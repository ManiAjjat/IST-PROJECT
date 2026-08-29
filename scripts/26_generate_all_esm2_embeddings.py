from pathlib import Path
import math
import os
import time

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
CACHE_DIR = PROJECT_DIR / "models" / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModel, AutoTokenizer


FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
RECOMMENDATION_FILE = PROJECT_DIR / "results" / "step43_esm2_batch_size_recommendation.csv"
NPY_OUTPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
CSV_OUTPUT = PROJECT_DIR / "derived" / "esm2_embeddings.csv"
METADATA_OUTPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
SUMMARY_OUTPUT = PROJECT_DIR / "results" / "step44_esm2_embedding_generation_summary.csv"
BATCH_LOG_OUTPUT = PROJECT_DIR / "results" / "step44_esm2_batch_log.csv"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EXPECTED_ROWS = 901
EXPECTED_HIDDEN_SIZE = 1280
EXPECTED_DEVELOPMENT = 720
EXPECTED_TEST = 181

print("=" * 94)
print("STEP 44 - GENERATE ESM-2 EMBEDDINGS FOR ALL 901 PEPTIDES")
print("=" * 94)

for required_file in (FEATURE_FILE, SPLIT_FILE, RECOMMENDATION_FILE):
    if not required_file.exists():
        raise FileNotFoundError(f"Required input not found: {required_file}")
if not CACHE_DIR.exists():
    raise FileNotFoundError(f"Step 42 model cache not found: {CACHE_DIR}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for Step 44 production embedding generation.")

device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

features = pd.read_csv(FEATURE_FILE)
split = pd.read_csv(SPLIT_FILE)
recommendation = pd.read_csv(RECOMMENDATION_FILE)
metadata_columns = [
    "ID", "sequence", "label", "binary_class", "inactive_source", "is_virtual_inactive"
]
missing_metadata = [column for column in metadata_columns if column not in features.columns]
if missing_metadata:
    raise ValueError(f"Feature table is missing metadata columns: {missing_metadata}")
if "split" not in split.columns:
    raise ValueError("Fixed split table is missing the split column.")
if len(features) != EXPECTED_ROWS or len(split) != EXPECTED_ROWS:
    raise ValueError("Expected 901 aligned feature and split rows.")
for column in ["ID", "sequence", "label", "binary_class", "inactive_source", "is_virtual_inactive"]:
    if not features[column].equals(split[column]):
        raise ValueError(f"Feature and split tables are not aligned for column: {column}")
if features["ID"].duplicated().any():
    raise ValueError("Peptide IDs are not unique.")
if features["sequence"].duplicated().any():
    raise ValueError("Peptide sequences are not unique.")
if int(split["split"].eq("development").sum()) != EXPECTED_DEVELOPMENT:
    raise ValueError("Unexpected development-set size.")
if int(split["split"].eq("test").sum()) != EXPECTED_TEST:
    raise ValueError("Unexpected test-set size.")

sequences = features["sequence"].astype(str).str.strip().str.upper().tolist()
sequence_lengths = np.asarray([len(sequence) for sequence in sequences], dtype=int)
if sequence_lengths.min() != 5 or sequence_lengths.max() != 38:
    raise ValueError("Unexpected peptide length range.")

if len(recommendation) != 1:
    raise ValueError("Step 43 recommendation must contain exactly one row.")
batch_size = int(recommendation.loc[0, "recommended_production_batch_size"])
if batch_size != 64:
    raise ValueError(f"Expected the verified Step 43 batch size 64, found {batch_size}.")
number_of_batches = math.ceil(EXPECTED_ROWS / batch_size)

print("\n44A. Production configuration:")
print("Model:", MODEL_NAME)
print("GPU:", gpu_name)
print("Total VRAM (GiB):", round(total_vram_gib, 3))
print("Model dtype: float16")
print("Saved embedding dtype: float32")
print("Batch size:", batch_size)
print("Number of peptides:", EXPECTED_ROWS)
print("Number of batches:", number_of_batches)
print("Original row order preserved:", True)
print("HF_HOME:", os.environ["HF_HOME"])
print("HF_HUB_OFFLINE:", os.environ["HF_HUB_OFFLINE"])

print("\n44B. Load model from local cache only:")
load_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    local_files_only=True,
    dtype=torch.float16,
)
model.eval()
model.to(device)
torch.cuda.synchronize(device)
model_load_seconds = time.perf_counter() - load_start
if int(model.config.hidden_size) != EXPECTED_HIDDEN_SIZE:
    raise ValueError(f"Unexpected hidden size: {model.config.hidden_size}")
if next(model.parameters()).device.type != "cuda":
    raise RuntimeError("Model is not on CUDA.")
print("Loaded locally:", True)
print("Model device:", next(model.parameters()).device)
print("Model parameter dtype:", next(model.parameters()).dtype)
print("Hidden size:", model.config.hidden_size)
print("Load seconds:", round(model_load_seconds, 3))

embeddings = np.empty((EXPECTED_ROWS, EXPECTED_HIDDEN_SIZE), dtype=np.float32)
batch_rows = []
production_start = time.perf_counter()

print("\n44C. Generate embeddings in original row order:")
for batch_number, start_row in enumerate(range(0, EXPECTED_ROWS, batch_size), start=1):
    end_row_exclusive = min(start_row + batch_size, EXPECTED_ROWS)
    batch_sequences = sequences[start_row:end_row_exclusive]
    actual_batch_size = len(batch_sequences)
    batch_lengths = sequence_lengths[start_row:end_row_exclusive]
    encoded = tokenizer(
        batch_sequences,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=False,
    )
    input_shape = tuple(encoded["input_ids"].shape)
    expected_token_width = int(batch_lengths.max()) + 2
    if input_shape != (actual_batch_size, expected_token_width):
        raise RuntimeError(f"Unexpected tokenized shape for batch {batch_number}: {input_shape}")
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    batch_start = time.perf_counter()
    with torch.inference_mode():
        hidden_states = model(**encoded).last_hidden_state
        expected_hidden_shape = (
            actual_batch_size, expected_token_width, EXPECTED_HIDDEN_SIZE
        )
        if tuple(hidden_states.shape) != expected_hidden_shape:
            raise RuntimeError(
                f"Unexpected hidden-state shape for batch {batch_number}: {tuple(hidden_states.shape)}"
            )
        pooled_vectors = []
        for local_index, sequence in enumerate(batch_sequences):
            valid_token_count = int(encoded["attention_mask"][local_index].sum().item())
            if valid_token_count != len(sequence) + 2:
                raise RuntimeError(
                    f"Attention mask mismatch at global row {start_row + local_index}."
                )
            residue_embeddings = hidden_states[
                local_index, 1 : valid_token_count - 1, :
            ]
            if tuple(residue_embeddings.shape) != (len(sequence), EXPECTED_HIDDEN_SIZE):
                raise RuntimeError(
                    f"Residue shape mismatch at global row {start_row + local_index}."
                )
            pooled_vectors.append(residue_embeddings.mean(dim=0))
        pooled_batch = torch.stack(pooled_vectors, dim=0)
    torch.cuda.synchronize(device)
    batch_seconds = time.perf_counter() - batch_start

    pooled_numpy = pooled_batch.detach().float().cpu().numpy()
    if pooled_numpy.shape != (actual_batch_size, EXPECTED_HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected pooled shape for batch {batch_number}.")
    missing_values = int(np.isnan(pooled_numpy).sum())
    non_finite_values = int((~np.isfinite(pooled_numpy)).sum())
    if missing_values or non_finite_values:
        raise RuntimeError(f"Invalid embedding values in batch {batch_number}.")
    embeddings[start_row:end_row_exclusive] = pooled_numpy

    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    batch_rows.append(
        {
            "batch_number": batch_number,
            "total_batches": number_of_batches,
            "start_embedding_row": start_row,
            "end_embedding_row_inclusive": end_row_exclusive - 1,
            "first_peptide_ID": int(features.iloc[start_row]["ID"]),
            "last_peptide_ID": int(features.iloc[end_row_exclusive - 1]["ID"]),
            "batch_size": actual_batch_size,
            "minimum_sequence_length": int(batch_lengths.min()),
            "maximum_sequence_length": int(batch_lengths.max()),
            "token_tensor_shape": str(input_shape),
            "hidden_state_shape": str(tuple(hidden_states.shape)),
            "pooled_shape": str(tuple(pooled_numpy.shape)),
            "inference_seconds": batch_seconds,
            "peptides_per_second": actual_batch_size / batch_seconds,
            "missing_values": missing_values,
            "non_finite_values": non_finite_values,
            "peak_gpu_allocated_gib": peak_allocated_gib,
            "peak_gpu_reserved_gib": peak_reserved_gib,
            "peak_reserved_percent_total_vram": 100 * peak_reserved_gib / total_vram_gib,
        }
    )
    print(
        f"Batch {batch_number:>2}/{number_of_batches}: rows {start_row}-{end_row_exclusive - 1}; "
        f"n={actual_batch_size}; pooled={pooled_numpy.shape}; "
        f"missing={missing_values}; non-finite={non_finite_values}; "
        f"{batch_seconds:.3f} s"
    )
    del encoded, hidden_states, pooled_vectors, pooled_batch, pooled_numpy

production_seconds = time.perf_counter() - production_start
batch_log = pd.DataFrame(batch_rows)
if len(batch_log) != number_of_batches:
    raise RuntimeError("Unexpected number of completed batches.")
if batch_log["batch_size"].iloc[:-1].ne(batch_size).any():
    raise RuntimeError("A non-final production batch did not contain 64 peptides.")
if int(batch_log.iloc[-1]["batch_size"]) != 5:
    raise RuntimeError("Final batch did not contain the expected 5 peptides.")

print("\n44D. Full-matrix integrity checks:")
if embeddings.shape != (EXPECTED_ROWS, EXPECTED_HIDDEN_SIZE):
    raise RuntimeError(f"Unexpected full matrix shape: {embeddings.shape}")
if embeddings.dtype != np.float32:
    raise RuntimeError(f"Unexpected full matrix dtype: {embeddings.dtype}")
missing_total = int(np.isnan(embeddings).sum())
non_finite_total = int((~np.isfinite(embeddings)).sum())
if missing_total or non_finite_total:
    raise RuntimeError("Full embedding matrix contains invalid values.")
unique_embedding_rows = int(np.unique(embeddings, axis=0).shape[0])
duplicate_embedding_rows = EXPECTED_ROWS - unique_embedding_rows
if duplicate_embedding_rows != 0:
    raise RuntimeError(f"Found {duplicate_embedding_rows} exactly duplicate embedding rows.")
print("Embedding matrix:", embeddings.shape)
print("dtype:", embeddings.dtype)
print("Missing:", missing_total)
print("Non-finite:", non_finite_total)
print("Unique embedding rows:", unique_embedding_rows)
print("Duplicate embedding rows:", duplicate_embedding_rows)

metadata = features[metadata_columns].copy()
metadata.insert(0, "embedding_row", np.arange(EXPECTED_ROWS, dtype=int))
metadata.insert(3, "sequence_length", sequence_lengths)
metadata["split"] = split["split"].to_numpy()
expected_metadata_columns = [
    "embedding_row", "ID", "sequence", "sequence_length", "label",
    "binary_class", "inactive_source", "is_virtual_inactive", "split",
]
metadata = metadata[expected_metadata_columns]
if not np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(EXPECTED_ROWS)):
    raise RuntimeError("Metadata embedding_row is not the original zero-based row order.")
if not metadata["ID"].equals(features["ID"]):
    raise RuntimeError("Metadata IDs are not aligned with the embedding rows.")
if not metadata["sequence"].equals(features["sequence"]):
    raise RuntimeError("Metadata sequences are not aligned with the embedding rows.")

NPY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
np.save(NPY_OUTPUT, embeddings)
embedding_columns = [f"esm2_{index:04d}" for index in range(1, EXPECTED_HIDDEN_SIZE + 1)]
pd.DataFrame(embeddings, columns=embedding_columns).to_csv(
    CSV_OUTPUT, index=False, float_format="%.9g"
)
metadata.to_csv(METADATA_OUTPUT, index=False)
batch_log.to_csv(BATCH_LOG_OUTPUT, index=False)

print("\n44E. Saved-artifact round-trip checks:")
reloaded_npy = np.load(NPY_OUTPUT)
if not np.array_equal(reloaded_npy, embeddings):
    raise RuntimeError("NPY round-trip did not preserve exact float32 values.")
reloaded_csv = pd.read_csv(CSV_OUTPUT).to_numpy(dtype=np.float32)
if reloaded_csv.shape != embeddings.shape or not np.array_equal(reloaded_csv, embeddings):
    raise RuntimeError("CSV round-trip did not preserve the float32 embedding matrix.")
reloaded_metadata = pd.read_csv(METADATA_OUTPUT)
if len(reloaded_metadata) != EXPECTED_ROWS:
    raise RuntimeError("Reloaded metadata has the wrong row count.")
if not reloaded_metadata["ID"].equals(features["ID"]):
    raise RuntimeError("Reloaded metadata IDs are misaligned.")
if not reloaded_metadata["sequence"].equals(features["sequence"]):
    raise RuntimeError("Reloaded metadata sequences are misaligned.")
development_count = int(reloaded_metadata["split"].eq("development").sum())
test_count = int(reloaded_metadata["split"].eq("test").sum())
if development_count != EXPECTED_DEVELOPMENT or test_count != EXPECTED_TEST:
    raise RuntimeError("Reloaded metadata split counts are incorrect.")
print("NPY exact round-trip:", True)
print("CSV float32 exact round-trip:", True)
print("Metadata ID/sequence alignment:", True)
print("Development/test rows:", development_count, test_count)

peak_allocated_gib = float(batch_log["peak_gpu_allocated_gib"].max())
peak_reserved_gib = float(batch_log["peak_gpu_reserved_gib"].max())
summary = pd.DataFrame(
    [
        {
            "model": MODEL_NAME,
            "cache_directory": str(CACHE_DIR),
            "local_cache_only": True,
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": gpu_name,
            "gpu_total_vram_gib": total_vram_gib,
            "model_dtype": str(next(model.parameters()).dtype),
            "saved_embedding_dtype": str(embeddings.dtype),
            "batch_size": batch_size,
            "number_of_batches": number_of_batches,
            "embedding_rows": embeddings.shape[0],
            "embedding_dimensions": embeddings.shape[1],
            "minimum_sequence_length": int(sequence_lengths.min()),
            "maximum_sequence_length": int(sequence_lengths.max()),
            "development_rows": development_count,
            "test_rows": test_count,
            "missing_values": missing_total,
            "non_finite_values": non_finite_total,
            "unique_embedding_rows": unique_embedding_rows,
            "duplicate_embedding_rows": duplicate_embedding_rows,
            "original_order_preserved": True,
            "padding_excluded_from_pooling": True,
            "special_tokens_excluded_from_pooling": True,
            "model_load_seconds": model_load_seconds,
            "production_inference_seconds": production_seconds,
            "overall_peptides_per_second": EXPECTED_ROWS / production_seconds,
            "peak_gpu_allocated_gib": peak_allocated_gib,
            "peak_gpu_reserved_gib": peak_reserved_gib,
            "npy_exact_round_trip": True,
            "csv_float32_exact_round_trip": True,
            "metadata_alignment_verified": True,
        }
    ]
)
summary.to_csv(SUMMARY_OUTPUT, index=False)

print("\n44Q. Output checks:")
for output_path in (NPY_OUTPUT, CSV_OUTPUT, METADATA_OUTPUT, BATCH_LOG_OUTPUT, SUMMARY_OUTPUT):
    print(output_path.name, "exists:", output_path.exists())

print("\n" + "=" * 94)
print("STEP 44 SUMMARY")
print("=" * 94)
print("Model:", MODEL_NAME)
print("GPU:", gpu_name)
print("Batch size:", batch_size)
print("Batches:", number_of_batches)
print("Embedding matrix:", embeddings.shape)
print("Embedding dtype:", embeddings.dtype)
print("Development/test:", development_count, test_count)
print("Missing values:", missing_total)
print("Non-finite values:", non_finite_total)
print("Unique embedding rows:", unique_embedding_rows)
print("Duplicate embedding rows:", duplicate_embedding_rows)
print("Production seconds:", round(production_seconds, 3))
print("Overall peptides/second:", round(EXPECTED_ROWS / production_seconds, 3))
print("Peak GPU allocated:", round(peak_allocated_gib, 3), "GiB")
print("Peak GPU reserved:", round(peak_reserved_gib, 3), "GiB")
print("\nPrimary NPY:", NPY_OUTPUT)
print("CSV:", CSV_OUTPUT)
print("Metadata:", METADATA_OUTPUT)
print("Batch log:", BATCH_LOG_OUTPUT)
print("Summary:", SUMMARY_OUTPUT)
print("\nSTEP 44 COMPLETED SUCCESSFULLY")
print("=" * 94)

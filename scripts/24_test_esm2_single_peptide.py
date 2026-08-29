from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
INPUT_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
CACHE_DIR = PROJECT_DIR / "models" / "huggingface_cache"
REPORT_OUTPUT = PROJECT_DIR / "results" / "step42_esm2_single_peptide_test.csv"
EMBEDDING_OUTPUT = PROJECT_DIR / "results" / "step42_single_peptide_embedding.npy"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EXPECTED_HIDDEN_SIZE = 1280
EXPECTED_TEST_ID = 1
EXPECTED_SEQUENCE = "AIGKFLHSAKKFGKAFVGEIMNS"

print("=" * 90)
print("STEP 42 - DOWNLOAD ESM-2 AND TEST ONE PEPTIDE")
print("=" * 90)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

print("\n42A. Runtime verification:")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the controlled Step 42 test.")
device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
gpu_properties = torch.cuda.get_device_properties(0)
total_vram_gib = gpu_properties.total_memory / (1024 ** 3)
print("GPU:", gpu_name)
print("GPU VRAM (GiB):", round(total_vram_gib, 3))
print("HF_HOME:", os.environ["HF_HOME"])

print("\n42B. Select exactly one peptide:")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Sequence table not found: {INPUT_FILE}")
peptide_table = pd.read_csv(INPUT_FILE)
required_columns = {"ID", "sequence"}
missing_columns = required_columns.difference(peptide_table.columns)
if missing_columns:
    raise ValueError(f"Sequence table is missing columns: {sorted(missing_columns)}")
selected = peptide_table.loc[peptide_table["ID"].eq(EXPECTED_TEST_ID)]
if len(selected) != 1:
    raise ValueError(f"Expected exactly one row for peptide ID {EXPECTED_TEST_ID}.")
test_id = int(selected.iloc[0]["ID"])
test_sequence = str(selected.iloc[0]["sequence"]).strip().upper()
if test_sequence != EXPECTED_SEQUENCE:
    raise ValueError(f"Unexpected sequence for peptide ID {EXPECTED_TEST_ID}: {test_sequence}")
print("Peptide ID:", test_id)
print("Sequence:", test_sequence)
print("Residues:", len(test_sequence))
print("Peptides selected for inference:", 1)

print("\n42C. Load tokenizer and model:")
download_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    dtype=torch.float16,
)
model.eval()
model.to(device)
torch.cuda.synchronize(device)
load_seconds = time.perf_counter() - download_start
model_allocated_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
model_reserved_gib = torch.cuda.memory_reserved(device) / (1024 ** 3)
hidden_size = int(model.config.hidden_size)
print("Model:", MODEL_NAME)
print("Model hidden size:", hidden_size)
print("Model dtype:", next(model.parameters()).dtype)
print("Model device:", next(model.parameters()).device)
print("Load/download seconds:", round(load_seconds, 3))
print("GPU allocated after model load (GiB):", round(model_allocated_gib, 3))
print("GPU reserved after model load (GiB):", round(model_reserved_gib, 3))
if hidden_size != EXPECTED_HIDDEN_SIZE:
    raise ValueError(f"Expected hidden size {EXPECTED_HIDDEN_SIZE}, found {hidden_size}.")
if next(model.parameters()).device.type != "cuda":
    raise RuntimeError("ESM-2 model was not moved to CUDA.")

print("\n42D. Tokenize and run one-peptide inference:")
encoded = tokenizer(
    test_sequence,
    return_tensors="pt",
    add_special_tokens=True,
    padding=False,
    truncation=False,
)
token_count = int(encoded["input_ids"].shape[1])
expected_token_count = len(test_sequence) + 2
if token_count != expected_token_count:
    raise ValueError(f"Expected {expected_token_count} tokens, found {token_count}.")
encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

torch.cuda.reset_peak_memory_stats(device)
inference_start = time.perf_counter()
with torch.inference_mode():
    output = model(**encoded)
torch.cuda.synchronize(device)
inference_seconds = time.perf_counter() - inference_start

last_hidden_state = output.last_hidden_state
expected_output_shape = (1, token_count, EXPECTED_HIDDEN_SIZE)
if tuple(last_hidden_state.shape) != expected_output_shape:
    raise ValueError(
        f"Expected model output shape {expected_output_shape}, found {tuple(last_hidden_state.shape)}."
    )

# Position 0 is the beginning token and the final position is the end token.
# Pool only the actual amino-acid residue representations.
residue_embeddings = last_hidden_state[0, 1 : len(test_sequence) + 1, :]
if tuple(residue_embeddings.shape) != (len(test_sequence), EXPECTED_HIDDEN_SIZE):
    raise ValueError(f"Unexpected residue embedding shape: {tuple(residue_embeddings.shape)}")
pooled_embedding = residue_embeddings.mean(dim=0)
pooled_numpy = pooled_embedding.detach().float().cpu().numpy()
if pooled_numpy.shape != (EXPECTED_HIDDEN_SIZE,):
    raise ValueError(f"Unexpected pooled shape: {pooled_numpy.shape}")

missing_values = int(np.isnan(pooled_numpy).sum())
non_finite_values = int((~np.isfinite(pooled_numpy)).sum())
if missing_values or non_finite_values:
    raise ValueError("The pooled embedding contains missing or non-finite values.")

peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
np.save(EMBEDDING_OUTPUT, pooled_numpy.astype(np.float32, copy=False))
saved_embedding = np.load(EMBEDDING_OUTPUT)
if saved_embedding.shape != (EXPECTED_HIDDEN_SIZE,) or saved_embedding.dtype != np.float32:
    raise RuntimeError("Saved embedding failed shape or dtype verification.")

print("Token count:", token_count)
print("Model output shape:", tuple(last_hidden_state.shape))
print("Residue embedding shape:", tuple(residue_embeddings.shape))
print("Pooled embedding shape:", pooled_numpy.shape)
print("Saved embedding dtype:", saved_embedding.dtype)
print("Missing values:", missing_values)
print("Non-finite values:", non_finite_values)
print("Inference seconds:", round(inference_seconds, 4))
print("Peak GPU allocated during inference (GiB):", round(peak_allocated, 3))
print("Peak GPU reserved during inference (GiB):", round(peak_reserved, 3))

report = pd.DataFrame(
    [
        {
            "model": MODEL_NAME,
            "cache_directory": str(CACHE_DIR),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": gpu_name,
            "gpu_total_vram_gib": total_vram_gib,
            "model_dtype": str(next(model.parameters()).dtype),
            "peptide_id": test_id,
            "sequence": test_sequence,
            "peptide_length": len(test_sequence),
            "token_count": token_count,
            "model_output_shape": str(tuple(last_hidden_state.shape)),
            "residue_embedding_shape": str(tuple(residue_embeddings.shape)),
            "hidden_dimension": hidden_size,
            "mean_pooled_shape": str(pooled_numpy.shape),
            "saved_embedding_dtype": str(saved_embedding.dtype),
            "missing_embedding_values": missing_values,
            "non_finite_embedding_values": non_finite_values,
            "model_load_and_download_seconds": load_seconds,
            "single_peptide_inference_seconds": inference_seconds,
            "model_gpu_allocated_gib": model_allocated_gib,
            "model_gpu_reserved_gib": model_reserved_gib,
            "peak_inference_gpu_allocated_gib": peak_allocated,
            "peak_inference_gpu_reserved_gib": peak_reserved,
            "peptides_processed": 1,
        }
    ]
)
report.to_csv(REPORT_OUTPUT, index=False)

print("\n42Q. Output checks:")
print("Report exists:", REPORT_OUTPUT.exists())
print("Embedding file exists:", EMBEDDING_OUTPUT.exists())

print("\n" + "=" * 90)
print("STEP 42 SUMMARY")
print("=" * 90)
print("Model:", MODEL_NAME)
print("GPU:", torch.cuda.get_device_name(0))
print("Peptide ID:", test_id)
print("Peptide length:", len(test_sequence))
print("Hidden dimension:", hidden_size)
print("Residue rows:", residue_embeddings.shape[0])
print("Mean-pooled dimensions:", pooled_numpy.shape[0])
print("Missing embedding values:", missing_values)
print("Non-finite embedding values:", non_finite_values)
print("Peak GPU allocated (GiB):", round(peak_allocated, 3))
print("Peak GPU reserved (GiB):", round(peak_reserved, 3))
print("\nReport:", REPORT_OUTPUT)
print("One-peptide embedding:", EMBEDDING_OUTPUT)
print("\nSTEP 42 COMPLETED SUCCESSFULLY")
print("=" * 90)

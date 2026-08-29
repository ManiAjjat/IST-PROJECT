from pathlib import Path
import gc
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


INPUT_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
BENCHMARK_OUTPUT = PROJECT_DIR / "results" / "step43_esm2_batch_size_benchmark.csv"
RECOMMENDATION_OUTPUT = PROJECT_DIR / "results" / "step43_esm2_batch_size_recommendation.csv"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EXPECTED_HIDDEN_SIZE = 1280
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MEMORY_LIMIT_PERCENT = 75.0

print("=" * 92)
print("STEP 43 - ESM-2 BATCH-SIZE BENCHMARK")
print("=" * 92)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the Step 43 benchmark.")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input table not found: {INPUT_FILE}")
if not CACHE_DIR.exists():
    raise FileNotFoundError(f"Step 42 cache not found: {CACHE_DIR}")

device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print("\n43A. Runtime:")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", gpu_name)
print("Total VRAM (GiB):", round(total_vram_gib, 3))
print("HF_HOME:", os.environ["HF_HOME"])
print("HF_HUB_OFFLINE:", os.environ["HF_HUB_OFFLINE"])

peptides = pd.read_csv(INPUT_FILE)
if not {"ID", "sequence"}.issubset(peptides.columns):
    raise ValueError("Input table must contain ID and sequence columns.")
if len(peptides) != 901 or peptides["ID"].duplicated().any():
    raise ValueError("Expected 901 peptides with unique IDs.")
peptides = peptides[["ID", "sequence"]].copy()
peptides["sequence"] = peptides["sequence"].astype(str).str.strip().str.upper()
peptides["sequence_length"] = peptides["sequence"].str.len()
peptides = peptides.sort_values(
    ["sequence_length", "ID"], ascending=[False, True]
).reset_index(drop=True)

print("\n43B. Conservative benchmark set:")
print("Dataset peptides:", len(peptides))
print("Maximum real sequence length:", int(peptides["sequence_length"].max()))
print("Longest 64 minimum length:", int(peptides.head(64)["sequence_length"].min()))
print("Batch sizes:", BATCH_SIZES)

print("\n43C. Load Step 42 cache only:")
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
if int(model.config.hidden_size) != EXPECTED_HIDDEN_SIZE:
    raise ValueError(f"Unexpected hidden size: {model.config.hidden_size}")
if next(model.parameters()).device.type != "cuda":
    raise RuntimeError("Model is not on CUDA.")
print("Model loaded from local cache:", MODEL_NAME)
print("Model dtype:", next(model.parameters()).dtype)
print("Hidden size:", model.config.hidden_size)
print("Network access required:", False)

# One unrecorded warm-up eliminates first-kernel initialization from timing.
warmup_sequence = peptides.iloc[0]["sequence"]
warmup = tokenizer(warmup_sequence, return_tensors="pt", add_special_tokens=True)
warmup = {name: tensor.to(device) for name, tensor in warmup.items()}
with torch.inference_mode():
    _ = model(**warmup).last_hidden_state
torch.cuda.synchronize(device)
del warmup, _
torch.cuda.empty_cache()

print("\n43D. Benchmark batches:")
benchmark_rows = []
for requested_batch_size in BATCH_SIZES:
    batch = peptides.head(requested_batch_size).copy()
    sequences = batch["sequence"].tolist()
    actual_batch_size = len(sequences)
    max_sequence_length = int(batch["sequence_length"].max())
    min_sequence_length = int(batch["sequence_length"].min())
    row = {
        "requested_batch_size": requested_batch_size,
        "actual_batch_size": actual_batch_size,
        "success": False,
        "error_type": "",
        "error_message": "",
        "inference_seconds": np.nan,
        "peptides_per_second": np.nan,
        "minimum_sequence_length": min_sequence_length,
        "maximum_sequence_length": max_sequence_length,
        "input_ids_shape": "",
        "attention_mask_shape": "",
        "model_output_shape": "",
        "pooled_embedding_shape": "",
        "embedding_dimension": EXPECTED_HIDDEN_SIZE,
        "missing_embedding_values": np.nan,
        "non_finite_embedding_values": np.nan,
        "peak_gpu_allocated_gib": np.nan,
        "peak_gpu_reserved_gib": np.nan,
        "peak_allocated_percent_total_vram": np.nan,
        "peak_reserved_percent_total_vram": np.nan,
    }
    encoded = None
    outputs = None
    pooled_batch = None
    try:
        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=False,
        )
        input_ids_shape = tuple(encoded["input_ids"].shape)
        attention_mask_shape = tuple(encoded["attention_mask"].shape)
        if input_ids_shape[0] != actual_batch_size:
            raise RuntimeError("Tokenizer returned the wrong batch dimension.")
        if input_ids_shape[1] != max_sequence_length + 2:
            raise RuntimeError("Tokenizer length does not equal residues plus two special tokens.")
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**encoded).last_hidden_state
            pooled_vectors = []
            for sample_index, sequence in enumerate(sequences):
                valid_token_count = int(encoded["attention_mask"][sample_index].sum().item())
                if valid_token_count != len(sequence) + 2:
                    raise RuntimeError("Attention mask does not match residue plus special-token count.")
                residue_embeddings = outputs[sample_index, 1 : valid_token_count - 1, :]
                if tuple(residue_embeddings.shape) != (len(sequence), EXPECTED_HIDDEN_SIZE):
                    raise RuntimeError("Residue-level shape validation failed.")
                pooled_vectors.append(residue_embeddings.mean(dim=0))
            pooled_batch = torch.stack(pooled_vectors, dim=0)
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - start

        expected_output_shape = (
            actual_batch_size, max_sequence_length + 2, EXPECTED_HIDDEN_SIZE
        )
        if tuple(outputs.shape) != expected_output_shape:
            raise RuntimeError(f"Unexpected model output shape: {tuple(outputs.shape)}")
        if tuple(pooled_batch.shape) != (actual_batch_size, EXPECTED_HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected pooled shape: {tuple(pooled_batch.shape)}")

        missing_values = int(torch.isnan(pooled_batch).sum().item())
        non_finite_values = int((~torch.isfinite(pooled_batch)).sum().item())
        if missing_values or non_finite_values:
            raise RuntimeError("Pooled embeddings contain missing or non-finite values.")

        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        row.update(
            {
                "success": True,
                "inference_seconds": inference_seconds,
                "peptides_per_second": actual_batch_size / inference_seconds,
                "input_ids_shape": str(input_ids_shape),
                "attention_mask_shape": str(attention_mask_shape),
                "model_output_shape": str(tuple(outputs.shape)),
                "pooled_embedding_shape": str(tuple(pooled_batch.shape)),
                "missing_embedding_values": missing_values,
                "non_finite_embedding_values": non_finite_values,
                "peak_gpu_allocated_gib": peak_allocated_gib,
                "peak_gpu_reserved_gib": peak_reserved_gib,
                "peak_allocated_percent_total_vram": 100 * peak_allocated_gib / total_vram_gib,
                "peak_reserved_percent_total_vram": 100 * peak_reserved_gib / total_vram_gib,
            }
        )
        print(
            f"Batch {actual_batch_size:>2}: success; {inference_seconds:.4f} s; "
            f"{actual_batch_size / inference_seconds:.2f} peptides/s; "
            f"peak reserved {peak_reserved_gib:.3f} GiB "
            f"({100 * peak_reserved_gib / total_vram_gib:.1f}%)"
        )
    except torch.cuda.OutOfMemoryError as exc:
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc).replace("\n", " ")
        print(f"Batch {actual_batch_size:>2}: CUDA out of memory; recorded and continuing.")
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc).replace("\n", " ")
        print(f"Batch {actual_batch_size:>2}: failed: {type(exc).__name__}: {exc}")
    finally:
        benchmark_rows.append(row)
        del encoded, outputs, pooled_batch
        gc.collect()
        torch.cuda.empty_cache()

benchmark_df = pd.DataFrame(benchmark_rows)
successful_df = benchmark_df.loc[
    benchmark_df["success"]
    & benchmark_df["peak_gpu_reserved_gib"].notna()
    & (benchmark_df["peak_reserved_percent_total_vram"] < MEMORY_LIMIT_PERCENT)
].copy()
if successful_df.empty:
    raise RuntimeError(
        f"No successful tested batch stayed below {MEMORY_LIMIT_PERCENT:.0f}% reserved VRAM."
    )
recommended_row = successful_df.sort_values("actual_batch_size").iloc[-1]
recommended_batch_size = int(recommended_row["actual_batch_size"])

recommendation_df = pd.DataFrame(
    [
        {
            "model": MODEL_NAME,
            "gpu": gpu_name,
            "gpu_total_vram_gib": total_vram_gib,
            "model_dtype": str(next(model.parameters()).dtype),
            "embedding_dimension": EXPECTED_HIDDEN_SIZE,
            "tested_batch_sizes": ",".join(map(str, BATCH_SIZES)),
            "successful_batch_sizes": ",".join(
                map(str, benchmark_df.loc[benchmark_df["success"], "actual_batch_size"].astype(int))
            ),
            "memory_safety_threshold_percent": MEMORY_LIMIT_PERCENT,
            "recommended_production_batch_size": recommended_batch_size,
            "recommended_peak_allocated_gib": float(recommended_row["peak_gpu_allocated_gib"]),
            "recommended_peak_reserved_gib": float(recommended_row["peak_gpu_reserved_gib"]),
            "recommended_peak_reserved_percent": float(
                recommended_row["peak_reserved_percent_total_vram"]
            ),
            "benchmark_maximum_sequence_length": int(peptides["sequence_length"].max()),
            "selection_rule": (
                "largest tested successful batch with peak reserved GPU memory below 75%"
            ),
            "local_cache_only": True,
            "full_dataset_embeddings_generated": False,
        }
    ]
)

BENCHMARK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
benchmark_df.to_csv(BENCHMARK_OUTPUT, index=False)
recommendation_df.to_csv(RECOMMENDATION_OUTPUT, index=False)

print("\n43Q. Output checks:")
print("Benchmark CSV exists:", BENCHMARK_OUTPUT.exists())
print("Recommendation CSV exists:", RECOMMENDATION_OUTPUT.exists())

print("\n" + "=" * 92)
print("STEP 43 SUMMARY")
print("=" * 92)
print("Model:", MODEL_NAME)
print("GPU:", gpu_name)
print("Total VRAM:", round(total_vram_gib, 3), "GiB")
print("Batch sizes tested:", len(BATCH_SIZES))
print("Successful batch sizes:", benchmark_df.loc[benchmark_df["success"], "actual_batch_size"].tolist())
print("Recommended production batch size:", recommended_batch_size)
print("\nBenchmark results:", BENCHMARK_OUTPUT)
print("Recommendation:", RECOMMENDATION_OUTPUT)
print("\nSTEP 43 COMPLETED SUCCESSFULLY")
print("=" * 92)

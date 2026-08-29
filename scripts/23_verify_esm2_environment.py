from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
INPUT_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
CACHE_DIR = PROJECT_DIR / "models" / "huggingface_cache"
REPORT_OUTPUT = PROJECT_DIR / "results" / "step41_esm2_environment_verification.csv"
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
EXPECTED_EMBEDDING_DIM = 1280
EXPECTED_PEPTIDES = 901
ESM2_MAX_RESIDUES = 1022

print("=" * 90)
print("STEP 41 - VERIFY ESM-2 ENVIRONMENT")
print("=" * 90)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

print("\n41A. Project directory:")
print(PROJECT_DIR)
print("\nProposed Hugging Face cache:")
print(CACHE_DIR)

print("\n41B. System information:")
print("Python:", platform.python_version())
print("Python executable:", sys.executable)
print("Platform:", platform.platform())
print("Processor:", platform.processor() or "Not reported")

print("\n41C. Core packages:")
print("NumPy version:", np.__version__)
print("pandas version:", pd.__version__)

def distribution_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


torch_version = distribution_version("torch")
torch_installed = torch_version != "not installed"
torch_import_ok = False
cuda_available = False
cuda_version = "not available"
cudnn_version = "not available"
gpu_count = 0
gpu_names = []
gpu_memory_gib = []
torch_error = ""
system_gpu_names = []
system_gpu_memory_mib = []
system_gpu_driver_versions = []

print("\n41D. PyTorch and CUDA check:")
try:
    import torch

    torch_import_ok = True
    torch_version = torch.__version__
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = str(torch.version.cuda) if torch.version.cuda else "not available"
    cudnn_raw = torch.backends.cudnn.version()
    cudnn_version = str(cudnn_raw) if cudnn_raw is not None else "not available"
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    print("PyTorch distribution installed:", torch_installed)
    print("PyTorch import successful:", torch_import_ok)
    print("PyTorch version:", torch_version)
    print("CUDA build version:", cuda_version)
    print("CUDA available:", cuda_available)
    print("cuDNN version:", cudnn_version)
    print("GPU count:", gpu_count)
    for gpu_index in range(gpu_count):
        properties = torch.cuda.get_device_properties(gpu_index)
        total_memory_gib = properties.total_memory / (1024 ** 3)
        gpu_names.append(properties.name)
        gpu_memory_gib.append(total_memory_gib)
        print(f"GPU {gpu_index}:", properties.name)
        print(f"GPU {gpu_index} total memory (GiB):", round(total_memory_gib, 2))
except Exception as exc:
    torch_error = f"{type(exc).__name__}: {exc}"
    print("PyTorch distribution installed:", torch_installed)
    print("PyTorch distribution version:", torch_version)
    print("PyTorch import successful:", torch_import_ok)
    print("PyTorch error:", torch_error)

try:
    nvidia_result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    for line in nvidia_result.stdout.strip().splitlines():
        name, memory_mib, driver_version = [value.strip() for value in line.split(",", maxsplit=2)]
        system_gpu_names.append(name)
        system_gpu_memory_mib.append(float(memory_mib))
        system_gpu_driver_versions.append(driver_version)
except (FileNotFoundError, subprocess.SubprocessError, ValueError):
    pass

print("NVIDIA GPUs detected by system driver:", len(system_gpu_names))
for gpu_index, (name, memory_mib, driver_version) in enumerate(
    zip(system_gpu_names, system_gpu_memory_mib, system_gpu_driver_versions)
):
    print(f"System GPU {gpu_index}:", name)
    print(f"System GPU {gpu_index} memory (MiB):", round(memory_mib))
    print(f"System GPU {gpu_index} driver:", driver_version)

transformers_version = distribution_version("transformers")
transformers_installed = transformers_version != "not installed"
transformers_import_ok = False
huggingface_hub_version = distribution_version("huggingface-hub")
esm_architecture_supported = False
auto_config_supported = False
transformers_error = ""

print("\n41E. Transformers and ESM support check:")
try:
    import transformers
    import huggingface_hub
    from transformers import AutoConfig, EsmConfig, EsmModel, EsmTokenizer

    transformers_import_ok = True
    transformers_version = transformers.__version__
    huggingface_hub_version = huggingface_hub.__version__
    esm_config = EsmConfig(hidden_size=EXPECTED_EMBEDDING_DIM)
    esm_architecture_supported = (
        EsmModel is not None
        and EsmTokenizer is not None
        and esm_config.hidden_size == EXPECTED_EMBEDDING_DIM
    )
    auto_config = AutoConfig.for_model("esm", hidden_size=EXPECTED_EMBEDDING_DIM)
    auto_config_supported = auto_config.model_type == "esm"
    print("Transformers distribution installed:", transformers_installed)
    print("Transformers import successful:", transformers_import_ok)
    print("Transformers version:", transformers_version)
    print("huggingface_hub version:", huggingface_hub_version)
    print("EsmConfig/EsmModel/EsmTokenizer import:", esm_architecture_supported)
    print("AutoConfig ESM registration:", auto_config_supported)
    print("Target model ID:", MODEL_ID)
    print("Expected pooled embedding dimensions:", EXPECTED_EMBEDDING_DIM)
    print("Network or model download attempted:", False)
except Exception as exc:
    transformers_error = f"{type(exc).__name__}: {exc}"
    print("Transformers distribution installed:", transformers_installed)
    print("Transformers distribution version:", transformers_version)
    print("Transformers import successful:", transformers_import_ok)
    print("huggingface_hub distribution version:", huggingface_hub_version)
    print("Transformers error:", transformers_error)

print("\n41F. Cache and disk check:")
disk_usage = shutil.disk_usage(CACHE_DIR)
disk_total_gib = disk_usage.total / (1024 ** 3)
disk_used_gib = disk_usage.used / (1024 ** 3)
disk_free_gib = disk_usage.free / (1024 ** 3)
cache_writable = os.access(CACHE_DIR, os.W_OK)
print("HF_HOME:", os.environ["HF_HOME"])
print("Cache directory exists:", CACHE_DIR.exists())
print("Cache directory writable:", cache_writable)
print("Disk total (GiB):", round(disk_total_gib, 2))
print("Disk used (GiB):", round(disk_used_gib, 2))
print("Disk free (GiB):", round(disk_free_gib, 2))

print("\n41G. Peptide dataset check:")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Traditional feature table not found: {INPUT_FILE}")
if not SPLIT_FILE.exists():
    raise FileNotFoundError(f"Fixed split table not found: {SPLIT_FILE}")

peptides = pd.read_csv(INPUT_FILE)
split = pd.read_csv(SPLIT_FILE)
required_sequence_columns = {"ID", "sequence", "label"}
missing_sequence_columns = required_sequence_columns.difference(peptides.columns)
if missing_sequence_columns:
    raise ValueError(f"Sequence table is missing columns: {sorted(missing_sequence_columns)}")
if len(peptides) != len(split):
    raise ValueError("Feature and split tables have different row counts.")
if peptides["ID"].duplicated().any():
    raise ValueError("Peptide IDs are not unique.")
if peptides["sequence"].isna().any():
    raise ValueError("Missing peptide sequences were found.")

sequences = peptides["sequence"].astype(str).str.strip().str.upper()
sequence_lengths = sequences.str.len()
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
invalid_character_rows = [
    index for index, sequence in enumerate(sequences)
    if not set(sequence).issubset(valid_amino_acids)
]
development_n = int(split["split"].eq("development").sum())
test_n = int(split["split"].eq("test").sum())
sequences_ready = (
    len(peptides) == EXPECTED_PEPTIDES
    and not invalid_character_rows
    and int(sequence_lengths.min()) > 0
)
lengths_within_limit = bool((sequence_lengths <= ESM2_MAX_RESIDUES).all())

print("Peptides:", len(peptides))
print("Development peptides:", development_n)
print("Test peptides:", test_n)
print("Unique IDs:", peptides["ID"].nunique())
print("Minimum sequence length:", int(sequence_lengths.min()))
print("Maximum sequence length:", int(sequence_lengths.max()))
print("Mean sequence length:", round(float(sequence_lengths.mean()), 3))
print("Invalid amino-acid rows:", len(invalid_character_rows))
print("Sequences ready:", sequences_ready)
print("Maximum supported residues used for check:", ESM2_MAX_RESIDUES)
print("All sequences within ESM-2 limit:", lengths_within_limit)

if cuda_available:
    recommended_device = "cuda"
elif system_gpu_names:
    recommended_device = "repair environment, then recheck CUDA for the detected NVIDIA GPU"
else:
    recommended_device = "cpu"
environment_ready = all(
    [
        torch_import_ok,
        transformers_import_ok,
        esm_architecture_supported,
        auto_config_supported,
        cache_writable,
        sequences_ready,
        lengths_within_limit,
    ]
)

report = pd.DataFrame(
    [
        {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "torch_installed": torch_installed,
            "torch_version": torch_version,
            "torch_import_successful": torch_import_ok,
            "torch_error": torch_error,
            "cuda_build_version": cuda_version,
            "cuda_available": cuda_available,
            "cudnn_version": cudnn_version,
            "gpu_count": gpu_count,
            "gpu_names": " | ".join(gpu_names),
            "gpu_total_memory_gib": " | ".join(f"{value:.2f}" for value in gpu_memory_gib),
            "system_nvidia_gpu_count": len(system_gpu_names),
            "system_nvidia_gpu_names": " | ".join(system_gpu_names),
            "system_nvidia_gpu_memory_mib": " | ".join(f"{value:.0f}" for value in system_gpu_memory_mib),
            "system_nvidia_driver_versions": " | ".join(system_gpu_driver_versions),
            "transformers_installed": transformers_installed,
            "transformers_version": transformers_version,
            "transformers_import_successful": transformers_import_ok,
            "transformers_error": transformers_error,
            "huggingface_hub_version": huggingface_hub_version,
            "esm_architecture_supported": esm_architecture_supported,
            "auto_config_esm_supported": auto_config_supported,
            "target_model_id": MODEL_ID,
            "expected_embedding_dimensions": EXPECTED_EMBEDDING_DIM,
            "model_download_attempted": False,
            "cache_directory": str(CACHE_DIR),
            "cache_writable": cache_writable,
            "disk_free_gib": disk_free_gib,
            "peptides": len(peptides),
            "development_peptides": development_n,
            "test_peptides": test_n,
            "minimum_sequence_length": int(sequence_lengths.min()),
            "maximum_sequence_length": int(sequence_lengths.max()),
            "mean_sequence_length": float(sequence_lengths.mean()),
            "invalid_amino_acid_rows": len(invalid_character_rows),
            "sequences_ready": sequences_ready,
            "esm2_max_residues_checked": ESM2_MAX_RESIDUES,
            "all_sequences_within_limit": lengths_within_limit,
            "recommended_device": recommended_device,
            "environment_ready_without_model_download": environment_ready,
        }
    ]
)
REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(REPORT_OUTPUT, index=False)

print("\n" + "=" * 90)
print("STEP 41 SUMMARY")
print("=" * 90)
print("Environment ready without model download:", environment_ready)
print("Recommended device:", recommended_device)
print("Target model:", MODEL_ID)
print("Expected pooled dimensions:", EXPECTED_EMBEDDING_DIM)
print("Peptides ready:", sequences_ready)
print("Lengths safely within limit:", lengths_within_limit)
print("Model downloaded:", False)
print("Environment report:", REPORT_OUTPUT)
print("\nSTEP 41 COMPLETED SUCCESSFULLY")
print("=" * 90)

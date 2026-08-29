from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
NPY_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
CSV_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.csv"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
FEATURE_INPUT = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_INPUT = PROJECT_DIR / "derived" / "fixed_split.csv"
CV_CSV_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.csv"
CV_NPZ_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
SUMMARY_OUTPUT = PROJECT_DIR / "results" / "step45_esm2_embedding_verification_summary.csv"
DIMENSION_OUTPUT = PROJECT_DIR / "results" / "step45_esm2_dimension_statistics.csv"
EXPECTED_SHAPE = (901, 1280)
EXPECTED_DTYPE = np.dtype("float32")
NEAR_ZERO_SD_THRESHOLD = 1e-8
ZERO_NORM_THRESHOLD = 1e-12

print("=" * 94)
print("STEP 45 - VERIFY ESM-2 EMBEDDING INTEGRITY")
print("=" * 94)

required_files = [
    NPY_INPUT, CSV_INPUT, METADATA_INPUT, FEATURE_INPUT, SPLIT_INPUT,
    CV_CSV_INPUT, CV_NPZ_INPUT,
]
for required_file in required_files:
    if not required_file.exists():
        raise FileNotFoundError(f"Required file not found: {required_file}")

print("\n45A. Load independently saved artifacts:")
X = np.load(NPY_INPUT, allow_pickle=False)
embedding_csv = pd.read_csv(CSV_INPUT)
metadata = pd.read_csv(METADATA_INPUT)
features = pd.read_csv(FEATURE_INPUT)
split = pd.read_csv(SPLIT_INPUT)
cv_table = pd.read_csv(CV_CSV_INPUT)
cv_indices = np.load(CV_NPZ_INPUT)
print("NPY shape:", X.shape)
print("NPY dtype:", X.dtype)
print("Embedding CSV shape:", embedding_csv.shape)
print("Metadata rows:", len(metadata))

if X.shape != EXPECTED_SHAPE:
    raise ValueError(f"Expected NPY shape {EXPECTED_SHAPE}, found {X.shape}.")
if X.dtype != EXPECTED_DTYPE:
    raise ValueError(f"Expected dtype float32, found {X.dtype}.")
if embedding_csv.shape != EXPECTED_SHAPE:
    raise ValueError("Embedding CSV has the wrong shape.")
if len(metadata) != EXPECTED_SHAPE[0]:
    raise ValueError("Metadata has the wrong row count.")

print("\n45B. Numerical integrity:")
missing_values = int(np.isnan(X).sum())
nonfinite_values = int((~np.isfinite(X)).sum())
unique_rows = int(np.unique(X, axis=0).shape[0])
duplicate_rows = X.shape[0] - unique_rows
if missing_values or nonfinite_values or duplicate_rows:
    raise ValueError("Embedding matrix failed missing/non-finite/duplicate checks.")
print("Missing values:", missing_values)
print("Non-finite values:", nonfinite_values)
print("Unique rows:", unique_rows)
print("Duplicate rows:", duplicate_rows)

print("\n45C. NPY/CSV agreement:")
csv_values = embedding_csv.to_numpy(dtype=np.float32)
csv_values_match = bool(np.array_equal(csv_values, X))
maximum_csv_absolute_difference = float(np.max(np.abs(csv_values - X)))
if not csv_values_match:
    raise ValueError("CSV values do not exactly reproduce the float32 NPY matrix.")
expected_columns = [f"esm2_{index:04d}" for index in range(1, 1281)]
csv_column_names_match = embedding_csv.columns.tolist() == expected_columns
if not csv_column_names_match:
    raise ValueError("Embedding CSV column names are incomplete or out of order.")
print("CSV values match NPY:", csv_values_match)
print("Maximum absolute difference:", maximum_csv_absolute_difference)
print("CSV dimension names/order correct:", csv_column_names_match)

print("\n45D. Metadata and original-order alignment:")
expected_metadata_columns = [
    "embedding_row", "ID", "sequence", "sequence_length", "label",
    "binary_class", "inactive_source", "is_virtual_inactive", "split",
]
if metadata.columns.tolist() != expected_metadata_columns:
    raise ValueError("Metadata columns are incomplete or out of order.")
embedding_row_match = bool(
    np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(EXPECTED_SHAPE[0]))
)
id_alignment = metadata["ID"].equals(features["ID"]) and metadata["ID"].equals(split["ID"])
sequence_alignment = (
    metadata["sequence"].equals(features["sequence"])
    and metadata["sequence"].equals(split["sequence"])
)
label_alignment = metadata["label"].equals(features["label"]) and metadata["label"].equals(split["label"])
binary_alignment = metadata["binary_class"].equals(features["binary_class"])
length_alignment = bool(
    np.array_equal(metadata["sequence_length"].to_numpy(), metadata["sequence"].str.len().to_numpy())
)
split_alignment = metadata["split"].equals(split["split"])
if not all(
    [embedding_row_match, id_alignment, sequence_alignment, label_alignment,
     binary_alignment, length_alignment, split_alignment]
):
    raise ValueError("Metadata alignment verification failed.")
print("embedding_row order:", embedding_row_match)
print("ID alignment:", id_alignment)
print("Sequence alignment:", sequence_alignment)
print("Label alignment:", label_alignment)
print("Sequence-length alignment:", length_alignment)
print("Split alignment:", split_alignment)

print("\n45E. Development/test preservation:")
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
if len(development_indices) != 720 or len(test_indices) != 181:
    raise ValueError("Expected a 720/181 development/test split.")
if np.intersect1d(development_indices, test_indices).size:
    raise ValueError("Development and test rows overlap.")
if np.union1d(development_indices, test_indices).size != EXPECTED_SHAPE[0]:
    raise ValueError("Development/test indices do not cover all embedding rows.")
X_dev = X[development_indices]
X_test = X[test_indices]
print("Development matrix:", X_dev.shape)
print("Test matrix:", X_test.shape)
print("Development/test overlap:", 0)

print("\n45F. Fixed 5-fold CV compatibility:")
if len(cv_table) != EXPECTED_SHAPE[0]:
    raise ValueError("CV table does not contain 901 rows.")
if not cv_table["ID"].equals(metadata["ID"]):
    raise ValueError("CV table IDs are not aligned with embedding rows.")
if not cv_table["sequence"].equals(metadata["sequence"]):
    raise ValueError("CV table sequences are not aligned with embedding rows.")
if not cv_table["split"].equals(metadata["split"]):
    raise ValueError("CV table split labels are not aligned with metadata.")
if not cv_table.loc[test_indices, "cv_fold"].eq(-1).all():
    raise ValueError("At least one locked-test peptide has a CV fold assignment.")
if not set(cv_table.loc[development_indices, "cv_fold"].unique()) == {1, 2, 3, 4, 5}:
    raise ValueError("Development CV labels are not exactly folds 1-5.")

saved_development = cv_indices["development_global_indices"].astype(int)
if not np.array_equal(saved_development, development_indices):
    raise ValueError("NPZ development indices do not match embedding metadata.")

all_cv_indices = []
all_validation_indices = []
fold_summary = []
for fold_number in range(1, 6):
    train_indices = cv_indices[f"fold{fold_number}_train"].astype(int)
    valid_indices = cv_indices[f"fold{fold_number}_valid"].astype(int)
    expected_valid = np.flatnonzero(cv_table["cv_fold"].eq(fold_number).to_numpy())
    expected_train = np.setdiff1d(development_indices, expected_valid)
    if not np.array_equal(valid_indices, expected_valid):
        raise ValueError(f"Fold {fold_number} validation indices do not match the CV table.")
    if not np.array_equal(train_indices, expected_train):
        raise ValueError(f"Fold {fold_number} training indices do not match development-minus-validation.")
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError(f"Fold {fold_number} has train/validation overlap.")
    if not np.all(np.isin(train_indices, development_indices)):
        raise ValueError(f"Fold {fold_number} training rows include non-development data.")
    if not np.all(np.isin(valid_indices, development_indices)):
        raise ValueError(f"Fold {fold_number} validation rows include non-development data.")
    fold_test_overlap = int(
        np.intersect1d(np.concatenate([train_indices, valid_indices]), test_indices).size
    )
    if fold_test_overlap:
        raise ValueError(f"Fold {fold_number} contains locked-test rows.")
    if X[train_indices].shape != (576, 1280) or X[valid_indices].shape != (144, 1280):
        raise ValueError(f"Fold {fold_number} embedding matrix sizes are incorrect.")
    all_cv_indices.extend(train_indices.tolist())
    all_cv_indices.extend(valid_indices.tolist())
    all_validation_indices.extend(valid_indices.tolist())
    fold_summary.append(f"fold{fold_number}:576/144")

cv_test_overlap = np.intersect1d(np.unique(all_cv_indices), test_indices)
validation_coverage = np.asarray(all_validation_indices, dtype=int)
validation_unique = np.unique(validation_coverage)
if not np.array_equal(validation_unique, development_indices):
    raise ValueError("Five validation folds do not cover every development row exactly once.")
if len(validation_coverage) != len(validation_unique):
    raise ValueError("A development row appears in more than one validation fold.")
print("Fold train/validation sizes:", ", ".join(fold_summary))
print("Development validation coverage:", len(validation_unique))
print("CV test overlap:", len(cv_test_overlap))

print("\n45G. Dimension statistics:")
dimension_means = X.mean(axis=0, dtype=np.float64)
dimension_sds = X.std(axis=0, dtype=np.float64, ddof=0)
dimension_variances = X.var(axis=0, dtype=np.float64, ddof=0)
dimension_minima = X.min(axis=0)
dimension_maxima = X.max(axis=0)
dimension_ranges = dimension_maxima - dimension_minima
development_means = X_dev.mean(axis=0, dtype=np.float64)
development_sds = X_dev.std(axis=0, dtype=np.float64, ddof=0)
constant_mask = dimension_ranges == 0
near_zero_mask = dimension_sds < NEAR_ZERO_SD_THRESHOLD
constant_dimensions = int(constant_mask.sum())
near_zero_variance_dimensions = int(near_zero_mask.sum())
dimension_statistics = pd.DataFrame(
    {
        "dimension_index_zero_based": np.arange(EXPECTED_SHAPE[1]),
        "dimension_name": expected_columns,
        "overall_mean": dimension_means,
        "overall_sd": dimension_sds,
        "overall_variance": dimension_variances,
        "overall_minimum": dimension_minima,
        "overall_maximum": dimension_maxima,
        "overall_range": dimension_ranges,
        "development_mean": development_means,
        "development_sd": development_sds,
        "constant_overall": constant_mask,
        "near_zero_sd_overall": near_zero_mask,
    }
)
print("Constant dimensions:", constant_dimensions)
print("Near-zero SD threshold:", NEAR_ZERO_SD_THRESHOLD)
print("Near-zero SD dimensions:", near_zero_variance_dimensions)
print("Minimum dimension SD:", float(dimension_sds.min()))
print("Median dimension SD:", float(np.median(dimension_sds)))
print("Maximum dimension SD:", float(dimension_sds.max()))

print("\n45H. Row norms:")
row_norms = np.linalg.norm(X.astype(np.float64), axis=1)
zero_norm_rows = int((row_norms <= ZERO_NORM_THRESHOLD).sum())
if zero_norm_rows:
    raise ValueError("One or more embedding rows have zero norm.")
print("Zero-norm rows:", zero_norm_rows)
print("Minimum row norm:", float(row_norms.min()))
print("Mean row norm:", float(row_norms.mean()))
print("Median row norm:", float(np.median(row_norms)))
print("Maximum row norm:", float(row_norms.max()))

SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
dimension_statistics.to_csv(DIMENSION_OUTPUT, index=False)
summary = pd.DataFrame(
    [
        {
            "embedding_rows": X.shape[0],
            "embedding_dimensions": X.shape[1],
            "embedding_dtype": str(X.dtype),
            "missing_values": missing_values,
            "non_finite_values": nonfinite_values,
            "unique_rows": unique_rows,
            "duplicate_rows": duplicate_rows,
            "constant_dimensions": constant_dimensions,
            "near_zero_sd_threshold": NEAR_ZERO_SD_THRESHOLD,
            "near_zero_sd_dimensions": near_zero_variance_dimensions,
            "minimum_dimension_sd": float(dimension_sds.min()),
            "median_dimension_sd": float(np.median(dimension_sds)),
            "maximum_dimension_sd": float(dimension_sds.max()),
            "zero_norm_rows": zero_norm_rows,
            "minimum_row_norm": float(row_norms.min()),
            "mean_row_norm": float(row_norms.mean()),
            "median_row_norm": float(np.median(row_norms)),
            "maximum_row_norm": float(row_norms.max()),
            "development_rows": len(development_indices),
            "test_rows": len(test_indices),
            "development_dimensions": X_dev.shape[1],
            "test_dimensions": X_test.shape[1],
            "cv_folds": 5,
            "cv_validation_coverage": len(validation_unique),
            "cv_test_overlap": len(cv_test_overlap),
            "csv_values_match_npy": csv_values_match,
            "maximum_csv_absolute_difference": maximum_csv_absolute_difference,
            "metadata_alignment_verified": True,
            "split_alignment_verified": True,
            "cv_alignment_verified": True,
            "classifier_trained": False,
        }
    ]
)
summary.to_csv(SUMMARY_OUTPUT, index=False)

print("\n45R. Output checks:")
print("Verification summary exists:", SUMMARY_OUTPUT.exists())
print("Dimension statistics exists:", DIMENSION_OUTPUT.exists())

print("\n" + "=" * 94)
print("STEP 45 SUMMARY")
print("=" * 94)
print("Embedding matrix:", X.shape)
print("dtype:", X.dtype)
print("Missing:", missing_values)
print("Non-finite:", nonfinite_values)
print("Unique rows:", unique_rows)
print("Duplicate rows:", duplicate_rows)
print("Constant dimensions:", constant_dimensions)
print("Near-zero SD dimensions:", near_zero_variance_dimensions)
print("Zero-norm rows:", zero_norm_rows)
print("Development matrix:", X_dev.shape)
print("Test matrix:", X_test.shape)
print("CV test overlap:", len(cv_test_overlap))
print("CSV values match NPY:", csv_values_match)
print("\nVerification summary:", SUMMARY_OUTPUT)
print("Dimension statistics:", DIMENSION_OUTPUT)
print("\nSTEP 45 COMPLETED SUCCESSFULLY")
print("=" * 94)

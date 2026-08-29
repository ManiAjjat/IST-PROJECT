from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
RAW_FILE = PROJECT_DIR / "data" / "ACPs_Lung_cancer.csv"
QC_OUTPUT = PROJECT_DIR / "results" / "traditional_feature_qc_summary.csv"

metadata_columns = [
    "ID",
    "sequence",
    "class",
    "original_class",
    "label",
    "binary_class",
    "inactive_source",
    "is_virtual_inactive",
]

df = pd.read_csv(FEATURE_FILE)
raw = pd.read_csv(RAW_FILE)
feature_columns = [column for column in df.columns if column not in metadata_columns]

print("=" * 70)
print("STEP 22 - COMPREHENSIVE QC OF TRADITIONAL FEATURES")
print("=" * 70)
print("Feature file:", FEATURE_FILE)

missing_total = int(df[feature_columns].isna().sum().sum())
numeric_values = df[feature_columns].apply(pd.to_numeric, errors="coerce")
nonfinite_total = int((~np.isfinite(numeric_values.to_numpy())).sum())
constant_features = [
    column for column in feature_columns if df[column].nunique(dropna=False) <= 1
]
duplicate_ids = int(df["ID"].duplicated().sum())
duplicate_sequences = int(df["sequence"].duplicated().sum())
aac_columns = [column for column in feature_columns if column.startswith("AAC_")]
aac_sums = df[aac_columns].sum(axis=1)
aac_failures = int((aac_sums.sub(1).abs() > 1e-9).sum())
ids_preserved = raw["ID"].equals(df["ID"])
sequences_preserved = raw["sequence"].equals(df["sequence"])
classes_preserved = raw["class"].equals(df["class"])
active_count = int((df["label"] == 1).sum())
inactive_count = int((df["label"] == 0).sum())

print("\n22A. Matrix dimensions:")
print("Total peptides:", len(df))
print("Total CSV columns:", len(df.columns))
print("ML features:", len(feature_columns))
print("Metadata columns:", len(metadata_columns))
print("\n22B. Feature validity:")
print("Missing feature values:", missing_total)
print("Non-finite feature values:", nonfinite_total)
print("Constant features:", constant_features)
print("\n22C. Duplicate and preservation checks:")
print("Duplicate IDs:", duplicate_ids)
print("Duplicate sequences:", duplicate_sequences)
print("IDs preserved:", ids_preserved)
print("Sequences preserved:", sequences_preserved)
print("Classes preserved:", classes_preserved)
print("\n22D. AAC check:")
print("AAC sum failures:", aac_failures)
print("Minimum AAC sum:", round(aac_sums.min(), 6))
print("Maximum AAC sum:", round(aac_sums.max(), 6))
print("\n22E. Class counts:")
print("Active peptides:", active_count)
print("Inactive peptides:", inactive_count)

if len(df) != 901 or len(df.columns) != 40 or len(feature_columns) != 32:
    raise ValueError("Unexpected traditional-feature matrix dimensions.")
if missing_total != 0 or nonfinite_total != 0:
    raise ValueError("Missing or non-finite feature values found.")
if constant_features:
    raise ValueError(f"Constant features found: {constant_features}")
if duplicate_ids != 0 or duplicate_sequences != 0:
    raise ValueError("Duplicate IDs or sequences found.")
if aac_failures != 0:
    raise ValueError("AAC values do not sum to 1 for every peptide.")
if not ids_preserved or not sequences_preserved or not classes_preserved:
    raise ValueError("Raw identifiers or classes were not preserved.")
if active_count != 99 or inactive_count != 802:
    raise ValueError("Unexpected binary class counts.")

range_table = pd.DataFrame(
    {
        "feature": feature_columns,
        "minimum": [df[column].min() for column in feature_columns],
        "maximum": [df[column].max() for column in feature_columns],
    }
)
range_table["n_unique"] = [df[column].nunique() for column in feature_columns]
range_table["missing"] = [df[column].isna().sum() for column in feature_columns]
range_table.to_csv(QC_OUTPUT, index=False)

print("\n" + "=" * 70)
print("STEP 22 SUMMARY")
print("=" * 70)
print("Total peptides:", len(df))
print("ML features:", len(feature_columns))
print("Missing feature values:", missing_total)
print("Non-finite feature values:", nonfinite_total)
print("Duplicate IDs:", duplicate_ids)
print("Duplicate sequences:", duplicate_sequences)
print("AAC sum failures:", aac_failures)
print("Constant features:", constant_features)
print("\nQC summary saved to:")
print(QC_OUTPUT)
print("\nSTEP 22 COMPLETED SUCCESSFULLY")
print("=" * 70)
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
INDEX_FILE = PROJECT_DIR / "derived" / "fixed_split_indices.npz"
QC_OUTPUT = PROJECT_DIR / "results" / "step28_split_verification.csv"

features = pd.read_csv(FEATURE_FILE)
split = pd.read_csv(SPLIT_FILE)
indices = np.load(INDEX_FILE)
development_indices = indices["development_indices"]
test_indices = indices["test_indices"]

if len(features) != len(split):
    raise ValueError("Master and split files have different row counts.")

ids_identical = features["ID"].equals(split["ID"])
sequences_identical = features["sequence"].equals(split["sequence"])
labels_identical = features["label"].equals(split["label"])
csv_development_indices = np.flatnonzero(split["split"].eq("development"))
csv_test_indices = np.flatnonzero(split["split"].eq("test"))
dev_indices_match = np.array_equal(csv_development_indices, development_indices)
test_indices_match = np.array_equal(csv_test_indices, test_indices)
overlap = set(development_indices).intersection(test_indices)
duplicate_split_ids = int(split["ID"].duplicated().sum())
duplicate_split_sequences = int(split["sequence"].duplicated().sum())

development = split[split["split"] == "development"]
test = split[split["split"] == "test"]
dev_csv_count = len(development)
test_csv_count = len(test)
dev_active = int((development["label"] == 1).sum())
dev_inactive = int((development["label"] == 0).sum())
test_active = int((test["label"] == 1).sum())
test_inactive = int((test["label"] == 0).sum())

summary = pd.DataFrame(
    [
        {"check": "total_peptides", "value": len(features)},
        {"check": "development_count", "value": dev_csv_count},
        {"check": "test_count", "value": test_csv_count},
        {"check": "development_active", "value": dev_active},
        {"check": "development_inactive", "value": dev_inactive},
        {"check": "test_active", "value": test_active},
        {"check": "test_inactive", "value": test_inactive},
        {"check": "overlap", "value": len(overlap)},
        {"check": "duplicate_ids", "value": duplicate_split_ids},
        {"check": "duplicate_sequences", "value": duplicate_split_sequences},
        {"check": "ids_preserved", "value": ids_identical},
        {"check": "sequences_preserved", "value": sequences_identical},
        {"check": "labels_preserved", "value": labels_identical},
        {"check": "development_indices_match", "value": dev_indices_match},
        {"check": "test_indices_match", "value": test_indices_match},
    ]
)

assert len(features) == 901
assert dev_csv_count == 720
assert test_csv_count == 181
assert dev_active == 79
assert dev_inactive == 641
assert test_active == 20
assert test_inactive == 161
assert len(overlap) == 0
assert duplicate_split_ids == 0
assert duplicate_split_sequences == 0
assert ids_identical and sequences_identical and labels_identical
assert dev_indices_match and test_indices_match

summary.to_csv(QC_OUTPUT, index=False)

print("\n" + "=" * 78)
print("STEP 28 SUMMARY")
print("=" * 78)
print("Total peptides:", len(features))
print("Development peptides:", dev_csv_count)
print("Test peptides:", test_csv_count)
print("Development Active:", dev_active)
print("Development Inactive:", dev_inactive)
print("Test Active:", test_active)
print("Test Inactive:", test_inactive)
print("IDs preserved:", ids_identical)
print("Sequences preserved:", sequences_identical)
print("Labels preserved:", labels_identical)
print("Development indices match:", dev_indices_match)
print("Test indices match:", test_indices_match)
print("Development/Test overlap:", len(overlap))
print("Duplicate IDs:", duplicate_split_ids)
print("Duplicate sequences:", duplicate_split_sequences)
print("\nQC output:")
print(QC_OUTPUT)
print("\nSTEP 28 COMPLETED SUCCESSFULLY")
print("=" * 78)
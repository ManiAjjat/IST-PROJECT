from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_OUTPUT = PROJECT_DIR / "derived" / "fixed_split.csv"
INDEX_OUTPUT = PROJECT_DIR / "derived" / "fixed_split_indices.npz"
SEED = 2026

df = pd.read_csv(FEATURE_FILE)
indices = np.arange(len(df))
development_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

split = np.full(len(df), "development", dtype=object)
split[test_indices] = "test"
split_df = df.copy()
split_df["split"] = split
split_df.to_csv(SPLIT_OUTPUT, index=False)
np.savez(
    INDEX_OUTPUT,
    development_indices=np.sort(development_indices),
    test_indices=np.sort(test_indices),
    seed=np.array(SEED),
)

development = split_df[split_df["split"] == "development"]
test = split_df[split_df["split"] == "test"]
overlap = set(development_indices).intersection(test_indices)
assigned = len(development_indices) + len(test_indices)
unassigned = len(df) - assigned
dev_active = int((development["label"] == 1).sum())
dev_inactive = int((development["label"] == 0).sum())
test_active = int((test["label"] == 1).sum())
test_inactive = int((test["label"] == 0).sum())

assert len(df) == 901
assert len(development) == 720
assert len(test) == 181
assert len(overlap) == 0
assert unassigned == 0
assert dev_active == 79
assert dev_inactive == 641
assert test_active == 20
assert test_inactive == 161

print("=" * 78)
print("STEP 27 - CREATE FIXED 80/20 TRAIN/TEST PARTITION")
print("=" * 78)
print("Total peptides:", len(df))
print("Development peptides:", len(development))
print("Test peptides:", len(test))
print("Development Active:", dev_active)
print("Development Inactive:", dev_inactive)
print("Test Active:", test_active)
print("Test Inactive:", test_inactive)
print("Development/Test overlap:", len(overlap))
print("Unassigned peptides:", unassigned)
print("Random seed:", SEED)
print("\nSplit assignment file:")
print(SPLIT_OUTPUT)
print("\nExact split indices:")
print(INDEX_OUTPUT)
print("\nSTEP 27 COMPLETED SUCCESSFULLY")
print("=" * 78)
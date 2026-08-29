from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
CV_OUTPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.csv"
CV_NPZ_OUTPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
FOLD_SUMMARY_OUTPUT = PROJECT_DIR / "results" / "step29_cv_fold_summary.csv"
N_SPLITS = 5
SEED = 2026

df = pd.read_csv(SPLIT_FILE)
development_mask = df["split"].eq("development").to_numpy()
development_indices_global = np.flatnonzero(development_mask)
test_indices_global = np.flatnonzero(~development_mask)
dev = df.iloc[development_indices_global].reset_index(drop=True)

splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_assignments = np.full(len(df), -1, dtype=int)
fold_train_indices = []
fold_valid_indices = []
fold_summary_rows = []

for fold_number, (train_dev, valid_dev) in enumerate(
    splitter.split(dev, dev["label"]),
    start=1,
):
    train_global = development_indices_global[train_dev]
    valid_global = development_indices_global[valid_dev]
    fold_assignments[valid_global] = fold_number
    fold_train_indices.append(np.sort(train_global))
    fold_valid_indices.append(np.sort(valid_global))
    fold_train = df.iloc[train_global]
    fold_valid = df.iloc[valid_global]
    train_valid_overlap = len(set(train_global).intersection(valid_global))
    fold_summary_rows.append(
        {
            "fold": fold_number,
            "training_n": len(fold_train),
            "validation_n": len(fold_valid),
            "training_active": int((fold_train["label"] == 1).sum()),
            "training_inactive": int((fold_train["label"] == 0).sum()),
            "validation_active": int((fold_valid["label"] == 1).sum()),
            "validation_inactive": int((fold_valid["label"] == 0).sum()),
            "active_fraction": float(fold_valid["label"].mean()),
            "train_validation_overlap": train_valid_overlap,
        }
    )

cv_output_df = df.copy()
cv_output_df["cv_fold"] = fold_assignments
cv_output_df.to_csv(CV_OUTPUT, index=False)

validation_counts_by_peptide = np.zeros(len(df), dtype=int)
for valid_indices in fold_valid_indices:
    validation_counts_by_peptide[valid_indices] += 1
development_validation_counts = validation_counts_by_peptide[
    development_indices_global
]
validation_once = int((development_validation_counts == 1).sum())
validation_zero = int((development_validation_counts == 0).sum())
validation_more_than_once = int((development_validation_counts > 1).sum())
test_cv_overlap = set(test_indices_global).intersection(
    np.flatnonzero(fold_assignments > 0)
)
fold_overlap_counts = [
    row["train_validation_overlap"] for row in fold_summary_rows
]
fold_summary = pd.DataFrame(fold_summary_rows)
fold_summary.to_csv(FOLD_SUMMARY_OUTPUT, index=False)

np.savez(
    CV_NPZ_OUTPUT,
    fold1_train=fold_train_indices[0],
    fold1_valid=fold_valid_indices[0],
    fold2_train=fold_train_indices[1],
    fold2_valid=fold_valid_indices[1],
    fold3_train=fold_train_indices[2],
    fold3_valid=fold_valid_indices[2],
    fold4_train=fold_train_indices[3],
    fold4_valid=fold_valid_indices[3],
    fold5_train=fold_train_indices[4],
    fold5_valid=fold_valid_indices[4],
    development_global_indices=development_indices_global,
)

assert len(dev) == 720
assert len(test_indices_global) == 181
assert len(test_cv_overlap) == 0
assert validation_once == 720
assert validation_zero == 0
assert validation_more_than_once == 0
assert fold_overlap_counts == [0, 0, 0, 0, 0]
assert all(row["validation_n"] == 144 for row in fold_summary_rows)

print("\n" + "=" * 80)
print("STEP 29 SUMMARY")
print("=" * 80)
print("Development peptides:", len(dev))
print("Independent test peptides used in CV:", len(test_cv_overlap))
print("Number of folds:", N_SPLITS)
print("Random seed:", SEED)
print("Validation exactly once:", validation_once)
print("Validation zero times:", validation_zero)
print("Validation more than once:", validation_more_than_once)
print("Train/validation overlaps:", fold_overlap_counts)
print("\nFold summary:")
print(fold_summary.to_string(index=False))
print("\nCV assignments:")
print(CV_OUTPUT)
print("\nCV indices:")
print(CV_NPZ_OUTPUT)
print("\nFold summary:")
print(FOLD_SUMMARY_OUTPUT)
print("\nSTEP 29 COMPLETED SUCCESSFULLY")
print("=" * 80)
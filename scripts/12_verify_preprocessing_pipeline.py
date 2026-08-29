from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
CV_FILE = PROJECT_DIR / "derived" / "fixed_cv_folds.csv"
CV_INDEX_FILE = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
QC_OUTPUT = PROJECT_DIR / "results" / "step30_preprocessing_verification.csv"

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
features = pd.read_csv(FEATURE_FILE)
split = pd.read_csv(SPLIT_FILE)
cv_assignments = pd.read_csv(CV_FILE)
cv_indices = np.load(CV_INDEX_FILE)
feature_columns = [column for column in features.columns if column not in metadata_columns]
feature_values = features[feature_columns].to_numpy(dtype=float)
missing_dev = int(np.isnan(feature_values[split["split"].eq("development")]).sum())
missing_test = int(np.isnan(feature_values[split["split"].eq("test")]).sum())
nonfinite_dev = int((~np.isfinite(feature_values[split["split"].eq("development")])).sum())
nonfinite_test = int((~np.isfinite(feature_values[split["split"].eq("test")])).sum())

if not split["ID"].equals(cv_assignments["ID"]):
    raise ValueError("Split and CV assignment rows are not aligned.")
observed_folds = sorted(
    int(value)
    for value in cv_assignments.loc[
        split["split"] == "development", "cv_fold"
    ].unique()
)
test_global_indices = set(np.flatnonzero(split["split"].eq("test")))
test_cv_overlap = set()
wrong_scaler_sample_counts = []
fold_rows = []

for fold_number in observed_folds:
    valid_global = np.flatnonzero(cv_assignments["cv_fold"].eq(fold_number))
    train_global = np.array(cv_indices[f"fold{fold_number}_train"])
    test_cv_overlap.update(set(valid_global).intersection(test_global_indices))
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(feature_values[train_global])
    scaled_valid = scaler.transform(feature_values[valid_global])
    samples_seen = int(scaler.n_samples_seen_)
    constant_training_features = [
        feature_columns[index]
        for index, scale in enumerate(scaler.scale_)
        if scale == 1.0 and np.ptp(feature_values[train_global, index]) == 0
    ]
    if samples_seen != len(train_global):
        wrong_scaler_sample_counts.append(fold_number)
    fold_rows.append(
        {
            "fold": fold_number,
            "training_rows": len(train_global),
            "validation_rows": len(valid_global),
            "scaler_samples_seen": samples_seen,
            "max_abs_training_scaled_mean": float(np.abs(scaled_train.mean(axis=0)).max()),
            "max_abs_training_scaled_std_error": float(np.abs(scaled_train.std(axis=0) - 1).max()),
            "max_abs_validation_scaled_mean": float(np.abs(scaled_valid.mean(axis=0)).max()),
            "constant_training_features": ",".join(constant_training_features),
            "test_overlap": len(set(valid_global).intersection(test_global_indices)),
        }
    )

if len(features) != 901 or len(feature_columns) != 32:
    raise ValueError("Unexpected feature matrix dimensions.")
if len(observed_folds) != 5:
    raise ValueError("Expected five observed CV folds.")
if missing_dev != 0 or missing_test != 0:
    raise ValueError("Missing feature values found.")
if nonfinite_dev != 0 or nonfinite_test != 0:
    raise ValueError("Non-finite feature values found.")
if test_cv_overlap:
    raise ValueError("Independent test peptides appeared in CV.")
if wrong_scaler_sample_counts:
    raise ValueError("A scaler was fitted on an unexpected number of samples.")

qc_df = pd.DataFrame(fold_rows)
qc_df.to_csv(QC_OUTPUT, index=False)

print("\n" + "=" * 80)
print("STEP 30 SUMMARY")
print("=" * 80)
print("Development peptides:", int(split["split"].eq("development").sum()))
print("Independent test peptides:", int(split["split"].eq("test").sum()))
print("ML features:", len(feature_columns))
print("CV folds:", len(observed_folds))
print("Test peptides appearing in CV:", len(test_cv_overlap))
print("Folds with incorrect scaler sample count:", wrong_scaler_sample_counts)
print("Development missing values:", missing_dev)
print("Development non-finite values:", nonfinite_dev)
print("Test missing values:", missing_test)
print("Test non-finite values:", nonfinite_test)
print("\nFold checks:")
print(qc_df.to_string(index=False))
print("\nScaling design:")
print("Logistic Regression / SVM:")
print("Training fold -> StandardScaler.fit -> StandardScaler.transform -> model.fit")
print("Validation fold -> SAME fitted scaler.transform -> model.predict")
print("Random Forest / XGBoost:")
print("Training fold -> model.fit")
print("Validation fold -> model.predict")
print("Independent test set: NOT USED during CV or hyperparameter selection.")
print("\nQC output:")
print(QC_OUTPUT)
print("\nSTEP 30 COMPLETED SUCCESSFULLY")
print("=" * 80)
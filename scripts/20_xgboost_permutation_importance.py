from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
CV_INDEX_FILE = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
PARAM_FILE = PROJECT_DIR / "results" / "step34_xgboost_best_params.json"
SUMMARY_OUTPUT = PROJECT_DIR / "results" / "step38_xgboost_permutation_importance.csv"
DETAIL_OUTPUT = PROJECT_DIR / "results" / "step38_xgboost_permutation_importance_fold_details.csv"
SEED = 2026
N_REPEATS = 30
N_FOLDS = 5

metadata_columns = [
    "ID", "sequence", "class", "original_class", "label",
    "binary_class", "inactive_source", "is_virtual_inactive",
]

for required_file in (FEATURE_FILE, SPLIT_FILE, CV_INDEX_FILE, PARAM_FILE):
    if not required_file.exists():
        raise FileNotFoundError(f"Required input not found: {required_file}")

df = pd.read_csv(FEATURE_FILE)
split = pd.read_csv(SPLIT_FILE)
if len(df) != len(split):
    raise ValueError("Feature and split tables have different row counts.")
if "label" not in df.columns or "split" not in split.columns:
    raise ValueError("Required label or split column is missing.")

feature_columns = [column for column in df.columns if column not in metadata_columns]
if len(feature_columns) != 32:
    raise ValueError(f"Expected 32 traditional features, found {len(feature_columns)}.")
if df[feature_columns].isna().any().any():
    raise ValueError("Traditional feature matrix contains missing values.")

X = df[feature_columns].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(split["split"].eq("development"))
test_indices = np.flatnonzero(split["split"].eq("test"))
development_index_set = set(development_indices.tolist())
if len(development_indices) != 720 or len(test_indices) != 181:
    raise ValueError("Expected 720 development and 181 locked-test peptides.")

with PARAM_FILE.open("r", encoding="utf-8") as parameter_file:
    saved_params = json.load(parameter_file)
best_params = {
    "n_estimators": int(saved_params["n_estimators"]),
    "max_depth": int(saved_params["max_depth"]),
    "learning_rate": float(saved_params["learning_rate"]),
    "subsample": float(saved_params["subsample"]),
    "colsample_bytree": float(saved_params["colsample_bytree"]),
}


def make_model(scale_pos_weight):
    return XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


cv_indices = np.load(CV_INDEX_FILE)
detail_rows = []
baseline_rows = []

print("=" * 90)
print("STEP 38 - CROSS-VALIDATED XGBOOST PERMUTATION IMPORTANCE")
print("=" * 90)
print("Frozen parameters:", best_params)
print("Features:", len(feature_columns))
print("Permutation repeats:", N_REPEATS)

for fold_number in range(1, N_FOLDS + 1):
    train_indices = cv_indices[f"fold{fold_number}_train"].astype(int)
    valid_indices = cv_indices[f"fold{fold_number}_valid"].astype(int)

    if not set(train_indices.tolist()).issubset(development_index_set):
        raise ValueError(f"Fold {fold_number} training indices include locked-test rows.")
    if not set(valid_indices.tolist()).issubset(development_index_set):
        raise ValueError(f"Fold {fold_number} validation indices include locked-test rows.")
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError(f"Fold {fold_number} has training-validation overlap.")
    if len(train_indices) != 576 or len(valid_indices) != 144:
        raise ValueError(f"Fold {fold_number} does not have the expected 576/144 sizes.")

    train_labels = y[train_indices]
    valid_labels = y[valid_indices]
    scale_pos_weight = int((train_labels == 0).sum()) / int(train_labels.sum())
    model = make_model(scale_pos_weight)
    model.fit(X[train_indices], train_labels)

    X_valid = X[valid_indices].copy()
    baseline_probability = model.predict_proba(X_valid)[:, 1]
    baseline_auroc = float(roc_auc_score(valid_labels, baseline_probability))
    baseline_rows.append({
        "fold": fold_number,
        "training_n": len(train_indices),
        "validation_n": len(valid_indices),
        "baseline_validation_AUROC": baseline_auroc,
    })

    for feature_index, feature_name in enumerate(feature_columns):
        drops = []
        permuted_aurocs = []
        for repeat_number in range(1, N_REPEATS + 1):
            rng = np.random.default_rng(SEED + fold_number * 100_000 + feature_index * 1_000 + repeat_number)
            X_permuted = X_valid.copy()
            X_permuted[:, feature_index] = rng.permutation(X_permuted[:, feature_index])
            permuted_probability = model.predict_proba(X_permuted)[:, 1]
            permuted_auroc = float(roc_auc_score(valid_labels, permuted_probability))
            permuted_aurocs.append(permuted_auroc)
            drops.append(baseline_auroc - permuted_auroc)

        detail_rows.append({
            "fold": fold_number,
            "feature": feature_name,
            "training_n": len(train_indices),
            "validation_n": len(valid_indices),
            "training_active": int(train_labels.sum()),
            "validation_active": int(valid_labels.sum()),
            "scale_pos_weight": scale_pos_weight,
            "baseline_validation_AUROC": baseline_auroc,
            "mean_permuted_AUROC": float(np.mean(permuted_aurocs)),
            "mean_AUROC_drop": float(np.mean(drops)),
            "sd_AUROC_drop_across_repeats": float(np.std(drops, ddof=1)),
            "min_AUROC_drop": float(np.min(drops)),
            "max_AUROC_drop": float(np.max(drops)),
            "n_repeats": N_REPEATS,
        })
    print(f"Fold {fold_number}/{N_FOLDS} completed: baseline AUROC {baseline_auroc:.6f}")

detail_df = pd.DataFrame(detail_rows)
if len(detail_df) != len(feature_columns) * N_FOLDS:
    raise RuntimeError("Unexpected number of feature-fold detail rows.")

summary_df = (
    detail_df.groupby("feature", sort=False)
    .agg(
        mean_cv_AUROC_drop=("mean_AUROC_drop", "mean"),
        sd_cv_AUROC_drop=("mean_AUROC_drop", lambda values: values.std(ddof=1)),
        min_fold_AUROC_drop=("mean_AUROC_drop", "min"),
        max_fold_AUROC_drop=("mean_AUROC_drop", "max"),
        mean_permutation_repeat_sd=("sd_AUROC_drop_across_repeats", "mean"),
        folds=("fold", "nunique"),
        permutation_repeats_per_fold=("n_repeats", "first"),
    )
    .reset_index()
    .sort_values(["mean_cv_AUROC_drop", "feature"], ascending=[False, True])
    .reset_index(drop=True)
)
summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(SUMMARY_OUTPUT, index=False)
detail_df.sort_values(["fold", "mean_AUROC_drop"], ascending=[True, False]).to_csv(
    DETAIL_OUTPUT, index=False
)

mean_baseline_auroc = float(pd.DataFrame(baseline_rows)["baseline_validation_AUROC"].mean())
positive_count = int((summary_df["mean_cv_AUROC_drop"] > 0).sum())
negative_count = int((summary_df["mean_cv_AUROC_drop"] < 0).sum())
zero_count = int((summary_df["mean_cv_AUROC_drop"] == 0).sum())

print("\n38P. Top 15 features:")
print(summary_df.head(15).to_string(index=False))
print("\n38P. Bottom five features:")
print(summary_df.tail(5).to_string(index=False))
print("\n38Q. Output checks:")
print("Summary CSV exists:", SUMMARY_OUTPUT.exists())
print("Fold-detail CSV exists:", DETAIL_OUTPUT.exists())

print("\n" + "=" * 90)
print("STEP 38 SUMMARY")
print("=" * 90)
print("Development peptides:", len(development_indices))
print("Independent test peptides used:", 0)
print("CV folds:", N_FOLDS)
print("Features tested:", len(feature_columns))
print("Permutation repeats:", N_REPEATS)
print("Total permutations:", len(feature_columns) * N_REPEATS * N_FOLDS)
print("Total feature-fold combinations:", len(detail_df))
print("Mean validation AUROC:", round(mean_baseline_auroc, 6))
print("Positive mean importance features:", positive_count)
print("Negative mean importance features:", negative_count)
print("Exactly zero mean importance features:", zero_count)
print("\nSummary output:", SUMMARY_OUTPUT)
print("Fold-detail output:", DETAIL_OUTPUT)
print("\nSTEP 38 COMPLETED SUCCESSFULLY")
print("=" * 90)

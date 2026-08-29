from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
EMBEDDING_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
TRADITIONAL_INPUT = PROJECT_DIR / "derived" / "traditional_features.csv"
CV_INDEX_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
CV_RESULTS_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_cv_results.csv"
CV_FOLD_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_cv_fold_results.csv"
TEST_METRICS_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_test_metrics.csv"
TEST_PRED_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_test_predictions.csv"
MODEL_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_model.joblib"
PARAM_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_best_params.json"
IMPORTANCE_OUTPUT = PROJECT_DIR / "results" / "step50_esm2_random_forest_feature_importance.csv"

SEED = 2026
N_ESTIMATORS = (300, 600)
MAX_DEPTHS = (None, 5, 10)
MIN_SAMPLES_LEAFS = (1, 3, 5)
DECISION_THRESHOLD = 0.5


def make_model(n_estimators, max_depth, min_samples_leaf):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )


print("=" * 98)
print("STEP 50 - ESM-2 RANDOM FOREST")
print("=" * 98)

X = np.load(EMBEDDING_INPUT, allow_pickle=False)
metadata = pd.read_csv(METADATA_INPUT)
traditional = pd.read_csv(TRADITIONAL_INPUT)
cv_indices = np.load(CV_INDEX_INPUT, allow_pickle=False)
y = metadata["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
feature_names = [f"esm2_{number:04d}" for number in range(1, X.shape[1] + 1)]

assert X.shape == (901, 1280) and X.dtype == np.float32 and np.isfinite(X).all()
assert len(metadata) == len(traditional) == len(X)
assert np.array_equal(metadata["embedding_row"], np.arange(len(X)))
assert np.array_equal(metadata["ID"], traditional["ID"])
assert np.array_equal(metadata["sequence"], traditional["sequence"])
assert np.array_equal(y, traditional["label"].to_numpy(dtype=int))
assert np.array_equal(cv_indices["development_global_indices"], development_indices)
assert len(development_indices) == 720 and len(test_indices) == 181
assert np.intersect1d(development_indices, test_indices).size == 0
assert feature_names[0] == "esm2_0001" and feature_names[-1] == "esm2_1280"

print("Data:", X.shape, X.dtype)
print("Representation: raw ESM-2; no scaling; no PCA")
print("Development active/inactive:", int(y[development_indices].sum()), "/", int((y[development_indices] == 0).sum()))
print("Test active/inactive:", int(y[test_indices].sum()), "/", int((y[test_indices] == 0).sum()))
print("scikit-learn:", sklearn.__version__)

fold_rows = []
configuration_order = 0
for n_estimators in N_ESTIMATORS:
    for max_depth in MAX_DEPTHS:
        for min_samples_leaf in MIN_SAMPLES_LEAFS:
            print(f"CV: trees={n_estimators}, depth={max_depth}, leaf={min_samples_leaf}")
            for fold_number in range(1, 6):
                train_indices = cv_indices[f"fold{fold_number}_train"]
                valid_indices = cv_indices[f"fold{fold_number}_valid"]
                assert len(train_indices) == 576 and len(valid_indices) == 144
                assert np.intersect1d(train_indices, valid_indices).size == 0
                assert np.intersect1d(train_indices, test_indices).size == 0
                assert np.intersect1d(valid_indices, test_indices).size == 0
                model = make_model(n_estimators, max_depth, min_samples_leaf)
                model.fit(X[train_indices], y[train_indices])
                assert model.n_features_in_ == 1280
                probability = model.predict_proba(X[valid_indices])[:, 1]
                prediction = (probability >= DECISION_THRESHOLD).astype(int)
                assert np.isfinite(probability).all()
                fold_rows.append({
                    "configuration_order": configuration_order,
                    "n_estimators": n_estimators,
                    "max_depth": "None" if max_depth is None else str(max_depth),
                    "min_samples_leaf": min_samples_leaf,
                    "fold": fold_number, "train_n": 576, "validation_n": 144,
                    "input_dimensions": model.n_features_in_,
                    "scaler_used": False, "pca_used": False,
                    "validation_AUROC": float(roc_auc_score(y[valid_indices], probability)),
                    "validation_AUPRC": float(average_precision_score(y[valid_indices], probability)),
                    "validation_MCC": float(matthews_corrcoef(y[valid_indices], prediction)),
                    "validation_F1": float(f1_score(y[valid_indices], prediction, zero_division=0)),
                    "locked_test_rows_used": 0,
                })
            configuration_order += 1

fold_df = pd.DataFrame(fold_rows)
assert fold_df.shape[0] == 90 and fold_df["locked_test_rows_used"].eq(0).all()
assert fold_df["input_dimensions"].eq(1280).all()
assert (~fold_df["scaler_used"]).all() and (~fold_df["pca_used"]).all()

groups = ["configuration_order", "n_estimators", "max_depth", "min_samples_leaf"]
cv_results = (
    fold_df.groupby(groups, sort=False).agg(
        mean_cv_AUROC=("validation_AUROC", "mean"), sd_cv_AUROC=("validation_AUROC", "std"),
        mean_cv_AUPRC=("validation_AUPRC", "mean"), sd_cv_AUPRC=("validation_AUPRC", "std"),
        mean_cv_MCC=("validation_MCC", "mean"), sd_cv_MCC=("validation_MCC", "std"),
        mean_cv_F1=("validation_F1", "mean"), sd_cv_F1=("validation_F1", "std"),
    ).reset_index().sort_values(
        ["mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC", "configuration_order"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
)
assert cv_results.shape[0] == 18
best = cv_results.iloc[0]
best_n_estimators = int(best["n_estimators"])
best_max_depth_text = str(best["max_depth"])
best_max_depth = None if best_max_depth_text == "None" else int(best_max_depth_text)
best_min_samples_leaf = int(best["min_samples_leaf"])
best_cv_auroc = float(best["mean_cv_AUROC"])
cv_results["selected"] = (
    cv_results["n_estimators"].eq(best_n_estimators)
    & cv_results["max_depth"].eq(best_max_depth_text)
    & cv_results["min_samples_leaf"].eq(best_min_samples_leaf)
)
assert cv_results["selected"].sum() == 1

# Freeze winner, fit all development rows, then evaluate the locked test once.
final_model = make_model(best_n_estimators, best_max_depth, best_min_samples_leaf)
final_model.fit(X[development_indices], y[development_indices])
assert final_model.n_features_in_ == 1280
test_probability = final_model.predict_proba(X[test_indices])[:, 1]
test_prediction = (test_probability >= DECISION_THRESHOLD).astype(int)
test_labels = y[test_indices]
assert np.isfinite(test_probability).all()
tn, fp, fn, tp = confusion_matrix(test_labels, test_prediction, labels=[0, 1]).ravel()
test_metrics = {
    "model": "ESM2_RandomForest", "representation": "raw_esm2",
    "best_n_estimators": best_n_estimators, "best_max_depth": best_max_depth_text,
    "best_min_samples_leaf": best_min_samples_leaf,
    "selection_metric": "mean_cv_AUROC with AUPRC and MCC tie-breakers",
    "best_mean_cv_AUROC": best_cv_auroc, "best_sd_cv_AUROC": float(best["sd_cv_AUROC"]),
    "decision_threshold": DECISION_THRESHOLD, "development_n": 720,
    "test_n": 181, "test_active": int(test_labels.sum()), "test_inactive": int((test_labels == 0).sum()),
    "test_AUROC": float(roc_auc_score(test_labels, test_probability)),
    "test_AUPRC": float(average_precision_score(test_labels, test_probability)),
    "test_MCC": float(matthews_corrcoef(test_labels, test_prediction)),
    "test_accuracy": float(accuracy_score(test_labels, test_prediction)),
    "test_precision": float(precision_score(test_labels, test_prediction, zero_division=0)),
    "test_recall": float(recall_score(test_labels, test_prediction, zero_division=0)),
    "test_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    "test_F1": float(f1_score(test_labels, test_prediction, zero_division=0)),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
}

predictions = traditional.iloc[test_indices][
    ["ID", "sequence", "class", "original_class", "label", "binary_class"]
].copy()
predictions["predicted_probability"] = test_probability
predictions["predicted_label"] = test_prediction
predictions["split"] = "test"
importance = pd.DataFrame({
    "feature": feature_names, "rf_impurity_importance": final_model.feature_importances_,
}).sort_values(["rf_impurity_importance", "feature"], ascending=[False, True]).reset_index(drop=True)
importance.insert(0, "rank", np.arange(1, len(importance) + 1))
assert len(importance) == 1280 and np.isclose(importance["rf_impurity_importance"].sum(), 1.0)

CV_RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv_results.drop(columns="configuration_order").to_csv(CV_RESULTS_OUTPUT, index=False)
fold_df.drop(columns="configuration_order").to_csv(CV_FOLD_OUTPUT, index=False)
pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_OUTPUT, index=False)
predictions.to_csv(TEST_PRED_OUTPUT, index=False)
importance.to_csv(IMPORTANCE_OUTPUT, index=False)
joblib.dump(final_model, MODEL_OUTPUT)
with PARAM_OUTPUT.open("w", encoding="utf-8") as handle:
    json.dump({
        "model": "ESM2_RandomForest", "representation": "raw_esm2",
        "scaler_used": False, "pca_used": False,
        "best_n_estimators": best_n_estimators, "best_max_depth": best_max_depth,
        "best_min_samples_leaf": best_min_samples_leaf,
        "n_estimators_tested": list(N_ESTIMATORS), "max_depth_tested": list(MAX_DEPTHS),
        "min_samples_leaf_tested": list(MIN_SAMPLES_LEAFS),
        "selection_metric": "mean_cv_AUROC with AUPRC and MCC tie-breakers",
        "class_weight": "balanced", "decision_threshold": DECISION_THRESHOLD,
        "random_state": SEED, "development_n": 720, "test_n": 181,
        "cv_configurations": 18, "cv_model_fits": 90,
    }, handle, indent=2)

for path in (CV_RESULTS_OUTPUT, CV_FOLD_OUTPUT, TEST_METRICS_OUTPUT, TEST_PRED_OUTPUT, MODEL_OUTPUT, PARAM_OUTPUT, IMPORTANCE_OUTPUT):
    assert path.exists() and path.stat().st_size > 0

print("\n" + "=" * 98)
print("STEP 50 SUMMARY")
print("=" * 98)
print("Configurations:", len(cv_results), "CV fits:", len(fold_df))
print("Selected n_estimators:", best_n_estimators)
print("Selected max_depth:", best_max_depth)
print("Selected min_samples_leaf:", best_min_samples_leaf)
print("Best mean CV AUROC:", round(best_cv_auroc, 6))
print("Test AUROC:", round(test_metrics["test_AUROC"], 6))
print("Test AUPRC:", round(test_metrics["test_AUPRC"], 6))
print("Test MCC:", round(test_metrics["test_MCC"], 6))
print("Test F1:", round(test_metrics["test_F1"], 6))
print("\nTop 10 configurations:")
print(cv_results.drop(columns="configuration_order").head(10).round(6).to_string(index=False))
print("\nTop 10 preliminary dimensions:")
print(importance.head(10).round(8).to_string(index=False))
print("\nCV results:\n", CV_RESULTS_OUTPUT)
print("\nCV fold results:\n", CV_FOLD_OUTPUT)
print("\nTest metrics:\n", TEST_METRICS_OUTPUT)
print("\nTest predictions:\n", TEST_PRED_OUTPUT)
print("\nSaved model:\n", MODEL_OUTPUT)
print("\nBest parameters:\n", PARAM_OUTPUT)
print("\nPreliminary importance:\n", IMPORTANCE_OUTPUT)
print("\nSTEP 50 COMPLETED SUCCESSFULLY")
print("=" * 98)

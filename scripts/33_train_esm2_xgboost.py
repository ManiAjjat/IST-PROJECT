from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
EMBEDDING_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
TRADITIONAL_INPUT = PROJECT_DIR / "derived" / "traditional_features.csv"
CV_INDEX_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
CV_RESULTS_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_cv_results.csv"
CV_FOLD_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_cv_fold_results.csv"
TEST_METRICS_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_test_metrics.csv"
TEST_PRED_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_test_predictions.csv"
MODEL_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_model.joblib"
PARAM_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_best_params.json"
IMPORTANCE_OUTPUT = PROJECT_DIR / "results" / "step51_esm2_xgboost_feature_importance.csv"

SEED = 2026
N_ESTIMATORS = (200, 500)
MAX_DEPTHS = (2, 3, 5)
LEARNING_RATES = (0.03, 0.10)
SUBSAMPLES = (0.8, 1.0)
COLSAMPLES = (0.6, 0.9)
DECISION_THRESHOLD = 0.5


def make_model(params, scale_pos_weight):
    return XGBClassifier(
        **params, scale_pos_weight=scale_pos_weight, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=-1,
    )


print("=" * 100)
print("STEP 51 - ESM-2 XGBOOST")
print("=" * 100)
X = np.load(EMBEDDING_INPUT, allow_pickle=False)
metadata = pd.read_csv(METADATA_INPUT)
traditional = pd.read_csv(TRADITIONAL_INPUT)
cv_indices = np.load(CV_INDEX_INPUT, allow_pickle=False)
y = metadata["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
feature_names = [f"esm2_{i:04d}" for i in range(1, X.shape[1] + 1)]

assert X.shape == (901, 1280) and X.dtype == np.float32 and np.isfinite(X).all()
assert len(metadata) == len(traditional) == len(X)
assert np.array_equal(metadata["embedding_row"], np.arange(len(X)))
assert np.array_equal(metadata["ID"], traditional["ID"])
assert np.array_equal(metadata["sequence"], traditional["sequence"])
assert np.array_equal(y, traditional["label"].to_numpy(dtype=int))
assert np.array_equal(cv_indices["development_global_indices"], development_indices)
assert len(development_indices) == 720 and len(test_indices) == 181
assert np.intersect1d(development_indices, test_indices).size == 0

print("Data:", X.shape, X.dtype)
print("Representation: raw ESM-2; no scaling; no PCA")
print("Development active/inactive:", int(y[development_indices].sum()), "/", int((y[development_indices] == 0).sum()))
print("Test active/inactive:", int(y[test_indices].sum()), "/", int((y[test_indices] == 0).sum()))
print("xgboost/scikit-learn:", xgboost.__version__, "/", sklearn.__version__)

fold_rows = []
configuration_order = 0
total_configurations = len(N_ESTIMATORS) * len(MAX_DEPTHS) * len(LEARNING_RATES) * len(SUBSAMPLES) * len(COLSAMPLES)
for n_estimators in N_ESTIMATORS:
    for max_depth in MAX_DEPTHS:
        for learning_rate in LEARNING_RATES:
            for subsample in SUBSAMPLES:
                for colsample_bytree in COLSAMPLES:
                    params = {
                        "n_estimators": n_estimators, "max_depth": max_depth,
                        "learning_rate": learning_rate, "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                    }
                    print(f"Configuration {configuration_order + 1}/{total_configurations}: {params}")
                    for fold_number in range(1, 6):
                        train_indices = cv_indices[f"fold{fold_number}_train"]
                        valid_indices = cv_indices[f"fold{fold_number}_valid"]
                        assert len(train_indices) == 576 and len(valid_indices) == 144
                        assert np.intersect1d(train_indices, valid_indices).size == 0
                        assert np.intersect1d(train_indices, test_indices).size == 0
                        assert np.intersect1d(valid_indices, test_indices).size == 0
                        train_labels = y[train_indices]
                        positives = int(train_labels.sum())
                        negatives = int((train_labels == 0).sum())
                        scale_pos_weight = negatives / positives
                        model = make_model(params, scale_pos_weight)
                        model.fit(X[train_indices], train_labels)
                        assert model.n_features_in_ == 1280
                        probability = model.predict_proba(X[valid_indices])[:, 1]
                        prediction = (probability >= DECISION_THRESHOLD).astype(int)
                        assert np.isfinite(probability).all()
                        fold_rows.append({
                            "configuration_order": configuration_order, **params,
                            "fold": fold_number, "train_n": 576, "validation_n": 144,
                            "train_active": positives, "train_inactive": negatives,
                            "scale_pos_weight": scale_pos_weight, "input_dimensions": 1280,
                            "scaler_used": False, "pca_used": False,
                            "validation_AUROC": float(roc_auc_score(y[valid_indices], probability)),
                            "validation_AUPRC": float(average_precision_score(y[valid_indices], probability)),
                            "validation_MCC": float(matthews_corrcoef(y[valid_indices], prediction)),
                            "validation_F1": float(f1_score(y[valid_indices], prediction, zero_division=0)),
                            "locked_test_rows_used": 0,
                        })
                    configuration_order += 1

fold_df = pd.DataFrame(fold_rows)
assert configuration_order == 48 and fold_df.shape[0] == 240
assert fold_df["input_dimensions"].eq(1280).all()
assert (~fold_df["scaler_used"]).all() and (~fold_df["pca_used"]).all()
assert fold_df["locked_test_rows_used"].eq(0).all()
assert np.allclose(fold_df["scale_pos_weight"], fold_df["train_inactive"] / fold_df["train_active"])

parameter_columns = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
cv_results = (
    fold_df.groupby(["configuration_order", *parameter_columns], sort=False).agg(
        mean_cv_AUROC=("validation_AUROC", "mean"), sd_cv_AUROC=("validation_AUROC", "std"),
        mean_cv_AUPRC=("validation_AUPRC", "mean"), sd_cv_AUPRC=("validation_AUPRC", "std"),
        mean_cv_MCC=("validation_MCC", "mean"), sd_cv_MCC=("validation_MCC", "std"),
        mean_cv_F1=("validation_F1", "mean"), sd_cv_F1=("validation_F1", "std"),
    ).reset_index().sort_values(
        ["mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC", "configuration_order"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
)
assert cv_results.shape[0] == 48
best = cv_results.iloc[0]
best_params = {
    "n_estimators": int(best["n_estimators"]), "max_depth": int(best["max_depth"]),
    "learning_rate": float(best["learning_rate"]), "subsample": float(best["subsample"]),
    "colsample_bytree": float(best["colsample_bytree"]),
}
best_cv_auroc = float(best["mean_cv_AUROC"])
cv_results["selected"] = cv_results[parameter_columns].astype(float).eq(pd.Series(best_params)).all(axis=1)
assert cv_results["selected"].sum() == 1

# Freeze winner; derive the final weight on development only; test once.
development_labels = y[development_indices]
final_scale_pos_weight = int((development_labels == 0).sum()) / int(development_labels.sum())
final_model = make_model(best_params, final_scale_pos_weight)
final_model.fit(X[development_indices], development_labels)
assert final_model.n_features_in_ == 1280
test_probability = final_model.predict_proba(X[test_indices])[:, 1]
test_prediction = (test_probability >= DECISION_THRESHOLD).astype(int)
test_labels = y[test_indices]
assert np.isfinite(test_probability).all()
tn, fp, fn, tp = confusion_matrix(test_labels, test_prediction, labels=[0, 1]).ravel()
test_metrics = {
    "model": "ESM2_XGBoost", "representation": "raw_esm2", **{f"best_{k}": v for k, v in best_params.items()},
    "final_scale_pos_weight": final_scale_pos_weight,
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
    "feature": feature_names, "xgboost_gain_importance": final_model.feature_importances_,
}).sort_values(["xgboost_gain_importance", "feature"], ascending=[False, True]).reset_index(drop=True)
importance.insert(0, "rank", np.arange(1, len(importance) + 1))
assert len(importance) == 1280 and np.isclose(importance["xgboost_gain_importance"].sum(), 1.0)

CV_RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv_results.drop(columns="configuration_order").to_csv(CV_RESULTS_OUTPUT, index=False)
fold_df.drop(columns="configuration_order").to_csv(CV_FOLD_OUTPUT, index=False)
pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_OUTPUT, index=False)
predictions.to_csv(TEST_PRED_OUTPUT, index=False)
importance.to_csv(IMPORTANCE_OUTPUT, index=False)
joblib.dump(final_model, MODEL_OUTPUT)
with PARAM_OUTPUT.open("w", encoding="utf-8") as handle:
    json.dump({
        "model": "ESM2_XGBoost", "representation": "raw_esm2",
        "scaler_used": False, "pca_used": False, **best_params,
        "n_estimators_tested": list(N_ESTIMATORS), "max_depth_tested": list(MAX_DEPTHS),
        "learning_rate_tested": list(LEARNING_RATES), "subsample_tested": list(SUBSAMPLES),
        "colsample_bytree_tested": list(COLSAMPLES),
        "selection_metric": "mean_cv_AUROC with AUPRC and MCC tie-breakers",
        "tree_method": "hist", "final_scale_pos_weight": final_scale_pos_weight,
        "random_state": SEED, "development_n": 720, "test_n": 181,
        "cv_configurations": 48, "cv_model_fits": 240,
    }, handle, indent=2)

for path in (CV_RESULTS_OUTPUT, CV_FOLD_OUTPUT, TEST_METRICS_OUTPUT, TEST_PRED_OUTPUT, MODEL_OUTPUT, PARAM_OUTPUT, IMPORTANCE_OUTPUT):
    assert path.exists() and path.stat().st_size > 0

print("\n" + "=" * 100)
print("STEP 51 SUMMARY")
print("=" * 100)
print("Configurations:", len(cv_results), "CV fits:", len(fold_df))
print("Selected n_estimators:", best_params["n_estimators"])
print("Selected max_depth:", best_params["max_depth"])
print("Selected learning_rate:", best_params["learning_rate"])
print("Selected subsample:", best_params["subsample"])
print("Selected colsample_bytree:", best_params["colsample_bytree"])
print("Final scale_pos_weight:", round(final_scale_pos_weight, 6))
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
print("\nSTEP 51 COMPLETED SUCCESSFULLY")
print("=" * 100)

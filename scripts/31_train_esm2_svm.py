from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
EMBEDDING_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
TRADITIONAL_INPUT = PROJECT_DIR / "derived" / "traditional_features.csv"
CV_INDEX_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
CV_RESULTS_OUTPUT = PROJECT_DIR / "results" / "step49_esm2_svm_cv_results.csv"
CV_FOLD_OUTPUT = PROJECT_DIR / "results" / "step49_esm2_svm_cv_fold_results.csv"
TEST_METRICS_OUTPUT = PROJECT_DIR / "results" / "step49_esm2_svm_test_metrics.csv"
TEST_PRED_OUTPUT = PROJECT_DIR / "results" / "step49_esm2_svm_test_predictions.csv"
MODEL_OUTPUT = PROJECT_DIR / "results" / "step49_esm2_svm_model.joblib"
PARAM_OUTPUT = PROJECT_DIR / "results" / "step49_esm2_svm_best_params.json"

SEED = 2026
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = ("scale", 0.001, 0.01, 0.1)
REPRESENTATIONS = (
    ("no_pca", None), ("pca_24", 24), ("pca_52", 52),
    ("pca_99", 99), ("pca_274", 274),
)
DECISION_THRESHOLD = 0.5


def make_pipeline(c_value, gamma_value, pca_components):
    steps = [("scaler", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
    steps.append(
        ("model", SVC(
            C=c_value, gamma=gamma_value, kernel="rbf", class_weight="balanced",
            probability=True, random_state=SEED,
        ))
    )
    return Pipeline(steps)


print("=" * 98)
print("STEP 49 - ESM-2 RBF-SVM")
print("=" * 98)

X = np.load(EMBEDDING_INPUT, allow_pickle=False)
metadata = pd.read_csv(METADATA_INPUT)
traditional = pd.read_csv(TRADITIONAL_INPUT)
cv_indices = np.load(CV_INDEX_INPUT, allow_pickle=False)
y = metadata["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())

assert X.shape == (901, 1280) and X.dtype == np.float32 and np.isfinite(X).all()
assert len(metadata) == len(traditional) == len(X)
assert np.array_equal(metadata["embedding_row"], np.arange(len(X)))
assert np.array_equal(metadata["ID"], traditional["ID"])
assert np.array_equal(metadata["sequence"], traditional["sequence"])
assert np.array_equal(y, traditional["label"].to_numpy(dtype=int))
assert np.array_equal(cv_indices["development_global_indices"], development_indices)
assert len(development_indices) == 720 and len(test_indices) == 181
assert np.intersect1d(development_indices, test_indices).size == 0

print("\nData:", X.shape, X.dtype)
print("Development active/inactive:", int(y[development_indices].sum()), "/", int((y[development_indices] == 0).sum()))
print("Test active/inactive:", int(y[test_indices].sum()), "/", int((y[test_indices] == 0).sum()))
print("scikit-learn:", sklearn.__version__)

fold_rows = []
warning_count = 0
for representation_order, (representation, pca_components) in enumerate(REPRESENTATIONS):
    for c_value in C_VALUES:
        for gamma_order, gamma_value in enumerate(GAMMA_VALUES):
            print(f"CV: {representation}, C={c_value:g}, gamma={gamma_value}")
            for fold_number in range(1, 6):
                train_indices = cv_indices[f"fold{fold_number}_train"]
                valid_indices = cv_indices[f"fold{fold_number}_valid"]
                assert len(train_indices) == 576 and len(valid_indices) == 144
                assert np.intersect1d(train_indices, valid_indices).size == 0
                assert np.intersect1d(train_indices, test_indices).size == 0
                assert np.intersect1d(valid_indices, test_indices).size == 0

                pipeline = make_pipeline(c_value, gamma_value, pca_components)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(X[train_indices], y[train_indices])
                fit_warnings = len(caught)
                warning_count += fit_warnings
                scaler_samples = int(np.asarray(pipeline.named_steps["scaler"].n_samples_seen_).max())
                assert scaler_samples == 576
                if pca_components is None:
                    pca_samples, output_dimensions = np.nan, X.shape[1]
                else:
                    pca_samples = int(pipeline.named_steps["pca"].n_samples_)
                    output_dimensions = int(pipeline.named_steps["pca"].n_components_)
                    assert pca_samples == 576 and output_dimensions == pca_components

                probability = pipeline.predict_proba(X[valid_indices])[:, 1]
                prediction = (probability >= DECISION_THRESHOLD).astype(int)
                assert np.isfinite(probability).all()
                fold_rows.append({
                    "representation": representation,
                    "representation_order": representation_order,
                    "pca_components": pca_components,
                    "C": c_value,
                    "gamma": str(gamma_value),
                    "gamma_order": gamma_order,
                    "fold": fold_number,
                    "train_n": len(train_indices),
                    "validation_n": len(valid_indices),
                    "scaler_fit_samples": scaler_samples,
                    "pca_fit_samples": pca_samples,
                    "output_dimensions": output_dimensions,
                    "validation_AUROC": float(roc_auc_score(y[valid_indices], probability)),
                    "validation_AUPRC": float(average_precision_score(y[valid_indices], probability)),
                    "validation_MCC": float(matthews_corrcoef(y[valid_indices], prediction)),
                    "validation_F1": float(f1_score(y[valid_indices], prediction, zero_division=0)),
                    "fit_warnings": fit_warnings,
                    "locked_test_rows_used": 0,
                })

fold_df = pd.DataFrame(fold_rows)
assert fold_df.shape[0] == 400
assert fold_df["scaler_fit_samples"].eq(576).all()
assert fold_df.loc[fold_df["pca_components"].notna(), "pca_fit_samples"].eq(576).all()
assert fold_df["locked_test_rows_used"].eq(0).all()

group_columns = ["representation", "representation_order", "pca_components", "C", "gamma", "gamma_order"]
cv_results = (
    fold_df.groupby(group_columns, dropna=False, sort=False)
    .agg(
        mean_cv_AUROC=("validation_AUROC", "mean"), sd_cv_AUROC=("validation_AUROC", "std"),
        mean_cv_AUPRC=("validation_AUPRC", "mean"), sd_cv_AUPRC=("validation_AUPRC", "std"),
        mean_cv_MCC=("validation_MCC", "mean"), sd_cv_MCC=("validation_MCC", "std"),
        mean_cv_F1=("validation_F1", "mean"), sd_cv_F1=("validation_F1", "std"),
        fit_warnings=("fit_warnings", "sum"),
    ).reset_index()
    .sort_values(
        ["mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC", "representation_order", "C", "gamma_order"],
        ascending=[False, False, False, True, True, True], kind="stable",
    ).reset_index(drop=True)
)
assert cv_results.shape[0] == 80
best_representation = str(cv_results.loc[0, "representation"])
best_pca_raw = cv_results.loc[0, "pca_components"]
best_pca_components = None if pd.isna(best_pca_raw) else int(best_pca_raw)
best_C = float(cv_results.loc[0, "C"])
best_gamma_text = str(cv_results.loc[0, "gamma"])
best_gamma = "scale" if best_gamma_text == "scale" else float(best_gamma_text)
best_cv_auroc = float(cv_results.loc[0, "mean_cv_AUROC"])
cv_results["selected"] = (
    cv_results["representation"].eq(best_representation)
    & cv_results["C"].eq(best_C) & cv_results["gamma"].eq(best_gamma_text)
)
assert cv_results["selected"].sum() == 1

# Freeze the CV winner, refit once on development, evaluate the test set once.
final_pipeline = make_pipeline(best_C, best_gamma, best_pca_components)
with warnings.catch_warnings(record=True) as final_caught:
    warnings.simplefilter("always")
    final_pipeline.fit(X[development_indices], y[development_indices])
final_warning_count = len(final_caught)
assert int(np.asarray(final_pipeline.named_steps["scaler"].n_samples_seen_).max()) == 720
if best_pca_components is not None:
    assert int(final_pipeline.named_steps["pca"].n_samples_) == 720

test_probability = final_pipeline.predict_proba(X[test_indices])[:, 1]
test_prediction = (test_probability >= DECISION_THRESHOLD).astype(int)
test_labels = y[test_indices]
assert np.isfinite(test_probability).all()
tn, fp, fn, tp = confusion_matrix(test_labels, test_prediction, labels=[0, 1]).ravel()
test_metrics = {
    "model": "ESM2_RBF-SVM", "representation": best_representation,
    "pca_components": best_pca_components, "best_C": best_C,
    "best_gamma": best_gamma_text, "selection_metric": "mean_cv_AUROC with AUPRC and MCC tie-breakers",
    "best_mean_cv_AUROC": best_cv_auroc, "best_sd_cv_AUROC": float(cv_results.loc[0, "sd_cv_AUROC"]),
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
    "cv_fit_warnings": warning_count, "final_fit_warnings": final_warning_count,
}

predictions = traditional.iloc[test_indices][
    ["ID", "sequence", "class", "original_class", "label", "binary_class"]
].copy()
predictions["predicted_probability"] = test_probability
predictions["predicted_label"] = test_prediction
predictions["split"] = "test"

CV_RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv_results.drop(columns=["representation_order", "gamma_order"]).to_csv(CV_RESULTS_OUTPUT, index=False)
fold_df.drop(columns=["representation_order", "gamma_order"]).to_csv(CV_FOLD_OUTPUT, index=False)
pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_OUTPUT, index=False)
predictions.to_csv(TEST_PRED_OUTPUT, index=False)
joblib.dump(final_pipeline, MODEL_OUTPUT)
with PARAM_OUTPUT.open("w", encoding="utf-8") as handle:
    json.dump({
        "model": "ESM2_RBF-SVM", "selected_representation": best_representation,
        "selected_pca_components": best_pca_components, "best_C": best_C,
        "best_gamma": best_gamma, "representations_tested": [x[0] for x in REPRESENTATIONS],
        "pca_components_tested": [x[1] for x in REPRESENTATIONS],
        "C_values_tested": list(C_VALUES), "gamma_values_tested": list(GAMMA_VALUES),
        "selection_metric": "mean_cv_AUROC with AUPRC and MCC tie-breakers",
        "kernel": "rbf", "class_weight": "balanced", "probability": True,
        "decision_threshold": DECISION_THRESHOLD, "random_state": SEED,
        "development_n": 720, "test_n": 181, "cv_configurations": 80,
        "cv_model_fits": 400, "cv_fit_warnings": warning_count,
        "final_fit_warnings": final_warning_count,
    }, handle, indent=2)

for path in (CV_RESULTS_OUTPUT, CV_FOLD_OUTPUT, TEST_METRICS_OUTPUT, TEST_PRED_OUTPUT, MODEL_OUTPUT, PARAM_OUTPUT):
    assert path.exists() and path.stat().st_size > 0

print("\n" + "=" * 98)
print("STEP 49 SUMMARY")
print("=" * 98)
print("Representation candidates:", len(REPRESENTATIONS))
print("C values:", len(C_VALUES))
print("Gamma values:", len(GAMMA_VALUES))
print("Configurations:", len(cv_results))
print("CV fits:", len(fold_df))
print("Selected representation:", best_representation)
print("Selected PCA components:", best_pca_components)
print("Selected C:", best_C)
print("Selected gamma:", best_gamma)
print("Best mean CV AUROC:", round(best_cv_auroc, 6))
print("Test AUROC:", round(test_metrics["test_AUROC"], 6))
print("Test AUPRC:", round(test_metrics["test_AUPRC"], 6))
print("Test MCC:", round(test_metrics["test_MCC"], 6))
print("Test F1:", round(test_metrics["test_F1"], 6))
print("CV/final fit warnings:", warning_count, "/", final_warning_count)
print("\nTop 10 configurations:")
print(cv_results.drop(columns=["representation_order", "gamma_order"]).head(10).round(6).to_string(index=False))
print("\nCV results:\n", CV_RESULTS_OUTPUT)
print("\nFold results:\n", CV_FOLD_OUTPUT)
print("\nTest metrics:\n", TEST_METRICS_OUTPUT)
print("\nTest predictions:\n", TEST_PRED_OUTPUT)
print("\nSaved model:\n", MODEL_OUTPUT)
print("\nBest parameters:\n", PARAM_OUTPUT)
print("\nSTEP 49 COMPLETED SUCCESSFULLY")
print("=" * 98)

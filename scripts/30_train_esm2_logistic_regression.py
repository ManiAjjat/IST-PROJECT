from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
EMBEDDING_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
TRADITIONAL_INPUT = PROJECT_DIR / "derived" / "traditional_features.csv"
CV_INDEX_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"

CV_RESULTS_OUTPUT = PROJECT_DIR / "results" / "step48_esm2_logistic_regression_cv_results.csv"
CV_FOLD_OUTPUT = PROJECT_DIR / "results" / "step48_esm2_logistic_regression_cv_fold_results.csv"
TEST_METRICS_OUTPUT = PROJECT_DIR / "results" / "step48_esm2_logistic_regression_test_metrics.csv"
TEST_PRED_OUTPUT = PROJECT_DIR / "results" / "step48_esm2_logistic_regression_test_predictions.csv"
MODEL_OUTPUT = PROJECT_DIR / "results" / "step48_esm2_logistic_regression_model.joblib"
PARAM_OUTPUT = PROJECT_DIR / "results" / "step48_esm2_logistic_regression_best_params.json"

SEED = 2026
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
REPRESENTATIONS = (
    ("no_pca", None),
    ("pca_24", 24),
    ("pca_52", 52),
    ("pca_99", 99),
    ("pca_274", 274),
)
DECISION_THRESHOLD = 0.5


def make_pipeline(c_value, pca_components):
    steps = [("scaler", StandardScaler())]
    if pca_components is not None:
        steps.append(
            ("pca", PCA(n_components=pca_components, svd_solver="full"))
        )
    steps.append(
        (
            "model",
            LogisticRegression(
                C=c_value,
                penalty="l2",
                class_weight="balanced",
                random_state=SEED,
                max_iter=5000,
            ),
        )
    )
    return Pipeline(steps)


print("=" * 96)
print("STEP 48 - ESM-2 LOGISTIC REGRESSION")
print("=" * 96)

X = np.load(EMBEDDING_INPUT, allow_pickle=False)
metadata = pd.read_csv(METADATA_INPUT)
traditional = pd.read_csv(TRADITIONAL_INPUT)
cv_indices = np.load(CV_INDEX_INPUT, allow_pickle=False)
y = metadata["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())

assert X.shape == (901, 1280)
assert X.dtype == np.float32
assert np.isfinite(X).all()
assert len(metadata) == len(traditional) == len(X)
assert np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(len(X)))
assert np.array_equal(metadata["ID"].to_numpy(), traditional["ID"].to_numpy())
assert np.array_equal(metadata["sequence"].to_numpy(), traditional["sequence"].to_numpy())
assert np.array_equal(y, traditional["label"].to_numpy(dtype=int))
assert np.array_equal(cv_indices["development_global_indices"], development_indices)
assert len(development_indices) == 720
assert len(test_indices) == 181
assert np.intersect1d(development_indices, test_indices).size == 0

print("\nData:")
print("Embedding matrix:", X.shape, X.dtype)
print("Development:", len(development_indices))
print("Locked test:", len(test_indices))
print("Development active/inactive:", int(y[development_indices].sum()), "/", int((y[development_indices] == 0).sum()))
print("Test active/inactive:", int(y[test_indices].sum()), "/", int((y[test_indices] == 0).sum()))
print("scikit-learn:", sklearn.__version__)

fold_rows = []
convergence_warning_count = 0

for representation_order, (representation, pca_components) in enumerate(REPRESENTATIONS):
    for c_value in C_VALUES:
        print(f"\nCV: representation={representation}, C={c_value:g}")
        for fold_number in range(1, 6):
            train_indices = cv_indices[f"fold{fold_number}_train"]
            valid_indices = cv_indices[f"fold{fold_number}_valid"]
            assert len(train_indices) == 576
            assert len(valid_indices) == 144
            assert np.intersect1d(train_indices, valid_indices).size == 0
            assert np.intersect1d(train_indices, test_indices).size == 0
            assert np.intersect1d(valid_indices, test_indices).size == 0

            pipeline = make_pipeline(c_value, pca_components)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(X[train_indices], y[train_indices])
            fold_convergence_warnings = sum(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            convergence_warning_count += fold_convergence_warnings

            scaler_fit_samples = int(
                np.asarray(pipeline.named_steps["scaler"].n_samples_seen_).max()
            )
            assert scaler_fit_samples == 576
            if pca_components is None:
                pca_fit_samples = np.nan
                output_dimensions = X.shape[1]
            else:
                pca_fit_samples = int(pipeline.named_steps["pca"].n_samples_)
                output_dimensions = int(pipeline.named_steps["pca"].n_components_)
                assert pca_fit_samples == 576
                assert output_dimensions == pca_components

            valid_probability = pipeline.predict_proba(X[valid_indices])[:, 1]
            valid_prediction = (valid_probability >= DECISION_THRESHOLD).astype(int)
            assert np.isfinite(valid_probability).all()

            fold_rows.append(
                {
                    "representation": representation,
                    "representation_order": representation_order,
                    "pca_components": pca_components,
                    "C": c_value,
                    "fold": fold_number,
                    "train_n": len(train_indices),
                    "validation_n": len(valid_indices),
                    "scaler_fit_samples": scaler_fit_samples,
                    "pca_fit_samples": pca_fit_samples,
                    "output_dimensions": output_dimensions,
                    "validation_AUROC": float(
                        roc_auc_score(y[valid_indices], valid_probability)
                    ),
                    "validation_AUPRC": float(
                        average_precision_score(y[valid_indices], valid_probability)
                    ),
                    "validation_MCC": float(
                        matthews_corrcoef(y[valid_indices], valid_prediction)
                    ),
                    "validation_F1": float(
                        f1_score(y[valid_indices], valid_prediction, zero_division=0)
                    ),
                    "convergence_warnings": fold_convergence_warnings,
                    "locked_test_rows_used": 0,
                }
            )

fold_df = pd.DataFrame(fold_rows)
assert fold_df.shape[0] == len(REPRESENTATIONS) * len(C_VALUES) * 5
assert fold_df["scaler_fit_samples"].eq(576).all()
assert fold_df.loc[fold_df["pca_components"].notna(), "pca_fit_samples"].eq(576).all()
assert fold_df["locked_test_rows_used"].eq(0).all()

cv_results = (
    fold_df.groupby(
        ["representation", "representation_order", "pca_components", "C"],
        dropna=False,
        sort=False,
    )
    .agg(
        mean_cv_AUROC=("validation_AUROC", "mean"),
        sd_cv_AUROC=("validation_AUROC", "std"),
        mean_cv_AUPRC=("validation_AUPRC", "mean"),
        sd_cv_AUPRC=("validation_AUPRC", "std"),
        mean_cv_MCC=("validation_MCC", "mean"),
        sd_cv_MCC=("validation_MCC", "std"),
        mean_cv_F1=("validation_F1", "mean"),
        sd_cv_F1=("validation_F1", "std"),
        convergence_warnings=("convergence_warnings", "sum"),
    )
    .reset_index()
    .sort_values(
        ["mean_cv_AUROC", "representation_order", "C"],
        ascending=[False, True, True],
        kind="stable",
    )
    .reset_index(drop=True)
)

best_representation = str(cv_results.loc[0, "representation"])
best_pca_value = cv_results.loc[0, "pca_components"]
best_pca_components = None if pd.isna(best_pca_value) else int(best_pca_value)
best_C = float(cv_results.loc[0, "C"])
best_cv_auroc = float(cv_results.loc[0, "mean_cv_AUROC"])
cv_results["selected"] = (
    cv_results["representation"].eq(best_representation)
    & cv_results["C"].eq(best_C)
)
assert cv_results.shape[0] == 25
assert cv_results["selected"].sum() == 1

# The configuration is now frozen. Fit once on all development rows and
# evaluate once on the locked test set.
final_pipeline = make_pipeline(best_C, best_pca_components)
with warnings.catch_warnings(record=True) as final_caught:
    warnings.simplefilter("always", ConvergenceWarning)
    final_pipeline.fit(X[development_indices], y[development_indices])
final_convergence_warnings = sum(
    issubclass(item.category, ConvergenceWarning) for item in final_caught
)

assert int(np.asarray(final_pipeline.named_steps["scaler"].n_samples_seen_).max()) == 720
if best_pca_components is not None:
    assert int(final_pipeline.named_steps["pca"].n_samples_) == 720

test_probability = final_pipeline.predict_proba(X[test_indices])[:, 1]
test_prediction = (test_probability >= DECISION_THRESHOLD).astype(int)
test_labels = y[test_indices]
assert np.isfinite(test_probability).all()
tn, fp, fn, tp = confusion_matrix(test_labels, test_prediction, labels=[0, 1]).ravel()

test_metrics = {
    "model": "ESM2_LogisticRegression",
    "representation": best_representation,
    "pca_components": best_pca_components,
    "best_C": best_C,
    "selection_metric": "mean_cv_AUROC",
    "best_mean_cv_AUROC": best_cv_auroc,
    "best_sd_cv_AUROC": float(cv_results.loc[0, "sd_cv_AUROC"]),
    "decision_threshold": DECISION_THRESHOLD,
    "development_n": len(development_indices),
    "test_n": len(test_indices),
    "test_active": int(test_labels.sum()),
    "test_inactive": int((test_labels == 0).sum()),
    "test_AUROC": float(roc_auc_score(test_labels, test_probability)),
    "test_AUPRC": float(average_precision_score(test_labels, test_probability)),
    "test_MCC": float(matthews_corrcoef(test_labels, test_prediction)),
    "test_accuracy": float(accuracy_score(test_labels, test_prediction)),
    "test_precision": float(precision_score(test_labels, test_prediction, zero_division=0)),
    "test_recall": float(recall_score(test_labels, test_prediction, zero_division=0)),
    "test_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    "test_F1": float(f1_score(test_labels, test_prediction, zero_division=0)),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "cv_convergence_warnings": convergence_warning_count,
    "final_convergence_warnings": final_convergence_warnings,
}

predictions = traditional.iloc[test_indices][
    ["ID", "sequence", "class", "original_class", "label", "binary_class"]
].copy()
predictions["predicted_probability"] = test_probability
predictions["predicted_label"] = test_prediction
predictions["split"] = "test"

CV_RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv_results.drop(columns="representation_order").to_csv(CV_RESULTS_OUTPUT, index=False)
fold_df.drop(columns="representation_order").to_csv(CV_FOLD_OUTPUT, index=False)
pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_OUTPUT, index=False)
predictions.to_csv(TEST_PRED_OUTPUT, index=False)
joblib.dump(final_pipeline, MODEL_OUTPUT)

with PARAM_OUTPUT.open("w", encoding="utf-8") as parameter_file:
    json.dump(
        {
            "model": "ESM2_LogisticRegression",
            "selected_representation": best_representation,
            "selected_pca_components": best_pca_components,
            "best_C": best_C,
            "representations_tested": [name for name, _ in REPRESENTATIONS],
            "pca_components_tested": [value for _, value in REPRESENTATIONS],
            "C_values_tested": list(C_VALUES),
            "selection_metric": "mean_cv_AUROC",
            "class_weight": "balanced",
            "penalty": "l2",
            "decision_threshold": DECISION_THRESHOLD,
            "random_state": SEED,
            "development_n": len(development_indices),
            "test_n": len(test_indices),
            "cv_configurations": len(cv_results),
            "cv_model_fits": len(fold_df),
            "cv_convergence_warnings": convergence_warning_count,
            "final_convergence_warnings": final_convergence_warnings,
        },
        parameter_file,
        indent=2,
    )

for output in (
    CV_RESULTS_OUTPUT,
    CV_FOLD_OUTPUT,
    TEST_METRICS_OUTPUT,
    TEST_PRED_OUTPUT,
    MODEL_OUTPUT,
    PARAM_OUTPUT,
):
    assert output.exists() and output.stat().st_size > 0

print("\n" + "=" * 96)
print("STEP 48 SUMMARY")
print("=" * 96)
print("Representation candidates:", len(REPRESENTATIONS))
print("C values:", len(C_VALUES))
print("CV configurations:", len(cv_results))
print("CV fits:", len(fold_df))
print("Selected representation:", best_representation)
print("Selected PCA components:", best_pca_components)
print("Selected C:", best_C)
print("Best mean CV AUROC:", round(best_cv_auroc, 6))
print("Test AUROC:", round(test_metrics["test_AUROC"], 6))
print("Test AUPRC:", round(test_metrics["test_AUPRC"], 6))
print("Test MCC:", round(test_metrics["test_MCC"], 6))
print("Test F1:", round(test_metrics["test_F1"], 6))
print("CV convergence warnings:", convergence_warning_count)
print("Final convergence warnings:", final_convergence_warnings)
print("\nTop 10 configurations:")
print(cv_results.drop(columns="representation_order").head(10).round(6).to_string(index=False))
print("\nCV results:")
print(CV_RESULTS_OUTPUT)
print("\nFold details:")
print(CV_FOLD_OUTPUT)
print("\nTest metrics:")
print(TEST_METRICS_OUTPUT)
print("\nTest predictions:")
print(TEST_PRED_OUTPUT)
print("\nSaved model:")
print(MODEL_OUTPUT)
print("\nBest parameters:")
print(PARAM_OUTPUT)
print("\nSTEP 48 COMPLETED SUCCESSFULLY")
print("=" * 96)

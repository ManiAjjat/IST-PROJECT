from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
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
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
CV_INDEX_FILE = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
CV_RESULTS_OUTPUT = PROJECT_DIR / "results" / "step31_logistic_regression_cv_results.csv"
TEST_METRICS_OUTPUT = PROJECT_DIR / "results" / "step31_logistic_regression_test_metrics.csv"
TEST_PRED_OUTPUT = PROJECT_DIR / "results" / "step31_logistic_regression_test_predictions.csv"
MODEL_OUTPUT = PROJECT_DIR / "results" / "step31_logistic_regression_model.joblib"
PARAM_OUTPUT = PROJECT_DIR / "results" / "step31_logistic_regression_best_params.json"
SEED = 2026
C_VALUES = [0.01, 0.1, 1, 10, 100]

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
split = pd.read_csv(SPLIT_FILE)
cv_indices = np.load(CV_INDEX_FILE)
feature_columns = [column for column in df.columns if column not in metadata_columns]
X = df[feature_columns].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(split["split"].eq("development"))
test_indices = np.flatnonzero(split["split"].eq("test"))


def make_pipeline(c_value):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )


cv_rows = []
for c_value in C_VALUES:
    fold_aurocs = []
    fold_auprcs = []
    fold_mccs = []
    for fold_number in range(1, 6):
        train_indices = cv_indices[f"fold{fold_number}_train"]
        valid_indices = cv_indices[f"fold{fold_number}_valid"]
        pipeline = make_pipeline(c_value)
        pipeline.fit(X[train_indices], y[train_indices])
        valid_probability = pipeline.predict_proba(X[valid_indices])[:, 1]
        valid_prediction = (valid_probability >= 0.5).astype(int)
        fold_aurocs.append(roc_auc_score(y[valid_indices], valid_probability))
        fold_auprcs.append(average_precision_score(y[valid_indices], valid_probability))
        fold_mccs.append(matthews_corrcoef(y[valid_indices], valid_prediction))
    cv_rows.append(
        {
            "C": c_value,
            "mean_cv_AUROC": float(np.mean(fold_aurocs)),
            "sd_cv_AUROC": float(np.std(fold_aurocs, ddof=1)),
            "mean_cv_AUPRC": float(np.mean(fold_auprcs)),
            "mean_cv_MCC": float(np.mean(fold_mccs)),
        }
    )

cv_results = pd.DataFrame(cv_rows).sort_values(
    ["mean_cv_AUROC", "C"], ascending=[False, True]
).reset_index(drop=True)
best_c = float(cv_results.loc[0, "C"])
cv_results["selected"] = cv_results["C"].eq(best_c)
cv_results.to_csv(CV_RESULTS_OUTPUT, index=False)

final_pipeline = make_pipeline(best_c)
final_pipeline.fit(X[development_indices], y[development_indices])
test_probability = final_pipeline.predict_proba(X[test_indices])[:, 1]
test_prediction = (test_probability >= 0.5).astype(int)
test_labels = y[test_indices]
tn, fp, fn, tp = confusion_matrix(test_labels, test_prediction, labels=[0, 1]).ravel()
test_metrics = {
    "model": "LogisticRegression",
    "best_C": best_c,
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
}
pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_OUTPUT, index=False)

predictions = df.iloc[test_indices][
    ["ID", "sequence", "class", "original_class", "label", "binary_class"]
].copy()
predictions["predicted_probability"] = test_probability
predictions["predicted_label"] = test_prediction
predictions["split"] = "test"
predictions.to_csv(TEST_PRED_OUTPUT, index=False)

joblib.dump(final_pipeline, MODEL_OUTPUT)
with PARAM_OUTPUT.open("w", encoding="utf-8") as parameter_file:
    json.dump(
        {
            "model": "LogisticRegression",
            "best_C": best_c,
            "C_values_tested": C_VALUES,
            "selection_metric": "mean_cv_AUROC",
            "class_weight": "balanced",
            "random_state": SEED,
            "development_n": len(development_indices),
            "test_n": len(test_indices),
        },
        parameter_file,
        indent=2,
    )

print("=" * 82)
print("STEP 31 - TRADITIONAL LOGISTIC REGRESSION")
print("=" * 82)
print("C values tested:", C_VALUES)
print("Best C:", best_c)
print("Mean CV AUROC:", round(float(cv_results.loc[0, "mean_cv_AUROC"]), 6))
print("Independent test AUROC:", round(test_metrics["test_AUROC"], 6))
print("Independent test AUPRC:", round(test_metrics["test_AUPRC"], 6))
print("Independent test MCC:", round(test_metrics["test_MCC"], 6))
print("\nCV results:")
print(CV_RESULTS_OUTPUT)
print("\nTest metrics:")
print(TEST_METRICS_OUTPUT)
print("\nTest predictions:")
print(TEST_PRED_OUTPUT)
print("\nSaved model:")
print(MODEL_OUTPUT)
print("\nBest parameters:")
print(PARAM_OUTPUT)
print("\nSTEP 31 COMPLETED SUCCESSFULLY")
print("=" * 82)
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
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
from xgboost import XGBClassifier


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SPLIT_FILE = PROJECT_DIR / "derived" / "fixed_split.csv"
CV_INDEX_FILE = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
CV_RESULTS_OUTPUT = PROJECT_DIR / "results" / "step34_xgboost_cv_results.csv"
TEST_METRICS_OUTPUT = PROJECT_DIR / "results" / "step34_xgboost_test_metrics.csv"
TEST_PRED_OUTPUT = PROJECT_DIR / "results" / "step34_xgboost_test_predictions.csv"
MODEL_OUTPUT = PROJECT_DIR / "results" / "step34_xgboost_model.joblib"
PARAM_OUTPUT = PROJECT_DIR / "results" / "step34_xgboost_best_params.json"
IMPORTANCE_OUTPUT = PROJECT_DIR / "results" / "step34_xgboost_feature_importance.csv"
SEED = 2026
N_ESTIMATORS = [200, 500]
MAX_DEPTHS = [2, 3, 5]
LEARNING_RATES = [0.03, 0.10]
SUBSAMPLES = [0.8, 1.0]
COLSAMPLES = [0.6, 0.9]

metadata_columns = [
    "ID", "sequence", "class", "original_class", "label",
    "binary_class", "inactive_source", "is_virtual_inactive",
]
df = pd.read_csv(FEATURE_FILE)
split = pd.read_csv(SPLIT_FILE)
cv_indices = np.load(CV_INDEX_FILE)
feature_columns = [c for c in df.columns if c not in metadata_columns]
X = df[feature_columns].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(split["split"].eq("development"))
test_indices = np.flatnonzero(split["split"].eq("test"))


def make_model(params, scale_pos_weight):
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


cv_rows = []
combination_count = 0
total_combinations = (
    len(N_ESTIMATORS) * len(MAX_DEPTHS) * len(LEARNING_RATES)
    * len(SUBSAMPLES) * len(COLSAMPLES)
)
for n_estimators in N_ESTIMATORS:
    for max_depth in MAX_DEPTHS:
        for learning_rate in LEARNING_RATES:
            for subsample in SUBSAMPLES:
                for colsample_bytree in COLSAMPLES:
                    combination_count += 1
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                    }
                    fold_aurocs, fold_auprcs, fold_mccs = [], [], []
                    for fold_number in range(1, 6):
                        train_indices = cv_indices[f"fold{fold_number}_train"]
                        valid_indices = cv_indices[f"fold{fold_number}_valid"]
                        train_labels = y[train_indices]
                        positives = int(train_labels.sum())
                        negatives = int((train_labels == 0).sum())
                        scale_pos_weight = negatives / positives
                        model = make_model(params, scale_pos_weight)
                        model.fit(X[train_indices], train_labels)
                        probability = model.predict_proba(X[valid_indices])[:, 1]
                        prediction = (probability >= 0.5).astype(int)
                        fold_aurocs.append(roc_auc_score(y[valid_indices], probability))
                        fold_auprcs.append(average_precision_score(y[valid_indices], probability))
                        fold_mccs.append(matthews_corrcoef(y[valid_indices], prediction))
                    cv_rows.append({
                        **params,
                        "mean_cv_AUROC": float(np.mean(fold_aurocs)),
                        "sd_cv_AUROC": float(np.std(fold_aurocs, ddof=1)),
                        "mean_cv_AUPRC": float(np.mean(fold_auprcs)),
                        "mean_cv_MCC": float(np.mean(fold_mccs)),
                    })
                    print(f"Combination {combination_count}/{total_combinations} completed")

cv_results = pd.DataFrame(cv_rows).sort_values(
    ["mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC"],
    ascending=[False, False, False],
).reset_index(drop=True)
best = cv_results.iloc[0]
best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_depth": int(best["max_depth"]),
    "learning_rate": float(best["learning_rate"]),
    "subsample": float(best["subsample"]),
    "colsample_bytree": float(best["colsample_bytree"]),
}
cv_results["selected"] = (
    cv_results[list(best_params)].astype(float).eq(pd.Series(best_params)).all(axis=1)
)
cv_results.to_csv(CV_RESULTS_OUTPUT, index=False)

development_labels = y[development_indices]
final_scale_pos_weight = int((development_labels == 0).sum()) / int(development_labels.sum())
final_model = make_model(best_params, final_scale_pos_weight)
final_model.fit(X[development_indices], development_labels)
test_probability = final_model.predict_proba(X[test_indices])[:, 1]
test_prediction = (test_probability >= 0.5).astype(int)
test_labels = y[test_indices]
tn, fp, fn, tp = confusion_matrix(test_labels, test_prediction, labels=[0, 1]).ravel()
test_metrics = {
    "model": "XGBoost",
    **{f"best_{key}": value for key, value in best_params.items()},
    "final_scale_pos_weight": final_scale_pos_weight,
    "test_n": len(test_indices), "test_active": int(test_labels.sum()),
    "test_inactive": int((test_labels == 0).sum()),
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
pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_OUTPUT, index=False)

predictions = df.iloc[test_indices][
    ["ID", "sequence", "class", "original_class", "label", "binary_class"]
].copy()
predictions["predicted_probability"] = test_probability
predictions["predicted_label"] = test_prediction
predictions["split"] = "test"
predictions.to_csv(TEST_PRED_OUTPUT, index=False)
joblib.dump(final_model, MODEL_OUTPUT)

importance = pd.DataFrame({
    "feature": feature_columns,
    "xgboost_gain_importance": final_model.feature_importances_,
}).sort_values("xgboost_gain_importance", ascending=False)
importance.to_csv(IMPORTANCE_OUTPUT, index=False)

with PARAM_OUTPUT.open("w", encoding="utf-8") as parameter_file:
    json.dump({
        "model": "XGBoost",
        **best_params,
        "n_estimators_tested": N_ESTIMATORS,
        "max_depth_tested": MAX_DEPTHS,
        "learning_rate_tested": LEARNING_RATES,
        "subsample_tested": SUBSAMPLES,
        "colsample_bytree_tested": COLSAMPLES,
        "selection_metric": "mean_cv_AUROC with AUPRC and MCC tie-breakers",
        "tree_method": "hist", "scale_pos_weight": final_scale_pos_weight,
        "random_state": SEED,
    }, parameter_file, indent=2)

print("=" * 86)
print("STEP 34 - TRADITIONAL XGBOOST")
print("=" * 86)
print("Combinations tested:", total_combinations)
print("Best parameters:", best_params)
print("Mean CV AUROC:", round(float(best["mean_cv_AUROC"]), 6))
print("Independent test AUROC:", round(test_metrics["test_AUROC"], 6))
print("Independent test AUPRC:", round(test_metrics["test_AUPRC"], 6))
print("Independent test MCC:", round(test_metrics["test_MCC"], 6))
print("\nCV results:", CV_RESULTS_OUTPUT)
print("Test metrics:", TEST_METRICS_OUTPUT)
print("Test predictions:", TEST_PRED_OUTPUT)
print("Saved model:", MODEL_OUTPUT)
print("Best parameters:", PARAM_OUTPUT)
print("Feature importance:", IMPORTANCE_OUTPUT)
print("\nSTEP 34 COMPLETED SUCCESSFULLY")
print("=" * 86)
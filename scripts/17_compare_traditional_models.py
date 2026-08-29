from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
METRIC_FILES = {
    "Logistic Regression": RESULTS_DIR / "step31_logistic_regression_test_metrics.csv",
    "RBF-SVM": RESULTS_DIR / "step32_svm_test_metrics.csv",
    "Random Forest": RESULTS_DIR / "step33_random_forest_test_metrics.csv",
    "XGBoost": RESULTS_DIR / "step34_xgboost_test_metrics.csv",
}
OUTPUT_FILE = RESULTS_DIR / "step35_traditional_model_comparison.csv"
MANUSCRIPT_OUTPUT = RESULTS_DIR / "step35_traditional_model_comparison_manuscript.csv"

rows = []
for model_name, metric_file in METRIC_FILES.items():
    metrics = pd.read_csv(metric_file)
    if len(metrics) != 1:
        raise ValueError(f"Expected one metrics row in {metric_file}")
    row = metrics.iloc[0].to_dict()
    row["model"] = model_name
    rows.append(row)

comparison = pd.DataFrame(rows)
test_columns = ["test_n", "test_active", "test_inactive"]
if comparison[test_columns].nunique().max() != 1:
    raise ValueError("Models were not evaluated on the same test composition.")
if comparison["test_n"].iloc[0] != 181:
    raise ValueError("Expected 181 independent test peptides.")

metric_map = {
    "test_AUROC": "AUROC",
    "test_AUPRC": "AUPRC",
    "test_MCC": "MCC",
    "test_accuracy": "Accuracy",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_specificity": "Specificity",
    "test_F1": "F1",
}
for source, target in metric_map.items():
    comparison[target] = comparison[source]

ranking_metrics = ["AUROC", "AUPRC", "MCC", "F1"]
for metric in ranking_metrics:
    comparison[f"{metric}_rank"] = comparison[metric].rank(
        ascending=False, method="min"
    )
comparison["overall_rank"] = comparison[
    [f"{metric}_rank" for metric in ranking_metrics]
].mean(axis=1)
comparison = comparison.sort_values(
    ["overall_rank", "AUROC", "MCC"], ascending=[True, False, False]
).reset_index(drop=True)

full_columns = [
    "model",
    "AUROC", "AUPRC", "MCC", "Accuracy", "Precision", "Recall",
    "Specificity", "F1", "TN", "FP", "FN", "TP",
    "AUROC_rank", "AUPRC_rank", "MCC_rank", "F1_rank", "overall_rank",
]
comparison[full_columns].round(6).to_csv(OUTPUT_FILE, index=False)

manuscript_table = comparison[
    [
        "model", "AUROC", "AUPRC", "MCC", "Accuracy", "Precision",
        "Recall", "Specificity", "F1",
    ]
].copy()
manuscript_table.round(6).to_csv(MANUSCRIPT_OUTPUT, index=False)

print("\n" + "=" * 84)
print("STEP 35 SUMMARY")
print("=" * 84)
print("Models compared:", len(comparison))
print("Independent test peptides:", int(comparison["test_n"].iloc[0]))
print("Active test peptides:", int(comparison["test_active"].iloc[0]))
print("Inactive test peptides:", int(comparison["test_inactive"].iloc[0]))
print("\nComparison:")
print(comparison[full_columns].round(6).to_string(index=False))
print("\nMain comparison table:")
print(OUTPUT_FILE)
print("\nManuscript-style table:")
print(MANUSCRIPT_OUTPUT)
print("\nSTEP 35 COMPLETED SUCCESSFULLY")
print("=" * 84)
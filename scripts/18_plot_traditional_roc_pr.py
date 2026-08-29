from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
ROC_PNG = FIGURE_DIR / "Step36_Traditional_Model_ROC.png"
ROC_PDF = FIGURE_DIR / "Step36_Traditional_Model_ROC.pdf"
PR_PNG = FIGURE_DIR / "Step36_Traditional_Model_PR.png"
PR_PDF = FIGURE_DIR / "Step36_Traditional_Model_PR.pdf"
CURVE_SUMMARY = RESULTS_DIR / "step36_traditional_model_curve_summary.csv"

prediction_files = {
    "Logistic Regression": RESULTS_DIR / "step31_logistic_regression_test_predictions.csv",
    "RBF-SVM": RESULTS_DIR / "step32_svm_test_predictions.csv",
    "Random Forest": RESULTS_DIR / "step33_random_forest_test_predictions.csv",
    "XGBoost": RESULTS_DIR / "step34_xgboost_test_predictions.csv",
}
colors = {
    "Logistic Regression": "#4C78A8",
    "RBF-SVM": "#F58518",
    "Random Forest": "#54A24B",
    "XGBoost": "#E45756",
}

prediction_tables = {}
reference_y = None
for model_name, prediction_file in prediction_files.items():
    table = pd.read_csv(prediction_file)
    required = {"ID", "label", "predicted_probability"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing prediction columns in {prediction_file}")
    table = table.sort_values("ID").reset_index(drop=True)
    current_y = table["label"].to_numpy()
    if reference_y is None:
        reference_y = current_y
        reference_ids = table["ID"].to_numpy()
    elif not (table["ID"].to_numpy() == reference_ids).all() or not (current_y == reference_y).all():
        raise ValueError("Prediction files do not use the same test peptides and labels.")
    prediction_tables[model_name] = table

active_n = int(reference_y.sum())
inactive_n = int((reference_y == 0).sum())
summary_rows = []

roc_figure, roc_axis = plt.subplots(figsize=(8.5, 7))
pr_figure, pr_axis = plt.subplots(figsize=(8.5, 7))
for model_name, table in prediction_tables.items():
    probability = table["predicted_probability"].to_numpy()
    false_positive_rate, true_positive_rate, _ = roc_curve(reference_y, probability)
    precision, recall, _ = precision_recall_curve(reference_y, probability)
    auroc = roc_auc_score(reference_y, probability)
    auprc = average_precision_score(reference_y, probability)
    summary_rows.append({"model": model_name, "AUROC": auroc, "AUPRC": auprc})
    roc_axis.plot(false_positive_rate, true_positive_rate, linewidth=2, color=colors[model_name], label=f"{model_name} (AUROC={auroc:.3f})")
    pr_axis.plot(recall, precision, linewidth=2, color=colors[model_name], label=f"{model_name} (AUPRC={auprc:.3f})")

roc_axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
roc_axis.set_title("Traditional-model ROC curves")
roc_axis.set_xlabel("False-positive rate")
roc_axis.set_ylabel("True-positive rate")
roc_axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
roc_axis.grid(alpha=0.25)
roc_figure.tight_layout(rect=(0, 0, 0.75, 1))
roc_figure.savefig(ROC_PNG, dpi=600, bbox_inches="tight", facecolor="white")
roc_figure.savefig(ROC_PDF, bbox_inches="tight", facecolor="white")
plt.close(roc_figure)

pr_axis.axhline(active_n / len(reference_y), linestyle="--", color="gray", linewidth=1, label=f"Prevalence baseline ({active_n / len(reference_y):.3f})")
pr_axis.set_title("Traditional-model Precision–Recall curves")
pr_axis.set_xlabel("Recall")
pr_axis.set_ylabel("Precision")
pr_axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
pr_axis.grid(alpha=0.25)
pr_figure.tight_layout(rect=(0, 0, 0.75, 1))
pr_figure.savefig(PR_PNG, dpi=600, bbox_inches="tight", facecolor="white")
pr_figure.savefig(PR_PDF, bbox_inches="tight", facecolor="white")
plt.close(pr_figure)

summary_df = pd.DataFrame(summary_rows).sort_values("AUROC", ascending=False)
summary_df.to_csv(CURVE_SUMMARY, index=False)

print("\n36M. Output checks:")
print("ROC PNG exists:", ROC_PNG.exists())
print("ROC PDF exists:", ROC_PDF.exists())
print("PR PNG exists:", PR_PNG.exists())
print("PR PDF exists:", PR_PDF.exists())
print("Summary CSV exists:", CURVE_SUMMARY.exists())
print("\nCurve metrics:")
print(summary_df.to_string(index=False))
print("\n" + "=" * 84)
print("STEP 36 SUMMARY")
print("=" * 84)
print("Models plotted:", len(prediction_tables))
print("Test peptides:", len(reference_y))
print("Active:", active_n)
print("Inactive:", inactive_n)
print("\nROC PNG:", ROC_PNG)
print("ROC PDF:", ROC_PDF)
print("PR PNG:", PR_PNG)
print("PR PDF:", PR_PDF)
print("Curve summary:", CURVE_SUMMARY)
print("\nSTEP 36 COMPLETED SUCCESSFULLY")
print("=" * 84)
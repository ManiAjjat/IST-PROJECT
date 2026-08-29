from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
COMPARISON_OUTPUT = RESULTS_DIR / "step52_esm2_model_comparison.csv"
MANUSCRIPT_OUTPUT = RESULTS_DIR / "step52_esm2_model_comparison_manuscript.csv"
CURVE_SUMMARY_OUTPUT = RESULTS_DIR / "step52_esm2_curve_summary.csv"
ROC_PNG = FIGURE_DIR / "Step52_ESM2_Model_ROC.png"
ROC_PDF = FIGURE_DIR / "Step52_ESM2_Model_ROC.pdf"
PR_PNG = FIGURE_DIR / "Step52_ESM2_Model_PR.png"
PR_PDF = FIGURE_DIR / "Step52_ESM2_Model_PR.pdf"
METRIC_PNG = FIGURE_DIR / "Step52_ESM2_Model_Metric_Comparison.png"
METRIC_PDF = FIGURE_DIR / "Step52_ESM2_Model_Metric_Comparison.pdf"

MODEL_FILES = {
    "Logistic Regression": (
        RESULTS_DIR / "step48_esm2_logistic_regression_test_metrics.csv",
        RESULTS_DIR / "step48_esm2_logistic_regression_test_predictions.csv",
    ),
    "RBF-SVM": (
        RESULTS_DIR / "step49_esm2_svm_test_metrics.csv",
        RESULTS_DIR / "step49_esm2_svm_test_predictions.csv",
    ),
    "Random Forest": (
        RESULTS_DIR / "step50_esm2_random_forest_test_metrics.csv",
        RESULTS_DIR / "step50_esm2_random_forest_test_predictions.csv",
    ),
    "XGBoost": (
        RESULTS_DIR / "step51_esm2_xgboost_test_metrics.csv",
        RESULTS_DIR / "step51_esm2_xgboost_test_predictions.csv",
    ),
}
MODEL_ORDER = list(MODEL_FILES)
COLORS = {
    "Logistic Regression": "#0072B2",
    "RBF-SVM": "#E69F00",
    "Random Forest": "#009E73",
    "XGBoost": "#D55E00",
}
LINESTYLES = {
    "Logistic Regression": "-", "RBF-SVM": "--",
    "Random Forest": "-.", "XGBoost": ":",
}
METRIC_COLORS = {
    "AUROC": "#0072B2", "AUPRC": "#E69F00",
    "MCC": "#009E73", "F1": "#CC79A7",
}
HATCHES = {"AUROC": "", "AUPRC": "//", "MCC": "xx", "F1": ".."}
DECISION_THRESHOLD = 0.5


def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


print("=" * 100)
print("STEP 52 - COMPARE ALL ESM-2 CLASSIFIERS")
print("=" * 100)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

prediction_tables = {}
comparison_rows = []
curve_rows = []
reference = None
metric_source_map = {
    "AUROC": "test_AUROC", "AUPRC": "test_AUPRC", "MCC": "test_MCC",
    "Accuracy": "test_accuracy", "Precision": "test_precision",
    "Recall": "test_recall", "Specificity": "test_specificity", "F1": "test_F1",
    "TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP",
}

for model_name, (metric_path, prediction_path) in MODEL_FILES.items():
    metrics_saved = pd.read_csv(metric_path)
    predictions = pd.read_csv(prediction_path)
    assert len(metrics_saved) == 1 and len(predictions) == 181
    required = {"ID", "sequence", "label", "predicted_probability", "predicted_label", "split"}
    assert required.issubset(predictions.columns)
    assert predictions["ID"].is_unique and predictions["split"].eq("test").all()
    assert predictions["predicted_probability"].between(0, 1).all()
    assert np.isfinite(predictions["predicted_probability"]).all()

    alignment = predictions[["ID", "sequence", "label"]].reset_index(drop=True)
    if reference is None:
        reference = alignment
    else:
        assert alignment.equals(reference)

    labels = predictions["label"].to_numpy(dtype=int)
    probabilities = predictions["predicted_probability"].to_numpy(dtype=float)
    expected_predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    assert np.array_equal(predictions["predicted_label"].to_numpy(dtype=int), expected_predictions)
    calculated = calculate_metrics(labels, probabilities)
    saved = metrics_saved.iloc[0]
    for target, source in metric_source_map.items():
        if target in {"TN", "FP", "FN", "TP"}:
            assert calculated[target] == int(saved[source])
        else:
            assert np.isclose(calculated[target], float(saved[source]), rtol=0, atol=1e-12)

    comparison_rows.append({"model": model_name, **calculated})
    curve_rows.append({
        "model": model_name, "test_n": len(labels),
        "test_active": int(labels.sum()), "test_inactive": int((labels == 0).sum()),
        "AUROC": calculated["AUROC"], "AUPRC": calculated["AUPRC"],
        "prevalence": float(labels.mean()),
    })
    prediction_tables[model_name] = predictions

assert reference is not None and len(reference) == 181
reference_y = reference["label"].to_numpy(dtype=int)
assert int(reference_y.sum()) == 20 and int((reference_y == 0).sum()) == 161

comparison = pd.DataFrame(comparison_rows)
ranking_metrics = ["AUROC", "AUPRC", "MCC", "F1"]
for metric in ranking_metrics:
    comparison[f"{metric}_rank"] = comparison[metric].rank(ascending=False, method="min").astype(int)
comparison["mean_descriptive_rank"] = comparison[[f"{m}_rank" for m in ranking_metrics]].mean(axis=1)
comparison = comparison.sort_values(
    ["mean_descriptive_rank", "AUROC", "MCC"], ascending=[True, False, False], kind="stable"
).reset_index(drop=True)
comparison.insert(0, "descriptive_order", np.arange(1, len(comparison) + 1))

full_columns = [
    "descriptive_order", "model", "AUROC", "AUPRC", "MCC", "Accuracy",
    "Precision", "Recall", "Specificity", "F1", "TN", "FP", "FN", "TP",
    "AUROC_rank", "AUPRC_rank", "MCC_rank", "F1_rank", "mean_descriptive_rank",
]
comparison[full_columns].to_csv(COMPARISON_OUTPUT, index=False)
comparison[[
    "model", "AUROC", "AUPRC", "MCC", "Accuracy", "Precision",
    "Recall", "Specificity", "F1",
]].to_csv(MANUSCRIPT_OUTPUT, index=False)
curve_summary = pd.DataFrame(curve_rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
curve_summary.to_csv(CURVE_SUMMARY_OUTPUT, index=False)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 14,
    "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ROC figure
fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
for model_name in MODEL_ORDER:
    table = prediction_tables[model_name]
    probability = table["predicted_probability"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(reference_y, probability)
    auroc = roc_auc_score(reference_y, probability)
    ax.plot(fpr, tpr, color=COLORS[model_name], linestyle=LINESTYLES[model_name],
            linewidth=2.2, label=f"{model_name} (AUROC={auroc:.3f})")
ax.plot([0, 1], [0, 1], color="#666666", linestyle=(0, (4, 3)), linewidth=1.2, label="Chance")
ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate",
       title="ESM-2 model ROC curves")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.20, linewidth=0.7)
ax.legend(
    loc="lower right", frameon=True, facecolor="white", edgecolor="none",
    framealpha=1.0,
)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.savefig(ROC_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(ROC_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Precision-recall figure
prevalence = float(reference_y.mean())
fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
for model_name in MODEL_ORDER:
    probability = prediction_tables[model_name]["predicted_probability"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(reference_y, probability)
    auprc = average_precision_score(reference_y, probability)
    ax.plot(recall, precision, color=COLORS[model_name], linestyle=LINESTYLES[model_name],
            linewidth=2.2, label=f"{model_name} (AUPRC={auprc:.3f})")
ax.axhline(prevalence, color="#666666", linestyle=(0, (4, 3)), linewidth=1.2,
           label=f"Prevalence baseline ({prevalence:.3f})")
ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision",
       title="ESM-2 model precision–recall curves")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.20, linewidth=0.7)
ax.legend(
    loc="lower left", frameon=True, facecolor="white", edgecolor="none",
    framealpha=1.0,
)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.savefig(PR_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PR_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Grouped metric comparison
plot_data = comparison.set_index("model").loc[MODEL_ORDER, ranking_metrics]
x = np.arange(len(MODEL_ORDER))
width = 0.19
offsets = (np.arange(len(ranking_metrics)) - 1.5) * width
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for index, metric in enumerate(ranking_metrics):
    values = plot_data[metric].to_numpy(dtype=float)
    bars = ax.bar(
        x + offsets[index], values, width=width, label=metric,
        color=METRIC_COLORS[metric], edgecolor="#333333", linewidth=0.5,
        hatch=HATCHES[metric],
    )
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
ax.set(title="ESM-2 model metric comparison", ylabel="Metric value", ylim=(0, 1.08))
ax.set_xticks(x, MODEL_ORDER)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.grid(axis="y", alpha=0.20, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.savefig(METRIC_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(METRIC_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

for output in (
    COMPARISON_OUTPUT, MANUSCRIPT_OUTPUT, CURVE_SUMMARY_OUTPUT,
    ROC_PNG, ROC_PDF, PR_PNG, PR_PDF, METRIC_PNG, METRIC_PDF,
):
    assert output.exists() and output.stat().st_size > 0

print("\nPrediction alignment: IDs, sequences, labels, and order passed")
print("Saved metrics independently reproduced from predictions: passed")
print("\nMetric leaders:")
for metric in ["AUROC", "AUPRC", "MCC", "F1", "Accuracy", "Precision", "Recall", "Specificity"]:
    maximum = comparison[metric].max()
    leaders = comparison.loc[np.isclose(comparison[metric], maximum), "model"].tolist()
    print(f"{metric}: {', '.join(leaders)} ({maximum:.6f})")
print("\nComparison:")
print(comparison[full_columns].round(6).to_string(index=False))
print("\nManuscript table:")
print(comparison[["model", "AUROC", "AUPRC", "MCC", "Accuracy", "Precision", "Recall", "Specificity", "F1"]].round(3).to_string(index=False))
print("\nComparison table:\n", COMPARISON_OUTPUT)
print("\nManuscript table:\n", MANUSCRIPT_OUTPUT)
print("\nCurve summary:\n", CURVE_SUMMARY_OUTPUT)
print("\nROC figure:\n", ROC_PNG)
print("\nPR figure:\n", PR_PNG)
print("\nMetric figure:\n", METRIC_PNG)
print("\nSTEP 52 COMPLETED SUCCESSFULLY")
print("=" * 100)

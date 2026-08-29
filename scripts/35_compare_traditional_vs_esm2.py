from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
OUTPUT_FILE = RESULTS_DIR / "step53_traditional_vs_esm2_comparison.csv"
MANUSCRIPT_OUTPUT = RESULTS_DIR / "step53_traditional_vs_esm2_comparison_manuscript.csv"
FIGURE_PNG = FIGURE_DIR / "Step53_Traditional_vs_ESM2_Metric_Deltas.png"
FIGURE_PDF = FIGURE_DIR / "Step53_Traditional_vs_ESM2_Metric_Deltas.pdf"

MODEL_FILES = {
    "Logistic Regression": {
        "traditional": (
            RESULTS_DIR / "step31_logistic_regression_test_metrics.csv",
            RESULTS_DIR / "step31_logistic_regression_test_predictions.csv",
        ),
        "esm2": (
            RESULTS_DIR / "step48_esm2_logistic_regression_test_metrics.csv",
            RESULTS_DIR / "step48_esm2_logistic_regression_test_predictions.csv",
        ),
    },
    "RBF-SVM": {
        "traditional": (
            RESULTS_DIR / "step32_svm_test_metrics.csv",
            RESULTS_DIR / "step32_svm_test_predictions.csv",
        ),
        "esm2": (
            RESULTS_DIR / "step49_esm2_svm_test_metrics.csv",
            RESULTS_DIR / "step49_esm2_svm_test_predictions.csv",
        ),
    },
    "Random Forest": {
        "traditional": (
            RESULTS_DIR / "step33_random_forest_test_metrics.csv",
            RESULTS_DIR / "step33_random_forest_test_predictions.csv",
        ),
        "esm2": (
            RESULTS_DIR / "step50_esm2_random_forest_test_metrics.csv",
            RESULTS_DIR / "step50_esm2_random_forest_test_predictions.csv",
        ),
    },
    "XGBoost": {
        "traditional": (
            RESULTS_DIR / "step34_xgboost_test_metrics.csv",
            RESULTS_DIR / "step34_xgboost_test_predictions.csv",
        ),
        "esm2": (
            RESULTS_DIR / "step51_esm2_xgboost_test_metrics.csv",
            RESULTS_DIR / "step51_esm2_xgboost_test_predictions.csv",
        ),
    },
}
MODEL_ORDER = list(MODEL_FILES)
METRICS = ["AUROC", "AUPRC", "MCC", "Accuracy", "Precision", "Recall", "Specificity", "F1"]
METRIC_SOURCE = {
    "AUROC": "test_AUROC", "AUPRC": "test_AUPRC", "MCC": "test_MCC",
    "Accuracy": "test_accuracy", "Precision": "test_precision",
    "Recall": "test_recall", "Specificity": "test_specificity", "F1": "test_F1",
}
COLORS = {
    "Logistic Regression": "#0072B2", "RBF-SVM": "#E69F00",
    "Random Forest": "#009E73", "XGBoost": "#D55E00",
}
HATCHES = {
    "Logistic Regression": "", "RBF-SVM": "//",
    "Random Forest": "xx", "XGBoost": "..",
}
THRESHOLD = 0.5


def recompute(labels, probabilities):
    predicted = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "MCC": float(matthews_corrcoef(labels, predicted)),
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


print("=" * 100)
print("STEP 53 - MATCHED TRADITIONAL VS ESM-2 COMPARISON")
print("=" * 100)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

rows = []
global_reference = None
for classifier, representation_files in MODEL_FILES.items():
    pair_reference = None
    pair_metrics = {}
    for representation, (metrics_path, predictions_path) in representation_files.items():
        saved = pd.read_csv(metrics_path)
        predictions = pd.read_csv(predictions_path)
        assert len(saved) == 1 and len(predictions) == 181
        required = {"ID", "sequence", "label", "predicted_probability", "predicted_label", "split"}
        assert required.issubset(predictions.columns)
        assert predictions["ID"].is_unique and predictions["split"].eq("test").all()
        assert predictions["predicted_probability"].between(0, 1).all()
        alignment = predictions[["ID", "sequence", "label"]].reset_index(drop=True)
        if pair_reference is None:
            pair_reference = alignment
        else:
            assert alignment.equals(pair_reference)
        if global_reference is None:
            global_reference = alignment
        else:
            assert alignment.equals(global_reference)

        labels = predictions["label"].to_numpy(dtype=int)
        probabilities = predictions["predicted_probability"].to_numpy(dtype=float)
        expected_label = (probabilities >= THRESHOLD).astype(int)
        assert np.array_equal(predictions["predicted_label"].to_numpy(dtype=int), expected_label)
        calculated = recompute(labels, probabilities)
        for metric, source in METRIC_SOURCE.items():
            assert np.isclose(calculated[metric], float(saved.iloc[0][source]), rtol=0, atol=1e-12)
        for field in ("TN", "FP", "FN", "TP"):
            assert calculated[field] == int(saved.iloc[0][field])
        pair_metrics[representation] = calculated

    row = {"classifier": classifier, "test_n": 181, "test_active": 20, "test_inactive": 161}
    for metric in METRICS:
        traditional_value = pair_metrics["traditional"][metric]
        esm2_value = pair_metrics["esm2"][metric]
        row[f"traditional_{metric}"] = traditional_value
        row[f"esm2_{metric}"] = esm2_value
        row[f"delta_{metric}"] = esm2_value - traditional_value
    for representation in ("traditional", "esm2"):
        for field in ("TN", "FP", "FN", "TP"):
            row[f"{representation}_{field}"] = pair_metrics[representation][field]
    rows.append(row)

assert global_reference is not None and len(global_reference) == 181
assert int(global_reference["label"].sum()) == 20
comparison = pd.DataFrame(rows)
assert comparison.shape[0] == 4
assert np.isfinite(comparison.select_dtypes("number")).all().all()
comparison.to_csv(OUTPUT_FILE, index=False)

manuscript_columns = ["classifier"]
for metric in ("AUROC", "AUPRC", "MCC", "F1"):
    manuscript_columns.extend([f"traditional_{metric}", f"esm2_{metric}", f"delta_{metric}"])
comparison[manuscript_columns].to_csv(MANUSCRIPT_OUTPUT, index=False)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})
figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
panels = [
    (axes[0], ["AUROC", "AUPRC"], "A", "Probability-ranking metrics", (-0.01, 0.285)),
    (axes[1], ["MCC", "Accuracy", "Precision", "Recall", "Specificity", "F1"],
     "B", "Threshold-dependent metrics (cutoff = 0.5)", (-0.12, 0.225)),
]
for axis, panel_metrics, panel_label, title, y_limits in panels:
    x = np.arange(len(panel_metrics))
    width = 0.19
    offsets = (np.arange(len(MODEL_ORDER)) - 1.5) * width
    for model_index, classifier in enumerate(MODEL_ORDER):
        values = comparison.set_index("classifier").loc[
            classifier, [f"delta_{metric}" for metric in panel_metrics]
        ].to_numpy(dtype=float)
        bars = axis.bar(
            x + offsets[model_index], values, width=width, label=classifier,
            color=COLORS[classifier], edgecolor="#333333", linewidth=0.5,
            hatch=HATCHES[classifier],
        )
        for bar, value in zip(bars, values):
            vertical_alignment = "bottom" if value >= 0 else "top"
            offset = 3 if value >= 0 else -3
            axis.annotate(
                f"{value:+.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=vertical_alignment, fontsize=7, rotation=90,
            )
    axis.axhline(0, color="#222222", linewidth=1.1)
    axis.set_xticks(x, panel_metrics)
    axis.set_ylabel("Delta (ESM-2 - traditional)")
    axis.set_ylim(*y_limits)
    axis.set_title(title, pad=9)
    axis.text(-0.07, 1.04, panel_label, transform=axis.transAxes,
              fontsize=13, fontweight="bold", va="bottom")
    axis.grid(axis="y", alpha=0.20, linewidth=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.30), frameon=False)
figure.suptitle("Matched traditional versus ESM-2 performance differences", fontsize=15, y=1.03)
figure.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
figure.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(figure)

for output in (OUTPUT_FILE, MANUSCRIPT_OUTPUT, FIGURE_PNG, FIGURE_PDF):
    assert output.exists() and output.stat().st_size > 0

print("\nAlignment across all eight prediction files: passed")
print("All saved metrics independently reproduced: passed")
print("\nMatched deltas:")
delta_columns = ["classifier", *[f"delta_{metric}" for metric in METRICS]]
print(comparison[delta_columns].round(6).to_string(index=False))
print("\nPositive-delta counts:")
for metric in METRICS:
    print(f"{metric}: {int((comparison[f'delta_{metric}'] > 0).sum())}/4")
print("\nLargest apparent gains:")
for metric in METRICS:
    index = comparison[f"delta_{metric}"].idxmax()
    print(f"{metric}: {comparison.loc[index, 'classifier']} ({comparison.loc[index, f'delta_{metric}']:+.6f})")
print("\nFull matched comparison:\n", OUTPUT_FILE)
print("\nManuscript comparison:\n", MANUSCRIPT_OUTPUT)
print("\nDelta figure:\n", FIGURE_PNG)
print("Figure PDF exists:", FIGURE_PDF.exists())
print("\n" + "=" * 100)
print("STEP 53 SUMMARY")
print("=" * 100)
print("Matched classifiers:", len(comparison))
print("Locked-test peptides:", 181)
print("Positive delta_AUROC classifiers:", int((comparison["delta_AUROC"] > 0).sum()))
print("Positive delta_AUPRC classifiers:", int((comparison["delta_AUPRC"] > 0).sum()))
print("Positive delta_MCC classifiers:", int((comparison["delta_MCC"] > 0).sum()))
print("Positive delta_F1 classifiers:", int((comparison["delta_F1"] > 0).sum()))
print("\nSTEP 53 COMPLETED SUCCESSFULLY")
print("=" * 100)

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\postdoc-work\ist-project")
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
MODEL_SPECS = [
    ("Traditional Logistic Regression", "Traditional", "Logistic Regression", "step31_logistic_regression_test_predictions.csv", "step31_logistic_regression_test_metrics.csv"),
    ("Traditional RBF-SVM", "Traditional", "RBF-SVM", "step32_svm_test_predictions.csv", "step32_svm_test_metrics.csv"),
    ("Traditional Random Forest", "Traditional", "Random Forest", "step33_random_forest_test_predictions.csv", "step33_random_forest_test_metrics.csv"),
    ("Traditional XGBoost", "Traditional", "XGBoost", "step34_xgboost_test_predictions.csv", "step34_xgboost_test_metrics.csv"),
    ("ESM-2 Logistic Regression", "ESM-2", "Logistic Regression", "step48_esm2_logistic_regression_test_predictions.csv", "step48_esm2_logistic_regression_test_metrics.csv"),
    ("ESM-2 RBF-SVM", "ESM-2", "RBF-SVM", "step49_esm2_svm_test_predictions.csv", "step49_esm2_svm_test_metrics.csv"),
    ("ESM-2 Random Forest", "ESM-2", "Random Forest", "step50_esm2_random_forest_test_predictions.csv", "step50_esm2_random_forest_test_metrics.csv"),
    ("ESM-2 XGBoost", "ESM-2", "XGBoost", "step51_esm2_xgboost_test_predictions.csv", "step51_esm2_xgboost_test_metrics.csv"),
]
SUMMARY_OUT = RESULTS / "step74_model_performance_bootstrap_summary.csv"
REPLICATES_OUT = RESULTS / "step74_model_performance_bootstrap_replicates.csv"
QC_OUT = RESULTS / "step74_model_performance_bootstrap_qc.csv"
FIGURE_PNG = FIGURES / "Step74_Model_Performance_Confidence_Intervals.png"
FIGURE_PDF = FIGURES / "Step74_Model_Performance_Confidence_Intervals.pdf"
N_BOOTSTRAP = 5000
SEED = 2026
THRESHOLD = 0.5
METRICS = ["AUROC", "AUPRC", "MCC", "F1", "Accuracy", "Sensitivity", "Specificity", "Precision"]


def calculate_metrics(y, probability):
    prediction = (probability >= THRESHOLD).astype(int)
    specificity = recall_score(y, prediction, pos_label=0, zero_division=0)
    return {
        "AUROC": roc_auc_score(y, probability),
        "AUPRC": average_precision_score(y, probability),
        "MCC": matthews_corrcoef(y, prediction),
        "F1": f1_score(y, prediction, zero_division=0),
        "Accuracy": accuracy_score(y, prediction),
        "Sensitivity": recall_score(y, prediction, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y, prediction, zero_division=0),
    }


print("=" * 110)
print("STEP 74 - STRATIFIED BOOTSTRAP CONFIDENCE INTERVALS FOR MODEL PERFORMANCE")
print("=" * 110)

loaded = {}
reference = None
existing_metric_rows = {}
for model, representation, classifier, prediction_file, metric_file in MODEL_SPECS:
    frame = pd.read_csv(RESULTS / prediction_file)
    required = ["ID", "sequence", "label", "predicted_probability", "predicted_label"]
    if set(required) - set(frame.columns):
        raise ValueError(f"{model}: missing prediction columns")
    if len(frame) != 181 or frame.ID.nunique() != 181:
        raise ValueError(f"{model}: expected 181 unique rows")
    aligned = frame[["ID", "sequence", "label"]].copy()
    if reference is None:
        reference = aligned
    elif not aligned.reset_index(drop=True).equals(reference.reset_index(drop=True)):
        raise ValueError(f"{model}: ID/sequence/label alignment failed")
    probability = frame.predicted_probability.to_numpy(float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError(f"{model}: invalid probabilities")
    if not np.array_equal((probability >= THRESHOLD).astype(int), frame.predicted_label.to_numpy(int)):
        raise ValueError(f"{model}: saved labels do not match threshold 0.5")
    loaded[model] = (representation, classifier, probability)
    existing_metric_rows[model] = pd.read_csv(RESULTS / metric_file).iloc[0]

y = reference.label.to_numpy(int)
active_indices = np.flatnonzero(y == 1)
inactive_indices = np.flatnonzero(y == 0)
if (len(y), len(active_indices), len(inactive_indices)) != (181, 20, 161):
    raise ValueError("Expected 181 peptides with 20 Active and 161 Inactive")

# One shared stratified bootstrap sample stream for all models.
rng = np.random.default_rng(SEED)
sample_indices = np.empty((N_BOOTSTRAP, len(y)), dtype=np.int32)
sample_hashes = []
for replicate in range(N_BOOTSTRAP):
    indices = np.concatenate([
        rng.choice(active_indices, size=len(active_indices), replace=True),
        rng.choice(inactive_indices, size=len(inactive_indices), replace=True),
    ]).astype(np.int32)
    sample_indices[replicate] = indices
    sample_hashes.append(hashlib.sha256(indices.tobytes()).hexdigest())

replicate_frames = []
summary_rows = []
maximum_existing_metric_error = 0.0
for model, representation, classifier, _, _ in MODEL_SPECS:
    _, _, probability = loaded[model]
    point = calculate_metrics(y, probability)
    existing = existing_metric_rows[model]
    existing_mapping = {
        "AUROC": "test_AUROC", "AUPRC": "test_AUPRC", "MCC": "test_MCC",
        "F1": "test_F1", "Accuracy": "test_accuracy", "Sensitivity": "test_recall",
        "Specificity": "test_specificity", "Precision": "test_precision",
    }
    for metric, existing_column in existing_mapping.items():
        maximum_existing_metric_error = max(maximum_existing_metric_error, abs(point[metric] - float(existing[existing_column])))

    values = {metric: np.empty(N_BOOTSTRAP, dtype=float) for metric in METRICS}
    for replicate, indices in enumerate(sample_indices):
        calculated = calculate_metrics(y[indices], probability[indices])
        for metric in METRICS:
            values[metric][replicate] = calculated[metric]
    replicate_frame = pd.DataFrame({
        "model": model, "representation": representation, "classifier": classifier,
        "replicate": np.arange(1, N_BOOTSTRAP + 1), "seed": SEED,
        "sample_index_sha256": sample_hashes, "replicate_n": len(y),
        "active_n": len(active_indices), "inactive_n": len(inactive_indices),
        **values,
    })
    replicate_frames.append(replicate_frame)

    row = {
        "model": model, "representation": representation, "classifier": classifier,
        "test_n": len(y), "active_n": len(active_indices), "inactive_n": len(inactive_indices),
        "classification_threshold": THRESHOLD, "bootstrap_replicates": N_BOOTSTRAP, "bootstrap_seed": SEED,
    }
    for metric in METRICS:
        low, high = np.percentile(values[metric], [2.5, 97.5])
        row[metric] = point[metric]
        row[f"{metric}_CI_low"] = low
        row[f"{metric}_CI_high"] = high
        row[f"{metric}_bootstrap_mean"] = values[metric].mean()
        row[f"{metric}_bootstrap_SE"] = values[metric].std(ddof=1)
    summary_rows.append(row)

replicates = pd.concat(replicate_frames, ignore_index=True)
summary = pd.DataFrame(summary_rows)
if len(replicates) != 40000 or len(summary) != 8:
    raise ValueError("Unexpected output row counts")
if not np.isfinite(replicates[METRICS].to_numpy()).all() or not np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all():
    raise ValueError("Non-finite metric values")
replicates.to_csv(REPLICATES_OUT, index=False)
summary.to_csv(SUMMARY_OUT, index=False)

hashes_per_replicate = replicates.groupby("replicate").sample_index_sha256.nunique()
rows_per_model = replicates.groupby("model").size()
replicate_ids_per_model = replicates.groupby("model").replicate.apply(lambda x: np.array_equal(x.to_numpy(), np.arange(1, N_BOOTSTRAP + 1)))
qc = pd.DataFrame([{
    "locked_test_peptides": len(y), "active": len(active_indices), "inactive": len(inactive_indices),
    "models": len(MODEL_SPECS), "bootstrap_seed": SEED, "bootstrap_replicates_per_model": N_BOOTSTRAP,
    "expected_replicate_rows": 40000, "observed_replicate_rows": len(replicates),
    "summary_rows": len(summary), "classification_threshold": THRESHOLD,
    "all_model_alignments_exact": True,
    "replicate_IDs_1_to_5000_for_every_model": bool(replicate_ids_per_model.all()),
    "rows_per_model_exactly_5000": bool((rows_per_model == N_BOOTSTRAP).all()),
    "one_sample_index_hash_per_replicate_across_all_models": bool((hashes_per_replicate == 1).all()),
    "replicate_n_every_row": bool((replicates.replicate_n == 181).all()),
    "replicate_active_n_every_row": bool((replicates.active_n == 20).all()),
    "replicate_inactive_n_every_row": bool((replicates.inactive_n == 161).all()),
    "all_metrics_finite": True,
    "maximum_point_estimate_error_vs_existing_test_metrics": maximum_existing_metric_error,
    "saved_point_estimates_reproduce_steps_31_34_and_48_51": maximum_existing_metric_error < 1e-12,
    "models_trained": False, "models_retrained": False,
    "thresholds_optimized": False, "recalibration_performed": False,
    "probabilities_changed": False, "predictions_changed": False,
}])
qc.to_csv(QC_OUT, index=False)

# Four-panel forest plot for manuscript metrics.
plot_metrics = ["AUROC", "AUPRC", "MCC", "F1"]
display = summary.iloc[::-1].reset_index(drop=True).copy()
display["short_model"] = display.representation.map({"Traditional": "Trad.", "ESM-2": "ESM-2"}) + " " + display.classifier
y_positions = np.arange(len(display))
colors = display.representation.map({"Traditional": "#4C78A8", "ESM-2": "#E45756"})
markers = display.representation.map({"Traditional": "o", "ESM-2": "D"})
fig, axes = plt.subplots(1, 4, figsize=(16.2, 6.3), sharey=True)
for panel, (ax, metric) in enumerate(zip(axes, plot_metrics)):
    point = display[metric].to_numpy()
    low = display[f"{metric}_CI_low"].to_numpy(); high = display[f"{metric}_CI_high"].to_numpy()
    for position, value, lower, upper, color, marker in zip(y_positions, point, low, high, colors, markers):
        ax.errorbar(value, position, xerr=np.array([[value - lower], [upper - value]]),
                    fmt=marker, color=color, ecolor=color, capsize=3.5, markersize=6.5, linewidth=1.6,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.set(title=metric, xlabel="Point estimate and 95% CI", yticks=y_positions, yticklabels=display.short_model)
    ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.16, 1.04, chr(65 + panel), transform=ax.transAxes, fontsize=13, fontweight="bold")
axes[0].set_ylabel("Frozen model")
from matplotlib.lines import Line2D
axes[-1].legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", label="Traditional", markersize=7),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#E45756", label="ESM-2", markersize=7),
], frameon=False, loc="lower right", fontsize=8.5)
fig.suptitle("Locked-test model performance with stratified-bootstrap uncertainty", fontsize=14, fontweight="bold")
fig.text(0.5, 0.025, "Intervals are 95% percentile intervals from 5,000 shared stratified bootstrap samples; threshold-dependent metrics use 0.50.", ha="center", fontsize=8.5)
plt.tight_layout(rect=[0.02, 0.07, 0.99, 0.93])
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nMain performance estimates and intervals:")
columns = ["model"]
for metric in plot_metrics:
    columns += [metric, f"{metric}_CI_low", f"{metric}_CI_high"]
print(summary[columns].round(6).to_string(index=False))
print("\nReplicate rows:", len(replicates))
print("Maximum point-estimate error vs frozen result files:", maximum_existing_metric_error)
print("\nSTEP 74 COMPLETED SUCCESSFULLY")
print("=" * 110)

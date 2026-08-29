from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

INPUTS = {
    "ESM-2 Logistic Regression": RESULTS_DIR / "step48_esm2_logistic_regression_test_predictions.csv",
    "ESM-2 RBF-SVM": RESULTS_DIR / "step49_esm2_svm_test_predictions.csv",
    "Fusion Logistic Regression": RESULTS_DIR / "step79_fusion_logistic_regression_test_predictions.csv",
    "Fusion RBF-SVM": RESULTS_DIR / "step79_fusion_svm_test_predictions.csv",
}
COMPARISONS = [
    ("Logistic Regression", "ESM-2 Logistic Regression", "Fusion Logistic Regression"),
    ("RBF-SVM", "ESM-2 RBF-SVM", "Fusion RBF-SVM"),
]
METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
REPLICATES = 5000
SEED = 2026
THRESHOLD = 0.5

SUMMARY_OUTPUT = RESULTS_DIR / "step80_fusion_vs_esm2_paired_bootstrap_summary.csv"
REPLICATE_OUTPUT = RESULTS_DIR / "step80_fusion_vs_esm2_paired_bootstrap_replicates.csv"
QC_OUTPUT = RESULTS_DIR / "step80_fusion_vs_esm2_paired_bootstrap_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step80_Fusion_vs_ESM2_Paired_Bootstrap.png"
FIGURE_PDF = FIGURES_DIR / "Step80_Fusion_vs_ESM2_Paired_Bootstrap.pdf"


print("=" * 108)
print("STEP 80 - PAIRED BOOTSTRAP: FUSION VS ESM-2-ONLY")
print("=" * 108)

tables = {}
reference = None
for model, path in INPUTS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen prediction input: {path}")
    table = pd.read_csv(path)
    if len(table) != 181 or table["ID"].nunique() != 181:
        raise ValueError(f"{model}: expected 181 unique test IDs.")
    if table["label"].value_counts().to_dict() != {0: 161, 1: 20}:
        raise ValueError(f"{model}: expected 20 Active and 161 Inactive peptides.")
    probability = table["predicted_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"{model}: invalid probabilities.")
    if not np.array_equal(table["predicted_label"].to_numpy(), (probability >= THRESHOLD).astype(int)):
        raise ValueError(f"{model}: saved labels do not equal probability >= 0.5.")
    identity = table[["ID", "sequence", "label"]].reset_index(drop=True)
    if reference is None:
        reference = identity
    elif not identity.equals(reference):
        raise ValueError(f"{model}: ID/sequence/label alignment failed.")
    tables[model] = table

y = reference["label"].to_numpy(dtype=int)
active_idx = np.flatnonzero(y == 1)
inactive_idx = np.flatnonzero(y == 0)
rng = np.random.default_rng(SEED)
sample_indices = np.empty((REPLICATES, 181), dtype=np.int64)
sample_hashes = []
for replicate in range(REPLICATES):
    indices = np.concatenate([
        rng.choice(active_idx, size=20, replace=True),
        rng.choice(inactive_idx, size=161, replace=True),
    ])
    sample_indices[replicate] = indices
    sample_hashes.append(hashlib.sha256(indices.tobytes()).hexdigest())


def calculate_metrics(y_true, probability):
    prediction = (probability >= THRESHOLD).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, probability),
        "AUPRC": average_precision_score(y_true, probability),
        "MCC": matthews_corrcoef(y_true, prediction),
        "F1": f1_score(y_true, prediction, zero_division=0),
    }


point_metrics = {}
for model, table in tables.items():
    point_metrics[model] = calculate_metrics(y, table["predicted_probability"].to_numpy(dtype=float))

summary_rows = []
replicate_frames = []
for classifier, esm_model, fusion_model in COMPARISONS:
    esm_probability = tables[esm_model]["predicted_probability"].to_numpy(dtype=float)
    fusion_probability = tables[fusion_model]["predicted_probability"].to_numpy(dtype=float)
    esm_values = {metric: np.empty(REPLICATES) for metric in METRICS}
    fusion_values = {metric: np.empty(REPLICATES) for metric in METRICS}

    for replicate, indices in enumerate(sample_indices):
        sampled_y = y[indices]
        esm_metrics = calculate_metrics(sampled_y, esm_probability[indices])
        fusion_metrics = calculate_metrics(sampled_y, fusion_probability[indices])
        for metric in METRICS:
            esm_values[metric][replicate] = esm_metrics[metric]
            fusion_values[metric][replicate] = fusion_metrics[metric]

    for metric in METRICS:
        delta = fusion_values[metric] - esm_values[metric]
        lower, upper = np.percentile(delta, [2.5, 97.5])
        if lower > 0:
            status = "favors_Fusion"
        elif upper < 0:
            status = "favors_ESM2_only"
        else:
            status = "includes_zero"
        p_le = float(np.mean(delta <= 0))
        p_ge = float(np.mean(delta >= 0))
        summary_rows.append({
            "classifier": classifier,
            "metric": metric,
            "esm2_model": esm_model,
            "fusion_model": fusion_model,
            "esm2_point": point_metrics[esm_model][metric],
            "fusion_point": point_metrics[fusion_model][metric],
            "observed_delta_fusion_minus_esm2": point_metrics[fusion_model][metric] - point_metrics[esm_model][metric],
            "bootstrap_mean_delta": float(delta.mean()),
            "bootstrap_se_delta": float(delta.std(ddof=1)),
            "ci_2_5": float(lower),
            "ci_97_5": float(upper),
            "ci_excludes_zero": bool(status != "includes_zero"),
            "interval_status": status,
            "bootstrap_fraction_fusion_greater": float(np.mean(delta > 0)),
            "bootstrap_fraction_esm2_greater": float(np.mean(delta < 0)),
            "bootstrap_fraction_equal": float(np.mean(delta == 0)),
            "descriptive_two_sided_tail_probability": min(1.0, 2.0 * min(p_le, p_ge)),
            "bootstrap_replicates": REPLICATES,
            "delta_definition": "Fusion_minus_ESM2",
        })
        replicate_frames.append(pd.DataFrame({
            "classifier": classifier,
            "metric": metric,
            "replicate": np.arange(1, REPLICATES + 1),
            "seed": SEED,
            "sample_index_sha256": sample_hashes,
            "replicate_n": 181,
            "active_n": 20,
            "inactive_n": 161,
            "esm2_value": esm_values[metric],
            "fusion_value": fusion_values[metric],
            "delta_fusion_minus_esm2": delta,
        }))

summary = pd.DataFrame(summary_rows)
replicates = pd.concat(replicate_frames, ignore_index=True)
if summary.shape[0] != 8 or replicates.shape[0] != 40000:
    raise ValueError("Unexpected Step-80 output row counts.")
if replicates[["classifier", "metric", "replicate"]].duplicated().any():
    raise ValueError("Duplicate classifier/metric/replicate key.")
if not np.isfinite(replicates[["esm2_value", "fusion_value", "delta_fusion_minus_esm2"]]).all().all():
    raise ValueError("Non-finite bootstrap metric detected.")

summary.to_csv(SUMMARY_OUTPUT, index=False)
replicates.to_csv(REPLICATE_OUTPUT, index=False)

# Two-panel paired-delta plot.
fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0), sharex=True, facecolor="white")
for ax, (classifier, _, _), panel in zip(axes, COMPARISONS, ["A", "B"]):
    ax.set_facecolor("white")
    plot = summary[summary["classifier"] == classifier].set_index("metric").loc[METRICS].reset_index()
    y_pos = np.arange(4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    for i, row in plot.iterrows():
        observed = row["observed_delta_fusion_minus_esm2"]
        lower = row["ci_2_5"]
        upper = row["ci_97_5"]
        supported = bool(row["ci_excludes_zero"])
        marker = "D" if supported else "o"
        color = "#C44E52" if supported else "#4C78A8"
        ax.errorbar(
            observed, y_pos[i], xerr=[[observed - lower], [upper - observed]],
            fmt=marker, color=color, ecolor=color, markeredgecolor="white",
            markersize=9, capsize=5, linewidth=1.7,
        )
        ax.text(upper, y_pos[i], f"  {observed:+.4f}", va="center", fontsize=9)
    ax.set_yticks(y_pos, METRICS)
    ax.invert_yaxis()
    ax.set_title(f"{panel}  {classifier}", loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("Paired difference: Fusion - ESM-2")
    ax.grid(axis="x", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Paired Bootstrap Comparison of Fusion and ESM-2-Only Models", fontsize=17, fontweight="bold", y=0.98)
fig.text(
    0.5, 0.02,
    "Positive values favor Fusion. Diamonds indicate intervals excluding zero; circles indicate intervals including zero. No multiplicity-adjusted claim is made.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0.04, 0.07, 0.99, 0.94])
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

point_identity_error = float(np.max(np.abs(
    summary["observed_delta_fusion_minus_esm2"]
    - (summary["fusion_point"] - summary["esm2_point"])
)))
replicate_identity_error = float(np.max(np.abs(
    replicates["delta_fusion_minus_esm2"]
    - (replicates["fusion_value"] - replicates["esm2_value"])
)))
hashes_per_replicate = replicates.groupby("replicate")["sample_index_sha256"].nunique()
qc = pd.DataFrame([{
    "prediction_files": 4,
    "rows_per_file": 181,
    "unique_ids_per_file": 181,
    "active": 20,
    "inactive": 161,
    "alignment_exact_across_four_files": True,
    "all_probabilities_finite_and_in_unit_interval": True,
    "saved_labels_equal_probability_ge_0_5": True,
    "bootstrap_seed": SEED,
    "bootstrap_replicates": REPLICATES,
    "shared_sample_indices_across_four_models": bool((hashes_per_replicate == 1).all()),
    "replicate_n_every_row": bool((replicates["replicate_n"] == 181).all()),
    "replicate_active_n_every_row": bool((replicates["active_n"] == 20).all()),
    "replicate_inactive_n_every_row": bool((replicates["inactive_n"] == 161).all()),
    "classification_threshold": THRESHOLD,
    "summary_rows": len(summary),
    "replicate_rows": len(replicates),
    "maximum_point_delta_identity_error": point_identity_error,
    "maximum_replicate_delta_identity_error": replicate_identity_error,
    "all_bootstrap_metrics_finite": True,
    "models_loaded": False,
    "models_retrained": False,
    "models_tuned": False,
    "recalibration_performed": False,
    "threshold_optimized": False,
    "probabilities_changed": False,
    "predictions_changed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nPaired summary:")
print(summary[[
    "classifier", "metric", "esm2_point", "fusion_point",
    "observed_delta_fusion_minus_esm2", "ci_2_5", "ci_97_5", "interval_status",
]].round(6).to_string(index=False))
print("\nSTEP 80 COMPLETED SUCCESSFULLY")
print("=" * 108)

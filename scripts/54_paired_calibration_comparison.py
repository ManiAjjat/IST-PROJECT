from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\postdoc-work\ist-project")
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
METRICS_FILE = RESULTS / "step71_calibration_metrics.csv"
BOOTSTRAP_FILE = RESULTS / "step71_calibration_bootstrap.csv"
SUMMARY_OUT = RESULTS / "step72_paired_calibration_comparison.csv"
DELTAS_OUT = RESULTS / "step72_paired_calibration_bootstrap_deltas.csv"
QC_OUT = RESULTS / "step72_paired_calibration_qc.csv"
FIGURE_PNG = FIGURES / "Step72_Traditional_vs_ESM2_Calibration_Deltas.png"
FIGURE_PDF = FIGURES / "Step72_Traditional_vs_ESM2_Calibration_Deltas.pdf"

PAIRS = {
    "Logistic Regression": ("Traditional Logistic Regression", "ESM-2 Logistic Regression"),
    "RBF-SVM": ("Traditional RBF-SVM", "ESM-2 RBF-SVM"),
    "Random Forest": ("Traditional Random Forest", "ESM-2 Random Forest"),
    "XGBoost": ("Traditional XGBoost", "ESM-2 XGBoost"),
}
METRICS = {
    "Brier": ("brier_score", "brier_score"),
    "Log loss": ("log_loss", "log_loss"),
    "ECE": ("ece_10_equal_width_bins", "ece_10_equal_width_bins"),
}
N_REPLICATES = 2000


print("=" * 112)
print("STEP 72 - PAIRED TRADITIONAL VS ESM-2 CALIBRATION COMPARISON")
print("=" * 112)

metrics = pd.read_csv(METRICS_FILE)
bootstrap = pd.read_csv(BOOTSTRAP_FILE)
if metrics.shape[0] != 8 or bootstrap.shape[0] != 16000:
    raise ValueError("Unexpected Step-71 input dimensions")
if metrics.model.nunique() != 8 or bootstrap.model.nunique() != 8:
    raise ValueError("Expected exactly eight models")
if set(metrics.model) != set(bootstrap.model):
    raise ValueError("Metric/bootstrap model sets differ")
for column in ["model", "replicate", "seed", "active_n", "inactive_n"] + [v[1] for v in METRICS.values()]:
    if column not in bootstrap.columns:
        raise ValueError(f"Missing bootstrap column: {column}")
if not np.isfinite(bootstrap[[v[1] for v in METRICS.values()]].to_numpy()).all():
    raise ValueError("Non-finite bootstrap metrics")
if not (bootstrap.groupby("model").size() == N_REPLICATES).all():
    raise ValueError("Each model must have 2000 bootstrap rows")
if not bootstrap.groupby("model").replicate.apply(lambda x: np.array_equal(np.sort(x), np.arange(1, N_REPLICATES + 1))).all():
    raise ValueError("Bootstrap replicate identifiers incomplete")
if not ((bootstrap.seed == 2026) & (bootstrap.active_n == 20) & (bootstrap.inactive_n == 161)).all():
    raise ValueError("Step-71 bootstrap design fields differ from expected")

summary_rows = []
delta_frames = []
for classifier, (traditional_model, esm2_model) in PAIRS.items():
    traditional_point = metrics.loc[metrics.model == traditional_model].iloc[0]
    esm2_point = metrics.loc[metrics.model == esm2_model].iloc[0]
    traditional_boot = bootstrap.loc[bootstrap.model == traditional_model].sort_values("replicate").reset_index(drop=True)
    esm2_boot = bootstrap.loc[bootstrap.model == esm2_model].sort_values("replicate").reset_index(drop=True)
    for column in ["replicate", "seed", "active_n", "inactive_n"]:
        if not np.array_equal(traditional_boot[column].to_numpy(), esm2_boot[column].to_numpy()):
            raise ValueError(f"{classifier}: bootstrap design alignment failed for {column}")

    for metric, (point_column, boot_column) in METRICS.items():
        traditional_value = float(traditional_point[point_column])
        esm2_value = float(esm2_point[point_column])
        observed_delta = esm2_value - traditional_value
        replicate_delta = esm2_boot[boot_column].to_numpy(float) - traditional_boot[boot_column].to_numpy(float)
        lower, upper = np.percentile(replicate_delta, [2.5, 97.5])
        excludes_zero = bool(lower > 0 or upper < 0)
        direction = "favors_ESM2" if upper < 0 else ("favors_traditional" if lower > 0 else "includes_zero")
        p_le = float(np.mean(replicate_delta <= 0))
        p_ge = float(np.mean(replicate_delta >= 0))
        summary_rows.append({
            "classifier": classifier,
            "traditional_model": traditional_model,
            "esm2_model": esm2_model,
            "metric": metric,
            "traditional_point": traditional_value,
            "esm2_point": esm2_value,
            "observed_delta_esm2_minus_traditional": observed_delta,
            "bootstrap_mean_delta": float(replicate_delta.mean()),
            "bootstrap_se_delta": float(replicate_delta.std(ddof=1)),
            "ci_2_5": float(lower),
            "ci_97_5": float(upper),
            "ci_excludes_zero": excludes_zero,
            "interval_direction": direction,
            "bootstrap_replicates": N_REPLICATES,
            "descriptive_two_sided_tail_probability": min(1.0, 2 * min(p_le, p_ge)),
            "lower_metric_is_better": True,
        })
        delta_frames.append(pd.DataFrame({
            "classifier": classifier,
            "metric": metric,
            "replicate": traditional_boot.replicate.to_numpy(),
            "traditional_value": traditional_boot[boot_column].to_numpy(float),
            "esm2_value": esm2_boot[boot_column].to_numpy(float),
            "delta_esm2_minus_traditional": replicate_delta,
        }))

summary = pd.DataFrame(summary_rows)
deltas = pd.concat(delta_frames, ignore_index=True)
if summary.shape != (12, 16) or deltas.shape != (24000, 6):
    raise ValueError(f"Unexpected outputs: {summary.shape}, {deltas.shape}")
if summary.duplicated(["classifier", "metric"]).any() or deltas.duplicated(["classifier", "metric", "replicate"]).any():
    raise ValueError("Duplicate comparison keys")

point_identity_error = float(np.max(np.abs(summary.observed_delta_esm2_minus_traditional - (summary.esm2_point - summary.traditional_point))))
bootstrap_identity_error = float(np.max(np.abs(deltas.delta_esm2_minus_traditional - (deltas.esm2_value - deltas.traditional_value))))
if point_identity_error > 1e-12 or bootstrap_identity_error > 1e-12:
    raise ValueError("Delta identity check failed")

summary.to_csv(SUMMARY_OUT, index=False)
deltas.to_csv(DELTAS_OUT, index=False)

below = int((summary.ci_97_5 < 0).sum())
above = int((summary.ci_2_5 > 0).sum())
includes = int(len(summary) - below - above)
qc = pd.DataFrame([{
    "step71_metric_rows": len(metrics),
    "step71_bootstrap_rows": len(bootstrap),
    "matched_classifier_pairs": len(PAIRS),
    "metrics": len(METRICS),
    "paired_comparisons": len(summary),
    "bootstrap_replicates_per_model": N_REPLICATES,
    "expected_bootstrap_delta_rows": 24000,
    "observed_bootstrap_delta_rows": len(deltas),
    "replicate_IDs_complete_and_aligned": True,
    "seed_and_class_counts_aligned_within_pairs": True,
    "delta_definition": "ESM2_minus_traditional",
    "lower_metric_is_better": True,
    "intervals_entirely_below_zero": below,
    "intervals_entirely_above_zero": above,
    "intervals_including_zero": includes,
    "maximum_point_delta_identity_error": point_identity_error,
    "maximum_bootstrap_delta_identity_error": bootstrap_identity_error,
    "new_bootstrap_sampling_performed": False,
    "models_loaded_for_training": False,
    "models_retrained": False,
    "probabilities_changed": False,
    "recalibration_performed": False,
    "model_selection_performed": False,
    "multiplicity_adjusted_inference_claimed": False,
}])
qc.to_csv(QC_OUT, index=False)

# Three-panel paired-delta forest plot.
classifier_order = list(PAIRS)
colors = {"Logistic Regression": "#0072B2", "RBF-SVM": "#D55E00", "Random Forest": "#009E73", "XGBoost": "#CC79A7"}
fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.8))
for panel, (ax, metric) in enumerate(zip(axes, METRICS)):
    plot = summary.loc[summary.metric == metric].set_index("classifier").loc[classifier_order].reset_index()
    y = np.arange(len(plot))
    ax.axvspan(plot.ci_2_5.min() - abs(plot.ci_2_5.min()) * 0.1 - 0.001, 0, color="#E8F3FA", alpha=0.55, zorder=0)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    for index, row in plot.iterrows():
        marker = "D" if row.ci_excludes_zero else "o"
        ax.errorbar(row.observed_delta_esm2_minus_traditional, y[index],
                    xerr=np.array([[row.observed_delta_esm2_minus_traditional - row.ci_2_5], [row.ci_97_5 - row.observed_delta_esm2_minus_traditional]]),
                    fmt=marker, color=colors[row.classifier], ecolor=colors[row.classifier],
                    markersize=7, capsize=4, linewidth=1.6)
        ax.annotate(f"{row.observed_delta_esm2_minus_traditional:+.4f}",
                    (row.observed_delta_esm2_minus_traditional, y[index]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color="black")
    ax.set(yticks=y, yticklabels=classifier_order, xlabel="ESM-2 minus Traditional", title=metric)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.05, chr(65 + panel), transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.suptitle("Paired calibration differences: ESM-2 vs traditional descriptors", fontsize=14, fontweight="bold")
fig.text(0.5, 0.075, "← Negative values favor ESM-2 | Positive values favor traditional descriptors →", ha="center", fontsize=9)
fig.text(0.5, 0.035, "Diamonds: 95% paired percentile interval excludes zero; circles: interval includes zero. Descriptive, not multiplicity-adjusted.", ha="center", fontsize=8.5)
plt.tight_layout(rect=[0.02, 0.13, 0.99, 0.92])
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nPaired calibration comparison:")
print(summary[["classifier", "metric", "traditional_point", "esm2_point", "observed_delta_esm2_minus_traditional", "ci_2_5", "ci_97_5", "interval_direction"]].round(6).to_string(index=False))
print("\nIntervals below zero:", below)
print("Intervals above zero:", above)
print("Intervals including zero:", includes)
print("Bootstrap delta rows:", len(deltas))
print("\nSTEP 72 COMPLETED SUCCESSFULLY")
print("=" * 112)

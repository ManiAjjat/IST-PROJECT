from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\postdoc-work\ist-project")
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

MODEL_SPECS = [
    ("Traditional Logistic Regression", "Traditional", "Logistic Regression", "step31_logistic_regression_test_predictions.csv"),
    ("Traditional RBF-SVM", "Traditional", "RBF-SVM", "step32_svm_test_predictions.csv"),
    ("Traditional Random Forest", "Traditional", "Random Forest", "step33_random_forest_test_predictions.csv"),
    ("Traditional XGBoost", "Traditional", "XGBoost", "step34_xgboost_test_predictions.csv"),
    ("ESM-2 Logistic Regression", "ESM-2", "Logistic Regression", "step48_esm2_logistic_regression_test_predictions.csv"),
    ("ESM-2 RBF-SVM", "ESM-2", "RBF-SVM", "step49_esm2_svm_test_predictions.csv"),
    ("ESM-2 Random Forest", "ESM-2", "Random Forest", "step50_esm2_random_forest_test_predictions.csv"),
    ("ESM-2 XGBoost", "ESM-2", "XGBoost", "step51_esm2_xgboost_test_predictions.csv"),
]

METRICS_OUT = RESULTS / "step71_calibration_metrics.csv"
BINS_OUT = RESULTS / "step71_calibration_bins.csv"
BOOTSTRAP_OUT = RESULTS / "step71_calibration_bootstrap.csv"
QC_OUT = RESULTS / "step71_calibration_qc.csv"
CURVES_PNG = FIGURES / "Step71_Model_Calibration_Curves.png"
CURVES_PDF = FIGURES / "Step71_Model_Calibration_Curves.pdf"
METRICS_PNG = FIGURES / "Step71_Calibration_Metrics.png"
METRICS_PDF = FIGURES / "Step71_Calibration_Metrics.pdf"

N_BOOTSTRAP = 2000
SEED = 2026
CLIP = 1e-15
BIN_EDGES = np.linspace(0.0, 1.0, 11)


def bin_indices(probability):
    # searchsorted assigns p=1.0 to index 10; clip closes the final bin.
    return np.clip(np.searchsorted(BIN_EDGES, probability, side="right") - 1, 0, 9)


def calibration_bins(y, probability, model, representation, classifier):
    assigned = bin_indices(probability)
    rows = []
    for index in range(10):
        mask = assigned == index
        if not mask.any():
            continue
        mean_probability = float(probability[mask].mean())
        observed_fraction = float(y[mask].mean())
        rows.append({
            "model": model,
            "representation": representation,
            "classifier": classifier,
            "bin": index + 1,
            "lower_bound": BIN_EDGES[index],
            "upper_bound": BIN_EDGES[index + 1],
            "interval": f"[{BIN_EDGES[index]:.1f},{BIN_EDGES[index + 1]:.1f}{']' if index == 9 else ')'}",
            "n": int(mask.sum()),
            "mean_predicted_probability": mean_probability,
            "observed_active_fraction": observed_fraction,
            "absolute_calibration_gap": abs(mean_probability - observed_fraction),
        })
    return pd.DataFrame(rows)


def ece_from_arrays(y, probability):
    assigned = bin_indices(probability)
    total = len(y)
    value = 0.0
    for index in range(10):
        mask = assigned == index
        if mask.any():
            value += mask.sum() / total * abs(probability[mask].mean() - y[mask].mean())
    return float(value)


def calibration_slope_intercept(y, probability):
    clipped = np.clip(probability, CLIP, 1.0 - CLIP)
    logits = np.log(clipped / (1.0 - clipped))
    def objective(parameters):
        linear = parameters[0] + parameters[1] * logits
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    def gradient(parameters):
        linear = parameters[0] + parameters[1] * logits
        fitted = np.empty_like(linear)
        positive = linear >= 0
        fitted[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exp_linear = np.exp(linear[~positive])
        fitted[~positive] = exp_linear / (1.0 + exp_linear)
        residual = fitted - y
        return np.array([residual.sum(), np.dot(residual, logits)])

    fit = minimize(objective, x0=np.array([0.0, 1.0]), jac=gradient, method="BFGS")
    return float(fit.x[0]), float(fit.x[1]), bool(fit.success)


print("=" * 104)
print("STEP 71 - PROBABILITY CALIBRATION AND RELIABILITY ANALYSIS")
print("=" * 104)

loaded = {}
reference = None
alignment_columns = ["ID", "sequence", "label", "binary_class"]
qc_rows = []
for model, representation, classifier, filename in MODEL_SPECS:
    frame = pd.read_csv(RESULTS / filename)
    required = alignment_columns + ["predicted_probability", "predicted_label"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{model}: missing columns {missing}")
    if len(frame) != 181 or frame.ID.nunique() != 181:
        raise ValueError(f"{model}: expected 181 unique test IDs")
    if reference is None:
        reference = frame[alignment_columns].copy()
    elif not frame[alignment_columns].reset_index(drop=True).equals(reference.reset_index(drop=True)):
        raise ValueError(f"{model}: alignment differs from reference")
    probability = frame.predicted_probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError(f"{model}: invalid probabilities")
    threshold_labels = (probability >= 0.5).astype(int)
    if not np.array_equal(threshold_labels, frame.predicted_label.to_numpy(dtype=int)):
        raise ValueError(f"{model}: saved labels do not equal probability >= 0.5")
    loaded[model] = (frame, representation, classifier, probability)

y = reference.label.to_numpy(dtype=int)
if int(y.sum()) != 20 or int((y == 0).sum()) != 161:
    raise ValueError("Expected 20 Active and 161 Inactive peptides")
prevalence = float(y.mean())
prevalence_brier = prevalence * (1.0 - prevalence)

# Generate the fixed stratified resampling indices once and reuse them for all models.
rng = np.random.default_rng(SEED)
active_indices = np.flatnonzero(y == 1)
inactive_indices = np.flatnonzero(y == 0)
bootstrap_indices = np.empty((N_BOOTSTRAP, len(y)), dtype=int)
for replicate in range(N_BOOTSTRAP):
    selected_active = rng.choice(active_indices, size=len(active_indices), replace=True)
    selected_inactive = rng.choice(inactive_indices, size=len(inactive_indices), replace=True)
    bootstrap_indices[replicate] = np.concatenate([selected_active, selected_inactive])

metrics_rows = []
bin_frames = []
bootstrap_rows = []
for model, representation, classifier, _ in MODEL_SPECS:
    frame, _, _, probability = loaded[model]
    bins = calibration_bins(y, probability, model, representation, classifier)
    bin_frames.append(bins)
    brier = float(brier_score_loss(y, probability))
    evaluated_log_loss = float(log_loss(y, np.clip(probability, CLIP, 1.0 - CLIP), labels=[0, 1]))
    ece = float((bins.n / len(y) * bins.absolute_calibration_gap).sum())
    mce = float(bins.absolute_calibration_gap.max())
    intercept, slope, converged = calibration_slope_intercept(y, probability)

    boot_brier = np.empty(N_BOOTSTRAP)
    boot_log_loss = np.empty(N_BOOTSTRAP)
    boot_ece = np.empty(N_BOOTSTRAP)
    for replicate, indices in enumerate(bootstrap_indices):
        y_boot = y[indices]
        p_boot = probability[indices]
        boot_brier[replicate] = np.mean((p_boot - y_boot) ** 2)
        p_clip = np.clip(p_boot, CLIP, 1.0 - CLIP)
        boot_log_loss[replicate] = -np.mean(y_boot * np.log(p_clip) + (1 - y_boot) * np.log(1 - p_clip))
        boot_ece[replicate] = ece_from_arrays(y_boot, p_boot)
        bootstrap_rows.append({
            "model": model, "representation": representation, "classifier": classifier,
            "replicate": replicate + 1, "seed": SEED,
            "brier_score": boot_brier[replicate], "log_loss": boot_log_loss[replicate],
            "ece_10_equal_width_bins": boot_ece[replicate],
            "active_n": len(active_indices), "inactive_n": len(inactive_indices),
        })
    brier_ci = np.percentile(boot_brier, [2.5, 97.5])
    log_loss_ci = np.percentile(boot_log_loss, [2.5, 97.5])
    ece_ci = np.percentile(boot_ece, [2.5, 97.5])
    metrics_rows.append({
        "model": model, "representation": representation, "classifier": classifier,
        "n": len(y), "active_n": int(y.sum()), "inactive_n": int((y == 0).sum()),
        "prevalence": prevalence, "prevalence_only_brier_benchmark": prevalence_brier,
        "mean_predicted_probability": probability.mean(),
        "brier_score": brier, "brier_ci_lower": brier_ci[0], "brier_ci_upper": brier_ci[1],
        "log_loss": evaluated_log_loss, "log_loss_ci_lower": log_loss_ci[0], "log_loss_ci_upper": log_loss_ci[1],
        "ece_10_equal_width_bins": ece, "ece_ci_lower": ece_ci[0], "ece_ci_upper": ece_ci[1],
        "mce_10_equal_width_bins": mce,
        "calibration_intercept": intercept, "calibration_slope": slope,
        "diagnostic_calibration_fit_converged": converged,
        "nonempty_calibration_bins": len(bins),
    })

metrics = pd.DataFrame(metrics_rows)
bins = pd.concat(bin_frames, ignore_index=True)
bootstrap = pd.DataFrame(bootstrap_rows)
metrics.to_csv(METRICS_OUT, index=False)
bins.to_csv(BINS_OUT, index=False)
bootstrap.to_csv(BOOTSTRAP_OUT, index=False)

for model, representation, classifier, _ in MODEL_SPECS:
    frame, _, _, probability = loaded[model]
    model_bins = bins.loc[bins.model == model]
    model_bootstrap = bootstrap.loc[bootstrap.model == model]
    weighted_prevalence = float((model_bins.n * model_bins.observed_active_fraction).sum() / len(y))
    weighted_probability = float((model_bins.n * model_bins.mean_predicted_probability).sum() / len(y))
    direct_brier = float(np.mean((probability - y) ** 2))
    saved_brier = float(metrics.loc[metrics.model == model, "brier_score"].iloc[0])
    saved_ece = float(metrics.loc[metrics.model == model, "ece_10_equal_width_bins"].iloc[0])
    bin_ece = float((model_bins.n / len(y) * model_bins.absolute_calibration_gap).sum())
    qc_rows.append({
        "model": model, "rows": len(frame), "unique_IDs": frame.ID.nunique(),
        "active_n": int(frame.label.sum()), "inactive_n": int((frame.label == 0).sum()),
        "alignment_exact": True, "probabilities_finite": np.isfinite(probability).all(),
        "probabilities_in_unit_interval": ((probability >= 0) & (probability <= 1)).all(),
        "saved_labels_equal_probability_ge_0_5": np.array_equal((probability >= 0.5).astype(int), frame.predicted_label),
        "predefined_bin_count": len(BIN_EDGES) - 1, "nonempty_bin_count": len(model_bins),
        "saved_bin_n_sum": int(model_bins.n.sum()),
        "weighted_observed_prevalence": weighted_prevalence,
        "prevalence_absolute_difference": abs(weighted_prevalence - prevalence),
        "weighted_mean_bin_probability": weighted_probability,
        "direct_mean_probability": float(probability.mean()),
        "mean_probability_absolute_difference": abs(weighted_probability - probability.mean()),
        "direct_brier_score": direct_brier, "saved_brier_score": saved_brier,
        "brier_absolute_difference": abs(direct_brier - saved_brier),
        "bin_reconstructed_ece": bin_ece, "saved_ece": saved_ece,
        "ece_absolute_difference": abs(bin_ece - saved_ece),
        "bootstrap_replicates": len(model_bootstrap),
        "bootstrap_active_n_every_replicate": bool((model_bootstrap.active_n == 20).all()),
        "bootstrap_inactive_n_every_replicate": bool((model_bootstrap.inactive_n == 161).all()),
        "stored_probability_max_absolute_change": 0.0,
        "recalibration_performed": False,
    })
qc = pd.DataFrame(qc_rows)
qc.to_csv(QC_OUT, index=False)

# Publication figure 1: reliability curves.
palette = {
    "Logistic Regression": "#0072B2", "RBF-SVM": "#D55E00",
    "Random Forest": "#009E73", "XGBoost": "#CC79A7",
}
markers = {"Logistic Regression": "o", "RBF-SVM": "s", "Random Forest": "^", "XGBoost": "D"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), constrained_layout=True)
for panel, (ax, representation) in enumerate(zip(axes, ["Traditional", "ESM-2"])):
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.3, label="Perfect calibration")
    for classifier in ["Logistic Regression", "RBF-SVM", "Random Forest", "XGBoost"]:
        part = bins.loc[(bins.representation == representation) & (bins.classifier == classifier)]
        ax.plot(part.mean_predicted_probability, part.observed_active_fraction,
                color=palette[classifier], marker=markers[classifier], linewidth=1.8,
                markersize=5.5, label=classifier)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal",
           xlabel="Mean predicted Active probability", ylabel="Observed Active fraction",
           title=representation + " models")
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.12, 1.06, chr(65 + panel), transform=ax.transAxes, fontsize=13, fontweight="bold")
axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Locked-test reliability curves (10 fixed equal-width bins)", fontsize=14, fontweight="bold")
fig.savefig(CURVES_PNG, dpi=420, facecolor="white")
fig.savefig(CURVES_PDF, facecolor="white")
plt.close(fig)

# Publication figure 2: point estimates with stratified bootstrap percentile intervals.
display = metrics.copy()
display["short_model"] = display.representation.str.replace("Traditional", "Trad.") + " " + display.classifier
display = display.iloc[::-1].reset_index(drop=True)
y_positions = np.arange(len(display))
fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.0), sharey=True, constrained_layout=True)
metric_specs = [
    ("brier_score", "brier_ci_lower", "brier_ci_upper", "Brier score", prevalence_brier),
    ("ece_10_equal_width_bins", "ece_ci_lower", "ece_ci_upper", "Expected calibration error", None),
    ("log_loss", "log_loss_ci_lower", "log_loss_ci_upper", "Log loss", None),
]
representation_colors = display.representation.map({"Traditional": "#4C78A8", "ESM-2": "#E45756"})
for panel, (ax, (value_col, low_col, high_col, title, benchmark)) in enumerate(zip(axes, metric_specs)):
    values = display[value_col].to_numpy()
    lower = display[low_col].to_numpy(); upper = display[high_col].to_numpy()
    for position, value, low, high, color in zip(y_positions, values, lower, upper, representation_colors):
        ax.errorbar(value, position, xerr=np.array([[value - low], [high - value]]),
                    fmt="none", ecolor=color, elinewidth=1.8, capsize=3)
    ax.scatter(values, y_positions, c=representation_colors, s=44, edgecolor="white", linewidth=0.6, zorder=3)
    if benchmark is not None:
        ax.axvline(benchmark, color="black", linestyle="--", linewidth=1.2, label=f"Prevalence-only: {benchmark:.3f}")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set(title=title, xlabel="Lower is better", yticks=y_positions, yticklabels=display.short_model)
    ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.15, 1.04, chr(65 + panel), transform=ax.transAxes, fontsize=13, fontweight="bold")
axes[0].set_ylabel("Frozen model")
fig.suptitle("Calibration metrics with 95% stratified-bootstrap intervals", fontsize=14, fontweight="bold")
fig.savefig(METRICS_PNG, dpi=420, facecolor="white")
fig.savefig(METRICS_PDF, facecolor="white")
plt.close(fig)

print(f"\nLocked-test prevalence: {prevalence:.6f}")
print(f"Prevalence-only Brier benchmark: {prevalence_brier:.6f}")
print("\nCalibration metrics:")
print(metrics[["model", "brier_score", "log_loss", "ece_10_equal_width_bins", "mce_10_equal_width_bins", "calibration_intercept", "calibration_slope"]].round(6).to_string(index=False))
print("\nOutput row counts:")
print("Metrics:", len(metrics), "Bins:", len(bins), "Bootstrap:", len(bootstrap), "QC:", len(qc))
print("\nSTEP 71 COMPLETED SUCCESSFULLY")
print("=" * 104)

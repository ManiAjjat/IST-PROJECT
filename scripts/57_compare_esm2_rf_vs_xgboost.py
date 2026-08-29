from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

SUMMARY_INPUT = RESULTS_DIR / "step74_model_performance_bootstrap_summary.csv"
REPLICATE_INPUT = RESULTS_DIR / "step74_model_performance_bootstrap_replicates.csv"
COMPARISON_OUTPUT = RESULTS_DIR / "step75_esm2_rf_vs_xgboost_paired_comparison.csv"
DELTA_OUTPUT = RESULTS_DIR / "step75_esm2_rf_vs_xgboost_bootstrap_deltas.csv"
QC_OUTPUT = RESULTS_DIR / "step75_esm2_rf_vs_xgboost_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step75_ESM2_RF_vs_XGBoost_Paired_Bootstrap.png"
FIGURE_PDF = FIGURES_DIR / "Step75_ESM2_RF_vs_XGBoost_Paired_Bootstrap.pdf"

RF_MODEL = "ESM-2 Random Forest"
XGB_MODEL = "ESM-2 XGBoost"
METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
EXPECTED_REPLICATES = 5000


print("=" * 100)
print("STEP 75 - PAIRED ESM-2 RANDOM FOREST VS XGBOOST COMPARISON")
print("=" * 100)

for path in (SUMMARY_INPUT, REPLICATE_INPUT):
    if not path.exists():
        raise FileNotFoundError(f"Missing Step-74 input: {path}")

summary_df = pd.read_csv(SUMMARY_INPUT)
replicate_df = pd.read_csv(REPLICATE_INPUT)

if summary_df.shape[0] != 8 or replicate_df.shape[0] != 40000:
    raise ValueError("Unexpected Step-74 input dimensions.")

required_summary = {"model", *METRICS}
required_replicates = {
    "model", "replicate", "sample_index_sha256", "replicate_n",
    "active_n", "inactive_n", *METRICS,
}
if not required_summary.issubset(summary_df.columns):
    raise ValueError("Step-74 summary is missing required columns.")
if not required_replicates.issubset(replicate_df.columns):
    raise ValueError("Step-74 replicate table is missing required columns.")

if not {RF_MODEL, XGB_MODEL}.issubset(set(summary_df["model"])):
    raise ValueError("One or both target models are absent from the summary.")
if not {RF_MODEL, XGB_MODEL}.issubset(set(replicate_df["model"])):
    raise ValueError("One or both target models are absent from the replicates.")

rf_summary = summary_df.loc[summary_df["model"] == RF_MODEL].iloc[0]
xgb_summary = summary_df.loc[summary_df["model"] == XGB_MODEL].iloc[0]
rf_boot = (
    replicate_df.loc[replicate_df["model"] == RF_MODEL]
    .sort_values("replicate").reset_index(drop=True)
)
xgb_boot = (
    replicate_df.loc[replicate_df["model"] == XGB_MODEL]
    .sort_values("replicate").reset_index(drop=True)
)

expected_ids = np.arange(1, EXPECTED_REPLICATES + 1)
if len(rf_boot) != EXPECTED_REPLICATES or len(xgb_boot) != EXPECTED_REPLICATES:
    raise ValueError("Each target model must have exactly 5,000 replicates.")
if not np.array_equal(rf_boot["replicate"].to_numpy(), expected_ids):
    raise ValueError("Random Forest replicate IDs are not exactly 1-5000.")
if not np.array_equal(xgb_boot["replicate"].to_numpy(), expected_ids):
    raise ValueError("XGBoost replicate IDs are not exactly 1-5000.")
if not np.array_equal(
    rf_boot["sample_index_sha256"].to_numpy(),
    xgb_boot["sample_index_sha256"].to_numpy(),
):
    raise ValueError("RF and XGBoost sample-index hashes are not paired.")

for frame in (rf_boot, xgb_boot):
    if not (frame["replicate_n"] == 181).all():
        raise ValueError("A replicate does not contain 181 observations.")
    if not (frame["active_n"] == 20).all() or not (frame["inactive_n"] == 161).all():
        raise ValueError("A replicate does not preserve the locked-test class counts.")
    if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite bootstrap metric encountered.")

comparison_rows = []
delta_frames = []
for metric in METRICS:
    rf_point = float(rf_summary[metric])
    xgb_point = float(xgb_summary[metric])
    rf_values = rf_boot[metric].to_numpy(dtype=float)
    xgb_values = xgb_boot[metric].to_numpy(dtype=float)
    deltas = xgb_values - rf_values
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    if lower > 0:
        status = "favors_XGBoost"
    elif upper < 0:
        status = "favors_Random_Forest"
    else:
        status = "includes_zero"

    less_equal_zero = float(np.mean(deltas <= 0))
    greater_equal_zero = float(np.mean(deltas >= 0))
    comparison_rows.append({
        "metric": metric,
        "random_forest_point": rf_point,
        "xgboost_point": xgb_point,
        "observed_delta_xgboost_minus_rf": xgb_point - rf_point,
        "bootstrap_mean_delta": float(np.mean(deltas)),
        "bootstrap_se_delta": float(np.std(deltas, ddof=1)),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "ci_excludes_zero": bool(status != "includes_zero"),
        "interval_status": status,
        "replicates": EXPECTED_REPLICATES,
        "bootstrap_fraction_xgboost_greater": float(np.mean(deltas > 0)),
        "bootstrap_fraction_rf_greater": float(np.mean(deltas < 0)),
        "bootstrap_fraction_equal": float(np.mean(deltas == 0)),
        "descriptive_two_sided_tail_probability": min(
            1.0, 2.0 * min(less_equal_zero, greater_equal_zero)
        ),
        "delta_definition": "XGBoost_minus_RandomForest",
    })
    delta_frames.append(pd.DataFrame({
        "metric": metric,
        "replicate": expected_ids,
        "sample_index_sha256": rf_boot["sample_index_sha256"].to_numpy(),
        "random_forest_value": rf_values,
        "xgboost_value": xgb_values,
        "delta_xgboost_minus_rf": deltas,
    }))

comparison_df = pd.DataFrame(comparison_rows)
delta_df = pd.concat(delta_frames, ignore_index=True)
if comparison_df.shape[0] != 4 or delta_df.shape[0] != 20000:
    raise ValueError("Unexpected Step-75 output dimensions.")
if delta_df[["metric", "replicate"]].duplicated().any():
    raise ValueError("Duplicate metric-replicate key.")

point_identity_error = float(np.max(np.abs(
    comparison_df["observed_delta_xgboost_minus_rf"]
    - (comparison_df["xgboost_point"] - comparison_df["random_forest_point"])
)))
replicate_identity_error = float(np.max(np.abs(
    delta_df["delta_xgboost_minus_rf"]
    - (delta_df["xgboost_value"] - delta_df["random_forest_value"])
)))

comparison_df.to_csv(COMPARISON_OUTPUT, index=False)
delta_df.to_csv(DELTA_OUTPUT, index=False)

plot_df = comparison_df.set_index("metric").loc[METRICS].reset_index()
y = np.arange(len(plot_df))
observed = plot_df["observed_delta_xgboost_minus_rf"].to_numpy()
lower = plot_df["ci_2_5"].to_numpy()
upper = plot_df["ci_97_5"].to_numpy()

fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor="white")
ax.set_facecolor("white")
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
for i, row in plot_df.iterrows():
    supported = bool(row["ci_excludes_zero"])
    color = "#C44E52" if supported else "#4C78A8"
    marker = "D" if supported else "o"
    ax.errorbar(
        observed[i], y[i],
        xerr=[[observed[i] - lower[i]], [upper[i] - observed[i]]],
        fmt=marker, color=color, ecolor=color, markeredgecolor="white",
        capsize=5, markersize=9, linewidth=1.7,
    )
    ax.text(upper[i], y[i], f"  {observed[i]:+.4f}", va="center", fontsize=9)
ax.set_yticks(y, METRICS)
ax.invert_yaxis()
ax.set_xlabel("Paired difference: ESM-2 XGBoost - ESM-2 Random Forest")
ax.set_title("Paired Bootstrap Comparison of the Two Leading ESM-2 Models", pad=12)
ax.grid(axis="x", alpha=0.20)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, 0.050, "Positive values favor XGBoost; negative values favor Random Forest.", ha="center", fontsize=9)
fig.text(
    0.5, 0.018,
    "Diamonds: 95% paired percentile interval excludes zero; circles: interval includes zero. "
    "No multiplicity-adjusted significance claim is made.",
    ha="center", fontsize=8.2,
)
fig.tight_layout(rect=[0.04, 0.09, 0.99, 0.96])
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

above = int((comparison_df["ci_2_5"] > 0).sum())
below = int((comparison_df["ci_97_5"] < 0).sum())
including = int(4 - above - below)
qc_df = pd.DataFrame([{
    "random_forest_model": RF_MODEL,
    "xgboost_model": XGB_MODEL,
    "metrics": 4,
    "bootstrap_replicates": EXPECTED_REPLICATES,
    "expected_summary_rows": 4,
    "observed_summary_rows": len(comparison_df),
    "expected_delta_rows": 20000,
    "observed_delta_rows": len(delta_df),
    "replicate_ids_exact_1_to_5000": True,
    "sample_index_hash_available": True,
    "sample_index_hash_pairing_verified": True,
    "shared_bootstrap_sampling": True,
    "delta_definition": "XGBoost_minus_RandomForest",
    "intervals_entirely_above_zero": above,
    "intervals_entirely_below_zero": below,
    "intervals_including_zero": including,
    "maximum_point_delta_identity_error": point_identity_error,
    "maximum_replicate_delta_identity_error": replicate_identity_error,
    "new_bootstrap_sampling_performed": False,
    "models_loaded_for_training": False,
    "models_retrained": False,
    "model_selection_performed": False,
    "threshold_optimized": False,
    "probabilities_changed": False,
    "multiplicity_adjusted_inference_claimed": False,
}])
qc_df.to_csv(QC_OUTPUT, index=False)

print("\nPaired results:")
print(comparison_df[[
    "metric", "random_forest_point", "xgboost_point",
    "observed_delta_xgboost_minus_rf", "ci_2_5", "ci_97_5", "interval_status",
]].round(6).to_string(index=False))
print(f"\nIntervals favoring XGBoost: {above}")
print(f"Intervals favoring Random Forest: {below}")
print(f"Intervals including zero: {including}")
print("\nSTEP 75 COMPLETED SUCCESSFULLY")
print("=" * 100)

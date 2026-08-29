from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

PERFORMANCE_INPUT = RESULTS_DIR / "step74_model_performance_bootstrap_summary.csv"
CALIBRATION_INPUT = RESULTS_DIR / "step71_calibration_metrics.csv"
UTILITY_INPUT = RESULTS_DIR / "step73_decision_curve_model_summary.csv"
ESM2_COMPARISON_INPUT = RESULTS_DIR / "step52_esm2_model_comparison.csv"
PAIRED_INPUT = RESULTS_DIR / "step75_esm2_rf_vs_xgboost_paired_comparison.csv"

INTEGRATED_OUTPUT = RESULTS_DIR / "step76_integrated_model_summary.csv"
RANK_OUTPUT = RESULTS_DIR / "step76_domain_rank_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step76_integrated_model_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step76_Integrated_Model_Performance_Map.png"
FIGURE_PDF = FIGURES_DIR / "Step76_Integrated_Model_Performance_Map.pdf"

MODEL_ORDER = [
    "Traditional Logistic Regression", "Traditional RBF-SVM",
    "Traditional Random Forest", "Traditional XGBoost",
    "ESM-2 Logistic Regression", "ESM-2 RBF-SVM",
    "ESM-2 Random Forest", "ESM-2 XGBoost",
]
PERFORMANCE_METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
CALIBRATION_RENAME = {
    "brier_score": "Brier",
    "ece_10_equal_width_bins": "ECE",
    "log_loss": "Log_loss",
}
UTILITY_RENAME = {
    "mean_net_benefit_0_05_to_0_20": "NB_0_05_to_0_20",
    "mean_net_benefit_0_05_to_0_50": "NB_0_05_to_0_50",
}


print("=" * 104)
print("STEP 76 - INTEGRATED MODEL STABILITY AND MULTI-CRITERIA RANKING SUMMARY")
print("=" * 104)

for path in (
    PERFORMANCE_INPUT, CALIBRATION_INPUT, UTILITY_INPUT,
    ESM2_COMPARISON_INPUT, PAIRED_INPUT,
):
    if not path.exists():
        raise FileNotFoundError(f"Missing required frozen input: {path}")

perf = pd.read_csv(PERFORMANCE_INPUT)
cal = pd.read_csv(CALIBRATION_INPUT)
utility = pd.read_csv(UTILITY_INPUT)
esm2 = pd.read_csv(ESM2_COMPARISON_INPUT)
paired = pd.read_csv(PAIRED_INPUT)

if perf.shape[0] != 8 or cal.shape[0] != 8 or utility.shape[0] != 8:
    raise ValueError("Expected eight model rows in Steps 71, 73, and 74.")
if esm2.shape[0] != 4 or paired.shape[0] != 4:
    raise ValueError("Expected four rows in the Step-52 and Step-75 checks.")

for frame, name in ((perf, "Step 74"), (cal, "Step 71"), (utility, "Step 73")):
    if frame["model"].duplicated().any() or set(frame["model"]) != set(MODEL_ORDER):
        raise ValueError(f"{name} model identities are not the expected eight models.")

perf_cols = ["model", "representation", "classifier", "test_n", "active_n", "inactive_n"]
for metric in PERFORMANCE_METRICS:
    perf_cols.extend([metric, f"{metric}_CI_low", f"{metric}_CI_high"])

integrated = perf[perf_cols].copy()
integrated = integrated.merge(
    cal[["model", *CALIBRATION_RENAME]].rename(columns=CALIBRATION_RENAME),
    on="model", how="left", validate="one_to_one",
)
integrated = integrated.merge(
    utility[["model", *UTILITY_RENAME]].rename(columns=UTILITY_RENAME),
    on="model", how="left", validate="one_to_one",
)

numeric_metrics = [
    "AUROC", "AUPRC", "MCC", "F1", "Brier", "ECE", "Log_loss",
    "NB_0_05_to_0_20", "NB_0_05_to_0_50",
]
if integrated[numeric_metrics].isna().any().any():
    raise ValueError("The integrated table contains missing metric values.")
if not np.isfinite(integrated[numeric_metrics].to_numpy(dtype=float)).all():
    raise ValueError("The integrated table contains non-finite metric values.")

# Average ranks retain ties. Lower rank is always better.
integrated["AUROC_rank"] = integrated["AUROC"].rank(ascending=False, method="average")
integrated["AUPRC_rank"] = integrated["AUPRC"].rank(ascending=False, method="average")
integrated["MCC_rank"] = integrated["MCC"].rank(ascending=False, method="average")
integrated["F1_rank"] = integrated["F1"].rank(ascending=False, method="average")
integrated["Brier_rank"] = integrated["Brier"].rank(ascending=True, method="average")
integrated["ECE_rank"] = integrated["ECE"].rank(ascending=True, method="average")
integrated["Log_loss_rank"] = integrated["Log_loss"].rank(ascending=True, method="average")
integrated["NB_0_05_to_0_20_rank"] = integrated["NB_0_05_to_0_20"].rank(ascending=False, method="average")
integrated["NB_0_05_to_0_50_rank"] = integrated["NB_0_05_to_0_50"].rank(ascending=False, method="average")

integrated["discrimination_rank"] = integrated[["AUROC_rank", "AUPRC_rank"]].mean(axis=1)
integrated["threshold_rank"] = integrated[["MCC_rank", "F1_rank"]].mean(axis=1)
integrated["calibration_rank"] = integrated[["Brier_rank", "ECE_rank", "Log_loss_rank"]].mean(axis=1)
integrated["decision_utility_rank"] = integrated[[
    "NB_0_05_to_0_20_rank", "NB_0_05_to_0_50_rank",
]].mean(axis=1)
integrated["mean_domain_rank"] = integrated[[
    "discrimination_rank", "threshold_rank", "calibration_rank", "decision_utility_rank",
]].mean(axis=1)
integrated["summary_order_only_not_model_selection"] = True

integrated = integrated.sort_values(
    ["mean_domain_rank", "model"], ascending=[True, True]
).reset_index(drop=True)
integrated.insert(0, "descriptive_summary_order", np.arange(1, 9))

rank_cols = [
    "descriptive_summary_order", "model", "representation", "classifier",
    "discrimination_rank", "threshold_rank", "calibration_rank",
    "decision_utility_rank", "mean_domain_rank",
    "summary_order_only_not_model_selection",
]
rank_summary = integrated[rank_cols].copy()

# Cross-source identity checks use only frozen values and add no ranking input.
esm2_names = {f"ESM-2 {m}" for m in esm2["model"].astype(str)}
esm2_identity_exact = esm2_names == set(MODEL_ORDER[4:])
esm2_max_error = 0.0
for _, row in esm2.iterrows():
    model = f"ESM-2 {row['model']}"
    target = integrated.loc[integrated["model"] == model].iloc[0]
    for metric in PERFORMANCE_METRICS:
        esm2_max_error = max(esm2_max_error, abs(float(row[metric]) - float(target[metric])))

paired_models_present = {
    "ESM-2 Random Forest", "ESM-2 XGBoost"
}.issubset(set(integrated["model"]))
paired_delta_max_error = 0.0
for _, row in paired.iterrows():
    metric = row["metric"]
    xgb = float(integrated.loc[integrated["model"] == "ESM-2 XGBoost", metric].iloc[0])
    rf = float(integrated.loc[integrated["model"] == "ESM-2 Random Forest", metric].iloc[0])
    paired_delta_max_error = max(
        paired_delta_max_error,
        abs((xgb - rf) - float(row["observed_delta_xgboost_minus_rf"])),
    )

integrated.to_csv(INTEGRATED_OUTPUT, index=False)
rank_summary.to_csv(RANK_OUTPUT, index=False)

# Visualization-only standardized desirability; it is not saved as a ranking input.
display_metrics = [
    "AUROC", "AUPRC", "MCC", "F1", "Brier", "ECE", "Log_loss",
    "NB_0_05_to_0_20", "NB_0_05_to_0_50",
]
display_labels = [
    "AUROC", "AUPRC", "MCC", "F1", "Brier", "ECE", "Log loss",
    "NB 0.05-0.20", "NB 0.05-0.50",
]
lower_better = {"Brier", "ECE", "Log_loss"}
z = np.empty((8, len(display_metrics)), dtype=float)
for j, metric in enumerate(display_metrics):
    values = integrated[metric].to_numpy(dtype=float)
    sd = values.std(ddof=0)
    if sd == 0:
        standardized = np.zeros_like(values)
    else:
        standardized = (values - values.mean()) / sd
    if metric in lower_better:
        standardized = -standardized
    z[:, j] = standardized

domain_cols = ["discrimination_rank", "threshold_rank", "calibration_rank", "decision_utility_rank"]
domain_labels = ["Discrimination", "Threshold", "Calibration", "Decision utility"]
rank_matrix = integrated[domain_cols].to_numpy(dtype=float)

fig = plt.figure(figsize=(17.2, 8.2), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
for ax in (ax1, ax2):
    ax.set_facecolor("white")

limit = max(2.0, float(np.max(np.abs(z))))
im1 = ax1.imshow(z, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
ax1.set_xticks(np.arange(len(display_labels)), display_labels, rotation=42, ha="right")
ax1.set_yticks(np.arange(8), integrated["model"])
ax1.set_title("A  Standardized metric desirability", loc="left", fontweight="bold", pad=12)
for i in range(8):
    for j, metric in enumerate(display_metrics):
        value = integrated.loc[i, metric]
        fmt = ".3f" if metric not in {"AUROC", "AUPRC", "MCC", "F1"} else ".3f"
        color = "white" if abs(z[i, j]) > 1.05 else "black"
        ax1.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7.5, color=color)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.02)
cbar1.set_label("Visualization-only standardized desirability\n(higher = better)")

im2 = ax2.imshow(rank_matrix, cmap="YlGn_r", vmin=1, vmax=8, aspect="auto")
ax2.set_xticks(np.arange(4), domain_labels, rotation=35, ha="right")
ax2.set_yticks(np.arange(8), [""] * 8)
ax2.set_title("B  Domain ranks", loc="left", fontweight="bold", pad=12)
for i in range(8):
    for j in range(4):
        val = rank_matrix[i, j]
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                 color="white" if val >= 6.0 else "black")
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.05, pad=0.03)
cbar2.set_label("Domain rank (lower = better)")

for ax in (ax1, ax2):
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("Integrated Performance Profile of Eight Frozen Classifiers", fontsize=18, fontweight="bold", y=0.985)
fig.text(
    0.5, 0.015,
    "Rows follow descriptive mean domain rank. Standardized colors are visual only; no super-score, new selection, or inference is performed.",
    ha="center", fontsize=10,
)
fig.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.90)
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

rank_recalc = integrated[[
    "discrimination_rank", "threshold_rank", "calibration_rank", "decision_utility_rank",
]].mean(axis=1)
qc = pd.DataFrame([{
    "models": 8,
    "traditional_models": 4,
    "esm2_models": 4,
    "integrated_rows": len(integrated),
    "rank_summary_rows": len(rank_summary),
    "performance_metrics_with_step74_intervals": 4,
    "calibration_metrics_without_new_intervals": 3,
    "decision_utility_metrics_without_new_intervals": 2,
    "all_source_model_alignments_exact": True,
    "all_integrated_metrics_finite": True,
    "step52_esm2_model_identities_exact": esm2_identity_exact,
    "maximum_step52_performance_value_error": esm2_max_error,
    "step75_paired_models_present": paired_models_present,
    "maximum_step75_observed_delta_error": paired_delta_max_error,
    "maximum_mean_domain_rank_identity_error": float(np.max(np.abs(rank_recalc - integrated["mean_domain_rank"]))),
    "average_rank_method_for_ties": True,
    "arbitrary_weighted_super_score_created": False,
    "visual_standardization_used_for_ranking": False,
    "models_trained_or_retrained": False,
    "new_resampling_performed": False,
    "new_thresholds_calculated": False,
    "primary_model_selected": False,
    "hypothesis_testing_performed": False,
    "ranks_interpreted_as_significance": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nDomain-rank summary:")
print(rank_summary.round(4).to_string(index=False))
print("\nSTEP 76 COMPLETED SUCCESSFULLY")
print("=" * 104)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
INPUT_FILE = RESULTS_DIR / "step54_paired_bootstrap_summary.csv"
PLOT_DATA_OUTPUT = RESULTS_DIR / "step55_bootstrap_ci_plot_data.csv"
PNG_OUTPUT = FIGURES_DIR / "Step55_Traditional_vs_ESM2_Bootstrap_CI.png"
PDF_OUTPUT = FIGURES_DIR / "Step55_Traditional_vs_ESM2_Bootstrap_CI.pdf"

CLASSIFIER_ORDER = ["Logistic Regression", "RBF-SVM", "Random Forest", "XGBoost"]
METRIC_ORDER = ["AUROC", "AUPRC", "MCC", "F1"]
PANEL_LABELS = ["A", "B", "C", "D"]
COLORS = {
    "Logistic Regression": "#0072B2",
    "RBF-SVM": "#E69F00",
    "Random Forest": "#009E73",
    "XGBoost": "#D55E00",
}


print("=" * 104)
print("STEP 55 - PUBLICATION-QUALITY BOOTSTRAP CONFIDENCE-INTERVAL FIGURE")
print("=" * 104)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
required = {
    "classifier", "metric", "observed_delta", "bootstrap_mean_delta",
    "bootstrap_standard_error", "ci_95_lower", "ci_95_upper",
    "ci_excludes_zero", "ci_relation_to_zero", "bootstrap_replicates",
}
assert required.issubset(df.columns)
assert len(df) == 16
assert set(df["classifier"]) == set(CLASSIFIER_ORDER)
assert set(df["metric"]) == set(METRIC_ORDER)
assert not df.duplicated(["classifier", "metric"]).any()
assert df["bootstrap_replicates"].eq(5000).all()
assert np.isfinite(df[[
    "observed_delta", "bootstrap_mean_delta", "bootstrap_standard_error",
    "ci_95_lower", "ci_95_upper",
]].to_numpy()).all()
assert (df["ci_95_lower"] <= df["observed_delta"]).all()
assert (df["observed_delta"] <= df["ci_95_upper"]).all()

plot_df = df.copy()
plot_df["classifier"] = pd.Categorical(
    plot_df["classifier"], categories=CLASSIFIER_ORDER, ordered=True
)
plot_df["metric"] = pd.Categorical(
    plot_df["metric"], categories=METRIC_ORDER, ordered=True
)
plot_df = plot_df.sort_values(["metric", "classifier"]).reset_index(drop=True)
plot_df = plot_df.rename(columns={
    "ci_95_lower": "percentile_95ci_lower",
    "ci_95_upper": "percentile_95ci_upper",
})
plot_df["marker"] = np.where(plot_df["ci_excludes_zero"], "diamond", "circle")
plot_df["direction"] = np.select(
    [plot_df["percentile_95ci_lower"] > 0, plot_df["percentile_95ci_upper"] < 0],
    ["favors_ESM2", "favors_traditional"],
    default="includes_zero",
)
plot_df.to_csv(PLOT_DATA_OUTPUT, index=False)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
fig.suptitle(
    "Paired-bootstrap differences: ESM-2 versus traditional descriptors",
    fontsize=14,
    y=1.025,
)

for axis, metric, panel_label in zip(axes.flat, METRIC_ORDER, PANEL_LABELS):
    panel = plot_df.loc[plot_df["metric"] == metric].copy()
    panel["classifier"] = panel["classifier"].astype("object")
    panel = panel.set_index("classifier").loc[CLASSIFIER_ORDER].reset_index()
    y_positions = np.arange(len(panel))[::-1]

    maximum = float(np.max(np.abs(panel[[
        "percentile_95ci_lower", "percentile_95ci_upper"
    ]].to_numpy())))
    axis_limit = maximum * 1.18
    axis.axvspan(-axis_limit, 0, color="#F2F2F2", zorder=0)
    axis.axvline(0, color="#333333", linewidth=1.15, linestyle="--", zorder=1)

    for y, row in zip(y_positions, panel.itertuples(index=False)):
        lower_error = row.observed_delta - row.percentile_95ci_lower
        upper_error = row.percentile_95ci_upper - row.observed_delta
        marker = "D" if row.ci_excludes_zero else "o"
        marker_size = 7.0 if row.ci_excludes_zero else 6.0
        axis.errorbar(
            row.observed_delta,
            y,
            xerr=np.array([[lower_error], [upper_error]]),
            fmt=marker,
            markersize=marker_size,
            markerfacecolor=COLORS[row.classifier],
            markeredgecolor="#222222",
            markeredgewidth=0.75,
            ecolor=COLORS[row.classifier],
            elinewidth=1.8,
            capsize=3.5,
            capthick=1.2,
            zorder=3,
        )

    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-0.65, len(panel) - 0.35)
    axis.set_yticks(y_positions, panel["classifier"])
    axis.set_xlabel(f"Delta {metric} (ESM-2 - traditional)")
    axis.set_title(f"{panel_label}   {metric}", loc="left", fontweight="bold", pad=8)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.text(
        0.02, -0.23, "Favors traditional", transform=axis.transAxes,
        ha="left", va="top", fontsize=7.5, color="#555555",
    )
    axis.text(
        0.98, -0.23, "Favors ESM-2", transform=axis.transAxes,
        ha="right", va="top", fontsize=7.5, color="#555555",
    )

fig.text(
    0.5, -0.025,
    "Points show observed paired differences; horizontal bars show 95% percentile bootstrap CIs "
    "(5,000 class-stratified replicates). Diamonds denote intervals excluding zero.",
    ha="center", va="top", fontsize=8.5,
)

fig.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

positive_supported = int((plot_df["percentile_95ci_lower"] > 0).sum())
negative_supported = int((plot_df["percentile_95ci_upper"] < 0).sum())
zero_including = int((~plot_df["ci_excludes_zero"]).sum())

assert positive_supported == 4
assert negative_supported == 0
assert zero_including == 12
assert PNG_OUTPUT.exists() and PDF_OUTPUT.exists() and PLOT_DATA_OUTPUT.exists()

print("Rows plotted:", len(plot_df))
print("Intervals entirely above zero:", positive_supported)
print("Intervals entirely below zero:", negative_supported)
print("Intervals including zero:", zero_including)
print("\nPlot-data table:")
print(PLOT_DATA_OUTPUT)
print("\nBootstrap CI PNG:")
print(PNG_OUTPUT)
print("\nBootstrap CI PDF:")
print(PDF_OUTPUT)
print("\nSTEP 55 COMPLETED SUCCESSFULLY")
print("=" * 104)

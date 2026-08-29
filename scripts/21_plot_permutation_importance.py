from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
INPUT_FILE = RESULTS_DIR / "step38_xgboost_permutation_importance.csv"
PLOT_DATA_OUTPUT = RESULTS_DIR / "step39_permutation_importance_plot_data.csv"
PNG_OUTPUT = FIGURE_DIR / "Step39_XGBoost_Permutation_Importance.png"
PDF_OUTPUT = FIGURE_DIR / "Step39_XGBoost_Permutation_Importance.pdf"
TOP_N = 15

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Step 38 summary not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
required_columns = {"rank", "feature", "mean_cv_AUROC_drop", "sd_cv_AUROC_drop"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Step 38 summary is missing columns: {sorted(missing_columns)}")
if len(df) != 32:
    raise ValueError(f"Expected 32 Step 38 features, found {len(df)}.")
if df["feature"].duplicated().any():
    raise ValueError("Step 38 summary contains duplicate feature names.")
if df[list(required_columns - {"feature"})].isna().any().any():
    raise ValueError("Step 38 ranking contains missing numeric values.")
if (df["sd_cv_AUROC_drop"] < 0).any():
    raise ValueError("Between-fold standard deviations cannot be negative.")

plot_df = (
    df.sort_values(["mean_cv_AUROC_drop", "feature"], ascending=[False, True])
    .head(TOP_N)
    .copy()
)
plot_df["rank"] = np.arange(1, TOP_N + 1)
plot_df = plot_df.rename(
    columns={"sd_cv_AUROC_drop": "sd_between_folds_AUROC_drop"}
)[["rank", "feature", "mean_cv_AUROC_drop", "sd_between_folds_AUROC_drop"]]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
plot_df.to_csv(PLOT_DATA_OUTPUT, index=False)

# Reverse the display order so the highest-ranked feature appears at the top.
display_df = plot_df.iloc[::-1].reset_index(drop=True)
y_positions = np.arange(len(display_df))
means = display_df["mean_cv_AUROC_drop"].to_numpy(dtype=float)
errors = display_df["sd_between_folds_AUROC_drop"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(10.5, 8.2))
bars = ax.barh(
    y_positions,
    means,
    xerr=errors,
    height=0.68,
    color="#4C78A8",
    edgecolor="white",
    linewidth=0.7,
    error_kw={"ecolor": "#333333", "elinewidth": 1.15, "capsize": 3.5, "capthick": 1.15},
)

ax.set_yticks(y_positions)
ax.set_yticklabels(display_df["feature"])
ax.set_xlabel("Mean decrease in validation AUROC after permutation")
ax.set_ylabel("Traditional feature")
ax.set_title("Cross-validated XGBoost permutation importance", fontsize=15, pad=14)
ax.axvline(0, color="#333333", linewidth=0.9)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

left_limit = min(-0.001, float(np.min(means - errors)) - 0.0007)
right_limit = float(np.max(means + errors)) + 0.0032
ax.set_xlim(left_limit, right_limit)
annotation_offset = (right_limit - left_limit) * 0.012
for bar, mean, error in zip(bars, means, errors):
    ax.text(
        mean + error + annotation_offset,
        bar.get_y() + bar.get_height() / 2,
        f"{mean:.4f}",
        va="center",
        ha="left",
        fontsize=8.5,
    )

ax.text(
    0.99,
    0.015,
    "Error bars: SD across 5 fixed development folds",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    color="#444444",
)
fig.tight_layout()
fig.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\n39Q. Output checks:")
print("PNG exists:", PNG_OUTPUT.exists())
print("PDF exists:", PDF_OUTPUT.exists())
print("Plot-data CSV exists:", PLOT_DATA_OUTPUT.exists())

print("\n39R. Top 5 plotted features:")
print(
    plot_df[
        ["rank", "feature", "mean_cv_AUROC_drop", "sd_between_folds_AUROC_drop"]
    ]
    .head(5)
    .round(6)
    .to_string(index=False)
)

print("\n" + "=" * 88)
print("STEP 39 SUMMARY")
print("=" * 88)
print("Total Step-38 features:", len(df))
print("Features plotted:", len(plot_df))
print("Top feature:", plot_df.iloc[0]["feature"])
print("Top mean AUROC drop:", round(float(plot_df.iloc[0]["mean_cv_AUROC_drop"]), 6))
print("\nPNG:", PNG_OUTPUT)
print("PDF:", PDF_OUTPUT)
print("Plot data:", PLOT_DATA_OUTPUT)
print("\nSTEP 39 COMPLETED SUCCESSFULLY")
print("=" * 88)

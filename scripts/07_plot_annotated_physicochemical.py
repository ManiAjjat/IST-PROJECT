from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
STATISTICS_FILE = PROJECT_DIR / "results" / "step24_physicochemical_statistics.csv"
FIGURE_DIR = PROJECT_DIR / "figures"
PNG_OUTPUT = FIGURE_DIR / "Step25_Physicochemical_Distributions_Annotated.png"
PDF_OUTPUT = FIGURE_DIR / "Step25_Physicochemical_Distributions_Annotated.pdf"

df = pd.read_csv(FEATURE_FILE)
stats = pd.read_csv(STATISTICS_FILE)
features = [
    "length",
    "molecular_weight",
    "net_charge_pH7_4",
    "isoelectric_point",
    "mean_eisenberg_hydrophobicity",
    "hydrophobic_moment",
    "boman_index",
]
titles = [
    "Peptide length",
    "Molecular weight (Da)",
    "Net charge at pH 7.4",
    "Isoelectric point",
    "Mean Eisenberg hydrophobicity",
    "Hydrophobic moment",
    "Boman index",
]
stat_lookup = stats.set_index("feature").to_dict(orient="index")
if set(features) != set(stat_lookup):
    raise ValueError("Step 24 statistics do not match the seven Step 23 features.")

plot_values = df[features].apply(pd.to_numeric, errors="coerce")
missing_values = int(plot_values.isna().sum().sum())
nonfinite_values = int((~np.isfinite(plot_values.to_numpy())).sum())
if missing_values != 0 or nonfinite_values != 0:
    raise ValueError("Plotting descriptors contain missing or non-finite values.")

active = df.loc[df["binary_class"] == "Active"]
inactive = df.loc[df["binary_class"] == "Inactive"]
active_n = len(active)
inactive_n = len(inactive)
colors = ["#4C78A8", "#E45756"]

figure, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
axes = axes.ravel()
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

for index, (feature, title) in enumerate(zip(features, titles)):
    axis = axes[index]
    values = [inactive[feature].to_numpy(), active[feature].to_numpy()]
    violin = axis.violinplot(
        values,
        positions=[1, 2],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.65)

    box = axis.boxplot(
        values,
        positions=[1, 2],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    lower = min(np.min(values[0]), np.min(values[1]))
    upper = max(np.max(values[0]), np.max(values[1]))
    spread = upper - lower or 1.0
    bracket_y = upper + 0.08 * spread
    bracket_height = 0.025 * spread
    axis.plot(
        [1, 1, 2, 2],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="black",
        linewidth=1.0,
    )
    stat = stat_lookup[feature]
    annotation = f"{stat['significance_symbol']}\nr_rb = {float(stat['rank_biserial_effect']):.2f}"
    axis.text(1.5, bracket_y + bracket_height, annotation, ha="center", va="bottom")
    axis.set_ylim(lower - 0.04 * spread, bracket_y + 0.18 * spread)
    axis.set_title(title)
    axis.set_xticks([1, 2], [f"Inactive\n(n={inactive_n})", f"Active\n(n={active_n})"])
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

axes[-1].axis("off")
figure.suptitle(
    "Physicochemical distributions of lung-cancer anticancer peptides",
    fontsize=15,
)
figure.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
figure.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(figure)

print("\n25L. Output checks:")
print("PNG exists:", PNG_OUTPUT.exists())
print("PDF exists:", PDF_OUTPUT.exists())
print("\nPNG:")
print(PNG_OUTPUT)
print("\nPDF:")
print(PDF_OUTPUT)
print("\n25M. Statistical annotations:")
for feature in features:
    stat = stat_lookup[feature]
    print(
        feature,
        "| FDR =",
        f"{float(stat['fdr_p_value']):.3e}",
        "| effect =",
        f"{float(stat['rank_biserial_effect']):.3f}",
        "|",
        stat["significance_symbol"],
    )

print("\n" + "=" * 78)
print("STEP 25 SUMMARY")
print("=" * 78)
print("Peptides plotted:", len(df))
print("Active:", active_n)
print("Inactive:", inactive_n)
print("Descriptors plotted:", len(features))
print("Statistical rows used:", len(stats))
print("Missing plotting values:", missing_values)
print("Non-finite plotting values:", nonfinite_values)
print("\nAnnotated PNG:")
print(PNG_OUTPUT)
print("\nAnnotated PDF:")
print(PDF_OUTPUT)
print("\nSTEP 25 COMPLETED SUCCESSFULLY")
print("=" * 78)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
FIGURE_DIR = PROJECT_DIR / "figures"
PNG_OUTPUT = FIGURE_DIR / "Step23_Physicochemical_Distributions.png"
PDF_OUTPUT = FIGURE_DIR / "Step23_Physicochemical_Distributions.pdf"
SUMMARY_OUTPUT = PROJECT_DIR / "results" / "step23_physicochemical_summary.csv"

df = pd.read_csv(FEATURE_FILE)
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

plot_values = df[features].apply(pd.to_numeric, errors="coerce")
missing_plot_values = int(plot_values.isna().sum().sum())
nonfinite_plot_values = int((~np.isfinite(plot_values.to_numpy())).sum())
if missing_plot_values != 0 or nonfinite_plot_values != 0:
    raise ValueError("Plotting descriptors contain missing or non-finite values.")

active = df.loc[df["binary_class"] == "Active"]
inactive = df.loc[df["binary_class"] == "Inactive"]
active_n = len(active)
inactive_n = len(inactive)

summary = pd.DataFrame(
    {
        "descriptor": features,
        "inactive_mean": [inactive[name].mean() for name in features],
        "active_mean": [active[name].mean() for name in features],
        "inactive_median": [inactive[name].median() for name in features],
        "active_median": [active[name].median() for name in features],
    }
)
summary.to_csv(SUMMARY_OUTPUT, index=False)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
figure, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
axes = axes.ravel()
colors = ["#4C78A8", "#E45756"]

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

    axis.set_title(title)
    axis.set_xticks([1, 2], [f"Inactive\n(n={inactive_n})", f"Active\n(n={active_n})"])
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

axes[-1].axis("off")
figure.suptitle("Physicochemical distributions of lung-cancer anticancer peptides", fontsize=15)
figure.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight")
figure.savefig(PDF_OUTPUT, bbox_inches="tight")
plt.close(figure)

print("=" * 72)
print("STEP 23 - PHYSICOCHEMICAL DISTRIBUTIONS")
print("=" * 72)
print("Active peptides:", active_n)
print("Inactive peptides:", inactive_n)
print("Descriptors plotted:", len(features))
print("Missing plotting values:", missing_plot_values)
print("Non-finite plotting values:", nonfinite_plot_values)
print("\nPNG:")
print(PNG_OUTPUT)
print("\nPDF:")
print(PDF_OUTPUT)
print("\nSummary table:")
print(SUMMARY_OUTPUT)
print("\nSTEP 23 COMPLETED SUCCESSFULLY")
print("=" * 72)
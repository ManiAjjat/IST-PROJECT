from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
INPUT_CSV = RESULTS_DIR / "step35_traditional_model_comparison.csv"
PLOT_DATA_CSV = RESULTS_DIR / "step37_traditional_model_metric_plot_data.csv"
OUTPUT_PNG = FIGURE_DIR / "Step37_Traditional_Model_Metric_Comparison.png"
OUTPUT_PDF = FIGURE_DIR / "Step37_Traditional_Model_Metric_Comparison.pdf"

model_order = ["Logistic Regression", "RBF-SVM", "Random Forest", "XGBoost"]
metrics = ["AUROC", "AUPRC", "MCC", "F1"]
colors = {
    "AUROC": "#4C78A8",
    "AUPRC": "#F58518",
    "MCC": "#54A24B",
    "F1": "#E45756",
}

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Step 35 comparison table not found: {INPUT_CSV}")

comparison = pd.read_csv(INPUT_CSV)
required_columns = {"model", *metrics}
missing_columns = required_columns.difference(comparison.columns)
if missing_columns:
    raise ValueError(f"Step 35 table is missing columns: {sorted(missing_columns)}")
if comparison["model"].duplicated().any():
    raise ValueError("Step 35 table contains duplicate model names.")

comparison = comparison.set_index("model")
missing_models = [model for model in model_order if model not in comparison.index]
if missing_models:
    raise ValueError(f"Step 35 table is missing models: {missing_models}")

plot_data = comparison.loc[model_order, metrics].reset_index()
if plot_data[metrics].isna().any().any():
    raise ValueError("One or more plotted metric values are missing.")
if not plot_data[metrics].apply(lambda column: column.between(0, 1)).all().all():
    raise ValueError("All plotted metrics must be between 0 and 1.")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
plot_data.to_csv(PLOT_DATA_CSV, index=False)

x_positions = np.arange(len(model_order))
bar_width = 0.19
offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * bar_width

figure, axis = plt.subplots(figsize=(11, 7))
for metric_index, metric in enumerate(metrics):
    values = plot_data[metric].to_numpy(dtype=float)
    bars = axis.bar(
        x_positions + offsets[metric_index],
        values,
        width=bar_width,
        label=metric,
        color=colors[metric],
        edgecolor="white",
        linewidth=0.6,
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)

axis.set_title("Traditional-model metric comparison", fontsize=15, pad=14)
axis.set_ylabel("Metric value")
axis.set_xticks(x_positions)
axis.set_xticklabels(model_order)
axis.set_ylim(0, 1.08)
axis.set_yticks(np.arange(0, 1.01, 0.1))
axis.grid(axis="y", alpha=0.25)
axis.set_axisbelow(True)
axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
figure.tight_layout()
figure.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
figure.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
plt.close(figure)

print("\n37A. Plotting data:")
print(plot_data.to_string(index=False))
print("\n37B. Metric leaders:")
for metric in metrics:
    leader_index = plot_data[metric].idxmax()
    print(f"{metric}: {plot_data.loc[leader_index, 'model']} ({plot_data.loc[leader_index, metric]:.6f})")

print("\n37C. Output checks:")
print("PNG exists:", OUTPUT_PNG.exists())
print("PDF exists:", OUTPUT_PDF.exists())
print("Plot data CSV exists:", PLOT_DATA_CSV.exists())
print("\n" + "=" * 86)
print("STEP 37 SUMMARY")
print("=" * 86)
print("Models:", ", ".join(model_order))
print("Metrics:", ", ".join(metrics))
print("Test peptides: 181")
print("Active: 20")
print("Inactive: 161")
print("\nPNG:", OUTPUT_PNG)
print("PDF:", OUTPUT_PDF)
print("Plot data:", PLOT_DATA_CSV)
print("\nSTEP 37 COMPLETED SUCCESSFULLY")
print("=" * 86)

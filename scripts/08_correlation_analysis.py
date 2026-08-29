from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
INPUT_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
CORR_OUTPUT = PROJECT_DIR / "results" / "step26_spearman_correlation_matrix.csv"
PAIR_OUTPUT = PROJECT_DIR / "results" / "step26_high_correlation_pairs.csv"
PNG_OUTPUT = PROJECT_DIR / "figures" / "Step26_Spearman_Correlation_Heatmap.png"
PDF_OUTPUT = PROJECT_DIR / "figures" / "Step26_Spearman_Correlation_Heatmap.pdf"

metadata_columns = [
    "ID",
    "sequence",
    "class",
    "original_class",
    "label",
    "binary_class",
    "inactive_source",
    "is_virtual_inactive",
]
df = pd.read_csv(INPUT_FILE)
feature_columns = [column for column in df.columns if column not in metadata_columns]
feature_values = df[feature_columns].apply(pd.to_numeric, errors="coerce")
missing_values = int(feature_values.isna().sum().sum())
nonfinite_values = int((~np.isfinite(feature_values.to_numpy())).sum())
if missing_values != 0 or nonfinite_values != 0:
    raise ValueError("Feature matrix contains missing or non-finite values.")
if len(feature_columns) != 32:
    raise ValueError("Expected 32 traditional features.")

corr_values, p_values = spearmanr(feature_values, axis=0)
corr_matrix = pd.DataFrame(corr_values, index=feature_columns, columns=feature_columns)
corr_matrix.to_csv(CORR_OUTPUT)

pairs = []
for row_index, feature_a in enumerate(feature_columns):
    for column_index in range(row_index + 1, len(feature_columns)):
        feature_b = feature_columns[column_index]
        rho = float(corr_matrix.loc[feature_a, feature_b])
        if abs(rho) >= 0.80:
            pairs.append({"feature_1": feature_a, "feature_2": feature_b, "spearman_rho": rho, "absolute_rho": abs(rho)})
pairs_df = pd.DataFrame(pairs).sort_values("absolute_rho", ascending=False)
pairs_df.to_csv(PAIR_OUTPUT, index=False)

figure, axis = plt.subplots(figsize=(15, 13))
image = axis.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
axis.set_xticks(range(len(feature_columns)), feature_columns, rotation=90, fontsize=7)
axis.set_yticks(range(len(feature_columns)), feature_columns, fontsize=7)
axis.set_title("Spearman correlation of traditional peptide features", pad=18)
colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
colorbar.set_label("Spearman rho")
figure.tight_layout()
figure.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
figure.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(figure)

print("=" * 78)
print("STEP 26 - SPEARMAN CORRELATION AND REDUNDANCY ANALYSIS")
print("=" * 78)
print("Peptides:", len(df))
print("Features:", len(feature_columns))
print("Missing feature values:", missing_values)
print("Non-finite feature values:", nonfinite_values)
print("High-correlation pairs (|rho| >= 0.80):", len(pairs_df))
print("\nStrongest pairs:")
print(pairs_df.head(20).to_string(index=False) if not pairs_df.empty else "None")
print("\nCorrelation matrix:")
print(CORR_OUTPUT)
print("\nHigh-correlation pairs:")
print(PAIR_OUTPUT)
print("\nHeatmap PNG:")
print(PNG_OUTPUT)
print("\nHeatmap PDF:")
print(PDF_OUTPUT)
print("\nSTEP 26 COMPLETED SUCCESSFULLY")
print("=" * 78)
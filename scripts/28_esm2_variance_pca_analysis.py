from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
NPY_FILE = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_FILE = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
PCA_VARIANCE_OUTPUT = RESULTS_DIR / "step46_esm2_pca_explained_variance.csv"
PCA_SUMMARY_OUTPUT = RESULTS_DIR / "step46_esm2_pca_variance_summary.csv"
SD_PNG = FIGURE_DIR / "Step46_ESM2_Dimension_SD_Distribution.png"
SD_PDF = FIGURE_DIR / "Step46_ESM2_Dimension_SD_Distribution.pdf"
PCA_PNG = FIGURE_DIR / "Step46_ESM2_PCA_Cumulative_Variance.png"
PCA_PDF = FIGURE_DIR / "Step46_ESM2_PCA_Cumulative_Variance.pdf"
EXPECTED_SHAPE = (901, 1280)
VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

print("=" * 94)
print("STEP 46 - ESM-2 VARIANCE AND DESCRIPTIVE PCA ANALYSIS")
print("=" * 94)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
for required_file in (NPY_FILE, METADATA_FILE):
    if not required_file.exists():
        raise FileNotFoundError(f"Required input not found: {required_file}")

print("\n46A. Load verified embeddings:")
X = np.load(NPY_FILE, allow_pickle=False)
metadata = pd.read_csv(METADATA_FILE)
if X.shape != EXPECTED_SHAPE or X.dtype != np.float32:
    raise ValueError(f"Unexpected embedding matrix: {X.shape}, {X.dtype}")
if len(metadata) != EXPECTED_SHAPE[0]:
    raise ValueError("Metadata row count does not match the embedding matrix.")
if not np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(EXPECTED_SHAPE[0])):
    raise ValueError("Metadata embedding rows are not aligned.")
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
if len(development_indices) != 720 or len(test_indices) != 181:
    raise ValueError("Expected 720 development and 181 test rows.")
X_dev = X[development_indices]
print("Full matrix:", X.shape)
print("Development matrix used:", X_dev.shape)
print("Test rows used for fitting/statistics:", 0)

print("\n46B. Development-only raw dimension variance:")
development_means = X_dev.mean(axis=0, dtype=np.float64)
development_sds = X_dev.std(axis=0, dtype=np.float64, ddof=0)
development_variances = X_dev.var(axis=0, dtype=np.float64, ddof=0)
if np.any(development_sds == 0):
    raise ValueError("Development matrix contains a constant dimension.")
total_raw_variance = float(development_variances.sum())
variance_shares = development_variances / total_raw_variance
sorted_variance_shares = np.sort(variance_shares)[::-1]
largest_dimension_variance_share = float(sorted_variance_shares[0])
top_10_dimension_variance_share = float(sorted_variance_shares[:10].sum())
top_50_dimension_variance_share = float(sorted_variance_shares[:50].sum())
top_100_dimension_variance_share = float(sorted_variance_shares[:100].sum())
print("Minimum SD:", float(development_sds.min()))
print("Median SD:", float(np.median(development_sds)))
print("Mean SD:", float(development_sds.mean()))
print("Maximum SD:", float(development_sds.max()))
print("Largest single-dimension raw variance share:", largest_dimension_variance_share)
print("Top 10 raw variance share:", top_10_dimension_variance_share)
print("Top 50 raw variance share:", top_50_dimension_variance_share)
print("Top 100 raw variance share:", top_100_dimension_variance_share)

print("\n46C. Development-only standardization and descriptive PCA:")
scaler = StandardScaler(with_mean=True, with_std=True)
X_dev_standardized = scaler.fit_transform(X_dev).astype(np.float64, copy=False)
standardized_means = X_dev_standardized.mean(axis=0)
standardized_sds = X_dev_standardized.std(axis=0, ddof=0)
if not np.allclose(standardized_means, 0.0, atol=1e-6):
    raise RuntimeError("Development-only standardization did not center all dimensions.")
if not np.allclose(standardized_sds, 1.0, atol=1e-6):
    raise RuntimeError("Development-only standardization did not scale all dimensions.")

pca = PCA(svd_solver="full")
pca.fit(X_dev_standardized)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
component_numbers = np.arange(1, len(explained_variance_ratio) + 1)
if len(explained_variance_ratio) != 720:
    raise RuntimeError("Expected PCA to return 720 sample-limited components.")
if not np.isclose(cumulative_variance[-1], 1.0, atol=1e-8):
    raise RuntimeError("PCA explained-variance ratios do not sum to one.")

components_needed = {
    threshold: int(np.searchsorted(cumulative_variance, threshold, side="left") + 1)
    for threshold in VARIANCE_THRESHOLDS
}
print("PCA components available:", len(component_numbers))
for threshold in VARIANCE_THRESHOLDS:
    print(f"{int(threshold * 100)}% variance PCs:", components_needed[threshold])

pca_variance_df = pd.DataFrame(
    {
        "principal_component": component_numbers,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": cumulative_variance,
        "singular_value": pca.singular_values_,
    }
)
pca_variance_df.to_csv(PCA_VARIANCE_OUTPUT, index=False)

summary_df = pd.DataFrame(
    [
        {
            "embedding_dimensions": X.shape[1],
            "development_rows_used": X_dev.shape[0],
            "test_rows_used": 0,
            "pca_input": "development-only standardized ESM-2 embeddings",
            "pca_solver": "full",
            "pca_components_available": len(component_numbers),
            "pcs_for_80_percent": components_needed[0.80],
            "pcs_for_90_percent": components_needed[0.90],
            "pcs_for_95_percent": components_needed[0.95],
            "pcs_for_99_percent": components_needed[0.99],
            "development_minimum_dimension_sd": float(development_sds.min()),
            "development_median_dimension_sd": float(np.median(development_sds)),
            "development_mean_dimension_sd": float(development_sds.mean()),
            "development_maximum_dimension_sd": float(development_sds.max()),
            "development_total_raw_variance": total_raw_variance,
            "largest_dimension_raw_variance_share": largest_dimension_variance_share,
            "top_10_dimensions_raw_variance_share": top_10_dimension_variance_share,
            "top_50_dimensions_raw_variance_share": top_50_dimension_variance_share,
            "top_100_dimensions_raw_variance_share": top_100_dimension_variance_share,
            "first_pc_explained_variance_ratio": float(explained_variance_ratio[0]),
            "first_10_pcs_cumulative_variance": float(cumulative_variance[9]),
            "first_50_pcs_cumulative_variance": float(cumulative_variance[49]),
            "first_100_pcs_cumulative_variance": float(cumulative_variance[99]),
            "pca_object_saved_for_modeling": False,
            "classifier_trained": False,
        }
    ]
)
summary_df.to_csv(PCA_SUMMARY_OUTPUT, index=False)

print("\n46D. Dimension-SD distribution figure:")
sd_figure, sd_axis = plt.subplots(figsize=(9, 6.5))
sd_axis.hist(
    development_sds,
    bins=45,
    color="#4C78A8",
    edgecolor="white",
    linewidth=0.6,
)
sd_axis.axvline(
    np.median(development_sds),
    color="#E45756",
    linestyle="--",
    linewidth=1.8,
    label=f"Median SD = {np.median(development_sds):.4f}",
)
sd_axis.set_title("ESM-2 dimension variability in the development set", fontsize=14, pad=13)
sd_axis.set_xlabel("Standard deviation across 720 development peptides")
sd_axis.set_ylabel("Number of ESM-2 dimensions")
sd_axis.grid(axis="y", alpha=0.25)
sd_axis.set_axisbelow(True)
sd_axis.legend(frameon=False)
sd_figure.tight_layout()
sd_figure.savefig(SD_PNG, dpi=600, bbox_inches="tight", facecolor="white")
sd_figure.savefig(SD_PDF, bbox_inches="tight", facecolor="white")
plt.close(sd_figure)

print("\n46E. PCA cumulative-variance figure:")
pca_figure, pca_axis = plt.subplots(figsize=(9, 6.5))
pca_axis.plot(component_numbers, cumulative_variance, color="#4C78A8", linewidth=2)
threshold_colors = ["#54A24B", "#F58518", "#E45756", "#B279A2"]
for threshold, color in zip(VARIANCE_THRESHOLDS, threshold_colors):
    count = components_needed[threshold]
    pca_axis.axhline(threshold, color=color, linestyle="--", linewidth=1, alpha=0.75)
    pca_axis.axvline(count, color=color, linestyle=":", linewidth=1, alpha=0.75)
    pca_axis.scatter([count], [cumulative_variance[count - 1]], color=color, s=35, zorder=3)
    pca_axis.annotate(
        f"{int(threshold * 100)}%: {count} PCs",
        xy=(count, cumulative_variance[count - 1]),
        xytext=(8, -14 if threshold < 0.99 else -22),
        textcoords="offset points",
        fontsize=9,
        color=color,
    )
pca_axis.set_title("Development-only standardized ESM-2 PCA", fontsize=14, pad=13)
pca_axis.set_xlabel("Number of principal components")
pca_axis.set_ylabel("Cumulative explained variance ratio")
pca_axis.set_xlim(1, len(component_numbers))
pca_axis.set_ylim(0, 1.02)
pca_axis.grid(alpha=0.25)
pca_axis.set_axisbelow(True)
pca_figure.tight_layout()
pca_figure.savefig(PCA_PNG, dpi=600, bbox_inches="tight", facecolor="white")
pca_figure.savefig(PCA_PDF, bbox_inches="tight", facecolor="white")
plt.close(pca_figure)

print("\n46F. Output checks:")
for output_path in (
    PCA_SUMMARY_OUTPUT, PCA_VARIANCE_OUTPUT, SD_PNG, SD_PDF, PCA_PNG, PCA_PDF
):
    print(output_path.name, "exists:", output_path.exists())

print("\n" + "=" * 94)
print("STEP 46 SUMMARY")
print("=" * 94)
print("Development matrix analyzed:", X_dev.shape)
print("Test rows used:", 0)
print("Original dimensions:", X_dev.shape[1])
print("80% variance PCs:", components_needed[0.80])
print("90% variance PCs:", components_needed[0.90])
print("95% variance PCs:", components_needed[0.95])
print("99% variance PCs:", components_needed[0.99])
print("Descriptive PCA object saved/reused:", False)
print("Classifier trained:", False)
print("\nPCA summary:", PCA_SUMMARY_OUTPUT)
print("PCA explained variance:", PCA_VARIANCE_OUTPUT)
print("SD figure PNG/PDF:", SD_PNG, SD_PDF)
print("PCA figure PNG/PDF:", PCA_PNG, PCA_PDF)
print("\nSTEP 46 COMPLETED SUCCESSFULLY")
print("=" * 94)

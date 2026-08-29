from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
DERIVED_DIR = PROJECT_DIR / "derived"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

TRADITIONAL_INPUT = DERIVED_DIR / "traditional_features.csv"
ESM2_INPUT = DERIVED_DIR / "esm2_embeddings.npy"
METADATA_INPUT = DERIVED_DIR / "esm2_embedding_metadata.csv"
SPLIT_INPUT = DERIVED_DIR / "fixed_split.csv"

PCA_OUTPUT = RESULTS_DIR / "step77_feature_space_pca_summary.csv"
CCA_OUTPUT = RESULTS_DIR / "step77_canonical_correlations.csv"
CORRELATION_OUTPUT = RESULTS_DIR / "step77_traditional_vs_esm2_pc_correlations.csv"
STRONGEST_OUTPUT = RESULTS_DIR / "step77_descriptor_strongest_esm2_pc_associations.csv"
QC_OUTPUT = RESULTS_DIR / "step77_feature_space_complementarity_qc.csv"
CCA_PNG = FIGURES_DIR / "Step77_Traditional_vs_ESM2_Canonical_Correlation.png"
CCA_PDF = FIGURES_DIR / "Step77_Traditional_vs_ESM2_Canonical_Correlation.pdf"
HEATMAP_PNG = FIGURES_DIR / "Step77_Descriptor_ESM2_PC_Association_Map.png"
HEATMAP_PDF = FIGURES_DIR / "Step77_Descriptor_ESM2_PC_Association_Map.pdf"

NON_FEATURE_COLUMNS = {
    "ID", "sequence", "class", "original_class", "label", "binary_class",
    "inactive_source", "is_virtual_inactive",
}
MAIN_DESCRIPTORS = [
    "length", "molecular_weight", "net_charge_pH7_4", "isoelectric_point",
    "mean_eisenberg_hydrophobicity", "hydrophobic_moment", "boman_index",
]
VARIANCE_TARGET = 0.90
CCA_CAP = 20


print("=" * 104)
print("STEP 77 - TRADITIONAL VS ESM-2 FEATURE-SPACE COMPLEMENTARITY")
print("=" * 104)

for path in (TRADITIONAL_INPUT, ESM2_INPUT, METADATA_INPUT, SPLIT_INPUT):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

traditional = pd.read_csv(TRADITIONAL_INPUT)
metadata = pd.read_csv(METADATA_INPUT)
split = pd.read_csv(SPLIT_INPUT)
esm2 = np.load(ESM2_INPUT)

if traditional.shape[0] != 901 or metadata.shape[0] != 901 or split.shape[0] != 901:
    raise ValueError("Expected 901 aligned peptide rows.")
if esm2.shape != (901, 1280) or esm2.dtype != np.float32:
    raise ValueError("Expected the frozen (901,1280) float32 ESM-2 matrix.")
if not np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(901)):
    raise ValueError("Embedding-row metadata is not exactly 0-900.")

for col in ("ID", "sequence", "label"):
    if not np.array_equal(traditional[col].to_numpy(), metadata[col].to_numpy()):
        raise ValueError(f"Traditional/metadata {col} alignment failed.")
    if not np.array_equal(traditional[col].to_numpy(), split[col].to_numpy()):
        raise ValueError(f"Traditional/split {col} alignment failed.")
if not np.array_equal(metadata["split"].to_numpy(), split["split"].to_numpy()):
    raise ValueError("Metadata/fixed-split assignments disagree.")

feature_columns = [c for c in traditional.columns if c not in NON_FEATURE_COLUMNS]
if len(feature_columns) != 32:
    raise ValueError(f"Expected 32 traditional features; found {len(feature_columns)}.")
if not set(MAIN_DESCRIPTORS).issubset(feature_columns):
    raise ValueError("A requested main descriptor is absent.")

dev_mask = split["split"].eq("development").to_numpy()
test_mask = split["split"].eq("test").to_numpy()
if dev_mask.sum() != 720 or test_mask.sum() != 181 or np.any(dev_mask & test_mask):
    raise ValueError("Expected a disjoint 720/181 development/test split.")

X_trad_dev = traditional.loc[dev_mask, feature_columns].to_numpy(dtype=np.float64)
X_esm_dev = esm2[dev_mask].astype(np.float64, copy=False)
if not np.isfinite(X_trad_dev).all() or not np.isfinite(X_esm_dev).all():
    raise ValueError("Development matrices contain non-finite values.")

# Every fitted transformation sees development rows only.
trad_scaler = StandardScaler()
esm_scaler = StandardScaler()
X_trad_scaled = trad_scaler.fit_transform(X_trad_dev)
X_esm_scaled = esm_scaler.fit_transform(X_esm_dev)

trad_pca_full = PCA(svd_solver="full")
esm_pca_full = PCA(svd_solver="full")
trad_scores_full = trad_pca_full.fit_transform(X_trad_scaled)
esm_scores_full = esm_pca_full.fit_transform(X_esm_scaled)

def count_for_variance(pca, target):
    return int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), target) + 1)

trad_n90 = count_for_variance(trad_pca_full, VARIANCE_TARGET)
esm_n90 = count_for_variance(esm_pca_full, VARIANCE_TARGET)
trad_scores = trad_scores_full[:, :trad_n90]
esm_scores = esm_scores_full[:, :esm_n90]
trad_cumulative = np.cumsum(trad_pca_full.explained_variance_ratio_)
esm_cumulative = np.cumsum(esm_pca_full.explained_variance_ratio_)

pca_rows = []
for representation, original_dimensions, pca, n90, cumulative in (
    ("Traditional", 32, trad_pca_full, trad_n90, trad_cumulative),
    ("ESM-2", 1280, esm_pca_full, esm_n90, esm_cumulative),
):
    pca_rows.append({
        "representation": representation,
        "development_rows_used_for_scaler_fit": int(pca.n_samples_),
        "development_rows_used_for_pca_fit": int(pca.n_samples_),
        "locked_test_rows_used_for_fit": 0,
        "original_dimensions": original_dimensions,
        "available_pcs": int(len(pca.explained_variance_ratio_)),
        "variance_target": VARIANCE_TARGET,
        "pcs_required_for_at_least_90pct": n90,
        "cumulative_variance_at_selected_count": float(cumulative[n90 - 1]),
        "cumulative_variance_before_selected_count": float(cumulative[n90 - 2]) if n90 > 1 else 0.0,
    })
pca_summary = pd.DataFrame(pca_rows)

cca_components = min(trad_n90, esm_n90, CCA_CAP)
cca = CCA(n_components=cca_components, scale=True, max_iter=5000, tol=1e-08)
trad_canonical, esm_canonical = cca.fit_transform(trad_scores, esm_scores)
cca_rows = []
for i in range(cca_components):
    correlation = float(np.corrcoef(trad_canonical[:, i], esm_canonical[:, i])[0, 1])
    cca_rows.append({
        "canonical_component": i + 1,
        "canonical_dimension": f"CC{i + 1}",
        "canonical_correlation": correlation,
        "squared_canonical_correlation": correlation ** 2,
        "development_rows": 720,
        "traditional_pc_count": trad_n90,
        "esm2_pc_count": esm_n90,
    })
cca_df = pd.DataFrame(cca_rows)

correlation_rows = []
for descriptor_index, descriptor in enumerate(feature_columns):
    values = X_trad_dev[:, descriptor_index]
    for pc_index in range(esm_n90):
        rho, pvalue = spearmanr(values, esm_scores[:, pc_index])
        correlation_rows.append({
            "traditional_descriptor": descriptor,
            "esm2_pc": f"ESM2_PC{pc_index + 1:02d}",
            "esm2_pc_number": pc_index + 1,
            "spearman_rho": float(rho),
            "absolute_spearman_rho": float(abs(rho)),
            "descriptive_unadjusted_p_value": float(pvalue),
            "development_rows": 720,
        })
correlations = pd.DataFrame(correlation_rows)

strongest = (
    correlations.sort_values(
        ["traditional_descriptor", "absolute_spearman_rho", "esm2_pc_number"],
        ascending=[True, False, True],
    )
    .groupby("traditional_descriptor", sort=False, as_index=False)
    .first()
)
strongest = strongest.rename(columns={
    "esm2_pc": "strongest_esm2_pc",
    "esm2_pc_number": "strongest_esm2_pc_number",
    "spearman_rho": "signed_spearman_rho",
    "absolute_spearman_rho": "strongest_absolute_spearman_rho",
    "descriptive_unadjusted_p_value": "descriptive_unadjusted_p_value_at_strongest_pc",
})
strongest["selected_for_modeling"] = False

pca_summary.to_csv(PCA_OUTPUT, index=False)
cca_df.to_csv(CCA_OUTPUT, index=False)
correlations.to_csv(CORRELATION_OUTPUT, index=False)
strongest.to_csv(STRONGEST_OUTPUT, index=False)

# Figure 1: canonical correlations.
fig, ax = plt.subplots(figsize=(10.0, 5.8), facecolor="white")
ax.set_facecolor("white")
x = np.arange(1, cca_components + 1)
ax.bar(x, cca_df["canonical_correlation"], color="#4C78A8", edgecolor="black", linewidth=0.5)
ax.plot(x, cca_df["canonical_correlation"], color="#1F4E79", marker="o", linewidth=1.2, markersize=4)
ax.set_xticks(x)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Canonical dimension")
ax.set_ylabel("In-sample canonical correlation")
ax.set_title("Development-Set Association Between Traditional and ESM-2 Feature Spaces", pad=12)
ax.grid(axis="y", alpha=0.22)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(
    0.5, 0.015,
    f"CCA used {trad_n90} traditional PCs and {esm_n90} ESM-2 PCs retained at >=90% variance; n=720 development peptides.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0.05, 0.06, 0.99, 0.96])
fig.savefig(CCA_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(CCA_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: seven biologically interpretable descriptors by retained ESM-2 PCs.
main_corr = (
    correlations.pivot(index="traditional_descriptor", columns="esm2_pc_number", values="spearman_rho")
    .loc[MAIN_DESCRIPTORS, np.arange(1, esm_n90 + 1)]
)
fig_width = max(14.0, 0.26 * esm_n90 + 5.5)
fig, ax = plt.subplots(figsize=(fig_width, 6.4), facecolor="white")
ax.set_facecolor("white")
im = ax.imshow(main_corr.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
ax.set_yticks(np.arange(len(MAIN_DESCRIPTORS)), MAIN_DESCRIPTORS)
tick_step = 2 if esm_n90 <= 60 else 5
tick_indices = np.arange(0, esm_n90, tick_step)
ax.set_xticks(tick_indices, [f"PC{i + 1}" for i in tick_indices], rotation=45, ha="right")
ax.set_xlabel("Retained ESM-2 principal components")
ax.set_title("Spearman Associations of Main Physicochemical Descriptors with ESM-2 PCs", pad=12)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Spearman rho")
fig.text(
    0.5, 0.015,
    "Development-set descriptive correlations only; ESM-2 PCs are not interpreted as single biological mechanisms.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0.05, 0.06, 0.99, 0.96])
fig.savefig(HEATMAP_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(HEATMAP_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "total_peptides": 901,
    "development_peptides": int(dev_mask.sum()),
    "locked_test_peptides": int(test_mask.sum()),
    "traditional_features": len(feature_columns),
    "esm2_dimensions": esm2.shape[1],
    "traditional_scaler_fit_rows": int(trad_scaler.n_samples_seen_),
    "esm2_scaler_fit_rows": int(esm_scaler.n_samples_seen_),
    "traditional_pca_fit_rows": int(trad_pca_full.n_samples_),
    "esm2_pca_fit_rows": int(esm_pca_full.n_samples_),
    "locked_test_rows_used_for_fitting_or_correlations": 0,
    "traditional_90pct_pc_count": trad_n90,
    "esm2_90pct_pc_count": esm_n90,
    "cca_components": cca_components,
    "cca_cap": CCA_CAP,
    "canonical_correlations_finite_and_in_unit_interval": bool(
        np.isfinite(cca_df["canonical_correlation"]).all()
        and cca_df["canonical_correlation"].between(0, 1).all()
    ),
    "descriptor_pc_correlation_rows": len(correlations),
    "expected_descriptor_pc_correlation_rows": 32 * esm_n90,
    "all_spearman_values_finite_and_in_unit_interval": bool(
        np.isfinite(correlations["spearman_rho"]).all()
        and correlations["spearman_rho"].between(-1, 1).all()
    ),
    "strongest_association_rows": len(strongest),
    "main_descriptors_in_figure": len(MAIN_DESCRIPTORS),
    "classifier_trained": False,
    "features_selected_for_later_models": False,
    "descriptors_removed": False,
    "esm2_dimensions_removed": False,
    "models_retuned": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print(f"Traditional PCs for >=90% variance: {trad_n90} ({trad_cumulative[trad_n90 - 1]:.6f})")
print(f"ESM-2 PCs for >=90% variance: {esm_n90} ({esm_cumulative[esm_n90 - 1]:.6f})")
print(f"CCA components: {cca_components}")
print("\nCanonical correlations:")
print(cca_df[["canonical_dimension", "canonical_correlation"]].round(6).to_string(index=False))
print("\nMain-descriptor strongest ESM-2 PC associations:")
print(strongest.loc[strongest["traditional_descriptor"].isin(MAIN_DESCRIPTORS), [
    "traditional_descriptor", "strongest_esm2_pc", "signed_spearman_rho",
]].round(6).to_string(index=False))
print("\nSTEP 77 COMPLETED SUCCESSFULLY")
print("=" * 104)

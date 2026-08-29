from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
DERIVED_DIR = PROJECT_DIR / "derived"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

TRADITIONAL_INPUT = DERIVED_DIR / "traditional_features.csv"
ESM2_INPUT = DERIVED_DIR / "esm2_embeddings.npy"
METADATA_INPUT = DERIVED_DIR / "esm2_embedding_metadata.csv"
SPLIT_INPUT = DERIVED_DIR / "fixed_split.csv"
FOLD_CSV_INPUT = DERIVED_DIR / "fixed_cv_folds.csv"
FOLD_NPZ_INPUT = DERIVED_DIR / "fixed_cv_folds.npz"

FOLD_OUTPUT = RESULTS_DIR / "step78_cv_cca_fold_summary.csv"
CORRELATION_OUTPUT = RESULTS_DIR / "step78_cv_canonical_correlations.csv"
DIMENSION_OUTPUT = RESULTS_DIR / "step78_cv_cca_dimension_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step78_cv_feature_complementarity_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step78_Cross_Validated_Canonical_Correlations.png"
FIGURE_PDF = FIGURES_DIR / "Step78_Cross_Validated_Canonical_Correlations.pdf"

NON_FEATURE_COLUMNS = {
    "ID", "sequence", "class", "original_class", "label", "binary_class",
    "inactive_source", "is_virtual_inactive",
}
VARIANCE_TARGET = 0.90
CCA_CAP = 10
FOLDS = range(1, 6)


print("=" * 108)
print("STEP 78 - CROSS-VALIDATED FEATURE-SPACE COMPLEMENTARITY")
print("=" * 108)

for path in (
    TRADITIONAL_INPUT, ESM2_INPUT, METADATA_INPUT, SPLIT_INPUT,
    FOLD_CSV_INPUT, FOLD_NPZ_INPUT,
):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

traditional = pd.read_csv(TRADITIONAL_INPUT)
metadata = pd.read_csv(METADATA_INPUT)
split = pd.read_csv(SPLIT_INPUT)
fold_csv = pd.read_csv(FOLD_CSV_INPUT)
fold_npz = np.load(FOLD_NPZ_INPUT)
esm2 = np.load(ESM2_INPUT)

if traditional.shape[0] != 901 or metadata.shape[0] != 901 or split.shape[0] != 901:
    raise ValueError("Expected 901 aligned peptide rows.")
if fold_csv.shape[0] != 901 or esm2.shape != (901, 1280):
    raise ValueError("Unexpected fold table or ESM-2 matrix shape.")
if not np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(901)):
    raise ValueError("Embedding metadata row mapping is invalid.")
for col in ("ID", "sequence", "label"):
    reference = traditional[col].to_numpy()
    if not np.array_equal(reference, metadata[col].to_numpy()):
        raise ValueError(f"Metadata {col} alignment failed.")
    if not np.array_equal(reference, split[col].to_numpy()):
        raise ValueError(f"Split {col} alignment failed.")
    if not np.array_equal(reference, fold_csv[col].to_numpy()):
        raise ValueError(f"Fold-table {col} alignment failed.")
if not np.array_equal(split["split"].to_numpy(), fold_csv["split"].to_numpy()):
    raise ValueError("Split assignments disagree.")

feature_columns = [c for c in traditional.columns if c not in NON_FEATURE_COLUMNS]
if len(feature_columns) != 32:
    raise ValueError(f"Expected 32 traditional features; found {len(feature_columns)}.")

dev_global = np.flatnonzero(split["split"].eq("development").to_numpy())
test_global = np.flatnonzero(split["split"].eq("test").to_numpy())
if len(dev_global) != 720 or len(test_global) != 181:
    raise ValueError("Expected exactly 720 development and 181 test peptides.")
if not np.array_equal(dev_global, fold_npz["development_global_indices"]):
    raise ValueError("NPZ development indices disagree with the fixed split.")
if set(fold_csv.loc[dev_global, "cv_fold"]) != {1, 2, 3, 4, 5}:
    raise ValueError("Development fold labels are not exactly 1-5.")
if not (fold_csv.loc[test_global, "cv_fold"] == -1).all():
    raise ValueError("A locked-test peptide has a development CV fold label.")

X_traditional = traditional[feature_columns].to_numpy(dtype=np.float64)
X_esm2 = esm2.astype(np.float64, copy=False)


def pc90_count(pca):
    return int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), VARIANCE_TARGET) + 1)


fold_rows = []
correlation_rows = []
qc_rows = []
all_validation_indices = []

for fold in FOLDS:
    train_idx = fold_npz[f"fold{fold}_train"].astype(int)
    validation_idx = fold_npz[f"fold{fold}_valid"].astype(int)
    all_validation_indices.extend(validation_idx.tolist())

    if len(train_idx) != 576 or len(validation_idx) != 144:
        raise ValueError(f"Fold {fold}: expected 576/144 rows.")
    train_validation_overlap = len(np.intersect1d(train_idx, validation_idx))
    test_overlap = len(np.intersect1d(np.union1d(train_idx, validation_idx), test_global))
    if train_validation_overlap != 0 or test_overlap != 0:
        raise ValueError(f"Fold {fold}: train/validation/test overlap detected.")
    if not np.isin(train_idx, dev_global).all() or not np.isin(validation_idx, dev_global).all():
        raise ValueError(f"Fold {fold}: a CV row is outside the development set.")

    trad_train_raw = X_traditional[train_idx]
    trad_valid_raw = X_traditional[validation_idx]
    esm_train_raw = X_esm2[train_idx]
    esm_valid_raw = X_esm2[validation_idx]

    trad_scaler = StandardScaler()
    esm_scaler = StandardScaler()
    trad_train_scaled = trad_scaler.fit_transform(trad_train_raw)
    trad_valid_scaled = trad_scaler.transform(trad_valid_raw)
    esm_train_scaled = esm_scaler.fit_transform(esm_train_raw)
    esm_valid_scaled = esm_scaler.transform(esm_valid_raw)

    trad_pca = PCA(svd_solver="full")
    esm_pca = PCA(svd_solver="full")
    trad_train_full = trad_pca.fit_transform(trad_train_scaled)
    trad_valid_full = trad_pca.transform(trad_valid_scaled)
    esm_train_full = esm_pca.fit_transform(esm_train_scaled)
    esm_valid_full = esm_pca.transform(esm_valid_scaled)

    trad_n90 = pc90_count(trad_pca)
    esm_n90 = pc90_count(esm_pca)
    trad_variance = float(np.cumsum(trad_pca.explained_variance_ratio_)[trad_n90 - 1])
    esm_variance = float(np.cumsum(esm_pca.explained_variance_ratio_)[esm_n90 - 1])
    cca_components = min(trad_n90, esm_n90, CCA_CAP)

    trad_train = trad_train_full[:, :trad_n90]
    trad_valid = trad_valid_full[:, :trad_n90]
    esm_train = esm_train_full[:, :esm_n90]
    esm_valid = esm_valid_full[:, :esm_n90]

    cca = CCA(n_components=cca_components, scale=True, max_iter=5000, tol=1e-08)
    trad_train_can, esm_train_can = cca.fit_transform(trad_train, esm_train)
    trad_valid_can, esm_valid_can = cca.transform(trad_valid, esm_valid)

    transformed_arrays = (
        trad_train, trad_valid, esm_train, esm_valid,
        trad_train_can, esm_train_can, trad_valid_can, esm_valid_can,
    )
    all_outputs_finite = all(np.isfinite(array).all() for array in transformed_arrays)
    if not all_outputs_finite:
        raise ValueError(f"Fold {fold}: a PCA or CCA output is non-finite.")

    validation_abs = []
    for component in range(cca_components):
        train_corr = float(np.corrcoef(
            trad_train_can[:, component], esm_train_can[:, component]
        )[0, 1])
        validation_corr = float(np.corrcoef(
            trad_valid_can[:, component], esm_valid_can[:, component]
        )[0, 1])
        validation_abs.append(abs(validation_corr))
        correlation_rows.append({
            "fold": fold,
            "canonical_component": component + 1,
            "canonical_dimension": f"CC{component + 1}",
            "training_correlation": train_corr,
            "validation_correlation": validation_corr,
            "absolute_validation_correlation": abs(validation_corr),
            "train_minus_validation_correlation": train_corr - validation_corr,
            "train_n": len(train_idx),
            "validation_n": len(validation_idx),
        })

    fold_rows.append({
        "fold": fold,
        "train_n": len(train_idx),
        "validation_n": len(validation_idx),
        "traditional_original_dimensions": 32,
        "esm2_original_dimensions": 1280,
        "traditional_pc90_count": trad_n90,
        "traditional_pc90_variance": trad_variance,
        "esm2_pc90_count": esm_n90,
        "esm2_pc90_variance": esm_variance,
        "cca_components": cca_components,
        "mean_abs_validation_CC1_CC3": float(np.mean(validation_abs[:3])),
        "mean_abs_validation_CC1_CC5": float(np.mean(validation_abs[:5])),
        "mean_abs_validation_all": float(np.mean(validation_abs)),
        "test_overlap": test_overlap,
        "train_validation_overlap": train_validation_overlap,
    })
    qc_rows.append({
        "fold": fold,
        "development_peptides": 720,
        "locked_test_peptides": 181,
        "train_n": len(train_idx),
        "validation_n": len(validation_idx),
        "traditional_scaler_fit_samples": int(trad_scaler.n_samples_seen_),
        "esm2_scaler_fit_samples": int(esm_scaler.n_samples_seen_),
        "traditional_pca_fit_samples": int(trad_pca.n_samples_),
        "esm2_pca_fit_samples": int(esm_pca.n_samples_),
        "cca_fit_samples": len(trad_train_can),
        "traditional_pc_count_training_determined": True,
        "esm2_pc_count_training_determined": True,
        "validation_rows_used_in_fit": 0,
        "locked_test_rows_used_in_fit_or_transform": 0,
        "train_validation_overlap": train_validation_overlap,
        "test_overlap": test_overlap,
        "all_pca_and_cca_outputs_finite": all_outputs_finite,
        "classifier_trained": False,
        "feature_selected": False,
        "representation_modified": False,
    })

fold_summary = pd.DataFrame(fold_rows)
canonical = pd.DataFrame(correlation_rows)
qc = pd.DataFrame(qc_rows)

if len(fold_summary) != 5 or len(canonical) != 50 or len(qc) != 5:
    raise ValueError("Unexpected Step-78 table dimensions.")
if sorted(all_validation_indices) != sorted(dev_global.tolist()):
    raise ValueError("Validation folds do not cover every development peptide exactly once.")
if canonical[["fold", "canonical_component"]].duplicated().any():
    raise ValueError("Duplicate fold/component key.")

dimension_summary = (
    canonical.groupby(["canonical_component", "canonical_dimension"], as_index=False)
    .agg(
        folds=("fold", "size"),
        mean_training_correlation=("training_correlation", "mean"),
        sd_training_correlation=("training_correlation", "std"),
        mean_validation_correlation=("validation_correlation", "mean"),
        sd_validation_correlation=("validation_correlation", "std"),
        mean_absolute_validation_correlation=("absolute_validation_correlation", "mean"),
        sd_absolute_validation_correlation=("absolute_validation_correlation", "std"),
        mean_train_minus_validation_correlation=("train_minus_validation_correlation", "mean"),
    )
)

fold_summary.to_csv(FOLD_OUTPUT, index=False)
canonical.to_csv(CORRELATION_OUTPUT, index=False)
dimension_summary.to_csv(DIMENSION_OUTPUT, index=False)
qc.to_csv(QC_OUTPUT, index=False)

# Mean train/validation correlations with validation SD across the five folds.
x = np.arange(1, 11)
train_mean = dimension_summary["mean_training_correlation"].to_numpy()
valid_mean = dimension_summary["mean_validation_correlation"].to_numpy()
valid_sd = dimension_summary["sd_validation_correlation"].to_numpy()

fig, ax = plt.subplots(figsize=(10.4, 6.2), facecolor="white")
ax.set_facecolor("white")
ax.plot(x, train_mean, marker="o", color="#4C78A8", linewidth=2, label="Training mean")
ax.errorbar(
    x, valid_mean, yerr=valid_sd, marker="D", color="#E45756",
    ecolor="#E45756", linewidth=2, capsize=5, label="Validation mean +/- SD",
)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("Canonical dimension")
ax.set_ylabel("Canonical correlation")
ax.set_title("Cross-Validated Traditional–ESM-2 Canonical Correlations", pad=12)
ax.grid(axis="y", alpha=0.22)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
fig.text(
    0.5, 0.018,
    "Each fold independently fitted scalers, >=90% PCA dimensions, and CCA on 576 training peptides; error bars show validation SD across five folds.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0.05, 0.07, 0.99, 0.96])
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nFold summary:")
print(fold_summary.round(6).to_string(index=False))
print("\nCanonical-dimension summary:")
print(dimension_summary.round(6).to_string(index=False))
print("\nSTEP 78 COMPLETED SUCCESSFULLY")
print("=" * 108)

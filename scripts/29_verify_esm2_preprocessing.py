from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
EMBEDDING_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
CV_NPZ_INPUT = PROJECT_DIR / "derived" / "fixed_cv_folds.npz"
SUMMARY_OUTPUT = PROJECT_DIR / "results" / "step47_esm2_preprocessing_verification_summary.csv"
DETAIL_OUTPUT = PROJECT_DIR / "results" / "step47_esm2_preprocessing_fold_details.csv"

PCA_COMPONENTS = (24, 52, 99, 274)
EXPECTED_ROWS = 901
EXPECTED_DIMENSIONS = 1280
EXPECTED_DEVELOPMENT = 720
EXPECTED_TEST = 181
EXPECTED_TRAIN = 576
EXPECTED_VALIDATION = 144


print("=" * 96)
print("STEP 47 - VERIFY LEAKAGE-SAFE ESM-2 PREPROCESSING")
print("=" * 96)

X = np.load(EMBEDDING_INPUT, allow_pickle=False)
metadata = pd.read_csv(METADATA_INPUT)
cv = np.load(CV_NPZ_INPUT, allow_pickle=False)

assert X.shape == (EXPECTED_ROWS, EXPECTED_DIMENSIONS)
assert len(metadata) == EXPECTED_ROWS
assert np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(EXPECTED_ROWS))
assert np.isfinite(X).all()

development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
assert len(development_indices) == EXPECTED_DEVELOPMENT
assert len(test_indices) == EXPECTED_TEST
assert np.intersect1d(development_indices, test_indices).size == 0
assert np.array_equal(cv["development_global_indices"], development_indices)

print("\nInputs:")
print("Embedding matrix:", X.shape, X.dtype)
print("Development rows:", len(development_indices))
print("Locked-test rows:", len(test_indices))
print("PCA candidates:", list(PCA_COMPONENTS))
print("scikit-learn:", sklearn.__version__)

details = []
wrong_scaler_folds = 0
wrong_pca_folds = 0
test_overlap_count = 0

for fold in range(1, 6):
    train_indices = cv[f"fold{fold}_train"]
    valid_indices = cv[f"fold{fold}_valid"]

    assert len(train_indices) == EXPECTED_TRAIN
    assert len(valid_indices) == EXPECTED_VALIDATION
    assert np.intersect1d(train_indices, valid_indices).size == 0
    assert np.isin(train_indices, development_indices).all()
    assert np.isin(valid_indices, development_indices).all()
    assert np.union1d(train_indices, valid_indices).size == EXPECTED_DEVELOPMENT

    fold_test_overlap = int(
        np.intersect1d(np.union1d(train_indices, valid_indices), test_indices).size
    )
    test_overlap_count += fold_test_overlap

    X_train = X[train_indices]
    X_valid = X[valid_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    scaler_fit_samples = int(np.asarray(scaler.n_samples_seen_).max())
    scaler_fit_correct = scaler_fit_samples == EXPECTED_TRAIN
    wrong_scaler_folds += int(not scaler_fit_correct)

    assert X_train_scaled.shape == (EXPECTED_TRAIN, EXPECTED_DIMENSIONS)
    assert X_valid_scaled.shape == (EXPECTED_VALIDATION, EXPECTED_DIMENSIONS)
    assert np.isfinite(X_train_scaled).all()
    assert np.isfinite(X_valid_scaled).all()
    assert np.allclose(scaler.mean_, X_train.mean(axis=0, dtype=np.float64))

    print(f"\nFold {fold}:")
    print("  Training:", X_train.shape)
    print("  Validation:", X_valid.shape)
    print("  Scaler samples seen:", scaler_fit_samples)
    print("  Locked-test overlap:", fold_test_overlap)

    for n_components in PCA_COMPONENTS:
        pca = PCA(n_components=n_components, svd_solver="full")
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_valid_pca = pca.transform(X_valid_scaled)
        pca_fit_samples = int(pca.n_samples_)
        pca_fit_correct = pca_fit_samples == EXPECTED_TRAIN
        wrong_pca_folds += int(not pca_fit_correct)

        assert X_train_pca.shape == (EXPECTED_TRAIN, n_components)
        assert X_valid_pca.shape == (EXPECTED_VALIDATION, n_components)
        assert np.isfinite(X_train_pca).all()
        assert np.isfinite(X_valid_pca).all()

        cumulative_variance = float(pca.explained_variance_ratio_.sum())
        details.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "input_dimensions": EXPECTED_DIMENSIONS,
                "scaler_fit_samples": scaler_fit_samples,
                "scaler_fit_training_only": scaler_fit_correct,
                "pca_components": n_components,
                "pca_fit_samples": pca_fit_samples,
                "pca_fit_training_only": pca_fit_correct,
                "train_output_rows": X_train_pca.shape[0],
                "train_output_dimensions": X_train_pca.shape[1],
                "validation_output_rows": X_valid_pca.shape[0],
                "validation_output_dimensions": X_valid_pca.shape[1],
                "cumulative_training_explained_variance": cumulative_variance,
                "train_output_finite": bool(np.isfinite(X_train_pca).all()),
                "validation_output_finite": bool(np.isfinite(X_valid_pca).all()),
                "train_validation_index_overlap": int(
                    np.intersect1d(train_indices, valid_indices).size
                ),
                "locked_test_index_overlap": fold_test_overlap,
            }
        )
        print(
            f"  PCA {n_components}: train {X_train_pca.shape}, "
            f"valid {X_valid_pca.shape}, fit samples {pca_fit_samples}, "
            f"training variance {cumulative_variance:.6f}"
        )

detail_df = pd.DataFrame(details)
assert len(detail_df) == 5 * len(PCA_COMPONENTS)
assert wrong_scaler_folds == 0
assert wrong_pca_folds == 0
assert test_overlap_count == 0
assert detail_df["train_validation_index_overlap"].eq(0).all()
assert detail_df["locked_test_index_overlap"].eq(0).all()

summary_df = pd.DataFrame(
    [
        {
            "embedding_rows": X.shape[0],
            "embedding_dimensions": X.shape[1],
            "development_rows": len(development_indices),
            "locked_test_rows": len(test_indices),
            "cv_folds": 5,
            "expected_train_rows_per_fold": EXPECTED_TRAIN,
            "expected_validation_rows_per_fold": EXPECTED_VALIDATION,
            "pca_candidates": ";".join(map(str, PCA_COMPONENTS)),
            "scaler_fits_audited": 5,
            "pca_fits_audited": len(detail_df),
            "incorrect_scaler_fit_counts": wrong_scaler_folds,
            "incorrect_pca_fit_counts": wrong_pca_folds,
            "train_validation_overlap_count": int(
                detail_df["train_validation_index_overlap"].sum()
            ),
            "test_peptides_used": test_overlap_count,
            "all_outputs_finite": bool(
                detail_df["train_output_finite"].all()
                and detail_df["validation_output_finite"].all()
            ),
            "classifier_trained": False,
            "preprocessing_objects_saved_for_modeling": False,
            "verification_passed": True,
        }
    ]
)

SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(SUMMARY_OUTPUT, index=False)
detail_df.to_csv(DETAIL_OUTPUT, index=False)

print("\n" + "=" * 96)
print("STEP 47 SUMMARY")
print("=" * 96)
print("Folds audited:", 5)
print("Scaler fits audited:", 5)
print("PCA fits audited:", len(detail_df))
print("Incorrect scaler fit counts:", wrong_scaler_folds)
print("Incorrect PCA fit counts:", wrong_pca_folds)
print("Test peptides used:", test_overlap_count)
print("Classifier trained:", False)
print("\nSummary:")
print(SUMMARY_OUTPUT)
print("\nFold details:")
print(DETAIL_OUTPUT)
print("\nSTEP 47 COMPLETED SUCCESSFULLY")
print("=" * 96)

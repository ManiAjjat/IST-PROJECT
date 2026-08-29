from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
FOLD_INPUT = DERIVED_DIR / "fixed_cv_folds.npz"

LR_CV_OUTPUT = RESULTS_DIR / "step79_fusion_logistic_regression_cv_results.csv"
LR_METRICS_OUTPUT = RESULTS_DIR / "step79_fusion_logistic_regression_test_metrics.csv"
LR_PRED_OUTPUT = RESULTS_DIR / "step79_fusion_logistic_regression_test_predictions.csv"
LR_MODEL_OUTPUT = RESULTS_DIR / "step79_fusion_logistic_regression_model.joblib"
LR_PARAMS_OUTPUT = RESULTS_DIR / "step79_fusion_logistic_regression_best_params.json"
SVM_CV_OUTPUT = RESULTS_DIR / "step79_fusion_svm_cv_results.csv"
SVM_METRICS_OUTPUT = RESULTS_DIR / "step79_fusion_svm_test_metrics.csv"
SVM_PRED_OUTPUT = RESULTS_DIR / "step79_fusion_svm_test_predictions.csv"
SVM_MODEL_OUTPUT = RESULTS_DIR / "step79_fusion_svm_model.joblib"
SVM_PARAMS_OUTPUT = RESULTS_DIR / "step79_fusion_svm_best_params.json"
COMPARISON_OUTPUT = RESULTS_DIR / "step79_fusion_model_comparison.csv"
QC_OUTPUT = RESULTS_DIR / "step79_fusion_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step79_Fusion_vs_Single_Representation.png"
FIGURE_PDF = FIGURES_DIR / "Step79_Fusion_vs_Single_Representation.pdf"

PCA_OPTIONS = [24, 52, 99]
LR_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
SVM_C_VALUES = [0.1, 1.0, 10.0, 100.0]
SVM_GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1]
SEED = 2026
THRESHOLD = 0.5
NON_FEATURE_COLUMNS = {
    "ID", "sequence", "class", "original_class", "label", "binary_class",
    "inactive_source", "is_virtual_inactive",
}


print("=" * 108)
print("STEP 79 - LEAKAGE-SAFE FEATURE-FUSION BENCHMARK")
print("=" * 108)

traditional = pd.read_csv(TRADITIONAL_INPUT)
metadata = pd.read_csv(METADATA_INPUT)
split = pd.read_csv(SPLIT_INPUT)
esm2 = np.load(ESM2_INPUT)
folds = np.load(FOLD_INPUT)

if traditional.shape[0] != 901 or metadata.shape[0] != 901 or split.shape[0] != 901:
    raise ValueError("Expected 901 aligned peptide rows.")
if esm2.shape != (901, 1280):
    raise ValueError("Expected a (901,1280) ESM-2 matrix.")
for col in ("ID", "sequence", "label"):
    if not np.array_equal(traditional[col].to_numpy(), metadata[col].to_numpy()):
        raise ValueError(f"Traditional/metadata {col} alignment failed.")
    if not np.array_equal(traditional[col].to_numpy(), split[col].to_numpy()):
        raise ValueError(f"Traditional/split {col} alignment failed.")
if not np.array_equal(metadata["split"].to_numpy(), split["split"].to_numpy()):
    raise ValueError("Split alignment failed.")

feature_columns = [c for c in traditional.columns if c not in NON_FEATURE_COLUMNS]
if len(feature_columns) != 32:
    raise ValueError("Expected exactly 32 traditional ML features.")

X_trad = traditional[feature_columns].to_numpy(dtype=np.float64)
X_esm = esm2.astype(np.float64, copy=False)
y = traditional["label"].to_numpy(dtype=int)
X_combined = np.hstack([X_trad, X_esm])
dev_idx = np.flatnonzero(split["split"].eq("development").to_numpy())
test_idx = np.flatnonzero(split["split"].eq("test").to_numpy())
if len(dev_idx) != 720 or len(test_idx) != 181:
    raise ValueError("Expected a 720/181 development/test split.")
if not np.array_equal(dev_idx, folds["development_global_indices"]):
    raise ValueError("Fixed-fold development indices disagree with the split.")


def metric_values(y_true, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y_true, probability),
        "AUPRC": average_precision_score(y_true, probability),
        "MCC": matthews_corrcoef(y_true, prediction),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "Specificity": tn / (tn + fp),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# Fit each fold/PCA preprocessing combination once; classifiers reuse frozen fold transforms.
cache = {}
preprocessing_audit = []
for fold in range(1, 6):
    train_idx = folds[f"fold{fold}_train"].astype(int)
    valid_idx = folds[f"fold{fold}_valid"].astype(int)
    if len(train_idx) != 576 or len(valid_idx) != 144:
        raise ValueError(f"Fold {fold}: invalid train/validation sizes.")
    if len(np.intersect1d(train_idx, valid_idx)) or len(np.intersect1d(np.union1d(train_idx, valid_idx), test_idx)):
        raise ValueError(f"Fold {fold}: overlap detected.")

    trad_scaler = StandardScaler()
    trad_train = trad_scaler.fit_transform(X_trad[train_idx])
    trad_valid = trad_scaler.transform(X_trad[valid_idx])
    esm_scaler = StandardScaler()
    esm_train_scaled = esm_scaler.fit_transform(X_esm[train_idx])
    esm_valid_scaled = esm_scaler.transform(X_esm[valid_idx])

    for n_components in PCA_OPTIONS:
        pca = PCA(n_components=n_components, svd_solver="full")
        esm_train_pca = pca.fit_transform(esm_train_scaled)
        esm_valid_pca = pca.transform(esm_valid_scaled)
        fused_train = np.hstack([trad_train, esm_train_pca])
        fused_valid = np.hstack([trad_valid, esm_valid_pca])
        if fused_train.shape != (576, 32 + n_components) or fused_valid.shape != (144, 32 + n_components):
            raise ValueError("Unexpected fused feature dimensions.")
        if not np.isfinite(fused_train).all() or not np.isfinite(fused_valid).all():
            raise ValueError("Non-finite fused values.")
        cache[(fold, n_components)] = (fused_train, fused_valid, y[train_idx], y[valid_idx])
        preprocessing_audit.append({
            "fold": fold, "pca_components": n_components,
            "traditional_scaler_fit_rows": int(trad_scaler.n_samples_seen_),
            "esm2_scaler_fit_rows": int(esm_scaler.n_samples_seen_),
            "pca_fit_rows": int(pca.n_samples_),
            "fusion_dimensions": 32 + n_components,
        })


def evaluate_cv(model_kind):
    rows = []
    configs = []
    if model_kind == "LR":
        for n in PCA_OPTIONS:
            for c in LR_C_VALUES:
                configs.append((n, c, None))
    else:
        for n in PCA_OPTIONS:
            for c in SVM_C_VALUES:
                for gamma in SVM_GAMMA_VALUES:
                    configs.append((n, c, gamma))

    for n_components, c_value, gamma in configs:
        fold_metrics = []
        warning_count = 0
        for fold in range(1, 6):
            X_train, X_valid, y_train, y_valid = cache[(fold, n_components)]
            if model_kind == "LR":
                model = LogisticRegression(
                    C=c_value, class_weight="balanced", max_iter=5000,
                    solver="liblinear", random_state=SEED,
                )
            else:
                model = SVC(
                    C=c_value, gamma=gamma, kernel="rbf", class_weight="balanced",
                    probability=True, random_state=SEED,
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_train, y_train)
            warning_count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            probability = model.predict_proba(X_valid)[:, 1]
            prediction = (probability >= THRESHOLD).astype(int)
            fold_metrics.append(metric_values(y_valid, probability, prediction))

        row = {
            "model": "Fusion Logistic Regression" if model_kind == "LR" else "Fusion RBF-SVM",
            "pca_components": n_components,
            "fusion_dimensions": 32 + n_components,
            "C": c_value,
            "gamma": gamma if gamma is not None else "not_applicable",
            "folds": 5,
            "cv_fit_warnings": warning_count,
        }
        for metric in ("AUROC", "AUPRC", "MCC", "F1", "Accuracy", "Precision", "Recall", "Specificity"):
            values = np.array([m[metric] for m in fold_metrics], dtype=float)
            row[f"mean_cv_{metric}"] = values.mean()
            row[f"sd_cv_{metric}"] = values.std(ddof=1)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC", "pca_components", "C"],
        ascending=[False, False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "selection_rank", np.arange(1, len(result) + 1))
    return result


lr_cv = evaluate_cv("LR")
svm_cv = evaluate_cv("SVM")
lr_cv.to_csv(LR_CV_OUTPUT, index=False)
svm_cv.to_csv(SVM_CV_OUTPUT, index=False)
lr_best = lr_cv.iloc[0]
svm_best = svm_cv.iloc[0]


def make_pipeline(model_kind, n_components, c_value, gamma=None):
    preprocess = ColumnTransformer([
        ("traditional", StandardScaler(), list(range(32))),
        ("esm2", Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(n_components), svd_solver="full")),
        ]), list(range(32, 1312))),
    ])
    if model_kind == "LR":
        classifier = LogisticRegression(
            C=float(c_value), class_weight="balanced", max_iter=5000,
            solver="liblinear", random_state=SEED,
        )
    else:
        classifier = SVC(
            C=float(c_value), gamma=gamma, kernel="rbf", class_weight="balanced",
            probability=True, random_state=SEED,
        )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def final_fit(model_kind, best_row, model_path, metrics_path, prediction_path, params_path):
    gamma = best_row["gamma"] if model_kind == "SVM" else None
    if model_kind == "SVM" and gamma != "scale":
        gamma = float(gamma)
    pipeline = make_pipeline(
        model_kind, int(best_row["pca_components"]), float(best_row["C"]), gamma,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_combined[dev_idx], y[dev_idx])
    probability = pipeline.predict_proba(X_combined[test_idx])[:, 1]
    prediction = (probability >= THRESHOLD).astype(int)
    values = metric_values(y[test_idx], probability, prediction)
    model_name = "Fusion Logistic Regression" if model_kind == "LR" else "Fusion RBF-SVM"
    metrics = pd.DataFrame([{
        "model": model_name, "representation": "Traditional_32_plus_ESM2_PCA",
        "pca_components": int(best_row["pca_components"]),
        "fusion_dimensions": 32 + int(best_row["pca_components"]),
        "best_C": float(best_row["C"]), "best_gamma": gamma if gamma is not None else "not_applicable",
        "selection_metric": "mean_cv_AUROC_then_AUPRC_then_MCC",
        "best_mean_cv_AUROC": float(best_row["mean_cv_AUROC"]),
        "decision_threshold": THRESHOLD, "development_n": 720,
        "test_n": 181, "test_active": int(y[test_idx].sum()),
        "test_inactive": int((y[test_idx] == 0).sum()),
        **{f"test_{k}": v for k, v in values.items()},
        "final_fit_warnings": len(caught),
    }])
    predictions = traditional.loc[test_idx, [
        "ID", "sequence", "class", "original_class", "label", "binary_class",
    ]].copy()
    predictions["predicted_probability"] = probability
    predictions["predicted_label"] = prediction
    predictions["split"] = "test"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    joblib.dump(pipeline, model_path)
    params = {
        "model": model_name, "pca_components": int(best_row["pca_components"]),
        "fusion_dimensions": 32 + int(best_row["pca_components"]),
        "C": float(best_row["C"]), "gamma": gamma,
        "class_weight": "balanced", "seed": SEED,
        "selection_metric": "mean_cv_AUROC_then_AUPRC_then_MCC",
        "development_rows": 720, "locked_test_rows": 181,
    }
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return pipeline, metrics, predictions


lr_model, lr_metrics, lr_predictions = final_fit(
    "LR", lr_best, LR_MODEL_OUTPUT, LR_METRICS_OUTPUT, LR_PRED_OUTPUT, LR_PARAMS_OUTPUT,
)
svm_model, svm_metrics, svm_predictions = final_fit(
    "SVM", svm_best, SVM_MODEL_OUTPUT, SVM_METRICS_OUTPUT, SVM_PRED_OUTPUT, SVM_PARAMS_OUTPUT,
)


def baseline_row(path, display_model, classifier, representation):
    row = pd.read_csv(path).iloc[0]
    return {
        "model": display_model, "classifier": classifier, "representation": representation,
        "AUROC": row["test_AUROC"], "AUPRC": row["test_AUPRC"],
        "MCC": row["test_MCC"], "F1": row["test_F1"],
        "Accuracy": row["test_accuracy"], "Precision": row["test_precision"],
        "Recall": row["test_recall"], "Specificity": row["test_specificity"],
        "TN": int(row["TN"]), "FP": int(row["FP"]), "FN": int(row["FN"]), "TP": int(row["TP"]),
        "frozen_existing_baseline": True,
    }


comparison_rows = [
    baseline_row(RESULTS_DIR / "step31_logistic_regression_test_metrics.csv", "Traditional Logistic Regression", "Logistic Regression", "Traditional"),
    baseline_row(RESULTS_DIR / "step48_esm2_logistic_regression_test_metrics.csv", "ESM-2 Logistic Regression", "Logistic Regression", "ESM-2"),
    baseline_row(RESULTS_DIR / "step32_svm_test_metrics.csv", "Traditional RBF-SVM", "RBF-SVM", "Traditional"),
    baseline_row(RESULTS_DIR / "step49_esm2_svm_test_metrics.csv", "ESM-2 RBF-SVM", "RBF-SVM", "ESM-2"),
]
for frame, classifier in ((lr_metrics, "Logistic Regression"), (svm_metrics, "RBF-SVM")):
    row = frame.iloc[0]
    comparison_rows.append({
        "model": row["model"], "classifier": classifier, "representation": "Fusion",
        **{m: row[f"test_{m}"] for m in ("AUROC", "AUPRC", "MCC", "F1", "Accuracy", "Precision", "Recall", "Specificity", "TN", "FP", "FN", "TP")},
        "frozen_existing_baseline": False,
    })
comparison = pd.DataFrame(comparison_rows)
comparison["representation_order"] = comparison["representation"].map({"Traditional": 1, "ESM-2": 2, "Fusion": 3})
comparison["classifier_order"] = comparison["classifier"].map({"Logistic Regression": 1, "RBF-SVM": 2})
comparison = comparison.sort_values(["classifier_order", "representation_order"]).drop(columns=["classifier_order", "representation_order"])
comparison.to_csv(COMPARISON_OUTPUT, index=False)

# Two-panel grouped comparison of the four main metrics.
metrics_to_plot = ["AUROC", "AUPRC", "MCC", "F1"]
colors = {"Traditional": "#4C78A8", "ESM-2": "#E45756", "Fusion": "#54A24B"}
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True, facecolor="white")
for ax, classifier, panel in zip(axes, ["Logistic Regression", "RBF-SVM"], ["A", "B"]):
    ax.set_facecolor("white")
    subset = comparison[comparison["classifier"] == classifier].set_index("representation").loc[["Traditional", "ESM-2", "Fusion"]]
    x = np.arange(len(metrics_to_plot))
    width = 0.24
    for offset, representation in zip([-width, 0, width], ["Traditional", "ESM-2", "Fusion"]):
        vals = subset.loc[representation, metrics_to_plot].to_numpy(dtype=float)
        bars = ax.bar(x + offset, vals, width, label=representation, color=colors[representation], edgecolor="black", linewidth=0.4)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, metrics_to_plot)
    ax.set_ylim(0.55, 1.04)
    ax.set_title(f"{panel}  {classifier}", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Locked-test metric")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Fusion versus Single-Representation Models", fontsize=17, fontweight="bold", y=0.98)
fig.text(0.5, 0.02, "Point estimates on the frozen 181-peptide test set; superiority is not inferred without paired uncertainty analysis.", ha="center", fontsize=9)
fig.tight_layout(rect=[0.04, 0.07, 0.99, 0.94])
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

audit = pd.DataFrame(preprocessing_audit)
qc = pd.DataFrame([{
    "total_rows": 901, "development_rows": 720, "test_rows": 181,
    "fixed_folds": 5, "fold_train_rows": 576, "fold_validation_rows": 144,
    "traditional_features": 32, "esm2_dimensions": 1280,
    "pca_candidates": "24,52,99", "lr_configurations": len(lr_cv),
    "svm_configurations": len(svm_cv), "total_cv_classifier_fits": 5 * (len(lr_cv) + len(svm_cv)),
    "preprocessing_fit_blocks": len(audit),
    "all_traditional_scaler_fit_rows_576": bool((audit.traditional_scaler_fit_rows == 576).all()),
    "all_esm2_scaler_fit_rows_576": bool((audit.esm2_scaler_fit_rows == 576).all()),
    "all_pca_fit_rows_576": bool((audit.pca_fit_rows == 576).all()),
    "fusion_dimensions_exact": bool((audit.fusion_dimensions == 32 + audit.pca_components).all()),
    "test_rows_used_during_cv": 0,
    "test_transformed_before_configuration_selection": False,
    "test_metric_used_for_selection": False,
    "row_sequence_label_alignment_exact": True,
    "all_transformed_values_finite": True,
    "lr_cv_rows": len(lr_cv), "svm_cv_rows": len(svm_cv),
    "comparison_rows": len(comparison),
    "selection_metric": "mean_cv_AUROC_then_AUPRC_then_MCC",
    "classification_threshold": THRESHOLD,
    "class_weight": "balanced", "seed": SEED,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nSelected fusion configurations:")
print("LR:", lr_best[["pca_components", "C", "mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC"]].to_dict())
print("SVM:", svm_best[["pca_components", "C", "gamma", "mean_cv_AUROC", "mean_cv_AUPRC", "mean_cv_MCC"]].to_dict())
print("\nMatched locked-test comparison:")
print(comparison.round(6).to_string(index=False))
print("\nSTEP 79 COMPLETED SUCCESSFULLY")
print("=" * 108)

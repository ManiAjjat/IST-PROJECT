from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
NOVELTY_FILE = RESULTS_DIR / "step66_test_to_development_sequence_similarity.csv"
PERFORMANCE_OUTPUT = RESULTS_DIR / "step67_sequence_novelty_performance.csv"
DELTA_OUTPUT = RESULTS_DIR / "step67_sequence_novelty_performance_deltas.csv"
SUBSET_OUTPUT = RESULTS_DIR / "step67_sequence_novelty_subset_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step67_sequence_novelty_qc.csv"
FIGURE1_PNG = FIGURES_DIR / "Step67_Sequence_Novelty_AUROC_AUPRC.png"
FIGURE1_PDF = FIGURES_DIR / "Step67_Sequence_Novelty_AUROC_AUPRC.pdf"
FIGURE2_PNG = FIGURES_DIR / "Step67_Sequence_Novelty_MCC_F1.png"
FIGURE2_PDF = FIGURES_DIR / "Step67_Sequence_Novelty_MCC_F1.pdf"

MODELS = [
    ("Traditional Logistic Regression", "Traditional", "Logistic Regression", "step31_logistic_regression_test_predictions.csv"),
    ("Traditional RBF-SVM", "Traditional", "RBF-SVM", "step32_svm_test_predictions.csv"),
    ("Traditional Random Forest", "Traditional", "Random Forest", "step33_random_forest_test_predictions.csv"),
    ("Traditional XGBoost", "Traditional", "XGBoost", "step34_xgboost_test_predictions.csv"),
    ("ESM-2 Logistic Regression", "ESM-2", "Logistic Regression", "step48_esm2_logistic_regression_test_predictions.csv"),
    ("ESM-2 RBF-SVM", "ESM-2", "RBF-SVM", "step49_esm2_svm_test_predictions.csv"),
    ("ESM-2 Random Forest", "ESM-2", "Random Forest", "step50_esm2_random_forest_test_predictions.csv"),
    ("ESM-2 XGBoost", "ESM-2", "XGBoost", "step51_esm2_xgboost_test_predictions.csv"),
]
SUBSETS = [
    ("all_test", "All test", None),
    ("novel_lt_0_95", "Similarity < 0.95", 0.95),
    ("novel_lt_0_90", "Similarity < 0.90", 0.90),
    ("novel_lt_0_80", "Similarity < 0.80", 0.80),
]


def calculate_metrics(y_true, probability):
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y_true, probability),
        "AUPRC": average_precision_score(y_true, probability),
        "MCC": matthews_corrcoef(y_true, predicted),
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


print("=" * 112)
print("STEP 67 - PERFORMANCE AS A FUNCTION OF SEQUENCE NOVELTY")
print("=" * 112)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
novelty = pd.read_csv(NOVELTY_FILE).sort_values("test_ID").reset_index(drop=True)
assert len(novelty) == 181 and novelty["test_ID"].is_unique
assert int(novelty["nearest_development_similarity"].ge(0.95).sum()) == 5
assert int(novelty["nearest_development_similarity"].ge(0.90).sum()) == 17
assert int(novelty["nearest_development_similarity"].ge(0.80).sum()) == 27

subset_rows = []
subset_ids = {}
for order, (subset, label, threshold) in enumerate(SUBSETS):
    selected = novelty if threshold is None else novelty.loc[novelty["nearest_development_similarity"] < threshold]
    ids = set(selected["test_ID"].astype(int))
    subset_ids[subset] = ids
    subset_rows.append({
        "subset_order": order, "subset": subset, "subset_label": label,
        "maximum_allowed_similarity_exclusive": threshold,
        "n": len(selected), "active": int(selected["test_label"].eq(1).sum()),
        "inactive": int(selected["test_label"].eq(0).sum()),
        "active_prevalence": selected["test_label"].mean(),
        "excluded_from_full_test": 181 - len(selected),
    })
subset_summary = pd.DataFrame(subset_rows)
assert subset_summary["n"].tolist() == [181, 176, 164, 154]
assert subset_ids["novel_lt_0_80"].issubset(subset_ids["novel_lt_0_90"])
assert subset_ids["novel_lt_0_90"].issubset(subset_ids["novel_lt_0_95"])
assert subset_ids["novel_lt_0_95"].issubset(subset_ids["all_test"])
subset_summary.to_csv(SUBSET_OUTPUT, index=False)

performance_rows = []
alignment_reference = None
prediction_threshold_consistent = True
for model_order, (model, representation, classifier, filename) in enumerate(MODELS):
    prediction = pd.read_csv(RESULTS_DIR / filename).sort_values("ID").reset_index(drop=True)
    assert len(prediction) == 181 and prediction["ID"].is_unique
    assert prediction["split"].eq("test").all()
    assert prediction["predicted_probability"].between(0, 1).all()
    assert np.isfinite(prediction["predicted_probability"]).all()
    aligned = prediction[["ID", "sequence", "label", "binary_class"]].merge(
        novelty[["test_ID", "test_sequence", "test_label", "test_class"]],
        left_on="ID", right_on="test_ID", how="inner", validate="one_to_one",
    )
    assert len(aligned) == 181
    assert aligned["sequence"].eq(aligned["test_sequence"]).all()
    assert aligned["label"].eq(aligned["test_label"]).all()
    assert aligned["binary_class"].eq(aligned["test_class"]).all()
    identity = prediction[["ID", "sequence", "label"]]
    if alignment_reference is None:
        alignment_reference = identity.copy()
    else:
        assert identity.equals(alignment_reference)
    prediction_threshold_consistent &= prediction["predicted_label"].eq(
        prediction["predicted_probability"].ge(0.5).astype(int)
    ).all()
    for subset_order, (subset, subset_label, threshold) in enumerate(SUBSETS):
        part = prediction.loc[prediction["ID"].isin(subset_ids[subset])].copy()
        metrics = calculate_metrics(part["label"].to_numpy(int), part["predicted_probability"].to_numpy(float))
        performance_rows.append({
            "model_order": model_order, "model": model, "representation": representation,
            "classifier": classifier, "subset_order": subset_order, "subset": subset,
            "subset_label": subset_label, "n": len(part),
            "active": int(part["label"].eq(1).sum()), "inactive": int(part["label"].eq(0).sum()),
            "active_prevalence": part["label"].mean(), "decision_threshold": 0.5,
            **metrics,
        })

performance = pd.DataFrame(performance_rows).sort_values(["model_order", "subset_order"]).reset_index(drop=True)
assert len(performance) == 32
assert performance.groupby("model")["subset"].nunique().eq(4).all()
assert performance[["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "Specificity", "F1"]].apply(
    lambda column: column.between(0, 1).all()
).all()
assert np.isfinite(performance[["AUROC", "AUPRC", "MCC", "Accuracy", "Precision", "Recall", "Specificity", "F1"]]).all().all()
assert (performance[["TN", "FP", "FN", "TP"]].sum(axis=1) == performance["n"]).all()
performance.to_csv(PERFORMANCE_OUTPUT, index=False)

metric_columns = ["AUROC", "AUPRC", "MCC", "Accuracy", "Precision", "Recall", "Specificity", "F1"]
full = performance.loc[performance["subset"].eq("all_test"), ["model"] + metric_columns].set_index("model")
delta = performance.copy()
for metric in metric_columns:
    delta[f"full_test_{metric}"] = delta["model"].map(full[metric])
    delta[f"delta_{metric}"] = delta[metric] - delta[f"full_test_{metric}"]
delta_columns = [
    "model_order", "model", "representation", "classifier", "subset_order", "subset", "subset_label",
    "n", "active", "inactive", "active_prevalence",
] + [value for metric in metric_columns for value in (f"full_test_{metric}", metric, f"delta_{metric}")]
delta = delta[delta_columns]
assert len(delta) == 32
assert np.allclose(delta.loc[delta["subset"].eq("all_test"), [f"delta_{m}" for m in metric_columns]], 0)
delta.to_csv(DELTA_OUTPUT, index=False)

colors = {"Logistic Regression": "#1B9E77", "RBF-SVM": "#D95F02", "Random Forest": "#7570B3", "XGBoost": "#E7298A"}
markers = {"Traditional": "o", "ESM-2": "s"}
linestyles = {"Traditional": "--", "ESM-2": "-"}
x = np.arange(4)

def make_metric_figure(metric_a, metric_b, png, pdf, title):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), facecolor="white", sharex=True)
    for ax, metric, panel in zip(axes, (metric_a, metric_b), ("A", "B")):
        for model, representation, classifier, _ in MODELS:
            part = performance.loc[performance["model"].eq(model)].sort_values("subset_order")
            ax.plot(x, part[metric], color=colors[classifier], marker=markers[representation],
                    linestyle=linestyles[representation], linewidth=1.8, markersize=6,
                    label=f"{representation} {classifier}")
        ax.set_xticks(x, ["All", "<0.95", "<0.90", "<0.80"])
        ax.set_xlabel("Maximum nearest-development similarity")
        ax.set_ylabel(metric)
        ax.set_title(f"{panel}  {metric}")
        ax.grid(color="#E0E0E0", linewidth=0.6); ax.set_axisbelow(True)
        for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.22, top=0.87, wspace=0.20)
    fig.savefig(png, dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

make_metric_figure("AUROC", "AUPRC", FIGURE1_PNG, FIGURE1_PDF,
                   "Frozen-model ranking performance across sequence-novel test subsets")
make_metric_figure("MCC", "F1", FIGURE2_PNG, FIGURE2_PDF,
                   "Frozen-model threshold performance across sequence-novel test subsets")

qc = pd.DataFrame([{
    "models": len(MODELS), "subsets": len(SUBSETS), "performance_rows": len(performance),
    "delta_rows": len(delta), "subset_summary_rows": len(subset_summary),
    "full_test_n": int(subset_summary.iloc[0]["n"]),
    "novel_lt_0_95_n": int(subset_summary.iloc[1]["n"]),
    "novel_lt_0_90_n": int(subset_summary.iloc[2]["n"]),
    "novel_lt_0_80_n": int(subset_summary.iloc[3]["n"]),
    "subset_nesting_valid": True, "prediction_alignment_all_models": True,
    "all_probabilities_finite_and_0_1": True,
    "saved_predictions_match_threshold_0_5": bool(prediction_threshold_consistent),
    "all_metric_values_finite": bool(np.isfinite(performance[metric_columns]).all().all()),
    "confusion_counts_sum_to_n": bool((performance[["TN", "FP", "FN", "TP"]].sum(axis=1) == performance["n"]).all()),
    "full_test_deltas_zero": bool(np.allclose(delta.loc[delta["subset"].eq("all_test"), [f"delta_{m}" for m in metric_columns]], 0)),
    "models_trained": False, "models_retrained": False,
    "thresholds_optimized": False, "decision_threshold_changed": False,
    "split_changed": False, "labels_changed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nSubset sizes:")
print(subset_summary[["subset_label", "n", "active", "inactive", "active_prevalence"]].to_string(index=False))
print("\nMost sequence-novel subset (<0.80):")
print(performance.loc[performance["subset"].eq("novel_lt_0_80"), [
    "model", "n", "active", "inactive", "AUROC", "AUPRC", "MCC", "F1"
]].sort_values("AUROC", ascending=False).round(6).to_string(index=False))
print("\nSTEP 67 COMPLETED SUCCESSFULLY")
print("=" * 112)

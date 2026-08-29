from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
SIMILARITY_INPUT = RESULTS_DIR / "step66_test_to_development_sequence_similarity.csv"

MODEL_OUTPUT = RESULTS_DIR / "step68_similarity_stratum_model_performance.csv"
CONSENSUS_OUTPUT = RESULTS_DIR / "step68_similarity_stratum_consensus_summary.csv"
PEPTIDE_OUTPUT = RESULTS_DIR / "step68_similarity_stratum_peptides.csv"
QC_OUTPUT = RESULTS_DIR / "step68_similarity_stratum_qc.csv"
ERROR_PNG = FIGURES_DIR / "Step68_Similarity_Stratum_Error_Rates.png"
ERROR_PDF = FIGURES_DIR / "Step68_Similarity_Stratum_Error_Rates.pdf"
DIFFICULTY_PNG = FIGURES_DIR / "Step68_Similarity_vs_Eight_Model_Difficulty.png"
DIFFICULTY_PDF = FIGURES_DIR / "Step68_Similarity_vs_Eight_Model_Difficulty.pdf"

MODELS = [
    ("Traditional Logistic Regression", "Traditional", "Logistic Regression", "trad_lr", "step31_logistic_regression_test_predictions.csv"),
    ("Traditional RBF-SVM", "Traditional", "RBF-SVM", "trad_svm", "step32_svm_test_predictions.csv"),
    ("Traditional Random Forest", "Traditional", "Random Forest", "trad_rf", "step33_random_forest_test_predictions.csv"),
    ("Traditional XGBoost", "Traditional", "XGBoost", "trad_xgb", "step34_xgboost_test_predictions.csv"),
    ("ESM-2 Logistic Regression", "ESM-2", "Logistic Regression", "esm2_lr", "step48_esm2_logistic_regression_test_predictions.csv"),
    ("ESM-2 RBF-SVM", "ESM-2", "RBF-SVM", "esm2_svm", "step49_esm2_svm_test_predictions.csv"),
    ("ESM-2 Random Forest", "ESM-2", "Random Forest", "esm2_rf", "step50_esm2_random_forest_test_predictions.csv"),
    ("ESM-2 XGBoost", "ESM-2", "XGBoost", "esm2_xgb", "step51_esm2_xgboost_test_predictions.csv"),
]

STRATA = [
    (0, "low_lt_0_80", "Low: <0.80", "<0.80"),
    (1, "moderate_0_80_to_lt_0_90", "Moderate: 0.80-<0.90", "0.80-<0.90"),
    (2, "high_0_90_to_lt_0_95", "High: 0.90-<0.95", "0.90-<0.95"),
    (3, "very_high_ge_0_95", "Very high: >=0.95", ">=0.95"),
]


def assign_stratum(value):
    if value < 0.80:
        return STRATA[0]
    if value < 0.90:
        return STRATA[1]
    if value < 0.95:
        return STRATA[2]
    return STRATA[3]


def safe_mean(values):
    return float(values.mean()) if len(values) else np.nan


print("=" * 112)
print("STEP 68 - PERFORMANCE STRATIFIED BY DEVELOPMENT-SET SEQUENCE SIMILARITY")
print("=" * 112)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

similarity = pd.read_csv(SIMILARITY_INPUT).sort_values("test_ID").reset_index(drop=True)
assert len(similarity) == 181 and similarity["test_ID"].is_unique
assert similarity["nearest_development_similarity"].between(0, 1).all()
assert np.isfinite(similarity["nearest_development_similarity"]).all()
assigned = similarity["nearest_development_similarity"].map(assign_stratum)
similarity[["stratum_order", "similarity_stratum", "similarity_stratum_label", "similarity_stratum_short"]] = pd.DataFrame(
    assigned.tolist(), index=similarity.index
)
stratum_counts = similarity.groupby("stratum_order", sort=True).size().tolist()
assert stratum_counts == [154, 10, 12, 5]
assert similarity["similarity_stratum"].notna().all()

peptides = similarity.copy()
model_rows = []
alignment_reference = None
threshold_consistent = True
wrong_columns = []
true_probability_columns = []

for model_order, (model, representation, classifier, short, filename) in enumerate(MODELS):
    prediction = pd.read_csv(RESULTS_DIR / filename).sort_values("ID").reset_index(drop=True)
    assert len(prediction) == 181 and prediction["ID"].is_unique
    assert prediction["split"].eq("test").all()
    assert prediction["predicted_probability"].between(0, 1).all()
    assert np.isfinite(prediction["predicted_probability"]).all()
    identity = prediction[["ID", "sequence", "label"]].copy()
    if alignment_reference is None:
        alignment_reference = identity
    else:
        assert identity.equals(alignment_reference)
    merged = peptides[["test_ID", "test_sequence", "test_label"]].merge(
        prediction[["ID", "sequence", "label", "predicted_probability", "predicted_label"]],
        left_on="test_ID", right_on="ID", how="left", validate="one_to_one",
    )
    assert len(merged) == 181
    assert merged["test_sequence"].eq(merged["sequence"]).all()
    assert merged["test_label"].eq(merged["label"]).all()
    frozen_predicted = merged["predicted_probability"].ge(0.5).astype(int)
    threshold_consistent &= merged["predicted_label"].eq(frozen_predicted).all()
    wrong = frozen_predicted.ne(merged["test_label"]).astype(int)
    true_probability = np.where(
        merged["test_label"].eq(1),
        merged["predicted_probability"],
        1.0 - merged["predicted_probability"],
    )
    wrong_column = f"{short}_wrong"
    probability_column = f"{short}_true_class_probability"
    peptides[wrong_column] = wrong.to_numpy()
    peptides[probability_column] = true_probability
    wrong_columns.append(wrong_column)
    true_probability_columns.append(probability_column)

    for stratum_order, stratum, stratum_label, stratum_short in STRATA:
        mask = peptides["similarity_stratum"].eq(stratum)
        part_wrong = peptides.loc[mask, wrong_column]
        part_probability = peptides.loc[mask, probability_column]
        correct_mask = part_wrong.eq(0)
        wrong_mask = part_wrong.eq(1)
        n = int(mask.sum())
        wrong_count = int(part_wrong.sum())
        model_rows.append({
            "model_order": model_order,
            "model": model,
            "representation": representation,
            "classifier": classifier,
            "stratum_order": stratum_order,
            "similarity_stratum": stratum,
            "similarity_stratum_label": stratum_label,
            "similarity_stratum_short": stratum_short,
            "n": n,
            "active": int(peptides.loc[mask, "test_label"].eq(1).sum()),
            "inactive": int(peptides.loc[mask, "test_label"].eq(0).sum()),
            "correct_count": n - wrong_count,
            "wrong_count": wrong_count,
            "accuracy": 1.0 - wrong_count / n,
            "error_rate": wrong_count / n,
            "mean_true_class_probability": part_probability.mean(),
            "median_true_class_probability": part_probability.median(),
            "mean_true_class_probability_when_correct": safe_mean(part_probability.loc[correct_mask]),
            "mean_true_class_probability_when_wrong": safe_mean(part_probability.loc[wrong_mask]),
            "decision_threshold": 0.5,
        })

model_performance = pd.DataFrame(model_rows).sort_values(["model_order", "stratum_order"]).reset_index(drop=True)
assert model_performance.shape[0] == 32
assert model_performance.groupby("model")["similarity_stratum"].nunique().eq(4).all()
assert (model_performance["correct_count"] + model_performance["wrong_count"] == model_performance["n"]).all()
assert np.allclose(model_performance["accuracy"] + model_performance["error_rate"], 1.0)
model_performance.to_csv(MODEL_OUTPUT, index=False)

peptides["traditional_models_wrong"] = peptides[wrong_columns[:4]].sum(axis=1).astype(int)
peptides["esm2_models_wrong"] = peptides[wrong_columns[4:]].sum(axis=1).astype(int)
peptides["total_models_wrong"] = peptides[wrong_columns].sum(axis=1).astype(int)
peptides["mean_true_class_probability_across_8_models"] = peptides[true_probability_columns].mean(axis=1)
peptides["consensus_8_of_8_error"] = peptides["total_models_wrong"].eq(8)
assert peptides["total_models_wrong"].between(0, 8).all()
assert peptides["traditional_models_wrong"].between(0, 4).all()
assert peptides["esm2_models_wrong"].between(0, 4).all()
assert peptides["mean_true_class_probability_across_8_models"].between(0, 1).all()

peptide_columns = [
    "test_ID", "test_sequence", "test_class", "test_label",
    "nearest_development_ID", "nearest_development_sequence",
    "nearest_development_class", "nearest_development_label", "same_true_class",
    "class_relation", "nearest_development_similarity", "nearest_tie_count",
    "stratum_order", "similarity_stratum", "similarity_stratum_label", "similarity_stratum_short",
] + wrong_columns + true_probability_columns + [
    "traditional_models_wrong", "esm2_models_wrong", "total_models_wrong",
    "mean_true_class_probability_across_8_models", "consensus_8_of_8_error",
]
peptides = peptides[peptide_columns].sort_values("test_ID").reset_index(drop=True)
peptides.to_csv(PEPTIDE_OUTPUT, index=False)


def summarize_consensus(part, stratum_order, stratum, stratum_label, stratum_short, scope, relation):
    n = len(part)
    row = {
        "stratum_order": stratum_order,
        "similarity_stratum": stratum,
        "similarity_stratum_label": stratum_label,
        "similarity_stratum_short": stratum_short,
        "aggregation_scope": scope,
        "nearest_neighbor_class_relation": relation,
        "n": n,
        "active": int(part["test_label"].eq(1).sum()),
        "inactive": int(part["test_label"].eq(0).sum()),
        "mean_total_models_wrong": part["total_models_wrong"].mean(),
        "median_total_models_wrong": part["total_models_wrong"].median(),
        "minimum_total_models_wrong": int(part["total_models_wrong"].min()),
        "maximum_total_models_wrong": int(part["total_models_wrong"].max()),
        "mean_traditional_models_wrong": part["traditional_models_wrong"].mean(),
        "mean_esm2_models_wrong": part["esm2_models_wrong"].mean(),
        "mean_true_class_probability_across_8_models": part["mean_true_class_probability_across_8_models"].mean(),
        "median_true_class_probability_across_8_models": part["mean_true_class_probability_across_8_models"].median(),
        "peptides_with_any_model_wrong": int(part["total_models_wrong"].gt(0).sum()),
        "peptides_with_any_model_wrong_rate": part["total_models_wrong"].gt(0).mean(),
        "consensus_8_of_8_error_count": int(part["consensus_8_of_8_error"].sum()),
    }
    for value in range(9):
        row[f"models_wrong_{value}_count"] = int(part["total_models_wrong"].eq(value).sum())
    return row


consensus_rows = []
for stratum_order, stratum, stratum_label, stratum_short in STRATA:
    stratum_part = peptides.loc[peptides["similarity_stratum"].eq(stratum)]
    consensus_rows.append(summarize_consensus(
        stratum_part, stratum_order, stratum, stratum_label, stratum_short, "all", "all"
    ))
    for relation in ("same", "opposite"):
        relation_part = stratum_part.loc[stratum_part["class_relation"].eq(relation)]
        assert len(relation_part) > 0
        consensus_rows.append(summarize_consensus(
            relation_part, stratum_order, stratum, stratum_label, stratum_short,
            "nearest_neighbor_class_relation", relation,
        ))

consensus = pd.DataFrame(consensus_rows).sort_values(["stratum_order", "aggregation_scope", "nearest_neighbor_class_relation"]).reset_index(drop=True)
assert len(consensus) == 12
assert consensus.loc[consensus["aggregation_scope"].eq("all"), "n"].tolist() == [154, 10, 12, 5]
assert (consensus[[f"models_wrong_{value}_count" for value in range(9)]].sum(axis=1) == consensus["n"]).all()
consensus.to_csv(CONSENSUS_OUTPUT, index=False)

# Figure 1: frozen error rates in mutually exclusive strata.
colors = {"Logistic Regression": "#1B9E77", "RBF-SVM": "#D95F02", "Random Forest": "#7570B3", "XGBoost": "#E7298A"}
markers = {"Traditional": "o", "ESM-2": "s"}
linestyles = {"Traditional": "--", "ESM-2": "-"}
x = np.arange(4)
fig, ax = plt.subplots(figsize=(10.8, 6.6), facecolor="white")
for model, representation, classifier, _, _ in MODELS:
    part = model_performance.loc[model_performance["model"].eq(model)].sort_values("stratum_order")
    ax.plot(
        x, part["error_rate"], color=colors[classifier], marker=markers[representation],
        linestyle=linestyles[representation], linewidth=1.8, markersize=7,
        label=f"{representation} {classifier}",
    )
ax.set_xticks(x, ["<0.80\n(n=154)", "0.80-<0.90\n(n=10)", "0.90-<0.95\n(n=12)", ">=0.95\n(n=5)"])
ax.set_xlabel("Nearest-development sequence-similarity stratum")
ax.set_ylabel("Error rate at frozen threshold 0.5")
ax.set_title("Frozen-model error rates across mutually exclusive similarity strata")
ax.grid(axis="y", color="#DEDEDE", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.text(0.01, 0.01, "Descriptive only; upper strata contain 10, 12, and 5 peptides.", fontsize=9, color="#444444")
fig.subplots_adjust(right=0.73, bottom=0.17, top=0.89)
fig.savefig(ERROR_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(ERROR_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: peptide-level similarity and eight-model difficulty.
relation_colors = {"same": "#377EB8", "opposite": "#E41A1C"}
fig, ax = plt.subplots(figsize=(11.2, 6.8), facecolor="white")
for relation in ("same", "opposite"):
    part = peptides.loc[peptides["class_relation"].eq(relation)]
    ax.scatter(
        part["nearest_development_similarity"], part["total_models_wrong"],
        s=38, alpha=0.64, color=relation_colors[relation], edgecolors="white", linewidths=0.45,
        label=f"Nearest analogue: {relation} class (n={len(part)})", zorder=2,
    )
hard = peptides.loc[peptides["consensus_8_of_8_error"]]
ax.scatter(
    hard["nearest_development_similarity"], hard["total_models_wrong"],
    s=135, facecolors="none", edgecolors="#111111", linewidths=1.8,
    label="8/8 consensus error", zorder=4,
)
annotation_positions = {
    40: (0.910, 8.47),
    48: (0.825, 8.47),
    56: (0.955, 7.05),
    68: (0.500, 8.47),
    145: (0.985, 8.47),
}
for _, row in hard.iterrows():
    position = annotation_positions[int(row["test_ID"])]
    ax.annotate(
        f"ID {int(row['test_ID'])}",
        (row["nearest_development_similarity"], row["total_models_wrong"]),
        xytext=position, textcoords="data", ha="center", fontsize=8, weight="bold", zorder=5,
        arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.7},
    )
for boundary, label in ((0.80, "0.80"), (0.90, "0.90"), (0.95, "0.95")):
    ax.axvline(boundary, color="#555555", linestyle=":", linewidth=1.1, zorder=1)
    ax.text(boundary, -0.60, label, ha="center", va="top", fontsize=8, color="#444444")
ax.set_xlabel("Nearest-development normalized edit similarity")
ax.set_ylabel("Number of frozen models wrong (0-8)")
ax.set_yticks(range(9))
ax.set_ylim(-0.75, 8.95)
ax.set_xlim(max(0, peptides["nearest_development_similarity"].min() - 0.025), 1.01)
ax.set_title("Sequence similarity versus eight-model peptide difficulty")
ax.grid(axis="y", color="#E3E3E3", linewidth=0.65)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.text(0.01, 0.01, "Points are descriptive locked-test observations; no relationship was fitted or tested.", fontsize=9, color="#444444")
fig.subplots_adjust(right=0.76, bottom=0.16, top=0.89)
fig.savefig(DIFFICULTY_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(DIFFICULTY_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

all_consensus = consensus.loc[consensus["aggregation_scope"].eq("all")].sort_values("stratum_order")
qc = pd.DataFrame([{
    "test_peptides": len(peptides),
    "unique_test_IDs": peptides["test_ID"].nunique(),
    "models": len(MODELS),
    "strata": len(STRATA),
    "expected_stratum_counts_match": stratum_counts == [154, 10, 12, 5],
    "strata_mutually_exclusive": True,
    "strata_cover_all_test_peptides": int(sum(stratum_counts)) == 181,
    "model_performance_rows": len(model_performance),
    "consensus_summary_rows": len(consensus),
    "peptide_rows": len(peptides),
    "prediction_alignment_all_models": True,
    "all_probabilities_finite_and_0_1": bool(np.isfinite(peptides[true_probability_columns]).all().all() and peptides[true_probability_columns].apply(lambda c: c.between(0, 1).all()).all()),
    "saved_predictions_match_threshold_0_5": bool(threshold_consistent),
    "model_counts_sum_to_stratum_n": bool((model_performance["correct_count"] + model_performance["wrong_count"] == model_performance["n"]).all()),
    "consensus_distributions_sum_to_n": bool((consensus[[f"models_wrong_{v}_count" for v in range(9)]].sum(axis=1) == consensus["n"]).all()),
    "same_opposite_rows_cover_each_stratum": bool(all(
        consensus.loc[(consensus["stratum_order"].eq(order)) & (consensus["aggregation_scope"].eq("nearest_neighbor_class_relation")), "n"].sum() == n
        for order, n in enumerate(stratum_counts)
    )),
    "consensus_8_of_8_errors": int(peptides["consensus_8_of_8_error"].sum()),
    "models_trained": False,
    "models_retrained": False,
    "significance_tests_performed": False,
    "decision_threshold_changed": False,
    "split_changed": False,
    "labels_changed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nMutually exclusive similarity strata:")
print(all_consensus[[
    "similarity_stratum_short", "n", "active", "inactive", "mean_total_models_wrong",
    "mean_true_class_probability_across_8_models", "consensus_8_of_8_error_count",
]].round(6).to_string(index=False))
print("\nSimilarity stratum x nearest-neighbor class relation:")
print(consensus.loc[consensus["aggregation_scope"].eq("nearest_neighbor_class_relation"), [
    "similarity_stratum_short", "nearest_neighbor_class_relation", "n",
    "mean_total_models_wrong", "mean_true_class_probability_across_8_models",
    "consensus_8_of_8_error_count",
]].round(6).to_string(index=False))
print("\n8/8 consensus errors:")
print(hard[[
    "test_ID", "test_class", "nearest_development_ID", "nearest_development_class",
    "class_relation", "nearest_development_similarity", "similarity_stratum_short",
]].to_string(index=False))
print("\nSTEP 68 COMPLETED SUCCESSFULLY")
print("=" * 112)

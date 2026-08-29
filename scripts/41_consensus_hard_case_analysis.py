from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"

RANKING_OUTPUT = RESULTS_DIR / "step59_consensus_hard_case_ranking.csv"
MANUSCRIPT_OUTPUT = RESULTS_DIR / "step59_consensus_hard_cases_manuscript.csv"
OVERLAP_OUTPUT = RESULTS_DIR / "step59_representation_error_overlap.csv"
QC_OUTPUT = RESULTS_DIR / "step59_consensus_hard_case_qc.csv"
HARD_MAP_PNG = FIGURES_DIR / "Step59_Eight_Model_Hard_Case_Map.png"
HARD_MAP_PDF = FIGURES_DIR / "Step59_Eight_Model_Hard_Case_Map.pdf"
OVERLAP_PNG = FIGURES_DIR / "Step59_Traditional_ESM2_Error_Overlap.png"
OVERLAP_PDF = FIGURES_DIR / "Step59_Traditional_ESM2_Error_Overlap.pdf"

MODEL_FILES = {
    "LR": (
        RESULTS_DIR / "step31_logistic_regression_test_predictions.csv",
        RESULTS_DIR / "step48_esm2_logistic_regression_test_predictions.csv",
    ),
    "SVM": (
        RESULTS_DIR / "step32_svm_test_predictions.csv",
        RESULTS_DIR / "step49_esm2_svm_test_predictions.csv",
    ),
    "RF": (
        RESULTS_DIR / "step33_random_forest_test_predictions.csv",
        RESULTS_DIR / "step50_esm2_random_forest_test_predictions.csv",
    ),
    "XGB": (
        RESULTS_DIR / "step34_xgboost_test_predictions.csv",
        RESULTS_DIR / "step51_esm2_xgboost_test_predictions.csv",
    ),
}
DESCRIPTORS = [
    "length", "molecular_weight", "net_charge_pH7_4", "isoelectric_point",
    "mean_eisenberg_hydrophobicity", "hydrophobic_moment", "boman_index",
]
EXPECTED_TRADITIONAL_ALL_WRONG = {40: 4, 48: 4, 56: 4, 67: 2, 68: 4, 145: 4, 149: 2}


def true_class_probability(label, active_probability):
    return active_probability if label == 1 else 1.0 - active_probability


def category(row):
    if row.traditional_wrong_count == 4 and row.esm2_wrong_count == 4:
        return "Consensus hard error"
    if row.total_wrong_count >= 6:
        return "Cross-representation difficult"
    if row.traditional_wrong_count >= 3 and row.esm2_wrong_count <= 1:
        return "Traditional-specific difficulty"
    if row.esm2_wrong_count >= 3 and row.traditional_wrong_count <= 1:
        return "ESM2-specific difficulty"
    if row.total_wrong_count >= 3:
        return "Mixed difficulty"
    return "Generally well classified"


print("=" * 110)
print("STEP 59 - EIGHT-MODEL CONSENSUS HARD-CASE RANKING")
print("=" * 110)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

loaded = {}
reference = None
for model, (traditional_path, esm2_path) in MODEL_FILES.items():
    traditional = pd.read_csv(traditional_path)
    esm2 = pd.read_csv(esm2_path)
    required = {"ID", "sequence", "label", "binary_class", "predicted_probability", "predicted_label", "split"}
    assert required.issubset(traditional.columns) and required.issubset(esm2.columns)
    assert len(traditional) == len(esm2) == 181
    alignment = traditional[["ID", "sequence", "label", "binary_class", "split"]].reset_index(drop=True)
    assert alignment.equals(esm2[["ID", "sequence", "label", "binary_class", "split"]].reset_index(drop=True))
    if reference is None:
        reference = alignment.copy()
    else:
        assert alignment.equals(reference)
    loaded[model] = {"traditional": traditional, "esm2": esm2}

assert reference["ID"].is_unique and reference["split"].eq("test").all()
features = pd.read_csv(FEATURE_FILE)
test_features = reference.merge(
    features[["ID", "sequence", "label", "binary_class", *DESCRIPTORS]],
    on=["ID", "sequence", "label", "binary_class"], how="left", validate="one_to_one",
)
assert len(test_features) == 181
assert np.isfinite(test_features[DESCRIPTORS].to_numpy(float)).all()
means = test_features[DESCRIPTORS].mean()
sds = test_features[DESCRIPTORS].std(ddof=1)
for descriptor in DESCRIPTORS:
    test_features[f"z_{descriptor}"] = (test_features[descriptor] - means[descriptor]) / sds[descriptor]
test_features["mean_absolute_descriptor_z"] = test_features[[f"z_{d}" for d in DESCRIPTORS]].abs().mean(axis=1)
test_features["maximum_absolute_descriptor_z"] = test_features[[f"z_{d}" for d in DESCRIPTORS]].abs().max(axis=1)
test_features["most_extreme_descriptor"] = test_features.apply(
    lambda row: max(DESCRIPTORS, key=lambda d: abs(row[f"z_{d}"])), axis=1
)

rows = []
for index, base in reference.iterrows():
    label = int(base["label"])
    row = {
        "ID": int(base["ID"]), "sequence": base["sequence"],
        "true_class": base["binary_class"], "y_true": label,
    }
    traditional_true_probabilities = []
    esm2_true_probabilities = []
    for model in MODEL_FILES:
        for representation in ("traditional", "esm2"):
            prediction_row = loaded[model][representation].iloc[index]
            probability = float(prediction_row["predicted_probability"])
            prediction = int(prediction_row["predicted_label"])
            assert 0 <= probability <= 1 and prediction == int(probability >= 0.5)
            correct = prediction == label
            true_probability = true_class_probability(label, probability)
            row[f"{model}_{representation}_probability_active"] = probability
            row[f"{model}_{representation}_true_class_probability"] = true_probability
            row[f"{model}_{representation}_correct"] = correct
            if representation == "traditional":
                traditional_true_probabilities.append(true_probability)
            else:
                esm2_true_probabilities.append(true_probability)
    traditional_correct = sum(row[f"{m}_traditional_correct"] for m in MODEL_FILES)
    esm2_correct = sum(row[f"{m}_esm2_correct"] for m in MODEL_FILES)
    row["traditional_correct_count"] = traditional_correct
    row["esm2_correct_count"] = esm2_correct
    row["traditional_wrong_count"] = 4 - traditional_correct
    row["esm2_wrong_count"] = 4 - esm2_correct
    row["total_wrong_count"] = row["traditional_wrong_count"] + row["esm2_wrong_count"]
    row["traditional_mean_true_class_probability"] = float(np.mean(traditional_true_probabilities))
    row["esm2_mean_true_class_probability"] = float(np.mean(esm2_true_probabilities))
    row["all_models_mean_true_class_probability"] = float(np.mean(
        traditional_true_probabilities + esm2_true_probabilities
    ))
    row["true_class_probability_gain"] = (
        row["esm2_mean_true_class_probability"] - row["traditional_mean_true_class_probability"]
    )
    row["traditional_mean_margin"] = row["traditional_mean_true_class_probability"] - 0.5
    row["esm2_mean_margin"] = row["esm2_mean_true_class_probability"] - 0.5
    row["representation_disagreement"] = abs(row["true_class_probability_gain"])
    row["wrong_count_difference_esm2_minus_traditional"] = (
        row["esm2_wrong_count"] - row["traditional_wrong_count"]
    )
    rows.append(row)

ranking = pd.DataFrame(rows).merge(
    test_features[["ID", "mean_absolute_descriptor_z", "maximum_absolute_descriptor_z", "most_extreme_descriptor"]],
    on="ID", how="left", validate="one_to_one",
)
ranking["difficulty_category"] = ranking.apply(category, axis=1)
ranking = ranking.sort_values(
    ["total_wrong_count", "all_models_mean_true_class_probability", "mean_absolute_descriptor_z", "ID"],
    ascending=[False, True, False, True],
).reset_index(drop=True)
ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
assert len(ranking) == 181 and ranking["rank"].tolist() == list(range(1, 182)) and ranking["ID"].is_unique

preferred_columns = [
    "rank", "ID", "sequence", "true_class", "y_true",
    "traditional_wrong_count", "esm2_wrong_count", "total_wrong_count",
    "traditional_correct_count", "esm2_correct_count",
    "traditional_mean_true_class_probability", "esm2_mean_true_class_probability",
    "all_models_mean_true_class_probability", "true_class_probability_gain",
    "traditional_mean_margin", "esm2_mean_margin", "representation_disagreement",
    "wrong_count_difference_esm2_minus_traditional",
    "LR_traditional_correct", "SVM_traditional_correct", "RF_traditional_correct", "XGB_traditional_correct",
    "LR_esm2_correct", "SVM_esm2_correct", "RF_esm2_correct", "XGB_esm2_correct",
    "mean_absolute_descriptor_z", "maximum_absolute_descriptor_z", "most_extreme_descriptor",
    "difficulty_category",
]
probability_columns = [
    column for column in ranking.columns
    if any(column.startswith(f"{model}_") for model in MODEL_FILES)
    and ("probability_active" in column or "_true_class_probability" in column)
]
ranking[[*preferred_columns, *probability_columns]].to_csv(RANKING_OUTPUT, index=False)

manuscript = ranking.loc[ranking["total_wrong_count"] >= 3, preferred_columns].copy()
manuscript.to_csv(MANUSCRIPT_OUTPUT, index=False)

overlap_matrix = pd.crosstab(ranking["traditional_wrong_count"], ranking["esm2_wrong_count"])
overlap_matrix = overlap_matrix.reindex(index=range(5), columns=range(5), fill_value=0)
overlap_table = overlap_matrix.copy()
overlap_table.columns = [f"esm2_wrong_{value}" for value in overlap_table.columns]
overlap_table.index.name = "traditional_wrong_count"
overlap_table.reset_index().to_csv(OVERLAP_OUTPUT, index=False)
assert int(overlap_matrix.to_numpy().sum()) == 181

# Independent post-reconstruction QC against Steps 56-58 expectations.
traditional_all_wrong = ranking.loc[ranking["traditional_wrong_count"] == 4, ["ID", "esm2_wrong_count"]]
reconstructed_expected = dict(zip(traditional_all_wrong["ID"], traditional_all_wrong["esm2_wrong_count"]))
expected_qc_passed = reconstructed_expected == EXPECTED_TRADITIONAL_ALL_WRONG
assert expected_qc_passed
consensus_ids = ranking.loc[
    ranking["traditional_wrong_count"].eq(4) & ranking["esm2_wrong_count"].eq(4), "ID"
].tolist()
assert set(consensus_ids) == {40, 48, 56, 68, 145}

probability_values = ranking[probability_columns].to_numpy(float)
qc_df = pd.DataFrame([{
    "peptides": len(ranking),
    "models_per_peptide": 8,
    "traditional_models": 4,
    "esm2_models": 4,
    "traditional_wrong_min": int(ranking["traditional_wrong_count"].min()),
    "traditional_wrong_max": int(ranking["traditional_wrong_count"].max()),
    "esm2_wrong_min": int(ranking["esm2_wrong_count"].min()),
    "esm2_wrong_max": int(ranking["esm2_wrong_count"].max()),
    "total_wrong_min": int(ranking["total_wrong_count"].min()),
    "total_wrong_max": int(ranking["total_wrong_count"].max()),
    "all_wrong_count_sums_match": bool((ranking["total_wrong_count"] == ranking["traditional_wrong_count"] + ranking["esm2_wrong_count"]).all()),
    "all_probabilities_finite": bool(np.isfinite(probability_values).all()),
    "all_probabilities_within_0_1": bool(((probability_values >= 0) & (probability_values <= 1)).all()),
    "overlap_matrix_sum": int(overlap_matrix.to_numpy().sum()),
    "ranking_unique_1_to_181": ranking["rank"].tolist() == list(range(1, 182)),
    "ids_unique": bool(ranking["ID"].is_unique),
    "traditional_all_wrong_expected_qc": expected_qc_passed,
    "consensus_hard_errors": len(consensus_ids),
    "manuscript_hard_cases": len(manuscript),
    "models_retrained": False,
    "hyperparameters_selected": False,
    "threshold_optimized": False,
    "test_set_model_selection": False,
}])
qc_df.to_csv(QC_OUTPUT, index=False)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 11,
    "axes.labelsize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Eight-model map for all peptides with at least three errors.
hard = manuscript.copy()
correct_columns = [
    "LR_traditional_correct", "SVM_traditional_correct", "RF_traditional_correct", "XGB_traditional_correct",
    "LR_esm2_correct", "SVM_esm2_correct", "RF_esm2_correct", "XGB_esm2_correct",
]
matrix = hard[correct_columns].to_numpy(dtype=int)
fig_height = max(5.2, 0.32 * len(hard) + 2.1)
fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
ax.imshow(matrix, cmap=ListedColormap(["#D55E00", "#009E73"]), vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(8), ["LR", "SVM", "RF", "XGB", "LR", "SVM", "RF", "XGB"])
ax.set_yticks(range(len(hard)), [f"#{r.rank} | ID {r.ID}" for r in hard.itertuples()])
for i in range(len(hard)):
    for j in range(8):
        ax.text(j, i, "Correct" if matrix[i, j] else "Wrong", ha="center", va="center",
                color="white", fontweight="bold", fontsize=7)
    record = hard.iloc[i]
    ax.text(8.2, i, f"Wrong {record.total_wrong_count}/8 | {record.true_class}",
            ha="left", va="center", fontsize=8)
ax.axvline(3.5, color="white", linewidth=5)
ax.text(1.5, -1.0, "Traditional descriptors", ha="center", va="center", fontweight="bold")
ax.text(5.5, -1.0, "ESM-2", ha="center", va="center", fontweight="bold")
ax.set_xlim(-0.5, 10.0)
ax.set_title("Eight-model outcomes for consensus difficult test peptides", pad=24)
ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(hard), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.4)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(HARD_MAP_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(HARD_MAP_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Representation error-overlap matrix.
fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
image = ax.imshow(overlap_matrix.to_numpy(), cmap="Blues", vmin=0, aspect="equal")
for i in range(5):
    for j in range(5):
        count = int(overlap_matrix.iloc[i, j])
        color = "white" if count > overlap_matrix.to_numpy().max() * 0.45 else "#222222"
        ax.text(j, i, str(count), ha="center", va="center", color=color, fontweight="bold", fontsize=11)
ax.set_xticks(range(5), range(5))
ax.set_yticks(range(5), range(5))
ax.set_xlabel("ESM-2 models wrong")
ax.set_ylabel("Traditional models wrong")
ax.set_title("Overlap of representation-level test errors")
colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
colorbar.set_label("Peptide count")
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
fig.savefig(OVERLAP_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(OVERLAP_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nTop ranked hard cases:")
print(ranking[preferred_columns[:14] + ["mean_absolute_descriptor_z", "most_extreme_descriptor", "difficulty_category"]].head(20).round(6).to_string(index=False))
print("\nDifficulty categories:")
print(ranking["difficulty_category"].value_counts().to_string())
print("\nRepresentation error-overlap matrix:")
print(overlap_matrix.to_string())
print("\nConsensus 8/8 error IDs:", consensus_ids)
print("Manuscript hard-case rows:", len(manuscript))
print("\nSTEP 59 COMPLETED SUCCESSFULLY")
print("=" * 110)

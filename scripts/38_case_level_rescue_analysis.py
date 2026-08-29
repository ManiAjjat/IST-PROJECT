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

TRANSITION_TABLE_OUTPUT = RESULTS_DIR / "step56_case_level_transition_table.csv"
CLASSIFIER_SUMMARY_OUTPUT = RESULTS_DIR / "step56_classifier_transition_summary.csv"
UNANIMOUS_ERROR_OUTPUT = RESULTS_DIR / "step56_traditional_unanimous_errors_esm2_rescue.csv"
UNANIMOUS_CORRECT_REGRESSION_OUTPUT = (
    RESULTS_DIR / "step56_traditional_unanimous_correct_esm2_regressions.csv"
)
SUMMARY_OUTPUT = RESULTS_DIR / "step56_case_level_summary.csv"
FIGURE_PNG = FIGURES_DIR / "Step56_Traditional_Error_ESM2_Rescue_Map.png"
FIGURE_PDF = FIGURES_DIR / "Step56_Traditional_Error_ESM2_Rescue_Map.pdf"

MODEL_FILES = {
    "Logistic Regression": (
        "lr", RESULTS_DIR / "step31_logistic_regression_test_predictions.csv",
        RESULTS_DIR / "step48_esm2_logistic_regression_test_predictions.csv",
    ),
    "RBF-SVM": (
        "svm", RESULTS_DIR / "step32_svm_test_predictions.csv",
        RESULTS_DIR / "step49_esm2_svm_test_predictions.csv",
    ),
    "Random Forest": (
        "rf", RESULTS_DIR / "step33_random_forest_test_predictions.csv",
        RESULTS_DIR / "step50_esm2_random_forest_test_predictions.csv",
    ),
    "XGBoost": (
        "xgb", RESULTS_DIR / "step34_xgboost_test_predictions.csv",
        RESULTS_DIR / "step51_esm2_xgboost_test_predictions.csv",
    ),
}
EXPECTED_UNANIMOUS_ERROR_IDS = [40, 48, 56, 67, 68, 145, 149]
TRANSITIONS = {
    (True, True): "stable_correct",
    (False, True): "rescue",
    (True, False): "regression",
    (False, False): "persistent_error",
}


print("=" * 108)
print("STEP 56 - CASE-LEVEL RESCUE AND REGRESSION ANALYSIS")
print("=" * 108)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

transition_df = None
classifier_summary_rows = []
for classifier, (prefix, traditional_path, esm2_path) in MODEL_FILES.items():
    traditional = pd.read_csv(traditional_path)
    esm2 = pd.read_csv(esm2_path)
    required = {"ID", "sequence", "class", "original_class", "label", "binary_class",
                "predicted_probability", "predicted_label", "split"}
    assert required.issubset(traditional.columns) and required.issubset(esm2.columns)
    assert len(traditional) == len(esm2) == 181
    align_columns = ["ID", "sequence", "class", "original_class", "label", "binary_class", "split"]
    assert traditional[align_columns].reset_index(drop=True).equals(
        esm2[align_columns].reset_index(drop=True)
    )
    if transition_df is None:
        transition_df = traditional[align_columns].copy().reset_index(drop=True)
    else:
        assert traditional[align_columns].reset_index(drop=True).equals(transition_df[align_columns])

    label = traditional["label"].to_numpy(int)
    traditional_prediction = traditional["predicted_label"].to_numpy(int)
    esm2_prediction = esm2["predicted_label"].to_numpy(int)
    traditional_correct = traditional_prediction == label
    esm2_correct = esm2_prediction == label
    transitions = [TRANSITIONS[(bool(t), bool(e))] for t, e in zip(traditional_correct, esm2_correct)]

    transition_df[f"{prefix}_traditional_probability"] = traditional["predicted_probability"].to_numpy(float)
    transition_df[f"{prefix}_traditional_prediction"] = traditional_prediction
    transition_df[f"{prefix}_traditional_correct"] = traditional_correct
    transition_df[f"{prefix}_esm2_probability"] = esm2["predicted_probability"].to_numpy(float)
    transition_df[f"{prefix}_esm2_prediction"] = esm2_prediction
    transition_df[f"{prefix}_esm2_correct"] = esm2_correct
    transition_df[f"{prefix}_transition"] = transitions

    counts = pd.Series(transitions).value_counts()
    stable_correct = int(counts.get("stable_correct", 0))
    rescue = int(counts.get("rescue", 0))
    regression = int(counts.get("regression", 0))
    persistent_error = int(counts.get("persistent_error", 0))
    traditional_correct_count = int(traditional_correct.sum())
    esm2_correct_count = int(esm2_correct.sum())
    assert sum((stable_correct, rescue, regression, persistent_error)) == 181
    assert stable_correct + regression == traditional_correct_count
    assert stable_correct + rescue == esm2_correct_count
    assert rescue - regression == esm2_correct_count - traditional_correct_count
    classifier_summary_rows.append({
        "classifier": classifier,
        "test_peptides": 181,
        "stable_correct": stable_correct,
        "rescued_by_esm2": rescue,
        "regressed_with_esm2": regression,
        "persistent_error": persistent_error,
        "traditional_correct": traditional_correct_count,
        "esm2_correct": esm2_correct_count,
        "correct_count_change": esm2_correct_count - traditional_correct_count,
        "rescue_minus_regression": rescue - regression,
        "traditional_accuracy": traditional_correct_count / 181,
        "esm2_accuracy": esm2_correct_count / 181,
        "accuracy_delta": (esm2_correct_count - traditional_correct_count) / 181,
        "transition_qc_passed": True,
    })

prefixes = [details[0] for details in MODEL_FILES.values()]
transition_df["traditional_models_correct"] = transition_df[
    [f"{p}_traditional_correct" for p in prefixes]
].sum(axis=1).astype(int)
transition_df["esm2_models_correct"] = transition_df[
    [f"{p}_esm2_correct" for p in prefixes]
].sum(axis=1).astype(int)
transition_df["esm2_minus_traditional_models_correct"] = (
    transition_df["esm2_models_correct"] - transition_df["traditional_models_correct"]
)
transition_df["traditional_all_wrong"] = transition_df["traditional_models_correct"].eq(0)
transition_df["traditional_all_correct"] = transition_df["traditional_models_correct"].eq(4)
transition_df["any_esm2_rescue"] = transition_df[
    [f"{p}_transition" for p in prefixes]
].eq("rescue").any(axis=1)
transition_df["any_esm2_regression"] = transition_df[
    [f"{p}_transition" for p in prefixes]
].eq("regression").any(axis=1)
assert transition_df["ID"].is_unique and transition_df["split"].eq("test").all()
transition_df.to_csv(TRANSITION_TABLE_OUTPUT, index=False)

classifier_summary = pd.DataFrame(classifier_summary_rows)
classifier_summary.to_csv(CLASSIFIER_SUMMARY_OUTPUT, index=False)

unanimous_errors = transition_df.loc[transition_df["traditional_all_wrong"]].copy()
assert unanimous_errors["ID"].tolist() == EXPECTED_UNANIMOUS_ERROR_IDS
error_columns = ["ID", "sequence", "label", "binary_class"]
for prefix in prefixes:
    error_columns.extend([
        f"{prefix}_esm2_probability", f"{prefix}_esm2_prediction",
        f"{prefix}_esm2_correct", f"{prefix}_transition",
    ])
error_columns.extend(["esm2_models_correct", "any_esm2_rescue"])
unanimous_error_output = unanimous_errors[error_columns].copy()
unanimous_error_output["esm2_rescue_fraction"] = (
    unanimous_error_output["esm2_models_correct"].astype(str) + "/4"
)
unanimous_error_output.to_csv(UNANIMOUS_ERROR_OUTPUT, index=False)

traditional_all_correct = transition_df.loc[transition_df["traditional_all_correct"]].copy()
unanimous_correct_regressions = traditional_all_correct.loc[
    traditional_all_correct["any_esm2_regression"]
].copy()
reverse_columns = ["ID", "sequence", "label", "binary_class"]
for prefix in prefixes:
    reverse_columns.extend([
        f"{prefix}_esm2_probability", f"{prefix}_esm2_prediction",
        f"{prefix}_esm2_correct", f"{prefix}_transition",
    ])
reverse_columns.extend(["esm2_models_correct", "any_esm2_regression"])
unanimous_correct_regressions[reverse_columns].to_csv(
    UNANIMOUS_CORRECT_REGRESSION_OUTPUT, index=False
)

rescued_by_at_least_one = int(unanimous_errors["esm2_models_correct"].ge(1).sum())
rescued_by_at_least_two = int(unanimous_errors["esm2_models_correct"].ge(2).sum())
rescued_by_majority = int(unanimous_errors["esm2_models_correct"].ge(3).sum())
rescued_by_all_four = int(unanimous_errors["esm2_models_correct"].eq(4).sum())
still_wrong_by_all_four = int(unanimous_errors["esm2_models_correct"].eq(0).sum())
total_model_rescues = int(classifier_summary["rescued_by_esm2"].sum())
total_model_regressions = int(classifier_summary["regressed_with_esm2"].sum())

summary_df = pd.DataFrame([{
    "test_peptides": len(transition_df),
    "traditional_unanimous_errors": len(unanimous_errors),
    "expected_unanimous_error_ids_match": unanimous_errors["ID"].tolist() == EXPECTED_UNANIMOUS_ERROR_IDS,
    "rescued_by_at_least_one_esm2_model": rescued_by_at_least_one,
    "rescued_by_at_least_two_esm2_models": rescued_by_at_least_two,
    "rescued_by_majority_esm2_models": rescued_by_majority,
    "rescued_by_all_four_esm2_models": rescued_by_all_four,
    "still_wrong_under_all_four_esm2_models": still_wrong_by_all_four,
    "traditional_unanimous_correct": len(traditional_all_correct),
    "traditional_unanimous_correct_with_esm2_regression": len(unanimous_correct_regressions),
    "total_classifier_level_rescues": total_model_rescues,
    "total_classifier_level_regressions": total_model_regressions,
    "net_rescue_minus_regression": total_model_rescues - total_model_regressions,
    "all_transition_partitions_181": bool(classifier_summary[[
        "stable_correct", "rescued_by_esm2", "regressed_with_esm2", "persistent_error"
    ]].sum(axis=1).eq(181).all()),
    "all_accuracy_change_qc_passed": bool(classifier_summary["transition_qc_passed"].all()),
    "models_loaded_or_retrained": False,
}])
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

# Seven-case rescue map: 1 = ESM-2 correct, 0 = ESM-2 wrong.
matrix = unanimous_errors[[f"{p}_esm2_correct" for p in prefixes]].to_numpy(dtype=int)
row_labels = [f"ID {row.ID} | {'Active' if row.label == 1 else 'Inactive'}" for row in unanimous_errors.itertuples()]
column_labels = ["LR", "SVM", "RF", "XGBoost"]
fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
cmap = ListedColormap(["#D55E00", "#009E73"])
ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(4), column_labels)
ax.set_yticks(np.arange(7), row_labels)
ax.set_xlabel("Frozen ESM-2 classifier")
ax.set_title("ESM-2 outcomes for seven unanimous traditional-model errors", pad=12)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, "Correct" if matrix[i, j] else "Wrong", ha="center", va="center",
                color="white", fontweight="bold", fontsize=8)
    ax.text(4.15, i, f"{matrix[i].sum()}/4", ha="left", va="center", fontsize=9, fontweight="bold")
ax.text(4.15, -0.65, "ESM-2 correct", ha="left", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlim(-0.5, 5.0)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nClassifier transitions:")
print(classifier_summary.to_string(index=False))
print("\nSeven traditional-unanimous errors:")
print(unanimous_error_output[["ID", "sequence", "label", "esm2_models_correct", "esm2_rescue_fraction"]].to_string(index=False))
print("\nTraditional unanimous errors:", len(unanimous_errors))
print("Rescued by >=1 ESM-2 model:", rescued_by_at_least_one)
print("Rescued by >=2 ESM-2 models:", rescued_by_at_least_two)
print("Rescued by majority of ESM-2 models:", rescued_by_majority)
print("Rescued by all four ESM-2 models:", rescued_by_all_four)
print("Still wrong under all four ESM-2 models:", still_wrong_by_all_four)
print("Traditional-all-correct peptides:", len(traditional_all_correct))
print("Traditional-all-correct cases with >=1 ESM regression:", len(unanimous_correct_regressions))
print("Total classifier-level rescues:", total_model_rescues)
print("Total classifier-level regressions:", total_model_regressions)
print("Net rescue minus regression:", total_model_rescues - total_model_regressions)
print("\nTransition table:", TRANSITION_TABLE_OUTPUT)
print("Classifier transition summary:", CLASSIFIER_SUMMARY_OUTPUT)
print("Seven-case rescue table:", UNANIMOUS_ERROR_OUTPUT)
print("Reverse regression table:", UNANIMOUS_CORRECT_REGRESSION_OUTPUT)
print("Summary:", SUMMARY_OUTPUT)
print("Rescue-map figure:", FIGURE_PNG)
print("\nSTEP 56 COMPLETED SUCCESSFULLY")
print("=" * 108)

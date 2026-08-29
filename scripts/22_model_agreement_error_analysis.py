from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"
AGREEMENT_OUTPUT = RESULTS_DIR / "step40_test_prediction_agreement.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "step40_model_agreement_summary.csv"
MISCLASSIFIED_OUTPUT = RESULTS_DIR / "step40_consistently_misclassified_peptides.csv"
MODEL_ERROR_OUTPUT = RESULTS_DIR / "step40_model_specific_errors.csv"
PNG_OUTPUT = FIGURE_DIR / "Step40_Traditional_Model_Agreement.png"
PDF_OUTPUT = FIGURE_DIR / "Step40_Traditional_Model_Agreement.pdf"

model_files = {
    "Logistic Regression": RESULTS_DIR / "step31_logistic_regression_test_predictions.csv",
    "RBF-SVM": RESULTS_DIR / "step32_svm_test_predictions.csv",
    "Random Forest": RESULTS_DIR / "step33_random_forest_test_predictions.csv",
    "XGBoost": RESULTS_DIR / "step34_xgboost_test_predictions.csv",
}
model_prefixes = {
    "Logistic Regression": "logistic_regression",
    "RBF-SVM": "rbf_svm",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
required_columns = {
    "ID", "sequence", "class", "original_class", "label", "binary_class",
    "predicted_probability", "predicted_label", "split",
}

tables = {}
for model, prediction_file in model_files.items():
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
    table = pd.read_csv(prediction_file).sort_values("ID").reset_index(drop=True)
    missing = required_columns.difference(table.columns)
    if missing:
        raise ValueError(f"{model} prediction file is missing columns: {sorted(missing)}")
    if len(table) != 181 or table["ID"].duplicated().any():
        raise ValueError(f"{model} must contain 181 unique test peptide IDs.")
    if not table["split"].eq("test").all():
        raise ValueError(f"{model} contains non-test rows.")
    if not table["predicted_probability"].between(0, 1).all():
        raise ValueError(f"{model} contains invalid probabilities.")
    threshold_predictions = (table["predicted_probability"] >= 0.5).astype(int)
    if not threshold_predictions.equals(table["predicted_label"].astype(int)):
        raise ValueError(f"{model} predictions are inconsistent with the 0.5 threshold.")
    tables[model] = table

reference = tables["Logistic Regression"]
metadata = ["ID", "sequence", "class", "original_class", "label", "binary_class", "split"]
for model, table in tables.items():
    for column in ["ID", "sequence", "label", "binary_class", "split"]:
        if not table[column].equals(reference[column]):
            raise ValueError(f"{model} is not aligned with the reference {column} values.")

agreement = reference[metadata].copy()
correct_columns = []
prediction_columns = []
for model, table in tables.items():
    prefix = model_prefixes[model]
    probability_column = f"{prefix}_probability"
    prediction_column = f"{prefix}_prediction"
    correct_column = f"{prefix}_correct"
    agreement[probability_column] = table["predicted_probability"].to_numpy(dtype=float)
    agreement[prediction_column] = table["predicted_label"].to_numpy(dtype=int)
    agreement[correct_column] = (
        agreement[prediction_column].to_numpy() == agreement["label"].to_numpy()
    ).astype(int)
    prediction_columns.append(prediction_column)
    correct_columns.append(correct_column)

agreement["models_correct"] = agreement[correct_columns].sum(axis=1).astype(int)
agreement["models_wrong"] = len(model_files) - agreement["models_correct"]
agreement["agreement_category"] = agreement["models_correct"].map(
    {
        0: "0 models correct (all wrong)",
        1: "1 model correct",
        2: "2 models correct",
        3: "3 models correct",
        4: "4 models correct (all correct)",
    }
)
agreement["unanimous_prediction"] = agreement[prediction_columns].nunique(axis=1).eq(1)
agreement["prediction_pattern_LR_SVM_RF_XGB"] = agreement[prediction_columns].astype(str).agg("".join, axis=1)

expected_counts = {0, 1, 2, 3, 4}
if not set(agreement["models_correct"].unique()).issubset(expected_counts):
    raise RuntimeError("Invalid number of correct models encountered.")

summary_rows = []
for models_correct in range(4, -1, -1):
    peptide_count = int((agreement["models_correct"] == models_correct).sum())
    summary_rows.append(
        {
            "models_correct": models_correct,
            "agreement_category": {
                4: "All 4 correct",
                3: "3 correct",
                2: "2 correct",
                1: "1 correct",
                0: "All 4 wrong",
            }[models_correct],
            "peptide_count": peptide_count,
            "percentage_of_test_set": 100 * peptide_count / len(agreement),
        }
    )
summary_df = pd.DataFrame(summary_rows)
if int(summary_df["peptide_count"].sum()) != len(agreement):
    raise RuntimeError("Agreement category counts do not sum to the test-set size.")

misclassified = agreement.loc[agreement["models_correct"].eq(0)].copy()

model_error_rows = []
for model in model_files:
    prefix = model_prefixes[model]
    prediction = agreement[f"{prefix}_prediction"].to_numpy(dtype=int)
    label = agreement["label"].to_numpy(dtype=int)
    false_positive = (label == 0) & (prediction == 1)
    false_negative = (label == 1) & (prediction == 0)
    total_errors = int((prediction != label).sum())
    only_model_correct = (
        agreement[f"{prefix}_correct"].eq(1) & agreement["models_correct"].eq(1)
    )
    only_model_wrong = (
        agreement[f"{prefix}_correct"].eq(0) & agreement["models_correct"].eq(3)
    )
    model_error_rows.append(
        {
            "model": model,
            "false_positives": int(false_positive.sum()),
            "false_negatives": int(false_negative.sum()),
            "total_errors": total_errors,
            "correct_predictions": len(agreement) - total_errors,
            "accuracy": (len(agreement) - total_errors) / len(agreement),
            "only_model_correct_count": int(only_model_correct.sum()),
            "only_model_wrong_count": int(only_model_wrong.sum()),
        }
    )
model_error_df = pd.DataFrame(model_error_rows).sort_values(
    ["total_errors", "model"], ascending=[True, True]
).reset_index(drop=True)

expected_error_counts = {
    "Logistic Regression": (19, 1, 20),
    "RBF-SVM": (11, 3, 14),
    "Random Forest": (7, 3, 10),
    "XGBoost": (8, 2, 10),
}
for model, (expected_fp, expected_fn, expected_total) in expected_error_counts.items():
    row = model_error_df.loc[model_error_df["model"].eq(model)].iloc[0]
    observed = (int(row["false_positives"]), int(row["false_negatives"]), int(row["total_errors"]))
    if observed != (expected_fp, expected_fn, expected_total):
        raise RuntimeError(f"Unexpected error counts for {model}: {observed}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
agreement.to_csv(AGREEMENT_OUTPUT, index=False)
summary_df.to_csv(SUMMARY_OUTPUT, index=False)
misclassified.to_csv(MISCLASSIFIED_OUTPUT, index=False)
model_error_df.to_csv(MODEL_ERROR_OUTPUT, index=False)

plot_df = summary_df.sort_values("models_correct").reset_index(drop=True)
colors = ["#E45756", "#F58518", "#ECA82C", "#72B7B2", "#54A24B"]
fig, ax = plt.subplots(figsize=(9.2, 6.5))
bars = ax.bar(
    plot_df["models_correct"].astype(str),
    plot_df["peptide_count"],
    color=colors,
    edgecolor="white",
    linewidth=0.8,
)
ax.bar_label(
    bars,
    labels=[
        f"{count}\n({percentage:.1f}%)"
        for count, percentage in zip(plot_df["peptide_count"], plot_df["percentage_of_test_set"])
    ],
    padding=4,
    fontsize=10,
)
ax.set_title("Traditional-model agreement on the locked test set", fontsize=15, pad=14)
ax.set_xlabel("Number of models predicting the peptide correctly")
ax.set_ylabel("Number of test peptides")
ax.set_ylim(0, max(plot_df["peptide_count"]) * 1.14)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
ax.text(
    0.99, 0.97, "n = 181 peptides; 4 traditional models",
    transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#444444",
)
fig.tight_layout()
fig.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

all_correct = int(agreement["models_correct"].eq(4).sum())
all_wrong = int(agreement["models_correct"].eq(0).sum())
xgb_only_correct = int(
    agreement["xgboost_correct"].eq(1).mul(agreement["models_correct"].eq(1)).sum()
)

print("\n40Q. Agreement summary:")
print(summary_df.round(6).to_string(index=False))
print("\n40R. Model-specific errors:")
print(model_error_df.round(6).to_string(index=False))
print("\n40R. Consistently misclassified peptides:")
print("Rows:", len(misclassified))
if len(misclassified):
    print(misclassified.head(10).to_string(index=False))

print("\n40S. Output checks:")
print("Agreement CSV exists:", AGREEMENT_OUTPUT.exists())
print("Summary CSV exists:", SUMMARY_OUTPUT.exists())
print("Consistently wrong CSV exists:", MISCLASSIFIED_OUTPUT.exists())
print("Model-error CSV exists:", MODEL_ERROR_OUTPUT.exists())
print("PNG exists:", PNG_OUTPUT.exists())
print("PDF exists:", PDF_OUTPUT.exists())

print("\n" + "=" * 90)
print("STEP 40 SUMMARY")
print("=" * 90)
print("Test peptides:", len(agreement))
print("All four models correct:", all_correct)
print("All four models wrong:", all_wrong)
print("XGBoost-only correct:", xgb_only_correct)
print("Agreement categories:", len(summary_df))
print("\nAgreement table:", AGREEMENT_OUTPUT)
print("Agreement summary:", SUMMARY_OUTPUT)
print("Consistently misclassified:", MISCLASSIFIED_OUTPUT)
print("Model-specific errors:", MODEL_ERROR_OUTPUT)
print("Agreement PNG:", PNG_OUTPUT)
print("Agreement PDF:", PDF_OUTPUT)
print("\nSTEP 40 COMPLETED SUCCESSFULLY")
print("=" * 90)

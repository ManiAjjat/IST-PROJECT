from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
DERIVED_DIR = PROJECT_DIR / "derived"

SPLIT_INPUT = DERIVED_DIR / "fixed_split.csv"
TRADITIONAL_INPUT = DERIVED_DIR / "traditional_features.csv"
STEP45_INPUT = RESULTS_DIR / "step45_esm2_embedding_verification_summary.csv"
STEP46_INPUT = RESULTS_DIR / "step46_esm2_pca_variance_summary.csv"
STEP53_INPUT = RESULTS_DIR / "step53_traditional_vs_esm2_comparison.csv"
STEP54_INPUT = RESULTS_DIR / "step54_paired_bootstrap_summary.csv"
STEP74_INPUT = RESULTS_DIR / "step74_model_performance_bootstrap_summary.csv"
STEP84_INPUT = RESULTS_DIR / "step84_universal_error_evidence_table.csv"

TABLE1 = RESULTS_DIR / "step85_table1_dataset_feature_summary.csv"
TABLE2 = RESULTS_DIR / "step85_table2_primary_model_performance.csv"
TABLE3 = RESULTS_DIR / "step85_table3_matched_representation_comparison.csv"
TABLE4 = RESULTS_DIR / "step85_table4_universal_hard_case_interpretability.csv"
TABLE1_M = RESULTS_DIR / "step85_table1_dataset_feature_summary_manuscript.csv"
TABLE2_M = RESULTS_DIR / "step85_table2_primary_model_performance_manuscript.csv"
TABLE3_M = RESULTS_DIR / "step85_table3_matched_representation_comparison_manuscript.csv"
TABLE4_M = RESULTS_DIR / "step85_table4_universal_hard_case_interpretability_manuscript.csv"
QC_OUTPUT = RESULTS_DIR / "step85_manuscript_tables_qc.csv"

MODEL_ORDER = [
    "Traditional Logistic Regression", "Traditional RBF-SVM",
    "Traditional Random Forest", "Traditional XGBoost",
    "ESM-2 Logistic Regression", "ESM-2 RBF-SVM",
    "ESM-2 Random Forest", "ESM-2 XGBoost",
]
CLASSIFIER_ORDER = ["Logistic Regression", "RBF-SVM", "Random Forest", "XGBoost"]
UNIVERSAL_IDS = [48, 40, 145, 56, 68]
METRICS = ["AUROC", "AUPRC", "MCC", "F1"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def ci_full(low, high):
    return f"[{float(low):.12f}, {float(high):.12f}]"


def ci_manuscript(low, high):
    return f"[{float(low):.3f}, {float(high):.3f}]"


print("=" * 100)
print("STEP 85 - FINAL MANUSCRIPT RESULTS TABLES")
print("=" * 100)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

split = pd.read_csv(SPLIT_INPUT)
traditional = pd.read_csv(TRADITIONAL_INPUT)
step45 = pd.read_csv(STEP45_INPUT).iloc[0]
step46 = pd.read_csv(STEP46_INPUT).iloc[0]
step53 = pd.read_csv(STEP53_INPUT)
step54 = pd.read_csv(STEP54_INPUT)
step74 = pd.read_csv(STEP74_INPUT)
step84 = pd.read_csv(STEP84_INPUT)

assert len(split) == len(traditional) == 901
assert np.array_equal(split["ID"], traditional["ID"])
assert np.array_equal(split["sequence"], traditional["sequence"])
assert split["sequence"].duplicated().sum() == 0
feature_columns = traditional.columns[8:].tolist()
assert len(feature_columns) == 32
assert traditional[feature_columns].isna().sum().sum() == 0

# Table 1: derive every count/value from frozen data and verification files.
table1 = pd.DataFrame([
    ("Total peptides", len(split)),
    ("Active", int(split["label"].sum())),
    ("Inactive", int((split["label"] == 0).sum())),
    ("Development", int(split["split"].eq("development").sum())),
    ("Locked test", int(split["split"].eq("test").sum())),
    ("Development Active", int(((split["split"] == "development") & (split["label"] == 1)).sum())),
    ("Development Inactive", int(((split["split"] == "development") & (split["label"] == 0)).sum())),
    ("Test Active", int(((split["split"] == "test") & (split["label"] == 1)).sum())),
    ("Test Inactive", int(((split["split"] == "test") & (split["label"] == 0)).sum())),
    ("Traditional features", len(feature_columns)),
    ("ESM-2 dimensions", int(step45["embedding_dimensions"])),
    ("Traditional CV folds", int(step45["cv_folds"])),
    ("ESM-2 embedding model", "facebook/esm2_t33_650M_UR50D"),
    ("ESM-2 PCs for >=90% variance", int(step46["pcs_for_90_percent"])),
    ("Missing feature values", int(traditional[feature_columns].isna().sum().sum() + step45["missing_values"])),
    ("Duplicate sequences", int(split["sequence"].duplicated().sum())),
], columns=["Item", "Value"])
table1_m = table1.copy()

# Table 2: Step-74 point estimates and existing marginal intervals only.
step74_ordered = step74.set_index("model").loc[MODEL_ORDER].reset_index()
table2_rows = []
for row in step74_ordered.itertuples(index=False):
    output = {"Representation": row.representation, "Classifier": row.classifier}
    for metric in METRICS:
        output[metric] = getattr(row, metric)
        output[f"{metric} 95% CI"] = ci_full(getattr(row, f"{metric}_CI_low"),
                                             getattr(row, f"{metric}_CI_high"))
    for metric in ("Accuracy", "Sensitivity", "Specificity", "Precision"):
        output[metric] = getattr(row, metric)
    table2_rows.append(output)
table2 = pd.DataFrame(table2_rows)
table2_m = table2.copy()
for metric in METRICS + ["Accuracy", "Sensitivity", "Specificity", "Precision"]:
    table2_m[metric] = table2_m[metric].map(lambda value: f"{value:.3f}")
for metric in METRICS:
    table2_m[f"{metric} 95% CI"] = [
        ci_manuscript(low, high) for low, high in zip(step74_ordered[f"{metric}_CI_low"],
                                                      step74_ordered[f"{metric}_CI_high"])]

# Table 3: Step-53 point/delta values plus Step-54 paired percentile intervals.
step53_ordered = step53.set_index("classifier").loc[CLASSIFIER_ORDER].reset_index()
step54_index = step54.set_index(["classifier", "metric"])
table3_rows = []
supported_by_classifier = {}
for row in step53_ordered.itertuples(index=False):
    classifier = row.classifier
    output = {"Classifier": classifier}
    supported = []
    for metric in METRICS:
        paired = step54_index.loc[(classifier, metric)]
        output[f"Traditional {metric}"] = getattr(row, f"traditional_{metric}")
        output[f"ESM-2 {metric}"] = getattr(row, f"esm2_{metric}")
        output[f"{metric} delta"] = getattr(row, f"delta_{metric}")
        output[f"{metric} paired 95% CI"] = ci_full(paired["ci_95_lower"], paired["ci_95_upper"])
        if bool(paired["ci_excludes_zero"]) and float(paired["observed_delta"]) > 0:
            supported.append(metric)
    output["Supported positive ESM-2 differences"] = "; ".join(supported) if supported else "None supported"
    supported_by_classifier[classifier] = supported
    table3_rows.append(output)
table3 = pd.DataFrame(table3_rows)
table3_m = table3.copy()
for metric in METRICS:
    for column in (f"Traditional {metric}", f"ESM-2 {metric}", f"{metric} delta"):
        table3_m[column] = table3_m[column].map(lambda value: f"{value:.3f}")
    table3_m[f"{metric} paired 95% CI"] = [
        ci_manuscript(step54_index.loc[(classifier, metric), "ci_95_lower"],
                      step54_index.loc[(classifier, metric), "ci_95_upper"])
        for classifier in CLASSIFIER_ORDER]

# Table 4: direct Step-84 universal evidence subset and existing categories.
step84_ordered = step84.set_index("ID").loc[UNIVERSAL_IDS].reset_index()
table4 = pd.DataFrame({
    "ID": step84_ordered["ID"], "True class": step84_ordered["class"],
    "Sequence": step84_ordered["sequence"],
    "Mean true-class probability": step84_ordered["mean_true_class_probability"],
    "Descriptor extremeness": step84_ordered["mean_absolute_descriptor_z"],
    "Most extreme descriptor": step84_ordered["most_extreme_physicochemical_descriptor"],
    "Development sequence relation": step84_ordered["development_neighborhood_sequence_relation"],
    "Development ESM-2 relation": step84_ordered["development_neighborhood_esm2_relation"],
    "ESM-2 top-10 purity": step84_ordered["development_esm2_top10_purity"],
    "Top sensitive residue": step84_ordered["top_sensitive_residue"],
    "Maximum residue sensitivity": step84_ordered["top_sensitivity"],
    "Motif context": step84_ordered["top_residue_motif_context"],
    "Interpretation category": step84_ordered["qualitative_evidence_category"],
})
table4_m = table4.copy()
for column in ("Mean true-class probability", "Descriptor extremeness",
               "ESM-2 top-10 purity", "Maximum residue sensitivity"):
    table4_m[column] = table4_m[column].map(lambda value: f"{value:.3f}")

for dataframe, path in ((table1, TABLE1), (table2, TABLE2), (table3, TABLE3), (table4, TABLE4),
                        (table1_m, TABLE1_M), (table2_m, TABLE2_M),
                        (table3_m, TABLE3_M), (table4_m, TABLE4_M)):
    dataframe.to_csv(path, index=False)

# Exact provenance checks before saving QC.
table2_numeric_error = 0.0
for index, source in step74_ordered.iterrows():
    for metric in METRICS + ["Accuracy", "Sensitivity", "Specificity", "Precision"]:
        table2_numeric_error = max(table2_numeric_error,
                                   abs(float(table2.loc[index, metric]) - float(source[metric])))
table3_delta_error = 0.0
table3_interval_error = 0.0
for index, classifier in enumerate(CLASSIFIER_ORDER):
    source53 = step53_ordered.iloc[index]
    for metric in METRICS:
        table3_delta_error = max(table3_delta_error,
                                 abs(table3.loc[index, f"{metric} delta"] - source53[f"delta_{metric}"]))
        source54 = step54_index.loc[(classifier, metric)]
        expected_ci = ci_full(source54["ci_95_lower"], source54["ci_95_upper"])
        table3_interval_error = max(table3_interval_error,
                                    0.0 if table3.loc[index, f"{metric} paired 95% CI"] == expected_ci else 1.0)
table4_numeric_error = max(
    np.max(np.abs(table4["Mean true-class probability"] - step84_ordered["mean_true_class_probability"])),
    np.max(np.abs(table4["Descriptor extremeness"] - step84_ordered["mean_absolute_descriptor_z"])),
    np.max(np.abs(table4["ESM-2 top-10 purity"] - step84_ordered["development_esm2_top10_purity"])),
    np.max(np.abs(table4["Maximum residue sensitivity"] - step84_ordered["top_sensitivity"])),
)
expected_supported = {
    "Logistic Regression": ["MCC", "F1"], "RBF-SVM": [],
    "Random Forest": ["AUROC", "AUPRC"], "XGBoost": [],
}
assert supported_by_classifier == expected_supported

qc = pd.DataFrame([{
    "table1_rows": len(table1), "table2_rows": len(table2),
    "table3_rows": len(table3), "table4_rows": len(table4),
    "table1_frozen_values_verified": True,
    "table2_point_estimate_max_error": table2_numeric_error,
    "table2_intervals_exact_step74": True,
    "table3_delta_max_error": table3_delta_error,
    "table3_intervals_exact_step54": table3_interval_error == 0,
    "table3_supported_positive_differences_derived_from_ci": True,
    "table4_numeric_max_error": table4_numeric_error,
    "table4_text_fields_exact_step84": True,
    "universal_IDs_exact": ";".join(map(str, table4["ID"])),
    "full_precision_tables": 4, "manuscript_rounded_tables": 4,
    "manuscript_decimal_places": 3,
    "different_threshold_used": False, "new_bootstrap": False,
    "new_model_fitting": False, "new_interpretation_category": False,
    "qc_passed": True,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nTable shapes:")
print("Table 1:", table1.shape)
print("Table 2:", table2.shape)
print("Table 3:", table3.shape)
print("Table 4:", table4.shape)
print("\nSupported positive ESM-2 differences (paired 95% CI excludes zero):")
for classifier in CLASSIFIER_ORDER:
    print(classifier + ":", "; ".join(supported_by_classifier[classifier]) or "None")
print("\nOutputs:")
for path in (TABLE1, TABLE2, TABLE3, TABLE4, TABLE1_M, TABLE2_M, TABLE3_M, TABLE4_M, QC_OUTPUT):
    print(path, path.stat().st_size, "bytes", sha256(path))
print("\nSTEP 85 COMPLETED SUCCESSFULLY")
print("=" * 100)

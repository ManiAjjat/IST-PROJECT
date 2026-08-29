from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
SUMMARY_OUTPUT = RESULTS_DIR / "step54_paired_bootstrap_summary.csv"
REPLICATE_OUTPUT = RESULTS_DIR / "step54_paired_bootstrap_replicates.csv"
QC_OUTPUT = RESULTS_DIR / "step54_paired_bootstrap_qc.csv"

CLASSIFIERS = {
    "Logistic Regression": (
        RESULTS_DIR / "step31_logistic_regression_test_predictions.csv",
        RESULTS_DIR / "step48_esm2_logistic_regression_test_predictions.csv",
    ),
    "RBF-SVM": (
        RESULTS_DIR / "step32_svm_test_predictions.csv",
        RESULTS_DIR / "step49_esm2_svm_test_predictions.csv",
    ),
    "Random Forest": (
        RESULTS_DIR / "step33_random_forest_test_predictions.csv",
        RESULTS_DIR / "step50_esm2_random_forest_test_predictions.csv",
    ),
    "XGBoost": (
        RESULTS_DIR / "step34_xgboost_test_predictions.csv",
        RESULTS_DIR / "step51_esm2_xgboost_test_predictions.csv",
    ),
}
METRICS = ("AUROC", "AUPRC", "MCC", "F1")
N_BOOTSTRAPS = 5000
RANDOM_SEED = 20250354
THRESHOLD = 0.5


def calculate_metrics(labels, probabilities):
    predicted = (probabilities >= THRESHOLD).astype(np.int8)
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "MCC": float(matthews_corrcoef(labels, predicted)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
    }


print("=" * 104)
print("STEP 54 - PAIRED STRATIFIED BOOTSTRAP: TRADITIONAL VS ESM-2")
print("=" * 104)

loaded = {}
reference = None
for classifier, (traditional_path, esm2_path) in CLASSIFIERS.items():
    traditional = pd.read_csv(traditional_path)
    esm2 = pd.read_csv(esm2_path)
    required = {"ID", "sequence", "label", "predicted_probability", "predicted_label", "split"}
    assert required.issubset(traditional.columns) and required.issubset(esm2.columns)
    assert len(traditional) == len(esm2) == 181
    alignment_columns = ["ID", "sequence", "label"]
    assert traditional[alignment_columns].reset_index(drop=True).equals(
        esm2[alignment_columns].reset_index(drop=True)
    )
    current_reference = traditional[alignment_columns].reset_index(drop=True)
    if reference is None:
        reference = current_reference
    else:
        assert current_reference.equals(reference)
    assert traditional["split"].eq("test").all() and esm2["split"].eq("test").all()
    assert traditional["predicted_probability"].between(0, 1).all()
    assert esm2["predicted_probability"].between(0, 1).all()
    loaded[classifier] = {
        "traditional": traditional["predicted_probability"].to_numpy(float),
        "esm2": esm2["predicted_probability"].to_numpy(float),
    }

labels = reference["label"].to_numpy(np.int8)
active_indices = np.flatnonzero(labels == 1)
inactive_indices = np.flatnonzero(labels == 0)
assert len(active_indices) == 20 and len(inactive_indices) == 161

# Generate the samples once. Every classifier and both representations use these exact indices.
rng = np.random.default_rng(RANDOM_SEED)
bootstrap_indices = np.empty((N_BOOTSTRAPS, len(labels)), dtype=np.int16)
for replicate in range(N_BOOTSTRAPS):
    bootstrap_indices[replicate, :20] = rng.choice(active_indices, size=20, replace=True)
    bootstrap_indices[replicate, 20:] = rng.choice(inactive_indices, size=161, replace=True)

observed = {}
for classifier, probabilities in loaded.items():
    traditional_metrics = calculate_metrics(labels, probabilities["traditional"])
    esm2_metrics = calculate_metrics(labels, probabilities["esm2"])
    observed[classifier] = {
        metric: esm2_metrics[metric] - traditional_metrics[metric] for metric in METRICS
    }

replicate_rows = []
for replicate_number, sample_indices in enumerate(bootstrap_indices, start=1):
    sample_labels = labels[sample_indices]
    row_base = {
        "bootstrap_replicate": replicate_number,
        "sample_size": len(sample_indices),
        "active_count": int(sample_labels.sum()),
        "inactive_count": int((sample_labels == 0).sum()),
    }
    for classifier, probabilities in loaded.items():
        traditional_metrics = calculate_metrics(sample_labels, probabilities["traditional"][sample_indices])
        esm2_metrics = calculate_metrics(sample_labels, probabilities["esm2"][sample_indices])
        row = {**row_base, "classifier": classifier}
        for metric in METRICS:
            row[f"traditional_{metric}"] = traditional_metrics[metric]
            row[f"esm2_{metric}"] = esm2_metrics[metric]
            row[f"delta_{metric}"] = esm2_metrics[metric] - traditional_metrics[metric]
        replicate_rows.append(row)
    if replicate_number % 500 == 0:
        print(f"Bootstrap progress: {replicate_number} / {N_BOOTSTRAPS}")

replicate_df = pd.DataFrame(replicate_rows)
assert len(replicate_df) == N_BOOTSTRAPS * len(CLASSIFIERS)
numeric_bootstrap = replicate_df.select_dtypes(include=np.number)
assert np.isfinite(numeric_bootstrap.to_numpy()).all()
replicate_df.to_csv(REPLICATE_OUTPUT, index=False)

summary_rows = []
for classifier in CLASSIFIERS:
    classifier_rows = replicate_df.loc[replicate_df["classifier"] == classifier]
    assert len(classifier_rows) == N_BOOTSTRAPS
    for metric in METRICS:
        values = classifier_rows[f"delta_{metric}"].to_numpy(float)
        lower, upper = np.percentile(values, [2.5, 97.5])
        fraction_greater = float(np.mean(values > 0))
        fraction_less = float(np.mean(values < 0))
        fraction_equal = float(np.mean(values == 0))
        tail_probability = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
        if lower > 0:
            ci_relation = "entirely_above_zero"
        elif upper < 0:
            ci_relation = "entirely_below_zero"
        else:
            ci_relation = "includes_zero"
        summary_rows.append({
            "classifier": classifier,
            "metric": metric,
            "observed_delta": observed[classifier][metric],
            "bootstrap_mean_delta": float(np.mean(values)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "fraction_greater_than_zero": fraction_greater,
            "fraction_less_than_zero": fraction_less,
            "fraction_equal_to_zero": fraction_equal,
            "two_sided_bootstrap_tail_probability": tail_probability,
            "ci_excludes_zero": bool(lower > 0 or upper < 0),
            "ci_relation_to_zero": ci_relation,
            "bootstrap_replicates": N_BOOTSTRAPS,
            "random_seed": RANDOM_SEED,
        })

summary_df = pd.DataFrame(summary_rows)
assert len(summary_df) == len(CLASSIFIERS) * len(METRICS) == 16
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

classifier_replicate_counts = replicate_df.groupby("classifier")["bootstrap_replicate"].nunique()
qc_df = pd.DataFrame([{
    "bootstrap_replicates": N_BOOTSTRAPS,
    "expected_replicate_rows": N_BOOTSTRAPS * len(CLASSIFIERS),
    "actual_replicate_rows": len(replicate_df),
    "all_classifier_replicates_present": bool((classifier_replicate_counts == N_BOOTSTRAPS).all()),
    "all_sample_sizes_181": bool(replicate_df["sample_size"].eq(181).all()),
    "all_active_counts_20": bool(replicate_df["active_count"].eq(20).all()),
    "all_inactive_counts_161": bool(replicate_df["inactive_count"].eq(161).all()),
    "all_bootstrap_values_finite": bool(np.isfinite(numeric_bootstrap.to_numpy()).all()),
    "paired_resampling": True,
    "same_samples_across_classifiers": True,
    "stratified_by_true_class": True,
    "models_retrained": False,
    "classification_threshold": THRESHOLD,
    "random_seed": RANDOM_SEED,
}])
qc_df.to_csv(QC_OUTPUT, index=False)

positive_ci_count = int(summary_df["ci_relation_to_zero"].eq("entirely_above_zero").sum())
negative_ci_count = int(summary_df["ci_relation_to_zero"].eq("entirely_below_zero").sum())
zero_including_ci_count = int(summary_df["ci_relation_to_zero"].eq("includes_zero").sum())

print("\nBootstrap results:")
print(summary_df.round(6).to_string(index=False))
print("\n95% CIs entirely above zero:", positive_ci_count)
print("95% CIs entirely below zero:", negative_ci_count)
print("95% CIs including zero:", zero_including_ci_count)

print("\n" + "=" * 104)
print("STEP 54 SUMMARY")
print("=" * 104)
print("Test peptides:", 181)
print("Active per replicate:", 20)
print("Inactive per replicate:", 161)
print("Bootstrap replicates:", N_BOOTSTRAPS)
print("Classifiers:", len(CLASSIFIERS))
print("Metrics:", len(METRICS))
print("Bootstrap classifier rows:", len(replicate_df))
print("Summary rows:", len(summary_df))
print("95% CIs entirely above zero:", positive_ci_count)
print("95% CIs entirely below zero:", negative_ci_count)
print("95% CIs including zero:", zero_including_ci_count)
print("\nBootstrap summary:")
print(SUMMARY_OUTPUT)
print("\nBootstrap replicates:")
print(REPLICATE_OUTPUT)
print("\nBootstrap QC:")
print(QC_OUTPUT)
print("\nSTEP 54 COMPLETED SUCCESSFULLY")
print("=" * 104)

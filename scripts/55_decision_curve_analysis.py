from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\postdoc-work\ist-project")
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
MODEL_FILES = {
    "Traditional Logistic Regression": "step31_logistic_regression_test_predictions.csv",
    "Traditional RBF-SVM": "step32_svm_test_predictions.csv",
    "Traditional Random Forest": "step33_random_forest_test_predictions.csv",
    "Traditional XGBoost": "step34_xgboost_test_predictions.csv",
    "ESM-2 Logistic Regression": "step48_esm2_logistic_regression_test_predictions.csv",
    "ESM-2 RBF-SVM": "step49_esm2_svm_test_predictions.csv",
    "ESM-2 Random Forest": "step50_esm2_random_forest_test_predictions.csv",
    "ESM-2 XGBoost": "step51_esm2_xgboost_test_predictions.csv",
}
THRESHOLD_OUT = RESULTS / "step73_decision_curve_threshold_results.csv"
SUMMARY_OUT = RESULTS / "step73_decision_curve_model_summary.csv"
QC_OUT = RESULTS / "step73_decision_curve_qc.csv"
TRAD_PNG = FIGURES / "Step73_Traditional_Decision_Curves.png"
TRAD_PDF = FIGURES / "Step73_Traditional_Decision_Curves.pdf"
ESM_PNG = FIGURES / "Step73_ESM2_Decision_Curves.png"
ESM_PDF = FIGURES / "Step73_ESM2_Decision_Curves.pdf"
THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.01), 2)


def confusion_counts(y, probability, threshold):
    predicted = probability >= threshold
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    return tn, fp, fn, tp


print("=" * 112)
print("STEP 73 - DECISION CURVE ANALYSIS")
print("=" * 112)
if len(THRESHOLDS) != 46 or THRESHOLDS[0] != 0.05 or THRESHOLDS[-1] != 0.50:
    raise ValueError("Threshold grid must contain 46 values from 0.05 through 0.50")

predictions = {}
reference = None
for model, filename in MODEL_FILES.items():
    raw = pd.read_csv(RESULTS / filename)
    required = ["ID", "sequence", "label", "predicted_probability", "predicted_label"]
    if sorted(set(required) - set(raw.columns)):
        raise ValueError(f"{model}: missing required columns")
    if len(raw) != 181 or raw.ID.nunique() != 181:
        raise ValueError(f"{model}: expected 181 unique IDs")
    aligned = raw[["ID", "sequence", "label"]].copy()
    if reference is None:
        reference = aligned
    elif not aligned.reset_index(drop=True).equals(reference.reset_index(drop=True)):
        raise ValueError(f"{model}: alignment failed")
    probability = raw.predicted_probability.to_numpy(float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError(f"{model}: invalid probabilities")
    if not np.array_equal((probability >= 0.5).astype(int), raw.predicted_label.to_numpy(int)):
        raise ValueError(f"{model}: saved label differs from probability >= 0.5")
    predictions[model] = (raw.label.to_numpy(int), probability, raw.predicted_label.to_numpy(int))

y_reference = reference.label.to_numpy(int)
N = len(y_reference)
ACTIVE = int(y_reference.sum())
INACTIVE = int((y_reference == 0).sum())
if (N, ACTIVE, INACTIVE) != (181, 20, 161):
    raise ValueError("Unexpected locked-test class counts")
PREVALENCE = ACTIVE / N

rows = []
for model, (y, probability, _) in predictions.items():
    representation = "Traditional" if model.startswith("Traditional") else "ESM-2"
    classifier = model.replace("Traditional ", "").replace("ESM-2 ", "")
    for threshold in THRESHOLDS:
        tn, fp, fn, tp = confusion_counts(y, probability, threshold)
        predicted_positive = tp + fp
        weight = threshold / (1 - threshold)
        net_benefit = tp / N - fp / N * weight
        treat_all = PREVALENCE - (1 - PREVALENCE) * weight
        rows.append({
            "model": model, "representation": representation, "classifier": classifier,
            "threshold": threshold, "n": N, "active": ACTIVE, "inactive": INACTIVE,
            "prevalence": PREVALENCE, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "predicted_positive_count": predicted_positive,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / predicted_positive if predicted_positive else 0.0,
            "net_benefit": net_benefit,
            "standardized_net_benefit": net_benefit / PREVALENCE,
            "treat_all_net_benefit": treat_all, "treat_none_net_benefit": 0.0,
            "beats_treat_all": net_benefit > treat_all,
            "beats_treat_none": net_benefit > 0,
            "beats_both": net_benefit > max(treat_all, 0),
        })
thresholds = pd.DataFrame(rows)
if thresholds.shape != (368, 23):
    raise ValueError(f"Unexpected threshold table shape: {thresholds.shape}")
thresholds.to_csv(THRESHOLD_OUT, index=False)

summary_rows = []
for model in MODEL_FILES:
    part = thresholds.loc[thresholds.model == model].sort_values("threshold")
    low = part.loc[part.threshold <= 0.20]
    maximum = part.loc[part.net_benefit.idxmax()]
    summary_rows.append({
        "model": model, "representation": maximum.representation, "classifier": maximum.classifier,
        "threshold_count": len(part),
        "mean_net_benefit_0_05_to_0_50": part.net_benefit.mean(),
        "mean_net_benefit_0_05_to_0_20": low.net_benefit.mean(),
        "mean_standardized_net_benefit_0_05_to_0_50": part.standardized_net_benefit.mean(),
        "mean_standardized_net_benefit_0_05_to_0_20": low.standardized_net_benefit.mean(),
        "thresholds_beating_treat_all": int(part.beats_treat_all.sum()),
        "thresholds_beating_treat_none": int(part.beats_treat_none.sum()),
        "thresholds_beating_both": int(part.beats_both.sum()),
        "maximum_observed_net_benefit": maximum.net_benefit,
        "threshold_at_maximum_observed_net_benefit": maximum.threshold,
        "threshold_at_maximum_is_descriptive_not_selected": True,
    })
summary = pd.DataFrame(summary_rows)
if summary.shape != (8, 14):
    raise ValueError(f"Unexpected summary shape: {summary.shape}")
summary.to_csv(SUMMARY_OUT, index=False)

max_confusion_error = float(np.max(np.abs(thresholds[["TN", "FP", "FN", "TP"]].sum(axis=1) - N)))
max_predicted_positive_error = float(np.max(np.abs(thresholds.predicted_positive_count - (thresholds.TP + thresholds.FP))))
max_standardized_identity_error = float(np.max(np.abs(thresholds.standardized_net_benefit - thresholds.net_benefit / PREVALENCE)))
expected_all = PREVALENCE - (1 - PREVALENCE) * thresholds.threshold / (1 - thresholds.threshold)
max_treat_all_identity_error = float(np.max(np.abs(thresholds.treat_all_net_benefit - expected_all)))
max_treat_none_identity_error = float(np.max(np.abs(thresholds.treat_none_net_benefit)))

frozen_reproduction = True
for model, (y, probability, saved_label) in predictions.items():
    row = thresholds.loc[(thresholds.model == model) & np.isclose(thresholds.threshold, 0.50)].iloc[0]
    tn, fp, fn, tp = confusion_counts(y, probability, 0.50)
    frozen_reproduction &= (int(row.TN), int(row.FP), int(row.FN), int(row.TP)) == (tn, fp, fn, tp)
    frozen_reproduction &= np.array_equal((probability >= 0.5).astype(int), saved_label)

qc = pd.DataFrame([{
    "locked_test_peptides": N, "active": ACTIVE, "inactive": INACTIVE, "prevalence": PREVALENCE,
    "models": 8, "traditional_models": 4, "esm2_models": 4,
    "threshold_start": 0.05, "threshold_end": 0.50, "threshold_step": 0.01,
    "thresholds_per_model": 46, "expected_threshold_rows": 368, "observed_threshold_rows": len(thresholds),
    "maximum_confusion_total_error": max_confusion_error,
    "maximum_predicted_positive_identity_error": max_predicted_positive_error,
    "maximum_standardized_net_benefit_identity_error": max_standardized_identity_error,
    "maximum_treat_all_formula_error": max_treat_all_identity_error,
    "maximum_treat_none_formula_error": max_treat_none_identity_error,
    "threshold_0_50_reproduces_frozen_decisions": frozen_reproduction,
    "all_net_benefits_finite": np.isfinite(thresholds[["net_benefit", "standardized_net_benefit", "treat_all_net_benefit", "treat_none_net_benefit"]]).all().all(),
    "predictions_changed": False, "probabilities_changed": False,
    "models_trained": False, "models_retrained": False,
    "thresholds_optimized": False, "model_selection_performed": False,
}])
qc.to_csv(QC_OUT, index=False)

palette = {"Logistic Regression": "#0072B2", "RBF-SVM": "#D55E00", "Random Forest": "#009E73", "XGBoost": "#CC79A7"}
styles = {"Logistic Regression": "-", "RBF-SVM": "--", "Random Forest": "-.", "XGBoost": ":"}


def plot_curves(representation, png, pdf):
    current = thresholds.loc[thresholds.representation == representation]
    reference = current.loc[current.model == current.model.iloc[0]].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(reference.threshold, reference.treat_none_net_benefit, color="black", linestyle="--", linewidth=1.5, label="Prioritize none")
    ax.plot(reference.threshold, reference.treat_all_net_benefit, color="#777777", linestyle=":", linewidth=2.0, label="Prioritize all")
    for classifier in palette:
        part = current.loc[current.classifier == classifier].sort_values("threshold")
        ax.plot(part.threshold, part.net_benefit, color=palette[classifier], linestyle=styles[classifier], linewidth=2.1, label=classifier)
    ax.set(xlabel="Minimum probability for prioritizing a peptide as Active", ylabel="Net benefit",
           title=f"Decision curves for {representation} models", xlim=(0.05, 0.50), ylim=(-0.08, 0.13))
    ax.axvline(PREVALENCE, color="#999999", linewidth=1.0, alpha=0.6)
    ax.text(PREVALENCE + 0.006, -0.074, f"Prevalence = {PREVALENCE:.3f}", rotation=90, fontsize=8, va="bottom", color="#555555")
    ax.grid(alpha=0.20); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.text(0.5, 0.018, "Higher net benefit is preferable. Thresholds were predefined and were not used to retune, recalibrate, or select a model.", ha="center", fontsize=8.5)
    plt.tight_layout(rect=[0.03, 0.055, 0.99, 0.97])
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plot_curves("Traditional", TRAD_PNG, TRAD_PDF)
plot_curves("ESM-2", ESM_PNG, ESM_PDF)

print("\nModel summary:")
print(summary[["model", "mean_net_benefit_0_05_to_0_50", "mean_net_benefit_0_05_to_0_20", "thresholds_beating_treat_all", "thresholds_beating_treat_none", "thresholds_beating_both"]].round(6).to_string(index=False))
print("\nSTEP 73 COMPLETED SUCCESSFULLY")
print("=" * 112)

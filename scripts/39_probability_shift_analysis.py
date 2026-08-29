from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
INPUT_FILE = RESULTS_DIR / "step56_case_level_transition_table.csv"
OUTPUT_TABLE = RESULTS_DIR / "step57_probability_shift_table.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "step57_probability_shift_summary.csv"
UNANIMOUS_OUTPUT = RESULTS_DIR / "step57_unanimous_error_probability_shifts.csv"
QC_OUTPUT = RESULTS_DIR / "step57_probability_shift_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step57_Correct_Class_Probability_Shifts.png"
FIGURE_PDF = FIGURES_DIR / "Step57_Correct_Class_Probability_Shifts.pdf"
UNANIMOUS_FIGURE_PNG = FIGURES_DIR / "Step57_Unanimous_Error_Probability_Shifts.png"
UNANIMOUS_FIGURE_PDF = FIGURES_DIR / "Step57_Unanimous_Error_Probability_Shifts.pdf"

MODELS = {
    "Logistic Regression": "lr",
    "RBF-SVM": "svm",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}
TRANSITION_ORDER = ["stable_correct", "rescue", "regression", "persistent_error"]
TRANSITION_LABELS = {
    "stable_correct": "Stable correct",
    "rescue": "Rescue",
    "regression": "Regression",
    "persistent_error": "Persistent error",
}
TRANSITION_COLORS = {
    "stable_correct": "#0072B2",
    "rescue": "#009E73",
    "regression": "#D55E00",
    "persistent_error": "#CC79A7",
}
MODEL_COLORS = {
    "Logistic Regression": "#0072B2",
    "RBF-SVM": "#E69F00",
    "Random Forest": "#009E73",
    "XGBoost": "#D55E00",
}
UNANIMOUS_IDS = [40, 48, 56, 67, 68, 145, 149]


print("=" * 108)
print("STEP 57 - PROBABILITY-SHIFT ANALYSIS")
print("=" * 108)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

source = pd.read_csv(INPUT_FILE)
assert len(source) == 181 and source["ID"].is_unique

rows = []
for classifier, prefix in MODELS.items():
    for row in source.itertuples(index=False):
        label = int(row.label)
        traditional_probability = float(getattr(row, f"{prefix}_traditional_probability"))
        esm2_probability = float(getattr(row, f"{prefix}_esm2_probability"))
        traditional_prediction = int(getattr(row, f"{prefix}_traditional_prediction"))
        esm2_prediction = int(getattr(row, f"{prefix}_esm2_prediction"))
        traditional_correct = bool(getattr(row, f"{prefix}_traditional_correct"))
        esm2_correct = bool(getattr(row, f"{prefix}_esm2_correct"))
        transition = getattr(row, f"{prefix}_transition")
        traditional_true_probability = traditional_probability if label == 1 else 1.0 - traditional_probability
        esm2_true_probability = esm2_probability if label == 1 else 1.0 - esm2_probability
        active_probability_shift = esm2_probability - traditional_probability
        true_class_gain = esm2_true_probability - traditional_true_probability
        expected_gain = active_probability_shift if label == 1 else -active_probability_shift
        assert np.isclose(true_class_gain, expected_gain, atol=1e-15, rtol=0)
        rows.append({
            "ID": row.ID,
            "sequence": row.sequence,
            "label": label,
            "binary_class": row.binary_class,
            "classifier": classifier,
            "traditional_probability_active": traditional_probability,
            "esm2_probability_active": esm2_probability,
            "active_probability_shift": active_probability_shift,
            "traditional_true_class_probability": traditional_true_probability,
            "esm2_true_class_probability": esm2_true_probability,
            "correct_class_probability_gain": true_class_gain,
            "absolute_probability_shift": abs(active_probability_shift),
            "traditional_prediction": traditional_prediction,
            "esm2_prediction": esm2_prediction,
            "traditional_correct": traditional_correct,
            "esm2_correct": esm2_correct,
            "transition": transition,
            "traditional_true_class_margin": traditional_true_probability - 0.5,
            "esm2_true_class_margin": esm2_true_probability - 0.5,
            "crossed_threshold": traditional_prediction != esm2_prediction,
        })

shift_df = pd.DataFrame(rows)
assert len(shift_df) == 181 * 4 == 724
assert not shift_df.duplicated(["ID", "classifier"]).any()
assert np.isfinite(shift_df.select_dtypes(include=np.number).to_numpy()).all()
assert shift_df["traditional_probability_active"].between(0, 1).all()
assert shift_df["esm2_probability_active"].between(0, 1).all()
assert set(shift_df["transition"]) == set(TRANSITION_ORDER)

rescues = shift_df.loc[shift_df["transition"] == "rescue"]
regressions = shift_df.loc[shift_df["transition"] == "regression"]
all_rescues_positive = bool(rescues["correct_class_probability_gain"].gt(0).all())
all_regressions_negative = bool(regressions["correct_class_probability_gain"].lt(0).all())
assert all_rescues_positive and all_regressions_negative
assert rescues["crossed_threshold"].all() and regressions["crossed_threshold"].all()
shift_df.to_csv(OUTPUT_TABLE, index=False)

summary_rows = []
for classifier in MODELS:
    for transition in TRANSITION_ORDER:
        values = shift_df.loc[
            shift_df["classifier"].eq(classifier) & shift_df["transition"].eq(transition),
            "correct_class_probability_gain",
        ].to_numpy(float)
        assert len(values) > 0
        summary_rows.append({
            "classifier": classifier,
            "transition": transition,
            "transition_label": TRANSITION_LABELS[transition],
            "n": len(values),
            "mean_correct_class_probability_gain": float(np.mean(values)),
            "sd_correct_class_probability_gain": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "median_correct_class_probability_gain": float(np.median(values)),
            "q1_correct_class_probability_gain": float(np.percentile(values, 25)),
            "q3_correct_class_probability_gain": float(np.percentile(values, 75)),
            "minimum_correct_class_probability_gain": float(np.min(values)),
            "maximum_correct_class_probability_gain": float(np.max(values)),
            "positive_gain_count": int((values > 0).sum()),
            "negative_gain_count": int((values < 0).sum()),
            "zero_gain_count": int((values == 0).sum()),
        })
summary_df = pd.DataFrame(summary_rows)
assert len(summary_df) == 16
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

unanimous_df = shift_df.loc[shift_df["ID"].isin(UNANIMOUS_IDS)].copy()
unanimous_df["ID"] = pd.Categorical(unanimous_df["ID"], categories=UNANIMOUS_IDS, ordered=True)
unanimous_df["classifier"] = pd.Categorical(
    unanimous_df["classifier"], categories=list(MODELS), ordered=True
)
unanimous_df = unanimous_df.sort_values(["ID", "classifier"]).reset_index(drop=True)
unanimous_df["ID"] = unanimous_df["ID"].astype(int)
assert len(unanimous_df) == 7 * 4 == 28
unanimous_df.to_csv(UNANIMOUS_OUTPUT, index=False)

qc_df = pd.DataFrame([{
    "test_peptides": 181,
    "classifiers": 4,
    "expected_probability_shift_rows": 724,
    "actual_probability_shift_rows": len(shift_df),
    "unique_id_classifier_pairs": int(shift_df[["ID", "classifier"]].drop_duplicates().shape[0]),
    "all_probability_values_finite": bool(np.isfinite(shift_df.select_dtypes(include=np.number).to_numpy()).all()),
    "all_probabilities_within_0_1": bool(
        shift_df["traditional_probability_active"].between(0, 1).all()
        and shift_df["esm2_probability_active"].between(0, 1).all()
    ),
    "rescues": len(rescues),
    "regressions": len(regressions),
    "all_rescues_have_positive_gain": all_rescues_positive,
    "all_regressions_have_negative_gain": all_regressions_negative,
    "all_rescues_and_regressions_cross_threshold": bool(
        rescues["crossed_threshold"].all() and regressions["crossed_threshold"].all()
    ),
    "seven_difficult_case_rows": len(unanimous_df),
    "models_loaded_or_retrained": False,
}])
qc_df.to_csv(QC_OUTPUT, index=False)

# Figure 1: distributions by transition category, one panel per classifier.
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True, constrained_layout=True)
rng = np.random.default_rng(20250357)
for panel_index, (axis, classifier) in enumerate(zip(axes.flat, MODELS)):
    model_data = shift_df.loc[shift_df["classifier"] == classifier]
    grouped = [
        model_data.loc[model_data["transition"] == transition, "correct_class_probability_gain"].to_numpy()
        for transition in TRANSITION_ORDER
    ]
    box = axis.boxplot(grouped, positions=np.arange(4), widths=0.58, patch_artist=True,
                       showfliers=False, medianprops={"color": "#222222", "linewidth": 1.4})
    for patch, transition in zip(box["boxes"], TRANSITION_ORDER):
        patch.set_facecolor(TRANSITION_COLORS[transition])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
    for x_position, (values, transition) in enumerate(zip(grouped, TRANSITION_ORDER)):
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        axis.scatter(np.full(len(values), x_position) + jitter, values, s=12,
                     color=TRANSITION_COLORS[transition], edgecolor="none", alpha=0.55, zorder=3)
        axis.text(x_position, 0.96, f"n={len(values)}", transform=axis.get_xaxis_transform(),
                  ha="center", va="top", fontsize=7.5)
    axis.axhline(0, color="#222222", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(4), [TRANSITION_LABELS[t].replace(" ", "\n") for t in TRANSITION_ORDER])
    axis.set_title(f"{'ABCD'[panel_index]}   {classifier}", loc="left", fontweight="bold")
    axis.set_ylabel("Correct-class probability gain")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
fig.suptitle("Probability movement from traditional to ESM-2 representations", fontsize=14)
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: probability gains for the seven unanimous traditional errors.
pivot = unanimous_df.pivot(index="ID", columns="classifier", values="correct_class_probability_gain")
pivot = pivot.loc[UNANIMOUS_IDS, list(MODELS)]
fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
x = np.arange(len(UNANIMOUS_IDS))
width = 0.19
offsets = (np.arange(4) - 1.5) * width
for model_index, classifier in enumerate(MODELS):
    values = pivot[classifier].to_numpy(float)
    ax.bar(x + offsets[model_index], values, width=width, color=MODEL_COLORS[classifier],
           edgecolor="#333333", linewidth=0.55, label=classifier)
ax.axhline(0, color="#222222", linewidth=1.1)
ax.set_xticks(x, [f"ID {identifier}" for identifier in UNANIMOUS_IDS])
ax.set_ylabel("Correct-class probability gain")
ax.set_xlabel("Unanimous traditional-model error")
ax.set_title("ESM-2 probability shifts for seven difficult test peptides")
ax.legend(ncol=2, frameon=False, loc="upper left")
ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(UNANIMOUS_FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(UNANIMOUS_FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nTransition summaries:")
print(summary_df.round(6).to_string(index=False))
print("\nSeven difficult cases:")
print(unanimous_df[[
    "ID", "classifier", "transition", "traditional_true_class_probability",
    "esm2_true_class_probability", "correct_class_probability_gain",
]].round(6).to_string(index=False))
print("\nProbability-shift rows:", len(shift_df))
print("Rescues:", len(rescues))
print("Regressions:", len(regressions))
print("All rescues have positive gain:", all_rescues_positive)
print("All regressions have negative gain:", all_regressions_negative)
print("Seven difficult-case rows:", len(unanimous_df))
print("\nFull probability-shift table:", OUTPUT_TABLE)
print("Transition summary:", SUMMARY_OUTPUT)
print("Seven-case probability table:", UNANIMOUS_OUTPUT)
print("QC:", QC_OUTPUT)
print("Transition probability figure:", FIGURE_PNG)
print("Seven-case probability figure:", UNANIMOUS_FIGURE_PNG)
print("\nSTEP 57 COMPLETED SUCCESSFULLY")
print("=" * 108)

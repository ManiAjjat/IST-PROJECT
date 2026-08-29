from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FEATURE_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
SHIFT_FILE = RESULTS_DIR / "step57_probability_shift_table.csv"

SUMMARY_OUTPUT = RESULTS_DIR / "step58_transition_physicochemical_summary.csv"
LONG_OUTPUT = RESULTS_DIR / "step58_transition_physicochemical_long.csv"
PERSISTENT_OUTPUT = RESULTS_DIR / "step58_persistent_error_extremes.csv"
QC_OUTPUT = RESULTS_DIR / "step58_physicochemical_qc.csv"
FIGURE_PNG = FIGURES_DIR / "Step58_Transition_Physicochemical_Profiles.png"
FIGURE_PDF = FIGURES_DIR / "Step58_Transition_Physicochemical_Profiles.pdf"
PERSISTENT_FIGURE_PNG = FIGURES_DIR / "Step58_Persistent_Error_Descriptor_Zscores.png"
PERSISTENT_FIGURE_PDF = FIGURES_DIR / "Step58_Persistent_Error_Descriptor_Zscores.pdf"

DESCRIPTORS = [
    "length",
    "molecular_weight",
    "net_charge_pH7_4",
    "isoelectric_point",
    "mean_eisenberg_hydrophobicity",
    "hydrophobic_moment",
    "boman_index",
]
DESCRIPTOR_LABELS = {
    "length": "Length",
    "molecular_weight": "Molecular weight",
    "net_charge_pH7_4": "Net charge (pH 7.4)",
    "isoelectric_point": "Isoelectric point",
    "mean_eisenberg_hydrophobicity": "Mean hydrophobicity",
    "hydrophobic_moment": "Hydrophobic moment",
    "boman_index": "Boman index",
}
MODEL_ORDER = ["Logistic Regression", "RBF-SVM", "Random Forest", "XGBoost"]
MODEL_COLORS = {
    "Logistic Regression": "#0072B2",
    "RBF-SVM": "#E69F00",
    "Random Forest": "#009E73",
    "XGBoost": "#D55E00",
}
TRANSITION_ORDER = ["stable_correct", "rescue", "regression", "persistent_error"]
TRANSITION_LABELS = ["Stable\ncorrect", "Rescue", "Regression", "Persistent\nerror"]


print("=" * 108)
print("STEP 58 - PHYSICOCHEMICAL CHARACTERIZATION OF PREDICTION-TRANSITION GROUPS")
print("=" * 108)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(FEATURE_FILE)
shift = pd.read_csv(SHIFT_FILE)
required_features = {"ID", "sequence", "label", "binary_class", *DESCRIPTORS}
assert required_features.issubset(features.columns)
assert len(shift) == 724 and not shift.duplicated(["ID", "classifier"]).any()

test_reference = shift[["ID", "sequence", "label", "binary_class"]].drop_duplicates().reset_index(drop=True)
assert len(test_reference) == 181 and test_reference["ID"].is_unique
test_features = test_reference.merge(
    features[["ID", "sequence", "label", "binary_class", *DESCRIPTORS]],
    on=["ID", "sequence", "label", "binary_class"], how="left", validate="one_to_one",
)
assert len(test_features) == 181
missing_values = int(test_features[DESCRIPTORS].isna().sum().sum())
nonfinite_values = int((~np.isfinite(test_features[DESCRIPTORS].to_numpy(float))).sum())
assert missing_values == 0 and nonfinite_values == 0

# Descriptive standardization uses the complete locked-test population only.
descriptor_means = test_features[DESCRIPTORS].mean()
descriptor_sds = test_features[DESCRIPTORS].std(ddof=1)
assert descriptor_sds.gt(0).all()
for descriptor in DESCRIPTORS:
    test_features[f"z_{descriptor}"] = (
        test_features[descriptor] - descriptor_means[descriptor]
    ) / descriptor_sds[descriptor]

joined = shift.merge(
    test_features[["ID", *DESCRIPTORS, *[f"z_{d}" for d in DESCRIPTORS]]],
    on="ID", how="left", validate="many_to_one",
)
assert len(joined) == 724

long_rows = []
for row in joined.itertuples(index=False):
    for descriptor in DESCRIPTORS:
        long_rows.append({
            "ID": row.ID,
            "sequence": row.sequence,
            "label": row.label,
            "binary_class": row.binary_class,
            "classifier": row.classifier,
            "transition": row.transition,
            "descriptor": descriptor,
            "descriptor_label": DESCRIPTOR_LABELS[descriptor],
            "raw_value": float(getattr(row, descriptor)),
            "test_set_zscore": float(getattr(row, f"z_{descriptor}")),
            "correct_class_probability_gain": row.correct_class_probability_gain,
        })
long_df = pd.DataFrame(long_rows)
assert len(long_df) == 724 * len(DESCRIPTORS) == 5068
assert np.isfinite(long_df[["raw_value", "test_set_zscore"]].to_numpy()).all()
long_df.to_csv(LONG_OUTPUT, index=False)

summary_rows = []
for classifier in MODEL_ORDER:
    for transition in TRANSITION_ORDER:
        for descriptor in DESCRIPTORS:
            group = long_df.loc[
                long_df["classifier"].eq(classifier)
                & long_df["transition"].eq(transition)
                & long_df["descriptor"].eq(descriptor)
            ]
            values = group["raw_value"].to_numpy(float)
            z_values = group["test_set_zscore"].to_numpy(float)
            assert len(values) > 0
            summary_rows.append({
                "classifier": classifier,
                "transition": transition,
                "descriptor": descriptor,
                "descriptor_label": DESCRIPTOR_LABELS[descriptor],
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean_test_set_zscore": float(np.mean(z_values)),
                "median_test_set_zscore": float(np.median(z_values)),
                "mean_absolute_test_set_zscore": float(np.mean(np.abs(z_values))),
            })
summary_df = pd.DataFrame(summary_rows)
assert len(summary_df) == 4 * 4 * 7 == 112
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

persistent_long = joined.loc[joined["transition"].eq("persistent_error")].copy()
assert len(persistent_long) == 32
persistent_counts = persistent_long.groupby("ID")["classifier"].nunique().rename("persistent_error_models")
persistent_peptides = test_features.loc[test_features["ID"].isin(persistent_counts.index)].copy()
persistent_peptides = persistent_peptides.merge(persistent_counts, on="ID", validate="one_to_one")
persistent_peptides["mean_absolute_physicochemical_zscore"] = persistent_peptides[
    [f"z_{d}" for d in DESCRIPTORS]
].abs().mean(axis=1)
persistent_peptides["maximum_absolute_physicochemical_zscore"] = persistent_peptides[
    [f"z_{d}" for d in DESCRIPTORS]
].abs().max(axis=1)
persistent_peptides["most_extreme_descriptor"] = persistent_peptides.apply(
    lambda row: max(DESCRIPTORS, key=lambda d: abs(row[f"z_{d}"])), axis=1
)
persistent_peptides = persistent_peptides.sort_values(
    ["persistent_error_models", "mean_absolute_physicochemical_zscore"],
    ascending=[False, False],
).reset_index(drop=True)
persistent_peptides.insert(0, "persistent_error_rank", np.arange(1, len(persistent_peptides) + 1))
persistent_peptides.to_csv(PERSISTENT_OUTPUT, index=False)

qc_df = pd.DataFrame([{
    "locked_test_peptides": len(test_features),
    "classifier_peptide_outcomes": len(joined),
    "descriptors_analyzed": len(DESCRIPTORS),
    "expected_long_rows": 724 * len(DESCRIPTORS),
    "actual_long_rows": len(long_df),
    "summary_rows": len(summary_df),
    "persistent_error_classifier_peptide_outcomes": len(persistent_long),
    "unique_peptides_with_persistent_error": len(persistent_peptides),
    "missing_descriptor_values": missing_values,
    "nonfinite_descriptor_values": nonfinite_values,
    "zscore_reference_rows": len(test_features),
    "zscore_sd_ddof": 1,
    "zscore_used_for_training": False,
    "hypothesis_tests_performed": False,
    "models_loaded_or_retrained": False,
}])
qc_df.to_csv(QC_OUTPUT, index=False)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Seven descriptive profile panels: group mean z-score, without inferential bars.
fig, axes = plt.subplots(2, 4, figsize=(12, 6.8), sharex=True, constrained_layout=True)
x = np.arange(4)
offsets = np.linspace(-0.18, 0.18, 4)
for panel_index, descriptor in enumerate(DESCRIPTORS):
    axis = axes.flat[panel_index]
    for model_index, classifier in enumerate(MODEL_ORDER):
        values = summary_df.loc[
            summary_df["classifier"].eq(classifier)
            & summary_df["descriptor"].eq(descriptor)
        ].set_index("transition").loc[TRANSITION_ORDER, "mean_test_set_zscore"].to_numpy()
        axis.plot(x + offsets[model_index], values, marker="o", markersize=4.5,
                  linewidth=1.2, color=MODEL_COLORS[classifier], label=classifier)
    axis.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
    axis.set_title(f"{'ABCDEFG'[panel_index]}   {DESCRIPTOR_LABELS[descriptor]}",
                   loc="left", fontweight="bold")
    axis.set_xticks(x, TRANSITION_LABELS, rotation=20, ha="right")
    axis.set_ylabel("Mean test-set z-score")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
axes.flat[7].axis("off")
handles, labels = axes.flat[0].get_legend_handles_labels()
axes.flat[7].legend(handles, labels, loc="center", frameon=False, title="Matched classifier")
fig.suptitle("Physicochemical profiles of prediction-transition groups", fontsize=14)
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Persistent-error heatmap ranked by persistence and overall extremeness.
heatmap = persistent_peptides[[f"z_{d}" for d in DESCRIPTORS]].to_numpy(float)
limit = max(2.0, float(np.ceil(np.max(np.abs(heatmap)) * 2) / 2))
fig_height = max(4.6, 0.30 * len(persistent_peptides) + 1.8)
fig, ax = plt.subplots(figsize=(9.5, fig_height), constrained_layout=True)
image = ax.imshow(heatmap, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
ax.set_xticks(np.arange(len(DESCRIPTORS)), [DESCRIPTOR_LABELS[d] for d in DESCRIPTORS],
              rotation=35, ha="right")
row_labels = [
    f"ID {row.ID} | {row.binary_class} | persistent {row.persistent_error_models}/4"
    for row in persistent_peptides.itertuples()
]
ax.set_yticks(np.arange(len(persistent_peptides)), row_labels)
ax.set_title("Physicochemical z-scores of persistent-error peptides", pad=10)
colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
colorbar.set_label("Test-set z-score")
ax.set_xticks(np.arange(-0.5, len(DESCRIPTORS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(persistent_peptides), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(PERSISTENT_FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PERSISTENT_FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nPersistent-error peptide ranking:")
print(persistent_peptides[[
    "persistent_error_rank", "ID", "binary_class", "persistent_error_models",
    "mean_absolute_physicochemical_zscore", "most_extreme_descriptor",
]].round(6).to_string(index=False))
print("\n58T. Output checks:")
outputs = [SUMMARY_OUTPUT, LONG_OUTPUT, PERSISTENT_OUTPUT, QC_OUTPUT, FIGURE_PNG,
           FIGURE_PDF, PERSISTENT_FIGURE_PNG, PERSISTENT_FIGURE_PDF]
for path in outputs:
    print(path.name, ":", path.exists())
print("\n" + "=" * 108)
print("STEP 58 SUMMARY")
print("=" * 108)
print("Locked-test peptides:", 181)
print("Classifier-peptide outcomes:", 724)
print("Descriptors analyzed:", len(DESCRIPTORS))
print("Persistent-error classifier-peptide outcomes:", len(persistent_long))
print("Unique peptides with >=1 persistent error:", len(persistent_peptides))
print("Missing descriptor values:", missing_values)
print("Non-finite descriptor values:", nonfinite_values)
print("\nTransition physicochemical summary:", SUMMARY_OUTPUT)
print("Long-format table:", LONG_OUTPUT)
print("Persistent-error extremes:", PERSISTENT_OUTPUT)
print("QC:", QC_OUTPUT)
print("Transition-profile figure:", FIGURE_PNG)
print("Persistent-error heatmap:", PERSISTENT_FIGURE_PNG)
print("\nSTEP 58 COMPLETED SUCCESSFULLY")
print("=" * 108)

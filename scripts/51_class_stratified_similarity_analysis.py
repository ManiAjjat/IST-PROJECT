from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
INPUT = RESULTS_DIR / "step68_similarity_stratum_peptides.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "step69_class_stratified_similarity_summary.csv"
CORRELATION_OUTPUT = RESULTS_DIR / "step69_class_similarity_correlations.csv"
RELATION_OUTPUT = RESULTS_DIR / "step69_class_neighbor_relation_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step69_class_similarity_qc.csv"
FIGURE1_PNG = FIGURES_DIR / "Step69_Class_Stratified_Difficulty.png"
FIGURE1_PDF = FIGURES_DIR / "Step69_Class_Stratified_Difficulty.pdf"
FIGURE2_PNG = FIGURES_DIR / "Step69_Class_Similarity_vs_Difficulty.png"
FIGURE2_PDF = FIGURES_DIR / "Step69_Class_Similarity_vs_Difficulty.pdf"

STRATA = ["<0.80", "0.80-<0.90", "0.90-<0.95", ">=0.95"]
CLASSES = ["Active", "Inactive"]
CLASS_COLORS = {"Active": "#D95F02", "Inactive": "#1B9E77"}
RELATION_COLORS = {"same": "#377EB8", "opposite": "#E41A1C"}


print("=" * 110)
print("STEP 69 - CLASS-STRATIFIED SIMILARITY-DIFFICULTY ANALYSIS")
print("=" * 110)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
required = [
    "test_ID", "test_sequence", "test_class", "test_label",
    "nearest_development_similarity", "similarity_stratum_short", "class_relation",
    "total_models_wrong", "traditional_models_wrong", "esm2_models_wrong",
    "mean_true_class_probability_across_8_models",
]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required Step-68 columns: {missing}")
assert len(df) == 181 and df["test_ID"].is_unique
assert df["test_label"].isin([0, 1]).all()
assert df["test_label"].eq(df["test_class"].eq("Active").astype(int)).all()
assert df["class_relation"].isin(["same", "opposite"]).all()
assert df["similarity_stratum_short"].isin(STRATA).all()

numeric_columns = [
    "nearest_development_similarity", "total_models_wrong", "traditional_models_wrong",
    "esm2_models_wrong", "mean_true_class_probability_across_8_models",
]
assert np.isfinite(df[numeric_columns].to_numpy(float)).all()
assert df["nearest_development_similarity"].between(0, 1).all()
assert df["total_models_wrong"].between(0, 8).all()
assert df["traditional_models_wrong"].between(0, 4).all()
assert df["esm2_models_wrong"].between(0, 4).all()
assert df["mean_true_class_probability_across_8_models"].between(0, 1).all()
assert (df["traditional_models_wrong"] + df["esm2_models_wrong"] == df["total_models_wrong"]).all()
assert df["test_class"].value_counts().to_dict() == {"Inactive": 161, "Active": 20}

summary_rows = []
for class_order, true_class in enumerate(CLASSES):
    for stratum_order, stratum in enumerate(STRATA):
        group = df.loc[df["test_class"].eq(true_class) & df["similarity_stratum_short"].eq(stratum)]
        n = len(group)
        summary_rows.append({
            "class_order": class_order,
            "true_class": true_class,
            "stratum_order": stratum_order,
            "similarity_stratum": stratum,
            "n": n,
            "same_class_neighbor_count": int(group["class_relation"].eq("same").sum()),
            "opposite_class_neighbor_count": int(group["class_relation"].eq("opposite").sum()),
            "mean_similarity": group["nearest_development_similarity"].mean(),
            "median_similarity": group["nearest_development_similarity"].median(),
            "minimum_similarity": group["nearest_development_similarity"].min(),
            "maximum_similarity": group["nearest_development_similarity"].max(),
            "mean_total_wrong": group["total_models_wrong"].mean(),
            "median_total_wrong": group["total_models_wrong"].median(),
            "minimum_total_wrong": int(group["total_models_wrong"].min()),
            "maximum_total_wrong": int(group["total_models_wrong"].max()),
            "mean_traditional_wrong": group["traditional_models_wrong"].mean(),
            "mean_esm2_wrong": group["esm2_models_wrong"].mean(),
            "mean_true_class_probability": group["mean_true_class_probability_across_8_models"].mean(),
            "median_true_class_probability": group["mean_true_class_probability_across_8_models"].median(),
            "any_model_wrong_count": int(group["total_models_wrong"].gt(0).sum()),
            "any_model_wrong_rate": group["total_models_wrong"].gt(0).mean(),
            "all_8_wrong_count": int(group["total_models_wrong"].eq(8).sum()),
        })

summary = pd.DataFrame(summary_rows).sort_values(["class_order", "stratum_order"]).reset_index(drop=True)
assert len(summary) == 8 and summary["n"].sum() == 181
assert summary.groupby("true_class")["n"].sum().to_dict() == {"Active": 20, "Inactive": 161}
assert (summary["same_class_neighbor_count"] + summary["opposite_class_neighbor_count"] == summary["n"]).all()
summary.to_csv(SUMMARY_OUTPUT, index=False)

correlation_rows = []
outcomes = [
    ("total_models_wrong", "total_models_wrong"),
    ("traditional_models_wrong", "traditional_models_wrong"),
    ("esm2_models_wrong", "esm2_models_wrong"),
    ("mean_true_class_probability_across_8_models", "mean_true_class_probability_across_8_models"),
]
for class_order, true_class in enumerate(CLASSES):
    group = df.loc[df["test_class"].eq(true_class)]
    for outcome_order, (outcome, column) in enumerate(outcomes):
        rho, nominal_p = spearmanr(
            group["nearest_development_similarity"].to_numpy(float),
            group[column].to_numpy(float),
        )
        correlation_rows.append({
            "class_order": class_order,
            "true_class": true_class,
            "n": len(group),
            "outcome_order": outcome_order,
            "similarity_variable": "nearest_development_similarity",
            "outcome": outcome,
            "spearman_rho": float(rho),
            "nominal_p_value_descriptive_only": float(nominal_p),
            "inferential_claim": False,
        })
correlations = pd.DataFrame(correlation_rows).sort_values(["class_order", "outcome_order"]).reset_index(drop=True)
assert len(correlations) == 8 and np.isfinite(correlations[["spearman_rho", "nominal_p_value_descriptive_only"]]).all().all()
correlations.to_csv(CORRELATION_OUTPUT, index=False)

relation_rows = []
for class_order, true_class in enumerate(CLASSES):
    for relation_order, relation in enumerate(["same", "opposite"]):
        group = df.loc[df["test_class"].eq(true_class) & df["class_relation"].eq(relation)]
        assert len(group) > 0
        relation_rows.append({
            "class_order": class_order,
            "true_class": true_class,
            "relation_order": relation_order,
            "nearest_development_relation": relation,
            "n": len(group),
            "mean_similarity": group["nearest_development_similarity"].mean(),
            "median_similarity": group["nearest_development_similarity"].median(),
            "mean_total_wrong": group["total_models_wrong"].mean(),
            "median_total_wrong": group["total_models_wrong"].median(),
            "mean_traditional_wrong": group["traditional_models_wrong"].mean(),
            "mean_esm2_wrong": group["esm2_models_wrong"].mean(),
            "mean_true_class_probability": group["mean_true_class_probability_across_8_models"].mean(),
            "median_true_class_probability": group["mean_true_class_probability_across_8_models"].median(),
            "any_model_wrong_count": int(group["total_models_wrong"].gt(0).sum()),
            "any_model_wrong_rate": group["total_models_wrong"].gt(0).mean(),
            "all_8_wrong_count": int(group["total_models_wrong"].eq(8).sum()),
        })
relations = pd.DataFrame(relation_rows).sort_values(["class_order", "relation_order"]).reset_index(drop=True)
assert len(relations) == 4
assert relations.groupby("true_class")["n"].sum().to_dict() == {"Active": 20, "Inactive": 161}
relations.to_csv(RELATION_OUTPUT, index=False)

# Figure 1: class-specific stratum means.
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), facecolor="white", sharey=True)
for ax, true_class in zip(axes, CLASSES):
    part = summary.loc[summary["true_class"].eq(true_class)].sort_values("stratum_order")
    x = np.arange(4)
    bars = ax.bar(x, part["mean_total_wrong"], color=CLASS_COLORS[true_class], alpha=0.85,
                  edgecolor="#222222", linewidth=0.65)
    ax.set_xticks(x, STRATA)
    ax.set_xlabel("Nearest-development similarity stratum")
    ax.set_ylabel("Mean number of frozen models wrong (0-8)")
    ax.set_title(f"{true_class} peptides (n={part['n'].sum()})")
    ax.set_ylim(0, 8.7)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, (_, row) in zip(bars, part.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                f"{row['mean_total_wrong']:.2f}\nn={int(row['n'])}",
                ha="center", va="bottom", fontsize=8.5)
fig.suptitle("Prediction difficulty across similarity strata within each true class")
fig.text(0.5, 0.015,
         "Predefined descriptive strata; sparse Active and upper-stratum cells are not stable population estimates.",
         ha="center", fontsize=8.8, color="#444444")
fig.subplots_adjust(bottom=0.16, top=0.87, wspace=0.13)
fig.savefig(FIGURE1_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE1_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: class-specific continuous views, colored by nearest-neighbor relation.
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), facecolor="white", sharey=True)
universal_label_positions = {
    40: (0.900, 8.47),
    48: (0.820, 8.47),
    56: (0.950, 7.10),
    68: (0.520, 8.47),
    145: (0.985, 8.47),
}
for ax, true_class in zip(axes, CLASSES):
    class_part = df.loc[df["test_class"].eq(true_class)]
    for relation in ["same", "opposite"]:
        part = class_part.loc[class_part["class_relation"].eq(relation)]
        ax.scatter(part["nearest_development_similarity"], part["total_models_wrong"],
                   s=42, alpha=0.68, color=RELATION_COLORS[relation], edgecolors="white",
                   linewidths=0.45, label=f"{relation.capitalize()} class (n={len(part)})")
    for boundary in [0.80, 0.90, 0.95]:
        ax.axvline(boundary, linestyle=":", color="#555555", linewidth=1.0)
    universal = class_part.loc[class_part["total_models_wrong"].eq(8)]
    ax.scatter(universal["nearest_development_similarity"], universal["total_models_wrong"],
               s=130, facecolors="none", edgecolors="#111111", linewidths=1.7,
               label="8/8 consensus error", zorder=4)
    for _, row in universal.iterrows():
        position = universal_label_positions[int(row["test_ID"])]
        ax.annotate(f"ID {int(row['test_ID'])}",
                    (row["nearest_development_similarity"], row["total_models_wrong"]),
                    xytext=position, textcoords="data", ha="center", fontsize=8, weight="bold",
                    arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.7})
    rho = correlations.loc[
        correlations["true_class"].eq(true_class) & correlations["outcome"].eq("total_models_wrong"),
        "spearman_rho",
    ].iloc[0]
    ax.text(0.03, 0.95, f"Descriptive Spearman rho = {rho:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.set_xlim(0.25, 1.01)
    ax.set_ylim(-0.45, 8.75)
    ax.set_yticks(range(9))
    ax.set_xlabel("Nearest-development normalized edit similarity")
    ax.set_ylabel("Frozen models wrong (0-8)")
    ax.set_title(f"{true_class} peptides (n={len(class_part)})")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=8.5)
fig.suptitle("Sequence similarity and frozen-model difficulty within true classes")
fig.text(0.5, 0.015,
         "No trend or causal model was fitted; vertical lines are the predefined 0.80, 0.90, and 0.95 boundaries.",
         ha="center", fontsize=8.8, color="#444444")
fig.subplots_adjust(bottom=0.16, top=0.87, wspace=0.13)
fig.savefig(FIGURE2_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURE2_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "locked_test_peptides": len(df),
    "unique_test_IDs": df["test_ID"].nunique(),
    "active": int(df["test_class"].eq("Active").sum()),
    "inactive": int(df["test_class"].eq("Inactive").sum()),
    "similarity_strata": len(STRATA),
    "expected_class_stratum_rows": 8,
    "observed_class_stratum_rows": len(summary),
    "expected_correlation_rows": 8,
    "observed_correlation_rows": len(correlations),
    "expected_relation_rows": 4,
    "observed_relation_rows": len(relations),
    "class_stratum_n_total": int(summary["n"].sum()),
    "minimum_total_wrong": int(df["total_models_wrong"].min()),
    "maximum_total_wrong": int(df["total_models_wrong"].max()),
    "all_numeric_values_finite": bool(np.isfinite(df[numeric_columns].to_numpy(float)).all()),
    "all_class_stratum_cells_nonempty": bool(summary["n"].gt(0).all()),
    "class_stratum_relation_counts_sum_to_n": bool((summary["same_class_neighbor_count"] + summary["opposite_class_neighbor_count"] == summary["n"]).all()),
    "traditional_plus_esm2_wrong_equals_total": bool((df["traditional_models_wrong"] + df["esm2_models_wrong"] == df["total_models_wrong"]).all()),
    "similarity_boundaries": "0.80,0.90,0.95",
    "nominal_p_values_saved_as_descriptive_diagnostics": True,
    "models_trained": False,
    "models_retrained": False,
    "trend_model_fitted": False,
    "significance_claimed": False,
    "decision_threshold_changed": False,
    "labels_changed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nClass-stratified summary:")
print(summary[["true_class", "similarity_stratum", "n", "same_class_neighbor_count",
               "opposite_class_neighbor_count", "mean_total_wrong", "mean_true_class_probability"]]
      .round(6).to_string(index=False))
print("\nDescriptive Spearman correlations:")
print(correlations[["true_class", "outcome", "spearman_rho"]].round(6).to_string(index=False))
print("\nClass and nearest-neighbor relation summary:")
print(relations[["true_class", "nearest_development_relation", "n", "mean_similarity",
                 "mean_total_wrong", "mean_true_class_probability", "all_8_wrong_count"]]
      .round(6).to_string(index=False))
print("\nSTEP 69 COMPLETED SUCCESSFULLY")
print("=" * 110)

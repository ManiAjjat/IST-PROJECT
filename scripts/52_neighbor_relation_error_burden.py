from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
INPUT = PROJECT_DIR / "results" / "step68_similarity_stratum_peptides.csv"
RESULTS = PROJECT_DIR / "results"
FIGURES = PROJECT_DIR / "figures"

BURDEN_OUT = RESULTS / "step70_neighbor_relation_error_burden.csv"
DIFF_OUT = RESULTS / "step70_neighbor_relation_differences.csv"
PAIRS_OUT = RESULTS / "step70_similarity_matched_pairs.csv"
MATCHED_SUMMARY_OUT = RESULTS / "step70_similarity_matched_summary.csv"
QC_OUT = RESULTS / "step70_neighbor_relation_qc.csv"
FIG1_PNG = FIGURES / "Step70_Neighbor_Relation_Error_Burden.png"
FIG1_PDF = FIGURES / "Step70_Neighbor_Relation_Error_Burden.pdf"
FIG2_PNG = FIGURES / "Step70_Matched_Neighbor_Relation_Differences.png"
FIG2_PDF = FIGURES / "Step70_Matched_Neighbor_Relation_Differences.pdf"

ID = "test_ID"
CLASS = "test_class"
RELATION = "class_relation"
STRATUM = "similarity_stratum_short"
SIM = "nearest_development_similarity"
TOTAL_WRONG = "total_models_wrong"
TRAD_WRONG = "traditional_models_wrong"
ESM_WRONG = "esm2_models_wrong"
CONFIDENCE = "mean_true_class_probability_across_8_models"

STRATA = ["<0.80", "0.80-<0.90", "0.90-<0.95", ">=0.95"]
POPULATIONS = ["All", "Active", "Inactive"]
RELATIONS = ["same", "opposite"]


def subset_population(frame, population):
    return frame if population == "All" else frame.loc[frame[CLASS] == population]


def summarize(frame, population, stratum, relation):
    part = subset_population(frame, population)
    if stratum != "All":
        part = part.loc[part[STRATUM] == stratum]
    part = part.loc[part[RELATION] == relation]
    if part.empty:
        return None
    wrong = part[TOTAL_WRONG]
    confidence = part[CONFIDENCE]
    return {
        "population": population,
        "similarity_stratum": stratum,
        "neighbor_class_relation": relation,
        "n": len(part),
        "active_n": int((part[CLASS] == "Active").sum()),
        "inactive_n": int((part[CLASS] == "Inactive").sum()),
        "mean_nearest_development_similarity": part[SIM].mean(),
        "median_nearest_development_similarity": part[SIM].median(),
        "mean_total_models_wrong": wrong.mean(),
        "median_total_models_wrong": wrong.median(),
        "mean_traditional_models_wrong": part[TRAD_WRONG].mean(),
        "mean_esm2_models_wrong": part[ESM_WRONG].mean(),
        "mean_true_class_probability": confidence.mean(),
        "median_true_class_probability": confidence.median(),
        "any_model_wrong_count": int((wrong > 0).sum()),
        "any_model_wrong_fraction": (wrong > 0).mean(),
        "consensus_8_of_8_error_count": int((wrong == 8).sum()),
        "consensus_8_of_8_error_fraction": (wrong == 8).mean(),
    }


print("=" * 104)
print("STEP 70 - SAME-CLASS VS OPPOSITE-CLASS HOMOLOGY ERROR-BURDEN ANALYSIS")
print("=" * 104)

df = pd.read_csv(INPUT)
required = [ID, CLASS, RELATION, STRATUM, SIM, TOTAL_WRONG, TRAD_WRONG, ESM_WRONG, CONFIDENCE]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if len(df) != 181 or df[ID].nunique() != 181:
    raise ValueError("Expected 181 unique locked-test peptides")
if set(df[CLASS]) != {"Active", "Inactive"}:
    raise ValueError("Unexpected class values")
if set(df[RELATION]) != set(RELATIONS):
    raise ValueError("Unexpected class-relation values")
if set(df[STRATUM]) != set(STRATA):
    raise ValueError("Unexpected similarity strata")
if not np.isfinite(df[[SIM, TOTAL_WRONG, TRAD_WRONG, ESM_WRONG, CONFIDENCE]]).all().all():
    raise ValueError("Non-finite analysis values")
if not np.array_equal((df[TRAD_WRONG] + df[ESM_WRONG]).to_numpy(), df[TOTAL_WRONG].to_numpy()):
    raise ValueError("Traditional plus ESM-2 wrong counts do not equal total wrong")

burden_rows = []
for population in POPULATIONS:
    for stratum in ["All"] + STRATA:
        for relation in RELATIONS:
            row = summarize(df, population, stratum, relation)
            if row is not None:
                burden_rows.append(row)
burden = pd.DataFrame(burden_rows)
burden.to_csv(BURDEN_OUT, index=False)

difference_rows = []
for population in POPULATIONS:
    for stratum in ["All"] + STRATA:
        cell = burden.loc[(burden.population == population) & (burden.similarity_stratum == stratum)]
        if set(cell.neighbor_class_relation) != set(RELATIONS):
            continue
        same = cell.loc[cell.neighbor_class_relation == "same"].iloc[0]
        opposite = cell.loc[cell.neighbor_class_relation == "opposite"].iloc[0]
        difference_rows.append({
            "population": population,
            "similarity_stratum": stratum,
            "same_n": int(same.n),
            "opposite_n": int(opposite.n),
            "opposite_minus_same_mean_total_models_wrong": opposite.mean_total_models_wrong - same.mean_total_models_wrong,
            "opposite_minus_same_mean_traditional_models_wrong": opposite.mean_traditional_models_wrong - same.mean_traditional_models_wrong,
            "opposite_minus_same_mean_esm2_models_wrong": opposite.mean_esm2_models_wrong - same.mean_esm2_models_wrong,
            "opposite_minus_same_mean_nearest_similarity": opposite.mean_nearest_development_similarity - same.mean_nearest_development_similarity,
            "same_minus_opposite_mean_true_class_probability": same.mean_true_class_probability - opposite.mean_true_class_probability,
            "same_minus_opposite_median_true_class_probability": same.median_true_class_probability - opposite.median_true_class_probability,
        })
differences = pd.DataFrame(difference_rows)
differences.to_csv(DIFF_OUT, index=False)

# Deterministic greedy one-to-one matching. Opposite-relation peptides are
# processed by true class, stratum order, then ascending ID. Candidate ties are
# resolved by absolute similarity difference and then ascending same-relation ID.
stratum_rank = {value: index for index, value in enumerate(STRATA)}
opposites = df.loc[df[RELATION] == "opposite"].copy()
opposites["_stratum_order"] = opposites[STRATUM].map(stratum_rank)
opposites = opposites.sort_values([CLASS, "_stratum_order", ID], kind="mergesort")
used_same_ids = set()
pair_rows = []
for _, opp in opposites.iterrows():
    candidates = df.loc[
        (df[RELATION] == "same")
        & (df[CLASS] == opp[CLASS])
        & (df[STRATUM] == opp[STRATUM])
        & (~df[ID].isin(used_same_ids))
    ].copy()
    if candidates.empty:
        pair_rows.append({
            "opposite_ID": opp[ID], "true_class": opp[CLASS], "similarity_stratum": opp[STRATUM],
            "matched": False, "unmatched_reason": "no unused same-relation peptide in same class and stratum",
            "opposite_similarity": opp[SIM], "same_ID": np.nan, "same_similarity": np.nan,
            "absolute_similarity_difference": np.nan,
            "opposite_total_models_wrong": opp[TOTAL_WRONG], "same_total_models_wrong": np.nan,
            "delta_total_models_wrong_opposite_minus_same": np.nan,
            "opposite_traditional_models_wrong": opp[TRAD_WRONG], "same_traditional_models_wrong": np.nan,
            "delta_traditional_models_wrong_opposite_minus_same": np.nan,
            "opposite_esm2_models_wrong": opp[ESM_WRONG], "same_esm2_models_wrong": np.nan,
            "delta_esm2_models_wrong_opposite_minus_same": np.nan,
            "opposite_mean_true_class_probability": opp[CONFIDENCE], "same_mean_true_class_probability": np.nan,
            "delta_true_class_probability_opposite_minus_same": np.nan,
        })
        continue
    candidates["_distance"] = (candidates[SIM] - opp[SIM]).abs()
    same = candidates.sort_values(["_distance", ID], kind="mergesort").iloc[0]
    used_same_ids.add(same[ID])
    pair_rows.append({
        "opposite_ID": opp[ID], "true_class": opp[CLASS], "similarity_stratum": opp[STRATUM],
        "matched": True, "unmatched_reason": "", "opposite_similarity": opp[SIM],
        "same_ID": int(same[ID]), "same_similarity": same[SIM],
        "absolute_similarity_difference": abs(opp[SIM] - same[SIM]),
        "opposite_total_models_wrong": opp[TOTAL_WRONG], "same_total_models_wrong": same[TOTAL_WRONG],
        "delta_total_models_wrong_opposite_minus_same": opp[TOTAL_WRONG] - same[TOTAL_WRONG],
        "opposite_traditional_models_wrong": opp[TRAD_WRONG], "same_traditional_models_wrong": same[TRAD_WRONG],
        "delta_traditional_models_wrong_opposite_minus_same": opp[TRAD_WRONG] - same[TRAD_WRONG],
        "opposite_esm2_models_wrong": opp[ESM_WRONG], "same_esm2_models_wrong": same[ESM_WRONG],
        "delta_esm2_models_wrong_opposite_minus_same": opp[ESM_WRONG] - same[ESM_WRONG],
        "opposite_mean_true_class_probability": opp[CONFIDENCE], "same_mean_true_class_probability": same[CONFIDENCE],
        "delta_true_class_probability_opposite_minus_same": opp[CONFIDENCE] - same[CONFIDENCE],
    })
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(PAIRS_OUT, index=False)

matched = pairs.loc[pairs.matched].copy()
matched_summary_rows = []
for population in POPULATIONS:
    part = matched if population == "All" else matched.loc[matched.true_class == population]
    entered = pairs if population == "All" else pairs.loc[pairs.true_class == population]
    dw = part.delta_total_models_wrong_opposite_minus_same
    dc = part.delta_true_class_probability_opposite_minus_same
    matched_summary_rows.append({
        "population": population, "opposite_peptides_entered": len(entered),
        "matched_pairs": len(part), "unmatched_opposite_peptides": int((~entered.matched).sum()),
        "mean_absolute_similarity_difference": part.absolute_similarity_difference.mean(),
        "median_absolute_similarity_difference": part.absolute_similarity_difference.median(),
        "mean_delta_total_models_wrong_opposite_minus_same": dw.mean(),
        "median_delta_total_models_wrong_opposite_minus_same": dw.median(),
        "opposite_more_models_wrong_count": int((dw > 0).sum()),
        "equal_models_wrong_count": int((dw == 0).sum()),
        "opposite_fewer_models_wrong_count": int((dw < 0).sum()),
        "mean_delta_traditional_models_wrong_opposite_minus_same": part.delta_traditional_models_wrong_opposite_minus_same.mean(),
        "mean_delta_esm2_models_wrong_opposite_minus_same": part.delta_esm2_models_wrong_opposite_minus_same.mean(),
        "mean_delta_true_class_probability_opposite_minus_same": dc.mean(),
        "median_delta_true_class_probability_opposite_minus_same": dc.median(),
        "opposite_lower_true_class_probability_count": int((dc < 0).sum()),
        "equal_true_class_probability_count": int(np.isclose(dc, 0).sum()),
        "opposite_higher_true_class_probability_count": int((dc > 0).sum()),
    })
matched_summary = pd.DataFrame(matched_summary_rows)
matched_summary.to_csv(MATCHED_SUMMARY_OUT, index=False)

relation_counts = df.groupby([CLASS, RELATION]).size().unstack(fill_value=0)
qc = pd.DataFrame([{
    "input_rows": len(df), "unique_IDs": df[ID].nunique(),
    "active_rows": int((df[CLASS] == "Active").sum()), "inactive_rows": int((df[CLASS] == "Inactive").sum()),
    "same_relation_rows": int((df[RELATION] == "same").sum()), "opposite_relation_rows": int((df[RELATION] == "opposite").sum()),
    "active_same_rows": int(relation_counts.loc["Active", "same"]), "active_opposite_rows": int(relation_counts.loc["Active", "opposite"]),
    "inactive_same_rows": int(relation_counts.loc["Inactive", "same"]), "inactive_opposite_rows": int(relation_counts.loc["Inactive", "opposite"]),
    "burden_rows": len(burden), "difference_rows": len(differences), "opposite_peptides_entered_matching": len(pairs),
    "matched_pairs": len(matched), "unmatched_opposite_peptides": int((~pairs.matched).sum()),
    "unique_same_IDs_used": matched.same_ID.nunique(), "same_ID_reuse_count": len(matched) - matched.same_ID.nunique(),
    "matched_class_mismatches": int((matched.true_class != matched.same_ID.map(df.set_index(ID)[CLASS])).sum()),
    "matched_stratum_mismatches": int((matched.similarity_stratum != matched.same_ID.map(df.set_index(ID)[STRATUM])).sum()),
    "maximum_matched_similarity_difference": matched.absolute_similarity_difference.max(),
    "all_numeric_analysis_values_finite": True,
    "traditional_plus_esm_equals_total": True,
    "significance_tests_performed": 0,
}])
qc.to_csv(QC_OUT, index=False)

# Figure 1: overall (not stratum-specific) descriptive error burden.
overall = burden.loc[burden.similarity_stratum == "All"].copy()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
x = np.arange(len(POPULATIONS)); width = 0.34
colors = {"same": "#2878B5", "opposite": "#D9534F"}
for offset, relation in zip([-width / 2, width / 2], RELATIONS):
    values = [overall.loc[(overall.population == p) & (overall.neighbor_class_relation == relation), "mean_total_models_wrong"].iloc[0] for p in POPULATIONS]
    bars = axes[0].bar(x + offset, values, width, label=f"{relation.capitalize()} class", color=colors[relation])
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    values = [overall.loc[(overall.population == p) & (overall.neighbor_class_relation == relation), "mean_true_class_probability"].iloc[0] for p in POPULATIONS]
    bars = axes[1].bar(x + offset, values, width, label=f"{relation.capitalize()} class", color=colors[relation])
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
axes[0].set(title="Eight-model error burden", ylabel="Mean number of models wrong (0–8)", xticks=x, xticklabels=POPULATIONS, ylim=(0, 2.3))
axes[1].set(title="Frozen prediction confidence", ylabel="Mean true-class probability", xticks=x, xticklabels=POPULATIONS, ylim=(0, 1.06))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.22); ax.set_axisbelow(True)
axes[1].legend(frameon=False, loc="lower left")
fig.suptitle("Error burden by nearest-development-neighbor class relation", fontsize=14, fontweight="bold")
fig.savefig(FIG1_PNG, dpi=420, facecolor="white")
fig.savefig(FIG1_PDF, facecolor="white")
plt.close(fig)

# Figure 2: paired deltas; positive wrong-count values favor the same-class match,
# whereas negative confidence values indicate lower confidence for the opposite-class peptide.
plot_pairs = matched.sort_values(["true_class", "similarity_stratum", "opposite_ID"], kind="mergesort").reset_index(drop=True)
labels = [f"{row.opposite_ID:g}→{row.same_ID:g}" for _, row in plot_pairs.iterrows()]
y = np.arange(len(plot_pairs)); point_colors = plot_pairs.true_class.map({"Active": "#E69F00", "Inactive": "#4C78A8"})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2), sharey=True, constrained_layout=True)
for ax, column, title, xlabel in [
    (axes[0], "delta_total_models_wrong_opposite_minus_same", "Error-count difference", "Opposite minus same: models wrong"),
    (axes[1], "delta_true_class_probability_opposite_minus_same", "Confidence difference", "Opposite minus same: true-class probability"),
]:
    values = plot_pairs[column].to_numpy()
    ax.hlines(y, 0, values, color=point_colors, linewidth=2, alpha=0.78)
    ax.scatter(values, y, c=point_colors, s=38, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set(title=title, xlabel=xlabel, yticks=y, yticklabels=labels)
    ax.grid(axis="x", alpha=0.22); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()
axes[0].set_ylabel("Opposite-class ID → matched same-class ID")
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00", label="Active", markersize=7), Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", label="Inactive", markersize=7)], frameon=False, loc="lower right")
fig.suptitle("Similarity-matched neighbor-relation differences", fontsize=14, fontweight="bold")
fig.savefig(FIG2_PNG, dpi=420, facecolor="white")
fig.savefig(FIG2_PDF, facecolor="white")
plt.close(fig)

print("\nInput relation counts:")
print(pd.crosstab(df[CLASS], df[RELATION]).to_string())
print("\nOverall burden:")
print(overall[["population", "neighbor_class_relation", "n", "mean_total_models_wrong", "mean_true_class_probability"]].round(6).to_string(index=False))
print("\nSimilarity-matched summary:")
print(matched_summary.round(6).to_string(index=False))
print("\nOutputs:")
for path in [BURDEN_OUT, DIFF_OUT, PAIRS_OUT, MATCHED_SUMMARY_OUT, QC_OUT, FIG1_PNG, FIG1_PDF, FIG2_PNG, FIG2_PDF]:
    print(path)
print("\nSTEP 70 COMPLETED SUCCESSFULLY")
print("=" * 104)

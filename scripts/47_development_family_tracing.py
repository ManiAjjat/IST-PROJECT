from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
CLUSTER_FILE = RESULTS_DIR / "step62_hard_case_cluster_assignments.csv"
TOP_FILE = RESULTS_DIR / "step63_hard_case_top_development_neighbors.csv"
STEP63_MAIN_FILE = RESULTS_DIR / "step63_hard_case_development_neighbors.csv"
RANKING_FILE = RESULTS_DIR / "step59_consensus_hard_case_ranking.csv"
METADATA_FILE = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
EDGE_OUTPUT = RESULTS_DIR / "step65_family_development_neighbor_edges.csv"
RECURRENT_OUTPUT = RESULTS_DIR / "step65_recurrent_development_neighbors.csv"
MEMBER_OUTPUT = RESULTS_DIR / "step65_family_member_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step65_family_tracing_qc.csv"
NETWORK_PNG = FIGURES_DIR / "Step65_Hard_Family_Development_Network.png"
NETWORK_PDF = FIGURES_DIR / "Step65_Hard_Family_Development_Network.pdf"
RECURRENT_PNG = FIGURES_DIR / "Step65_Recurrent_Development_Neighbors.png"
RECURRENT_PDF = FIGURES_DIR / "Step65_Recurrent_Development_Neighbors.pdf"
EXPECTED_IDS = {33, 40, 43, 47, 48, 149}
REPRESENTATION_NAMES = {
    "normalized_edit_similarity": "sequence",
    "esm2_cosine_similarity": "esm2",
}


print("=" * 112)
print("STEP 65 - DEVELOPMENT-SET FAMILY TRACING AND CLASS-CONFLICT MAP")
print("=" * 112)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

clusters = pd.read_csv(CLUSTER_FILE)
top = pd.read_csv(TOP_FILE)
step63_main = pd.read_csv(STEP63_MAIN_FILE)
ranking = pd.read_csv(RANKING_FILE)
metadata = pd.read_csv(METADATA_FILE)
at_080 = clusters.loc[np.isclose(clusters["threshold"], 0.80, rtol=0, atol=1e-12)]
non_singletons = at_080.loc[at_080["cluster_size"] > 1]
component_ids = non_singletons["cluster_id"].unique()
assert len(component_ids) == 1
family = non_singletons.loc[non_singletons["cluster_id"] == component_ids[0]].sort_values("ID")
family_ids = set(family["ID"].astype(int))
assert len(family) == 6 and family_ids == EXPECTED_IDS
assert family["true_class"].eq("Inactive").all()

edges = top.loc[top["hard_case_ID"].isin(family_ids)].copy()
assert len(edges) == 60
edges["representation"] = edges["similarity_type"].map(REPRESENTATION_NAMES)
assert edges["representation"].notna().all() and set(edges["representation"]) == {"sequence", "esm2"}
assert edges.groupby(["hard_case_ID", "representation"])["neighbor_rank"].apply(
    lambda values: set(values) == {1, 2, 3, 4, 5}
).all()
edges = edges.rename(columns={
    "hard_case_ID": "family_ID", "hard_case_sequence": "family_sequence",
    "hard_case_class": "family_true_class", "hard_case_total_wrong_count": "family_total_wrong_count",
    "hard_case_consensus_8_of_8_error": "family_consensus_8_of_8",
})
edges["same_true_class"] = edges["class_relation"].eq("same")
edges["edge_class"] = np.where(edges["same_true_class"], "same_class", "opposite_class")
edge_columns = [
    "family_ID", "family_sequence", "family_true_class", "family_total_wrong_count",
    "family_consensus_8_of_8", "representation", "neighbor_rank",
    "development_neighbor_ID", "development_neighbor_sequence", "development_neighbor_class",
    "same_true_class", "similarity", "edge_class",
]
edges = edges[edge_columns].sort_values(["family_ID", "representation", "neighbor_rank"]).reset_index(drop=True)

development = metadata.loc[metadata["split"] == "development"]
dev_ids = set(development["ID"].astype(int))
test_ids = set(metadata.loc[metadata["split"] == "test", "ID"].astype(int))
assert set(edges["development_neighbor_ID"]).issubset(dev_ids)
assert set(edges["development_neighbor_ID"]).isdisjoint(test_ids)
assert np.isfinite(edges["similarity"]).all()
assert edges.loc[edges["representation"] == "sequence", "similarity"].between(0, 1).all()
assert edges.loc[edges["representation"] == "esm2", "similarity"].between(-1, 1).all()
edges.to_csv(EDGE_OUTPUT, index=False)

recurrent_rows = []
for (representation, neighbor_id), part in edges.groupby(
    ["representation", "development_neighbor_ID"], sort=False
):
    linked_ids = sorted(part["family_ID"].astype(int).unique())
    recurrent_rows.append({
        "representation": representation,
        "development_neighbor_ID": int(neighbor_id),
        "development_neighbor_sequence": part["development_neighbor_sequence"].iloc[0],
        "development_neighbor_class": part["development_neighbor_class"].iloc[0],
        "number_of_family_members_linked": len(linked_ids),
        "total_edge_count": len(part),
        "minimum_neighbor_rank": int(part["neighbor_rank"].min()),
        "mean_similarity": part["similarity"].mean(),
        "maximum_similarity": part["similarity"].max(),
        "same_class_edges": int(part["edge_class"].eq("same_class").sum()),
        "opposite_class_edges": int(part["edge_class"].eq("opposite_class").sum()),
        "linked_family_IDs": ";".join(map(str, linked_ids)),
    })
recurrent = pd.DataFrame(recurrent_rows).sort_values(
    ["representation", "number_of_family_members_linked", "minimum_neighbor_rank",
     "mean_similarity", "development_neighbor_ID"],
    ascending=[True, False, True, False, True],
).reset_index(drop=True)
recurrent["recurrence_rank_within_representation"] = recurrent.groupby("representation").cumcount() + 1
assert recurrent["number_of_family_members_linked"].le(6).all()
assert (recurrent["same_class_edges"] + recurrent["opposite_class_edges"]).eq(
    recurrent["total_edge_count"]
).all()
recurrent.to_csv(RECURRENT_OUTPUT, index=False)

nearest_opposite = step63_main.loc[step63_main["ID"].isin(family_ids), [
    "ID", "sequence_nearest_opposite_ID", "esm2_nearest_opposite_ID"
]].set_index("ID")
member_rows = []
for (family_id, representation), part in edges.groupby(["family_ID", "representation"], sort=True):
    part = part.sort_values("neighbor_rank")
    nearest = part.iloc[0]
    same_count = int(part["same_true_class"].sum())
    opposite_count = len(part) - same_count
    sequence_opposite = int(nearest_opposite.loc[family_id, "sequence_nearest_opposite_ID"])
    esm_opposite = int(nearest_opposite.loc[family_id, "esm2_nearest_opposite_ID"])
    member_rows.append({
        "family_ID": int(family_id), "family_sequence": part["family_sequence"].iloc[0],
        "family_true_class": part["family_true_class"].iloc[0],
        "family_total_wrong_count": int(part["family_total_wrong_count"].iloc[0]),
        "family_consensus_8_of_8": bool(part["family_consensus_8_of_8"].iloc[0]),
        "representation": representation, "top5_same_class_count": same_count,
        "top5_opposite_class_count": opposite_count,
        "top5_same_class_fraction": same_count / 5,
        "top5_opposite_class_fraction": opposite_count / 5,
        "mean_top5_similarity": part["similarity"].mean(),
        "nearest_development_neighbor_ID": int(nearest["development_neighbor_ID"]),
        "nearest_neighbor_class": nearest["development_neighbor_class"],
        "nearest_neighbor_similarity": float(nearest["similarity"]),
        "nearest_opposite_sequence_ID": sequence_opposite,
        "nearest_opposite_esm2_ID": esm_opposite,
        "same_recurrent_opposite_neighbor_across_representations": sequence_opposite == esm_opposite,
    })
member_summary = pd.DataFrame(member_rows).sort_values(["family_ID", "representation"]).reset_index(drop=True)
assert len(member_summary) == 12
member_summary.to_csv(MEMBER_OUTPUT, index=False)

# Network includes recurring neighbors or a rank-1 neighbor for at least one family member.
include_ids = set(recurrent.loc[recurrent["number_of_family_members_linked"] >= 2, "development_neighbor_ID"])
include_ids.update(edges.loc[edges["neighbor_rank"] == 1, "development_neighbor_ID"])
network_edges = edges.loc[edges["development_neighbor_ID"].isin(include_ids)].copy()
neighbor_info = network_edges.drop_duplicates("development_neighbor_ID").set_index("development_neighbor_ID")
family_order = sorted(family_ids)
neighbor_order = sorted(include_ids, key=lambda value: (
    0 if neighbor_info.loc[value, "development_neighbor_class"] == "Active" else 1, value
))
family_y = dict(zip(family_order, np.linspace(0.92, 0.08, len(family_order))))
neighbor_y = dict(zip(neighbor_order, np.linspace(0.95, 0.05, len(neighbor_order))))
fig, ax = plt.subplots(figsize=(13.2, 9.0), facecolor="white")
for row in network_edges.itertuples(index=False):
    color = "#D73027" if row.development_neighbor_class == "Active" else "#4575B4"
    ax.plot([0.18, 0.82], [family_y[row.family_ID], neighbor_y[row.development_neighbor_ID]],
            linestyle="-" if row.representation == "esm2" else "--",
            linewidth=0.7 + 2.0 * row.similarity, color=color, alpha=0.42, zorder=1)
for family_id in family_order:
    consensus = family_id in {40, 48}
    ax.scatter(0.18, family_y[family_id], s=650 if consensus else 510, marker="s",
               facecolor="#FEE08B", edgecolor="#B2182B" if consensus else "#333333",
               linewidth=3.2 if consensus else 1.5, zorder=4)
    ax.text(0.145, family_y[family_id], f"ID {family_id}", ha="right", va="center", fontsize=11,
            fontweight="bold" if consensus else "normal")
for neighbor_id in neighbor_order:
    info = neighbor_info.loc[neighbor_id]
    active = info["development_neighbor_class"] == "Active"
    ax.scatter(0.82, neighbor_y[neighbor_id], s=390, marker="^" if active else "o",
               facecolor="#D73027" if active else "#4575B4", edgecolor="white", linewidth=1.2, zorder=4)
    ax.text(0.85, neighbor_y[neighbor_id], f"ID {neighbor_id} ({'A' if active else 'I'})",
            ha="left", va="center", fontsize=9.5)
ax.text(0.18, 1.02, "Hard Inactive family", ha="center", fontsize=13, fontweight="bold")
ax.text(0.82, 1.02, "Selected development neighbors", ha="center", fontsize=13, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.06); ax.axis("off")
ax.set_title("Repeated development-neighbor links of the six-member hard-case family", pad=22)
legend = [
    Line2D([0], [0], color="#555555", linestyle="-", label="ESM-2 edge"),
    Line2D([0], [0], color="#555555", linestyle="--", label="Sequence edge"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#D73027", label="Development Active"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#4575B4", label="Development Inactive"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#FEE08B", markeredgecolor="#B2182B",
           markeredgewidth=2.5, label="Hard family 8/8 failure"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig(NETWORK_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(NETWORK_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qualifying = recurrent.loc[recurrent["number_of_family_members_linked"] >= 2].copy()
if qualifying["development_neighbor_ID"].nunique() < 5:
    supplement = recurrent.loc[~recurrent["development_neighbor_ID"].isin(qualifying["development_neighbor_ID"])].copy()
    needed = 5 - qualifying["development_neighbor_ID"].nunique()
    extra_ids = supplement.sort_values(
        ["number_of_family_members_linked", "minimum_neighbor_rank", "mean_similarity", "development_neighbor_ID"],
        ascending=[False, True, False, True],
    )["development_neighbor_ID"].drop_duplicates().head(needed)
    qualifying = pd.concat([qualifying, recurrent.loc[recurrent["development_neighbor_ID"].isin(extra_ids)]])
plot_ids = sorted(qualifying["development_neighbor_ID"].unique(), key=lambda value: (
    -int(recurrent.loc[recurrent["development_neighbor_ID"].eq(value), "number_of_family_members_linked"].max()), value
))
plot_data = recurrent.loc[recurrent["development_neighbor_ID"].isin(plot_ids)].copy()
lookup = plot_data.pivot(index="development_neighbor_ID", columns="representation",
                         values="number_of_family_members_linked").reindex(plot_ids).fillna(0)
class_lookup = recurrent.drop_duplicates("development_neighbor_ID").set_index("development_neighbor_ID")["development_neighbor_class"]
y = np.arange(len(plot_ids)); height = 0.36
fig, ax = plt.subplots(figsize=(10.8, max(5.8, 0.48 * len(plot_ids) + 2.3)), facecolor="white")
ax.barh(y - height / 2, lookup.get("sequence", pd.Series(0, index=lookup.index)), height,
        color="#377EB8", label="Sequence")
ax.barh(y + height / 2, lookup.get("esm2", pd.Series(0, index=lookup.index)), height,
        color="#E41A1C", label="ESM-2")
labels = [f"ID {value} ({'A' if class_lookup[value] == 'Active' else 'I'})" for value in plot_ids]
ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 6.7)
ax.set_xticks(range(7)); ax.set_xlabel("Number of six hard-family members linked in top five")
ax.set_ylabel("Development peptide ID (class)")
ax.set_title("Recurrent development neighbors of the six-member hard Inactive family")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.7); ax.set_axisbelow(True)
ax.legend(frameon=False)
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3, fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(RECURRENT_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(RECURRENT_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "step62_threshold": 0.80, "non_singleton_component_count": len(component_ids),
    "family_size": len(family), "expected_family_IDs_recovered": family_ids == EXPECTED_IDS,
    "family_IDs": ";".join(map(str, sorted(family_ids))), "input_top_neighbor_rows": len(edges),
    "representation_count": edges["representation"].nunique(),
    "representations_exact": ";".join(sorted(edges["representation"].unique())),
    "ranks_1_to_5_each": bool(edges.groupby(["family_ID", "representation"])["neighbor_rank"].apply(
        lambda values: set(values) == {1, 2, 3, 4, 5}).all()),
    "all_neighbors_in_development": bool(set(edges["development_neighbor_ID"]).issubset(dev_ids)),
    "test_neighbors_entering_edges": len(set(edges["development_neighbor_ID"]) & test_ids),
    "all_similarities_finite": bool(np.isfinite(edges["similarity"]).all()),
    "sequence_similarities_within_0_1": bool(edges.loc[edges["representation"].eq("sequence"), "similarity"].between(0, 1).all()),
    "esm2_similarities_within_minus1_1": bool(edges.loc[edges["representation"].eq("esm2"), "similarity"].between(-1, 1).all()),
    "maximum_recurrent_count": int(recurrent["number_of_family_members_linked"].max()),
    "recurrent_counts_at_most_6": bool(recurrent["number_of_family_members_linked"].le(6).all()),
    "edge_class_totals_match": bool((recurrent["same_class_edges"] + recurrent["opposite_class_edges"]).eq(recurrent["total_edge_count"]).all()),
    "network_development_nodes": len(include_ids), "network_edges": len(network_edges),
    "models_fitted": False, "thresholds_optimized": False, "labels_changed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nTop recurrent sequence neighbors:")
print(recurrent.loc[recurrent["representation"].eq("sequence"), [
    "development_neighbor_ID", "development_neighbor_class", "number_of_family_members_linked",
    "minimum_neighbor_rank", "mean_similarity", "linked_family_IDs"
]].head(10).round(6).to_string(index=False))
print("\nTop recurrent ESM-2 neighbors:")
print(recurrent.loc[recurrent["representation"].eq("esm2"), [
    "development_neighbor_ID", "development_neighbor_class", "number_of_family_members_linked",
    "minimum_neighbor_rank", "mean_similarity", "linked_family_IDs"
]].head(10).round(6).to_string(index=False))
majority = member_summary.assign(majority_opposite=member_summary["top5_opposite_class_count"] > 2)
print("\nMajority-opposite top-five by sequence:", int(majority.loc[majority["representation"].eq("sequence"), "majority_opposite"].sum()))
print("Majority-opposite top-five by ESM-2:", int(majority.loc[majority["representation"].eq("esm2"), "majority_opposite"].sum()))
print("Same nearest opposite across representations:", int(member_summary.drop_duplicates("family_ID")["same_recurrent_opposite_neighbor_across_representations"].sum()))
print("\nSTEP 65 COMPLETED SUCCESSFULLY")
print("=" * 112)

from pathlib import Path

from Bio.Align import PairwiseAligner
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

HARD_FILE = RESULTS_DIR / "step59_consensus_hard_cases_manuscript.csv"
STEP61_FILE = RESULTS_DIR / "step61_sequence_nearest_neighbor_analysis.csv"

PAIRWISE_OUTPUT = RESULTS_DIR / "step62_hard_case_pairwise_similarity.csv"
ASSIGNMENT_OUTPUT = RESULTS_DIR / "step62_hard_case_cluster_assignments.csv"
CLUSTER_OUTPUT = RESULTS_DIR / "step62_hard_case_cluster_summary.csv"
THRESHOLD_OUTPUT = RESULTS_DIR / "step62_cluster_threshold_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step62_sequence_cluster_qc.csv"
HEATMAP_PNG = FIGURES_DIR / "Step62_Hard_Case_Sequence_Similarity_Map.png"
HEATMAP_PDF = FIGURES_DIR / "Step62_Hard_Case_Sequence_Similarity_Map.pdf"
NETWORK_PNG = FIGURES_DIR / "Step62_Hard_Case_Cluster_Network.png"
NETWORK_PDF = FIGURES_DIR / "Step62_Hard_Case_Cluster_Network.pdf"

THRESHOLDS = (0.70, 0.80, 0.90)
NETWORK_THRESHOLD = 0.80


def connected_components(ids, edges):
    adjacency = {peptide_id: set() for peptide_id in ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(ids)
    components = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        components.append(sorted(component))
    return sorted(components, key=lambda members: min(members))


def class_pattern(active_count, inactive_count):
    if active_count and inactive_count:
        return "Mixed-class"
    if active_count:
        return "Active-only"
    return "Inactive-only"


print("=" * 110)
print("STEP 62 - CLASS-AWARE HARD-CASE SEQUENCE-FAMILY CLUSTERING")
print("=" * 110)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

hard = pd.read_csv(HARD_FILE).sort_values("rank").reset_index(drop=True)
step61 = pd.read_csv(STEP61_FILE)
required = {"rank", "ID", "sequence", "true_class", "y_true", "total_wrong_count"}
assert required.issubset(hard.columns) and required.issubset(step61.columns)
assert len(hard) == 15 and hard["ID"].is_unique and hard["total_wrong_count"].ge(3).all()
aligned_step61 = hard[["ID", "sequence", "true_class", "y_true"]].merge(
    step61[["ID", "sequence", "true_class", "y_true"]],
    on=["ID", "sequence", "true_class", "y_true"], how="inner", validate="one_to_one",
)
assert len(aligned_step61) == 15

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 0.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -1.0
aligner.extend_gap_score = -1.0

n_hard = len(hard)
similarity_matrix = np.eye(n_hard, dtype=np.float64)
pair_rows = []
for left in range(n_hard):
    for right in range(left + 1, n_hard):
        left_row = hard.iloc[left]
        right_row = hard.iloc[right]
        distance = -float(aligner.score(left_row["sequence"], right_row["sequence"]))
        similarity = 1.0 - distance / max(len(left_row["sequence"]), len(right_row["sequence"]))
        similarity_matrix[left, right] = similarity_matrix[right, left] = similarity
        pair_rows.append({
            "ID_1": int(left_row["ID"]),
            "ID_2": int(right_row["ID"]),
            "class_1": left_row["true_class"],
            "class_2": right_row["true_class"],
            "total_wrong_1": int(left_row["total_wrong_count"]),
            "total_wrong_2": int(right_row["total_wrong_count"]),
            "consensus_error_1": bool(left_row["total_wrong_count"] == 8),
            "consensus_error_2": bool(right_row["total_wrong_count"] == 8),
            "normalized_edit_similarity": similarity,
            "same_true_class": bool(left_row["y_true"] == right_row["y_true"]),
        })
pairwise = pd.DataFrame(pair_rows)
assert len(pairwise) == 105
assert np.allclose(similarity_matrix, similarity_matrix.T)
assert np.allclose(np.diag(similarity_matrix), 1.0)
assert np.isfinite(similarity_matrix).all()
assert np.all((similarity_matrix >= 0) & (similarity_matrix <= 1))
pairwise.to_csv(PAIRWISE_OUTPUT, index=False)

hard_by_id = hard.set_index("ID")
similarity_lookup = {
    frozenset((int(row.ID_1), int(row.ID_2))): float(row.normalized_edit_similarity)
    for row in pairwise.itertuples(index=False)
}
assignment_rows = []
cluster_rows = []
threshold_rows = []
components_by_threshold = {}

for threshold in THRESHOLDS:
    edges = [
        (int(row.ID_1), int(row.ID_2))
        for row in pairwise.itertuples(index=False)
        if row.normalized_edit_similarity >= threshold
    ]
    components = connected_components(hard["ID"].astype(int).tolist(), edges)
    components_by_threshold[threshold] = components
    for cluster_id, members in enumerate(components, start=1):
        member_frame = hard_by_id.loc[members]
        active_count = int(member_frame["y_true"].eq(1).sum())
        inactive_count = int(member_frame["y_true"].eq(0).sum())
        pattern = class_pattern(active_count, inactive_count)
        internal_similarities = [
            similarity_lookup[frozenset((members[left], members[right]))]
            for left in range(len(members))
            for right in range(left + 1, len(members))
        ]
        for member in members:
            row = hard_by_id.loc[member]
            assignment_rows.append({
                "threshold": threshold,
                "cluster_id": cluster_id,
                "cluster_minimum_member_ID": min(members),
                "ID": member,
                "sequence": row["sequence"],
                "true_class": row["true_class"],
                "total_wrong_count": int(row["total_wrong_count"]),
                "consensus_8_of_8_error": bool(row["total_wrong_count"] == 8),
                "cluster_size": len(members),
                "cluster_class_composition": pattern,
            })
        cluster_rows.append({
            "threshold": threshold,
            "cluster_id": cluster_id,
            "cluster_minimum_member_ID": min(members),
            "cluster_size": len(members),
            "member_IDs": ";".join(map(str, members)),
            "active_count": active_count,
            "inactive_count": inactive_count,
            "class_pattern": pattern,
            "mean_total_wrong_count": member_frame["total_wrong_count"].mean(),
            "max_total_wrong_count": int(member_frame["total_wrong_count"].max()),
            "consensus_8_of_8_count": int(member_frame["total_wrong_count"].eq(8).sum()),
            "mean_within_cluster_similarity": (
                np.mean(internal_similarities) if internal_similarities else np.nan
            ),
            "minimum_within_cluster_similarity": (
                np.min(internal_similarities) if internal_similarities else np.nan
            ),
            "maximum_within_cluster_similarity": (
                np.max(internal_similarities) if internal_similarities else np.nan
            ),
        })
    component_patterns = []
    for members in components:
        member_frame = hard_by_id.loc[members]
        component_patterns.append(class_pattern(
            int(member_frame["y_true"].eq(1).sum()), int(member_frame["y_true"].eq(0).sum())
        ))
    threshold_rows.append({
        "threshold": threshold,
        "hard_cases": n_hard,
        "number_of_clusters": len(components),
        "singleton_clusters": sum(len(members) == 1 for members in components),
        "non_singleton_clusters": sum(len(members) > 1 for members in components),
        "largest_cluster_size": max(map(len, components)),
        "mixed_class_clusters": component_patterns.count("Mixed-class"),
        "inactive_only_clusters": component_patterns.count("Inactive-only"),
        "active_only_clusters": component_patterns.count("Active-only"),
        "pairs_above_threshold": len(edges),
        "proportion_pairs_above_threshold": len(edges) / len(pairwise),
    })

assignments = pd.DataFrame(assignment_rows)
clusters = pd.DataFrame(cluster_rows)
threshold_summary = pd.DataFrame(threshold_rows)
assignments.to_csv(ASSIGNMENT_OUTPUT, index=False)
clusters.to_csv(CLUSTER_OUTPUT, index=False)
threshold_summary.to_csv(THRESHOLD_OUTPUT, index=False)

for threshold in THRESHOLDS:
    threshold_assignments = assignments.loc[assignments["threshold"].eq(threshold)]
    threshold_clusters = clusters.loc[clusters["threshold"].eq(threshold)]
    assert len(threshold_assignments) == n_hard
    assert threshold_assignments["ID"].nunique() == n_hard
    assert threshold_assignments.groupby("cluster_id")["cluster_size"].first().sum() == n_hard
    assert (threshold_clusters["active_count"] + threshold_clusters["inactive_count"]).eq(
        threshold_clusters["cluster_size"]
    ).all()
    assert threshold_clusters["consensus_8_of_8_count"].le(threshold_clusters["cluster_size"]).all()
    minimum_ids = threshold_clusters["cluster_minimum_member_ID"].tolist()
    assert minimum_ids == sorted(minimum_ids)
    assert threshold_clusters["cluster_id"].tolist() == list(range(1, len(threshold_clusters) + 1))

labels = [
    f"{row.ID} {row.true_class[0]} {row.total_wrong_count}/8"
    for row in hard.itertuples(index=False)
]
fig, ax = plt.subplots(figsize=(12.0, 10.4), facecolor="white")
image = ax.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(np.arange(n_hard), labels, rotation=55, ha="right")
ax.set_yticks(np.arange(n_hard), labels)
ax.set_title("Normalized edit similarity among the 15 consensus hard cases")
ax.set_xlabel("Hard cases in Step-59 rank order")
ax.set_ylabel("Hard cases in Step-59 rank order")
for index in range(n_hard):
    for column in range(n_hard):
        value = similarity_matrix[index, column]
        ax.text(
            column, index, f"{value:.2f}", ha="center", va="center", fontsize=6.4,
            color="white" if value < 0.66 else "black",
        )
ax.axhline(4.5, color="#D73027", linewidth=1.6)
ax.axvline(4.5, color="#D73027", linewidth=1.6)
for tick in list(ax.get_xticklabels())[:5] + list(ax.get_yticklabels())[:5]:
    tick.set_color("#B2182B")
    tick.set_fontweight("bold")
colorbar = fig.colorbar(image, ax=ax, pad=0.02)
colorbar.set_label("Normalized edit similarity")
fig.tight_layout()
fig.savefig(HEATMAP_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(HEATMAP_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

network_components = components_by_threshold[NETWORK_THRESHOLD]
network_edges = pairwise.loc[pairwise["normalized_edit_similarity"] >= NETWORK_THRESHOLD]
positions = {}
n_components = len(network_components)
centers = []
for component_index in range(n_components):
    angle = 2 * np.pi * component_index / max(n_components, 1) + np.pi / 2
    centers.append((3.25 * np.cos(angle), 3.25 * np.sin(angle)))
for center, members in zip(centers, network_components):
    if len(members) == 1:
        positions[members[0]] = center
    else:
        radius = min(1.10, 0.52 + 0.10 * len(members))
        for member_index, member in enumerate(members):
            angle = 2 * np.pi * member_index / len(members) + np.pi / 2
            positions[member] = (
                center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)
            )

fig, ax = plt.subplots(figsize=(12.0, 10.2), facecolor="white")
for edge in network_edges.itertuples(index=False):
    x1, y1 = positions[int(edge.ID_1)]
    x2, y2 = positions[int(edge.ID_2)]
    ax.plot(
        [x1, x2], [y1, y2], color="#737373",
        linewidth=1.0 + 4.0 * (edge.normalized_edit_similarity - NETWORK_THRESHOLD)
        / (1.0 - NETWORK_THRESHOLD), alpha=0.62, zorder=1,
    )
for row in hard.itertuples(index=False):
    x, y = positions[int(row.ID)]
    active = row.y_true == 1
    consensus = row.total_wrong_count == 8
    ax.scatter(
        [x], [y], s=820, marker="s" if active else "o",
        color="#377EB8" if active else "#E66101",
        edgecolor="#111111" if consensus else "#666666",
        linewidth=3.5 if consensus else 1.2, zorder=3,
    )
    ax.text(
        x, y, f"{row.ID}\n{row.true_class[0]} {row.total_wrong_count}/8",
        ha="center", va="center", fontsize=8.4,
        color="white", fontweight="bold", zorder=4,
    )
ax.set_title("Hard-case sequence-family network at normalized edit similarity >= 0.80")
ax.set_aspect("equal")
ax.axis("off")
legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#E66101",
           markeredgecolor="#666666", markersize=12, label="Inactive"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#377EB8",
           markeredgecolor="#666666", markersize=12, label="Active"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#BDBDBD",
           markeredgecolor="#111111", markeredgewidth=3, markersize=12,
           label="8/8 consensus failure (thick outline)"),
    Line2D([0, 1], [0, 0], color="#737373", linewidth=2, label="Similarity >= 0.80"),
]
ax.legend(
    handles=legend, frameon=False, loc="upper center",
    bbox_to_anchor=(0.5, -0.015), ncol=2,
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(NETWORK_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(NETWORK_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc_rows = []
for threshold in THRESHOLDS:
    threshold_assignments = assignments.loc[assignments["threshold"].eq(threshold)]
    threshold_clusters = clusters.loc[clusters["threshold"].eq(threshold)]
    qc_rows.append({
        "threshold": threshold,
        "hard_case_peptides": n_hard,
        "unique_ids": hard["ID"].nunique(),
        "pairwise_rows": len(pairwise),
        "pairwise_similarities_finite": bool(np.isfinite(pairwise["normalized_edit_similarity"]).all()),
        "similarity_range_within_0_1": bool(pairwise["normalized_edit_similarity"].between(0, 1).all()),
        "matrix_symmetric": bool(np.allclose(similarity_matrix, similarity_matrix.T)),
        "diagonal_self_similarity_one": bool(np.allclose(np.diag(similarity_matrix), 1.0)),
        "assignment_rows": len(threshold_assignments),
        "assignment_unique_ids": threshold_assignments["ID"].nunique(),
        "sum_cluster_sizes": int(threshold_clusters["cluster_size"].sum()),
        "cluster_ids_deterministic": bool(
            threshold_clusters["cluster_minimum_member_ID"].tolist()
            == sorted(threshold_clusters["cluster_minimum_member_ID"].tolist())
            and threshold_clusters["cluster_id"].tolist()
            == list(range(1, len(threshold_clusters) + 1))
        ),
        "class_counts_match_cluster_sizes": bool(
            (threshold_clusters["active_count"] + threshold_clusters["inactive_count"])
            .eq(threshold_clusters["cluster_size"]).all()
        ),
        "consensus_counts_valid": bool(
            threshold_clusters["consensus_8_of_8_count"].le(threshold_clusters["cluster_size"]).all()
        ),
        "models_trained": False,
        "threshold_optimized": False,
        "labels_changed": False,
    })
qc = pd.DataFrame(qc_rows)
qc.to_csv(QC_OUTPUT, index=False)

print("\nThreshold summary:")
print(threshold_summary.to_string(index=False))
print("\nClusters at threshold 0.80:")
print(
    clusters.loc[clusters["threshold"].eq(NETWORK_THRESHOLD), [
        "cluster_id", "cluster_size", "member_IDs", "active_count", "inactive_count",
        "class_pattern", "consensus_8_of_8_count", "minimum_within_cluster_similarity",
    ]].round(6).to_string(index=False)
)
print("\nOutput checks:")
for path in (
    PAIRWISE_OUTPUT, ASSIGNMENT_OUTPUT, CLUSTER_OUTPUT, THRESHOLD_OUTPUT, QC_OUTPUT,
    HEATMAP_PNG, HEATMAP_PDF, NETWORK_PNG, NETWORK_PDF,
):
    print(path.name, ":", path.exists())
print("\nSTEP 62 COMPLETED SUCCESSFULLY")
print("=" * 110)

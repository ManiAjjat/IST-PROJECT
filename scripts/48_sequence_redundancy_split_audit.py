from pathlib import Path

from Bio.Align import PairwiseAligner
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
METADATA_FILE = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
TEST_DEV_OUTPUT = RESULTS_DIR / "step66_test_to_development_sequence_similarity.csv"
ASSIGNMENT_OUTPUT = RESULTS_DIR / "step66_sequence_family_assignments.csv"
FAMILY_SUMMARY_OUTPUT = RESULTS_DIR / "step66_sequence_family_summary.csv"
THRESHOLD_SUMMARY_OUTPUT = RESULTS_DIR / "step66_similarity_threshold_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step66_sequence_redundancy_qc.csv"
SIMILARITY_FIGURE_PNG = FIGURES_DIR / "Step66_Test_to_Development_Similarity.png"
SIMILARITY_FIGURE_PDF = FIGURES_DIR / "Step66_Test_to_Development_Similarity.pdf"
REDUNDANCY_FIGURE_PNG = FIGURES_DIR / "Step66_Sequence_Family_Redundancy.png"
REDUNDANCY_FIGURE_PDF = FIGURES_DIR / "Step66_Sequence_Family_Redundancy.pdf"
THRESHOLDS = (0.80, 0.90, 0.95)


def connected_components(adjacency):
    n = adjacency.shape[0]
    unseen = set(range(n))
    components = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = set(np.flatnonzero(adjacency[node])) & unseen
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        components.append(sorted(component))
    components.sort(key=lambda values: int(ids[values].min()))
    return components


print("=" * 112)
print("STEP 66 - SEQUENCE REDUNDANCY AND DEVELOPMENT-TEST HOMOLOGY AUDIT")
print("=" * 112)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(METADATA_FILE).sort_values("ID").reset_index(drop=True)
assert len(metadata) == 901 and metadata["ID"].is_unique
assert set(metadata["split"]) == {"development", "test"}
assert int(metadata["split"].eq("development").sum()) == 720
assert int(metadata["split"].eq("test").sum()) == 181
assert int(metadata.loc[metadata["split"].eq("development"), "label"].eq(1).sum()) == 79
assert int(metadata.loc[metadata["split"].eq("test"), "label"].eq(1).sum()) == 20
assert set(metadata["binary_class"]) == {"Active", "Inactive"}
ids = metadata["ID"].to_numpy(int)
sequences = metadata["sequence"].astype(str).to_numpy()
labels = metadata["label"].to_numpy(int)
splits = metadata["split"].to_numpy()

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 0.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -1.0
aligner.extend_gap_score = -1.0
n = len(metadata)
similarity = np.eye(n, dtype=np.float64)
total_pairs = n * (n - 1) // 2
completed = 0
for i in range(n - 1):
    first = sequences[i]
    for j in range(i + 1, n):
        second = sequences[j]
        distance = -float(aligner.score(first, second))
        value = 1.0 - distance / max(len(first), len(second))
        similarity[i, j] = value
        similarity[j, i] = value
        completed += 1
    if (i + 1) % 100 == 0:
        print(f"Pairwise progress: {completed:,}/{total_pairs:,}")
assert completed == total_pairs == 405450
assert np.isfinite(similarity).all()
assert np.all((similarity >= 0) & (similarity <= 1))
assert np.allclose(similarity, similarity.T)
assert np.all(np.diag(similarity) == 1)

development_indices = np.flatnonzero(splits == "development")
test_indices = np.flatnonzero(splits == "test")
development_ids = ids[development_indices]
test_rows = []
for index in test_indices:
    values = similarity[index, development_indices].astype(np.float64)
    maximum = values.max()
    tied_local = np.flatnonzero(np.isclose(values, maximum, rtol=0, atol=1e-12))
    chosen_local = tied_local[np.argmin(development_ids[tied_local])]
    chosen = development_indices[chosen_local]
    test_rows.append({
        "test_ID": int(ids[index]), "test_sequence": sequences[index],
        "test_class": metadata.iloc[index]["binary_class"], "test_label": int(labels[index]),
        "nearest_development_ID": int(ids[chosen]),
        "nearest_development_sequence": sequences[chosen],
        "nearest_development_class": metadata.iloc[chosen]["binary_class"],
        "nearest_development_label": int(labels[chosen]),
        "same_true_class": bool(labels[index] == labels[chosen]),
        "class_relation": "same" if labels[index] == labels[chosen] else "opposite",
        "nearest_development_similarity": float(maximum),
        "nearest_tie_count": len(tied_local),
        "at_least_0_80": bool(maximum >= 0.80),
        "at_least_0_90": bool(maximum >= 0.90),
        "at_least_0_95": bool(maximum >= 0.95),
        "exact_1_00": bool(np.isclose(maximum, 1.0, rtol=0, atol=1e-12)),
    })
test_dev = pd.DataFrame(test_rows).sort_values("test_ID").reset_index(drop=True)
assert len(test_dev) == 181 and test_dev["test_ID"].is_unique
test_dev.to_csv(TEST_DEV_OUTPUT, index=False)

assignment_rows = []
family_rows = []
threshold_rows = []
for threshold in THRESHOLDS:
    adjacency = similarity >= threshold
    np.fill_diagonal(adjacency, False)
    components = connected_components(adjacency)
    edge_count = int(np.triu(adjacency, 1).sum())
    for family_number, component in enumerate(components, 1):
        member_ids = sorted(map(int, ids[component]))
        dev_count = int(np.sum(splits[component] == "development"))
        test_count = int(np.sum(splits[component] == "test"))
        active_count = int(np.sum(labels[component] == 1))
        inactive_count = len(component) - active_count
        split_composition = (
            "cross_split" if dev_count and test_count else
            "development_only" if dev_count else "test_only"
        )
        label_composition = (
            "mixed_label" if active_count and inactive_count else
            "Active_only" if active_count else "Inactive_only"
        )
        family_rows.append({
            "threshold": threshold, "family_id": family_number,
            "family_minimum_member_ID": min(member_ids), "family_size": len(component),
            "development_count": dev_count, "test_count": test_count,
            "active_count": active_count, "inactive_count": inactive_count,
            "split_composition": split_composition, "label_composition": label_composition,
            "cross_split_family": bool(dev_count and test_count),
            "mixed_label_family": bool(active_count and inactive_count),
            "member_IDs": ";".join(map(str, member_ids)),
        })
        for index in component:
            assignment_rows.append({
                "threshold": threshold, "family_id": family_number,
                "family_minimum_member_ID": min(member_ids), "family_size": len(component),
                "ID": int(ids[index]), "sequence": sequences[index],
                "split": splits[index], "label": int(labels[index]),
                "binary_class": metadata.iloc[index]["binary_class"],
                "split_composition": split_composition, "label_composition": label_composition,
                "cross_split_family": bool(dev_count and test_count),
                "mixed_label_family": bool(active_count and inactive_count),
            })
    fam = pd.DataFrame([row for row in family_rows if row["threshold"] == threshold])
    threshold_rows.append({
        "threshold": threshold, "sequence_family_count": len(components),
        "singleton_families": int(fam["family_size"].eq(1).sum()),
        "non_singleton_families": int(fam["family_size"].gt(1).sum()),
        "largest_family_size": int(fam["family_size"].max()),
        "development_only_families": int(fam["split_composition"].eq("development_only").sum()),
        "test_only_families": int(fam["split_composition"].eq("test_only").sum()),
        "cross_split_families": int(fam["cross_split_family"].sum()),
        "mixed_label_families": int(fam["mixed_label_family"].sum()),
        "same_label_families": int((~fam["mixed_label_family"]).sum()),
        "graph_edges": edge_count,
        "test_peptides_in_cross_split_families": int(fam.loc[fam["cross_split_family"], "test_count"].sum()),
        "development_peptides_in_cross_split_families": int(fam.loc[fam["cross_split_family"], "development_count"].sum()),
    })

assignments = pd.DataFrame(assignment_rows).sort_values(["threshold", "family_id", "ID"]).reset_index(drop=True)
family_summary = pd.DataFrame(family_rows).sort_values(["threshold", "family_id"]).reset_index(drop=True)
threshold_summary = pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)
assert len(assignments) == 901 * 3
assert assignments.groupby("threshold")["ID"].nunique().eq(901).all()
assert family_summary.groupby("threshold")["family_size"].sum().eq(901).all()
assert (threshold_summary["development_only_families"] + threshold_summary["test_only_families"]
        + threshold_summary["cross_split_families"]).eq(threshold_summary["sequence_family_count"]).all()
assert (threshold_summary["mixed_label_families"] + threshold_summary["same_label_families"]).eq(
    threshold_summary["sequence_family_count"]
).all()
assignments.to_csv(ASSIGNMENT_OUTPUT, index=False)
family_summary.to_csv(FAMILY_SUMMARY_OUTPUT, index=False)
threshold_summary.to_csv(THRESHOLD_SUMMARY_OUTPUT, index=False)

values = test_dev["nearest_development_similarity"].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), facecolor="white")
axes[0].hist(values, bins=np.linspace(0, 1, 26), color="#377EB8", edgecolor="white")
for threshold, color in zip(THRESHOLDS, ("#FDAE61", "#F46D43", "#D73027")):
    axes[0].axvline(threshold, color=color, linestyle="--", linewidth=1.5, label=f"{threshold:.2f}")
axes[0].set_xlabel("Nearest development normalized edit similarity")
axes[0].set_ylabel("Locked-test peptide count")
axes[0].set_title("A  Distribution")
axes[0].legend(title="Threshold", frameon=False)
sorted_values = np.sort(values)
axes[1].plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color="#4D4D4D", linewidth=2)
for threshold, color in zip(THRESHOLDS, ("#FDAE61", "#F46D43", "#D73027")):
    axes[1].axvline(threshold, color=color, linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Nearest development normalized edit similarity")
axes[1].set_ylabel("Cumulative fraction of locked-test peptides")
axes[1].set_title("B  Empirical cumulative distribution")
for ax in axes:
    ax.grid(color="#E0E0E0", linewidth=0.6); ax.set_axisbelow(True)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
fig.suptitle("Locked-test similarity to the closest development peptide")
fig.tight_layout()
fig.savefig(SIMILARITY_FIGURE_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(SIMILARITY_FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

x = np.arange(len(THRESHOLDS)); width = 0.25
fig, ax = plt.subplots(figsize=(11.3, 6.5), facecolor="white")
for offset, column, label, color in (
    (-width, "development_only_families", "Development only", "#4575B4"),
    (0, "test_only_families", "Test only", "#FDAE61"),
    (width, "cross_split_families", "Cross split", "#D73027"),
):
    bars = ax.bar(x + offset, threshold_summary[column], width, label=label, color=color)
    ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xticks(x, [f">= {value:.2f}" for value in THRESHOLDS])
ax.set_ylabel("Sequence-family count")
ax.set_xlabel("Normalized edit-similarity edge threshold")
ax.set_title("Sequence-family redundancy across the development/test split")
ax.legend(frameon=False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.6); ax.set_axisbelow(True)
for spine in ("top", "right"): ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(REDUNDANCY_FIGURE_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(REDUNDANCY_FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

exact_matches = int(test_dev["exact_1_00"].sum())
qc = pd.DataFrame([{
    "peptides": n, "development_peptides": len(development_indices), "test_peptides": len(test_indices),
    "unique_pairwise_comparisons": total_pairs, "similarity_matrix_symmetric": bool(np.allclose(similarity, similarity.T)),
    "diagonal_exactly_one": bool(np.all(np.diag(similarity) == 1)),
    "all_similarities_finite": bool(np.isfinite(similarity).all()),
    "all_similarities_within_0_1": bool(np.all((similarity >= 0) & (similarity <= 1))),
    "exact_test_development_matches": exact_matches,
    "test_rows": len(test_dev), "assignment_rows": len(assignments),
    "thresholds_exact": ";".join(f"{value:.2f}" for value in THRESHOLDS),
    "all_assignment_thresholds_cover_901": bool(assignments.groupby("threshold")["ID"].nunique().eq(901).all()),
    "family_sizes_cover_901": bool(family_summary.groupby("threshold")["family_size"].sum().eq(901).all()),
    "nearest_development_ids_valid": bool(set(test_dev["nearest_development_ID"]).issubset(set(development_ids))),
    "test_ids_disjoint_from_development_candidates": bool(set(ids[test_indices]).isdisjoint(development_ids)),
    "models_trained": False, "split_changed": False, "labels_changed": False,
    "thresholds_optimized": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\n" + "=" * 112)
print("STEP 66 SUMMARY")
print("=" * 112)
print("Peptides:", n)
print("Development/Test: 720/181")
print("Pairwise comparisons:", total_pairs)
print("Exact test-development sequence matches:", exact_matches)
for threshold in THRESHOLDS:
    count = int((test_dev["nearest_development_similarity"] >= threshold).sum())
    print(f"Test peptides with nearest development similarity >= {threshold:.2f}: {count}/181")
print("\nThreshold family summary:")
print(threshold_summary.to_string(index=False))
print("\nSTEP 66 COMPLETED SUCCESSFULLY")
print("=" * 112)

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

RANKING_FILE = RESULTS_DIR / "step59_consensus_hard_case_ranking.csv"
EMBEDDING_FILE = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
METADATA_FILE = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"

FULL_OUTPUT = RESULTS_DIR / "step61_sequence_nearest_neighbor_analysis.csv"
HARD_OUTPUT = RESULTS_DIR / "step61_hard_case_nearest_neighbors.csv"
GROUP_SUMMARY_OUTPUT = RESULTS_DIR / "step61_nearest_neighbor_group_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step61_sequence_similarity_qc.csv"
PROXIMITY_FIGURE_PNG = FIGURES_DIR / "Step61_Hard_Case_Opposite_Class_Proximity.png"
PROXIMITY_FIGURE_PDF = FIGURES_DIR / "Step61_Hard_Case_Opposite_Class_Proximity.pdf"
MAP_FIGURE_PNG = FIGURES_DIR / "Step61_Hard_Case_Nearest_Neighbor_Map.png"
MAP_FIGURE_PDF = FIGURES_DIR / "Step61_Hard_Case_Nearest_Neighbor_Map.pdf"

HARD_LABEL = "Hard cases (>=3/8 wrong)"
REFERENCE_LABEL = "Generally well classified"


def choose_neighbor(similarities, candidate_indices, ids):
    """Maximum similarity, with the smallest ID as the deterministic tie-breaker."""
    candidate_similarities = similarities[candidate_indices]
    best_similarity = candidate_similarities.max()
    tied = candidate_indices[np.isclose(candidate_similarities, best_similarity, rtol=0, atol=1e-12)]
    return int(tied[np.argmin(ids[tied])]), float(best_similarity), int(len(tied))


def summarize_group(frame, group_name, metrics):
    rows = []
    for metric in metrics:
        values = frame[metric].astype(float)
        rows.append({
            "analysis_group": group_name,
            "metric": metric,
            "n": len(values),
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "median": values.median(),
            "minimum": values.min(),
            "maximum": values.max(),
            "positive_margin_count": int((values > 0).sum()) if metric.endswith("margin") else np.nan,
            "zero_margin_count": int(np.isclose(values, 0, atol=1e-12).sum())
            if metric.endswith("margin") else np.nan,
            "negative_margin_count": int((values < 0).sum()) if metric.endswith("margin") else np.nan,
        })
    return rows


print("=" * 110)
print("STEP 61 - SEQUENCE SIMILARITY AND NEAREST-NEIGHBOR ANALYSIS")
print("=" * 110)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ranking = pd.read_csv(RANKING_FILE)
metadata = pd.read_csv(METADATA_FILE)
embeddings = np.load(EMBEDDING_FILE)
assert len(ranking) == 181 and ranking["ID"].is_unique
assert embeddings.shape == (901, 1280) and embeddings.dtype == np.float32
assert len(metadata) == 901 and metadata["embedding_row"].tolist() == list(range(901))

test_metadata = metadata.loc[metadata["split"] == "test"].copy()
aligned = ranking[[
    "rank", "ID", "sequence", "true_class", "y_true", "total_wrong_count", "difficulty_category"
]].merge(
    test_metadata[["embedding_row", "ID", "sequence", "label", "binary_class"]],
    on=["ID", "sequence"], how="left", validate="one_to_one",
)
assert len(aligned) == 181 and aligned["embedding_row"].notna().all()
assert aligned["y_true"].eq(aligned["label"]).all()
assert aligned["true_class"].eq(aligned["binary_class"]).all()

sequences = aligned["sequence"].tolist()
labels = aligned["y_true"].to_numpy(int)
ids = aligned["ID"].to_numpy(int)
n_peptides = len(aligned)

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 0.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -1.0
aligner.extend_gap_score = -1.0

edit_similarity = np.eye(n_peptides, dtype=np.float64)
pair_count = 0
for left in range(n_peptides):
    for right in range(left + 1, n_peptides):
        edit_distance = -float(aligner.score(sequences[left], sequences[right]))
        similarity = 1.0 - edit_distance / max(len(sequences[left]), len(sequences[right]))
        edit_similarity[left, right] = edit_similarity[right, left] = similarity
        pair_count += 1
assert pair_count == n_peptides * (n_peptides - 1) // 2 == 16290
assert np.allclose(edit_similarity, edit_similarity.T)
assert np.all((edit_similarity >= -1e-12) & (edit_similarity <= 1 + 1e-12))

test_embedding_rows = aligned["embedding_row"].to_numpy(int)
test_embeddings = embeddings[test_embedding_rows].astype(np.float64)
norms = np.linalg.norm(test_embeddings, axis=1)
assert np.all(norms > 0)
normalized_embeddings = test_embeddings / norms[:, None]
esm_similarity = normalized_embeddings @ normalized_embeddings.T
esm_similarity = np.clip(esm_similarity, -1.0, 1.0)
assert np.allclose(esm_similarity, esm_similarity.T, atol=1e-12)

rows = []
all_indices = np.arange(n_peptides)
for index, base in aligned.iterrows():
    same_candidates = all_indices[(labels == labels[index]) & (all_indices != index)]
    opposite_candidates = all_indices[labels != labels[index]]
    assert len(same_candidates) > 0 and len(opposite_candidates) > 0

    seq_same, seq_same_similarity, seq_same_ties = choose_neighbor(
        edit_similarity[index], same_candidates, ids
    )
    seq_opposite, seq_opposite_similarity, seq_opposite_ties = choose_neighbor(
        edit_similarity[index], opposite_candidates, ids
    )
    esm_same, esm_same_similarity, esm_same_ties = choose_neighbor(
        esm_similarity[index], same_candidates, ids
    )
    esm_opposite, esm_opposite_similarity, esm_opposite_ties = choose_neighbor(
        esm_similarity[index], opposite_candidates, ids
    )

    seq_margin = seq_opposite_similarity - seq_same_similarity
    esm_margin = esm_opposite_similarity - esm_same_similarity
    if seq_margin > 1e-12 and esm_margin > 1e-12:
        pattern = "Both favor opposite class"
    elif seq_margin > 1e-12 and esm_margin < -1e-12:
        pattern = "Sequence opposite; ESM same"
    elif seq_margin < -1e-12 and esm_margin > 1e-12:
        pattern = "Sequence same; ESM opposite"
    elif seq_margin < -1e-12 and esm_margin < -1e-12:
        pattern = "Both favor same class"
    else:
        pattern = "At least one tied neighborhood"

    rows.append({
        "rank": int(base["rank"]),
        "ID": int(base["ID"]),
        "sequence": base["sequence"],
        "true_class": base["true_class"],
        "y_true": int(base["y_true"]),
        "total_wrong_count": int(base["total_wrong_count"]),
        "difficulty_category": base["difficulty_category"],
        "analysis_group": HARD_LABEL if base["total_wrong_count"] >= 3 else REFERENCE_LABEL,
        "same_class_candidate_count": len(same_candidates),
        "opposite_class_candidate_count": len(opposite_candidates),
        "sequence_nearest_same_ID": ids[seq_same],
        "sequence_nearest_same_sequence": sequences[seq_same],
        "sequence_nearest_same_similarity": seq_same_similarity,
        "sequence_nearest_same_tie_count": seq_same_ties,
        "sequence_nearest_opposite_ID": ids[seq_opposite],
        "sequence_nearest_opposite_sequence": sequences[seq_opposite],
        "sequence_nearest_opposite_similarity": seq_opposite_similarity,
        "sequence_nearest_opposite_tie_count": seq_opposite_ties,
        "sequence_opposite_class_proximity_margin": seq_margin,
        "esm2_nearest_same_ID": ids[esm_same],
        "esm2_nearest_same_sequence": sequences[esm_same],
        "esm2_nearest_same_similarity": esm_same_similarity,
        "esm2_nearest_same_tie_count": esm_same_ties,
        "esm2_nearest_opposite_ID": ids[esm_opposite],
        "esm2_nearest_opposite_sequence": sequences[esm_opposite],
        "esm2_nearest_opposite_similarity": esm_opposite_similarity,
        "esm2_nearest_opposite_tie_count": esm_opposite_ties,
        "esm2_opposite_class_proximity_margin": esm_margin,
        "neighborhood_pattern": pattern,
    })

analysis = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
hard_cases = analysis.loc[analysis["total_wrong_count"] >= 3].copy()
reference = analysis.loc[analysis["difficulty_category"] == "Generally well classified"].copy()
assert len(hard_cases) == 15 and len(reference) == 166
assert set(hard_cases["ID"]) | set(reference["ID"]) == set(analysis["ID"])
analysis.to_csv(FULL_OUTPUT, index=False)
hard_cases.to_csv(HARD_OUTPUT, index=False)

summary_metrics = [
    "sequence_nearest_same_similarity", "sequence_nearest_opposite_similarity",
    "sequence_opposite_class_proximity_margin", "esm2_nearest_same_similarity",
    "esm2_nearest_opposite_similarity", "esm2_opposite_class_proximity_margin",
]
group_summary = pd.DataFrame(
    summarize_group(hard_cases, HARD_LABEL, summary_metrics)
    + summarize_group(reference, REFERENCE_LABEL, summary_metrics)
)
group_summary.to_csv(GROUP_SUMMARY_OUTPUT, index=False)

labels_for_plot = [f"{row.ID} ({row.true_class[0]})" for row in hard_cases.itertuples(index=False)]
y = np.arange(len(hard_cases))
height = 0.37
fig, ax = plt.subplots(figsize=(11.4, 8.2), facecolor="white")
ax.barh(
    y - height / 2, hard_cases["sequence_opposite_class_proximity_margin"], height,
    color="#377EB8", label="Normalized edit similarity",
)
ax.barh(
    y + height / 2, hard_cases["esm2_opposite_class_proximity_margin"], height,
    color="#E41A1C", label="ESM-2 cosine similarity",
)
ax.axvline(0, color="#222222", linewidth=1.1)
ax.set_yticks(y, labels_for_plot)
ax.invert_yaxis()
ax.set_xlabel("Opposite-class minus same-class nearest-neighbor similarity")
ax.set_ylabel("Step-59 rank: peptide ID (true class)")
ax.set_title("Opposite-class proximity of consensus hard cases")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(PROXIMITY_FIGURE_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(PROXIMITY_FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

map_columns = [
    "sequence_nearest_same_similarity", "sequence_nearest_opposite_similarity",
    "esm2_nearest_same_similarity", "esm2_nearest_opposite_similarity",
]
neighbor_id_columns = [
    "sequence_nearest_same_ID", "sequence_nearest_opposite_ID",
    "esm2_nearest_same_ID", "esm2_nearest_opposite_ID",
]
map_labels = ["Sequence\nsame", "Sequence\nopposite", "ESM-2\nsame", "ESM-2\nopposite"]
map_values = hard_cases[map_columns].to_numpy(float)
fig, ax = plt.subplots(figsize=(9.4, 9.0), facecolor="white")
image = ax.imshow(map_values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(4), map_labels)
ax.set_yticks(np.arange(len(hard_cases)), labels_for_plot)
ax.set_xlabel("Nearest-neighbor type")
ax.set_ylabel("Step-59 rank: peptide ID (true class)")
ax.set_title("Hard-case nearest-neighbor similarities and neighbor IDs")
for row_index in range(len(hard_cases)):
    for column_index in range(4):
        value = map_values[row_index, column_index]
        neighbor_id = int(hard_cases.iloc[row_index][neighbor_id_columns[column_index]])
        ax.text(
            column_index, row_index, f"{value:.3f}\nID {neighbor_id}",
            ha="center", va="center", fontsize=8.0,
            color="white" if value < 0.66 else "black",
        )
colorbar = fig.colorbar(image, ax=ax, pad=0.02)
colorbar.set_label("Similarity")
fig.tight_layout()
fig.savefig(MAP_FIGURE_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(MAP_FIGURE_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

consensus_errors = hard_cases.loc[hard_cases["total_wrong_count"] == 8]
hard_count = len(hard_cases)
hard_sequence_closer_opposite = int(
    (hard_cases["sequence_opposite_class_proximity_margin"] > 0).sum()
)
hard_esm_closer_opposite = int(
    (hard_cases["esm2_opposite_class_proximity_margin"] > 0).sum()
)
qc = pd.DataFrame([{
    "locked_test_peptides": n_peptides,
    "active_test_peptides": int((labels == 1).sum()),
    "inactive_test_peptides": int((labels == 0).sum()),
    "unordered_sequence_pairs": pair_count,
    "embedding_dimensions": test_embeddings.shape[1],
    "hard_cases": hard_count,
    "consensus_8_of_8_errors": len(consensus_errors),
    "hard_sequence_closer_opposite": hard_sequence_closer_opposite,
    "hard_esm2_closer_opposite": hard_esm_closer_opposite,
    "self_neighbors_found": int(
        (analysis["ID"] == analysis["sequence_nearest_same_ID"]).sum()
        + (analysis["ID"] == analysis["esm2_nearest_same_ID"]).sum()
    ),
    "sequence_similarity_symmetric": bool(np.allclose(edit_similarity, edit_similarity.T)),
    "esm2_similarity_symmetric": bool(np.allclose(esm_similarity, esm_similarity.T, atol=1e-12)),
    "sequence_similarity_within_0_1": bool(
        np.all((edit_similarity >= -1e-12) & (edit_similarity <= 1 + 1e-12))
    ),
    "esm2_similarity_finite": bool(np.isfinite(esm_similarity).all()),
    "all_neighbor_ids_valid": bool(
        analysis[neighbor_id_columns].isin(set(ids)).all().all()
    ),
    "all_neighbor_classes_correct": True,
    "groups_cover_all_test_peptides": bool(len(hard_cases) + len(reference) == n_peptides),
    "models_trained": False,
    "labels_changed": False,
    "thresholds_changed": False,
    "model_selection_performed": False,
}])

id_to_label = dict(zip(ids, labels))
for row in analysis.itertuples(index=False):
    assert id_to_label[row.sequence_nearest_same_ID] == row.y_true
    assert id_to_label[row.esm2_nearest_same_ID] == row.y_true
    assert id_to_label[row.sequence_nearest_opposite_ID] != row.y_true
    assert id_to_label[row.esm2_nearest_opposite_ID] != row.y_true
assert qc.at[0, "self_neighbors_found"] == 0
qc.to_csv(QC_OUTPUT, index=False)

print("\nFive consensus 8/8 errors:")
print(
    consensus_errors[[
        "ID", "sequence_nearest_same_ID", "sequence_nearest_same_similarity",
        "sequence_nearest_opposite_ID", "sequence_nearest_opposite_similarity",
        "sequence_opposite_class_proximity_margin", "esm2_nearest_same_ID",
        "esm2_nearest_same_similarity", "esm2_nearest_opposite_ID",
        "esm2_nearest_opposite_similarity", "esm2_opposite_class_proximity_margin",
        "neighborhood_pattern",
    ]].round(6).to_string(index=False)
)

print("\n61S. Output checks:")
outputs = [
    FULL_OUTPUT, HARD_OUTPUT, GROUP_SUMMARY_OUTPUT, QC_OUTPUT,
    PROXIMITY_FIGURE_PNG, PROXIMITY_FIGURE_PDF, MAP_FIGURE_PNG, MAP_FIGURE_PDF,
]
for path in outputs:
    print(path.name, ":", path.exists())

print("\n" + "=" * 110)
print("STEP 61 SUMMARY")
print("=" * 110)
print("Locked-test peptides:", n_peptides)
print("Hard cases:", hard_count)
print("Consensus 8/8 errors:", len(consensus_errors))
print("Hard cases closer to opposite class by sequence:", hard_sequence_closer_opposite, "/", hard_count)
print("Hard cases closer to opposite class by ESM-2:", hard_esm_closer_opposite, "/", hard_count)
print("\nFull analysis:")
print(FULL_OUTPUT)
print("\nHard-case analysis:")
print(HARD_OUTPUT)
print("\nGroup summary:")
print(GROUP_SUMMARY_OUTPUT)
print("\nQC:")
print(QC_OUTPUT)
print("\nProximity figure:")
print(PROXIMITY_FIGURE_PNG)
print("\nNearest-neighbor map:")
print(MAP_FIGURE_PNG)
print("\nSTEP 61 COMPLETED SUCCESSFULLY")
print("=" * 110)

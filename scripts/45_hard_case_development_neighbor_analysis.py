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

MAIN_OUTPUT = RESULTS_DIR / "step63_hard_case_development_neighbors.csv"
TOP_NEIGHBOR_OUTPUT = RESULTS_DIR / "step63_hard_case_top_development_neighbors.csv"
GROUP_SUMMARY_OUTPUT = RESULTS_DIR / "step63_development_neighbor_group_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step63_development_neighbor_qc.csv"
PROXIMITY_PNG = FIGURES_DIR / "Step63_Hard_Case_Development_Proximity.png"
PROXIMITY_PDF = FIGURES_DIR / "Step63_Hard_Case_Development_Proximity.pdf"
MAP_PNG = FIGURES_DIR / "Step63_Hard_Case_Development_Neighbor_Map.png"
MAP_PDF = FIGURES_DIR / "Step63_Hard_Case_Development_Neighbor_Map.pdf"


def best_index(similarities, candidates, candidate_ids):
    values = similarities[candidates]
    maximum = values.max()
    tied = candidates[np.isclose(values, maximum, rtol=0, atol=1e-12)]
    return int(tied[np.argmin(candidate_ids[tied])]), float(maximum), len(tied)


def top_indices(similarities, candidate_ids, count=5):
    return np.lexsort((candidate_ids, -similarities))[:count]


def summarize(frame, group, metrics):
    rows = []
    for metric in metrics:
        values = frame[metric].astype(float)
        rows.append({
            "analysis_group": group,
            "metric": metric,
            "n": len(values),
            "mean": values.mean(),
            "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
            "median": values.median(),
            "minimum": values.min(),
            "maximum": values.max(),
            "positive_count": int((values > 0).sum()) if metric.endswith("margin") else np.nan,
            "zero_count": int(np.isclose(values, 0, rtol=0, atol=1e-12).sum())
            if metric.endswith("margin") else np.nan,
            "negative_count": int((values < 0).sum()) if metric.endswith("margin") else np.nan,
        })
    return rows


print("=" * 112)
print("STEP 63 - HARD-CASE DEVELOPMENT-SET NEAREST-NEIGHBOR ANALYSIS")
print("=" * 112)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ranking = pd.read_csv(RANKING_FILE)
metadata = pd.read_csv(METADATA_FILE)
embeddings = np.load(EMBEDDING_FILE)
assert len(ranking) == 181 and ranking["ID"].is_unique
assert embeddings.shape == (901, 1280) and embeddings.dtype == np.float32
assert len(metadata) == 901 and metadata["embedding_row"].tolist() == list(range(901))

hard = ranking.loc[ranking["total_wrong_count"] >= 3].sort_values("rank").reset_index(drop=True)
development = metadata.loc[metadata["split"] == "development"].sort_values("ID").reset_index(drop=True)
test_metadata = metadata.loc[metadata["split"] == "test"]
assert len(hard) == 15 and hard["ID"].is_unique
assert len(development) == 720 and development["ID"].is_unique
assert int(development["label"].eq(1).sum()) == 79
assert int(development["label"].eq(0).sum()) == 641
assert set(hard["ID"]).isdisjoint(development["ID"])
hard_alignment = hard[["ID", "sequence", "y_true", "true_class"]].merge(
    test_metadata[["ID", "sequence", "label", "binary_class"]],
    on=["ID", "sequence"], how="inner", validate="one_to_one",
)
assert len(hard_alignment) == 15
assert hard_alignment["y_true"].eq(hard_alignment["label"]).all()
assert hard_alignment["true_class"].eq(hard_alignment["binary_class"]).all()

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 0.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -1.0
aligner.extend_gap_score = -1.0

edit_similarity = np.empty((len(hard), len(development)), dtype=np.float64)
comparison_count = 0
for hard_index, hard_sequence in enumerate(hard["sequence"]):
    for development_index, development_sequence in enumerate(development["sequence"]):
        distance = -float(aligner.score(hard_sequence, development_sequence))
        edit_similarity[hard_index, development_index] = (
            1.0 - distance / max(len(hard_sequence), len(development_sequence))
        )
        comparison_count += 1
assert comparison_count == 15 * 720 == 10800
assert np.isfinite(edit_similarity).all()
assert np.all((edit_similarity >= 0) & (edit_similarity <= 1))

hard_embedding_rows = test_metadata.set_index("ID").loc[hard["ID"], "embedding_row"].to_numpy(int)
development_embedding_rows = development["embedding_row"].to_numpy(int)
hard_embeddings = embeddings[hard_embedding_rows].astype(np.float64)
development_embeddings = embeddings[development_embedding_rows].astype(np.float64)
hard_embeddings /= np.linalg.norm(hard_embeddings, axis=1, keepdims=True)
development_embeddings /= np.linalg.norm(development_embeddings, axis=1, keepdims=True)
esm_similarity = np.clip(hard_embeddings @ development_embeddings.T, -1.0, 1.0)
assert np.isfinite(esm_similarity).all()

development_ids = development["ID"].to_numpy(int)
development_labels = development["label"].to_numpy(int)
main_rows = []
top_rows = []

for hard_index, query in hard.iterrows():
    query_label = int(query["y_true"])
    same_candidates = np.flatnonzero(development_labels == query_label)
    opposite_candidates = np.flatnonzero(development_labels != query_label)
    seq_same, seq_same_value, seq_same_ties = best_index(
        edit_similarity[hard_index], same_candidates, development_ids
    )
    seq_opposite, seq_opposite_value, seq_opposite_ties = best_index(
        edit_similarity[hard_index], opposite_candidates, development_ids
    )
    esm_same, esm_same_value, esm_same_ties = best_index(
        esm_similarity[hard_index], same_candidates, development_ids
    )
    esm_opposite, esm_opposite_value, esm_opposite_ties = best_index(
        esm_similarity[hard_index], opposite_candidates, development_ids
    )
    exact_indices = np.flatnonzero(np.isclose(edit_similarity[hard_index], 1.0, rtol=0, atol=1e-12))
    exact_same = [int(development_ids[index]) for index in exact_indices if development_labels[index] == query_label]
    exact_opposite = [int(development_ids[index]) for index in exact_indices if development_labels[index] != query_label]

    sequence_top = top_indices(edit_similarity[hard_index], development_ids, 5)
    esm_top = top_indices(esm_similarity[hard_index], development_ids, 5)
    for similarity_type, similarities, indices in (
        ("normalized_edit_similarity", edit_similarity[hard_index], sequence_top),
        ("esm2_cosine_similarity", esm_similarity[hard_index], esm_top),
    ):
        for neighbor_rank, development_index in enumerate(indices, start=1):
            neighbor = development.iloc[development_index]
            top_rows.append({
                "hard_case_rank": int(query["rank"]),
                "hard_case_ID": int(query["ID"]),
                "hard_case_sequence": query["sequence"],
                "hard_case_class": query["true_class"],
                "hard_case_total_wrong_count": int(query["total_wrong_count"]),
                "hard_case_consensus_8_of_8_error": bool(query["total_wrong_count"] == 8),
                "similarity_type": similarity_type,
                "neighbor_rank": neighbor_rank,
                "development_neighbor_ID": int(neighbor["ID"]),
                "development_neighbor_sequence": neighbor["sequence"],
                "development_neighbor_class": neighbor["binary_class"],
                "class_relation": "same" if int(neighbor["label"]) == query_label else "opposite",
                "similarity": float(similarities[development_index]),
                "exact_sequence_match": bool(np.isclose(
                    edit_similarity[hard_index, development_index], 1.0, rtol=0, atol=1e-12
                )),
            })

    main_rows.append({
        "rank": int(query["rank"]),
        "ID": int(query["ID"]),
        "sequence": query["sequence"],
        "true_class": query["true_class"],
        "y_true": query_label,
        "total_wrong_count": int(query["total_wrong_count"]),
        "consensus_8_of_8_error": bool(query["total_wrong_count"] == 8),
        "same_class_development_candidates": len(same_candidates),
        "opposite_class_development_candidates": len(opposite_candidates),
        "sequence_nearest_same_ID": int(development.iloc[seq_same]["ID"]),
        "sequence_nearest_same_sequence": development.iloc[seq_same]["sequence"],
        "sequence_nearest_same_similarity": seq_same_value,
        "sequence_nearest_same_tie_count": seq_same_ties,
        "sequence_nearest_opposite_ID": int(development.iloc[seq_opposite]["ID"]),
        "sequence_nearest_opposite_sequence": development.iloc[seq_opposite]["sequence"],
        "sequence_nearest_opposite_similarity": seq_opposite_value,
        "sequence_nearest_opposite_tie_count": seq_opposite_ties,
        "sequence_opposite_class_proximity_margin": seq_opposite_value - seq_same_value,
        "esm2_nearest_same_ID": int(development.iloc[esm_same]["ID"]),
        "esm2_nearest_same_sequence": development.iloc[esm_same]["sequence"],
        "esm2_nearest_same_similarity": esm_same_value,
        "esm2_nearest_same_tie_count": esm_same_ties,
        "esm2_nearest_opposite_ID": int(development.iloc[esm_opposite]["ID"]),
        "esm2_nearest_opposite_sequence": development.iloc[esm_opposite]["sequence"],
        "esm2_nearest_opposite_similarity": esm_opposite_value,
        "esm2_nearest_opposite_tie_count": esm_opposite_ties,
        "esm2_opposite_class_proximity_margin": esm_opposite_value - esm_same_value,
        "sequence_top5_same_count": int(sum(development_labels[index] == query_label for index in sequence_top)),
        "sequence_top5_opposite_count": int(sum(development_labels[index] != query_label for index in sequence_top)),
        "esm2_top5_same_count": int(sum(development_labels[index] == query_label for index in esm_top)),
        "esm2_top5_opposite_count": int(sum(development_labels[index] != query_label for index in esm_top)),
        "exact_sequence_matches_to_development": len(exact_indices),
        "exact_same_class_match_count": len(exact_same),
        "exact_opposite_class_match_count": len(exact_opposite),
        "exact_same_class_match_IDs": ";".join(map(str, exact_same)),
        "exact_opposite_class_match_IDs": ";".join(map(str, exact_opposite)),
    })

main_df = pd.DataFrame(main_rows).sort_values("rank").reset_index(drop=True)
top_neighbors_df = pd.DataFrame(top_rows).sort_values(
    ["hard_case_rank", "similarity_type", "neighbor_rank"]
).reset_index(drop=True)
assert len(main_df) == 15 and len(top_neighbors_df) == 150
assert top_neighbors_df.groupby(["hard_case_ID", "similarity_type"]).size().eq(5).all()
main_df.to_csv(MAIN_OUTPUT, index=False)
top_neighbors_df.to_csv(TOP_NEIGHBOR_OUTPUT, index=False)

summary_metrics = [
    "sequence_nearest_same_similarity", "sequence_nearest_opposite_similarity",
    "sequence_opposite_class_proximity_margin", "esm2_nearest_same_similarity",
    "esm2_nearest_opposite_similarity", "esm2_opposite_class_proximity_margin",
]
group_rows = []
group_rows += summarize(main_df, "All hard cases", summary_metrics)
group_rows += summarize(
    main_df.loc[main_df["consensus_8_of_8_error"]], "Consensus 8/8 failures", summary_metrics
)
group_rows += summarize(
    main_df.loc[~main_df["consensus_8_of_8_error"]], "Other hard cases", summary_metrics
)
group_summary = pd.DataFrame(group_rows)
group_summary.to_csv(GROUP_SUMMARY_OUTPUT, index=False)

labels = [f"{row.ID} ({row.true_class[0]})" for row in main_df.itertuples(index=False)]
y = np.arange(len(main_df))
height = 0.37
fig, ax = plt.subplots(figsize=(11.6, 8.3), facecolor="white")
ax.barh(
    y - height / 2, main_df["sequence_opposite_class_proximity_margin"], height,
    color="#377EB8", label="Normalized edit similarity",
)
ax.barh(
    y + height / 2, main_df["esm2_opposite_class_proximity_margin"], height,
    color="#E41A1C", label="ESM-2 cosine similarity",
)
ax.axvline(0, color="#222222", linewidth=1.1)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Nearest opposite-class minus same-class development similarity")
ax.set_ylabel("Step-59 rank: hard-case ID (true class)")
ax.set_title("Development-set opposite-class proximity of hard test peptides")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(PROXIMITY_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(PROXIMITY_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

map_columns = [
    "sequence_nearest_same_similarity", "sequence_nearest_opposite_similarity",
    "esm2_nearest_same_similarity", "esm2_nearest_opposite_similarity",
]
map_id_columns = [
    "sequence_nearest_same_ID", "sequence_nearest_opposite_ID",
    "esm2_nearest_same_ID", "esm2_nearest_opposite_ID",
]
map_labels = ["Sequence\nsame", "Sequence\nopposite", "ESM-2\nsame", "ESM-2\nopposite"]
map_values = main_df[map_columns].to_numpy(float)
fig, ax = plt.subplots(figsize=(9.6, 9.1), facecolor="white")
image = ax.imshow(map_values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(4), map_labels)
ax.set_yticks(np.arange(len(main_df)), labels)
ax.set_xlabel("Nearest development-neighbor type")
ax.set_ylabel("Step-59 rank: hard-case ID (true class)")
ax.set_title("Hard-case nearest development neighbors and similarities")
for row_index in range(len(main_df)):
    for column_index in range(4):
        value = map_values[row_index, column_index]
        neighbor_id = int(main_df.iloc[row_index][map_id_columns[column_index]])
        ax.text(
            column_index, row_index, f"{value:.3f}\nID {neighbor_id}",
            ha="center", va="center", fontsize=8.0,
            color="white" if value < 0.66 else "black",
        )
colorbar = fig.colorbar(image, ax=ax, pad=0.02)
colorbar.set_label("Similarity")
fig.tight_layout()
fig.savefig(MAP_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(MAP_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

development_active = int(development["label"].eq(1).sum())
development_inactive = int(development["label"].eq(0).sum())
exact_sequence_match_count = int(main_df["exact_sequence_matches_to_development"].gt(0).sum())
exact_opposite_conflict_count = int(main_df["exact_opposite_class_match_count"].gt(0).sum())
neighbor_id_columns = [
    "sequence_nearest_same_ID", "sequence_nearest_opposite_ID",
    "esm2_nearest_same_ID", "esm2_nearest_opposite_ID",
]
qc = pd.DataFrame([{
    "development_peptides": len(development),
    "development_active": development_active,
    "development_inactive": development_inactive,
    "hard_test_cases": len(main_df),
    "consensus_8_of_8_failures": int(main_df["consensus_8_of_8_error"].sum()),
    "edit_comparisons": comparison_count,
    "embedding_dimensions": embeddings.shape[1],
    "top_neighbor_rows": len(top_neighbors_df),
    "top_rows_per_case_and_similarity": bool(
        top_neighbors_df.groupby(["hard_case_ID", "similarity_type"]).size().eq(5).all()
    ),
    "edit_similarities_finite": bool(np.isfinite(edit_similarity).all()),
    "edit_similarities_within_0_1": bool(np.all((edit_similarity >= 0) & (edit_similarity <= 1))),
    "esm2_similarities_finite": bool(np.isfinite(esm_similarity).all()),
    "all_neighbor_ids_in_development": bool(main_df[neighbor_id_columns].isin(set(development_ids)).all().all()),
    "all_neighbor_classes_correct": True,
    "exact_sequence_match_hard_cases": exact_sequence_match_count,
    "exact_opposite_class_conflict_hard_cases": exact_opposite_conflict_count,
    "test_development_id_overlap": len(set(hard["ID"]) & set(development["ID"])),
    "models_trained": False,
    "labels_changed": False,
    "cv_folds_changed": False,
    "hyperparameters_changed": False,
    "threshold_0_5_changed": False,
    "representation_selected": False,
}])
id_to_label = development.set_index("ID")["label"].to_dict()
for row in main_df.itertuples(index=False):
    assert id_to_label[row.sequence_nearest_same_ID] == row.y_true
    assert id_to_label[row.esm2_nearest_same_ID] == row.y_true
    assert id_to_label[row.sequence_nearest_opposite_ID] != row.y_true
    assert id_to_label[row.esm2_nearest_opposite_ID] != row.y_true
qc.to_csv(QC_OUTPUT, index=False)

print("\nFive consensus 8/8 failures:")
print(
    main_df.loc[main_df["consensus_8_of_8_error"], [
        "ID", "sequence_nearest_same_ID", "sequence_nearest_same_similarity",
        "sequence_nearest_opposite_ID", "sequence_nearest_opposite_similarity",
        "sequence_opposite_class_proximity_margin", "esm2_nearest_same_ID",
        "esm2_nearest_same_similarity", "esm2_nearest_opposite_ID",
        "esm2_nearest_opposite_similarity", "esm2_opposite_class_proximity_margin",
        "exact_sequence_matches_to_development", "exact_opposite_class_match_count",
    ]].round(6).to_string(index=False)
)
print("\nOutput checks:")
for path in (
    MAIN_OUTPUT, TOP_NEIGHBOR_OUTPUT, GROUP_SUMMARY_OUTPUT, QC_OUTPUT,
    PROXIMITY_PNG, PROXIMITY_PDF, MAP_PNG, MAP_PDF,
):
    print(path.name, ":", path.exists())
print("\n" + "=" * 112)
print("STEP 63 SUMMARY")
print("=" * 112)
print("Development peptides:", len(development))
print("Development Active:", development_active)
print("Development Inactive:", development_inactive)
print("Hard locked-test cases:", len(main_df))
print("Consensus 8/8 failures:", int(main_df["consensus_8_of_8_error"].sum()))
print("Top-development-neighbor rows:", len(top_neighbors_df))
print("Exact sequence matches to development:", exact_sequence_match_count)
print("Exact opposite-class conflicts:", exact_opposite_conflict_count)
print("\nSTEP 63 COMPLETED SUCCESSFULLY")
print("=" * 112)

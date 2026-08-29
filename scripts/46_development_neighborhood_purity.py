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
PURITY_OUTPUT = RESULTS_DIR / "step64_development_neighborhood_purity.csv"
BALANCED_OUTPUT = RESULTS_DIR / "step64_balanced_neighbor_similarity.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "step64_neighborhood_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step64_neighborhood_qc.csv"
PURITY_PNG = FIGURES_DIR / "Step64_Hard_Case_Neighborhood_Purity.png"
PURITY_PDF = FIGURES_DIR / "Step64_Hard_Case_Neighborhood_Purity.pdf"
MARGIN_PNG = FIGURES_DIR / "Step64_Balanced_Development_Similarity_Margins.png"
MARGIN_PDF = FIGURES_DIR / "Step64_Balanced_Development_Similarity_Margins.pdf"
K_VALUES = (3, 5, 10, 20)


def ordered_indices(similarities, candidate_ids, candidates=None):
    if candidates is None:
        candidates = np.arange(len(similarities))
    candidates = np.asarray(candidates, dtype=int)
    order = np.lexsort((candidate_ids[candidates], -similarities[candidates]))
    return candidates[order]


print("=" * 112)
print("STEP 64 - DEVELOPMENT-SET NEIGHBORHOOD PURITY AND CLASS-CONFLICT ANALYSIS")
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
alignment = hard[["ID", "sequence", "y_true", "true_class"]].merge(
    test_metadata[["ID", "sequence", "label", "binary_class"]],
    on=["ID", "sequence"], how="inner", validate="one_to_one",
)
assert len(alignment) == 15
assert alignment["y_true"].eq(alignment["label"]).all()
assert alignment["true_class"].eq(alignment["binary_class"]).all()

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 0.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -1.0
aligner.extend_gap_score = -1.0
edit_similarity = np.empty((15, 720), dtype=np.float64)
for i, query in enumerate(hard["sequence"]):
    for j, candidate in enumerate(development["sequence"]):
        distance = -float(aligner.score(query, candidate))
        edit_similarity[i, j] = 1.0 - distance / max(len(query), len(candidate))
assert np.isfinite(edit_similarity).all()
assert np.all((edit_similarity >= 0) & (edit_similarity <= 1))

hard_rows = test_metadata.set_index("ID").loc[hard["ID"], "embedding_row"].to_numpy(int)
dev_rows = development["embedding_row"].to_numpy(int)
hard_embeddings = embeddings[hard_rows].astype(np.float64)
dev_embeddings = embeddings[dev_rows].astype(np.float64)
hard_embeddings /= np.linalg.norm(hard_embeddings, axis=1, keepdims=True)
dev_embeddings /= np.linalg.norm(dev_embeddings, axis=1, keepdims=True)
esm_similarity = np.clip(hard_embeddings @ dev_embeddings.T, -1.0, 1.0)
assert np.isfinite(esm_similarity).all()

dev_ids = development["ID"].to_numpy(int)
dev_labels = development["label"].to_numpy(int)
purity_rows = []
balanced_rows = []
top_k_unique = True
for i, query in hard.iterrows():
    y = int(query["y_true"])
    same_candidates = np.flatnonzero(dev_labels == y)
    opposite_candidates = np.flatnonzero(dev_labels != y)
    for representation, similarities in (
        ("sequence", edit_similarity[i]), ("esm2", esm_similarity[i])
    ):
        natural_order = ordered_indices(similarities, dev_ids)
        for k in K_VALUES:
            selected = natural_order[:k]
            top_k_unique &= len(np.unique(dev_ids[selected])) == k
            same_mask = dev_labels[selected] == y
            same_count = int(same_mask.sum())
            opposite_count = k - same_count
            purity_rows.append({
                "hard_case_rank": int(query["rank"]), "hard_case_ID": int(query["ID"]),
                "hard_case_sequence": query["sequence"], "hard_case_class": query["true_class"],
                "y_true": y, "total_wrong_count": int(query["total_wrong_count"]),
                "consensus_8_of_8_error": bool(query["total_wrong_count"] == 8),
                "representation": representation, "k": k,
                "same_class_count": same_count, "opposite_class_count": opposite_count,
                "same_class_fraction": same_count / k,
                "opposite_class_fraction": opposite_count / k,
                "neighborhood_purity": same_count / k,
                "class_conflict_score": opposite_count / k,
                "mean_topk_similarity": float(similarities[selected].mean()),
                "mean_same_class_similarity": float(similarities[selected][same_mask].mean())
                if same_count else np.nan,
                "mean_opposite_class_similarity": float(similarities[selected][~same_mask].mean())
                if opposite_count else np.nan,
                "top_k_development_IDs": ";".join(map(str, dev_ids[selected])),
                "top_k_class_relations": ";".join(
                    "same" if value else "opposite" for value in same_mask
                ),
            })
        top_same = ordered_indices(similarities, dev_ids, same_candidates)[:5]
        top_opposite = ordered_indices(similarities, dev_ids, opposite_candidates)[:5]
        mean_same = float(similarities[top_same].mean())
        mean_opposite = float(similarities[top_opposite].mean())
        balanced_rows.append({
            "hard_case_rank": int(query["rank"]), "hard_case_ID": int(query["ID"]),
            "hard_case_sequence": query["sequence"], "hard_case_class": query["true_class"],
            "y_true": y, "total_wrong_count": int(query["total_wrong_count"]),
            "consensus_8_of_8_error": bool(query["total_wrong_count"] == 8),
            "representation": representation, "same_class_neighbor_count": len(top_same),
            "opposite_class_neighbor_count": len(top_opposite),
            "mean_top5_same_class_similarity": mean_same,
            "mean_top5_opposite_class_similarity": mean_opposite,
            "balanced_similarity_margin": mean_opposite - mean_same,
            "top5_same_class_IDs": ";".join(map(str, dev_ids[top_same])),
            "top5_opposite_class_IDs": ";".join(map(str, dev_ids[top_opposite])),
        })

purity = pd.DataFrame(purity_rows).sort_values(
    ["hard_case_rank", "representation", "k"]
).reset_index(drop=True)
balanced = pd.DataFrame(balanced_rows).sort_values(
    ["hard_case_rank", "representation"]
).reset_index(drop=True)
assert len(purity) == 120 and len(balanced) == 30
assert set(purity["k"]) == set(K_VALUES)
assert (purity["same_class_count"] + purity["opposite_class_count"]).eq(purity["k"]).all()
assert np.allclose(purity["same_class_fraction"] + purity["opposite_class_fraction"], 1)
assert purity["neighborhood_purity"].between(0, 1).all()
assert balanced["same_class_neighbor_count"].eq(5).all()
assert balanced["opposite_class_neighbor_count"].eq(5).all()
assert np.allclose(
    balanced["balanced_similarity_margin"],
    balanced["mean_top5_opposite_class_similarity"]
    - balanced["mean_top5_same_class_similarity"],
)
purity.to_csv(PURITY_OUTPUT, index=False)
balanced.to_csv(BALANCED_OUTPUT, index=False)

groups = [
    ("Consensus 8/8 failures", lambda frame: frame["consensus_8_of_8_error"]),
    ("Other hard cases", lambda frame: ~frame["consensus_8_of_8_error"]),
    ("All hard cases", lambda frame: pd.Series(True, index=frame.index)),
]
summary_rows = []
for group_name, selector in groups:
    for (representation, k), part in purity.loc[selector(purity)].groupby(
        ["representation", "k"], sort=True
    ):
        values = part["neighborhood_purity"]
        summary_rows.append({
            "analysis_type": "natural_top_k", "analysis_group": group_name,
            "representation": representation, "k": int(k), "n": len(part),
            "mean_neighborhood_purity": values.mean(),
            "median_neighborhood_purity": values.median(),
            "purity_below_0_5_count": int((values < 0.5).sum()),
            "purity_equal_0_5_count": int(np.isclose(values, 0.5, rtol=0, atol=1e-12).sum()),
            "purity_above_0_5_count": int((values > 0.5).sum()),
            "mean_balanced_margin": np.nan, "median_balanced_margin": np.nan,
            "opposite_favored_count": np.nan, "same_favored_count": np.nan,
            "tie_count": np.nan,
        })
    for representation, part in balanced.loc[selector(balanced)].groupby("representation", sort=True):
        values = part["balanced_similarity_margin"]
        summary_rows.append({
            "analysis_type": "balanced_top5_per_class", "analysis_group": group_name,
            "representation": representation, "k": 10, "n": len(part),
            "mean_neighborhood_purity": np.nan, "median_neighborhood_purity": np.nan,
            "purity_below_0_5_count": np.nan, "purity_equal_0_5_count": np.nan,
            "purity_above_0_5_count": np.nan, "mean_balanced_margin": values.mean(),
            "median_balanced_margin": values.median(),
            "opposite_favored_count": int((values > 0).sum()),
            "same_favored_count": int((values < 0).sum()),
            "tie_count": int(np.isclose(values, 0, rtol=0, atol=1e-12).sum()),
        })
summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUTPUT, index=False)

row_labels = [f"{row.ID} ({row.true_class[0]})" for row in hard.itertuples(index=False)]
fig, axes = plt.subplots(1, 2, figsize=(12.0, 8.5), facecolor="white", sharey=True)
for ax, representation, title in zip(
    axes, ("sequence", "esm2"), ("A  Normalized edit similarity", "B  ESM-2 cosine similarity")
):
    matrix = purity.loc[purity["representation"] == representation].pivot(
        index="hard_case_ID", columns="k", values="neighborhood_purity"
    ).loc[hard["ID"], list(K_VALUES)].to_numpy()
    image = ax.imshow(matrix, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [f"k={k}" for k in K_VALUES])
    ax.set_yticks(range(15), row_labels)
    ax.set_title(title)
    ax.set_xlabel("Natural development neighborhood size")
    for row in range(15):
        for column in range(4):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value < 0.25 or value > 0.80 else "black")
axes[0].set_ylabel("Step-59 rank: hard-case ID (true class)")
colorbar = fig.colorbar(image, ax=axes, pad=0.025, fraction=0.035)
colorbar.set_label("Same-class neighborhood purity")
fig.suptitle("Development-set neighborhood purity of hard test peptides", y=0.98)
fig.subplots_adjust(left=0.15, right=0.90, bottom=0.08, top=0.91, wspace=0.12)
fig.savefig(PURITY_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(PURITY_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

sequence_margin = balanced.loc[balanced["representation"] == "sequence"].set_index(
    "hard_case_ID"
).loc[hard["ID"], "balanced_similarity_margin"].to_numpy()
esm_margin = balanced.loc[balanced["representation"] == "esm2"].set_index(
    "hard_case_ID"
).loc[hard["ID"], "balanced_similarity_margin"].to_numpy()
ypos = np.arange(15)
fig, ax = plt.subplots(figsize=(11.3, 8.3), facecolor="white")
ax.scatter(sequence_margin, ypos - 0.14, s=54, color="#377EB8", label="Sequence", zorder=3)
ax.scatter(esm_margin, ypos + 0.14, s=54, color="#E41A1C", marker="D", label="ESM-2", zorder=3)
for row in range(15):
    ax.plot([sequence_margin[row], esm_margin[row]], [ypos[row] - 0.14, ypos[row] + 0.14],
            color="#BDBDBD", linewidth=0.8, zorder=1)
ax.axvline(0, color="#222222", linewidth=1.1)
ax.set_yticks(ypos, row_labels)
ax.invert_yaxis()
ax.set_xlabel("Mean top-5 opposite-class similarity minus mean top-5 same-class similarity")
ax.set_ylabel("Step-59 rank: hard-case ID (true class)")
ax.set_title("Class-balanced development-neighborhood similarity margins")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(MARGIN_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(MARGIN_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "development_peptides": len(development),
    "development_active": int(development["label"].eq(1).sum()),
    "development_inactive": int(development["label"].eq(0).sum()),
    "hard_cases": len(hard),
    "consensus_8_of_8_failures": int(hard["total_wrong_count"].eq(8).sum()),
    "sequence_development_comparisons": int(edit_similarity.size),
    "esm2_development_comparisons": int(esm_similarity.size),
    "purity_rows": len(purity), "balanced_rows": len(balanced),
    "k_values_exact": ";".join(map(str, sorted(purity["k"].unique()))),
    "all_top_k_ids_unique": bool(top_k_unique),
    "test_candidate_overlap": len(set(hard["ID"]) & set(development["ID"])),
    "counts_sum_to_k": bool((purity["same_class_count"] + purity["opposite_class_count"]).eq(purity["k"]).all()),
    "fractions_sum_to_one": bool(np.allclose(purity["same_class_fraction"] + purity["opposite_class_fraction"], 1)),
    "all_similarities_finite": bool(np.isfinite(edit_similarity).all() and np.isfinite(esm_similarity).all()),
    "all_purity_within_0_1": bool(purity["neighborhood_purity"].between(0, 1).all()),
    "balanced_five_same_each": bool(balanced["same_class_neighbor_count"].eq(5).all()),
    "balanced_five_opposite_each": bool(balanced["opposite_class_neighbor_count"].eq(5).all()),
    "balanced_margin_identity": bool(np.allclose(
        balanced["balanced_similarity_margin"],
        balanced["mean_top5_opposite_class_similarity"] - balanced["mean_top5_same_class_similarity"]
    )),
    "models_trained": False, "models_retrained": False,
    "thresholds_optimized": False, "labels_changed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

focus = purity.loc[
    purity["consensus_8_of_8_error"] & purity["representation"].eq("esm2") & purity["k"].eq(10),
    ["hard_case_ID", "same_class_count", "opposite_class_count", "neighborhood_purity"],
].merge(
    balanced.loc[balanced["consensus_8_of_8_error"] & balanced["representation"].eq("esm2"),
                 ["hard_case_ID", "mean_top5_same_class_similarity",
                  "mean_top5_opposite_class_similarity", "balanced_similarity_margin"]],
    on="hard_case_ID", validate="one_to_one",
).sort_values("hard_case_ID")
print("\nFive consensus 8/8 failures - ESM-2 top-10 and balanced top-5-per-class:")
print(focus.round(6).to_string(index=False))
print("\nOutput checks:")
for path in (PURITY_OUTPUT, BALANCED_OUTPUT, SUMMARY_OUTPUT, QC_OUTPUT,
             PURITY_PNG, PURITY_PDF, MARGIN_PNG, MARGIN_PDF):
    print(path.name, ":", path.exists())
print("\nPurity rows:", len(purity))
print("Balanced rows:", len(balanced))
print("STEP 64 COMPLETED SUCCESSFULLY")
print("=" * 112)

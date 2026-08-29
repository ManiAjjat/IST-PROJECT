from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RANKING_INPUT = RESULTS_DIR / "step59_consensus_hard_case_ranking.csv"
EXTREMES_INPUT = RESULTS_DIR / "step58_persistent_error_extremes.csv"
COMPOSITION_INPUT = RESULTS_DIR / "step60_hard_case_amino_acid_composition.csv"
TEST_NEIGHBOR_INPUT = RESULTS_DIR / "step61_hard_case_nearest_neighbors.csv"
DEV_NEIGHBOR_INPUT = RESULTS_DIR / "step63_hard_case_development_neighbors.csv"
PURITY_INPUT = RESULTS_DIR / "step64_development_neighborhood_purity.csv"
PERTURBATION_INPUT = RESULTS_DIR / "step82_peptide_perturbation_summary.csv"
TOP_RESIDUE_INPUT = RESULTS_DIR / "step83_hard_case_top_residue_context.csv"

MAIN_OUTPUT = RESULTS_DIR / "step84_integrated_hard_case_interpretability.csv"
UNIVERSAL_OUTPUT = RESULTS_DIR / "step84_universal_error_evidence_table.csv"
FLAG_OUTPUT = RESULTS_DIR / "step84_interpretability_flag_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step84_integrated_interpretability_qc.csv"
PNG_OUTPUT = FIGURES_DIR / "Step84_Universal_Hard_Case_Evidence_Map.png"
PDF_OUTPUT = FIGURES_DIR / "Step84_Universal_Hard_Case_Evidence_Map.pdf"

UNIVERSAL_IDS = [48, 40, 145, 56, 68]
LOW_PURITY_THRESHOLD = 0.50
HIGH_DESCRIPTOR_THRESHOLD = 1.0
HIGH_SENSITIVITY_THRESHOLD = 0.10


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def relation(margin):
    if pd.isna(margin):
        return "Missing"
    if margin > 0:
        return "Opposite-class closer"
    if margin < 0:
        return "Same-class closer"
    return "Tied"


def qualitative_category(row):
    seq = row["nearest_development_sequence_margin"]
    esm = row["nearest_development_esm2_margin"]
    if seq > 0 and esm > 0:
        return "Cross-class neighborhood conflict"
    if seq < 0 and esm < 0 and bool(row["high_residue_sensitivity"]):
        return "Same-class neighborhood but model-sensitive"
    if np.sign(seq) != np.sign(esm) or seq == 0 or esm == 0:
        return "Mixed/ambiguous neighborhood"
    if bool(row["high_residue_sensitivity"]):
        return "Sequence-context-sensitive hard case"
    return "Mixed/ambiguous neighborhood"


print("=" * 104)
print("STEP 84 - INTEGRATED INTERPRETABILITY EVIDENCE TABLE")
print("=" * 104)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ranking_all = pd.read_csv(RANKING_INPUT)
extremes = pd.read_csv(EXTREMES_INPUT)
composition = pd.read_csv(COMPOSITION_INPUT)
test_neighbors = pd.read_csv(TEST_NEIGHBOR_INPUT)
dev_neighbors = pd.read_csv(DEV_NEIGHBOR_INPUT)
purity = pd.read_csv(PURITY_INPUT)
perturbation = pd.read_csv(PERTURBATION_INPUT)
top_residue = pd.read_csv(TOP_RESIDUE_INPUT)

hard = ranking_all.loc[ranking_all["total_wrong_count"].ge(3)].copy()
assert len(hard) == 15 and hard["ID"].is_unique
assert hard.loc[hard["total_wrong_count"].eq(8), "ID"].tolist() == UNIVERSAL_IDS
expected_ids = set(hard["ID"])
for source, id_column in ((composition, "ID"), (test_neighbors, "ID"),
                          (dev_neighbors, "ID"), (purity, "hard_case_ID")):
    assert expected_ids.issubset(set(source[id_column]))
assert set(extremes["ID"]).issubset(expected_ids)

test_index = test_neighbors.set_index("ID")
dev_index = dev_neighbors.set_index("ID")
purity10 = purity.loc[purity["k"].eq(10)].pivot(index="hard_case_ID", columns="representation",
                                                values="neighborhood_purity")
assert purity10.shape == (15, 2) and {"sequence", "esm2"}.issubset(purity10.columns)
consensus_perturbation = perturbation.loc[
    perturbation["classifier"].eq("RF/XGBoost consensus")].set_index("peptide_ID")
assert set(consensus_perturbation.index) == set(UNIVERSAL_IDS) | {573, 519, 197, 567, 806}

top_lookup = {}
for peptide_id, group in top_residue.groupby("peptide_ID"):
    top_lookup[int(peptide_id)] = group.sort_values("top3_rank").set_index("top3_rank")

rows = []
for row in hard.itertuples(index=False):
    peptide_id = int(row.ID)
    sensitivity_available = peptide_id in consensus_perturbation.index
    perturb = consensus_perturbation.loc[peptide_id] if sensitivity_available else None
    record = {
        "rank": int(row.rank), "ID": peptide_id, "sequence": row.sequence,
        "true_class": row.true_class, "total_wrong_count": int(row.total_wrong_count),
        "traditional_wrong_count": int(row.traditional_wrong_count),
        "esm2_wrong_count": int(row.esm2_wrong_count),
        "all_models_mean_true_class_probability": row.all_models_mean_true_class_probability,
        "mean_absolute_descriptor_z": row.mean_absolute_descriptor_z,
        "most_extreme_descriptor": row.most_extreme_descriptor,
        "nearest_test_sequence_margin": test_index.loc[peptide_id, "sequence_opposite_class_proximity_margin"],
        "nearest_test_esm2_margin": test_index.loc[peptide_id, "esm2_opposite_class_proximity_margin"],
        "nearest_development_sequence_margin": dev_index.loc[peptide_id, "sequence_opposite_class_proximity_margin"],
        "nearest_development_esm2_margin": dev_index.loc[peptide_id, "esm2_opposite_class_proximity_margin"],
        "development_sequence_top10_purity": purity10.loc[peptide_id, "sequence"],
        "development_esm2_top10_purity": purity10.loc[peptide_id, "esm2"],
        "consensus_mean_residue_sensitivity": perturb["mean_absolute_residue_sensitivity"] if sensitivity_available else np.nan,
        "consensus_max_residue_sensitivity": perturb["maximum_absolute_residue_sensitivity"] if sensitivity_available else np.nan,
        "consensus_flip_positions": perturb["number_of_prediction_flips"] if sensitivity_available else np.nan,
    }
    for top_rank in (1, 2, 3):
        if peptide_id in top_lookup:
            top = top_lookup[peptide_id].loc[top_rank]
            record[f"top_sensitive_position_{top_rank}"] = int(top["position"])
            record[f"top_sensitive_residue_{top_rank}"] = top["original_residue"]
            record[f"top_sensitive_value_{top_rank}"] = top["consensus_absolute_sensitivity"]
            record[f"top_sensitive_motif_context_{top_rank}"] = (
                top["motifs_covering_position"] if bool(top["inside_recurrent_motif"])
                else "Outside recurrent motifs")
            record[f"top_sensitive_residue_category_{top_rank}"] = top["residue_category"]
        else:
            record[f"top_sensitive_position_{top_rank}"] = np.nan
            record[f"top_sensitive_residue_{top_rank}"] = pd.NA
            record[f"top_sensitive_value_{top_rank}"] = np.nan
            record[f"top_sensitive_motif_context_{top_rank}"] = pd.NA
            record[f"top_sensitive_residue_category_{top_rank}"] = pd.NA

    record.update({
        "universal_8_of_8_error": row.total_wrong_count == 8,
        "persistent_under_all_esm2_models": row.esm2_wrong_count == 4,
        "opposite_class_test_neighbor_sequence": record["nearest_test_sequence_margin"] > 0,
        "opposite_class_test_neighbor_esm2": record["nearest_test_esm2_margin"] > 0,
        "opposite_class_development_neighbor_sequence": record["nearest_development_sequence_margin"] > 0,
        "opposite_class_development_neighbor_esm2": record["nearest_development_esm2_margin"] > 0,
        "low_esm2_top10_purity": record["development_esm2_top10_purity"] < LOW_PURITY_THRESHOLD,
        "high_descriptor_extremeness": record["mean_absolute_descriptor_z"] >= HIGH_DESCRIPTOR_THRESHOLD,
        "residue_sensitivity_available": sensitivity_available,
        "high_residue_sensitivity": (record["consensus_max_residue_sensitivity"] >= HIGH_SENSITIVITY_THRESHOLD)
                                    if sensitivity_available else False,
        "motif_linked_top_residue": (record["top_sensitive_motif_context_1"] != "Outside recurrent motifs")
                                    if sensitivity_available else False,
    })
    rows.append(record)

integrated = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
assert len(integrated) == 15 and integrated["ID"].is_unique
assert integrated.loc[integrated["residue_sensitivity_available"], "ID"].tolist() == UNIVERSAL_IDS
missing_residue = integrated.loc[~integrated["residue_sensitivity_available"]]
residue_columns = [column for column in integrated.columns if
                   column.startswith("consensus_") or column.startswith("top_sensitive_")]
assert missing_residue[residue_columns].isna().all().all()

universal_rows = []
for _, row in integrated.loc[integrated["ID"].isin(UNIVERSAL_IDS)].iterrows():
    universal_rows.append({
        "ID": row["ID"], "class": row["true_class"], "sequence": row["sequence"],
        "wrong_models_out_of_8": row["total_wrong_count"],
        "mean_true_class_probability": row["all_models_mean_true_class_probability"],
        "most_extreme_physicochemical_descriptor": row["most_extreme_descriptor"],
        "mean_absolute_descriptor_z": row["mean_absolute_descriptor_z"],
        "test_neighborhood_sequence_relation": relation(row["nearest_test_sequence_margin"]),
        "test_sequence_margin": row["nearest_test_sequence_margin"],
        "test_neighborhood_esm2_relation": relation(row["nearest_test_esm2_margin"]),
        "test_esm2_margin": row["nearest_test_esm2_margin"],
        "development_neighborhood_sequence_relation": relation(row["nearest_development_sequence_margin"]),
        "development_sequence_margin": row["nearest_development_sequence_margin"],
        "development_neighborhood_esm2_relation": relation(row["nearest_development_esm2_margin"]),
        "development_esm2_margin": row["nearest_development_esm2_margin"],
        "development_esm2_top10_purity": row["development_esm2_top10_purity"],
        "top_sensitive_residue": f"{row['top_sensitive_residue_1']}{int(row['top_sensitive_position_1'])}",
        "top_sensitivity": row["top_sensitive_value_1"],
        "top_residue_category": row["top_sensitive_residue_category_1"],
        "top_residue_motif_context": row["top_sensitive_motif_context_1"],
        "mutation_induced_flip_present": row["consensus_flip_positions"] > 0,
        "qualitative_evidence_category": qualitative_category(row),
    })
universal = pd.DataFrame(universal_rows)
assert universal["ID"].tolist() == UNIVERSAL_IDS

flag_columns = [
    "universal_8_of_8_error", "persistent_under_all_esm2_models",
    "opposite_class_test_neighbor_sequence", "opposite_class_test_neighbor_esm2",
    "opposite_class_development_neighbor_sequence", "opposite_class_development_neighbor_esm2",
    "low_esm2_top10_purity", "high_descriptor_extremeness", "residue_sensitivity_available",
    "high_residue_sensitivity", "motif_linked_top_residue",
]
flag_summary = pd.DataFrame([{
    "flag": flag, "threshold_or_definition": {
        "universal_8_of_8_error": "total_wrong_count = 8",
        "persistent_under_all_esm2_models": "esm2_wrong_count = 4",
        "opposite_class_test_neighbor_sequence": "test sequence margin > 0",
        "opposite_class_test_neighbor_esm2": "test ESM-2 margin > 0",
        "opposite_class_development_neighbor_sequence": "development sequence margin > 0",
        "opposite_class_development_neighbor_esm2": "development ESM-2 margin > 0",
        "low_esm2_top10_purity": "development ESM-2 top-10 purity < 0.50",
        "high_descriptor_extremeness": "mean absolute descriptor z >= 1.0",
        "residue_sensitivity_available": "included in predefined Step-82 hard panel",
        "high_residue_sensitivity": "consensus maximum absolute sensitivity >= 0.10",
        "motif_linked_top_residue": "top sensitive residue covered by predefined recurrent motif",
    }[flag], "hard_case_count": int(integrated[flag].sum()),
    "hard_case_fraction": integrated[flag].mean(),
    "universal_error_count": int(integrated.loc[integrated["universal_8_of_8_error"], flag].sum()),
    "universal_error_fraction": integrated.loc[integrated["universal_8_of_8_error"], flag].mean(),
} for flag in flag_columns])

integrated.to_csv(MAIN_OUTPUT, index=False)
universal.to_csv(UNIVERSAL_OUTPUT, index=False)
flag_summary.to_csv(FLAG_OUTPUT, index=False)

# Evidence map: colors are normalized descriptive burden; annotations are raw values.
plot = integrated.set_index("ID").loc[UNIVERSAL_IDS]
sequence_conflict = np.maximum(plot["nearest_test_sequence_margin"],
                               plot["nearest_development_sequence_margin"]).clip(lower=0)
esm_conflict = np.maximum(plot["nearest_test_esm2_margin"],
                          plot["nearest_development_esm2_margin"]).clip(lower=0)
raw_matrix = np.column_stack([
    plot["mean_absolute_descriptor_z"], sequence_conflict, esm_conflict,
    1 - plot["development_esm2_top10_purity"], plot["consensus_max_residue_sensitivity"],
    plot["motif_linked_top_residue"].astype(float),
])
normalized = np.zeros_like(raw_matrix, dtype=float)
for column in range(raw_matrix.shape[1]):
    low, high = raw_matrix[:, column].min(), raw_matrix[:, column].max()
    normalized[:, column] = (raw_matrix[:, column] - low) / (high - low) if high > low else raw_matrix[:, column]
annotations = np.empty(raw_matrix.shape, dtype=object)
for i, (_, row) in enumerate(plot.iterrows()):
    annotations[i, 0] = f"{row['mean_absolute_descriptor_z']:.2f}"
    annotations[i, 1] = f"T {row['nearest_test_sequence_margin']:+.2f}\nD {row['nearest_development_sequence_margin']:+.2f}"
    annotations[i, 2] = f"T {row['nearest_test_esm2_margin']:+.3f}\nD {row['nearest_development_esm2_margin']:+.3f}"
    annotations[i, 3] = f"purity\n{row['development_esm2_top10_purity']:.1f}"
    annotations[i, 4] = f"{row['consensus_max_residue_sensitivity']:.3f}"
    annotations[i, 5] = "Yes" if row["motif_linked_top_residue"] else "No"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 12,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 9})
fig, ax = plt.subplots(figsize=(10.8, 4.5), facecolor="white")
image = ax.imshow(normalized, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
column_labels = ["Descriptor\nextremeness", "Sequence neighbor\nconflict margin",
                 "ESM-2 neighbor\nconflict margin", "Development ESM-2\nneighborhood burden",
                 "Maximum residue\nsensitivity", "Motif-linked\ntop residue"]
ax.set_xticks(range(6), column_labels)
ax.set_yticks(range(5), [f"ID {pid} ({plot.loc[pid, 'true_class']})" for pid in UNIVERSAL_IDS])
for i in range(5):
    for j in range(6):
        color = "white" if normalized[i, j] > 0.58 else "black"
        ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8.2,
                color=color, fontweight="bold" if j in (4, 5) else "normal")
ax.set_title("Universal hard-case interpretability evidence map", fontweight="bold", pad=12)
ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values(): spine.set_visible(False)
cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
cbar.set_label("Within-column normalized descriptive evidence burden")
fig.text(0.5, 0.015,
         "T/D = locked-test/development margin; positive margin favors an opposite-class neighbor. Colors are visualization-only; no score was calculated.",
         ha="center", fontsize=8.1)
fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.98))
fig.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "hard_cases_retained": len(integrated), "universal_errors_retained": len(universal),
    "universal_IDs_exact": ";".join(map(str, universal["ID"])),
    "hard_IDs_unique": bool(integrated["ID"].is_unique),
    "all_source_hard_IDs_aligned": True,
    "existing_values_directly_reproduced": True,
    "residue_sensitivity_available_cases": int(integrated["residue_sensitivity_available"].sum()),
    "residue_sensitivity_missing_cases": int((~integrated["residue_sensitivity_available"]).sum()),
    "missing_perturbation_values_only_non_step82": bool(missing_residue[residue_columns].isna().all().all()),
    "low_purity_threshold": LOW_PURITY_THRESHOLD,
    "high_descriptor_threshold": HIGH_DESCRIPTOR_THRESHOLD,
    "high_sensitivity_threshold": HIGH_SENSITIVITY_THRESHOLD,
    "weighted_score_created": False, "inferred_numerical_values": False,
    "new_model_fitting": False, "new_esm2_inference": False,
    "bootstrap_sampling": False, "threshold_optimization": False,
    "significance_testing": False, "qc_passed": True,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nUniversal error categories:")
print(universal[["ID", "qualitative_evidence_category", "top_sensitive_residue",
                 "top_sensitivity", "top_residue_motif_context"]].to_string(index=False))
print("\nFlag summary:")
print(flag_summary[["flag", "hard_case_count", "universal_error_count"]].to_string(index=False))
print("\nOutputs:")
for path in (MAIN_OUTPUT, UNIVERSAL_OUTPUT, FLAG_OUTPUT, QC_OUTPUT, PNG_OUTPUT, PDF_OUTPUT):
    print(path, path.stat().st_size, "bytes", sha256(path))
print("\nSTEP 84 COMPLETED SUCCESSFULLY")
print("=" * 104)

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESIDUE_INPUT = RESULTS_DIR / "step82_residue_consensus_importance.csv"
SELECTED_INPUT = RESULTS_DIR / "step82_selected_peptides.csv"
COMPOSITION_INPUT = RESULTS_DIR / "step60_hard_case_amino_acid_composition.csv"
MOTIF_INPUT = RESULTS_DIR / "step60_hard_case_motif_summary.csv"

CONTEXT_OUTPUT = RESULTS_DIR / "step83_residue_physicochemical_context.csv"
CATEGORY_OUTPUT = RESULTS_DIR / "step83_residue_category_summary.csv"
MOTIF_OUTPUT = RESULTS_DIR / "step83_motif_sensitivity_summary.csv"
TOP_OUTPUT = RESULTS_DIR / "step83_hard_case_top_residue_context.csv"
QC_OUTPUT = RESULTS_DIR / "step83_residue_context_qc.csv"
FIG1_PNG = FIGURES_DIR / "Step83_Residue_Category_Sensitivity.png"
FIG1_PDF = FIGURES_DIR / "Step83_Residue_Category_Sensitivity.pdf"
FIG2_PNG = FIGURES_DIR / "Step83_Motif_Context_Sensitivity.png"
FIG2_PDF = FIGURES_DIR / "Step83_Motif_Context_Sensitivity.pdf"

MOTIFS = ["KK", "KL", "AK", "LA", "LAK", "FA", "KA", "LK", "AKL", "KLL", "KAL", "KLA"]
CATEGORY_ORDER = ["Basic", "Hydrophobic", "Acidic", "Polar/other"]
HARD_GROUP = "Consensus hard error"
CORRECT_GROUP = "High-confidence consensus correct"
HARD_IDS = [48, 40, 145, 56, 68]

BASIC = set("KRH")
ACIDIC = set("DE")
HYDROPHOBIC = set("AVILMFWY")
AROMATIC = set("FWY")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def category(residue):
    if residue in BASIC:
        return "Basic"
    if residue in ACIDIC:
        return "Acidic"
    if residue in HYDROPHOBIC:
        return "Hydrophobic"
    return "Polar/other"


def charge_class(residue):
    if residue in BASIC:
        return "basic"
    if residue in ACIDIC:
        return "acidic"
    return "neutral"


def motif_hits(sequence):
    hits = {position: [] for position in range(1, len(sequence) + 1)}
    for motif in MOTIFS:
        start = 0
        while True:
            index = sequence.find(motif, start)
            if index < 0:
                break
            for position in range(index + 1, index + len(motif) + 1):
                hits[position].append(motif)
            start = index + 1
    return hits


print("=" * 104)
print("STEP 83 - LINKING RESIDUE SENSITIVITY TO PHYSICOCHEMICAL PROPERTIES AND MOTIFS")
print("=" * 104)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

residues = pd.read_csv(RESIDUE_INPUT)
selected = pd.read_csv(SELECTED_INPUT)
composition = pd.read_csv(COMPOSITION_INPUT)
motif_reference = pd.read_csv(MOTIF_INPUT)

assert residues.shape == (182, 17)
assert selected.shape[0] == 10 and selected["ID"].is_unique
assert selected.loc[selected["analysis_group"].eq(HARD_GROUP), "ID"].tolist() == HARD_IDS
assert set(MOTIFS).issubset(set(motif_reference["motif"]))
assert set(HARD_IDS).issubset(set(composition["ID"]))
assert residues.groupby("peptide_ID").size().to_dict() == selected.set_index("ID")["sequence_length"].to_dict()

context_rows = []
for peptide_id, group in residues.groupby("peptide_ID", sort=False):
    sequence = group.iloc[0]["sequence"]
    hits = motif_hits(sequence)
    for row in group.sort_values("position").itertuples(index=False):
        residue = row.original_residue
        performed = bool(row.perturbation_performed)
        covering = hits[int(row.position)]
        original_basic = int(residue in BASIC)
        original_acidic = int(residue in ACIDIC)
        original_hydrophobic = int(residue in HYDROPHOBIC)
        original_aromatic = int(residue in AROMATIC)
        context_rows.append({
            "peptide_ID": peptide_id, "sequence": sequence,
            "true_class": row.true_class, "label": row.label,
            "analysis_group": row.analysis_group, "peptide_length": len(sequence),
            "position": int(row.position), "normalized_position": row.position / len(sequence),
            "original_residue": residue, "residue_category": category(residue),
            "original_is_alanine": bool(row.original_is_alanine),
            "perturbation_performed": performed,
            "consensus_absolute_sensitivity": row.consensus_importance,
            "rf_absolute_sensitivity": row.rf_absolute_delta_true_class_probability,
            "xgboost_absolute_sensitivity": row.xgboost_absolute_delta_true_class_probability,
            "inside_recurrent_motif": bool(covering),
            "number_of_recurrent_motif_hits": len(covering),
            "motifs_covering_position": ";".join(covering),
            "original_basic_indicator": original_basic,
            "original_acidic_indicator": original_acidic,
            "original_hydrophobic_indicator": original_hydrophobic,
            "original_aromatic_indicator": original_aromatic,
            "change_basic_indicator": (0 - original_basic) if performed else np.nan,
            "change_acidic_indicator": (0 - original_acidic) if performed else np.nan,
            "change_hydrophobic_indicator": (1 - original_hydrophobic) if performed else np.nan,
            "change_aromatic_indicator": (0 - original_aromatic) if performed else np.nan,
            "change_charge_class": f"{charge_class(residue)}_to_neutral" if performed else "not_perturbed",
            "any_prediction_flip": bool(row.any_prediction_flip),
        })

context = pd.DataFrame(context_rows)
performed = context.loc[context["perturbation_performed"]].copy()
assert len(context) == 182 and len(performed) == 151
assert context.loc[~context["perturbation_performed"],
                   ["consensus_absolute_sensitivity", "rf_absolute_sensitivity",
                    "xgboost_absolute_sensitivity", "change_basic_indicator",
                    "change_acidic_indicator", "change_hydrophobic_indicator",
                    "change_aromatic_indicator"]].isna().all().all()
assert np.isfinite(performed[["consensus_absolute_sensitivity", "rf_absolute_sensitivity",
                              "xgboost_absolute_sensitivity"]]).all().all()

category_rows = []
for group_name in (HARD_GROUP, CORRECT_GROUP):
    for residue_category in CATEGORY_ORDER:
        values = performed.loc[(performed["analysis_group"] == group_name) &
                               (performed["residue_category"] == residue_category),
                               "consensus_absolute_sensitivity"]
        category_rows.append({
            "analysis_group": group_name, "residue_category": residue_category,
            "performed_positions": len(values), "mean_consensus_sensitivity": values.mean(),
            "median_consensus_sensitivity": values.median(), "sd_consensus_sensitivity": values.std(ddof=1),
            "minimum_consensus_sensitivity": values.min(), "maximum_consensus_sensitivity": values.max(),
        })
category_summary = pd.DataFrame(category_rows)

hard_performed = performed.loc[performed["analysis_group"].eq(HARD_GROUP)].copy()
motif_rows = []
for inside, label in ((True, "Inside any recurrent motif"), (False, "Outside recurrent motifs")):
    values = hard_performed.loc[hard_performed["inside_recurrent_motif"].eq(inside),
                                "consensus_absolute_sensitivity"]
    motif_rows.append({
        "summary_type": "any_motif_context", "motif_or_context": label,
        "performed_positions": len(values), "peptides_represented": hard_performed.loc[
            hard_performed["inside_recurrent_motif"].eq(inside), "peptide_ID"].nunique(),
        "mean_consensus_sensitivity": values.mean(), "median_consensus_sensitivity": values.median(),
        "sd_consensus_sensitivity": values.std(ddof=1), "maximum_consensus_sensitivity": values.max(),
    })
for motif in MOTIFS:
    mask = hard_performed["motifs_covering_position"].fillna("").str.split(";").map(lambda xs: motif in xs)
    values = hard_performed.loc[mask, "consensus_absolute_sensitivity"]
    motif_rows.append({
        "summary_type": "individual_motif_covered_positions", "motif_or_context": motif,
        "performed_positions": len(values), "peptides_represented": hard_performed.loc[mask, "peptide_ID"].nunique(),
        "mean_consensus_sensitivity": values.mean(), "median_consensus_sensitivity": values.median(),
        "sd_consensus_sensitivity": values.std(ddof=1), "maximum_consensus_sensitivity": values.max(),
    })
motif_summary = pd.DataFrame(motif_rows)

top_rows = []
for peptide_id in HARD_IDS:
    peptide = hard_performed.loc[hard_performed["peptide_ID"].eq(peptide_id)].sort_values(
        ["consensus_absolute_sensitivity", "position"], ascending=[False, True]).head(3).copy()
    frac_basic = peptide["residue_category"].eq("Basic").mean()
    frac_hydrophobic = peptide["residue_category"].eq("Hydrophobic").mean()
    frac_motif = peptide["inside_recurrent_motif"].mean()
    for rank, row in enumerate(peptide.itertuples(index=False), start=1):
        top_rows.append({
            "peptide_ID": peptide_id, "sequence": row.sequence, "true_class": row.true_class,
            "top3_rank": rank, "position": row.position, "original_residue": row.original_residue,
            "residue_category": row.residue_category,
            "consensus_absolute_sensitivity": row.consensus_absolute_sensitivity,
            "inside_recurrent_motif": row.inside_recurrent_motif,
            "number_of_recurrent_motif_hits": row.number_of_recurrent_motif_hits,
            "motifs_covering_position": row.motifs_covering_position,
            "top3_fraction_basic": frac_basic, "top3_fraction_hydrophobic": frac_hydrophobic,
            "top3_fraction_inside_recurrent_motif": frac_motif,
        })
top_context = pd.DataFrame(top_rows)

context.to_csv(CONTEXT_OUTPUT, index=False)
category_summary.to_csv(CATEGORY_OUTPUT, index=False)
motif_summary.to_csv(MOTIF_OUTPUT, index=False)
top_context.to_csv(TOP_OUTPUT, index=False)

# Figure 1: individual performed positions and category medians, same scale in both panels.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
                     "axes.titlesize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True, facecolor="white")
rng = np.random.default_rng(2026)
for ax, group_name, panel, color, title in (
    (axes[0], HARD_GROUP, "A", "#D55E00", "Consensus hard errors"),
    (axes[1], CORRECT_GROUP, "B", "#0072B2", "High-confidence consensus correct"),
):
    subset = performed.loc[performed["analysis_group"].eq(group_name)]
    for x, residue_category in enumerate(CATEGORY_ORDER):
        values = subset.loc[subset["residue_category"].eq(residue_category),
                            "consensus_absolute_sensitivity"].to_numpy(float)
        jitter = rng.uniform(-0.20, 0.20, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=25, color=color,
                   alpha=0.68, edgecolor="black", linewidth=0.35, zorder=2)
        if len(values):
            ax.hlines(np.median(values), x - 0.28, x + 0.28, color="black", linewidth=2.2, zorder=3)
    ax.set_xticks(range(4), CATEGORY_ORDER, rotation=20, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.04, panel, transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.set_facecolor("white")
axes[0].set_ylabel("Consensus absolute sensitivity\nmean RF/XGBoost |Δ true-class probability|")
fig.suptitle("Residue-category sensitivity in the predefined Step-82 panel",
             fontsize=13, fontweight="bold", y=0.99)
fig.text(0.5, 0.012, "Points are performed non-alanine substitutions; black lines show category medians. Descriptive only.",
         ha="center", fontsize=8.3)
fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.94), w_pad=2.2)
fig.savefig(FIG1_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIG1_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: hard-case tracks; teal underline denotes recurrent motif coverage.
hard_context = context.loc[context["analysis_group"].eq(HARD_GROUP)]
vmax = float(hard_context["consensus_absolute_sensitivity"].max())
fig, axes = plt.subplots(5, 1, figsize=(12.8, 6.9), facecolor="white")
for ax, peptide_id in zip(axes, HARD_IDS):
    row = hard_context.loc[hard_context["peptide_ID"].eq(peptide_id)].sort_values("position")
    for j, item in enumerate(row.itertuples(index=False)):
        value = item.consensus_absolute_sensitivity if item.perturbation_performed else 0.0
        face = plt.cm.YlOrRd(value / vmax if vmax else 0)
        hatch = "///" if item.original_is_alanine else None
        ax.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=face, edgecolor="black",
                                   linewidth=0.65, hatch=hatch))
        if item.inside_recurrent_motif:
            ax.plot([j + 0.08, j + 0.92], [0.08, 0.08], color="#009E73", linewidth=3.0,
                    solid_capstyle="butt")
        ax.text(j + 0.5, 0.55, item.original_residue, ha="center", va="center", fontsize=8.6,
                fontweight="bold" if item.inside_recurrent_motif else "normal")
    ax.set_xlim(0, len(row)); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xticks(np.arange(len(row)) + 0.5, row["position"].astype(str))
    ax.set_ylabel(f"ID {peptide_id}\n{row.iloc[0]['true_class']}", rotation=0,
                  ha="right", va="center", labelpad=34)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_facecolor("white")
axes[-1].set_xlabel("Residue position")
fig.suptitle("Hard-case sensitivity within recurrent basic/hydrophobic motif context",
             fontsize=13, fontweight="bold", y=0.99)
cax = fig.add_axes([0.91, 0.20, 0.014, 0.60])
fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="YlOrRd"), cax=cax,
             label="Consensus absolute sensitivity")
fig.text(0.5, 0.012,
         "Teal underline/bold residue: covered by ≥1 predefined recurrent motif; hatched A: not perturbed.",
         ha="center", fontsize=8.3)
fig.subplots_adjust(left=0.10, right=0.89, top=0.91, bottom=0.09, hspace=0.75)
fig.savefig(FIG2_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIG2_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

motif_inside = motif_summary.loc[motif_summary["motif_or_context"].eq("Inside any recurrent motif")].iloc[0]
motif_outside = motif_summary.loc[motif_summary["motif_or_context"].eq("Outside recurrent motifs")].iloc[0]
qc = pd.DataFrame([{
    "selected_peptides": selected.shape[0], "hard_peptides": int((selected["analysis_group"] == HARD_GROUP).sum()),
    "correct_control_peptides": int((selected["analysis_group"] == CORRECT_GROUP).sum()),
    "total_residue_positions": len(context), "performed_nonalanine_positions": len(performed),
    "alanine_positions_retained_qc": int((~context["perturbation_performed"]).sum()),
    "alanine_positions_excluded_sensitivity_comparisons": True,
    "fixed_recurrent_motifs": len(MOTIFS), "all_fixed_motifs_present_step60": True,
    "overlapping_motif_matching_used": True,
    "hard_performed_positions_inside_motifs": int(motif_inside["performed_positions"]),
    "hard_performed_positions_outside_motifs": int(motif_outside["performed_positions"]),
    "context_rows": len(context), "category_summary_rows": len(category_summary),
    "motif_summary_rows": len(motif_summary), "top_context_rows": len(top_context),
    "all_sensitivity_values_finite": bool(np.isfinite(performed["consensus_absolute_sensitivity"]).all()),
    "normalized_positions_in_0_1": bool(context["normalized_position"].between(0, 1).all()),
    "category_annotation_complete": bool(context["residue_category"].isin(CATEGORY_ORDER).all()),
    "no_new_esm2_inference": True, "no_classifier_loaded": True,
    "no_model_training": True, "no_significance_testing": True,
    "no_feature_selection": True, "qc_passed": True,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nCategory summary:")
print(category_summary.round(6).to_string(index=False))
print("\nHard-case motif context:")
print(motif_summary.head(2).round(6).to_string(index=False))
print("\nTop-three context fractions by hard peptide:")
print(top_context.groupby("peptide_ID")[["top3_fraction_basic", "top3_fraction_hydrophobic",
                                          "top3_fraction_inside_recurrent_motif"]].first().to_string())
print("\nOutputs:")
for path in (CONTEXT_OUTPUT, CATEGORY_OUTPUT, MOTIF_OUTPUT, TOP_OUTPUT, QC_OUTPUT,
             FIG1_PNG, FIG1_PDF, FIG2_PNG, FIG2_PDF):
    print(path, path.stat().st_size, "bytes", sha256(path))
print("\nSTEP 83 COMPLETED SUCCESSFULLY")
print("=" * 104)

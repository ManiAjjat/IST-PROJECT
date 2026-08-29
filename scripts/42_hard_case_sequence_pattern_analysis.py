from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
RANKING_FILE = RESULTS_DIR / "step59_consensus_hard_case_ranking.csv"

COMPOSITION_OUTPUT = RESULTS_DIR / "step60_hard_case_amino_acid_composition.csv"
GROUP_OUTPUT = RESULTS_DIR / "step60_group_amino_acid_summary.csv"
MOTIF_OUTPUT = RESULTS_DIR / "step60_hard_case_motif_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step60_sequence_pattern_qc.csv"
COMPOSITION_PNG = FIGURES_DIR / "Step60_Hard_Case_Amino_Acid_Composition.png"
COMPOSITION_PDF = FIGURES_DIR / "Step60_Hard_Case_Amino_Acid_Composition.pdf"
MOTIF_PNG = FIGURES_DIR / "Step60_Hard_Case_Motif_Enrichment.png"
MOTIF_PDF = FIGURES_DIR / "Step60_Hard_Case_Motif_Enrichment.pdf"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
RESIDUE_SETS = {
    "basic_KRH": set("KRH"),
    "acidic_DE": set("DE"),
    "aromatic_FWY": set("FWY"),
    "hydrophobic_AVILMFWY": set("AVILMFWY"),
    "small_AGST": set("AGST"),
    "proline_glycine_PG": set("PG"),
    "strong_basic_KR": set("KR"),
}
HARD_LABEL = "Hard cases (>=3/8 wrong)"
REFERENCE_LABEL = "Generally well classified"


def longest_run(sequence, residue_set):
    longest = current = 0
    for residue in sequence:
        current = current + 1 if residue in residue_set else 0
        longest = max(longest, current)
    return longest


def maximum_window_fraction(sequence, residue_set, window_size=5):
    width = min(window_size, len(sequence))
    return max(
        sum(residue in residue_set for residue in sequence[start : start + width]) / width
        for start in range(len(sequence) - width + 1)
    )


def repeated_ngram_count(sequence):
    repeated = 0
    for n in (2, 3):
        counts = Counter(sequence[start : start + n] for start in range(len(sequence) - n + 1))
        repeated += sum(count - 1 for count in counts.values() if count > 1)
    return repeated


def sequence_features(sequence):
    length = len(sequence)
    counts = Counter(sequence)
    row = {"sequence_length": length}
    for amino_acid in AMINO_ACIDS:
        row[f"fraction_{amino_acid}"] = counts[amino_acid] / length
    for name, residue_set in RESIDUE_SETS.items():
        row[name] = sum(counts[residue] for residue in residue_set) / length
    row["KR_minus_DE_fraction"] = row["strong_basic_KR"] - row["acidic_DE"]
    row["K_minus_R_fraction"] = row["fraction_K"] - row["fraction_R"]
    row["max_KR_run"] = longest_run(sequence, set("KR"))
    row["max_KRH_run"] = longest_run(sequence, set("KRH"))
    row["max_KR_fraction_window5"] = maximum_window_fraction(sequence, set("KR"))
    row["max_hydrophobic_fraction_window5"] = maximum_window_fraction(
        sequence, set("AVILMFWY")
    )
    states = [
        "B" if residue in set("KR") else "H" if residue in set("AVILMFWY") else "O"
        for residue in sequence
    ]
    row["basic_hydrophobic_transition_fraction"] = (
        sum({left, right} == {"B", "H"} for left, right in zip(states, states[1:]))
        / max(length - 1, 1)
    )
    row["repeated_di_tripeptide_excess_count"] = repeated_ngram_count(sequence)
    return row


def motif_counters(frame, n):
    occurrence = Counter()
    presence = Counter()
    for sequence in frame["sequence"]:
        motifs = [sequence[start : start + n] for start in range(len(sequence) - n + 1)]
        occurrence.update(motifs)
        presence.update(set(motifs))
    return occurrence, presence


print("=" * 108)
print("STEP 60 - HARD-CASE SEQUENCE PATTERN ANALYSIS")
print("=" * 108)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ranking = pd.read_csv(RANKING_FILE)
required = {"rank", "ID", "sequence", "true_class", "y_true", "total_wrong_count", "difficulty_category"}
assert required.issubset(ranking.columns)
assert len(ranking) == 181 and ranking["ID"].is_unique
assert ranking["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()

hard = ranking.loc[ranking["total_wrong_count"] >= 3].copy()
reference = ranking.loc[ranking["difficulty_category"] == "Generally well classified"].copy()
assert len(hard) == 15 and len(reference) == 166
assert set(hard["ID"]).isdisjoint(reference["ID"])
assert set(hard["ID"]) | set(reference["ID"]) == set(ranking["ID"])

feature_rows = []
for row in ranking.itertuples(index=False):
    features = sequence_features(row.sequence)
    features.update({
        "ID": int(row.ID), "rank": int(row.rank), "sequence": row.sequence,
        "true_class": row.true_class, "y_true": int(row.y_true),
        "total_wrong_count": int(row.total_wrong_count),
        "analysis_group": HARD_LABEL if row.total_wrong_count >= 3 else REFERENCE_LABEL,
    })
    feature_rows.append(features)
features = pd.DataFrame(feature_rows)

identity_columns = [
    "rank", "ID", "sequence", "true_class", "y_true", "total_wrong_count", "analysis_group"
]
metric_columns = [column for column in features.columns if column not in identity_columns]
features = features[identity_columns + metric_columns]
hard_features = features.loc[features["analysis_group"] == HARD_LABEL].copy()
reference_features = features.loc[features["analysis_group"] == REFERENCE_LABEL].copy()
hard_features.to_csv(COMPOSITION_OUTPUT, index=False)

summary_rows = []
for metric in metric_columns:
    hard_values = hard_features[metric].astype(float)
    reference_values = reference_features[metric].astype(float)
    summary_rows.append({
        "metric": metric,
        "hard_n": len(hard_values),
        "hard_mean": hard_values.mean(),
        "hard_sd": hard_values.std(ddof=1),
        "hard_median": hard_values.median(),
        "hard_min": hard_values.min(),
        "hard_max": hard_values.max(),
        "reference_n": len(reference_values),
        "reference_mean": reference_values.mean(),
        "reference_sd": reference_values.std(ddof=1),
        "reference_median": reference_values.median(),
        "reference_min": reference_values.min(),
        "reference_max": reference_values.max(),
        "hard_minus_reference_mean": hard_values.mean() - reference_values.mean(),
        "hard_to_reference_mean_ratio": (
            hard_values.mean() / reference_values.mean() if reference_values.mean() != 0 else np.nan
        ),
    })
group_summary = pd.DataFrame(summary_rows)
group_summary.to_csv(GROUP_OUTPUT, index=False)

motif_rows = []
for n in (2, 3):
    hard_occurrence, hard_presence = motif_counters(hard, n)
    reference_occurrence, reference_presence = motif_counters(reference, n)
    all_motifs = sorted(set(hard_occurrence) | set(reference_occurrence))
    hard_total_windows = sum(max(len(sequence) - n + 1, 0) for sequence in hard["sequence"])
    reference_total_windows = sum(
        max(len(sequence) - n + 1, 0) for sequence in reference["sequence"]
    )
    for motif in all_motifs:
        hard_prevalence = hard_presence[motif] / len(hard)
        reference_prevalence = reference_presence[motif] / len(reference)
        motif_rows.append({
            "n": n,
            "motif": motif,
            "hard_peptides_with_motif": hard_presence[motif],
            "hard_peptide_prevalence": hard_prevalence,
            "reference_peptides_with_motif": reference_presence[motif],
            "reference_peptide_prevalence": reference_prevalence,
            "prevalence_difference": hard_prevalence - reference_prevalence,
            "descriptive_prevalence_ratio_smoothed": (
                (hard_presence[motif] + 0.5) / (len(hard) + 1.0)
            ) / ((reference_presence[motif] + 0.5) / (len(reference) + 1.0)),
            "hard_occurrences": hard_occurrence[motif],
            "hard_occurrences_per_100_windows": 100 * hard_occurrence[motif] / hard_total_windows,
            "reference_occurrences": reference_occurrence[motif],
            "reference_occurrences_per_100_windows": (
                100 * reference_occurrence[motif] / reference_total_windows
            ),
        })
motifs = pd.DataFrame(motif_rows).sort_values(
    ["hard_peptides_with_motif", "prevalence_difference", "n", "motif"],
    ascending=[False, False, True, True],
).reset_index(drop=True)
motifs.insert(0, "descriptive_rank", np.arange(1, len(motifs) + 1))
motifs.to_csv(MOTIF_OUTPUT, index=False)

aa_metrics = [f"fraction_{amino_acid}" for amino_acid in AMINO_ACIDS]
aa_summary = group_summary.set_index("metric").loc[aa_metrics]
hard_aa = aa_summary["hard_mean"].to_numpy() * 100
reference_aa = aa_summary["reference_mean"].to_numpy() * 100
x = np.arange(len(AMINO_ACIDS))
width = 0.39
fig, ax = plt.subplots(figsize=(13.2, 6.8), facecolor="white")
ax.bar(x - width / 2, hard_aa, width, color="#D95F02", label=f"Hard cases (n={len(hard)})")
ax.bar(
    x + width / 2, reference_aa, width, color="#1B9E77",
    label=f"Generally well classified (n={len(reference)})",
)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_xticks(x, AMINO_ACIDS)
ax.set_ylabel("Mean amino-acid fraction (%)")
ax.set_xlabel("Amino acid")
ax.set_title("Amino-acid composition of consensus hard cases and reference peptides")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(COMPOSITION_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(COMPOSITION_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

plot_motifs = motifs.loc[motifs["hard_peptides_with_motif"] >= 2].nlargest(
    18, ["prevalence_difference", "hard_peptides_with_motif"]
).sort_values("prevalence_difference")
assert len(plot_motifs) > 0
y = np.arange(len(plot_motifs))
fig, ax = plt.subplots(figsize=(10.8, 8.2), facecolor="white")
ax.barh(
    y, plot_motifs["prevalence_difference"] * 100,
    color=["#D95F02" if value >= 0 else "#1B9E77" for value in plot_motifs["prevalence_difference"]],
)
ax.axvline(0, color="#333333", linewidth=1.0)
ax.set_yticks(y, plot_motifs["motif"])
ax.set_xlabel("Hard-case minus reference peptide prevalence (percentage points)")
ax.set_ylabel("Di-/tripeptide motif")
ax.set_title("Recurrent short sequence patterns in consensus hard cases")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
for position, value in zip(y, plot_motifs["prevalence_difference"] * 100):
    ax.text(
        value + (0.45 if value >= 0 else -0.45), position, f"{value:+.1f}",
        va="center", ha="left" if value >= 0 else "right", fontsize=8.5,
    )
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(MOTIF_PNG, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(MOTIF_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "test_peptides": len(ranking),
    "hard_case_peptides": len(hard_features),
    "reference_peptides": len(reference_features),
    "group_overlap": len(set(hard_features["ID"]) & set(reference_features["ID"])),
    "group_union": len(set(hard_features["ID"]) | set(reference_features["ID"])),
    "unique_ids": features["ID"].nunique(),
    "valid_amino_acid_sequences": bool(ranking["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()),
    "hard_output_rows": len(hard_features),
    "group_summary_rows": len(group_summary),
    "motif_summary_rows": len(motifs),
    "all_composition_values_finite": bool(np.isfinite(features[metric_columns].to_numpy(float)).all()),
    "all_aa_fractions_sum_to_one": bool(np.allclose(features[aa_metrics].sum(axis=1), 1.0)),
    "motif_prevalences_within_0_1": bool(
        motifs[["hard_peptide_prevalence", "reference_peptide_prevalence"]].ge(0).all().all()
        and motifs[["hard_peptide_prevalence", "reference_peptide_prevalence"]].le(1).all().all()
    ),
    "hard_threshold_applied": bool(hard_features["total_wrong_count"].ge(3).all()),
    "reference_definition_applied": bool(
        reference["difficulty_category"].eq("Generally well classified").all()
    ),
    "models_retrained": False,
    "model_selection_performed": False,
    "inferential_motif_tests_performed": False,
}])
qc.to_csv(QC_OUTPUT, index=False)

focus_metrics = [
    "fraction_K", "fraction_R", "fraction_L", "fraction_F", "fraction_A",
    "strong_basic_KR", "KR_minus_DE_fraction", "max_KR_run",
    "max_KR_fraction_window5", "hydrophobic_AVILMFWY",
    "max_hydrophobic_fraction_window5",
]
print("\nGroup comparison for requested focus metrics:")
print(
    group_summary.set_index("metric").loc[focus_metrics, [
        "hard_mean", "reference_mean", "hard_minus_reference_mean"
    ]].round(6).to_string()
)
print("\nTop recurrent motifs (present in at least two hard cases):")
print(
    motifs.loc[motifs["hard_peptides_with_motif"] >= 2, [
        "motif", "hard_peptides_with_motif", "hard_peptide_prevalence",
        "reference_peptide_prevalence", "prevalence_difference"
    ]].head(15).round(6).to_string(index=False)
)
print("\nOutput checks:")
for path in (
    COMPOSITION_OUTPUT, GROUP_OUTPUT, MOTIF_OUTPUT, QC_OUTPUT,
    COMPOSITION_PNG, COMPOSITION_PDF, MOTIF_PNG, MOTIF_PDF,
):
    print(f"  {path.name}: {path.exists()}")
print("\nSTEP 60 COMPLETED SUCCESSFULLY")
print("=" * 108)

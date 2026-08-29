from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
MANUSCRIPT_FIGURES_DIR = PROJECT_DIR / "manuscript" / "figures"
SUPPLEMENTARY_FIGURES_DIR = PROJECT_DIR / "manuscript" / "supplementary_figures"

MAIN_OUTPUT = RESULTS_DIR / "step86_main_figure_selection.csv"
SUPPLEMENT_OUTPUT = RESULTS_DIR / "step86_supplementary_figure_manifest.csv"
QC_OUTPUT = RESULTS_DIR / "step86_figure_organization_qc.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENTARY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

main_specs = [
    {
        "figure_number": "Figure 1",
        "manuscript_title": "Comparative predictive performance of traditional descriptor- and ESM-2-based classifiers on the locked test set.",
        "source_step": "Step 74",
        "source_stem": "Step74_Model_Performance_Confidence_Intervals",
        "destination_stem": "Figure1_Model_Performance",
        "primary_message": "Primary locked-test AUROC, AUPRC, MCC, and F1 estimates with stratified-bootstrap 95% confidence intervals for all eight frozen models.",
        "results_section": "Predictive performance of traditional and ESM-2 classifiers",
    },
    {
        "figure_number": "Figure 2",
        "manuscript_title": "Predictive discrimination across predefined sequence-novelty subsets of the locked test set.",
        "source_step": "Step 67",
        "source_stem": "Step67_Sequence_Novelty_AUROC_AUPRC",
        "destination_stem": "Figure2_Sequence_Novelty_Performance",
        "primary_message": "AUROC and AUPRC remain high after progressively excluding test peptides with close development-set sequence analogues; captions must state the changing subset sizes and class prevalence.",
        "results_section": "Sequence-similarity sensitivity and generalization",
    },
    {
        "figure_number": "Figure 3",
        "manuscript_title": "Integrated sequence, neighborhood, and residue-level evidence underlying universal prediction failures.",
        "source_step": "Step 84",
        "source_stem": "Step84_Universal_Hard_Case_Evidence_Map",
        "destination_stem": "Figure3_Universal_Hard_Case_Interpretability",
        "primary_message": "Integrated descriptive evidence for the five universal 8/8 errors; signals indicate model behavior and do not establish incorrect labels or biochemical causality.",
        "results_section": "Consensus hard-case and residue-level interpretation",
    },
]

main_rows: list[dict] = []
qc_rows: list[dict] = []

for spec in main_specs:
    source_png = FIGURES_DIR / f"{spec['source_stem']}.png"
    source_pdf = FIGURES_DIR / f"{spec['source_stem']}.pdf"
    destination_png = MANUSCRIPT_FIGURES_DIR / f"{spec['destination_stem']}.png"
    destination_pdf = MANUSCRIPT_FIGURES_DIR / f"{spec['destination_stem']}.pdf"

    for path in (source_png, source_pdf):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source figure: {path}")

    main_rows.append(
        {
            "figure_number": spec["figure_number"],
            "manuscript_title": spec["manuscript_title"],
            "source_step": spec["source_step"],
            "source_png": str(source_png.relative_to(PROJECT_DIR)),
            "source_pdf": str(source_pdf.relative_to(PROJECT_DIR)),
            "destination_png": str(destination_png.relative_to(PROJECT_DIR)),
            "destination_pdf": str(destination_pdf.relative_to(PROJECT_DIR)),
            "primary_message": spec["primary_message"],
            "results_section": spec["results_section"],
        }
    )

    for file_format, source, destination in (
        ("PNG", source_png, destination_png),
        ("PDF", source_pdf, destination_pdf),
    ):
        source_hash_before = sha256(source)
        source_size_before = source.stat().st_size
        source_mtime_before = source.stat().st_mtime_ns
        shutil.copy2(source, destination)
        source_hash_after = sha256(source)
        destination_hash = sha256(destination)
        qc_rows.append(
            {
                "figure_number": spec["figure_number"],
                "source_step": spec["source_step"],
                "file_format": file_format,
                "source_file": str(source.relative_to(PROJECT_DIR)),
                "destination_file": str(destination.relative_to(PROJECT_DIR)),
                "source_exists": source.is_file(),
                "destination_exists": destination.is_file(),
                "source_size_bytes": source_size_before,
                "destination_size_bytes": destination.stat().st_size,
                "source_sha256_before_copy": source_hash_before,
                "source_sha256_after_copy": source_hash_after,
                "destination_sha256": destination_hash,
                "source_unchanged": source_hash_before == source_hash_after
                and source_size_before == source.stat().st_size
                and source_mtime_before == source.stat().st_mtime_ns,
                "source_destination_hash_match": source_hash_before == destination_hash,
                "figure_regenerated": False,
                "data_recalculated": False,
                "labels_changed": False,
                "scientific_results_changed": False,
            }
        )

supplement_rows = [
    {"supplementary_number": "S1", "source_step": "Step 36", "source_figure": "Step36_Traditional_Model_ROC; Step36_Traditional_Model_PR", "analysis_topic": "Traditional-model ROC and precision-recall curves", "recommended_status": "Include", "reason": "Provides curve-level detail supporting the traditional-model performance results."},
    {"supplementary_number": "S2", "source_step": "Step 39", "source_figure": "Step39_XGBoost_Permutation_Importance", "analysis_topic": "Traditional XGBoost permutation importance", "recommended_status": "Include", "reason": "Supports interpretation of the strongest traditional classifier without displacing primary performance evidence."},
    {"supplementary_number": "S3", "source_step": "Step 52", "source_figure": "Step52_ESM2_Model_ROC; Step52_ESM2_Model_PR", "analysis_topic": "ESM-2 model ROC and precision-recall curves", "recommended_status": "Include", "reason": "Provides curve-level detail for the four frozen ESM-2 classifiers."},
    {"supplementary_number": "S4", "source_step": "Step 55", "source_figure": "Step55_Traditional_vs_ESM2_Bootstrap_CI", "analysis_topic": "Matched traditional-versus-ESM-2 paired-bootstrap differences", "recommended_status": "Include", "reason": "Directly supports the matched representation comparisons summarized in Table 3."},
    {"supplementary_number": "S5", "source_step": "Step 60", "source_figure": "Step60_Hard_Case_Amino_Acid_Composition; Step60_Hard_Case_Motif_Enrichment", "analysis_topic": "Hard-case amino-acid composition and recurrent motifs", "recommended_status": "Include", "reason": "Provides sequence-level context underlying the integrated hard-case evidence."},
    {"supplementary_number": "S6", "source_step": "Step 61", "source_figure": "Step61_Hard_Case_Opposite_Class_Proximity", "analysis_topic": "Opposite-class nearest-neighbor proximity", "recommended_status": "Include", "reason": "Shows descriptive cross-class proximity among difficult peptides."},
    {"supplementary_number": "S7", "source_step": "Step 62", "source_figure": "Step62_Hard_Case_Sequence_Similarity_Map", "analysis_topic": "Within-hard-case sequence similarity", "recommended_status": "Include", "reason": "Documents family concentration and diversity within the hard-case set."},
    {"supplementary_number": "S8", "source_step": "Step 66", "source_figure": "Step66_Test_to_Development_Similarity", "analysis_topic": "Development-test sequence homology audit", "recommended_status": "Include", "reason": "Documents the absence of exact overlap and the distribution of nearest-development similarity."},
    {"supplementary_number": "S9", "source_step": "Step 67", "source_figure": "Step67_Sequence_Novelty_MCC_F1", "analysis_topic": "Threshold performance across sequence-novelty subsets", "recommended_status": "Include", "reason": "Complements main Figure 2, with explicit caution because the strictest subset contains only four Active peptides."},
    {"supplementary_number": "S10", "source_step": "Step 71", "source_figure": "Step71_Model_Calibration_Curves", "analysis_topic": "Probability calibration and reliability", "recommended_status": "Include", "reason": "Adds probability-reliability evidence not captured by discrimination metrics."},
    {"supplementary_number": "S11", "source_step": "Step 73", "source_figure": "Step73_Traditional_Decision_Curves; Step73_ESM2_Decision_Curves", "analysis_topic": "Decision-curve net benefit", "recommended_status": "Include", "reason": "Provides threshold-dependent clinical-utility context for both representation branches."},
    {"supplementary_number": "S12", "source_step": "Step 77", "source_figure": "Step77_Traditional_vs_ESM2_Canonical_Correlation; Step77_Descriptor_ESM2_PC_Association_Map", "analysis_topic": "In-sample feature-space complementarity", "recommended_status": "Optional", "reason": "Useful representation analysis, but Step 78 provides the stronger held-out validation perspective."},
    {"supplementary_number": "S13", "source_step": "Step 78", "source_figure": "Step78_Cross_Validated_Canonical_Correlations", "analysis_topic": "Cross-validated feature-space complementarity", "recommended_status": "Include", "reason": "Evaluates whether traditional-ESM-2 alignment persists on held-out development folds."},
    {"supplementary_number": "S14", "source_step": "Step 81", "source_figure": "Step81_ESM2_Feature_Importance_Stability", "analysis_topic": "Stable latent-dimension usage across CV folds", "recommended_status": "Optional", "reason": "Adds computational interpretability, while latent dimensions lack direct biological meaning."},
    {"supplementary_number": "S15", "source_step": "Step 82", "source_figure": "Step82_Hard_Case_Residue_Sensitivity", "analysis_topic": "Residue-level alanine perturbation sensitivity", "recommended_status": "Include", "reason": "Provides the residue-level evidence summarized in main Figure 3."},
    {"supplementary_number": "S16", "source_step": "Step 83", "source_figure": "Step83_Motif_Context_Sensitivity", "analysis_topic": "Motif context of residue sensitivity", "recommended_status": "Include", "reason": "Connects recurrent hard-case motifs to prediction-sensitive positions without implying mechanism."},
    {"supplementary_number": "S17", "source_step": "Step 76", "source_figure": "Step76_Integrated_Model_Performance_Map", "analysis_topic": "Descriptive multi-domain model ranking", "recommended_status": "Archive only", "reason": "The mean domain rank is descriptive and could be mistaken for a formal model-selection criterion."},
    {"supplementary_number": "S18", "source_step": "Step 79", "source_figure": "Step79_Fusion_vs_Single_Representation", "analysis_topic": "Feature-fusion point-estimate comparison", "recommended_status": "Optional", "reason": "The controlled negative fusion result is useful but not central to the three-figure manuscript narrative."},
    {"supplementary_number": "S19", "source_step": "Step 80", "source_figure": "Step80_Fusion_vs_ESM2_Paired_Bootstrap", "analysis_topic": "Paired fusion-versus-ESM-2 uncertainty", "recommended_status": "Optional", "reason": "Supports a concise negative-result paragraph if supplementary space permits."},
]

write_csv(
    MAIN_OUTPUT,
    main_rows,
    ["figure_number", "manuscript_title", "source_step", "source_png", "source_pdf", "destination_png", "destination_pdf", "primary_message", "results_section"],
)
write_csv(
    SUPPLEMENT_OUTPUT,
    supplement_rows,
    ["supplementary_number", "source_step", "source_figure", "analysis_topic", "recommended_status", "reason"],
)
write_csv(
    QC_OUTPUT,
    qc_rows,
    ["figure_number", "source_step", "file_format", "source_file", "destination_file", "source_exists", "destination_exists", "source_size_bytes", "destination_size_bytes", "source_sha256_before_copy", "source_sha256_after_copy", "destination_sha256", "source_unchanged", "source_destination_hash_match", "figure_regenerated", "data_recalculated", "labels_changed", "scientific_results_changed"],
)

assert len(main_rows) == 3
assert [row["source_step"] for row in main_rows] == ["Step 74", "Step 67", "Step 84"]
assert len(qc_rows) == 6
assert all(row["source_exists"] and row["destination_exists"] for row in qc_rows)
assert all(row["source_unchanged"] and row["source_destination_hash_match"] for row in qc_rows)
assert all(not row[key] for row in qc_rows for key in ("figure_regenerated", "data_recalculated", "labels_changed", "scientific_results_changed"))
assert {row["recommended_status"] for row in supplement_rows} <= {"Include", "Optional", "Archive only"}

print("=" * 92)
print("STEP 86 - FINAL MAIN-FIGURE SELECTION AND MANUSCRIPT ORGANIZATION")
print("=" * 92)
print(f"Main figures selected: {len(main_rows)}")
print(f"Copied files verified byte-for-byte: {sum(row['source_destination_hash_match'] for row in qc_rows)}/6")
print(f"Supplementary manifest entries: {len(supplement_rows)}")
print(f"Main-figure directory: {MANUSCRIPT_FIGURES_DIR}")
print(f"Supplementary directory: {SUPPLEMENTARY_FIGURES_DIR}")
print(f"Main selection: {MAIN_OUTPUT}")
print(f"Supplementary manifest: {SUPPLEMENT_OUTPUT}")
print(f"QC: {QC_OUTPUT}")
print("STEP 86 COMPLETED SUCCESSFULLY")
print("=" * 92)

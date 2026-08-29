from pathlib import Path
import hashlib
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(r"E:\postdoc-work\ist-project")
SOURCE_DIR = PROJECT / "figures"
OUTPUT_DIR = PROJECT / "manuscript" / "main_figures_from_existing"
RESULTS_DIR = PROJECT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE_OUTPUT = RESULTS_DIR / "step86c_existing_figure_main_architecture.csv"
PANEL_OUTPUT = RESULTS_DIR / "step86c_existing_figure_panel_manifest.csv"
COMPOSITE_QC_OUTPUT = RESULTS_DIR / "step86c_existing_figure_composite_qc.csv"
SOURCE_QC_OUTPUT = RESULTS_DIR / "step86c_existing_source_integrity_qc.csv"

FIGURES = [
    (
        1,
        "Dataset and representation landscape",
        "Figure1_Dataset_and_Representation_Landscape",
        [
            ("Step25_Physicochemical_Distributions_Annotated.png", "Physicochemical landscape"),
            ("Step26_Spearman_Correlation_Heatmap.png", "Descriptor correlation structure"),
            ("Step46_ESM2_Dimension_SD_Distribution.png", "ESM-2 dimension variability"),
            ("Step46_ESM2_PCA_Cumulative_Variance.png", "ESM-2 cumulative variance"),
            ("Step66_Sequence_Family_Redundancy.png", "Sequence-family redundancy audit"),
            ("Step66_Test_to_Development_Similarity.png", "Test-to-development similarity"),
        ],
    ),
    (
        2,
        "Traditional and ESM-2 predictive performance",
        "Figure2_Traditional_and_ESM2_Performance",
        [
            ("Step36_Traditional_Model_ROC.png", "Traditional ROC curves"),
            ("Step36_Traditional_Model_PR.png", "Traditional precision-recall curves"),
            ("Step37_Traditional_Model_Metric_Comparison.png", "Traditional metric comparison"),
            ("Step52_ESM2_Model_ROC.png", "ESM-2 ROC curves"),
            ("Step52_ESM2_Model_PR.png", "ESM-2 precision-recall curves"),
            ("Step52_ESM2_Model_Metric_Comparison.png", "ESM-2 metric comparison"),
            ("Step74_Model_Performance_Confidence_Intervals.png", "Bootstrap performance uncertainty"),
        ],
    ),
    (
        3,
        "Matched improvements, calibration, and decision utility",
        "Figure3_Matched_Performance_Calibration_and_Utility",
        [
            ("Step53_Traditional_vs_ESM2_Metric_Deltas.png", "Matched representation deltas"),
            ("Step55_Traditional_vs_ESM2_Bootstrap_CI.png", "Paired bootstrap intervals"),
            ("Step71_Model_Calibration_Curves.png", "Reliability curves"),
            ("Step71_Calibration_Metrics.png", "Calibration metrics"),
            ("Step72_Traditional_vs_ESM2_Calibration_Deltas.png", "Matched calibration deltas"),
            ("Step73_Traditional_Decision_Curves.png", "Traditional decision curves"),
            ("Step73_ESM2_Decision_Curves.png", "ESM-2 decision curves"),
            ("Step75_ESM2_RF_vs_XGBoost_Paired_Bootstrap.png", "Leading ESM-2 tree-model comparison"),
        ],
    ),
    (
        4,
        "Sequence novelty and homology-aware generalization",
        "Figure4_Sequence_Novelty_and_Generalization",
        [
            ("Step67_Sequence_Novelty_AUROC_AUPRC.png", "Novelty-stratified ranking metrics"),
            ("Step67_Sequence_Novelty_MCC_F1.png", "Novelty-stratified threshold metrics"),
            ("Step68_Similarity_Stratum_Error_Rates.png", "Similarity-stratum error rates"),
            ("Step68_Similarity_vs_Eight_Model_Difficulty.png", "Similarity versus consensus difficulty"),
            ("Step69_Class_Similarity_vs_Difficulty.png", "Class-aware similarity and difficulty"),
            ("Step69_Class_Stratified_Difficulty.png", "Class-stratified difficulty"),
            ("Step70_Matched_Neighbor_Relation_Differences.png", "Matched neighbor-relation differences"),
            ("Step70_Neighbor_Relation_Error_Burden.png", "Neighbor relation and error burden"),
        ],
    ),
    (
        5,
        "Representation complementarity and ESM-2 interpretability",
        "Figure5_Representation_Complementarity_and_Interpretability",
        [
            ("Step77_Traditional_vs_ESM2_Canonical_Correlation.png", "Cross-space canonical correlation"),
            ("Step77_Descriptor_ESM2_PC_Association_Map.png", "Descriptor-to-ESM-2 PC associations"),
            ("Step78_Cross_Validated_Canonical_Correlations.png", "Cross-validated shared structure"),
            ("Step79_Fusion_vs_Single_Representation.png", "Fusion versus single representations"),
            ("Step80_Fusion_vs_ESM2_Paired_Bootstrap.png", "Fusion paired-bootstrap evidence"),
            ("Step81_ESM2_Feature_Importance_Stability.png", "Stable latent-dimension importance"),
            ("Step82_Hard_vs_Correct_Residue_Sensitivity.png", "Hard-versus-correct residue sensitivity"),
            ("Step82_Hard_Case_Residue_Sensitivity.png", "Hard-case residue tracks"),
            ("Step83_Residue_Category_Sensitivity.png", "Residue-category sensitivity"),
            ("Step83_Motif_Context_Sensitivity.png", "Motif-context sensitivity"),
        ],
    ),
    (
        6,
        "Hard-case anatomy and error interpretation",
        "Figure6_Hard_Case_Anatomy_and_Error_Interpretation",
        [
            ("Step56_Traditional_Error_ESM2_Rescue_Map.png", "Traditional errors and ESM-2 rescue"),
            ("Step57_Correct_Class_Probability_Shifts.png", "Correct-class probability shifts"),
            ("Step58_Transition_Physicochemical_Profiles.png", "Prediction-transition profiles"),
            ("Step59_Eight_Model_Hard_Case_Map.png", "Eight-model hard-case map"),
            ("Step59_Traditional_ESM2_Error_Overlap.png", "Representation error overlap"),
            ("Step60_Hard_Case_Amino_Acid_Composition.png", "Hard-case amino-acid composition"),
            ("Step60_Hard_Case_Motif_Enrichment.png", "Recurrent hard-case motifs"),
            ("Step61_Hard_Case_Opposite_Class_Proximity.png", "Opposite-class proximity"),
            ("Step64_Hard_Case_Neighborhood_Purity.png", "Development-neighborhood purity"),
            ("Step84_Universal_Hard_Case_Evidence_Map.png", "Universal-error evidence synthesis"),
        ],
    ),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def get_font(size, bold=False):
    candidate = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    if candidate.exists():
        return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_text(draw, text, max_width, initial_size, bold=False, minimum=30):
    size = initial_size
    while size >= minimum:
        font = get_font(size, bold=bold)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= max_width:
            return font
        size -= 2
    return get_font(minimum, bold=bold)


def assemble(number, title, stem, panels):
    column_width = 2800
    gap = 100
    outer = 100
    main_header = 210
    panel_header = 135
    images = []
    for source_name, panel_title in panels:
        source = SOURCE_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        scale = column_width / image.width
        scaled = image.resize((column_width, max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        images.append((source, panel_title, image.size, scaled))

    rows = math.ceil(len(images) / 2)
    row_heights = []
    for row in range(rows):
        items = images[row * 2:(row + 1) * 2]
        row_heights.append(panel_header + max(item[3].height for item in items))

    canvas_width = outer * 2 + column_width * 2 + gap
    canvas_height = outer * 2 + main_header + sum(row_heights) + gap * (rows - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    headline = f"Figure {number}. {title}"
    headline_font = fit_text(draw, headline, canvas_width - outer * 2, 82, bold=True, minimum=52)
    draw.text((outer, outer), headline, fill="black", font=headline_font)

    y = outer + main_header
    panel_rows = []
    for row in range(rows):
        row_items = images[row * 2:(row + 1) * 2]
        for col, (source, panel_title, original_size, scaled) in enumerate(row_items):
            x = outer + col * (column_width + gap)
            index = row * 2 + col
            letter = chr(ord("A") + index)
            draw.rounded_rectangle(
                (x, y, x + column_width, y + panel_header - 20),
                radius=18,
                fill=(242, 246, 250),
                outline=(60, 60, 60),
                width=3,
            )
            letter_font = get_font(64, bold=True)
            title_font = fit_text(draw, panel_title, column_width - 190, 50, bold=True, minimum=32)
            draw.text((x + 30, y + 24), letter, fill=(0, 70, 125), font=letter_font)
            draw.text((x + 130, y + 30), panel_title, fill="black", font=title_font)
            image_y = y + panel_header
            canvas.paste(scaled, (x, image_y))
            panel_rows.append({
                "figure_number": number,
                "figure_title": title,
                "panel": letter,
                "panel_title": panel_title,
                "source_png": str(source),
                "source_width_px": original_size[0],
                "source_height_px": original_size[1],
                "source_sha256_before": sha256(source),
                "composite_png": str(OUTPUT_DIR / f"{stem}.png"),
                "composite_pdf": str(OUTPUT_DIR / f"{stem}.pdf"),
                "scientific_content_redrawn": False,
            })
        y += row_heights[row] + (gap if row < rows - 1 else 0)

    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    canvas.save(png, dpi=(300, 300), optimize=True)
    canvas.save(pdf, "PDF", resolution=300.0)
    return panel_rows, {
        "figure_number": number,
        "figure_title": title,
        "panel_count": len(panels),
        "source_panels_reused_intact": len(panels),
        "scientific_panels_redrawn": 0,
        "png": str(png),
        "pdf": str(pdf),
        "png_width_px": canvas.width,
        "png_height_px": canvas.height,
        "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size,
        "png_sha256": sha256(png),
        "pdf_sha256": sha256(pdf),
        "supersedes_step86b": True,
    }


print("=" * 100)
print("STEP 86C - ASSEMBLE EXISTING FIGURES INTO SIX MAIN-MANUSCRIPT COMPOSITES")
print("=" * 100)

all_sources = []
for _, _, _, panels in FIGURES:
    all_sources.extend(SOURCE_DIR / name for name, _ in panels)
before_hashes = {str(path): sha256(path) for path in all_sources}

panel_rows = []
architecture_rows = []
for number, title, stem, panels in FIGURES:
    rows, architecture = assemble(number, title, stem, panels)
    panel_rows.extend(rows)
    architecture_rows.append(architecture)
    print(f"Figure {number}: {len(panels)} unchanged source panels assembled")

source_qc_rows = []
for path in all_sources:
    after = sha256(path)
    source_qc_rows.append({
        "source_png": str(path),
        "sha256_before": before_hashes[str(path)],
        "sha256_after": after,
        "unchanged": after == before_hashes[str(path)],
    })

composite_qc_rows = []
for row in architecture_rows:
    for output_type in ("png", "pdf"):
        path = Path(row[output_type])
        valid = path.exists() and path.stat().st_size > 0
        if output_type == "pdf" and valid:
            raw = path.read_bytes()
            valid = raw.startswith(b"%PDF") and b"%%EOF" in raw[-2048:]
        if output_type == "png" and valid:
            with Image.open(path) as image:
                valid = image.format == "PNG" and image.width > 0 and image.height > 0
        composite_qc_rows.append({
            "figure_number": row["figure_number"],
            "output_type": output_type.upper(),
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "format_valid": valid,
        })

pd.DataFrame(architecture_rows).to_csv(ARCHITECTURE_OUTPUT, index=False)
pd.DataFrame(panel_rows).to_csv(PANEL_OUTPUT, index=False)
pd.DataFrame(composite_qc_rows).to_csv(COMPOSITE_QC_OUTPUT, index=False)
pd.DataFrame(source_qc_rows).to_csv(SOURCE_QC_OUTPUT, index=False)

assert len(architecture_rows) == 6
assert len(panel_rows) == 49
assert all(row["unchanged"] for row in source_qc_rows)
assert all(row["format_valid"] for row in composite_qc_rows)

print(f"Existing panels assigned: {len(panel_rows)}")
print("Source panels changed: 0")
print("Scientific panels redrawn: 0")
print(f"Output directory: {OUTPUT_DIR}")
print("STEP 86C COMPLETED SUCCESSFULLY")

from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


Image.MAX_IMAGE_PIXELS = None

PROJECT = Path(r"E:\postdoc-work\ist-project")
SOURCE_DIR = PROJECT / "figures"
OUTPUT_DIR = PROJECT / "manuscript" / "main_figures_high_impact"
RESULTS_DIR = PROJECT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE_CSV = RESULTS_DIR / "step86e_high_impact_main_figure_architecture.csv"
PANEL_CSV = RESULTS_DIR / "step86e_high_impact_panel_manifest.csv"
OUTPUT_QC_CSV = RESULTS_DIR / "step86e_high_impact_output_qc.csv"
SOURCE_QC_CSV = RESULTS_DIR / "step86e_high_impact_source_integrity_qc.csv"

# Each nested list is one tightly packed row. Scientific panels are reused;
# only surrounding white margins are removed and panel letters are overlaid.
FIGURES = [
    {
        "number": 1,
        "title": "Peptide landscape, representations, and split integrity",
        "stem": "Figure1_Peptide_Landscape_and_Representation_Integrity",
        "rows": [
            ["Step25_Physicochemical_Distributions_Annotated.png"],
            ["Step26_Spearman_Correlation_Heatmap.png", "Step46_ESM2_PCA_Cumulative_Variance.png"],
            ["Step66_Sequence_Family_Redundancy.png", "Step66_Test_to_Development_Similarity.png"],
        ],
    },
    {
        "number": 2,
        "title": "Predictive performance and matched representation gains",
        "stem": "Figure2_Predictive_Performance_and_Representation_Gains",
        "rows": [
            ["Step36_Traditional_Model_ROC.png", "Step36_Traditional_Model_PR.png",
             "Step52_ESM2_Model_ROC.png", "Step52_ESM2_Model_PR.png"],
            ["Step53_Traditional_vs_ESM2_Metric_Deltas.png", "Step55_Traditional_vs_ESM2_Bootstrap_CI.png"],
            ["Step74_Model_Performance_Confidence_Intervals.png"],
        ],
    },
    {
        "number": 3,
        "title": "Calibration, clinical utility, and leading-model trade-offs",
        "stem": "Figure3_Calibration_Utility_and_Leading_Model_Tradeoffs",
        "rows": [
            ["Step71_Model_Calibration_Curves.png", "Step71_Calibration_Metrics.png"],
            ["Step72_Traditional_vs_ESM2_Calibration_Deltas.png",
             "Step75_ESM2_RF_vs_XGBoost_Paired_Bootstrap.png"],
            ["Step73_Traditional_Decision_Curves.png", "Step73_ESM2_Decision_Curves.png"],
        ],
    },
    {
        "number": 4,
        "title": "Sequence novelty and homology-aware generalization",
        "stem": "Figure4_Sequence_Novelty_and_Homology_Aware_Generalization",
        "rows": [
            ["Step67_Sequence_Novelty_AUROC_AUPRC.png", "Step67_Sequence_Novelty_MCC_F1.png"],
            ["Step68_Similarity_Stratum_Error_Rates.png",
             "Step68_Similarity_vs_Eight_Model_Difficulty.png"],
            ["Step69_Class_Similarity_vs_Difficulty.png", "Step70_Neighbor_Relation_Error_Burden.png"],
        ],
    },
    {
        "number": 5,
        "title": "Representation complementarity and stable ESM-2 signals",
        "stem": "Figure5_Representation_Complementarity_and_Stable_ESM2_Signals",
        "rows": [
            ["Step77_Traditional_vs_ESM2_Canonical_Correlation.png",
             "Step77_Descriptor_ESM2_PC_Association_Map.png"],
            ["Step78_Cross_Validated_Canonical_Correlations.png",
             "Step79_Fusion_vs_Single_Representation.png"],
            ["Step80_Fusion_vs_ESM2_Paired_Bootstrap.png",
             "Step81_ESM2_Feature_Importance_Stability.png"],
        ],
    },
    {
        "number": 6,
        "title": "Consensus hard cases and sequence-context interpretation",
        "stem": "Figure6_Consensus_Hard_Cases_and_Sequence_Context",
        "rows": [
            ["Step59_Eight_Model_Hard_Case_Map.png", "Step84_Universal_Hard_Case_Evidence_Map.png"],
            ["Step60_Hard_Case_Amino_Acid_Composition.png", "Step60_Hard_Case_Motif_Enrichment.png"],
            ["Step82_Hard_vs_Correct_Residue_Sensitivity.png", "Step82_Hard_Case_Residue_Sensitivity.png"],
            ["Step83_Residue_Category_Sensitivity.png", "Step83_Motif_Context_Sensitivity.png"],
        ],
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def get_font(size):
    path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def trim_white(image, padding=24, threshold=248):
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    # Pixels darker than threshold constitute content. Near-white antialiasing
    # alone does not expand the crop, while axes, labels, and footnotes remain.
    mask = gray.point(lambda value: 255 if value < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb, (0, 0, rgb.width, rgb.height)
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(rgb.width, bbox[2] + padding)
    bottom = min(rgb.height, bbox[3] + padding)
    return rgb.crop((left, top, right, bottom)), (left, top, right, bottom)


def place_letter(canvas, x, y, letter):
    draw = ImageDraw.Draw(canvas)
    size = 74
    fnt = get_font(size)
    # A thin white stroke separates the letter from any source background;
    # there is deliberately no box, banner, or panel title strip.
    draw.text((x + 12, y + 6), letter, font=fnt, fill="black",
              stroke_width=5, stroke_fill="white")


def build_figure(spec):
    canvas_width = 4800
    side_margin = 55
    top_bottom_margin = 55
    horizontal_gap = 55
    vertical_gap = 65
    available_width = canvas_width - 2 * side_margin

    loaded_rows = []
    panel_records = []
    letter_index = 0
    for row_index, filenames in enumerate(spec["rows"], start=1):
        loaded = []
        for filename in filenames:
            source = SOURCE_DIR / filename
            if not source.exists():
                raise FileNotFoundError(source)
            with Image.open(source) as handle:
                original = handle.convert("RGB")
            cropped, crop_box = trim_white(original)
            loaded.append((source, original.size, cropped, crop_box))

        usable = available_width - horizontal_gap * (len(loaded) - 1)
        aspect_sum = sum(item[2].width / item[2].height for item in loaded)
        common_height = round(usable / aspect_sum)
        scaled = []
        consumed = 0
        for index, item in enumerate(loaded):
            source, original_size, cropped, crop_box = item
            if index == len(loaded) - 1:
                target_width = usable - consumed
            else:
                target_width = round(common_height * cropped.width / cropped.height)
                consumed += target_width
            resized = cropped.resize((target_width, common_height), Image.Resampling.LANCZOS)
            scaled.append((source, original_size, crop_box, cropped.size, resized))
        loaded_rows.append((row_index, common_height, scaled))

    canvas_height = (2 * top_bottom_margin + sum(row[1] for row in loaded_rows)
                     + vertical_gap * (len(loaded_rows) - 1))
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    y = top_bottom_margin
    for row_index, row_height, panels in loaded_rows:
        x = side_margin
        for source, original_size, crop_box, cropped_size, panel in panels:
            letter = chr(ord("A") + letter_index)
            canvas.paste(panel, (x, y))
            place_letter(canvas, x, y, letter)
            panel_records.append({
                "figure_number": spec["number"],
                "figure_title": spec["title"],
                "row": row_index,
                "panel": letter,
                "source_png": str(source),
                "source_sha256_before": sha256(source),
                "original_width_px": original_size[0],
                "original_height_px": original_size[1],
                "crop_left": crop_box[0], "crop_top": crop_box[1],
                "crop_right": crop_box[2], "crop_bottom": crop_box[3],
                "cropped_width_px": cropped_size[0],
                "cropped_height_px": cropped_size[1],
                "placed_width_px": panel.width,
                "placed_height_px": panel.height,
                "scientific_content_redrawn": False,
            })
            x += panel.width + horizontal_gap
            letter_index += 1
        y += row_height + vertical_gap

    png = OUTPUT_DIR / f"{spec['stem']}.png"
    pdf = OUTPUT_DIR / f"{spec['stem']}.pdf"
    tiff = OUTPUT_DIR / f"{spec['stem']}.tiff"
    canvas.save(png, dpi=(600, 600), optimize=True)
    canvas.save(pdf, "PDF", resolution=600.0)
    canvas.save(tiff, compression="tiff_lzw", dpi=(600, 600))
    for record in panel_records:
        record.update({"output_png": str(png), "output_pdf": str(pdf), "output_tiff": str(tiff)})
    architecture = {
        "figure_number": spec["number"], "figure_title": spec["title"],
        "panel_count": len(panel_records), "row_count": len(spec["rows"]),
        "canvas_width_px": canvas.width, "canvas_height_px": canvas.height,
        "scientific_panels_redrawn": 0, "boxed_panel_headers": 0,
        "source_whitespace_trimmed": True, "png": str(png), "pdf": str(pdf),
        "tiff": str(tiff), "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size, "tiff_bytes": tiff.stat().st_size,
    }
    return panel_records, architecture


print("=" * 100)
print("STEP 86E - HIGH-IMPACT MAIN-MANUSCRIPT FIGURE REVISION")
print("=" * 100)

source_paths = []
for spec in FIGURES:
    for row in spec["rows"]:
        source_paths.extend(SOURCE_DIR / filename for filename in row)
before_hashes = {str(path): sha256(path) for path in source_paths}

panel_rows = []
architecture_rows = []
for spec in FIGURES:
    panels, architecture = build_figure(spec)
    panel_rows.extend(panels)
    architecture_rows.append(architecture)
    print(f"Figure {spec['number']}: {len(panels)} panels in {len(spec['rows'])} balanced rows")

source_qc_rows = []
for path in source_paths:
    after = sha256(path)
    source_qc_rows.append({"source_png": str(path), "sha256_before": before_hashes[str(path)],
                           "sha256_after": after, "unchanged": before_hashes[str(path)] == after})

output_qc_rows = []
for architecture in architecture_rows:
    for kind in ("png", "pdf", "tiff"):
        path = Path(architecture[kind])
        valid = path.exists() and path.stat().st_size > 0
        if kind in ("png", "tiff") and valid:
            with Image.open(path) as handle:
                expected = "PNG" if kind == "png" else "TIFF"
                valid = handle.format == expected and handle.width > 0 and handle.height > 0
        if kind == "pdf" and valid:
            raw = path.read_bytes()
            valid = raw.startswith(b"%PDF") and b"%%EOF" in raw[-2048:]
        output_qc_rows.append({"figure_number": architecture["figure_number"],
                               "format": kind.upper(), "path": str(path),
                               "bytes": path.stat().st_size if path.exists() else 0,
                               "format_valid": valid})

pd.DataFrame(architecture_rows).to_csv(ARCHITECTURE_CSV, index=False)
pd.DataFrame(panel_rows).to_csv(PANEL_CSV, index=False)
pd.DataFrame(output_qc_rows).to_csv(OUTPUT_QC_CSV, index=False)
pd.DataFrame(source_qc_rows).to_csv(SOURCE_QC_CSV, index=False)

assert len(architecture_rows) == 6
assert len(panel_rows) == 38
assert all(row["unchanged"] for row in source_qc_rows)
assert all(row["format_valid"] for row in output_qc_rows)
assert all(row["boxed_panel_headers"] == 0 for row in architecture_rows)

print(f"Selected high-value source panels: {len(panel_rows)}")
print("Boxed headers: 0")
print("Scientific panels redrawn: 0")
print(f"Output directory: {OUTPUT_DIR}")
print("STEP 86E COMPLETED SUCCESSFULLY")

from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


Image.MAX_IMAGE_PIXELS = None

PROJECT = Path(r"E:\postdoc-work\ist-project")
SOURCE_DIR = PROJECT / "figures"
OUTPUT_DIR = PROJECT / "manuscript" / "supplementary_figures_high_impact"
RESULTS_DIR = PROJECT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE_CSV = RESULTS_DIR / "step86f_high_impact_supplementary_architecture.csv"
PANEL_CSV = RESULTS_DIR / "step86f_high_impact_supplementary_panel_manifest.csv"
OUTPUT_QC_CSV = RESULTS_DIR / "step86f_high_impact_supplementary_output_qc.csv"
SOURCE_QC_CSV = RESULTS_DIR / "step86f_high_impact_supplementary_source_integrity_qc.csv"

# Every original panel left outside the revised main-text architecture is retained.
# Rows are curated by topic and aspect ratio; no panel title boxes are added.
SUPPLEMENTS = [
    {
        "number": 1,
        "title": "Additional descriptor and model diagnostics",
        "stem": "Supplementary_Figure_S1_Additional_Descriptor_and_Model_Diagnostics",
        "rows": [
            ["Step23_Physicochemical_Distributions.png"],
            ["Step39_XGBoost_Permutation_Importance.png", "Step40_Traditional_Model_Agreement.png"],
            ["Step76_Integrated_Model_Performance_Map.png"],
        ],
    },
    {
        "number": 2,
        "title": "Probability shifts and hard-case proximity",
        "stem": "Supplementary_Figure_S2_Probability_Shifts_and_Hard_Case_Proximity",
        "rows": [
            ["Step57_Unanimous_Error_Probability_Shifts.png", "Step58_Persistent_Error_Descriptor_Zscores.png"],
            ["Step61_Hard_Case_Nearest_Neighbor_Map.png", "Step62_Hard_Case_Sequence_Similarity_Map.png"],
        ],
    },
    {
        "number": 3,
        "title": "Sequence families and development-neighborhood context",
        "stem": "Supplementary_Figure_S3_Sequence_Families_and_Development_Neighborhoods",
        "rows": [
            ["Step62_Hard_Case_Cluster_Network.png", "Step63_Hard_Case_Development_Neighbor_Map.png"],
            ["Step63_Hard_Case_Development_Proximity.png", "Step64_Balanced_Development_Similarity_Margins.png"],
            ["Step65_Hard_Family_Development_Network.png", "Step65_Recurrent_Development_Neighbors.png"],
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
    draw.text((x + 12, y + 6), letter, font=get_font(74), fill="black",
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
                "supplementary_figure": f"S{spec['number']}",
                "figure_title": spec["title"], "row": row_index, "panel": letter,
                "source_png": str(source), "source_sha256_before": sha256(source),
                "original_width_px": original_size[0], "original_height_px": original_size[1],
                "crop_left": crop_box[0], "crop_top": crop_box[1],
                "crop_right": crop_box[2], "crop_bottom": crop_box[3],
                "cropped_width_px": cropped_size[0], "cropped_height_px": cropped_size[1],
                "placed_width_px": panel.width, "placed_height_px": panel.height,
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
        "supplementary_figure": f"S{spec['number']}", "figure_title": spec["title"],
        "panel_count": len(panel_records), "row_count": len(spec["rows"]),
        "canvas_width_px": canvas.width, "canvas_height_px": canvas.height,
        "scientific_panels_redrawn": 0, "boxed_panel_headers": 0,
        "source_whitespace_trimmed": True, "png": str(png), "pdf": str(pdf),
        "tiff": str(tiff), "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size, "tiff_bytes": tiff.stat().st_size,
    }
    return panel_records, architecture


print("=" * 100)
print("STEP 86F - HIGH-IMPACT SUPPLEMENTARY-FIGURE REVISION")
print("=" * 100)

source_paths = []
for spec in SUPPLEMENTS:
    for row in spec["rows"]:
        source_paths.extend(SOURCE_DIR / filename for filename in row)
assert len(source_paths) == 14
assert len({path.name for path in source_paths}) == 14
before_hashes = {str(path): sha256(path) for path in source_paths}

panel_rows = []
architecture_rows = []
for spec in SUPPLEMENTS:
    panels, architecture = build_figure(spec)
    panel_rows.extend(panels)
    architecture_rows.append(architecture)
    print(f"Supplementary Figure S{spec['number']}: {len(panels)} panels in {len(spec['rows'])} rows")

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
                valid = handle.format == ("PNG" if kind == "png" else "TIFF") and handle.width > 0 and handle.height > 0
        if kind == "pdf" and valid:
            raw = path.read_bytes()
            valid = raw.startswith(b"%PDF") and b"%%EOF" in raw[-2048:]
        output_qc_rows.append({"supplementary_figure": architecture["supplementary_figure"],
                               "format": kind.upper(), "path": str(path),
                               "bytes": path.stat().st_size if path.exists() else 0,
                               "format_valid": valid})

pd.DataFrame(architecture_rows).to_csv(ARCHITECTURE_CSV, index=False)
pd.DataFrame(panel_rows).to_csv(PANEL_CSV, index=False)
pd.DataFrame(output_qc_rows).to_csv(OUTPUT_QC_CSV, index=False)
pd.DataFrame(source_qc_rows).to_csv(SOURCE_QC_CSV, index=False)

assert len(architecture_rows) == 3
assert len(panel_rows) == 14
assert all(row["unchanged"] for row in source_qc_rows)
assert all(row["format_valid"] for row in output_qc_rows)
assert all(row["boxed_panel_headers"] == 0 for row in architecture_rows)

print(f"Supplementary panels retained: {len(panel_rows)}")
print("Boxed headers: 0")
print("Scientific panels redrawn: 0")
print(f"Output directory: {OUTPUT_DIR}")
print("STEP 86F COMPLETED SUCCESSFULLY")

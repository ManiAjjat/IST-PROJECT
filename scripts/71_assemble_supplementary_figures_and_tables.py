from pathlib import Path
import hashlib
import math
import shutil

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(r"E:\postdoc-work\ist-project")
FIGURE_DIR = PROJECT / "figures"
RESULTS_DIR = PROJECT / "results"
SUPPLEMENT_DIR = PROJECT / "manuscript" / "supplementary_figures"
TABLE_DIR = PROJECT / "manuscript" / "tables"
SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_PANEL_MANIFEST = RESULTS_DIR / "step86c_existing_figure_panel_manifest.csv"
SUPP_ARCHITECTURE = RESULTS_DIR / "step86d_supplementary_figure_architecture.csv"
SUPP_PANEL_MANIFEST = RESULTS_DIR / "step86d_supplementary_panel_manifest.csv"
SUPP_QC = RESULTS_DIR / "step86d_supplementary_composite_qc.csv"
SOURCE_QC = RESULTS_DIR / "step86d_supplementary_source_integrity_qc.csv"
TABLE_MANIFEST = RESULTS_DIR / "step86d_main_text_table_manifest.csv"
TABLE_QC = RESULTS_DIR / "step86d_main_text_table_qc.csv"

SUPPLEMENTS = [
    (
        1,
        "Additional model and descriptor diagnostics",
        "Supplementary_Figure_S1_Additional_Model_and_Descriptor_Diagnostics",
        [
            ("Step23_Physicochemical_Distributions.png", "Complete physicochemical distributions"),
            ("Step39_XGBoost_Permutation_Importance.png", "Traditional XGBoost permutation importance"),
            ("Step40_Traditional_Model_Agreement.png", "Traditional-model agreement and errors"),
            ("Step76_Integrated_Model_Performance_Map.png", "Integrated multi-domain model summary"),
        ],
    ),
    (
        2,
        "Additional probability-shift and hard-case proximity analyses",
        "Supplementary_Figure_S2_Probability_Shifts_and_Hard_Case_Proximity",
        [
            ("Step57_Unanimous_Error_Probability_Shifts.png", "Unanimous-error probability shifts"),
            ("Step58_Persistent_Error_Descriptor_Zscores.png", "Persistent-error descriptor extremeness"),
            ("Step61_Hard_Case_Nearest_Neighbor_Map.png", "Hard-case nearest-neighbor map"),
            ("Step62_Hard_Case_Sequence_Similarity_Map.png", "Within-hard-case sequence similarity"),
        ],
    ),
    (
        3,
        "Sequence families and development-neighborhood context",
        "Supplementary_Figure_S3_Sequence_Families_and_Development_Neighborhoods",
        [
            ("Step62_Hard_Case_Cluster_Network.png", "Hard-case sequence-cluster network"),
            ("Step63_Hard_Case_Development_Neighbor_Map.png", "Development-neighbor evidence map"),
            ("Step63_Hard_Case_Development_Proximity.png", "Development-set proximity margins"),
            ("Step64_Balanced_Development_Similarity_Margins.png", "Class-balanced development margins"),
            ("Step65_Hard_Family_Development_Network.png", "Hard-family development network"),
            ("Step65_Recurrent_Development_Neighbors.png", "Recurrent development neighbors"),
        ],
    ),
]

TABLES = [
    (1, "Dataset and feature summary", "step85_table1_dataset_feature_summary.csv",
     "step85_table1_dataset_feature_summary_manuscript.csv"),
    (2, "Primary model performance", "step85_table2_primary_model_performance.csv",
     "step85_table2_primary_model_performance_manuscript.csv"),
    (3, "Matched representation comparison", "step85_table3_matched_representation_comparison.csv",
     "step85_table3_matched_representation_comparison_manuscript.csv"),
    (4, "Universal hard-case interpretability", "step85_table4_universal_hard_case_interpretability.csv",
     "step85_table4_universal_hard_case_interpretability_manuscript.csv"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def font(size, bold=False):
    path = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def fit_font(draw, text, width, start=52, minimum=30, bold=True):
    for size in range(start, minimum - 1, -2):
        candidate = font(size, bold)
        box = draw.textbbox((0, 0), text, font=candidate)
        if box[2] - box[0] <= width:
            return candidate
    return font(minimum, bold)


def assemble(number, title, stem, panels):
    column_width, gap, outer, figure_header, panel_header = 2800, 100, 100, 210, 135
    loaded = []
    for filename, panel_title in panels:
        source = FIGURE_DIR / filename
        if not source.exists():
            raise FileNotFoundError(source)
        with Image.open(source) as handle:
            original = handle.convert("RGB")
        original_size = original.size
        scale = column_width / original.width
        resized = original.resize(
            (column_width, max(1, round(original.height * scale))), Image.Resampling.LANCZOS
        )
        loaded.append((source, panel_title, original_size, resized))

    rows = math.ceil(len(loaded) / 2)
    heights = [
        panel_header + max(item[3].height for item in loaded[row * 2:(row + 1) * 2])
        for row in range(rows)
    ]
    width = outer * 2 + column_width * 2 + gap
    height = outer * 2 + figure_header + sum(heights) + gap * (rows - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headline = f"Supplementary Figure S{number}. {title}"
    draw.text((outer, outer), headline, fill="black",
              font=fit_font(draw, headline, width - outer * 2, start=78, minimum=48))

    y = outer + figure_header
    panel_records = []
    for row in range(rows):
        for col, (source, panel_title, original_size, image) in enumerate(loaded[row * 2:(row + 1) * 2]):
            index = row * 2 + col
            letter = chr(ord("A") + index)
            x = outer + col * (column_width + gap)
            draw.rounded_rectangle((x, y, x + column_width, y + panel_header - 20), radius=18,
                                   fill=(242, 246, 250), outline=(60, 60, 60), width=3)
            draw.text((x + 30, y + 24), letter, fill=(0, 70, 125), font=font(64, True))
            draw.text((x + 130, y + 30), panel_title, fill="black",
                      font=fit_font(draw, panel_title, column_width - 190))
            canvas.paste(image, (x, y + panel_header))
            panel_records.append({
                "supplementary_figure": f"S{number}", "panel": letter,
                "panel_title": panel_title, "source_png": str(source),
                "source_width_px": original_size[0], "source_height_px": original_size[1],
                "source_sha256_before": sha256(source),
                "scientific_content_redrawn": False,
            })
        y += heights[row] + (gap if row < rows - 1 else 0)

    png = SUPPLEMENT_DIR / f"{stem}.png"
    pdf = SUPPLEMENT_DIR / f"{stem}.pdf"
    canvas.save(png, dpi=(300, 300), optimize=True)
    canvas.save(pdf, "PDF", resolution=300.0)
    for record in panel_records:
        record["composite_png"] = str(png)
        record["composite_pdf"] = str(pdf)
    architecture = {
        "supplementary_figure": f"S{number}", "title": title,
        "panel_count": len(panels), "scientific_panels_redrawn": 0,
        "png": str(png), "pdf": str(pdf), "png_width_px": width,
        "png_height_px": height, "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size, "png_sha256": sha256(png),
        "pdf_sha256": sha256(pdf),
    }
    return panel_records, architecture


print("=" * 100)
print("STEP 86D - SUPPLEMENTARY FIGURE ASSEMBLY AND MAIN-TEXT TABLE ORGANIZATION")
print("=" * 100)

main_used = set(Path(path).name for path in pd.read_csv(MAIN_PANEL_MANIFEST)["source_png"])
all_existing = set(path.name for path in FIGURE_DIR.glob("*.png"))
supplement_sources = [FIGURE_DIR / filename for _, _, _, panels in SUPPLEMENTS for filename, _ in panels]
supplement_names = {path.name for path in supplement_sources}
leftovers = all_existing - main_used
assert supplement_names == leftovers, (
    f"Supplement assignment mismatch; unassigned={sorted(leftovers - supplement_names)}, "
    f"unexpected={sorted(supplement_names - leftovers)}"
)

before_hashes = {str(path): sha256(path) for path in supplement_sources}
panel_rows, architecture_rows = [], []
for number, title, stem, panels in SUPPLEMENTS:
    records, architecture = assemble(number, title, stem, panels)
    panel_rows.extend(records)
    architecture_rows.append(architecture)
    print(f"Supplementary Figure S{number}: {len(panels)} unchanged panels")

source_qc_rows = []
for path in supplement_sources:
    after = sha256(path)
    source_qc_rows.append({"source_png": str(path), "sha256_before": before_hashes[str(path)],
                           "sha256_after": after, "unchanged": before_hashes[str(path)] == after})

composite_qc_rows = []
for row in architecture_rows:
    for kind in ("png", "pdf"):
        path = Path(row[kind])
        valid = path.exists() and path.stat().st_size > 0
        if kind == "png" and valid:
            with Image.open(path) as handle:
                valid = handle.format == "PNG" and handle.width > 0 and handle.height > 0
        if kind == "pdf" and valid:
            raw = path.read_bytes()
            valid = raw.startswith(b"%PDF") and b"%%EOF" in raw[-2048:]
        composite_qc_rows.append({"supplementary_figure": row["supplementary_figure"],
                                  "format": kind.upper(), "path": str(path),
                                  "bytes": path.stat().st_size if path.exists() else 0,
                                  "format_valid": valid})

table_rows, table_qc_rows = [], []
for number, title, full_name, manuscript_name in TABLES:
    for version, source_name, destination_name in (
        ("full_precision", full_name, f"Table{number}_{title.replace(' ', '_')}_Full_Precision.csv"),
        ("manuscript_ready", manuscript_name, f"Table{number}_{title.replace(' ', '_')}_Manuscript.csv"),
    ):
        source = RESULTS_DIR / source_name
        destination = TABLE_DIR / destination_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        source_hash, destination_hash = sha256(source), sha256(destination)
        rows = len(pd.read_csv(destination))
        table_rows.append({"table_number": number, "table_title": title, "version": version,
                           "source": str(source), "destination": str(destination), "rows": rows,
                           "sha256": destination_hash})
        table_qc_rows.append({"table_number": number, "version": version,
                              "source_destination_identical": source_hash == destination_hash,
                              "destination_exists": destination.exists(), "rows": rows})

pd.DataFrame(architecture_rows).to_csv(SUPP_ARCHITECTURE, index=False)
pd.DataFrame(panel_rows).to_csv(SUPP_PANEL_MANIFEST, index=False)
pd.DataFrame(composite_qc_rows).to_csv(SUPP_QC, index=False)
pd.DataFrame(source_qc_rows).to_csv(SOURCE_QC, index=False)
pd.DataFrame(table_rows).to_csv(TABLE_MANIFEST, index=False)
pd.DataFrame(table_qc_rows).to_csv(TABLE_QC, index=False)

assert len(panel_rows) == 14
assert len(table_rows) == 8
assert all(row["unchanged"] for row in source_qc_rows)
assert all(row["format_valid"] for row in composite_qc_rows)
assert all(row["source_destination_identical"] for row in table_qc_rows)

print(f"Previously unused figures assigned: {len(panel_rows)}")
print(f"Main-text table files copied: {len(table_rows)}")
print(f"Supplementary figure folder: {SUPPLEMENT_DIR}")
print(f"Main-text table folder: {TABLE_DIR}")
print("STEP 86D COMPLETED SUCCESSFULLY")

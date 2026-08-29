from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
INPUT_FILE = PROJECT_DIR / "derived" / "lung_binary_labeled.csv"
OUTPUT_FILE = PROJECT_DIR / "derived" / "traditional_features_step16.csv"

df = pd.read_csv(INPUT_FILE)
df["length"] = df["sequence"].astype(str).str.len()

print("=" * 70)
print("STEP 16 - CALCULATE PEPTIDE LENGTH")
print("=" * 70)
print("Input:", INPUT_FILE)
print("Output:", OUTPUT_FILE)

print("\n16G. Overall length statistics:")
print("Minimum:", df["length"].min())
print("Maximum:", df["length"].max())
print("Mean:", round(df["length"].mean(), 2))
print("Median:", df["length"].median())
print("Standard deviation:", round(df["length"].std(), 2))

print("\n16H. Length statistics by binary class:")
class_stats = df.groupby("binary_class")["length"].agg(
    count="count",
    mean="mean",
    median="median",
    minimum="min",
    maximum="max",
    std="std",
)
print(class_stats.round(2))

missing_length = int(df["length"].isna().sum())
zero_length = int((df["length"] <= 0).sum())
print("\n16I. Length safety checks:")
print("Missing length values:", missing_length)
print("Zero/negative lengths:", zero_length)
if missing_length != 0:
    raise ValueError("STOP: Missing peptide-length values found.")
if zero_length != 0:
    raise ValueError("STOP: Invalid peptide lengths found.")

columns_to_save = [
    "ID",
    "sequence",
    "class",
    "original_class",
    "label",
    "binary_class",
    "inactive_source",
    "is_virtual_inactive",
    "length",
]
feature_df = df[columns_to_save].copy()
feature_df.to_csv(OUTPUT_FILE, index=False)

print("\n16J. Feature dataset saved successfully:")
print(OUTPUT_FILE)
print("\n" + "=" * 70)
print("STEP 16 SUMMARY")
print("=" * 70)
print("Total peptides:", len(feature_df))
print("Traditional features currently calculated: 1")
print("Feature: peptide length")
print("Missing length values:", missing_length)
print("Invalid lengths:", zero_length)
print("\nSTEP 16 COMPLETED SUCCESSFULLY")
print("=" * 70)


# =========================================================
# STEP 17: CALCULATE AMINO-ACID COMPOSITION (AAC)
# =========================================================
print("\nSTEP 17 - AMINO-ACID COMPOSITION (AAC)")
print("=" * 70)


amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aac_columns = [f"AAC_{amino_acid}" for amino_acid in amino_acids]

for amino_acid, column in zip(amino_acids, aac_columns):
    df[column] = df["sequence"].astype(str).str.count(amino_acid) / df["length"]

df["AAC_sum_check"] = df[aac_columns].sum(axis=1)
aac_sum_failures = int((df["AAC_sum_check"].sub(1).abs() > 1e-9).sum())
missing_aac = int(df[aac_columns].isna().sum().sum())
aac_min = float(df[aac_columns].min().min())
aac_max = float(df[aac_columns].max().max())

print("\n17G. AAC sum quality check:")
print("Minimum AAC sum:", round(df["AAC_sum_check"].min(), 6))
print("Maximum AAC sum:", round(df["AAC_sum_check"].max(), 6))
print("Peptides failing AAC-sum check:", aac_sum_failures)
print("Missing AAC values:", missing_aac)
print("Minimum AAC value:", round(aac_min, 6))
print("Maximum AAC value:", round(aac_max, 6))

if aac_sum_failures != 0:
    raise ValueError("STOP: AAC values do not sum to 1 for every peptide.")
if missing_aac != 0:
    raise ValueError("STOP: Missing AAC values found.")
if aac_min < 0 or aac_max > 1:
    raise ValueError("STOP: AAC value outside 0-1 range.")

print("\n17I. Mean AAC by binary class:")
mean_aac_by_class = df.groupby("binary_class")[aac_columns].mean().T
print(mean_aac_by_class.round(4).to_string())

df = df.drop(columns=["AAC_sum_check"])
metadata_columns = [
    "ID",
    "sequence",
    "class",
    "original_class",
    "label",
    "binary_class",
    "inactive_source",
    "is_virtual_inactive",
]
traditional_features = ["length"] + aac_columns
step17_output = PROJECT_DIR / "derived" / "traditional_features_step17.csv"
step17_df = df[metadata_columns + traditional_features].copy()
step17_df.to_csv(step17_output, index=False)

print("\n" + "=" * 70)
print("STEP 17 SUMMARY")
print("=" * 70)
print("Total peptides:", len(step17_df))
print("Peptide length features: 1")
print("AAC features:", len(aac_columns))
print("Total traditional features:", len(traditional_features))
print("AAC sum failures:", aac_sum_failures)
print("Missing AAC values:", missing_aac)
print("\nOutput file:")
print(step17_output)
print("\nSTEP 17 COMPLETED SUCCESSFULLY")
print("=" * 70)


# =========================================================
# STEP 18: MOLECULAR WEIGHT AND NET CHARGE
# =========================================================
print("\n" + "=" * 70)
print("STEP 18 - MOLECULAR WEIGHT AND NET CHARGE AT pH 7.4")
print("=" * 70)

from Bio.SeqUtils.ProtParam import ProteinAnalysis


def calculate_molecular_weight(sequence):
    sequence = str(sequence).strip().upper()
    return ProteinAnalysis(sequence).molecular_weight()


def calculate_net_charge(sequence, pH=7.4):
    sequence = str(sequence).strip().upper()
    return ProteinAnalysis(sequence).charge_at_pH(pH)


step18_df = step17_df.copy()
step18_df["molecular_weight"] = step18_df["sequence"].apply(
    calculate_molecular_weight
)
step18_df["net_charge_pH7_4"] = step18_df["sequence"].apply(
    calculate_net_charge
)

step18_output = PROJECT_DIR / "derived" / "traditional_features_step18.csv"
step18_df.to_csv(step18_output, index=False)

print("\n18A. Feature calculation checks:")
print("Missing molecular weights:", int(step18_df["molecular_weight"].isna().sum()))
print("Missing net charges:", int(step18_df["net_charge_pH7_4"].isna().sum()))
print(
    "Molecular weight range:",
    round(step18_df["molecular_weight"].min(), 2),
    "to",
    round(step18_df["molecular_weight"].max(), 2),
)
print(
    "Net charge range:",
    round(step18_df["net_charge_pH7_4"].min(), 4),
    "to",
    round(step18_df["net_charge_pH7_4"].max(), 4),
)

assert len(step18_df) == 901, "Expected 901 peptides."
assert int(step18_df["molecular_weight"].isna().sum()) == 0
assert int(step18_df["net_charge_pH7_4"].isna().sum()) == 0

print("\n18B. First five new feature values:")
print(
    step18_df[
        ["ID", "sequence", "molecular_weight", "net_charge_pH7_4"]
    ].head().to_string(index=False)
)

print("\nOutput file:")
print(step18_output)
print("\n" + "=" * 70)
print("STEP 18 COMPLETED SUCCESSFULLY")
print("=" * 70)


# =========================================================
# STEP 19: pI, AROMATICITY, AND INSTABILITY INDEX
# =========================================================
print("\n" + "=" * 70)
print("STEP 19 - pI, AROMATICITY, AND INSTABILITY INDEX")
print("=" * 70)

step19_df = step18_df.copy()

print("\n19A. Calculating isoelectric point (pI)...")
step19_df["isoelectric_point"] = step19_df["sequence"].apply(
    lambda sequence: ProteinAnalysis(str(sequence).strip().upper()).isoelectric_point()
)
print("pI calculation completed.")

print("\n19B. Calculating aromaticity...")
step19_df["aromaticity"] = step19_df["sequence"].apply(
    lambda sequence: ProteinAnalysis(str(sequence).strip().upper()).aromaticity()
)
print("Aromaticity calculation completed.")

print("\n19C. Calculating instability index...")
step19_df["instability_index"] = step19_df["sequence"].apply(
    lambda sequence: ProteinAnalysis(str(sequence).strip().upper()).instability_index()
)
print("Instability-index calculation completed.")

print("\n19D. First 10 peptides:")
print(
    step19_df[
        [
            "ID",
            "sequence",
            "isoelectric_point",
            "aromaticity",
            "instability_index",
        ]
    ]
    .head(10)
    .round(4)
    .to_string(index=False)
)

print("\n19E. Overall pI statistics:")
print(step19_df["isoelectric_point"].describe().round(3))

step19_output = PROJECT_DIR / "derived" / "traditional_features_step19.csv"
step19_df.to_csv(step19_output, index=False)

new_feature_columns = [
    "isoelectric_point",
    "aromaticity",
    "instability_index",
]
missing_step19 = int(step19_df[new_feature_columns].isna().sum().sum())
print("\n19F. Safety checks:")
print("Missing Step 19 values:", missing_step19)
print("Rows:", len(step19_df))
print("Columns:", len(step19_df.columns))
print("ML features:", len(step19_df.columns) - len(metadata_columns))
if missing_step19 != 0:
    raise ValueError("STOP: Missing Step 19 descriptor values found.")
if len(step19_df) != 901:
    raise ValueError("Expected 901 peptides.")
if len(step19_df.columns) != 34:
    raise ValueError("Expected 34 total columns.")

print("\nOutput file:")
print(step19_output)
print("\n" + "=" * 70)
print("STEP 19 COMPLETED SUCCESSFULLY")
print("=" * 70)


# =========================================================
# STEP 20: HYDROPHOBICITY AND HYDROPHOBIC MOMENT
# =========================================================
print("\n" + "=" * 70)
print("STEP 20 - HYDROPHOBICITY AND HYDROPHOBIC MOMENT")
print("=" * 70)

from modlamp.descriptors import PeptideDescriptor

sequences = step19_df["sequence"].astype(str).str.strip().str.upper().tolist()
hydrophobicity_values = []
moment_values = []

for sequence in sequences:
    global_descriptor = PeptideDescriptor([sequence], scalename="Eisenberg")
    global_descriptor.calculate_global()
    hydrophobicity_values.append(float(global_descriptor.descriptor[0][0]))

    moment_descriptor = PeptideDescriptor([sequence], scalename="Eisenberg")
    moment_descriptor.calculate_moment(angle=100)
    moment_values.append(float(moment_descriptor.descriptor[0][0]))

step20_df = step19_df.copy()
step20_df["mean_eisenberg_hydrophobicity"] = hydrophobicity_values
step20_df["hydrophobic_moment"] = moment_values

missing_hydrophobicity = int(
    step20_df["mean_eisenberg_hydrophobicity"].isna().sum()
)
missing_moment = int(step20_df["hydrophobic_moment"].isna().sum())
step20_features = [
    column
    for column in step20_df.columns
    if column not in metadata_columns
]

print("\n20K. Feature checks:")
print("Missing hydrophobicity values:", missing_hydrophobicity)
print("Missing hydrophobic-moment values:", missing_moment)
print(
    "Hydrophobicity range:",
    round(step20_df["mean_eisenberg_hydrophobicity"].min(), 6),
    "to",
    round(step20_df["mean_eisenberg_hydrophobicity"].max(), 6),
)
print(
    "Hydrophobic-moment range:",
    round(step20_df["hydrophobic_moment"].min(), 6),
    "to",
    round(step20_df["hydrophobic_moment"].max(), 6),
)

if missing_hydrophobicity != 0 or missing_moment != 0:
    raise ValueError("STOP: Missing Step 20 descriptor values found.")
if len(step20_df) != 901:
    raise ValueError("Expected 901 peptides.")
if len(step20_features) != 28:
    raise ValueError("Expected 28 traditional ML features.")

step20_output = PROJECT_DIR / "derived" / "traditional_features_step20.csv"
step20_df.to_csv(step20_output, index=False)

print("\n20L. Final summary:")
print("Total peptides:", len(step20_df))
print("Total traditional ML features:", len(step20_features))
print("Missing hydrophobicity values:", missing_hydrophobicity)
print("Missing hydrophobic-moment values:", missing_moment)
print("\nOutput file:")
print(step20_output)
print("\nSTEP 20 COMPLETED SUCCESSFULLY")
print("=" * 70)


# =========================================================
# STEP 21: FINAL GLOBAL PHYSICOCHEMICAL DESCRIPTORS
# =========================================================
print("\n" + "=" * 70)
print("STEP 21 - FINAL GLOBAL PHYSICOCHEMICAL DESCRIPTORS")
print("=" * 70)

import numpy as np
from modlamp.descriptors import GlobalDescriptor

step21_df = step20_df.copy()
sequences = step21_df["sequence"].astype(str).str.strip().str.upper().tolist()
global_methods = {
    "charge_density": "charge_density",
    "aliphatic_index": "aliphatic_index",
    "boman_index": "boman_index",
    "hydrophobic_ratio": "hydrophobic_ratio",
}

for column, method_name in global_methods.items():
    descriptor = GlobalDescriptor(sequences)
    getattr(descriptor, method_name)()
    step21_df[column] = descriptor.descriptor[:, 0]

new_step21_features = list(global_methods)
new_values = step21_df[new_step21_features].to_numpy(dtype=float)
missing_step21 = int(step21_df[new_step21_features].isna().sum().sum())
nonfinite_step21 = int((~np.isfinite(new_values)).sum())
step21_features = [
    column
    for column in step21_df.columns
    if column not in metadata_columns
]

print("\n21D. Descriptor mapping:")
for column, method_name in global_methods.items():
    print(f"{column} -> GlobalDescriptor.{method_name}")
print("\n21E. New descriptor checks:")
print("Missing new descriptor values:", missing_step21)
print("Non-finite new descriptor values:", nonfinite_step21)
print("Traditional ML features:", len(step21_features))
print("Total CSV columns:", len(step21_df.columns))

if missing_step21 != 0:
    raise ValueError("STOP: Missing Step 21 descriptor values found.")
if nonfinite_step21 != 0:
    raise ValueError("STOP: Non-finite Step 21 descriptor values found.")
if len(step21_df) != 901:
    raise ValueError("Expected 901 peptides.")
if len(step21_features) != 32:
    raise ValueError("Expected 32 traditional ML features.")
if len(step21_df.columns) != 40:
    raise ValueError("Expected 40 total CSV columns.")

step21_output = PROJECT_DIR / "derived" / "traditional_features.csv"
step21_df.to_csv(step21_output, index=False)

print("\nFinal traditional-feature file:")
print(step21_output)
print("\nSTEP 21 COMPLETED SUCCESSFULLY")
print("=" * 70)
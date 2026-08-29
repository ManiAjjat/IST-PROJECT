from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
DATA_FILE = PROJECT_DIR / "data" / "ACPs_Lung_cancer.csv"

print("=" * 70)
print("ACTIVE/INACTIVE ANTICANCER PEPTIDE PREDICTION - LUNG CANCER")
print("=" * 70)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nDataset location:")
print(DATA_FILE)

print("\nChecking whether dataset exists...")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

print("Dataset found successfully!")
df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("Column names:", df.columns.tolist())

print("\nFirst five peptides:")
print(df.head().to_string(index=False))

print("\nOriginal class distribution:")
print(df["class"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 70)


# =========================================================
# STEP 13: RAW DATASET QUALITY CONTROL
# =========================================================
print("\nSTEP 13 - RAW DATASET QUALITY CONTROL")
print("=" * 70)

print("\n13A. Missing values in each column:")
print(df.isna().sum())

id_text = df["ID"].astype("string").str.strip()
sequence_text = df["sequence"].astype("string").str.strip()
class_text = df["class"].astype("string").str.strip()

empty_id_count = int(id_text.fillna("").eq("").sum())
empty_sequence_count = int(sequence_text.fillna("").eq("").sum())
empty_class_count = int(class_text.fillna("").eq("").sum())

print("\n13B. Empty text values:")
print("Empty ID values:", empty_id_count)
print("Empty sequence values:", empty_sequence_count)
print("Empty class values:", empty_class_count)

duplicate_ids = int(df["ID"].duplicated(keep=False).sum())
duplicate_sequences = int(sequence_text.duplicated(keep=False).sum())

print("\n13C. Duplicate IDs:")
print(duplicate_ids)

print("\n13D. Exact duplicate peptide sequences:")
print(duplicate_sequences)

sequence_lengths = sequence_text.fillna("").str.len()
print("\n13E. Peptide length statistics:")
print("Minimum length:", int(sequence_lengths.min()))
print("Maximum length:", int(sequence_lengths.max()))
print("Mean length:", float(sequence_lengths.mean()))
print("Median length:", float(sequence_lengths.median()))

lowercase_count = int(sequence_text.fillna("").str.contains(r"[a-z]", regex=True).sum())
whitespace_count = int(sequence_text.fillna("").str.contains(r"\s", regex=True).sum())
canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
noncanonical_rows = []

for row_index, sequence in sequence_text.items():
    if pd.isna(sequence):
        invalid_characters = {"<MISSING>"}
    else:
        invalid_characters = set(str(sequence)) - canonical_amino_acids
    if invalid_characters:
        noncanonical_rows.append((row_index, invalid_characters))

print("\n13F. Sequences containing lowercase letters:")
print(lowercase_count)

print("\n13G. Sequences containing whitespace:")
print(whitespace_count)

print("\n13H. Sequences containing non-canonical amino acids:")
print(len(noncanonical_rows))
if noncanonical_rows:
    print(noncanonical_rows[:10])

expected_classes = {
    "very active",
    "mod. active",
    "inactive - exp",
    "inactive - virtual",
}
observed_classes = set(class_text.dropna())

print("\n13I. Observed classes:")
print(sorted(observed_classes))
print("\nExpected classes:")
print(sorted(expected_classes))
missing_classes = expected_classes - observed_classes
unexpected_classes = observed_classes - expected_classes
print("\nMissing expected classes:")
print(sorted(missing_classes))
print("\nUnexpected classes:")
print(sorted(unexpected_classes))

print("\n" + "=" * 70)
print("STEP 13 QUALITY-CONTROL SUMMARY")
print("=" * 70)
print("Total rows:", len(df))
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate IDs:", duplicate_ids)
print("Duplicate sequences:", duplicate_sequences)
print("Sequences with lowercase letters:", lowercase_count)
print("Sequences with whitespace:", whitespace_count)
print("Sequences with non-canonical characters:", len(noncanonical_rows))
print("Missing expected classes:", len(missing_classes))
print("Unexpected classes:", len(unexpected_classes))
print("\nSTEP 13 COMPLETED")
print("=" * 70)


# =========================================================
# STEP 14: CREATE BINARY ACTIVE/INACTIVE LABELS
# =========================================================
print("\nSTEP 14 - CREATE BINARY ACTIVE/INACTIVE LABELS")
print("=" * 70)

binary_df = df.copy()
binary_df["original_class"] = binary_df["class"].astype(str).str.strip()


def assign_label(original_class):
    if original_class in {"very active", "mod. active"}:
        return 1
    if original_class in {"inactive - exp", "inactive - virtual"}:
        return 0
    raise ValueError(f"Unexpected class: {original_class}")


def assign_binary_class(label):
    return "Active" if label == 1 else "Inactive"


def assign_inactive_source(original_class):
    if original_class == "inactive - virtual":
        return "virtual"
    if original_class == "inactive - exp":
        return "experimental"
    return "active"


binary_df["label"] = binary_df["original_class"].apply(assign_label)
binary_df["binary_class"] = binary_df["label"].apply(assign_binary_class)
binary_df["inactive_source"] = binary_df["original_class"].apply(assign_inactive_source)
binary_df["is_virtual_inactive"] = (
    binary_df["original_class"] == "inactive - virtual"
).astype(int)

print("\n14H. Binary label distribution:")
print(binary_df["label"].value_counts().sort_index())
print("\nHuman-readable binary distribution:")
print(binary_df["binary_class"].value_counts())

active_count = int((binary_df["label"] == 1).sum())
inactive_count = int((binary_df["label"] == 0).sum())
virtual_count = int(binary_df["is_virtual_inactive"].sum())
experimental_inactive_count = int(
    (binary_df["inactive_source"] == "experimental").sum()
)

print("\n14I. Verification counts:")
print("Active peptides:", active_count)
print("Inactive peptides:", inactive_count)
print("Experimental inactive:", experimental_inactive_count)
print("Virtual inactive:", virtual_count)
print("Total peptides:", len(binary_df))

assert len(binary_df) == 901, "Expected 901 total peptides."
assert active_count == 99, "Expected 99 Active peptides."
assert inactive_count == 802, "Expected 802 Inactive peptides."
assert experimental_inactive_count == 52, (
    "Expected 52 experimental inactive peptides."
)
assert virtual_count == 750, "Expected 750 virtual inactive peptides."
print("\nAll Step 14 count checks passed.")

binary_df = binary_df[
    [
        "ID",
        "sequence",
        "class",
        "original_class",
        "label",
        "binary_class",
        "inactive_source",
        "is_virtual_inactive",
    ]
]

output_file = PROJECT_DIR / "derived" / "lung_binary_labeled.csv"
binary_df.to_csv(output_file, index=False)
print("\n14L. New labeled dataset saved to:")
print(output_file)
print("\n" + "=" * 70)
print("STEP 14 COMPLETED SUCCESSFULLY")
print("=" * 70)
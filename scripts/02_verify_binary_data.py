from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
RAW_FILE = PROJECT_DIR / "data" / "ACPs_Lung_cancer.csv"
BINARY_FILE = PROJECT_DIR / "derived" / "lung_binary_labeled.csv"

raw = pd.read_csv(RAW_FILE)
binary = pd.read_csv(BINARY_FILE)

print("=" * 70)
print("STEP 15 - VERIFY BINARY DATASET INTEGRITY")
print("=" * 70)
print("Raw dataset:", RAW_FILE)
print("Binary dataset:", BINARY_FILE)

active_count = int((binary["label"] == 1).sum())
inactive_count = int((binary["label"] == 0).sum())
virtual_count = int(binary["is_virtual_inactive"].sum())

ids_identical = raw["ID"].equals(binary["ID"])
sequences_identical = raw["sequence"].equals(binary["sequence"])
classes_identical = raw["class"].equals(binary["class"])

expected_labels = raw["class"].isin(["very active", "mod. active"]).astype(int)
label_mismatches = int((binary["label"] != expected_labels).sum())
expected_virtual = (raw["class"] == "inactive - virtual").astype(int)
virtual_mismatches = int(
    (binary["is_virtual_inactive"] != expected_virtual).sum()
)

experimental_count = int(
    binary["inactive_source"].astype(str).str.lower().eq("experimental").sum()
)
print("\n15M. Experimental inactive count:")
print(experimental_count)
if experimental_count != 52:
    raise ValueError(f"Expected 52 experimental inactive peptides, found {experimental_count}")

missing_values = int(binary.isna().sum().sum())
print("\n15N. Total missing values:")
print(missing_values)
if missing_values != 0:
    raise ValueError("STOP: Missing values found in binary dataset.")

duplicate_ids = int(binary["ID"].duplicated().sum())
duplicate_sequences = int(binary["sequence"].duplicated().sum())
print("\n15O. Duplicate check:")
print("Duplicate IDs:", duplicate_ids)
print("Duplicate sequences:", duplicate_sequences)
if duplicate_ids != 0:
    raise ValueError("STOP: Duplicate IDs found.")
if duplicate_sequences != 0:
    raise ValueError("STOP: Duplicate peptide sequences found.")

if len(binary) != 901:
    raise ValueError("Expected 901 total peptides.")
if active_count != 99:
    raise ValueError("Expected 99 Active peptides.")
if inactive_count != 802:
    raise ValueError("Expected 802 Inactive peptides.")
if virtual_count != 750:
    raise ValueError("Expected 750 virtual inactive peptides.")
if not ids_identical or not sequences_identical or not classes_identical:
    raise ValueError("Raw identifying columns were not preserved.")
if label_mismatches != 0 or virtual_mismatches != 0:
    raise ValueError("Binary label mapping does not match the raw classes.")

print("\n" + "=" * 70)
print("STEP 15 FINAL VERIFICATION SUMMARY")
print("=" * 70)
print("Total peptides:", len(binary))
print("Active peptides:", active_count)
print("Inactive peptides:", inactive_count)
print("Virtual inactive peptides:", virtual_count)
print("Experimental inactive peptides:", experimental_count)
print("Missing values:", missing_values)
print("Duplicate IDs:", duplicate_ids)
print("Duplicate sequences:", duplicate_sequences)
print("IDs preserved:", ids_identical)
print("Sequences preserved:", sequences_identical)
print("Original classes preserved:", classes_identical)
print("Binary-label mapping errors:", label_mismatches)
print("Virtual-indicator errors:", virtual_mismatches)
print("\nSTEP 15 COMPLETED SUCCESSFULLY")
print("=" * 70)
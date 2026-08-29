from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
INPUT_FILE = PROJECT_DIR / "derived" / "traditional_features.csv"
OUTPUT_FILE = PROJECT_DIR / "results" / "step24_physicochemical_statistics.csv"
MANUSCRIPT_OUTPUT = (
    PROJECT_DIR / "results" / "step24_physicochemical_statistics_manuscript.csv"
)

df = pd.read_csv(INPUT_FILE)
features = [
    "length",
    "molecular_weight",
    "net_charge_pH7_4",
    "isoelectric_point",
    "mean_eisenberg_hydrophobicity",
    "hydrophobic_moment",
    "boman_index",
]
display_names = {
    "length": "Peptide length",
    "molecular_weight": "Molecular weight",
    "net_charge_pH7_4": "Net charge at pH 7.4",
    "isoelectric_point": "Isoelectric point",
    "mean_eisenberg_hydrophobicity": "Mean Eisenberg hydrophobicity",
    "hydrophobic_moment": "Hydrophobic moment",
    "boman_index": "Boman index",
}

active = df.loc[df["binary_class"] == "Active"]
inactive = df.loc[df["binary_class"] == "Inactive"]
active_n = len(active)
inactive_n = len(inactive)
tested_values = df[features].apply(pd.to_numeric, errors="coerce")
missing_values = int(tested_values.isna().sum().sum())
nonfinite_values = int((~np.isfinite(tested_values.to_numpy())).sum())
if missing_values != 0 or nonfinite_values != 0:
    raise ValueError("Tested descriptors contain missing or non-finite values.")

rows = []
for feature in features:
    active_values = active[feature].to_numpy(dtype=float)
    inactive_values = inactive[feature].to_numpy(dtype=float)
    test = mannwhitneyu(active_values, inactive_values, alternative="two-sided")
    u_statistic = float(test.statistic)
    raw_p_value = float(test.pvalue)
    rank_biserial_effect = 2 * u_statistic / (active_n * inactive_n) - 1
    absolute_effect = abs(rank_biserial_effect)
    if absolute_effect < 0.10:
        effect_magnitude = "Negligible"
    elif absolute_effect < 0.30:
        effect_magnitude = "Small"
    elif absolute_effect < 0.50:
        effect_magnitude = "Moderate"
    else:
        effect_magnitude = "Large"
    if rank_biserial_effect > 0:
        direction = "Active higher"
    elif rank_biserial_effect < 0:
        direction = "Active lower"
    else:
        direction = "No difference"
    rows.append(
        {
            "feature": feature,
            "display_name": display_names[feature],
            "active_n": active_n,
            "inactive_n": inactive_n,
            "active_median": float(np.median(active_values)),
            "inactive_median": float(np.median(inactive_values)),
            "active_mean": float(np.mean(active_values)),
            "inactive_mean": float(np.mean(inactive_values)),
            "u_statistic": u_statistic,
            "raw_p_value": raw_p_value,
            "rank_biserial_effect": rank_biserial_effect,
            "effect_magnitude": effect_magnitude,
            "direction": direction,
        }
    )

results_df = pd.DataFrame(rows)
ordered = np.argsort(results_df["raw_p_value"].to_numpy())
fdr_values = np.empty(len(results_df), dtype=float)
running_minimum = 1.0
for rank in range(len(ordered) - 1, -1, -1):
    index = ordered[rank]
    adjusted = results_df.loc[index, "raw_p_value"] * len(results_df) / (rank + 1)
    running_minimum = min(running_minimum, adjusted)
    fdr_values[index] = min(running_minimum, 1.0)
results_df["fdr_p_value"] = fdr_values
results_df["significance_symbol"] = results_df["fdr_p_value"].map(
    lambda value: "****"
    if value < 0.0001
    else "***"
    if value < 0.001
    else "**"
    if value < 0.01
    else "*"
    if value < 0.05
    else "ns"
)
results_df = results_df[
    [
        "feature",
        "display_name",
        "active_n",
        "inactive_n",
        "active_median",
        "inactive_median",
        "active_mean",
        "inactive_mean",
        "u_statistic",
        "raw_p_value",
        "fdr_p_value",
        "rank_biserial_effect",
        "effect_magnitude",
        "direction",
        "significance_symbol",
    ]
]
results_df.to_csv(OUTPUT_FILE, index=False)

manuscript_df = results_df[
    [
        "display_name",
        "active_n",
        "inactive_n",
        "active_median",
        "inactive_median",
        "raw_p_value",
        "fdr_p_value",
        "rank_biserial_effect",
        "effect_magnitude",
        "direction",
        "significance_symbol",
    ]
].copy()
manuscript_df.to_csv(MANUSCRIPT_OUTPUT, index=False)

significant_count = int((results_df["fdr_p_value"] < 0.05).sum())
print("\n" + "=" * 76)
print("STEP 24 - ACTIVE VS INACTIVE STATISTICAL COMPARISON")
print("=" * 76)
print(results_df[
    [
        "display_name",
        "active_median",
        "inactive_median",
        "fdr_p_value",
        "rank_biserial_effect",
        "effect_magnitude",
        "direction",
        "significance_symbol",
    ]
].to_string(index=False))
print("\n24O. Statistical table saved:")
print(OUTPUT_FILE)
print("\nConcise statistical table saved:")
print(MANUSCRIPT_OUTPUT)
print("\n" + "=" * 76)
print("STEP 24 SUMMARY")
print("=" * 76)
print("Total peptides:", len(df))
print("Active peptides:", active_n)
print("Inactive peptides:", inactive_n)
print("Descriptors tested:", len(features))
print("Missing tested values:", missing_values)
print("Non-finite tested values:", nonfinite_values)
print("FDR-significant descriptors:", significant_count)
print("\nStatistical test: Mann-Whitney U, two-sided")
print("Multiple-testing correction: Benjamini-Hochberg FDR")
print("Effect size: Rank-biserial correlation")
print("\nMain results:")
print(OUTPUT_FILE)
print("\nManuscript-style results:")
print(MANUSCRIPT_OUTPUT)
print("\nSTEP 24 COMPLETED SUCCESSFULLY")
print("=" * 76)
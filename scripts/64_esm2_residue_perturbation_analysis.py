from pathlib import Path
import hashlib
import os

PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
CACHE_DIR = PROJECT_DIR / "models" / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EMBEDDING_DIMENSION = 1280
BATCH_SIZE = 64
THRESHOLD = 0.5
HARD_IDS = [48, 40, 145, 56, 68]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

RANKING_INPUT = RESULTS_DIR / "step59_consensus_hard_case_ranking.csv"
METADATA_INPUT = PROJECT_DIR / "derived" / "esm2_embedding_metadata.csv"
EMBEDDING_INPUT = PROJECT_DIR / "derived" / "esm2_embeddings.npy"
RF_MODEL_INPUT = RESULTS_DIR / "step50_esm2_random_forest_model.joblib"
XGB_MODEL_INPUT = RESULTS_DIR / "step51_esm2_xgboost_model.joblib"
RF_PRED_INPUT = RESULTS_DIR / "step50_esm2_random_forest_test_predictions.csv"
XGB_PRED_INPUT = RESULTS_DIR / "step51_esm2_xgboost_test_predictions.csv"

DETAIL_OUTPUT = RESULTS_DIR / "step82_residue_perturbation_details.csv"
CONSENSUS_OUTPUT = RESULTS_DIR / "step82_residue_consensus_importance.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "step82_peptide_perturbation_summary.csv"
QC_OUTPUT = RESULTS_DIR / "step82_residue_perturbation_qc.csv"
SELECTED_OUTPUT = RESULTS_DIR / "step82_selected_peptides.csv"
FIG1_PNG = FIGURES_DIR / "Step82_Hard_Case_Residue_Sensitivity.png"
FIG1_PDF = FIGURES_DIR / "Step82_Hard_Case_Residue_Sensitivity.pdf"
FIG2_PNG = FIGURES_DIR / "Step82_Hard_vs_Correct_Residue_Sensitivity.png"
FIG2_PDF = FIGURES_DIR / "Step82_Hard_vs_Correct_Residue_Sensitivity.pdf"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def residue_category(residue):
    if residue in "KRH":
        return "Basic"
    if residue in "DE":
        return "Acidic"
    if residue in "AVILMFWY":
        return "Hydrophobic"
    return "Polar/other"


def true_probability(probability_active, label):
    return probability_active if label == 1 else 1.0 - probability_active


def embed_sequences(sequences, tokenizer, language_model, device):
    batches = []
    for start in range(0, len(sequences), BATCH_SIZE):
        current = sequences[start:start + BATCH_SIZE]
        encoded = tokenizer(current, return_tensors="pt", add_special_tokens=True,
                            padding=True, truncation=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = language_model(**encoded).last_hidden_state
            pooled = []
            for row, sequence in enumerate(current):
                valid_tokens = int(encoded["attention_mask"][row].sum().item())
                if valid_tokens != len(sequence) + 2:
                    raise AssertionError("Tokenizer residue/special-token length mismatch")
                residues = hidden[row, 1:valid_tokens - 1, :]
                if tuple(residues.shape) != (len(sequence), EMBEDDING_DIMENSION):
                    raise AssertionError("Unexpected residue embedding shape")
                pooled.append(residues.mean(dim=0))
        pooled_np = torch.stack(pooled).float().cpu().numpy().astype(np.float32)
        if pooled_np.shape != (len(current), EMBEDDING_DIMENSION):
            raise AssertionError("Unexpected pooled embedding shape")
        if not np.isfinite(pooled_np).all():
            raise AssertionError("Non-finite pooled embedding")
        batches.append(pooled_np)
        del encoded, hidden, pooled
    return np.vstack(batches)


print("=" * 104)
print("STEP 82 - RESIDUE-LEVEL PERTURBATION ANALYSIS OF ESM-2 PREDICTIONS")
print("=" * 104)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ranking = pd.read_csv(RANKING_INPUT)
metadata = pd.read_csv(METADATA_INPUT)
step44 = np.load(EMBEDDING_INPUT, allow_pickle=False)
rf_predictions = pd.read_csv(RF_PRED_INPUT)
xgb_predictions = pd.read_csv(XGB_PRED_INPUT)

assert len(ranking) == 181 and ranking["ID"].is_unique
hard = ranking.loc[ranking["ID"].isin(HARD_IDS)].copy()
assert hard["ID"].tolist() == HARD_IDS
assert hard["total_wrong_count"].eq(8).all()
controls = (ranking.loc[ranking["total_wrong_count"].eq(0)]
            .sort_values(["all_models_mean_true_class_probability", "ID"],
                         ascending=[False, True]).head(5).copy())
assert len(hard) == len(controls) == 5
hard["analysis_group"] = "Consensus hard error"
controls["analysis_group"] = "High-confidence consensus correct"
selected = pd.concat([hard, controls], ignore_index=True)
selected["panel_order"] = np.arange(1, 11)
selected["sequence_length"] = selected["sequence"].str.len()
selected["selection_rule"] = np.where(
    selected["analysis_group"].eq("Consensus hard error"),
    "Predefined Step-59 8/8 error ID", "Top five all-model true-class probabilities among 0/8 errors")
assert selected["ID"].is_unique and len(selected) == 10
assert selected["sequence"].map(lambda s: set(s) <= STANDARD_AA).all()

alignment = metadata.set_index("ID").loc[selected["ID"]]
assert alignment["sequence"].tolist() == selected["sequence"].tolist()
assert alignment["label"].astype(int).tolist() == selected["y_true"].astype(int).tolist()
assert alignment["split"].eq("test").all()
step44_selected = step44[alignment["embedding_row"].to_numpy(dtype=int)]

for predictions in (rf_predictions, xgb_predictions):
    aligned = predictions.set_index("ID").loc[selected["ID"]]
    assert aligned["sequence"].tolist() == selected["sequence"].tolist()
    assert aligned["label"].astype(int).tolist() == selected["y_true"].astype(int).tolist()

rf_model = joblib.load(RF_MODEL_INPUT)
xgb_model = joblib.load(XGB_MODEL_INPUT)
assert rf_model.n_features_in_ == xgb_model.n_features_in_ == EMBEDDING_DIMENSION

if not torch.cuda.is_available():
    raise RuntimeError("Step 82 requires the confirmed CUDA environment")
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
language_model = AutoModel.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True, dtype=torch.float16)
language_model.eval().to(device)
assert next(language_model.parameters()).device.type == "cuda"
assert next(language_model.parameters()).dtype == torch.float16
assert int(language_model.config.hidden_size) == EMBEDDING_DIMENSION
print("ESM-2 loaded from local cache:", MODEL_NAME)
print("Network required: False")
print("Device/dtype:", device, next(language_model.parameters()).dtype)

original_sequences = selected["sequence"].tolist()
original_embeddings = embed_sequences(original_sequences, tokenizer, language_model, device)
numerator = np.sum(original_embeddings * step44_selected, axis=1, dtype=np.float64)
denominator = (np.linalg.norm(original_embeddings.astype(np.float64), axis=1) *
               np.linalg.norm(step44_selected.astype(np.float64), axis=1))
cosine = numerator / denominator
assert np.isfinite(cosine).all() and (cosine > 0.999).all()

rf_original = rf_model.predict_proba(original_embeddings)[:, 1]
xgb_original = xgb_model.predict_proba(original_embeddings)[:, 1]
rf_frozen = rf_predictions.set_index("ID").loc[selected["ID"], "predicted_probability"].to_numpy(float)
xgb_frozen = xgb_predictions.set_index("ID").loc[selected["ID"], "predicted_probability"].to_numpy(float)
rf_frozen_label = rf_predictions.set_index("ID").loc[selected["ID"], "predicted_label"].to_numpy(int)
xgb_frozen_label = xgb_predictions.set_index("ID").loc[selected["ID"], "predicted_label"].to_numpy(int)
rf_regen_label = (rf_original >= THRESHOLD).astype(int)
xgb_regen_label = (xgb_original >= THRESHOLD).astype(int)
if not np.array_equal(rf_regen_label, rf_frozen_label):
    raise RuntimeError("Regenerated RF prediction crossed the 0.5 threshold")
if not np.array_equal(xgb_regen_label, xgb_frozen_label):
    raise RuntimeError("Regenerated XGBoost prediction crossed the 0.5 threshold")

selected["step44_embedding_cosine_similarity"] = cosine
selected["rf_frozen_probability_active"] = rf_frozen
selected["rf_regenerated_probability_active"] = rf_original
selected["rf_absolute_probability_discrepancy"] = np.abs(rf_original - rf_frozen)
selected["xgboost_frozen_probability_active"] = xgb_frozen
selected["xgboost_regenerated_probability_active"] = xgb_original
selected["xgboost_absolute_probability_discrepancy"] = np.abs(xgb_original - xgb_frozen)
selected["regenerated_prediction_threshold_crossings"] = 0

# Generate one mutant for every non-alanine position, retaining explicit alanine rows later.
mutant_records = []
for peptide_index, row in selected.iterrows():
    sequence = row["sequence"]
    for position, residue in enumerate(sequence, start=1):
        if residue == "A":
            continue
        mutant = sequence[:position - 1] + "A" + sequence[position:]
        assert len(mutant) == len(sequence)
        assert sum(a != b for a, b in zip(sequence, mutant)) == 1
        assert set(mutant) <= STANDARD_AA
        mutant_records.append({"peptide_index": peptide_index, "position": position,
                               "mutant_sequence": mutant})
mutants = pd.DataFrame(mutant_records)
mutant_embeddings = embed_sequences(mutants["mutant_sequence"].tolist(), tokenizer,
                                    language_model, device)
mutants["rf_probability_active"] = rf_model.predict_proba(mutant_embeddings)[:, 1]
mutants["xgboost_probability_active"] = xgb_model.predict_proba(mutant_embeddings)[:, 1]
assert np.isfinite(mutants[["rf_probability_active", "xgboost_probability_active"]]).all().all()
assert mutants[["rf_probability_active", "xgboost_probability_active"]].ge(0).all().all()
assert mutants[["rf_probability_active", "xgboost_probability_active"]].le(1).all().all()
mutant_lookup = mutants.set_index(["peptide_index", "position"])

detail_rows = []
for peptide_index, row in selected.iterrows():
    sequence = row["sequence"]
    label = int(row["y_true"])
    for position, residue in enumerate(sequence, start=1):
        performed = residue != "A"
        for classifier, original_active, frozen_active, mutant_column in (
            ("ESM-2 Random Forest", rf_original[peptide_index], rf_frozen[peptide_index], "rf_probability_active"),
            ("ESM-2 XGBoost", xgb_original[peptide_index], xgb_frozen[peptide_index], "xgboost_probability_active"),
        ):
            mutant_active = (float(mutant_lookup.loc[(peptide_index, position), mutant_column])
                             if performed else np.nan)
            original_true = true_probability(float(original_active), label)
            mutant_true = true_probability(mutant_active, label) if performed else np.nan
            delta_active = mutant_active - original_active if performed else np.nan
            delta_true = mutant_true - original_true if performed else np.nan
            mutant_prediction = int(mutant_active >= THRESHOLD) if performed else np.nan
            detail_rows.append({
                "peptide_ID": row["ID"], "sequence": sequence,
                "true_class": row["true_class"], "label": label,
                "analysis_group": row["analysis_group"], "position": position,
                "original_residue": residue, "residue_category": residue_category(residue),
                "mutant_residue": "A", "original_is_alanine": residue == "A",
                "perturbation_performed": performed, "classifier": classifier,
                "frozen_original_probability_active": frozen_active,
                "original_probability_active": original_active,
                "mutant_probability_active": mutant_active,
                "delta_probability_active": delta_active,
                "original_true_class_probability": original_true,
                "mutant_true_class_probability": mutant_true,
                "delta_true_class_probability": delta_true,
                "absolute_delta_true_class_probability": abs(delta_true) if performed else np.nan,
                "original_prediction": int(original_active >= THRESHOLD),
                "mutant_prediction": mutant_prediction,
                "prediction_flip": bool(mutant_prediction != int(original_active >= THRESHOLD)) if performed else False,
                "step44_embedding_cosine_similarity": cosine[peptide_index],
            })

details = pd.DataFrame(detail_rows)
details["importance_rank"] = np.nan
performed_mask = details["perturbation_performed"]
details.loc[performed_mask, "importance_rank"] = (
    details.loc[performed_mask].groupby(["peptide_ID", "classifier"])
    ["absolute_delta_true_class_probability"].rank(method="first", ascending=False))

consensus_rows = []
for (peptide_id, position), group in details.groupby(["peptide_ID", "position"], sort=False):
    first = group.iloc[0]
    performed = bool(first["perturbation_performed"])
    values = group["absolute_delta_true_class_probability"].to_numpy(float)
    consensus_rows.append({
        "peptide_ID": peptide_id, "sequence": first["sequence"],
        "true_class": first["true_class"], "label": first["label"],
        "analysis_group": first["analysis_group"], "position": position,
        "original_residue": first["original_residue"],
        "residue_category": first["residue_category"],
        "original_is_alanine": first["original_is_alanine"],
        "perturbation_performed": performed,
        "rf_absolute_delta_true_class_probability": values[0] if performed else np.nan,
        "xgboost_absolute_delta_true_class_probability": values[1] if performed else np.nan,
        "consensus_importance": np.nanmean(values) if performed else np.nan,
        "rf_prediction_flip": bool(group.iloc[0]["prediction_flip"]),
        "xgboost_prediction_flip": bool(group.iloc[1]["prediction_flip"]),
        "any_prediction_flip": bool(group["prediction_flip"].any()),
    })
consensus = pd.DataFrame(consensus_rows)
consensus["consensus_importance_rank"] = np.nan
cmask = consensus["perturbation_performed"]
consensus.loc[cmask, "consensus_importance_rank"] = (
    consensus.loc[cmask].groupby("peptide_ID")["consensus_importance"]
    .rank(method="first", ascending=False))

summary_rows = []
for (peptide_id, classifier), group in details.groupby(["peptide_ID", "classifier"], sort=False):
    valid = group.loc[group["perturbation_performed"]]
    maximum = valid.sort_values(["absolute_delta_true_class_probability", "position"],
                                ascending=[False, True]).iloc[0]
    summary_rows.append({
        "peptide_ID": peptide_id, "sequence": group.iloc[0]["sequence"],
        "true_class": group.iloc[0]["true_class"], "analysis_group": group.iloc[0]["analysis_group"],
        "classifier": classifier, "mean_absolute_residue_sensitivity": valid["absolute_delta_true_class_probability"].mean(),
        "median_absolute_residue_sensitivity": valid["absolute_delta_true_class_probability"].median(),
        "maximum_absolute_residue_sensitivity": maximum["absolute_delta_true_class_probability"],
        "maximum_sensitivity_position": int(maximum["position"]),
        "maximum_sensitivity_residue": maximum["original_residue"],
        "number_of_perturbations": len(valid), "number_of_prediction_flips": int(valid["prediction_flip"].sum()),
        "fraction_of_perturbations_causing_flips": valid["prediction_flip"].mean(),
    })
for peptide_id, group in consensus.groupby("peptide_ID", sort=False):
    valid = group.loc[group["perturbation_performed"]]
    maximum = valid.sort_values(["consensus_importance", "position"], ascending=[False, True]).iloc[0]
    summary_rows.append({
        "peptide_ID": peptide_id, "sequence": group.iloc[0]["sequence"],
        "true_class": group.iloc[0]["true_class"], "analysis_group": group.iloc[0]["analysis_group"],
        "classifier": "RF/XGBoost consensus", "mean_absolute_residue_sensitivity": valid["consensus_importance"].mean(),
        "median_absolute_residue_sensitivity": valid["consensus_importance"].median(),
        "maximum_absolute_residue_sensitivity": maximum["consensus_importance"],
        "maximum_sensitivity_position": int(maximum["position"]),
        "maximum_sensitivity_residue": maximum["original_residue"],
        "number_of_perturbations": len(valid), "number_of_prediction_flips": int(valid["any_prediction_flip"].sum()),
        "fraction_of_perturbations_causing_flips": valid["any_prediction_flip"].mean(),
    })
summary = pd.DataFrame(summary_rows)

selected_columns = ["panel_order", "ID", "sequence", "sequence_length", "true_class", "y_true",
                    "analysis_group", "total_wrong_count", "all_models_mean_true_class_probability",
                    "selection_rule", "step44_embedding_cosine_similarity",
                    "rf_frozen_probability_active", "rf_regenerated_probability_active",
                    "rf_absolute_probability_discrepancy", "xgboost_frozen_probability_active",
                    "xgboost_regenerated_probability_active", "xgboost_absolute_probability_discrepancy",
                    "regenerated_prediction_threshold_crossings"]
selected[selected_columns].to_csv(SELECTED_OUTPUT, index=False)
details.to_csv(DETAIL_OUTPUT, index=False)
consensus.to_csv(CONSENSUS_OUTPUT, index=False)
summary.to_csv(SUMMARY_OUTPUT, index=False)

# Figure 1: sequence-cell sensitivity tracks for the five predefined hard errors.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
hard_consensus = consensus.loc[consensus["analysis_group"].eq("Consensus hard error")]
vmax = float(hard_consensus["consensus_importance"].max())
fig, axes = plt.subplots(5, 1, figsize=(12.5, 6.8), facecolor="white")
for ax, peptide_id in zip(axes, HARD_IDS):
    row = hard_consensus.loc[hard_consensus["peptide_ID"].eq(peptide_id)].sort_values("position")
    for j, item in enumerate(row.itertuples(index=False)):
        value = item.consensus_importance if item.perturbation_performed else 0.0
        color = plt.cm.YlOrRd(value / vmax if vmax else 0)
        hatch = "///" if item.original_is_alanine else None
        linewidth = 2.2 if item.any_prediction_flip else 0.7
        ax.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=color, edgecolor="black",
                                   linewidth=linewidth, hatch=hatch))
        label = item.original_residue + ("*" if item.any_prediction_flip else "")
        ax.text(j + 0.5, 0.5, label, ha="center", va="center", fontsize=9,
                fontweight="bold" if item.any_prediction_flip else "normal")
    ax.set_xlim(0, len(row)); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xticks(np.arange(len(row)) + 0.5, row["position"].astype(str))
    ax.set_ylabel(f"ID {peptide_id}\n{row.iloc[0]['true_class']}", rotation=0,
                  ha="right", va="center", labelpad=35)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_facecolor("white")
axes[-1].set_xlabel("Residue position")
fig.suptitle("A  Consensus hard-case residue sensitivity to alanine substitution",
             fontsize=13, fontweight="bold", y=0.99)
cax = fig.add_axes([0.91, 0.20, 0.014, 0.60])
fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="YlOrRd"), cax=cax,
             label="Consensus mean |Δ true-class probability|")
fig.text(0.5, 0.012, "Asterisk/thick border: at least one classifier prediction flipped; hatched A: no substitution performed.",
         ha="center", fontsize=8.3)
fig.tight_layout(rect=(0.08, 0.045, 0.90, 0.95), h_pad=0.65)
fig.savefig(FIG1_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIG1_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: descriptive peptide summaries for consensus sensitivity.
plot_summary = summary.loc[summary["classifier"].eq("RF/XGBoost consensus")].copy()
plot_summary["short_group"] = plot_summary["analysis_group"].map({
    "Consensus hard error": "Hard error", "High-confidence consensus correct": "Consensus correct"})
colors = {"Hard error": "#D55E00", "Consensus correct": "#0072B2"}
metrics = [("mean_absolute_residue_sensitivity", "Mean |Δ true-class probability|"),
           ("maximum_absolute_residue_sensitivity", "Maximum |Δ true-class probability|"),
           ("number_of_prediction_flips", "Positions flipping ≥1 classifier")]
fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), facecolor="white")
for panel, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
    for x, group_name in enumerate(["Hard error", "Consensus correct"]):
        values = plot_summary.loc[plot_summary["short_group"].eq(group_name), metric].to_numpy(float)
        offsets = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(np.full(len(values), x) + offsets, values, s=54, color=colors[group_name],
                   edgecolor="black", linewidth=0.6, zorder=3, label=group_name if panel == 0 else None)
        ax.hlines(values.mean(), x - 0.22, x + 0.22, color="black", linewidth=2.0, zorder=4)
    ax.set_xticks([0, 1], ["Hard error\n(n=5)", "Consensus correct\n(n=5)"])
    ax.set_ylabel(ylabel); ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.12, 1.04, chr(65 + panel), transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_facecolor("white")
fig.suptitle("Hard-error versus high-confidence correct peptide sensitivity",
             fontsize=13, fontweight="bold", y=0.99)
fig.text(0.5, 0.012, "Points are individual peptides; horizontal lines are group means. Descriptive only—no significance tests.",
         ha="center", fontsize=8.3)
fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.94), w_pad=2.0)
fig.savefig(FIG2_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIG2_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

expected_positions = int(selected["sequence_length"].sum())
alanine_positions = int(sum(sequence.count("A") for sequence in original_sequences))
performed_positions = expected_positions - alanine_positions
assert len(details) == expected_positions * 2
assert len(consensus) == expected_positions
assert int(details["perturbation_performed"].sum()) == performed_positions * 2
assert details.loc[~details["perturbation_performed"],
                   ["mutant_probability_active", "delta_probability_active",
                    "mutant_true_class_probability", "delta_true_class_probability",
                    "absolute_delta_true_class_probability"]].isna().all().all()

qc = pd.DataFrame([{
    "selected_peptides": len(selected), "consensus_hard_peptides": len(hard),
    "consensus_correct_peptides": len(controls), "selected_active": int(selected["y_true"].sum()),
    "selected_inactive": int((selected["y_true"] == 0).sum()),
    "total_residue_positions": expected_positions, "alanine_positions_unperturbed": alanine_positions,
    "unique_mutants_generated": len(mutants), "detail_rows": len(details),
    "consensus_rows": len(consensus), "summary_rows": len(summary),
    "esm2_model": MODEL_NAME, "model_loaded_local_cache": True,
    "network_required": False, "device": str(device), "model_dtype": str(next(language_model.parameters()).dtype),
    "embedding_dimension": EMBEDDING_DIMENSION, "all_embeddings_finite": True,
    "minimum_step44_cosine_similarity": cosine.min(),
    "maximum_rf_probability_discrepancy": np.max(np.abs(rf_original - rf_frozen)),
    "maximum_xgboost_probability_discrepancy": np.max(np.abs(xgb_original - xgb_frozen)),
    "rf_regenerated_label_matches": bool(np.array_equal(rf_regen_label, rf_frozen_label)),
    "xgboost_regenerated_label_matches": bool(np.array_equal(xgb_regen_label, xgb_frozen_label)),
    "regenerated_threshold_crossings": 0,
    "all_probabilities_finite_and_bounded": True,
    "mutant_lengths_preserved": True, "single_residue_changes_only": True,
    "standard_amino_acids_only": True, "rf_model_retrained": False,
    "xgboost_model_retrained": False, "model_selection_performed": False,
    "threshold_optimization_performed": False, "locked_test_labels_interpretation_only": True,
    "qc_passed": True,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nSelected panel:")
print(selected[selected_columns].to_string(index=False))
print("\nPerturbations:", performed_positions, "unique mutants;")
print("alanine positions retained as unperturbed:", alanine_positions)
print("minimum original/Step-44 cosine similarity:", round(float(cosine.min()), 9))
print("maximum RF/XGBoost probability discrepancy:",
      round(float(np.max(np.abs(rf_original - rf_frozen))), 9), "/",
      round(float(np.max(np.abs(xgb_original - xgb_frozen))), 9))
print("\nOutputs:")
for path in (SELECTED_OUTPUT, DETAIL_OUTPUT, CONSENSUS_OUTPUT, SUMMARY_OUTPUT, QC_OUTPUT,
             FIG1_PNG, FIG1_PDF, FIG2_PNG, FIG2_PDF):
    print(path, path.stat().st_size, "bytes", sha256(path))
print("\nSTEP 82 COMPLETED SUCCESSFULLY")
print("=" * 104)

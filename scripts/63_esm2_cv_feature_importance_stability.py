from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


PROJECT_DIR = Path(r"E:\postdoc-work\ist-project")
DERIVED_DIR = PROJECT_DIR / "derived"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

EMBEDDING_INPUT = DERIVED_DIR / "esm2_embeddings.npy"
METADATA_INPUT = DERIVED_DIR / "esm2_embedding_metadata.csv"
CV_INPUT = DERIVED_DIR / "fixed_cv_folds.npz"

DETAIL_OUTPUT = RESULTS_DIR / "step81_esm2_cv_feature_importance_fold_details.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "step81_esm2_cv_feature_importance_summary.csv"
OVERLAP_OUTPUT = RESULTS_DIR / "step81_esm2_feature_importance_overlap.csv"
QC_OUTPUT = RESULTS_DIR / "step81_esm2_feature_importance_qc.csv"
PNG_OUTPUT = FIGURES_DIR / "Step81_ESM2_Feature_Importance_Stability.png"
PDF_OUTPUT = FIGURES_DIR / "Step81_ESM2_Feature_Importance_Stability.pdf"

SEED = 2026
N_FOLDS = 5
N_FEATURES = 1280


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def xgb_normalized_gain(model, n_features):
    raw = model.get_booster().get_score(importance_type="gain")
    values = np.zeros(n_features, dtype=np.float64)
    for key, value in raw.items():
        index = int(key[1:]) if key.startswith("f") else int(key)
        values[index] = float(value)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise AssertionError("XGBoost fold has no positive finite gain importance")
    return values / total


print("=" * 100)
print("STEP 81 - STABLE ESM-2 FEATURE IMPORTANCE ACROSS CV FOLDS")
print("=" * 100)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(EMBEDDING_INPUT, allow_pickle=False)
metadata = pd.read_csv(METADATA_INPUT)
cv = np.load(CV_INPUT, allow_pickle=False)
y = metadata["label"].to_numpy(dtype=int)
development_indices = np.flatnonzero(metadata["split"].eq("development").to_numpy())
test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
features = np.array([f"esm2_{i:04d}" for i in range(1, N_FEATURES + 1)])

assert X.shape == (901, N_FEATURES)
assert X.dtype == np.float32 and np.isfinite(X).all()
assert len(metadata) == 901
assert np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(901))
assert len(development_indices) == 720 and len(test_indices) == 181
assert np.array_equal(cv["development_global_indices"], development_indices)
assert np.intersect1d(development_indices, test_indices).size == 0
assert features[0] == "esm2_0001" and features[-1] == "esm2_1280"

print("Embedding matrix:", X.shape, X.dtype)
print("Development/test rows:", len(development_indices), "/", len(test_indices))
print("Locked-test rows used for fitting or importance: 0")
print("scikit-learn/xgboost:", sklearn.__version__, "/", xgboost.__version__)

detail_parts = []
fit_audit = []
for fold in range(1, N_FOLDS + 1):
    train_idx = cv[f"fold{fold}_train"]
    valid_idx = cv[f"fold{fold}_valid"]
    assert len(train_idx) == 576 and len(valid_idx) == 144
    assert np.intersect1d(train_idx, valid_idx).size == 0
    assert np.intersect1d(train_idx, test_indices).size == 0
    assert np.intersect1d(valid_idx, test_indices).size == 0
    assert np.isin(train_idx, development_indices).all()

    models = [
        ("ESM-2 Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )),
        ("ESM-2 XGBoost", XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.6, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=float((y[train_idx] == 0).sum() / (y[train_idx] == 1).sum()),
            random_state=SEED, n_jobs=-1,
        )),
    ]

    for model_name, model in models:
        print(f"Fitting fold {fold}: {model_name}")
        model.fit(X[train_idx], y[train_idx])
        assert model.n_features_in_ == N_FEATURES
        if model_name == "ESM-2 Random Forest":
            importance = np.asarray(model.feature_importances_, dtype=np.float64)
            importance_definition = "mean decrease in impurity, normalized"
        else:
            importance = xgb_normalized_gain(model, N_FEATURES)
            importance_definition = "gain, normalized within fold"

        assert importance.shape == (N_FEATURES,)
        assert np.isfinite(importance).all() and (importance >= 0).all()
        assert np.isclose(importance.sum(), 1.0, rtol=1e-10, atol=1e-10)
        order = np.lexsort((np.arange(N_FEATURES), -importance))
        ranks = np.empty(N_FEATURES, dtype=int)
        ranks[order] = np.arange(1, N_FEATURES + 1)
        detail_parts.append(pd.DataFrame({
            "model": model_name,
            "fold": fold,
            "feature": features,
            "importance": importance,
            "rank_within_fold": ranks,
            "nonzero": importance > 0,
            "importance_definition": importance_definition,
        }))
        fit_audit.append({
            "model": model_name, "fold": fold, "train_n": len(train_idx),
            "validation_n": len(valid_idx), "input_dimensions": model.n_features_in_,
            "importance_sum": importance.sum(), "nonzero_dimensions": int((importance > 0).sum()),
            "train_validation_overlap": 0, "test_overlap": 0,
        })

details = pd.concat(detail_parts, ignore_index=True)
audit = pd.DataFrame(fit_audit)
assert details.shape[0] == 2 * N_FOLDS * N_FEATURES
assert details.groupby(["model", "fold"]).size().eq(N_FEATURES).all()
assert details.groupby("model").size().eq(N_FOLDS * N_FEATURES).all()
assert audit.shape[0] == 10

summary_rows = []
for (model_name, feature), group in details.groupby(["model", "feature"], sort=False):
    ranks = group["rank_within_fold"].to_numpy()
    values = group["importance"].to_numpy()
    summary_rows.append({
        "model": model_name,
        "feature": feature,
        "mean_importance": values.mean(),
        "median_importance": np.median(values),
        "sd_importance": values.std(ddof=1),
        "mean_rank": ranks.mean(),
        "median_rank": np.median(ranks),
        "number_of_folds_nonzero": int(group["nonzero"].sum()),
        "number_of_folds_top10": int((ranks <= 10).sum()),
        "number_of_folds_top20": int((ranks <= 20).sum()),
        "number_of_folds_top50": int((ranks <= 50).sum()),
        "stable_top20": bool((ranks <= 20).sum() >= 3),
        "stable_top50": bool((ranks <= 50).sum() >= 4),
    })

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values(
    ["model", "mean_importance", "feature"], ascending=[True, False, True]
).reset_index(drop=True)
summary["rank_by_mean_importance"] = summary.groupby("model").cumcount() + 1
assert summary.shape[0] == 2 * N_FEATURES

overlap_rows = []
for criterion in ("stable_top20", "stable_top50"):
    rf = set(summary.loc[(summary["model"] == "ESM-2 Random Forest") & summary[criterion], "feature"])
    xgb = set(summary.loc[(summary["model"] == "ESM-2 XGBoost") & summary[criterion], "feature"])
    intersection = rf & xgb
    union = rf | xgb
    overlap_rows.append({
        "stability_criterion": criterion,
        "rf_stable_count": len(rf),
        "xgboost_stable_count": len(xgb),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard_index": len(intersection) / len(union) if union else np.nan,
        "intersection_features": ";".join(sorted(intersection)),
        "rf_only_features": ";".join(sorted(rf - xgb)),
        "xgboost_only_features": ";".join(sorted(xgb - rf)),
    })
overlap = pd.DataFrame(overlap_rows)

details.to_csv(DETAIL_OUTPUT, index=False)
summary.to_csv(SUMMARY_OUTPUT, index=False)
overlap.to_csv(OVERLAP_OUTPUT, index=False)

# Publication figure: top 20 by mean CV importance, with between-fold SD.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})
fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), facecolor="white")
panels = [
    ("ESM-2 Random Forest", "A", "#0072B2", "Mean decrease in impurity"),
    ("ESM-2 XGBoost", "B", "#D55E00", "Mean normalized gain"),
]
for ax, (model_name, letter, color, xlabel) in zip(axes, panels):
    top = summary.loc[summary["model"].eq(model_name)].nsmallest(20, "rank_by_mean_importance")
    top = top.sort_values("mean_importance", ascending=True)
    positions = np.arange(len(top))
    ax.barh(positions, top["mean_importance"], xerr=top["sd_importance"],
            color=color, alpha=0.88, edgecolor="black", linewidth=0.45,
            error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2.2})
    ax.set_yticks(positions, top["feature"])
    ax.set_xlabel(f"{xlabel} across five folds (mean ± SD)")
    ax.set_title(model_name, fontweight="bold", pad=10)
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="left", va="bottom")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")

fig.suptitle("Stability of ESM-2 latent-dimension importance across development CV folds",
             fontsize=13, fontweight="bold", color="black", y=0.995)
fig.text(0.5, 0.012,
         "Error bars show between-fold SD (n = 5 fitted fold models); latent dimensions are computational coordinates, not biological determinants.",
         ha="center", va="bottom", fontsize=8.3, color="black")
fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.965), w_pad=3.0)
fig.savefig(PNG_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

qc = pd.DataFrame([{
    "development_rows": len(development_indices),
    "locked_test_rows": len(test_indices),
    "locked_test_rows_used": 0,
    "fixed_folds": N_FOLDS,
    "train_rows_each_fold": int(audit["train_n"].min()),
    "validation_rows_each_fold": int(audit["validation_n"].min()),
    "rf_fits": int((audit["model"] == "ESM-2 Random Forest").sum()),
    "xgboost_fits": int((audit["model"] == "ESM-2 XGBoost").sum()),
    "input_dimensions_every_fit": bool(audit["input_dimensions"].eq(N_FEATURES).all()),
    "importance_sums_all_close_to_one": bool(np.allclose(audit["importance_sum"], 1.0, atol=1e-10)),
    "all_importance_finite": bool(np.isfinite(details["importance"]).all()),
    "all_importance_nonnegative": bool(details["importance"].ge(0).all()),
    "dimensions_per_model_fold": N_FEATURES,
    "rows_per_model": N_FOLDS * N_FEATURES,
    "fold_detail_rows": len(details),
    "summary_rows": len(summary),
    "overlap_rows": len(overlap),
    "train_validation_overlap": int(audit["train_validation_overlap"].sum()),
    "test_overlap": int(audit["test_overlap"].sum()),
    "hyperparameter_tuning_performed": False,
    "test_evaluation_performed": False,
    "feature_selection_fed_back": False,
    "rf_importance_definition": "feature_importances_ (normalized mean decrease in impurity)",
    "xgboost_importance_definition": "normalized gain from Booster.get_score",
    "random_state": SEED,
    "qc_passed": True,
}])
qc.to_csv(QC_OUTPUT, index=False)

print("\nStable-dimension overlap:")
print(overlap[["stability_criterion", "rf_stable_count", "xgboost_stable_count",
               "intersection_count", "union_count", "jaccard_index"]].to_string(index=False))
for model_name in ("ESM-2 Random Forest", "ESM-2 XGBoost"):
    print(f"\nTop 10 by mean importance - {model_name}:")
    print(summary.loc[summary["model"].eq(model_name),
                      ["feature", "mean_importance", "sd_importance", "number_of_folds_top20",
                       "number_of_folds_top50"]].head(10).round(6).to_string(index=False))

print("\nOutputs:")
for path in (DETAIL_OUTPUT, SUMMARY_OUTPUT, OVERLAP_OUTPUT, QC_OUTPUT, PNG_OUTPUT, PDF_OUTPUT):
    print(path, path.stat().st_size, "bytes", sha256(path))
print("\nSTEP 81 COMPLETED SUCCESSFULLY")
print("=" * 100)

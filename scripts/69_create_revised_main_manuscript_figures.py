from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT = Path(r"E:\postdoc-work\ist-project")
DERIVED = PROJECT / "derived"
RESULTS = PROJECT / "results"
OUT = PROJECT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ARCH_OUT = RESULTS / "step86b_revised_main_figure_architecture.csv"
PANEL_OUT = RESULTS / "step86b_main_figure_panel_sources.csv"
QC_OUT = RESULTS / "step86b_revised_main_figure_qc.csv"

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#777777"
LIGHT = "#E9EEF3"
BLACK = "#111111"
MODEL_COLORS = [BLUE, SKY, GREEN, ORANGE, "#332288", PURPLE, "#44AA99", VERMILION]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def read(rel: str) -> pd.DataFrame:
    path = PROJECT / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def panel(ax, letter: str, title: str) -> None:
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")
    ax.set_title(title, loc="left", fontweight="bold", pad=7)


def clean(ax, grid: str | None = "y") -> None:
    if grid:
        ax.grid(axis=grid, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def model_short(name: str) -> str:
    return (name.replace("Traditional ", "Trad. ")
                .replace("ESM-2 Logistic Regression", "ESM-2 LR")
                .replace("Traditional Logistic Regression", "Trad. LR")
                .replace("Random Forest", "RF"))


traditional = read("derived/traditional_features.csv")
split = read("derived/fixed_split.csv")
folds = read("derived/fixed_cv_folds.csv")
perf = read("results/step74_model_performance_bootstrap_summary.csv")
paired = read("results/step54_paired_bootstrap_summary.csv")
calibration = read("results/step71_calibration_metrics.csv")
decision = read("results/step73_decision_curve_model_summary.csv")
novelty = read("results/step67_sequence_novelty_performance.csv")
cvcca = read("results/step78_cv_canonical_correlations.csv")
cvfold = read("results/step78_cv_cca_fold_summary.csv")
pcassoc = read("results/step77_traditional_vs_esm2_pc_correlations.csv")
fusion = read("results/step79_fusion_model_comparison.csv")
fusion_delta = read("results/step80_fusion_vs_esm2_paired_bootstrap_summary.csv")
importance = read("results/step81_esm2_cv_feature_importance_summary.csv")
overlap = read("results/step81_esm2_feature_importance_overlap.csv")
peptide_sens = read("results/step82_peptide_perturbation_summary.csv")
residue = read("results/step83_residue_physicochemical_context.csv")
category = read("results/step83_residue_category_summary.csv")
motif = read("results/step83_motif_sensitivity_summary.csv")
hard = read("results/step59_consensus_hard_cases_manuscript.csv")
integrated = read("results/step84_integrated_hard_case_interpretability.csv")
universal = read("results/step84_universal_error_evidence_table.csv")

assert len(traditional) == 901 and len(split) == 901
assert (split["split"] == "development").sum() == 720
assert (split["split"] == "test").sum() == 181
assert len(perf) == 8 and len(hard) == 15 and len(universal) == 5
assert universal["ID"].tolist() == [48, 40, 145, 56, 68]


# -------------------------------------------------------------------------
# Figure 1: study design, peptide landscape, computational framework
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(13.2, 12.2), constrained_layout=True)
gs = fig.add_gridspec(4, 4, height_ratios=[1.05, 1, 1, 1.08])

ax = fig.add_subplot(gs[0, :])
ax.axis("off")
panel(ax, "A", "Study workflow")
workflow = ["901 peptides", "Sequence QC", "32 descriptors\n+ 1,280 ESM-2", "720 development\n181 locked test", "Fivefold CV", "4 matched\nclassifiers", "Uncertainty,\nnovelty, interpretation"]
xs = np.linspace(0.07, 0.93, len(workflow))
for i, (x, txt) in enumerate(zip(xs, workflow)):
    box = FancyBboxPatch((x - 0.055, 0.35), 0.11, 0.30, boxstyle="round,pad=0.012",
                         facecolor=[LIGHT, "#F7E7C6", "#DDEBF7", "#E2F0D9"][i % 4],
                         edgecolor=BLACK, linewidth=0.8, transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(x, 0.50, txt, ha="center", va="center", transform=ax.transAxes, fontsize=7.5)
    if i < len(workflow) - 1:
        ax.annotate("", xy=(xs[i + 1] - 0.065, 0.50), xytext=(x + 0.065, 0.50),
                    xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.1, color=GRAY))

ax = fig.add_subplot(gs[1, 0])
panel(ax, "B", "Dataset composition")
counts = pd.crosstab(split["split"], split["binary_class"]).reindex(["development", "test"])
bottom = np.zeros(2)
for cls, color in [("Inactive", SKY), ("Active", VERMILION)]:
    vals = counts[cls].to_numpy()
    ax.bar([0, 1], vals, bottom=bottom, color=color, label=cls, edgecolor="white")
    for x, v, b in zip([0, 1], vals, bottom):
        ax.text(x, b + v / 2, str(int(v)), ha="center", va="center", fontsize=8)
    bottom += vals
ax.set_xticks([0, 1], ["Development\n(n=720)", "Locked test\n(n=181)"])
ax.set_ylabel("Peptides")
ax.legend(frameon=False, loc="upper right")
clean(ax)

ax = fig.add_subplot(gs[1, 1])
panel(ax, "C", "Peptide length distribution")
for label, color, shift in [(0, SKY, -0.12), (1, VERMILION, 0.12)]:
    vals = traditional.loc[traditional["label"] == label, "length"].to_numpy()
    parts = ax.violinplot(vals, positions=[label + shift], widths=0.22, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color); body.set_edgecolor(BLACK); body.set_alpha(0.8)
    parts["cmedians"].set_color(BLACK)
ax.set_xticks([-0.12, 1.12], ["Inactive\n(n=802)", "Active\n(n=99)"])
ax.set_ylabel("Length (residues)")
clean(ax)

ax = fig.add_subplot(gs[1, 2:])
panel(ax, "D", "Key physicochemical distributions")
features = ["net_charge_pH7_4", "isoelectric_point", "mean_eisenberg_hydrophobicity", "hydrophobic_moment"]
labels = ["Net charge", "pI", "Hydrophobicity", "Hydrophobic moment"]
for j, (feat, lab) in enumerate(zip(features, labels)):
    for k, (cls, color) in enumerate([(0, SKY), (1, VERMILION)]):
        vals = traditional.loc[traditional.label == cls, feat]
        z = (vals - traditional[feat].mean()) / traditional[feat].std(ddof=1)
        pos = j + (-0.16 if cls == 0 else 0.16)
        bp = ax.boxplot(z, positions=[pos], widths=0.28, patch_artist=True, showfliers=False,
                        medianprops=dict(color=BLACK), whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY))
        bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.78)
ax.axhline(0, color=GRAY, lw=0.7)
ax.set_xticks(range(4), labels)
ax.set_ylabel("Standardized value")
ax.legend([Rectangle((0, 0), 1, 1, fc=SKY), Rectangle((0, 0), 1, 1, fc=VERMILION)], ["Inactive", "Active"], frameon=False, ncol=2)
clean(ax)

ax = fig.add_subplot(gs[2, :2])
panel(ax, "E", "Traditional-feature correlation structure")
corr = read("results/step26_spearman_correlation_matrix.csv").set_index("Unnamed: 0")
sel = ["length", "molecular_weight", "net_charge_pH7_4", "isoelectric_point", "mean_eisenberg_hydrophobicity", "hydrophobic_moment", "boman_index"]
short = ["Length", "MW", "Charge", "pI", "Hydrophobicity", "Hydrophobic moment", "Boman"]
mat = corr.loc[sel, sel].to_numpy()
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(sel)), short, rotation=35, ha="right")
ax.set_yticks(range(len(sel)), short)
for i in range(len(sel)):
    for j in range(len(sel)):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(mat[i,j]) > 0.55 else BLACK)
fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman rho")

ax = fig.add_subplot(gs[2, 2])
panel(ax, "F", "Representation overview")
ax.bar([0, 1], [32, 1280], color=[ORANGE, BLUE], width=0.65)
ax.set_yscale("log")
ax.set_xticks([0, 1], ["Traditional", "ESM-2"])
ax.set_ylabel("Dimensions (log scale)")
for x, v in enumerate([32, 1280]): ax.text(x, v * 1.15, f"{v:,}", ha="center", fontweight="bold")
clean(ax)

ax = fig.add_subplot(gs[2, 3])
panel(ax, "G", "Leakage-safe model evaluation")
ax.axis("off")
dev_counts = folds.loc[folds.split == "development", "cv_fold"].value_counts().sort_index()
ax.text(0.5, 0.93, "Development set (n=720)", ha="center", fontweight="bold", transform=ax.transAxes)
for i, (fold, n) in enumerate(dev_counts.items()):
    y = 0.78 - i * 0.11
    ax.add_patch(Rectangle((0.08, y), 0.64, 0.065, transform=ax.transAxes, fc=LIGHT, ec=GRAY))
    ax.add_patch(Rectangle((0.72, y), 0.16, 0.065, transform=ax.transAxes, fc=ORANGE, ec=GRAY))
    ax.text(0.40, y + 0.032, "Train 576", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
    ax.text(0.80, y + 0.032, f"Val {int(n)}", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
    ax.text(0.02, y + 0.032, f"F{fold}", va="center", transform=ax.transAxes)
ax.add_patch(FancyBboxPatch((0.10, 0.08), 0.78, 0.15, boxstyle="round,pad=0.02", transform=ax.transAxes, fc="#FDE9E7", ec=VERMILION, lw=1.2))
ax.text(0.49, 0.155, "Locked test (n=181)\nuntouched until configuration freeze", ha="center", va="center", transform=ax.transAxes, fontweight="bold", fontsize=7)

fig.suptitle("Figure 1. Study design, peptide landscape, and computational framework", fontsize=14, fontweight="bold")
f1 = save(fig, "Figure1_Study_Design_and_Peptide_Landscape")


# -------------------------------------------------------------------------
# Figure 2: comparative performance and uncertainty
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(13.2, 13.0), constrained_layout=True)
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.1, 1.05])
metrics = ["AUROC", "AUPRC", "MCC", "F1"]
for idx, metric in enumerate(metrics):
    ax = fig.add_subplot(gs[0, idx])
    panel(ax, chr(65 + idx), f"{metric} with 95% bootstrap CI")
    d = perf.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    values = d[metric].to_numpy()
    lo = d[f"{metric}_CI_low"].to_numpy(); hi = d[f"{metric}_CI_high"].to_numpy()
    colors = [ORANGE if r == "Traditional" else BLUE for r in d.representation]
    ax.errorbar(values, y, xerr=[values-lo, hi-values], fmt="none", ecolor=GRAY, capsize=2, lw=1)
    ax.scatter(values, y, c=colors, s=30, edgecolor=BLACK, linewidth=0.4, zorder=3)
    ax.set_yticks(y, [model_short(x) for x in d.model] if idx == 0 else [""] * len(d))
    ax.set_xlim(max(-0.1, min(lo)-0.05), min(1.03, max(hi)+0.05))
    ax.set_xlabel(metric)
    clean(ax)

ax = fig.add_subplot(gs[1, :2])
panel(ax, "E", "Matched ESM-2 minus Traditional differences")
metric_pos = {m: i for i, m in enumerate(metrics)}
classifiers = ["Logistic Regression", "RBF-SVM", "Random Forest", "XGBoost"]
offsets = np.linspace(-0.24, 0.24, 4)
for ci, classifier in enumerate(classifiers):
    dd = paired[paired.classifier == classifier].set_index("metric").loc[metrics]
    x = np.arange(4) + offsets[ci]
    v = dd.observed_delta.to_numpy(); lo = dd.ci_95_lower.to_numpy(); hi = dd.ci_95_upper.to_numpy()
    ax.errorbar(x, v, yerr=[v-lo, hi-v], marker=["o","s","^","D"][ci], color=MODEL_COLORS[ci], lw=1, capsize=2, label=classifier)
ax.axhline(0, color=BLACK, lw=0.9)
ax.set_xticks(range(4), metrics)
ax.set_ylabel("Paired delta (ESM-2 − Traditional)")
ax.legend(frameon=False, ncol=2)
clean(ax)

ax = fig.add_subplot(gs[1, 2:])
panel(ax, "F", "Calibration comparison")
cal_metrics = [("brier_score", "Brier"), ("ece_10_equal_width_bins", "ECE"), ("log_loss", "Log loss")]
for j, (col, lab) in enumerate(cal_metrics):
    vals = calibration[col].to_numpy()
    x = np.full(len(vals), j) + np.linspace(-0.22, 0.22, len(vals))
    ax.scatter(x, vals, c=[ORANGE if r == "Traditional" else BLUE for r in calibration.representation], s=32, edgecolor=BLACK, linewidth=0.3)
ax.set_xticks(range(3), [x[1] for x in cal_metrics])
ax.set_ylabel("Calibration error (lower is better)")
ax.legend([Rectangle((0,0),1,1,fc=ORANGE), Rectangle((0,0),1,1,fc=BLUE)], ["Traditional", "ESM-2"], frameon=False)
clean(ax)

ax = fig.add_subplot(gs[2, :])
panel(ax, "G", "Decision utility across predefined threshold ranges")
d = decision.sort_values("mean_net_benefit_0_05_to_0_20", ascending=True)
y = np.arange(len(d)); h = 0.34
ax.barh(y-h/2, d.mean_net_benefit_0_05_to_0_20, h, color=GREEN, label="Mean NB, 0.05–0.20")
ax.barh(y+h/2, d.mean_net_benefit_0_05_to_0_50, h, color=PURPLE, label="Mean NB, 0.05–0.50")
ax.set_yticks(y, [model_short(x) for x in d.model])
ax.set_xlabel("Mean net benefit")
ax.legend(frameon=False, ncol=2, loc="upper left")
clean(ax, "x")
ax.text(0.99, 0.96, "Frozen predictions; no threshold optimized", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=GRAY)
fig.suptitle("Figure 2. Comparative predictive performance and uncertainty", fontsize=14, fontweight="bold")
f2 = save(fig, "Figure2_Comparative_Performance_and_Uncertainty")


# -------------------------------------------------------------------------
# Figure 3: novelty, generalization, complementarity
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(13.2, 12.0), constrained_layout=True)
gs = fig.add_gridspec(3, 4)
subsets = novelty[["subset_order", "subset", "subset_label", "n", "active", "inactive"]].drop_duplicates().sort_values("subset_order")
for idx, metric in enumerate(["AUROC", "AUPRC"]):
    ax = fig.add_subplot(gs[0, idx*2:(idx+1)*2])
    panel(ax, chr(65+idx), f"Sequence-novelty {metric}")
    for m_idx, (model, dd) in enumerate(novelty.groupby("model", sort=False)):
        dd = dd.sort_values("subset_order")
        ax.plot(dd.subset_order, dd[metric], marker="o", ms=3.5, lw=1.1,
                color=MODEL_COLORS[m_idx], label=model_short(model))
    labels = [f"{r.subset_label}\nn={r.n}, A={r.active}" for r in subsets.itertuples()]
    ax.set_xticks(subsets.subset_order, labels)
    ax.set_ylim(0.45 if metric == "AUPRC" else 0.80, 1.015)
    ax.set_ylabel(metric)
    if idx == 0: ax.legend(frameon=False, ncol=2, fontsize=6.2)
    clean(ax)

ax = fig.add_subplot(gs[1, :2])
panel(ax, "C", "Cross-validated canonical correlations")
summary_cc = cvcca.groupby("canonical_component").agg(train=("training_correlation","mean"), valid=("validation_correlation","mean"), sd=("validation_correlation","std")).reset_index()
x = summary_cc.canonical_component.to_numpy()
ax.plot(x, summary_cc.train, "o-", color=ORANGE, label="Training")
ax.errorbar(x, summary_cc.valid, yerr=summary_cc.sd, fmt="s-", color=BLUE, capsize=2, label="Validation mean ± SD")
ax.set_xticks(x, [f"CC{i}" for i in x])
ax.set_ylim(0.90, 1.005)
ax.set_ylabel("Canonical correlation")
ax.legend(frameon=False)
clean(ax)

ax = fig.add_subplot(gs[1, 2:])
panel(ax, "D", "Descriptor–ESM-2 PC associations")
desc = ["length", "molecular_weight", "net_charge_pH7_4", "isoelectric_point", "mean_eisenberg_hydrophobicity", "hydrophobic_moment", "boman_index"]
dlabels = ["Length", "MW", "Charge", "pI", "Hydrophobicity", "Hydrophobic moment", "Boman"]
pcs = list(range(1, 13))
pivot = pcassoc[pcassoc.traditional_descriptor.isin(desc) & pcassoc.esm2_pc_number.isin(pcs)].pivot(index="traditional_descriptor", columns="esm2_pc_number", values="spearman_rho").loc[desc, pcs]
im = ax.imshow(pivot, cmap="RdBu_r", vmin=-0.65, vmax=0.65, aspect="auto")
ax.set_yticks(range(len(desc)), dlabels); ax.set_xticks(range(12), [f"PC{i:02d}" for i in pcs], rotation=45, ha="right")
fig.colorbar(im, ax=ax, shrink=0.78, label="Spearman rho")

ax = fig.add_subplot(gs[2, :2])
panel(ax, "E", "Fold-specific dimensions for ≥90% variance")
vals = [cvfold.traditional_pc90_count, cvfold.esm2_pc90_count]
bp = ax.boxplot(vals, tick_labels=["Traditional", "ESM-2"], patch_artist=True, widths=0.55)
for box, color in zip(bp["boxes"], [ORANGE, BLUE]): box.set_facecolor(color); box.set_alpha(0.8)
for i, v in enumerate(vals, start=1): ax.scatter(np.full(len(v), i)+np.linspace(-0.06,0.06,len(v)), v, color=BLACK, s=18, zorder=3)
ax.set_ylabel("PC count")
ax.text(0.03, 0.96, f"Traditional: {vals[0].min()}–{vals[0].max()}\nESM-2: {vals[1].min()}–{vals[1].max()}", transform=ax.transAxes, va="top")
clean(ax)

ax = fig.add_subplot(gs[2, 2:])
panel(ax, "F", "Fusion versus ESM-2-only paired differences")
for ci, classifier in enumerate(["Logistic Regression", "RBF-SVM"]):
    dd = fusion_delta[fusion_delta.classifier == classifier].set_index("metric").loc[metrics]
    pos = np.arange(4) + (-0.09 if ci == 0 else 0.09)
    v = dd.observed_delta_fusion_minus_esm2.to_numpy()
    ax.errorbar(pos, v, yerr=[v-dd.ci_2_5.to_numpy(), dd.ci_97_5.to_numpy()-v], fmt=["o","s"][ci], color=[VERMILION, BLUE][ci], capsize=2, label=classifier)
ax.axhline(0, color=BLACK, lw=0.9)
ax.set_xticks(range(4), metrics)
ax.set_ylabel("Paired delta (Fusion − ESM-2)")
ax.legend(frameon=False)
clean(ax)
fig.suptitle("Figure 3. Generalization, sequence novelty, and representation complementarity", fontsize=14, fontweight="bold")
f3 = save(fig, "Figure3_Generalization_and_Complementarity")


# -------------------------------------------------------------------------
# Figure 4: latent dimensions to sequence context
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(13.2, 14.0), constrained_layout=True)
gs = fig.add_gridspec(4, 4, height_ratios=[1.1, 0.72, 1, 1.25])
for idx, model in enumerate(["ESM-2 Random Forest", "ESM-2 XGBoost"]):
    ax = fig.add_subplot(gs[0, idx*2:(idx+1)*2])
    panel(ax, chr(65+idx), f"{model.replace('ESM-2 ', '')} stable latent dimensions")
    dd = importance[importance.model == model].nsmallest(12, "rank_by_mean_importance").sort_values("mean_importance")
    y = np.arange(len(dd))
    ax.barh(y, dd.mean_importance, xerr=dd.sd_importance, color=[SKY, ORANGE][idx], ecolor=GRAY, capsize=2)
    ax.set_yticks(y, dd.feature)
    ax.set_xlabel("Mean CV importance ± SD")
    clean(ax, "x")

ax = fig.add_subplot(gs[1, :])
panel(ax, "C", "Cross-model stable-top-20 overlap")
ax.axis("off")
ov = overlap.loc[overlap.stability_criterion == "stable_top20"].iloc[0]
shared = ov.intersection_features.split(";")
ax.add_patch(FancyBboxPatch((0.08,0.18),0.34,0.62,boxstyle="round,pad=0.02",fc="#DDEBF7",ec=BLUE,transform=ax.transAxes))
ax.add_patch(FancyBboxPatch((0.58,0.18),0.34,0.62,boxstyle="round,pad=0.02",fc="#FCE4D6",ec=ORANGE,transform=ax.transAxes))
ax.text(0.25,0.68,f"RF stable top 20\n{int(ov.rf_stable_count)} dimensions",ha="center",transform=ax.transAxes,fontweight="bold")
ax.text(0.75,0.68,f"XGBoost stable top 20\n{int(ov.xgboost_stable_count)} dimensions",ha="center",transform=ax.transAxes,fontweight="bold")
ax.text(0.50,0.44,"Shared (n=7)\n"+", ".join(shared),ha="center",va="center",transform=ax.transAxes,fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5",fc="white",ec=GREEN,lw=1.4))
ax.text(0.50,0.12,f"Jaccard index = {ov.jaccard_index:.2f}; latent coordinates are model-use signals, not biological mechanisms",ha="center",transform=ax.transAxes,color=GRAY)

ax = fig.add_subplot(gs[2, :2])
panel(ax, "D", "Hard-case versus high-confidence-correct sensitivity")
cons = peptide_sens[peptide_sens.classifier == "RF/XGBoost consensus"]
groups = ["Consensus hard error", "High-confidence consensus-correct"]
for i, (g, color) in enumerate(zip(groups, [VERMILION, BLUE])):
    dd = cons[cons.analysis_group == g]
    ax.scatter(np.full(len(dd), i)+np.linspace(-0.08,0.08,len(dd)), dd.mean_absolute_residue_sensitivity, color=color, edgecolor=BLACK, label="Mean sensitivity" if i==0 else None)
    ax.scatter(np.full(len(dd), i)+np.linspace(-0.08,0.08,len(dd)), dd.maximum_absolute_residue_sensitivity, color=color, marker="^", edgecolor=BLACK, label="Maximum sensitivity" if i==0 else None)
ax.set_xticks([0,1],["Universal hard errors\n(n=5)","Consensus-correct\n(n=5)"])
ax.set_ylabel("Absolute change in true-class probability")
ax.legend(frameon=False)
clean(ax)

ax = fig.add_subplot(gs[2, 2])
panel(ax, "E", "Residue-category sensitivity")
cats = ["Basic","Hydrophobic","Acidic","Polar/other"]
for i,g in enumerate(groups):
    dd=category[category.analysis_group==g].set_index("residue_category").reindex(cats)
    ax.bar(np.arange(4)+(i-0.5)*0.34,dd.mean_consensus_sensitivity,0.34,color=[VERMILION,BLUE][i],label=["Hard","Correct"][i])
ax.set_xticks(range(4),["Basic","Hydro.","Acidic","Polar/\nother"])
ax.set_ylabel("Mean sensitivity")
ax.legend(frameon=False)
clean(ax)

ax = fig.add_subplot(gs[2, 3])
panel(ax, "F", "Motif-context sensitivity")
dd=motif[motif.summary_type=="any_motif_context"].set_index("motif_or_context")
names=["Inside any recurrent motif","Outside recurrent motifs"]
ax.bar([0,1],dd.loc[names,"mean_consensus_sensitivity"],color=[PURPLE,GRAY])
ax.set_xticks([0,1],["Motif-covered","Outside motifs"])
ax.set_ylabel("Mean hard-case sensitivity")
for i,v in enumerate(dd.loc[names,"mean_consensus_sensitivity"]): ax.text(i,v+0.008,f"{v:.3f}",ha="center")
clean(ax)

ax = fig.add_subplot(gs[3, :])
panel(ax, "G", "Representative sequence-level alanine scans")
ids=[48,40,145]
ax.set_xlim(0.5,max(residue[residue.peptide_ID.isin(ids)].peptide_length)+0.5); ax.set_ylim(-0.6,2.6)
for row_y,pid in enumerate(ids[::-1]):
    dd=residue[(residue.peptide_ID==pid)&residue.perturbation_performed].sort_values("position")
    seq=dd.sequence.iloc[0]
    y=row_y
    sc=ax.scatter(dd.position,np.full(len(dd),y),c=dd.consensus_absolute_sensitivity,cmap="viridis",norm=Normalize(0,0.85),s=180,marker="s",edgecolor=np.where(dd.inside_recurrent_motif,"#000000","#BBBBBB"),linewidth=1.0)
    for r in dd.itertuples(): ax.text(r.position,y,r.original_residue,ha="center",va="center",fontsize=7,color="white" if r.consensus_absolute_sensitivity>0.42 else BLACK,fontweight="bold")
    ax.text(0.1,y,f"ID {pid}",ha="right",va="center",fontweight="bold")
ax.set_yticks([]); ax.set_xlabel("Residue position (black outline = recurrent-motif context)")
fig.colorbar(sc,ax=ax,orientation="horizontal",pad=0.12,shrink=0.55,label="Consensus absolute sensitivity")
fig.suptitle("Figure 4. Model interpretation from latent dimensions to sequence context",fontsize=14,fontweight="bold")
f4=save(fig,"Figure4_ESM2_Interpretability")


# -------------------------------------------------------------------------
# Figure 5: anatomy of universal failures
# -------------------------------------------------------------------------
fig=plt.figure(figsize=(13.2,13.5),constrained_layout=True)
gs=fig.add_gridspec(3,4,height_ratios=[0.75,1.25,1.3])

ax=fig.add_subplot(gs[0,:2]); ax.axis("off"); panel(ax,"A","Hard-case hierarchy")
levels=[("Locked test",181,BLUE),("≥3 of 8 models wrong",15,ORANGE),("Universal 8/8 errors",5,VERMILION)]
for i,(name,n,color) in enumerate(levels):
    width=0.82-i*0.20; x=(1-width)/2; y=0.72-i*0.27
    ax.add_patch(FancyBboxPatch((x,y),width,0.18,boxstyle="round,pad=0.01",fc=color,ec=BLACK,alpha=0.82,transform=ax.transAxes))
    ax.text(0.5,y+0.09,f"{name}: n={n}",ha="center",va="center",transform=ax.transAxes,color="white" if i!=1 else BLACK,fontweight="bold")

ax=fig.add_subplot(gs[0,2:]); panel(ax,"B","Eight-model error patterns among 15 hard cases")
ax.set_title("Eight-model error patterns among 15 hard cases", loc="left", fontweight="bold", pad=24)
cols=["LR_traditional_correct","SVM_traditional_correct","RF_traditional_correct","XGB_traditional_correct","LR_esm2_correct","SVM_esm2_correct","RF_esm2_correct","XGB_esm2_correct"]
mat=(~hard[cols].astype(bool)).astype(int).to_numpy()
ax.imshow(mat,cmap=matplotlib.colors.ListedColormap(["#E8F3F8",VERMILION]),vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(8),["LR","SVM","RF","XGB","LR","SVM","RF","XGB"],rotation=45,ha="right")
ax.set_yticks(range(15),[f"{r.ID} ({r.total_wrong_count}/8)" for r in hard.itertuples()])
ax.axvline(3.5,color=BLACK,lw=1.2)
ax.text(0.25,1.01,"Traditional",ha="center",va="bottom",fontweight="bold",transform=ax.transAxes)
ax.text(0.75,1.01,"ESM-2",ha="center",va="bottom",fontweight="bold",transform=ax.transAxes)

u=integrated[integrated.universal_8_of_8_error].set_index("ID").loc[[48,40,145,56,68]]
ax=fig.add_subplot(gs[1,:2]); panel(ax,"C","Neighborhood conflict margins")
margin_cols=["nearest_test_sequence_margin","nearest_test_esm2_margin","nearest_development_sequence_margin","nearest_development_esm2_margin"]
mlabels=["Test sequence","Test ESM-2","Development sequence","Development ESM-2"]
x=np.arange(5)
for j,(col,lab) in enumerate(zip(margin_cols,mlabels)):
    ax.plot(x,u[col],marker=["o","s","^","D"][j],label=lab,lw=1)
ax.axhline(0,color=BLACK,lw=0.9)
ax.set_xticks(x,[str(i) for i in u.index]); ax.set_xlabel("Universal-error ID"); ax.set_ylabel("Same minus opposite-class similarity margin")
ax.legend(frameon=False,ncol=2); clean(ax)

ax=fig.add_subplot(gs[1,2:]); panel(ax,"D","Development-neighborhood purity")
w=0.36; x=np.arange(5)
ax.bar(x-w/2,u.development_sequence_top10_purity,w,color=ORANGE,label="Sequence")
ax.bar(x+w/2,u.development_esm2_top10_purity,w,color=BLUE,label="ESM-2")
ax.axhline(0.5,color=VERMILION,ls="--",lw=1,label="Low-purity criterion (<0.50)")
ax.set_xticks(x,[str(i) for i in u.index]); ax.set_xlabel("Universal-error ID"); ax.set_ylabel("Same-class fraction among top 10")
ax.set_ylim(0,1); ax.legend(frameon=False,ncol=2); clean(ax)

ax=fig.add_subplot(gs[2,:2]); panel(ax,"E","Integrated universal-error evidence map")
evidence=pd.DataFrame(index=u.index)
evidence["Descriptor\nextremeness"]=np.clip(u.mean_absolute_descriptor_z/1.5,0,1)
evidence["Sequence\nconflict"]=u.opposite_class_development_neighbor_sequence.astype(float)
evidence["ESM-2\nconflict"]=u.opposite_class_development_neighbor_esm2.astype(float)
evidence["Low ESM-2\npurity"]=1-u.development_esm2_top10_purity
evidence["Residue\nsensitivity"]=np.clip(u.consensus_max_residue_sensitivity/0.85,0,1)
evidence["Motif-linked\ntop residue"]=u.motif_linked_top_residue.astype(float)
im=ax.imshow(evidence,cmap="cividis",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(evidence.shape[1]),evidence.columns); ax.set_yticks(range(5),[str(i) for i in evidence.index]); ax.set_ylabel("Peptide ID")
for i in range(5):
    for j in range(evidence.shape[1]): ax.text(j,i,f"{evidence.iloc[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if evidence.iloc[i,j]<0.35 else BLACK)
fig.colorbar(im,ax=ax,shrink=0.75,label="Normalized descriptive indicator")

ax=fig.add_subplot(gs[2,2:]); panel(ax,"F","Universal-error sequence and residue summary"); ax.axis("off")
headers=["ID","Class","Top residue","Sensitivity","Motif context","Evidence category"]
rows=[]
for r in universal.itertuples(): rows.append([r.ID,r._1 if hasattr(r,'_1') else r[2],r.top_sensitive_residue,f"{r.top_sensitivity:.3f}",r.top_residue_motif_context,r.qualitative_evidence_category])
# Use direct column access because 'class' is not a valid namedtuple attribute.
rows=[[int(r.ID),r["class"],r.top_sensitive_residue,f"{r.top_sensitivity:.3f}",r.top_residue_motif_context,r.qualitative_evidence_category] for _,r in universal.iterrows()]
table=ax.table(cellText=rows,colLabels=headers,cellLoc="left",colLoc="left",loc="center",colWidths=[0.07,0.10,0.12,0.12,0.16,0.43])
table.auto_set_font_size(False); table.set_fontsize(6.8); table.scale(1,1.8)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor("#D0D0D0"); cell.set_facecolor(LIGHT if r==0 else "white");
    if r==0: cell.set_text_props(fontweight="bold")
ax.text(0.0,0.02,"Descriptive model-interpretability evidence; not evidence of label error or biochemical causality.",transform=ax.transAxes,color=GRAY,fontsize=7)
fig.suptitle("Figure 5. Anatomy of universal model failures",fontsize=14,fontweight="bold")
f5=save(fig,"Figure5_Universal_Model_Failures")


figures = [
    ("Figure 1", "Study design, peptide landscape, and computational framework", f1, "A-G"),
    ("Figure 2", "Comparative predictive performance and uncertainty", f2, "A-G"),
    ("Figure 3", "Generalization, sequence novelty, and representation complementarity", f3, "A-F"),
    ("Figure 4", "Model interpretation from latent dimensions to sequence context", f4, "A-G"),
    ("Figure 5", "Anatomy of universal model failures", f5, "A-F"),
]

architecture_rows=[]
for number,title,(png,pdf),panels in figures:
    architecture_rows.append({"figure_number":number,"manuscript_title":title,"panel_range":panels,"png":str(png.relative_to(PROJECT)),"pdf":str(pdf.relative_to(PROJECT)),"status":"Main figure - frozen in Step 86B"})

panel_rows = [
    ("Figure 1","A","Study workflow","derived/traditional_features.csv; derived/fixed_split.csv","Schematic from frozen study counts and design"),
    ("Figure 1","B","Dataset composition","derived/fixed_split.csv","Counts by split and class"),
    ("Figure 1","C","Peptide length distribution","derived/traditional_features.csv","All 901 peptides"),
    ("Figure 1","D","Physicochemical distributions","derived/traditional_features.csv","Four biologically relevant descriptors"),
    ("Figure 1","E","Traditional correlation structure","results/step26_spearman_correlation_matrix.csv","Selected seven-descriptor matrix"),
    ("Figure 1","F","Representation overview","derived/traditional_features.csv; derived/esm2_embeddings.npy","Dimension counts only"),
    ("Figure 1","G","Leakage-safe modeling","derived/fixed_cv_folds.csv","Five fixed development folds; test untouched"),
    ("Figure 2","A-D","Performance with uncertainty","results/step74_model_performance_bootstrap_summary.csv","Frozen point estimates and existing 95% intervals"),
    ("Figure 2","E","Matched representation deltas","results/step54_paired_bootstrap_summary.csv","ESM-2 minus Traditional"),
    ("Figure 2","F","Calibration","results/step71_calibration_metrics.csv","Brier, ECE, log loss"),
    ("Figure 2","G","Decision utility","results/step73_decision_curve_model_summary.csv","Predefined threshold-range means"),
    ("Figure 3","A-B","Sequence-novelty performance","results/step67_sequence_novelty_performance.csv","AUROC and AUPRC with n/Active annotations"),
    ("Figure 3","C","Cross-validated CCA","results/step78_cv_canonical_correlations.csv","Training and held-out validation correlations"),
    ("Figure 3","D","Descriptor-PC association","results/step77_traditional_vs_esm2_pc_correlations.csv","Seven descriptors by ESM-2 PC01-PC12"),
    ("Figure 3","E","PC90 dimensionality","results/step78_cv_cca_fold_summary.csv","Fold-contained PCA counts"),
    ("Figure 3","F","Fusion comparison","results/step80_fusion_vs_esm2_paired_bootstrap_summary.csv","Fusion minus ESM-2 paired deltas"),
    ("Figure 4","A-B","Stable latent importance","results/step81_esm2_cv_feature_importance_summary.csv","RF and XGBoost"),
    ("Figure 4","C","Stable-feature overlap","results/step81_esm2_feature_importance_overlap.csv","Stable-top-20 intersection"),
    ("Figure 4","D","Hard-vs-correct residue sensitivity","results/step82_peptide_perturbation_summary.csv","Predefined ten-peptide panel"),
    ("Figure 4","E-F","Residue category and motif context","results/step83_residue_category_summary.csv; results/step83_motif_sensitivity_summary.csv","Descriptive summaries"),
    ("Figure 4","G","Representative alanine scans","results/step83_residue_physicochemical_context.csv","IDs 48, 40, 145"),
    ("Figure 5","A-B","Hard-case hierarchy and error patterns","results/step59_consensus_hard_cases_manuscript.csv","15 hard cases and five universal errors"),
    ("Figure 5","C-D","Neighborhood conflict and purity","results/step84_integrated_hard_case_interpretability.csv","Five universal errors"),
    ("Figure 5","E-F","Integrated evidence and residue summary","results/step84_integrated_hard_case_interpretability.csv; results/step84_universal_error_evidence_table.csv","Descriptive interpretability evidence"),
]
panel_columns=["figure_number","panel","panel_title","source_files","transformation_or_scope"]
panel_dicts=[dict(zip(panel_columns,row)) for row in panel_rows]

with ARCH_OUT.open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(architecture_rows[0])); w.writeheader(); w.writerows(architecture_rows)
with PANEL_OUT.open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=panel_columns); w.writeheader(); w.writerows(panel_dicts)

qc=[]
for number,title,(png,pdf),panels in figures:
    for fmt,path in [("PNG",png),("PDF",pdf)]:
        qc.append({"figure_number":number,"format":fmt,"path":str(path.relative_to(PROJECT)),"exists":path.is_file(),"size_bytes":path.stat().st_size,"sha256":sha256(path),"white_background":True,"unified_font_and_palette":True,"panel_labels_present":True,"source_data_frozen":True,"model_retraining":False,"new_bootstrap":False,"threshold_optimization":False,"scientific_result_changed":False})
with QC_OUT.open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(qc[0])); w.writeheader(); w.writerows(qc)

assert len(architecture_rows)==5 and len(qc)==10
assert all(r["exists"] and r["size_bytes"]>1000 for r in qc)
print("STEP 86B COMPLETED SUCCESSFULLY")
print("Main composite figures:",len(architecture_rows))
print("Panel provenance rows:",len(panel_dicts))
print("Figure files verified:",len(qc))
print(ARCH_OUT)
print(PANEL_OUT)
print(QC_OUT)

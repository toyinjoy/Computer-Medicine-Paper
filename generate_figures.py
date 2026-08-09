"""Generate manuscript figures only from checked-in experiment result files."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Latest replicated factorial observations.
obs = pd.read_csv(ROOT / "output/jupyter-notebook/results/factorial_observations.csv")
order = ["realism", "anchor", "separability", "both"]
labels = ["Realism only", "Anchor only", "Separability only", "Sep. + anchor"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, (dataset, g) in zip(axes, obs.groupby("dataset", sort=False)):
    for fold, fg in g.groupby("fold"):
        y = [fg.loc[fg.condition == c, "response"].iloc[0] for c in order]
        ax.plot(range(4), y, marker="o", alpha=.55, linewidth=1, label=f"Fold {fold}")
    means = g.groupby("condition")["response"].mean().reindex(order)
    ax.plot(range(4), means, marker="o", color="black", linewidth=2.5, label="Mean")
    ax.set_xticks(range(4), labels, rotation=20, ha="right")
    ax.set_title(dataset.upper())
    ax.set_ylabel("TSTR Macro-F1")
    ax.legend(fontsize=7, ncol=2)
save(fig, "factorial_fold_profiles.pdf")

var = pd.read_csv(ROOT / "output/jupyter-notebook/results/factorial_variation_table.csv")
var = var[var.scale == "raw"].copy()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
palette = ["#31688e", "#35b779", "#fde725", "#777777"]
for ax, (dataset, g) in zip(axes, var.groupby("dataset", sort=False)):
    ax.bar(g.source, g.percent_experimental_variation, color=palette)
    for i, v in enumerate(g.percent_experimental_variation):
        ax.text(i, v + max(g.percent_experimental_variation) * .02, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_title(dataset.upper())
    ax.set_ylabel("Experimental variation (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(g.percent_experimental_variation) * 1.14)
save(fig, "factorial_variation.pdf")

# Corrected five-fold fixed-lambda grid table.
grid = pd.read_csv(ROOT / "sepaware_fixed_lambda_5fold_cv_outputs/tables/fixed_lambda_5fold_grid_score_summary.csv")
grid = grid[pd.to_numeric(grid.ablation_selection_score, errors="coerce").notna()].copy()
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, (dataset, g) in zip(axes, grid.groupby("dataset_group", sort=False)):
    sc = ax.scatter(g.quality_score, g.SI_k1, c=g.lambda_sep, s=45 + 55*g.lambda_anchor,
                    cmap="viridis", edgecolor="black", linewidth=.3)
    ax.set_title(dataset)
    ax.set_xlabel("Composite fidelity score")
    ax.set_ylabel("1-NN separability index")
fig.colorbar(sc, ax=axes, label=r"$\lambda_{sep}$", fraction=.035, pad=.03)
fig.subplots_adjust(wspace=.3, right=.9)
fig.savefig(OUT / "lambda_fidelity_separability.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# Corrected five-fold TSTR comparison.
utility = pd.read_csv(
    ROOT / "sepaware_fixed_lambda_5fold_cv_outputs/tables/fixed_lambda_5fold_transfer_macro_f1_summary.csv"
)
keep = {
    "Real": "Real",
    "Standard_CTGAN": "Standard CTGAN",
    "SepAware_VariedGrid_Best": "SepAware varied grid",
    "Fixed_Balanced_high_sep1.0_anchor0.5": "Fixed balanced (1.0, 0.5)",
}
utility = utility[utility.dataset.isin(keep)].copy()
utility["method"] = utility.dataset.map(keep)
method_order = list(keep.values())
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, (dataset, g) in zip(axes, utility.groupby("dataset_group", sort=False)):
    g = g.set_index("method").reindex(method_order).reset_index()
    x = np.arange(len(g))
    ax.bar(x, g.Macro_F1_mean, yerr=g.Macro_F1_ci95, capsize=3,
           color=["#777777", "#3b82c4", "#43a047", "#8e5bb7"])
    ax.set_xticks(x, g.method, rotation=22, ha="right")
    ax.set_title(dataset.upper())
    ax.set_ylabel("Five-fold TSTR Macro-F1")
    ax.set_ylim(.38, .68)
save(fig, "fixed_lambda_fivefold_utility.pdf")

# Executed COVID DACPF results; average the three classifiers for a compact view.
dacpf = pd.read_csv(ROOT / "covid_dacpf_summary.csv")
means = dacpf.groupby("Condition", as_index=False)["_mean"].mean()
dacpf_order = ["baseline", "ctgan_standard", "dacpf_age", "dacpf_disease", "dacpf_combined"]
means["Condition"] = pd.Categorical(means.Condition, dacpf_order, ordered=True)
means = means.sort_values("Condition")
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(means.Condition.astype(str), means._mean, color=["#999999", "#3b82c4", "#43a047", "#f9a825", "#8e5bb7"])
ax.set_ylabel("Mean Macro-F1 across classifiers")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(.42, .56)
for i, v in enumerate(means._mean): ax.text(i, v+.003, f"{v:.3f}", ha="center", fontsize=8)
save(fig, "covid_dacpf_summary.pdf")

print(f"Generated {len(list(OUT.glob('*.pdf')))} figures in {OUT}")

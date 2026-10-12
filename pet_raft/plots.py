"""
plots.py
--------
All figures for the two-output kinetic prediction pipeline.

Figures:
  fig1_conversion_distribution.png  — histograms of both targets
  fig2_parity_plots.png             — predicted vs actual, both targets
  fig3_cv_comparison.png            — CV stability
  fig4_shap_global.png              — SHAP feature importance per target
  fig5_shap_waterfall.png           — local SHAP, both targets
  fig6_kinetic_curves.png           — actual vs predicted kinetic curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from pet_raft.models import MODEL_COLORS
from pet_raft.config import PIPELINE, TARGET_COLUMNS


def _save(fig, filename):
    out_dir = PIPELINE["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=PIPELINE["dpi"], bbox_inches="tight")
    print(f"[Plot]  Saved → {path}")
    plt.show()
    plt.close(fig)


def plot_conversion_distribution(y):
    """Histograms for monomer_conv and ctm_conv."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col_i, color, label in [
        (ax1, 0, "#2E86AB", "Monomer Conversion"),
        (ax2, 1, "#E84855", "CTM Conversion"),
    ]:
        vals = y[:, col_i]
        ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(vals.mean(), color="black", ls="--", lw=2,
                   label=f"Mean = {vals.mean():.1f}%")
        ax.set_xlabel(f"{label} (%)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Dataset: Conversion Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig1_conversion_distribution.png")


def plot_parity_plots(test_results, y_test):
    """Parity plots — two rows (monomer top, CTM bottom), 10 models each."""
    model_order = ["LR","Ridge","Lasso","SVR","k-NN","RF","GBR","XGBoost","CatBoost","MLP"]
    labels      = ["Monomer conv", "CTM conv"]
    fig, axes   = plt.subplots(2, 10, figsize=(28, 8))

    for col_i, name in enumerate(model_order):
        res   = test_results[name]
        color = MODEL_COLORS.get(name, "#666666")
        for row_i, (t_i, label) in enumerate(zip([0, 1], labels)):
            ax     = axes[row_i][col_i]
            y_pred = res["y_pred"][:, t_i]
            y_true = y_test[:, t_i]
            r2_key = "r2_mon" if t_i == 0 else "r2_ctm"
            mae_key = "mae_mon" if t_i == 0 else "mae_ctm"

            ax.scatter(y_pred, y_true, alpha=0.6, s=20,
                       color=color, edgecolors="white", lw=0.3)
            ax.plot([0,100],[0,100],"k--",lw=1,alpha=0.4)
            ax.set_xlim(0,105); ax.set_ylim(0,105)
            if row_i == 0:
                ax.set_title(name, fontsize=9, fontweight="bold", color=color)
            if col_i == 0:
                ax.set_ylabel(f"{label}\nActual (%)", fontsize=8)
            ax.set_xlabel("Predicted (%)", fontsize=8)
            ax.text(0.05, 0.88,
                    f"R²={res[r2_key]:.2f}\nMAE={res[mae_key]:.1f}",
                    transform=ax.transAxes, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="white",alpha=0.8))

    plt.suptitle("Parity Plots — Top: Monomer Conv | Bottom: CTM Conv",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig2_parity_plots.png")


def plot_cv_comparison(cv_results, fold_r2s, ttest_results, best_name):
    """CV bar chart and boxplots."""
    names = list(cv_results.keys())
    r2s   = [cv_results[n]["r2_avg_mean"] for n in names]
    sds   = [cv_results[n]["r2_avg_sd"]   for n in names]
    x, w  = np.arange(len(names)), 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

    bars = ax1.bar(x, r2s, w, color="#2E86AB", alpha=0.85, label="Avg R²")
    ax1.errorbar(x, r2s, yerr=sds, fmt="none", color="black", capsize=4, lw=1.5)

    for i, name in enumerate(names):
        sig = ttest_results.get(name, {}).get("sig", "")
        if sig and sig not in ("—", "n.s."):
            ax1.text(x[i], r2s[i]+sds[i]+0.02, sig,
                     ha="center", fontsize=9, fontweight="bold")

    bars[names.index(best_name)].set_edgecolor("gold")
    bars[names.index(best_name)].set_linewidth(2.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Avg R² (monomer + CTM)", fontsize=11)
    ax1.set_title("CV Performance", fontsize=12, fontweight="bold")
    ax1.legend(); ax1.set_ylim(0, 1.15); ax1.grid(axis="y", alpha=0.25)

    fold_data = [fold_r2s[n] for n in names]
    bp = ax2.boxplot(fold_data, tick_labels=names, patch_artist=True,
                     medianprops=dict(color="white", lw=2))
    palette = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, col in zip(bp["boxes"], palette):
        patch.set_facecolor(col); patch.set_alpha(0.8)

    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Avg R² per Fold", fontsize=11)
    ax2.set_title("Fold-wise Stability", fontsize=12, fontweight="bold")
    ax2.axhline(0, color="grey", lw=0.5, ls="--"); ax2.grid(axis="y", alpha=0.25)

    plt.suptitle("5-Fold CV — Multi-Output Regression",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_cv_comparison.png")


def plot_shap_global(shap_values_list, X_test_df, mean_abs_shap, top_n=10):
    """Global SHAP bar chart for both targets side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, target_name, shap_vals in zip(axes, TARGET_COLUMNS, shap_values_list):
        mas    = mean_abs_shap[target_name].head(top_n)
        feats  = [f.replace("Mon_","Mon·").replace("CTM_","CTM·")
                  for f in mas.index[::-1]]
        colors = ["#E84855" if "Mon" in f
                  else "#2E86AB" if "CTM" in f
                  else "#3BB273" for f in feats]

        ax.barh(feats, mas.values[::-1], color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Mean |SHAP value| (pps)", fontsize=10)
        ax.set_title(f"Top {top_n} Features — {target_name}",
                     fontsize=12, fontweight="bold")
        ax.legend(handles=[
            mpatches.Patch(color="#E84855", label="Monomer"),
            mpatches.Patch(color="#2E86AB", label="CTM"),
            mpatches.Patch(color="#3BB273", label="Numerical"),
        ], fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Global SHAP Feature Importance — Both Targets",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig4_shap_global.png")


def plot_shap_waterfall(local_explanations, explainer_list, best_name):
    """Local SHAP waterfall: 2×2 grid (target × low/high)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for row_i, target_name in enumerate(TARGET_COLUMNS):
        for col_i, case in enumerate(["low", "high"]):
            ax   = axes[row_i][col_i]
            info = local_explanations[target_name][case]
            top_sv    = info["top_sv"]
            top_feats = [f.replace("Mon_","Mon·").replace("CTM_","CTM·")
                         for f in info["top_feats"]]

            positions  = np.arange(len(top_sv))
            bar_colors = ["#E84855" if s > 0 else "#2E86AB" for s in top_sv[::-1]]

            ax.barh(positions, top_sv[::-1], color=bar_colors,
                    alpha=0.85, edgecolor="white", height=0.6)
            ax.set_yticks(positions)
            ax.set_yticklabels(top_feats[::-1], fontsize=9)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP value (pps)", fontsize=10)
            label = "Lowest" if case == "low" else "Highest"
            ax.set_title(
                f"{target_name} — {label} conversion\n"
                f"Actual: {info['y_actual']:.1f}%  "
                f"Predicted: {info['y_pred']:.1f}%",
                fontsize=10, fontweight="bold"
            )
            ax.grid(axis="x", alpha=0.2)

    plt.suptitle(f"Local SHAP Waterfall — {best_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig5_shap_waterfall.png")


def plot_kinetic_curves(df, best_model, scaler, selected_features):
    """
    Actual vs predicted kinetic curves for each CTM.
    Shows both monomer and CTM conversion over time.
    """
    from pet_raft.descriptors import build_feature_matrix
    from pet_raft.features    import engineer_features, align_features

    ctm_names = sorted(df["ctm"].unique().tolist())
    monomer   = df["monomer"].iloc[0]
    feed_pct  = 10

    fig, axes = plt.subplots(1, len(ctm_names),
                              figsize=(6 * len(ctm_names), 5))
    if len(ctm_names) == 1:
        axes = [axes]

    colors = {"monomer_conv": "#2E86AB", "ctm_conv": "#E84855"}

    for ax, ctm in zip(axes, ctm_names):
        subset = df[(df["ctm"] == ctm) &
                    (df["ctm_feed_mol_pct"] == feed_pct)].sort_values("time_hours")

        if subset.empty:
            ax.set_title(f"{ctm}\n(no data)"); continue

        times = subset["time_hours"].values

        for target, color in colors.items():
            if target in subset.columns:
                label = "TEGMA (actual)" if target == "monomer_conv" \
                        else f"{ctm} (actual)"
                ax.plot(times, subset[target].values, "o-",
                        color=color, lw=2.5, ms=8,
                        markerfacecolor="white", markeredgewidth=2,
                        label=label)

        # Predicted curves
        try:
            X_sub = build_feature_matrix(subset)
            X_eng = engineer_features(X_sub)
            X_aln = align_features(X_eng, selected_features)
            X_sc  = scaler.transform(X_aln)
            y_hat = best_model.predict(X_sc)

            for t_i, (target, color) in enumerate(colors.items()):
                label = "TEGMA (predicted)" if t_i == 0 \
                        else f"{ctm} (predicted)"
                ax.plot(times, y_hat[:, t_i], "--",
                        color=color, lw=1.5, alpha=0.7, label=label)
        except Exception as e:
            print(f"  Could not plot predictions for {ctm}: {e}")

        ax.set_xlabel("Time (hours)", fontsize=11)
        ax.set_ylabel("Conversion (%)", fontsize=11)
        ax.set_title(f"{ctm}\n({feed_pct} mol% CTM)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    plt.suptitle("Kinetic Curves — Actual vs Predicted",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig6_kinetic_curves.png")

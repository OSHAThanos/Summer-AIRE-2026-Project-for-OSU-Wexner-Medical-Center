"""
run.py
------
Entry point — right-click → Run in IntelliJ.

Predicts BOTH monomer_conv AND ctm_conv simultaneously at each timepoint.

SWITCHING TO REAL DATA:
  Set USE_REAL_DATA = True
  Set REAL_DATA_PATH to your CSV

Your CSV needs ONLY 6 columns:
  monomer | ctm | ctm_feed_mol_pct | time_hours | monomer_conv | ctm_conv
  TEGMA   | CDTPA-HEMA | 10 | 1 | 29.0 | 13.17
  TEGMA   | CDTPA-HEAA | 10 | 2 | 18.0 | 83.30
  ...

Bablu's data is already formatted at: data/kinetic_data_processed.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pet_raft.config      import PIPELINE, TARGET_COLUMNS
from pet_raft.dataset     import generate_synthetic_dataset, load_real_dataset
from pet_raft.descriptors import build_feature_matrix
from pet_raft.features    import engineer_features, fit_scaler, align_features
from pet_raft.train       import (train_and_evaluate, select_best_model,
                                   paired_ttest_vs_best, save_results_csv)
from pet_raft.explain     import run_shap_analysis, get_local_explanations
from pet_raft.plots       import (plot_conversion_distribution,
                                   plot_parity_plots,
                                   plot_cv_comparison,
                                   plot_shap_global,
                                   plot_shap_waterfall,
                                   plot_kinetic_curves)

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_REAL_DATA  = False
REAL_DATA_PATH = "data/kinetic_data_processed.csv"

# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg = PIPELINE
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("=" * 65)
    print("  CTM Kinetic Prediction Pipeline")
    print("  Targets: monomer_conv AND ctm_conv (two-output)")
    print("=" * 65)

    # Step 1 — Dataset
    print("\n[Step 1]  Dataset")
    df = load_real_dataset(REAL_DATA_PATH) if USE_REAL_DATA \
         else generate_synthetic_dataset(cfg["n_synthetic_samples"],
                                          cfg["random_state"])

    # Step 2 — Descriptors
    print("\n[Step 2]  Molecular descriptor calculation")
    X_raw = build_feature_matrix(df)

    # Step 3 — Feature engineering
    print("\n[Step 3]  Feature engineering")
    X_eng = engineer_features(X_raw, cfg["variance_threshold"],
                               cfg["correlation_threshold"])

    y = df[TARGET_COLUMNS].values
    print(f"           Targets: {TARGET_COLUMNS}")
    print(f"           monomer_conv: mean={y[:,0].mean():.1f}%  "
          f"range={y[:,0].min():.1f}–{y[:,0].max():.1f}%")
    print(f"           ctm_conv:     mean={y[:,1].mean():.1f}%  "
          f"range={y[:,1].min():.1f}–{y[:,1].max():.1f}%")

    # Step 4 — Split + scale
    print("\n[Step 4]  Train/test split (80/20) + z-score normalisation")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_eng, y, test_size=cfg["test_size"], random_state=cfg["random_state"])
    scaler            = fit_scaler(X_tr.values)
    X_tr_sc           = scaler.transform(X_tr.values)
    X_te_sc           = scaler.transform(X_te.values)
    selected_features = X_eng.columns.tolist()
    print(f"           Train: {X_tr_sc.shape[0]} | "
          f"Test: {X_te_sc.shape[0]} | "
          f"Features: {X_tr_sc.shape[1]}")

    # Step 5 — Train
    print("\n[Step 5]  Model training — two outputs simultaneously")
    cv_results, test_results, fold_r2s, trained = train_and_evaluate(
        X_tr_sc, y_tr, X_te_sc, y_te,
        cfg["n_cv_folds"], cfg["random_state"])

    # Step 6 — Select
    print("\n[Step 6]  Model selection")
    best_name, _ = select_best_model(cv_results, test_results)
    best_model   = trained[best_name]

    # Step 7 — Stats
    print("\n[Step 7]  Paired t-test")
    ttest = paired_ttest_vs_best(fold_r2s, best_name)

    # Step 8 — SHAP
    print("\n[Step 8]  SHAP analysis")
    X_te_df = pd.DataFrame(X_te_sc, columns=selected_features)
    shap_list, mean_abs_shap, explainer_list = run_shap_analysis(
        best_model, X_tr_sc, X_te_sc,
        selected_features, best_name, cfg["shap_top_n"])
    local_expl = get_local_explanations(
        shap_list, test_results, y_te, selected_features, best_name)

    # Step 9 — Figures
    print("\n[Step 9]  Generating figures...")
    plot_conversion_distribution(y)
    plot_parity_plots(test_results, y_te)
    plot_cv_comparison(cv_results, fold_r2s, ttest, best_name)
    plot_shap_global(shap_list, X_te_df, mean_abs_shap, cfg["shap_top_n"])
    plot_shap_waterfall(local_expl, explainer_list, best_name)
    plot_kinetic_curves(df, best_model, scaler, selected_features)

    # Step 10 — Save
    save_results_csv(cv_results, test_results, cfg["output_dir"])

    # Summary
    b  = test_results[best_name]
    cv = cv_results[best_name]
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Data source:             {'Real' if USE_REAL_DATA else 'Synthetic'}")
    print(f"  Samples:                 {len(df)}")
    print(f"  Features:                {X_tr_sc.shape[1]}")
    print(f"  Best model:              {best_name}")
    print(f"  Test R² (monomer_conv):  {b['r2_mon']:.3f}")
    print(f"  Test R² (ctm_conv):      {b['r2_ctm']:.3f}")
    print(f"  Test R² (average):       {b['r2_avg']:.3f}")
    print(f"  CV R² avg (mean ± SD):   {cv['r2_avg_mean']:.3f} ± {cv['r2_avg_sd']:.3f}")
    print(f"\n  Top SHAP — monomer_conv:")
    for feat, val in mean_abs_shap[TARGET_COLUMNS[0]].head(3).items():
        tag = "[Mon]" if feat.startswith("Mon_") else \
              "[CTM]" if feat.startswith("CTM_") else "[Num]"
        print(f"    {tag} {feat:<40} {val:.4f}")
    print(f"\n  Top SHAP — ctm_conv:")
    for feat, val in mean_abs_shap[TARGET_COLUMNS[1]].head(3).items():
        tag = "[Mon]" if feat.startswith("Mon_") else \
              "[CTM]" if feat.startswith("CTM_") else "[Num]"
        print(f"    {tag} {feat:<40} {val:.4f}")
    print(f"\n  Outputs saved to: ./{cfg['output_dir']}/")
    print("=" * 65)


if __name__ == "__main__":
    main()

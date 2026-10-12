"""
explain.py
----------
SHAP analysis for two-output kinetic prediction.
Runs separately for monomer_conv and ctm_conv.
"""

import numpy as np
import pandas as pd
import shap
import warnings
from pet_raft.config import TARGET_COLUMNS


def run_shap_analysis(model, X_train, X_test, feature_names,
                       model_name, top_n=10):
    """
    SHAP analysis for both targets.

    Returns
    -------
    shap_values_list : [monomer_shap_array, ctm_shap_array]
    mean_abs_shap    : {target_name: pd.Series ranked by mean |SHAP|}
    explainer_list   : [explainer_for_mon, explainer_for_ctm]
    """
    print(f"\n[SHAP]  Running analysis on {model_name} for both targets...")

    shap_values_list = []
    mean_abs_shap    = {}
    explainer_list   = []

    for t_i, target_name in enumerate(TARGET_COLUMNS):
        print(f"        Target {t_i+1}: {target_name}")

        # Create a single-output predictor for this target
        def make_predictor(m, idx):
            def predict(X):
                return m.predict(X)[:, idx]
            return predict

        predictor  = make_predictor(model, t_i)
        background = shap.sample(X_train, min(50, len(X_train)))

        try:
            explainer  = shap.KernelExplainer(predictor, background)
            shap_vals  = explainer.shap_values(X_test)
        except Exception as e:
            warnings.warn(f"SHAP failed for {target_name}: {e}")
            shap_vals = np.zeros((len(X_test), len(feature_names)))
            explainer = None

        shap_df = pd.DataFrame(shap_vals, columns=feature_names)
        mas     = shap_df.abs().mean().sort_values(ascending=False)

        print(f"        Top {top_n} features:")
        for i, (feat, val) in enumerate(mas.head(top_n).items(), 1):
            tag = "[Mon]" if feat.startswith("Mon_") else \
                  "[CTM]" if feat.startswith("CTM_") else "[Num]"
            print(f"          {i:2}. {feat:<40} {val:.4f}  {tag}")

        shap_values_list.append(shap_vals)
        mean_abs_shap[target_name] = mas
        explainer_list.append(explainer)

    return shap_values_list, mean_abs_shap, explainer_list


def get_local_explanations(shap_values_list, test_results,
                            y_test, feature_names, best_name, top_n=8):
    """Local SHAP for lowest and highest conversion per target."""
    local = {}
    for t_i, target_name in enumerate(TARGET_COLUMNS):
        y_pred = test_results[best_name]["y_pred"][:, t_i]
        y_true = y_test[:, t_i]
        sv     = shap_values_list[t_i]

        idx_low  = int(np.argmin(y_pred))
        idx_high = int(np.argmax(y_pred))

        local[target_name] = {}
        for key, idx in [("low", idx_low), ("high", idx_high)]:
            shap_row = sv[idx]
            top_idx  = np.argsort(np.abs(shap_row))[-top_n:][::-1]
            local[target_name][key] = {
                "idx":       idx,
                "top_feats": [feature_names[i] for i in top_idx],
                "top_sv":    shap_row[top_idx],
                "y_actual":  float(y_true[idx]),
                "y_pred":    float(y_pred[idx]),
            }
    return local

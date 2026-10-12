"""
train.py
--------
Multi-output training: predicts monomer_conv AND ctm_conv simultaneously.
Uses MultiOutputRegressor wrapper for single-output models.
Tree ensembles (RF, GBR, XGBoost, CatBoost) support multi-output natively.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from pet_raft.models import get_models
from pet_raft.config import PIPELINE, TARGET_COLUMNS


def _wrap(model, name):
    """Wrap in MultiOutputRegressor if model doesn't natively support multi-output."""
    native = ["RF", "XGBoost"]
    return model if name in native else MultiOutputRegressor(model)


def _metrics(y_true, y_pred):
    """R², RMSE, MAE per target + average."""
    results = {}
    for i, name in enumerate(TARGET_COLUMNS):
        yt, yp = y_true[:, i], y_pred[:, i]
        results[name] = {
            "r2":   r2_score(yt, yp),
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "mae":  mean_absolute_error(yt, yp),
        }
    results["mean"] = {
        "r2":   np.mean([results[n]["r2"]   for n in TARGET_COLUMNS]),
        "rmse": np.mean([results[n]["rmse"] for n in TARGET_COLUMNS]),
        "mae":  np.mean([results[n]["mae"]  for n in TARGET_COLUMNS]),
    }
    return results


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       n_folds=None, random_state=None):
    """
    Train all 10 models with k-fold CV.
    Both monomer_conv and ctm_conv predicted simultaneously.
    """
    cfg          = PIPELINE
    n_folds      = n_folds      or cfg["n_cv_folds"]
    random_state = random_state or cfg["random_state"]

    models_raw = get_models()
    kf         = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_results, test_results, fold_r2s, trained = {}, {}, {}, {}

    t0, t1 = TARGET_COLUMNS[0], TARGET_COLUMNS[1]
    print(f"\n[Train]  {n_folds}-fold CV — targets: {TARGET_COLUMNS}\n")
    print(f"  {'Model':<12} {'CV R²(mon)':>11} {'CV R²(ctm)':>11} "
          f"{'CV avg':>8} {'±SD':>7} {'Test mon':>10} {'Test ctm':>10}")
    print("  " + "-" * 75)

    for name, model_raw in models_raw.items():
        model = _wrap(model_raw, name)

        fold_r2_mon, fold_r2_ctm, fold_r2_avg = [], [], []

        for tr_idx, val_idx in kf.split(X_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            m = _metrics(y_train[val_idx], model.predict(X_train[val_idx]))
            fold_r2_mon.append(m[t0]["r2"])
            fold_r2_ctm.append(m[t1]["r2"])
            fold_r2_avg.append(m["mean"]["r2"])

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        m_test      = _metrics(y_test, y_pred_test)

        cv_results[name] = {
            "r2_mon_mean": np.mean(fold_r2_mon),
            "r2_ctm_mean": np.mean(fold_r2_ctm),
            "r2_avg_mean": np.mean(fold_r2_avg),
            "r2_avg_sd":   np.std(fold_r2_avg),
            "r2_mean":     np.mean(fold_r2_avg),  # alias for select_best_model
            "r2_sd":       np.std(fold_r2_avg),   # alias
            "rmse_mean":   m_test["mean"]["rmse"],
            "mae_mean":    m_test["mean"]["mae"],
        }
        test_results[name] = {
            "r2_mon":  m_test[t0]["r2"],
            "r2_ctm":  m_test[t1]["r2"],
            "r2_avg":  m_test["mean"]["r2"],
            "r2":      m_test["mean"]["r2"],       # alias for plots
            "rmse":    m_test["mean"]["rmse"],      # alias
            "mae":     m_test["mean"]["mae"],       # alias
            "rmse_mon": m_test[t0]["rmse"],
            "rmse_ctm": m_test[t1]["rmse"],
            "mae_mon":  m_test[t0]["mae"],
            "mae_ctm":  m_test[t1]["mae"],
            "y_pred":   y_pred_test,
        }
        fold_r2s[name] = np.array(fold_r2_avg)
        trained[name]  = model

        print(f"  {name:<12} "
              f"{np.mean(fold_r2_mon):>11.3f} "
              f"{np.mean(fold_r2_ctm):>11.3f} "
              f"{np.mean(fold_r2_avg):>8.3f} "
              f"{np.std(fold_r2_avg):>7.3f} "
              f"{m_test[t0]['r2']:>10.3f} "
              f"{m_test[t1]['r2']:>10.3f}")

    return cv_results, test_results, fold_r2s, trained


def select_best_model(cv_results, test_results):
    """Best model = highest test R² (avg) minus CV SD."""
    rows = []
    for name in cv_results:
        rows.append({
            "model":     name,
            "cv_r2":     cv_results[name]["r2_avg_mean"],
            "cv_sd":     cv_results[name]["r2_avg_sd"],
            "test_r2":   test_results[name]["r2_avg"],
            "test_r2_mon": test_results[name]["r2_mon"],
            "test_r2_ctm": test_results[name]["r2_ctm"],
        })
    df = pd.DataFrame(rows)
    df["score"] = df["test_r2"] - df["cv_sd"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    best = df.loc[0, "model"]

    print(f"\n[Selection]  Model ranking:")
    print(df[["model","cv_r2","cv_sd","test_r2_mon","test_r2_ctm","score"]]
          .to_string(index=False))
    print(f"\n  ★  Best: {best}  (avg R²={test_results[best]['r2_avg']:.3f}  "
          f"mon={test_results[best]['r2_mon']:.3f}  "
          f"ctm={test_results[best]['r2_ctm']:.3f})")
    return best, df


def paired_ttest_vs_best(fold_r2s, best_name):
    """Paired t-test vs best model."""
    best_r2s = fold_r2s[best_name]
    results  = {}
    print(f"\n[Statistics]  Paired t-test vs {best_name}:")
    for name, r2s in fold_r2s.items():
        if name == best_name:
            results[name] = {"t": 0.0, "p": 1.0, "sig": "—"}
            continue
        t, p = stats.ttest_rel(best_r2s, r2s)
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        results[name] = {"t": t, "p": p, "sig": sig}
        print(f"  {name:<12}  t={t:+.3f}  p={p:.4f}  {sig}")
    return results


def save_results_csv(cv_results, test_results, output_dir):
    """Save all metrics to CSV."""
    import os
    rows = []
    for name in cv_results:
        rows.append({
            "model":         name,
            "cv_r2_avg":     round(cv_results[name]["r2_avg_mean"], 4),
            "cv_sd":         round(cv_results[name]["r2_avg_sd"],   4),
            "test_r2_mon":   round(test_results[name]["r2_mon"],    4),
            "test_r2_ctm":   round(test_results[name]["r2_ctm"],    4),
            "test_r2_avg":   round(test_results[name]["r2_avg"],    4),
            "rmse_mon":      round(test_results[name]["rmse_mon"],  3),
            "rmse_ctm":      round(test_results[name]["rmse_ctm"],  3),
        })
    df = pd.DataFrame(rows).sort_values("test_r2_avg", ascending=False)
    path = os.path.join(output_dir, "results_summary.csv")
    df.to_csv(path, index=False)
    print(f"[Train]  Results saved → {path}")
    return df
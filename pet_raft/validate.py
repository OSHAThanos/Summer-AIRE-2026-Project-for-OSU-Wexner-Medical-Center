"""
validate.py
-----------
External validation on unseen monomers.
Mirrors paper Table 1 and the generalizability assessment.
"""

import numpy as np
import pandas as pd

from pet_raft.descriptors import build_feature_matrix
from pet_raft.features import engineer_features, align_features
from pet_raft.config import PIPELINE


def external_validate(model, scaler, selected_features,
                       ext_df, y_true,
                       variance_threshold=None,
                       correlation_threshold=None):
    """
    Run external validation on unseen monomers.

    Parameters
    ----------
    model             : fitted best model
    scaler            : fitted StandardScaler from training
    selected_features : list of feature names from training set
    ext_df            : pd.DataFrame with same schema as training df
                        (monomer_smiles, raft_smiles, pc_smiles,
                         raft_rse, raft_frag_eff, raft_delta_h)
    y_true            : array-like — ground truth conversion values
    variance_threshold    : passed to engineer_features
    correlation_threshold : passed to engineer_features

    Returns
    -------
    results_df : pd.DataFrame with monomer, actual, predicted, abs_error
    mae        : float
    rmse       : float
    """
    cfg = PIPELINE
    variance_threshold    = variance_threshold    or cfg["variance_threshold"]
    correlation_threshold = correlation_threshold or cfg["correlation_threshold"]

    print("\n[Validate]  External validation on unseen monomers...")

    # Compute descriptors
    X_ext_raw = build_feature_matrix(ext_df)

    # Feature engineering
    X_ext_eng = engineer_features(X_ext_raw, variance_threshold,
                                   correlation_threshold)

    # Align to training feature set (add missing cols as 0, drop extras)
    X_ext_aligned = align_features(X_ext_eng, selected_features)

    # Scale using training scaler
    X_ext_scaled = scaler.transform(X_ext_aligned)

    # Predict
    y_pred = model.predict(X_ext_scaled)
    y_true = np.array(y_true)

    abs_errors = np.abs(y_true - y_pred)
    mae  = float(abs_errors.mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

    results_df = pd.DataFrame({
        "monomer":   ext_df["monomer"].values,
        "actual":    y_true,
        "predicted": np.round(y_pred, 2),
        "abs_error": np.round(abs_errors, 2),
    })

    print(f"\n        {'Monomer':<10} {'Actual':>8} {'Predicted':>10} {'|Error|':>9}")
    print("        " + "-" * 42)
    for _, row in results_df.iterrows():
        print(f"        {row['monomer']:<10} {row['actual']:>8.1f} "
              f"{row['predicted']:>10.2f} {row['abs_error']:>9.2f}")
    print(f"\n        MAE  = {mae:.2f} pps")
    print(f"        RMSE = {rmse:.2f} pps")

    return results_df, mae, rmse

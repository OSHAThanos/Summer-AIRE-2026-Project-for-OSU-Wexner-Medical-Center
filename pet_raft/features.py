"""
features.py
-----------
Feature engineering pipeline:
  1. Handle NaN / Inf values
  2. Variance filter  (threshold < 0.05)
  3. Correlation filter (|r| > 0.95)
  4. Z-score normalization (fit on train, transform test)

Mirrors the paper's preprocessing (208 raw → ~83 retained features).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Replace Inf with NaN, then fill NaN with column median."""
    X = X.replace([np.inf, -np.inf], np.nan)
    n_nan = X.isna().any(axis=0).sum()
    if n_nan > 0:
        X = X.fillna(X.median())
        print(f"[Features]    NaN/Inf: filled {n_nan} columns with column median")
    return X


def variance_filter(X: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Remove near-zero variance features."""
    mask = X.var() >= threshold
    n_dropped = (~mask).sum()
    X = X.loc[:, mask]
    print(f"[Features]    Variance filter (< {threshold}): "
          f"dropped {n_dropped} → {X.shape[1]} remaining")
    return X


def correlation_filter(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated features (keep the first of each pair).
    Preserves collinearity while minimising redundancy.
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    X = X.drop(columns=drop_cols)
    print(f"[Features]    Correlation filter (|r| > {threshold}): "
          f"dropped {len(drop_cols)} → {X.shape[1]} remaining")
    return X


def engineer_features(X: pd.DataFrame,
                       variance_threshold: float = 0.05,
                       correlation_threshold: float = 0.95) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Apply to the entire dataset before train/test split.

    Parameters
    ----------
    X                     : raw feature matrix from build_feature_matrix()
    variance_threshold    : features with var < this are dropped
    correlation_threshold : feature pairs with |r| > this, one is dropped

    Returns
    -------
    pd.DataFrame — filtered feature matrix
    """
    n_start = X.shape[1]
    print(f"[Features]    Engineering: {n_start} raw features")

    X = clean_features(X)
    X = variance_filter(X, variance_threshold)
    X = correlation_filter(X, correlation_threshold)

    print(f"[Features]    ✓  Final: {X.shape[1]} features "
          f"(reduced from {n_start})")
    return X


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit z-score normalizer on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def align_features(X: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """
    Align a feature matrix to a reference column list.
    Adds missing columns as 0, drops extra columns.
    Used to align external validation features to training features.
    """
    for col in reference_columns:
        if col not in X.columns:
            X[col] = 0.0
    return X[reference_columns]

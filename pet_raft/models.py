"""
models.py
---------
Instantiates all 10 ML models evaluated in the paper.
Hyperparameters are centralised in config.py.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from pet_raft.config import MODEL_PARAMS


def get_models() -> dict:
    """
    Return all 10 models from the paper as a name -> estimator dict.
    Order matches the paper's presentation (linear → kernel →
    instance-based → ensemble → neural network).
    """
    p = MODEL_PARAMS
    return {
        # --- Linear / regularised ---
        "LR":       LinearRegression(**p["LR"]),
        "Ridge":    Ridge(**p["Ridge"]),
        "Lasso":    Lasso(**p["Lasso"]),

        # --- Kernel ---
        "SVR":      SVR(**p["SVR"]),

        # --- Instance-based ---
        "k-NN":     KNeighborsRegressor(**p["k-NN"]),

        # --- Tree ensembles ---
        "RF":       RandomForestRegressor(**p["RF"]),
        "GBR":      GradientBoostingRegressor(**p["GBR"]),
        "XGBoost":  XGBRegressor(**p["XGBoost"]),
        "CatBoost": CatBoostRegressor(**p["CatBoost"]),

        # --- Neural network ---
        "MLP":      MLPRegressor(**p["MLP"]),
    }


# Models expected to perform best (for ordering in plots)
ENSEMBLE_MODELS  = ["RF", "GBR", "XGBoost", "CatBoost"]
LINEAR_MODELS    = ["LR", "Ridge", "Lasso"]
OTHER_MODELS     = ["SVR", "k-NN", "MLP"]
MODEL_ORDER      = LINEAR_MODELS + OTHER_MODELS + ENSEMBLE_MODELS

# Color palette for plots (matching paper style)
MODEL_COLORS = {
    "CatBoost": "#E84855",
    "XGBoost":  "#2E86AB",
    "RF":       "#3BB273",
    "GBR":      "#7B2D8B",
    "LR":       "#888888",
    "Ridge":    "#AAAAAA",
    "Lasso":    "#BBBBBB",
    "SVR":      "#FF9F1C",
    "k-NN":     "#FFBF69",
    "MLP":      "#CBF3F0",
}

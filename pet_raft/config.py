"""
config.py
---------
Configuration for the CTM kinetic prediction pipeline.

TWO TARGET VARIABLES (predicted simultaneously):
  monomer_conv  — monomer conversion % at each timepoint
  ctm_conv      — CTM conversion % at each timepoint

Components (confirmed from Bablu's structural drawings):
  4 monomers:   nBA, NIPAM, TEGMA, Styrene
  3 CTMs:       CDTPA-HEMA, CDTPA-HEAA, CDTPA-APMA
  Feed ratios:  5, 10, 25, 50 mol% CTM
  Timepoints:   fixed across all experiments (confirm with PI)

TO ADD A NEW CTM:
  Add one entry to CTM_LIBRARY. Everything else updates automatically.

TO SWITCH TO REAL DATA:
  In run.py set USE_REAL_DATA = True and REAL_DATA_PATH = "data/your_file.csv"
  Your CSV needs: monomer | ctm | ctm_feed_mol_pct | time_hours | monomer_conv | ctm_conv
"""

# =============================================================================
# MONOMER LIBRARY
# Format: name -> (SMILES, base_conv_estimate)
# base_conv_estimate used ONLY for synthetic data — NOT by the ML pipeline
# =============================================================================
MONOMER_LIBRARY = {
    "nBA":     ("C=CC(=O)OCCCC",          65.0),  # n-butyl acrylate
    "NIPAM":   ("C=CC(=O)NC(C)C",         72.0),  # N-isopropylacrylamide
    "TEGMA":   ("C=C(C)C(=O)OCCOCCOCCO",  55.0),  # triethylene glycol methacrylate
    "Styrene": ("C=Cc1ccccc1",             40.0),  # styrene
}

# =============================================================================
# CTM LIBRARY
# All three structures confirmed from Bablu's structural drawings (image6.png)
# Atom counts verified: HEMA=C25N1O6S3, HEAA=C24N2O5S3, APMA=C26N3O4S3
#
# Shared backbone: dodecyl-S-C(=S)-S-C(CN)(CH2COOH)(CH2-linker-vinyl)
# Dodecyl = C12 chain
#
# Format: name -> (SMILES, rse, frag_eff, delta_h)
#   rse      = radical stabilization energy (kcal/mol)
#   frag_eff = fragmentation efficiency (0-1)
#   delta_h  = enthalpy of RAFT stability (kcal/mol)
# =============================================================================
CTM_LIBRARY = {

    # CDTPA-HEMA — confirmed C25N1O6S3
    # Linker:    ester + ethylene glycol  -C(=O)-O-CH2CH2-O-
    # Vinyl end: methacrylate             -C(=O)-C(CH3)=CH2
    "CDTPA-HEMA": (
        "C=C(C)C(=O)OCCOC(=O)CC(C#N)(CC(=O)O)SC(=S)SCCCCCCCCCCCC",
        -31.5, 0.88, -7.4,
    ),

    # CDTPA-HEAA — confirmed C24N2O5S3
    # Linker:    ester + ethylene + amide  -C(=O)-O-CH2CH2-NH-
    # Vinyl end: acrylate                  -C(=O)-CH=CH2
    "CDTPA-HEAA": (
        "C=CC(=O)NCCOC(=O)CC(C#N)(CC(=O)O)SC(=S)SCCCCCCCCCCCC",
        -31.5, 0.86, -7.1,
    ),

    # CDTPA-APMA — confirmed C26N3O4S3
    # Linker:    double amide + propyl spacer  -C(=O)-NH-CH2CH2CH2-NH-
    # Vinyl end: methacrylamide                -C(=O)-C(CH3)=CH2
    # Key: no ether oxygen, two amide bonds, propyl (not ethyl) spacer
    "CDTPA-APMA": (
        "C=C(C)C(=O)NCCCNC(=O)CC(C#N)(CC(=O)O)SC(=S)SCCCCCCCCCCCC",
        -31.5, 0.85, -6.9,
    ),

    # ── Add new CTMs here as Bablu sends structures ───────────────────────
    # "CDTPA-NewCTM": ("SMILES", rse, frag_eff, delta_h),
    
   
# "CDTPA-NewCTM": (
#       "SMILES_string_here",
#       None, None, None,
# ),


}

# =============================================================================
# FEED RATIOS (mol% CTM)
# =============================================================================
CTM_FEED_RATIOS = [5, 10, 25, 50]

# =============================================================================
# TIMEPOINTS (hours) — fixed across all experiments
# Update once PI confirms exact set (1,2,3,4,5,6 or 1,2,4,6,8,10)
# =============================================================================
TIMEPOINTS = [1, 2, 3, 4, 5, 6, 8]

# =============================================================================
# TARGET VARIABLES — two outputs predicted simultaneously
# =============================================================================
TARGET_COLUMNS = ["monomer_conv", "ctm_conv"]

# =============================================================================
# EXTERNAL VALIDATION
# =============================================================================
EXTERNAL_MONOMERS = {}
EXTERNAL_Y_TRUE   = {}

# =============================================================================
# PIPELINE SETTINGS
# =============================================================================
PIPELINE = {
    "n_synthetic_samples": 200,
    "random_state":        42,
    "variance_threshold":    0.05,
    "correlation_threshold": 0.95,
    "test_size":  0.20,
    "n_cv_folds": 5,
    "shap_top_n": 10,
    "output_dir": "outputs",
    "dpi":        150,
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
MODEL_PARAMS = {
    "LR":       {},
    "Ridge":    {"alpha": 1.0},
    "Lasso":    {"alpha": 0.1, "max_iter": 5000},
    "SVR":      {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
    "k-NN":     {"n_neighbors": 5},
    "RF":       {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    "GBR":      {"n_estimators": 200, "learning_rate": 0.05,
                 "max_depth": 4, "random_state": 42},
    "XGBoost":  {"n_estimators": 200, "learning_rate": 0.05,
                 "max_depth": 4, "random_state": 42, "verbosity": 0},
    "CatBoost": {"iterations": 500, "learning_rate": 0.05, "depth": 6,
                 "random_state": 42, "verbose": 0, "loss_function": "RMSE"},
    "MLP":      {"hidden_layer_sizes": (128, 64, 32), "max_iter": 1000,
                 "random_state": 42, "early_stopping": True,
                 "validation_fraction": 0.1},
}

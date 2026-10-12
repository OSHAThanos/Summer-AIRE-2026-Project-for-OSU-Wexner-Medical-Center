# Data Directory

Place your real PET-RAFT dataset here as a CSV file.

## Required columns

| Column            | Type   | Description                                      |
|-------------------|--------|--------------------------------------------------|
| `monomer_smiles`  | str"/Users/dhoopshikhabasgeet/PET-RAFT Polymerization Prediction/.venv/bin/python" /Users/dhoopshikhabasgeet/PET-RAFT Polymerization Prediction/run.py 
=================================================================
PET-RAFT Monomer Conversion Prediction Pipeline
Replication: Alemdag et al. (2026) Sci. Rep. 16:5947
=================================================================

[Step 1]  Dataset
[Dataset]  Synthetic: 152 samples | mean=48.0% | SD=20.1% | range=4.0–98.0%

[Step 2]  Molecular descriptor calculation
[Descriptors]  Computing RDKit 2D descriptors...
152 samples × 3 components × 217 descriptors
✓  Raw feature matrix: 152 × 654

[Step 3]  Feature engineering
[Features]    Engineering: 654 raw features
[Features]    NaN/Inf: filled 8 columns with column median
[Features]    Variance filter (< 0.05): dropped 383 → 271 remaining
[Features]    Correlation filter (|r| > 0.95): dropped 205 → 66 remaining
[Features]    ✓  Final: 66 features (reduced from 654)

[Step 4]  Train/test split (80/20) + z-score normalisation
Train: 121 | Test: 31 | Features: 66

[Step 5]  Model training and 5-fold CV evaluation

[Train]  5-fold CV across 10 models

Model           CV R²     ±SD   CV RMSE   CV MAE   Test R²     RMSE      MAE
--------------------------------------------------------------------------------
LR              0.501   0.084     13.67    11.19     0.674    10.84     8.44
Ridge           0.504   0.085     13.62    11.14     0.675    10.82     8.41
Lasso           0.516   0.084     13.46    10.96     0.676    10.80     8.42
SVR             0.025   0.171     19.13    16.02     0.100    18.01    14.31
k-NN           -0.012   0.253     19.48    15.89     0.221    16.76    12.77
RF              0.414   0.162     14.68    11.50     0.413    14.55    11.35
GBR             0.233   0.241     16.73    12.89     0.239    16.56    13.07
XGBoost         0.245   0.228     16.63    12.98     0.280    16.11    12.68
CatBoost        0.290   0.203     16.14    12.56     0.293    15.96    12.62
MLP             0.151   0.300     17.71    14.20     0.509    13.30    10.47

[Step 6]  Model selection

[Selection]  Stability-adjusted model ranking:
model     cv_r2    cv_sd  test_r2     score
Lasso  0.515610 0.083764 0.676073  0.592309
Ridge  0.503621 0.085310 0.675243  0.589932
LR  0.500537 0.084198 0.674127  0.589929
RF  0.413645 0.161776 0.412610  0.250834
MLP  0.151091 0.300191 0.508916  0.208725
CatBoost  0.289730 0.203132 0.293319  0.090187
XGBoost  0.244532 0.228035 0.280224  0.052189
GBR  0.232560 0.241220 0.238714 -0.002507
k-NN -0.012125 0.252688 0.220649 -0.032040
SVR  0.024607 0.170728 0.100111 -0.070617

★  Best model: Lasso (test R²=0.676, SD=0.084)

[Step 7]  Paired t-test comparison

[Statistics]  Paired t-test vs Lasso:
LR            t=+6.420  p=0.0030  **
Ridge         t=+7.118  p=0.0021  **
SVR           t=+7.214  p=0.0020  **
k-NN          t=+4.081  p=0.0151  *
RF            t=+2.384  p=0.0756  n.s.
GBR           t=+3.385  p=0.0277  *
XGBoost       t=+3.424  p=0.0267  *
CatBoost      t=+3.261  p=0.0310  *
MLP           t=+2.528  p=0.0648  n.s.

[Step 8]  SHAP analysis

[SHAP]  Running analysis on Lasso...
100%|██████████| 31/31 [00:01<00:00, 29.56it/s]
KernelExplainer — approximate SHAP values

        Top 10 features by mean |SHAP value|:
        Rank  Feature                                        Mean |SHAP|
        ----------------------------------------------------------------
        1     Mon_VSA_EState7                                     6.9996  [Monomer]
        2     Mon_SMR_VSA5                                        5.7989  [Monomer]
        3     Mon_MolWt                                           4.0824  [Monomer]
        4     RAFT_EState_VSA4                                    1.0609  [RAFT]
        5     RAFT_EState_VSA7                                    0.7237  [RAFT]
        6     Photo_HallKierAlpha                                 0.6451  [PC]
        7     RAFT_VSA_EState9                                    0.6359  [RAFT]
        8     RAFT_SlogP_VSA4                                     0.6002  [RAFT]
        9     RAFT_BalabanJ                                       0.5541  [RAFT]
        10    RAFT_fr_aryl_methyl                                 0.4957  [RAFT]

        Top 3 features account for 76.9% of mean |SHAP| sum

[Step 9]  External validation

[Validate]  External validation on unseen monomers...
[Descriptors]  Computing RDKit 2D descriptors...
4 samples × 3 components × 217 descriptors
✓  Raw feature matrix: 4 × 654
[Features]    Engineering: 654 raw features
[Features]    NaN/Inf: filled 8 columns with column median
[Features]    Variance filter (< 0.05): dropped 571 → 83 remaining
[Features]    Correlation filter (|r| > 0.95): dropped 70 → 13 remaining
[Features]    ✓  Final: 13 features (reduced from 654)

        Monomer      Actual  Predicted   |Error|
        ------------------------------------------
        MA             87.0      54.62     32.38
        DEA            92.0      54.62     37.38
        MMA            65.0      54.62     10.38
        NAM            95.0      54.62     40.38

        MAE  = 30.13 pps
        RMSE = 32.34 pps

[Step 10] Generating figures...
/Users/dhoopshikhabasgeet/PET-RAFT Polymerization Prediction/.venv/lib/python3.13/site-packages/sklearn/utils/validation.py:2684: UserWarning: X has feature names, but StandardScaler was fitted without feature names
warnings.warn(
[Plot]  Saved → outputs/fig1_conversion_distribution.png
[Plot]  Saved → outputs/fig2_parity_plots.png
[Plot]  Saved → outputs/fig3_cv_comparison.png
[Plot]  Saved → outputs/fig4_shap_global.png
[Plot]  Saved → outputs/fig5_shap_waterfall.png
[Plot]  Saved → outputs/fig6_external_validation.png
[Train]  Results saved → outputs/results_summary.csv

=================================================================
PIPELINE COMPLETE
=================================================================
Data source:        Synthetic
Samples:            152
Final features:     66
Best model:         Lasso
Test R²:            0.676
Test RMSE:          10.80 pps
Test MAE:           8.42 pps
CV R² (mean ± SD):  0.516 ± 0.084
External MAE:       30.13 pps

Top SHAP features:
[Mon] Mon_VSA_EState7                           6.9996
[Mon] Mon_SMR_VSA5                              5.7989
[Mon] Mon_MolWt                                 4.0824
[RAFT] RAFT_EState_VSA4                          1.0609
[RAFT] RAFT_EState_VSA7                          0.7237

Outputs saved to:   ./outputs/
=================================================================

Process finished with exit code 0ing | SMILES string of the monomer                     |
| `raft_smiles`     | string | SMILES string of the RAFT agent                  |
| `pc_smiles`       | string | SMILES string of the photocatalyst               |
| `raft_rse`        | float  | Radical stabilization energy (kcal/mol)          |
| `raft_frag_eff`   | float  | Fragmentation efficiency (dimensionless, 0–1)    |
| `raft_delta_h`    | float  | Enthalpy of RAFT stability (kcal/mol)            |
| `conversion`      | float  | Monomer conversion % — TARGET VARIABLE (0–100)   |

## Optional label columns (used in plots only)

| Column          | Type   | Example       |
|-----------------|--------|---------------|
| `monomer`       | string | DMA           |
| `raft_agent`    | string | BTPA          |
| `photocatalyst` | string | ZnTPP         |

## Source

This pipeline was built to replicate the dataset from:
  Ng, G. et al. (2021). Screening RAFT agents and photocatalysts to mediate
  PET-RAFT polymerization using a high throughput approach.
  Polymer Chemistry, 12, 6548–6560.

The thermodynamic parameters (RSE, fragmentation efficiency, ΔH) for
RAFT agents can be sourced from:
  Ng, G. et al. (2021) — Supplementary Information, Table S1.

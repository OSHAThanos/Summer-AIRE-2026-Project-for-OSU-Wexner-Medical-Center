# CTM Kinetic Conversion Prediction
### Machine learning pipeline for predicting monomer and CTM conversion kinetics
### Based on the methodology of Alemdag et al. (2026) — *Scientific Reports* 16:5947

---

## What this project does

Predicts **monomer conversion %** and **CTM conversion %** simultaneously over time for copolymerisation reactions between a monomer and a chain transfer monomer (CTM).

CTMs are bifunctional molecules — they act as both a RAFT chain transfer agent and a comonomer. The pipeline learns how CTM molecular structure, monomer identity, feed ratio, and reaction time together determine the kinetic conversion profile of each species.

---

## Your system

| Component | Options |
|---|---|
| Monomers | nBA, NIPAM, TEGMA, Styrene |
| CTMs | CDTPA-HEMA, CDTPA-HEAA, CDTPA-APMA (expandable) |
| Feed ratios | 5, 10, 25, 50 mol% CTM |
| Timepoints | Fixed across all experiments |
| Targets | monomer_conv (%) and ctm_conv (%) |

---

## Project structure

```
CTM-Kinetic-Prediction/
├── run.py                  ← ONLY FILE YOU RUN
├── requirements.txt
├── data/
│   ├── kinetic_data_processed.csv   ← Bablu's real kinetic data
│   └── README_data.md               ← how to format your CSV
├── outputs/                ← all figures saved here automatically
└── pet_raft/
    ├── __init__.py
    ├── config.py           ← CTM/monomer structures + settings
    ├── dataset.py          ← synthetic generator + real data loader
    ├── descriptors.py      ← RDKit descriptor calculation
    ├── features.py         ← variance/correlation filtering
    ├── models.py           ← 10 ML models
    ├── train.py            ← CV training loop (two-output)
    ├── explain.py          ← SHAP analysis (both targets)
    ├── validate.py         ← external validation
    └── plots.py            ← all 6 figures including kinetic curves
```

---

## Setup in IntelliJ

1. **Open project**: File → Open → select this folder
2. **Activate virtual environment** in terminal:
   ```bash
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run**: right-click `run.py` → Run

---

## Switching between synthetic and real data

Open `run.py` and change these two lines:

```python
# Synthetic data (for testing the pipeline)
USE_REAL_DATA  = False

# Real data (Bablu's NMR measurements)
USE_REAL_DATA  = True
REAL_DATA_PATH = "data/kinetic_data_processed.csv"
```

---

## Real data CSV format

Your CSV needs exactly these 6 columns:

| Column | What it is | Example |
|---|---|---|
| `monomer` | monomer name | `TEGMA` |
| `ctm` | CTM name | `CDTPA-HEMA` |
| `ctm_feed_mol_pct` | mol% CTM in feed | `10` |
| `time_hours` | reaction time in hours | `2` |
| `monomer_conv` | monomer conversion % from ¹H NMR | `60.0` |
| `ctm_conv` | CTM conversion % from ¹H NMR | `46.8` |

SMILES strings and thermodynamic parameters are looked up automatically from `config.py` — you never write them in your CSV.

---

## Adding a new CTM

Open `pet_raft/config.py` and add one entry to `CTM_LIBRARY`:

```python
"CDTPA-NewCTM": (
    "SMILES_string_here",
    rse_value,
    frag_eff_value,
    delta_h_value,
),
```

Then use the name in your CSV. Everything else updates automatically.

---

## Outputs (saved to `outputs/` folder)

| File | What it shows |
|---|---|
| `fig1_conversion_distribution.png` | Histogram of both conversion targets |
| `fig2_parity_plots.png` | Predicted vs actual for all 10 models |
| `fig3_cv_comparison.png` | CV stability comparison |
| `fig4_shap_global.png` | Which descriptors matter most per target |
| `fig5_shap_waterfall.png` | Why specific reactions gave high/low conversion |
| `fig6_kinetic_curves.png` | Actual vs predicted conversion curves over time |
| `results_summary.csv` | All model metrics in one table |

---

## Current status

- [x] Pipeline built and validated on synthetic data
- [x] Updated for CTM kinetic system (two-output: monomer + CTM conversion)
- [x] Three CTM structures verified from Bablu's structural drawings
- [x] Running on Bablu's 21 real data points (TEGMA + 3 CTMs at 10 mol%)
- [ ] Waiting for glovebox kinetic data — more monomers and feed ratios
- [ ] Full model training on complete dataset (~250 points expected)

---

## Why the predictions are not yet accurate

The current real dataset has 21 data points — all from the same monomer (TEGMA) at one feed ratio (10 mol%). With only 16 training rows the model cannot learn the kinetic shape reliably. It needs data from at least 2-3 monomers across multiple feed ratios before predictions become meaningful.

The kinetic curves figure (Fig 6) is the most useful output at this stage — it shows the actual experimental data clearly and demonstrates the dramatic differences between the three CTM systems.

---

## Key finding from current data

The three CTMs show completely different kinetic profiles despite identical conditions:
- **CDTPA-APMA** — slow incorporation with a lag phase, both species track closely
- **CDTPA-HEAA** — CTM incorporates explosively fast (83% by hour 2 vs 1% monomer)
- **CDTPA-HEMA** — steady parallel incorporation, monomer slightly faster throughout

The SHAP analysis identified `CTM_PEOE_VSA1` (partial charge surface area of the CTM) as the top predictor — this descriptor changes directly with O vs N substitution in the linker group.

---

## Reference

Alemdag, B., Kocaarslan, A., & Kabay, G. (2026).
*Leveraging molecular descriptors and explainable machine learning for monomer
conversion prediction in photoinduced electron transfer-reversible
addition-fragmentation chain transfer polymerization.*
Scientific Reports, 16, 5947.
https://doi.org/10.1038/s41598-025-33553-y

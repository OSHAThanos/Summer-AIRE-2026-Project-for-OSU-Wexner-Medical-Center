"""
dataset.py
----------
Data preparation for the two-output kinetic prediction pipeline.

TARGET VARIABLES:
  monomer_conv — monomer conversion % at each timepoint
  ctm_conv     — CTM conversion % at each timepoint

REAL DATA CSV FORMAT:
  monomer | ctm | ctm_feed_mol_pct | time_hours | monomer_conv | ctm_conv
  TEGMA   | CDTPA-HEMA | 10 | 1 | 29.0 | 13.17
  TEGMA   | CDTPA-HEMA | 10 | 2 | 60.0 | 46.80
  ...

Bablu's processed data: data/kinetic_data_processed.csv (21 real points)
"""

import numpy as np
import pandas as pd
from pet_raft.config import (MONOMER_LIBRARY, CTM_LIBRARY,
                              CTM_FEED_RATIOS, TIMEPOINTS,
                              TARGET_COLUMNS, PIPELINE)


def generate_synthetic_dataset(n_samples=None, random_state=None):
    """
    Generate synthetic kinetic dataset with two output variables.

    Kinetics follow a first-order-like model per CTM:
      conversion(t) = max_conv × (1 - exp(-rate × t))

    Each CTM has different rates for monomer vs CTM incorporation,
    based on Bablu's observed kinetic behaviour:
      CDTPA-HEMA: monomer faster throughout
      CDTPA-HEAA: CTM races ahead dramatically early
      CDTPA-APMA: both slow, tracking each other closely
    """
    cfg          = PIPELINE
    n_samples    = n_samples    or cfg["n_synthetic_samples"]
    random_state = random_state or cfg["random_state"]

    rng       = np.random.RandomState(random_state)
    mon_names = list(MONOMER_LIBRARY.keys())
    ctm_names = list(CTM_LIBRARY.keys())

    CTM_KINETICS = {
        "CDTPA-HEMA": {"mon_rate": 0.45, "ctm_rate": 0.30,
                        "mon_max": 95,   "ctm_max": 90},
        "CDTPA-HEAA": {"mon_rate": 0.20, "ctm_rate": 0.90,
                        "mon_max": 75,   "ctm_max": 93},
        "CDTPA-APMA": {"mon_rate": 0.25, "ctm_rate": 0.20,
                        "mon_max": 70,   "ctm_max": 58},
    }

    records = []
    for _ in range(n_samples):
        mon      = rng.choice(mon_names)
        ctm      = rng.choice(ctm_names)
        feed_pct = rng.choice(CTM_FEED_RATIOS)
        time     = rng.choice(TIMEPOINTS)

        mon_smiles, base_conv          = MONOMER_LIBRARY[mon]
        ctm_smiles, rse, frag_eff, dh  = CTM_LIBRARY[ctm]
        kin                            = CTM_KINETICS[ctm]

        mon_scale = base_conv / 65.0
        mon_conv  = kin["mon_max"] * mon_scale * (1 - np.exp(-kin["mon_rate"] * time))
        ctm_conv  = kin["ctm_max"] * (1 - np.exp(-kin["ctm_rate"] * time))

        # Feed ratio effect on monomer conversion
        mon_conv += -(feed_pct - 10) * 0.3

        # Experimental noise
        mon_conv += rng.normal(0, 5)
        ctm_conv += rng.normal(0, 5)

        mon_conv = float(np.clip(mon_conv, 0, 98))
        ctm_conv = float(np.clip(ctm_conv, 0, 98))

        records.append({
            "monomer":          mon,
            "ctm":              ctm,
            "ctm_feed_mol_pct": feed_pct,
            "time_hours":       time,
            "monomer_smiles":   mon_smiles,
            "ctm_smiles":       ctm_smiles,
            "ctm_rse":          rse,
            "ctm_frag_eff":     frag_eff,
            "ctm_delta_h":      dh,
            "monomer_conv":     round(mon_conv, 2),
            "ctm_conv":         round(ctm_conv, 2),
        })

    df = pd.DataFrame(records)
    print(f"[Dataset]  Synthetic kinetic dataset: {len(df)} samples")
    print(f"           monomer_conv — mean={df['monomer_conv'].mean():.1f}% | "
          f"range={df['monomer_conv'].min():.1f}–{df['monomer_conv'].max():.1f}%")
    print(f"           ctm_conv     — mean={df['ctm_conv'].mean():.1f}% | "
          f"range={df['ctm_conv'].min():.1f}–{df['ctm_conv'].max():.1f}%")
    print(f"           CTMs:        {df['ctm'].value_counts().to_dict()}")
    print(f"           Timepoints:  {sorted(df['time_hours'].unique().tolist())}")
    return df


def load_real_dataset(path):
    """
    Load real kinetic dataset from CSV.

    Required columns:
        monomer           — must match MONOMER_LIBRARY key
        ctm               — must match CTM_LIBRARY key
        ctm_feed_mol_pct  — mol% CTM in feed
        time_hours        — reaction time in hours
        monomer_conv      — monomer conversion % (TARGET 1)
        ctm_conv          — CTM conversion %     (TARGET 2)

    SMILES and thermodynamic params auto-filled from config.py.
    """
    required = ["monomer", "ctm", "ctm_feed_mol_pct",
                "time_hours", "monomer_conv", "ctm_conv"]

    df = pd.read_csv(path)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Required: monomer | ctm | ctm_feed_mol_pct | "
            f"time_hours | monomer_conv | ctm_conv"
        )

    unknown_mon = set(df["monomer"]) - set(MONOMER_LIBRARY.keys())
    if unknown_mon:
        raise ValueError(f"Unknown monomers: {unknown_mon} — add to config.py")

    unknown_ctm = set(df["ctm"]) - set(CTM_LIBRARY.keys())
    if unknown_ctm:
        raise ValueError(f"Unknown CTMs: {unknown_ctm} — add to config.py")

    df["monomer_smiles"] = df["monomer"].map({k: v[0] for k, v in MONOMER_LIBRARY.items()})
    df["ctm_smiles"]     = df["ctm"].map({k: v[0] for k, v in CTM_LIBRARY.items()})
    df["ctm_rse"]        = df["ctm"].map({k: v[1] for k, v in CTM_LIBRARY.items()})
    df["ctm_frag_eff"]   = df["ctm"].map({k: v[2] for k, v in CTM_LIBRARY.items()})
    df["ctm_delta_h"]    = df["ctm"].map({k: v[3] for k, v in CTM_LIBRARY.items()})

    print(f"[Dataset]  Real data: {len(df)} samples from '{path}'")
    print(f"           monomer_conv — mean={df['monomer_conv'].mean():.1f}% | "
          f"range={df['monomer_conv'].min():.1f}–{df['monomer_conv'].max():.1f}%")
    print(f"           ctm_conv     — mean={df['ctm_conv'].mean():.1f}% | "
          f"range={df['ctm_conv'].min():.1f}–{df['ctm_conv'].max():.1f}%")
    print(f"           CTMs:        {df['ctm'].value_counts().to_dict()}")
    print(f"           Timepoints:  {sorted(df['time_hours'].unique().tolist())}")
    return df

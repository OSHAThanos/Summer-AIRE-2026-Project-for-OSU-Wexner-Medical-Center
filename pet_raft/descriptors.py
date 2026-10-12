"""
descriptors.py
--------------
SMILES → RDKit 2D molecular descriptors for the CTM kinetic pipeline.

Features:
  Mon_*             — monomer descriptors    (~208)
  CTM_*             — CTM descriptors        (~208)
  CTM_thermo_*      — thermodynamic params   (3)
  ctm_feed_mol_pct  — feed ratio             (1)
  time_hours        — reaction time          (1)  ← kinetic variable

Total raw features: ~421 before filtering
"""

import numpy as np
import pandas as pd
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
]
_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


def smiles_to_descriptors(smiles: str, prefix: str = "") -> dict:
    
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def cached_descriptor(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return
    
    values = cached_descriptor(smiles)

    """Compute all RDKit 2D descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"Could not parse SMILES: {smiles!r} — filling with NaN")
        return {f"{prefix}{n}": np.nan for n in DESCRIPTOR_NAMES}
    try:
        values = _CALCULATOR.CalcDescriptors(mol)
    except Exception as e:
        warnings.warn(f"Descriptor calculation failed: {e}")
        values = [np.nan] * len(DESCRIPTOR_NAMES)
    return {f"{prefix}{n}": v for n, v in zip(DESCRIPTOR_NAMES, values)}


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for kinetic prediction.

    Mon_* descriptors from monomer SMILES
    CTM_* descriptors from CTM SMILES
    Numerical features added directly:
      ctm_rse, ctm_frag_eff, ctm_delta_h  — thermodynamic
      ctm_feed_mol_pct                     — condition variable
      time_hours                           — kinetic variable
    """
    print("[Descriptors]  Computing RDKit 2D descriptors...")
    print(f"               {len(df)} samples × 2 components × "
          f"{len(DESCRIPTOR_NAMES)} descriptors + 5 numerical features")

    mon_records = [smiles_to_descriptors(s, "Mon_") for s in df["monomer_smiles"]]
    ctm_records = [smiles_to_descriptors(s, "CTM_") for s in df["ctm_smiles"]]

    mon_df = pd.DataFrame(mon_records)
    ctm_df = pd.DataFrame(ctm_records)

    numerical_df = pd.DataFrame({
        "CTM_thermo_RSE":     df["ctm_rse"].values,
        "CTM_thermo_FragEff": df["ctm_frag_eff"].values,
        "CTM_thermo_DeltaH":  df["ctm_delta_h"].values,
        "ctm_feed_mol_pct":   df["ctm_feed_mol_pct"].values,
        "time_hours":         df["time_hours"].values,
    })

    X = pd.concat(
        [mon_df.reset_index(drop=True),
         ctm_df.reset_index(drop=True),
         numerical_df.reset_index(drop=True)],
        axis=1,
    )

    print(f"               ✓  Raw feature matrix: {X.shape[0]} × {X.shape[1]}")
    return X

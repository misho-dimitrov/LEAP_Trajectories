#!/usr/bin/env python
"""run_growth_curves.py
=======================
Growth-curve models for developmental FC (wDC) trajectories.

Parcellation strategy
---------------------
* **Cortical** – Schaefer-200 parcels collapsed to Yeo networks (7 or 17).
* **Subcortical** – Bilateral regions: average of Left/Right for each
  structure (Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala,
  Accumbens) + Brainstem (already bilateral).
* **Cerebellar** – Lobules mapped to 7 MDTB functional domains
  (King et al. 2019; Guell et al. 2018; Buckner et al. 2011):
    1. Cereb_Motor        (I-IV, V)
    2. Cereb_Attention     (VI)
    3. Cereb_Executive     (Crus I)
    4. Cereb_DMN           (Crus II)
    5. Cereb_Language      (VIIb)
    6. Cereb_Somatomotor   (VIIIa, VIIIb)
    7. Cereb_Limbic        (IX, X)

Models
------
  Model 1 – Age model (cross-sectional + longitudinal pooled):
      wDC ~ age × group + sex + (1 + age | subject)              [LMM]
      wDC ~ bs(age) × group + sex + (1 | subject)                [GAMM]

  Model 2 – Longitudinal model:
      wDC ~ time_elapsed × baseline_age × group + sex
            + (1 + time_elapsed | subject)                       [LMM]
      wDC ~ bs(time_elapsed) × group + baseline_age × group
            + sex + (1 | subject)                                [GAMM]

LMM models include random slopes; GAMM models are intercept-only
(random slopes on spline bases are rarely identifiable).

Random-slope identifiability is assessed after fitting:
  - fit.converged must be True
  - the random-slope variance must exceed _MIN_SLOPE_VAR
Slopes that fail either check are flagged slope_reliable=False in
random_slopes.csv and should not be used for downstream analysis.

Both are fit as:
  (a) LMM  – fully parametric linear mixed model
  (b) GAMM – B-spline basis expansion inside a LMM

Permutation testing
-------------------
Group labels are shuffled across subjects (preserving within-subject
repeated-measures structure); all models are re-fit on the permuted data.
Per-term permutation p-values are computed as:
    p_perm = (n_perm_|z| >= observed_|z| + 1) / (n_perms + 1)

FDR correction
--------------
Benjamini–Hochberg FDR is applied to both parametric (``p``) and
permutation (``p_perm``) p-values.  Default scope:
  ``per_model_term`` – correct across networks within each model × term
  combination.  This is the standard neuroimaging approach that answers
  "which regions show a significant age × group interaction?"
Other scopes: ``per_model``, ``all``.

Synthetic T3
------------
When master_wdc.csv contains only T1/T2, synthetic T3 wDC is generated
for a random subset of eligible subjects (those with a valid t3_ageyrs).
Once real T3 rows are added to master_wdc.csv the synthetic generation
is automatically skipped – no code changes needed.

Usage
-----
    python run_growth_curves.py                      # defaults
    python run_growth_curves.py --yeo 17             # 17-network cortical
    python run_growth_curves.py --n-perms 0          # skip permutations
    python run_growth_curves.py --n-perms 5000       # full permutation run
    python run_growth_curves.py --no-synthetic-t3
    python run_growth_curves.py --help
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------
#  Path helpers
# ---------------------------------------------------------------------------
try:
    from rs_paths import (
        default_master_wdc_csv,
        default_behaviour_csv,
        default_qc_csv,
        rs_project_root,
    )
except ImportError:
    def rs_project_root() -> Path:
        return Path(__file__).resolve().parents[1]

    def default_master_wdc_csv() -> Path:
        return rs_project_root() / "reports" / "fc" / "master_wdc.csv"

    def default_behaviour_csv() -> Path:
        return rs_project_root().parent / "Behaviour" / "df.csv"

    def default_qc_csv(timepoint: int = 1) -> Path:
        root = rs_project_root()
        stem = f"LEAP{int(timepoint)}_RS-fMRI_QC_final"
        return root / f"{stem}.csv"


def default_growth_curves_output_dir() -> Path:
    return rs_project_root() / "reports" / "growth_curves"


LOG = logging.getLogger("growth_curves")


# ============================================================================
#  CONSTANTS
# ============================================================================

# --- Cortical: Yeo networks ------------------------------------------------

_CORTICAL_RE = re.compile(
    r"^7Networks_([LR]H)_([A-Za-z]+)(?:_([A-Za-z]+))?_(\d+)$"
)

YEO_7_ORDER = [
    "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
]

# Approximate 7-net sub-region → 17-net mapping (Schaefer-200 based on 7-net
# atlas, so some networks cannot be split further).
YEO_17_FROM_SUBREGION: Dict[str, str] = {
    "Vis":                      "Vis",
    "SomMot":                   "SomMot",
    "DorsAttn_FEF":             "DorsAttn_A",
    "DorsAttn_PrCv":            "DorsAttn_A",
    "DorsAttn_Post":            "DorsAttn_B",
    "SalVentAttn_FrOperIns":    "SalVentAttn_A",
    "SalVentAttn_Med":          "SalVentAttn_A",
    "SalVentAttn_PrC":          "SalVentAttn_A",
    "SalVentAttn_ParOper":      "SalVentAttn_A",
    "SalVentAttn_PFCl":         "SalVentAttn_B",
    "SalVentAttn_TempOccPar":   "SalVentAttn_B",
    "Limbic_OFC":               "Limbic_A",
    "Limbic_TempPole":          "Limbic_B",
    "Cont_Par":                 "Cont_A",
    "Cont_PFCl":                "Cont_A",
    "Cont_Cing":                "Cont_A",
    "Cont_pCun":                "Cont_A",
    "Cont_PFCmp":               "Cont_B",
    "Cont_Temp":                "Cont_B",
    "Cont_OFC":                 "Cont_C",
    "Cont_PFCv":                "Cont_C",
    "Default_Par":              "Default_A",
    "Default_PFC":              "Default_A",
    "Default_pCunPCC":          "Default_A",
    "Default_PFCdPFCm":         "Default_B",
    "Default_Temp":             "Default_B",
    "Default_PFCv":             "Default_C",
    "Default_PHC":              "Default_C",
}

# --- Subcortical: bilateral structures -------------------------------------

SUBCORTICAL_STRUCTURES = {
    "Accumbens", "Amygdala", "Caudate", "Hippocampus",
    "Pallidum", "Putamen", "Thalamus",
}

# --- Cerebellar: MDTB 7-domain mapping ------------------------------------
# Based on King et al. (2019) "Functional boundaries in the human cerebellum
# revealed by a multi-domain task battery", Guell et al. (2018), and
# Buckner et al. (2011).  Since parcels are at the lobule level (not voxel),
# the mapping assigns each lobule to its *dominant* functional domain.
#
# The Diedrichsen lab atlas repo (https://github.com/DiedrichsenLab/
# cerebellar_atlases) provides voxel-level parcellations (e.g.
# atl-NettekovenSym32) that could refine this if sub-lobular labels become
# available.

MDTB_LOBULE_TO_DOMAIN: Dict[str, str] = {
    "I-IV":    "Cereb_Motor",          # primary motor
    "V":       "Cereb_Motor",          # primary motor
    "VI":      "Cereb_Attention",      # attentional / premotor bridge
    "Crus I":  "Cereb_Executive",      # executive / working memory
    "Crus II": "Cereb_DMN",            # default-mode / mentalising
    "VIIb":    "Cereb_Language",       # language processing
    "VIIIa":   "Cereb_Somatomotor",   # secondary somatomotor
    "VIIIb":   "Cereb_Somatomotor",   # secondary somatomotor
    "IX":      "Cereb_Limbic",         # limbic / vestibular
    "X":       "Cereb_Limbic",         # limbic / vestibular
}

MDTB_DOMAIN_ORDER = sorted(set(MDTB_LOBULE_TO_DOMAIN.values()))

# --- Group / plotting constants --------------------------------------------

GROUP_LABEL_MAP = {1: "TD", 2: "ASD", 3: "ID_control", 4: "ID_ASD"}
GROUP_REF = "TD"
MISSING_SENTINEL = 999

GROUP_COLOURS: Dict[str, str] = {
    "TD":         "#2196F3",
    "ASD":        "#F44336",
    "ID_ASD":     "#FF9800",
    "ID_control": "#4CAF50",
}

# Random-effects formula for LMM models (None = intercept-only).
# GAMMs are always intercept-only; random slopes on spline bases are
# rarely identifiable with ≤3 timepoints per subject.
LMM_RE_FORMULAS: Dict[str, Optional[str]] = {
    "M1_LMM": "~age_c",  # age_c = age − mean(age), created in _prepare_network_df
    "M2_LMM": "~time_elapsed",
}

# Minimum random-slope variance below which an estimate is flagged as
# unreliable (boundary / near-zero solution).
_MIN_SLOPE_VAR: float = 1e-6


# ============================================================================
#  Parcel → Network assignment
# ============================================================================

def _parse_cortical_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (yeo7_network, subregion_key) for a cortical Schaefer label."""
    m = _CORTICAL_RE.match(label)
    if m is None:
        return None, None
    network = m.group(2)
    subregion = m.group(3)
    key = f"{network}_{subregion}" if subregion else network
    return network, key


def assign_network(
    label: str,
    n_networks: int = 7,
    include_subcortical: bool = True,
) -> Optional[str]:
    """Map a parcel *label* to an analysis-level region name.

    Cortical parcels → Yeo network.
    Subcortical → bilateral structure name (e.g. ``Thalamus``).
    Cerebellar → MDTB 7-domain (e.g. ``Cereb_Executive``).
    Brainstem → ``Brainstem``.
    """
    # --- Cortical ---
    net7, subkey = _parse_cortical_label(label)
    if net7 is not None:
        return net7 if n_networks == 7 else YEO_17_FROM_SUBREGION.get(subkey, net7)

    # --- Non-cortical ---
    if not include_subcortical:
        return None

    if label == "Brain-Stem":
        return "Brainstem"

    # Vermis cerebellar lobules → MDTB domain
    if label.startswith("Vermis "):
        lobule = label[len("Vermis "):]
        return MDTB_LOBULE_TO_DOMAIN.get(lobule)

    # Left / Right → subcortical bilateral OR cerebellar MDTB domain
    for side in ("Left ", "Right "):
        if label.startswith(side):
            structure = label[len(side):]
            if structure in SUBCORTICAL_STRUCTURES:
                return structure                       # e.g. "Thalamus"
            return MDTB_LOBULE_TO_DOMAIN.get(structure)  # e.g. "Cereb_Motor"

    return None


# ============================================================================
#  Data loading & preparation
# ============================================================================

def load_wdc(path: Path) -> pd.DataFrame:
    """Load master_wdc.csv → tidy long DataFrame."""
    LOG.info("Loading wDC from %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    assert {"subject", "site", "timepoint", "label", "wdc"}.issubset(df.columns), (
        f"Expected columns {{subject, site, timepoint, label, wdc}}, got {set(df.columns)}"
    )

    df = df[df["subject"].notna()].copy()
    df["subject"] = (
        df["subject"]
        .astype(float)   # already float, but be explicit
        .astype(int)     # drop .0
        .astype(str)     # now safe to use str methods
        .str[:6]         # first 6 digits                                # → "111297"
    )

    # T3 uses "London_KCL" folder — unify to "KCL" so ComBat treats it as one site
    df["site"] = df["site"].replace({"London_KCL": "KCL"})

    return df


# ============================================================================
#  Preprocessing: FD residualization & ComBat
# ============================================================================

def _load_qc_fd(
    qc_path: Path,
    subject_col: str = "subject",
    fd_col: str = "meanFD",
) -> pd.DataFrame:
    """Load a QC file and return a subject → meanFD mapping.

    Handles naming variations across LEAP timepoints:
      LEAP1: subject / meanFD   (TSV)
      LEAP2: subjects / meanFD  (CSV)
      LEAP3: Subjects / Mean_FD (CSV)
    """
    qc = None
    for reader in (lambda p: pd.read_csv(p, sep="\t"), pd.read_csv):
        try:
            qc = reader(qc_path)
            if len(qc.columns) > 1:
                break
        except Exception:
            continue
    if qc is None or len(qc.columns) <= 1:
        LOG.warning("Could not read QC file %s", qc_path)
        return pd.DataFrame(columns=[subject_col, fd_col])

    # Normalise subject column: subjects / Subjects → subject
    col_lower = {c: c.lower().rstrip("s") if c.lower().rstrip("s") == "subject" else c
                 for c in qc.columns}
    # Normalise FD column: Mean_FD → meanFD
    for c in list(col_lower.keys()):
        if c.lower().replace("_", "") == "meanfd":
            col_lower[c] = fd_col
    qc = qc.rename(columns=col_lower)

    if subject_col not in qc.columns or fd_col not in qc.columns:
        LOG.warning("QC file %s missing required columns (%s, %s); found %s",
                    qc_path, subject_col, fd_col, list(qc.columns))
        return pd.DataFrame(columns=[subject_col, fd_col])

    qc = qc[[subject_col, fd_col]].copy()
    qc[subject_col] = qc[subject_col].astype(str).str.strip().str[:6]
    qc[fd_col] = pd.to_numeric(qc[fd_col], errors="coerce")
    return qc


def attach_fd_to_wdc(
    wdc: pd.DataFrame,
    qc_paths: Dict[str, Path],
    subject_col: str = "subject",
    fd_col: str = "meanFD",
) -> pd.DataFrame:
    """Attach meanFD to the long-format parcel-level wDC table.

    Parameters
    ----------
    qc_paths : dict
        Mapping of timepoint label ("T1", "T2", "T3") → QC file path.
    """
    wdc = wdc.copy()
    wdc["__meanFD"] = np.nan
    for tp, qc_path in qc_paths.items():
        if qc_path is None or not qc_path.exists():
            LOG.info("No QC file for %s – skipping FD attachment.", tp)
            continue
        qc = _load_qc_fd(qc_path, subject_col, fd_col)
        if qc.empty:
            continue
        tp_mask = wdc["timepoint"] == tp
        tp_subjects = wdc.loc[tp_mask, subject_col].values
        fd_map = qc.set_index(subject_col)[fd_col]
        wdc.loc[tp_mask, "__meanFD"] = [
            fd_map.get(s, np.nan) for s in tp_subjects
        ]
        n_matched = wdc.loc[tp_mask, "__meanFD"].notna().sum() // max(
            wdc.loc[tp_mask, "label"].nunique(), 1
        )
        LOG.info("%s: attached meanFD for %d subjects.", tp, n_matched)
    return wdc


def residualize_wdc_on_fd(wdc: pd.DataFrame) -> pd.DataFrame:
    """Regress wDC on meanFD within each timepoint × parcel and replace with residuals."""
    if "__meanFD" not in wdc.columns:
        return wdc
    wdc = wdc.copy()
    for tp in wdc["timepoint"].unique():
        for lbl in wdc["label"].unique():
            mask = (
                (wdc["timepoint"] == tp)
                & (wdc["label"] == lbl)
                & wdc["__meanFD"].notna()
            )
            if mask.sum() < 3:
                continue
            fd = wdc.loc[mask, "__meanFD"].values.astype(float)
            w = wdc.loc[mask, "wdc"].values.astype(float)
            valid = np.isfinite(fd) & np.isfinite(w)
            if valid.sum() < 3:
                continue
            # Fit linear regression: wdc ~ meanFD (per parcel)
            x = fd[valid]
            y = w[valid]
            slope = np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0) if np.var(x, ddof=0) > 0 else 0.0
            intercept = np.mean(y) - slope * np.mean(x)
            residuals = w.copy()
            residuals[valid] = y - (slope * x + intercept) + np.mean(y)
            wdc.loc[mask, "wdc"] = residuals
    LOG.info("FD residualization complete.")
    return wdc


def combat_harmonize_wdc(
    wdc: pd.DataFrame,
    demo: pd.DataFrame,
    age_col: str = "t1_ageyrs",
    sex_col: str = "t1_sex",
    group_col: str = "t1_group",
    iq_col: str = "t1_fsiq",
) -> pd.DataFrame:
    """Apply ComBat harmonization to parcel-level wDC per timepoint.

    Biological covariates (age, sex, group) are included so that ComBat
    preserves variance associated with them.
    """
    try:
        from neuroHarmonize import harmonizationLearn
    except ImportError:
        LOG.warning("neuroHarmonize not installed – skipping ComBat.")
        return wdc

    wdc = wdc.copy()
    # Build a subject-level demographics lookup
    demo_sub = demo[["subject"]].copy()
    for col in (age_col, sex_col, group_col, iq_col):
        if col in demo.columns:
            demo_sub[col] = demo[col].values
    demo_sub["subject"] = demo_sub["subject"].astype(str).str.strip().str[:6]
    demo_sub = demo_sub.drop_duplicates(subset=["subject"])

    age_map = dict(zip(demo_sub["subject"], pd.to_numeric(demo_sub.get(age_col, pd.Series(dtype=float)), errors="coerce"))) if age_col in demo_sub.columns else {}
    if sex_col in demo_sub.columns:
        sex_codes = pd.Categorical(demo_sub[sex_col].astype(str)).codes.astype(float)
        sex_codes[sex_codes < 0] = np.nan
        sex_map = dict(zip(demo_sub["subject"], sex_codes))
    else:
        sex_map = {}
    if group_col in demo_sub.columns:
        grp_codes = pd.Categorical(demo_sub[group_col].astype(str)).codes.astype(float)
        grp_codes[grp_codes < 0] = np.nan
        grp_map = dict(zip(demo_sub["subject"], grp_codes))
    else:
        grp_map = {}
    iq_map = dict(zip(demo_sub["subject"], pd.to_numeric(demo_sub.get(iq_col, pd.Series(dtype=float)), errors="coerce"))) if iq_col in demo_sub.columns else {}

    for tp in sorted(wdc["timepoint"].unique()):
        tp_mask = wdc["timepoint"] == tp
        tp_df = wdc.loc[tp_mask]

        # Pivot to wide: subjects × parcels
        tp_wide = tp_df.pivot_table(
            index="subject", columns="label", values="wdc", aggfunc="mean"
        )
        if tp_wide.shape[0] < 5:
            LOG.info("%s: too few subjects (%d) for ComBat – skipping.", tp, tp_wide.shape[0])
            continue

        subj_order = tp_wide.index.tolist()
        parcels = tp_wide.columns.tolist()
        x_arr = tp_wide.values.astype(float)

        # Build covariates
        site_per_subj = []
        for s in subj_order:
            s_sites = tp_df.loc[tp_df["subject"] == s, "site"].astype(str).unique()
            site_per_subj.append(s_sites[0] if len(s_sites) > 0 else "unknown")

        cov = pd.DataFrame({"SITE": site_per_subj})
        if age_map:
            cov["AGE"] = [age_map.get(s, np.nan) for s in subj_order]
        if sex_map:
            cov["SEX"] = [sex_map.get(s, np.nan) for s in subj_order]
        if grp_map:
            cov["GROUP"] = [grp_map.get(s, np.nan) for s in subj_order]
        if iq_map:
            cov["IQ"] = [iq_map.get(s, np.nan) for s in subj_order]

        # Drop zero-variance features
        v = np.var(x_arr, axis=0)
        keep = np.isfinite(v) & (v > 0)
        if not np.any(keep):
            continue

        x_sub = x_arr[:, keep]
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                learn_result = harmonizationLearn(x_sub, cov)
            if isinstance(learn_result, (tuple, list)) and len(learn_result) == 2:
                _, x_adj = learn_result
            else:
                continue
        except Exception as exc:
            LOG.warning("%s: ComBat failed (%s) – skipping.", tp, exc)
            continue

        x_adj_arr = np.asarray(x_adj, dtype=float)
        finite_mask = np.isfinite(x_adj_arr)
        if not finite_mask.all():
            LOG.warning(
                "%s: ComBat produced %d non-finite values – keeping originals for those entries.",
                tp, int((~finite_mask).sum()),
            )
            # Only overwrite entries where ComBat produced finite results
            orig = x_arr[:, keep].copy()
            x_arr[:, keep] = x_adj_arr
            x_arr[:, keep][~finite_mask] = orig[~finite_mask]
        else:
            x_arr[:, keep] = x_adj_arr

        # Map adjusted values back to long format
        adj_wide = pd.DataFrame(x_arr, index=subj_order, columns=parcels)
        adj_long = adj_wide.stack().reset_index()
        adj_long.columns = ["subject", "label", "wdc_adj"]

        # Merge adjusted values back
        tp_idx = wdc.index[tp_mask]
        wdc_tp = wdc.loc[tp_idx].merge(
            adj_long, on=["subject", "label"], how="left"
        )
        adjusted_mask = wdc_tp["wdc_adj"].notna()
        if adjusted_mask.any():
            wdc.loc[tp_idx[adjusted_mask.values], "wdc"] = wdc_tp.loc[adjusted_mask, "wdc_adj"].values

        LOG.info("%s: ComBat harmonization applied (%d subjects, %d parcels).",
                 tp, len(subj_order), int(np.sum(keep)))

    return wdc


def load_demographics(path: Path) -> pd.DataFrame:
    """Load behaviour/demographics CSV → cleaned DataFrame."""
    LOG.info("Loading demographics from %s", path)
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip().str.lower()

    if "subjects" in raw.columns and "subject" not in raw.columns:
        raw = raw.rename(columns={"subjects": "subject"})

    raw["subject"] = raw["subject"].astype(str).str.strip().str[:6] # ← truncate to first 6 digits
    
    keep = [
        "subject", "t1_ageyrs", "t2_ageyrs", "t3_ageyrs",
        "t1_sex", "t1_group", "t1_fsiq",
    ]
    if "group" in raw.columns:
        keep.append("group")
    keep = [c for c in keep if c in raw.columns]
    df = raw[keep].copy()

    for col in ("t1_ageyrs", "t2_ageyrs", "t3_ageyrs", "t1_sex", "t1_fsiq"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] >= MISSING_SENTINEL, col] = np.nan

    if "group" in df.columns:
        df["group_str"] = (
            df["group"]
            .astype(str)
            .str.replace("-", "_", regex=False)
            .replace({"nan": np.nan, "999": np.nan})
        )
    elif "t1_group" in df.columns:
        df["group_str"] = df["t1_group"].map(GROUP_LABEL_MAP)
    else:
        raise ValueError("No group column found in demographics CSV.")

    df = df[df["group_str"].isin(["ASD", "TD"])].copy()
    LOG.info("Keeping %d subjects after group filter (ASD + TD only).", len(df))

    if "t1_sex" in df.columns:
        df["sex"] = df["t1_sex"] - 1

    if "t1_fsiq" in df.columns:
        df["iq"] = df["t1_fsiq"]

    return df


def collapse_to_networks(
    wdc: pd.DataFrame,
    n_networks: int = 7,
    include_subcortical: bool = True,
) -> pd.DataFrame:
    """Average wDC within regions for each subject × timepoint.

    Cortical → Yeo networks (mean across parcels in the same network).
    Subcortical → bilateral mean (Left + Right average per structure).
    Cerebellar → MDTB domain mean (Left + Right + Vermis per domain).
    """
    wdc = wdc.copy()
    wdc["network"] = wdc["label"].apply(
        lambda lbl: assign_network(lbl, n_networks, include_subcortical)
    )
    wdc = wdc.dropna(subset=["network"])
    agg = (
    wdc
    .groupby(["subject", "timepoint", "network", "site"], as_index=False)["wdc"]
    .mean()
    )
    n_cortical = agg["network"].isin(
        YEO_7_ORDER if n_networks == 7
        else sorted(set(YEO_17_FROM_SUBREGION.values()))
    ).sum()
    LOG.info(
        "Collapsed %d parcels → %d regions "
        "(yeo-%d cortical, subcortical+cerebellar=%s): %s",
        wdc["label"].nunique(), agg["network"].nunique(),
        n_networks, include_subcortical,
        sorted(agg["network"].unique()),
    )
    return agg


# ============================================================================
#  Synthetic T3 generation
# ============================================================================

def generate_synthetic_t3(
    wdc_net: pd.DataFrame,
    demo: pd.DataFrame,
    frac: float = 0.40,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic T3 wDC rows for a random subset of eligible subjects.

    Eligible = subjects who have T2 wDC **and** a valid t3_ageyrs.
    Synthetic wDC = subject's T2 value + Gaussian noise calibrated to
    5 % of per-network T2 standard deviation.
    """
    rng = np.random.default_rng(seed)

    t2_subs = set(wdc_net.loc[wdc_net["timepoint"] == "T2", "subject"].unique())
    t3_age_subs = set(demo.loc[demo["t3_ageyrs"].notna(), "subject"].unique())
    already_t3 = set(wdc_net.loc[wdc_net["timepoint"] == "T3", "subject"].unique())

    eligible = sorted((t2_subs & t3_age_subs) - already_t3)
    if not eligible:
        LOG.info("No eligible subjects for synthetic T3 – skipping.")
        return pd.DataFrame(columns=["subject", "timepoint", "network", "wdc", "synthetic", "site"])
    n_select = max(1, int(len(eligible) * frac))
    selected = set(rng.choice(eligible, size=n_select, replace=False))

    LOG.info(
        "Generating synthetic T3 for %d / %d eligible subjects (%.0f%%)",
        n_select, len(eligible), frac * 100,
    )

    t2_data = wdc_net[wdc_net["timepoint"] == "T2"]
    net_sd = t2_data.groupby("network")["wdc"].std().to_dict()

    rows: list[dict] = []
    for _, r in t2_data.iterrows():
        if r["subject"] not in selected:
            continue
        sd = net_sd.get(r["network"], 1.0)
        rows.append({
            "subject":   r["subject"],
            "timepoint": "T3",
            "network":   r["network"],
            "wdc":       r["wdc"] + rng.normal(0, 0.05 * sd),
            "synthetic": True,
            "site":      r.get("site", np.nan),
        })

    wdc_t3 = pd.DataFrame(rows)
    LOG.info("  → %d synthetic T3 rows created.", len(wdc_t3))
    return wdc_t3


# ============================================================================
#  Build analysis-ready long-format table
# ============================================================================

def build_long_df(
    wdc_net: pd.DataFrame,
    demo: pd.DataFrame,
) -> pd.DataFrame:
    """Merge network wDC with demographics; compute time_elapsed & baseline_age.

    Returns one row per subject × timepoint × network with columns:
        subject, timepoint, time_elapsed, age, baseline_age,
        group, sex, network, wdc, synthetic
    """
    age_cols = {"T1": "t1_ageyrs", "T2": "t2_ageyrs", "T3": "t3_ageyrs"}
    demo_rows: list[dict] = []
    for _, r in demo.iterrows():
        base_age = r.get("t1_ageyrs", np.nan)
        for tp_str, age_col in age_cols.items():
            age_val = r.get(age_col, np.nan)
            if pd.isna(age_val):
                continue
            demo_rows.append({
                "subject":      r["subject"],
                "timepoint":    tp_str,
                "age":          float(age_val),
                "baseline_age": float(base_age) if pd.notna(base_age) else np.nan,
                "time_elapsed": float(age_val) - float(base_age)
                                if pd.notna(base_age) else np.nan,
                "group":        r.get("group_str", np.nan),
                "sex":          r.get("sex", np.nan),
                "iq":           r.get("iq", np.nan),
            })

    demo_long = pd.DataFrame(demo_rows)
    merged = wdc_net.merge(demo_long, on=["subject", "timepoint"], how="inner")

    if "synthetic" not in merged.columns:
        merged["synthetic"] = False
    merged["synthetic"] = merged["synthetic"].fillna(False).astype(bool)

    required = ["age", "baseline_age", "group", "sex", "iq", "wdc", "site"]
    n_before = len(merged)
    merged = merged.dropna(subset=required)
    n_dropped = n_before - len(merged)
    if n_dropped:
        LOG.info("Dropped %d rows with missing required variables.", n_dropped)

    LOG.info(
        "Long DF ready: %d rows, %d subjects, %d regions, timepoints %s",
        len(merged), merged["subject"].nunique(),
        merged["network"].nunique(), sorted(merged["timepoint"].unique()),
    )
    return merged


# ============================================================================
#  Model fitting helpers
# ============================================================================

def _safe_mixedlm(
    formula: str,
    data: pd.DataFrame,
    groups: str,
    tag: str = "",
    re_formula: Optional[str] = None,
) -> Tuple[Optional[object], bool]:
    """Fit a statsmodels MixedLM; return (result, converged) or (None, False).

    Convergence warnings are captured and logged rather than suppressed so
    callers know whether the estimates are trustworthy.
    """
    caught_conv: List[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula, data=data, groups=data[groups],
                re_formula=re_formula,
            )
            result = model.fit(reml=True, method="lbfgs", maxiter=500)
        for w in caught:
            if issubclass(w.category, (ConvergenceWarning, RuntimeWarning)):
                caught_conv.append(str(w.message))
        if caught_conv:
            LOG.warning(
                "Convergence issues [%s]: %s", tag, "; ".join(caught_conv),
            )
        converged = bool(getattr(result, "converged", True)) and not caught_conv
        return result, converged
    except Exception as exc:
        LOG.warning("Model failed [%s]: %s", tag, exc)
        return None, False


def _extract_coefficients(
    fit, model_tag: str, network: str, converged: bool = True,
) -> List[dict]:
    """Pull fixed-effect coefficients from a fitted MixedLM result."""
    if fit is None:
        return []
    rows = []
    for term in fit.params.index:
        ci = fit.conf_int()
        rows.append({
            "network":   network,
            "model":     model_tag,
            "term":      term,
            "estimate":  fit.params[term],
            "std_err":   fit.bse.get(term, np.nan),
            "z":         fit.tvalues.get(term, np.nan),
            "p":         fit.pvalues.get(term, np.nan),
            "ci_lower":  ci.loc[term, 0] if term in ci.index else np.nan,
            "ci_upper":  ci.loc[term, 1] if term in ci.index else np.nan,
            "nobs":      fit.nobs,
            "ngroups":   fit.k_groups if hasattr(fit, "k_groups") else np.nan,
            "aic":       fit.aic if hasattr(fit, "aic") else np.nan,
            "bic":       fit.bic if hasattr(fit, "bic") else np.nan,
            "converged": converged,
        })
    return rows


def _extract_random_slopes(
    fit,
    model_tag: str,
    network: str,
    slope_var_name: str,
    converged: bool,
) -> List[dict]:
    """Extract per-subject random intercepts and slopes from a fitted MixedLM.

    Returns one row per subject with columns:
        subject, model, network, slope_var_name,
        re_intercept, re_slope, slope_var, slope_reliable

    ``slope_reliable`` is True only when the model converged *and* the
    estimated random-slope variance exceeds ``_MIN_SLOPE_VAR``.
    """
    if fit is None or not hasattr(fit, "random_effects"):
        return []

    cov_re = getattr(fit, "cov_re", None)
    if cov_re is None or cov_re.shape == (1, 1):
        return []  # intercept-only model – nothing to extract

    # When re_formula is used, statsmodels labels cov_re columns after the
    # patsy design matrix: intercept → "Intercept", slope → the variable name
    # (e.g. "age").  Without re_formula the single column is "Group".
    # Look up slope_var_name directly to avoid any ambiguity.
    if slope_var_name not in cov_re.columns:
        LOG.warning(
            "Slope variable '%s' not found in cov_re for %s / %s "
            "(available: %s) – skipping random slope extraction.",
            slope_var_name, model_tag, network, list(cov_re.columns),
        )
        return []
    slope_col = slope_var_name
    slope_var = float(cov_re.loc[slope_col, slope_col])
    slope_reliable = (
        converged
        and np.isfinite(slope_var)
        and slope_var >= _MIN_SLOPE_VAR
    )

    if not slope_reliable:
        LOG.warning(
            "Random slope unreliable for %s / %s "
            "(converged=%s, slope_var=%s).",
            model_tag, network, converged,
            f"{slope_var:.2e}" if np.isfinite(slope_var) else "NaN",
        )

    rows = []
    for subj, re_vals in fit.random_effects.items():
        re_dict = re_vals.to_dict() if hasattr(re_vals, "to_dict") else dict(re_vals)
        # Intercept key is "Intercept" when re_formula is used, "Group" otherwise.
        intercept_key = "Intercept" if "Intercept" in re_dict else "Group"
        rows.append({
            "subject":        subj,
            "model":          model_tag,
            "network":        network,
            "slope_var_name": slope_var_name,
            "re_intercept":   re_dict.get(intercept_key, np.nan),
            "re_slope":       re_dict.get(slope_col, np.nan),
            "slope_var":      slope_var,
            "slope_reliable": slope_reliable,
        })
    return rows


def _build_formulas(bs_df: int, has_longitudinal: bool) -> Dict[str, str]:
    """Return a dict of {model_tag: formula} for all models to fit."""
    formulas: Dict[str, str] = {
        "M1_LMM":  "wdc ~ age_c * C(group) + sex + iq + C(site)",
        "M1_GAMM": f"wdc ~ bs(age_c, df={bs_df}) * C(group) + sex + iq + C(site)",
    }
    if has_longitudinal:
        formulas["M2_LMM"] = (
            "wdc ~ time_elapsed * baseline_age * C(group) + sex + iq + C(site)"
        )
        formulas["M2_GAMM"] = (
            f"wdc ~ bs(time_elapsed, df={bs_df}) * C(group)"
            f" + baseline_age * C(group) + sex + iq + C(site)"
        )
    return formulas


def _prepare_network_df(long_df: pd.DataFrame, network: str) -> Optional[pd.DataFrame]:
    """Subset long_df for *network*, set group as Categorical with TD ref."""
    df = long_df[long_df["network"] == network].copy()
    if len(df) < 30:
        LOG.warning("Skipping '%s': only %d observations.", network, len(df))
        return None
    other_groups = sorted(g for g in df["group"].unique() if g != GROUP_REF)
    df["group"] = pd.Categorical(
        df["group"], categories=[GROUP_REF] + other_groups,
    )
    df["age_c"] = df["age"] - df["age"].mean()
    return df


# ============================================================================
#  Model fitting (parametric)
# ============================================================================

def fit_models_for_network(
    long_df: pd.DataFrame,
    network: str,
    bs_df: int = 4,
) -> Tuple[List[dict], List[dict]]:
    """Fit all models for one *network*.

    Returns
    -------
    fixed_rows : list of dict
        Fixed-effect coefficient rows (one per term per model).
    slope_rows : list of dict
        Per-subject random-slope rows for LMM models only.
    """
    df = _prepare_network_df(long_df, network)
    if df is None:
        return [], []

    tp_counts = df.groupby("subject")["timepoint"].nunique()
    has_longitudinal = tp_counts.mean() >= 1.3
    formulas = _build_formulas(bs_df, has_longitudinal)

    fixed_rows: List[dict] = []
    slope_rows: List[dict] = []

    for model_tag in sorted(formulas):
        re_formula = LMM_RE_FORMULAS.get(model_tag)  # None for GAMMs
        fit, converged = _safe_mixedlm(
            formulas[model_tag], df, "subject",
            tag=f"{model_tag}|{network}",
            re_formula=re_formula,
        )
        fixed_rows.extend(_extract_coefficients(fit, model_tag, network, converged))
        if re_formula is not None:
            slope_var_name = re_formula.lstrip("~").strip()
            slope_rows.extend(
                _extract_random_slopes(
                    fit, model_tag, network, slope_var_name, converged,
                )
            )

    if not has_longitudinal:
        LOG.info(
            "Skipping Model 2 for '%s': mean %.1f timepoints/subject.",
            network, tp_counts.mean(),
        )

    return fixed_rows, slope_rows


def run_all_models(
    long_df: pd.DataFrame,
    bs_df: int = 4,
    n_perms: int = 5000,
    n_jobs: int = -1,
    seed: int = 42,
    fdr_scope: str = "per_model_term",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit growth-curve models (+ optional permutation tests) for every region.

    Returns
    -------
    results : pd.DataFrame
        Fixed-effect results with parametric p-values, permutation p-values
        (``p_perm``), FDR-corrected versions, and a ``converged`` flag.
    random_slopes : pd.DataFrame
        Per-subject random-slope estimates from LMM models, with a
        ``slope_reliable`` flag indicating identifiability.  Only use rows
        where ``slope_reliable`` is True for downstream analysis.
    """
    networks = sorted(long_df["network"].unique())
    LOG.info("Fitting models for %d regions …", len(networks))

    all_rows: List[dict] = []
    all_slope_rows: List[dict] = []
    rng = np.random.default_rng(seed)

    for i, net in enumerate(networks, 1):
        LOG.info("  [%d/%d] %s – fitting parametric models …", i, len(networks), net)
        net_results, net_slopes = fit_models_for_network(long_df, net, bs_df=bs_df)
        all_slope_rows.extend(net_slopes)

        # ── Permutation testing ─────────────────────────────────────────
        if n_perms > 0 and net_results:
            LOG.info(
                "  [%d/%d] %s – running %d permutations (n_jobs=%s) …",
                i, len(networks), net, n_perms,
                "all cores" if n_jobs == -1 else n_jobs,
            )
            perm_pvals = _run_permutations_for_network(
                long_df, net, net_results,
                n_perms=n_perms, n_jobs=n_jobs, bs_df=bs_df,
                seed=rng.integers(0, 2**31),
            )
            for row in net_results:
                key = (row["model"], row["term"])
                row["p_perm"] = perm_pvals.get(key, np.nan)
        else:
            for row in net_results:
                row["p_perm"] = np.nan

        all_rows.extend(net_results)

    results = pd.DataFrame(all_rows)
    if not results.empty:
        results = _add_fdr(results, scope=fdr_scope, p_col="p", fdr_col="p_fdr")
        if n_perms > 0:
            results = _add_fdr(
                results, scope=fdr_scope, p_col="p_perm", fdr_col="p_perm_fdr",
            )

    random_slopes = pd.DataFrame(all_slope_rows)
    return results, random_slopes


# ============================================================================
#  Permutation testing
# ============================================================================

def _permutation_worker(
    perm_seed: int,
    df_net: pd.DataFrame,
    subjects: np.ndarray,
    groups: np.ndarray,
    formulas: Dict[str, str],
    group_ref: str,
    re_formulas: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[Tuple[str, str], float]:
    """Single permutation: shuffle group labels, fit models, return |z| stats."""
    rng = np.random.default_rng(perm_seed)
    perm_groups = rng.permutation(groups)
    group_map = dict(zip(subjects, perm_groups))

    df = df_net.copy()
    df["group"] = df["subject"].map(group_map)
    other = sorted(g for g in df["group"].unique() if g != group_ref)
    df["group"] = pd.Categorical(df["group"], categories=[group_ref] + other)

    z_stats: Dict[Tuple[str, str], float] = {}
    for tag, formula in formulas.items():
        re_formula = (re_formulas or {}).get(tag)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula, data=df, groups=df["subject"],
                    re_formula=re_formula,
                )
                fit = model.fit(reml=True, method="lbfgs", maxiter=500)
            for term in fit.tvalues.index:
                z_stats[(tag, term)] = abs(fit.tvalues[term])
        except Exception:
            pass  # convergence failure → skip this model for this permutation
    return z_stats


def _run_permutations_for_network(
    long_df: pd.DataFrame,
    network: str,
    observed_results: List[dict],
    n_perms: int = 5000,
    n_jobs: int = -1,
    bs_df: int = 4,
    seed: int = 42,
) -> Dict[Tuple[str, str], float]:
    """Run permutation test for one network; return {(model, term): p_perm}."""
    df_net = _prepare_network_df(long_df, network)
    if df_net is None:
        return {}

    # Subject-level group assignments
    sub_group = df_net.groupby("subject")["group"].first()
    subjects = sub_group.index.values
    groups = sub_group.values.astype(str)

    tp_counts = df_net.groupby("subject")["timepoint"].nunique()
    has_longitudinal = tp_counts.mean() >= 1.3
    formulas = _build_formulas(bs_df, has_longitudinal)

    # Observed |z| per (model, term)
    observed_z: Dict[Tuple[str, str], float] = {}
    for row in observed_results:
        z_val = row.get("z", np.nan)
        if not np.isnan(z_val):
            observed_z[(row["model"], row["term"])] = abs(z_val)

    # Generate seeds for reproducible parallel permutations
    rng = np.random.default_rng(seed)
    perm_seeds = rng.integers(0, 2**31, size=n_perms)

    t0 = time.time()
    null_dist: List[dict] = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_permutation_worker)(
            int(s), df_net, subjects, groups, formulas, GROUP_REF, LMM_RE_FORMULAS,
        )
        for s in perm_seeds
    )
    elapsed = time.time() - t0
    LOG.info(
        "    Permutations done in %.1f s (%.3f s/perm).",
        elapsed, elapsed / max(n_perms, 1),
    )

    # Count how often permuted |z| >= observed |z|
    perm_pvals: Dict[Tuple[str, str], float] = {}
    for key, obs_z in observed_z.items():
        count = sum(1 for nd in null_dist if nd.get(key, 0.0) >= obs_z)
        perm_pvals[key] = (count + 1) / (n_perms + 1)

    # Log convergence rate (count per-model convergences, not per-term)
    n_expected = n_perms * len(formulas)
    n_actual = sum(
        1 for nd in null_dist for tag in formulas
        if any(k[0] == tag for k in nd)
    )
    if n_expected > 0:
        conv_rate = n_actual / n_expected * 100
        if conv_rate < 90:
            LOG.warning(
                "    Low convergence: %.0f%% of permutation models converged.",
                conv_rate,
            )

    return perm_pvals


# ============================================================================
#  FDR correction
# ============================================================================

def _add_fdr(
    results: pd.DataFrame,
    scope: str = "per_model_term",
    p_col: str = "p",
    fdr_col: str = "p_fdr",
) -> pd.DataFrame:
    """Append Benjamini–Hochberg FDR-corrected p-values.

    Parameters
    ----------
    scope : {"per_model_term", "per_model", "all"}
        ``per_model_term`` – correct across networks within each model × term
            (standard neuroimaging approach).
        ``per_model`` – correct across all terms × networks within each model.
        ``all`` – correct across all p-values (most conservative).
    """
    results = results.copy()
    results[fdr_col] = np.nan

    if p_col not in results.columns:
        return results

    if scope == "all":
        pvals = results[p_col].values.astype(float)
        valid = ~np.isnan(pvals)
        if valid.sum() >= 2:
            _, fdr, _, _ = multipletests(pvals[valid], method="fdr_bh")
            fdr_full = np.full_like(pvals, np.nan)
            fdr_full[valid] = fdr
            results[fdr_col] = fdr_full
    else:
        groupby_cols = (
            ["model", "term"] if scope == "per_model_term" else ["model"]
        )
        for _, grp in results.groupby(groupby_cols):
            pvals = grp[p_col].values.astype(float)
            valid = ~np.isnan(pvals)
            if valid.sum() < 2:
                continue
            _, fdr, _, _ = multipletests(pvals[valid], method="fdr_bh")
            fdr_full = np.full_like(pvals, np.nan)
            fdr_full[valid] = fdr
            results.loc[grp.index, fdr_col] = fdr_full

    return results


# ============================================================================
#  Plotting
# ============================================================================

def _get_colour(group: str) -> str:
    return GROUP_COLOURS.get(group, "#9E9E9E")


def plot_spaghetti_age(
    long_df: pd.DataFrame, network: str, output_dir: Path,
) -> None:
    """Spaghetti plot: wDC vs age, individual trajectories + group smooth."""
    df = long_df[long_df["network"] == network].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for group in sorted(df["group"].unique()):
        gdf = df[df["group"] == group]
        colour = _get_colour(group)
        for _subj, sdf in gdf.groupby("subject"):
            sdf = sdf.sort_values("age")
            ax.plot(sdf["age"], sdf["wdc"],
                    color=colour, alpha=0.12, linewidth=0.4)
        if len(gdf) > 10:
            smoothed = sm_lowess(
                gdf["wdc"].values, gdf["age"].values,
                frac=0.6, return_sorted=True,
            )
            ax.plot(smoothed[:, 0], smoothed[:, 1],
                    color=colour, linewidth=2.5, label=group)
        else:
            ax.plot([], [], color=colour, linewidth=2.5, label=group)

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Mean wDC")
    ax.set_title(f"{network}")
    ax.legend(title="Group", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"spaghetti_age_{network}.png", dpi=150)
    plt.close(fig)


def plot_spaghetti_time(
    long_df: pd.DataFrame, network: str, output_dir: Path,
) -> None:
    """Spaghetti plot: wDC vs time_elapsed, coloured by group."""
    df = long_df[long_df["network"] == network].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for group in sorted(df["group"].unique()):
        gdf = df[df["group"] == group]
        colour = _get_colour(group)
        for _subj, sdf in gdf.groupby("subject"):
            sdf = sdf.sort_values("time_elapsed")
            ax.plot(sdf["time_elapsed"], sdf["wdc"],
                    color=colour, alpha=0.12, linewidth=0.4)
        if len(gdf) > 10:
            smoothed = sm_lowess(
                gdf["wdc"].values, gdf["time_elapsed"].values,
                frac=0.6, return_sorted=True,
            )
            ax.plot(smoothed[:, 0], smoothed[:, 1],
                    color=colour, linewidth=2.5, label=group)
        else:
            ax.plot([], [], color=colour, linewidth=2.5, label=group)

    ax.set_xlabel("Time elapsed from baseline (years)")
    ax.set_ylabel("Mean wDC")
    ax.set_title(f"{network}")
    ax.legend(title="Group", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"spaghetti_time_{network}.png", dpi=150)
    plt.close(fig)


def generate_all_plots(long_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate spaghetti plots for every region."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    networks = sorted(long_df["network"].unique())
    LOG.info("Generating plots for %d regions …", len(networks))
    for net in networks:
        plot_spaghetti_age(long_df, net, plot_dir)
        plot_spaghetti_time(long_df, net, plot_dir)
    LOG.info("Plots saved to %s", plot_dir)


# ============================================================================
#  Main
# ============================================================================

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Growth-curve models for developmental FC trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--wdc", type=Path, default=None,
        help="Path to master_wdc.csv  (default: auto-detected via rs_paths).",
    )
    p.add_argument(
        "--demo", type=Path, default=None,
        help="Path to demographics CSV (default: auto-detected via rs_paths).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: reports/growth_curves).",
    )
    p.add_argument(
        "--yeo", type=int, choices=[7, 17], default=7,
        help="Yeo network granularity: 7 (default) or 17.",
    )
    p.add_argument(
        "--include-subcortical", action="store_true", default=True,
        help="Include subcortical / cerebellar / brainstem (default: True).",
    )
    p.add_argument(
        "--no-subcortical", dest="include_subcortical", action="store_false",
        help="Drop all non-cortical parcels.",
    )
    p.add_argument(
        "--synthetic-t3", action="store_true", default=True,
        help="Generate synthetic T3 wDC where missing (default: True).",
    )
    p.add_argument(
        "--no-synthetic-t3", dest="synthetic_t3", action="store_false",
        help="Do not generate any synthetic T3 data.",
    )
    p.add_argument(
        "--synthetic-frac", type=float, default=0.40,
        help="Fraction of eligible subjects to synthesise T3 for (default: 0.40).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--bs-df", type=int, default=4,
        help="B-spline degrees of freedom for the GAMM models (default: 4).",
    )
    # ── Preprocessing ───────────────────────────────────────────────────
    p.add_argument(
        "--use-fd-residualize",
        type=lambda x: str(x).lower() == "true", default=True,
        help="Residualize wDC on mean FD before network averaging (default: True).",
    )
    p.add_argument(
        "--use-combat",
        type=lambda x: str(x).lower() == "true", default=True,
        help="Apply ComBat harmonization across sites before network averaging (default: True).",
    )
    p.add_argument(
        "--qc-fd-col", type=str, default="meanFD",
        help="FD column name in the QC files (default: meanFD).",
    )
    # ── Permutation testing ─────────────────────────────────────────────
    p.add_argument(
        "--n-perms", type=int, default=5000,
        help="Number of permutations for inference (default: 5000).  "
             "Set to 0 to skip permutation testing entirely.",
    )
    p.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel jobs for permutations: -1 = all cores (default).",
    )
    # ── FDR ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--fdr-scope",
        choices=["per_model_term", "per_model", "all"],
        default="per_model_term",
        help="FDR correction scope (default: per_model_term). "
             "'per_model_term' corrects across regions within each model × "
             "term – the standard neuroimaging approach.  'per_model' corrects "
             "across all terms × regions within each model.  'all' corrects "
             "across everything (most conservative).",
    )
    # ── Plotting ────────────────────────────────────────────────────────
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    wdc_path  = args.wdc  or default_master_wdc_csv()
    demo_path = args.demo or default_behaviour_csv()
    out_dir   = args.out  or default_growth_curves_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Configuration: yeo=%d, subcortical=%s, n_perms=%d, fdr_scope=%s",
             args.yeo, args.include_subcortical, args.n_perms, args.fdr_scope)

    # 1. Load data -----------------------------------------------------------
    wdc_raw = load_wdc(wdc_path)
    demo    = load_demographics(demo_path)

    # 1b. FD residualization -------------------------------------------------
    if args.use_fd_residualize:
        try:
            qc_paths = {
                "T1": default_qc_csv(timepoint=1),
                "T2": default_qc_csv(timepoint=2),
                "T3": default_qc_csv(timepoint=3),
            }
        except Exception:
            qc_paths = {}
            LOG.warning("Could not resolve QC paths – skipping FD residualization.")
        if qc_paths:
            wdc_raw = attach_fd_to_wdc(wdc_raw, qc_paths, fd_col=args.qc_fd_col)
            wdc_raw = residualize_wdc_on_fd(wdc_raw)

    # 1c. ComBat harmonization -----------------------------------------------
    if args.use_combat:
        wdc_raw = combat_harmonize_wdc(wdc_raw, demo)

    # 2. Collapse parcels to regions -----------------------------------------
    wdc_net = collapse_to_networks(
        wdc_raw,
        n_networks=args.yeo,
        include_subcortical=args.include_subcortical,
    )

    # 3. Synthetic T3 --------------------------------------------------------
    if args.synthetic_t3:
        real_t3 = wdc_net[wdc_net["timepoint"] == "T3"]
        if real_t3.empty or real_t3["subject"].nunique() < 10:
            synth = generate_synthetic_t3(
                wdc_net, demo,
                frac=args.synthetic_frac,
                seed=args.seed,
            )
            wdc_net = pd.concat([wdc_net, synth], ignore_index=True)
        else:
            LOG.info(
                "Real T3 data found for %d subjects – skipping synthetic generation.",
                real_t3["subject"].nunique(),
            )

    # 4. Build long-format analysis table ------------------------------------
    long_df = build_long_df(wdc_net, demo)
    long_path = out_dir / "long_data.csv"
    long_df.to_csv(long_path, index=False)
    LOG.info("Long-format data saved to %s", long_path)

    # 5. Fit models (+ permutation tests) ------------------------------------
    results, random_slopes = run_all_models(
        long_df,
        bs_df=args.bs_df,
        n_perms=args.n_perms,
        n_jobs=args.n_jobs,
        seed=args.seed,
        fdr_scope=args.fdr_scope,
    )

    if not results.empty:
        res_path = out_dir / "model_results.csv"
        results.to_csv(res_path, index=False)
        LOG.info("Model results saved to %s", res_path)
        for model_tag in results["model"].unique():
            sub = results[results["model"] == model_tag]
            sub.to_csv(out_dir / f"results_{model_tag}.csv", index=False)
    else:
        LOG.warning("No model results to save (all models failed?).")

    if not random_slopes.empty:
        rs_path = out_dir / "random_slopes.csv"
        random_slopes.to_csv(rs_path, index=False)
        n_reliable = int(random_slopes["slope_reliable"].sum())
        LOG.info(
            "Random slopes saved to %s (%d / %d subject×network entries reliable).",
            rs_path, n_reliable, len(random_slopes),
        )
    else:
        LOG.info("No random slopes extracted.")

    # 6. Plots ---------------------------------------------------------------
    if not args.no_plots:
        generate_all_plots(long_df, out_dir)

    LOG.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()

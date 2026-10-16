#!/usr/bin/env python3

"""Two-block PLS (SVD) linking rs-fMRI wDC features (T1) to behavioral measures.

Uses pyls (netneurolab/pypyls) for behavioral PLS-SVD analysis:
- Loads wDC imaging features + behavior table
- Merges on subject ID
- Drops special missing codes
- ComBat harmonization across sites
- Residualizes (harmonized) imaging features on mean FD, age, sex and IQ
- Runs behavioral PLS on the full sample via pyls.behavioral_pls()
  (automatic extraction of max modes, built-in permutation & bootstrap testing)
- Cross-validation to assess stability/generalizability of significant modes

Supported models (--models):
  sdq     – SDQ subscales (emotional / conduct / hyperactivity / peer / prosocial, parent)
  prl     – PRL (perseverative errors / Win-Stay / Lose-Shift)
  autism  – SRS rawscore / RBS total / SSP total
  ashq    – ASHQ Total
  tas     – TAS subscales (identify / describe / external thinking)

All models share the same T1 imaging preprocessing pipeline; only the
behavioural columns (and any sign flips) differ. Results land in
{outdir}/{model_name}/.

Workflow:
  1. Run PLS on full sample → modes, saliences, permutation p-values
  2. Cross-validation to assess stability of significant modes:
     For each CV fold:
       - Hold out test participants (~20%)
       - Project held-out brain & behavior data through full-sample saliences:
           u_test = X_test @ brain_saliences
           v_test = Y_test @ behavior_saliences
       - Compute correlation between projected brain and behavior scores
     Average correlations across folds

Notes:
- wDC values are used directly (no PCA dimensionality reduction)
- pyls handles z-scoring internally via cross-correlation matrix
- Mode signs are arbitrary; pyls bootstraps handle Procrustes rotation
- Max number of PLS modes is automatically determined by pyls as min(p, q)

Date: 2026-02-11
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import argparse
import math
import os
from pathlib import Path
from typing import Optional


def _auto_install_requirements_if_missing() -> None:
    """Install Python dependencies if they are missing."""

    if str(os.environ.get("RS_AUTO_INSTALL", "1")).lower() in {"0", "false", "no"}:
        return

    required_modules = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "neuroHarmonize",
        "matplotlib",
        "pyls",
        "tqdm",
        "h5py",
    ]
    missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
    if not missing:
        return

    req_file = Path(__file__).resolve().parent / "requirements.txt"
    if not req_file.exists():
        print(
            "Missing required modules: "
            + ", ".join(missing)
            + f". requirements.txt not found at {req_file}; cannot auto-install.",
            flush=True,
        )
        return

    print(
        "Missing required modules: "
        + ", ".join(missing)
        + f". Installing from {req_file}...",
        flush=True,
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


_auto_install_requirements_if_missing()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt

try:
    import pyls as pyls_pkg
except ImportError:
    raise ImportError(
        "pyls is required. Install via: pip install git+https://github.com/netneurolab/pypyls.git"
    )

try:
    from rs_paths import (
        default_behaviour_csv,
        default_master_wdc_csv,
        default_pls_output_dir,
        default_qc_csv,
    )
except ImportError:  # pragma: no cover
    from scripts.rs_paths import (  # type: ignore
        default_behaviour_csv,
        default_master_wdc_csv,
        default_pls_output_dir,
        default_qc_csv,
    )

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    delayed = None  # type: ignore


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "sdq": {
        "behav_vars": [
            "t1_sdq_emotional_p", "t1_sdq_conduct_p", "t1_sdq_hyperactivity_p",
            "t1_sdq_peer_p", "t1_sdq_prosocial_p",
        ],
        "sign_flip_vars": ["t1_sdq_prosocial_p"],
        "missing_codes": "777,999",
        "label": "SDQ subscales – emotional / conduct / hyperactivity / peer / prosocial (parent)",
    },
    "prl": {
        "behav_vars": ["t1_prl_perE_prop", "t1_prl_WS", "t1_prl_LS"],
        "sign_flip_vars": ["t1_prl_WS", "t1_prl_LS"],
        "missing_codes": "",
        "label": "PRL (perseverative errors / Win-Stay / Lose-Shift)",
    },
    "autism": {
        "behav_vars": ["t1_srs_rawscore", "t1_rbs_total", "t1_ssp_total"],
        "sign_flip_vars": [],
        "missing_codes": "777,999",
        "label": "Autism spectrum (SRS rawscore / RBS total / SSP total)",
    },
    "ashq": {
        "behav_vars": ["t1_ASHQ_total"],
        "sign_flip_vars": [],
        "missing_codes": "777,999",
        "label": "ASHQ Total",
    },
    "tas": {
        "behav_vars": ["t1_tas_identify", "t1_tas_describe", "t1_tas_external"],
        "sign_flip_vars": [],
        "missing_codes": "777,999",
        "label": "TAS subscales – identify / describe / external thinking",
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dir_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {path}")
    return any(path.iterdir())


def _cleanup_previous_outputs_if_manifest_present(outdir: Path) -> None:
    """Remove files listed in an existing manifest.csv (if present)."""
    manifest = outdir / "manifest.csv"
    if not manifest.exists():
        return
    try:
        mf = pd.read_csv(manifest)
    except Exception:
        return
    if "filename" not in mf.columns:
        return
    for name in mf["filename"].dropna().astype(str).tolist():
        p = outdir / name
        try:
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:
            pass


def benjamini_hochberg_fdr(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_pvals * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[sorted_indices] = adjusted
    return restored


def _try_read_table(path: Path) -> pd.DataFrame:
    header = ""
    try:
        with path.open("r") as _fh:
            header = _fh.readline()
    except Exception:
        header = ""

    preferred_seps: list[Optional[str]] = []
    if "\t" in header:
        preferred_seps = ["\t", ",", None]
    else:
        preferred_seps = [",", "\t", None]

    last_err: Optional[Exception] = None
    for sep in preferred_seps:
        try:
            if sep is None:
                df = pd.read_csv(path, sep=r"\s+", engine="python")
            else:
                df = pd.read_csv(path, sep=sep)
            if (sep in (",", "\t")) and df.shape[1] == 1 and ("\t" in header or "," in header):
                continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Unable to read table: {path}. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_inputs(
    cohort_csv: Path,
    behav_csv: Path,
    subject_col: str,
    behav_subject_col: str,
    behav_cols_csv: str,
    max_subjects: int,
    group_col: Optional[str],
    behav_group_col: str,
    behav_group_keep_csv: str,
    imaging_long_label_col: str,
    imaging_long_value_col: str,
    imaging_feature_regex: str,
    age_col: str = "t1_ageyrs",
    sex_col: str = "t1_sex",
    iq_col: str = "t1_fsiq",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cohort = pd.read_csv(cohort_csv)
    cohort[subject_col] = cohort[subject_col].astype(str)

    # Keep only T1 data when a timepoint column is present
    if "timepoint" in cohort.columns:
        cohort = cohort[cohort["timepoint"] == "T1"].drop(columns=["timepoint"])
        print(f"Filtered to T1: {len(cohort)} rows retained.")

    # Support long-format cohort tables (subject/site/label/wdc → pivot to wide)
    did_pivot = False
    if (
        imaging_long_label_col in cohort.columns
        and imaging_long_value_col in cohort.columns
        and subject_col in cohort.columns
    ):
        idx_cols = [subject_col]
        if group_col and group_col in cohort.columns:
            idx_cols.append(group_col)
        cohort = (
            cohort.pivot_table(
                index=idx_cols,
                columns=imaging_long_label_col,
                values=imaging_long_value_col,
                aggfunc="mean",
            )
            .reset_index()
        )
        cohort.columns = [str(c) for c in cohort.columns]
        did_pivot = True

    behav = pd.read_csv(behav_csv)
    behav_subject_col_eff = behav_subject_col or subject_col
    behav[behav_subject_col_eff] = behav[behav_subject_col_eff].astype(str)

    # Optional: restrict to selected behaviour-group levels
    behav_group_col = str(behav_group_col or "").strip()
    behav_group_keep_csv = str(behav_group_keep_csv or "").strip()
    if behav_group_col and behav_group_keep_csv and behav_group_col in behav.columns:
        keep_levels = [v.strip() for v in behav_group_keep_csv.split(",") if v.strip()]
        if keep_levels:
            keep_set = {str(v) for v in keep_levels}
            behav_group_series = behav[behav_group_col].astype(str)
            behav = behav.loc[behav_group_series.isin(keep_set)].copy()

    exclude = {subject_col}
    if group_col and group_col in cohort.columns:
        exclude.add(group_col)

    if did_pivot:
        imaging_cols = [c for c in cohort.columns if c not in exclude]
    else:
        imaging_cols: list[str] = []
        imaging_feature_regex = str(imaging_feature_regex or "").strip()
        if imaging_feature_regex:
            imaging_cols = cohort.filter(regex=imaging_feature_regex).columns.tolist()
        if not imaging_cols:
            candidates = [c for c in cohort.columns if c not in exclude]
            if candidates:
                coerced = cohort[candidates].apply(pd.to_numeric, errors="coerce")
                imaging_cols = [c for c in candidates if bool(coerced[c].notna().any())]

    if not imaging_cols:
        raise ValueError(
            "No imaging feature columns found in cohort CSV. "
            "If using a wide-format file, pass --imaging-feature-regex, "
            "or use a long-format file with label/value columns."
        )

    behav_cols = [c.strip() for c in behav_cols_csv.split(",") if c.strip()]
    if not behav_cols:
        raise ValueError("No behavior columns specified.")
    missing_b = [c for c in behav_cols if c not in behav.columns]
    if missing_b:
        raise ValueError(f"Missing behavior columns: {', '.join(missing_b)}")

    keep_cols = [subject_col, *imaging_cols]
    if group_col and group_col in cohort.columns:
        keep_cols.append(group_col)
    cohort_small = cohort[keep_cols]

    extra_behav_cols: list[str] = []
    if behav_group_col and behav_group_col in behav.columns and behav_group_col not in behav_cols:
        extra_behav_cols.append(behav_group_col)
    # Carry age, sex, and IQ columns through the merge for ComBat covariates
    for _demo_col in (age_col, sex_col, iq_col):
        if _demo_col and _demo_col in behav.columns and _demo_col not in behav_cols and _demo_col not in extra_behav_cols:
            extra_behav_cols.append(_demo_col)

    behav_merge_cols = [behav_subject_col_eff, *behav_cols, *extra_behav_cols]
    behav_merge_cols = list(dict.fromkeys(behav_merge_cols))

    merged = cohort_small.merge(
        behav[behav_merge_cols],
        left_on=subject_col,
        right_on=behav_subject_col_eff,
        how="inner",
    )
    if behav_subject_col_eff != subject_col:
        merged = merged.drop(columns=[behav_subject_col_eff])

    complete_cols = imaging_cols + behav_cols
    merged_cc = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=complete_cols)

    if max_subjects and max_subjects > 0:
        merged_cc = merged_cc.sort_values(subject_col).head(max_subjects)

    merged_cc = merged_cc.reset_index(drop=True)
    return merged_cc, imaging_cols, behav_cols


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------


def to_matrix(df: pd.DataFrame, cols: list[str], do_scale: bool) -> np.ndarray:
    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if do_scale:
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=0)
        std = np.where(std == 0, 1.0, std)
        mat = (mat - mean) / std
    mat[~np.isfinite(mat)] = 0.0
    return mat


def fit_scaler(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z


# ---------------------------------------------------------------------------
# Imaging preprocessing (full-sample)
# ---------------------------------------------------------------------------


def build_nuisance_matrix(
    df: pd.DataFrame,
    fd_col: Optional[str],
    age_col: Optional[str],
    sex_col: Optional[str],
    iq_col: Optional[str],
) -> tuple[np.ndarray, list[str]]:
    """Build the nuisance design matrix (mFD, age, sex, IQ), without intercept."""
    cols: dict[str, np.ndarray] = {}
    for name, col in (("mFD", fd_col), ("AGE", age_col), ("IQ", iq_col)):
        if col and col in df.columns:
            cols[name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if sex_col and sex_col in df.columns:
        sex = df[sex_col].astype("string")
        dummies = pd.get_dummies(sex, prefix="SEX", drop_first=True, dtype=float)
        dummies[sex.isna().to_numpy()] = np.nan
        for c in dummies.columns:
            cols[c] = dummies[c].to_numpy(dtype=float)
    if not cols:
        return np.empty((len(df), 0)), []
    return np.column_stack(list(cols.values())), list(cols.keys())


def residualize_features_on_covariates_full(x: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Regress each imaging feature on nuisance covariates (OLS) and keep residuals.

    Fitted on subjects with complete covariates; subjects with missing
    covariates are left unadjusted.
    """
    x = np.asarray(x, dtype=float).copy()
    c = np.asarray(covariates, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    if c.shape[1] == 0:
        return x

    rows = np.all(np.isfinite(c), axis=1) & np.all(np.isfinite(x), axis=1)
    n_cov = c.shape[1]
    if rows.sum() <= n_cov + 1:
        return x

    # Drop constant covariates, which are collinear with the intercept
    c = c[:, np.std(c[rows], axis=0) > 0]
    if c.shape[1] == 0:
        return x

    design = np.column_stack([np.ones(int(rows.sum())), c[rows]])
    beta, *_ = np.linalg.lstsq(design, x[rows], rcond=None)
    x[rows] = x[rows] - design @ beta

    x[~np.isfinite(x)] = 0.0
    return x


def combat_harmonize_full(
    x: np.ndarray,
    site_labels: np.ndarray,
    df: pd.DataFrame,
    age_col: str = "t1_ageyrs",
    sex_col: str = "t1_sex",
    group_col: str = "t1_group",
    iq_col: str = "t1_fsiq",
    fd_col: Optional[str] = None,
) -> np.ndarray:
    """Apply ComBat harmonization on the full dataset.

    Biological covariates (age, sex, group, IQ) and mean FD are included so
    that ComBat preserves variance associated with them (mFD is regressed out
    afterwards in the residualisation step).
    """
    try:
        from neuroHarmonize import harmonizationLearn
    except ImportError as e:
        raise RuntimeError("Missing dependency 'neuroHarmonize'.") from e

    x_arr = np.asarray(x, dtype=float).copy()
    cov = pd.DataFrame({"SITE": pd.Series(site_labels, dtype=str)})
    if age_col and age_col in df.columns:
        cov["AGE"] = pd.to_numeric(df[age_col], errors="coerce").to_numpy(dtype=float)
    if sex_col and sex_col in df.columns:
        sex_codes = pd.Categorical(df[sex_col].astype(str)).codes.astype(float)
        sex_codes[sex_codes < 0] = np.nan
        cov["SEX"] = sex_codes
    if group_col and group_col in df.columns:
        grp_codes = pd.Categorical(df[group_col].astype(str)).codes.astype(float)
        grp_codes[grp_codes < 0] = np.nan
        if len(np.unique(grp_codes[np.isfinite(grp_codes)])) > 1:
            cov["GROUP"] = grp_codes
    if iq_col and iq_col in df.columns:
        cov["IQ"] = pd.to_numeric(df[iq_col], errors="coerce").to_numpy(dtype=float)
    if fd_col and fd_col in df.columns:
        fd = pd.to_numeric(df[fd_col], errors="coerce").to_numpy(dtype=float).copy()
        # Median-impute missing mFD (covariate only) so ComBat does not fail on NaN
        fd[~np.isfinite(fd)] = np.nanmedian(fd) if np.isfinite(fd).any() else 0.0
        cov["MFD"] = fd

    # Drop zero-variance features for ComBat
    v = np.var(x_arr, axis=0)
    keep = np.isfinite(v) & (v > 0)
    if not np.any(keep):
        x_arr[~np.isfinite(x_arr)] = 0.0
        return x_arr

    x_sub = x_arr[:, keep]
    learn_result = harmonizationLearn(x_sub, cov)
    if isinstance(learn_result, (tuple, list)) and len(learn_result) == 2:
        _, x_adj = learn_result
    else:
        x_adj = x_sub  # fallback

    x_arr[:, keep] = np.asarray(x_adj, dtype=float)
    x_arr[~np.isfinite(x_arr)] = 0.0
    return x_arr


def drop_rows_missing_covariates(
    df: pd.DataFrame,
    cols: list[Optional[str]],
    subject_col: str,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Drop subjects with missing values in any covariate used for ComBat/residualisation.

    ComBat (neuroHarmonize) and the OLS residualisation cannot handle missing
    covariates, and partially adjusted subjects would be inconsistent with the rest.
    """
    present = [c for c in dict.fromkeys(cols) if c and c in df.columns]
    if not present:
        return df
    bad = df[present].isna().any(axis=1).to_numpy()
    if not bad.any():
        return df
    print(f"Dropping {int(bad.sum())}/{len(df)} subjects with missing covariates ({', '.join(present)}).")
    if outdir is not None:
        df.loc[bad, [subject_col, *present]].to_csv(outdir / "subjects_dropped_missing_covariates.csv", index=False)
    return df.loc[~bad].reset_index(drop=True)


def preprocess_imaging_full(
    df: pd.DataFrame,
    imaging_cols: list[str],
    site_col: str,
    fd_col: Optional[str],
    use_combat: bool,
    age_col: str = "t1_ageyrs",
    sex_col: str = "t1_sex",
    group_col: str = "t1_group",
    iq_col: str = "t1_fsiq",
    residualize: bool = True,
) -> np.ndarray:
    """Full-sample imaging preprocessing: ComBat → residualize on mFD, age, sex, IQ.

    ComBat is applied first (age/sex/group/IQ preserved as biological
    covariates); the harmonised features are then residualised on the
    nuisance covariates. Residualisation runs only if `residualize` is True.

    No z-scoring (pyls handles normalization internally via cross-correlation).
    No PCA (wDC values are used directly).
    """
    x = df[imaging_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ComBat harmonisation (first)
    if use_combat and site_col in df.columns:
        site = df[site_col].astype(str).to_numpy()
        x = combat_harmonize_full(x, site, df, age_col=age_col, sex_col=sex_col, group_col=group_col, iq_col=iq_col, fd_col=fd_col if residualize else None)

    # Residualisation on mFD + age + sex + IQ (after ComBat)
    if residualize:
        nuisance, names = build_nuisance_matrix(df, fd_col, age_col, sex_col, iq_col)
        print(f"Residualising on: {', '.join(names) if names else '(none)'}")
        x = residualize_features_on_covariates_full(x, nuisance)

    return x


# ---------------------------------------------------------------------------
# Behavior diagnostics
# ---------------------------------------------------------------------------


def behav_diagnostics(
    df: pd.DataFrame,
    behav_cols: list[str],
    outlier_z: float,
    red_thresh: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    braw = df[behav_cols].apply(pd.to_numeric, errors="coerce")
    variances = braw.var(axis=0, ddof=1)

    out_rows = []
    for col in behav_cols:
        x = braw[col].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - mu) / sd
        idx = np.where(np.isfinite(z) & (np.abs(z) > outlier_z))[0]
        for i in idx.tolist():
            out_rows.append({"variable": col, "row_index": int(i + 1)})
    outliers = pd.DataFrame(out_rows)

    cor = braw.corr(method="pearson", min_periods=1)
    red_pairs = []
    if cor.shape[0] >= 2:
        cols = cor.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = cor.iloc[i, j]
                if pd.notna(r) and abs(r) > red_thresh:
                    red_pairs.append({"var1": cols[i], "var2": cols[j], "r": float(r)})
    redundancy = pd.DataFrame(red_pairs)
    return variances, outliers, cor, redundancy


# ---------------------------------------------------------------------------
# Missing code handling
# ---------------------------------------------------------------------------


def _parse_missing_codes(codes_csv: str) -> tuple[set[str], set[float]]:
    codes = [c.strip() for c in codes_csv.split(",") if c.strip()]
    codes_str = set(codes)
    codes_num: set[float] = set()
    for c in codes:
        try:
            codes_num.add(float(c))
        except Exception:
            pass
    return codes_str, codes_num


def find_special_missing_rows(
    df: pd.DataFrame,
    subject_col: str,
    behav_cols: list[str],
    codes_csv: str,
) -> pd.DataFrame:
    codes_str, codes_num = _parse_missing_codes(codes_csv)
    rows = []
    for i in range(len(df)):
        for col in behav_cols:
            val = df.iloc[i][col]
            val_str = "" if pd.isna(val) else str(val).strip()
            val_num = None
            try:
                val_num = float(val_str)
            except Exception:
                val_num = None
            is_special = (val_str in codes_str) or (val_num is not None and val_num in codes_num)
            if is_special:
                rows.append(
                    {subject_col: str(df.iloc[i][subject_col]), "variable": col, "value": val_str}
                )
    return pd.DataFrame(rows)


def drop_special_missing_rows(
    df: pd.DataFrame,
    behav_cols: list[str],
    codes_csv: str,
) -> pd.DataFrame:
    codes_str, codes_num = _parse_missing_codes(codes_csv)

    def row_has_special(row: pd.Series) -> bool:
        for col in behav_cols:
            v = row[col]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s in codes_str:
                return True
            try:
                if float(s) in codes_num:
                    return True
            except Exception:
                pass
        return False

    mask_keep = ~df.apply(row_has_special, axis=1)
    return df.loc[mask_keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# QC / FD attachment
# ---------------------------------------------------------------------------


def attach_fd_from_qc(
    df: pd.DataFrame,
    qc_path: Path,
    subject_col: str,
    qc_subject_col: str,
    fd_col: str,
    out_fd_col: str,
) -> pd.DataFrame:
    qc = _try_read_table(qc_path)
    if qc_subject_col not in qc.columns:
        raise ValueError(f"QC subject column not found: {qc_subject_col}")
    if fd_col not in qc.columns:
        raise ValueError(f"QC FD column not found: {fd_col}")

    qc = qc[[qc_subject_col, fd_col]].copy()
    qc[qc_subject_col] = qc[qc_subject_col].astype(str)
    qc[fd_col] = pd.to_numeric(qc[fd_col], errors="coerce")

    merged = df.merge(qc, left_on=subject_col, right_on=qc_subject_col, how="inner")
    if qc_subject_col != subject_col:
        merged = merged.drop(columns=[qc_subject_col])
    merged = merged.rename(columns={fd_col: out_fd_col})
    return merged


# ---------------------------------------------------------------------------
# Fold building
# ---------------------------------------------------------------------------


def build_folds(
    df: pd.DataFrame,
    kfolds: int,
    seed: int,
    group_col: Optional[str],
    strategy: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(df)

    if not group_col or group_col not in df.columns:
        k = min(kfolds, n)
        folds = np.tile(np.arange(1, k + 1), int(math.ceil(n / k)))[:n]
        rng.shuffle(folds)
        return folds

    if strategy == "random":
        k = min(kfolds, n)
        folds = np.zeros(n, dtype=int)
        groups = df[group_col].astype(str).to_numpy()
        for g in pd.unique(groups):
            idx = np.where(groups == g)[0]
            rng.shuffle(idx)
            for j, ii in enumerate(idx):
                folds[ii] = (j % k) + 1
        return folds

    groups = df[group_col].astype(str).to_numpy()
    uniq = pd.unique(groups)

    if strategy == "leave-group-out":
        group_to_fold = {g: i + 1 for i, g in enumerate(uniq)}
        return np.array([group_to_fold[g] for g in groups], dtype=int)

    # group-kfold
    k = min(kfolds, len(uniq))
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    group_to_fold = {g: (i % k) + 1 for i, g in enumerate(shuffled)}
    return np.array([group_to_fold[g] for g in groups], dtype=int)


def splits_from_folds(folds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    kfolds = int(np.max(folds))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, kfolds + 1):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        splits.append((train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------


def _corrcoef_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 3:
        return float("nan")
    if float(np.nanstd(a[mask])) == 0.0 or float(np.nanstd(b[mask])) == 0.0:
        return float("nan")
    r = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return r if np.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------
# Cross-validation of PLS modes using full-sample saliences
# ---------------------------------------------------------------------------


def cv_assess_modes(
    X: np.ndarray,
    Y: np.ndarray,
    x_weights: np.ndarray,
    y_weights: np.ndarray,
    significant_modes: list[int],
    df: pd.DataFrame,
    group_col: Optional[str],
    kfolds: int,
    seed: int,
    cv_strategy: str,
) -> dict:
    """Cross-validation to assess stability/generalizability of PLS modes.

    For each fold:
      1. Hold out test participants
      2. Project held-out data through full-sample saliences:
           u_test = X_test @ brain_saliences[:, mode]
           v_test = Y_test @ behavior_saliences[:, mode]
      3. Compute correlation(u_test, v_test)

    Args:
        X: Preprocessed imaging data (n_subjects x n_features)
        Y: Behavior data (n_subjects x n_behav) — same as passed to pyls
        x_weights: Brain saliences from full-sample PLS (n_features x n_modes)
        y_weights: Behavior saliences from full-sample PLS (n_behav x n_modes)
        significant_modes: 0-indexed list of significant mode indices to assess
        df: DataFrame (for fold building)
        group_col: Column for stratified fold building
        kfolds: Number of CV folds
        seed: Random seed
        cv_strategy: Fold strategy ('random', 'group-kfold', 'leave-group-out')

    Returns:
        dict with keys: r, abs_r, fisher_z, n_test, folds, significant_modes
    """
    folds = build_folds(df, kfolds, seed, group_col, cv_strategy)
    splits = splits_from_folds(folds)

    n_modes = len(significant_modes)
    n_splits = len(splits)
    r = np.full((n_splits, n_modes), np.nan, dtype=float)
    abs_r = np.full((n_splits, n_modes), np.nan, dtype=float)
    fisher_z = np.full((n_splits, n_modes), np.nan, dtype=float)
    n_test = np.full(n_splits, np.nan, dtype=float)

    for i, (train_idx, test_idx) in enumerate(splits):
        n_test[i] = len(test_idx)
        for j, mode in enumerate(significant_modes):
            u_test = X[test_idx] @ x_weights[:, mode]
            v_test = Y[test_idx] @ y_weights[:, mode]
            rr = _corrcoef_1d(u_test, v_test)
            r[i, j] = rr
            abs_r[i, j] = abs(rr) if np.isfinite(rr) else np.nan
            if np.isfinite(rr) and abs(rr) < 1.0:
                fisher_z[i, j] = float(np.arctanh(rr))

    return {
        "r": r,
        "abs_r": abs_r,
        "fisher_z": fisher_z,
        "n_test": n_test,
        "folds": folds,
        "significant_modes": significant_modes,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Behavioral PLS (pyls) on rs-fMRI wDC features (T1) vs behavioral measures. "
            f"Valid model names: {', '.join(MODEL_CONFIGS)}."
        )
    )

    p.add_argument(
        "--cohort-csv", type=Path, default=default_master_wdc_csv(),
        help="Path to imaging feature CSV (wide or long format).",
    )
    p.add_argument(
        "--behav-csv", type=Path, default=default_behaviour_csv(),
        help="Path to behavioral CSV containing the questionnaire columns.",
    )
    p.add_argument(
        "--models", type=str, default="all",
        help=(
            f"Comma-separated model names to run, or 'all'. "
            f"Valid: {', '.join(MODEL_CONFIGS)}."
        ),
    )
    p.add_argument("--subject-col", type=str, default="subject")
    p.add_argument("--behav-subject-col", type=str, default="subjects")
    p.add_argument(
        "--behav-group-col", type=str, default="t1_group",
        help="Behaviour table column used to restrict subjects by group.",
    )
    p.add_argument(
        "--behav-group-keep", type=str, default="1,2",
        help="Comma-separated allowed values for --behav-group-col.",
    )

    p.add_argument("--qc-csv", type=Path, default=default_qc_csv(timepoint=1))
    p.add_argument("--qc-subject-col", type=str, default="subject")
    p.add_argument("--qc-fd-col", type=str, default="meanFD")

    p.add_argument("--max-subjects", type=int, default=0)
    p.add_argument(
        "--outdir", type=Path, default=default_pls_output_dir(),
        help="Base output directory. Per-model outputs go to {outdir}/{model_name}/.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Allow writing into a non-empty outdir.",
    )

    p.add_argument("--nperms", type=int, default=5000,
                    help="Number of permutations for significance testing (pyls).")
    p.add_argument("--nboots", type=int, default=5000,
                    help="Number of bootstrap resamples for reliability testing (pyls).")
    p.add_argument("--kfolds", type=int, default=5,
                    help="Number of CV folds for stability assessment.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--use-combat", type=lambda x: str(x).lower() == "true", default=True,
        help="If True, run ComBat harmonisation on full sample before PLS.",
    )
    p.add_argument(
        "--use-fd-residualize", type=lambda x: str(x).lower() == "true", default=True,
        help="If True, after ComBat regress imaging features on meanFD (if QC available), age, sex and IQ; "
             "behavioural variables are regressed on age, sex and IQ.",
    )

    p.add_argument("--behav-outlier-z", type=float, default=3.0)
    p.add_argument("--behav-redundancy-thresh", type=float, default=0.8)
    p.add_argument("--drop-nzv", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--drop-redundant", type=lambda x: str(x).lower() == "true", default=False)

    p.add_argument("--group-col", type=str, default="site")
    p.add_argument("--age-col", type=str, default="t1_ageyrs",
                    help="Column name for age (used as ComBat and residualisation covariate).")
    p.add_argument("--sex-col", type=str, default="t1_sex",
                    help="Column name for sex (used as ComBat and residualisation covariate).")
    p.add_argument("--iq-col", type=str, default="t1_fsiq",
                    help="Column name for IQ (used as ComBat and residualisation covariate).")

    p.add_argument("--imaging-feature-regex", type=str, default="")
    p.add_argument("--imaging-long-label-col", type=str, default="label")
    p.add_argument("--imaging-long-value-col", type=str, default="wdc")

    p.add_argument(
        "--cv-strategy", type=str, default="random",
        choices=["random", "group-kfold", "leave-group-out"],
    )

    p.add_argument(
        "--perm-alpha", type=float, default=0.05,
        help="Significance threshold for identifying significant PLS modes from permutation p-values.",
    )
    p.add_argument(
        "--n-proc", type=str, default=None,
        help="Number of parallel workers for pyls (permutation/bootstrap). "
             "Use 'max' for all cores, or an integer. Default: serial.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_model(
    opts: argparse.Namespace,
    model_name: str,
    config: dict,
    n_proc: Optional[int],
) -> None:
    """Run the full PLS pipeline for one behavioural model (--models entry)."""
    label = config["label"]
    behav_vars: list[str] = config["behav_vars"]
    behav_cols_csv = ",".join(behav_vars)
    missing_codes = config.get("missing_codes", "777,999")
    sign_flip_vars: list[str] = config.get("sign_flip_vars", [])

    outdir = opts.outdir / model_name
    print(f"\n{'=' * 64}")
    print(f"Model: {model_name}  —  {label}")
    print(f"Behaviour variables: {behav_vars}")
    print(f"Output: {outdir}")
    print(f"{'=' * 64}")

    ensure_dir(outdir)
    if _dir_has_files(outdir) and not bool(getattr(opts, "overwrite", False)):
        print(f"SKIP: {outdir} is not empty. Pass --overwrite to re-run.")
        return
    if bool(getattr(opts, "overwrite", False)):
        _cleanup_previous_outputs_if_manifest_present(outdir)

    # ---------------------------------------------------------------
    # 1. Load & merge data
    # ---------------------------------------------------------------

    df, imaging_cols, behav_cols = read_inputs(
        opts.cohort_csv,
        opts.behav_csv,
        opts.subject_col,
        opts.behav_subject_col,
        behav_cols_csv,
        opts.max_subjects,
        opts.group_col,
        opts.behav_group_col,
        opts.behav_group_keep,
        opts.imaging_long_label_col,
        opts.imaging_long_value_col,
        opts.imaging_feature_regex,
        age_col=opts.age_col,
        sex_col=opts.sex_col,
        iq_col=opts.iq_col,
    )

    # Special missing codes
    miss_rows = find_special_missing_rows(df, opts.subject_col, behav_cols, missing_codes)
    miss_rows.to_csv(outdir / "behavior_special_missing_rows.csv", index=False)
    n_before = len(df)
    df = drop_special_missing_rows(df, behav_cols, missing_codes)
    n_after = len(df)

    # Attach FD from QC
    fd_col_internal = "__meanFD"
    if fd_col_internal in df.columns:
        raise ValueError(f"Column '{fd_col_internal}' already exists in the dataset.")
    if bool(opts.use_fd_residualize) and opts.qc_csv and opts.qc_csv.exists():
        df = attach_fd_from_qc(
            df,
            qc_path=opts.qc_csv,
            subject_col=opts.subject_col,
            qc_subject_col=opts.qc_subject_col,
            fd_col=opts.qc_fd_col,
            out_fd_col=fd_col_internal,
        )
        df = df.reset_index(drop=True)

    # ComBat and residualisation need complete covariates
    if bool(opts.use_combat) or bool(opts.use_fd_residualize):
        df = drop_rows_missing_covariates(
            df,
            [
                opts.age_col, opts.sex_col, opts.iq_col,
                opts.behav_group_col if bool(opts.use_combat) else None,
                fd_col_internal if bool(opts.use_fd_residualize) else None,
            ],
            opts.subject_col,
            outdir,
        )

    # Sign flips for conceptual alignment
    sign_flipped: list[str] = []
    for col in sign_flip_vars:
        if col in behav_cols and col in df.columns:
            df[col] = -1.0 * pd.to_numeric(df[col], errors="coerce")
            sign_flipped.append(col)

    # Behavior diagnostics
    variances, outliers, cor, redundancy = behav_diagnostics(
        df, behav_cols, outlier_z=opts.behav_outlier_z, red_thresh=opts.behav_redundancy_thresh,
    )
    pd.DataFrame({"variable": variances.index, "variance": variances.values}).to_csv(
        outdir / "behavior_variances.csv", index=False
    )
    outliers.to_csv(outdir / "behavior_outliers.csv", index=False)
    cor.reset_index().rename(columns={"index": "variable"}).to_csv(
        outdir / "behavior_cor_matrix.csv", index=False
    )
    redundancy.to_csv(outdir / "behavior_redundancy_pairs.csv", index=False)

    # Optional NZV / redundancy dropping
    dropped_log = []
    if opts.drop_nzv:
        nzv = variances[variances < 1e-8].index.tolist()
        dropped = [c for c in behav_cols if c in nzv]
        for c in dropped:
            dropped_log.append({"variable": c, "reason": "near-zero variance"})
        behav_cols = [c for c in behav_cols if c not in dropped]

    if opts.drop_redundant and not redundancy.empty:
        for _, row in redundancy.iterrows():
            v2 = row["var2"]
            if v2 in behav_cols:
                behav_cols.remove(v2)
                dropped_log.append(
                    {"variable": v2, "reason": f"redundant with {row['var1']} (r={float(row['r']):.3f})"}
                )

    pd.DataFrame({"behavior_feature": behav_cols}).to_csv(outdir / "behavior_used.csv", index=False)
    if dropped_log:
        pd.DataFrame(dropped_log).to_csv(outdir / "dropped_columns_log.csv", index=False)

    # Save combined input
    df.to_csv(outdir / "combined_input.csv", index=False)

    # ---------------------------------------------------------------
    # 2. Preprocess imaging (full sample): ComBat → residualize (mFD, age, sex, IQ)
    # ---------------------------------------------------------------

    print(f"Preprocessing imaging: {len(imaging_cols)} wDC features, {len(df)} subjects...")
    X = preprocess_imaging_full(
        df=df,
        imaging_cols=imaging_cols,
        site_col=opts.group_col,
        fd_col=fd_col_internal if fd_col_internal in df.columns else None,
        use_combat=bool(opts.use_combat),
        age_col=opts.age_col,
        sex_col=opts.sex_col,
        group_col=opts.behav_group_col,
        iq_col=opts.iq_col,
        residualize=bool(opts.use_fd_residualize),
    )

    # Behaviour matrix: residualise on age, sex, IQ (same as imaging, without mFD).
    # pyls handles z-scoring internally.
    Y = df[behav_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(opts.use_fd_residualize):
        nuisance_y, names_y = build_nuisance_matrix(df, None, opts.age_col, opts.sex_col, opts.iq_col)
        print(f"Residualising behaviour on: {', '.join(names_y) if names_y else '(none)'}")
        Y = residualize_features_on_covariates_full(Y, nuisance_y)

    print(f"X shape: {X.shape} (subjects x wDC features)")
    print(f"Y shape: {Y.shape} (subjects x behavior features)")

    # ---------------------------------------------------------------
    # 3. Site summary (no multi-group PLS — data is ComBat-harmonized)
    # ---------------------------------------------------------------

    if opts.group_col and opts.group_col in df.columns:
        site_counts = df[opts.group_col].astype(str).value_counts().sort_index()
        print(f"Site distribution: {site_counts.to_dict()}")

    # ---------------------------------------------------------------
    # 4. Run behavioral PLS on full sample via pyls (pooled, no groups)
    # ---------------------------------------------------------------

    n_modes_max = min(X.shape[1], Y.shape[1])
    print(
        f"Running behavioral PLS (pyls): max {n_modes_max} modes, "
        f"n_perm={opts.nperms}, n_boot={opts.nboots}..."
    )

    pls_results = pyls_pkg.behavioral_pls(
        X, Y,
        n_perm=int(opts.nperms),
        n_boot=int(opts.nboots),
        n_split=0,
        rotate=False,  # Procrustes rotation disabled (pyls bug: dimension mismatch in compute.procrustes)
        ci=95,
        seed=int(opts.seed),
        verbose=True,
        n_proc=n_proc,
    )

    # ---------------------------------------------------------------
    # 5. Extract results
    # ---------------------------------------------------------------

    x_weights = pls_results["x_weights"]    # (n_features, n_modes) — brain saliences
    y_weights = pls_results["y_weights"]    # (n_behav, n_modes) — behavior saliences
    x_scores = pls_results["x_scores"]      # (n_subjects, n_modes) — brain scores
    y_scores = pls_results["y_scores"]      # (n_subjects, n_modes) — behavior scores
    singvals = pls_results["singvals"]       # (n_modes,) — singular values
    varexp = pls_results["varexp"]           # (n_modes,) — variance explained per mode

    n_modes = len(singvals)
    print(f"PLS extracted {n_modes} modes (max possible = {n_modes_max})")

    # Permutation p-values
    perm_pvals = pls_results["permres"]["pvals"]
    perm_pvals_fdr = benjamini_hochberg_fdr(np.asarray(perm_pvals, dtype=float))

    # Identify significant modes
    alpha = float(opts.perm_alpha)
    significant_modes = [m for m in range(n_modes) if perm_pvals[m] < alpha]
    print(f"Permutation p-values: {perm_pvals}")
    print(f"Permutation p-values (FDR): {perm_pvals_fdr}")
    print(f"Significant modes (p < {alpha}): {[m + 1 for m in significant_modes]}")

    # Compute per-mode correlations (full sample)
    full_corrs = np.array(
        [_corrcoef_1d(x_scores[:, m], y_scores[:, m]) for m in range(n_modes)],
        dtype=float,
    )

    # ---------------------------------------------------------------
    # 6. Save PLS outputs
    # ---------------------------------------------------------------

    # Saliences / weights
    wX_df = pd.DataFrame({"feature": imaging_cols})
    wY_df = pd.DataFrame({"feature": behav_cols})
    for mi in range(n_modes):
        wX_df[f"salience_mode{mi + 1}"] = x_weights[:, mi]
        wY_df[f"salience_mode{mi + 1}"] = y_weights[:, mi]
    wX_df.to_csv(outdir / "saliences_X.csv", index=False)
    wY_df.to_csv(outdir / "saliences_Y.csv", index=False)

    # Scores
    scores_df = pd.DataFrame({opts.subject_col: df[opts.subject_col].astype(str).tolist()})
    for mi in range(n_modes):
        scores_df[f"u{mi + 1}"] = x_scores[:, mi]
        scores_df[f"v{mi + 1}"] = y_scores[:, mi]
    scores_df.to_csv(outdir / "scores.csv", index=False)

    # Full-sample mode summary
    mode_summary_rows = []
    for mi in range(n_modes):
        mode_summary_rows.append({
            "mode": mi + 1,
            "singular_value": float(singvals[mi]),
            "variance_explained": float(varexp[mi]),
            "correlation": float(full_corrs[mi]),
            "perm_pval": float(perm_pvals[mi]),
            "perm_pval_fdr": float(perm_pvals_fdr[mi]),
            "significant": perm_pvals[mi] < alpha,
        })
    mode_summary = pd.DataFrame(mode_summary_rows)
    mode_summary.to_csv(outdir / "mode_summary.csv", index=False)

    # Permutation null distribution
    if "permres" in pls_results and "perm_singval" in pls_results["permres"]:
        perm_null_sv = pls_results["permres"]["perm_singval"]
        perm_null_df = pd.DataFrame(perm_null_sv)
        perm_null_df.columns = [f"null_sv_mode{m + 1}" for m in range(perm_null_df.shape[1])]
        perm_null_df.insert(0, "iter", np.arange(1, len(perm_null_df) + 1))
        perm_null_df.to_csv(outdir / "permutation_null.csv", index=False)

    # Bootstrap results
    if "bootres" in pls_results and pls_results["bootres"] is not None:
        bootres = pls_results["bootres"]

        # Bootstrap ratios for brain saliences (x_weights_normed)
        if "x_weights_normed" in bootres:
            bsr_x = bootres["x_weights_normed"]  # (n_features, n_modes)
            bsr_X_df = pd.DataFrame({"feature": imaging_cols})
            for mi in range(bsr_x.shape[1]):
                bsr_X_df[f"bsr_mode{mi + 1}"] = bsr_x[:, mi]
            bsr_X_df.to_csv(outdir / "bootstrap_ratios_X.csv", index=False)

        if "x_weights_stderr" in bootres:
            se_x = bootres["x_weights_stderr"]
            se_X_df = pd.DataFrame({"feature": imaging_cols})
            for mi in range(se_x.shape[1]):
                se_X_df[f"se_mode{mi + 1}"] = se_x[:, mi]
            se_X_df.to_csv(outdir / "bootstrap_stderr_X.csv", index=False)

        # Bootstrap behavioral loadings
        if "y_loadings" in bootres:
            y_load = bootres["y_loadings"]  # (n_behav, n_modes)
            y_load_df = pd.DataFrame({"feature": behav_cols})
            for mi in range(y_load.shape[1]):
                y_load_df[f"loading_mode{mi + 1}"] = y_load[:, mi]
            y_load_df.to_csv(outdir / "bootstrap_y_loadings.csv", index=False)

        if "y_loadings_ci" in bootres:
            y_ci = bootres["y_loadings_ci"]  # (n_behav, n_modes, 2)
            y_ci_df = pd.DataFrame({"feature": behav_cols})
            for mi in range(y_ci.shape[1]):
                y_ci_df[f"ci_lo_mode{mi + 1}"] = y_ci[:, mi, 0]
                y_ci_df[f"ci_hi_mode{mi + 1}"] = y_ci[:, mi, 1]
            y_ci_df.to_csv(outdir / "bootstrap_y_loadings_ci.csv", index=False)

    # Y loadings from full-sample model
    if "y_loadings" in pls_results:
        y_loadings_full = pls_results["y_loadings"]
        y_load_full_df = pd.DataFrame({"feature": behav_cols})
        for mi in range(y_loadings_full.shape[1]):
            y_load_full_df[f"loading_mode{mi + 1}"] = y_loadings_full[:, mi]
        y_load_full_df.to_csv(outdir / "y_loadings.csv", index=False)

    # ---------------------------------------------------------------
    # 7. Cross-validation of significant modes
    # ---------------------------------------------------------------

    cv_result = None
    if significant_modes:
        print(
            f"Running {opts.kfolds}-fold CV to assess stability of "
            f"{len(significant_modes)} significant mode(s)..."
        )
        cv_result = cv_assess_modes(
            X=X,
            Y=Y,
            x_weights=x_weights,
            y_weights=y_weights,
            significant_modes=significant_modes,
            df=df,
            group_col=opts.group_col,
            kfolds=int(opts.kfolds),
            seed=int(opts.seed) + 777,
            cv_strategy=opts.cv_strategy,
        )

        # Per-fold detail
        cv_rows = []
        for i in range(cv_result["r"].shape[0]):
            for j, mode in enumerate(significant_modes):
                rr = cv_result["r"][i, j]
                cv_rows.append({
                    "fold": i + 1,
                    "mode": mode + 1,
                    "r": float(rr) if np.isfinite(rr) else np.nan,
                    "abs_r": float(cv_result["abs_r"][i, j]) if np.isfinite(cv_result["abs_r"][i, j]) else np.nan,
                    "fisher_z": float(cv_result["fisher_z"][i, j]) if np.isfinite(cv_result["fisher_z"][i, j]) else np.nan,
                    "n_test": int(cv_result["n_test"][i]),
                })
        cv_detail = pd.DataFrame(cv_rows)
        cv_detail.to_csv(outdir / "cv_fold_correlations.csv", index=False)

        # CV summary
        cv_summ_rows = []
        for j, mode in enumerate(significant_modes):
            r_vals = cv_result["r"][:, j]
            abs_r_vals = cv_result["abs_r"][:, j]
            fz_vals = cv_result["fisher_z"][:, j]
            cv_summ_rows.append({
                "mode": mode + 1,
                "mean_r": float(np.nanmean(r_vals)),
                "sd_r": float(np.nanstd(r_vals, ddof=1)),
                "mean_abs_r": float(np.nanmean(abs_r_vals)),
                "sd_abs_r": float(np.nanstd(abs_r_vals, ddof=1)),
                "mean_fisher_z": float(np.nanmean(fz_vals)),
                "sd_fisher_z": float(np.nanstd(fz_vals, ddof=1)),
                "perm_pval": float(perm_pvals[mode]),
                "perm_pval_fdr": float(perm_pvals_fdr[mode]),
            })
        cv_summary = pd.DataFrame(cv_summ_rows)
        cv_summary.to_csv(outdir / "cv_summary.csv", index=False)

        print("CV summary:")
        for _, row in cv_summary.iterrows():
            print(
                f"  Mode {int(row['mode'])}: mean r = {row['mean_r']:.4f} "
                f"(sd = {row['sd_r']:.4f}), mean |r| = {row['mean_abs_r']:.4f}"
            )
    else:
        print("No significant modes found — skipping CV assessment.")
        # Write empty CV files for consistency
        pd.DataFrame(columns=["fold", "mode", "r", "abs_r", "fisher_z", "n_test"]).to_csv(
            outdir / "cv_fold_correlations.csv", index=False
        )
        pd.DataFrame(columns=["mode", "mean_r", "sd_r", "mean_abs_r", "sd_abs_r",
                               "mean_fisher_z", "sd_fisher_z", "perm_pval", "perm_pval_fdr"]).to_csv(
            outdir / "cv_summary.csv", index=False
        )

    # ---------------------------------------------------------------
    # 8. Summary text
    # ---------------------------------------------------------------

    summary_lines: list[str] = []
    summary_lines.append("=" * 60)
    summary_lines.append("PLS Analysis Summary (pyls behavioral PLS)")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Subjects: {len(df)}")
    summary_lines.append(f"Imaging features (wDC): {len(imaging_cols)}")
    summary_lines.append(f"Behavior features: {len(behav_cols)}")
    summary_lines.append(f"PLS modes extracted: {n_modes} (max = min(p, q) = {n_modes_max})")
    summary_lines.append(f"ComBat harmonization (applied first): {bool(opts.use_combat)}")
    summary_lines.append(
        "Residualization (applied after ComBat): "
        f"imaging on mFD, age, sex, IQ; behaviour on age, sex, IQ: {bool(opts.use_fd_residualize)}"
    )
    summary_lines.append(f"Permutations: {opts.nperms}")
    summary_lines.append(f"Bootstraps: {opts.nboots}")
    summary_lines.append(f"Seed: {opts.seed}")
    summary_lines.append("")

    summary_lines.append("Full-sample mode results:")
    for mi in range(n_modes):
        sig_marker = " ***" if mi in significant_modes else ""
        summary_lines.append(
            f"  Mode {mi + 1}: sv = {singvals[mi]:.4f}, "
            f"varexp = {varexp[mi]:.4f}, "
            f"r = {full_corrs[mi]:.4f}, "
            f"perm_p = {perm_pvals[mi]:.6f}, "
            f"perm_p_fdr = {perm_pvals_fdr[mi]:.6f}"
            f"{sig_marker}"
        )

    summary_lines.append("")
    summary_lines.append(f"Significant modes (p < {alpha}): {[m + 1 for m in significant_modes]}")

    if cv_result is not None and significant_modes:
        summary_lines.append("")
        summary_lines.append(f"Cross-validation ({opts.kfolds}-fold, strategy={opts.cv_strategy}):")
        for j, mode in enumerate(significant_modes):
            r_vals = cv_result["r"][:, j]
            summary_lines.append(
                f"  Mode {mode + 1}: mean r = {np.nanmean(r_vals):.4f} "
                f"(sd = {np.nanstd(r_vals, ddof=1):.4f})"
            )

    summary_lines.append("")
    summary_lines.append(f"Dropped due to special missing codes: {n_before - n_after}")
    if sign_flipped:
        summary_lines.append("Applied sign flip to: " + ", ".join(sign_flipped))
    if opts.group_col and opts.group_col in df.columns:
        site_counts = df[opts.group_col].astype(str).value_counts().sort_index()
        summary_lines.append(f"Site distribution: {site_counts.to_dict()}")
    summary_lines.append("Groups mode: pooled (no multi-group PLS; ComBat-harmonized data)")

    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n" + "\n".join(summary_lines))

    # ---------------------------------------------------------------
    # 9. Manifest
    # ---------------------------------------------------------------

    manifest_entries = [
        ("behavior_used.csv", "Behavior features used in PLS"),
        ("behavior_variances.csv", "Variance of each behavioral feature"),
        ("behavior_outliers.csv", "Subjects flagged as outliers by z-score threshold"),
        ("behavior_cor_matrix.csv", "Correlation matrix of behavioral features"),
        ("behavior_redundancy_pairs.csv", "High-correlation behavioral pairs"),
        ("behavior_special_missing_rows.csv", "Rows with special missing codes (dropped)"),
        ("combined_input.csv", "Merged input data after preprocessing"),
        ("saliences_X.csv", "Brain saliences (wDC feature weights) per PLS mode"),
        ("saliences_Y.csv", "Behavior saliences (weights) per PLS mode"),
        ("scores.csv", "Subject scores on each PLS mode (u and v)"),
        ("mode_summary.csv", "Per-mode summary: singular values, variance explained, correlations, p-values"),
        ("cv_fold_correlations.csv", "Per-fold CV correlations for significant modes"),
        ("cv_summary.csv", "CV summary statistics for significant modes"),
        ("summary.txt", "Full analysis summary"),
        ("manifest.csv", "This file"),
    ]

    if (outdir / "permutation_null.csv").exists():
        manifest_entries.append(("permutation_null.csv", "Permutation null distribution of singular values"))
    if dropped_log:
        manifest_entries.insert(1, ("dropped_columns_log.csv", "Log of dropped behavioral columns"))
    if "y_loadings" in pls_results:
        manifest_entries.append(("y_loadings.csv", "Behavioral loadings from full-sample PLS"))
    if "bootres" in pls_results and pls_results["bootres"] is not None:
        bootres_manifest = pls_results["bootres"]
        if "x_weights_normed" in bootres_manifest:
            manifest_entries.append(("bootstrap_ratios_X.csv", "Bootstrap ratios for brain saliences"))
        if "x_weights_stderr" in bootres_manifest:
            manifest_entries.append(("bootstrap_stderr_X.csv", "Bootstrap standard errors for brain saliences"))
        if "y_loadings" in bootres_manifest:
            manifest_entries.append(("bootstrap_y_loadings.csv", "Bootstrap behavioral loadings"))
        if "y_loadings_ci" in bootres_manifest:
            manifest_entries.append(("bootstrap_y_loadings_ci.csv", "Bootstrap CIs for behavioral loadings"))

    # Save pyls results object for later use
    hdf5_saved = False
    try:
        pyls_pkg.save_results(str(outdir / "pls_results.hdf5"), pls_results)
        hdf5_saved = True
        print(f"pyls results object saved to {outdir / 'pls_results.hdf5'}")
    except Exception as e:
        print(f"Warning: Could not save pyls results object ({e})")

    if hdf5_saved:
        manifest_entries.append(("pls_results.hdf5", "Full pyls results object (HDF5)"))

    pd.DataFrame(manifest_entries, columns=["filename", "description"]).to_csv(
        outdir / "manifest.csv", index=False
    )

    print(f"\nDone. Outputs saved to {outdir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    opts = parse_args()

    if opts.models.strip().lower() == "all":
        selected_models = list(MODEL_CONFIGS.keys())
    else:
        selected_models = [m.strip() for m in opts.models.split(",") if m.strip()]
        invalid = [m for m in selected_models if m not in MODEL_CONFIGS]
        if invalid:
            raise ValueError(
                f"Unknown model(s): {invalid}. Valid: {list(MODEL_CONFIGS.keys())}"
            )

    print(f"Models to run: {selected_models}")

    n_proc: Optional[int] = None
    if opts.n_proc is not None:
        if opts.n_proc.strip().lower() == "max":
            n_proc = -1
        else:
            try:
                n_proc = int(opts.n_proc)
            except ValueError:
                n_proc = None

    ensure_dir(opts.outdir)

    for model_name in selected_models:
        try:
            _run_model(opts, model_name, MODEL_CONFIGS[model_name], n_proc)
        except Exception as e:
            print(f"ERROR running model '{model_name}': {e}")
            print("Skipping this model.")
            continue


if __name__ == "__main__":
    main()

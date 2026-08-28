#!/usr/bin/env python3

"""PLS linking rs-fMRI wDC developmental trajectory slopes to behavioural trajectory slopes.

Input: per-subject LMM random-slope BLUPs from:
  - run_growth_curves_with_random_slopes.py          → brain (network) slopes
  - run_growth_curves_with_random_slopes_behavioural.py → behavioural slopes

Because both LMMs include site, sex, age, and IQ as fixed effects, ComBat and
FD residualization are disabled by default.  Use --use-combat true to override.

Supported models (--models):
  autism  – SRS rawscore / RBS total / SSP total
  sdq     – SDQ Total Difficulties (parent)
  prl     – PRL perseverative errors / Win-Stay / Lose-Shift
  ashq    – ASHQ Total
  tas     – TAS Total
  vabs    – Vineland ABC Standard Score
  whoqol  – WHOQoL-Bref Overall QoL & General Health

All models share the same brain slopes; only the behavioural columns differ.
Results land in {outdir}/{model_name}/.

Date: 2026-08-03
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
        default_growth_curves_output_dir,
        rs_project_root,
    )
except ImportError:  # pragma: no cover
    from scripts.rs_paths import (  # type: ignore
        default_growth_curves_output_dir,
        rs_project_root,
    )

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    delayed = None  # type: ignore


def _default_brain_slopes_csv() -> Path:
    return default_growth_curves_output_dir() / "random_slopes.csv"


def _default_behav_slopes_csv() -> Path:
    return rs_project_root() / "reports" / "growth_curves_behavioural" / "random_slopes.csv"


def _default_pls_trajectories_output_dir() -> Path:
    return rs_project_root() / "reports" / "pls_trajectories"


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "autism": {
        "behav_vars": ["srs_rawscore", "rbs_total", "ssp_total"],
        "label": "Autism spectrum (SRS rawscore / RBS total / SSP total)",
    },
    "sdq": {
        "behav_vars": ["sdq_emotional_p", "sdq_conduct_p", "sdq_hyperactivity_p", "sdq_peer_p", "sdq_prosocial_p"],
        "label": "SDQ subscales – emotional / conduct / hyperactivity / peer / prosocial (parent)",
    },
    "prl": {
        "behav_vars": ["prl_perE_prop", "prl_WS", "prl_LS"],
        "label": "PRL (perseverative errors / Win-Stay / Lose-Shift)",
    },
    "ashq": {
        "behav_vars": ["ashq_total"],
        "label": "ASHQ Total",
    },
    "tas": {
        "behav_vars": ["tas_identify", "tas_describe", "tas_external"],
        "label": "TAS subscales – identify / describe / external thinking",
    },
    "vabs": {
        "behav_vars": ["vineland_abc"],
        "label": "Vineland ABC Standard Score",
    },
    "whoqol": {
        "behav_vars": ["whoqol_overall"],
        "label": "WHOQoL-Bref Overall QoL & General Health",
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
# Random-slope data loading
# ---------------------------------------------------------------------------


def load_random_slopes_wide(
    path: Path,
    model_tag: str,
    feature_col: str,
    subject_col: str,
    reliable_only: bool = True,
    feature_filter: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a random_slopes.csv and pivot to wide format (subjects × features).

    Reads `re_slope` (the per-subject BLUP) as the feature value.
    Filters to `model_tag` and optionally to slope_reliable == True rows.
    """
    df = _try_read_table(path)

    for required_col in ("model", subject_col, feature_col, "re_slope"):
        if required_col not in df.columns:
            raise ValueError(
                f"Column '{required_col}' not found in {path}. "
                f"Available: {df.columns.tolist()}"
            )

    df = df[df["model"] == model_tag].copy()
    if df.empty:
        available = _try_read_table(path)["model"].unique().tolist()
        raise ValueError(
            f"No rows for model_tag='{model_tag}' in {path}. "
            f"Available: {sorted(available)}"
        )

    if reliable_only and "slope_reliable" in df.columns:
        reliable_mask = df["slope_reliable"].astype(str).str.lower() == "true"
        n_feat_ok = df.loc[reliable_mask, feature_col].nunique()
        n_feat_all = df[feature_col].nunique()
        print(
            f"  slope_reliable filter [{path.name}, model={model_tag}]: "
            f"{n_feat_ok}/{n_feat_all} features pass"
        )
        df = df[reliable_mask].copy()
        if df.empty:
            raise ValueError(
                f"No reliable slopes for model_tag='{model_tag}' in {path}."
            )

    if feature_filter is not None:
        df = df[df[feature_col].isin(feature_filter)].copy()
        missing_feats = [f for f in feature_filter if f not in df[feature_col].unique()]
        if missing_feats:
            print(f"  Warning: features absent after reliability filter: {missing_feats}")
        if df.empty:
            raise ValueError(
                f"No rows remain after feature filter {feature_filter} "
                f"for model_tag='{model_tag}' in {path}."
            )

    wide = (
        df.pivot_table(
            index=subject_col,
            columns=feature_col,
            values="re_slope",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns = [str(c) for c in wide.columns]
    wide[subject_col] = wide[subject_col].astype(str)

    feature_cols = [c for c in wide.columns if c != subject_col]
    print(
        f"  → {len(wide)} subjects × {len(feature_cols)} features "
        f"from {path.name} (model={model_tag})"
    )
    return wide, feature_cols


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


def residualize_features_on_fd_full(x: np.ndarray, fd: np.ndarray) -> np.ndarray:
    """Regress each imaging feature on mean FD across the full dataset."""
    x = np.asarray(x, dtype=float).copy()
    fd = np.asarray(fd, dtype=float)
    fd = np.where(np.isfinite(fd), fd, np.nan)

    for j in range(x.shape[1]):
        y = x[:, j]
        mask = np.isfinite(fd) & np.isfinite(y)
        if np.sum(mask) < 3:
            continue
        xm = fd[mask]
        ym = y[mask]
        if np.nanstd(ym) < 1e-15:
            continue
        vx = np.var(xm)
        if not np.isfinite(vx) or vx == 0:
            continue
        b = float(np.cov(xm, ym, ddof=0)[0, 1] / vx)
        a = float(np.mean(ym) - b * np.mean(xm))
        pred = a + b * fd
        finite_pred = np.isfinite(pred)
        x[finite_pred, j] = y[finite_pred] - pred[finite_pred]

    x[~np.isfinite(x)] = 0.0
    return x


def combat_harmonize_full(
    x: np.ndarray,
    site_labels: np.ndarray,
    df: pd.DataFrame,
    age_col: str = "t1_ageyrs",
    sex_col: str = "t1_sex",
    group_col: str = "t1_group",
) -> np.ndarray:
    """Apply ComBat harmonization on the full dataset.

    Biological covariates (age, sex, group) are included so that
    ComBat preserves variance associated with them.
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


def preprocess_imaging_full(
    df: pd.DataFrame,
    imaging_cols: list[str],
    site_col: str,
    fd_col: Optional[str],
    use_combat: bool,
    age_col: str = "t1_ageyrs",
    sex_col: str = "t1_sex",
    group_col: str = "t1_group",
) -> np.ndarray:
    """Full-sample imaging preprocessing: FD residualize → ComBat.

    No z-scoring (pyls handles normalization internally via cross-correlation).
    No PCA (wDC values are used directly).
    """
    x = df[imaging_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # FD residualisation
    if fd_col and fd_col in df.columns:
        fd = pd.to_numeric(df[fd_col], errors="coerce").to_numpy(float)
        x = residualize_features_on_fd_full(x, fd)

    # ComBat harmonisation
    if use_combat and site_col in df.columns:
        site = df[site_col].astype(str).to_numpy()
        x = combat_harmonize_full(x, site, df, age_col=age_col, sex_col=sex_col, group_col=group_col)

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
# Core PLS execution (factored for multi-model loop)
# ---------------------------------------------------------------------------


def run_pls_for_model(
    df: pd.DataFrame,
    imaging_cols: list[str],
    behav_cols_in: list[str],
    opts: argparse.Namespace,
    outdir: Path,
    model_label: str,
    n_proc: Optional[int],
) -> None:
    """Run the full PLS pipeline for one brain × behaviour model."""
    ensure_dir(outdir)
    behav_cols = list(behav_cols_in)

    # FD attachment (disabled by default; LMM already handled motion)
    fd_col_internal = "__meanFD"
    has_fd = False
    if bool(getattr(opts, "use_fd_residualize", False)):
        qc_csv = getattr(opts, "qc_csv", None)
        if qc_csv and Path(str(qc_csv)).exists():
            df = attach_fd_from_qc(
                df,
                qc_path=Path(str(qc_csv)),
                subject_col=opts.subject_col,
                qc_subject_col=opts.qc_subject_col,
                fd_col=opts.qc_fd_col,
                out_fd_col=fd_col_internal,
            )
            df = df.reset_index(drop=True)
            has_fd = True

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

    df.to_csv(outdir / "combined_input.csv", index=False)

    print(f"  Preprocessing: {len(imaging_cols)} brain features, {len(df)} subjects...")
    X = preprocess_imaging_full(
        df=df,
        imaging_cols=imaging_cols,
        site_col=opts.group_col,
        fd_col=fd_col_internal if has_fd else None,
        use_combat=bool(opts.use_combat),
        age_col=opts.age_col,
        sex_col=opts.sex_col,
        group_col="",
    )

    Y = df[behav_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  X: {X.shape}  (subjects × brain trajectory slopes)")
    print(f"  Y: {Y.shape}  (subjects × behavioural trajectory slopes)")

    if opts.group_col and opts.group_col in df.columns:
        site_counts = df[opts.group_col].astype(str).value_counts().sort_index()
        print(f"  Site distribution: {site_counts.to_dict()}")

    n_modes_max = min(X.shape[1], Y.shape[1])
    print(
        f"  Running behavioral PLS: max {n_modes_max} modes, "
        f"n_perm={opts.nperms}, n_boot={opts.nboots}..."
    )

    pls_results = pyls_pkg.behavioral_pls(
        X, Y,
        n_perm=int(opts.nperms),
        n_boot=int(opts.nboots),
        n_split=0,
        rotate=False,
        ci=95,
        seed=int(opts.seed),
        verbose=True,
        n_proc=n_proc,
    )

    x_weights = pls_results["x_weights"]
    y_weights = pls_results["y_weights"]
    x_scores  = pls_results["x_scores"]
    y_scores  = pls_results["y_scores"]
    singvals  = pls_results["singvals"]
    varexp    = pls_results["varexp"]

    n_modes = len(singvals)
    print(f"  PLS extracted {n_modes} modes (max = {n_modes_max})")

    perm_pvals = pls_results["permres"]["pvals"]
    perm_pvals_fdr = benjamini_hochberg_fdr(np.asarray(perm_pvals, dtype=float))
    alpha = float(opts.perm_alpha)
    significant_modes = [m for m in range(n_modes) if perm_pvals[m] < alpha]

    print(f"  Permutation p-values: {perm_pvals}")
    print(f"  Permutation p-values (FDR): {perm_pvals_fdr}")
    print(f"  Significant modes (p < {alpha}): {[m + 1 for m in significant_modes]}")

    full_corrs = np.array(
        [_corrcoef_1d(x_scores[:, m], y_scores[:, m]) for m in range(n_modes)],
        dtype=float,
    )

    wX_df = pd.DataFrame({"feature": imaging_cols})
    wY_df = pd.DataFrame({"feature": behav_cols})
    for mi in range(n_modes):
        wX_df[f"salience_mode{mi + 1}"] = x_weights[:, mi]
        wY_df[f"salience_mode{mi + 1}"] = y_weights[:, mi]
    wX_df.to_csv(outdir / "saliences_X.csv", index=False)
    wY_df.to_csv(outdir / "saliences_Y.csv", index=False)

    scores_df = pd.DataFrame({opts.subject_col: df[opts.subject_col].astype(str).tolist()})
    for mi in range(n_modes):
        scores_df[f"u{mi + 1}"] = x_scores[:, mi]
        scores_df[f"v{mi + 1}"] = y_scores[:, mi]
    scores_df.to_csv(outdir / "scores.csv", index=False)

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
    pd.DataFrame(mode_summary_rows).to_csv(outdir / "mode_summary.csv", index=False)

    if "permres" in pls_results and "perm_singval" in pls_results["permres"]:
        perm_null_sv = pls_results["permres"]["perm_singval"]
        perm_null_df = pd.DataFrame(perm_null_sv)
        perm_null_df.columns = [f"null_sv_mode{m + 1}" for m in range(perm_null_df.shape[1])]
        perm_null_df.insert(0, "iter", np.arange(1, len(perm_null_df) + 1))
        perm_null_df.to_csv(outdir / "permutation_null.csv", index=False)

    if "bootres" in pls_results and pls_results["bootres"] is not None:
        bootres = pls_results["bootres"]
        if "x_weights_normed" in bootres:
            bsr_x = bootres["x_weights_normed"]
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
        if "y_loadings" in bootres:
            y_load = bootres["y_loadings"]
            y_load_df = pd.DataFrame({"feature": behav_cols})
            for mi in range(y_load.shape[1]):
                y_load_df[f"loading_mode{mi + 1}"] = y_load[:, mi]
            y_load_df.to_csv(outdir / "bootstrap_y_loadings.csv", index=False)
        if "y_loadings_ci" in bootres:
            y_ci = bootres["y_loadings_ci"]
            y_ci_df = pd.DataFrame({"feature": behav_cols})
            for mi in range(y_ci.shape[1]):
                y_ci_df[f"ci_lo_mode{mi + 1}"] = y_ci[:, mi, 0]
                y_ci_df[f"ci_hi_mode{mi + 1}"] = y_ci[:, mi, 1]
            y_ci_df.to_csv(outdir / "bootstrap_y_loadings_ci.csv", index=False)

    if "y_loadings" in pls_results:
        y_loadings_full = pls_results["y_loadings"]
        y_load_full_df = pd.DataFrame({"feature": behav_cols})
        for mi in range(y_loadings_full.shape[1]):
            y_load_full_df[f"loading_mode{mi + 1}"] = y_loadings_full[:, mi]
        y_load_full_df.to_csv(outdir / "y_loadings.csv", index=False)

    cv_result = None
    if significant_modes:
        print(
            f"  Running {opts.kfolds}-fold CV for "
            f"{len(significant_modes)} significant mode(s)..."
        )
        cv_result = cv_assess_modes(
            X=X, Y=Y,
            x_weights=x_weights, y_weights=y_weights,
            significant_modes=significant_modes,
            df=df,
            group_col=opts.group_col,
            kfolds=int(opts.kfolds),
            seed=int(opts.seed) + 777,
            cv_strategy=opts.cv_strategy,
        )

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
        pd.DataFrame(cv_rows).to_csv(outdir / "cv_fold_correlations.csv", index=False)

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

        print("  CV summary:")
        for _, row in cv_summary.iterrows():
            print(
                f"    Mode {int(row['mode'])}: mean r = {row['mean_r']:.4f} "
                f"(sd = {row['sd_r']:.4f}), mean |r| = {row['mean_abs_r']:.4f}"
            )
    else:
        print("  No significant modes — skipping CV.")
        pd.DataFrame(columns=["fold", "mode", "r", "abs_r", "fisher_z", "n_test"]).to_csv(
            outdir / "cv_fold_correlations.csv", index=False
        )
        pd.DataFrame(columns=["mode", "mean_r", "sd_r", "mean_abs_r", "sd_abs_r",
                               "mean_fisher_z", "sd_fisher_z", "perm_pval", "perm_pval_fdr"]).to_csv(
            outdir / "cv_summary.csv", index=False
        )

    summary_lines: list[str] = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"PLS Trajectory Analysis: {model_label}")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Subjects: {len(df)}")
    summary_lines.append(f"Brain features (network trajectory slopes): {len(imaging_cols)}")
    summary_lines.append(f"Behaviour features (trajectory slopes): {len(behav_cols)}")
    summary_lines.append(f"PLS modes extracted: {n_modes} (max = {n_modes_max})")
    summary_lines.append(f"Slopes model tag: {opts.slopes_model_tag}")
    summary_lines.append(f"FD residualization: {bool(getattr(opts, 'use_fd_residualize', False))}")
    summary_lines.append(f"ComBat harmonization: {bool(opts.use_combat)}")
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
    if opts.group_col and opts.group_col in df.columns:
        site_counts = df[opts.group_col].astype(str).value_counts().sort_index()
        summary_lines.append(f"Site distribution: {site_counts.to_dict()}")

    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n" + "\n".join(summary_lines))

    manifest_entries = [
        ("behavior_used.csv", "Behaviour trajectory slope features used in PLS"),
        ("behavior_variances.csv", "Variance of each behavioural slope feature"),
        ("behavior_outliers.csv", "Subjects flagged as outliers by z-score threshold"),
        ("behavior_cor_matrix.csv", "Correlation matrix of behavioural slope features"),
        ("behavior_redundancy_pairs.csv", "High-correlation behavioural slope pairs"),
        ("combined_input.csv", "Merged input data (brain + behavioural slopes)"),
        ("saliences_X.csv", "Brain saliences (network slope weights) per PLS mode"),
        ("saliences_Y.csv", "Behaviour saliences (slope weights) per PLS mode"),
        ("scores.csv", "Subject scores on each PLS mode (u and v)"),
        ("mode_summary.csv", "Per-mode summary: singular values, variance, correlations, p-values"),
        ("cv_fold_correlations.csv", "Per-fold CV correlations for significant modes"),
        ("cv_summary.csv", "CV summary statistics for significant modes"),
        ("summary.txt", "Full analysis summary"),
        ("manifest.csv", "This file"),
    ]
    if (outdir / "permutation_null.csv").exists():
        manifest_entries.append(("permutation_null.csv", "Permutation null distribution of singular values"))
    if dropped_log:
        manifest_entries.insert(1, ("dropped_columns_log.csv", "Log of dropped behavioural columns"))
    if "y_loadings" in pls_results:
        manifest_entries.append(("y_loadings.csv", "Behavioural loadings from full-sample PLS"))
    if "bootres" in pls_results and pls_results["bootres"] is not None:
        br = pls_results["bootres"]
        if "x_weights_normed" in br:
            manifest_entries.append(("bootstrap_ratios_X.csv", "Bootstrap ratios for brain saliences"))
        if "x_weights_stderr" in br:
            manifest_entries.append(("bootstrap_stderr_X.csv", "Bootstrap standard errors for brain saliences"))
        if "y_loadings" in br:
            manifest_entries.append(("bootstrap_y_loadings.csv", "Bootstrap behavioural loadings"))
        if "y_loadings_ci" in br:
            manifest_entries.append(("bootstrap_y_loadings_ci.csv", "Bootstrap CIs for behavioural loadings"))

    hdf5_saved = False
    try:
        pyls_pkg.save_results(str(outdir / "pls_results.hdf5"), pls_results)
        hdf5_saved = True
        print(f"  pyls results saved to {outdir / 'pls_results.hdf5'}")
    except Exception as e:
        print(f"  Warning: could not save pyls results object ({e})")
    if hdf5_saved:
        manifest_entries.append(("pls_results.hdf5", "Full pyls results object (HDF5)"))

    pd.DataFrame(manifest_entries, columns=["filename", "description"]).to_csv(
        outdir / "manifest.csv", index=False
    )
    print(f"  Outputs saved to {outdir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Behavioral PLS on rs-fMRI wDC developmental trajectory slopes "
            "vs behavioural trajectory slopes. "
            f"Valid model names: {', '.join(MODEL_CONFIGS)}."
        )
    )

    p.add_argument(
        "--brain-slopes-csv", type=Path, default=_default_brain_slopes_csv(),
        help="random_slopes.csv from run_growth_curves_with_random_slopes.py.",
    )
    p.add_argument(
        "--behav-slopes-csv", type=Path, default=_default_behav_slopes_csv(),
        help="random_slopes.csv from run_growth_curves_with_random_slopes_behavioural.py.",
    )
    p.add_argument(
        "--slopes-model-tag", type=str, default="M2_LMM",
        help="Model tag to extract (e.g. M1_LMM, M2_LMM).",
    )
    p.add_argument(
        "--slopes-brain-feature-col", type=str, default="network",
        help="Column identifying brain features in the brain slopes CSV.",
    )
    p.add_argument(
        "--slopes-behav-feature-col", type=str, default="variable",
        help="Column identifying behavioural features in the behav slopes CSV.",
    )
    p.add_argument(
        "--slopes-reliable-only", type=lambda x: str(x).lower() != "false", default=True,
        help="Exclude rows where slope_reliable=False (default: True).",
    )
    p.add_argument(
        "--models", type=str, default="all",
        help=(
            f"Comma-separated model names to run, or 'all'. "
            f"Valid: {', '.join(MODEL_CONFIGS)}."
        ),
    )
    p.add_argument("--subject-col", type=str, default="subject")
    p.add_argument(
        "--demo-csv", type=Path, default=None,
        help="Optional demographics CSV to merge site/age/sex (defaults to df.csv).",
    )
    p.add_argument("--demo-subject-col", type=str, default="subjects")

    p.add_argument(
        "--outdir", type=Path, default=_default_pls_trajectories_output_dir(),
        help="Base output directory. Per-model outputs go to {outdir}/{model_name}/.",
    )
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--nperms", type=int, default=5000)
    p.add_argument("--nboots", type=int, default=5000)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perm-alpha", type=float, default=0.05)
    p.add_argument(
        "--n-proc", type=str, default=None,
        help="Parallel workers for pyls. Use 'max' for all cores.",
    )

    p.add_argument(
        "--use-combat", type=lambda x: str(x).lower() == "true", default=False,
        help="ComBat harmonisation on brain slopes (default: False; LMM handles site).",
    )
    p.add_argument(
        "--use-fd-residualize", type=lambda x: str(x).lower() == "true", default=False,
        help="FD residualisation on brain slopes (default: False; LMM handles motion).",
    )
    p.add_argument("--group-col", type=str, default="site")
    p.add_argument("--age-col", type=str, default="t1_ageyrs")
    p.add_argument("--sex-col", type=str, default="t1_sex")

    p.add_argument("--qc-csv", type=Path, default=None)
    p.add_argument("--qc-subject-col", type=str, default="subject")
    p.add_argument("--qc-fd-col", type=str, default="meanFD")

    p.add_argument("--max-subjects", type=int, default=0)
    p.add_argument("--behav-outlier-z", type=float, default=3.0)
    p.add_argument("--behav-redundancy-thresh", type=float, default=0.8)
    p.add_argument("--drop-nzv", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--drop-redundant", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument(
        "--cv-strategy", type=str, default="random",
        choices=["random", "group-kfold", "leave-group-out"],
    )

    return p.parse_args()


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

    print(f"\nLoading brain trajectory slopes from {opts.brain_slopes_csv}...")
    if not opts.brain_slopes_csv.exists():
        raise FileNotFoundError(
            f"Brain slopes file not found: {opts.brain_slopes_csv}\n"
            "Run run_growth_curves_with_random_slopes.py first."
        )
    brain_wide, imaging_cols_all = load_random_slopes_wide(
        opts.brain_slopes_csv,
        model_tag=opts.slopes_model_tag,
        feature_col=opts.slopes_brain_feature_col,
        subject_col=opts.subject_col,
        reliable_only=opts.slopes_reliable_only,
    )

    print(f"\nLoading behavioural trajectory slopes from {opts.behav_slopes_csv}...")
    if not opts.behav_slopes_csv.exists():
        raise FileNotFoundError(
            f"Behavioural slopes file not found: {opts.behav_slopes_csv}\n"
            "Run run_growth_curves_with_random_slopes_behavioural.py first."
        )
    behav_slopes_raw = _try_read_table(opts.behav_slopes_csv)

    demo_df: Optional[pd.DataFrame] = None
    demo_csv_path = opts.demo_csv
    if demo_csv_path is None:
        try:
            from rs_paths import default_behaviour_csv
            demo_csv_path = default_behaviour_csv()
        except ImportError:
            pass
    if demo_csv_path is not None and Path(str(demo_csv_path)).exists():
        try:
            demo_df = pd.read_csv(demo_csv_path)
            demo_df[opts.demo_subject_col] = demo_df[opts.demo_subject_col].astype(str)
            print(f"Loaded demographics from {demo_csv_path} ({len(demo_df)} rows)")
        except Exception as e:
            print(f"Warning: could not load demographics CSV ({e})")

    ensure_dir(opts.outdir)

    for model_name in selected_models:
        config = MODEL_CONFIGS[model_name]
        model_label = config["label"]
        model_behav_vars: list[str] = config["behav_vars"]
        outdir = opts.outdir / model_name

        print(f"\n{'=' * 64}")
        print(f"Model: {model_name}  —  {model_label}")
        print(f"Behaviour variables: {model_behav_vars}")
        print(f"Output: {outdir}")
        print(f"{'=' * 64}")

        ensure_dir(outdir)
        if _dir_has_files(outdir) and not bool(getattr(opts, "overwrite", False)):
            print(f"SKIP: {outdir} is not empty. Pass --overwrite to re-run.")
            continue
        if bool(getattr(opts, "overwrite", False)):
            _cleanup_previous_outputs_if_manifest_present(outdir)

        # Pivot behavioural slopes for this model
        try:
            behav_model_df = behav_slopes_raw.copy()
            if "model" in behav_model_df.columns:
                behav_model_df = behav_model_df[
                    behav_model_df["model"] == opts.slopes_model_tag
                ]
            if opts.slopes_reliable_only and "slope_reliable" in behav_model_df.columns:
                behav_model_df = behav_model_df[
                    behav_model_df["slope_reliable"].astype(str).str.lower() == "true"
                ]
            behav_model_df = behav_model_df[
                behav_model_df[opts.slopes_behav_feature_col].isin(model_behav_vars)
            ].copy()

            missing_vars = [
                v for v in model_behav_vars
                if v not in behav_model_df[opts.slopes_behav_feature_col].unique()
            ]
            if missing_vars:
                print(f"  Warning: variables absent after reliability filter: {missing_vars}")
            if behav_model_df.empty:
                print(f"  SKIP: no reliable behavioural slopes for {model_behav_vars}.")
                continue

            behav_wide = (
                behav_model_df.pivot_table(
                    index=opts.subject_col,
                    columns=opts.slopes_behav_feature_col,
                    values="re_slope",
                    aggfunc="mean",
                )
                .reset_index()
            )
            behav_wide.columns = [str(c) for c in behav_wide.columns]
            behav_wide[opts.subject_col] = behav_wide[opts.subject_col].astype(str)
            behav_cols = [c for c in behav_wide.columns if c != opts.subject_col]
            print(f"  Behavioural slopes: {len(behav_wide)} subjects × {len(behav_cols)} features")

        except Exception as e:
            print(f"  ERROR loading behavioural slopes for '{model_name}': {e}")
            print("  Skipping this model.")
            continue

        df = brain_wide.merge(behav_wide, on=opts.subject_col, how="inner")
        imaging_cols = imaging_cols_all.copy()
        if opts.max_subjects and opts.max_subjects > 0:
            df = df.sort_values(opts.subject_col).head(opts.max_subjects)
        df = df.reset_index(drop=True)
        print(f"  Subjects after brain ∩ behaviour merge: {len(df)}")

        if len(df) < 10:
            print(f"  SKIP: too few subjects ({len(df)}) after merge.")
            continue

        if demo_df is not None:
            demo_cols = [opts.demo_subject_col]
            for c in [opts.group_col, opts.age_col, opts.sex_col]:
                if c and c in demo_df.columns and c not in demo_cols:
                    demo_cols.append(c)
            demo_sub = demo_df[list(dict.fromkeys(demo_cols))].copy()
            if "timepoint" in demo_sub.columns:
                demo_sub = demo_sub[demo_sub["timepoint"] == "T1"].drop(columns=["timepoint"])
            demo_sub = demo_sub.drop_duplicates(subset=[opts.demo_subject_col])
            df = df.merge(demo_sub, left_on=opts.subject_col,
                          right_on=opts.demo_subject_col, how="left")
            if opts.demo_subject_col != opts.subject_col and opts.demo_subject_col in df.columns:
                df = df.drop(columns=[opts.demo_subject_col])
            df = df.reset_index(drop=True)

        try:
            run_pls_for_model(
                df=df,
                imaging_cols=imaging_cols,
                behav_cols_in=behav_cols,
                opts=opts,
                outdir=outdir,
                model_label=model_label,
                n_proc=n_proc,
            )
        except Exception as e:
            import traceback
            print(f"  ERROR in PLS for '{model_name}': {e}")
            traceback.print_exc()
            print("  Continuing with next model...")

    print(f"\n{'=' * 64}")
    print(f"All done. Results in {opts.outdir}")
    print(f"{'=' * 64}")

if __name__ == "__main__":
    main()

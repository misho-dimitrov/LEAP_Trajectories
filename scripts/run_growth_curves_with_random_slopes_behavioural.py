#!/usr/bin/env python
"""run_growth_curves_with_random_slopes_behavioural.py
=======================================================
Growth-curve models for developmental behavioural trajectories.

Outcomes
--------
Eleven behavioural variables (using whichever timepoints are available):
  - SDQ Total Difficulties – parent (T1/T2/T3)
  - SRS Raw Score – parent (T1/T2/T3; T3 sourced from t3_srs_rawscore_total)
  - RBS Total (T1/T2/T3)
  - SSP Total (T1/T2/T3)
  - PRL perseverative errors proportion (T1/T2/T3)
  - PRL Win-Stay (T1/T2/T3)
  - PRL Lose-Shift (T1/T2/T3)
  - WHOQoL-Bref Overall QoL / General Health – raw (T1/T3 only; no T2)
  - ASHQ Total (T1/T2/T3)
  - TAS Total (T1/T2/T3)
  - Vineland ABC Standard Score (T1/T2/T3)

Models
------
  Model 1 – Age model (cross-sectional + longitudinal pooled):
      outcome ~ age × group + sex + site + (1 + age | subject)      [LMM]
      outcome ~ bs(age) × group + sex + site + (1 | subject)         [GAMM]

  Model 2 – Longitudinal model (only when mean timepoints/subject ≥ 1.3):
      outcome ~ time_elapsed × baseline_age × group + sex + site
                + (1 + time_elapsed | subject)                        [LMM]
      outcome ~ bs(time_elapsed) × group + baseline_age × group
                + sex + site + (1 | subject)                          [GAMM]

LMM models include random slopes; GAMM models are intercept-only
(random slopes on spline bases are rarely identifiable).

Random-slope identifiability is assessed after fitting:
  - fit.converged must be True
  - the random-slope variance must exceed _MIN_SLOPE_VAR
Slopes that fail either check are flagged slope_reliable=False in
random_slopes.csv.

Permutation testing
-------------------
Group labels are shuffled across subjects (preserving within-subject
repeated-measures structure); all models are re-fit on the permuted data.
Per-term permutation p-values are computed as:
    p_perm = (n_perm_|z| >= observed_|z| + 1) / (n_perms + 1)

FDR correction
--------------
Benjamini–Hochberg FDR is applied to both parametric (``p``) and
permutation (``p_perm``) p-values.  Default scope: ``per_model_term``.

Usage
-----
    python run_growth_curves_with_random_slopes_behavioural.py
    python run_growth_curves_with_random_slopes_behavioural.py --n-perms 0
    python run_growth_curves_with_random_slopes_behavioural.py --help
"""

from __future__ import annotations

import argparse
import logging
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
        default_behaviour_csv,
        rs_project_root,
    )
except ImportError:
    def rs_project_root() -> Path:
        return Path(__file__).resolve().parents[1]

    def default_behaviour_csv() -> Path:
        return rs_project_root().parent / "Behaviour" / "df.csv"


def default_growth_curves_output_dir() -> Path:
    return rs_project_root() / "reports" / "growth_curves_behavioural"


LOG = logging.getLogger("growth_curves_behavioural")


# ============================================================================
#  CONSTANTS
# ============================================================================

# --- Group / plotting constants --------------------------------------------

GROUP_LABEL_MAP = {1: "TD", 2: "ASD", 3: "ID_control", 4: "ID_ASD"}
GROUP_REF = "TD"
MISSING_SENTINEL = 999

# Special numeric codes used as NA placeholders in df.csv
MISSING_CODES: set = {777, 999}

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
    "M1_LMM": "~age_c",  # age_c = age − mean(age), created in _prepare_variable_df
    "M2_LMM": "~time_elapsed",
}

# Minimum random-slope variance below which an estimate is flagged as
# unreliable (boundary / near-zero solution).
_MIN_SLOPE_VAR: float = 1e-6

# ---------------------------------------------------------------------------
#  Behavioural variable definitions
#
#  Each entry has:
#    name     – short identifier used in output filenames / columns
#    label    – human-readable label for plot axes
#    tp_cols  – dict mapping timepoint ("T1"/"T2"/"T3") to the raw
#               column name in df.csv.  Timepoints absent from the dict
#               are simply not collected for that variable.
#
#  Note: SRS uses a different column name at T3 (t3_srs_rawscore_total
#  instead of t{N}_srs_rawscore).  WHOQoL has no T2 column in df.csv.
# ---------------------------------------------------------------------------

BEHAVIOURAL_VARIABLES: List[Dict] = [
    {
        "name":    "sdq_total_p",
        "label":   "SDQ Total Difficulties (parent)",
        "tp_cols": {
            "T1": "t1_sdq_total_difficulties_p",
            "T2": "t2_sdq_total_difficulties_p",
            "T3": "t3_sdq_total_difficulties_p",
        },
    },
    {
        "name":    "sdq_emotional_p",
        "label":   "SDQ Emotional Problems (parent)",
        "tp_cols": {
            "T1": "t1_sdq_emotional_p",
            "T2": "t2_sdq_emotional_p",
            "T3": "t3_sdq_emotional_p",
        },
    },
    {
        "name":    "sdq_conduct_p",
        "label":   "SDQ Conduct Problems (parent)",
        "tp_cols": {
            "T1": "t1_sdq_conduct_p",
            "T2": "t2_sdq_conduct_p",
            "T3": "t3_sdq_conduct_p",
        },
    },
    {
        "name":    "sdq_hyperactivity_p",
        "label":   "SDQ Hyperactivity (parent)",
        "tp_cols": {
            "T1": "t1_sdq_hyperactivity_p",
            "T2": "t2_sdq_hyperactivity_p",
            "T3": "t3_sdq_hyperactivity_p",
        },
    },
    {
        "name":    "sdq_peer_p",
        "label":   "SDQ Peer Problems (parent)",
        "tp_cols": {
            "T1": "t1_sdq_peer_p",
            "T2": "t2_sdq_peer_p",
            "T3": "t3_sdq_peer_p",
        },
    },
    {
        "name":    "sdq_prosocial_p",
        "label":   "SDQ Prosocial Behaviour (parent)",
        "tp_cols": {
            "T1": "t1_sdq_prosocial_p",
            "T2": "t2_sdq_prosocial_p",
            "T3": "t3_sdq_prosocial_p",
        },
    },
    {
        "name":    "srs_rawscore",
        "label":   "SRS Raw Score (parent)",
        "tp_cols": {
            "T1": "t1_srs_rawscore",
            "T2": "t2_srs_rawscore",
            "T3": "t3_srs_rawscore_total",  # column name differs at T3
        },
    },
    {
        "name":    "rbs_total",
        "label":   "RBS Total",
        "tp_cols": {
            "T1": "t1_rbs_total",
            "T2": "t2_rbs_total",
            "T3": "t3_rbs_total",
        },
    },
    {
        "name":    "ssp_total",
        "label":   "SSP Total",
        "tp_cols": {
            "T1": "t1_ssp_total",
            "T2": "t2_ssp_total",
            "T3": "t3_ssp_total",
        },
    },
    {
        "name":    "prl_perE_prop",
        "label":   "PRL Perseverative Errors (proportion)",
        "tp_cols": {
            "T1": "t1_prl_perE_prop",
            "T2": "t2_prl_perE_prop",
            "T3": "t3_prl_perE_prop",
        },
    },
    {
        "name":    "prl_WS",
        "label":   "PRL Win-Stay",
        "tp_cols": {
            "T1": "t1_prl_WS",
            "T2": "t2_prl_WS",
            "T3": "t3_prl_WS",
        },
    },
    {
        "name":    "prl_LS",
        "label":   "PRL Lose-Shift",
        "tp_cols": {
            "T1": "t1_prl_LS",
            "T2": "t2_prl_LS",
            "T3": "t3_prl_LS",
        },
    },
    {
        "name":  "whoqol_overall",
        "label": "WHOQoL-Bref Overall QoL & General Health (raw)",
        "tp_cols": {
            "T1": "t1_whoqolbref_overall_quality_of_life_and_general_health_raw",
            # T2 column not available in df.csv
            "T3": "t3_whoqolbref_overall_quality_of_life_and_general_health_raw",
        },
    },
    {
        "name":    "ashq_total",
        "label":   "ASHQ Total",
        "tp_cols": {
            "T1": "t1_ashq_total",
            "T2": "t2_ashq_total",
            "T3": "t3_ashq_total",
        },
    },
    {
        "name":    "tas_total",
        "label":   "TAS Total",
        "tp_cols": {
            "T1": "t1_tas_total",
            "T2": "t2_tas_total",
            "T3": "t3_tas_total",
        },
    },
    {
        "name":    "tas_identify",
        "label":   "TAS Difficulty Identifying Feelings",
        "tp_cols": {
            "T1": "t1_tas_identify",
            "T2": "t2_tas_identify",
            "T3": "t3_tas_identify",
        },
    },
    {
        "name":    "tas_describe",
        "label":   "TAS Difficulty Describing Feelings",
        "tp_cols": {
            "T1": "t1_tas_describe",
            "T2": "t2_tas_describe",
            "T3": "t3_tas_describe",
        },
    },
    {
        "name":    "tas_external",
        "label":   "TAS Externally Oriented Thinking",
        "tp_cols": {
            "T1": "t1_tas_external",
            "T2": "t2_tas_external",
            "T3": "t3_tas_external",
        },
    },
    {
        "name":    "vineland_abc",
        "label":   "Vineland ABC Standard Score",
        "tp_cols": {
            "T1": "t1_vabsabcabc_standard",
            "T2": "t2_vabsabcabc_standard",
            "T3": "t3_vabsabcabc_standard",
        },
    },
]


# ============================================================================
#  Data loading & preparation
# ============================================================================

def load_demographics_wide(path: Path) -> pd.DataFrame:
    """Load df.csv → wide-format DataFrame with demographics + all behavioural columns.

    Applies missing-code handling (777 and 999 → NaN) to every numeric
    behavioural column, and restricts to ASD + TD subjects only.
    """
    LOG.info("Loading demographics/behaviour from %s", path)
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip().str.lower()

    if "subjects" in raw.columns and "subject" not in raw.columns:
        raw = raw.rename(columns={"subjects": "subject"})

    raw["subject"] = raw["subject"].astype(str).str.strip().str[:6]

    # Collect all behavioural columns referenced in BEHAVIOURAL_VARIABLES
    behav_all_cols: List[str] = []
    for var in BEHAVIOURAL_VARIABLES:
        for col in var["tp_cols"].values():
            col_lc = col.lower()
            if col_lc in raw.columns and col_lc not in behav_all_cols:
                behav_all_cols.append(col_lc)

    keep = ["subject", "t1_ageyrs", "t2_ageyrs", "t3_ageyrs",
            "t1_sex", "t1_group", "t1_site", "t1_fsiq"]
    keep = [c for c in keep if c in raw.columns] + behav_all_cols
    df = raw[keep].copy()

    # Replace sentinel values with NaN in demographic numeric columns
    for col in ("t1_ageyrs", "t2_ageyrs", "t3_ageyrs", "t1_sex", "t1_fsiq"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] >= MISSING_SENTINEL, col] = np.nan

    # Replace 777 and 999 with NaN in every behavioural column
    for col in behav_all_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(MISSING_CODES), col] = np.nan

    # Group filter: keep ASD + TD only
    if "t1_group" in df.columns:
        df["group_str"] = df["t1_group"].map(GROUP_LABEL_MAP)
    else:
        raise ValueError("Column 't1_group' not found in demographics CSV.")

    df = df[df["group_str"].isin(["ASD", "TD"])].copy()
    LOG.info("Keeping %d subjects after group filter (ASD + TD only).", len(df))

    # Sex: t1_sex is coded as 1/2 (or -1/1) in LEAP; convert so that the
    # contrast is a simple 0/1 numeric predictor.
    if "t1_sex" in df.columns:
        df["sex"] = df["t1_sex"] - 1

    if "t1_fsiq" in df.columns:
        df["iq"] = df["t1_fsiq"]

    # Site: convert numeric code to string for C(site) in model formulas
    if "t1_site" in df.columns:
        df["site"] = df["t1_site"].apply(
            lambda v: str(int(v)) if pd.notna(v) and float(v) < MISSING_SENTINEL else np.nan
        )

    return df.reset_index(drop=True)


def build_behavioural_long_df(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide demographics+behaviour table → long format.

    Returns one row per subject × timepoint × variable with columns:
        subject, timepoint, age, baseline_age, time_elapsed,
        group, sex, site, variable, value
    """
    age_cols = {"T1": "t1_ageyrs", "T2": "t2_ageyrs", "T3": "t3_ageyrs"}
    rows: List[dict] = []

    for _, r in wide_df.iterrows():
        base_age = r.get("t1_ageyrs", np.nan)

        for var in BEHAVIOURAL_VARIABLES:
            for tp, src_col in var["tp_cols"].items():
                src_col_lc = src_col.lower()
                if src_col_lc not in wide_df.columns:
                    continue
                val = r.get(src_col_lc, np.nan)
                if pd.isna(val):
                    continue  # no data for this subject × timepoint × variable

                age_col = age_cols.get(tp)
                age_val = r.get(age_col, np.nan) if age_col else np.nan
                if pd.isna(age_val):
                    continue  # no age → cannot place on timeline

                rows.append({
                    "subject":      r["subject"],
                    "timepoint":    tp,
                    "age":          float(age_val),
                    "baseline_age": float(base_age) if pd.notna(base_age) else np.nan,
                    "time_elapsed": float(age_val) - float(base_age)
                                    if pd.notna(base_age) else np.nan,
                    "group":        r.get("group_str", np.nan),
                    "sex":          r.get("sex", np.nan),
                    "iq":           r.get("iq", np.nan),
                    "site":         r.get("site", np.nan),
                    "variable":     var["name"],
                    "value":        float(val),
                })

    long_df = pd.DataFrame(rows)

    required = ["age", "baseline_age", "group", "sex", "iq", "site", "value"]
    n_before = len(long_df)
    long_df = long_df.dropna(subset=required)
    n_dropped = n_before - len(long_df)
    if n_dropped:
        LOG.info("Dropped %d rows with missing required variables.", n_dropped)

    LOG.info(
        "Behavioural long DF ready: %d rows, %d subjects, %d variables, timepoints %s",
        len(long_df), long_df["subject"].nunique(),
        long_df["variable"].nunique(), sorted(long_df["timepoint"].unique()),
    )
    return long_df


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
    fit, model_tag: str, variable: str, converged: bool = True,
) -> List[dict]:
    """Pull fixed-effect coefficients from a fitted MixedLM result."""
    if fit is None:
        return []
    rows = []
    for term in fit.params.index:
        ci = fit.conf_int()
        rows.append({
            "variable":  variable,
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
    variable: str,
    slope_var_name: str,
    converged: bool,
) -> List[dict]:
    """Extract per-subject random intercepts and slopes from a fitted MixedLM.

    Returns one row per subject with columns:
        subject, model, variable, slope_var_name,
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
            slope_var_name, model_tag, variable, list(cov_re.columns),
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
            model_tag, variable, converged,
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
            "variable":       variable,
            "slope_var_name": slope_var_name,
            "re_intercept":   re_dict.get(intercept_key, np.nan),
            "re_slope":       re_dict.get(slope_col, np.nan),
            "slope_var":      slope_var,
            "slope_reliable": slope_reliable,
        })
    return rows


def _build_formulas(var_name: str, bs_df: int, has_longitudinal: bool) -> Dict[str, str]:
    """Return a dict of {model_tag: formula} for all models to fit."""
    formulas: Dict[str, str] = {
        "M1_LMM":  f"{var_name} ~ age_c * C(group) + sex + iq + C(site)",
        "M1_GAMM": f"{var_name} ~ bs(age_c, df={bs_df}) * C(group) + sex + iq + C(site)",
    }
    if has_longitudinal:
        formulas["M2_LMM"] = (
            f"{var_name} ~ time_elapsed * baseline_age * C(group) + sex + iq + C(site)"
        )
        formulas["M2_GAMM"] = (
            f"{var_name} ~ bs(time_elapsed, df={bs_df}) * C(group)"
            f" + baseline_age * C(group) + sex + iq + C(site)"
        )
    return formulas


def _prepare_variable_df(long_df: pd.DataFrame, var_name: str) -> Optional[pd.DataFrame]:
    """Subset long_df for *var_name*, rename value column, set group as Categorical."""
    df = long_df[long_df["variable"] == var_name].copy()
    if len(df) < 30:
        LOG.warning("Skipping '%s': only %d observations.", var_name, len(df))
        return None
    # Rename generic "value" column to var_name so patsy formulas resolve it
    df = df.rename(columns={"value": var_name})
    other_groups = sorted(g for g in df["group"].unique() if g != GROUP_REF)
    df["group"] = pd.Categorical(
        df["group"], categories=[GROUP_REF] + other_groups,
    )
    df["age_c"] = df["age"] - df["age"].mean()
    return df


# ============================================================================
#  Model fitting (parametric)
# ============================================================================

def fit_models_for_variable(
    long_df: pd.DataFrame,
    var_name: str,
    bs_df: int = 4,
) -> Tuple[List[dict], List[dict]]:
    """Fit all models for one *var_name*.

    Returns
    -------
    fixed_rows : list of dict
        Fixed-effect coefficient rows (one per term per model).
    slope_rows : list of dict
        Per-subject random-slope rows for LMM models only.
    """
    df = _prepare_variable_df(long_df, var_name)
    if df is None:
        return [], []

    tp_counts = df.groupby("subject")["timepoint"].nunique()
    has_longitudinal = tp_counts.mean() >= 1.3
    formulas = _build_formulas(var_name, bs_df, has_longitudinal)

    fixed_rows: List[dict] = []
    slope_rows: List[dict] = []

    for model_tag in sorted(formulas):
        re_formula = LMM_RE_FORMULAS.get(model_tag)  # None for GAMMs
        fit, converged = _safe_mixedlm(
            formulas[model_tag], df, "subject",
            tag=f"{model_tag}|{var_name}",
            re_formula=re_formula,
        )
        fixed_rows.extend(_extract_coefficients(fit, model_tag, var_name, converged))
        if re_formula is not None:
            slope_var_name = re_formula.lstrip("~").strip()
            slope_rows.extend(
                _extract_random_slopes(
                    fit, model_tag, var_name, slope_var_name, converged,
                )
            )

    if not has_longitudinal:
        LOG.info(
            "Skipping Model 2 for '%s': mean %.1f timepoints/subject.",
            var_name, tp_counts.mean(),
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
    var_names = [v["name"] for v in BEHAVIOURAL_VARIABLES]
    LOG.info("Fitting models for %d behavioural variables …", len(var_names))

    all_rows: List[dict] = []
    all_slope_rows: List[dict] = []
    rng = np.random.default_rng(seed)

    for i, var_name in enumerate(var_names, 1):
        LOG.info("  [%d/%d] %s – fitting parametric models …", i, len(var_names), var_name)
        var_results, var_slopes = fit_models_for_variable(long_df, var_name, bs_df=bs_df)
        all_slope_rows.extend(var_slopes)

        # ── Permutation testing ─────────────────────────────────────────
        if n_perms > 0 and var_results:
            LOG.info(
                "  [%d/%d] %s – running %d permutations (n_jobs=%s) …",
                i, len(var_names), var_name, n_perms,
                "all cores" if n_jobs == -1 else n_jobs,
            )
            perm_pvals = _run_permutations_for_variable(
                long_df, var_name, var_results,
                n_perms=n_perms, n_jobs=n_jobs, bs_df=bs_df,
                seed=rng.integers(0, 2**31),
            )
            for row in var_results:
                key = (row["model"], row["term"])
                row["p_perm"] = perm_pvals.get(key, np.nan)
        else:
            for row in var_results:
                row["p_perm"] = np.nan

        all_rows.extend(var_results)

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


def _run_permutations_for_variable(
    long_df: pd.DataFrame,
    var_name: str,
    observed_results: List[dict],
    n_perms: int = 5000,
    n_jobs: int = -1,
    bs_df: int = 4,
    seed: int = 42,
) -> Dict[Tuple[str, str], float]:
    """Run permutation test for one variable; return {(model, term): p_perm}."""
    df_var = _prepare_variable_df(long_df, var_name)
    if df_var is None:
        return {}

    # Subject-level group assignments
    sub_group = df_var.groupby("subject")["group"].first()
    subjects = sub_group.index.values
    groups = sub_group.values.astype(str)

    tp_counts = df_var.groupby("subject")["timepoint"].nunique()
    has_longitudinal = tp_counts.mean() >= 1.3
    formulas = _build_formulas(var_name, bs_df, has_longitudinal)

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
            int(s), df_var, subjects, groups, formulas, GROUP_REF, LMM_RE_FORMULAS,
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

def _get_var_label(var_name: str) -> str:
    """Return the human-readable label for *var_name*."""
    for v in BEHAVIOURAL_VARIABLES:
        if v["name"] == var_name:
            return v["label"]
    return var_name


def _get_colour(group: str) -> str:
    return GROUP_COLOURS.get(group, "#9E9E9E")


def plot_spaghetti_age(
    long_df: pd.DataFrame, var_name: str, output_dir: Path,
) -> None:
    """Spaghetti plot: outcome vs age, individual trajectories + group smooth."""
    df = long_df[long_df["variable"] == var_name].copy()
    if df.empty:
        return
    y_label = _get_var_label(var_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group in sorted(df["group"].unique()):
        gdf = df[df["group"] == group]
        colour = _get_colour(group)
        for _subj, sdf in gdf.groupby("subject"):
            sdf = sdf.sort_values("age")
            ax.plot(sdf["age"], sdf["value"],
                    color=colour, alpha=0.12, linewidth=0.4)
        if len(gdf) > 10:
            smoothed = sm_lowess(
                gdf["value"].values, gdf["age"].values,
                frac=0.6, return_sorted=True,
            )
            ax.plot(smoothed[:, 0], smoothed[:, 1],
                    color=colour, linewidth=2.5, label=group)
        else:
            ax.plot([], [], color=colour, linewidth=2.5, label=group)

    ax.set_xlabel("Age (years)")
    ax.set_ylabel(y_label)
    ax.set_title(var_name)
    ax.legend(title="Group", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"spaghetti_age_{var_name}.png", dpi=150)
    plt.close(fig)


def plot_spaghetti_time(
    long_df: pd.DataFrame, var_name: str, output_dir: Path,
) -> None:
    """Spaghetti plot: outcome vs time_elapsed, coloured by group."""
    df = long_df[long_df["variable"] == var_name].copy()
    if df.empty:
        return
    y_label = _get_var_label(var_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group in sorted(df["group"].unique()):
        gdf = df[df["group"] == group]
        colour = _get_colour(group)
        for _subj, sdf in gdf.groupby("subject"):
            sdf = sdf.sort_values("time_elapsed")
            ax.plot(sdf["time_elapsed"], sdf["value"],
                    color=colour, alpha=0.12, linewidth=0.4)
        if len(gdf) > 10:
            smoothed = sm_lowess(
                gdf["value"].values, gdf["time_elapsed"].values,
                frac=0.6, return_sorted=True,
            )
            ax.plot(smoothed[:, 0], smoothed[:, 1],
                    color=colour, linewidth=2.5, label=group)
        else:
            ax.plot([], [], color=colour, linewidth=2.5, label=group)

    ax.set_xlabel("Time elapsed from baseline (years)")
    ax.set_ylabel(y_label)
    ax.set_title(var_name)
    ax.legend(title="Group", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"spaghetti_time_{var_name}.png", dpi=150)
    plt.close(fig)


def generate_all_plots(long_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate spaghetti plots for every behavioural variable."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    var_names = sorted(long_df["variable"].unique())
    LOG.info("Generating plots for %d variables …", len(var_names))
    for var_name in var_names:
        plot_spaghetti_age(long_df, var_name, plot_dir)
        plot_spaghetti_time(long_df, var_name, plot_dir)
    LOG.info("Plots saved to %s", plot_dir)


# ============================================================================
#  Main
# ============================================================================

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Growth-curve models for developmental behavioural trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--demo", type=Path, default=None,
        help="Path to demographics/behaviour CSV (default: auto-detected via rs_paths).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: reports/growth_curves_behavioural).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--bs-df", type=int, default=4,
        help="B-spline degrees of freedom for the GAMM models (default: 4).",
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
             "'per_model_term' corrects across variables within each model × "
             "term.  'per_model' corrects across all terms × variables within "
             "each model.  'all' corrects across everything.",
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

    demo_path = args.demo or default_behaviour_csv()
    out_dir   = args.out  or default_growth_curves_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("n_perms=%d, fdr_scope=%s, bs_df=%d",
             args.n_perms, args.fdr_scope, args.bs_df)

    # 1. Load data -----------------------------------------------------------
    wide_df = load_demographics_wide(demo_path)

    # 2. Build long-format analysis table ------------------------------------
    long_df = build_behavioural_long_df(wide_df)
    long_path = out_dir / "long_data.csv"
    long_df.to_csv(long_path, index=False)
    LOG.info("Long-format data saved to %s", long_path)

    # 3. Fit models (+ permutation tests) ------------------------------------
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
            "Random slopes saved to %s (%d / %d subject×variable entries reliable).",
            rs_path, n_reliable, len(random_slopes),
        )
    else:
        LOG.info("No random slopes extracted.")

    # 4. Plots ---------------------------------------------------------------
    if not args.no_plots:
        generate_all_plots(long_df, out_dir)

    LOG.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()

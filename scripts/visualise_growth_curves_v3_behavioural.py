#!/usr/bin/env python
"""visualise_growth_curves_v3_behavioural.py
============================================
Post-hoc visualisation of growth-curve model results for behavioural outcomes.

Reads the saved ``model_results.csv`` produced by
``run_growth_curves_with_random_slopes_behavioural.py`` and generates
publication-ready summary figures:

1. **Summary heatmap** – behavioural variables x ALL model terms, cells
   coloured by z-statistic with significance annotations.
2. **Forest plots** – effect size +/- 95% CI for the main group interaction
   terms across all outcome variables.
3. **Trajectory plots** – individual and group-mean trajectories at T1/T2/T3
   for each outcome, coloured by group (ASD / TD).

Usage
-----
    python visualise_growth_curves_v3_behavioural.py              # defaults
    python visualise_growth_curves_v3_behavioural.py --model M2_LMM
    python visualise_growth_curves_v3_behavioural.py --help
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

try:
    from rs_paths import rs_project_root
except ImportError:
    def rs_project_root() -> Path:  # type: ignore[misc]
        return Path(__file__).resolve().parents[1]

def default_growth_curves_output_dir() -> Path:
    return rs_project_root() / "reports" / "growth_curves_behavioural"

LOG = logging.getLogger("vis_growth")

# ---------------------------------------------------------------------------
#  Variable display helpers
# ---------------------------------------------------------------------------

DISPLAY_NAMES: Dict[str, str] = {
    # SDQ
    "sdq_total_p":         "SDQ Total",
    "sdq_emotional_p":     "SDQ Emotional",
    "sdq_conduct_p":       "SDQ Conduct",
    "sdq_hyperactivity_p": "SDQ Hyperactivity",
    "sdq_peer_p":          "SDQ Peer Problems",
    "sdq_prosocial_p":     "SDQ Prosocial",
    # Social
    "srs_rawscore":        "SRS Raw Score",
    # Sensory / Behaviour
    "rbs_total":           "RBS Total",
    "ssp_total":           "SSP Total",
    # PRL
    "prl_perE_prop":       "PRL Persev. Errors",
    "prl_WS":              "PRL Win-Stay",
    "prl_LS":              "PRL Lose-Shift",
    # QoL / Adaptive
    "ashq_total":          "ASHQ Total",
    "tas_total":           "TAS Total",
    "tas_identify":        "TAS Identify",
    "tas_describe":        "TAS Describe",
    "tas_external":        "TAS External",
}

REGION_ORDER: List[str] = [
    # Symptoms
    "sdq_total_p", "sdq_emotional_p", "sdq_conduct_p",
    "sdq_hyperactivity_p", "sdq_peer_p", "sdq_prosocial_p",
    "srs_rawscore",
    # Sensory / Behaviour
    "rbs_total", "ssp_total",
    "prl_perE_prop", "prl_WS", "prl_LS",
    # QoL / Adaptive
    "ashq_total",
    "tas_total", "tas_identify", "tas_describe", "tas_external",
]

SECTION_LABELS = {
    0:  "Symptoms",
    7:  "Sensory / Behaviour",
    12: "QoL / Adaptive",
}

# Approximate MNI centroid coordinates for MDTB-derived cerebellar networks.
_CEREB_MNI_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
    "Cereb_Motor":       (  0, -55, -25),
    "Cereb_Somatomotor": (  0, -60, -32),
    "Cereb_Attention":   ( 28, -72, -32),
    "Cereb_Executive":   ( 32, -68, -42),
    "Cereb_DMN":         (-15, -72, -42),
    "Cereb_Language":    ( 42, -63, -36),
    "Cereb_Limbic":      (  0, -48, -22),
}

_CEREB_SPHERE_RADIUS_MM = 8


def _display(region: str) -> str:
    return DISPLAY_NAMES.get(region, region)


def _ordered_regions(available: set) -> List[str]:
    ordered = [r for r in REGION_ORDER if r in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def _sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _safe_xerr(est: float, ci_lo: float, ci_hi: float):
    """Return (lo_err, hi_err) for errorbar, clamped to >= 0."""
    lo = max(0.0, float(est) - float(ci_lo)) if (
        np.isfinite(ci_lo) and np.isfinite(est)) else 0.0
    hi = max(0.0, float(ci_hi) - float(est)) if (
        np.isfinite(ci_hi) and np.isfinite(est)) else 0.0
    return lo, hi


# ---------------------------------------------------------------------------
#  Term classification helpers
# ---------------------------------------------------------------------------

_ALL_TERM_PATTERNS: List[Tuple[str, str]] = [
    (r"^Intercept$",                                              "Intercept"),
    (r"^C\(group\)\[T\.\w+\]$",                                  "Group (ASD)"),
    (r"^C\(sex\)\[T\.\w+\]$",                                    "Sex"),
    (r"^sex$",                                                    "Sex"),
    (r"^age$",                                                    "Age"),
    (r"^baseline_age$",                                           "Baseline Age"),
    (r"^time_elapsed$",                                           "Time"),
    (r"^bs\(age[^)]*\)\[\d+\]$",                                 "Age (spline)"),
    (r"^bs\(baseline_age[^)]*\)\[\d+\]$",                        "Baseline Age (spline)"),
    (r"^bs\(time_elapsed[^)]*\)\[\d+\]$",                        "Time (spline)"),
    (r"^age:time_elapsed$|^time_elapsed:age$",                    "Age × Time"),
    (r"^baseline_age:time_elapsed$|^time_elapsed:baseline_age$",  "Baseline Age × Time"),
    (r"^bs\(age[^)]*\)\[\d+\]:bs\(time_elapsed[^)]*\)\[\d+\]$",  "Age(spl) × Time(spl)"),
    (r"^C\(sex\)\[T\.\w+\]:age$|^age:C\(sex\)\[T\.\w+\]$",               "Age × Sex"),
    (r"^C\(sex\)\[T\.\w+\]:baseline_age$|^baseline_age:C\(sex\)\[T\.\w+\]$",
                                                                  "Baseline Age × Sex"),
    (r"^C\(sex\)\[T\.\w+\]:time_elapsed$|^time_elapsed:C\(sex\)\[T\.\w+\]$",
                                                                  "Time × Sex"),
    (r"^bs\(age[^)]*\)\[\d+\]:C\(sex\)\[T\.\w+\]$"
     r"|^C\(sex\)\[T\.\w+\]:bs\(age[^)]*\)\[\d+\]$",            "Age(spline) × Sex"),
    (r"^bs\(time_elapsed[^)]*\)\[\d+\]:C\(sex\)\[T\.\w+\]$"
     r"|^C\(sex\)\[T\.\w+\]:bs\(time_elapsed[^)]*\)\[\d+\]$",   "Time(spline) × Sex"),
    (r".*C\(sex\).*age.*time.*|.*time.*age.*C\(sex\).*",         "Time × Age × Sex"),
    (r"^C\(group\)\[T\.\w+\]:age$|^age:C\(group\)\[T\.\w+\]$",           "Age × Group"),
    (r"^C\(group\)\[T\.\w+\]:baseline_age$|^baseline_age:C\(group\)\[T\.\w+\]$",
                                                                  "Baseline Age × Group"),
    (r"^C\(group\)\[T\.\w+\]:time_elapsed$|^time_elapsed:C\(group\)\[T\.\w+\]$",
                                                                  "Time × Group"),
    (r"^bs\(age[^)]*\)\[\d+\]:C\(group\)\[T\.\w+\]$"
     r"|^C\(group\)\[T\.\w+\]:bs\(age[^)]*\)\[\d+\]$",          "Age(spline) × Group"),
    (r"^bs\(time_elapsed[^)]*\)\[\d+\]:C\(group\)\[T\.\w+\]$"
     r"|^C\(group\)\[T\.\w+\]:bs\(time_elapsed[^)]*\)\[\d+\]$", "Time(spline) × Group"),
    (r".*time_elapsed.*baseline_age.*C\(group\).*"
     r"|.*C\(group\).*baseline_age.*time_elapsed.*",             "Time × Age × Group"),
    (r".*bs\(time_elapsed[^)]*\)\[\d+\].*bs\(age[^)]*\)\[\d+\].*C\(group\).*"
     r"|.*C\(group\).*bs\(age[^)]*\)\[\d+\].*bs\(time_elapsed[^)]*\)\[\d+\].*",
                                                                  "Time(spl) × Age(spl) × Group"),
    (r".*C\(sex\).*C\(group\).*|.*C\(group\).*C\(sex\).*",       "Sex × Group"),
    # Nuisance regressors – consolidated so they never produce per-site
    # blank columns in the heatmap.
    (r"^C\(site\)\[T\.\w+\]$",                                    "Site"),
    # Variance / covariance components emitted by statsmodels MixedLM.
    # Intercept-only models emit "Group Var"; random-slope models emit
    # "Intercept Var" (plus slope and covariance terms below).
    (r"^(?:Group Var|Intercept Var)$",                            "RE: Intercept Var"),
    # Any remaining "X Var" term is the random-slope variance for predictor X
    # (e.g. "age Var" in M1_LMM, "time_elapsed Var" in M2_LMM).
    (r"\w[\w ]*\bVar\b",                                          "RE: Slope Var"),
    # Cross-term between intercept and slope.
    (r"\bCov\b",                                                  "RE: Slope Cov"),
    # Residual variance.
    (r"^Residual$",                                               "RE: Residual"),
]

_GROUP_TERM_LABELS = {
    "Group (ASD)", "Age × Group", "Baseline Age × Group",
    "Time × Group", "Age(spline) × Group", "Time(spline) × Group",
    "Time × Age × Group", "Time(spl) × Age(spl) × Group", "Sex × Group",
}


def classify_group_term(term: str) -> Optional[str]:
    """Return a label if *term* is a group-related term, else None."""
    label = _simplify_term(term)
    return label if label in _GROUP_TERM_LABELS else None


def _simplify_term(term: str) -> str:
    """Map a raw model term string to a concise, human-readable label."""
    for pat, label in _ALL_TERM_PATTERNS:
        if re.search(pat, term):
            return label
    simplified = term
    simplified = re.sub(r"C\((\w+)\)\[T\.(\w+)\]", r"\1[\2]", simplified)
    simplified = re.sub(r"bs\((\w+)[^)]*\)\[\d+\]", r"\1(spl)", simplified)
    return simplified


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["term_label"] = df["term"].apply(_simplify_term)
    df["is_group_term"] = df["term"].apply(lambda t: classify_group_term(t) is not None)
    return df


# ============================================================================
#  Shared heatmap helper
# ============================================================================

def _prepare_heatmap_data(
    mdf: pd.DataFrame,
    p_col: str,
    group_terms_only: bool = False,
):
    if group_terms_only:
        mdf = mdf[mdf["is_group_term"]]

    agg_rows = []
    for (net, tl), grp in mdf.groupby(["variable", "term_label"]):
        if len(grp) == 1:
            agg_rows.append(grp.iloc[0])
        else:
            best = grp.loc[grp[p_col].idxmin()] if grp[p_col].notna().any() else grp.iloc[0]
            agg_rows.append(best)
    mdf_agg = pd.DataFrame(agg_rows)

    regions = _ordered_regions(set(mdf_agg["variable"]))
    terms   = sorted(mdf_agg["term_label"].unique())

    z_mat    = np.full((len(regions), len(terms)), np.nan)
    star_mat = np.empty((len(regions), len(terms)), dtype=object)
    star_mat[:] = ""
    # True wherever a row exists in the results, even when z is NaN
    # (boundary / degenerate SE).  Distinguishes "no data" from "z undefined".
    has_data = np.zeros((len(regions), len(terms)), dtype=bool)

    for i, reg in enumerate(regions):
        for j, t in enumerate(terms):
            row = mdf_agg[(mdf_agg["variable"] == reg) & (mdf_agg["term_label"] == t)]
            if row.empty:
                continue
            row = row.iloc[0]
            z_mat[i, j]    = row["z"]
            star_mat[i, j] = _sig_star(row[p_col])
            has_data[i, j] = True

    return regions, terms, z_mat, star_mat, has_data


def _render_heatmap_axes(ax, fig, regions, terms, z_mat, star_mat,
                         title="", show_yticklabels=True, cell_fontsize=7,
                         has_data=None):
    vmax = np.nanmax(np.abs(z_mat)) if np.any(np.isfinite(z_mat)) else 3
    vmax = max(vmax, 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(z_mat, aspect="auto", cmap="RdBu_r", norm=norm)

    for i in range(len(regions)):
        for j in range(len(terms)):
            z_val = z_mat[i, j]
            star  = star_mat[i, j]
            if np.isnan(z_val):
                # Show '~' when the row exists but z is undefined (boundary /
                # degenerate SE — e.g. random-slope variance hitting zero).
                if has_data is not None and has_data[i, j]:
                    ax.text(j, i, "~", ha="center", va="center",
                            fontsize=cell_fontsize, color="#AAAAAA")
                continue
            txt    = f"{z_val:.1f}{star}" if star else f"{z_val:.1f}"
            colour = "white" if abs(z_val) > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=cell_fontsize, color=colour,
                    fontweight="bold" if star else "normal")

    # Draw section separators and annotate section labels in the left margin.
    sec_items = sorted(SECTION_LABELS.items())  # [(0, "Cortical"), (7, "Subcortical"), ...]
    for k, (sec_start, sec_label) in enumerate(sec_items):
        if 0 < sec_start < len(regions):
            ax.axhline(sec_start - 0.5, color="grey", linewidth=0.8, linestyle="--")
        # Compute the midpoint row of this section for the label.
        sec_end = sec_items[k + 1][0] if k + 1 < len(sec_items) else len(regions)
        mid_row = (sec_start + min(sec_end, len(regions)) - 1) / 2
        if show_yticklabels and sec_start < len(regions):
            ax.text(
                -0.5, mid_row, sec_label,
                ha="right", va="center", fontsize=8,
                color="grey", style="italic",
                transform=ax.get_yaxis_transform(),
            )

    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(regions)))
    if show_yticklabels:
        ax.set_yticklabels([_display(r) for r in regions], fontsize=9)
    else:
        ax.set_yticklabels([])

    if title:
        ax.set_title(title, fontsize=11)

    return im


# ============================================================================
#  1. Standalone heatmap figure
# ============================================================================

def plot_heatmap(
    results: pd.DataFrame,
    models: List[str],
    output_dir: Path,
    alpha: float = 0.05,
    use_perm: bool = False,
    group_terms_only: bool = False,
) -> None:
    p_col = "p_perm_fdr" if (use_perm and "p_perm_fdr" in results.columns) else "p_fdr"

    for model in models:
        mdf = results[results["model"] == model].copy()
        if mdf.empty:
            LOG.info("No rows for model %s – skipping heatmap.", model)
            continue

        regions, terms, z_mat, star_mat, has_data = _prepare_heatmap_data(
            mdf, p_col, group_terms_only=group_terms_only
        )
        if not regions or not terms:
            continue

        fig_height = max(5, 0.4 * len(regions) + 1.5)
        fig_width  = max(4, 1.1 * len(terms) + 3)
        fig, ax    = plt.subplots(figsize=(fig_width, fig_height))

        p_label  = "perm-FDR" if use_perm else "FDR"
        term_str = "group terms only" if group_terms_only else "all terms"
        title    = (f"{model} — Effects ({term_str})\n"
                    f"(* p<.05  ** p<.01  *** p<.001;  {p_label} corrected)")

        im = _render_heatmap_axes(ax, fig, regions, terms, z_mat, star_mat,
                                  title=title, has_data=has_data)
        fig.colorbar(im, ax=ax, shrink=0.7, label="z-statistic")

        fig.tight_layout()
        suffix = "_group" if group_terms_only else "_all"
        fname  = output_dir / f"heatmap{suffix}_{model}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved %s", fname)


# ============================================================================
#  2. Forest plots
# ============================================================================

def plot_forest(
    results: pd.DataFrame,
    models: List[str],
    output_dir: Path,
    target_terms: Optional[List[str]] = None,
    alpha: float = 0.05,
    use_perm: bool = False,
) -> None:
    p_col = "p_perm_fdr" if (use_perm and "p_perm_fdr" in results.columns) else "p_fdr"

    if target_terms is None:
        grp = results[results["is_group_term"]].copy()
        target_terms = (
            grp.groupby("term")["variable"]
            .nunique()
            .sort_values(ascending=False)
            .head(4)
            .index.tolist()
        )

    for model in models:
        mdf = results[results["model"] == model].copy()
        terms_in_model = [t for t in target_terms if t in mdf["term"].values]
        if not terms_in_model:
            continue

        n_panels = len(terms_in_model)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(5 * n_panels, max(5, 0.35 * 22 + 1)),
            sharey=True,
        )
        if n_panels == 1:
            axes = [axes]

        regions = _ordered_regions(set(mdf["variable"]))
        y_pos   = np.arange(len(regions))

        for ax, term in zip(axes, terms_in_model):
            tdf = mdf[mdf["term"] == term].set_index("variable")

            estimates, ci_low, ci_high, colours = [], [], [], []
            for reg in regions:
                if reg in tdf.index:
                    row = tdf.loc[reg]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    estimates.append(row["estimate"])
                    ci_low.append(row["ci_lower"])
                    ci_high.append(row["ci_upper"])
                    sig = row[p_col] < alpha if pd.notna(row[p_col]) else False
                    colours.append("#D32F2F" if sig else "#90A4AE")
                else:
                    estimates.append(np.nan)
                    ci_low.append(np.nan)
                    ci_high.append(np.nan)
                    colours.append("#E0E0E0")

            estimates = np.array(estimates, dtype=float)
            ci_low    = np.array(ci_low, dtype=float)
            ci_high   = np.array(ci_high, dtype=float)
            valid = np.isfinite(estimates)

            for idx in np.where(valid)[0]:
                lo, hi = _safe_xerr(estimates[idx], ci_low[idx], ci_high[idx])
                ax.errorbar(
                    estimates[idx], y_pos[idx],
                    xerr=[[lo], [hi]],
                    fmt="o", color=colours[idx], markersize=6,
                    elinewidth=1.5, capsize=3,
                )

            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
            for sec_idx in SECTION_LABELS:
                if 0 < sec_idx < len(regions):
                    ax.axhline(sec_idx - 0.5, color="#BDBDBD",
                               linewidth=0.6, linestyle=":")

            ax.set_yticks(y_pos)
            ax.set_yticklabels([_display(r) for r in regions], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Estimate (β)", fontsize=10)
            ax.set_title(_simplify_term(term), fontsize=10, fontweight="bold")
            ax.tick_params(axis="x", labelsize=8)

        legend_elements = [
            Patch(facecolor="#D32F2F", label=f"Significant ({p_col.replace('_', ' ')} < {alpha})"),
            Patch(facecolor="#90A4AE", label="Not significant"),
        ]
        axes[-1].legend(handles=legend_elements, loc="lower right",
                        fontsize=8, frameon=True, framealpha=0.9)

        fig.suptitle(f"{model} — Group interaction effects (ASD vs TD)",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        fname = output_dir / f"forest_group_{model}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved %s", fname)


# ============================================================================
#  3. Brain surface / glass-brain maps
# ============================================================================

def _build_yeo7_stat_map(region_stats: Dict[str, float]) -> Optional[object]:
    try:
        from nilearn import datasets
        import nibabel as nib
    except ImportError:
        LOG.warning("nilearn/nibabel not available – skipping cortical brain maps.")
        return None

    atlas    = datasets.fetch_atlas_yeo_2011()
    yeo_img  = nib.load(atlas["maps"]) if isinstance(atlas["maps"], (str, Path)) else atlas["maps"]
    yeo_data = np.asarray(yeo_img.dataobj, dtype=float)

    YEO_LABEL_TO_NETWORK = {1: "Vis", 2: "SomMot", 3: "DorsAttn",
                             4: "SalVentAttn", 5: "Limbic", 6: "Cont", 7: "Default"}

    stat_data = np.zeros_like(yeo_data, dtype=float)
    for label_int, net_name in YEO_LABEL_TO_NETWORK.items():
        if net_name in region_stats:
            stat_data[yeo_data == label_int] = region_stats[net_name]

    return nib.Nifti1Image(stat_data, yeo_img.affine, yeo_img.header)


_HO_SUBCORT_MAP: Dict[str, str] = {
    "Left Accumbens":   "Accumbens",  "Right Accumbens":   "Accumbens",
    "Left Amygdala":    "Amygdala",   "Right Amygdala":    "Amygdala",
    "Left Caudate":     "Caudate",    "Right Caudate":     "Caudate",
    "Left Hippocampus": "Hippocampus","Right Hippocampus": "Hippocampus",
    "Left Pallidum":    "Pallidum",   "Right Pallidum":    "Pallidum",
    "Left Putamen":     "Putamen",    "Right Putamen":     "Putamen",
    "Left Thalamus":    "Thalamus",   "Right Thalamus":    "Thalamus",
    "Brain-Stem":       "Brainstem",
}


def _build_subcortical_stat_map(region_stats: Dict[str, float]) -> Optional[object]:
    try:
        from nilearn import datasets
        import nibabel as nib
    except ImportError:
        return None

    atlas      = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    atlas_img  = nib.load(atlas["maps"]) if isinstance(atlas["maps"], (str, Path)) else atlas["maps"]
    atlas_data = np.asarray(atlas_img.dataobj, dtype=float)
    labels     = atlas["labels"]

    stat_data = np.zeros_like(atlas_data, dtype=float)
    for idx, lbl in enumerate(labels):
        our_name = _HO_SUBCORT_MAP.get(lbl)
        if our_name and our_name in region_stats:
            stat_data[atlas_data == idx] = region_stats[our_name]

    return nib.Nifti1Image(stat_data, atlas_img.affine, atlas_img.header)


def _build_cerebellar_stat_map(region_stats: Dict[str, float]) -> Optional[object]:
    try:
        import nibabel as nib
        from nilearn import image
        from nilearn.image import new_img_like
    except ImportError:
        LOG.warning("nilearn/nibabel not available – skipping cerebellar brain maps.")
        return None

    try:
        from nilearn.datasets import load_mni152_template
        ref_img = load_mni152_template(resolution=2)
    except Exception:
        try:
            from nilearn.datasets import load_mni152_template
            ref_img = load_mni152_template()
        except Exception as exc:
            LOG.warning("Could not load MNI template: %s", exc)
            return None

    affine   = ref_img.affine
    shape    = ref_img.shape[:3]
    stat_vol = np.zeros(shape, dtype=float)
    inv_affine = np.linalg.inv(affine)

    for net_name, mni_xyz in _CEREB_MNI_CENTROIDS.items():
        if net_name not in region_stats:
            continue
        stat_val = region_stats[net_name]
        if stat_val == 0.0:
            continue

        mni_h    = np.array([*mni_xyz, 1.0])
        vox_f    = inv_affine @ mni_h
        cx, cy, cz = int(round(vox_f[0])), int(round(vox_f[1])), int(round(vox_f[2]))

        vox_size = float(np.abs(affine[0, 0]))
        r_vox    = int(np.ceil(_CEREB_SPHERE_RADIUS_MM / vox_size))

        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dz in range(-r_vox, r_vox + 1):
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                        continue
                    dist_mm = np.sqrt((dx ** 2 + dy ** 2 + dz ** 2)) * vox_size
                    if dist_mm <= _CEREB_SPHERE_RADIUS_MM:
                        weight = np.exp(-0.5 * (dist_mm / (_CEREB_SPHERE_RADIUS_MM / 2)) ** 2)
                        if abs(stat_val * weight) > abs(stat_vol[nx, ny, nz]):
                            stat_vol[nx, ny, nz] = stat_val * weight

    return nib.Nifti1Image(stat_vol, affine)


def plot_brain_maps(
    results: pd.DataFrame,
    models: List[str],
    output_dir: Path,
    target_terms: Optional[List[str]] = None,
    alpha: float = 0.05,
    use_perm: bool = False,
) -> None:
    try:
        from nilearn import plotting
    except ImportError:
        LOG.warning("nilearn not available – skipping brain maps.")
        return

    p_col = "p_perm_fdr" if (use_perm and "p_perm_fdr" in results.columns) else "p_fdr"

    if target_terms is None:
        grp = results[results["is_group_term"]].copy()
        target_terms = (
            grp.groupby("term")["network"]
            .nunique()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )

    cortical_nets  = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}
    subcort_nets   = {"Accumbens", "Amygdala", "Caudate", "Hippocampus",
                      "Pallidum", "Putamen", "Thalamus", "Brainstem"}
    cerebellar_nets = set(_CEREB_MNI_CENTROIDS.keys())

    for model in models:
        mdf = results[results["model"] == model].copy()

        for term in target_terms:
            tdf = mdf[mdf["term"] == term].copy()
            if tdf.empty:
                continue

            best_rows = []
            for net, grp in tdf.groupby("network"):
                best = grp.loc[grp[p_col].idxmin()] if grp[p_col].notna().any() else grp.iloc[0]
                best_rows.append(best)
            tdf = pd.DataFrame(best_rows)

            all_stats      = dict(zip(tdf["network"], tdf["z"]))
            cortical_stats = {k: v for k, v in all_stats.items() if k in cortical_nets}
            subcort_stats  = {k: v for k, v in all_stats.items() if k in subcort_nets}
            cereb_stats    = {k: v for k, v in all_stats.items() if k in cerebellar_nets}

            cortical_img  = _build_yeo7_stat_map(cortical_stats)    if cortical_stats  else None
            subcort_img   = _build_subcortical_stat_map(subcort_stats) if subcort_stats else None
            cerebellar_img = _build_cerebellar_stat_map(cereb_stats) if cereb_stats     else None

            if cortical_img is not None:
                vmax = max(abs(v) for v in cortical_stats.values())
                vmax = max(vmax, 0.5)
                fig = plt.figure(figsize=(14, 4))
                plotting.plot_stat_map(
                    cortical_img,
                    display_mode="ortho",
                    cut_coords=[0, -20, 30],
                    cmap="RdBu_r",
                    colorbar=True,
                    vmax=vmax,
                    symmetric_cbar=True,
                    title=f"{model}: {_simplify_term(term)}\n(Cortical Yeo-7, z-statistic)",
                    figure=fig,
                )
                fname = output_dir / f"brain_cortical_{model}_{_safe_fname(term)}.png"
                fig.savefig(fname, dpi=200, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved %s", fname)

            if subcort_img is not None:
                vmax = max(abs(v) for v in subcort_stats.values())
                vmax = max(vmax, 0.5)
                fig = plt.figure(figsize=(14, 4))
                plotting.plot_glass_brain(
                    subcort_img,
                    display_mode="ortho",
                    cmap="RdBu_r",
                    colorbar=True,
                    vmax=vmax,
                    symmetric_cbar=True,
                    title=f"{model}: {_simplify_term(term)}\n(Subcortical, z-statistic)",
                    figure=fig,
                    plot_abs=False,
                )
                fname = output_dir / f"brain_subcort_{model}_{_safe_fname(term)}.png"
                fig.savefig(fname, dpi=200, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved %s", fname)

            if cerebellar_img is not None:
                vmax = max(abs(v) for v in cereb_stats.values())
                vmax = max(vmax, 0.5)
                fig = plt.figure(figsize=(14, 4))
                plotting.plot_glass_brain(
                    cerebellar_img,
                    display_mode="ortho",
                    cmap="RdBu_r",
                    colorbar=True,
                    vmax=vmax,
                    symmetric_cbar=True,
                    title=f"{model}: {_simplify_term(term)}\n(Cerebellar networks, z-statistic)",
                    figure=fig,
                    plot_abs=False,
                )
                fname = output_dir / f"brain_cerebellar_{model}_{_safe_fname(term)}.png"
                fig.savefig(fname, dpi=200, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved %s", fname)

            imgs_to_merge = [i for i in [cortical_img, subcort_img, cerebellar_img]
                             if i is not None]
            if len(imgs_to_merge) >= 2:
                try:
                    from nilearn import image as nli_image
                    merged = imgs_to_merge[0]
                    for extra in imgs_to_merge[1:]:
                        merged = nli_image.math_img(
                            "np.where(np.abs(a) >= np.abs(b), a, b)",
                            a=merged, b=extra,
                        )
                    vmax = max(abs(v) for v in all_stats.values()) if all_stats else 3
                    vmax = max(vmax, 0.5)
                    fig = plt.figure(figsize=(14, 5))
                    plotting.plot_glass_brain(
                        merged,
                        display_mode="lyrz",
                        cmap="RdBu_r",
                        colorbar=True,
                        vmax=vmax,
                        symmetric_cbar=True,
                        title=(f"{model}: {_simplify_term(term)}"
                               "  (z-statistic, all regions)"),
                        figure=fig,
                        plot_abs=False,
                    )
                    fname = output_dir / f"brain_combined_{model}_{_safe_fname(term)}.png"
                    fig.savefig(fname, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    LOG.info("Saved %s", fname)
                except Exception as e:
                    LOG.warning("Combined brain map failed: %s", e)


def _safe_fname(term: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", term)[:60]


# ============================================================================
#  4. Cerebellar bar-chart (dot-plot) figure
# ============================================================================

def plot_cerebellar_bar(
    results: pd.DataFrame,
    models: List[str],
    output_dir: Path,
    alpha: float = 0.05,
    use_perm: bool = False,
) -> None:
    p_col       = "p_perm_fdr" if (use_perm and "p_perm_fdr" in results.columns) else "p_fdr"
    cereb_nets  = [r for r in REGION_ORDER if r.startswith("Cereb_")]

    for model in models:
        mdf = results[results["model"] == model].copy()

        cereb_mask  = mdf["network"].isin(cereb_nets)
        terms_avail = sorted(mdf.loc[cereb_mask, "term_label"].unique())
        if not terms_avail:
            LOG.info("No cerebellar rows for model %s – skipping cerebellar bar plot.", model)
            continue

        agg_rows = []
        for (net, tl), grp in mdf[cereb_mask].groupby(["network", "term_label"]):
            best = grp.loc[grp[p_col].idxmin()] if grp[p_col].notna().any() else grp.iloc[0]
            agg_rows.append(best)
        mdf_agg = pd.DataFrame(agg_rows)

        n_terms  = len(terms_avail)
        n_cols   = min(n_terms, 4)
        n_rows   = int(np.ceil(n_terms / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )

        available_cereb = [r for r in cereb_nets if r in mdf_agg["network"].values]

        for ax_idx, term in enumerate(terms_avail):
            ax  = axes[ax_idx // n_cols][ax_idx % n_cols]
            tdf = mdf_agg[mdf_agg["term_label"] == term].set_index("network")

            zvals, cis_lo, cis_hi, colours, labels_y = [], [], [], [], []
            for net in available_cereb:
                if net not in tdf.index:
                    continue
                row   = tdf.loc[net]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                zvals.append(row["z"])
                cis_lo.append(row.get("ci_lower", np.nan))
                cis_hi.append(row.get("ci_upper", np.nan))
                sig    = row[p_col] < alpha if pd.notna(row[p_col]) else False
                colours.append("#D32F2F" if sig else "#90A4AE")
                labels_y.append(_display(net))

            if not zvals:
                ax.set_visible(False)
                continue

            zvals  = np.array(zvals, dtype=float)
            cis_lo = np.array(cis_lo, dtype=float)
            cis_hi = np.array(cis_hi, dtype=float)
            y_pos  = np.arange(len(zvals))

            for i in range(len(zvals)):
                lo_e, hi_e = _safe_xerr(zvals[i], cis_lo[i], cis_hi[i])
                xerr_args = {"xerr": [[lo_e], [hi_e]], "elinewidth": 1.5, "capsize": 3}
                ax.errorbar(zvals[i], y_pos[i], fmt="o",
                            color=colours[i], markersize=8, **xerr_args)

            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_y, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("z-statistic", fontsize=9)
            ax.set_title(term, fontsize=10, fontweight="bold")
            ax.tick_params(axis="x", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        for ax_idx in range(n_terms, n_rows * n_cols):
            axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

        legend_elements = [
            Patch(facecolor="#D32F2F", label=f"Significant ({p_col.replace('_', ' ')} < {alpha})"),
            Patch(facecolor="#90A4AE", label="Not significant"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=2, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

        p_label = "perm-FDR" if use_perm else "FDR"
        fig.suptitle(
            f"{model} — Cerebellar network effects (ASD vs TD)\n"
            f"(* p < .05, ** p < .01, *** p < .001;  {p_label} corrected)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fname = output_dir / f"cerebellar_bar_{model}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved %s", fname)


# ============================================================================
#  5. Summary figure — full heatmap (left) + forest plot (right)
# ============================================================================

def plot_summary_figure(
    results: pd.DataFrame,
    model: str,
    output_dir: Path,
    alpha: float = 0.05,
    use_perm: bool = False,
) -> None:
    p_col = "p_perm_fdr" if (use_perm and "p_perm_fdr" in results.columns) else "p_fdr"

    mdf = results[results["model"] == model].copy()
    if mdf.empty:
        return

    regions, terms, z_mat, star_mat, has_data = _prepare_heatmap_data(mdf, p_col,
                                                                       group_terms_only=False)
    if not regions or not terms:
        return

    forest_term_label = None
    for candidate in ["Time × Group", "Time(spline) × Group",
                      "Age × Group", "Age(spline) × Group", "Group (ASD)"]:
        if candidate in terms:
            forest_term_label = candidate
            break
    if forest_term_label is None:
        forest_term_label = terms[0]

    heat_w   = max(4, 1.1 * len(terms) + 3)
    forest_w = 5
    fig_w    = heat_w + forest_w + 1
    fig_h    = max(6, 0.4 * len(regions) + 2)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[heat_w, forest_w], wspace=0.08)

    ax_heat = fig.add_subplot(gs[0])
    p_label = "perm-FDR" if use_perm else "FDR"
    heat_title = (f"{model} — All terms\n"
                  f"(* p<.05  ** p<.01  *** p<.001;  {p_label} corrected)")
    im = _render_heatmap_axes(
        ax_heat, fig, regions, terms, z_mat, star_mat,
        title=heat_title, show_yticklabels=True,
        cell_fontsize=max(5, 8 - max(0, len(terms) - 8)),
        has_data=has_data,
    )
    fig.colorbar(im, ax=ax_heat, shrink=0.6, label="z-statistic", pad=0.02)

    ax_forest = fig.add_subplot(gs[1])

    agg_rows = []
    for (net, tl), grp in mdf.groupby(["variable", "term_label"]):
        best = grp.loc[grp[p_col].idxmin()] if grp[p_col].notna().any() else grp.iloc[0]
        agg_rows.append(best)
    mdf_agg = pd.DataFrame(agg_rows)

    fdf   = mdf_agg[mdf_agg["term_label"] == forest_term_label].set_index("variable")
    y_pos = np.arange(len(regions))

    for idx, reg in enumerate(regions):
        if reg not in fdf.index:
            continue
        row = fdf.loc[reg]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        est    = row.get("estimate", np.nan)
        ci_lo  = row.get("ci_lower", np.nan)
        ci_hi  = row.get("ci_upper", np.nan)
        if not np.isfinite(est):
            continue
        sig    = row[p_col] < alpha if pd.notna(row[p_col]) else False
        colour = "#D32F2F" if sig else "#90A4AE"
        lo_e, hi_e = _safe_xerr(est, ci_lo, ci_hi)
        xerr_kw = {"xerr": [[lo_e], [hi_e]], "elinewidth": 1.5, "capsize": 3}
        ax_forest.errorbar(est, y_pos[idx], fmt="o",
                           color=colour, markersize=6, **xerr_kw)

    ax_forest.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
    for sec_idx in SECTION_LABELS:
        if 0 < sec_idx < len(regions):
            ax_forest.axhline(sec_idx - 0.5, color="#BDBDBD",
                               linewidth=0.6, linestyle=":")

    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels([])
    ax_forest.invert_yaxis()
    ax_forest.set_xlabel("Estimate (β) ± 95% CI", fontsize=10)
    ax_forest.set_title(forest_term_label, fontsize=11, fontweight="bold")
    ax_forest.tick_params(axis="x", labelsize=8)
    ax_forest.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Patch(facecolor="#D32F2F", label=f"Significant ({p_col.replace('_', ' ')} < {alpha})"),
        Patch(facecolor="#90A4AE", label="Not significant"),
    ]
    ax_forest.legend(handles=legend_elements, loc="lower right",
                     fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fname = output_dir / f"summary_{model}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", fname)


# ============================================================================
#  6. FC rate-of-change vs baseline age, by group
# ============================================================================

_GROUP_COLOURS = {"TD": "#1976D2", "ASD": "#D32F2F"}
_GROUP_FILL    = {"TD": "#90CAF9", "ASD": "#EF9A9A"}


def plot_fc_rate_by_age(
    results: pd.DataFrame,
    models: List[str],
    output_dir: Path,
    age_range: Optional[Tuple[float, float]] = None,
    n_age_points: int = 60,
    alpha: float = 0.05,
    use_perm: bool = False,
) -> None:
    """Line plot: predicted rate of FC change (per year) vs baseline age, per group."""
    p_col = "p_perm_fdr" if (use_perm and "p_perm_fdr" in results.columns) else "p_fdr"

    def _se_from_ci(row_: pd.Series) -> float:
        lo = row_.get("ci_lower", np.nan)
        hi = row_.get("ci_upper", np.nan)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            return (hi - lo) / (2 * 1.96)
        return 0.0

    for model in models:
        mdf = results[results["model"] == model].copy()
        if mdf.empty:
            continue

        has_age_mod = mdf["term_label"].isin(
            ["Baseline Age × Time", "Time × Age × Group"]
        ).any()

        regions = _ordered_regions(set(mdf["network"]))
        if not regions:
            continue

        if age_range is None:
            _lo, _hi = 6.0, 18.0
        else:
            _lo, _hi = age_range
        ages = np.linspace(_lo, _hi, n_age_points)

        n_reg  = len(regions)
        n_cols = min(n_reg, 4)
        n_rows = int(np.ceil(n_reg / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.8 * n_cols, 3.6 * n_rows),
            squeeze=False,
        )

        for reg_idx, region in enumerate(regions):
            ax  = axes[reg_idx // n_cols][reg_idx % n_cols]
            tdf = mdf[mdf["network"] == region]

            agg = {}
            for tl, grp in tdf.groupby("term_label"):
                best = (grp.loc[grp[p_col].idxmin()]
                        if grp[p_col].notna().any() else grp.iloc[0])
                agg[tl] = best

            def coef(label: str, col: str = "estimate") -> float:
                row_ = agg.get(label)
                return float(row_[col]) if row_ is not None and pd.notna(row_.get(col)) else 0.0

            def se(label: str) -> float:
                row_ = agg.get(label)
                return _se_from_ci(row_) if row_ is not None else 0.0

            b_t   = coef("Time")
            b_tg  = coef("Time × Group")
            b_at  = coef("Baseline Age × Time")
            b_atg = coef("Time × Age × Group")

            se_t   = se("Time")
            se_tg  = se("Time × Group")
            se_at  = se("Baseline Age × Time")
            se_atg = se("Time × Age × Group")

            slope_td  = b_t  +  b_at  * ages
            slope_asd = (b_t + b_tg) + (b_at + b_atg) * ages

            se_td_line  = np.sqrt(se_t**2  + (ages * se_at)**2)
            se_asd_line = np.sqrt(se_t**2 + se_tg**2 + (ages**2) * (se_at**2 + se_atg**2))

            sig_rows = tdf[tdf["term_label"].isin(
                ["Time × Group", "Time × Age × Group"]
            )]
            is_sig = (sig_rows[p_col] < alpha).any() if not sig_rows.empty else False

            for slope, se_line, group_key in [
                (slope_td,  se_td_line,  "TD"),
                (slope_asd, se_asd_line, "ASD"),
            ]:
                colour = _GROUP_COLOURS[group_key]
                fill_c = _GROUP_FILL[group_key]
                ax.plot(ages, slope, color=colour, linewidth=2,
                        label=group_key,
                        linestyle="-" if group_key == "TD" else "--")
                ax.fill_between(
                    ages, slope - se_line, slope + se_line,
                    color=fill_c, alpha=0.35,
                )

            ax.axhline(0, color="grey", linewidth=0.6, linestyle=":", zorder=0)
            ax.set_xlabel("Baseline Age (years)", fontsize=9)
            ax.set_ylabel("ΔFC / year", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

            region_label = _display(region)
            if is_sig:
                region_label += "  *"
            ax.set_title(region_label, fontsize=10,
                         fontweight="bold" if is_sig else "normal")

        for reg_idx in range(n_reg, n_rows * n_cols):
            axes[reg_idx // n_cols][reg_idx % n_cols].set_visible(False)

        handles = [
            plt.Line2D([0], [0], color=_GROUP_COLOURS["TD"],  linewidth=2,
                       linestyle="-",  label="TD"),
            plt.Line2D([0], [0], color=_GROUP_COLOURS["ASD"], linewidth=2,
                       linestyle="--", label="ASD"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

        age_note = ("Age-modulated slope shown" if has_age_mod
                    else "No age × time interaction in model — slopes are flat")
        p_label  = "perm-FDR" if use_perm else "FDR"
        suptitle_str = (
            f"{model} — Predicted rate of FC change vs baseline age\n"
            f"(* region with significant group interaction, {p_label} < {alpha})\n"
            f"{age_note}  |  shading ≈ ±1 SE (zero-covariance approximation)"
        )
        fig.suptitle(suptitle_str, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fname = output_dir / f"fc_rate_by_age_{model}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved %s", fname)


# ============================================================================
#  7. wDC trajectories: T1 → T3, by group
# ============================================================================

# Expected column names in the raw wDC data file.  Override via keyword args
# to ``plot_wdc_t1_t3`` if your file uses different names.
_WDC_DEFAULTS = dict(
    subject_col   = "subject",         # column name in long_data.csv
    timepoint_col = "timepoint",       # values expected: "T1", "T3" (others ignored)
    elapsed_col   = "time_elapsed",    # numeric; e.g. months or years since T1
    group_col     = "group",           # values expected: "ASD", "TD"
    network_col   = "network",         # must match REGION_ORDER labels
    wdc_col       = "wdc",             # lowercase in long_data.csv
)

# Fallback x-position for T3 when elapsed_col is absent / all-NaN
_T3_NOMINAL_ELAPSED = 2.0


def plot_wdc_t1_t3(
    raw_data: pd.DataFrame,
    output_dir: Path,
    alpha: float = 0.05,
    subject_col: str   = _WDC_DEFAULTS["subject_col"],
    timepoint_col: str = _WDC_DEFAULTS["timepoint_col"],
    elapsed_col: str   = _WDC_DEFAULTS["elapsed_col"],
    group_col: str     = _WDC_DEFAULTS["group_col"],
    network_col: str   = _WDC_DEFAULTS["network_col"],
    wdc_col: str       = _WDC_DEFAULTS["wdc_col"],
    jitter_seed: int   = 42,
) -> None:
    """Plot wDC at T1 and T3 for each brain network, split by group (ASD / TD).

    Layout
    ------
    One panel per network (following REGION_ORDER), arranged in a 4-column grid.
    Within each panel:

    * **Thin grey lines** connect each subject's T1 → T3 wDC values (paired
      trajectories).
    * **Coloured markers** show individual wDC values (circles = T1, squares = T3),
      jittered slightly on x to reduce overplotting.
    * **Thick coloured lines** show the group mean with a ± 1 SD ribbon.

    X-axis positioning
    ------------------
    T1 is anchored at x = 0.  T3 is placed at the median ``elapsed_col`` value
    across all subjects for that timepoint (or ``_T3_NOMINAL_ELAPSED`` = 2.0 if
    the elapsed column is absent or entirely NaN).

    Parameters
    ----------
    raw_data:
        Long-format DataFrame.  Must contain at minimum the columns named by
        the ``*_col`` arguments (see ``_WDC_DEFAULTS`` for defaults).
    output_dir:
        Directory where ``wdc_t1_t3.png`` will be written.
    alpha:
        Reserved for future significance annotation; not used currently.
    subject_col, timepoint_col, elapsed_col, group_col, network_col, wdc_col:
        Column-name overrides.
    jitter_seed:
        Random seed for reproducible x-jitter on individual data points.
    """
    rng = np.random.default_rng(jitter_seed)

    # ── Validate required columns ─────────────────────────────────────────────
    required = {subject_col, timepoint_col, group_col, network_col, wdc_col}
    missing  = required - set(raw_data.columns)
    if missing:
        LOG.error(
            "plot_wdc_t1_t3: raw data is missing required column(s): %s — skipping.",
            missing,
        )
        return

    has_elapsed = (
        elapsed_col in raw_data.columns
        and raw_data[elapsed_col].notna().any()
    )

    # ── Filter to T1 and T3 only ──────────────────────────────────────────────
    df = raw_data[raw_data[timepoint_col].isin(["T1", "T3"])].copy()
    if df.empty:
        LOG.warning("plot_wdc_t1_t3: no T1/T3 rows found in raw data — skipping.")
        return

    # Normalise group labels (upper-case, stripped) for consistent colour lookup
    df[group_col] = df[group_col].astype(str).str.upper().str.strip()

    # ── Determine x-position for T3 ───────────────────────────────────────────
    if has_elapsed:
        t3_x = float(df.loc[df[timepoint_col] == "T3", elapsed_col].median())
        if not np.isfinite(t3_x) or t3_x <= 0:
            t3_x = _T3_NOMINAL_ELAPSED
    else:
        t3_x = _T3_NOMINAL_ELAPSED

    tp_to_x = {"T1": 0.0, "T3": t3_x}
    df["_x"] = df[timepoint_col].map(tp_to_x)

    # ── Grid layout ───────────────────────────────────────────────────────────
    regions = _ordered_regions(set(df[network_col].unique()))
    n_reg   = len(regions)
    if n_reg == 0:
        LOG.warning("plot_wdc_t1_t3: no recognisable network labels — skipping.")
        return

    n_cols = min(n_reg, 4)
    n_rows = int(np.ceil(n_reg / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    groups_present = sorted(df[group_col].unique())

    for reg_idx, region in enumerate(regions):
        ax  = axes[reg_idx // n_cols][reg_idx % n_cols]
        rdf = df[df[network_col] == region].copy()

        if rdf.empty:
            ax.set_visible(False)
            continue

        # ── Per-subject paired trajectories (thin grey lines) ─────────────────
        for subj in rdf[subject_col].unique():
            sdf = rdf[rdf[subject_col] == subj].sort_values("_x")
            if len(sdf) < 2:
                continue
            ax.plot(
                sdf["_x"], sdf[wdc_col],
                color="#BDBDBD", linewidth=0.7, alpha=0.5, zorder=1,
            )

        # ── Individual scatter + group mean ± SD ──────────────────────────────
        for group_key in groups_present:
            colour = _GROUP_COLOURS.get(group_key, "#607D8B")
            fill_c = _GROUP_FILL.get(group_key,   "#B0BEC5")
            gdf    = rdf[rdf[group_col] == group_key]

            # Scatter: circles for T1, squares for T3
            for tp, marker in [("T1", "o"), ("T3", "s")]:
                tdf = gdf[gdf[timepoint_col] == tp]
                if tdf.empty:
                    continue
                x_nom  = tp_to_x[tp]
                # Small x-jitter (±5 % of the T1→T3 interval) to reduce overplot
                jitter = rng.uniform(-0.05 * t3_x, 0.05 * t3_x, size=len(tdf))
                ax.scatter(
                    x_nom + jitter, tdf[wdc_col],
                    color=colour, marker=marker,
                    s=28, alpha=0.55, linewidths=0, zorder=2,
                )

            # Group mean ± 1 SD ribbon
            xs, means, sds = [], [], []
            for tp in ["T1", "T3"]:
                vals = gdf[gdf[timepoint_col] == tp][wdc_col].dropna()
                if vals.empty:
                    continue
                xs.append(tp_to_x[tp])
                means.append(vals.mean())
                sds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

            if len(xs) >= 2:
                xs    = np.array(xs)
                means = np.array(means)
                sds   = np.array(sds)
                ax.plot(
                    xs, means,
                    color=colour, linewidth=2.5, zorder=3,
                    label=group_key,
                    linestyle="-" if group_key == "TD" else "--",
                )
                ax.fill_between(
                    xs, means - sds, means + sds,
                    color=fill_c, alpha=0.30, zorder=2,
                )

        # ── Axes decoration ───────────────────────────────────────────────────
        ax.set_xticks([0.0, t3_x])
        ax.set_xticklabels(["T1", "T3"], fontsize=9)
        ax.set_xlim(-0.15 * t3_x, 1.2 * t3_x)
        ax.set_xlabel(
            f"Time elapsed ({elapsed_col})" if has_elapsed else "Timepoint",
            fontsize=9,
        )
        ax.set_ylabel("wDC", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(_display(region), fontsize=10, fontweight="bold")

    # ── Hide surplus subplots ─────────────────────────────────────────────────
    for reg_idx in range(n_reg, n_rows * n_cols):
        axes[reg_idx // n_cols][reg_idx % n_cols].set_visible(False)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = []
    for group_key in groups_present:
        colour = _GROUP_COLOURS.get(group_key, "#607D8B")
        ls     = "-" if group_key == "TD" else "--"
        legend_handles.append(
            plt.Line2D([0], [0], color=colour, linewidth=2, linestyle=ls,
                       label=f"{group_key} mean ± SD")
        )
    legend_handles += [
        plt.Line2D([0], [0], color="#BDBDBD", linewidth=1,
                   label="Individual trajectory"),
        plt.Line2D([0], [0], color="none", marker="o", markerfacecolor="#888",
                   markersize=7, label="T1"),
        plt.Line2D([0], [0], color="none", marker="s", markerfacecolor="#888",
                   markersize=7, label="T3"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles),
        fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01),
    )

    tp_note = (f"T3 x-position = median elapsed time ({t3_x:.2f})"
               if has_elapsed else f"T3 x-position = nominal ({t3_x})")
    fig.suptitle(
        f"wDC at T1 and T3 by group\n({tp_note}; individual points jittered on x)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    fname = output_dir / "wdc_t1_t3.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", fname)


def load_raw_data(path: Path) -> Optional[pd.DataFrame]:
    """Load and minimally validate the raw wDC long-format data file."""
    if not path.exists():
        LOG.error("Raw data file not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        LOG.info("Loaded raw data: %d rows, %d cols from %s", *df.shape, path)
        return df
    except Exception as exc:
        LOG.error("Could not read raw data file %s: %s", path, exc)
        return None


# ============================================================================
#  CLI
# ============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Visualise growth-curve model results (reads model_results.csv).",
    )
    p.add_argument("--results", type=Path, default=None,
                   help="Path to model_results.csv (default: auto-detected).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory for figures (default: <results_dir>/vis).")
    p.add_argument("--model", nargs="*", default=None,
                   help="Model(s) to visualise (default: all).")
    p.add_argument("--use-perm", action="store_true",
                   help="Use permutation-FDR p-values instead of parametric FDR.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold (default: 0.05).")
    p.add_argument("--no-brain", action="store_true",
                   help="Skip brain surface/glass-brain maps (faster, no nilearn needed).")
    p.add_argument("--group-terms-only", action="store_true",
                   help="Heatmaps show only group-related terms (original behaviour).")
    p.add_argument("--age-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="Baseline-age axis range for FC-rate plot (default: 6 18).")
    # ── wDC T1/T3 plot ────────────────────────────────────────────────────────
    p.add_argument(
        "--raw-data", type=Path, default=None,
        help=("Path to long-format raw wDC CSV (default: long_data.csv alongside "
              "model_results.csv).  Columns used: subject, timepoint (T1/T3), "
              "time_elapsed, group, network, wdc.  All other columns and any "
              "timepoints other than T1/T3 are ignored.  Override column names "
              "with the --wdc-* flags below."),
    )
    p.add_argument("--wdc-subject-col",   default=_WDC_DEFAULTS["subject_col"],
                   metavar="COL", help="Subject-ID column (default: subject).")
    p.add_argument("--wdc-timepoint-col", default=_WDC_DEFAULTS["timepoint_col"],
                   metavar="COL", help="Timepoint column (default: timepoint).")
    p.add_argument("--wdc-elapsed-col",   default=_WDC_DEFAULTS["elapsed_col"],
                   metavar="COL", help="Time-elapsed column (default: time_elapsed).")
    p.add_argument("--wdc-group-col",     default=_WDC_DEFAULTS["group_col"],
                   metavar="COL", help="Group column (default: group).")
    p.add_argument("--wdc-network-col",   default=_WDC_DEFAULTS["network_col"],
                   metavar="COL", help="Network column (default: network).")
    p.add_argument("--wdc-col",           default=_WDC_DEFAULTS["wdc_col"],
                   metavar="COL", help="wDC value column (default: wdc).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    gc_dir        = default_growth_curves_output_dir()
    results_path  = args.results or (gc_dir / "model_results.csv")
    if not results_path.exists():
        LOG.error("Results file not found: %s", results_path)
        raise SystemExit(1)

    out_dir = args.out or (gc_dir / "vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    LOG.info("Loaded %d rows from %s", len(results), results_path)

    models = args.model or sorted(results["model"].unique())
    LOG.info("Models to visualise: %s", models)

    # 1. Heatmaps
    LOG.info("Generating heatmaps …")
    plot_heatmap(results, models, out_dir,
                 alpha=args.alpha, use_perm=args.use_perm,
                 group_terms_only=args.group_terms_only)

    # 2. Forest plots
    LOG.info("Generating forest plots …")
    plot_forest(results, models, out_dir,
                alpha=args.alpha, use_perm=args.use_perm)

    # 3. Summary figures
    LOG.info("Generating summary figures …")
    for m in models:
        plot_summary_figure(results, m, out_dir,
                            alpha=args.alpha, use_perm=args.use_perm)

    LOG.info("All figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
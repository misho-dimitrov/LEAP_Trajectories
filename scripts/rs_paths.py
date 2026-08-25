"""Shared path helpers for the RS workspace scripts.

Keeps default input/output paths consistent across scripts (e.g. estimate_FC.py and run_pls.py)
without hardcoding user-specific absolute paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def rs_project_root() -> Path:
    """Return the RS workspace root (the folder containing Data/, reports/, scripts/)."""

    return Path(__file__).resolve().parents[1]


def default_fc_output_dir() -> Path:
    return rs_project_root() / "reports" / "fc"


def default_master_wdc_csv(output_dir: Optional[Path] = None) -> Path:
    return (output_dir or default_fc_output_dir()) / "master_wdc.csv"


def default_qc_csv(timepoint: int = 1) -> Path:
    """Return the QC file for *timepoint*, trying .tsv then .csv."""
    root = rs_project_root()

    # T3 uses a different naming convention
    if int(timepoint) == 3:
        p = root / "LEAP3_QC_raport_all_sites.csv"
        if p.exists():
            return p
        # Fallback
        return p

    stem = f"LEAP{int(timepoint)}_RS-fMRI_QC_final"
    for ext in (".tsv", ".csv"):
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    # Fallback (will fail later with a clear FileNotFoundError)
    return root / f"{stem}.csv"


def default_pls_output_dir() -> Path:
    return rs_project_root() / "reports" / "pls"


def default_pls_sdq_self_output_dir() -> Path:
    return rs_project_root() / "reports" / "pls_sdq_self"


def default_pls_prl_output_dir() -> Path:
    return rs_project_root() / "reports" / "pls_prl"


def default_pls_core_autistic_output_dir() -> Path:
    return rs_project_root() / "reports" / "pls_core_autistic"


def default_growth_curves_output_dir() -> Path:
    return rs_project_root() / "reports" / "growth_curves"


def default_behaviour_csv() -> Path:
    """Default behavior table path.

    Expected layout: <Analysis>/RS (this repo) and <Analysis>/Behaviour/df.csv as a sibling.
    """

    return rs_project_root().parent / "Behaviour" / "df.csv"


def default_sdq_self_xlsx() -> Path:
    """Default SDQ baseline self/parent report Excel path.

    Expected layout: <Analysis>/Behaviour/SDQ_BL_selfparent_25112021 (1).xlsx.
    """

    return rs_project_root().parent / "Behaviour" / "SDQ_BL_selfparent_25112021 (1).xlsx"

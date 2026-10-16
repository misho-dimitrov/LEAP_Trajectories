#!/usr/bin/env python3

"""Run PLS for Group 1 (Neurotypical) only.

Thin wrapper around run_pls.py that sets:
  --behav-group-keep 1
  --outdir reports/pls/NT
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inject default arguments (can still be overridden from the command line)
_DEFAULTS = {
    "--behav-group-keep": "1",
    "--outdir": str(Path(__file__).resolve().parents[1] / "reports" / "pls" / "NT"),
}


def _inject_defaults() -> None:
    """Prepend default --key=value pairs unless already specified on the CLI."""
    existing_keys = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
    extras = [f"{k}={v}" for k, v in _DEFAULTS.items() if k not in existing_keys]
    sys.argv[1:1] = extras


if __name__ == "__main__":
    _inject_defaults()

    # Import after argv patching so parse_args picks up the defaults
    from run_pls import main  # type: ignore

    main()

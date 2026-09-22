# SPDX-License-Identifier: Apache-2.0
"""Regenerate the shipped full reference fronts (see fronts_full.py).

    python -m mootation.benchmarks.make_fronts [KEY|FAMILY ...] [--no-cache | --cache-only]

A separate entry point because the registry imports fronts_full itself, and
running an already-imported module as __main__ gets a runpy warning.
"""

from .fronts_full import main

if __name__ == "__main__":
    raise SystemExit(main())

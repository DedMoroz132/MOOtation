# SPDX-License-Identifier: Apache-2.0
"""The quality indicators a campaign can record, and which way each is better.

Kept apart from `metrics` because this part must import without NumPy: the
config checks, the command line and the TUI need the names, not the arithmetic.
"""

from __future__ import annotations

METRIC_NAMES = ("igd", "igdp", "igdp_norm", "eps", "eps_norm", "hv", "hv_h")

# Higher is better for these; lower for every other indicator.
HIGHER_IS_BETTER = frozenset({"hv", "hv_h"})

# These compare the answer set with a sample of the true Pareto front, so a
# problem without one gives None for them.
NEEDS_FRONT = frozenset({"igd", "igdp", "igdp_norm", "eps", "eps_norm"})

DESCRIPTIONS = {
    "igd": "mean distance from the reference front to the nearest point; raw units. "
           "Not Pareto-compliant: for comparison with published numbers, not for ranking",
    "igdp": "IGD+: only the part of each offset that is dominated counts; raw units",
    "igdp_norm": "IGD+ with objectives and front divided by nadir - ideal; scale-free",
    "eps": "additive epsilon: the smallest shift in every objective that makes the set "
           "weakly dominate the reference front; raw units",
    "eps_norm": "additive epsilon after the nadir - ideal normalisation; scale-free",
    "hv": "hypervolume, objectives normalised by ideal and nadir, reference point 1.1",
    "hv_h": "hypervolume with the reference point at 1 + 1/H, H from the problem's default "
            "population (Ishibuchi, Imada, Setoguchi & Nojima, GECCO 2017)",
}

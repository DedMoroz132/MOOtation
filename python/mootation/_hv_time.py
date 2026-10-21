# SPDX-License-Identifier: Apache-2.0
"""The time warning of the exact hypervolume contributions (task 14).

A module of its own, not minimize.py: importing the submodule
mootation.minimize would put the module in the place of the package's
function mootation.minimize."""

from __future__ import annotations

# Task 14 (the owner's word of 2026-10-21): SMS-EMOA and MO-CMA-ES compute
# the hypervolume contributions exactly at every number of objectives by
# default; from EXACT_HV_SLOW_M objectives on a step can take seconds and a run
# of 25 000 evaluations hours. minimize() warns, a campaign names such jobs
# before it runs them.
EXACT_HV_SLOW_M = 8


def exact_hv_time_warning(algorithm: str, n_objs: int, knobs: dict) -> "str | None":
    """The warning for a run of SMS-EMOA or MO-CMA-ES whose contributions are
    exact at n_objs >= EXACT_HV_SLOW_M objectives (hv::ContribChoice's rule:
    exact unless n_samples > 0 is given without race or mc, or a threshold
    hv_exact_max_m below n_objs with race or mc); None otherwise."""
    if algorithm not in ("sms_emoa", "mo_cma_es") or int(n_objs) < EXACT_HV_SLOW_M:
        return None
    if algorithm == "sms_emoa":
        approx = knobs.get("hv_contrib") in ("race", "mc")
        n_samples = knobs.get("n_samples")
        if n_samples is not None and not approx and float(n_samples) > 0:
            return None
        t = knobs.get("hv_exact_max_m")
        if t is not None and int(n_objs) > int(t):
            return None
        alternative = "hv_exact_max_m with hv_contrib = 'race' or 'mc' estimates instead (they choose badly from five objectives)"
    else:
        alternative = "MO-CMA-ES has no estimate (its step size needs exact contributions)"
    return (f"{algorithm} computes the hypervolume contributions exactly at {int(n_objs)} objectives: "
            f"a step can take seconds and a run of 25 000 evaluations hours; {alternative}")

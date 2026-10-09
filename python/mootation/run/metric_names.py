# SPDX-License-Identifier: Apache-2.0
"""The quality indicators a campaign can record, and which way each is better.

Kept apart from `metrics` because this part must import without NumPy: the
config checks, the command line and the TUI need the names, not the arithmetic.
"""

from __future__ import annotations

METRIC_NAMES = ("igd", "igdp", "igdp_norm", "gdp", "eps", "eps_norm", "hv", "hv_h",
                "roi_dist", "range_cover", "nd_share", "dup_share", "igdx", "cr", "pdist",
                "tau90", "n_final", "r2", "gap_max", "nn_cv", "gdp_norm",
                "gdp_min", "g_min", "g_median", "ps_dmin")

# What a run's variation operators did, recorded with [campaign] operator_stats
# = true (the binding's per-step bookkeeping, mootation._core.operator_stats):
# along the trajectory per interval, in `final` over the whole run. Not
# computable afterwards, so --recompute refuses them; the tables read them.
RUN_STATS = ("oob_share", "oob_var_share", "survival_share", "offspring_nd_share",
             "step_mean")

# What a run on a constrained problem records besides (task 6 of 2026-10-09,
# item 5.2), unasked: feas_share wherever an indicator is measured, which is
# then measured on the feasible part of the answer (metrics.compute), and
# fe_first_feas in `final`.
FEASIBILITY = ("feas_share", "fe_first_feas")

# What a run on a noisy copy (benchmarks/noisy.py) records besides, its
# indicators being measured on the noise-free objectives (task 6, item 7).
NOISE = ("true_dominated_share",)

# Everything a table can be asked for.
TABLE_NAMES = METRIC_NAMES + RUN_STATS + FEASIBILITY + NOISE

# Higher is better for these; lower for every other indicator. nd_share,
# range_cover and pdist are diagnostics rather than quality indicators,
# oriented the way a healthy run moves.
HIGHER_IS_BETTER = frozenset({"hv", "hv_h", "range_cover", "nd_share", "cr", "pdist",
                              "feas_share"})

# These compare the answer set with a sample of the true Pareto front, so a
# problem without one gives None for them.
NEEDS_FRONT = frozenset({"igd", "igdp", "igdp_norm", "gdp", "eps", "eps_norm", "tau90",
                         "gap_max", "gdp_norm", "gdp_min"})

# The same indicators against the dense reference of four objectives and more
# (benchmarks.registry.dense_reference, task 6 of 2026-10-09, item 9) are
# recorded beside them under the name with this suffix: igdp_norm_ref_v3.
DENSE_SUFFIX = "_ref_v3"


def base_name(name: str) -> str:
    """A metric's name without the dense reference's suffix."""
    stem = name[:-len(DENSE_SUFFIX)] if name.endswith(DENSE_SUFFIX) else None
    return stem if stem in NEEDS_FRONT else name


def is_table_name(name: str) -> bool:
    """Can a table be asked for `name`: TABLE_NAMES, and the front indicators
    against the dense reference."""
    return name in TABLE_NAMES or base_name(name) != name

# These look at the decision variables: all of them need the solutions; igdx,
# cr and ps_dmin a sample of the Pareto SET besides (BenchProblem.pareto_set),
# and g_min and g_median the problem's distance function (BenchProblem.
# distance_fn), which only some problems have.
NEEDS_X = frozenset({"igdx", "cr", "pdist", "g_min", "g_median", "ps_dmin"})
NEEDS_SET = frozenset({"igdx", "cr", "ps_dmin"})

# The hypervolumes: exact above three objectives costs seconds per call, which
# is why a campaign can keep them off the trajectory (trajectory_hv_max_m).
HV_NAMES = frozenset({"hv", "hv_h"})

DESCRIPTIONS = {
    "igd": "mean distance from the reference front to the nearest point; raw units. "
           "Not Pareto-compliant: for comparison with published numbers, not for ranking",
    "igdp": "IGD+: only the part of each offset that is dominated counts; raw units",
    "igdp_norm": "IGD+ with objectives and front divided by nadir - ideal; scale-free",
    "gdp": "GD+: the mean over the SET of the IGD+ distance to the nearest reference "
           "point — convergence only, where IGD+ mixes in coverage; raw units. "
           "Not Pareto-compliant: a set that dominates another can score worse",
    "eps": "additive epsilon: the smallest shift in every objective that makes the set "
           "weakly dominate the reference front; raw units",
    "eps_norm": "additive epsilon after the nadir - ideal normalisation; scale-free",
    "hv": "hypervolume, objectives normalised by ideal and nadir, reference point 1.1",
    "hv_h": "hypervolume with the reference point at 1 + 1/H, H from the problem's default "
            "population (Ishibuchi, Imada, Setoguchi & Nojima, GECCO 2017)",
    "roi_dist": "distance from the set to the box [ideal, nadir] in normalised units, "
                "0 once any point is inside; separates runs whose hypervolume is 0 "
                "(the COCO bbob-biobj convention); no reference front needed",
    "range_cover": "the worst objective's covered share of [ideal, nadir] (clipped to "
                   "the box); falls early when a population collapses onto part of "
                   "the front; per objective in range_cover_each; no reference front needed",
    "nd_share": "share of the set that no other member dominates; no reference front needed",
    "dup_share": "share of the set that repeats an objective vector already in it; "
                 "no reference front needed",
    "igdx": "IGD in the decision space against a Pareto-set sample, variables normalised "
            "by the bounds, cyclic ones wrapped (Tanabe & Ishibuchi 2019, Eq. 5)",
    "cr": "cover rate: how much of the Pareto set's extent in each variable the solutions "
          "span, 1 = all (Tanabe & Ishibuchi 2019, Eqs. 7-8)",
    "pdist": "mean pairwise distance between solutions in normalised variables: spread in "
             "the decision space; no reference needed",
    "tau90": "the 0.9 quantile of the IGD+ distances d+(z, A) over the reference sample, "
             "normalised like igdp_norm: 90 % of the front lies within tau90 of the set; "
             "igdp_norm is their mean, their maximum the Euclidean counterpart of eps_norm; "
             "weakly Pareto-compliant; coverage_curve gives the share within 0.01 ... 0.2",
    "n_final": "the number of points in the answer: hv, IGD+ and eps never get worse by "
               "adding points, so read them next to it when set sizes differ",
    "r2": "discrete R2 (Brockhoff, Wagner & Trautmann 2015, Def. 4): the mean over the "
          "Das-Dennis weights of hv_h's lattice of the "
          "best weighted Tchebycheff value max_i w_i (f_i - ideal_i)/(nadir_i - ideal_i) "
          "in the set; weakly Pareto-compliant, linear in the objective count where the "
          "hypervolume turns Monte-Carlo, and non-zero where hv is 0",
    "gdp_norm": "GD+ with objectives and front divided by nadir - ideal, like igdp_norm: "
                "how far the set is from the front, scale-free, so one threshold reads "
                "alike on every problem (--cover's progress criterion)",
    "gap_max": "the largest hole: the maximum over the reference sample of the IGD+ "
               "distance d+(z, A), normalised like igdp_norm — tau90 taken at 100 %; "
               "coverage, blind to how many points fill the rest",
    "nn_cv": "evenness: the coefficient of variation of the nearest-neighbour distances "
             "among the distinct non-dominated points, objectives normalised by ideal and "
             "nadir; 0 = evenly spaced, larger = clumped; no reference front needed",
    "gdp_min": "convergence without coverage: the smallest IGD+ distance d+(a, Z) of any "
               "point of the set to the reference, normalised like igdp_norm — has the set "
               "reached the front anywhere; it cannot go below the reference's resolution, "
               "given beside it (gdp_min_ref_step, the median distance between neighbouring "
               "reference points; gdp_min_below_step when lower)",
    "g_min": "the smallest g(x) - g* in the set, the problem's own distance to the Pareto "
             "set (BenchProblem.distance_fn: ZDT, DTLZ and their variants): exact, no "
             "reference sample",
    "g_median": "the median g(x) - g* over the set (see g_min)",
    "ps_dmin": "the smallest distance from a solution to a Pareto-set sample, variables "
               "normalised by the bounds, cyclic ones wrapped, as igdx's: for the problems "
               "with a Pareto set and no distance function",
    "oob_share": "share of offspring with a variable outside the box before repair",
    "oob_var_share": "share of offspring variables outside the box before repair",
    "survival_share": "share of offspring that entered the next population",
    "offspring_nd_share": "share of offspring no member of the parent population dominates",
    "step_mean": "mean distance from an offspring to the nearest parent, over ||ub - lb||",
    "feas_share": "a constrained problem's: the share of the answer that satisfies every "
                  "constraint; every other indicator is measured on that part, and an answer "
                  "without one has no value (an empty run, counted apart and ranked worst)",
    "fe_first_feas": "a constrained problem's: the evaluation that first gave a feasible "
                     "point; none when the run found none",
    "true_dominated_share": "a noisy copy's: the share of the answer that another of its "
                            "points dominates by the noise-free objectives — what the noise "
                            "let through",
}

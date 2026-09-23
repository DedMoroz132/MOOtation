# SPDX-License-Identifier: Apache-2.0
"""The quality indicators a campaign can record, and which way each is better.

Kept apart from `metrics` because this part must import without NumPy: the
config checks, the command line and the TUI need the names, not the arithmetic.
"""

from __future__ import annotations

METRIC_NAMES = ("igd", "igdp", "igdp_norm", "gdp", "eps", "eps_norm", "hv", "hv_h",
                "roi_dist", "range_cover", "nd_share", "dup_share", "igdx", "cr", "pdist",
                "tau90", "n_final")

# What a run's variation operators did, recorded with [campaign] operator_stats
# = true (the binding's per-step bookkeeping, mootation._core.operator_stats):
# along the trajectory per interval, in `final` over the whole run. Not
# computable afterwards, so --recompute refuses them; the tables read them.
RUN_STATS = ("oob_share", "oob_var_share", "survival_share", "offspring_nd_share",
             "step_mean")

# Everything a table can be asked for.
TABLE_NAMES = METRIC_NAMES + RUN_STATS

# Higher is better for these; lower for every other indicator. nd_share,
# range_cover and pdist are diagnostics rather than quality indicators,
# oriented the way a healthy run moves.
HIGHER_IS_BETTER = frozenset({"hv", "hv_h", "range_cover", "nd_share", "cr", "pdist"})

# These compare the answer set with a sample of the true Pareto front, so a
# problem without one gives None for them.
NEEDS_FRONT = frozenset({"igd", "igdp", "igdp_norm", "gdp", "eps", "eps_norm", "tau90"})

# These look at the decision variables: all of them need the solutions, and
# igdx and cr a sample of the Pareto SET besides (BenchProblem.pareto_set),
# which only some problems have.
NEEDS_X = frozenset({"igdx", "cr", "pdist"})
NEEDS_SET = frozenset({"igdx", "cr"})

# The hypervolumes: exact above three objectives costs seconds per call, which
# is why a campaign can keep them off the trajectory (trajectory_hv_max_m).
HV_NAMES = frozenset({"hv", "hv_h"})

DESCRIPTIONS = {
    "igd": "mean distance from the reference front to the nearest point; raw units. "
           "Not Pareto-compliant: for comparison with published numbers, not for ranking",
    "igdp": "IGD+: only the part of each offset that is dominated counts; raw units",
    "igdp_norm": "IGD+ with objectives and front divided by nadir - ideal; scale-free",
    "gdp": "GD+: the mean over the SET of the IGD+ distance to the nearest reference "
           "point — convergence only, where IGD+ mixes in coverage; raw units",
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
    "oob_share": "share of offspring with a variable outside the box before repair",
    "oob_var_share": "share of offspring variables outside the box before repair",
    "survival_share": "share of offspring that entered the next population",
    "offspring_nd_share": "share of offspring no member of the parent population dominates",
    "step_mean": "mean distance from an offspring to the nearest parent, over ||ub - lb||",
}

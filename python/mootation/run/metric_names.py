# SPDX-License-Identifier: Apache-2.0
"""The quality indicators a campaign can record, and which way each is better.

Kept apart from `metrics` because this part must import without NumPy: the
config checks, the command line and the TUI need the names, not the arithmetic.
"""

from __future__ import annotations

METRIC_NAMES = ("igd", "igdp", "igdp_norm", "gdp", "eps", "eps_norm", "hv", "hv_h",
                "roi_dist", "range_cover", "nd_share", "dup_share", "igdx", "cr", "pdist",
                "tau90", "n_final", "r2", "gap_max", "nn_cv", "gdp_norm",
                "gdp_min", "g_min", "g_median", "ps_dmin", "i_cmop")

# What a run's variation operators did, recorded with [campaign] operator_stats
# = true (the binding's per-step bookkeeping, mootation._core.operator_stats):
# along the trajectory per interval, in `final` over the whole run. Not
# computable afterwards, so --recompute refuses them; the tables read them.
RUN_STATS = ("oob_share", "oob_var_share", "survival_share", "offspring_nd_share",
             "step_mean")

# The same bookkeeping of crossover = de / de_pbest and portfolio (task 7 of
# 2026-10-10, §2.2 and §2.4; include/mootation/operators/de_crossover.hpp,
# real_crossover.hpp): the SHADE memory (its cells' mean and quartiles), the
# mean CR and F drawn and kept; per operator of a portfolio its share of the
# children and the share of them the selection kept, PORTFOLIO_STATS + name.
DE_STATS = ("de_mu_cr_mean", "de_mu_cr_q1", "de_mu_cr_q3", "de_mu_f_mean", "de_mu_f_q1",
            "de_mu_f_q3", "de_cr_mean", "de_f_mean", "de_cr_success_mean", "de_f_success_mean")
PORTFOLIO_STATS = ("portfolio_share_", "portfolio_kept_")

# mosa's (task 7, §1; algorithms/mosa.hpp), the state at the record: the
# temperature, the step scale, the archive's size, the reseeds so far; with
# sa_step = adaptive the three steps' probabilities, with sa_energy = tch the
# weights' mean and spread over the chains per objective, SA_WEIGHT_STATS + j.
SA_STATS = ("sa_T", "sa_s", "sa_archive_size", "sa_reseeds", "sa_p_laplace", "sa_p_de",
            "sa_p_sbx")
SA_WEIGHT_STATS = ("sa_w_mean_", "sa_w_sd_")

# The learned coordinates' (coords = pca, pca_nd, randrot; task 7, supplement
# 1): the spectrum with operator_stats, and with [campaign] coords_diag the
# principal angles against the distance block too (coords_diag.py).
COORDS_STATS = ("coords_ratio", "coords_lead_share", "coords_updates", "coords_angle_max",
                "coords_angle_mean")

# What a run on a constrained problem records besides (task 6 of 2026-10-09,
# item 5.2), unasked: feas_share wherever an indicator is measured, which is
# then measured on the feasible part of the answer (metrics.compute), and
# fe_first_feas in `final`.
FEASIBILITY = ("feas_share", "fe_first_feas")

# What a run on a noisy copy (benchmarks/noisy.py) records besides, its
# indicators being measured on the noise-free objectives (task 6, item 7).
NOISE = ("true_dominated_share",)
# ... and every final indicator twice more (task 7, §4): on the non-dominated
# set by the noise-free values of every point evaluated (_oracle), and on the
# noisy values of the answer, as the algorithm saw them (_observed).
NOISE_MODES = ("_oracle", "_observed")

# Typed variables (mootation.var_types, task 7, §10) and bit strings
# (benchmarks/binary.py, supplement 2, §3.4), in `final`.
TYPED = ("dup_x_share", "cover_gen", "cover_evals", "cover_archive_gen", "cover_archive_evals")

# Everything a table can be asked for.
TABLE_NAMES = (METRIC_NAMES + RUN_STATS + DE_STATS + SA_STATS + COORDS_STATS + FEASIBILITY
               + NOISE + TYPED)

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
    """Can a table be asked for `name`: TABLE_NAMES, the front indicators
    against the dense reference, a portfolio's per-operator shares and mosa's
    per-objective weights."""
    for mode in NOISE_MODES:
        if name.endswith(mode) and name[:-len(mode)] in METRIC_NAMES + NOISE:
            return True
    return (name in TABLE_NAMES or base_name(name) != name
            or name.startswith(PORTFOLIO_STATS + SA_WEIGHT_STATS))

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
    "i_cmop": "Vodopija et al.'s indicator for a constrained problem (IEEE TEVC 2025, "
              "Eq. 12): at most 1 with a feasible point (minus the normalised hypervolume, "
              "or the distance to [ideal, nadir]), 1 + the least violation without one; "
              "lower is better",
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
    "de_mu_cr_mean": "crossover de/de_pbest with SHADE: the mean of the memory's mu_CR",
    "de_mu_cr_q1": "the first quartile of the memory's mu_CR",
    "de_mu_cr_q3": "the third quartile of the memory's mu_CR",
    "de_mu_f_mean": "the mean of the memory's mu_F",
    "de_mu_f_q1": "the first quartile of the memory's mu_F",
    "de_mu_f_q3": "the third quartile of the memory's mu_F",
    "de_cr_mean": "the mean CR of the children drawn",
    "de_f_mean": "the mean F of the children drawn",
    "de_cr_success_mean": "the mean CR of the children the selection kept",
    "de_f_success_mean": "the mean F of the children the selection kept",
    "dup_x_share": "typed variables: the share of the evaluations whose decoded point "
                   "had been evaluated before",
    "cover_gen": "a bit-string problem's: the first generation whose population holds every "
                 "front vector; NaN when none within the budget (cover_reached false)",
    "cover_evals": "the evaluations spent by then",
    "cover_archive_gen": "the same for the inner archive (inner_archive = nd)",
    "cover_archive_evals": "the evaluations spent by then",
    "coords_ratio": "coords: the largest eigenvalue of the last update over the smallest "
                    "one above 1e-12 of it",
    "coords_lead_share": "coords: the share of the first M - 1 components in the spectrum",
    "coords_updates": "coords: the updates of the learned basis so far",
    "coords_angle_max": "coords_diag: the largest principal angle, in degrees, between the "
                        "distance block and the learned minor subspace",
    "coords_angle_mean": "coords_diag: their mean",
    "sa_T": "mosa: the temperature at the record",
    "sa_s": "mosa: the step scale at the record, over ub - lb",
    "sa_archive_size": "mosa: the archive's size at the record",
    "sa_reseeds": "mosa: the steps so far that restarted a chain from the archive",
    "sa_p_laplace": "mosa with sa_step = adaptive: the Laplace step's probability",
    "sa_p_de": "mosa with sa_step = adaptive: the differential step's probability",
    "sa_p_sbx": "mosa with sa_step = adaptive: the SBX step's probability",
    "feas_share": "a constrained problem's: the share of the answer that satisfies every "
                  "constraint; every other indicator is measured on that part, and an answer "
                  "without one has no value (an empty run, counted apart and ranked worst)",
    "fe_first_feas": "a constrained problem's: the evaluation that first gave a feasible "
                     "point; none when the run found none",
    "true_dominated_share": "a noisy copy's: the share of the answer that another of its "
                            "points dominates by the noise-free objectives — what the noise "
                            "let through",
}

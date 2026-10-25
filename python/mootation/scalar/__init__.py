# SPDX-License-Identifier: Apache-2.0
"""Scalarization with a local solver (task 9 of 2026-10-18, part A1) and the
methods beside it — mosqp and boostdms — and the polishing (polish.py): in C++
since task 17, stage 6 (include/mootation/scalar/, _core.scalar), with the
library's own SQP (slsqp.hpp: Nocedal & Wright's line-search SQP on
Lawson & Hanson's least-squares subproblem), Nelder-Mead (nelder_mead.hpp:
Lagarias et al. 1998 with Gao & Han's adaptive coefficients) and CMA-ES
(cmaes.hpp). This module names them and hands minimize()'s arguments over;
the methods, their knobs, plans and readings (SCALAR-1 to SCALAR-12, MS-n,
BD-n) are described in include/mootation/scalar/scalar.hpp, mosqp.hpp and
boostdms.hpp.

THE ALGORITHM `scalar`, and its presets under their own names:

  eps_sqp  tch_sqp  nbi_sqp  nnc_sqp  eps_nm  eps_cmaes

  scalarization  eps | tch | nbi | nnc | ps
  local          sqp | nm | cmaes
  sub_plan       grid | laumanns | gap
  tch_weights    inverse | lattice
  n_sub, sub_budget, fd_step, local_tol, cma_sigma0, eps_objective

and mosqp (mosqp_tau, mosqp_mu, mosqp_beta, mosqp_spread, fd_step) and
boostdms (boost_scalarization = eps | minmax). Every evaluation goes through
fn; the run's first points are pop_size uniform points from the seed, the
rows of the design `init` (designs.py), or the seed population.

THE VERSION of these methods' trajectories is in every result's settings,
method_version (_core.scalar.METHOD_VERSION): 3 since sqp_line_search =
accept became the default (25.10); 2 since task 17, stage 6 (the library's
solvers and random numbers; 1, absent, was SciPy's SLSQP and Nelder-Mead and
NumPy's generator).
"""

from __future__ import annotations

PRESETS = {
    "scalar": {},
    "eps_sqp": {"scalarization": "eps", "local": "sqp"},
    "tch_sqp": {"scalarization": "tch", "local": "sqp"},
    "nbi_sqp": {"scalarization": "nbi", "local": "sqp"},
    "nnc_sqp": {"scalarization": "nnc", "local": "sqp"},
    "eps_nm": {"scalarization": "eps", "local": "nm"},
    "eps_cmaes": {"scalarization": "eps", "local": "cmaes"},
}
# the methods of task 9 beside the scalarization methods, on the same
# evaluator, answer and observer
OTHERS = ("boostdms", "mosqp")
NAMES = tuple(PRESETS) + OTHERS


def _module():
    """_core.scalar, or an ImportError that says what is missing."""
    try:
        from .. import _core
        return _core.scalar
    except (ImportError, AttributeError) as e:
        raise ImportError("the scalarization methods (mootation.scalar) run in the compiled "
                          "extension, mootation._core, which is not built or too old") from e


class ScalarResult:
    """Shaped like the binding's Result."""

    def __init__(self, d, ignored, n_failed, eq_tol, unique):
        self.objectives = d["objectives"]
        self.variables = d["variables"]
        self.cv = d["cv"]
        self.active_n = len(self.objectives)
        self.ignored = list(ignored)
        self.evaluations = int(d["evaluations"])
        self.unique_evaluations = int(unique)
        self.n_failed = int(n_failed)
        self.eq_tol = float(eq_tol)
        self.operators = []
        self.settings = [tuple(p) for p in d["settings"]]
        self.operator_totals = {}
        self.inner_archive = []
        self.inner_archive_variables = []
        self.binary_variables = []
        # the objectives of every subproblem's solution, anchors first, in the
        # order solved (feasible ones; a diagnostic, not the answer)
        self.solutions = d.get("solutions", [])


def run(name, fn, constraints, *, bounds, n_objs: int, n_cons: int, n_eq: int, eq_tol: float,
        eq_tol0, pop_size: int, budget: int, seed: int, knobs: dict, on_generation=None,
        record_every: int = 0, init: str = "uniform", init_share: float = 0.0,
        seed_points=None, unique_count=None, guard=None):
    core = _module()
    design = None
    if seed_points is None and init != "uniform":
        from .. import designs
        n0 = pop_size
        if init_share:
            n0 = max(n0, int(round(float(init_share) * budget)))
        design = [list(map(float, x)) for x in designs.design(init, n0, bounds, seed)]
    if name == "mosqp":
        design = None                 # mosqp starts from the box's centre and its line
    seeds = None if seed_points is None else [(list(x), list(f), list(g)) for x, f, g in seed_points]
    d = core.run(name, fn, constraints, bounds, int(n_objs), int(n_cons), int(n_eq), float(eq_tol),
                 eq_tol0, int(pop_size), int(budget), int(seed), dict(knobs),
                 on_generation=on_generation if int(record_every) > 0 else None,
                 record_every=int(record_every), design=design,
                 seed_points=seeds if name != "mosqp" else None)
    n_failed = guard.n_failed if guard is not None else d["n_failed"]
    unique = unique_count() if unique_count is not None else d["evaluations"]
    return ScalarResult(d, d["ignored"], n_failed, eq_tol if n_eq else 0.0, unique)

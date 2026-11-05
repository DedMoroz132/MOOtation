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
boostdms (boost_scalarization = eps | minmax).

THE SINGLE-OBJECTIVE SOLVERS (task 18, item 2.3; n_objs = 1 only): the local
solvers alone, each with its knobs in scalar, once from the initial plan's
best point (scalar.hpp, run_single; SINGLE-1, SINGLE-2):

  slsqp        fd_step, local_tol, sqp_line_search
  nelder_mead  local_tol
  cmaes        cma_sigma0, local_tol

and, in the problem's own coordinates, bounds -inf / +inf allowed (task 18,
stage 1; scalar.hpp, run_local; LOCAL-1 to LOCAL-4):

  lm           lm_ftol, lm_xtol, lm_gtol, lm_loss, lm_f_scale, lm_gauge (the
               problem's residuals; Moré 1978; lm_gauge = on: the step off the
               problem's gauge_basis, task 19)
  lbfgsb       lbfgsb_m, lbfgsb_pgtol, lbfgsb_ftol (Byrd, Lu, Nocedal, Zhu 1995)
  eki          eki_ensemble, eki_init, eki_sigma0, eki_gamma, eki_perturb,
               eki_noise, eki_tau (the residuals; ensemble Kalman inversion,
               Iglesias, Law, Stuart 2013; task 19)

nelder_mead and eps_nm take nm_coefficients = adaptive | standard (task 19).
A run with residuals answers result.profile(j, ...) (lm.hpp's profiles).

grad_cost (1) for all but eki: what a call of the problem's gradient costs of
the budget (evaluator.hpp, THE COUNTS). The problem may give its gradient
(grad), its residuals (residuals, the objective then 1/2 ||r||^2) and their
Jacobian (jac), and a start x0 (minimize()).

THE POPULATION METHODS of stage 4 (task 18, §4; n_objs = 1 only, every
bound finite), each with its paper's defaults (scalar.hpp, run_population,
where their knobs are listed; population.hpp, POP-1 to POP-8):

  de  jade  shade  lshade  ga_sbx  ssga  g3_pcx  chc  pso  spso2011  sep_cmaes

np sets the population (0: the paper's); pop_size does not (POP-6).

Every evaluation goes through fn; the run's first points are pop_size uniform
points from the seed, the rows of the design `init` (designs.py), or the seed
population.

THE VERSION of these methods' trajectories is in every result's settings,
method_version (_core.scalar.METHOD_VERSION): 4 since sqp_line_search =
powell became the default (26.10), 3 since accept did (25.10); 2 since task 17, stage 6 (the library's
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
# the population methods of stage 4 (task 18, §4; scalar.hpp, run_population)
POPULATION = ("de", "jade", "shade", "lshade", "ga_sbx", "ssga", "g3_pcx", "chc", "pso", "spso2011",
              "sep_cmaes")
# the single-objective algorithms (task 18, item 2.3, stage 1 and stage 4)
SINGLE = ("slsqp", "nelder_mead", "cmaes", "lm", "lbfgsb", "eki") + POPULATION
# the methods beside the scalarization methods, on the same evaluator,
# answer and observer: those of task 9 and the single-objective algorithms.
# None lays its run out over the budget (budget.py) but lshade, whose
# population schedule runs to the budget (POP-8)
OTHERS = ("boostdms", "mosqp") + tuple(s for s in SINGLE if s != "lshade")
NAMES = tuple(PRESETS) + ("boostdms", "mosqp") + SINGLE


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
        self.grad_calls = int(d.get("grad_calls", 0))
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
        # one objective: a record each time the best point improved, (evaluations,
        # grad_calls, f, cv) at that point (improvements.hpp)
        self.improvements = [tuple(e) for e in d.get("improvements", [])]
        self._lsq = None                # (residuals, jac, bounds) of a least-squares run

    def profile(self, j: int, grid=None, *, alpha: float = 0.95, df: int = 1, sigma2: float = 0.0,
                max_steps: int = 50, x=None) -> dict:
        """The profile of variable j about x (the best point by default) of a
        run with residuals (lm.hpp, profile; task 19, §19.3.3): for each value
        of x_j the least 1/2 ||r||^2 with x_j fixed, by lm from the neighbouring
        node; the threshold chi2_quantile(alpha, df) / 2 (times sigma2 where
        the residuals are not divided by their sigma; sigma2 = 0: relative),
        the interval [lower, upper] where the profile crosses it, and the kind
        (identifiable, practically_non_identifiable,
        structurally_non_identifiable; Raue et al. 2009). grid: the values,
        or None for steps from (J^T J)^+ (LMP-2). Its evaluations are the
        residuals' own calls, outside the run's budget and counts."""
        if self._lsq is None:
            raise ValueError("profile: the run had no residuals (least squares only)")
        residuals, jac, bounds = self._lsq
        x_hat = [float(v) for v in (self.variables[0] if x is None else x)]
        m = len(residuals(x_hat))
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        return _module().lm_profile(lambda v: list(residuals(v)), jac, x_hat, m, int(j),
                                    bounds=(lo, hi), grid=None if grid is None else [float(v) for v in grid],
                                    alpha=float(alpha), df=int(df), sigma2=float(sigma2),
                                    max_steps=int(max_steps))


def run(name, fn, constraints, *, bounds, n_objs: int, n_cons: int, n_eq: int, eq_tol: float,
        eq_tol0, pop_size: int, budget: int, seed: int, knobs: dict, on_generation=None,
        record_every: int = 0, init: str = "uniform", init_share: float = 0.0,
        seed_points=None, unique_count=None, guard=None, grad=None, residuals=None, jac=None,
        x0=None, gauge_basis=None):
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
                 seed_points=seeds if name != "mosqp" else None, grad=grad, residuals=residuals,
                 jac=jac, x0=None if x0 is None else [float(v) for v in x0],
                 gauge_basis=gauge_basis)
    n_failed = guard.n_failed if guard is not None else d["n_failed"]
    unique = unique_count() if unique_count is not None else d["evaluations"]
    return ScalarResult(d, d["ignored"], n_failed, eq_tol if n_eq else 0.0, unique)

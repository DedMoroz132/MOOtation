# SPDX-License-Identifier: Apache-2.0
"""One function for the common case.

`_core.run` wants a Problem object with three fields set and a Config object
with the knobs you care about. That is the right shape for a binding — it maps
onto the C++ one-to-one and hides nothing — but it means every caller writes
the same six lines. `minimize` writes them.

Anything it cannot express is still reachable: build `Problem` and `Config`
yourself and call `run`. This function adds no capability, only brevity.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from . import _core

# Every optional knob the binding accepts. Kept as a tuple rather than
# **kwargs-into-setattr so that a typo is an error naming the alternatives,
# instead of a silently ignored argument — a run configured with an ignored
# parameter is not the run that was asked for.
KNOBS = ("eta_c", "eta_m", "pc", "pm", "T", "delta", "nr", "kappa",
         "K", "n_clusters", "theta", "alpha", "F", "CR", "div")

def minimize(
    fn: Callable[[Sequence[float]], Sequence[float]],
    bounds: Iterable[tuple[float, float]],
    n_objs: int,
    *,
    algorithm: str = "nsga2",
    pop_size: int = 100,
    n_gen: int = 250,
    seed: int = 0,
    constraints: Callable[[Sequence[float]], Sequence[float]] | None = None,
    n_cons: int = 0,
    batch: Callable[[list], list] | None = None,
    seed_population=None,
    on_size_mismatch: str = "truncate",
    save_population=None,
    log_evaluations=None,
    **knobs,
):
    """Minimize `fn` over `bounds` and return the final population.

    fn          takes one decision vector, returns `n_objs` objective values.
    bounds      one (lower, upper) pair per decision variable.
    n_objs      how many objectives `fn` returns.
    algorithm   any name from `mootation.algorithms()`.
    constraints takes one decision vector, returns `n_cons` values; each <= 0
                means satisfied. Supplying it turns constraint handling on.
    batch       an optional vectorized evaluator: takes a list of decision
                vectors, returns a list of objective vectors. It crosses into
                Python once per generation instead of once per individual,
                which is the difference that matters when `fn` is cheap.
    seed_population
                start from a previous run instead of a random population: a
                path written by `save_population`, or the object
                `mootation.load_population` returns. Costs ZERO evaluations —
                the objectives come from the file. Only variables and
                objectives are carried, so a population saved by NSGA-II can
                seed a MOEA/D run.
    on_size_mismatch
                what to do when the seed population is not `pop_size` long:
                truncate | pad | error. Routine rather than exceptional, since
                NSGA-III and M2M constrain pop_size themselves.
    save_population
                path to write the final population to, in the same format
                `seed_population` reads.
    log_evaluations
                path to a CSV of EVERY evaluation, including the ones the
                optimizer discarded. Off unless given: no file is opened and
                nothing is wrapped.
    **knobs     any of KNOBS. A knob the chosen algorithm does not have is
                reported in `result.ignored` rather than dropped.

    Returns the binding's Result: `.objectives`, `.variables`, `.cv`,
    `.active_n`, `.ignored`.
    """
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    if not bounds:
        raise ValueError("bounds is empty — there are no decision variables")
    for i, (lo, hi) in enumerate(bounds):
        if not lo < hi:
            raise ValueError(
                f"bounds[{i}] = ({lo}, {hi}): lower must be < upper")
    if n_objs < 1:
        raise ValueError(f"n_objs must be >= 1, got {n_objs}")

    known = set(_core.algorithms())
    if algorithm not in known:
        near = sorted(n for n in known if n.startswith(algorithm[:3]))
        hint = f"; did you mean: {', '.join(near[:5])}" if near else ""
        raise ValueError(f"unknown algorithm '{algorithm}'{hint}")

    bad = sorted(set(knobs) - set(KNOBS))
    if bad:
        raise TypeError(
            f"unknown parameter(s): {', '.join(bad)}. "
            f"Known: {', '.join(KNOBS)}")

    log = None
    if log_evaluations is not None:
        from .persistence import EvaluationLog
        log = EvaluationLog(log_evaluations,
                            meta={"algorithm": algorithm, "pop_size": pop_size,
                                  "n_gen": n_gen, "seed": seed})
        fn = log.wrap(fn, constraints)
        if batch is not None:
            # A batched evaluator bypasses `fn`, so the wrapper would never
            # see anything. Refusing is better than writing an empty log and
            # letting the caller believe the run was recorded.
            raise ValueError(
                "log_evaluations cannot be combined with batch=: the batched "
                "evaluator replaces the per-point function the log wraps. "
                "Log inside your batch function instead.")

    problem = _core.Problem()
    problem.bounds = bounds
    problem.n_objectives = int(n_objs)

    if constraints is not None:
        if n_cons < 1:
            # Counted once rather than guessed per call: the binding needs the
            # width up front, and calling the user's function during setup to
            # find it out would spend an evaluation they did not ask for.
            probe = list(constraints([lo for lo, _ in bounds]))
            n_cons = len(probe)
        problem.n_limits = int(n_cons)

        def _eval_with_cons(x, _f=fn, _c=constraints):
            return list(_f(x))

        problem.evaluate = _eval_with_cons
        problem.limits_batch = lambda X, _c=constraints: [list(_c(x)) for x in X]
    else:
        problem.n_limits = 0

    if batch is not None:
        problem.evaluate_batch = lambda X, _b=batch: [list(r) for r in _b(X)]
    if constraints is None:
        problem.evaluate = lambda x, _f=fn: list(_f(x))

    cfg = _core.Config()
    cfg.pop_size = int(pop_size)
    cfg.n_gen = int(n_gen)
    cfg.seed = int(seed)
    if constraints is not None:
        cfg.constraint_mode = _core.ConstraintMode.FEASIBILITY
    for k, v in knobs.items():
        setattr(cfg, k, v)

    if seed_population is not None:
        from .persistence import fit_population, load_population
        pop = (load_population(seed_population)
               if isinstance(seed_population, (str, bytes)) or
               hasattr(seed_population, "__fspath__")
               else seed_population)
        if len(pop.variables[0]) != len(bounds):
            raise ValueError(
                f"the seed population has {len(pop.variables[0])} variables "
                f"but bounds gives {len(bounds)}; a warm start cannot change "
                "the number of decision variables")
        if pop.objectives and len(pop.objectives[0]) != n_objs:
            raise ValueError(
                f"the seed population has {len(pop.objectives[0])} objectives "
                f"but n_objs is {n_objs}; a warm start cannot change the "
                "number of objectives")
        pop = fit_population(pop, int(pop_size), on_size_mismatch)
        cfg.seed_variables = [list(v) for v in pop.variables]
        cfg.seed_objectives = [list(v) for v in pop.objectives]

    try:
        result = _core.run(algorithm, problem, cfg)
    finally:
        if log is not None:
            log.close()

    if save_population is not None:
        from .persistence import save_population as _save
        _save(result, save_population,
              meta={"algorithm": algorithm, "pop_size": pop_size,
                    "n_gen": n_gen, "seed": seed, "n_objs": n_objs})
    return result

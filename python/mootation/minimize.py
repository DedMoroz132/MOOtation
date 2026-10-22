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

import warnings
from typing import Callable, Iterable, Sequence

from . import _core

# Every optional knob the binding accepts. Kept as a tuple rather than
# **kwargs-into-setattr so that a typo is an error naming the alternatives,
# instead of a silently ignored argument — a run configured with an ignored
# parameter is not the run that was asked for.
# The knobs after the per-knob setters (settings.hpp, is_core_param and
# is_core_word) travel in Config.params and Config.words, a list knob
# (portfolio) as comma-separated words.
PARAM_KNOBS = ("ref_delta", "de_p", "de_memory", "p_de", "sa_chains", "sa_reseed", "sa_t_ratio",
               "sa_s0", "sa_s_end", "sa_archive_cap", "sa_psa_alpha", "sa_adapt_window",
               "coords_warmup", "coords_update", "spu_frac", "inner_archive_cap",
               "archive_reuse_p", "dn_cf", "vmax_frac", "vel_on_bound_low", "vel_on_bound_up",
               "w_min", "w_max", "c_min", "c_max", "turbulence_k", "turbulence_p",
               "archive_size", "eps_archive_eta", "r2_weights", "r2_offset", "bc_lambda",
               "paes_depth", "mads_tau", "mads_delta0", "glods_search_every",
               "glods_local_tol", "rm_extension", "mosqp_tau", "mosqp_mu", "mosqp_beta", "mosqp_spread",
               "n_sub",
               "sub_budget", "fd_step",
               "local_tol", "cma_sigma0", "eps_objective", "n_samples", "hv_exact_max_m",
               "race_eps", "race_delta", "race_max_samples", "race_abs_tol", "hv_incremental",
               "n_offspring", "n_ref_points", "ada")
WORD_KNOBS = ("de_adapt", "de_adapt_weight", "de_archive", "de_base", "de_index",
              "portfolio", "portfolio_credit",
              "sa_energy", "sa_step", "sa_weights", "sa_tch_accept", "coords", "survival",
              "inner_archive", "bin_crossover", "mating", "smpso_preset", "chi_mode",
              "turbulence", "r_per_dim", "archive_prune", "leader_select", "leader_offer", "r2_mode",
              "r2_space", "bc_selection", "mads_select", "mads_poll", "mads_search", "mads_cache",
              "cache_check",
              "dms_cache", "paes_cache",
              "boost_scalarization", "scalarization",
              "local", "sub_plan", "tch_weights", "hv_algo", "hv_contrib")
# Polishing at the end of the budget (task 9, A2; mootation.scalar.polish):
# every algorithm takes them, and minimize() does the polishing itself.
POLISH_KNOBS = ("polish", "polish_share", "polish_points")


def direct_path(fn, *, native: str = "auto", algorithm: str = "nsga2", constraints=None,
                eq_constraints=None, batch=None, log_evaluations=None, var_types=None,
                budget_counts: str = "calls", max_failure_share=None, seed_population=None,
                **knobs):
    """(the _core.Native the core evaluates fn as, why not): minimize's direct
    path (task 11 of 2026-10-18, §1.3). fn is a problem of the C++ suite when
    it carries one (a benchmark's evaluate, benchmarks/native.py), and the core
    evaluates it itself where nothing of the caller's stands between: no batch
    or constraints but the problem's own, no eq_constraints, log_evaluations,
    binary or categorical var_types (integer and discrete ones the core decodes
    as the Python path does: task 13 of 2026-10-19, §3b), budget_counts =
    "unique", max_failure_share, seed_population or polishing, and a core, not
    a scalarization method. Takes minimize's arguments; a campaign asks it
    before it runs one."""
    if native not in ("auto", "off"):
        raise ValueError(f"native = '{native}': auto or off")
    obj = getattr(fn, "native", None)
    if obj is None:
        return None, "fn is not a problem of the C++ suite"
    if native == "off":
        return None, "native = 'off'"
    from .scalar import NAMES as SCALAR_NAMES
    own = (constraints is None or getattr(constraints, "native", None) is obj) and \
          (batch is None or getattr(batch, "__self__", None) is obj)
    why = ("a scalarization method runs in Python" if algorithm in SCALAR_NAMES else
           "batch or constraints of the caller's" if not own else
           "eq_constraints" if eq_constraints is not None else
           "log_evaluations" if log_evaluations is not None else
           "binary or categorical var_types" if var_types is not None and any(
               t == "bin" or (isinstance(t, (tuple, list)) and t[0] == "cat")
               for t in var_types) else
           "budget_counts = 'unique'" if budget_counts != "calls" else
           "max_failure_share" if max_failure_share is not None else
           "seed_population" if seed_population is not None else
           "polish" if str(knobs.get("polish", "off")) != "off" else None)
    return (None, why) if why else (obj, None)
KNOBS = ("eta_c", "eta_m", "pc", "pm", "T", "delta", "nr", "kappa",
         "K", "n_clusters", "theta", "alpha", "F", "CR", "div", "normalize",
         "bound_repair", "crossover", "mutation", "mutation_scale", "mixture_q", "blx_alpha",
         "sbx_var_prob", "crowding_space", "dms_init", "dms_poll", "step_share",
         "F_spread") + PARAM_KNOBS + WORD_KNOBS + POLISH_KNOBS

from ._hv_time import exact_hv_time_warning  # noqa: E402  (task 14: exact at every M)


def minimize(
    fn: Callable[[Sequence[float]], Sequence[float]],
    bounds: Iterable[tuple[float, float]],
    n_objs: int,
    *,
    algorithm: str = "nsga2",
    pop_size: int = 100,
    n_gen: int = 250,
    max_evaluations: int = 0,
    seed: int = 0,
    constraints: Callable[[Sequence[float]], Sequence[float]] | None = None,
    n_cons: int = 0,
    eq_constraints: Callable[[Sequence[float]], Sequence[float]] | None = None,
    n_eq: int = 0,
    eq_tol: float = 1e-4,
    eq_tol_schedule: str = "fixed",
    eq_tol0: float | None = None,
    constraint_mode: str = "feasibility",
    eps_schedule: dict | None = None,
    batch: Callable[[list], list] | None = None,
    batch_with_constraints: bool = False,
    seed_population=None,
    on_size_mismatch: str = "truncate",
    save_population=None,
    log_evaluations=None,
    on_generation=None,
    record_every: int = 0,
    operator_stats: bool = False,
    init: str = "uniform",
    init_share: float = 0.0,
    var_types=None,
    budget_counts: str = "calls",
    int_repair: str = "lamarck",
    eta_int: float | None = None,
    on_failure: str = "infeasible",
    failure_exceptions=Exception,
    max_failure_share: float | None = None,
    failure_window: int = 100,
    native: str = "auto",
    on_rows=None,
    **knobs,
):
    """Minimize `fn` over `bounds` and return the final population.

    fn          takes one decision vector, returns `n_objs` objective values.
    bounds      one (lower, upper) pair per decision variable.
    n_objs      how many objectives `fn` returns.
    algorithm   any name from `mootation.algorithms()`.
    constraints takes one decision vector, returns `n_cons` values; each <= 0
                means satisfied. Supplying it turns constraint handling on.
    eq_constraints, n_eq
                equalities h(x) = 0: takes one decision vector, returns n_eq
                values (n_eq must be given). An equality holds where
                |h| <= eq_tol and adds max(|h| - eq_tol, 0) to the violation
                cv, beside the inequalities' (data_vault.hpp, EQUALITY
                CONSTRAINTS). Turns constraint handling on.
    eq_tol      the equalities' tolerance, 1e-4 by default: Liu and Wang's,
                with the same violation (the DOC problems, IEEE TEVC 23(5),
                2019, Eq. 3); the RWMOP competition's guidelines (Kumar et
                al. 2021) take 1e-4 too, but count |h| itself beyond it.
    constraint_mode
                with constraints: "feasibility" (the default) ranks a feasible
                point above every infeasible one and infeasible ones by their
                violation; "eps_level" counts a violation up to a level eps as
                none (Takahama and Sakai's epsilon-level comparison), eps set
                once per generation by Fan et al.'s improved rule (Soft
                Computing 23(23), 2019, Eq. 8) and 0 from 80 % of the budget
                on (python/mootation_module.cpp, RunConfig::eps_level). Every
                core that takes constraints takes it; result.cv is the
                violation itself. "none" (task 15, §5): the algorithm sees the
                objectives only — the constraints are still evaluated, and
                result.cv is the violation, so the feasible part of the answer
                can be measured (Tanabe & Oyama, CEC 2017: NSGA-II without any
                handling approximates C1-DTLZ1's, C1-DTLZ3's and C2-DTLZ2's
                feasible fronts well).
    eps_schedule
                with "eps_level": {"tau": 0.1, "alpha": 0.95, "theta": 0.05,
                "tc": 0.8}, the paper's values, any of them replaced; tc is
                the share of the budget from which eps is 0.
    eq_tol_schedule, eq_tol0
                "fixed" (the default): eq_tol throughout. "linear": the
                tolerance falls from eq_tol0 to eq_tol in proportion to the
                evaluations spent (it needs max_evaluations; with a design,
                over the evaluations left to the algorithm); every point's
                violation follows the current tolerance, and the answer's cv
                is taken at eq_tol (result.eq_tol). An observer reads the
                tolerance of the moment from mootation._core.current_eq_tol().
    batch       an optional vectorized evaluator: takes a list of decision
                vectors, returns a list of objective vectors. It crosses into
                Python once per generation instead of once per individual,
                which is the difference that matters when `fn` is cheap. A
                core that evaluates a whole generation at once hands it the
                whole generation (docs/running.md lists which do), so `batch`
                may spread the rows over processes; the points of a core that
                evaluates one at a time come one row at a time. With
                `constraints`, those are asked for every row of the batch
                (one call of `constraints` per row).
    batch_with_constraints
                True: `batch` computes the constraints too, and returns the
                pair (objective rows, constraint rows) — for a model whose
                objectives and constraints come out of ONE computation, which
                then runs once per point. No `constraints` then; n_cons as
                with them (without it, one point at the lower bounds is
                computed to count them).
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
                optimizer discarded; of a `batch`, its rows in order, written
                a batch at a time. Off unless given: no file is opened and
                nothing is wrapped.
    max_evaluations
                stop once this many evaluations have been spent instead of
                after `n_gen` steps, which is then ignored. Use it whenever
                algorithms are compared: a step is not a generation for every
                core (NIMMO evaluates one offspring per step, MOEA/D-DRA and
                -AWA a fifth of the population). Checked between steps, so a
                generational core overshoots by less than one generation, and
                `record_every` then counts `pop_size` evaluations per unit.
    on_generation, record_every
                an observer `on_generation(gen, objectives)` called after
                setup (gen 0) and after every `record_every`-th generation
                with the current answer set's objective rows. This is how a
                convergence trajectory is recorded without re-running at
                several budgets. Off unless record_every > 0. The observer
                gets the rows' variables, when it needs them, from
                mootation._core.current_variables().
    operator_stats
                keep per-step statistics of the variation operators, read with
                mootation._core.operator_stats() from on_generation: offspring,
                oob_share, oob_var_share, survival_share, offspring_nd_share,
                step_mean. Off by default; it never changes the run's result.
                result.operators lists the operators used either way.
    init        the first population: "uniform", the algorithm's own random
                start (the default, as in its paper), or a design of
                mootation.designs ("sobol", "lhs", "halton_rand", "sobol_owen"),
                evaluated here through `fn` — the pop_size evaluations the
                algorithm's own start would cost —
                and handed over as a seed population.
    init_share  with a design: the share of max_evaluations it takes (0: just
                pop_size points). A larger design is a plan of experiments
                before the optimization: the algorithm starts from pop_size of
                its points, the nondominated first, the rest by DSS
                (designs.start_points), and has what is left of the budget.
    on_failure  what a failed evaluation does (mootation.failures): an
                exception of `failure_exceptions` (Exception) from fn,
                constraints or batch, a non-finite value or a None row of a
                batch. "infeasible" (the default): the point is dominated by
                every point that did not fail and is never in the answer;
                result.n_failed counts them. "raise": the run stops.
    max_failure_share, failure_window
                with "infeasible": stop with FailureShareExceeded once more
                than this share of the last failure_window (100) evaluations
                failed; off unless given.
    var_types   one entry per variable: "real", "int", ("discrete", values),
                "bin" or ("cat", labels) (mootation.var_types); fn gets the
                decoded point, a categorical variable as its label.
    budget_counts
                "calls" (the default): max_evaluations counts calls, and fn
                runs for every one. "unique": a decoded point evaluated before
                is answered from memory, and max_evaluations counts distinct
                points — what a solver run costs (mootation.unique); a
                run whose core asks for nothing new in 10 000 steps ends
                there (a space smaller than the budget).
                result.unique_evaluations counts the distinct points either
                way, result.evaluations the calls.
    native      "auto" (the default): fn a problem of the C++ suite (the
                evaluate of a benchmark of mootation.benchmarks: ZDT, DTLZ and
                its variants, MaF, ZCAT and the rotations over them), the core
                evaluates it itself, without calling Python, wherever nothing
                of the caller's stands between (direct_path says what does).
                The values are the same; the crossing into Python goes, and
                result.native_seconds is the time the evaluations took. "off":
                fn is called as any function.
    on_rows     with the core evaluating the problem itself (native): an
                observer of every evaluation, on_rows(X, F, G), arrays of the
                points evaluated, their objectives and their constraint values
                (no columns without constraints), in the order evaluated, a
                batch at a time — before every on_generation call, at the end
                and every few thousand rows in between; a design's points at
                once, before the run. An error where the core does not evaluate
                the problem itself.
    **knobs     any of KNOBS. A knob the chosen algorithm does not have is
                reported in `result.ignored` rather than dropped.

    Returns the binding's Result: `.objectives`, `.variables`, `.cv`,
    `.active_n`, `.ignored`, `.evaluations`, `.unique_evaluations`,
    `.n_failed`, `.eq_tol`.
    """
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    if not bounds:
        raise ValueError("bounds is empty — there are no decision variables")
    for i, (lo, hi) in enumerate(bounds):
        gene = (var_types is not None and i < len(var_types)
                and (var_types[i] == "bin" or (isinstance(var_types[i], (tuple, list))
                                               and var_types[i][0] == "cat")))
        if not lo < hi and not gene:            # a bit's or a label's bounds are ignored
            raise ValueError(
                f"bounds[{i}] = ({lo}, {hi}): lower must be < upper")
    if n_objs < 1:
        raise ValueError(f"n_objs must be >= 1, got {n_objs}")
    slow = exact_hv_time_warning(algorithm, n_objs, knobs)
    if slow:
        warnings.warn(slow, stacklevel=2)

    from .scalar import NAMES as SCALAR_NAMES
    known = set(_core.algorithms()) | set(SCALAR_NAMES)
    if algorithm not in known:
        near = sorted(n for n in known if n.startswith(algorithm[:3]))
        hint = f"; did you mean: {', '.join(near[:5])}" if near else ""
        raise ValueError(f"unknown algorithm '{algorithm}'{hint}")

    bad = sorted(set(knobs) - set(KNOBS))
    if bad:
        raise TypeError(
            f"unknown parameter(s): {', '.join(bad)}. "
            f"Known: {', '.join(KNOBS)}")
    # polishing at the end of the budget (task 9, A2; mootation.scalar.polish)
    from .scalar import polish as _polish
    polish_how, polish_share, polish_points = _polish.check(knobs, max_evaluations)

    # A problem of the C++ suite the core evaluates itself (direct_path)
    direct, why = direct_path(fn, native=native, algorithm=algorithm, constraints=constraints,
                              eq_constraints=eq_constraints, batch=batch,
                              log_evaluations=log_evaluations, var_types=var_types,
                              budget_counts=budget_counts, max_failure_share=max_failure_share,
                              seed_population=seed_population, **knobs)
    if on_rows is not None and direct is None:
        raise ValueError(f"on_rows observes a problem the core evaluates itself; here it "
                         f"does not: {why}")
    if direct is not None:
        if int(n_objs) != direct.n_obj or len(bounds) != direct.n_var:
            raise ValueError(f"the problem of the C++ suite has {direct.n_var} variables and "
                             f"{direct.n_obj} objectives")
        constraints, batch, batch_with_constraints = None, None, False

    # Equality constraints (task 7, supplement 3, §3): the core takes them after
    # the inequalities, as one function of n_cons + n_eq values, and turns each
    # h into |h| - eq_tol itself, so that a moving tolerance reaches every point.
    if eq_tol_schedule not in ("fixed", "linear"):
        raise ValueError(f"eq_tol_schedule = '{eq_tol_schedule}': fixed or linear")
    has_ineq = constraints is not None
    if eq_constraints is not None:
        if int(n_eq) < 1:
            raise ValueError("eq_constraints needs n_eq, the number of equalities it returns")
        if not float(eq_tol) >= 0.0:
            raise ValueError(f"eq_tol must be >= 0, got {eq_tol}")
        if eq_tol_schedule == "linear":
            if eq_tol0 is None or not float(eq_tol0) >= float(eq_tol):
                raise ValueError("eq_tol_schedule = 'linear' needs eq_tol0 >= eq_tol, "
                                 "the tolerance at the start")
            if not max_evaluations:
                raise ValueError("eq_tol_schedule = 'linear' runs over the evaluation "
                                 "budget: give max_evaluations")
        if has_ineq:
            constraints = lambda x, _c=constraints, _h=eq_constraints: (    # noqa: E731
                list(_c(x)) + list(_h(x)))
        else:
            constraints = lambda x, _h=eq_constraints: list(_h(x))         # noqa: E731
    elif n_eq or eq_tol_schedule != "fixed" or eq_tol0 is not None:
        raise ValueError("n_eq, eq_tol_schedule and eq_tol0 go with eq_constraints")
    n_eq = int(n_eq) if eq_constraints is not None else 0

    # One computation for the objectives and the constraints (task 10, §4.2):
    # the batch's constraint rows are kept, by point, until the core asks for
    # them right after the objectives (the binding asks limits_batch for the
    # rows it has just evaluated); a point asked for outside a batch (a warm
    # start's, a design's) is computed alone.
    if batch_with_constraints:
        if batch is None:
            raise ValueError("batch_with_constraints needs batch, returning (objectives, "
                             "constraints)")
        if constraints is not None or eq_constraints is not None:
            raise ValueError("batch_with_constraints: the constraints come from batch; give "
                             "no constraints or eq_constraints")
        from .unique import _key
        both, last = batch, {}

        def objectives_part(X, _b=both):
            F, G = _b(X)
            last.clear()
            for x, g in zip(X, G):
                last[_key(x)] = None if g is None else list(g)
            return F

        def constraints_part(x, _b=both):
            k = _key(x)
            if k in last and last[k] is not None:
                return last[k]
            return list(_b([x])[1][0])

        batch, constraints, has_ineq = objectives_part, constraints_part, True

    # Failed evaluations (mootation.failures; task 7, supplement 3, §2): the
    # guard turns the user's exceptions and None rows into NaN, which the core
    # marks failed; with "raise" nothing is caught and the core raises on NaN.
    if on_failure not in ("infeasible", "raise"):
        raise ValueError(f"on_failure = '{on_failure}': infeasible or raise")
    guard = None
    if on_failure == "infeasible":
        from .failures import Guard
        guard = Guard(n_objs, n_cons, failure_exceptions, max_failure_share, failure_window)
        fn = guard.fn(fn)
        if constraints is not None:
            constraints = guard.constraints(constraints)
        if batch is not None:
            batch = guard.batch(batch)
    elif max_failure_share is not None:
        raise ValueError("max_failure_share needs on_failure = 'infeasible'")

    log = None
    if log_evaluations is not None:
        from .persistence import EvaluationLog
        log = EvaluationLog(log_evaluations,
                            meta={"algorithm": algorithm, "pop_size": pop_size,
                                  "n_gen": n_gen, "seed": seed})
        fn = log.wrap(fn, constraints, guard)
        if batch is not None:                       # the rows of a batch, a batch at a time
            batch = log.wrap_batch(batch, constraints, guard)

    # Distinct decoded points (unique.py; supplement 3, §1.2): counted at the
    # caller's side of var_types, so that the key is the decoded point; with
    # budget_counts = "unique" a repeat is answered from memory.
    if budget_counts not in ("calls", "unique"):
        raise ValueError(f"budget_counts = '{budget_counts}': calls or unique")
    from .unique import Points
    points = Points(cache=budget_counts == "unique")
    fn = points.fn(fn)
    if constraints is not None:
        constraints = points.constraints(constraints)
    if batch is not None:
        batch = points.batch(batch)

    # Integer, discrete, binary and categorical variables (var_types.py; task 7,
    # §10, supplement 2, §3, supplement 3, §1): the core sees the real, integer
    # and discrete ones as its real genome and the bits and categorical ones as
    # its integer genome; fn, constraints and batch get the caller's decoded
    # variables in the caller's order.
    if int_repair not in ("lamarck", "baldwin"):
        raise ValueError(f"int_repair = '{int_repair}': lamarck or baldwin")
    layout = None
    if var_types is not None:
        from .var_types import Layout
        layout = Layout(var_types, list(bounds))
        if layout.trivial():
            layout = None
    n_bin = 0
    if layout is not None:
        user_fn, user_cons, user_batch = fn, constraints, batch
        fn = lambda xc, _f=user_fn: _f(layout.to_user(xc))                  # noqa: E731
        if user_cons is not None:
            constraints = lambda xc, _c=user_cons: _c(layout.to_user(xc))   # noqa: E731
        if user_batch is not None:
            batch = lambda X, _b=user_batch: _b([layout.to_user(x) for x in X])  # noqa: E731
        bounds = layout.core_bounds()
        n_bin = len(layout.genes)
        if n_bin and init != "uniform":
            raise ValueError("init designs are real-valued: not with binary or categorical "
                             "variables")

    # polishing: every evaluation recorded, the algorithm on the budget without
    # the polishing's share
    recorder, total_evaluations = None, int(max_evaluations or 0)
    if polish_how is not None:
        if layout is not None:
            raise ValueError("polish takes real variables only: no var_types other than 'real'")
        recorder = _polish.Recorder()
        fn = recorder.fn(fn)
        constraints = recorder.constraints(constraints)
        batch = recorder.batch(batch)
        max_evaluations = total_evaluations - int(round(polish_share * total_evaluations))

    problem = _core.Problem()
    problem.bounds = bounds
    problem.n_objectives = int(n_objs)
    problem.n_bin = n_bin

    if direct is not None:
        problem.native = direct
        problem.n_limits = n_cons = direct.n_con

        def rows_seen(X, F, G, _user=on_rows):
            points.seen.update(map(tuple, X.tolist()))      # result.unique_evaluations
            if _user is not None:
                _user(X, F, G)
        problem.on_rows = rows_seen
    elif constraints is not None:
        if not has_ineq:
            n_cons = n_eq                   # the equalities alone
        elif n_cons < 1:
            # Counted once rather than guessed per call: the binding needs the
            # width up front, and calling the user's function during setup to
            # find it out would spend an evaluation they did not ask for.
            probe = list(constraints([lo for lo, _ in bounds] + [0.0] * n_bin))
            n_cons = len(probe)             # with the equalities, if any
        else:
            n_cons = int(n_cons) + n_eq
        problem.n_limits = int(n_cons)
        if guard is not None:
            guard.n_cons = int(n_cons)

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

    def _polished(result):
        """The algorithm's result, polished when polish is on (scalar/polish.py)."""
        if polish_how is None:
            return result
        knobs_s = {k: v for k, v in knobs.items() if k in ("fd_step", "local_tol")}
        return _polish.run(
            result, recorder, how=polish_how, points=polish_points, fn=fn,
            constraints=constraints, bounds=bounds, n_objs=int(n_objs),
            n_cons=int(n_cons) if constraints is not None else 0, n_eq=n_eq,
            eq_tol=float(eq_tol), pop_size=int(pop_size),
            budget=max(0, total_evaluations - recorder.calls),
            on_generation=on_generation if int(record_every) > 0 else None,
            record_every=int(record_every), guard=guard,
            **{k: float(v) for k, v in knobs_s.items()})

    if algorithm in SCALAR_NAMES:
        # scalarization with a local solver (mootation.scalar): Python, not a core
        if layout is not None:
            raise ValueError(f"{algorithm} takes real variables only: no var_types other "
                             "than 'real'")
        if constraint_mode != "feasibility":
            raise ValueError(f"{algorithm} takes constraint_mode = 'feasibility' only")
        if init != "uniform":
            from . import designs
            if init not in designs.KINDS:
                raise ValueError(f"init = '{init}': uniform, or a design: "
                                 f"{', '.join(designs.KINDS)}")
            if seed_population is not None:
                raise ValueError("init and seed_population both give the first population")
        elif init_share:
            raise ValueError("init_share needs a design: init = 'sobol'")
        if int(max_evaluations) < 0:
            raise ValueError(f"max_evaluations must be >= 0, got {max_evaluations}")
        seeds = None
        if seed_population is not None:
            from .persistence import fit_population, load_population
            pop = (load_population(seed_population)
                   if isinstance(seed_population, (str, bytes)) or
                   hasattr(seed_population, "__fspath__") else seed_population)
            pop = fit_population(pop, int(pop_size), on_size_mismatch)
            seeds = [(list(x), list(f), list(constraints(list(x))) if constraints is not None
                      else []) for x, f in zip(pop.variables, pop.objectives)]
        from . import scalar
        try:
            result = scalar.run(
                algorithm, fn, constraints, bounds=bounds, n_objs=int(n_objs),
                n_cons=int(n_cons) if constraints is not None else 0, n_eq=n_eq,
                eq_tol=float(eq_tol),
                eq_tol0=float(eq_tol0) if eq_tol_schedule == "linear" else None,
                pop_size=int(pop_size),
                budget=int(max_evaluations) if max_evaluations else int(pop_size) * int(n_gen),
                seed=int(seed), knobs={k: v for k, v in knobs.items() if k not in POLISH_KNOBS},
                on_generation=on_generation, record_every=int(record_every),
                init=init, init_share=float(init_share), seed_points=seeds,
                unique_count=lambda: points.count, guard=guard)
            result = _polished(result)
        finally:
            if log is not None:
                log.close()
        if save_population is not None:
            from .persistence import save_population as _save
            _save(result, save_population,
                  meta={"algorithm": algorithm, "pop_size": pop_size,
                        "n_gen": n_gen, "seed": seed, "n_objs": n_objs})
        return result

    cfg = _core.Config()
    cfg.pop_size = int(pop_size)
    cfg.n_gen = int(n_gen)
    if max_evaluations:
        if int(max_evaluations) < 0:
            raise ValueError(f"max_evaluations must be >= 0, got {max_evaluations}")
        cfg.max_evaluations = int(max_evaluations)
    cfg.seed = int(seed)
    cfg.fail_raise = on_failure == "raise"
    cfg.unique_evaluations = lambda: points.count
    cfg.budget_unique = budget_counts == "unique"
    if constraint_mode not in ("feasibility", "eps_level", "none"):
        raise ValueError(f"constraint_mode = '{constraint_mode}': feasibility, eps_level or none")
    constrained = constraints is not None or (direct is not None and direct.n_con > 0)
    if constraint_mode == "eps_level" and not constrained:
        raise ValueError("constraint_mode = 'eps_level' needs constraints")
    if eps_schedule is not None and constraint_mode != "eps_level":
        raise ValueError("eps_schedule goes with constraint_mode = 'eps_level'")
    if constrained and constraint_mode != "none":     # none: the core ranks by the objectives
        cfg.constraint_mode = _core.ConstraintMode.FEASIBILITY
    if constraint_mode == "eps_level":
        # the vault's epsilon level on the feasibility-first ranking
        cfg.eps_level = True
        sched = dict(eps_schedule or {})
        bad = sorted(set(sched) - {"tau", "alpha", "theta", "tc"})
        if bad:
            raise ValueError(f"eps_schedule: unknown key(s) {bad}; tau, alpha, theta, tc")
        for k, v in sched.items():
            setattr(cfg, "eps_" + k, float(v))
    if n_eq:
        cfg.n_eq = n_eq
        cfg.eq_tol = float(eq_tol)
        if eq_tol_schedule == "linear":
            cfg.eq_linear = True
            cfg.eq_tol0 = float(eq_tol0)
    if layout is not None:
        cfg.var_kinds = layout.var_kinds()
        cfg.discrete_values = layout.discrete_values()
        cfg.gene_levels = layout.gene_levels()
        cfg.lamarck = int_repair == "lamarck"
        if any(k != 0 for k in cfg.var_kinds):
            cfg.eta_int = 3.0 if eta_int is None else float(eta_int)
    params, words = {}, {}
    for k, v in knobs.items():
        if k in POLISH_KNOBS:
            continue                        # minimize's own (task 9, A2)
        if k in PARAM_KNOBS:
            params[k] = float(v)
            continue
        if k in WORD_KNOBS:
            words[k] = ",".join(v) if isinstance(v, (list, tuple)) else str(v)
            continue
        try:
            setattr(cfg, k, v)
        except TypeError:
            # T, nr, K, n_clusters and div are integers in the binding, and
            # pybind11 refuses a float for them. A whole number arrives as a
            # float from JSON, from arithmetic or from a config layer, and it
            # means the integer; a fractional one is a mistake worth naming.
            if isinstance(v, float) and v.is_integer():
                setattr(cfg, k, int(v))
            elif isinstance(v, float):
                raise TypeError(f"parameter {k} takes an integer, got {v!r}") from None
            else:
                raise
    if params:
        cfg.params = params
    if words:
        cfg.words = words
    cfg.operator_stats = bool(operator_stats)
    if on_generation is not None and int(record_every) > 0:
        cfg.on_generation = on_generation
        cfg.record_every = int(record_every)

    if init != "uniform":
        # A design as the seed population: evaluated here, so that it costs
        # what the algorithm's own start costs; the core then plants it
        # without evaluating it again (setup_with_seed).
        from . import designs
        if init not in designs.KINDS:
            raise ValueError(f"init = '{init}': uniform, or a design: {', '.join(designs.KINDS)}")
        if seed_population is not None:
            raise ValueError("init and seed_population both give the first population")
        n_design = int(pop_size)
        if init_share:
            if not max_evaluations:
                raise ValueError("init_share is a share of max_evaluations, which is not given")
            n_design = max(n_design, int(round(float(init_share) * int(max_evaluations))))
        rows = [list(x) for x in designs.design(init, n_design, bounds, seed)]
        if direct is not None:
            import numpy as np
            XD = np.asarray(rows, dtype=float).reshape(len(rows), len(bounds))
            FD, GD = direct.evaluate_with_constraints(XD)
            problem.on_rows(XD, FD, GD)                    # the design's points, before the run
            F, G = FD.tolist(), GD.tolist()
        else:
            F = ([list(r) for r in batch(rows)] if batch is not None
                 else [list(fn(x)) for x in rows])
        pick = designs.start_points(F, int(pop_size))
        cfg.seed_variables = [rows[i] for i in pick]
        cfg.seed_objectives = [F[i] for i in pick]
        if direct is not None and direct.n_con > 0:
            cfg.seed_limits = [G[i] for i in pick]
        elif constraints is not None:
            cfg.seed_limits = [list(constraints(rows[i])) for i in pick]
        if max_evaluations:
            left = int(max_evaluations) - n_design
            if left <= 0:
                raise ValueError(f"the design takes {n_design} of the {max_evaluations} "
                                 f"evaluations and leaves none to {algorithm}")
            # the core counts its own calls; distinct points include the design's
            cfg.max_evaluations = left if budget_counts == "calls" else int(max_evaluations)
    elif init_share:
        raise ValueError("init_share needs a design: init = 'sobol'")

    if seed_population is not None:
        from .persistence import fit_population, load_population
        pop = (load_population(seed_population)
               if isinstance(seed_population, (str, bytes)) or
               hasattr(seed_population, "__fspath__")
               else seed_population)
        if layout is not None and len(pop.variables[0]) == len(layout.kinds) and \
                not getattr(pop, "binary", None):
            # the caller's variables, bits included: the core's reals and bits
            parts = [layout.split(x) for x in pop.variables]
            from .persistence import LoadedPopulation
            pop = LoadedPopulation([p[0] for p in parts], pop.objectives, pop.cv, pop.meta,
                                   [p[1] for p in parts])
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
        if n_bin:
            if not getattr(pop, "binary", None):
                raise ValueError("the seed population carries no bits for the binary variables")
            cfg.seed_binary = [list(b) for b in pop.binary]
        cfg.seed_objectives = [list(v) for v in pop.objectives]
        if constraints is not None:
            # The population file carries x, f and the aggregate cv only; the
            # algorithms compare the individual constraint values, so they are
            # recomputed here from `constraints`. That costs one constraint
            # call per seed individual and no objective evaluations. With
            # binary or categorical variables `constraints` takes the core's
            # whole genome, the reals and then the integer genome, as the core
            # hands a point to it (var_types.Layout.to_user).
            genes = pop.binary if n_bin else [[]] * len(pop.variables)
            cfg.seed_limits = [list(constraints(list(x) + list(b)))
                               for x, b in zip(pop.variables, genes)]

    try:
        result = _polished(_core.run(algorithm, problem, cfg))
    except ValueError as e:
        if n_bin and ("binary" in str(e) or "continuous" in str(e)):
            # the 24 cores without an integer genome refuse it by name
            raise ValueError(f"{algorithm}: incompatible by construction: takes no binary or "
                             f"categorical variables (the integer genome; docs/running.md "
                             f"lists the cores that do): {e}") from e
        raise
    finally:
        if log is not None:
            log.close()

    if save_population is not None:
        from .persistence import save_population as _save
        _save(result, save_population,
              meta={"algorithm": algorithm, "pop_size": pop_size,
                    "n_gen": n_gen, "seed": seed, "n_objs": n_objs})
    # the result's variables are the core's real genome (integer and discrete
    # values as floats), its binary_variables the integer genome: bits, and
    # categorical variables' label indices (var_types.py; Layout.to_user)
    return result

# SPDX-License-Identifier: Apache-2.0
"""Scalarization with a local solver (task 9 of 2026-10-18, part A1): the
front as a sequence of single-objective subproblems, each solved by a local
method. Not evolutionary, and not a C++ core: the solvers are SciPy's (BSD-3,
called as a dependency) and a CMA-ES written from Hansen's tutorial, every
evaluation still goes through fn, and minimize() runs these algorithms by
name like any other.

THE ALGORITHM `scalar`, and its presets under their own names:

  eps_sqp  tch_sqp  nbi_sqp  nnc_sqp  eps_nm  eps_cmaes

  scalarization  eps | tch | nbi | nnc | ps            (subproblems.py)
  local          sqp | nm | cmaes                      (solvers.py)
  sub_plan       grid | laumanns | gap
  tch_weights    inverse | lattice: tch's weights from the lattice point d,
                 1/d (the rays F̄ ∝ d, SCALAR-3) or d itself (the plain grid)
  n_sub          subproblems after the anchors; 0: from the budget (SCALAR-1)
  sub_budget     evaluations a subproblem may take; 0: what is left over the
                 subproblems left, so that what one does not use the next get
  fd_step        sqp's forward-difference step in the normalized box (1.49e-8)
  local_tol      the solvers' tolerance (1e-14)
  cma_sigma0     cmaes's initial step in the normalized box (0.3; the
                 tutorial's 0.3 (b - a), footnote to Figure 6)
  eps_objective  the objective eps minimizes, 1-based (1)

THE PLAN.
  1. The initial plan: pop_size points drawn uniformly from the seed, or the
     design of `init` (with `init_share`), or the seed population — the
     points the first subproblems start from.
  2. The anchors: for every objective k, the lexicographic minimum of
     (f_k, f_{k+1}, ..., f_{k-1}) — min f_k, then min f_{k+1} with f_k no
     worse, and so on (SCALAR-9). They give the normalization (z*, z^N;
     subproblems.Frame).
  3. The subproblems of the plan, each started from the archive's best point
     for it (feasible first, then its value): with sub_plan =
       grid      the lattice, in lattice order (neighbours one after another:
                 NBI's "continuation", Das & Dennis §5.3);
       laumanns  eps only: the adaptive scheme of Laumanns, Thiele & Zitzler
                 (Dagstuhl 04461, 2005, Fig. 1-2; EJOR 169(3), 2006): the
                 constraint levels come from the points already found, every
                 cell is searched once, and an empty or dominated cell is
                 never searched again; a cell yields a point by two runs, min
                 f_k, then min the others with f_k no worse. On a continuous
                 problem the strict bound is f_j <= e_j - δ with δ = 1/(L + 1)
                 of the normalized range, L the grid's levels per objective
                 (the increments §6 of the paper asks for there; SCALAR-10);
       gap       OUR rule, not Laumanns's: of a lattice four times finer, the
                 subproblem whose parameter lies farthest from those of the
                 points found so far (subproblems.coords), one after another.
  4. The answer: the archive's feasible non-dominated points, reduced to
     pop_size by DSS (evaluator.py). The run ends when the plan is done or the
     budget spent, whichever comes first: a plan that ends early leaves
     evaluations unspent, as DMS does.

CONSTRAINTS. sqp takes the problem's inequalities (with a margin of 1e-10:
SCALAR-8) and equalities directly; nm and cmaes through the penalty in which a
feasible point is always better (solvers.py, SCALAR-2). Only the feasibility
rule: constraint_mode = eps_level is refused. Real variables only.

READINGS (declared):
  SCALAR-1. n_sub from the budget: what is left after the plan's start and the
    anchors, over 10 (n + 1) evaluations a subproblem for sqp — about ten
    iterations from a neighbour's solution — and 100 n for nm and cmaes; at
    most pop_size, at least 1. A subproblem may take what is left over the
    subproblems left; an anchor's stage at least a quarter of the budget left
    after the initial plan over the m² stages — the frame rests on them.
  SCALAR-2. nm, cmaes: an infeasible point is worth the worst feasible value
    known for its subproblem plus its violation (Deb's rule as a penalty).
  SCALAR-3. tch: w_i = 1 / max(d_i, 1e-6) for the lattice's direction d, so
    that the solutions lie on the rays F̄ ∝ d the lattice spreads evenly;
    the reference point is z*.
  SCALAR-4. nnc: the normal constraints (Ā_m − Ā_i)ᵀ(F̄ − X̄) <= 0 and min F̄_m,
    as the materials of task 9 give the 2003 paper (its text was not
    available). The 2026 summary's Eq. (80), (F̄ − X̄)ᵀ(A_i − X̄) <= 0 with
    min F̄_k, leads at m = 2 and k = m straight to the anchor of f_m, so it is
    not read as the method.
  SCALAR-5. ps: ā on the anchors' simplex, r its quasi-normal reversed (NBI's
    geometry; Eichfelder's adaptive choice of ā could not be checked against
    her texts and is not implemented).
  SCALAR-6. sqp stops a subproblem at a failed evaluation of an iterate, and
    at a finite-difference step that fails both ways.
  SCALAR-7. cmaes: a point outside the box is evaluated at its clip, plus the
    squared distance to it times (1 + |value|); one run per subproblem, until
    σ·max D < local_tol or the allowance is spent.
  SCALAR-8. sqp asks g(x) <= -1e-10 of the problem's inequalities, so that a
    solution on the boundary is feasible by the vault's rule (g <= 0).
  SCALAR-9. The anchors are lexicographic minima in cyclic order: ours. Das &
    Dennis and Messac et al. take the individual minima as they come, which
    on a front with a degenerate edge (DTLZ2: (0, 0, 1) minimizes f1 and f2
    alike) makes two anchors one and the simplex flat; a stage keeps the
    earlier objectives within 1e-9 (relative) of their minima. A stage runs
    from the archive's best point for it and, when it differs, from the
    point of least value too: the best feasible point can sit where the
    gradient vanishes (DTLZ2's corner x1 = 1 for f3), and sqp never leaves it.
  SCALAR-10. laumanns: the grid's first bound is +∞ (no constraint), the
    mirror of the paper's lower bound 0 for maximization.
  SCALAR-12. The anchors' stages see the objectives scaled by the initial
    plan's range (its minimum to 0, its maximum to 1): SLSQP stops early on a
    raw objective of 1e7 (bbob-biobj f2's ellipsoid).
  SCALAR-11. A subproblem whose solver ends where it started runs again, with
    what is left of its allowance, from the initial plan's best point for it:
    the archive's best is often an anchor or a neighbour's solution at a
    corner of the parameterization where the gradients vanish (DTLZ's
    x_i in {0, 1}), and a gradient method stays there.
"""

from __future__ import annotations

import math

PRESETS = {
    "scalar": {},
    "eps_sqp": {"scalarization": "eps", "local": "sqp"},
    "tch_sqp": {"scalarization": "tch", "local": "sqp"},
    "nbi_sqp": {"scalarization": "nbi", "local": "sqp"},
    "nnc_sqp": {"scalarization": "nnc", "local": "sqp"},
    "eps_nm": {"scalarization": "eps", "local": "nm"},
    "eps_cmaes": {"scalarization": "eps", "local": "cmaes"},
}
# Python-level algorithms of task 9 beside the scalarization methods, on the
# same evaluator, answer and observer (boostdms.py)
OTHERS = ("boostdms",)
NAMES = tuple(PRESETS) + OTHERS

WORDS = {
    "scalarization": ("eps", "tch", "nbi", "nnc", "ps"),
    "tch_weights": ("inverse", "lattice"),
    "local": ("sqp", "nm", "cmaes"),
    "sub_plan": ("grid", "laumanns", "gap"),
}
NUMBERS = {"n_sub": 0, "sub_budget": 0, "fd_step": 1.4901161193847656e-08,
           "local_tol": 1e-14, "cma_sigma0": 0.3, "eps_objective": 1}
DEFAULTS = {"scalarization": "eps", "local": "sqp", "sub_plan": "grid",
            "tch_weights": "inverse", **NUMBERS}


class ScalarResult:
    """Shaped like the binding's Result."""

    def __init__(self, ev, idx, ignored, settings, n_failed, eq_tol, unique, solutions=()):
        self.objectives = [list(map(float, ev.F[i])) for i in idx]
        self.variables = [list(map(float, ev.X[i])) for i in idx]
        self.cv = [float(ev.cv(i)) for i in idx]
        self.active_n = len(idx)
        self.ignored = list(ignored)
        self.evaluations = int(ev.spent)
        self.unique_evaluations = int(unique)
        self.n_failed = int(n_failed)
        self.eq_tol = float(eq_tol)
        self.operators = []
        self.settings = list(settings)
        self.operator_totals = {}
        self.inner_archive = []
        self.inner_archive_variables = []
        self.binary_variables = []
        # the objectives of every subproblem's solution, anchors first, in the
        # order solved (feasible ones; a diagnostic, not the answer)
        self.solutions = [list(map(float, ev.F[i])) for i in solutions]


def settings_of(name: str, knobs: dict) -> dict:
    """The preset's settings with the knobs given over them, checked."""
    s = dict(DEFAULTS)
    s.update(PRESETS[name])
    for k, v in knobs.items():
        if k in WORDS:
            if v not in WORDS[k]:
                raise ValueError(f"{k} = '{v}': {', '.join(WORDS[k])}")
            s[k] = v
        elif k in NUMBERS:
            s[k] = float(v)
    if s["n_sub"] < 0 or s["n_sub"] != math.floor(s["n_sub"]):
        raise ValueError("n_sub must be a whole number >= 0 (0: from the budget)")
    if s["sub_budget"] < 0 or s["sub_budget"] != math.floor(s["sub_budget"]):
        raise ValueError("sub_budget must be a whole number >= 0 (0: a share of what is left)")
    for k in ("fd_step", "local_tol", "cma_sigma0"):
        if not s[k] > 0.0:
            raise ValueError(f"{k} must be > 0")
    if s["sub_plan"] == "laumanns" and s["scalarization"] != "eps":
        raise ValueError("sub_plan = 'laumanns' is a scheme of the eps-constraint method: "
                         "scalarization = 'eps'")
    return s


def _simplex(m: int, count: int):
    """Das-Dennis lattice points without the vertices (the anchors), at least
    `count` of them (the smallest H that gives as many)."""
    from math import comb
    import numpy as np
    from ..run.metrics import das_dennis
    if m < 2:
        return np.zeros((0, m))
    h = 2
    while comb(h + m - 1, m - 1) - m < count:
        h += 1
    L = das_dennis(m, h)
    return L[np.max(L, axis=1) < 1.0]


def _levels(m: int, count: int):
    """eps: per constrained objective L levels j/(L + 1), L^(m-1) >= count."""
    import numpy as np
    if m < 2:
        return np.zeros((0, 0)), 1
    L = max(1, int(math.ceil(count ** (1.0 / (m - 1)) - 1e-9)))
    axis = np.arange(1, L + 1) / (L + 1)
    grids = np.meshgrid(*([axis] * (m - 1)), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1), L


def run(name, fn, constraints, *, bounds, n_objs: int, n_cons: int, n_eq: int, eq_tol: float,
        eq_tol0, pop_size: int, budget: int, seed: int, knobs: dict, on_generation=None,
        record_every: int = 0, init: str = "uniform", init_share: float = 0.0,
        seed_points=None, unique_count=None, guard=None):
    import numpy as np
    from . import solvers as S
    from . import subproblems as P
    from .evaluator import BudgetSpent, Evaluator, SubBudgetSpent

    if name == "boostdms":
        return _run_boostdms(fn, constraints, bounds=bounds, n_objs=n_objs, n_cons=n_cons,
                             n_eq=n_eq, eq_tol=eq_tol, eq_tol0=eq_tol0, pop_size=pop_size,
                             budget=budget, seed=seed, knobs=knobs, on_generation=on_generation,
                             record_every=record_every, init=init, init_share=init_share,
                             seed_points=seed_points, unique_count=unique_count, guard=guard)
    known = set(WORDS) | set(NUMBERS)
    ignored = sorted(k for k in knobs if k not in known)
    s = settings_of(name, {k: v for k, v in knobs.items() if k in known})
    m, n = int(n_objs), len(bounds)
    k_eps = int(s["eps_objective"]) - 1
    if not 0 <= k_eps < m:
        raise ValueError(f"eps_objective = {s['eps_objective']:g}: 1 to {m}")
    rng = np.random.default_rng(int(seed))
    ev = Evaluator(fn, constraints, bounds=bounds, n_objs=m, n_cons=n_cons, n_eq=n_eq,
                   eq_tol=eq_tol, eq_tol0=eq_tol0, budget=budget, pop_size=pop_size,
                   on_generation=on_generation, record_every=record_every)
    lo, hi = ev.lo, ev.hi
    width = hi - lo

    def best_start(sub, frame, by_value=False, among=None):
        """The archive's best point for sub: feasible first, then its value; or,
        by_value, the least value whatever it violates. among: these indices
        only (the initial plan's)."""
        pool = range(len(ev.X)) if among is None else among
        ok = [i for i in pool if not ev.failed[i]]
        if not ok:
            return None
        FB = frame.bar(np.array([ev.F[i] for i in ok]))
        value, viol = sub.merits(FB)
        if by_value:
            return ok[int(np.argmin(value))]
        viol = np.where(viol > S.SLACK, viol, 0.0) + np.array([ev.cv(i) for i in ok])
        feas = viol <= 0.0
        if feas.any():
            j = int(np.flatnonzero(feas)[np.argmin(value[feas])])
        else:
            j = int(np.argmin(viol))
        return ok[j]

    plan = {"left": 0}                       # subproblems still to come, this one included
    solutions = []

    def solve(sub, frame, two_starts=False, floor=0):
        """One subproblem: its start, its allowance, its solver; its best point.
        two_starts (the anchors): from the best point and from the one of least
        value too, when they differ (SCALAR-9)."""
        starts = [best_start(sub, frame)]
        if starts[0] is None:
            return None
        if two_starts:
            other = best_start(sub, frame, by_value=True)
            if other != starts[0]:
                starts.append(other)
        left = ev.budget - ev.spent
        share = left // max(1, plan["left"])
        allowance = (int(min(left, s["sub_budget"])) if s["sub_budget"] > 0
                     else int(min(left, max(share, floor))))
        plan["left"] = max(0, plan["left"] - 1)
        end = ev.spent + max(1, allowance)
        run_ = S.Run(sub, ev, frame)

        def from_(start, limit):
            ev.limit = limit
            run_.touched.append(start)
            y0 = (ev.X[start] - lo) / width
            if s["local"] == "sqp":
                S.sqp(run_, y0, sub.t_star(frame.bar(ev.F[start])), fd_step=s["fd_step"],
                      tol=s["local_tol"])
            elif s["local"] == "nm":
                S.nm(run_, y0, [start], tol=s["local_tol"])
            else:
                S.cmaes(run_, y0, [start], sigma0=s["cma_sigma0"], tol=s["local_tol"], rng=rng)

        try:
            for k, start in enumerate(starts):
                # the second start gets what the first left of the allowance
                from_(start, end if k == len(starts) - 1 else
                      ev.spent + max(1, (end - ev.spent) // 2))
            if not S.moved(run_, starts, S.best_of(run_, run_.touched)) and ev.spent < end:
                # not moved from where it started: again from the initial plan's
                # best point for this subproblem (SCALAR-11)
                again = best_start(sub, frame, among=plan["initial"])
                if again is not None and again not in starts:
                    from_(again, end)
        finally:
            ev.limit = ev.budget
            best = S.best_of(run_, run_.touched)
        return best

    def found(sub, frame, i):
        """Did subproblem sub yield a feasible point at i?"""
        if i is None or ev.failed[i] or ev.cv(i) > 0.0:
            return False
        return sub.merit(frame.bar(ev.F[i]))[1] <= S.SLACK

    try:
        # 1. the initial plan
        if seed_points is not None:
            for x, f, g in seed_points:
                ev.plant(np.asarray(x, float), f, g)
        else:
            if init != "uniform":
                from .. import designs
                n0 = pop_size
                if init_share:
                    n0 = max(n0, int(round(float(init_share) * budget)))
                rows = designs.design(init, n0, bounds, seed)
            else:
                rows = lo + rng.random((pop_size, n)) * width
            for x in rows:
                ev.index(np.asarray(x, float))
        plan["initial"] = list(range(len(ev.X)))
        if not ev.recorded:
            ev.record()                      # gen 0, unless the plan's last point gave it

        # 2. the anchors, in the scale of the initial plan (SCALAR-12)
        ok = [i for i in plan["initial"] if not ev.failed[i]]
        if ok:
            FP = np.array([ev.F[i] for i in ok])
            identity = P.Frame.from_anchors(np.vstack([FP.min(axis=0), FP.max(axis=0)]))
        else:
            identity = P.Frame(m)
        k_sub = int(s["n_sub"])
        cost = 10 * (n + 1) if s["local"] == "sqp" else 100 * n
        if k_sub == 0:
            k_sub = int(min(pop_size, max(1, (ev.budget - ev.spent) // cost - m * m)))
        s["n_sub"] = float(k_sub)
        plan["left"] = m * m + k_sub
        floor = (ev.budget - ev.spent) // (4 * m * m)       # SCALAR-1
        anchors = []
        for k in range(m):
            # lexicographic, by cycle: f_k, then f_{k+1}, ..., each stage keeping
            # the ones before no worse (SCALAR-9)
            rows, best = [], None
            for o in [(k + r) % m for r in range(m)]:
                a0 = np.zeros(m)
                a0[o] = 1.0
                stage = P.Sub("lexi", a0, ineq=rows)
                b = solve(stage, identity, two_starts=True, floor=floor)
                if b is None or not found(stage, identity, b):
                    break
                best = b
                fo = float(identity.bar(ev.F[b])[o])
                bound = np.zeros(m)
                bound[o] = -1.0
                rows = rows + [(bound, 0.0, fo + 1e-9 * max(1.0, abs(fo)))]
            if best is not None:
                anchors.append(ev.F[best])
                solutions.append(best)
        if len(anchors) < m:
            raise BudgetSpent
        frame = P.Frame.from_anchors(np.array(anchors))

        # 3. the plan
        kind = s["scalarization"]

        def build(p):
            if kind == "eps":
                return P.eps(m, k_eps, p)
            if kind == "tch":
                return P.tch(m, p, s["tch_weights"])
            if kind == "nbi":
                return P.nbi(frame, p)
            if kind == "ps":
                return P.ps(frame, p)
            return P.nnc(frame, p)

        def params(count):
            return _levels(m, count)[0] if kind == "eps" else _simplex(m, count)

        if m == 1:
            pass
        elif s["sub_plan"] == "grid":
            grid = params(k_sub)
            plan["left"] = len(grid)
            for p in grid:
                sub = build(p)
                i = solve(sub, frame)
                if found(sub, frame, i):
                    solutions.append(i)
        elif s["sub_plan"] == "gap":
            cand = list(params(4 * k_sub))
            got = [frame.bar(a) for a in anchors]
            plan["left"] = k_sub
            for _ in range(min(k_sub, len(cand))):
                C = np.array(cand)
                have = P.coords(kind, frame, np.array(got), k_eps)
                d = np.min(np.linalg.norm(C[:, None, :] - have[None, :, :], axis=2), axis=1)
                j = int(np.argmax(d))
                sub = build(cand.pop(j))
                i = solve(sub, frame)
                if found(sub, frame, i):
                    got.append(frame.bar(ev.F[i]))
                    solutions.append(i)
        else:
            _laumanns(m, k_eps, k_sub, frame, solve, found, ev, P, plan, solutions)
    except (BudgetSpent, SubBudgetSpent):
        pass

    ev.over = True
    idx = ev.answer()
    ev.record(final=True)
    settings = [(k, f"{v:g}" if isinstance(v, float) else str(v)) for k, v in s.items()]
    n_failed = guard.n_failed if guard is not None else int(sum(ev.failed))
    unique = unique_count() if unique_count is not None else ev.spent
    return ScalarResult(ev, idx, ignored, settings, n_failed,
                        ev.eq_tol_end if n_eq else 0.0, unique, solutions)


def _run_boostdms(fn, constraints, *, bounds, n_objs, n_cons, n_eq, eq_tol, eq_tol0, pop_size,
                  budget, seed, knobs, on_generation, record_every, init, init_share,
                  seed_points, unique_count, guard):
    """boostdms.py on the evaluator: knobs, start, answer."""
    import numpy as np
    from .boostdms import run_boostdms
    from .evaluator import BudgetSpent, Evaluator
    ignored = sorted(k for k in knobs if k != "boost_scalarization")
    how = str(knobs.get("boost_scalarization", "eps"))
    if how not in ("eps", "minmax"):
        raise ValueError(f"boost_scalarization = '{how}': eps or minmax")
    ev = Evaluator(fn, constraints, bounds=bounds, n_objs=n_objs, n_cons=n_cons, n_eq=n_eq,
                   eq_tol=eq_tol, eq_tol0=eq_tol0, budget=budget, pop_size=pop_size,
                   on_generation=on_generation, record_every=record_every)
    start = None
    try:
        if seed_points is not None:
            start = [ev.plant(np.asarray(x, float), f, g) for x, f, g in seed_points]
        elif init != "uniform":
            from .. import designs
            n0 = max(pop_size, int(round(float(init_share) * budget))) if init_share else pop_size
            start = [ev.index(np.asarray(x, float)) for x in designs.design(init, n0, bounds, seed)]
    except BudgetSpent:
        pass
    idx = run_boostdms(ev, n_objs=n_objs, pop_size=pop_size, how=how, start=start)
    ev.record(final=True)
    n_failed = guard.n_failed if guard is not None else int(sum(ev.failed))
    unique = unique_count() if unique_count is not None else ev.spent
    return ScalarResult(ev, idx, ignored, [("boost_scalarization", how)], n_failed,
                        ev.eq_tol_end if n_eq else 0.0, unique)


def _laumanns(m, k, k_sub, frame, solve, found, ev, P, plan, solutions):
    """Laumanns, Thiele & Zitzler's adaptive ε-constraint scheme (Fig. 1 and
    2), for minimization in the normalized objectives: the bounds are upper
    ones, sorted from +∞ down, and a cell's strict bound f_j < e_j is
    f_j <= e_j - δ."""
    import numpy as np
    others = [j for j in range(m) if j != k]
    L = _levels(m, k_sub)[1]
    delta = 1.0 / (L + 1)
    e = {j: [math.inf] for j in others}          # descending: index 0 is +∞
    P_found, Q, R = [], [], []
    plan["left"] = k_sub
    while True:
        p = len(P_found)
        i = (p + 1) ** (m - 1)
        new = False
        while i >= 0:
            eps, r = [], i
            for j in others:                     # getBounds
                d = r % (p + 1)
                r //= p + 1
                eps.append(e[j][d])
            eps_t = tuple(eps)
            skip = any(all(qq >= ee for qq, ee in zip(q, eps)) for q in Q) or eps_t in R
            if not skip:
                levels = [x - delta for x in eps]
                first = P.eps(m, k, levels)
                y = solve(first, frame)
                Fy = frame.bar(ev.F[y]) if y is not None else None
                if not found(first, frame, y) or any(np.all(z <= Fy) for z in P_found):
                    Q.append(eps_t)
                else:
                    a0 = np.ones(m)
                    a0[k] = 0.0
                    bound = np.zeros(m)
                    bound[k] = -1.0
                    second = P.Sub("eps", a0, ineq=first.ineq + [(bound, 0.0, float(Fy[k]))])
                    x = solve(second, frame)
                    if not found(second, frame, x):
                        x = y
                    Fx = frame.bar(ev.F[x])
                    P_found.append(Fx)
                    solutions.append(x)
                    R.append(eps_t)
                    for j in others:             # updateBounds
                        e[j].append(float(Fx[j]))
                        e[j].sort(reverse=True)
                    new = True
                    break
            i -= 1
        if not new:
            return

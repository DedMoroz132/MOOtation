# SPDX-License-Identifier: Apache-2.0
"""Polishing at the end of the budget (task 9, A2): any algorithm, then a local
solver on some of the points it found.

  polish         off (the default: nothing changes) | sqp | nm
  polish_share   the share of max_evaluations kept for the polishing (0.2);
                 the algorithm runs on the rest
  polish_points  how many points to polish; 0: from what the share buys, as
                 n_sub (scalar, SCALAR-1), at most pop_size

THE POINTS are the feasible non-dominated points of every evaluation the
algorithm made, chosen by DSS (run/archive.py) when there are more; without a
feasible point, the least violating ones.

POLISHING a point a is the ε-constraint subproblem "improve one objective,
the others no worse than at a": min f_k  s.t.  f_i <= f_i(a) for every i,
started at a, with k going round the objectives from one point to the next
(POLISH-1). Every point it evaluates goes to the archive, and its result is
never worse than a: a itself is feasible for it.

THE ANSWER is the non-dominated points of the algorithm's answer and the
polished points, reduced to pop_size by DSS when there are more (POLISH-2):
a polished point replaces what it dominates and nothing is lost. The observer
goes on after the algorithm's last record, every record_every * pop_size
evaluations.

READINGS:
  POLISH-1. The objective improved goes round the objectives (the first point
    f_1, the second f_2, ...) in DSS order; the objectives are normalized by
    the range of the points chosen.
  POLISH-2. The answer after polishing is the non-dominated union of the
    algorithm's answer and the polished points, not the archive's.
"""

from __future__ import annotations

import math


class Recorder:
    """Every point fn and constraints are called on, with what they returned."""

    def __init__(self):
        self.f, self.g, self.order = {}, {}, []
        self.calls = 0

    def fn(self, fn):
        def wrapped(x):
            y = fn(x)
            self.calls += 1
            key = tuple(float(v) for v in x)
            if key not in self.f:
                self.order.append(key)
            self.f[key] = [float(v) for v in y]
            return y
        return wrapped

    def constraints(self, constraints):
        if constraints is None:
            return None

        def wrapped(x):
            g = constraints(x)
            self.g[tuple(float(v) for v in x)] = [float(v) for v in g]
            return g
        return wrapped

    def batch(self, batch):
        if batch is None:
            return None

        def wrapped(X):
            Y = batch(X)
            for x, y in zip(X, Y):
                self.calls += 1
                key = tuple(float(v) for v in x)
                if key not in self.f:
                    self.order.append(key)
                self.f[key] = [float(v) for v in y] if y is not None else [math.nan]
            return Y
        return wrapped

    def points(self):
        return [(list(k), self.f[k], self.g.get(k)) for k in self.order]


class PolishedResult:
    """The binding's Result, after polishing."""

    def __init__(self, base, ev, idx, extra_evals, n_failed, settings):
        self.objectives = [list(map(float, ev.F[i])) for i in idx]
        self.variables = [list(map(float, ev.X[i])) for i in idx]
        self.cv = [float(ev.cv(i)) for i in idx]
        self.active_n = len(idx)
        self.ignored = list(base.ignored)
        self.evaluations = int(extra_evals)
        self.unique_evaluations = getattr(base, "unique_evaluations", -1)
        self.n_failed = int(n_failed)
        self.eq_tol = float(base.eq_tol)
        self.operators = list(getattr(base, "operators", []))
        self.settings = list(getattr(base, "settings", [])) + list(settings)
        self.operator_totals = dict(getattr(base, "operator_totals", {}))
        self.inner_archive = list(getattr(base, "inner_archive", []))
        self.inner_archive_variables = list(getattr(base, "inner_archive_variables", []))
        self.binary_variables = []


def check(knobs: dict, max_evaluations: int):
    """(how, share, points) from the knobs, checked; how is None when off."""
    how = str(knobs.get("polish", "off"))
    if how not in ("off", "sqp", "nm"):
        raise ValueError(f"polish = '{how}': off, sqp or nm")
    share = float(knobs.get("polish_share", 0.2))
    points = float(knobs.get("polish_points", 0))
    if how == "off":
        return None, share, points
    if not max_evaluations:
        raise ValueError("polish takes a share of max_evaluations, which is not given")
    if not 0.0 < share < 1.0:
        raise ValueError(f"polish_share = {share}: in (0, 1)")
    if points < 0 or points != math.floor(points):
        raise ValueError("polish_points must be a whole number >= 0 (0: from the budget)")
    return how, share, int(points)


def run(base, recorder, *, how, points, fn, constraints, bounds, n_objs, n_cons, n_eq,
        eq_tol, pop_size, budget, on_generation=None, record_every=0, guard=None,
        fd_step=1.4901161193847656e-08, local_tol=1e-14):
    import numpy as np
    from . import solvers as S
    from . import subproblems as P
    from .evaluator import BudgetSpent, Evaluator, SubBudgetSpent

    m, n = int(n_objs), len(bounds)
    ev = Evaluator(fn, constraints, bounds=bounds, n_objs=m, n_cons=n_cons, n_eq=n_eq,
                   eq_tol=eq_tol, eq_tol0=None, budget=budget, pop_size=pop_size,
                   on_generation=on_generation, record_every=record_every)
    ev.recorded = True                                # the algorithm has had gen 0
    ev.gen0 = -(-int(base.evaluations) // max(1, pop_size))
    for x, f, g in recorder.points():
        ev.plant(np.asarray(x, float), f, g if g is not None else [])
    base_idx = [ev.where[np.ascontiguousarray(x, dtype=float).tobytes()]
                for x in base.variables
                if np.ascontiguousarray(x, dtype=float).tobytes() in ev.where]
    chosen = ev.answer(None)
    k_pts = points or int(min(pop_size, max(1, budget // (10 * (n + 1)))))
    chosen = chosen[:k_pts]
    polished = []
    if chosen:
        FC = np.array([ev.F[i] for i in chosen])
        frame = P.Frame.from_anchors(np.vstack([FC.min(axis=0), FC.max(axis=0)]))
        lo, width = ev.lo, ev.hi - ev.lo
        try:
            for j, a in enumerate(chosen):
                k = j % m
                Fa = frame.bar(ev.F[a])
                rows = []
                for i in range(m):
                    row = np.zeros(m)
                    row[i] = -1.0
                    rows.append((row, 0.0, float(Fa[i])))
                a0 = np.zeros(m)
                a0[k] = 1.0
                sub = P.Sub("polish", a0, ineq=rows)
                left = ev.budget - ev.spent
                ev.limit = ev.spent + max(1, left // (len(chosen) - j))
                run_ = S.Run(sub, ev, frame)
                run_.touched.append(a)
                y0 = (ev.X[a] - lo) / width
                try:
                    if how == "sqp":
                        S.sqp(run_, y0, 0.0, fd_step=fd_step, tol=local_tol)
                    else:
                        S.nm(run_, y0, [a], tol=local_tol)
                except SubBudgetSpent:
                    pass
                finally:
                    ev.limit = ev.budget
                polished.append(S.best_of(run_, run_.touched))
        except BudgetSpent:
            pass
    ev.over = True
    # the answer: the algorithm's answer and the polished points, non-dominated
    pool = list(dict.fromkeys(base_idx + [i for i in polished if i is not None]))
    ok = [i for i in pool if not ev.failed[i]]
    feas = [i for i in ok if ev.cv(i) <= 0.0]
    if feas:
        F = np.array([ev.F[i] for i in feas])
        nd = [feas[a] for a in range(len(feas))
              if not np.any(np.all(F <= F[a], axis=1) & np.any(F < F[a], axis=1))]
        nd = list(dict.fromkeys(nd))
        if len(nd) > pop_size:
            from ..run.archive import dss_order
            nd = [nd[a] for a in dss_order(np.array([ev.F[i] for i in nd]), pop_size)]
        idx = nd
    else:
        idx = sorted(ok, key=lambda i: (ev.cv(i), i))[:pop_size]
    if ev.observer is not None:
        F = [list(map(float, ev.F[i])) for i in idx]
        X = [list(map(float, ev.X[i])) for i in idx]
        ev.observer(ev.gen0 + -(-ev.spent // max(1, pop_size)), F, X)
    # the guard counts the failures of the algorithm and of the polishing alike
    n_failed = guard.n_failed if guard is not None else int(base.n_failed)
    settings = [("polish", how), ("polish_points", str(len(chosen))),
                ("polish_evaluations", str(ev.spent))]
    return PolishedResult(base, ev, idx, recorder.calls, n_failed, settings)

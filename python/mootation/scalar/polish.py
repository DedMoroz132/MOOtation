# SPDX-License-Identifier: Apache-2.0
"""Polishing at the end of the budget (task 9, A2): any algorithm, then a local
solver on some of the points it found. In C++ since task 17, stage 6
(include/mootation/scalar/polish.hpp, _core.scalar.polish); here the knobs'
check, the recording of the algorithm's evaluations and the result.

  polish         off (the default: nothing changes) | sqp | nm
  polish_share   the share of max_evaluations kept for the polishing (0.2);
                 the algorithm runs on the rest
  polish_points  how many points to polish; 0: from what the share buys, as
                 n_sub (scalar, SCALAR-1), at most pop_size

THE POINTS are the feasible non-dominated points of every evaluation the
algorithm made, chosen by DSS (dss.hpp) when there are more; without a
feasible point, the least violating ones.

POLISHING a point a is the ε-constraint subproblem "improve one objective,
the others no worse than at a": min f_k  s.t.  f_i <= f_i(a) for every i,
started at a, with k going round the objectives from one point to the next
(POLISH-1). Every point it evaluates goes to the archive; a itself is
feasible for it, and its result is no worse than a up to the subproblem's
slack (POLISH-3).

THE ANSWER is the non-dominated points of the algorithm's answer and the
polished points; when there are more than pop_size, first one point weakly
dominating each point of the algorithm's answer, then the rest in DSS order
(POLISH-2). A polished point replaces what it dominates and nothing of the
algorithm's answer is lost. The observer goes on after the algorithm's last
record, every record_every * pop_size evaluations.

READINGS:
  POLISH-1. The objective improved goes round the objectives (the first point
    f_1, the second f_2, ...) in DSS order; the objectives are normalized by
    the range of the points chosen.
  POLISH-2. The answer after polishing is the non-dominated union of the
    algorithm's answer and the polished points, not the archive's; cut to
    pop_size, it keeps a point weakly dominating each point of the
    algorithm's answer first (so it can hold more than pop_size when the
    algorithm's answer does).
  POLISH-3. The subproblem's constraints f_i <= f_i(a) hold to a slack
    (SLACK, solvers.hpp, in the normalized objectives): a result a hair worse
    than a in one objective does not dominate a, and both stay.
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

    def __init__(self, base, d, extra_evals, n_failed, settings):
        self.objectives = d["objectives"]
        self.variables = d["variables"]
        self.cv = d["cv"]
        self.active_n = len(self.objectives)
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
    from . import _module
    core = _module()
    d = core.polish(fn, constraints, bounds, int(n_objs), int(n_cons), int(n_eq),
                    float(eq_tol), int(pop_size), int(budget), recorder.points(),
                    [list(map(float, x)) for x in base.variables], int(base.evaluations),
                    how, int(points),
                    on_generation=on_generation if int(record_every) > 0 else None,
                    record_every=int(record_every), fd_step=float(fd_step),
                    local_tol=float(local_tol))
    # the guard counts the failures of the algorithm and of the polishing alike
    n_failed = guard.n_failed if guard is not None else int(base.n_failed)
    settings = [("polish", how), ("polish_points", str(d["chosen"])),
                ("polish_evaluations", str(d["evaluations"])),
                ("polish_version", str(core.METHOD_VERSION))]
    return PolishedResult(base, d, recorder.calls, n_failed, settings)

# SPDX-License-Identifier: Apache-2.0
"""The local solvers of one subproblem.

All three work in the box normalized to [0, 1]^n (y = (x - lo) / (hi - lo))
and spend evaluations only through the Evaluator, so the budget and the
archive see every point. A solver stops at its own tolerance, at the
subproblem's allowance (SubBudgetSpent) or at the run's budget (BudgetSpent,
passed on); what it found is in the archive either way, and the subproblem's
result is the best point it evaluated (best_of).

  sqp    SciPy's SLSQP (Kraft 1988; BSD-3, called, not copied). Gradients by
         forward differences with step fd_step in y (backwards at the upper
         bound; a step that fails is taken the other way once): n evaluations
         a gradient, every one counted. The problem's
         inequalities and equalities go to SLSQP as constraints of their own
         (equalities as h = 0). A failed evaluation ends the subproblem: SLSQP
         cannot step over a hole in the function (SCALAR-6).
  nm     SciPy's Nelder-Mead with adaptive parameters (adaptive=True: Gao & Han,
         Comput. Optim. Appl. 51(1), 2012) and SciPy's handling of the box.
  cmaes  cmaes.py; the box by evaluating the clipped point and adding the
         squared distance to it times (1 + |value|) (SCALAR-7).
For nm and cmaes the subproblem's own constraints and the problem's are a
penalty in which a feasible point is always better (SCALAR-2): an infeasible
point is worth the worst feasible value known for this subproblem — over the
archive at its start, then over what it evaluates — plus its violation.
"""

from __future__ import annotations

import math

import numpy as np

from .evaluator import BudgetSpent, SubBudgetSpent


class FailedPoint(Exception):
    """sqp met a failed evaluation."""


# A subproblem's own rows count as met within this (normalized units): a
# solver meets an equality to rounding, not exactly. The problem's
# constraints keep the vault's rule (cv <= 0).
SLACK = 1e-9


class Run:
    """One subproblem's bookkeeping: the points it evaluated, and their worth."""

    def __init__(self, sub, ev, frame):
        self.sub, self.ev, self.frame = sub, ev, frame
        self.lo, self.hi = ev.lo, ev.hi
        self.width = ev.hi - ev.lo
        self.touched = []

    def x_of(self, y):
        return self.lo + np.clip(y, 0.0, 1.0) * self.width

    def at(self, y) -> int:
        i = self.ev.index(self.x_of(y))
        self.touched.append(i)
        return i

    def worth(self, i):
        """(value, violation): the subproblem's merit, its rows' violation
        beyond SLACK plus the problem's cv."""
        if self.ev.failed[i]:
            return math.inf, math.inf
        value, viol = self.sub.merit(self.frame.bar(self.ev.F[i]))
        return value, (viol if viol > SLACK else 0.0) + self.ev.cv(i)


def best_of(run: Run, indices):
    """The feasible point of least value, else the least violating one."""
    best, key = None, None
    for i in indices:
        v, c = run.worth(i)
        k = (0, v, i) if c <= 0.0 else (1, c, i)
        if key is None or k < key:
            best, key = i, k
    return best


def moved(run: Run, starts, best) -> bool:
    """Did the subproblem get better than its best start, beyond 1e-9 relative?"""
    v1, c1 = run.worth(best)
    v0, c0 = min((run.worth(i) for i in starts), key=lambda w: (w[1] > 0.0, w[1], w[0]))
    if c0 > 0.0:
        return c1 < c0 * (1.0 - 1e-9)
    return c1 <= 0.0 and v1 < v0 - 1e-9 * max(1.0, abs(v0))


# ── sqp ─────────────────────────────────────────────────────────────────────
def sqp(run: Run, y0, t0: float, *, fd_step: float, tol: float):
    from scipy.optimize import minimize as sp_minimize
    sub, ev, frame = run.sub, run.ev, run.frame
    n = len(y0)
    k_ineq = ev.n_cons - ev.n_eq
    Ai, bi, ci = sub.mats(sub.ineq)
    Ae, be, ce = sub.mats(sub.eq)
    jac_memo = {}

    def values(z):
        i = run.at(z[:n])
        if ev.failed[i]:
            raise FailedPoint
        return i, frame.bar(ev.F[i]), ev.G[i]

    def jac_at(z):
        """∂F̄/∂y (m × n) and ∂G/∂y by forward differences."""
        y = np.clip(z[:n], 0.0, 1.0)
        key = y.tobytes()
        if key in jac_memo:
            return jac_memo[key]
        _, Fb, G = values(z)
        JF = np.zeros((len(Fb), n))
        JG = np.zeros((len(G), n))

        def at(j, h):
            yj = y.copy()
            yj[j] += h
            i = run.at(yj)
            return None if ev.failed[i] else (frame.bar(ev.F[i]), ev.G[i])

        for j in range(n):
            h = fd_step if y[j] + fd_step <= 1.0 else -fd_step
            got = at(j, h)
            if got is None and 0.0 <= y[j] - h <= 1.0:
                h = -h                       # a failed step: the other way once
                got = at(j, h)
            if got is None:
                raise FailedPoint
            Fj, Gj = got
            JF[:, j] = (Fj - Fb) / h
            if len(G):
                JG[:, j] = (Gj - G) / h
        jac_memo[key] = (JF, JG)
        return JF, JG

    t_of = (lambda z: z[n]) if sub.aux else (lambda z: 0.0)

    def fun(z):
        _, Fb, _ = values(z)
        return float(sub.a0 @ Fb + sub.b0 * t_of(z) + sub.c0)

    def fun_jac(z):
        JF, _ = jac_at(z)
        g = sub.a0 @ JF
        return np.append(g, sub.b0) if sub.aux else g

    def c_ineq(z):
        _, Fb, G = values(z)
        return np.concatenate([Ai @ Fb + bi * t_of(z) + ci, -G[:k_ineq] - 1e-10])

    def c_ineq_jac(z):
        JF, JG = jac_at(z)
        J = np.vstack([Ai @ JF, -JG[:k_ineq]]) if len(JG) else Ai @ JF
        return np.hstack([J, np.concatenate([bi, np.zeros(k_ineq)])[:, None]]) if sub.aux else J

    def c_eq(z):
        _, Fb, G = values(z)
        return np.concatenate([Ae @ Fb + be * t_of(z) + ce, G[k_ineq:]])

    def c_eq_jac(z):
        JF, JG = jac_at(z)
        J = np.vstack([Ae @ JF, JG[k_ineq:]]) if len(JG) else Ae @ JF
        return np.hstack([J, np.concatenate([be, np.zeros(ev.n_eq)])[:, None]]) if sub.aux else J

    cons = []
    if len(bi) or k_ineq:
        cons.append({"type": "ineq", "fun": c_ineq, "jac": c_ineq_jac})
    if len(be) or ev.n_eq:
        cons.append({"type": "eq", "fun": c_eq, "jac": c_eq_jac})
    z0 = np.append(y0, t0) if sub.aux else np.asarray(y0, float)
    bounds = [(0.0, 1.0)] * n + ([(None, None)] if sub.aux else [])
    try:
        return sp_minimize(fun, z0, jac=fun_jac, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100000, "ftol": tol})
    except (SubBudgetSpent, FailedPoint) as e:
        return e


# ── nm and cmaes: a penalty in which feasible is better ─────────────────────
class Penalty:
    """value if feasible, else (worst feasible value known) + violation."""

    def __init__(self, run: Run, start):
        self.run = run
        self.worst = -math.inf
        for i in start:
            v, c = run.worth(i)
            if c <= 0.0 and math.isfinite(v):
                self.worst = max(self.worst, v)
        if not math.isfinite(self.worst):
            self.worst = 0.0

    def __call__(self, y) -> float:
        i = self.run.at(y)
        v, c = self.run.worth(i)
        if not math.isfinite(c):
            return math.inf                    # a failed evaluation
        if c <= 0.0:
            self.worst = max(self.worst, v)
            return v
        return self.worst + c


def nm(run: Run, y0, start, *, tol: float) -> None:
    from scipy.optimize import minimize as sp_minimize
    pen = Penalty(run, start)
    try:
        sp_minimize(pen, np.asarray(y0, float), method="Nelder-Mead",
                    bounds=[(0.0, 1.0)] * len(y0),
                    options={"adaptive": True, "xatol": tol, "fatol": tol, "maxfev": 10 ** 9,
                             "maxiter": 10 ** 9})
    except SubBudgetSpent:
        pass


def cmaes(run: Run, y0, start, *, sigma0: float, tol: float, rng) -> None:
    from .cmaes import CMAES
    pen = Penalty(run, start)
    es = CMAES(y0, sigma0, rng)
    try:
        while es.spread() > tol:
            Y = es.ask()
            vals = []
            for y in Y:
                yc = np.clip(y, 0.0, 1.0)
                v = pen(yc)
                d2 = float(np.sum((y - yc) ** 2))
                vals.append(v + d2 * (1.0 + abs(v)) if math.isfinite(v) else math.inf)
            es.tell(vals)
    except SubBudgetSpent:
        pass

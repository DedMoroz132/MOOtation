# SPDX-License-Identifier: Apache-2.0
"""The one road to fn for the scalarization methods: every evaluation is
counted against the budget, kept in the archive and reported to the observer.

THE BUDGET. A call of fn is an evaluation, whoever asks for it: a solver's
iterate, a finite-difference step, a trial point of a line search, a point of
the initial plan. A point asked for twice is evaluated once (the archive keeps
it by its bytes), so the number of calls of fn is the number counted. Asking
for a point past the budget raises BudgetSpent, and past a subproblem's
allowance SubBudgetSpent: the solver stops there, and what it found is in the
archive.

THE ARCHIVE holds every evaluation: x, f, the constraint values and whether it
failed. A point is feasible when every inequality is <= 0 and every equality
within the tolerance of the moment (|h| <= eq_tol), and its violation cv is the
vault's (data_vault.hpp): sum max(g, 0) + sum max(|h| - eq_tol, 0). A failed
evaluation (a NaN from the guard, mootation.failures) is in the archive, marked,
and never in the answer.

THE ANSWER, as every algorithm's, is pop_size points: the feasible
non-dominated points of the archive, reduced by DSS (run/archive.py) when there
are more; without a feasible point, the least violating ones. The observer
gets it after the initial plan (gen 0), then every record_every * pop_size
evaluations and at the end, with the variables beside the objectives.
"""

from __future__ import annotations

import math

import numpy as np


class BudgetSpent(Exception):
    """The run's evaluations are spent."""


class SubBudgetSpent(Exception):
    """The current subproblem's allowance is spent."""


class Evaluator:
    def __init__(self, fn, constraints, *, bounds, n_objs: int, n_cons: int, n_eq: int,
                 eq_tol: float, eq_tol0: float | None, budget: int, pop_size: int,
                 on_generation=None, record_every: int = 0):
        self.fn, self.constraints = fn, constraints
        self.lo = np.array([b[0] for b in bounds], float)
        self.hi = np.array([b[1] for b in bounds], float)
        self.m, self.n_cons, self.n_eq = int(n_objs), int(n_cons), int(n_eq)
        self.eq_tol_end = float(eq_tol)
        self.eq_tol0 = None if eq_tol0 is None else float(eq_tol0)
        self.budget, self.pop_size = int(budget), int(pop_size)
        self.observer = on_generation if record_every and on_generation is not None else None
        self.stride = max(1, int(record_every)) * self.pop_size
        self.next_record = self.stride
        self.X, self.F, self.G, self.failed = [], [], [], []
        self.where = {}                       # x's bytes -> archive index
        self.spent = 0
        self.limit = self.budget              # the current subproblem's last evaluation
        self.nd = []                          # feasible non-dominated archive indices
        self.ndF = np.zeros((0, self.m))      # ... and their objectives
        self.nd_tol = None                    # the tolerance self.nd was judged at
        self.over = False                     # the run has ended: the final tolerance
        self.recorded = False                 # gen 0 has gone to the observer
        self.gen0 = 0                         # gens before this evaluator (polish.py)
        self._ok_i, self._ok_X, self._ok_upto = [], None, 0   # ok_points()

    # ── evaluating ──────────────────────────────────────────────────────────
    def index(self, x) -> int:
        """The archive index of x, evaluating it if it is new."""
        x = np.ascontiguousarray(x, dtype=float)
        key = x.tobytes()
        i = self.where.get(key)
        if i is not None:
            return i
        if self.spent >= self.budget:
            raise BudgetSpent
        if self.spent >= self.limit:
            raise SubBudgetSpent
        xs = [float(v) for v in x]
        f = [float(v) for v in self.fn(xs)]
        if len(f) != self.m:
            raise ValueError(f"fn returned {len(f)} values for {self.m} objectives")
        g = [float(v) for v in self.constraints(xs)] if self.constraints is not None else []
        bad = not all(math.isfinite(v) for v in f + g)
        return self._keep(x, f, g, bad, key, count=True)

    def plant(self, x, f, g) -> int:
        """A point whose values are known (a seed population): archived, not counted."""
        x = np.ascontiguousarray(x, dtype=float)
        key = x.tobytes()
        if key in self.where:
            return self.where[key]
        g = list(g) if g is not None else []
        bad = not all(math.isfinite(float(v)) for v in list(f) + g)
        return self._keep(x, [float(v) for v in f], [float(v) for v in g], bad, key, count=False)

    def _keep(self, x, f, g, bad, key, count) -> int:
        i = len(self.X)
        self.X.append(x.copy())
        self.F.append(np.asarray(f, float))
        self.G.append(np.asarray(g, float))
        self.failed.append(bool(bad))
        self.where[key] = i
        if count:
            self.spent += 1
        if not bad and self.nd_tol == self.eq_tol() and self.cv(i) <= 0.0:
            self._offer(i)
        if self.observer is not None and count and self.spent >= self.next_record:
            while self.next_record <= self.spent:
                self.next_record += self.stride
            self.record()
        return i

    def ok_points(self):
        """(indices, X) of the evaluations that did not fail, X as an array
        kept from call to call and extended by the new rows only."""
        if self._ok_upto < len(self.X):
            new = [i for i in range(self._ok_upto, len(self.X)) if not self.failed[i]]
            if new:
                self._ok_i.extend(new)
                rows = np.array([self.X[i] for i in new])
                self._ok_X = rows if self._ok_X is None else np.vstack([self._ok_X, rows])
            self._ok_upto = len(self.X)
        X = self._ok_X if self._ok_X is not None else np.zeros((0, len(self.lo)))
        return self._ok_i, X

    # ── feasibility ─────────────────────────────────────────────────────────
    def eq_tol(self) -> float:
        """The equalities' tolerance of the moment (eq_tol_schedule = linear: from
        eq_tol0 to eq_tol in proportion to the evaluations spent; eq_tol itself
        once the run is over, for the answer)."""
        if self.eq_tol0 is None or self.budget <= 0 or self.over:
            return self.eq_tol_end
        share = min(1.0, self.spent / self.budget)
        return self.eq_tol0 + (self.eq_tol_end - self.eq_tol0) * share

    def violation(self, g, tol=None) -> float:
        if not len(g):
            return 0.0
        tol = self.eq_tol() if tol is None else tol
        k = len(g) - self.n_eq
        v = float(np.sum(np.maximum(g[:k], 0.0)))
        if self.n_eq:
            v += float(np.sum(np.maximum(np.abs(g[k:]) - tol, 0.0)))
        return v

    def cv(self, i, tol=None) -> float:
        return math.inf if self.failed[i] else self.violation(self.G[i], tol)

    # ── the answer ──────────────────────────────────────────────────────────
    def _offer(self, i) -> None:
        f = self.F[i]
        if len(self.nd):
            if np.any(np.all(self.ndF <= f, axis=1)):
                return                        # dominated or equal: not added
            keep = ~np.all(f <= self.ndF, axis=1)
            self.nd = [j for j, k in zip(self.nd, keep) if k]
            self.ndF = self.ndF[keep]
        self.nd.append(i)
        self.ndF = np.vstack([self.ndF, f[None, :]]) if len(self.nd) > 1 else f[None, :].copy()

    def _rebuild(self) -> None:
        tol = self.eq_tol()
        self.nd, self.ndF, self.nd_tol = [], np.zeros((0, self.m)), tol
        for i in range(len(self.X)):
            if not self.failed[i] and self.cv(i, tol) <= 0.0:
                self._offer(i)

    def answer(self, k=None):
        """Archive indices of the answer: up to k (pop_size) feasible
        non-dominated points by DSS, or the least violating ones."""
        k = self.pop_size if k is None else int(k)
        if self.nd_tol != self.eq_tol():
            self._rebuild()
        if self.nd:
            if len(self.nd) <= k:
                return list(self.nd)
            from ..run.archive import dss_order
            order = dss_order(np.array([self.F[i] for i in self.nd]), k)
            return [self.nd[j] for j in order]
        ok = [i for i in range(len(self.X)) if not self.failed[i]]
        ok.sort(key=lambda i: (self.cv(i), i))
        return ok[:k]

    def record(self, final: bool = False) -> None:
        """The answer to the observer: gen 0 the first time (after the initial
        plan), then the evaluations spent in units of pop_size, rounded up at
        the end."""
        if self.observer is None:
            return
        idx = self.answer()
        F = [list(map(float, self.F[i])) for i in idx]
        X = [list(map(float, self.X[i])) for i in idx]
        gen = (0 if not self.recorded else self.gen0 +
               (-(-self.spent // self.pop_size) if final else self.spent // self.pop_size))
        self.recorded = True
        self.observer(gen, F, X)

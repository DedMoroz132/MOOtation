# SPDX-License-Identifier: Apache-2.0
"""Failed evaluations (task 7 of 2026-10-10, supplement 3, §2).

A finite-element model can crash, not converge or return NaN, and in an
engineering problem that may be a large share of the points tried: a hidden
constraint, in Le Digabel & Wild's taxonomy (Optim. Eng. 25(2), 2024) one that
is non-quantifiable, unrelaxable, simulation-based and unknown. minimize()
runs the user's functions through a Guard:

  * an exception of `failure_exceptions` (Exception by default) raised by fn,
    constraints or batch, a non-finite value among the objectives or the
    constraints, or a None row from batch is a failure. The guard hands the
    core NaN in its place, and the core (data_vault.hpp, FAILED EVALUATIONS)
    gives the point a penalty: dominated by every point that did not fail,
    infeasible beyond any infeasible one, equal to every other failure, and
    never in the answer, the observers' rows, the ideal point or the archive;
  * on_failure = "raise" catches nothing, and a non-finite value raises in the
    core: a failure stops the run, for debugging;
  * max_failure_share (off by default) stops the run with an error once more
    than that share of the last failure_window evaluations failed.

A failure is an evaluation and counts in the budget; result.n_failed counts
them. A hung evaluation (a timeout) is not handled here: the batch function or
the job that runs the solver has to stop it and return None or raise.
"""

from __future__ import annotations

import math
from collections import deque


class FailureShareExceeded(RuntimeError):
    """More than max_failure_share of the last failure_window evaluations failed."""


def finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class Guard:
    """Catches the failures of one run's functions (see the module's docstring)."""

    def __init__(self, n_objs: int, n_cons: int = 0, exceptions=Exception,
                 max_share: float | None = None, window: int = 100):
        if max_share is not None and not 0.0 <= max_share < 1.0:
            raise ValueError(f"max_failure_share = {max_share}: a share in [0, 1)")
        if window < 1:
            raise ValueError(f"failure_window = {window}: at least 1")
        self.n_objs, self.n_cons = int(n_objs), int(n_cons)
        self.exceptions = exceptions
        self.max_share, self.window = max_share, int(window)
        self.recent = deque(maxlen=self.window)
        self.n_failed = 0
        self.last_error = None          # the text of the last failure, for the log

    def _note(self, failed: bool, error: str | None = None) -> None:
        self.last_error = error if failed else None
        self.n_failed += failed
        self.recent.append(bool(failed))
        if (self.max_share is not None and len(self.recent) == self.window
                and sum(self.recent) > self.max_share * self.window):
            raise FailureShareExceeded(
                f"{sum(self.recent)} of the last {self.window} evaluations failed, more than "
                f"max_failure_share = {self.max_share}; the run stops (the last failure: "
                f"{error})")

    def _late(self, error: str) -> None:
        """The constraints of an evaluation whose objectives were fine failed."""
        if self.recent and not self.recent[-1]:
            self.recent[-1] = True
            self.n_failed += 1
        self.last_error = error

    def fn(self, f):
        def guarded(x, _f=f):
            try:
                out = list(_f(x))
            except self.exceptions as e:
                self._note(True, f"{type(e).__name__}: {e}")
                return [math.nan] * self.n_objs
            ok = finite(out)
            self._note(not ok, None if ok else "a non-finite objective")
            return out
        return guarded

    def residuals(self, r):
        """A problem's residuals (minimize's residuals=): as fn, the stand-in
        NaN as many as the first call that did not fail gave; a failure before
        any is raised."""
        width = []

        def guarded(x, _r=r):
            try:
                out = list(_r(x))
            except self.exceptions as e:
                if not width:
                    raise
                self._note(True, f"{type(e).__name__}: {e}")
                return [math.nan] * width[0]
            if not width:
                width.append(len(out))
            ok = finite(out)
            self._note(not ok, None if ok else "a non-finite residual")
            return out
        return guarded

    def constraints(self, c):
        def guarded(x, _c=c):
            try:
                out = list(_c(x))
            except self.exceptions as e:
                if self.n_cons <= 0:    # minimize's probe for the width: nothing to stand in
                    raise
                self._late(f"{type(e).__name__}: {e}")
                return [math.nan] * self.n_cons
            if not finite(out):
                self._late("a non-finite constraint")
            return out
        return guarded

    def batch(self, b):
        def guarded(X, _b=b):
            try:
                rows = _b(X)
            except self.exceptions as e:
                for _ in X:
                    self._note(True, f"{type(e).__name__}: {e}")
                return [[math.nan] * self.n_objs for _ in X]
            out = []
            for r in rows:
                if r is None:
                    self._note(True, "None from batch")
                    out.append([math.nan] * self.n_objs)
                    continue
                r = list(r)
                ok = finite(r)
                self._note(not ok, None if ok else "a non-finite objective")
                out.append(r)
            return out
        return guarded

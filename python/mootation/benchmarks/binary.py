# SPDX-License-Identifier: Apache-2.0
"""Bit-string problems of the runtime theory (task 7 of 2026-10-10, supplement 2, §3).

All three are maximization problems of a bit string x of length n; the
library minimizes, so the registered objectives are their negatives.

  OneMinMax  (OMM_n<n>)       f1 = |x|_0, the zeros; f2 = |x|_1, the ones.
                              Every string is Pareto optimal: the front is the
                              n + 1 pairs (a, n - a).
  LOTZ       (LOTZ_n<n>)      f1 = the leading ones, f2 = the trailing zeros.
                              The front is the n + 1 pairs (a, n - a), the
                              strings 1^a 0^(n - a).
  OneJumpZeroJump_k (OJZJ_n<n>_k<k>), Definition 3 of Bian, Zhou, Li & Qian
                              (IJCAI 2023), after Doerr & Zheng (AAAI 2021),
                              2 <= k < n/2:
                              f1 = k + |x|_1 if |x|_1 <= n - k or x = 1^n,
                                   n - |x|_1 otherwise;
                              f2 = k + |x|_0 if |x|_0 <= n - k or x = 0^n,
                                   n - |x|_0 otherwise.
                              The front is {(a, n + 2k - a) | a in [2k..n] or
                              a = k or a = n + k}: n - 2k + 3 points.

Fronts and frames are exact, from the formulas (tests: against every one of the
2^n strings for n <= 12). Theirs is the paper's measure: the first generation
at which the population holds every front vector (run.campaign: cover_gen).
"""

from __future__ import annotations

from typing import List

import numpy as np

# (name, n, k): the registered sizes; OJZJ_n10_k3 is the paper's check (§4.9)
SIZES = (("OMM", 10, 0), ("OMM", 20, 0), ("OMM", 50, 0),
         ("LOTZ", 10, 0), ("LOTZ", 20, 0), ("LOTZ", 50, 0),
         ("OJZJ", 10, 2), ("OJZJ", 10, 3), ("OJZJ", 20, 3), ("OJZJ", 20, 4))


def name(kind: str, n: int, k: int = 0) -> str:
    return f"{kind}_n{n}_k{k}" if kind == "OJZJ" else f"{kind}_n{n}"


def objectives(kind: str, x, k: int = 0) -> List[float]:
    """The maximized objectives (f1, f2) of a bit string."""
    b = [1 if v else 0 for v in x]
    n = len(b)
    ones = sum(b)
    zeros = n - ones
    if kind == "OMM":
        return [float(zeros), float(ones)]
    if kind == "LOTZ":
        lo = next((i for i, v in enumerate(b) if v == 0), n)
        tz = next((i for i, v in enumerate(reversed(b)) if v == 1), n)
        return [float(lo), float(tz)]
    if kind == "OJZJ":
        f1 = k + ones if (ones <= n - k or ones == n) else n - ones
        f2 = k + zeros if (zeros <= n - k or zeros == n) else n - zeros
        return [float(f1), float(f2)]
    raise ValueError(kind)


def evaluate(kind: str, x, k: int = 0) -> List[float]:
    """The registered (minimized) objectives: the negatives."""
    return [-v for v in objectives(kind, x, k)]


def front(kind: str, n: int, k: int = 0) -> np.ndarray:
    """The Pareto front of the maximization problem, as (f1, f2) rows."""
    if kind in ("OMM", "LOTZ"):
        return np.array([[a, n - a] for a in range(n + 1)], float)
    a = [k] + list(range(2 * k, n + 1)) + [n + k]
    return np.array([[v, n + 2 * k - v] for v in a], float)


def frame(kind: str, n: int, k: int = 0):
    """(ideal, nadir) of the minimized front."""
    F = -front(kind, n, k)
    return tuple(F.min(axis=0)), tuple(F.max(axis=0))

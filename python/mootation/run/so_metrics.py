# SPDX-License-Identifier: Apache-2.0
"""The single-objective measures of a run in Python: the cross-check of
include/mootation/so_metrics.hpp (task 18, stage 1, §1.3), which is the one
implementation (_core.so_metrics). The definitions and readings (SOM-1 to
SOM-7) are that header's: targets absolute (COCO, H22 §III-F item 2) or
relative (Moré and Wild, item 1), the runtime to a target, ERT, the ECDF and
the value at a fixed budget, read off a run's improvement records
(evaluations, grad_calls, f, cv); the simulated restarts (H16 §3.2, H22
§V-A), bit for bit: std::mt19937(seed), rnd::below and rnd::shuffle
(random.hpp) are written out here.

Python's 10 ** k is the C library's pow, the header's dm::pow the library's
own: the absolute grid may differ in the last bit, and a test holds the two
within that.
"""

from __future__ import annotations

import math

REFERENCE_VERSION = 1                     # SO_REFERENCE_VERSION (so_metrics.hpp, kReferenceVersion)


def coco_precisions(k_min: int = -8, k_max: int = 2) -> list:
    return [10.0 ** ((5 * k_max - i) / 5.0) for i in range(5 * (k_max - k_min) + 1)]


def coco_table_precisions() -> list:
    return [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7]


def more_wild_factors() -> list:
    return [1e-1, 1e-3, 1e-5, 1e-7]


def default_precisions(kind: str) -> list:
    if kind == "absolute":
        return coco_precisions()
    if kind == "relative":
        return more_wild_factors()
    raise ValueError(f"so_metrics: targets 'absolute' or 'relative', not '{kind}'")


def targets(kind: str, eps, f_ref: float, f_x0: float) -> list:
    if kind == "absolute":
        return [f_ref + e for e in eps]
    if kind == "relative":
        return [f_ref + e * (f_x0 - f_ref) for e in eps]
    raise ValueError(f"so_metrics: targets 'absolute' or 'relative', not '{kind}'")


def cost(record, grad_cost: float = 1.0) -> float:
    return float(record[0]) + grad_cost * float(record[1])


def runtime(records, target: float, grad_cost: float = 1.0):
    for r in records:
        if r[3] <= 0.0 and r[2] <= target:
            return cost(r, grad_cost)
    return None


def ert(runtimes, lengths) -> float:
    if len(runtimes) != len(lengths):
        raise ValueError("so_metrics: a length per run")
    total, hits = 0.0, 0
    for t, n in zip(runtimes, lengths):
        if t is not None:
            total += t
            hits += 1
        else:
            total += n
    if hits == 0:
        return 0.0 if total == 0.0 else math.inf
    return total / hits


def ecdf(runtimes, xs) -> list:
    n = len(runtimes)
    return [0.0 if not n else sum(1 for t in runtimes if t is not None and t <= x) / n for x in xs]


class _Words:
    """std::mt19937(seed)'s 32-bit words: Matsumoto & Nishimura's MT19937
    (the standard's std::mt19937, its seed by init_genrand), written out so
    that the mirror depends on no library's generator."""

    def __init__(self, seed: int):
        mt = [seed & 0xFFFFFFFF]
        for i in range(1, 624):
            mt.append((1812433253 * (mt[-1] ^ (mt[-1] >> 30)) + i) & 0xFFFFFFFF)
        self._mt, self._i = mt, 624

    def _twist(self) -> None:
        mt = self._mt
        for k in range(624):
            y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7FFFFFFF)
            mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self._i = 0

    def word32(self) -> int:
        if self._i >= 624:
            self._twist()
        y = self._mt[self._i]
        self._i += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return y ^ (y >> 18)


def _below(g: _Words, n: int) -> int:
    """rnd::below for n <= 2^32 (Lemire, random.hpp)."""
    m = g.word32() * n
    low = m & 0xFFFFFFFF
    if low < n:
        t = ((1 << 32) - n) % n
        while low < t:
            m = g.word32() * n
            low = m & 0xFFFFFFFF
    return m >> 32


def simulated_runtimes(runtimes, lengths, n: int, seed: int, first_permuted: bool = True) -> list:
    """N simulated runtimes off runs at one target (so_metrics.hpp, SOM-6,
    SOM-7); [] when no run hit it."""
    if len(runtimes) != len(lengths):
        raise ValueError("so_metrics: a length per run")
    k = len(runtimes)
    if not any(t is not None for t in runtimes):
        return []
    g = _Words(seed)
    perm = list(range(k))
    if first_permuted:
        for i in range(k, 1, -1):                       # rnd::shuffle: Fisher-Yates from the last
            j = _below(g, i)
            if j != i - 1:
                perm[i - 1], perm[j] = perm[j], perm[i - 1]
    out = []
    for s in range(n):
        total = 0.0
        i = perm[s % k] if first_permuted else _below(g, k)
        while runtimes[i] is None:
            total += lengths[i]
            i = _below(g, k)
        out.append(total + runtimes[i])
    return out


def value_at(records, budget: float, grad_cost: float = 1.0):
    best = None
    for r in records:
        if cost(r, grad_cost) > budget:
            break
        if r[3] <= 0.0 and (best is None or r[2] < best):
            best = r[2]
    return best


def peaks_found(X, F, height: float, eps: float, r: float) -> int:
    """The CEC 2013 niching report's Algorithm 1 (so_metrics.hpp,
    peaks_found): the points by F, largest first; one within eps of the peak
    height counts when it is farther than r from every one counted before."""
    order = sorted(range(len(X)), key=lambda i: -F[i])           # stable: equal F in their order
    seeds = []
    for i in order:
        if not abs(height - F[i]) <= eps:
            continue
        if all(sum((a - b) ** 2 for a, b in zip(X[i], X[s])) ** 0.5 > r for s in seeds):
            seeds.append(i)
    return len(seeds)

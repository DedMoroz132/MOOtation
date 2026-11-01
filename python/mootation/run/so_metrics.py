# SPDX-License-Identifier: Apache-2.0
"""The single-objective measures of a run in Python: the cross-check of
include/mootation/so_metrics.hpp (task 18, stage 1, §1.3), which is the one
implementation (_core.so_metrics). The definitions and readings (SOM-1 to
SOM-4) are that header's: targets absolute (COCO, H22 §III-F item 2) or
relative (Moré and Wild, item 1), the runtime to a target, ERT, the ECDF and
the value at a fixed budget, read off a run's improvement records
(evaluations, grad_calls, f, cv).

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


def value_at(records, budget: float, grad_cost: float = 1.0):
    best = None
    for r in records:
        if cost(r, grad_cost) > budget:
            break
        if r[3] <= 0.0 and (best is None or r[2] < best):
            best = r[2]
    return best

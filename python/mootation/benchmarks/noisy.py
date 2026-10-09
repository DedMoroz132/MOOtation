# SPDX-License-Identifier: Apache-2.0
"""Noisy copies of benchmark problems: a control written for this library, not
a published suite (task 6 of 2026-10-09, item 7). They stand in for the noise
of a mesh and a solver behind a real objective.

The copy of a problem returns

    f~_i = f_i + sigma (nadir_i - ideal_i) z_i,   z_i ~ N(0, 1),

independently over the objectives and the evaluations, ideal and nadir the
base problem's frame. Its name is the base's with the level after it:
`<base>_n1` for sigma = 0.01 and `<base>_n5` for sigma = 0.05, of ZDT1, ZDT4,
DTLZ1_3D, DTLZ2_3D and WFG7_3D (ZDT1_n1, DTLZ2_3D_n5, ...). Front, frame,
Pareto set and distance function are the base's.

THE STREAM. As with the uninformative problems (uninformative.py), z is
deterministic from (the run's seed, the evaluation's number): a campaign hands
each run a fresh evaluator (`BenchProblem.make_evaluator(seed)`), and two runs
with the same seed see the same noise. z by Box-Muller from two uniforms of
the splitmix64 hash, in a domain of its own. The plain `evaluate`, for a
caller without a seed, draws the noise from x instead: the same x, the same
values.

WHAT IS MEASURED. The algorithm sees f~; a campaign measures every indicator
on the noise-free f at the same x (`BenchProblem.true_objectives`, the base's
`evaluate`, called again), and records beside them true_dominated_share, the
share of the answer that another of its points dominates by the true values:
the points the noise let through.
"""

from __future__ import annotations

import math
import struct
from typing import Callable, List

from .uninformative import _MASK, _splitmix64, _unit

BASES = ("ZDT1", "ZDT4", "DTLZ1_3D", "DTLZ2_3D", "WFG7_3D")
SIGMA = {"n1": 0.01, "n5": 0.05}
_DOMAIN = 3 << 61                       # apart from uninformative.py's two


def name(base: str, level: str) -> str:
    return f"{base}_{level}"


def is_noisy(problem: str) -> bool:
    """Is this registry name a noisy copy (`<base>_n1`, `<base>_n5`)?"""
    return "_" in problem and problem.rsplit("_", 1)[1] in SIGMA


def base_of(problem: str) -> str:
    return problem.rsplit("_", 1)[0]


def normal(seed: int, k: int, i: int) -> float:
    """z of objective i at evaluation k of the run with this seed: N(0, 1)."""
    u1 = _unit(_splitmix64((seed & _MASK) ^ _splitmix64(_DOMAIN + 4 * k + 2 * i)))
    u2 = _unit(_splitmix64((seed & _MASK) ^ _splitmix64(_DOMAIN + 4 * k + 2 * i + 1)))
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


class Evaluator:
    """One run's objective function: the base's values plus the k-th draw of
    the stream at the k-th call. `scale` is sigma (nadir - ideal), per
    objective."""

    def __init__(self, base: Callable, scale, seed: int):
        self.base = base
        self.scale = [float(s) for s in scale]
        self.seed = int(seed)
        self.k = 0

    def __call__(self, x) -> List[float]:
        f = self.base(list(x))
        out = [v + s * normal(self.seed, self.k, i)
               for i, (v, s) in enumerate(zip(f, self.scale))]
        self.k += 1
        return out


def evaluate_by_x(base: Callable, scale, x) -> List[float]:
    """Without a seed: the noise drawn from x itself, the same for the same x."""
    h = 0
    for v in x:
        h = _splitmix64(h ^ struct.unpack("<Q", struct.pack("<d", float(v)))[0])
    out = []
    for i, (v, s) in enumerate(zip(base(list(x)), scale)):
        u1, u2 = _unit(_splitmix64(h + 2 * i + 1)), _unit(_splitmix64(h + 2 * i + 2))
        out.append(v + s * math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2))
    return out

# SPDX-License-Identifier: Apache-2.0
"""Uninformative objectives: a probe of an algorithm's structural bias.

Every objective value is an independent U(0, 1) draw that does not depend on
x at all, so no region of the decision space is better than another: where an
algorithm's final population ends up is a property of the ALGORITHM — of its
operators, its repair at the bounds, its selection — not of the problem. A
population piled up at the bounds or in the centre of the box is a bias the
algorithm will carry onto every problem. (Task 2 of 2026-09-23, A3; the
method is a simplification of Kudela, van Stein, Bäck & Kononova, GECCO 2026,
who run 120 PlatEMO algorithms on five such problems through the BIAS
toolbox; the analysis here is `--bias` on the campaign command line.)

  uninformative_n02_2D, uninformative_n10_2D
      M = 2, n = 2 and 10, x in [0, 1]^n; no Pareto front, no reference.

The values are deterministic from (the run's seed, the evaluation's number):
two runs with the same seed see the same stream, whatever they ask for. That
needs a counter, so the problem hands a campaign a fresh evaluator per run
(`BenchProblem.make_evaluator(seed)`). The plain `evaluate`, for a caller
without a seed, draws from x instead — the same x gets the same values —
which is just as uninformative.
"""

from __future__ import annotations

import struct
from typing import List

M = 2
SIZES = (2, 10)
_MASK = (1 << 64) - 1


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _unit(h: int) -> float:
    """53 high bits of a 64-bit hash as a double in [0, 1)."""
    return (h >> 11) * (1.0 / (1 << 53))


def value(seed: int, k: int, i: int) -> float:
    """Objective i of evaluation k in the run with this seed: U(0, 1)."""
    h = _splitmix64((seed & _MASK) ^ _splitmix64(k * M + i))
    return _unit(h)


class Evaluator:
    """One run's objective function: the k-th call returns value(seed, k, .)."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.k = 0

    def __call__(self, x) -> List[float]:
        f = [value(self.seed, self.k, i) for i in range(M)]
        self.k += 1
        return f


def evaluate_by_x(x) -> List[float]:
    """Without a seed: values drawn from x itself, the same for the same x."""
    h = 0
    for v in x:
        h = _splitmix64(h ^ struct.unpack("<Q", struct.pack("<d", float(v)))[0])
    return [_unit(_splitmix64(h + i + 1)) for i in range(M)]


def name(n: int) -> str:
    return f"uninformative_n{n:02d}_{M}D"

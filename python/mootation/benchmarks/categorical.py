# SPDX-License-Identifier: Apache-2.0
"""catZDT1: ZDT1 whose distance function a categorical variable chooses (a
synthetic problem of MOOtation; task 7 of 2026-10-10, supplement 3, §1.4).

x = (c, x1, ..., x10), c one of three labels, x1 .. x10 in [0, 1]:

    f1 = x1,   f2 = g_c (1 - sqrt(x1 / g_c)),   y = x2 .. x10 (nine of them)

  "rastrigin"  g = 1 + 10 * 9 + sum((10 y - 5)^2 - 10 cos(4 pi (10 y - 5)))
               — ZDT4's, its variables mapped from [-5, 5] to [0, 1]: least
               1 at y = 0.5, and 21^9 local optima around it;
  "linear"     g = 1.02 + 9 mean(y) — ZDT1's raised by 0.02: least 1.02 at
               y = 0, one basin;
  "quadratic"  g = 1.05 + 9 mean((y - 0.75)^2): least 1.05 at y = 0.75.

THE FRONT is ZDT1's, f2 = 1 - sqrt(f1), and only "rastrigin" reaches it. At a
given f1, f2 = g - sqrt(f1 g) grows with g (its derivative 1 - sqrt(f1 / g)
/ 2 > 0 for f1 <= 1 <= g), so each label's best curve lies above the one of
any smaller least g: "linear" and "quadratic" give the fronts of g = 1.02 and
1.05, close and dominated. The easy labels lure an algorithm away from the
multimodal one that holds the front; the test checks the three curves.

The evaluator takes the label, or its index (a caller that does not decode
var_types).
"""

from __future__ import annotations

import math
from typing import List, Tuple

LABELS: Tuple[str, ...] = ("rastrigin", "linear", "quadratic")
N_REAL = 10
LEAST_G = {"rastrigin": 1.0, "linear": 1.02, "quadratic": 1.05}


def g(label: str, y) -> float:
    n = len(y)
    if label == "rastrigin":
        return 1.0 + 10.0 * n + sum((10.0 * v - 5.0) ** 2
                                    - 10.0 * math.cos(4.0 * math.pi * (10.0 * v - 5.0))
                                    for v in y)
    if label == "linear":
        return 1.02 + 9.0 * sum(y) / n
    if label == "quadratic":
        return 1.05 + 9.0 * sum((v - 0.75) ** 2 for v in y) / n
    raise ValueError(f"catZDT1: the label {label!r} is not one of {LABELS}")


def label_of(c) -> str:
    if isinstance(c, str):
        return c
    return LABELS[min(max(int(round(c)), 0), len(LABELS) - 1)]


def evaluate(x) -> List[float]:
    c, x1, y = label_of(x[0]), float(x[1]), [float(v) for v in x[2:]]
    gc = g(c, y)
    return [x1, gc * (1.0 - math.sqrt(x1 / gc))]


def bounds() -> List[Tuple[float, float]]:
    return [(0.0, float(len(LABELS) - 1))] + [(0.0, 1.0)] * N_REAL


def var_types() -> list:
    return [("cat", list(LABELS))] + ["real"] * N_REAL


def front(n: int):
    import numpy as np
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])

# SPDX-License-Identifier: Apache-2.0
"""What a batch of points must still compute one point at a time.

The problems evaluate a whole batch of points with NumPy (BenchProblem.
evaluate_batch; task 10 of 2026-10-18, §4.1), and must give, bit for bit, the
values the point-by-point code gave before. Elementwise arithmetic and sums
along a row are the same either way. These are not:

  * a matrix times a vector is BLAS's gemv, a matrix times the batch its gemm,
    which adds in another order: rows_matvec keeps one gemv per point (rWFG,
    rcPolygon); the rotations of the C++ suite's families take matvec_pairwise,
    an order of NumPy's that C++ repeats;
  * a dot product x @ x is BLAS's ddot, np.sum(x * x) a pairwise sum:
    rows_dot keeps one ddot per point;
  * x ** 2, x ** 0.5 and x ** -1 of an array are NumPy's x * x, sqrt and 1/x,
    of a Python float (or a NumPy scalar) the C library's pow, which on Windows
    differs in the last bit for about one value in two thousand: spow keeps
    the library's pow wherever the old code raised a scalar;
  * NumPy's sin, cos, exp and power of an array are the C library's on some
    machines only: with AVX-512 NumPy has an exp of its own and, on Linux,
    takes sin, cos and power from Intel's SVML, which differ from the C
    library's in the last bit now and then. Where the point code called
    math's of one float, the batch calls math_sin, math_cos and math_exp (and
    spow): NumPy's where the two agree on this machine (agrees), math's at
    every element where they do not;
  * math.tanh and math.hypot are not NumPy's tanh and hypot: each() keeps them;
  * a sum along the rows (np.sum(..., axis=1), np.mean) is the pairwise sum of
    each row, as of the row alone, only where the rows lie in memory one after
    the other: picking columns with an index array (W[:, js]) lays the result
    out by columns, and NumPy then adds each row's terms one by one — take
    np.ascontiguousarray first.
"""

from __future__ import annotations

import math

import numpy as np


def rows(X, n_vars: int) -> np.ndarray:
    """The batch as an (n, n_vars) float array."""
    return np.asarray(X, dtype=float).reshape(-1, n_vars)


def rows_matvec(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    """M @ x for every row x, each by itself (gemv)."""
    out = np.empty((len(X), M.shape[0]))
    for i in range(len(X)):
        out[i] = M @ X[i]
    return out


def matvec_pairwise(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """A d for a vector d, or for every row d of D, each entry the pairwise sum
    of its row's products, np.sum(A * d, axis=1): NumPy's order, which the C++
    suite repeats (include/mootation/suite/common.hpp, matvec). A @ d is BLAS's
    gemv, whose order is its own and may change with the machine and the
    library; the rotated problems of the suite's families take this one
    (task 11 of 2026-10-18; rotated.py)."""
    A = np.ascontiguousarray(A, dtype=float)
    D = np.asarray(D, dtype=float)
    if D.ndim == 1:
        return np.sum(A * D, axis=1)
    out = np.empty((len(D), A.shape[0]))
    for i in range(len(D)):
        out[i] = np.sum(A * D[i], axis=1)
    return out


def rows_dot(X: np.ndarray) -> np.ndarray:
    """x @ x for every row x, each by itself (ddot)."""
    return np.array([x @ x for x in X], dtype=float)


def pow_each(a: np.ndarray, e: float) -> np.ndarray:
    """a_i ** e by the C library's pow, as of a Python float, in a's shape."""
    a = np.asarray(a, dtype=float)
    return np.array([v ** e for v in a.ravel().tolist()], dtype=float).reshape(a.shape)


# The exponents NumPy's power computes otherwise than the C library's pow:
# x * x, sqrt(x) and 1 / x (checked on NumPy 2.5: every other exponent the
# tests met, integer or not, gives pow's value).
_NUMPY_OWN = (2.0, 0.5, -1.0)


def spow(a, e: float) -> np.ndarray:
    """a ** e elementwise with the value pow gives a scalar: NumPy's power where
    it agrees with pow on this machine, never at the exponents NumPy computes
    its own way."""
    e = float(e)
    if e not in _NUMPY_OWN and agrees(("pow", e), lambda x: np.power(x, e),
                                      lambda v: v ** e, "pow" if e.is_integer() else "pow+"):
        return np.power(a, e)
    return pow_each(a, e)


def each(fn, a) -> np.ndarray:
    """fn (a math function of one float) at every element of a, in a's shape,
    where NumPy's own version differs from the C library's."""
    a = np.asarray(a, dtype=float)
    return np.array([fn(v) for v in a.ravel().tolist()], dtype=float).reshape(a.shape)


# Whether NumPy's function of an array gives, bit for bit, the C library's of
# each float, on a fixed sample of 20 000 values: two implementations that
# differ at all differ on far more than one value in 20 000, and one
# implementation (the same C library) agrees everywhere. Decided once per
# process and function.
_AGREES: dict = {}


def _sample(kind: str) -> np.ndarray:
    rng = np.random.default_rng(20261018)
    if kind == "exp":
        return np.concatenate([rng.uniform(-50.0, 50.0, 10000), rng.uniform(-700.0, 700.0, 10000)])
    if kind == "pow+":                                # a fractional power: positive bases
        return np.concatenate([rng.uniform(0.0, 2.0, 10000), 10.0 ** rng.uniform(-6, 6, 10000)])
    if kind == "pow":                                 # an integer power: of either sign
        return np.concatenate([rng.uniform(-2.0, 2.0, 10000), rng.uniform(-50.0, 50.0, 10000)])
    return np.concatenate([rng.uniform(-10.0, 10.0, 10000), rng.uniform(-1e4, 1e4, 10000)])


def agrees(key, numpy_fn, math_fn, kind: str) -> bool:
    """numpy_fn of an array is math_fn of each of its floats, on this machine."""
    if key not in _AGREES:
        x = _sample(kind)
        theirs = np.array([math_fn(v) for v in x.tolist()], dtype=float)
        with np.errstate(all="ignore"):
            _AGREES[key] = np.asarray(numpy_fn(x), dtype=float).tobytes() == theirs.tobytes()
    return _AGREES[key]


def math_sin(a) -> np.ndarray:
    """math.sin of every element of a."""
    return np.sin(a) if agrees("sin", np.sin, math.sin, "trig") else each(math.sin, a)


def math_cos(a) -> np.ndarray:
    """math.cos of every element of a."""
    return np.cos(a) if agrees("cos", np.cos, math.cos, "trig") else each(math.cos, a)


def math_exp(a) -> np.ndarray:
    """math.exp of every element of a."""
    return np.exp(a) if agrees("exp", np.exp, math.exp, "exp") else each(math.exp, a)

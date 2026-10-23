# SPDX-License-Identifier: Apache-2.0
"""Distinct decoded points: the count, and a cache for budget_counts = "unique"
(task 7 of 2026-10-10, supplement 3, §1.2).

An algorithm asks for the same point more often than one expects: elitist
copies, and with integer, discrete or categorical variables every child that
rounds back onto a point already seen. minimize() runs fn, constraints and
batch through Points at the caller's side of var_types, so that the key is
the DECODED point — integers rounded, discrete values snapped to their list,
categorical variables as their labels — not the real "underlay" the core
mutates. Points counts the distinct ones (result.unique_evaluations); with
cache=True it also answers a repeat from memory without calling fn, and the
core counts its budget in distinct points (budget_counts = "unique"), which
is what a solver run costs. The default, "calls", calls fn for every request
as before: a run is unchanged bit for bit, a stochastic fn draws anew.

The key is the point itself, exact: a repeat is a copy, not a neighbour.

THE COUNT'S MEMORY. A run of 10^6 evaluations of 100 variables kept every
point as a tuple of Python floats, about 3 kB a point: 3 GB a run, and a
campaign of eighteen such runs at once took the machine down (task 17,
2026-10-23). The distinct points are now remembered by a 16-byte digest of
the key (BLAKE2b): numbers as float64, -0.0 as 0.0 (they are equal), a NaN
equal to nothing (as in a tuple of fresh floats), anything else by its repr.
Equal keys give equal digests; two different points share one with a
probability of about 2^-128 a pair. The cache of budget_counts = "unique"
keeps the points themselves: it answers with their values.
"""

from __future__ import annotations

import hashlib
import itertools
import struct

_NAN = itertools.count()          # every NaN a value of its own


def _digest(k) -> bytes:
    """16 bytes standing for the key k (a tuple, _key's): see the module's docstring."""
    parts = []
    for v in k:
        if isinstance(v, (int, float)) and not (isinstance(v, int) and abs(v) > 2 ** 53):
            if v != v:
                parts.append(b"n" + next(_NAN).to_bytes(8, "little"))
            else:
                parts.append(b"d" + struct.pack("<d", float(v) + 0.0))
        else:
            r = repr(v).encode()
            parts.append(b"r" + len(r).to_bytes(4, "little") + r)
    return hashlib.blake2b(b"".join(parts), digest_size=16).digest()


def row_digests(X) -> list:
    """_digest of every row of a float64 array (n x d), the same bytes laid
    out at once: "d" and the eight bytes of each value."""
    import numpy as np
    X = np.ascontiguousarray(np.asarray(X, np.float64).reshape(len(X), -1) + 0.0)   # -0.0 as 0.0
    n, d = X.shape
    B = np.empty((n, d, 9), np.uint8)
    B[:, :, 0] = ord("d")
    B[:, :, 1:] = X.view(np.uint8).reshape(n, d, 8)
    nan = np.isnan(X).any(axis=1)
    return [_digest(X[i].tolist()) if nan[i] else
            hashlib.blake2b(B[i].tobytes(), digest_size=16).digest() for i in range(n)]


def _key(x) -> tuple:
    try:
        k = tuple(x)
        hash(k)
        return k
    except TypeError:                     # an unhashable label
        return tuple(repr(v) for v in x)


class Points:
    def __init__(self, cache: bool = False):
        self.cache = cache
        self.seen: set = set()
        self.f: dict = {}
        self.g: dict = {}

    @property
    def count(self) -> int:
        return len(self.seen)

    def fn(self, f):
        def counted(x, _f=f):
            k = _key(x)
            if self.cache and k in self.f:
                return self.f[k]
            out = list(_f(x))
            self.seen.add(_digest(k))
            if self.cache:
                self.f[k] = out
            return out
        return counted

    def constraints(self, c):
        def counted(x, _c=c):
            k = _key(x)
            if self.cache and k in self.g:
                return self.g[k]
            out = list(_c(x))
            if self.cache:
                self.g[k] = out
            return out
        return counted

    def batch(self, b):
        def counted(X, _b=b):
            keys = [_key(x) for x in X]
            todo = [i for i, k in enumerate(keys) if not (self.cache and k in self.f)]
            rows = _b([X[i] for i in todo]) if todo else []
            got = dict(zip(todo, rows))
            out = []
            for i, k in enumerate(keys):
                if i in got:
                    row = got[i]
                    self.seen.add(_digest(k))
                    if self.cache and row is not None:
                        self.f[k] = list(row)
                    out.append(row)
                else:
                    out.append(self.f[k])
            return out
        return counted

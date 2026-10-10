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
"""

from __future__ import annotations


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
            self.seen.add(k)
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
                    self.seen.add(k)
                    if self.cache and row is not None:
                        self.f[k] = list(row)
                    out.append(row)
                else:
                    out.append(self.f[k])
            return out
        return counted

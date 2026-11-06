# SPDX-License-Identifier: Apache-2.0
"""Symmetries of a problem (task 19, §19.2.1): the Python cross-check of
include/mootation/scalar/symmetry.hpp.

blocks: a list of (offset, size, count) — count blocks of size coordinates
from offset on may be permuted. align(x, ref, blocks, canon=None): canon(x)
where given, then in every group the permutation of x's blocks that
minimizes the sum of squared distances to ref's blocks (the Hungarian
method, Munkres 1957, §1; the readings SYM-1 ... SYM-4 of the header).
distance(x, y) = ||align(x, y) - y||. sort_blocks(x, blocks): the built-in
canon of sym = canon where the problem gives none (SYM-5).
"""

from __future__ import annotations

import math


def hungarian(c):
    """The assignment of least cost of the square matrix c (a list of rows):
    a list, the column given to each row. Munkres's steps, the first
    non-covered zero in row-major order (SYM-1)."""
    n = len(c)
    if n == 0:
        return []
    a = [list(map(float, row)) for row in c]
    if any(len(row) != n for row in a):
        raise ValueError("hungarian: an n x n matrix is needed")
    if not all(math.isfinite(v) for row in a for v in row):
        raise ValueError("hungarian: the entries must be finite")
    for i in range(n):
        m = min(a[i])
        a[i] = [v - m for v in a[i]]
    for j in range(n):
        m = min(a[i][j] for i in range(n))
        for i in range(n):
            a[i][j] -= m
    none = n
    star_row, star_col, prime_row = [none] * n, [none] * n, [none] * n
    row_cov, col_cov = [False] * n, [False] * n
    for i in range(n):
        for j in range(n):
            if a[i][j] == 0.0 and star_row[i] == none and star_col[j] == none:
                star_row[i], star_col[j] = j, i

    def cover_starred():
        for j in range(n):
            col_cov[j] = star_col[j] != none
        return sum(col_cov)

    covered = cover_starred()
    while covered < n:
        z = None
        for i in range(n):
            if row_cov[i]:
                continue
            for j in range(n):
                if not col_cov[j] and a[i][j] == 0.0:
                    z = (i, j)
                    break
            if z is not None:
                break
        if z is None:                                  # Step 3
            h = min(a[i][j] for i in range(n) if not row_cov[i] for j in range(n) if not col_cov[j])
            for i in range(n):
                for j in range(n):
                    if row_cov[i]:
                        a[i][j] += h
                    if not col_cov[j]:
                        a[i][j] -= h
            continue
        i, j = z
        prime_row[i] = j
        if star_row[i] != none:
            row_cov[i] = True
            col_cov[star_row[i]] = False
            continue
        while True:                                    # Step 2
            i_star = star_col[j]
            star_row[i], star_col[j] = j, i
            if i_star == none:
                break
            j_prime = prime_row[i_star]
            star_row[i_star] = none
            i, j = i_star, j_prime
        prime_row = [none] * n
        row_cov = [False] * n
        covered = cover_starred()
    return list(star_row)


def check(blocks, n):
    """The groups within n variables, not overlapping (SYM-4)."""
    used = [False] * n
    for off, size, count in blocks:
        if size < 1 or count < 1 or off + size * count > n:
            raise ValueError("symmetry: a group of blocks needs size, count >= 1 within the variables")
        for k in range(off, off + size * count):
            if used[k]:
                raise ValueError("symmetry: the groups of blocks overlap")
            used[k] = True


def align(x, ref, blocks, canon=None):
    """x aligned to ref: canon first (where given), then each group's blocks
    permuted to ref's."""
    if len(x) != len(ref):
        raise ValueError("symmetry: x and ref differ in length")
    y = [float(v) for v in (canon(list(x)) if canon is not None else x)]
    out = list(y)
    for off, s, m in blocks:
        cost = [[_squares(y, ref, off + i * s, off + j * s, s) for j in range(m)] for i in range(m)]
        for i, j in enumerate(hungarian(cost)):
            out[off + j * s:off + (j + 1) * s] = y[off + i * s:off + (i + 1) * s]
    return out


def _squares(a, b, i0, j0, s):
    """sum (a[i0 + k] - b[j0 + k])^2 in order, as the header adds (CPython's
    sum() of floats is compensated since 3.12)."""
    t = 0.0
    for k in range(s):
        d = a[i0 + k] - b[j0 + k]
        t += d * d
    return t


def sort_blocks(x, blocks):
    """x with each group's blocks sorted lexicographically (SYM-5)."""
    out = [float(v) for v in x]
    for off, s, m in blocks:
        parts = sorted(out[off + k * s:off + (k + 1) * s] for k in range(m))
        for k, part in enumerate(parts):
            out[off + k * s:off + (k + 1) * s] = part
    return out


def distance(x, y, blocks, canon=None) -> float:
    a = align(x, y, blocks, canon)
    return math.sqrt(_squares(a, y, 0, 0, len(y)))

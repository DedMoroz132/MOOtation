# SPDX-License-Identifier: Apache-2.0
"""The Remez exchange algorithm on Python's standard library (decimal for the
function values, fractions for the linear systems), for the polynomial
coefficients of include/mootation/detmath.hpp (task 14 of 2026-10-20,
supplement 2, D7.3).

remez(F, W, a, b, powers) finds the coefficients c_k of
P(t) = sum_k c_k t^powers[k] that minimise max over [a, b] of
|W(t) (F(t) - P(t))|, F and W given as functions of a Decimal returning a
Decimal. The steps (Muller 2016, §3.6, Remez's algorithm):

  1. a reference of n + 1 points (n = len(powers)): the extrema of the
     Chebyshev polynomial of degree n mapped onto [a, b];
  2. solve P(t_i) + (-1)^i E / W(t_i) = F(t_i), i = 0..n, for the c_k and E
     (exactly, in fractions);
  3. the new reference: on a grid refined around every sign change, the
     point of largest |error| in each run of one sign — n + 1 alternating
     points, the global extremum among them;
  4. stop when the largest |error| is within 1e-4 of |E| (relative).

round_to_double(...) then rounds the coefficients to doubles one at a time
from the highest power down, re-solving for the lower ones after each
rounding (the high coefficients matter least, so their rounding is absorbed
by the others) — the simple scheme; Brisebarre and Chevillard 2007 only if
it is not accurate enough.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

PREC = 80


def _D(v):
    return v if isinstance(v, Decimal) else Decimal(v)


def _solve(rows, rhs):
    """Gaussian elimination in fractions (exact)."""
    n = len(rows)
    A = [[Fraction(v) for v in row] + [Fraction(r)] for row, r in zip(rows, rhs)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(A[i][col]))
        A[col], A[piv] = A[piv], A[col]
        for i in range(n):
            if i != col and A[i][col] != 0:
                f = A[i][col] / A[col][col]
                A[i] = [x - f * y for x, y in zip(A[i], A[col])]
    return [A[i][n] / A[i][i] for i in range(n)]


def _pw(t, p):
    """t^p with 0^0 = 1 (decimal calls 0 ** 0 invalid)."""
    return Decimal(1) if p == 0 else t ** p


def _poly(coeffs, powers, t):
    s = Decimal(0)
    for c, p in zip(coeffs, powers):
        s += c * _pw(t, p)
    return s


def _cheb_reference(a, b, m):
    """m points: the extrema of T_{m-1} on [a, b], ascending."""
    from math import cos, pi
    pts = []
    for i in range(m):
        x = -cos(pi * i / (m - 1))
        pts.append((a + b) / 2 + (b - a) / 2 * Decimal(repr(x)))
    return pts


def _error(F, W, coeffs, powers, t):
    return W(t) * (F(t) - _poly(coeffs, powers, t))


def _new_reference(F, W, coeffs, powers, a, b, m, grid):
    """The point of largest |error| in each run of one sign on the grid
    (refined by golden-section search between its grid neighbours), m of
    them; the largest |error| seen."""
    ts = [a + (b - a) * i / grid for i in range(grid + 1)]
    es = [_error(F, W, coeffs, powers, t) for t in ts]
    runs = []                                   # (index of the run's max, sign)
    for i, e in enumerate(es):
        if e == 0:
            continue
        if runs and (runs[-1][1] > 0) == (e > 0):
            if abs(e) > abs(es[runs[-1][0]]):
                runs[-1] = (i, e)
        else:
            runs.append((i, e))
    pts = []
    for i, e in runs:
        lo = ts[i - 1] if i > 0 else ts[i]
        hi = ts[i + 1] if i + 1 < len(ts) else ts[i]
        t, _ = _golden(lambda u: abs(_error(F, W, coeffs, powers, u)), lo, hi, ts[i], abs(e))
        pts.append((t, _error(F, W, coeffs, powers, t)))
    # too many runs: drop the smaller end until m remain (keeping alternation)
    while len(pts) > m:
        if abs(pts[0][1]) < abs(pts[-1][1]):
            pts.pop(0)
        else:
            pts.pop()
    if len(pts) < m:
        raise RuntimeError(f"only {len(pts)} alternating extrema, need {m}")
    return [t for t, _ in pts], max(abs(e) for _, e in pts)


def _golden(f, lo, hi, t0, f0, iters=60):
    """Maximise f on [lo, hi] (golden section), starting knowledge t0, f0."""
    if hi <= lo:
        return t0, f0
    g = (Decimal(5).sqrt() - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    t = (a + b) / 2
    ft = f(t)
    best = max([(f0, t0), (ft, t), (f(lo), lo), (f(hi), hi)], key=lambda p: p[0])
    return best[1], best[0]


def remez(F, W, a, b, powers, grid=2000, max_iter=30, fixed=()):
    """Coefficients (Decimal) of the minimax P for powers on [a, b], and the
    level |E|. fixed: (power, value) pairs already decided (rounded); they
    stay in P and are not solved for."""
    with localcontext() as ctx:
        ctx.prec = PREC
        a, b = _D(a), _D(b)
        fixed = list(fixed)
        free = [p for p in powers if p not in dict(fixed)]

        def G(t):                                  # F minus the fixed part
            return F(t) - sum((Decimal(v) * _pw(t, p) for p, v in fixed), Decimal(0))

        m = len(free) + 1
        ref = _cheb_reference(a, b, m)
        if W(a) == 0:                              # a weight vanishing at a: no
            ref[0] = a + (ref[1] - a) / 8          # extremum there, start inside
        level = None
        for _ in range(max_iter):
            rows, rhs = [], []
            for i, t in enumerate(ref):
                rows.append([Fraction(_pw(t, p)) for p in free] + [Fraction((-1) ** i) / Fraction(W(t))])
                rhs.append(Fraction(G(t)))
            sol = _solve(rows, rhs)
            coeffs = [Decimal(c.numerator) / Decimal(c.denominator) for c in sol[:-1]]
            E = abs(Decimal(sol[-1].numerator) / Decimal(sol[-1].denominator))
            ref, worst = _new_reference(G, W, coeffs, free, a, b, m, grid)
            level = worst
            if E > 0 and (worst - E) / E < Decimal("1e-4"):
                break
        out = dict(fixed)
        out.update({p: c for p, c in zip(free, coeffs)})
        return [out[p] for p in powers], level


def round_to_double(F, W, a, b, powers, keep=(), **kw):
    """Doubles for every coefficient: round the highest free power, re-solve
    the rest, repeat. keep: (power, double) fixed from the start (e.g. the
    leading 1 of r + r^3 ...). Returns (doubles, level of the rounded P)."""
    fixed = list(keep)
    for p in sorted((p for p in powers if p not in dict(fixed)), reverse=True):
        coeffs, _ = remez(F, W, a, b, powers, fixed=fixed, **kw)
        fixed.append((p, float(coeffs[powers.index(p)])))
    d = dict(fixed)
    doubles = [d[p] for p in powers]
    with localcontext() as ctx:
        ctx.prec = PREC
        dec = [Decimal(v) for v in doubles]
        A, B = _D(a), _D(b)
        grid = kw.get("grid", 2000)
        level = max(abs(_error(F, W, dec, powers, A + (B - A) * i / (4 * grid))) for i in range(4 * grid + 1))
    return doubles, level

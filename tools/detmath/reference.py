# SPDX-License-Identifier: Apache-2.0
"""High-precision values of the functions of include/mootation/detmath.hpp, on
Python's standard library only (decimal), for the accuracy tests (task 14 of
2026-10-20, supplement 2, D7.3).

Every value is computed with DIGITS significant decimal digits and returned as
a Decimal; `ulp_error(got, exact)` is the error of a double in units in the last
place of the exact value. decimal has exp, ln and sqrt of its own; pi, the sine
and cosine and the arctangent are the series below:

    pi          Machin: 16 atan(1/5) - 4 atan(1/239)
    sin, cos    x = k pi/2 + r, |r| <= pi/4 (pi to DIGITS + the digits of k),
                then the Taylor series of sin r and cos r
    atan        atan x = 2 atan(x / (1 + sqrt(1 + x^2))) until |x| < 1/8,
                then the Taylor series
    acos        2 atan(sqrt((1 - x) / (1 + x))); acos(-1) = pi
    pow         exp(y ln |x|), the sign of x^y for x < 0 and an odd integer y
    hypot       sqrt(x^2 + y^2)

A double converts to a Decimal exactly (Decimal(float)), and float(Decimal) is
correctly rounded, so the correctly rounded result of f(x) is
float(value(f, x)). The special inputs (zeros, infinities, NaN, overflow) are
not handled here: the tests take those from C23 Annex F directly.
"""

import math
from decimal import Decimal, localcontext

DIGITS = 60                     # significant digits of every value


def _atan_small(x, prec):
    """atan x for |x| < 1/8 by its Taylor series, at prec digits."""
    with localcontext() as c:
        c.prec = prec
        x2 = x * x
        term, total, n = x, x, 1
        eps = Decimal(10) ** (-prec - 2)
        while True:
            term = -term * x2
            n += 2
            t = term / n
            if abs(t) < eps * (abs(total) + eps):
                return total
            total += t


def _atan(x, prec):
    """atan x for any finite x, at prec digits (argument halving)."""
    with localcontext() as c:
        c.prec = prec + 10
        halvings = 0
        while abs(x) >= Decimal(1) / 8:
            x = x / (1 + (1 + x * x).sqrt())
            halvings += 1
        return _atan_small(x, prec + 10) * (2 ** halvings)


_PI = {}


def pi(prec):
    """pi to prec digits (cached)."""
    if prec not in _PI:
        with localcontext() as c:
            c.prec = prec + 10
            p = 16 * _atan_small(Decimal(1) / 5, prec + 10) - 4 * _atan_small(Decimal(1) / 239, prec + 10)
        with localcontext() as c:
            c.prec = prec
            _PI[prec] = +p
    return _PI[prec]


def _sin_cos_series(r, prec):
    """(sin r, cos r) for |r| <= pi/4 by their Taylor series."""
    with localcontext() as c:
        c.prec = prec
        r2 = r * r
        eps = Decimal(10) ** (-prec - 2)
        s, term, n = r, r, 1
        while abs(term) > eps:
            term = -term * r2 / ((n + 1) * (n + 2))
            n += 2
            s += term
        co, term, n = Decimal(1), Decimal(1), 0
        while abs(term) > eps:
            term = -term * r2 / ((n + 1) * (n + 2))
            n += 2
            co += term
        return s, co


def _reduce(x):
    """(k mod 4, r) with x = k pi/2 + r, |r| <= pi/4, r to DIGITS digits."""
    ax = abs(x)
    extra = max(0, ax.adjusted()) + 10        # the digits k takes from pi
    prec = DIGITS + extra + 10
    with localcontext() as c:
        c.prec = prec
        half_pi = pi(prec) / 2
        k = (x / half_pi).to_integral_value()           # round half even
        r = x - k * half_pi
        return int(k) % 4, r


def sin_cos(x):
    k, r = _reduce(Decimal(x))
    s, co = _sin_cos_series(r, DIGITS + 10)
    return [(s, co), (co, -s), (-s, -co), (-co, s)][k]


def value(f, x, y=None):
    """The value of f at the double x (and y), as a Decimal of DIGITS digits."""
    with localcontext() as c:
        c.prec = DIGITS + 20
        X = Decimal(x)
        if f == "exp":
            v = X.exp()
        elif f == "log":
            v = X.ln()
        elif f == "sin":
            v = sin_cos(x)[0]
        elif f == "cos":
            v = sin_cos(x)[1]
        elif f == "tan":
            s, co = sin_cos(x)
            v = s / co
        elif f == "acos":
            if x == -1.0:
                v = pi(DIGITS + 20)
            else:
                v = 2 * _atan(((1 - X) / (1 + X)).sqrt(), DIGITS + 20)
        elif f == "pow":
            Y = Decimal(y)
            if X == 0:
                raise ValueError("pow at x = 0: Annex F, not this function")
            t = Y * abs(X).ln()
            if t > 800:                        # beyond any double: overflow
                v = Decimal("Infinity")
            elif t < -800:                     # below half the least subnormal
                v = Decimal(0)
            else:
                v = t.exp()
            if X < 0:
                if Y != Y.to_integral_value():
                    raise ValueError("pow of a negative x at a non-integer y")
                if int(Y) % 2:
                    v = -v
        elif f == "hypot":
            Y = Decimal(y)
            v = (X * X + Y * Y).sqrt()
        else:
            raise ValueError(f)
    with localcontext() as c:
        c.prec = DIGITS
        return +v


def ulp_of(exact):
    """The ulp of the exact value: 2^(e - 52) for 2^e <= |exact| < 2^(e+1),
    2^-1074 below the normal range: the ulp of the real number, not of a
    double near it."""
    a = abs(exact)
    if a == 0:
        return Decimal(2) ** -1074
    r = abs(float(a))
    if r == math.inf:
        r = math.ldexp(1.0, 1024)
    e = math.frexp(r)[1] - 1                   # 2^e <= r < 2^(e+1)
    with localcontext() as c:
        c.prec = DIGITS + 20
        if Decimal(2) ** e > a:                # a just below a power of two
            e -= 1
    return Decimal(2) ** max(e - 52, -1074)


def ulp_error(got, exact):
    """|got - exact| / ulp(exact), as a float."""
    with localcontext() as c:
        c.prec = DIGITS + 20
        return float(abs(Decimal(got) - exact) / ulp_of(exact))


def double_double(exact):
    """(hi, lo): hi = the correctly rounded double, lo = the rounded rest."""
    hi = float(exact)
    with localcontext() as c:
        c.prec = DIGITS + 20
        lo = float(exact - Decimal(hi))
    return hi, lo

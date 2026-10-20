# SPDX-License-Identifier: Apache-2.0
"""Writes include/mootation/detmath_coeffs.hpp: the constants and polynomial
coefficients of include/mootation/detmath.hpp, as hex-float literals (task 14
of 2026-10-20, supplement 2, D7.3). Python's standard library only.

    python tools/detmath/gen_coeffs.py [--check]

--check compares the header in the tree with what this script makes (exit 1
on a difference) instead of writing it.

Constants: the parts of pi/2 for the Cody-Waite reduction of sin, cos and tan
(three of 33 bits, so that k * part is exact for k <= 2^20, and the rest
rounded), pi/2 and pi as hi + lo, 2/pi; the values of the library's calls on
constants (exp(8) etc.), correctly rounded.

Polynomials (Remez, remez.py; coefficients rounded one at a time from the
highest, the rest re-solved): for |r| <= pi/4 and t = r^2,
    sin r = r + r^3 S(t),      S of degree 5,
    cos r = 1 - t/2 + t^2 C(t),  C of degree 5;
and for 0 <= s <= 1/2, t = s^2,
    asin s = s + s^3 A(t),     A of degree 12.
Each minimises the relative error of the whole function; the level of the
rounded polynomial is printed and written into the header.

exp (Tang, ACM TOMS 15(2), 1989): x = (32m + j) ln2/32 + r, |r| <= ln2/64;
ln2/32 = kExpL1 + kExpL2, kExpL1 with 33 significant bits (N kExpL1 exact for
|N| < 2^20); 2^(j/32) = kExpS[j][0] + kExpS[j][1], the lead with its six
lowest bits zero (Tang's choice); e^r - 1 ~ r + A1 r^2 + ... + A5 r^6 on
|r| <= 0.010831 (a little wider than ln2/64, as Tang). The thresholds: the
largest x with exp(x) finite, the smallest with exp(x) not rounding to 0.

log (Tang, ACM TOMS 16(4), 1990): outside (e^-1/16, e^1/16), x = 2^m (F + f),
F = 1 + j/128, |f| <= 2^-8 (Procedure 1): log x = m ln2 + log F + log(1 + u),
u = f/F; ln2 and log F as lead + trail, the leads multiples of 2^-39 (so m
lead + lead is exact); log(1 + u) - u ~ u^2 (A1 + ... + u^4 A5) on
|u| <= 2^-8 (absolute error: |log x| >= 1/16 here). Inside (Procedure 2):
alpha = 2f/(2 + f), f = x - 1; log x = alpha + alpha^3 B(alpha^2),
B of degree 3 on alpha^2 <= 2^-8 (relative error). pow takes a third
polynomial: log(1 + u) - u + u^2/2 ~ u^3 (D0 + ... ) on |u| <= 2^-8.
"""

import math
import pathlib
import sys
from decimal import Decimal, localcontext

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reference as R        # noqa: E402
import remez                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "include" / "mootation" / "detmath_coeffs.hpp"
PREC = 80


def hexf(v):
    """A double as a C++ hex-float literal (exact)."""
    if v == 0:
        return "0.0"
    return float(v).hex()


def truncated(v, bits):
    """v (a Decimal) rounded to nearest with `bits` significant bits."""
    with localcontext() as c:
        c.prec = PREC
        e = math.frexp(float(v))[1]             # 2^(e-1) <= |v| < 2^e
        scale = Decimal(2) ** (bits - e)
        return float((v * scale).to_integral_value() / scale)


def pi_parts():
    """pi/2 = P1 + P2 + P3 + P4: P1..P3 with 33 bits, P4 rounded."""
    with localcontext() as c:
        c.prec = PREC
        rest = R.pi(PREC) / 2
        parts = []
        for _ in range(3):
            p = truncated(rest, 33)
            parts.append(p)
            rest -= Decimal(p)
        parts.append(float(rest))
        return parts


def hi_lo(v):
    with localcontext() as c:
        c.prec = PREC
        h = float(v)
        return h, float(v - Decimal(h))


def poly(name, F, W, b, degree, Wtrue):
    powers = list(range(degree + 1))
    doubles, _ = remez.round_to_double(F, W, 0, b, powers, grid=600)
    with localcontext() as c:
        c.prec = remez.PREC
        dec = [Decimal(v) for v in doubles]
        B = Decimal(b)
        level = max(abs(Wtrue(B * i / 4000) * (F(B * i / 4000) - remez._poly(dec, powers, B * i / 4000)))
                    for i in range(4001))
    print(f"{name}: degree {degree}, relative error of the rounded polynomial 2^{math.log2(float(level)):.2f}")
    return doubles, level


def poly2(name, F, W, a, b, degree, Wtrue):
    """poly on [a, b] (a < 0 allowed)."""
    powers = list(range(degree + 1))
    doubles, _ = remez.round_to_double(F, W, a, b, powers, grid=600)
    with localcontext() as c:
        c.prec = remez.PREC
        dec = [Decimal(v) for v in doubles]
        A, B = Decimal(a), Decimal(b)
        pts = [A + (B - A) * i / 4000 for i in range(4001)]
        level = max(abs(Wtrue(t) * (F(t) - remez._poly(dec, powers, t))) for t in pts)
    print(f"{name}: degree {degree}, relative error of the rounded polynomial 2^{math.log2(float(level)):.2f}")
    return doubles, level


def main():
    with localcontext() as c:
        c.prec = PREC
        pi = R.pi(PREC)
        pi4sq = (pi / 4) ** 2

        def sin_series(t):
            r = t.sqrt()
            return R._sin_cos_series(r, PREC)

        def sF(t):
            if t == 0:
                return Decimal(-1) / 6
            r = t.sqrt()
            return (sin_series(t)[0] - r) / (t * r)

        def sW(t):                                   # |r^3 dS| / |sin r|
            if t == 0:
                return Decimal(0)
            r = t.sqrt()
            return t * r / sin_series(t)[0]

        def cF(t):
            if t == 0:
                return Decimal(1) / 24
            return (sin_series(t)[1] - 1 + t / 2) / (t * t)

        def cWtrue(t):                               # |t^2 dC| / cos r
            return t * t / sin_series(t)[1]

        def cW(t):                                   # nonzero at 0: Remez keeps its extrema
            tt = t + pi4sq / 64
            return tt * tt / sin_series(t)[1]

        def aval(t):
            s = t.sqrt()
            return R._atan(s / (1 - t).sqrt(), PREC)

        def aF(t):
            if t == 0:
                return Decimal(1) / 6
            s = t.sqrt()
            return (aval(t) - s) / (t * s)

        def aWtrue(t):
            if t == 0:
                return Decimal(0)
            s = t.sqrt()
            return t * s / aval(t)

        def aW(t):
            if t == 0:
                return Decimal("0.25") / 64
            s = t.sqrt()
            return (t + Decimal("0.25") / 64) * s / aval(t)

        S, s_level = poly("sin", sF, sW, pi4sq, 5, sW)
        C, c_level = poly("cos", cF, cW, pi4sq, 5, cWtrue)
        A, a_level = poly("asin", aF, aW, Decimal("0.25"), 12, aWtrue)

        def eF(t):
            if t == 0:
                return Decimal(1) / 2
            return (t.exp() - 1 - t) / (t * t)

        ea = Decimal("0.010831")

        def eW(t):                                   # nonzero at 0
            tt = t * t + ea * ea / 64
            return tt / t.exp()

        def eWtrue(t):
            return t * t / t.exp()

        E, e_level = poly2("exp", eF, eW, -ea, ea, 4, eWtrue)
        ln2 = Decimal(2).ln()
        L1 = truncated(ln2 / 32, 33)
        L2 = float(ln2 / 32 - Decimal(L1))
        inv_l = float(32 / ln2)
        ES = []
        for j in range(32):
            v = (Decimal(j) / 32 * ln2).exp()
            lead = truncated(v, 47)                  # 53 - 6 bits
            ES.append((lead, float(v - Decimal(lead))))
        # the largest double x with exp(x) <= the largest double (no overflow
        # once rounded): exp(x) < (2 - 2^-53) 2^1023; the smallest x with
        # exp(x) > 2^-1075 (rounds to the least subnormal, not 0)
        big = Decimal(2) ** 1024 * (1 - Decimal(2) ** -54)
        x_over = float(big.ln())
        while Decimal(x_over).exp() >= big:
            x_over = math.nextafter(x_over, -math.inf)
        while Decimal(math.nextafter(x_over, math.inf)).exp() < big:
            x_over = math.nextafter(x_over, math.inf)
        tiny = Decimal(2) ** -1075
        x_under = float(tiny.ln())
        while Decimal(x_under).exp() <= tiny:
            x_under = math.nextafter(x_under, math.inf)
        while Decimal(math.nextafter(x_under, -math.inf)).exp() > tiny:
            x_under = math.nextafter(x_under, -math.inf)

        # log, Procedure 1
        two = Decimal(2)
        lg2 = two.ln()
        unit = two ** -39
        def lead39(v):
            return float((v / unit).to_integral_value() * unit)
        lg2_lead = lead39(lg2)
        lg2_trail = float(lg2 - Decimal(lg2_lead))
        CT = []
        for j in range(129):
            v = (1 + Decimal(j) / 128).ln()
            lead = lead39(v)
            CT.append((lead, float(v - Decimal(lead))))
        ua = Decimal(2) ** -8

        def lF(u):
            if u == 0:
                return Decimal(-1) / 2
            return ((1 + u).ln() - u) / (u * u)

        def lW(u):                                   # absolute, nonzero at 0
            return u * u + ua * ua / 64

        def lWtrue(u):
            return u * u

        LA, l_level = poly2("log, Procedure 1", lF, lW, -ua, ua, 4, lWtrue)

        def l3F(u):                                  # (log(1+u) - u + u^2/2) / u^3
            if u == 0:
                return Decimal(1) / 3
            return ((1 + u).ln() - u + u * u / 2) / (u * u * u)

        def l3W(u):                                  # relative to log(1+u) ~ u
            return u * u + ua * ua / 64

        def l3Wtrue(u):
            return u * u

        LD, d_level = poly2("log tail for pow", l3F, l3W, -ua, ua, 5, l3Wtrue)
        va = Decimal("0.0625") ** 2

        def l2F(v):                                  # (log((1+a/2)/(1-a/2)) - a) / a^3
            if v == 0:
                return Decimal(1) / 12
            a = v.sqrt()
            return (((1 + a / 2) / (1 - a / 2)).ln() - a) / (a * v)

        def l2W(v):
            return v + va / 64

        def l2Wtrue(v):
            return v

        LB, b_level = poly2("log, Procedure 2", l2F, l2W, 0, va, 3, l2Wtrue)
        e16 = (Decimal(1) / 16).exp()
        t2 = float(e16)
        if Decimal(t2) <= e16:
            t2 = math.nextafter(t2, math.inf)
        em16 = (Decimal(-1) / 16).exp()
        t1 = float(em16)
        if Decimal(t1) >= em16:
            t1 = math.nextafter(t1, -math.inf)

        P = pi_parts()
        pio2 = hi_lo(pi / 2)
        pi_hl = hi_lo(pi)
        two_over_pi = float(2 / pi)
        consts = {
            "kPi": float(pi),
            "kLn2": float(Decimal(2).ln()),
            "kExp8": float(Decimal(8).exp()),
            "kExp40": float(Decimal(40).exp()),
            "kExpMinus40": float(Decimal(-40).exp()),
            "kExpMinus04": float(Decimal("-0.4").exp()),
            "kExpMinus1": float(Decimal(-1).exp()),
        }

    lines = [
        "#pragma once",
        "// SPDX-License-Identifier: Apache-2.0",
        "// ============================================================================",
        "// GENERATED by tools/detmath/gen_coeffs.py — do not edit by hand.",
        "// The constants and polynomial coefficients of detmath.hpp (task 14 of",
        "// 2026-10-20, supplement 2), exact as hex-float literals. Remez on Python's",
        "// standard library; the relative error of each rounded polynomial on its",
        "// interval is given beside it.",
        "// ============================================================================",
        "",
        "namespace mootation::dm::coeffs {",
        "",
        "// pi/2 = kPio2Part[0] + [1] + [2] + [3]: the first three with 33 significant",
        "// bits (k * part exact for k <= 2^20), the last the rest rounded (Cody-Waite)",
        "inline constexpr double kPio2Part[4] = {" + ", ".join(hexf(v) for v in P) + "};",
        f"inline constexpr double kPio2Hi = {hexf(pio2[0])};",
        f"inline constexpr double kPio2Lo = {hexf(pio2[1])};",
        f"inline constexpr double kPiHi = {hexf(pi_hl[0])};",
        f"inline constexpr double kPiLo = {hexf(pi_hl[1])};",
        f"inline constexpr double kTwoOverPi = {hexf(two_over_pi)};",
        "",
        "// correctly rounded values of the library's calls on constants",
    ]
    for k, v in consts.items():
        lines.append(f"inline constexpr double {k} = {hexf(v)};")
    lines += [
        "",
        f"// sin r = r + r^3 S(r^2), |r| <= pi/4; relative error 2^{math.log2(float(s_level)):.1f}",
        "inline constexpr double kSin[6] = {" + ", ".join(hexf(v) for v in S) + "};",
        f"// cos r = 1 - r^2/2 + r^4 C(r^2), |r| <= pi/4; relative error 2^{math.log2(float(c_level)):.1f}",
        "inline constexpr double kCos[6] = {" + ", ".join(hexf(v) for v in C) + "};",
        f"// asin s = s + s^3 A(s^2), 0 <= s <= 1/2; relative error 2^{math.log2(float(a_level)):.1f}",
        "inline constexpr double kAsin[13] = {" + ", ".join(hexf(v) for v in A) + "};",
        "",
        "// exp (Tang 1989): 32/ln2; ln2/32 = kExpL1 + kExpL2 (kExpL1: 33 bits)",
        f"inline constexpr double kExpInvL = {hexf(inv_l)};",
        f"inline constexpr double kExpL1 = {hexf(L1)};",
        f"inline constexpr double kExpL2 = {hexf(L2)};",
        "// exp(x) overflows above kExpOver and rounds to 0 below kExpUnder",
        f"inline constexpr double kExpOver = {hexf(x_over)};",
        f"inline constexpr double kExpUnder = {hexf(x_under)};",
        f"// e^r - 1 ~ r + r^2 (A1 + r (A2 + ... + r A5)), |r| <= 0.010831; relative error 2^{math.log2(float(e_level)):.1f}",
        "inline constexpr double kExpA[5] = {" + ", ".join(hexf(v) for v in E) + "};",
        "// log (Tang 1990): Procedure 2 for kLogT1 < x < kLogT2 (e^-1/16, e^1/16)",
        f"inline constexpr double kLogT1 = {hexf(t1)};",
        f"inline constexpr double kLogT2 = {hexf(t2)};",
        "// ln2 = kLog2Lead + kLog2Trail, the lead a multiple of 2^-39",
        f"inline constexpr double kLog2Lead = {hexf(lg2_lead)};",
        f"inline constexpr double kLog2Trail = {hexf(lg2_trail)};",
        f"// log(1 + u) - u ~ u^2 (A1 + u (A2 + ... + u A5)), |u| <= 2^-8; absolute error 2^{math.log2(float(l_level)):.1f}",
        "inline constexpr double kLogA[5] = {" + ", ".join(hexf(v) for v in LA) + "};",
        f"// log((1 + a/2)/(1 - a/2)) - a ~ a^3 B(a^2), a^2 <= 2^-8; relative error 2^{math.log2(float(b_level)):.1f}",
        "inline constexpr double kLogB[4] = {" + ", ".join(hexf(v) for v in LB) + "};",
        f"// pow: log(1 + u) - u + u^2/2 ~ u^3 D(u), |u| <= 2^-8; relative error 2^{math.log2(float(d_level)):.1f}",
        "inline constexpr double kLogD[6] = {" + ", ".join(hexf(v) for v in LD) + "};",
        "// log(1 + j/128) = kLogC[j][0] + kLogC[j][1], j = 0..128, leads multiples of 2^-39",
        "inline constexpr double kLogC[129][2] = {",
    ] + [f"    {{{hexf(a)}, {hexf(b)}}}," for a, b in CT] + [
        "};",
        "// 2^(j/32) = kExpS[j][0] + kExpS[j][1], the first with its six lowest bits 0",
        "inline constexpr double kExpS[32][2] = {",
    ] + [f"    {{{hexf(a)}, {hexf(b)}}}," for a, b in ES] + [
        "};",
        "",
        "}   // namespace mootation::dm::coeffs",
        "",
    ]
    text = "\n".join(lines)
    if "--check" in sys.argv:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != text:
            print("detmath_coeffs.hpp differs from what gen_coeffs.py makes")
            sys.exit(1)
        print("detmath_coeffs.hpp is up to date")
        return
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

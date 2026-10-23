#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Elementary functions with the same bits on every platform (task 14 of
// 2026-10-20, §2 and supplement 2). IEEE 754 rounds +, -, *, / and sqrt
// correctly, so a function built from those alone, with fixed constants in a
// fixed order, gives the same result wherever it runs — provided nothing
// fuses a multiply and an add (fp_contract.hpp, §3) and the arithmetic is
// SSE2/NEON double, not x87 (§5 item 4). The C standard does not fix the last
// bit of the C library's functions, so they may differ between platforms, and
// one seed then gives two runs.
//
// The functions, written for this library, in the namespace dm: exp, log,
// pow, sin, cos, tan, acos, hypot — the elementary functions the cores,
// operators, distributions and the suite of problems call (task 14, D7.1) —
// and asin and atan (task 17, supplement 3, stage 3a: the suite's MW5-8).
// Accuracy: within 1 ulp for exp, log, sin, cos; within 2 ulp for the rest;
// measured against values to 60 digits (tools/detmath/, tests/test_detmath.cpp,
// the table in tests/data/README.md). Special values (zeros, infinities,
// NaN, overflow) as C23 Annex F says (ISO/IEC 9899:2024, F.10); odd functions
// odd and even ones even, exactly (F.10, 3).
//
// sin, cos, tan: every finite x. Up to |x| = 2^20 pi/2 (about 1.6e6) the
// reduction takes pi/2 in four parts, three of 33 bits, so k * part is exact
// for k <= 2^20 (Cody & Waite); above it, Payne and Hanek's reduction by the
// bits of 2/pi (Muller, Elementary Functions, 3rd ed., 2016, §11.4; task 17,
// supplement 5, §4 — DOC4's sine takes arguments up to 3e8): the same
// accuracy, and the bits below 2^20 pi/2 those of before.
//
// Sources, by function:
//   exp      P. T. P. Tang, Table-driven implementation of the exponential
//            function in IEEE floating-point arithmetic, ACM TOMS 15(2), 1989
//   log      P. T. P. Tang, ... of the logarithm function ..., ACM TOMS 16(4),
//            1990 (Procedures 1 and 2)
//   pow      Tang's log and exp with the reduced argument, log x and y log x
//            carried as double-word numbers
//   sin, cos, tan
//            W. J. Cody, W. Waite, Software Manual for the Elementary
//            Functions, Prentice-Hall, 1980, ch. 8-9 (reduction by pi in
//            parts); above 2^20 pi/2 M. Payne, R. Hanek, SIGNUM Newsletter
//            18, 1983, as Muller 2016, §11.4, with its worst case for the
//            reduction (Table 11.2); kernels on |r| <= pi/4; tan as the
//            quotient of the two kernels in double-word arithmetic
//   acos     Cody & Waite, ch. 10 (asin of the half angle near +-1)
//   asin     ibid., ch. 10, acos's kernel
//   atan     ibid., ch. 11 (the reduction to |f| <= 2 - sqrt 3), the reduced
//            argument carried as a double-word number
//   hypot    C. F. Borges, Algorithm 1014, ACM TOMS 47(1), 2020: scaling by a
//            power of two, h = sqrt(x^2 + y^2) corrected by tau / (2h) as in
//            his Algorithm 5, with Dekker's exact products for his FMAs
// Double-word arithmetic without a fused multiply-add: Joldes, Muller &
// Popescu, ACM TOMS 44(2), 2017 (Algorithms 1, 2, 4, 6, 7, 17), with the
// bounds of Muller & Rideau, ACM TOMS 48(1), 2022; Veltkamp's splitting and
// Dekker's product (Dekker, Numer. Math. 18, 1971; Handbook of
// Floating-Point Arithmetic, 2nd ed., 2018, Algorithms 4.9 and 4.10).
// Polynomial coefficients: minimax by the Remez algorithm (J.-M. Muller,
// Elementary Functions, 3rd ed., 2016), tools/detmath/gen_coeffs.py, written
// to detmath_coeffs.hpp.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "detmath_coeffs.hpp"
#include "fp_contract.hpp"

// The helpers that return a struct (log_split, reduce_pio2, log_dw, ...):
// MSVC does not inline them on its own, and on x64 Windows a struct of more
// than 8 bytes comes back through memory, so the call and the reload cost
// more than their arithmetic (log took 16 ns instead of 4; task 14, D9-FMA).
// Inlining changes no value. GCC and Clang inline them already.
#if defined(_MSC_VER)
#define MOOTATION_FORCEINLINE __forceinline
#else
#define MOOTATION_FORCEINLINE inline
#endif

namespace mootation::dm {

// The values of the library's calls on constants, correctly rounded
// (tools/detmath/gen_coeffs.py): what acos(-1), log(2), exp(8), ... are.
inline constexpr double kPi = coeffs::kPi;
inline constexpr double kLn2 = coeffs::kLn2;
inline constexpr double kExp8 = coeffs::kExp8;
inline constexpr double kExp40 = coeffs::kExp40;
inline constexpr double kExpMinus40 = coeffs::kExpMinus40;
inline constexpr double kExpMinus04 = coeffs::kExpMinus04;
inline constexpr double kExpMinus1 = coeffs::kExpMinus1;

namespace detail {

// The bits of a double and back: memcpy, no union or pointer cast; a double
// and a uint64_t read as wholes, whatever the byte order.
inline std::uint64_t bits(double x)
{
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

inline double from_bits(std::uint64_t u)
{
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

// v rounded to the nearest integer, ties to even, for |v| < 2^51: adding and
// subtracting 1.5 2^52 leaves no fraction bits (two IEEE additions; no call
// into the C library, as std::floor or std::nearbyint may be).
inline double round_to_integer(double v)
{
    constexpr double shift = 0x1.8p52;
    return (v + shift) - shift;
}

// 2^k for -1022 <= k <= 1023, from its bits (exact).
inline double pow2i(std::int32_t k)
{
    return from_bits(static_cast<std::uint64_t>(k + 1023) << 52);
}

// The binary exponent of a normal positive double: 2^e <= x < 2^(e+1).
inline std::int32_t exponent_of(double x)
{
    return static_cast<std::int32_t>((bits(x) >> 52) & 0x7FF) - 1023;
}

// A double-word number: the unevaluated sum h + l, |l| <= ulp(h) / 2.
struct DW {
    double h, l;
};

// Fast2Sum (Joldes et al. 2017, Algorithm 1): s + t = a + b exactly when
// |a| >= |b| (or a or b is 0).
inline DW fast_two_sum(double a, double b)
{
    const double s = a + b;
    const double z = s - a;
    return {s, b - z};
}

// 2Sum (Joldes et al. 2017, Algorithm 2): s + t = a + b exactly, any a, b.
inline DW two_sum(double a, double b)
{
    const double s = a + b;
    const double a1 = s - b;
    const double b1 = s - a1;
    const double da = a - a1;
    const double db = b - b1;
    return {s, da + db};
}

// Veltkamp's splitting (Muller et al., Handbook of Floating-Point
// Arithmetic, 2nd ed., 2018, Algorithm 4.9; Dekker 1971, (6.1)-(6.2)) with
// s = 27, C = 2^27 + 1: x = x_h + x_l, x_h with 26 significant bits, x_l with
// 26. No overflow for |x| < 2^996.
inline DW split(double x)
{
    const double gamma = 134217729.0 * x;
    const double delta = x - gamma;
    const double xh = gamma + delta;
    return {xh, x - xh};
}

// Dekker's product (Handbook, Algorithm 4.10): r1 = RN(xy), r1 + r2 = xy
// exactly when the exponents of x and y add up to at least -969 (no
// underflow of the error) and nothing overflows (Theorem 4.8).
inline DW two_prod(double x, double y)
{
    const DW X = split(x);
    const DW Y = split(y);
    const double r1 = x * y;
    const double t1 = -r1 + X.h * Y.h;
    const double t2 = t1 + X.h * Y.l;
    const double t3 = t2 + X.l * Y.h;
    return {r1, t3 + X.l * Y.l};
}

// (x_h, x_l) + y: Joldes et al. 2017, Algorithm 4 (DWPlusFP); relative
// error <= 2u^2 (u = 2^-53).
inline DW dw_plus_fp(DW x, double y)
{
    const DW s = two_sum(x.h, y);
    const double v = x.l + s.l;
    return fast_two_sum(s.h, v);
}

// (x_h, x_l) + (y_h, y_l): Algorithm 6 (AccurateDWPlusDW); <= 3u^2 + 13u^3.
inline DW dw_plus_dw(DW x, DW y)
{
    const DW s = two_sum(x.h, y.h);
    const DW t = two_sum(x.l, y.l);
    const double c = s.l + t.h;
    const DW v = fast_two_sum(s.h, c);
    const double w = t.l + v.l;
    return fast_two_sum(v.h, w);
}

// (x_h, x_l) * y: Algorithm 7 (DWTimesFP1); <= 3u^2/2 + 4u^3.
inline DW dw_times_fp(DW x, double y)
{
    const DW c = two_prod(x.h, y);
    const double cl2 = x.l * y;
    const DW t = fast_two_sum(c.h, cl2);
    const double tl2 = t.l + c.l;
    return fast_two_sum(t.h, tl2);
}

// (x_h, x_l) / (y_h, y_l): Algorithm 17 (DWDivDW2); <= 15u^2 + 56u^3.
inline DW dw_div_dw(DW x, DW y)
{
    const double th = x.h / y.h;
    const DW r = dw_times_fp(y, th);
    const double ph = x.h - r.h;                       // exact
    const double dl = x.l - r.l;
    const double d = ph + dl;
    const double tl = d / y.h;
    return fast_two_sum(th, tl);
}

// ── sin, cos, tan ───────────────────────────────────────────────────────────

constexpr double kQuarterPi = 0x1.921fb54442d18p-1;   // pi/4 rounded
constexpr double kTrigMax = 0x1.921fb54442d18p+20;    // 2^20 pi/2 rounded

// x = k pi/2 + r, |r| <= pi/4 (a hair more where x * 2/pi rounds across a
// half), r as a double-word number; q = k mod 4. x >= 0, finite.
struct Reduced {
    DW r;
    std::uint32_t q;
};

MOOTATION_FORCEINLINE Reduced reduce_pio2(double x)
{
    if (x <= kQuarterPi)
        return {{x, 0.0}, 0};
    const double k = round_to_integer(x * coeffs::kTwoOverPi);
    const double* P = coeffs::kPio2Part;
    // k * P[0], k * P[1], k * P[2] are exact (k <= 2^20, 33-bit parts);
    // x - k P[0] is exact by Sterbenz's lemma (k P[0] is within a factor 2 of x)
    const double t = x - k * P[0];
    DW r = two_sum(t, -(k * P[1]));
    r = dw_plus_fp(r, -(k * P[2]));
    r = dw_plus_fp(r, -(k * P[3]));
    const std::uint32_t q = static_cast<std::uint32_t>(static_cast<std::int64_t>(k) & 3);
    return {r, q};
}

// Payne and Hanek's reduction (Muller 2016, §11.4) of x > 2^20 pi/2, finite:
// x = X 2^E, X the 53-bit integer significand, and x 2/pi = X 2^E sum b_i 2^-i
// over the bits b_i of 2/pi (coeffs::kTwoOverPiBits). A bit with i <= E - 2
// adds a multiple of 4 to x 2/pi — nothing to k mod 4 or to r — and is left
// out; the 192 bits from b_{max(1, E - 1)} on are taken, as the integer W, and
// the bits after them change the fraction of x 2/pi by less than 2^-137, while
// no double lies closer than 2^-61 to a multiple of pi/2 (Muller, Table 11.2:
// 6381956970095103 2^797). Z = X W, in 32-bit words; its bits from sh = i0 +
// 191 - E up give k mod 4, the 192 below sh the fraction f, k rounded to the
// nearest (f >= 1/2: k + 1, f - 1); r = f pi/2, a double-word number from the
// first 159 bits of |f| on.
inline Reduced reduce_large(double x)
{
    const std::uint64_t u = bits(x);
    const std::int32_t E = static_cast<std::int32_t>((u >> 52) & 0x7ff) - 1075;
    const std::uint64_t X = (u & ((std::uint64_t{1} << 52) - 1)) | (std::uint64_t{1} << 52);
    const std::int32_t i0 = E - 1 > 1 ? E - 1 : 1;
    // the 32 bits b_i .. b_{i+31} of 2/pi (i from 1)
    auto word = [](std::int32_t i) {
        const std::int32_t w = (i - 1) / 32, off = (i - 1) % 32;
        const std::uint32_t* T = coeffs::kTwoOverPiBits;
        return off == 0 ? T[w] : static_cast<std::uint32_t>((T[w] << off) | (T[w + 1] >> (32 - off)));
    };
    // W, least significant 32-bit word first; X in two words
    std::uint64_t w[6];
    for (int j = 0; j < 6; ++j) w[j] = word(i0 + 32 * (5 - j));
    const std::uint64_t xw[2] = {X & 0xffffffffu, X >> 32};
    std::uint64_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        std::uint64_t carry = 0;
        for (int j = 0; j < 6; ++j) {
            const std::uint64_t t = xw[i] * w[j] + z[i + j] + carry;   // < 2^64
            z[i + j] = t & 0xffffffffu;
            carry = t >> 32;
        }
        z[i + 6] += carry;
    }
    auto bit = [&](std::int32_t b) -> std::uint32_t {
        return b < 0 ? 0u : static_cast<std::uint32_t>((z[b / 32] >> (b % 32)) & 1u);
    };
    const std::int32_t sh = i0 + 191 - E;
    std::uint32_t q = bit(sh) | (bit(sh + 1) << 1);
    // the fraction's 192 bits, the first (weight 1/2) at f[0]
    std::uint32_t f[192];
    for (std::int32_t t = 0; t < 192; ++t) f[t] = bit(sh - 1 - t);
    const bool up = f[0] != 0;
    if (up) {                                       // 1 - f: two's complement
        q = (q + 1) & 3u;
        std::uint32_t borrow = 0;
        for (std::int32_t t = 191; t >= 0; --t) {
            const std::uint32_t d = 0u - f[t] - borrow;
            borrow = (f[t] | borrow) != 0 ? 1u : 0u;
            f[t] = d & 1u;
        }
    }
    std::int32_t lead = 0;
    while (lead < 192 && f[lead] == 0) ++lead;
    // three 53-bit pieces from the leading bit on, bit t weighing 2^-(t + 1)
    auto piece = [&](std::int32_t from) {
        std::uint64_t v = 0;
        for (std::int32_t t = from; t < from + 53; ++t) v = (v << 1) | (t < 192 ? f[t] : 0u);
        return static_cast<double>(v) * pow2i(-(from + 53));
    };
    DW g = fast_two_sum(piece(lead), piece(lead + 53));
    g = dw_plus_fp(g, piece(lead + 106));
    DW r = dw_times_fp(g, coeffs::kPio2Hi);
    r = dw_plus_fp(r, g.h * coeffs::kPio2Lo);
    if (up) r = {-r.h, -r.l};
    return {r, q};
}

// x = k pi/2 + r of any finite x >= 0: Cody and Waite's up to 2^20 pi/2,
// Payne and Hanek's above.
MOOTATION_FORCEINLINE Reduced reduce(double x)
{
    return x <= kTrigMax ? reduce_pio2(x) : reduce_large(x);
}

// sin(r_h + r_l) = sin r_h + r_l cos r_h, cos r_h ~ 1 - r_h^2/2 (the rest of
// r_l cos r_h is below 2^-110 r_h).
inline DW sin_kernel_dw(DW r)
{
    const double* S = coeffs::kSin;
    const double z = r.h * r.h;
    const double s = S[0] + z * (S[1] + z * (S[2] + z * (S[3] + z * (S[4] + z * S[5]))));
    const double y = (r.h * z) * s;
    return fast_two_sum(r.h, y + r.l * (1.0 - 0.5 * z));
}

inline double sin_kernel(DW r)
{
    return sin_kernel_dw(r).h;
}

// cos(r_h + r_l) = cos r_h - r_l sin r_h; 1 - r_h^2/2 with its rounding error
// kept (Fast2Sum of 1 and -r_h^2/2, |r_h^2/2| < 1).
inline DW cos_kernel_dw(DW r)
{
    const double* C = coeffs::kCos;
    const double z = r.h * r.h;
    const double c = C[0] + z * (C[1] + z * (C[2] + z * (C[3] + z * (C[4] + z * C[5]))));
    const double hz = 0.5 * z;
    const double w = 1.0 - hz;
    const double tail = ((1.0 - w) - hz) + ((z * z) * c - r.h * r.l);
    return fast_two_sum(w, tail);
}

inline double cos_kernel(DW r)
{
    return cos_kernel_dw(r).h;
}

}   // namespace detail

// Whether |x| is within the area of Cody and Waite's reduction (above it,
// Payne and Hanek's: the same accuracy).
inline bool trig_in_domain(double x)
{
    return std::fabs(x) <= detail::kTrigMax;
}

inline double sin(double x)
{
    const double ax = std::fabs(x);
    if (!(ax < std::numeric_limits<double>::infinity()))
        return x - x;                                  // NaN for NaN and +-inf
    if (ax < 0x1p-26)
        return x;                                      // sin x = x to the last bit
    const detail::Reduced R = detail::reduce(ax);
    double v;
    switch (R.q) {
    case 0: v = detail::sin_kernel(R.r); break;
    case 1: v = detail::cos_kernel(R.r); break;
    case 2: v = -detail::sin_kernel(R.r); break;
    default: v = -detail::cos_kernel(R.r); break;
    }
    return x < 0.0 ? -v : v;
}

inline double cos(double x)
{
    const double ax = std::fabs(x);
    if (!(ax < std::numeric_limits<double>::infinity()))
        return x - x;
    if (ax < 0x1p-27)
        return 1.0;                                    // cos x = 1 to the last bit
    const detail::Reduced R = detail::reduce(ax);
    switch (R.q) {
    case 0: return detail::cos_kernel(R.r);
    case 1: return -detail::sin_kernel(R.r);
    case 2: return -detail::cos_kernel(R.r);
    default: return detail::sin_kernel(R.r);
    }
}

// tan = sin / cos of the reduced argument (-cos / sin in odd quarters), the
// two as double-word numbers and their quotient by DWDivDW2.
inline double tan(double x)
{
    const double ax = std::fabs(x);
    if (!(ax < std::numeric_limits<double>::infinity()))
        return x - x;
    if (ax < 0x1p-27)
        return x;                                      // tan x = x to the last bit
    const detail::Reduced R = detail::reduce(ax);
    const detail::DW s = detail::sin_kernel_dw(R.r);
    const detail::DW c = detail::cos_kernel_dw(R.r);
    const double v = (R.q & 1u) ? -detail::dw_div_dw(c, s).h : detail::dw_div_dw(s, c).h;
    return x < 0.0 ? -v : v;
}

// ── acos ────────────────────────────────────────────────────────────────────

namespace detail {

// asin s - s = s^3 A(s^2), 0 <= s <= 1/2; returns t A(t) for t = s^2.
MOOTATION_FORCEINLINE double asin_tail(double t)
{
    const double* A = coeffs::kAsin;
    double p = A[12];
    for (int i = 11; i >= 0; --i)
        p = A[i] + t * p;
    return t * p;
}

}   // namespace detail

// acos x (Cody & Waite 1980, ch. 10): |x| <= 1/2: pi/2 - asin x; x > 1/2:
// 2 asin s; x < -1/2: pi - 2 asin s; s = sqrt((1 - |x|)/2), with the
// rounding error of the square root carried (Dekker's exact product).
inline double acos(double x)
{
    if (x != x)
        return x;                                      // NaN
    const double ax = std::fabs(x);
    if (ax > 1.0)
        return std::numeric_limits<double>::quiet_NaN();
    if (ax <= 0.5) {
        const double q = x * detail::asin_tail(x * x);  // asin x = x + q
        const detail::DW a = detail::two_sum(coeffs::kPio2Hi, -x);
        return a.h + (a.l + (coeffs::kPio2Lo - q));
    }
    const double z = (1.0 - ax) * 0.5;                  // exact
    if (z == 0.0)
        return x > 0.0 ? 0.0 : coeffs::kPiHi;           // acos(1) = +0, acos(-1) = pi
    const double s = std::sqrt(z);
    const detail::DW ss = detail::two_prod(s, s);
    const double sl = ((z - ss.h) - ss.l) / (2.0 * s);  // sqrt(z) ~ s + sl
    const double tail = sl + s * detail::asin_tail(z);  // asin sqrt(z) ~ s + tail
    if (x > 0.0)
        return 2.0 * (s + tail);
    const detail::DW a = detail::two_sum(coeffs::kPiHi, -2.0 * s);
    return a.h + (a.l + (coeffs::kPiLo - 2.0 * tail));
}

// ── asin, atan ──────────────────────────────────────────────────────────────

// asin x (Cody & Waite 1980, ch. 10), acos's kernel: |x| <= 1/2: x + x^3
// A(x^2); |x| > 1/2: pi/2 - 2 asin s, s = sqrt((1 - |x|)/2), the rounding
// error of the square root carried (Dekker's exact product). Odd, exactly.
inline double asin(double x)
{
    if (x != x)
        return x;                                      // NaN
    const double ax = std::fabs(x);
    if (ax > 1.0)
        return std::numeric_limits<double>::quiet_NaN();
    if (ax <= 0.5)
        return x + x * detail::asin_tail(x * x);
    const double z = (1.0 - ax) * 0.5;                  // exact
    if (z == 0.0)
        return x > 0.0 ? coeffs::kPio2Hi : -coeffs::kPio2Hi;
    const double s = std::sqrt(z);
    const detail::DW ss = detail::two_prod(s, s);
    const double sl = ((z - ss.h) - ss.l) / (2.0 * s);  // sqrt(z) ~ s + sl
    const double tail = sl + s * detail::asin_tail(z);  // asin sqrt(z) ~ s + tail
    const detail::DW a = detail::two_sum(coeffs::kPio2Hi, -2.0 * s);
    const double r = a.h + (a.l + (coeffs::kPio2Lo - 2.0 * tail));
    return x > 0.0 ? r : -r;
}

namespace detail {

// atan f - f = f^3 T(f^2), |f| <= 2 - sqrt 3; returns t T(t) for t = f^2.
MOOTATION_FORCEINLINE double atan_tail(double t)
{
    constexpr int n = static_cast<int>(sizeof(coeffs::kAtan) / sizeof(coeffs::kAtan[0]));
    double p = coeffs::kAtan[n - 1];
    for (int i = n - 2; i >= 0; --i)
        p = coeffs::kAtan[i] + t * p;
    return t * p;
}

}   // namespace detail

// atan x (Cody & Waite 1980, ch. 11): f = |x|, 1/f above 1 (atan = pi/2 -
// atan f), then (sqrt3 f - 1) / (sqrt3 + f) above 2 - sqrt 3 (atan = pi/6 +
// atan f), so that |f| <= 2 - sqrt 3; atan f = f + f^3 T(f^2); pi/6, pi/2,
// pi/3 added as hi + lo. Their reduction in plain floating point loses up to
// two ulps near its seams (sqrt3 f - 1 cancels); here f is carried as a
// double-word number through it (Joldes et al. 2017, Algorithms 4, 6, 7, 17).
// Odd, exactly; atan(+-inf) = +-pi/2.
inline double atan(double x)
{
    if (x != x)
        return x;                                      // NaN
    double f = std::fabs(x);
    if (f == std::numeric_limits<double>::infinity())
        return std::signbit(x) ? -coeffs::kPio2Hi : coeffs::kPio2Hi;
    double fl = 0.0;                                    // f + fl
    int n = 0;
    if (f > 1.0) {
        const double r = 1.0 / f;
        if (f < 0x1p500) {                              // Dekker's split of f stays finite
            const detail::DW p = detail::two_prod(f, r);
            fl = ((1.0 - p.h) - p.l) / f;               // 1/f ~ r + fl
        }
        f = r;
        n = 2;
    }
    if (f > coeffs::kTwoMinusSqrt3) {
        const detail::DW s3{coeffs::kSqrt3Hi, coeffs::kSqrt3Lo};
        detail::DW num = detail::dw_times_fp(s3, f);
        num = detail::dw_plus_fp(detail::DW{num.h, num.l + coeffs::kSqrt3Hi * fl}, -1.0);
        const detail::DW den = detail::dw_plus_dw(s3, detail::DW{f, fl});
        const detail::DW q = detail::dw_div_dw(num, den);
        f = q.h;
        fl = q.l;
        ++n;
    }
    const double rl = fl + f * detail::atan_tail(f * f);   // atan(f + fl) ~ f + rl
    double v = f + rl;
    if (n == 1) {
        const detail::DW a = detail::two_sum(coeffs::kPio6Hi, f);
        v = a.h + (a.l + (coeffs::kPio6Lo + rl));
    } else if (n == 2) {
        const detail::DW a = detail::two_sum(coeffs::kPio2Hi, -f);
        v = a.h + (a.l + (coeffs::kPio2Lo - rl));
    } else if (n == 3) {
        const detail::DW a = detail::two_sum(coeffs::kPio3Hi, -f);
        v = a.h + (a.l + (coeffs::kPio3Lo - rl));
    }
    return std::signbit(x) ? -v : v;
}

// ── hypot ───────────────────────────────────────────────────────────────────

// sqrt(x^2 + y^2) without overflow or underflow in between (Borges 2020): the
// larger scaled into [1, 2) by a power of two, the squares and their sum as
// double-word numbers, h = sqrt of the sum corrected by tau / (2h), tau =
// x^2 + y^2 - h^2 from the exact parts (his Algorithm 5, with Dekker's products
// where he takes FMAs). ay < ax 2^-27: ax itself (his section 3).
inline double hypot(double x, double y)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    double ax = std::fabs(x), ay = std::fabs(y);
    if (ax == inf || ay == inf)
        return inf;                                     // even with a NaN (F.10.4.4)
    if (ax != ax || ay != ay)
        return x + y;                                   // NaN
    if (ax < ay) {
        const double t = ax;
        ax = ay;
        ay = t;
    }
    if (ay == 0.0 || ay < ax * 0x1p-27)
        return ax;                                      // ax (1 + (ay/ax)^2/2) rounds to ax
    double post = 1.0;
    if (ax > 0x1p500) {
        ax *= 0x1p-600;
        ay *= 0x1p-600;
        post = 0x1p600;
    } else if (ax < 0x1p-500) {
        ax *= 0x1p600;
        ay *= 0x1p600;
        post = 0x1p-600;
    }
    const std::int32_t e = detail::exponent_of(ax);
    const double down = detail::pow2i(-e);
    ax *= down;                                         // [1, 2), exact
    ay *= down;                                         // >= 2^-27, exact
    const detail::DW S = detail::dw_plus_dw(detail::two_prod(ax, ax), detail::two_prod(ay, ay));
    const double h = std::sqrt(S.h);
    const detail::DW hh = detail::two_prod(h, h);
    const double corr = (((S.h - hh.h) - hh.l) + S.l) / (2.0 * h);
    return ((h + corr) * detail::pow2i(e)) * post;
}

// ── exp ─────────────────────────────────────────────────────────────────────

namespace detail {

// e^(x_h + x_l) for kExpUnder <= x_h <= kExpOver, |x_l| <= ulp(x_h): Tang,
// ACM TOMS 15(2), 1989, steps 2-4 (x_l = 0 for exp; pow passes the low part
// of y log x). x = (32 M + J) ln2/32 + R1 + R2, |R1 + R2| <= ln2/64;
// e^x = 2^M (S_lead(J) + (S_trail(J) + S (R1 + (R2 + Q)))).
MOOTATION_FORCEINLINE double exp_dw(double xh, double xl)
{
    const double N = round_to_integer(xh * coeffs::kExpInvL);
    const std::int32_t n = static_cast<std::int32_t>(N);
    const std::int32_t j = n & 31;                      // N mod 32, >= 0
    const std::int32_t m = (n - j) / 32;
    const double R1 = xh - N * coeffs::kExpL1;          // exact (N kExpL1 exact, cancellation)
    const double R2 = xl - N * coeffs::kExpL2;
    const double R = R1 + R2;
    const double* A = coeffs::kExpA;
    const double Q = (R * R) * (A[0] + R * (A[1] + R * (A[2] + R * (A[3] + R * A[4]))));
    const double P = R1 + (R2 + Q);
    const double Sl = coeffs::kExpS[j][0];
    const double St = coeffs::kExpS[j][1];
    const double S = Sl + St;
    const double y = Sl + (St + S * P);
    if (m >= 1024)
        return (y * 2.0) * pow2i(m - 1);
    if (m <= -1022)                                     // a subnormal result: one rounding
        return (y * pow2i(m + 600)) * 0x1p-600;
    return y * pow2i(m);
}

}   // namespace detail

inline double exp(double x)
{
    if (x != x)
        return x;
    if (x > coeffs::kExpOver)
        return std::numeric_limits<double>::infinity();
    if (x < coeffs::kExpUnder)
        return 0.0;
    if (std::fabs(x) < 0x1p-54)
        return 1.0 + x;                                 // rounds to 1 (Tang, step 1)
    return detail::exp_dw(x, 0.0);
}

// ── log ─────────────────────────────────────────────────────────────────────

namespace detail {

// v with the lowest 29 bits of its significand cleared: 24 significant
// bits, truncated (Tang 1990, Procedure 2, step 3).
inline double truncate24(double v)
{
    return from_bits(bits(v) & ~((std::uint64_t{1} << 29) - 1));
}

// log(1 + f) for e^-1/16 < 1 + f < e^1/16: Tang 1990, Procedure 2.
inline double log_near1(double f)
{
    const double g = 1.0 / (2.0 + f);
    const double u = (2.0 * f) * g;
    const double v = u * u;
    const double* B = coeffs::kLogB;
    const double q = (u * v) * (B[0] + v * (B[1] + v * (B[2] + v * B[3])));
    const double u1 = truncate24(u);
    const double f1 = truncate24(f);
    const double f2 = f - f1;
    const double u2 = ((2.0 * (f - u1) - u1 * f1) - u1 * f2) * g;
    return u1 + (u2 + q);
}

// x = 2^m Y, 1 <= Y < 2; F = 1 + j/128 nearest Y, f = Y - F (exact), for a
// positive normal or subnormal x (Tang 1990, section 3.2, step 3).
struct LogSplit {
    std::int32_t m;
    std::int32_t j;
    double F, f;
};

MOOTATION_FORCEINLINE LogSplit log_split(double x)
{
    std::int32_t m = 0;
    if (x < 0x1p-1022) {
        x *= 0x1p54;                                    // exact
        m = -54;
    }
    m += exponent_of(x);
    // Y: x's significand with the exponent of 1 (exact; also for x >= 2^1023)
    const double Y = from_bits((bits(x) & ((std::uint64_t{1} << 52) - 1)) | (std::uint64_t{1023} << 52));
    const double J = round_to_integer(Y * 128.0);         // 128..256
    const double F = J * 0x1p-7;
    return {m, static_cast<std::int32_t>(J) - 128, F, Y - F};
}

}   // namespace detail

// log x (Tang, ACM TOMS 16(4), 1990): near 1 Procedure 2; elsewhere
// Procedure 1, log x = m ln2 + log F + log(1 + f/F), the leads of m ln2 and
// log F adding up exactly.
inline double log(double x)
{
    if (x != x)
        return x;
    if (x < 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0)
        return -std::numeric_limits<double>::infinity();
    if (x == std::numeric_limits<double>::infinity())
        return x;
    if (x > coeffs::kLogT1 && x < coeffs::kLogT2)
        return detail::log_near1(x - 1.0);              // x - 1 exact
    const detail::LogSplit s = detail::log_split(x);
    const double md = static_cast<double>(s.m);
    const double Llead = md * coeffs::kLog2Lead + coeffs::kLogC[s.j][0];    // exact
    const double Ltrail = md * coeffs::kLog2Trail + coeffs::kLogC[s.j][1];
    const double u = s.f / s.F;
    const double* A = coeffs::kLogA;
    const double q = (u * u) * (A[0] + u * (A[1] + u * (A[2] + u * (A[3] + u * A[4]))));
    return Llead + (u + (q + Ltrail));
}

// ── pow ─────────────────────────────────────────────────────────────────────

namespace detail {

// log x as a double-word number, x > 0 finite: Tang's two procedures with
// the reduced argument and the leading terms in double-word arithmetic
// (Joldes et al. 2017), for pow.
MOOTATION_FORCEINLINE DW log_dw(double x)
{
    if (x > coeffs::kLogT1 && x < coeffs::kLogT2) {
        // log(1 + f) = a + a^3 B(a^2), a = 2f / (2 + f) as a double-word
        const double f = x - 1.0;                       // exact
        const DW a = dw_div_dw({2.0 * f, 0.0}, two_sum(2.0, f));
        const double v = a.h * a.h;
        const double* B = coeffs::kLogB;
        const double q = (a.h * v) * (B[0] + v * (B[1] + v * (B[2] + v * B[3])));
        return fast_two_sum(a.h, a.l + q);
    }
    const LogSplit s = log_split(x);
    const double md = static_cast<double>(s.m);
    const double Llead = md * coeffs::kLog2Lead + coeffs::kLogC[s.j][0];    // exact
    const double Ltrail = md * coeffs::kLog2Trail + coeffs::kLogC[s.j][1];
    // u = f / F as a double-word number
    const double uh = s.f / s.F;
    const DW r = two_prod(uh, s.F);
    const double ul = ((s.f - r.h) - r.l) / s.F;
    // log(1 + u) = u - u^2/2 + u^3 D(u)
    const DW p = two_prod(uh, uh);
    const double* D = coeffs::kLogD;
    const double cube = (uh * p.h) * (D[0] + uh * (D[1] + uh * (D[2] + uh * (D[3] + uh * (D[4] + uh * D[5])))));
    const double rest = ((ul - uh * ul) - 0.5 * p.l) + cube;
    const DW t = two_sum(uh, -0.5 * p.h);
    const DW head = two_sum(Llead, t.h);
    return dw_plus_fp(head, t.l + (rest + Ltrail));
}

MOOTATION_FORCEINLINE bool is_integer(double y)
{
    return std::floor(y) == y;
}

// An odd integer: |y| < 2^53 (above it every double is even).
MOOTATION_FORCEINLINE bool is_odd_integer(double y)
{
    return is_integer(y) && std::fabs(y) < 0x1p53 && !is_integer(0.5 * y);
}

}   // namespace detail

// x^y: the special cases of C23 Annex F (F.10.4.5); y = 2, -1, 1/2 as x x,
// 1/x, sqrt x (each correctly rounded, so what any correctly rounding pow
// gives); otherwise e^(y log |x|), log |x| and its product with y as
// double-word numbers, Tang's exp of the double-word result.
inline double pow(double x, double y)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    if (y == 0.0)
        return 1.0;                                     // even for a NaN x
    if (x == 1.0)
        return 1.0;                                     // even for a NaN y
    if (x != x || y != y)
        return x + y;                                   // NaN
    if (y == 2.0)
        return x * x;
    if (y == -1.0)
        return 1.0 / x;
    if (y == 0.5 && x != 0.0 && x != -inf)
        return std::sqrt(x);                            // NaN for x < 0
    const double ax = std::fabs(x);
    const bool odd = detail::is_odd_integer(y);
    if (std::fabs(y) == inf) {
        if (ax == 1.0)
            return 1.0;                                 // x = -1
        return (ax < 1.0) == (y < 0.0) ? inf : 0.0;
    }
    if (x == 0.0) {
        if (y < 0.0)
            return odd ? 1.0 / x : inf;                 // +-inf for an odd y
        return odd ? x : 0.0;
    }
    if (ax == inf) {
        if (x > 0.0)
            return y < 0.0 ? 0.0 : inf;
        if (y < 0.0)
            return odd ? -0.0 : 0.0;
        return odd ? -inf : inf;
    }
    if (x < 0.0 && !detail::is_integer(y))
        return std::numeric_limits<double>::quiet_NaN();
    const bool negative = x < 0.0 && odd;
    double r;
    if (std::fabs(y) > 0x1p64) {
        // |log |x|| >= 2^-54 for |x| != 1, so |y log |x|| > 2^10: out of range
        r = (ax > 1.0) == (y > 0.0) ? inf : 0.0;
    } else {
        const detail::DW z = detail::dw_times_fp(detail::log_dw(ax), y);
        if (z.h > coeffs::kExpOver)
            r = inf;
        else if (z.h < coeffs::kExpUnder)
            r = 0.0;
        else if (std::fabs(z.h) < 0x1p-54)
            r = 1.0 + z.h;
        else
            r = detail::exp_dw(z.h, z.l);
    }
    return negative ? -r : r;
}

}   // namespace mootation::dm

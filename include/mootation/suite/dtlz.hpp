#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DTLZ1-7 (Deb, Thiele, Laumanns & Zitzler 2002) as benchmarks/dtlz.py
// evaluates them, the variants of dtlz_variants.py (IDTLZ1/2, minusDTLZ2,
// SDTLZ1/2) and the constraints of C-DTLZ (c_dtlz.py, Jain & Deb 2014, read as
// there). Positions x_1..x_{M-1}, the distance block from x_M on; the order of
// every operation is that of the NumPy rows (common.hpp).
// ============================================================================

#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::dtlz {

// wideDTLZ3's distance variables in [0, 1.2] (dtlz_variants.WIDE_UPPER)
constexpr double kWideUpper = 1.2;

// g_1 = 100 (k + sum((x - 1/2)^2 - cos(20 pi (x - 1/2)))) (DTLZ1, DTLZ3)
inline double g1(const double* xm, int k)
{
    std::vector<double> t(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        const double d = xm[j] - 0.5;
        t[static_cast<std::size_t>(j)] = d * d - dm::cos(20.0 * kPi * d);
    }
    return 100.0 * (k + row_sum(t));
}

// g_2 = sum (x - 1/2)^2 (DTLZ2, 4, 5)
inline double g2(const double* xm, int k)
{
    std::vector<double> t(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        const double d = xm[j] - 0.5;
        t[static_cast<std::size_t>(j)] = d * d;
    }
    return row_sum(t);
}

// g_6 = sum x^0.1 (DTLZ6)
inline double g6(const double* xm, int k)
{
    std::vector<double> t(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) t[static_cast<std::size_t>(j)] = dm::pow(xm[j], 0.1);
    return row_sum(t);
}

// f_1 = base prod(c_1..c_{M-1}), f_i = base prod(c_1..c_{M-i}) last_{M-i+1}
// (i = 2..M-1), f_M = base last_1 (dtlz._products)
inline void products(const double* c, const double* last, double base, int M, double* f)
{
    f[0] = base * row_prod(c, static_cast<std::size_t>(M - 1));
    for (int i = 2; i < M; ++i)
        f[i - 1] = base * row_prod(c, static_cast<std::size_t>(M - i)) * last[M - i];
    f[M - 1] = base * last[0];
}

// The concave shape of DTLZ2-4 (alpha = 100 for DTLZ4), base 1 + g.
inline void concave(const double* y, double g, int M, double alpha, double* f)
{
    std::vector<double> c(static_cast<std::size_t>(M - 1)), s(static_cast<std::size_t>(M - 1));
    for (int i = 0; i < M - 1; ++i) {
        const double xp = alpha != 1.0 ? dm::pow(y[i], alpha) : y[i];
        c[static_cast<std::size_t>(i)] = dm::cos(xp * kPi / 2.0);
        s[static_cast<std::size_t>(i)] = dm::sin(xp * kPi / 2.0);
    }
    products(c.data(), s.data(), 1.0 + g, M, f);
}

// The degenerate shape of DTLZ5 and DTLZ6 (dtlz._dtlz_degenerate).
inline void degenerate(const double* y, double g, int M, double* f)
{
    std::vector<double> c(static_cast<std::size_t>(M - 1)), s(static_cast<std::size_t>(M - 1));
    std::vector<double> theta(static_cast<std::size_t>(M - 1));
    theta[0] = y[0] * kPi / 2.0;
    if (M > 2) {
        const double denom = 4.0 * (1.0 + g);
        for (int i = 1; i < M - 1; ++i)
            theta[static_cast<std::size_t>(i)] = (kPi / denom) * (1.0 + 2.0 * g * y[i]);
    }
    for (int i = 0; i < M - 1; ++i) {
        c[static_cast<std::size_t>(i)] = dm::cos(theta[static_cast<std::size_t>(i)]);
        s[static_cast<std::size_t>(i)] = dm::sin(theta[static_cast<std::size_t>(i)]);
    }
    products(c.data(), s.data(), 1.0 + g, M, f);
}

// DTLZ`which` (1..7) of y, n variables, M objectives.
inline void evaluate(int which, const double* y, int n, int M, double* f)
{
    const double* xm = y + (M - 1);
    const int k = n - M + 1;
    switch (which) {
        case 1: {
            const double base = 0.5 * (1.0 + g1(xm, k));
            std::vector<double> last(static_cast<std::size_t>(M - 1));
            for (int i = 0; i < M - 1; ++i) last[static_cast<std::size_t>(i)] = 1.0 - y[i];
            products(y, last.data(), base, M, f);
            return;
        }
        case 2: concave(y, g2(xm, k), M, 1.0, f); return;
        case 3: concave(y, g1(xm, k), M, 1.0, f); return;
        case 4: concave(y, g2(xm, k), M, 100.0, f); return;
        case 5: degenerate(y, g2(xm, k), M, f); return;
        case 6: degenerate(y, g6(xm, k), M, f); return;
        case 7: {
            const double g = 1.0 + 9.0 / k * row_sum(xm, static_cast<std::size_t>(k));
            std::vector<double> t(static_cast<std::size_t>(M - 1));
            for (int i = 0; i < M - 1; ++i) {
                f[i] = y[i];
                t[static_cast<std::size_t>(i)] = (y[i] / (1.0 + g)) * (1.0 + dm::sin(3.0 * kPi * y[i]));
            }
            const double h = static_cast<double>(M) - row_sum(t);
            f[M - 1] = (1.0 + g) * h;
            return;
        }
        default:
            return;
    }
}

// ---- the variants (dtlz_variants.py) ------------------------------------
// IDTLZ1: 0.5 (1 + g) - f of DTLZ1; IDTLZ2: (1 + g) - f of DTLZ2.
inline void inverted(int which, const double* y, int n, int M, double* f)
{
    const double* xm = y + (M - 1);
    const int k = n - M + 1;
    evaluate(which, y, n, M, f);
    const double top = which == 1 ? 0.5 * (1.0 + g1(xm, k)) : 1.0 + g2(xm, k);
    for (int i = 0; i < M; ++i) f[i] = top - f[i];
}

// ---- C-DTLZ's constraints (c_dtlz.constraints_rows), <= 0 satisfied ------
enum class Cons { C1_DTLZ1, C1_DTLZ3, C2_DTLZ2, C2_Convex, C3_DTLZ1, C3_DTLZ4 };

// The radius r of C1-DTLZ3, C2-DTLZ2 and C2-ConvexDTLZ2 at M = 3 and 5 (Jain
// & Deb, IEEE TEVC 18(4), 2014, Part II; c_dtlz.C1_DTLZ3_R, C2_DTLZ2_R,
// C2_CONVEX_R); 0 elsewhere.
inline double cdtlz_radius(Cons which, int M)
{
    if (M != 3 && M != 5) return 0.0;
    switch (which) {
        case Cons::C1_DTLZ3: return M == 3 ? 9.0 : 12.5;
        case Cons::C2_DTLZ2: return M == 3 ? 0.4 : 0.5;
        case Cons::C2_Convex: return 0.225;
        default: return 0.0;
    }
}

// Part I's convex DTLZ2: f_i^4 (i < M), f_M^2 (c_dtlz.convex).
inline void convex(double* f, int M)
{
    for (int i = 0; i < M - 1; ++i) f[i] = dm::pow(f[i], 4.0);
    f[M - 1] = f[M - 1] * f[M - 1];
}

// r2: r^2 of the problem at M (C1-DTLZ3, C2-DTLZ2, C2-ConvexDTLZ2); centre:
// 1/sqrt(M), C2-DTLZ2's middle sphere.
inline void constraints(Cons which, const double* f, int M, double r2, double centre, double* g)
{
    std::vector<double> sq(static_cast<std::size_t>(M));
    for (int i = 0; i < M; ++i) sq[static_cast<std::size_t>(i)] = f[i] * f[i];
    const double s2 = row_sum(sq);
    switch (which) {
        case Cons::C1_DTLZ1:
            g[0] = -(1.0 - f[M - 1] / 0.6 - row_sum(f, static_cast<std::size_t>(M - 1)) / 0.5);
            return;
        case Cons::C1_DTLZ3:
            g[0] = -((s2 - 16.0) * (s2 - r2));
            return;
        case Cons::C2_DTLZ2: {
            double best = 0.0;
            for (int i = 0; i < M; ++i) {
                const double corner = dm::pow(f[i] - 1.0, 2.0) + (s2 - dm::pow(f[i], 2.0)) - r2;
                best = i == 0 ? corner : nan_min(best, corner);
            }
            std::vector<double> d(static_cast<std::size_t>(M));
            for (int i = 0; i < M; ++i) {
                const double e = f[i] - centre;
                d[static_cast<std::size_t>(i)] = e * e;
            }
            g[0] = nan_min(best, row_sum(d) - r2);
            return;
        }
        case Cons::C2_Convex: {
            const double mean = row_sum(f, static_cast<std::size_t>(M)) / M;
            std::vector<double> d(static_cast<std::size_t>(M));
            for (int i = 0; i < M; ++i) {
                const double e = f[i] - mean;
                d[static_cast<std::size_t>(i)] = e * e;
            }
            g[0] = -(row_sum(d) - r2);
            return;
        }
        case Cons::C3_DTLZ1: {
            const double s = row_sum(f, static_cast<std::size_t>(M));
            for (int j = 0; j < M; ++j) g[j] = -(f[j] / 0.5 + (s - f[j]) - 1.0);
            return;
        }
        case Cons::C3_DTLZ4:
            for (int j = 0; j < M; ++j) {
                const double p = dm::pow(f[j], 2.0);
                g[j] = -(p / 4.0 + (s2 - p) - 1.0);
            }
            return;
    }
}

}   // namespace mootation::suite::dtlz

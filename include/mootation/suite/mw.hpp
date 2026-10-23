#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MW1-14 (Ma & Wang, IEEE TEVC 23(6), 2019, Eqs. 12-28), as benchmarks/mw.py
// evaluates them (task 17, supplement 3, stage 3a): g1_rows .. g3_rows,
// objectives_rows and constraints_f one for one, in the order of their NumPy
// rows (common.hpp); where the point formula raised a scalar (_rows.spow) a
// float's ** is dm::pow, x ** 2 too; np.arctan and np.arcsin dm::atan and
// dm::asin. mw.py stays the reference. The constraints in the library's sign
// (mw.py, THE SIGN); m objectives (MW4, MW8, MW14 three or more, the rest two).
// ============================================================================

#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::mw {

constexpr double kSqrt2 = 1.4142135623730951;      // np.sqrt(2.0)

inline bool scalable(int i) { return i == 4 || i == 8 || i == 14; }

inline int n_con(int i)
{
    constexpr int c[14] = {1, 1, 2, 1, 3, 1, 2, 1, 1, 3, 4, 2, 2, 1};
    return c[i - 1];
}

// the distance function of each problem: 1, 2 or 3 (mw.DISTANCE)
inline int distance(int i)
{
    constexpr int d[14] = {1, 2, 3, 1, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3};
    return d[i - 1];
}

// g1, g2, g3 (Eqs. 12-14) of the n variables x, m objectives
inline double g(int kind, const double* x, int n, int m)
{
    std::vector<double> t;
    for (int j = m - 1; j < n; ++j) {                 // i = j + 1 = m .. n
        if (kind == 1) {
            const double z = dm::pow(x[j], static_cast<double>(n - m));
            const double d = z - 0.5 - j / (2.0 * n);
            t.push_back(1.0 - dm::exp(-10.0 * (d * d)));
        } else if (kind == 2) {
            const double d = x[j] - static_cast<double>(j) / n;
            const double z = 1.0 - dm::exp(-10.0 * (d * d));
            t.push_back(1.5 + (0.1 / n) * z * z - 1.5 * dm::cos(2.0 * kPi * z));
        } else {
            const double a = x[j - 1] - 0.5;
            const double d = x[j] + a * a - 1.0;
            t.push_back(2.0 * (d * d));
        }
    }
    return 1.0 + row_sum(t);
}

// sqrt(max(0, v)), max as Python's (a NaN gives 0; mw.py MW-3)
inline double root(double v) { return std::sqrt(v > 0.0 ? v : 0.0); }

inline void objectives(int i, const double* x, int n, int m, double* f)
{
    const double gv = g(distance(i), x, n, m);
    const double x1 = x[0];
    if (i == 1) {
        f[0] = x1;
        f[1] = gv * (1.0 - 0.85 * x1 / gv);
        return;
    }
    if (i == 2 || i == 3) {
        f[0] = x1;
        f[1] = gv * (1.0 - x1 / gv);
        return;
    }
    if (i == 4 || i == 8) {
        // F = g; F[1:] *= x[m-2], .., x[0] (sin of (pi/2) x for MW8); F[:-1] *=
        // the running product of 1 - x (cos) reversed
        std::vector<double> cp(static_cast<std::size_t>(m - 1));
        double run = 1.0;
        for (int k = 0; k < m - 1; ++k) {
            const double v = i == 4 ? 1.0 - x[k] : dm::cos(0.5 * kPi * x[k]);
            run = k == 0 ? v : run * v;
            cp[static_cast<std::size_t>(k)] = run;
        }
        for (int j = 0; j < m; ++j) f[j] = gv;
        for (int j = 1; j < m; ++j) f[j] = f[j] * (i == 4 ? x[m - 1 - j] : dm::sin(0.5 * kPi * x[m - 1 - j]));
        for (int j = 0; j < m - 1; ++j) f[j] = f[j] * cp[static_cast<std::size_t>(m - 2 - j)];
        return;
    }
    if (i == 14) {
        std::vector<double> h(static_cast<std::size_t>(m - 1));
        for (int k = 0; k < m - 1; ++k) {
            f[k] = x[k];
            h[static_cast<std::size_t>(k)] = 6.0 - dm::exp(x[k]) - 1.5 * dm::sin(1.1 * kPi * (x[k] * x[k]));
        }
        f[m - 1] = gv / (m - 1) * row_sum(h);
        return;
    }
    const double f1 = i == 10 ? gv * dm::pow(x1, 15.0) : gv * x1;
    const double t = f1 / gv;
    double f2 = 0.0;
    switch (i) {
        case 5:
        case 7: f2 = gv * root(1.0 - dm::pow(t, 2.0)); break;
        case 6: f2 = gv * root(1.1 * 1.1 - dm::pow(t, 2.0)); break;
        case 9: f2 = gv * (1.0 - dm::pow(t, 0.6)); break;
        case 10: f2 = gv * (1.0 - dm::pow(t, 2.0)); break;
        case 11: f2 = gv * root(2.0 - dm::pow(t, 2.0)); break;
        case 12: f2 = gv * (0.85 - 0.8 * t - 0.08 * std::abs(dm::sin(3.2 * kPi * t))); break;
        default: f2 = gv * (5.0 - dm::exp(t) - 0.5 * std::abs(dm::sin(3.0 * kPi * t))); break;   // 13
    }
    f[0] = f1;
    f[1] = f2;
}

// np.arctan(f2 / f1), the division as NumPy's (0 / 0 NaN, x / 0 +-inf)
inline double angle(double f2, double f1) { return dm::atan(f2 / f1); }

inline void constraints(int i, const double* F, int m, double* c)
{
    const double f1 = F[0], f2 = F[1];
    std::vector<double> ff(static_cast<std::size_t>(m));
    for (int k = 0; k < m; ++k) ff[static_cast<std::size_t>(k)] = F[k] * F[k];
    const double sq = row_sum(ff);
    switch (i) {
        case 1:
        case 2: {
            const double l = kSqrt2 * f2 - kSqrt2 * f1;
            const double b = i == 1 ? 2.0 : 3.0;
            c[0] = -(1.0 - f1 - f2 + 0.5 * dm::pow(dm::sin(b * kPi * l), 8.0));
            return;
        }
        case 3: {
            const double l = kSqrt2 * f2 - kSqrt2 * f1;
            const double s = dm::sin(0.75 * kPi * l);
            c[0] = -(1.05 - f1 - f2 + 0.45 * dm::pow(s, 6.0));
            c[1] = 0.85 - f1 - f2 + 0.3 * (s * s);
            return;
        }
        case 4: {
            const double l = F[m - 1] - row_sum(F, static_cast<std::size_t>(m - 1));
            c[0] = -((1.0 + 0.4 * dm::pow(dm::sin(2.5 * kPi * l), 8.0)) - row_sum(F, static_cast<std::size_t>(m)));
            return;
        }
        case 5: {
            const double l1 = angle(f2, f1);
            const double l2 = 0.5 * kPi - 2.0 * std::abs(l1 - 0.25 * kPi);
            const double a = 1.7 - 0.2 * dm::sin(2.0 * l1);
            const double s = dm::sin(6.0 * dm::pow(l2, 3.0));
            const double u = 1.0 + 0.5 * s, v = 1.0 - 0.45 * s;
            c[0] = -(a * a - sq);
            c[1] = u * u - sq;
            c[2] = v * v - sq;
            return;
        }
        case 6: {
            const double l = dm::pow(dm::cos(6.0 * dm::pow(angle(f2, f1), 4.0)), 10.0);
            const double a = f1 / (1.0 + 0.15 * l), b = f2 / (1.0 + 0.75 * l);
            c[0] = -(1.0 - a * a - b * b);
            return;
        }
        case 7: {
            const double s = dm::sin(4.0 * angle(f2, f1));
            const double a = 1.2 + 0.4 * dm::pow(s, 16.0), b = 1.15 - 0.2 * dm::pow(s, 8.0);
            c[0] = -(a * a - sq);
            c[1] = b * b - sq;
            return;
        }
        case 8: {
            const double s = dm::sin(6.0 * dm::asin(F[m - 1] / std::sqrt(sq)));
            const double a = 1.25 - 0.5 * (s * s);
            c[0] = -(a * a - sq);
            return;
        }
        case 9: {
            const double t1 = (1.0 - 0.64 * f1 * f1 - f2) * (1.0 - 0.36 * f1 * f1 - f2);
            const double p = f1 + 0.35, q = f1 + 0.15;
            const double t2 = 1.8225000000000002 - p * p - f2;            // 1.35 ** 2
            const double t3 = 1.3224999999999998 - q * q - f2;            // 1.15 ** 2
            c[0] = nan_min(t1, t2 * t3);
            return;
        }
        case 10: {
            const double s = f1 * f1;
            c[0] = -((2.0 - 4.0 * s - f2) * (2.0 - 8.0 * s - f2));
            c[1] = (2.0 - 2.0 * s - f2) * (2.0 - 16.0 * s - f2);
            c[2] = (1.0 - s - f2) * (1.2 - 1.2 * s - f2);
            return;
        }
        case 11: {
            const double s = f1 * f1;
            c[0] = -((3.0 - s - f2) * (3.0 - 2.0 * s - f2));
            c[1] = (3.0 - 0.625 * s - f2) * (3.0 - 7.0 * s - f2);
            c[2] = -((1.62 - 0.18 * s - f2) * (1.125 - 0.125 * s - f2));
            c[3] = (2.07 - 0.23 * s - f2) * (0.63 - 0.07 * s - f2);
            return;
        }
        case 12: {
            const double t1 = 1.0 - 0.8 * f1 - f2 + 0.08 * dm::sin(2.0 * kPi * (f2 - f1 / 1.5));
            const double t2 = 1.0 - 0.625 * f1 - f2 + 0.08 * dm::sin(2.0 * kPi * (f2 - f1 / 1.6));
            const double t3 = 1.4 - 0.875 * f1 - f2 + 0.08 * dm::sin(2.0 * kPi * (f2 / 1.4 - f1 / 1.6));
            const double t4 = 1.8 - 1.125 * f1 - f2 + 0.08 * dm::sin(2.0 * kPi * (f2 / 1.8 - f1 / 1.6));
            c[0] = t1 * t4;
            c[1] = -(t2 * t3);
            return;
        }
        case 13: {
            const double w = 0.5 * dm::sin(3.0 * kPi * f1);
            const double t1 = 5.0 - dm::exp(f1) - w - f2;
            const double t2 = 5.0 - (1.0 + f1 + 0.5 * f1 * f1) - w - f2;
            const double t3 = 5.0 - (1.0 + 0.7 * f1) - w - f2;
            const double t4 = 5.0 - (1.0 + 0.4 * f1) - w - f2;
            c[0] = t1 * t4;
            c[1] = -(t2 * t3);
            return;
        }
        default: {                                     // 14
            std::vector<double> a(static_cast<std::size_t>(m - 1));
            for (int k = 0; k < m - 1; ++k) {
                const double P = F[k];
                a[static_cast<std::size_t>(k)] = 6.1 - (1.0 + P + 0.5 * P * P + 1.5 * dm::sin(1.1 * kPi * P * P));
            }
            c[0] = -(row_sum(a) / (m - 1) - F[m - 1]);
            return;
        }
    }
}

inline void evaluate(int i, const double* x, int n, int m, double* f, double* c)
{
    objectives(i, x, n, m, f);
    constraints(i, f, m, c);
}

}   // namespace mootation::suite::mw

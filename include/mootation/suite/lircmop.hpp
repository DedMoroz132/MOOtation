#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// LIR-CMOP1-14 (Fan et al., Soft Computing 23(23), 2019, Table 6 of
// arXiv:1707.08767v1), as benchmarks/lir_cmop.py evaluates them (task 17,
// supplement 3, stage 3a): g_values, objectives and constraints_c one for
// one, in the order of their NumPy rows (common.hpp). lir_cmop.py stays the
// reference. J1 the odd variables from the third, J2 the even ones (1-based);
// the constraints in the library's sign, -c.
// ============================================================================

#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::lircmop {

constexpr double kShift = 0.7057;
constexpr double kA = 0.51, kB = 0.5, kC = 20.0, kR = 0.1;
constexpr double kCosT = 0.7071067811865476;      // math.cos(-0.25 pi)
constexpr double kSinT = -0.7071067811865476;     // math.sin(-0.25 pi)
constexpr double kCosA = 0.7071067811865476;      // math.cos(0.25 pi)
constexpr double kSinA = 0.7071067811865476;      // math.sin(0.25 pi)

inline int n_obj(int i) { return i >= 13 ? 3 : 2; }

inline int n_con(int i)
{
    if (i <= 2) return 2;
    if (i <= 4) return 3;
    if (i <= 6) return 2;
    if (i <= 8) return 3;
    if (i <= 12) return 2;
    return i == 13 ? 2 : 3;
}

// (g1, g2) of the n variables x (lir_cmop.g_values)
inline void g_values(int i, const double* x, int n, double& g1, double& g2)
{
    std::vector<double> t1, t2;
    const double u = x[0];
    for (int j = 2; j < n; j += 2) {                   // J1: 0-based 2, 4, ...
        const double s = i <= 4 ? dm::sin(0.5 * kPi * u) : dm::sin(0.5 * (j + 1) / 30.0 * kPi * u);
        t1.push_back((x[j] - s) * (x[j] - s));
    }
    for (int j = 1; j < n; j += 2) {                   // J2: 0-based 1, 3, ...
        const double c = i <= 4 ? dm::cos(0.5 * kPi * u) : dm::cos(0.5 * (j + 1) / 30.0 * kPi * u);
        t2.push_back((x[j] - c) * (x[j] - c));
    }
    g1 = row_sum(t1);
    g2 = row_sum(t2);
}

inline bool sqrt_shape(int i) { return i == 2 || i == 4 || i == 5 || i == 7 || i == 10 || i == 11; }

// one ellipse of 5-12: ((f1 - p) cos t - (f2 - q) sin t)^2 / a^2 + (...)^2 / b^2 - r
inline double ellipse(double F1, double F2, double p, double q, double a, double b)
{
    const double d1 = F1 - p, d2 = F2 - q;
    const double u = d1 * kCosT - d2 * kSinT, v = d1 * kSinT + d2 * kCosT;
    return u * u / (a * a) + v * v / (b * b) - kR;
}

// f and the constraints (<= 0 satisfied) of the point x of n variables
inline void evaluate(int i, const double* x, int n, double* f, double* out)
{
    if (i >= 13) {
        std::vector<double> t;
        for (int j = 2; j < n; ++j) t.push_back(10.0 * ((x[j] - 0.5) * (x[j] - 0.5)));
        const double g1 = row_sum(t);
        const double c1 = dm::cos(0.5 * kPi * x[0]), s1 = dm::sin(0.5 * kPi * x[0]);
        const double rad = kShift + 1.0 + g1;
        f[0] = rad * c1 * dm::cos(0.5 * kPi * x[1]);
        f[1] = rad * c1 * dm::sin(0.5 * kPi * x[1]);
        f[2] = rad * s1;
        const double sq[3] = {f[0] * f[0], f[1] * f[1], f[2] * f[2]};
        const double gg = row_sum(sq, 3);
        out[0] = -((gg - 9.0) * (gg - 4.0));
        out[1] = -((gg - 3.61) * (gg - 3.24));
        if (i == 14) out[2] = -((gg - 3.0625) * (gg - 2.56));
        return;
    }
    double g1 = 0.0, g2 = 0.0;
    g_values(i, x, n, g1, g2);
    const double h = sqrt_shape(i) ? 1.0 - std::sqrt(x[0]) : 1.0 - x[0] * x[0];
    if (i <= 4) {
        f[0] = x[0] + g1;
        f[1] = h + g2;
        out[0] = -((kA - g1) * (g1 - kB));
        out[1] = -((kA - g2) * (g2 - kB));
        if (i >= 3) out[2] = -(dm::sin(kC * kPi * x[0]) - 0.5);
        return;
    }
    if (i <= 8) {
        f[0] = x[0] + 10.0 * g1 + kShift;
        f[1] = h + 10.0 * g2 + kShift;
    } else {
        f[0] = (1.0 + kShift) * x[0] * (10.0 * g1 + 1.0);
        f[1] = (1.0 + kShift) * h * (10.0 * g2 + 1.0);
    }
    switch (i) {
        case 5:
            out[0] = -ellipse(f[0], f[1], 1.6, 1.6, 2.0, 4.0);
            out[1] = -ellipse(f[0], f[1], 2.5, 2.5, 2.0, 8.0);
            return;
        case 6:
            out[0] = -ellipse(f[0], f[1], 1.8, 1.8, 2.0, 8.0);
            out[1] = -ellipse(f[0], f[1], 2.8, 2.8, 2.0, 8.0);
            return;
        case 7:
        case 8:
            out[0] = -ellipse(f[0], f[1], 1.2, 1.2, 2.0, 6.0);
            out[1] = -ellipse(f[0], f[1], 2.25, 2.25, 2.5, 12.0);
            out[2] = -ellipse(f[0], f[1], 3.5, 3.5, 2.5, 10.0);
            return;
        default:
            break;
    }
    // 9-12: one ellipse and the sine
    static constexpr double kEll[4][4] = {{1.4, 1.4, 1.5, 6.0}, {1.1, 1.2, 2.0, 4.0},
                                          {1.2, 1.2, 1.5, 5.0}, {1.6, 1.6, 1.5, 6.0}};
    static constexpr double kSineE[4] = {2.0, 1.0, 2.1, 2.5};
    const double* q = kEll[i - 9];
    out[0] = -ellipse(f[0], f[1], q[0], q[1], q[2], q[3]);
    out[1] = -(f[0] * kSinA + f[1] * kCosA - dm::sin(4.0 * kPi * (f[0] * kCosA - f[1] * kSinA)) - kSineE[i - 9]);
}

}   // namespace mootation::suite::lircmop

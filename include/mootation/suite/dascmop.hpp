#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DAS-CMOP1-9 (Fan et al., Evolutionary Computation 28(3), 2020, Table 2,
// Eq. 7), as benchmarks/dascmop.py evaluates them (task 17, supplement 3,
// stage 3a): g_values, shape and constraints_c one for one, in the order of
// their NumPy rows (common.hpp). dascmop.py stays the reference.
//
// The difficulty triplet's constants (b, d, e, r) come from Python
// (dascmop.params): e is +inf where zeta = 0 (c2 = g - d) and equals d where
// zeta = 1 (c2 = EQ_EPS - |g - d|, Eq. 13). The constraints in the library's
// sign, -c: two objectives 11 (sin, g, nine ellipses), three 7 (sin, cos, g,
// three spheres, the centre's).
// ============================================================================

#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::dascmop {

constexpr double kA = 20.0, kEqEps = 1e-6;
constexpr double kAk2 = 0.3, kBk2 = 1.2;
constexpr double kCosT = 0.7071067811865476;      // math.cos(-0.25 pi)
constexpr double kSinT = -0.7071067811865476;     // math.sin(-0.25 pi)
constexpr double kP[9] = {0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0};
constexpr double kQ[9] = {1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5};
constexpr double kInvSqrt3 = 0.5773502691896258;  // 1.0 / math.sqrt(3.0)

inline int n_obj(int i) { return i <= 6 ? 2 : 3; }
inline int n_con(int i) { return i <= 6 ? 11 : 7; }

// g of the n variables x (dascmop.g_values)
inline double g_value(int i, const double* x, int n)
{
    std::vector<double> t;
    if (i <= 3) {
        const double s = dm::sin(0.5 * kPi * x[0]);
        for (int j = 1; j < n; ++j) t.push_back((x[j] - s) * (x[j] - s));
        return row_sum(t);
    }
    if (i <= 8) {
        const int first = i <= 6 ? 1 : 2;
        for (int j = first; j < n; ++j) {
            const double z = x[j] - 0.5;
            t.push_back(z * z - dm::cos(20.0 * kPi * z));
        }
        return static_cast<double>(n - first) + row_sum(t);
    }
    const double sum12 = x[0] + x[1];
    for (int j = 2; j < n; ++j) {
        const double c = 0.25 * (j + 1) / n * kPi;       // 0.25 * j / n * pi, j counted from 1
        const double d = x[j] - dm::cos(c * sum12);
        t.push_back(d * d);
    }
    return row_sum(t);
}

// the objectives without g (dascmop.shape)
inline void shape(int i, double x1, double x2, double* f)
{
    if (i == 1 || i == 4) {
        f[0] = x1;
        f[1] = 1.0 - x1 * x1;
    } else if (i == 2 || i == 5) {
        f[0] = x1;
        f[1] = 1.0 - std::sqrt(x1);
    } else if (i == 3 || i == 6) {
        f[0] = x1;
        f[1] = 1.0 - std::sqrt(x1) + 0.5 * std::abs(dm::sin(5.0 * kPi * x1));
    } else if (i == 7) {
        f[0] = x1 * x2;
        f[1] = x2 * (1.0 - x1);
        f[2] = 1.0 - x2;
    } else {
        const double c1 = dm::cos(0.5 * kPi * x1), s1 = dm::sin(0.5 * kPi * x1);
        f[0] = c1 * dm::cos(0.5 * kPi * x2);
        f[1] = c1 * dm::sin(0.5 * kPi * x2);
        f[2] = s1;
    }
}

// f and the constraints (<= 0 satisfied) of the point x of n variables;
// bder = (b, d, e, r) of the triplet
inline void evaluate(int i, const double* x, int n, const std::vector<double>& bder, double* f, double* out)
{
    const double b = bder[0], d = bder[1], e = bder[2], r = bder[3];
    const double g = g_value(i, x, n);
    const int M = n_obj(i);
    shape(i, x[0], x[1], f);
    for (int k = 0; k < M; ++k) f[k] = f[k] + g;
    const double c_g = std::isinf(e) ? g - d : e == d ? kEqEps - std::abs(g - d) : (e - g) * (g - d);
    if (M == 2) {
        out[0] = -(dm::sin(kA * kPi * x[0]) - b);
        out[1] = -c_g;
        for (int k = 0; k < 9; ++k) {
            const double f1 = f[0] - kP[k], f2 = f[1] - kQ[k];
            const double u = f1 * kCosT - f2 * kSinT, v = f1 * kSinT + f2 * kCosT;
            out[2 + k] = -(u * u / kAk2 + v * v / kBk2 - r);
        }
        return;
    }
    out[0] = -(dm::sin(kA * kPi * x[0]) - b);
    out[1] = -(dm::cos(kA * kPi * x[1]) - b);
    out[2] = -c_g;
    const double sq[3] = {f[0] * f[0], f[1] * f[1], f[2] * f[2]};
    const double total = row_sum(sq, 3);
    double centre[3];
    for (int k = 0; k < 3; ++k) {
        out[3 + k] = -(total - sq[k] + (f[k] - 1.0) * (f[k] - 1.0) - r * r);
        const double c = f[k] - kInvSqrt3;
        centre[k] = c * c;
    }
    out[6] = -(row_sum(centre, 3) - r * r);
}

}   // namespace mootation::suite::dascmop

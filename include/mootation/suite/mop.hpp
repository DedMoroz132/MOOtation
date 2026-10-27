#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOP1-7, the imbalanced test problems, as benchmarks/mop.py's mop_rows
// evaluates them: 10 variables, MOP1-5 two objectives, MOP6-7 three. The
// order of every operation is that of the NumPy rows there (common.hpp): the
// distance terms elementwise, summed in NumPy's pairwise order; t ** 0.6 and
// a scalar's ** (spow) pow, and sin, cos and exp, dm::'s.
// ============================================================================

#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::mop {

constexpr int kN = 10;

inline int n_obj(int which) { return which >= 6 ? 3 : 2; }

// g_a = sum(-0.9 t^2 + |t|^0.6) and g_b = sum(|t| / (1 + e^|t|)) over the
// t_j = x_j - offset of the variables from `first` on
inline double g_a(const double* x, int first, double offset)
{
    std::vector<double> v;
    for (int j = first; j < kN; ++j) {
        const double t = x[j] - offset;
        v.push_back(-0.9 * t * t + dm::pow(std::fabs(t), 0.6));
    }
    return row_sum(v);
}

inline double g_b(const double* x, double offset)
{
    std::vector<double> v;
    for (int j = 1; j < kN; ++j) {
        const double at = std::fabs(x[j] - offset);
        v.push_back(at / (1.0 + dm::exp(at)));
    }
    return row_sum(v);
}

// which = 1..7; x the point of 10 variables; f its objectives
inline void evaluate(int which, const double* x, double* f)
{
    const double x1 = x[0];
    if (which == 6 || which == 7) {
        const double x2 = x[1];
        const double g = 2.0 * dm::sin(kPi * x1) * g_a(x, 2, x1 * x2);
        if (which == 6) {
            f[0] = (1 + g) * x1 * x2;
            f[1] = (1 + g) * x1 * (1 - x2);
            f[2] = (1 + g) * (1 - x1);
            return;
        }
        const double c1 = dm::cos(0.5 * kPi * x1), c2 = dm::cos(0.5 * kPi * x2);
        const double s1 = dm::sin(0.5 * kPi * x1), s2 = dm::sin(0.5 * kPi * x2);
        f[0] = (1 + g) * c1 * c2;
        f[1] = (1 + g) * c1 * s2;
        f[2] = (1 + g) * s1;
        return;
    }
    const double offset = dm::sin(0.5 * kPi * x1);
    if (which == 1 || which == 5) {
        const double ga = g_a(x, 1, offset);
        const double g = (which == 1 ? 2.0 * dm::sin(kPi * x1) : 2.0 * std::fabs(dm::cos(kPi * x1))) * ga;
        f[0] = (1 + g) * x1;
        f[1] = (1 + g) * (1 - std::sqrt(x1));
        return;
    }
    const double gb = g_b(x, offset);
    if (which == 3) {
        const double g = 10.0 * dm::sin(0.5 * kPi * x1) * gb;
        f[0] = (1 + g) * dm::cos(0.5 * kPi * x1);
        f[1] = (1 + g) * dm::sin(0.5 * kPi * x1);
        return;
    }
    const double g = 10.0 * dm::sin(kPi * x1) * gb;
    f[0] = (1 + g) * x1;
    if (which == 2) f[1] = (1 + g) * (1 - x1 * x1);
    else f[1] = (1 + g) * (1 - dm::pow(x1, 0.5) * dm::pow(dm::cos(2 * kPi * x1), 2.0));       // MOP4
}

}   // namespace mootation::suite::mop

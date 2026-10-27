#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// BT1-9, the biased test problems (Li, Zhang & Deng, "Biased Multiobjective
// Optimization and Decomposition Algorithm", IEEE Trans. Cybern. 47(1),
// 2017, Appendix A), as benchmarks/bt.py's bt_rows evaluates them: 30
// variables, BT1-8 two objectives, BT9 three. The order of every operation is
// that of the NumPy rows there (common.hpp): a column y_j = x_j - offset(j)
// per index set, the bias transform elementwise, the set's values summed in
// NumPy's pairwise order; a scalar's ** (spow) and sin, cos and exp dm::'s.
// ============================================================================

#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::bt {

constexpr int kN = 30;

// D1 and D2: the distance's bias
inline double d1(double g, double theta) { return g * g + (1.0 - dm::exp(-g * g / theta)) / 5.0; }
inline double d2(double g, double theta) { return g * g + dm::pow(std::fabs(g), theta) / 5.0; }

// S1 and S2: the position's bias (S2's branches as np.where picks them)
inline double s1(double x, double gamma) { return dm::pow(std::fabs(x), gamma); }

inline double s2(double x, double gamma)
{
    auto p = [gamma](double base) { return dm::pow(base < 0.0 ? 0.0 : base, gamma); };
    if (x < 0.25) return (1 - p(1 - 4 * x)) / 4.0;
    if (x < 0.50) return (1 + p(4 * x - 1)) / 4.0;
    if (x < 0.75) return (3 - p(3 - 4 * x)) / 4.0;
    return (3 + p(4 * x - 3)) / 4.0;
}

inline double q(double z) { return 4.0 * z * z - dm::cos(8.0 * kPi * z) + 1.0; }

// _idx_sets: the 1-indexed j = m..n with j mod m = k, for k = 0..m-1
inline std::vector<int> index_set(int m, int k)
{
    std::vector<int> out;
    for (int j = m; j <= kN; ++j)
        if (j % m == k) out.push_back(j);
    return out;
}

// _sin_off(j) = sin(j pi / (2 N))
inline double sin_off(int j) { return dm::sin(j * kPi / (2 * kN)); }

// the transform `t` of y_j = x_j - offset(j) over the index set, summed
template <class Offset, class Transform>
inline double set_sum(const double* x, int m, int k, Offset offset, Transform t)
{
    std::vector<double> v;
    for (int j : index_set(m, k)) v.push_back(t(x[j - 1] - offset(j)));
    return row_sum(v);
}

// which = 1..9; x the point of 30 variables; f its objectives
inline void evaluate(int which, const double* x, double* f)
{
    const double x1 = x[0];
    if (which == 9) {
        const double x2 = x[1];
        double s[3];
        for (int k = 0; k < 3; ++k)
            s[k] = 10.0 * set_sum(x, 3, k, sin_off, [](double y) { return d1(y, 1e-9); });
        const double c1 = dm::cos(0.5 * kPi * x1), c2 = dm::cos(0.5 * kPi * x2);
        const double si1 = dm::sin(0.5 * kPi * x1), si2 = dm::sin(0.5 * kPi * x2);
        f[0] = c1 * c2 + s[0];
        f[1] = c1 * si2 + s[1];
        f[2] = si1 + s[2];
        return;
    }
    double theta = 0.0;
    switch (which) {
        case 1: theta = 1e-10; break;
        case 3: theta = 1e-8; break;
        case 4: theta = 1e-8; break;
        case 5: theta = 1e-10; break;
        case 6: theta = 1e-4; break;
        case 7: theta = 1e-3; break;
        case 8: theta = 1e-3; break;
        default: break;
    }
    const double s7 = which == 7 ? dm::sin(6 * kPi * x1) : 0.0;
    auto offset = [which, x1, s7](int j) {
        if (which == 6 || which == 8) return dm::pow(x1, 0.5 + 1.5 * (j - 1) / (kN - 1));
        if (which == 7) return s7;
        return sin_off(j);
    };
    double a = 0.0, b = 0.0;
    if (which == 2) {
        a = set_sum(x, 2, 0, offset, [](double y) { return d2(y, 0.2); });
        b = set_sum(x, 2, 1, offset, [](double y) { return d2(y, 0.2); });
    } else if (which == 8) {
        a = set_sum(x, 2, 0, offset, [theta](double y) { return q(d1(y, theta)); });
        b = set_sum(x, 2, 1, offset, [theta](double y) { return q(d1(y, theta)); });
    } else {
        a = set_sum(x, 2, 0, offset, [theta](double y) { return d1(y, theta); });
        b = set_sum(x, 2, 1, offset, [theta](double y) { return d1(y, theta); });
    }
    if (which == 3 || which == 4) {
        const double s = which == 3 ? s1(x1, 0.02) : s2(x1, 0.06);
        f[0] = s + a;
        f[1] = 1 - std::sqrt(s) + b;
        return;
    }
    if (which == 5) {
        f[0] = x1 + a;
        f[1] = (1 - x1) * (1 - x1 * dm::sin(8.5 * kPi * x1)) + b;
        return;
    }
    f[0] = x1 + a;
    f[1] = 1 - std::sqrt(x1) + b;
}

}   // namespace mootation::suite::bt

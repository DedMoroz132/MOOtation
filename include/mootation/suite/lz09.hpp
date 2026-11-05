#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// LZ09 F1-F9, the test problems with complicated Pareto sets: H. Li and Q.
// Zhang, "Multiobjective Optimization Problems With Complicated Pareto Sets,
// MOEA/D and NSGA-II", IEEE TEVC 13(2): 284-302, 2009 (source
// moead-de_li2009), Table I, read off the PDF (task 20 of 2026-11-02,
// §20.1.3). benchmarks/lz09.py is the cross-check; the order of every
// operation is its NumPy rows' (common.hpp): a set's terms summed pairwise,
// F8's cosines multiplied left to right, sin, cos and pow dm::'s.
//
// Every instance is the paper's (2)-(3), f_i = alpha_i(x_I) + beta_i(x_II -
// g(x_I)): the Pareto set is x_II = g(x_I) (Theorem 1; optimum below) over
// x_1 in [0, 1] (F6: x_1, x_2 in [0, 1]^2), the front alpha's: f2 = 1 -
// sqrt(f1) (F1-F5, F7, F8), f2 = 1 - f1^2 (F9), the octant of the unit sphere
// (F6). n = 30 for F1-F5 and F9, 10 for F6-F8 (§V-A); the box [0, 1]^n for
// F1, F7, F8, [0, 1] x [-1, 1]^(n-1) for F2-F5, F9, [0, 1]^2 x [-2, 2]^(n-2)
// for F6.
//
// DEVIATIONS AND READINGS (the paper's text where libraries differ):
//   LZ09-1. J1 = {j odd, 2 <= j <= n} into f1, J2 = {j even} into f2 (F6: j -
//     1, j - 2 and j multiples of 3, 3 <= j <= n, into f1, f2, f3), as the
//     table prints them; j counts the variables from 1. jMetal's code swaps
//     J1 and J2 and stretches every variable from [0, 1] (the task's reading
//     P5), a variant of its own, not here yet.
//   LZ09-2. F8's cosine is cos(20 y_j pi / sqrt(j)), j the variable's number
//     as printed (jMetal: the index within the set).
// ============================================================================

#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::lz09 {

inline int n_var(int which) { return which >= 6 && which <= 8 ? 10 : 30; }
inline int n_obj(int which) { return which == 6 ? 3 : 2; }

inline Bounds bounds(int which)
{
    const int n = n_var(which);
    if (which == 1 || which == 7 || which == 8) return box(n, 0.0, 1.0);
    Bounds b = box(n, which == 6 ? -2.0 : -1.0, which == 6 ? 2.0 : 1.0);
    b[0] = {0.0, 1.0};
    if (which == 6) b[1] = {0.0, 1.0};
    return b;
}

// the first variable of x_II (1-indexed): 3 for F6, 2 otherwise
inline int first(int which) { return which == 6 ? 3 : 2; }

// g(x_I)_j, the optimal x_j given x_1 (and x_2): lz09.optimum
inline double optimum(int which, const double* x, int j)
{
    const int n = n_var(which);
    const double x1 = x[0], jp = j * kPi / n;
    if (which == 6) return 2.0 * x[1] * dm::sin(2.0 * kPi * x1 + jp);
    if (which == 1 || which == 7 || which == 8) return dm::pow(x1, 0.5 * (1.0 + 3.0 * (j - 2) / (n - 2)));
    if (which == 2 || which == 9) return dm::sin(6.0 * kPi * x1 + jp);
    const bool odd = j % 2 == 1;
    if (which == 3) return odd ? 0.8 * x1 * dm::cos(6.0 * kPi * x1 + jp) : 0.8 * x1 * dm::sin(6.0 * kPi * x1 + jp);
    if (which == 4)
        return odd ? 0.8 * x1 * dm::cos((6.0 * kPi * x1 + jp) / 3.0) : 0.8 * x1 * dm::sin(6.0 * kPi * x1 + jp);
    const double a = 0.3 * x1 * x1 * dm::cos(24.0 * kPi * x1 + 4.0 * j * kPi / n) + 0.6 * x1;       // F5
    return odd ? a * dm::cos(6.0 * kPi * x1 + jp) : a * dm::sin(6.0 * kPi * x1 + jp);
}

// which = 1..9; x the point of n_var(which) variables; f its objectives
inline void evaluate(int which, const double* x, double* f)
{
    const int n = n_var(which), m = n_obj(which);
    // the sets: j with (j - shift) % m == 0 from `first` on, shift 1, 2, 0
    // for F6's J1, J2, J3, and odd / even for the others
    auto member = [which](int j, int k) { return which == 6 ? (j - (k == 2 ? 0 : k + 1)) % 3 == 0 : j % 2 == (k == 0 ? 1 : 0); };
    double s[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < m; ++k) {
        std::vector<double> sq, cs;
        for (int j = first(which); j <= n; ++j) {
            if (!member(j, k)) continue;
            const double y = x[j - 1] - optimum(which, x, j);
            if (which == 7) {
                sq.push_back(4.0 * y * y - dm::cos(8.0 * y * kPi) + 1.0);
            } else {
                sq.push_back(y * y);
                if (which == 8) cs.push_back(dm::cos(20.0 * y * kPi / std::sqrt(static_cast<double>(j))));
            }
        }
        const double c = 2.0 / static_cast<double>(sq.size());
        s[k] = which == 8 ? c * (4.0 * row_sum(sq) - 2.0 * row_prod(cs.data(), cs.size()) + 2.0) : c * row_sum(sq);
    }
    const double x1 = x[0];
    if (which == 6) {
        const double x2 = x[1];
        const double c1 = dm::cos(0.5 * x1 * kPi), c2 = dm::cos(0.5 * x2 * kPi);
        const double s1 = dm::sin(0.5 * x1 * kPi), s2 = dm::sin(0.5 * x2 * kPi);
        f[0] = c1 * c2 + s[0];
        f[1] = c1 * s2 + s[1];
        f[2] = s1 + s[2];
        return;
    }
    f[0] = x1 + s[0];
    f[1] = (which == 9 ? 1.0 - x1 * x1 : 1.0 - std::sqrt(x1)) + s[1];
}

}   // namespace mootation::suite::lz09

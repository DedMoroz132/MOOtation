#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// UF1-UF10, the unconstrained problems of the CEC 2009 competition: Q. Zhang,
// A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, S. Tiwari, "Multiobjective
// optimization Test Instances for the CEC 2009 Special Session and
// Competition", Technical Report CES-487, University of Essex, "April 20, 2009
// DRAFT" (source cec2009-uf-cf_zhang2008), §II; task 20 of 2026-11-02,
// §20.1.1. benchmarks/uf.py is the cross-check; the order of every operation
// is its NumPy rows' (common.hpp): a set's terms summed pairwise, the cosines
// multiplied left to right, sin, cos, exp and pow dm::'s.
//
// n = 30; UF1-UF7 two objectives, UF8-UF10 three; UF5 N = 10, epsilon = 0.1;
// UF6 N = 2, epsilon = 0.1; UF9 epsilon = 0.1 (the contest's values). The
// boxes: [0, 1] x [-1, 1]^(n-1) (UF1, UF2, UF5-UF7), [0, 1]^n (UF3), [0, 1] x
// [-2, 2]^(n-1) (UF4), [0, 1]^2 x [-2, 2]^(n-2) (UF8-UF10). UF1, UF2 and UF3
// are LZ09's F2, F5 and F8 at n = 30 (lz09.hpp). The fronts and the Pareto
// sets are the report's ("Its PF", "Its PS"; uf.py has them).
//
// DEVIATIONS AND READINGS (the report's text):
//   UF-1. UF10's sums: f2's over J2 and f3's over J3, where the report prints
//     J1 for all three (the task's reading P1): pagmo2, PlatEMO and jMetal sum
//     so, and the report's own factors 2/|J2| and 2/|J3| say so.
//   UF-2. Harmless misprints read as meant: UF3's "the same as those of F1"
//     (UF1's), UF4's "y_i = x_j - ..." (y_j), UF4's "Its PF is f2 = 1 -
//     f2^2" (1 - f1^2), UF10's "f1^2 + f2^2 + f3^3 = 1" (f3^2).
// UF11-UF13 are not here: they need the competition's data files.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::uf {

constexpr int kN = 30;
constexpr double kEps = 0.1;                         // UF5, UF6, UF9

inline int n_obj(int which) { return which >= 8 ? 3 : 2; }
inline int parts(int which) { return which == 5 ? 10 : 2; }   // the N of UF5 and UF6

inline Bounds bounds(int which)
{
    if (which == 3) return box(kN, 0.0, 1.0);
    if (which >= 8) {
        Bounds b = box(kN, -2.0, 2.0);
        b[0] = b[1] = {0.0, 1.0};
        return b;
    }
    Bounds b = box(kN, which == 4 ? -2.0 : -1.0, which == 4 ? 2.0 : 1.0);
    b[0] = {0.0, 1.0};
    return b;
}

// the first variable of the distance part (1-indexed): 3 for UF8-UF10
inline int first(int which) { return which >= 8 ? 3 : 2; }

// the optimal x_j given x_1 (and x_2): uf.optimum
inline double optimum(int which, const double* x, int j)
{
    const double x1 = x[0], jp = j * kPi / kN;
    if (which >= 8) return 2.0 * x[1] * dm::sin(2.0 * kPi * x1 + jp);
    if (which == 3) return dm::pow(x1, 0.5 * (1.0 + 3.0 * (j - 2) / (kN - 2)));
    if (which == 2) {
        const double a = 0.3 * x1 * x1 * dm::cos(24.0 * kPi * x1 + 4.0 * j * kPi / kN) + 0.6 * x1;
        return j % 2 == 1 ? a * dm::cos(6.0 * kPi * x1 + jp) : a * dm::sin(6.0 * kPi * x1 + jp);
    }
    return dm::sin(6.0 * kPi * x1 + jp);
}

// which = 1..10; x the point of 30 variables; f its objectives
inline void evaluate(int which, const double* x, double* f)
{
    const int m = n_obj(which);
    auto member = [which](int j, int k) {
        if (which >= 8) return (j - (k == 2 ? 0 : k + 1)) % 3 == 0;   // J1: j - 1, J2: j - 2, J3: j a multiple of 3
        return j % 2 == (k == 0 ? 1 : 0);                             // J1 odd, J2 even
    };
    double s[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < m; ++k) {
        std::vector<double> t, cs;
        for (int j = first(which); j <= kN; ++j) {
            if (!member(j, k)) continue;
            const double y = x[j - 1] - optimum(which, x, j);
            if (which == 3 || which == 6) {
                t.push_back(y * y);
                cs.push_back(dm::cos(20.0 * y * kPi / std::sqrt(static_cast<double>(j))));
            } else if (which == 4) {
                const double a = std::fabs(y);
                t.push_back(a / (1.0 + dm::exp(2.0 * a)));
            } else if (which == 5) {
                t.push_back(2.0 * y * y - dm::cos(4.0 * kPi * y) + 1.0);
            } else if (which == 10) {
                t.push_back(4.0 * y * y - dm::cos(8.0 * kPi * y) + 1.0);
            } else {
                t.push_back(y * y);
            }
        }
        const double sum = which == 3 || which == 6 ? 4.0 * row_sum(t) - 2.0 * row_prod(cs.data(), cs.size()) + 2.0
                                                    : row_sum(t);
        s[k] = 2.0 / static_cast<double>(t.size()) * sum;
    }
    const double x1 = x[0];
    if (which >= 8) {
        const double x2 = x[1];
        if (which == 9) {
            const double mx = std::max(0.0, (1.0 + kEps) * (1.0 - 4.0 * (2.0 * x1 - 1.0) * (2.0 * x1 - 1.0)));
            f[0] = 0.5 * (mx + 2.0 * x1) * x2 + s[0];
            f[1] = 0.5 * (mx - 2.0 * x1 + 2.0) * x2 + s[1];
            f[2] = 1.0 - x2 + s[2];
            return;
        }
        const double c1 = dm::cos(0.5 * x1 * kPi), c2 = dm::cos(0.5 * x2 * kPi);
        const double s1 = dm::sin(0.5 * x1 * kPi), s2 = dm::sin(0.5 * x2 * kPi);
        f[0] = c1 * c2 + s[0];
        f[1] = c1 * s2 + s[1];
        f[2] = s1 + s[2];
        return;
    }
    if (which == 4) {
        f[0] = x1 + s[0];
        f[1] = 1.0 - x1 * x1 + s[1];
        return;
    }
    if (which == 5 || which == 6) {
        const double N = parts(which);
        const double e = which == 5 ? (1.0 / (2.0 * N) + kEps) * std::fabs(dm::sin(2.0 * N * kPi * x1))
                                    : std::max(0.0, 2.0 * (1.0 / (2.0 * N) + kEps) * dm::sin(2.0 * N * kPi * x1));
        f[0] = x1 + e + s[0];
        f[1] = 1.0 - x1 + e + s[1];
        return;
    }
    if (which == 7) {
        const double r = dm::pow(x1, 0.2);
        f[0] = r + s[0];
        f[1] = 1.0 - r + s[1];
        return;
    }
    f[0] = x1 + s[0];
    f[1] = 1.0 - std::sqrt(x1) + s[1];
}

}   // namespace mootation::suite::uf

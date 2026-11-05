#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CF1-CF10, the constrained problems of the CEC 2009 competition: Q. Zhang,
// A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, S. Tiwari, CES-487, "April 20,
// 2009 DRAFT" (source cec2009-uf-cf_zhang2008), §III; task 20 of 2026-11-02,
// §20.1.2. benchmarks/cf.py is the cross-check; the order of every operation
// is its NumPy rows' (common.hpp), sin, cos, exp, pow and acos dm::'s. The
// registry's names are CF1-CEC2009 .. CF10-CEC2009 (jMetal's cf/CF1 .. CF16
// is another set).
//
// n = 10; CF1-CF7 two objectives, CF8-CF10 three; (N, a) = (10, 1) for CF1,
// (2, 1) for CF2 and CF3, (2, 4) for CF8, (2, 3) for CF9, (2, 1) for CF10. The
// boxes: [0, 1]^n (CF1), [0, 1] x [-1, 1]^(n-1) (CF2), [0, 1] x [-2,
// 2]^(n-1) (CF3-CF7), [0, 1]^2 x [-4, 4]^(n-2) (CF8), [0, 1]^2 x [-2,
// 2]^(n-2) (CF9, CF10). The report writes c(x) >= 0; the library's
// constraint is g = -c (g <= 0 feasible). The Pareto sets and fronts:
// cf.py's docstring; pareto_point below.
//
// DEVIATIONS AND READINGS:
//   CF-1. No clipping of x into the box inside the problem (the task's
//     reading P3): PlatEMO clips in CF8 and CF9; the box is the algorithm's.
//   CF-2. The competition's tolerance (§IV, c >= -1e-6) belongs to its
//     assessment, not to the problem: g = -c as printed. CF1's and CF8's
//     Pareto points sit on the constraint's boundary, where the computed g is
//     a rounding error either side of 0.
//   CF-3. CF8-CF10 divide by 1 - f3^2, 0 at the front's pole f3 = 1 (x_1 = 1,
//     the J3 terms at their optimum): there the constraint is 0/0 or x/0 and
//     comes out NaN, as the printed formula and PlatEMO's code give it.
//   CF-4. CF8-CF10 as in the version of 20 April 2009 (the task's reading P2:
//     the competition's final report says earlier versions had misprints
//     there; results of 2008-2009 on CF8-CF10 are no reference).
//   CF-5. |sin| in CF1 and CF8, sin without the modulus in CF2, CF3, CF9 and
//     CF10; the threshold of CF4's and CF5's h_2 is 3/2 (1 - sqrt(2)/2): as
//     printed.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::cf {

constexpr int kN = 10;

inline int n_obj(int which) { return which >= 8 ? 3 : 2; }
inline int n_con(int which) { return which == 6 || which == 7 ? 2 : 1; }

// N and a of CF1-CF3 and CF8-CF10
inline int parts(int which) { return which == 1 ? 10 : 2; }
inline double amplitude(int which) { return which == 8 ? 4.0 : which == 9 ? 3.0 : 1.0; }

inline Bounds bounds(int which)
{
    if (which == 1) return box(kN, 0.0, 1.0);
    if (which >= 8) {
        const double w = which == 8 ? 4.0 : 2.0;
        Bounds b = box(kN, -w, w);
        b[0] = b[1] = {0.0, 1.0};
        return b;
    }
    const double w = which == 2 ? 1.0 : 2.0;
    Bounds b = box(kN, -w, w);
    b[0] = {0.0, 1.0};
    return b;
}

inline int first(int which) { return which >= 8 ? 3 : 2; }

// the x_j at which y_j = 0, given x_1 (and x_2): cf.optimum
inline double optimum(int which, const double* x, int j)
{
    const double x1 = x[0], jp = j * kPi / kN;
    if (which >= 8) return 2.0 * x[1] * dm::sin(2.0 * kPi * x1 + jp);
    const bool odd = j % 2 == 1;
    if (which == 1) return dm::pow(x1, 0.5 * (1.0 + 3.0 * (j - 2) / (kN - 2)));
    if (which == 3 || which == 4) return dm::sin(6.0 * kPi * x1 + jp);
    if (which == 2 || which == 7) {
        const bool sine = which == 2 ? odd : !odd;
        return sine ? dm::sin(6.0 * kPi * x1 + jp) : dm::cos(6.0 * kPi * x1 + jp);
    }
    return odd ? 0.8 * x1 * dm::cos(6.0 * kPi * x1 + jp) : 0.8 * x1 * dm::sin(6.0 * kPi * x1 + jp);   // CF5, CF6
}

inline double sign(double v) { return v > 0.0 ? 1.0 : v < 0.0 ? -1.0 : 0.0; }

// which = 1..10; x the point of 10 variables; f its objectives, g = -c
inline void evaluate(int which, const double* x, double* f, double* g)
{
    const int m = n_obj(which);
    const double x1 = x[0];
    auto member = [which](int j, int k) {
        if (which >= 8) return (j - (k == 2 ? 0 : k + 1)) % 3 == 0;
        return j % 2 == (k == 0 ? 1 : 0);
    };
    double s[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < m; ++k) {
        std::vector<double> t, cs;
        for (int j = first(which); j <= kN; ++j) {
            if (!member(j, k)) continue;
            const double y = x[j - 1] - optimum(which, x, j);
            if (which == 10) {
                t.push_back(4.0 * y * y - dm::cos(8.0 * kPi * y) + 1.0);
            } else if (which == 3) {
                t.push_back(y * y);
                cs.push_back(dm::cos(20.0 * y * kPi / std::sqrt(static_cast<double>(j))));
            } else if ((which == 4 || which == 5) && j == 2) {
                const double threshold = 1.5 * (1.0 - std::sqrt(2.0) / 2.0);
                t.push_back(y < threshold ? std::fabs(y) : 0.125 + (y - 1.0) * (y - 1.0));
            } else if (which == 5 || (which == 7 && j != 2 && j != 4)) {
                t.push_back(2.0 * y * y - dm::cos(4.0 * kPi * y) + 1.0);
            } else {
                t.push_back(y * y);
            }
        }
        if (which >= 4 && which <= 7) s[k] = row_sum(t);                      // sum of h_j(y_j)
        else if (which == 3)
            s[k] = 2.0 / static_cast<double>(t.size()) * (4.0 * row_sum(t) - 2.0 * row_prod(cs.data(), cs.size()) + 2.0);
        else s[k] = 2.0 / static_cast<double>(t.size()) * row_sum(t);
    }
    if (which >= 8) {
        const double x2 = x[1];
        const double c1 = dm::cos(0.5 * x1 * kPi), c2 = dm::cos(0.5 * x2 * kPi);
        const double s1 = dm::sin(0.5 * x1 * kPi), s2 = dm::sin(0.5 * x2 * kPi);
        f[0] = c1 * c2 + s[0];
        f[1] = c1 * s2 + s[1];
        f[2] = s1 + s[2];
        const double N = parts(which), a = amplitude(which);
        const double r = 1.0 - f[2] * f[2];
        const double sn = dm::sin(N * kPi * ((f[0] * f[0] - f[1] * f[1]) / r + 1.0));
        const double c = (f[0] * f[0] + f[1] * f[1]) / r - a * (which == 8 ? std::fabs(sn) : sn) - 1.0;
        g[0] = -c;
        return;
    }
    if (which <= 3) {
        const double N = parts(which), a = amplitude(which);
        double c = 0.0;
        if (which == 1) {
            f[0] = x1 + s[0];
            f[1] = 1.0 - x1 + s[1];
            c = f[0] + f[1] - a * std::fabs(dm::sin(N * kPi * (f[0] - f[1] + 1.0))) - 1.0;
        } else if (which == 2) {
            f[0] = x1 + s[0];
            f[1] = 1.0 - std::sqrt(x1) + s[1];
            const double q = std::sqrt(f[0]);
            const double t = f[1] + q - a * dm::sin(N * kPi * (q - f[1] + 1.0)) - 1.0;
            c = t / (1.0 + dm::exp(4.0 * std::fabs(t)));
        } else {
            f[0] = x1 + s[0];
            f[1] = 1.0 - x1 * x1 + s[1];
            c = f[1] + f[0] * f[0] - a * dm::sin(N * kPi * (f[0] * f[0] - f[1] + 1.0)) - 1.0;
        }
        g[0] = -c;
        return;
    }
    if (which == 4 || which == 5) {
        f[0] = x1 + s[0];
        f[1] = 1.0 - x1 + s[1];
        const double base = dm::sin(6.0 * kPi * x1 + 2.0 * kPi / kN);
        double c = x[1] - (which == 4 ? base : 0.8 * x1 * base) - 0.5 * x1 + 0.25;
        if (which == 4) c = c / (1.0 + dm::exp(4.0 * std::fabs(c)));
        g[0] = -c;
        return;
    }
    f[0] = x1 + s[0];
    f[1] = (1.0 - x1) * (1.0 - x1) + s[1];
    const double u = 1.0 - x1;
    const double a = 0.5 * u - u * u;
    const double b = 0.25 * std::sqrt(u) - 0.5 * u;
    const double k = which == 7 ? 1.0 : 0.8 * x1;
    g[0] = -(x[1] - k * dm::sin(6.0 * kPi * x1 + 2.0 * kPi / kN) - sign(a) * std::sqrt(std::fabs(a)));
    g[1] = -(x[3] - k * dm::sin(6.0 * kPi * x1 + 4.0 * kPi / kN) - sign(b) * std::sqrt(std::fabs(b)));
}

// a Pareto-optimal point given x_1 (and x_2) in x: cf.pareto_set's rows — y_j
// = 0 but CF4's and CF5's y_2 and CF6's and CF7's y_2 and y_4, which the
// constraints hold away from 0
inline void pareto_point(int which, double* x)
{
    for (int j = first(which); j <= kN; ++j) x[j - 1] = optimum(which, x, j);
    const double x1 = x[0];
    if (which == 4 || which == 5) {
        x[1] += x1 <= 0.5 ? 0.0 : x1 <= 0.75 ? 0.5 * x1 - 0.25 : 1.0;
    } else if (which == 6 || which == 7) {
        const double u = 1.0 - x1;
        x[1] += std::sqrt(std::max(0.0, 0.5 * u - u * u));
        x[3] += std::sqrt(std::max(0.0, 0.25 * std::sqrt(u) - 0.5 * u));
    }
}

}   // namespace mootation::suite::cf

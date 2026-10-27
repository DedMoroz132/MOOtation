#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// catZDT1, as benchmarks/categorical.py evaluates it: ZDT1 whose distance
// function g the first variable picks, the index of one of the labels
// rastrigin, linear and quadratic (rounded half to even, clamped), then
// x1 .. x10. The order of every operation is that of the Python there
// (common.hpp): its sums term by term from 0, a float's ** pow and cos dm::'s.
// ============================================================================

#include <cmath>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::catzdt {

constexpr int kReal = 10;                             // categorical.N_REAL

// x = (c, x1, ..., x10); f two objectives
inline void evaluate(const double* x, double* f)
{
    double c = std::nearbyint(x[0]);
    c = c < 0.0 ? 0.0 : c > 2.0 ? 2.0 : c;
    const double x1 = x[1];
    const double* y = x + 2;
    const int n = kReal - 1;
    double s = 0.0, g = 0.0;
    if (c == 0.0) {                                   // rastrigin
        for (int i = 0; i < n; ++i)
            s = s + (dm::pow(10.0 * y[i] - 5.0, 2.0) - 10.0 * dm::cos(4.0 * kPi * (10.0 * y[i] - 5.0)));
        g = 1.0 + 10.0 * n + s;
    } else if (c == 1.0) {                            // linear
        for (int i = 0; i < n; ++i) s = s + y[i];
        g = 1.02 + 9.0 * s / n;
    } else {                                          // quadratic
        for (int i = 0; i < n; ++i) s = s + dm::pow(y[i] - 0.75, 2.0);
        g = 1.05 + 9.0 * s / n;
    }
    f[0] = x1;
    f[1] = g * (1.0 - std::sqrt(x1 / g));
}

}   // namespace mootation::suite::catzdt

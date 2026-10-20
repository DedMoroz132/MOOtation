#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ZDT1-4 and ZDT6 (Zitzler, Deb & Thiele 2000), as benchmarks/registry.py,
// _zdt_register, evaluates them: ZDT1-3 at 30 variables, ZDT4 and ZDT6 at 10.
// The order of every operation is that of the NumPy rows there (common.hpp):
// a float's ** of the point code is pow (spow), math's sin, cos and exp of one
// float dm::'s, and ZDT4's sum term by term from 0, as Python's sum added it.
// ============================================================================

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::zdt {

// g = 1 + 9 sum(x_2..x_n) / (n - 1) of ZDT1-3
inline double g_linear(const double* y, int n)
{
    return 1.0 + 9.0 * row_sum(y + 1, static_cast<std::size_t>(n - 1)) / (n - 1);
}

// which = 1, 2, 3, 4 or 6; y the point of n variables; f two objectives
inline void evaluate(int which, const double* y, int n, double* f)
{
    const double f1 = y[0];
    switch (which) {
        case 1: {
            const double g = g_linear(y, n);
            f[0] = f1;
            f[1] = g * (1.0 - std::sqrt(f1 / g));
            return;
        }
        case 2: {
            const double g = g_linear(y, n);
            f[0] = f1;
            f[1] = g * (1.0 - dm::pow(f1 / g, 2.0));
            return;
        }
        case 3: {
            const double g = g_linear(y, n);
            f[0] = f1;
            f[1] = g * (1.0 - std::sqrt(f1 / g) - (f1 / g) * dm::sin(10.0 * kPi * f1));
            return;
        }
        case 4: {
            double s = 0.0;
            for (int j = 1; j < n; ++j)
                s = s + (dm::pow(y[j], 2.0) - 10.0 * dm::cos(4.0 * kPi * y[j]));
            const double g = 1.0 + 10.0 * (n - 1) + s;
            f[0] = f1;
            f[1] = g * (1.0 - std::sqrt(f1 / g));
            return;
        }
        case 6: {
            const double h = 1.0 - dm::exp(-4.0 * y[0]) * dm::pow(dm::sin(6.0 * kPi * y[0]), 6.0);
            const double g = 1.0 + 9.0 * dm::pow(row_sum(y + 1, static_cast<std::size_t>(n - 1)) / 9.0, 0.25);
            f[0] = h;
            f[1] = g * (1.0 - dm::pow(h / g, 2.0));
            return;
        }
        default:
            return;
    }
}

}   // namespace mootation::suite::zdt

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The classical test problems the papers of the verification set use (task 17,
// supplement 2, §3): the problems to check the cores against their papers,
// not atlas problems. Written in C++ from the start; the tests compare them
// with the formulas in NumPy.
//
//   SCH   Deb, Pratap, Agarwal & Meyarivan, IEEE TEVC 6(2), 2002, Table I
//         (source nsga2_deb2002): n = 1, [-10^3, 10^3]; f = (x^2, (x - 2)^2)
//   FON   ibid.: n = 3, [-4, 4]; f_k = 1 - exp(-sum (x_i -+ 1/sqrt 3)^2)
//   POL   ibid.: n = 2, [-pi, pi]; f1 = 1 + (A1 - B1)^2 + (A2 - B2)^2,
//         f2 = (x1 + 3)^2 + (x2 + 1)^2
//   KUR   ibid.: n = 3, [-5, 5]; f1 = sum_{i<n} -10 exp(-0.2 sqrt(x_i^2 +
//         x_{i+1}^2)), f2 = sum |x_i|^0.8 + 5 sin(x_i^3) (the page's "5 sin
//         x_i^3", checked on the PDF)
//   SPH   Zitzler, Laumanns & Thiele, TIK-Report 103, 2001, Table 1 (source
//         spea2_zitzler2001), SPH-m: n = 100, [-10^3, 10^3];
//         f_j = sum_{i != j} x_i^2 + (x_j - 1)^2, j = 1..m (m = 2, 3)
//   QV    ibid.: n = 100, [-5, 5]; f1 = (mean(x_i^2 - 10 cos(2 pi x_i) +
//         10))^(1/4), f2 the same of x_i - 1.5
//   KURZ  ibid., the errata's KUR (the version their runs used): n = 100,
//         [-10^3, 10^3]; f1 = sum |x_i|^0.8 + 5 sin^3(x_i) + 3.5828,
//         f2 = sum_{i<n} 1 - exp(-0.2 sqrt(x_i^2 + x_{i+1}^2))
//
// Every sum term by term from 0, exp, sin, cos and pow the library's (dm::).
// ============================================================================

#include <cmath>
#include <string>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::classic {

enum class Which { SCH, FON, POL, KUR, SPH, QV, KURZ };

inline bool parse(const std::string& family, Which& w)
{
    if (family == "SCH") w = Which::SCH;
    else if (family == "FON") w = Which::FON;
    else if (family == "POL") w = Which::POL;
    else if (family == "KUR") w = Which::KUR;
    else if (family == "SPH") w = Which::SPH;
    else if (family == "QV") w = Which::QV;
    else if (family == "KURZ") w = Which::KURZ;
    else return false;
    return true;
}

// f of the n variables x, M objectives (SPH: M = m; the others 2)
inline void evaluate(Which w, const double* x, int n, int M, double* f)
{
    switch (w) {
        case Which::SCH:
            f[0] = x[0] * x[0];
            f[1] = (x[0] - 2.0) * (x[0] - 2.0);
            return;
        case Which::FON: {
            const double c = 1.0 / std::sqrt(3.0);
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < n; ++i) {
                s1 = s1 + (x[i] - c) * (x[i] - c);
                s2 = s2 + (x[i] + c) * (x[i] + c);
            }
            f[0] = 1.0 - dm::exp(-s1);
            f[1] = 1.0 - dm::exp(-s2);
            return;
        }
        case Which::POL: {
            const double A1 = 0.5 * dm::sin(1.0) - 2.0 * dm::cos(1.0) + dm::sin(2.0) - 1.5 * dm::cos(2.0);
            const double A2 = 1.5 * dm::sin(1.0) - dm::cos(1.0) + 2.0 * dm::sin(2.0) - 0.5 * dm::cos(2.0);
            const double B1 = 0.5 * dm::sin(x[0]) - 2.0 * dm::cos(x[0]) + dm::sin(x[1]) - 1.5 * dm::cos(x[1]);
            const double B2 = 1.5 * dm::sin(x[0]) - dm::cos(x[0]) + 2.0 * dm::sin(x[1]) - 0.5 * dm::cos(x[1]);
            f[0] = 1.0 + (A1 - B1) * (A1 - B1) + (A2 - B2) * (A2 - B2);
            f[1] = (x[0] + 3.0) * (x[0] + 3.0) + (x[1] + 1.0) * (x[1] + 1.0);
            return;
        }
        case Which::KUR: {
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i + 1 < n; ++i)
                s1 = s1 + (-10.0 * dm::exp(-0.2 * std::sqrt(x[i] * x[i] + x[i + 1] * x[i + 1])));
            for (int i = 0; i < n; ++i)
                s2 = s2 + (dm::pow(std::abs(x[i]), 0.8) + 5.0 * dm::sin(x[i] * x[i] * x[i]));
            f[0] = s1;
            f[1] = s2;
            return;
        }
        case Which::SPH:
            for (int j = 0; j < M; ++j) {
                double s = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double d = i == j ? x[i] - 1.0 : x[i];
                    s = s + d * d;
                }
                f[j] = s;
            }
            return;
        case Which::QV: {
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double y = x[i] - 1.5;
                s1 = s1 + (x[i] * x[i] - 10.0 * dm::cos(2.0 * kPi * x[i]) + 10.0);
                s2 = s2 + (y * y - 10.0 * dm::cos(2.0 * kPi * y) + 10.0);
            }
            f[0] = dm::pow(s1 / n, 0.25);
            f[1] = dm::pow(s2 / n, 0.25);
            return;
        }
        case Which::KURZ: {
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double s = dm::sin(x[i]);
                s1 = s1 + (dm::pow(std::abs(x[i]), 0.8) + 5.0 * (s * s * s) + 3.5828);
            }
            for (int i = 0; i + 1 < n; ++i)
                s2 = s2 + (1.0 - dm::exp(-0.2 * std::sqrt(x[i] * x[i] + x[i + 1] * x[i + 1])));
            f[0] = s1;
            f[1] = s2;
            return;
        }
    }
}

}   // namespace mootation::suite::classic

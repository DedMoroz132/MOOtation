#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The 13 scalable functions of X. Yao, Y. Liu, G. Lin, "Evolutionary
// programming made faster", IEEE TEVC 3(2), 1999, as JADE's Table II prints
// them (Zhang & Sanderson 2009; source jade_zhang2009), with its initial
// ranges; every minimum is 0. f6 is the step function sum floor(x + 0.5)^2
// (the converted table lost the floor); f7's noise, rand[0, 1), comes from the
// caller's generator. Elementary functions by dm:: (the same bits everywhere).
// For the tests of scalar/ (jade, shade, lshade, ...).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <mootation/detmath.hpp>
#include <mootation/random.hpp>

namespace yao1999 {

using Vec = std::vector<double>;
namespace dm = mootation::dm;

inline const double PI = 3.141592653589793;

inline double u(double x, double a, double k, double m)
{
    if (x > a) return k * dm::pow(x - a, m);
    if (x < -a) return k * dm::pow(-x - a, m);
    return 0.0;
}

inline double sin2(double t)
{
    const double s = dm::sin(t);
    return s * s;
}

// the initial range [-r, r]^D of f_id
inline double range(int id)
{
    const double r[14] = {0, 100, 10, 100, 100, 30, 100, 1.28, 500, 5.12, 32, 600, 50, 50};
    return r[id];
}

inline double f(int id, const Vec& x, std::mt19937* noise = nullptr)
{
    const std::size_t D = x.size();
    const double n = static_cast<double>(D);
    double s = 0.0, p = 1.0;
    switch (id) {
    case 1:
        for (double v : x) s += v * v;
        return s;
    case 2:
        for (double v : x) {
            s += std::fabs(v);
            p *= std::fabs(v);
        }
        return s + p;
    case 3: {
        double c = 0.0;
        for (double v : x) {
            c += v;
            s += c * c;
        }
        return s;
    }
    case 4:
        for (double v : x) s = std::max(s, std::fabs(v));
        return s;
    case 5:
        for (std::size_t i = 0; i + 1 < D; ++i) {
            const double a = x[i + 1] - x[i] * x[i];
            s += 100 * a * a + (x[i] - 1) * (x[i] - 1);
        }
        return s;
    case 6:
        for (double v : x) {
            const double t = std::floor(v + 0.5);
            s += t * t;
        }
        return s;
    case 7:
        for (std::size_t i = 0; i < D; ++i) {
            const double t = x[i] * x[i];
            s += static_cast<double>(i + 1) * t * t;
        }
        return s + (noise ? mootation::rnd::uniform01(*noise) : 0.0);
    case 8:
        for (double v : x) s += -v * dm::sin(std::sqrt(std::fabs(v)));
        return s + n * 418.98288727243369;
    case 9:
        for (double v : x) s += v * v - 10 * dm::cos(2 * PI * v) + 10;
        return s;
    case 10: {
        double c = 0.0;
        for (double v : x) {
            s += v * v;
            c += dm::cos(2 * PI * v);
        }
        return -20 * dm::exp(-0.2 * std::sqrt(s / n)) - dm::exp(c / n) + 20 + dm::exp(1.0);
    }
    case 11:
        for (std::size_t i = 0; i < D; ++i) {
            s += x[i] * x[i];
            p *= dm::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
        }
        return s / 4000 - p + 1;
    case 12: {
        auto y = [&](std::size_t i) { return 1 + (x[i] + 1) / 4; };
        s = 10 * sin2(PI * y(0));
        for (std::size_t i = 0; i + 1 < D; ++i) s += (y(i) - 1) * (y(i) - 1) * (1 + 10 * sin2(PI * y(i + 1)));
        s += (y(D - 1) - 1) * (y(D - 1) - 1);
        double pen = 0.0;
        for (double v : x) pen += u(v, 10, 100, 4);
        return PI / n * s + pen;
    }
    case 13: {
        s = sin2(3 * PI * x[0]);
        for (std::size_t i = 0; i + 1 < D; ++i) s += (x[i] - 1) * (x[i] - 1) * (1 + sin2(3 * PI * x[i + 1]));
        s += (x[D - 1] - 1) * (x[D - 1] - 1) * (1 + sin2(2 * PI * x[D - 1]));
        double pen = 0.0;
        for (double v : x) pen += u(v, 5, 100, 4);
        return 0.1 * s + pen;
    }
    }
    return 0.0;
}

}   // namespace yao1999

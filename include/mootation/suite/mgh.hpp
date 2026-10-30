#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The 35 test functions of J.J. Moré, B.S. Garbow, K.E. Hillstrom, "Testing
// unconstrained optimization software", ACM TOMS 7(1) 1981, pp. 17-41 (source
// mgh-test-problems_more1981; "MGH"), §3: each as m residuals f_i(x) of n
// variables with their Jacobian (MGH's SSQFCN and SSQJAC), the standard
// starting point x_s and INITPT's FACTOR * x_s (§2: all components FACTOR
// where x_s = 0 and FACTOR != 1). Least squares: min sum f_i^2 (MGH (3.1),
// the function the minima of (d) are given for); unconstrained minimization:
// the same sum, its gradient 2 J^T f.
//
// The numbers are MGH's, written as functions, not taken from their Fortran.
// Readings:
//   MGH-1. Gulf (11): the printed |y_i mi x_2| is read |y_i - x_2|, the one
//     reading under which (d) f = 0 at (50, 25, 1.5) holds.
//   MGH-2. Helical valley (7): theta is not defined at x_1 = 0; we take its
//     limit from x_1 > 0, sign(x_2) / 4 (0 where x_2 = 0 too).
//   MGH-3. Brown and Dennis (16): x_s = (25, 5, -5, -1) as MGH print it
//     (Moré 1978, §8, prints (25, 5, -5, 1) for the same function).
//   MGH-4. The sizes MGH leave open take the caller's m (and n where the
//     function is "n variable").
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "../fp_contract.hpp"

namespace mootation::suite::mgh {

using Vec = std::vector<double>;

inline constexpr int kCount = 35;

inline const char* name(int id)
{
    static const char* names[kCount] = {
        "rosenbrock", "freudenstein_roth", "powell_badly_scaled", "brown_badly_scaled", "beale",
        "jennrich_sampson", "helical_valley", "bard", "gaussian", "meyer", "gulf", "box_3d",
        "powell_singular", "wood", "kowalik_osborne", "brown_dennis", "osborne_1", "biggs_exp6",
        "osborne_2", "watson", "extended_rosenbrock", "extended_powell_singular", "penalty_1",
        "penalty_2", "variably_dimensioned", "trigonometric", "brown_almost_linear",
        "discrete_boundary_value", "discrete_integral_equation", "broyden_tridiagonal", "broyden_banded",
        "linear_full_rank", "linear_rank_1", "linear_rank_1_zero", "chebyquad"};
    if (id < 1 || id > kCount) throw std::invalid_argument("mgh: problems 1 to 35");
    return names[id - 1];
}

// n and m where MGH fix them (0: the caller's, MGH-4)
inline std::size_t fixed_n(int id)
{
    static const std::size_t n[kCount] = {2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 11, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return n[id - 1];
}

inline std::size_t fixed_m(int id, std::size_t n)
{
    static const std::size_t m[kCount] = {2, 2, 2, 3, 3, 0, 3, 15, 15, 16, 0, 0, 4, 6, 11, 0, 33, 0, 65, 31,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (m[id - 1]) return m[id - 1];
    switch (id) {
    case 21: case 22: case 26: case 27: case 28: case 29: case 30: case 31: return n;
    case 23: return n + 1;
    case 24: return 2 * n;
    case 25: return n + 2;
    default: return 0;
    }
}

namespace detail {

inline const double* bard_y()
{
    static const double y[15] = {0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
                                 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39};
    return y;
}
inline double gaussian_y(int i)                      // i = 1..15, symmetric about 8
{
    static const double y[8] = {0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989};
    return y[i <= 8 ? i - 1 : 15 - i];
}
inline const double* meyer_y()
{
    static const double y[16] = {34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744,
                                 8261,  7030,  6005,  5147,  4427,  3820,  3307,  2872};
    return y;
}
inline const double* kowalik_y()
{
    static const double y[11] = {0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                                 0.0456, 0.0342, 0.0323, 0.0235, 0.0246};
    return y;
}
inline const double* kowalik_u()
{
    static const double u[11] = {4.0000, 2.0000, 1.0000, 0.5000, 0.2500, 0.1670,
                                 0.1250, 0.1000, 0.0833, 0.0714, 0.0625};
    return u;
}
inline const double* osborne1_y()
{
    static const double y[33] = {0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751,
                                 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490,
                                 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406};
    return y;
}
inline const double* osborne2_y()
{
    static const double y[65] = {
        1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608,
        0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661,
        0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428,
        0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559,
        0.597, 0.625, 0.739, 0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054};
    return y;
}

inline constexpr double kPi = 3.141592653589793;

inline double theta(double x1, double x2)            // helical valley (MGH-2)
{
    if (x1 > 0.0) return std::atan(x2 / x1) / (2.0 * kPi);
    if (x1 < 0.0) return std::atan(x2 / x1) / (2.0 * kPi) + 0.5;
    return x2 > 0.0 ? 0.25 : x2 < 0.0 ? -0.25 : 0.0;
}

// the shifted Chebyshev polynomials T_k(x) = T_k(2x - 1), k = 0..K, and their derivatives
inline void chebyshev(double x, std::size_t K, Vec& T, Vec& dT)
{
    const double y = 2.0 * x - 1.0;
    T.assign(K + 1, 0.0);
    dT.assign(K + 1, 0.0);
    T[0] = 1.0;
    if (K >= 1) {
        T[1] = y;
        dT[1] = 2.0;
    }
    for (std::size_t k = 1; k < K; ++k) {
        T[k + 1] = 2.0 * y * T[k] - T[k - 1];
        dT[k + 1] = 4.0 * T[k] + 2.0 * y * dT[k] - dT[k - 1];
    }
}

}   // namespace detail

// The standard starting point x_s (n variables)
inline Vec standard_start(int id, std::size_t n)
{
    switch (id) {
    case 1: return {-1.2, 1.0};
    case 2: return {0.5, -2.0};
    case 3: return {0.0, 1.0};
    case 4: return {1.0, 1.0};
    case 5: return {1.0, 1.0};
    case 6: return {0.3, 0.4};
    case 7: return {-1.0, 0.0, 0.0};
    case 8: return {1.0, 1.0, 1.0};
    case 9: return {0.4, 1.0, 0.0};
    case 10: return {0.02, 4000.0, 250.0};
    case 11: return {5.0, 2.5, 0.15};
    case 12: return {0.0, 10.0, 20.0};
    case 13: return {3.0, -1.0, 0.0, 1.0};
    case 14: return {-3.0, -1.0, -3.0, -1.0};
    case 15: return {0.25, 0.39, 0.415, 0.39};
    case 16: return {25.0, 5.0, -5.0, -1.0};   // MGH-3
    case 17: return {0.5, 1.5, -1.0, 0.01, 0.02};
    case 18: return {1.0, 2.0, 1.0, 1.0, 1.0, 1.0};
    case 19: return {1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5};
    default: break;
    }
    Vec x(n, 0.0);
    const double h = 1.0 / static_cast<double>(n + 1);
    for (std::size_t j = 0; j < n; ++j) {
        const double jj = static_cast<double>(j + 1), t = jj * h;
        switch (id) {
        case 20: x[j] = 0.0; break;
        case 21: x[j] = j % 2 == 0 ? -1.2 : 1.0; break;
        case 22: x[j] = j % 4 == 0 ? 3.0 : j % 4 == 1 ? -1.0 : j % 4 == 2 ? 0.0 : 1.0; break;
        case 23: x[j] = jj; break;
        case 24: x[j] = 0.5; break;
        case 25: x[j] = 1.0 - jj / static_cast<double>(n); break;
        case 26: x[j] = 1.0 / static_cast<double>(n); break;
        case 27: x[j] = 0.5; break;
        case 28: case 29: x[j] = t * (t - 1.0); break;
        case 30: case 31: x[j] = -1.0; break;
        case 32: case 33: case 34: x[j] = 1.0; break;
        case 35: x[j] = t; break;
        default: throw std::invalid_argument("mgh: problems 1 to 35");
        }
    }
    return x;
}

// INITPT (MGH §2)
inline Vec start(int id, std::size_t n, double factor)
{
    Vec x = standard_start(id, n);
    bool zero = true;
    for (double v : x) zero = zero && v == 0.0;
    for (double& v : x) v = zero && factor != 1.0 ? factor : factor * v;
    return x;
}

// f (m entries) and, where J is not null, the Jacobian row by row (m x n)
inline void evaluate(int id, std::size_t n, std::size_t m, const Vec& x, Vec& f, Vec* Jp = nullptr)
{
    f.assign(m, 0.0);
    Vec dummy;
    Vec& J = Jp ? *Jp : dummy;
    if (Jp) J.assign(m * n, 0.0);
    auto at = [&](std::size_t i, std::size_t j) -> double& { return J[i * n + j]; };
    const bool jac = Jp != nullptr;
    const double dn = static_cast<double>(n), dm = static_cast<double>(m);
    switch (id) {
    case 1:
        f[0] = 10.0 * (x[1] - x[0] * x[0]);
        f[1] = 1.0 - x[0];
        if (jac) { at(0, 0) = -20.0 * x[0]; at(0, 1) = 10.0; at(1, 0) = -1.0; }
        break;
    case 2:
        f[0] = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1];
        f[1] = -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1];
        if (jac) {
            at(0, 0) = 1.0; at(0, 1) = (10.0 - 3.0 * x[1]) * x[1] - 2.0;
            at(1, 0) = 1.0; at(1, 1) = (3.0 * x[1] + 2.0) * x[1] - 14.0;
        }
        break;
    case 3:
        f[0] = 1e4 * x[0] * x[1] - 1.0;
        f[1] = std::exp(-x[0]) + std::exp(-x[1]) - 1.0001;
        if (jac) {
            at(0, 0) = 1e4 * x[1]; at(0, 1) = 1e4 * x[0];
            at(1, 0) = -std::exp(-x[0]); at(1, 1) = -std::exp(-x[1]);
        }
        break;
    case 4:
        f[0] = x[0] - 1e6;
        f[1] = x[1] - 2e-6;
        f[2] = x[0] * x[1] - 2.0;
        if (jac) { at(0, 0) = 1.0; at(1, 1) = 1.0; at(2, 0) = x[1]; at(2, 1) = x[0]; }
        break;
    case 5: {
        const double y[3] = {1.5, 2.25, 2.625};
        for (std::size_t i = 0; i < 3; ++i) {
            const double p = std::pow(x[1], static_cast<double>(i + 1));
            f[i] = y[i] - x[0] * (1.0 - p);
            if (jac) {
                at(i, 0) = -(1.0 - p);
                at(i, 1) = x[0] * static_cast<double>(i + 1) * std::pow(x[1], static_cast<double>(i));
            }
        }
        break;
    }
    case 6:
        for (std::size_t i = 0; i < m; ++i) {
            const double ii = static_cast<double>(i + 1), e1 = std::exp(ii * x[0]), e2 = std::exp(ii * x[1]);
            f[i] = 2.0 + 2.0 * ii - (e1 + e2);
            if (jac) { at(i, 0) = -ii * e1; at(i, 1) = -ii * e2; }
        }
        break;
    case 7: {
        const double r2 = x[0] * x[0] + x[1] * x[1], r = std::sqrt(r2);
        f[0] = 10.0 * (x[2] - 10.0 * detail::theta(x[0], x[1]));
        f[1] = 10.0 * (r - 1.0);
        f[2] = x[2];
        if (jac) {
            const double c = 100.0 / (2.0 * detail::kPi * r2);
            at(0, 0) = c * x[1]; at(0, 1) = -c * x[0]; at(0, 2) = 10.0;
            at(1, 0) = 10.0 * x[0] / r; at(1, 1) = 10.0 * x[1] / r;
            at(2, 2) = 1.0;
        }
        break;
    }
    case 8:
        for (std::size_t i = 0; i < 15; ++i) {
            const double u = static_cast<double>(i + 1), v = 16.0 - u, w = std::min(u, v);
            const double d = v * x[1] + w * x[2];
            f[i] = detail::bard_y()[i] - (x[0] + u / d);
            if (jac) { at(i, 0) = -1.0; at(i, 1) = u * v / (d * d); at(i, 2) = u * w / (d * d); }
        }
        break;
    case 9:
        for (std::size_t i = 0; i < 15; ++i) {
            const double t = (8.0 - static_cast<double>(i + 1)) / 2.0, d = t - x[2];
            const double e = std::exp(-x[1] * d * d / 2.0);
            f[i] = x[0] * e - detail::gaussian_y(static_cast<int>(i + 1));
            if (jac) { at(i, 0) = e; at(i, 1) = -x[0] * e * d * d / 2.0; at(i, 2) = x[0] * e * x[1] * d; }
        }
        break;
    case 10:
        for (std::size_t i = 0; i < 16; ++i) {
            const double t = 45.0 + 5.0 * static_cast<double>(i + 1), d = t + x[2];
            const double e = std::exp(x[1] / d);
            f[i] = x[0] * e - detail::meyer_y()[i];
            if (jac) { at(i, 0) = e; at(i, 1) = x[0] * e / d; at(i, 2) = -x[0] * e * x[1] / (d * d); }
        }
        break;
    case 11:                                         // MGH-1
        for (std::size_t i = 0; i < m; ++i) {
            const double t = static_cast<double>(i + 1) / 100.0;
            const double y = 25.0 + std::pow(-50.0 * std::log(t), 2.0 / 3.0);
            const double a = std::fabs(y - x[1]), p = std::pow(a, x[2]), e = std::exp(-p / x[0]);
            f[i] = e - t;
            if (jac) {
                const double sg = y - x[1] > 0.0 ? 1.0 : y - x[1] < 0.0 ? -1.0 : 0.0;
                at(i, 0) = e * p / (x[0] * x[0]);
                at(i, 1) = a > 0.0 ? e * x[2] * std::pow(a, x[2] - 1.0) * sg / x[0] : 0.0;
                at(i, 2) = a > 0.0 ? -e * p * std::log(a) / x[0] : 0.0;
            }
        }
        break;
    case 12:
        for (std::size_t i = 0; i < m; ++i) {
            const double t = 0.1 * static_cast<double>(i + 1);
            const double e1 = std::exp(-t * x[0]), e2 = std::exp(-t * x[1]), c = std::exp(-t) - std::exp(-10.0 * t);
            f[i] = e1 - e2 - x[2] * c;
            if (jac) { at(i, 0) = -t * e1; at(i, 1) = t * e2; at(i, 2) = -c; }
        }
        break;
    case 13: case 22:
        for (std::size_t b = 0; b + 3 < n; b += 4) {
            const double s5 = std::sqrt(5.0), s10 = std::sqrt(10.0);
            const double a = x[b + 1] - 2.0 * x[b + 2], c = x[b] - x[b + 3];
            f[b] = x[b] + 10.0 * x[b + 1];
            f[b + 1] = s5 * (x[b + 2] - x[b + 3]);
            f[b + 2] = a * a;
            f[b + 3] = s10 * c * c;
            if (jac) {
                at(b, b) = 1.0; at(b, b + 1) = 10.0;
                at(b + 1, b + 2) = s5; at(b + 1, b + 3) = -s5;
                at(b + 2, b + 1) = 2.0 * a; at(b + 2, b + 2) = -4.0 * a;
                at(b + 3, b) = 2.0 * s10 * c; at(b + 3, b + 3) = -2.0 * s10 * c;
            }
        }
        break;
    case 14: {
        const double s90 = std::sqrt(90.0), s10 = std::sqrt(10.0);
        f[0] = 10.0 * (x[1] - x[0] * x[0]);
        f[1] = 1.0 - x[0];
        f[2] = s90 * (x[3] - x[2] * x[2]);
        f[3] = 1.0 - x[2];
        f[4] = s10 * (x[1] + x[3] - 2.0);
        f[5] = (x[1] - x[3]) / s10;
        if (jac) {
            at(0, 0) = -20.0 * x[0]; at(0, 1) = 10.0;
            at(1, 0) = -1.0;
            at(2, 2) = -2.0 * s90 * x[2]; at(2, 3) = s90;
            at(3, 2) = -1.0;
            at(4, 1) = s10; at(4, 3) = s10;
            at(5, 1) = 1.0 / s10; at(5, 3) = -1.0 / s10;
        }
        break;
    }
    case 15:
        for (std::size_t i = 0; i < 11; ++i) {
            const double u = detail::kowalik_u()[i];
            const double num = u * u + u * x[1], den = u * u + u * x[2] + x[3];
            f[i] = detail::kowalik_y()[i] - x[0] * num / den;
            if (jac) {
                at(i, 0) = -num / den; at(i, 1) = -x[0] * u / den;
                at(i, 2) = x[0] * num * u / (den * den); at(i, 3) = x[0] * num / (den * den);
            }
        }
        break;
    case 16:
        for (std::size_t i = 0; i < m; ++i) {
            const double t = static_cast<double>(i + 1) / 5.0;
            const double a = x[0] + t * x[1] - std::exp(t), b = x[2] + x[3] * std::sin(t) - std::cos(t);
            f[i] = a * a + b * b;
            if (jac) { at(i, 0) = 2.0 * a; at(i, 1) = 2.0 * a * t; at(i, 2) = 2.0 * b; at(i, 3) = 2.0 * b * std::sin(t); }
        }
        break;
    case 17:
        for (std::size_t i = 0; i < 33; ++i) {
            const double t = 10.0 * static_cast<double>(i), e4 = std::exp(-t * x[3]), e5 = std::exp(-t * x[4]);
            f[i] = detail::osborne1_y()[i] - (x[0] + x[1] * e4 + x[2] * e5);
            if (jac) { at(i, 0) = -1.0; at(i, 1) = -e4; at(i, 2) = -e5; at(i, 3) = x[1] * t * e4; at(i, 4) = x[2] * t * e5; }
        }
        break;
    case 18:
        for (std::size_t i = 0; i < m; ++i) {
            const double t = 0.1 * static_cast<double>(i + 1);
            const double y = std::exp(-t) - 5.0 * std::exp(-10.0 * t) + 3.0 * std::exp(-4.0 * t);
            const double e1 = std::exp(-t * x[0]), e2 = std::exp(-t * x[1]), e5 = std::exp(-t * x[4]);
            f[i] = x[2] * e1 - x[3] * e2 + x[5] * e5 - y;
            if (jac) {
                at(i, 0) = -t * x[2] * e1; at(i, 1) = t * x[3] * e2; at(i, 2) = e1;
                at(i, 3) = -e2; at(i, 4) = -t * x[5] * e5; at(i, 5) = e5;
            }
        }
        break;
    case 19:
        for (std::size_t i = 0; i < 65; ++i) {
            const double t = static_cast<double>(i) / 10.0;
            const double d9 = t - x[8], d10 = t - x[9], d11 = t - x[10];
            const double e1 = std::exp(-t * x[4]), e2 = std::exp(-d9 * d9 * x[5]), e3 = std::exp(-d10 * d10 * x[6]),
                         e4 = std::exp(-d11 * d11 * x[7]);
            f[i] = detail::osborne2_y()[i] - (x[0] * e1 + x[1] * e2 + x[2] * e3 + x[3] * e4);
            if (jac) {
                at(i, 0) = -e1; at(i, 1) = -e2; at(i, 2) = -e3; at(i, 3) = -e4;
                at(i, 4) = x[0] * t * e1;
                at(i, 5) = x[1] * d9 * d9 * e2; at(i, 6) = x[2] * d10 * d10 * e3; at(i, 7) = x[3] * d11 * d11 * e4;
                at(i, 8) = -2.0 * x[1] * x[5] * d9 * e2; at(i, 9) = -2.0 * x[2] * x[6] * d10 * e3;
                at(i, 10) = -2.0 * x[3] * x[7] * d11 * e4;
            }
        }
        break;
    case 20:
        for (std::size_t i = 0; i < 29; ++i) {
            const double t = static_cast<double>(i + 1) / 29.0;
            double s1 = 0.0, s2 = 0.0, p = 1.0;          // p = t^(j-1)
            for (std::size_t j = 0; j < n; ++j) {
                if (j >= 1) s1 += static_cast<double>(j) * x[j] * (p / t);
                s2 += x[j] * p;
                p *= t;
            }
            f[i] = s1 - s2 * s2 - 1.0;
            if (jac) {
                p = 1.0;
                for (std::size_t j = 0; j < n; ++j) {
                    at(i, j) = (j >= 1 ? static_cast<double>(j) * (p / t) : 0.0) - 2.0 * s2 * p;
                    p *= t;
                }
            }
        }
        f[29] = x[0];
        f[30] = x[1] - x[0] * x[0] - 1.0;
        if (jac) { at(29, 0) = 1.0; at(30, 0) = -2.0 * x[0]; at(30, 1) = 1.0; }
        break;
    case 21:
        for (std::size_t b = 0; b + 1 < n; b += 2) {
            f[b] = 10.0 * (x[b + 1] - x[b] * x[b]);
            f[b + 1] = 1.0 - x[b];
            if (jac) { at(b, b) = -20.0 * x[b]; at(b, b + 1) = 10.0; at(b + 1, b) = -1.0; }
        }
        break;
    case 23: {
        const double sa = std::sqrt(1e-5);
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            f[j] = sa * (x[j] - 1.0);
            s += x[j] * x[j];
            if (jac) { at(j, j) = sa; at(n, j) = 2.0 * x[j]; }
        }
        f[n] = s - 0.25;
        break;
    }
    case 24: {
        const double sa = std::sqrt(1e-5), e1 = std::exp(-0.1);
        f[0] = x[0] - 0.2;
        if (jac) at(0, 0) = 1.0;
        for (std::size_t i = 1; i < n; ++i) {
            const double ii = static_cast<double>(i + 1);
            const double y = std::exp(ii / 10.0) + std::exp((ii - 1.0) / 10.0);
            const double a = std::exp(x[i] / 10.0), b = std::exp(x[i - 1] / 10.0);
            f[i] = sa * (a + b - y);
            if (jac) { at(i, i) = sa * a / 10.0; at(i, i - 1) = sa * b / 10.0; }
        }
        for (std::size_t i = n; i + 1 < 2 * n; ++i) {   // i (1-based i + 1) in (n, 2n): x_{i - n + 1}
            const std::size_t j = i - n + 1;
            const double a = std::exp(x[j] / 10.0);
            f[i] = sa * (a - e1);
            if (jac) at(i, j) = sa * a / 10.0;
        }
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            s += (dn - static_cast<double>(j)) * x[j] * x[j];
            if (jac) at(2 * n - 1, j) = 2.0 * (dn - static_cast<double>(j)) * x[j];
        }
        f[2 * n - 1] = s - 1.0;
        break;
    }
    case 25: {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            f[j] = x[j] - 1.0;
            s += static_cast<double>(j + 1) * (x[j] - 1.0);
            if (jac) at(j, j) = 1.0;
        }
        f[n] = s;
        f[n + 1] = s * s;
        if (jac)
            for (std::size_t j = 0; j < n; ++j) {
                at(n, j) = static_cast<double>(j + 1);
                at(n + 1, j) = 2.0 * s * static_cast<double>(j + 1);
            }
        break;
    }
    case 26: {
        double c = 0.0;
        for (std::size_t j = 0; j < n; ++j) c += std::cos(x[j]);
        for (std::size_t i = 0; i < n; ++i) {
            const double ii = static_cast<double>(i + 1);
            f[i] = dn - c + ii * (1.0 - std::cos(x[i])) - std::sin(x[i]);
            if (jac) {
                for (std::size_t j = 0; j < n; ++j) at(i, j) = std::sin(x[j]);
                at(i, i) += ii * std::sin(x[i]) - std::cos(x[i]);
            }
        }
        break;
    }
    case 27: {
        double s = 0.0, p = 1.0;
        for (std::size_t j = 0; j < n; ++j) {
            s += x[j];
            p *= x[j];
        }
        for (std::size_t i = 0; i + 1 < n; ++i) {
            f[i] = x[i] + s - (dn + 1.0);
            if (jac) {
                for (std::size_t j = 0; j < n; ++j) at(i, j) = 1.0;
                at(i, i) = 2.0;
            }
        }
        f[n - 1] = p - 1.0;
        if (jac)
            for (std::size_t j = 0; j < n; ++j) {
                double q = 1.0;
                for (std::size_t k = 0; k < n; ++k)
                    if (k != j) q *= x[k];
                at(n - 1, j) = q;
            }
        break;
    }
    case 28: {
        const double h = 1.0 / (dn + 1.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i + 1) * h, u = x[i] + t + 1.0;
            const double xm = i > 0 ? x[i - 1] : 0.0, xp = i + 1 < n ? x[i + 1] : 0.0;
            f[i] = 2.0 * x[i] - xm - xp + h * h * u * u * u / 2.0;
            if (jac) {
                at(i, i) = 2.0 + 1.5 * h * h * u * u;
                if (i > 0) at(i, i - 1) = -1.0;
                if (i + 1 < n) at(i, i + 1) = -1.0;
            }
        }
        break;
    }
    case 29: {
        const double h = 1.0 / (dn + 1.0);
        Vec t(n), u3(n), du3(n);
        for (std::size_t j = 0; j < n; ++j) {
            t[j] = static_cast<double>(j + 1) * h;
            const double u = x[j] + t[j] + 1.0;
            u3[j] = u * u * u;
            du3[j] = 3.0 * u * u;
        }
        for (std::size_t i = 0; i < n; ++i) {
            double a = 0.0, b = 0.0;
            for (std::size_t j = 0; j <= i; ++j) a += t[j] * u3[j];
            for (std::size_t j = i + 1; j < n; ++j) b += (1.0 - t[j]) * u3[j];
            f[i] = x[i] + h * ((1.0 - t[i]) * a + t[i] * b) / 2.0;
            if (jac)
                for (std::size_t j = 0; j < n; ++j) {
                    const double w = j <= i ? (1.0 - t[i]) * t[j] : t[i] * (1.0 - t[j]);
                    at(i, j) = (i == j ? 1.0 : 0.0) + h * w * du3[j] / 2.0;
                }
        }
        break;
    }
    case 30:
        for (std::size_t i = 0; i < n; ++i) {
            const double xm = i > 0 ? x[i - 1] : 0.0, xp = i + 1 < n ? x[i + 1] : 0.0;
            f[i] = (3.0 - 2.0 * x[i]) * x[i] - xm - 2.0 * xp + 1.0;
            if (jac) {
                at(i, i) = 3.0 - 4.0 * x[i];
                if (i > 0) at(i, i - 1) = -1.0;
                if (i + 1 < n) at(i, i + 1) = -2.0;
            }
        }
        break;
    case 31:
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t lo = i >= 5 ? i - 5 : 0, hi = std::min(n - 1, i + 1);
            double s = 0.0;
            for (std::size_t j = lo; j <= hi; ++j)
                if (j != i) {
                    s += x[j] * (1.0 + x[j]);
                    if (jac) at(i, j) = -(1.0 + 2.0 * x[j]);
                }
            f[i] = x[i] * (2.0 + 5.0 * x[i] * x[i]) + 1.0 - s;
            if (jac) at(i, i) = 2.0 + 15.0 * x[i] * x[i];
        }
        break;
    case 32: {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) s += x[j];
        for (std::size_t i = 0; i < m; ++i) {
            f[i] = (i < n ? x[i] : 0.0) - 2.0 / dm * s - 1.0;
            if (jac)
                for (std::size_t j = 0; j < n; ++j) at(i, j) = (i == j ? 1.0 : 0.0) - 2.0 / dm;
        }
        break;
    }
    case 33: {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) s += static_cast<double>(j + 1) * x[j];
        for (std::size_t i = 0; i < m; ++i) {
            f[i] = static_cast<double>(i + 1) * s - 1.0;
            if (jac)
                for (std::size_t j = 0; j < n; ++j) at(i, j) = static_cast<double>((i + 1) * (j + 1));
        }
        break;
    }
    case 34: {
        double s = 0.0;
        for (std::size_t j = 1; j + 1 < n; ++j) s += static_cast<double>(j + 1) * x[j];
        f[0] = -1.0;
        f[m - 1] = -1.0;
        for (std::size_t i = 1; i + 1 < m; ++i) {
            f[i] = static_cast<double>(i) * s - 1.0;
            if (jac)
                for (std::size_t j = 1; j + 1 < n; ++j) at(i, j) = static_cast<double>(i * (j + 1));
        }
        break;
    }
    case 35: {
        Vec T, dT;
        for (std::size_t j = 0; j < n; ++j) {
            detail::chebyshev(x[j], m, T, dT);
            for (std::size_t i = 0; i < m; ++i) {
                f[i] += T[i + 1] / dn;
                if (jac) at(i, j) = dT[i + 1] / dn;
            }
        }
        for (std::size_t i = 0; i < m; ++i) {
            const double k = static_cast<double>(i + 1);
            if ((i + 1) % 2 == 0) f[i] += 1.0 / (k * k - 1.0);   // - integral = + 1 / (k^2 - 1)
        }
        break;
    }
    default:
        throw std::invalid_argument("mgh: problems 1 to 35");
    }
}

}   // namespace mootation::suite::mgh

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The CEC 2013 niching benchmark (task 18, stage 3, §3.5): X. Li, A.
// Engelbrecht, M.G. Epitropakis, "Benchmark Functions for CEC'2013 Special
// Session and Competition on Niching Methods for Multimodal Function
// Optimization", technical report, 2013 (source niching-benchmark-
// cec2013_li2013, L13), §II and Tables I and IV: twelve functions, twenty
// problems (the competition's numbering 1 ... 20, Table IV's order):
//
//    1 F1 Five-Uneven-Peak Trap, D 1    11 F9  CF1, D 2
//    2 F2 Equal Maxima, D 1             12 F10 CF2, D 2
//    3 F3 Uneven Decreasing Maxima, D 1 13 F11 CF3, D 2
//    4 F4 Himmelblau, D 2               14 F11 CF3, D 3
//    5 F5 Six-Hump Camel Back, D 2      15 F12 CF4, D 3
//    6 F6 Shubert, D 2                  16 F11 CF3, D 5
//    7 F7 Vincent, D 2                  17 F12 CF4, D 5
//    8 F6 Shubert, D 3                  18 F11 CF3, D 10
//    9 F7 Vincent, D 3                  19 F12 CF4, D 10
//   10 F8 Modified Rastrigin, D 2       20 F12 CF4, D 20
//
// value(x) is the report's F, to be maximized; the library minimizes -F. Each
// problem carries Table IV's r, peak height and number of global optima and
// Table I's MaxFEs, for the peak ratio (so_metrics.hpp, peaks_found).
//
// The composition functions (§II-I): CF(x) = sum_i w_i C f_i(z_i) /
// |f_max^i|, z_i = ((x - o_i) / λ_i) M_i (a row vector times M_i), f_max^i =
// f_i((x* / λ_i) M_i) with x* = (5, ..., 5), C = 2000, bias 0; w_i = exp(-|x -
// o_i|^2 / (2 D σ_i^2)), every w_i but the largest times (1 - max w^10), then
// normalized. Weierstrass α = 0.5, β = 3, k_max = 20; EF8F2 the 1-D Griewank
// of Rosenbrock's term of (z_i, z_i+1), round to (z_D, z_1).
//
// DEVIATIONS AND READINGS (the definitions against the report):
//   CEC-1. F5 is -[(4 - 2.1x² + x⁴/3)x² + xy + (4y² - 4)y²], without the
//     factor 4 the report prints: Table IV's height 1.03163 and the published
//     peak ratios (Tables II-III: 1 at every ε) are of this form (owner's В2).
//   CEC-2. The report gives no shifts o_i nor rotations M_i of the
//     composition functions (the competition's data files have them): ours,
//     o_i uniform in [-5, 5]^D, M_i of CF3 and CF4 Haar rotations (condition
//     number one), from std::mt19937 seeded by instances::seed({20261105,
//     problem, D}): o_1 ... o_n, then M_1 ... M_n. CF1 and CF2 use the
//     identity, as the report says. Numbers of F9-F12 are thus comparable to
//     the published ones in shape only.
//   CEC-3. Where every w_i is 0 (far from every o_i; outside the box only)
//     the weights are equal, 1/n, as the normalization 0/0 leaves them.
//   CEC-4. F8 at D = 2 has k = (3, 4) (the report's 2-D instance); its
//     formula is printed under the name F9, a misprint.
//   CEC-5. Ties of the largest weight: every w_i equal to the largest keeps
//     its value.
//   CEC-6. EF8F2 is evaluated at z + 1, as CEC 2005's expanded F8F2 (x - o +
//     1), whose code the report's is based on: its minimum is then at z = 0,
//     and CF(o_i) = 0, as the report states; at z itself it would be at (1,
//     ..., 1) and no optimum of CF3 or CF4 would be at o_i.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../detmath.hpp"
#include "../random.hpp"
#include "instances.hpp"

namespace mootation::suite::cec2013 {

constexpr int kProblems = 20;
constexpr double kPi = 3.14159265358979323846;

// Table IV and Table I for problem p (1 ... 20)
struct Info {
    int function = 0, D = 0;          // F1 ... F12, the dimension
    double lo = 0.0, hi = 0.0;        // the box (F5: x's; y in [-1.1, 1.1])
    double r = 0.0, height = 0.0;     // the niche radius, the global optima's value
    int n_global = 0;
    long long max_fes = 0;
};

inline Info info(int p)
{
    struct Row {
        int f, D;
        double lo, hi, r, height;
        int n;
        long long fes;
    };
    static const Row rows[kProblems] = {
        {1, 1, 0.0, 30.0, 0.01, 200.0, 2, 50000},        {2, 1, 0.0, 1.0, 0.01, 1.0, 5, 50000},
        {3, 1, 0.0, 1.0, 0.01, 1.0, 1, 50000},           {4, 2, -6.0, 6.0, 0.01, 200.0, 4, 50000},
        {5, 2, -1.9, 1.9, 0.5, 1.03163, 2, 50000},       {6, 2, -10.0, 10.0, 0.5, 186.731, 18, 200000},
        {7, 2, 0.25, 10.0, 0.2, 1.0, 36, 200000},        {6, 3, -10.0, 10.0, 0.5, 2709.0935, 81, 400000},
        {7, 3, 0.25, 10.0, 0.2, 1.0, 216, 400000},       {8, 2, 0.0, 1.0, 0.01, -2.0, 12, 200000},
        {9, 2, -5.0, 5.0, 0.01, 0.0, 6, 200000},         {10, 2, -5.0, 5.0, 0.01, 0.0, 8, 200000},
        {11, 2, -5.0, 5.0, 0.01, 0.0, 6, 200000},        {11, 3, -5.0, 5.0, 0.01, 0.0, 6, 400000},
        {12, 3, -5.0, 5.0, 0.01, 0.0, 8, 400000},        {11, 5, -5.0, 5.0, 0.01, 0.0, 6, 400000},
        {12, 5, -5.0, 5.0, 0.01, 0.0, 8, 400000},        {11, 10, -5.0, 5.0, 0.01, 0.0, 6, 400000},
        {12, 10, -5.0, 5.0, 0.01, 0.0, 8, 400000},       {12, 20, -5.0, 5.0, 0.01, 0.0, 8, 400000}};
    if (p < 1 || p > kProblems) throw std::invalid_argument("cec2013: problem 1 ... 20, got " + std::to_string(p));
    const Row& r = rows[p - 1];
    return {r.f, r.D, r.lo, r.hi, r.r, r.height, r.n, r.fes};
}

// The global optima's value F to full precision, and whether it is exact
// (the others: Nelder-Mead from the optima to 1e-15, test_cec2013; Table IV
// prints 0.99999983 as 1.0, 1.0316284535 as 1.03163, ...)
struct Best {
    double value = 0.0;
    bool exact = true;
};

inline Best best(int p)
{
    switch (p) {
    case 3: return {0.99999982845447266, false};
    case 5: return {1.0316284534898774, false};
    case 6: return {186.73090883102387, false};
    case 8: return {2709.0935055728273, false};
    default: return {info(p).height, true};
    }
}

inline std::vector<double> lower(int p)
{
    const Info i = info(p);
    std::vector<double> v(static_cast<std::size_t>(i.D), i.lo);
    if (i.function == 5) v[1] = -1.1;
    return v;
}

inline std::vector<double> upper(int p)
{
    const Info i = info(p);
    std::vector<double> v(static_cast<std::size_t>(i.D), i.hi);
    if (i.function == 5) v[1] = 1.1;
    return v;
}

namespace detail {

inline double sin6(double t)
{
    const double s = dm::sin(t), s2 = s * s;
    return s2 * s2 * s2;
}

// the basic functions of the compositions (§II-I), at z (D values)
inline double sphere(const std::vector<double>& z)
{
    double s = 0.0;
    for (double v : z) s += v * v;
    return s;
}

inline double griewank(const std::vector<double>& z)
{
    double s = 0.0, p = 1.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        s += z[i] * z[i] / 4000.0;
        p *= dm::cos(z[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    return s - p + 1.0;
}

inline double rastrigin(const std::vector<double>& z)
{
    double s = 0.0;
    for (double v : z) s += v * v - 10.0 * dm::cos(2.0 * kPi * v) + 10.0;
    return s;
}

inline double weierstrass(const std::vector<double>& z)
{
    double s = 0.0, c = 0.0, a = 1.0, b = 1.0;
    for (int k = 0; k <= 20; ++k) {                                   // α = 0.5, β = 3
        for (double v : z) s += a * dm::cos(2.0 * kPi * b * (v + 0.5));
        c += a * dm::cos(2.0 * kPi * b * 0.5);
        a *= 0.5;
        b *= 3.0;
    }
    return s - static_cast<double>(z.size()) * c;
}

inline double ef8f2(const std::vector<double>& z)
{
    const std::size_t D = z.size();
    double s = 0.0;
    for (std::size_t i = 0; i < D; ++i) {
        const double a = z[i] + 1.0, b = z[(i + 1) % D] + 1.0;                   // CEC-6
        const double f2 = 100.0 * (a * a - b) * (a * a - b) + (a - 1.0) * (a - 1.0);
        s += f2 * f2 / 4000.0 - dm::cos(f2) + 1.0;
    }
    return s;
}

enum class Basic { sphere, griewank, rastrigin, weierstrass, ef8f2 };

inline double basic(Basic b, const std::vector<double>& z)
{
    switch (b) {
    case Basic::sphere: return sphere(z);
    case Basic::griewank: return griewank(z);
    case Basic::rastrigin: return rastrigin(z);
    case Basic::weierstrass: return weierstrass(z);
    case Basic::ef8f2: return ef8f2(z);
    }
    return 0.0;
}

}   // namespace detail

// One problem, its composition's draws made (CEC-2).
class Problem {
public:
    Problem() = default;

    explicit Problem(int p) : info_(info(p))
    {
        const int f = info_.function;
        if (f < 9) return;
        using B = detail::Basic;
        if (f == 9) {
            basics_ = {B::griewank, B::griewank, B::weierstrass, B::weierstrass, B::sphere, B::sphere};
            sigma_ = {1, 1, 1, 1, 1, 1};
            lambda_ = {1, 1, 8, 8, 1.0 / 5.0, 1.0 / 5.0};
        } else if (f == 10) {
            basics_ = {B::rastrigin, B::rastrigin, B::weierstrass, B::weierstrass,
                       B::griewank,  B::griewank,  B::sphere,      B::sphere};
            sigma_ = {1, 1, 1, 1, 1, 1, 1, 1};
            lambda_ = {1, 1, 10, 10, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 7.0, 1.0 / 7.0};
        } else if (f == 11) {
            basics_ = {B::ef8f2, B::ef8f2, B::weierstrass, B::weierstrass, B::griewank, B::griewank};
            sigma_ = {1, 1, 2, 2, 2, 2};
            lambda_ = {1.0 / 4.0, 1.0 / 10.0, 2, 1, 2, 5};
        } else {
            basics_ = {B::rastrigin, B::rastrigin, B::ef8f2,       B::ef8f2,
                       B::weierstrass, B::weierstrass, B::griewank, B::griewank};
            sigma_ = {1, 1, 1, 1, 1, 2, 2, 2};
            lambda_ = {4, 1, 4, 1, 1.0 / 10.0, 1.0 / 5.0, 1.0 / 10.0, 1.0 / 40.0};
        }
        const std::size_t n = basics_.size(), D = static_cast<std::size_t>(info_.D);
        std::mt19937 g(instances::seed({20261105u, static_cast<std::uint64_t>(p), static_cast<std::uint64_t>(D)}));
        o_.assign(n, std::vector<double>(D));
        for (auto& o : o_)
            for (double& v : o) v = -5.0 + 10.0 * rnd::uniform01(g);
        M_.assign(n, std::vector<double>(D * D, 0.0));
        for (auto& M : M_) {
            if (f >= 11) {
                M = instances::haar(g, info_.D);
            } else {
                for (std::size_t i = 0; i < D; ++i) M[i * D + i] = 1.0;
            }
        }
        fmax_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> y(D, 5.0 / lambda_[i]);
            fmax_[i] = detail::basic(basics_[i], times(y, i));
        }
    }

    const Info& about() const { return info_; }
    // the shifts o_i of a composition function (empty for F1 ... F8)
    const std::vector<std::vector<double>>& optima() const { return o_; }

    // the report's F at x (to be maximized)
    double value(const std::vector<double>& x) const
    {
        switch (info_.function) {
        case 1: return trap(x[0]);
        case 2: return detail::sin6(5.0 * kPi * x[0]);
        case 3: {
            const double t = (x[0] - 0.08) / 0.854;
            return dm::exp(-2.0 * dm::log(2.0) * t * t) * detail::sin6(5.0 * kPi * (dm::pow(x[0], 0.75) - 0.05));
        }
        case 4: {
            const double a = x[0] * x[0] + x[1] - 11.0, b = x[0] + x[1] * x[1] - 7.0;
            return 200.0 - a * a - b * b;
        }
        case 5: {                                                                // CEC-1
            const double u = x[0] * x[0], v = x[1] * x[1];
            return -((4.0 - 2.1 * u + u * u / 3.0) * u + x[0] * x[1] + (4.0 * v - 4.0) * v);
        }
        case 6: {
            double p = 1.0;
            for (double xi : x) {
                double s = 0.0;
                for (int j = 1; j <= 5; ++j) s += j * dm::cos((j + 1.0) * xi + j);
                p *= s;
            }
            return -p;
        }
        case 7: {
            double s = 0.0;
            for (double xi : x) s += dm::sin(10.0 * dm::log(xi));
            return s / static_cast<double>(x.size());
        }
        case 8: {                                                                // CEC-4
            static const double k[2] = {3.0, 4.0};
            double s = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) s += 10.0 + 9.0 * dm::cos(2.0 * kPi * k[i] * x[i]);
            return -s;
        }
        default: return -composition(x);
        }
    }

private:
    Info info_;
    std::vector<detail::Basic> basics_;
    std::vector<double> sigma_, lambda_, fmax_;
    std::vector<std::vector<double>> o_, M_;

    static double trap(double x)
    {
        if (x < 2.5) return 80.0 * (2.5 - x);
        if (x < 5.0) return 64.0 * (x - 2.5);
        if (x < 7.5) return 64.0 * (7.5 - x);
        if (x < 12.5) return 28.0 * (x - 7.5);
        if (x < 17.5) return 28.0 * (17.5 - x);
        if (x < 22.5) return 32.0 * (x - 17.5);
        if (x < 27.5) return 32.0 * (27.5 - x);
        return 80.0 * (x - 27.5);
    }

    // y M_i, y a row vector
    std::vector<double> times(const std::vector<double>& y, std::size_t i) const
    {
        const std::size_t D = y.size();
        std::vector<double> z(D, 0.0);
        for (std::size_t j = 0; j < D; ++j)
            for (std::size_t k = 0; k < D; ++k) z[j] += y[k] * M_[i][k * D + j];
        return z;
    }

    double composition(const std::vector<double>& x) const
    {
        const std::size_t n = basics_.size(), D = x.size();
        std::vector<double> w(n), fit(n);
        double wmax = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double d2 = 0.0;
            std::vector<double> y(D);
            for (std::size_t k = 0; k < D; ++k) {
                d2 += (x[k] - o_[i][k]) * (x[k] - o_[i][k]);
                y[k] = (x[k] - o_[i][k]) / lambda_[i];
            }
            w[i] = dm::exp(-d2 / (2.0 * static_cast<double>(D) * sigma_[i] * sigma_[i]));
            wmax = std::max(wmax, w[i]);
            fit[i] = 2000.0 * detail::basic(basics_[i], times(y, i)) / std::fabs(fmax_[i]);
        }
        double sum = 0.0;
        const double damp = 1.0 - dm::pow(wmax, 10.0);
        for (std::size_t i = 0; i < n; ++i) {
            if (w[i] != wmax) w[i] *= damp;                                      // CEC-5
            sum += w[i];
        }
        double cf = 0.0;
        for (std::size_t i = 0; i < n; ++i) cf += (sum > 0.0 ? w[i] / sum : 1.0 / static_cast<double>(n)) * fit[i];   // CEC-3
        return cf;
    }
};

}   // namespace mootation::suite::cec2013

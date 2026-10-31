#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// RCM01-18, 21-23, 25, 26, 28 and 29: the mechanical, chemical and process
// problems of RWMOP (Kumar et al., Swarm and Evolutionary Computation 67,
// 2021) as benchmarks/rwmop.py writes them, the registered ones. The order of
// every operation is that of the Python there (common.hpp): a float's ** is
// pow (dm::pow), even at 2; math's sqrt std::sqrt, exp and log dm::'s; an
// integer variable rounded half to even (Python's round); an integer's ** of
// an integer exact. The constraints are rwmop.evaluate's: the inequalities
// in the library's sign, then |h| - EQ_EPS of the equalities.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::rcm {

constexpr double kEqEps = 1e-4;                       // rwmop.EQ_EPS

struct Size {
    int n_var = 0, n_obj = 0, n_con = 0;
};

// the registered problems' sizes (rwmop.PROBLEMS, n_cons); n_var 0: not one
inline Size size(int which)
{
    switch (which) {
        case 1: return {4, 2, 2};
        case 2: return {5, 2, 5};
        case 3: return {3, 2, 3};
        case 4: return {4, 2, 4};
        case 5: return {4, 2, 4};
        case 6: return {7, 2, 11};
        case 7: return {4, 2, 1};
        case 8: return {7, 3, 10};
        case 9: return {4, 2, 0};
        case 10: return {2, 2, 2};
        case 11: return {3, 5, 7};
        case 12: return {4, 2, 1};
        case 13: return {7, 3, 11};
        case 14: return {5, 2, 8};
        case 15: return {3, 2, 8};
        case 16: return {2, 2, 2};
        case 17: return {6, 3, 9};
        case 18: return {3, 2, 3};
        case 21: return {6, 2, 4};
        case 22: return {9, 2, 6};
        case 23: return {6, 2, 5};
        case 25: return {2, 2, 2};
        case 26: return {3, 2, 2};
        case 28: return {8, 2, 9};
        case 29: return {7, 2, 9};
        default: return {};
    }
}

namespace detail {

inline double P(double x, double e) { return dm::pow(x, e); }
inline double rnd(double v) { return std::nearbyint(v) + 0.0; }    // Python's round: half to even

const double kSq2 = std::sqrt(2.0);                   // rwmop.SQ2

// RCM15's wire diameters (rwmop.WIRE)
inline const std::vector<double>& wire()
{
    static const std::vector<double> w = {
        0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018,
        0.020, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.080,
        0.092, 0.105, 0.120, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
        0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500};
    return w;
}

// _nearest: the first of the values nearest v
inline double nearest(const std::vector<double>& values, double v)
{
    double best = values[0];
    for (double w : values)
        if (std::fabs(w - v) < std::fabs(best - v)) best = w;
    return best;
}

inline double reducer_stress(double a, double x2, double x3, double c, double x6)
{
    return std::sqrt(P(745.0 * a / (x2 * x3), 2.0) + c) / (0.1 * P(x6, 3.0));
}

// RCM21's response surfaces (rwmop._C21): the constant, x1..x6, x1^2..x6^2,
// then x_i x_j for i < j in order
struct Surface {
    double c0;
    double lin[6], sq[6], cross[15];
};

inline const Surface* surfaces()
{
    static const Surface s[2] = {
        {1.3667145844797,
         {-0.00904459793976106, -0.0016193573938033, -0.00758531275221425, -0.00440727360327102,
          -0.00572216860791644, -0.00936039926190721},
         {2.62510221107328e-6, 4.92982681358861e-7, 2.25524989067108e-6, 1.84605439400301e-6,
          2.17175358243416e-6, 3.90158043948054e-6},
         {4.55276994245781e-7, -6.37013576290982e-7, 8.26736480446359e-7, 5.66352809442276e-8,
          -3.20213897443278e-7, 1.18015467772812e-8, 9.25820391546515e-8, -1.05705364119837e-7,
          -4.74797783014687e-7, -5.02319867013788e-7, 9.54284258085225e-7, 1.80533309229454e-7,
          -1.07938022118477e-6, -1.81370642220182e-7, -2.24238851688047e-7}},
        {-1.19896668942683,
         {3.04107017009774, 1.23535701600191, 2.13882039381528, 2.33495178382303, 2.68632494801975,
          3.43918953617606},
         {-7.89144544980703e-4, -2.06085185698215e-4, -7.15269900037858e-4, -7.8449237573837e-4,
          -9.31396896237177e-4, -1.40826531972195e-3},
         {-1.60434988248392e-4, 2.0824655419411e-4, -3.0530659653553e-4, -8.10145973591615e-5,
          6.94728759651311e-5, 1.18015467772812e-8, 9.25820391546515e-8, -1.05705364119837e-7,
          1.69935290196781e-4, 2.32421829190088e-5, -2.0808624041163476e-4, 1.75576341867273e-5,
          2.68422081654044e-4, 4.39852066801981e-5, 2.96785446021357e-5}}};
    return s;
}

// _surface: its sums term by term from 0, as rwmop's loops add them
inline double surface(const Surface& c, const double* x)
{
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < 6; ++i) s1 = s1 + c.lin[i] * x[i];
    for (int i = 0; i < 6; ++i) s2 = s2 + c.sq[i] * x[i] * x[i];
    double v = c.c0 + s1 + s2;
    int k = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j) v += c.cross[k++] * x[i] * x[j];
    return v;
}

}   // namespace detail

// which: a registered problem; x its variables; f its objectives; g its
// constraints (size(which).n_con of them)
// The box of RCM<which> (Kumar et al. 2021, §2.1-2.3; rwmop.PROBLEMS)
inline Bounds bounds(int which)
{
    const double SQ2 = detail::kSq2;
    switch (which) {
        case 1: return {{1, 99}, {1, 99}, {10, 200}, {10, 200}};
        case 2: return {{0.05, 0.5}, {0.2, 0.5}, {0.2, 0.6}, {0.35, 0.5}, {3, 6}};
        case 3: return {{1e-5, 100}, {1e-5, 100}, {1, 3}};
        case 4: return {{0.125, 5}, {0.1, 10}, {0.1, 10}, {0.125, 5}};
        case 5: return {{55, 80}, {75, 110}, {1000, 3000}, {11, 20}};
        case 6:
        case 13: return {{2.6, 3.6}, {0.7, 0.8}, {17, 28}, {7.3, 8.3}, {7.3, 8.3}, {2.9, 3.9}, {5, 5.5}};
        case 7: return box(4, 12, 60);
        case 8: return {{0.5, 1.5}, {0.45, 1.35}, {0.5, 1.5}, {0.5, 1.5}, {0.875, 2.625}, {0.4, 1.2}, {0.4, 1.2}};
        case 9: return {{1, 3}, {SQ2, 3}, {SQ2, 3}, {1, 3}};
        case 10: return {{0.1, 2}, {0.5, 2.5}};
        case 11: return {{0.01, 0.45}, {0.01, 0.1}, {0.01, 0.1}};
        case 12: return {{10, 80}, {10, 50}, {0.9, 5}, {0.9, 5}};
        case 14: return {{60, 80}, {90, 110}, {1, 3}, {0, 1000}, {2, 9}};
        case 15: return {{1, 70}, {0.6, 3}, {0.009, 0.5}};
        case 16: return {{0.01, 0.05}, {0.2, 1}};
        case 17: return {{150, 274.32}, {20, 32.31}, {13, 25}, {10, 11.71}, {14, 18}, {0.63, 0.75}};
        case 18: return {{136, 146}, {56, 68}, {1.4, 2.2}};
        case 21: return {{1.3, 1.7}, {2.5, 3.5}, {1.3, 1.7}, {1.3, 1.7}, {1.3, 1.7}, {1.3, 1.7}};
        case 22: return {{0, 100}, {0, 200}, {0, 100}, {0, 100}, {0, 100}, {0, 100}, {0, 200}, {0, 100}, {0, 200}};
        case 23: return {{0, 1}, {0, 1}, {0, 1}, {0, 1}, {1e-5, 16}, {1e-5, 16}};
        case 25: return {{0, 1.6}, {0, 1}};
        case 26: return {{0.5, 1.4}, {0.5, 1.4}, {0, 1}};
        case 28: {
            Bounds b = box(8, 0, 100);
            b[6] = b[7] = {0, 1};
            return b;
        }
        case 29: {
            Bounds b = box(7, 0, 100);
            b[3] = b[4] = b[5] = b[6] = {0, 1};
            return b;
        }
        default: return {};
    }
}

inline void evaluate(int which, const double* x, double* f, double* g)
{
    using detail::P;
    using detail::rnd;
    const double SQ2 = detail::kSq2;
    std::vector<double> gv, hv;
    switch (which) {
        case 1: {
            const double x1 = rnd(x[0]), x2 = rnd(x[1]), x3 = x[2], x4 = x[3];
            const double z1 = 0.0625 * x1, z2 = 0.0625 * x2;
            f[0] = 1.7781 * z2 * P(x3, 2.0) + 0.6224 * z1 * x3 * x4 + 3.1661 * P(z1, 2.0) * x4 +
                   19.84 * P(z1, 2.0) * x3;
            f[1] = -kPi * P(x3, 2.0) * x4 - 4.0 / 3.0 * kPi * P(x3, 3.0);
            gv = {0.00954 * x3 - z2, 0.0193 * x3 - z1};
            break;
        }
        case 2: {
            const double d1 = x[0], d2 = x[1], d3 = x[2], b = x[3], L = x[4];
            const double r1 = 100.0, r2 = 2770.0, r3 = 7780.0, E1 = 1.6, E2 = 70.0, E3 = 200.0;
            const double c1 = 500.0, c2 = 1500.0, c3 = 800.0;
            const double EI = 2.0 * b / 3.0 *
                              (E1 * P(d1, 3.0) + E2 * (P(d2, 3.0) - P(d1, 3.0)) + E3 * (P(d3, 3.0) - P(d2, 3.0)));
            const double mu = 2.0 * b * (r1 * d1 + r2 * (d2 - d1) + r3 * (d3 - d2));
            if (mu == 0.0 || EI / mu < 0.0) {        // undefined: a failed evaluation
                const double nan = std::numeric_limits<double>::quiet_NaN();
                f[0] = f[1] = nan;
                for (int k = 0; k < 5; ++k) g[k] = nan;
                return;
            }
            f[0] = -kPi / (2.0 * P(L, 2.0)) * std::sqrt(EI / mu);
            f[1] = 2.0 * b * L * (c1 * d1 + c2 * (d2 - d1) + c3 * (d3 - d2));
            gv = {mu * L - 2800.0, d1 - d2, d2 - d1 - 0.15, d2 - d3, d3 - d2 - 0.01};
            break;
        }
        case 3: {
            const double x1 = x[0], x2 = x[1], x3 = x[2];
            f[0] = x1 * std::sqrt(16.0 + P(x3, 2.0)) + x2 * std::sqrt(1.0 + P(x3, 2.0));
            f[1] = 20.0 * std::sqrt(16.0 + P(x3, 2.0)) / (x3 * x1);
            const double g3 = 80.0 * std::sqrt(1.0 + P(x3, 2.0)) / (x3 * x2) - 1e5;
            gv = {f[0] - 0.1, f[1] - 1e5, g3};
            break;
        }
        case 4: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
            const double Pl = 6000.0, L = 14.0, E = 30e6, G = 12e6, tau_max = 13600.0, sigma_max = 30000.0;
            f[0] = 1.10471 * P(x1, 2.0) * x2 + 0.04811 * x3 * x4 * (14.0 + x2);
            f[1] = 4.0 * Pl * P(L, 3.0) / (E * x4 * P(x3, 3.0));
            const double t1 = Pl / (SQ2 * x1 * x2);
            const double M = Pl * (L + x2 / 2.0);
            const double R = std::sqrt(P(x2, 2.0) / 4.0 + P((x1 + x3) / 2.0, 2.0));
            const double J = 2.0 * (SQ2 * x1 * x2 * (P(x2, 2.0) / 12.0 + P((x1 + x3) / 2.0, 2.0)));
            const double t2 = M * R / J;
            const double tau = std::sqrt(P(t1, 2.0) + 2.0 * t1 * t2 * x2 / (2.0 * R) + P(t2, 2.0));
            const double sigma = 6.0 * Pl * L / (x4 * P(x3, 2.0));
            const double Pc = 4.013 * E * std::sqrt(P(x3, 2.0) * P(x4, 6.0) / 36.0) / P(L, 2.0) *
                              (1.0 - x3 / (2.0 * L) * std::sqrt(E / (4.0 * G)));
            gv = {tau - tau_max, sigma - sigma_max, x1 - x4, Pl - Pc};
            break;
        }
        case 5: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
            const double s2 = P(x2, 2.0) - P(x1, 2.0), s3 = P(x2, 3.0) - P(x1, 3.0);
            f[0] = 4.9e-5 * s2 * (x4 - 1.0);
            f[1] = 9.82e6 * (s2 / (x3 * x4 * s3));
            gv = {20.0 - (x2 - x1), x3 / (3.14 * s2) - 0.4, 2.22e-3 * x3 * s3 / P(s2, 2.0) - 1.0,
                  900.0 - 2.66e-2 * x3 * x4 * s3 / s2};
            break;
        }
        case 6: {
            const double x1 = x[0], x2 = x[1], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
            const double x3 = rnd(x[2]), x3sq = x3 * x3;          // an integer's ** 2: exact
            f[0] = 0.7854 * x1 * P(x2, 2.0) * (10.0 * x3sq / 3.0 + 14.933 * x3 - 43.0934) -
                   1.508 * x1 * (P(x6, 2.0) + P(x7, 2.0)) + 7.477 * (P(x6, 3.0) + P(x7, 3.0)) +
                   0.7854 * (x4 * P(x6, 2.0) + x5 * P(x7, 2.0));
            f[1] = detail::reducer_stress(x4, x2, x3, 1.69e7, x6);
            gv = {1.0 / (x1 * P(x2, 2.0) * x3) - 1.0 / 27.0,
                  1.0 / (x1 * P(x2, 2.0) * x3sq) - 1.0 / 397.5,
                  P(x4, 3.0) / (x2 * x3 * P(x6, 4.0)) - 1.0 / 1.93,
                  P(x5, 3.0) / (x2 * x3 * P(x7, 4.0)) - 1.0 / 1.93,
                  x2 * x3 - 40.0,
                  x1 / x2 - 12.0,
                  -x1 / x2 + 5.0,
                  1.9 - x4 + 1.5 * x6,
                  1.9 - x5 + 1.1 * x7,
                  f[1] - 1300.0,
                  detail::reducer_stress(x5, x2, x3, 1.575e8, x7) - 1100.0};
            break;
        }
        case 7: {
            const double x1 = rnd(x[0]), x2 = rnd(x[1]), x3 = rnd(x[2]), x4 = rnd(x[3]);
            f[0] = std::fabs(6.931 - x3 * x4 / (x1 * x2));
            f[1] = std::max({x1, x2, x3, x4});
            gv = {f[0] / 6.931 - 0.5};
            break;
        }
        case 8: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
            const double V_mbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2;
            const double V_fd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6;
            f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7;
            f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3;
            f[2] = 0.5 * (V_mbp + V_fd);
            gv = {-1.0 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3,
                  -0.32 + 0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 +
                      0.0154464 * x6,
                  -0.32 + 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * P(x2, 2.0),
                  -0.32 + 0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 -
                      0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * P(x2, 2.0),
                  -32.0 + 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728,
                  -32.0 + 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7,
                  -32.0 + 46.36 - 9.9 * x2 - 4.4505 * x1,
                  f[1] - 4.0,
                  V_mbp - 9.9,
                  V_fd - 15.7};
            break;
        }
        case 9: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
            const double F = 10.0, E = 2e5, L = 200.0;
            f[0] = L * (2.0 * x1 + SQ2 * x2 + SQ2 * x3 + x4);
            f[1] = F * L / E * (2.0 / x1 + 2.0 * SQ2 / x2 - 2.0 * SQ2 / x3 + 2.0 / x4);
            break;
        }
        case 10: {
            const double x1 = x[0], x2 = x[1];
            const double rho = 0.283, h = 100.0, Pl = 104.0, E = 3e7, s0 = 2e4;
            f[0] = 2.0 * rho * h * x2 * std::sqrt(1.0 + P(x1, 2.0));
            f[1] = rho * h * P(1.0 + P(x1, 2.0), 1.5) * P(1.0 + P(x1, 4.0), 0.5) /
                   (2.0 * SQ2 * E * P(x1, 2.0) * x2);
            const double w = P(1.0 + P(x1, 2.0), 0.5) / (2.0 * SQ2 * x1 * x2);
            gv = {Pl * (1.0 + x1) * w - s0, Pl * (-x1 + 1.0) * w - s0};
            break;
        }
        case 11: {
            const double x1 = x[0], x2 = x[1], x3 = x[2];
            const double a = x1 * x2;
            f[0] = 106780.37 * (x2 + x3) + 61704.67;
            f[1] = 3000.0 * x1;
            f[2] = 2.62314586e3 * x2;
            f[3] = 572250.0 * dm::exp(-3.975 * x2 + 9.9 * x3 + 2.74);
            f[4] = 25.0 * (1.39 / a + 4940.0 * x3 - 80.0);
            gv = {-1.0 + (0.00139 / a + 4.94 * x3 - 0.08),
                  -1.0 + (0.000306 / a + 1.082 * x3 - 0.0986),
                  -50000.0 + (12.307 / a + 49408.24 * x3 + 4051.02),
                  -16000.0 + (12.098 / a + 8046.33 * x3 - 696.71),
                  -10000.0 + (2.138 / a + 7883.39 * x3 - 705.04),
                  -2000.0 + (0.417 * a + 1721.26 * x3 - 136.54),
                  -550.0 + (0.164 / a + 631.13 * x3 - 54.48)};
            break;
        }
        case 12: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
            const double Pl = 600.0, L = 200.0, E = 20000.0;
            const double web = x1 - 2.0 * x4;
            const double I4 = x3 * P(web, 3.0) + 2.0 * x2 * x4 * (4.0 * P(x4, 2.0) + 3.0 * x1 * web);
            f[0] = 2.0 * x2 * x4 + x3 * web;
            f[1] = Pl * P(L, 3.0) / (4.0 * E * I4);
            gv = {-16.0 + 180000.0 * x1 / I4 + 15000.0 * x2 / (web * P(x3, 3.0) + 2.0 * x4 * P(x2, 3.0))};
            break;
        }
        case 13: {
            const double x1 = x[0], x2 = x[1], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
            const double x3 = rnd(x[2]), x3sq = x3 * x3;
            f[0] = 0.7854 * P(x2, 2.0) * x1 * (14.9334 / x3 - 43.0934 + 3.3333 * x3sq) +
                   0.7854 * (x5 * P(x7, 2.0) + x4 * P(x6, 2.0)) - 1.508 * x1 * (P(x7, 2.0) + P(x6, 2.0)) +
                   7.477 * (P(x7, 3.0) + P(x6, 3.0));
            f[1] = 10.0 * P(x6, -3.0) * std::sqrt(16.91e6 + P(745.0 * x4 / (x2 * x3), 2.0));
            f[2] = 10.0 * P(x7, -3.0) * std::sqrt(157.5e6 + P(745.0 * x5 / (x2 * x3), 2.0));
            gv = {1.0 / (x1 * P(x2, 2.0) * x3) - 1.0 / 27.0,
                  1.0 / (x1 * P(x2, 2.0) * x3sq) - 1.0 / 397.5,
                  1.0 / (x2 * P(x6, 4.0) * x3 * P(x4, -3.0)) - 1.0 / 1.93,
                  1.0 / (x2 * P(x7, 4.0) * x3 * P(x5, -3.0)) - 1.0 / 1.93,
                  f[1] - 1100.0,
                  f[2] - 850.0,
                  x2 * x3 - 40.0,
                  -x1 / x2 + 5.0,
                  x1 / x2 - 12.0,
                  1.5 * x6 - x4 + 1.9,
                  1.1 * x7 - x5 + 1.9};
            break;
        }
        case 14: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4];
            const double rho = 7.8e-6, dR = 20.0, L_max = 30.0, mu = 0.6, V_max = 10.0, delta = 0.5, s = 1.5;
            const double T_max = 15.0, n = 250.0, I_z = 55.0, M_s = 40.0, M_f = 3.0, p_max = 1.0;
            const double M_h = 2.0 / 3.0 * mu * x4 * x5 * (P(x2, 3.0) - P(x1, 3.0)) / (P(x2, 2.0) - P(x1, 2.0));
            const double w = kPi * n / 30.0;
            const double A = kPi * (P(x2, 2.0) - P(x1, 2.0));
            const double p_rz = x4 / A;
            const double R_sr = 2.0 / 3.0 * (P(x2, 3.0) - P(x1, 3.0)) / (P(x2, 2.0) * P(x1, 2.0));
            const double V_sr = kPi * R_sr * n / 30.0;
            const double T = I_z * w / (M_h + M_f);
            f[0] = kPi * (P(x2, 2.0) - P(x1, 2.0)) * x3 * (x5 + 1.0) * rho;
            f[1] = T;
            gv = {-p_max + p_rz,           p_rz * V_sr - V_max * p_max, dR + x1 - x2,
                  -L_max + (x5 + 1.0) * (x3 + delta), s * M_s - M_h, -T,
                  -V_max + V_sr,           T - T_max};
            break;
        }
        case 15: {
            const double x1 = rnd(x[0]), x2 = x[1];
            const double x3 = detail::nearest(detail::wire(), x[2]);
            const double C_f = (4.0 * x2 / x3 - 1.0) / (4.0 * x2 / x3 - 4.0) + 0.615 * x3 / x2;
            const double K = 11.5e6 * P(x3, 4.0) / (8.0 * x1 * P(x2, 3.0));
            const double sigma_p = 300.0 / K;
            const double l_f = 1000.0 / K + 1.05 * (x1 + 2.0) * x3;
            f[0] = P(kPi, 2.0) * x2 * P(x3, 2.0) * (x1 + 2.0) / 4.0;
            f[1] = 8000.0 * C_f * x2 / (kPi * P(x3, 3.0));
            gv = {f[1] - 189000.0, l_f - 14.0,     0.2 - x3, x2 - 3.0, 3.0 - x2 / x3, sigma_p - 6.0,
                  sigma_p + 700.0 / K + 1.05 * (x1 + 2.0) * x3 - l_f, 1.25 - 700.0 / K};
            break;
        }
        case 16: {
            const double x1 = x[0], x2 = x[1];
            const double Pl = 1.0, E = 307e8, rho = 7800.0, Sy = 3e5, d_max = 0.05;
            f[0] = 0.25 * rho * kPi * x2 * P(x1, 2.0);
            f[1] = 64.0 * Pl * P(x2, 3.0) / (3.0 * E * kPi * P(x1, 4.0));
            gv = {-Sy + 32.0 * Pl * x2 / (kPi * P(x1, 3.0)), -d_max + f[1]};
            break;
        }
        case 17: {
            const double L = x[0], B = x[1], D = x[2], T = x[3], V_k = x[4], C_B = x[5];
            const double a = 4977.06 * P(C_B, 2.0) - 8105.61 * C_B + 4456.51;
            const double b = -10847.2 * P(C_B, 2.0) + 12817.0 * C_B - 6960.32;
            const double F_n = 0.5144 / P(9.8065 * L, 0.5);
            const double Pw = P(1.025 * L * B * T * C_B, 0.67) * P(V_k, 3.0) / (a + b * F_n);
            const double W_s = 0.034 * P(L, 1.7) * P(B, 0.6) * P(D, 0.4) * P(C_B, 0.5);
            const double W_o = P(L, 0.8) * P(B, 0.6) * P(D, 0.3) * P(C_B, 0.1);
            const double W_m = 0.17 * P(Pw, 0.9);
            const double ls = W_s + W_o + W_m;
            const double D_wt = 1.025 * L * B * T * C_B - ls;
            const double F_c = 4.56e-5 * Pw + 0.2;
            const double D_cwt = D_wt - F_c * (5000.0 * V_k / 24.0 + 5.0) - 2.0 * P(D_wt, 0.5);
            const double R_trp = 350.0 / (5000.0 * V_k / 24.0 + 2.0 * (D_cwt / 8000.0 + 0.5));
            const double ac = D_cwt * R_trp;
            const double S_d = 5000.0 * V_k / 24.0;
            const double C_c = 0.26 * (2000.0 * P(W_s, 0.85) + 3500.0 * W_o + 2400.0 * P(Pw, 0.8));
            const double C_r = 40000.0 * P(D_wt, 0.3);
            const double C_v = (105.0 * F_c * S_d + 6.3 * P(D_wt, 0.8)) * R_trp;
            f[0] = (C_c + C_r + C_v) / ac;
            f[1] = ls;
            f[2] = -ac;
            gv = {-L / B + 6.0,
                  -15.0 + L / D,
                  -19.0 + L / T,
                  -0.45 * P(D_wt, 0.31) + T,
                  -0.7 * D - 0.7 + T,
                  -0.32 + F_n,
                  -0.53 * T - (0.085 * C_B - 0.002) * P(B, 2.0) / (T * C_B) + (1.0 + 0.52 * D) + 0.07 * B,
                  -D_wt + 3000.0,
                  -500000.0 + D_wt};
            break;
        }
        case 18: {
            const double hh = x[0], w = x[1], t = x[2];
            const double Ea = 14496.5, Fa = 234.9;
            const double E = (-70973.4 + 958.656 * w + 614.173 * hh - 3.827 * w * hh + 57.023 * w * t +
                              63.274 * hh * t - 3.582 * P(w, 2.0) - 1.4842 * P(hh, 2.0) - 1890.174 * P(t, 2.0));
            const double F = (111.854 - 20.210 * w + 7.560 * hh - 0.025 * w * hh + 2.731 * w * t - 1.479 * hh * t +
                              0.165 * P(w, 2.0));
            f[0] = Ea / E;
            f[1] = F / Fa;
            gv = {-(hh - 136.0) * (146.0 - hh), -(w - 58.0) * (66.0 - w), -(t - 1.4) * (2.2 - t)};
            break;
        }
        case 21: {
            f[0] = detail::surface(detail::surfaces()[0], x);
            f[1] = detail::surface(detail::surfaces()[1], x);
            gv = {f[0] - 5.0, -f[0], f[1] - 28.0, -f[1]};
            break;
        }
        case 22: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6], x8 = x[7],
                         x9 = x[8];
            f[0] = -9.0 * x1 - 15.0 * x2 + 6.0 * x3 + 16.0 * x4;
            f[1] = 10.0 * (x5 + x6);
            hv = {x7 + x8 - x4 - x3, x1 - x5 - x7, x2 - x6 - x8, x9 * x7 + x9 * x8 - 3.0 * x3 - x4};
            gv = {x9 * x7 + 2.0 * x5 - 2.5 * x1, x9 * x8 + 2.0 * x6 - 1.5 * x2};
            break;
        }
        case 23: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5];
            const double k1 = 0.09755988, k2 = 0.99 * k1, k3 = 0.0391908, k4 = 0.9 * k3;
            f[0] = -x4;
            f[1] = P(x5, 0.5) + P(x6, 0.5);
            hv = {k1 * x5 * x2 + x1 - 1.0, k3 * x5 * x3 + x3 + x1 - 1.0, k2 * x6 * x2 - x1 + x2,
                  k4 * x6 * x4 + x2 - x1 + x4 - x3};
            gv = {P(x5, 0.5) + P(x6, 0.5) - 4.0};
            break;
        }
        case 25: {
            const double x1 = x[0], x2 = rnd(x[1]);
            f[0] = x2 + 2.0 * x1;
            f[1] = -P(x1, 2.0) - x2;
            gv = {-P(x1, 2.0) - x2 + 1.25, x1 + x2 - 1.6};
            break;
        }
        case 26: {
            const double x1 = x[0], x2 = x[1], x3 = rnd(x[2]);
            f[0] = -x3 + x2 + 2.0 * x1;
            f[1] = -P(x1, 2.0) - x2 + x1 * x3;
            gv = {x2 - x1 + x3};
            hv = {-2.0 * dm::exp(-x2) + x1};
            break;
        }
        case 28: {
            const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5];
            const double x7 = rnd(x[6]), x8 = rnd(x[7]);
            f[0] = 7.5 * x7 + 5.5 * x8 + 7.0 * x5 + 6.0 * x6 + 5.0 * (x1 + x2);
            f[1] = x1 + x2;
            hv = {x7 + x8 - 1.0, x3 - 0.9 * (1.0 - dm::exp(-0.5 * x5)) * x1, x4 - 0.8 * (1.0 - dm::exp(-0.4 * x6)) * x2,
                  x3 + x4 - 10.0, x3 * x7 + x4 * x8 - 10.0};
            gv = {x5 - 10.0 * x7, x6 - 10.0 * x8, x1 - 20.0 * x7, x2 - 20.0 * x8};
            break;
        }
        case 29: {
            const double x1 = x[0], x2 = x[1], x3 = x[2];
            const double x4 = rnd(x[3]), x5 = rnd(x[4]), x6 = rnd(x[5]), x7 = rnd(x[6]);
            const double tail = P(1.0 - x1, 2.0) + P(2.0 - x2, 2.0) + P(3.0 - x3, 2.0);
            f[0] = P(1.0 - x4, 2.0) + P(1.0 - x5, 2.0) + P(1.0 - x6, 2.0) - dm::log(1.0 + x7) + tail;
            f[1] = tail;
            // an integer's ** of an integer: exact
            gv = {x1 + x2 + x3 + x4 + x5 + x6 - 5.0,
                  x6 * x6 * x6 + P(x1, 2.0) + P(x2, 2.0) + P(x3, 2.0) - 5.5,
                  x1 + x4 - 1.2,
                  x2 + x5 - 1.8,
                  x3 + x6 - 2.5,
                  x1 + x7 - 1.2,
                  x5 * x5 + P(x2, 2.0) - 1.64,
                  x6 * x6 + P(x3, 2.0) - 4.25,
                  x5 * x5 + P(x3, 2.0) - 4.64};
            break;
        }
        default:
            return;
    }
    std::size_t k = 0;
    for (double v : gv) g[k++] = v;
    for (double v : hv) g[k++] = std::fabs(v) - kEqEps;
}

}   // namespace mootation::suite::rcm

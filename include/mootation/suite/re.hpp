#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The RE and CRE real-world problems (Tanabe & Ishibuchi, Applied Soft
// Computing 89, 2020), as benchmarks/re.py evaluates them (task 17,
// supplement 3, stage 3a): its code, the port of reproblems' Python version,
// expression for expression in its order. np.power(v, 2) is v * v and
// np.power(v, 0.5) sqrt(v), as NumPy computes them; every other power dm::pow,
// exp dm::exp; np.round rounds half to even; the nearest value of a list the
// first of equal distances (argmin). re.py stays the reference.
//
// RE91 draws its four normals from x itself (re.re91_by_x: SplitMix64 over
// the bits of x); a run's seeded stream (re.RE91Evaluator) is Python's.
// CRE's constraints are the violations, max(-g, 0) (<= 0 satisfied as they
// stand); an RE that reformulates a CRE has their sum as its last objective.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::re {

enum class Which { RE21, RE22, RE22c, RE23, RE24, RE25, RE31, RE32, RE33, RE33c, RE34, RE35, RE36, RE37, RE41,
                   RE42, RE61, RE91, CRE21, CRE22, CRE23, CRE24, CRE25, CRE31, CRE32, CRE51 };

struct Size { int n_var, n_obj, n_con; };

inline bool parse(const std::string& f, Which& w)
{
    static const struct { const char* name; Which w; } names[] = {
        {"RE21", Which::RE21}, {"RE22", Which::RE22}, {"RE22c", Which::RE22c}, {"RE23", Which::RE23},
        {"RE24", Which::RE24}, {"RE25", Which::RE25}, {"RE31", Which::RE31}, {"RE32", Which::RE32},
        {"RE33", Which::RE33}, {"RE33c", Which::RE33c}, {"RE34", Which::RE34}, {"RE35", Which::RE35},
        {"RE36", Which::RE36}, {"RE37", Which::RE37}, {"RE41", Which::RE41}, {"RE42", Which::RE42},
        {"RE61", Which::RE61}, {"RE91", Which::RE91}, {"CRE21", Which::CRE21}, {"CRE22", Which::CRE22},
        {"CRE23", Which::CRE23}, {"CRE24", Which::CRE24}, {"CRE25", Which::CRE25}, {"CRE31", Which::CRE31},
        {"CRE32", Which::CRE32}, {"CRE51", Which::CRE51}};
    for (const auto& e : names)
        if (f == e.name) {
            w = e.w;
            return true;
        }
    return false;
}

inline Size size(Which w)
{
    switch (w) {
        case Which::RE21: return {4, 2, 0};
        case Which::RE22: case Which::RE22c: return {3, 2, 0};
        case Which::RE23: return {4, 2, 0};
        case Which::RE24: return {2, 2, 0};
        case Which::RE25: return {3, 2, 0};
        case Which::RE31: return {3, 3, 0};
        case Which::RE32: case Which::RE33: case Which::RE33c: return {4, 3, 0};
        case Which::RE34: return {5, 3, 0};
        case Which::RE35: return {7, 3, 0};
        case Which::RE36: return {4, 3, 0};
        case Which::RE37: return {4, 3, 0};
        case Which::RE41: return {7, 4, 0};
        case Which::RE42: return {6, 4, 0};
        case Which::RE61: return {3, 6, 0};
        case Which::RE91: return {7, 9, 0};
        case Which::CRE21: return {3, 2, 3};
        case Which::CRE22: return {4, 2, 4};
        case Which::CRE23: return {4, 2, 4};
        case Which::CRE24: return {7, 2, 11};
        case Which::CRE25: return {4, 2, 1};
        case Which::CRE31: return {7, 3, 10};
        case Which::CRE32: return {6, 3, 9};
        case Which::CRE51: return {3, 5, 7};
    }
    return {0, 0, 0};
}

constexpr double kSqrt2 = 1.4142135623730951;      // np.sqrt(2)

// The box of a problem as reproblems has it (re.RE; a CRE problem: its RE
// counterpart's; RE33c: x4 in [2, 11], re.RE33C_BOUNDS). RE21's from F = 10,
// sigma = 10: [F / sigma, 3 F / sigma], x2 and x3 from sqrt(2) F / sigma.
inline Bounds bounds(Which w)
{
    switch (w) {
        case Which::RE21: return {{1.0, 3.0}, {kSqrt2, 3.0}, {kSqrt2, 3.0}, {1.0, 3.0}};
        case Which::RE22:
        case Which::RE22c: return {{0.2, 15}, {0.0, 20}, {0.0, 40}};
        case Which::RE23: return {{1, 100}, {1, 100}, {10, 200}, {10, 240}};
        case Which::RE24: return {{0.5, 4}, {0.5, 50}};
        case Which::RE25: return {{1, 70}, {0.6, 3}, {0.09, 0.5}};
        case Which::RE31:
        case Which::CRE21: return {{0.00001, 100.0}, {0.00001, 100.0}, {1.0, 3.0}};                  // truss
        case Which::RE32:
        case Which::CRE22: return {{0.125, 5.0}, {0.1, 10.0}, {0.1, 10.0}, {0.125, 5.0}};           // weld
        case Which::RE33:
        case Which::CRE23: return {{55, 80}, {75, 110}, {1000, 3000}, {11, 20}};                    // brake
        case Which::RE33c: return {{55, 80}, {75, 110}, {1000, 3000}, {2, 11}};
        case Which::RE34: return box(5, 1, 3);
        case Which::RE35:
        case Which::CRE24: return {{2.6, 3.6}, {0.7, 0.8}, {17, 28}, {7.3, 8.3}, {7.3, 8.3}, {2.9, 3.9}, {5.0, 5.5}};
        case Which::RE36:
        case Which::CRE25: return box(4, 12, 60);                                                    // gear
        case Which::RE37: return box(4, 0, 1);
        case Which::RE41:
        case Which::RE91:
        case Which::CRE31: return {{0.5, 1.5}, {0.45, 1.35}, {0.5, 1.5}, {0.5, 1.5},              // car
                                   {0.875, 2.625}, {0.4, 1.2}, {0.4, 1.2}};
        case Which::RE42:
        case Which::CRE32: return {{150.0, 274.32}, {20.0, 32.31}, {13.0, 25.0}, {10.0, 11.71},   // ship
                                   {14.0, 18.0}, {0.63, 0.75}};
        case Which::RE61:
        case Which::CRE51: return {{0.01, 0.45}, {0.01, 0.10}, {0.01, 0.10}};                      // water
    }
    return {};
}

// RE22's list of x1's values as reproblems has it ("3,10" for 3.10: 3 and 10;
// 7.00 missing), and RE22c's, the source's (re.RE22_VALUES, RE22C_VALUES)
constexpr double kRE22[78] = {0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32,
                              1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60,
                              2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0,
                              4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0,
                              6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0,
                              11.06, 11.85, 12.0, 13.0, 14.0, 15.0};
constexpr double kRE22c[77] = {0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32,
                               1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60,
                               2.64, 2.79, 2.80, 3.0, 3.08, 3.10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0,
                               4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0,
                               6.16, 6.32, 6.60, 7.00, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27,
                               11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0};
constexpr double kRE25[42] = {0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018,
                              0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08,
                              0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
                              0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5};

// the value of the list nearest v: np.abs(values - v).argmin(), the first of
// equal distances (a NaN distance, the first NaN)
inline double nearest(const double* values, int n, double v)
{
    int best = 0;
    double bd = std::abs(values[0] - v);
    for (int i = 1; i < n && !std::isnan(bd); ++i) {
        const double d = std::abs(values[i] - v);
        if (d < bd || std::isnan(d)) {
            best = i;
            bd = d;
        }
    }
    return values[best];
}

// np.round: to the nearest integer, half to even
inline double np_round(double v) { return std::nearbyint(v); }

// re._violations: -g where g < 0, else 0
inline double violation(double g) { return g < 0.0 ? -g : 0.0; }

// ---- the shared engineering models: the objectives and the violations ------
inline void truss(const double* x, double* f, double* v)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2];
    f[0] = x1 * std::sqrt(16.0 + (x3 * x3)) + x2 * std::sqrt(1.0 + x3 * x3);
    f[1] = (20.0 * std::sqrt(16.0 + (x3 * x3))) / (x1 * x3);
    v[0] = violation(0.1 - f[0]);
    v[1] = violation(100000.0 - f[1]);
    v[2] = violation(100000 - ((80.0 * std::sqrt(1.0 + x3 * x3)) / (x3 * x2)));
}

inline void weld(const double* x, double* f, double* v)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    const double P = 6000, L = 14, E = 30 * 1e6, G = 12 * 1e6, tauMax = 13600, sigmaMax = 30000;
    f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2);
    f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3);
    const double M = P * (L + (x2 / 2));
    const double h = (x1 + x3) / 2.0;
    double tmpVar = ((x2 * x2) / 4.0) + h * h;
    const double R = std::sqrt(tmpVar);
    tmpVar = ((x2 * x2) / 12.0) + h * h;
    const double J = 2 * kSqrt2 * x1 * x2 * tmpVar;
    const double tauDashDash = (M * R) / J;
    const double tauDash = P / (kSqrt2 * x1 * x2);
    tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash);
    const double tau = std::sqrt(tmpVar);
    const double sigma = (6 * P * L) / (x4 * x3 * x3);
    tmpVar = 4.013 * E * std::sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L);
    const double tmpVar2 = (x3 / (2 * L)) * std::sqrt(E / (4 * G));
    const double PC = tmpVar * (1 - tmpVar2);
    v[0] = violation(tauMax - tau);
    v[1] = violation(sigmaMax - sigma);
    v[2] = violation(x4 - x1);
    v[3] = violation(PC - P);
}

inline void brake(const double* x, double* f, double* v)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    f[0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0);
    f[1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1));
    const double d = x2 * x2 - x1 * x1;
    v[0] = violation((x2 - x1) - 20.0);
    v[1] = violation(0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1))));
    v[2] = violation(1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / (d * d));
    v[3] = violation((2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0);
}

inline void reducer(const double* x, double* f, double* v)
{
    const double x1 = x[0], x2 = x[1], x3 = np_round(x[2]), x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
    f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934)
           - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
           + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7);
    const double a = (745.0 * x4) / (x2 * x3);
    f[1] = std::sqrt(a * a + 1.69 * 1e7) / (0.1 * x6 * x6 * x6);
    const double b = (745.0 * x5) / (x2 * x3);
    const double g[11] = {-(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0,
                          -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5,
                          -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93,
                          -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93,
                          -(x2 * x3) + 40.0,
                          -(x1 / x2) + 12.0,
                          -5.0 + (x1 / x2),
                          -1.9 + x4 - 1.5 * x6,
                          -1.9 + x5 - 1.1 * x7,
                          -f[1] + 1300.0,
                          -std::sqrt(b * b + 1.575 * 1e8) / (0.1 * x7 * x7 * x7) + 1100.0};
    for (int i = 0; i < 11; ++i) v[i] = violation(g[i]);
}

inline void gear(const double* x, double* f, double* v)
{
    const double x1 = np_round(x[0]), x2 = np_round(x[1]), x3 = np_round(x[2]), x4 = np_round(x[3]);
    f[0] = std::abs(6.931 - ((x3 / x1) * (x4 / x2)));
    double top = x1;                                   // Python's max: the first of the largest
    for (double c : {x2, x3, x4})
        if (c > top) top = c;
    f[1] = top;
    v[0] = violation(0.5 - (f[0] / 6.931));
}

inline void car(const double* x, double* f, double* v)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
    f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7;
    f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3;
    const double Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2;
    const double Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6;
    f[2] = 0.5 * (Vmbp + Vfd);
    const double g[10] = {
        1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3),
        0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6),
        0.32 - (0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7
                + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2),
        0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2),
        32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7),
        32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728),
        32 - (46.36 - 9.9 * x2 - 4.4505 * x1),
        4 - f[1],
        9.9 - Vmbp,
        15.7 - Vfd};
    for (int i = 0; i < 10; ++i) v[i] = violation(g[i]);
}

inline void ship(const double* x, double* f, double* v)
{
    const double x_L = x[0], x_B = x[1], x_D = x[2], x_T = x[3], x_Vk = x[4], x_CB = x[5];
    const double displacement = 1.025 * x_L * x_B * x_T * x_CB;
    const double V = 0.5144 * x_Vk;
    const double g = 9.8065;
    const double Fn = V / std::sqrt(g * x_L);
    const double a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51;
    const double b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32;
    const double power = (dm::pow(displacement, 2.0 / 3.0) * dm::pow(x_Vk, 3.0)) / (a + (b * Fn));
    const double outfit_weight = 1.0 * dm::pow(x_L, 0.8) * dm::pow(x_B, 0.6) * dm::pow(x_D, 0.3) * dm::pow(x_CB, 0.1);
    const double steel_weight = 0.034 * dm::pow(x_L, 1.7) * dm::pow(x_B, 0.7) * dm::pow(x_D, 0.4)
                                * std::sqrt(x_CB);
    const double machinery_weight = 0.17 * dm::pow(power, 0.9);
    const double light_ship_weight = steel_weight + outfit_weight + machinery_weight;
    const double ship_cost = 1.3 * ((2000.0 * dm::pow(steel_weight, 0.85)) + (3500.0 * outfit_weight)
                                    + (2400.0 * dm::pow(power, 0.8)));
    const double capital_costs = 0.2 * ship_cost;
    const double DWT = displacement - light_ship_weight;
    const double running_costs = 40000.0 * dm::pow(DWT, 0.3);
    const double round_trip_miles = 5000.0;
    const double sea_days = (round_trip_miles / 24.0) * x_Vk;
    const double handling_rate = 8000.0;
    const double daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2;
    const double fuel_price = 100.0;
    const double fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price;
    const double port_cost = 6.3 * dm::pow(DWT, 0.8);
    const double fuel_carried = daily_consumption * (sea_days + 5.0);
    const double miscellaneous_DWT = 2.0 * std::sqrt(DWT);
    const double cargo_DWT = DWT - fuel_carried - miscellaneous_DWT;
    const double port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5);
    const double RTPA = 350.0 / (sea_days + port_days);
    const double voyage_costs = (fuel_cost + port_cost) * RTPA;
    const double annual_costs = capital_costs + running_costs + voyage_costs;
    const double annual_cargo = cargo_DWT * RTPA;
    f[0] = annual_costs / annual_cargo;
    f[1] = light_ship_weight;
    f[2] = -annual_cargo;
    const double KB = 0.53 * x_T;
    const double BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB);
    const double KG = 1.0 + 0.52 * x_D;
    const double c[9] = {(x_L / x_B) - 6.0,
                         -(x_L / x_D) + 15.0,
                         -(x_L / x_T) + 19.0,
                         0.45 * dm::pow(DWT, 0.31) - x_T,
                         0.7 * x_D + 0.7 - x_T,
                         500000.0 - DWT,
                         DWT - 3000.0,
                         0.32 - Fn,
                         (KB + BMT - KG) - (0.07 * x_B)};
    for (int i = 0; i < 9; ++i) v[i] = violation(c[i]);
}

inline void water(const double* x, double* f, double* v)
{
    f[0] = 106780.37 * (x[1] + x[2]) + 61704.67;
    f[1] = 3000 * x[0];
    f[2] = 305700 * 2289 * x[1] / dm::pow(0.06 * 2289, 0.65);
    f[3] = 250 * 2289 * dm::exp(-39.75 * x[1] + 9.9 * x[2] + 2.74);
    f[4] = 25 * (1.39 / (x[0] * x[1]) + 4940 * x[2] - 80);
    const double g[7] = {1 - (0.00139 / (x[0] * x[1]) + 4.94 * x[2] - 0.08),
                         1 - (0.000306 / (x[0] * x[1]) + 1.082 * x[2] - 0.0986),
                         50000 - (12.307 / (x[0] * x[1]) + 49408.24 * x[2] + 4051.02),
                         16000 - (2.098 / (x[0] * x[1]) + 8046.33 * x[2] - 696.71),
                         10000 - (2.138 / (x[0] * x[1]) + 7883.39 * x[2] - 705.04),
                         2000 - (0.417 * x[0] * x[1] + 1721.26 * x[2] - 136.54),
                         550 - (0.164 / (x[0] * x[1]) + 631.13 * x[2] - 54.48)};
    for (int i = 0; i < 7; ++i) v[i] = violation(g[i]);
}

// ---- the RE problems that are not reformulations of a CRE -------------------
inline void re21(const double* x, double* f)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    const double F = 10.0, E = 2.0 * 1e5, L = 200.0;
    f[0] = L * ((2 * x1) + kSqrt2 * x2 + std::sqrt(x3) + x4);
    f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * kSqrt2 / x2) - (2.0 * kSqrt2 / x3) + (2.0 / x4));
}

inline void re22(const double* x, bool corrected, double* f)
{
    const double x1 = corrected ? nearest(kRE22c, 77, x[0]) : nearest(kRE22, 78, x[0]);
    const double x2 = x[1], x3 = x[2];
    if (x2 == 0.0) {                                   // g divides by x2: a failed evaluation
        f[0] = f[1] = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    f[0] = (29.4 * x1) + (0.6 * x2 * x3);
    const double g0 = violation((x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0);
    const double g1 = violation(4.0 - (x3 / x2));
    f[1] = g0 + g1;
}

inline void re23(const double* x, double* f)
{
    const double x1 = 0.0625 * np_round(x[0]), x2 = 0.0625 * np_round(x[1]), x3 = x[2], x4 = x[3];
    f[0] = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3);
    const double g0 = violation(x1 - (0.0193 * x3));
    const double g1 = violation(x2 - (0.00954 * x3));
    const double g2 = violation((kPi * x3 * x3 * x4) + ((4.0 / 3.0) * (kPi * x3 * x3 * x3)) - 1296000);
    f[1] = g0 + g1 + g2;
}

inline void re24(const double* x, double* f)
{
    const double x1 = x[0], x2 = x[1];
    f[0] = x1 + (120 * x2);
    const double E = 700000, sigma_b_max = 700, tau_max = 450, delta_max = 1.5;
    const double sigma_k = (E * x1 * x1) / 100;
    const double sigma_b = 4500 / (x1 * x2);
    const double tau = 1800 / x2;
    const double delta = (56.2 * 10000) / (E * x1 * x2 * x2);
    const double g[4] = {violation(1 - (sigma_b / sigma_b_max)), violation(1 - (tau / tau_max)),
                         violation(1 - (delta / delta_max)), violation(1 - (sigma_b / sigma_k))};
    f[1] = g[0] + g[1] + g[2] + g[3];
}

inline void re25(const double* x, double* f)
{
    const double x1 = np_round(x[0]), x2 = x[1], x3 = nearest(kRE25, 42, x[2]);
    f[0] = (kPi * kPi * x2 * x3 * x3 * (x1 + 2)) / 4.0;
    const double Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2);
    const double Fmax = 1000.0, S = 189000.0, G = 11.5 * 1e+6;
    const double K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2);
    const double lmax = 14.0;
    const double lf = (Fmax / K) + 1.05 * (x1 + 2) * x3;
    const double Fp = 300.0;
    const double sigmaP = Fp / K;
    const double sigmaPM = 6, sigmaW = 1.25;
    const double g[6] = {-((8 * Cf * Fmax * x2) / (kPi * x3 * x3 * x3)) + S,
                         -lf + lmax,
                         -3 + (x2 / x3),
                         -sigmaP + sigmaPM,
                         -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf,
                         sigmaW - ((Fmax - Fp) / K)};
    double s = violation(g[0]);
    for (int i = 1; i < 6; ++i) s = s + violation(g[i]);
    f[1] = s;
}

inline void re34(const double* x, double* f)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4];
    f[0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4)
           + (4.4559504 * x5);
    f[1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4)
           + (0.0861 * x1 * x5) + (0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3)
           + (0.1764 * x4 * x4);
    f[2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3)
           - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2)
           + (0.0109 * x4 * x4);
}

inline void re37(const double* x, double* f)
{
    const double xAlpha = x[0], xHA = x[1], xOA = x[2], xOPTT = x[3];
    f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT)
           - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA)
           - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA)
           - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA)
           + (0.0184 * xOPTT * xOPTT);
    f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT)
           + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA)
           - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA)
           + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA)
           + (0.0192 * xOPTT * xOPTT);
    f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT)
           - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA)
           + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA)
           + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT)
           + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha)
           - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA)
           - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA);
}

inline double max0(double v) { return 0.0 < v ? v : 0.0; }   // Python's max(0.0, v)

inline void re91(const double* x, const double* z, double* f)
{
    const double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
    const double x8 = 0.006 * (z[0]) + 0.345, x9 = 0.006 * (z[1]) + 0.192;
    const double x10 = 10 * (z[2]) + 0.0, x11 = 10 * (z[3]) + 0.0;
    f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.75 * x5 + 0.00001 * x6 + 2.73 * x7;
    f[1] = max0((1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) / 1.0);
    f[2] = max0((0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5
                 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11)
                / 0.32);
    f[3] = max0((0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6
                 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6
                 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10
                 - 0.018 * x2 * x2)
                / 0.32);
    f[4] = max0((0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2)
                / 0.32);
    const double tmp = ((28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8
                         + 0.32 * x9 * x10)
                        + (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10
                           - 9.98 * x7 * x8 + 22 * x8 * x9)
                        + (46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10))
                       / 3;
    f[5] = max0(tmp / 32);
    f[6] = max0((4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11)
                / 4.0);
    f[7] = max0((10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10
                 + 0.028 * x6 * x10)
                / 9.9);
    f[8] = max0((16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11
                 - 0.000786 * x11 * x11)
                / 15.7);
}

inline std::uint64_t splitmix64(std::uint64_t z)
{
    z = z + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// 53 high bits of a 64-bit hash as a double in [0, 1) (uninformative._unit)
inline double unit(std::uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }

// RE91 without a seed: the four normals drawn from x itself (re.re91_by_x)
inline void re91_by_x(const double* x, double* f)
{
    std::uint64_t h = 0;
    for (int i = 0; i < 7; ++i) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x[i], sizeof bits);
        h = splitmix64(h ^ bits);
    }
    double z[4];
    for (int i = 0; i < 4; ++i) {
        const double u1 = unit(splitmix64(h + 2 * static_cast<std::uint64_t>(i) + 1));
        const double u2 = unit(splitmix64(h + 2 * static_cast<std::uint64_t>(i) + 2));
        z[i] = std::sqrt(-2.0 * dm::log(1.0 - u1)) * dm::cos(2.0 * kPi * u2);
    }
    re91(x, z, f);
}

// an RE problem from a CRE model: its objectives, then the sum of the
// violations in their order (re._reformulated)
template <class Model>
inline void reformulated(Model model, const double* x, int m, int c, double* f)
{
    double v[11];
    model(x, f, v);
    double s = v[0];
    for (int i = 1; i < c; ++i) s = s + v[i];
    f[m] = s;
}

// f (and g, the violations, for a CRE) of the point x
inline void evaluate(Which w, const double* x, double* f, double* g)
{
    switch (w) {
        case Which::RE21: re21(x, f); return;
        case Which::RE22: re22(x, false, f); return;
        case Which::RE22c: re22(x, true, f); return;
        case Which::RE23: re23(x, f); return;
        case Which::RE24: re24(x, f); return;
        case Which::RE25: re25(x, f); return;
        case Which::RE31: reformulated(truss, x, 2, 3, f); return;
        case Which::RE32: reformulated(weld, x, 2, 4, f); return;
        case Which::RE33: case Which::RE33c: reformulated(brake, x, 2, 4, f); return;
        case Which::RE34: re34(x, f); return;
        case Which::RE35: reformulated(reducer, x, 2, 11, f); return;
        case Which::RE36: reformulated(gear, x, 2, 1, f); return;
        case Which::RE37: re37(x, f); return;
        case Which::RE41: reformulated(car, x, 3, 10, f); return;
        case Which::RE42: reformulated(ship, x, 3, 9, f); return;
        case Which::RE61: reformulated(water, x, 5, 7, f); return;
        case Which::RE91: re91_by_x(x, f); return;
        case Which::CRE21: truss(x, f, g); return;
        case Which::CRE22: weld(x, f, g); return;
        case Which::CRE23: brake(x, f, g); return;
        case Which::CRE24: reducer(x, f, g); return;
        case Which::CRE25: gear(x, f, g); return;
        case Which::CRE31: car(x, f, g); return;
        case Which::CRE32: ship(x, f, g); return;
        case Which::CRE51: water(x, f, g); return;
    }
}

}   // namespace mootation::suite::re

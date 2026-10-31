#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DOC1-9 (Liu & Wang, IEEE TEVC 23(5), 2019, supplementary S-I), as
// benchmarks/doc.py evaluates them (task 17, supplement 3, stage 3a): its
// scalar code one for one — a float's ** dm::pow (x ** 2 too, as there),
// math's sqrt, log, sin and cos, Python's sum term by term from 0, Python's
// min (the first of equal values; a NaN stays where it stands first). DOC2's
// products with g19's matrices are sums term by term from 0 as doc.py writes
// them out (_dot). doc.py stays the reference.
//
// The constraints, <= 0 satisfied: the inequalities, then each equality h
// as |h| - 1e-4 (the paper's Eq. 3; doc.constraints).
// ============================================================================

#include <cmath>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::doc {

constexpr double kEqTol = 1e-4;

inline int n_obj(int i) { return i >= 8 ? 3 : 2; }
inline int n_var(int i)
{
    constexpr int n[9] = {6, 16, 10, 8, 8, 11, 11, 10, 11};
    return n[i - 1];
}
// inequalities + equalities
inline int n_con(int i)
{
    constexpr int c[9] = {7, 7, 10, 6, 9, 10, 6, 7, 14};
    return c[i - 1];
}

// The box of DOC<i> (supplementary S-I): x1 (and DOC3's, DOC8's and DOC9's
// x2) in [0, 1], the rest the CEC 2006 problem's own (doc.BOUNDS)
inline Bounds bounds(int i)
{
    switch (i) {
        case 1: return {{0.0, 1.0}, {78.0, 102.0}, {33.0, 45.0}, {27.0, 45.0}, {27.0, 45.0}, {27.0, 45.0}};
        case 2: {
            Bounds b = box(16, 0.0, 10.0);
            b[0] = {0.0, 1.0};
            return b;
        }
        case 3: return {{0.0, 1.0}, {0.0, 1.0},   {0.0, 300.0}, {0.0, 100.0}, {0.0, 200.0},
                        {0.0, 100.0}, {0.0, 1.0}, {0.0, 100.0}, {0.0, 200.0}, {0.0, 0.03}};
        case 4:
        case 6: {
            Bounds b = box(i == 4 ? 8 : 11, -10.0, 10.0);
            b[0] = {0.0, 1.0};
            return b;
        }
        case 5: return {{0.0, 1.0},     {0.0, 1000.0}, {0.0, 40.0}, {0.0, 40.0},
                        {100.0, 300.0}, {6.3, 6.7},    {5.9, 6.4},  {4.5, 6.25}};
        case 7: {
            Bounds b = box(11, 0.0, 10.0);
            b[0] = {0.0, 1.0};
            return b;
        }
        case 8: return {{0.0, 1.0},     {0.0, 1.0},     {500.0, 1000.0}, {1000.0, 2000.0}, {5000.0, 6000.0},
                        {100.0, 500.0}, {100.0, 500.0}, {100.0, 500.0},  {100.0, 500.0},   {100.0, 500.0}};
        default: {                                         // 9
            Bounds b = box(11, -1.0, 10.0);
            b[0] = b[1] = {0.0, 1.0};
            return b;
        }
    }
}

// g19 of CEC 2006 (DOC2): A (10 x 5), C (5 x 5), D, E, B
constexpr double kG19A[10][5] = {{-16.0, 2.0, 0.0, 1.0, 0.0},  {0.0, -2.0, 0.0, 0.4, 2.0},
                                 {-3.5, 0.0, 2.0, 0.0, 0.0},   {0.0, -2.0, 0.0, -4.0, -1.0},
                                 {0.0, -9.0, -2.0, 1.0, -2.8}, {2.0, 0.0, -4.0, 0.0, 0.0},
                                 {-1.0, -1.0, -1.0, -1.0, -1.0}, {-1.0, -2.0, -3.0, -2.0, -1.0},
                                 {1.0, 2.0, 3.0, 4.0, 5.0},    {1.0, 1.0, 1.0, 1.0, 1.0}};
constexpr double kG19C[5][5] = {{30.0, -20.0, -10.0, 32.0, -10.0}, {-20.0, 39.0, -6.0, -31.0, 32.0},
                                {-10.0, -6.0, 10.0, -6.0, -10.0},  {32.0, -31.0, -6.0, 39.0, -20.0},
                                {-10.0, 32.0, -10.0, -20.0, 30.0}};
constexpr double kG19D[5] = {4.0, 8.0, 10.0, 6.0, 2.0};
constexpr double kG19E[5] = {-15.0, -27.0, -36.0, -18.0, -12.0};
constexpr double kG19B[10] = {-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0};
// g14 (DOC7)
constexpr double kG14C[10] = {-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708,
                              -26.662, -22.179};
// DOC2's circles: (p, q) on f2 = 1 - sqrt(f1), radius 0.15
constexpr double kCircles[3][2] = {{0.125, 0.6464466094067263},     // 1/8, 1 - sqrt(2)/4
                                   {0.5, 0.2928932188134524},       // 1/2, 1 - sqrt(2)/2
                                   {0.875, 0.06458565330651467}};   // 7/8, 1 - sqrt(14)/4

inline double sq(double v) { return dm::pow(v, 2.0); }

// Python's min of a and b in that order: b only where b < a
inline double py_min(double a, double b) { return b < a ? b : a; }

// f1 + f2 - 1 - |sin(10 pi (f1 - f2 + 1))|, DOC4-7's constraint
inline double sine(double f1, double f2)
{
    return f1 + f2 - 1.0 - std::abs(dm::sin(10.0 * kPi * (f1 - f2 + 1.0)));
}

// the constraints on the objectives, <= 0 satisfied (doc.objective_constraints);
// returns how many it wrote
inline int objective_constraints(int i, const double* f, double* out)
{
    const double f1 = f[0], f2 = f[1];
    switch (i) {
        case 1:
        case 9:
            out[0] = 1.0 - (sq(f1) + sq(f2));
            return 1;
        case 2: {
            out[0] = 1.0 - std::sqrt(f1) - f2;
            double best = 0.0;
            for (int c = 0; c < 3; ++c) {
                const double v = sq(f1 - kCircles[c][0]) + sq(f2 - kCircles[c][1]) - 0.0225;
                best = c == 0 ? v : py_min(best, v);
            }
            out[1] = best;
            return 2;
        }
        case 3: {
            const double d = f1 - f2;
            out[0] = 1.0 - (sq(f1) + sq(f2));
            out[1] = 0.1 - std::abs(d - 0.5);
            out[2] = 0.1 - std::abs(d);
            out[3] = 0.1 - std::abs(d + 0.5);
            return 4;
        }
        case 4:
            out[0] = 1.0 - f1 - f2;
            out[1] = -sine(f1, f2);
            return 2;
        case 5:
            out[0] = 1.0 - f1 - f2;
            out[1] = -sine(f1, f2);
            out[2] = (f1 - 0.8) * (f2 - 0.6);
            return 3;
        case 6:
            out[0] = 1.0 - f1 - f2;
            out[1] = -(f1 - 0.5) * sine(f1, f2);
            return 2;
        case 7:
            out[0] = 1.0 - f1 - f2;
            out[1] = -(f1 - 0.5) * sine(f1, f2);
            out[2] = 0.1 - std::abs(f1 - f2);
            return 3;
        default:                                          // 8
            out[0] = -(f[2] - 0.4) * (f[2] - 0.6);
            return 1;
    }
}

// f and the constraints of the point x (doc.evaluate, doc.constraints)
inline void evaluate(int i, const double* x, double* f, double* out)
{
    const double* X = x - 1;                             // X[k] is the paper's x_k
    double ineq[16], eq[5];
    int n_ineq = 0, n_eq = 0;
    switch (i) {
        case 1: {
            const double g = 5.3578547 * sq(X[4]) + 0.8356891 * X[2] * X[6] + 37.293239 * X[2]
                             - 10125.6023282166;
            f[0] = X[1];
            f[1] = g - std::sqrt(X[1]);
            const double a = 0.0056858 * X[3] * X[6] + 0.0006262 * X[2] * X[5] - 0.0022053 * X[4] * X[6];
            const double b = 0.0071317 * X[3] * X[6] + 0.0029955 * X[2] * X[3] + 0.0021813 * sq(X[4]);
            const double c = 0.0047026 * X[4] * X[6] + 0.0012547 * X[2] * X[4] + 0.0019085 * X[4] * X[5];
            n_ineq = objective_constraints(1, f, ineq);
            const double rest[6] = {85.334407 + a - 92.0, -85.334407 - a, 80.51249 + b - 110.0,
                                    -80.51249 - b + 90.0, 9.300961 + c - 25.0, -9.300961 - c + 20.0};
            for (double v : rest) ineq[n_ineq++] = v;
            break;
        }
        case 2: {
            const double* y = X + 12;                    // x12..x16
            const double* z = X + 2;                     // x2..x11
            // y C y: (y C)_k term by term, then its sum with y
            double yC[5], yCy = 0.0, d3 = 0.0, Bz = 0.0;
            for (int k = 0; k < 5; ++k) {
                double s = 0.0;
                for (int j = 0; j < 5; ++j) s = s + y[j] * kG19C[j][k];
                yC[k] = s;
            }
            for (int k = 0; k < 5; ++k) yCy = yCy + yC[k] * y[k];
            for (int k = 0; k < 5; ++k) d3 = d3 + kG19D[k] * dm::pow(y[k], 3.0);
            for (int j = 0; j < 10; ++j) Bz = Bz + kG19B[j] * z[j];
            const double g = yCy + 2.0 * d3 - Bz - 31.6555929502;
            f[0] = X[1];
            f[1] = g - dm::pow(X[1], 1.0 / 3.0);
            n_ineq = objective_constraints(2, f, ineq);
            for (int k = 0; k < 5; ++k) {
                double Cty = 0.0, zA = 0.0;
                for (int j = 0; j < 5; ++j) Cty = Cty + kG19C[j][k] * y[j];
                for (int j = 0; j < 10; ++j) zA = zA + z[j] * kG19A[j][k];
                ineq[n_ineq++] = -2.0 * Cty - 3.0 * kG19D[k] * (y[k] * y[k]) - kG19E[k] + zA;
            }
            break;
        }
        case 3: {
            const double g = -9.0 * X[6] - 15.0 * X[9] + 6.0 * X[2] + 16.0 * X[3] + 10.0 * (X[7] + X[8])
                             + 401.0551;
            f[0] = X[1];
            f[1] = g - X[1];
            n_ineq = objective_constraints(3, f, ineq);
            ineq[n_ineq++] = X[10] * X[4] + 0.02 * X[7] - 0.025 * X[6];
            ineq[n_ineq++] = X[10] * X[5] + 0.02 * X[8] - 0.015 * X[9];
            eq[n_eq++] = X[2] + X[3] - X[4] - X[5];
            eq[n_eq++] = 0.03 * X[2] + 0.01 * X[3] - X[10] * (X[4] + X[5]);
            eq[n_eq++] = X[4] + X[7] - X[6];
            eq[n_eq++] = X[5] + X[8] - X[9];
            break;
        }
        case 4: {
            const double g = sq(X[2] - 10.0) + 5.0 * sq(X[3] - 12.0) + dm::pow(X[4], 4.0)
                             + 3.0 * sq(X[5] - 11.0) + 10.0 * dm::pow(X[6], 6.0) + 7.0 * sq(X[7])
                             + dm::pow(X[8], 4.0) - 4.0 * X[7] * X[8] - 10.0 * X[7] - 8.0 * X[8]
                             - 679.6300573745;
            f[0] = X[1];
            f[1] = g - std::sqrt(X[1]);
            n_ineq = objective_constraints(4, f, ineq);
            ineq[n_ineq++] = -127.0 + 2.0 * sq(X[2]) + 3.0 * dm::pow(X[3], 4.0) + X[4] + 4.0 * sq(X[5])
                             + 5.0 * X[6];
            ineq[n_ineq++] = -282.0 + 7.0 * X[2] + 3.0 * X[3] + 10.0 * sq(X[4]) + X[5] - X[6];
            ineq[n_ineq++] = -196.0 + 23.0 * X[2] + sq(X[3]) + 6.0 * sq(X[7]) - 8.0 * X[8];
            ineq[n_ineq++] = 4.0 * sq(X[2]) + sq(X[3]) - 3.0 * X[2] * X[3] + 2.0 * sq(X[4])
                             + 5.0 * X[7] - 11.0 * X[8];
            break;
        }
        case 5: {
            const double g = X[2] - 192.724510070035;
            f[0] = X[1];
            f[1] = g - std::sqrt(X[1]);
            n_ineq = objective_constraints(5, f, ineq);
            ineq[n_ineq++] = -X[2] + 35.0 * dm::pow(X[3], 0.6) + 35.0 * dm::pow(X[4], 0.6);
            eq[n_eq++] = -300.0 * X[4] + 7500.0 * X[6] - 7500.0 * X[7] - 25.0 * X[5] * X[6]
                         + 25.0 * X[5] * X[7] + X[4] * X[5];
            eq[n_eq++] = 100.0 * X[3] + 155.365 * X[5] + 2500.0 * X[8] - X[3] * X[5] - 25.0 * X[5] * X[8]
                         - 15536.5;
            eq[n_eq++] = -X[6] + dm::log(-X[5] + 900.0);
            eq[n_eq++] = -X[7] + dm::log(X[5] + 300.0);
            eq[n_eq++] = -X[8] + dm::log(-2.0 * X[5] + 700.0);
            break;
        }
        case 6: {
            const double g = sq(X[2]) + sq(X[3]) + X[2] * X[3] - 14.0 * X[2] - 16.0 * X[3]
                             + sq(X[4] - 10.0) + 4.0 * sq(X[5] - 5.0) + sq(X[6] - 3.0)
                             + 2.0 * sq(X[7] - 1.0) + 5.0 * sq(X[8]) + 7.0 * sq(X[9] - 11.0)
                             + 2.0 * sq(X[10] - 10.0) + sq(X[11] - 7.0) + 21.693790931900001;
            f[0] = X[1];
            f[1] = g - std::sqrt(X[1]);
            n_ineq = objective_constraints(6, f, ineq);
            ineq[n_ineq++] = -105.0 + 4.0 * X[2] + 5.0 * X[3] - 3.0 * X[8] + 9.0 * X[9];
            ineq[n_ineq++] = 10.0 * X[2] - 8.0 * X[3] - 17.0 * X[8] + 2.0 * X[9];
            ineq[n_ineq++] = -8.0 * X[2] + 2.0 * X[3] + 5.0 * X[10] - 2.0 * X[11] - 12.0;
            ineq[n_ineq++] = 3.0 * sq(X[2] - 2.0) + 4.0 * sq(X[3] - 3.0) + 2.0 * sq(X[4])
                             - 7.0 * X[5] - 120.0;
            ineq[n_ineq++] = 5.0 * sq(X[2]) + 8.0 * X[3] + sq(X[4] - 6.0) - 2.0 * X[5] - 40.0;
            ineq[n_ineq++] = sq(X[2]) + 2.0 * sq(X[3] - 2.0) - 2.0 * X[2] * X[3] + 14.0 * X[6]
                             - 6.0 * X[7];
            ineq[n_ineq++] = 0.5 * sq(X[2] - 8.0) + 2.0 * sq(X[3] - 4.0) + 3.0 * sq(X[6]) - X[7] - 30.0;
            ineq[n_ineq++] = -3.0 * X[2] + 6.0 * X[3] + 12.0 * sq(X[10] - 8.0) - 7.0 * X[11];
            break;
        }
        case 7: {
            const double* y = X + 2;                     // x2..x11
            double s = 0.0, g = 0.0;
            for (int k = 0; k < 10; ++k) s = s + y[k];
            for (int k = 0; k < 10; ++k)
                if (y[k] > 0.0) g = g + y[k] * (kG14C[k] + dm::log(y[k] / s));
            g = g + 48.7648884;
            f[0] = X[1];
            f[1] = g - std::sqrt(X[1]);
            n_ineq = objective_constraints(7, f, ineq);
            eq[n_eq++] = X[2] + 2.0 * X[3] + 2.0 * X[4] + X[7] + X[11] - 2.0;
            eq[n_eq++] = X[5] + 2.0 * X[6] + X[7] + X[8] - 1.0;
            eq[n_eq++] = X[4] + X[8] + X[9] + 2.0 * X[10] + X[11] - 1.0;
            break;
        }
        case 8: {
            const double g = X[3] + X[4] + X[5] - 7048.2480205286;
            f[0] = X[1] * X[2] * g;
            f[1] = X[1] * (1.0 - X[2]) * g;
            f[2] = (1.0 - X[1]) * g;
            n_ineq = objective_constraints(8, f, ineq);
            ineq[n_ineq++] = -1.0 + 0.0025 * (X[6] + X[8]);
            ineq[n_ineq++] = -1.0 + 0.0025 * (X[7] + X[9] - X[6]);
            ineq[n_ineq++] = -1.0 + 0.01 * (X[10] - X[7]);
            ineq[n_ineq++] = -X[3] * X[8] + 833.33252 * X[6] + 100.0 * X[3] - 83333.333;
            ineq[n_ineq++] = -X[4] * X[9] + 1250.0 * X[7] + X[4] * X[6] - 1250.0 * X[6];
            ineq[n_ineq++] = -X[5] * X[10] + 1250000.0 + X[5] * X[7] - 2500.0 * X[7];
            break;
        }
        default: {                                       // 9
            const double g = -0.5 * (X[3] * X[6] - X[4] * X[5] + X[5] * X[11] - X[7] * X[11]
                                     + X[7] * X[10] - X[8] * X[9]) + 1.8660254038;
            const double a = 0.5 * kPi * X[1], b = 0.5 * kPi * X[2];
            f[0] = dm::cos(a) * dm::cos(b) * g;
            f[1] = dm::cos(a) * dm::sin(b) * g;
            f[2] = dm::sin(a) * g;
            n_ineq = objective_constraints(9, f, ineq);
            const double rest[13] = {
                sq(X[5]) + sq(X[6]) - 1.0,
                sq(X[11]) - 1.0,
                sq(X[7]) + sq(X[8]) - 1.0,
                sq(X[3]) + sq(X[4] - X[11]) - 1.0,
                sq(X[3] - X[7]) + sq(X[4] - X[8]) - 1.0,
                sq(X[3] - X[9]) + sq(X[4] - X[10]) - 1.0,
                sq(X[5] - X[7]) + sq(X[6] - X[8]) - 1.0,
                sq(X[5] - X[9]) + sq(X[6] - X[10]) - 1.0,
                sq(X[9]) + sq(X[10] - X[11]) - 1.0,
                X[4] * X[5] - X[3] * X[6],
                -X[5] * X[11],
                X[7] * X[11],
                X[8] * X[9] - X[7] * X[10]};
            for (double v : rest) ineq[n_ineq++] = v;
            break;
        }
    }
    int k = 0;
    for (int j = 0; j < n_ineq; ++j) out[k++] = ineq[j];
    for (int j = 0; j < n_eq; ++j) out[k++] = std::abs(eq[j]) - kEqTol;
}

}   // namespace mootation::suite::doc

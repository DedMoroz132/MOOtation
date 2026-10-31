#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The multimodal problems of the CEC 2019 competition (Liang, Qu, Gong & Yue,
// technical report, 2019, §1.3): SYM-PART simple and rotated, Omni-test and
// MMF1-15, as benchmarks/mmo.py evaluates them (task 17, supplement 3, stage
// 3a): its scalar code one for one — a float's ** dm::pow (x ** 2 too, as
// there), math's and NumPy's sin, cos, exp of one float dm::'s, Omni-test's
// sums term by term from 0. mmo.py stays the reference. pareto_set below is
// mmo.pareto_set: every part of the global Pareto set, its grids NumPy's
// linspace, _argmin's least value of g by the same grid and golden section.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::mmo {

enum class Which { SymSimple, SymRotated, Omni, MMF1, MMF1z, MMF1e, MMF2, MMF3, MMF4, MMF5, MMF6, MMF7, MMF8,
                   MMF9, MMF10, MMF11, MMF12, MMF13, MMF14, MMF14a, MMF15, MMF15a };

inline bool parse(const std::string& f, Which& w)
{
    static const struct { const char* name; Which w; } names[] = {
        {"SYM-PART_simple", Which::SymSimple}, {"SYM-PART_rotated", Which::SymRotated},
        {"Omni-test", Which::Omni}, {"MMF1", Which::MMF1}, {"MMF1_z", Which::MMF1z},
        {"MMF1_e", Which::MMF1e}, {"MMF2", Which::MMF2}, {"MMF3", Which::MMF3}, {"MMF4", Which::MMF4},
        {"MMF5", Which::MMF5}, {"MMF6", Which::MMF6}, {"MMF7", Which::MMF7}, {"MMF8", Which::MMF8},
        {"MMF9", Which::MMF9}, {"MMF10", Which::MMF10}, {"MMF11", Which::MMF11}, {"MMF12", Which::MMF12},
        {"MMF13", Which::MMF13}, {"MMF14", Which::MMF14}, {"MMF14_a", Which::MMF14a},
        {"MMF15", Which::MMF15}, {"MMF15_a", Which::MMF15a}};
    for (const auto& e : names)
        if (f == e.name) {
            w = e.w;
            return true;
        }
    return false;
}

inline int n_obj(Which w) { return w >= Which::MMF14 ? 3 : 2; }

// the number of variables (Omni-test: any)
inline int n_var(Which w)
{
    if (w == Which::Omni) return 0;
    return w == Which::MMF13 || w >= Which::MMF14 ? 3 : 2;
}

constexpr double kA = 1.0, kB = 10.0, kC = 8.0;      // SYM-PART
constexpr double kCosOmega = 0.7071067811865476;   // math.cos(0.25 pi)
constexpr double kSinOmega = 0.7071067811865476;   // math.sin(0.25 pi)
constexpr double kKz = 3.0;                        // MMF1_z
constexpr double kAe = 2.718281828459045;          // MMF1_e: math.e
constexpr int kNp = 2;                             // MMF9-15
constexpr int kQ = 4;                              // MMF12
constexpr double kMinus2Ln2 = -1.3862943611198906; // -2.0 * math.log(2.0)
constexpr double kSqrt2 = 1.4142135623730951;      // math.sqrt(2.0)

// The box of the report's problem (§1.3; mmo.BOUNDS; Omni-test: n variables
// in [0, 6]). MMF1_e's x2 in [-a^3, a^3] with a = e: math.e ** 3 as CPython
// computes it (the C library's pow), a unit in the last place under e^3
// rounded (20.085536923187668), so that both engines draw the same points.
inline Bounds bounds(Which w, int n)
{
    constexpr double kAe3 = 20.085536923187664;
    switch (w) {
        case Which::SymSimple:
        case Which::SymRotated: return box(2, -20.0, 20.0);
        case Which::Omni: return box(n, 0.0, 6.0);
        case Which::MMF1:
        case Which::MMF1z:
        case Which::MMF7: return {{1.0, 3.0}, {-1.0, 1.0}};
        case Which::MMF1e: return {{1.0, 3.0}, {-kAe3, kAe3}};
        case Which::MMF2: return {{0.0, 1.0}, {0.0, 2.0}};
        case Which::MMF3: return {{0.0, 1.0}, {0.0, 1.5}};
        case Which::MMF4: return {{-1.0, 1.0}, {0.0, 2.0}};
        case Which::MMF5: return {{1.0, 3.0}, {-1.0, 3.0}};
        case Which::MMF6: return {{1.0, 3.0}, {-1.0, 2.0}};
        case Which::MMF8: return {{-kPi, kPi}, {0.0, 9.0}};
        case Which::MMF9:
        case Which::MMF10:
        case Which::MMF11: return box(2, 0.1, 1.1);
        case Which::MMF12: return box(2, 0.0, 1.0);
        case Which::MMF13: return box(3, 0.1, 1.1);
        case Which::MMF14:
        case Which::MMF14a:
        case Which::MMF15:
        case Which::MMF15a: return box(3, 0.0, 1.0);
    }
    return {};
}

inline double sq(double v) { return dm::pow(v, 2.0); }

// v ** 2 of a float (pow) or, rows = true, of an array (NumPy's v * v): the
// g's below are called on one float (evaluate, the golden section of
// _argmin) and on a grid of them (_argmin's first step)
inline double sq(double v, bool rows) { return rows ? v * v : dm::pow(v, 2.0); }

// the g of MMF9-15 (mmo._g9 .. _g15)
inline double g9(double x) { return 2.0 - dm::pow(dm::sin(kNp * kPi * x), 6.0); }
inline double g10(double x, bool rows = false)
{
    return 2.0 - dm::exp(-sq((x - 0.2) / 0.004, rows)) - 0.8 * dm::exp(-sq((x - 0.6) / 0.4, rows));
}
inline double g11(double x, bool rows = false)
{
    return 2.0 - dm::exp(kMinus2Ln2 * sq((x - 0.1) / 0.8, rows)) * dm::pow(dm::sin(kNp * kPi * x), 6.0);
}
inline double g14(double x) { return 2.0 - sq(dm::sin(kNp * kPi * x)); }
inline double g15(double x, bool rows = false)
{
    return 2.0 - dm::exp(kMinus2Ln2 * sq((x - 0.1) / 0.8, rows)) * sq(dm::sin(kNp * kPi * x), rows);
}

// t_i = sgn(t^) min(|t^|, 1), t^ = sgn(v) ceil((|v| - half) / period)
inline double sym_t(double v, double half, double period)
{
    const double th = v != 0.0 ? std::copysign(1.0, v) * std::ceil((std::abs(v) - half) / period) : 0.0;
    if (th == 0.0) return 0.0;
    const double a = std::abs(th);
    return std::copysign(1.0 < a ? 1.0 : a, th);
}

inline void sym_part(double r1, double r2, double* f)
{
    const double t1 = sym_t(r1, kA + kC / 2.0, 2.0 * kA + kC);
    const double t2 = sym_t(r2, kB / 2.0, kB);
    const double p1 = r1 - t1 * (kC + 2.0 * kA), p2 = r2 - t2 * kB;
    f[0] = sq(p1 + kA) + sq(p2);
    f[1] = sq(p1 - kA) + sq(p2);
}

// f of the point x of n variables (mmo.evaluate)
inline void evaluate(Which w, const double* x, int n, double* f)
{
    switch (w) {
        case Which::SymSimple:
            sym_part(x[0], x[1], f);
            return;
        case Which::SymRotated:
            sym_part(kCosOmega * x[0] - kSinOmega * x[1], kSinOmega * x[0] + kCosOmega * x[1], f);
            return;
        case Which::Omni: {
            double s = 0.0, c = 0.0;
            for (int i = 0; i < n; ++i) s = s + dm::sin(kPi * x[i]);
            for (int i = 0; i < n; ++i) c = c + dm::cos(kPi * x[i]);
            f[0] = s;
            f[1] = c;
            return;
        }
        case Which::MMF1:
        case Which::MMF1z:
        case Which::MMF1e:
        case Which::MMF5:
        case Which::MMF6:
        case Which::MMF7: {
            const double d = std::abs(x[0] - 2.0);
            if (w == Which::MMF7) {
                const double s = (0.3 * d * d * dm::cos(24.0 * kPi * d + 4.0 * kPi) + 0.6 * d)
                                 * dm::sin(6.0 * kPi * d + kPi);
                f[0] = d;
                f[1] = 1.0 - std::sqrt(d) + sq(x[1] - s);
                return;
            }
            double s = dm::sin(6.0 * kPi * d + kPi);
            if (w == Which::MMF1z && x[0] < 2.0) s = dm::sin(2.0 * kKz * kPi * d + kPi);
            else if (w == Which::MMF1z) s = dm::sin(2.0 * kPi * d + kPi);
            else if (w == Which::MMF1e && x[0] >= 2.0) s = dm::pow(kAe, x[0]) * s;
            double y = x[1];
            if (w == Which::MMF5 && y > 1.0) y -= 2.0;
            else if (w == Which::MMF6 && y > 1.0) y -= 1.0;
            f[0] = d;
            f[1] = 1.0 - std::sqrt(d) + 2.0 * sq(y - s);
            return;
        }
        case Which::MMF2:
        case Which::MMF3: {
            const double x1 = x[0], x2 = x[1];
            bool first = false;
            double shift = 0.0;
            if (w == Which::MMF2) {
                first = x2 <= 1.0;
                shift = 1.0;
            } else {
                first = x2 <= 0.5 || (0.5 < x2 && x2 < 1.0 && 0.25 < x1);
                shift = 0.5;
            }
            const double u = x2 - std::sqrt(x1) - (first ? 0.0 : shift);
            const double ccoef = first ? 2.0 : 1.0;
            f[0] = x1;
            f[1] = 1.0 - std::sqrt(x1) + 2.0 * (4.0 * u * u - ccoef * dm::cos(20.0 * u * kPi / kSqrt2) + 2.0);
            return;
        }
        case Which::MMF4: {
            const double y = x[1] >= 1.0 ? x[1] - 1.0 : x[1];
            f[0] = std::abs(x[0]);
            f[1] = 1.0 - x[0] * x[0] + 2.0 * sq(y - dm::sin(kPi * std::abs(x[0])));
            return;
        }
        case Which::MMF8: {
            const double sa = dm::sin(std::abs(x[0]));
            const double y = x[1] > 4.0 ? x[1] - 4.0 : x[1];
            f[0] = sa;
            f[1] = std::sqrt(1.0 - sa * sa) + 2.0 * sq(y - sa - std::abs(x[0]));
            return;
        }
        case Which::MMF9:
        case Which::MMF10:
        case Which::MMF11: {
            const double g = w == Which::MMF9 ? g9(x[1]) : w == Which::MMF10 ? g10(x[1]) : g11(x[1]);
            f[0] = x[0];
            f[1] = g / x[0];
            return;
        }
        case Which::MMF12: {
            const double g = g11(x[1]), f1 = x[0];
            const double h = 1.0 - sq(f1 / g) - (f1 / g) * dm::sin(2.0 * kPi * kQ * f1);
            f[0] = f1;
            f[1] = g * h;
            return;
        }
        case Which::MMF13: {
            const double t = x[1] + std::sqrt(x[2]);
            f[0] = x[0];
            f[1] = g11(t) / x[0];
            return;
        }
        case Which::MMF14:
        case Which::MMF14a:
        case Which::MMF15:
        case Which::MMF15a: {
            const double x2 = x[1], x3 = x[2];
            const double shifted = x3 - 0.5 * dm::sin(kPi * x2) + 1.0 / (2 * kNp);
            const double g = w == Which::MMF14  ? g14(x3)
                           : w == Which::MMF14a ? g14(shifted)
                           : w == Which::MMF15  ? g15(x3)
                                                : g15(shifted);
            const double a = 0.5 * kPi * x[0], b = 0.5 * kPi * x2;
            f[0] = dm::cos(a) * dm::cos(b) * (1.0 + g);
            f[1] = dm::cos(a) * dm::sin(b) * (1.0 + g);
            f[2] = dm::sin(a) * (1.0 + g);
            return;
        }
    }
}

// ---- the global Pareto sets (mmo.pareto_set) -------------------------------

// Where g is least on [lo, hi] (mmo._argmin): its least value on a grid of
// 200 001 points (the first of equal ones), then golden section around it
// until the bracket is below 1e-16, 200 steps at most.
inline double argmin(double (*g)(double, bool), double lo, double hi)
{
    const std::vector<double> t = linspace(lo, hi, 200001);
    std::size_t k = 0;
    double best = g(t[0], true);
    for (std::size_t i = 1; i < t.size() && !std::isnan(best); ++i) {
        const double v = g(t[i], true);
        if (v < best || std::isnan(v)) {
            best = v;
            k = i;
        }
    }
    double a = t[k > 0 ? k - 1 : 0], b = t[k + 1 < t.size() ? k + 1 : t.size() - 1];
    constexpr double r = 0.6180339887498949;          // (math.sqrt(5.0) - 1.0) / 2.0
    for (int i = 0; i < 200; ++i) {
        const double c = b - r * (b - a), d = a + r * (b - a);
        if (g(c, false) < g(d, false)) b = d;
        else a = c;
        if (b - a < 1e-16) break;
    }
    return 0.5 * (a + b);
}

// n points split as evenly as can be over k parts (mmo._even)
inline int even(int n, int k, int i) { return n / k + (i < n % k ? 1 : 0); }

// The rows of the global Pareto set, about n of them, every part (mmo.
// pareto_set), flat, n_var values a row; empty for none
inline std::vector<double> pareto_set(Which w, int n)
{
    std::vector<double> out;
    auto row2 = [&](double a, double b) { out.push_back(a); out.push_back(b); };
    auto row3 = [&](double a, double b, double c) { out.push_back(a); out.push_back(b); out.push_back(c); };
    switch (w) {
        case Which::SymSimple:
        case Which::SymRotated: {
            int part = 0;
            for (int t1 = -1; t1 <= 1; ++t1)
                for (int t2 = -1; t2 <= 1; ++t2) {
                    const int m = even(n, 9, part++);
                    for (double p1 : linspace(-kA, kA, m > 2 ? m : 2)) {
                        const double r1 = t1 * (kC + 2.0 * kA) + p1, r2 = t2 * kB;
                        if (w == Which::SymRotated) row2(kCosOmega * r1 + kSinOmega * r2, -kSinOmega * r1 + kCosOmega * r2);
                        else row2(r1, r2);
                    }
                }
            return out;
        }
        case Which::Omni: {
            int part = 0;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int c = 0; c < 3; ++c) {
                        const int k = even(n, 27, part++);
                        for (double u : linspace(1.0, 1.5, k > 2 ? k : 2)) row3(2.0 * a + u, 2.0 * b + u, 2.0 * c + u);
                    }
            return out;
        }
        case Which::MMF1:
        case Which::MMF1z:
        case Which::MMF1e:
        case Which::MMF7:
            for (double x1 : linspace(1.0, 3.0, n > 3 ? n : 3)) {
                const double d = std::abs(x1 - 2.0);
                double x2 = 0.0;
                if (w == Which::MMF7) {
                    x2 = (0.3 * d * d * dm::cos(24.0 * kPi * d + 4.0 * kPi) + 0.6 * d) * dm::sin(6.0 * kPi * d + kPi);
                } else {
                    x2 = dm::sin(6.0 * kPi * d + kPi);
                    if (w == Which::MMF1z)
                        x2 = x1 < 2.0 ? dm::sin(2.0 * kKz * kPi * d + kPi) : dm::sin(2.0 * kPi * d + kPi);
                    else if (w == Which::MMF1e && x1 >= 2.0)
                        x2 = dm::pow(kAe, x1) * x2;
                }
                row2(x1, x2);
            }
            return out;
        case Which::MMF2:
        case Which::MMF3:
            for (double x2 : linspace(0.0, 1.0, n > 2 ? n : 2))
                if (w == Which::MMF2 || x2 < 1.0) row2(x2 * x2, x2);   // MMF3: x2 = 1 is the second branch's
            return out;
        case Which::MMF4: {
            const int k = n / 4 > 2 ? n / 4 : 2;
            for (int up = 0; up <= 1; ++up)
                for (double sign : {1.0, -1.0})
                    for (double a : linspace(0.0, 1.0, k)) {
                        const double x2 = dm::sin(kPi * a) + up;
                        if (up == 0 ? x2 < 1.0 : x2 >= 1.0) row2(sign * a, x2);
                    }
            return out;
        }
        case Which::MMF5:
        case Which::MMF6: {
            const std::vector<double> x1 = linspace(1.0, 3.0, n / 2 > 3 ? n / 2 : 3);
            std::vector<double> s;
            for (double v : x1) s.push_back(dm::sin(6.0 * kPi * std::abs(v - 2.0) + kPi));
            for (std::size_t i = 0; i < x1.size(); ++i) row2(x1[i], s[i]);
            for (std::size_t i = 0; i < x1.size(); ++i) {
                const double up = s[i] + (w == Which::MMF5 ? 2.0 : 1.0);
                if (up > 1.0) row2(x1[i], up);
            }
            return out;
        }
        case Which::MMF8: {
            const std::vector<double> x1 = linspace(-kPi, kPi, n / 2 > 3 ? n / 2 : 3);
            std::vector<double> x2;
            for (double v : x1) x2.push_back(dm::sin(std::abs(v)) + std::abs(v));
            for (std::size_t i = 0; i < x1.size(); ++i) row2(x1[i], x2[i]);
            for (std::size_t i = 0; i < x1.size(); ++i)
                if (x2[i] + 4.0 > 4.0) row2(x1[i], x2[i] + 4.0);
            return out;
        }
        case Which::MMF9:
        case Which::MMF10:
        case Which::MMF11: {
            const int k = w == Which::MMF9 ? n / kNp : n;
            const std::vector<double> x1 = linspace(0.1, 1.1, k > 2 ? k : 2);
            if (w == Which::MMF9) {
                for (int i = 0; i < kNp; ++i)
                    for (double v : x1) row2(v, (0.5 + i) / kNp);
                return out;
            }
            const double t = w == Which::MMF10 ? argmin(g10, 0.1, 1.1) : argmin(g11, 0.1, 1.1);
            for (double v : x1) row2(v, t);
            return out;
        }
        case Which::MMF12: {
            const double t = argmin(g11, 0.0, 1.0), g = g11(t);
            // the f1 of the non-dominated part of f2 = g h(f1, g) (mmo._mmf12_pieces)
            const std::vector<double> f1 = linspace(0.0, 1.0, 200001);
            std::vector<double> kept;
            double best = 0.0;
            for (std::size_t i = 0; i < f1.size(); ++i) {
                const double u = f1[i] / g;
                const double f2 = g * (1.0 - u * u - u * dm::sin(2.0 * kPi * kQ * f1[i]));
                if (i == 0 || f2 < best) kept.push_back(f1[i]);
                best = i == 0 ? f2 : nan_min(best, f2);
            }
            for (double v : linspace(0.0, static_cast<double>(kept.size() - 1), n > 2 ? n : 2))
                row2(kept[static_cast<std::size_t>(v)], t);
            return out;
        }
        case Which::MMF13: {
            const double t = argmin(g11, 0.1 + std::sqrt(0.1), 1.1 + std::sqrt(1.1));
            const int k = std::max(static_cast<int>(std::sqrt(static_cast<double>(n))), 2);
            const double top = std::min(1.1, sq(t - 0.1));
            const std::vector<double> a = linspace(0.1, 1.1, k), b = linspace(0.1, top, k);
            for (double x3 : b)
                for (double x1 : a) {
                    const double x2 = t - std::sqrt(x3);
                    row3(x1, std::isnan(x2) ? x2 : (x2 > 0.1 ? x2 : 0.1), x3);   // np.maximum(.., 0.1)
                }
            return out;
        }
        case Which::MMF14:
        case Which::MMF14a:
        case Which::MMF15:
        case Which::MMF15a: {
            const int parts = w == Which::MMF14 || w == Which::MMF14a ? kNp : 1;
            const int k = std::max(static_cast<int>(std::sqrt(static_cast<double>(n) / parts)), 2);
            const std::vector<double> a = linspace(0.0, 1.0, k);
            const double t = w == Which::MMF15 || w == Which::MMF15a ? argmin(g15, 0.0, 1.0) : 0.0;
            for (int i = 0; i < parts; ++i)
                for (double x2 : a)
                    for (double x1 : a) {
                        double x3 = 0.0;
                        if (w == Which::MMF14) x3 = (0.5 + i) / kNp;
                        else if (w == Which::MMF14a) x3 = 0.5 * dm::sin(kPi * x2) + static_cast<double>(i) / kNp;
                        else if (w == Which::MMF15) x3 = t;
                        else x3 = 0.5 * dm::sin(kPi * x2) + t - 1.0 / (2 * kNp);
                        if (x3 >= 0.0 && x3 <= 1.0) row3(x1, x2, x3);
                    }
            return out;
        }
    }
    return out;
}

}   // namespace mootation::suite::mmo

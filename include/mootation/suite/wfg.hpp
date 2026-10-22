#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// WFG1-9 (Huband, Hingston, Barone & While, IEEE TEVC 10(5), 2006, Table
// XIV), as benchmarks/wfg.py evaluates them (task 17, supplement 1, stage 1):
// its scalar kernels one for one, every operation in their order — a float's
// ** pow (dm::pow; x ** 2 is x * x, as there), math's sin and cos dm::'s,
// floor, abs, min and max as Python's, the sums term by term from 0, and
// NumPy's cumsum of WFG7-9 the same running sum. wfg.py stays the reference.
//
// k is the number of position variables (2 (M - 1) for WFG; M - 1 for MaF10-12,
// which are WFG1, WFG2 and WFG9 with that k and l = 10), n = k + l.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::wfg {

using Vec = std::vector<double>;

inline double s_linear(double y, double A)
{
    return std::abs(y - A) / std::abs(std::floor(A - y) + A);
}

inline double s_decept(double y, double A, double B, double C)
{
    const double inner = std::abs(y - A) - B;
    const double t1 = std::floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B);
    const double t2 = std::floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B);
    return 1.0 + inner * (t1 + t2 + 1.0 / B);
}

inline double s_multi(double y, int A, double B, double C)
{
    const double denom = 2.0 * (std::floor(C - y) + C);
    const double u = std::abs(y - C) / denom;
    const double num = 1.0 + dm::cos((4.0 * A + 2.0) * kPi * (0.5 - u)) + 4.0 * B * u * u;
    return num / (B + 2.0);
}

inline double b_flat(double y, double A, double B, double C)
{
    const double t1 = std::min(0.0, std::floor(y - B)) * (A * (B - y) / B);
    const double t2 = std::min(0.0, std::floor(C - y)) * ((1.0 - A) * (y - C) / (1.0 - C));
    const double v = A + t1 - t2;
    return std::max(0.0, std::min(1.0, v));
}

inline double b_param(double y, double u, double A, double B, double C)
{
    const double v = A - (1.0 - 2.0 * u) * std::abs(std::floor(0.5 - u) + A);
    return dm::pow(y, B + (C - B) * v);
}

// r_nonsep of the n values t[0..n) with degree A
inline double r_nonsep(const double* t, int n, int A)
{
    const double ceil_A_half = std::ceil(A / 2.0);
    double total = 0.0;
    for (int j = 0; j < n; ++j) {
        double s = t[j];
        for (int kk = 0; kk < A - 1; ++kk) s += std::abs(t[j] - t[(j + kk + 1) % n]);
        total += s;
    }
    const double denom = (static_cast<double>(n) / A) * ceil_A_half * (1.0 + 2.0 * A - 2.0 * ceil_A_half);
    return total / denom;
}

// r_sum over the M - 1 chunks of the k positions and over the rest; w empty:
// uniform weights (wfg.py's _UNIF)
inline Vec r_sum_chunks(const Vec& t, int k, int M, const Vec& w)
{
    const int chunk = k / (M - 1), len = static_cast<int>(t.size());
    Vec out(static_cast<std::size_t>(M));
    auto part = [&](int lo, int hi) {
        if (w.empty()) {
            double s = 0.0;
            for (int i = lo; i < hi; ++i) s += t[static_cast<std::size_t>(i)];
            return s / (hi - lo);
        }
        double sw = 0.0, s = 0.0;
        for (int i = lo; i < hi; ++i) {
            s += w[static_cast<std::size_t>(i)] * t[static_cast<std::size_t>(i)];
            sw += w[static_cast<std::size_t>(i)];
        }
        return s / sw;
    };
    for (int m = 0; m < M - 1; ++m) out[static_cast<std::size_t>(m)] = part(m * chunk, m * chunk + chunk);
    out[static_cast<std::size_t>(M - 1)] = part(k, len);
    return out;
}

inline Vec r_nonsep_chunks(const Vec& t, int A, int k, int M, int l)
{
    const int chunk = k / (M - 1);
    Vec out(static_cast<std::size_t>(M));
    for (int m = 0; m < M - 1; ++m) out[static_cast<std::size_t>(m)] = r_nonsep(t.data() + m * chunk, chunk, A);
    out[static_cast<std::size_t>(M - 1)] = r_nonsep(t.data() + k, static_cast<int>(t.size()) - k, l);
    return out;
}

// the shapes of the M objectives from the M - 1 values x
inline Vec shape_linear(const Vec& x)
{
    const int M = static_cast<int>(x.size()) + 1;
    Vec h(static_cast<std::size_t>(M));
    double cp = 1.0;
    for (double v : x) cp *= v;
    h[0] = cp;
    for (int m = 1; m < M - 1; ++m) {
        double acc = 1.0;
        for (int i = 0; i < M - m - 1; ++i) acc *= x[static_cast<std::size_t>(i)];
        h[static_cast<std::size_t>(m)] = acc * (1.0 - x[static_cast<std::size_t>(M - m - 1)]);
    }
    h[static_cast<std::size_t>(M - 1)] = 1.0 - x[0];
    return h;
}

inline Vec shape_convex(const Vec& x)
{
    const int M = static_cast<int>(x.size()) + 1;
    Vec s(x.size()), omc(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        s[i] = dm::sin(kPi * x[i] / 2.0);
        omc[i] = 1.0 - dm::cos(kPi * x[i] / 2.0);
    }
    Vec h(static_cast<std::size_t>(M));
    double cp = 1.0;
    for (double v : omc) cp *= v;
    h[0] = cp;
    for (int m = 1; m < M - 1; ++m) {
        double acc = 1.0;
        for (int i = 0; i < M - m - 1; ++i) acc *= omc[static_cast<std::size_t>(i)];
        h[static_cast<std::size_t>(m)] = acc * (1.0 - s[static_cast<std::size_t>(M - m - 1)]);
    }
    h[static_cast<std::size_t>(M - 1)] = 1.0 - s[0];
    return h;
}

inline Vec shape_concave(const Vec& x)
{
    const int M = static_cast<int>(x.size()) + 1;
    Vec s(x.size()), c(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        s[i] = dm::sin(kPi * x[i] / 2.0);
        c[i] = dm::cos(kPi * x[i] / 2.0);
    }
    Vec h(static_cast<std::size_t>(M));
    double cp = 1.0;
    for (double v : s) cp *= v;
    h[0] = cp;
    for (int m = 1; m < M - 1; ++m) {
        double acc = 1.0;
        for (int i = 0; i < M - m - 1; ++i) acc *= s[static_cast<std::size_t>(i)];
        h[static_cast<std::size_t>(m)] = acc * c[static_cast<std::size_t>(M - m - 1)];
    }
    h[static_cast<std::size_t>(M - 1)] = c[0];
    return h;
}

// x of the shapes from t; degenerate: A_1 = 1, the others 0 (WFG3), else all 1
inline Vec calc_x(const Vec& t, bool degenerate)
{
    const std::size_t M = t.size();
    const double tM = t[M - 1];
    Vec x(M);
    for (std::size_t i = 0; i + 1 < M; ++i) {
        const double A = degenerate && i > 0 ? 0.0 : 1.0;
        x[i] = std::max(tM, A) * (t[i] - 0.5) + 0.5;
    }
    x[M - 1] = tM;
    return x;
}

inline void final_values(const Vec& x, const Vec& h, int M, double* f)
{
    for (int m = 0; m < M; ++m) f[m] = x[static_cast<std::size_t>(M - 1)] + 2.0 * (m + 1) * h[static_cast<std::size_t>(m)];
}

inline Vec head(const Vec& x)
{
    return Vec(x.begin(), x.end() - 1);
}

// np.cumsum(y[::-1])[::-1]: cs[i] = y[n-1] + ... + y[i], added from the end
inline Vec tail_sums(const Vec& y)
{
    Vec cs(y);
    for (std::size_t i = y.size() - 1; i-- > 0;) cs[i] = cs[i + 1] + y[i];
    return cs;
}

// np.cumsum(y): cs[i] = y[0] + ... + y[i]
inline Vec head_sums(const Vec& y)
{
    Vec cs(y);
    for (std::size_t i = 1; i < y.size(); ++i) cs[i] = cs[i - 1] + y[i];
    return cs;
}

// b_param's constants as WFG7-9 call it
constexpr double kBpA = 0.98 / 49.98, kBpB = 0.02, kBpC = 50.0;

// WFG<which> of the n variables z, M objectives, k positions; f M values
inline void evaluate(int which, const double* z, int n, int M, int k, double* f)
{
    const int l = n - k;
    Vec y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(i)] = z[i] / (2.0 * (i + 1));
    const std::size_t uk = static_cast<std::size_t>(k), un = static_cast<std::size_t>(n);
    switch (which) {
        case 1: {
            Vec t(y);
            for (std::size_t i = uk; i < un; ++i) t[i] = s_linear(y[i], 0.35);
            for (std::size_t i = uk; i < un; ++i) t[i] = b_flat(t[i], 0.8, 0.75, 0.85);
            for (double& v : t) v = dm::pow(v, 0.02);
            Vec w(un);
            for (std::size_t i = 0; i < un; ++i) w[i] = 2.0 * static_cast<double>(i + 1);
            const Vec x = calc_x(r_sum_chunks(t, k, M, w), false);
            Vec h = shape_convex(head(x));
            const double x1 = x[0];
            // (...) ** 1.0 there: Python's pow gives the value itself
            h[static_cast<std::size_t>(M - 1)] =
                1.0 - x1 - dm::cos(10.0 * kPi * x1 + kPi / 2.0) / (10.0 * kPi);
            final_values(x, h, M, f);
            return;
        }
        case 2:
        case 3: {
            Vec t1(y);
            for (std::size_t i = uk; i < un; ++i) t1[i] = s_linear(y[i], 0.35);
            Vec t2(uk + static_cast<std::size_t>(l / 2));
            for (std::size_t i = 0; i < uk; ++i) t2[i] = t1[i];
            for (int i = 0; i < l / 2; ++i) t2[uk + static_cast<std::size_t>(i)] = r_nonsep(t1.data() + k + 2 * i, 2, 2);
            const Vec x = calc_x(r_sum_chunks(t2, k, M, Vec{}), which == 3);
            if (which == 3) {
                final_values(x, shape_linear(head(x)), M, f);
                return;
            }
            Vec h = shape_convex(head(x));
            const double x1 = x[0];
            h[static_cast<std::size_t>(M - 1)] = 1.0 - x1 * dm::pow(dm::cos(5.0 * kPi * x1), 2.0);
            final_values(x, h, M, f);
            return;
        }
        case 4:
        case 5: {
            Vec t(un);
            for (std::size_t i = 0; i < un; ++i)
                t[i] = which == 4 ? s_multi(y[i], 30, 10.0, 0.35) : s_decept(y[i], 0.35, 0.001, 0.05);
            const Vec x = calc_x(r_sum_chunks(t, k, M, Vec{}), false);
            final_values(x, shape_concave(head(x)), M, f);
            return;
        }
        case 6: {
            Vec t(y);
            for (std::size_t i = uk; i < un; ++i) t[i] = s_linear(y[i], 0.35);
            const Vec x = calc_x(r_nonsep_chunks(t, k / (M - 1), k, M, l), false);
            final_values(x, shape_concave(head(x)), M, f);
            return;
        }
        case 7:
        case 8: {
            Vec t(y);
            if (which == 7) {
                const Vec cs = tail_sums(y);                 // the mean of y_{i+1..n} at i
                for (std::size_t i = 0; i < uk; ++i)
                    t[i] = b_param(y[i], cs[i + 1] / static_cast<double>(n - 1 - static_cast<int>(i)), kBpA, kBpB, kBpC);
            } else {
                const Vec cs = head_sums(y);                 // the mean of y_1..y_{i-1} at i
                for (std::size_t i = uk; i < un; ++i)
                    t[i] = b_param(y[i], cs[i - 1] / static_cast<double>(i), kBpA, kBpB, kBpC);
            }
            for (std::size_t i = uk; i < un; ++i) t[i] = s_linear(t[i], 0.35);
            const Vec x = calc_x(r_sum_chunks(t, k, M, Vec{}), false);
            final_values(x, shape_concave(head(x)), M, f);
            return;
        }
        case 9: {
            const Vec cs = tail_sums(y);
            Vec t1(y);
            for (std::size_t i = 0; i + 1 < un; ++i)
                t1[i] = b_param(y[i], cs[i + 1] / static_cast<double>(n - 1 - static_cast<int>(i)), kBpA, kBpB, kBpC);
            Vec t2(un);
            for (std::size_t i = 0; i < uk; ++i) t2[i] = s_decept(t1[i], 0.35, 0.001, 0.05);
            for (std::size_t i = uk; i < un; ++i) t2[i] = s_multi(t1[i], 30, 95.0, 0.35);
            const Vec x = calc_x(r_nonsep_chunks(t2, k / (M - 1), k, M, l), false);
            final_values(x, shape_concave(head(x)), M, f);
            return;
        }
        default:
            return;
    }
}

}   // namespace mootation::suite::wfg

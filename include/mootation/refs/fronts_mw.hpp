#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The reference fronts and Pareto sets of MW1-14 (Ma & Wang, IEEE TEVC 23(6),
// 2019) in C++ — task 17, §2.1: benchmarks/mw.py's THE FRONTS, function for
// function (named beside each), the constraints the suite's
// (suite/mw.hpp's constraints, in objective space). mw.py stays the
// cross-check.
//
//   MW4      the simplex sum f = 1; MW8 the feasible part of the sphere;
//   MW14     the left records of h(t) = 6 - e^t - 1.5 sin(1.1 pi t^2), drawn
//            (the library's generator), and the corners;
//   the others (two objectives): the least feasible reachable f2 at 20 001
//            values of f1 — of the angle for MW5-7, scanned in f2 for MW1
//            and MW3 — refined where neighbours lie far apart or one has a
//            point and the other none, nondominated (dense_front), then about
//            n of them evenly by length along each piece (_spread).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "../run_metrics.hpp"
#include "../suite/mw.hpp"
#include "sample.hpp"

namespace mootation::refs::mw {

constexpr std::size_t kResolution = 20001, kScanResolution = 5001;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

inline bool radial(int i) { return i >= 5 && i <= 7; }
inline bool scanned(int i) { return i == 1 || i == 3; }

// mw.upper: x1's bound
inline double upper(int i)
{
    switch (i) {
        case 6: return 1.1;
        case 11: return std::sqrt(2.0);
        case 13:
        case 14: return 1.5;
        default: return 1.0;
    }
}

// mw.F1_RANGE
inline double f1_range(int i)
{
    switch (i) {
        case 9:
        case 10:
        case 12: return 2.0;
        case 11:
        case 13: return 2.5;
        default: return 1.0;
    }
}

// mw._feasible: every constraint of the objective vector f <= tol
inline bool feasible(int i, const double* f, int m, double tol = 0.0)
{
    double c[4] = {0.0, 0.0, 0.0, 0.0};
    suite::mw::constraints(i, f, m, c);
    for (int k = 0; k < suite::mw::n_con(i); ++k)
        if (!(c[k] <= tol)) return false;
    return true;
}

inline bool feasible2(int i, double f1, double f2, double tol = 0.0)
{
    const double f[2] = {f1, f2};
    return feasible(i, f, 2, tol);
}

// mw._shape
inline double shape(int i, double t)
{
    switch (i) {
        case 1: return 1.0 - 0.85 * t;
        case 3: return 1.0 - t;
        case 9: return 1.0 - dm::pow(t, 0.6);
        case 10: return 1.0 - dm::pow(t, 2.0);
        case 11: return std::sqrt(std::max(0.0, 2.0 - dm::pow(t, 2.0)));
        case 12: return 0.85 - 0.8 * t - 0.08 * std::fabs(dm::sin(3.2 * dm::kPi * t));
        default: return 5.0 - dm::exp(t) - 0.5 * std::fabs(dm::sin(3.0 * dm::kPi * t));
    }
}

// np.geomspace(a, b, n): 10^(linspace(log10 a, log10 b, n)), the ends exact
inline std::vector<double> geomspace(double a, double b, std::size_t n)
{
    const double ln10 = dm::log(10.0);
    std::vector<double> e = linspace(dm::log(a) / ln10, dm::log(b) / ln10, n), out(n);
    for (std::size_t k = 0; k < n; ++k) out[k] = dm::pow(10.0, e[k]);
    out.front() = a;
    out.back() = b;
    return out;
}

// mw._least_over_g: min of g s(f1/g) over g >= max(1, f1/top): a grid of g,
// then 60 golden-section steps round its best
inline double least_over_g(int i, double f1)
{
    static const std::vector<double> grid = [] {
        std::vector<double> g = linspace(1.0, 3.0, 2001);
        const std::vector<double> tail = geomspace(3.0, 70.0, 200);
        g.insert(g.end(), tail.begin() + 1, tail.end());
        return g;
    }();
    const double golden = (std::sqrt(5.0) - 1.0) / 2.0;
    const double base = std::max(1.0, f1 / upper(i));
    std::vector<double> G(grid.size());
    std::size_t k = 0;                                 // np.argmin: the first least
    double best = 0.0;
    for (std::size_t j = 0; j < grid.size(); ++j) {
        G[j] = base * grid[j];
        const double v = G[j] * shape(i, f1 / G[j]);
        if (j == 0 || v < best) {
            best = v;
            k = j;
        }
    }
    double lo = G[k > 0 ? k - 1 : 0], hi = G[std::min(k + 1, grid.size() - 1)];
    for (int it = 0; it < 60; ++it) {
        const double c = hi - golden * (hi - lo), d = lo + golden * (hi - lo);
        if (c * shape(i, f1 / c) < d * shape(i, f1 / d)) hi = d;
        else lo = c;
    }
    const double mid = 0.5 * (lo + hi);
    return std::min(best, mid * shape(i, f1 / mid));
}

// mw.floor: the least reachable f2 at f1
inline double floor_at(int i, double f1)
{
    if (scanned(i)) return shape(i, f1);
    if (i == 9 || i == 10 || i == 11) {
        const double top = upper(i);
        return f1 <= top ? shape(i, std::min(f1, top)) : 0.0;
    }
    return least_over_g(i, f1);
}

// mw._roots: where each factor of the constraints is zero, as f2 at f1
inline std::vector<double> roots(int i, double f1)
{
    const double q = f1 * f1;
    auto sq = [](double v) { return dm::pow(v, 2.0); };
    if (i == 9)
        return {1 - 0.64 * q, 1 - 0.36 * q, sq(1.35) - sq(f1 + 0.35), sq(1.15) - sq(f1 + 0.15)};
    if (i == 10) return {2 - 4 * q, 2 - 8 * q, 2 - 2 * q, 2 - 16 * q, 1 - q, 1.2 - 1.2 * q};
    if (i == 11)
        return {3 - q, 3 - 2 * q, 3 - 0.625 * q, 3 - 7 * q, 1.62 - 0.18 * q, 1.125 - 0.125 * q, 2.07 - 0.23 * q,
                0.63 - 0.07 * q};
    if (i == 13) {
        const double w = 0.5 * dm::sin(3.0 * dm::kPi * f1);
        return {5 - dm::exp(f1) - w, 5 - (1 + f1 + 0.5 * q) - w, 5 - (1 + 0.7 * f1) - w, 5 - (1 + 0.4 * f1) - w};
    }
    std::vector<double> out;                         // MW12: the decreasing roots, 60 bisections
    static constexpr double kLines[4][4] = {{1.0, 0.8, 1.0, 1.5}, {1.0, 0.625, 1.0, 1.6},
                                            {1.4, 0.875, 1.4, 1.6}, {1.8, 1.125, 1.8, 1.6}};
    for (const auto& L : kLines) {
        const double line = L[0] - L[1] * f1;
        double lo = line - 0.1, hi = line + 0.1;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (line - mid + 0.08 * dm::sin(2.0 * dm::kPi * (mid / L[2] - f1 / L[3])) > 0.0) lo = mid;
            else hi = mid;
        }
        out.push_back(0.5 * (lo + hi));
    }
    return out;
}

// mw._scan: MW1, MW3 — the least feasible f2 from the floor up (step 2e-4, f1
// + f2 <= 1.5 + step), 60 bisections from the first feasible step down; NaN
// where none
inline double scan(int i, double f1, double step = 2e-4)
{
    const double L = floor_at(i, f1);
    const std::size_t steps = static_cast<std::size_t>(0.5 / step) + 2;
    for (std::size_t j = 0; j < steps; ++j) {
        const double f2 = L + step * static_cast<double>(j);
        if (!(feasible2(i, f1, f2) && f1 + f2 <= 1.5 + step)) continue;
        if (j == 0) return L;
        double b = L + step * static_cast<double>(j - 1), c = f2;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (b + c);
            if (feasible2(i, f1, mid)) c = mid;
            else b = mid;
        }
        return c;
    }
    return kNaN;
}

// mw._radius: MW5-7, the least feasible radius along the ray at angle th
inline double radius(int i, double th)
{
    if (i == 5) {
        const double l2 = 0.5 * dm::kPi - 2.0 * std::fabs(th - 0.25 * dm::kPi);
        double s = dm::sin(6.0 * dm::pow(l2, 3.0));
        if (std::fabs(s) < 1e-12) s = 0.0;
        return 1.0 + std::max(0.5 * s, -0.45 * s);
    }
    if (i == 7) return std::max(1.0, 1.15 - 0.2 * dm::pow(dm::sin(4.0 * th), 8.0));
    return 1.1;
}

// mw._curve at one parameter: the point (f1, f2; f2 NaN where none) and
// whether it is the least reachable one
inline void curve(int i, double u, double* f, bool& low)
{
    if (radial(i)) {
        const double r = radius(i, u);
        f[0] = r * dm::cos(u);
        f[1] = r * dm::sin(u);
        for (int k = 0; k < 2; ++k)
            if (std::fabs(f[k]) < 1e-15) f[k] = 0.0;
        if (!feasible2(i, f[0], f[1], 1e-12)) f[1] = kNaN;
        low = r == (i == 6 ? 1.1 : 1.0);
        return;
    }
    const double lo = floor_at(i, u);
    f[0] = u;
    if (scanned(i)) {
        f[1] = scan(i, u);
    } else {
        double best = kInf;                          // mw._least_feasible of [floor] + roots
        std::vector<double> C{lo};
        for (double r : roots(i, u)) C.push_back(r);
        for (double c : C)
            if (c >= lo && c < best && feasible2(i, u, c, 1e-12)) best = c;
        f[1] = std::isfinite(best) ? best : kNaN;
    }
    low = f[1] == lo;
}

// mw._refined: midpoints added where neighbours lie farther apart than twice
// the median step, where one has a point and the other none, and where one
// is the least reachable point and the other not; at most 45 rounds
inline Rows refined(int i, std::vector<double> u, int rounds = 45)
{
    std::vector<double> F1(u.size()), F2(u.size());
    std::vector<char> low(u.size());
    for (std::size_t k = 0; k < u.size(); ++k) {
        double f[2];
        bool l = false;
        curve(i, u[k], f, l);
        F1[k] = f[0];
        F2[k] = f[1];
        low[k] = l;
    }
    auto steps = [&](std::vector<double>& st) {
        st.assign(u.size() - 1, 0.0);
        for (std::size_t k = 0; k + 1 < u.size(); ++k) {
            const double a = F1[k + 1] - F1[k], b = F2[k + 1] - F2[k];
            st[k] = std::sqrt(a * a + b * b);
        }
    };
    std::vector<double> st, ok_steps;
    steps(st);
    for (std::size_t k = 0; k + 1 < u.size(); ++k)
        if (std::isfinite(F2[k]) && std::isfinite(F2[k + 1])) ok_steps.push_back(st[k]);
    const double target = 2.0 * run_metrics::median(ok_steps);
    for (int r = 0; r < rounds; ++r) {
        steps(st);
        std::vector<double> mids;
        for (std::size_t k = 0; k + 1 < u.size(); ++k) {
            const bool a = std::isfinite(F2[k]), b = std::isfinite(F2[k + 1]), both = a && b;
            const bool wide = ((both && st[k] > target) || (a != b) || (both && low[k] != low[k + 1])) &&
                              (u[k + 1] - u[k] > 1e-12);
            if (wide) mids.push_back(0.5 * (u[k] + u[k + 1]));
        }
        if (mids.empty()) break;
        for (double m : mids) {
            double f[2];
            bool l = false;
            curve(i, m, f, l);
            u.push_back(m);
            F1.push_back(f[0]);
            F2.push_back(f[1]);
            low.push_back(l);
        }
        std::vector<std::size_t> order(u.size());
        for (std::size_t k = 0; k < u.size(); ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&u](std::size_t a, std::size_t b) { return u[a] < u[b]; });
        std::vector<double> u2, a2, b2;
        std::vector<char> l2;
        for (auto k : order) {
            u2.push_back(u[k]);
            a2.push_back(F1[k]);
            b2.push_back(F2[k]);
            l2.push_back(low[k]);
        }
        u.swap(u2);
        F1.swap(a2);
        F2.swap(b2);
        low.swap(l2);
    }
    Rows F = rows_of(0, 2);
    for (std::size_t k = 0; k < u.size(); ++k)
        if (std::isfinite(F2[k])) {
            const double r[2] = {F1[k], F2[k]};
            push_row(F, r);
        }
    return F;
}

// mw._nondominated: the nondominated rows sorted by f1 (then f2), strict
inline Rows nondominated_sorted(const Rows& F)
{
    std::vector<std::size_t> order(F.n);
    for (std::size_t k = 0; k < F.n; ++k) order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&F](std::size_t a, std::size_t b) {
        if (F.row(a)[0] != F.row(b)[0]) return F.row(a)[0] < F.row(b)[0];
        return F.row(a)[1] < F.row(b)[1];
    });
    std::vector<std::size_t> keep;
    double best = kInf;
    for (auto k : order)
        if (F.row(k)[1] < best) {
            keep.push_back(k);
            best = F.row(k)[1];
        }
    return take(F, keep);
}

// mw.dense_front: the two-objective front, sorted by f1
inline Rows dense_front(int i)
{
    if (i == 2) {
        Rows F = rows_of(0, 2);
        for (double t : linspace(0.0, 1.0, kResolution)) {
            const double r[2] = {t, 1.0 - t};
            push_row(F, r);
        }
        return F;
    }
    std::vector<double> u;
    if (radial(i)) {
        u = linspace(0.0, 0.5 * dm::kPi, kResolution);
        if (i == 5) {                                // the zeros of s: the discrete points
            for (int k = 0; k < 8; ++k) {
                const double l2 = dm::pow(static_cast<double>(k) * dm::kPi / 6.0, 1.0 / 3.0);
                u.push_back(0.25 * dm::kPi + (0.5 * dm::kPi - l2) / 2.0);
                u.push_back(0.25 * dm::kPi - (0.5 * dm::kPi - l2) / 2.0);
            }
            std::sort(u.begin(), u.end());           // np.unique
            u.erase(std::unique(u.begin(), u.end()), u.end());
        }
    } else {
        u = linspace(0.0, f1_range(i), scanned(i) ? kScanResolution : kResolution);
    }
    Rows F = refined(i, u);
    if (i == 11) {
        const double r[2] = {1.0, 1.0};              // the isolated point, exactly
        push_row(F, r);
    }
    return nondominated_sorted(F);
}

// mw._spread: about n rows of the sorted front, evenly by length along it,
// the ends of every piece kept (a step over 20 median steps separates two);
// piece: each row's piece, counted from 0 along the front
inline Rows spread(const Rows& F, std::size_t n, std::vector<std::size_t>* piece = nullptr)
{
    if (F.n < 2) {
        if (piece) piece->assign(F.n, 0);
        return F;
    }
    std::vector<double> st(F.n - 1);
    for (std::size_t k = 0; k + 1 < F.n; ++k) {
        const double a = F.row(k + 1)[0] - F.row(k)[0], b = F.row(k + 1)[1] - F.row(k)[1];
        st[k] = std::sqrt(a * a + b * b);
    }
    const double med = run_metrics::median(st);
    std::vector<char> gap(st.size());
    for (std::size_t k = 0; k < st.size(); ++k) gap[k] = st[k] > 20.0 * med;
    std::vector<std::size_t> of(F.n, 0);             // the piece of each row of F
    for (std::size_t k = 0; k + 1 < F.n; ++k) of[k + 1] = of[k] + (gap[k] ? 1 : 0);
    if (F.n <= n) {
        if (piece) *piece = of;
        return F;
    }
    std::vector<double> s(F.n, 0.0);                 // the length along it, gaps counting nothing
    for (std::size_t k = 0; k + 1 < F.n; ++k) s[k + 1] = s[k] + (gap[k] ? 0.0 : st[k]);
    std::vector<std::size_t> idx;
    for (std::size_t k = 0; k < F.n; ++k)
        if (k == 0 || k + 1 == F.n || (k > 0 && gap[k - 1]) || (k + 1 < F.n && gap[k])) idx.push_back(k);
    const std::size_t ends = idx.size();
    for (double t : linspace(0.0, s.back(), std::max<std::size_t>(n > ends ? n - ends : 0, 2))) {
        const std::size_t k = static_cast<std::size_t>(std::lower_bound(s.begin(), s.end(), t) - s.begin());
        idx.push_back(std::min(k, F.n - 1));
    }
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    if (piece) {
        piece->clear();
        for (std::size_t k : idx) piece->push_back(of[k]);
    }
    return take(F, idx);
}

// mw._mw14_records: the strict left records of h(t) on [0, 1.5]
inline const std::vector<double>& mw14_records()
{
    static const std::vector<double> rec = [] {
        const std::vector<double> t = linspace(0.0, 1.5, 200001);
        std::vector<double> out;
        double prior = kInf;
        for (double v : t) {
            const double h = 6.0 - dm::exp(v) - 1.5 * dm::sin(1.1 * dm::kPi * v * v);
            if (h < prior) out.push_back(v);
            prior = std::min(prior, h);
        }
        return out;
    }();
    return rec;
}

// mw._mw14_front: the corners of the records' range, then n draws of records;
// f_m = the mean of h over the positions
inline Rows mw14_front(std::size_t m, std::size_t n)
{
    const std::vector<double>& rec = mw14_records();
    const std::size_t d = m - 1;
    Rows P = rows_of(0, d);
    std::vector<double> row(d);
    // np.meshgrid(*[[r0, r1]] * d) (indexing "xy": the first two swap)
    // .reshape(d, -1).T: corner c's coordinate j is bit j of its index in
    // the grid's C order, with the first two axes swapped
    for (std::size_t c = 0; c < (std::size_t{1} << d); ++c) {
        std::vector<std::size_t> bit(d);
        for (std::size_t j = 0; j < d; ++j) bit[j] = (c >> (d - 1 - j)) & 1;   // C order of the grid's axes
        if (d >= 2) std::swap(bit[0], bit[1]);
        for (std::size_t j = 0; j < d; ++j) row[j] = bit[j] ? rec.back() : rec.front();
        push_row(P, row.data());
    }
    std::mt19937 g(static_cast<std::uint32_t>(20261010 + m));
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < d; ++j) row[j] = rec[static_cast<std::size_t>(rnd::below(g, rec.size()))];
        push_row(P, row.data());
    }
    Rows F = rows_of(P.n, m);
    std::vector<double> h(d);
    for (std::size_t k = 0; k < P.n; ++k) {
        for (std::size_t j = 0; j < d; ++j) {
            const double p = P.row(k)[j];
            F.row(k)[j] = p;
            h[j] = 6.0 - dm::exp(p) - 1.5 * dm::sin(1.1 * dm::kPi * p * p);
        }
        F.row(k)[d] = np::mean(h.data(), d);
    }
    return F;
}

// mw.front; piece: each row's piece where the front is the sorted
// two-objective one (spread), left empty elsewhere
inline Rows front(int i, std::size_t m, std::size_t n, std::vector<std::size_t>* piece = nullptr)
{
    if (i == 4) return simplex(m, n);
    if (i == 8) {
        const Rows S = sphere_oct(m, 2 * n);
        Rows F = rows_of(0, m);
        for (std::size_t k = 0; k < S.n; ++k)
            if (feasible(i, S.row(k), static_cast<int>(m), 1e-12)) push_row(F, S.row(k));
        return F;
    }
    if (i == 14) return mw14_front(m, n);
    static std::map<int, Rows> dense;                    // once per process, as lru_cache
    static std::mutex lock;
    Rows D;
    {
        std::lock_guard<std::mutex> g(lock);
        auto it = dense.find(i);
        if (it == dense.end()) it = dense.emplace(i, dense_front(i)).first;
        D = it->second;
    }
    return spread(D, n, piece);
}

}   // namespace mootation::refs::mw

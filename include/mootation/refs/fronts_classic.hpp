#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The fronts and frames of the verification set's problems that are not in
// closed form (suite/classic.hpp; benchmarks/classic.py; task 17, addendum 7,
// section 2.1): POL, KUR, QV and KURZ.
//
// THE FRAMES. Two objectives: the ideal is each objective's minimum, the
// nadir each objective at the point where the other takes its minimum. The
// points are known exactly or by a search along one variable:
//   QV    f1 at x = 0, f2 at x = 1.5 (every variable): 0 both; the nadir
//         f1(1.5) = f2(0) = 22.25^(1/4)
//   POL   f2's minimum 0 at (-3, -1); f1's 1 where B(x) = A — every root of
//         the 2 x 2 system in [-pi, pi]^2 (a grid of the residual's local
//         minima, then Newton's method), the nadir of f2 the least f2 over
//         them
//   KUR   f1's minimum at x = 0; f2 = sum g(x_i), g(x) = |x|^0.8 +
//         5 sin(x^3), at x_i = t* = argmin g over [-5, 5]
//   KURZ  f2's minimum 0 at x = 0; f1 = sum g(x_i) + 3.5828 n, g(x) =
//         |x|^0.8 + 5 sin^3 x, at x_i = t* = argmin g over [-10^3, 10^3]
// argmin g: the least value of a grid along [a, b], refined by golden
// section in the cells about it; the next-deepest local minimum of the grid
// is kept (it shows the minimum is the only one: minimum_of's `second`).
// The objectives are the suite's (classic::evaluate), at those points.
//
// THE FRONTS (classic.py, pareto_front): POL's and KUR's the non-dominated
// points of a grid of the box (1501^2 and 161^3 points), then of a finer
// grid of the cell about every one of them (21^2, 11^3 offsets); QV's the
// lower-left convex hull of the one-variable front of (r(x), r(x - 1.5)),
// r(y) = y^2 - 10 cos(2 pi y) + 10, x in [0, 1.5] (300 001 points), its
// segments sampled at 200 points each, then ^(1/4). n rows of either evenly
// by index along it. KURZ has none (its paper: the true Pareto set "cannot
// be determined").
// ============================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../archive_select.hpp"
#include "../detmath.hpp"
#include "../suite/classic.hpp"
#include "sample.hpp"

namespace mootation::refs::classic {

using suite::classic::Which;

struct Minimum {
    double x = 0.0, g = 0.0;                      // argmin and min
    double second_x = 0.0, second_g = std::numeric_limits<double>::infinity();
};

// the least g over [a, b]: a grid of `cells` steps, refined by golden section
// in [x - h, x + h] about its least point
inline Minimum minimum_of(const std::function<double(double)>& g, double a, double b, std::size_t cells)
{
    const double h = (b - a) / static_cast<double>(cells);
    std::vector<double> v(cells + 1);
    for (std::size_t i = 0; i <= cells; ++i) v[i] = g(i == cells ? b : a + static_cast<double>(i) * h);
    std::size_t best = 0;
    for (std::size_t i = 1; i <= cells; ++i)
        if (v[i] < v[best]) best = i;
    Minimum m;
    for (std::size_t i = 1; i < cells; ++i)               // the other local minima of the grid
        if (v[i] <= v[i - 1] && v[i] <= v[i + 1] && (i + 1 < best || i > best + 1) && v[i] < m.second_g) {
            m.second_g = v[i];
            m.second_x = a + static_cast<double>(i) * h;
        }
    double lo = std::max(a, a + (static_cast<double>(best) - 1.0) * h);
    double hi = std::min(b, a + (static_cast<double>(best) + 1.0) * h);
    const double r = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = hi - r * (hi - lo), d = lo + r * (hi - lo), gc = g(c), gd = g(d);
    for (int it = 0; it < 200 && hi - lo > 0.0; ++it) {
        if (gc < gd) {
            hi = d;
            d = c;
            gd = gc;
            c = hi - r * (hi - lo);
            gc = g(c);
        } else {
            lo = c;
            c = d;
            gc = gd;
            d = lo + r * (hi - lo);
            gd = g(d);
        }
        if (c == d) break;
    }
    m.x = gc < gd ? c : d;
    m.g = std::min(gc, gd);
    const double at_grid = a + static_cast<double>(best) * h;
    if (v[best] < m.g) {                                 // the grid point itself is lower
        m.x = at_grid;
        m.g = v[best];
    }
    return m;
}

inline void objectives(Which w, const std::vector<double>& x, double* f)
{
    suite::classic::evaluate(w, x.data(), static_cast<int>(x.size()), 2, f);
}

// POL: every root of B(x) = A in [-pi, pi]^2
inline std::vector<std::vector<double>> pol_roots()
{
    const double lo = -dm::kPi, hi = dm::kPi;
    auto residual = [](double x0, double x1) {
        double f[2];
        objectives(Which::POL, {x0, x1}, f);
        return f[0] - 1.0;
    };
    constexpr int k = 401;
    std::vector<double> axis = linspace(lo, hi, k), R(static_cast<std::size_t>(k * k));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) R[static_cast<std::size_t>(i * k + j)] = residual(axis[i], axis[j]);
    const double A1 = 0.5 * dm::sin(1.0) - 2.0 * dm::cos(1.0) + dm::sin(2.0) - 1.5 * dm::cos(2.0);
    const double A2 = 1.5 * dm::sin(1.0) - dm::cos(1.0) + 2.0 * dm::sin(2.0) - 0.5 * dm::cos(2.0);
    std::vector<std::vector<double>> roots;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            const double r0 = R[static_cast<std::size_t>(i * k + j)];
            bool least = true;
            for (int di = -1; di <= 1 && least; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const int a = i + di, b = j + dj;
                    if ((di || dj) && a >= 0 && a < k && b >= 0 && b < k && R[static_cast<std::size_t>(a * k + b)] < r0) {
                        least = false;
                        break;
                    }
                }
            if (!least) continue;
            double x0 = axis[i], x1 = axis[j];                   // Newton on A - B(x) = 0
            for (int it = 0; it < 100; ++it) {
                const double s0 = dm::sin(x0), c0 = dm::cos(x0), s1 = dm::sin(x1), c1 = dm::cos(x1);
                const double F1 = A1 - (0.5 * s0 - 2.0 * c0 + s1 - 1.5 * c1);
                const double F2 = A2 - (1.5 * s0 - c0 + 2.0 * s1 - 0.5 * c1);
                // d(-B)/dx
                const double j11 = -(0.5 * c0 + 2.0 * s0), j12 = -(c1 + 1.5 * s1);
                const double j21 = -(1.5 * c0 + s0), j22 = -(2.0 * c1 + 0.5 * s1);
                const double det = j11 * j22 - j12 * j21;
                if (det == 0.0) break;
                const double d0 = (F1 * j22 - F2 * j12) / det, d1 = (j11 * F2 - j21 * F1) / det;
                x0 -= d0;
                x1 -= d1;
                if (std::abs(d0) + std::abs(d1) < 1e-15) break;
            }
            if (x0 < lo || x0 > hi || x1 < lo || x1 > hi || !(residual(x0, x1) <= 1e-24)) continue;
            bool known = false;
            for (const auto& r : roots)
                if (std::abs(r[0] - x0) + std::abs(r[1] - x1) < 1e-9) known = true;
            if (!known) roots.push_back({x0, x1});
        }
    if (roots.empty()) throw std::runtime_error("POL: no root of B(x) = A found");
    return roots;
}

struct Frame2 {
    double ideal[2], nadir[2];
};

// the frame, at the minimisers of the two objectives
inline Frame2 frame(Which w, std::size_t n)
{
    std::vector<double> a(n, 0.0), b(n, 0.0);         // f1's minimiser, f2's
    if (w == Which::QV) {
        std::fill(b.begin(), b.end(), 1.5);
    } else if (w == Which::POL) {
        const auto roots = pol_roots();
        double best = std::numeric_limits<double>::infinity();
        for (const auto& r : roots) {
            double f[2];
            objectives(Which::POL, r, f);
            if (f[1] < best) {
                best = f[1];
                a = r;
            }
        }
        b = {-3.0, -1.0};
    } else if (w == Which::KUR) {
        const Minimum m = minimum_of([](double t) { return dm::pow(std::abs(t), 0.8) + 5.0 * dm::sin(t * t * t); },
                                     -5.0, 5.0, 1000000);
        std::fill(b.begin(), b.end(), m.x);
    } else if (w == Which::KURZ) {
        const Minimum m = minimum_of(
            [](double t) {
                const double s = dm::sin(t);
                return dm::pow(std::abs(t), 0.8) + 5.0 * (s * s * s);
            },
            -1000.0, 1000.0, 20000000);
        std::fill(a.begin(), a.end(), m.x);
    } else {
        throw std::invalid_argument("refs::classic::frame: POL, KUR, QV or KURZ");
    }
    double fa[2], fb[2];
    objectives(w, a, fa);
    objectives(w, b, fb);
    return Frame2{{fa[0], fb[1]}, {fb[0], fa[1]}};
}

// classic._nondominated_2d: the rows no other row dominates, sorted by f1
// (then f2); each with its x
inline void nondominated_2d(Rows& F, Rows& X)
{
    std::vector<std::size_t> order(F.n);
    for (std::size_t i = 0; i < F.n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t p, std::size_t q) {
        const double* a = F.row(p);
        const double* b = F.row(q);
        return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
    });
    std::vector<std::size_t> keep;
    double low = std::numeric_limits<double>::infinity();
    for (std::size_t i : order) {
        if (F.row(i)[1] < low) keep.push_back(i);
        low = std::min(low, F.row(i)[1]);
    }
    F = take(F, keep);
    X = take(X, keep);
}

// classic._even: n rows, evenly by index (np.round of numpy.linspace)
inline Rows even(const Rows& F, std::size_t n)
{
    if (F.n <= n) return F;
    std::vector<std::size_t> idx;
    for (double v : linspace(0.0, static_cast<double>(F.n - 1), n))
        idx.push_back(static_cast<std::size_t>(std::nearbyint(v)));
    return take(F, idx);
}

// the points of numpy.meshgrid(*[axis] * d, indexing="ij"), the last fastest
inline Rows grid(const std::vector<double>& axis, std::size_t d)
{
    std::size_t total = 1;
    for (std::size_t j = 0; j < d; ++j) total *= axis.size();
    Rows X = rows_of(total, d);
    for (std::size_t r = 0; r < total; ++r) {
        std::size_t q = r;
        for (std::size_t j = d; j-- > 0;) {
            X.row(r)[j] = axis[q % axis.size()];
            q /= axis.size();
        }
    }
    return X;
}

// classic._grid_front
inline Rows grid_front(Which w)
{
    const std::size_t d = w == Which::POL ? 2 : 3;
    const double lo = w == Which::POL ? -dm::kPi : -5.0, hi = -lo;
    const std::size_t k = d == 2 ? 1501 : 161;
    Rows X = grid(linspace(lo, hi, k), d), F = rows_of(X.n, 2);
    for (std::size_t i = 0; i < X.n; ++i)
        suite::classic::evaluate(w, X.row(i), static_cast<int>(d), 2, F.row(i));
    nondominated_2d(F, X);
    const double step = (hi - lo) / static_cast<double>(k - 1);
    const Rows D = grid(linspace(-step, step, d == 2 ? 21 : 11), d);
    Rows Y = rows_of(X.n * D.n, d), G = rows_of(X.n * D.n, 2);
    for (std::size_t i = 0; i < X.n; ++i)
        for (std::size_t t = 0; t < D.n; ++t) {
            double* y = Y.row(i * D.n + t);
            for (std::size_t j = 0; j < d; ++j) y[j] = std::min(std::max(X.row(i)[j] + D.row(t)[j], lo), hi);
            suite::classic::evaluate(w, y, static_cast<int>(d), 2, G.row(i * D.n + t));
        }
    Rows FF = F, XX = X;
    FF.v.insert(FF.v.end(), G.v.begin(), G.v.end());
    FF.n += G.n;
    XX.v.insert(XX.v.end(), Y.v.begin(), Y.v.end());
    XX.n += Y.n;
    nondominated_2d(FF, XX);
    return FF;
}

// classic._qv_front
inline Rows qv_front()
{
    const std::vector<double> x = linspace(0.0, 1.5, 300001);
    auto r = [](double y) { return y * y - 10.0 * dm::cos(2.0 * dm::kPi * y) + 10.0; };
    Rows F = rows_of(x.size(), 2), X = rows_of(x.size(), 1);
    for (std::size_t i = 0; i < x.size(); ++i) {
        F.row(i)[0] = r(x[i]);
        F.row(i)[1] = r(x[i] - 1.5);
        X.row(i)[0] = x[i];
    }
    nondominated_2d(F, X);
    std::vector<std::size_t> hull{0};                    // the lower convex hull, left to right
    for (std::size_t i = 1; i < F.n; ++i) {
        while (hull.size() >= 2) {
            const double* a = F.row(hull[hull.size() - 2]);
            const double* b = F.row(hull.back());
            const double* p = F.row(i);
            if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0.0) hull.pop_back();
            else break;
        }
        hull.push_back(i);
    }
    Rows P = rows_of(0, 2);
    for (std::size_t s = 0; s + 1 < hull.size(); ++s) {
        const double* a = F.row(hull[s]);
        const double* b = F.row(hull[s + 1]);
        for (int t = 0; t < 200; ++t) {              // numpy.linspace(a, b, 200, endpoint=False)
            const double row[2] = {a[0] + t * ((b[0] - a[0]) / 200.0), a[1] + t * ((b[1] - a[1]) / 200.0)};
            push_row(P, row);
        }
    }
    push_row(P, F.row(hull.back()));
    for (double& v : P.v) v = dm::pow(v, 0.25);
    return P;
}

}   // namespace mootation::refs::classic

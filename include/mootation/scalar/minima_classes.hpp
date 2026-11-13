#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The minima of a multistart, classified (task 26, item 24.2.6): for every
// two minima of a log (strategy.hpp, MinimaLog), whether they are one point
// but for the problem's symmetries ("copy"), two points of one basin with no
// barrier between them ("connected"), or two minima ("distinct").
//
//   classify_minima(minima, f, lo, hi, sym, min_dist, nodes, tol):
//     copy       sym is on and, after b is aligned to a (or both made
//                canonical, symmetry.hpp), the largest coordinate difference
//                in the box scaled to [0, 1] is at most min_dist (STRAT-4's
//                distance), MC-1;
//     connected  otherwise, on the segment from a to b (b aligned to a where
//                sym is on), f at `nodes` interior points z_t = a + t (b - a),
//                t = i / (nodes + 1), never exceeds max(f(a), f(b)) + tol;
//     distinct   otherwise; the barrier is the largest f on the segment less
//                max(f(a), f(b)).
//   The shares are of the M (M - 1) / 2 pairs. The evaluations of f are the
//   utility's own (counted in its result), not any run's budget.
//
// What it answers: whether the minima a multistart found are many minima or
// one degenerate one seen many times (a symmetry, a flat valley). The
// barrier is the simplest one, f sampled on the segment between the two.
//
// READINGS:
//   MC-1. "Copy" needs sym on: a log made with min_sym on has merged such
//     minima already; one made without keeps them apart, and this tells them.
//   MC-2. The segment is straight in the problem's coordinates, inside the
//     box (it is convex); periodic variables are not taken the short way.
//   MC-3. tol (default 0) absorbs rounding: a flat valley with f equal along
//     it is "connected".
//   MC-4. A non-finite f on the segment is a barrier (+inf): the minima are
//     "distinct".
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "strategy.hpp"
#include "symmetry.hpp"

namespace mootation::scalar {

enum class MinimaPair { copy, connected, distinct };

struct MinimaPairClass {
    std::size_t a = 0, b = 0;             // indices into the minima, a < b
    MinimaPair kind = MinimaPair::distinct;
    double barrier = 0.0;                 // distinct: the barrier; else <= tol
};

struct MinimaClasses {
    std::vector<MinimaPairClass> pairs;
    double copy_share = 0.0, connected_share = 0.0, distinct_share = 0.0;
    long long evaluations = 0;
};

inline MinimaClasses classify_minima(const std::vector<strategy::Minimum>& minima,
                                     const std::function<double(const std::vector<double>&)>& f,
                                     const std::vector<double>& lo, const std::vector<double>& hi,
                                     const symmetry::Sym& sym, double min_dist, int nodes = 9,
                                     double tol = 0.0)
{
    if (nodes < 1) throw std::invalid_argument("classify_minima: nodes >= 1");
    MinimaClasses out;
    const std::size_t m = minima.size();
    auto scaled_gap = [&](const std::vector<double>& u, const std::vector<double>& v) {
        double d = 0.0;
        for (std::size_t j = 0; j < u.size(); ++j) {
            const double w = hi[j] - lo[j];
            const double s = std::isfinite(w) && w > 0.0 ? w : 1.0;
            d = std::max(d, std::fabs(u[j] - v[j]) / s);
        }
        return d;
    };
    std::size_t copies = 0, connected = 0;
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = a + 1; b < m; ++b) {
            MinimaPairClass c;
            c.a = a;
            c.b = b;
            const std::vector<double>& xa0 = minima[a].x;
            std::vector<double> ka, kb;
            const std::vector<double>& xa = sym.mode == symmetry::Mode::canon ? sym.view(xa0, xa0, ka) : xa0;
            const std::vector<double>& xb = sym.view(minima[b].x, xa0, kb);
            if (sym.on() && scaled_gap(xa, xb) <= min_dist) {                      // MC-1
                c.kind = MinimaPair::copy;
                ++copies;
                out.pairs.push_back(c);
                continue;
            }
            const double top = std::max(minima[a].f, minima[b].f);
            double worst = -std::numeric_limits<double>::infinity();
            std::vector<double> z(xa.size());
            for (int i = 1; i <= nodes; ++i) {
                const double t = static_cast<double>(i) / static_cast<double>(nodes + 1);
                for (std::size_t j = 0; j < z.size(); ++j) z[j] = xa[j] + t * (xb[j] - xa[j]);
                const double v = f(z);
                ++out.evaluations;
                worst = std::max(worst, std::isfinite(v) ? v : std::numeric_limits<double>::infinity());   // MC-4
            }
            c.barrier = worst - top;
            if (worst <= top + tol) {
                c.kind = MinimaPair::connected;
                ++connected;
            }
            out.pairs.push_back(c);
        }
    const double n = static_cast<double>(out.pairs.size());
    if (n > 0) {
        out.copy_share = static_cast<double>(copies) / n;
        out.connected_share = static_cast<double>(connected) / n;
        out.distinct_share = 1.0 - out.copy_share - out.connected_share;
    }
    return out;
}

}   // namespace mootation::scalar

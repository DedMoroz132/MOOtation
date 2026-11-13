// SPDX-License-Identifier: Apache-2.0
// The minima log (task 26, item 24.2.6; scalar/strategy.hpp, STRAT-4):
//   * the search by a window of f gives the log the scan of every minimum
//     gave: on random answers clustered about a few points, with many f
//     within min_df of each other, the same minima (x, f, start, found) and
//     the same joined() after every add, against a copy of the scan;
//   * 10^5 answers at D = 30 are logged in time linear in their number;
//   * a NaN f never joins, and nothing joins it.

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/minima_classes.hpp>
#include <mootation/scalar/strategy.hpp>

#include "harness.hpp"

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

// the log before task 26: every minimum scanned in the order opened (STRAT-4)
struct ScanLog {
    Vec lo, hi;
    double dist, df;
    std::vector<sc::strategy::Minimum> list;
    std::size_t joined = 0;

    double distance(const Vec& a, const Vec& b) const
    {
        double d = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) d = std::max(d, std::fabs(a[j] - b[j]) / (hi[j] - lo[j]));
        return d;
    }

    void add(const Vec& x, double f, long long start)
    {
        for (std::size_t q = 0; q < list.size(); ++q) {
            auto& m = list[q];
            if (distance(m.x, x) <= dist && std::fabs(m.f - f) <= df) {
                joined = q;
                ++m.found;
                if (f < m.f) {
                    m.x = x;
                    m.f = f;
                    m.start = start;
                }
                return;
            }
        }
        joined = list.size();
        list.push_back({x, f, start, 1});
    }
};

bool same_list(const std::vector<sc::strategy::Minimum>& a, const std::vector<sc::strategy::Minimum>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].x != b[i].x || !(a[i].f == b[i].f || (std::isnan(a[i].f) && std::isnan(b[i].f))) ||
            a[i].start != b[i].start || a[i].found != b[i].found)
            return false;
    return true;
}

}   // namespace

int main()
{
    using mootation::testing::check;
    namespace rnd = mootation::rnd;

    // ── the window gives the scan's log ─────────────────────────────────────
    for (int trial = 0; trial < 6; ++trial) {
        const std::size_t D = 3;
        const Vec lo(D, -1.0), hi(D, 1.0);
        const double min_dist = trial % 2 ? 0.05 : 0.2, min_df = trial < 3 ? 1e-3 : 0.05;
        sc::strategy::MinimaLog log(lo, hi, min_dist, min_df);
        ScanLog scan{lo, hi, min_dist, min_df, {}, 0};
        std::mt19937 g(static_cast<unsigned>(1234 + trial));
        std::vector<Vec> centres(8, Vec(D));
        for (auto& c : centres) for (double& v : c) v = -1.0 + 2.0 * rnd::uniform01(g);
        bool joined_alike = true;
        for (long long s = 0; s < 4000; ++s) {
            const Vec& c = centres[static_cast<std::size_t>(rnd::uniform01(g) * 8.0) % 8];
            Vec x(D);
            for (std::size_t j = 0; j < D; ++j) x[j] = c[j] + 0.1 * (rnd::uniform01(g) - 0.5);
            // f on a coarse grid: many answers tie or fall within min_df of others
            const double f = std::floor(rnd::uniform01(g) * 40.0) * 0.01 + (s % 7 == 0 ? 1e-4 : 0.0);
            log.add(x, f, s);
            scan.add(x, f, s);
            joined_alike = joined_alike && log.joined() == scan.joined;
        }
        check(joined_alike, "window log: joined() as the scan's after every add, trial " + std::to_string(trial));
        check(same_list(log.list(), scan.list), "window log: the scan's minima, trial " + std::to_string(trial) +
                                                     " (" + std::to_string(scan.list.size()) + " minima)");
    }

    // ── NaN ─────────────────────────────────────────────────────────────────
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        sc::strategy::MinimaLog log({0.0}, {1.0}, 0.5, 1.0);
        log.add({0.5}, nan, 0);
        log.add({0.5}, nan, 1);
        log.add({0.5}, 0.0, 2);
        log.add({0.5}, 0.5, 3);
        check(log.list().size() == 3 && log.list()[2].found == 2 && log.joined() == 2,
              "a NaN f never joins, nothing joins it");
    }

    // ── 10^5 answers at D = 30 ──────────────────────────────────────────────
    {
        const std::size_t D = 30;
        const Vec lo(D, 0.0), hi(D, 1.0);
        sc::strategy::MinimaLog log(lo, hi, 1e-3, 1e-8);
        std::mt19937 g(77);
        Vec x(D);
        const auto t0 = std::chrono::steady_clock::now();
        for (long long s = 0; s < 100000; ++s) {
            for (double& v : x) v = rnd::uniform01(g);
            log.add(x, rnd::uniform01(g), s);
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "  10^5 answers at D = 30: " << sec << " s, " << log.list().size() << " minima\n";
        check(log.list().size() == 100000 && sec < 20.0, "10^5 answers at D = 30 in linear time");
    }

    // ── classify_minima (minima_classes.hpp, MC-1 ... MC-4) ─────────────────
    {
        using sc::MinimaPair;
        // a double well: minima at -1 and 1, f(0) = 1 between them
        auto well = [](const Vec& x) { return (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0); };
        std::vector<sc::strategy::Minimum> two{{{-1.0}, 0.0, 0, 1}, {{1.0}, 0.0, 1, 1}};
        sc::symmetry::Sym off;
        const auto c = sc::classify_minima(two, well, {-2.0}, {2.0}, off, 1e-3);
        check(c.pairs.size() == 1 && c.pairs[0].kind == MinimaPair::distinct && std::fabs(c.pairs[0].barrier - 1.0) < 1e-12 &&
                  c.distinct_share == 1.0 && c.evaluations == 9,
              "classify: a double well's two minima are distinct, barrier f(0) = 1");
        // a flat valley: f = x1^2, minima along x0 = anything
        auto valley = [](const Vec& x) { return x[1] * x[1]; };
        std::vector<sc::strategy::Minimum> flat{{{0.1, 0.0}, 0.0, 0, 1}, {{0.9, 0.0}, 0.0, 1, 1}};
        const auto v = sc::classify_minima(flat, valley, {0.0, -1.0}, {1.0, 1.0}, off, 1e-3);
        check(v.pairs[0].kind == MinimaPair::connected && v.connected_share == 1.0,
              "classify: two points of a flat valley are connected");
        // two blocks of one coordinate: g(x0) + g(x1), minima (0.2, 0.7) and (0.7, 0.2) are copies
        auto g = [](double t) { return (t - 0.2) * (t - 0.2) * (t - 0.7) * (t - 0.7); };
        auto sym2 = [&](const Vec& x) { return g(x[0]) + g(x[1]); };
        std::vector<sc::strategy::Minimum> perm{{{0.2, 0.7}, 0.0, 0, 1}, {{0.7, 0.2}, 0.0, 1, 1}, {{0.2, 0.2}, 0.0, 2, 1}};
        sc::symmetry::Sym al;
        al.mode = sc::symmetry::Mode::align;
        al.groups = {sc::symmetry::Blocks{0, 1, 2}};
        const auto p = sc::classify_minima(perm, sym2, {0.0, 0.0}, {1.0, 1.0}, al, 1e-6);
        check(p.pairs.size() == 3 && p.pairs[0].kind == MinimaPair::copy && p.pairs[1].kind == MinimaPair::distinct &&
                  p.pairs[2].kind == MinimaPair::distinct && std::fabs(p.copy_share - 1.0 / 3.0) < 1e-15,
              "classify: a permutation of blocks is a copy (align), the others distinct");
        const auto q = sc::classify_minima(perm, sym2, {0.0, 0.0}, {1.0, 1.0}, off, 1e-6);
        check(q.copy_share == 0.0 && q.distinct_share == 1.0, "classify: without sym no copies (MC-1)");
        auto nanf = [](const Vec&) { return std::numeric_limits<double>::quiet_NaN(); };
        const auto n = sc::classify_minima(flat, nanf, {0.0, -1.0}, {1.0, 1.0}, off, 1e-3);
        check(n.pairs[0].kind == MinimaPair::distinct, "classify: a non-finite f on the segment is a barrier (MC-4)");
    }

    return mootation::testing::report("test_minima");
}

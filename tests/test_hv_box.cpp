// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// hv_box.hpp: the hypervolume by box decomposition (Lacour, Klamroth & Fonseca
// 2017) against WFG (hypervolume.hpp) on 10 000 random sets at 3 to 6
// objectives — fronts, dominated points, copies, equal coordinates, points on
// the reference point's faces — within 1e-12 relative; the contributions
// (hv_contribution.hpp) within 1e-12 of the set's hypervolume. Small cases by
// hand. hv_algo = "auto" (automatic, task 13 of 2026-10-19, §3a) on 10 000
// random sets at 2 to 10 objectives: box's value bit for bit at 3 to 8, WFG's
// elsewhere, and within 1e-11 of WFG everywhere.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/hv_box.hpp>
#include <mootation/hv_contribution.hpp>
#include <mootation/hypervolume.hpp>

namespace {

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        if (failed <= 20) std::printf("  FAIL  %s\n", what.c_str());
    }
}

}   // namespace

int main()
{
    namespace H = mootation::hypervolume;
    // by hand: one point; two in 2-D; a 3-D staircase
    {
        const std::vector<double> r = {1.0, 1.0, 1.0};
        check(H::box({{0.5, 0.5, 0.5}}, r) == 0.125, "one point");
        check(std::abs(H::box({{0.2, 0.6}, {0.6, 0.2}}, {1.0, 1.0}) - 0.48) < 1e-15, "two points");
        const double v = H::box({{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}}, r);
        check(std::abs(v - 0.5) < 1e-15, "three points of the 3-D staircase");
        check(H::box({{2.0, 0.5, 0.5}}, r) == 0.0, "outside the reference point: nothing");
        check(H::box({{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.6, 0.6, 0.6}}, r) == 0.125,
              "copies and a dominated point add nothing");
    }
    std::mt19937_64 rng(20261018);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> g(0.0, 1.0);
    double worst = 0.0, worst_c = 0.0;
    for (int t = 0; t < 10000; ++t) {
        const std::size_t m = 3 + t % 4;
        const std::size_t n = 1 + rng() % (m <= 4 ? 60 : 30);
        const int q = 1 << (2 + rng() % 4);
        std::vector<std::vector<double>> F;
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> f(m);
            const double kind = u(rng);
            if (i > 0 && kind < 0.1) {                         // a copy
                F.push_back(F[rng() % F.size()]);
                continue;
            }
            double s = 0.0;
            for (auto& v : f) { v = std::abs(g(rng)); s += v * v; }
            for (auto& v : f) {
                v = (1.0 + 0.3 * u(rng) * (kind < 0.3)) * v / std::sqrt(s);   // some behind the front
                if (kind > 0.8) v = std::floor(v * q) / q;     // equal coordinates
            }
            F.push_back(f);
        }
        std::vector<double> r(m, 1.1);
        if (t % 7 == 0) r[0] = F[0][0];                      // a point on the reference's face
        const double a = H::box(F, r), b = H::wfg(F, r);
        const double rel = std::abs(a - b) / std::max(std::abs(b), 1e-300);
        worst = std::max(worst, b > 0.0 ? rel : std::abs(a));
        check(b > 0.0 ? rel <= 1e-12 : a == 0.0,
              "set " + std::to_string(t) + " (M = " + std::to_string(m) + "): box " +
                  std::to_string(a) + ", wfg " + std::to_string(b));
        if (t % 5 == 0) {                                     // contributions: the nondominated rows
            std::vector<std::vector<double>> P;
            for (std::size_t i = 0; i < F.size(); ++i) {
                bool dominated = false;
                for (std::size_t k = 0; k < F.size() && !dominated; ++k) {
                    if (k == i) continue;
                    bool le = true, lt = false;
                    for (std::size_t j = 0; j < m; ++j) {
                        le = le && F[k][j] <= F[i][j];
                        lt = lt || F[k][j] < F[i][j];
                    }
                    dominated = le && lt;
                }
                if (!dominated) P.push_back(F[i]);
            }
            const auto cb = mootation::hv::contributions_box(P, r);
            const auto cw = mootation::hv::contributions_exact(P, r);
            double d = 0.0;
            for (std::size_t i = 0; i < P.size(); ++i) d = std::max(d, std::abs(cb[i] - cw[i]));
            worst_c = std::max(worst_c, b > 0.0 ? d / b : d);
            check(d <= 1e-12 * std::max(b, 1e-300) || (b == 0.0 && d == 0.0),
                  "contributions of set " + std::to_string(t));
        }
    }
    // ── hv_algo = "auto" ────────────────────────────────────────────────────
    double worst_a = 0.0;
    {
        std::mt19937_64 rng_a(20261020);
        std::uniform_real_distribution<double> ua(0.0, 1.0);
        std::normal_distribution<double> ga(0.0, 1.0);
        for (int t = 0; t < 10000; ++t) {
            const std::size_t m = 2 + t % 9;
            const std::size_t n = 1 + rng_a() % (m <= 6 ? 40 : m <= 8 ? 20 : 10);
            std::vector<double> F, r(m, 1.1);
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<double> f(m);
                double s = 0.0;
                for (auto& v : f) { v = std::abs(ga(rng_a)); s += v * v; }
                for (auto& v : f) v = (1.0 + 0.2 * ua(rng_a)) * v / std::sqrt(s);
                F.insert(F.end(), f.begin(), f.end());
            }
            const double a = H::automatic(F.data(), n, m, r.data());
            const double w = H::wfg(F.data(), n, m, r.data());
            const double same = H::auto_takes_box(m) ? H::box(F.data(), n, m, r.data()) : w;
            check(a == same, "auto takes " + std::string(H::auto_takes_box(m) ? "box" : "wfg") +
                                 " at m = " + std::to_string(m) + ", set " + std::to_string(t));
            const double d = std::abs(a - w);
            worst_a = std::max(worst_a, w > 0.0 ? d / w : d);
            check(d <= 1e-11 * std::max(w, 1e-300) || (w == 0.0 && d == 0.0),
                  "auto within 1e-11 of wfg, set " + std::to_string(t));
        }
    }
    check(!H::auto_takes_box(2) && H::auto_takes_box(3) && H::auto_takes_box(8) &&
              !H::auto_takes_box(9), "auto: box at 3 to 8 objectives");

    std::printf("hv_box: %d/%d checks passed (largest relative difference %.3g, of the "
                "contributions %.3g; auto against wfg %.3g)%s\n", checks - failed, checks,
                worst, worst_c, worst_a, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}

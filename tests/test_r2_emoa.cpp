// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// r2_emoa.hpp (task 9, B3): the R2 indicator R2-EMOA selects by.
//   * r2_exact_2d against the discrete R2 of Def. 4 with 200 001 weights
//     uniform in weight space, on random two-objective sets: the exact R2 of
//     Schäpermeier et al. 2024 is the limit of the discrete one;
//   * one point: the closed form of their §3.1;
//   * dominated points and copies change nothing;
//   * the contributions by §5.2 of Brockhoff et al. 2015 (second best minus
//     best, per weight) equal Def. 5, R2(A \ {a}) − R2(A), computed by brute
//     force with the same weights, at two and three objectives;
//   * the exact contributions: positive for distinct non-dominated points,
//     0 for a copy.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include <mootation/algorithms/r2_emoa.hpp>
#include <mootation/individuals.hpp>

namespace {

int g_checks = 0, g_failed = 0;

void check(bool ok, const char* what)
{
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL  %s\n", what);
    }
}

using mootation::r2_exact_2d;

// Def. 4 with the given weights, the utopian point at the origin.
double r2_discrete(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& W)
{
    double s = 0.0;
    for (const auto& w : W) {
        double best = 1e300;
        for (const auto& a : A) {
            double u = 0.0;
            for (std::size_t j = 0; j < w.size(); ++j) u = std::max(u, w[j] * a[j]);
            best = std::min(best, u);
        }
        s += best;
    }
    return s / static_cast<double>(W.size());
}

// The protected contributions, for the test.
struct Probe : mootation::R2EMOACore<mootation::NSGAII_Individual> {
    std::vector<double> contributions(const std::vector<std::vector<double>>& Fv,
                                      const std::vector<std::vector<double>>& Q)
    {
        return this->other_contributions(Fv, Q);
    }
};

}   // namespace

int main()
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // the exact R2 is the limit of the discrete one
    const int K = 200001;
    std::vector<std::vector<double>> W2;
    for (int i = 0; i < K; ++i) {
        const double w = static_cast<double>(i) / (K - 1);
        W2.push_back({w, 1.0 - w});
    }
    int agree = 0;
    for (int t = 0; t < 20; ++t) {
        const int n = 1 + static_cast<int>(U(rng) * 12);
        std::vector<std::vector<double>> A;
        std::vector<std::pair<double, double>> P;
        for (int i = 0; i < n; ++i) {                      // a front: f2 = 1 − f1^p, shifted
            const double x = U(rng), p = 0.5 + U(rng);
            A.push_back({0.1 + x, 0.1 + 1.0 - std::pow(x, p)});
        }
        for (const auto& a : A) P.emplace_back(a[0], a[1]);
        agree += std::abs(r2_exact_2d(P) - r2_discrete(A, W2)) <= 1e-6;
    }
    check(agree == 20, "exact R2 = discrete R2 with 200 001 weights, within 1e-6 (20 sets)");

    // one point (§3.1)
    {
        const double y1 = 0.3, y2 = 0.8, ws = y2 / (y1 + y2);
        const double want = 0.5 * y2 * (1.0 - (1.0 - ws) * (1.0 - ws)) + 0.5 * y1 * (1.0 - ws * ws);
        check(std::abs(r2_exact_2d({{y1, y2}}) - want) <= 1e-15, "one point: the closed form");
    }

    // dominated points and copies
    {
        const std::vector<std::pair<double, double>> P = {{0.2, 0.9}, {0.5, 0.4}, {0.9, 0.1}};
        auto Q = P;
        Q.push_back({0.6, 0.5});                           // dominated by (0.5, 0.4)
        Q.push_back({0.5, 0.4});                           // a copy
        check(r2_exact_2d(P) == r2_exact_2d(Q), "dominated points and copies change nothing");
    }

    // §5.2's contributions are Def. 5's
    for (int m : {2, 3}) {
        Probe probe;
        probe.set_word("r2_mode", "discrete");
        probe.set_param("r2_weights", 301);
        const auto& W = probe.weights(static_cast<std::size_t>(m));
        int same = 0;
        for (int t = 0; t < 10; ++t) {
            std::vector<std::vector<double>> F;
            for (int i = 0; i < 8; ++i) {                  // a sphere's front, + the utopia (0, ...) later
                std::vector<double> f(static_cast<std::size_t>(m));
                double s = 0.0;
                for (auto& v : f) { v = 0.05 + U(rng); s += v * v; }
                for (auto& v : f) v /= std::sqrt(s);
                F.push_back(f);
            }
            auto Q = F;
            Q.push_back(std::vector<double>(static_cast<std::size_t>(m), 0.0));   // r* = 0
            const auto c = probe.contributions(F, Q);
            const double all = r2_discrete(F, W);
            bool ok = true;
            for (std::size_t a = 0; a < F.size(); ++a) {
                auto without = F;
                without.erase(without.begin() + static_cast<std::ptrdiff_t>(a));
                ok = ok && std::abs((r2_discrete(without, W) - all) - c[a]) <= 1e-12;
            }
            same += ok;
        }
        check(same == 10, m == 2 ? "§5.2 = Def. 5 at two objectives (10 sets)"
                                 : "§5.2 = Def. 5 at three objectives (10 sets)");
    }

    // the exact contributions
    {
        Probe probe;
        const std::vector<std::vector<double>> F = {{0.2, 0.9}, {0.5, 0.4}, {0.9, 0.1}, {0.5, 0.4}};
        auto Q = F;
        Q.push_back({0.0, 0.0});
        const auto c = probe.contributions(F, Q);
        check(c[0] > 0 && c[2] > 0 && c[1] == 0 && c[3] == 0,
              "exact: a copy contributes 0, the distinct points more");
    }

    std::printf("r2_emoa: %d/%d checks passed%s\n", g_checks - g_failed, g_checks,
                g_failed ? ", FAILED" : "");
    return g_failed == 0 ? 0 : 1;
}

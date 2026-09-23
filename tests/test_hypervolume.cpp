// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// hypervolume.hpp: the exact WFG hypervolume and the Monte-Carlo count.
//
//   (1) closed forms: one box, two boxes with their overlap, a 2-D staircase,
//       the unit simplex corners in 3-D;
//   (2) what adds nothing: points on or beyond the reference point, NaN rows,
//       dominated points and duplicates;
//   (3) random sets of up to 10 points in 2 to 6 objectives against the
//       inclusion-exclusion sum over every subset — an independent formula,
//       exact, and affordable at that size (2^10 subsets);
//   (4) covered() against a brute-force count, and the Monte-Carlo estimate
//       it gives converging on the exact value.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include <mootation/hypervolume.hpp>

namespace {

int g_checks = 0, g_failed = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL  %s\n", what);
    }
}

void check_close(double got, double want, double tol, const char* what) {
    ++g_checks;
    if (!(std::abs(got - want) <= tol)) {
        ++g_failed;
        std::printf("  FAIL  %s: got %.15g, want %.15g\n", what, got, want);
    }
}

using Rows = std::vector<std::vector<double>>;

// sum over nonempty subsets S of (-1)^(|S|+1) * prod_j (r_j - max_{i in S} p_ij),
// counting only points strictly better than r in every objective
double inclusion_exclusion(const Rows& P, const std::vector<double>& r) {
    Rows in;
    for (const auto& p : P) {
        bool ok = true;
        for (std::size_t j = 0; j < r.size(); ++j) ok = ok && p[j] < r[j];
        if (ok) in.push_back(p);
    }
    const std::size_t n = in.size(), m = r.size();
    double total = 0.0;
    for (unsigned long s = 1; s < (1ul << n); ++s) {
        std::vector<double> worst(m, -std::numeric_limits<double>::infinity());
        int bits = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!(s >> i & 1ul)) continue;
            ++bits;
            for (std::size_t j = 0; j < m; ++j) worst[j] = std::max(worst[j], in[i][j]);
        }
        double v = 1.0;
        for (std::size_t j = 0; j < m; ++j) v *= r[j] - worst[j];
        total += (bits % 2 ? v : -v);
    }
    return total;
}

}  // namespace

int main() {
    using mootation::hypervolume::covered;
    using mootation::hypervolume::wfg;

    // ── (1) closed forms ────────────────────────────────────────────────────
    check_close(wfg(Rows{{0.2, 0.3, 0.4}}, {1, 1, 1}), 0.8 * 0.7 * 0.6, 1e-15, "one box");
    check_close(wfg(Rows{{0.0, 0.5}, {0.5, 0.0}}, {1, 1}), 0.75, 1e-15, "two boxes in 2-D");
    check_close(wfg(Rows{{0.0, 0.5, 0.0}, {0.5, 0.0, 0.0}}, {1, 1, 1}), 0.75, 1e-15,
                "two boxes in 3-D, overlap subtracted once");
    check_close(wfg(Rows{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {2, 2, 2}), 3 * 4.0 - 3 * 2.0 + 1.0,
                1e-12, "the three unit corners against (2, 2, 2)");
    {
        Rows stairs;
        for (int i = 0; i < 10; ++i) stairs.push_back({0.1 * i, 0.9 - 0.1 * i});
        // row i adds a strip 0.1 high and 1 - 0.1 i wide: 0.1 * (1 + 0.9 + ... + 0.1)
        check_close(wfg(stairs, {1, 1}), 0.55, 1e-12, "a 2-D staircase of ten points");
    }

    // ── (2) what adds nothing ───────────────────────────────────────────────
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check_close(wfg(Rows{{1.0, 0.0}, {0.5, 1.0}}, {1, 1}), 0.0, 0.0,
                "points on the reference point's faces add nothing");
    check_close(wfg(Rows{{nan, 0.1}, {0.5, 0.5}}, {1, 1}), 0.25, 1e-15, "a NaN row adds nothing");
    check_close(wfg(Rows{{0.5, 0.5}, {0.6, 0.6}, {0.5, 0.5}, {0.5, 0.7}}, {1, 1}), 0.25, 1e-15,
                "dominated points and duplicates add nothing");
    check_close(wfg(Rows{}, {1, 1, 1}), 0.0, 0.0, "the empty set");
    check_close(wfg(Rows{{0.3}}, {1}), 0.7, 1e-15, "one objective");

    // ── (3) random sets against inclusion-exclusion ─────────────────────────
    std::mt19937_64 rng(20260922);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int agree = 0, total = 0;
    double worst = 0.0;
    for (int m = 2; m <= 6; ++m) {
        for (int rep = 0; rep < 60; ++rep) {
            const int n = 1 + static_cast<int>(rng() % 10);
            Rows P(n, std::vector<double>(m));
            for (auto& p : P) {
                // points near a sphere, so that few dominate each other, and
                // a few copies and snapped coordinates to exercise the ties
                double norm = 0.0;
                for (auto& v : p) { v = U(rng); norm += v * v; }
                for (auto& v : p) v = v / std::sqrt(norm) * (0.8 + 0.4 * U(rng));
                if (U(rng) < 0.2) p[rng() % m] = 0.5;
            }
            if (n > 2 && U(rng) < 0.3) P[1] = P[0];
            std::vector<double> r(m, 1.1);
            const double a = wfg(P, r), b = inclusion_exclusion(P, r);
            const double err = std::abs(a - b);
            worst = std::max(worst, err);
            agree += err <= 1e-12;
            ++total;
        }
    }
    check(agree == total, "WFG equals inclusion-exclusion on 300 random sets");
    std::printf("  random sets: %d/%d agree, largest difference %.3g\n", agree, total, worst);

    // ── (4) the Monte-Carlo count ───────────────────────────────────────────
    {
        const int m = 4, n = 40, k = 20000;
        std::vector<double> F(n * m), S(k * m);
        for (auto& v : F) v = 0.2 + 0.8 * U(rng);
        for (auto& v : S) v = 1.1 * U(rng);
        std::size_t brute = 0;
        for (int i = 0; i < k; ++i) {
            bool hit = false;
            for (int p = 0; p < n && !hit; ++p) {
                bool le = true;
                for (int j = 0; j < m && le; ++j) le = F[p * m + j] <= S[i * m + j];
                hit = le;
            }
            brute += hit;
        }
        check(covered(F.data(), n, m, S.data(), k) == brute,
              "covered() counts what a brute-force scan counts");
        const double box = std::pow(1.1, m);
        const double est = box * static_cast<double>(brute) / k;
        const std::vector<double> ref(m, 1.1);
        const double exact = wfg(F.data(), n, m, ref.data());
        // a binomial share from 20 000 samples: 4 standard errors
        const double p = static_cast<double>(brute) / k;
        check_close(est, exact, 4.0 * box * std::sqrt(p * (1 - p) / k),
                    "the Monte-Carlo estimate lands within four standard errors");
        check(covered(F.data(), n, m, S.data(), 0) == 0, "no samples, no hits");
    }

    std::printf("hypervolume: %d/%d checks passed%s\n", g_checks - g_failed, g_checks,
                g_failed ? ", FAILED" : "");
    return g_failed == 0 ? 0 : 1;
}

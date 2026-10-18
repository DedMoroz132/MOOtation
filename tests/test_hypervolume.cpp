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

#include <mootation/hv_contribution.hpp>
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

// hv::contributions_mc as it was before task 10 (every other point in the
// count, not only those that can reach the box): the reference the leaner
// count must equal, value for value, with the same generator.
template <typename RNG>
std::vector<double> contributions_mc_all(const std::vector<std::vector<double>>& F,
                                         const std::vector<double>& r, std::size_t per, RNG& rng)
{
    const std::size_t n = F.size(), m = r.size();
    std::vector<double> c(n, 0.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::vector<double> others, S(per * m), b(m);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        for (std::size_t k = 0; k < m; ++k) b[k] = r[k];
        others.clear();
        bool covered_whole = false;
        for (std::size_t q = 0; q < n && !covered_whole; ++q) {
            if (q == i) continue;
            std::size_t worse = 0, at = m;
            for (std::size_t j = 0; j < m; ++j)
                if (F[q][j] > p[j]) { ++worse; at = j; }
            covered_whole = worse == 0;
            if (worse == 1) b[at] = std::min(b[at], F[q][at]);
            others.insert(others.end(), F[q].begin(), F[q].end());
        }
        if (covered_whole) continue;
        double vol = 1.0;
        for (std::size_t k = 0; k < m; ++k) vol *= std::max(0.0, b[k] - p[k]);
        if (vol <= 0.0) continue;
        for (std::size_t s = 0; s < per; ++s)
            for (std::size_t k = 0; k < m; ++k) S[s * m + k] = p[k] + U01(rng) * (b[k] - p[k]);
        const std::size_t hit = others.empty() ? 0
            : mootation::hypervolume::covered(others.data(), others.size() / m, m, S.data(), per);
        c[i] = vol * static_cast<double>(per - hit) / static_cast<double>(per);
    }
    return c;
}

// hv::contributions_exact (M >= 3) as it was before task 10: a vector per
// row and hypervolume::wfg per point; the reference for the flat buffer.
std::vector<double> contributions_exact_rows(const std::vector<std::vector<double>>& F,
                                             const std::vector<double>& r)
{
    const std::size_t n = F.size(), m = r.size();
    std::vector<double> c(n, 0.0);
    std::vector<std::vector<double>> L;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        bool in = true;
        for (std::size_t k = 0; k < m; ++k) in = in && p[k] < r[k];
        if (!in) continue;
        double own = 1.0;
        for (std::size_t k = 0; k < m; ++k) own *= r[k] - p[k];
        L.clear();
        for (std::size_t q = 0; q < n; ++q) {
            if (q == i) continue;
            std::vector<double> l(m);
            for (std::size_t k = 0; k < m; ++k) l[k] = std::max(F[q][k], p[k]);
            L.push_back(std::move(l));
        }
        c[i] = std::max(0.0, own - (L.empty() ? 0.0 : mootation::hypervolume::wfg(L, r)));
    }
    return c;
}

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

    // the Monte-Carlo contributions count only the points that can reach a
    // point's box: the same values as counting every point, the same draws
    {
        std::mt19937 gen(31);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        bool same = true;
        for (int t = 0; t < 40 && same; ++t) {
            const std::size_t m = 4 + static_cast<std::size_t>(t % 4);
            const std::size_t n = 20 + static_cast<std::size_t>(t);
            std::vector<std::vector<double>> P(n, std::vector<double>(m));
            for (auto& row : P) {
                double s2 = 0.0;
                for (auto& v : row) { v = unit(gen); s2 += v * v; }
                for (auto& v : row) v /= std::sqrt(s2);           // a front on the sphere
            }
            if (t % 5 == 0) P[3] = P[2];                          // a copy
            const std::vector<double> ref(m, 1.2);
            std::mt19937 a(static_cast<unsigned>(t)), b(static_cast<unsigned>(t));
            same = mootation::hv::contributions_mc(P, ref, 500, a) ==
                   contributions_mc_all(P, ref, 500, b);
        }
        check(same, "the Monte-Carlo contributions with the reaching points alone");
        // the exact ones, from one flat buffer and one evaluator
        for (int t = 0; t < 60 && same; ++t) {
            const std::size_t m = 3 + static_cast<std::size_t>(t % 3);
            const std::size_t n = 2 + static_cast<std::size_t>(t);
            std::vector<std::vector<double>> P(n, std::vector<double>(m));
            for (auto& row : P) {
                double s2 = 0.0;
                for (auto& v : row) { v = unit(gen); s2 += v * v; }
                for (auto& v : row) v /= std::sqrt(s2);
            }
            if (t % 4 == 0) P[1] = P[0];                          // a copy
            if (t % 7 == 0) P[0][0] = 5.0;                        // a point beyond r
            const std::vector<double> ref(m, 1.1);
            same = mootation::hv::contributions_exact(P, ref) == contributions_exact_rows(P, ref);
        }
        check(same, "the exact contributions from one flat buffer");
    }

    std::printf("hypervolume: %d/%d checks passed%s\n", g_checks - g_failed, g_checks,
                g_failed ? ", FAILED" : "");
    return g_failed == 0 ? 0 : 1;
}

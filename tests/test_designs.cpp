// SPDX-License-Identifier: Apache-2.0
// The designs and the ablation baselines on the library's generators
// (include/mootation/designs.hpp, ablations.hpp; task 17, stage 5): one seed
// one plan, another seed another; the points in [0, 1); the first 2^m points of
// the scrambled Sobol sequences one in each of the 2^m intervals of a
// coordinate, and the first 2^m of two coordinates one in each elementary box
// of area 2^-m (a (0, m, 2)-net: the scrambles keep it); the first b^m points
// of a randomized Halton coordinate of base b one in each of its b^m
// intervals; every column of a Latin hypercube one point in each of its n
// cells; the ablations spend the budget exactly, the same with the same seed;
// past 4096 points lhs anneals without the n x n matrices (LHS-1): the same
// maximin order, a Latin hypercube still.

#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <mootation/ablations.hpp>
#include <mootation/designs.hpp>
#include <mootation/random.hpp>

namespace ds = mootation::designs;
namespace ab = mootation::ablations;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// every one of the `cells` equal intervals of coordinate j holds one point
static bool stratified(const std::vector<double>& U, std::size_t n, int d, int j, std::size_t cells)
{
    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < n; ++i)
        seen.insert(static_cast<std::size_t>(U[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(j)] *
                                             static_cast<double>(cells)));
    return seen.size() == cells && n == cells;
}

// every elementary box 2^-a x 2^-b (a + b = m) of coordinates 0 and 1 holds one point
static bool net2(const std::vector<double>& U, int d, int m)
{
    const std::size_t n = std::size_t{1} << m;
    for (int a = 0; a <= m; ++a) {
        std::set<std::pair<std::size_t, std::size_t>> seen;
        for (std::size_t i = 0; i < n; ++i)
            seen.insert({static_cast<std::size_t>(U[i * static_cast<std::size_t>(d)] * static_cast<double>(1u << a)),
                         static_cast<std::size_t>(U[i * static_cast<std::size_t>(d) + 1] *
                                                  static_cast<double>(1u << (m - a)))});
        if (seen.size() != n) return false;
    }
    return true;
}

int main()
{
    for (const std::string kind : {"uniform", "sobol", "halton_rand", "sobol_owen", "lhs"}) {
        const auto A = ds::design(kind, 40, 3, 7), B = ds::design(kind, 40, 3, 7), C = ds::design(kind, 40, 3, 8);
        CHECK(A == B && A != C && A.size() == 120);
        for (double u : A) CHECK(u >= 0.0 && u < 1.0);
    }
    for (const std::string kind : {"sobol", "sobol_owen"}) {
        const auto U = ds::design(kind, 64, 4, 3);
        for (int j = 0; j < 4; ++j) CHECK(stratified(U, 64, 4, j, 64));
        CHECK(net2(U, 4, 6));
    }
    const auto H = ds::halton(81, 2, 5);
    CHECK(stratified(H, 81, 2, 1, 81));                  // base 3, 3^4 points
    const auto H2 = ds::halton(64, 2, 5);
    CHECK(stratified(H2, 64, 2, 0, 64));                 // base 2, 2^6 points
    CHECK(ds::halton_digits(2) == 53);
    const auto L = ds::lhs(25, 4, 11);
    for (int j = 0; j < 4; ++j) CHECK(stratified(L, 25, 4, j, 25));

    // the ablations: the budget exactly, the same run with the same seed
    const std::vector<double> lo{0.0, 0.0, 0.0}, hi{1.0, 1.0, 1.0};
    long long calls = 0;
    auto evaluate = [&calls](const std::vector<double>& x) {
        ++calls;
        return std::vector<double>{x[0], 1.0 - std::sqrt(x[0]) + x[1] + x[2]};
    };
    auto first = [](const ab::Rows& F, std::size_t k) {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < F.size() && i < k; ++i) idx.push_back(i);
        return idx;
    };
    const auto r1 = ab::random_selection_ea(evaluate, lo, hi, 20, 450, 3, nullptr);
    CHECK(calls == 450);
    const auto r2 = ab::random_selection_ea(evaluate, lo, hi, 20, 450, 3, nullptr);
    CHECK(r1.F == r2.F && r1.X == r2.X && r1.F.size() == 20);
    calls = 0;
    const auto g1 = ab::gsemo(evaluate, lo, hi, 20, 300, 4, first, nullptr);
    CHECK(calls == 300 && g1.steps == 15);
    const auto g2 = ab::gsemo(evaluate, lo, hi, 20, 300, 4, first, nullptr);
    CHECK(g1.F == g2.F && !g1.F.empty());
    for (const auto& a : g1.F)                           // the population: no member dominates another
        for (const auto& b : g1.F)
            CHECK(!(a[0] <= b[0] && a[1] <= b[1] && (a[0] < b[0] || a[1] < b[1])));

    // task 24, §24.3.2: the Sobol designs past 100 dimensions (sobol.hpp up
    // to 21 201); a coordinate's scrambling is drawn for it alone, so the
    // first 100 of 300 coordinates are the 100 coordinates' design
    for (const char* kind : {"sobol", "sobol_owen"}) {
        const auto W = mootation::designs::design(kind, 16, 300, 5);
        const auto N = mootation::designs::design(kind, 16, 100, 5);
        CHECK(W.size() == 16 * 300);
        bool prefix = true;
        for (std::size_t i = 0; i < 16; ++i)
            for (std::size_t j = 0; j < 100; ++j) prefix = prefix && W[i * 300 + j] == N[i * 100 + j];
        CHECK(prefix);
        for (double u : W) CHECK(u >= 0.0 && u < 1.0);
    }

    // task 24, §24.3.2, LHS-1: past kLeanN points lhs anneals without the
    // n x n matrices. Its maximin order is the matrices' one, value for value;
    // its annealing keeps a Latin hypercube; past 4096 points the design is
    // one, a point in each cell of every column, the same with the same seed.
    {
        std::mt19937 g(3);
        const std::size_t n = 60, k = 3;
        std::vector<int> L0(n * k);
        for (std::size_t c = 0; c < k; ++c) {
            std::vector<int> col(n);
            for (std::size_t r = 0; r < n; ++r) col[r] = static_cast<int>(r);
            mootation::rnd::shuffle(col.begin(), col.end(), g);
            for (std::size_t r = 0; r < n; ++r) L0[r * k + c] = col[r];
        }
        CHECK(ds::maximin_order_lean(L0, n, k) == ds::maximin_order(ds::squared_distances(L0, n, k)));
        for (int p : {1, 10}) {
            std::mt19937 h(9);
            const auto A = ds::anneal_lean(L0, n, k, p, h);
            CHECK(ds::maximin_order_lean(A, n, k) == ds::maximin_order(ds::squared_distances(A, n, k)));
            for (std::size_t c = 0; c < k; ++c) {
                std::set<int> levels;
                for (std::size_t r = 0; r < n; ++r) levels.insert(A[r * k + c]);
                CHECK(levels.size() == n && *levels.begin() == 0 && *levels.rbegin() == static_cast<int>(n) - 1);
            }
        }
        const std::size_t big = ds::kLeanN + 1;
        const auto P = ds::lhs(big, 2, 11), Q = ds::lhs(big, 2, 11);
        CHECK(P == Q && P.size() == 2 * big);
        for (int j = 0; j < 2; ++j) CHECK(stratified(P, big, 2, j, big));
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("designs: the library's generators, the nets and the strata, the ablations\n");
    return 0;
}

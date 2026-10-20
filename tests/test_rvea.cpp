// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// rvea.hpp, the parents of RVEA and RVEA* (task 13 of 2026-10-19, §2; RVEA-3,
// RVEA*-6; rvea_parent_order):
//   * a full population (n_parents = n, n even and odd): the order of one
//     shuffle gone through once, as before RVEA-3, the same random numbers
//     drawn — so the same runs;
//   * a shrunk one (20 of 126): every round holds each survivor once, and
//     the distinct pairs of a generation are as many as those of random pairs
//     (about 54 of 63), not the 10 of one shuffle gone through again.
// ============================================================================

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <mootation/algorithms/rvea.hpp>

namespace {

int checks = 0, failed = 0;

void check(bool ok, const char* what)
{
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("FAIL: %s\n", what);
    }
}

// The order before RVEA-3: one shuffle, gone through round after round.
std::vector<int> one_shuffle(int n_parents, int n, std::mt19937& rng)
{
    std::vector<int> perm(static_cast<std::size_t>(n_parents)), order;
    std::iota(perm.begin(), perm.end(), 0);
    mootation::rnd::shuffle(perm.begin(), perm.end(), rng);       // as rvea.hpp
    for (int k = 0; k < n + n % 2; ++k) order.push_back(perm[static_cast<std::size_t>(k % n_parents)]);
    return order;
}

std::size_t distinct_pairs(const std::vector<int>& order, int n)
{
    std::set<std::pair<int, int>> pairs;
    for (int i = 0; i < n; i += 2)
        pairs.insert(std::minmax(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(i) + 1]));
    return pairs.size();
}

} // namespace

int main()
{
    // ── a full population: as before, the same draws ────────────────────────
    for (int n : {90, 91, 126, 210, 275}) {
        for (unsigned seed = 1; seed <= 20; ++seed) {
            std::mt19937 a(seed), b(seed);
            const bool same = mootation::rvea_parent_order(n, n, a) == one_shuffle(n, n, b) &&
                              a() == b();
            check(same, "full population: the order and the draws of one shuffle");
        }
    }

    // ── 20 of 126: rounds of every survivor, pairs as random ones ───────────
    const int np = 20, n = 126, gens = 2000;
    std::mt19937 rng(7), old_rng(7), pick(11);
    double now = 0.0, before = 0.0, random = 0.0;
    bool rounds = true;
    for (int g = 0; g < gens; ++g) {
        const auto order = mootation::rvea_parent_order(np, n, rng);
        for (int r = 0; r * np < n; ++r) {
            std::vector<int> part(order.begin() + r * np,
                                  order.begin() + std::min<int>((r + 1) * np, n));
            std::sort(part.begin(), part.end());
            rounds = rounds && std::adjacent_find(part.begin(), part.end()) == part.end();
        }
        now += static_cast<double>(distinct_pairs(order, n));
        before += static_cast<double>(distinct_pairs(one_shuffle(np, n, old_rng), n));
        std::uniform_int_distribution<int> u(0, np - 1);
        std::vector<int> rand_order;
        while (static_cast<int>(rand_order.size()) < n) {
            const int x = u(pick);
            int y = u(pick);
            while (y == x) y = u(pick);
            rand_order.push_back(x);
            rand_order.push_back(y);
        }
        random += static_cast<double>(distinct_pairs(rand_order, n));
    }
    now /= gens;
    before /= gens;
    random /= gens;
    std::printf("20 of 126: distinct pairs of 63 per generation: %.2f (random pairs %.2f; "
                "one shuffle %.2f)\n", now, random, before);
    check(rounds, "every round holds each survivor at most once");
    check(before == 10.0, "one shuffle gone through again: 10 distinct pairs");
    check(now > random - 2.0 && now < random + 2.0, "the distinct pairs as many as random pairs'");

    std::printf("rvea: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}

// SPDX-License-Identifier: Apache-2.0
// The integer genome (operators/bit_flip.hpp; task 7 of 2026-10-10,
// supplement 3, §1): a gene of k > 2 levels mutates to one of its other
// values, each alike, never to itself; a bit flips and draws nothing; a random
// gene is uniform over its levels and a bit's draw is the old one; the vault
// clamps a gene to its levels and a bit to 0 or 1; the mutation rate is
// 1/n, at most 1/2.

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <mootation/random.hpp>

#include "harness.hpp"

namespace {

using namespace mootation;
using mootation::testing::check;

void test_mutation()
{
    std::cout << "-- a gene mutates to another value, each alike --\n";
    ops::gene_levels() = {5, 2};
    std::mt19937 rng(7);
    std::vector<int> count(5, 0);
    bool never_same = true;
    for (int t = 0; t < 40000; ++t) {
        std::vector<int> g = {t % 5, 0};
        const int before = g[0];
        ops::mutate_gene(g, 0, rng);
        never_same = never_same && g[0] != before && g[0] >= 0 && g[0] < 5;
        ++count[static_cast<std::size_t>(g[0])];
    }
    check(never_same, "a gene of 5 levels never keeps its value, and stays in 0..4");
    bool even = true;
    for (int c : count) even = even && c > 7600 && c < 8400;     // 8000 each
    check(even, "each value as often as the others (40 000 draws)");
    // a bit flips and draws nothing
    std::mt19937 a(3), b(3);
    std::vector<int> g = {0, 1};
    ops::mutate_gene(g, 1, a);
    check(g[1] == 0 && a() == b(), "a bit flips without a draw");
}

void test_random_and_clamp()
{
    std::cout << "-- a random gene, the clamp, the rate --\n";
    ops::gene_levels() = {3, 2};
    std::mt19937 a(11), b(11);
    // the library's own distribution (random.hpp, task 14): the standard
    // library's draws differ between implementations (libc++ from libstdc++
    // and MSVC's), the library's do not
    mootation::rnd::uniform_int_distribution<int> old(0, 1);
    bool same = true;
    for (int t = 0; t < 1000; ++t) same = same && ops::random_gene(1, a) == old(b);
    check(same, "a random bit is the draw of rnd::uniform_int_distribution<int>(0, 1)");
    std::vector<int> seen(3, 0);
    for (int t = 0; t < 3000; ++t) ++seen[static_cast<std::size_t>(ops::random_gene(0, a))];
    check(seen[0] > 900 && seen[1] > 900 && seen[2] > 900, "a gene of 3 levels: all three");
    check(ops::clamp_gene(0, 7) == 2 && ops::clamp_gene(0, -1) == 0 && ops::clamp_gene(0, 1) == 1,
          "a gene is clamped to its levels");
    check(ops::clamp_gene(1, 7) == 1 && ops::clamp_gene(1, -1) == 1 && ops::clamp_gene(1, 0) == 0,
          "a bit is 1 for anything but 0, as before");
    check(ops::gene_mutation_rate(1) == 0.5 && ops::gene_mutation_rate(2) == 0.5
          && ops::gene_mutation_rate(8) == 0.125, "the rate 1/n, at most 1/2");
    ops::gene_levels().clear();
    check(ops::gene_levels_of(0) == 2 && ops::clamp_gene(0, 3) == 1, "no levels: every gene a bit");
}

}   // namespace

int main()
{
    std::cout << "the integer genome\n";
    test_mutation();
    test_random_and_clamp();
    return mootation::testing::report("genes");
}

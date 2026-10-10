#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <random>
#include <vector>

namespace mootation::ops {

// The levels of every gene of the integer genome (DataVault's
// binary_variables; task 7 of 2026-10-10, supplement 3, §1): a categorical
// variable of k values is a gene holding 0 .. k-1, a bit a gene of 2 levels.
// Set for a run by the binding, thread-local like pm_eta_per_variable; empty,
// or 2 or less: every gene a bit, as before categorical variables existed.
inline std::vector<int>& gene_levels()
{
    thread_local std::vector<int> v;
    return v;
}

inline int gene_levels_of(int j)
{
    const auto& l = gene_levels();
    const auto u = static_cast<std::size_t>(j);
    return j >= 0 && u < l.size() && l[u] > 2 ? l[u] : 2;
}

// A new value of gene j: a bit flips; a gene of k > 2 levels takes one of the
// other k - 1 values, each alike (pymoo's, PlatEMO's and Optuna's rule). A
// bit draws nothing, so a binary run is what it was.
template <typename RNG>
inline void mutate_gene(std::vector<int>& g, int j, RNG& rng)
{
    int& gene = g[static_cast<std::size_t>(j)];
    const int k = gene_levels_of(j);
    if (k <= 2) { gene ^= 1; return; }
    std::uniform_int_distribution<int> other(0, k - 2);
    const int v = other(rng);
    gene = v < gene ? v : v + 1;
}

// A random value of gene j, uniform over its levels (an initial population).
// For a bit the draw of uniform_int_distribution<int>(0, 1), as before.
template <typename RNG>
inline int random_gene(int j, RNG& rng)
{
    std::uniform_int_distribution<int> level(0, gene_levels_of(j) - 1);
    return level(rng);
}

// Gene j's value as the vault stores it: a bit is 1 for anything but 0, as
// before; a gene of k > 2 levels is clamped to [0, k - 1].
inline int clamp_gene(int j, int v)
{
    const int k = gene_levels_of(j);
    if (k <= 2) return v != 0 ? 1 : 0;
    return v < 0 ? 0 : (v > k - 1 ? k - 1 : v);
}

// The probability that a gene changes: 1/n_genes, at most 1/2 — pymoo's cap
// (Mutation.get_prob_var): with a single gene, 1 would change every child's,
// and a child could never keep its parents' category or bit.
inline double gene_mutation_rate(int n_genes)
{
    return n_genes > 1 ? 1.0 / static_cast<double>(n_genes) : 0.5;
}

// Bit-flip mutation: each gene changes (mutate_gene) with probability
// gene_mutation_rate(n_bits).
template <typename RNG>
inline void bit_flip_mutation(std::vector<int>& bvars, int n_bits, RNG& rng)
{
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double p = gene_mutation_rate(n_bits);
    for (int j = 0; j < n_bits; ++j)
        if (uni(rng) < p) mutate_gene(bvars, j, rng);
}

} // namespace mootation::ops

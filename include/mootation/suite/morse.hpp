#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// morse_cluster (task 20, package 24-27, §24.6.2): the Morse cluster of N
// atoms, E = sum_{i<j} V(r_ij), V(r) = [exp(rho (1 - r)) - 1]^2 - 1, x in
// R^3N (atom after atom); the gradient analytic. One objective, not least
// squares. The pair potential: A. Grosso, M. Locatelli, F. Schoen, "A
// population based approach for hard global optimization problems based on
// dissimilarity measures" (source
// population-dissimilarity-global-opt_grosso2007_preprint), §4.1; rho the
// instance's `rho`.
//
// The minima: the putative global minima at rho = 14 of A. Grosso, M.
// Locatelli, F. Schoen, "An experimental analysis of a population based
// approach for global optimization", 2007 (source
// experimental-analysis-population_grosso2007), Table 1: N = 30, 55, 61, 79
// (morse_best_known; MORSE-2). The preprint's PBH runs (Table 2) are on the
// same four and on N = 43, whose value neither paper prints.
//
// The box [-L, L]^3N, L = 1 + (3N / (4 pi sqrt 2))^(1/3) (MORSE-1).
// Symmetries, blocks, gauge directions, canon: lj.hpp's (Cluster). The start:
// atoms uniform in the box, from (20261102, crc32("morse_cluster:start"),
// instance, N).
//
// READINGS:
//   MORSE-1. The sources give no box. Ours is lj_cluster's (LJ-1) with the
//     pair minimum at r = 1: the fcc cell of nearest neighbours at 1 has the
//     volume 1/sqrt 2 per atom.
//   MORSE-2. Table 1 prints three decimals: a run at the minimum's structure
//     may end up to 5e-4 below the table's value.
//   MORSE-3. f(x_opt) = f_opt is checked where the structure is known
//     exactly: N = 2, 3, 4 as the segment, triangle and tetrahedron of edge 1,
//     every pair at V's minimum -1, so E = -N (N - 1) / 2 at any rho, the
//     least E can be (tests/test_suite.cpp).
// ============================================================================

#include <cmath>
#include <limits>

#include "lj.hpp"

namespace mootation::suite::sym {

// Table 1 of Grosso, Locatelli & Schoen (rho = 14; NaN elsewhere)
inline double morse_best_known(int N, double rho)
{
    if (rho != 14.0) return std::numeric_limits<double>::quiet_NaN();
    switch (N) {
        case 30: return -106.835;
        case 55: return -220.646;
        case 61: return -249.587;
        case 79: return -336.798;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

class MorseCluster final : public Cluster {
public:
    double rho = 0.0;

    explicit MorseCluster(const Instance& in) : Cluster(in, "morse_cluster", 1.0 / std::sqrt(2.0))   // MORSE-1
    {
        if (!(in.rho > 0.0)) throw std::invalid_argument("morse_cluster: rho > 0");
        rho = in.rho;
    }

    double value(const double* x) const override
    {
        double sum = 0.0;
        for (int i = 0; i < atoms; ++i)
            for (int j = i + 1; j < atoms; ++j) {
                const double e = dm::exp(rho * (1.0 - std::sqrt(squared(x, i, j)))) - 1.0;
                sum += e * e - 1.0;
            }
        return sum;
    }

    // dE/dx_i = V'(r) (x_i - x_j) / r, V'(r) = -2 rho exp(rho (1 - r)) (exp(rho (1 - r)) - 1)
    void gradient(const double* x, double* g) const override
    {
        std::fill(g, g + n, 0.0);
        for (int i = 0; i < atoms; ++i)
            for (int j = i + 1; j < atoms; ++j) {
                const double r = std::sqrt(squared(x, i, j)), a = dm::exp(rho * (1.0 - r));
                const double de = -2.0 * rho * a * (a - 1.0) / r;
                for (int c = 0; c < 3; ++c) {
                    const double t = de * (x[3 * i + c] - x[3 * j + c]);
                    g[3 * i + c] += t;
                    g[3 * j + c] -= t;
                }
            }
    }
};

}   // namespace mootation::suite::sym

// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Running a bundled benchmark, with and without its constraint.
//
// Shows the two things the problem set exists for: one macro turns a spec into
// a usable Problem<>, and a spec that declares limits works with
// constraint_mode without any further wiring.
//
// Build:
//   g++ -std=c++17 -O2 -Iinclude examples/benchmark_problem.cpp -o bench
// ============================================================================

#include <cstddef>
#include <iostream>

#include <mootation/mootation.hpp>
#include <mootation/problems/benchmarks.hpp>

// DTLZ2 with 3 objectives and 12 variables, and the same problem with the
// constraint x0 <= 0.5. Each macro call declares the tag and its Problem<>.
MOOTATION_DEFINE_PROBLEM(DTLZ2_3D, NSGAII_Individual,
                         mootation::problems::DTLZ2<3, 12>)
MOOTATION_DEFINE_PROBLEM(CDTLZ2_3D, NSGAII_Individual,
                         mootation::problems::CDTLZ2<3, 12>)

namespace {

constexpr int      POP  = 91;
constexpr int      GENS = 200;
constexpr unsigned SEED = 12345;

// Distance to the true DTLZ2 front, which is the unit sphere: | ||f|| - 1 |.
template <typename Ind>
double mean_front_error(mootation::DataVault<Ind>& v)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < v.active_n(); ++i) {
        double n2 = 0.0;
        for (double c : v.objectives_of(i)) n2 += c * c;
        sum += std::abs(std::sqrt(n2) - 1.0);
    }
    return v.active_n() ? sum / static_cast<double>(v.active_n()) : 0.0;
}

} // namespace

int main()
{
    using namespace mootation;

    // ── 1. Unconstrained ────────────────────────────────────────────────────
    {
        Problem<DTLZ2_3D>   prob;
        DataVault<DTLZ2_3D> vault(POP, prob);
        Optimizer<DTLZ2_3D, NSGAIICore<DTLZ2_3D>> opt(std::move(vault), defer_setup);
        opt.get_algorithm().set_seed(SEED);
        opt.setup();
        opt.optimize(GENS);

        std::cout << "DTLZ2      mean distance to front: "
                  << mean_front_error(opt.get_vault()) << '\n';
    }

    // ── 2. The same problem with x0 <= 0.5, constraint handling on ──────────
    {
        Problem<CDTLZ2_3D>   prob;
        DataVault<CDTLZ2_3D> vault(POP, prob);
        Optimizer<CDTLZ2_3D, NSGAIICore<CDTLZ2_3D>> opt(std::move(vault), defer_setup);

        auto& alg = opt.get_algorithm();
        alg.set_seed(SEED);
        alg.constraint_mode = ConstraintMode::FEASIBILITY;   // off by default
        opt.setup();
        opt.optimize(GENS);

        auto&       v = opt.get_vault();
        std::size_t feasible = 0;
        for (std::size_t i = 0; i < v.active_n(); ++i)
            if (v.get_cv(i) <= 0.0) ++feasible;

        std::cout << "C-DTLZ2    mean distance to front: " << mean_front_error(v)
                  << ", feasible " << feasible << " of " << v.active_n() << '\n';
    }

    return 0;
}

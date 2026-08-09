// SPDX-License-Identifier: Apache-2.0
// custom_problem.cpp — how to define and solve YOUR OWN problem, fully
// self-contained (no external evaluator, everything lives in this file).
//
// Problem: Kursawe (1991) — a classic bi-objective benchmark with a
// non-convex, disconnected Pareto front. 3 real variables in [-5, 5]:
//
//   f1(x) = sum_{i=1..2} [ -10 * exp(-0.2 * sqrt(x_i^2 + x_{i+1}^2)) ]
//   f2(x) = sum_{i=1..3} [ |x_i|^0.8 + 5 * sin(x_i^3) ]
//
// Recipe for a custom problem (duck-typed, no virtuals):
//   1. Declare a tag individual derived from the individual type of the
//      algorithm you want to run (here: NSGAII_Individual).
//   2. Specialize mootation::Problem<> for that tag: report the numbers of
//      real/binary variables, objectives and constraint limits, provide
//      per-variable bounds, and implement calc_objs().
//   3. Instantiate DataVault + Optimizer with the tag type as usual.
//
// Build & run:
//   g++ -std=c++17 -O2 -Iinclude examples/custom_problem.cpp -o custom_problem
//   ./custom_problem

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

#include <mootation/mootation.hpp>

// ---------------------------------------------------------------------------
// 1. Tag individual: "an NSGA-II individual evaluated on Kursawe".
// ---------------------------------------------------------------------------
namespace mootation::problems {
struct Kursawe_NSGAII : public NSGAII_Individual {};
} // namespace mootation::problems

// ---------------------------------------------------------------------------
// 2. Problem specialization for the tag.
// ---------------------------------------------------------------------------
namespace mootation {

template <>
class Problem<problems::Kursawe_NSGAII> {
    static constexpr int NVARS = 3;

public:
    // One {lower, upper} pair per real variable (std::nullopt = unbounded).
    std::vector<std::pair<std::optional<double>, std::optional<double>>>
        bounds = {{-5.0, 5.0}, {-5.0, 5.0}, {-5.0, 5.0}};

    int get_vars_n()     const { return NVARS; } // real-valued variables
    int get_bin_vars_n() const { return 0; }     // no binary part
    int get_objs_n()     const { return 2; }     // f1, f2
    int get_lims_n()     const { return 0; }     // unconstrained

    void calc_objs(problems::Kursawe_NSGAII& ind) const {
        const auto& x = ind.variables;

        double f1 = 0.0;
        for (int i = 0; i + 1 < NVARS; ++i)
            f1 += -10.0 * std::exp(-0.2 * std::sqrt(x[i] * x[i] +
                                                    x[i + 1] * x[i + 1]));

        double f2 = 0.0;
        for (int i = 0; i < NVARS; ++i)
            f2 += std::pow(std::abs(x[i]), 0.8) +
                  5.0 * std::sin(x[i] * x[i] * x[i]);

        ind.objectives[0] = f1;
        ind.objectives[1] = f2;
    }
};

} // namespace mootation

// ---------------------------------------------------------------------------
// 3. Solve it.
// ---------------------------------------------------------------------------
int main() {
    using namespace mootation;
    using Ind = problems::Kursawe_NSGAII;

    Problem<Ind>   prob;
    DataVault<Ind> vault(100, prob);   // population size 100

    Optimizer<Ind, NSGAIICore<Ind>> opt(std::move(vault));
    opt.optimize(250);                 // 250 generations

    auto& v = opt.get_vault();
    io::save_population(v, "pop_kursawe.txt",
                        "NSGA-II | Kursawe | pop=100 | iter=250");

    std::cout << "=== NSGA-II on Kursawe (first 10 of "
              << v.active_n() << " solutions) ===\n"
              << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < std::min<std::size_t>(10, v.active_n()); ++i) {
        const auto& f = v.objectives_of(i);
        std::cout << "f1 = " << std::setw(9) << f[0]
                  << "   f2 = " << std::setw(9) << f[1] << '\n';
    }
    return 0;
}

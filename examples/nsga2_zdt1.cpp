// SPDX-License-Identifier: Apache-2.0
// NSGA-II on plain ZDT1.
#include <iomanip>
#include <iostream>

#include <mootation/mootation.hpp>

int main()
{
    using namespace mootation;

    Problem<NSGAII_Individual>   prob;
    DataVault<NSGAII_Individual> vault(100, prob);

    // Seeded so this example reproduces run to run.
    //
    // `defer_setup` is required for that: the one-argument Optimizer
    // constructor calls setup() from its own body, and setup() draws the
    // initial population there and then. Calling set_seed() afterwards would
    // change nothing that has already been drawn. With defer_setup the caller
    // owns the order — configure first, then setup().
    Optimizer<NSGAII_Individual, NSGAIICore<NSGAII_Individual>>
        opt(std::move(vault), defer_setup);

    opt.get_algorithm().set_seed(12345);
    opt.setup();

    opt.optimize(500);

    auto& v = opt.get_vault();
    io::save_population(v, "pop_nsga2.txt",
                        "NSGA-II | ZDT1 | pop=100 | iter=500");

    std::cout << "\n=== NSGA-II (ZDT1) ===\n";
    std::cout << std::fixed << std::setprecision(5);
    for (std::size_t i = 0; i < std::min<std::size_t>(10, v.active_n()); ++i) {
        const auto& o = v.objectives_of(i);
        std::cout << "f1=" << o[0] << "  f2=" << o[1] << "\n";
    }
    return 0;
}

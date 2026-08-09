// SPDX-License-Identifier: Apache-2.0
// IBEA e+ on plain ZDT1 (6 real, 0 bin).
#include <iomanip>
#include <iostream>

#include <mootation/mootation.hpp>

int main()
{
    using namespace mootation;

    Problem<IBEA_Individual>   prob;
    DataVault<IBEA_Individual> vault(100, prob);
    Optimizer<IBEA_Individual, IBEAePlusCore<IBEA_Individual>>
        opt(std::move(vault));
    opt.optimize(500);

    auto& v = opt.get_vault();
    io::save_population(v, "pop_ibea.txt",
                        "IBEA e+ | ZDT1 | pop=100 | iter=500");

    std::cout << "\n=== IBEA e+ (ZDT1) ===\n";
    std::cout << std::fixed << std::setprecision(5);
    for (std::size_t i = 0; i < std::min<std::size_t>(10, v.active_n()); ++i) {
        const auto& o = v.objectives_of(i);
        std::cout << "f1=" << o[0] << "  f2=" << o[1] << "\n";
    }
    return 0;
}

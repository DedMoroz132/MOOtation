// SPDX-License-Identifier: Apache-2.0
// IBEA e+ on the mixed problem ZDT1Mixed (2 real + 6 bin).
#include <iomanip>
#include <iostream>

#include <mootation/mootation.hpp>
#include "../problems/zdt1_mixed.hpp"

int main()
{
    using namespace mootation;
    using Ind = problems::ZDT1Mixed_IBEA;

    Problem<Ind>   prob;
    DataVault<Ind> vault(60, prob);
    Optimizer<Ind, IBEAePlusCore<Ind>> opt(std::move(vault));
    opt.optimize(400);

    auto& v = opt.get_vault();
    std::cout << "\n=== IBEA e+: 2 real + 6 binary ===\n";
    std::cout << std::fixed << std::setprecision(4);

    for (std::size_t i = 0; i < std::min<std::size_t>(10, v.active_n()); ++i) {
        int ones = 0;
        for (int j = 0; j < v.bin_vars_n(); ++j)
            ones += v.get_bin_variable(i, j);
        const auto& o = v.objectives_of(i);
        std::cout << "f1=" << o[0] << "  f2=" << o[1]
                  << "  bits=" << ones << "/" << v.bin_vars_n() << "\n";
    }
    io::save_population(v, "pop_mixed.txt",
                        "IBEA e+ mixed | 2real+6bin | pop=60 | iter=400");
    return 0;
}

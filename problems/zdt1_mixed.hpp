#pragma once
// SPDX-License-Identifier: Apache-2.0

// ZDT1-mixed: 2 real + 6 binary variables.
// g depends on the number of ones in the bit mask.
// Optimum: all bits = 0, x1 = 0 → g = 1 → the pure ZDT1 front.
// With bits = 6 → g = 10 → the front degrades.
//
// Tag: ZDT1Mixed — a pseudo-type on which the specialization is made.
// This keeps the default Problem<IBEA_Individual> untouched.

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "../include/mootation/individuals.hpp"
#include "../include/mootation/problem.hpp"

namespace mootation::problems {

// Wrapper individual for the Problem<> specialization.
struct ZDT1Mixed_IBEA   : public IBEA_Individual   {};
struct ZDT1Mixed_NSGAII : public NSGAII_Individual {};

} // namespace mootation::problems

namespace mootation {

template <>
class Problem<problems::ZDT1Mixed_IBEA> {
    static constexpr int NREAL = 2;
    static constexpr int NBIN  = 6;
public:
    int get_vars_n()     const { return NREAL; }
    int get_bin_vars_n() const { return NBIN;  }
    int get_objs_n()     const { return 2;     }
    int get_lims_n()     const { return 0;     }

    std::vector<std::pair<std::optional<double>, std::optional<double>>>
        bounds = {{0.0, 1.0}, {0.0, 1.0}};

    void calc_objs(problems::ZDT1Mixed_IBEA& ind) const {
        double ones = 0.0;
        for (int b : ind.binary_variables) ones += b;
        double g = 1.0 + 9.0 * (ind.variables[1] + ones / NBIN) / 2.0;
        ind.objectives[0] = ind.variables[0];
        ind.objectives[1] = g * (1.0 - std::sqrt(ind.objectives[0] / g));
    }
};

template <>
class Problem<problems::ZDT1Mixed_NSGAII> {
    static constexpr int NREAL = 2;
    static constexpr int NBIN  = 6;
public:
    int get_vars_n()     const { return NREAL; }
    int get_bin_vars_n() const { return NBIN;  }
    int get_objs_n()     const { return 2;     }
    int get_lims_n()     const { return 0;     }

    std::vector<std::pair<std::optional<double>, std::optional<double>>>
        bounds = {{0.0, 1.0}, {0.0, 1.0}};

    void calc_objs(problems::ZDT1Mixed_NSGAII& ind) const {
        double ones = 0.0;
        for (int b : ind.binary_variables) ones += b;
        double g = 1.0 + 9.0 * (ind.variables[1] + ones / NBIN) / 2.0;
        ind.objectives[0] = ind.variables[0];
        ind.objectives[1] = g * (1.0 - std::sqrt(ind.objectives[0] / g));
    }
};

} // namespace mootation

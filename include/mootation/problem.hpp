#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "individuals.hpp"

namespace mootation {

// Default Problem template — ZDT1 on 6 real-valued variables.
// The user writes their own specialization: see problems/zdt1.hpp as an example.
//
// Contract (duck-typed, no virtual):
//   int  get_vars_n()     const;
//   int  get_bin_vars_n() const;
//   int  get_objs_n()     const;
//   int  get_lims_n()     const;
//   void calc_objs(Ind_type& ind) const;
//   std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds;
template <typename Ind_type>
class Problem {
private:
    int num_vars = 6, num_objs = 2, num_lims = 0, num_bin_vars = 0;

public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0},
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    void set_bin_vars(int n)  { num_bin_vars = n; }
    int  get_vars_n()     const { return num_vars; }
    int  get_bin_vars_n() const { return num_bin_vars; }
    int  get_objs_n()     const { return num_objs; }
    int  get_lims_n()     const { return num_lims; }

    void calc_objs(Ind_type& ind) const {
        // ZDT1: f1 = x0; f2 = g(1 - sqrt(f1/g)), g = 1 + 9*sum(x1..x5)/5
        double g = 1.0 + 9.0 * (ind.variables[1] + ind.variables[2] +
                                ind.variables[3] + ind.variables[4] +
                                ind.variables[5]) / 5.0;
        ind.objectives[0] = ind.variables[0];
        ind.objectives[1] = g * (1.0 - std::sqrt(ind.objectives[0] / g));
    }
};

} // namespace mootation

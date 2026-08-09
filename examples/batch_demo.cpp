// SPDX-License-Identifier: Apache-2.0
// Batch executor demo — simulates an external solver
// that takes a batch of individuals and returns a batch of objectives.
// In production this will be a pybind11/Python call (MAPDL/E+).
//
// To keep Problem::calc_objs free of any background role (we want the
// objectives to be computed ONLY by the batch executor), an "empty"
// Problem specialization is used.
//
// Pipeline:
//   step():
//     expand() → set_variables N times → sync() → batch_executor(req, resp)
//                                                    ↑
//                                          this is where Python plugs in
#include <iostream>
#include <iomanip>
#include <cmath>

#include <mootation/mootation.hpp>

namespace mootation {

// Tag individual and a no-op Problem: calc_objs does nothing,
// all computation is delegated to the batch executor.
struct BatchOnly_IBEA : public IBEA_Individual {};

template <>
class Problem<BatchOnly_IBEA> {
public:
    int get_vars_n()     const { return 6; }
    int get_bin_vars_n() const { return 0; }
    int get_objs_n()     const { return 2; }
    int get_lims_n()     const { return 0; }
    std::vector<std::pair<std::optional<double>, std::optional<double>>>
        bounds = {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}};
    void calc_objs(BatchOnly_IBEA&) const {
        // intentionally empty — the real computation is done by the batch executor
    }
};

} // namespace mootation

int main()
{
    using namespace mootation;

    Problem<BatchOnly_IBEA>   prob;
    DataVault<BatchOnly_IBEA> vault(40, prob);

    int n_batch_calls = 0;
    int n_individuals_evaluated = 0;

    // Simulation of an external pool: receive a batch, compute ZDT1, return it.
    vault.set_batch_executor([&](const BatchRequest& req, BatchResponse& resp) {
        ++n_batch_calls;
        n_individuals_evaluated += static_cast<int>(req.size());

        resp.resize(req.size(), /*n_objs*/ 2, /*n_lims*/ 0);
        for (std::size_t i = 0; i < req.size(); ++i) {
            const auto& x = req.variables[i];
            // ZDT1: f1 = x0; f2 = g (1 - sqrt(f1/g)); g = 1 + 9*sum(x1..x5)/5
            double g = 1.0 + 9.0 * (x[1] + x[2] + x[3] + x[4] + x[5]) / 5.0;
            resp.objectives[i][0] = x[0];
            resp.objectives[i][1] = g * (1.0 - std::sqrt(x[0] / g));
        }
    });

    Optimizer<BatchOnly_IBEA, IBEAePlusCore<BatchOnly_IBEA>>
        opt(std::move(vault));
    opt.optimize(50);

    auto& v = opt.get_vault();
    std::cout << "\n=== batch_demo ===\n"
              << "batch calls: " << n_batch_calls
              << "  individuals: " << n_individuals_evaluated << "\n"
              << "front (top 10 by f1):\n";
    std::cout << std::fixed << std::setprecision(5);
    for (std::size_t i = 0; i < std::min<std::size_t>(10, v.active_n()); ++i) {
        const auto& o = v.objectives_of(i);
        std::cout << "  f1=" << o[0] << "  f2=" << o[1] << "\n";
    }
    return 0;
}

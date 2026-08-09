#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <functional>
#include <vector>

namespace mootation {

// ============================================================
//  BatchExecutor: interface for batch evaluation of objectives.
//
//  Why: an external evaluator (MAPDL/EnergyPlus/...) is often more
//  efficient when it takes a batch of individuals at once — the worker
//  pool stays maximally busy. A slot-by-slot executor (set_executor)
//  does not provide this.
//
//  How it works (BATCH eval mode):
//    1. Algorithm: set_variables N times → each slot gets dirty=1
//    2. vault.sync() gathers all dirty real_idx + their vars/bin_vars
//       into a BatchRequest and calls batch_executor(req, resp)
//    3. The executor must fill resp.objectives and resp.limits
//       (one row per dirty slot, in the same order).
//    4. DataVault writes obj/limits back into the slots, computes CV,
//       and clears dirty.
//
//  Executor contract:
//    - resp.objectives.size() == req.size()
//    - each resp.objectives[i].size() == n_objectives
//    - resp.limits.size() == 0  OR  == req.size()
//      (if empty — all limits are interpreted as zeros = OK)
//    - each resp.limits[i].size() == n_limits (if limits are provided)
//    - the order strictly matches req.real_indices / req.variables
//    - the executor may throw an exception — sync() will skip the
//      write-back and keep dirty=1 (individuals not evaluated →
//      ensure_ready will retry)
// ============================================================

struct BatchRequest {
    std::vector<std::size_t>          real_indices;     // r-indices of dirty slots
    std::vector<std::vector<double>>  variables;        // [N][num_vars]
    std::vector<std::vector<int>>     binary_variables; // [N][num_bin_vars] (may be empty if bin=0)

    std::size_t size() const { return real_indices.size(); }
};

struct BatchResponse {
    std::vector<std::vector<double>> objectives;        // [N][num_objs]
    std::vector<std::vector<double>> limits;            // [N][num_lims] or empty

    void resize(std::size_t n, int n_objs, int n_lims) {
        objectives.assign(n, std::vector<double>(n_objs, 0.0));
        if (n_lims > 0) limits.assign(n, std::vector<double>(n_lims, 0.0));
        else            limits.clear();
    }
};

using BatchExecutor = std::function<void(const BatchRequest&, BatchResponse&)>;

} // namespace mootation

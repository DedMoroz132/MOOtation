#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The resumable strategies' runs behind the interface of resumable.hpp (task
// 31, layer 3): which strategies have one, and start_resumable. Included at
// the end of strategy.hpp, after every strategy. The runs themselves:
// basin_hopping and ils — Walk (search.hpp); cluster_ga — ClusterGa
// (cluster_ga.hpp); ga_sbx and ssga — GaRun (scalar.hpp).
// ============================================================================

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "resumable.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

inline const std::vector<std::string>& resumable_names()
{
    static const std::vector<std::string> v{"basin_hopping", "ils", "cluster_ga", "ga_sbx", "ssga"};
    return v;
}

inline std::unique_ptr<Resumable> start_resumable(const std::string& name, const Problem& P, double eq_tol,
                                                  bool has_tol0, double eq_tol0, int pop_size, long long budget,
                                                  unsigned seed, const Knobs& knobs, Observer observer,
                                                  int record_every, const Start& start)
{
    if (!detail::listed(resumable_names(), name))
        throw std::invalid_argument(name + " is not resumable (resumable.hpp: resumable_names())");
    // RES-8: the time of scalar::run: the problem with its deadline and clock, and what the result gets after
    const detail::RunTime t = detail::run_time(P, knobs);
    std::unique_ptr<Problem> own = t.timed ? std::make_unique<Problem>(detail::timed_problem(P, t)) : nullptr;
    const Problem& Q = own ? *own : P;
    const Knobs& kn = t.timed ? t.rest : knobs;
    auto tail = [t, name](Result& r) { detail::run_tail(name, t, r); };
    if (name == "basin_hopping" || name == "ils") {
        auto w = std::make_unique<Walk>(name, Q, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, kn,
                                        std::move(observer), record_every, start);
        w->own = std::move(own);
        w->tail = tail;
        return w;
    }
    if (name == "cluster_ga") {
        auto c = std::make_unique<ClusterGa>(Q, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, kn,
                                             std::move(observer), record_every, start);
        c->own = std::move(own);
        c->tail = tail;
        return c;
    }
    auto g = std::make_unique<GaRun>(name, Q, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, kn,
                                     std::move(observer), record_every, start);
    g->own = std::move(own);
    g->tail = tail;
    return g;
}

}   // namespace mootation::scalar

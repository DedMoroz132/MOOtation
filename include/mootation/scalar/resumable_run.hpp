#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The resumable strategies' runs behind the interface of resumable.hpp (task
// 31, layer 3): which strategies have one, and start_resumable. Included at
// the end of strategy.hpp, after every strategy.
// ============================================================================

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "resumable.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

inline const std::vector<std::string>& resumable_names()
{
    static const std::vector<std::string> v{};
    return v;
}

inline std::unique_ptr<Resumable> start_resumable(const std::string& name, const Problem& P, double eq_tol,
                                                  bool has_tol0, double eq_tol0, int pop_size, long long budget,
                                                  unsigned seed, const Knobs& knobs, Observer observer,
                                                  int record_every, const Start& start)
{
    (void)P;
    (void)eq_tol;
    (void)has_tol0;
    (void)eq_tol0;
    (void)pop_size;
    (void)budget;
    (void)seed;
    (void)knobs;
    (void)observer;
    (void)record_every;
    (void)start;
    throw std::invalid_argument(name + " is not resumable (resumable.hpp: resumable_names())");
}

}   // namespace mootation::scalar

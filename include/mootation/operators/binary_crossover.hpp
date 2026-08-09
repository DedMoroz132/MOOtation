#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <vector>

namespace mootation::ops {

// Uniform crossover for binary variables.
template <typename RNG>
inline void binary_crossover(const std::vector<int>& p1,
                             const std::vector<int>& p2,
                             std::vector<int>& c1,
                             std::vector<int>& c2,
                             RNG& rng)
{
    std::uniform_int_distribution<int> coin(0, 1);
    int nb = static_cast<int>(p1.size());
    c1.resize(nb); c2.resize(nb);
    for (int j = 0; j < nb; ++j) {
        if (coin(rng)) { c1[j] = p1[j]; c2[j] = p2[j]; }
        else           { c1[j] = p2[j]; c2[j] = p1[j]; }
    }
}

} // namespace mootation::ops

#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <vector>

namespace mootation::ops {

// Bit-flip mutation: each bit is inverted with probability 1/n_bits.
template <typename RNG>
inline void bit_flip_mutation(std::vector<int>& bvars, int n_bits, RNG& rng)
{
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double p = 1.0 / static_cast<double>(n_bits > 0 ? n_bits : 1);
    for (int j = 0; j < n_bits; ++j)
        if (uni(rng) < p) bvars[j] ^= 1;
}

} // namespace mootation::ops

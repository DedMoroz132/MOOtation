#pragma once
// SPDX-License-Identifier: Apache-2.0

#include "bc_spx.hpp"

namespace mootation {

// ============================================================================
// BC with REX — Tanabe & Ishibuchi, GECCO 2019: the best-children selection of
// bc_spx.hpp (BCCore, its parameters and readings BC-1..5) with the real-coded
// ensemble crossover, ξ ~ N(0, 1/(k − 1)) (Table 1; operators/multi_parent.hpp).
// Real variables only.
// ============================================================================

template <typename Ind_t>
class BCREXCore : public BCCore<Ind_t> {
public:
    BCREXCore() : BCCore<Ind_t>(ops::Crossover::REX) {}
};

}   // namespace mootation

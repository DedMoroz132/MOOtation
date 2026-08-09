// SPDX-License-Identifier: Apache-2.0
#pragma once
// ============================================================================
// What to do when a seed population does not have pop_size individuals.
//
// This lives in its own header for the same reason constraint_mode.hpp does:
// Settings needs to name the policy, and io/population.hpp needs to implement
// it. Without a shared header the plain settings struct would have to include
// the whole population/vault machinery to spell one enum.
//
// A mismatch is routine rather than exceptional. A population saved by one run
// is often reused at a different pop_size, and for the NSGA-III family and the
// M2M decomposition pop_size is constrained by the algorithm itself, so the
// same file simply cannot fit every run it is useful for. The policy belongs to
// the caller because the right answer depends on why they are restarting.
// ============================================================================

namespace mootation {

enum class SizeMismatch {
    Error,      // refuse: the caller wanted exactly this population
    Truncate,   // keep the first pop_size rows (the file is sorted by f1)
    Pad         // cycle the rows until pop_size is reached
};

} // namespace mootation

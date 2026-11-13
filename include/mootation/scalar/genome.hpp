#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A genome's decoding (task 24, §24.1.2): what Problem::decode (evaluator.hpp)
// gives back for a genome x of fixed length — the canonical genome, written
// back into the population (Lamarckian: the genome becomes x_canon before it
// is evaluated), the genes that are active (empty: all; the variables a
// local search may move, §24.1.1), and whether the genome decodes at all.
// A genome that does not (valid = false; a grammatical evolution's genome
// past its wrapping limit) is not evaluated and costs nothing: the GA gives
// it the worst values (ga.hpp GA-11). The structural genomes of task 27
// (trees, grammatical evolution) pack into such a vector and come with
// their own operators (Problem::crossover, mutate, repair, sample).
// ============================================================================

#include <vector>

namespace mootation::scalar {

struct Decoded {
    std::vector<double> x_canon;
    std::vector<char> active;
    bool valid = true;
};

}   // namespace mootation::scalar

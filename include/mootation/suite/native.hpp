#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A problem of the C++ suite (suite.hpp) as the run evaluates it: the
// interface alone, without the arithmetic. The Python binding's run code
// (python/mootation_module.cpp) calls problems through it and includes nothing
// else of the suite, whose arithmetic is compiled in python/suite_module.cpp
// alone, without contraction into fused multiply-adds (common.hpp).
// ============================================================================

namespace mootation::suite {

// x of n_var values to n_obj objectives in f and n_con constraint values in g
// (<= 0 satisfied; g may be null when n_con is 0). const and without state:
// one object serves any number of threads.
struct Native {
    int n_var = 0, n_obj = 0, n_con = 0;
    virtual ~Native() = default;
    virtual void evaluate(const double* x, double* f, double* g) const = 0;
};

}   // namespace mootation::suite

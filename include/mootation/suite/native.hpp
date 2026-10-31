#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A problem of the C++ suite (suite.hpp) as the run evaluates it: the
// interface alone, without the arithmetic. The Python binding's run code
// (python/mootation_module.cpp) calls problems through it and includes nothing
// else of the suite, whose arithmetic is compiled in python/suite_module.cpp
// alone, without contraction into fused multiply-adds (common.hpp).
// ============================================================================

#include <stdexcept>

namespace mootation::suite {

// x of n_var values to n_obj objectives in f and n_con constraint values in g
// (<= 0 satisfied; g may be null when n_con is 0). const and without state:
// one object serves any number of threads.
//
// A least-squares problem (task 18, stage 1: MGH, NIST) gives its n_res
// residuals too, its objective 1/2 ||r||^2, and with has_jacobian their
// Jacobian J (n_res rows of n_var, row-major; J may be null).
struct Native {
    int n_var = 0, n_obj = 0, n_con = 0;
    int n_res = 0;                     // 0: no residuals
    bool has_jacobian = false;
    virtual ~Native() = default;
    virtual void evaluate(const double* x, double* f, double* g) const = 0;
    virtual void residuals(const double*, double*, double*) const
    {
        throw std::logic_error("suite: the problem gives no residuals");
    }
};

}   // namespace mootation::suite

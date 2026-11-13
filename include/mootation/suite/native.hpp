#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A problem of the C++ suite (suite.hpp) as the run evaluates it: the
// interface alone, without the arithmetic. The Python binding's run code
// (python/mootation_module.cpp) calls problems through it and includes nothing
// else of the suite, whose arithmetic is compiled in python/suite_module.cpp
// alone, without contraction into fused multiply-adds (common.hpp).
// ============================================================================

#include <limits>
#include <stdexcept>
#include <vector>

namespace mootation::suite {

// count blocks of size coordinates from offset, which may be interchanged
// (task 19, §19.1: blocks of a problem with symmetries)
struct Blocks {
    int offset = 0, size = 0, count = 0;
};

// x of n_var values to n_obj objectives in f and n_con constraint values in g
// (<= 0 satisfied; g may be null when n_con is 0). const and without state:
// one object serves any number of threads.
//
// A least-squares problem (task 18, stage 1: MGH, NIST) gives its n_res
// residuals too, its objective 1/2 ||r||^2, and with has_jacobian their
// Jacobian J (n_res rows of n_var, row-major; J may be null).
//
// A problem with symmetries (task 19, §19.1: sym.hpp's families) says which
// coordinates may be interchanged (blocks; the rest are free), gives its
// canonical representative canon(x) where it has one (f(canon(x)) = f(x),
// canon(canon(x)) = canon(x)), and gauge_dim directions along which its
// residuals (or its objective) do not change to first order, the columns of
// gauge_basis(x) (n_var rows of gauge_dim, row-major). linear: the parameters
// the residuals are linear in (for variable projection). An objective that is
// not a sum of squares may give its gradient (has_gradient). start(x0): the
// problem's own start where it has one (has_start). res_blocks: the sizes of
// the residuals' blocks, in order (task 19, §19.5; they add up to n_res;
// empty: one block), which blocks_objectives.hpp makes objectives of; the
// problem's own objective stays 1/2 ||r||^2 whatever they say.
struct Native {
    int n_var = 0, n_obj = 0, n_con = 0;
    int n_res = 0;                     // 0: no residuals
    bool has_jacobian = false;
    std::vector<Blocks> blocks;        // empty: no interchangeable coordinates
    std::vector<int> linear;
    bool has_canon = false, has_gradient = false, has_start = false;
    int gauge_dim = 0;
    std::vector<int> res_blocks;       // empty: one block of n_res
    // a measure of success of the problem's own (task 24, §24.1.5; scalar/
    // evaluator.hpp THE SUCCESS): asked at the feasible points with f <=
    // post_at in a campaign with [campaign] success = "user"; each call costs
    // success_cost of the budget (declared). It is const, as evaluate is.
    bool has_success = false;
    double post_at = std::numeric_limits<double>::infinity(), success_cost = 0.0;
    virtual ~Native() = default;
    virtual void evaluate(const double* x, double* f, double* g) const = 0;
    virtual void residuals(const double*, double*, double*) const
    {
        throw std::logic_error("suite: the problem gives no residuals");
    }
    virtual void canon(double*) const { throw std::logic_error("suite: the problem has no canonical form"); }
    virtual void gauge_basis(const double*, double*) const
    {
        throw std::logic_error("suite: the problem gives no gauge directions");
    }
    virtual void gradient(const double*, double*) const
    {
        throw std::logic_error("suite: the problem gives no gradient");
    }
    virtual void start(double*) const { throw std::logic_error("suite: the problem has no start of its own"); }
    virtual bool accept_success(const double*) const
    {
        throw std::logic_error("suite: the problem has no measure of success of its own");
    }
};

}   // namespace mootation::suite

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Residual blocks as objectives (task 19, §19.5): a problem whose residuals
// come in B blocks, r = (r_1, ..., r_B) (Native::res_blocks), seen as a
// problem of B objectives
//
//   f_b(x) = 1/2 w_b ||r_b(x)||^2,   b = 1 ... B,
//
// one call of the residuals for all B. The idea is H.V. Gupta, S.
// Sorooshian, P.O. Yapo, "Toward improved calibration of hydrologic models:
// Multiple and noncommensurable measures of information", Water Resour. Res.
// 34(4), 1998, 751-763 (source multiobjective-calibration_gupta1998): the
// calibration stated as min F(theta) = {f_1(theta), ..., f_m(theta)} (their
// (2)), a Pareto set of trade-offs, rather than one weighted sum w_1 f_1 +
// ... + w_m f_m (their (3)). The paper has no algorithm to reproduce here:
// the adapter only states the problem, which any multi-objective core then
// solves; their measures (DRMS, BIAS, NSC) are not sums of squares and are
// not the adapter's.
//
// Readings (OURS):
//   BO-1. f_b: the squares of block b's residuals added in order, then
//     (0.5 w_b) times that sum; w_b = 1 gives 0.5 s_b, the suite's 1/2
//     ||r||^2 of one block, bit for bit.
//   BO-2. total(f) = sum_b f_b, added in block order, is the problem's
//     objective at M = 1 (1/2 sum_b w_b ||r_b||^2) up to rounding: the same
//     nonnegative terms grouped by block. Each of the two sums is the exact
//     sum of its terms times (1 + e_i), |e_i| <= gamma_{n_res} = n_res u /
//     (1 - n_res u) (no term takes part in more than n_res - 1 additions,
//     and the factor w_b is one rounding more; N.J. Higham, Accuracy and
//     Stability of Numerical Algorithms, 2nd ed., 2002, §4.2, after (4.4);
//     source accuracy-stability_higham2002), so they differ by at most 2
//     gamma_{n_res} of it. One block of weight 1: the same bit for bit.
//   BO-3. Any B >= 1; above 8 the multi-objective cores lose their selection
//     pressure, and the Python front (minimize) warns. The adapter itself
//     says nothing (a library header).
//   BO-4. What else the problem gives passes through: its residuals and
//     their Jacobian, its symmetries (blocks, canon: each f_b is invariant
//     where r is), its gauge directions and its start. Its gradient does not
//     (one objective's). A problem with constraints, or with more than one
//     objective, is refused: the residual problems of the suite have neither.
//   BO-5. The weights are the adapter's (default 1 each), not the problem's:
//     the problem's own objective is 1/2 ||r||^2 (native.hpp).
// ============================================================================

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "native.hpp"

namespace mootation::suite {

// The blocks' sizes of a problem with n_res residuals (empty res_blocks: one
// block), checked.
inline std::vector<int> residual_blocks(const std::vector<int>& sizes, int n_res)
{
    if (n_res < 1) throw std::invalid_argument("blocks_objectives: the problem gives no residuals");
    if (sizes.empty()) return {n_res};
    long long sum = 0;
    for (int s : sizes) {
        if (s < 1) throw std::invalid_argument("blocks_objectives: every block needs a residual");
        sum += s;
    }
    if (sum != n_res) throw std::invalid_argument("blocks_objectives: the blocks' sizes must add up to n_res");
    return sizes;
}

// f_b of the residuals r (BO-1); weights empty: 1 each.
inline void block_objectives(const double* r, const std::vector<int>& sizes, const std::vector<double>& weights,
                             double* f)
{
    std::size_t at = 0;
    for (std::size_t b = 0; b < sizes.size(); ++b) {
        double s = 0.0;
        for (int k = 0; k < sizes[b]; ++k, ++at) s += r[at] * r[at];
        f[b] = (0.5 * (weights.empty() ? 1.0 : weights[b])) * s;
    }
}

// sum_b f_b in block order (BO-2).
inline double block_total(const double* f, std::size_t B)
{
    double t = 0.0;
    for (std::size_t b = 0; b < B; ++b) t += f[b];
    return t;
}

class BlocksObjectives final : public Native {
public:
    BlocksObjectives(std::shared_ptr<const Native> problem, std::vector<double> weights = {})
        : p_(std::move(problem)), weights_(std::move(weights))
    {
        if (!p_) throw std::invalid_argument("blocks_objectives: no problem");
        if (p_->n_obj != 1 || p_->n_con != 0)
            throw std::invalid_argument("blocks_objectives: a problem of one objective without constraints (BO-4)");
        sizes_ = residual_blocks(p_->res_blocks, p_->n_res);
        if (!weights_.empty() && weights_.size() != sizes_.size())
            throw std::invalid_argument("blocks_objectives: one weight per block (" + std::to_string(sizes_.size()) +
                                        ")");
        for (double w : weights_)
            if (!(w >= 0.0) || !std::isfinite(w))
                throw std::invalid_argument("blocks_objectives: the weights must be finite and >= 0");
        n_var = p_->n_var;
        n_obj = static_cast<int>(sizes_.size());
        n_res = p_->n_res;
        has_jacobian = p_->has_jacobian;
        blocks = p_->blocks;
        linear = p_->linear;
        has_canon = p_->has_canon;
        has_start = p_->has_start;
        gauge_dim = p_->gauge_dim;
        res_blocks = p_->res_blocks;
    }

    void evaluate(const double* x, double* f, double*) const override
    {
        std::vector<double> r(static_cast<std::size_t>(n_res));
        p_->residuals(x, r.data(), nullptr);
        block_objectives(r.data(), sizes_, weights_, f);
    }

    void residuals(const double* x, double* r, double* J) const override { p_->residuals(x, r, J); }
    void canon(double* x) const override { p_->canon(x); }
    void gauge_basis(const double* x, double* N) const override { p_->gauge_basis(x, N); }
    void start(double* x) const override { p_->start(x); }

    double total(const double* f) const { return block_total(f, sizes_.size()); }
    const std::vector<int>& sizes() const { return sizes_; }
    const std::vector<double>& weights() const { return weights_; }
    const Native& problem() const { return *p_; }

private:
    std::shared_ptr<const Native> p_;
    std::vector<double> weights_;
    std::vector<int> sizes_;
};

}   // namespace mootation::suite

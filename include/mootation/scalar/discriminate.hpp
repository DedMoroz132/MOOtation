#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// T-optimum designs for discriminating between two models (task 26, item
// 24.7.6): A. C. Atkinson, V. V. Fedorov, "The design of experiments for
// discriminating between two rival models", Biometrika 62(1) (1975) 57-70
// (source design-experiments-discriminating_atkinson1975; "AF75a").
//
// The first model is taken as true, its responses eta_t(x) known at the
// candidate points x_1 ... x_C; the rival model is fitted to them by
// weighted least squares, fit2(p) -> its predictions at the candidates under
// the design weights p (any fit: a linear least squares, lm, the atomic
// measure of atomic_measure.hpp — the caller's). Then
//   psi_2(x, xi) = {eta_t(x) - eta_2(x, theta_2(xi))}^2,
//   Delta_2(xi) = sum_i p_i psi_2(x_i, xi)                     (AF75a (2)),
// and the sequential construction of §3 with sigma = 0: at stage s the
// candidate of the largest psi_2 receives one more trial, xi_{s+1} = (1 -
// alpha_s) xi_s + alpha_s xi(x_{s+1}), alpha_s = 1/(N + 1), N the trials so
// far (AF75a §3, Example 2's "alpha_s = 1/(N + 1)"). By the equivalence
// theorem (Theorem 1) the T-optimum design has sup_x psi_2 = Delta_2 at its
// points: the run stops where max psi_2 <= (1 + tol) Delta_2, or after
// `steps` trials.
//
// READINGS:
//   TD-1. The candidates are a finite set (AF75a's Example 2 searches a grid
//     of 21 points): the sup over X is the largest psi over them; the first
//     candidate on ties.
//   TD-2. fit2 receives the weights of every candidate (zero off the design)
//     and must return C predictions; a design whose weights do not determine
//     the rival's parameters is the caller's to avoid (AF75a starts from
//     three points for a three-parameter rival).
//   TD-3. The weights returned are the trials over N (the design xi_N); the
//     trials themselves are kept, and the candidate chosen at each stage.
//   TD-4. AF75a's Example 2 gives eta_t(1) = -0.131; 4.5 - 1.5 e - 2/e is
//     -0.313 (its other values, -1.488 at -1 and the maximum 1.036 at 0.144,
//     agree): read as the digits transposed.
//   Not here: several models and T_p-optimality (Atkinson & Fedorov 1975b),
//   the designs with the error sigma > 0 (simulated responses).
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace mootation::scalar::discriminate {

struct Result {
    std::vector<long long> trials;       // per candidate
    std::vector<double> weights;          // trials / N (TD-3)
    std::vector<std::size_t> chosen;      // the candidate added at each stage
    double delta = 0.0;                   // Delta_2 of the final design
    double psi_max = 0.0;                 // the largest psi_2 at the final design
    std::size_t argmax = 0;
    long long steps = 0;
    bool certified = false;               // psi_max <= (1 + tol) delta
};

inline Result t_optimal(const std::vector<double>& eta_t,
                        const std::function<std::vector<double>(const std::vector<double>& p)>& fit2,
                        const std::vector<std::size_t>& initial, long long steps, double tol = 1e-3)
{
    const std::size_t C = eta_t.size();
    if (C == 0) throw std::invalid_argument("t_optimal: no candidates");
    if (initial.empty()) throw std::invalid_argument("t_optimal: an initial design of at least one trial");
    if (!fit2) throw std::invalid_argument("t_optimal: the rival model's fit");
    Result r;
    r.trials.assign(C, 0);
    long long N = 0;
    for (std::size_t i : initial) {
        if (i >= C) throw std::invalid_argument("t_optimal: an initial trial past the candidates");
        ++r.trials[i];
        ++N;
    }
    auto evaluate = [&]() {
        std::vector<double> p(C);
        for (std::size_t c = 0; c < C; ++c) p[c] = static_cast<double>(r.trials[c]) / static_cast<double>(N);
        const std::vector<double> pred = fit2(p);
        if (pred.size() != C) throw std::runtime_error("t_optimal: the fit gave predictions of another length (TD-2)");
        double delta = 0.0, top = -1.0;
        std::size_t arg = 0;
        for (std::size_t c = 0; c < C; ++c) {
            const double d = eta_t[c] - pred[c], psi = d * d;
            delta += p[c] * psi;
            if (psi > top) {                                          // TD-1: the first on ties
                top = psi;
                arg = c;
            }
        }
        r.weights = p;
        r.delta = delta;
        r.psi_max = top;
        r.argmax = arg;
        r.certified = top <= (1.0 + tol) * delta;
    };
    evaluate();
    for (long long s = 0; s < steps && !r.certified; ++s) {
        ++r.trials[r.argmax];                                         // alpha_s = 1/(N + 1)
        r.chosen.push_back(r.argmax);
        ++N;
        ++r.steps;
        evaluate();
    }
    return r;
}

}   // namespace mootation::scalar::discriminate

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The problem's hooks for the strategies over minima (task 24, §24.1.3,
// §24.1.6; evaluator.hpp, Problem): the defaults where the problem gives none.
//
//   distance_of(P, x, y)  the problem's distance, else the Euclidean distance
//                         after the problem's align (where it gives one) or
//                         symmetry::distance on its blocks and canon (SYM-1
//                         ... SYM-11): invariant to the problem's symmetries;
//                         a periodic coordinate the shorter way (SYM-11)
//   restricted(P, xf, idx) the problem's restrict_to, else the default: the
//                         variables idx free, the others frozen at xf — the
//                         subproblem's variable k is x[idx[k]], in idx's
//                         order, its bounds P.lo/hi[idx[k]]; its values,
//                         gradient, residuals and Jacobian the problem's at
//                         the assembled point embed(xf, idx, z) (their
//                         columns idx); idx all the variables in order: the
//                         problem itself (RES-1)
//   embed(xf, idx, z)     the way back: xf with x[idx[k]] = z[k]
//   active_of(P, x, who)  the variables a local search may move at x: the
//                         problem's decode(x).active (empty: all; a genome
//                         that does not decode: none)
//
// READINGS
//   RES-1. Off by default (a strategy that asks for a subproblem). The
//     default subproblem keeps the problem's box (on idx), its
//     constraints, targets (stop_at as it is, f_sub(z) being f(embed(z));
//     accept_success at the assembled point), its periodic and var_repair
//     masks (on idx), and those groups of sym_blocks whose coordinates are
//     all in idx and there in their order, one after another (remapped to
//     the subproblem's positions; a group cut by idx, or laid out otherwise,
//     is dropped); not canon or align (they are the whole problem's), gauge
//     directions, start or hooks of task 24 (perturb, distance, canon_key,
//     local_search, the genome's): a subproblem with those is the user's
//     restrict_to.
// ============================================================================

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "evaluator.hpp"
#include "symmetry.hpp"

namespace mootation::scalar {

inline double distance_of(const Problem& P, const std::vector<double>& x, const std::vector<double>& y)
{
    if (P.distance) return P.distance(x, y);
    if (x.size() != y.size()) throw std::invalid_argument("distance_of: x and y differ in length");
    const std::vector<double> period = symmetry::periods(P.periodic, P.lo, P.hi);
    if (!P.align) return symmetry::distance(x, y, P.sym_blocks, P.canon, period);
    std::vector<double> a;
    P.align(x, y, a);
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = period.empty() ? a[i] - y[i] : symmetry::difference(a[i], y[i], period[i]);
        sum += d * d;
    }
    return std::sqrt(sum);
}

// the variables a local search may move at x (task 24, §24.1.1, §24.1.2;
// ULS-7, SRCH-21): the problem's decode(x).active, 1 where variable j may
// move; empty: all of them (no decode, or decode gives no mask); a genome
// that does not decode: none
inline std::vector<char> active_of(const Problem& P, const std::vector<double>& x, const std::string& who)
{
    if (!P.decode) return {};
    Decoded d;
    try {
        d = P.decode(x);
    } catch (const std::exception& e) {
        throw std::runtime_error(who + ": the user's decode failed: " + e.what());
    }
    if (!d.valid) return std::vector<char>(x.size(), 0);
    if (!d.active.empty() && d.active.size() != x.size())
        throw std::invalid_argument(who + ": decode gave " + std::to_string(d.active.size()) + " active flags for " +
                                    std::to_string(x.size()) + " variables");
    return d.active;
}

// the subproblem's z back in the whole: x_frozen with x[idx[k]] = z[k]
inline std::vector<double> embed(const std::vector<double>& x_frozen, const std::vector<std::size_t>& idx,
                                 const std::vector<double>& z)
{
    if (z.size() != idx.size())
        throw std::invalid_argument("embed: " + std::to_string(z.size()) + " values for " +
                                    std::to_string(idx.size()) + " variables");
    std::vector<double> x = x_frozen;
    for (std::size_t k = 0; k < idx.size(); ++k) x.at(idx[k]) = z[k];
    return x;
}

inline Problem restricted(const Problem& P, const std::vector<double>& x_frozen, const std::vector<std::size_t>& idx)
{
    using Vec = std::vector<double>;
    if (P.restrict_to) return P.restrict_to(x_frozen, idx);
    const std::size_t n = P.lo.size();
    if (x_frozen.size() != n)
        throw std::invalid_argument("restricted: x_frozen has " + std::to_string(x_frozen.size()) + " values for " +
                                    std::to_string(n) + " variables");
    for (std::size_t j : idx)
        if (j >= n) throw std::invalid_argument("restricted: variable " + std::to_string(j) + " past n");
    bool all = idx.size() == n;
    for (std::size_t k = 0; k < idx.size() && all; ++k) all = idx[k] == k;
    if (all) return P;                                                               // RES-1
    auto whole = std::make_shared<const Problem>(P);
    auto full = [x_frozen, idx](const Vec& z) {
        Vec x = x_frozen;
        for (std::size_t k = 0; k < idx.size(); ++k) x[idx[k]] = z[k];
        return x;
    };
    auto columns = [idx](const Rows& J) {
        Rows out(J.size(), Vec(idx.size()));
        for (std::size_t a = 0; a < J.size(); ++a)
            for (std::size_t k = 0; k < idx.size(); ++k) out[a][k] = J[a][idx[k]];
        return out;
    };
    Problem Q;
    for (std::size_t j : idx) {
        Q.lo.push_back(P.lo[j]);
        Q.hi.push_back(P.hi[j]);
        if (!P.periodic.empty()) Q.periodic.push_back(P.periodic[j]);
        if (!P.var_repair.empty()) Q.var_repair.push_back(P.var_repair[j]);
    }
    Q.m = P.m;
    Q.n_cons = P.n_cons;
    Q.n_eq = P.n_eq;
    Q.n_res = P.n_res;
    Q.blocks = P.blocks;
    Q.weights = P.weights;
    Q.stop_at = P.stop_at;
    for (const symmetry::Blocks& b : P.sym_blocks) {                                    // RES-1
        std::size_t first = idx.size();
        bool kept = true;
        for (std::size_t t = 0; t < b.count * b.size && kept; ++t) {
            std::size_t at = idx.size();
            for (std::size_t q = 0; q < idx.size(); ++q)
                if (idx[q] == b.offset + t) at = q;
            if (t == 0) first = at;
            kept = at < idx.size() && at == first + t;
        }
        if (kept) Q.sym_blocks.push_back({first, b.size, b.count});
    }
    Q.post_at = P.post_at;
    Q.success_cost = P.success_cost;
    if (P.evaluate) Q.evaluate = [whole, full](const Vec& z, Vec& f, Vec& g) { whole->evaluate(full(z), f, g); };
    if (P.gradient)
        Q.gradient = [whole, full, idx, columns](const Vec& z, Vec& gf, Rows& JG) {
            Vec g;
            Rows J;
            whole->gradient(full(z), g, J);
            gf.assign(idx.size(), 0.0);
            for (std::size_t k = 0; k < idx.size(); ++k) gf[k] = g[idx[k]];
            JG = columns(J);
        };
    if (P.value_grad)
        Q.value_grad = [whole, full, idx, columns](const Vec& z, Vec& f, Vec& g, Vec& gf, Rows& JG) {
            Vec gg;
            Rows J;
            whole->value_grad(full(z), f, g, gg, J);
            gf.assign(idx.size(), 0.0);
            for (std::size_t k = 0; k < idx.size(); ++k) gf[k] = gg[idx[k]];
            JG = columns(J);
        };
    if (P.residuals) Q.residuals = [whole, full](const Vec& z, Vec& r) { whole->residuals(full(z), r); };
    if (P.jacobian)
        Q.jacobian = [whole, full, columns](const Vec& z, Rows& J) {
            Rows Jf;
            whole->jacobian(full(z), Jf);
            J = columns(Jf);
        };
    if (P.residuals_jac)
        Q.residuals_jac = [whole, full, columns](const Vec& z, Vec& r, Rows& J) {
            Rows Jf;
            whole->residuals_jac(full(z), r, Jf);
            J = columns(Jf);
        };
    if (P.accept_success)
        Q.accept_success = [whole, full](const Vec& z) { return whole->accept_success(full(z)); };
    return Q;
}

}   // namespace mootation::scalar

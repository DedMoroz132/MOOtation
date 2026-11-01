#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Wrappers over any problem (task 20 of 2026-11-02, §20.2.1): a permutation,
// a rotation and a shift of the whole decision vector, chiefly for the
// single-objective solvers. The run sees x; the problem underneath is
// evaluated at y = T(x). Off unless asked for: a problem without a wrapper is
// evaluated as it was, and every instance number 0 makes T the identity, bit
// for bit.
//
// THE MAP, from the run's x to the problem's y, in this fixed order:
//
//   shift    u = x - s                      the whole problem moves with its
//                                           box: x in [lb + s, ub + s]
//   rotate   v = c + Q (u - c)              Q a k x k rotation, c its centre
//   perm     y_i = v_pi(i)                  the box permuted the same way
//
// so that the problem is first permuted, then turned, then moved, and the
// chain from x is the one of the sources: Tanabe & Fukunaga 2014 evaluate a
// "distributed" function at z = (y_P1, ..., y_PD) of y = x - o (§2,
// "Adjacent functions", the permutation of their reference [20]); bbob turns
// x - x_opt (Hansen et al. 2009, z = R (x - x_opt)). Its inverse:
//
//   v_pi(i) = y_i,   u = c + Q^T (v - c),   x = u + s,
//
// gives the start x0 and the optimum x_opt in the run's coordinates; f_opt
// is the problem's. The derivatives follow the chain rule with dy/dx = P Q
// ((P v)_i = v_pi(i)): a Jacobian J_y of residuals becomes J_y P Q, row by
// row Q^T (P^T row), and a gradient g_y becomes Q^T (P^T g_y) (pull).
//
// THE INSTANCES. Each wrapper has its own instance number (0: off) and draws
// from its own key, the maps' way (suite.hpp, THE MAPS; instances.hpp):
// (20261102, kind, crc32(name), n, instance), kind 1 perm, 2 rotate, 3 shift,
// name the problem's name in the registry (or a user's problem's own name):
//   perm     pi = rnd::shuffle of 0 .. n - 1 (Fisher-Yates from the last
//            element down)
//   rotate   Q = instances::haar(g, n), a rotation from the Haar distribution,
//            uniform over the rotations as Iorio & Li 2006 ask of theirs
//            (§3.1, Algorithm 1: Gram-Schmidt of normal vectors; haar's QR
//            with the signs of R's diagonal fixed is the same distribution)
//   shift    s_j = scale_j (2 U_j - 1), U_j = rnd::uniform01, j = 0 .. n - 1
//            in the run's order of the variables
// benchmarks/wrap.py draws the same with library_rng.py: pi and s bit for
// bit, Q to a few ulps (NumPy's QR against haar's; as for the maps).
//
// What the products add up: Q d and Q^T d are rows' pairwise sums
// (common.hpp, matvec: np.sum(A * d, axis=1)), as the maps' are.
//
// READINGS:
//   WRAP-1. The order shift, rotate, perm from x (above). The three commute
//     up to their own parameters (a shift after a permutation is another
//     shift), so the order names an instance, not a different construction.
//   WRAP-2. The centre of the rotation is the optimum, c = P^T x_opt
//     (c_pi(i) = x_opt_i, the optimum in v's coordinates), where x_opt is
//     known (bbob: the instance's), so that the optimum stays where it was
//     relative to the box; else the origin, Iorio & Li's centre ("the
//     rotation matrix is used to rotate about the origin", §3.1). The box has
//     no centre to offer: a problem with a box is not rotated (WRAP-3).
//   WRAP-3. A rotation only for a problem without a box (every bound
//     infinite: MGH, NIST) or whose formula is defined on all of R^n and is
//     marked so (defined_everywhere; bbob, whose functions are defined
//     outside [-5, 5]^D and penalise it themselves where they do, f_pen):
//     its box stays where it was in the run's coordinates (shifted by s), the
//     points of it go where Q sends them. A problem with a box otherwise is
//     refused: Q carries points out of the box, and every rule that brings
//     them back changes the problem (rWFG's clipping makes plateaus;
//     rotated.py). bbob turns its variables itself (R, Q): this rotation
//     comes on top of the instance's, which is allowed and is still a
//     rotation of the same function. Multi-objective problems are refused:
//     their rotations are the r* maps, which keep the front.
//   WRAP-4. The scale of the shift, per variable of the run: the width of its
//     box, ub - lb, where both bounds are finite; else max(1, |a|_inf), a =
//     x_opt where known, else x0, else 1 alone. So s moves a box by up to its
//     width, and a box-less problem by about the size of its optimum or its
//     start — the size at which steps proportional to |x| (finite
//     differences, Nelder-Mead's initial simplex, lm's scaling, relative
//     tolerances) change. shift_scale multiplies it (1 by default).
//   WRAP-5. This is a shift of the whole problem, box and all: relative to
//     its box the optimum does not move. Moving the optimum inside the box is
//     the maps' (shiftDTLZ, shiftZDT; suite.hpp).
//   WRAP-6. perm and shift know the variables only as a box: a problem whose
//     variables carry more structure (types other than real, blocks or
//     symmetries) is the caller's to refuse, or to permute along; a shift or
//     a rotation of integer variables would make them real, and is refused by
//     the caller.
//
// Sources: R. Tanabe, A. Fukunaga, "Reevaluating exponential crossover in
// differential evolution", PPSN XIII, 2014 (source
// de-exponential-crossover_tanabe2014); A.W. Iorio, X. Li, "Rotated test
// problems for assessing the performance of multi-objective optimization
// algorithms", GECCO 2006, 683-690 (source rotated-problems_iorio2006);
// N. Hansen, S. Finck, R. Ros, A. Auger, "Real-parameter black-box
// optimization benchmarking 2009: noiseless functions definitions", INRIA
// RR-6829 (source bbob-functions-def_hansen2009), §0.2 and the definitions.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../random.hpp"
#include "common.hpp"
#include "instances.hpp"
#include "native.hpp"
#include "suite.hpp"

namespace mootation::suite {

// Which wrappers, and their instances (0: off).
struct WrapSpec {
    std::string name;                       // the problem's name: the key's crc32
    int perm = 0, rotate = 0, shift = 0;    // instance numbers
    double shift_scale = 1.0;               // WRAP-4
    bool defined_everywhere = false;        // WRAP-3: the formula holds on all of R^n
};

constexpr std::uint64_t kWrapKey = 20261102;

namespace wrap_detail {

// Q^T d of a k x k row-major Q, each entry the pairwise sum of its column's
// products (matvec of Q's transpose)
inline void matvec_t(const std::vector<double>& Q, std::size_t k, const double* d, double* out)
{
    std::vector<double> p(k);
    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t i = 0; i < k; ++i) p[i] = Q[i * k + j] * d[i];
        out[j] = row_sum(p);
    }
}

inline std::mt19937 generator(int kind, const std::string& name, int n, int instance)
{
    return std::mt19937(instances::seed({kWrapKey, static_cast<std::uint64_t>(kind), instances::crc32(name),
                                         static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(instance)}));
}

}   // namespace wrap_detail

// The map x -> y of a WrapSpec over a problem of n variables (see THE MAP).
class WrapMap {
public:
    // base: the problem's box; x_opt and x0 its optimum and start, where
    // known (empty otherwise)
    WrapMap(const WrapSpec& w, int n, int n_obj, const Bounds& base, const std::vector<double>& x_opt = {},
            const std::vector<double>& x0 = {})
        : w_(w), n_(n)
    {
        const std::size_t nu = static_cast<std::size_t>(n);
        if (n < 1) fail("a problem of no variables");
        if (base.size() != nu) fail("the box has " + std::to_string(base.size()) + " variables, not n");
        if (!x_opt.empty() && x_opt.size() != nu) fail("x_opt of n values");
        if (!x0.empty() && x0.size() != nu) fail("x0 of n values");
        if (w.perm < 0 || w.rotate < 0 || w.shift < 0) fail("an instance number below 0");
        if (!(w.shift_scale > 0.0 && std::isfinite(w.shift_scale))) fail("shift_scale: a finite number above 0");
        if (active() && w.name.empty()) fail("the key needs the problem's name");

        perm_.resize(nu);
        for (std::size_t i = 0; i < nu; ++i) perm_[i] = static_cast<int>(i);
        if (w.perm > 0) {
            std::mt19937 g = wrap_detail::generator(1, w.name, n, w.perm);
            rnd::shuffle(perm_.begin(), perm_.end(), g);
        }
        // the box of v (and of u): v_pi(i) lies in the problem's box of y_i
        Bounds vbox(nu);
        for (std::size_t i = 0; i < nu; ++i) vbox[static_cast<std::size_t>(perm_[i])] = base[i];

        if (w.rotate > 0) {
            if (n_obj > 1) fail("rotate: a multi-objective problem (its rotations are the r* maps, WRAP-3)");
            bool boxed = false;
            for (const auto& b : base) boxed = boxed || std::isfinite(b.first) || std::isfinite(b.second);
            if (boxed && !w.defined_everywhere)
                fail("rotate: a problem with a box (a rotation carries points out of it, and bringing them "
                     "back changes the problem; WRAP-3)");
            std::mt19937 g = wrap_detail::generator(2, w.name, n, w.rotate);
            Q_ = instances::haar(g, n);
            centre_.assign(nu, 0.0);                       // WRAP-2
            if (!x_opt.empty())
                for (std::size_t i = 0; i < nu; ++i) centre_[static_cast<std::size_t>(perm_[i])] = x_opt[i];
        }

        shift_.assign(nu, 0.0);
        if (w.shift > 0) {
            double size = 1.0;                             // WRAP-4, a box-less variable's scale
            const std::vector<double>& a = !x_opt.empty() ? x_opt : x0;
            for (double v : a) size = std::max(size, std::abs(v));
            std::mt19937 g = wrap_detail::generator(3, w.name, n, w.shift);
            for (std::size_t j = 0; j < nu; ++j) {
                const auto& b = vbox[j];
                const double scale = std::isfinite(b.first) && std::isfinite(b.second) ? b.second - b.first : size;
                shift_[j] = w.shift_scale * scale * (2.0 * rnd::uniform01(g) - 1.0);
            }
        }
        box_.resize(nu);
        for (std::size_t j = 0; j < nu; ++j) box_[j] = {vbox[j].first + shift_[j], vbox[j].second + shift_[j]};
    }

    bool active() const { return w_.perm > 0 || w_.rotate > 0 || w_.shift > 0; }
    int n_var() const { return n_; }
    const WrapSpec& spec() const { return w_; }

    // y = T(x): the point the problem is evaluated at
    void to_problem(const double* x, double* y) const
    {
        const std::size_t n = static_cast<std::size_t>(n_);
        std::vector<double> v(x, x + n);
        if (w_.shift > 0)
            for (std::size_t j = 0; j < n; ++j) v[j] = v[j] - shift_[j];
        if (w_.rotate > 0) {
            std::vector<double> d(n), r(n);
            for (std::size_t j = 0; j < n; ++j) d[j] = v[j] - centre_[j];
            matvec(Q_, n, d.data(), r.data());
            for (std::size_t j = 0; j < n; ++j) v[j] = centre_[j] + r[j];
        }
        for (std::size_t i = 0; i < n; ++i) y[i] = v[static_cast<std::size_t>(perm_[i])];
    }

    // x = T^-1(y): a point of the problem (its start, its optimum) in the
    // run's coordinates
    void to_run(const double* y, double* x) const
    {
        const std::size_t n = static_cast<std::size_t>(n_);
        std::vector<double> u(n);
        for (std::size_t i = 0; i < n; ++i) u[static_cast<std::size_t>(perm_[i])] = y[i];
        if (w_.rotate > 0) {
            std::vector<double> d(n), r(n);
            for (std::size_t j = 0; j < n; ++j) d[j] = u[j] - centre_[j];
            wrap_detail::matvec_t(Q_, n, d.data(), r.data());
            for (std::size_t j = 0; j < n; ++j) u[j] = centre_[j] + r[j];
        }
        for (std::size_t j = 0; j < n; ++j) x[j] = w_.shift > 0 ? u[j] + shift_[j] : u[j];
    }

    std::vector<double> to_run(const std::vector<double>& y) const
    {
        std::vector<double> x(y.size());
        to_run(y.data(), x.data());
        return x;
    }

    // g_x = (dy/dx)^T g_y = Q^T (P^T g_y): a gradient, or a row of a Jacobian,
    // in the run's coordinates
    void pull(const double* g_y, double* g_x) const
    {
        const std::size_t n = static_cast<std::size_t>(n_);
        std::vector<double> w(n);
        for (std::size_t i = 0; i < n; ++i) w[static_cast<std::size_t>(perm_[i])] = g_y[i];
        if (w_.rotate > 0) wrap_detail::matvec_t(Q_, n, w.data(), g_x);
        else std::copy(w.begin(), w.end(), g_x);
    }

    // the box in the run's coordinates
    const Bounds& bounds() const { return box_; }

    const std::vector<int>& permutation() const { return perm_; }     // pi
    const std::vector<double>& rotation() const { return Q_; }         // Q, n x n row-major (empty: none)
    const std::vector<double>& centre() const { return centre_; }      // c (empty: none)
    const std::vector<double>& shift() const { return shift_; }        // s

    // the instance, for a run's record: which wrappers, their numbers, the key
    std::string describe() const
    {
        std::string s = "shift " + std::to_string(w_.shift) + ", rotate " + std::to_string(w_.rotate) + ", perm " +
                        std::to_string(w_.perm) + " of '" + w_.name + "' (key " + std::to_string(kWrapKey) +
                        ", kind, crc32 " + std::to_string(instances::crc32(w_.name)) + ", n " +
                        std::to_string(n_) + ", instance)";
        if (w_.shift > 0 && w_.shift_scale != 1.0) s += ", shift_scale " + std::to_string(w_.shift_scale);
        return s;
    }

private:
    WrapSpec w_;
    int n_ = 0;
    std::vector<int> perm_;
    std::vector<double> Q_, centre_, shift_;
    Bounds box_;

    [[noreturn]] void fail(const std::string& why) const
    {
        throw std::invalid_argument("wrap: " + (w_.name.empty() ? std::string("a problem") : w_.name) + ": " +
                                    why);
    }
};

// A problem seen through a WrapMap: its values at x are the problem's at
// T(x), its residuals' Jacobian J_y P Q. const and without state, as Native.
class Wrapped final : public Native {
public:
    Wrapped(std::shared_ptr<const Native> problem, WrapMap map) : p_(std::move(problem)), m_(std::move(map))
    {
        if (!p_) throw std::invalid_argument("wrap: no problem");
        if (p_->n_var != m_.n_var()) throw std::invalid_argument("wrap: the map is not of the problem's n");
        n_var = p_->n_var;
        n_obj = p_->n_obj;
        n_con = p_->n_con;
        n_res = p_->n_res;
        has_jacobian = p_->has_jacobian;
    }

    void evaluate(const double* x, double* f, double* g) const override
    {
        std::vector<double> y(static_cast<std::size_t>(n_var));
        m_.to_problem(x, y.data());
        p_->evaluate(y.data(), f, g);
    }

    void residuals(const double* x, double* r, double* J) const override
    {
        const std::size_t n = static_cast<std::size_t>(n_var), m = static_cast<std::size_t>(n_res);
        std::vector<double> y(n), Jy(J ? m * n : 0);
        m_.to_problem(x, y.data());
        p_->residuals(y.data(), r, J ? Jy.data() : nullptr);
        if (J)
            for (std::size_t row = 0; row < m; ++row) m_.pull(Jy.data() + row * n, J + row * n);
    }

    const WrapMap& map() const { return m_; }
    const Native& problem() const { return *p_; }

private:
    std::shared_ptr<const Native> p_;
    WrapMap m_;
};

// A problem of the suite (suite.hpp, make) wrapped: its box from the suite,
// bbob's x_opt from its instance where none is given, bbob marked defined
// everywhere (WRAP-3). x0: the problem's start, where it has one (WRAP-4).
inline std::unique_ptr<Wrapped> wrap(Spec spec, WrapSpec w, std::vector<double> x0 = {},
                                     std::vector<double> x_opt = {})
{
    auto problem = std::make_shared<const Problem>(std::move(spec));
    const Spec& s = problem->spec();
    if (s.family == "bbob") {
        w.defined_everywhere = true;
        if (x_opt.empty())
            x_opt = s.bbob.empty() ? bbob::instance(s.bbob_fid, s.bbob_iid, s.n_var).x_opt : s.bbob[0].x_opt;
    }
    WrapMap map(w, problem->n_var, problem->n_obj, problem->bounds(), x_opt, x0);
    return std::make_unique<Wrapped>(std::move(problem), std::move(map));
}

}   // namespace mootation::suite

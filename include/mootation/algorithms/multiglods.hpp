#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../dss.hpp"
#include "../sobol.hpp"

namespace mootation {

// ============================================================================
// MultiGLODS — global and local multiobjective optimization by direct search
// A.L. Custódio, J.F.A. Madeira, "MultiGLODS: global and local multiobjective
// optimization using direct search", J. Glob. Optim. 72(2):323-345, 2018,
// doi:10.1007/s10898-018-0618-1 (the manuscript: multiglods_custodio2018_draft;
// §2 Algorithms 2.1-2.2, §4).
//
// A list of points (x; α_x; r_x; i_x): step size, comparison radius, active
// flag; points are never removed. One step() is one iteration of Algorithm
// 2.1:
//   search — after three unsuccessful iterations in a row (§4), n points of
//     the Sobol sequence (sobol.hpp), added by add();
//   poll — when the search added no active point: the active point not yet
//     local (α >= 10⁻³, §4) of the largest step size; the coordinate
//     directions [I −I], every one (complete poll, §4); the poll points
//     carry (α_k, α_k‖d‖) into add();
//   the outcome — success (a new active point): the step sizes kept (§4,
//     γ = 1), each new active point's radius at least d_max α; unsuccessful
//     (the list unchanged): the poll centre's α halved; merging (only
//     inactive points added): nothing changes.
// add() (Algorithm 2.2), each new point x against the list:
//   far from every y (‖x − y‖ > r_y): added active, (α_0, r_0)        (line 2)
//   else, over the comparable y (‖x − y‖ <= r_y): x dominating y turns y
//     inactive, and x takes the largest α_y of them and its r_y; a y
//     dominating x marks pdom; x is active when pdom = 0; x is added when it
//     dominates an active point, or when nothing comparable dominates it
//     (integer lattices, ρ ≡ 0: line 6) — with the inherited (α, r) from a
//     search, its own from a poll                                  (lines 7-8)
// Settings (§4): the start — n points evenly spaced on the line between the
// bounds and the centre of the box; α_0 = n·max_i (u_i − l_i); r_0 = d_max α_0
// = α_0 (d_max = 1 for [I −I]); halving; the run ends when every active
// point's α is below 10⁻³, or at the budget.
// The answer: the non-dominated points of the list (the global front's
// approximation: the active points' non-dominated set, §2), reduced to
// pop_size by DSS.
//
// THE KNOBS: glods_search_every (3), glods_local_tol (10⁻³).
//
// READINGS (declared):
//   MG-1. The box normalized to [0, 1]^n, as dms does: α_0 = n, distances
//     and radii in the normalized box (the paper works in the problem's
//     units; with its α_0 = n·max range the two agree on boxes of one
//     range).
//   MG-2. The Sobol sequence unscrambled, from its index 1, continued from
//     one search to the next; a point outside the domain (constraints) is
//     evaluated and refused by the extreme barrier. Above 100 variables the
//     search draws uniform points from the seed instead.
//   MG-3. Constraints: the extreme barrier — an infeasible point (cv > 0)
//     never enters the list (constraint_mode FEASIBILITY); NONE ignores
//     them. A start without a feasible point keeps the least violating.
//   MG-4. The order of the list: insertion; the poll centre's ties (equal
//     α): the first.
//   MG-5. Poll points outside the box are not evaluated (the extreme
//     barrier, as dms).
//   MG-6. "Each time that three consecutive unsuccessful iterations
//     occurred" (§4): a success or a merging iteration breaks the run, and a
//     search starts the count again.
// Continuous variables only.
// ============================================================================

template <typename Ind_t>
class MultiGLODSCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    struct Entry {
        std::vector<double> x;     // normalized
        std::vector<double> f;
        double cv = 0.0;
        double alpha = 0.0, r = 0.0;
        bool active = false;
        VaultPoint pt{};
    };

    std::vector<Entry> list_;
    std::vector<double> lo_, hi_;
    double alpha0_ = 1.0, r0_ = 1.0, local_tol_ = 1e-3;
    int search_every_ = 3, failures_ = 0;
    std::unique_ptr<sobol::Generator> sobol_;
    std::mt19937 rng_{0};

    static bool dominates(const std::vector<double>& a, const std::vector<double>& b)
    {
        bool strict = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) strict = true;
        }
        return strict;
    }

    bool feasible(const Entry& e) const
    {
        return constraint_mode == ConstraintMode::NONE || e.cv <= 0.0;
    }

    static double dist(const std::vector<double>& a, const std::vector<double>& b)
    {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) s += (a[j] - b[j]) * (a[j] - b[j]);
        return std::sqrt(s);
    }

    std::vector<double> denorm(const std::vector<double>& u) const
    {
        std::vector<double> x(u.size());
        for (std::size_t j = 0; j < u.size(); ++j) x[j] = lo_[j] + u[j] * (hi_[j] - lo_[j]);
        return x;
    }

    std::vector<Entry> evaluate(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U)
    {
        std::vector<Entry> out;
        if (U.empty()) return out;
        const int keep = static_cast<int>(vault.active_n());
        const int base = vault.expand(static_cast<int>(U.size()));
        for (std::size_t i = 0; i < U.size(); ++i)
            vault.set_variables(static_cast<std::size_t>(base) + i, denorm(U[i]));
        vault.sync();
        for (std::size_t i = 0; i < U.size(); ++i) {
            const std::size_t v = static_cast<std::size_t>(base) + i;
            Entry e;
            e.x = U[i];
            e.f = vault.objectives_of(v);
            e.cv = vault.get_cv(v);
            e.pt = vault.point_of(v);
            out.push_back(std::move(e));
        }
        vault.reduce(keep);
        return out;
    }

    // Algorithm 2.2; the indices of the points added active.
    std::vector<std::size_t> add(std::vector<Entry> L2)
    {
        std::vector<std::size_t> added_active;
        bool any_added = false;
        for (auto& x : L2) {
            if (!feasible(x)) continue;                                        // MG-3
            double far = std::numeric_limits<double>::infinity();
            for (const auto& y : list_) far = std::min(far, dist(x.x, y.x) - y.r);
            if (list_.empty() || far > 0.0) {                                  // lines 1-2
                x.alpha = alpha0_; x.r = r0_; x.active = true;
                added_active.push_back(list_.size());
                list_.push_back(std::move(x));
                any_added = true;
                continue;
            }
            bool present = false;
            for (const auto& y : list_) present = present || y.x == x.x;
            if (present) continue;
            double a = 0.0, ra = 0.0;
            int idom = 0;
            bool pdom = false;
            for (auto& y : list_) {
                if (dist(x.x, y.x) - y.r > 0.0) continue;                      // line 3
                if (dominates(x.f, y.f)) {
                    idom += y.active ? 1 : 0;
                    y.active = false;                                          // line 4
                    if (y.alpha > a) { a = y.alpha; ra = y.r; }
                } else if (dominates(y.f, x.f)) {
                    pdom = true;
                }
            }
            const bool active = !pdom;                                         // line 5
            if (a == 0.0) { a = alpha0_; ra = r0_; }
            if (idom > 0 || !pdom) {                                           // line 6, ρ ≡ 0
                if (x.alpha == 0.0) { x.alpha = a; x.r = ra; }                 // line 7
                x.active = active;                                             // line 8: its own
                if (active) added_active.push_back(list_.size());
                list_.push_back(std::move(x));
                any_added = true;
            }
        }
        added_ = any_added;
        return added_active;
    }
    bool added_ = false;

    void publish(DataVault<Ind_t>& vault)
    {
        std::vector<std::size_t> nd;
        for (std::size_t i = 0; i < list_.size(); ++i) {
            bool beaten = false;
            for (std::size_t j = 0; j < list_.size() && !beaten; ++j)
                beaten = j != i && (dominates(list_[j].f, list_[i].f) ||
                                    (j < i && list_[j].f == list_[i].f));
            if (!beaten) nd.push_back(i);
        }
        std::vector<std::vector<double>> F;
        for (std::size_t i : nd) F.push_back(list_[i].f);
        const auto idx = dss::order(F, static_cast<std::size_t>(vault.pop_size()));
        vault.reduce(0);
        vault.expand(static_cast<int>(idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) vault.put_point(i, list_[nd[idx[i]]].pt);
    }

    void start(DataVault<Ind_t>& vault, std::vector<Entry> first)
    {
        list_.clear();
        failures_ = 0;
        bool feasible_one = false;
        for (const auto& e : first) feasible_one = feasible_one || feasible(e);
        if (!feasible_one && !first.empty()) {                                  // MG-3
            auto best = std::min_element(first.begin(), first.end(),
                [](const Entry& a, const Entry& b) { return a.cv < b.cv; });
            best->cv = 0.0;
            Entry e = *best;
            first = {e};
        }
        for (auto& e : first) e.alpha = 0.0;                                    // search points
        add(std::move(first));
        publish(vault);
    }

    void read_bounds(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("multiglods: continuous variables only");
        const auto& b = vault.get_bounds();
        lo_.resize(b.size());
        hi_.resize(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (!b[j].first || !b[j].second)
                throw std::invalid_argument("multiglods: every variable needs finite bounds");
            lo_[j] = *b[j].first;
            hi_[j] = *b[j].second;
        }
        const int n = static_cast<int>(b.size());
        alpha0_ = static_cast<double>(n);                                       // MG-1
        r0_ = alpha0_;
        if (n <= sobol::max_dims) sobol_ = std::make_unique<sobol::Generator>(n);
    }

public:
    // No random number at the defaults (the Sobol sequence is deterministic):
    // the seed is accepted, as every core's is, and changes nothing below
    // 101 variables (MG-2).
    static constexpr bool deterministic = true;

    void set_seed(unsigned s) { rng_.seed(s); }

    bool set_param(const std::string& k, double v)
    {
        if (k == "glods_search_every") {
            if (!(v >= 1.0) || v != std::floor(v) || v > 1e9)
                throw std::invalid_argument("glods_search_every must be a whole number >= 1");
            search_every_ = static_cast<int>(v);
            return true;
        }
        if (k == "glods_local_tol") {
            if (!(v > 0.0)) throw std::invalid_argument("glods_local_tol must be > 0");
            local_tol_ = v;
            return true;
        }
        return false;
    }

    std::size_t list_size() const { return list_.size(); }

    // Every active point local (α below 10⁻³): the paper's stopping rule.
    bool finished() const
    {
        bool any = false;
        for (const auto& e : list_)
            if (e.active) { any = true; if (e.alpha >= local_tol_) return false; }
        return any;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        const std::size_t n = lo_.size();
        std::vector<std::vector<double>> S;
        if (n >= 2)
            for (std::size_t i = 0; i < n; ++i)
                S.emplace_back(n, static_cast<double>(i) / static_cast<double>(n - 1));
        S.emplace_back(n, 0.5);                                                 // the centre
        vault.reduce(0);
        start(vault, evaluate(vault, S));
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        std::vector<Entry> first;
        for (std::size_t v = 0; v < vault.active_n(); ++v) {
            Entry e;
            const auto& x = vault.variables_of(v);
            e.x.resize(x.size());
            for (std::size_t j = 0; j < x.size(); ++j) e.x[j] = (x[j] - lo_[j]) / (hi_[j] - lo_[j]);
            e.f = vault.objectives_of(v);
            e.cv = vault.get_cv(v);
            e.pt = vault.point_of(v);
            first.push_back(std::move(e));
        }
        start(vault, std::move(first));
    }

    void step(DataVault<Ind_t>& vault)
    {
        for (auto& e : list_) refresh_copy(vault, e.pt, e.f, e.cv, true);
        if (finished()) return;
        const std::size_t n = lo_.size();
        // the search, after search_every_ failures in a row
        if (failures_ >= search_every_) {
            failures_ = 0;                                                       // MG-6
            std::vector<std::vector<double>> A;
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (std::size_t i = 0; i < n; ++i) {
                if (sobol_) A.push_back(sobol_->next());
                else {                                                           // MG-2
                    std::vector<double> u(n);
                    for (auto& t : u) t = U(rng_);
                    A.push_back(std::move(u));
                }
            }
            auto got = evaluate(vault, A);
            for (auto& e : got) e.alpha = 0.0;
            const auto act = add(std::move(got));
            if (!act.empty()) { publish(vault); return; }
        }
        // the poll centre: active, not local, the largest α (MG-4)
        std::size_t c = list_.size();
        for (std::size_t i = 0; i < list_.size(); ++i)
            if (list_[i].active && list_[i].alpha >= local_tol_ &&
                (c == list_.size() || list_[i].alpha > list_[c].alpha)) c = i;
        if (c == list_.size()) { publish(vault); return; }
        const Entry centre = list_[c];
        std::vector<std::vector<double>> P;
        for (int sign : {+1, -1})
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<double> u = centre.x;
                u[j] += sign * centre.alpha;
                if (u[j] < 0.0 || u[j] > 1.0) continue;                         // MG-5
                P.push_back(std::move(u));
            }
        auto got = evaluate(vault, P);
        for (auto& e : got) { e.alpha = centre.alpha; e.r = centre.alpha; }    // ‖d‖ = 1
        const auto act = add(std::move(got));
        if (!act.empty()) {                                                      // success
            failures_ = 0;
            for (std::size_t i : act) list_[i].r = std::max(list_[i].r, list_[i].alpha);
        } else if (!added_) {                                                    // unsuccessful
            ++failures_;
            list_[c].alpha *= 0.5;
        } else {
            failures_ = 0;                                                       // merging: MG-6
        }
        publish(vault);
    }
};

}   // namespace mootation

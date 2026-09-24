#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/bound_repair.hpp"
#include "../operators/de_mutation.hpp"

namespace mootation {

// ============================================================================
// GDE3 — The third Evolution Step of Generalized Differential Evolution
// S. Kukkonen, J. Lampinen — IEEE CEC 2005, pp. 443-450,
// doi:10.1109/CEC.2005.1554717
//                     (source: gde3_kukkonen2005, KanGAL Report 2005013 — the
//                      report version of the same paper)
//
// One generation (Eq. 3):
//   1. for every x_i: r1, r2, r3 mutually different and different from i,
//      u_i = DE/rand/1/bin: u_j = x_{j,r3} + F·(x_{j,r1} − x_{j,r2}) if
//      rand_j < CR or j = j_rand, else x_{j,i} — every trial built from
//      generation G;
//   2. select: u_i replaces x_i if u_i ⪯_c x_i (weak constraint-domination);
//      otherwise, if u_i is feasible and x_i ⊀ u_i, u_i is ADDED to the
//      population ("both vectors are selected"); otherwise it is dropped;
//   3. the population, between NP and 2·NP, is reduced to NP: while it is too
//      large, remove the vector x that dominates no one and whose crowding
//      distance is the smallest among the vectors not dominating it — "sorted
//      based on non-dominance and crowdedness ... the worst population members
//      ... are removed", one at a time.
// Constraint-domination ≺_c (§2): a feasible vector beats an infeasible one;
// two infeasible ones compare by dominance in the space of constraint
// violations (each constraint's max(0, g_j), not their sum); two feasible
// ones by dominance in the objectives. ⪯_c is the same with weak dominance.
// With a single objective and no constraints this is DE/rand/1/bin (Eq. 4).
//
// Parameters. The paper sets F and CR per problem and gives no default (§5:
// CR = 0.2, F = 0.2 on the bi-objective test problem (5) and on DTLZ1/DTLZ4;
// 0.9/0.5 on the spring design; 0.9/0.1 on CTP1/CTP2). Default here: CR = 0.2,
// F = 0.2, the paper's setting on its test-function problems (§5.2, §5.5);
// both are knobs (F, CR). At CR = 1 the trial is x_r3 + F·(x_r1 − x_r2)
// entirely, which is invariant to a rotation of the search space. NP ≥ 4
// (Eq. 3's input), refused otherwise.
//
// Declared readings and deviations:
//   GDE3-1. The removal rule (step 3). Eq. 3's condition, read literally,
//     compares crowding distances across fronts; the text states the intent,
//     NSGA-II's ordering. Implemented: non-dominated sorting by ≺_c; whole
//     fronts are kept while they fit; from the front that overflows the member
//     with the smallest crowding distance is removed, the distances of that
//     front recomputed, and so on — the one-at-a-time pruning the paper cites
//     as its improvement ([19], Kukkonen & Deb, then unpublished). A tie in
//     the smallest distance removes the member first in population order.
//   GDE3-2. Crowding distance is NSGA-II's (§III-B of Deb et al. 2002, Eq. 3's
//     "CD"), each objective normalised by its range on the front, at every M.
//     The better distance for M > 2 that Kukkonen & Deb published in 2006 is
//     not part of this paper.
//   GDE3-3. Bounds. "initial bounds" x^(lo), x^(hi) bound the initialisation
//     only; the paper does not say what happens to a trial variable outside
//     them. Here a mutant variable outside the box is repaired by
//     bound_repair (operators/bound_repair.hpp), `reflect` by default — the
//     provisional default for new operators until experiment E3 picks one.
//     `resample` and `native` are refused (DE cannot do them).
//   GDE3-4. Every trial is evaluated in full. The paper notes that
//     constraint-first evaluation can skip objective evaluations (§4); the
//     library has no partial evaluation.
//   GDE3-5. constraint_mode NONE (the default) ignores constraints, as every
//     core does; FEASIBILITY is the paper's ≺_c above; CDP and EPS_CONSTRAINT
//     act as FEASIBILITY.
//   GDE3-6. Continuous variables only; a problem with binary ones is refused.
// ============================================================================
template <typename Ind_t>
class GDE3Core {
public:
    // Supported: NONE, FEASIBILITY (the paper's ≺_c); CDP and EPS_CONSTRAINT
    // act as FEASIBILITY
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double F_  = 0.2;   // §5.2, §5.5
    double CR_ = 0.2;   // §5.2, §5.5
    ops::BoundRepair repair_ = ops::BoundRepair::Reflect;   // GDE3-3
    std::mt19937 rng_{std::random_device{}()};

    bool constrained() const { return constraint_mode != ConstraintMode::NONE; }

    // a ⪯ b (strict = false) or a ≺ b (strict = true), minimisation
    static bool dominates(const std::vector<double>& a, const std::vector<double>& b,
                          bool strict)
    {
        bool better = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) better = true;
        }
        return better || !strict;
    }

    static std::vector<double> violations(DataVault<Ind_t>& vault, int v)
    {
        std::vector<double> g = vault.limits_of(static_cast<std::size_t>(v));
        for (double& x : g) x = std::max(0.0, x);
        return g;
    }

    // a ⪯_c b (strict = false) or a ≺_c b (strict = true), §2
    bool cdom(DataVault<Ind_t>& vault, int a, int b, bool strict)
    {
        if (constrained()) {
            const bool fa = vault.get_cv(static_cast<std::size_t>(a)) <= 0.0;
            const bool fb = vault.get_cv(static_cast<std::size_t>(b)) <= 0.0;
            if (fa && !fb) return true;
            if (!fa && fb) return false;
            if (!fa && !fb) return dominates(violations(vault, a), violations(vault, b), strict);
        }
        const std::vector<double> fa = vault.objectives_of(static_cast<std::size_t>(a));
        return dominates(fa, vault.objectives_of(static_cast<std::size_t>(b)), strict);
    }

    // Non-dominated sorting of [0, n) by ≺_c.
    std::vector<std::vector<int>> sort_fronts(DataVault<Ind_t>& vault, int n)
    {
        std::vector<std::vector<int>> S(static_cast<std::size_t>(n));
        std::vector<int> count(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                if (cdom(vault, i, j, true))      { S[i].push_back(j); ++count[j]; }
                else if (cdom(vault, j, i, true)) { S[j].push_back(i); ++count[i]; }
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> cur;
        for (int i = 0; i < n; ++i) if (count[i] == 0) cur.push_back(i);
        int rank = 0;
        while (!cur.empty()) {
            std::vector<int> next;
            for (int i : cur) {
                vault.get_ind(static_cast<std::size_t>(i)).rank = rank;
                for (int j : S[i]) if (--count[j] == 0) next.push_back(j);
            }
            std::sort(next.begin(), next.end());
            fronts.push_back(std::move(cur));
            cur = std::move(next);
            ++rank;
        }
        return fronts;
    }

    // NSGA-II's crowding distance over one front (GDE3-2).
    void crowding(DataVault<Ind_t>& vault, const std::vector<int>& front)
    {
        const int m = vault.objs_n();
        for (int v : front) vault.get_ind(static_cast<std::size_t>(v)).crowding_distance = 0.0;
        if (front.empty()) return;
        std::vector<int> s = front;
        for (int k = 0; k < m; ++k) {
            auto f = [&](int v) { return vault.objectives_of(static_cast<std::size_t>(v))[k]; };
            std::stable_sort(s.begin(), s.end(), [&](int a, int b) { return f(a) < f(b); });
            const double inf = std::numeric_limits<double>::infinity();
            vault.get_ind(static_cast<std::size_t>(s.front())).crowding_distance = inf;
            vault.get_ind(static_cast<std::size_t>(s.back())).crowding_distance  = inf;
            const double range = f(s.back()) - f(s.front());
            if (range < 1e-14) continue;
            for (std::size_t i = 1; i + 1 < s.size(); ++i)
                vault.get_ind(static_cast<std::size_t>(s[i])).crowding_distance +=
                    (f(s[i + 1]) - f(s[i - 1])) / range;
        }
    }

    // Reduce [0, total) to the first n slots (step 3, GDE3-1).
    void prune(DataVault<Ind_t>& vault, int total, int n)
    {
        const auto fronts = sort_fronts(vault, total);
        std::vector<int> keep;
        keep.reserve(static_cast<std::size_t>(n));
        for (const auto& front : fronts) {
            const int room = n - static_cast<int>(keep.size());
            if (room <= 0) break;
            if (static_cast<int>(front.size()) <= room) {
                keep.insert(keep.end(), front.begin(), front.end());
                continue;
            }
            std::vector<int> last = front;
            while (static_cast<int>(last.size()) > room) {
                crowding(vault, last);
                std::size_t worst = 0;
                for (std::size_t i = 1; i < last.size(); ++i)
                    if (vault.get_ind(static_cast<std::size_t>(last[i])).crowding_distance <
                        vault.get_ind(static_cast<std::size_t>(last[worst])).crowding_distance)
                        worst = i;
                last.erase(last.begin() + static_cast<std::ptrdiff_t>(worst));
            }
            crowding(vault, last);
            keep.insert(keep.end(), last.begin(), last.end());
        }
        // move keep[i] to slot i
        std::vector<int> pos(static_cast<std::size_t>(total)), at(static_cast<std::size_t>(total));
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at.begin(), at.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int want = keep[static_cast<std::size_t>(i)];
            const int cur  = pos[static_cast<std::size_t>(want)];
            if (cur == i) continue;
            const int other = at[static_cast<std::size_t>(i)];
            vault.swap_active(static_cast<std::size_t>(i), static_cast<std::size_t>(cur));
            pos[static_cast<std::size_t>(want)]  = i;
            pos[static_cast<std::size_t>(other)] = cur;
            at[static_cast<std::size_t>(i)]      = want;
            at[static_cast<std::size_t>(cur)]    = other;
        }
    }

    static void require(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("gde3: continuous variables only (GDE3-6)");
        if (vault.pop_size() < 4)
            throw std::invalid_argument("gde3: NP >= 4 (Eq. 3)");
    }

public:
    GDE3Core() = default;

    void set_seed(unsigned s) { rng_.seed(s); }
    void set_F(double f)      { F_ = f; }
    void set_CR(double c)     { CR_ = c; }
    void set_bound_repair(ops::BoundRepair r) {
        ops::require_repair(r, "gde3", false, false);
        repair_ = r;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        require(vault);
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        std::vector<double> x(static_cast<std::size_t>(vault.vars_n()));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                const double lo = bounds[j].first.value_or(0.0);
                const double hi = bounds[j].second.value_or(1.0);
                x[static_cast<std::size_t>(j)] = lo + U01(rng_) * (hi - lo);
            }
            vault.set_variables(static_cast<std::size_t>(i), x);
        }
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) { require(vault); }

    void step(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        const int base = vault.expand(n);                  // u_i in slot base + i
        std::uniform_int_distribution<int> pick(0, n - 1);
        std::vector<double> u;
        for (int i = 0; i < n; ++i) {
            int r1, r2, r3;
            do r1 = pick(rng_); while (r1 == i);
            do r2 = pick(rng_); while (r2 == i || r2 == r1);
            do r3 = pick(rng_); while (r3 == i || r3 == r1 || r3 == r2);
            ops::de_rand_1_bin(vault.variables_of(static_cast<std::size_t>(r3)),
                               vault.variables_of(static_cast<std::size_t>(r1)),
                               vault.variables_of(static_cast<std::size_t>(r2)),
                               vault.variables_of(static_cast<std::size_t>(i)),
                               u, bounds, F_, CR_, repair_, rng_);
            vault.set_variables(static_cast<std::size_t>(base + i), u);
        }
        vault.sync();

        // Select: the trial replaces its parent, joins the population, or goes.
        std::vector<int> extra;
        for (int i = 0; i < n; ++i) {
            const int t = base + i;
            if (cdom(vault, t, i, false)) {                 // u_i ⪯_c x_i
                vault.swap_active(static_cast<std::size_t>(i), static_cast<std::size_t>(t));
                continue;
            }
            const bool feasible = !constrained() || vault.get_cv(static_cast<std::size_t>(t)) <= 0.0;
            const std::vector<double> fx = vault.objectives_of(static_cast<std::size_t>(i));
            if (feasible && !dominates(fx, vault.objectives_of(static_cast<std::size_t>(t)), true))
                extra.push_back(t);                        // x_i ⊀ u_i: both stay
        }
        // the added trials to slots n, n+1, ... (extra is ascending, extra[k] ≥ n + k)
        for (std::size_t k = 0; k < extra.size(); ++k)
            vault.swap_active(static_cast<std::size_t>(base) + k, static_cast<std::size_t>(extra[k]));
        const int total = n + static_cast<int>(extra.size());
        if (total > n) prune(vault, total, n);
        vault.reduce(n);
    }
};

}   // namespace mootation

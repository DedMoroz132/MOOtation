#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../sdr.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/real_crossover.hpp"
#include "../operators/real_mutation.hpp"
#include "../operators/variation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

// ============================================================================
// NSGA-II/SDR — NSGA-II with the strengthened dominance relation
// Y. Tian, R. Cheng, X. Zhang, Y. Su, Y. Jin, "A Strengthened Dominance
// Relation Considering Convergence and Diversity for Evolutionary
// Many-Objective Optimization", IEEE TEVC 23(2):331-345, 2019,
// doi:10.1109/TEVC.2018.2866854          (source: nsga2-sdr_tian2019_accepted)
// The relation itself, (12)-(15), and the library's rules SDR-1..3 where the
// paper is silent: sdr.hpp.
//
// Algorithm 1 — NSGA-II (nsga2.hpp) with Pareto dominance replaced by SDR:
//   1. Binary tournament by (layer, crowding distance)            (line 6)
//   2. SBX + polynomial mutation: N offspring                      (line 7)
//   3. R = P ∪ Q (2N); normalise R by its own minima and maxima    (line 9)
//   4. Deb's fast non-dominated sort of R by SDR, θ̄ from R itself  (line 10)
//   5. Crowding distance per layer; whole layers kept, the last
//      one cut by descending crowding distance                     (11-16)
// The initial population is sorted the same way, θ̄ from P (lines 2-4).
// One generation: N evaluations, one batch. The answer is the population.
//
// Defaults: NSGA-II's, as the core nsga2 has them (Deb et al. 2002, §IV-A):
//   pc = 0.9, pm = 1/n, eta_c = eta_m = 20. The paper's runs (§IV-A.3) used
//   pc = 1 for every algorithm; set pc = 1 for them. Whether the default
//   should be the paper's pc = 1 is an open question, to be settled by a
//   separate check; until then it stays nsga2's 0.9.
// The sort is Deb's with every pair compared directly, O(M N²) for the
//   angles and O(N²) comparisons: SDR is not transitive (Property 3), so no
//   faster sort applies (the paper, after Property 3).
//
// Deviations and readings:
//   SDR-A duplicates stay: the population is always N. Two copies of a
//         point have θ = 0 and one Con, so neither dominates the other.
//   SDR-B crowding distance, the tournament and the cut of the last layer
//         are nsga2's, in the raw objectives (nsga2.hpp, its deviations
//         included: CD over the layer's own range, tournament with
//         replacement, a tie goes to the second).
// Extension beyond the paper — constraints (constraint_mode FEASIBILITY or
//   CDP, the two alike, off by default): a feasible point dominates an
//   infeasible one, of two infeasible ones the smaller violation dominates,
//   and SDR decides between two feasible ones only — its normalisation and
//   θ̄ over the feasible part of R alone. The tournament compares
//   feasibility first, as nsga2's does.
// The switchable operators (crossover, mutation, bound_repair; the mixed
//   real + binary genome) are nsga2's, called the same way. The relation has
//   no knob.
// ============================================================================
template <typename Ind_t>
class NSGAIISDRCore {
public:
    // Supported: NONE, FEASIBILITY, CDP
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double eta_c_ = 20.0;
    double eta_m_ = 20.0;
    double pc_    = 0.9;
    ops::CrossoverSpec xover_;
    std::optional<double> pm_;
    ops::MutationSpec  mut_;
    std::mt19937 rng_{std::random_device{}()};
    double theta_bar_ = 0.0;     // of the last sort
    double first_share_ = 0.0;   // |F_1| / |R| of the last sort

    // The layers of the first n active points: Deb's sort by SDR among the
    // feasible ones, feasibility first (the header). Sets rank.
    std::vector<std::vector<int>> sdr_sort(DataVault<Ind_t>& vault, int n)
    {
        const bool constrained = constraint_mode == ConstraintMode::CDP ||
                                 constraint_mode == ConstraintMode::FEASIBILITY;
        std::vector<double> cv(static_cast<std::size_t>(n), 0.0);
        if (constrained)
            for (int i = 0; i < n; ++i) cv[static_cast<std::size_t>(i)] = vault.get_cv(i);

        // the frame of the feasible part (all of R without constraints)
        std::vector<std::size_t> at(static_cast<std::size_t>(n), 0);
        std::vector<std::vector<double>> F;
        for (int i = 0; i < n; ++i)
            if (cv[static_cast<std::size_t>(i)] <= 0.0) {
                at[static_cast<std::size_t>(i)] = F.size();
                F.push_back(vault.objectives_of(i));
            }
        const sdr::Frame fr = sdr::make_frame(F);

        auto fronts = sdr::fast_sort(n, [&](int i, int j) {
            const double ci = cv[static_cast<std::size_t>(i)];
            const double cj = cv[static_cast<std::size_t>(j)];
            if (ci > 0.0) return cj > 0.0 && ci < cj;    // infeasible: by violation
            if (cj > 0.0) return true;                    // feasible over infeasible
            return sdr::dominates(fr, at[static_cast<std::size_t>(i)],
                                  at[static_cast<std::size_t>(j)]);
        });
        for (std::size_t k = 0; k < fronts.size(); ++k)
            for (int i : fronts[k]) vault.get_ind(i).rank = static_cast<int>(k);
        theta_bar_ = fr.theta_bar;
        first_share_ = n > 0 ? static_cast<double>(fronts[0].size()) / n : 0.0;
        return fronts;
    }

    // NSGA-II's crowding distance (§III-B of Deb et al. 2002), as in nsga2.hpp.
    void assign_crowding_distance(DataVault<Ind_t>& vault, const std::vector<int>& front)
    {
        const int m = vault.objs_n();
        const int l = static_cast<int>(front.size());
        if (l == 0) return;
        for (int v : front) vault.get_ind(v).crowding_distance = 0.0;
        for (int obj = 0; obj < m; ++obj) {
            std::vector<int> sorted = front;
            std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                return vault.objectives_of(a)[obj] < vault.objectives_of(b)[obj];
            });
            vault.get_ind(sorted.front()).crowding_distance = std::numeric_limits<double>::infinity();
            vault.get_ind(sorted.back()).crowding_distance  = std::numeric_limits<double>::infinity();
            const double range = vault.objectives_of(sorted.back())[obj] -
                                 vault.objectives_of(sorted.front())[obj];
            if (range < 1e-14) continue;
            for (int i = 1; i < l - 1; ++i) {
                const double prev = vault.objectives_of(sorted[i - 1])[obj];
                const double next = vault.objectives_of(sorted[i + 1])[obj];
                vault.get_ind(sorted[i]).crowding_distance += (next - prev) / range;
            }
        }
    }

    void sort_and_crowd(DataVault<Ind_t>& vault, int n,
                        std::vector<std::vector<int>>* out = nullptr)
    {
        auto fronts = sdr_sort(vault, n);
        for (auto& front : fronts) assign_crowding_distance(vault, front);
        if (out) *out = std::move(fronts);
    }

    // The tournament's comparison: feasibility first under constraint
    // handling, then the lower layer, then the larger crowding distance; on
    // a tie b.
    int better(DataVault<Ind_t>& vault, int a, int b)
    {
        if (constraint_mode == ConstraintMode::FEASIBILITY ||
            constraint_mode == ConstraintMode::CDP) {
            const double cva = vault.get_cv(a), cvb = vault.get_cv(b);
            const bool a_feas = cva <= 0.0, b_feas = cvb <= 0.0;
            if ( a_feas && !b_feas) return a;
            if (!a_feas &&  b_feas) return b;
            if (!a_feas && !b_feas) return (cva < cvb) ? a : b;
        }
        const auto& ia = vault.get_ind(a);
        const auto& ib = vault.get_ind(b);
        if (ia.rank < ib.rank) return a;
        if (ib.rank < ia.rank) return b;
        return (ia.crowding_distance > ib.crowding_distance) ? a : b;
    }

    int tournament(DataVault<Ind_t>& vault, std::uniform_int_distribution<int>& dist)
    {
        const int a = dist(rng_), b = dist(rng_);
        return better(vault, a, b);
    }

public:
    NSGAIISDRCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_pm(double p)            { pm_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }
    void set_crossover(ops::Crossover c)      { xover_.kind = c; }
    void set_mutation(ops::Mutation m)        { mut_.kind = m; }
    void set_mutation_scale(double s)         { mut_.scale = ops::checked_mutation_scale(s); }
    void set_mixture_q(double q)              { mut_.q = q; }
    void set_blx_alpha(double a)              { xover_.alpha = a; }
    void set_F(double f)                      { xover_.de.set.F  = ops::checked_de_F(f); }
    void set_CR(double c)                     { xover_.de.set.CR = ops::checked_de_CR(c); }
    bool set_param(const std::string& k, double v)             { return xover_.set_param(k, v); }
    bool set_word(const std::string& k, const std::string& w)  { return xover_.set_word(k, w); }
    void set_bound_repair(ops::BoundRepair r) {
        ops::require_repair(r, "nsga2_sdr", true, false);
        xover_.repair = r; mut_.repair = r;
    }

    // θ̄ and the share of the first layer in the set of the last sort (R
    // after a step, P after setup): diagnostics and tests.
    double theta_bar() const { return theta_bar_; }
    double first_layer_share() const { return first_share_; }

    void setup(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                const double lo = bounds[j].first .value_or(0.0);
                const double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist_real(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bvars[j] = ops::random_gene(j, rng_);
            if (vault.bin_vars_n() > 0)
                vault.set_all_variables(i, vars, bvars);
            else
                vault.set_variables(i, vars);
        }
        vault.sync();
        sort_and_crowd(vault, n);                       // Algorithm 1, lines 2-4
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        sort_and_crowd(vault, vault.pop_size());
    }

    void step(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        xover_.observe(vault, n, rng_);  // de, de_pbest, portfolio: the last step's verdict
        // de_pbest: a random one of the best ⌈p·n⌉, by layer, then crowding distance
        std::vector<int> order;
        auto best = [&](double p) {
            if (order.empty()) {
                order.resize(static_cast<std::size_t>(n));
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    const auto& ia = vault.get_ind(a);
                    const auto& ib = vault.get_ind(b);
                    if (ia.rank != ib.rank) return ia.rank < ib.rank;
                    return ia.crowding_distance > ib.crowding_distance;
                });
            }
            const int k = std::max(1, static_cast<int>(std::ceil(p * n)));
            std::uniform_int_distribution<int> top(0, k - 1);
            return vault.variables_of(order[static_cast<std::size_t>(top(rng_))]);
        };

        const int off_base = vault.expand(n);  // [off_base, off_base+n) — offspring slots
        auto draw = [&] { return vault.variables_of(tournament(vault, dist_int)); };
        const bool bin = vault.bin_vars_n() > 0;
        auto parent = [&](std::vector<double>& x, std::vector<int>& bv) {
            const int p = tournament(vault, dist_int);
            for (int j = 0; j < vault.vars_n(); ++j) x[j] = vault.get_variable(p, j);
            if (bin) bv = vault.binary_variables_of(static_cast<std::size_t>(p));
        };

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        std::vector<int> bv1, bv2;
        for (int i = 0; i < n; i += 2) {                // Algorithm 1, lines 6-7
            parent(pv1, bv1);
            parent(pv2, bv2);
            const double pm = pm_ ? *pm_ : (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::vary(xover_, mut_, pv1, pv2, draw, best, n, c1, c2, bounds,
                      eta_c_, pc_, eta_m_, pm, rng_, [] { return -1; });
            xover_.finish(c1, c2, i + 1 < n ? -1 : 0);

            if (bin) {
                std::vector<int> bc1 = bv1, bc2 = bv2;
                ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i + 1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();
        select(vault);
    }

    // The environmental selection on the 2N active points, Algorithm 1,
    // lines 9-16. Public for the tests.
    void select(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        std::vector<std::vector<int>> fronts;
        sort_and_crowd(vault, 2 * n, &fronts);

        std::vector<int> survivors;
        survivors.reserve(static_cast<std::size_t>(n));
        for (auto& front : fronts) {
            const int have = static_cast<int>(survivors.size());
            if (have + static_cast<int>(front.size()) <= n) {
                for (int v : front) survivors.push_back(v);
            } else {
                const int needed = n - have;
                std::partial_sort(front.begin(), front.begin() + needed, front.end(),
                    [&](int a, int b) {
                        return vault.get_ind(a).crowding_distance
                             > vault.get_ind(b).crowding_distance;
                    });
                for (int i = 0; i < needed; ++i) survivors.push_back(front[i]);
            }
            if (static_cast<int>(survivors.size()) == n) break;
        }

        // survivors[i] to position i: pos[v] is where the point first at v is
        // now, at_pos[p] the point now at p (as nsga2.hpp)
        std::vector<int> pos(static_cast<std::size_t>(2 * n));
        std::vector<int> at_pos(static_cast<std::size_t>(2 * n));
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int want = survivors[static_cast<std::size_t>(i)];
            const int cur  = pos[static_cast<std::size_t>(want)];
            if (cur == i) continue;
            const int other = at_pos[static_cast<std::size_t>(i)];
            vault.swap_active(i, cur);
            pos[static_cast<std::size_t>(want)]  = i;
            pos[static_cast<std::size_t>(other)] = cur;
            at_pos[static_cast<std::size_t>(i)]   = want;
            at_pos[static_cast<std::size_t>(cur)] = other;
        }
        vault.reduce(n);
    }
};

} // namespace mootation

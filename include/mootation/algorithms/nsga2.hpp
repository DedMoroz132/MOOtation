#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

// ============================================================================
// NSGA-II — A Fast and Elitist Multiobjective Genetic Algorithm
// K. Deb, A. Pratap, S. Agarwal, T. Meyarivan — IEEE TEVC 6(2), 2002
// doi:10.1109/4235.996017          (source: deb2002)
//
// Generational scheme:
//   1. Binary tournament by ≺_n (rank, then crowding distance; CDP — §VI)
//   2. SBX (eta_c=20, pc=0.9) + polynomial mutation (eta_m=20, pm=1/n)
//   3. R_t = P_t ∪ Q_t (2N) → fast non-dominated sort (§III-A)
//   4. Whole fronts are accepted; the last front — selection by descending CD (§III-B)
//   5. Permute survivors into [0,N), reduce to N
//
// Defaults = §IV-A: pc=0.9, pm=1/n, eta_c=20, eta_m=20.
// Deviations (MINOR, deliberately left unchanged; internal audit):
//   - CD normalization uses min/max of the current FRONT (the paper is ambiguous);
//   - tournament "with replacement" (a==b is possible) instead of two permutations;
//   - deterministic tournament tie-breaks (on a tie — `b`);
//   - binary genome: uniform crossover p=1 instead of single-point pc=0.9.
// Extensions beyond the paper: constraint_mode FEASIBILITY (off by default;
//   CDP follows §VI of the paper); mixed real+binary genome.
// ============================================================================
template <typename Ind_t>
class NSGAIICore {
public:
    // Supported: NONE, FEASIBILITY, CDP
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 20.0;
    double       eta_m_ = 20.0;
    double       pc_    = 0.9;   // §IV-A: "crossover probability of p_c = 0.9"
    std::mt19937 rng_{std::random_device{}()};

    bool dominates_plain(const std::vector<double>& a,
                         const std::vector<double>& b) const
    {
        bool any_better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) any_better = true;
        }
        return any_better;
    }

    // CDP: Constrained Domination
    bool dominates_cdp(const std::vector<double>& fa, double cva,
                       const std::vector<double>& fb, double cvb) const
    {
        bool a_feas = (cva <= 0.0);
        bool b_feas = (cvb <= 0.0);
        if ( a_feas && !b_feas) return true;
        if (!a_feas &&  b_feas) return false;
        if (!a_feas && !b_feas) return cva < cvb;
        return dominates_plain(fa, fb);
    }

    std::vector<std::vector<int>>
    fast_nondominated_sort(DataVault<Ind_t>& vault, int n)
    {
        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::CDP ||
            constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        std::vector<std::vector<int>> S(n);
        std::vector<int>              np(n, 0);

        for (int i = 0; i < n; ++i) {
            const auto& fi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& fj = vault.objectives_of(j);
                bool i_dom_j, j_dom_i;
                if (constraint_mode == ConstraintMode::CDP ||
                    constraint_mode == ConstraintMode::FEASIBILITY) {
                    i_dom_j = dominates_cdp(fi, cvs[i], fj, cvs[j]);
                    j_dom_i = dominates_cdp(fj, cvs[j], fi, cvs[i]);
                } else {
                    i_dom_j = dominates_plain(fi, fj);
                    j_dom_i = dominates_plain(fj, fi);
                }
                if (i_dom_j)      S[i].push_back(j);
                else if (j_dom_i) ++np[i];
            }
            if (np[i] == 0) vault.get_ind(i).rank = 0;
        }

        std::vector<std::vector<int>> fronts;
        std::vector<int> front0;
        for (int i = 0; i < n; ++i)
            if (np[i] == 0) front0.push_back(i);
        fronts.push_back(front0);

        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> next_front;
            for (int i : fronts[k]) {
                for (int j : S[i]) {
                    if (--np[j] == 0) {
                        vault.get_ind(j).rank = k + 1;
                        next_front.push_back(j);
                    }
                }
            }
            fronts.push_back(next_front);
            ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    void assign_crowding_distance(DataVault<Ind_t>& vault,
                                  const std::vector<int>& front)
    {
        int m = vault.objs_n();
        int l = static_cast<int>(front.size());
        if (l == 0) return;

        for (int v : front) vault.get_ind(v).crowding_distance = 0.0;

        for (int obj = 0; obj < m; ++obj) {
            std::vector<int> sorted = front;
            std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                return vault.objectives_of(a)[obj] < vault.objectives_of(b)[obj];
            });

            vault.get_ind(sorted.front()).crowding_distance =
                std::numeric_limits<double>::infinity();
            vault.get_ind(sorted.back()).crowding_distance  =
                std::numeric_limits<double>::infinity();

            double f_min = vault.objectives_of(sorted.front())[obj];
            double f_max = vault.objectives_of(sorted.back())[obj];
            double range = f_max - f_min;
            if (range < 1e-14) continue;

            for (int i = 1; i < l - 1; ++i) {
                double prev = vault.objectives_of(sorted[i - 1])[obj];
                double next = vault.objectives_of(sorted[i + 1])[obj];
                vault.get_ind(sorted[i]).crowding_distance += (next - prev) / range;
            }
        }
    }

    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist)
    {
        int a = dist(rng_), b = dist(rng_);
        const auto& ia = vault.get_ind(a);
        const auto& ib = vault.get_ind(b);

        if (constraint_mode == ConstraintMode::FEASIBILITY ||
            constraint_mode == ConstraintMode::CDP) {
            double cva = vault.get_cv(a);
            double cvb = vault.get_cv(b);
            bool a_feas = (cva <= 0.0);
            bool b_feas = (cvb <= 0.0);
            if ( a_feas && !b_feas) return a;
            if (!a_feas &&  b_feas) return b;
            if (!a_feas && !b_feas) return (cva < cvb) ? a : b;
        }
        if (ia.rank < ib.rank) return a;
        if (ib.rank < ia.rank) return b;
        return (ia.crowding_distance > ib.crowding_distance) ? a : b;
    }

public:
    NSGAIICore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::uniform_int_distribution<int>     dist_bin (0, 1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist_real(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bvars[j] = dist_bin(rng_);

            if (vault.bin_vars_n() > 0)
                vault.set_all_variables(i, vars, bvars);
            else
                vault.set_variables(i, vars);
        }
        vault.sync();

        auto fronts = fast_nondominated_sort(vault, n);
        for (auto& front : fronts) assign_crowding_distance(vault, front);
    }

    // ============================================================
    //  setup_seeded — for resume via Optimizer::setup_with_seed.
    //  The vault is already seeded with the seed_individual → skip the
    //  random generation, only recompute ranks + crowding distance.
    // ============================================================
    void setup_seeded(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        auto fronts = fast_nondominated_sort(vault, n);
        for (auto& front : fronts) assign_crowding_distance(vault, front);
    }

    void step(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        int off_base = vault.expand(n);  // [off_base, off_base+n) — offspring slots

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_int);
            int p2 = tournament(vault, dist_int);

            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            // §IV-A: pc=0.9, pm=1/n
            double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);

            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n());
                std::vector<int> bc1, bc2;
                for (int j = 0; j < vault.bin_vars_n(); ++j) {
                    bv1[j] = vault.get_bin_variable(p1, j);
                    bv2[j] = vault.get_bin_variable(p2, j);
                }
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

        auto fronts = fast_nondominated_sort(vault, n * 2);
        for (auto& front : fronts) assign_crowding_distance(vault, front);

        std::vector<int> survivors;
        survivors.reserve(n);
        for (auto& front : fronts) {
            if (static_cast<int>(survivors.size()) + static_cast<int>(front.size()) <= n) {
                for (int v : front) survivors.push_back(v);
            } else {
                int needed = n - static_cast<int>(survivors.size());
                std::partial_sort(front.begin(), front.begin() + needed, front.end(),
                    [&](int a, int b) {
                        return vault.get_ind(a).crowding_distance
                             > vault.get_ind(b).crowding_distance;
                    });
                for (int i = 0; i < needed; ++i) survivors.push_back(front[i]);
                break;
            }
            if (static_cast<int>(survivors.size()) == n) break;
        }

        // Permutation: survivors[i] goes to position i.
        // pos[v] = current virtual position of the individual originally at v.
        // at_pos[p] = virtual "original id" currently located at position p.
        // Both maps are updated in O(1) per swap → O(N) total instead of O(N²).
        std::vector<int> pos   (n * 2);
        std::vector<int> at_pos(n * 2);
        std::iota(pos.begin(),    pos.end(),    0);
        std::iota(at_pos.begin(), at_pos.end(), 0);

        for (int i = 0; i < n; ++i) {
            int want = survivors[i];
            int cur  = pos[want];
            if (cur == i) continue;

            int other = at_pos[i];
            vault.swap_active(i, cur);
            pos[want]    = i;
            pos[other]   = cur;
            at_pos[i]    = want;
            at_pos[cur]  = other;
        }

        vault.reduce(n);
    }
};

} // namespace mootation

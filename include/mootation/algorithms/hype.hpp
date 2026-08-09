#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// HypE — Hypervolume Estimation Algorithm for Multiobjective Optimization
// J. Bader, E. Zitzler — Evolutionary Computation 19(1), 2011
// doi:10.1162/EVCO_a_00009
//
// Generational scheme (Alg. 4):
//   1. Mating (Alg. 5): fitness I_h^k with k = |P| (alpha_i == 1, the full
//      I_h); binary tournament "if v_a > v_b then a else b".
//   2. Variation: SBX (eta_c = 20, §6.1 "the SBX-20 operator") + polynomial
//      mutation (p_m = 1/n by §6.1's delegation; eta_m = 20 is conventional,
//      not from the paper — see PAPER DEFAULTS) producing N offspring.
//   3. Environmental selection (Alg. 6): non-dominated sorting with greedy
//      front transfer; the last front Q' that does not fit is truncated
//      iteratively: fitness I_h^k over Q' (k = |Q|+|Q'|−N, decremented after
//      each removal), the worst is dropped; ties at the minimum are "selected
//      uniformly at random" (§5).
//
// Computing I_h^k (§5): for n <= 3 objectives it is EXACT, Alg. 1/2
// (hypeIndicatorExact: recursive slicing along the HSO principle
// [Zitzler 2001; While 2006; cf. Emmerich 2005] with alpha-weighted contributions
// cells, Theorem 3: alpha_i = Prod_{j=1..i-1}(k−j)/(|P|−j)); for n > 3 it is a
// Monte-Carlo estimate (Alg. 3, box Eq. 16–17, M = 10000 samples by default,
// §6.1).
//
// Reference set R: in the paper this is an external fixed parameter (a Require
// of every algorithm); set it with set_reference_point().
// DECLARED DEVIATION (the default when R is not supplied): an adaptive point
// r_i = max_i f_i · (1 ± 0.1) over the current pool. This is a practical
// default for problems of unknown scale; the paper does NOT define such a
// construction.
// DECLARED DEVIATION (domain narrowing): only |R| = 1 is supported, while the
// paper allows a reference SET (§3.3 works a two-point example). At |R| = 1 the
// UR filter of Alg.2 line 4 and the "∃r∈R : s ≼ r" test of Alg.3 line 11 both
// degenerate into a per-dimension bound, which is exactly what slice_rec and
// hype_fitness_mc implement; for |R| > 1 they would not.
//
// PAPER DEFAULTS (§6.1): M = 10000; η_c = 20 ("the SBX-20 operator");
//   p_c = 1.0 and p_m = 1/n via §6.1's delegation of the PROBABILITIES to
//   Deb et al. (2005).
// BEYOND THE PAPER — conventional default: η_m = 20. §6.1 specifies only "a
//   polynomial distribution ... for mutation"; the delegation sentence covers
//   "the recombination and mutation probabilities", and a distribution index is
//   not a probability, so the η_m NUMBER appears nowhere in the paper.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY
// (CDP sorting, penalty fitness for infeasible solutions), binary variables.
// ============================================================================

#include <algorithm>
#include <cmath>
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

template <typename Ind_t>
class HypECore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    int          n_samples_  = 10000;  // M: MC sample count (paper §6.1)
    double       ref_offset_ = 0.1;    // adaptive ref point offset (deviation)
    double       eta_c_      = 20.0;
    double       eta_m_      = 20.0;
    std::mt19937 rng_{std::random_device{}()};

    // External fixed reference point R (paper: Require of Alg. 1/3/5/6).
    // Empty — adaptive default is used (documented deviation, see header).
    std::vector<double> ref_point_ext_;

    // ── dominance helpers ─────────────────────────────────────────────────

    bool dominates_plain(const std::vector<double>& a,
                         const std::vector<double>& b) const {
        bool any_better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) any_better = true;
        }
        return any_better;
    }

    bool dominates_cdp(const std::vector<double>& fa, double cva,
                       const std::vector<double>& fb, double cvb) const {
        const bool a_feas = (cva <= 0.0), b_feas = (cvb <= 0.0);
        if ( a_feas && !b_feas) return true;
        if (!a_feas &&  b_feas) return false;
        if (!a_feas && !b_feas) return cva < cvb;
        return dominates_plain(fa, fb);
    }

    // ── non-dominated sort returning front vectors ─────────────────────────
    std::vector<std::vector<int>>
    fast_nondominated_sort(DataVault<Ind_t>& vault, int n) {
        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        std::vector<std::vector<int>> S(n);
        std::vector<int>              np(n, 0);

        for (int i = 0; i < n; ++i) {
            const auto& fi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& fj = vault.objectives_of(j);
                bool i_dom_j, j_dom_i;
                if (constraint_mode == ConstraintMode::FEASIBILITY) {
                    i_dom_j = dominates_cdp(fi, cvs[i], fj, cvs[j]);
                    j_dom_i = dominates_cdp(fj, cvs[j], fi, cvs[i]);
                } else {
                    i_dom_j = dominates_plain(fi, fj);
                    j_dom_i = dominates_plain(fj, fi);
                }
                if (i_dom_j) S[i].push_back(j);
                else if (j_dom_i) ++np[i];
            }
        }

        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for (int i = 0; i < n; ++i) if (np[i] == 0) f0.push_back(i);
        fronts.push_back(f0);

        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> next;
            for (int i : fronts[k])
                for (int j : S[i])
                    if (--np[j] == 0) next.push_back(j);
            fronts.push_back(next);
            ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── reference point ───────────────────────────────────────────────────
    // External R when set; otherwise the adaptive default
    // r_i = max_{x in pool} f_i(x) · (1 ± ref_offset_) (deviation, see header).
    std::vector<double> reference_point(DataVault<Ind_t>& vault, int n) {
        if (!ref_point_ext_.empty()) return ref_point_ext_;
        int m = vault.objs_n();
        std::vector<double> rp(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int k = 0; k < m; ++k)
                rp[k] = std::max(rp[k], o[k]);
        }
        for (int k = 0; k < m; ++k) {
            double val = rp[k];
            rp[k] = (val >= 0.0) ? val * (1.0 + ref_offset_)
                                  : val * (1.0 - ref_offset_);
        }
        return rp;
    }

    // ── α coefficients (Theorem 3): α_u = Π_{j=1}^{u-1} (k−j)/(NF−j) ──────
    static std::vector<double> alpha_coeffs(int k, int NF) {
        int keff = std::min(k, NF);
        std::vector<double> alpha(std::max(keff, 1) + 1, 0.0);
        if (keff >= 1) alpha[1] = 1.0;
        for (int u = 2; u <= keff; ++u) {
            double denom = static_cast<double>(NF - (u - 1));
            if (denom <= 0.0) { alpha[u] = 0.0; continue; }
            alpha[u] = alpha[u - 1]
                     * static_cast<double>(k - (u - 1)) / denom;
        }
        return alpha;
    }

    // ── exact I_h^k via recursive objective-space partitioning ─────────────
    // (Alg. 1/2, doSlicing). pts — objective vectors of the evaluated set,
    // ref — single reference point; cells beyond ref are excluded (a cell
    // whose lower edge equals ref_d is outside H(P,R) and is skipped).
    // The candidate list carries the UP-filter incrementally (Line 3).
    void slice_rec(const std::vector<std::vector<double>>& pts,
                   const std::vector<double>& ref,
                   int keff, int level, double V,
                   const std::vector<int>& cand,
                   const std::vector<double>& alpha,
                   std::vector<double>& fit) const {
        if (level == 0) {                       // end of recursion (Lines 5-14)
            int u = static_cast<int>(cand.size());
            if (u >= 1 && u <= keff) {
                double c = alpha[u] / static_cast<double>(u) * V;
                for (int a : cand) fit[a] += c;
            }
            return;
        }
        int d = level - 1;                      // scan dimension (Lines 18-26)
        std::vector<int> order = cand;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return pts[a][d] < pts[b][d];
        });
        std::vector<int> child;
        child.reserve(order.size());
        std::size_t ptr = 0;
        while (ptr < order.size()) {
            double lo = pts[order[ptr]][d];
            if (lo >= ref[d]) break;            // slice outside the ref bound
            while (ptr < order.size() && pts[order[ptr]][d] <= lo)
                child.push_back(order[ptr++]);
            double hi = (ptr < order.size())
                        ? std::min(pts[order[ptr]][d], ref[d])
                        : ref[d];
            if (hi > lo)
                slice_rec(pts, ref, keff, level - 1, V * (hi - lo),
                          child, alpha, fit);
        }
    }

    // Exact fitness (computeHypervolume, Alg. 1): fills fit[0..NF) for the
    // point set pts with parameter k.
    void hype_fitness_exact(const std::vector<std::vector<double>>& pts,
                            const std::vector<double>& ref, int k,
                            std::vector<double>& fit) const {
        int NF = static_cast<int>(pts.size());
        fit.assign(NF, 0.0);
        if (NF == 0 || k <= 0) return;
        int keff = std::min(k, NF);
        auto alpha = alpha_coeffs(k, NF);
        std::vector<int> cand(NF);
        std::iota(cand.begin(), cand.end(), 0);
        slice_rec(pts, ref, keff, static_cast<int>(ref.size()),
                  1.0, cand, alpha, fit);
    }

    // MC fitness estimation (estimateHypervolume, Alg. 3): box Eq. 16-17,
    // M = n_samples_; credit α_{|A|}/|A| · V/M to each a in A (Lines 9-26).
    void hype_fitness_mc(const std::vector<std::vector<double>>& pts,
                         const std::vector<double>& ref, int k,
                         std::vector<double>& fit) {
        int NF = static_cast<int>(pts.size());
        int m  = static_cast<int>(ref.size());
        fit.assign(NF, 0.0);
        if (NF == 0 || k <= 0) return;

        std::vector<double> lo(m, std::numeric_limits<double>::max());
        for (const auto& p : pts)
            for (int j = 0; j < m; ++j) lo[j] = std::min(lo[j], p[j]);

        double vol = 1.0;
        for (int j = 0; j < m; ++j)
            vol *= std::max(0.0, ref[j] - lo[j]);
        if (vol < 1e-30) return;                // degenerate box

        int keff = std::min(k, NF);
        auto alpha = alpha_coeffs(k, NF);

        std::vector<std::uniform_real_distribution<double>> dists;
        dists.reserve(m);
        for (int j = 0; j < m; ++j) dists.emplace_back(lo[j], ref[j]);

        const double inv_M = 1.0 / static_cast<double>(n_samples_);
        std::vector<double> s(m);
        std::vector<int> A;
        for (int smp = 0; smp < n_samples_; ++smp) {
            for (int j = 0; j < m; ++j) s[j] = dists[j](rng_);
            // s <= r (single reference point) holds by box construction.
            A.clear();
            for (int i = 0; i < NF; ++i) {
                bool dom = true;
                for (int j = 0; j < m; ++j)
                    if (pts[i][j] > s[j]) { dom = false; break; }
                if (dom) A.push_back(i);
            }
            int ai = static_cast<int>(A.size());
            if (ai < 1 || ai > keff) continue;
            double contrib = alpha[ai] / static_cast<double>(ai) * inv_M * vol;
            for (int idx : A) fit[idx] += contrib;
        }
    }

    // ── fitness assignment for a subset of vault slots ─────────────────────
    // Exact (n <= 3, Alg. 1) or MC (n > 3, Alg. 3) — paper §5 / Alg. 5/6.
    // FEASIBILITY mode (extension): infeasible get a large negative fitness
    // and are excluded from the HV computation; the α denominator is the
    // feasible-set size.
    void assign_fitness(DataVault<Ind_t>& vault, const std::vector<int>& idx,
                        const std::vector<double>& ref, int k) {
        int m = vault.objs_n();

        std::vector<int> feas;
        feas.reserve(idx.size());
        for (int i : idx) {
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double cv = vault.get_cv(i);
                if (cv > 0.0) {
                    vault.get_ind(i).fitness = -(1e6 + cv);
                    continue;
                }
            }
            feas.push_back(i);
        }
        if (feas.empty()) return;

        std::vector<std::vector<double>> pts;
        pts.reserve(feas.size());
        for (int i : feas) pts.push_back(vault.objectives_of(i));

        std::vector<double> fit;
        if (m <= 3) hype_fitness_exact(pts, ref, k, fit);
        else        hype_fitness_mc   (pts, ref, k, fit);

        for (std::size_t t = 0; t < feas.size(); ++t)
            vault.get_ind(feas[t]).fitness = fit[t];
    }

    // ── binary tournament on fitness (Alg. 5: if v_a > v_b then a else b) ──
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_), b = dist(rng_);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double cva = vault.get_cv(a), cvb = vault.get_cv(b);
            bool a_feas = (cva <= 0.0), b_feas = (cvb <= 0.0);
            if ( a_feas && !b_feas) return a;
            if (!a_feas &&  b_feas) return b;
            if (!a_feas && !b_feas) return (cva < cvb) ? a : b;
        }
        return (vault.get_ind(a).fitness > vault.get_ind(b).fitness) ? a : b;
    }

    // ── environmental selection (Algorithm 6) ────────────────────────────
    void environmental_selection(DataVault<Ind_t>& vault, int pool_size,
                                 int target_n) {
        auto fronts = fast_nondominated_sort(vault, pool_size);

        // Greedily accept fronts (Lines 5-16).
        std::vector<int> accepted;
        accepted.reserve(target_n);
        std::vector<int> last_front;

        for (auto& front : fronts) {
            int total = static_cast<int>(accepted.size() + front.size());
            if (total <= target_n) {
                for (int v : front) accepted.push_back(v);
                if (static_cast<int>(accepted.size()) == target_n) break;
            } else {
                last_front = front;
                break;
            }
        }

        // Truncate Q' = last_front (Lines 17-34): k = |Q|+|Q'|-N, then for
        // each removal recompute I_h^k over Q' only, remove the worst
        // (ties broken uniformly at random, §5), k--.
        if (!last_front.empty()) {
            int needed    = target_n - static_cast<int>(accepted.size());
            int to_remove = static_cast<int>(last_front.size()) - needed;

            // R is fixed during truncation (adaptive default: computed once
            // over the full pool; identical on every removal iteration).
            auto rp = reference_point(vault, pool_size);

            for (int rem = 0; rem < to_remove; ++rem) {
                int k_cur = to_remove - rem;       // remaining removals
                assign_fitness(vault, last_front, rp, k_cur);

                // Worst (lowest fitness); random tie-break (§5, item 2).
                double worst_fit = vault.get_ind(last_front[0]).fitness;
                for (int li : last_front)
                    worst_fit = std::min(worst_fit,
                                         vault.get_ind(li).fitness);
                std::vector<int> ties;
                for (int li = 0; li < static_cast<int>(last_front.size()); ++li)
                    if (vault.get_ind(last_front[li]).fitness == worst_fit)
                        ties.push_back(li);
                int worst_li = ties[std::uniform_int_distribution<int>(
                    0, static_cast<int>(ties.size()) - 1)(rng_)];

                last_front[worst_li] = last_front.back();
                last_front.pop_back();
            }

            for (int v : last_front) accepted.push_back(v);
        }

        // Rearrange vault: move accepted[i] to slot i.
        std::vector<int> pos(pool_size), at_pos(pool_size);
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);

        for (int i = 0; i < target_n; ++i) {
            int want = accepted[i];
            int cur  = pos[want];
            if (cur == i) continue;
            int other = at_pos[i];
            vault.swap_active(i, cur);
            pos[want]  = i;    pos[other] = cur;
            at_pos[i]  = want; at_pos[cur] = other;
        }
        vault.reduce(target_n);
    }

    // Mating-selection fitness over the current population (Alg. 5 Line 2-6:
    // k = |P|, so exact for n <= 3, MC otherwise).
    void mating_fitness(DataVault<Ind_t>& vault, int n) {
        auto rp = reference_point(vault, n);
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        assign_fitness(vault, idx, rp, n);
    }

public:
    HypECore() = default;

    void set_n_samples    (int n)     { n_samples_  = n; }
    void set_ref_offset   (double r)  { ref_offset_ = r; }
    void set_eta_crossover(double e)  { eta_c_ = e; }
    void set_eta_mutation (double e)  { eta_m_ = e; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // External fixed reference point R (paper-exact mode). Pass an empty
    // vector to return to the adaptive default.
    void set_reference_point(const std::vector<double>& r) { ref_point_ext_ = r; }

    // ── setup: random init → evaluate → mating-selection fitness ──────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::uniform_int_distribution<int>     dist_bin (0, 1);
        std::vector<double> vars (vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist_real(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = dist_bin(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables    (i, vars);
        }
        vault.sync();
        mating_fitness(vault, n);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        mating_fitness(vault, vault.pop_size());
    }

    // ── step: one full generation (Algorithm 4) ────────────────────────────
    //
    //   mating selection (tournament on current fitness) → variation →
    //   environmental selection (pool = P ∪ O → N) →
    //   mating fitness for the next generation (k = N over the new P)
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // ── expand: active = 2n ────────────────────────────────────────────
        vault.expand(vault.pop_size());

        // ── breed N offspring into [n, 2n) ────────────────────────────────
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_int);
            int p2 = tournament(vault, dist_int);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, rng_);
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
                vault.set_all_variables(n + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(n + i + 1, c2, bc2);
            } else {
                vault.set_variables(n + i, c1);
                if (i + 1 < n) vault.set_variables(n + i + 1, c2);
            }
        }

        // ── evaluate offspring ─────────────────────────────────────────────
        vault.sync();

        // ── environmental selection: pool [0, 2n) → n survivors ───────────
        environmental_selection(vault, n * 2, n);

        // ── recompute mating-selection fitness for the next generation ─────
        mating_fitness(vault, n);
    }
};

} // namespace mootation

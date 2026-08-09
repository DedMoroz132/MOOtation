#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/DD — An Evolutionary Many-Objective Optimization Algorithm Based on
//           Dominance and Decomposition
// K. Li, K. Deb, Q. Zhang, S. Kwong — IEEE Transactions on Evolutionary
// Computation 19(5), 2015
// doi:10.1109/TEVC.2014.2373386          (source: li2015)
//
// Generation scheme (Alg.1, steady-state, SEQUENTIALLY for i = 1..N):
//   1. MATING_SELECTION (Alg.3): with prob. δ, k SUBREGION indexes are picked
//      from E(i); parents — random among the individuals ASSOCIATED with
//      these subregions; if the selected subregions are empty — random from
//      P; with prob. 1−δ — random from P. Under constraint_mode=FEASIBILITY
//      each parent is selected by a binary CV tournament (Alg.8, C-MOEA/DD
//      §VI-B).
//   2. VARIATION: SBX (p_c = 1.0, η_c = 30) + PM (p_m = 1/n, η_m = 20);
//      one offspring xc; one objective function evaluation (FE/generation = N).
//   3. UPDATE_POPULATION (Alg.4): update z*; associate xc via Eq.6
//      (minimal acute angle); P' = P ∪ {xc}; incremental non-domination
//      level update [66]. Victim selection:
//        l = 1            → LOCATE_WORST(P') (Alg.5);
//        l > 1, |F_l| = 1 → if |Φ^l| > 1 — remove x^l (line 10); if
//                           Φ^l is isolated — x^l SURVIVES, remove
//                           LOCATE_WORST(P') (lines 12–13, "second chance");
//        l > 1, |F_l| > 1 → Φ^h — the most crowded subregion among those
//                           associated with F_l (tie Eq.7); |Φ^h| > 1 →
//                           argmax g^pbi over the WHOLE Φ^h (Eq.8, lines 18–19);
//                           |Φ^h| = 1 → LOCATE_WORST(P') (lines 21–22).
//      The offspring enters P' and may itself be removed; if the offspring
//      is removed, the effects of the incremental insertion are ROLLED BACK
//      (arbitration fix #3).
//   4. C-MOEA/DD (constraint_mode=FEASIBILITY, §VI-A, Alg.7): if P' contains
//      infeasible members — sort by CV in descending order, the first NOT
//      isolated one is removed; if all are isolated — the maximal CV.
//
// Defaults = §IV-D (Table) and §III/§VI: p_c = 1.0, η_c = 30, p_m = 1/n,
//   η_m = 20, θ_PBI = 5.0, T = 20, δ = 0.9; weights — Das–Dennis; when a
//   single-layer lattice is unattainable — the two-layer scheme (Eq.5,
//   τ = 0.5, das_dennis::generate_two_layer).
// Deviations:
//   - |S| = 1: one offspring of the SBX pair is used (Alg.1 line 6 allows
//     several; one offspring yields exactly N FE per generation).
//   - Eq.6: the angle is computed w.r.t. (F(x) − z*), not the raw F(x) —
//     the standard reading, adopted by the audits.
//   - z* is the running component-wise MINIMUM over everything evaluated. §II-B
//     defines the ideal vector with a STRICT inequality, z*_i < min_{x∈Ω}
//     f_i(x), i.e. strictly below the true optimum, which no online estimator
//     can satisfy from observed points alone. The consequence is confined to
//     PBI: with z*_i equal to an attained value, a solution sitting exactly at
//     z* has d_1 = d_2 = 0 and therefore g^pbi = 0, the unbeatable score, and
//     any solution on the ray through z* has d_2 = 0. MOEA/D-AWA solves the
//     same problem by subtracting a fixed 1e-7 (see moead_awa.hpp); this port
//     does not, because §IV-D of this paper reports no such offset and the
//     degenerate case only arises at exact equality.
//   - pbi_value in an individual is an informational cache from the moment
//     of association; in selection the PBI is RECOMPUTED from the current z*
//     — a cached value goes stale as soon as z* moves.
//   - the two mating parents are forced DISTINCT (up to 10 retries); Alg.3 says
//     only "randomly select two solutions". A harmless strengthening, but the
//     retry loop draws a data-dependent number of uniforms, so the RNG stream
//     differs from a plain pair of picks.
// Extensions beyond the paper (disabled by default / active only when binary
// variables are present): UNIFORM crossover + bit-flip for bin_vars
// (general-purpose; see moead.hpp — no one-point operator exists in this
// library).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../detail/math_compat.hpp"
#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class MOEADDCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (for sources see the file header) ──────────────────
    double       theta_ = 5.0;   // §IV-D-4: θ = 5.0
    int          T_     = 20;    // §IV-D-5: T = 20
    double       delta_ = 0.9;   // §IV-D-6: δ = 0.9
    double       eta_c_ = 30.0;  // §IV-D-1: η_c = 30
    double       eta_m_ = 20.0;  // §IV-D-1: η_m = 20
    double       pc_    = 1.0;   // §IV-D-1: p_c = 1.0
    double       pm_    = -1.0;  // §IV-D-1: p_m = 1/n; <0 → auto 1/n
    int          mating_k_ = 2;  // k parents (Alg.3); SBX requires 2
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> W_;     // weight vectors [N][M]
    std::vector<std::vector<int>>    B_;     // neighbourhoods E(i) [N][T]
    std::vector<double>              ideal_; // z*

    double pm_eff(int nv) const {
        return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);
    }

    // ── Weight vector generation (Alg.2 lines 2–12; Eq.5 via das_dennis) ───
    void init_weights(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
    }

    void build_neighbourhood(int n) {
        int T = std::min(T_, n);
        B_.resize(n);
        for (int i=0;i<n;++i) {
            std::vector<std::pair<double,int>> dists;
            dists.reserve(n);
            for (int j=0;j<n;++j) {
                double d=0.0;
                for (std::size_t k=0;k<W_[i].size();++k) {double diff=W_[i][k]-W_[j][k];d+=diff*diff;}
                dists.emplace_back(d,j);
            }
            std::partial_sort(dists.begin(),dists.begin()+T,dists.end());
            B_[i].resize(T); for(int k=0;k<T;++k) B_[i][k]=dists[k].second;
        }
    }

    // ── PBI (Eq.2–4): g^pbi = d1 + θ·d2 — always from the CURRENT z* ───────
    double pbi(DataVault<Ind_t>& vault, int v, int w_idx) const {
        int m = vault.objs_n();
        const auto& fw = W_[w_idx];
        const auto& fo = vault.objectives_of(v);
        double ww = 0.0;
        for (int j=0;j<m;++j) ww += fw[j]*fw[j];
        double norm_w = std::sqrt(ww);
        if (norm_w < 1e-14) return 0.0;
        double dot = 0.0;
        for (int j=0;j<m;++j) dot += (fo[j]-ideal_[j])*fw[j];
        double d1 = dot/norm_w;
        double d2sq = 0.0;
        for (int j=0;j<m;++j) {
            double proj = d1*fw[j]/norm_w;
            double diff = (fo[j]-ideal_[j]) - proj;
            d2sq += diff*diff;
        }
        return d1 + theta_*std::sqrt(d2sq);
    }

    // ── Cosine of the acute angle between (f(x)−z*) and weight w (Eq.6) ───
    // argmin angle ⟺ argmax cos. Returns cos∈[0,1]; −1 on degeneracy.
    double acute_cos(DataVault<Ind_t>& vault, int v, int w_idx) const {
        int m = vault.objs_n();
        const auto& fw = W_[w_idx];
        const auto& fo = vault.objectives_of(v);
        double dot=0.0, ww=0.0, ff=0.0;
        for (int j=0;j<m;++j) {
            double f = fo[j]-ideal_[j];
            dot += f*fw[j]; ww += fw[j]*fw[j]; ff += f*f;
        }
        double denom = std::sqrt(ww)*std::sqrt(ff);
        return (denom > 1e-14) ? dot/denom : -1.0;
    }

    // Associate individual v with a subregion via Eq.6 (minimal acute angle).
    void associate(DataVault<Ind_t>& vault, int v) {
        double best_cos = -std::numeric_limits<double>::infinity();
        int    best_r   = 0;
        for (int r=0;r<static_cast<int>(W_.size());++r) {
            double c = acute_cos(vault, v, r);
            if (c > best_cos) { best_cos = c; best_r = r; }
        }
        vault.get_ind(v).subregion_idx = best_r;
        vault.get_ind(v).pbi_value     = pbi(vault, v, best_r);
    }

    // ── Dominance — PURE Pareto dominance on objectives ────────────────────
    // Constraints are handled by Alg.7/8 of the paper, not by a CDP scheme
    // (C-MOEA/DD), while the paper keeps NDS unconditional.
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        const auto& fa=vault.objectives_of(a);
        const auto& fb=vault.objectives_of(b);
        bool better=false;
        for(std::size_t i=0;i<fa.size();++i) {
            if(fa[i]>fb[i]) return false; if(fa[i]<fb[i]) better=true;
        }
        return better;
    }

    // ── Fast NDS (initialisation, Alg.2 line 16) ───────────────────────────
    void compute_fronts(DataVault<Ind_t>& vault, int n) {
        std::vector<std::vector<int>> S(n);
        std::vector<int> np(n,0);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
            if(i==j) continue;
            if(dominates(vault,i,j)) S[i].push_back(j);
            else if(dominates(vault,j,i)) ++np[i];
        }
        std::vector<int> f0;
        for(int i=0;i<n;++i) if(np[i]==0) { vault.get_ind(i).rank=0; f0.push_back(i); }
        std::vector<std::vector<int>> fronts; fronts.push_back(f0);
        int k=0;
        while(!fronts[k].empty()) {
            std::vector<int> nxt;
            for(int i:fronts[k]) for(int j:S[i]) if(--np[j]==0) { vault.get_ind(j).rank=k+1; nxt.push_back(j); }
            fronts.push_back(nxt); ++k;
        }
    }

    // ── Initial association (Alg.2 line 17) ────────────────────────────────
    // "each member of P is initially associated with a unique subregion in a
    // random manner" — a random bijection individual ↔ subregion, NOT an
    // association by acute angle (which is what Alg.2 line 17 does not say).
    void assign_subregions_random(DataVault<Ind_t>& vault, int n) {
        int N = static_cast<int>(W_.size());
        std::vector<int> perm(N);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);
        for (int i=0;i<n;++i) {
            int r = perm[i % N];
            vault.get_ind(i).subregion_idx = r;
            vault.get_ind(i).pbi_value     = pbi(vault, i, r);
            vault.get_ind(i).niche_count   = 1;
        }
    }

    // ── Incremental non-domination level update on xc insertion ([66]) ────
    // Pool: [0,n) + xc (scratch). Correctness on the "replace the worst"
    // path confirmed by arbitration (internal arbitration review, item 1).
    void incremental_rank_insert(DataVault<Ind_t>& vault, int n, int xc_slot) {
        int xc_rank = 0;
        for (int i = 0; i < n; ++i) {
            if (dominates(vault, i, xc_slot))
                xc_rank = std::max(xc_rank, vault.get_ind(i).rank + 1);
        }
        vault.get_ind(xc_slot).rank = xc_rank;
        std::vector<int> to_check;
        for (int i = 0; i < n; ++i)
            if (dominates(vault, xc_slot, i) &&
                vault.get_ind(i).rank <= xc_rank)
                to_check.push_back(i);
        while (!to_check.empty()) {
            int v = to_check.back(); to_check.pop_back();
            int new_rank = 0;
            // v's rank = max rank of all its dominators + 1.
            for (int i = 0; i <= n; ++i) {   // n+1 slots: [0,n) + xc
                int slot = (i < n) ? i : xc_slot;
                if (slot == v) continue;
                if (dominates(vault, slot, v))
                    new_rank = std::max(new_rank, vault.get_ind(slot).rank + 1);
            }
            if (new_rank != vault.get_ind(v).rank) {
                vault.get_ind(v).rank = new_rank;
                for (int i = 0; i < n; ++i)
                    if (dominates(vault, v, i))
                        to_check.push_back(i);
            }
        }
    }

    // ── Level update AFTER removal (called after swap_active) ─────────────
    // New pool = active [0,n) (with xc in the victim's place); the removed
    // individual sits in scratch. Ranks of individuals dominated by the
    // removed one may decrease, so the update runs over the
    // up-to-date pool, which already contains xc (previously xc was ignored).
    void rank_remove_update(DataVault<Ind_t>& vault, int n, int removed_slot) {
        std::vector<int> to_check;
        for (int i = 0; i < n; ++i)
            if (dominates(vault, removed_slot, i))
                to_check.push_back(i);
        while (!to_check.empty()) {
            int v = to_check.back(); to_check.pop_back();
            int new_rank = 0;
            for (int i = 0; i < n; ++i) {
                if (i == v) continue;
                if (dominates(vault, i, v))
                    new_rank = std::max(new_rank, vault.get_ind(i).rank + 1);
            }
            if (new_rank < vault.get_ind(v).rank) {
                vault.get_ind(v).rank = new_rank;
                for (int i = 0; i < n; ++i)
                    if (dominates(vault, v, i))
                        to_check.push_back(i);
            }
        }
    }

    // ── Σ g^pbi over the members of subregion r in pool [0,n)+scratch (Eq.7)
    double subregion_pbi_sum(DataVault<Ind_t>& vault, int n, int scratch,
                             int r) {
        double s = 0.0;
        for (int i = 0; i <= n; ++i) {
            int slot = (i < n) ? i : scratch;
            if (vault.get_ind(slot).subregion_idx == r)
                s += pbi(vault, slot, r);
        }
        return s;
    }

    // ── LOCATE_WORST (Alg.5) ───────────────────────────────────────────────
    // 1. Φ^h — the most crowded subregion in P' (tie — Eq.7: max Σ g^pbi);
    // 2. R — members of Φ^h at the worst non-domination level WITHIN Φ^h;
    // 3. x' = argmax g^pbi over R.
    int locate_worst(DataVault<Ind_t>& vault, int n, int scratch,
                     const std::vector<int>& niche) {
        int N = static_cast<int>(W_.size());
        int h = -1; double h_sum = -1.0;
        for (int r = 0; r < N; ++r) {
            if (niche[r] == 0) continue;
            if (h < 0 || niche[r] > niche[h]) { h = r; h_sum = -1.0; }
            else if (niche[r] == niche[h]) {
                if (h_sum < 0.0) h_sum = subregion_pbi_sum(vault, n, scratch, h);
                double s2 = subregion_pbi_sum(vault, n, scratch, r);
                if (s2 > h_sum) { h = r; h_sum = s2; }
            }
        }
        // R: the worst level within Φ^h.
        int worst_rank = -1;
        for (int i = 0; i <= n; ++i) {
            int slot = (i < n) ? i : scratch;
            if (vault.get_ind(slot).subregion_idx == h)
                worst_rank = std::max(worst_rank, vault.get_ind(slot).rank);
        }
        int worst = -1; double worst_pbi = -std::numeric_limits<double>::infinity();
        for (int i = 0; i <= n; ++i) {
            int slot = (i < n) ? i : scratch;
            const auto& ind = vault.get_ind(slot);
            if (ind.subregion_idx != h || ind.rank != worst_rank) continue;
            double p = pbi(vault, slot, h);
            if (p > worst_pbi) { worst_pbi = p; worst = slot; }
        }
        return worst;
    }

    // ── Victim selection by Alg.4 (unconstrained case) ─────────────────────
    int find_victim_unconstrained(DataVault<Ind_t>& vault, int n, int scratch,
                                  const std::vector<int>& niche) {
        // l: number of levels (via the maximal rank in the pool).
        int max_rank = 0;
        for (int i = 0; i <= n; ++i) {
            int slot = (i < n) ? i : scratch;
            max_rank = std::max(max_rank, vault.get_ind(slot).rank);
        }
        if (max_rank == 0)                       // l = 1 (Alg.4 lines 4–6)
            return locate_worst(vault, n, scratch, niche);

        std::vector<int> Fl;                     // the last front
        for (int i = 0; i <= n; ++i) {
            int slot = (i < n) ? i : scratch;
            if (vault.get_ind(slot).rank == max_rank) Fl.push_back(slot);
        }

        if (Fl.size() == 1) {                    // |F_l| = 1 (lines 8–14)
            int xl = Fl[0];
            int sub_l = vault.get_ind(xl).subregion_idx;
            if (niche[sub_l] > 1)
                return xl;                       // line 10: remove x^l
            // Φ^l is isolated → x^l survives, "second chance": remove the
            // worst from the most crowded subregion (lines 12–13). The
            // offspring DOES enter P here — Alg.5 admits it.
            return locate_worst(vault, n, scratch, niche);
        }

        // |F_l| > 1 (lines 15–24): Φ^h — the most crowded among the
        // subregions associated with the solutions of F_l (tie — Eq.7 over
        // the whole Φ).
        std::vector<int> subs;
        for (int slot : Fl) {
            int s = vault.get_ind(slot).subregion_idx;
            if (std::find(subs.begin(), subs.end(), s) == subs.end())
                subs.push_back(s);
        }
        int h = subs[0]; double h_sum = -1.0;
        for (std::size_t k = 1; k < subs.size(); ++k) {
            int r = subs[k];
            if (niche[r] > niche[h]) { h = r; h_sum = -1.0; }
            else if (niche[r] == niche[h]) {
                if (h_sum < 0.0) h_sum = subregion_pbi_sum(vault, n, scratch, h);
                double s2 = subregion_pbi_sum(vault, n, scratch, r);
                if (s2 > h_sum) { h = r; h_sum = s2; }
            }
        }
        if (niche[h] > 1) {
            // lines 18–19, Eq.8: argmax g^pbi over the WHOLE Φ^h (the victim
            // is possibly NOT from the last front).
            int worst = -1;
            double worst_pbi = -std::numeric_limits<double>::infinity();
            for (int i = 0; i <= n; ++i) {
                int slot = (i < n) ? i : scratch;
                if (vault.get_ind(slot).subregion_idx != h) continue;
                double p = pbi(vault, slot, h);
                if (p > worst_pbi) { worst_pbi = p; worst = slot; }
            }
            return worst;
        }
        // lines 21–22: |Φ^h| = 1 — all members of F_l are isolated → LOCATE_WORST.
        return locate_worst(vault, n, scratch, niche);
    }

    // ── Victim selection by Alg.7 (C-MOEA/DD, infeasible present in P') ───
    // Sort the infeasible by CV in descending order; the first one whose
    // subregion is NOT isolated is removed; if all are isolated — S(1)
    // (the maximal CV).
    int find_victim_constrained(DataVault<Ind_t>& vault, int n, int scratch,
                                const std::vector<int>& niche) {
        std::vector<std::pair<double,int>> S;   // (cv, slot), cv > 0
        for (int i = 0; i <= n; ++i) {
            int slot = (i < n) ? i : scratch;
            double cv = vault.get_cv(slot);
            if (cv > 0.0) S.emplace_back(cv, slot);
        }
        std::sort(S.begin(), S.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        for (const auto& [cv, slot] : S) {
            (void)cv;
            if (niche[vault.get_ind(slot).subregion_idx] > 1)
                return slot;                     // lines 14–19
        }
        return S.front().second;                 // lines 20–22: all isolated
    }

    // ── MATING_SELECTION (Alg.3): candidates ──────────────────────────────
    std::vector<int> mating_candidates(DataVault<Ind_t>& vault, int i, int n) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        std::vector<int> cand;
        if (U01(rng_) < delta_) {
            // k random SUBREGION indexes from E(i) — subregion indexes, not
            // individual slot numbers; the two are not interchangeable here.
            std::vector<int> sel = B_[i];
            std::shuffle(sel.begin(), sel.end(), rng_);
            int k = std::min<int>(mating_k_, static_cast<int>(sel.size()));
            sel.resize(k);
            for (int s = 0; s < n; ++s) {
                int sub = vault.get_ind(s).subregion_idx;
                for (int w : sel)
                    if (sub == w) { cand.push_back(s); break; }
            }
        }
        // "if no solution in the selected subregions" (Alg.3 line 3) or the
        // global branch (line 9): random from the whole P.
        if (cand.empty()) {
            cand.resize(n);
            std::iota(cand.begin(), cand.end(), 0);
        }
        return cand;
    }

    // One parent from the candidates; under FEASIBILITY — CV tournament (Alg.8).
    int pick_parent(DataVault<Ind_t>& vault, const std::vector<int>& cand) {
        std::uniform_int_distribution<int> d(0, static_cast<int>(cand.size())-1);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            int a = cand[d(rng_)], b = cand[d(rng_)];
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            if (ca < cb) return a;
            if (cb < ca) return b;
            std::uniform_int_distribution<int> coin(0, 1);
            return coin(rng_) ? a : b;           // RANDOM_PICK (Alg.8 line 6)
        }
        return cand[d(rng_)];
    }

    // ── UPDATE_POPULATION (Alg.4 / Alg.7) for one offspring in scratch ────
    void update_population(DataVault<Ind_t>& vault, int n, int scratch,
                           std::vector<int>& rank_backup) {
        int m = vault.objs_n();

        // Update z* (xc is already evaluated).
        const auto& fo = vault.objectives_of(scratch);
        for (int j=0;j<m;++j) ideal_[j] = std::min(ideal_[j], fo[j]);

        // Line 1: associate xc via Eq.6.
        associate(vault, scratch);

        // Snapshot of P's ranks (for rollback if the offspring gets discarded).
        for (int s = 0; s < n; ++s) rank_backup[s] = vault.get_ind(s).rank;

        // Line 3: incremental level update of P' = P ∪ {xc}.
        incremental_rank_insert(vault, n, scratch);

        // Niche counts over the pool P' (n+1 individuals).
        int N = static_cast<int>(W_.size());
        std::vector<int> niche(N, 0);
        for (int s = 0; s <= n; ++s) {
            int slot = (s < n) ? s : scratch;
            ++niche[vault.get_ind(slot).subregion_idx];
        }
        for (int s = 0; s <= n; ++s) {
            int slot = (s < n) ? s : scratch;
            vault.get_ind(slot).niche_count =
                niche[vault.get_ind(slot).subregion_idx];
        }

        // Victim selection: Alg.7 if infeasible present (FEASIBILITY), else Alg.4.
        bool any_infeas = false;
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            for (int s = 0; s <= n && !any_infeas; ++s) {
                int slot = (s < n) ? s : scratch;
                if (vault.get_cv(slot) > 0.0) any_infeas = true;
            }
        }
        int worst = any_infeas
                  ? find_victim_constrained(vault, n, scratch, niche)
                  : find_victim_unconstrained(vault, n, scratch, niche);

        if (worst == scratch) {
            // The offspring is discarded (P ← P' \ {xc}) → roll back the
            // effects of incremental_rank_insert (arbitration fix #3: stale ranks).
            for (int s = 0; s < n; ++s)
                vault.get_ind(s).rank = rank_backup[s];
        } else {
            // P ← P' \ {x'}: xc takes the victim's position; the victim goes
            // to scratch.
            vault.swap_active(worst, scratch);
            // Line 26: level update of the new P (with xc in the pool).
            rank_remove_update(vault, n, scratch);
        }
    }

public:
    MOEADDCore() = default;

    void set_theta        (double t) { theta_ = t; }
    void set_T            (int t)    { T_     = t; }
    void set_delta        (double d) { delta_ = d; }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_    = p; }
    void set_pm           (double p) { pm_    = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        init_weights(n, m);
        build_neighbourhood(n);
        ideal_.assign(m, std::numeric_limits<double>::max());

        // Alg.2 line 1: random sampling from Ω.
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int>     db(0,1);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bvars(vault.bin_vars_n());
        for (int i=0;i<n;++i) {
            for(int j=0;j<vault.vars_n();++j){double lo=bounds[j].first.value_or(0.0),hi=bounds[j].second.value_or(1.0);vars[j]=lo+dr(rng_)*(hi-lo);}
            for(int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars); else vault.set_variables(i,vars);
        }
        vault.sync();
        for(int i=0;i<n;++i){const auto& o=vault.objectives_of(i);for(int j=0;j<m;++j) ideal_[j]=std::min(ideal_[j],o[j]);}
        // Alg.2 line 16: NDS; line 17: random bijection individual ↔ subregion.
        compute_fronts(vault, n);
        assign_subregions_random(vault, n);
        vault.expand(1); vault.reduce(n+1);   // scratch slot at index n
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(), m=vault.objs_n();
        if(W_.empty()){init_weights(n,m);build_neighbourhood(n);}
        ideal_.assign(m,std::numeric_limits<double>::max());
        for(int i=0;i<n;++i){const auto& o=vault.objectives_of(i);for(int j=0;j<m;++j) ideal_[j]=std::min(ideal_[j],o[j]);}
        compute_fronts(vault,n);
        assign_subregions_random(vault,n);
        vault.expand(1); vault.reduce(n+1);
    }

    // ── step: one generation (Alg.1 lines 3–9, sequentially i=1..N) ───────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int scratch = n;   // scratch slot at active index n
        const auto& bounds = vault.get_bounds();

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        std::vector<int> rank_backup(n);

        for (int i = 0; i < n; ++i) {
            // Alg.1 line 4: MATING_SELECTION (Alg.3 / Alg.8).
            auto cand = mating_candidates(vault, i, n);
            int pa = pick_parent(vault, cand);
            int pb = pick_parent(vault, cand);
            for (int t = 0; t < 10 && pb == pa && cand.size() > 1; ++t)
                pb = pick_parent(vault, cand);

            // Alg.1 line 5: VARIATION — SBX(p_c, η_c) + PM(p_m, η_m);
            // one offspring c1 is used (|S| = 1, see the file header).
            for(int j=0;j<vault.vars_n();++j){pv1[j]=vault.get_variable(pa,j);pv2[j]=vault.get_variable(pb,j);}
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_,
                                     pm_eff(vault.vars_n()), rng_);
            if(vault.bin_vars_n()>0){
                std::vector<int> bv1(vault.bin_vars_n()),bv2(vault.bin_vars_n()),bc1,bc2;
                for(int j=0;j<vault.bin_vars_n();++j){bv1[j]=vault.get_bin_variable(pa,j);bv2[j]=vault.get_bin_variable(pb,j);}
                ops::binary_crossover(bv1,bv2,bc1,bc2,rng_);
                ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
                vault.set_all_variables(scratch,c1,bc1);
            } else vault.set_variables(scratch,c1);
            // Exactly ONE objective function call per offspring (FE/generation = N).
            vault.refresh_objectives(scratch);

            // Alg.1 line 7: UPDATE_POPULATION (Alg.4 / Alg.7).
            update_population(vault, n, scratch, rank_backup);
        }
    }
};

} // namespace mootation

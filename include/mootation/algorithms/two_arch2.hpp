#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Two_Arch2 — An Improved Two-Archive Algorithm for Many-Objective Optimization
// Handing Wang, Licheng Jiao, Xin Yao — IEEE TEVC 19(4):524-541, 2015
// doi:10.1109/TEVC.2014.2350987
//
//
// MECHANISM (Section III, Fig. 2). Two fixed-size archives kept in parallel:
//   CA (Convergence Archive) — selected by the I_epsilon+ indicator (IBEA-style).
//   DA (Diversity Archive)   — Pareto-based; the FINAL OUTPUT (Sec III-A).
// Per generation (Fig. 2):
//   1. Reproduction: crossover BETWEEN CA and DA (one parent from each) AND
//      mutation on CA — as TWO INDEPENDENT operators (Sec III-A: "crossover and
//      mutation are independent in Two_Arch2"; Sec IV-E.1/E.2). The n_DA
//      offspring split into a crossover stream (CA×DA children, no mutation) and
//      a mutation stream (pure mutants of CA members, no crossover).
//   2. Update CA by I_epsilon+ (Alg. 2): add offspring to CA, then while
//      |CA|>n_CA delete the member with minimal fitness loss F and update the
//      rest by F(x)+=e^{-I_eps+(x*,x)/0.05} (Eq. 2, Alg. 2).
//   3. Update DA by Pareto dominance (Sec III-C.1, Alg. 3): keep only
//      nondominated; if |DA|>n_DA, run the diversity SELECTION (not deletion):
//      take boundary solutions first, then iteratively add the solution most
//      DIFFERENT (max of min L_p-distance) to those already selected.
//   The L_p (Minkowski) distance uses p = 1/m, m = #objectives (Sec III-C.2):
//      d(a,b) = ( sum_i |a_i - b_i|^p )^{1/p},  p = 1/m  (fractional norm).
//   Final result returned to the vault = DA (Sec III-A).
//
// I_epsilon+ (Eq. 1): I_eps+(x1,x2) = min over eps s.t. f_i(x1)-eps <= f_i(x2)
//   for all i  ==>  I_eps+(x1,x2) = max_i ( f_i(x1) - f_i(x2) ).
// Fitness (Eq. 2):  F(x1) = sum_{x2 != x1} -e^{ -I_eps+(x2,x1)/kappa },
//   kappa = 0.05 (paper). Larger F = better (less negative); the worst (most
//   negative, smallest F) is removed first in Alg. 2.
//
// DEFAULTS: n_CA = 100 (Sec IV-C); n_DA = pop_size (final output size,
//   Sec IV-A/IV-C); kappa = 0.05 (Eq. 2); p = 1/m (Sec III-C.2);
//   SBX eta_c=20, pc=1.0; poly mutation eta_m=20, pm=1/n. (Paper uses
//   eta=15 for both operators; see deviation TA2-5 — we keep the library
//   defaults of 20 to match the rest of MOOtation and the harness contract.)
//
// DECLARED DEVIATIONS:
//   TA2-1 (MINOR). I_eps+ computed on RAW objectives (no per-axis [0,1]
//     normalization). IBEA normalizes objectives before I_eps+; the Two_Arch2
//     paper does not explicitly state normalization for CA. On the unit-scaled
//     DTLZ/WFG fronts used here this is immaterial; declared for honesty.
//   TA2-2 (MINOR). DA "boundary solutions with maximal or minimal objective
//     values" (Alg. 3 line 3) implemented as, per objective i, the argmin and
//     argmax of f_i (up to 2m points, de-duplicated). If that already fills DA
//     it is truncated; the paper does not specify a tie/overflow rule here.
//   TA2-3 (MINOR). If after Pareto filtering |DA| <= n_DA, DA is kept as-is
//     (no padding) — DA may temporarily hold fewer than n_DA solutions early on,
//     exactly as Alg. 3 implies (selection only triggers on overflow).
//     Objective-identical solutions are also collapsed to one before the Alg. 3
//     selection, which is a second way |DA| can fall below n_DA. The paper
//     never mentions duplicates; the collapse is provably neutral for the
//     SELECTED set — L_p distance is non-negative, so a duplicate of an
//     already-selected point has similarity exactly 0 and Alg. 3's argmax can
//     never pick it — but it does change how many solutions DA ends up holding.
//   TA2-4 (INTENTIONAL). Reproduction realizes crossover and mutation as TWO
//     INDEPENDENT operators, per Sec III-A ("crossover and mutation are
//     independent in Two_Arch2") and Sec IV-E (variation experiments treat
//     mutation-on-archive and crossover-parent-selection separately). The n_DA
//     offspring are split: (a) crossover children from CA×DA pairs (first SBX
//     child, no mutation); (b) pure polynomial mutants of CA members (no
//     crossover). Stream sizes are n_DA/2 and n_DA-n_DA/2 (the paper fixes only
//     pop/FE budget, not a stream ratio; ~50/50 preserves the offspring count
//     n_DA and the FE contract). An earlier version chained the operators (the
//     SBX child of CA x DA was always mutated), contradicting the operator
//     independence of §III-A; it is now two independent streams.
//   TA2-5 (MINOR). Operator distribution indices eta_c=eta_m=20 (library
//     default) instead of the paper's 15. Peripheral; tunable via setters.
//   TA2-6 (MINOR). Real genome only; binary is out of scope.
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP switches BOTH archives to Deb's feasibility rules:
//   the DA Pareto filter uses constrained domination, and the CA indicator
//   fitness gets an additive penalty so infeasible members are the first to be
//   truncated (worst CV first). handingwang2015 studies unconstrained problems
//   only, so this is an extension, not a transcription.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class Two_Arch2Core {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    Two_Arch2Core() = default;

    void set_seed(unsigned s)        { rng_.seed(s); }
    void set_ca_size(int n)          { n_ca_req_ = n; }   // n_CA (default 100)
    void set_da_size(int n)          { n_da_req_ = n; }   // n_DA (default pop_size)
    void set_kappa(double k)         { kappa_ = k; }      // Eq. 2 (default 0.05)
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_pm(double p)            { pm_ = p; }

private:
    int    n_ca_req_ = 100;     // requested CA capacity (Sec IV-C)
    int    n_da_req_ = -1;      // requested DA capacity; <0 => pop_size
    double kappa_    = 0.05;    // Eq. 2 scaling
    double eta_c_    = 20.0;
    double eta_m_    = 20.0;
    double pc_       = 1.0;
    double pm_       = -1.0;    // <0 => 1/n
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; double cv = 0.0; };

    std::vector<Sol> CA_;       // Convergence Archive (I_eps+)
    std::vector<Sol> DA_;       // Diversity Archive (Pareto) — final output
    int m_ = 0;                 // #objectives
    int n_ca_ = 0, n_da_ = 0;   // effective capacities

    double pm_eff(int nv) const { return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0); }

    // ── Domination on raw objectives (minimization). Under FEASIBILITY/CDP
    //    this is Deb's constrained domination; with NONE it is plain Pareto. ──
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }
    bool dominates(const std::vector<double>& a, const std::vector<double>& b) const {
        return detail::pareto_dominates(a, b);
    }

    // ── I_eps+ (Eq. 1): max_i ( f_i(x1) - f_i(x2) ) ─────────────────────────
    double i_eps_plus(const std::vector<double>& x1,
                      const std::vector<double>& x2) const {
        double v = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < m_; ++i) v = std::max(v, x1[i] - x2[i]);
        return v;
    }

    // ── Minkowski L_p distance, p = 1/m (Sec III-C.2). Fractional norm. ──────
    double lp_dist(const std::vector<double>& a, const std::vector<double>& b) const {
        double p = (m_ > 0) ? 1.0 / static_cast<double>(m_) : 1.0;
        double s = 0.0;
        for (int i = 0; i < m_; ++i) s += std::pow(std::abs(a[i] - b[i]), p);
        return std::pow(s, 1.0 / p);
    }

    // ── CA update by I_eps+ (Eq. 2 fitness, Alg. 2 truncation) ──────────────
    // Offspring already merged into CA_; trim down to n_ca_.
    void update_ca() {
        int sz = static_cast<int>(CA_.size());
        if (sz <= n_ca_) return;

        // F(x1) = sum_{x2 != x1} -e^{ -I_eps+(x2,x1)/kappa }   (Eq. 2)
        // Under FEASIBILITY/CDP an infeasible member is pushed below every
        // feasible one by an additive penalty on F (larger F is better here),
        // so Alg. 2 removes infeasible members first, worst-CV first. The
        // indicator itself is left untouched.
        std::vector<double> F(sz, 0.0);
        std::vector<char> alive(sz, 1);
        for (int i = 0; i < sz; ++i)
            for (int j = 0; j < sz; ++j) {
                if (i == j) continue;
                F[i] += -std::exp(-i_eps_plus(CA_[j].objs, CA_[i].objs) / kappa_);
            }
        if (constraint_mode != ConstraintMode::NONE)
            for (int i = 0; i < sz; ++i)
                if (CA_[i].cv > 0.0) F[i] -= 1e12 * (1.0 + CA_[i].cv);

        int live = sz;
        while (live > n_ca_) {
            // Alg. 2 line 3: find x* with minimal F (worst convergence value).
            int worst = -1; double wv = std::numeric_limits<double>::infinity();
            for (int i = 0; i < sz; ++i)
                if (alive[i] && F[i] < wv) { wv = F[i]; worst = i; }
            if (worst < 0) break;
            // Alg. 2 line 5: update remaining F(x) += e^{ -I_eps+(x*,x)/kappa }.
            for (int i = 0; i < sz; ++i)
                if (alive[i] && i != worst)
                    F[i] += std::exp(-i_eps_plus(CA_[worst].objs, CA_[i].objs) / kappa_);
            alive[worst] = 0;
            --live;
        }
        std::vector<Sol> kept; kept.reserve(n_ca_);
        for (int i = 0; i < sz; ++i) if (alive[i]) kept.push_back(CA_[i]);
        CA_.swap(kept);
    }

    // ── DA update by Pareto + L_p diversity SELECTION (Alg. 3) ───────────────
    // pool = current DA_ + nondominated offspring; keep nondominated, then if
    // overflow run the selection of Alg. 3.
    void update_da(const std::vector<Sol>& candidates) {
        // Merge.
        std::vector<Sol> pool = DA_;
        pool.insert(pool.end(), candidates.begin(), candidates.end());

        // Keep only nondominated within the pool (Sec III-C.1: only
        // nondominated solutions can be added to DA).
        std::vector<Sol> nd;
        int P = static_cast<int>(pool.size());
        std::vector<char> dom(P, 0);
        for (int i = 0; i < P; ++i) {
            if (dom[i]) continue;
            for (int j = 0; j < P; ++j) {
                if (i == j) continue;
                if (dominates(pool[j], pool[i])) { dom[i] = 1; break; }
                // duplicate guard: identical objectives -> keep first only
                if (j < i && pool[j].objs == pool[i].objs) { dom[i] = 1; break; }
            }
        }
        for (int i = 0; i < P; ++i) if (!dom[i]) nd.push_back(pool[i]);

        if (static_cast<int>(nd.size()) <= n_da_) {   // TA2-3: no overflow
            DA_.swap(nd);
            return;
        }

        // Alg. 3: selection in overflowed DA.
        int K = static_cast<int>(nd.size());
        std::vector<char> taken(K, 0);
        std::vector<int> selected;
        selected.reserve(n_da_);

        // line 3: boundary solutions (per objective: min and max). TA2-2.
        for (int d = 0; d < m_ && static_cast<int>(selected.size()) < n_da_; ++d) {
            int imin = 0, imax = 0;
            for (int i = 1; i < K; ++i) {
                if (nd[i].objs[d] < nd[imin].objs[d]) imin = i;
                if (nd[i].objs[d] > nd[imax].objs[d]) imax = i;
            }
            for (int b : {imin, imax}) {
                if (!taken[b] && static_cast<int>(selected.size()) < n_da_) {
                    taken[b] = 1; selected.push_back(b);
                }
            }
        }

        // Similarity[i] = min L_p distance from i to any selected solution.
        std::vector<double> sim(K, std::numeric_limits<double>::infinity());
        for (int i = 0; i < K; ++i) {
            if (taken[i]) { sim[i] = -1.0; continue; }
            for (int s : selected)
                sim[i] = std::min(sim[i], lp_dist(nd[i].objs, nd[s].objs));
        }

        // line 4-9: iteratively add the most DIFFERENT (max Similarity).
        while (static_cast<int>(selected.size()) < n_da_) {
            int best = -1; double bv = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < K; ++i)
                if (!taken[i] && sim[i] > bv) { bv = sim[i]; best = i; }
            if (best < 0) break;
            taken[best] = 1; selected.push_back(best);
            // refresh Similarity against the newly added solution.
            for (int i = 0; i < K; ++i)
                if (!taken[i])
                    sim[i] = std::min(sim[i], lp_dist(nd[i].objs, nd[best].objs));
        }

        std::vector<Sol> out; out.reserve(selected.size());
        for (int s : selected) out.push_back(nd[s]);
        DA_.swap(out);
    }

    // Evaluate a variable vector into a Sol via the scratch slot.
    Sol eval(DataVault<Ind_t>& vault, int scratch, std::vector<double> vars) {
        Sol s; s.vars = std::move(vars);
        vault.set_variables(scratch, s.vars);
        vault.refresh_objectives(scratch);
        s.objs = vault.objectives_of(scratch);
        if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(scratch);
        return s;
    }

    // ── Reproduction (Sec III-A / Fig. 2 / Sec IV-E): crossover between CA and
    //    DA, and mutation on CA — as TWO INDEPENDENT operators. Returns n_da_
    //    offspring split into a crossover stream and a mutation stream. ────────
    // The paper: "Two_Arch2 makes crossover between CA and DA but mutation on CA
    //   only during the process of reproduction (crossover and mutation are
    //   INDEPENDENT in Two_Arch2)". Chaining the operators — SBX on a CA x DA
    //   pair followed by ALWAYS mutating the child — contradicts that
    //   independence and §IV-E, whose experiments ("mutation on CA/DA/union"
    //   and "crossover CA x DA / union / CA / DA") only make sense for
    //   standalone operators. Split into two streams: (a) crossover children
    //   from CA x DA, unmutated; (b) pure mutants of CA members, uncrossed.
    std::vector<Sol> reproduce(DataVault<Ind_t>& vault, int scratch) {
        const auto& b = vault.get_bounds();
        int nv = vault.vars_n();
        std::vector<Sol> off;
        off.reserve(n_da_);

        // Guard: if an archive is empty, fall back to the non-empty one.
        const std::vector<Sol>& A = !CA_.empty() ? CA_ : DA_;  // CA side
        const std::vector<Sol>& B = !DA_.empty() ? DA_ : CA_;  // DA side
        if (A.empty() || B.empty()) return off;

        std::uniform_int_distribution<int> dA(0, static_cast<int>(A.size()) - 1);
        std::uniform_int_distribution<int> dB(0, static_cast<int>(B.size()) - 1);

        // Split the n_DA offspring budget between the two independent streams
        // (article gives pop/FE budget but no stream ratio; ~50/50 keeps the
        // total offspring count = n_DA and the FE contract intact — TA2-4).
        int n_cross = n_da_ / 2;
        int n_mut   = n_da_ - n_cross;

        // (a) Crossover stream: one parent from CA, one from DA; SBX; NO mutation.
        for (int k = 0; k < n_cross; ++k) {
            const Sol& pca = A[dA(rng_)];   // parent from CA (Sec IV-E.2)
            const Sol& pda = B[dB(rng_)];   // parent from DA
            std::vector<double> c1, c2;
            ops::sbx(pca.vars, pda.vars, c1, c2, b, eta_c_, pc_, rng_);
            off.push_back(eval(vault, scratch, std::move(c1)));
        }

        // (b) Mutation stream: pure polynomial mutants of CA members; NO crossover.
        for (int k = 0; k < n_mut; ++k) {
            std::vector<double> mv = A[dA(rng_)].vars;   // member of CA
            ops::polynomial_mutation(mv, b, eta_m_, pm_eff(nv), rng_);
            off.push_back(eval(vault, scratch, std::move(mv)));
        }
        return off;
    }

    // Build CA/DA from an initial population.
    void seed_archives(const std::vector<Sol>& P) {
        // CA: all individuals merged, trimmed by I_eps+.
        CA_ = P;
        update_ca();
        // DA: nondominated + diversity selection from the same pool.
        DA_.clear();
        update_da(P);
    }

    void write_back(DataVault<Ind_t>& vault) {
        // Final output = DA (Sec III-A).
        const std::vector<Sol>& out = !DA_.empty() ? DA_ : CA_;
        vault.reduce(0);
        vault.expand(static_cast<int>(out.size()));
        for (int i = 0; i < static_cast<int>(out.size()); ++i)
            vault.seed_individual(static_cast<std::size_t>(i),
                                  out[i].vars, out[i].objs, {}, {});
    }

public:
    void setup(DataVault<Ind_t>& vault) {
        m_    = vault.objs_n();
        n_ca_ = (n_ca_req_ > 0) ? n_ca_req_ : 100;
        n_da_ = (n_da_req_ > 0) ? n_da_req_ : vault.pop_size();

        int N = vault.pop_size();
        const auto& bd = vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0, 1.0);
        std::vector<double> vars(vault.vars_n());
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bd[j].first.value_or(0.0), hi = bd[j].second.value_or(1.0);
                vars[j] = lo + d(rng_) * (hi - lo);
            }
            vault.set_variables(i, vars);
        }
        vault.sync();

        std::vector<Sol> P;
        P.reserve(N);
        for (int i = 0; i < N; ++i) {
            Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
            if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
            P.push_back(std::move(s));
        }
        seed_archives(P);
        write_back(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        m_    = vault.objs_n();
        n_ca_ = (n_ca_req_ > 0) ? n_ca_req_ : 100;
        n_da_ = (n_da_req_ > 0) ? n_da_req_ : vault.pop_size();

        std::vector<Sol> P;
        int na = static_cast<int>(vault.active_n());
        P.reserve(na);
        for (int i = 0; i < na; ++i) {
            Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
            if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
            P.push_back(std::move(s));
        }
        seed_archives(P);
        write_back(vault);
    }

    // One generation (Fig. 2): reproduce -> update CA (I_eps+) -> update DA.
    void step(DataVault<Ind_t>& vault) {
        int scratch = vault.expand(1);

        // 1. Reproduction: independent crossover (CA×DA) + mutation (CA) streams.
        std::vector<Sol> off = reproduce(vault, scratch);

        // 2. Update CA by I_eps+ (Alg. 2): merge offspring into CA, then trim.
        CA_.insert(CA_.end(), off.begin(), off.end());
        update_ca();

        // 3. Update DA by Pareto dominance + L_p selection (Alg. 3).
        update_da(off);

        // Final result population = DA.
        write_back(vault);
    }
};

} // namespace mootation

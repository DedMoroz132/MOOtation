#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/D-M2M — Decomposition of an MOP into a number of simple multiobjective
// subproblems.
// H.-L. Liu, F. Gu, Q. Zhang — IEEE Transactions on Evolutionary Computation
// 18(3), 2014.
// doi:10.1109/TEVC.2013.2281533            (source: liu2014)
//
// IDEA (§II-A). K unit direction vectors v¹..v^K in R₊ᵐ are chosen.
// The space is divided into K subregions (Eq.2):
//   Ω_k = { u ∈ R₊ᵐ : <u,v^k> ≤ <u,v^j> ∀ j },
// where <u,v> is the acute angle between the vectors. That is, a point u
// belongs to Ω_k if v^k is the closest to u by angle among all K directions.
// Each subregion = a simple multiobjective subproblem (Eq.3), solved jointly.
//
// SCHEME (Algorithm 1):
//   1.  Init: K×S random points from [a,b]ⁿ, evaluate F, distribute over
//       P_1..P_K via the Allocation algorithm (Algorithm 2).
//   2.  while not stop:
//   4.    R = ∅
//   5–11. for k=1..K: foreach x∈P_k: pick a random y∈P_k; z=GA(x,y);
//         F(z); R ∪= {z}.          (exactly K·S = N offspring per generation)
//   12.   Q = R ∪ (∪ P_k)          (union of 2N solutions)
//   13.   Allocation(Q) → P_1..P_K (Algorithm 2)
//   15.   output the non-dominated solutions of ∪P_k.
//
// ALLOCATION (Algorithm 2), for each k:
//   • P_k := solutions of Q whose F lies in Ω_k (v^k is the closest by angle);
//   • |P_k| < S → ADD S−|P_k| RANDOM solutions from Q (line 4);
//   • |P_k| > S → non-dominated sorting [4] (NSGA-II), REMOVE the
//     (|P_k|−S) worst by rank (line 6-7).
//   Guarantees exactly S solutions in each P_k.
//
// DEFAULTS (= experimental section §III-A):
//   • Operators — Liu & Li 2009 [20] (§III-A(1): «crossover and mutation
//     operators with the same control parameters in [20]»): the annealed
//     arithmetic crossover Eq.(5) z = x + rc·(x − y) and the mutation Eq.(6)
//     with the "≥1 component" guarantee; the rc/rm step decays with gen/Max_gen
//     (operators/liuli_crossover.hpp, arbitration notes LL-1..LL-5).
//     P_m=1/n ([20] §V).
//     FIX 2026-07-07 (source-fidelity review): previously SBX/PM stood here,
//     FALSELY attributed to source [20] — there is no SBX/PM in liu2009.
//   • The annealing needs Max_gen: §III-A(4) of the paper — 3000 generations.
//     In the library — set_t_max(int), default 1000 (moead_awa/adaw convention);
//     the caller MUST set the real budget, otherwise the annealing schedule does
//     not match the paper, and for gen>t_max the operator degenerates into
//     copying.
//   • 2-obj.: K=S=10  →  N=100.   3-obj.: K=S=17  →  N=289.
//   • Direction vectors — "uniformly from the unit sphere in the first octant".
//   • Stopping — by generation count (outer loop).
//
// RELATION TO THE LIBRARY'S pop_size. In the paper N = K·S is the total
//   population size. K (the number of subregions) is set via set_K();
//   S = pop_size / K (must divide evenly, otherwise an exception). Direction
//   vectors: if K is attainable by the lattice — the Das–Dennis lattice on the
//   simplex, normalized to unit length; otherwise — a deterministic generator of
//   exactly K uniform directions (see M2M-2 / FIX 2026-07-08). The actual K =
//   the requested set_K for ANY K≥1.
//
// DECLARED DEVIATIONS:
//   M2M-1 (DEVIATION, paper text vs reference code). Algorithm 2 line 4:
//     when solutions are lacking, the subregion is refilled with RANDOM ones
//     from Q (the letter of the paper). The authors' reference implementation in
//     PlatEMO instead refills with the solutions having the smallest angle to
//     v^k. Here we follow the LETTER of the paper (random refill);
//     this is also ablation point A4/A5 ("hard quotas + refill").
//   M2M-2 (MINOR). Direction vectors: "uniformly from the unit sphere" —
//     for K attainable by the lattice, implemented as the Das–Dennis simplex
//     lattice normalized to unit length (the canonical way; the reference code
//     does the same).
//     FIX 2026-07-08: for K unattainable by the lattice —
//     a deterministic generator of exactly K uniform directions
//     (uniform_sphere_directions: candidates from a fine lattice → FPS
//     by angle → Riesz s-energy minimization, s=2, 60 iterations; clamp into the
//     first octant). Mechanics ported from sms_m2m.hpp (the reference
//     implementation; SMSM2M-5 there).
//     Removes a previously undeclared consequence: the paper's canonical
//     3-objective setup K=S=17 (m=3, N=289) was not reproducible — the lattice
//     yielded 21, and pop_size 289 (=17·17) is not divisible by 21 →
//     std::invalid_argument. Now an arbitrary K/S instantiates without an
//     exception (for m=2 K=S=10, N=100 — as before, bit-for-bit via the
//     lattice). NOTE: the Liu–Li operator (§III-A, FIX waves 1/2) is untouched —
//     only the direction generator was added.
//   M2M-3 (MINOR). For |P_k|>S the paper's non-dominated sorting [4] does not
//     specify the tie-break within the boundary front. NSGA-II crowding
//     distance is used (the paper cites [4] precisely for the sorting).
//   M2M-4 (MINOR). Before the angles are computed, the objective vectors are
//     translated by subtracting the coordinate-wise minimum of the pool
//     (origin-shift). The paper §II-A assumes f≥0 and the footnote "shift
//     f_i+M"; the reference code subtracts the min.
//   M2M-5 (MINOR). y∈P_k is chosen randomly ≠ x (up to 5 attempts to separate);
//     the paper (line 7) allows y=x. Consistent with the moead.hpp style.
//   M2M-6 (retired, FIX 2026-07-07). Previously: "SBX yields two children — the
//     first is taken". Operator [20] produces exactly one offspring — line 8
//     («generate a new solution z», singular) is now followed literally.
//   M2M-D (DIAGNOSTIC, added 2026-08-06 — not a deviation, context for the
//     convergence suite). This port reports mean 1.02 / best 0.010 on DTLZ2
//     (M=3, pop 90, 200 generations), which is why it carries a known_issue
//     marker. Two things narrow it:
//     (a) NOT an output-filtering artefact. Alg.1 line 15 says "find all the
//         nondominated solutions in ∪P_k and output them", and the Output is
//         declared as "Ψ: a set of nondominated solutions", while this port
//         presents the whole ∪P_k. Filtering to the non-dominated subset was
//         the obvious suspect; measured, 71 of the 90 are already mutually
//         non-dominated and the filtered mean is 1.085 — WORSE, not better.
//         Hypothesis excluded.
//     (b) DTLZ2 is not this paper's problem. §III-C states the test instances
//         are "modified ZDT and DTLZ instances ... g(x) functions used in our
//         modified instances are DIFFERENT from those in their original
//         versions", chosen so that "both MOEA/D-DE and NSGA-II cannot locate
//         the global PF on any instance". M2M's per-subregion quota is a
//         diversity device for exactly that regime; plain DTLZ2 is the opposite
//         case, where conventional MOEAs already do well. On DTLZ2 the reported
//         error equals g = Σ(x_i−0.5)², so mean 1.02 says the distance
//         variables sit near random (uniform x gives E[g] ≈ 0.83).
//         A fair check means running the paper's own MOP1-MOP7.
//   M2M-7 (MINOR). set_eta_crossover/set_eta_mutation/set_pc are no-op shims
//     for API uniformity (operator [20] has no η/p_c; the crossover is
//     unconditional).
//   M2M-8 (CONSTRAINT, not a deviation — documented for callers).
//     The M2M decomposition partitions the population into K subpopulations of
//     equal size S, so pop_size MUST be an exact multiple of K:
//         pop_size = K * S,  S >= 2
//     Any other pop_size throws std::invalid_argument from setup(). This is
//     structural, not an implementation shortcut: §III-A defines the scheme in
//     terms of K equally sized subpopulations, and unequal S would silently
//     change the per-subregion selection pressure.
//     K itself is unconstrained since the 2026-07-08 fix above (M2M-2) — it no
//     longer has to be a Das–Dennis lattice cardinality. Only the divisibility
//     of pop_size by K remains.
//     Practical consequence: a benchmark that fixes one pop_size across all
//     algorithms (e.g. 91 = the m=3, H=12 lattice count used by the NSGA-III
//     family) cannot feed that value to M2M unchanged — 91 is not divisible by
//     the default K=10. Either pick pop_size = K*S, or call set_K() with a
//     divisor of pop_size.
//     Deliberately NOT auto-corrected: silently rounding pop_size to K*⌊N/K⌋
//     would change the function-evaluation budget, which is exactly the
//     quantity a benchmark holds fixed when comparing algorithms. Failing
//     loudly is the honest behaviour. Revisit only together with a warning
//     channel and a getter for the effective population size.
//
// EXTENSIONS BEYOND THE PAPER (disabled by default):
//   • Binary/mixed genome: UNIFORM crossover + bit-flip (as in moead.hpp —
//     general-purpose, not the one-point operator of MOEA/D's own MOKP study;
//     active only when bin_vars_n() > 0).
//   • constraint_mode FEASIBILITY/CDP (M2M-C): the 2014 paper is unconstrained,
//     so this is an extension. It makes the per-subregion NSGA-II truncation
//     (Alg.1 step for |P_k| > S) use Deb's constrained domination, which is the
//     only preference relation M2M has — the angular partition and the random
//     shortage refill are geometry and stay as they are. Consequence worth
//     knowing: a subregion whose members are all infeasible still keeps S of
//     them (the quota is unconditional), ranked by CV.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../detail/sphere_directions.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
// FIX 2026-07-07 (source-fidelity review): SBX/PM replaced with the
// Liu & Li 2009 [20] operator Eq.(5)-(6), as required by §III-A(1) of the paper.
#include "../operators/liuli_crossover.hpp"

namespace mootation {

template <typename Ind_t>
class MOEADM2MCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;  // see header: beyond the paper

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    int    K_req_  = 10;     // requested number of subregions (§III-A: K)
    double pm_     = -1.0;   // [20] §V: P_m=1/n; <0 → auto 1/n_vars
    // FIX 2026-07-07: Max_gen for the annealing of operator [20] (§III-A(4):
    // 3000 generations); default 1000 is the library convention, see
    // header/set_t_max.
    int    t_max_  = 1000;
    std::mt19937 rng_{std::random_device{}()};

    // ── runtime state ──────────────────────────────────────────────────────
    std::vector<std::vector<double>> V_;   // K unit direction vectors [K][m]
    int K_  = 0;                           // actual number of subregions
    int S_  = 0;                           // subpopulation size = N / K
    int gen_ = 0;                          // generation counter (for the annealing of [20])

    double pm_eff(int nv) const {
        return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);
    }

    // Acute angle between u and v (both taken in the translated space).
    static double acute_angle(const std::vector<double>& u,
                              const std::vector<double>& v) {
        double dot = 0.0, nu = 0.0, nv = 0.0;
        for (std::size_t k = 0; k < u.size(); ++k) {
            dot += u[k] * v[k];
            nu  += u[k] * u[k];
            nv  += v[k] * v[k];
        }
        double denom = std::sqrt(nu) * std::sqrt(nv);
        if (denom < 1e-300) return 0.0;     // degenerate u (at the origin) → angle 0
        double c = dot / denom;
        if (c >  1.0) c =  1.0;
        if (c < -1.0) c = -1.0;
        return std::acos(c);
    }

    // ── Exactly K uniform directions on the first-octant unit sphere ─────────
    // FIX 2026-07-08: arbitrary K for values unattainable by
    // the lattice (the paper's canonical setup K=S=17, m=3, N=289 — the
    // Das–Dennis lattice yielded 21 → pop_size 289 was rejected with an
    // exception, i.e. the 3-objective run was not reproducible). The mechanics
    // are ported VERBATIM from sms_m2m.hpp (the reference implementation; the
    // issue is already solved and declared there as SMSM2M-5): deterministic
    // (no RNG used) — candidates from the smallest Das–Dennis lattice with ≥2K
    // points on the sphere; FPS by angle starting from candidate 0; then 60
    // iterations of Riesz s-energy repulsion (s=2, step 0.05, clamp into the
    // first octant + renormalization) — the scheme is the deterministic Riesz
    // s-energy (C2 Energy) one.
    // ── Direction vectors: Das–Dennis simplex → normalization onto the sphere ─
    // FIX 2026-07-08: if K is attainable by the lattice —
    //   the previous generate_auto path (bit-for-bit the old behavior);
    //   otherwise — the deterministic arbitrary-K generator
    //   uniform_sphere_directions (as in sms_m2m.hpp). Removes the exception for
    //   the canonical 3-objective setup K=S=17, N=289.
    void build_directions(int m) {
        auto W = das_dennis::generate_auto(m, K_req_);
        if (static_cast<int>(W.size()) != K_req_)
            W = detail::uniform_sphere_directions(m, K_req_);     // arbitrary K (FIX)
        V_.clear();
        V_.reserve(W.size());
        for (auto& w : W) {
            double n2 = 0.0;
            for (double wi : w) n2 += wi * wi;
            double nn = std::sqrt(std::max(n2, 1e-300));
            for (double& wi : w) wi /= nn;
            V_.push_back(std::move(w));
        }
        K_ = static_cast<int>(V_.size());
    }

    // ── NSGA-II selection of the best S via non-dominated sorting + CD ───────
    // objs — objective vectors of the candidates (size c); returns the local
    // indices (in [0,c)) of exactly S solutions that remain. Requires c > S.
    std::vector<int> select_best_S(const std::vector<std::vector<double>>& objs,
                                   const std::vector<double>& cvs,
                                   int S) const {
        int c = static_cast<int>(objs.size());
        int m = c ? static_cast<int>(objs[0].size()) : 0;

        // 1) fast non-dominated sorting
        std::vector<std::vector<int>> dominated(c);
        std::vector<int> dom_count(c, 0);
        // Constrained domination when constraint_mode is on (M2M-C).
        auto dominates = [&](int a, int b) {
            return detail::dominates(constraint_mode, objs[a], cvs[a],
                                                      objs[b], cvs[b]);
        };
        (void)m;
        std::vector<std::vector<int>> fronts;
        std::vector<int> first;
        for (int p = 0; p < c; ++p) {
            for (int q = 0; q < c; ++q) {
                if (p == q) continue;
                if (dominates(p, q))      dominated[p].push_back(q);
                else if (dominates(q, p)) ++dom_count[p];
            }
            if (dom_count[p] == 0) first.push_back(p);
        }
        fronts.push_back(first);
        while (!fronts.back().empty()) {
            std::vector<int> next;
            for (int p : fronts.back()) {
                for (int q : dominated[p]) {
                    if (--dom_count[q] == 0) next.push_back(q);
                }
            }
            if (next.empty()) break;
            fronts.push_back(std::move(next));
        }

        // 2) take whole fronts while they fit
        std::vector<int> kept;
        kept.reserve(S);
        std::size_t fi = 0;
        for (; fi < fronts.size(); ++fi) {
            if (static_cast<int>(kept.size() + fronts[fi].size()) > S) break;
            for (int p : fronts[fi]) kept.push_back(p);
            if (static_cast<int>(kept.size()) == S) return kept;
        }
        // 3) boundary front — fill up by crowding distance (descending)
        if (fi < fronts.size() && static_cast<int>(kept.size()) < S) {
            const auto& F = fronts[fi];
            int fn = static_cast<int>(F.size());
            std::vector<double> cd(fn, 0.0);
            for (int k = 0; k < m; ++k) {
                std::vector<int> order(fn);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](int a, int b){ return objs[F[a]][k] < objs[F[b]][k]; });
                cd[order.front()] = std::numeric_limits<double>::infinity();
                cd[order.back()]  = std::numeric_limits<double>::infinity();
                double fmin = objs[F[order.front()]][k];
                double fmax = objs[F[order.back()]][k];
                double range = fmax - fmin;
                if (range < 1e-300) continue;
                for (int t = 1; t < fn - 1; ++t)
                    cd[order[t]] += (objs[F[order[t+1]]][k]
                                   - objs[F[order[t-1]]][k]) / range;
            }
            std::vector<int> order(fn);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b){ return cd[a] > cd[b]; });
            int need = S - static_cast<int>(kept.size());
            for (int t = 0; t < need; ++t) kept.push_back(F[order[t]]);
        }
        return kept;
    }

    // ── Allocation (Algorithm 2): distributes the current active pool into K ─
    // blocks of S solutions each. After the call active_n() == K·S,
    // block k = [k*S,(k+1)*S).
    void allocate(DataVault<Ind_t>& vault) {
        int M_act = static_cast<int>(vault.active_n());
        int m     = vault.objs_n();
        int N     = K_ * S_;

        // snapshot of the pool (vars/objs/bvars/lims) BEFORE any restructuring
        std::vector<std::vector<double>> objs(M_act), vars(M_act), lims(M_act);
        std::vector<std::vector<int>>    bvars(M_act);
        std::vector<double>              cvs(M_act, 0.0);
        for (int i = 0; i < M_act; ++i) {
            objs[i]  = vault.objectives_of(i);
            vars[i]  = vault.variables_of(i);
            bvars[i] = vault.binary_variables_of(i);
            lims[i]  = vault.limits_of(i);
            if (constraint_mode != ConstraintMode::NONE) cvs[i] = vault.get_cv(i);
        }

        // origin-shift: subtract the coordinate-wise minimum of the pool (M2M-4)
        std::vector<double> fmin(m, std::numeric_limits<double>::max());
        for (int i = 0; i < M_act; ++i)
            for (int k = 0; k < m; ++k) fmin[k] = std::min(fmin[k], objs[i][k]);
        std::vector<std::vector<double>> U(M_act, std::vector<double>(m));
        for (int i = 0; i < M_act; ++i)
            for (int k = 0; k < m; ++k) U[i][k] = objs[i][k] - fmin[k];

        // partition: each i → the subregion closest by angle (Eq.2)
        std::vector<std::vector<int>> bucket(K_);
        for (int i = 0; i < M_act; ++i) {
            int    best   = 0;
            double best_a = acute_angle(U[i], V_[0]);
            for (int k = 1; k < K_; ++k) {
                double a = acute_angle(U[i], V_[k]);
                if (a < best_a) { best_a = a; best = k; }
            }
            bucket[best].push_back(i);
        }

        // survivors[k] — exactly S source indices per subregion (they may be
        // duplicated across subregions via random refill — this is allowed)
        std::uniform_int_distribution<int> pick(0, M_act - 1);
        std::vector<int> order;        // length N, source indices in block order
        order.reserve(N);
        for (int k = 0; k < K_; ++k) {
            const auto& mem = bucket[k];
            if (static_cast<int>(mem.size()) == S_) {
                for (int i : mem) order.push_back(i);
            } else if (static_cast<int>(mem.size()) > S_) {
                // selection of the best S (NDS + CD)
                std::vector<std::vector<double>> mo;
                std::vector<double> mcv;
                mo.reserve(mem.size()); mcv.reserve(mem.size());
                for (int i : mem) { mo.push_back(objs[i]); mcv.push_back(cvs[i]); }
                std::vector<int> keep = select_best_S(mo, mcv, S_);
                for (int li : keep) order.push_back(mem[li]);
            } else {
                // shortage: all members + S−|mem| random ones from the whole pool (M2M-1)
                for (int i : mem) order.push_back(i);
                int need = S_ - static_cast<int>(mem.size());
                for (int t = 0; t < need; ++t) order.push_back(pick(rng_));
            }
        }

        // materialization: keep N slots and seed them in block order
        // (seed_individual does not re-evaluate — objs are taken from the snapshot)
        vault.reduce(N);
        for (int p = 0; p < N; ++p) {
            int src = order[p];
            vault.seed_individual(static_cast<std::size_t>(p),
                                  vars[src], objs[src], bvars[src], lims[src]);
        }
    }

    // ── one offspring: operator [20] Eq.(5)-(6), base x ∈ P_k, partner y ─────
    // FIX 2026-07-07 (source-fidelity review): was SBX+PM.
    void breed(DataVault<Ind_t>& vault, int x, int y, int dst) {
        int nv = vault.vars_n();
        const auto& bounds = vault.get_bounds();
        std::vector<double> pv1(nv), pv2(nv), c1;
        for (int j = 0; j < nv; ++j) {
            pv1[j] = vault.get_variable(x, j);
            pv2[j] = vault.get_variable(y, j);
        }
        ops::liuli_crossover(pv1, pv2, c1, bounds, gen_, t_max_, rng_);
        ops::liuli_mutation(c1, bounds, pm_eff(nv), gen_, t_max_, rng_);

        int nb = vault.bin_vars_n();
        if (nb > 0) {
            std::vector<int> bv1(nb), bv2(nb), bc1, bc2;
            for (int j = 0; j < nb; ++j) {
                bv1[j] = vault.get_bin_variable(x, j);
                bv2[j] = vault.get_bin_variable(y, j);
            }
            ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
            ops::bit_flip_mutation(bc1, nb, rng_);
            vault.set_all_variables(dst, c1, bc1);
        } else {
            vault.set_variables(dst, c1);
        }
    }

    void init_S_and_dirs(DataVault<Ind_t>& vault) {
        int m = vault.objs_n();
        int N = vault.pop_size();
        build_directions(m);
        if (K_ < 1)
            throw std::invalid_argument("MOEADM2MCore: K < 1");
        if (N % K_ != 0)
            throw std::invalid_argument("MOEADM2MCore: pop_size=" + std::to_string(N) +
                " is not divisible by the number of subregions K=" + std::to_string(K_) +
                ". Set pop_size = K*S (for m=2 K exactly equals set_K).");
        S_ = N / K_;
        if (S_ < 2)
            throw std::invalid_argument("MOEADM2MCore: S = pop_size/K < 2 (the crossover requires ≥2 "
                "solutions in a subpopulation).");
    }

public:
    MOEADM2MCore() = default;

    void set_K(int k)             { K_req_ = k; }
    void set_n_clusters(int k)    { K_req_ = k; }   // alias (for the binding)
    // FIX 2026-07-07: Max_gen of the annealing of operator [20] (see header).
    void set_t_max(int t)         { if (t > 0) t_max_ = t; }
    // M2M-7: no-op shims (operator [20] has no η/p_c; the crossover is unconditional).
    void set_eta_crossover(double){}
    void set_eta_mutation(double) {}
    void set_pc(double)           {}
    void set_pm(double p)         { pm_ = p; }
    void set_seed(unsigned s)     { rng_.seed(s); }

    // ── setup: K×S random initialization + initial Allocation (line 1) ───────
    void setup(DataVault<Ind_t>& vault) {
        init_S_and_dirs(vault);
        gen_ = 0;                          // FIX 2026-07-07: annealing of [20]
        int N = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<int>     dist_bin(0, 1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = dist_bin(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables(i, vars);
        }
        vault.sync();
        allocate(vault);   // Algorithm 2 over the initial K·S points
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        init_S_and_dirs(vault);
        gen_ = 0;                          // FIX 2026-07-07: annealing of [20]
        allocate(vault);   // distribute the seeded population over the subregions
    }

    // ── step: one generation of Algorithm 1 (lines 4-13) ────────────────────
    void step(DataVault<Ind_t>& vault) {
        ++gen_;   // FIX 2026-07-07: first mating at gen=1 (as in [20] Step 1/6)
        int N = K_ * S_;
        // line 5-11: each subregion produces S offspring
        vault.expand(N);                       // offspring slots [N, 2N)
        for (int k = 0; k < K_; ++k) {
            for (int j = 0; j < S_; ++j) {
                int x = k * S_ + j;
                // random y ∈ P_k, preferably ≠ x (M2M-5)
                std::uniform_int_distribution<int> dy(0, S_ - 1);
                int yj = dy(rng_);
                for (int att = 0; att < 5 && yj == j; ++att) yj = dy(rng_);
                int y = k * S_ + yj;
                breed(vault, x, y, N + k * S_ + j);
            }
        }
        vault.sync();
        // line 12-13: Q = R ∪ (∪P_k) is already in active [0,2N); Allocation → blocks
        allocate(vault);
    }

    // compatibility with the base API
    void environmental_selection(DataVault<Ind_t>& vault, int /*target_n*/) {
        allocate(vault);
    }
};

} // namespace mootation

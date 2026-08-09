#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SMS-M2M — S Metric Selection + M2M Population Decomposition.
// L. Chen, H.-L. Liu, C. Lu, Y.-m. Cheung, J. Zhang — "A Novel Evolutionary
// Multi-objective Algorithm Based on S Metric Selection and M2M Population
// Decomposition", in: Proc. 18th Asia Pacific Symposium on Intelligent and
// Evolutionary Systems, Vol. 2 (Proceedings in Adaptation, Learning and
// Optimization 2), Springer, 2015.
// doi:10.1007/978-3-319-13356-0_35
//
// IDEA. The MOEA/D-M2M framework (Liu et al. 2014, doi:10.1109/TEVC.2013.2281533):
// the space is divided by K unit direction vectors into subregions Ω_k (Eq.2),
// each with its own subpopulation of size S. DIFFERENCE from MOEA/D-M2M: when a
// subpopulation is truncated (|P_k|>S), instead of non-dominated sorting,
// SMS selection is used (S-metric / hypervolume selection as in SMS-EMOA),
// applied INSIDE a subregion — this makes the expensive HV computation cheaper
// (O((N/K)^{m-1}) instead of O(N^{m-1}); §2.9). Additionally — mating with a
// cross-subpopulation partner with probability P_c (Algorithm 1).
//
// SCHEME (Algorithm 1):
//   Init: S·K random points, F, Allocation → P_1..P_K.
//   while gen<max_gen:
//     R=∅
//     for k=1..K: foreach x∈P_k:
//        r~U[0,1]; if r<P_c: y from Q\P_k (another subregion); else y from P_k;
//        z=GA(x,y); F(z); R∪={z}.
//     Q := R ∪ (∪P_k)
//     Allocation(Q) → P_1..P_K       (Algorithm 2, SMS truncation)
//     Q := ∪P_k
//   output the non-dominated solutions of ∪P_k.
//
// ALLOCATION (Algorithm 2):
//   ∀k: P_k = {Q : F∈Ω_k};
//       |P_k|<S → ADD S−|P_k| RANDOM solutions from Q;
//       |P_k|>S → SMS removes |P_k|−S solutions ONE BY ONE.
//
// SMS TRUNCATION (= SMS-EMOA, Beume et al. 2007, §2.3): each removal —
//   1. non-dominated sorting of P_k; the LAST (worst) front is taken;
//   2. if it holds 1 solution — that one is removed; otherwise the solution
//      with the MINIMAL exclusive hypervolume contribution ΔS = HV(F)−HV(F\{s})
//      is removed.
//   HV reference point: r_j = 1.1·max_j(P_k) (as in the reference SMS-EMOA).
//
// DEFAULTS (= §2.7):
//   • Operators as in [5] (SMS-EMOA): SBX η_c=20, p_c(SBX)=1.0; PM η_m=20.
//   • P_c (cross-subpopulation mating) = 0.8.
//   • P_m (mutation) = 1/popsiz   (letter of §2.7: «Pm = 1/popsiz»; see SMSM2M-2).
//   • 2-obj.: K=S=10, N=100; 3-obj.: K=S=17, N=300/289.
//   • Directions — "uniformly from the unit sphere"; stopping — generations.
//   In the library: N=pop_size, K=set_K (ANY K≥1, see SMSM2M-5), S=N/K
//   (must divide evenly). The paper's setup K=S=17 (N=289) is reproducible.
//
// DECLARED DEVIATIONS:
//   SMSM2M-1 (DEVIATION, letter of the paper). Algorithm 2: refill with RANDOM
//     solutions from Q (as in the M2M primary source). The M2M reference in
//     PlatEMO refills by angle; we follow the letter of the paper. This is also
//     ablation point A3.
//   SMSM2M-2 (DEVIATION, letter of the paper). §2.7 sets p_m=1/popsiz (and NOT
//     the usual 1/n_vars). Implemented literally: pm=1/pop_size. It can be
//     overridden to 1/n_vars via set_pm if desired.
//   SMSM2M-3 (MINOR). SMS-EMOA has a "modified" variant with the number of
//     dominators d(s,P) (§2.3); here the CANONICAL HV variant is used
//     (last front + min exclusive HV). The paper: «utilizes the same SMS
//     with SMS-EMOA» — we take the basic HV selection.
//   SMSM2M-4 (MINOR). HV reference point r_j=1.1·max_j(P_k) (SMS-EMOA
//     reference); if max_j≤0 → max_j+1. The paper's H-metric y*=(1,…,1) — for
//     selection only the relative contributions matter, not the absolute r.
//   SMSM2M-5 (MINOR). Directions: the letter of the paper §2.7 — «uniformly
//     selected from the unit sphere in the first octant», i.e. an ARBITRARY K
//     is allowed. If K is attainable by the Das–Dennis lattice — the lattice
//     normalized onto the sphere (as in moead_m2m; the previous behavior
//     bit-for-bit). Otherwise — a deterministic generator of exactly K uniform
//     directions: candidates from a fine lattice → farthest-point sampling by
//     angle → Riesz s-energy minimization (s=2, 60 repulsion iterations on the
//     sphere with a clamp into the first octant; the in-project precedent is
//     C2 Energy).
//     FIX 2026-07-07 (source-fidelity review): previously only generate_auto
//     was used — for m=3 K=17 is unattainable (the lattice gave 21 → pop_size
//     289 was rejected with an exception), i.e. the paper's three-objective
//     setup K=S=17 was not reproducible; this consequence had not been
//     declared.
//   SMSM2M-9 (CONSTRAINT, not a deviation — documented for callers).
//     Same structural requirement as moead_m2m (see M2M-8 there): the M2M
//     decomposition needs K equally sized subpopulations, so
//         pop_size = K * S,  S >= 2
//     and any other pop_size throws std::invalid_argument from setup().
//     K is unconstrained (SMSM2M-5); only the divisibility remains. Not
//     auto-corrected on purpose — rounding pop_size would move the
//     function-evaluation budget that a benchmark holds fixed.
//
//   SMSM2M-6..8 (MINOR). As in moead_m2m: origin-shift before the angles; y≠x
//     in the P_k branch (up to 5 attempts); SBX yields 2 children — the first
//     is taken.
//
// EXTENSIONS BEYOND THE PAPER (disabled by default): binary/mixed genome;
//   constraint_mode FEASIBILITY/CDP (SMSM2M-C). The paper is unconstrained, so
//   this is an extension. It makes the non-dominated sort inside sms_reduce a
//   CONSTRAINED one, so the worst front — the one SMS then thins by
//   hypervolume contribution — is populated by feasibility rules first. The HV
//   contribution itself is left on raw objectives: it is a measure, not a
//   preference. The per-subregion quota S stays unconditional, so an
//   all-infeasible subregion still keeps S members, ordered by CV.
//
// COST. SMS computes the HV contribution inside subregions — expensive for
//   large m (like HypE). For smoke tests m=2 suffices.
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
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class SMSM2MCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;  // beyond the paper

private:
    int    K_req_   = 10;
    double eta_c_   = 20.0;
    double eta_m_   = 20.0;
    double pc_      = 1.0;     // SBX crossover rate ([5])
    double pm_      = -1.0;    // <0 → auto 1/pop_size (§2.7, SMSM2M-2)
    double Pc_cross_= 0.8;     // §2.7: cross-subpopulation mating
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> V_;
    int K_ = 0;
    int S_ = 0;
    int N_pop_ = 0;            // for pm=1/popsiz

    double pm_eff() const {
        return (pm_ > 0.0) ? pm_ : (N_pop_ > 0 ? 1.0 / N_pop_ : 0.0);
    }

    static double acute_angle(const std::vector<double>& u,
                              const std::vector<double>& v) {
        double dot = 0.0, nu = 0.0, nv = 0.0;
        for (std::size_t k = 0; k < u.size(); ++k) {
            dot += u[k]*v[k]; nu += u[k]*u[k]; nv += v[k]*v[k];
        }
        double denom = std::sqrt(nu) * std::sqrt(nv);
        if (denom < 1e-300) return 0.0;
        double c = dot / denom;
        if (c >  1.0) c =  1.0;
        if (c < -1.0) c = -1.0;
        return std::acos(c);
    }

    // ── Exactly K uniform directions on the first-octant sphere (SMSM2M-5) ──
    // FIX 2026-07-07 (source-fidelity review): arbitrary K for values
    // unattainable by the lattice (the paper's setup K=S=17, m=3).
    // Deterministic (no RNG used): candidates — the smallest Das–Dennis lattice
    // with ≥2K points on the sphere; FPS by angle from candidate 0 (angular);
    // then 60 iterations of Riesz s-energy repulsion (s=2, step 0.05, clamp
    // into the first octant + renormalization) — the scheme is the
    // deterministic Riesz s-energy (C2 Energy) one.
    void build_directions(int m) {
        // K attainable by the lattice — the previous path (bit-for-bit the old behavior)
        auto W = das_dennis::generate_auto(m, K_req_);
        if (static_cast<int>(W.size()) != K_req_)
            W = detail::uniform_sphere_directions(m, K_req_);     // arbitrary K (FIX)
        V_.clear(); V_.reserve(W.size());
        for (auto& w : W) {
            double n2 = 0.0; for (double wi : w) n2 += wi*wi;
            double nn = std::sqrt(std::max(n2, 1e-300));
            for (double& wi : w) wi /= nn;
            V_.push_back(std::move(w));
        }
        K_ = static_cast<int>(V_.size());
    }

    // ── Hypervolume (HSO slicing), maximization of the union [0,q] ────────────
    // q-points = (ref - f), clamped into [0,∞). Returns the volume of the union.
    static double hv_union(std::vector<std::vector<double>> q) {
        if (q.empty()) return 0.0;
        int m = static_cast<int>(q[0].size());
        if (m == 1) {
            double mx = 0.0; for (auto& p : q) mx = std::max(mx, p[0]);
            return mx;
        }
        // sort by the last coordinate, descending
        std::sort(q.begin(), q.end(),
                  [m](const std::vector<double>& a, const std::vector<double>& b){
                      return a[m-1] > b[m-1];
                  });
        double vol = 0.0;
        int n = static_cast<int>(q.size());
        for (int i = 0; i < n; ++i) {
            double next = (i+1 < n) ? q[i+1][m-1] : 0.0;
            double thick = q[i][m-1] - next;
            if (thick <= 0.0) continue;
            // projection of points [0..i] onto the first m-1 coordinates
            std::vector<std::vector<double>> proj;
            proj.reserve(i+1);
            for (int t = 0; t <= i; ++t)
                proj.emplace_back(q[t].begin(), q[t].begin() + (m-1));
            vol += thick * hv_union(proj);
        }
        return vol;
    }

    static double hv_of(const std::vector<std::vector<double>>& objs,
                        const std::vector<int>& idx,
                        const std::vector<double>& ref) {
        int m = static_cast<int>(ref.size());
        std::vector<std::vector<double>> q;
        q.reserve(idx.size());
        for (int i : idx) {
            std::vector<double> qi(m);
            for (int k = 0; k < m; ++k)
                qi[k] = std::max(0.0, ref[k] - objs[i][k]);
            q.push_back(std::move(qi));
        }
        return hv_union(std::move(q));
    }

    // fast non-dominated sort → vector of ranks (0 = best front)
    std::vector<int> nd_ranks(const std::vector<std::vector<double>>& objs,
                              const std::vector<int>& idx,
                              const std::vector<double>& cvs) const {
        int c = static_cast<int>(idx.size());
        int m = c ? static_cast<int>(objs[idx[0]].size()) : 0;
        // Constrained domination when constraint_mode is on (SMSM2M-C).
        auto dom = [&](int a, int b){
            return detail::dominates(constraint_mode,
                                     objs[idx[a]], cvs[idx[a]],
                                     objs[idx[b]], cvs[idx[b]]);
        };
        (void)m;
        std::vector<int> dom_count(c, 0), rank(c, 0);
        std::vector<std::vector<int>> doms(c);
        std::vector<int> front;
        for (int p = 0; p < c; ++p) {
            for (int qy = 0; qy < c; ++qy) {
                if (p == qy) continue;
                if (dom(p, qy)) doms[p].push_back(qy);
                else if (dom(qy, p)) ++dom_count[p];
            }
            if (dom_count[p] == 0) { rank[p] = 0; front.push_back(p); }
        }
        int r = 0;
        while (!front.empty()) {
            std::vector<int> nxt;
            for (int p : front)
                for (int qy : doms[p])
                    if (--dom_count[qy] == 0) { rank[qy] = r+1; nxt.push_back(qy); }
            front = std::move(nxt); ++r;
        }
        return rank;
    }

    // SMS removal: remove (cnt) solutions from members ONE BY ONE. Returns the
    // indices (values of members) that REMAIN (size = members.size()-cnt).
    std::vector<int> sms_reduce(const std::vector<std::vector<double>>& objs,
                                const std::vector<double>& cvs,
                                std::vector<int> members, int cnt) const {
        int m = static_cast<int>(objs[members[0]].size());
        for (int rep = 0; rep < cnt; ++rep) {
            if (static_cast<int>(members.size()) <= 1) break;
            // 1) non-dominated sorting; take the worst front
            std::vector<int> rank = nd_ranks(objs, members, cvs);
            int maxr = *std::max_element(rank.begin(), rank.end());
            std::vector<int> last_local;  // positions in members
            for (int i = 0; i < (int)members.size(); ++i)
                if (rank[i] == maxr) last_local.push_back(i);

            int victim_local;
            if (last_local.size() == 1) {
                victim_local = last_local[0];
            } else {
                // 2) reference point r_j = 1.1*max_j over ALL members (SMSM2M-4)
                std::vector<double> ref(m, -std::numeric_limits<double>::max());
                for (int i : members)
                    for (int k = 0; k < m; ++k) ref[k] = std::max(ref[k], objs[i][k]);
                for (int k = 0; k < m; ++k)
                    ref[k] = (ref[k] > 0.0) ? ref[k]*1.1 : ref[k] + 1.0;

                // exclusive HV contribution of each member of the last front
                std::vector<int> last_idx;     // global indices of the last front
                last_idx.reserve(last_local.size());
                for (int lp : last_local) last_idx.push_back(members[lp]);
                double hv_all = hv_of(objs, last_idx, ref);
                double worst_contrib = std::numeric_limits<double>::max();
                victim_local = last_local[0];
                for (std::size_t t = 0; t < last_idx.size(); ++t) {
                    std::vector<int> without;
                    without.reserve(last_idx.size()-1);
                    for (std::size_t u = 0; u < last_idx.size(); ++u)
                        if (u != t) without.push_back(last_idx[u]);
                    double contrib = hv_all - hv_of(objs, without, ref);
                    if (contrib < worst_contrib) {
                        worst_contrib = contrib;
                        victim_local  = last_local[t];
                    }
                }
            }
            members.erase(members.begin() + victim_local);
        }
        return members;
    }

    // ── Allocation (Algorithm 2) ────────────────────────────────────────────
    void allocate(DataVault<Ind_t>& vault) {
        int M_act = static_cast<int>(vault.active_n());
        int m     = vault.objs_n();
        int N     = K_ * S_;

        std::vector<std::vector<double>> objs(M_act), vars(M_act), lims(M_act);
        std::vector<std::vector<int>>    bvars(M_act);
        std::vector<double>              cvs(M_act, 0.0);
        for (int i = 0; i < M_act; ++i) {
            if (constraint_mode != ConstraintMode::NONE) cvs[i] = vault.get_cv(i);
            objs[i]  = vault.objectives_of(i);
            vars[i]  = vault.variables_of(i);
            bvars[i] = vault.binary_variables_of(i);
            lims[i]  = vault.limits_of(i);
        }

        std::vector<double> fmin(m, std::numeric_limits<double>::max());
        for (int i = 0; i < M_act; ++i)
            for (int k = 0; k < m; ++k) fmin[k] = std::min(fmin[k], objs[i][k]);
        std::vector<std::vector<double>> U(M_act, std::vector<double>(m));
        for (int i = 0; i < M_act; ++i)
            for (int k = 0; k < m; ++k) U[i][k] = objs[i][k] - fmin[k];

        std::vector<std::vector<int>> bucket(K_);
        for (int i = 0; i < M_act; ++i) {
            int best = 0; double best_a = acute_angle(U[i], V_[0]);
            for (int k = 1; k < K_; ++k) {
                double a = acute_angle(U[i], V_[k]);
                if (a < best_a) { best_a = a; best = k; }
            }
            bucket[best].push_back(i);
        }

        std::uniform_int_distribution<int> pick(0, M_act - 1);
        std::vector<int> order; order.reserve(N);
        for (int k = 0; k < K_; ++k) {
            const auto& mem = bucket[k];
            if ((int)mem.size() == S_) {
                for (int i : mem) order.push_back(i);
            } else if ((int)mem.size() > S_) {
                std::vector<int> keep = sms_reduce(objs, cvs, mem, (int)mem.size() - S_);
                for (int i : keep) order.push_back(i);
            } else {
                for (int i : mem) order.push_back(i);
                int need = S_ - (int)mem.size();
                for (int t = 0; t < need; ++t) order.push_back(pick(rng_));
            }
        }

        vault.reduce(N);
        for (int p = 0; p < N; ++p) {
            int src = order[p];
            vault.seed_individual((std::size_t)p,
                                  vars[src], objs[src], bvars[src], lims[src]);
        }
    }

    void breed(DataVault<Ind_t>& vault, int x, int y, int dst) {
        int nv = vault.vars_n();
        const auto& bounds = vault.get_bounds();
        std::vector<double> pv1(nv), pv2(nv), c1, c2;
        for (int j = 0; j < nv; ++j) {
            pv1[j] = vault.get_variable(x, j);
            pv2[j] = vault.get_variable(y, j);
        }
        ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
        ops::polynomial_mutation(c1, bounds, eta_m_, pm_eff(), rng_);
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
        N_pop_ = vault.pop_size();
        build_directions(m);
        if (K_ < 1) throw std::invalid_argument("SMSM2MCore: K < 1");
        if (N_pop_ % K_ != 0)
            throw std::invalid_argument("SMSM2MCore: pop_size=" + std::to_string(N_pop_) +
                " is not divisible by K=" + std::to_string(K_) + " (pop_size=K*S is required).");
        S_ = N_pop_ / K_;
        if (S_ < 2)
            throw std::invalid_argument("SMSM2MCore: S=pop_size/K < 2.");
    }

public:
    SMSM2MCore() = default;

    void set_K(int k)              { K_req_ = k; }
    void set_n_clusters(int k)     { K_req_ = k; }
    void set_cross_prob(double p)  { Pc_cross_ = p; }
    void set_eta_crossover(double e){ eta_c_ = e; }
    void set_eta_mutation(double e) { eta_m_ = e; }
    void set_pc(double p)          { pc_ = p; }
    void set_pm(double p)          { pm_ = p; }
    void set_seed(unsigned s)      { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        init_S_and_dirs(vault);
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
                vars[j] = lo + dist(rng_)*(hi-lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = dist_bin(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables(i, vars);
        }
        vault.sync();
        allocate(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        init_S_and_dirs(vault);
        allocate(vault);
    }

    void step(DataVault<Ind_t>& vault) {
        int N = K_ * S_;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        vault.expand(N);
        for (int k = 0; k < K_; ++k) {
            for (int j = 0; j < S_; ++j) {
                int x = k*S_ + j;
                int y;
                if (uni(rng_) < Pc_cross_ && K_ > 1) {
                    // y from Q\P_k: an index outside block k
                    std::uniform_int_distribution<int> dout(0, N - S_ - 1);
                    int r = dout(rng_);
                    if (r >= k*S_) r += S_;
                    y = r;
                } else {
                    std::uniform_int_distribution<int> din(0, S_ - 1);
                    int yj = din(rng_);
                    for (int att = 0; att < 5 && yj == j; ++att) yj = din(rng_);
                    y = k*S_ + yj;
                }
                breed(vault, x, y, N + k*S_ + j);
            }
        }
        vault.sync();
        allocate(vault);
    }

    void environmental_selection(DataVault<Ind_t>& vault, int /*target_n*/) {
        allocate(vault);
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// GrEA — A Grid-Based Evolutionary Algorithm for Many-Objective Optimization
// S. Yang, M. Li, X. Liu, J. Zheng — IEEE TEVC 17(5), 2013
// doi:10.1109/TEVC.2012.2227145
//
// Generational scheme (Algorithm 1):
//   1. Grid setting (Eq.3-6, an adaptive grid over P) plus fitness assignment
//      on the current population P
//   2. Mating (Alg.2): p dominates q, or p grid-dominates q -> p; symmetrically
//      for q; otherwise the smaller GCD; otherwise at random. Variation: SBX
//      (pc=1) + PM (pm=1/n) -> P''
//   3. Environmental selection (Alg.4) on P u P'' (2N): non-dominated sorting
//      with whole fronts accepted; on the critical front F_i, its own grid,
//      Initialization (Alg.5: GR = Eq.9, GCPD = Eq.11, GCD <- 0), then greedy
//      Findout_best (Alg.7: min GR -> min GCD -> min GCPD) -> Q;
//      GCD_calculation (Alg.6: dynamic GCD relative to the archive) plus
//      GR_adjustment (Alg.3: E(q) -> +(M+2), G(q) -> +M, neighbours -> +PD)
//
// Definitions: grid dominance Eq.7; GD(x,y) = Sum |G_i(x) − G_i(y)| (Eq.8);
//   GR = Sum G_i (Eq.9); GCD(x) = Sum_{y: GD<M} (M − GD) (Eq.10); GCPD is the
//   normalized Euclidean distance to the best corner of the point's own
//   hyperbox, where smaller is better (Eq.11).
//
// PAPER DEFAULTS (§IV-C.1): pc=1.0, pm=1/n, eta_c=20, eta_m=20.
//   div has NO single paper-wide value. §IV-C.4 sets it per problem via
//   Table II (range 4-50). §V-B sweeps div over [5,50] on DTLZ2 and reports the
//   best values as 16/9/9/8/7/7/8/9 for 3/4/5/6/8/10/12/15 objectives, and for
//   an UNKNOWN problem it recommends "a division value around 9 ... a slightly
//   larger div if the problem is hard to converge, and a slightly lower value
//   if coverage of the Pareto front is more emphasized". The library default is
//   div=8 — inside that recommended band, but NOT a paper value: reproducing
//   any specific experiment requires set_div.
// DECLARED DEVIATIONS: the PD propagation in GR_adjustment is performed even
//   without increasing PD(p); this is behaviourally equivalent to the letter of
//   Alg.3 line 11 by the transitivity of grid dominance.
// EXTENSIONS BEYOND THE PAPER: constraint_mode FEASIBILITY (CDP; off by
//   default; the grid is built over the feasible subset); real+binary genome.
//
// Individual: GrEA_Individual (individuals.hpp) carries grid_coord, gr, gcd;
//   GCPD is held locally in the environmental selection as an unordered_map.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
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
class GrEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int          div_  = 8;     // paper: per-problem (see header); use set_div
    double       eta_c_ = 20.0; // §IV-C.1
    double       eta_m_ = 20.0; // §IV-C.1
    double       pc_    = 1.0;  // §IV-C.1: «A crossover probability pc = 1.0»
    std::mt19937 rng_{std::random_device{}()};

    // ── Grid setup for pool [0, n) ─────────────────────────────────────────
    struct GridParams {
        std::vector<double> lb, d;
        int m, div;
    };

    // `n` is unused: the grid is built from `subset`, not from the whole
    // population. Kept in the signature for call-site symmetry with the other
    // per-generation helpers.
    GridParams build_grid(DataVault<Ind_t>& vault, int /*n*/,
                          const std::vector<int>& subset) const {
        int m = vault.objs_n();
        GridParams gp; gp.m = m; gp.div = div_;
        gp.lb.resize(m); gp.d.resize(m);
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (int i : subset) {
            const auto& o = vault.objectives_of(i);
            for (int j=0;j<m;++j) { fmin[j]=std::min(fmin[j],o[j]); fmax[j]=std::max(fmax[j],o[j]); }
        }
        for (int j=0;j<m;++j) {
            double span = std::max(fmax[j]-fmin[j], 1e-14);
            gp.lb[j] = fmin[j] - span/(2.0*div_);
            gp.d [j] = (fmax[j]+span/(2.0*div_) - gp.lb[j]) / div_;
            if (gp.d[j] < 1e-14) gp.d[j] = 1.0;
        }
        return gp;
    }

    // G_k(x) ∈ [0, div-1].
    int grid_coord(double fk, const GridParams& gp, int k) const {
        int g = static_cast<int>((fk - gp.lb[k]) / gp.d[k]);
        return std::max(0, std::min(div_-1, g));
    }

    void compute_grid_coords(DataVault<Ind_t>& vault,
                             const std::vector<int>& pool,
                             const GridParams& gp) {
        int m = gp.m;
        for (int i : pool) {
            const auto& o = vault.objectives_of(i);
            auto& ind = vault.get_ind(i);
            ind.grid_coord.resize(m);
            for (int j=0;j<m;++j)
                ind.grid_coord[j] = grid_coord(o[j], gp, j);
        }
    }

    // GD(x, y) = Σ |G_i(x) - G_i(y)|
    int grid_diff(DataVault<Ind_t>& vault, int a, int b) const {
        const auto& ga = vault.get_ind(a).grid_coord;
        const auto& gb = vault.get_ind(b).grid_coord;
        int d = 0;
        for (std::size_t k=0;k<ga.size();++k) d += std::abs(ga[k]-gb[k]);
        return d;
    }

    // Grid dominance (Yang et al. 2013, Eq.7): a ≺grid b ⟺
    //   ∀i G_i(a) ≤ G_i(b)  ∧  ∃j G_j(a) < G_j(b).
    bool grid_dominates(DataVault<Ind_t>& vault, int a, int b) const {
        const auto& ga = vault.get_ind(a).grid_coord;
        const auto& gb = vault.get_ind(b).grid_coord;
        bool strict = false;
        for (std::size_t k=0;k<ga.size();++k) {
            if (ga[k] > gb[k]) return false;
            if (ga[k] < gb[k]) strict = true;
        }
        return strict;
    }

    // ── Pareto dominance (Eq.2); CDP under FEASIBILITY ─────────────────────
    bool pareto_dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return ca < cb;        // both infeasible -> smaller CV
        }
        const auto& oa = vault.objectives_of(a);
        const auto& ob = vault.objectives_of(b);
        bool strict = false;
        for (std::size_t k = 0; k < oa.size(); ++k) {
            if (oa[k] > ob[k]) return false;
            if (oa[k] < ob[k]) strict = true;
        }
        return strict;
    }

    // ── Fast nondominated sort (Deb 2002) -> fronts (Algorithm 4, line 2) ───
    std::vector<std::vector<int>>
    nondominated_sort(DataVault<Ind_t>& vault, const std::vector<int>& pool) {
        int n = static_cast<int>(pool.size());
        std::vector<std::vector<int>> S(n);
        std::vector<int> ndom(n, 0);
        std::vector<int> f0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (pareto_dominates(vault, pool[i], pool[j])) S[i].push_back(j);
                else if (pareto_dominates(vault, pool[j], pool[i])) ++ndom[i];
            }
            if (ndom[i] == 0) f0.push_back(i);
        }
        std::vector<std::vector<int>> fronts;
        fronts.push_back({});
        for (int idx : f0) fronts.back().push_back(pool[idx]);
        std::vector<int> cur = f0;
        while (!cur.empty()) {
            std::vector<int> next;
            for (int i : cur)
                for (int j : S[i])
                    if (--ndom[j] == 0) next.push_back(j);
            if (next.empty()) break;
            fronts.push_back({});
            for (int idx : next) fronts.back().push_back(pool[idx]);
            cur = next;
        }
        return fronts;
    }

    // ── GCPD (Yang et al. 2013, Eq.11): the normalized Euclidean distance to
    //    the best (utopian) corner of the point's OWN hyperbox; SMALLER IS
    //    BETTER.
    //    GCPD(x)=sqrt(Σ_k ((F_k(x) − (lb_k + G_k(x)·d_k))/d_k)^2). ───────────
    double gcpd_eq11(DataVault<Ind_t>& vault, int i, const GridParams& gp) const {
        const auto& o  = vault.objectives_of(i);
        const auto& gc = vault.get_ind(i).grid_coord;
        double s = 0.0;
        for (int k = 0; k < gp.m; ++k) {
            double corner = gp.lb[k] + static_cast<double>(gc[k]) * gp.d[k];
            double t = (o[k] - corner) / gp.d[k];
            s += t * t;
        }
        return std::sqrt(s);
    }

    // ── GR (Eq.9) + static GCD (Eq.10) over the pool, for mating (Alg.2).
    //    Here GCD is computed relative to the whole population, as in Fitness
    //    assignment (Algorithm 1, line 4). For environmental selection, GCD is
    //    built dynamically from scratch (Algorithm 5/6) — see
    //    environmental_selection.
    void compute_gr_gcd(DataVault<Ind_t>& vault,
                        const std::vector<int>& pool,
                        const GridParams& gp) {
        int m = gp.m;
        for (int i : pool) {
            auto& ind = vault.get_ind(i);
            int gr = 0;
            for (int j = 0; j < m; ++j) gr += ind.grid_coord[j];
            ind.gr = gr;
            int gcd = 0;
            for (int j : pool) {
                if (j == i) continue;
                int gd = grid_diff(vault, i, j);
                if (gd < m) gcd += m - gd;
            }
            ind.gcd = gcd;
        }
    }

    // ── GR adjustment for environmental selection ─────────────────────────
    // Yang et al. 2013, Algorithm 3 (GR adjustment). When an individual q is
    // selected into the archive, the GR of its "related" individuals is
    // penalized, i.e. increased, since a smaller GR is better. Three groups,
    // with penalties M+2 / M / [0,M-1]:
    //   E(q)  = {p : GD(p,q)=0}              -> GR += M+2
    //   G(q)  = {p : q grid-dominates p}     -> GR += M
    //   neighbours p in N(q) & NG(q), p not in E(q):
    //     PD(p) = max(0, M - GD(p,q));
    //     for r grid-dominated by p (r not in G(q) u E(q)):
    //     PD(r) = max(PD(r), PD(p)); finally GR(p) += PD(p) for every
    //     p in NG(q), p not in E(q).
    // The code used to adjust GCD instead, which is a different mechanism.
    // Only GR is updated now (Algorithm 3); there is no scalar fitness any
    // more — selection and mating work directly on the GR/GCD/GCPD criteria.
    void adjust_gr(DataVault<Ind_t>& vault,
                   int q,
                   const std::vector<int>& pool) {
        int m = vault.objs_n();

        // Classification and PD.
        std::vector<int> PD;            // punishment degree, indexed by pool position
        std::vector<char> inE, inG, inNG;
        PD.reserve(pool.size());
        inE.reserve(pool.size()); inG.reserve(pool.size()); inNG.reserve(pool.size());
        for (int p : pool) {
            int gd = grid_diff(vault, p, q);
            bool e  = (gd == 0);
            bool g  = grid_dominates(vault, q, p);   // q ≺grid p
            bool ng = !g;                            // q ⊀grid p
            inE.push_back(e); inG.push_back(g); inNG.push_back(ng);
            PD.push_back(0);
        }

        // Group 1: E(q) -> +(M+2);  Group 2: G(q) -> +M.
        for (std::size_t k=0;k<pool.size();++k) {
            int p = pool[k];
            if (p == q) continue;
            if (inE[k]) {
                vault.get_ind(p).gr += (m + 2);
            } else if (inG[k]) {
                vault.get_ind(p).gr += m;
            }
        }

        // Group 3, the neighbours (Algorithm 3, lines 7-19): PD over the
        // neighbours N(q) & NG(q) \ E(q), propagated to the grid-dominated.
        for (std::size_t k=0;k<pool.size();++k) {
            int p = pool[k];
            if (p == q || inE[k] || !inNG[k]) continue;
            int gd = grid_diff(vault, p, q);
            if (gd < m) {
                int pd_p = m - gd;
                if (PD[k] < pd_p) PD[k] = pd_p;
                // propagate to r grid-dominated by p, with r not in G(q) u E(q):
                for (std::size_t r=0;r<pool.size();++r) {
                    if (r == k) continue;
                    int rp = pool[r];
                    if (inG[r] || inE[r]) continue;
                    if (grid_dominates(vault, p, rp)) {       // p ≺grid r
                        if (PD[r] < PD[k]) PD[r] = PD[k];
                    }
                }
            }
        }
        // Final step (lines 20-22): GR(p) += PD(p) for p in NG(q), p not in E(q).
        for (std::size_t k=0;k<pool.size();++k) {
            int p = pool[k];
            if (p == q || inE[k] || !inNG[k] || PD[k] == 0) continue;
            vault.get_ind(p).gr += PD[k];
        }
    }

    // ── GCD calculation (Yang et al. 2013, Algorithm 6): dynamically update
    //    the GCD of the neighbours of the selected q within the candidate
    //    front Fi, relative to the growing archive. ───────────────────────────
    void gcd_calculation(DataVault<Ind_t>& vault,
                         const std::vector<int>& Fi, int q) {
        int m = vault.objs_n();
        for (int p : Fi) {
            int gd = grid_diff(vault, p, q);
            if (gd < m) vault.get_ind(p).gcd += (m - gd);
        }
    }

    // ── Findout_best (Algorithm 7): min GR → min GCD → min GCPD. ───────────
    int findout_best(DataVault<Ind_t>& vault, const std::vector<int>& Fi,
                     const std::unordered_map<int,double>& gcpd) const {
        int q = Fi[0];
        for (std::size_t i = 1; i < Fi.size(); ++i) {
            int p = Fi[i];
            const auto& ip = vault.get_ind(p);
            const auto& iq = vault.get_ind(q);
            if (ip.gr < iq.gr) { q = p; }
            else if (ip.gr == iq.gr) {
                if (ip.gcd < iq.gcd) { q = p; }
                else if (ip.gcd == iq.gcd) {
                    if (gcpd.at(p) < gcpd.at(q)) q = p;
                }
            }
        }
        return q;
    }

    // ── Environmental selection (Yang et al. 2013, Algorithm 4) ────────────
    // Non-dominated sort of the pool -> accept the whole leading fronts -> on
    // the critical front Fi: its own grid, init (GR Eq.9, GCPD Eq.11, GCD=0),
    // then greedily: Findout_best -> into the archive -> GCD_calculation
    // (Alg.6, dynamic) + GR_adjustment (Alg.3). Here GCD is built FROM SCRATCH
    // relative to the selected archive.
    std::vector<int> environmental_selection(DataVault<Ind_t>& vault,
                                             const std::vector<int>& pool, int N) {
        auto fronts = nondominated_sort(vault, pool);
        std::vector<int> Q;
        Q.reserve(N);
        std::size_t fi = 0;
        while (fi < fronts.size() &&
               static_cast<int>(Q.size() + fronts[fi].size()) <= N) {
            for (int x : fronts[fi]) Q.push_back(x);
            ++fi;
        }
        if (static_cast<int>(Q.size()) == N || fi >= fronts.size())
            return Q;

        std::vector<int> Fi = fronts[fi];                  // the critical front
        std::vector<int> feas = Fi;
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            std::vector<int> ff;
            for (int x : Fi) if (vault.get_cv(x) <= 0.0) ff.push_back(x);
            if (!ff.empty()) feas = ff;
        }
        auto gp = build_grid(vault, static_cast<int>(Fi.size()), feas);
        compute_grid_coords(vault, Fi, gp);

        // Initialization (Algorithm 5): GR (Eq.9), GCPD (Eq.11), GCD ← 0.
        std::unordered_map<int,double> gcpd;
        gcpd.reserve(Fi.size() * 2);
        for (int p : Fi) {
            auto& ind = vault.get_ind(p);
            int gr = 0;
            for (int k = 0; k < gp.m; ++k) gr += ind.grid_coord[k];
            ind.gr  = gr;
            ind.gcd = 0;
            gcpd[p] = gcpd_eq11(vault, p, gp);
        }

        while (static_cast<int>(Q.size()) < N && !Fi.empty()) {
            int q = findout_best(vault, Fi, gcpd);
            Q.push_back(q);
            Fi.erase(std::find(Fi.begin(), Fi.end(), q));
            gcd_calculation(vault, Fi, q);   // Algorithm 6 — dynamic GCD
            adjust_gr(vault, q, Fi);         // Algorithm 3 — GR adjustment
        }
        return Q;
    }

    // ── Mating selection (Yang et al. 2013, Algorithm 2) ───────────────────
    // p≺q ∨ p≺grid q → p ; q≺p ∨ q≺grid p → q ; GCD(p)<GCD(q) → p ;
    // GCD(q) < GCD(p) -> q; otherwise at random. Pareto dominance with CDP
    // under FEASIBILITY. (The tournament used to run on a scalarized GR
    // fitness.)
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int p = dist(rng_), q = dist(rng_);
        if (pareto_dominates(vault, p, q) || grid_dominates(vault, p, q)) return p;
        if (pareto_dominates(vault, q, p) || grid_dominates(vault, q, p)) return q;
        if (vault.get_ind(p).gcd < vault.get_ind(q).gcd) return p;
        if (vault.get_ind(q).gcd < vault.get_ind(p).gcd) return q;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return (u(rng_) < 0.5) ? p : q;
    }

    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool_size) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool_size), at_pos(pool_size);
        std::iota(pos.begin(),pos.end(),0); std::iota(at_pos.begin(),at_pos.end(),0);
        for (int i=0;i<n;++i) {
            int want=survivors[i], cur=pos[want];
            if(cur==i) continue;
            int other=at_pos[i];
            vault.swap_active(i,cur);
            pos[want]=i; pos[other]=cur; at_pos[i]=want; at_pos[cur]=other;
        }
        vault.reduce(n);
    }

public:
    GrEACore() = default;

    void set_div          (int d)    { div_   = std::max(1, d); }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_    = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int>     db(0,1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i=0;i<n;++i) {
            for(int j=0;j<vault.vars_n();++j){double lo=bounds[j].first.value_or(0.0),hi=bounds[j].second.value_or(1.0);vars[j]=lo+dr(rng_)*(hi-lo);}
            for(int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars); else vault.set_variables(i,vars);
        }
        vault.sync();
        std::vector<int> pool(n); std::iota(pool.begin(),pool.end(),0);
        auto gp = build_grid(vault, n, pool);
        compute_grid_coords(vault, pool, gp);
        compute_gr_gcd(vault, pool, gp);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        std::vector<int> pool(n); std::iota(pool.begin(),pool.end(),0);
        auto gp = build_grid(vault, n, pool);
        compute_grid_coords(vault, pool, gp);
        compute_gr_gcd(vault, pool, gp);
    }

    // ── step (Yang et al. 2013, Algorithm 1: one generational cycle) ───────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n-1);

        // (Algorithm 1, lines 3-4) Grid setting + fitness assignment on the
        // current population P=[0,n), feeding mating (Algorithm 2 uses grid
        // dominance and GCD).
        {
            std::vector<int> P(n); std::iota(P.begin(), P.end(), 0);
            std::vector<int> feas = P;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                std::vector<int> ff;
                for (int i : P) if (vault.get_cv(i) <= 0.0) ff.push_back(i);
                if (!ff.empty()) feas = ff;
            }
            auto gp = build_grid(vault, n, feas);
            compute_grid_coords(vault, P, gp);
            compute_gr_gcd(vault, P, gp);
        }

        // (lines 5-6) Mating selection (Algorithm 2) + variation → [off_base,+n).
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i=0;i<n;i+=2) {
            int p1=tournament(vault,dist_int), p2=tournament(vault,dist_int);
            for(int j=0;j<vault.vars_n();++j){pv1[j]=vault.get_variable(p1,j);pv2[j]=vault.get_variable(p2,j);}
            // §IV-C.1: pc=1.0, pm=1/n
            double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::sbx(pv1,pv2,c1,c2,bounds,eta_c_,pc_,rng_);
            ops::polynomial_mutation(c1,bounds,eta_m_,pm,rng_);
            ops::polynomial_mutation(c2,bounds,eta_m_,pm,rng_);
            if(vault.bin_vars_n()>0){
                std::vector<int> bv1(vault.bin_vars_n()),bv2(vault.bin_vars_n()),bc1,bc2;
                for(int j=0;j<vault.bin_vars_n();++j){bv1[j]=vault.get_bin_variable(p1,j);bv2[j]=vault.get_bin_variable(p2,j);}
                ops::binary_crossover(bv1,bv2,bc1,bc2,rng_);
                ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
                if(i+1<n) ops::bit_flip_mutation(bc2,vault.bin_vars_n(),rng_);
                vault.set_all_variables(off_base + i,c1,bc1);
                if(i+1<n) vault.set_all_variables(off_base + i+1,c2,bc2);
            } else { vault.set_variables(off_base + i,c1); if(i+1<n) vault.set_variables(off_base + i+1,c2); }
        }
        vault.sync();

        // (line 7) Environmental selection (Algorithm 4) on P u P'' = [0, 2n).
        int pool_size = n * 2;
        std::vector<int> pool(pool_size); std::iota(pool.begin(),pool.end(),0);
        std::vector<int> survivors = environmental_selection(vault, pool, n);
        rearrange(vault, survivors, pool_size);
    }
};

} // namespace mootation

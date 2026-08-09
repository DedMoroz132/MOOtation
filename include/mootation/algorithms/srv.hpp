#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaOEA/SRV — Many-Objective EA with Self-Guided Reference Vector strategy
// Liu, Lin, Wong, Coello Coello, Li, Ming, Zhang — IEEE Trans. Cybernetics, 2022
// doi:10.1109/TCYB.2020.2971638
//
// Generational scheme (Algorithm 4):
//   1. SBX + polynomial mutation on P -> Q (N offspring); U = P + Q.
//   2. Non-dominated sorting of U -> St (the leading fronts, |St| >= N);
//      Pc = St, normalized per Eq.2 (z^nad = the max over Pc).
//   3. theta_c per Eq.17 (G = gen+1); N SRVs = SRVStrategy::extract(Pc, N,
//      theta_c) — adapted every generation (tau=1, the best setting in
//      Table IV).
//   4. APP: each x in Pc goes to the nearest SRV by angle theta(x, rv_i)
//      (Eq.6), giving N subsets (diversity).
//   5. ESS: a single fixed RV r* (ideal -> nadir in the normalized space,
//      = (1..1)/sqrt(m)); I_c = d1(x, r*) (Eq.5); ranking within the subsets
//      gives the levels S^L_1..S^L_L; the last level is filled at random.
//
// PAPER DEFAULTS (§IV-B): the operators follow NSGA-III [21] — SBX p_c=1.0 /
//   eta_c=30, PM p_m=1/n / eta_m=20. The paper defers the exact values to
//   Table A.II of a supplementary file that is unavailable, so the [21]
//   convention is used.
// DECLARED DEVIATIONS: z^nad = the max over St (Alg.4 line 5 only says
//   "normalize by (2)"); mating uses random pairs, as the paper specifies no
//   parent-selection operator.
//   G_max DEPENDENCE (contract, not optional). The theta_c annealing schedule
//     (Eq.17) is driven by t_max_, which defaults to 1000. A caller who never
//     calls set_t_max therefore runs a schedule computed against a budget that
//     has nothing to do with the real one: at 200 generations the anneal never
//     leaves its early regime, and past 1000 it saturates. set_t_max is
//     effectively mandatory for reproducing the paper.

// EXTENSIONS BEYOND THE PAPER: binary variables (uniform crossover +
//   bit-flip), constraint_mode FEASIBILITY (off by default).
// ============================================================================
// Notable fixes: (1) theta_min_c per Eq.16 — theta_min() now computes the true
// angles (the change lives in srv_strategy.hpp); (2) when the RV set is padded
// with leftover V0 points, those are now projected onto the unit sphere
// (SRV-2); (3) with an odd n the last pair used to be p1==p2, i.e. SBX of an
// individual with its own clone — the partner is now re-drawn from the rest
// (SRV-3).

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"
#include "srv_strategy.hpp"

namespace mootation {

template <typename Ind_t>
class SRVCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_  = 30.0;   // NSGA-III [21] convention (Table A.II, suppl.)
    double       eta_m_  = 20.0;
    double       pc_     = 1.0;    // SBX pair probability (NSGA-III [21])
    int          t_max_  = 1000;
    std::mt19937 rng_{std::random_device{}()};
    int          gen_    = 0;

    // Initial Das-Dennis vectors (for θ_min_c computation and fallback)
    std::vector<std::vector<double>> V0_;
    double theta_min_c_ = 0.0;

    SRVStrategy<Ind_t> srv_;

    // ── Das-Dennis generation ────────────────────────────────────────────────
    void init_V0(int n, int m) {
        V0_ = das_dennis::generate_exact(m, n);
        // theta_c^min per Eq.16 from the original RVs; theta_min() computes
        // the true angles, normalizing internally.
        theta_min_c_ = SRVStrategy<Ind_t>::theta_min(V0_);
    }


    // ── Dominance ────────────────────────────────────────────────────────────
    bool dominates(DataVault<Ind_t>& vault, int a, int b) {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca=vault.get_cv(a), cb=vault.get_cv(b);
            bool af=(ca<=0.0), bf=(cb<=0.0);
            if(af&&!bf) return true;
            if(!af&&bf) return false;
            if(!af&&!bf) return ca<cb;
        }
        const auto& fa=vault.objectives_of(a); const auto& fb=vault.objectives_of(b);
        bool better=false;
        for(std::size_t i=0;i<fa.size();++i){
            if(fa[i]>fb[i]) return false;
            if(fa[i]<fb[i]) better=true;
        }
        return better;
    }

    // ── Fast NDS → returns fronts over [0, pool) ─────────────────────────────
    std::vector<std::vector<int>> fast_nds(DataVault<Ind_t>& vault, int pool) {
        std::vector<std::vector<int>> S(pool); std::vector<int> np(pool,0);
        for(int i=0;i<pool;++i) for(int j=0;j<pool;++j){
            if(i==j) continue;
            if(dominates(vault,i,j)) S[i].push_back(j);
            else if(dominates(vault,j,i)) ++np[i];
        }
        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for(int i=0;i<pool;++i) if(np[i]==0) f0.push_back(i);
        fronts.push_back(f0);
        int k=0;
        while(!fronts[k].empty()){
            std::vector<int> nxt;
            for(int i:fronts[k]) for(int j:S[i]) if(--np[j]==0) nxt.push_back(j);
            fronts.push_back(nxt); ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── Environmental selection (Algorithm 4 §ESS) ────────────────────────────
    // Pc = St (first l fronts with |St| ≥ N)
    // APP: assign each x ∈ Pc to closest SRV (angle)
    // ESS: rank by d1(x, r*) within each subset; pick level by level
    std::vector<int> env_select(DataVault<Ind_t>& vault, int pool, int N) {
        int m = vault.objs_n();

        // ── Build St ─────────────────────────────────────────────────────────
        auto fronts = fast_nds(vault, pool);
        std::vector<int> St;
        for (auto& front : fronts) {
            for (int v : front) St.push_back(v);
            if (static_cast<int>(St.size()) >= N) break;
        }

        // ── Normalise objectives in St → f'(x), EdI(x) ───────────────────────
        std::vector<double> zstar(m, std::numeric_limits<double>::max());
        std::vector<double> znad (m,-std::numeric_limits<double>::max());
        for (int v : St) {
            const auto& o = vault.objectives_of(v);
            for (int j=0;j<m;++j){ zstar[j]=std::min(zstar[j],o[j]); znad[j]=std::max(znad[j],o[j]); }
        }
        auto f_prime = [&](int v) -> std::vector<double> {
            const auto& o = vault.objectives_of(v);
            std::vector<double> fp(m);
            for(int j=0;j<m;++j){
                double range=std::max(znad[j]-zstar[j],1e-14);
                fp[j]=(o[j]-zstar[j])/range;
            }
            return fp;
        };
        auto edi = [&](const std::vector<double>& fp) -> double {
            double s=0.0; for(double x:fp) s+=x*x; return std::sqrt(s);
        };

        // ── SRV: extract N reference vectors from Pc = St ────────────────────
        // Liu et al. 2022, Algorithm 1/4: Pc = St, so the adaptive RVs are
        // extracted from St (the leading fronts of the merged pool, |St| >= N).
        // Extracting from the population [0,N), a subset of St, was simpler but
        // wrong. The extract(vault, St, N, theta_c) overload is used: St are the
        // clustering points, N is the number of clusters/RVs.
        double theta_c = SRVStrategy<Ind_t>::compute_theta_c(theta_min_c_, gen_+1, t_max_);
        auto RVs = srv_.extract(vault, St, N, theta_c);

        // Ensure we have exactly N RVs (pad with V0 if needed).
        // SRV-2: V0 holds Das-Dennis simplex points, whose norm is not 1. When
        // padding the array of unit RVs they are PROJECTED onto the unit
        // sphere; otherwise the angular association, which reads the dot
        // product as a cosine, would mix scales.
        while (static_cast<int>(RVs.size()) < N) {
            std::vector<double> v = V0_[RVs.size()];
            double nrm = 0.0;
            for (double x : v) nrm += x * x;
            nrm = std::sqrt(std::max(nrm, 1e-28));
            for (double& x : v) x /= nrm;
            RVs.push_back(std::move(v));
        }
        RVs.resize(N);

        // ── APP: assign each x ∈ St to closest RV by angle ──────────────────
        // Also compute d1(x, r*) for ESS
        // r* = normalised(f'(centroid of St)) connecting ideal to nadir
        // Actually per paper: r* joins ideal point to nadir point (Eq. 5)
        // r* = nadir direction in normalised space = (1,1,...,1)/sqrt(m)
        std::vector<double> rstar(m, 1.0/std::sqrt(static_cast<double>(m)));

        std::vector<int> assoc(pool, 0);  // RV association for each x in St
        std::vector<double> d1(pool, 0.0);
        auto fstar = [&](const std::vector<double>& fp) -> std::vector<double> {
            double e = std::max(edi(fp), 1e-14);
            std::vector<double> fs(m); for(int j=0;j<m;++j) fs[j]=fp[j]/e; return fs;
        };

        for (int v : St) {
            auto fp  = f_prime(v);
            auto fs  = fstar(fp);
            // Association to closest RV
            double best_cos = -std::numeric_limits<double>::max();
            for (int r=0; r<N; ++r) {
                double dot=0.0; for(int j=0;j<m;++j) dot+=fs[j]*RVs[r][j];
                if(dot>best_cos){best_cos=dot; assoc[v]=r;}
            }
            vault.get_ind(v).ref_vector_idx = assoc[v];
            // d1(x, r*) = projection of f'(x) onto r* [Eq. 5]
            // = (f'(x) · r*) / ||r*||  but ||r*||=1 already
            double proj=0.0; for(int j=0;j<m;++j) proj+=fp[j]*rstar[j];
            d1[v] = proj;
            vault.get_ind(v).apd = proj;
        }

        // ── ESS: build N subsets SRV_i, sort by d1, assign Ic-rank ───────────
        // Each x gets an Ic rank = position in its subset sorted by d1 (lower=better)
        // Then divide all x into levels SIc_1, SIc_2, ... by Ic rank
        // Fill Pt+1: take SIc_1 + SIc_2 + ... until N; last level random
        // Ic rank of x = position in subset_i sorted ascending by d1[x]

        // Build subsets
        std::vector<std::vector<int>> subsets(N);
        for (int v : St) subsets[assoc[v]].push_back(v);

        // Sort each subset by d1 ascending
        for (auto& sub : subsets)
            std::sort(sub.begin(), sub.end(), [&](int a, int b){ return d1[a]<d1[b]; });

        // Ic rank: position within subset (0 = best)
        std::vector<int> ic_rank(pool, 0);
        for (auto& sub : subsets)
            for (int pos=0; pos<static_cast<int>(sub.size()); ++pos)
                ic_rank[sub[pos]] = pos;

        // Build levels: SIc_k = {x : ic_rank[x] == k-1}
        // Max rank = max subset size
        int max_rank = 0;
        for (int v : St) max_rank = std::max(max_rank, ic_rank[v]);

        std::vector<int> survivors;
        survivors.reserve(N);
        for (int rank=0; rank<=max_rank && static_cast<int>(survivors.size())<N; ++rank) {
            std::vector<int> level;
            for (int v : St) if (ic_rank[v]==rank) level.push_back(v);
            int remaining = N - static_cast<int>(survivors.size());
            if (static_cast<int>(level.size()) <= remaining) {
                for (int v : level) survivors.push_back(v);
            } else {
                // Random selection from this level (Algorithm 4, line 18)
                std::shuffle(level.begin(), level.end(), rng_);
                for (int i=0; i<remaining; ++i) survivors.push_back(level[i]);
            }
        }
        return survivors;
    }

    // ── Rearrange vault ──────────────────────────────────────────────────────
    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool), at_pos(pool);
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);
        for (int i=0; i<n; ++i) {
            int want=survivors[i], cur=pos[want];
            if(cur==i) continue;
            int other=at_pos[i];
            vault.swap_active(i, cur);
            pos[want]=i; pos[other]=cur;
            at_pos[i]=want; at_pos[cur]=other;
        }
        vault.reduce(n);
    }

public:
    SRVCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_    = p; }
    void set_t_max        (int t)    { t_max_ = t; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        gen_ = 0;
        init_V0(n, m);

        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int>     db(0,1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i=0; i<n; ++i) {
            for (int j=0;j<vault.vars_n();++j) {
                double lo=bounds[j].first.value_or(0.0), hi=bounds[j].second.value_or(1.0);
                vars[j]=lo+dr(rng_)*(hi-lo);
            }
            for (int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars);
            else vault.set_variables(i,vars);
        }
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(), m=vault.objs_n(); gen_=0;
        if(V0_.empty()) init_V0(n,m);
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();

        // ── Breed N offspring ─────────────────────────────────────────────────
        vault.expand(vault.pop_size());
        std::vector<int> perm(n); std::iota(perm.begin(),perm.end(),0);
        std::shuffle(perm.begin(), perm.end(), rng_);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;  // 1/n
        for (int i=0; i<n; i+=2) {
            int p1=perm[i], p2;
            if (i+1 < n) {
                p2 = perm[i+1];
            } else if (n >= 2) {
                // With an odd n the last "pair" was p1==p2, i.e. SBX of an
                // individual with its own clone. The partner is re-drawn at
                // random from the other n−1 individuals.
                std::uniform_int_distribution<int> dp(0, n-2);
                int r = dp(rng_);
                p2 = perm[(r >= i) ? r+1 : r];
            } else {
                p2 = p1;   // n==1: no partner available (degenerate case)
            }
            for(int j=0;j<vault.vars_n();++j){pv1[j]=vault.get_variable(p1,j);pv2[j]=vault.get_variable(p2,j);}
            ops::sbx(pv1,pv2,c1,c2,bounds,eta_c_,pc_,rng_);
            ops::polynomial_mutation(c1,bounds,eta_m_,pm,rng_);
            ops::polynomial_mutation(c2,bounds,eta_m_,pm,rng_);
            if (vault.bin_vars_n()>0) {
                std::vector<int> bv1(vault.bin_vars_n()),bv2(vault.bin_vars_n()),bc1,bc2;
                for(int j=0;j<vault.bin_vars_n();++j){bv1[j]=vault.get_bin_variable(p1,j);bv2[j]=vault.get_bin_variable(p2,j);}
                ops::binary_crossover(bv1,bv2,bc1,bc2,rng_);
                ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
                if(i+1<n) ops::bit_flip_mutation(bc2,vault.bin_vars_n(),rng_);
                vault.set_all_variables(n+i,c1,bc1);
                if(i+1<n) vault.set_all_variables(n+i+1,c2,bc2);
            } else {
                vault.set_variables(n+i,c1);
                if(i+1<n) vault.set_variables(n+i+1,c2);
            }
        }
        vault.sync();

        // ── Environmental selection over 2N pool ──────────────────────────────
        auto survivors = env_select(vault, n*2, n);
        rearrange(vault, survivors, n*2);
        ++gen_;
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SRV-MOEA/D — MOEA/D with Self-Guided Reference Vectors (SRV) as weights
// after: Liu, Lin, Wong, Coello Coello, Li, Ming, Zhang — IEEE TCYB, 2022
// doi:10.1109/TCYB.2020.2971638
//
// *** EXPERIMENTAL — NOT A TRANSCRIPTION OF THE PAPER ***
// Liu et al. 2022 embed SRV into generational MaOEAs of the APP+ESS class
// (NSGA-III, theta-DEA, EFR-RR — §III-C); steady-state MOEA/D (Zhang & Li 2007)
// is NOT covered by that paper. This file is an extrapolation "in the spirit
// of" it. Known risks:
//   * The SRV directions are substituted DIRECTLY as Tchebycheff weights
//     lambda. The optimum of a subproblem with weight lambda lies along
//     ~(1/lambda_1,...,1/lambda_m), so without the WS transformation
//     (lambda_j proportional to 1/d_j) the subproblem geometry is the mirror
//     image of the SRV intent, where the RVs follow the shape of the PF
//     (SRVM-3).
//   * Pc is the current population [0,N), because steady-state MOEA/D has no
//     2N pool. At k=N the ADM clustering degenerates into the identity — the
//     centroids are simply all N points — so the "SRVs" are approximately the
//     normalized population members (SRVM-4). The paper requires |Pc| >= N with
//     Pc = S_t of the merged pool.
//
// Generational scheme:
//   1. theta_c per Eq.17 (theta_c^min per Eq.16 from the Das-Dennis lattice,
//      using true angles — see srv_strategy.hpp);
//      W = SRVStrategy::extract(P, N, theta_c) every generation, followed by a
//      rebuild of the neighbourhoods B_i.
//   2. For each subproblem, in random order: the pool is B_i with probability
//      delta=0.9, otherwise the whole population; two parents; SBX + PM ->
//      offspring y.
//   3. Update z*; replace up to n_r=2 neighbours when
//      g^tch(y|lambda) <= g^tch(x|lambda), with
//      g^tch = max_j lambda_j·|f_j − z*_j|.
//
// DEFAULTS: T=20, delta=0.9, n_r=2; p_c=1.0, p_m=1/n, eta_c=eta_m=20 — the
//   MOEA/D conventions. The paper specifies none of these for this carrier;
//   for its own APP+ESS carriers it uses eta_c=30.
// DECLARED DEVIATIONS:
//   SRVM-A. G_max dependence (contract, not optional). The theta_c annealing
//     schedule (Eq.17) is driven by t_max_, which defaults to 1000. A caller
//     who never calls set_t_max runs a schedule computed against a budget that
//     has nothing to do with the real one. set_t_max is effectively mandatory.
//   SRVM-C. constraint_mode FEASIBILITY/CDP applies feasibility rules to the
//     Step-2.4 replacement test — the only preference relation this carrier
//     has. The SRV extraction and the neighbourhood rebuild are geometry.
//   SRVM-B. Step-2.4 replacement TRANSFERS the offspring's already computed
//     objectives (seed_individual). Re-evaluating instead would cost one extra
//     FE per replacement, i.e. N + (replacements) per generation instead of N.
//   (The substantive readings SRVM-1..SRVM-4 are stated in the status block and
//   the notable-fixes block above; this list exists so the file is not read as
//   claiming "no deviations".)
// EXTENSIONS BEYOND THE PAPER: the entire file (see the status above); binary
//   variables (off by default).
// ============================================================================
// Notable fixes: (1) SRVM-1 was a real bug — setup_seeded did not compute
// theta_min_c_, so theta_c(G=1) = 0 and rho(x) (Eq.9) evaluated 0/0 = NaN in
// the first generation; theta_min is now computed in init_W, the path COMMON
// to setup and setup_seeded (as in srv_nsga3.hpp); (2) theta_min comes from the
// corrected SRVStrategy::theta_min, which uses true angles; (3) the RNG seed is
// std::random_device plus set_seed, replacing time(nullptr); (4) explicit
// p_c/p_m in the current SBX/PM signatures; (5) the EXPERIMENTAL status and the
// SRVM-3/4 risks are stated in this header.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"
#include "srv_strategy.hpp"

namespace mootation {

template <typename Ind_t>
class SRVMOEADCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_   = 20.0;  // MOEA/D convention (beyond the paper, see header)
    double       eta_m_   = 20.0;  // MOEA/D convention
    double       pc_      = 1.0;   // MOEA/D convention: the SBX pair always crosses
    int          T_       = 20;    // neighbourhood size
    double       delta_   = 0.9;   // neighbourhood selection prob
    int          nr_      = 2;     // replacement limit
    int          t_max_   = 1000;
    std::mt19937 rng_{std::random_device{}()};
    int          current_gen_ = 0;

    SRVStrategy<Ind_t> srv_;
    double theta_min_c_ = 0.0;
    std::vector<std::vector<double>> W_;      // weight vectors (SRV centroids)
    std::vector<std::vector<int>>    B_;      // neighbourhood
    std::vector<double>              ideal_;

    // ── Das-Dennis init (warm start / fallback) ────────────────────────────
    void init_W(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
        // SRVM-1: theta_c^min (Eq.16) is computed HERE, the path common to
        // setup and setup_seeded. Computing it only in setup left the seeded
        // path with theta_c(G=1) = 0, hence 0/0 = NaN in rho(x). theta_min()
        // uses true angles (srv_strategy.hpp); Eq.16 is defined on the
        // ORIGINAL RVs.
        theta_min_c_ = SRVStrategy<Ind_t>::theta_min(W_);
    }


    // ── Neighbourhood (Euclidean distance between weight vectors) ──────────
    void build_neighbourhood(int n) {
        int T=std::min(T_,n);
        B_.resize(n);
        for (int i=0;i<n;++i) {
            std::vector<std::pair<double,int>> dists;
            dists.reserve(n);
            for (int j=0;j<n;++j) {
                double d=0.0;
                for (std::size_t k=0;k<W_[i].size();++k) {
                    double diff=W_[i][k]-W_[j][k]; d+=diff*diff;
                }
                dists.emplace_back(d,j);
            }
            std::partial_sort(dists.begin(),dists.begin()+T,dists.end());
            B_[i].resize(T);
            for(int k=0;k<T;++k) B_[i][k]=dists[k].second;
        }
    }

    // ── Tchebycheff scalarisation ──────────────────────────────────────────
    double tche(const std::vector<double>& f, const std::vector<double>& w) const {
        double g=0.0;
        for (std::size_t j=0;j<f.size();++j) g=std::max(g,w[j]*std::abs(f[j]-ideal_[j]));
        return g;
    }

public:
    SRVMOEADCore() = default;

    void set_eta_crossover(double e) { eta_c_=e; }
    void set_eta_mutation (double e) { eta_m_=e; }
    void set_pc           (double p) { pc_  =p; }
    void set_T            (int t)    { T_=t; }
    void set_delta        (double d) { delta_=d; }
    void set_nr           (int r)    { nr_=r; }
    void set_t_max        (int t)    { t_max_=t; }
    void set_seed         (unsigned s){ rng_.seed(s); }
    // SRVStrategy has no kmeans_iter field: the iteration count of the
    // modified k-means is fixed at 2m by Algorithm 3 of Liu et al. 2022.
    // An earlier version referenced a non-existent srv_.kmeans_iter, which
    // failed to compile when called. This is now a safe no-op that keeps the
    // API surface.
    void set_kmeans_iter  (int /*k*/) { /* iterations fixed at 2m (paper) */ }

    void setup(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(), m=vault.objs_n();
        current_gen_=0;
        init_W(n,m);   // also computes theta_min_c_ (Eq.16)
        build_neighbourhood(n);
        ideal_.assign(m,std::numeric_limits<double>::max());
        const auto& bounds=vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int> db(0,1);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bvars(vault.bin_vars_n());
        for(int i=0;i<n;++i){
            for(int j=0;j<vault.vars_n();++j){
                double lo=bounds[j].first.value_or(0.0), hi=bounds[j].second.value_or(1.0);
                vars[j]=lo+dr(rng_)*(hi-lo);
            }
            for(int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars);
            else vault.set_variables(i,vars);
        }
        vault.sync();
        for(int i=0;i<n;++i){
            const auto& o=vault.objectives_of(i);
            for(int j=0;j<m;++j) ideal_[j]=std::min(ideal_[j],o[j]);
        }
        vault.expand(1); vault.reduce(n+1);   // scratch slot at index n
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(), m=vault.objs_n();
        current_gen_=0;
        // SRVM-1, as in srv_nsga3.hpp: init_W also computes theta_min_c_.
        // The seeded path used to leave theta_min = 0, giving theta_c(G=1) = 0
        // and 0/0 = NaN in rho(x) in the first generation. When W_ carries over
        // from a previous run (SRV centroids), theta_min is recomputed from a
        // fresh Das-Dennis lattice, since Eq.16 is defined on the ORIGINAL
        // RVs.
        if (W_.empty() || theta_min_c_ <= 0.0) {
            init_W(n,m);
            build_neighbourhood(n);
        }
        ideal_.assign(m,std::numeric_limits<double>::max());
        for(int i=0;i<n;++i){
            const auto& o=vault.objectives_of(i);
            for(int j=0;j<m;++j) ideal_[j]=std::min(ideal_[j],o[j]);
        }
        vault.expand(1); vault.reduce(n+1);
    }

    void step(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(); int m=vault.objs_n();
        const auto& bounds=vault.get_bounds();
        int scratch=n;

        // Liu et al. 2022, §III-C: the RVs are adjusted EVERY generation, not
        // once per 10% of t_max. The source Pc is the current population [0,n):
        // steady-state MOEA/D has no merged 2N pool, since an offspring
        // replaces neighbours immediately, and [0,n) is the maintained elite
        // set — the Pc = U variant the paper allows for MaOEA/Ds.
        {
            double tc = SRVStrategy<Ind_t>::compute_theta_c(theta_min_c_, current_gen_+1, t_max_);
            W_=srv_.extract(vault,n,tc);
            build_neighbourhood(n);
        }

        std::vector<int> order(n); std::iota(order.begin(),order.end(),0);
        std::shuffle(order.begin(),order.end(),rng_);

        std::uniform_real_distribution<double> U01(0.0,1.0);
        std::vector<double> pv1(vault.vars_n()),pv2(vault.vars_n()),c1,c2;
        // MOEA/D convention (beyond the paper): p_c = 1.0, p_m = 1/n.
        double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;

        for (int idx : order) {
            const auto& Bi=B_[idx];
            std::vector<int> pool;
            if (U01(rng_)<delta_) pool=Bi;
            else { pool.resize(n); std::iota(pool.begin(),pool.end(),0); }

            std::uniform_int_distribution<int> dp(0,(int)pool.size()-1);
            int pa=pool[dp(rng_)], pb;
            for(int tries=0;tries<10;++tries){pb=pool[dp(rng_)];if(pb!=pa) break;}

            for(int j=0;j<vault.vars_n();++j){
                pv1[j]=vault.get_variable(pa,j); pv2[j]=vault.get_variable(pb,j);
            }
            ops::sbx(pv1,pv2,c1,c2,bounds,eta_c_,pc_,rng_);
            ops::polynomial_mutation(c1,bounds,eta_m_,pm,rng_);

            if(vault.bin_vars_n()>0){
                std::vector<int> bv1(vault.bin_vars_n()),bv2(vault.bin_vars_n()),bc1,bc2;
                for(int j=0;j<vault.bin_vars_n();++j){
                    bv1[j]=vault.get_bin_variable(pa,j); bv2[j]=vault.get_bin_variable(pb,j);
                }
                ops::binary_crossover(bv1,bv2,bc1,bc2,rng_);
                ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
                vault.set_all_variables(scratch,c1,bc1);
            } else {
                vault.set_variables(scratch,c1);
            }
            vault.refresh_objectives(scratch);

            const auto& fy=vault.objectives_of(scratch);
            for(int j=0;j<m;++j) ideal_[j]=std::min(ideal_[j],fy[j]);

            std::vector<int> B_order=Bi;
            std::shuffle(B_order.begin(),B_order.end(),rng_);
            int cnt=0;
            for(int nb:B_order){
                if(cnt>=nr_) break;
                double gy=tche(fy,W_[nb]);
                double gx=tche(vault.objectives_of(nb),W_[nb]);
                // Step 2.4 replacement. Under FEASIBILITY/CDP the comparison is
                // feasibility-first (SRVM-C): a feasible incumbent is never
                // replaced by an infeasible offspring, and two infeasible
                // solutions compare by CV.
                bool take;
                if(constraint_mode!=ConstraintMode::NONE){
                    double cvy=vault.get_cv(scratch), cvx=vault.get_cv(nb);
                    take = detail::better_scalar(constraint_mode,gy,cvy,gx,cvx)
                           || (cvy<=0.0 && cvx<=0.0 && gy<=gx);
                } else {
                    take = (gy<=gx);
                }
                if(take){
                    // Step 2.4 transfers the ALREADY COMPUTED objectives of the
                    // offspring; it does not re-evaluate. set_variables would
                    // mark the slot dirty and the following refresh_objectives
                    // would spend one extra FE per replacement, so a generation
                    // would cost N + (number of replacements) instead of N.
                    // Same pattern as moead.hpp.
                    vault.seed_individual(nb,
                                          vault.variables_of(scratch),
                                          vault.objectives_of(scratch),
                                          vault.binary_variables_of(scratch),
                                          vault.limits_of(scratch));
                    ++cnt;
                }
            }
        }
        ++current_gen_;
    }
};

} // namespace mootation

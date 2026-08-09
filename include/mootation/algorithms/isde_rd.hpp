#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ISDE+RD — Evolutionary Multi/Many-objective Algorithm based on I_SDE^+ and
// Region Decomposition.
// Zixian Lin, Hailin Liu, Fangqing Gu — School of Applied Mathematics,
//   Guangdong University of Technology, Guangzhou, China. 14th International
//   Conference on Computational Intelligence and Security (CIS), 2018.
// doi:10.1109/CIS2018.2018.00015
//
// IDEA (a per-subregion indicator, but a DIRECTED one). M2M framework: K
// subregions defined by direction vectors v_k (acute angle), S_k = floor(N/K).
// Within a subregion, selection uses the I_SDE^+ indicator DIRECTED by v_k
// through the weighted sum WS_k.
//
// I_SDE^+ (Eq.5-9): WS_k(p)=Σ v_k^i·f_i(p); P_WS={q∈P_k: WS_k(q)<WS_k(p)};
//   shift q'_j = max(p_j, q_j); I_SDE^+(p) = min_{q in P_WS} d(p,q');
//   an empty P_WS gives +inf.
//
// DEFAULTS: K=10 (a free parameter — the paper does NOT report the
//   experimental number of subregions, see ISDE-2), intra-subregion mating
//   probability 0.7; S_k = floor(N/K); SBX eta_c=20 / pc=1; PM eta_m=20 /
//   pm=1/n.
//
// DECLARED DEVIATIONS:
//   ISDE-1 (MINOR). The SDE shift is q'_j = max(p_j, q_j); d is Euclidean and
//     translation-invariant.
//   ISDE-2 (ARBITRATED).
//     Direction vectors V. The paper does NOT specify how to generate them:
//     §II-B delegates to the "decomposition strategy of MOEA/D-M2M [4]", and
//     Alg.3 takes "V: K unit direction vectors" as an INPUT. The "K" row of
//     Table II (4/6/10/7/9 for m=2/3/6/8/10) is the POSITION parameter of the
//     WFG problems — the table's own footnote says "K - Position vector,
//     L - Distance vector", D = K+L, and every value is a multiple of m−1 as
//     WFG requires — and NOT the algorithm's subregion count. The experimental
//     number of subregions is never reported, so there is nothing to reproduce.
//     Arbitration: the directions are Das–Dennis (das_dennis::generate_auto)
//     with L2 normalization.
//     NOTE: for an unattainable K_req_, generate_auto returns the NEAREST
//     attainable lattice size FROM ABOVE, so the actual K_ equals that lattice
//     size (for example m=6: 10 -> 21; m=8: 7 -> 8). The substitution of K is
//     declared LOUDLY here; it used to be silent.
//     This differs from the two other M2M ports in this library, and the
//     difference is deliberate rather than forced. An exact-K generator is
//     available — detail::uniform_sphere_directions(m, K) in
//     detail/sphere_directions.hpp (Das–Dennis candidates -> FPS by angle ->
//     Riesz s-energy repulsion, no RNG) — and moead_m2m and sms_m2m both REFUSE
//     the round-up, falling back to it whenever the lattice misses K.
//     RESOLVED (2026-08-08). The default stays the lattice: MOEA/D-M2M's own
//     decomposition — the thing §II-B delegates to — is a lattice one, and the
//     paper reports no K to reproduce, so switching the default would replace
//     a traceable choice with an untraceable one. The cost is real and now
//     escapable: at m=6 a requested K=10 becomes K=21, halving S_k = floor(N/K)
//     and with it the per-subregion pool, so set_exact_directions(true) routes
//     the build through the shared generator and honours K_req_ exactly. It
//     draws no random numbers, so flipping it moves the geometry without
//     shifting the RNG stream — the two settings are comparable run to run.
//     The lattice path also warns loudly when it substitutes K; it used to be
//     silent, and setup_seeded used to skip the warning entirely (both entry
//     points now share build_directions).
//   ISDE-3 (DEVIATION). The angular ASSOCIATION is computed in the shifted
//     space (f − z*), where z* is the running ideal over the pool. Without the
//     shift, negative objectives (ZDT3 has f2 < 0) push points into another
//     quadrant, the angular division goes wrong and segments are lost. The WS
//     ordering and the SDE distance are translation-invariant and need no
//     shift. Verified: ZDT3 spread 0.42 -> 0.84.
//   ISDE-4 (MINOR). Above S, removal proceeds one at a time with I_SDE^+
//     recomputed; P_WS = empty gives +inf.
//   ISDE-5 (MINOR). One offspring per member (SBX, first child); real-valued
//     genome, binary out of scope.
//   ISDE-6 (MINOR, consequence of the sizing formula). The active population
//     after each step() has exactly K_*S_ members. S_k = floor(N/K) is the
//     paper's own formula, but Alg.1's Output and Alg.3 line 17 both promise N
//     individuals, and the two disagree whenever K does not divide N. The
//     answer set is then SMALLER than pop_size (m=6, pop_size=132 -> K_=21,
//     S_=6, so 126 are returned) — and LARGER when the substituted K_ exceeds
//     pop_size, because the S_<1 -> 1 clamp forces one member per subregion.
//   ISDE-7 (MINOR). Intra-subregion mating retries up to 5 times to avoid
//     q == p; Alg.3 line 8 says only "randomly choose q from P_k" and imposes
//     no such condition. Two effects: at |P_k| = 2 the paper self-mates half
//     the time while this code almost never does, and the retry loop consumes
//     a data-dependent number of RNG draws (1 to 6 per mating), so the stream
//     position is not the same as a single unconditional draw. Kept rather
//     than deleted — removing it would itself change the stream.
//
// CONSTRAINTS (beyond the paper, off by default). ISDE+RD's only preference
//   relation is the per-subregion truncation by I_SDE^+ (smallest indicator
//   value is removed first). constraint_mode FEASIBILITY/CDP makes that
//   feasibility-first (ISDE-C): infeasible members are removed before any
//   feasible one, worst CV first, and only then does the indicator decide. The
//   WS ordering, the SDE shift and the angular association are untouched. The
//   paper is unconstrained.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../detail/sphere_directions.hpp"
#include "../data_vault.hpp"
#include "../warn.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class ISDERDCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int    K_req_=10;
    bool   exact_K_=false;   // ISDE-2: opt out of the Das-Dennis round-up
    double inprob_=0.7;
    double eta_c_=20.0, eta_m_=20.0, pc_=1.0, pm_=-1.0;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };
    std::vector<std::vector<double>> V_;
    std::vector<std::vector<Sol>> subpop_;
    int K_=0, m_=0, N_=0, S_=0;
    std::vector<double> z_;   // running ideal (ISDE-3): shift for the angular association

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }
    static double cosang(const std::vector<double>& a, const std::vector<double>& b){
        double d=0,na=0,nb=0; for(std::size_t i=0;i<a.size();++i){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
        double q=std::sqrt(na)*std::sqrt(nb); if(q<1e-300) return 1.0; return std::clamp(d/q,-1.0,1.0);
    }
    static std::vector<double> unit(std::vector<double> f){ double n=0; for(double v:f) n+=v*v; n=std::sqrt(std::max(n,1e-300)); for(double&v:f) v/=n; return f; }
    int assoc(const std::vector<double>& f) const {
        // ISDE-3: the angular association runs in the shifted space (f−z*);
        // otherwise negative objectives (ZDT3 f2 < 0) break the angular
        // division around the origin.
        std::vector<double> s(m_); for(int k=0;k<m_;++k) s[k]= z_.empty()? f[k] : f[k]-z_[k];
        int best=0; double bc=cosang(s,V_[0]); for(int i=1;i<K_;++i){double c=cosang(s,V_[i]); if(c>bc){bc=c;best=i;}} return best;
    }
    double ws(const std::vector<double>& f, int k) const { double s=0; for(int i=0;i<m_;++i) s+=V_[k][i]*f[i]; return s; }
    static double dist(const std::vector<double>& p, const std::vector<double>& q){
        double s=0; for(std::size_t j=0;j<p.size();++j){ double qq=std::max(p[j],q[j]); double d=p[j]-qq; s+=d*d; } return std::sqrt(s);
    }
    double isdeplus(int idx, const std::vector<Sol>& G, int k) const {
        double wp=ws(G[idx].objs,k); double best=std::numeric_limits<double>::infinity();
        for(int j=0;j<(int)G.size();++j){ if(j==idx) continue; if(ws(G[j].objs,k)<wp){ double d=dist(G[idx].objs,G[j].objs); if(d<best) best=d; } }
        return best;
    }

    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch){
        const auto& b=vault.get_bounds(); int nv=vault.vars_n();
        std::vector<double> c1,c2;
        ops::sbx(x.vars,y.vars,c1,c2,b,eta_c_,pc_,rng_);
        ops::polynomial_mutation(c1,b,eta_m_,pm_eff(nv),rng_);
        Sol z; z.vars=c1; vault.set_variables(scratch,c1); vault.refresh_objectives(scratch); z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        return z;
    }
    std::vector<Sol> flat() const { std::vector<Sol> P; for(auto&s:subpop_) for(auto&x:s) P.push_back(x); return P; }
    void store_arch(DataVault<Ind_t>& vault){ auto P=flat(); vault.reduce(0); vault.expand((int)P.size());
        for(int i=0;i<(int)P.size();++i) vault.seed_individual((std::size_t)i,P[i].vars,P[i].objs,{},{}); }

    void allocate(const std::vector<Sol>& pool){
        // running ideal over the pool (the shift for the angular association)
        z_.assign(m_,std::numeric_limits<double>::max());
        for(const auto& s:pool) for(int k=0;k<m_;++k) z_[k]=std::min(z_[k],s.objs[k]);
        std::vector<std::vector<int>> bk(K_);
        for(int i=0;i<(int)pool.size();++i) bk[assoc(pool[i].objs)].push_back(i);
        std::uniform_int_distribution<int> dp(0,(int)pool.size()-1);
        subpop_.assign(K_,{});
        for(int k=0;k<K_;++k){
            std::vector<Sol> G; for(int i:bk[k]) G.push_back(pool[i]);
            if((int)G.size()==S_){ subpop_[k]=G; }
            else if((int)G.size()<S_){ subpop_[k]=G; int need=S_-(int)G.size(); for(int t=0;t<need;++t) subpop_[k].push_back(pool[dp(rng_)]); }
            else { // above S: remove one at a time by I_SDE^+
                std::vector<int> alive(G.size()); std::iota(alive.begin(),alive.end(),0);
                while((int)alive.size()>S_){
                    std::vector<Sol> cur; for(int a:alive) cur.push_back(G[a]);
                    int worst=-1; double wv=std::numeric_limits<double>::infinity();
                    // Worst by I_SDE^+ — smaller is worse. Under
                    // FEASIBILITY/CDP an infeasible member is worse than any
                    // feasible one and infeasible members are ordered by CV
                    // (ISDE-C), so the removal loop drains infeasibility first.
                    double wcv=-1.0; bool inf_seen=false;
                    for(int a=0;a<(int)alive.size();++a){
                        const Sol& sa=cur[a];
                        if(constraint_mode!=ConstraintMode::NONE && sa.cv>0.0){
                            if(!inf_seen || sa.cv>wcv){ inf_seen=true; wcv=sa.cv; worst=a; }
                            continue;
                        }
                        if(inf_seen) continue;
                        double v=isdeplus(a,cur,k); if(v<wv){wv=v;worst=a;}
                    }
                    if(worst<0) worst=0;
                    alive.erase(alive.begin()+worst);
                }
                for(int a:alive) subpop_[k].push_back(G[a]);
            }
        }
    }

public:
    ISDERDCore() = default;
    void set_K(int k){ K_req_=k; }
    // ISDE-2. false (default): K is rounded UP to the nearest attainable
    // Das-Dennis lattice size, matching MOEA/D-M2M's lattice decomposition,
    // which is what §II-B of the paper delegates to. true: K_req_ is honoured
    // exactly via detail::uniform_sphere_directions, matching moead_m2m and
    // sms_m2m. The generator draws no random numbers, so the switch changes
    // the geometry without shifting the RNG stream.
    void set_exact_directions(bool b){ exact_K_=b; }
    void set_n_clusters(int k){ K_req_=k; }
    void set_inprob(double p){ inprob_=p; }
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double p){ pc_=p; }
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    // ISDE-2: the actual K_ equals the Das-Dennis lattice size >= K_req_ unless
    // exact_K_ is set, in which case the deterministic sphere generator hits
    // K_req_ exactly. Shared by setup and setup_seeded so the diagnostic and
    // the geometry cannot drift apart between the two entry points.
    void build_directions(){
        auto Vr=das_dennis::generate_auto(m_,K_req_);
        if((int)Vr.size()!=K_req_){
            if(exact_K_){
                Vr=detail::uniform_sphere_directions(m_,K_req_);
            } else {
                warn_lazy([&]{ return "isde_rd: K=" + std::to_string(K_req_) +
                               " is not an attainable Das-Dennis lattice size for m=" +
                               std::to_string(m_) + "; using K=" +
                               std::to_string(Vr.size()) +
                               " instead, which changes S_k=floor(N/K) (see ISDE-2)."
                               " set_exact_directions(true) hits K exactly"; });
            }
        }
        V_.clear(); for(auto&v:Vr) V_.push_back(unit(v)); K_=(int)V_.size();
        S_=N_/K_; if(S_<1) S_=1;
    }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size();
        build_directions();
        const auto& bd=vault.get_bounds(); std::uniform_real_distribution<double> d(0.0,1.0);
        std::vector<double> vars(vault.vars_n());
        for(int i=0;i<N_;++i){ for(int j=0;j<vault.vars_n();++j){double lo=bd[j].first.value_or(0.0),hi=bd[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);} vault.set_variables(i,vars);}
        vault.sync();
        std::vector<Sol> P; for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            P.push_back(s);}
        allocate(P);
    }
    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size();
        build_directions();
        std::vector<Sol> P; for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); P.push_back(s);}
        allocate(P);
    }

    void step(DataVault<Ind_t>& vault){
        int scratch=vault.expand(1);
        std::uniform_real_distribution<double> uni(0,1);
        std::vector<Sol> flatP=flat();
        std::uniform_int_distribution<int> dall(0,(int)flatP.size()-1);
        std::vector<Sol> O;
        for(int k=0;k<K_;++k){
            auto& Pk=subpop_[k]; int sz=(int)Pk.size(); if(sz==0) continue;
            for(int j=0;j<sz;++j){
                Sol q;
                if(uni(rng_)<inprob_ && sz>1){ int qq=std::uniform_int_distribution<int>(0,sz-1)(rng_); for(int a=0;a<5&&qq==j;++a) qq=std::uniform_int_distribution<int>(0,sz-1)(rng_); q=Pk[qq]; }
                else q=flatP[dall(rng_)];
                O.push_back(breed(Pk[j],q,vault,scratch));
            }
        }
        std::vector<Sol> merged=flatP; for(auto&s:O) merged.push_back(s);
        allocate(merged);
        store_arch(vault);
    }
};

} // namespace mootation

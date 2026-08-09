#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// RD-EMO — A Region Division based Decomposition approach for Evolutionary
// Many-objective Optimization.
// R. Liu, J. Liu, R. Zhou, C. Lian, R. Bian — Knowledge-Based Systems (2020),
// article 105518.
// doi:10.1016/j.knosys.2020.105518
//
// IDEA (region-oriented slot allocation; cf. NAEMO). The objective space is
// divided into regions by reference lines (Das–Dennis). Attributes: the region
// degree RD_i (the number of solutions in the region) and the region sparse
// rate RSR_i. Mating is biased toward sparse regions; the update (Alg.2) has an
// offspring evict the WORST by PBI from the DENSEST region, or from its own
// region if that one is no sparser.
//
// PBI (Eq.2): g=d1+θ·d2, d1=((F−z*)·λ)/‖λ‖, d2=‖(F−z*)−d1·λ/‖λ‖‖.
//
// PAPER DEFAULTS (§4.1, Table 4): SBX eta_c=30 / p_c=0.9; PM eta_m=20 /
//   p_m=1/n; theta=5; T=20, W=T/4 (§3.2: "W = T/4 in our experiments",
//   so W is derived, not independent — set_T rescales it and there is no
//   set_W);
//   popsiz=pop_size; ref points=pop_size.
//
// DECLARED DEVIATIONS:
//   RDE-1 (NOT A DEVIATION — kept as a note). θ = 5 IS stated: §4.1 says
//     "θ = 5 suggested in MOEA/D is set in RD-EMO". Settable via set_theta.
//   RDE-2 (MINOR). A region is argmin d2 to a reference line, i.e. the
//     "nearest reference line".
//   RDE-3 (DEVIATION — filling a hole, not overriding a value). z* is the
//     RUNNING ideal: the per-objective minimum over the population. The paper
//     never says how z* is obtained. §3.1 introduces it only as "an ideal point
//     z*" inside Eq.2, and Fig.1/Fig.2 draw every reference line as emanating
//     from the axis origin, which is where the origin reading comes from — but
//     no section, §4.1 included, states it. The suite the paper runs (DTLZ,
//     WFG) has f ≥ 0 throughout, so the distinction never surfaces there. It
//     surfaces immediately outside it: ZDT3 has f2 < 0, and with z* pinned at
//     the origin those points fall into a different orthant, the angular region
//     division misassigns them and PF segments are lost. The running ideal
//     coincides with the origin whenever f ≥ 0 and the minimum is attained at
//     0, and stays well-defined otherwise. Verified: ZDT3 spread 0.43 -> 0.84.
//   RDE-4 (MINOR). The distance between regions is the Euclidean distance
//     between reference points; T and W are truncated to n−1.
//   RDE-5 (MINOR). Reference points = Das–Dennis(pop_size) via generate_auto,
//     with automatic layering.
//   RDE-6 (MINOR). Alg.1 line 20 applies SBX ONCE per pair and takes the
//     complementary pair (x_new, x''_new). This port calls ops::sbx twice on
//     the same pair, keeping c1 from the first call and c2 from the second.
//     Per-child marginals are unchanged (sbx swaps the two output positions
//     with probability 1/2), but the siblings are decorrelated and the RNG
//     consumption per pair is doubled: with p_c=0.9 the gate is drawn twice, so
//     paired application makes both children parent copies together with
//     probability 0.1, whereas split application makes exactly one a copy with
//     probability 0.18. Both children are consumed sequentially by the Alg.2
//     eviction loop, so it is the joint law that drives the trajectory. Not
//     collapsed to one call: that would itself change the RNG stream.
//   RDE-7 (CONSTRAINT, documented for callers). §3.3 requires an EVEN popsiz —
//     "population size is set to the number of reference points; otherwise ...
//     plus one" — because SBX produces offspring in pairs. pop_size comes from
//     the caller and is neither rounded nor rejected here, so an odd pop_size
//     breeds pop_size−1 offspring and the per-generation FE budget is
//     pop_size−1. Not auto-corrected on purpose: rounding pop_size would move a
//     function-evaluation budget that a benchmark holds fixed (same rule as
//     sms_m2m.hpp).
//
// NOTE (not a deviation). Algorithm 3 line 24 increments RSR_n, which read
//   literally would make RSR_1 = 0 in the paper's own Fig.2 example. The §3.2
//   prose and its worked example fix the subscript as RSR_j: with
//   RD = (0,1,3,1,1) and W = 2 the paper states RSR = (2,1,2,2,2), which this
//   code reproduces exactly. The Algorithm-3 subscript is a typo.
//
// CONSTRAINTS (beyond the paper, off by default). RD-EMO's only preference
//   relation is the Alg.2 eviction by PBI, so constraint_mode FEASIBILITY/CDP
//   applies feasibility-first there: the evicted member is the infeasible one
//   with the largest CV if any exist, otherwise the largest PBI as in the
//   paper; and an infeasible offspring can never evict a feasible incumbent.
//   The region division itself is untouched. The paper is unconstrained.
// EXTENSIONS BEYOND THE PAPER (off by default): binary genome.
// ============================================================================

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

namespace mootation {

template <typename Ind_t>
class RDEMOCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int    T_ = 20, W_ = 5;
    double theta_ = 5.0;
    double eta_c_ = 30.0, eta_m_ = 20.0, pc_ = 0.9, pm_ = -1.0;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; std::vector<int> bvars; double cv=0.0; };

    std::vector<std::vector<double>> R_;          // [N] reference points (lambda directions)
    std::vector<double>              Rnorm_;      // ‖λ_i‖
    std::vector<std::vector<int>>    U_;          // [N] the T nearest regions
    std::vector<std::vector<int>>    V_;          // [N] the W nearest regions
    std::vector<Sol>                 pop_;
    std::vector<int>                 region_of_;
    int N_=0, m_=0, popsiz_=0;
    std::vector<double> z_;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }

    double d2_to(const std::vector<double>& f, int i) const {
        double dot=0; for(int k=0;k<m_;++k) dot+=(f[k]-z_[k])*R_[i][k];
        double d1=dot/Rnorm_[i], s2=0;
        for(int k=0;k<m_;++k){ double pr=(f[k]-z_[k])-d1*R_[i][k]/Rnorm_[i]; s2+=pr*pr; }
        return std::sqrt(std::max(s2,0.0));
    }
    double pbi(const std::vector<double>& f, int i) const {
        double dot=0; for(int k=0;k<m_;++k) dot+=(f[k]-z_[k])*R_[i][k];
        double d1=dot/Rnorm_[i], s2=0;
        for(int k=0;k<m_;++k){ double pr=(f[k]-z_[k])-d1*R_[i][k]/Rnorm_[i]; s2+=pr*pr; }
        return d1 + theta_*std::sqrt(std::max(s2,0.0));
    }
    int assoc(const std::vector<double>& f) const {
        int best=0; double bd=d2_to(f,0);
        for(int i=1;i<N_;++i){ double d=d2_to(f,i); if(d<bd){bd=d;best=i;} }
        return best;
    }
    // RDE-3: z* is the running ideal (minimum over the population).
    void upd_ideal_pop(){
        for(const auto& s:pop_) for(int k=0;k<m_;++k) z_[k]=std::min(z_[k],s.objs[k]);
    }

    void build_regions(int m){
        R_=das_dennis::generate_auto(m, popsiz_);
        N_=(int)R_.size();
        Rnorm_.assign(N_,0.0);
        for(int i=0;i<N_;++i){ double s=0; for(double v:R_[i]) s+=v*v; Rnorm_[i]=std::sqrt(std::max(s,1e-300)); }
        int Te=std::min(T_,N_-1), We=std::min(W_,N_-1);
        if(Te<1) Te=1;
        if(We<1) We=1;
        U_.assign(N_,{}); V_.assign(N_,{});
        for(int i=0;i<N_;++i){
            std::vector<std::pair<double,int>> d;
            for(int j=0;j<N_;++j){ if(j==i) continue;
                double s=0; for(int k=0;k<m;++k){double df=R_[i][k]-R_[j][k];s+=df*df;}
                d.emplace_back(std::sqrt(s),j); }
            std::sort(d.begin(),d.end());
            for(int t=0;t<Te;++t) U_[i].push_back(d[t].second);
            for(int t=0;t<We;++t) V_[i].push_back(d[t].second);
        }
    }

    std::vector<int> region_degrees() const {
        std::vector<int> RD(N_,0);
        for(int r:region_of_) RD[r]++;
        return RD;
    }

    Sol breed_child(const Sol& x, const Sol& y, bool first,
                    DataVault<Ind_t>& vault, int scratch){
        const auto& b=vault.get_bounds(); int nv=vault.vars_n();
        std::vector<double> c1,c2;
        ops::sbx(x.vars,y.vars,c1,c2,b,eta_c_,pc_,rng_);
        std::vector<double>& c = first ? c1 : c2;
        ops::polynomial_mutation(c,b,eta_m_,pm_eff(nv),rng_);
        Sol z; z.vars=c;
        if(vault.bin_vars_n()>0){
            std::vector<int> bc1,bc2;
            ops::binary_crossover(x.bvars,y.bvars,bc1,bc2,rng_);
            std::vector<int>& bc = first ? bc1 : bc2;
            ops::bit_flip_mutation(bc,vault.bin_vars_n(),rng_);
            z.bvars=bc; vault.set_all_variables(scratch,c,bc);
        } else {
            vault.set_variables(scratch,c);
        }
        vault.refresh_objectives(scratch);
        z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        return z;
    }

    void store_arch(DataVault<Ind_t>& vault){
        vault.reduce(0); vault.expand((int)pop_.size());
        for(int i=0;i<(int)pop_.size();++i)
            vault.seed_individual((std::size_t)i,pop_[i].vars,pop_[i].objs,pop_[i].bvars,{});
    }

    // Worst member of a region: largest PBI (Alg.2). Under FEASIBILITY/CDP the
    // infeasible members are worst by definition, ordered by descending CV, so
    // an infeasible incumbent is always evicted before any feasible one.
    int worst_in_region(int reg) const {
        int worst=-1; double wg=-std::numeric_limits<double>::max();
        double wcv=-1.0; bool found_infeasible=false;
        const bool cm = (constraint_mode!=ConstraintMode::NONE);
        for(int i=0;i<(int)pop_.size();++i){
            if(region_of_[i]!=reg) continue;
            if(cm && pop_[i].cv>0.0){
                if(!found_infeasible || pop_[i].cv>wcv){ found_infeasible=true; wcv=pop_[i].cv; worst=i; }
                continue;
            }
            if(found_infeasible) continue;
            double g=pbi(pop_[i].objs,reg); if(g>wg){wg=g;worst=i;}
        }
        return worst;
    }
    void remove_at(int idx){
        int last=(int)pop_.size()-1;
        pop_[idx]=pop_[last]; region_of_[idx]=region_of_[last];
        pop_.pop_back(); region_of_.pop_back();
    }

public:
    RDEMOCore() = default;
    void set_T(int t){ if(t>0){T_=t;W_=std::max(1,t/4);} }
    void set_theta(double t){ if(t>0) theta_=t; }
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double p){ pc_=p; }
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); popsiz_=vault.pop_size();
        z_.assign(m_,std::numeric_limits<double>::max());  // z* = running ideal
        build_regions(m_);
        const auto& b=vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0,1.0);
        std::uniform_int_distribution<int> dbn(0,1);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bv(vault.bin_vars_n());
        for(int i=0;i<popsiz_;++i){
            for(int j=0;j<vault.vars_n();++j){double lo=b[j].first.value_or(0.0),hi=b[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);}
            for(int j=0;j<vault.bin_vars_n();++j) bv[j]=dbn(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bv); else vault.set_variables(i,vars);
        }
        vault.sync();
        pop_.clear(); region_of_.clear();
        for(int i=0;i<popsiz_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s); }
        upd_ideal_pop();
        for(auto& s:pop_) region_of_.push_back(assoc(s.objs));
    }

    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); popsiz_=vault.pop_size();
        z_.assign(m_,std::numeric_limits<double>::max());
        build_regions(m_);
        pop_.clear(); region_of_.clear();
        for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s); }
        upd_ideal_pop();
        for(auto& s:pop_) region_of_.push_back(assoc(s.objs));
    }

    void step(DataVault<Ind_t>& vault){
        int scratch=vault.expand(1);
        std::uniform_int_distribution<int> dp(0,(int)pop_.size()-1);

        // update the running ideal and re-associate (z* may have dropped)
        upd_ideal_pop();
        region_of_.clear();
        for(auto& s:pop_) region_of_.push_back(assoc(s.objs));

        // RD and RSR over the current population
        std::vector<int> RD=region_degrees();
        std::vector<int> RSR(N_,0);
        for(int i=0;i<N_;++i){ int c=0; for(int j:V_[i]) if(RD[j]>0) ++c; RSR[i]=c; }
        std::vector<std::vector<int>> members(N_);
        for(int i=0;i<(int)pop_.size();++i) members[region_of_[i]].push_back(i);

        // ── Algorithm 1: selection → Q ──
        std::vector<Sol> Q; Q.reserve(popsiz_);
        int pairs = popsiz_/2;
        for(int j=0;j<pairs;++j){
            int a=dp(rng_), b=dp(rng_);
            int na=region_of_[a], nb=region_of_[b];
            int win=a, nwin=na;
            if(RD[nb]<RD[na] || (RD[nb]==RD[na] && RSR[nb]<RSR[na])){ win=b; nwin=nb; }
            std::vector<int> cand;
            for(int rg:U_[nwin]) if(RD[rg]>0) cand.push_back(rg);
            int x2;
            if(!cand.empty()){
                int rg=cand[std::uniform_int_distribution<int>(0,(int)cand.size()-1)(rng_)];
                x2=members[rg][std::uniform_int_distribution<int>(0,(int)members[rg].size()-1)(rng_)];
            } else {
                x2=dp(rng_);
            }
            Q.push_back(breed_child(pop_[win],pop_[x2],true ,vault,scratch));
            Q.push_back(breed_child(pop_[win],pop_[x2],false,vault,scratch));
        }

        // ── Algorithm 2: update ──
        for(auto& q:Q){
            for(int kk=0;kk<m_;++kk) z_[kk]=std::min(z_[kk],q.objs[kk]);   // running ideal
            std::vector<int> rd=region_degrees();
            int n=0; for(int i=1;i<N_;++i) if(rd[i]>rd[n]) n=i;
            int k=assoc(q.objs);
            if(rd[k]==0 || rd[k]<rd[n]){
                int p=worst_in_region(n);
                // Feasibility-first: this branch would trade a member of the
                // densest region for the offspring purely on diversity grounds.
                // An infeasible offspring must not displace a feasible member,
                // so it is discarded instead. (Inert when constraint_mode=NONE.)
                if(constraint_mode!=ConstraintMode::NONE && q.cv>0.0 &&
                   p>=0 && pop_[p].cv<=0.0) continue;
                if(p>=0) remove_at(p);
                pop_.push_back(q); region_of_.push_back(k);
            } else {
                pop_.push_back(q); region_of_.push_back(k);
                int pp=worst_in_region(k);
                if(pp>=0) remove_at(pp);
            }
        }
        store_arch(vault);
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Liu & Gu 2011 — An Improved NSGA-II Algorithm Based on Sub-regional Search.
// H.-L. Liu, F. Gu — IEEE Congress on Evolutionary Computation (CEC) 2011,
// pp. 1906-1911.
// doi:10.1109/CEC.2011.5949848          (source: liu2011)
//
// IDEA (the earliest sub-regional ancestor of the M2M family). The objective
// space is divided A PRIORI into S = ⌈√N⌉ fixed subregions by uniform
// central vectors W^i (first octant). Each subregion stores:
//   • an internal set — the ≤ l_i best (NSGA-II: NDS + crowding distance);
//   • an external set — 5·l_i solutions, a "diversity reservoir".
// Mating: an internal individual with R=1 (non-dominated within its internal
// set) × a random one from the external set of the SAME subregion → NSGA-II
// selection. Non-directional (NDS), caps WITHOUT refill (internal may be
// < l_i). This is the point of contrast with M2M (equal-quota refill) and with
// the direction-based methods.
//
// SCHEME (Section II-D):
//   Step 1 Init: S, l_i=⌈N/S⌉, central vectors W; generate 5N random points,
//     distribute them over the subregions; internal = the ≤l_i best (NDS+CD),
//     external = 5·l_i random ones from the whole 5N pool.
//   Step 2 Mating: the set of internal solutions with R=1 (if |R1|<N — top up
//     with random internal ones to N); for each x^i (subregion t): x^j randomly
//     from external_t; crossover+mutation → 1 offspring; N offspring in total.
//   Step 3 Update: 3.1 distribute (N offspring ∪ all internal) over the
//     subregions; 3.2 internal_k = the ≤l_k best (NDS+CD); 3.3 the rejected
//     ones (beyond l_k) RANDOMLY replace the same number of members of
//     external_k.
//   Step 4 gen++; every T generations recompute R (R=1 ⟺ non-dominated
//     within the internal set).
//
// Shift/association (II-A): f_i ← f_i − f̄_i^gen (population minimum) → first
//   octant; subregion = the smallest angle to the central vectors W^i.
//
// DEFAULTS (§IV-A):
//   • N=pop_size; S=⌈√N⌉; l_i=⌈N/S⌉; external = 5·l_i; T=50.
//   • Operators — Liu & Li 2009 [1]. The reference is in §II-C: «We perform the
//     crossover and mutation used in [1] between x^i and x^j». §II-D Step 2
//     instead says «a crossover operator as equation(3) and a mutation operator
//     as equation(4)» — liu2011 contains no Eq.(3) or Eq.(4) at all (only
//     Eq.(1) and Eq.(2)), so the operator is resolved through §II-C.
//     What that gives: the annealed arithmetic crossover Eq.(5)
//     x̃ = xⁱ + rc·(xⁱ − xʲ) and the mutation Eq.(6) with the "≥1 component"
//     guarantee; the rc/rm step decays with gen/Max_gen (see
//     operators/liuli_crossover.hpp, including the arbitration notes
//     LL-1..LL-5 on the typos of paper [1]). P_m = 1/n ([1] §V).
//     FIX 2026-07-07 (source-fidelity review): previously SBX/PM stood here
//     with the FALSE reference "as in [1]" — there is no SBX/PM in liu2009.
//   • The annealing needs Max_gen: liu2011 §IV-A sets Max_gen=⌊(300000−5N)/N⌋
//     (300000 FE). In the library — set_t_max(int), default 1000 (the
//     moead_awa/adaw convention); the caller MUST set the real budget,
//     otherwise the annealing schedule does not match the paper, and for
//     gen>t_max the operator degenerates into copying (rc=rm=0).
//   • Stop by generations; the paper uses 300000 FE (= ⌊(300000−5N)/N⌋
//     generations, the same bracket reading as above — see LG-3b).
//
// DECLARED DEVIATIONS:
//   LG-1 (MINOR). R=1 = "non-dominated within its internal set" (the paper:
//     «non-dominated solutions within internal set», recomputed every T);
//     offspring are marked R=1 on entry.
//   LG-2 (MINOR). The NSGA-II selection of internal — canonical NDS + crowding
//     distance (the paper cites NSGA-II [6]); the boundary-front tie-break is
//     by CD.
//   LG-3 (DEVIATION — the paper contradicts itself, and this port breaks the
//     tie in the direction that overshoots). §II-B states a strict equality:
//     "l_i indicates the upper limit of i-th sub-region with Σ_{i=1}^S l_i = N".
//     §IV-A prescribes l_i = ⌈N/S⌉, uniform across sub-regions, which gives
//     Σ l_i = S·⌈N/S⌉ ≥ N — equal only when S divides N. The two cannot both
//     hold. This port follows §IV-A, so the returned population can exceed
//     pop_size: at N=91, m=3 we get S=⌈√91⌉=10, l_i=10, Σ l_i=100.
//     There is no "≈" anywhere in the paper; an earlier version of this entry
//     attributed one to it, which was false.
//   LG-3b (READING). §IV-A prints ceiling brackets for all three of S, l_i and
//     Max_gen and then defines the bracket, in the same sentence, as "the
//     largest integer of not greater than x" — i.e. floor. This port reads
//     them as true ceilings for S and l_i and as a floor for Max_gen. The
//     direction matters: under a floor reading of l_i, Σ l_i ≤ N and the answer
//     set never exceeds pop_size; under the ceiling reading it does (see LG-3).
//     Note that floor does not restore the §II-B equality either — at N=91,
//     S=9, floor(91/9)=10 gives Σ=90 ≠ 91.
//   LG-4 (MINOR). The central vectors W — a Das–Dennis lattice of S points
//     normalized to unit length (the paper: «uniformly distributed unit
//     vectors»).
//     NOTE: S = ⌈√N⌉ is generally NOT an attainable Das–Dennis lattice size.
//     generate_auto returns the nearest attainable size FROM ABOVE and the
//     effective S becomes that size — at m=5, N=100 a requested S=10 comes back
//     as 15, because n_vectors(5,H) runs 5, 15, 35, … The substitution then
//     propagates into l_i = ⌈N/S⌉ and into the external-set size 5·l_i, so all
//     three quantities in the DEFAULTS block above can differ from what a
//     reader computes from N alone.
//   LG-5 (MINOR). Shift by the minimum of the internal population; association
//     by the acute angle (cos) of the shifted f.
//   LG-6 (MINOR). External may contain duplicates (random
//     initialization/replacement — per the letter of the paper). (The former
//     part about "SBX yields 2 children" is retired: operator [1] produces
//     exactly one offspring.)
//   LG-6b (MINOR). Step 3.3 says "the remained individuals replace the same
//     number of external individuals randomly" without saying whether the
//     positions are drawn with or without replacement. This port draws them
//     i.i.d., so two rejected individuals can land in the same external slot
//     and FEWER than |rej| distinct members are displaced — the reservoir
//     turns over slightly more slowly than "the same number" suggests. Small
//     at the paper's settings (N=91, S=10, |external_k|=50, |rej|≈9 gives ≈0.7
//     collisions per region per generation, ≈8% of writes). Sampling distinct
//     slots via a partial shuffle would change the RNG draw count.
//   LG-7 (AMBIGUOUS→resolved). For |R1|>N (possible when Σl_i>N, LG-3) the
//     paper only says «all of these individuals take part in mating» and
//     «generate N new individuals», without specifying the order.
//     FIX 2026-07-07 (source-fidelity review): instead of the deterministic
//     truncation "the first N in region-major order" (it systematically cut
//     off the tail regions) — RANDOM truncation to N (shuffle).
//   ---------------------------------------------------------------------
//   OPEN — UNDER INVESTIGATION (noted 2026-08-04, convergence smoke suite).
//   ---------------------------------------------------------------------
//   On DTLZ2 (M=3, n=12, pop_size=91, 200 generations, Liu–Li Eq.(5)-(6) with
//   p_m = 1/n and t_max left at the class default of 1000 — NOT SBX/PM, which
//   this class does not use at all) this
//   implementation reports:
//       mean distance of the population to the true front  = 1.010
//       distance of the single best solution               = 0.003
//       final population size                              = 100
//   Every other algorithm in the suite lands at mean <= 0.25, and 55 of 60
//   land below 0.015.
//
//   Reading of the two numbers: best = 0.003 means the search DOES reach the
//   front, so this is not a convergence-speed artefact of the sub-regional
//   scheme. A population mean of 1.01 alongside it means most of the retained
//   population sits roughly one unit away from the front — i.e. survivors are
//   being kept that should not be. That points at environmental selection or
//   at the external set, not at variation.
//
//   The population size of 100 > 91 is explained and declared by LG-3/LG-3b
//   (l_i = ⌈N/S⌉ is uniform across sub-regions, so Σ l_i ≥ N — the paper's own
//   §II-B equality is violated by its own §IV-A formula) together with LG-7
//   (truncation when |R1| > N). Whether the overshoot and the mean are the same
//   defect or two separate ones is NOT yet established.
//
//   NARROWED 2026-08-06, by measurement:
//   (a) NOT an output-filtering artefact. Step 5 outputs "non-dominated
//       solutions" while store_arch presents the union of the internal sets,
//       which by design also holds locally-best-but-globally-dominated points.
//       Filtering the presented set to the non-dominated subset was the obvious
//       suspect. Measured: 84 of the 100 returned solutions are already
//       mutually non-dominated, and the filtered mean is 0.944 against 1.010 —
//       the anomaly survives the filter. Hypothesis excluded.
//   (b) What the number actually says. On DTLZ2 the front is ‖f‖ = 1 and
//       ‖f‖ = 1 + g, so the reported error IS g = Σ(x_i − 0.5)² over the ten
//       distance variables. A mean of 1.01 therefore means g ≈ 1, while
//       uniformly random x gives E[g] = 10/12 ≈ 0.83. The distance variables
//       are barely being optimised at all, even though the population is spread
//       and mutually non-dominated in the objective directions. Whatever is
//       wrong is in convergence pressure, not in diversity.
//   (c) Out-of-domain benchmark. §IV tests this algorithm on UF1-UF9 (CEC 2009)
//       and reports IGD 0.0081 on UF1 — the UF suite is built around
//       COMPLICATED PARETO SETS, which is the structure Fig.2's argument
//       ("individuals of the same sub-region are adjacent in decision space")
//       depends on. DTLZ2 appears nowhere in the paper. The smoke suite is
//       therefore judging this port on a problem its source never claimed.
//       That does not excuse the number, but it does mean a fair check requires
//       running UF1 and comparing against the paper's own table.
//
//   NOT ISOLATED (added 2026-08-04). moead_m2m.hpp shows the same signature at
//   pop=90: mean 1.022, best 0.010. Two controls from the same family rule out
//   "this decomposition is simply weak on DTLZ2" — at identical settings
//   sms_m2m reaches mean 0.0002 and moead_am2m 0.013.
//
//   What the two affected files have in common, as candidate leads (none
//   verified): both build sub-regional populations with per-region slot quotas,
//   and both were switched from SBX/PM to the Liu-Li annealing operator during
//   the 2026-07 fix waves. Note that moead_am2m was switched to Liu-Li too and
//   is fine, so the operator alone does not explain it.
//
//   Deliberately not "fixed" here: the correct resolution has to come from
//   §IV-A of the paper (liu2011, doi:10.1109/CEC.2011.5949848), not from
//   tuning until the number looks better. Scheduled for the primary-source
//   verification pass. Until then treat this algorithm's diversity behaviour
//   as unverified.
//
//   LG-8 (MINOR). set_eta_crossover/set_eta_mutation/set_pc are no-op shims
//     for API uniformity (operator [1] has no η/p_c; the crossover is
//     unconditional).
//
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP makes the NSGA-II selection of each internal set (Step 3.2)
//   and the R=1 non-dominance flag constrained, so a sub-region keeps feasible
//   solutions in preference to infeasible ones and only mates from R=1 members
//   chosen under the same rule. The external reservoir stays random, as in the
//   paper. The paper is unconstrained.
// EXTENSIONS BEYOND THE PAPER (disabled): binary genome.
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
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
// FIX 2026-07-07 (source-fidelity review): SBX/PM replaced with the
// Liu & Li 2009 [1] operator Eq.(5)-(6), as required by §II-C of the paper.
#include "../operators/liuli_crossover.hpp"

namespace mootation {

template <typename Ind_t>
class LiuGu2011Core {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int    T_ = 50;
    // FIX 2026-07-07: Max_gen for the annealing of operator [1] (liu2011 §IV-A:
    // ⌊(300000−5N)/N⌋); default 1000 is the library convention (moead_awa/adaw),
    // the caller must pass the real budget via set_t_max.
    int    t_max_ = 1000;
    double pm_ = -1.0;                 // P_m Eq.(6); <0 → auto 1/n ([1] §V)
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; std::vector<int> bvars; double cv=0.0; int rank=0; };

    std::vector<std::vector<double>> W_;          // [S] central vectors (unit)
    std::vector<std::vector<Sol>>    internal_;   // [S]
    std::vector<std::vector<Sol>>    external_;   // [S]
    std::vector<int>                 l_;          // [S] caps
    int S_=0, m_=0, N_=0, gen_=0;
    std::vector<double> z_;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }

    static double cosine(const std::vector<double>& a, const std::vector<double>& b){
        double dot=0,na=0,nb=0;
        for(std::size_t i=0;i<a.size();++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
        double d=std::sqrt(na)*std::sqrt(nb);
        if(d<1e-300) return 1.0;
        return std::clamp(dot/d,-1.0,1.0);
    }
    static std::vector<double> unit(const std::vector<double>& f){
        double n=0; for(double v:f) n+=v*v; n=std::sqrt(std::max(n,1e-300));
        std::vector<double> u(f.size()); for(std::size_t i=0;i<f.size();++i) u[i]=f[i]/n; return u;
    }
    static bool dominates(const std::vector<double>& a, const std::vector<double>& b){
        bool ne=false;
        for(std::size_t k=0;k<a.size();++k){ if(a[k]>b[k]) return false; if(a[k]<b[k]) ne=true; }
        return ne;
    }
    // Constraint-aware form (Deb's constrained domination when the mode is on).
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }

    void upd_ideal_from(const std::vector<Sol>& P){
        z_.assign(m_, std::numeric_limits<double>::max());
        for(const auto& s:P) for(int k=0;k<m_;++k) z_[k]=std::min(z_[k],s.objs[k]);
        if(P.empty()) z_.assign(m_,0.0);
    }
    int assoc(const std::vector<double>& f) const {
        std::vector<double> sh(m_); for(int k=0;k<m_;++k) sh[k]=f[k]-z_[k];
        int best=0; double bc=-2.0;
        for(int i=0;i<S_;++i){ double c=cosine(sh,W_[i]); if(c>bc){bc=c;best=i;} }
        return best;
    }

    // NSGA-II ordering (best→worst) by NDS + crowding distance.
    std::vector<int> nsga2_order(const std::vector<Sol>& P) const {
        int c=(int)P.size(); if(c==0) return {};
        std::vector<std::vector<int>> doms(c); std::vector<int> dc(c,0);
        std::vector<std::vector<int>> fronts; std::vector<int> f0;
        for(int p=0;p<c;++p){
            for(int q=0;q<c;++q){ if(p==q) continue;
                if(dominates(P[p],P[q])) doms[p].push_back(q);
                else if(dominates(P[q],P[p])) ++dc[p]; }
            if(dc[p]==0) f0.push_back(p);
        }
        fronts.push_back(f0);
        while(!fronts.back().empty()){
            std::vector<int> nx;
            for(int p:fronts.back()) for(int q:doms[p]) if(--dc[q]==0) nx.push_back(q);
            if(nx.empty()) break;
            fronts.push_back(std::move(nx));
        }
        std::vector<int> order;
        for(auto& F:fronts){
            int fn=(int)F.size();
            std::vector<double> cd(fn,0.0);
            for(int k=0;k<m_;++k){
                std::vector<int> o(fn); std::iota(o.begin(),o.end(),0);
                std::sort(o.begin(),o.end(),[&](int a,int b){return P[F[a]].objs[k]<P[F[b]].objs[k];});
                cd[o.front()]=cd[o.back()]=std::numeric_limits<double>::infinity();
                double rng=P[F[o.back()]].objs[k]-P[F[o.front()]].objs[k];
                if(rng<1e-300) continue;
                for(int t=1;t<fn-1;++t) cd[o[t]]+=(P[F[o[t+1]]].objs[k]-P[F[o[t-1]]].objs[k])/rng;
            }
            std::vector<int> o(fn); std::iota(o.begin(),o.end(),0);
            std::sort(o.begin(),o.end(),[&](int a,int b){return cd[a]>cd[b];});
            for(int t:o) order.push_back(F[t]);
        }
        return order;
    }

    // mark R=1 (rank=0) for the non-dominated ones within the internal set of each subregion
    void recompute_R(){
        for(int k=0;k<S_;++k){
            auto& I=internal_[k]; int c=(int)I.size();
            for(int p=0;p<c;++p){
                bool nd=true;
                for(int q=0;q<c;++q){ if(p!=q && dominates(I[q],I[p])){nd=false;break;} }
                I[p].rank = nd?0:1;
            }
        }
    }

    // FIX 2026-07-07 (source-fidelity review): operator
    // [1]=liu2009 Eq.(5)-(6) instead of SBX/PM. x — the internal parent (base),
    // y — the partner from the external set of the same subregion (§II-C);
    // one offspring.
    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch){
        const auto& b=vault.get_bounds(); int nv=vault.vars_n();
        std::vector<double> c1;
        ops::liuli_crossover(x.vars,y.vars,c1,b,gen_,t_max_,rng_);
        ops::liuli_mutation(c1,b,pm_eff(nv),gen_,t_max_,rng_);
        Sol z; z.vars=c1; z.rank=0;            // the offspring is marked R=1
        if(vault.bin_vars_n()>0){
            std::vector<int> bc1,bc2;
            ops::binary_crossover(x.bvars,y.bvars,bc1,bc2,rng_);
            ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
            z.bvars=bc1; vault.set_all_variables(scratch,c1,bc1);
        } else {
            vault.set_variables(scratch,c1);
        }
        vault.refresh_objectives(scratch);
        z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        return z;
    }

    void store_arch(DataVault<Ind_t>& vault){
        std::vector<Sol> P;
        for(int k=0;k<S_;++k) for(auto& s:internal_[k]) P.push_back(s);
        vault.reduce(0); vault.expand((int)P.size());
        for(int i=0;i<(int)P.size();++i)
            vault.seed_individual((std::size_t)i,P[i].vars,P[i].objs,P[i].bvars,{});
    }

public:
    LiuGu2011Core() = default;
    void set_T(int t){ if(t>0) T_=t; }
    // FIX 2026-07-07: Max_gen of the annealing of operator [1] (see header).
    void set_t_max(int t){ if(t>0) t_max_=t; }
    // LG-8: no-op shims (operator [1] has no η/p_c; the crossover is unconditional).
    void set_eta_crossover(double){}
    void set_eta_mutation(double){}
    void set_pc(double){}
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        S_=(int)std::ceil(std::sqrt((double)N_));
        auto Wr=das_dennis::generate_auto(m_,S_);
        W_.clear(); for(auto& w:Wr) W_.push_back(unit(w)); S_=(int)W_.size();
        l_.assign(S_,(int)std::ceil((double)N_/S_));

        int scratch=0;            // use active slot 0 as scratch
        const auto& b=vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0,1.0);
        std::uniform_int_distribution<int> dbn(0,1);
        std::vector<Sol> pool; pool.reserve(5*N_);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bv(vault.bin_vars_n());
        for(int i=0;i<5*N_;++i){
            for(int j=0;j<vault.vars_n();++j){double lo=b[j].first.value_or(0.0),hi=b[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);}
            for(int j=0;j<vault.bin_vars_n();++j) bv[j]=dbn(rng_);
            Sol s; s.vars=vars; s.bvars=bv;
            if(vault.bin_vars_n()>0) vault.set_all_variables(scratch,vars,bv); else vault.set_variables(scratch,vars);
            vault.refresh_objectives(scratch); s.objs=vault.objectives_of(scratch);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(scratch);
            pool.push_back(std::move(s));
        }
        upd_ideal_from(pool);
        std::vector<std::vector<int>> bk(S_);
        for(int i=0;i<(int)pool.size();++i) bk[assoc(pool[i].objs)].push_back(i);
        internal_.assign(S_,{}); external_.assign(S_,{});
        std::uniform_int_distribution<int> dp(0,(int)pool.size()-1);
        for(int k=0;k<S_;++k){
            std::vector<Sol> reg; for(int i:bk[k]) reg.push_back(pool[i]);
            if((int)reg.size()<=l_[k]){ internal_[k]=reg; }
            else { auto ord=nsga2_order(reg); for(int t=0;t<l_[k];++t) internal_[k].push_back(reg[ord[t]]); }
            int es=5*l_[k];
            for(int t=0;t<es;++t) external_[k].push_back(pool[dp(rng_)]);
        }
        recompute_R();
        store_arch(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        S_=(int)std::ceil(std::sqrt((double)N_));
        auto Wr=das_dennis::generate_auto(m_,S_);
        W_.clear(); for(auto& w:Wr) W_.push_back(unit(w)); S_=(int)W_.size();
        l_.assign(S_,(int)std::ceil((double)N_/S_));
        std::vector<Sol> pool;
        for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pool.push_back(s); }
        upd_ideal_from(pool);
        std::vector<std::vector<int>> bk(S_);
        for(int i=0;i<(int)pool.size();++i) bk[assoc(pool[i].objs)].push_back(i);
        internal_.assign(S_,{}); external_.assign(S_,{});
        std::uniform_int_distribution<int> dp(0,std::max(0,(int)pool.size()-1));
        for(int k=0;k<S_;++k){
            std::vector<Sol> reg; for(int i:bk[k]) reg.push_back(pool[i]);
            if((int)reg.size()<=l_[k]) internal_[k]=reg;
            else { auto ord=nsga2_order(reg); for(int t=0;t<l_[k];++t) internal_[k].push_back(reg[ord[t]]); }
            int es=5*l_[k];
            for(int t=0;t<es && !pool.empty();++t) external_[k].push_back(pool[dp(rng_)]);
            if(external_[k].empty() && !internal_[k].empty()) external_[k]=internal_[k];
        }
        recompute_R();
        store_arch(vault);
    }

    void step(DataVault<Ind_t>& vault){
        ++gen_;
        int scratch=vault.expand(1);
        std::vector<std::pair<int,int>> R1;
        for(int k=0;k<S_;++k) for(int j=0;j<(int)internal_[k].size();++j)
            if(internal_[k][j].rank==0) R1.push_back({k,j});
        std::vector<std::pair<int,int>> all_int;
        for(int k=0;k<S_;++k) for(int j=0;j<(int)internal_[k].size();++j) all_int.push_back({k,j});
        std::vector<std::pair<int,int>> mating=R1;
        if((int)mating.size()<N_ && !all_int.empty()){
            std::uniform_int_distribution<int> da(0,(int)all_int.size()-1);
            while((int)mating.size()<N_) mating.push_back(all_int[da(rng_)]);
        } else if((int)mating.size()>N_){
            // FIX 2026-07-07 (LG-7): the paper does not specify the truncation
            // order — random N instead of "the first N in region-major order"
            // (the former deterministic truncation cut off the tail regions).
            std::shuffle(mating.begin(),mating.end(),rng_);
            mating.resize(N_);
        }
        std::vector<Sol> Q; Q.reserve(N_);
        for(int idx=0; idx<N_ && !mating.empty(); ++idx){
            auto pr = mating[idx % mating.size()];
            int t=pr.first, j=pr.second;
            const Sol& xi = internal_[t][j];
            const Sol& xj = external_[t].empty() ? xi
                            : external_[t][std::uniform_int_distribution<int>(0,(int)external_[t].size()-1)(rng_)];
            Q.push_back(breed(xi,xj,vault,scratch));
        }
        std::vector<Sol> allp;
        for(auto& kk:internal_) for(auto& s:kk) allp.push_back(s);
        for(auto& s:Q) allp.push_back(s);
        upd_ideal_from(allp);

        std::vector<std::vector<Sol>> reg(S_);
        for(int k=0;k<S_;++k) for(auto& s:internal_[k]) reg[assoc(s.objs)].push_back(s);
        for(auto& s:Q) reg[assoc(s.objs)].push_back(s);

        for(int k=0;k<S_;++k){
            auto& R=reg[k];
            if((int)R.size()<=l_[k]){
                internal_[k]=R;
            } else {
                auto ord=nsga2_order(R);
                std::vector<Sol> keep, rej;
                for(int t=0;t<(int)ord.size();++t) (t<l_[k]?keep:rej).push_back(R[ord[t]]);
                internal_[k]=keep;
                if(!external_[k].empty()){
                    std::uniform_int_distribution<int> de(0,(int)external_[k].size()-1);
                    for(auto& s:rej) external_[k][de(rng_)]=s;
                }
            }
        }
        if(gen_%T_==0) recompute_R();
        store_arch(vault);
    }
};

} // namespace mootation

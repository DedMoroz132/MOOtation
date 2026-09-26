#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Liu & Gu 2011 — An Improved NSGA-II Algorithm Based on Sub-regional Search.
// H.-L. Liu, F. Gu — IEEE Congress on Evolutionary Computation (CEC) 2011,
// pp. 1906-1911.
// doi:10.1109/CEC.2011.5949848          (source: nsga2-subregional_liu2011)
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
//   LG-4 (READING; FIXED 2026-09-06, full-paper checklist). The central
//     vectors W — «uniformly distributed unit vectors on the first quadrant of
//     the unit hyper-sphere», no construction given. When S = ⌈√N⌉ is a
//     Das–Dennis lattice size the lattice normalized to unit length is used
//     (m=3, N=91: S=10 = H=3); otherwise the deterministic arbitrary-K
//     generator detail::uniform_sphere_directions shared with moead_m2m /
//     sms_m2m. S therefore stays EXACTLY ⌈√N⌉, and l_i = ⌈N/S⌉ and the
//     external size 5·l_i follow from N alone. Previously generate_auto
//     silently substituted the nearest attainable lattice size from above
//     (m=5, N=100: S=10 became 15) and l_i / 5·l_i moved with it.
//   LG-5 (MINOR). Shift by the minimum of the internal population; association
//     by the acute angle (cos) of the shifted f.
//   LG-6 (FIXED 2026-09-06, full-paper checklist). §II-B: «Randomly select
//     5·l_i individuals from the initial population to constitute the external
//     set» — a random SUBSET of the 5N pool (sampling without replacement,
//     sample_idx). Previously the draws were i.i.d., so one individual could
//     occupy several slots of the same external set. One individual may still
//     belong to the external sets of several sub-regions: every set is drawn
//     from the whole pool, as the paper says. (Operator [1] produces exactly
//     one offspring — the old remark about SBX pairs is retired.)
//   LG-6b (FIXED 2026-09-06, full-paper checklist). Step 3.3: «the remained
//     individuals replace the same number of external individuals randomly»
//     — |rej| DISTINCT external slots are drawn (partial Fisher–Yates) and
//     overwritten, so exactly "the same number" of members is displaced.
//     Previously the slots were drawn i.i.d. (≈8% of writes collided at the
//     paper's settings). When |rej| exceeds |external_k| — no rule in the
//     paper — a random |external_k| of the rejected replace the whole set.
//   LG-9 (FIXED 2026-09-06, full-paper checklist). §II-C: «If M_gen < N, we
//     randomly select N − M_gen individuals from the internal set to
//     participate in mating» — distinct internal members that are NOT already
//     marked R=1 (sample_idx over the rest). Only when the whole internal
//     population is smaller than N (possible: caps without refill) are the
//     remaining places filled i.i.d. from all internal members. Previously the
//     top-up was i.i.d. over all internal members, R=1 ones included.
//   LG-7 (AMBIGUOUS→resolved). For |R1|>N (possible when Σl_i>N, LG-3) the
//     paper only says «all of these individuals take part in mating» and
//     «generate N new individuals», without specifying the order.
//     FIX 2026-07-07 (source-fidelity review): instead of the deterministic
//     truncation "the first N in region-major order" (it systematically cut
//     off the tail regions) — RANDOM truncation to N (shuffle).
//   ---------------------------------------------------------------------
//   RESOLVED 2026-09 (second primary-source pass) — was "OPEN — UNDER
//   INVESTIGATION" since 2026-08-04.
//   ---------------------------------------------------------------------
//   Symptom: on DTLZ2 (M=3, n=12, N=91) at 200 generations the population
//   mean distance to the front was 1.01 while the best was 0.003.
//
//   Finding: not a defect of this port. The port was re-verified line by line
//   against liu2011 §II-B..D and §IV-A, and the operator against liu2009
//   §III-A Eq.(5)-(6) including the sign of the exponent in rm (an A/B with
//   the +a reading made things slightly WORSE, so the paper's -a stands).
//   What the number measures is a budget mismatch: the paper stops after
//   300,000 evaluations, i.e. Max_gen = ⌈(300000 − 5N)/N⌉ ≈ 3300 generations,
//   and the annealing schedule of the operator is calibrated to that.
//
//   Measured at the same settings, varying only the budget (seed 20260804):
//       gens    mean    median   best     distance vars at a box bound
//        200    0.939   0.762    0.0031   28.5 %
//       1000    0.633   0.014    0.0006   16.9 %
//       3000    0.028   0.000    0.0000    1.1 %
//   The population is bimodal: most members converge early (the median is
//   0.014 by 1000 generations) while a shrinking minority sits at the box
//   bounds of the distance variables and dominates the MEAN. That minority
//   is produced by the extrapolating crossover of Eq.(5), x + rc·(x − y),
//   whose out-of-bound repair lands between the bound and the parent, and it
//   is displaced slowly because selection inside a sub-region (NDS + crowding
//   distance, as the paper prescribes) exerts little convergence pressure
//   among mutually non-dominated members. moead_am2m, which shares the
//   operator but selects by scalarization, shows none of it — consistent with
//   the mechanism, not with a bug in the operator.
//   Cone width is NOT the lever: for the sibling moead_m2m, narrower cones
//   (K=30, S=3) made it catastrophically worse, and the paper's own K=S=17
//   no better than K=10.
//
//   Consequence for the convergence suite: this algorithm is run at the
//   paper's budget (3000 generations) rather than the suite's default 200,
//   and passes the ordinary thresholds there without any relaxation. The
//   known_issue marker is gone. See tests/test_convergence.cpp.
//
//   Not established and not claimed: performance on the paper's own UF1-UF10
//   instances against Table I. DTLZ2 appears nowhere in liu2011.
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
#include "../detail/sphere_directions.hpp"
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
    // Liu-Li repair (bound_repair, 2026-09-23): the paper's own rule by
    // default; resample works in the mutation only (liuli_crossover.hpp).
    ops::BoundRepair repair_ = ops::BoundRepair::Native;
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

    // k distinct indices out of [0,n) — partial Fisher–Yates (sampling without
    // replacement; LG-6 / LG-6b / LG-9).
    std::vector<int> sample_idx(int n, int k){
        std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
        k=std::clamp(k,0,n);
        for(int t=0;t<k;++t){ int r=std::uniform_int_distribution<int>(t,n-1)(rng_); std::swap(idx[t],idx[r]); }
        idx.resize(k); return idx;
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
        ops::liuli_crossover(x.vars,y.vars,c1,b,gen_,t_max_,rng_,repair_);
        ops::liuli_mutation(c1,b,pm_eff(nv),gen_,t_max_,rng_,repair_);
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
    void set_bound_repair(ops::BoundRepair r) {
        ops::require_repair(r, "liu_gu2011", false, true); repair_ = r; }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        S_=(int)std::ceil(std::sqrt((double)N_));
        auto Wr=das_dennis::generate_auto(m_,S_);
        if((int)Wr.size()!=S_) Wr=detail::uniform_sphere_directions(m_,S_);   // LG-4: S = ⌈√N⌉ exactly
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
        for(int k=0;k<S_;++k){
            std::vector<Sol> reg; for(int i:bk[k]) reg.push_back(pool[i]);
            if((int)reg.size()<=l_[k]){ internal_[k]=reg; }
            else { auto ord=nsga2_order(reg); for(int t=0;t<l_[k];++t) internal_[k].push_back(reg[ord[t]]); }
            // §II-B: "Randomly select 5·l_i individuals from the initial population"
            // — a random subset of the 5N pool, distinct members (LG-6).
            int es=std::min(5*l_[k],(int)pool.size());
            for(int i:sample_idx((int)pool.size(),es)) external_[k].push_back(pool[i]);
        }
        recompute_R();
        store_arch(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        S_=(int)std::ceil(std::sqrt((double)N_));
        auto Wr=das_dennis::generate_auto(m_,S_);
        if((int)Wr.size()!=S_) Wr=detail::uniform_sphere_directions(m_,S_);   // LG-4: S = ⌈√N⌉ exactly
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
        for(int k=0;k<S_;++k){
            std::vector<Sol> reg; for(int i:bk[k]) reg.push_back(pool[i]);
            if((int)reg.size()<=l_[k]) internal_[k]=reg;
            else { auto ord=nsga2_order(reg); for(int t=0;t<l_[k];++t) internal_[k].push_back(reg[ord[t]]); }
            int es=std::min(5*l_[k],(int)pool.size());       // LG-6: distinct members
            for(int i:sample_idx((int)pool.size(),es)) external_[k].push_back(pool[i]);
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
            // §II-C: "randomly select N − M_gen individuals from the internal set"
            // — distinct internal members not already marked R=1 (LG-9); repeats
            // only when the whole internal population is smaller than N.
            std::vector<std::pair<int,int>> rest;
            for(auto& pr:all_int) if(internal_[pr.first][pr.second].rank!=0) rest.push_back(pr);
            int need=N_-(int)mating.size();
            for(int i:sample_idx((int)rest.size(),std::min(need,(int)rest.size()))) mating.push_back(rest[i]);
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
                    // Step 3.3: "the remained individuals replace the same number of
                    // external individuals randomly" — |rej| DISTINCT slots (LG-6b).
                    // |rej| > |external_k|: a random |external_k| of them replace the
                    // whole set (the paper has no rule for this case).
                    int ne=(int)external_[k].size(), nr=(int)rej.size();
                    if(nr>ne){ std::shuffle(rej.begin(),rej.end(),rng_); nr=ne; }
                    auto slots=sample_idx(ne,nr);
                    for(int t=0;t<nr;++t) external_[k][slots[t]]=rej[t];
                }
            }
        }
        if(gen_%T_==0) recompute_R();
        store_arch(vault);
    }
};

} // namespace mootation

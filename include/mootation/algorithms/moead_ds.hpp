#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/DDS — "Dynamical decomposition and selection based evolutionary
// algorithm for many-objective optimization".
// Q. Bao, M. Wang, G. Dai, X. Chen, Z. Song — Applied Soft Computing (2023),
// article 110295 (PII S1568494623003137).
// doi:10.1016/j.asoc.2023.110295
//
// IDEA (FPS pivots + scalar HPF). The reference points are the SOLUTIONS
// THEMSELVES, projected onto the unit hyperplane Σf''=1 (NBI). The DDS
// environmental selection: dynamic decomposition — iteratively find pivot p =
// the solution farthest (by d2 between reference points) from the already
// selected set Q (FPS); among the candidates S_A "close to the pivot", the
// non-dominated solution minimizing HPF (the scalar d1+θ·d2) is taken. This is
// a relative of the FPS-based family of selection strategies, but with a
// global NBI scalarization rather than a local one.
//
// NORMALIZATION (Alg.2): z_min/z_nad (intercepts via the ASF extreme points
//   Eq.9; ASF weights — NSGA-III convention: 1 on the own axis, 1e-6 on the
//   others, see arbitration DDS-6); f''_i=(f_i−z*_i)/(z_nad_i−z*_i) (Eq.10);
//   reference point R(x)=f''/Σf'' on the hyperplane Σ=1 (Eq.11).
// d1(x)=(Σf''(x)−1)/√m (perpendicular to the hyperplane, signed, Eq.14).
// d2(x,y)=‖R(x)−R(y)‖ (Eq.3). HPF=d1+θ·d2, θ=G/Gmax (Eq.12,16).
//
// SCHEME (Algorithm 1):
//   init P (N); DDS(P,N).
//   while: O=∅; for i=1..N: P'=Mating(P_i) (prob. σ — K=√N nearest by d2,
//     otherwise 2 random); R=createOffspring(SBX+PM); O∪=R. P=P∪O; P=DDS(P,N).
// DDS(Alg.3): normalization; Q=extremes (rank 0); W=rest;
//   while |Q|<N: p=argmax_W distance(x,Q); S_A={x∈W: d2(x,p)≤distance(x,Q)};
//     s=argmin_{nondom(S_A)} HPF(x,p); Q∪=s; update distance.
// Mating(Alg.4): rand<σ → K=√N min-d2 neighbors, random mate; otherwise 2
//   random.
//
// DEFAULTS (§5): K=⌊√N⌋, σ=0.9; SBX η_c=20/p_c=1.0; PM η_m=20/p_m=1/d;
//   θ=G/Gmax. N=pop_size. Gmax — via set_t_max (generations).
//
// DECLARED DEVIATIONS:
//   DDS-1 (MINOR). createOffspring — SBX yields 2 children, the first is taken
//     (1 offspring per i → |O|=N → P∪O=2N → DDS selects N).
//   DDS-2 (MINOR). z_nad — hyperplane intercepts via the extreme points
//     (NSGA-III); on degeneration, fallback to the coordinate-wise max.
//   DDS-3 (MINOR). nondominated within S_A — front 0; if S_A/the front is
//     empty, s=pivot.
//   DDS-4 (MINOR). θ=G/Gmax = gen/t_max. The paper itself is of two minds
//     about what G counts: the prose under Eq.16 says "G is the current
//     evaluations and G_max is the maximum evaluations", while Alg.1 line 13
//     increments G once per outer iteration, i.e. per GENERATION. The two
//     readings coincide up to a constant here — this algorithm spends exactly N
//     evaluations per generation — so θ traces the same ramp either way and the
//     choice is free. (Generations instead of NFE; proportional.) The counter is advanced at the START of step(), so the
//     first DDS runs at θ = 1/Gmax, whereas Alg.1 increments G at line 13 —
//     after the line-12 DDS — and therefore runs its first in-loop DDS at
//     θ = 0. The whole schedule is shifted by one generation, i.e. by 1/Gmax.
//   DDS-5 (MINOR). Real-valued genome; binary is outside the coverage (NONE).
//   DDS-6 (ARBITRATION). FIX 2026-07-07 (source-fidelity review):
//     the literal Eq.9 inverts the ASF weights («when j=i, w=0» → 1e-6 on the
//     own axis, 1 on the others): for m≥3 the "extremes" degenerate into
//     near-ideal points (≈argmin f_i) → the hyperplane is built on the wrong
//     points, the intercepts get underestimated, or the DDS-2 fallback kicks
//     in. This contradicts the paper itself: «The calculation method can be
//     referred to NSGA-III [22]» and «the intercept … is now at f''_i=1 [22]».
//     The NSGA-III convention is adopted: w=1 on the own axis, 1e-6 on the
//     others (for m=2 the choice of points coincides).
//     Previously the code implemented the literal formula while the header
//     claimed "as in NSGA-III" — the code has been brought in line with the
//     declaration.
//   DDS-7 (MINOR). Mating d2 is recomputed from a FRESH normalization of the
//     N-solution parent population. Alg.1 line 8 instead consumes the d2
//     returned by the line-12 DDS run over the merged 2N pool. normalize()
//     derives its frame from z_min and from ASF-based intercepts, both
//     set-dependent, so the two frames differ and so do the resulting
//     distances. (Carrying d2 out of dds() would also remove the duplicate
//     work.)
//
// CONSTRAINTS (beyond the paper, off by default). The DDS pivot loop picks,
//   among the candidates near a pivot, the NON-DOMINATED one with the smallest
//   HPF. constraint_mode FEASIBILITY/CDP makes that non-dominance test a
//   constrained one (DDS-C), so a feasible candidate is preferred whenever one
//   exists in S_A and infeasible candidates are ordered by CV. The HPF scalar
//   and the normalization are untouched. The paper is unconstrained.
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
class MOEADDSCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double sigma_=0.9;
    double eta_c_=20.0, eta_m_=20.0, pc_=1.0, pm_=-1.0;
    int    t_max_=1000;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };
    std::vector<Sol> pop_;
    int N_=0, m_=0, gen_=0, K_=10;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }
    bool dom(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv); }

    // normalization of a set → f'' (scale on the hyperplane), R (ref on Σ=1), d1
    void normalize(const std::vector<Sol>& P, std::vector<std::vector<double>>& R,
                   std::vector<double>& d1) const {
        int n=(int)P.size();
        std::vector<double> zmin(m_,std::numeric_limits<double>::max());
        for(auto&s:P) for(int k=0;k<m_;++k) zmin[k]=std::min(zmin[k],s.objs[k]);
        // extreme points via ASF (Eq.9; weights — NSGA-III convention, DDS-6)
        // FIX 2026-07-07 (source-fidelity review): was
        // w=(k==i)?1e-6:1.0 (the literal Eq.9, an inversion against NSGA-III
        // [22], which the paper refers to) — now 1 on the own axis, 1e-6 on
        // the others.
        std::vector<int> ext(m_,0);
        for(int i=0;i<m_;++i){
            int best=0; double bg=std::numeric_limits<double>::max();
            for(int j=0;j<n;++j){
                double mx=-1e300;
                for(int k=0;k<m_;++k){ double w=(k==i)?1.0:1e-6; double v=(P[j].objs[k]-zmin[k])/w; if(v>mx) mx=v; }
                if(mx<bg){bg=mx;best=j;}
            }
            ext[i]=best;
        }
        // intercepts a_i: solve Z b = 1, a_i = 1/b_i ; Z[i][k]=f'(ext_i)[k]
        std::vector<double> a(m_, 0.0);
        std::vector<std::vector<double>> Z(m_,std::vector<double>(m_,0.0));
        for(int i=0;i<m_;++i) for(int k=0;k<m_;++k) Z[i][k]=P[ext[i]].objs[k]-zmin[k];
        std::vector<double> b(m_,1.0); bool ok=true;
        // Gaussian elimination
        std::vector<std::vector<double>> A=Z; std::vector<double> rhs(m_,1.0);
        for(int i=0;i<m_;++i){
            int piv=i; for(int r=i+1;r<m_;++r) if(std::abs(A[r][i])>std::abs(A[piv][i])) piv=r;
            if(std::abs(A[piv][i])<1e-12){ ok=false; break; }
            std::swap(A[i],A[piv]); std::swap(rhs[i],rhs[piv]);
            for(int r=0;r<m_;++r){ if(r==i) continue; double f=A[r][i]/A[i][i]; for(int c=i;c<m_;++c) A[r][c]-=f*A[i][c]; rhs[r]-=f*rhs[i]; }
        }
        if(ok){ for(int i=0;i<m_;++i){ b[i]=rhs[i]/A[i][i]; if(b[i]<=1e-12){ok=false;break;} a[i]=1.0/b[i]; } }
        if(!ok){ for(int k=0;k<m_;++k){ double mx=zmin[k]; for(auto&s:P) mx=std::max(mx,s.objs[k]); a[k]=std::max(mx-zmin[k],1e-12); } }
        R.assign(n,std::vector<double>(m_,0.0)); d1.assign(n,0.0);
        for(int j=0;j<n;++j){
            std::vector<double> fpp(m_); double sum=0;
            for(int k=0;k<m_;++k){ fpp[k]=(P[j].objs[k]-zmin[k])/a[k]; sum+=fpp[k]; }
            d1[j]=(sum-1.0)/std::sqrt((double)m_);
            double s2=(std::abs(sum)>1e-300)?sum:1.0;
            for(int k=0;k<m_;++k) R[j][k]=fpp[k]/s2;
        }
        // would the extreme indices be useful to the caller via an ext closure? — rebuilt there instead
        (void)b;
    }
    static double d2(const std::vector<double>& a, const std::vector<double>& b){
        double s=0; for(std::size_t k=0;k<a.size();++k){double d=a[k]-b[k];s+=d*d;} return std::sqrt(s);
    }
    std::vector<int> extreme_idx(const std::vector<Sol>& P) const {
        int n=(int)P.size();
        std::vector<double> zmin(m_,std::numeric_limits<double>::max());
        for(auto&s:P) for(int k=0;k<m_;++k) zmin[k]=std::min(zmin[k],s.objs[k]);
        std::vector<int> ext;
        // FIX 2026-07-07 (source-fidelity review): ASF weights
        // as in normalize() — NSGA-III convention (DDS-6), was the Eq.9 inversion.
        for(int i=0;i<m_;++i){ int best=0; double bg=std::numeric_limits<double>::max();
            for(int j=0;j<n;++j){ double mx=-1e300; for(int k=0;k<m_;++k){double w=(k==i)?1.0:1e-6; double v=(P[j].objs[k]-zmin[k])/w; if(v>mx)mx=v;} if(mx<bg){bg=mx;best=j;} }
            if(std::find(ext.begin(),ext.end(),best)==ext.end()) ext.push_back(best); }
        return ext;
    }

    // DDS: select N out of pool
    std::vector<Sol> dds(const std::vector<Sol>& pool){
        int n=(int)pool.size();
        if(n<=N_) return pool;
        std::vector<std::vector<double>> R; std::vector<double> d1; normalize(pool,R,d1);
        double theta=std::min(1.0,(double)gen_/std::max(1,t_max_));
        std::vector<char> inQ(n,0);
        std::vector<int> ext=extreme_idx(pool);
        std::vector<int> Q;
        for(int e:ext){ if(!inQ[e]){ inQ[e]=1; Q.push_back(e);} }
        std::vector<int> Wv; for(int i=0;i<n;++i) if(!inQ[i]) Wv.push_back(i);
        std::vector<double> distQ(n,std::numeric_limits<double>::max());
        for(int x:Wv){ double mn=1e300; for(int q:Q) mn=std::min(mn,d2(R[x],R[q])); distQ[x]=mn; }
        while((int)Q.size()<N_ && !Wv.empty()){
            // pivot = argmax distQ
            int p=-1; double bd=-1; for(int x:Wv){ if(distQ[x]>bd){bd=distQ[x];p=x;} }
            // S_A
            std::vector<int> SA; for(int x:Wv){ if(d2(R[x],R[p])<=distQ[x]) SA.push_back(x); }
            if(SA.empty()) SA.push_back(p);
            // nondominated in SA
            std::vector<int> nd;
            for(int x:SA){ bool ok=true; for(int y:SA){ if(x!=y && dom(pool[y],pool[x])){ok=false;break;} } if(ok) nd.push_back(x); }
            if(nd.empty()) nd=SA;
            // argmin HPF
            int s=nd[0]; double bg=1e300;
            for(int x:nd){ double hpf=d1[x]+theta*d2(R[x],R[p]); if(hpf<bg){bg=hpf;s=x;} }
            inQ[s]=1; Q.push_back(s);
            Wv.erase(std::remove(Wv.begin(),Wv.end(),s),Wv.end());
            for(int x:Wv) distQ[x]=std::min(distQ[x],d2(R[x],R[s]));
        }
        // if there were >N extremes (rare) — truncate
        std::vector<Sol> out; for(int i=0;i<(int)Q.size() && (int)out.size()<N_;++i) out.push_back(pool[Q[i]]);
        while((int)out.size()<N_) out.push_back(pool[std::uniform_int_distribution<int>(0,n-1)(rng_)]);
        return out;
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
    void store_arch(DataVault<Ind_t>& vault){ vault.reduce(0); vault.expand((int)pop_.size());
        for(int i=0;i<(int)pop_.size();++i) vault.seed_individual((std::size_t)i,pop_[i].vars,pop_[i].objs,{},{}); }

public:
    MOEADDSCore() = default;
    void set_sigma(double s){ sigma_=s; }
    void set_t_max(int t){ if(t>0) t_max_=t; }
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double p){ pc_=p; }
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0; K_=std::max(2,(int)std::floor(std::sqrt((double)N_)));
        const auto& bd=vault.get_bounds(); std::uniform_real_distribution<double> d(0.0,1.0);
        std::vector<double> vars(vault.vars_n());
        for(int i=0;i<N_;++i){ for(int j=0;j<vault.vars_n();++j){double lo=bd[j].first.value_or(0.0),hi=bd[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);} vault.set_variables(i,vars);}
        vault.sync();
        pop_.clear(); for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s);}
    }
    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0; K_=std::max(2,(int)std::floor(std::sqrt((double)N_)));
        pop_.clear(); for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s);}
    }

    void step(DataVault<Ind_t>& vault){
        ++gen_;
        int scratch=vault.expand(1);
        // normalization of the current P for d2-based mating
        std::vector<std::vector<double>> R; std::vector<double> d1; normalize(pop_,R,d1);
        std::uniform_real_distribution<double> uni(0,1);
        std::uniform_int_distribution<int> di(0,N_-1);
        std::vector<Sol> O; O.reserve(N_);
        for(int i=0;i<N_;++i){
            int p1=i, p2;
            if(uni(rng_)<sigma_){
                // K nearest to i by d2
                std::vector<std::pair<double,int>> dd;
                for(int j=0;j<N_;++j){ if(j==i) continue; dd.emplace_back(d2(R[i],R[j]),j); }
                int kk=std::min(K_,(int)dd.size()); std::partial_sort(dd.begin(),dd.begin()+kk,dd.end());
                p2=dd[std::uniform_int_distribution<int>(0,kk-1)(rng_)].second;
            } else { p1=di(rng_); p2=di(rng_); }
            O.push_back(breed(pop_[p1],pop_[p2],vault,scratch));
        }
        std::vector<Sol> merged=pop_; for(auto&s:O) merged.push_back(s);
        pop_=dds(merged);
        store_arch(vault);
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// APRD — A Multi-objective Evolutionary Algorithm with Adaptive Parallel Region
// Decomposition.
// Hongyan Chen, Hai-Lin Liu, Fangqing Gu, Lei Chen — Guangdong University of
//   Technology, Guangzhou, China. 13th International Conference on Advanced
//   Computational Intelligence (ICACI), 2021.
// doi:10.1109/ICACI52617.2021.9435909
//
// IDEA. Unlike M2M, which decomposes by different WEIGHTS, APRD decomposes by
// different IDEAL POINTS; subregion centres are taken from the individuals
// themselves (Min-Max), and slots are distributed by subregion size without
// refill.
// A hyperplane L approximates the PF trend by multiple linear regression over
// the archive and is shifted to the origin; the direction V=(−b2,...,−bn,1) is
// perpendicular to L. The population is projected onto L (clamped to the L
// segment, §II-C); K subregion centres are chosen Min-Max (Euclidean) among the
// projections; individuals are assigned by PARALLEL distance (Eq.2-3); the slot
// counts S_i follow the subregion sizes WITHOUT refill; the subregion ideal
// points are Min-Max; offspring selection is by PBI (Eq.7-8).
//
// SCHEME (Algorithm 1):
//   P is random; N_a=2N; Arc = the non-dominated members of P.
//   while not stop:
//     if |Arc| < N_a (early stage): NSGA-II (Q = crossover/mutation of P;
//        Arc = Update(Arc u Q); P = NDS+CD selection from P u Q).
//     else (late stage): mate within the subpopulations P_1..P_K, which are the
//        output of Algorithm 3 from the PREVIOUS generation (on first entry, the
//        initial partition of P per §II-A); for x in P_k take y from P_k with
//        probability p, otherwise from P; z = crossover/mutation; Q u= z;
//        Arc = Update(Arc u Q); fit L over Arc; rebuild P_1..P_K from
//        (union P_k) u Q via Algorithm 3.
//   Algorithm 2 Update(R,N_a): NDS; if |NDS| <= N_a then Arc = NDS, otherwise
//      Max-Min down to N_a, with the distance read as Euclidean — see APRD-9,
//      the paper is contradictory here.
//   Algorithm 3: K centres (Min-Max over the projections) -> assignment
//      (parallel distance) -> slots S_i (no refill) -> subregion ideal points Z
//      (Min-Max) -> PBI selection.
//
// PBI. Implemented as d1 = ((F−Z)·V̂), d2 = ‖F − (Z + d1·V̂)‖ with V̂ = V/‖V‖,
//   g = d1 + θ·d2 — i.e. d2 is the component of (F−Z) ORTHOGONAL to V, the
//   canonical Zhang & Li form. Note V here is the NORMAL of the fitted
//   hyperplane, not a per-subproblem weight: §II decomposes "by different ideal
//   points, but not by different weight vectors", so every subregion shares V
//   and differs only in Z.
//   THE PAPER'S PRINTED Eq.(7)-(8) DIFFER, and are not self-consistent:
//     Eq.7  d1 = ‖(F−Z)ᵀV‖ / ‖V‖        Eq.8  d2 = ‖F − (Z − d1·V)‖
//   (a) Eq.8 multiplies d1 by the UNNORMALIZED V, while Eq.7 already divided by
//       ‖V‖ — so d1·V has magnitude d1·‖V‖ and the two agree only if ‖V‖ = 1,
//       which V = (−b2,…,−bn,1) is not.
//   (b) Eq.8 SUBTRACTS the projection: F − Z + d1·V is (F−Z) plus its own
//       projection, not the orthogonal residual, so it does not measure a
//       distance to the line at all.
//   (c) Eq.7 wraps the dot product in bars, making d1 unsigned, whereas PBI's
//       d1 is signed (a solution beyond the reference point should score
//       better, not the same as one short of it).
//   All three are read as misprints and the canonical form is used. This is an
//   arbitration, not a transcription — see APRD-10.
//
// PAPER DEFAULTS (§III): K=10, p=0.7, N_a=2N, theta=5, N=pop_size; termination
//   by generation count.
// BEYOND THE PAPER — library defaults: SBX eta_c=20, p_c=1.0; PM eta_m=20,
//   p_m=1/n. §III lists only K, p, N_a, theta, N and the generation budget; the
//   paper says "crossover and mutation operators" and gives no numeric
//   parameters for either. The values above are the usual PlatEMO/NSGA-II
//   convention and are NOT attributable to §III.
//
// DECLARED DEVIATIONS:
//   APRD-1 (MINOR). The hyperplane is a least-squares regression of f_m on
//     f_1..f_{m-1}, shifted to the origin (the intercept is dropped); on a
//     degenerate system V = (1,...,1).
//   APRD-2 (IMPLEMENTED). L segment: point projections are clamped into
//     [Lmin,Lmax], the coordinate-wise bounds of the Arc projections onto L
//     (§II-C / Fig.2, "replace projections outside L with nearest endpoint").
//     See project_L / refresh_L.
//   APRD-3 (MINOR). PBI sign and projection are written in canonical form (d1
//     signed, proj = Z + d1 * V_hat); the paper writes "Z − d1 V", which is
//     dimensionally inconsistent, so the standard correct form is used.
//   APRD-4 (MINOR). Min-Max (Euclidean) for centres and ideal points: the first
//     element is random, as Min-Max in §II inherits a random start, then
//     farthest-to-set; slots S_i without refill (skipped when S_i = |P_i|),
//     §II-D2. The per-ideal-point argmin of Alg.3 Step 4 runs over the FULL
//     bucket with no exclusion of already-selected individuals, so one
//     individual may occupy several slots — as in the MOEA/D family. The paper
//     states no exclusion rule. With Min-Max-spread Z duplicates are rare but
//     possible.
//   APRD-5 (MINOR). Case I NSGA-II: (rank, CD) tournament plus SBX/PM, then
//     NDS+CD selection.
//   APRD-6 (MINOR). SBX yields the first child; y != x when mating inside a
//     subpopulation (Case II).
//   APRD-7 (resolved).
//     update_archive receives Arc u Q in BOTH cases (Alg.1: "Arc <-
//     UpdateArchive(Arc u Q, N_a)"), not pop u Q as it once did. The archive is
//     cumulative, the condition |Arc| >= N_a = 2N is reachable, and the switch
//     from Case I to Case II works.
//   APRD-8 (resolved).
//     the partition P_1..P_K is built by Algorithm 3 (update_subpops) and
//     REUSED by the next generation's mating (Alg.1: for k, foreach x in P_k),
//     rather than recomputed by a random-start FPS at every step.
//     On the first entry into Case II, when no stored partition exists, P is
//     partitioned per §II-A: projections onto L (from Arc), K Min-Max centres,
//     association by parallel distance (Eq.2-3). Case I invalidates the
//     partition. Individuals added by the Alg.3 fallback top-up are assigned to
//     the nearest centre (par_dist).
//   APRD-9 (READING — the paper contradicts itself). Algorithm 2 truncates the
//     archive by Euclidean max-min in objective space (maxmin_euclid). The
//     prose of §II-B specifies the maximum ANGLE, borrowing Max-Min from
//     M2M [2]; §II-D1 contrasts the two metrics explicitly, but does so only
//     for the subregion CENTRES, where it states plainly that the Euclidean
//     distance is used rather than the angle. Algorithm 2's own pseudocode
//     writes a generic d(x_i, x_j) with no metric named. This implementation
//     follows the pseudocode and reads d as Euclidean. An angular variant would
//     substitute an acute-angle measure (cf. ar_moea.hpp). The reading is not
//     free: on a strongly curved front the two metrics keep different survivors.
//
//   APRD-10 (ARBITRATION — the printed PBI equations are misprinted).
//     Eq.(7)-(8) as printed are dimensionally inconsistent (d1 already divided
//     by ‖V‖, then multiplied by the unnormalized V) and subtract the
//     projection where the orthogonal residual is meant, so they do not define
//     a distance to the reference line. The canonical Zhang & Li PBI is used
//     instead: d1 = (F−Z)·V̂, d2 = ‖(F−Z) − d1·V̂‖, V̂ = V/‖V‖. d1 is kept
//     SIGNED, against Eq.7's absolute-value bars. See the PBI block above for
//     the three separate discrepancies.
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP makes the archive's non-dominated filter (Alg.2) and the
//   environmental non-dominated sort constrained, so an infeasible solution
//   enters the archive only when nothing feasible competes for the slot. The
//   PBI selection inside a subregion stays on raw objectives. The paper is
//   unconstrained.
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
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class APRDCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int    K_ = 10;
    double p_ = 0.7;
    double theta_ = 5.0;
    double eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0, pm_ = -1.0;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; std::vector<int> bvars; double cv=0.0; };

    std::vector<Sol> pop_;        // population (N)
    std::vector<Sol> arc_;        // archive (<= N_a)
    // FIX 2026-07-07, APRD-8:
    // stored partition P_1..P_K (indices into pop_): the output of Algorithm 3,
    // reused by Case II mating; empty means no partition (Case I or start).
    std::vector<std::vector<int>> Pk_;
    std::vector<double> V_;       // hyperplane normal (direction)
    std::vector<double> Lmin_, Lmax_;   // L-segment bounds (Arc projections)
    int N_=0, m_=0, Na_=0;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }

    static bool dominates(const std::vector<double>& a, const std::vector<double>& b){
        bool ne=false;
        for(std::size_t k=0;k<a.size();++k){ if(a[k]>b[k]) return false; if(a[k]<b[k]) ne=true; }
        return ne;
    }
    // Constraint-aware form (Deb's constrained domination when the mode is on).
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }
    static double edist(const std::vector<double>& a, const std::vector<double>& b){
        double s=0; for(std::size_t k=0;k<a.size();++k){double d=a[k]-b[k];s+=d*d;} return std::sqrt(s);
    }

    // front 0 (the non-dominated members) of a set
    std::vector<int> nondominated(const std::vector<Sol>& R) const {
        std::vector<int> nd;
        int c=(int)R.size();
        for(int p=0;p<c;++p){ bool ok=true;
            for(int q=0;q<c;++q){ if(p!=q && dominates(R[q],R[p])){ok=false;break;} }
            if(ok) nd.push_back(p);
        }
        return nd;
    }

    // Min-Max (Euclidean) FPS: pick cnt indices out of pts
    std::vector<int> maxmin_euclid(const std::vector<std::vector<double>>& pts, int cnt){
        int n=(int)pts.size(); std::vector<int> sel;
        if(n==0||cnt<=0) return sel;
        std::vector<char> used(n,0);
        int r=std::uniform_int_distribution<int>(0,n-1)(rng_);
        sel.push_back(r); used[r]=1;
        while((int)sel.size()<cnt){
            int b=-1; double bd=-1.0;
            for(int i=0;i<n;++i){ if(used[i]) continue;
                double mind=std::numeric_limits<double>::max();
                for(int s:sel){ double d=edist(pts[i],pts[s]); if(d<mind) mind=d; }
                if(mind>bd){ bd=mind; b=i; } }
            if(b<0) break; sel.push_back(b); used[b]=1;
        }
        for(int i=0;(int)sel.size()<cnt;++i) sel.push_back(sel[i%std::max(1,(int)sel.size())]);
        return sel;
    }

    // Least squares: regress f[m-1] on f[0..m-2]; returns coef[0..m-1]
    // (intercept, k1, ...)
    std::vector<double> regress(const std::vector<Sol>& A) const {
        int p=m_;                       // coefficient count (intercept + m-1)
        std::vector<std::vector<double>> ATA(p,std::vector<double>(p,0.0));
        std::vector<double> ATy(p,0.0);
        for(const auto& s:A){
            std::vector<double> row(p); row[0]=1.0;
            for(int k=0;k<m_-1;++k) row[k+1]=s.objs[k];
            double y=s.objs[m_-1];
            for(int a=0;a<p;++a){ ATy[a]+=row[a]*y; for(int b=0;b<p;++b) ATA[a][b]+=row[a]*row[b]; }
        }
        // Gaussian elimination
        std::vector<double> coef(p,0.0);
        for(int i=0;i<p;++i){
            int piv=i; for(int r=i+1;r<p;++r) if(std::abs(ATA[r][i])>std::abs(ATA[piv][i])) piv=r;
            if(std::abs(ATA[piv][i])<1e-12) return {};   // degenerate
            std::swap(ATA[i],ATA[piv]); std::swap(ATy[i],ATy[piv]);
            for(int r=0;r<p;++r){ if(r==i) continue; double f=ATA[r][i]/ATA[i][i];
                for(int c=i;c<p;++c) ATA[r][c]-=f*ATA[i][c]; ATy[r]-=f*ATy[i]; }
        }
        for(int i=0;i<p;++i) coef[i]=ATy[i]/ATA[i][i];
        return coef;
    }

    void estimate_hyperplane(){
        auto coef=regress(arc_);
        V_.assign(m_,0.0);
        if((int)coef.size()==m_){
            for(int k=0;k<m_-1;++k) V_[k]=-coef[k+1];
            V_[m_-1]=1.0;
        } else {
            for(int k=0;k<m_;++k) V_[k]=1.0;   // fallback
        }
    }

    // projection of f onto the hyperplane {x·V = 0} through the origin
    std::vector<double> project(const std::vector<double>& f) const {
        double fv=0,vv=0; for(int k=0;k<m_;++k){fv+=f[k]*V_[k];vv+=V_[k]*V_[k];}
        std::vector<double> q(m_); double c=(vv>1e-300)?fv/vv:0.0;
        for(int k=0;k<m_;++k) q[k]=f[k]-c*V_[k]; return q;
    }
    // L segment: bounds of the Arc projections onto L (§II-C, Fig.2).
    void refresh_L(){
        Lmin_.assign(m_, std::numeric_limits<double>::max());
        Lmax_.assign(m_,-std::numeric_limits<double>::max());
        if(arc_.empty()){ Lmin_.clear(); Lmax_.clear(); return; }
        for(const auto& s:arc_){ auto q=project(s.objs); for(int k=0;k<m_;++k){ Lmin_[k]=std::min(Lmin_[k],q[k]); Lmax_[k]=std::max(Lmax_[k],q[k]); } }
    }
    std::vector<double> project_L(const std::vector<double>& f) const {
        auto q=project(f);
        if(!Lmin_.empty()) for(int k=0;k<m_;++k) q[k]=std::clamp(q[k],Lmin_[k],Lmax_[k]);
        return q;
    }
    // parallel (in-hyperplane) distance between F and a centre Y on L
    double par_dist(const std::vector<double>& F, const std::vector<double>& Y) const {
        std::vector<double> d(m_); for(int k=0;k<m_;++k) d[k]=F[k]-Y[k];
        double dv=0,vv=0; for(int k=0;k<m_;++k){dv+=d[k]*V_[k];vv+=V_[k]*V_[k];}
        double c=(vv>1e-300)?dv/vv:0.0, s2=0;
        for(int k=0;k<m_;++k){ double pr=d[k]-c*V_[k]; s2+=pr*pr; }
        return std::sqrt(std::max(s2,0.0));
    }
    double pbi(const std::vector<double>& F, const std::vector<double>& Z) const {
        double vv=0; for(int k=0;k<m_;++k) vv+=V_[k]*V_[k];
        double vn=std::sqrt(std::max(vv,1e-300));
        double d1=0; for(int k=0;k<m_;++k) d1+=(F[k]-Z[k])*V_[k]/vn;
        double s2=0; for(int k=0;k<m_;++k){ double pr=F[k]-(Z[k]+d1*V_[k]/vn); s2+=pr*pr; }
        return d1 + theta_*std::sqrt(std::max(s2,0.0));
    }

    // NSGA-II ordering (best to worst) by NDS+CD over a list of Sol
    std::vector<int> nsga2_order(const std::vector<Sol>& P) const {
        int c=(int)P.size(); if(c==0) return {};
        std::vector<std::vector<int>> doms(c); std::vector<int> dc(c,0);
        std::vector<std::vector<int>> fr; std::vector<int> f0;
        for(int p=0;p<c;++p){ for(int q=0;q<c;++q){ if(p==q) continue;
            if(dominates(P[p],P[q])) doms[p].push_back(q);
            else if(dominates(P[q],P[p])) ++dc[p]; }
            if(dc[p]==0) f0.push_back(p); }
        fr.push_back(f0);
        while(!fr.back().empty()){ std::vector<int> nx;
            for(int p:fr.back()) for(int q:doms[p]) if(--dc[q]==0) nx.push_back(q);
            if(nx.empty()) break; fr.push_back(std::move(nx)); }
        std::vector<int> order;
        for(auto& F:fr){ int fn=(int)F.size(); std::vector<double> cd(fn,0.0);
            for(int k=0;k<m_;++k){ std::vector<int> o(fn); std::iota(o.begin(),o.end(),0);
                std::sort(o.begin(),o.end(),[&](int a,int b){return P[F[a]].objs[k]<P[F[b]].objs[k];});
                cd[o.front()]=cd[o.back()]=std::numeric_limits<double>::infinity();
                double rg=P[F[o.back()]].objs[k]-P[F[o.front()]].objs[k]; if(rg<1e-300) continue;
                for(int t=1;t<fn-1;++t) cd[o[t]]+=(P[F[o[t+1]]].objs[k]-P[F[o[t-1]]].objs[k])/rg; }
            std::vector<int> o(fn); std::iota(o.begin(),o.end(),0);
            std::sort(o.begin(),o.end(),[&](int a,int b){return cd[a]>cd[b];});
            for(int t:o) order.push_back(F[t]); }
        return order;
    }

    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch){
        const auto& b=vault.get_bounds(); int nv=vault.vars_n();
        std::vector<double> c1,c2;
        ops::sbx(x.vars,y.vars,c1,c2,b,eta_c_,pc_,rng_);
        ops::polynomial_mutation(c1,b,eta_m_,pm_eff(nv),rng_);
        Sol z; z.vars=c1;
        if(vault.bin_vars_n()>0){
            std::vector<int> bc1,bc2;
            ops::binary_crossover(x.bvars,y.bvars,bc1,bc2,rng_);
            ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
            z.bvars=bc1; vault.set_all_variables(scratch,c1,bc1);
        } else { vault.set_variables(scratch,c1); }
        vault.refresh_objectives(scratch);
        z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        return z;
    }

    void update_archive(std::vector<Sol> R){
        auto nd=nondominated(R);
        std::vector<Sol> NDS; for(int i:nd) NDS.push_back(R[i]);
        if((int)NDS.size()<=Na_){ arc_=NDS; return; }
        std::vector<std::vector<double>> pts; for(auto& s:NDS) pts.push_back(s.objs);
        auto sel=maxmin_euclid(pts,Na_);
        arc_.clear(); for(int i:sel) arc_.push_back(NDS[i]);
    }

    void store_arch(DataVault<Ind_t>& vault){
        vault.reduce(0); vault.expand((int)pop_.size());
        for(int i=0;i<(int)pop_.size();++i)
            vault.seed_individual((std::size_t)i,pop_[i].vars,pop_[i].objs,pop_[i].bvars,{});
    }

    // Algorithm 3: build new subpopulations from the pool R (~2N) -> new pop_ (N)
    void update_subpops(const std::vector<Sol>& R){
        int n=(int)R.size();
        std::vector<std::vector<double>> proj(n);
        for(int i=0;i<n;++i) proj[i]=project_L(R[i].objs);
        auto ci=maxmin_euclid(proj,K_);
        int K=(int)ci.size();
        std::vector<std::vector<double>> centers; for(int i:ci) centers.push_back(proj[i]);
        std::vector<std::vector<int>> bucket(K);
        for(int i=0;i<n;++i){ int best=0; double bd=par_dist(R[i].objs,centers[0]);
            for(int k=1;k<K;++k){ double d=par_dist(R[i].objs,centers[k]); if(d<bd){bd=d;best=k;} }
            bucket[best].push_back(i); }
        std::vector<int> S(K,0), cnt(K);
        for(int k=0;k<K;++k){ cnt[k]=(int)bucket[k].size(); if(cnt[k]>0) S[k]=1; }
        int sum=0; for(int s:S) sum+=s;
        std::vector<int> ord(K); std::iota(ord.begin(),ord.end(),0);
        std::sort(ord.begin(),ord.end(),[&](int a,int b){return cnt[a]>cnt[b];});
        int guard=0;
        while(sum<N_ && guard<100*N_+100){
            bool any=false;
            for(int k:ord){ if(sum>=N_) break; if(cnt[k]>0 && S[k]<cnt[k]){ ++S[k]; ++sum; any=true; } }
            if(!any) break; ++guard;
        }
        // FIX 2026-07-07, APRD-8:
        // Algorithm 3 produces NEW subpopulations P_1..P_K; the partition is
        // stored (newk = the subpopulation index of each selected individual)
        // for the next step's mating.
        std::vector<Sol> newpop; std::vector<int> newk;
        for(int k=0;k<K;++k){
            if(S[k]<=0 || bucket[k].empty()) continue;
            std::vector<std::vector<double>> mp; for(int i:bucket[k]) mp.push_back(proj[i]);
            auto zi=maxmin_euclid(mp,S[k]);
            for(int j=0;j<S[k];++j){
                const std::vector<double>& Z = proj[bucket[k][zi[j]]];
                int best=-1; double bg=std::numeric_limits<double>::max();
                for(int i:bucket[k]){ double g=pbi(R[i].objs,Z); if(g<bg){bg=g;best=i;} }
                newpop.push_back(R[best]); newk.push_back(k);
            }
        }
        if((int)newpop.size()<N_){
            auto o=nsga2_order(R);
            for(int idx:o){ if((int)newpop.size()>=N_) break;
                // fallback beyond the paper: assign to the nearest centre
                // (par_dist, Eq.2-3)
                int bk=0; double bd=par_dist(R[idx].objs,centers[0]);
                for(int k=1;k<K;++k){ double d=par_dist(R[idx].objs,centers[k]); if(d<bd){bd=d;bk=k;} }
                newpop.push_back(R[idx]); newk.push_back(bk);
            }
        }
        if((int)newpop.size()>N_){ newpop.resize(N_); newk.resize(N_); }
        pop_=newpop;
        Pk_.assign(K,{});
        for(int i=0;i<(int)pop_.size();++i) Pk_[newk[i]].push_back(i);
    }

    // FIX 2026-07-07, APRD-8:
    // initial partition of P into K subpopulations on the first entry into
    // Case II (§II-A:
    // «the population P is divided into K subpopulation … by the proposed
    // adaptive parallel region decomposition strategy"): project pop_ onto L,
    // K Min-Max centres, association by parallel distance (Eq.2-3).
    void partition_pop(){
        int n=(int)pop_.size();
        std::vector<std::vector<double>> proj(n);
        for(int i=0;i<n;++i) proj[i]=project_L(pop_[i].objs);
        auto ci=maxmin_euclid(proj,K_); int K=(int)ci.size();
        if(K<=0){ Pk_.clear(); return; }
        std::vector<std::vector<double>> centers; for(int i:ci) centers.push_back(proj[i]);
        Pk_.assign(K,{});
        for(int i=0;i<n;++i){ int best=0; double bd=par_dist(pop_[i].objs,centers[0]);
            for(int k=1;k<K;++k){ double d=par_dist(pop_[i].objs,centers[k]); if(d<bd){bd=d;best=k;} }
            Pk_[best].push_back(i); }
    }

public:
    APRDCore() = default;
    void set_K(int k){ K_=k; }
    void set_n_clusters(int k){ K_=k; }
    void set_p(double p){ p_=p; }
    void set_theta(double t){ if(t>0) theta_=t; }
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double v){ pc_=v; }
    void set_pm(double v){ pm_=v; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); Na_=2*N_;
        const auto& b=vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0,1.0);
        std::uniform_int_distribution<int> dbn(0,1);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bv(vault.bin_vars_n());
        for(int i=0;i<N_;++i){
            for(int j=0;j<vault.vars_n();++j){double lo=b[j].first.value_or(0.0),hi=b[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);}
            for(int j=0;j<vault.bin_vars_n();++j) bv[j]=dbn(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bv); else vault.set_variables(i,vars);
        }
        vault.sync();
        pop_.clear(); for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s); }
        update_archive(pop_);
        V_.assign(m_,1.0);
        Pk_.clear();   // APRD-8: no partition yet
    }

    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); Na_=2*N_;
        pop_.clear(); for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s); }
        update_archive(pop_);
        V_.assign(m_,1.0);
        Pk_.clear();   // APRD-8: no partition yet
    }

    void step(DataVault<Ind_t>& vault){
        int scratch=vault.expand(1);
        std::uniform_real_distribution<double> uni(0.0,1.0);

        if((int)arc_.size()<Na_){
            // ── Case I: NSGA-II ──
            std::vector<Sol> Q; Q.reserve(N_);
            auto ord=nsga2_order(pop_);
            std::vector<int> rankpos((int)pop_.size());
            for(int r=0;r<(int)ord.size();++r) rankpos[ord[r]]=r;
            std::uniform_int_distribution<int> dp(0,(int)pop_.size()-1);
            auto tour=[&](){ int a=dp(rng_),b=dp(rng_); return rankpos[a]<rankpos[b]?a:b; };
            for(int i=0;i<N_;++i){ int x=tour(),y=tour(); Q.push_back(breed(pop_[x],pop_[y],vault,scratch)); }
            // FIX 2026-07-07, APRD-7:
            // Alg.1: Arc <- UpdateArchive(Arc u Q, N_a) — a cumulative archive.
            // With pop u Q the archive lost its memory and Case II never engaged.
            std::vector<Sol> AQ=arc_; for(auto& s:Q) AQ.push_back(s);
            update_archive(AQ);
            std::vector<Sol> R=pop_; for(auto& s:Q) R.push_back(s);
            auto o2=nsga2_order(R);
            std::vector<Sol> np; for(int t=0;t<N_ && t<(int)o2.size();++t) np.push_back(R[o2[t]]);
            pop_=np;
            Pk_.clear();   // APRD-8: NSGA-II selection invalidates the partition
        } else {
            // ── Case II: parallel region decomposition ──
            // FIX 2026-07-07, APRD-8:
            // mating runs over P_1..P_K, the output of Algorithm 3 from the
            // previous generation (it used to be recomputed by a random-start
            // FPS every step). On the first entry into Case II, P is
            // partitioned per §II-A.
            if(Pk_.empty()){
                estimate_hyperplane();
                refresh_L();
                partition_pop();
            }
            int n=(int)pop_.size();
            std::vector<Sol> Q; Q.reserve(N_);
            std::uniform_int_distribution<int> dall(0,n-1);
            for(int k=0;k<(int)Pk_.size();++k){
                int sz=(int)Pk_[k].size();
                for(int j=0;j<sz;++j){
                    int x=Pk_[k][j], y;
                    if(uni(rng_)<p_ && sz>1){ int yy=std::uniform_int_distribution<int>(0,sz-1)(rng_);
                        for(int a=0;a<5&&yy==j;++a) yy=std::uniform_int_distribution<int>(0,sz-1)(rng_);
                        y=Pk_[k][yy]; }
                    else y=dall(rng_);
                    Q.push_back(breed(pop_[x],pop_[y],vault,scratch));
                }
            }
            // FIX 2026-07-07, APRD-7:
            // Alg.1: Arc <- UpdateArchive(Arc u Q, N_a) — cumulative archive.
            std::vector<Sol> AQ=arc_; for(auto& s:Q) AQ.push_back(s);
            update_archive(AQ);
            // Alg.1: fit L over Arc, then (union P_k) u Q -> Algorithm 3
            // (new P_1..P_K).
            std::vector<Sol> R; for(auto& kk:Pk_) for(int i:kk) R.push_back(pop_[i]);
            for(auto& s:Q) R.push_back(s);
            estimate_hyperplane();
            refresh_L();
            update_subpops(R);
        }
        store_arch(vault);
    }
};

} // namespace mootation

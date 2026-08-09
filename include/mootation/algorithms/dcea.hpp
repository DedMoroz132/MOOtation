#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DCEA — A novel clustering-based evolutionary algorithm with objective space
//   decomposition for multi/many-objective optimization.
// W. Zheng, Y. Tan, Z. Yan, M. Yang — Information Sciences 677 (2024), 120940.
// doi:10.1016/j.ins.2024.120940
//
// IDEA (an M2M descendant: neighbour refill + NDS). The objective space is
// divided by W reference vectors V into subspaces Y^w (by the acute angle,
// Eq.2), S=N/W solutions per subspace. The mating pool of each individual is
// built by K-means clustering (CMEI, Alg.4). Offspring — by an adaptive
// sin-cos operator (SCAGO, Alg.5). Environment (ESM, Alg.6): merge P∪Q,
// decompose by V; on a shortfall in Y^w — a NEIGHBOUR refill (neighbours by
// the Λ weights), on an excess — NDS removal of the worst (as in Alg.1). The
// reference V are periodically updated from the cluster centres (Eq.8-9).
//
// Generational scheme (Algorithm 2):
//   Init (Alg.3): P is random; S=N/W; V (W), Λ (N) are uniform; decompose F_p
//     by V (Eq.2) and balance it with Algorithm 1 to exactly S per subspace →
//     Y^w; SP^w from Y^w (Alg.3 l.6-7).
//   while NFE≤NFEmax:
//     M ← CMEI(F_p,…)   (K-means, W clusters; the pool of an individual =
//                        the solutions of the clusters of its Y^w)
//     Q ← SCAGO(M,P)    (sin-cos Eq.4-6 + PM; MF>1→explore, MF=1→exploit)
//     P ← ESM(Q,P,V,Λ,b) (P∪Q; Y^w: <S neighbour-refill / >S NDS removal;
//                         SP^w ← Y^w, P=∪SP^w — SP is inherited by the next
//                         generation)
//     every 10000 NFE: update V from the cluster centres.
//
// CMEI (Alg.4): K-means(W) on F_p; cluster → the nearest Y^w (centre→nearest
//   V); the pool of an individual from SP^i = the union of the solutions of
//   the clusters in Y^i (CS), otherwise SP^i; SP is an input of Alg.4
//   (inherited from ESM/Init, not recomputed).
// SCAGO (Alg.5): μ=1−√(NFE/NFEmax); r1=2π·rnd, r2=2·rnd — ONE draw per
//   generation (l.4), the sin/cos choice — per offspring (l.11/19); NDS on
//   Y^i → MF; r0=NFE/NFEmax+1 (MF>1) or 1−NFE/NFEmax (MF=1);
//   y_t=x+r0·{sin|cos}(r1)·|r2·p−x|; y=y_t+μ(p−x); repair; PM (see DCEA-9 —
//   Alg.5 has the opposite order).
// ESM (Alg.6): R=P∪Q; f'=f−z_min (Eq.7); by V → Y^w; by Λ → τ;
//   |Y^w|<S: B^w=the b nearest Λ to v^w; TS=the solutions of those τ;
//     DC=TS∖Y^w; |DC|>need → random from DC, otherwise random from R;
//     |Y^w|>S: NDS removal of the worst;
//   SP^w ← Y^w (l.14/18/20); P = ∪SP^w (l.23).
//
// PAPER DEFAULTS (§4.1.1; §5.1.1 uses W=20, b=15 for all MaOPs):
//   W=10 (2-obj.) / 20 (3-obj.), b=10/15; N=pop_size; S=N/W.
//   NFEmax = pop_size·t_max (via set_t_max / the runner).
// ASSUMED — NOT GIVEN BY THE PAPER: PM η_m=20, p_m=1/n. The paper names
//   polynomial mutation only (§3.1.3, Alg.5 l.26) and states that W and b are
//   the only parameters that need setting; it gives no distribution index and
//   no mutation probability anywhere. The values above are the NSGA-II/PlatEMO
//   convention. SBX appears in neither the paper nor this implementation.
//
// DECLARED DEVIATIONS:
//   DCEA-1 (MINOR). Eq.8-9 is implemented with ρ=1: no history KC_NFE is
//     collected and no ρ-row average is taken — V is simply overwritten with
//     the current generation's K-means centres, unit-normalized. The paper
//     itself says ρ "is set simply, not in order to learn the shape of the PF",
//     so ρ=1 is an admissible degradation; the unit normalization is harmless
//     because the association is angular.
//     Second, the period is counted in GENERATIONS (upd_period_ = 100), which
//     equals the paper's 10000 NFE only when N = 100; at any other population
//     size the real period is 100·N evaluations.
//   DCEA-2 (MINOR). The Y^w/τ association — the acute angle to V/Λ on the
//     translated f−z_min (Eq.7). z_min here is a MONOTONE RUNNING ideal point:
//     initialized in setup() and only ever lowered as individuals are
//     evaluated, never recomputed. The gloss of Eq.7 specifies the minimum over
//     the PREVIOUS F_R, refreshed when Algorithm 6 ends, so once a
//     record-holding individual is discarded the running z_min sits strictly
//     below min(R). Transiently the code's z is therefore <= the paper's.
//   DCEA-3 (MINOR). K-means: Lloyd ≤ kmeans_it_ iterations (the paper: 100;
//     capped here for cost, convergence at small W is sufficient).
//   DCEA-4 (MINOR). NDS removal when |Y^w|>S: the tie-break inside the
//     boundary front is by crowding distance (the paper: "lowest ranks", the
//     tie is not specified).
//   DCEA-5 (MINOR). The genome is real-valued (the SCA operator lives in the
//     decision space); a binary genome is out of coverage (NONE).
//   DCEA-6. SP^w IS INHERITED from ESM (Alg.6 l.14/18/20 "Obtain SP^w based
//     on Y^w", l.23 "P = ∪SP^w") and from the initialization (Alg.3 l.6:
//     Eq.2 + Algorithm 1 — balancing to exactly S: a shortfall → a random
//     top-up from P, an excess → NDS removal of the worst), rather than
//     being recomputed by the Eq.2 association every generation. Notable
//     fix: it used to be recomputed that way, which was a DEVIATION — the
//     group sizes were then arbitrary, and MF and the SCAGO fallback pools
//     were computed over different groups. Extension beyond the paper: when
//     N is not a multiple of W, the ESM/Init slots missing up to N are
//     topped up with random individuals whose owner is assigned by Eq.2.
//   DCEA-7. r1, r2 are drawn ONCE per generation — Alg.5 l.4 sets them
//     before the loops over w and j. Notable fix: the earlier AMBIGUOUS
//     reading drew them per offspring. The sin/cos choice stays per
//     offspring (Alg.5 l.11/19: rand()<0.5 inside the loop).
//   DCEA-8 (MINOR). The number of subspaces W is rounded UP to the nearest
//     Das–Dennis lattice cardinality: setup() calls
//     das_dennis::generate_auto(m, W_req_) and then sets W_ = V_.size(). The
//     paper's own 3-objective default W=20 is not a lattice size for m=3
//     (H=4 gives 15, H=5 gives 21), so the shipped configuration runs at W=21.
//     Consequence worth stating plainly: S = N/W then floors, W·S < N, and the
//     "extension beyond the paper" top-up path documented in DCEA-6 fires every
//     generation rather than as an edge case. To reproduce the paper exactly,
//     choose N and W so that W is attainable and W divides N.
//   DCEA-9 (MINOR). Order of repair and mutation. Alg.5 lines 26-27 read
//     "y <- PM(y); repair y if necessary"; this code clamps into the bounds
//     first and mutates second. ops::polynomial_mutation clamps its own output,
//     so the population stays inside the box either way, but
//     PM(clamp(y)) != clamp(PM(y)) whenever the SCA step leaves the box.
//
//   DCEA-10 (AMBIGUOUS — the paper does not name the set). Alg.5 line 6 says
//     "perform the nondominated sorting method to obtain the maximum number of
//     front as MF" and names no population. MF is the switch between the two
//     regimes: MF > 1 gives r0 = NFE/NFEmax + 1 > 1 (exploration), MF = 1 gives
//     r0 = 1 − NFE/NFEmax < 1 (exploitation), so the reading changes the
//     operator's behaviour throughout the run. This port reads it PER SUBSPACE,
//     because line 6 sits inside the loop over w (line 5) — the textual
//     reading. The alternative, one sorting over the whole population, is
//     available via set_mf_scope_global(true).
//     Measured on DTLZ2 (M=3, n=12, pop 91, 200 generations, seed 20260804):
//       per-subspace  mean 0.1108  best 0.0101
//       global        mean 0.0863  best 0.0065
//     The global reading scores better here, but not by enough to overturn the
//     textual one on the strength of a single problem and seed, so the default
//     follows the pseudocode's nesting. Worth knowing when comparing this port
//     against the paper's tables: DCEA is one of the weaker performers in this
//     library's smoke suite under either reading.
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP makes every non-dominated sort a constrained one — the
//   ESM/ESD front accumulation and the mating rank both read it — so an
//   infeasible solution can only survive when nothing feasible competes for
//   its slot. The clustering and the SCAGO operator are untouched. The paper
//   is unconstrained.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../detail/math_compat.hpp"
#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"

namespace mootation {

template <typename Ind_t>
class DCEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int    W_req_ = 10, b_ = 10;
    // DCEA-3 / DCEA-1: both depart from the paper (100 Lloyd iterations; the V
    // update every 10000 NFE) and neither is reachable from the public API.
    int    kmeans_it_ = 30, upd_period_ = 100;
    double eta_m_ = 20.0, pm_ = -1.0;
    int    t_max_ = 1000;
    bool   mf_global_ = false;   // DCEA-10: scope of Alg.5 line 6
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };

    std::vector<std::vector<double>> V_, Lambda_;   // [W],[N] unit
    std::vector<Sol> pop_;
    // DCEA-6: the subpopulations SP^w (indices into pop_) — inherited from
    // ESM (Alg.6 l.14/18/20) / the initialization (Alg.3 l.6-7); they are the
    // input of CMEI and SCAGO.
    std::vector<std::vector<int>> SP_;
    std::vector<double> z_;
    int W_=0, m_=0, N_=0, S_=0, gen_=0;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }
    static double cosang(const std::vector<double>& a, const std::vector<double>& b){
        double d=0,na=0,nb=0; for(std::size_t i=0;i<a.size();++i){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
        double q=std::sqrt(na)*std::sqrt(nb); if(q<1e-300) return 1.0; return std::clamp(d/q,-1.0,1.0);
    }
    static std::vector<double> unit(std::vector<double> f){
        double n=0; for(double v:f) n+=v*v; n=std::sqrt(std::max(n,1e-300)); for(double&v:f) v/=n; return f;
    }
    bool dom(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }
    std::vector<double> shifted(const std::vector<double>& f) const {
        std::vector<double> s(m_); for(int k=0;k<m_;++k) s[k]=f[k]-z_[k]; return s;
    }
    int assoc(const std::vector<double>& f, const std::vector<std::vector<double>>& set) const {
        auto s=shifted(f); int best=0; double bc=cosang(s,set[0]);
        for(int i=1;i<(int)set.size();++i){ double c=cosang(s,set[i]); if(c>bc){bc=c;best=i;} }
        return best;
    }
    void upd_ideal(const std::vector<double>& f){ for(int k=0;k<m_;++k) z_[k]=std::min(z_[k],f[k]); }

    // NDS: the number of fronts (MF) for a set of indices into pool
    int max_front(const std::vector<int>& idx, const std::vector<Sol>& pool) const {
        int c=(int)idx.size(); if(c==0) return 0;
        std::vector<int> dc(c,0); std::vector<std::vector<int>> dl(c);
        std::vector<int> front;
        for(int p=0;p<c;++p){ for(int q=0;q<c;++q){ if(p==q) continue;
            if(dom(pool[idx[p]],pool[idx[q]])) dl[p].push_back(q);
            else if(dom(pool[idx[q]],pool[idx[p]])) ++dc[p]; }
            if(dc[p]==0) front.push_back(p); }
        int mf=0;
        while(!front.empty()){ ++mf; std::vector<int> nx;
            for(int p:front) for(int q:dl[p]) if(--dc[q]==0) nx.push_back(q);
            front=std::move(nx); }
        return mf;
    }
    // NDS+CD order (best→worst) for a set of indices into pool
    std::vector<int> nds_order(const std::vector<int>& idx, const std::vector<Sol>& pool) const {
        int c=(int)idx.size(); std::vector<int> dc(c,0); std::vector<std::vector<int>> dl(c);
        std::vector<std::vector<int>> fr; std::vector<int> f0;
        for(int p=0;p<c;++p){ for(int q=0;q<c;++q){ if(p==q) continue;
            if(dom(pool[idx[p]],pool[idx[q]])) dl[p].push_back(q);
            else if(dom(pool[idx[q]],pool[idx[p]])) ++dc[p]; }
            if(dc[p]==0) f0.push_back(p); }
        fr.push_back(f0);
        while(!fr.back().empty()){ std::vector<int> nx; for(int p:fr.back()) for(int q:dl[p]) if(--dc[q]==0) nx.push_back(q);
            if(nx.empty()) break;
            fr.push_back(std::move(nx)); }
        std::vector<int> order;
        for(auto& F:fr){ int fn=(int)F.size(); std::vector<double> cd(fn,0.0);
            for(int k=0;k<m_;++k){ std::vector<int> o(fn); std::iota(o.begin(),o.end(),0);
                std::sort(o.begin(),o.end(),[&](int a,int b){return pool[idx[F[a]]].objs[k]<pool[idx[F[b]]].objs[k];});
                cd[o.front()]=cd[o.back()]=std::numeric_limits<double>::infinity();
                double rg=pool[idx[F[o.back()]]].objs[k]-pool[idx[F[o.front()]]].objs[k]; if(rg<1e-300) continue;
                for(int t=1;t<fn-1;++t) cd[o[t]]+=(pool[idx[F[o[t+1]]]].objs[k]-pool[idx[F[o[t-1]]]].objs[k])/rg; }
            std::vector<int> o(fn); std::iota(o.begin(),o.end(),0);
            std::sort(o.begin(),o.end(),[&](int a,int b){return cd[a]>cd[b];});
            for(int t:o) order.push_back(idx[F[t]]); }
        return order;
    }

    // K-means (Lloyd) on the shifted objectives of pop_; returns assignment +
    // centers
    void kmeans(std::vector<int>& assign, std::vector<std::vector<double>>& centers){
        int n=(int)pop_.size();
        std::vector<std::vector<double>> X(n);
        for(int i=0;i<n;++i) X[i]=shifted(pop_[i].objs);
        // init: the first W_ distinct ones (shuffle)
        std::vector<int> perm(n); std::iota(perm.begin(),perm.end(),0);
        std::shuffle(perm.begin(),perm.end(),rng_);
        centers.assign(W_,std::vector<double>(m_,0.0));
        for(int k=0;k<W_;++k) centers[k]=X[perm[k]];
        assign.assign(n,0);
        for(int it=0;it<kmeans_it_;++it){
            bool ch=false;
            for(int i=0;i<n;++i){ int best=0; double bd=1e300;
                for(int k=0;k<W_;++k){ double d=0; for(int t=0;t<m_;++t){double df=X[i][t]-centers[k][t];d+=df*df;} if(d<bd){bd=d;best=k;} }
                if(assign[i]!=best){assign[i]=best;ch=true;} }
            std::vector<std::vector<double>> sum(W_,std::vector<double>(m_,0.0)); std::vector<int> cnt(W_,0);
            for(int i=0;i<n;++i){ ++cnt[assign[i]]; for(int t=0;t<m_;++t) sum[assign[i]][t]+=X[i][t]; }
            for(int k=0;k<W_;++k) if(cnt[k]>0) for(int t=0;t<m_;++t) centers[k][t]=sum[k][t]/cnt[k];
            if(!ch) break;
        }
    }

    void store_arch(DataVault<Ind_t>& vault){
        vault.reduce(0); vault.expand((int)pop_.size());
        for(int i=0;i<(int)pop_.size();++i) vault.seed_individual((std::size_t)i,pop_[i].vars,pop_[i].objs,{},{});
    }

    // DCEA-6: the initial decomposition (Alg.3 l.6-7): the Eq.2 association by
    // V → Y_t^w, then Algorithm 1 — balancing to exactly S per subspace (a
    // shortfall → a random top-up from P, l.3; an excess → NDS removal of the
    // worst, l.5-6); SP^w from Y^w.
    // Extension beyond the paper: when W·S < N (N is not a multiple of W) the
    // missing slots are topped up with random individuals whose owner is
    // assigned by Eq.2.
    void init_subpops(){
        SP_.assign(W_,{});
        int n=(int)pop_.size();
        if(n==0) return;
        std::vector<std::vector<int>> Yt(W_);
        for(int i=0;i<n;++i) Yt[assoc(pop_[i].objs,V_)].push_back(i);
        std::uniform_int_distribution<int> dr(0,n-1);
        int total=0;
        for(int w=0;w<W_;++w){
            std::vector<int> Yw=Yt[w];
            if((int)Yw.size()>S_){                       // Alg.1 l.5-6: NDS, remove the worst
                auto ord=nds_order(Yw,pop_);
                ord.resize(S_);
                Yw=ord;
            } else {
                while((int)Yw.size()<S_) Yw.push_back(dr(rng_));   // Alg.1 l.3: random from P
            }
            SP_[w]=Yw; total+=(int)Yw.size();
        }
        while(total<n){                                  // extension: N%W≠0
            int i=dr(rng_);
            SP_[assoc(pop_[i].objs,V_)].push_back(i); ++total;
        }
    }

public:
    DCEACore() = default;
    void set_W(int w){ W_req_=w; }
    void set_n_clusters(int w){ W_req_=w; }
    void set_b(int b){ b_=b; }
    void set_mf_scope_global(bool g){ mf_global_=g; }   // DCEA-10
    void set_t_max(int t){ if(t>0) t_max_=t; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_eta_crossover(double){}
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        auto Vr=das_dennis::generate_auto(m_,W_req_); V_.clear(); for(auto&v:Vr) V_.push_back(unit(v)); W_=(int)V_.size();
        if(N_%W_!=0) S_=std::max(1,N_/W_); else S_=N_/W_;
        S_=N_/W_; if(S_<1) S_=1;
        auto Lr=das_dennis::generate_auto(m_,N_); Lambda_.clear(); for(auto&v:Lr) Lambda_.push_back(unit(v));
        const auto& bd=vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0,1.0);
        std::vector<double> vars(vault.vars_n());
        for(int i=0;i<N_;++i){ for(int j=0;j<vault.vars_n();++j){double lo=bd[j].first.value_or(0.0),hi=bd[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);} vault.set_variables(i,vars); }
        vault.sync();
        pop_.clear(); for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s);}
        z_.assign(m_,std::numeric_limits<double>::max()); for(auto&s:pop_) upd_ideal(s.objs);
        init_subpops();   // DCEA-6: Alg.3 l.6-7 (Eq.2 + Algorithm 1)
    }
    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        auto Vr=das_dennis::generate_auto(m_,W_req_); V_.clear(); for(auto&v:Vr) V_.push_back(unit(v)); W_=(int)V_.size();
        S_=N_/W_; if(S_<1) S_=1;
        auto Lr=das_dennis::generate_auto(m_,N_); Lambda_.clear(); for(auto&v:Lr) Lambda_.push_back(unit(v));
        pop_.clear(); for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s);}
        z_.assign(m_,std::numeric_limits<double>::max()); for(auto&s:pop_) upd_ideal(s.objs);
        init_subpops();   // DCEA-6: Alg.3 l.6-7 (Eq.2 + Algorithm 1)
    }

    void step(DataVault<Ind_t>& vault){
        ++gen_;
        int scratch=vault.expand(1);
        const auto& bd=vault.get_bounds();
        double frac = std::min(1.0, (double)gen_/std::max(1,t_max_));   // ≈ NFE/NFEmax

        // DCEA-6: SP^w is NOT recomputed by the Eq.2 association — SP_ is
        // used, inherited from the ESM of the previous generation / the
        // initialization (Alg.3 l.6-7; Alg.4 receives SP as an input,
        // Alg.6 l.14/18/20 "Obtain SP^w based on Y^w").

        // ── CMEI: K-means → pools ──
        std::vector<int> kasg; std::vector<std::vector<double>> kcen;
        kmeans(kasg,kcen);
        // cluster → subspace (centre→nearest V)
        std::vector<int> cluster_sub(W_);
        for(int c=0;c<W_;++c) cluster_sub[c]=assoc_center(kcen[c]);
        std::vector<std::vector<int>> CS(W_);     // the solutions of subspace w
        for(int i=0;i<(int)pop_.size();++i) CS[cluster_sub[kasg[i]]].push_back(i);

        // ── SCAGO: offspring ──
        double mu=1.0-std::sqrt(frac);
        std::uniform_real_distribution<double> uni(0.0,1.0);
        // DCEA-7: r1, r2 — ONE draw per generation (Alg.5 l.4: "Set r1=2π*rnd
        // and r2=2*rnd", before the loops over w and j); it used to be one draw
        // per offspring. The sin/cos choice stays per offspring
        // (Alg.5 l.11/19).
        double r1=2.0*M_PI*uni(rng_), r2=2.0*uni(rng_);
        // DCEA-10: Alg.5 line 6 says "perform the nondominated sorting method
        // to obtain the maximum number of front as MF" without naming the set.
        // Read per-subspace by default (line 6 sits inside the loop over w);
        // set_mf_scope_global(true) reads it over the whole population instead.
        int mf_all = 0;
        if (mf_global_) {
            std::vector<int> all((int)pop_.size());
            for (int i=0;i<(int)pop_.size();++i) all[i]=i;
            mf_all = max_front(all, pop_);
        }
        std::vector<Sol> Q; Q.reserve(N_);
        for(int w=0;w<W_;++w){
            if(SP_[w].empty()) continue;
            int mf = mf_global_ ? mf_all : max_front(SP_[w],pop_);
            for(int idx:SP_[w]){
                const std::vector<int>& pool = !CS[w].empty()? CS[w] : SP_[w];
                int p=pool[std::uniform_int_distribution<int>(0,(int)pool.size()-1)(rng_)];
                double r0 = (mf>1)? frac+1.0 : 1.0-frac;
                std::vector<double> y(vault.vars_n());
                bool sgn = uni(rng_)<0.5;
                for(int t=0;t<vault.vars_n();++t){
                    double xv=pop_[idx].vars[t], pv=pop_[p].vars[t];
                    double mv = (sgn? std::sin(r1):std::cos(r1)) * std::abs(r2*pv - xv);
                    double yt = xv + r0*mv;
                    y[t] = yt + mu*(pv - xv);
                    double lo=bd[t].first.value_or(0.0),hi=bd[t].second.value_or(1.0);
                    y[t]=std::clamp(y[t],lo,hi);
                }
                ops::polynomial_mutation(y,bd,eta_m_,pm_eff(vault.vars_n()),rng_);
                vault.set_variables(scratch,y); vault.refresh_objectives(scratch);
                Sol z; z.vars=y; z.objs=vault.objectives_of(scratch);
                if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
                upd_ideal(z.objs); Q.push_back(z);
            }
        }

        // ── ESM ──
        std::vector<Sol> R=pop_; for(auto&s:Q) R.push_back(s);
        // assoc by V and Λ
        std::vector<std::vector<int>> Y(W_), tau((int)Lambda_.size());
        for(int i=0;i<(int)R.size();++i){ Y[assoc(R[i].objs,V_)].push_back(i); tau[assoc(R[i].objs,Lambda_)].push_back(i); }
        // the neighbours B^w: the b nearest Λ to v^w (Euclidean)
        // DCEA-6: owner[t] = the subspace w of slot t of the new population —
        // "Obtain SP^w based on Y^w" (Alg.6 l.14/18/20); SP_ is rebuilt from
        // owner.
        std::vector<Sol> newpop; std::vector<int> owner;
        for(int w=0;w<W_;++w){
            std::vector<int> Yw=Y[w];
            if((int)Yw.size()==S_){ for(int i:Yw){ newpop.push_back(R[i]); owner.push_back(w); } }
            else if((int)Yw.size()>S_){
                auto ord=nds_order(Yw,R);
                for(int t=0;t<S_;++t){ newpop.push_back(R[ord[t]]); owner.push_back(w); }
            } else {
                std::vector<bool> inYw(R.size(),false); for(int i:Yw) inYw[i]=true;
                for(int i:Yw){ newpop.push_back(R[i]); owner.push_back(w); }
                int need=S_-(int)Yw.size();
                // B^w = the b nearest Λ to V_[w]
                std::vector<std::pair<double,int>> dd;
                for(int l=0;l<(int)Lambda_.size();++l){ double s=0; for(int k=0;k<m_;++k){double df=V_[w][k]-Lambda_[l][k];s+=df*df;} dd.emplace_back(std::sqrt(s),l); }
                int bb=std::min(b_,(int)dd.size()); std::partial_sort(dd.begin(),dd.begin()+bb,dd.end());
                std::vector<int> DC;
                for(int t=0;t<bb;++t) for(int i:tau[dd[t].second]) if(!inYw[i]) DC.push_back(i);
                std::uniform_int_distribution<int> dr(0,(int)R.size()-1);
                if((int)DC.size()>need){
                    std::shuffle(DC.begin(),DC.end(),rng_);
                    for(int t=0;t<need;++t){ newpop.push_back(R[DC[t]]); owner.push_back(w); }
                } else {
                    for(int t=0;t<need;++t){ newpop.push_back(R[dr(rng_)]); owner.push_back(w); }
                }
            }
        }
        if((int)newpop.size()>N_){ newpop.resize(N_); owner.resize(N_); }
        while((int)newpop.size()<N_){
            // extension beyond the paper (N%W≠0): top up with a random
            // individual, the owner assigned by Eq.2
            int ri=std::uniform_int_distribution<int>(0,(int)R.size()-1)(rng_);
            newpop.push_back(R[ri]); owner.push_back(assoc(R[ri].objs,V_));
        }
        pop_=newpop;
        SP_.assign(W_,{});
        for(int t=0;t<(int)pop_.size();++t) SP_[owner[t]].push_back(t);

        // ── periodic update of V from the cluster centres (DCEA-1) ──
        if(gen_%upd_period_==0){
            for(int c=0;c<W_;++c){ auto u=unit(kcen[c]); if(u.size()==(std::size_t)m_) V_[c]=u; }
        }
        store_arch(vault);
    }

private:
    int assoc_center(const std::vector<double>& c) const {
        int best=0; double bc=cosang(c,V_[0]);
        for(int i=1;i<W_;++i){ double cc=cosang(c,V_[i]); if(cc>bc){bc=cc;best=i;} }
        return best;
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// HLMEA — Two-stage hybrid learning-based multi-objective evolutionary
//         algorithm based on objective space decomposition.
// W. Zheng, J. Sun — Information Sciences 610 (2022) 1163-1186.
// doi:10.1016/j.ins.2022.08.030    (source: 1-s2.0-S0020025522009136-main)
// FIX 2026-08-09 (Crossref sweep): the DOI read 10.1016/j.ins.2022.08.077,
//   which resolves to "Interpretable fuzzy clustering using unsupervised fuzzy
//   decision trees" — an unrelated Information Sciences paper. Authors and
//   page range added; they were missing entirely.
//
// IDEA (an M2M descendant: k-means drives mating; HV is used ONLY as the
// operator switch). MOEA/D-M2M framework (W subregions, Eq.2; the allocation of
// [17] = NDS truncation plus random refill, Alg.1). Two stages:
//   Stage 1 (gen <= α·Gmax): within each subregion, operator Eq.4
//     (y = x + F0·(x−x_r), F0 adaptive) + PM; Q ∪ P -> allocation (Alg.1).
//   Stage 2 (gen > α·Gmax): within each subregion, K-means(K) in DECISION
//     space; the mating pool B is the individual's cluster C^i with probability
//     β, otherwise the subregion P_w; the operator is DE: Eq.5 (rand/1/bin)
//     when δ >= 0, otherwise Eq.6 (current/1), plus PM; the environment is
//     NSGA-II (NDS+CD) selecting N; allocation (Alg.1); every 2 generations δ
//     is refreshed from the HV of the cluster centres (Alg.4):
//     δ = HV(c_old)/HV(c_new) − 1.
//
// Eq.4: y=x+F0·(x−x_r), F0=(2r−1)·(1−r)^(−(1−gen/Gmax)^0.8).
// Eq.5: y_i = x_r1 + F1·(x_r2−x_r3) if rand<=CR1 else x_i (DE/rand/1/bin).
// Eq.6: y_i = x  + F2·(x_r1−x_r2)  if rand<=CR2 else x_i (DE/current/1).
//
// PAPER DEFAULTS (§4-§5): W is objective-count-dependent — §4.3 sets "W = 10
//   and W = 15 for MOPs with two and three objectives, respectively", §4.8 sets
//   W = 10 for the MaOP suite (m = 4, 5, 10). K=3 with <=100 K-means iterations
//   (§4.3); β=0.9 (§4.3/§5.3);
//   α=0.5 (MOP)/0.3 (MaOP); F1=0.5,CR1=1.0,F2=0.5,CR2=0.6;
//   PM η_m=20, p_m=1/n; HV-ref z*=(1,…,1). N=pop_size; S=N/W.
//
// DECLARED DEVIATIONS:
//   HLMEA-1 (RESOLVED). The header once claimed that "β and K are not stated
//     unambiguously in the tables". That was wrong: §4.3 gives both in plain
//     text — K=3 ("the number of K in K-means is set as 3"; §5.4 reports K=2
//     and K>=4 as worse) and β=0.9 (also §5.3). The defaults now match the
//     paper: K=3, β=0.9, both settable.
//   HLMEA-2 (MINOR). The allocation of [17]: NDS truncation plus random refill
//     (as in moead_m2m); the boundary-front tie is broken by crowding distance.
//   HLMEA-3 (MINOR). HV of the cluster centres via HSO slicing. Alg.4 gives
//     three lines — H_old, H_new, δ = H_old/H_new − 1 — and never names a
//     reference point, so this port fixes one: ref = 1.1·max over the UNION
//     c_old ∪ c_new, shared by both evaluations. A shared reference is not
//     optional. hv_of is positively homogeneous of degree m, so with a
//     per-set reference any uniform contraction of the centres toward the
//     origin — i.e. convergence by a factor λ<1 — gives H_new = λ^m·H_old and
//     δ = λ^{−m} − 1 > 0 unconditionally, while under any common reference the
//     same contraction gives δ < 0. Since δ ≥ 0 selects Eq.5 (exploration) and
//     δ < 0 selects Eq.6 (exploitation), a per-set reference would invert the
//     switch for the whole run. §3.1 calls line 3 "a promotion rate of two
//     point sets", which requires one reference by construction.
//   HLMEA-4 (MINOR). K-means: Lloyd in decision space with an early exit on
//     stabilization; the iteration limit is 100 per §4.3 (it had been 30 as an
//     "implementation choice", but the paper states 100 explicitly).
//   HLMEA-5 (MINOR). The Φ_w association uses the acute angle to V on f−z_min.
//   HLMEA-6 (MINOR). Real-valued genome; binary is out of scope.
//   HLMEA-9 (MINOR, consequence of the sizing formula). The Alg.1 allocation
//     fills every subregion to exactly S = floor(N/W), so the active
//     population after each step() has W*S members — SHORT of pop_size
//     whenever W does not divide N. At the paper's own m=3 default W=15, a
//     pop_size of 91 (the natural Das-Dennis lattice size for three
//     objectives, and what the NSGA-III family wants) yields 15*6 = 90.
//     The paper assumes N = W*S throughout and never says what to do with a
//     remainder, so the floor is the faithful reading; it is NOT silently
//     corrected, because rounding pop_size would change the search.
//     Unlike ISDE+RD, where K is the user's parameter, W here is chosen BY
//     THE LIBRARY from the objective count, so the shortfall would otherwise
//     be invisible: set_W pins an explicit divisor, and a warning fires when
//     the default does not divide pop_size.
//   HLMEA-8 (MINOR). Alg.3 says "randomly select" one (line 2), three (line 6)
//     or two (line 9) solutions from the mating pool and never requires them to
//     be distinct from each other or from x. This port samples with replacement,
//     literally. A collision is not a malfunction: r2=r3 in Eq.5 degenerates to
//     y=x_r1 (a copy of a pool member, then PM), r1=r2 in Eq.6 and x_r=x in
//     Eq.4 degenerate to y=x (then PM). Collisions are common here because the
//     pool is a K-means cluster of one subregion (|P_w|=N/W, split K=3 ways, so
//     typically 2-4 members), which is exactly the regime the paper designed;
//     forcing distinctness would silently change the operator the authors
//     measured. Note that Eq.5/Eq.6 already re-sample per VARIABLE via the CR
//     test, so a collision costs one component, not one offspring.
//   HLMEA-7 (MINOR). W follows the paper's per-m branch by default: 15 at m=3,
//     10 otherwise. set_W pins an explicit value and disables the branch. The
//     requested count then goes through das_dennis::generate_auto, so the
//     effective W is the nearest attainable lattice size >= the request (at
//     m=3 a request of 15 is exact; at m=2 any W is exact).
//
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP routes the NDS used by the Alg.1 allocation and by the
//   stage-2 NSGA-II environment through Deb's constrained domination. The
//   subregion association, the HV switch and the DE operators are untouched:
//   they are geometry, not preference. The paper is unconstrained.
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
#include "../warn.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"

namespace mootation {

template <typename Ind_t>
class HLMEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // FIX 2026-07-07: K_=5→3,
    // kmeans_it_ 30 -> 100 — §4.3 in plain text: "the number of K in K-means is
    // set as 3 and the maximum number of iterations in K-means is set as 100»;
    // §5.4: IGD is worse at K=2 and K>=4; the optimum is K=3.
    // W_req_ = 0 means "use the paper's objective-count-dependent default"
    // (§4.3: 10 for m=2, 15 for m=3; §4.8: 10 for m>=4). set_W pins it.
    int    W_req_=0, K_=3, kmeans_it_=100, t_max_=1000;
    double alpha_=0.5, beta_=0.9;   // §5.3: "minimum IGD ...
    // when β=0.9" (the paper's text; the image caption saying it grows with β
    // is an OCR error).
    double F1_=0.5, CR1_=1.0, F2_=0.5, CR2_=0.6;
    double eta_m_=20.0, pm_=-1.0;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };
    std::vector<std::vector<double>> V_;
    std::vector<std::vector<Sol>> subpop_;
    std::vector<double> z_;
    std::vector<std::vector<double>> c_old_;   // stage-2 centres of the previous round (objs)
    double delta_=0.0;
    int W_=0, m_=0, N_=0, S_=0, gen_=0;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }
    static double cosang(const std::vector<double>& a, const std::vector<double>& b){
        double d=0,na=0,nb=0; for(std::size_t i=0;i<a.size();++i){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
        double q=std::sqrt(na)*std::sqrt(nb); if(q<1e-300) return 1.0; return std::clamp(d/q,-1.0,1.0);
    }
    static std::vector<double> unit(std::vector<double> f){ double n=0; for(double v:f) n+=v*v; n=std::sqrt(std::max(n,1e-300)); for(double&v:f) v/=n; return f; }
    bool dom(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv); }
    void upd_ideal(const std::vector<double>& f){ for(int k=0;k<m_;++k) z_[k]=std::min(z_[k],f[k]); }
    int assoc(const std::vector<double>& f) const {
        std::vector<double> s(m_); for(int k=0;k<m_;++k) s[k]=f[k]-z_[k];
        int best=0; double bc=cosang(s,V_[0]); for(int i=1;i<W_;++i){double c=cosang(s,V_[i]); if(c>bc){bc=c;best=i;}} return best;
    }

    std::vector<int> nds_order(const std::vector<Sol>& P) const {
        int c=(int)P.size(); std::vector<int> dc(c,0); std::vector<std::vector<int>> dl(c);
        std::vector<std::vector<int>> fr; std::vector<int> f0;
        for(int p=0;p<c;++p){ for(int q=0;q<c;++q){ if(p==q) continue;
            if(dom(P[p],P[q])) dl[p].push_back(q); else if(dom(P[q],P[p])) ++dc[p]; }
            if(dc[p]==0) f0.push_back(p); }
        fr.push_back(f0);
        while(!fr.back().empty()){ std::vector<int> nx; for(int p:fr.back()) for(int q:dl[p]) if(--dc[q]==0) nx.push_back(q); if(nx.empty()) break; fr.push_back(std::move(nx)); }
        std::vector<int> order;
        for(auto&F:fr){ int fn=(int)F.size(); std::vector<double> cd(fn,0.0);
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

    // allocation of [17]: pool -> W subpopulations of size S (angle association;
    // NDS truncation when above S, random refill when below)
    void allocate(const std::vector<Sol>& pool){
        std::vector<std::vector<int>> bk(W_);
        for(int i=0;i<(int)pool.size();++i) bk[assoc(pool[i].objs)].push_back(i);
        std::uniform_int_distribution<int> dp(0,(int)pool.size()-1);
        subpop_.assign(W_,{});
        for(int w=0;w<W_;++w){
            if((int)bk[w].size()==S_){ for(int i:bk[w]) subpop_[w].push_back(pool[i]); }
            else if((int)bk[w].size()>S_){
                std::vector<Sol> sub; for(int i:bk[w]) sub.push_back(pool[i]);
                auto ord=nds_order(sub); for(int t=0;t<S_;++t) subpop_[w].push_back(sub[ord[t]]);
            } else {
                for(int i:bk[w]) subpop_[w].push_back(pool[i]);
                int need=S_-(int)bk[w].size(); for(int t=0;t<need;++t) subpop_[w].push_back(pool[dp(rng_)]);
            }
        }
    }

    void kmeans(const std::vector<Sol>& P, std::vector<int>& assign, int K){
        int n=(int)P.size(); int nv=(int)P[0].vars.size();
        K=std::min(K,n);
        std::vector<std::vector<double>> cen(K);
        std::vector<int> perm(n); std::iota(perm.begin(),perm.end(),0); std::shuffle(perm.begin(),perm.end(),rng_);
        for(int k=0;k<K;++k) cen[k]=P[perm[k]].vars;
        assign.assign(n,0);
        for(int it=0;it<kmeans_it_;++it){ bool ch=false;
            for(int i=0;i<n;++i){ int best=0; double bd=1e300;
                for(int k=0;k<K;++k){ double d=0; for(int t=0;t<nv;++t){double df=P[i].vars[t]-cen[k][t];d+=df*df;} if(d<bd){bd=d;best=k;} }
                if(assign[i]!=best){assign[i]=best;ch=true;} }
            std::vector<std::vector<double>> sm(K,std::vector<double>(nv,0.0)); std::vector<int> cnt(K,0);
            for(int i=0;i<n;++i){ ++cnt[assign[i]]; for(int t=0;t<nv;++t) sm[assign[i]][t]+=P[i].vars[t]; }
            for(int k=0;k<K;++k) if(cnt[k]>0) for(int t=0;t<nv;++t) cen[k][t]=sm[k][t]/cnt[k];
            if(!ch) break; }
    }

    // HV (HSO) of the union, for δ
    static double hv_union(std::vector<std::vector<double>> q){
        if(q.empty()) return 0.0; int m=(int)q[0].size();
        if(m==1){ double mx=0; for(auto&p:q) mx=std::max(mx,p[0]); return mx; }
        std::sort(q.begin(),q.end(),[m](const std::vector<double>&a,const std::vector<double>&b){return a[m-1]>b[m-1];});
        double vol=0; int n=(int)q.size();
        for(int i=0;i<n;++i){ double nx=(i+1<n)?q[i+1][m-1]:0.0; double th=q[i][m-1]-nx; if(th<=0) continue;
            std::vector<std::vector<double>> pr; for(int t=0;t<=i;t++) pr.emplace_back(q[t].begin(),q[t].begin()+(m-1));
            vol+=th*hv_union(pr); }
        return vol;
    }
    // Reference point for the Alg.4 promotion rate: max over the SUPPLIED pool,
    // inflated by 10%. Alg.4 compares H_old and H_new as "a promotion rate of
    // two point sets", which is only meaningful against ONE reference — hence
    // the pool is c_old ∪ c_new, never one set at a time. See HLMEA-3.
    std::vector<double> hv_ref(const std::vector<std::vector<double>>& C) const {
        std::vector<double> ref(m_,-1e300);
        for(auto&c:C) for(int k=0;k<m_;++k) ref[k]=std::max(ref[k],c[k]);
        for(int k=0;k<m_;++k) ref[k]=(ref[k]>-1e299)?((ref[k]>0)?ref[k]*1.1:ref[k]+1.0):1.0;
        return ref;
    }
    double hv_of(const std::vector<std::vector<double>>& C,
                 const std::vector<double>& ref) const {
        if(C.empty()) return 0.0;
        std::vector<std::vector<double>> q;
        for(auto&c:C){ std::vector<double> qi(m_); for(int k=0;k<m_;++k) qi[k]=std::max(0.0,ref[k]-c[k]); q.push_back(qi); }
        return hv_union(std::move(q));
    }

    std::vector<Sol> flat() const { std::vector<Sol> P; for(auto&s:subpop_) for(auto&x:s) P.push_back(x); return P; }
    void store_arch(DataVault<Ind_t>& vault){ auto P=flat(); vault.reduce(0); vault.expand((int)P.size());
        for(int i=0;i<(int)P.size();++i) vault.seed_individual((std::size_t)i,P[i].vars,P[i].objs,{},{}); }

    Sol eval(std::vector<double> y, const std::vector<std::pair<std::optional<double>,std::optional<double>>>& bd,
             DataVault<Ind_t>& vault, int scratch){
        for(int t=0;t<(int)y.size();++t){ double lo=bd[t].first.value_or(0.0),hi=bd[t].second.value_or(1.0); y[t]=std::clamp(y[t],lo,hi); }
        ops::polynomial_mutation(y,bd,eta_m_,pm_eff((int)y.size()),rng_);
        vault.set_variables(scratch,y); vault.refresh_objectives(scratch);
        Sol z; z.vars=y; z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        upd_ideal(z.objs); return z;
    }

public:
    HLMEACore() = default;
    void set_W(int w){ W_req_=w; }
    // Paper default for W, resolved once m is known (see W_req_).
    int W_default() const { return (m_==3) ? 15 : 10; }
    void set_n_clusters(int w){ W_req_=w; }
    void set_K(int k){ K_=k; }
    void set_alpha(double a){ alpha_=a; }
    void set_beta(double b){ beta_=b; }
    void set_t_max(int t){ if(t>0) t_max_=t; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_eta_crossover(double){}
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0; delta_=0.0;
        c_old_.clear();   // stage-2 HV history is per-run state, not per-object
        int w_req=(W_req_>0)?W_req_:W_default();
        auto Vr=das_dennis::generate_auto(m_,w_req); V_.clear(); for(auto&v:Vr) V_.push_back(unit(v)); W_=(int)V_.size();
        S_=N_/W_; if(S_<1) S_=1;
        if(W_*S_ != N_)
            warn_lazy([&]{ return "hlmea: W=" + std::to_string(W_) +
                           " does not divide pop_size=" + std::to_string(N_) +
                           "; the answer set will hold W*floor(N/W)=" +
                           std::to_string(W_*S_) + " individuals, not " +
                           std::to_string(N_) + " (see HLMEA-9). "
                           "set_W with a divisor of pop_size to avoid it"; });
        const auto& bd=vault.get_bounds(); std::uniform_real_distribution<double> d(0.0,1.0);
        std::vector<double> vars(vault.vars_n());
        for(int i=0;i<N_;++i){ for(int j=0;j<vault.vars_n();++j){double lo=bd[j].first.value_or(0.0),hi=bd[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);} vault.set_variables(i,vars);}
        vault.sync();
        std::vector<Sol> P; z_.assign(m_,std::numeric_limits<double>::max());
        for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            upd_ideal(s.objs); P.push_back(s);}
        allocate(P);
    }
    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0; delta_=0.0;
        c_old_.clear();   // stage-2 HV history is per-run state, not per-object
        int w_req=(W_req_>0)?W_req_:W_default();
        auto Vr=das_dennis::generate_auto(m_,w_req); V_.clear(); for(auto&v:Vr) V_.push_back(unit(v)); W_=(int)V_.size();
        S_=N_/W_; if(S_<1) S_=1;
        if(W_*S_ != N_)
            warn_lazy([&]{ return "hlmea: W=" + std::to_string(W_) +
                           " does not divide pop_size=" + std::to_string(N_) +
                           "; the answer set will hold W*floor(N/W)=" +
                           std::to_string(W_*S_) + " individuals, not " +
                           std::to_string(N_) + " (see HLMEA-9). "
                           "set_W with a divisor of pop_size to avoid it"; });
        std::vector<Sol> P; z_.assign(m_,std::numeric_limits<double>::max());
        for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            upd_ideal(s.objs); P.push_back(s);}
        allocate(P);
    }

    void step(DataVault<Ind_t>& vault){
        ++gen_;
        int scratch=vault.expand(1);
        const auto& bd=vault.get_bounds();
        bool stage1 = gen_ <= alpha_*t_max_;

        if(stage1){
            // FIX 2026-08-08: a per-generation uniform draw used to be taken and
            // discarded here (a leftover). Alg.3 line 1-3 draws only x_r and the
            // r inside Eq.4, so the extra draw implemented nothing; removing it
            // shifts the RNG stream but restores the paper's draw count.
            std::vector<Sol> Q;
            for(int w=0;w<W_;++w){
                auto& Pw=subpop_[w]; int sz=(int)Pw.size(); if(sz==0) continue;
                for(int j=0;j<sz;++j){
                    int rr=std::uniform_int_distribution<int>(0,sz-1)(rng_);
                    double rv=std::uniform_real_distribution<double>(0,1)(rng_);
                    double F0=(2*rv-1)*std::pow(std::max(1e-12,1-rv), -std::pow(1.0-(double)gen_/std::max(1,t_max_),0.8));
                    std::vector<double> y(vault.vars_n());
                    for(int t=0;t<vault.vars_n();++t) y[t]=Pw[j].vars[t]+F0*(Pw[j].vars[t]-Pw[rr].vars[t]);
                    Q.push_back(eval(y,bd,vault,scratch));
                }
            }
            auto P=flat(); for(auto&s:Q) P.push_back(s);
            allocate(P);
        } else {
            std::vector<Sol> Q;
            std::vector<std::vector<double>> c_new;       // cluster centres (objs)
            for(int w=0;w<W_;++w){
                auto& Pw=subpop_[w]; int sz=(int)Pw.size(); if(sz==0) continue;
                std::vector<int> kasg; kmeans(Pw,kasg,K_);
                int Keff=0; for(int a:kasg) Keff=std::max(Keff,a+1);
                // cluster centres in objective space
                std::vector<std::vector<double>> cs(Keff,std::vector<double>(m_,0.0)); std::vector<int> cc(Keff,0);
                for(int i=0;i<sz;++i){ ++cc[kasg[i]]; for(int k=0;k<m_;++k) cs[kasg[i]][k]+=Pw[i].objs[k]; }
                for(int c=0;c<Keff;++c) if(cc[c]>0){ for(int k=0;k<m_;++k) cs[c][k]/=cc[c]; c_new.push_back(cs[c]); }
                for(int j=0;j<sz;++j){
                    int ci=kasg[j];
                    std::vector<int> Cidx; for(int i=0;i<sz;++i) if(kasg[i]==ci) Cidx.push_back(i);
                    bool useC = std::uniform_real_distribution<double>(0,1)(rng_)<beta_;
                    std::vector<int> Pwall(sz); std::iota(Pwall.begin(),Pwall.end(),0);
                    std::vector<int>& pool = useC? Cidx : Pwall;   // mating pool B (Alg.2)
                    auto pick=[&](){ return pool[std::uniform_int_distribution<int>(0,(int)pool.size()-1)(rng_)]; };
                    std::vector<double> y(vault.vars_n());
                    if(delta_>=0){
                        int r1=pick(),r2=pick(),r3=pick();
                        for(int t=0;t<vault.vars_n();++t){
                            if(std::uniform_real_distribution<double>(0,1)(rng_)<=CR1_)
                                y[t]=Pw[r1].vars[t]+F1_*(Pw[r2].vars[t]-Pw[r3].vars[t]);
                            else y[t]=Pw[j].vars[t];
                        }
                    } else {
                        int r1=pick(),r2=pick();
                        for(int t=0;t<vault.vars_n();++t){
                            if(std::uniform_real_distribution<double>(0,1)(rng_)<=CR2_)
                                y[t]=Pw[j].vars[t]+F2_*(Pw[r1].vars[t]-Pw[r2].vars[t]);
                            else y[t]=Pw[j].vars[t];
                        }
                    }
                    Q.push_back(eval(y,bd,vault,scratch));
                }
            }
            // NSGA-II environment
            auto P=flat(); for(auto&s:Q) P.push_back(s);
            auto ord=nds_order(P);
            std::vector<Sol> np; for(int t=0;t<N_ && t<(int)ord.size();++t) np.push_back(P[ord[t]]);
            allocate(np);
            // δ is refreshed every 2 generations
            if(gen_%2==0){
                if(!c_old_.empty() && !c_new.empty()){
                    // ONE reference for both sets (see hv_ref / HLMEA-3).
                    std::vector<std::vector<double>> both=c_old_;
                    both.insert(both.end(), c_new.begin(), c_new.end());
                    auto ref=hv_ref(both);
                    double ho=hv_of(c_old_,ref), hn=hv_of(c_new,ref);
                    delta_ = (hn>1e-300)? (ho/hn - 1.0) : 0.0;
                }
                c_old_=c_new;
            }
        }
        store_arch(vault);
    }
};

} // namespace mootation

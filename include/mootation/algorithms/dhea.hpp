#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DHEA — Dynamic decomposition and hyper-distance based many-objective
//        evolutionary algorithm.
// X. Wang, F. Zhang, M. Yao — Complex & Intelligent Systems (2024);
// accepted 23 Aug 2024, published online 19 Dec 2024.
// doi:10.1007/s40747-024-01637-3          (source: s40747-024-01637-3)
//
// IDEA (FPS scaffold built from individuals + hAPD). No predefined reference
// vectors: "pivot solutions" are chosen by FPS on ANGLE (max angular distance
// to those already chosen), then the population is clustered by the nearest
// pivot. Inside each cluster the selection is by hAPD = (1+P(θ))·d_h, where
// d_h is the hyper-distance (projection onto the cluster's local hyperplane,
// convergence) and P(θ) is the angular penalty (diversity). The scalarization
// is hAPD, not a quality indicator.
//
// SCHEME (Algorithm 1): init; Knee=extreme, Div; loop: mating (binary
//   tournament on dominance→Knee→Div) → variation (SBX+PM) →
//   EnvironmentalSelection.
// EnvironmentalSelection (Alg.2): NDS(Q), keep the fronts up to the critical
//   one (≥N); normalization; Piv=extremes; while |Piv|<N: add the solution
//   with the max min-angle to Piv (FPS); clustering by the nearest pivot
//   (angle); pick 1 from each cluster (Alg.3).
// Selection (Alg.3): extremes — automatic; otherwise Case I (|cluster|=1)→it;
//   Case II (the local ideal coincides with a solution)→it; Case III → min
//   hAPD, Knee=argmin d_h (l.19), outward — the labels of the chosen ones
//   (l.27-28).
// hAPD (Eq.12-14): d_h=‖F_T‖·cos(F_T,w), F_T=F'−z^{l_ideal}, w=the local
//   nadir (max of the translated); P(θ)=m·(t/tmax)^α·sinθ,
//   θ=angle(F_T(solution),F_T(pivot)) — the apex is at the cluster's LOCAL
//   ideal (Fig.6: ∠eOf).
//
// PAPER DEFAULTS: α=2 — stated outright in §Experimental setup, item (3):
//   "In DHEA, α = 2 is the same as that in RVEA, which does not require special
//   design"; settable via set_alpha. N=pop_size; SBX η_c=20/pc=1;
//   PM η_m=20/pm=1/n. t_max — via set_t_max.
//
// CONFORMANCE NOTES (not deviations — recorded because the choice is visible
// in the code and a reader may wonder):
//   (a) Normalization follows Eq.(8) exactly — ideal = componentwise min,
//       nadir = componentwise max over the RETAINED pool (Alg.2 lines 2-3),
//       with no NSGA-III-style intercept construction anywhere. Degenerate
//       ranges (r ≤ 1e-12) map to F' = 0.
//   (b) The Eq.(9) ASF for the extremes is evaluated on that same F', not on
//       raw or merely translated objectives (Alg.2 line 3 precedes line 6).
//       The distinction is not cosmetic: without the range divisor the argmin
//       changes on any differently scaled objective set, and DHEA's own suite
//       includes WFG1-WFG9, where f_i ∈ [0, 2i]. The extremes seed Piv, are
//       auto-selected out of their clusters, and are labelled knee=true for the
//       mating tournament, so a wrong pick propagates through all three.
//
// DECLARED DEVIATIONS:
//   DHEA-1 (MINOR). Mating — binary tournament (dominance→Knee→Div); variation
//     SBX(first child)+PM; pairs taken sequentially from the mating pool.
//   DHEA-2 (MINOR). w (the local nadir) with protection against zero
//     components (1e-12); cos is clamped to [−1,1].
//   DHEA-3 (MINOR). real-valued genome; binary is beyond coverage.
//
// NOTABLE FIXES:
//   1) The ASF weights of the extremes (Eq.9) used to be inverted (1e-6 on the
//      axis of interest, 1.0 on the others); they are now w=1 when k==i and
//      1e-6 otherwise ("if i≠j, w=10⁻⁶, otherwise w=1"; the reference for the
//      same formula is maoea_3c.hpp).
//   2) θ in P(θ) (Eq.14) used to be computed between the globally normalized
//      vectors F; it is now the angle between F_T(solution)=F'−z^{l_ideal} and
//      F_T(pivot)=F'_piv−z^{l_ideal}, with the apex O at the cluster's local
//      ideal — the same local frame that d_h uses (Eq.13/14; Alg.3 l.20
//      transPopObj; Fig.6 ∠eOf). Implementation: th=angle(FT,FTpiv) in the
//      Case III loop.
//   3) Full life cycle of the Knee labels: init Knee(Extreme)=true (Alg.1 l.3,
//      PRESENT); Case I/II/extreme — knee=true on the selected one (PRESENT);
//      Case III — the label goes to argmin d_h (Alg.3 l.19) with the final
//      filter Knee=Knee(finalChoose) (Alg.3 l.28): the solution selected by
//      min hAPD carries knee ONLY if it is also argmin d_h (otherwise
//      knee=false); the random top-up (an extension beyond the paper) — an
//      explicit reset knee=false, so that a stale flag does not leak into the
//      mating tournament of the next generation.
//
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP switches every dominance test to Deb's constrained
//   domination: the Alg.2 front accumulation that decides which solutions
//   reach the pivot/clustering stage at all, and the first two levels of the
//   mating tournament. hAPD itself stays on raw objectives — it is a distance,
//   not a preference relation. The paper is unconstrained.
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
class DHEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double alpha_=2.0;
    double eta_c_=20.0, eta_m_=20.0, pc_=1.0, pm_=-1.0;
    int    t_max_=1000;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; double cv=0.0; bool knee=false; double div=0.0; };
    std::vector<Sol> pop_;
    int N_=0, m_=0, gen_=0;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }
    bool dom(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv); }
    static double angle(const std::vector<double>& a, const std::vector<double>& b){
        double d=0,na=0,nb=0; for(std::size_t k=0;k<a.size();++k){d+=a[k]*b[k];na+=a[k]*a[k];nb+=b[k]*b[k];}
        double q=std::sqrt(na)*std::sqrt(nb); if(q<1e-300) return 0.0; return std::acos(std::clamp(d/q,-1.0,1.0));
    }

    // normalization: F' = (f-ideal)/(nadir-ideal)
    std::vector<std::vector<double>> normalize(const std::vector<Sol>& P) const {
        int n=(int)P.size();
        std::vector<double> zmin(m_,1e300), zmax(m_,-1e300);
        for(auto&s:P) for(int k=0;k<m_;++k){ zmin[k]=std::min(zmin[k],s.objs[k]); zmax[k]=std::max(zmax[k],s.objs[k]); }
        std::vector<std::vector<double>> F(n,std::vector<double>(m_));
        for(int i=0;i<n;++i) for(int k=0;k<m_;++k){ double r=zmax[k]-zmin[k]; F[i][k]=(r>1e-12)?(P[i].objs[k]-zmin[k])/r:0.0; }
        return F;
    }
    std::vector<int> nds_fronts_until_N(const std::vector<Sol>& Q, std::vector<int>& crit_keep) const {
        int n=(int)Q.size();
        std::vector<int> dc(n,0); std::vector<std::vector<int>> dl(n);
        std::vector<int> f0;
        for(int p=0;p<n;++p){ for(int q=0;q<n;++q){ if(p==q) continue;
            if(dom(Q[p],Q[q])) dl[p].push_back(q); else if(dom(Q[q],Q[p])) ++dc[p]; }
            if(dc[p]==0) f0.push_back(p); }
        std::vector<int> keep; std::vector<int> cur=f0;
        while(!cur.empty()){
            // Alg.2 line 2: WHOLE fronts are accumulated until |keep| >= N;
            // the critical front is trimmed later, by Piv/clustering, not here.
            for(int p:cur) keep.push_back(p);
            if((int)keep.size()>=N_) break;
            std::vector<int> nx; for(int p:cur) for(int q:dl[p]) if(--dc[q]==0) nx.push_back(q); cur=std::move(nx);
        }
        crit_keep=keep; return keep;
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

    void compute_div(std::vector<Sol>& P, const std::vector<std::vector<double>>& F) const {
        int n=(int)P.size();
        for(int i=0;i<n;++i){
            std::vector<double> ang; ang.reserve(n);
            for(int j=0;j<n;++j) if(j!=i) ang.push_back(angle(F[i],F[j]));
            std::sort(ang.begin(),ang.end());
            int kk=std::min(m_,(int)ang.size()); double s=0; for(int t=0;t<kk;++t) s+=ang[t];
            P[i].div = kk>0? s/kk : 0.0;
        }
    }

    // extreme points (Eq.9): ASF(x, w_i) = max_k F'_k(x)/w_{i,k} with w=1 on the
    // axis of interest and 1e-6 on the others; argmin over the pool per axis.
    // The input is the Eq.(8)-normalized matrix, not raw objectives: Alg.2 runs
    // line 3 (normalization) before line 6 (extremes), and Eq.9 is written on
    // f'. Taking (f_k − z_k) without the range divisor would agree only when
    // every objective has the same range — see conformance note (b).
    std::vector<int> find_extremes(const std::vector<std::vector<double>>& F) const {
        int n=(int)F.size();
        std::vector<int> ext;
        for(int i=0;i<m_;++i){ int best=0; double bg=1e300;
            for(int j=0;j<n;++j){ double mx=-1e300; for(int k=0;k<m_;++k){double w=(k==i)?1.0:1e-6; double v=F[j][k]/w; if(v>mx)mx=v;} if(mx<bg){bg=mx;best=j;} }
            if(std::find(ext.begin(),ext.end(),best)==ext.end()) ext.push_back(best); }
        return ext;
    }

public:
    DHEACore() = default;
    void set_alpha(double a){ alpha_=a; }
    void set_t_max(int t){ if(t>0) t_max_=t; }
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double p){ pc_=p; }
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        const auto& bd=vault.get_bounds(); std::uniform_real_distribution<double> d(0.0,1.0);
        std::vector<double> vars(vault.vars_n());
        for(int i=0;i<N_;++i){ for(int j=0;j<vault.vars_n();++j){double lo=bd[j].first.value_or(0.0),hi=bd[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);} vault.set_variables(i,vars);}
        vault.sync();
        pop_.clear(); for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s);}
        auto F=normalize(pop_); compute_div(pop_,F);
        for(int e:find_extremes(F)) pop_[e].knee=true;      // Alg.1 l.3: Knee(Extreme)=true
    }
    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        pop_.clear(); for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            pop_.push_back(s);}
        auto F=normalize(pop_); compute_div(pop_,F);
        for(int e:find_extremes(F)) pop_[e].knee=true;      // Alg.1 l.3: Knee(Extreme)=true
    }

    void step(DataVault<Ind_t>& vault){
        ++gen_;
        int scratch=vault.expand(1);
        std::uniform_int_distribution<int> di(0,N_-1);
        // mating: binary tournament (dominance → knee → div)
        auto tour=[&](){ int a=di(rng_),b=di(rng_);
            if(dom(pop_[a],pop_[b])) return a;
            if(dom(pop_[b],pop_[a])) return b;
            if(pop_[a].knee!=pop_[b].knee) return pop_[a].knee?a:b;
            return (pop_[a].div>pop_[b].div)?a:b; };
        std::vector<Sol> O; O.reserve(N_);
        for(int i=0;i<N_;++i){ int p1=tour(),p2=tour(); O.push_back(breed(pop_[p1],pop_[p2],vault,scratch)); }

        // env selection
        std::vector<Sol> Q=pop_; for(auto&s:O) Q.push_back(s);
        std::vector<int> keepIdx; nds_fronts_until_N(Q,keepIdx);
        std::vector<Sol> P; for(int i:keepIdx) P.push_back(Q[i]);
        int np=(int)P.size();
        auto F=normalize(P);

        // extreme points (Eq.9): min ASF along the axes
        std::vector<int> ext=find_extremes(F);

        // dyn. decomposition: pivot = FPS on angle
        int Npiv=std::min(N_,np);
        std::vector<char> isPiv(np,0); std::vector<int> piv;
        for(int e:ext){ if(!isPiv[e]){isPiv[e]=1;piv.push_back(e);} }
        std::vector<double> dmin(np,1e300);
        for(int j=0;j<np;++j){ if(isPiv[j]) continue; double mn=1e300; for(int p:piv) mn=std::min(mn,angle(F[j],F[p])); dmin[j]=mn; }
        while((int)piv.size()<Npiv){
            int best=-1; double bd=-1; for(int j=0;j<np;++j){ if(isPiv[j]) continue; if(dmin[j]>bd){bd=dmin[j];best=j;} }
            if(best<0) break;
            isPiv[best]=1; piv.push_back(best);
            for(int j=0;j<np;++j){ if(isPiv[j]) continue; dmin[j]=std::min(dmin[j],angle(F[j],F[best])); }
        }
        // clustering: each non-pivot → the nearest pivot
        int P_=(int)piv.size();
        std::vector<std::vector<int>> cluster(P_);
        std::vector<int> pivOfIdx(np,-1); for(int c=0;c<P_;++c){ pivOfIdx[piv[c]]=c; cluster[c].push_back(piv[c]); }
        for(int j=0;j<np;++j){ if(isPiv[j]) continue;
            int best=0; double ba=angle(F[j],F[piv[0]]); for(int c=1;c<P_;++c){ double a=angle(F[j],F[piv[c]]); if(a<ba){ba=a;best=c;} }
            cluster[best].push_back(j); }

        // pick 1 from the cluster
        std::vector<char> isExt(np,0); for(int e:ext) isExt[e]=1;
        std::vector<Sol> next; next.reserve(N_);
        double Pt=std::pow(std::min(1.0,(double)gen_/std::max(1,t_max_)),alpha_);
        for(int c=0;c<P_ && (int)next.size()<N_;++c){
            auto& cl=cluster[c];
            // if the pivot is an extreme: auto-select the extreme
            int extInCl=-1; for(int idx:cl) if(isExt[idx]) extInCl=idx;
            if(extInCl>=0){ Sol s=P[extInCl]; s.knee=true; next.push_back(s); continue; }
            if((int)cl.size()==1){ Sol s=P[cl[0]]; s.knee=true; next.push_back(s); continue; }
            // local ideal
            std::vector<double> li(m_,1e300); for(int idx:cl) for(int k=0;k<m_;++k) li[k]=std::min(li[k],P[idx].objs[k]);
            int coincide=-1; for(int idx:cl){ bool eq=true; for(int k=0;k<m_;++k) if(std::abs(P[idx].objs[k]-li[k])>1e-12){eq=false;break;} if(eq){coincide=idx;break;} }
            if(coincide>=0){ Sol s=P[coincide]; s.knee=true; next.push_back(s); continue; }
            // Case III: hAPD. F_T=F'-li' (on the normalized ones). local nadir = max translated
            std::vector<double> li2(m_,1e300); for(int idx:cl) for(int k=0;k<m_;++k) li2[k]=std::min(li2[k],F[idx][k]);
            std::vector<double> w(m_,0.0); for(int idx:cl) for(int k=0;k<m_;++k) w[k]=std::max(w[k],F[idx][k]-li2[k]);
            for(int k=0;k<m_;++k) if(w[k]<1e-12) w[k]=1e-12;
            int pivc=piv[c];
            // θ is computed in the cluster's LOCAL ideal frame (apex O at z^{l_ideal}) — the
            //   angle between F_T(solution)=F'-li2 and F_T(pivot)=F'_piv-li2, as for d_h
            //   (Eq.13/14; Alg.3 l.20 transPopObj; Fig.6 ∠eOf). Previously — the global F.
            std::vector<double> FTpiv(m_); for(int k=0;k<m_;++k) FTpiv[k]=F[pivc][k]-li2[k];
            // full Knee cycle in Case III — the label goes to argmin d_h (Alg.3 l.19),
            //   filter Knee=Knee(finalChoose) (Alg.3 l.28): the selected solution
            //   (min hAPD) carries knee ONLY if it is also argmin d_h, else knee is reset.
            int best=-1; double bg=1e300;
            int kneeIdx=-1; double bdh=1e300;
            for(int idx:cl){
                std::vector<double> FT(m_); for(int k=0;k<m_;++k) FT[k]=F[idx][k]-li2[k];
                double nft=0; for(double v:FT) nft+=v*v; nft=std::sqrt(nft);
                double dotw=0,nw=0; for(int k=0;k<m_;++k){dotw+=FT[k]*w[k];nw+=w[k]*w[k];}
                double cosw = (nft*std::sqrt(nw)>1e-300)? std::clamp(dotw/(nft*std::sqrt(nw)),-1.0,1.0):0.0;
                double dh = nft*cosw;
                double th = angle(FT,FTpiv);   // local frame: ∠(F_T(solution),F_T(pivot))
                double Pth = m_*Pt*std::sin(th);
                double hapd=(1.0+Pth)*dh;
                if(dh<bdh){bdh=dh;kneeIdx=idx;}          // Alg.3 l.19: Knee on argmin d_h
                if(hapd<bg){bg=hapd;best=idx;}           // Alg.3 l.23: finalChoose at min hAPD
            }
            if(best<0) best=cl[0];
            Sol s=P[best]; s.knee=(best==kneeIdx); next.push_back(s);  // Alg.3 l.28: knee only if the selected = argmin d_h
        }
        // top-up on a shortfall (extension beyond the paper): a copy WITHOUT the
        // knee label — an explicit reset, so that a stale knee flag does not leak
        // into the mating tournament (Alg.3 l.2/l.28: knee only on the ones
        // selected by the algorithm).
        while((int)next.size()<N_ && np>0){ Sol s=P[std::uniform_int_distribution<int>(0,np-1)(rng_)]; s.knee=false; next.push_back(s); }
        if((int)next.size()>N_) next.resize(N_);
        pop_=next;
        auto F2=normalize(pop_); compute_div(pop_,F2);
        store_arch(vault);
    }
};

} // namespace mootation

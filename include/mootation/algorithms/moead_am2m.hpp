#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/D-AM2M — the algorithm of "Adaptively Allocating Search Effort in
// Challenging Many-Objective Optimization Problems" (M2M + adaptive direction
// vectors/weights derived from individuals; the AM2M name is the paper's own,
// the descriptive subtitle is this library's).
// H.-L. Liu, L. Chen, Q. Zhang, K. Deb — IEEE Transactions on Evolutionary
// Computation 22(3), 2018, pp. 433-448 (journal version of conf. liu2016).
// doi:10.1109/TEVC.2017.2725902     (source: liu2018, conf: liu2016)
//
// IDEA. The MOEA/D-M2M framework, but the direction vectors v¹..v^K and the
// weights w^{k,j} are NOT fixed by a lattice — they are ADAPTIVELY built from
// the current population by the Max-Min method (FPS by angle): directions from
// the whole pool, weights from the individuals inside a subregion. Selection
// inside a subregion — Tchebycheff (Eq.4).
//
// SCHEME (Algorithm 1, journal):
//   Init («Uniformly initialize direction/weight vectors and population, use
//   them to set subpopulation P_k», gen=1): K UNIFORM directions and N
//   UNIFORM weights (§V-C: Das–Dennis lattice, two-layer if needed);
//   the weights are distributed over the subregions (max cos to V), the holder
//   of each weight is the best-by-Tchebycheff member of the initial population.
//   FIX 2026-07-07 (source-fidelity review): previously the initialization was
//   ADAPTIVE (adapt() on a random population) contrary to Alg.1/§V-C and
//   undeclared. while not stop:
//     Q ← one offspring per member of P (II-C); if mod(gen,G)==0 → adaptation
//     (III-A/B, reseating); else → split Q by V and update the holders (II-D).
//
// NORMALIZATION (FIX 2026-06-13, AM2M-norm). All geometry (directions,
// Ω_k association, weights, Tchebycheff) is computed in the normalized space
// f'=(f−z^{min})/(z^{nad}−z^{min}), where z^{min}/z^{nad} is a snapshot
// frozen at initialization (over the initial P) and afterwards ONLY at each
// adaptation (over P∪Q). The paper introduces no normalization at all
// (angles/Tchebycheff Eq.4 are in raw objectives w.r.t. z); the normalization
// is our own deviation AM2M-norm. Without it, early badly-scaled generations
// (large g ⇒ f2≫f1) project onto the unit sphere into ONE bundle, the Max-Min
// directions collapse near a single axis and do NOT recover → diversity
// collapse. (M2M is protected from this by the fixed Das–Dennis lattice; the
// adaptive AM2M is not.)
// FIX 2026-07-07 (source-fidelity review): frame drift removed — previously
// step() re-took the snapshot set_norm(P) EVERY generation, contrary to this
// very declaration (the weights were applied in a "foreign" coordinate frame).
//
// REPRODUCTION OPERATOR (§V-C of liu2018): «The crossover and mutation
//   operators with the same control parameters in MOEA/D-M2M [8] … are used in
//   MOEA/D-AM2M». Reference [8] = liu2014 (M2M), whose operators (see
//   moead_m2m.hpp §III-A(1)) are Liu & Li 2009 [20 in liu2014]: the annealed
//   arithmetic crossover Eq.(5) z = x + rc·(x − y) and the mutation Eq.(6),
//   rc/rm decay with gen/Max_gen (operators/liuli_crossover.hpp). That is, §V-C
//   transitively prescribes the Liu–Li operator for AM2M, and NOT SBX/PM.
//   FIX 2026-07-08 (wave 2 of the source-fidelity review):
//   breed() switched from SBX+PM to Liu–Li Eq.(5)-(6) — per §V-C via M2M [8]
//   (the same operator and the same fix as in moead_m2m.hpp, wave 1). The former
//   declaration AM2M-3 «η_c=20, operators as in M2M [8]» was a false
//   attribution: there is no SBX/PM in M2M [8]/[20]. The annealing needs
//   Max_gen: set_t_max(int), default 1000 (library convention); the caller must
//   set the real budget, otherwise for gen ≥ t_max the operator degenerates
//   into copying the parent.
//
// DEFAULTS (§V-C): Liu–Li operator Eq.(5)-(6) (p_m=1/n, Max_gen=set_t_max,
//   default 1000); K=set_K; G=100; N=pop_size.
//
// DECLARED DEVIATIONS:
//   AM2M-1 (DEVIATION). α (probability of within-subpopulation mating) is not
//     specified by the paper → 0.7.
//   AM2M-2 (MINOR). Weights = L1 normalization of the Max-Min-selected
//     individuals (Eq.4 Σw=1).
//   AM2M-3 (retired, FIX 2026-07-08). Previously: «η_c=20, operators as in
//     M2M [8]» — falsely attributed SBX/PM to source M2M [8]; §V-C via
//     [8]=liu2014 prescribes the Liu–Li operator (see the REPRODUCTION OPERATOR
//     block above).
//   AM2M-norm (FIX). Geometry in the normalized [0,1] space (see above);
//     the paper says «project to unit sphere» with no explicit normalization,
//     but on problems with badly scaled objectives the normalization is
//     necessary for the adaptation to work.
//     The frame is fixed at init and at adaptations (since 2026-07-07 — without
//     drift).
//   AM2M-5 (MINOR). S_k>|P̃_k| → Max-Min fill-up by repetition; S_k=0 → the
//     subregion is empty (after the uniform initialization this is possible
//     even before the first adaptation: an empty weight bucket of a direction).
//   AM2M-6 (MINOR). The Liu–Li operator yields ONE offspring z=x+rc·(x−y) from
//     the base x; y≠x for within-subregion mating. (FIX 2026-07-08: previously
//     «SBX→first child» — obsolete after switching the operator to Liu–Li, see
//     the REPRODUCTION OPERATOR block.)
//   AM2M-7 (MINOR). Uniform initial weights: das_dennis::generate_auto(m,N)
//     may return slightly more than N vectors (fallback lattice) — truncated
//     to N; weight holders are chosen as the Tchebycheff argmin without removal
//     (duplicates are possible — just as in adapt()). K directions: the actual
//     K = the lattice size (as in moead_m2m).
//   AM2M-8 (MINOR). §III-B specifies only an INCREMENT loop for the slot quota
//     (steps 1-4: S_k = ⌊|P̃_k|/2⌋, except S_k = 1 when |P̃_k| = 1; then raise
//     the largest S_k until Σ S_k = |P|). The |P̃_k| = 1 rule can push Σ S_k
//     ABOVE |P|, a case the paper never addresses, so a symmetric DECREMENT
//     loop was added. That loop is bounded (bail-out at 10K+10 iterations) and
//     can therefore terminate with Σ S_k > N when K is large and one subregion
//     dominates — in which case the active population after step() exceeds
//     pop_size(), because the store writes exactly Σ S_k slots.
//
// CONSTRAINTS (beyond the paper, off by default). AM2M's only preference
//   relation is the per-weight Tchebycheff argmin that picks each holder, so
//   constraint_mode FEASIBILITY/CDP makes THAT comparison feasibility-first
//   (AM2M-C): feasible beats infeasible, two infeasible compare by CV, two
//   feasible compare by g. The adaptive directions, the Max-Min weight
//   construction and the normalization frame are geometry and stay as they
//   are. The paper is unconstrained.
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
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"
// FIX 2026-07-08 (wave 2 of the source-fidelity review):
// SBX/PM in breed() replaced with the Liu & Li 2009 operator Eq.(5)-(6), as
// required by §V-C of liu2018 via M2M [8] (see header, REPRODUCTION OPERATOR
// block).
#include "../operators/liuli_crossover.hpp"

namespace mootation {

template <typename Ind_t>
class MOEADAM2MCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int    K_req_ = 10;
    int    G_     = 100;
    double alpha_ = 0.7;
    double eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0, pm_ = -1.0;
    // FIX 2026-07-08: Max_gen for the annealing of the Liu–Li operator [20]
    // (§V-C via M2M [8]); default 1000 is the library convention
    // (moead_m2m/moead_awa/adaw), the caller sets the real budget via set_t_max.
    int    t_max_ = 1000;
    std::mt19937 rng_{std::random_device{}()};

    struct Sol { std::vector<double> vars, objs; std::vector<int> bvars; double cv=0.0; };

    std::vector<std::vector<double>>                 V_;     // [K] direction vectors (unit, in normalized space)
    std::vector<std::vector<std::vector<double>>>    Wk_;    // [K][S_k] weights (Σ=1, in normalized space)
    std::vector<std::vector<Sol>>                    hold_;  // [K][S_k] holders
    std::vector<double>                              z_;     // ideal point (global, monotone)
    std::vector<double>                              zn_, range_;  // normalization snapshot (ideal, nadir-ideal)
    int K_ = 0, m_ = 0, N_ = 0, gen_ = 0;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }

    static double cosine(const std::vector<double>& a, const std::vector<double>& b) {
        double dot=0,na=0,nb=0;
        for (std::size_t i=0;i<a.size();++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
        double d=std::sqrt(na)*std::sqrt(nb);
        if (d<1e-300) return 1.0;
        double c=dot/d; return std::clamp(c,-1.0,1.0);
    }
    static std::vector<double> unit(const std::vector<double>& f){
        double n=0; for(double v:f) n+=v*v; n=std::sqrt(std::max(n,1e-300));
        std::vector<double> u(f.size()); for(std::size_t i=0;i<f.size();++i) u[i]=f[i]/n; return u;
    }
    static std::vector<double> l1(const std::vector<double>& f){
        double s=0; for(double v:f) s+=std::abs(v); if(s<1e-300) s=1.0;
        std::vector<double> w(f.size());
        for(std::size_t i=0;i<f.size();++i) w[i]=std::max(std::abs(f[i])/s,1e-6);
        return w;
    }
    // normalized objective f' = (f−zn_)/range_, clamped ≥0
    std::vector<double> norm(const std::vector<double>& f) const {
        std::vector<double> r(m_);
        for(int k=0;k<m_;++k) r[k]=std::max(0.0,(f[k]-zn_[k])/range_[k]);
        return r;
    }

    // Max-Min (FPS by angle): pick cnt indices out of the unit vectors pts.
    std::vector<int> max_min(const std::vector<std::vector<double>>& pts, int cnt){
        int n=(int)pts.size();
        std::vector<int> sel;
        if (n==0||cnt<=0) return sel;
        std::vector<char> used(n,0);
        std::uniform_int_distribution<int> ri(0,n-1);
        int r=ri(rng_);
        int best=-1; double bc=2.0;
        for(int i=0;i<n;++i){ double c=cosine(pts[i],pts[r]); if(c<bc){bc=c;best=i;} }
        sel.push_back(best); used[best]=1;
        while((int)sel.size()<cnt){
            int b=-1; double bv=2.0;
            for(int i=0;i<n;++i){
                if(used[i]) continue;
                double maxc=-2.0;
                for(int s:sel){ double c=cosine(pts[i],pts[s]); if(c>maxc) maxc=c; }
                if(maxc<bv){ bv=maxc; b=i; }
            }
            if(b<0) break;
            sel.push_back(b); used[b]=1;
        }
        for(int i=0;(int)sel.size()<cnt;++i) sel.push_back(sel[i%std::max(1,(int)sel.size())]);
        return sel;
    }

    int assoc(const std::vector<double>& f) const {
        auto fn=norm(f);
        int best=0; double bestc=-2.0;
        for(int k=0;k<K_;++k){ double c=cosine(fn,V_[k]); if(c>bestc){bestc=c;best=k;} }
        return best;
    }
    // Tchebycheff in the normalized space: g = max_i f'_i / w_i
    double tcheby(const std::vector<double>& f, const std::vector<double>& w) const {
        auto fn=norm(f);
        double g=-std::numeric_limits<double>::max();
        for(int i=0;i<m_;++i){ double v=fn[i]/std::max(w[i],1e-6); if(v>g) g=v; }
        return g;
    }
    void upd_ideal(const std::vector<double>& f){
        for(int i=0;i<m_;++i) z_[i]=std::min(z_[i],f[i]);
    }
    // normalization snapshot over the pool big
    void set_norm(const std::vector<Sol>& big){
        zn_.assign(m_, std::numeric_limits<double>::max());
        std::vector<double> nad(m_,-std::numeric_limits<double>::max());
        for(const auto& s:big) for(int k=0;k<m_;++k){ zn_[k]=std::min(zn_[k],s.objs[k]); nad[k]=std::max(nad[k],s.objs[k]); }
        range_.assign(m_,1.0);
        for(int k=0;k<m_;++k) range_[k]=std::max(nad[k]-zn_[k],1e-12);
    }

    // FIX 2026-07-08 (wave 2 of the source-fidelity review):
    // was SBX+PM. §V-C of liu2018 via M2M [8] prescribes the
    // Liu & Li 2009 operator Eq.(5)-(6): one offspring z = x + rc·(x − y), base
    // x ∈ P_k, rc/rm annealing by gen_/t_max_. eta_c_/pc_ are not used by this
    // operator (the setters are kept for API compatibility).
    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch){
        int nv=vault.vars_n(); const auto& b=vault.get_bounds();
        std::vector<double> c1;
        ops::liuli_crossover(x.vars,y.vars,c1,b,gen_,t_max_,rng_);
        ops::liuli_mutation(c1,b,pm_eff(nv),gen_,t_max_,rng_);
        Sol z; z.vars=c1;
        if(vault.bin_vars_n()>0){
            std::vector<int> bc1,bc2;
            ops::binary_crossover(x.bvars,y.bvars,bc1,bc2,rng_);
            ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
            z.bvars=bc1;
            vault.set_all_variables(scratch,c1,bc1);
        } else {
            vault.set_variables(scratch,c1);
        }
        vault.refresh_objectives(scratch);
        z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        upd_ideal(z.objs);
        return z;
    }

    std::vector<Sol> flat_pop() const {
        std::vector<Sol> P;
        for(int k=0;k<K_;++k) for(const auto& s:hold_[k]) P.push_back(s);
        return P;
    }

    // adaptation of V/S_k/weights/holders from the merged pool big (P∪Q)
    void adapt(const std::vector<Sol>& big){
        set_norm(big);   // freeze the normalization for this cycle
        // 1) Max-Min directions from big (normalized objectives projected onto the sphere)
        std::vector<std::vector<double>> proj; proj.reserve(big.size());
        for(const auto& s:big) proj.push_back(unit(norm(s.objs)));
        int K=std::min(K_req_,(int)big.size());
        std::vector<int> di=max_min(proj,K);
        V_.clear(); for(int idx:di) V_.push_back(unit(norm(big[idx].objs)));
        K_=(int)V_.size();

        // 2) association of big with V
        std::vector<std::vector<int>> bucket(K_);
        for(int i=0;i<(int)big.size();++i) bucket[assoc(big[i].objs)].push_back(i);

        // 3) S_k
        std::vector<int> S(K_,0);
        for(int k=0;k<K_;++k){ int sz=(int)bucket[k].size(); S[k]=(sz==1)?1:sz/2; }
        int sum=0; for(int s:S) sum+=s;
        if(sum<N_){
            std::vector<int> order;
            for(int k=0;k<K_;++k) if(S[k]>0) order.push_back(k);
            std::sort(order.begin(),order.end(),[&](int a,int b){return S[a]<S[b];});
            if(order.empty()) order.push_back(0);
            int idx=0; while(sum<N_){ ++S[order[idx%order.size()]]; ++sum; ++idx; }
        } else if(sum>N_){
            std::vector<int> order;
            for(int k=0;k<K_;++k) order.push_back(k);
            std::sort(order.begin(),order.end(),[&](int a,int b){return S[a]>S[b];});
            int idx=0; while(sum>N_){ if(S[order[idx%order.size()]]>1){--S[order[idx%order.size()]];--sum;} ++idx; if(idx>10*K_+10) break; }
        }

        // 4) weights and holders (all in the normalized space)
        Wk_.assign(K_,{}); hold_.assign(K_,{});
        for(int k=0;k<K_;++k){
            if(S[k]<=0) continue;
            const auto& bk=bucket[k];
            std::vector<std::vector<double>> pk; pk.reserve(bk.size());
            for(int i:bk) pk.push_back(unit(norm(big[i].objs)));
            std::vector<int> wi = pk.empty() ? std::vector<int>{} : max_min(pk,S[k]);
            for(int j=0;j<S[k];++j){
                if(!wi.empty()) Wk_[k].push_back(l1(norm(big[bk[wi[j]]].objs)));
                else            Wk_[k].push_back(std::vector<double>(m_,1.0/m_));
            }
            for(int j=0;j<S[k];++j){
                const auto& w=Wk_[k][j];
                // Holder = argmin Tchebycheff. Under FEASIBILITY/CDP the
                // comparison is feasibility-first (AM2M-C), so an infeasible
                // candidate can hold a weight only if nothing feasible is in
                // the pool.
                int best=-1; double bg=std::numeric_limits<double>::max();
                double bcv=std::numeric_limits<double>::max();
                auto consider=[&](int i){
                    double g=tcheby(big[i].objs,w);
                    if(best<0 || detail::better_scalar(constraint_mode,g,big[i].cv,bg,bcv)){
                        bg=g; bcv=big[i].cv; best=i;
                    }
                };
                if(!bk.empty()){ for(int i:bk) consider(i); }
                else           { for(int i=0;i<(int)big.size();++i) consider(i); }
                hold_[k].push_back(big[best]);
            }
        }
    }

    // FIX 2026-07-07 (source-fidelity review): uniform initialization per Alg.1
    // («Uniformly initialize direction/weight vectors») and §V-C (Das–Dennis,
    // two-layer if needed) instead of the adaptive one.
    // The holder of each weight is the best-by-Tchebycheff member of the
    // initial population P (the same rule as in adapt(); duplicates are
    // possible, AM2M-7).
    void uniform_init(const std::vector<Sol>& P){
        set_norm(P);   // AM2M-norm: the frame is fixed here, afterwards — at adaptations
        // 1) K uniform directions: Das–Dennis lattice → unit sphere
        auto Vr = das_dennis::generate_auto(m_, K_req_);
        V_.clear(); V_.reserve(Vr.size());
        for (auto& v : Vr) V_.push_back(unit(v));
        K_ = (int)V_.size();
        // 2) N uniform weights (§V-C); the fallback lattice may yield >N → truncation
        auto Wl = das_dennis::generate_auto(m_, N_);
        if ((int)Wl.size() > N_) Wl.resize(N_);
        // 3) weights over the subregions: max cos to a direction (Eq.4: w∈Ω_k);
        //    l1 — Σw=1, clamp w_i≥1e-6 (Eq.4 requires w>0)
        Wk_.assign(K_,{}); hold_.assign(K_,{});
        for (const auto& w : Wl) {
            int best=0; double bc=-2.0;
            for(int k=0;k<K_;++k){ double c=cosine(w,V_[k]); if(c>bc){bc=c;best=k;} }
            Wk_[best].push_back(l1(w));
        }
        // 4) weight holders from the initial population («use them to set P_k»)
        for(int k=0;k<K_;++k)
            for(const auto& w : Wk_[k]){
                int best=0; double bg=std::numeric_limits<double>::max();
                for(int i=0;i<(int)P.size();++i){
                    double g=tcheby(P[i].objs,w);
                    if(g<bg){bg=g;best=i;}
                }
                hold_[k].push_back(P[best]);
            }
    }

public:
    MOEADAM2MCore() = default;
    void set_K(int k){ K_req_=k; }
    void set_n_clusters(int k){ K_req_=k; }
    void set_G(int g){ if(g>0) G_=g; }
    // FIX 2026-07-08: Max_gen of the annealing of the Liu–Li operator [20]
    // (§V-C via M2M [8]).
    void set_t_max(int t){ if(t>0) t_max_=t; }
    void set_alpha(double a){ alpha_=a; }
    // eta_c_/pc_ are not used by the Liu–Li operator; the setters are kept for
    // API compatibility (no effect on reproduction, FIX 2026-07-08).
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double p){ pc_=p; }
    void set_pm(double p){ pm_=p; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        const auto& bounds=vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0,1.0);
        std::uniform_int_distribution<int> db(0,1);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bv(vault.bin_vars_n());
        for(int i=0;i<N_;++i){
            for(int j=0;j<vault.vars_n();++j){double lo=bounds[j].first.value_or(0.0),hi=bounds[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);}
            for(int j=0;j<vault.bin_vars_n();++j) bv[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bv); else vault.set_variables(i,vars);
        }
        vault.sync();
        z_.assign(m_,std::numeric_limits<double>::max());
        std::vector<Sol> P; P.reserve(N_);
        for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            upd_ideal(s.objs); P.push_back(s); }
        uniform_init(P);   // FIX 2026-07-07: Alg.1/§V-C (was adapt(P))
        store_arch(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); gen_=0;
        z_.assign(m_,std::numeric_limits<double>::max());
        std::vector<Sol> P; P.reserve(N_);
        for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i); s.bvars=vault.binary_variables_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            upd_ideal(s.objs); P.push_back(s); }
        uniform_init(P);   // FIX 2026-07-07: Alg.1/§V-C (was adapt(P))
        store_arch(vault);
    }

    void store_arch(DataVault<Ind_t>& vault){
        std::vector<Sol> P=flat_pop();
        vault.reduce(0); vault.expand((int)P.size());
        for(int i=0;i<(int)P.size();++i)
            vault.seed_individual((std::size_t)i,P[i].vars,P[i].objs,P[i].bvars,{});
    }

    void step(DataVault<Ind_t>& vault){
        ++gen_;
        std::uniform_real_distribution<double> uni(0.0,1.0);
        int scratch=vault.expand(1);

        std::vector<Sol> Q; Q.reserve(N_);
        std::vector<Sol> Pflat=flat_pop();
        // FIX 2026-07-07 (source-fidelity review): set_norm(Pflat) used to be
        // here every generation — frame drift contrary to the AM2M-norm
        // declaration; the z^min/z^nad snapshot now changes only in
        // uniform_init() and adapt() (the weights and geometry stay in the
        // frame of the last adaptation).
        for(int k=0;k<K_;++k){
            int nk=(int)hold_[k].size();
            for(int j=0;j<nk;++j){
                const Sol& x=hold_[k][j];
                Sol y;
                if(uni(rng_)<alpha_ && nk>1){
                    std::uniform_int_distribution<int> dj(0,nk-1);
                    int yy=dj(rng_); for(int a=0;a<5&&yy==j;++a) yy=dj(rng_);
                    y=hold_[k][yy];
                } else {
                    std::uniform_int_distribution<int> dp(0,(int)Pflat.size()-1);
                    int tries=0,pi; do{ pi=dp(rng_); ++tries; } while(tries<10 && assoc(Pflat[pi].objs)==k);
                    y=Pflat[pi];
                }
                Q.push_back(breed(x,y,vault,scratch));
            }
        }

        if(gen_%G_==0){
            std::vector<Sol> big=flat_pop();
            big.insert(big.end(),Q.begin(),Q.end());
            adapt(big);
        } else {
            std::vector<std::vector<int>> Qk(K_);
            for(int i=0;i<(int)Q.size();++i) Qk[assoc(Q[i].objs)].push_back(i);
            for(int k=0;k<K_;++k){
                int nk=(int)hold_[k].size();
                if(nk==0) continue;
                for(int qi:Qk[k]){
                    const Sol& z=Q[qi];
                    // Which holders the offspring beats. Under FEASIBILITY/CDP
                    // the comparison is feasibility-first (AM2M-C): a feasible
                    // holder is never displaced by an infeasible offspring, and
                    // two infeasible solutions compare by CV.
                    std::vector<int> beat;
                    for(int j=0;j<nk;++j){
                        double gz=tcheby(z.objs,Wk_[k][j]);
                        double gh=tcheby(hold_[k][j].objs,Wk_[k][j]);
                        bool better;
                        if(constraint_mode!=ConstraintMode::NONE)
                            better=detail::better_scalar(constraint_mode,
                                                         gz,z.cv,gh,hold_[k][j].cv);
                        else
                            better=(gz<gh);
                        if(better) beat.push_back(j);
                    }
                    if(!beat.empty()){
                        std::uniform_int_distribution<int> dbi(0,(int)beat.size()-1);
                        hold_[k][beat[dbi(rng_)]]=z;
                    }
                }
            }
        }
        store_arch(vault);
    }
};

} // namespace mootation
// (FIX 2026-07-08, wave 2 of the source-fidelity review: see header — the
// Liu–Li operator instead of SBX/PM.)

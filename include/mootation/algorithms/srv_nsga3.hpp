#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// NSGA-III/S — NSGA-III with Self-Guided Reference Vectors (SRV)
// Liu, Lin, Wong, Coello Coello, Li, Ming, Zhang — IEEE Trans. Cybernetics, 2022
// doi:10.1109/TCYB.2020.2971638
//
// Generational scheme (§III-C, Fig.5: SRV is embedded into NSGA-III "without
// changing other procedures"):
//   1. Tournament (random on a tie) -> SBX + PM -> Q; U = P + Q.
//   2. Non-dominated sorting of U -> St (plus the critical front Fl);
//      Pc = St u Fl.
//   3. theta_c per Eq.17 (G = gen+1, theta_c^min per Eq.16 from the original
//      Das-Dennis lattice); V = SRVStrategy::extract(Pc, N, theta_c), every
//      generation (tau = 1).
//   4. NSGA-III normalization: z_min, the ASF extremes, the hyperplane and its
//      intercepts; on degeneracy or negative intercepts, z^nad = the max over
//      Pc.
//   5. Association by the perpendicular distance d2 to the lines of V; NSGA-III
//      niching on the critical front (minimum-rho niche, nearest or random).
//
// PAPER DEFAULTS (§IV-B): the operators follow NSGA-III [21] — SBX p_c=1.0 /
//   eta_c=30, PM p_m=1/n / eta_m=20. Table A.II of the supplementary file is
//   unavailable, so the [21] convention is used.
// DECLARED DEVIATIONS: the SRVs are extracted under the max-normalization of
//   srv_strategy while the association runs under the NSGA-III intercept
//   normalization (SRVN-2, a mixing of spaces); set_kmeans_iter is a no-op,
//   since the iteration count is fixed at 2m by Alg.3.
//   SRVN-5: the ideal z^min and the ASF extremes of the NSGA-III intercept
//   normalization now follow the CUMULATIVE scheme of deb2014 §IV-C ("ever
//   found from the start of the simulation"); they used to be computed over the
//   current S_t, which was an undeclared deviation. The reference is nsga3.hpp.
//   G_max DEPENDENCE (contract, not optional). The theta_c annealing schedule
//     (Eq.17) is driven by t_max_, which defaults to 1000. A caller who never
//     calls set_t_max therefore runs a schedule computed against a budget that
//     has nothing to do with the real one: at 200 generations the anneal never
//     leaves its early regime, and past 1000 it saturates. set_t_max is
//     effectively mandatory for reproducing the paper.

//   Note that the SRV extraction in srv_strategy keeps its own max
//   normalization (SRVN-2); this change touches only the NSGA-III
//   association and niching path.
// EXTENSIONS BEYOND THE PAPER (off by default): binary variables,
//   constraint_mode FEASIBILITY.
// ============================================================================
// Notable fixes: (1) setup_seeded did not compute theta_min_c_, so theta_c = 0
// and rho(x) evaluated to NaN in the first generation (SRVN-1); init_V now
// computes theta_min, as in srv.hpp; (2) theta_min per Eq.16 uses true angles
// (srv_strategy.hpp); (3) negative or missing intercepts did not produce the
// fallback z^nad = max over Pc that the paper requires — a per-objective
// fallback was added, where previously only a singular matrix was caught.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"
#include "srv_strategy.hpp"

namespace mootation {

template <typename Ind_t>
class SRVNSGA3Core {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_   = 30.0;  // NSGA-III [21] convention (Table A.II, suppl.)
    double       eta_m_   = 20.0;
    double       pc_      = 1.0;   // SBX pair probability (NSGA-III [21])
    int          t_max_   = 1000;
    std::mt19937 rng_{std::random_device{}()};
    int          current_gen_ = 0;

    SRVStrategy<Ind_t> srv_;
    std::vector<std::vector<double>> V_;     // current unit reference vectors
    double theta_min_c_ = 0.0;

    // ── Persistent NSGA-III normalization state (deb2014 §IV-C) ───────────
    // The ideal z^min and the ASF extremes are CUMULATIVE ("ever found from the
    //   start of the simulation", Alg.2), not per current S_t. The reference
    //   for the cumulative scheme is nsga3.hpp (zmin_hist_ / extreme_hist_).
    std::vector<double>              zmin_hist_;     // historical ideal point
    std::vector<std::vector<double>> extreme_hist_;  // [m] historical F vectors

    // ── Das-Dennis init ────────────────────────────────────────────────────
    void init_V(int n, int m) {
        V_ = das_dennis::generate_exact(m, n);
        // SRVN-1: theta_c^min (Eq.16) is computed HERE, the path common to
        // setup and setup_seeded. Computing it only in setup left the seeded
        // path with theta_c(G=1) = 0, hence 0/0 = NaN in rho(x). theta_min()
        // uses true angles.
        theta_min_c_ = SRVStrategy<Ind_t>::theta_min(V_);
    }


    // ── Dominance ──────────────────────────────────────────────────────────
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca=vault.get_cv(a), cb=vault.get_cv(b);
            bool af=(ca<=0.0), bf=(cb<=0.0);
            if(af&&!bf) return true;
            if(!af&&bf) return false;
            if(!af&&!bf) return ca<cb;
        }
        const auto& fa=vault.objectives_of(a); const auto& fb=vault.objectives_of(b);
        bool better=false;
        for (std::size_t i=0;i<fa.size();++i) {
            if (fa[i]>fb[i]) return false;
            if (fa[i]<fb[i]) better=true;
        }
        return better;
    }

    // ── Fast NDS ───────────────────────────────────────────────────────────
    std::vector<std::vector<int>> fast_nds(DataVault<Ind_t>& vault, int n) {
        std::vector<std::vector<int>> S(n); std::vector<int> np(n,0);
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
            if (i==j) continue;
            if (dominates(vault,i,j)) S[i].push_back(j);
            else if (dominates(vault,j,i)) ++np[i];
        }
        std::vector<std::vector<int>> fronts; std::vector<int> f0;
        for (int i=0;i<n;++i) if(np[i]==0) { vault.get_ind(i).rank=0; f0.push_back(i); }
        fronts.push_back(f0);
        int k=0;
        while (!fronts[k].empty()) {
            std::vector<int> nxt;
            for (int i:fronts[k]) for (int j:S[i])
                if (--np[j]==0) { vault.get_ind(j).rank=k+1; nxt.push_back(j); }
            fronts.push_back(nxt); ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── Update of the cumulative normalization state (deb2014 §IV-C) ───────
    // z^min is the minimum over the union of all S_tau; the extremes are the
    // min-ASF members of {previously found} u S_t ("ever found from the start
    // of the simulation"). The scheme is identical to nsga3.hpp.
    void update_norm_state(DataVault<Ind_t>& vault, const std::vector<int>& St) {
        int m=vault.objs_n();
        if (zmin_hist_.empty())    zmin_hist_.assign(m, std::numeric_limits<double>::max());
        if (extreme_hist_.empty()) extreme_hist_.assign(m, {});
        for (int v:St) { const auto& o=vault.objectives_of(v); for (int j=0;j<m;++j) zmin_hist_[j]=std::min(zmin_hist_[j],o[j]); }
        const double eps=1e-6;
        auto asf=[&](const std::vector<double>& o, int axis) {
            double val=-std::numeric_limits<double>::max();
            for (int j=0;j<m;++j) { double w=(axis==j)?1.0:eps; val=std::max(val,(o[j]-zmin_hist_[j])/w); }
            return val;
        };
        for (int i=0;i<m;++i) {
            std::vector<double> best=extreme_hist_[i];
            double best_asf=best.empty()?std::numeric_limits<double>::max():asf(best,i);
            for (int v:St) { const auto& o=vault.objectives_of(v); double a=asf(o,i); if (a<best_asf) { best_asf=a; best=o; } }
            extreme_hist_[i]=std::move(best);
        }
    }

    // ── NSGA-III normalization (ASF hyperplane, cumulative z^min/extremes) ─
    struct NP { std::vector<double> zmin, intercepts; };
    NP compute_norm(DataVault<Ind_t>& vault, const std::vector<int>& St) {
        int m=vault.objs_n(); NP np;
        update_norm_state(vault, St);   // cumulative (deb2014 §IV-C)
        np.zmin = zmin_hist_;
        np.intercepts.assign(m,1.0);
        // z^nad = the max over Pc — the paper's fallback: "for some
        // degenerate cases like no intercepts in certain directions or
        // negative intercepts, z_i^nadir is set as the maximal value of
        // f_i(x) from all x in Pc».
        std::vector<double> nadir(m,-std::numeric_limits<double>::max());
        for (int v:St) { const auto& o=vault.objectives_of(v); for (int j=0;j<m;++j) nadir[j]=std::max(nadir[j],o[j]); }
        bool deg=false; for (int i=0;i<m&&deg==false;++i) if(extreme_hist_[i].empty()) deg=true;
        if (deg==false) {
            std::vector<std::vector<double>> A(m,std::vector<double>(m+1));
            for (int i=0;i<m;++i) {
                const auto& o=extreme_hist_[i];
                for (int j=0;j<m;++j) A[i][j]=o[j]-np.zmin[j];
                A[i][m]=1.0;
            }
            for (int col=0;col<m&&deg==false;++col) {
                int piv=col; for (int r=col+1;r<m;++r) if(std::abs(A[r][col])>std::abs(A[piv][col])) piv=r;
                std::swap(A[col],A[piv]);
                if (std::abs(A[col][col])<1e-12) { deg=true; break; }
                for (int r=col+1;r<m;++r) { double f=A[r][col]/A[col][col]; for (int k=col;k<=m;++k) A[r][k]-=f*A[col][k]; }
            }
            if (deg==false) {
                std::vector<double> x(m);
                for (int i=m-1;i>=0;--i) { x[i]=A[i][m]; for (int j=i+1;j<m;++j) x[i]-=A[i][j]*x[j]; x[i]/=A[i][i]; }
                // The intercept is a_i = 1/x[i]. Previously x[i] < 0 produced
                // a NEGATIVE denominator, i.e. a degenerate normalization,
                // while x[i] ~ 0 ("no intercept") produced a magic 1.0. There
                // is now a per-objective fallback to nadir_i − zmin_i.
                for (int i=0;i<m;++i) {
                    if (x[i] > 1e-12) np.intercepts[i] = 1.0 / x[i];
                    else              np.intercepts[i] = nadir[i] - np.zmin[i];
                }
            }
        }
        if (deg) {
            for (int j=0;j<m;++j) np.intercepts[j]=nadir[j]-np.zmin[j];
        }
        return np;
    }

    std::vector<double> fn(DataVault<Ind_t>& vault, int v, const NP& np) const {
        int m=vault.objs_n(); const auto& o=vault.objectives_of(v);
        std::vector<double> f(m);
        for (int j=0;j<m;++j) { double d=np.intercepts[j]; f[j]=(std::abs(d)>1e-12)?(o[j]-np.zmin[j])/d:0.0; f[j]=std::max(f[j],0.0); }
        return f;
    }

    // ── Association (perp distance to reference lines) ─────────────────────
    void associate(DataVault<Ind_t>& vault, const std::vector<int>& St,
                   const std::vector<std::vector<double>>& fn_map) {
        int m=vault.objs_n(); int nref=static_cast<int>(V_.size());
        int sz=static_cast<int>(St.size());
        for (int si=0;si<sz;++si) {
            double best_d=std::numeric_limits<double>::max(); int best_r=0;
            for (int r=0;r<nref;++r) {
                const auto& rv=V_[r];
                double rr=0.0; for (int j=0;j<m;++j) rr+=rv[j]*rv[j];
                if (rr<1e-14) continue;
                double dot=0.0; for (int j=0;j<m;++j) dot+=fn_map[si][j]*rv[j];
                double t=dot/rr; double d2=0.0;
                for (int j=0;j<m;++j) { double diff=fn_map[si][j]-t*rv[j]; d2+=diff*diff; }
                double d=std::sqrt(d2);
                if (d<best_d) { best_d=d; best_r=r; }
            }
            vault.get_ind(St[si]).ref_point_idx=best_r;
            vault.get_ind(St[si]).norm_distance=best_d;
        }
    }

    // ── NSGA-III niching ───────────────────────────────────────────────────
    void niching(DataVault<Ind_t>& vault, int K,
                 const std::vector<int>& Fl,
                 const std::vector<int>& St_minus_Fl,
                 std::vector<int>& chosen) {
        int nref=static_cast<int>(V_.size());
        std::vector<int> rho(nref,0);
        for (int v:St_minus_Fl) ++rho[vault.get_ind(v).ref_point_idx];
        std::vector<int> fl_rem=Fl;
        for (int pick=0;pick<K;++pick) {
            if (fl_rem.empty()) break;
            std::vector<int> rwf;
            for (int r=0;r<nref;++r) {
                bool has=false; for (int v:fl_rem) if(vault.get_ind(v).ref_point_idx==r) { has=true; break; }
                if (has) rwf.push_back(r);
            }
            if (rwf.empty()) break;
            int min_rho=std::numeric_limits<int>::max(); for (int r:rwf) min_rho=std::min(min_rho,rho[r]);
            std::vector<int> cands; for (int r:rwf) if(rho[r]==min_rho) cands.push_back(r);
            std::uniform_int_distribution<int> dc(0,(int)cands.size()-1);
            int j_star=cands[dc(rng_)];
            std::vector<int> assoc; for (int v:fl_rem) if(vault.get_ind(v).ref_point_idx==j_star) assoc.push_back(v);
            int sel;
            if (rho[j_star]==0) sel=*std::min_element(assoc.begin(),assoc.end(),[&](int a,int b){ return vault.get_ind(a).norm_distance<vault.get_ind(b).norm_distance; });
            else { std::uniform_int_distribution<int> da(0,(int)assoc.size()-1); sel=assoc[da(rng_)]; }
            chosen.push_back(sel); ++rho[j_star];
            fl_rem.erase(std::find(fl_rem.begin(),fl_rem.end(),sel));
        }
    }

    // ── Tournament (random when both feasible — NSGA-III style) ───────────
    int tournament(DataVault<Ind_t>& vault, std::uniform_int_distribution<int>& dist) {
        int a=dist(rng_), b=dist(rng_);
        if (constraint_mode==ConstraintMode::FEASIBILITY) {
            double ca=vault.get_cv(a), cb=vault.get_cv(b);
            bool af=(ca<=0.0), bf=(cb<=0.0);
            if(af&&!bf) return a;
            if(!af&&bf) return b;
            if(!af&&!bf) return (ca<cb)?a:b;
        }
        std::uniform_int_distribution<int> coin(0,1); return coin(rng_)?a:b;
    }

    void rearrange(DataVault<Ind_t>& vault, const std::vector<int>& sv, int pool) {
        int n=static_cast<int>(sv.size());
        std::vector<int> pos(pool),at_pos(pool);
        std::iota(pos.begin(),pos.end(),0); std::iota(at_pos.begin(),at_pos.end(),0);
        for (int i=0;i<n;++i) {
            int want=sv[i],cur=pos[want]; if(cur==i) continue;
            int other=at_pos[i]; vault.swap_active(i,cur);
            pos[want]=i; pos[other]=cur; at_pos[i]=want; at_pos[cur]=other;
        }
        vault.reduce(n);
    }

public:
    SRVNSGA3Core() = default;

    void set_eta_crossover(double e) { eta_c_=e; }
    void set_eta_mutation (double e) { eta_m_=e; }
    void set_pc           (double p) { pc_   =p; }
    void set_t_max        (int t)    { t_max_=t; }
    void set_seed         (unsigned s){ rng_.seed(s); }
    // SRVStrategy has no kmeans_iter field: the iteration count of the
    // modified k-means is fixed at 2m by Algorithm 3 of Liu et al. 2022.
    // An earlier version referenced a non-existent srv_.kmeans_iter, which
    // failed to compile when called. This is now a safe no-op that keeps the
    // API surface.
    void set_kmeans_iter  (int /*k*/) { /* iterations fixed at 2m (paper) */ }

    void setup(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(), m=vault.objs_n();
        current_gen_=0;
        // The cumulative normalization history is counted "from the start of
        // the simulation" (deb2014 §IV-C).
        zmin_hist_.clear(); extreme_hist_.clear();
        init_V(n,m);   // also computes theta_min_c_ (Eq.16)
        const auto& bounds=vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int> db(0,1);
        std::vector<double> vars(vault.vars_n()); std::vector<int> bvars(vault.bin_vars_n());
        for (int i=0;i<n;++i) {
            for (int j=0;j<vault.vars_n();++j) {
                double lo=bounds[j].first.value_or(0.0), hi=bounds[j].second.value_or(1.0);
                vars[j]=lo+dr(rng_)*(hi-lo);
            }
            for (int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if (vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars);
            else vault.set_variables(i,vars);
        }
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(), m=vault.objs_n();
        current_gen_=0;
        // reset the cumulative history (see setup).
        zmin_hist_.clear(); extreme_hist_.clear();
        // SRVN-1, as in srv.hpp: init_V also computes theta_min_c_. The
        // seeded path used to leave theta_min = 0, giving NaN in rho(x) in the
        // first generation. V_ is overwritten by the SRVs every step, so when
        // V_ carries over from a previous run, theta_min is recomputed from a
        // fresh Das-Dennis lattice (Eq.16 is defined on the ORIGINAL RVs).
        if (V_.empty() || theta_min_c_ <= 0.0) init_V(n,m);
    }

    void step(DataVault<Ind_t>& vault) {
        int n=vault.pop_size(); const auto& bounds=vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0,n-1);

        // Liu et al. 2022, §III-C: the SRVs are refreshed EVERY generation
        // from St (below, after the non-dominated sort), not once per 10% of
        // t_max and not from the parents [0,n).

        vault.expand(vault.pop_size());

        std::vector<double> pv1(vault.vars_n()),pv2(vault.vars_n()),c1,c2;
        double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;  // 1/n
        for (int i=0;i<n;i+=2) {
            int p1=tournament(vault,dist_int), p2=tournament(vault,dist_int);
            for (int j=0;j<vault.vars_n();++j) {
                pv1[j]=vault.get_variable(p1,j); pv2[j]=vault.get_variable(p2,j);
            }
            ops::sbx(pv1,pv2,c1,c2,bounds,eta_c_,pc_,rng_);
            ops::polynomial_mutation(c1,bounds,eta_m_,pm,rng_);
            ops::polynomial_mutation(c2,bounds,eta_m_,pm,rng_);
            if (vault.bin_vars_n()>0) {
                std::vector<int> bv1(vault.bin_vars_n()),bv2(vault.bin_vars_n()),bc1,bc2;
                for(int j=0;j<vault.bin_vars_n();++j) {
                    bv1[j]=vault.get_bin_variable(p1,j); bv2[j]=vault.get_bin_variable(p2,j);
                }
                ops::binary_crossover(bv1,bv2,bc1,bc2,rng_);
                ops::bit_flip_mutation(bc1,vault.bin_vars_n(),rng_);
                if(i+1<n) ops::bit_flip_mutation(bc2,vault.bin_vars_n(),rng_);
                vault.set_all_variables(n+i,c1,bc1);
                if(i+1<n) vault.set_all_variables(n+i+1,c2,bc2);
            } else {
                vault.set_variables(n+i,c1);
                if(i+1<n) vault.set_variables(n+i+1,c2);
            }
        }
        vault.sync();

        auto fronts=fast_nds(vault,n*2);

        std::vector<int> St; std::vector<int> Fl;
        for (auto& front:fronts) {
            if (static_cast<int>(St.size()+front.size())<=n) {
                for(int v:front) St.push_back(v);
                if((int)St.size()==n) break;
            } else { Fl=front; break; }
        }

        // Pc = St (Liu 2022 §III-C): the leading fronts of the merged pool
        // (>= N) = St u Fl.
        std::vector<int> St_full = St;
        for (int v : Fl) St_full.push_back(v);

        // SRVs: refreshed EVERY generation, sourced from St, not the parents.
        {
            double tc = SRVStrategy<Ind_t>::compute_theta_c(
                            theta_min_c_, current_gen_+1, t_max_);
            V_ = srv_.extract(vault, St_full, n, tc);   // N = n clusters/RVs
        }

        std::vector<int> survivors;
        if (Fl.empty()) {
            survivors=St;
        } else {
            std::vector<int> St_minus_Fl=St;
            int K=n-(int)St.size();

            auto np=compute_norm(vault,St_full);
            int sfsz=static_cast<int>(St_full.size());
            std::vector<std::vector<double>> fn_map(sfsz);
            for (int si=0;si<sfsz;++si) fn_map[si]=fn(vault,St_full[si],np);

            associate(vault,St_full,fn_map);

            std::vector<int> chosen;
            niching(vault,K,Fl,St_minus_Fl,chosen);
            survivors=St_minus_Fl;
            for(int v:chosen) survivors.push_back(v);
        }
        rearrange(vault,survivors,n*2);
        ++current_gen_;
    }
};

} // namespace mootation

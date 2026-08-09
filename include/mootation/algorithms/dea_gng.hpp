#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DEA-GNG — Decomposition-based EA adapting Reference Vectors and Scalarizing
// Functions by Growing Neural Gas.
// Y. Liu, H. Ishibuchi, N. Masuyama, Y. Nojima — IEEE Transactions on
//   Evolutionary Computation 24(3):439-453, 2020.
// doi:10.1109/TEVC.2019.2926151
//
// IDEA (references = GNG nodes over the individuals + adaptive PBI). A Growing
// Neural Gas network is trained on the archive of non-dominated signals A_S and
// learns the topology of the PF. The GNG nodes (expanded over the range of the
// signals and combined with the uniform R_u) serve as the reference vectors;
// the θ in the PBI of each node is chosen from the curvature (the angles of the
// GNG edges). Selection is NSGA-III-style with these references and a
// per-reference PBI.
//
// Generational scheme (Algorithm 1):
//   init; z*; R=R_u=Uniform(N); A_S=∅.
//   while g<Gmax: P'=Mating(P); P''=Reproduction(P'); z*=ideal(P''∪z*);
//     if g<(1−α)Gmax: A_S=ArchiveUpdate(A_S∪P'',N_S); [R_node,V_edge]=GNG(A_S);
//       R=RefAdapt(R_u,R_node,A_S); F^S=ScalarAdapt(R,V_edge);
//     P=EnvSelection(P∪P'',R,F^S,N,z*).
// GNG variant (§III-D): nodes carry HP; a new node (Step 0/8) starts at
//   HP_max. Per signal the winner r_a is restored to HP_max, the runner-up r_b
//   is left UNCHANGED, and every other live node loses one HP. The runner-up
//   exemption is load-bearing, not an oversight: with only r_a and r_b alive
//   nobody is penalised, so the network cannot die out. Dead (HP=0) nodes are
//   removed; isolated nodes are NOT removed for being isolated. Edges are
//   undirected with a single age per edge (Step 0/3/6/7). age_max, λ, ε_a,
//   ε_nb, α, δ are standard GNG.
// RefAdapt (Alg.3-4): expansion (stretch the sub-networks over the range of
//   A_S) + combination (R = R'_u∪R'_node, remove the R_u vectors closer than
//   d_min=min(d_p,d_u) to R_p; d_p is the mean distance over the pairs of R_p
//   nodes CONNECTED by an edge).
// ScalarAdapt (Eq.7-8): θ=max(0,1/tan(ζ')), ζ'=max(0,ζ_min−ε), ζ_k=angle(r, the
//   edge r_nb−r); an obtuse ζ' (≥π/2) ⇒ θ=0; an isolated node and R'_u → θ=∞
//   (d2 only). PBI f^S=d1+θ·d2.
// Mating (§III-B): binary tournament — primary is the non-dominated sorting
//   rank, secondary the c_j of the individual's reference vector (association
//   as in Alg.2); a tie is broken at random.
//
// PAPER DEFAULTS (§IV): N=pop_size; N_S=M·N; ε=0.05π (M<5) / 0.15π (M≥5) —
//   §IV-A, setter set_eps_theta; HP_max=2|A_S|, age_max=N, λ=⌊0.2N⌋, ε_a=0.2,
//   ε_nb=0.01, α_err=0.5, δ=0.9; max nodes=N; α (GNG stop)=0.1; SBX
//   η_c=20/pc=1; PM η_m=20/pm=1/n.
//
// DECLARED DEVIATIONS:
//   GNG-1 (MINOR). The A_S update (III-C) is approximated: front-0, and when
//     >N_S an FPS diversity selection in objective space (without the exact
//     niching by c_J of the paper).
//   GNG-2 (MINOR). sub-networks = the connected components of the edges (BFS).
//   GNG-3 (MINOR). NSGA-III selection: normalization by z*/z' (max over F_1);
//     association by angle; niches c_j; PBI f^S_J for the critical front.
//   GNG-4 (MINOR). θ=∞ is emulated by a large value (1e6); R'_u → d2.
//   GNG-5 (MINOR). real-valued genome; binary is out of coverage (NONE).
//   GNG-θ. ζ_k is the angle between r and the EDGE (r_nb − r) per Eq.8 — not
//     the angle between two node positions, which is what an earlier version
//     computed.
//   GNG-6 (AMBIGUOUS — the paper mixes two frames; the raw one is chosen).
//     ζ (Eq.8) is measured on the RAW, un-expanded GNG node geometry: both the
//     reference vector and the edge come from node_obj, matching V_edge as
//     emitted by Alg.1 line 12, and the resulting θ is then attached to the
//     Eq.5-EXPANDED vector. The paper does not fix the space — Alg.1 line 14
//     hands Scalarizing_Function_Adaptation the EXPANDED R together with the
//     RAW V_edge, and expansion is a per-component affine map, so an angle is
//     not invariant under it. The pre-expansion reading is adopted because it
//     is the only one in which both operands live in the same frame. The same
//     mixing affects the raw-vs-normalized scale of R'_node.
//
// EXTENSIONS BEYOND THE PAPER: constraint_mode exists for API uniformity and
//   does not change the DEA-GNG logic (NONE).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "../detail/math_compat.hpp"
#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class DEAGNGCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double alpha_stop_=0.1, eps_theta_=0.05*M_PI;
    // ε for ζ' depends on the number of objectives (§IV-A: 0.05π for M<5,
    //   0.15π for M≥5); eps_user_set_ is the flag for an explicit setting by
    //   the user (the setter takes priority).
    bool   eps_user_set_=false;
    double ea_=0.2, enb_=0.01, aerr_=0.5, ddecay_=0.9;
    double eta_c_=20.0, eta_m_=20.0, pc_=1.0, pm_=-1.0;
    int    t_max_=1000;
    std::mt19937 rng_{std::random_device{}()};
    static constexpr double INF_THETA=1e6;

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };
    struct GNode { std::vector<double> pos; double err=0; int hp=0; bool alive=true; };
    std::vector<GNode> gn_;
    std::vector<std::map<int,int>> gadj_;   // node→(node→age)
    int sig_count_=0, age_max_=100, lambda_=20, maxnode_=100;

    std::vector<Sol> pop_, AS_;
    std::vector<std::vector<double>> Ru_;
    std::vector<std::vector<double>> R_;
    std::vector<double> Rtheta_;
    std::vector<double> z_;
    int N_=0, m_=0, NS_=0, g_=0;

    double pm_eff(int nv) const { return (pm_>0.0)?pm_:(nv>0?1.0/nv:0.0); }
    double uni01(){ return std::uniform_real_distribution<double>(0,1)(rng_); }
    bool dom(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv); }
    static double edist(const std::vector<double>& a, const std::vector<double>& b){ double s=0; for(std::size_t k=0;k<a.size();++k){double d=a[k]-b[k];s+=d*d;} return std::sqrt(s); }
    static double ang(const std::vector<double>& a, const std::vector<double>& b){
        double d=0,na=0,nb=0; for(std::size_t k=0;k<a.size();++k){d+=a[k]*b[k];na+=a[k]*a[k];nb+=b[k]*b[k];}
        double q=std::sqrt(na)*std::sqrt(nb); if(q<1e-300) return 0.0; return std::acos(std::clamp(d/q,-1.0,1.0));
    }
    void upd_ideal(const std::vector<double>& f){ for(int k=0;k<m_;++k) z_[k]=std::min(z_[k],f[k]); }

    std::vector<int> nondominated(const std::vector<Sol>& P) const {
        std::vector<int> nd; int n=(int)P.size();
        for(int p=0;p<n;++p){ bool ok=true; for(int q=0;q<n;++q){ if(p!=q && dom(P[q],P[p])){ok=false;break;} } if(ok) nd.push_back(p); }
        return nd;
    }

    void archive_update(const std::vector<Sol>& Ppp){
        std::vector<Sol> U=AS_; for(auto&s:Ppp) U.push_back(s);
        auto nd=nondominated(U);
        std::vector<Sol> F0; for(int i:nd) F0.push_back(U[i]);
        if((int)F0.size()<=NS_){ AS_=F0; return; }
        std::vector<std::vector<double>> pts; for(auto&s:F0) pts.push_back(s.objs);
        std::vector<int> sel; std::vector<char> used(F0.size(),0);
        int first=std::uniform_int_distribution<int>(0,(int)F0.size()-1)(rng_); sel.push_back(first); used[first]=1;
        while((int)sel.size()<NS_){ int b=-1; double bd=-1; for(int i=0;i<(int)F0.size();++i){ if(used[i])continue; double mn=1e300; for(int s:sel) mn=std::min(mn,edist(pts[i],pts[s])); if(mn>bd){bd=mn;b=i;} } if(b<0)break; sel.push_back(b); used[b]=1; }
        AS_.clear(); for(int i:sel) AS_.push_back(F0[i]);
    }

    void gng_update(){
        if(AS_.empty()) return;
        std::vector<double> mn(m_,1e300),mx(m_,-1e300);
        for(auto&s:AS_) for(int k=0;k<m_;++k){ mn[k]=std::min(mn[k],s.objs[k]); mx[k]=std::max(mx[k],s.objs[k]); }
        auto norm=[&](const std::vector<double>& o){ std::vector<double> r(m_); for(int k=0;k<m_;++k){double d=mx[k]-mn[k]; r[k]=(d>1e-12)?(o[k]-mn[k])/d:0.0;} return r; };
        int HPmax=2*(int)AS_.size();
        if((int)gn_.size()<2){
            gn_.clear(); gadj_.clear();
            for(int t=0;t<2;++t){ GNode nd; nd.pos=norm(AS_[std::uniform_int_distribution<int>(0,(int)AS_.size()-1)(rng_)].objs); nd.hp=HPmax; gn_.push_back(nd); gadj_.push_back({}); }
            gadj_[0][1]=0; gadj_[1][0]=0;
        }
        std::vector<int> order(AS_.size()); std::iota(order.begin(),order.end(),0); std::shuffle(order.begin(),order.end(),rng_);
        for(int si:order){
            std::vector<double> xi=norm(AS_[si].objs);
            ++sig_count_;
            int a=-1,b=-1; double da=1e300,db=1e300;
            for(int i=0;i<(int)gn_.size();++i){ if(!gn_[i].alive) continue; double d=edist(gn_[i].pos,xi); if(d<da){db=da;b=a;da=d;a=i;} else if(d<db){db=d;b=i;} }
            if(a<0) continue;
            for(int i=0;i<(int)gn_.size();++i){ if(!gn_[i].alive) continue; if(i==a) gn_[i].hp=HPmax; else if(i==b){} else --gn_[i].hp; }
            // Step 3 — symmetric edge ageing. Notable fix: a GNG edge is
            //   undirected with one single age; the age increment of ALL edges
            //   from the winner r_a is mirrored into the reverse copy
            //   gadj_[nb][a], so that both directed entries always carry one
            //   and the same age.
            for(auto& e:gadj_[a]){ ++e.second; auto it=gadj_[e.first].find(a); if(it!=gadj_[e.first].end()) it->second=e.second; }
            gn_[a].err += da*da;
            for(int k=0;k<m_;++k) gn_[a].pos[k]+=ea_*(xi[k]-gn_[a].pos[k]);
            for(auto& e:gadj_[a]){ int nb=e.first; if(!gn_[nb].alive) continue; for(int k=0;k<m_;++k) gn_[nb].pos[k]+=enb_*(xi[k]-gn_[nb].pos[k]); }
            if(b>=0){ gadj_[a][b]=0; gadj_[b][a]=0; }
            // Step 7 — symmetric edge removal. Notable fix: the age of an edge
            //   is one and the same for both sides: at age>age_max we erase
            //   BOTH directed copies (gadj_[i][j] and gadj_[j][i]), otherwise a
            //   "half-alive" reverse entry would be left behind, which is not
            //   representable in the model of the paper.
            for(int i=0;i<(int)gn_.size();++i){ std::vector<int> del; for(auto&e:gadj_[i]) if(e.second>age_max_) del.push_back(e.first); for(int j:del){ gadj_[i].erase(j); gadj_[j].erase(i); } }
            for(int i=0;i<(int)gn_.size();++i){ if(gn_[i].alive && gn_[i].hp<=0){ gn_[i].alive=false; for(auto&e:gadj_[i]) gadj_[e.first].erase(i); gadj_[i].clear(); } }
            int aliveN=0; for(auto&g:gn_) if(g.alive) ++aliveN;
            if(sig_count_%std::max(1,lambda_)==0 && aliveN<maxnode_){
                int q=-1; double me=-1; for(int i=0;i<(int)gn_.size();++i){ if(gn_[i].alive && gn_[i].err>me){me=gn_[i].err;q=i;} }
                if(q>=0 && !gadj_[q].empty()){
                    int f=-1; double mf=-1; for(auto&e:gadj_[q]){ int nb=e.first; if(gn_[nb].alive && gn_[nb].err>mf){mf=gn_[nb].err;f=nb;} }
                    if(f>=0){
                        GNode nn; nn.pos.resize(m_); for(int k=0;k<m_;++k) nn.pos[k]=0.5*(gn_[q].pos[k]+gn_[f].pos[k]); nn.hp=HPmax;
                        gn_[q].err*=aerr_; gn_[f].err*=aerr_; nn.err=gn_[q].err;
                        int idx=(int)gn_.size(); gn_.push_back(nn); gadj_.push_back({});
                        gadj_[q].erase(f); gadj_[f].erase(q);
                        gadj_[q][idx]=0; gadj_[idx][q]=0; gadj_[f][idx]=0; gadj_[idx][f]=0;
                    }
                }
            }
            for(auto&g:gn_) if(g.alive) g.err*=ddecay_;
        }
    }

    void reference_adaptation(){
        std::vector<int> idx; for(int i=0;i<(int)gn_.size();++i) if(gn_[i].alive) idx.push_back(i);
        if(idx.empty()){ R_=Ru_; Rtheta_.assign(R_.size(),INF_THETA); return; }
        std::vector<double> mn(m_,1e300),mx(m_,-1e300);
        for(auto&s:AS_) for(int k=0;k<m_;++k){ mn[k]=std::min(mn[k],s.objs[k]); mx[k]=std::max(mx[k],s.objs[k]); }
        std::map<int,int> g2l; for(int t=0;t<(int)idx.size();++t) g2l[idx[t]]=t;
        std::vector<std::vector<double>> node_obj(idx.size());
        for(int t=0;t<(int)idx.size();++t){ node_obj[t].resize(m_); for(int k=0;k<m_;++k) node_obj[t][k]=gn_[idx[t]].pos[k]*(mx[k]-mn[k])+mn[k]; }
        std::vector<int> comp(idx.size(),-1); int Q=0;
        for(int t=0;t<(int)idx.size();++t){ if(comp[t]>=0) continue; std::queue<int> bfs; bfs.push(t); comp[t]=Q;
            while(!bfs.empty()){ int u=bfs.front();bfs.pop(); for(auto&e:gadj_[idx[u]]){ if(g2l.count(e.first)){ int v=g2l[e.first]; if(comp[v]<0){comp[v]=Q;bfs.push(v);} } } } ++Q; }
        std::vector<std::vector<int>> sigOf(Q);
        for(int s=0;s<(int)AS_.size();++s){
            int bq=0; double bd=1e300; for(int t=0;t<(int)idx.size();++t){ double d=edist(AS_[s].objs,node_obj[t]); if(d<bd){bd=d;bq=comp[t];} } sigOf[bq].push_back(s); }
        std::vector<std::vector<double>> Rnode;
        // rp2t[i] — the local index of the GNG node (in idx/node_obj) that
        //   produced the i-th element of Rnode/Rp. Needed for d_p over the
        //   pairs CONNECTED by edges (Alg.4 l.6).
        std::vector<int> rp2t;
        for(int q=0;q<Q;++q){
            std::vector<int> nq; for(int t=0;t<(int)idx.size();++t) if(comp[t]==q) nq.push_back(t);
            std::vector<double> fminN(m_,1e300),fmaxN(m_,-1e300);
            for(int t:nq) for(int k=0;k<m_;++k){ fminN[k]=std::min(fminN[k],node_obj[t][k]); fmaxN[k]=std::max(fmaxN[k],node_obj[t][k]); }
            std::vector<double> fminA(m_,1e300),fmaxA(m_,-1e300);
            if(!sigOf[q].empty()) for(int s:sigOf[q]) for(int k=0;k<m_;++k){ fminA[k]=std::min(fminA[k],AS_[s].objs[k]); fmaxA[k]=std::max(fmaxA[k],AS_[s].objs[k]); }
            else { fminA=fminN; fmaxA=fmaxN; }
            for(int t:nq){ std::vector<double> r(m_);
                for(int k=0;k<m_;++k){ double dn=fmaxN[k]-fminN[k]; double frac=(dn>1e-12)?(node_obj[t][k]-fminN[k])/dn:0.0; r[k]=frac*(fmaxA[k]-fminA[k])+fminA[k]; }
                Rnode.push_back(r); rp2t.push_back(t); }
        }
        std::vector<std::vector<double>> Rp;
        for(auto& rn:Rnode){ double s=0; for(double v:rn) s+=v; if(s<1e-300) s=1; std::vector<double> rp(m_); for(int k=0;k<m_;++k) rp[k]=rn[k]/s; Rp.push_back(rp); }
        // d_p — the mean Euclidean distance between EVERY pair of nodes in R_p
        //   CONNECTED by an edge (Alg.4, line 6), and not the mean NN distance.
        //   Connectivity is taken from the GNG topology (gadj_); each edge is
        //   counted once.
        std::vector<int> t2rp(idx.size(),-1);
        for(int i=0;i<(int)rp2t.size();++i) t2rp[rp2t[i]]=i;
        double dp=0; int cntp=0;
        for(int i=0;i<(int)Rp.size();++i){ int t=rp2t[i];
            for(auto&e:gadj_[idx[t]]){ if(!g2l.count(e.first)) continue; int v=g2l[e.first]; int j=t2rp[v];
                if(j>i){ dp+=edist(Rp[i],Rp[j]); ++cntp; } } }
        if(cntp>0) dp/=cntp; else dp=1e300;   // no edges (all nodes isolated) ⇒ d_p does not constrain
        double du=1e300; for(int i=0;i<(int)Ru_.size();++i) for(int j=i+1;j<(int)Ru_.size();++j){ double d=edist(Ru_[i],Ru_[j]); if(d<du) du=d; }
        double dmin=std::min(dp,du);
        R_.clear(); Rtheta_.clear();
        for(auto& ru:Ru_){ double dd=1e300; for(auto& rp:Rp) dd=std::min(dd,edist(ru,rp)); if(dd>=dmin){ R_.push_back(ru); Rtheta_.push_back(INF_THETA); } }
        int ri=0;
        for(int q=0;q<Q;++q){ std::vector<int> nq; for(int t=0;t<(int)idx.size();++t) if(comp[t]==q) nq.push_back(t);
            for(int t:nq){
                double zmin=1e300; int deg=0;
                for(auto&e:gadj_[idx[t]]){ if(g2l.count(e.first)){ int v=g2l[e.first];
                    // Eq.8: ζ_k = the angle between the reference vector r and the EDGE v_edge=r_nb−r
                    std::vector<double> edge(m_); for(int k=0;k<m_;++k) edge[k]=node_obj[v][k]-node_obj[t][k];
                    // Do NOT fold an obtuse angle down to an acute one.
                    //   Eq.7: θ=max(0,1/tan(ζ')). For an obtuse ζ' (≥π/2) tan<0 ⇒ θ=0 (pure
                    //   convergence). The former folding a=min(a,π−a) inverted this into a large θ.
                    double a=ang(node_obj[t],edge);   // the true angle r↔edge in [0,π]
                    if(a<zmin) zmin=a; ++deg; } }
                double theta;
                if(deg==0) theta=INF_THETA;
                // Eq.7 exactly. ζ'=max(0,ζ_min−ε).
                //   ζ'≈0 (edge ∥ r) ⇒ tan→0 ⇒ θ→∞; ζ'∈(0,π/2) ⇒ θ=1/tanζ'>0;
                //   ζ'≥π/2 (obtuse angle) ⇒ 1/tanζ'≤0 ⇒ max(0,·)=0.
                else { double zp=std::max(0.0,zmin-eps_theta_); theta = (zp>1e-9)? std::max(0.0,1.0/std::tan(zp)) : INF_THETA; }
                R_.push_back(Rnode[ri]); Rtheta_.push_back(theta);
                ++ri;
            }
        }
        if(R_.empty()){ R_=Ru_; Rtheta_.assign(R_.size(),INF_THETA); }
    }

    double pbi(const std::vector<double>& fp, const std::vector<double>& r, double theta) const {
        double rn2=0; for(double v:r) rn2+=v*v; double rn=std::sqrt(std::max(rn2,1e-300));
        double d1=0; for(int k=0;k<m_;++k) d1+=fp[k]*r[k]/rn;
        double s2=0; for(int k=0;k<m_;++k){ double pr=fp[k]-d1*r[k]/rn; s2+=pr*pr; }
        double d2=std::sqrt(std::max(s2,0.0));
        if(theta>=INF_THETA) return d2;
        return d1+theta*d2;
    }

    std::vector<Sol> env_select(const std::vector<Sol>& U){
        int n=(int)U.size();
        std::vector<int> dc(n,0); std::vector<std::vector<int>> dl(n); std::vector<std::vector<int>> fr; std::vector<int> f0;
        for(int p=0;p<n;++p){ for(int q=0;q<n;++q){ if(p==q)continue; if(dom(U[p],U[q])) dl[p].push_back(q); else if(dom(U[q],U[p])) ++dc[p]; } if(dc[p]==0) f0.push_back(p); }
        fr.push_back(f0);
        while(!fr.back().empty()){ std::vector<int> nx; for(int p:fr.back()) for(int q:dl[p]) if(--dc[q]==0) nx.push_back(q); if(nx.empty()) break; fr.push_back(std::move(nx)); }
        std::vector<int> Pidx; int fi=0;
        for(; fi<(int)fr.size(); ++fi){ if((int)(Pidx.size()+fr[fi].size())>N_) break; for(int p:fr[fi]) Pidx.push_back(p); if((int)Pidx.size()==N_) break; }
        if((int)Pidx.size()>=N_){ std::vector<Sol> out; for(int t=0;t<N_;++t) out.push_back(U[Pidx[t]]); return out; }
        std::vector<int> crit = (fi<(int)fr.size())? fr[fi] : std::vector<int>{};
        std::vector<double> zp(m_,-1e300);
        for(int p:fr[0]) for(int k=0;k<m_;++k) zp[k]=std::max(zp[k],U[p].objs[k]);
        auto fnorm=[&](const std::vector<double>& o){ std::vector<double> r(m_); for(int k=0;k<m_;++k){double d=zp[k]-z_[k]; r[k]=(d>1e-12)?(o[k]-z_[k])/d:(o[k]-z_[k]); } return r; };
        int R=(int)R_.size();
        std::vector<int> cj(R,0);
        for(int p:Pidx){ auto fp=fnorm(U[p].objs); int best=0; double ba=ang(fp,R_[0]); for(int r=1;r<R;++r){double a=ang(fp,R_[r]); if(a<ba){ba=a;best=r;}} ++cj[best]; }
        std::vector<std::vector<int>> Delta(R);
        for(int p:crit){ auto fp=fnorm(U[p].objs); int best=0; double ba=ang(fp,R_[0]); for(int r=1;r<R;++r){double a=ang(fp,R_[r]); if(a<ba){ba=a;best=r;}} Delta[best].push_back(p); }
        std::vector<char> Cact(R,1);
        std::vector<int> chosen=Pidx;
        while((int)chosen.size()<N_){
            int J=-1; for(int r=0;r<R;++r){ if(!Cact[r]) continue; if(J<0||cj[r]<cj[J]) J=r; }
            if(J<0) break;
            if(Delta[J].empty()){ Cact[J]=0; continue; }
            int pick;
            if(cj[J]==0){ int best=-1; double bg=1e300; for(int p:Delta[J]){ auto fp=fnorm(U[p].objs); double g=pbi(fp,R_[J],Rtheta_[J]); if(g<bg){bg=g;best=p;} } pick=best; }
            else { pick=Delta[J][std::uniform_int_distribution<int>(0,(int)Delta[J].size()-1)(rng_)]; }
            Delta[J].erase(std::remove(Delta[J].begin(),Delta[J].end(),pick),Delta[J].end());
            chosen.push_back(pick); ++cj[J];
        }
        while((int)chosen.size()<N_ && fi<(int)fr.size()){ for(int p:fr[fi]){ if((int)chosen.size()>=N_) break; if(std::find(chosen.begin(),chosen.end(),p)==chosen.end()) chosen.push_back(p);} ++fi; }
        std::vector<Sol> out; for(int t=0;t<N_ && t<(int)chosen.size();++t) out.push_back(U[chosen[t]]);
        while((int)out.size()<N_) out.push_back(U[std::uniform_int_distribution<int>(0,n-1)(rng_)]);
        return out;
    }

    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch){
        const auto& b=vault.get_bounds(); int nv=vault.vars_n();
        std::vector<double> c1,c2; ops::sbx(x.vars,y.vars,c1,c2,b,eta_c_,pc_,rng_);
        ops::polynomial_mutation(c1,b,eta_m_,pm_eff(nv),rng_);
        Sol z; z.vars=c1; vault.set_variables(scratch,c1); vault.refresh_objectives(scratch); z.objs=vault.objectives_of(scratch);
        if(constraint_mode!=ConstraintMode::NONE) z.cv=vault.get_cv(scratch);
        upd_ideal(z.objs); return z;
    }
    void store_arch(DataVault<Ind_t>& vault){ vault.reduce(0); vault.expand((int)pop_.size());
        for(int i=0;i<(int)pop_.size();++i) vault.seed_individual((std::size_t)i,pop_[i].vars,pop_[i].objs,{},{}); }

    // Keys of the mating tournament (§III-B): primary — the non-dominated
    //   sorting rank of pop_ (rank[i], smaller=better); secondary — the c_j of
    //   the reference vector the individual is associated with (association by
    //   angle, as in Alg.2). Normalization (f−z*)/(z'−z*), z'=max over F_1.
    void mating_keys(std::vector<int>& rank, std::vector<int>& cj_of) const {
        int n=(int)pop_.size(); rank.assign(n,0); cj_of.assign(n,0);
        if(n==0) return;
        // Non-dominated sorting of pop_ (fast non-dominated sort) → rank = front index.
        std::vector<int> dc(n,0); std::vector<std::vector<int>> dl(n); std::vector<int> cur;
        for(int p=0;p<n;++p){ for(int q=0;q<n;++q){ if(p==q)continue; if(dom(pop_[p],pop_[q])) dl[p].push_back(q); else if(dom(pop_[q],pop_[p])) ++dc[p]; } if(dc[p]==0){ rank[p]=0; cur.push_back(p);} }
        int fidx=0;
        while(!cur.empty()){ std::vector<int> nx; for(int p:cur) for(int q:dl[p]) if(--dc[q]==0){ rank[q]=fidx+1; nx.push_back(q);} cur.swap(nx); ++fidx; }
        // Association of each individual of pop_ to the nearest r∈R_ by angle;
        // c_j = the number of individuals at r.
        std::vector<int> f0; for(int p=0;p<n;++p) if(rank[p]==0) f0.push_back(p);
        std::vector<double> zp(m_,-1e300); for(int p:f0) for(int k=0;k<m_;++k) zp[k]=std::max(zp[k],pop_[p].objs[k]);
        auto fnorm=[&](const std::vector<double>& o){ std::vector<double> r(m_); for(int k=0;k<m_;++k){double d=zp[k]-z_[k]; r[k]=(d>1e-12)?(o[k]-z_[k])/d:(o[k]-z_[k]); } return r; };
        int R=(int)R_.size(); if(R==0) return;
        std::vector<int> assoc(n,0); std::vector<int> cj(R,0);
        for(int p=0;p<n;++p){ auto fp=fnorm(pop_[p].objs); int best=0; double ba=ang(fp,R_[0]); for(int r=1;r<R;++r){double a=ang(fp,R_[r]); if(a<ba){ba=a;best=r;}} assoc[p]=best; ++cj[best]; }
        for(int p=0;p<n;++p) cj_of[p]=cj[assoc[p]];
    }

public:
    DEAGNGCore() = default;
    void set_t_max(int t){ if(t>0) t_max_=t; }
    void set_eta_crossover(double e){ eta_c_=e; }
    void set_eta_mutation(double e){ eta_m_=e; }
    void set_pc(double p){ pc_=p; }
    void set_pm(double p){ pm_=p; }
    // Public setter of the ε for ζ' (§IV-A). An explicit setting disables the
    //   choice by M.
    void set_eps_theta(double e){ eps_theta_=e; eps_user_set_=true; }
    void set_seed(unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); g_=0; NS_=m_*N_;
        age_max_=N_; lambda_=std::max(1,(int)std::floor(0.2*N_)); maxnode_=N_;
        // ε from the number of objectives (§IV-A) unless set explicitly.
        if(!eps_user_set_) eps_theta_=(m_<5)?0.05*M_PI:0.15*M_PI;
        auto Ur=das_dennis::generate_auto(m_,N_); Ru_=Ur; R_=Ru_; Rtheta_.assign(R_.size(),INF_THETA);
        const auto& bd=vault.get_bounds(); std::uniform_real_distribution<double> d(0.0,1.0);
        std::vector<double> vars(vault.vars_n());
        for(int i=0;i<N_;++i){ for(int j=0;j<vault.vars_n();++j){double lo=bd[j].first.value_or(0.0),hi=bd[j].second.value_or(1.0);vars[j]=lo+d(rng_)*(hi-lo);} vault.set_variables(i,vars);}
        vault.sync();
        pop_.clear(); z_.assign(m_,std::numeric_limits<double>::max());
        for(int i=0;i<N_;++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            upd_ideal(s.objs); pop_.push_back(s);}
        AS_.clear(); gn_.clear(); gadj_.clear(); sig_count_=0;
    }
    void setup_seeded(DataVault<Ind_t>& vault){
        m_=vault.objs_n(); N_=vault.pop_size(); g_=0; NS_=m_*N_;
        age_max_=N_; lambda_=std::max(1,(int)std::floor(0.2*N_)); maxnode_=N_;
        // ε from the number of objectives (§IV-A) unless set explicitly.
        if(!eps_user_set_) eps_theta_=(m_<5)?0.05*M_PI:0.15*M_PI;
        auto Ur=das_dennis::generate_auto(m_,N_); Ru_=Ur; R_=Ru_; Rtheta_.assign(R_.size(),INF_THETA);
        pop_.clear(); z_.assign(m_,std::numeric_limits<double>::max());
        for(int i=0;i<(int)vault.active_n();++i){ Sol s; s.vars=vault.variables_of(i); s.objs=vault.objectives_of(i);
            if(constraint_mode!=ConstraintMode::NONE) s.cv=vault.get_cv(i);
            upd_ideal(s.objs); pop_.push_back(s);}
        AS_.clear(); gn_.clear(); gadj_.clear(); sig_count_=0;
    }

    void step(DataVault<Ind_t>& vault){
        ++g_;
        int scratch=vault.expand(1);
        std::uniform_int_distribution<int> di(0,(int)pop_.size()-1);
        // Mating tournament per §III-B — primary the non-dominated sorting
        //   rank, secondary the c_j of the reference vector (smaller=better);
        //   a tie on both criteria is broken at random. Notable fix: it used to
        //   be pairwise dominance only.
        std::vector<int> mrank, mcj; mating_keys(mrank, mcj);
        auto tour=[&](){ int a=di(rng_),b=di(rng_);
            if(mrank[a]!=mrank[b]) return (mrank[a]<mrank[b])?a:b;      // primary: lower front
            if(mcj[a]!=mcj[b])     return (mcj[a]<mcj[b])?a:b;          // secondary: smaller c_j
            return (uni01()<0.5)?a:b; };                                // tie — at random
        std::vector<Sol> Pp; Pp.reserve(N_);
        for(int i=0;i<N_;++i){ int p1=tour(),p2=tour(); Pp.push_back(breed(pop_[p1],pop_[p2],vault,scratch)); }
        if(g_ < (int)((1.0-alpha_stop_)*t_max_)){
            archive_update(Pp);
            gng_update();
            reference_adaptation();
        }
        std::vector<Sol> U=pop_; for(auto&s:Pp) U.push_back(s);
        pop_=env_select(U);
        store_arch(vault);
    }
};

} // namespace mootation

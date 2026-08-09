#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CLIA — A Many-Objective Evolutionary Algorithm with Two Interacting Processes:
//        Cascade Clustering and Reference Point Incremental Learning
// H. Ge, M. Zhao, L. Sun, Z. Wang, G. Tan, Q. Zhang, C.L.P. Chen
// IEEE TEVC 2019, doi:10.1109/TEVC.2018.2874465
//
// MECHANISM (§III). Two interacting processes:
//   (A) Cascade Clustering (CC, §III-A, Alg.1) — the 2N -> N generational
//       selection:
//       1. Frontier Identification (§III-A-1): NDSort down to the first front;
//          split the pool into frontiers F and non-frontiers NF.
//       2. Bi-level Clustering + Intraclass Sort (§III-A-2, Alg.1):
//          - each f in F attaches to the nearest reference vector z by
//            sin(angle(f,z)); the active z are those that received at least one
//            frontier.
//          - within a cluster, sort by PDM ascending (Eq.1):
//              PDM(o,z) = mean(o) + alpha*||o||_2*sin(o,z), alpha=5 (as in PBI);
//            the cluster centre is the f with the smallest PDM.
//          - each nf in NF attaches to the cluster with the nearest centre
//            (Euclidean); within a cluster, sort by d(nf,centre) ascending.
//          - the cluster selection queue is S = <sorted F, sorted NF>.
//       3. Round-robin Picking (§III-A-3, Alg.1): take the head of each queue in
//          turn until |P| = N. The centres (heads) are retained.
//   (B) Reference Point Incremental Learning (RPIL, §III-B, Alg.2) —
//       reference-vector adaptation:
//       - Status sampler: "stable" when the set of active reference vectors has
//         not changed for theta generations.
//         theta = min(20, max(5, ceil(maxFEs/2e4))) (§IV-B).
//       - Trigger (Alg.2): if isStable AND |Z_A| < N, train a classifier on the
//         reference points labelled {active = +, inactive = −} (projected from
//         M to M−1), raise the density D <- D+1, generate
//         Z = generateReference(D,M), score it and keep the n = 2N best
//         (delta = the n-th best score) (§IV-B).
//
// PAPER DEFAULTS (§IV-A/B, Table): alpha=5; SBX eta_c=20, p_c=1; PM eta_m=20,
//   p_m=1/n; theta=min(20,max(5,ceil(maxFEs/2e4))) with maxFEs ~= t_max*N;
//   n_keep=2N.
// REFERENCE POINTS: the paper fixes no count of its own — the initial set is a
//   uniform simplex sampling that the RPIL process then reduces and
//   re-densifies (§III-B, Fig.4). This port requests pop_size points by
//   default, or set_K(k) points if the caller asks, and hands the request to
//   das_dennis::generate_auto, which returns the nearest attainable lattice
//   size >= the request. set_K therefore takes a NUMBER OF POINTS, not a
//   division count (the same contract as every other set_K in this library).
//
// DECLARED DEVIATIONS:
//   CLIA-1 (MAJOR, the RPIL classifier). The paper uses an INCREMENTAL SVM
//     (Gaussian kernel <S,C> = <0.056,10>) on reference points projected by
//     Gram-Schmidt from M to M−1, to estimate the "effective regions" of the
//     simplex. What is implemented here is a functionally equivalent
//     non-parametric estimator: the score of each dense reference point is its
//     Gaussian-weighted proximity (width ~ the kernel scale) to the ACTIVE
//     samples minus its proximity to the INACTIVE samples on the simplex,
//     i.e. a k-NN / Parzen kernel. The n = 2N best-scoring points are kept
//     (delta = the n-th best), as in Alg.2. This preserves the PURPOSE of RPIL
//     — generate dense reference points in effective regions and prune
//     ineffective ones at the boundary — without the full incremental online
//     SVM machinery, which would be a separate library and outside the scope of
//     a header-only C++17 project. The "incremental" aspect is realized as
//     accumulation of samples from previous cycles.
//     HONEST CAVEAT: this is NOT the paper's "margin reuse", which reuses only
//     the reserved samples near the SVM boundary. Here samp_active_ and
//     samp_inactive_ accumulate the ENTIRE history of active/inactive reference
//     points over a run and are never cleared, so the scoring cost grows with
//     the number of cycles. The behaviour is unaffected.
//   CLIA-2 (MINOR). The M -> M−1 Gram-Schmidt projection for the classifier is
//     replaced by working directly on the simplex, where the points already lie
//     on Sum = 1. Under an isotropic kernel this is equivalent, since the
//     projection only removes a redundant coordinate.
//   CLIA-3 (MINOR). PDM uses ||o||_2 * sin(o,z), the full form of Eq.1; the
//     pseudocode of Alg.1 writes mean(f) + sin(z,f). Eq.1 in the text takes
//     precedence.
//   CLIA-4. "Denser reference points" (ge2019, footnote 3): for M < 8 the
//     density of the SINGLE-LAYER lattice is raised (D <- D+1); for M >= 8 the
//     TWO-LAYER Das-Dennis scheme is used (dense boundary Hb, sparse inner
//     Hi = Hb/2), as in NSGA-III. Using a single-layer generate(M, H_active+D)
//     at M >= 8 was a deviation.
//   CLIA-5 (MINOR). One offspring per pair (SBX, first child), as elsewhere in
//     this library; parents are mated at random from the pool (§III-C,
//     "evolve").
//   CLIA-6 (MINOR). Real-valued genome; binary and mixed are out of scope
//     (constraint_mode=NONE).
//
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP makes the Frontier Identification step of Cascade
//   Clustering (§III-A-1) a CONSTRAINED non-dominated sort, which is the gate
//   into every later stage. The RPIL learner and the angular clustering are
//   untouched. The paper is unconstrained.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class CLIACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    CLIACore() = default;
    void set_seed(unsigned s) { rng_.seed(s); }
    void set_t_max(int t) { t_max_ = t; }
    void set_K(int k) { K_req_ = k; }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e) { eta_m_ = e; }
    void set_pc(double p) { pc_ = p; }
    void set_pm(double p) { pm_ = p; }
    void set_alpha(double a) { alpha_ = a; }
    void set_n_keep_factor(double f) { n_keep_factor_ = f; }  // n = f·N (default 2)

    void setup(DataVault<Ind_t>& vault);
    void setup_seeded(DataVault<Ind_t>& vault);
    void step(DataVault<Ind_t>& vault);

private:
    std::mt19937 rng_{std::random_device{}()};
    int    t_max_ = 250, t_ = 0;
    int    K_req_ = 0;            // requested NUMBER of reference points (0 -> pop_size)
    double eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0, pm_ = -1.0;
    double alpha_ = 5.0;          // Eq.1, as in PBI
    double n_keep_factor_ = 2.0;  // §IV-B: n = 2N

    int m_ = 0, N_ = 0;
    int theta_ = 5;               // §IV-B status-sampler threshold
    int H_base_ = 0;              // base lattice density (divisions)
    int extra_density_ = 0;       // D (Alg.2): extra divisions added by RPIL

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };

    // Reference vectors (Z), normalized to unit length for the angles.
    std::vector<std::vector<double>> Z_;        // active / current reference vectors
    // Status-sampler state: the history of active index sets.
    std::vector<char> last_active_mask_;
    int  stable_count_ = 0;
    // RPIL sample accumulation on the simplex (CLIA-1): the ENTIRE history of
    // active/inactive reference points is kept for the run — this is not the
    // paper's "margin reuse", and it is never cleared.
    std::vector<std::vector<double>> samp_active_, samp_inactive_;

    double pm_eff(int nv) const { return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0); }

    static std::vector<double> unit(std::vector<double> v) {
        double n = 0; for (double x : v) n += x * x;
        n = std::sqrt(std::max(n, 1e-300));
        for (double& x : v) x /= n; return v;
    }
    // sine of the angle between o and z (z a reference vector); the DM term of Eq.1.
    static double sin_angle(const std::vector<double>& o, const std::vector<double>& z) {
        double oo = 0, zz = 0, oz = 0;
        for (std::size_t i = 0; i < o.size(); ++i) { oo += o[i]*o[i]; zz += z[i]*z[i]; oz += o[i]*z[i]; }
        double num = oo * zz - oz * oz;            // ||o||²||z||² − (oᵀz)²
        if (num < 0) num = 0;
        double den = std::sqrt(std::max(zz, 1e-300));
        // ||o||₂·sin = sqrt(num)/||z||
        return std::sqrt(num) / den;               // = ||o||·sin(o,z)
    }
    static double l2(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0; for (std::size_t i = 0; i < a.size(); ++i) { double d = a[i]-b[i]; s += d*d; } return std::sqrt(s);
    }
    static double norm2(const std::vector<double>& a) { double s=0; for(double x:a) s+=x*x; return std::sqrt(s); }

    // PDM (Eq.1): mean(o) + α·||o||₂·sin(o,z). osin = ||o||·sin precomputed.
    double pdm(const std::vector<double>& o, double osin) const {
        double mean = 0; for (double x : o) mean += x; mean /= (double)o.size();
        return mean + alpha_ * osin;
    }

    bool dominates(const std::vector<double>& a, const std::vector<double>& b) const {
        bool better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) better = true;
        }
        return better;
    }
    // Constraint-aware form (Deb's constrained domination when the mode is on).
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }
    // §III-A-1: identify first non-dominated front; return mask (1=frontier).
    std::vector<char> frontier_mask(const std::vector<Sol>& P) const {
        int n = (int)P.size();
        std::vector<char> fm(n, 1);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && dominates(P[j],P[i])) { fm[i] = 0; break; }
        return fm;
    }

    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch) {
        const auto& b = vault.get_bounds(); int nv = vault.vars_n();
        std::vector<double> c1, c2;
        ops::sbx(x.vars, y.vars, c1, c2, b, eta_c_, pc_, rng_);          // SBX
        ops::polynomial_mutation(c1, b, eta_m_, pm_eff(nv), rng_);       // PM
        Sol z; z.vars = c1;
        vault.set_variables(scratch, c1); vault.refresh_objectives(scratch);
        z.objs = vault.objectives_of(scratch);
        if (constraint_mode != ConstraintMode::NONE) z.cv = vault.get_cv(scratch);
        return z;
    }

    // ── Cascade Clustering (Alg.1): select N out of pool ───────────────────
    std::vector<Sol> cascade_select(const std::vector<Sol>& pool,
                                    std::vector<char>& active_out) {
        int n = (int)pool.size();
        int R = (int)Z_.size();
        active_out.assign(R, 0);

        // 1. Frontier identification (§III-A-1)
        std::vector<char> fm = frontier_mask(pool);

        // 2a. Attach frontiers to nearest ref-vector by sin angle; precompute osin.
        std::vector<std::vector<int>> cl_F(R);   // frontier indices per ref
        std::vector<double> osin(n, 0.0);        // ||o||·sin(o,z*) of attached
        for (int i = 0; i < n; ++i) {
            if (!fm[i]) continue;
            int best = 0; double bs = sin_angle(pool[i].objs, Z_[0]);
            for (int r = 1; r < R; ++r) {
                double s = sin_angle(pool[i].objs, Z_[r]);
                if (s < bs) { bs = s; best = r; }
            }
            cl_F[best].push_back(i); osin[i] = bs; active_out[best] = 1;
        }

        // 2b. Intraclass sort frontiers by PDM (Eq.1) ascend; center = min PDM.
        struct Cluster { std::vector<int> F, NF; int center = -1; };
        std::vector<Cluster> clusters;
        std::vector<int> ref_of_cluster;
        for (int r = 0; r < R; ++r) {
            if (cl_F[r].empty()) continue;
            Cluster c;
            c.F = cl_F[r];
            std::sort(c.F.begin(), c.F.end(), [&](int a, int b) {
                return pdm(pool[a].objs, osin[a]) < pdm(pool[b].objs, osin[b]);
            });
            c.center = c.F.front();          // smallest PDM
            clusters.push_back(std::move(c));
            ref_of_cluster.push_back(r);
        }
        int C = (int)clusters.size();
        if (C == 0) {  // degenerate: no frontier attached anywhere — keep best N by mean
            std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                double ma=0,mb=0; for(double v:pool[a].objs) ma+=v; for(double v:pool[b].objs) mb+=v; return ma<mb; });
            std::vector<Sol> out; for (int k = 0; k < std::min(N_, n); ++k) out.push_back(pool[idx[k]]);
            return out;
        }

        // 2c. Attach non-frontiers to nearest cluster CENTER (euclid); sort by dist.
        for (int i = 0; i < n; ++i) {
            if (fm[i]) continue;
            int best = 0; double bd = l2(pool[i].objs, pool[clusters[0].center].objs);
            for (int c = 1; c < C; ++c) {
                double d = l2(pool[i].objs, pool[clusters[c].center].objs);
                if (d < bd) { bd = d; best = c; }
            }
            clusters[best].NF.push_back(i);
        }
        for (auto& c : clusters) {
            std::sort(c.NF.begin(), c.NF.end(), [&](int a, int b) {
                return l2(pool[a].objs, pool[c.center].objs)
                     < l2(pool[b].objs, pool[c.center].objs);
            });
        }

        // 3. Round-robin picking (Alg.1): queue = ⟨F, NF⟩.
        std::vector<std::vector<int>> queue(C);
        for (int c = 0; c < C; ++c) {
            queue[c] = clusters[c].F;
            queue[c].insert(queue[c].end(), clusters[c].NF.begin(), clusters[c].NF.end());
        }
        std::vector<std::size_t> head(C, 0);
        std::vector<Sol> out; out.reserve(N_);
        std::vector<char> picked(n, 0);
        int ci = 0, guard = 0, maxguard = C * (n + 2);
        while ((int)out.size() < N_ && guard++ < maxguard) {
            if (head[ci] < queue[ci].size()) {
                int idx = queue[ci][head[ci]++];
                if (!picked[idx]) { picked[idx] = 1; out.push_back(pool[idx]); }
            }
            ci = (ci + 1) % C;
        }
        return out;
    }

    // ── RPIL (Alg.2): adapt Z_ when stable & |active|<N ────────────────────
    // CLIA-1: a non-parametric Parzen estimator replaces the incremental SVM.
    void update_status_and_learn(const std::vector<char>& active_mask) {
        // Status sampler (§III-B): stable if the active mask has been unchanged
        // for theta generations.
        if (active_mask == last_active_mask_) stable_count_++;
        else { stable_count_ = 0; last_active_mask_ = active_mask; }

        int n_active = 0; for (char c : active_mask) n_active += c ? 1 : 0;

        // Trigger (Alg.2): isStable(θ) AND |Z_A| < N.
        if (stable_count_ < theta_ || n_active >= N_) return;

        // Accumulate samples (the "incremental" aspect): active reference
        // points on the simplex are positives, inactive ones are negatives. The
        // ENTIRE run history is kept, not only near-boundary samples — see
        // CLIA-1.
        int R = (int)Z_.size();
        for (int r = 0; r < R; ++r) {
            // normalize onto the simplex (Sum = 1) for the classifier
            std::vector<double> p = Z_[r]; double s = 0; for (double v : p) s += v;
            if (s < 1e-300) continue; for (double& v : p) v /= s;
            if (active_mask[r]) samp_active_.push_back(p);
            else                samp_inactive_.push_back(p);
        }
        if (samp_active_.empty()) { stable_count_ = 0; return; }

        // Generate denser reference points (Alg.2: D←D+1; CLIA-4).
        // Per ge2019 footnote 3 ("For M < 8, we
        //   increase the lattice density. Else, we increase the density of
        //   reference points in the boundary layer and the inner layers"), the
        //   dense points at M >= 8 are generated by the TWO-LAYER Das-Dennis
        //   scheme, as in NSGA-III. A single-layer lattice at large M either
        //   explodes combinatorially or is far too sparse. For M < 8 the
        //   single-layer density is used. The two-layer reference is
        //   nsga3.hpp / das_dennis.hpp (dense boundary Hb > sparse inner Hi;
        //   deb2014 §V).
        extra_density_++;
        int H = H_base_ + extra_density_;
        std::vector<std::vector<double>> dense;
        if (m_ >= 8) {
            int Hb = std::max(2, H);              // dense boundary layer
            int Hi = std::max(1, Hb / 2);         // sparse inner layer (Hb > Hi)
            dense = das_dennis::generate_two_layer(m_, Hb, Hi);
        } else {
            dense = das_dennis::generate(m_, H);  // on the simplex, Sum = 1
        }
        if (dense.empty()) { stable_count_ = 0; return; }

        // Score each dense point (CLIA-1: Parzen kernel; S≈kernel scale).
        const double bw = 0.056;                    // kernel scale ⟨S⟩ (§IV-B)
        const double inv2s2 = 1.0 / (2.0 * bw * bw);
        std::vector<double> score(dense.size(), 0.0);
        for (std::size_t k = 0; k < dense.size(); ++k) {
            double pos = 0, neg = 0;
            for (const auto& a : samp_active_)   { double d = l2(dense[k], a); pos += std::exp(-d*d*inv2s2); }
            for (const auto& a : samp_inactive_) { double d = l2(dense[k], a); neg += std::exp(-d*d*inv2s2); }
            double na = (double)samp_active_.size(), ni = (double)samp_inactive_.size();
            score[k] = (na > 0 ? pos/na : 0.0) - (ni > 0 ? neg/ni : 0.0);
        }

        // Keep n = n_keep_factor·N best by score; δ = n-th best (§IV-B).
        int n_keep = std::max(N_, (int)std::lround(n_keep_factor_ * N_));
        std::vector<int> ord(dense.size()); std::iota(ord.begin(), ord.end(), 0);
        if ((int)ord.size() > n_keep)
            std::nth_element(ord.begin(), ord.begin() + n_keep, ord.end(),
                             [&](int a, int b) { return score[a] > score[b]; });
        int keep = std::min<int>(n_keep, (int)ord.size());

        std::vector<std::vector<double>> newZ;
        newZ.reserve(keep);
        for (int i = 0; i < keep; ++i) newZ.push_back(unit(dense[ord[i]]));
        if (!newZ.empty()) Z_ = std::move(newZ);

        // reset sampler after adaptation
        stable_count_ = 0;
        last_active_mask_.clear();
    }

    void compute_theta() {
        // §IV-B: θ = min(20, max(5, ceil(maxFEs/2e4))), maxFEs ≈ t_max·N.
        double maxFEs = (double)t_max_ * (double)N_;
        int th = (int)std::ceil(maxFEs / 2e4);
        theta_ = std::min(20, std::max(5, th));
    }

    void init_refs() {
        int req = (K_req_ > 0) ? K_req_ : N_;
        auto Vr = das_dennis::generate_auto(m_, req);
        Z_.clear(); for (auto& v : Vr) Z_.push_back(unit(v));
        // restore the base division count H for the next density increase
        H_base_ = das_dennis::find_H_le(m_, (long long)Vr.size());
        if (H_base_ < 1) H_base_ = 1;
        extra_density_ = 0;
        last_active_mask_.assign(Z_.size(), 0);
        stable_count_ = 0;
        samp_active_.clear(); samp_inactive_.clear();
    }

    void write_pop(DataVault<Ind_t>& vault, const std::vector<Sol>& P) {
        vault.reduce(0); vault.expand((int)P.size());
        for (int i = 0; i < (int)P.size(); ++i)
            vault.seed_individual((std::size_t)i, P[i].vars, P[i].objs, {}, {});
    }

    std::vector<Sol> pop_;   // current population (N)
};

// ── setup ────────────────────────────────────────────────────────────────
template <typename Ind_t>
void CLIACore<Ind_t>::setup(DataVault<Ind_t>& vault) {
    m_ = vault.objs_n(); N_ = vault.pop_size();
    init_refs(); compute_theta(); t_ = 0;
    const auto& bd = vault.get_bounds();
    std::uniform_real_distribution<double> d(0.0, 1.0);
    std::vector<double> vars(vault.vars_n());
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < vault.vars_n(); ++j) {
            double lo = bd[j].first.value_or(0.0), hi = bd[j].second.value_or(1.0);
            vars[j] = lo + d(rng_) * (hi - lo);
        }
        vault.set_variables(i, vars);
    }
    vault.sync();
    pop_.clear();
    for (int i = 0; i < N_; ++i) {
        Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
        if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
        pop_.push_back(std::move(s));
    }
}

template <typename Ind_t>
void CLIACore<Ind_t>::setup_seeded(DataVault<Ind_t>& vault) {
    m_ = vault.objs_n(); N_ = vault.pop_size();
    init_refs(); compute_theta(); t_ = 0;
    pop_.clear();
    for (int i = 0; i < (int)vault.active_n(); ++i) {
        Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
        if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
        pop_.push_back(std::move(s));
    }
}

// ── step (one generation) ──────────────────────────────────────────────────
template <typename Ind_t>
void CLIACore<Ind_t>::step(DataVault<Ind_t>& vault) {
    int scratch = vault.expand(1);
    int sz = (int)pop_.size();
    if (sz == 0) return;
    std::uniform_int_distribution<int> pick(0, sz - 1);

    // (1) Evolve: offspring of size N (§III-C). Random mating over population.
    std::vector<Sol> O; O.reserve(sz);
    for (int i = 0; i < sz; ++i) {
        int a = pick(rng_), b = pick(rng_);
        for (int t = 0; t < 5 && b == a; ++t) b = pick(rng_);
        O.push_back(breed(pop_[a], pop_[b], vault, scratch));   // CLIA-5: one child
    }

    // (2) Potential population 2N = P ∪ O.
    std::vector<Sol> pool = pop_;
    for (auto& s : O) pool.push_back(s);

    // (3) Cascade clustering selection → next P; collect ref activities.
    std::vector<char> active_mask;
    pop_ = cascade_select(pool, active_mask);

    // (4) RPIL: status sampler + incremental learning when stable & under-active.
    update_status_and_learn(active_mask);

    ++t_;
    write_pop(vault, pop_);
}

} // namespace mootation

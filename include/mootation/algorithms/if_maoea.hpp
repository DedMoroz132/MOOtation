#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// IF-MaOEA — A many-objective evolutionary algorithm combining Simplified
//        Hypervolume and a method for reference point sampling based on
//        Angular relationship.
// T. Chao, S. Wang, S. Wang, M. Yang — Applied Soft Computing 2024.
// doi:10.1016/j.asoc.2024.111881   (PII S1568494624006550)
// Applied Soft Computing 163 (2024) 111881.
// FIX 2026-08-09 (Crossref sweep): the DOI read 10.1016/j.asoc.2024.111872,
//   which resolves to a bridge-tower buffeting-reliability paper. The article
//   number and the DOI suffix agree at 111881.
//
//
// NAMING NOTE. The paper calls the method IF-MaOEA ("improved-fitness-based
// many-objective optimization algorithm"). The class name keeps that acronym;
// the two load-bearing mechanisms named in the paper title are Simplified
// Hypervolume (S_HV) and reference-point sampling by angular relationship.
//
// GENERATIONAL SCHEME (Algorithm 1):
//   1. RefPoint <- angular-relationship sampling (§2.2, Eq.2-4), N points.
//   2. z (ideal) and z^nad (nadir) are recomputed inside normalize() from
//      whichever pool it is handed — P ∪ AS for mating, P ∪ O for
//      environmental selection, AS ∪ P for the archive sieve. They are NOT
//      carried as class state. See IF-MaOEA-7: the paper is contradictory here.
//   3. AS — external archive, initially empty.
//   4. while: ASS = [P, AS]; P' = MatingPoolSelection(ASS,...) (Alg.2);
//      O = Variation(P') (SBX + polynomial mutation);
//      CA = P ∪ O; P = EnvironmentalSelection(CA,...) (Alg.3);
//      AS = ND(AS, P, ...) — the same sieve as environmental selection.
//
// REFERENCE POINT SAMPLING (§2.2, Eq.2-4):
//   Angular coordinates ϑ_j ∈ Ω = {0, π/2L, ..., π/2} constrained by Eq.2:
//   Σ_{j<=m} ϑ_j ∈ [(m-1)π/2, m·π/2] for m<M (the bounds are NON-strict — per
//   Eq.3 and the Fig.3 example, see IF-MaOEA-1), and Σ_{j=1..M} ϑ_j =
//   (M-1)π/2; the set cardinality is C(L+M-1, M-1).
//   W_concave: projection cos(ϑ_j), divided by the norm (Steps 3-4).
//   W_unit: the same lattice in linear coordinates u_j = (L−i_j)/L — the
//   "traditional" unit simplex with the same M and L, in 1-to-1 correspondence
//   with the angle sets; W_convex = 2·W_unit − W_concave (Eq.4). Working set =
//   W_concave ∪ W_convex (∪ inner; Step 5 adds an inner layer).
//
// FITNESS (Alg.2, Eq.5-7):
//   norm f'_i = (f_i − z)/(z^nad − z) (Eq.5).
//   cosθ_ij = f'_i·Ref_j / (‖f'_i‖‖Ref_j‖); the max over j is taken (the
//     nearest vector), i.e. the minimum angle (Eq.6).
//   S_HV V(j): per objective, take the MAX among the neighbouring solutions
//     (the nearest one on each side along that objective) as the reference
//     point; the volume is the product of the gaps; the MIN volume over
//     objectives is taken (§2.3-B; the details live in [43], which is not
//     available — see IF-MaOEA-2 for the competing reading).
//   fitness(j) = V(j)·cosθ_ij / ‖f'_j‖ (Eq.7) — larger is better.
//
// ENVIRONMENTAL SELECTION (Alg.3, §2.4):
//   NondominatedSort(CA); whole fronts are accepted up to the critical one, k.
//   Within the critical front: first the "optimally distributed individuals" —
//   for each RefPoint, the individual closest to its VERTEX by EUCLIDEAN
//   distance in the normalized space (§2.4, the IGD concept; the selection was
//   previously by angle, which is a different criterion — see the note at
//   optimally_distributed); if there are more than needed, take those with the
//   smaller ‖f'‖ (Euclidean distance to the ideal; the paper says "smaller
//   Euclidean distance" without naming the target); if fewer, fill up by
//   maximum fitness (Eq.7).
//
// PAPER SETTINGS (§3.1, Table 2): the population size N and the
//   function-evaluation budget, per objective count M. Table 2 carries only
//   (p1, p2), N, D and FE.
// BEYOND THE PAPER — conventional defaults: SBX η_c=20, p_c=1.0; PM η_m=20,
//   p_m=1/n. The paper names SBX and polynomial mutation (§2.1) but gives NO
//   numeric η_c, p_c, η_m or p_m anywhere, so these are not attributable to
//   §3.1 or Table 2.
//   L (the number of angle divisions) is chosen so that |RefPoint| ≈ N.
//
// DECLARED DEVIATIONS:
//   IF-MaOEA-1 (RESOLVED).
//     The header used to claim a scope cut — "W_convex is NOT computed" —
//     while the code did compute it, with TWO bugs: (a) a strict upper bound on
//     the partial sums of Eq.2 discarded every set with ϑ1=π/2, losing the
//     entire edge w1=0 (15 points instead of 21 at M=3, L=5, missing the unit
//     vectors (0,1,0) and (0,0,1)); (b) W_unit was built inverted (i_j/Σi_j
//     instead of (L−i_j)/L), so the convex branch of Eq.4 produced mirrored
//     points clamped into the edges. Both fixed: the bounds are NON-strict (per
//     Eq.3 and Fig.3 of the paper — the printed Eq.2 with a strict "<"
//     contradicts the paper's own example), and W_unit = (L−i_j)/L is derived
//     from the SAME angle indices (1-to-1 pairing, Σ=1 by construction).
//     Verified by execution: M=3, L=5 yields exactly 21 sets (= C(7,2))
//     including the unit vectors; angles (π/10, 2π/5, π/2) give
//     W_convex = (0.845, 0.155, 0), matching the paper. The v<0 clamp in
//     build_refpoints is kept as a numerical guard — effectively inactive once
//     the pairing is correct.
//   IF-MaOEA-2 (AMBIGUOUS — two readings, one chosen). For each objective k the
//     "neighbour" is the nearest solution with a LARGER f_k, and that whole
//     solution vector serves as the reference point; the volume is the product
//     of the positive gaps, and the MINIMUM volume over objectives is taken.
//     The competing reading is a single constructed corner point assembled
//     component-wise from the per-objective maxima of several neighbours, which
//     is closer to what Fig. 8 depicts. The per-objective reference-solution
//     reading was chosen because only it makes §2.3-B's phrase "the minimum
//     volume value among all objective functions" operative — a single corner
//     point yields one volume, not one per objective. [43], which holds the
//     details, is not available. Consequence worth knowing: because the
//     reference always has a larger k-th objective it can never dominate the
//     candidate, so on a mutually non-dominated layer every volume is positive.
//   IF-MaOEA-3 (DEVIATION — a different operation, not merely a
//     reconstruction). §2.4 invokes the "reference point adjustment strategy"
//     of [44] = AR-MOEA. That sentence does describe an operation: AR-MOEA's
//     Alg. 4 AdjustLocation moves each reference point to the orthogonal
//     projection of its associated solution onto the reference vector.
//     THAT PROJECTION IS NOT IMPLEMENTED HERE. adapt_refpoints ports AR-MOEA's
//     set-level Operation 3 instead — keep the valid reference points, then top
//     up with the directions of actual solutions by max-min angle, rebuilding
//     from Ref0_ every generation.
//     Why: the paper fixes neither the schedule nor the pool for the
//     projection, and this file uses Ref_ only through angles and a
//     simplex projection, where relocating a point along its own ray changes
//     little. The effect obtained is the intended one — boundary coverage on
//     convex and irregular fronts — but by a different mechanism than [44]
//     specifies. An earlier version of this entry claimed the paper gives "a
//     single sentence with no formula"; that understated what §2.4 says.
//   IF-MaOEA-4 (MINOR). One offspring per pair (SBX, first child), as in
//     ISDE+RD and others in this library; |O| = N.
//   IF-MaOEA-5 (MINOR). Real-valued genome; binary is out of scope.
//   IF-MaOEA-6 (MINOR). The archive AS is truncated by the same sieve to ≤N;
//     ASS = P ∪ AS.
//   IF-MaOEA-7 (AMBIGUOUS — the paper contradicts itself on the normalization
//     range). Alg.1 lines 3-5 and 11 build a PERSISTENT Range = [z, z^nad]:
//     z^nad is taken from the INITIAL population and z is updated monotonically
//     by the offspring. §2.3-A, under Eq.5, instead defines z = min over ASS and
//     z^nad = max over ASS, i.e. over the CURRENT pool. The two cannot both
//     hold. This code follows §2.3-A and recomputes both inside normalize() for
//     each pool it is given. A visible consequence: Alg.1 line 11, which
//     updates z from the offspring, has no counterpart here — under
//     recomputation it would be redundant.
//   IF-MaOEA-8 (PAPER MISPRINT, resolved from the paper's own text). Alg.2
//     line 8 reads "Calculate the MINIMUM cosine of the angle between ASS and
//     RefPoint". Taken literally that selects the FARTHEST reference vector,
//     and since fitness = S_HV·Cosine/Norm is maximized (lines 14-17), it would
//     reward misalignment. The body text is unambiguous the other way: §2.3-A
//     says each individual is associated with "the reference vector that has
//     the smallest angle relative to it", and "selecting offspring with smaller
//     angles relative to the reference vector is likely to yield a more uniform
//     distribution". Smallest angle = LARGEST cosine, so this port maximizes
//     cos θ_ij over j (Eq.6).
//   IF-MaOEA-9 (PAPER PSEUDOCODE BROKEN, resolved from §2.4). Alg.3 as printed
//     cannot run: (a) lines 5-10 accumulate the critical-layer picks into p and
//     then "return Q" at line 12 without ever merging p into Q, and line 8's
//     "p <- argmax fitness(F_k)" overwrites p instead of appending, so the loop
//     at line 7 never terminates; (b) the guard at line 4, |Front_k| > N-|Q|,
//     is false in the exact-fit case |Front_k| = N-|Q|, which skips the whole
//     block and returns a Q SHORTER than N. This port follows the explanatory
//     paragraph instead: whole fronts are accepted while the running total
//     stays <= N (so the exact-fit case is absorbed normally), and the critical
//     front contributes the optimally distributed individuals first, trimmed by
//     smallest ||f'|| when they overflow and topped up by descending fitness
//     (Eq.7) when they fall short.
// EXTENSIONS BEYOND THE PAPER (off by default): constraint_mode exists for API
//   uniformity and does not change the IF-MaOEA logic (NONE).
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
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class IFMaOEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    IFMaOEACore() = default;
    void set_seed(unsigned s) { rng_.seed(s); }
    void set_t_max(int t) { t_max_ = t; }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e) { eta_m_ = e; }
    void set_pc(double p) { pc_ = p; }
    void set_pm(double p) { pm_ = p; }

private:
    std::mt19937 rng_{std::random_device{}()};
    int    t_max_ = 250, t_ = 0;
    double eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0, pm_ = -1.0;  // §3.1 Table 2

    struct Sol { std::vector<double> vars, objs; double cv=0.0; };

    int m_ = 0, N_ = 0;
    std::vector<std::vector<double>> Ref_;   // reference points (unit-norm rows)
    std::vector<std::vector<double>> Ref0_;  // static base set (used by §2.4)
    std::vector<Sol> P_;                      // current population
    std::vector<Sol> AS_;                     // external archive

    double pm_eff(int nv) const { return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0); }

    // ── reference-point sampling based on angular relationship (§2.2) ────────
    // Recursively enumerate angular coordinates ϑ_j ∈ Ω with constraints Eq.2,
    // project via cosine (Step 3) and normalise (Step 4) → W_concave.
    // Enumerates the angle sets (Eq.2; the partial-sum bounds are NON-strict —
    // see the note below). Each set yields TWO points on the simplex (Σ=1):
    // W_concave (cosine projection of the angles — concave PF) and W_unit (the
    // same lattice in linear coordinates u_j = (L−i_j)/L — the "traditional"
    // unit simplex, i.e. Das–Dennis with the same M and L, in 1-to-1
    // correspondence with the angle sets).
    // Per Eq.4: W_convex = 2·W_unit − W_concave (see build_refpoints).
    static void enum_angles(int M, int L, int dim, double sum_so_far,
                            std::vector<int>& idx,
                            std::vector<std::vector<double>>& out_concave,
                            std::vector<std::vector<double>>& out_unit)
    {
        const double step = (M_PI / 2.0) / L;          // Ω increment
        if (dim == M - 1) {
            // last angle fixed by Σϑ = (M-1)π/2
            double last = (M - 1) * M_PI / 2.0 - sum_so_far;
            if (last < -1e-9 || last > M_PI / 2.0 + 1e-9) return;
            double k = last / step;
            double kr = std::round(k);
            if (std::abs(k - kr) > 1e-6) return;
            std::vector<double> ang(M);
            for (int j = 0; j < M - 1; ++j) ang[j] = idx[j] * step;
            ang[M - 1] = std::max(0.0, std::min(M_PI / 2.0, last));
            // W_concave: cosine projection, normalized onto the simplex (Steps 3-4)
            std::vector<double> wc(M); double sc = 0.0;
            for (int j = 0; j < M; ++j) { wc[j] = std::cos(ang[j]); if (wc[j] < 0) wc[j] = 0; sc += wc[j]; }
            if (sc < 1e-12) return;
            for (int j = 0; j < M; ++j) wc[j] /= sc;
            // FIX 2026-07-07: W_unit —
            // A point of the TRADITIONAL unit simplex (Das–Dennis with the same
            // M and L), paired 1-to-1 with the angle set: u_j = (L−i_j)/L;
            // Σu = 1 by construction (Σi_j = (M−1)·L), and the monotonicity
            // agrees with the cosine (i_j=0 → u_j=1, i_j=L → u_j=0). The
            // earlier i_j/Σi_j form was inverted, which made the convex branch
            // of Eq.4 produce mirrored points. Check (M=3, L=5, angles
            // (π/10, 2π/5, π/2)): W_unit=(0.8, 0.2, 0), hence W_convex =
            // 2·W_unit − W_concave = (0.845, 0.155, 0) — matching the paper.
            std::vector<double> wu(M);
            for (int j = 0; j < M - 1; ++j) wu[j] = (double)(L - idx[j]) / (double)L;
            wu[M - 1] = (double)(L - (int)kr) / (double)L;
            out_concave.push_back(wc);
            out_unit.push_back(wu);
            return;
        }
        int m = dim + 1;  // 1-based count of assigned angles after this one
        for (int i = 0; i <= L; ++i) {
            double a = i * step;
            double ns = sum_so_far + a;
            // The partial-sum bounds are NON-strict:
            // (m-1)·π/2 <= Σ <= m·π/2 for m<M.
            // The printed Eq.2 uses a strict "<", but the paper's own example is
            // unambiguously non-strict: Ω includes π/2, Eq.3 states only
            // non-strict inequalities, and the Fig.3 table (M=3, L=5) lists 21
            // sets, six of them with ϑ1=π/2. The strict bound removed the whole
            // edge w1=0 (15 points instead of 21, missing the unit vectors
            // (0,1,0)/(0,0,1)); with the non-strict bound the enumeration yields
            // exactly C(L+M-1, M-1) sets — 1-to-1 with the W_unit lattice.
            double lo = (m - 1) * M_PI / 2.0 - 1e-9;
            double hi = m * M_PI / 2.0 + 1e-9;
            if (ns < lo || ns > hi) continue;
            idx[dim] = i;
            enum_angles(M, L, dim + 1, ns, idx, out_concave, out_unit);
        }
    }

    static std::vector<std::vector<double>> angular_sampling(int M, int L) {
        std::vector<std::vector<double>> conc, unit;
        std::vector<int> idx(M, 0);
        enum_angles(M, L, 0, 0.0, idx, conc, unit);
        return conc;
    }

    // Reference set of §2.2 + Eq.4: per angular sample, W_concave (concave PF)
    // and W_convex = 2·W_unit − W_concave (convex PF). Covers both front types.
    void build_refpoints() {
        std::vector<std::vector<double>> conc, unit;
        for (int L = 1; L <= 60; ++L) {
            conc.clear(); unit.clear();
            std::vector<int> idx(m_, 0);
            enum_angles(m_, L, 0, 0.0, idx, conc, unit);
            if ((int)conc.size() * 2 >= N_) break;     // concave+convex ≈ N
        }
        if (conc.empty()) {                            // fallback
            conc = das_dennis::generate_auto(m_, std::max(1, N_ / 2));
            unit = conc;
        }
        std::vector<std::vector<double>> set;
        for (std::size_t i = 0; i < conc.size(); ++i) {
            set.push_back(conc[i]);                     // W_concave
            std::vector<double> wx(m_); double s = 0.0; // W_convex (Eq.4)
            for (int k = 0; k < m_; ++k) {
                double v = 2.0 * unit[i][k] - conc[i][k];
                if (v < 0.0) v = 0.0;
                wx[k] = v; s += v;
            }
            if (s > 1e-9) { for (int k = 0; k < m_; ++k) wx[k] /= s; set.push_back(wx); }
        }
        // Step 5: inner layer contracted toward the centroid (deb2014 §V)
        if ((int)set.size() < N_) {
            std::vector<std::vector<double>> inner = set;
            double c = 1.0 / m_;
            for (auto& w : inner) for (double& wi : w) wi = wi / 2.0 + c / 2.0;
            set.insert(set.end(), inner.begin(), inner.end());
        }
        // unit-normalize (L2) for cosθ (Eq.6)
        Ref_.clear();
        for (auto& w : set) {
            double n = 0; for (double v : w) n += v * v;
            n = std::sqrt(std::max(n, 1e-300));
            for (double& v : w) v /= n;
            Ref_.push_back(w);
        }
        Ref0_ = Ref_;                  // keep the static base set (§2.4)
    }

    // ── §2.4 Reference-point adjustment ──────────────────────────────────────
    // A port of the AR-MOEA principle ([44] Tian et al. 2017) — an INDEPENDENT
    // copy that does not depend on ar_moea.hpp. Motivation: on convex and
    // irregular fronts the optimally distributed individuals cluster in the
    // centre and the boundaries stay uncovered. The adaptation keeps the "valid"
    // reference points (nearest by angle to at least one solution) and tops them
    // up with DIRECTIONS of the actual solutions by max-min angle, up to |Ref0_|
    // (as in AR-MOEA Op.2-3). Rebuilt from Ref0_ on every call, so nothing
    // accumulates and the set cannot degenerate. Modifies ONLY Ref_ for the
    // subsequent optimally_distributed; the caller restores Ref0_.
    void adapt_refpoints(const std::vector<std::vector<double>>& F,
                         const std::vector<int>& idxs) {
        const int NR = static_cast<int>(Ref0_.size());
        if (idxs.empty() || NR == 0) { Ref_ = Ref0_; return; }
        // unit directions of the candidate solutions
        std::vector<std::vector<double>> S; S.reserve(idxs.size());
        for (int i : idxs) {
            double n = norm2(F[i]); if (n < 1e-300) continue;
            std::vector<double> d(m_);
            for (int k = 0; k < m_; ++k) d[k] = F[i][k] / n;
            S.push_back(std::move(d));
        }
        if (S.empty()) { Ref_ = Ref0_; return; }
        // R^valid: the Ref0_ points that are nearest by angle to >= 1 solution
        std::vector<char> valid(NR, 0);
        for (const auto& s : S) {
            int arg = 0; double cm = -2.0;
            for (int r = 0; r < NR; ++r) { double c = cosang(s, Ref0_[r]); if (c > cm) { cm = c; arg = r; } }
            valid[arg] = 1;
        }
        std::vector<std::vector<double>> Rp;
        for (int r = 0; r < NR; ++r) if (valid[r]) Rp.push_back(Ref0_[r]);
        // top up with solution directions by max-min angle, up to |Ref0_|
        auto ang = [&](const std::vector<double>& a, const std::vector<double>& b) {
            return std::acos(cosang(a, b)); };
        std::vector<char> used(S.size(), 0);
        std::vector<double> minang(S.size(), std::numeric_limits<double>::max());
        for (std::size_t i = 0; i < S.size(); ++i)
            for (const auto& r : Rp) minang[i] = std::min(minang[i], ang(S[i], r));
        while (static_cast<int>(Rp.size()) < NR) {
            int best = -1;
            for (std::size_t i = 0; i < S.size(); ++i)
                if (!used[i] && (best < 0 || minang[i] > minang[best])) best = static_cast<int>(i);
            if (best < 0) break;
            used[best] = 1; Rp.push_back(S[best]);
            for (std::size_t i = 0; i < S.size(); ++i)
                minang[i] = std::min(minang[i], ang(S[i], S[best]));
        }
        Ref_ = std::move(Rp);
    }

    // ── helpers ─────────────────────────────────────────────────────────────
    static double cosang(const std::vector<double>& a, const std::vector<double>& b) {
        double d = 0, na = 0, nb = 0;
        for (std::size_t i = 0; i < a.size(); ++i) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        double q = std::sqrt(na) * std::sqrt(nb);
        if (q < 1e-300) return 1.0;
        return std::clamp(d / q, -1.0, 1.0);
    }
    static double norm2(const std::vector<double>& a) {
        double s = 0; for (double v : a) s += v * v; return std::sqrt(s);
    }

    bool dominates(const std::vector<double>& a, const std::vector<double>& b) const {
        return detail::pareto_dominates(a, b);
    }
    // Constrained form used by the Alg.3 non-dominated sort.
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }

    // Normalisation z / z^nad over a pool (Eq.5). Returns normalised objs.
    std::vector<std::vector<double>> normalize(const std::vector<Sol>& pool) const {
        std::vector<double> z(m_, std::numeric_limits<double>::max());
        std::vector<double> zn(m_, -std::numeric_limits<double>::max());
        for (const auto& s : pool)
            for (int k = 0; k < m_; ++k) { z[k] = std::min(z[k], s.objs[k]); zn[k] = std::max(zn[k], s.objs[k]); }
        std::vector<std::vector<double>> F(pool.size(), std::vector<double>(m_));
        for (std::size_t i = 0; i < pool.size(); ++i)
            for (int k = 0; k < m_; ++k) {
                double d = zn[k] - z[k]; if (d < 1e-12) d = 1e-12;
                F[i][k] = (pool[i].objs[k] - z[k]) / d;
            }
        return F;
    }

    // S_HV (§2.3-B, Eq.7, Fig.8, [43]): for each objective k, take the nearest
    // larger neighbour along axis k as the reference point, then form the BOX
    // VOLUME between the candidate and that reference = product of the gaps
    // (ref_d - F[idx][d]) over ALL objectives d. The S_HV is the MIN volume
    // over k. A larger S_HV means better uniformity (Eq.7).
    double s_hv(int idx, const std::vector<std::vector<double>>& F) const {
        double min_vol = std::numeric_limits<double>::infinity();
        for (int k = 0; k < m_; ++k) {
            // reference point for objective k: nearest larger neighbour along
            // axis k supplies its full coordinate vector as the reference.
            double up = std::numeric_limits<double>::infinity();
            int ref = -1;
            for (std::size_t j = 0; j < F.size(); ++j) {
                if ((int)j == idx) continue;
                if (F[j][k] > F[idx][k] && F[j][k] < up) { up = F[j][k]; ref = (int)j; }
            }
            // box volume = product over all objectives of (ref_d - F[idx][d]),
            // guarded so each edge is strictly positive (numerically safe).
            double vol = 1.0;
            for (int d = 0; d < m_; ++d) {
                double rd = (ref >= 0) ? F[ref][d] : 1.0;  // boundary: upper bound
                double gap = rd - F[idx][d];
                if (gap < 1e-12) gap = 1e-12;              // guard zero/negative
                vol *= gap;
            }
            if (vol < min_vol) min_vol = vol;
        }
        if (!std::isfinite(min_vol)) min_vol = 0.0;
        return min_vol;
    }

    // fitness(j) = V(j)·cosθ_ij / ‖f'_j‖ (Eq.7). cosθ = max cos to any RefPoint.
    std::vector<double> fitness_all(const std::vector<std::vector<double>>& F) const {
        std::vector<double> fit(F.size(), 0.0);
        for (std::size_t j = 0; j < F.size(); ++j) {
            double cmax = -1.0;
            for (const auto& r : Ref_) { double c = cosang(F[j], r); if (c > cmax) cmax = c; }
            if (cmax < 0) cmax = 0;
            double nf = norm2(F[j]); if (nf < 1e-12) nf = 1e-12;
            double V = s_hv((int)j, F);
            fit[j] = V * cmax / nf;
        }
        return fit;
    }

    // Indices of "optimally distributed individuals": for each RefPoint, the
    // member of `idxs` closest to its VERTEX by Euclidean distance (§2.4).
    // The paper (§2.4, the IGD concept) defines an optimally distributed
    // individual as the one closest to the REFERENCE POINT ("the vertex of the
    // reference vector") by EUCLIDEAN distance. Selecting by angle (max cos)
    // is a different criterion: a distant individual lying exactly along the ray
    // wins on angle while losing on Euclidean distance.
    // The vertex is a point on the simplex (Fig.3(c)); Ref_ is stored
    // L2-normalized (for cosθ, Eq.6), so each reference point is mapped back
    // onto the simplex by dividing by the sum of its coordinates — for the
    // solution directions coming from adapt_refpoints, that is their
    // intersection with the simplex.
    std::vector<int> optimally_distributed(const std::vector<int>& idxs,
                                           const std::vector<std::vector<double>>& F) const {
        std::vector<int> chosen;
        std::vector<char> picked(F.size(), 0);
        std::vector<double> vert(m_);
        for (const auto& r : Ref_) {
            double sr = 0.0; for (double c : r) sr += c;
            for (int k = 0; k < m_; ++k) vert[k] = (sr > 1e-12) ? r[k] / sr : r[k];
            int best = -1; double bd = std::numeric_limits<double>::max();
            for (int i : idxs) {
                double d2 = 0.0;
                for (int k = 0; k < m_; ++k) { double t = F[i][k] - vert[k]; d2 += t * t; }
                if (d2 < bd) { bd = d2; best = i; }
            }
            if (best >= 0 && !picked[best]) { picked[best] = 1; chosen.push_back(best); }
        }
        return chosen;
    }

    // ── breeding ────────────────────────────────────────────────────────────
    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch) {
        const auto& b = vault.get_bounds(); int nv = vault.vars_n();
        std::vector<double> c1, c2;
        ops::sbx(x.vars, y.vars, c1, c2, b, eta_c_, pc_, rng_);
        ops::polynomial_mutation(c1, b, eta_m_, pm_eff(nv), rng_);
        Sol z; z.vars = c1;
        vault.set_variables(scratch, c1); vault.refresh_objectives(scratch);
        z.objs = vault.objectives_of(scratch);
        if (constraint_mode != ConstraintMode::NONE) z.cv = vault.get_cv(scratch);
        return z;
    }

    // ── mating pool selection (Alg.2): binary tournament on fitness ─────────
    std::vector<Sol> mating_pool(const std::vector<Sol>& ASS) {
        auto F = normalize(ASS);
        auto fit = fitness_all(F);
        std::vector<Sol> Pp;
        std::uniform_int_distribution<int> di(0, (int)ASS.size() - 1);
        for (int j = 0; j < N_; ++j) {
            int p = di(rng_), q = di(rng_);
            Pp.push_back(fit[p] > fit[q] ? ASS[p] : ASS[q]);
        }
        return Pp;
    }

    // ── environmental selection (Alg.3) ─────────────────────────────────────
    std::vector<Sol> environmental(const std::vector<Sol>& CA, int N) {
        int n = (int)CA.size();
        // fast non-dominated sort
        std::vector<int> ndom(n, 0);
        std::vector<std::vector<int>> doms(n);
        std::vector<int> rank(n, -1);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (dominates(CA[i], CA[j])) doms[i].push_back(j);
                else if (dominates(CA[j], CA[i])) ndom[i]++;
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> cur;
        for (int i = 0; i < n; ++i) if (ndom[i] == 0) { rank[i] = 0; cur.push_back(i); }
        int r = 0;
        while (!cur.empty()) {
            fronts.push_back(cur);
            std::vector<int> nxt;
            for (int i : cur) for (int j : doms[i]) if (--ndom[j] == 0) { rank[j] = r + 1; nxt.push_back(j); }
            cur = nxt; ++r;
        }

        std::vector<int> Q;
        std::size_t fi = 0;
        for (; fi < fronts.size(); ++fi) {
            if ((int)(Q.size() + fronts[fi].size()) > N) break;
            for (int i : fronts[fi]) Q.push_back(i);
        }
        if ((int)Q.size() < N && fi < fronts.size()) {
            int need = N - (int)Q.size();
            const std::vector<int>& Fk = fronts[fi];

            // normalise whole CA for angle / fitness in critical layer
            auto Fnorm = normalize(CA);

            // §2.4: adapt the reference points to the current critical layer
            // (the AR-MOEA port), then pick the optimally distributed ones;
            // afterwards restore the static base set for everything else.
            adapt_refpoints(Fnorm, Fk);
            auto od = optimally_distributed(Fk, Fnorm);
            Ref_ = Ref0_;
            std::vector<char> inFk(n, 0); for (int i : Fk) inFk[i] = 1;

            if ((int)od.size() >= need) {
                // pick `need` with smaller Euclidean distance to ideal (‖f'‖)
                std::sort(od.begin(), od.end(), [&](int a, int b) {
                    return norm2(Fnorm[a]) < norm2(Fnorm[b]);
                });
                for (int t = 0; t < need; ++t) Q.push_back(od[t]);
            } else {
                std::vector<char> taken(n, 0);
                for (int i : od) { Q.push_back(i); taken[i] = 1; }
                int still = need - (int)od.size();
                // remaining Fk by descending fitness (Eq.7)
                auto fit = fitness_all(Fnorm);
                std::vector<int> rest;
                for (int i : Fk) if (!taken[i]) rest.push_back(i);
                std::sort(rest.begin(), rest.end(), [&](int a, int b) { return fit[a] > fit[b]; });
                for (int t = 0; t < still && t < (int)rest.size(); ++t) Q.push_back(rest[t]);
            }
        }
        std::vector<Sol> out;
        for (int i : Q) out.push_back(CA[i]);
        return out;
    }

    void read_pop(DataVault<Ind_t>& vault, std::vector<Sol>& P) {
        P.clear();
        for (std::size_t i = 0; i < vault.active_n(); ++i) {
            Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
            if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
            P.push_back(s);
        }
    }
    void write_pop(DataVault<Ind_t>& vault, const std::vector<Sol>& P) {
        vault.reduce(0); vault.expand((int)P.size());
        for (int i = 0; i < (int)P.size(); ++i)
            vault.seed_individual((std::size_t)i, P[i].vars, P[i].objs, {}, {});
    }

public:
    void setup(DataVault<Ind_t>& vault) {
        m_ = vault.objs_n(); N_ = vault.pop_size();
        build_refpoints();
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
        read_pop(vault, P_);
        AS_.clear();
        t_ = 0;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        m_ = vault.objs_n(); N_ = vault.pop_size();
        build_refpoints();
        read_pop(vault, P_);
        AS_.clear();
        t_ = 0;
    }

    void step(DataVault<Ind_t>& vault) {
        int scratch = vault.expand(1);

        // ASS = [P, AS]  (Alg.1 line 8)
        std::vector<Sol> ASS = P_;
        ASS.insert(ASS.end(), AS_.begin(), AS_.end());

        // P' = MatingPoolSelection(ASS) (Alg.2)
        std::vector<Sol> Pp = mating_pool(ASS);

        // O = Variation(P') — one child per consecutive pair (IF-MaOEA-4)
        std::vector<Sol> O;
        for (int i = 0; i < (int)Pp.size(); ++i) {
            const Sol& a = Pp[i];
            const Sol& b = Pp[(i + 1) % Pp.size()];
            O.push_back(breed(a, b, vault, scratch));
        }

        // CA = P ∪ O
        std::vector<Sol> CA = P_;
        CA.insert(CA.end(), O.begin(), O.end());

        // P = EnvironmentalSelection(CA) (Alg.3)
        P_ = environmental(CA, N_);

        // AS = ND(AS, P) — same sieve (IF-MaOEA-6)
        std::vector<Sol> arch = AS_;
        arch.insert(arch.end(), P_.begin(), P_.end());
        AS_ = environmental(arch, N_);

        // expose current P to the vault
        write_pop(vault, P_);

        ++t_;
    }
};

} // namespace mootation

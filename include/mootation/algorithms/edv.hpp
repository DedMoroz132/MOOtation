#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// EDV — A many-objective evolutionary algorithm with Epsilon-indicator
//       Direction Vector
// Y. Yang, J. Luo, L. Huang, et al. — Applied Soft Computing 76, 2019, 326-355
// doi:10.1016/j.asoc.2018.11.041
//
// Generational scheme (Alg.1):
//   1. Alg.3: N offspring; parents come from a binary tournament over the
//      archive EA (line 4); SBX (p_c=1, eta_c=20) + polynomial mutation
//      (p_m=1/n, eta_m=20). The offspring y lies in Omega by construction,
//      since the bounded operators clamp: "F(y) in Omega" in Alg.3 line 7 is a
//      typo in the paper for y in Omega, and a separate check would be dead
//      code. The ideal point z* is updated by the offspring (line 9,
//      monotonically).
//   2. Alg.4 (Update EA): S = the non-dominated members of EA u Y;
//      if |S| <= 2N then EA = S (the EA size is variable, up to 2N);
//      otherwise: Step 1 — for each of the N direction vectors d_i (a two-layer
//      Das-Dennis lattice, Alg.2), the solution of S closest by angle is moved
//      into EA if that angle is below a_i, the minimum angle from d_i to its
//      neighbours; Step 2 — exactly |S|−2N of the remaining S are removed, the
//      worst by the eps+ fitness of Eq.4,
//      F(x) = Σ −exp(−I_{eps+}(y,x)/(c·v)) with c = max|I_{eps+}|, using the
//      incremental update of Eq.5; the rest moves into EA, giving |EA| = 2N.
//
//   I_{eps+}(a,b) = max_i (f_i(a) − f_i(b)) (Eq.2); angles are measured from
//   f − z* (an assumption: the paper does not specify the role of z* in the
//   angle); arccos is used instead of the paper's sin theta, which is
//   monotonically equivalent on [0, pi/2].
//
// PAPER DEFAULTS (§4.2.4): v (kappa) = 0.05, p_c=1, p_m=1/n, eta=20.
//   NOTE the paper is internally inconsistent on v: §2.2 gives v=0.01 while
//   §4.2.4 gives v=0.05. The experimental value 0.05 of §4.2.4 is used.
//   It is NOT inconsistent on p_m — see EDV-1.
// ASSUMPTIONS: Alg.3 line 4 is literally "Pa = binary tournament(EA)" — the
//   paper never says what the tournament compares. This port compares the eps+
//   fitness of Eq.4 as computed inside EA, larger being better, which is the
//   only ranking the algorithm already maintains over EA.
// DECLARED DEVIATIONS:
//   EDV-1 (DEVIATION). p_m = 1/n_vars, not the paper's literal 1/pop_size.
//     §4.2.4 is explicit and self-consistent here — "the mutation probability
//     (p_m) is set to 1/pops. The pops is equal to the pop_size of the
//     algorithm" — so this is a deliberate departure from the text, not a
//     resolved ambiguity: it is read as a slip against the Deb polynomial-
//     mutation convention the paper itself cites. The gap is large, not
//     cosmetic: at the paper's own m=3 setting (pop_size 300, DTLZ n=7) it is
//     1/300 against 1/7. Compare sms_m2m.hpp (SMSM2M-2), which meets the same
//     phrasing in a different paper and implements it literally.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY —
//   CDP in the dominance relation; binary variables.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../detail/math_compat.hpp"
#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../individuals.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

struct EDV_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class EDVCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double KAPPA_ = 0.05;   // v, §4.2.4 (vs 0.01 in §2.2 — see header)

    double       eta_c_ = 20.0;
    double       eta_m_ = 20.0;
    double       pc_    = 1.0;
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> D_;     // direction vectors (N of them)
    std::vector<double>              a_;     // a_i = min angle from d_i to its neighbours
    std::vector<double>              ideal_; // z*, the monotone ideal point

    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return ca < cb;
        }
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        bool strict = false;
        for (std::size_t k = 0; k < fa.size(); ++k) {
            if (fa[k] > fb[k]) return false;
            if (fa[k] < fb[k]) strict = true;
        }
        return strict;
    }

    static double vnorm(const std::vector<double>& f) {
        double s = 0.0; for (double x : f) s += x * x; return std::sqrt(s);
    }

    static double angle(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        double denom = vnorm(a) * vnorm(b);
        if (denom < 1e-30) return M_PI / 2.0;
        double c = dot / denom;
        c = std::max(-1.0, std::min(1.0, c));
        return std::acos(c);
    }

    static double eps_plus(const std::vector<double>& a, const std::vector<double>& b) {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; if (d > w) w = d; }
        return w;
    }

    void compute_min_angles() {
        int H = static_cast<int>(D_.size());
        a_.assign(H, M_PI / 2.0);
        for (int i = 0; i < H; ++i) {
            double best = M_PI;
            for (int j = 0; j < H; ++j) {
                if (i == j) continue;
                best = std::min(best, angle(D_[i], D_[j]));
            }
            a_[i] = (best < M_PI) ? best : (M_PI / 2.0);
        }
    }

    void update_ideal(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j)
            ideal_[j] = std::min(ideal_[j], f[j]);
    }

    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool_size) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool_size), at_pos(pool_size);
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);
        for (int i = 0; i < n; ++i) {
            int want = survivors[i], cur = pos[want];
            if (cur == i) continue;
            int other = at_pos[i];
            vault.swap_active(i, cur);
            pos[want] = i; pos[other] = cur;
            at_pos[i] = want; at_pos[cur] = other;
        }
        vault.reduce(n);
    }

public:
    EDVCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        D_ = das_dennis::generate_exact(m, n);   // Alg.2: two-layer lattice
        compute_min_angles();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0, 1.0);
        std::uniform_int_distribution<int>     db(0, 1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first.value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dr(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = db(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables(i, vars);
        }
        vault.sync();
        // Alg.2 (auxiliary step): z* is initialized from EA.
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        if (D_.empty()) { D_ = das_dennis::generate_exact(m, n); compute_min_angles(); }
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));
    }

    void step(DataVault<Ind_t>& vault) {
        int N = vault.pop_size();
        int n_ea = vault.parents_n();            // |EA| ∈ [1, 2N]
        int m = vault.objs_n();
        const auto& bounds = vault.get_bounds();

        // ── Eq.4 fitness inside EA — the binary tournament criterion ───────
        std::vector<double> fit(n_ea, 0.0);
        {
            double c = 1e-12;
            for (int i = 0; i < n_ea; ++i)
                for (int j = 0; j < n_ea; ++j)
                    if (i != j)
                        c = std::max(c, std::abs(eps_plus(vault.objectives_of(i),
                                                          vault.objectives_of(j))));
            for (int i = 0; i < n_ea; ++i) {
                double s = 0.0;
                for (int j = 0; j < n_ea; ++j)
                    if (i != j)
                        s += -std::exp(-eps_plus(vault.objectives_of(j),
                                                 vault.objectives_of(i)) / (c * KAPPA_));
                fit[i] = s;
            }
        }
        std::uniform_int_distribution<int> dist_ea(0, n_ea - 1);
        auto tournament = [&]() {
            int a = dist_ea(rng_), b = dist_ea(rng_);
            return (fit[a] >= fit[b]) ? a : b;
        };

        // ── Alg.3: N offspring, binary tournament(EA) + SBX + PM ───────────
        int off_base = vault.expand(N);
        int nv = vault.vars_n();
        double pm = (nv > 0) ? 1.0 / nv : 0.0;
        std::vector<double> pv1(nv), pv2(nv), c1, c2;
        for (int i = 0; i < N; i += 2) {
            int p1 = tournament(), p2 = tournament();
            for (int j = 0; j < nv; ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()), bc1, bc2;
                for (int j = 0; j < vault.bin_vars_n(); ++j) {
                    bv1[j] = vault.get_bin_variable(p1, j);
                    bv2[j] = vault.get_bin_variable(p2, j);
                }
                ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i + 1 < N) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < N) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < N) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();

        int pool = n_ea + N;

        // ── Alg.3 line 9: monotone update of z* from the offspring ─────────
        for (int i = off_base; i < pool; ++i)
            update_ideal(vault.objectives_of(i));

        // shifted objectives f' = f − z* (for the angles; eps+ is
        // translation-invariant)
        std::vector<std::vector<double>> Fp(pool, std::vector<double>(m, 0.0));
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) Fp[i][j] = o[j] - ideal_[j];
        }

        // ── Alg.4 line 2: the non-dominated set S from EA u Y ──────────────
        std::vector<int> S;
        for (int i = 0; i < pool; ++i) {
            bool dom = false;
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                if (dominates(vault, j, i)) { dom = true; break; }
            }
            if (!dom) S.push_back(i);
        }
        int Np = static_cast<int>(S.size());   // N' = size(S)

        std::vector<int> survivors;

        if (Np <= 2 * N) {
            // Alg.4 lines 4–5: EA = S. The size is variable, dominated
            // solutions are always discarded, and there is no top-up.
            survivors = S;
        } else {
            // Step 1 (lines 7–12): over the direction vectors.
            std::vector<char> inS(pool, 0);
            for (int v : S) inS[v] = 1;
            std::vector<int> retained;
            int H = static_cast<int>(D_.size());
            for (int i = 0; i < H; ++i) {
                int best = -1; double bd = std::numeric_limits<double>::max();
                for (int v : S) {
                    if (!inS[v]) continue;
                    double ang = angle(Fp[v], D_[i]);
                    if (ang < bd) { bd = ang; best = v; }
                }
                if (best >= 0 && bd < a_[i]) { retained.push_back(best); inS[best] = 0; }
            }
            std::vector<int> rem;
            for (int v : S) if (inS[v]) rem.push_back(v);

            // Step 2 (lines 13–26): remove exactly N' − 2N of the worst by
            // Eq.4 from the remainder of S, with the incremental Eq.5 update.
            int need_remove = Np - 2 * N;
            std::vector<char> alive(pool, 0);
            for (int v : rem) alive[v] = 1;
            if (need_remove > 0 && !rem.empty()) {
                double c = 1e-12;
                for (int i : rem) for (int j : rem) if (i != j)
                    c = std::max(c, std::abs(eps_plus(Fp[i], Fp[j])));
                std::vector<double> F(pool, 0.0);
                for (int i : rem) {
                    double s = 0;
                    for (int j : rem) if (i != j)
                        s += -std::exp(-eps_plus(Fp[j], Fp[i]) / (c * KAPPA_));
                    F[i] = s;
                }
                for (int t = 0; t < need_remove; ++t) {
                    int worst = -1; double wf = std::numeric_limits<double>::max();
                    for (int v : rem) if (alive[v] && F[v] < wf) { wf = F[v]; worst = v; }
                    if (worst < 0) break;
                    alive[worst] = 0;
                    for (int v : rem) if (alive[v] && v != worst)
                        F[v] += std::exp(-eps_plus(Fp[worst], Fp[v]) / (c * KAPPA_));
                }
            }
            survivors = retained;
            for (int v : rem) if (alive[v]) survivors.push_back(v);
        }

        // guard: EA cannot be empty — at least one non-dominated solution exists.
        if (survivors.empty()) survivors.push_back(0);

        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation

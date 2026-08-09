#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// VaEA — A Vector Angle-Based Evolutionary Algorithm for Unconstrained
//        Many-Objective Optimization
// Y. Xiang, Y. Zhou, M. Li, Z. Chen — IEEE TEVC 21(1), 2017
// doi:10.1109/TEVC.2016.2587808
//
// Generational scheme (Alg.1-2):
//   1. Q: parents drawn at random from P (§II-A; Alg.1 line 4 nominally calls
//      Mating_selection, but §II-D states "we do not use any special
//      reproduction operations"), SBX (eta_c=30, pc=1) +
//      PM (eta_m=20, pm=1/n)
//   2. S = P ∪ Q (2N): normalization (Alg.3) f' = (f−z^min)/(z^max−z^min),
//      fit = Σf' (Eq.3, smaller is better); NDS; whole fronts are accepted
//   3. Association (Alg.4): when P = ∅, first the m EXTREME solutions
//      (Definition 3: minimum angle to the unit vectors (1,0,…),…,(0,…,1)),
//      then the m best by fit; thereafter θ and γ are the minimum angle and
//      the target neighbour in P for each x in F_l
//   4. Niching (Alg.5): maximum-vector-angle-first (Alg.6: add
//      x_ρ = argmax θ, update θ/γ) plus worse-elimination (Alg.7: x_μ =
//      argmin θ; if θ(x_μ) < σ = (π/2)/(N+1), x_μ was not added and
//      fit(y_r) > fit(x_μ), then replace y_r <- x_μ; θ update: scenario 1
//      (γ(x_j) != γ(x_μ)) applies when angle < θ; scenario 2
//      (γ(x_j) = γ(x_μ)) sets θ(x_j) = angle UNCONDITIONALLY, lines 14-15)
//
// PAPER DEFAULTS (§III-C-3): pc=1.0, pm=1/n, eta_c=30, eta_m=20.
// DECLARED DEVIATIONS (all MINOR, deliberate):
//   - Definition 1/3: on multiple minima the first is taken, not a random one;
//   - a defensive top-up of P from the remaining pool in degenerate cases
//     (line 1 of Alg.6 in the paper simply returns P);
//   - angle() = π/2 at zero norm, i.e. an individual sitting at the ideal
//     point — the paper leaves this case undefined.
// EXTENSIONS BEYOND THE PAPER: constraint_mode FEASIBILITY (CDP; off by
//   default, as the paper is unconstrained); mixed real+binary genome.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../detail/math_compat.hpp"
#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../individuals.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

struct VaEA_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class VaEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 30.0;  // §III-C-3: «its distribution index is η_c = 30»
    double       eta_m_ = 20.0;  // §III-C-3
    double       pc_    = 1.0;   // §III-C-3: «crossover probability … set to 1.0»
    std::mt19937 rng_{std::random_device{}()};

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

    std::vector<std::vector<int>> fast_nds(DataVault<Ind_t>& vault, int pool) {
        std::vector<std::vector<int>> S(pool);
        std::vector<int> ndom(pool, 0), cur;
        for (int i = 0; i < pool; ++i) {
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                if (dominates(vault, i, j)) S[i].push_back(j);
                else if (dominates(vault, j, i)) ++ndom[i];
            }
            if (ndom[i] == 0) cur.push_back(i);
        }
        std::vector<std::vector<int>> fronts;
        while (!cur.empty()) {
            fronts.push_back(cur);
            std::vector<int> next;
            for (int i : cur)
                for (int j : S[i])
                    if (--ndom[j] == 0) next.push_back(j);
            cur = next;
        }
        return fronts;
    }

    static double vnorm(const std::vector<double>& f) {
        double s = 0.0; for (double x : f) s += x * x; return std::sqrt(s);
    }

    // angle(x,y) ∈ [0, π/2]
    static double angle(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        double denom = vnorm(a) * vnorm(b);
        if (denom < 1e-30) return M_PI / 2.0;
        double c = dot / denom;
        c = std::max(-1.0, std::min(1.0, c));
        return std::acos(c);
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
    VaEACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
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
    }

    void setup_seeded(DataVault<Ind_t>&) {}

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n, m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_N(0, n - 1);

        // ── Mating (random parents) + Variation -> offspring [N, 2N) ───────
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = dist_N(rng_), p2 = dist_N(rng_);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            // §III-C-3: pc=1.0, pm=1/n
            double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
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
                if (i + 1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();

        int pool = n * 2;

        // ── Normalization (Alg.3): f' and fit = Σf' over the union S ───────
        std::vector<double> zmin(m,  std::numeric_limits<double>::max());
        std::vector<double> zmax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                zmin[j] = std::min(zmin[j], o[j]);
                zmax[j] = std::max(zmax[j], o[j]);
            }
        }
        std::vector<std::vector<double>> Fn(pool, std::vector<double>(m, 0.0));
        std::vector<double> fit(pool, 0.0);
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            double s = 0.0;
            for (int j = 0; j < m; ++j) {
                double range = zmax[j] - zmin[j];
                Fn[i][j] = (range > 1e-14) ? (o[j] - zmin[j]) / range : 0.0;
                s += Fn[i][j];
            }
            fit[i] = s;
        }

        // ── NDS -> whole fronts, then the critical front Fl ────────────────
        auto fronts = fast_nds(vault, pool);
        std::vector<int> P;
        std::vector<int> Fl;
        for (const auto& fr : fronts) {
            if (static_cast<int>(P.size() + fr.size()) <= N) {
                for (int v : fr) P.push_back(v);
                if (static_cast<int>(P.size()) == N) break;
            } else {
                Fl = fr;
                break;
            }
        }

        // ── Niching (Alg.5): maximum-vector-angle-first + worse-elimination ─
        if (static_cast<int>(P.size()) < N && !Fl.empty()) {
            int T = static_cast<int>(Fl.size());
            std::vector<char>   flag(T, 0);
            std::vector<double> theta(T, std::numeric_limits<double>::max());
            std::vector<int>    gamma(T, -1);
            const double sigma = (M_PI / 2.0) / (N + 1);

            // bootstrap when P is empty (the first front exceeds N) —
            // Alg.4 lines 1-7: (1) the m EXTREME solutions (Definition 3:
            // minimum angle to the unit vectors (1,0,…,0)…(0,…,0,1));
            // (2) the m best by fit, after Sort(Fl) by fitness — without
            // duplicates, and allowing for K < 2m.
            if (P.empty()) {
                // (1) the extreme solutions e_i
                std::vector<double> axis(m, 0.0);
                for (int i = 0; i < m && static_cast<int>(P.size()) < N; ++i) {
                    std::fill(axis.begin(), axis.end(), 0.0);
                    axis[i] = 1.0;
                    int best = -1;
                    double ba = std::numeric_limits<double>::max();
                    for (int j = 0; j < T; ++j) {
                        if (flag[j]) continue;
                        double a = angle(Fn[Fl[j]], axis);
                        if (a < ba) { ba = a; best = j; }
                    }
                    if (best >= 0) { P.push_back(Fl[best]); flag[best] = 1; }
                }
                // (2) the m best by convergence (fit, smaller is better)
                std::vector<int> ord;
                ord.reserve(T);
                for (int j = 0; j < T; ++j) if (!flag[j]) ord.push_back(j);
                std::sort(ord.begin(), ord.end(),
                          [&](int a, int b) { return fit[Fl[a]] < fit[Fl[b]]; });
                for (int i = 0; i < m && i < static_cast<int>(ord.size()) &&
                                static_cast<int>(P.size()) < N; ++i) {
                    P.push_back(Fl[ord[i]]); flag[ord[i]] = 1;
                }
            }
            // association
            for (int j = 0; j < T; ++j) {
                if (flag[j]) continue;
                for (int k = 0; k < static_cast<int>(P.size()); ++k) {
                    double a = angle(Fn[Fl[j]], Fn[P[k]]);
                    if (a < theta[j]) { theta[j] = a; gamma[j] = k; }
                }
            }

            while (static_cast<int>(P.size()) < N) {
                // ρ = argmax θ among the unselected
                int rho = -1; double bestmax = -1.0;
                for (int j = 0; j < T; ++j)
                    if (!flag[j] && theta[j] > bestmax) { bestmax = theta[j]; rho = j; }
                // μ = argmin θ among the unselected
                int mu = -1; double bestmin = std::numeric_limits<double>::max();
                for (int j = 0; j < T; ++j)
                    if (!flag[j] && theta[j] < bestmin) { bestmin = theta[j]; mu = j; }
                if (rho < 0) break;

                // maximum-vector-angle-first (Alg.6): add x_ρ
                P.push_back(Fl[rho]); flag[rho] = 1;
                int newidx = static_cast<int>(P.size()) - 1;
                for (int j = 0; j < T; ++j) {
                    if (flag[j]) continue;
                    double a = angle(Fn[Fl[j]], Fn[Fl[rho]]);
                    if (a < theta[j]) { theta[j] = a; gamma[j] = newidx; }
                }

                // worse-elimination (Alg.7): conditional replacement y_r <- x_μ
                if (mu >= 0 && !flag[mu] && theta[mu] < sigma) {
                    int r = gamma[mu];
                    if (r >= 0 && r < static_cast<int>(P.size()) &&
                        fit[P[r]] > fit[Fl[mu]]) {
                        P[r] = Fl[mu]; flag[mu] = 1;
                        for (int j = 0; j < T; ++j) {
                            if (flag[j]) continue;
                            double a = angle(Fn[Fl[j]], Fn[Fl[mu]]);
                            if (gamma[j] == r) {
                                // Scenario 2 (Alg.7 lines 14-15): x_j was
                                // associated with the replaced y_r, so θ(x_j)
                                // is updated UNCONDITIONALLY (γ stays r, which
                                // is now x_μ). Otherwise θ would hold the angle
                                // to an individual no longer in P.
                                theta[j] = a;
                            } else if (a < theta[j]) {
                                // Scenario 1 (lines 9-13).
                                theta[j] = a; gamma[j] = r;
                            }
                        }
                    }
                }
            }
        }

        // guard: if fewer than N for any reason, top up from the pool
        if (static_cast<int>(P.size()) < N) {
            std::vector<char> inP(pool, 0);
            for (int v : P) inP[v] = 1;
            for (int v = 0; v < pool && static_cast<int>(P.size()) < N; ++v)
                if (!inP[v]) { P.push_back(v); inP[v] = 1; }
        }
        if (static_cast<int>(P.size()) > N) P.resize(N);

        rearrange(vault, P, pool);
    }
};

} // namespace mootation

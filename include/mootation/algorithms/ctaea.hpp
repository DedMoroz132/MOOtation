#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"
#include "../random.hpp"

namespace mootation {

// ============================================================================
// C-TAEA — Two-Archive Evolutionary Algorithm for Constrained Multiobjective
// Optimization
// K. Li, R. Chen, G. Fu, X. Yao — IEEE TEVC 23(2):303-315, 2019
// doi:10.1109/TEVC.2018.2855411            (source: c-taea_li2019_arxiv)
// (task 7 of 2026-10-10, supplement 3, §4.3)
//
// Two archives of N each, N the number of weight vectors W (a Das-Dennis
// lattice, as NSGA-III's: pop_size must be one of its sizes):
//   CA (the population, the answer) — convergence and feasibility (§3.2,
//      Algorithm 2): of CA ∪ Q the feasible ones; too many — the
//      non-dominated levels, then from the most crowded subregion the one of
//      the closest pair worse by the Tchebycheff function; too few — the
//      infeasible ones by the levels of (CV, g_tch), the last level cut by CV.
//   DA (the vault's archive) — diversity, constraints left out (§3.3,
//      Algorithm 3): of DA ∪ Q, round itr = 1, 2, ... gives each subregion
//      that holds fewer than itr members of the CA the best non-dominated
//      member of DA ∪ Q there, by g_tch, until N are chosen.
// Mating (§3.4, Algorithms 4-5): rho_c, rho_d the shares of CA ∪ DA that the
// non-dominated members of the CA and of the DA make; the first parent is a
// binary tournament in the CA if rho_c > rho_d, else in the DA; the second in
// the CA with probability rho_c, else in the DA; the tournament takes a
// feasible one over an infeasible one, a dominating one among two feasible,
// at random otherwise. Subregions (Algorithm 1): a point goes to the weight
// whose line it is perpendicularly closest to, its objectives taken from the
// ideal point of the set at hand (CTAEA-2).
//
// READINGS (declared):
//   CTAEA-1. Algorithm 2's infeasible branch (lines 23-30) begins with
//     S <- S_c and loops while |S| < N (the printed |S_c| < N cannot end), and
//     line 29 cuts the level added last; Algorithm 3 loops while |S| < N and
//     line 11's w^c is Eq. 13's w^i. pymoo's port of the authors' code reads
//     them so.
//   CTAEA-2. Eq. 8 divides by z_nad - z*; the authors' code (as pymoo
//     carries it) only subtracts z*, and so does this: with the division, the
//     DA's far infeasible members squeeze the front onto a few weights. z* is
//     the least of each objective over the set at hand;
//     g_tch (Eq. 11) and Eq. 9's distances are taken from it. A zero weight
//     counts as 1e-4 (pymoo's ASF).
//   CTAEA-3. The supplement with the operators' settings is not at hand:
//     SBX (eta_c 30, pc 1) and polynomial mutation (eta_m 20, pm 1/n), and
//     one child of each pair, N pairs per generation — the authors' code as
//     pymoo carries it. The DA starts as
//     a copy of the first CA.
//   CTAEA-4. Feasibility is cv == 0 by the vault (so EPS_LEVEL relaxes it);
//     with constraint_mode NONE every point is feasible.
// ============================================================================

template <typename Ind_t>
class CTAEACore {
public:
    // NONE: the constraints are left out; every other mode: cv as given
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double eta_c_ = 30.0, pc_ = 1.0, eta_m_ = 20.0, pm_ = -1.0;   // CTAEA-3
    std::mt19937 rng_{std::random_device{}()};
    std::vector<std::vector<double>> W_;

    using Vec = std::vector<double>;

    // ── the points the procedures work on ────────────────────────────────
    struct Pt {
        Vec f;
        double cv;
        int src;          // >= 0: active slot; < 0: archive entry -(src + 1)
    };

    double cv_of(double cv) const { return constraint_mode == ConstraintMode::NONE ? 0.0 : cv; }

    static bool dominates(const Vec& a, const Vec& b)
    {
        bool better = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) better = true;
        }
        return better;
    }

    // non-dominated levels of the points `idx` under `key`
    template <typename Key>
    static std::vector<std::vector<int>> levels(const std::vector<int>& idx, Key key)
    {
        const std::size_t n = idx.size();
        std::vector<std::vector<int>> S(n);
        std::vector<int> count(n, 0);
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b) {
                if (dominates(key(idx[a]), key(idx[b])))      { S[a].push_back(static_cast<int>(b)); ++count[b]; }
                else if (dominates(key(idx[b]), key(idx[a]))) { S[b].push_back(static_cast<int>(a)); ++count[a]; }
            }
        std::vector<std::vector<int>> out;
        std::vector<int> cur;
        for (std::size_t a = 0; a < n; ++a) if (count[a] == 0) cur.push_back(static_cast<int>(a));
        while (!cur.empty()) {
            std::vector<int> next, front;
            for (int a : cur) {
                front.push_back(idx[static_cast<std::size_t>(a)]);
                for (int b : S[static_cast<std::size_t>(a)])
                    if (--count[static_cast<std::size_t>(b)] == 0) next.push_back(b);
            }
            std::stable_sort(next.begin(), next.end());
            out.push_back(std::move(front));
            cur = std::move(next);
        }
        return out;
    }

    // the objectives of the points `idx` taken from their ideal point z*
    // (CTAEA-2)
    static std::vector<Vec> from_ideal(const std::vector<Pt>& P, const std::vector<int>& idx)
    {
        std::vector<Vec> out(P.size());
        if (idx.empty()) return out;
        const std::size_t m = P[static_cast<std::size_t>(idx[0])].f.size();
        Vec lo(m, std::numeric_limits<double>::infinity());
        for (int i : idx)
            for (std::size_t j = 0; j < m; ++j)
                lo[j] = std::min(lo[j], P[static_cast<std::size_t>(i)].f[j]);
        for (int i : idx) {
            Vec& v = out[static_cast<std::size_t>(i)];
            v.resize(m);
            for (std::size_t j = 0; j < m; ++j) v[j] = P[static_cast<std::size_t>(i)].f[j] - lo[j];
        }
        return out;
    }

    // Algorithm 1: the subregion of every point of `idx` (into `region`)
    void associate(const std::vector<Vec>& F, const std::vector<int>& idx, std::vector<int>& region) const
    {
        for (int i : idx) {
            const Vec& x = F[static_cast<std::size_t>(i)];
            int best = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < W_.size(); ++k) {
                const Vec& w = W_[k];
                double ww = 0.0, wx = 0.0;
                for (std::size_t j = 0; j < w.size(); ++j) { ww += w[j] * w[j]; wx += w[j] * x[j]; }
                double d2 = 0.0;
                for (std::size_t j = 0; j < w.size(); ++j) {
                    const double e = x[j] - wx / ww * w[j];
                    d2 += e * e;
                }
                if (d2 < best_d) { best_d = d2; best = static_cast<int>(k); }
            }
            region[static_cast<std::size_t>(i)] = best;
        }
    }

    // Eq. 11 from z* (CTAEA-2)
    double g_tch(const Vec& f, int k) const
    {
        const Vec& w = W_[static_cast<std::size_t>(k)];
        double g = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < f.size(); ++j) g = std::max(g, std::abs(f[j]) / std::max(w[j], 1e-4));
        return g;
    }

    // Algorithm 2: the indices into P (CA ∪ Q) of the new CA
    std::vector<int> update_ca(const std::vector<Pt>& P, int N)
    {
        std::vector<int> Sc, SI;
        for (int i = 0; i < static_cast<int>(P.size()); ++i)
            (P[static_cast<std::size_t>(i)].cv <= 0.0 ? Sc : SI).push_back(i);
        if (static_cast<int>(Sc.size()) == N) return Sc;
        std::vector<int> region(P.size(), -1);
        if (static_cast<int>(Sc.size()) > N) {
            auto fronts = levels(Sc, [&](int i) -> const Vec& { return P[static_cast<std::size_t>(i)].f; });
            std::vector<int> S;
            for (const auto& front : fronts) {
                if (static_cast<int>(S.size()) >= N) break;
                S.insert(S.end(), front.begin(), front.end());
            }
            if (static_cast<int>(S.size()) == N) return S;
            const std::vector<Vec> F = from_ideal(P, S);
            associate(F, S, region);
            std::vector<int> alive = S;
            while (static_cast<int>(alive.size()) > N) {
                std::vector<int> count(W_.size(), 0);
                for (int i : alive) ++count[static_cast<std::size_t>(region[static_cast<std::size_t>(i)])];
                const int most = *std::max_element(count.begin(), count.end());
                std::vector<int> crowded;
                for (std::size_t k = 0; k < count.size(); ++k)
                    if (count[k] == most) crowded.push_back(static_cast<int>(k));
                rnd::uniform_int_distribution<std::size_t> pick(0, crowded.size() - 1);
                const int k = crowded[pick(rng_)];                         // ties at random
                std::vector<int> in;
                for (int i : alive) if (region[static_cast<std::size_t>(i)] == k) in.push_back(i);
                // Eq. 9: each one's distance to its nearest in the subregion;
                // S_t the ones at the least; x^w the worst of them by g_tch
                std::vector<double> dist(in.size(), std::numeric_limits<double>::infinity());
                for (std::size_t a = 0; a < in.size(); ++a)
                    for (std::size_t b = 0; b < in.size(); ++b) {
                        if (a == b) continue;
                        double d2 = 0.0;
                        const Vec& fa = F[static_cast<std::size_t>(in[a])];
                        const Vec& fb = F[static_cast<std::size_t>(in[b])];
                        for (std::size_t j = 0; j < fa.size(); ++j) d2 += (fa[j] - fb[j]) * (fa[j] - fb[j]);
                        dist[a] = std::min(dist[a], std::sqrt(d2));
                    }
                const double least = *std::min_element(dist.begin(), dist.end());
                int worst = -1;
                double worst_g = -std::numeric_limits<double>::infinity();
                for (std::size_t a = 0; a < in.size(); ++a) {
                    if (dist[a] != least) continue;
                    const double g = g_tch(F[static_cast<std::size_t>(in[a])], k);
                    if (g > worst_g) { worst_g = g; worst = in[a]; }
                }
                alive.erase(std::find(alive.begin(), alive.end(), worst));
            }
            return alive;
        }
        // too few feasible: the infeasible by the levels of (CV, g_tch), Eq. 12
        const std::vector<Vec> F = from_ideal(P, SI);
        associate(F, SI, region);
        std::vector<Vec> key(P.size());
        for (int i : SI)
            key[static_cast<std::size_t>(i)] = {P[static_cast<std::size_t>(i)].cv,
                                                g_tch(F[static_cast<std::size_t>(i)],
                                                      region[static_cast<std::size_t>(i)])};
        auto fronts = levels(SI, [&](int i) -> const Vec& { return key[static_cast<std::size_t>(i)]; });
        std::vector<int> S = Sc;                                       // CTAEA-1
        for (auto& front : fronts) {
            if (static_cast<int>(S.size()) >= N) break;
            if (static_cast<int>(S.size() + front.size()) > N) {       // cut the last level by CV
                std::stable_sort(front.begin(), front.end(), [&](int a, int b) {
                    return P[static_cast<std::size_t>(a)].cv < P[static_cast<std::size_t>(b)].cv;
                });
                front.resize(static_cast<std::size_t>(N) - S.size());
            }
            S.insert(S.end(), front.begin(), front.end());
        }
        return S;
    }

    // Algorithm 3: the indices into H (DA ∪ Q) of the new DA, given the new CA
    std::vector<int> update_da(const std::vector<Pt>& H, const std::vector<Pt>& CA, int N) const
    {
        std::vector<int> all_h(H.size()), all_c(CA.size());
        std::iota(all_h.begin(), all_h.end(), 0);
        std::iota(all_c.begin(), all_c.end(), 0);
        const std::vector<Vec> Fh = from_ideal(H, all_h), Fc = from_ideal(CA, all_c);
        std::vector<int> rh(H.size()), rc(CA.size());
        associate(Fh, all_h, rh);
        associate(Fc, all_c, rc);
        std::vector<int> in_ca(W_.size(), 0);
        for (int r : rc) ++in_ca[static_cast<std::size_t>(r)];
        std::vector<std::vector<int>> by_region(W_.size());
        for (int i : all_h) by_region[static_cast<std::size_t>(rh[static_cast<std::size_t>(i)])].push_back(i);
        std::vector<int> S;
        for (int itr = 1; static_cast<int>(S.size()) < N; ++itr) {
            for (std::size_t k = 0; k < W_.size() && static_cast<int>(S.size()) < N; ++k) {
                for (int t = in_ca[k]; t < itr && static_cast<int>(S.size()) < N; ++t) {
                    auto& cand = by_region[k];
                    if (cand.empty()) break;
                    // the best non-dominated one there, by g_tch (Eq. 13)
                    int best = -1;
                    double best_g = std::numeric_limits<double>::infinity();
                    for (int a : cand) {
                        bool dominated = false;
                        for (int b : cand)
                            if (b != a && dominates(H[static_cast<std::size_t>(b)].f, H[static_cast<std::size_t>(a)].f)) {
                                dominated = true;
                                break;
                            }
                        if (dominated) continue;
                        const double g = g_tch(Fh[static_cast<std::size_t>(a)], static_cast<int>(k));
                        if (g < best_g) { best_g = g; best = a; }
                    }
                    cand.erase(std::find(cand.begin(), cand.end(), best));
                    S.push_back(best);
                }
            }
        }
        return S;
    }

    // Algorithm 5 over the points `pool`
    const Pt& tournament(const std::vector<const Pt*>& pool)
    {
        rnd::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
        const std::size_t a = pick(rng_);
        std::size_t b = pick(rng_);
        if (pool.size() > 1) while (b == a) b = pick(rng_);
        const Pt& x = *pool[a];
        const Pt& y = *pool[b];
        const bool fx = x.cv <= 0.0, fy = y.cv <= 0.0;
        rnd::uniform_int_distribution<int> coin(0, 1);
        if (fx && fy) {
            if (dominates(x.f, y.f)) return x;
            if (dominates(y.f, x.f)) return y;
            return coin(rng_) ? x : y;
        }
        if (fx) return x;
        if (fy) return y;
        return coin(rng_) ? x : y;
    }

    Pt active_pt(DataVault<Ind_t>& vault, int v)
    {
        return Pt{vault.objectives_of(static_cast<std::size_t>(v)),
                  cv_of(vault.get_cv(static_cast<std::size_t>(v))), v};
    }
    Pt archive_pt(DataVault<Ind_t>& vault, std::size_t i)
    {
        return Pt{vault.archive_objectives_of(i), cv_of(vault.archive_cv(i)), -static_cast<int>(i) - 1};
    }

    void require(DataVault<Ind_t>& vault)
    {
        if (W_.empty()) W_ = das_dennis::generate_exact(vault.objs_n(), vault.pop_size());
    }

public:
    CTAEACore() = default;

    void set_seed(unsigned s) { rng_.seed(s); }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_pm(double p)            { pm_ = p; }

    void setup(DataVault<Ind_t>& vault)
    {
        require(vault);
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        rnd::uniform_real_distribution<double> U01(0.0, 1.0);
        Vec x(static_cast<std::size_t>(vault.vars_n()));
        std::vector<int> bx(static_cast<std::size_t>(vault.bin_vars_n()));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                const double lo = bounds[j].first.value_or(0.0);
                const double hi = bounds[j].second.value_or(1.0);
                x[static_cast<std::size_t>(j)] = lo + U01(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bx[static_cast<std::size_t>(j)] = ops::random_gene(j, rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(static_cast<std::size_t>(i), x, bx);
            else                        vault.set_variables(static_cast<std::size_t>(i), x);
        }
        vault.sync();
        setup_seeded(vault);
    }

    // the DA starts as a copy of the CA (CTAEA-3)
    void setup_seeded(DataVault<Ind_t>& vault)
    {
        require(vault);
        vault.archive_clear();
        for (int i = 0; i < vault.pop_size(); ++i) vault.archive_push(static_cast<std::size_t>(i));
    }

    void step(DataVault<Ind_t>& vault)
    {
        const int N = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        const int nv = vault.vars_n(), nb = vault.bin_vars_n();
        const double pm = pm_ >= 0.0 ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);

        // Algorithm 4: rho_c, rho_d over CA ∪ DA
        std::vector<Pt> ca, da;
        for (int i = 0; i < N; ++i) ca.push_back(active_pt(vault, i));
        for (std::size_t i = 0; i < vault.archive_size(); ++i) da.push_back(archive_pt(vault, i));
        std::vector<const Pt*> hm, pca, pda;
        for (const auto& p : ca) { hm.push_back(&p); pca.push_back(&p); }
        for (const auto& p : da) { hm.push_back(&p); pda.push_back(&p); }
        int nd_c = 0, nd_d = 0;
        for (std::size_t a = 0; a < hm.size(); ++a) {
            bool dominated = false;
            for (std::size_t b = 0; b < hm.size() && !dominated; ++b)
                dominated = b != a && dominates(hm[b]->f, hm[a]->f);
            if (!dominated) ++(a < ca.size() ? nd_c : nd_d);
        }
        const double rho_c = static_cast<double>(nd_c) / static_cast<double>(hm.size());
        const double rho_d = static_cast<double>(nd_d) / static_cast<double>(hm.size());

        auto vars_of = [&](const Pt& p) -> const Vec& {
            return p.src >= 0 ? vault.variables_of(static_cast<std::size_t>(p.src))
                              : vault.archive_variables_of(static_cast<std::size_t>(-p.src - 1));
        };
        auto bits_of = [&](const Pt& p) -> const std::vector<int>& {
            return p.src >= 0 ? vault.binary_variables_of(static_cast<std::size_t>(p.src))
                              : vault.archive_bin_variables_of(static_cast<std::size_t>(-p.src - 1));
        };

        // Q: N children, one of each pair (CTAEA-3)
        rnd::uniform_real_distribution<double> U01(0.0, 1.0);
        std::vector<Vec> qx;
        std::vector<std::vector<int>> qb;
        while (static_cast<int>(qx.size()) < N) {
            const Pt& p1 = tournament(rho_c > rho_d ? pca : pda);
            const Pt& p2 = tournament(U01(rng_) < rho_c ? pca : pda);
            Vec c1, c2;
            ops::sbx(vars_of(p1), vars_of(p2), c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            qx.push_back(std::move(c1));
            if (nb > 0) {
                std::vector<int> b1, b2;
                ops::binary_crossover(bits_of(p1), bits_of(p2), b1, b2, rng_);
                ops::bit_flip_mutation(b1, nb, rng_);
                qb.push_back(std::move(b1));
            }
        }
        const int base = vault.expand(N);
        for (int i = 0; i < N; ++i) {
            if (nb > 0) vault.set_all_variables(static_cast<std::size_t>(base + i), qx[static_cast<std::size_t>(i)],
                                                qb[static_cast<std::size_t>(i)]);
            else        vault.set_variables(static_cast<std::size_t>(base + i), qx[static_cast<std::size_t>(i)]);
        }
        vault.sync();

        // Algorithm 2 over CA ∪ Q (the active slots), Algorithm 3 over DA ∪ Q
        std::vector<Pt> hc;
        for (int i = 0; i < base + N; ++i) hc.push_back(active_pt(vault, i));
        const std::vector<int> keep_ca = update_ca(hc, N);
        std::vector<Pt> new_ca, hd = da;
        for (int i : keep_ca) new_ca.push_back(hc[static_cast<std::size_t>(i)]);
        for (int i = 0; i < N; ++i) hd.push_back(hc[static_cast<std::size_t>(base + i)]);
        const std::vector<int> keep_da = update_da(hd, new_ca, N);

        // the new DA: the kept entries stay, the chosen children are copied in
        std::vector<char> stay(da.size(), 0);
        std::vector<int> add;
        for (int i : keep_da) {
            if (static_cast<std::size_t>(i) < da.size()) stay[static_cast<std::size_t>(i)] = 1;
            else add.push_back(hd[static_cast<std::size_t>(i)].src);
        }
        for (std::size_t i = da.size(); i-- > 0;) if (!stay[i]) vault.archive_erase(i);
        for (int v : add) vault.archive_push(static_cast<std::size_t>(v));

        // the new CA into slots [0, N)
        std::vector<int> pos(static_cast<std::size_t>(base + N)), at(static_cast<std::size_t>(base + N));
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at.begin(), at.end(), 0);
        for (int i = 0; i < N; ++i) {
            const int want = keep_ca[static_cast<std::size_t>(i)];
            const int cur  = pos[static_cast<std::size_t>(want)];
            if (cur == i) continue;
            const int other = at[static_cast<std::size_t>(i)];
            vault.swap_active(static_cast<std::size_t>(i), static_cast<std::size_t>(cur));
            pos[static_cast<std::size_t>(want)]  = i;
            pos[static_cast<std::size_t>(other)] = cur;
            at[static_cast<std::size_t>(i)]      = want;
            at[static_cast<std::size_t>(cur)]    = other;
        }
        vault.reduce(N);
    }
};

}   // namespace mootation

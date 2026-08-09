#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// EMyO/C — Clustering-Based Selection for Evolutionary Many-Objective
//          Optimization
// R. Denysiuk, L. Costa, I. Espírito Santo — PPSN XIII, LNCS 8672, 2014,
//   538-547
// doi:10.1007/978-3-319-10762-2_53
//
// Generational scheme ((mu+lambda), §2):
//   1. Mating: the pool is the whole population — every member is selected.
//   2. Variation (each x yields one offspring x'): pick two distinct a, b from
//      the pool; v = a − b; apply polynomial mutation (eta_m=20, p_m=1/n) to
//      the difference vector v; clamp v_j into [−delta_j, delta_j] with
//      delta_j = (ub_j−lb_j)/2 (Eq.2–3); x'_j = x_j + v_j when rand < CR,
//      otherwise x_j (Eq.4); repair into [lb,ub] (Eq.5); if x' equals x, redraw
//      a and b until they differ.
//   3. Reference point z: initialized from the starting population; on every
//      offspring evaluation its components are lowered monotonically.
//   4. Environmental selection P u Q -> N: NDS with whole fronts accepted;
//      the critical front Fl is truncated to k as follows: Step 1
//      dref = ||f−z||_2 (before any transformation); Step 2 shift by f−z
//      (Eq.6); Step 3 project onto Sum f = 1 by dividing by the component sum
//      (Eq.7); Step 4 agglomerative clustering into k clusters with
//      AVERAGE-linkage (the mean of all pairwise distances, UPGMA);
//      Step 5 the cluster representative is argmin dref.
//
// PAPER DEFAULTS (§3.2): CR=0.15, eta_m=20, p_m=1/n.
// ASSUMPTIONS (only where the paper genuinely leaves a gap):
//   (1) v = a − b, with NO scaling factor F. §3.1 says only "a difference
//       vector v is calculated using these individuals" and defers the DE idea
//       to [2]; Eq.4 then adds v to the parent unscaled, so a factor would have
//       to come from nowhere.
//   (2) The BOX HANDED TO POLYNOMIAL MUTATION when it mutates v is
//       [−(ub−lb), +(ub−lb)] — the range a difference of two in-box variables
//       can actually take. The paper says PM is applied to v but never says in
//       what box, and PM's δ1/δ2 are position-dependent, so a box has to be
//       chosen. NOT to be confused with the Eq.2/Eq.3 clamp below, which the
//       paper DOES specify: after mutation v is clipped to [−δ_j, +δ_j] with
//       δ_j = (ub_j − lb_j)/2. Both are implemented; they are different steps.
// DECLARED DEVIATIONS: the "until different" retry is capped at 10 attempts,
//   as a guard against looping forever.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY —
//   CDP inside the non-dominated sort; binary variables (bit-flip).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../individuals.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"

namespace mootation {

struct EMyOC_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class EMyOCCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_m_ = 20.0;
    double       cr_    = 0.15;    // CR = 0.15 (§3.2)
    std::mt19937 rng_{std::random_device{}()};

    std::vector<double> ideal_;    // z, the historical (monotone) ideal point

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
            for (int i : cur) for (int j : S[i]) if (--ndom[j] == 0) next.push_back(j);
            cur = next;
        }
        return fronts;
    }

    // Average-linkage agglomeration of the projected Fl points into k clusters.
    // AVERAGE-LINKAGE (UPGMA via Lance-Williams)
    std::vector<std::vector<int>>
    agglomerate_average(const std::vector<int>& Fl,
                        const std::vector<std::vector<double>>& Proj, int k) {
        int n = static_cast<int>(Fl.size());
        std::vector<std::vector<int>> cl(n);
        for (int i = 0; i < n; ++i) cl[i] = {Fl[i]};
        if (n <= k) return cl;
        // matrix of mean pairwise distances between clusters (points at first)
        std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double s = 0.0;
                for (std::size_t t = 0; t < Proj[Fl[i]].size(); ++t) {
                    double d = Proj[Fl[i]][t] - Proj[Fl[j]][t]; s += d * d;
                }
                D[i][j] = D[j][i] = std::sqrt(s);
            }
        std::vector<int> alive(n, 1), sz(n, 1);
        int count = n;
        while (count > k) {
            int ba = -1, bb = -1; double bd = std::numeric_limits<double>::max();
            for (int a = 0; a < n; ++a) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; ++b) {
                    if (!alive[b]) continue;
                    if (D[a][b] < bd) { bd = D[a][b]; ba = a; bb = b; }
                }
            }
            if (ba < 0) break;
            // Lance-Williams for average-linkage (UPGMA):
            //   d(ba∪bb, c) = (|ba|·d(ba,c) + |bb|·d(bb,c)) / (|ba|+|bb|)
            int na = sz[ba], nb = sz[bb];
            for (int c = 0; c < n; ++c) {
                if (!alive[c] || c == ba || c == bb) continue;
                double nd = (na * D[ba][c] + nb * D[bb][c]) / (na + nb);
                D[ba][c] = D[c][ba] = nd;
            }
            cl[ba].insert(cl[ba].end(), cl[bb].begin(), cl[bb].end());
            sz[ba] = na + nb; alive[bb] = 0; --count;
        }
        std::vector<std::vector<int>> out;
        for (int i = 0; i < n; ++i) if (alive[i]) out.push_back(cl[i]);
        return out;
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

    void update_ideal(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j)
            ideal_[j] = std::min(ideal_[j], f[j]);
    }

public:
    EMyOCCore() = default;

    void set_eta_crossover(double)   {}             // EMyO/C does not use SBX
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_cr           (double c) { cr_ = c; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
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
        // z_j = min_{1<=i<=mu} f_j(x_i); afterwards only offspring update it.
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n, m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_N(0, n - 1);
        std::uniform_real_distribution<double> dr(0.0, 1.0);

        // ════ VARIATION: DE difference vector + PM (Eq.2-5) ════
        int off_base = vault.expand(n);
        int nv = vault.vars_n();
        double pm = (nv > 0) ? 1.0 / nv : 0.0;   // p_m = 1/n (§3.2)
        // PM is applied to the difference vector v, whose natural bounds are
        // [−(ub−lb), (ub−lb)] — the difference of two variables from [lb,ub];
        // v is then clamped to [−delta, delta] with delta = (ub−lb)/2.
        std::vector<std::pair<std::optional<double>, std::optional<double>>> dbounds(nv);
        for (int j = 0; j < nv; ++j) {
            double lo = bounds[j].first.value_or(0.0);
            double hi = bounds[j].second.value_or(1.0);
            dbounds[j] = std::make_pair(std::optional<double>(-(hi - lo)),
                                        std::optional<double>(hi - lo));
        }
        for (int i = 0; i < n; ++i) {
            std::vector<double> child(nv);
            bool differs = false;
            for (int attempt = 0; attempt < 10 && !differs; ++attempt) {
                int a = dist_N(rng_), b = dist_N(rng_);
                // FIX 2026-07-08:
                // The donor redraw is guarded by the population size n.
                // Previously: while(a==i); while(b==i||b==a) — at n<=2 there is
                // no third distinct individual, so the second loop could never
                // terminate and the process hung. The conditions now relax with
                // n: a != i is required only at n >= 2; b not in {i,a} only at
                // n >= 3, where three distinct individuals exist. At n >= 3 the
                // semantics are unchanged (a != i, b != i, b != a); at n = 2,
                // b == i is allowed; at n = 1, a == b == i gives a zero
                // difference vector, so the offspring equals the parent, and
                // the outer attempt < 10 cap bounds the loop.
                if (n >= 2) while (a == i) a = dist_N(rng_);
                if (n >= 3) while (b == i || b == a) b = dist_N(rng_);
                else if (n == 2) while (b == a) b = dist_N(rng_);
                std::vector<double> v(nv);
                for (int j = 0; j < nv; ++j) v[j] = vault.get_variable(a, j) - vault.get_variable(b, j);
                ops::polynomial_mutation(v, dbounds, eta_m_, pm, rng_);
                for (int j = 0; j < nv; ++j) {
                    double lo = bounds[j].first.value_or(0.0);
                    double hi = bounds[j].second.value_or(1.0);
                    double dj = (hi - lo) / 2.0;
                    if (v[j] < -dj) v[j] = -dj; else if (v[j] > dj) v[j] = dj;
                    double xj = vault.get_variable(i, j);
                    double cj = (dr(rng_) < cr_) ? xj + v[j] : xj;
                    cj = std::min(std::max(cj, lo), hi);
                    child[j] = cj;
                    if (std::abs(cj - xj) > 1e-15) differs = true;
                }
            }
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bc(vault.bin_vars_n());
                for (int j = 0; j < vault.bin_vars_n(); ++j) bc[j] = vault.get_bin_variable(i, j);
                ops::bit_flip_mutation(bc, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i, child, bc);
            } else {
                vault.set_variables(off_base + i, child);
            }
        }
        vault.sync();

        int pool = n * 2;
        // z is updated by the evaluated offspring, monotonically (§2).
        for (int i = off_base; i < pool; ++i)
            update_ideal(vault.objectives_of(i));
        const std::vector<double>& z = ideal_;

        // ════ ENV-SELECTION: non-dominated sort + accept whole fronts ════
        auto fronts = fast_nds(vault, pool);
        std::vector<int> survivors;
        std::vector<int> Fl;
        for (const auto& fr : fronts) {
            if (static_cast<int>(survivors.size() + fr.size()) <= N) {
                for (int v : fr) survivors.push_back(v);
                if (static_cast<int>(survivors.size()) == N) break;
            } else { Fl = fr; break; }
        }

        if (static_cast<int>(survivors.size()) < N && !Fl.empty()) {
            int kk = N - static_cast<int>(survivors.size());
            // ── Step 1: dref = ||f−z||; Step 2: shift f−z; Step 3: project /Sum f ──
            std::vector<double> dref(pool, 0.0);
            std::vector<std::vector<double>> Proj(pool, std::vector<double>(m, 0.0));
            for (int v : Fl) {
                const auto& o = vault.objectives_of(v);
                double s2 = 0.0, sum = 0.0;
                std::vector<double> tr(m);
                for (int j = 0; j < m; ++j) { tr[j] = o[j] - z[j]; s2 += tr[j]*tr[j]; sum += tr[j]; }
                dref[v] = std::sqrt(s2);
                for (int j = 0; j < m; ++j) Proj[v][j] = (sum > 1e-14) ? tr[j]/sum : 0.0;
            }
            // ── Step 4: average-linkage into k clusters; Step 5: representative = min dref ──
            auto clusters = agglomerate_average(Fl, Proj, kk);
            for (auto& c : clusters) {
                int best = c[0]; double bd = dref[c[0]];
                for (int v : c) if (dref[v] < bd) { bd = dref[v]; best = v; }
                survivors.push_back(best);
            }
        }

        if (static_cast<int>(survivors.size()) > N) survivors.resize(N);

        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation

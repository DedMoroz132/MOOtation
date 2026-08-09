#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// R2-IBEA — R2 Indicator Based Evolutionary Algorithm for Multiobjective
//           Optimization
// D. H. Phan, J. Suzuki — 2013 IEEE Congress on Evolutionary Computation
// (CEC), pp. 1836-1845.
// doi:10.1109/CEC.2013.6557783
// FIX 2026-08-09 (Crossref sweep): the DOI read 10.1109/CEC.2013.6557868,
//   which resolves to MOMBI (Hernandez Gomez & Coello Coello) — a different
//   paper from the same proceedings.
//
// Generational scheme (Algorithm 2):
//   1. Mating: binary tournament on the BINARY pairwise R2 indicator (§IV-C):
//      I_R2(a,b) and I_R2(b,a) (Eq. 4) are compared at the current z*; the
//      smaller own value wins; ties are broken at random.
//   2. Variation: SBX with P_c = 0.9 (Alg. 2 Line 8, Table I) and polynomial
//      mutation with P_m = 1/n per variable (Table I, jMetal semantics);
//      mu offspring are produced.
//   3. R_g = P_g ∪ O_g (2mu); z* is updated per Eq. 5 (Line 20).
//   4. Fitness F(x) = Σ_{y≠x} −e^{−I_R2(y,x)/κ} (Line 21).
//   5. Environmental selection: iteratively remove the worst, updating
//      F(x) += e^{−I_R2(x*,x)/κ} (Lines 22–26) until |R_g| = mu.
//
// Weight generation (Algorithm 1, offline in setup): tmax add+prune
// iterations; when |W| > |V| the vector with the smallest HV contribution is
// removed, I_HV(w) = HV(W) − HV(W\{w}), reference point (2,…,2)
// (Table I / Fig. 1).
// The paper does not prescribe how HV is computed; here:
//   m = 2 — exact contribution (staircase sort, O(K log K) per prune);
//   m = 3 — exact contribution (leave-one-out: contribution(w) =
//           Vol([w,ref]) − HV3D(clipped competitors), sweeping the third
//           coordinate);
//   m > 3 — Monte-Carlo with an adaptive sample count: exclusive-dominance
//           counters accumulate (base max(20000, 200·|W|) samples, doubling
//           while fewer than |W|/2 counters are non-zero, budget 16x the base);
//           the sampler is seeded from rng_, so set_seed keeps it reproducible.
// In every branch the argmin tie is broken at random.
//
// PAPER DEFAULTS (Table I; t_max = 10,000 comes from §IV-A / Fig. 1, not from
//   Table I): |V| = mu = pop_size, κ = 0.005, tmax = 10000,
// P_c = 0.9, P_m = 1/n, η_c = η_m = 20 (the jMetal defaults; Table I gives no
// η values).
// DECLARED DEVIATIONS: when random() > P_c the offspring are copies of the
// parents and are still ADDED to O_g (jMetal SBX semantics; the letter of
// Alg. 2 Lines 8–17 would re-draw the parents instead).
//   (MINOR) the binary tournament draws two DISTINCT individuals by rejection
//   sampling; §IV-C does not require distinctness. A harmless strengthening,
//   but not free: the rejection loop consumes extra RNG draws, so the stream
//   differs from a plain pair of uniform picks. Kept as is — removing the loop
//   would itself change the stream.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY
// (CDP preference in the tournament and a fitness penalty), binary variables
// (uniform crossover + bit-flip).
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class R2IBEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ---- hyperparameters (Table I) ----
    double       kappa_ = 0.005;   // indicator scaling factor
    int          tmax_  = 10000;   // weight vector generation iterations
    double       pc_    = 0.9;     // SBX crossover probability (Alg. 2 Line 8)
    double       eta_c_ = 20.0;
    double       eta_m_ = 20.0;
    std::mt19937 rng_{std::random_device{}()};

    // Weight vectors — generated once in setup(), shape [|V|][m].
    std::vector<std::vector<double>> weight_vectors_;

    // Adaptive reference point z* (Eq. 5). Persistent: tournaments of
    // generation g use the z* updated at Line 20 of generation g-1
    // (setup initialises it from P_0).
    std::vector<double> zstar_;

    static constexpr double WREF_ = 2.0;   // HV reference point (2,...,2), Table I

    // ------------------------------------------------------------------ //
    //  Weight pruning (Algorithm 1, Lines 9-11): exact HV contributions
    // ------------------------------------------------------------------ //

    // m = 2: exact exclusive areas via the sorted staircase, ref (2,2).
    // Weakly dominated / duplicate vectors get contribution 0.
    static void contrib_exact_2d(const std::vector<std::vector<double>>& W,
                                 std::vector<double>& contrib) {
        int K = static_cast<int>(W.size());
        contrib.assign(K, 0.0);
        std::vector<int> order(K);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (W[a][0] != W[b][0]) return W[a][0] < W[b][0];
            return W[a][1] < W[b][1];
        });
        std::vector<int> stair;                      // non-dominated staircase
        double best_y = std::numeric_limits<double>::max();
        for (int idx : order)
            if (W[idx][1] < best_y) { stair.push_back(idx); best_y = W[idx][1]; }
        int T = static_cast<int>(stair.size());
        for (int t = 0; t < T; ++t) {
            double x_next = (t + 1 < T) ? W[stair[t + 1]][0] : WREF_;
            double y_prev = (t > 0)     ? W[stair[t - 1]][1] : WREF_;
            contrib[stair[t]] = (x_next - W[stair[t]][0])
                              * (y_prev - W[stair[t]][1]);
        }
    }

    // Exact 3D hypervolume w.r.t. ref (2,2,2) — sweep along the 3rd coordinate
    // with a 2D non-dominated staircase (minimisation). Points must be <= ref.
    static double hv3d(std::vector<std::array<double, 3>>& pts) {
        if (pts.empty()) return 0.0;
        std::sort(pts.begin(), pts.end(),
                  [](const auto& a, const auto& b) { return a[2] < b[2]; });
        // staircase: sorted by x ascending, y strictly descending
        std::vector<std::pair<double, double>> st;
        auto area = [&st]() {
            double s = 0.0;
            for (std::size_t t = 0; t < st.size(); ++t) {
                double x_next = (t + 1 < st.size()) ? st[t + 1].first : WREF_;
                s += (x_next - st[t].first) * (WREF_ - st[t].second);
            }
            return s;
        };
        double vol = 0.0, cur_area = 0.0, z_prev = pts[0][2];
        for (const auto& p : pts) {
            vol += cur_area * (p[2] - z_prev);
            z_prev = p[2];
            // insert (x, y) into the staircase
            double x = p[0], y = p[1];
            // predecessor with greatest x' <= x has the lowest y' among them
            auto it = std::upper_bound(st.begin(), st.end(), x,
                [](double v, const std::pair<double, double>& e)
                { return v < e.first; });
            if (it != st.begin() && std::prev(it)->second <= y)
                continue;                            // dominated — skip
            // erase now-dominated successors (x' >= x, y' >= y)
            auto er = it;
            while (er != st.end() && er->second >= y) ++er;
            it = st.erase(it, er);
            st.insert(it, {x, y});
            cur_area = area();
        }
        vol += cur_area * (WREF_ - z_prev);
        return vol;
    }

    // m = 3: exact contributions, leave-one-out with clipped opponents:
    // I_HV(w_i) = Vol([w_i, ref]) − HV3D({max(w_i, w_q) : q != i}).
    static void contrib_exact_3d(const std::vector<std::vector<double>>& W,
                                 std::vector<double>& contrib) {
        int K = static_cast<int>(W.size());
        contrib.assign(K, 0.0);
        std::vector<std::array<double, 3>> clipped;
        clipped.reserve(K);
        for (int i = 0; i < K; ++i) {
            double box = (WREF_ - W[i][0]) * (WREF_ - W[i][1])
                       * (WREF_ - W[i][2]);
            clipped.clear();
            for (int q = 0; q < K; ++q) {
                if (q == i) continue;
                clipped.push_back({std::max(W[i][0], W[q][0]),
                                   std::max(W[i][1], W[q][1]),
                                   std::max(W[i][2], W[q][2])});
            }
            contrib[i] = box - hv3d(clipped);
        }
    }

    // m > 3: Monte-Carlo exclusive-dominance counters in [0,2]^m, sampler
    // seeded from rng_; adaptive sample count (accumulating, doubling while
    // fewer than half of the counters are non-zero, budget 16x base).
    void contrib_mc(const std::vector<std::vector<double>>& W, int m,
                    std::vector<double>& contrib) {
        int K = static_cast<int>(W.size());
        std::vector<long long> cnt(K, 0);
        std::uniform_real_distribution<double> U(0.0, WREF_);
        const long long base =
            std::max<long long>(20000, 200LL * static_cast<long long>(K));
        long long total = 0, chunk = base;
        std::vector<double> s(m);
        while (true) {
            for (long long it = 0; it < chunk; ++it) {
                for (int j = 0; j < m; ++j) s[j] = U(rng_);
                int dom_idx = -1, dom_cnt = 0;
                for (int i = 0; i < K; ++i) {
                    bool dom = true;
                    for (int j = 0; j < m; ++j)
                        if (W[i][j] > s[j]) { dom = false; break; }
                    if (dom) { ++dom_cnt; dom_idx = i; if (dom_cnt > 1) break; }
                }
                if (dom_cnt == 1) ++cnt[dom_idx];
            }
            total += chunk;
            int nonzero = 0;
            for (long long c : cnt) if (c > 0) ++nonzero;
            if (nonzero >= K / 2 || total >= 16 * base) break;
            chunk = total;                           // double the sample count
        }
        contrib.assign(K, 0.0);
        for (int i = 0; i < K; ++i) contrib[i] = static_cast<double>(cnt[i]);
    }

    // Remove the vector with the smallest HV contribution (Alg. 1 Lines 9-11);
    // ties are broken uniformly at random.
    void prune_min_hv_contrib(std::vector<std::vector<double>>& W, int m) {
        int K = static_cast<int>(W.size());
        if (K == 0) return;
        std::vector<double> contrib;
        if      (m == 2) contrib_exact_2d(W, contrib);
        else if (m == 3) contrib_exact_3d(W, contrib);
        else             contrib_mc(W, m, contrib);
        double mn = *std::min_element(contrib.begin(), contrib.end());
        std::vector<int> ties;
        for (int i = 0; i < K; ++i) if (contrib[i] == mn) ties.push_back(i);
        int pick = ties[std::uniform_int_distribution<int>(
            0, static_cast<int>(ties.size()) - 1)(rng_)];
        W.erase(W.begin() + pick);
    }

    // Algorithm 1: full tmax add+prune loop, no early exit.
    void generate_weight_vectors(int n_vectors, int m) {
        std::vector<std::vector<double>> W;
        W.reserve(n_vectors + 1);
        std::uniform_real_distribution<double> U(0.0, 1.0);

        auto draw_simplex = [&]() {
            std::vector<double> x(m - 1);
            for (int j = 0; j < m - 1; ++j) x[j] = U(rng_);
            std::sort(x.begin(), x.end());
            std::vector<double> w(m);
            w[0] = x[0];
            for (int j = 1; j < m - 1; ++j) w[j] = x[j] - x[j - 1];
            w[m - 1] = 1.0 - x[m - 2];
            return w;
        };

        for (int t = 0; t < tmax_; ++t) {
            W.push_back(draw_simplex());
            if (static_cast<int>(W.size()) > n_vectors)
                prune_min_hv_contrib(W, m);
        }
        // Guarantee exactly n_vectors even if tmax_ was set too small.
        while (static_cast<int>(W.size()) < n_vectors)
            W.push_back(draw_simplex());
        while (static_cast<int>(W.size()) > n_vectors)
            W.pop_back();

        weight_vectors_ = std::move(W);
    }

    // ------------------------------------------------------------------ //
    //  Adaptive Reference Point (Eq. 5):
    //  z*_i = min_{x in pool} f_i(x) − max_j { range_j(pool) }
    // ------------------------------------------------------------------ //
    std::vector<double> compute_reference_point(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int k = 0; k < m; ++k) {
                fmin[k] = std::min(fmin[k], o[k]);
                fmax[k] = std::max(fmax[k], o[k]);
            }
        }
        double max_range = 0.0;
        for (int k = 0; k < m; ++k)
            max_range = std::max(max_range, fmax[k] - fmin[k]);
        std::vector<double> zstar(m);
        for (int k = 0; k < m; ++k) zstar[k] = fmin[k] - max_range;
        return zstar;
    }

    // ------------------------------------------------------------------ //
    //  Binary R2 indicator (Eq. 4):
    //  I_R2(x, y) = R2({x}, V, z*) − R2({x ∪ y}, V, z*)
    //             = (1/|V|) Σ_v [ t_x − min(t_x, t_y) ],
    //  t_a = max_j v_j |z*_j − f_j(a)|.
    // ------------------------------------------------------------------ //
    double ir2(const std::vector<double>& fx,
               const std::vector<double>& fy,
               const std::vector<double>& zstar) const {
        int m = static_cast<int>(fx.size());
        double sum = 0.0;
        for (const auto& v : weight_vectors_) {
            double tx = 0.0, ty = 0.0;
            for (int j = 0; j < m; ++j) {
                double vx = v[j] * std::abs(zstar[j] - fx[j]);
                double vy = v[j] * std::abs(zstar[j] - fy[j]);
                if (vx > tx) tx = vx;
                if (vy > ty) ty = vy;
            }
            sum += tx - std::min(tx, ty);
        }
        return sum / static_cast<double>(weight_vectors_.size());
    }

    // ------------------------------------------------------------------ //
    //  Fitness assignment (Algorithm 2, Line 21):
    //  F(x_i) = Σ_{y_j != x_i} −e^{−I_R2(y_j, x_i)/κ}     higher is better.
    //  FEASIBILITY mode (extension): infeasible get a large penalty.
    // ------------------------------------------------------------------ //
    void calculate_fitness(DataVault<Ind_t>& vault, int n,
                           const std::vector<double>& zstar) {
        std::vector<double> cvs(n, 0.0);
        if (constraint_mode != ConstraintMode::NONE)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        for (int i = 0; i < n; ++i) {
            if (constraint_mode == ConstraintMode::FEASIBILITY && cvs[i] > 0.0) {
                vault.get_ind(i).fitness = -(1e6 + cvs[i]);
                continue;
            }
            const auto& fi = vault.objectives_of(i);
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (constraint_mode == ConstraintMode::FEASIBILITY && cvs[j] > 0.0)
                    continue;
                sum += -std::exp(-ir2(vault.objectives_of(j), fi, zstar) / kappa_);
            }
            vault.get_ind(i).fitness = sum;
        }
    }

    // ------------------------------------------------------------------ //
    //  Environmental selection (Algorithm 2, Lines 22-26): iteratively remove
    //  the lowest-fitness individual, then update the remaining fitness:
    //  F(x_i) += e^{−I_R2(x*, x_i)/κ} (Line 25). z* fixed during the loop
    //  (updated once per generation at Line 20).
    // ------------------------------------------------------------------ //
    void environmental_selection(DataVault<Ind_t>& vault, int target_n,
                                 const std::vector<double>& zstar) {
        while (static_cast<int>(vault.active_n()) > target_n) {
            int curr = static_cast<int>(vault.active_n());

            int worst = 0;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double worst_cv  = vault.get_cv(0);
                double worst_fit = vault.get_ind(0).fitness;
                for (int i = 1; i < curr; ++i) {
                    double cv_i  = vault.get_cv(i);
                    double fit_i = vault.get_ind(i).fitness;
                    bool i_inf = (cv_i     > 0.0);
                    bool w_inf = (worst_cv > 0.0);
                    if (i_inf && !w_inf) {
                        worst = i; worst_cv = cv_i; worst_fit = fit_i;
                    } else if (i_inf && w_inf) {
                        if (cv_i > worst_cv) { worst = i; worst_cv = cv_i; worst_fit = fit_i; }
                    } else if (!i_inf && !w_inf) {
                        if (fit_i < worst_fit) { worst = i; worst_cv = cv_i; worst_fit = fit_i; }
                    }
                }
            } else {
                for (int i = 1; i < curr; ++i)
                    if (vault.get_ind(i).fitness < vault.get_ind(worst).fitness)
                        worst = i;
            }

            // Update fitness of the remaining individuals (Line 25).
            const auto& fw = vault.objectives_of(worst);
            double cv_worst = (constraint_mode != ConstraintMode::NONE)
                              ? vault.get_cv(worst) : 0.0;
            for (int i = 0; i < curr; ++i) {
                if (i == worst) continue;
                if (constraint_mode == ConstraintMode::FEASIBILITY) {
                    if (vault.get_cv(i) > 0.0) continue;  // already penalised
                    if (cv_worst > 0.0) continue;          // worst was infeasible
                }
                double val = ir2(fw, vault.objectives_of(i), zstar);
                vault.get_ind(i).fitness += std::exp(-val / kappa_);
            }

            vault.swap_active(worst, curr - 1);
            vault.reduce(curr - 1);
        }
    }

    // ------------------------------------------------------------------ //
    //  Binary tournament (§IV-C): draws two individuals — DISTINCT by this
    //  port's choice, not by the paper's requirement (see the deviation in the
    //  header) — compares
    //  the pair with the binary R2 indicator — a is superior when
    //  I_R2(a,b) < I_R2(b,a) (Sec. IV-B monotonicity); on a tie one of them
    //  is selected uniformly at random.
    //  FEASIBILITY mode (extension): feasible beats infeasible first.
    // ------------------------------------------------------------------ //
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_);
        int b = a;
        if (dist.b() > dist.a())
            do { b = dist(rng_); } while (b == a);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double cva = vault.get_cv(a), cvb = vault.get_cv(b);
            bool a_feas = (cva <= 0.0), b_feas = (cvb <= 0.0);
            if ( a_feas && !b_feas) return a;
            if (!a_feas &&  b_feas) return b;
            if (!a_feas && !b_feas) return (cva < cvb) ? a : b;
        }
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        double iab = ir2(fa, fb, zstar_);
        double iba = ir2(fb, fa, zstar_);
        if (iab < iba) return a;
        if (iba < iab) return b;
        return (std::uniform_int_distribution<int>(0, 1)(rng_) == 0) ? a : b;
    }

public:
    R2IBEACore() = default;

    void set_kappa        (double k)  { kappa_ = k; }
    void set_tmax         (int t)     { tmax_  = t; }
    void set_pc           (double p)  { pc_    = p; }
    void set_eta_crossover(double e)  { eta_c_ = e; }
    void set_eta_mutation (double e)  { eta_m_ = e; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // ------------------------------------------------------------------ //
    //  setup: generate weight vectors → random init → evaluate → z* → fitness
    // ------------------------------------------------------------------ //
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        // Generate |V| = µ weight vectors (Table I).
        generate_weight_vectors(n, m);

        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::uniform_int_distribution<int>     dist_bin (0, 1);
        std::vector<double> vars (vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist_real(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = dist_bin(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables    (i, vars);
        }
        vault.sync();

        zstar_ = compute_reference_point(vault, n);
        calculate_fitness(vault, n, zstar_);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        if (weight_vectors_.empty())
            generate_weight_vectors(n, m);
        zstar_ = compute_reference_point(vault, n);
        calculate_fitness(vault, n, zstar_);
    }

    // ------------------------------------------------------------------ //
    //  step: one full generation (Algorithm 2)
    //   mating (binary R2 tournaments with last generation's z*) →
    //   SBX(P_c)/PM → sync → z* update over R_g (Line 20) →
    //   fitness over R_g (Line 21) → environmental selection (Lines 22-26)
    // ------------------------------------------------------------------ //
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // ---- expand: active = 2n ----
        vault.expand(vault.pop_size());

        // ---- breed n offspring into slots [n, 2n) ----
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_int);
            int p2 = tournament(vault, dist_int);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n());
                std::vector<int> bc1, bc2;
                for (int j = 0; j < vault.bin_vars_n(); ++j) {
                    bv1[j] = vault.get_bin_variable(p1, j);
                    bv2[j] = vault.get_bin_variable(p2, j);
                }
                ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i + 1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(n + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(n + i + 1, c2, bc2);
            } else {
                vault.set_variables(n + i, c1);
                if (i + 1 < n) vault.set_variables(n + i + 1, c2);
            }
        }

        // ---- evaluate offspring ----
        vault.sync();

        // ---- Algorithm 2, Line 20: update z* from R_g = P ∪ O (size 2n) ----
        zstar_ = compute_reference_point(vault, n * 2);

        // ---- fitness assignment over the combined pool R_g (Line 21) ----
        calculate_fitness(vault, n * 2, zstar_);

        // ---- environmental selection: remove worst until n remain ----
        environmental_selection(vault, n, zstar_);

        // ---- reduce to n ----
        if (static_cast<int>(vault.active_n()) > n)
            vault.reduce(n);
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// IBEA-eps+-clean ("eps+ mIBEA") — canonical IBEA-eps+ plus
// the mIBEA dominance filter.
//
// This file is a COMPOSITION of two published algorithms, not an
// implementation of a paper of its own, so it carries no DOI of its own. Both
// parents are cited instead:
//   base       E. Zitzler, S. Kuenzli — "Indicator-Based Selection in
//              Multiobjective Search", PPSN VIII, LNCS 3242, 2004, pp. 832-842.
//              doi:10.1007/978-3-540-30217-9_84
//              Contributes: Alg.1+2, the I_eps+ indicator, the frozen
//              normalization and c = max|I| for the removal loop.
//   addition   W. Li, E. Ozcan, R. John, J. H. Drake, A. Neumann, M. Wagner —
//              "A modified indicator-based evolutionary algorithm (mIBEA)",
//              IEEE CEC 2017, pp. 1047-1054.
//              doi:10.1109/CEC.2017.7969423
//              Contributes: Step 2.1 only — the fast non-dominated sort that
//              reduces the pool to front 0 before scaling.
//
// PURPOSE. A like-for-like eps+ counterpart of mIBEA, so that the contribution
// of "pruning dominated solutions" can be separated from that of "per-removal
// renormalization" without an indicator mismatch — mIBEA uses I_HD, this uses
// I_eps+.
//
// Composition (the primary implementations are left untouched):
//   - base: IBEAePlusCore (ibea_eplus.hpp) — Zitzler & Kuenzli 2004, Alg.1+2;
//     the normalization and c = max|I| are FROZEN for the removal loop, as the
//     canonical algorithm requires;
//   - addition: mIBEA Step 2.1 (Li et al., CEC 2017, Alg.2; mibea.hpp) —
//     fast non-dominated sorting, with the pool reduced to front 0 BEFORE
//     scaling; also applied in setup(), following the Alg.2 flow
//     (Step 1 -> Step 2.1 -> Step 2.2/3).
//   - after the filter the pool may hold fewer than mu individuals, as in
//     mIBEA, and is deliberately NOT topped up; mating samples from the actual
//     parents_n(), and the offspring count is always mu.
//
// Operator defaults are those of ibea_eplus.hpp (kappa=0.05, SBX eta=20,
// pc=1.0, pm=0.01, paper-exact), which isolates the effect of the filter when
// no overrides are supplied.
// NOTE when comparing against mIBEA: that algorithm uses pc=0.9 and pm=1/n, so
// align them through the eta_c / pc / pm setters if a matched comparison is
// wanted.
//
// The constraint branches (beyond the paper, off by default) mirror
// ibea_eplus.hpp; the front-0 filter under CDP/FEASIBILITY uses constrained
// domination, mirroring mibea.hpp.
// ============================================================================

#include <algorithm>
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
class IBEAePlusCleanCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // Defaults follow ibea_eplus.hpp (see its note on footnote 3).
    double       kappa_ = 0.05;
    double       eta_c_ = 20.0;
    double       eta_m_ = 20.0;
    double       pc_    = 1.0;
    double       pm_    = 0.01;
    std::mt19937 rng_{std::random_device{}()};

    // ── Dominance helpers (mirroring mibea.hpp) ────────────────────────────
    bool dominates_plain(const std::vector<double>& a,
                         const std::vector<double>& b) const {
        bool better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) better = true;
        }
        return better;
    }
    bool dominates_cdp(const std::vector<double>& fa, double cva,
                       const std::vector<double>& fb, double cvb) const {
        bool af = (cva <= 0.0), bf = (cvb <= 0.0);
        if ( af && !bf) return true;
        if (!af &&  bf) return false;
        if (!af && !bf) return cva < cvb;
        return dominates_plain(fa, fb);
    }

    // ── mIBEA Step 2.1: reduce the pool to the non-dominated front 0 ───────
    // Differs from mibea.hpp only in storing the ranks in a local vector, since
    // IBEA_Individual has no rank field; the behaviour is identical.
    void filter_to_nondominated(DataVault<Ind_t>& vault) {
        int n = static_cast<int>(vault.active_n());
        if (n <= 1) return;

        std::vector<double> cvs(n, 0.0);
        const bool use_cdp = (constraint_mode == ConstraintMode::CDP ||
                              constraint_mode == ConstraintMode::FEASIBILITY);
        if (use_cdp)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        std::vector<char> dominated(n, 0);
        for (int i = 0; i < n; ++i) {
            const auto& fi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& fj = vault.objectives_of(j);
                bool j_dom_i = use_cdp
                    ? dominates_cdp(fj, cvs[j], fi, cvs[i])
                    : dominates_plain(fj, fi);
                if (j_dom_i) { dominated[i] = 1; break; }
            }
        }

        // Partition: move the non-dominated solutions to the front of the
        // active range. dominated[] is reordered in step with swap_active.
        int write = 0;
        for (int i = 0; i < n; ++i) {
            if (dominated[i]) continue;
            if (write != i) {
                vault.swap_active(write, i);
                std::swap(dominated[write], dominated[i]);
            }
            ++write;
        }
        if (write < n && write > 0) vault.reduce(write);
    }

    // ── I_eps+ / normalization / c — UNCHANGED from ibea_eplus.hpp ─────────
    double eps_indicator_with_cv(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 double cv_a, double cv_b,
                                 const std::vector<double>& fmin,
                                 const std::vector<double>& fmax) const
    {
        double worst = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) {
            double range   = fmax[i] - fmin[i];
            double shift_a = (cv_a > 0.0) ? cv_a / (range > 1e-14 ? range : 1.0) : 0.0;
            double shift_b = (cv_b > 0.0) ? cv_b / (range > 1e-14 ? range : 1.0) : 0.0;
            double an = (range > 1e-14) ? (a[i] - fmin[i]) / range : 0.0;
            double bn = (range > 1e-14) ? (b[i] - fmin[i]) / range : 0.0;
            an += shift_a;
            bn += shift_b;
            worst = std::max(worst, an - bn);
        }
        return worst;
    }

    double eps_indicator_norm(const std::vector<double>& a,
                              const std::vector<double>& b,
                              const std::vector<double>& fmin,
                              const std::vector<double>& fmax) const
    {
        double worst = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) {
            double range = fmax[i] - fmin[i];
            double an = (range > 1e-14) ? (a[i] - fmin[i]) / range : 0.0;
            double bn = (range > 1e-14) ? (b[i] - fmin[i]) / range : 0.0;
            worst = std::max(worst, an - bn);
        }
        return worst;
    }

    void calc_obj_bounds(DataVault<Ind_t>& vault, int n,
                         std::vector<double>& fmin,
                         std::vector<double>& fmax) const
    {
        int m = vault.objs_n();
        fmin.assign(m,  std::numeric_limits<double>::max());
        fmax.assign(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int k = 0; k < m; ++k) {
                fmin[k] = std::min(fmin[k], o[k]);
                fmax[k] = std::max(fmax[k], o[k]);
            }
        }
    }

    double calc_c(DataVault<Ind_t>& vault, int n,
                  const std::vector<double>& fmin,
                  const std::vector<double>& fmax,
                  const std::vector<double>& cvs) const
    {
        double c = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& oi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double v;
                if (constraint_mode == ConstraintMode::FEASIBILITY) {
                    if (cvs[i] > 0.0 || cvs[j] > 0.0) continue;
                    v = std::abs(
                        eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax));
                } else if (constraint_mode == ConstraintMode::EPS_CONSTRAINT) {
                    v = std::abs(
                        eps_indicator_with_cv(vault.objectives_of(j), oi,
                                              cvs[j], cvs[i], fmin, fmax));
                } else {
                    v = std::abs(
                        eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax));
                }
                if (v > c) c = v;
            }
        }
        return (c > 1e-14) ? c : 1.0;
    }

    double resolved_pm(const DataVault<Ind_t>& vault) const {
        if (pm_ >= 0.0) return pm_;
        int nv = vault.vars_n();
        return (nv > 0) ? 1.0 / static_cast<double>(nv) : 0.0;
    }

public:
    IBEAePlusCleanCore() = default;

    void set_kappa(double k)         { kappa_ = k; }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }
    void set_pc(double p)            { pc_    = p; }
    void set_pm(double p)            { pm_    = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<int>    dist_bin(0, 1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bvars[j] = dist_bin(rng_);

            if (vault.bin_vars_n() > 0)
                vault.set_all_variables(i, vars, bvars);
            else
                vault.set_variables(i, vars);
        }
        vault.sync();
        // mIBEA Alg.2: Step 1 -> Step 2.1 -> Step 2.2/Step 3, so the first
        // fitness is already computed over the non-dominated pool.
        filter_to_nondominated(vault);
        calculate_fitness(vault, static_cast<int>(vault.active_n()));
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        filter_to_nondominated(vault);
        calculate_fitness(vault, static_cast<int>(vault.active_n()));
    }

    // UNCHANGED from ibea_eplus.hpp.
    void calculate_fitness(DataVault<Ind_t>& vault, int n)
    {
        std::vector<double> fmin, fmax;
        calc_obj_bounds(vault, n, fmin, fmax);

        std::vector<double> cvs(n, 0.0);
        if (constraint_mode != ConstraintMode::NONE)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        double c = calc_c(vault, n, fmin, fmax, cvs);

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            const auto& oi = vault.objectives_of(i);

            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                if (cvs[i] > 0.0) {
                    vault.get_ind(i).fitness = -1e12 * (1.0 + cvs[i]);
                    continue;
                }
                for (int j = 0; j < n; ++j) {
                    if (i == j || cvs[j] > 0.0) continue;
                    sum += -std::exp(
                        -eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax)
                        / (c * kappa_));
                }
            } else if (constraint_mode == ConstraintMode::EPS_CONSTRAINT) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    double ind = eps_indicator_with_cv(
                        vault.objectives_of(j), oi,
                        cvs[j], cvs[i], fmin, fmax);
                    sum += -std::exp(-ind / (c * kappa_));
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    sum += -std::exp(
                        -eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax)
                        / (c * kappa_));
                }
            }
            vault.get_ind(i).fitness = sum;
        }
    }

    // UNCHANGED from ibea_eplus.hpp: the normalization and c are FROZEN for
    // the loop, as canonical Alg.2 requires. If active_n <= target_n after the
    // filter, this is a no-op.
    void environmental_selection(DataVault<Ind_t>& vault, int target_n)
    {
        int n0 = static_cast<int>(vault.active_n());
        if (n0 <= target_n) return;
        std::vector<double> fmin, fmax;
        calc_obj_bounds(vault, n0, fmin, fmax);

        std::vector<double> cvs0(n0, 0.0);
        if (constraint_mode != ConstraintMode::NONE)
            for (int i = 0; i < n0; ++i) cvs0[i] = vault.get_cv(i);
        double c = calc_c(vault, n0, fmin, fmax, cvs0);

        while (static_cast<int>(vault.active_n()) > target_n) {
            int curr = static_cast<int>(vault.active_n());

            int worst = 0;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double worst_cv  = vault.get_cv(0);
                double worst_fit = vault.get_ind(0).fitness;
                for (int i = 1; i < curr; ++i) {
                    double cv_i  = vault.get_cv(i);
                    double fit_i = vault.get_ind(i).fitness;
                    bool i_inf = (cv_i    > 0.0);
                    bool w_inf = (worst_cv > 0.0);
                    if (i_inf && !w_inf) {
                        worst = i; worst_cv = cv_i; worst_fit = fit_i;
                    } else if (!i_inf && w_inf) {
                        // worst stays
                    } else if (i_inf && w_inf) {
                        if (cv_i > worst_cv) { worst = i; worst_cv = cv_i; worst_fit = fit_i; }
                    } else {
                        if (fit_i < worst_fit) { worst = i; worst_cv = cv_i; worst_fit = fit_i; }
                    }
                }
            } else {
                for (int i = 1; i < curr; ++i)
                    if (vault.get_ind(i).fitness < vault.get_ind(worst).fitness)
                        worst = i;
            }

            const std::vector<double> ow = vault.objectives_of(worst);
            double cv_worst = (constraint_mode != ConstraintMode::NONE)
                              ? vault.get_cv(worst) : 0.0;

            for (int i = 0; i < curr; ++i) {
                if (i == worst) continue;
                double ind_val;
                if (constraint_mode == ConstraintMode::FEASIBILITY) {
                    if (cv_worst > 0.0 || vault.get_cv(i) > 0.0) continue;
                    ind_val = eps_indicator_norm(ow, vault.objectives_of(i),
                                                 fmin, fmax);
                } else if (constraint_mode == ConstraintMode::EPS_CONSTRAINT) {
                    double cv_i = vault.get_cv(i);
                    ind_val = eps_indicator_with_cv(ow, vault.objectives_of(i),
                                                    cv_worst, cv_i, fmin, fmax);
                } else {
                    ind_val = eps_indicator_norm(ow, vault.objectives_of(i),
                                                 fmin, fmax);
                }
                vault.get_ind(i).fitness += std::exp(-ind_val / (c * kappa_));
            }
            vault.swap_active(worst, curr - 1);
            vault.reduce(curr - 1);
        }
    }

    void step(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();

        // After the previous step()'s filter, active_n may be below pop_size:
        // read the parents BEFORE expand() and write the offspring into slots
        // [off_base, off_base+n).
        int n_parents = vault.parents_n();
        std::uniform_int_distribution<int> dist_int(0, n_parents - 1);
        int off_base = vault.expand(n);
        double pm = resolved_pm(vault);

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int a = dist_int(rng_), b = dist_int(rng_);
            int p1 = (vault.get_ind(a).fitness > vault.get_ind(b).fitness) ? a : b;
            int cc = dist_int(rng_), d = dist_int(rng_);
            int p2 = (vault.get_ind(cc).fitness > vault.get_ind(d).fitness) ? cc : d;

            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);

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
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();

        // mIBEA Step 2.1: reduce the merged pool to front 0 BEFORE scaling.
        filter_to_nondominated(vault);

        // Steps 2.2 + 3-4: fitness over the filtered pool (normalized on that
        // same pool), then environmental selection down to pop_size, frozen for
        // the loop.
        calculate_fitness(vault, static_cast<int>(vault.active_n()));
        environmental_selection(vault, n);
    }
};

} // namespace mootation

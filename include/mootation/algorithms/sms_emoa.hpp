#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../hv_contribution.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/real_crossover.hpp"
#include "../operators/real_mutation.hpp"

namespace mootation {

// ============================================================================
// SMS-EMOA — S-metric selection EMOA
// M. Emmerich, N. Beume, B. Naujoks, "An EMO Algorithm Using the Hypervolume
// Measure as Selection Criterion", EMO 2005, LNCS 3410, pp. 62-76,
// doi:10.1007/978-3-540-31880-4_5            (source: sms-emoa_emmerich2005)
// N. Beume, B. Naujoks, M. Emmerich, "SMS-EMOA: Multiobjective selection
// based on dominated hypervolume", EJOR 181(3):1653-1669, 2007,
// doi:10.1016/j.ejor.2006.08.008                (source: sms-emoa_beume2007)
// — the journal version: the 2005 algorithm (Algorithms 1-2, Eq. 3), the
// reference point (§2.1.3), the "dp" variant (§2.2), the settings (§3.2).
// N. Beume, "Hypervolume-based Metaheuristics for Multiobjective
// Optimization", PhD thesis, TU Dortmund, 2011, §3.1   (source:
//                                   sms-emoa_beume2011_thesis)
// — restates the basic algorithm (Algorithm 3.1) and gives the parameters.
//
// A steady-state (μ+1) scheme; one step() is one offspring, one evaluation:
//   1. two parents drawn uniformly from the population, with replacement
//      (thesis §3.1, "contrarily to the binary tournament used in most other
//      EMOA");
//   2. SBX, one of its two children chosen uniformly, then polynomial
//      mutation (thesis Algorithm 3.2; footnote 1 of both papers: the
//      operators of ε-MOEA, from the KanGAL code);
//   3. Q = P ∪ {q}; non-dominated sorting of Q into F_1 … F_v;
//   4. if |F_v| = 1 that point goes; otherwise the point of F_v with the
//      smallest hypervolume contribution ΔS(s, F_v) (EJOR Algorithm 2 and
//      Eq. 3, as in 2005) goes. At M = 2 the two extremes of F_v are always
//      kept — "We decided to omit y_ref and always keep these extremal
//      solutions" (EJOR §2.1.3; 2005 §3.2) — and the others compare by
//      EJOR Eq. 6 (2005 Eq. 4), which needs no reference point. From M = 3
//      the contributions are taken against the ADAPTIVE reference point
//      r = nad(Q) + (1, …, 1), "the vector of the currently worst objective
//      values increased by 1.0" (EJOR §2.1.3; the thesis's Def. 2.6 and
//      Algorithm 3.1, line 9).
// Parameters (thesis §3.4.1, "chosen according to the studies of Deb et al.
// (2003)"; EJOR §3.2 gives no values, only "the same parameter settings as
// in the benchmark from Deb et al."): η_c = 15, η_m = 20, p_c = 1,
// p_m = 1/n. EJOR's runs: μ = 100; 20 000 evaluations on ZDT, 30 000 on
// DTLZ, 100 000 on DTLZ3. SBX and PM are the NSGA-II implementation's
// (operators/sbx.hpp, poly_mutation.hpp), as in the thesis's experiments.
// The crossover and mutation are switchable as in NSGA-II (task 2, B3).
//
// Declared readings and deviations:
//   SMS-1. Several fronts. EJOR has two Reduce procedures: the basic one
//     (Algorithm 2, Eq. 3, as in 2005) removes the smallest ΔS of the worst
//     front F_v whatever the number of fronts; "SMS-EMOA dp" (§2.2,
//     Algorithm 3, Eq. 5) removes, when there are several fronts, the point
//     of F_v dominated by the most points of Q, else the smallest ΔS of F_1.
//     Both are benchmarked (Tables 1-2); the thesis (§3.1, "Variants",
//     Eq. 3.1) credits dp to Naujoks et al. (CEC 2005) and leaves it out.
//     Implemented: the basic one (the thesis's Algorithm 3.1); dp is not.
//   SMS-2. Contributions (hv_contribution.hpp): exact for M ≤ 3 (EJOR Eq. 6,
//     the 2005 paper's Eq. 4, at M = 2; the exclusive hypervolume by WFG at
//     M = 3, where EJOR §2.3 uses the exact O(μ³) grid algorithm of Naujoks
//     et al.); Monte-Carlo for M ≥ 4, which EJOR does not cover (it calls
//     SMS-EMOA "hardly applicable" there, §5), as HypE does (task: "при
//     M = 5 — Монте-Карло, как у HypE"), 10 000 samples a step shared out
//     over the points of F_v, each point sampled in its own bounding box.
//     set_n_samples changes the number; set_n_samples(0) makes every M exact.
//   SMS-3. A tie in the smallest contribution removes the point that comes
//     first in the population, the offspring last.
//   SMS-4. Two objectives, two points. When F_v holds only two points at
//     M = 2, both are extremes and "always keep" cannot hold; the papers do
//     not say which goes, and here their contributions against
//     r = nad(Q) + 1 decide — the thesis's rule, which it uses at every M
//     ("However, the usage of the adaptive reference point is recommended",
//     §3.1, "Variants of SMS-EMOA"). With three points or more the order at
//     M = 2 does not depend on the objectives' scale (2005 §3.2); from M = 3
//     the offset of r is absolute and it does (see
//     tests/test_scale_invariance.cpp). The extremes are the least f_1 and
//     the least f_2 of F_v, the lower other objective breaking a tie (so of
//     two copies one is kept and the other contributes nothing).
//   SMS-5. constraint_mode NONE (the default) ignores constraints;
//     FEASIBILITY and CDP sort by Deb's constrained domination (a feasible
//     point beats an infeasible one, two infeasible ones compare by their
//     violation); EPS_CONSTRAINT acts as FEASIBILITY. The papers give the
//     selection no constraint handling; EJOR's constrained airfoil case
//     (§4) repeats the variation up to 1000 times to meet the geometric
//     constraints and says no more.
//   SMS-6. A binary or mixed genome uses the library's uniform crossover and
//     bit-flip mutation, as NSGA-II does; the papers are real-valued.
// ============================================================================
template <typename Ind_t>
class SMSEMOACore {
public:
    // Supported: NONE, FEASIBILITY, CDP (SMS-5)
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double eta_c_ = 15.0;   // thesis §3.4.1
    double eta_m_ = 20.0;   // thesis §3.4.1
    double pc_    = 1.0;    // thesis §3.4.1
    int    n_samples_ = 10000;   // SMS-2
    ops::CrossoverSpec xover_;
    ops::MutationSpec  mut_;
    std::mt19937 rng_{std::random_device{}()};

    static bool dominates(const std::vector<double>& a, const std::vector<double>& b)
    {
        bool better = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) better = true;
        }
        return better;
    }

    // Front index (0 = best) of every one of [0, n).
    std::vector<int> ranks(DataVault<Ind_t>& vault, int n)
    {
        const bool cons = constraint_mode != ConstraintMode::NONE;
        std::vector<std::vector<double>> F(static_cast<std::size_t>(n));
        std::vector<double> cv(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            F[i] = vault.objectives_of(static_cast<std::size_t>(i));
            if (cons) cv[i] = vault.get_cv(static_cast<std::size_t>(i));
        }
        auto beats = [&](int a, int b) {
            if (cons) {
                const bool fa = cv[a] <= 0.0, fb = cv[b] <= 0.0;
                if (fa != fb) return fa;
                if (!fa) return cv[a] < cv[b];
            }
            return dominates(F[a], F[b]);
        };
        std::vector<std::vector<int>> S(static_cast<std::size_t>(n));
        std::vector<int> count(static_cast<std::size_t>(n), 0), rank(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                if (beats(i, j))      { S[i].push_back(j); ++count[j]; }
                else if (beats(j, i)) { S[j].push_back(i); ++count[i]; }
            }
        std::vector<int> cur;
        for (int i = 0; i < n; ++i) if (count[i] == 0) cur.push_back(i);
        for (int r = 0; !cur.empty(); ++r) {
            std::vector<int> next;
            for (int i : cur) {
                rank[i] = r;
                for (int j : S[i]) if (--count[j] == 0) next.push_back(j);
            }
            cur = std::move(next);
        }
        return rank;
    }

    // Reduce(Q): the index in [0, n) of the point to remove (steps 3-4).
    int worst(DataVault<Ind_t>& vault, int n)
    {
        const auto rank = ranks(vault, n);
        const int v = *std::max_element(rank.begin(), rank.end());
        std::vector<int> last;
        for (int i = 0; i < n; ++i) if (rank[i] == v) last.push_back(i);
        if (last.size() == 1) return last[0];
        const std::size_t m = static_cast<std::size_t>(vault.objs_n());
        std::vector<double> r(m, -std::numeric_limits<double>::infinity());
        for (int i = 0; i < n; ++i) {
            const auto& f = vault.objectives_of(static_cast<std::size_t>(i));
            for (std::size_t k = 0; k < m; ++k) r[k] = std::max(r[k], f[k]);
        }
        for (double& x : r) x += 1.0;                      // r = nad(Q) + 1
        std::vector<std::vector<double>> Fv;
        for (int i : last) Fv.push_back(vault.objectives_of(static_cast<std::size_t>(i)));
        const bool mc = m >= 4 && n_samples_ > 0;
        const std::size_t per = mc ? std::max<std::size_t>(
            1, static_cast<std::size_t>(n_samples_) / Fv.size()) : 0;
        auto c = mc ? hv::contributions_mc(Fv, r, per, rng_)
                    : hv::contributions_exact(Fv, r);
        if (m == 2 && Fv.size() > 2) {                     // SMS-4: the two extremes stay
            std::size_t a = 0, b = 0;
            for (std::size_t t = 1; t < Fv.size(); ++t) {
                if (Fv[t][0] < Fv[a][0] || (Fv[t][0] == Fv[a][0] && Fv[t][1] < Fv[a][1])) a = t;
                if (Fv[t][1] < Fv[b][1] || (Fv[t][1] == Fv[b][1] && Fv[t][0] < Fv[b][0])) b = t;
            }
            c[a] = c[b] = std::numeric_limits<double>::infinity();
        }
        std::size_t arg = 0;
        for (std::size_t t = 1; t < c.size(); ++t)
            if (c[t] < c[arg]) arg = t;
        for (std::size_t t = 0; t < last.size(); ++t)
            vault.get_ind(static_cast<std::size_t>(last[t])).fitness = c[t];
        return last[arg];
    }

public:
    SMSEMOACore() = default;

    void set_seed(unsigned s)             { rng_.seed(s); }
    void set_eta_crossover(double e)      { eta_c_ = e; }
    void set_eta_mutation(double e)       { eta_m_ = e; }
    void set_pc(double p)                 { pc_ = p; }
    void set_n_samples(int s)             { n_samples_ = s; }
    void set_crossover(ops::Crossover c)  { xover_.kind = c; }
    void set_mutation(ops::Mutation m)    { mut_.kind = m; }
    void set_mutation_scale(double s)     { mut_.scale = s; }
    void set_mixture_q(double q)          { mut_.q = q; }
    void set_blx_alpha(double a)          { xover_.alpha = a; }
    void set_bound_repair(ops::BoundRepair b) {
        ops::require_repair(b, "sms_emoa", true, false);
        xover_.repair = b; mut_.repair = b;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        std::uniform_int_distribution<int> bit(0, 1);
        std::vector<double> x(static_cast<std::size_t>(vault.vars_n()));
        std::vector<int> bx(static_cast<std::size_t>(vault.bin_vars_n()));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                const double lo = bounds[j].first.value_or(0.0);
                const double hi = bounds[j].second.value_or(1.0);
                x[static_cast<std::size_t>(j)] = lo + U01(rng_) * (hi - lo);
            }
            for (auto& b : bx) b = bit(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(static_cast<std::size_t>(i), x, bx);
            else                        vault.set_variables(static_cast<std::size_t>(i), x);
        }
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>&) {}

    void step(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> pick(0, n - 1);
        const int a = pick(rng_), b = pick(rng_);
        const std::vector<double> x1 = vault.variables_of(static_cast<std::size_t>(a));
        const std::vector<double> x2 = vault.variables_of(static_cast<std::size_t>(b));
        // one more parent, uniformly as the two, for the crossovers that take
        // more than two (B2)
        auto draw = [&] { return vault.variables_of(static_cast<std::size_t>(pick(rng_))); };
        std::vector<double> c1, c2;
        xover_.apply_any(x1, x2, draw, c1, c2, bounds, eta_c_, pc_, rng_);
        std::uniform_int_distribution<int> coin(0, 1);
        const bool second = coin(rng_) == 1;
        std::vector<double>& child = second ? c2 : c1;
        const double pm = vault.vars_n() > 0 ? 1.0 / vault.vars_n() : 0.0;
        mut_.apply(child, bounds, eta_m_, pm, rng_);

        const int q = vault.expand(1);
        if (vault.bin_vars_n() > 0) {
            std::vector<int> b1(static_cast<std::size_t>(vault.bin_vars_n())), b2 = b1, bc1, bc2;
            for (int j = 0; j < vault.bin_vars_n(); ++j) {
                b1[static_cast<std::size_t>(j)] = vault.get_bin_variable(static_cast<std::size_t>(a), j);
                b2[static_cast<std::size_t>(j)] = vault.get_bin_variable(static_cast<std::size_t>(b), j);
            }
            ops::binary_crossover(b1, b2, bc1, bc2, rng_);
            std::vector<int>& bchild = second ? bc2 : bc1;
            ops::bit_flip_mutation(bchild, vault.bin_vars_n(), rng_);
            vault.set_all_variables(static_cast<std::size_t>(q), child, bchild);
        } else {
            vault.set_variables(static_cast<std::size_t>(q), child);
        }
        vault.sync();

        const int out = worst(vault, n + 1);
        if (out != n) vault.swap_active(static_cast<std::size_t>(out), static_cast<std::size_t>(n));
        vault.reduce(n);
    }
};

}   // namespace mootation

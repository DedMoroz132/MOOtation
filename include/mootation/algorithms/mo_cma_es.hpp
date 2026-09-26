#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../hv_contribution.hpp"

namespace mootation {

// ============================================================================
// MO-CMA-ES, the steady-state (μ+1) variant with the population-based notion
// of success (MO-CMA-ES^P)
// T. Voß, N. Hansen, C. Igel, "Improved Step Size Adaptation for the
// MO-CMA-ES", GECCO 2010, pp. 487-494, doi:10.1145/1830483.1830573
//                                                  (source: mo-cma-es_voss2010)
// The algorithm itself is Igel, Hansen & Roth, "Covariance Matrix Adaptation
// for Multi-objective Optimization", Evol. Comput. 15(1):1-28, 2007,
// doi:10.1162/evco.2007.15.1.1 (source: mo-cma-es_igel2007); the 2010 paper
// restates it (§2, Algorithm 1) and gives its defaults "as given in [14]".
//
// An individual is [x, p̄_succ, σ, p_c, C]. One step() is one offspring
// (Algorithm 1 with λ = 1, line 4a):
//   1. a parent a_i drawn uniformly from ndom(Q), the non-dominated members;
//   2. the offspring copies a_i and samples x' ~ x_i + σ_i·N(0, C_i);
//   3. Q ← Q ∪ {a'}; the member ranked last by ≺_Q (Eq. 4: level of
//      non-dominance, then contribution rank — the smallest hypervolume
//      contribution in the last front goes first) is removed;
//   4. succ = 1 if the offspring was not the one removed (Eq. 9, "selected
//      for the next parent population");
//   5. offspring (lines 9-16): p̄_succ ← (1 − c_p)·p̄_succ + c_p·succ,
//      σ ← σ·exp((p̄_succ − p_target) / (d·(1 − p_target))); if p̄_succ <
//      p_thresh: p_c ← (1 − c_c)·p_c + √(c_c(2 − c_c))·(x' − x_i)/σ_i and
//      C ← (1 − c_cov)·C + c_cov·p_c·p_cᵀ; else p_c ← (1 − c_c)·p_c and
//      C ← (1 − c_cov)·C + c_cov·(p_c·p_cᵀ + c_c(2 − c_c)·C);
//      parent (lines 17-18): the same p̄_succ and σ update with the same succ.
// Defaults (§2, "as given in [14]"; [14]'s Table 1 at λ = 1): d = 1 + n/2,
// p_target = 2/11 (MOCMA-9), c_p = p_target/(2 + p_target) = 1/12,
// c_c = 2/(n + 2), c_cov = 2/(n² + 6),
// p_thresh = 0.44; σ_0 = 0.6 of the range; box constraints by the penalised
// fitness of Eq. 5, f(feasible(x)) + α‖x − feasible(x)‖², α = 10⁻⁶, feasible
// the L1-closest point of the box (the clip).
// Contribution ranks (§2): the reference point is "dominated by all
// individuals" and chosen "such that all boundary elements get the highest
// contribution ranks"; ties are broken at random.
//
// Declared readings and deviations:
//   MOCMA-1 (task, as DMS-1). The search runs in coordinates normalised to
//     the box, u = (x − ℓ)/(u − ℓ): the paper's σ_0 = 0.6·(x^u − x^l) is
//     stated for equal ranges, and normalising makes it literal for unequal
//     ones (equivalently, C_0 = diag of the squared ranges). The 2007 paper
//     does the same by hand: σ_0 is 60 % of x_2^u − x_2^l, and ZDT4's first
//     variable is "rescaled to [−5, 5]" to give it that range (§4.2). The
//     penalty of Eq. 5 is measured in the same coordinates.
//   MOCMA-2. The individual keeps its unclipped x; the library's population
//     holds feasible(x) and its true objectives f(feasible(x)), so the answer
//     is always inside the box. Selection compares the penalised values.
//   MOCMA-3. Initial state: p̄_succ = p_target, p_c = 0, C = I (the 2007
//     paper, §2.1 "Initialization"); x and σ "must be chosen problem
//     dependent" there — here x is uniform in the box and σ = σ_0.
//   MOCMA-4. The reference point is r = nad + (nad − ideal) over Q in every
//     objective (nad + 1 where the range is zero), which every member
//     dominates, and the boundary elements of the front being ranked — its
//     best point in each objective — are given an infinite contribution. The
//     paper leaves r free under exactly those two conditions; this choice
//     scales with the objectives, so the selection does not depend on their
//     units. "Boundary element" is the 2010 paper's (the argmin of each
//     objective); the 2007 paper (§3.2.3) calls boundary every element whose
//     contribution depends on the reference point — the same two points at
//     M = 2, more at M ≥ 3.
//   MOCMA-5. Contributions as SMS-EMOA's (hv_contribution.hpp): exact for
//     M ≤ 3, Monte-Carlo for M ≥ 4 (10 000 samples a step; set_n_samples, 0
//     = exact at every M).
//   MOCMA-6. C is factorised by Cholesky for every offspring (O(n³)); the
//     paper's implementation updates the factor incrementally (Suttorp,
//     Hansen & Igel 2009), which samples the same distribution faster.
//   MOCMA-7. constraint_mode NONE (the default) ignores general constraints;
//     FEASIBILITY and CDP rank by Deb's constrained domination; EPS_CONSTRAINT
//     acts as FEASIBILITY. The paper handles box constraints only.
//   MOCMA-8. Continuous variables only; a problem with binary ones is refused.
//   MOCMA-9. p_target = 2/11 ≈ 0.1818, the 2007 paper's value. The 2010
//     paper prints (5 + √(1/2))⁻¹ ≈ 0.1752, the root over the whole 1/2
//     (HAL version, PDF page 4), while it gives its defaults "as given in [14]":
//     [14]'s Table 1 has 1/(5 + √λ/2), the root over λ alone (p. 5), and every
//     MO-CMA-ES individual is a (1+1)-ES, λ = 1. The 2010 value is Table 1's
//     at λ = 2 — a slip in restating it, taken as such.
// ============================================================================
template <typename Ind_t>
class MOCMAESCore {
public:
    // Supported: NONE, FEASIBILITY, CDP (MOCMA-7)
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double sigma0_    = 0.6;      // §2
    double alpha_pen_ = 1e-6;     // §2, Eq. 5
    double p_thresh_  = 0.44;     // §2
    int    n_samples_ = 10000;    // MOCMA-5
    std::vector<double> lo_, hi_;
    std::mt19937 rng_{std::random_device{}()};

    int    n_ = 0;
    double d_ = 0, p_target_ = 0, c_p_ = 0, c_c_ = 0, c_cov_ = 0;

    void constants(int n)
    {
        n_ = n;
        const double nd = static_cast<double>(n);
        d_        = 1.0 + nd / 2.0;
        p_target_ = 2.0 / 11.0;                  // 1/(5 + √λ/2) at λ = 1, MOCMA-9
        c_p_      = p_target_ / (2.0 + p_target_);
        c_c_      = 2.0 / (nd + 2.0);
        c_cov_    = 2.0 / (nd * nd + 6.0);
    }

    void read_bounds(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("mo_cma_es: continuous variables only (MOCMA-8)");
        const auto& b = vault.get_bounds();
        lo_.resize(b.size());
        hi_.resize(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (!b[j].first || !b[j].second)
                throw std::invalid_argument("mo_cma_es: every variable needs finite bounds");
            lo_[j] = *b[j].first;
            hi_[j] = *b[j].second;
        }
        constants(static_cast<int>(b.size()));
    }

    // feasible(u) (the clip) in problem units, and ‖u − feasible(u)‖²
    std::vector<double> feasible(const std::vector<double>& u, double& dist2) const
    {
        std::vector<double> x(u.size());
        dist2 = 0.0;
        for (std::size_t j = 0; j < u.size(); ++j) {
            const double c = std::min(1.0, std::max(0.0, u[j]));
            dist2 += (u[j] - c) * (u[j] - c);
            x[j] = lo_[j] + c * (hi_[j] - lo_[j]);
        }
        return x;
    }

    void init_strategy(Ind_t& a) const
    {
        const std::size_t n = static_cast<std::size_t>(n_);
        a.sigma  = sigma0_;
        a.p_succ = p_target_;
        a.pc.assign(n, 0.0);
        a.C.assign(n * n, 0.0);
        for (std::size_t j = 0; j < n; ++j) a.C[j * n + j] = 1.0;
    }

    // Eq. 6 and 7
    void update_step(Ind_t& a, double succ) const
    {
        a.p_succ = (1.0 - c_p_) * a.p_succ + c_p_ * succ;
        a.sigma *= std::exp((a.p_succ - p_target_) / (d_ * (1.0 - p_target_)));
    }

    // Lower-triangular A with A·Aᵀ = C (row-major); a tiny ridge if C has
    // lost definiteness to rounding.
    static std::vector<double> cholesky(const std::vector<double>& C, std::size_t n)
    {
        for (double ridge = 0.0;; ridge = ridge == 0.0 ? 1e-12 : ridge * 10.0) {
            std::vector<double> A(n * n, 0.0);
            bool ok = true;
            for (std::size_t i = 0; i < n && ok; ++i)
                for (std::size_t j = 0; j <= i; ++j) {
                    double s = C[i * n + j] + (i == j ? ridge : 0.0);
                    for (std::size_t k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
                    if (i == j) {
                        if (!(s > 0.0)) { ok = false; break; }
                        A[i * n + i] = std::sqrt(s);
                    } else {
                        A[i * n + j] = s / A[j * n + j];
                    }
                }
            if (ok) return A;
            if (ridge > 1.0) throw std::runtime_error("mo_cma_es: covariance matrix not positive definite");
        }
    }

    static bool dominates(const std::vector<double>& a, const std::vector<double>& b)
    {
        bool better = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) better = true;
        }
        return better;
    }

    // Levels of non-dominance of [0, n), on the penalised values.
    std::vector<int> levels(DataVault<Ind_t>& vault, int n)
    {
        const bool cons = constraint_mode != ConstraintMode::NONE;
        std::vector<double> cv(static_cast<std::size_t>(n), 0.0);
        if (cons)
            for (int i = 0; i < n; ++i) cv[static_cast<std::size_t>(i)] = vault.get_cv(static_cast<std::size_t>(i));
        auto beats = [&](int a, int b) {
            if (cons) {
                const bool fa = cv[static_cast<std::size_t>(a)] <= 0.0;
                const bool fb = cv[static_cast<std::size_t>(b)] <= 0.0;
                if (fa != fb) return fa;
                if (!fa) return cv[static_cast<std::size_t>(a)] < cv[static_cast<std::size_t>(b)];
            }
            return dominates(vault.get_ind(static_cast<std::size_t>(a)).fpen,
                             vault.get_ind(static_cast<std::size_t>(b)).fpen);
        };
        std::vector<std::vector<int>> S(static_cast<std::size_t>(n));
        std::vector<int> count(static_cast<std::size_t>(n), 0), lv(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                if (beats(i, j))      { S[static_cast<std::size_t>(i)].push_back(j); ++count[static_cast<std::size_t>(j)]; }
                else if (beats(j, i)) { S[static_cast<std::size_t>(j)].push_back(i); ++count[static_cast<std::size_t>(i)]; }
            }
        std::vector<int> cur;
        for (int i = 0; i < n; ++i) if (count[static_cast<std::size_t>(i)] == 0) cur.push_back(i);
        for (int l = 0; !cur.empty(); ++l) {
            std::vector<int> next;
            for (int i : cur) {
                lv[static_cast<std::size_t>(i)] = l;
                for (int j : S[static_cast<std::size_t>(i)])
                    if (--count[static_cast<std::size_t>(j)] == 0) next.push_back(j);
            }
            cur = std::move(next);
        }
        return lv;
    }

    // The member of [0, n) ranked last by ≺_Q (step 3); sets every rank.
    int last_ranked(DataVault<Ind_t>& vault, int n)
    {
        const auto lv = levels(vault, n);
        for (int i = 0; i < n; ++i) vault.get_ind(static_cast<std::size_t>(i)).rank = lv[static_cast<std::size_t>(i)];
        const int v = *std::max_element(lv.begin(), lv.end());
        std::vector<int> last;
        for (int i = 0; i < n; ++i) if (lv[static_cast<std::size_t>(i)] == v) last.push_back(i);
        if (last.size() == 1) return last[0];

        const std::size_t m = static_cast<std::size_t>(vault.objs_n());
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> lo(m, inf), hi(m, -inf);
        for (int i = 0; i < n; ++i) {
            const auto& f = vault.get_ind(static_cast<std::size_t>(i)).fpen;
            for (std::size_t k = 0; k < m; ++k) { lo[k] = std::min(lo[k], f[k]); hi[k] = std::max(hi[k], f[k]); }
        }
        std::vector<double> r(m);
        for (std::size_t k = 0; k < m; ++k)                  // MOCMA-4
            r[k] = hi[k] > lo[k] ? hi[k] + (hi[k] - lo[k]) : hi[k] + 1.0;
        std::vector<std::vector<double>> Fv;
        for (int i : last) Fv.push_back(vault.get_ind(static_cast<std::size_t>(i)).fpen);
        const bool mc = m >= 4 && n_samples_ > 0;
        auto c = mc ? hv::contributions_mc(Fv, r, std::max<std::size_t>(
                          1, static_cast<std::size_t>(n_samples_) / Fv.size()), rng_)
                    : hv::contributions_exact(Fv, r);
        for (std::size_t k = 0; k < m; ++k) {                  // boundary elements
            std::size_t b = 0;
            for (std::size_t t = 1; t < Fv.size(); ++t) if (Fv[t][k] < Fv[b][k]) b = t;
            c[b] = inf;
        }
        double least = inf;
        for (double x : c) least = std::min(least, x);
        std::vector<int> tied;
        for (std::size_t t = 0; t < c.size(); ++t) if (c[t] == least) tied.push_back(last[t]);
        if (tied.size() == 1) return tied[0];
        std::uniform_int_distribution<std::size_t> pick(0, tied.size() - 1);  // "at random"
        return tied[pick(rng_)];
    }

    void place(DataVault<Ind_t>& vault, int v, const std::vector<double>& u)
    {
        double dist2 = 0.0;
        vault.set_variables(static_cast<std::size_t>(v), feasible(u, dist2));
        auto& a = vault.get_ind(static_cast<std::size_t>(v));
        a.x = u;
        a.dist2 = dist2;
    }

    void penalise(DataVault<Ind_t>& vault, int v)
    {
        auto& a = vault.get_ind(static_cast<std::size_t>(v));
        a.fpen = vault.objectives_of(static_cast<std::size_t>(v));
        for (double& f : a.fpen) f += alpha_pen_ * a.dist2;        // Eq. 5
    }

public:
    MOCMAESCore() = default;

    void set_seed(unsigned s)       { rng_.seed(s); }
    void set_n_samples(int s)       { n_samples_ = s; }
    void set_initial_sigma(double s){ sigma0_ = s; }

    void setup(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        const int mu = vault.pop_size();
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        std::vector<double> u(static_cast<std::size_t>(n_));
        for (int i = 0; i < mu; ++i) {
            for (auto& x : u) x = U01(rng_);
            place(vault, i, u);
            init_strategy(vault.get_ind(static_cast<std::size_t>(i)));
        }
        vault.sync();
        for (int i = 0; i < mu; ++i) penalise(vault, i);
        const auto lv = levels(vault, mu);
        for (int i = 0; i < mu; ++i) vault.get_ind(static_cast<std::size_t>(i)).rank = lv[static_cast<std::size_t>(i)];
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        const int mu = vault.pop_size();
        for (int i = 0; i < mu; ++i) {
            auto& a = vault.get_ind(static_cast<std::size_t>(i));
            const auto& x = vault.variables_of(static_cast<std::size_t>(i));
            a.x.resize(x.size());
            for (std::size_t j = 0; j < x.size(); ++j) a.x[j] = (x[j] - lo_[j]) / (hi_[j] - lo_[j]);
            a.dist2 = 0.0;
            init_strategy(a);
            penalise(vault, i);
        }
        const auto lv = levels(vault, mu);
        for (int i = 0; i < mu; ++i) vault.get_ind(static_cast<std::size_t>(i)).rank = lv[static_cast<std::size_t>(i)];
    }

    void step(DataVault<Ind_t>& vault)
    {
        const int mu = vault.pop_size();
        const std::size_t n = static_cast<std::size_t>(n_);
        // 1. a parent from ndom(Q) (line 4a)
        std::vector<int> nd;
        for (int i = 0; i < mu; ++i) if (vault.get_ind(static_cast<std::size_t>(i)).rank == 0) nd.push_back(i);
        std::uniform_int_distribution<std::size_t> pick(0, nd.size() - 1);
        const int p = nd[pick(rng_)];
        // 2. the offspring (lines 5-6)
        const int q = vault.expand(1);
        {
            const Ind_t parent = vault.get_ind(static_cast<std::size_t>(p));
            const auto A = cholesky(parent.C, n);
            std::normal_distribution<double> N01(0.0, 1.0);
            std::vector<double> z(n), u = parent.x;
            for (auto& v : z) v = N01(rng_);
            for (std::size_t i = 0; i < n; ++i) {
                double y = 0.0;
                for (std::size_t k = 0; k <= i; ++k) y += A[i * n + k] * z[k];
                u[i] += parent.sigma * y;
            }
            auto& child = vault.get_ind(static_cast<std::size_t>(q));
            child.sigma = parent.sigma; child.p_succ = parent.p_succ;
            child.pc = parent.pc;       child.C = parent.C;
            place(vault, q, u);
        }
        vault.sync();
        penalise(vault, q);
        // 3-4. selection and success (Eq. 9)
        const int out = last_ranked(vault, mu + 1);
        const double succ = out == q ? 0.0 : 1.0;
        // 5. strategy updates (lines 9-18)
        {
            auto& parent = vault.get_ind(static_cast<std::size_t>(p));
            auto& child  = vault.get_ind(static_cast<std::size_t>(q));
            const double sigma_p = parent.sigma;
            update_step(child, succ);
            if (child.p_succ < p_thresh_) {
                const double w = std::sqrt(c_c_ * (2.0 - c_c_));
                for (std::size_t j = 0; j < n; ++j)
                    child.pc[j] = (1.0 - c_c_) * child.pc[j] + w * (child.x[j] - parent.x[j]) / sigma_p;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        child.C[i * n + j] = (1.0 - c_cov_) * child.C[i * n + j] +
                                             c_cov_ * child.pc[i] * child.pc[j];
            } else {
                for (auto& v : child.pc) v *= (1.0 - c_c_);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        child.C[i * n + j] = (1.0 - c_cov_) * child.C[i * n + j] +
                                             c_cov_ * (child.pc[i] * child.pc[j] +
                                                       c_c_ * (2.0 - c_c_) * child.C[i * n + j]);
            }
            update_step(parent, succ);
        }
        // the one ranked last leaves
        if (out != mu) vault.swap_active(static_cast<std::size_t>(out), static_cast<std::size_t>(mu));
        vault.reduce(mu);
    }
};

}   // namespace mootation

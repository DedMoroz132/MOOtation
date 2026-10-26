#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../detmath.hpp"
#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../dss.hpp"
#include "../operators/bound_repair.hpp"
#include "../operators/learned_coords.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"
#include "../random.hpp"

namespace mootation {

// ============================================================================
// MOSA — multi-objective simulated annealing over an archive (task 7 of
// 2026-10-10, §1 and §6): a composition of this library. Which mechanism
// comes from where, checked against the papers (sources
// dbmosa_smith2008_final-submission, amosa_bandyopadhyay2008_accepted,
// psa_czyzak1998):
//   * DBMOSA (Smith, Everson, Fieldsend, Murphy & Misra, IEEE TEVC
//     12(3):323-342, 2008, doi:10.1109/TEVC.2007.904345): the archive of
//     EVERY non-dominated point (unconstrained, §III); the energy "share of
//     the archive dominating the point" (Eq. 12), here against A' = A
//     without the point, without their refinement by the attainment surface
//     (§IV); the Laplace perturbation (§V-B: there one variable at random,
//     with per-variable scalings adapted to traversal and location moves;
//     here every variable with probability 1/n and one scale falling over
//     the budget); T_0 from the energy changes over ln 2 (§V-A: there the
//     mean positive change over a burn-in; here the median over pairs of
//     start points); the restart from the archive (§II-C, "periodically
//     restarting the annealer from a randomly chosen element of F"; here the
//     better by energy of two random members, with probability sa_reseed);
//   * AMOSA (Bandyopadhyay, Saha, Maulik & Deb, IEEE TEVC 12(3):269-283,
//     2008, doi:10.1109/TEVC.2007.900837): the energies against an archive.
//     Its archive is bounded (HL and SL, single-linkage clustering, §III-B)
//     — sa_archive_cap > 0 is that idea with DSS — and its "informed
//     reseeding" (Case 3(a): a new point that dominates the current one but
//     not the archive points dominating it gives way, with probability
//     1/(1 + exp(−Δdom_min)), to the closest of those) is NOT used here;
//   * Pareto simulated annealing (Czyżak & Jaszkiewicz, J. Multi-Crit.
//     Decis. Anal. 7(1):34-47, 1998,
//     doi:10.1002/(SICI)1099-1360(199801)7:1<34::AID-MCDA161>3.0.CO;2-6):
//     chains with weights that adapt to push them apart (the rule of its
//     Algorithm, with α = 1.05) and the acceptance by the weighted sum of
//     the objectives' differences (rule SL, which its experiments use);
//   * the steps: the library's SBX and polynomial mutation, a differential
//     step, the Laplace move above;
//   * the combined energy, the Chebyshev energy against the archive's ideal
//     point, the step "SBX or DE" and the adaptive choice of step: ours.
//
// ONE STEP IS ONE EVALUATION. A run:
//   1. START: a Latin hypercube of max(n + 2, 20) points, all evaluated and
//      offered to the archive (a seeded population — the campaign's `init` —
//      instead, when there is one).
//   2. ARCHIVE A: every non-dominated point evaluated, with its variables; a
//      point some member weakly dominates is refused, the members it weakly
//      dominates go. sa_archive_cap > 0 thins it by DSS beyond that size.
//   3. ENERGY of f against A' = A without f (span: A's range per objective,
//      a zero range counting as 1):
//        eps  −min_{a∈A'} max_i (a_i − f_i)/span_i, the additive ε of A'
//             against f — negative where f extends the front;
//        dom  the share of A' that dominates f (DBMOSA's Eq. 12 divides
//             |F̃_x'| − |F̃_x| by one |F̃|, F̃ = A ∪ {x, x'}; here each
//             share has its own A');
//        cmb  (the default) dom when it is > 0, eps otherwise: the pressure
//             of dom on a dominated point, the spread of eps among the rest;
//        tch  max_i w_k,i (f_i − z_i)/span_i, z A's ideal point, w_k the
//             weights of chain k (PSA, below).
//   4. TEMPERATURE: T_0 = the median |ΔE| between pairs of start points
//      (the energy differences the acceptance compares) / ln 2 — half the
//      uphill moves between start points accepted; T falls geometrically to
//      T_0·sa_t_ratio (10⁻⁴) at the last step of the budget.
//   5. STEP (sa_step) from the chain's current point x:
//        laplace  every variable with probability 1/n (at least one) moves by
//                 Laplace noise of scale s·(ub − lb), s falling geometrically
//                 from sa_s0 (0.2) to sa_s_end (0.002) over the budget;
//        de       with probability p_de (0.7) x + F·(x_a − x_b), x_a and x_b
//                 two archive members, every variable (CR = 1); otherwise
//                 the Laplace move;
//        sbx      the library's SBX of x and a random archive member (η_c 20,
//                 every variable with probability ½, p_c = 1), one of the
//                 two children at random, then polynomial mutation (η_m 20,
//                 p_m = 1/n);
//        mix      (the default) sbx or de (p_de = 1), half and half;
//        adaptive (§6) laplace, de or sbx with probability ∝ (its successes
//                 in the last sa_adapt_window steps (200) + 1), a success
//                 being a new point the archive took — Borg's credit over a
//                 window. The probabilities are recorded.
//      DE needs three archive members and SBX two; with fewer the step is
//      the Laplace move. A variable outside the box is repaired by
//      bound_repair (reflect by default; the old value is the parent for
//      midpoint; resample and native are refused) and counted.
//   6. ACCEPTANCE: the new point's energy against the archive BEFORE it
//      joins, then the point is offered to the archive and the current
//      point's energy recomputed against today's archive; Metropolis,
//      exp(−ΔE/T). With tch the default compares the weighted sum of the
//      objectives' differences, Σ w_k,i (f'_i − f_i)/span_i, as PSA does
//      (sa_tch_accept = weighted_sum: PSA's rule SL, P = min{1, exp(−Σ λ_j
//      (f_j(y) − f_j(x))/T)} under minimization — the paper prints the sign
//      of the exponent the other way round, which Fig. 1 contradicts);
//      chebyshev compares the energies, which is not PSA's rule C (that is
//      max_j λ_j (f_j(y) − f_j(x)), x itself the reference point).
//   7. RESEEDING: with probability sa_reseed (1/50) per step the chain's
//      current point becomes the better by energy of two random archive
//      members (DBMOSA's restart from the archive, above).
//   8. CHAINS: sa_chains (1; 10 with tch) advance in turn over one archive.
//      With tch their weights start at the corners of the simplex (0.9 e_i
//      + 0.1/M) and continue Dirichlet (PSA starts every chain from random
//      weights); sa_weights = psa (the default with tch) adapts them after
//      every step of chain k as PSA's Algorithm does: x' the current point of
//      another chain nearest to x_k (here in the span-normalised objectives;
//      the paper names no distance) among those neither dominating x_k nor
//      dominated by it; every objective where x_k is better than x' or equal
//      to it has its weight multiplied by α (sa_psa_alpha, 1.05), every other
//      divided by it, the sum then put back to 1 — chain k moves further
//      where it beats its neighbour; without such an x' the weights are
//      drawn at random again (Dirichlet(1)). sa_weights = fixed keeps the
//      start weights. The mean and spread of the weights over the chains are
//      recorded.
// The schedules (T, s) need the budget in steps: the run's t_max (one
// evaluation a step; corrected after the first step under an evaluation
// budget).
//
// THE ANSWER: the population is the archive reduced to N by DSS (dss.hpp, as
// DMS's) — published every N steps and at the last one, the cadence at which
// an observer reads it. Annealing is sequential: one evaluation at a time,
// nothing to evaluate in parallel.
//
// THE ARCHIVE'S PASSES (task 11 of 2026-10-18, §3): the archive's objectives
// are kept beside it by columns (ac_), which the energies and the archive's
// rule read objective by objective over all the members at once instead of
// member by member; its span and ideal point are computed again only when it
// changed (a member joining moves the bounds by itself; one leaving from the
// upper bound has them computed again: one leaving from the lower bound cannot,
// the point that drove it out being at or below it); the energies' ε only
// where it is the answer (eps, and cmb with no
// member dominating f). The same comparisons and the same expressions, each
// member's maximum over the objectives in their order and the minimum over the
// members in theirs: the same run, bit for bit.
//
// MOSA-c. Constraints (constraint_mode not NONE): the archive keeps feasible
// points only, or, while none is feasible, the least violating one; an
// infeasible new point is accepted only from an infeasible current point
// whose violation it does not exceed, a feasible one always from an
// infeasible one.
//
// Knobs: sa_energy, sa_step, p_de, F, sa_chains, sa_reseed, sa_t_ratio,
// sa_s0, sa_s_end, sa_archive_cap, sa_weights, sa_psa_alpha, sa_tch_accept,
// sa_adapt_window, eta_c, eta_m, pm, bound_repair.
// ============================================================================

enum class SAEnergy { Cmb, Eps, Dom, Tch };
enum class SAStep { Mix, Laplace, DE, SBX, Adaptive };

// The run's annealing record, as the binding's operator_stats reads it: the
// current temperature and step scale, the archive's size, the reseedings so
// far, with tch the weights' mean and standard deviation over the chains per
// objective, with sa_step = adaptive the steps' probabilities.
inline std::vector<std::pair<std::string, double>>& mosa_record()
{
    thread_local std::vector<std::pair<std::string, double>> r;
    return r;
}

template <typename Ind_t>
class MOSACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    struct Point {
        std::vector<double> x, f;
        double cv = 0.0;
        VaultPoint pt{};                  // the point whole (data_vault.hpp, WRITING BACK)
    };
    using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;

private:
    SAEnergy energy_ = SAEnergy::Cmb;
    SAStep   step_kind_ = SAStep::Mix;
    bool     psa_ = true;                 // sa_weights: psa | fixed
    bool     weighted_sum_ = true;        // sa_tch_accept: weighted_sum | chebyshev
    double   p_de_ = 0.7, F_ = 0.5, reseed_ = 1.0 / 50.0, t_ratio_ = 1e-4;
    double   s0_ = 0.2, s_end_ = 0.002, psa_alpha_ = 1.05;
    double   eta_c_ = 20.0, eta_m_ = 20.0;
    std::optional<double> pm_;
    double pc_ = 1.0;   // SBX probability (knob pc)
    int      chains_ = 0;                 // 0: 1, or 10 with tch
    int      cap_ = 0;                    // 0: no cap
    int      window_ = 200;
    ops::BoundRepair repair_ = ops::BoundRepair::Reflect;
    ops::LearnedCoords coords_;           // coords: the sbx step in learned coordinates
    int      t_max_ = 1000;
    std::mt19937 rng_{0};

    std::vector<Point> archive_;
    std::vector<std::vector<double>> ac_;  // archive_'s objectives by columns (THE ARCHIVE'S PASSES)
    mutable bool stale_ = true;           // the bounds to be computed again from ac_
    mutable std::vector<double> lo_c_, hi_c_, span_c_, ideal_c_;
    mutable std::vector<char> same_, weak_, strict_;   // the passes' flags, one per member
    mutable std::vector<double> mx_;
    std::vector<Point> cur_;              // the chains' current points
    unsigned long long present_seen_ = 0; // the vault's present() they are brought to
    std::vector<std::vector<double>> W_;  // tch: each chain's weights
    std::vector<double> lo_, hi_;
    double   T0_ = 1e-3, T_ = 1e-3, s_ = 0.2;
    long long k_ = 0;                     // steps done
    long long reseeds_ = 0;
    std::deque<std::pair<int, bool>> hist_;   // adaptive: (step, success) per step
    std::vector<double> probs_ = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};   // laplace, de, sbx

    bool constrained() const { return constraint_mode != ConstraintMode::NONE; }

    static bool dominates(const std::vector<double>& a, const std::vector<double>& b)
    {
        bool strict = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) strict = true;
        }
        return strict;
    }
    static bool weakly(const std::vector<double>& a, const std::vector<double>& b)
    {
        for (std::size_t j = 0; j < a.size(); ++j) if (a[j] > b[j]) return false;
        return true;
    }

    int K() const { return chains_ > 0 ? chains_ : (energy_ == SAEnergy::Tch ? 10 : 1); }

    std::size_t n_obj() const { return archive_.empty() ? 0 : archive_[0].f.size(); }

    // the archive changed: ac_ from archive_ again, span and ideal to be computed
    void archive_changed()
    {
        const std::size_t m = n_obj();
        ac_.assign(m, std::vector<double>(archive_.size()));
        for (std::size_t i = 0; i < archive_.size(); ++i)
            for (std::size_t j = 0; j < m; ++j) ac_[j][i] = archive_[i].f[j];
        stale_ = true;
    }

    void bounds_of_archive() const
    {
        const std::size_t m = n_obj();
        std::vector<double> lo(m, std::numeric_limits<double>::infinity()), hi(m, -std::numeric_limits<double>::infinity());
        for (std::size_t j = 0; j < m; ++j)
            for (double v : ac_[j]) {
                lo[j] = std::min(lo[j], v);
                hi[j] = std::max(hi[j], v);
            }
        lo_c_ = lo;
        hi_c_ = hi;
        stale_ = false;
        bounds_derived();
    }

    // span and ideal from the bounds
    void bounds_derived() const
    {
        const std::size_t m = lo_c_.size();
        span_c_.assign(m, 0.0);
        for (std::size_t j = 0; j < m; ++j) span_c_[j] = hi_c_[j] > lo_c_[j] ? hi_c_[j] - lo_c_[j] : 1.0;
        ideal_c_ = lo_c_;
    }

    const std::vector<double>& span() const
    {
        if (stale_) bounds_of_archive();
        return span_c_;
    }
    const std::vector<double>& ideal() const
    {
        if (stale_) bounds_of_archive();
        return ideal_c_;
    }

    // ── the steps ────────────────────────────────────────────────────────────
    void repair_into_box(std::vector<double>& y, const std::vector<double>& parent)
    {
        ops::RepairTally tally;
        for (std::size_t j = 0; j < y.size(); ++j)
            if (y[j] < lo_[j] || y[j] > hi_[j]) {
                tally.out();
                y[j] = ops::repair_value(y[j], lo_[j], hi_[j], parent[j], repair_, rng_);
            }
    }

    std::vector<double> laplace(const std::vector<double>& x)
    {
        const std::size_t n = x.size();
        rnd::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<char> mask(n, 0);
        bool any = false;
        for (std::size_t j = 0; j < n; ++j) { mask[j] = u(rng_) < 1.0 / static_cast<double>(n); any = any || mask[j]; }
        if (!any) mask[rnd::uniform_int_distribution<std::size_t>(0, n - 1)(rng_)] = 1;
        rnd::exponential_distribution<double> ex(1.0);
        std::vector<double> y(x);
        for (std::size_t j = 0; j < n; ++j)
            if (mask[j]) {
                const double sign = u(rng_) < 0.5 ? -1.0 : 1.0;
                y[j] += sign * ex(rng_) * s_ * (hi_[j] - lo_[j]);
            }
        repair_into_box(y, x);
        return y;
    }

    // coords (task 7, supplement 1; operators/learned_coords.hpp): with
    // learned coordinates the SBX and the mutation act in y, the basis learned
    // from the archive (the non-dominated points, so pca and pca_nd agree),
    // a generation being pop_size steps.
    std::vector<double> sbx_move(const std::vector<double>& x, const Bounds& bounds, int N)
    {
        rnd::uniform_int_distribution<std::size_t> any(0, archive_.size() - 1);
        const std::vector<double> xa = archive_[any(rng_)].x;
        const long long gen = k_ / std::max(1, N);
        if (coords_.due(gen)) {
            std::vector<std::vector<double>> X, F;
            for (const auto& a : archive_) { X.push_back(a.x); F.push_back(a.f); }
            coords_.fit_at(gen, X, F, bounds, rng_);
        }
        const bool learned = coords_.active();
        std::vector<double> c1, c2;
        if (learned) {
            ops::require_repair(repair_, ops::coords_name(coords_.kind), false, false);
            ops::sbx(coords_.to_y(x), coords_.to_y(xa), c1, c2, coords_.open(x.size()), eta_c_,
                     pc_, rng_);
        } else {
            ops::sbx(x, xa, c1, c2, bounds, eta_c_, pc_, rng_);
        }
        rnd::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> y = u(rng_) < 0.5 ? c1 : c2;
        const double pm = pm_ ? *pm_ : 1.0 / static_cast<double>(x.size());
        if (!learned) {
            ops::polynomial_mutation(y, bounds, eta_m_, pm, rng_);
            return y;
        }
        ops::polynomial_mutation(y, coords_.mutation_bounds(y), eta_m_, pm, rng_);
        return coords_.back(y, bounds, x, repair_, rng_);
    }

    // 0 laplace, 1 de, 2 sbx — the step actually taken
    int choose(rnd::uniform_real_distribution<double>& u)
    {
        const std::size_t na = archive_.size();
        switch (step_kind_) {
            case SAStep::Laplace: return 0;
            case SAStep::DE:      return (na >= 3 && u(rng_) < p_de_) ? 1 : 0;
            case SAStep::SBX:     return na >= 2 ? 2 : 0;
            case SAStep::Mix: {
                const bool use_sbx = u(rng_) < 0.5;
                if (use_sbx && na >= 2) return 2;
                return na >= 3 ? 1 : 0;
            }
            case SAStep::Adaptive: {
                rnd::discrete_distribution<int> d(probs_.begin(), probs_.end());
                const int op = d(rng_);
                if (op == 1 && na < 3) return 0;
                if (op == 2 && na < 2) return 0;
                return op;
            }
        }
        return 0;
    }

    // ── energies ─────────────────────────────────────────────────────────────
    double chebyshev(const std::vector<double>& f, int k, const std::vector<double>& sp,
                     const std::vector<double>& z) const
    {
        double e = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < f.size(); ++j) e = std::max(e, W_[static_cast<std::size_t>(k)][j] * (f[j] - z[j]) / sp[j]);
        return e;
    }
    // The acceptance's ΔE from x to y for chain k.
    double delta(const Point& from, const Point& to, int k) const
    {
        if (energy_ == SAEnergy::Tch && weighted_sum_) {
            const auto& sp = span();
            double d = 0.0;
            for (std::size_t j = 0; j < to.f.size(); ++j)
                d += W_[static_cast<std::size_t>(k)][j] * (to.f[j] - from.f[j]) / sp[j];
            return d;
        }
        return energy(to.f, k) - energy(from.f, k);
    }

    void publish(DataVault<Ind_t>& vault)
    {
        std::vector<std::vector<double>> F;
        F.reserve(archive_.size());
        for (const auto& a : archive_) F.push_back(a.f);
        const auto idx = dss::order(F, static_cast<std::size_t>(vault.pop_size()));
        vault.reduce(0);
        vault.expand(static_cast<int>(idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const Point& p = archive_[idx[i]];
            vault.put_point(i, p.pt);
        }
    }

    std::vector<Point> evaluate(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& X)
    {
        std::vector<Point> out;
        const int keep = static_cast<int>(vault.active_n());
        const int base = vault.expand(static_cast<int>(X.size()));
        for (std::size_t i = 0; i < X.size(); ++i)
            vault.set_variables(static_cast<std::size_t>(base) + i, X[i]);
        vault.sync();
        for (std::size_t i = 0; i < X.size(); ++i) {
            const std::size_t v = static_cast<std::size_t>(base) + i;
            out.push_back({X[i], vault.objectives_of(v), constrained() ? vault.get_cv(v) : 0.0,
                           vault.point_of(v)});
        }
        vault.reduce(keep);
        return out;
    }

    void read_bounds(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("mosa: continuous variables only");
        const auto& b = vault.get_bounds();
        lo_.resize(b.size());
        hi_.resize(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (!b[j].first || !b[j].second)
                throw std::invalid_argument("mosa: every variable needs finite bounds");
            lo_[j] = *b[j].first;
            hi_[j] = *b[j].second;
        }
    }

    // The start (1-2, 4, 8) from the evaluated start points.
    void start(DataVault<Ind_t>& vault, const std::vector<Point>& first)
    {
        archive_.clear();
        archive_changed();
        for (const auto& p : first) offer(p);
        const int k = K();
        const std::size_t m = first.empty() ? 0 : first[0].f.size();
        W_.assign(static_cast<std::size_t>(k), std::vector<double>(m, 1.0 / static_cast<double>(std::max<std::size_t>(1, m))));
        if (energy_ == SAEnergy::Tch) {
            rnd::gamma_distribution<double> g(1.0, 1.0);
            for (int c = 0; c < k; ++c) {
                auto& w = W_[static_cast<std::size_t>(c)];
                if (static_cast<std::size_t>(c) < m) {
                    for (std::size_t j = 0; j < m; ++j) w[j] = 0.1 / static_cast<double>(m) + (j == static_cast<std::size_t>(c) ? 0.9 : 0.0);
                } else {
                    double s = 0.0;
                    for (auto& v : w) { v = g(rng_); s += v; }
                    for (auto& v : w) v /= s;
                }
            }
        }
        // T_0: the median |ΔE| between pairs of start points, per chain
        std::vector<double> d;
        for (int c = 0; c < k; ++c)
            for (std::size_t a = 0; a < first.size(); ++a)
                for (std::size_t b = a + 1; b < first.size(); ++b) {
                    const double x = std::abs(delta(first[a], first[b], c));
                    if (x > 0.0 && std::isfinite(x)) d.push_back(x);
                }
        if (!d.empty()) {
            std::nth_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
            double med = d[d.size() / 2];
            if (d.size() % 2 == 0) {
                const double lower = *std::max_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2));
                med = 0.5 * (med + lower);
            }
            T0_ = med / dm::kLn2;
        } else {
            T0_ = 1e-3;
        }
        T_ = T0_;
        s_ = s0_;
        cur_.clear();
        for (int c = 0; c < k; ++c) {
            std::size_t j = 0;
            if (energy_ == SAEnergy::Tch) {
                double best = std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < archive_.size(); ++i) {
                    const double e = energy(archive_[i].f, c);
                    if (e < best) { best = e; j = i; }
                }
            } else {
                j = rnd::uniform_int_distribution<std::size_t>(0, archive_.size() - 1)(rng_);
            }
            cur_.push_back(archive_[j]);
        }
        k_ = 0;
        reseeds_ = 0;
        hist_.clear();
        probs_.assign(3, 1.0 / 3.0);
        publish(vault);
        record();
    }

    void record()
    {
        auto& r = mosa_record();
        r.clear();
        r.emplace_back("sa_T", T_);
        r.emplace_back("sa_s", s_);
        r.emplace_back("sa_archive_size", static_cast<double>(archive_.size()));
        r.emplace_back("sa_reseeds", static_cast<double>(reseeds_));
        if (energy_ == SAEnergy::Tch && !W_.empty()) {
            const std::size_t m = W_[0].size();
            for (std::size_t j = 0; j < m; ++j) {
                double mean = 0.0, sq = 0.0;
                for (const auto& w : W_) mean += w[j];
                mean /= static_cast<double>(W_.size());
                for (const auto& w : W_) sq += (w[j] - mean) * (w[j] - mean);
                r.emplace_back("sa_w_mean_" + std::to_string(j + 1), mean);
                r.emplace_back("sa_w_sd_" + std::to_string(j + 1), std::sqrt(sq / static_cast<double>(W_.size())));
            }
        }
        if (step_kind_ == SAStep::Adaptive) {
            r.emplace_back("sa_p_laplace", probs_[0]);
            r.emplace_back("sa_p_de", probs_[1]);
            r.emplace_back("sa_p_sbx", probs_[2]);
        }
    }

public:
    MOSACore() = default;

    // ── knobs ────────────────────────────────────────────────────────────────
    void set_seed(unsigned s)              { rng_.seed(s); }
    void set_t_max(int t)                  { t_max_ = std::max(1, t); }
    void set_F(double f)                   { F_ = f; }
    void set_eta_crossover(double e)       { eta_c_ = e; }
    void set_pc(double p)                  { pc_ = p; }
    void set_eta_mutation(double e)        { eta_m_ = e; }
    void set_pm(double p)                  { pm_ = p; }
    void set_bound_repair(ops::BoundRepair r)
    {
        ops::require_repair(r, "mosa", false, false);
        repair_ = r;
    }
    bool set_param(const std::string& k, double v)
    {
        if (coords_.set_param(k, v)) return true;
        auto positive = [&](const char* name) {
            if (!(v > 0.0)) throw std::invalid_argument(std::string(name) + " must be > 0");
        };
        auto whole = [&](const char* name, double least) {
            if (!(v >= least) || v != std::floor(v))
                throw std::invalid_argument(std::string(name) + " must be an integer >= " +
                                            std::to_string(static_cast<int>(least)));
        };
        if (k == "p_de")        { if (!(v >= 0.0 && v <= 1.0)) throw std::invalid_argument("p_de must be in [0, 1]"); p_de_ = v; return true; }
        if (k == "sa_chains")   { whole("sa_chains", 1); chains_ = static_cast<int>(v); return true; }
        if (k == "sa_reseed")   { if (!(v >= 0.0 && v <= 1.0)) throw std::invalid_argument("sa_reseed must be in [0, 1]"); reseed_ = v; return true; }
        if (k == "sa_t_ratio")  { positive("sa_t_ratio"); t_ratio_ = v; return true; }
        if (k == "sa_s0")       { positive("sa_s0"); s0_ = v; return true; }
        if (k == "sa_s_end")    { positive("sa_s_end"); s_end_ = v; return true; }
        if (k == "sa_archive_cap") { whole("sa_archive_cap", 0); cap_ = static_cast<int>(v); return true; }
        if (k == "sa_psa_alpha") { if (!(v >= 1.0)) throw std::invalid_argument("sa_psa_alpha must be >= 1"); psa_alpha_ = v; return true; }
        if (k == "sa_adapt_window") { whole("sa_adapt_window", 1); window_ = static_cast<int>(v); return true; }
        return false;
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (coords_.set_word(k, w)) return true;
        auto bad = [&](const char* words) {
            return std::invalid_argument(k + " = '" + w + "': " + words);
        };
        if (k == "sa_energy") {
            if (w == "cmb") energy_ = SAEnergy::Cmb;
            else if (w == "eps") energy_ = SAEnergy::Eps;
            else if (w == "dom") energy_ = SAEnergy::Dom;
            else if (w == "tch") energy_ = SAEnergy::Tch;
            else throw bad("cmb, eps, dom or tch");
            return true;
        }
        if (k == "sa_step") {
            if (w == "mix") step_kind_ = SAStep::Mix;
            else if (w == "laplace") step_kind_ = SAStep::Laplace;
            else if (w == "de") step_kind_ = SAStep::DE;
            else if (w == "sbx") step_kind_ = SAStep::SBX;
            else if (w == "adaptive") step_kind_ = SAStep::Adaptive;
            else throw bad("mix, laplace, de, sbx or adaptive");
            return true;
        }
        if (k == "sa_weights") {
            if (w != "psa" && w != "fixed") throw bad("psa or fixed");
            psa_ = w == "psa";
            return true;
        }
        if (k == "sa_tch_accept") {
            if (w != "weighted_sum" && w != "chebyshev") throw bad("weighted_sum or chebyshev");
            weighted_sum_ = w == "weighted_sum";
            return true;
        }
        return false;
    }

    // ── what the tests read ──────────────────────────────────────────────────
    const std::vector<Point>& archive() const { return archive_; }
    double temperature() const { return T_; }
    double initial_temperature() const { return T0_; }
    double step_scale() const { return s_; }
    long long reseeds() const { return reseeds_; }
    long long steps() const { return k_; }
    const std::vector<std::vector<double>>& weights() const { return W_; }
    const std::vector<double>& step_probabilities() const { return probs_; }

    // Public for the tests: the archive's rule (2), the differential step
    // (5), PSA's update of chain c's weights (8), the adaptive step's
    // record of one step (§6), and the chains' current points and weights.
    // Offer p; true if it was taken (a success of the step).
    bool offer(const Point& p)
    {
        if (constrained()) {
            const bool holds_infeasible = !archive_.empty() && archive_[0].cv > 0.0;
            if (p.cv > 0.0) {
                if (archive_.empty() || (holds_infeasible && p.cv < archive_[0].cv)) {
                    archive_.assign(1, p);
                    archive_changed();
                    return true;
                }
                return false;
            }
            if (holds_infeasible) { archive_.assign(1, p); archive_changed(); return true; }
        }
        const std::size_t m = p.f.size(), n = archive_.size();
        if (ac_.size() != m) {                                   // the first point: m columns
            ac_.assign(m, std::vector<double>());
            for (const auto& a : archive_)
                for (std::size_t j = 0; j < m; ++j) ac_[j].push_back(a.f[j]);
        }
        // weak_: the member weakly dominates p; strict_ (reused): p weakly dominates it
        weak_.assign(n, 1);
        strict_.assign(n, 1);
        for (std::size_t j = 0; j < m; ++j) {
            const double* c = ac_[j].data();
            const double fj = p.f[j];
            char* w = weak_.data();
            char* b = strict_.data();
            for (std::size_t i = 0; i < n; ++i) {
                w[i] = static_cast<char>(w[i] & !(c[i] > fj));
                b[i] = static_cast<char>(b[i] & !(fj > c[i]));
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            if (weak_[i]) return false;
        bool finite = true;
        for (double v : p.f) finite = finite && std::isfinite(v);
        bool keep_bounds = !stale_ && finite && lo_c_.size() == m;
        std::size_t kept = 0;                                    // the members p weakly dominates go
        for (std::size_t i = 0; i < n; ++i) {
            if (strict_[i]) {
                for (std::size_t j = 0; j < m && keep_bounds; ++j)
                    keep_bounds = !(ac_[j][i] == hi_c_[j]);      // it held an upper bound
                continue;
            }
            if (kept != i) {
                archive_[kept] = std::move(archive_[i]);
                for (std::size_t j = 0; j < m; ++j) ac_[j][kept] = ac_[j][i];
            }
            ++kept;
        }
        archive_.resize(kept);
        archive_.push_back(p);
        for (std::size_t j = 0; j < m; ++j) {
            ac_[j].resize(kept);
            ac_[j].push_back(p.f[j]);
        }
        if (keep_bounds) {
            for (std::size_t j = 0; j < m; ++j) {
                lo_c_[j] = std::min(lo_c_[j], p.f[j]);
                hi_c_[j] = std::max(hi_c_[j], p.f[j]);
            }
            bounds_derived();
        } else {
            stale_ = true;
        }
        if (cap_ > 0 && static_cast<int>(archive_.size()) > cap_) {
            std::vector<std::vector<double>> F;
            for (const auto& a : archive_) F.push_back(a.f);
            const auto idx = dss::order(F, static_cast<std::size_t>(cap_));
            std::vector<Point> kept_points;
            for (std::size_t i : idx) kept_points.push_back(archive_[i]);
            archive_ = std::move(kept_points);
            archive_changed();
        }
        return true;
    }

    std::vector<double> differential(const std::vector<double>& x)
    {
        rnd::uniform_int_distribution<std::size_t> any(0, archive_.size() - 1);
        const std::size_t a = any(rng_);
        std::size_t b = any(rng_);
        while (b == a) b = any(rng_);
        std::vector<double> y(x);
        for (std::size_t j = 0; j < x.size(); ++j) y[j] = x[j] + F_ * (archive_[a].x[j] - archive_[b].x[j]);
        repair_into_box(y, x);
        return y;
    }

    void psa_update(int c)
    {
        const auto& sp = span();
        const auto& x = cur_[static_cast<std::size_t>(c)].f;
        int near = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int o = 0; o < static_cast<int>(cur_.size()); ++o) {
            if (o == c) continue;
            const auto& y = cur_[static_cast<std::size_t>(o)].f;
            if (dominates(x, y) || dominates(y, x)) continue;
            double d = 0.0;
            for (std::size_t j = 0; j < x.size(); ++j) d += (x[j] - y[j]) * (x[j] - y[j]) / (sp[j] * sp[j]);
            if (d < best) { best = d; near = o; }
        }
        auto& w = W_[static_cast<std::size_t>(c)];
        double s = 0.0;
        if (near < 0) {                       // no x': random weights (PSA)
            rnd::gamma_distribution<double> g(1.0, 1.0);
            for (auto& v : w) { v = g(rng_); s += v; }
            for (auto& v : w) v /= s;
            return;
        }
        const auto& y = cur_[static_cast<std::size_t>(near)].f;
        for (std::size_t j = 0; j < w.size(); ++j) {   // x better or equal on j: up
            w[j] = x[j] <= y[j] ? w[j] * psa_alpha_ : w[j] / psa_alpha_;
            s += w[j];
        }
        for (auto& v : w) v /= s;
    }

    void note_step(int op, bool success)
    {
        hist_.emplace_back(op, success);
        while (static_cast<int>(hist_.size()) > window_) hist_.pop_front();
        std::vector<double> w(3, 1.0);
        for (const auto& h : hist_) if (h.second) w[static_cast<std::size_t>(h.first)] += 1.0;
        const double s = w[0] + w[1] + w[2];
        for (std::size_t i = 0; i < 3; ++i) probs_[i] = w[i] / s;
    }
    const std::vector<Point>& current() const { return cur_; }
    void set_current(const std::vector<Point>& c) { cur_ = c; }
    void set_weights(const std::vector<std::vector<double>>& w) { W_ = w; }

    // The energy of f for chain k against the archive (3 above).
    double energy(const std::vector<double>& f, int k) const
    {
        if (archive_.empty()) return 0.0;
        const auto& sp = span();
        if (energy_ == SAEnergy::Tch) return chebyshev(f, k, sp, ideal());
        const std::size_t m = f.size(), n = archive_.size();
        // the members other than f (same_ not set) and those dominating it
        same_.assign(n, 1);
        weak_.assign(n, 1);
        strict_.assign(n, 0);
        for (std::size_t j = 0; j < m; ++j) {
            const double* c = ac_[j].data();
            const double fj = f[j];
            char* sa = same_.data();
            char* w = weak_.data();
            char* st = strict_.data();
            for (std::size_t i = 0; i < n; ++i) {
                sa[i] = static_cast<char>(sa[i] & (c[i] == fj));
                w[i] = static_cast<char>(w[i] & !(c[i] > fj));
                st[i] = static_cast<char>(st[i] | (c[i] < fj));
            }
        }
        std::size_t others = 0, dom = 0;
        for (std::size_t i = 0; i < n; ++i) {
            others += !same_[i];
            dom += !same_[i] && weak_[i] && strict_[i];
        }
        if (others == 0) return 0.0;
        const double share = static_cast<double>(dom) / static_cast<double>(others);
        if (energy_ == SAEnergy::Dom) return share;
        if (energy_ == SAEnergy::Cmb && share > 0.0) return share;
        // ε of the members other than f against f: each member's maximum over
        // the objectives in their order, then the minimum over the members
        mx_.assign(n, -std::numeric_limits<double>::infinity());
        for (std::size_t j = 0; j < m; ++j) {
            const double* c = ac_[j].data();
            const double fj = f[j], sj = sp[j];
            double* x = mx_.data();
            for (std::size_t i = 0; i < n; ++i) x[i] = std::max(x[i], (c[i] - fj) / sj);
        }
        double eps = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i)
            if (!same_[i]) eps = std::min(eps, mx_[i]);
        return -eps;
    }

    // ── the run ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        const std::size_t n = lo_.size();
        const std::size_t S = std::max<std::size_t>(n + 2, 20);
        std::vector<std::vector<double>> X(S, std::vector<double>(n));
        rnd::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<std::size_t> perm(S);
        for (std::size_t j = 0; j < n; ++j) {                 // a Latin hypercube
            std::iota(perm.begin(), perm.end(), std::size_t{0});
            rnd::shuffle(perm.begin(), perm.end(), rng_);
            for (std::size_t i = 0; i < S; ++i)
                X[i][j] = lo_[j] + (static_cast<double>(perm[i]) + u(rng_)) / static_cast<double>(S) * (hi_[j] - lo_[j]);
        }
        vault.reduce(0);
        start(vault, evaluate(vault, X));
    }

    // A warm start (the campaign's init): the seeded points instead of the
    // Latin hypercube.
    void setup_seeded(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        std::vector<Point> first;
        for (std::size_t v = 0; v < vault.active_n(); ++v)
            first.push_back({vault.variables_of(v), vault.objectives_of(v),
                             constrained() ? vault.get_cv(v) : 0.0, vault.point_of(v)});
        start(vault, first);
    }

    void step(DataVault<Ind_t>& vault)
    {
        // the archive and the chains' points, brought to the vault's present
        // when it moved (data_vault.hpp, WRITING BACK)
        if (vault.present() != present_seen_) {
            for (auto* pts : {&archive_, &cur_})
                for (auto& p : *pts) refresh_copy(vault, p.pt, p.f, p.cv, constrained());
            present_seen_ = vault.present();
            archive_changed();
        }
        const int c = static_cast<int>(k_ % static_cast<long long>(cur_.size()));
        const double frac = t_max_ > 1 ? std::min(1.0, static_cast<double>(k_) / static_cast<double>(t_max_ - 1)) : 1.0;
        T_ = T0_ * dm::pow(t_ratio_, frac);
        s_ = s0_ * dm::pow(s_end_ / s0_, frac);
        rnd::uniform_real_distribution<double> u(0.0, 1.0);
        Point& x = cur_[static_cast<std::size_t>(c)];
        const int op = choose(u);
        std::vector<double> y;
        if (op == 2) y = sbx_move(x.x, vault.get_bounds(), vault.pop_size());
        else if (op == 1) y = differential(x.x);
        else y = laplace(x.x);
        const Point p = evaluate(vault, {y})[0];

        // 6: the new point against the archive before it joins (with tch and
        // the weighted sum: the difference in the archive's span before it)
        const bool ws = energy_ == SAEnergy::Tch && weighted_sum_;
        const double before = ws ? delta(x, p, c) : energy(p.f, c);
        const bool success = offer(p);
        bool accept;
        if (constrained() && (p.cv > 0.0 || x.cv > 0.0)) {           // MOSA-c
            accept = p.cv > 0.0 ? (x.cv > 0.0 && p.cv <= x.cv) : true;
        } else {
            const double d = ws ? before : before - energy(x.f, c);
            accept = d <= 0.0 || u(rng_) < dm::exp(-d / std::max(T_, 1e-300));
        }
        if (accept) x = p;
        if (u(rng_) < reseed_ && archive_.size() > 1) {               // 7
            rnd::uniform_int_distribution<std::size_t> any(0, archive_.size() - 1);
            const std::size_t a = any(rng_);
            std::size_t b = any(rng_);
            while (b == a) b = any(rng_);
            x = energy(archive_[a].f, c) <= energy(archive_[b].f, c) ? archive_[a] : archive_[b];
            ++reseeds_;
        }
        if (energy_ == SAEnergy::Tch && psa_ && cur_.size() > 1) psa_update(c);
        if (step_kind_ == SAStep::Adaptive) note_step(op, success);    // §6
        ++k_;
        if (k_ % vault.pop_size() == 0 || k_ >= t_max_) publish(vault);
        record();
    }
};

}   // namespace mootation

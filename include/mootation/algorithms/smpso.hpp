#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../detail/constrained.hpp"
#include "../operators/bound_repair.hpp"
#include "../operators/poly_mutation.hpp"
#include "../random.hpp"

// ============================================================================
// SMPSO — Speed-constrained Multi-objective PSO (task 9 of 2026-10-18, B1).
//
// A.J. Nebro, J.J. Durillo, J. García-Nieto, C.A. Coello Coello, F. Luna,
// E. Alba, "SMPSO: A new PSO-based metaheuristic for multi-objective
// optimization", IEEE Symp. on Computational Intelligence in Multi-Criteria
// Decision-Making (MCDM) 2009, pp. 66-73, doi:10.1109/MCDM.2009.4938830
// (source smpso_nebro2009, §II and Algorithm 1). Its departure point, OMOPSO:
// M. Reyes Sierra, C.A. Coello Coello, "Improving PSO-based multi-objective
// optimization using crowding, mutation and ε-dominance", EMO 2005, LNCS 3410,
// pp. 505-519, doi:10.1007/978-3-540-31880-4_35 (source
// omopso_reyes-sierra2005, §3 and Fig. 1).
//
// Algorithm 1 (one step() = lines 5-11, the swarm evaluated in one batch):
//   for each particle i: a leader g from the leaders archive (a binary
//   tournament by crowding distance, §II-B); r1, r2, C1, C2 drawn; Eq. 2
//     v = w·v + C1·r1·(p_i − x) + C2·r2·(g − x)
//   times the constriction factor χ (Eq. 3-4), then bounded per variable to
//   ±δ_j, δ_j = (u_j − l_j)/2 (Eq. 5-6); x ← x + v (Eq. 1), a variable outside
//   the box put back on it and its velocity multiplied by a factor (§II-C);
//   turbulence: polynomial mutation of a share of the swarm (§II-C);
//   evaluation; every particle offered to the leaders archive (non-dominated,
//   bounded, the most crowded leaving first, §II-B); p_i replaced by the new
//   x when p_i does not dominate it. The answer is the leaders archive
//   (Algorithm 1, line 13), written into the active slots after every step.
//
// THE PRESETS (smpso_preset). The paper leaves w, the sign of χ, the
// velocity at the bound and the exact turbulence open, and the reference
// codes differ on them; each preset is one reading, every part of it a knob:
//   jmetal (default)  χ as Eq. 3 prints it, without a modulus (negative for
//                     φ > 4); w = 0.1; |v| ≤ ½·width; clip; velocity × (−1)
//                     at the bound; PM (η = 20, 1/n) on particles i % 6 == 0;
//                     archive by crowding, one point removed at a time. As
//                     jMetal's SMPSO and SMPSOBuilder (MIT) do.
//   platemo           χ with the modulus, |2 − φ − √(φ² − 4φ)|, φ = C1 + C2
//                     (χ = 1 for φ ≤ 4, which is the modulus form at φ = 4);
//                     w ~ U[0.1, 0.5] per particle; velocity × 0.001 at the
//                     bound (the value §II-C reports as slightly better); PM
//                     on each particle with probability 0.15 (§II-C: "15 % of
//                     the particles").
//   fair              χ with the modulus; w = 0.1; the position put back by
//                     bound_repair = reflect (operators/bound_repair.hpp);
//                     velocity × (−1); turbulence and archive as jmetal.
//   omopso            OMOPSO (its §3): no χ (1); w ~ U[0.1, 0.5]; C1, C2 ~
//                     U[1.5, 2.0]; no velocity bound; clip, velocity × (−1);
//                     turbulence by thirds — non-uniform mutation on
//                     i % 3 == 0, uniform on i % 3 == 1, none on the rest
//                     (both with probability 1/n per variable; perturbation
//                     0.5, as jMetal's OMOPSO); only the particles that
//                     replaced their p_i offered as leaders, the leaders cut
//                     once a step by crowding; the answer an ε-dominance
//                     archive of every leader accepted (η = 0.0075, the
//                     value its §4 uses on ZDT). Its non-uniform mutation
//                     follows the run's progress: budget-dependent.
// Common to all: swarm = pop_size, r1, r2 ~ U[0, 1] and C1, C2 ~ U[c_min,
// c_max] = U[1.5, 2.5] (§II-C) drawn once per particle, the leaders archive
// of size pop_size (§IV-A: 100 particles, archive 100).
//
// THE KNOBS (each overrides its preset's value, whatever the order given):
//   chi_mode          signed | abs | none        (χ of Eq. 3, its modulus, or 1)
//   vmax_frac         the velocity bound as a share of the width, 0.5 (Eq. 6);
//                     0 = no bound
//   bound_repair      what a position outside the box becomes (clip)
//   vel_on_bound_low, vel_on_bound_up
//                     the factor on the velocity of a variable put back on
//                     its lower or upper bound
//   w_min, w_max      the inertia weight, drawn U[w_min, w_max] per particle
//                     (not drawn when equal)
//   c_min, c_max      the range of C1 and C2
//   r_per_dim         no | yes: r1, r2 per particle (as the references) or
//                     per variable
//   turbulence        every_k | prob | thirds | off; turbulence_k (6),
//                     turbulence_p (0.15); eta_m (20) and pm (1/n) of the PM
//   archive_size      the leaders archive's bound (pop_size)
//   archive_prune     one_by_one (the most crowded removed after each
//                     addition, crowding recomputed) | one_shot (the whole
//                     swarm added, then one crowding and the tail cut)
//   leader_select     crowding_tournament | random
//   leader_offer      all | improved: every particle offered as a leader, or
//                     only those that replaced their p_i (OMOPSO §3)
//   eps_archive_eta   the ε of the omopso answer archive (0.0075)
//
// READINGS (declared):
//   SMPSO-1. "χ" multiplies the whole of Eq. 2 including w·v ("the resulting
//     velocity is then multiplied by the constriction factor", §II-A).
//   SMPSO-2. The leader is drawn for every particle; §II-B's tournament reads
//     "pbest" where the leader is meant. The tournament keeps the first of the
//     two on a tie of crowding. A leader added since the last estimate counts
//     0 until the next one, which every step makes after its additions.
//   SMPSO-3. p_i is replaced unless it dominates the new position, by Deb's
//     constrained domination under constraint handling (constraint_mode not
//     NONE), Pareto dominance otherwise; the same comparison admits leaders.
//   SMPSO-4. A leader whose objectives equal a member's exactly is not added.
//   SMPSO-5. The velocity starts at 0, p_i at the initial position (Alg. 1,
//     line 1).
//   SMPSO-6. Eq. 4 sets φ = 1 for C1 + C2 <= 4, which makes Eq. 3 complex;
//     χ = 1 there, as jMetal has it (and the modulus form gives at φ = 4).
//   SMPSO-7. OMOPSO's mutation: three parts of equal size, none, uniform and
//     non-uniform (OMOPSO §3; the SMPSO paper's §II-C says 30 % each and none
//     on the rest); here interleaved by index as jMetal's OMOPSO does
//     (i % 3). The ranges of the uniform and non-uniform mutation are not in
//     the paper: jMetal's perturbation 0.5.
//   SMPSO-8. platemo's turbulence draws for each particle (probability 0.15),
//     jmetal's mutates i % 6 == 0, a sixth: §II-C's "15 %" read both ways.
//   SMPSO-9. OMOPSO's ε-dominance is relative, f_i(x1)/(1 + ε) <= f_i(x2)
//     (§3), and its archive procedure is not given; the archive here is
//     jMetal's: additive boxes floor(f/η), in one box the point nearer the
//     corner. The relative form has no box for f = 0, which the ZDT fronts
//     reach.
// Continuous variables only: a binary or categorical genome is refused.
// ============================================================================

namespace mootation {

enum class SMPSOChi { Signed, Abs, None };
enum class SMPSOTurbulence { EveryK, Prob, Thirds, Off };

template <typename Ind_t>
class SMPSOCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    // The resolved settings of one preset (THE PRESETS).
    struct Settings {
        SMPSOChi chi = SMPSOChi::Signed;
        double w_min = 0.1, w_max = 0.1;
        double c_min = 1.5, c_max = 2.5;
        double vmax_frac = 0.5;
        double vel_low = -1.0, vel_up = -1.0;
        ops::BoundRepair repair = ops::BoundRepair::Clip;
        SMPSOTurbulence turbulence = SMPSOTurbulence::EveryK;
        int turbulence_k = 6;
        double turbulence_p = 0.15;
        bool r_per_dim = false;
        bool one_shot = false;
        bool leader_random = false;
        bool offer_improved = false;
        bool eps_answer = false;
        double eps_eta = 0.0075;
    };

    static Settings preset(const std::string& name)
    {
        Settings s;
        if (name == "jmetal") return s;
        if (name == "platemo") {
            s.chi = SMPSOChi::Abs;
            s.w_min = 0.1; s.w_max = 0.5;
            s.vel_low = s.vel_up = 0.001;
            s.turbulence = SMPSOTurbulence::Prob;
            return s;
        }
        if (name == "fair") {
            s.chi = SMPSOChi::Abs;
            s.repair = ops::BoundRepair::Reflect;
            return s;
        }
        if (name == "omopso") {
            s.chi = SMPSOChi::None;
            s.w_min = 0.1; s.w_max = 0.5;
            s.c_min = 1.5; s.c_max = 2.0;
            s.vmax_frac = 0.0;
            s.turbulence = SMPSOTurbulence::Thirds;
            s.one_shot = true;
            s.offer_improved = true;
            s.eps_answer = true;
            return s;
        }
        throw std::invalid_argument("smpso_preset = '" + name + "': jmetal, platemo, fair or omopso");
    }

private:
    struct Point {                       // a particle's position, its best, a leader
        std::vector<double> f;
        double cv = 0.0;
        double crowd = 0.0;              // leaders only
        VaultPoint pt{};                 // the point whole (data_vault.hpp, WRITING BACK)
    };

    std::string preset_ = "jmetal";
    // the knobs given: applied over the preset at setup, whatever the order
    std::optional<SMPSOChi> o_chi_;
    std::optional<double> o_w_min_, o_w_max_, o_c_min_, o_c_max_, o_vmax_, o_vlow_, o_vup_,
                          o_turb_p_, o_eps_eta_;
    std::optional<ops::BoundRepair> o_repair_;
    std::optional<SMPSOTurbulence> o_turb_;
    std::optional<int> o_turb_k_, o_archive_size_;
    std::optional<bool> o_r_per_dim_, o_one_shot_, o_leader_random_, o_offer_improved_;
    double eta_m_ = 20.0, pm_ = -1.0;    // pm < 0: 1/n

    Settings s_;
    int archive_size_ = 0;
    std::vector<Point> swarm_, best_, leaders_, eps_;
    std::vector<std::vector<double>> v_;
    std::vector<double> lo_, hi_;
    int t_ = 0, t_max_ = 250;
    std::mt19937 rng_{0};

    bool dominates(const Point& a, const Point& b) const
    {
        return detail::dominates(constraint_mode, a.f, a.cv, b.f, b.cv);
    }

    double uni(double a, double b) { return rnd::uniform_real_distribution<double>(a, b)(rng_); }

    double chi(double c1, double c2) const
    {
        if (s_.chi == SMPSOChi::None) return 1.0;
        const double phi = c1 + c2;
        if (phi <= 4.0) return 1.0;                         // Eq. 4
        const double den = 2.0 - phi - std::sqrt(phi * phi - 4.0 * phi);
        return 2.0 / (s_.chi == SMPSOChi::Abs ? std::abs(den) : den);
    }

    // The crowding distance over the leaders (NSGA-II's, as jMetal's estimator:
    // all infinite at two points or fewer; an objective with no spread skipped).
    void crowding(std::vector<Point>& A) const
    {
        const std::size_t n = A.size();
        for (auto& a : A) a.crowd = 0.0;
        if (n <= 2) {
            for (auto& a : A) a.crowd = std::numeric_limits<double>::infinity();
            return;
        }
        const std::size_t m = A.front().f.size();
        std::vector<std::size_t> idx(n);
        for (std::size_t k = 0; k < m; ++k) {
            std::iota(idx.begin(), idx.end(), std::size_t{0});
            std::stable_sort(idx.begin(), idx.end(),
                             [&](std::size_t a, std::size_t b) { return A[a].f[k] < A[b].f[k]; });
            const double lo = A[idx.front()].f[k], hi = A[idx.back()].f[k];
            if (lo == hi) continue;
            A[idx.front()].crowd = A[idx.back()].crowd = std::numeric_limits<double>::infinity();
            for (std::size_t j = 1; j + 1 < n; ++j)
                A[idx[j]].crowd += (A[idx[j + 1]].f[k] - A[idx[j - 1]].f[k]) / (hi - lo);
        }
    }

    // A point offered to a non-dominated list: refused when a member dominates
    // it or has its objectives exactly (SMPSO-4); the members it dominates
    // leave. True when it entered.
    bool offer(std::vector<Point>& L, const Point& p) const
    {
        for (const auto& q : L)
            if (dominates(q, p) || q.f == p.f) return false;
        L.erase(std::remove_if(L.begin(), L.end(), [&](const Point& q) { return dominates(p, q); }),
                L.end());
        L.push_back(p);
        return true;
    }

    // The most crowded leaders out until the bound holds (archive_prune).
    void prune(std::vector<Point>& L) const
    {
        if (static_cast<int>(L.size()) <= archive_size_) return;
        crowding(L);
        if (s_.one_shot) {
            std::vector<std::size_t> idx(L.size());
            std::iota(idx.begin(), idx.end(), std::size_t{0});
            std::stable_sort(idx.begin(), idx.end(),
                             [&](std::size_t a, std::size_t b) { return L[a].crowd > L[b].crowd; });
            std::vector<Point> kept;
            for (int k = 0; k < archive_size_; ++k) kept.push_back(L[idx[static_cast<std::size_t>(k)]]);
            L = std::move(kept);
            return;
        }
        std::size_t worst = 0;                              // the first of the least crowded
        for (std::size_t k = 1; k < L.size(); ++k)
            if (L[k].crowd < L[worst].crowd) worst = k;
        L.erase(L.begin() + static_cast<std::ptrdiff_t>(worst));
    }

    // jMetal's ε-dominance comparison (EpsilonDominanceComparator, MIT): -1
    // when a ε-dominates b, 1 when b does a, 0 when neither; in one box the
    // point nearer the box's corner.
    int eps_compare(const Point& a, const Point& b) const
    {
        bool one = false, two = false;
        for (std::size_t k = 0; k < a.f.size(); ++k) {
            const double va = std::floor(a.f[k] / s_.eps_eta), vb = std::floor(b.f[k] / s_.eps_eta);
            if (va < vb) { one = true; if (two) return 0; }
            else if (vb < va) { two = true; if (one) return 0; }
        }
        if (!one && !two) {
            double da = 0.0, db = 0.0;
            for (std::size_t k = 0; k < a.f.size(); ++k) {
                const double ia = std::floor(a.f[k] / s_.eps_eta), ib = std::floor(b.f[k] / s_.eps_eta);
                da += std::pow(a.f[k] - ia * s_.eps_eta, 2.0);
                db += std::pow(b.f[k] - ib * s_.eps_eta, 2.0);
            }
            return da < db ? -1 : 1;
        }
        return two ? 1 : -1;
    }

    void offer_eps(const Point& p)
    {
        for (auto it = eps_.begin(); it != eps_.end();) {
            const int flag = eps_compare(p, *it);
            if (flag < 0) { it = eps_.erase(it); continue; }
            if (flag > 0 || it->f == p.f) return;
            ++it;
        }
        eps_.push_back(p);
    }

    void add_leaders(const std::vector<Point>& P)
    {
        for (const auto& p : P) {
            const bool in = offer(leaders_, p);
            if (in && s_.eps_answer) offer_eps(p);
            if (!s_.one_shot) prune(leaders_);
        }
        if (s_.one_shot) prune(leaders_);
        crowding(leaders_);                                  // the estimate the next step reads
    }

    const Point& leader()
    {
        rnd::uniform_int_distribution<std::size_t> any(0, leaders_.size() - 1);
        const std::size_t a = any(rng_);
        if (s_.leader_random) return leaders_[a];
        const std::size_t b = any(rng_);
        return leaders_[a].crowd >= leaders_[b].crowd ? leaders_[a] : leaders_[b];
    }

    // jMetal's uniform and non-uniform mutation (UniformMutation,
    // NonUniformMutation, MIT): probability 1/n per variable, perturbation
    // 0.5, the value clipped to the box.
    void uniform_mutation(std::vector<double>& x)
    {
        ops::note_operator("omopso_uniform", "clip");
        const double p = 1.0 / static_cast<double>(x.size());
        for (std::size_t j = 0; j < x.size(); ++j) {
            if (!(uni(0.0, 1.0) < p)) continue;
            const double r = uni(0.0, 1.0);
            x[j] = std::clamp(x[j] + (r - 0.5) * 0.5, lo_[j], hi_[j]);
        }
    }

    void non_uniform_mutation(std::vector<double>& x)
    {
        ops::note_operator("omopso_non_uniform", "clip");
        const double p = 1.0 / static_cast<double>(x.size());
        const double shrink = std::pow(1.0 - static_cast<double>(t_) / static_cast<double>(t_max_), 0.5);
        auto delta = [&](double y) { return y * (1.0 - std::pow(uni(0.0, 1.0), shrink)); };
        for (std::size_t j = 0; j < x.size(); ++j) {
            if (!(uni(0.0, 1.0) < p)) continue;
            const double r = uni(0.0, 1.0);
            const double d = r <= 0.5 ? delta(hi_[j] - x[j]) : delta(lo_[j] - x[j]);
            x[j] = std::clamp(x[j] + d, lo_[j], hi_[j]);
        }
    }

    void turbulence(std::size_t i, std::vector<double>& x, const DataVault<Ind_t>& vault)
    {
        const double pm = pm_ > 0.0 ? pm_ : 1.0 / static_cast<double>(x.size());
        switch (s_.turbulence) {
            case SMPSOTurbulence::EveryK:
                if (i % static_cast<std::size_t>(s_.turbulence_k) == 0)
                    ops::polynomial_mutation(x, vault.get_bounds(), eta_m_, pm, rng_);
                break;
            case SMPSOTurbulence::Prob:
                if (uni(0.0, 1.0) < s_.turbulence_p)
                    ops::polynomial_mutation(x, vault.get_bounds(), eta_m_, pm, rng_);
                break;
            case SMPSOTurbulence::Thirds:
                if (i % 3 == 0) non_uniform_mutation(x);
                else if (i % 3 == 1) uniform_mutation(x);
                break;
            case SMPSOTurbulence::Off:
                break;
        }
    }

    // Evaluate positions in one batch, in the slots after the answer.
    std::vector<Point> evaluate(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& X)
    {
        const int keep = static_cast<int>(vault.active_n());
        const int base = vault.expand(static_cast<int>(X.size()));
        for (std::size_t i = 0; i < X.size(); ++i)
            vault.set_variables(static_cast<std::size_t>(base) + i, X[i]);
        vault.sync();
        std::vector<Point> out(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            const std::size_t v = static_cast<std::size_t>(base) + i;
            out[i].f = vault.objectives_of(v);
            out[i].cv = constraint_mode != ConstraintMode::NONE ? vault.get_cv(v) : 0.0;
            out[i].pt = vault.point_of(v);
        }
        vault.reduce(keep);
        return out;
    }

    // The answer — the leaders, or omopso's ε-archive — into the active slots.
    void publish(DataVault<Ind_t>& vault)
    {
        const auto& A = s_.eps_answer && !eps_.empty() ? eps_ : leaders_;
        vault.reduce(0);
        vault.expand(static_cast<int>(A.size()));
        for (std::size_t i = 0; i < A.size(); ++i) vault.put_point(i, A[i].pt);
    }

    void start(DataVault<Ind_t>& vault, std::vector<Point> first)
    {
        swarm_ = std::move(first);
        best_ = swarm_;
        v_.assign(swarm_.size(), std::vector<double>(lo_.size(), 0.0));
        leaders_.clear();
        eps_.clear();
        add_leaders(swarm_);
        t_ = 1;
        publish(vault);
    }

    void resolve(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("smpso: continuous variables only");
        const auto& b = vault.get_bounds();
        lo_.resize(b.size());
        hi_.resize(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (!b[j].first || !b[j].second)
                throw std::invalid_argument("smpso: every variable needs finite bounds");
            lo_[j] = *b[j].first;
            hi_[j] = *b[j].second;
        }
        s_ = preset(preset_);
        if (o_chi_) s_.chi = *o_chi_;
        if (o_w_min_) s_.w_min = *o_w_min_;
        if (o_w_max_) s_.w_max = *o_w_max_;
        if (o_c_min_) s_.c_min = *o_c_min_;
        if (o_c_max_) s_.c_max = *o_c_max_;
        if (o_vmax_) s_.vmax_frac = *o_vmax_;
        if (o_vlow_) s_.vel_low = *o_vlow_;
        if (o_vup_) s_.vel_up = *o_vup_;
        if (o_repair_) s_.repair = *o_repair_;
        if (o_turb_) s_.turbulence = *o_turb_;
        if (o_turb_k_) s_.turbulence_k = *o_turb_k_;
        if (o_turb_p_) s_.turbulence_p = *o_turb_p_;
        if (o_r_per_dim_) s_.r_per_dim = *o_r_per_dim_;
        if (o_one_shot_) s_.one_shot = *o_one_shot_;
        if (o_leader_random_) s_.leader_random = *o_leader_random_;
        if (o_offer_improved_) s_.offer_improved = *o_offer_improved_;
        if (o_eps_eta_) s_.eps_eta = *o_eps_eta_;
        if (s_.w_min > s_.w_max || s_.c_min > s_.c_max)
            throw std::invalid_argument("smpso: w_min <= w_max and c_min <= c_max");
        archive_size_ = o_archive_size_ ? *o_archive_size_ : vault.pop_size();
    }

public:
    SMPSOCore() = default;

    void set_seed(unsigned s) { rng_.seed(s); }
    void set_t_max(int t)     { t_max_ = std::max(1, t); }
    void set_eta_mutation(double e) { eta_m_ = e; }
    void set_pm(double p)           { pm_ = p; }
    void set_bound_repair(ops::BoundRepair r)
    {
        ops::require_repair(r, "smpso", false, false);
        o_repair_ = r;
    }

    const Settings& settings() const { return s_; }
    std::size_t leaders_size() const { return leaders_.size(); }

    // Every knob as the run used it, the preset resolved (the binding hands
    // them to the caller: Result.settings, a campaign's meta.json).
    std::vector<std::pair<std::string, std::string>> effective_settings() const
    {
        auto num = [](double v) {                            // the shortest that reads back
            for (int digits = 15;; ++digits) {
                std::ostringstream o;
                o.precision(digits);
                o << v;
                if (digits == 17 || std::stod(o.str()) == v) return o.str();
            }
        };
        const char* chi = s_.chi == SMPSOChi::Signed ? "signed" : s_.chi == SMPSOChi::Abs ? "abs" : "none";
        const char* turb = s_.turbulence == SMPSOTurbulence::EveryK ? "every_k"
                         : s_.turbulence == SMPSOTurbulence::Prob   ? "prob"
                         : s_.turbulence == SMPSOTurbulence::Thirds ? "thirds" : "off";
        return {
            {"smpso_preset", preset_}, {"chi_mode", chi},
            {"w_min", num(s_.w_min)}, {"w_max", num(s_.w_max)},
            {"c_min", num(s_.c_min)}, {"c_max", num(s_.c_max)},
            {"vmax_frac", num(s_.vmax_frac)}, {"bound_repair", ops::bound_repair_name(s_.repair)},
            {"vel_on_bound_low", num(s_.vel_low)}, {"vel_on_bound_up", num(s_.vel_up)},
            {"r_per_dim", s_.r_per_dim ? "yes" : "no"},
            {"turbulence", turb}, {"turbulence_k", num(s_.turbulence_k)},
            {"turbulence_p", num(s_.turbulence_p)}, {"eta_m", num(eta_m_)},
            {"pm", pm_ > 0.0 ? num(pm_) : "1/n"},
            {"archive_size", num(archive_size_)},
            {"archive_prune", s_.one_shot ? "one_shot" : "one_by_one"},
            {"leader_select", s_.leader_random ? "random" : "crowding_tournament"},
            {"leader_offer", s_.offer_improved ? "improved" : "all"},
            {"answer", s_.eps_answer ? "eps_archive" : "leaders"},
            {"eps_archive_eta", num(s_.eps_eta)},
        };
    }

    bool set_param(const std::string& k, double v)
    {
        auto whole = [&](const char* what) {
            if (!(v >= 1.0) || v != std::floor(v) || v > 1e9)
                throw std::invalid_argument(std::string(what) + " must be a whole number >= 1");
            return static_cast<int>(v);
        };
        if (k == "vmax_frac") {
            if (!(v >= 0.0)) throw std::invalid_argument("vmax_frac must be >= 0 (0: no bound)");
            o_vmax_ = v; return true;
        }
        if (k == "vel_on_bound_low") { o_vlow_ = v; return true; }
        if (k == "vel_on_bound_up")  { o_vup_ = v; return true; }
        if (k == "w_min") { o_w_min_ = v; return true; }
        if (k == "w_max") { o_w_max_ = v; return true; }
        if (k == "c_min") { o_c_min_ = v; return true; }
        if (k == "c_max") { o_c_max_ = v; return true; }
        if (k == "turbulence_k") { o_turb_k_ = whole("turbulence_k"); return true; }
        if (k == "turbulence_p") {
            if (!(v >= 0.0 && v <= 1.0)) throw std::invalid_argument("turbulence_p must be in [0, 1]");
            o_turb_p_ = v; return true;
        }
        if (k == "archive_size") { o_archive_size_ = whole("archive_size"); return true; }
        if (k == "eps_archive_eta") {
            if (!(v > 0.0)) throw std::invalid_argument("eps_archive_eta must be > 0");
            o_eps_eta_ = v; return true;
        }
        return false;
    }

    bool set_word(const std::string& k, const std::string& w)
    {
        auto bad = [&](const char* words) {
            return std::invalid_argument(k + " = '" + w + "': " + words);
        };
        if (k == "smpso_preset") { preset(w); preset_ = w; return true; }
        if (k == "chi_mode") {
            if (w == "signed") o_chi_ = SMPSOChi::Signed;
            else if (w == "abs") o_chi_ = SMPSOChi::Abs;
            else if (w == "none") o_chi_ = SMPSOChi::None;
            else throw bad("signed, abs or none");
            return true;
        }
        if (k == "turbulence") {
            if (w == "every_k") o_turb_ = SMPSOTurbulence::EveryK;
            else if (w == "prob") o_turb_ = SMPSOTurbulence::Prob;
            else if (w == "thirds") o_turb_ = SMPSOTurbulence::Thirds;
            else if (w == "off") o_turb_ = SMPSOTurbulence::Off;
            else throw bad("every_k, prob, thirds or off");
            return true;
        }
        if (k == "r_per_dim") {
            if (w != "yes" && w != "no") throw bad("yes or no");
            o_r_per_dim_ = w == "yes"; return true;
        }
        if (k == "archive_prune") {
            if (w != "one_by_one" && w != "one_shot") throw bad("one_by_one or one_shot");
            o_one_shot_ = w == "one_shot"; return true;
        }
        if (k == "leader_select") {
            if (w != "crowding_tournament" && w != "random") throw bad("crowding_tournament or random");
            o_leader_random_ = w == "random"; return true;
        }
        if (k == "leader_offer") {
            if (w != "all" && w != "improved") throw bad("all or improved");
            o_offer_improved_ = w == "improved"; return true;
        }
        return false;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        resolve(vault);
        const std::size_t S = static_cast<std::size_t>(vault.pop_size());
        std::vector<std::vector<double>> X(S, std::vector<double>(lo_.size()));
        for (auto& x : X)
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = uni(lo_[j], hi_[j]);
        vault.reduce(0);
        start(vault, evaluate(vault, X));
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        resolve(vault);
        std::vector<Point> first;
        for (std::size_t v = 0; v < vault.active_n(); ++v) {
            Point p;
            p.f = vault.objectives_of(v);
            p.cv = constraint_mode != ConstraintMode::NONE ? vault.get_cv(v) : 0.0;
            p.pt = vault.point_of(v);
            first.push_back(std::move(p));
        }
        start(vault, std::move(first));
    }

    void step(DataVault<Ind_t>& vault)
    {
        // the points kept from the last step, brought to the vault's present
        // (data_vault.hpp, WRITING BACK): the tolerance, the level and the
        // failures' penalty move between steps
        const bool with_cv = constraint_mode != ConstraintMode::NONE;
        for (auto* P : {&swarm_, &best_, &leaders_, &eps_})
            for (auto& p : *P) refresh_copy(vault, p.pt, p.f, p.cv, with_cv);
        const std::size_t S = swarm_.size(), n = lo_.size();
        std::vector<std::vector<double>> X(S);
        for (std::size_t i = 0; i < S; ++i) {
            const Point& g = leader();                       // SMPSO-2
            const std::vector<double>& x = swarm_[i].pt.variables;
            const std::vector<double>& p = best_[i].pt.variables;
            const std::vector<double>& gx = g.pt.variables;
            double r1 = 0.0, r2 = 0.0;
            if (!s_.r_per_dim) { r1 = uni(0.0, 1.0); r2 = uni(0.0, 1.0); }
            const double c1 = uni(s_.c_min, s_.c_max), c2 = uni(s_.c_min, s_.c_max);
            const double w = s_.w_min < s_.w_max ? uni(s_.w_min, s_.w_max) : s_.w_min;
            const double k = chi(c1, c2);
            X[i] = x;
            for (std::size_t j = 0; j < n; ++j) {
                if (s_.r_per_dim) { r1 = uni(0.0, 1.0); r2 = uni(0.0, 1.0); }
                double vj = k * (w * v_[i][j] + c1 * r1 * (p[j] - x[j]) + c2 * r2 * (gx[j] - x[j]));
                if (s_.vmax_frac > 0.0) {                    // Eq. 5-6
                    const double d = s_.vmax_frac * (hi_[j] - lo_[j]);
                    vj = std::clamp(vj, -d, d);
                }
                v_[i][j] = vj;
            }
        }
        ops::note_operator("smpso_velocity", ops::bound_repair_name(s_.repair));
        for (std::size_t i = 0; i < S; ++i) {
            ops::RepairTally tally;
            for (std::size_t j = 0; j < n; ++j) {
                const double y = X[i][j] + v_[i][j];            // Eq. 1
                if (y < lo_[j] || y > hi_[j]) {
                    tally.out();
                    X[i][j] = ops::repair_value(y, lo_[j], hi_[j], X[i][j], s_.repair, rng_);
                    v_[i][j] *= y < lo_[j] ? s_.vel_low : s_.vel_up;
                } else {
                    X[i][j] = y;
                }
            }
        }
        for (std::size_t i = 0; i < S; ++i) turbulence(i, X[i], vault);
        std::vector<Point> next = evaluate(vault, X);
        std::vector<Point> offered;                          // leader_offer
        for (std::size_t i = 0; i < S; ++i)
            if (!s_.offer_improved || !dominates(best_[i], next[i])) offered.push_back(next[i]);
        add_leaders(offered);
        for (std::size_t i = 0; i < S; ++i)                  // SMPSO-3
            if (!dominates(best_[i], next[i])) best_[i] = next[i];
        swarm_ = std::move(next);
        ++t_;
        publish(vault);
    }
};

}   // namespace mootation

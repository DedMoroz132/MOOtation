#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../dss.hpp"
#include "../random.hpp"

namespace mootation {

// ============================================================================
// DMulti-MADS — mesh adaptive direct multisearch
// J. Bigeon, S. Le Digabel, L. Salomon, "DMulti-MADS: mesh adaptive direct
// multisearch for bound-constrained blackbox multiobjective optimization",
// Comput. Optim. Appl. 79(2):301-338, 2021, doi:10.1007/s10589-021-00272-9
// (the preprint, Les Cahiers du GERAD G-2020-25: dmulti-mads_bigeon2021_draft;
// §3, §4 Algorithms 2-4, §6.3).
//
// A list L of non-dominated points, each with its own frame size Δ (in the
// box normalized to [0, 1]^n); its mesh size δ = min(Δ, Δ²) (§3). One step()
// is one iteration of Algorithm 2:
//   1. the incumbent (x_k, Δ_k): of the points of the largest Δ, by
//      selectCurrentIncumbent (Algorithm 4: one of them; of two, when L has
//      two, the one of the largest objective value; else the largest γ, the
//      scaled distance to its neighbours along an objective, doubled at the
//      ends);
//   2. the search (§6.3, "speculative"): s = x_k + w_k, w_k = x_k − x_h the
//      target direction from the poll centre x_h of the iteration that found
//      x_k, when no failure has been seen at x_k; if s dominates x_k, the
//      iteration succeeds without a poll;
//   3. the poll: n + 1 directions (§6.3, after [6]) scaled to the frame and
//      rounded to the mesh, P = {x_k + δ z}, each step cut to the box (DMM-7);
//   4. success when a new point dominates x_k (the strict strategy, §6.3's
//      best); the new non-dominated points enter L by updateList (Algorithm
//      3): dominating part of L — they replace it, frame τ⁻¹Δ_k; extending L
//      (a better value of some objective than every member) — τ⁻¹Δ_k;
//      filling L — Δ_k. A failure shrinks the incumbent's Δ to τΔ.
// The list starts from DMS's line (n points on the box's diagonal, §6.3:
// "the linesearch starting strategy exposed in [21]"), its non-dominated
// points only: Algorithm 3 assumes every L^k non-dominated. The run stops when
// the incumbent's mesh size is below 1e-9 (§6.3), which is every member's,
// or at the budget. The answer: the list reduced to pop_size by DSS, as dms.
//
// THE KNOBS: mads_tau (τ, 1/2), mads_delta0 (Δ_0, 0.1 of the range),
// mads_select (spread: Algorithm 4 | first: the first of the largest Δ in
// list order), mads_poll (complete | opportunistic: stop the poll at the
// first success), mads_search (speculative | none), mads_cache (on | off;
// DMM-8).
//
// READINGS (declared). [6] = Audet, Ianni, Le Digabel, Tribes, SIAM J. Optim.
// 24(2):621-642, 2014 (n + 1 directions), checked; [8] = Audet, Le Digabel,
// Tribes, SIAM J. Optim. 29(2):1164-1189, 2019 (the granular mesh) and [1]
// (OrthoMADS) not available; the NOMAD 4.6.0 user guide read, not its code.
// PRELIMINARY: the poll (DMM-1, DMM-2) and the bounds (DMM-6, DMM-7) are our
// reading until checked against [8] and [1] as well.
//   DMM-1. The poll directions: an orthonormal basis H = I − 2vvᵀ (a
//     Householder matrix of a unit vector v drawn from the seed, as
//     OrthoMADS's [1] Halton vector, here from the run's generator), each
//     column turned towards the target direction w_k (at random without
//     one), and their negative sum: n + 1 directions, a positive basis.
//     This is [6]'s MADS(suc,neg) (§3.2: d kept where dᵀw_k >= 0, else -d;
//     d_k = −Σ d, (3.1)), NOMAD's ORTHO N+1 NEG; the paper's "n + 1
//     directions without models" (§6.3) and the guide's "DMulti-MADS ...
//     cannot use the ORTHO N+1 QUAD strategy" agree. Ours: a real Householder
//     basis rounded to the mesh, not OrthoMADS's integer one, and random
//     turns without w_k ([6] names H_k itself the simplest choice).
//   DMM-2. The mesh: the classic MADS one, δ = min(Δ, Δ²) (Algorithm 2), in
//     the normalized box; a direction u becomes the mesh vector
//     z = round(u / ‖u‖_∞ · Δ/δ), so that ‖δz‖_∞ = Δ. Not the granular mesh
//     of [8], which NOMAD and the paper's runs use.
//   DMM-3. τ = 1/2 and Δ_0 = 0.1: the paper's runs leave both to [8]'s mesh.
//   DMM-4. The target direction taken onto the current mesh by rounding; a
//     speculative point outside the box or equal to x_k is not evaluated.
//   DMM-5. Constraints: the extreme barrier (§4.1) — a point with cv > 0 is
//     infeasible and never enters L (constraint_mode FEASIBILITY; NONE
//     ignores them). An initial list without a feasible point keeps the
//     least violating one, as dms (DMS-4). The progressive barrier of
//     DMulti-MADS-PB (Bigeon et al. 2024) is not implemented.
//   DMM-6. Points outside the box: the extreme barrier, not evaluated. So
//     the sources: [6] §1 (after MADS), "infeasible trial points are
//     immediately rejected from consideration"; the NOMAD guide has bounds
//     set in the parameters, never in the blackbox, and lists NOMAD 3's
//     SNAP_TO_BOUNDS as "Not implemented" in NOMAD 4 (no projection). Near
//     a vertex or a face of the box (an optimum on the bound: ZDT, DTLZ7) an
//     iteration that evaluates no point at all is common.
//   DMM-7. A poll step that would leave the box stops at its last mesh point
//     inside: z_r is cut to the integers with 0 <= x_r + δz_r <= 1, so the
//     point stays on the mesh and in the frame; a point equal to x_k or to an
//     earlier poll point is dropped. The paper leaves the bounds to the
//     extreme barrier (§4.1) and to [6], [8]. Every direction of DMM-1 moves
//     every coordinate, so from a poll centre with k active bounds a direction
//     stays in the box with probability about 2^-k: a list on a vertex
//     (ZDT1's and DTLZ7's diagonal leave only x = 0) shrank to the stop of
//     §6.3 in a dozen iterations, after some 40 evaluations. Ours: neither
//     [6] nor the NOMAD guide cuts a step to the box; why the paper's runs
//     do not stop at such a vertex ([1]'s integer basis, [8]'s mesh?) is
//     open until those are read.
//   DMM-8 (task 15 of 2026-10-21, §8; the default since 2026-10-22). The
//     cache V^k of Algorithm 1 ("Update the cache V^{k+1}"): MADS evaluates
//     no point twice — the NOMAD 4.6.0 user guide: "before starting an
//     evaluation, the cache is scanned for the point". mads_cache = on (the
//     default) keeps every point evaluated (its normalized variables, bit
//     for bit) with its values; a trial point found there takes them and
//     spends no evaluation. Off: a point met again is evaluated again, the
//     runs before 2026-10-22 (which spent up to a fifth of 10^4 evaluations
//     so on DTLZ2_5D and MaF1_5D).
//   DMM-9 (fixed 2026-10-22, task 16). A failure shrinks the frame of the
//     member polled — found by its variables and its values. It was found by
//     its variables alone: on a noisy problem without the cache a point met
//     twice entered the list twice (other values, the same x), the first copy
//     shrank and the incumbent, the second, never did, and the run idled to
//     the runner's limit (10 000 steps evaluating nothing). With the cache
//     (DMM-8, the default) no point is evaluated twice, so the default runs
//     do not change.
// Continuous variables only.
// ============================================================================

template <typename Ind_t>
class DMultiMADSCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    struct Entry {
        std::vector<double> x;        // normalized to [0, 1]^n
        std::vector<double> f;
        double cv = 0.0;
        double delta = 0.1;           // frame size Δ
        std::vector<double> from;     // the poll centre it was found around (target direction)
        bool failed_here = false;     // a failure has been seen with it as incumbent
        VaultPoint pt{};
    };

    std::vector<Entry> list_;
    std::vector<double> lo_, hi_;
    double tau_ = 0.5, delta0_ = 0.1, tol_ = 1e-9;
    bool spread_ = true, opportunistic_ = false, speculative_ = true;
    bool cache_ = true;                                    // DMM-8
    std::map<std::vector<double>, Entry> seen_;            // V^k: the points evaluated, by x
    long long cache_hits_ = 0;
    std::mt19937 rng_{0};

    static bool dominates(const Entry& a, const Entry& b)
    {
        bool strict = false;
        for (std::size_t j = 0; j < a.f.size(); ++j) {
            if (a.f[j] > b.f[j]) return false;
            if (a.f[j] < b.f[j]) strict = true;
        }
        return strict;
    }

    bool feasible(const Entry& e) const
    {
        return constraint_mode == ConstraintMode::NONE || e.cv <= 0.0;
    }

    static double mesh(double delta) { return std::min(delta, delta * delta); }

    std::vector<double> denorm(const std::vector<double>& u) const
    {
        std::vector<double> x(u.size());
        for (std::size_t j = 0; j < u.size(); ++j) x[j] = lo_[j] + u[j] * (hi_[j] - lo_[j]);
        return x;
    }

    std::vector<Entry> evaluate(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U)
    {
        if (cache_) return evaluate_cached(vault, U);
        return evaluate_all(vault, U);
    }

    // DMM-8: the points of U in V^k take its values; the rest are evaluated
    // (in U's order) and enter it.
    std::vector<Entry> evaluate_cached(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U)
    {
        std::vector<std::vector<double>> fresh;
        for (const auto& u : U)
            if (seen_.find(u) == seen_.end()
                && std::find(fresh.begin(), fresh.end(), u) == fresh.end()) fresh.push_back(u);
        for (auto& e : evaluate_all(vault, fresh)) seen_.emplace(e.x, e);
        std::vector<Entry> out;
        for (const auto& u : U) out.push_back(seen_.at(u));
        cache_hits_ += static_cast<long long>(U.size() - fresh.size());
        return out;
    }

    std::vector<Entry> evaluate_all(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U)
    {
        std::vector<Entry> out;
        if (U.empty()) return out;
        const int keep = static_cast<int>(vault.active_n());
        const int base = vault.expand(static_cast<int>(U.size()));
        for (std::size_t i = 0; i < U.size(); ++i)
            vault.set_variables(static_cast<std::size_t>(base) + i, denorm(U[i]));
        vault.sync();
        for (std::size_t i = 0; i < U.size(); ++i) {
            const std::size_t v = static_cast<std::size_t>(base) + i;
            Entry e;
            e.x = U[i];
            e.f = vault.objectives_of(v);
            e.cv = vault.get_cv(v);
            e.pt = vault.point_of(v);
            out.push_back(std::move(e));
        }
        vault.reduce(keep);
        return out;
    }

    // Algorithm 3, the new points of one iteration (already mutually
    // non-dominated, feasible) into L with their frame sizes.
    void update_list(std::vector<Entry> add, double delta_k)
    {
        for (auto& a : add) {
            bool dominates_some = false;
            for (const auto& e : list_) dominates_some = dominates_some || dominates(a, e);
            if (dominates_some) {                                           // line 2-3
                list_.erase(std::remove_if(list_.begin(), list_.end(),
                                           [&](const Entry& e) { return dominates(a, e); }),
                            list_.end());
                a.delta = delta_k / tau_;
                list_.push_back(std::move(a));
                continue;
            }
            bool beaten = false;
            for (const auto& e : list_)
                beaten = beaten || dominates(e, a) || e.f == a.f;
            if (beaten) continue;
            bool extends = list_.empty();                                    // line 4
            for (std::size_t i = 0; i < a.f.size() && !extends; ++i) {
                double best = std::numeric_limits<double>::infinity();
                for (const auto& e : list_) best = std::min(best, e.f[i]);
                extends = a.f[i] < best;
            }
            a.delta = extends ? delta_k / tau_ : delta_k;                     // line 5 / 7
            list_.push_back(std::move(a));
        }
    }

    // Algorithm 4.
    std::size_t select_incumbent() const
    {
        double dmax = 0.0;
        for (const auto& e : list_) dmax = std::max(dmax, e.delta);
        std::vector<std::size_t> top;
        for (std::size_t i = 0; i < list_.size(); ++i)
            if (list_[i].delta == dmax) top.push_back(i);
        if (top.size() == 1 || !spread_) return top.front();
        const std::size_t m = list_.front().f.size();
        if (top.size() == 2 && list_.size() == 2) {
            std::size_t best = top[0];
            double fmax = -std::numeric_limits<double>::infinity();
            for (std::size_t i : top)
                for (double v : list_[i].f) if (v > fmax) { fmax = v; best = i; }
            return best;
        }
        // γ_i(x_j) for every member, by objective; the largest over the objectives
        std::vector<double> gamma(list_.size(), 0.0);
        std::vector<std::size_t> o(list_.size());
        for (std::size_t i = 0; i < m; ++i) {
            std::iota(o.begin(), o.end(), std::size_t{0});
            std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
                return list_[a].f[i] < list_[b].f[i];
            });
            const std::size_t L = o.size();
            const double span = list_[o.back()].f[i] - list_[o.front()].f[i];
            if (!(span > 0.0) || L < 2) continue;
            for (std::size_t t = 0; t < L; ++t) {
                double g;
                if (t == 0) g = 2.0 * (list_[o[1]].f[i] - list_[o[0]].f[i]) / span;
                else if (t == L - 1) g = 2.0 * (list_[o[L - 1]].f[i] - list_[o[L - 2]].f[i]) / span;
                else g = (list_[o[t + 1]].f[i] - list_[o[t - 1]].f[i]) / span;
                gamma[o[t]] = std::max(gamma[o[t]], g);
            }
        }
        std::size_t best = top.front();
        for (std::size_t i : top) if (gamma[i] > gamma[best]) best = i;
        return best;
    }

    // DMM-1, DMM-2: the n + 1 poll points of the frame Δ around x.
    std::vector<std::vector<double>> poll_points(const std::vector<double>& x, double Delta,
                                                 const std::vector<double>& w)
    {
        const std::size_t n = x.size();
        rnd::normal_distribution<double> N(0.0, 1.0);
        std::vector<double> v(n);
        double vv = 0.0;
        for (auto& c : v) { c = N(rng_); vv += c * c; }
        std::vector<std::vector<double>> dirs;
        std::vector<double> neg(n, 0.0);
        rnd::bernoulli_distribution coin(0.5);
        for (std::size_t c = 0; c < n; ++c) {
            std::vector<double> h(n);
            for (std::size_t r = 0; r < n; ++r)
                h[r] = (r == c ? 1.0 : 0.0) - 2.0 * v[r] * v[c] / vv;
            double dot = 0.0;
            for (std::size_t r = 0; r < n; ++r) dot += h[r] * (w.empty() ? 0.0 : w[r]);
            const bool flip = w.empty() ? coin(rng_) : dot < 0.0;
            if (flip) for (auto& t : h) t = -t;
            for (std::size_t r = 0; r < n; ++r) neg[r] -= h[r];
            dirs.push_back(std::move(h));
        }
        dirs.push_back(neg);
        const double delta = mesh(Delta);
        const double steps = std::round(Delta / delta);
        std::vector<std::vector<double>> P;
        for (const auto& d : dirs) {
            double inf = 0.0;
            for (double t : d) inf = std::max(inf, std::abs(t));
            if (!(inf > 0.0)) continue;
            std::vector<double> u = x;
            bool inside = true, moved = false;
            for (std::size_t r = 0; r < n; ++r) {
                double z = std::round(d[r] / inf * steps);
                z = std::min(z, std::floor((1.0 - x[r]) / delta));    // DMM-7: the last
                z = std::max(z, std::ceil(-x[r] / delta));            // mesh point inside
                moved = moved || z != 0.0;
                u[r] = x[r] + delta * z;
                inside = inside && u[r] >= 0.0 && u[r] <= 1.0;
            }
            const bool again = std::find(P.begin(), P.end(), u) != P.end();
            if (inside && moved && !again) P.push_back(std::move(u));  // DMM-6
        }
        return P;
    }

    void publish(DataVault<Ind_t>& vault)
    {
        std::vector<std::vector<double>> F;
        for (const auto& e : list_) F.push_back(e.f);
        const auto idx = dss::order(F, static_cast<std::size_t>(vault.pop_size()));
        vault.reduce(0);
        vault.expand(static_cast<int>(idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) vault.put_point(i, list_[idx[i]].pt);
    }

    void start(DataVault<Ind_t>& vault, std::vector<Entry> first)
    {
        list_.clear();
        std::vector<Entry> ok;
        for (auto& e : first) if (feasible(e)) ok.push_back(e);
        if (ok.empty() && !first.empty()) {                         // DMM-5
            auto best = std::min_element(first.begin(), first.end(),
                [](const Entry& a, const Entry& b) { return a.cv < b.cv; });
            best->cv = 0.0;
            ok.push_back(*best);
        }
        // L^0 non-dominated, as Algorithm 3 assumes of every L^k: a point
        // dominated by another, or with the objectives of an earlier one, is
        // left out (copied, not moved: every point is compared with every other)
        std::vector<Entry> nd;
        for (std::size_t a = 0; a < ok.size(); ++a) {
            bool beaten = false;
            for (std::size_t b = 0; b < ok.size() && !beaten; ++b)
                beaten = b != a && (dominates(ok[b], ok[a]) || (b < a && ok[b].f == ok[a].f));
            if (!beaten) {
                nd.push_back(ok[a]);
                nd.back().delta = delta0_;
            }
        }
        list_ = std::move(nd);
        publish(vault);
    }

    void read_bounds(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("dmulti_mads: continuous variables only");
        const auto& b = vault.get_bounds();
        lo_.resize(b.size());
        hi_.resize(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (!b[j].first || !b[j].second)
                throw std::invalid_argument("dmulti_mads: every variable needs finite bounds");
            lo_[j] = *b[j].first;
            hi_[j] = *b[j].second;
        }
    }

public:
    void set_seed(unsigned s) { rng_.seed(s); }

    bool set_param(const std::string& k, double v)
    {
        if (k == "mads_tau") {
            if (!(v > 0.0 && v < 1.0)) throw std::invalid_argument("mads_tau must be in (0, 1)");
            tau_ = v;
            return true;
        }
        if (k == "mads_delta0") {
            if (!(v > 0.0 && v <= 1.0)) throw std::invalid_argument("mads_delta0 must be in (0, 1]");
            delta0_ = v;
            return true;
        }
        return false;
    }

    bool set_word(const std::string& k, const std::string& w)
    {
        auto bad = [&](const char* words) {
            return std::invalid_argument(k + " = '" + w + "': " + words);
        };
        if (k == "mads_select") {
            if (w != "spread" && w != "first") throw bad("spread or first");
            spread_ = w == "spread";
            return true;
        }
        if (k == "mads_poll") {
            if (w != "complete" && w != "opportunistic") throw bad("complete or opportunistic");
            opportunistic_ = w == "opportunistic";
            return true;
        }
        if (k == "mads_search") {
            if (w != "speculative" && w != "none") throw bad("speculative or none");
            speculative_ = w == "speculative";
            return true;
        }
        if (k == "mads_cache") {                                            // DMM-8
            if (w != "off" && w != "on") throw bad("off or on");
            cache_ = w == "on";
            return true;
        }
        return false;
    }

    std::size_t list_size() const { return list_.size(); }
    long long cache_hits() const { return cache_hits_; }               // DMM-8: evaluations saved

    // The incumbent's mesh below 1e-9 (§6.3): it has the largest Δ, so every
    // member's is.
    bool finished() const
    {
        if (list_.empty()) return false;
        double dmax = 0.0;
        for (const auto& e : list_) dmax = std::max(dmax, e.delta);
        return mesh(dmax) < tol_;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        const std::size_t n = lo_.size();
        std::vector<std::vector<double>> S;
        if (n < 2) S.emplace_back(n, 0.5);
        else
            for (std::size_t i = 0; i < n; ++i)
                S.emplace_back(n, static_cast<double>(i) / static_cast<double>(n - 1));
        vault.reduce(0);
        start(vault, evaluate(vault, S));
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        std::vector<Entry> first;
        for (std::size_t v = 0; v < vault.active_n(); ++v) {
            Entry e;
            const auto& x = vault.variables_of(v);
            e.x.resize(x.size());
            for (std::size_t j = 0; j < x.size(); ++j) e.x[j] = (x[j] - lo_[j]) / (hi_[j] - lo_[j]);
            e.f = vault.objectives_of(v);
            e.cv = vault.get_cv(v);
            e.pt = vault.point_of(v);
            first.push_back(std::move(e));
        }
        start(vault, std::move(first));
    }

    void step(DataVault<Ind_t>& vault)
    {
        for (auto& e : list_) refresh_copy(vault, e.pt, e.f, e.cv, true);
        if (list_.empty() || finished()) return;
        const std::size_t k = select_incumbent();
        const Entry inc = list_[k];
        const double Delta = inc.delta;
        const double delta = mesh(Delta);
        std::vector<double> w;
        if (!inc.from.empty())
            for (std::size_t r = 0; r < inc.x.size(); ++r) w.push_back(inc.x[r] - inc.from[r]);
        std::vector<Entry> got;
        bool success = false;
        if (speculative_ && !w.empty() && !inc.failed_here) {                 // search
            std::vector<double> s = inc.x;
            bool inside = true, moved = false;
            for (std::size_t r = 0; r < s.size(); ++r) {
                const double z = std::round(w[r] / delta);                      // DMM-4
                moved = moved || z != 0.0;
                s[r] += delta * z;
                inside = inside && s[r] >= 0.0 && s[r] <= 1.0;
            }
            if (inside && moved) {
                got = evaluate(vault, {s});
                success = feasible(got.front()) && dominates(got.front(), inc);
            }
        }
        if (!success) {                                                         // poll
            const auto P = poll_points(inc.x, Delta, w);
            if (opportunistic_) {
                for (const auto& p : P) {
                    auto one = evaluate(vault, {p});
                    const bool win = feasible(one.front()) && dominates(one.front(), inc);
                    got.push_back(std::move(one.front()));
                    if (win) { success = true; break; }
                }
            } else {
                auto all = evaluate(vault, P);
                for (auto& e : all) {
                    success = success || (feasible(e) && dominates(e, inc));
                    got.push_back(std::move(e));
                }
            }
        }
        // the new feasible points, non-dominated among themselves, into L
        std::vector<Entry> add;
        for (auto& e : got) {
            if (!feasible(e)) continue;
            e.from = inc.x;
            add.push_back(std::move(e));
        }
        std::vector<Entry> nd;
        for (std::size_t a = 0; a < add.size(); ++a) {
            bool beaten = false;
            for (std::size_t b = 0; b < add.size() && !beaten; ++b)
                beaten = b != a && (dominates(add[b], add[a]) || (b < a && add[b].f == add[a].f));
            if (!beaten) nd.push_back(add[a]);
        }
        update_list(std::move(nd), Delta);
        if (!success)
            for (auto& e : list_)
                if (e.x == inc.x && e.f == inc.f) { e.delta = tau_ * Delta; e.failed_here = true; break; }   // DMM-9
        publish(vault);
    }
};

}   // namespace mootation

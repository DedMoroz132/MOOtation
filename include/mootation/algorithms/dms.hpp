#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../dss.hpp"
#include "../random.hpp"

namespace mootation {

// The initial list of DMS: one point, n on the diagonal of the box, or n drawn
// from the seed (a Latin hypercube; DMS-6).
enum class DMSInit { Line, Single, Random };

inline const char* dms_init_name(DMSInit i)
{
    return i == DMSInit::Single ? "single" : i == DMSInit::Random ? "random" : "line";
}

inline std::optional<DMSInit> parse_dms_init(const std::string& s)
{
    if (s == "line")   return DMSInit::Line;
    if (s == "single") return DMSInit::Single;
    if (s == "random") return DMSInit::Random;
    return std::nullopt;
}

// Which point of the list is polled next: the first still worth polling
// (DMS-2), or one of them at random, the poll directions in random order too
// (DMS-6).
enum class DMSPoll { First, Random };

inline const char* dms_poll_name(DMSPoll p) { return p == DMSPoll::Random ? "random" : "first"; }

inline std::optional<DMSPoll> parse_dms_poll(const std::string& s)
{
    if (s == "first")  return DMSPoll::First;
    if (s == "random") return DMSPoll::Random;
    return std::nullopt;
}

// ============================================================================
// DMS — Direct MultiSearch for multiobjective optimization
// A. L. Custódio, J. F. A. Madeira, A. I. F. Vaz, L. N. Vicente,
// SIAM J. Optim. 21(3):1109-1140, 2011, doi:10.1137/10079731X
//                                   (source: dms_custodio2011, the preprint)
//
// The first method in the library that is not evolutionary. Its state is a
// LIST L of nondominated points, each with its own step size α, not a
// population (Algorithm 3.1):
//   1. select a poll centre (x_k; α_k) from L;
//   2. (search step: none — the paper's DMS variants in §6 use none)
//   3. poll: evaluate F at x_k + α_k·d for d in D = [I  −I] (2n points), and
//      filter L ∪ L_add to its nondominated points (Algorithm 3.2);
//      L_trial = L_filtered (Algorithm 3.3);
//   4. success (L_trial ≠ L_k): the new points join with step α_k and α_k is
//      kept (γ = 1); failure: α_k ← α_k/2 (β1 = β2 = 1/2);
//   the poll centre moves to the end of the list (§6.1: "always add points to
//   the end of the list and move a point already selected as a poll center to
//   the end of the list (at the end of an iteration)").
// Stops when every α in L is below α_ε = 10^-3 (§6.1) — finished() — or when
// the budget runs out. One step() is one poll. Initial list (§6.1): DMS(n,line),
// "the best variant", by default — S_0 = {ℓ + i/(n−1)(u − ℓ), i = 0..n−1},
// its nondominated points with α_0 = 1 — or DMS(1), x_0 = (u + ℓ)/2
// (set_init / knob dms_init). Points outside the box are infeasible under the
// extreme barrier (2) and are NOT evaluated (their F would be +∞).
//
// The answer is the list; compared with a population it is reduced to N by
// the same DSS selection every other set in the library goes through
// (dss.hpp): the active population after every step is that selection, so the
// observers, the trajectory and the final result all read it. While the list
// is shorter than N the answer is the whole list, fewer than N points.
//
// Declared deviations and readings:
//   DMS-1 (task, 2026-09-23). The variables are normalised to [0, 1]^n: poll
//     steps are α·(u − ℓ) per variable, α_0 = 1, α_ε = 10^-3 of the range.
//     The paper polls in the problem's own units with α_0 = 1, which on a
//     [−5, 5] box is a tenth of the range and on [0, 1] all of it; the
//     normalisation makes the step scale-free, as every other operator here.
//   DMS-2. The poll centre is the first point IN LIST ORDER whose α is at
//     least α_ε. The paper orders the list and picks a point, and stops only
//     when every α is below α_ε; it does not say whether a converged point at
//     the head is polled again. Polling it would spend 2n evaluations on steps
//     the stopping rule already considers exhausted.
//   DMS-3. The paper evaluates every feasible poll point, including one equal
//     to a point already in the list (its SP1 example, iteration 1, does);
//     so does this port. The union then keeps one copy ("the set union should
//     not allow element repetition").
//   DMS-4. Constraints: with constraint_mode FEASIBILITY a point with cv > 0
//     is infeasible and filtered out, the extreme barrier (2); with NONE the
//     limits are ignored, as in every core. An initial list with no feasible
//     point keeps the least violating one, so that polling can start.
//   DMS-5. Continuous variables only; a problem with binary ones is refused.
//   DMS-6 (task 5, 2026-10-01; off by default). The method has no random
//     number, so its ten seeds of a campaign are one run, and the line's
//     points and their dyadic steps (α = 1, 1/2, 1/4, ... of the range) land
//     exactly on the middle and the bounds of the box, where DTLZ1-4 and ZDT
//     have their optima. To tell its search from those two: dms_init = random
//     starts from a Latin hypercube of n points drawn from the seed (one point
//     when n < 2), and dms_poll = random polls a point drawn among those still
//     worth polling, in a random order of the directions. Neither is in the
//     paper.
//   DMS-7 (task 16, supplement 1; task 17, supplement 2). A cache: a point
//     polled again takes its values from the run's cache of every point
//     evaluated, as DMulti-MADS's V^k (DMM-8), brought to the vault's present
//     (the tolerance, the level, the failures' penalty), and costs no
//     evaluation; its step size is the poll's, as a new point's would be.
//     On by default (dms_cache = on; the owner's decision of 2026-10-23,
//     TASK16a §5: IGD+ better on 8 of 11 problems at 10^4): a departure from
//     the paper, which keeps no cache — Algorithm 3.1 evaluates every poll
//     set, Appendix A polls x_0 again at iteration 1 — and where on DTLZ2_3D
//     at 2 000 evaluations about three calls in five ask for a point already
//     evaluated. The same points in the same order as without it (only the
//     budget goes further), but where the run itself reads the count of
//     evaluations (eq_tol_schedule = linear): TASK17 cache_equivalence.
//     dms_cache = off evaluates them again, as the paper.
// ============================================================================
template <typename Ind_t>
class DMSCore {
public:
    // Supported: NONE, FEASIBILITY (the extreme barrier); CDP and
    // EPS_CONSTRAINT act as FEASIBILITY: a list has no ranking to shift
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    struct Entry {
        std::vector<double> x;   // normalised to [0, 1]^n
        std::vector<double> f;
        double cv = 0.0;
        double alpha = 1.0;
        VaultPoint pt{};           // the point as the vault holds it (data_vault.hpp, WRITING BACK)
    };

    std::vector<Entry> list_;
    DMSInit init_  = DMSInit::Line;
    DMSPoll poll_  = DMSPoll::First;
    double  alpha0_ = 1.0;
    double  tol_    = 1e-3;
    std::vector<double> lo_, hi_;
    std::mt19937 rng_{0};           // drawn from only by DMS-6's knobs
    bool cache_ = true;                                    // DMS-7, as the run uses it
    std::string cache_mode_ = "on";                        // dms_cache: on, off, force
    std::string cache_note_ = "on";
    bool cache_check_ = false;                             // every hit evaluated again (a check)
    std::map<std::vector<double>, Entry> seen_;            // every point evaluated, by x
    long long cache_hits_ = 0;

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

    std::vector<double> denorm(const std::vector<double>& u) const
    {
        std::vector<double> x(u.size());
        for (std::size_t j = 0; j < u.size(); ++j) x[j] = lo_[j] + u[j] * (hi_[j] - lo_[j]);
        return x;
    }

    // Evaluate points (normalised) in one batch: slots after the answer set;
    // with the cache (DMS-7) only those not evaluated before, the others
    // taken from it, in U's order either way.
    std::vector<Entry> evaluate(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U,
                                double alpha)
    {
        if (!cache_) return evaluate_all(vault, U, alpha);
        std::vector<std::vector<double>> fresh;
        for (const auto& u : U)
            if (seen_.find(u) == seen_.end()
                && std::find(fresh.begin(), fresh.end(), u) == fresh.end()) fresh.push_back(u);
        for (auto& e : evaluate_all(vault, fresh, alpha)) seen_.emplace(e.x, e);
        std::vector<Entry> out;
        for (const auto& u : U) {
            Entry e = seen_.at(u);
            e.alpha = alpha;
            if (std::find(fresh.begin(), fresh.end(), u) == fresh.end()) {
                // a hit, evaluated at an earlier step: brought to the vault's
                // present as the list is (WRITING BACK; task 17, supplement 2)
                refresh_copy(vault, e.pt, e.f, e.cv, true);
                if (cache_check_) {
                    const Entry n = evaluate_all(vault, {u}, alpha).front();
                    check_cache_hit("dms", e.f, e.cv, e.pt.failed, n.f, n.cv, n.pt.failed);
                }
            }
            out.push_back(std::move(e));
        }
        cache_hits_ += static_cast<long long>(U.size() - fresh.size());
        return out;
    }

    std::vector<Entry> evaluate_all(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U,
                                    double alpha)
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
            e.alpha = alpha;
            out.push_back(std::move(e));
        }
        vault.reduce(keep);
        return out;
    }

    // Algorithm 3.2: L ∪ add filtered to its nondominated points, the list's
    // order kept and the added points appended in their order; returns
    // whether any added point survived.
    bool filter(std::vector<Entry> add)
    {
        std::vector<char> alive(add.size(), 1);
        for (std::size_t a = 0; a < add.size(); ++a) {
            if (!feasible(add[a])) { alive[a] = 0; continue; }
            for (const auto& e : list_)
                if (e.x == add[a].x || dominates(e, add[a])) { alive[a] = 0; break; }
            for (std::size_t b = 0; b < a && alive[a]; ++b)
                if (alive[b] && (add[b].x == add[a].x || dominates(add[b], add[a]))) alive[a] = 0;
            if (!alive[a]) continue;
            for (std::size_t b = 0; b < a; ++b)
                if (alive[b] && dominates(add[a], add[b])) alive[b] = 0;
        }
        std::vector<Entry> kept;
        for (auto& e : list_) {
            bool beaten = false;
            for (std::size_t a = 0; a < add.size() && !beaten; ++a)
                beaten = alive[a] && dominates(add[a], e);
            if (!beaten) kept.push_back(std::move(e));
        }
        bool any = false;
        for (std::size_t a = 0; a < add.size(); ++a)
            if (alive[a]) { kept.push_back(std::move(add[a])); any = true; }
        list_ = std::move(kept);
        return any;
    }

    // The active population := the DSS selection of N from the list.
    void publish(DataVault<Ind_t>& vault)
    {
        std::vector<std::vector<double>> F;
        F.reserve(list_.size());
        for (const auto& e : list_) F.push_back(e.f);
        const auto idx = dss::order(F, static_cast<std::size_t>(vault.pop_size()));
        vault.reduce(0);
        vault.expand(static_cast<int>(idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const Entry& e = list_[idx[i]];
            vault.put_point(i, e.pt);
            if constexpr (std::is_base_of_v<DMS_Individual, Ind_t>)
                vault.get_ind(i).alpha = e.alpha;
        }
    }

    void start(DataVault<Ind_t>& vault, std::vector<Entry> first)
    {
        list_.clear();
        std::vector<Entry> ok;
        for (auto& e : first) if (feasible(e)) ok.push_back(e);
        if (ok.empty() && !first.empty()) {             // DMS-4: the least violating
            auto best = std::min_element(first.begin(), first.end(),
                [](const Entry& a, const Entry& b) { return a.cv < b.cv; });
            best->cv = 0.0;
            ok.push_back(*best);
        }
        filter(std::move(ok));
        for (auto& e : list_) e.alpha = alpha0_;
        publish(vault);
    }

    void read_bounds(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("dms: continuous variables only (DMS-5)");
        const auto& b = vault.get_bounds();
        lo_.resize(b.size());
        hi_.resize(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (!b[j].first || !b[j].second)
                throw std::invalid_argument("dms: every variable needs finite bounds");
            lo_[j] = *b[j].first;
            hi_[j] = *b[j].second;
        }
    }

public:
    // No random number anywhere in the method: the seed is accepted, as every
    // core's is, and changes nothing at the defaults (tests/
    // test_reproducibility.cpp holds it to that); only DMS-6's knobs draw.
    static constexpr bool deterministic = true;

    DMSCore() = default;

    void set_seed(unsigned s)       { rng_.seed(s); }
    void set_init(DMSInit i)        { init_ = i; }
    void set_poll(DMSPoll p)        { poll_ = p; }
    void set_step_tolerance(double t) { tol_ = t; }

    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "dms_cache") {                                            // DMS-7
            if (w != "on" && w != "off" && w != "force")
                throw std::invalid_argument("dms_cache = '" + w + "': on, off or force");
            cache_mode_ = w;
            return true;
        }
        if (k == "cache_check") {
            if (w != "on" && w != "off")
                throw std::invalid_argument("cache_check = '" + w + "': on or off");
            cache_check_ = w == "on";
            return true;
        }
        return false;
    }
    long long cache_hits() const { return cache_hits_; }               // DMS-7: evaluations saved
    std::vector<std::pair<std::string, std::string>> effective_settings() const
    {
        return {{"dms_cache", cache_note_}};
    }

    // The list itself (normalised variables, objectives, step sizes).
    std::size_t list_size() const { return list_.size(); }
    double      alpha_of(std::size_t i) const { return list_.at(i).alpha; }

    // Every step size below α_ε: the paper's stopping rule.
    bool finished() const
    {
        for (const auto& e : list_) if (e.alpha >= tol_) return false;
        return !list_.empty();
    }

    // The cache as the run uses it: off on a stochastic problem but where
    // forced (task 17, supplement 2, §1.3; problem.hpp, is_stochastic).
    void choose_cache(const DataVault<Ind_t>& vault)
    {
        cache_ = cache_mode_ == "force" || (cache_mode_ == "on" && !vault.stochastic());
        cache_note_ = cache_mode_ == "on" && !cache_ ? "off: the problem is stochastic"
                                                    : cache_ ? "on" : "off";
    }

    void setup(DataVault<Ind_t>& vault)
    {
        choose_cache(vault);
        read_bounds(vault);
        const std::size_t n = lo_.size();
        std::vector<std::vector<double>> S;
        if (init_ == DMSInit::Random) {                  // DMS-6: a Latin hypercube
            const std::size_t k = n < 2 ? 1 : n;
            S.assign(k, std::vector<double>(n));
            rnd::uniform_real_distribution<double> u(0.0, 1.0);
            std::vector<std::size_t> perm(k);
            for (std::size_t j = 0; j < n; ++j) {
                std::iota(perm.begin(), perm.end(), std::size_t{0});
                rnd::shuffle(perm.begin(), perm.end(), rng_);
                for (std::size_t i = 0; i < k; ++i)
                    S[i][j] = (static_cast<double>(perm[i]) + u(rng_)) / static_cast<double>(k);
            }
        } else if (init_ == DMSInit::Single || n < 2) {
            S.emplace_back(n, 0.5);                      // x_0 = (u + ℓ)/2
        } else {
            for (std::size_t i = 0; i < n; ++i)
                S.emplace_back(n, static_cast<double>(i) / static_cast<double>(n - 1));
        }
        vault.reduce(0);
        start(vault, evaluate(vault, S, alpha0_));
    }

    // A warm start: the seeded population's nondominated points, α_0 each.
    void setup_seeded(DataVault<Ind_t>& vault)
    {
        choose_cache(vault);
        read_bounds(vault);
        std::vector<Entry> first;
        for (std::size_t v = 0; v < vault.active_n(); ++v) {
            Entry e;
            const auto& x = vault.variables_of(v);
            e.x.resize(x.size());
            for (std::size_t j = 0; j < x.size(); ++j)
                e.x[j] = (x[j] - lo_[j]) / (hi_[j] - lo_[j]);
            e.f = vault.objectives_of(v);
            e.cv = vault.get_cv(v);
            e.pt = vault.point_of(v);
            first.push_back(std::move(e));
        }
        start(vault, std::move(first));
    }

    void step(DataVault<Ind_t>& vault)
    {
        // the list kept from the last step, brought to the vault's present
        // (data_vault.hpp, WRITING BACK): the tolerance, the level and the
        // failures' penalty move between steps
        for (auto& e : list_) refresh_copy(vault, e.pt, e.f, e.cv, true);
        // DMS-2: the first point in list order still worth polling; DMS-6:
        // one of them at random
        std::vector<std::size_t> open;
        for (std::size_t i = 0; i < list_.size(); ++i)
            if (list_[i].alpha >= tol_) {
                open.push_back(i);
                if (poll_ == DMSPoll::First) break;
            }
        if (open.empty()) return;                          // finished()
        const std::size_t k = poll_ == DMSPoll::First ? open.front()
            : open[rnd::uniform_int_distribution<std::size_t>(0, open.size() - 1)(rng_)];
        const Entry centre = list_[k];
        const std::size_t n = centre.x.size();
        std::vector<std::vector<double>> P;
        for (int sign : {+1, -1})                           // D = [I  −I]
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<double> u = centre.x;
                u[j] += sign * centre.alpha;
                if (u[j] < 0.0 || u[j] > 1.0) continue;     // extreme barrier (2)
                P.push_back(std::move(u));
            }
        if (poll_ == DMSPoll::Random) rnd::shuffle(P.begin(), P.end(), rng_);
        const bool success = filter(evaluate(vault, P, centre.alpha));
        // the centre: α kept on success, halved on failure; to the end of the list
        for (std::size_t i = 0; i < list_.size(); ++i) {
            if (list_[i].x != centre.x) continue;
            Entry c = list_[i];
            if (!success) c.alpha *= 0.5;
            list_.erase(list_.begin() + static_cast<std::ptrdiff_t>(i));
            list_.push_back(std::move(c));
            break;
        }
        publish(vault);
    }
};

}   // namespace mootation

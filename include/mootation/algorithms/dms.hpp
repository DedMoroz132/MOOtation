#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../dss.hpp"

namespace mootation {

// The initial list of DMS: one point, or n on the diagonal of the box.
enum class DMSInit { Line, Single };

inline const char* dms_init_name(DMSInit i) { return i == DMSInit::Single ? "single" : "line"; }

inline std::optional<DMSInit> parse_dms_init(const std::string& s)
{
    if (s == "line")   return DMSInit::Line;
    if (s == "single") return DMSInit::Single;
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
    };

    std::vector<Entry> list_;
    DMSInit init_  = DMSInit::Line;
    double  alpha0_ = 1.0;
    double  tol_    = 1e-3;
    std::vector<double> lo_, hi_;

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

    // Evaluate points (normalised) in one batch: slots after the answer set.
    std::vector<Entry> evaluate(DataVault<Ind_t>& vault, const std::vector<std::vector<double>>& U,
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
            vault.seed_individual(i, denorm(e.x), e.f);
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
    // core's is, and changes nothing (tests/test_reproducibility.cpp holds it
    // to that).
    static constexpr bool deterministic = true;

    DMSCore() = default;

    void set_seed(unsigned)         {}
    void set_init(DMSInit i)        { init_ = i; }
    void set_step_tolerance(double t) { tol_ = t; }

    // The list itself (normalised variables, objectives, step sizes).
    std::size_t list_size() const { return list_.size(); }
    double      alpha_of(std::size_t i) const { return list_.at(i).alpha; }

    // Every step size below α_ε: the paper's stopping rule.
    bool finished() const
    {
        for (const auto& e : list_) if (e.alpha >= tol_) return false;
        return !list_.empty();
    }

    void setup(DataVault<Ind_t>& vault)
    {
        read_bounds(vault);
        const std::size_t n = lo_.size();
        std::vector<std::vector<double>> S;
        if (init_ == DMSInit::Single || n < 2) {
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
            first.push_back(std::move(e));
        }
        start(vault, std::move(first));
    }

    void step(DataVault<Ind_t>& vault)
    {
        // DMS-2: the first point in list order still worth polling
        std::size_t k = list_.size();
        for (std::size_t i = 0; i < list_.size(); ++i)
            if (list_[i].alpha >= tol_) { k = i; break; }
        if (k == list_.size()) return;                     // finished()
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

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Crossovers beyond SBX (task 2 of 2026-09-23, B2), two children per pair.
//
//   uniform    every variable of the first child from one parent, the
//              second child's from the other, the parent chosen by a fair
//              coin per variable — the shuffle_random mechanism of JEGA
//              (Dakota's MOGA) for two parents; cannot leave the box
//   blx_alpha  every variable drawn uniformly from [min − α·I, max + α·I],
//              I = |x1_i − x2_i| (Eshelman & Schaffer, "Real-Coded Genetic
//              Algorithms and Interval-Schemata", FOGA 2, 1993, pp. 187-202,
//              §2 and Fig. 1, in words; source blx-alpha_eshelman1993); α = 0.5 by
//              default, the value at which a child is as likely outside its
//              parents as between them (§3.2), the one §4.2 runs. Two children
//              per mating (§4.1), each on its own draws — this library's
//              reading. The paper says nothing of a value outside the range:
//              here it is repaired (bound_repair.hpp, reflect by default,
//              provisionally; resample redraws it; the parent for midpoint is
//              the child's own, x1 for the first)
//
// Both cross the pair with probability pc, as SBX does, and burn no random
// number for that when pc >= 1.
//
// Multi-parent (multi_parent.hpp, formulas and sources there): spx and rex
// take n + 1 parents, undx and pcx three. A host with pairwise mating calls
// apply_any: the pair it selected is crossed with probability pc, as above,
// and only then are the other parents drawn with the host's own selection,
// each one different from those already taken (MP-6); two children come out
// (undx its symmetric pair, the others two independent draws). apply() takes
// two parents only and refuses these kinds. The default SBX path draws
// nothing more than before.
//
// de and de_pbest (de_crossover.hpp) take the pair and two more parents the
// same way; portfolio draws one of several kinds per mating (below).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unordered_map>

#include "bound_repair.hpp"
#include "de_crossover.hpp"
#include "learned_coords.hpp"
#include "multi_parent.hpp"
#include "sbx.hpp"
#include "../random.hpp"

namespace mootation::ops {

enum class Crossover { SBX, Uniform, BlxAlpha, SPX, REX, UNDX, PCX, DE, DEPBest, Portfolio,
                       None };

inline const char* crossover_name(Crossover c)
{
    switch (c) {
        case Crossover::SBX:      return "sbx";
        case Crossover::Uniform:  return "uniform";
        case Crossover::BlxAlpha: return "blx_alpha";
        case Crossover::SPX:      return "spx";
        case Crossover::REX:      return "rex";
        case Crossover::UNDX:     return "undx";
        case Crossover::PCX:      return "pcx";
        case Crossover::DE:       return "de";
        case Crossover::DEPBest:  return "de_pbest";
        case Crossover::Portfolio: return "portfolio";
        case Crossover::None:     return "none";
    }
    return "?";
}

inline std::optional<Crossover> parse_crossover(const std::string& s)
{
    for (Crossover c : {Crossover::SBX, Crossover::Uniform, Crossover::BlxAlpha,
                        Crossover::SPX, Crossover::REX, Crossover::UNDX, Crossover::PCX,
                        Crossover::DE, Crossover::DEPBest, Crossover::Portfolio,
                        Crossover::None})
        if (s == crossover_name(c)) return c;
    return std::nullopt;
}

template <typename RNG>
inline void uniform_crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                              std::vector<double>& c1, std::vector<double>& c2,
                              double pc, RNG& rng)
{
    note_operator("uniform", "none");
    rnd::uniform_real_distribution<double> uni(0.0, 1.0);
    c1 = p1; c2 = p2;
    if (pc < 1.0 && uni(rng) > pc) return;
    for (std::size_t j = 0; j < p1.size(); ++j)
        if (uni(rng) < 0.5) std::swap(c1[j], c2[j]);
}

template <typename RNG>
inline void blx_alpha_crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                                std::vector<double>& c1, std::vector<double>& c2,
                                const std::vector<std::pair<std::optional<double>,
                                                            std::optional<double>>>& bounds,
                                double alpha, double pc, BoundRepair repair, RNG& rng)
{
    require_repair(repair, "blx_alpha", true, false);
    note_operator("blx_alpha", bound_repair_name(repair));
    rnd::uniform_real_distribution<double> uni(0.0, 1.0);
    c1 = p1; c2 = p2;
    if (pc < 1.0 && uni(rng) > pc) return;
    RepairTally t1, t2;
    for (std::size_t j = 0; j < p1.size(); ++j) {
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        const double I  = std::abs(p1[j] - p2[j]);
        const double a  = std::min(p1[j], p2[j]) - alpha * I;
        const double b  = std::max(p1[j], p2[j]) + alpha * I;
        auto child = [&](double parent, RepairTally& tally) {
            auto draw = [&] { return a + (b - a) * uni(rng); };
            double v = draw();
            if (v >= lo && v <= hi) return v;
            tally.out();
            if (repair == BoundRepair::Resample) {
                for (int t = 0; t < resample_tries() && (v < lo || v > hi); ++t) v = draw();
                return std::clamp(v, lo, hi);
            }
            return repair_value(v, lo, hi, parent, repair, rng);
        };
        c1[j] = child(p1[j], t1);
        c2[j] = child(p2[j], t2);
    }
}

// ── the choice, as the switchable algorithms hold it ────────────────────────
// PORTFOLIO (task 7 of 2026-10-10, §2.4): crossover = "portfolio" draws the
// operator of every mating from the list `portfolio` (sbx, de, pcx and undx by
// default; any of the kinds above but portfolio itself), both children of a
// mating from the same one, by `portfolio_credit`:
//   uniform  every operator alike;
//   borg     probability ∝ (the operator's children in the archive + 1), the
//            archive the non-dominated set of the children the host's
//            selection kept, each tagged with its operator — after Borg
//            (Hadka & Reed, Evol. Comput. 21(2):231-259, 2013), whose archive
//            is an ε-box archive of every offspring: here no ε, and only the
//            children that survived (not checked against the paper yet).
// The operator of every child goes to portfolio_record(): its children and
// how many of them the selection kept, which operator_stats() reports.
enum class PortfolioCredit { Uniform, Borg };

inline std::optional<PortfolioCredit> parse_portfolio_credit(const std::string& s)
{
    if (s == "uniform") return PortfolioCredit::Uniform;
    if (s == "borg")    return PortfolioCredit::Borg;
    return std::nullopt;
}

struct PortfolioCount { long long children = 0, kept = 0, t_children = 0, t_kept = 0; };
// {operator name: counts since the binding last read them}, thread-local.
inline std::vector<std::pair<std::string, PortfolioCount>>& portfolio_record()
{
    thread_local std::vector<std::pair<std::string, PortfolioCount>> r;
    return r;
}
inline PortfolioCount& portfolio_count(const char* op)
{
    auto& r = portfolio_record();
    for (auto& kv : r) if (kv.first == op) return kv.second;
    r.emplace_back(op, PortfolioCount{});
    return r.back().second;
}

// The records of de, de_pbest and portfolio, cleared at the start of a run.
inline void reset_crossover_records()
{
    de_record() = DERecord{};
    portfolio_record().clear();
    coords_record() = CoordsRecord{};
}

// {name: value} of the records since the last call (totals: over the run),
// the read counters reset: the de_* and portfolio_* entries of the binding's
// operator_stats() and operator_totals.
inline std::vector<std::pair<std::string, double>> crossover_stats(bool totals)
{
    std::vector<std::pair<std::string, double>> out;
    auto& d = de_record();
    if (d.used) {
        auto quart = [&](std::vector<double> v, const char* name) {
            if (v.empty()) return;
            std::stable_sort(v.begin(), v.end());
            auto q = [&](double t) {
                const double x = t * static_cast<double>(v.size() - 1);
                const std::size_t i = static_cast<std::size_t>(x);
                const double f = x - static_cast<double>(i);
                return i + 1 < v.size() ? v[i] * (1.0 - f) + v[i + 1] * f : v[i];
            };
            double mean = 0.0;
            for (double x : v) mean += x;
            out.emplace_back(std::string(name) + "_mean", mean / static_cast<double>(v.size()));
            out.emplace_back(std::string(name) + "_q1", q(0.25));
            out.emplace_back(std::string(name) + "_q3", q(0.75));
        };
        quart(d.mcr, "de_mu_cr");
        quart(d.mf, "de_mu_f");
        const long long n = totals ? d.t_drawn : d.drawn, k = totals ? d.t_ok : d.ok;
        if (n > 0) {
            out.emplace_back("de_cr_mean", (totals ? d.t_cr_sum : d.cr_sum) / static_cast<double>(n));
            out.emplace_back("de_f_mean", (totals ? d.t_f_sum : d.f_sum) / static_cast<double>(n));
        }
        if (k > 0) {
            out.emplace_back("de_cr_success_mean", (totals ? d.t_cr_ok_sum : d.cr_ok_sum) / static_cast<double>(k));
            out.emplace_back("de_f_success_mean", (totals ? d.t_f_ok_sum : d.f_ok_sum) / static_cast<double>(k));
        }
        if (!totals) { d.drawn = d.ok = 0; d.cr_sum = d.f_sum = d.cr_ok_sum = d.f_ok_sum = 0.0; }
    }
    long long all = 0;
    for (const auto& kv : portfolio_record()) all += totals ? kv.second.t_children : kv.second.children;
    for (auto& kv : portfolio_record()) {
        const long long c = totals ? kv.second.t_children : kv.second.children;
        const long long k = totals ? kv.second.t_kept : kv.second.kept;
        if (all > 0) out.emplace_back("portfolio_share_" + kv.first, static_cast<double>(c) / static_cast<double>(all));
        if (c > 0) out.emplace_back("portfolio_kept_" + kv.first, static_cast<double>(k) / static_cast<double>(c));
        if (!totals) kv.second.children = kv.second.kept = 0;
    }
    // coords (learned_coords.hpp): the spectrum of the last update — the
    // largest eigenvalue over the smallest one above 1e-12 of it, and the
    // share of the first M - 1 components
    const auto& co = coords_record();
    if (co.active && !co.eig.empty()) {
        double sum = 0.0, lead = 0.0, low = co.eig.front();
        for (std::size_t i = 0; i < co.eig.size(); ++i) {
            sum += std::max(co.eig[i], 0.0);
            if (static_cast<int>(i) < co.m - 1) lead += std::max(co.eig[i], 0.0);
            if (co.eig[i] > 1e-12 * co.eig.front()) low = co.eig[i];
        }
        if (co.eig.front() > 0.0) {
            out.emplace_back("coords_ratio", co.eig.front() / low);
            out.emplace_back("coords_lead_share", lead / sum);
        }
        out.emplace_back("coords_updates", static_cast<double>(co.updates));
    }
    return out;
}

struct CrossoverSpec {
    Crossover   kind   = Crossover::SBX;
    double      alpha  = 0.5;                      // blx_alpha
    BoundRepair repair = BoundRepair::Reflect;     // provisional, see real_mutation.hpp
    DEState     de;                                // de, de_pbest (de_crossover.hpp)
    std::vector<Crossover> portfolio = {Crossover::SBX, Crossover::DE, Crossover::PCX,
                                        Crossover::UNDX};
    PortfolioCredit credit = PortfolioCredit::Uniform;
    LearnedCoords coords;                          // coords (learned_coords.hpp, variation.hpp)

    bool multi() const { return is_multi(kind); }

    static bool is_multi(Crossover k)
    {
        return k == Crossover::SPX || k == Crossover::REX ||
               k == Crossover::UNDX || k == Crossover::PCX;
    }

    // Parents one application takes: n + 1 for spx and rex, 3 for undx and
    // pcx, 4 for de and de_pbest.
    int parents(int n) const { return parents_of(kind, n); }

    static int parents_of(Crossover k, int n)
    {
        if (k == Crossover::SPX || k == Crossover::REX) return n + 1;
        if (k == Crossover::UNDX || k == Crossover::PCX) return 3;
        if (k == Crossover::DE || k == Crossover::DEPBest) return 4;
        return 2;
    }

    // Does the run need observe() and finish() at every step (de, de_pbest,
    // portfolio)?
    bool observing() const
    {
        return kind == Crossover::DE || kind == Crossover::DEPBest ||
               kind == Crossover::Portfolio;
    }

    // The knobs of de, de_pbest and portfolio (settings.hpp): the core's
    // set_param / set_word hand them here; false: not one of them.
    bool set_param(const std::string& k, double v)
    {
        if (coords.set_param(k, v)) return true;
        if (k == "de_p") {
            if (!(v > 0.0 && v <= 1.0)) throw std::invalid_argument("de_p must be in (0, 1]");
            de.set.p = v;
            return true;
        }
        if (k == "de_memory") {
            if (!(v >= 1.0) || v != std::floor(v))
                throw std::invalid_argument("de_memory must be an integer >= 1");
            de.set.memory = static_cast<int>(v);
            return true;
        }
        return false;
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (coords.set_word(k, w)) return true;
        if (k == "de_adapt") {
            auto a = parse_de_adapt(w);
            if (!a) throw std::invalid_argument("de_adapt = '" + w + "': none or shade");
            de.set.adapt = *a;
            return true;
        }
        if (k == "de_adapt_weight") {
            auto a = parse_de_weight(w);
            if (!a) throw std::invalid_argument("de_adapt_weight = '" + w + "': equal or contribution");
            de.set.weight = *a;
            return true;
        }
        if (k == "de_archive") {
            if (w != "yes" && w != "no") throw std::invalid_argument("de_archive = '" + w + "': yes or no");
            de.set.archive = w == "yes";
            return true;
        }
        if (k == "portfolio_credit") {
            auto c = parse_portfolio_credit(w);
            if (!c) throw std::invalid_argument("portfolio_credit = '" + w + "': uniform or borg");
            credit = *c;
            return true;
        }
        if (k == "portfolio") {                  // comma-separated kinds
            std::vector<Crossover> v;
            std::size_t a = 0;
            while (a <= w.size()) {
                const std::size_t b = std::min(w.find(',', a), w.size());
                std::size_t x = a, y = b;              // the item without spaces
                while (x < y && w[x] == ' ') ++x;
                while (y > x && w[y - 1] == ' ') --y;
                const std::string item = w.substr(x, y - x);
                auto c = parse_crossover(item);
                if (!c || *c == Crossover::Portfolio)
                    throw std::invalid_argument("portfolio: '" + item + "' is not a crossover");
                v.push_back(*c);
                a = b + 1;
            }
            if (v.empty()) throw std::invalid_argument("portfolio is empty");
            portfolio = std::move(v);
            return true;
        }
        return false;
    }

    template <typename RNG>
    void apply(const std::vector<double>& p1, const std::vector<double>& p2,
               std::vector<double>& c1, std::vector<double>& c2,
               const std::vector<std::pair<std::optional<double>,
                                           std::optional<double>>>& bounds,
               double eta_c, double pc, RNG& rng) const
    {
        apply_two(kind, p1, p2, c1, c2, bounds, eta_c, pc, rng);
    }

    // Two children of a multi-parent kind from P, P[0] and P[1] the pair and
    // at least parents(n) members in all.
    template <typename RNG>
    void cross_multi(const ParentSet& P, std::vector<double>& c1, std::vector<double>& c2,
                     const Bounds& bounds, RNG& rng) const
    {
        cross_multi_kind(kind, P, c1, c2, bounds, rng);
    }

    // The hosts' entry: the pair they selected, and draw_parent(), their own
    // mating selection returning a member's variables, for the parents a
    // multi-parent kind takes beyond two. The pair is crossed with
    // probability pc; when it is not, no further parent is drawn. A drawn
    // parent equal to one already taken is drawn again, at most 10·k times
    // in one application (MP-6). best(p) returns a random member of the
    // host's best ⌈p·N⌉ (de_pbest; de_crossover.hpp), N the population size.
    template <typename RNG, typename Draw, typename Best>
    void apply_any(const std::vector<double>& p1, const std::vector<double>& p2, Draw&& draw_parent,
                   Best&& best, int N, std::vector<double>& c1, std::vector<double>& c2,
                   const Bounds& bounds, double eta_c, double pc, RNG& rng)
    {
        Crossover k = kind;
        if (k == Crossover::Portfolio) k = pick(rng);
        last_ = k;
        crossed_ = false;
        if (k != Crossover::DE && k != Crossover::DEPBest && !is_multi(k)) {
            apply_two(k, p1, p2, c1, c2, bounds, eta_c, pc, rng);
            return;
        }
        require_repair(repair, crossover_name(k), false, false);
        note_operator(crossover_name(k), bound_repair_name(repair));
        c1 = p1; c2 = p2;
        if (pc < 1.0) {
            rnd::uniform_real_distribution<double> uni(0.0, 1.0);
            if (uni(rng) > pc) return;
        }
        crossed_ = true;
        ParentSet P{p1, p2};
        const int need = parents_of(k, static_cast<int>(p1.size()));
        int redraws = 0;
        while (static_cast<int>(P.size()) < need) {
            std::vector<double> x = draw_parent();
            const bool taken = std::find(P.begin(), P.end(), x) != P.end();
            if (taken && redraws++ < 10 * need) continue;
            P.push_back(std::move(x));
        }
        if (k == Crossover::DE || k == Crossover::DEPBest) {
            auto r2 = [&] { return de.population_or_archive(draw_parent, N, rng); };
            // p of current-to-pbest: de_p, or with SHADE rand[2/N, 0.2] per
            // child (SHADE's Eq. 20; de_crossover.hpp)
            auto p_of = [&] {
                if (de.set.adapt != DEAdapt::SHADE) return de.set.p;
                const double lo = std::min(0.2, 2.0 / static_cast<double>(N));
                rnd::uniform_real_distribution<double> u(lo, 0.2);
                return u(rng);
            };
            de_children(P, k == Crossover::DEPBest, [&] { return best(p_of()); }, r2, c1, c2,
                        bounds, repair, de, rng);
            return;
        }
        cross_multi_kind(k, P, c1, c2, bounds, rng);
    }

    // The same, for a host without an order of its own (de_pbest refused).
    template <typename RNG, typename Draw>
    void apply_any(const std::vector<double>& p1, const std::vector<double>& p2, Draw&& draw_parent,
                   std::vector<double>& c1, std::vector<double>& c2, const Bounds& bounds,
                   double eta_c, double pc, RNG& rng)
    {
        auto no_best = [](double) -> std::vector<double> {
            throw std::invalid_argument("crossover 'de_pbest' needs the host's order of its "
                                        "members, which this algorithm does not give");
        };
        apply_any(p1, p2, std::forward<Draw>(draw_parent), no_best, 0, c1, c2, bounds, eta_c,
                  pc, rng);
    }

    // After the host's mutation, the children it evaluates: c1 and c2, or
    // with only_one the one of them it keeps (0 or 1). de and de_pbest
    // follow them to the selection's verdict; the portfolio counts them.
    void finish(const std::vector<double>& c1, const std::vector<double>& c2, int only_one = -1)
    {
        if (!observing()) return;
        if ((last_ == Crossover::DE || last_ == Crossover::DEPBest) && crossed_)
            de.finish(c1, c2, only_one);
        if (kind == Crossover::Portfolio) {
            auto& cnt = portfolio_count(crossover_name(last_));
            if (only_one != 1) { pending_.push_back({c1, last_}); ++cnt.children; ++cnt.t_children; }
            if (only_one != 0) { pending_.push_back({c2, last_}); ++cnt.children; ++cnt.t_children; }
        }
    }

    // At the start of every step, the population the host's selection left:
    // the verdict on the last step's children (de_crossover.hpp; the
    // portfolio's archive and counts), and the learned coordinates' update
    // (learned_coords.hpp) — `offspring` the children this step will make,
    // n by default (sms_emoa: 1).
    template <typename Vault, typename RNG>
    void observe(Vault& vault, int n, RNG& rng, int offspring = -1)
    {
        const bool learn = coords.kind != CoordsKind::Axis;
        if (!observing() && !learn) return;
        std::vector<std::vector<double>> X, F;
        X.reserve(static_cast<std::size_t>(n));
        F.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            X.push_back(vault.variables_of(static_cast<std::size_t>(i)));
            F.push_back(vault.objectives_of(static_cast<std::size_t>(i)));
        }
        if (observing()) {
            de.observe(X, F, rng);
            if (kind == Crossover::Portfolio) observe_portfolio(X, F);
        }
        if (learn) coords.observe(X, F, offspring < 0 ? n : offspring, vault.get_bounds(), rng);
    }

private:
    struct Tagged { std::vector<double> x; Crossover op; };
    struct Kept   { std::vector<double> f; Crossover op; };
    std::vector<Tagged> pending_;          // portfolio children awaiting the verdict
    std::vector<Kept>   archive_;          // borg: the non-dominated kept children
    Crossover           last_ = Crossover::SBX;
    bool                crossed_ = false;

    template <typename RNG>
    Crossover pick(RNG& rng) const
    {
        std::vector<double> w(portfolio.size(), 1.0);
        if (credit == PortfolioCredit::Borg)
            for (std::size_t i = 0; i < portfolio.size(); ++i)
                for (const auto& a : archive_) w[i] += a.op == portfolio[i] ? 1.0 : 0.0;
        rnd::discrete_distribution<std::size_t> d(w.begin(), w.end());
        return portfolio[d(rng)];
    }

    void observe_portfolio(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& F)
    {
        std::unordered_map<std::string, std::vector<std::size_t>> rows;
        for (std::size_t i = 0; i < X.size(); ++i) rows[de_detail::key(X[i])].push_back(i);
        auto dominates = [](const std::vector<double>& a, const std::vector<double>& b) {
            bool lt = false;
            for (std::size_t j = 0; j < a.size(); ++j) {
                if (a[j] > b[j]) return false;
                lt = lt || a[j] < b[j];
            }
            return lt;
        };
        for (const auto& c : pending_) {
            auto it = rows.find(de_detail::key(c.x));
            if (it == rows.end() || it->second.empty()) continue;
            const std::size_t r = it->second.back();
            it->second.pop_back();
            auto& cnt = portfolio_count(crossover_name(c.op));
            ++cnt.kept;
            ++cnt.t_kept;
            if (credit != PortfolioCredit::Borg) continue;
            const auto& f = F[r];
            bool beaten = false;
            for (const auto& a : archive_)
                if (dominates(a.f, f) || a.f == f) { beaten = true; break; }
            if (beaten) continue;
            archive_.erase(std::remove_if(archive_.begin(), archive_.end(),
                                          [&](const Kept& a) { return dominates(f, a.f); }),
                           archive_.end());
            archive_.push_back({f, c.op});
        }
        pending_.clear();
    }

    template <typename RNG>
    void apply_two(Crossover k, const std::vector<double>& p1, const std::vector<double>& p2,
                   std::vector<double>& c1, std::vector<double>& c2,
                   const std::vector<std::pair<std::optional<double>,
                                               std::optional<double>>>& bounds,
                   double eta_c, double pc, RNG& rng) const
    {
        switch (k) {
            case Crossover::SBX:      sbx(p1, p2, c1, c2, bounds, eta_c, pc, rng); break;
            case Crossover::Uniform:  uniform_crossover(p1, p2, c1, c2, pc, rng); break;
            case Crossover::BlxAlpha:
                blx_alpha_crossover(p1, p2, c1, c2, bounds, alpha, pc, repair, rng); break;
            case Crossover::None:     // the children are the parents (task 7, supplement 2)
                c1 = p1; c2 = p2;
                note_operator("none", "none");
                break;
            default:
                throw std::invalid_argument(std::string("crossover '") + crossover_name(k) +
                                            "' takes more than two parents: call apply_any");
        }
    }

    template <typename RNG>
    void cross_multi_kind(Crossover k, const ParentSet& P, std::vector<double>& c1,
                          std::vector<double>& c2, const Bounds& bounds, RNG& rng) const
    {
        const int n = static_cast<int>(P[0].size());
        if (static_cast<int>(P.size()) < parents_of(k, n))
            throw std::invalid_argument(std::string("crossover '") + crossover_name(k) +
                                        "': too few parents");
        switch (k) {
            case Crossover::SPX: {
                const double eps = std::sqrt(static_cast<double>(n) + 2.0);
                const auto g = mp_detail::centre(P);
                c1 = spx_child(P, eps, rng); mp_detail::repair(c1, g, bounds, repair, rng);
                c2 = spx_child(P, eps, rng); mp_detail::repair(c2, g, bounds, repair, rng);
                break;
            }
            case Crossover::REX: {
                const auto g = mp_detail::centre(P);
                c1 = rex_child(P, rng); mp_detail::repair(c1, g, bounds, repair, rng);
                c2 = rex_child(P, rng); mp_detail::repair(c2, g, bounds, repair, rng);
                break;
            }
            case Crossover::UNDX: {
                auto pr = undx_pair(P[0], P[1], P[2], 0.5, 0.35, rng);
                std::vector<double> m(P[0].size());
                for (std::size_t j = 0; j < m.size(); ++j) m[j] = 0.5 * (P[0][j] + P[1][j]);
                c1 = std::move(pr.first);  mp_detail::repair(c1, m, bounds, repair, rng);
                c2 = std::move(pr.second); mp_detail::repair(c2, m, bounds, repair, rng);
                break;
            }
            case Crossover::PCX: {
                auto a = pcx_child(P, 0.1, 0.1, rng);
                c1 = std::move(a.first);  mp_detail::repair(c1, P[a.second], bounds, repair, rng);
                auto b = pcx_child(P, 0.1, 0.1, rng);
                c2 = std::move(b.first);  mp_detail::repair(c2, P[b.second], bounds, repair, rng);
                break;
            }
            default: break;
        }
    }
};

}   // namespace mootation::ops

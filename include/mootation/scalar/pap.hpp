#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// pap (task 19, §19.6; B20): the population-based algorithm portfolio of F.
// Peng, K. Tang, G. Chen, X. Yao, "Population-Based Algorithm Portfolios for
// Numerical Optimization", IEEE TEVC 14(5), 2010, 782-800 (source
// portfolio_peng2010), Fig. 1 and §II-A, §II-C: m constituent algorithms,
// each on its own subpopulation, the time shared among them before the
// search; every migration_interval generations, for each constituent, the
// migration_size best individuals of the other m - 1 subpopulations are
// copied into its subpopulation and its migration_size worst discarded; PAP
// stops when the evaluations are spent. §IV-B: migration_size = 1,
// migration_interval = MAX_GEN / 20. Included from strategy.hpp.
//
// The constituents are the single-objective population methods of stage 4
// (de, jade, shade, lshade, ga_sbx, ssga, g3_pcx, pso, spso2011) and CMA-ES
// (cmaes.hpp), each with its paper's defaults and its subpopulation's size.
//
//   pap_members   a JSON array of {"solver": name, "np": size}; by default
//                 de 70, pso 16, cmaes 14 (Table III's DE+PSO+ES, with de for
//                 SaNSDE and pso for wPSO)
//   pap_interval  generations between migrations (0: MAX_GEN / 20)
//   pap_size      migration_size (1)
//
// Readings (OURS):
//   PAP-1. The constituents run generation by generation in lockstep, in
//     the order given: a generation of each is one of its ask/tell steps, so
//     the evaluations go to each in proportion to what its generation asks
//     (its subpopulation for the generational methods; g3_pcx's lambda
//     offspring, ssga's one child). MAX_GEN = budget / (the evaluations of
//     one generation of all), at least 1.
//   PAP-2. A migration takes the emigrants of every constituent from the
//     subpopulations as they stand before it ("combine the subpopulations of
//     the remaining m - 1 algorithms, identify and duplicate the best
//     migration_size individuals"); they cost nothing (copies). A migrant
//     takes the place of the worst member where it is better (evo.hpp,
//     replace_worst): the subpopulation with the emigrants, less its worst.
//   PAP-3. PSO's "gbest": the migrant replaces the worst particle's best and
//     position, its velocity 0, and gbest is the best of the bests again
//     (§II-C: "updating the gbest once a better solution emerges in the whole
//     population"); spso2011 likewise, its L_i at the next tell.
//   PAP-4. CMA-ES's "covariance matrix updated after the migration": the
//     migrant is injected (cmaes.hpp, inject; task 19, §19.4.3) into its next
//     generation, which updates the distribution with it; it is evaluated
//     again, free (the archive's). CMA-ES runs in the box normalized to [0,
//     1]^n, sigma0 = 0.3, its mean uniform in the box, a point outside
//     evaluated at its clip plus the squared distance times 1 + |f| (as
//     sep_cmaes, POP-7); its subpopulation is its last generation.
//   PAP-5. A constituent that ends (its done(): a collapsed population)
//     leaves its time to the others: they go on until the budget is spent,
//     in proportion to their generations as before (§II-C: "in proportion to
//     the sizes of their subpopulations").
//   PAP-6. Constituent k's generator is start_seed(seed, k) (strategy.hpp
//     STRAT-1): one constituent, no migration, is that method's own run, bit
//     for bit. lshade's MAX_NFE is its share of the budget, np / sum(np).
//   PAP-7. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) for the emigrants and the migrants; CMA-ES's value of an
//     infeasible point is +inf.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../json.hpp"
#include "cmaes.hpp"
#include "de.hpp"
#include "evo.hpp"
#include "ga.hpp"
#include "jade.hpp"
#include "lshade.hpp"
#include "population.hpp"
#include "pso.hpp"
#include "shade.hpp"
#include "spso2011.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace pap {

struct Individual {
    Vec x;
    double f = 0.0, v = 0.0;
};

// A constituent: one generation at a time, its best individuals, a migrant.
class Member {
public:
    explicit Member(unsigned seed) : rng(seed) {}
    virtual ~Member() = default;
    std::mt19937 rng;                 // before the method, which keeps a reference
    virtual Rows ask() = 0;
    virtual void tell(const Vec& f, const Vec& v) = 0;
    virtual void migrate(const Individual& m) = 0;
    virtual std::vector<Individual> best(std::size_t k) const = 0;
    virtual bool done() const = 0;
};

inline std::vector<Individual> best_of(const Rows& X, const Vec& F, const Vec& V, std::size_t k)
{
    std::vector<std::size_t> order(X.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return evo::better(F[a], V[a], F[b], V[b]); });
    std::vector<Individual> out;
    for (std::size_t i = 0; i < k && i < order.size(); ++i) out.push_back({X[order[i]], F[order[i]], V[order[i]]});
    return out;
}

// A method of stage 4 (PSO's population: the particles' bests)
template <class K>
class Kernel final : public Member {
public:
    template <class Options>
    Kernel(unsigned seed, const Vec& lo, const Vec& hi, const Options& o, bool can_end)
        : Member(seed), k_(lo, hi, o, rng), can_end_(can_end) {}
    Rows ask() override { return k_.ask(); }
    void tell(const Vec& f, const Vec& v) override { k_.tell(f, v); }
    void migrate(const Individual& m) override { k_.migrate(m.x, m.f, m.v); }
    std::vector<Individual> best(std::size_t k) const override
    {
        return best_of(k_.population(), k_.values(), k_.violations(), k);
    }
    bool done() const override
    {
        if constexpr (has_done) return can_end_ && k_.done();
        else return false;
    }

private:
    template <class T, class = void>
    struct Done : std::false_type {};
    template <class T>
    struct Done<T, std::void_t<decltype(std::declval<const T&>().done())>> : std::true_type {};
    static constexpr bool has_done = Done<K>::value;
    K k_;
    bool can_end_;
};

// CMA-ES in the box normalized (PAP-4)
class Cma final : public Member {
public:
    Cma(unsigned seed, const Vec& lo, const Vec& hi, std::size_t lambda) : Member(seed), lo_(lo), hi_(hi)
    {
        Vec y0(lo.size());
        for (double& v : y0) v = rnd::uniform01(rng);
        es_ = std::make_unique<CMAES>(y0, 0.3, rng, static_cast<int>(lambda));
    }
    Rows ask() override
    {
        const Rows Y = es_->ask();
        X_.assign(Y.size(), Vec(lo_.size()));
        d2_.assign(Y.size(), 0.0);
        for (std::size_t k = 0; k < Y.size(); ++k)
            for (std::size_t j = 0; j < lo_.size(); ++j) {
                const double yc = std::min(1.0, std::max(0.0, Y[k][j]));
                d2_[k] += (Y[k][j] - yc) * (Y[k][j] - yc);
                X_[k][j] = lo_[j] + yc * (hi_[j] - lo_[j]);
            }
        return X_;
    }
    void tell(const Vec& f, const Vec& v) override
    {
        F_ = f;
        V_ = v;
        Vec values(f.size());
        for (std::size_t k = 0; k < f.size(); ++k)
            values[k] = std::isfinite(f[k]) && !(evo::key(v[k]) > 0.0)
                            ? f[k] + d2_[k] * (1.0 + std::fabs(f[k]))
                            : std::numeric_limits<double>::infinity();
        es_->tell(values);
    }
    void migrate(const Individual& m) override
    {
        Vec y(lo_.size());
        for (std::size_t j = 0; j < y.size(); ++j) y[j] = (m.x[j] - lo_[j]) / (hi_[j] - lo_[j]);
        es_->inject({y});
    }
    std::vector<Individual> best(std::size_t k) const override { return best_of(X_, F_, V_, k); }
    bool done() const override { return false; }

private:
    Vec lo_, hi_;
    std::unique_ptr<CMAES> es_;
    Rows X_;
    Vec F_, V_, d2_;
};

struct Spec {
    std::string solver;
    std::size_t np = 0;
};

inline std::vector<Spec> members_of(const std::string& text)
{
    const json::Value a = json::parse(text);
    if (a.kind != json::Value::Kind::Array || a.items.empty())
        throw std::invalid_argument("pap: pap_members must be a non-empty array of {\"solver\": name, \"np\": size}");
    std::vector<Spec> out;
    for (const json::Value& e : a.items) {
        const json::Value* s = e.kind == json::Value::Kind::Object ? e.find("solver") : nullptr;
        const json::Value* n = e.kind == json::Value::Kind::Object ? e.find("np") : nullptr;
        if (s == nullptr || s->kind != json::Value::Kind::String || n == nullptr || n->kind != json::Value::Kind::Int ||
            n->i < 1 || e.keys.size() != 2)
            throw std::invalid_argument("pap: a member is {\"solver\": name, \"np\": size >= 1}");
        out.push_back({s->s, static_cast<std::size_t>(n->i)});
    }
    return out;
}

inline std::unique_ptr<Member> member(const Spec& s, unsigned seed, const Vec& lo, const Vec& hi, long long nfe)
{
    if (s.solver == "de") {
        de::Options o;
        o.np = s.np;
        return std::make_unique<Kernel<de::DE>>(seed, lo, hi, o, true);
    }
    if (s.solver == "jade") {
        jade::Options o;
        o.np = s.np;
        return std::make_unique<Kernel<jade::JADE>>(seed, lo, hi, o, true);
    }
    if (s.solver == "shade") {
        shade::Options o;
        o.np = s.np;
        return std::make_unique<Kernel<shade::SHADE>>(seed, lo, hi, o, true);
    }
    if (s.solver == "lshade") {
        lshade::Options o;
        o.r_ninit = static_cast<double>(s.np) / static_cast<double>(lo.size());
        o.max_nfe = std::max(1LL, nfe);                                                     // PAP-6
        return std::make_unique<Kernel<lshade::LSHADE>>(seed, lo, hi, o, true);
    }
    if (s.solver == "ga_sbx" || s.solver == "ssga") {
        ga::Options o = s.solver == "ssga" ? ga::ssga() : ga::ga_sbx();
        o.mu = s.np;
        return std::make_unique<Kernel<ga::GA>>(seed, lo, hi, o, true);
    }
    if (s.solver == "g3_pcx") {
        ga::G3Options o;
        o.n = s.np;
        return std::make_unique<Kernel<ga::G3>>(seed, lo, hi, o, false);
    }
    if (s.solver == "pso") {
        pso::Options o;
        o.n = s.np;
        return std::make_unique<Kernel<pso::PSO>>(seed, lo, hi, o, true);
    }
    if (s.solver == "spso2011") {
        spso2011::Options o;
        o.n = s.np;
        return std::make_unique<Kernel<spso2011::SPSO2011>>(seed, lo, hi, o, false);
    }
    if (s.solver == "cmaes") return std::make_unique<Cma>(seed, lo, hi, s.np);
    throw std::invalid_argument("pap: '" + s.solver +
                                "' is not a member (de, jade, shade, lshade, ga_sbx, ssga, g3_pcx, pso, spso2011, cmaes)");
}

}   // namespace pap

inline Result run_pap(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every)
{
    using namespace strategy;
    StrategyKnobs k("pap", knobs, {"pap_members", "pap_interval", "pap_size"});
    std::string text = "[{\"np\":70,\"solver\":\"de\"},{\"np\":16,\"solver\":\"pso\"},{\"np\":14,\"solver\":\"cmaes\"}]";
    if (const Knob* v = k.find("pap_members")) {
        if (!v->is_word) throw std::invalid_argument("pap: pap_members must be a JSON array");
        text = v->word;
    }
    k.settings.push_back({"pap_members", text});
    const long long interval_knob = k.whole("pap_interval", 0, 0);
    const long long size = k.whole("pap_size", 1, 1);
    if (P.m != 1) throw std::invalid_argument("pap is a single-objective algorithm: n_objs must be 1");
    search::finite_box("pap", P);
    const std::vector<pap::Spec> specs = pap::members_of(text);
    std::size_t total = 0;
    for (const pap::Spec& s : specs) total += s.np;

    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    std::vector<std::unique_ptr<pap::Member>> m;
    for (std::size_t i = 0; i < specs.size(); ++i)
        m.push_back(pap::member(specs[i], start_seed(seed, i), P.lo, P.hi,
                                budget * static_cast<long long>(specs[i].np) / static_cast<long long>(total)));
    long long generations = 0, migrations = 0, interval = interval_knob;
    try {
        int idle = 0;
        bool first = true;
        long long per_generation = 0;
        while (true) {
            bool fresh = false, any = false;
            long long asked = 0;
            for (auto& a : m) {
                if (a->done()) continue;                                                        // PAP-5
                any = true;
                const Rows X = a->ask();
                Vec f, v;
                fresh = population::values(ev, X, f, v) || fresh;
                if (first && !ev.recorded) ev.record();
                asked += static_cast<long long>(X.size());
                a->tell(f, v);
            }
            first = false;
            if (!any) break;
            ++generations;
            if (generations == 1) {                                                            // PAP-1
                per_generation = std::max(1LL, asked);
                const long long max_gen = std::max(1LL, budget / per_generation);
                if (interval == 0) interval = std::max(1LL, max_gen / 20);
            }
            idle = fresh ? 0 : idle + 1;
            if (idle >= population::kIdleGenerations) break;                                  // POP-2
            if (m.size() > 1 && generations > 1 && generations % interval == 0) {             // PAP-2
                std::vector<std::vector<pap::Individual>> out;
                for (auto& a : m) out.push_back(a->best(static_cast<std::size_t>(size)));
                for (std::size_t i = 0; i < m.size(); ++i) {
                    std::vector<pap::Individual> pool;
                    for (std::size_t j = 0; j < m.size(); ++j)
                        if (j != i) pool.insert(pool.end(), out[j].begin(), out[j].end());
                    std::stable_sort(pool.begin(), pool.end(), [](const pap::Individual& a, const pap::Individual& b) {
                        return evo::better(a.f, a.v, b.f, b.v);
                    });
                    for (std::size_t q = 0; q < pool.size() && q < static_cast<std::size_t>(size); ++q)
                        m[i]->migrate(pool[q]);
                }
                ++migrations;
            }
        }
    } catch (const BudgetSpent&) {
    }
    MinimaLog log(P.lo, P.hi, 1e-3, 1e-8);
    return search::finish(ev, k, log,
                          {{"migration_interval", std::to_string(interval)}, {"generations", std::to_string(generations)},
                           {"migrations", std::to_string(migrations)}});
}

}   // namespace mootation::scalar

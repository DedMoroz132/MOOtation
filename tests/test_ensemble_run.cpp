// SPDX-License-Identifier: Apache-2.0
// The ensemble of resumable runs (task 31, layer 4; scalar/ensemble.hpp):
//   * the ensemble's rounds and exchanges on a small resumable run of this
//     test (Toy): one member without an exchange is the member's run; K
//     members without an exchange (or with migrants = 0) are K independent
//     runs; every topology, pick and place of migrate, and swap over a knob
//     and over a parameter, give the same run on 1 thread and on 3; the
//     members' spending and the exchanges' evaluations stay within the
//     budget; migrants and exchanges are taken;
//   * on the library's resumable strategies (where resumable_names() has
//     them): ensemble with one member = the strategy's run, bit for bit
//     (basin_hopping, cluster_ga, ga_sbx); K members with exchange_every = 0
//     or migrants = 0 = K independent runs with the members' seeds and
//     budgets; 1 thread = 4 threads; replica_ladder with rl_mode = resume and
//     swap_every = 0 = rl_mode = pieces, bit for bit;
//   * the ensemble's refusals.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& kv : r.settings)
        if (kv.first == key) return kv.second;
    return "";
}

double u01(std::mt19937& g) { return mootation::rnd::uniform01(g); }

// Rastrigin in n variables
sc::Problem rastrigin(std::size_t n)
{
    sc::Problem P;
    P.lo.assign(n, -5.12);
    P.hi.assign(n, 5.12);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
        f.assign(1, s);
    };
    return P;
}

// f_L(x) = sum (x_j - L)^2 in [-4, 4]^n: a parameter that moves the minimum
sc::Problem shifted(std::size_t n, double L)
{
    sc::Problem P;
    P.lo.assign(n, -4.0);
    P.hi.assign(n, 4.0);
    P.m = 1;
    P.param = L;
    P.evaluate = [L](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (double v : x) s += (v - L) * (v - L);
        f.assign(1, s);
    };
    P.at_param = [n](double L2) { return shifted(n, L2); };
    return P;
}

// A small resumable run: pop members, each step every member moved by a
// uniform step and kept where better. Its own evaluator and generator; a
// piece is whole steps.
class Toy : public sc::Resumable {
public:
    Toy(const sc::Problem& P, long long budget, unsigned seed, std::size_t pop, const sc::Start& st)
        : ev_(P, 0.0, false, 0.0, budget, 1, {}, 0), rng_(seed), pop_(pop), start_(st) {}

    const std::string& name() const override { return name_; }
    void begin() override
    {
        if (begun_) return;
        begun_ = true;
        try {
            for (std::size_t q = 0; q < pop_; ++q) {
                Vec x(ev_.lo.size());
                if (q == 0 && start_.has_x0) {
                    x = start_.x0;
                } else {
                    for (std::size_t j = 0; j < x.size(); ++j) x[j] = ev_.lo[j] + u01(rng_) * (ev_.hi[j] - ev_.lo[j]);
                }
                cur_.push_back(eval(x));
            }
        } catch (const sc::BudgetSpent&) {
            over_ = true;
        }
    }
    bool advance(const sc::Piece& piece) override
    {
        begin();
        const double from = ev_.counted();
        long long taken = 0;
        while (!over_) {
            if ((piece.steps > 0 && taken >= piece.steps) || (piece.spend > 0.0 && ev_.counted() - from >= piece.spend))
                break;
            try {
                for (sc::Member& m : cur_) {
                    Vec y = m.x;
                    for (std::size_t j = 0; j < y.size(); ++j)
                        y[j] = (std::min)(ev_.hi[j], (std::max)(ev_.lo[j], y[j] + 0.5 * (2.0 * u01(rng_) - 1.0)));
                    const sc::Member c = eval(y);
                    if (c.f < m.f) m = c;
                }
            } catch (const sc::BudgetSpent&) {
                over_ = true;
            }
            ++taken;
            ++steps_;
        }
        return !over_;
    }
    bool done() const override { return over_; }
    long long steps() const override { return steps_; }
    const sc::Evaluator& ledger() const override { return ev_; }
    void set_budget(long long budget) override { ev_.budget = ev_.limit = budget; }
    std::vector<sc::Member> members() const override { return cur_; }
    std::vector<sc::Member> best_points(std::size_t k) const override
    {
        return k == 0 ? std::vector<sc::Member>{} : std::vector<sc::Member>{best()};
    }
    long long admit(const std::vector<sc::Member>& in, sc::Place how) override
    {
        long long taken = 0;
        std::vector<char> fresh(cur_.size(), 0);
        for (std::size_t q = 0; q < in.size(); ++q) {
            if (how == sc::Place::all) {
                if (q >= cur_.size()) break;
                cur_[q] = planted(in[q]);
                ++taken;
                continue;
            }
            std::size_t w = cur_.size();
            for (std::size_t i = 0; i < cur_.size(); ++i)
                if (!fresh[i] && (w == cur_.size() || cur_[i].f > cur_[w].f)) w = i;
            if (w == cur_.size()) break;
            if (how == sc::Place::worst_if_better && !(in[q].f < cur_[w].f)) continue;
            cur_[w] = planted(in[q]);
            fresh[w] = 1;
            ++taken;
        }
        return taken;
    }
    long long admit(const std::vector<sc::Member>& in, const sc::PlaceRule& rule) override
    {
        long long taken = 0;
        for (const sc::Member& m : in) {
            const long long i = rule(cur_, m);
            if (i < 0) continue;
            cur_[static_cast<std::size_t>(i)] = planted(m);
            ++taken;
        }
        return taken;
    }
    sc::Operators operators() const override { return {}; }
    void set_operators(const sc::Operators&) override {}
    sc::Member best() const override { return sc::best_member(ev_); }
    std::vector<std::pair<std::string, std::string>> summary() const override
    {
        return {{"steps", std::to_string(steps_)}};
    }
    sc::Result result() override
    {
        over_ = true;
        sc::Result r;
        const sc::Member b = best();
        if (!b.x.empty()) {
            r.variables = {b.x};
            r.objectives = {{b.f}};
            r.cv = {b.cv};
        }
        r.evaluations = ev_.spent;
        r.used = ev_.counted();
        r.settings = summary();
        r.stopped_by = "budget";
        for (const sc::Member& m : cur_) r.population.push_back(m.x);
        return r;
    }

private:
    sc::Evaluator ev_;
    std::mt19937 rng_;
    std::size_t pop_;
    sc::Start start_;
    std::string name_ = "toy";
    bool begun_ = false, over_ = false;
    long long steps_ = 0;
    std::vector<sc::Member> cur_;

    sc::Member eval(const Vec& x)
    {
        if (ev_.left() <= 0) throw sc::BudgetSpent{};
        const std::size_t i = ev_.index(x);
        return sc::member_of(ev_, x, ev_.F[i][0], ev_.cv(i));
    }
    sc::Member planted(const sc::Member& m)
    {
        const std::size_t i = ev_.plant(m.x, {m.f}, m.g);
        return sc::member_of(ev_, m.x, ev_.F[i][0], ev_.cv(i));
    }
};

bool same_result(const sc::Result& a, const sc::Result& b)
{
    return a.variables == b.variables && a.objectives == b.objectives && a.cv == b.cv &&
           a.evaluations == b.evaluations && a.grad_calls == b.grad_calls && a.used == b.used &&
           a.population == b.population;
}

bool same_outcome(const sc::ens::Outcome& a, const sc::ens::Outcome& b)
{
    bool ok = a.results.size() == b.results.size() && a.rounds == b.rounds && a.pieces == b.pieces &&
              a.offered == b.offered && a.accepted == b.accepted && a.tried == b.tried && a.swapped == b.swapped &&
              a.own_evaluations == b.own_evaluations && a.why == b.why;
    for (std::size_t m = 0; ok && m < a.results.size(); ++m) ok = same_result(a.results[m], b.results[m]);
    return ok;
}

// an ensemble of K toys on P with the members' seeds and budgets as the strategy gives them
sc::ens::Plan toy_plan(const sc::Problem& P, std::size_t K, long long budget, unsigned seed)
{
    sc::ens::Plan plan;
    const std::vector<long long> share = sc::split_budget(budget, K);
    plan.members.resize(K);
    for (std::size_t m = 0; m < K; ++m) {
        plan.members[m].problem = &P;
        plan.members[m].seed = sc::strategy::start_seed(seed, static_cast<std::uint64_t>(m));
        plan.members[m].budget = share[m];
    }
    plan.stream = sc::strategy::start_seed(seed, 0xE45Eu);
    return plan;
}

sc::ens::Make toy_maker(std::size_t pop)
{
    return [pop](const sc::ens::Spec& s, std::size_t) {
        return std::unique_ptr<sc::Resumable>(std::make_unique<Toy>(*s.problem, s.budget, s.seed, pop, s.start));
    };
}

sc::ens::Outcome toy_run(const sc::Problem& P, long long budget, const sc::ens::Plan& plan, std::size_t pop)
{
    sc::strategy::StopRule stop;
    return sc::ens::run(P, budget, plan, stop, toy_maker(pop));
}

long long spent_of(const sc::ens::Outcome& o)
{
    long long s = o.own_evaluations;
    for (const sc::Result& r : o.results) s += r.evaluations;
    return s;
}

bool has_resumable(const std::string& name)
{
    for (const std::string& n : sc::resumable_names())
        if (n == name) return true;
    return false;
}

const char* kNM = "{\"solver\":\"nelder_mead\",\"local_tol\":1e-8}";

sc::Result run(const std::string& name, const sc::Problem& P, const sc::Knobs& k, long long budget, unsigned seed,
               const sc::Start& st = {})
{
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, st);
}

// the settings of two runs alike but for the threads knob's row
bool same_rows(const sc::Result& a, const sc::Result& b)
{
    std::vector<std::pair<std::string, std::string>> x, y;
    for (const auto& kv : a.settings)
        if (kv.first != "threads") x.push_back(kv);
    for (const auto& kv : b.settings)
        if (kv.first != "threads") y.push_back(kv);
    return x == y;
}

// the ensemble's settings begin with the member's own
bool member_rows_first(const sc::Result& member, const sc::Result& ensemble)
{
    if (ensemble.settings.size() < member.settings.size()) return false;
    for (std::size_t q = 0; q < member.settings.size(); ++q)
        if (member.settings[q].first != ensemble.settings[q].first ||
            (member.settings[q].first != "seconds" && member.settings[q].second != ensemble.settings[q].second &&
             member.settings[q].first != "wall_seconds" && member.settings[q].first != "cpu_seconds"))
            return false;
    return true;
}

}   // namespace

int main()
{
    const sc::Problem R4 = rastrigin(4);
    const sc::Start none;

    // ── the core on the toy run ──────────────────────────────────────────────
    {
        // one member without an exchange = the member's run
        sc::ens::Plan one = toy_plan(R4, 1, 400, 11);
        const sc::ens::Outcome o1 = toy_run(R4, 400, one, 3);
        Toy alone(R4, 400, 11, 3, none);
        alone.advance(sc::Piece());
        const sc::Result a = alone.result();
        check(o1.ran[0] && same_result(o1.results[0], a) && o1.rounds == 1,
              "toy: one member without an exchange = the member's run");

        // K members, in pieces, without an exchange = K independent runs
        for (int variant = 0; variant < 3; ++variant) {
            sc::ens::Plan p = toy_plan(R4, 3, 1000, 5);
            if (variant >= 1) p.piece.steps = 4;
            if (variant == 2) {
                p.exchange = sc::ens::Exchange::migrate;
                p.migrants = 0;
            }
            const sc::ens::Outcome o = toy_run(R4, 1000, p, 3);
            bool alike = true;
            for (std::size_t m = 0; m < 3; ++m) {
                Toy t(R4, p.members[m].budget, p.members[m].seed, 3, none);
                t.advance(sc::Piece());
                alike = alike && o.ran[m] && same_result(o.results[m], t.result());
            }
            const char* what[3] = {"one round", "pieces of 4 steps", "migrants = 0"};
            check(alike && spent_of(o) <= 1000, std::string("toy: 3 members, ") + what[variant] +
                                                    " = 3 independent runs with the members' seeds and budgets");
        }

        // migrate: every topology, pick and place; 1 thread = 3 threads; within the budget
        const sc::ens::Topology tops[3] = {sc::ens::Topology::ring, sc::ens::Topology::full, sc::ens::Topology::random};
        const sc::ens::Pick picks[3] = {sc::ens::Pick::best, sc::ens::Pick::random, sc::ens::Pick::diverse};
        const sc::Place places[2] = {sc::Place::worst, sc::Place::worst_if_better};
        bool all_alike = true, all_within = true, any_taken = true;
        for (const auto t : tops)
            for (const auto pk : picks)
                for (const auto pl : places) {
                    sc::ens::Plan p = toy_plan(R4, 4, 1200, 3);
                    p.piece.steps = 5;
                    p.exchange = sc::ens::Exchange::migrate;
                    p.topology = t;
                    p.pick = pk;
                    p.place = pl;
                    p.migrants = 2;
                    const sc::ens::Outcome o1t = toy_run(R4, 1200, p, 4);
                    p.threads = 3;
                    const sc::ens::Outcome o3t = toy_run(R4, 1200, p, 4);
                    all_alike = all_alike && same_outcome(o1t, o3t);
                    all_within = all_within && spent_of(o1t) <= 1200;
                    any_taken = any_taken && o1t.offered > 0 && o1t.accepted > 0 && o1t.accepted <= o1t.offered;
                }
        check(all_alike, "toy: migrate, every topology, pick and place: 1 thread = 3 threads");
        check(all_within, "toy: migrate: the members' evaluations within the budget");
        check(any_taken, "toy: migrate: migrants offered and taken, never more taken than offered");
        {
            sc::ens::Plan p = toy_plan(R4, 3, 900, 8);
            p.piece.steps = 5;
            p.exchange = sc::ens::Exchange::migrate;
            p.place = sc::Place::worst;
            const sc::ens::Outcome o = toy_run(R4, 900, p, 3);
            check(o.offered == o.accepted && o.offered % 3 == 0 && o.offered >= 3 * (o.rounds - 1) &&
                      o.offered <= 3 * o.rounds,
                  "toy: ring, one best migrant, worst: 3 placed after every round the members go on from, " +
                      std::to_string(o.offered) + " in " + std::to_string(o.rounds) + " rounds");
        }

        // swap over a knob (the temperatures given)
        {
            sc::ens::Plan p = toy_plan(R4, 3, 900, 2);
            p.piece.steps = 3;
            p.exchange = sc::ens::Exchange::swap;
            const double T[3] = {0.5, 2.0, 8.0};
            for (std::size_t m = 0; m < 3; ++m) p.members[m].temperature = T[m];
            const sc::ens::Outcome o1t = toy_run(R4, 900, p, 1);
            p.threads = 3;
            const sc::ens::Outcome o3t = toy_run(R4, 900, p, 1);
            check(same_outcome(o1t, o3t) && o1t.tried > 0 && o1t.swapped > 0 && o1t.own_evaluations == 0,
                  "toy: swap over a knob: tried " + std::to_string(o1t.tried) + ", taken " + std::to_string(o1t.swapped) +
                      ", 1 thread = 3 threads, no evaluation of its own");
        }

        // swap over a parameter and migrate between levels: carried, evaluated, charged
        {
            const sc::Problem S = shifted(3, 0.0);
            const double lv[3] = {0.5, 1.0, 1.5};
            for (int kind = 0; kind < 2; ++kind) {
                sc::ens::Plan p = toy_plan(S, 3, 900, 4);
                p.piece.steps = 3;
                for (std::size_t m = 0; m < 3; ++m) {
                    p.members[m].level = lv[m];
                    p.members[m].own = std::make_shared<sc::Problem>(sc::problem_at(S, lv[m]));
                    p.members[m].problem = p.members[m].own.get();
                }
                if (kind == 0) {
                    p.exchange = sc::ens::Exchange::swap;
                    p.over_param = true;
                } else {
                    p.exchange = sc::ens::Exchange::migrate;
                    p.topology = sc::ens::Topology::full;
                    p.migrants = 2;
                }
                const sc::ens::Outcome o1t = toy_run(S, 900, p, 2);
                p.threads = 2;
                const sc::ens::Outcome o3t = toy_run(S, 900, p, 2);
                const char* what = kind == 0 ? "swap over a parameter" : "migrate between levels";
                check(same_outcome(o1t, o3t) && o1t.own_evaluations > 0 && spent_of(o1t) <= 900,
                      std::string("toy: ") + what + ": " + std::to_string(o1t.own_evaluations) +
                          " evaluations of the exchange, " + std::to_string(spent_of(o1t)) +
                          " in all <= 900, 1 thread = 2 threads");
            }
        }
    }

    // ── the library's resumable strategies ───────────────────────────────────
    const std::string bh = std::string("{\"inner\":") + kNM + ",\"solver\":\"basin_hopping\"}";
    const std::string cga = std::string("{\"cga_pop\":4,\"inner\":") + kNM + ",\"solver\":\"cluster_ga\"}";
    const std::string gas = "{\"np\":12,\"solver\":\"ga_sbx\"}";
    if (!has_resumable("basin_hopping") || !has_resumable("cluster_ga") || !has_resumable("ga_sbx")) {
        std::printf("skip  the library's resumable strategies: resumable_names() lacks them (task 31, layer 3)\n");
    } else {
        const sc::Start from = [] {
            sc::Start s;
            s.has_x0 = true;
            s.x0 = {2.0, -1.0, 0.5, 3.0};
            return s;
        }();
        // one member = the strategy's run, bit for bit
        for (const std::string& json : {bh, cga, gas}) {
            const sc::strategy::Inner in = sc::strategy::inner_of("inner", word(json));
            const sc::Result plain = run(in.name, R4, in.knobs, 3000, 7, from);
            const sc::Result one = run("ensemble", R4, {{"inner", word(json)}}, 3000, 7, from);
            const sc::Result pieces =
                run("ensemble", R4, {{"inner", word(json)}, {"exchange_every", number(3)}}, 3000, 7, from);
            check(same_result(plain, one) && member_rows_first(plain, one) && same_result(plain, pieces),
                  "ensemble with one member = " + in.name + "'s run, bit for bit (one round; rounds of 3 steps)");
        }
        // K members without an exchange = K independent runs
        for (const std::string& json : {bh, cga}) {
            const sc::strategy::Inner in = sc::strategy::inner_of("inner", word(json));
            const sc::Knobs mk{{"inner", word(json)}, {"ens_K", number(3)}};
            const sc::Result r0 = run("ensemble", R4, mk, 3001, 9, from);
            sc::Knobs mk2 = mk;
            mk2.push_back({"exchange_every", number(2)});
            mk2.push_back({"migrants", number(0)});
            const sc::Result r1 = run("ensemble", R4, mk2, 3001, 9, from);
            const std::vector<long long> share = sc::split_budget(3001, 3);
            std::string want = "[";
            long long evals = 0;
            for (std::size_t m = 0; m < 3; ++m) {
                const sc::Result q = run(in.name, R4, in.knobs, share[m], sc::strategy::start_seed(9, m), from);
                want += (m ? "," : "") + sc::detail::g_format(q.objectives[0][0]);
                evals += q.evaluations;
            }
            want += "]";
            check(setting(r0, "ens_best") == want && r0.evaluations == evals && setting(r1, "ens_best") == want &&
                      r1.evaluations == evals,
                  in.name + ": 3 members, exchange_every = 0 and migrants = 0 = 3 independent runs, " + want);
        }
        // 1 thread = 4 threads with exchanges; within the budget. Each local search capped (start_budget), so
        // that the members go on for many rounds (a search of nelder_mead to 1e-8 alone spends hundreds)
        const std::string bhc = std::string("{\"inner\":") + kNM + ",\"solver\":\"basin_hopping\",\"start_budget\":40}";
        const std::string cgac =
            std::string("{\"cga_pop\":4,\"inner\":") + kNM + ",\"solver\":\"cluster_ga\",\"start_budget\":40}";
        {
            const sc::Knobs mig{{"inner", word(cgac)}, {"ens_K", number(4)}, {"exchange_every", number(2)},
                                {"topology", word("full")}, {"migrants", number(2)}, {"migrant_pick", word("diverse")}};
            sc::Knobs mig4 = mig;
            mig4.push_back({"threads", number(4)});
            const sc::Result a = run("ensemble", R4, mig, 8000, 3, from), b = run("ensemble", R4, mig4, 8000, 3, from);
            check(same_result(a, b) && same_rows(a, b),
                  "cluster_ga islands (full, 2 diverse migrants every 2 generations): 1 thread = 4 threads, " +
                      setting(a, "rounds") + " rounds");
            check(a.evaluations <= 8000 && setting(a, "migrants_offered") != "0" && setting(a, "migrants_accepted") != "0",
                  "cluster_ga islands: migrants offered " + setting(a, "migrants_offered") + ", accepted " +
                      setting(a, "migrants_accepted") + ", " + std::to_string(a.evaluations) + " evaluations <= 8000");
            const std::string temps = "[{\"bh_temperature\":0.5},{\"bh_temperature\":2},{\"bh_temperature\":8}]";
            const sc::Knobs sw{{"inner", word(bhc)},         {"exchange", word("swap")},
                               {"exchange_every", number(3)}, {"swap_knob", word("bh_temperature")},
                               {"member_knobs", word(temps)}};
            sc::Knobs sw4 = sw;
            sw4.push_back({"threads", number(4)});
            const sc::Result c = run("ensemble", R4, sw, 6000, 3, from), d = run("ensemble", R4, sw4, 6000, 3, from);
            check(same_result(c, d) && same_rows(c, d),
                  "basin_hopping replicas over bh_temperature: 1 thread = 4 threads, " + setting(c, "rounds") + " rounds");
            check(setting(c, "swaps_tried") != "0" && c.evaluations <= 6000,
                  "basin_hopping replicas: swaps " + setting(c, "swaps_accepted") + " of " + setting(c, "swaps_tried") +
                      ", " + std::to_string(c.evaluations) + " evaluations <= 6000");
        }
        // member_params, member_seeds: a level each, the answer from the problem itself
        {
            const sc::Problem S = shifted(4, 0.0);
            const sc::Knobs mp{{"inner", word(bhc)},
                               {"member_params", word("[null, 1.0, 2.0]")},
                               {"member_seeds", word("[[[1,1,1,1]], null, [[2,2,2,2],[0,0,0,0]]]")},
                               {"exchange_every", number(3)},
                               {"topology", word("full")}};
            const sc::Result r = run("ensemble", S, mp, 6000, 5);
            const std::string ex = setting(r, "exchange_evaluations");
            check(!r.objectives.empty() && setting(r, "ens_answer") == "0" && r.evaluations <= 6000 &&
                      std::stoll(ex.empty() ? "0" : ex) > 0,
                  "member_params: migrants between levels evaluated (" + ex + "), " + std::to_string(r.evaluations) +
                      " evaluations <= 6000, the answer member 0's (the problem itself), " + setting(r, "rounds") + " rounds");
        }
        // replica_ladder: rl_mode = resume, swap_every = 0 = pieces, bit for bit
        {
            const sc::Problem S = shifted(2, 1.0);
            const std::string ibh = std::string("{\"inner\":") + kNM + ",\"solver\":\"basin_hopping\"}";
            const sc::Knobs kp{{"inner", word(ibh)}, {"rl_over", word("param")}, {"rl_levels", word("[0.5, 1, 1.5]")}};
            sc::Knobs kr = kp;
            kr.push_back({"rl_mode", word("resume")});
            const sc::Result a = run("replica_ladder", S, kp, 3000, 7), b = run("replica_ladder", S, kr, 3000, 7);
            check(same_result(a, b) && setting(a, "replica_best") == setting(b, "replica_best") &&
                      setting(b, "rl_mode") == "resume",
                  "replica_ladder over a parameter: rl_mode = resume with swap_every = 0 = pieces, bit for bit");
            const sc::Knobs kk{{"inner", word(ibh)}, {"rl_knob", word("bh_temperature")}, {"rl_levels", word("[0.5, 2, 8]")}};
            sc::Knobs kk2 = kk;
            kk2.push_back({"rl_mode", word("resume")});
            const sc::Result c = run("replica_ladder", R4, kk, 3000, 7), d = run("replica_ladder", R4, kk2, 3000, 7);
            check(same_result(c, d) && setting(c, "replica_best") == setting(d, "replica_best"),
                  "replica_ladder over a knob: rl_mode = resume with swap_every = 0 = pieces, bit for bit");
            sc::Knobs ks = kr;
            ks.push_back({"swap_every", number(150)});
            const sc::Result e = run("replica_ladder", S, ks, 3000, 7), f = run("replica_ladder", S, ks, 3000, 7);
            check(same_result(e, f) && setting(e, "swaps_tried") != "0" && e.evaluations <= 3000,
                  "replica_ladder, rl_mode = resume, swap_every = 150: exchanges tried " + setting(e, "swaps_tried") +
                      ", " + std::to_string(e.evaluations) + " evaluations, the same run twice");
        }
    }

    // ── refusals ─────────────────────────────────────────────────────────────
    {
        auto refused = [&](const sc::Knobs& k) {
            try {
                run("ensemble", R4, k, 500, 1);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        const std::string lb = "{\"solver\":\"lbfgsb\"}";
        check(refused({{"inner", word(lb)}}), "ensemble: an inner strategy that is not resumable refused");
        check(refused({}), "ensemble: no inner strategy refused");
        if (has_resumable("basin_hopping")) {
            check(refused({{"inner", word(bh)}, {"ens_K", number(2)}, {"member_params", word("[1, 2, 3]")}}),
                  "ensemble: member lists of another length than ens_K refused");
            check(refused({{"inner", word(bh)}, {"ens_K", number(2)}, {"exchange", word("swap")}}),
                  "ensemble: swap over a knob without swap_knob refused");
            check(refused({{"inner", word(bh)}, {"ens_K", number(2)}, {"exchange", word("swap")},
                           {"swap_over", word("param")}}),
                  "ensemble: swap over a parameter without member_params refused");
            check(refused({{"inner", word(bh)}, {"member_seeds", word("[[[1, 2]]]")}}),
                  "ensemble: a seed point of another length refused");
        }
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

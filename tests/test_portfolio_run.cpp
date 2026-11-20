// SPDX-License-Identifier: Apache-2.0
// The portfolio of resumable runs (task 32, layer 1; scalar/portfolio_run.hpp):
//   * the core on a small resumable run of this test (Toy): one member is the
//     member's run with every rule; op_select = fixed without share_best is
//     the K independent runs with the budget split evenly; two members, one
//     on a constant problem and one on the sphere: adaptive pursuit takes
//     the second's P to 1 - (K - 1) op_pmin and gives it that share of the
//     pieces; successive halving keeps the better half each round; 1 thread
//     = 3 threads; the members within the budget;
//   * on the library's resumable strategies: one member = the strategy's
//     run, bit for bit (basin_hopping, cluster_ga, ga_sbx; every rule, pieces
//     of steps and of budget); three different strategies with fixed = three
//     independent runs; 1 thread = 4 threads with share_best; leftover keep
//     and share; successive halving;
//   * the refusals.

#include <algorithm>
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
namespace json = mootation::json;
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

sc::Problem sphere(std::size_t n)
{
    sc::Problem P;
    P.lo.assign(n, -4.0);
    P.hi.assign(n, 4.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (double v : x) s += v * v;
        f.assign(1, s);
    };
    return P;
}

// a problem without improvements: f = c everywhere
sc::Problem constant(std::size_t n, double c)
{
    sc::Problem P;
    P.lo.assign(n, -4.0);
    P.hi.assign(n, 4.0);
    P.m = 1;
    P.evaluate = [c](const Vec&, Vec& f, Vec&) { f.assign(1, c); };
    return P;
}

// A small resumable run (as test_ensemble_run's): pop members, each step every
// member moved by a uniform step and kept where better. scaled: the step
// 0.5 sqrt(f / n) of the member's f (scale-invariant, so that a run on the
// sphere keeps improving), else 0.5
class Toy : public sc::Resumable {
public:
    Toy(const sc::Problem& P, long long budget, unsigned seed, std::size_t pop, const sc::Start& st, bool scaled = false)
        : ev_(P, 0.0, false, 0.0, budget, 1, {}, 0), rng_(seed), pop_(pop), start_(st), scaled_(scaled) {}

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
        const double from = ev_.counted();
        begin();
        long long taken = 0;
        while (!over_) {
            if ((piece.steps > 0 && taken >= piece.steps) || (piece.spend > 0.0 && ev_.counted() - from >= piece.spend)) {
                if (ev_.left() <= 0) over_ = true;
                break;
            }
            if (ev_.left() <= 0) {
                over_ = true;
                break;
            }
            try {
                for (sc::Member& m : cur_) {
                    Vec y = m.x;
                    const double step = scaled_ ? 0.5 * std::sqrt(m.f / static_cast<double>(y.size())) : 0.5;
                    for (std::size_t j = 0; j < y.size(); ++j)
                        y[j] = (std::min)(ev_.hi[j], (std::max)(ev_.lo[j], y[j] + step * (2.0 * u01(rng_) - 1.0)));
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
        for (const sc::Member& m : in) {
            std::size_t w = cur_.size();
            for (std::size_t i = 0; i < cur_.size(); ++i)
                if (!fresh[i] && (w == cur_.size() || cur_[i].f > cur_[w].f)) w = i;
            if (w == cur_.size()) break;
            if (how == sc::Place::worst_if_better && !(m.f < cur_[w].f)) continue;
            cur_[w] = planted(m);
            fresh[w] = 1;
            ++taken;
        }
        return taken;
    }
    long long admit(const std::vector<sc::Member>&, const sc::PlaceRule&) override { return 0; }
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
    bool scaled_ = false;
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

std::string dumped(const json::Value& v) { return json::dump(v, -1, ",", ":"); }

bool same_outcome(const sc::pf::Outcome& a, const sc::pf::Outcome& b)
{
    bool ok = a.results.size() == b.results.size() && a.rounds == b.rounds && a.pieces == b.pieces &&
              a.offered == b.offered && a.accepted == b.accepted && a.withheld == b.withheld && a.P == b.P &&
              a.Q == b.Q && a.why == b.why && dumped(a.history) == dumped(b.history) &&
              dumped(a.halving) == dumped(b.halving);
    for (std::size_t m = 0; ok && m < a.results.size(); ++m) ok = same_result(a.results[m], b.results[m]);
    return ok;
}

// a portfolio of toys, member m on problems[m], with the strategy's seeds
sc::pf::Plan toy_plan(const std::vector<const sc::Problem*>& problems, unsigned seed, sc::pf::Rule rule)
{
    sc::pf::Plan plan;
    plan.members.resize(problems.size());
    for (std::size_t m = 0; m < problems.size(); ++m) {
        plan.members[m].problem = problems[m];
        plan.members[m].name = "toy";
        plan.members[m].seed = sc::strategy::start_seed(seed, static_cast<std::uint64_t>(m));
    }
    plan.rule = rule;
    if (rule == sc::pf::Rule::matching || rule == sc::pf::Rule::pursuit)
        plan.pmin = problems.size() > 1 ? 0.5 / static_cast<double>(problems.size()) : 0.0;
    plan.stream = sc::strategy::start_seed(seed, 0x9F01u);
    plan.piece.steps = 1;
    return plan;
}

sc::pf::Outcome toy_run(long long budget, const sc::pf::Plan& plan, std::size_t pop, bool scaled = false)
{
    sc::strategy::StopRule stop;
    const sc::pf::Make make = [pop, scaled](const sc::pf::Spec& s, long long b, std::size_t) {
        return std::unique_ptr<sc::Resumable>(std::make_unique<Toy>(*s.problem, b, s.seed, pop, s.start, scaled));
    };
    return sc::pf::run(*plan.members[0].problem, budget, plan, stop, make);
}

long long spent_of(const sc::pf::Outcome& o)
{
    long long s = 0;
    for (const sc::Result& r : o.results) s += r.evaluations;
    return s;
}

const char* kNM = "{\"solver\":\"nelder_mead\",\"local_tol\":1e-8}";

sc::Result run(const std::string& name, const sc::Problem& P, const sc::Knobs& k, long long budget, unsigned seed,
               const sc::Start& st = {})
{
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, st);
}

bool same_rows(const sc::Result& a, const sc::Result& b)
{
    std::vector<std::pair<std::string, std::string>> x, y;
    for (const auto& kv : a.settings)
        if (kv.first != "threads") x.push_back(kv);
    for (const auto& kv : b.settings)
        if (kv.first != "threads") y.push_back(kv);
    return x == y;
}

// the portfolio's settings begin with the member's own
bool member_rows_first(const sc::Result& member, const sc::Result& portfolio)
{
    if (portfolio.settings.size() < member.settings.size()) return false;
    for (std::size_t q = 0; q < member.settings.size(); ++q)
        if (member.settings[q].first != portfolio.settings[q].first ||
            (member.settings[q].first != "seconds" && member.settings[q].second != portfolio.settings[q].second &&
             member.settings[q].first != "wall_seconds" && member.settings[q].first != "cpu_seconds"))
            return false;
    return true;
}

// a JSON list of the runs' answers' f, as pf_best writes it
std::string answers(const std::vector<sc::Result>& rs)
{
    json::Value a = json::Value::array();
    for (const sc::Result& r : rs)
        a.items.push_back(r.objectives.empty() ? json::Value::null() : json::Value::number(r.objectives[0][0]));
    return dumped(a);
}

}   // namespace

int main()
{
    const sc::Start none;
    const sc::Problem R4 = rastrigin(4);

    // ── the core on the toy run ──────────────────────────────────────────────
    {
        // one member = the member's run, with every rule
        const sc::pf::Rule rules[4] = {sc::pf::Rule::fixed, sc::pf::Rule::matching, sc::pf::Rule::pursuit,
                                       sc::pf::Rule::halving};
        Toy alone(R4, 400, sc::strategy::start_seed(11, 0), 3, none);
        alone.advance(sc::Piece());
        const sc::Result a = alone.result();
        bool all_alike = true;
        for (const auto rule : rules)
            for (long long steps : {1LL, 3LL}) {
                sc::pf::Plan p = toy_plan({&R4}, 11, rule);
                p.piece.steps = steps;
                const sc::pf::Outcome o = toy_run(400, p, 3);
                all_alike = all_alike && o.ran[0] && same_result(o.results[0], a);
            }
        check(all_alike, "toy: one member = the member's run, every rule, pieces of 1 and 3 steps");

        // fixed, share_best off = K independent runs with the budget split evenly
        for (long long rp : {1LL, 4LL}) {
            sc::pf::Plan p = toy_plan({&R4, &R4, &R4}, 5, sc::pf::Rule::fixed);
            p.piece.steps = 2;
            p.round_pieces = rp;
            const sc::pf::Outcome o = toy_run(1000, p, 3);
            const std::vector<long long> share = sc::split_budget(1000, 3);
            bool alike = true;
            for (std::size_t m = 0; m < 3; ++m) {
                Toy t(R4, share[m], p.members[m].seed, 3, none);
                t.advance(sc::Piece());
                alike = alike && o.ran[m] && same_result(o.results[m], t.result());
            }
            check(alike && spent_of(o) <= 1000, "toy: fixed, 3 members, " + std::to_string(rp) +
                                                     " piece(s) a round = 3 independent runs of the even shares");
        }

        // a constant problem and the sphere: adaptive pursuit takes the sphere's P to 1 - (K - 1) pmin
        const sc::Problem C1 = constant(10, 1.0), C2 = constant(10, 2.0), S10 = sphere(10);
        for (std::size_t K : {2u, 3u}) {
            std::vector<const sc::Problem*> pr;
            for (std::size_t m = 0; m + 1 < K; ++m) pr.push_back(m == 0 ? &C1 : &C2);
            pr.push_back(&S10);
            sc::pf::Plan p = toy_plan(pr, 21, sc::pf::Rule::pursuit);
            p.piece.steps = 2;
            const long long budget = 6000;
            const sc::pf::Outcome o = toy_run(budget, p, 3, true);
            const double pmax = 1.0 - static_cast<double>(K - 1) * p.pmin;
            // the pieces of the second half
            long long late = 0, late_s = 0;
            for (const json::Value& row : o.history.items)
                if (row.items[0].i >= o.rounds / 2) {
                    ++late;
                    late_s += row.items[1].i == static_cast<long long>(K - 1) ? 1 : 0;
                }
            const double frac = late ? static_cast<double>(late_s) / static_cast<double>(late) : 0.0;
            char line[256];
            std::snprintf(line, sizeof line,
                          "toy: %zu members, %zu constant + sphere, adaptive pursuit: P(sphere) %.12f -> 1 - (K-1) "
                          "op_pmin = %.12f; its share of the later pieces %.3f, %lld rounds",
                          K, K - 1, o.P[K - 1], pmax, frac, o.rounds);
            check(std::fabs(o.P[K - 1] - pmax) < 1e-9 && std::fabs(frac - pmax) < 0.08 && spent_of(o) <= budget, line);
            sc::pf::Plan q = p;
            q.rule = sc::pf::Rule::matching;
            const sc::pf::Outcome om = toy_run(budget, q, 3, true);
            std::snprintf(line, sizeof line, "toy: %zu members, probability matching: P(sphere) %.6f > P(constant) %.6f",
                          K, om.P[K - 1], om.P[0]);
            check(om.P[K - 1] > om.P[0] && spent_of(om) <= budget, line);
        }

        // successive halving on four constant problems: the better half each round; r_0 + r_1 over the
        // members spend the whole budget (4 x 150 + 2 x 300 = 1200), so there is no last piece (PF-6)
        {
            const sc::Problem c4 = constant(3, 4.0), c3 = constant(3, 3.0), c2 = constant(3, 2.0), c1 = constant(3, 1.0);
            sc::pf::Plan p = toy_plan({&c4, &c2, &c3, &c1}, 3, sc::pf::Rule::halving);
            const sc::pf::Outcome o = toy_run(1200, p, 3);
            check(dumped(o.halving) == "[[0,1,2,3],[1,3]]" && spent_of(o) == 1200 && o.pieces[3] == 2 &&
                      o.pieces[1] == 2 && o.pieces[0] == 1 && o.pieces[2] == 1,
                  "toy: successive halving keeps the better half: " + dumped(o.halving) + ", " +
                      std::to_string(spent_of(o)) + " of 1200 spent");
        }

        // 1 thread = 3 threads; within the budget; share_best hands points
        {
            bool alike = true, within = true, shared = true;
            for (const auto rule : rules)
                for (int sb = 0; sb < 2; ++sb) {
                    sc::pf::Plan p = toy_plan({&R4, &R4, &R4, &R4}, 7, rule);
                    p.share_best = sb == 1;
                    if (rule != sc::pf::Rule::halving) {
                        p.round_pieces = 4;
                        p.piece.steps = 3;
                    }
                    const sc::pf::Outcome o1 = toy_run(1500, p, 3);
                    p.threads = 3;
                    const sc::pf::Outcome o3 = toy_run(1500, p, 3);
                    alike = alike && same_outcome(o1, o3);
                    within = within && spent_of(o1) <= 1500;
                    if (sb == 1) shared = shared && o1.offered > 0 && o1.accepted > 0 && o1.accepted <= o1.offered;
                }
            check(alike, "toy: every rule, share_best off and on, 4 pieces a round: 1 thread = 3 threads");
            check(within, "toy: every rule: the members within the budget");
            check(shared, "toy: share_best: the best point offered and taken");
        }
    }

    // ── the library's resumable strategies ───────────────────────────────────
    const std::string bh = std::string("{\"inner\":") + kNM + ",\"solver\":\"basin_hopping\"}";
    const std::string cga = std::string("{\"cga_pop\":4,\"inner\":") + kNM + ",\"solver\":\"cluster_ga\"}";
    const std::string gas = "{\"np\":12,\"solver\":\"ga_sbx\"}";
    const sc::Start from = [] {
        sc::Start s;
        s.has_x0 = true;
        s.x0 = {2.0, -1.0, 0.5, 3.0};
        return s;
    }();
    {
        // one member = the strategy's run, bit for bit
        for (const std::string& js : {bh, cga, gas}) {
            const sc::strategy::Inner in = sc::strategy::inner_of("inner", word(js));
            const sc::Result plain = run(in.name, R4, in.knobs, 3000, 7, from);
            bool alike = true;
            for (const char* sel : {"fixed", "adaptive_pursuit", "probability_matching", "successive_halving"}) {
                const sc::Knobs k1{{"members", word("[" + js + "]")}, {"op_select", word(sel)}};
                const sc::Result one = run("portfolio_run", R4, k1, 3000, 7, from);
                alike = alike && same_result(plain, one) && member_rows_first(plain, one);
                if (std::string(sel) == "successive_halving") continue;
                sc::Knobs k3 = k1, k97 = k1;
                k3.push_back({"piece_steps", number(3)});
                k97.push_back({"piece_budget", number(97)});
                alike = alike && same_result(plain, run("portfolio_run", R4, k3, 3000, 7, from)) &&
                        same_result(plain, run("portfolio_run", R4, k97, 3000, 7, from));
            }
            check(alike, "portfolio_run with one member = " + in.name +
                             "'s run, bit for bit (every rule; pieces of 1 step, 3 steps, 97 of the budget)");
        }

        // fixed, share_best off: three different strategies = three independent runs of the even shares
        const std::string members = "[" + bh + "," + cga + "," + gas + "]";
        const std::vector<long long> share = sc::split_budget(3001, 3);
        std::vector<sc::Result> plain;
        long long evals = 0;
        for (std::size_t m = 0; m < 3; ++m) {
            const sc::strategy::Inner in = sc::strategy::inner_of("inner", word(m == 0 ? bh : m == 1 ? cga : gas));
            plain.push_back(run(in.name, R4, in.knobs, share[m], sc::strategy::start_seed(9, m), from));
            evals += plain.back().evaluations;
        }
        const std::string want = answers(plain);
        const sc::Knobs fx{{"members", word(members)}};
        const sc::Result r0 = run("portfolio_run", R4, fx, 3001, 9, from);
        sc::Knobs fx3 = fx;
        fx3.push_back({"piece_steps", number(2)});
        fx3.push_back({"round_pieces", number(3)});
        fx3.push_back({"threads", number(3)});
        const sc::Result r1 = run("portfolio_run", R4, fx3, 3001, 9, from);
        check(setting(r0, "pf_best") == want && r0.evaluations == evals && setting(r1, "pf_best") == want &&
                  r1.evaluations == evals,
              "fixed, basin_hopping + cluster_ga + ga_sbx = 3 independent runs of the even shares, " + want +
                  " (pieces of 1 step; 3 pieces of 2 steps a round, 3 threads)");

        // 1 thread = 4 threads, adaptive pursuit with share_best; within the budget
        const std::string bhc = std::string("{\"inner\":") + kNM + ",\"solver\":\"basin_hopping\",\"start_budget\":40}";
        const std::string cgac =
            std::string("{\"cga_pop\":4,\"inner\":") + kNM + ",\"solver\":\"cluster_ga\",\"start_budget\":40}";
        const std::string capped = "[" + bhc + "," + cgac + "," + gas + "," + bhc + "]";
        for (const char* sel : {"adaptive_pursuit", "fixed"}) {
            const sc::Knobs ap{{"members", word(capped)}, {"op_select", word(sel)}, {"share_best", word("on")},
                               {"round_pieces", number(4)}, {"piece_steps", number(2)}};
            sc::Knobs ap4 = ap;
            ap4.push_back({"threads", number(4)});
            const sc::Result a = run("portfolio_run", R4, ap, 8000, 3, from), b = run("portfolio_run", R4, ap4, 8000, 3, from);
            check(same_result(a, b) && same_rows(a, b) && a.evaluations <= 8000 &&
                      setting(a, "pf_shared_accepted") != "0",
                  std::string(sel) + ", share_best on, 4 pieces a round: 1 thread = 4 threads, " + setting(a, "rounds") +
                      " rounds, " + std::to_string(a.evaluations) + " evaluations <= 8000, best points taken " +
                      setting(a, "pf_shared_accepted") + " of " + setting(a, "pf_shared_offered") + ", shares " +
                      setting(a, "pf_share"));
        }

        // leftover: a member ended by its stall leaves its rest unspent (keep) or to the others (share)
        {
            const std::string bhs = std::string("{\"inner\":") + kNM +
                                    ",\"solver\":\"basin_hopping\",\"start_budget\":40,\"stall\":2}";
            const std::string two = "[" + bhs + "," + cgac + "]";
            const sc::Knobs keep{{"members", word(two)}};
            sc::Knobs shared = keep;
            shared.push_back({"leftover", word("share")});
            const sc::Result a = run("portfolio_run", R4, keep, 6000, 4, from);
            const sc::Result b = run("portfolio_run", R4, shared, 6000, 4, from);
            const std::string w = setting(a, "pf_withheld");
            const long long withheld = w.empty() ? 0 : std::stoll(w);
            check(setting(a, "pf_stopped_by").rfind("[\"stall\"", 0) == 0 && withheld > 0 &&
                      a.evaluations <= 6000 - withheld && setting(b, "pf_withheld").empty() &&
                      b.evaluations > a.evaluations && b.evaluations <= 6000,
                  "leftover: basin_hopping ended by its stall " + setting(a, "pf_stopped_by") + "; keep: " + w +
                      " withheld, " + std::to_string(a.evaluations) + " evaluations; share: " +
                      std::to_string(b.evaluations) + " evaluations <= 6000");
        }

        // successive halving: four members, two rounds (then the last to the end where the pool has some left)
        {
            const sc::Knobs sh{{"members", word(capped)}, {"op_select", word("successive_halving")}};
            const sc::Result a = run("portfolio_run", R4, sh, 6000, 5, from), b = run("portfolio_run", R4, sh, 6000, 5, from);
            const json::Value h = json::parse(setting(a, "pf_halving"));
            check(h.items.size() >= 2 && h.items.size() <= 3 && h.items[0].items.size() == 4 &&
                      h.items[1].items.size() == 2 && (h.items.size() == 2 || h.items[2].items.size() == 1) &&
                      a.evaluations <= 6000 && same_result(a, b),
                  "successive halving: rounds " + setting(a, "pf_halving") + ", " + std::to_string(a.evaluations) +
                      " evaluations <= 6000, the answer member " + setting(a, "pf_answer"));
        }
    }

    // ── refusals ─────────────────────────────────────────────────────────────
    {
        auto refused = [&](const sc::Knobs& k) {
            try {
                run("portfolio_run", R4, k, 500, 1);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        const std::string two = "[" + bh + "," + gas + "]";
        check(refused({}), "portfolio_run: no members refused");
        check(refused({{"members", word("[{\"solver\":\"lbfgsb\"}]")}}), "portfolio_run: a member not resumable refused");
        check(refused({{"members", word("[1, 2]")}}), "portfolio_run: a member that is not a solver refused");
        check(refused({{"members", word(two)}, {"member_seeds", word("[null]")}}),
              "portfolio_run: member_seeds of another length refused");
        check(refused({{"members", word(two)}, {"op_select", word("adaptive_pursuit")}, {"op_pmin", number(0.5)}}),
              "portfolio_run: op_pmin >= 1/K refused");
        check(refused({{"members", word(two)}, {"op_select", word("successive_halving")}, {"piece_steps", number(2)}}),
              "portfolio_run: pieces with successive_halving refused");
        check(refused({{"members", word(two)}, {"round_pieces", number(0)}}), "portfolio_run: round_pieces = 0 refused");
        check(refused({{"members", word(two)}, {"member_rng", word("[1, -2]")}}),
              "portfolio_run: a negative member_rng refused");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

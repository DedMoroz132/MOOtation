// SPDX-License-Identifier: Apache-2.0
// The resumable run of a strategy (task 31, layer 3; scalar/resumable.hpp):
//   * the interface can be implemented and driven through a pointer to it;
//   * best_member and member_of read the archive;
//   * carry_points' helpers: takes_population, carry_start, split_budget;
//   * a strategy without the resumable run is refused by start_resumable;
//   * RES-1, the layer's check: basin_hopping, ils, cluster_ga (one
//     population, islands, multi_dynamic, afpo, operator portfolios), ga_sbx
//     and ssga — scalar::run, the resumable run in one piece, in pieces of one
//     step and in pieces of a spend are one run, bit for bit: the same points
//     evaluated in the same order, the same answer, counts and settings; the
//     interface's reads between the pieces change nothing;
//   * set_budget before the first piece is the run with that budget;
//   * admit: the member planted (not evaluated), by each Place and by the
//     caller's rule; cluster_ga's islands refuse it; on a problem with
//     residuals (SEED-2) the member is evaluated and counted, its f the
//     problem's, for basin_hopping and ga_sbx;
//   * operators and set_operators carry the portfolios' weights; a portfolio
//     of other operators is refused;
//   * the time of scalar::run (max_seconds) in the result (RES-8);
//   * carry_points in shrink (PARAM-10): 1 is the run of before; N points
//     carried to every level, as instances of basin_hopping or as cluster_ga's
//     seed points; a solver that is not resumable carries one; beside
//     shrink_carry = population refused.

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mootation/scalar/resumable.hpp>

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

sc::Problem sphere(std::size_t n)
{
    sc::Problem P;
    P.lo.assign(n, -5.0);
    P.hi.assign(n, 5.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (double v : x) s += v * v;
        f.assign(1, s);
    };
    return P;
}

double rastrigin_of(const Vec& x)
{
    double s = 10.0 * static_cast<double>(x.size());
    for (double v : x) s += v * v - 10.0 * std::cos(2.0 * 3.14159265358979323846 * v);
    return s;
}

// Rastrigin in 4 variables; every point evaluated goes to *seen, in order
sc::Problem rastrigin(std::vector<Vec>* seen)
{
    sc::Problem P;
    P.lo.assign(4, -5.12);
    P.hi.assign(4, 5.12);
    P.m = 1;
    P.evaluate = [seen](const Vec& x, Vec& f, Vec&) {
        if (seen) seen->push_back(x);
        f.assign(1, rastrigin_of(x));
    };
    auto shift = [](std::size_t j) {
        return [j](Vec& y, std::mt19937& g) { y[j] += 0.5 * (2.0 * mootation::rnd::uniform01(g) - 1.0); };
    };
    P.mutate_ops = {{"x0", shift(0)}, {"x1", shift(1)}, {"x2", shift(2)}};
    auto blend = [](double w) {
        return [w](const Vec& a, const Vec& b, Vec& c1, Vec& c2, std::mt19937&) {
            for (std::size_t j = 0; j < a.size(); ++j) {
                c1[j] = w * a[j] + (1.0 - w) * b[j];
                c2[j] = (1.0 - w) * a[j] + w * b[j];
            }
        };
    };
    P.crossover_ops = {{"half", blend(0.5)}, {"third", blend(1.0 / 3.0)}};
    return P;
}

// Rosenbrock's residuals in [-5, 5]^2, with their Jacobian (SEED-2)
sc::Problem rosenbrock_residuals()
{
    sc::Problem P;
    P.lo.assign(2, -5.0);
    P.hi.assign(2, 5.0);
    P.m = 1;
    P.residuals = [](const Vec& x, Vec& r) { r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]}; };
    P.jacobian = [](const Vec& x, sc::Rows& J) { J = {{-20.0 * x[0], 10.0}, {-1.0, 0.0}}; };
    return P;
}

// Rastrigin in 2 variables on [-L, L]^2: a parameter for shrink (PARAM-10)
sc::Problem rastrigin_at(double L)
{
    sc::Problem P;
    P.lo.assign(2, -L);
    P.hi.assign(2, L);
    P.m = 1;
    P.param = L;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, rastrigin_of(x)); };
    P.at_param = [](double L2) { return rastrigin_at(L2); };
    P.rescale = [](Vec& x, double from, double to) {
        for (double& v : x) v *= to / from;
    };
    return P;
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.cv == b.cv &&
           a.evaluations == b.evaluations && a.grad_calls == b.grad_calls && a.settings == b.settings &&
           a.ignored == b.ignored && a.population == b.population && a.stopped_by == b.stopped_by &&
           a.used == b.used;
}

const char* kNM = "{\"solver\":\"nelder_mead\"}";

// The smallest run behind the interface: a step evaluates the next point of
// a line, one member. It shows the interface can be implemented and driven.
class Line : public sc::Resumable {
public:
    explicit Line(const sc::Problem& P) : ev_(P, 0.0, false, 0.0, 10, 1, {}, 0) {}

    const std::string& name() const override { return name_; }
    void begin() override
    {
        if (begun_) return;
        begun_ = true;
        at(0);
    }
    bool advance(const sc::Piece& piece) override
    {
        begin();
        const double from = ev_.counted();
        long long taken = 0;
        while (!done()) {
            if (piece.steps > 0 && taken >= piece.steps) break;
            if (piece.spend > 0.0 && ev_.counted() - from >= piece.spend) break;
            at(++steps_);
            ++taken;
        }
        return !done();
    }
    bool done() const override { return ev_.left() <= 0; }
    long long steps() const override { return steps_; }
    const sc::Evaluator& ledger() const override { return ev_; }
    void set_budget(long long budget) override { ev_.budget = ev_.limit = budget; }
    std::vector<sc::Member> members() const override { return {cur_}; }
    std::vector<sc::Member> best_points(std::size_t k) const override
    {
        return k == 0 ? std::vector<sc::Member>{} : std::vector<sc::Member>{best_};
    }
    long long admit(const std::vector<sc::Member>& in, sc::Place) override
    {
        if (in.empty()) return 0;
        cur_ = in[0];
        return 1;
    }
    long long admit(const std::vector<sc::Member>& in, const sc::PlaceRule& rule) override
    {
        long long taken = 0;
        for (const sc::Member& m : in)
            if (rule(members(), m) == 0) {
                cur_ = m;
                ++taken;
            }
        return taken;
    }
    sc::Operators operators() const override { return {}; }
    void set_operators(const sc::Operators&) override {}
    sc::Member best() const override { return best_; }
    std::vector<std::pair<std::string, std::string>> summary() const override
    {
        return {{"steps", std::to_string(steps_)}};
    }
    sc::Result result() override
    {
        sc::Result r;
        r.evaluations = ev_.spent;
        r.settings = summary();
        return r;
    }

private:
    sc::Evaluator ev_;
    std::string name_ = "line";
    bool begun_ = false;
    long long steps_ = 0;
    sc::Member cur_, best_;

    void at(long long s)
    {
        const Vec x(ev_.lo.size(), -4.0 + 0.5 * static_cast<double>(s));
        const std::size_t i = ev_.index(x);
        cur_.x = x;
        cur_.f = ev_.F[i][0];
        cur_.cv = 0.0;
        if (cur_.f < best_.f) best_ = cur_;
    }
};

struct Case {
    std::string name;
    sc::Knobs knobs;
};

std::unique_ptr<sc::Resumable> start(const Case& c, const sc::Problem& P, long long budget)
{
    return sc::start_resumable(c.name, P, 1e-6, false, 0.0, 1, budget, 11, c.knobs, {}, 0, {});
}

// the interface's reads, between pieces (they must change nothing)
void read_all(const sc::Resumable& r)
{
    (void)r.members();
    (void)r.best_points(3);
    (void)r.best();
    (void)r.operators();
    (void)r.summary();
    (void)r.steps();
    (void)r.ledger().left();
}

}   // namespace

int main()
{
    // the interface, driven through a pointer to it
    {
        const sc::Problem P = sphere(2);
        std::unique_ptr<sc::Resumable> r = std::make_unique<Line>(P);
        r->advance(sc::Piece{0.0, 3});
        check(r->steps() == 3 && r->ledger().spent == 4, "a piece of 3 steps: begin and 3 steps, 4 evaluations");
        r->advance(sc::Piece{2.0, 0});
        check(r->steps() == 5 && r->ledger().spent == 6, "a piece of spend 2: 2 more steps");
        r->set_budget(9);
        check(!r->advance(), "advance() runs to the run's end");
        check(r->ledger().spent == 9 && r->done(), "set_budget: the run ends at the new budget");
        check(r->best().f == 0.0, "the best point: the line crosses 0");
        const sc::Member b = sc::best_member(r->ledger());
        check(b.x == Vec{0.0, 0.0} && b.f == 0.0 && b.cv == 0.0 && b.g.empty(),
              "best_member: the archive's lowest f, with its values");
        check(sc::member_of(r->ledger(), {9.0, 9.0}, 1.0, 2.0).f == 1.0 &&
                  sc::member_of(r->ledger(), {-4.0, -4.0}, 1.0, 2.0).f == 32.0,
              "member_of: the archive's values where it has x, else those given");
        sc::Member m;
        m.x = {1.0, 1.0};
        m.f = 2.0;
        m.cv = 0.0;
        check(r->admit({m}, sc::Place::worst) == 1 && r->members()[0].f == 2.0, "admit by a Place");
        check(r->admit({m}, [](const std::vector<sc::Member>&, const sc::Member&) { return -1LL; }) == 0,
              "admit by the caller's rule: none");
        check(r->result().settings[0].second == "8", "result: the summary");
    }
    // carry_points' helpers (RES-7)
    {
        check(!sc::takes_population("cluster_ga", {}), "cluster_ga alone takes one start point");
        check(sc::takes_population("cluster_ga", {{"cga_init", word("seed_points")}}),
              "cluster_ga with cga_init = seed_points takes a population");
        check(sc::takes_population("ga_sbx", {{"warm_start", word("on")}}) &&
                  !sc::takes_population("ssga", {}) && !sc::takes_population("basin_hopping", {}),
              "ga_sbx with warm_start = on takes a population; ssga alone and basin_hopping do not");
        const sc::Start st = sc::carry_start({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        check(st.has_x0 && st.x0 == Vec{1.0, 2.0} && st.has_seed_points && st.seed_points.size() == 2 &&
                  st.seed_points[1].x == Vec{5.0, 6.0} && st.seed_points[1].f.empty(),
              "carry_start: x0 the first point, seed points the others, their values unknown");
        const sc::Start one = sc::carry_start({{1.0, 2.0}});
        check(one.has_x0 && !one.has_seed_points, "carry_start of one point: x0 alone");
        check(sc::split_budget(10, 3) == std::vector<long long>{4, 3, 3} &&
                  sc::split_budget(9, 3) == std::vector<long long>{3, 3, 3},
              "split_budget: equal shares, the first ones a unit more");
    }
    // a strategy without the resumable run
    {
        const sc::Problem P = sphere(2);
        bool refused = false;
        try {
            sc::start_resumable("lbfgsb", P, 0.0, false, 0.0, 1, 100, 1, {}, {}, 0, {});
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "lbfgsb is refused by start_resumable");
    }

    // ── RES-1: one piece = k pieces, bit for bit ─────────────────────────────
    const sc::Knob nm = word(kNM);
    const std::vector<Case> cases{
        {"basin_hopping", {{"inner", nm}, {"start_budget", number(60)}}},
        {"basin_hopping", {{"inner", nm}, {"start_budget", number(60)}, {"step", word("portfolio")},
                           {"op_select", word("adaptive_pursuit")}}},
        {"ils", {{"inner", nm}, {"start_budget", number(60)}, {"ils_accept", word("restart")}, {"ils_restart", number(3)}}},
        {"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(6)}}},
        {"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(6)}, {"mutation", word("ops")},
                        {"crossover", word("ops")}, {"cga_mut_rate", number(0.5)}, {"op_select", word("adaptive_pursuit")}}},
        {"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(4)}, {"cga_islands", number(3)},
                        {"cga_migrate_every", number(2)}}},
        {"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(5)},
                        {"cga_replace", word("multi_dynamic")}}},
        {"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(5)}, {"cga_replace", word("afpo")}}},
        {"ga_sbx", {{"np", number(20)}}},
        {"ga_sbx", {{"np", number(20)}, {"ga_mutation", word("ops")}, {"op_select", word("probability_matching")}}},
        {"ssga", {{"np", number(20)}}},
    };
    const long long budget = 6000;                       // every strategy cut into several pieces
    for (std::size_t q = 0; q < cases.size(); ++q) {
        const Case& c = cases[q];
        const std::string tag = c.name + " #" + std::to_string(q);
        std::vector<Vec> s0, s1, s2, s3;
        const sc::Problem P0 = rastrigin(&s0), P1 = rastrigin(&s1), P2 = rastrigin(&s2), P3 = rastrigin(&s3);
        const sc::Result r0 = sc::run(c.name, P0, 1e-6, false, 0.0, 1, budget, 11, c.knobs, {}, 0, {});
        std::unique_ptr<sc::Resumable> a = start(c, P1, budget);
        a->advance();
        const sc::Result r1 = a->result();
        std::unique_ptr<sc::Resumable> b = start(c, P2, budget);
        long long pieces = 0;
        while (b->advance(sc::Piece{0.0, 1})) {
            read_all(*b);
            ++pieces;
        }
        const sc::Result r2 = b->result();
        std::unique_ptr<sc::Resumable> d = start(c, P3, budget);
        long long cuts = 0;
        while (d->advance(sc::Piece{97.0, 0})) {
            read_all(*d);
            ++cuts;
        }
        const sc::Result r3 = d->result();
        check(same(r0, r1) && s0 == s1, tag + ": scalar::run = the resumable run in one piece (" +
                                            std::to_string(r0.evaluations) + " evaluations)");
        check(same(r0, r2) && s0 == s2 && pieces > 3,
              tag + ": = " + std::to_string(pieces + 1) + " pieces of one step, the same points in the same order");
        check(same(r0, r3) && s0 == s3 && cuts > 3, tag + ": = " + std::to_string(cuts + 1) + " pieces of spend 97");
    }

    // set_budget before the first piece: the run with that budget (RES-5)
    {
        const Case c{"basin_hopping", {{"inner", nm}, {"start_budget", number(60)}}};
        std::vector<Vec> s0, s1;
        const sc::Problem P0 = rastrigin(&s0), P1 = rastrigin(&s1);
        const sc::Result r0 = sc::run(c.name, P0, 1e-6, false, 0.0, 1, 700, 11, c.knobs, {}, 0, {});
        std::unique_ptr<sc::Resumable> a = start(c, P1, 3000);
        a->set_budget(700);
        a->advance();
        check(same(r0, a->result()) && s0 == s1, "set_budget(700) before the first piece: the run of budget 700");
    }

    // admit (RES-3, RES-4)
    {
        const sc::Problem P = rastrigin(nullptr);
        sc::Member opt;
        opt.x = Vec(4, 0.0);
        opt.f = 0.0;
        opt.cv = 0.0;
        const Case bh{"basin_hopping", {{"inner", nm}, {"start_budget", number(60)}}};
        std::unique_ptr<sc::Resumable> w = start(bh, P, 1500);
        w->begin();
        const long long spent = w->ledger().spent;
        check(w->admit({opt}, sc::Place::worst) == 1 && w->members().size() == 1 && w->members()[0].f == 0.0 &&
                  w->ledger().spent == spent && w->best().f == 0.0,
              "basin_hopping: admit sets the walk's point, planted, not evaluated; the run's best is it");
        sc::Member worse = opt;
        worse.x = Vec(4, 1.0);
        worse.f = rastrigin_of(worse.x);
        check(w->admit({worse}, sc::Place::worst_if_better) == 0, "basin_hopping: worst_if_better keeps a better point");
        w->advance();
        check(w->result().objectives[0][0] == 0.0, "basin_hopping: the walk goes on from the admitted point");

        const Case cg{"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(5)}}};
        std::unique_ptr<sc::Resumable> g = start(cg, P, 1500);
        g->advance(sc::Piece{0.0, 2});
        const long long before = g->ledger().spent;
        check(g->admit({opt}, sc::Place::worst_if_better) == 1 && g->admit({opt}, sc::Place::worst_if_better) == 0 &&
                  g->ledger().spent == before,
              "cluster_ga: worst_if_better takes the optimum once (step 3's cga_de), not evaluated");
        bool has = false;
        for (const sc::Member& m : g->members()) has = has || m.f == 0.0;
        check(has && g->members().size() == 5 && g->best_points(1)[0].f == 0.0, "cluster_ga: the member is in, best first");
        const std::vector<sc::Member> two{worse, worse};
        check(g->admit(two, sc::Place::all) == 2 && g->members()[0].x == worse.x && g->members()[1].x == worse.x,
              "cluster_ga: all replaces the first members");
        check(g->admit({opt}, [](const std::vector<sc::Member>&, const sc::Member&) { return 4LL; }) == 1 &&
                  g->members()[4].f == 0.0,
              "cluster_ga: the caller's rule picks the place");
        const Case isl{"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(3)},
                                      {"cga_islands", number(2)}}};
        std::unique_ptr<sc::Resumable> i2 = start(isl, P, 1500);
        i2->begin();
        bool refused = false;
        try {
            i2->admit({opt}, sc::Place::worst);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused && i2->members().size() == 6, "cluster_ga with islands: members are every island's; admit refused");

        const Case ga{"ga_sbx", {{"np", number(10)}}};
        std::unique_ptr<sc::Resumable> h = start(ga, P, 1500);
        h->begin();
        const long long at = h->ledger().spent;
        check(h->admit({opt}, sc::Place::worst) == 1 && h->best_points(1)[0].f == 0.0 && h->ledger().spent == at &&
                  h->members().size() == 10,
              "ga_sbx: admit puts the optimum in the worst member's place, not evaluated");
        h->advance();
        check(h->result().objectives[0][0] == 0.0, "ga_sbx: the run keeps it (elitist)");
    }

    // admit on a problem with residuals (SEED-2, the owner's choice of 18.11): a member carries no
    // residuals, so it is evaluated and counted, and its f is the problem's, not the one it brought
    {
        const sc::Problem R = rosenbrock_residuals();
        sc::Member at_min;
        at_min.x = {1.0, 1.0};
        at_min.f = 5.0;                                // wrong on purpose: the problem's f there is 0
        at_min.cv = 0.0;
        // basin_hopping's first lm run ends at (1, 1) itself, already in the archive: a point off the minimum
        sc::Member off_min;
        off_min.x = {0.5, 0.25};
        off_min.f = 0.0;                               // wrong on purpose: the problem's f there is 0.125
        off_min.cv = 0.0;
        const Case bh{"basin_hopping", {{"inner", word("{\"solver\":\"lm\"}")}, {"start_budget", number(60)}}};
        std::unique_ptr<sc::Resumable> w = start(bh, R, 1500);
        w->begin();
        const long long spent = w->ledger().spent;
        check(w->admit({off_min}, sc::Place::worst) == 1 && w->ledger().spent == spent + 1 &&
                  w->members().size() == 1 && std::fabs(w->members()[0].f - 0.125) < 1e-15,
              "basin_hopping on residuals: admit evaluates the member and counts it, its f the problem's (SEED-2)");
        w->advance();
        check(w->done() && w->result().objectives[0][0] == 0.0 && w->result().evaluations <= 1500,
              "basin_hopping on residuals: the walk goes on from it to the end (SEED-2)");

        const Case ga{"ga_sbx", {{"np", number(10)}}};
        std::unique_ptr<sc::Resumable> h = start(ga, R, 1500);
        h->begin();
        const long long at = h->ledger().spent;
        check(h->admit({at_min}, sc::Place::worst) == 1 && h->ledger().spent == at + 1 &&
                  h->best_points(1)[0].f == 0.0 && h->members().size() == 10,
              "ga_sbx on residuals: admit evaluates the member and counts it, its f the problem's (SEED-2)");
        h->advance();
        check(h->result().objectives[0][0] == 0.0, "ga_sbx on residuals: the run keeps it (elitist)");
    }

    // operators (RES-6)
    {
        const sc::Problem P = rastrigin(nullptr);
        const Case c{"cluster_ga", {{"inner", nm}, {"start_budget", number(60)}, {"cga_pop", number(6)},
                                    {"mutation", word("ops")}, {"crossover", word("ops")}, {"cga_mut_rate", number(0.5)},
                                    {"op_select", word("adaptive_pursuit")}}};
        std::unique_ptr<sc::Resumable> a = start(c, P, 1500), b = start(c, P, 1500);
        a->advance(sc::Piece{0.0, 5});
        const sc::Operators o = a->operators();
        b->set_operators(o);
        check(o.mutation.on() && o.crossover.on() && b->operators().mutation.P == o.mutation.P &&
                  b->operators().crossover.Q == o.crossover.Q && o.mutation.P != std::vector<double>(3, 1.0 / 3.0),
              "cluster_ga: the portfolios' weights carried by set_operators");
        sc::Operators other = o;
        other.mutation.names = {"a", "b", "c"};
        bool refused = false;
        try {
            b->set_operators(other);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "set_operators: a portfolio of other operators refused");
        const Case w{"basin_hopping", {{"inner", nm}, {"start_budget", number(60)}}};
        std::unique_ptr<sc::Resumable> h = start(w, P, 1500);
        h->advance(sc::Piece{0.0, 12});
        sc::Operators s = h->operators();
        check(s.has_step && s.step != 0.05, "basin_hopping: the adapted step, " + std::to_string(s.step));
        s.step = 0.2;
        h->set_operators(s);
        check(h->operators().step == 0.2, "basin_hopping: set_operators sets the step");
    }

    // the time of scalar::run (RES-8)
    {
        const sc::Problem P = rastrigin(nullptr);
        const Case c{"basin_hopping", {{"inner", nm}, {"start_budget", number(60)}, {"max_seconds", number(1000)}}};
        const sc::Result r0 = sc::run(c.name, P, 1e-6, false, 0.0, 1, 800, 11, c.knobs, {}, 0, {});
        std::unique_ptr<sc::Resumable> a = start(c, P, 800);
        a->advance();
        const sc::Result r1 = a->result();
        check(same(r0, r1) && setting(r1, "max_seconds") == "1000" && setting(r1, "time_up") == "no" &&
                  !setting(r1, "spend").empty(),
              "max_seconds: the deadline's settings and the ledger's, as scalar::run's");
    }

    // carry_points in shrink (PARAM-10)
    {
        const sc::Problem P = rastrigin_at(5.12);
        sc::Start from;
        from.has_x0 = true;
        from.x0 = {3.0, -2.0};
        const char* bh = "{\"inner\":{\"solver\":\"nelder_mead\"},\"solver\":\"basin_hopping\",\"start_budget\":60}";
        const char* cga = "{\"cga_init\":\"seed_points\",\"cga_pop\":4,\"inner\":{\"solver\":\"nelder_mead\"},"
                          "\"solver\":\"cluster_ga\",\"start_budget\":60}";
        auto shrink = [&](const char* inner, double carry) {
            sc::Knobs k{{"shrink_step", word("ladder")}, {"shrink_ladder", word("[5.12, 4, 3]")},
                        {"inner", word(inner)}};
            if (carry > 0) k.push_back({"carry_points", number(carry)});
            return sc::run("shrink", P, 1e-6, false, 0.0, 1, 3000, 7, k, {}, 0, from);
        };
        const sc::Result d = shrink(bh, 0), one = shrink(bh, 1);
        check(d.objectives == one.objectives && d.variables == one.variables && d.evaluations == one.evaluations &&
                  setting(one, "shrink_carried_points") == "[0,1,1]",
              "carry_points = 1: the run of before, bit for bit");
        const sc::Result three = shrink(bh, 3), again = shrink(bh, 3);
        check(setting(three, "shrink_carried_points") == "[0,3,3]" && same(three, again) && three.evaluations <= 3000,
              "carry_points = 3, basin_hopping: 3 points to every level, an instance each, " +
                  setting(three, "shrink_carried_points"));
        const sc::Result pop = shrink(cga, 3);
        check(setting(pop, "shrink_carried_points") == "[0,3,3]" && setting(pop, "init_seeded") == "2",
              "carry_points = 3, cluster_ga with seed_points: one run from the 3 points (x0 and 2 seeds)");
        // what carry promises a population ga: the points arrive as x0 and the design's rows, evaluated first
        {
            std::vector<Vec> seen;
            sc::Problem G = rastrigin_at(4.0);
            G.evaluate = [&seen](const Vec& x, Vec& f, Vec&) {
                seen.push_back(x);
                f.assign(1, rastrigin_of(x));
            };
            const sc::Knobs gk{{"np", number(10)}, {"warm_start", word("on")}};
            const sc::strategy::Inner gi{"ga_sbx", gk};
            const sc::Rows pts{{1.0, -1.0}, {2.0, 0.5}, {-3.0, 3.5}};
            const sc::detail::CarryLevel cl =
                sc::detail::carry_level(gi, G, 1e-6, false, 0.0, 1, 500, 7, sc::Start(), pts, 3);
            check(seen.size() >= 3 && seen[0] == pts[0] && seen[1] == pts[1] && seen[2] == pts[2] &&
                      cl.parts.size() == 1 && cl.parts[0].evaluations <= 500 && !cl.points.empty() &&
                      cl.points.size() <= 3 && setting(cl.parts[0], "warm_start") == "on",
                  "carry_points = 3, ga_sbx with warm_start: one run, the 3 points its first rows, evaluated first");
        }
        const sc::Result lb = shrink(kNM, 3);
        check(setting(lb, "shrink_carried_points") == "[0,1,1]",
              "carry_points = 3, nelder_mead (not resumable): it holds one point");
        bool refused = false;
        try {
            sc::Knobs k{{"shrink_carry", word("population")}, {"carry_points", number(2)}, {"inner", word(cga)}};
            sc::run("shrink", P, 1e-6, false, 0.0, 1, 1000, 7, k, {}, 0, from);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "carry_points beside shrink_carry = population refused");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

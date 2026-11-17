// SPDX-License-Identifier: Apache-2.0
// The resumable run of a strategy (task 31, layer 3; scalar/resumable.hpp):
//   * the interface can be implemented and driven through a pointer to it;
//   * best_member and member_of read the archive;
//   * carry_points' helpers: takes_population, carry_start, split_budget;
//   * a strategy without the resumable run is refused by start_resumable.

#include <cstdio>
#include <memory>
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
    // a strategy without the resumable run (RES: resumable_names)
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
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

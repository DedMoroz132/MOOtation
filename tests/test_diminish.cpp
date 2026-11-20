// SPDX-License-Identifier: Apache-2.0
// The diminishing returns of the campaign's loops (task 32, layer 5;
// scalar/stop_rule.hpp STOP-8, portfolio_run.hpp PF-10, campaign/family.hpp
// FAM-11, campaign/components.hpp COMP-12), on synthetic problems whose
// round k gains exactly c q^k (q = 1/2, so that every total is exact):
//   * the rule told the gains: the ratio rule ends after the first round k
//     with g_k < diminish_ratio g_1, the absolute one after the first with
//     g_k < diminish_abs, at the round counted by hand; rounds not measured
//     neither end the loop nor give g_1; g_1 is the first gain > 0;
//   * portfolio_run's rounds (one member whose best goes down by c q^k at
//     its k-th step; the first round is not measured: no point before it);
//     on the library's members, 1 thread = 4 threads, stopped_by and
//     pf_gains in the settings;
//   * the family's campaign (instance n's value goes down by q^k at its k-th
//     evaluation, a piece one evaluation): the round of as many pieces as
//     instances, with records given (round 1 measured) and without (round 1
//     not: new records); a campaign stopped by max_pieces, loaded and run on,
//     is the campaign in one piece; a stage after one the rule ended
//     measures anew; a call without diminish_* drops the measure;
//   * the loop over the parts (a part's quality goes down by q^k at its k-th
//     evaluation, instance n the sum of its parts' qualities): without a
//     bank and with one, the loop loaded and run on;
//   * the refusals.
// The default runs (no diminish_*) are checked bit for bit against the
// commit before the layer by the drivers of the task's report.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/campaign/components.hpp>
#include <mootation/campaign/family.hpp>
#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
namespace fam = mootation::campaign::fam;
namespace comp = mootation::campaign::comp;
namespace json = mootation::json;
using Vec = std::vector<double>;
using Key = comp::Key;
using Diminish = sc::strategy::Diminish;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const std::vector<std::pair<std::string, std::string>>& s, const std::string& key)
{
    for (const auto& kv : s)
        if (kv.first == key) return kv.second;
    return "";
}

std::string read_text(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

std::string utf8(const std::filesystem::path& p)
{
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

std::filesystem::path temp_dir(const std::string& what)
{
    const std::filesystem::path d = std::filesystem::temp_directory_path() /
                                    ("mootation_test_diminish_" + what + "_" + std::to_string(std::rand()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

bool throws(const std::function<void()>& f)
{
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// q^k, q = 1/2
double half_pow(long long k) { return std::ldexp(1.0, static_cast<int>(-k)); }

// the gains [c q^1, c q^2, ...] as the settings write them (null: not measured)
std::string gains_text(const std::vector<double>& g)
{
    json::Value a = json::Value::array();
    for (double v : g) a.items.push_back(std::isnan(v) ? json::Value::null() : json::Value::number(v));
    return json::dump(a);
}

// THE LADDER: a problem whose j-th evaluation (j = 0, 1, ...) gives
// top - c (q + q^2 + ... + q^j), whatever the point: its value goes down by
// c q^j at the j-th. The calls are counted in *calls (the test's).
sc::Problem ladder(std::size_t dim, double top, double c, std::shared_ptr<long long> calls)
{
    sc::Problem P;
    P.lo.assign(dim, 0.0);
    P.hi.assign(dim, 1.0);
    P.m = 1;
    P.evaluate = [top, c, calls](const Vec&, Vec& f, Vec&) {
        const long long j = (*calls)++;
        double s = 0.0;
        for (long long i = 1; i <= j; ++i) s += half_pow(i);
        f.assign(1, top - c * s);
    };
    return P;
}

// a resumable run of one point a step on the ladder (each step a new point,
// so that the archive never answers from its memory)
class Walker : public sc::Resumable {
public:
    Walker(const sc::Problem& P, long long budget) : ev_(P, 0.0, false, 0.0, budget, 1, {}, 0) {}

    const std::string& name() const override { return name_; }
    void begin() override
    {
        if (begun_) return;
        begun_ = true;
        step_point();
    }
    bool advance(const sc::Piece& piece) override
    {
        begin();
        long long taken = 0;
        while (!over_ && taken < (piece.steps > 0 ? piece.steps : 1)) {
            if (ev_.left() <= 0) {
                over_ = true;
                break;
            }
            step_point();
            ++taken;
            ++steps_;
        }
        return !over_;
    }
    bool done() const override { return over_; }
    long long steps() const override { return steps_; }
    const sc::Evaluator& ledger() const override { return ev_; }
    void set_budget(long long budget) override { ev_.budget = ev_.limit = budget; }
    std::vector<sc::Member> members() const override { return {best()}; }
    std::vector<sc::Member> best_points(std::size_t) const override { return {best()}; }
    long long admit(const std::vector<sc::Member>&, sc::Place) override { return 0; }
    long long admit(const std::vector<sc::Member>&, const sc::PlaceRule&) override { return 0; }
    sc::Operators operators() const override { return {}; }
    void set_operators(const sc::Operators&) override {}
    sc::Member best() const override { return sc::best_member(ev_); }
    std::vector<std::pair<std::string, std::string>> summary() const override { return {}; }
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
        r.stopped_by = "budget";
        return r;
    }

private:
    sc::Evaluator ev_;
    std::string name_ = "walker";
    bool begun_ = false, over_ = false;
    long long steps_ = 0, points_ = 0;

    void step_point()
    {
        if (ev_.left() <= 0) {
            over_ = true;
            return;
        }
        ev_.index(Vec{static_cast<double>(points_++) / 4096.0});
    }
};

bool same_result(const sc::Result& a, const sc::Result& b)
{
    std::vector<std::pair<std::string, std::string>> x, y;
    for (const auto& kv : a.settings)
        if (kv.first != "threads") x.push_back(kv);
    for (const auto& kv : b.settings)
        if (kv.first != "threads") y.push_back(kv);
    return a.variables == b.variables && a.objectives == b.objectives && a.cv == b.cv &&
           a.evaluations == b.evaluations && a.used == b.used && a.population == b.population &&
           a.stopped_by == b.stopped_by && x == y;
}

// ── the family: instances 1, 2, 3, weights 1, 2, 4, instance n a ladder of
// one variable from 4 down by q^k (a piece one evaluation) ─────────────────
struct LadderFamily {
    std::map<long long, std::shared_ptr<long long>> calls;
    fam::Family F;

    LadderFamily()
    {
        for (long long n : {1, 2, 3}) calls[n] = std::make_shared<long long>(0);
        F.instances = {1, 2, 3};
        auto c = calls;
        F.problem_at = [c](long long n) { return ladder(1, 4.0, 1.0, c.at(n)); };
        F.weight = fam::table_of({{1, 1.0}, {2, 2.0}, {3, 4.0}}, "weight");
    }
    long long all_calls() const
    {
        long long s = 0;
        for (const auto& kv : calls) s += *kv.second;
        return s;
    }
};

fam::Options family_options()
{
    fam::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 4;
    o.budget = 1000;
    o.piece_budget = 1;
    o.seed = 3;
    return o;
}

// records given to every instance (their evaluation j = 0: the value 4)
void give_all(const fam::Family& F, fam::Bank& B)
{
    fam::Options o = family_options();
    fam::Campaign c(F, B, o);
    for (long long n : F.instances) c.give(n, Vec{0.5});
}

std::string bank_text_of(const fam::Bank& B, const fam::Family& F)
{
    return fam::bank_text(B, fam::total_of(B, F.weight), F.instances.size(), "");
}

// ── the parts: keys {1}, {2}, {3}, part {s} a ladder of one variable from 2
// down by q^k; instance n = 1, 2, 3 is part {n}, instance 4 parts {1} and {3}:
// a round (one piece a part) gains 5 q^k in sum_n f(n) (w = 1) ─────────────
struct LadderParts {
    std::map<Key, std::shared_ptr<long long>> calls;
    comp::Components C;
    fam::Family F;

    LadderParts()
    {
        for (long long s : {1, 2, 3}) calls[Key{s}] = std::make_shared<long long>(0);
        C.keys = {Key{1}, Key{2}, Key{3}};
        auto c = calls;
        C.component_problem = [c](const Key& k) { return ladder(1, 2.0, 1.0, c.at(k)); };
        C.assemble = [](long long n, const comp::View& v) {
            comp::Assembled a;
            const std::vector<Key> use = n == 4 ? std::vector<Key>{Key{1}, Key{3}} : std::vector<Key>{Key{n}};
            double f = 0.0;
            for (const Key& k : use) {
                if (!v.has(k)) return a;
                f += v.quality(k);
            }
            a.ok = true;
            a.f = f;
            a.x = Vec{f};                    // instance n's point: its value (problem f(x) = x_0)
            a.used_keys = use;
            return a;
        };
        F.instances = {1, 2, 3, 4};
        F.problem_at = [](long long) {
            sc::Problem P;
            P.lo.assign(1, -100.0);
            P.hi.assign(1, 100.0);
            P.m = 1;
            P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, x[0]); };
            return P;
        };
    }
};

comp::Options parts_options()
{
    comp::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 4;
    o.budget = 1000;
    o.piece_budget = 1;
    o.seed = 5;
    return o;
}

void give_parts(LadderParts& L, comp::Table& T, fam::Bank* B)
{
    comp::Options o = parts_options();
    comp::Loop l(L.F, L.C, T, B, o);
    for (long long s : {1, 2, 3}) l.give(Key{s}, Vec{0.5});
}

}   // namespace

int main()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // ── the rule told the gains c q^k ────────────────────────────────────────
    {
        auto end_round = [](double ratio, double abs, double c, const std::vector<double>& before) {
            Diminish d;
            d.ratio = ratio;
            d.abs = abs;
            for (double g : before)
                if (d.round(g)) return -d.rounds;                // ended on a round given before the ladder
            for (long long k = 1; k <= 200; ++k)
                if (d.round(c * half_pow(k))) return d.rounds;
            return 0LL;
        };
        // ratio 0.1: q^(k-1) < 0.1 first at k = 5 (1/16); 0.2: at k = 4 (1/8); c does not matter
        check(end_round(0.1, 0.0, 1.0, {}) == 5, "the ratio 0.1: ends after round 5 (q^4 = 1/16 < 0.1 <= q^3)");
        check(end_round(0.2, 0.0, 3.0, {}) == 4, "the ratio 0.2, c = 3: ends after round 4 (q^3 = 1/8 < 0.2 <= q^2)");
        check(end_round(0.125, 0.0, 1.0, {}) == 5, "the ratio 1/8: q^3 = 1/8 is not less, ends after round 5");
        // abs: c q^k < a; c = 1, a = 0.01: k = 7 (1/128); c = 3, a = 0.05: 3/64 < 0.05 <= 3/32, k = 6
        check(end_round(0.0, 0.01, 1.0, {}) == 7, "diminish_abs 0.01: ends after round 7 (q^7 = 1/128)");
        check(end_round(0.0, 0.05, 3.0, {}) == 6, "diminish_abs 0.05, c = 3: ends after round 6 (3/64)");
        check(end_round(0.0, 0.5, 1.0, {}) == 2, "diminish_abs 0.5: round 1's 1/2 is not less, round 2's 1/4 is");
        check(end_round(0.0, 0.75, 1.0, {}) == 1, "diminish_abs 0.75: the first measured round ends it");
        check(end_round(0.1, 0.01, 1.0, {}) == 5, "both: the first that fires (the ratio at 5)");
        check(end_round(0.01, 0.05, 1.0, {}) == 5, "both: the first that fires (abs at 5: 1/32 < 0.05)");
        // rounds not measured: neither end nor give g1
        check(end_round(0.1, 0.0, 1.0, {nan, nan}) == 7, "two rounds not measured first: g1 is round 3's, ends at 7");
        check(end_round(0.0, 0.01, 1.0, {nan}) == 8, "a round not measured, then abs: round 1 + 7");
        {
            Diminish d;
            d.ratio = 0.5;
            bool ended = d.round(0.0);
            for (int k = 0; k < 50; ++k) ended = ended || d.round(0.0) || d.round(-1.0);
            check(!ended && !d.has_ref, "gains of 0 (and losses) before any gain: no g1, the ratio does not end the loop");
            check(!d.round(1.0) && d.has_ref && d.ref == 1.0 && !d.round(0.5) && d.round(0.25),
                  "... the first gain > 0 is g1 (1); 0.5 is not less than 0.5 g1, 0.25 is");
            Diminish e;
            e.ratio = 0.5;
            e.round(2.0);
            check(e.round(-1.0), "after g1 a loss (a negative gain) ends it");
        }
        {
            Diminish d;
            check(!d.on() && !d.round(0.0) && !d.round(-5.0), "off (0, 0): never ends");
            d = Diminish();
            d.ratio = 0.1;
            d.round(nan);
            d.round(0.5);
            d.round(0.25);
            check(gains_text(d.gains) == "[null, 0.5, 0.25]" && json::dump(d.gains_json()) == "[null, 0.5, 0.25]",
                  "the gains as JSON, null where not measured");
        }
        check(throws([] { Diminish::check(-0.1, 0.0, "t"); }) && throws([] { Diminish::check(0.0, -1.0, "t"); }) &&
                  throws([&] { Diminish::check(nan, 0.0, "t"); }) &&
                  throws([] { Diminish::check(0.0, std::numeric_limits<double>::infinity(), "t"); }) &&
                  !throws([] { Diminish::check(0.0, 0.0, "t"); }) && !throws([] { Diminish::check(2.0, 3.0, "t"); }),
              "the knobs: >= 0 and finite");
    }

    // ── portfolio_run's rounds (pf::run on the walker) ───────────────────────
    {
        // one member: round 1 = its first point and one step (not measured: no point before), round k >= 2
        // gains c q^k (c = 2). ratio 0.1 against g1 = round 2's: q^(k-2) < 0.1 first at k = 6;
        // abs 0.03: 2 q^k < 0.03 first at k = 7 (2/128)
        auto walk = [](double ratio, double abs, long long budget) {
            auto calls = std::make_shared<long long>(0);
            const sc::Problem P = ladder(1, 10.0, 2.0, calls);
            sc::pf::Plan plan;
            plan.members.resize(1);
            plan.members[0].problem = &P;
            plan.members[0].name = "walker";
            plan.piece.steps = 1;
            plan.diminish.ratio = ratio;
            plan.diminish.abs = abs;
            sc::strategy::StopRule stop;
            const sc::pf::Make make = [](const sc::pf::Spec& s, long long b, std::size_t) {
                return std::unique_ptr<sc::Resumable>(std::make_unique<Walker>(*s.problem, b));
            };
            return sc::pf::run(P, budget, plan, stop, make);
        };
        std::vector<double> want{nan};
        for (long long k = 2; k <= 6; ++k) want.push_back(2.0 * half_pow(k));
        const sc::pf::Outcome a = walk(0.1, 0.0, 1000);
        check(a.why == "diminishing" && a.rounds == 6 && gains_text(a.diminish.gains) == gains_text(want),
              "portfolio rounds, the ratio 0.1: ends after round 6; gains [null, 2q^2 ... 2q^6] exactly (" +
                  std::to_string(a.rounds) + " rounds, " + gains_text(a.diminish.gains) + ")");
        const sc::pf::Outcome b = walk(0.0, 0.03, 1000);
        check(b.why == "diminishing" && b.rounds == 7, "portfolio rounds, diminish_abs 0.03: ends after round 7 (" +
                                                           std::to_string(b.rounds) + ")");
        const sc::pf::Outcome c = walk(0.0, 0.0, 30);
        check(c.why != "diminishing" && c.rounds == 29 && c.diminish.gains.empty(),
              "off: the walker runs to its budget (29 rounds of 30 evaluations), nothing measured");
    }

    // ── portfolio_run on the library's members: stopped_by, pf_gains, 1 thread = 4 threads ──
    {
        sc::Problem S;                                           // Rastrigin, 4 variables
        S.lo.assign(4, -5.12);
        S.hi.assign(4, 5.12);
        S.m = 1;
        S.evaluate = [](const Vec& x, Vec& f, Vec&) {
            double s = 40.0;
            for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
            f.assign(1, s);
        };
        const std::string members =
            "[{\"solver\":\"ga_sbx\"},{\"solver\":\"cluster_ga\",\"inner\":{\"solver\":\"nelder_mead\"}},"
            "{\"solver\":\"ssga\"}]";
        auto run_with = [&](const std::string& sel, int threads, double ratio) {
            sc::Knobs k{{"members", word(members)}, {"op_select", word(sel)}, {"round_pieces", number(3)},
                        {"threads", number(threads)}};
            if (ratio > 0.0) k.push_back({"diminish_ratio", number(ratio)});
            return sc::run("portfolio_run", S, 1e-6, false, 0.0, 12, 20000, 7, k, {}, 0, {});
        };
        for (const std::string sel : {"fixed", "adaptive_pursuit"}) {
            const sc::Result r1 = run_with(sel, 1, 0.05);
            const sc::Result r4 = run_with(sel, 4, 0.05);
            const sc::Result off = run_with(sel, 1, 0.0);
            const std::string gains = setting(r1.settings, "pf_gains");
            check(r1.stopped_by == "diminishing" && setting(r1.settings, "stopped_by") == "diminishing" &&
                      gains.rfind("[null,", 0) == 0,
                  sel + ": the library's members end by the diminishing returns (stopped_by " + r1.stopped_by +
                      ", pf_gains " + gains.substr(0, 300) + ")");
            check(same_result(r1, r4), sel + ": diminish_ratio, 1 thread = 4 threads (the result and the settings)");
            check(r1.used < off.used && setting(off.settings, "pf_gains").empty() &&
                      setting(off.settings, "stopped_by").empty() && setting(off.settings, "diminish_ratio").empty(),
                  sel + ": it ends before the budget's run (" + std::to_string(r1.used) + " of " +
                      std::to_string(off.used) + "); without the knob no row of it");
            check(setting(r1.settings, "diminish_ratio") == "0.05", sel + ": the knob in the settings");
        }
        check(throws([&] {
                  sc::run("portfolio_run", S, 1e-6, false, 0.0, 12, 2000, 7,
                          sc::Knobs{{"members", word(members)}, {"diminish_ratio", number(-0.5)}}, {}, 0, {});
              }),
              "portfolio_run: diminish_ratio < 0 refused");
    }

    // ── the family's campaign ────────────────────────────────────────────────
    {
        // records given (each instance's value 4): round 1 (one piece each) gains (1 + 2 + 4) q = 7 q, round k 7 q^k
        auto campaign = [](double ratio, double abs, bool given, LadderFamily& L, fam::Bank& B) {
            if (given) give_all(L.F, B);
            fam::Options o = family_options();
            o.diminish_ratio = ratio;
            o.diminish_abs = abs;
            return fam::run(L.F, B, o);
        };
        {
            LadderFamily L;
            fam::Bank B;
            const fam::Report r = campaign(0.1, 0.0, true, L, B);
            std::vector<double> want;
            for (long long k = 1; k <= 5; ++k) want.push_back(7.0 * half_pow(k));
            check(r.stopped_by == "diminishing" && B.pieces.size() == 15 &&
                      setting(r.settings, "family_gains") == gains_text(want) &&
                      setting(r.settings, "family_stopped_by") == "diminishing",
                  "family, records given, ratio 0.1: ends after round 5 (15 pieces), gains 7q ... 7q^5 (" +
                      std::to_string(B.pieces.size()) + " pieces, " + setting(r.settings, "family_gains") + ")");
            check(L.all_calls() == 3 + 15, "one evaluation a piece (" + std::to_string(L.all_calls()) + " calls)");
            check(r.total == 7.0 * (4.0 - (1.0 - half_pow(5))), "the total exactly 7 (4 - (q + ... + q^5))");
        }
        {
            LadderFamily L;
            fam::Bank B;
            const fam::Report r = campaign(0.0, 0.06, true, L, B);
            check(r.stopped_by == "diminishing" && B.pieces.size() == 21,
                  "family, diminish_abs 0.06: 7 q^k < 0.06 first at k = 7 (21 pieces; " +
                      std::to_string(B.pieces.size()) + ")");
        }
        {
            LadderFamily L;
            fam::Bank B;
            const fam::Report r = campaign(0.1, 0.0, false, L, B);
            // without records: round 1 makes them (its evaluations j = 0: not measured), round k >= 2 gains 7 q^(k-1);
            // g1 = round 2's 7 q, and q^(k-2) < 0.1 first at k = 6
            check(r.stopped_by == "diminishing" && B.pieces.size() == 18 &&
                      setting(r.settings, "family_gains").rfind("[null, 3.5,", 0) == 0,
                  "family without records: round 1 not measured (new records), ends after round 6 (" +
                      std::to_string(B.pieces.size()) + " pieces, " + setting(r.settings, "family_gains") + ")");
        }
        {
            LadderFamily L;
            fam::Bank B;
            give_all(L.F, B);
            fam::Options o = family_options();
            o.diminish_ratio = 0.1;
            o.round_pieces = 6;                                  // two pieces an instance a round: 7 (q^(2k-1) + q^(2k))
            const fam::Report r = fam::run(L.F, B, o);
            // g_k = 7 * 3 q^(2k), g1 = 21/4; g_k < 0.1 g1: q^(2k-2) < 0.1 first at k = 3
            check(r.stopped_by == "diminishing" && B.pieces.size() == 18,
                  "family, round_pieces 6: ends after round 3 (18 pieces; " + std::to_string(B.pieces.size()) + ")");
        }
        // loaded and run on = one piece; the file's diminish state
        {
            const std::filesystem::path dir = temp_dir("family");
            LadderFamily L1;
            fam::Bank one;
            give_all(L1.F, one);
            fam::Options o = family_options();
            o.diminish_ratio = 0.1;
            o.bank_file = utf8(dir / "one.json");
            fam::run(L1.F, one, o);
            const std::string one_text = read_text(dir / "one.json");
            LadderFamily L2;
            fam::Bank part;
            give_all(L2.F, part);
            o.bank_file = utf8(dir / "part.json");
            bool same = true;
            for (long long stop : {4LL, 8LL, 13LL}) {
                o.max_pieces = stop;
                fam::Bank b = part;
                const fam::Report r = fam::run(L2.F, b, o);
                same = same && r.stopped_by == "pieces";
                part = fam::load_bank(o.bank_file);
                same = same && bank_text_of(part, L2.F) == bank_text_of(b, L2.F);
            }
            o.max_pieces = 0;
            const fam::Report r = fam::run(L2.F, part, o);
            const std::string part_text = read_text(dir / "part.json");
            check(same && r.stopped_by == "diminishing" && part_text == one_text,
                  "family: stopped by max_pieces (4, 8, 13: mid-round), loaded and run on = one piece, byte for byte");
            check(one_text.find("\"diminish\": {\"start\": 15,") != std::string::npos &&
                      one_text.find("\"diminish_ratio\": 0.1") != std::string::npos,
                  "the file keeps the measure (state's diminish) and the options");
            // a stage after the one the rule ended: a new measure from piece 15, g1 = 7 q^6, ends at its round 5
            fam::Bank stage = fam::load_bank(o.bank_file);
            const fam::Report s = fam::run(L2.F, stage, o);
            check(s.stopped_by == "diminishing" && stage.pieces.size() == 30 && stage.measure.start == 30 &&
                      stage.measure.rule.gains.size() == 5 && stage.measure.rule.ref == 7.0 * half_pow(6),
                  "family: the next stage measures anew (ends after its round 5: 30 pieces)");
            // a call without diminish_* drops the measure: the bank as the campaign without it
            fam::Options plain = family_options();
            plain.max_pieces = 33;
            fam::Bank dropped = stage;
            fam::run(L2.F, dropped, plain);
            check(!dropped.measure.on && bank_text_of(dropped, L2.F).find("diminish") == std::string::npos &&
                      setting(fam::settings_of(dropped, 0.0, 3, ""), "family_gains").empty(),
                  "family: a call without diminish_* drops the measure (no diminish in the bank's text)");
            std::filesystem::remove_all(dir);
        }
        // without diminish_*: the text names nothing of it
        {
            LadderFamily L;
            fam::Bank B;
            give_all(L.F, B);
            fam::Options o = family_options();
            o.max_pieces = 9;
            const fam::Report r = fam::run(L.F, B, o);
            check(r.stopped_by == "pieces" && bank_text_of(B, L.F).find("diminish") == std::string::npos &&
                      B.options.find("round_pieces") == nullptr && setting(r.settings, "family_gains").empty(),
                  "family off: no measure, no options of it, no family_gains");
        }
        // refusals
        {
            LadderFamily L;
            fam::Bank B;
            fam::Options o = family_options();
            o.diminish_ratio = -0.1;
            const bool a = throws([&] { fam::run(L.F, B, o); });
            o.diminish_ratio = 0.1;
            o.round_pieces = -1;
            const bool b = throws([&] { fam::run(L.F, B, o); });
            o.round_pieces = 0;
            o.diminish_abs = std::numeric_limits<double>::infinity();
            const bool c = throws([&] { fam::run(L.F, B, o); });
            check(a && b && c, "family: diminish_ratio < 0, round_pieces < 0, diminish_abs infinite refused");
        }
    }

    // ── the loop over the parts ──────────────────────────────────────────────
    {
        // a round (one piece a part) gains 5 q^k; ratio 0.1: k = 5 (15 pieces); abs 0.04: 5 q^k < 0.04 at k = 7
        for (const bool with_bank : {false, true}) {
            const std::string at = with_bank ? "parts with a bank" : "parts without a bank";
            {
                LadderParts L;
                comp::Table T;
                fam::Bank B;
                give_parts(L, T, with_bank ? &B : nullptr);
                comp::Options o = parts_options();
                o.diminish_ratio = 0.1;
                const comp::Report r = comp::run(L.F, L.C, T, with_bank ? &B : nullptr, o);
                std::vector<double> want;
                for (long long k = 1; k <= 5; ++k) want.push_back(5.0 * half_pow(k));
                check(r.stopped_by == "diminishing" && T.pieces.size() == 15 &&
                          setting(r.settings, "components_gains") == gains_text(want) &&
                          setting(r.settings, "components_stopped_by") == "diminishing",
                      at + ", ratio 0.1: ends after round 5 (15 pieces), gains 5q ... 5q^5 (" +
                          std::to_string(T.pieces.size()) + ", " + setting(r.settings, "components_gains") + ")");
                if (with_bank)
                    check(fam::total_of(B, nullptr) == 5.0 * (2.0 - (1.0 - half_pow(5))) && B.records.size() == 4,
                          at + ": the bank's total exactly 5 (2 - (q + ... + q^5))");
            }
            {
                LadderParts L;
                comp::Table T;
                fam::Bank B;
                give_parts(L, T, with_bank ? &B : nullptr);
                comp::Options o = parts_options();
                o.diminish_abs = 0.04;
                const comp::Report r = comp::run(L.F, L.C, T, with_bank ? &B : nullptr, o);
                check(r.stopped_by == "diminishing" && T.pieces.size() == 21,
                      at + ", diminish_abs 0.04: ends after round 7 (21 pieces; " + std::to_string(T.pieces.size()) +
                          ")");
            }
        }
        // loaded and run on = one piece (the table and the bank)
        {
            const std::filesystem::path dir = temp_dir("parts");
            comp::Options o = parts_options();
            o.diminish_ratio = 0.1;
            LadderParts L1;
            comp::Table T1;
            fam::Bank B1;
            give_parts(L1, T1, &B1);
            o.table_file = utf8(dir / "t1.json");
            o.bank_file = utf8(dir / "b1.json");
            comp::run(L1.F, L1.C, T1, &B1, o);
            LadderParts L2;
            comp::Table T2;
            fam::Bank B2;
            give_parts(L2, T2, &B2);
            o.table_file = utf8(dir / "t2.json");
            o.bank_file = utf8(dir / "b2.json");
            bool same = true;
            for (long long stop : {2LL, 7LL, 11LL}) {
                o.max_pieces = stop;
                same = same && comp::run(L2.F, L2.C, T2, &B2, o).stopped_by == "pieces";
                T2 = comp::load_table(o.table_file);
                B2 = fam::load_bank(o.bank_file);
            }
            o.max_pieces = 0;
            const comp::Report r = comp::run(L2.F, L2.C, T2, &B2, o);
            check(same && r.stopped_by == "diminishing" &&
                      read_text(dir / "t2.json") == read_text(dir / "t1.json") &&
                      read_text(dir / "b2.json") == read_text(dir / "b1.json"),
                  "parts: stopped by max_pieces (2, 7, 11), loaded and run on = one piece (table and bank, byte for byte)");
            check(read_text(dir / "t1.json").find("\"diminish\": {\"start\": 15,") != std::string::npos &&
                      read_text(dir / "b1.json").find("diminish") == std::string::npos,
                  "the table keeps the measure; the bank's file (the loop's, not the campaign's) none");
            std::filesystem::remove_all(dir);
        }
        // off: nothing of it
        {
            LadderParts L;
            comp::Table T;
            give_parts(L, T, nullptr);
            comp::Options o = parts_options();
            o.max_pieces = 6;
            const comp::Report r = comp::run(L.F, L.C, T, nullptr, o);
            check(r.stopped_by == "pieces" && !T.measure.on && comp::table_text(T, 3, "").find("diminish") == std::string::npos &&
                      setting(r.settings, "components_gains").empty(),
                  "parts off: no measure, no components_gains");
        }
        // without keys: done, the measure not begun
        {
            LadderParts L;
            L.C.keys.clear();
            comp::Table T;
            comp::Options o = parts_options();
            o.diminish_ratio = 0.1;
            const comp::Report r = comp::run(L.F, L.C, T, nullptr, o);
            check(r.stopped_by == "done" && !T.measure.on, "parts without keys: done, nothing measured (COMP-11)");
        }
        {
            LadderParts L;
            comp::Table T;
            comp::Options o = parts_options();
            o.diminish_abs = -1.0;
            const bool a = throws([&] { comp::run(L.F, L.C, T, nullptr, o); });
            o.diminish_abs = 0.0;
            o.round_pieces = -2;
            const bool b = throws([&] { comp::run(L.F, L.C, T, nullptr, o); });
            check(a && b, "parts: diminish_abs < 0, round_pieces < 0 refused");
        }
    }

    if (g_failed) {
        std::printf("%d check(s) FAILED\n", g_failed);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}

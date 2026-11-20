// SPDX-License-Identifier: Apache-2.0
// A family of problems: the bank of the best solutions and the transfer
// between instances (task 32, layer 2; campaign/family.hpp):
//   * the family "sphere of dimension n" (n = 1 ... 6), the transfer adding
//     or taking off the last coordinate: the closure from one good record
//     lifts the neighbours, round by round, to the values the test works
//     out; closure_rounds stops it;
//   * a certify that rejects everything leaves the bank as it was; one that
//     accepts gives the record its certified f, cv 0, and its cost in used();
//   * the bank's file read back is the bank; a campaign stopped by
//     max_pieces, or by its budget inside a closure, loaded and run on, is
//     the campaign in one piece (every pick rule);
//   * one instance, one piece, no transfer: the bank's record is the
//     scalar::run of the solver, bit for bit (basin_hopping, cluster_ga,
//     layer 1's portfolio_run, nelder_mead); a problem's at_param levels;
//   * the pick rules piece by piece; the weighted total; the export into a
//     run store; the refusals.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/campaign/family.hpp>
#include <mootation/campaign/family_store.hpp>
#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
namespace fam = mootation::campaign::fam;
namespace st = mootation::store;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }

std::string read_text(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// a path as UTF-8 text (u8string is a std::u8string in C++20)
std::string utf8(const std::filesystem::path& p)
{
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

std::filesystem::path temp_dir(const std::string& what)
{
    const std::filesystem::path d = std::filesystem::temp_directory_path() /
                                    ("mootation_test_family_" + what + "_" + std::to_string(std::rand()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// the optimum's coordinates of the sphere family: a_j = 0.5 + 0.1 j
double a_of(std::size_t j) { return 0.5 + 0.1 * static_cast<double>(j); }

// instance n: sum_{j < n} (x_j - a_j)^2 on [-2, 2]^n
sc::Problem sphere_n(long long n)
{
    sc::Problem P;
    const std::size_t d = static_cast<std::size_t>(n);
    P.lo.assign(d, -2.0);
    P.hi.assign(d, 2.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) s += (x[j] - a_of(j)) * (x[j] - a_of(j));
        f.assign(1, s);
    };
    return P;
}

// the transfer: the last coordinate taken off (n - 1) or a 0 added (n + 1)
std::vector<fam::Candidate> sphere_transfer(long long n, const Vec& x)
{
    std::vector<fam::Candidate> out;
    fam::Candidate down{n - 1, Vec(x.begin(), x.end() - (x.empty() ? 0 : 1))};
    fam::Candidate up{n + 1, x};
    up.x.push_back(0.0);
    out.push_back(down);
    out.push_back(up);
    return out;
}

fam::Family sphere_family(bool with_transfer)
{
    fam::Family F;
    F.instances = {6, 5, 4, 3, 2, 1};                                     // kept in ascending order (FAM-1)
    F.problem_at = sphere_n;
    if (with_transfer) F.transfer = sphere_transfer;
    return F;
}

Vec optimum(long long n)
{
    Vec x;
    for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) x.push_back(a_of(j));
    return x;
}

std::string text_of(const fam::Bank& B, const fam::Family& F)
{
    const double total = fam::total_of(B, F.weight);
    return fam::bank_text(B, total, F.instances.size(), "");
}

fam::Options campaign_options(const std::string& pick, long long budget)
{
    fam::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.budget = budget;
    o.piece_budget = 60;
    o.pick = pick;
    o.seed = 11;
    return o;
}

// ── the closure ─────────────────────────────────────────────────────────────

void closure_lifts_the_neighbours()
{
    const fam::Family F = sphere_family(true);
    fam::Options o = campaign_options("round_robin", 1000);
    o.max_pieces = 0;
    {
        fam::Bank B;
        fam::Campaign c(F, B, o);
        c.give(4, optimum(4));
        // the closure runs first in run(); a campaign of 0 pieces: max_pieces would be
        // 0 = none, so run it with the budget that the closure alone spends
        fam::Options o2 = o;
        o2.budget = 11;                       // 1 (given) + 2 + 4 + 3 + 1 evaluations of the rounds
        fam::Campaign c2(F, B, o2);
        const fam::Report r = c2.run();
        const double a4 = a_of(4), a5 = a_of(5);
        check(B.records.size() == 6, "closure: all six instances have a record");
        bool exact = true;
        for (long long n = 1; n <= 4; ++n) exact = exact && B.records.at(n).f == 0.0 && B.records.at(n).x == optimum(n);
        check(exact, "closure: instances 1 ... 4 at their optimum, f = 0 exactly");
        check(B.records.at(5).f == a4 * a4 && B.records.at(6).f == a4 * a4 + a5 * a5,
              "closure: instance 5 at a_4^2, 6 at a_4^2 + a_5^2 (a 0 added)");
        check(B.records.at(3).source == "transfer" && B.records.at(3).from == 4 && B.records.at(1).from == 2 &&
                  B.records.at(4).source == "given",
              "closure: the records' sources and the instances they came from");
        check(B.closure_rounds == std::vector<long long>{4}, "closure: 4 rounds (the 4th makes nothing better)");
        check(B.used == 11.0 && B.transfer.offered == 10 && B.transfer.accepted == 5 && B.transfer_skipped == 2,
              "closure: 10 candidates evaluated and offered, 5 taken, n = 0 and 7 skipped, used 11");
        check(r.stopped_by == "budget" && B.pieces.empty() && B.now.empty(),
              "closure: done before the budget ended, no piece run");
        check(B.history.size() == 6 && B.history.back().second == r.total && r.total == a4 * a4 + (a4 * a4 + a5 * a5),
              "closure: the history of the total, its last = the total");
    }
    {
        fam::Bank B;
        fam::Options o1 = o;
        o1.closure_rounds = 1;
        o1.budget = 3;
        fam::Campaign c(F, B, o1);
        c.give(4, optimum(4));
        c.run();
        check(B.records.size() == 3 && B.records.count(3) && B.records.count(5) && B.closure_rounds.size() == 1 &&
                  B.closure_rounds[0] == 1,
              "closure_rounds = 1: only the neighbours 3 and 5");
    }
}

// ── certify ─────────────────────────────────────────────────────────────────

void certify_rejecting_everything()
{
    fam::Family F = sphere_family(true);
    fam::Bank B;
    {
        fam::Options o = campaign_options("round_robin", 600);
        fam::Campaign c(F, B, o);
        c.give(3, optimum(3));
        c.run();
    }
    const std::map<long long, fam::Record> before = B.records;
    const std::string records_before = [&] {
        std::string s;
        for (const auto& [n, r] : before) s += mootation::json::dump(fam::detail::record_json(n, r)) + "\n";
        return s;
    }();
    F.certify = [](long long, const Vec&) { return fam::Certified{false, 0.0, 0.5}; };
    fam::Options o = campaign_options("round_robin", 1200);
    const long long run_acc = B.run.accepted, tr_acc = B.transfer.accepted;
    fam::Campaign c(F, B, o);
    c.give(2, optimum(2));
    c.run();
    std::string records_after;
    for (const auto& [n, r] : B.records) records_after += mootation::json::dump(fam::detail::record_json(n, r)) + "\n";
    check(records_after == records_before, "certify rejects everything: the bank's records unchanged");
    check(B.run.accepted == run_acc && B.transfer.accepted == tr_acc && B.given.rejected == 1 && B.run.rejected > 0,
          "certify rejects everything: nothing accepted, the pieces' answers rejected");
    check(B.used <= 1200.0 + 0.5, "certify rejects everything: within the budget but for one certify's cost (FAM-5)");

    // a certify that accepts: the certified f, cv 0, the cost in used()
    fam::Family G = sphere_family(false);
    G.certify = [](long long n, const Vec&) { return fam::Certified{true, 100.0 + static_cast<double>(n), 2.5}; };
    fam::Bank C;
    fam::Options og = campaign_options("round_robin", 1000);
    fam::Campaign cg(G, C, og);
    cg.give(2, optimum(2));
    check(C.records.at(2).f == 102.0 && C.records.at(2).cv == 0.0 && C.used == 2.5,
          "certify accepts: f certified, cv 0, its cost the only spending (the point not evaluated)");
}

// ── the file and going on ───────────────────────────────────────────────────

void file_and_going_on(const std::string& pick)
{
    fam::Family F = sphere_family(true);
    if (pick != "round_robin") {
        F.weight = fam::table_of({{1, 1.0}, {2, 2.0}, {3, 1.0}, {4, 0.5}, {5, 1.0}, {6, 3.0}}, "weight");
        F.reference = [](long long) { return 0.0; };
    }
    const long long B2 = 1500;
    const std::filesystem::path dir = temp_dir("file_" + pick);
    // the campaign in one piece
    fam::Bank one;
    fam::Options o = campaign_options(pick, B2);
    {
        fam::Campaign c(F, one, o);
        c.give(1, Vec{0.3});
        c.run();
    }
    const std::string want = text_of(one, F);
    // read back
    fam::save_bank(utf8(dir / "one.json"), one, fam::total_of(one, F.weight), F.instances.size(), "");
    const fam::Bank back = fam::load_bank(utf8(dir / "one.json"));
    check(text_of(back, F) == want, pick + ": the bank read back from its file is the bank, byte for byte");
    check(!std::filesystem::exists(dir / "one.json.tmp"), pick + ": no temporary file left");

    // stopped by max_pieces, saved every piece, loaded, run on
    {
        fam::Bank A;
        fam::Options oa = o;
        oa.max_pieces = 7;
        oa.bank_file = utf8(dir / "a.json");
        fam::Campaign c(F, A, oa);
        c.give(1, Vec{0.3});
        const fam::Report ra = c.run();
        fam::Bank A2 = fam::load_bank(oa.bank_file);
        fam::Options ob = o;
        ob.bank_file = oa.bank_file;
        fam::Campaign c2(F, A2, ob);
        c2.run();
        check(ra.stopped_by == "pieces" && A.pieces.size() == 7, pick + ": the first call stopped by max_pieces at 7");
        check(text_of(A2, F) == want, pick + ": max_pieces, file, going on = the campaign in one piece");
        check(read_text(ob.bank_file) == fam::bank_text(A2, fam::total_of(A2, F.weight), F.instances.size(), "budget"),
              pick + ": the file at the end is the bank, stopped_by budget");
    }
    // stopped by the budget inside a closure (a piece of the whole budget before it)
    {
        long long B1 = 0;
        for (std::size_t p = 0; p + 1 < one.pieces.size() && B1 == 0; ++p) {
            const double u0 = one.pieces[p].used_at, u1 = one.pieces[p + 1].used_at;
            if (one.pieces[p].budget == 60 && u1 - u0 >= 62.0) B1 = static_cast<long long>(u1) - 1;
        }
        check(B1 > 0, pick + ": the campaign has a closure of two candidates or more after a whole piece");
        fam::Bank A;
        fam::Options oa = o;
        oa.budget = B1;
        oa.bank_file = utf8(dir / "b.json");
        fam::Campaign c(F, A, oa);
        c.give(1, Vec{0.3});
        c.run();
        check(!A.now.empty(), pick + ": the first call ended inside a closure (kept in the bank)");
        fam::Bank A2 = fam::load_bank(oa.bank_file);
        fam::Campaign c2(F, A2, o);
        c2.run();
        check(text_of(A2, F) == want, pick + ": the budget inside a closure, file, going on = the campaign in one piece");
    }
    // saving changes nothing
    {
        fam::Bank S;
        fam::Options os = o;
        os.bank_file = utf8(dir / "s.json");
        os.save_every = 3;
        fam::Campaign c(F, S, os);
        c.give(1, Vec{0.3});
        c.run();
        check(S.options.find("save_every")->i == 3, pick + ": the file keeps the call's options (save_every 3)");
        S.options = one.options;                                           // the one option that differs
        check(text_of(S, F) == want, pick + ": a campaign that saves its bank (every 3 pieces) = one that does not");
    }
    std::filesystem::remove_all(dir);
}

// ── one instance, one piece: scalar::run ────────────────────────────────────

void one_instance_is_the_run(const std::string& solver, const sc::Knobs& knobs)
{
    fam::Family F;
    F.instances = {5};
    F.problem_at = sphere_n;
    fam::Options o;
    o.solver = solver;
    o.knobs = knobs;
    o.pop_size = 10;
    o.budget = o.piece_budget = 700;
    o.max_pieces = 1;
    o.seed = 7;
    fam::Bank B;
    fam::run(F, B, o);
    const sc::Problem Q = sphere_n(5);
    const sc::Result r = sc::run(solver, Q, 1e-4, false, 0.0, 10, 700, 7, knobs, {}, 0, sc::Start{});
    check(B.records.size() == 1 && B.records.at(5).x == r.variables[0] && B.records.at(5).f == r.objectives[0][0] &&
              B.used == r.used && B.records.at(5).source == "run",
          solver + ": one instance, one piece = scalar::run, bit for bit (x, f, used)");
}

void levels_of_a_parameter()
{
    // a problem with at_param: the sphere around (L, L)
    sc::Problem P = sphere_n(2);
    P.param = 1.0;
    P.at_param = [](double L) {
        sc::Problem Q = sphere_n(2);
        Q.param = L;
        Q.evaluate = [L](const Vec& x, Vec& f, Vec&) { f.assign(1, (x[0] - L) * (x[0] - L) + (x[1] - L) * (x[1] - L)); };
        return Q;
    };
    fam::Family F;
    F.instances = {0, 1};
    F.problem_at = fam::levels_of(P);
    fam::Bank B;
    const fam::Options o = campaign_options("round_robin", 100);
    fam::Campaign c(F, B, o);
    c.give(1, Vec{1.0, 1.0});
    c.give(0, Vec{1.0, 1.0});
    check(B.records.at(1).f == 0.0 && B.records.at(0).f == 2.0, "levels_of: instance n is the problem at the level n");
}

// ── the picks ───────────────────────────────────────────────────────────────

void picks_piece_by_piece()
{
    {
        fam::Family F = sphere_family(false);
        fam::Bank B;
        fam::Options o = campaign_options("round_robin", 100000);
        o.max_pieces = 8;
        fam::run(F, B, o);
        std::vector<long long> got;
        for (const auto& p : B.pieces) got.push_back(p.n);
        check(got == std::vector<long long>({1, 2, 3, 4, 5, 6, 1, 2}), "round_robin: 1 ... 6, then again");
    }
    for (const std::string pick : {"gap", "gap_age"}) {
        fam::Family F = sphere_family(false);
        const std::map<long long, double> w{{1, 1.0}, {2, 2.0}, {3, 1.0}, {4, 0.5}, {5, 1.0}, {6, 3.0}};
        F.weight = fam::table_of(w, "weight");
        F.reference = fam::table_of({{1, 0.0}, {2, 0.0}, {3, 0.01}, {4, 0.0}, {5, 0.0}, {6, 0.0}}, "reference");
        fam::Bank B;
        fam::Options o = campaign_options(pick, 100000);
        o.piece_budget = 15;
        {
            fam::Campaign c(F, B, o);
            c.give(3, Vec{0.0, 0.0, 0.0});
            c.give(5, Vec{0.5, 0.6, 0.7, 0.8, 0.9});
        }
        bool all = true;
        for (long long k = 1; k <= 20; ++k) {
            // the pick the rule asks for, from the bank before the piece
            long long want = -1;
            double best = -std::numeric_limits<double>::infinity();
            long long best_age = -1;
            for (long long n = 1; n <= 6; ++n) {
                const auto t = B.touched.find(n);
                const long long age = static_cast<long long>(B.pieces.size()) - (t == B.touched.end() ? -1 : t->second);
                double p;
                if (pick == "gap") {
                    const auto it = B.records.find(n);
                    p = it == B.records.end() ? std::numeric_limits<double>::infinity()
                                              : w.at(n) * (it->second.f - (n == 3 ? 0.01 : 0.0));
                } else {
                    p = w.at(n) * static_cast<double>(age);
                }
                if (want < 0 || p > best || (pick == "gap" && p == best && age > best_age)) {
                    want = n;
                    best = p;
                    best_age = age;
                }
            }
            fam::Options ok = o;
            ok.max_pieces = k;
            fam::run(F, B, ok);
            all = all && B.pieces.size() == static_cast<std::size_t>(k) && B.pieces.back().n == want;
            if (k == 1 && pick == "gap")
                check(B.pieces.back().n == 1, "gap: an instance without a record first (the lowest n of them)");
        }
        check(all, pick + ": 20 pieces, each the instance the rule asks for from the bank before it");
        if (pick == "gap_age") {
            long long twos = 0, ones = 0;
            for (const auto& p : B.pieces) {
                twos += p.n == 6 ? 1 : 0;
                ones += p.n == 1 ? 1 : 0;
            }
            check(twos > ones, "gap_age: the instance of weight 3 more often than one of weight 1");
        }
    }
}

// ── the total, the export, the refusals ─────────────────────────────────────

void total_export_refusals()
{
    fam::Family F = sphere_family(true);
    F.weight = fam::table_of({{1, 1.0}, {2, 2.0}, {3, 1.0}, {4, 0.5}, {5, 1.0}, {6, 3.0}}, "weight");
    fam::Bank B;
    fam::Options o = campaign_options("round_robin", 500);
    fam::Campaign c(F, B, o);
    c.give(2, Vec{0.0, 0.0});
    const fam::Report r = c.run();
    double s = 0.0;
    for (const auto& [n, rec] : B.records) s += F.weight(n) * rec.f;
    check(r.total == s && c.total() == s, "the total: sum w(n) f(n) in ascending n");
    std::string missing;
    for (const auto& kv : r.settings)
        if (kv.first == "family_missing") missing = kv.second;
    check(missing == std::to_string(6 - B.records.size()), "family_missing: the instances without a record");

    const std::filesystem::path store = temp_dir("store");
    const st::Entry e = fam::export_bank_to_store(store, "family|sphere|bank|11|500", B, r.total, 6, 4242, "host");
    bool found = false;
    for (const auto& f : std::filesystem::directory_iterator(store)) {
        if (f.path().extension() != ".raw") continue;
        const std::string raw = read_text(f.path());
        const auto* buf = reinterpret_cast<const std::uint8_t*>(raw.data());
        const st::RecordAt at = st::parse_record(buf, raw.size(), static_cast<std::size_t>(e.offset));
        if (!at.ok || at.key != "family|sphere|bank|11|500") continue;
        const auto blocks = st::decode_run(buf + at.body, at.body_size);
        const st::Block* bank = st::run_json::named(blocks, "bank.json");
        const st::Block* rec = st::run_json::named(blocks, "records.csv");
        found = bank && rec && std::string(bank->data.begin(), bank->data.end()) == fam::bank_text(B, r.total, 6, "") &&
                rec->shape == std::vector<std::uint64_t>({B.records.size(), 7}) && e.status == "done";
    }
    check(found, "export: the bank one record of the store — bank.json the file's text, records.csv a row a record");
    std::filesystem::remove_all(store);

    auto refused = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const fam::Family plain = sphere_family(false);
    fam::Options og = campaign_options("gap", 100);
    check(refused([&] { fam::Bank b; fam::Campaign x(plain, b, og); }), "refused: gap without a reference");
    fam::Options ow = campaign_options("best", 100);
    check(refused([&] { fam::Bank b; fam::Campaign x(plain, b, ow); }), "refused: an unknown pick");
    check(refused([&] { fam::Bank b; fam::run(sphere_family(false), b, campaign_options("round_robin", 0)); }),
          "refused: budget 0");
    fam::Family twice = sphere_family(false);
    twice.instances.push_back(3);
    check(refused([&] { fam::Bank b; fam::Campaign x(twice, b, o); }), "refused: an instance twice");
    check(refused([&] {
              fam::Bank b;
              b.records[9].x = Vec(9, 0.0);
              fam::Campaign x(plain, b, o);
          }),
          "refused: a bank with an instance outside the family");
    check(refused([&] {
              fam::Bank b;
              b.records[2].x = Vec(3, 0.0);
              fam::Campaign x(plain, b, o);
          }),
          "refused: a bank's point not of its problem's length");
    check(refused([&] {
              fam::Bank b;
              fam::Campaign x(plain, b, o);
              x.give(2, Vec{0.0});
          }),
          "refused: a point given not of its problem's length");
    std::string text = fam::bank_text(B, r.total, 6, "");
    text.replace(text.find("\"version\": 1"), 12, "\"version\": 9");
    check(refused([&] { fam::bank_of_text(text); }), "refused: a bank file of another version");
}

}   // namespace

int main()
{
    try {
        closure_lifts_the_neighbours();
        certify_rejecting_everything();
        for (const std::string pick : {"round_robin", "gap", "gap_age"}) file_and_going_on(pick);
        one_instance_is_the_run("nelder_mead", {});
        const std::string nm = "{\"solver\":\"nelder_mead\",\"local_tol\":1e-8}";
        one_instance_is_the_run("basin_hopping", {{"inner", word(nm)}});
        one_instance_is_the_run("cluster_ga", {{"inner", word(nm)}, {"cga_pop", sc::Knob{false, "", 4.0}}});
        one_instance_is_the_run(
            "portfolio_run",
            {{"members", word("[{\"inner\":" + nm + ",\"solver\":\"basin_hopping\"},{\"np\":12,\"solver\":\"ga_sbx\"}]")},
             {"op_select", word("adaptive_pursuit")}});
        levels_of_a_parameter();
        picks_piece_by_piece();
        total_export_refusals();
    } catch (const std::exception& e) {
        std::printf("FAIL  exception: %s\n", e.what());
        ++g_failed;
    }
    if (g_failed) {
        std::printf("%d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}

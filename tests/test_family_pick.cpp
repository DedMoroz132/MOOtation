// SPDX-License-Identifier: Apache-2.0
// The family's picks of task 33.1 (campaign/family.hpp FAM-12, FAM-13):
//   * pick = gap_utility: an instance with the largest gap whose pieces
//     never better it. gap gives it every piece; gap_utility lowers its
//     utility by 0.95 a failed piece (MOEA/D-DRA's Step 5) and goes on to the
//     next instances once the utility times the gap is below their gaps — at
//     the piece the test works out; utility_factor = 0.5 sooner;
//   * fresh_first: records that came by an assembly (a label given) or a
//     transfer, and not yet worked on, get their pieces before the records of
//     runs, under every pick; without fresh records, the pick of before;
//   * every pick, with and without fresh_first, 20 pieces, each the instance
//     the rule asks for from the bank before it;
//   * the utilities in the bank's file, read back; a campaign stopped by
//     max_pieces, loaded and run on, is the campaign in one piece; a bank of
//     the other picks has nothing new in its file; the refusals.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/campaign/family.hpp>
#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
namespace fam = mootation::campaign::fam;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

std::string utf8(const std::filesystem::path& p)
{
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

std::filesystem::path temp_dir(const std::string& what)
{
    const std::filesystem::path d = std::filesystem::temp_directory_path() /
                                    ("mootation_test_family_pick_" + what + "_" + std::to_string(std::rand()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// instance 1: f = 2 on [-1, 1], a piece never betters it; instances 2 ... 5:
// f = (x - 0.3)^2. The reference 0, the weight 1.
sc::Problem problem_of(long long n)
{
    sc::Problem P;
    P.lo.assign(1, -1.0);
    P.hi.assign(1, 1.0);
    P.m = 1;
    if (n == 1)
        P.evaluate = [](const Vec&, Vec& f, Vec&) { f.assign(1, 2.0); };
    else
        P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, (x[0] - 0.3) * (x[0] - 0.3)); };
    return P;
}

fam::Family family(bool with_transfer)
{
    fam::Family F;
    F.instances = {1, 2, 3, 4, 5};
    F.problem_at = problem_of;
    F.reference = [](long long) { return 0.0; };
    // the transfer: n's point for n + 1 (the same box)
    if (with_transfer) F.transfer = [](long long n, const Vec& x) { return std::vector<fam::Candidate>{{n + 1, x}}; };
    return F;
}

fam::Options options(const std::string& pick, long long max_pieces)
{
    fam::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.budget = 100000;
    o.piece_budget = 30;
    o.max_pieces = max_pieces;
    o.pick = pick;
    o.seed = 5;
    return o;
}

std::vector<long long> pieces_of(const fam::Bank& B, std::size_t from)
{
    std::vector<long long> v;
    for (std::size_t p = from; p < B.pieces.size(); ++p) v.push_back(B.pieces[p].n);
    return v;
}

std::string text_of(const fam::Bank& B)
{
    return fam::bank_text(B, fam::total_of(B, {}), 5, "");
}

// the bank of the stuck instance: every instance given x = -1 (instance 1:
// gap 2, the others 1.69)
fam::Bank stuck_bank()
{
    fam::Bank B;
    fam::Campaign c(family(false), B, options("gap", 1));
    for (long long n = 1; n <= 5; ++n) c.give(n, Vec{-1.0});
    return B;
}

// FAM-13 from the bank (the test's own reading)
bool fresh_in(const fam::Bank& B, long long n)
{
    const auto it = B.records.find(n);
    if (it == B.records.end() || it->second.source == "run") return false;
    const auto t = B.touched.find(n);
    return t == B.touched.end() || t->second <= it->second.piece;
}

// FAM-6, FAM-12, FAM-13 from the bank before the piece (weight 1, reference 0)
long long want_pick(const fam::Bank& B, const std::string& pick, bool fresh_first)
{
    std::vector<long long> among;
    for (long long n = 1; n <= 5; ++n)
        if (fresh_first && fresh_in(B, n)) among.push_back(n);
    if (pick == "round_robin") {
        for (std::size_t k = 0; k < 5; ++k) {
            const long long n = static_cast<long long>((B.cursor + k) % 5) + 1;
            if (among.empty() || fresh_in(B, n)) return n;
        }
    }
    if (among.empty()) among = {1, 2, 3, 4, 5};
    const double inf = std::numeric_limits<double>::infinity();
    long long want = -1, best_age = -1;
    double best = -inf;
    for (long long n : among) {
        const auto t = B.touched.find(n);
        const long long age = static_cast<long long>(B.pieces.size()) - (t == B.touched.end() ? -1 : t->second);
        double p = static_cast<double>(age);
        if (pick != "gap_age") {
            const auto it = B.records.find(n);
            p = (it == B.records.end() || it->second.cv > 0.0) ? inf : it->second.f;
            const auto u = B.utility.find(n);
            if (pick == "gap_utility" && p > 0.0 && p < inf && u != B.utility.end()) p *= u->second;
        }
        if (want < 0 || p > best || (pick != "gap_age" && p == best && age > best_age)) {
            want = n;
            best = p;
            best_age = age;
        }
    }
    return want;
}

// ── gap_utility: the stuck instance ─────────────────────────────────────────

void stuck_instance()
{
    {
        fam::Bank B = stuck_bank();
        fam::run(family(false), B, options("gap", 20));
        check(pieces_of(B, 0) == std::vector<long long>(20, 1) && B.records.at(1).source == "given" &&
                  B.run.accepted == 0 && B.records.at(1).f == 2.0,
              "gap: all 20 pieces to the stuck instance 1 (its record unchanged)");
    }
    {
        fam::Bank B = stuck_bank();
        fam::run(family(false), B, options("gap_utility", 20));
        // 2 * 0.95^k > 1.69 for k = 0 ... 3, 2 * 0.95^4 = 1.629 < 1.69: piece 4 goes to instance 2
        // (the instances 2 ... 5 tie at 1.69 and at their age: the lowest n), then 3, 4, 5;
        // then instance 1 again, the others' gaps below 2 * 0.95^k all along
        std::vector<long long> want{1, 1, 1, 1, 2, 3, 4, 5};
        while (want.size() < 20) want.push_back(1);
        check(pieces_of(B, 0) == want, "gap_utility: 4 failed pieces on instance 1, then 2, 3, 4, 5, then 1 again");
        double pi = 1.0;
        for (int k = 0; k < 16; ++k) pi = 0.95 * pi;
        check(B.utility.at(1) == pi, "gap_utility: instance 1's utility 0.95^16 after 16 failed pieces");
        bool ones = true;
        for (long long n = 2; n <= 5; ++n) ones = ones && B.utility.at(n) == 1.0 && B.records.at(n).f < 1e-3;
        check(ones, "gap_utility: instances 2 ... 5 bettered by their piece, utility 1");
    }
    {
        fam::Bank B = stuck_bank();
        fam::Options o = options("gap_utility", 6);
        o.utility_factor = 0.5;
        fam::run(family(false), B, o);
        check(pieces_of(B, 0) == std::vector<long long>({1, 2, 3, 4, 5, 1}),
              "utility_factor 0.5: one failed piece on instance 1 (2 * 0.5 = 1 < 1.69), then 2 ... 5");
    }
}

// ── fresh_first ─────────────────────────────────────────────────────────────

// records of runs on every instance (round_robin, 5 pieces of 3 evaluations),
// then an assembled record for 3 whose closure carries its point to 4 and 5
fam::Bank fresh_bank()
{
    fam::Bank B;
    fam::Options o = options("round_robin", 5);
    o.piece_budget = 3;
    fam::run(family(false), B, o);
    fam::Campaign c(family(true), B, options("round_robin", 5));
    c.give(3, Vec{0.3}, "assembled");
    c.closure();
    return B;
}

void fresh_first()
{
    const fam::Bank B0 = fresh_bank();
    check(B0.records.at(1).source == "run" && B0.records.at(2).source == "run" &&
              B0.records.at(3).source == "assembled" && B0.records.at(4).source == "transfer" &&
              B0.records.at(5).source == "transfer" && B0.records.at(4).f == 0.0 && B0.pieces.size() == 5,
          "fresh: records of runs on 1, 2; an assembly on 3, transferred to 4 and 5");
    bool fresh_ok = !fresh_in(B0, 1) && !fresh_in(B0, 2) && fresh_in(B0, 3) && fresh_in(B0, 4) && fresh_in(B0, 5);
    check(fresh_ok, "fresh: the assembled and transferred records are fresh, the runs' are not");
    const std::map<std::string, std::vector<long long>> plain{
        {"round_robin", {1, 2, 3, 4}}, {"gap", {1, 1, 1, 1}}, {"gap_age", {1, 2, 3, 4}}, {"gap_utility", {1, 1, 1, 1}}};
    for (const auto& [pick, want_plain] : plain) {
        fam::Bank A = B0, F = B0;
        fam::run(family(true), A, options(pick, 9));
        fam::Options of = options(pick, 9);
        of.fresh_first = true;
        fam::run(family(true), F, of);
        check(pieces_of(A, 5) == want_plain, pick + ": without fresh_first the pick of before");
        check(pieces_of(F, 5) == std::vector<long long>({3, 4, 5, 1}),
              pick + " + fresh_first: the fresh records 3, 4, 5 first (in the pick's order), then the pick of before");
        check(!fresh_in(F, 3) && !fresh_in(F, 4) && !fresh_in(F, 5), pick + ": a piece given, the record no longer fresh");
    }
    // the bank of only run records: fresh_first changes nothing
    {
        fam::Bank A, F;
        fam::Options o = options("gap", 12);
        fam::run(family(false), A, o);
        o.fresh_first = true;
        fam::run(family(false), F, o);
        F.options = A.options;                                      // the one option that differs
        check(text_of(F) == text_of(A), "fresh_first without fresh records: the bank of before, byte for byte");
    }
}

// ── every pick, piece by piece ──────────────────────────────────────────────

void picks_piece_by_piece()
{
    for (const std::string pick : {"round_robin", "gap", "gap_age", "gap_utility"}) {
        for (const bool ff : {false, true}) {
            fam::Bank B = fresh_bank();
            bool all = true;
            for (long long k = 6; k <= 25; ++k) {
                const long long want = want_pick(B, pick, ff);
                fam::Options o = options(pick, k);
                o.fresh_first = ff;
                fam::run(family(true), B, o);
                all = all && B.pieces.size() == static_cast<std::size_t>(k) && B.pieces.back().n == want;
            }
            check(all, pick + (ff ? " + fresh_first" : "") + ": 20 pieces, each the one the rule asks for");
        }
    }
}

// ── the file, the old banks, the refusals ───────────────────────────────────

void file_and_going_on()
{
    const std::filesystem::path dir = temp_dir("file");
    fam::Options o = options("gap_utility", 20);
    o.fresh_first = true;
    o.utility_factor = 0.8;
    fam::Bank one = stuck_bank();
    fam::run(family(true), one, o);
    const std::string want = text_of(one);
    check(want.find("\"utility\": [[1, ") != std::string::npos,
          "file: the utilities in the state");
    fam::save_bank(utf8(dir / "one.json"), one, fam::total_of(one, {}), 5, "");
    const fam::Bank back = fam::load_bank(utf8(dir / "one.json"));
    check(back.utility == one.utility && text_of(back) == want, "file: read back, the utilities and the bank byte for byte");
    {
        fam::Bank A = stuck_bank();
        fam::Options oa = o;
        oa.max_pieces = 7;
        oa.bank_file = utf8(dir / "a.json");
        fam::run(family(true), A, oa);
        fam::Bank A2 = fam::load_bank(oa.bank_file);
        fam::run(family(true), A2, o);
        check(A.pieces.size() == 7 && text_of(A2) == want, "file: max_pieces 7, the file, going on = the campaign in one piece");
    }
    // the other picks: nothing new in the bank's text
    for (const std::string pick : {"round_robin", "gap", "gap_age"}) {
        fam::Bank B = stuck_bank();
        fam::run(family(true), B, options(pick, 8));
        const std::string t = text_of(B);
        check(B.utility.empty() && t.find("utility") == std::string::npos && t.find("fresh_first") == std::string::npos,
              pick + ": no utility, utility_factor or fresh_first in the bank's file");
        const fam::Bank b = fam::bank_of_text(t);
        check(b.utility.empty() && text_of(b) == t, pick + ": a bank without utilities read back (all of them 1)");
    }
    std::filesystem::remove_all(dir);
}

void refusals()
{
    auto refused = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    fam::Family plain = family(false);
    plain.reference = nullptr;
    check(refused([&] { fam::Bank b; fam::Campaign x(plain, b, options("gap_utility", 1)); }),
          "refused: gap_utility without a reference");
    for (const double u : {0.0, 1.0, -0.5, std::numeric_limits<double>::quiet_NaN()}) {
        fam::Options o = options("gap_utility", 1);
        o.utility_factor = u;
        check(refused([&] { fam::Bank b; fam::Campaign x(family(false), b, o); }),
              "refused: utility_factor " + std::to_string(u));
    }
    fam::Bank B = stuck_bank();
    std::string t = text_of(B);
    t.replace(t.find("\"state\": {"), 10, "\"state\": {\"utility\": [[1]], ");
    check(refused([&] { fam::bank_of_text(t); }), "refused: a utility not [n, pi]");
}

}   // namespace

int main()
{
    try {
        stuck_instance();
        fresh_first();
        picks_piece_by_piece();
        file_and_going_on();
        refusals();
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

// SPDX-License-Identifier: Apache-2.0
// budget_more (task 33.3; campaign/family.hpp FAM-14, campaign/components.hpp
// COMP-14) on synthetic families: instance n the problem n / 4 + sum (x_j -
// 0.5)^2 in dimension 20 (the family); twelve parts {1} ... {12}, part {k}
// the problem k / 4 + sum (x_j - 0.5)^2 in dimension 20, instance n
// assembled at no cost from parts {n} and {n mod 12 + 1} (the parts).
//   * two calls of budget_more = B from the file are one call of budget =
//     2B, bit for bit (the bank; the table, with and without its journal);
//     the file's options keep the call's budget and budget_more;
//   * a budget not above what the file had spent: no piece, stopped_by
//     budget, and the settings say what was spent; a budget above it says
//     nothing;
//   * off by default: a file without budget_more in its options, the
//     settings as before; the refusals.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
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
                                    ("mootation_test_budget_more_" + what + "_" + std::to_string(std::rand()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::string setting(const std::vector<std::pair<std::string, std::string>>& s, const std::string& key)
{
    for (const auto& kv : s)
        if (kv.first == key) return kv.second;
    return "";
}

bool has_setting(const std::vector<std::pair<std::string, std::string>>& s, const std::string& key)
{
    for (const auto& kv : s)
        if (kv.first == key) return true;
    return false;
}

const std::size_t kDim = 20;
const long long kPieceBudget = 60, kB = 600;   // ten pieces a stage

sc::Problem quadratic(double b)
{
    sc::Problem P;
    P.lo.assign(kDim, -2.0);
    P.hi.assign(kDim, 2.0);
    P.m = 1;
    P.evaluate = [b](const Vec& x, Vec& f, Vec&) {
        double s = b;
        for (double v : x) s += (v - 0.5) * (v - 0.5);
        f.assign(1, s);
    };
    return P;
}

// ── the family ──────────────────────────────────────────────────────────────

const long long kInstances = 6;

fam::Family family()
{
    fam::Family F;
    for (long long n = 1; n <= kInstances; ++n) F.instances.push_back(n);
    F.problem_at = [](long long n) { return quadratic(0.25 * static_cast<double>(n)); };
    F.weight = [](long long n) { return static_cast<double>(n); };
    return F;
}

fam::Options family_options(const std::string& file)
{
    fam::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.piece_budget = kPieceBudget;
    o.seed = 7;
    o.bank_file = file;
    return o;
}

void family_two_stages()
{
    const fam::Family F = family();
    const std::filesystem::path d = temp_dir("family");
    const std::string one = utf8(d / "one.json"), two = utf8(d / "two.json");

    fam::Options o = family_options(one);
    o.budget = 2 * kB;
    fam::Bank B0;
    const fam::Report r0 = fam::run(F, B0, o);
    const fam::Bank L0 = fam::load_bank(one);
    check(L0.options.find("budget_more") == nullptr && !has_setting(r0.settings, "family_budget_spent"),
          "family, budget = 2B: no budget_more in the file's options, nothing said of the budget");

    fam::Options m = family_options(two);
    m.budget_more = kB;
    fam::Bank B1;
    const fam::Report r1 = fam::run(F, B1, m);
    std::printf("      family: the first stage spent %s of %lld in %zu pieces\n", json::float_repr(B1.used).c_str(), kB,
                B1.pieces.size());
    check(B1.used == static_cast<double>(kB) && r1.stopped_by == "budget",
          "family: the first stage of budget_more = B ended at its budget exactly");
    fam::Bank B2 = fam::load_bank(two);
    const fam::Report r2 = fam::run(F, B2, m);
    const fam::Bank L2 = fam::load_bank(two);
    const json::Value* b = L2.options.find("budget");
    const json::Value* bm = L2.options.find("budget_more");
    check(b && b->i == 2 * kB && bm && bm->i == kB,
          "family: the second stage's file keeps the call's budget (spent + B) and budget_more");
    fam::Bank E = L2;
    E.options = L0.options;
    check(r2.stopped_by == r0.stopped_by && r2.total == r0.total &&
              fam::bank_text(E, fam::total_of(E, {}), F.instances.size(), r2.stopped_by) ==
                  fam::bank_text(L0, fam::total_of(L0, {}), F.instances.size(), r0.stopped_by),
          "family: two stages of budget_more = B are one of budget = 2B, bit for bit (the options aside)");
    check(r2.settings == r0.settings, "family: the settings of the two are those of the one");

    // a budget not above what the file had spent
    fam::Options s = family_options(two);
    s.budget = 2 * kB;
    fam::Bank B3 = fam::load_bank(two);
    const std::size_t before = B3.pieces.size();
    const fam::Report r3 = fam::run(F, B3, s);
    check(r3.stopped_by == "budget" && B3.pieces.size() == before &&
              setting(r3.settings, "family_budget_spent") == std::to_string(2 * kB),
          "family: budget = what the file spent: no piece, stopped_by budget, family_budget_spent said");
    s.budget = 2 * kB + kPieceBudget;
    fam::Bank B4 = fam::load_bank(two);
    const fam::Report r4 = fam::run(F, B4, s);
    check(B4.pieces.size() == before + 1 && !has_setting(r4.settings, "family_budget_spent"),
          "family: a budget above it gives its piece and says nothing");
    std::filesystem::remove_all(d);
}

void family_refusals()
{
    const fam::Family F = family();
    auto refused = [&](fam::Options o, const std::string& what) {
        fam::Bank B;
        bool thrown = false;
        try {
            fam::run(F, B, o);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        check(thrown, "family refuses " + what);
    };
    fam::Options o = family_options("");
    o.budget = kB;
    o.budget_more = kB;
    refused(o, "budget and budget_more both");
    o.budget = 0;
    o.budget_more = -1;
    refused(o, "budget_more < 0");
    o.budget_more = 0;
    refused(o, "neither budget nor budget_more");
}

// ── the parts ───────────────────────────────────────────────────────────────

const std::size_t kParts = 12;

Key other_of(long long n) { return Key{n % static_cast<long long>(kParts) + 1}; }

comp::Components parts()
{
    comp::Components C;
    for (long long k = 1; k <= static_cast<long long>(kParts); ++k) C.keys.push_back(Key{k});
    C.component_problem = [](const Key& k) { return quadratic(0.25 * static_cast<double>(k.at(0))); };
    C.assemble = [](long long n, const comp::View& v) {
        comp::Assembled a;
        a.cost = 0.0;
        if (!v.has(Key{n}) || !v.has(other_of(n))) return a;
        a.ok = true;
        a.f = v.quality(Key{n}) + v.quality(other_of(n));
        a.used_keys = {Key{n}, other_of(n)};
        return a;
    };
    return C;
}

fam::Family part_family()
{
    fam::Family F;
    for (long long n = 1; n <= static_cast<long long>(kParts); ++n) F.instances.push_back(n);
    F.weight = [](long long n) { return static_cast<double>(n); };
    return F;
}

comp::Options part_options(const std::string& pick, const std::string& file, bool journal)
{
    comp::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.piece_budget = kPieceBudget;
    o.pick = pick;
    o.epsilon = 1.0 / 1024.0;
    o.seed = 5;
    o.table_file = file;
    o.table_journal = journal;
    return o;
}

void parts_two_stages()
{
    const fam::Family F = part_family();
    const comp::Components C = parts();
    const std::filesystem::path d = temp_dir("parts");
    for (const std::string pick : {"round_robin", "sensitivity"}) {
        for (const bool journal : {false, true}) {
            const std::string what = pick + (journal ? ", with the journal" : "");
            const std::string one = utf8(d / (pick + (journal ? "_j" : "") + "_one.json"));
            const std::string two = utf8(d / (pick + (journal ? "_j" : "") + "_two.json"));

            comp::Options o = part_options(pick, one, journal);
            o.budget = 2 * kB;
            comp::Table T0;
            const comp::Report r0 = comp::run(F, C, T0, nullptr, o);
            const comp::Table L0 = comp::load_table(one);
            check(L0.options.find("budget_more") == nullptr && !has_setting(r0.settings, "components_budget_spent"),
                  "parts, " + what + ", budget = 2B: no budget_more in the options, nothing said");

            comp::Options m = part_options(pick, two, journal);
            m.budget_more = kB;
            comp::Table T1;
            const comp::Report r1 = comp::run(F, C, T1, nullptr, m);
            check(T1.used == static_cast<double>(kB) && r1.stopped_by == "budget",
                  "parts, " + what + ": the first stage ended at its budget exactly");
            comp::Table T2 = comp::load_table(two);
            const comp::Report r2 = comp::run(F, C, T2, nullptr, m);
            comp::Table L2 = comp::load_table(two);
            const json::Value* b = L2.options.find("budget");
            const json::Value* bm = L2.options.find("budget_more");
            check(b && b->i == 2 * kB && bm && bm->i == kB,
                  "parts, " + what + ": the second stage's file keeps the call's budget and budget_more");
            L2.options = L0.options;
            check(r2.stopped_by == r0.stopped_by && r2.settings == r0.settings &&
                      comp::table_text(L2, kParts, r2.stopped_by) == comp::table_text(L0, kParts, r0.stopped_by),
                  "parts, " + what + ": two stages of budget_more = B are one of budget = 2B, bit for bit");

            comp::Options s = part_options(pick, two, journal);
            s.budget = 2 * kB;
            comp::Table T3 = comp::load_table(two);
            const std::size_t before = T3.pieces.size();
            const comp::Report r3 = comp::run(F, C, T3, nullptr, s);
            check(r3.stopped_by == "budget" && T3.pieces.size() == before &&
                      setting(r3.settings, "components_budget_spent") == std::to_string(2 * kB),
                  "parts, " + what + ": budget = what the file spent: no piece, components_budget_spent said");
        }
    }
    std::filesystem::remove_all(d);
}

void parts_refusals()
{
    const fam::Family F = part_family();
    const comp::Components C = parts();
    auto refused = [&](comp::Options o, const std::string& what) {
        comp::Table T;
        bool thrown = false;
        try {
            comp::run(F, C, T, nullptr, o);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        check(thrown, "parts refuse " + what);
    };
    comp::Options o = part_options("round_robin", "", false);
    o.budget = kB;
    o.budget_more = kB;
    refused(o, "budget and budget_more both");
    o.budget = 0;
    o.budget_more = -1;
    refused(o, "budget_more < 0");
    o.budget_more = 0;
    refused(o, "neither budget nor budget_more");
}

}   // namespace

int main()
{
    std::srand(33003);
    try {
        family_two_stages();
        family_refusals();
        parts_two_stages();
        parts_refusals();
    } catch (const std::exception& e) {
        std::printf("FAIL  an exception: %s\n", e.what());
        ++g_failed;
    }
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}

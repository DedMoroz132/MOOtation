// SPDX-License-Identifier: Apache-2.0
// THE USER'S GENOME (task 24, §24.1.2; ga.hpp GA-9 ... GA-11) for ga_sbx and
// ssga:
//   * the user's crossover and mutation being SBX and the polynomial
//     mutation with the run's generator: the builtin run, bit for bit;
//   * a decoder that writes the genome back unchanged: the same run; one
//     that sorts it: only sorted genomes are evaluated;
//   * a genome that does not decode is not evaluated and costs nothing;
//   * ga_init = user: the first population is the sampler's;
//   * repair: builtin clips, user is the problem's, off leaves the row, and a
//     row outside the box, of the wrong length or not finite is refused with
//     its number; a knob without its hook is refused.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/operators/poly_mutation.hpp>
#include <mootation/operators/sbx.hpp>
#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
namespace ops = mootation::ops;
using Vec = std::vector<double>;
using Rows = std::vector<Vec>;
using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// a sum of shifted squares in [0, 1]^4 that writes down the points it is asked for
static sc::Problem squares(std::shared_ptr<Rows> asked)
{
    sc::Problem P;
    P.lo.assign(4, 0.0);
    P.hi.assign(4, 1.0);
    P.m = 1;
    P.evaluate = [asked](const Vec& x, Vec& f, Vec&) {
        if (asked) asked->push_back(x);
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - 0.2 * static_cast<double>(i + 1)) * (x[i] - 0.2 * static_cast<double>(i + 1));
        f[0] = s;
    };
    return P;
}

static bool same(const sc::Result& a, const sc::Result& b)
{
    if (a.improvements.size() != b.improvements.size()) return false;
    for (std::size_t i = 0; i < a.improvements.size(); ++i)
        if (std::memcmp(&a.improvements[i], &b.improvements[i], sizeof a.improvements[i]) != 0) return false;
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations;
}

static sc::Knob word(const std::string& w) { return {true, w, 0.0}; }

int main()
{
    const Bounds box(4, {0.0, 1.0});
    const sc::Knobs user_ops{{"ga_crossover", word("user")}, {"ga_mutation", word("user")}};
    for (const std::string m : {"ga_sbx", "ssga"}) {
        // the user's operators = the builtin's: bit for bit
        sc::Problem A = squares(nullptr), B = squares(nullptr);
        B.crossover = [box](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& rng) {
            ops::sbx(p1, p2, c1, c2, box, 2.0, 0.9, rng);
        };
        B.mutate = [box](Vec& x, std::mt19937& rng) { ops::polynomial_mutation(x, box, 20.0, 0.25, rng); };
        const sc::Result a = sc::run(m, A, 1e-6, false, 0.0, 20, 3000, 7, {}, {}, 0, {});
        const sc::Result b = sc::run(m, B, 1e-6, false, 0.0, 20, 3000, 7, user_ops, {}, 0, {});
        check(same(a, b), m + ": the user's SBX and polynomial mutation are the builtin run, bit for bit");
        // a decoder writing the genome back unchanged: the same run
        sc::Problem C = squares(nullptr);
        C.decode = [](const Vec& x) { return sc::Decoded{x, {}, true}; };
        const sc::Result c = sc::run(m, C, 1e-6, false, 0.0, 20, 3000, 7, {}, {}, 0, {});
        check(same(a, c) && c.decode_invalid == 0, m + ": an unchanged write-back is the builtin run");
        // a decoder that sorts: only sorted genomes are evaluated
        auto asked = std::make_shared<Rows>();
        sc::Problem D = squares(asked);
        D.decode = [](const Vec& x) {
            Vec y = x;
            std::sort(y.begin(), y.end());
            return sc::Decoded{y, {}, true};
        };
        const sc::Result d = sc::run(m, D, 1e-6, false, 0.0, 20, 3000, 7, {}, {}, 0, {});
        bool sorted = !asked->empty();
        for (const Vec& x : *asked) sorted = sorted && std::is_sorted(x.begin(), x.end());
        check(sorted && std::is_sorted(d.variables[0].begin(), d.variables[0].end()),
              m + ": the canonical genome written back, the only one evaluated");
        // genomes that do not decode: not evaluated, no budget
        auto seen = std::make_shared<Rows>();
        sc::Problem E = squares(seen);
        E.decode = [](const Vec& x) { return sc::Decoded{x, {}, x[0] <= 0.5}; };
        const sc::Result e = sc::run(m, E, 1e-6, false, 0.0, 20, 3000, 7, {}, {}, 0, {});
        bool none = true;
        for (const Vec& x : *seen) none = none && x[0] <= 0.5;
        check(none && e.decode_invalid > 0 && e.evaluations == static_cast<long long>(seen->size()),
              m + ": an invalid genome is not evaluated and costs nothing");
        std::printf("  %-6s builtin f %.3g; sorted %.3g; %lld genomes did not decode (%lld evaluations)\n", m.c_str(),
                    a.objectives[0][0], d.objectives[0][0], e.decode_invalid, e.evaluations);
        // ga_init = user: the first population is the sampler's
        auto first = std::make_shared<Rows>();
        sc::Problem S = squares(first);
        S.sample = [](Vec& x, std::mt19937& rng) {
            for (double& v : x) v = 0.25 + 0.01 * mootation::rnd::uniform01(rng);
        };
        sc::run(m, S, 1e-6, false, 0.0, 20, 300, 7, {{"ga_init", word("user")}}, {}, 0, {});
        bool sampled = first->size() >= 20;
        for (std::size_t i = 0; sampled && i < 20; ++i)
            for (double v : (*first)[i]) sampled = sampled && v >= 0.25 && v < 0.26;
        check(sampled, m + ": ga_init = user, the first population from sample");
    }

    // repair: a crossover that leaves the box
    {
        sc::Problem P = squares(nullptr);
        P.crossover = [](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937&) {
            c1 = p1;
            c2 = p2;
            c1[1] = 1.5;                                 // outside [0, 1]
        };
        const sc::Result clipped =
            sc::run("ga_sbx", P, 1e-6, false, 0.0, 20, 2000, 7, {{"ga_crossover", word("user")}}, {}, 0, {});
        check(!clipped.objectives.empty(), "ga_repair = builtin: the row clipped to the box");
        P.repair = [](Vec& x) { x[1] = 0.4; };
        const sc::Result repaired = sc::run("ga_sbx", P, 1e-6, false, 0.0, 20, 2000, 7,
                                            {{"ga_crossover", word("user")}, {"ga_repair", word("user")}}, {}, 0, {});
        check(!repaired.objectives.empty(), "ga_repair = user: the problem's repair");
        std::string why;
        try {
            sc::run("ga_sbx", P, 1e-6, false, 0.0, 20, 2000, 7, {{"ga_crossover", word("user")}, {"ga_repair", word("off")}},
                    {}, 0, {});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find("individual 0 of generation 1: outside the box (variable 1)") != std::string::npos,
              "ga_repair = off: a row outside the box refused: '" + why + "'");
    }
    // a wrong length, a value not finite, a knob without its hook
    auto refused = [](sc::Problem P, const sc::Knobs& k, const std::string& part) {
        std::string why;
        try {
            sc::run("ssga", P, 1e-6, false, 0.0, 20, 500, 7, k, {}, 0, {});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find(part) != std::string::npos, "refused: '" + why + "' (wanted: " + part + ")");
    };
    {
        sc::Problem P = squares(nullptr);
        P.mutate = [](Vec& x, std::mt19937&) { x.push_back(0.0); };
        refused(P, {{"ga_mutation", word("user")}}, "5 values for 4 variables");
        P.mutate = [](Vec& x, std::mt19937&) { x[2] = std::nan(""); };
        refused(P, {{"ga_mutation", word("user")}, {"ga_repair", word("off")}}, "a value not finite (variable 2)");
        refused(squares(nullptr), {{"ga_crossover", word("user")}}, "the problem gives no crossover");
        refused(squares(nullptr), {{"ga_init", word("user")}}, "the problem gives no sample");
        sc::Problem Q = squares(nullptr);
        Q.decode = [](const Vec&) { return sc::Decoded{{0.5}, {}, true}; };
        refused(Q, {}, "the decoder's canonical genome, individual 0 of generation 0: 1 values for 4 variables");
    }

    if (failures) {
        std::printf("test_genome_ops: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_genome_ops: all checks passed\n");
    return 0;
}

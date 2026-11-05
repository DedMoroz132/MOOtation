// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/pap.hpp (task 19, §19.6):
//   - evo::replace_worst: a better migrant takes the worst's place (the last
//     of equals), a worse one stays out; DE's and PSO's migrate (PSO: the
//     particle's best and position, its velocity 0, gbest again);
//   - one member, no migration, is that method's own run, bit for bit: de,
//     pso, ga_sbx (PAP-6);
//   - the default portfolio (de 70, pso 16, cmaes 14) on Rastrigin n = 5:
//     MAX_GEN / 20 between migrations, the migrations made, every
//     evaluation counted once (the migrants cost nothing), the budget kept;
//     the same seed, the same run;
//   - refusals: an unknown member, a member without np, a bound infinite.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& [k, v] : r.settings)
        if (k == key) return v;
    return "";
}

sc::Problem rastrigin(int n, std::shared_ptr<long long> calls)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -5.12);
    P.hi.assign(static_cast<std::size_t>(n), 5.12);
    P.m = 1;
    P.evaluate = [calls](const Vec& x, Vec& f, Vec&) {
        ++*calls;
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
        f.assign(1, s);
    };
    return P;
}

void migrants()
{
    namespace evo = sc::evo;
    sc::Rows X{{0.0}, {1.0}, {2.0}};
    Vec F{3.0, 5.0, 5.0}, V{0.0, 0.0, 0.0};
    const std::size_t a = evo::replace_worst(X, F, V, {9.0}, 4.0, 0.0);
    const std::size_t b = evo::replace_worst(X, F, V, {8.0}, 6.0, 0.0);
    check(a == 2 && X[2] == Vec{9.0} && F[2] == 4.0 && b == 3 && X[1] == Vec{1.0},
          "replace_worst: a better migrant takes the last worst's place, a worse one stays out (PAP-2)");
    std::mt19937 rng(3);
    sc::de::Options o;
    o.np = 6;
    sc::de::DE de(Vec(2, -1.0), Vec(2, 1.0), o, rng);
    const sc::Rows X0 = de.ask();
    Vec f(6), v(6, 0.0);
    for (std::size_t i = 0; i < 6; ++i) f[i] = X0[i][0] * X0[i][0] + X0[i][1] * X0[i][1];
    de.tell(f, v);
    de.migrate({0.0, 0.0}, 0.0, 0.0);
    check(de.population()[de.best()] == Vec{0.0, 0.0} && de.values()[de.best()] == 0.0,
          "de.migrate: the migrant in the population, its best");
    sc::pso::Options po;
    po.n = 5;
    sc::pso::PSO pso(Vec(2, -1.0), Vec(2, 1.0), po, rng);
    const sc::Rows P0 = pso.ask();
    Vec fp(5);
    for (std::size_t i = 0; i < 5; ++i) fp[i] = P0[i][0] * P0[i][0] + P0[i][1] * P0[i][1];
    pso.tell(fp, Vec(5, 0.0));
    pso.migrate({0.0, 0.0}, 0.0, 0.0);
    const std::size_t g = pso.best();
    check(pso.population()[g] == Vec{0.0, 0.0} && pso.positions()[g] == Vec{0.0, 0.0},
          "pso.migrate: the worst particle's best and position, gbest the migrant (PAP-3)");
}

void one_member_is_the_method()
{
    for (const auto& [name, np] : {std::pair<std::string, int>{"de", 30}, {"pso", 20}, {"ga_sbx", 40}}) {
        auto c1 = std::make_shared<long long>(0), c2 = std::make_shared<long long>(0);
        const std::string members = "[{\"np\":" + std::to_string(np) + ",\"solver\":\"" + name + "\"}]";
        const sc::Result p = sc::run("pap", rastrigin(4, c1), 1e-6, false, 0.0, 10, 3000, 9,
                                     {{"pap_members", word(members)}}, {}, 0, {});
        const sc::Result d = sc::run(name, rastrigin(4, c2), 1e-6, false, 0.0, 10, 3000, 9,
                                     {{"np", number(np)}}, {}, 0, {});
        check(p.variables == d.variables && p.objectives == d.objectives && p.evaluations == d.evaluations &&
                  *c1 == *c2 && setting(p, "migrations") == "0",
              "one member " + name + " (np " + std::to_string(np) + "): " + name + "'s own run, bit for bit (" +
                  std::to_string(p.evaluations) + " evaluations)");
    }
}

void the_portfolio()
{
    auto c1 = std::make_shared<long long>(0), c2 = std::make_shared<long long>(0);
    const sc::Result a = sc::run("pap", rastrigin(5, c1), 1e-6, false, 0.0, 10, 20000, 4, {}, {}, 0, {});
    const sc::Result b = sc::run("pap", rastrigin(5, c2), 1e-6, false, 0.0, 10, 20000, 4, {}, {}, 0, {});
    const long long gens = std::stoll(setting(a, "generations")), migr = std::stoll(setting(a, "migrations"));
    check(setting(a, "migration_interval") == "10" && migr == (gens - 1) / 10 && a.evaluations <= 20000 &&
              *c1 == a.evaluations && std::isfinite(a.objectives[0][0]),
          "de 70 + pso 16 + cmaes 14, Rastrigin n = 5, 20000: every 10 generations (MAX_GEN 200 / 20), " +
              std::to_string(migr) + " migrations in " + std::to_string(gens) + " generations, f = " +
              std::to_string(a.objectives[0][0]) + ", each evaluation counted once");
    check(a.variables == b.variables && a.evaluations == b.evaluations && *c1 == *c2, "the same seed, the same run");
    auto refused = [](const sc::Problem& P, const sc::Knobs& k) {
        try {
            sc::run("pap", P, 1e-6, false, 0.0, 10, 100, 1, k, {}, 0, {});
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    auto c = std::make_shared<long long>(0);
    sc::Problem open = rastrigin(2, c);
    open.hi[0] = std::numeric_limits<double>::infinity();
    check(refused(rastrigin(2, c), {{"pap_members", word("[{\"np\":10,\"solver\":\"nsga2\"}]")}}) &&
              refused(rastrigin(2, c), {{"pap_members", word("[{\"solver\":\"de\"}]")}}) &&
              refused(rastrigin(2, c), {{"pap_members", word("[]")}}) && refused(open, {}),
          "refused: an unknown member, a member without np, no members, a bound infinite");
}

}   // namespace

int main()
{
    migrants();
    one_member_is_the_method();
    the_portfolio();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

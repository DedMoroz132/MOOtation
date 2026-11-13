// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/catcma.hpp (Hamano et al., GECCO 2024):
//   - Table 1 at N_co = N_ca = 5: lambda and the margin q^min = (1 -
//     0.73^(1/N_ca)) / (K - 1) (CC-2); q^(0) = 1/K, delta^(0) = 1;
//   - Proposition 4.1: with every q_{n,1} at 1 - (K_n - 1) q^min the joint
//     probability of the optimal categories is 0.73, and the number of
//     non-optimal samples stays at most lambda - floor(lambda / 2) with
//     probability >= 0.95 for lambda = 6 .. 200 (the binomial with 0.27);
//   - the margin (28)-(29) after every update: each q_n sums to 1 and no
//     q_{n,k} falls below q^min;
//   - the paper's experiment, short (§5, Table 2, m^(0) ~ U[-3, 3], sigma^(0)
//     = 1, N_co = N_ca = K = 10, 11 runs of 10^4 evaluations): SPHERECOM and
//     MCPROXIMITY reach 1e-10 in the median; the margin of §5.2 does better
//     than the large one 1 / (N_ca (K - 1)) (Figure 4's direction); CatCMA
//     does better than its ablation without the enhancements (§5.3, Figure
//     6's direction);
//   - update = printed ((24)-(25) as the PDF prints them, CC-11) stalls
//     there: SPHERECOM's median stays above 0.1 where the default reaches
//     1e-10;
//   - N_ca = 0 is the plain CMA-ES and solves the sphere; bad arguments are
//     refused (CC-9);
//   - a checksum of the runs, the same bits on every platform.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/catcma.hpp>

namespace {

namespace C = mootation::scalar::catcma;
namespace rnd = mootation::rnd;
using C::Vec;
using Fn = std::function<double(const C::Point&)>;

int g_failed = 0;
std::uint64_t g_hash = 1469598103934665603ull;     // FNV-1a over the runs' bits

void mix(double v)
{
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof b);
    g_hash = (g_hash ^ b) * 1099511628211ull;
}

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

// Table 2, the optimal category the first (index 0)
double sphere_com(const C::Point& p)
{
    double s = 0.0;
    for (double v : p.x) s += v * v;
    for (int c : p.c) s += c == 0 ? 0.0 : 1.0;
    return s;
}

double mc_proximity(const C::Point& p, int K)
{
    double s = 0.0;
    for (std::size_t n = 0; n < p.x.size(); ++n) {
        const double z = static_cast<double>(p.c[n]) / K;
        s += (p.x[n] - z) * (p.x[n] - z) + z;
    }
    return s;
}

// the best value after `budget` evaluations; `each` sees the solver after every tell
double run(const Fn& f, int n, int K, long long budget, unsigned seed, const C::Options& o,
           const std::function<void(const C::CatCMA&)>& each = nullptr)
{
    std::mt19937 rng(seed);
    Vec m(static_cast<std::size_t>(n));
    for (double& v : m) v = -3.0 + 6.0 * rnd::uniform01(rng);
    C::CatCMA es(m, 1.0, std::vector<int>(static_cast<std::size_t>(n), K), rng, o);
    double best = 1e300;
    for (long long evals = 0; evals < budget;) {
        const auto P = es.ask();
        Vec v(P.size());
        for (std::size_t i = 0; i < P.size(); ++i) {
            v[i] = f(P[i]);
            if (evals < budget) best = std::min(best, v[i]);
            ++evals;
        }
        es.tell(v);
        if (each) each(es);
    }
    mix(best);
    return best;
}

double median(const Fn& f, int n, int K, long long budget, const C::Options& o)
{
    std::vector<double> b;
    for (unsigned s = 1; s <= 11; ++s) b.push_back(run(f, n, K, budget, s, o));
    std::sort(b.begin(), b.end());
    return b[5];
}

void table1()
{
    std::mt19937 rng(1);
    const C::CatCMA es(Vec(5, 0.0), 1.0, std::vector<int>(5, 4), rng);
    check(es.lambda() == 10, "lambda = 4 + floor(3 ln (N_co + N_ca)) = 10");
    const double want = (1.0 - std::pow(0.73, 1.0 / 5.0)) / 3.0;
    check(std::fabs(es.margin(0) / want - 1.0) < 1e-14 && std::fabs(C::default_margin(5, 4) / want - 1.0) < 1e-14,
          "q^min = (1 - 0.73^(1/N_ca)) / (K - 1) (CC-2)");
    bool uniform = true;
    for (const Vec& q : es.q())
        for (double t : q) uniform = uniform && t == 0.25;
    check(uniform && es.delta() == 1.0, "q^(0) = 1/K, delta^(0) = 1");
}

void proposition()
{
    for (int Nca : {1, 3, 10, 40})
        for (int K : {2, 5, 10}) {
            const double qmin = C::default_margin(static_cast<std::size_t>(Nca), K);
            const double all = std::pow(1.0 - (K - 1) * qmin, Nca);
            if (std::fabs(all - 0.73) > 1e-12) {
                check(false, "prod q_{n,1} at the margin = 0.73: N_ca " + std::to_string(Nca) + ", K " + std::to_string(K));
                return;
            }
        }
    check(true, "prod q_{n,1} at the margin = 0.73 (N_ca = 1, 3, 10, 40; K = 2, 5, 10)");
    double worst = 1.0;
    for (int lam = 6; lam <= 200; ++lam) {
        const int k = lam - lam / 2;
        double term = std::pow(0.73, lam), cdf = 0.0;      // P(X = 0), X ~ B(lam, 0.27)
        for (int i = 0; i <= k; ++i) {
            cdf += term;
            term *= static_cast<double>(lam - i) / (i + 1.0) * (0.27 / 0.73);
        }
        worst = std::min(worst, cdf);
    }
    char line[160];
    std::snprintf(line, sizeof line, "Pr(lambda_non <= lambda - floor(lambda/2)) >= 0.95, lambda = 6..200 (least %.4f)", worst);
    check(worst >= 0.95, line);
}

void margins()
{
    const int n = 5, K = 5;
    C::Options o;
    double low = 1.0, sum_err = 0.0;
    double qmin = 0.0;
    run(sphere_com, n, K, 3000, 7, o, [&](const C::CatCMA& es) {
        qmin = es.margin(0);
        for (const Vec& q : es.q()) {
            double s = 0.0;
            for (double t : q) {
                s += t;
                low = std::min(low, t);
            }
            sum_err = std::max(sum_err, std::fabs(s - 1.0));
        }
    });
    char line[200];
    std::snprintf(line, sizeof line, "after every update: |sum q_n - 1| <= %.1e, min q_{n,k} / q^min = %.6f (28)-(29)", sum_err, low / qmin);
    check(sum_err < 1e-12 && low >= qmin * (1.0 - 1e-12), line);
}

void experiment()
{
    const int N = 10, K = 10;
    const long long budget = 10000;
    C::Options rec, large, abl;
    large.q_min = 1.0 / (N * (K - 1.0));
    abl.enhanced = false;
    const Fn mcp = [](const C::Point& p) { return mc_proximity(p, 10); };
    struct Row {
        const char* name;
        Fn f;
    };
    for (const Row& r : {Row{"SPHERECOM", sphere_com}, Row{"MCPROXIMITY", mcp}}) {
        const double a = median(r.f, N, K, budget, rec);
        const double b = median(r.f, N, K, budget, large);
        const double c = median(r.f, N, K, budget, abl);
        char line[240];
        std::snprintf(line, sizeof line, "%s, N = K = 10, 1e4 evaluations: median %.2e; large margin %.2e; ablation %.2e",
                      r.name, a, b, c);
        check(a < 1e-10 && b > 1e3 * a && c > 1e3 * a, line);
    }
}

void printed()
{
    C::Options lit;
    lit.update = C::Update::printed;
    const double a = median(sphere_com, 10, 10, 10000, lit);
    char line[200];
    std::snprintf(line, sizeof line, "update = printed, SPHERECOM N = K = 10, 1e4 evaluations: median %.2e, stalled (CC-11)", a);
    check(a > 0.1, line);
}

void plain()
{
    std::mt19937 rng(3);
    C::CatCMA es(Vec(4, 2.0), 0.5, {}, rng);
    double best = 1e300;
    for (int g = 0; g < 300; ++g) {
        const auto P = es.ask();
        Vec v(P.size());
        for (std::size_t i = 0; i < P.size(); ++i) {
            v[i] = sphere_com(P[i]);
            best = std::min(best, v[i]);
        }
        es.tell(v);
    }
    mix(best);
    check(es.lambda() == 8 && best < 1e-20, "N_ca = 0: lambda = 4 + floor(3 ln 4) = 8, the sphere below 1e-20 in 300 generations");
}

void refusals()
{
    auto refused = [](const std::function<void()>& g) {
        try {
            g();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    std::mt19937 rng(1);
    C::Options big;
    big.q_min = 0.6;
    check(refused([&] { C::CatCMA es(Vec{}, 1.0, {3}, rng); }) &&
              refused([&] { C::CatCMA es(Vec(2, 0.0), 1.0, {1}, rng); }) &&
              refused([&] { C::CatCMA es(Vec(2, 0.0), 0.0, {3}, rng); }) &&
              refused([&] { C::CatCMA es(Vec(2, 0.0), 1.0, {2}, rng, big); }) &&
              refused([&] {
                  C::CatCMA es(Vec(2, 0.0), 1.0, {3}, rng);
                  es.ask();
                  es.tell(Vec(1, 0.0));
              }),
          "refused: N_co = 0, K_n < 2, sigma <= 0, q^min K > 1, a tell of the wrong length (CC-9)");
}

}   // namespace

int main()
{
    table1();
    proposition();
    margins();
    experiment();
    printed();
    plain();
    refusals();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0x54fc979e74626851ull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}

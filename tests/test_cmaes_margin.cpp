// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/cmaes.hpp, task 25's options:
//   eigen_every (CMA-5, CMA-7): 1 = the old run bit for bit; k = 5 decomposes
//     every fifth tell and still solves an ellipsoid; 0 gives H09's interval
//     max(1, floor(1/((c_1 + c_mu) 10 D))) (1 at D = 20, 2 at D = 300);
//   CMA-ES with margin (HM22): margin_values with margin_alpha = 0 is the old
//     run bit for bit; with alpha = 1/(N lambda) (HM22 §5.1) SphereInt and
//     SphereOneMax at N = 20 (10 continuous + 10 integer in [-10, 10] or
//     binary) reach f < 1e-10 in every one of 20 runs, the evaluations of the
//     order of HM22 Table 2 (3840 and 3876 median);
//   encode: the midpoint thresholds; the options are checked.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/cmaes.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

using mootation::scalar::CMAES;
using mootation::scalar::CMAESOptions;

double ellipsoid(const std::vector<double>& x)
{
    double f = 0.0;
    const double n1 = static_cast<double>(x.size() - 1);
    for (std::size_t i = 0; i < x.size(); ++i) f += std::pow(1e3, static_cast<double>(i) / n1) * x[i] * x[i];
    return f;
}

// the means after `iters` iterations and the points of the last ask
std::vector<double> trace(const CMAESOptions& o, int iters, std::vector<std::vector<double>>* last = nullptr)
{
    std::mt19937 r(3);
    CMAES es(std::vector<double>(10, 1.0), 0.5, r, 0, o);
    std::vector<double> means;
    for (int t = 0; t < iters; ++t) {
        const auto X = es.ask();
        std::vector<double> f;
        for (const auto& x : X) f.push_back(ellipsoid(x));
        es.tell(f);
        means.insert(means.end(), es.mean().begin(), es.mean().end());
        if (last) *last = X;
    }
    return means;
}

void defaults_bit_for_bit()
{
    std::vector<std::vector<double>> a, b, c;
    const auto base = trace(CMAESOptions(), 200, &a);
    CMAESOptions o1;
    o1.eigen_every = 1;
    const auto one = trace(o1, 200, &b);
    CMAESOptions o2;
    o2.margin_alpha = 0.0;
    o2.margin_values.assign(10, {});
    for (int j = 5; j < 10; ++j) o2.margin_values[j] = {-1.0, 0.0, 1.0, 2.0};
    const auto off = trace(o2, 200, &c);
    check(one == base && b == a, "eigen_every = 1: the default run bit for bit (200 iterations)");
    check(off == base && c == a, "margin_values with margin_alpha = 0: the default run bit for bit");
}

void lazy()
{
    std::mt19937 r(5);
    CMAESOptions o;
    o.eigen_every = 5;
    CMAES es(std::vector<double>(20, 1.0), 0.5, r, 0, o);
    double best = 1e300;
    long long it = 0;
    for (; it < 20000 && best > 1e-10; ++it) {
        const auto X = es.ask();
        std::vector<double> f;
        for (const auto& x : X) {
            f.push_back(ellipsoid(x));
            best = std::min(best, f.back());
        }
        es.tell(f);
    }
    std::printf("      eigen_every = 5, ellipsoid D = 20: f = %.2g after %lld iterations, %lld decompositions\n", best, it,
                es.decompositions());
    check(best <= 1e-10 && es.decompositions() == it / 5, "eigen_every = 5: the ellipsoid solved, a decomposition every fifth tell");
    std::mt19937 r2(1);
    CMAESOptions a;
    a.eigen_every = 0;
    const CMAES e20(std::vector<double>(20, 0.0), 1.0, r2, 0, a), e300(std::vector<double>(300, 0.0), 1.0, r2, 0, a);
    const auto h09 = [](const CMAES& e, double D) {
        return std::max(1, static_cast<int>(std::floor(1.0 / (((e.c_mu() + 2.0 / ((D + 1.3) * (D + 1.3) + e.mu_eff())) * 10.0 * D)))));
    };
    std::printf("      eigen_every = 0: interval %d at D = 20, %d at D = 300\n", e20.eigen_every(), e300.eigen_every());
    check(e20.eigen_every() == h09(e20, 20.0) && e300.eigen_every() == h09(e300, 300.0) && e300.eigen_every() > 1,
          "eigen_every = 0: H09's interval max(1, floor(1/((c_1 + c_mu) 10 D)))");
}

// HM22 §5: SphereInt / SphereOneMax on the encoded point; N = 20, half discrete
struct Run {
    bool success;
    long long evals;
};

Run mixed(bool binary, double alpha, unsigned seed)
{
    const std::size_t N = 20, Nco = 10;
    std::mt19937 init(seed + 1000);
    std::vector<double> m(N);
    for (std::size_t j = 0; j < N; ++j)
        m[j] = (binary && j >= Nco) ? 0.0 : 1.0 + 2.0 * mootation::rnd::uniform01(init);   // HM22 §3
    CMAESOptions o;
    o.margin_values.assign(N, {});
    for (std::size_t j = Nco; j < N; ++j) {
        if (binary) o.margin_values[j] = {0.0, 1.0};
        else
            for (int v = -10; v <= 10; ++v) o.margin_values[j].push_back(v);
    }
    std::mt19937 r(seed);
    CMAES probe(m, 1.0, r, 0, CMAESOptions());
    o.margin_alpha = alpha > 0.0 ? alpha : 0.0;
    if (alpha < 0.0) o.margin_alpha = 1.0 / (static_cast<double>(N) * static_cast<double>(probe.lambda()));
    CMAES es(m, 1.0, r, 0, o);
    long long evals = 0;
    for (int it = 0; it < 20000; ++it) {
        const auto V = es.ask();
        std::vector<double> f;
        for (const auto& v : V) {
            const auto x = es.encode(v);
            double s = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                if (binary && j >= Nco) s += 1.0 - x[j];           // OneMax: N_bi - sum x
                else s += x[j] * x[j];
            }
            f.push_back(s);
            ++evals;
            if (s < 1e-10) return {true, evals};
        }
        es.tell(f);
        if (es.stop() != CMAES::Stop::none && es.stop() != CMAES::Stop::stagnation) break;
    }
    return {false, evals};
}

void margin()
{
    for (bool binary : {false, true}) {
        int ok = 0, ok_plain = 0;
        std::vector<long long> evals;
        for (unsigned s = 1; s <= 20; ++s) {
            const Run a = mixed(binary, -1.0, s);   // alpha = 1/(N lambda)
            if (a.success) {
                ++ok;
                evals.push_back(a.evals);
            }
            ok_plain += mixed(binary, 0.0, s).success ? 1 : 0;
        }
        std::sort(evals.begin(), evals.end());
        const long long med = evals.empty() ? -1 : evals[evals.size() / 2];
        std::printf("      %s N = 20: margin %d/20 (median %lld evaluations; HM22 Table 2: %s), without margin %d/20\n",
                    binary ? "SphereOneMax" : "SphereInt", ok, med, binary ? "3876" : "3840", ok_plain);
        check(ok == 20 && med > 0 && med < 3 * (binary ? 3876 : 3840),
              std::string(binary ? "SphereOneMax" : "SphereInt") + ", N = 20: the margin solves 20 of 20, evaluations of HM22's order");
    }
}

void encode_and_checks()
{
    std::mt19937 r(1);
    CMAESOptions o;
    o.margin_alpha = 0.01;
    o.margin_values = {{}, {0.0, 1.0}, {1.0, 2.0, 4.0}};
    CMAES es(std::vector<double>(3, 0.0), 1.0, r, 0, o);
    check(es.encode({0.3, 0.5, 3.0}) == std::vector<double>{0.3, 0.0, 2.0} &&
              es.encode({0.3, 0.51, 3.01}) == std::vector<double>{0.3, 1.0, 4.0},
          "encode: x <= threshold takes the lower value (HM22 §4.1)");
    bool threw = false;
    try {
        CMAESOptions bad;
        bad.margin_values = {{1.0}};
        CMAES e(std::vector<double>(1, 0.0), 1.0, r, 0, bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a discrete coordinate with one value is refused");
}

}   // namespace

int main()
{
    defaults_bit_for_bit();
    lazy();
    margin();
    encode_and_checks();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}

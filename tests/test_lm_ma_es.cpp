// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/lm_ma_es.hpp (task 25, item 24.3.1):
//   the sphere at n = 128 from a point of [-5, 5]^n with sigma = 3 (L18
//     §4.1's setup) reaches f_target = 1e-10;
//   a rotated ellipsoid (condition 1e4, n = 64) reaches 1e-10: the m vectors
//     learn the metric;
//   one seed, one run (bit for bit); the budget is never exceeded;
//   small n (n = 5, 10): the readings LMMA-1 keep the run finite.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/lm_ma_es.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace lm = mootation::lm_ma_es;

double sphere(const std::vector<double>& x)
{
    double s = 0.0;
    for (double v : x) s += v * v;
    return s;
}

// An ellipsoid of condition 1e4 rotated by the pairwise rotations of 30
// degrees of coordinates (2k, 2k+1) and then (2k+1, 2k+2)
double ellipsoid(const std::vector<double>& x)
{
    std::vector<double> y = x;
    const double c = std::cos(0.5235987755982988), s = std::sin(0.5235987755982988);
    for (std::size_t k = 0; k + 1 < y.size(); k += 2) {
        const double a = y[k], b = y[k + 1];
        y[k] = c * a - s * b;
        y[k + 1] = s * a + c * b;
    }
    for (std::size_t k = 1; k + 1 < y.size(); k += 2) {
        const double a = y[k], b = y[k + 1];
        y[k] = c * a - s * b;
        y[k + 1] = s * a + c * b;
    }
    double f = 0.0;
    const double n1 = static_cast<double>(y.size() - 1);
    for (std::size_t k = 0; k < y.size(); ++k) f += std::pow(1e4, static_cast<double>(k) / n1) * y[k] * y[k];
    return f;
}

std::vector<double> start(std::size_t n, unsigned seed)
{
    std::mt19937 g(seed);
    std::vector<double> y(n);
    for (double& v : y) v = -5.0 + 10.0 * mootation::rnd::uniform01(g);
    return y;
}

void sphere_128()
{
    lm::Options o;
    o.sigma0 = 3.0;
    o.f_target = 1e-10;
    o.max_evals = 2000000;
    std::mt19937 r(1);
    const lm::Result R = lm::minimize(sphere, start(128, 7), r, o);
    std::printf("      sphere n = 128: f = %.3g after %lld evaluations (%lld iterations)\n", R.f, R.evaluations,
                R.iterations);
    check(R.stop == lm::Stop::target && R.f <= 1e-10 && R.evaluations <= o.max_evals, "sphere, n = 128: f <= 1e-10");

    std::mt19937 r2(1);
    const lm::Result R2 = lm::minimize(sphere, start(128, 7), r2, o);
    check(R2.x == R.x && R2.f == R.f && R2.evaluations == R.evaluations && R2.mean == R.mean, "one seed, one run");
}

void ellipsoid_64()
{
    lm::Options o;
    o.sigma0 = 3.0;
    o.f_target = 1e-10;
    o.max_evals = 3000000;
    std::mt19937 r(2);
    const lm::Result R = lm::minimize(ellipsoid, start(64, 9), r, o);
    std::printf("      rotated ellipsoid n = 64, condition 1e4: f = %.3g after %lld evaluations\n", R.f, R.evaluations);
    check(R.stop == lm::Stop::target, "rotated ellipsoid, n = 64: f <= 1e-10");
}

void small_and_budget()
{
    bool ok = true;
    for (std::size_t n : {2u, 5u, 10u}) {
        lm::Options o;
        o.f_target = 1e-10;
        o.max_evals = 200000;
        std::mt19937 r(3);
        const lm::Result R = lm::minimize(sphere, std::vector<double>(n, 1.0), r, o);
        std::printf("      sphere n = %zu: f = %.3g after %lld evaluations\n", n, R.f, R.evaluations);
        ok = ok && R.stop == lm::Stop::target && std::isfinite(R.sigma);
    }
    check(ok, "n = 2, 5, 10 (2 lambda >= n: LMMA-1): the sphere is solved");
    lm::Options o;
    o.max_evals = 1000;
    std::mt19937 r(4);
    const lm::Result R = lm::minimize(sphere, start(30, 1), r, o);
    check(R.stop == lm::Stop::budget && R.evaluations <= 1000 && R.evaluations > 1000 - 14,
          "a budget of 1000: whole iterations up to it, never past it");
}

}   // namespace

int main()
{
    sphere_128();
    ellipsoid_64();
    small_and_budget();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}

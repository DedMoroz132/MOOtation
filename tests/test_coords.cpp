// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// coords — the variation in learned coordinates (operators/learned_coords.hpp,
// variation.hpp; task 7 of 2026-10-10, supplement 1): the basis is complete
// and orthonormal, randrot's is a fresh rotation reproducible by its seed, the
// offspring distribution of pca turns with the population, children come back
// into the box, and a warm-up that never ends is coords = axis, bit for bit,
// in NSGA-II, SMS-EMOA and mosa's sbx step.
// ============================================================================

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "harness.hpp"

MOOTATION_TEST_PROBLEM(CoordsZDT1, NSGAII_Individual, ZDT1Spec)

namespace {

using namespace mootation;
using namespace mootation::testing;
using Vec = std::vector<double>;
using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;

struct FakeVault {
    std::vector<Vec> X, F;
    Bounds box;
    const Vec& variables_of(std::size_t i) const { return X[i]; }
    const Vec& objectives_of(std::size_t i) const { return F[i]; }
    const Bounds& get_bounds() const { return box; }
};

double orthonormality_error(const std::vector<Vec>& R)
{
    double worst = 0.0;
    for (std::size_t a = 0; a < R.size(); ++a)
        for (std::size_t b = 0; b < R.size(); ++b) {
            double d = 0.0;
            for (std::size_t i = 0; i < R.size(); ++i) d += R[i][a] * R[i][b];
            worst = std::max(worst, std::abs(d - (a == b ? 1.0 : 0.0)));
        }
    return worst;
}

FakeVault population(std::size_t m, std::size_t n, double lo, double hi, unsigned seed)
{
    FakeVault v;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (std::size_t i = 0; i < m; ++i) {
        Vec x(n);
        // a stretched cloud along a slanted direction, so that the spectrum is
        // not flat
        const double t = u(rng);
        for (std::size_t j = 0; j < n; ++j)
            x[j] = lo + (hi - lo) * (0.5 + 0.3 * (t - 0.5) * (j % 2 ? 1.0 : -1.0)
                                     + 0.05 * (u(rng) - 0.5));
        v.X.push_back(x);
        v.F.push_back({t, 1.0 - t});
    }
    v.box = Bounds(n, {lo, hi});
    return v;
}

void test_basis()
{
    std::cout << "-- the basis: complete, orthonormal, decreasing; few points --\n";
    std::mt19937 rng(3);
    for (std::size_t m : {std::size_t(40), std::size_t(4)}) {
        auto v = population(m, 8, 0.0, 1.0, 5);
        ops::LearnedCoords co;
        co.kind = ops::CoordsKind::PCA;
        co.fit(v.X, v.F, v.box, rng);
        check(co.basis().size() == 8 && orthonormality_error(co.basis()) < 1e-10,
              "pca, " + std::to_string(m) + " points of 8 variables: R is 8 x 8 and orthonormal");
        bool down = true;
        for (std::size_t i = 1; i < co.eigenvalues().size(); ++i)
            down = down && co.eigenvalues()[i] <= co.eigenvalues()[i - 1] + 1e-15;
        check(down, "the eigenvalues decrease");
        double back = 0.0;
        for (const auto& x : v.X) {
            const Vec z = co.to_x(co.to_y(x));
            for (std::size_t j = 0; j < x.size(); ++j) back = std::max(back, std::abs(z[j] - x[j]));
        }
        check(back < 1e-12, "x -> y -> x is the identity");
    }
}

void test_randrot()
{
    std::cout << "-- randrot: orthogonal, new at every update, the seed's own --\n";
    auto v = population(30, 6, 0.0, 1.0, 7);
    ops::LearnedCoords a, b;
    a.kind = b.kind = ops::CoordsKind::RandRot;
    std::mt19937 ra(11), rb(11);
    a.fit(v.X, v.F, v.box, ra);
    const auto first = a.basis();
    a.fit(v.X, v.F, v.box, ra);
    b.fit(v.X, v.F, v.box, rb);
    check(orthonormality_error(a.basis()) < 1e-12, "R is orthogonal");
    check(a.basis() != first, "the next update draws another rotation");
    check(b.basis() == first, "the same seed draws the same rotation");
}

// Offspring of SBX and polynomial mutation in pca coordinates from a
// population, and from the population turned by M: the second's children,
// turned back, have the first's mean and covariance (10^4 children each).
void test_equivariance()
{
    std::cout << "-- pca: the offspring turn with the population --\n";
    const std::size_t n = 4;
    auto v = population(40, n, -1000.0, 1000.0, 13);
    for (auto& x : v.X) for (double& c : x) c = (c + 1000.0) / 2000.0;     // a cloud near 0.5
    std::mt19937 qrng(17);
    const auto M = ops::coords_detail::haar(n, qrng);
    FakeVault w = v;
    for (auto& x : w.X) {                                    // x M
        Vec y(n, 0.0);
        for (std::size_t c = 0; c < n; ++c)
            for (std::size_t i = 0; i < n; ++i) y[c] += x[i] * M[i][c];
        x = y;
    }
    auto offspring = [&](FakeVault& pop, unsigned seed, bool turn_back) {
        ops::CrossoverSpec xo;
        xo.coords.kind = ops::CoordsKind::PCA;
        xo.coords.warmup = 0;
        ops::MutationSpec mu;
        std::mt19937 rng(seed);
        xo.observe(pop, static_cast<int>(pop.X.size()), rng);
        std::uniform_int_distribution<std::size_t> any(0, pop.X.size() - 1);
        auto draw = [&] { return pop.X[any(rng)]; };
        auto best = [&](double) { return pop.X[0]; };
        std::vector<Vec> kids;
        Vec c1, c2;
        for (int t = 0; t < 5000; ++t) {
            const Vec& a = pop.X[any(rng)];
            const Vec& b = pop.X[any(rng)];
            ops::vary(xo, mu, a, b, draw, best, 40, c1, c2, pop.box, 15.0, 0.9, 20.0, 0.25, rng,
                      [] { return -1; });
            for (const Vec* c : {&c1, &c2}) {
                Vec y = *c;
                if (turn_back) {                             // y Mᵀ
                    Vec z(n, 0.0);
                    for (std::size_t i = 0; i < n; ++i)
                        for (std::size_t k = 0; k < n; ++k) z[i] += y[k] * M[i][k];
                    y = z;
                }
                kids.push_back(y);
            }
        }
        return kids;
    };
    const auto A = offspring(v, 21, false);
    const auto B = offspring(w, 22, true);
    auto moments = [&](const std::vector<Vec>& K, Vec& mean, std::vector<Vec>& cov) {
        mean.assign(n, 0.0);
        for (const auto& k : K) for (std::size_t j = 0; j < n; ++j) mean[j] += k[j];
        for (double& m : mean) m /= static_cast<double>(K.size());
        cov.assign(n, Vec(n, 0.0));
        for (const auto& k : K)
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t b = 0; b < n; ++b) cov[a][b] += (k[a] - mean[a]) * (k[b] - mean[b]);
        for (auto& r : cov) for (double& c : r) c /= static_cast<double>(K.size() - 1);
    };
    Vec ma, mb;
    std::vector<Vec> ca, cb;
    moments(A, ma, ca);
    moments(B, mb, cb);
    bool mean_ok = true, cov_ok = true;
    for (std::size_t a = 0; a < n; ++a) {
        const double se = std::sqrt((ca[a][a] + cb[a][a]) / static_cast<double>(A.size()));
        mean_ok = mean_ok && std::abs(ma[a] - mb[a]) < 4.0 * se;
        for (std::size_t b = 0; b < n; ++b)
            cov_ok = cov_ok && std::abs(ca[a][b] - cb[a][b]) <
                               0.1 * std::sqrt(ca[a][a] * ca[b][b]) + 1e-12;
    }
    check(mean_ok, "the mean of the children turns with the population (4 standard errors)");
    check(cov_ok, "their covariance too (to 0.1 of the scale)");
}

void test_box()
{
    std::cout << "-- children come back into the box --\n";
    const std::size_t n = 5;
    auto v = population(30, n, 0.0, 1.0, 29);
    for (auto& x : v.X) x[0] = 0.999;                         // against a bound
    bool inside = true;
    for (auto kind : {ops::CoordsKind::PCA, ops::CoordsKind::PCAND, ops::CoordsKind::RandRot}) {
        ops::CrossoverSpec xo;
        xo.coords.kind = kind;
        xo.coords.warmup = 0;
        ops::MutationSpec mu;
        std::mt19937 rng(31);
        xo.observe(v, 30, rng);
        std::uniform_int_distribution<std::size_t> any(0, v.X.size() - 1);
        auto draw = [&] { return v.X[any(rng)]; };
        auto best = [&](double) { return v.X[0]; };
        Vec c1, c2;
        for (int t = 0; t < 2000; ++t) {
            ops::vary(xo, mu, v.X[any(rng)], v.X[any(rng)], draw, best, 30, c1, c2, v.box, 15.0,
                      1.0, 20.0, 1.0, rng, [] { return -1; });
            for (const Vec* c : {&c1, &c2})
                for (double x : *c) inside = inside && x >= 0.0 && x <= 1.0;
        }
    }
    check(inside, "pca, pca_nd, randrot: every child inside [0, 1]^n after reflect");
}

template <typename Core>
std::vector<Vec> run(const std::string& coords, double warmup, unsigned seed)
{
    Problem<CoordsZDT1> prob;
    DataVault<CoordsZDT1> vault(20, prob);
    Optimizer<CoordsZDT1, Core> opt(std::move(vault), defer_setup);
    auto& core = opt.get_algorithm();
    core.set_seed(seed);
    if (!coords.empty()) {
        core.set_word("coords", coords);
        core.set_param("coords_warmup", warmup);
    }
    opt.setup();
    for (int g = 0; g < 60; ++g) opt.step();
    auto& v = opt.get_vault();
    std::vector<Vec> out;
    for (std::size_t i = 0; i < v.active_n(); ++i) out.push_back(v.variables_of(i));
    return out;
}

void test_endless_warmup_is_axis()
{
    std::cout << "-- a warm-up that never ends is axis, bit for bit --\n";
    check(run<NSGAIICore<CoordsZDT1>>("", 0, 5) == run<NSGAIICore<CoordsZDT1>>("pca", 1e9, 5),
          "nsga2: pca with coords_warmup 1e9 runs the axis run");
    check(run<SMSEMOACore<CoordsZDT1>>("", 0, 5) == run<SMSEMOACore<CoordsZDT1>>("pca", 1e9, 5),
          "sms_emoa: the same");
    check(run<MOSACore<CoordsZDT1>>("", 0, 5) == run<MOSACore<CoordsZDT1>>("pca", 1e9, 5),
          "mosa: the same");
    check(run<NSGAIICore<CoordsZDT1>>("", 0, 5) != run<NSGAIICore<CoordsZDT1>>("pca", 0, 5),
          "nsga2: pca from the start is another run");
    check(run<MOSACore<CoordsZDT1>>("", 0, 5) != run<MOSACore<CoordsZDT1>>("pca", 0, 5),
          "mosa: the same (its sbx step, mix by default)");
    bool refused = false;
    try { run<NSGAIICore<CoordsZDT1>>("diagonal", 0, 5); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "an unknown coords word is refused");
}

}   // namespace

int main()
{
    std::cout << "coords\n";
    test_basis();
    test_randrot();
    test_equivariance();
    test_box();
    test_endless_warmup_is_axis();
    return mootation::testing::report("coords");
}

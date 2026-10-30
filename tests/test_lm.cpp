// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/lm.hpp (task 18, stage 2) on least-squares problems whose solutions
// are known, with the Jacobian given and by differences:
//   - Rosenbrock, Freudenstein & Roth (its local minimum 24.4921...), Powell's
//     singular function, Wood's, a linear model of full rank: the minimum, and
//     the evaluations the method takes (the values below, ours: Moré 1978's
//     rules; MINPACK lmdif, whose rules differ in places, takes 54, 30, 1002,
//     326, 13);
//   - lm_geodesic reaches the same minima;
//   - lm_loss = huber, cauchy: the weights' fixed point is the M-estimate
//     (psi(r)^T J = 0 there);
//   - the diagnostics: the singular values of a matrix known exactly, a
//     rank-deficient model (one small singular value), the covariance against
//     (J^T J)^-1;
//   - check_jacobian finds a wrong column;
//   - a checksum of the iterates' bits (the same on every platform: these
//     problems take only + - * / and sqrt).
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mootation/scalar/lm.hpp>

namespace {

namespace LM = mootation::scalar::lm;
using LM::Vec;
using mootation::lsq::Matrix;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

struct Case {
    const char* name;
    std::size_t n, m;
    Vec x0, x_min;          // x_min empty: only the cost is known
    double cost_min;
    LM::Problem p;
};

LM::Problem make(std::size_t n, std::size_t m, std::function<void(const Vec&, Vec&)> r,
                 std::function<void(const Vec&, Matrix&)> J = nullptr)
{
    LM::Problem p;
    p.n = n;
    p.m = m;
    p.residuals = std::move(r);
    p.jacobian = std::move(J);
    return p;
}

std::vector<Case> cases(bool with_jacobian)
{
    std::vector<Case> c;
    c.push_back({"rosenbrock", 2, 2, {-1.2, 1.0}, {1.0, 1.0}, 0.0,
                 make(2, 2, [](const Vec& x, Vec& r) { r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]}; },
                      [](const Vec& x, Matrix& J) {
                          J(0, 0) = -20.0 * x[0]; J(0, 1) = 10.0;
                          J(1, 0) = -1.0;         J(1, 1) = 0.0;
                      })});
    c.push_back({"freudenstein_roth", 2, 2, {0.5, -2.0}, {}, 24.492126863534967,
                 make(2, 2,
                      [](const Vec& x, Vec& r) {
                          r = {-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1],
                               -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]};
                      },
                      [](const Vec& x, Matrix& J) {
                          J(0, 0) = 1.0; J(0, 1) = 10.0 * x[1] - 3.0 * x[1] * x[1] - 2.0;
                          J(1, 0) = 1.0; J(1, 1) = 3.0 * x[1] * x[1] + 2.0 * x[1] - 14.0;
                      })});
    const double s5 = std::sqrt(5.0), s10 = std::sqrt(10.0), s90 = std::sqrt(90.0);
    c.push_back({"powell_singular", 4, 4, {3.0, -1.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 0.0}, 0.0,
                 make(4, 4,
                      [=](const Vec& x, Vec& r) {
                          const double a = x[1] - 2.0 * x[2], b = x[0] - x[3];
                          r = {x[0] + 10.0 * x[1], s5 * (x[2] - x[3]), a * a, s10 * b * b};
                      },
                      [=](const Vec& x, Matrix& J) {
                          const double a = x[1] - 2.0 * x[2], b = x[0] - x[3];
                          for (double& v : J.a) v = 0.0;
                          J(0, 0) = 1.0; J(0, 1) = 10.0;
                          J(1, 2) = s5;  J(1, 3) = -s5;
                          J(2, 1) = 2.0 * a; J(2, 2) = -4.0 * a;
                          J(3, 0) = 2.0 * s10 * b; J(3, 3) = -2.0 * s10 * b;
                      })});
    c.push_back({"wood", 4, 6, {-3.0, -1.0, -3.0, -1.0}, {1.0, 1.0, 1.0, 1.0}, 0.0,
                 make(4, 6,
                      [=](const Vec& x, Vec& r) {
                          r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0], s90 * (x[3] - x[2] * x[2]), 1.0 - x[2],
                               s10 * (x[1] + x[3] - 2.0), (x[1] - x[3]) / s10};
                      },
                      [=](const Vec& x, Matrix& J) {
                          for (double& v : J.a) v = 0.0;
                          J(0, 0) = -20.0 * x[0]; J(0, 1) = 10.0;
                          J(1, 0) = -1.0;
                          J(2, 2) = -2.0 * s90 * x[2]; J(2, 3) = s90;
                          J(3, 2) = -1.0;
                          J(4, 1) = s10; J(4, 3) = s10;
                          J(5, 1) = 1.0 / s10; J(5, 3) = -1.0 / s10;
                      })});
    // r_i = x_i - 2/m sum x - 1 (i < n), -2/m sum x - 1 (i >= n): the minimum
    // x_i = -1, cost (m - n) / 2
    c.push_back({"linear_full_rank", 5, 10, {1.0, 1.0, 1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, -1.0}, 2.5,
                 make(5, 10,
                      [](const Vec& x, Vec& r) {
                          double s = 0.0;
                          for (double v : x) s += v;
                          r.assign(10, -2.0 / 10.0 * s - 1.0);
                          for (std::size_t i = 0; i < 5; ++i) r[i] = x[i] - 2.0 / 10.0 * s - 1.0;
                      },
                      [](const Vec&, Matrix& J) {
                          for (std::size_t i = 0; i < 10; ++i)
                              for (std::size_t j = 0; j < 5; ++j) J(i, j) = (i == j ? 1.0 : 0.0) - 0.2;
                      })});
    if (!with_jacobian)
        for (Case& k : c) k.p.jacobian = nullptr;
    return c;
}

bool reached(const Case& k, const LM::Result& r, double tol)
{
    bool ok = std::fabs(r.cost - k.cost_min) <= tol * std::max(1.0, k.cost_min);
    for (std::size_t j = 0; j < k.x_min.size(); ++j) ok = ok && std::fabs(r.x[j] - k.x_min[j]) <= 1e-4;
    if (!ok) std::printf("      %s: cost %.17g, x0 %.17g\n", k.name, r.cost, r.x[0]);
    return ok;
}

std::uint64_t mix(std::uint64_t h, double v)
{
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return (h ^ b) * 1099511628211ull;
}

}   // namespace

int main()
{
    std::uint64_t hash = 1469598103934665603ull;

    // the minima, by differences; the evaluations the method takes
    {
        const long long nfev[] = {39, 31, 1001, 310, 13};   // powell_singular stops at max_nfev
        const std::vector<Case> cs = cases(false);
        for (std::size_t i = 0; i < cs.size(); ++i) {
            const LM::Result r = LM::solve(cs[i].p, cs[i].x0, LM::Options{});
            const double tol = cs[i].x_min.empty() ? 1e-9 : 1e-8;
            check(reached(cs[i], r, tol), std::string("differences: ") + cs[i].name + " reaches its minimum");
            check(r.nfev == nfev[i], std::string("differences: ") + cs[i].name + " takes " +
                                         std::to_string(nfev[i]) + " evaluations, got " + std::to_string(r.nfev));
            for (double v : r.x) hash = mix(hash, v);
            hash = mix(hash, static_cast<double>(r.nfev));
        }
    }
    // the same with the Jacobian, and with lm_geodesic
    for (bool geodesic : {false, true}) {
        const std::vector<Case> cs = cases(true);
        for (const Case& k : cs) {
            LM::Options o;
            o.geodesic = geodesic;
            const LM::Result r = LM::solve(k.p, k.x0, o);
            check(reached(k, r, k.x_min.empty() ? 1e-9 : 1e-8),
                  std::string(geodesic ? "geodesic: " : "jacobian: ") + k.name + " reaches its minimum");
            for (double v : r.x) hash = mix(hash, v);
            hash = mix(hash, static_cast<double>(r.nfev));
        }
    }

    // lm_loss: a line fitted through points with two outliers; at the
    // solution sum_i psi(r_i) dr_i/dx = 0
    {
        const std::size_t m = 12;
        Vec t(m), y(m);
        for (std::size_t i = 0; i < m; ++i) {
            t[i] = static_cast<double>(i);
            y[i] = 1.0 + 0.5 * t[i] + (i % 3 == 0 ? 0.01 : -0.01);
        }
        y[3] += 5.0;
        y[8] -= 7.0;
        LM::Problem p = make(2, m, [&](const Vec& x, Vec& r) {
            r.resize(m);
            for (std::size_t i = 0; i < m; ++i) r[i] = x[0] + x[1] * t[i] - y[i];
        });
        for (auto loss : {LM::Options::Loss::huber, LM::Options::Loss::cauchy}) {
            LM::Options o;
            o.loss = loss;
            o.f_scale = 0.1;
            const LM::Result r = LM::solve_robust(p, {0.0, 0.0}, o);
            double g0 = 0.0, g1 = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                const double psi = LM::weight_of(loss, o.f_scale, r.r[i]) * r.r[i];
                g0 += psi;
                g1 += psi * t[i];
            }
            const char* name = loss == LM::Options::Loss::huber ? "huber" : "cauchy";
            check(std::fabs(g0) < 1e-7 && std::fabs(g1) < 1e-6,
                  std::string("lm_loss = ") + name + ": psi(r)^T J = 0 at the solution");
            check(std::fabs(r.x[0] - 1.0) < 0.05 && std::fabs(r.x[1] - 0.5) < 0.01,
                  std::string("lm_loss = ") + name + ": the outliers do not move the line");
            for (double v : r.x) hash = mix(hash, v);
        }
    }

    // the diagnostics
    {
        Matrix J(4, 3);                         // columns orthogonal: singular values 4, 2, 1
        J(0, 0) = 2.0; J(1, 0) = 2.0; J(2, 0) = 2.0; J(3, 0) = 2.0;
        J(0, 1) = 1.0; J(1, 1) = -1.0; J(2, 1) = 1.0; J(3, 1) = -1.0;
        J(0, 2) = 0.5; J(1, 2) = 0.5; J(2, 2) = -0.5; J(3, 2) = -0.5;
        const LM::Diagnostics d = LM::diagnose(J, 0.5, 0.0);
        check(std::fabs(d.sv[0] - 4.0) < 1e-14 && std::fabs(d.sv[1] - 2.0) < 1e-14 &&
                  std::fabs(d.sv[2] - 1.0) < 1e-14 && d.small == 0,
              "diagnose: the singular values 4, 2, 1");
        // s^2 = 2 cost / (m - n) = 1; (J^T J)^-1 = diag(1/16, 1/4, 1)
        check(std::fabs(d.cov(0, 0) - 1.0 / 16.0) < 1e-15 && std::fabs(d.cov(1, 1) - 0.25) < 1e-15 &&
                  std::fabs(d.cov(2, 2) - 1.0) < 1e-14 && std::fabs(d.cov(0, 1)) < 1e-15 &&
                  std::fabs(d.corr(0, 0) - 1.0) < 1e-14,
              "diagnose: the covariance (J^T J)^-1 and the correlations");
        Matrix K(4, 3);                         // the third column the first's: rank 2
        for (std::size_t i = 0; i < 4; ++i) {
            K(i, 0) = J(i, 0);
            K(i, 1) = J(i, 1);
            K(i, 2) = J(i, 0);
        }
        const LM::Diagnostics e = LM::diagnose(K, 0.5, 2.0);
        check(e.small == 1 && e.sigma2 == 2.0 && e.sigma2_given, "diagnose: one small singular value; sigma2 given");
    }

    // check_jacobian
    {
        Case k = cases(true)[3];                // Wood
        LM::Problem p = k.p;
        const LM::JacobianCheck ok = LM::check_jacobian(p, {0.3, 0.7, -0.2, 1.1});
        bool small = true;
        for (double e : ok.column_error) small = small && e < 1e-6;
        check(small, "check_jacobian: Wood's Jacobian agrees with differences");
        auto J0 = p.jacobian;
        p.jacobian = [J0](const Vec& x, Matrix& J) {
            J0(x, J);
            J(2, 2) *= 1.01;
        };
        const LM::JacobianCheck bad = LM::check_jacobian(p, {0.3, 0.7, -0.2, 1.1});
        check(bad.worst[0] == 2 && bad.column_error[2] > 1e-3, "check_jacobian: the wrong column comes first");
    }

    std::printf("checksum %016llx\n", static_cast<unsigned long long>(hash));
    check(hash == 0xf74755ffd406621eull, "the iterates' bits as on MSVC and clang (checksum f74755ffd406621e)");
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

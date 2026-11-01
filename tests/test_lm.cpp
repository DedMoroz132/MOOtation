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
//     problems take only + - * / and sqrt);
//   - task 19, §19.3: equalities (solve_eq) against the KKT system, on the
//     sphere, lowrank balanced; the gauge basis on lowrank (steps orthogonal
//     to N, the same minimum, J Sigma J^T against the gauge fixed
//     explicitly); profiles (a linear model's parabola and interval, a flat
//     one), chi^2 quantiles.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/lm.hpp>
#include <mootation/suite/factor.hpp>

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

// ── task 19, §19.3: equalities, the gauge basis, profiles ───────────────────
// A x = b by Gaussian elimination with partial pivoting (the KKT systems below)
Vec gauss(Matrix A, Vec b)
{
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t i = c + 1; i < n; ++i)
            if (std::fabs(A(i, c)) > std::fabs(A(piv, c))) piv = i;
        for (std::size_t j = 0; j < n; ++j) std::swap(A(c, j), A(piv, j));
        std::swap(b[c], b[piv]);
        for (std::size_t i = c + 1; i < n; ++i) {
            const double t = A(i, c) / A(c, c);
            for (std::size_t j = c; j < n; ++j) A(i, j) -= t * A(c, j);
            b[i] -= t * b[c];
        }
    }
    Vec x(n);
    for (std::size_t ii = n; ii-- > 0;) {
        double s = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j) s -= A(ii, j) * x[j];
        x[ii] = s / A(ii, ii);
    }
    return x;
}

double max_abs_diff(const Vec& a, const Vec& b)
{
    double d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

// the lowrank problem of the suite (suite/factor.hpp) as an lm problem
LM::Problem lowrank_problem(const mootation::suite::sym::Lowrank& lr)
{
    LM::Problem p;
    p.n = static_cast<std::size_t>(lr.n);
    p.m = static_cast<std::size_t>(lr.m);
    p.residuals = [&lr](const Vec& x, Vec& r) {
        r.assign(static_cast<std::size_t>(lr.m), 0.0);
        lr.residuals(x.data(), r.data(), nullptr);
    };
    p.jacobian = [&lr](const Vec& x, Matrix& J) {
        J = Matrix(static_cast<std::size_t>(lr.m), static_cast<std::size_t>(lr.n));
        Vec r(static_cast<std::size_t>(lr.m));
        lr.residuals(x.data(), r.data(), J.a.data());
    };
    return p;
}

mootation::suite::sym::Lowrank lowrank(int rows, int cols, int rank, double noise)
{
    mootation::suite::sym::Instance in;
    in.teacher = rank;
    in.shape = {rows, cols};
    in.noise = noise;
    return mootation::suite::sym::Lowrank(in);
}

// a number in a check's message, three digits
std::string sci(double v)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3g", v);
    return buf;
}

void task19()
{
    std::mt19937 g(19);
    auto unit = [&g]() { return 2.0 * mootation::rnd::uniform01(g) - 1.0; };

    // 19.3.2: a linear least-squares problem with linear equalities, against the KKT system
    {
        const std::size_t m = 8, n = 4, pc = 2;
        Matrix E(m, n), C(pc, n);
        Vec f(m), d(pc);
        for (double& v : E.a) v = unit();
        for (double& v : f) v = unit();
        for (double& v : C.a) v = unit();
        for (double& v : d) v = unit();
        Matrix K(n + pc, n + pc);                     // [E^T E  C^T; C  0] [x; lambda] = [E^T f; d]
        Vec rhs(n + pc, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = 0; k < m; ++k) K(i, j) += E(k, i) * E(k, j);
            for (std::size_t k = 0; k < m; ++k) rhs[i] += E(k, i) * f[k];
            for (std::size_t c = 0; c < pc; ++c) K(i, n + c) = K(n + c, i) = C(c, i);
        }
        for (std::size_t c = 0; c < pc; ++c) rhs[n + c] = d[c];
        const Vec kkt = gauss(K, rhs);
        const Vec x_kkt(kkt.begin(), kkt.begin() + static_cast<std::ptrdiff_t>(n));
        const Vec l_kkt(kkt.begin() + static_cast<std::ptrdiff_t>(n), kkt.end());
        LM::Problem p = make(n, m, [&](const Vec& x, Vec& r) {
            r.assign(m, 0.0);
            for (std::size_t k = 0; k < m; ++k) {
                for (std::size_t j = 0; j < n; ++j) r[k] += E(k, j) * x[j];
                r[k] -= f[k];
            }
        }, [&](const Vec&, Matrix& J) { J = E; });
        LM::Equalities eq;
        eq.C = C;
        eq.d = d;
        const LM::EqResult a = LM::solve_eq(p, eq, Vec(n, 0.0), LM::Options{});
        check(a.eliminated && max_abs_diff(a.fit.x, x_kkt) <= 1e-12 && max_abs_diff(a.lambda, l_kkt) <= 1e-11,
              "solve_eq: linear equalities eliminated = the KKT solution (x " +
                  sci(max_abs_diff(a.fit.x, x_kkt)) + ", lambda " + sci(max_abs_diff(a.lambda, l_kkt)) + ")");
        p.lo.assign(n, -1e6);                         // a finite box: the bound-constrained Lagrangian
        p.hi.assign(n, 1e6);
        const LM::EqResult b = LM::solve_eq(p, eq, Vec(n, 0.0), LM::Options{});
        check(!b.eliminated && b.converged && max_abs_diff(b.fit.x, x_kkt) <= 1e-8 &&
                  max_abs_diff(b.lambda, l_kkt) <= 1e-6,
              "solve_eq: the same with a box, Algorithm 17.4 (x " + sci(max_abs_diff(b.fit.x, x_kkt)) +
                  ", " + sci(b.outer) + " outer iterations)");
    }
    // 19.3.2: min 1/2 ||x - a||^2 on ||x||^2 = 1 -> a / ||a||; lambda = (||a|| - 1) / 2
    for (int variant = 0; variant < 3; ++variant) {
        const Vec a = {3.0, -1.0, 2.0};
        const double na = std::sqrt(14.0);
        LM::Problem p = make(3, 3, [&](const Vec& x, Vec& r) {
            r.resize(3);
            for (std::size_t i = 0; i < 3; ++i) r[i] = x[i] - a[i];
        }, [](const Vec&, Matrix& J) {
            J = Matrix(3, 3);
            for (std::size_t i = 0; i < 3; ++i) J(i, i) = 1.0;
        });
        LM::Equalities eq;
        eq.q = 1;
        eq.h = [](const Vec& x, Vec& h) { h = {x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1.0}; };
        if (variant != 1)
            eq.jacobian = [](const Vec& x, Matrix& A) {
                A = Matrix(1, 3);
                for (std::size_t j = 0; j < 3; ++j) A(0, j) = 2.0 * x[j];
            };
        if (variant == 2) {
            p.lo.assign(3, -2.0);
            p.hi.assign(3, 2.0);
        }
        const LM::EqResult r = LM::solve_eq(p, eq, {0.5, 0.5, 0.5}, LM::Options{});
        const Vec want = {a[0] / na, a[1] / na, a[2] / na};
        const char* what[] = {"the Jacobian of h", "h by differences", "and a box"};
        check(r.converged && max_abs_diff(r.fit.x, want) <= 1e-6 && std::fabs(r.lambda[0] - (na - 1.0) / 2.0) <= 1e-5,
              std::string("solve_eq: min ||x - a|| on the sphere, ") + what[variant] + " (" +
                  sci(max_abs_diff(r.fit.x, want)) + ", lambda " + sci(r.lambda[0]) + ", converged " + sci(r.converged) + ", outer " + sci(r.outer) + ", pg " + sci(r.projected_gradient) + ", c " + sci(r.c_norm) + ")");
    }
    // 19.3.2: lowrank balanced, U^T U = V^T V (the upper triangle): a solution exists (f = 0)
    {
        const auto lr = lowrank(6, 5, 2, 0.0);
        LM::Problem p = lowrank_problem(lr);
        const int w = lr.rows + lr.cols, r = lr.rank;
        LM::Equalities eq;
        eq.q = static_cast<std::size_t>(r * (r + 1) / 2);
        eq.h = [&lr, w, r](const Vec& x, Vec& h) {
            h.clear();
            for (int a = 0; a < r; ++a)
                for (int b = a; b < r; ++b) {
                    double s = 0.0;
                    for (int i = 0; i < lr.rows; ++i) s += x[static_cast<std::size_t>(a * w + i)] * x[static_cast<std::size_t>(b * w + i)];
                    for (int j = 0; j < lr.cols; ++j)
                        s -= x[static_cast<std::size_t>(a * w + lr.rows + j)] * x[static_cast<std::size_t>(b * w + lr.rows + j)];
                    h.push_back(s);
                }
        };
        Vec x0(p.n);
        lr.start(x0.data());
        const LM::EqResult res = LM::solve_eq(p, eq, x0, LM::Options{});
        check(res.converged && res.c_norm <= 1e-10 && res.fit.cost <= 1e-16,
              "solve_eq: lowrank with U^T U = V^T V (||c|| " + sci(res.c_norm) + ", f " +
                  sci(res.fit.cost) + ")");
    }

    // 19.3.1: the gauge basis on lowrank with noise
    {
        const auto lr = lowrank(8, 6, 2, 0.05);
        LM::Problem p = lowrank_problem(lr);
        p.gauge = [&lr](const Vec& x, Matrix& N) {
            N = Matrix(static_cast<std::size_t>(lr.n), static_cast<std::size_t>(lr.gauge_dim));
            lr.gauge(x.data(), N.a.data());
        };
        Vec x0(p.n);
        lr.start(x0.data());
        std::vector<Vec> its{x0};
        LM::Options plain;
        plain.scale = false;
        plain.iterates = &its;
        const LM::Result rp = LM::solve(p, x0, plain);
        double worst = 0.0;
        for (std::size_t k = 0; k + 1 < its.size(); ++k) {
            Matrix N;
            p.gauge(its[k], N);
            Vec dx(p.n);
            for (std::size_t i = 0; i < p.n; ++i) dx[i] = its[k + 1][i] - its[k][i];
            for (std::size_t c = 0; c < N.cols; ++c) {
                double dot = 0.0, nn = 0.0;
                for (std::size_t i = 0; i < p.n; ++i) {
                    dot += N(i, c) * dx[i];
                    nn += N(i, c) * N(i, c);
                }
                // the step to 1e-12, the difference of the iterates to their own rounding (x + s)
                const double allowed = std::sqrt(nn) * (1e-12 * LM::norm(dx) + 2.2e-16 * LM::norm(its[k + 1]));
                worst = std::max(worst, std::fabs(dot) / allowed);
            }
        }
        check(its.size() > 3 && worst <= 1.0, "lm gauge: the steps are orthogonal to N (scale = false; " +
                                                     sci(static_cast<double>(its.size() - 1)) + " steps, worst share of the tolerance " + sci(worst) + ")");
        const LM::Result rg = LM::solve(p, x0, LM::Options{});
        LM::Problem free = p;
        free.gauge = nullptr;
        const LM::Result rf = LM::solve(free, x0, LM::Options{});
        check(std::fabs(rg.cost - rf.cost) <= 1e-8 * rf.cost && std::fabs(rp.cost - rf.cost) <= 1e-8 * rf.cost &&
                  rg.gauge_k == 4 && rg.gauge_ratio <= 1e-12,
              "lm gauge: the same minimum as without it; k = 4, ||J N|| / ||J|| = " + sci(rg.gauge_ratio));
        // J Sigma J^T on the complement against the gauge fixed explicitly: U's top r x r block = I
        Matrix J(p.m, p.n), N;
        p.jacobian(rg.x, J);
        p.gauge(rg.x, N);
        const LM::Diagnostics dg = LM::diagnose_gauge(J, N, rg.cost, 1.0);
        const int w = lr.rows + lr.cols, r = lr.rank;
        double T[2][2], A[2][2];                      // T = U's top block, A = T^-1
        for (int i = 0; i < r; ++i)
            for (int k = 0; k < r; ++k) T[i][k] = rg.x[static_cast<std::size_t>(k * w + i)];
        const double det = T[0][0] * T[1][1] - T[0][1] * T[1][0];
        A[0][0] = T[1][1] / det;
        A[1][1] = T[0][0] / det;
        A[0][1] = -T[0][1] / det;
        A[1][0] = -T[1][0] / det;
        Vec xf(p.n, 0.0);                             // U A, V A^-T = V T^T
        for (int k = 0; k < r; ++k) {
            for (int i = 0; i < lr.rows; ++i)
                for (int l = 0; l < r; ++l)
                    xf[static_cast<std::size_t>(k * w + i)] += rg.x[static_cast<std::size_t>(l * w + i)] * A[l][k];
            for (int j = 0; j < lr.cols; ++j)
                for (int l = 0; l < r; ++l)
                    xf[static_cast<std::size_t>(k * w + lr.rows + j)] += rg.x[static_cast<std::size_t>(l * w + lr.rows + j)] * T[k][l];
        }
        Matrix Jx(p.m, p.n);
        p.jacobian(xf, Jx);
        std::vector<std::size_t> keep;
        for (int k = 0; k < r; ++k)
            for (int i = 0; i < w; ++i)
                if (i >= r) keep.push_back(static_cast<std::size_t>(k * w + i));
        Matrix Jfix(p.m, keep.size());
        for (std::size_t i = 0; i < p.m; ++i)
            for (std::size_t c = 0; c < keep.size(); ++c) Jfix(i, c) = Jx(i, keep[c]);
        const LM::Diagnostics df = LM::diagnose(Jfix, rg.cost, 1.0);
        double diff = 0.0, big = 0.0;
        for (std::size_t a = 0; a < p.m; ++a)
            for (std::size_t b = 0; b < p.m; ++b) {
                double s1 = 0.0, s2 = 0.0;
                for (std::size_t i = 0; i < p.n; ++i)
                    for (std::size_t j = 0; j < p.n; ++j) s1 += J(a, i) * dg.cov(i, j) * J(b, j);
                for (std::size_t i = 0; i < keep.size(); ++i)
                    for (std::size_t j = 0; j < keep.size(); ++j) s2 += Jfix(a, i) * df.cov(i, j) * Jfix(b, j);
                diff = std::max(diff, std::fabs(s1 - s2));
                big = std::max(big, std::fabs(s1));
            }
        check(diff <= 1e-8 * big, "lm gauge: J Sigma J^T on the complement = with U's top block fixed to I (" +
                                       sci(diff / big) + ")");
    }

    // 19.3.3: chi^2 quantiles; df = 2 is -2 log(1 - alpha) exactly
    check(std::fabs(LM::chi2_quantile(0.95, 2) + 2.0 * std::log(0.05)) <= 1e-12 &&
              std::fabs(LM::chi2_quantile(0.95, 1) - 3.841458820694124) <= 1e-10 &&
              std::fabs(LM::chi2_quantile(0.95, 3) - 7.814727903251178) <= 1e-10 &&
              std::fabs(LM::chi2_quantile(0.6826894921370859, 1) - 1.0) <= 1e-10,
          "chi2_quantile: 5.99146 (exact), 3.84146, 7.81473, 1 at one sigma");
    // 19.3.3: the profile of a linear model is the parabola, its interval the classical one
    {
        const std::size_t m = 10, n = 3;
        Matrix E(m, n);
        Vec f(m);
        for (double& v : E.a) v = unit();
        for (double& v : f) v = unit();
        LM::Problem p = make(n, m, [&](const Vec& x, Vec& r) {
            r.assign(m, 0.0);
            for (std::size_t k = 0; k < m; ++k) {
                for (std::size_t j = 0; j < n; ++j) r[k] += E(k, j) * x[j];
                r[k] -= f[k];
            }
        }, [&](const Vec&, Matrix& J) { J = E; });
        LM::Options tight;
        tight.ftol = tight.xtol = 1e-15;
        const LM::Result fit = LM::solve(p, Vec(n, 0.0), tight);
        Matrix EtE(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = 0; k < m; ++k) EtE(i, j) += E(k, i) * E(k, j);
        const std::size_t j = 1;
        Vec ej(n, 0.0);
        ej[j] = 1.0;
        const double cjj = gauss(EtE, ej)[j];         // ((E^T E)^-1)_jj
        const LM::Profile pr = LM::profile(p, fit.x, j, tight);
        double worst = 0.0;
        for (const LM::ProfilePoint& pt : pr.points) {
            const double want = (pt.value - fit.x[j]) * (pt.value - fit.x[j]) / (2.0 * cjj);
            worst = std::max(worst, std::fabs((pt.cost - pr.cost_hat) - want));
        }
        const double half = std::sqrt(3.841458820694124 * cjj);
        check(worst <= 1e-10 && pr.kind == LM::Profile::Kind::identifiable && pr.relative &&
                  std::fabs(pr.lower - (fit.x[j] - half)) <= 1e-7 && std::fabs(pr.upper - (fit.x[j] + half)) <= 1e-7,
              "profile: a linear model's is the parabola (" + sci(worst) + "), its interval x_j -+ (3.84 C_jj)^1/2 (" +
                  sci(static_cast<double>(pr.points.size())) + " nodes)");
    }
    // 19.3.3: lowrank without the gauge basis: a flat profile, structurally non-identifiable
    {
        const auto lr = lowrank(6, 5, 2, 0.0);
        LM::Problem p = lowrank_problem(lr);
        LM::Options tight;
        tight.ftol = tight.xtol = 1e-15;
        Vec x0(p.n);
        lr.start(x0.data());
        const LM::Result fit = LM::solve(p, x0, tight);
        LM::ProfileOptions po;
        po.step = 0.2;
        po.max_steps = 4;
        const LM::Profile pr = LM::profile(p, fit.x, 0, tight, po);
        check(pr.kind == LM::Profile::Kind::structurally_non_identifiable && pr.points.size() == 9,
              "profile: lowrank's u_1[0] is flat (structurally non-identifiable)");
    }
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

    task19();

    std::printf("checksum %016llx\n", static_cast<unsigned long long>(hash));
    check(hash == 0xf74755ffd406621eull, "the iterates' bits as on MSVC and clang (checksum f74755ffd406621e)");
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

// SPDX-License-Identifier: Apache-2.0
// T-optimum discrimination (task 26, item 24.7.6; scalar/discriminate.hpp),
// Atkinson & Fedorov 1975a, Example 2: eta_t(x) = 4.5 - 1.5 e^x - 2 e^-x
// true, the quadratic rival fitted by weighted least squares, 21 candidates
// on [-1, 1] in steps of 0.1, the initial design -1, 0, 1, alpha_s = 1/(N +
// 1). The paper: Delta_2(xi*) = 1.087e-3 on four points, the largest
// difference of the responses 0.033. The sequential design approaches it
// from below; the equivalence theorem's certificate (max psi = Delta) tightens.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mootation/detmath.hpp>
#include <mootation/scalar/discriminate.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

// the weighted least squares quadratic through (x_c, y_c) with weights p_c:
// the 3 x 3 normal equations, Gaussian elimination with partial pivoting
std::vector<double> quadratic_fit(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& p)
{
    double A[3][4] = {};
    for (std::size_t c = 0; c < x.size(); ++c) {
        const double f[3] = {1.0, x[c], x[c] * x[c]};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) A[i][j] += p[c] * f[i] * f[j];
            A[i][3] += p[c] * f[i] * y[c];
        }
    }
    for (int k = 0; k < 3; ++k) {
        int piv = k;
        for (int i = k + 1; i < 3; ++i)
            if (std::fabs(A[i][k]) > std::fabs(A[piv][k])) piv = i;
        for (int j = 0; j < 4; ++j) std::swap(A[k][j], A[piv][j]);
        for (int i = k + 1; i < 3; ++i) {
            const double m = A[i][k] / A[k][k];
            for (int j = k; j < 4; ++j) A[i][j] -= m * A[k][j];
        }
    }
    double t[3];
    for (int i = 2; i >= 0; --i) {
        double s = A[i][3];
        for (int j = i + 1; j < 3; ++j) s -= A[i][j] * t[j];
        t[i] = s / A[i][i];
    }
    std::vector<double> pred(x.size());
    for (std::size_t c = 0; c < x.size(); ++c) pred[c] = t[0] + t[1] * x[c] + t[2] * x[c] * x[c];
    return pred;
}

}   // namespace

int main()
{
    namespace td = mootation::scalar::discriminate;
    std::vector<double> x(21), eta(21);
    for (int c = 0; c < 21; ++c) {
        x[static_cast<std::size_t>(c)] = -1.0 + 0.1 * c;
        eta[static_cast<std::size_t>(c)] = 4.5 - 1.5 * mootation::dm::exp(x[static_cast<std::size_t>(c)]) -
                                           2.0 * mootation::dm::exp(-x[static_cast<std::size_t>(c)]);
    }
    const double xm = 0.5 * std::log(4.0 / 3.0);                 // eta_t' = 0
    const double em = 4.5 - 1.5 * mootation::dm::exp(xm) - 2.0 * mootation::dm::exp(-xm);
    check(std::fabs(eta[0] - (-1.488)) < 1e-3 && std::fabs(em - 1.036) < 1e-3 && std::fabs(xm - 0.144) < 1e-3 &&
              std::fabs(eta[20] - (-0.313)) < 1e-3,
          "eta_t(-1) = -1.488, the maximum 1.036 at 0.144 (AF75a Example 2); eta_t(1) = -0.313, printed -0.131 (TD-4)");
    auto fit2 = [&](const std::vector<double>& p) { return quadratic_fit(x, eta, p); };
    const td::Result r = td::t_optimal(eta, fit2, {0, 10, 20}, 20000, 1e-3);
    int support = 0;
    for (double w : r.weights) support += w > 0.02 ? 1 : 0;
    std::printf("  Delta = %.6g, max psi = %.6g, steps %lld, support %d, sqrt(max psi) = %.4f\n", r.delta, r.psi_max,
                r.steps, support, std::sqrt(r.psi_max));
    check(r.delta > 1.06e-3 && r.delta <= 1.0875e-3, "Delta_2 approaches 1.087e-3 from below (AF75a §2)");
    check(r.psi_max < 1.02 * r.delta, "the equivalence theorem's certificate: max psi within 2 % of Delta");
    check(support == 4, "the T-optimum design's four points");
    check(std::fabs(std::sqrt(r.psi_max) - 0.033) < 1e-3, "the largest difference of the responses 0.033");
    const td::Result again = td::t_optimal(eta, fit2, {0, 10, 20}, 20000, 1e-3);
    check(again.trials == r.trials && again.delta == r.delta, "the same design twice");
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

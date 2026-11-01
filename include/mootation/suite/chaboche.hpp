#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// chaboche1d (task 19, §19.1.4): uniaxial elastoplasticity under strain
// control with N back stresses, fitted to a teacher's stresses.
//
// The model of J.L. Chaboche, "Constitutive equations for cyclic plasticity
// and cyclic viscoplasticity", Int. J. Plasticity 5 (1989) 247-302 (source
// chaboche1d_chaboche1989), in tension-compression, its (8) and (10), (12)-(13):
//   sigma = E (eps - eps_p);
//   f = |sigma - X| - R - sigma_y <= 0, X = sum_k X_k (their k is our sigma_y);
//   d eps_p = dp sign(sigma - X);
//   dX_k = C_k d eps_p - gamma_k X_k dp (Armstrong-Frederick; (5)'s 2/3 goes
//     with the uniaxial reading, (8): h = C - gamma X sign(sigma - X));
//   R = Q (1 - exp(-b p)) (Voce, their (13)).
// The parameters x = (E, sigma_y, Q, b, C_1, gamma_1, ..., C_N, gamma_N);
// symmetry: the pairs (C_k, gamma_k) may be interchanged (blocks: N of 2 from
// 4); canon sorts them by gamma, then C. The residuals: the student's stress
// minus the data at every step of the program; the Jacobian is not given
// (forward differences in the solver).
//
// The integration, step n -> n + 1 at the strain eps_{n+1}: the elastic
// trial and the return to the yield surface of J.C. Simo, T.J.R. Hughes,
// Computational Inelasticity, Springer 1998 (source return-mapping_simo1998),
// §1.4.2 (BOX 1.4: trial state, consistency, update) with a scalar Newton
// iteration for the consistency parameter as their §3.3 does for nonlinear
// hardening. Simo & Hughes have no Armstrong-Frederick law (their closed
// form, (1.4.22)-(1.4.23), is for linear hardening); its implicit
// discretization is ours (CH-1):
//   trial: sigma_tr = E (eps_{n+1} - eps_p,n), xi = sigma_tr - sum X_k,n,
//     f_tr = |xi| - R(p_n) - sigma_y; elastic where f_tr <= 0;
//   else, s = sign(xi), Delta p > 0 from g(Delta p) = 0:
//     X_k = (X_k,n + C_k Delta p s) / (1 + gamma_k Delta p)   (backward Euler)
//     g = |xi| - E Delta p - sum_k (s X_k - s X_k,n) - R(p_n + Delta p) - sigma_y
//     g' = -E - sum_k (C_k - gamma_k s X_k,n) / (1 + gamma_k Delta p)^2
//          - Q b exp(-b (p_n + Delta p));
//   eps_p = eps_p,n + Delta p s, p = p_n + Delta p, sigma = sigma_tr - E Delta p s.
// With |X_k,n| <= C_k / gamma_k (which the update keeps, from X = 0) g is
// decreasing and convex, so Newton's iteration from Delta p = 0 increases to
// the root (Simo & Hughes §3.3's argument for (3.3.5)); it stops where g <= 0
// or the step is below 4 eps Delta p, 100 iterations at most.
//
// The program (CH-2): symmetric cycles 0 -> +a -> -a -> 0 at the amplitudes a
// = 0.4 %, 0.8 %, 1.2 %, two cycles each, in steps of 0.04 % strain (480
// steps; residuals at every step). The teacher (CH-3): E = 200 000, sigma_y =
// 200, Q = 100, b = 10, C_k = 60 000 / 10^(k - 1), gamma_k = 600 / 10^(k - 1)
// (stresses in MPa). The noise (sym.hpp, SYM-1): one normal draw per step,
// from (20261102, crc32("chaboche1d"), instance, N_t, steps).
//
// READINGS:
//   CH-1. The backward-Euler update of the Armstrong-Frederick back stresses
//     and the scalar consistency equation in Delta p are our derivation; the
//     structure (trial, consistency, local Newton) is Simo & Hughes'.
//   CH-2. The program's amplitudes, cycles and step are ours.
//   CH-3. The teacher's values and the box are ours: E in [5e4, 5e5],
//     sigma_y in [10, 1000], Q in [0, 1000], b in [0.1, 100], C_k in [0,
//     2e5], gamma_k in [0, 5000]; the start (inside it, by lm's rule for a
//     start on a bound) E = 150 000, sigma_y = 150, Q = 50, b = 5, C_k =
//     30 000 / 10^(k - 1), gamma_k = 300 / 10^(k - 1).
// ============================================================================

#include <cmath>
#include <stdexcept>

#include "sym.hpp"

namespace mootation::suite::sym {

class Chaboche final : public Family {
public:
    int N = 0;                                    // the student's pairs
    Vec strain, y;                                // the program and the data

    explicit Chaboche(const Instance& in)
    {
        if (in.teacher < 1) throw std::invalid_argument("chaboche1d: N_t >= 1 back stresses");
        for (double a : {0.004, 0.008, 0.012}) {      // CH-2
            const int quarter = static_cast<int>(a / 0.0004 + 0.5);
            for (int cycle = 0; cycle < 2; ++cycle) {
                for (int i = 1; i <= quarter; ++i) strain.push_back(0.0004 * i);
                for (int i = quarter - 1; i >= -quarter; --i) strain.push_back(0.0004 * i);
                for (int i = -quarter + 1; i <= 0; ++i) strain.push_back(0.0004 * i);
            }
        }
        Vec teacher = {200000.0, 200.0, 100.0, 10.0};   // CH-3
        double scale = 1.0;
        for (int k = 0; k < in.teacher; ++k) {
            teacher.push_back(60000.0 / scale);
            teacher.push_back(600.0 / scale);
            scale = scale * 10.0;
        }
        y.assign(strain.size(), 0.0);
        stresses(teacher.data(), in.teacher, y.data());
        std::mt19937 g = generator({kKey, instances::crc32("chaboche1d"), static_cast<std::uint64_t>(in.instance),
                                    static_cast<std::uint64_t>(in.teacher), strain.size()});
        add_noise(g, in.noise, y);
        N = in.student > 0 ? in.student : in.teacher;
        if (N == in.teacher) optimum = teacher;
        n = 4 + 2 * N;
        m = static_cast<int>(strain.size());
        has_canon = true;
        blocks = {{4, 2, N}};
        bounds = {{5e4, 5e5}, {10.0, 1000.0}, {0.0, 1000.0}, {0.1, 100.0}};
        for (int k = 0; k < N; ++k) {
            bounds.emplace_back(0.0, 2e5);
            bounds.emplace_back(0.0, 5000.0);
        }
    }

    void residuals(const double* x, double* r, double*) const override
    {
        stresses(x, N, r);
        for (std::size_t i = 0; i < y.size(); ++i) r[i] = r[i] - y[i];
    }

    void canonical(double* x) const override
    {
        sort_blocks(x, blocks[0], [](const Vec& b) { return b[1]; });
    }

    void start(double* x) const override
    {
        x[0] = 150000.0;
        x[1] = 150.0;
        x[2] = 50.0;
        x[3] = 5.0;
        double scale = 1.0;
        for (int k = 0; k < N; ++k) {
            x[4 + 2 * k] = 30000.0 / scale;
            x[5 + 2 * k] = 300.0 / scale;
            scale = scale * 10.0;
        }
    }

    // the stress at every step of the program for the parameters p with
    // `pairs` back stresses
    void stresses(const double* p, int pairs, double* out) const
    {
        const double E = p[0], sy = p[1], Q = p[2], b = p[3];
        double eps_p = 0.0, pp = 0.0;
        Vec X(static_cast<std::size_t>(pairs), 0.0), Xn(X);
        for (std::size_t step = 0; step < strain.size(); ++step) {
            const double trial = E * (strain[step] - eps_p);
            double back = 0.0;
            for (double v : X) back += v;
            const double xi = trial - back;
            const double f_tr = std::fabs(xi) - Q * (1.0 - dm::exp(-b * pp)) - sy;
            if (f_tr <= 0.0) {
                out[step] = trial;
                continue;
            }
            const double s = xi < 0.0 ? -1.0 : 1.0;
            Xn = X;
            double dp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double g = std::fabs(xi) - E * dp - Q * (1.0 - dm::exp(-b * (pp + dp))) - sy;
                double dg = -E - Q * b * dm::exp(-b * (pp + dp));
                for (std::size_t k = 0; k < X.size(); ++k) {
                    const double C = p[4 + 2 * k], gam = p[5 + 2 * k], den = 1.0 + gam * dp;
                    g -= s * ((Xn[k] + C * dp * s) / den) - s * Xn[k];
                    dg -= (C - gam * s * Xn[k]) / (den * den);
                }
                if (g <= 0.0) break;
                const double step_dp = -g / dg;
                dp += step_dp;
                if (step_dp <= 4.0 * kEps * dp) break;
            }
            for (std::size_t k = 0; k < X.size(); ++k) {
                const double C = p[4 + 2 * k], gam = p[5 + 2 * k];
                X[k] = (Xn[k] + C * dp * s) / (1.0 + gam * dp);
            }
            eps_p += dp * s;
            pp += dp;
            out[step] = trial - E * dp * s;
        }
    }
};

}   // namespace mootation::suite::sym

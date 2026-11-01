#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// sumexp (task 19, §19.1.1): a sum of exponentials fitted to a teacher's.
//
// The teacher y(t) = sum_{k=1..K_t} a_k exp(-t / tau_k) on a grid of N points;
// the student's unknowns (a_k, s_k = log tau_k), k = 1..K_s, in that order,
// pair after pair; the residuals r_i = sum_k a_k exp(-t_i exp(-s_k)) - y_i,
// the Jacobian dr_i/da_k = E_ik = exp(-t_i exp(-s_k)), dr_i/ds_k = a_k E_ik
// t_i exp(-s_k). Symmetry: the K_s pairs may be interchanged (blocks: K_s of
// 2); canon sorts them by s (then a, SYM-3). The a_k enter linearly (linear:
// 0, 2, 4, ...), for variable projection.
//
// The definition is ours (§19.1.1); the scale of difficulty is the
// resolution limit of A.A. Istratov, O.F. Vyvenko, "Exponential analysis in
// physical phenomena", Rev. Sci. Instrum. 70(2) 1999, 1233-1257 (source
// sumexp_istratov1999), p. 1237, (12)-(13): two decay rates are resolved only
// if their ratio exceeds delta = exp(pi / omega_max), cosh(pi omega_max) = pi
// SNR^2 (resolution(snr) below; their Table I: 2.44, 1.88, 1.63 at SNR =
// 10^2, 10^3, 10^4). An instance's ratio tau_{k+1} / tau_k above
// resolution(1 / noise) is resolvable, below it is not.
//
// Instances:
//   the default: tau_k = ratio^(k - 1) (tau_1 = 1), a_k = 1, the grid t_i =
//     t_lo (t_hi / t_lo)^(i / (N - 1)), i = 0..N-1, t_lo = tau_1 / 10, t_hi =
//     10 tau_{K_t} (SE-1);
//   variant "lanczos": Lanczos's example as Istratov & Vyvenko reproduce it
//     (p. 1235, Fig. 2): the teacher f_3(t) = 0.0951 e^-t + 0.8607 e^-3t +
//     1.5576 e^-5t, on t = 0, 0.05, ..., 1.15 (24 points, SE-2); a student
//     of K_s = 2 is their f_2 = 2.202 e^-4.45t + 0.305 e^-1.58t's class.
// The data are the teacher evaluated by the student's model at (a_k, log
// tau_k), so that the teacher's point is the student's minimum, f = 0 bit
// for bit, where K_s = K_t and noise = 0; then the noise (sym.hpp, SYM-1):
// one normal draw per point in order, from the key (20261102,
// crc32("sumexp"), instance, K_t, N).
//
// READINGS:
//   SE-1. The default teacher's amplitudes (1) and grid (two decades beyond
//     the fastest and slowest decay, logarithmic as Istratov & Vyvenko's
//     sampling argument on p. 1237 has it) are ours.
//   SE-2. Lanczos's grid is not in the corpus (Istratov & Vyvenko's Ref. 30);
//     we take that of the NIST StRD files Lanczos1-3, which tabulate the same
//     f_3 at 24 points t = 0(0.05)1.15 (suite/nist_data.hpp).
//   SE-3. The box: a_k in [-2 Y, 2 Y], Y = max |y_i|; s_k in [log t_min, log
//     t_max], the smallest positive and the largest t of the grid (decays
//     faster or slower than the grid can see are outside). The start: a_k =
//     Y / K_s, s_k evenly inside the box's s range, (k - 1/2) / K_s of it.
// ============================================================================

#include <cmath>
#include <stdexcept>
#include <string>

#include "sym.hpp"

namespace mootation::suite::sym {

// Istratov & Vyvenko's resolution limit (12)-(13) at a signal-to-noise ratio:
// exp(pi / omega), omega = acosh(pi snr^2) / pi
inline double resolution(double snr)
{
    const double c = dm::kPi * snr * snr;
    const double omega = dm::log(c + std::sqrt(c * c - 1.0)) / dm::kPi;
    return dm::exp(dm::kPi / omega);
}

class Sumexp final : public Family {
public:
    Vec t, y;
    int K = 0;                        // the student's terms

    explicit Sumexp(const Instance& in)
    {
        Vec a, s;                     // the teacher's (a_k, log tau_k)
        if (in.variant == "lanczos") {
            a = {0.0951, 0.8607, 1.5576};
            for (double rate : {1.0, 3.0, 5.0}) s.push_back(-dm::log(rate));
            for (int i = 0; i < 24; ++i) t.push_back(0.05 * i);
        } else {
            if (!in.variant.empty()) throw std::invalid_argument("sumexp: variant lanczos or none");
            if (in.teacher < 1 || in.points < 2 || !(in.ratio > 1.0))
                throw std::invalid_argument("sumexp: K_t >= 1, N >= 2 points, ratio > 1");
            double tau = 1.0;
            for (int k = 0; k < in.teacher; ++k) {
                a.push_back(1.0);
                s.push_back(dm::log(tau));
                if (k + 1 < in.teacher) tau = tau * in.ratio;
            }
            const double lo = dm::log(0.1), hi = dm::log(10.0 * tau);
            for (int i = 0; i < in.points; ++i)
                t.push_back(dm::exp(lo + (hi - lo) * (static_cast<double>(i) / (in.points - 1))));
        }
        const int teacher = static_cast<int>(a.size());
        y.assign(t.size(), 0.0);
        for (std::size_t i = 0; i < t.size(); ++i) y[i] = model(a.data(), s.data(), teacher, t[i]);
        std::mt19937 g = generator({kKey, instances::crc32("sumexp"), static_cast<std::uint64_t>(in.instance),
                                    static_cast<std::uint64_t>(teacher), t.size()});
        add_noise(g, in.noise, y);

        K = in.student > 0 ? in.student : teacher;
        if (K == teacher)
            for (int k = 0; k < K; ++k) {
                optimum.push_back(a[static_cast<std::size_t>(k)]);
                optimum.push_back(s[static_cast<std::size_t>(k)]);
            }
        n = 2 * K;
        m = static_cast<int>(t.size());
        has_jacobian = has_canon = true;
        blocks = {{0, 2, K}};
        for (int k = 0; k < K; ++k) linear.push_back(2 * k);
        double big = 0.0, tmin = kInf, tmax = 0.0;
        for (double v : y) big = std::max(big, std::fabs(v));
        for (double v : t) {
            if (v > 0.0) tmin = std::min(tmin, v);
            tmax = std::max(tmax, v);
        }
        s_lo_ = dm::log(tmin);
        s_hi_ = dm::log(tmax);
        y_big_ = big;
        for (int k = 0; k < K; ++k) {                       // SE-3
            bounds.emplace_back(-2.0 * big, 2.0 * big);
            bounds.emplace_back(s_lo_, s_hi_);
        }
    }

    void residuals(const double* x, double* r, double* J) const override
    {
        for (std::size_t i = 0; i < t.size(); ++i) {
            r[i] = model(x, x + 1, K, t[i], 2) - y[i];
            if (!J) continue;
            for (int k = 0; k < K; ++k) {
                const double rate = dm::exp(-x[2 * k + 1]);
                const double e = dm::exp(-t[i] * rate);
                double* row = J + static_cast<int>(i) * n;
                row[2 * k] = e;
                row[2 * k + 1] = x[2 * k] * e * t[i] * rate;
            }
        }
    }

    void canonical(double* x) const override
    {
        sort_blocks(x, blocks[0], [](const Vec& b) { return b[1]; });
    }

    void start(double* x) const override
    {
        for (int k = 0; k < K; ++k) {                       // SE-3
            x[2 * k] = y_big_ / K;
            x[2 * k + 1] = s_lo_ + (s_hi_ - s_lo_) * ((k + 0.5) / K);
        }
    }

private:
    double s_lo_ = 0.0, s_hi_ = 0.0, y_big_ = 0.0;

    // sum_k a_k exp(-t exp(-s_k)), a and s every stride-th entry
    static double model(const double* a, const double* s, int terms, double t, int stride = 1)
    {
        double sum = 0.0;
        for (int k = 0; k < terms; ++k) sum += a[k * stride] * dm::exp(-t * dm::exp(-s[k * stride]));
        return sum;
    }
};

}   // namespace mootation::suite::sym

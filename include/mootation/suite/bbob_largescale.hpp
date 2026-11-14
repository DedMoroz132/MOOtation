#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// bbob-largescale (task 20, package 24-27, §24.6.5, variant (b): written from
// the papers, no code of COCO's): the 24 bbob functions in any dimension, the
// family "bbob_largescale", one objective, f1..f24 at an instance drawn here.
//
// K. Varelas, O.A. El Hara, D. Brockhoff, N. Hansen, D.M. Nguyen, T. Tusar, A.
// Auger, "Benchmarking large-scale continuous optimizers: the bbob-largescale
// testbed, a COCO software guide and beyond", Applied Soft Computing 97
// (2020) (source benchmarking-large-scale_varelas2020), §4.2-4.6, Tables 1-3,
// Algorithm 1; the same suite in O.A. Elhara, K. Varelas, D.M. Nguyen, T.
// Tusar, D. Brockhoff, N. Hansen, A. Auger, "COCO: the large scale black-box
// optimization benchmarking (bbob-largescale) test suite", arXiv 1903.06396
// (source bbob-largescale_elhara2019_arxiv), §2.2-2.5. The transformations
// T_osz, T_asy, Lambda^alpha, f_pen are bbob's (bbob.hpp, Hansen et al. 2009).
//
// WHAT CHANGES FROM BBOB (Varelas §4.2-4.6):
//   * every rotation R (and Q) is P_left B P_right: B block-diagonal, its
//     blocks Haar rotations of size s = min(n, 40) (the last smaller), P_left
//     and P_right permutations of n_s = n truncated uniform swaps within r_s =
//     floor(n / 3) (Algorithm 1);
//   * gamma(n) = min(1, 40 / n) multiplies the functions the tables say (all
//     but f16-f23's six: Weierstrass, Schaffers F7 twice, Griewank-Rosenbrock,
//     Schwefel, Gallagher twice, Katsuura);
//   * Discus, Bent Cigar, Sharp Ridge: ceil(n / 40) distinguished axes;
//   * the Rosenbrocks (f8, f9) and f19: max(1, sqrt(s) / 8) for sqrt(n) / 8;
//     f8's and f9's x_opt in [-3, 3]^n, f9 z = c R (x - x_opt) + 1.
// At n <= 40 (s = n, gamma = 1, one axis) each function is bbob's with R = B
// up to the permutations, which tests/test_suite.cpp holds to: bbob.hpp's
// instance handed over (of_bbob) gives bbob.hpp's values (LS-3 for f14).
//
// AN INSTANCE (instance below) is drawn by the library's generator keyed
// (20261113, crc32("bbob_largescale"), fid, iid, D): x_opt uniform on
// [-4, 4]^D ([-3, 3]^D for f8 and f9, drawn after the first, as bbob's f8);
// f_opt bbob's Cauchy; R, then Q (each: the blocks in order, P_left, P_right);
// f5's, f20's and f24's signs and their x_opt; f19's x_opt the point where
// z = 1, R^T 1 / (2c); f21's and f22's peaks as bbob.hpp draws them.
//
// READINGS:
//   LS-1. The instances are ours (as bbob's, BBOB-4), not COCO's: COCO's
//     generator is code, not a paper.
//   LS-2. The blocks are our Haar rotations (instances.hpp, Householder QR
//     with R's signs), Varelas's Gram-Schmidt of normal matrices: the same
//     distribution, other numbers.
//   LS-3. Both tables print f14 without bbob's square root: taken as printed;
//     at n <= 40 f14 - f_opt is the square of bbob's.
//   LS-4. "An additional constant is added to the z vector" (§4.6) gives no
//     number beyond Table 1's z = c (x - x_opt) + 1: taken as the table.
//   LS-5. f8's sum runs to n - 1 (Varelas; Elhara's n is a misprint: z_{n+1}
//     does not exist).
//   LS-6. f21's and f22's B ("block-diagonal without permutations"): the
//     instance's R without its permutations.
//   LS-7. Algorithm 1 with 0-based indices: pi a uniform permutation
//     (rnd::shuffle of 0..n-1), the k-th first swap variable i = pi(k), j
//     uniform on {lb..ub} \ {i}, lb = max(0, i - r_s), ub = min(n - 1, i +
//     r_s) (lb + below(ub - lb), stepping over i); no swap where r_s = 0
//     (n < 3). (P v)_i = v[p_i].
//   LS-8. gamma: the tables. Elhara's text leaves Griewank-Rosenbrock out of
//     the exceptions; both tables print f19 without gamma.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bbob.hpp"
#include "common.hpp"
#include "instances.hpp"

namespace mootation::suite::bbob_ls {

using Vec = std::vector<double>;

constexpr int kBlock = 40;                     // s = min(n, 40)

// R = P_left B P_right (Varelas (1)): (P v)_i = v[p_i], B's blocks row-major
// one after another
struct Rotation {
    int n = 0, s = 0;
    std::vector<int> left, right;
    Vec B;

    int blocks() const { return (n + s - 1) / s; }
    int size_of(int b) const { return std::min(s, n - b * s); }

    // out = B v, or B^T v
    void block_product(const double* v, double* out, bool transposed) const
    {
        std::size_t at = 0;
        for (int b = 0, o = 0; b < blocks(); o += size_of(b), ++b) {
            const int k = size_of(b);
            for (int i = 0; i < k; ++i) {
                double sum = 0.0;
                for (int j = 0; j < k; ++j)
                    sum += (transposed ? B[at + static_cast<std::size_t>(j * k + i)]
                                       : B[at + static_cast<std::size_t>(i * k + j)]) * v[o + j];
                out[o + i] = sum;
            }
            at += static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
        }
    }

    // out = R v
    void apply(const double* v, double* out) const
    {
        Vec a(static_cast<std::size_t>(n)), b(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = v[right[static_cast<std::size_t>(i)]];
        block_product(a.data(), b.data(), false);
        for (int i = 0; i < n; ++i) out[i] = b[static_cast<std::size_t>(left[static_cast<std::size_t>(i)])];
    }

    // out = R^T v = P_right^T B^T P_left^T v
    void apply_transposed(const double* v, double* out) const
    {
        Vec a(static_cast<std::size_t>(n)), b(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(left[static_cast<std::size_t>(i)])] = v[i];
        block_product(a.data(), b.data(), true);
        for (int i = 0; i < n; ++i) out[right[static_cast<std::size_t>(i)]] = b[static_cast<std::size_t>(i)];
    }
};

// Algorithm 1 (LS-7): n swaps within floor(n / 3)
inline std::vector<int> truncated_swaps(std::mt19937& g, int n)
{
    std::vector<int> p(static_cast<std::size_t>(n)), pi(static_cast<std::size_t>(n));
    std::iota(p.begin(), p.end(), 0);
    std::iota(pi.begin(), pi.end(), 0);
    rnd::shuffle(pi.begin(), pi.end(), g);
    const int range = n / 3;
    for (int k = 0; k < n; ++k) {
        const int i = pi[static_cast<std::size_t>(k)];
        const int lb = std::max(0, i - range), ub = std::min(n - 1, i + range);
        if (ub == lb) continue;
        int j = lb + static_cast<int>(rnd::below(g, static_cast<std::uint64_t>(ub - lb)));
        if (j >= i) ++j;
        std::swap(p[static_cast<std::size_t>(i)], p[static_cast<std::size_t>(j)]);
    }
    return p;
}

// the blocks in order, then P_left, then P_right
inline Rotation rotation(std::mt19937& g, int n)
{
    Rotation R;
    R.n = n;
    R.s = std::min(n, kBlock);
    for (int b = 0; b < R.blocks(); ++b) {
        const Vec q = instances::haar(g, R.size_of(b));
        R.B.insert(R.B.end(), q.begin(), q.end());
    }
    R.left = truncated_swaps(g, n);
    R.right = truncated_swaps(g, n);
    return R;
}

// a rotation of one block and no permutations: bbob's dense R (the check's)
inline Rotation dense(const Vec& M, int n)
{
    Rotation R;
    R.n = R.s = n;
    R.B = M;
    R.left.resize(static_cast<std::size_t>(n));
    std::iota(R.left.begin(), R.left.end(), 0);
    R.right = R.left;
    return R;
}

// One function at one instance in n variables
struct Instance {
    int fid = 0;
    Vec x_opt;
    double f_opt = 0.0;
    Rotation R, Q;
    Vec l10, lam, sign, w, C, y;                // as bbob::Function's
};

// bbob.hpp's instance in n <= 40 variables as one of this suite: R and Q one
// block, no permutations (the check against bbob, tests/test_suite.cpp)
inline Instance of_bbob(const bbob::Function& f, int n)
{
    Instance in;
    in.fid = f.fid;
    in.x_opt = f.x_opt;
    in.f_opt = f.f_opt;
    if (!f.R.empty()) in.R = dense(f.R, n);
    if (!f.Q.empty()) in.Q = dense(f.Q, n);
    in.l10 = f.l10;
    in.lam = f.lam;
    in.sign = f.sign;
    in.w = f.w;
    in.C = f.C;
    in.y = f.y;
    return in;
}

// max(1, sqrt(s) / 8): f8, f9, f19
inline double rosenbrock_scale(int n) { return std::max(1.0, std::sqrt(static_cast<double>(std::min(n, kBlock))) / 8.0); }

// Function fid at instance iid in n variables (the header's AN INSTANCE)
inline Instance instance(int fid, int iid, int n)
{
    if (fid < 1 || fid > 24 || n < 2 || iid < 1)
        throw std::invalid_argument("suite: bbob_largescale: f1..f24, n >= 2, instance 1 or more");
    std::mt19937 g(instances::seed({20261113u, instances::crc32("bbob_largescale"), static_cast<std::uint64_t>(fid),
                                    static_cast<std::uint64_t>(iid), static_cast<std::uint64_t>(n)}));
    Instance in;
    in.fid = fid;
    in.x_opt = bbob::uniform(g, -4.0, 4.0, n);
    if (fid == 8 || fid == 9) in.x_opt = bbob::uniform(g, -3.0, 3.0, n);
    const double c = rnd::cauchy(g, 0.0, 100.0);
    in.f_opt = clip(std::nearbyint(100.0 * c) / 100.0, -1000.0, 1000.0);
    if (bbob::uses_r(fid)) in.R = rotation(g, n);
    if (bbob::uses_q(fid)) in.Q = rotation(g, n);
    if (bbob::uses_l10(fid)) in.l10 = bbob::lam(10.0, n);
    if (bbob::alpha_of(fid) > 0.0) in.lam = bbob::lam(bbob::alpha_of(fid), n);
    if (bbob::uses_sign(fid)) {
        const double scale = fid == 5 ? 5.0 : fid == 20 ? 0.5 * 4.2096874633 : 0.5 * 2.5;
        for (int i = 0; i < n; ++i) in.sign.push_back(rnd::uniform01(g) < 0.5 ? -1.0 : 1.0);
        for (int i = 0; i < n; ++i) in.x_opt[static_cast<std::size_t>(i)] = scale * in.sign[static_cast<std::size_t>(i)];
    }
    if (fid == 19) {                                         // z = c R x + 1/2 = 1
        const Vec half(static_cast<std::size_t>(n), 0.5 / rosenbrock_scale(n));
        in.R.apply_transposed(half.data(), in.x_opt.data());
    }
    if (fid == 21 || fid == 22) {                            // bbob.hpp's draws
        const int peaks = bbob::peaks_of(fid);
        const double q = peaks - 2.0;
        in.w.push_back(10.0);
        for (int j = 0; j < peaks - 1; ++j) in.w.push_back(1.1 + 8.0 * j / q);
        Vec alpha;
        for (int j = 0; j < peaks - 1; ++j) alpha.push_back(dm::pow(1000.0, 2.0 * j / q));
        rnd::shuffle(alpha.begin(), alpha.end(), g);
        alpha.insert(alpha.begin(), fid == 21 ? 1000.0 : 1000.0 * 1000.0);
        const std::size_t N = static_cast<std::size_t>(n);
        in.C.resize(static_cast<std::size_t>(peaks) * N);
        for (int k = 0; k < peaks; ++k) {
            const double a = alpha[static_cast<std::size_t>(k)];
            double* l = in.C.data() + static_cast<std::size_t>(k) * N;
            for (int i = 0; i < n; ++i) l[i] = dm::pow(a, 0.5 * i / (n - 1.0));
            rnd::shuffle(l, l + n, g);
            const double scale = dm::pow(a, 0.25);
            for (int i = 0; i < n; ++i) l[i] = l[i] / scale;
        }
        const double first = fid == 21 ? 4.0 : 3.92, rest = fid == 21 ? 5.0 : 4.9;
        in.y.resize(static_cast<std::size_t>(peaks) * N);
        for (int k = 0; k < peaks; ++k) {
            const double lo = k == 0 ? -first : -rest, hi = k == 0 ? first : rest;
            double* y = in.y.data() + static_cast<std::size_t>(k) * N;
            for (std::size_t i = 0; i < N; ++i) y[i] = lo + (hi - lo) * rnd::uniform01(g);
        }
        in.x_opt.assign(in.y.begin(), in.y.begin() + n);
    }
    return in;
}

class Objective {
public:
    Objective() = default;

    Objective(Instance in, int n) : in_(std::move(in)), n_(n)
    {
        const std::size_t N = static_cast<std::size_t>(n);
        const int id = in_.fid;
        if (id < 1 || id > 24 || n < 2 || in_.x_opt.size() != N) fail("f1..f24, x_opt of n values");
        if (bbob::uses_r(id) && in_.R.n != n) fail("R of n");
        if (bbob::uses_q(id) && in_.Q.n != n) fail("Q of n");
        gamma_ = std::min(1.0, 40.0 / n);
        axes_ = (n + kBlock - 1) / kBlock;
        c_ = rosenbrock_scale(n);
        for (int i = 0; i < n; ++i) {
            const double t = i / (n - 1.0);
            cond6_.push_back(dm::pow(10.0, 6.0 * t));
            cond2_.push_back(dm::pow(10.0, 2.0 * t));
            powers_.push_back(2.0 + 4.0 * t);
        }
        if (id == 5)
            for (int i = 0; i < n; ++i)
                slope_.push_back(bbob::np_sign(in_.x_opt[static_cast<std::size_t>(i)]) * dm::pow(10.0, i / (n - 1.0)));
        if (id == 16) {
            double three = 1.0, half = 1.0;
            for (int k = 0; k < 12; ++k) {
                half_k_[k] = half;
                wave_k_[k] = bbob::kTwoPi * three;
                f0_ = f0_ + half * dm::cos(wave_k_[k] * 0.5);
                three = 3.0 * three;
                half = 0.5 * half;
            }
        }
        if (id == 23) katsuura_ = 10.0 / dm::pow(static_cast<double>(n), 1.2);
        if (id == 24) {
            lunacek_s_ = 1.0 - 1.0 / (2.0 * std::sqrt(n + 20.0) - 8.2);
            mu1_ = -std::sqrt((2.5 * 2.5 - 1.0) / lunacek_s_);
        }
    }

    const Instance& instance() const { return in_; }

    double operator()(const double* x) const
    {
        Vec d(static_cast<std::size_t>(n_));
        for (int i = 0; i < n_; ++i) d[static_cast<std::size_t>(i)] = x[i] - in_.x_opt[static_cast<std::size_t>(i)];
        const double fo = in_.f_opt, g = gamma_;
        switch (in_.fid) {
            case 1: return g * dot(d, d) + fo;
            case 2: return g * weighted(cond6_, osz(d)) + fo;
            case 3: { const Vec z = scale(in_.l10, asy(osz(d), 0.2)); return g * (rastrigin(z) + dot(z, z)) + fo; }
            case 4: return g * f4_sum(d) + 100.0 * pen(x) + fo;
            case 5: return g * f5_sum(x) + fo;
            case 6: return f6(d) + fo;
            case 7: return f7(x, d) + fo;
            case 8: { Vec z = d; for (double& v : z) v = c_ * v + 1.0; return g * rosenbrock(z) + fo; }
            case 9: { Vec z = rot(in_.R, d); for (double& v : z) v = c_ * v + 1.0; return g * rosenbrock(z) + fo; }
            case 10: return g * weighted(cond6_, osz(rot(in_.R, d))) + fo;
            case 11: { const Vec z = osz(rot(in_.R, d)); return g * (1e6 * head(z) + tail(z)) + fo; }
            case 12: { const Vec z = rot(in_.R, asy(rot(in_.R, d), 0.5)); return g * (head(z) + 1e6 * tail(z)) + fo; }
            case 13: { const Vec z = qlr(in_.l10, d); return g * (head(z) + 100.0 * std::sqrt(tail(z))) + fo; }
            case 14: return g * f14_sum(rot(in_.R, d)) + fo;               // LS-3
            case 15: {
                const Vec z = rot(in_.R, scale(in_.l10, rot(in_.Q, asy(osz(rot(in_.R, d)), 0.2))));
                return g * (rastrigin(z) + dot(z, z)) + fo;
            }
            case 16: return f16(x, d) + fo;
            case 17: return schaffer(x, d, in_.l10) + fo;
            case 18: return schaffer(x, d, in_.lam) + fo;
            case 19: return f19(x) + fo;
            case 20: return f20(x) + fo;
            case 21: case 22: return f21(x) + fo;
            case 23: return f23(x, d) + fo;
            default: return f24(x) + fo;
        }
    }

private:
    Instance in_;
    int n_ = 0;
    double gamma_ = 1.0, c_ = 1.0, f0_ = 0.0, katsuura_ = 0.0, lunacek_s_ = 0.0, mu1_ = 0.0;
    int axes_ = 1;
    Vec cond6_, cond2_, powers_, slope_;
    double half_k_[12] = {}, wave_k_[12] = {};

    [[noreturn]] void fail(const std::string& why) const
    {
        throw std::invalid_argument("suite: bbob_largescale f" + std::to_string(in_.fid) + ": " + why);
    }

    std::size_t N() const { return static_cast<std::size_t>(n_); }
    static double dot(const Vec& a, const Vec& b)
    {
        Vec t(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) t[i] = a[i] * b[i];
        return row_sum(t);
    }
    Vec rot(const Rotation& R, const Vec& v) const { Vec out(N()); R.apply(v.data(), out.data()); return out; }
    static Vec scale(const Vec& l, Vec v) { for (std::size_t i = 0; i < v.size(); ++i) v[i] = l[i] * v[i]; return v; }
    static Vec osz(Vec v) { for (double& e : v) e = bbob::t_osz(e); return v; }
    Vec asy(const Vec& v, double beta) const { Vec out(N()); bbob::t_asy(v.data(), n_, beta, out.data()); return out; }
    // Q Lambda R d
    Vec qlr(const Vec& l, const Vec& d) const { return rot(in_.Q, scale(l, rot(in_.R, d))); }

    double weighted(const Vec& c, const Vec& z) const
    {
        Vec t(N());
        for (std::size_t i = 0; i < N(); ++i) t[i] = c[i] * z[i] * z[i];
        return row_sum(t);
    }
    // the sums of z_i^2 over the first ceil(n / 40) axes and over the rest
    double head(const Vec& z) const
    {
        Vec t(z.begin(), z.begin() + axes_);
        for (double& v : t) v = v * v;
        return row_sum(t);
    }
    double tail(const Vec& z) const
    {
        Vec t(z.begin() + axes_, z.end());
        for (double& v : t) v = v * v;
        return row_sum(t);
    }
    double pen(const double* x) const
    {
        Vec t(N());
        for (std::size_t i = 0; i < N(); ++i) {
            const double m = bbob::np_maximum(0.0, std::abs(x[i]) - 5.0);
            t[i] = m * m;
        }
        return row_sum(t);
    }
    double rastrigin(const Vec& z) const
    {
        Vec c(N());
        for (std::size_t i = 0; i < N(); ++i) c[i] = dm::cos(bbob::kTwoPi * z[i]);
        return 10.0 * (n_ - row_sum(c));
    }
    double rosenbrock(const Vec& z) const
    {
        Vec t(N() - 1);
        for (std::size_t i = 0; i + 1 < N(); ++i) {
            const double a = z[i] * z[i] - z[i + 1], b = z[i] - 1.0;
            t[i] = 100.0 * a * a + b * b;
        }
        return row_sum(t);                                              // LS-5
    }

    double f4_sum(const Vec& d) const
    {
        Vec z(N()), t(N());
        for (std::size_t i = 0; i < N(); ++i) {
            const double v = bbob::t_osz(d[i]), l = in_.l10[i];
            z[i] = (v > 0.0 && i % 2 == 0 ? 10.0 * l : l) * v;
            t[i] = z[i] * z[i];
        }
        return rastrigin(z) + row_sum(t);
    }

    double f5_sum(const double* x) const
    {
        Vec t(N());
        for (std::size_t i = 0; i < N(); ++i) {
            const double xo = in_.x_opt[i], s = slope_[i];
            const double z = xo * x[i] < 25.0 ? x[i] : xo;
            t[i] = 5.0 * std::abs(s) - s * z;
        }
        return row_sum(t);
    }

    double f6(const Vec& d) const
    {
        const Vec z = qlr(in_.l10, d);
        Vec t(N());
        for (std::size_t i = 0; i < N(); ++i) {
            const double v = (z[i] * in_.x_opt[i] > 0.0 ? 100.0 : 1.0) * z[i];
            t[i] = v * v;
        }
        return dm::pow(bbob::t_osz(gamma_ * row_sum(t)), 0.9);
    }

    double f7(const double* x, const Vec& d) const
    {
        const Vec zh = scale(in_.l10, rot(in_.R, d));
        Vec zt(N());
        for (std::size_t i = 0; i < N(); ++i)
            zt[i] = std::abs(zh[i]) > 0.5 ? std::floor(0.5 + zh[i]) : std::floor(0.5 + 10.0 * zh[i]) / 10.0;
        const Vec z = rot(in_.Q, zt);
        return gamma_ * 0.1 * bbob::np_maximum(std::abs(zh[0]) / 1e4, weighted(cond2_, z)) + pen(x);
    }

    double f14_sum(const Vec& z) const
    {
        Vec t(N());
        for (std::size_t i = 0; i < N(); ++i) t[i] = dm::pow(std::abs(z[i]), powers_[i]);
        return row_sum(t);
    }

    double f16(const double* x, const Vec& d) const
    {
        const Vec z = rot(in_.R, scale(in_.lam, rot(in_.Q, osz(rot(in_.R, d)))));
        Vec s(N());
        for (std::size_t i = 0; i < N(); ++i) {
            double acc = 0.0;
            for (int k = 0; k < 12; ++k) acc = acc + half_k_[k] * dm::cos(wave_k_[k] * (z[i] + 0.5));
            s[i] = acc;
        }
        const double t = row_sum(s) / n_ - f0_;
        return 10.0 * (t * t * t) + (10.0 / n_) * pen(x);
    }

    double schaffer(const double* x, const Vec& d, const Vec& l) const
    {
        const Vec z = scale(l, rot(in_.Q, asy(rot(in_.R, d), 0.5)));
        Vec t(N() - 1);
        for (std::size_t i = 0; i + 1 < N(); ++i) {
            const double s = std::sqrt(z[i] * z[i] + z[i + 1] * z[i + 1]), rt = std::sqrt(s);
            const double sn = dm::sin(50.0 * dm::pow(s, 0.2));
            t[i] = rt + rt * (sn * sn);
        }
        const double inner = row_sum(t) / (n_ - 1.0);
        return inner * inner + 10.0 * pen(x);
    }

    double f19(const double* x) const
    {
        Vec z = rot(in_.R, Vec(x, x + n_)), t(N() - 1);
        for (double& v : z) v = c_ * v + 0.5;
        for (std::size_t i = 0; i + 1 < N(); ++i) {
            const double a = z[i] * z[i] - z[i + 1], b = z[i] - 1.0, s = 100.0 * a * a + b * b;
            t[i] = s / 4000.0 - dm::cos(s);
        }
        return 10.0 / (n_ - 1.0) * row_sum(t) + 10.0;
    }

    double f20(const double* x) const
    {
        Vec two_abs(N()), xh(N()), z(N()), t(N());
        for (std::size_t i = 0; i < N(); ++i) {
            two_abs[i] = 2.0 * std::abs(in_.x_opt[i]);
            xh[i] = 2.0 * in_.sign[i] * x[i];
        }
        for (std::size_t i = 0; i < N(); ++i) {
            const double zh = i == 0 ? xh[0] : xh[i] + 0.25 * (xh[i - 1] - two_abs[i - 1]);
            z[i] = 100.0 * (in_.l10[i] * (zh - two_abs[i]) + two_abs[i]);
        }
        for (std::size_t i = 0; i < N(); ++i) t[i] = z[i] * dm::sin(std::sqrt(std::abs(z[i])));
        const double term = -row_sum(t) / (100.0 * n_);
        for (double& v : z) v = v / 100.0;
        return term + bbob::kSchwefel + 100.0 * pen(z.data());
    }

    // B (x - y_k) without the permutations (LS-6)
    double f21(const double* x) const
    {
        Vec d(N()), u(N()), q(N());
        double best = 0.0;
        const int peaks = bbob::peaks_of(in_.fid);
        for (int k = 0; k < peaks; ++k) {
            const double* y = in_.y.data() + static_cast<std::size_t>(k) * N();
            const double* C = in_.C.data() + static_cast<std::size_t>(k) * N();
            for (std::size_t i = 0; i < N(); ++i) d[i] = x[i] - y[i];
            in_.R.block_product(d.data(), u.data(), false);
            for (std::size_t i = 0; i < N(); ++i) q[i] = C[i] * u[i] * u[i];
            const double v = in_.w[static_cast<std::size_t>(k)] * dm::exp(-row_sum(q) / (2.0 * n_));
            best = k == 0 ? v : bbob::np_maximum(best, v);
        }
        return bbob::py_square(bbob::t_osz(10.0 - best)) + pen(x);
    }

    double f23(const double* x, const Vec& d) const
    {
        const Vec z = qlr(in_.lam, d);
        Vec p(N());
        for (std::size_t i = 0; i < N(); ++i) {
            double s = 0.0, two = 1.0;
            for (int j = 1; j <= 32; ++j) {
                two = 2.0 * two;
                const double v = two * z[i];
                s = s + std::abs(v - std::nearbyint(v)) / two;
            }
            p[i] = dm::pow(1.0 + (static_cast<double>(i) + 1.0) * s, katsuura_);
        }
        const double k = 10.0 / (static_cast<double>(n_) * n_);
        return k * row_prod(p.data(), N()) - k + pen(x);
    }

    double f24(const double* x) const
    {
        Vec a(N()), aa(N()), bb(N());
        for (std::size_t i = 0; i < N(); ++i) {
            const double xh = 2.0 * in_.sign[i] * x[i];
            a[i] = xh - 2.5;
            aa[i] = a[i] * a[i];
            bb[i] = (xh - mu1_) * (xh - mu1_);
        }
        const double funnel = std::min(row_sum(aa), 1.0 * n_ + lunacek_s_ * row_sum(bb));
        const Vec z = qlr(in_.lam, a);
        return gamma_ * (funnel + rastrigin(z)) + 1e4 * pen(x);
    }
};

}   // namespace mootation::suite::bbob_ls

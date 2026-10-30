#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// bbob-biobj and bbob-biobj-mixint (task 12 of 2026-10-19, §1): the ten bbob
// functions of benchmarks/bbob.py, paired as benchmarks/bbob_biobj.py pairs
// them, and the integer variables of benchmarks/bbob_biobj_mixint.py, with the
// values those modules compute (to a few ulps: the elementary functions here
// are the library's own, dm::, those of the modules the C library's).
//
// bbob itself (task 18, stage 2): all 24 functions, one objective, in 2 to 40
// variables (Objective; the suite's family "bbob"). The fourteen the pairs do
// not use (f3-f5, f7, f9-f12, f16, f18, f19, f22-f24) are written from the
// report as the ten were, in the same manner, and bbob.py repeats them as the
// cross-check. Readings of the report for them (bbob.py's BBOB-5..8):
//   BBOB-5. f9 and f19 draw no x_opt: z = max(1, sqrt(D)/8) R x + 1/2 with
//     z_opt = 1, so x_opt = R^T 1 / (2 max(1, sqrt(D)/8)); f(x_opt) is f_opt
//     to rounding, not exactly.
//   BBOB-6. f23's [.] (the nearest integer) rounds half to even.
//   BBOB-7. f22 as printed after the errata: alpha_1 = 1000^2, y_1 on
//     [-3.92, 3.92]^D, the other peaks on [-4.9, 4.9]^D.
//   BBOB-8. f16's inner sum runs over k = 0..11 in order, each term
//     2^-k cos(2 pi 3^k (z_i + 1/2)) with 2 pi 3^k taken first; f0 the same at
//     z_i = 0. f23's over j = 1..32 in order.
//
// N. Hansen, S. Finck, R. Ros, A. Auger. "Real-Parameter Black-Box
// Optimization Benchmarking 2009: Noiseless Functions Definitions." INRIA
// RR-6829, with the 2019 errata: the functions (Sections 1-5) and the
// transformations T_osz, T_asy, Lambda^alpha and f_pen (Section 0.2).
// D. Brockhoff, A. Auger, N. Hansen, T. Tusar. "Using Well-Understood
// Single-Objective Functions in Multiobjective Black-Box Optimization Test
// Suites." Evolutionary Computation 30(2):165-193, 2022: F = (f_alpha, f_beta).
// T. Tusar, D. Brockhoff, N. Hansen. "Mixed-Integer Benchmark Problems for
// Single- and Bi-Objective Optimization." GECCO 2019, Section 3.1.1: the
// integer j of a variable stands for the level z_{j+1}.
//
// WHAT AN INSTANCE DRAWS (task 17, supplement 5, §3): x_opt, f_opt, the
// rotations R and Q, f20's signs, f21's peaks — drawn here (instance, pair
// below), by the library's generator keyed (20260916, fid, iid, D), in
// bbob.py's order, draw for draw (bbob.py draws the same through
// library_rng.py, the cross-check's; the fourteen of task 18 in the same
// order: x_opt, f_opt, R, Q, then the signs or the peaks, the draws a
// function does not use left out): Hansen et al.'s distributions (Section
// 0.1), not COCO's generator, which no paper states — the instances are ours
// and are not COCO's (BBOB-4), and from 2026-10-24 (version 2) not the ones
// NumPy's default_rng drew before either. A spec may still hand the numbers
// over (`bbob`). Python keeps the problem's frame, the ideal and the nadir.
// The readings BBOB-1..4, BIOBJ-1..2 and MIXINT-1..3 are those of the Python
// modules.
//
// THE ORDER OF THE ARITHMETIC is that of the NumPy code, as in the rest of the
// suite (common.hpp): sums along a row pairwise, x ** 2 of an array x * x,
// log, exp, sin, cos and pow the library's own (dm::, detmath.hpp; task 14 of
// 2026-10-20), and the grouping of every Python expression kept. Python's
// v ** 2 of a float (pow_each) is pow of |v| (py_square), not v * v.
//
// THE PRODUCTS OF VECTORS AND MATRICES. bbob.py takes R x, x . x and the
// peaks' (x - y_i) R^T through NumPy's @, that is through the BLAS NumPy is
// built with, and a BLAS adds in an order of its own, with fused
// multiply-adds. Its values are kept: the order of OpenBLAS 0.3.34's SkylakeX
// kernels (the scipy-openblas of NumPy 2.5), read off NumPy's results and
// checked against them for every D from 2 to 40 (gemv, gemm) and every length
// up to 15 (ddot), is written out below (dot, matvec, gemm_row). A machine
// whose NumPy has another BLAS or kernel gives other last bits in the Python
// code, as it always did; this code gives the same values everywhere. The
// sizes the orders were read off for bound bbob-biobj's D to 2..15; bbob's
// single functions in 16 to 40 variables keep the same orders, whose ddot
// NumPy then takes from another kernel (the cross-check's tolerance).
//
// Built without contraction into fused multiply-adds (suite_module.cpp):
// every fused multiply-add here is explicit (fused), rounded once, whatever
// the machine.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../detmath.hpp"
#include "../random.hpp"
#include "common.hpp"
#include "instances.hpp"

#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#endif

namespace mootation::suite {

namespace bbob {

constexpr int kMinD = 2;
constexpr int kMaxD = 15;                       // bbob-biobj: the sizes the BLAS orders were read off for
constexpr int kMaxSoloD = 40;                   // bbob: n = 2, 3, 5, 10, 20, 40
constexpr int kPeaks = 101;                     // f21 (f22: 21)
constexpr double kTwoPi = 2.0 * kPi;            // 2.0 * math.pi, as Python folds it first
constexpr double kSchwefel = 4.189828872724339; // f20's constant

// the peaks of f21 and f22
inline int peaks_of(int fid) { return fid == 22 ? 21 : kPeaks; }

// whether function fid takes R, Q, Lambda^10, another Lambda^alpha, the signs
inline bool uses_r(int id) { return !(id <= 5 || id == 8 || id == 20); }
inline bool uses_q(int id)
{
    return id == 6 || id == 7 || id == 13 || id == 15 || id == 16 || id == 17 || id == 18 || id == 23 || id == 24;
}
inline bool uses_l10(int id) { return id == 3 || id == 4 || id == 6 || id == 7 || id == 13 || id == 15 || id == 17 || id == 20; }
inline double alpha_of(int id) { return id == 16 ? 0.01 : id == 18 ? 1000.0 : (id == 23 || id == 24) ? 100.0 : 0.0; }
inline bool uses_sign(int id) { return id == 5 || id == 20 || id == 24; }

// One bbob function at one instance in D variables: what bbob.py's Instance
// holds, and for bbob-biobj-mixint the levels its integers stand for.
struct Function {
    int fid = 0;                                // 1 .. 24 (bbob-biobj: 1, 2, 6, 8, 13, 14, 15, 17, 20 or 21)
    std::vector<double> x_opt;                  // D
    double f_opt = 0.0;
    std::vector<double> R, Q;                   // D x D, row-major (uses_r, uses_q)
    std::vector<double> l10;                    // the diagonal of Lambda^10 (uses_l10)
    std::vector<double> lam;                    // the diagonal of Lambda^alpha, alpha_of: f16, f18, f23, f24
    std::vector<double> sign;                   // f5, f20, f24: the vector 1+-
    std::vector<double> w, C, y;                // f21, f22: the weights, the peaks x D diagonals of C_i, the peaks y_i
    std::vector<std::vector<double>> levels;    // mixint: z_1 .. z_l of each integer variable, in this objective
};

// ---- a fused multiply-add ----------------------------------------------------
// a * b + c rounded once, as IEEE 754 defines it and the BLAS kernels'
// instruction computes it. std::fma gives that value on any machine, but MSVC
// compiles it to a call into the C library; on x86-64 with MSVC, therefore,
// the instruction itself where the CPU has it (FMA3, and the system saving
// the AVX registers; asked once). The same value either way
// (tests/test_suite.cpp holds the two equal).
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
inline bool cpu_has_fma()
{
    int r[4];
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    return fma && osxsave && avx && (_xgetbv(0) & 6) == 6;
}

inline const bool kFmaInstruction = cpu_has_fma();

inline double fused(double a, double b, double c)
{
    if (kFmaInstruction) return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
    return std::fma(a, b, c);
}
#else
inline double fused(double a, double b, double c) { return std::fma(a, b, c); }
#endif

// ---- the products in OpenBLAS's order (THE PRODUCTS) ------------------------

// x . y of n <= 15 terms (ddot): fused multiply-adds one after another from 0.
inline double dot(const double* x, const double* y, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s = fused(x[i], y[i], s);
    return 0.0 + s;
}

// One row of M x (dgemv; NumPy hands the row-major M to it transposed). The
// rows go in blocks of four, then a pair, then a single row, each with a
// kernel of its own. Of a row's D terms, the first 4 floor(D / 4) go into
// lanes, term i into lane i mod L, from 0:
//   a row of a block of four: four lanes of fused multiply-adds, summed
//     (l0 + l2) + (l1 + l3);
//   a row of the pair: two lanes, each term a product, then a sum: l0 + l1;
//   the single row: four such lanes, (l0 + l2) + (l1 + l3);
// and that sum lands on the zero the result starts from. The D mod 4 terms
// left over (t, u, v) come after: one, fma(m_t, x_t, s); two,
// s + fma(m_t, x_t, m_u x_u); three, s + fma(m_v, x_v, fma(m_t, x_t, m_u x_u)).
inline double matvec_row(const double* m, const double* x, int D, int row)
{
    const int main = D - D % 4, left = D % 4;
    const int blocks = 4 * (D / 4), pair = (D % 4) / 2 * 2;
    double s = 0.0;
    if (row < blocks) {
        double l[4] = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < main; ++i) l[i % 4] = fused(m[i], x[i], l[i % 4]);
        s = (l[0] + l[2]) + (l[1] + l[3]);
    } else if (row < blocks + pair) {
        double l[2] = {0.0, 0.0};
        for (int i = 0; i < main; ++i) l[i % 2] = l[i % 2] + m[i] * x[i];
        s = l[0] + l[1];
    } else {
        double l[4] = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < main; ++i) l[i % 4] = l[i % 4] + m[i] * x[i];
        s = (l[0] + l[2]) + (l[1] + l[3]);
    }
    s = 0.0 + s;
    const int t = main;
    if (left == 1) return fused(m[t], x[t], s);
    if (left == 2) return s + fused(m[t], x[t], m[t + 1] * x[t + 1]);
    if (left == 3) return s + fused(m[t + 2], x[t + 2], fused(m[t], x[t], m[t + 1] * x[t + 1]));
    return s;
}

// M x of a D x D row-major M (rows_matvec's M @ x).
inline void matvec(const std::vector<double>& M, int D, const double* x, double* out)
{
    for (int r = 0; r < D; ++r) out[r] = matvec_row(M.data() + static_cast<std::size_t>(r) * D, x, D, r);
}

// One row of (x - y) R^T (dgemm): entry j the fused multiply-adds of
// (x - y)_k R_jk one after another over k, from 0.
inline void gemm_row(const double* d, const std::vector<double>& R, int D, double* out)
{
    for (int j = 0; j < D; ++j) {
        const double* r = R.data() + static_cast<std::size_t>(j) * D;
        double s = 0.0;
        for (int k = 0; k < D; ++k) s = fused(d[k], r[k], s);
        out[j] = s;
    }
}

// ---- NumPy's scalar loops and Python's power --------------------------------

// np.sign: 0 at either zero, NaN stays NaN
inline double np_sign(double x)
{
    return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : x == 0.0 ? 0.0 : x;
}

// np.maximum(a, b): a NaN wins
inline double np_maximum(double a, double b)
{
    return (a >= b || std::isnan(a)) ? a : b;
}

// Python's v ** 2 of a float (bbob.py's pow_each): CPython takes the sign off
// a negative base itself, then calls pow (here dm::pow, which gives |v| |v|)
inline double py_square(double v)
{
    if (std::isnan(v)) return v;
    return dm::pow(std::abs(v), 2.0);
}

// ---- the transformations (Hansen et al., Section 0.2) -----------------------

// T_osz of one value: sign(x) exp(x^ + 0.049 (sin(c1 x^) + sin(c2 x^)))
inline double t_osz(double x)
{
    const double xh = x != 0.0 ? dm::log(std::abs(x)) : 0.0;
    const double c1 = x > 0.0 ? 10.0 : 5.5;
    const double c2 = x > 0.0 ? 7.9 : 3.1;
    return np_sign(x) * dm::exp(xh + 0.049 * (dm::sin(c1 * xh) + dm::sin(c2 * xh)));
}

// T_asy^beta of a D-vector: x_i^(1 + beta (i-1)/(D-1) sqrt(x_i)) where x_i > 0
inline void t_asy(const double* x, int D, double beta, double* out)
{
    for (int i = 0; i < D; ++i) {
        const double e = 1.0 + beta * (i / (D - 1.0)) * std::sqrt(std::abs(x[i]));
        out[i] = x[i] > 0.0 ? dm::pow(std::abs(x[i]), e) : x[i];
    }
}

// f_pen: sum_i max(0, |x_i| - 5)^2
inline double f_pen(const double* x, int D)
{
    double t[kMaxSoloD] = {};
    for (int i = 0; i < D; ++i) {
        const double m = np_maximum(0.0, std::abs(x[i]) - 5.0);
        t[i] = m * m;
    }
    return row_sum(t, static_cast<std::size_t>(D));
}

// ---- the 24 functions (bbob.py's _f1 .. _f24, one point) --------------------

class Objective {
public:
    Objective() = default;

    Objective(Function f, int D) : f_(std::move(f)), D_(D)
    {
        check();
        prepare();
    }

    const Function& function() const { return f_; }

    // the value at x (D values, real)
    double operator()(const double* x) const
    {
        double d[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) d[i] = x[i] - f_.x_opt[static_cast<std::size_t>(i)];
        switch (f_.fid) {
            case 1: return dot(d, d, D_) + f_.f_opt;
            case 2: return f2(d);
            case 3: return f3(d);
            case 4: return f4(x, d);
            case 5: return f5(x);
            case 6: return f6(d);
            case 7: return f7(x, d);
            case 8: return f8(d);
            case 9: return f9(x);
            case 10: return f10(d);
            case 11: return f11(d);
            case 12: return f12(d);
            case 13: return f13(d);
            case 14: return f14(d);
            case 15: return f15(d);
            case 16: return f16(x, d);
            case 17: return f17(x, d, f_.l10);
            case 18: return f17(x, d, f_.lam);
            case 19: return f19(x);
            case 20: return f20(x);
            case 21: case 22: return f21(x);
            case 23: return f23(x, d);
            default: return f24(x);
        }
    }

private:
    Function f_;
    int D_ = 0;
    std::vector<double> cond_;                  // f2, f10: 10^(6 (i-1)/(D-1)); f7: 10^(2 (i-1)/(D-1));
                                                // f14: the exponents 2 + 4 (i-1)/(D-1); f5: s_i
    std::vector<double> two_abs_;               // f20: 2 |x_opt| (BBOB-2)
    double c_ = 0.0;                            // f9, f19: max(1, sqrt(D)/8); f23: 10/D^1.2; f24: s
    double mu1_ = 0.0;                          // f24
    double f0_ = 0.0;                           // f16
    double half_k_[12] = {}, wave_k_[12] = {};  // f16: 2^-k and 2 pi 3^k

    void fail(const std::string& why) const
    {
        throw std::invalid_argument("suite: bbob f" + std::to_string(f_.fid) + ": " + why);
    }

    // the sizes the function reads, before any evaluation
    void check() const
    {
        const std::size_t n = static_cast<std::size_t>(D_);
        if (D_ < kMinD || D_ > kMaxSoloD) fail("D from 2 to 40");
        if (f_.x_opt.size() != n) fail("x_opt of D values");
        const int id = f_.fid;
        if (id < 1 || id > 24) fail("not one of the 24");
        if (uses_r(id) && f_.R.size() != n * n) fail("R of D x D");
        if (uses_q(id) && f_.Q.size() != n * n) fail("Q of D x D");
        if (uses_l10(id) && f_.l10.size() != n) fail("Lambda^10 of D values");
        if (alpha_of(id) > 0.0 && f_.lam.size() != n) fail("Lambda^alpha of D values");
        if (uses_sign(id) && f_.sign.size() != n) fail("the signs of D values");
        const std::size_t peaks = static_cast<std::size_t>(peaks_of(id));
        if ((id == 21 || id == 22) && (f_.w.size() != peaks || f_.C.size() != peaks * n || f_.y.size() != peaks * n))
            fail(std::to_string(peaks) + " peaks");
    }

    // the constants bbob.py computes in every call, once
    void prepare()
    {
        const int D = D_;
        if (f_.fid == 2 || f_.fid == 10)
            for (int i = 0; i < D; ++i) cond_.push_back(dm::pow(10.0, 6.0 * i / (D - 1.0)));
        if (f_.fid == 7)
            for (int i = 0; i < D; ++i) cond_.push_back(dm::pow(10.0, 2.0 * i / (D - 1.0)));
        if (f_.fid == 14)
            for (int i = 0; i < D; ++i) cond_.push_back(2.0 + 4.0 * i / (D - 1.0));
        if (f_.fid == 5)
            for (int i = 0; i < D; ++i)
                cond_.push_back(np_sign(f_.x_opt[static_cast<std::size_t>(i)]) * dm::pow(10.0, i / (D - 1.0)));
        if (f_.fid == 20)
            for (double v : f_.x_opt) two_abs_.push_back(2.0 * std::abs(v));
        if (f_.fid == 9 || f_.fid == 19) c_ = std::max(1.0, std::sqrt(static_cast<double>(D)) / 8.0);
        if (f_.fid == 23) c_ = 10.0 / dm::pow(static_cast<double>(D), 1.2);
        if (f_.fid == 24) {
            c_ = 1.0 - 1.0 / (2.0 * std::sqrt(D + 20.0) - 8.2);
            mu1_ = -std::sqrt((2.5 * 2.5 - 1.0) / c_);
        }
        if (f_.fid == 16) {
            double three = 1.0, half = 1.0;
            for (int k = 0; k < 12; ++k) {
                half_k_[k] = half;
                wave_k_[k] = kTwoPi * three;
                f0_ = f0_ + half * dm::cos(wave_k_[k] * 0.5);
                three = 3.0 * three;
                half = 0.5 * half;
            }
        }
    }

    // 10 (D - sum cos(2 pi z_i)): f3, f4, f15, f24's Rastrigin part
    double rastrigin(const double* z) const
    {
        double c[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) c[i] = dm::cos(kTwoPi * z[i]);
        return 10.0 * (D_ - row_sum(c, static_cast<std::size_t>(D_)));
    }

    // sum_i 100 (z_i^2 - z_{i+1})^2 + (z_i - 1)^2: f8, f9's Rosenbrock
    double rosenbrock(const double* z) const
    {
        double t[kMaxSoloD] = {};
        for (int i = 0; i < D_ - 1; ++i) {
            const double a = z[i] * z[i] - z[i + 1];
            const double b = z[i] - 1.0;
            t[i] = 100.0 * a * a + b * b;
        }
        return row_sum(t, static_cast<std::size_t>(D_ - 1));
    }

    // Q Lambda^10 R d
    void qlr(const double* d, double* z) const
    {
        double a[kMaxSoloD] = {};
        matvec(f_.R, D_, d, a);
        for (int i = 0; i < D_; ++i) a[i] = f_.l10[static_cast<std::size_t>(i)] * a[i];
        matvec(f_.Q, D_, a, z);
    }

    double f2(const double* d) const
    {
        double t[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) {
            const double z = t_osz(d[i]);
            t[i] = cond_[static_cast<std::size_t>(i)] * z * z;
        }
        return row_sum(t, static_cast<std::size_t>(D_)) + f_.f_opt;
    }

    double f6(const double* d) const
    {
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        qlr(d, z);
        for (int i = 0; i < D_; ++i) {
            const double s = z[i] * f_.x_opt[static_cast<std::size_t>(i)] > 0.0 ? 100.0 : 1.0;
            const double v = s * z[i];
            t[i] = v * v;
        }
        return dm::pow(t_osz(row_sum(t, static_cast<std::size_t>(D_))), 0.9) + f_.f_opt;
    }

    double f8(const double* d) const
    {
        const double c = std::max(1.0, std::sqrt(static_cast<double>(D_)) / 8.0);
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) z[i] = c * d[i] + 1.0;
        for (int i = 0; i < D_ - 1; ++i) {
            const double a = z[i] * z[i] - z[i + 1];
            const double b = z[i] - 1.0;
            t[i] = 100.0 * a * a + b * b;
        }
        return row_sum(t, static_cast<std::size_t>(D_ - 1)) + f_.f_opt;
    }

    double f13(const double* d) const
    {
        double z[kMaxSoloD] = {};
        qlr(d, z);
        return py_square(z[0]) + 100.0 * std::sqrt(dot(z + 1, z + 1, D_ - 1)) + f_.f_opt;
    }

    double f14(const double* d) const
    {
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, d, z);
        for (int i = 0; i < D_; ++i) t[i] = dm::pow(std::abs(z[i]), cond_[static_cast<std::size_t>(i)]);
        return std::sqrt(row_sum(t, static_cast<std::size_t>(D_))) + f_.f_opt;
    }

    double f15(const double* d) const
    {
        double a[kMaxSoloD] = {}, b[kMaxSoloD] = {}, z[kMaxSoloD] = {}, c[kMaxSoloD] = {};
        matvec(f_.R, D_, d, a);
        for (int i = 0; i < D_; ++i) a[i] = t_osz(a[i]);
        t_asy(a, D_, 0.2, b);
        matvec(f_.Q, D_, b, a);
        for (int i = 0; i < D_; ++i) a[i] = f_.l10[static_cast<std::size_t>(i)] * a[i];
        matvec(f_.R, D_, a, z);
        for (int i = 0; i < D_; ++i) c[i] = dm::cos(kTwoPi * z[i]);
        return (10.0 * (D_ - row_sum(c, static_cast<std::size_t>(D_))) + dot(z, z, D_)) + f_.f_opt;
    }

    // f17 with Lambda^10, f18 with Lambda^1000
    double f17(const double* x, const double* d, const std::vector<double>& lam) const
    {
        double a[kMaxSoloD] = {}, b[kMaxSoloD] = {}, z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, d, a);
        t_asy(a, D_, 0.5, b);
        matvec(f_.Q, D_, b, a);
        for (int i = 0; i < D_; ++i) z[i] = lam[static_cast<std::size_t>(i)] * a[i];
        for (int i = 0; i < D_ - 1; ++i) {                    // BBOB-3: i = 1 .. D-1
            const double s = std::sqrt(z[i] * z[i] + z[i + 1] * z[i + 1]);
            const double rt = std::sqrt(s);
            const double sn = dm::sin(50.0 * dm::pow(s, 0.2));
            t[i] = rt + rt * (sn * sn);
        }
        const double inner = row_sum(t, static_cast<std::size_t>(D_ - 1)) / (D_ - 1.0);
        return inner * inner + 10.0 * f_pen(x, D_) + f_.f_opt;
    }

    double f20(const double* x) const
    {
        double xh[kMaxSoloD] = {}, z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) xh[i] = 2.0 * f_.sign[static_cast<std::size_t>(i)] * x[i];
        for (int i = 0; i < D_; ++i) {
            const double two_abs = two_abs_[static_cast<std::size_t>(i)];
            const double zh = i == 0 ? xh[0] : xh[i] + 0.25 * (xh[i - 1] - two_abs_[static_cast<std::size_t>(i - 1)]);
            z[i] = 100.0 * (f_.l10[static_cast<std::size_t>(i)] * (zh - two_abs) + two_abs);
        }
        for (int i = 0; i < D_; ++i) t[i] = z[i] * dm::sin(std::sqrt(std::abs(z[i])));
        const double term = -row_sum(t, static_cast<std::size_t>(D_)) / (100.0 * D_);
        for (int i = 0; i < D_; ++i) z[i] = z[i] / 100.0;
        return term + kSchwefel + 100.0 * f_pen(z, D_) + f_.f_opt;
    }

    // f21 with 101 peaks, f22 with 21
    double f21(const double* x) const
    {
        double d[kMaxSoloD] = {}, u[kMaxSoloD] = {}, q[kMaxSoloD] = {};
        double best = 0.0;
        const int peaks = peaks_of(f_.fid);
        for (int k = 0; k < peaks; ++k) {
            const double* y = f_.y.data() + static_cast<std::size_t>(k) * D_;
            const double* C = f_.C.data() + static_cast<std::size_t>(k) * D_;
            for (int i = 0; i < D_; ++i) d[i] = x[i] - y[i];
            gemm_row(d, f_.R, D_, u);
            for (int i = 0; i < D_; ++i) q[i] = C[i] * u[i] * u[i];
            const double quad = row_sum(q, static_cast<std::size_t>(D_));
            const double v = f_.w[static_cast<std::size_t>(k)] * dm::exp(-quad / (2.0 * D_));
            best = k == 0 ? v : np_maximum(best, v);
        }
        return py_square(t_osz(10.0 - best)) + f_pen(x, D_) + f_.f_opt;
    }

    // ---- the fourteen of task 18 ----

    double f3(const double* d) const
    {
        double a[kMaxSoloD] = {}, z[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) a[i] = t_osz(d[i]);
        t_asy(a, D_, 0.2, z);
        for (int i = 0; i < D_; ++i) z[i] = f_.l10[static_cast<std::size_t>(i)] * z[i];
        return (rastrigin(z) + dot(z, z, D_)) + f_.f_opt;
    }

    // s_i = 10 lambda_ii where z_i > 0 at the odd i (1-based), lambda_ii else
    double f4(const double* x, const double* d) const
    {
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) {
            const double v = t_osz(d[i]);
            const double l = f_.l10[static_cast<std::size_t>(i)];
            z[i] = (v > 0.0 && i % 2 == 0 ? 10.0 * l : l) * v;
            t[i] = z[i] * z[i];
        }
        return rastrigin(z) + row_sum(t, static_cast<std::size_t>(D_)) + 100.0 * f_pen(x, D_) + f_.f_opt;
    }

    double f5(const double* x) const
    {
        double t[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) {
            const double xo = f_.x_opt[static_cast<std::size_t>(i)], s = cond_[static_cast<std::size_t>(i)];
            const double z = xo * x[i] < 25.0 ? x[i] : xo;
            t[i] = 5.0 * std::abs(s) - s * z;
        }
        return row_sum(t, static_cast<std::size_t>(D_)) + f_.f_opt;
    }

    double f7(const double* x, const double* d) const
    {
        double zh[kMaxSoloD] = {}, zt[kMaxSoloD] = {}, z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, d, zh);
        for (int i = 0; i < D_; ++i) {
            zh[i] = f_.l10[static_cast<std::size_t>(i)] * zh[i];
            zt[i] = std::abs(zh[i]) > 0.5 ? std::floor(0.5 + zh[i]) : std::floor(0.5 + 10.0 * zh[i]) / 10.0;
        }
        matvec(f_.Q, D_, zt, z);
        for (int i = 0; i < D_; ++i) t[i] = cond_[static_cast<std::size_t>(i)] * z[i] * z[i];
        const double big = np_maximum(std::abs(zh[0]) / 1e4, row_sum(t, static_cast<std::size_t>(D_)));
        return 0.1 * big + f_pen(x, D_) + f_.f_opt;
    }

    // z = max(1, sqrt(D)/8) R x + 1/2 (BBOB-5)
    double f9(const double* x) const
    {
        double z[kMaxSoloD] = {};
        matvec(f_.R, D_, x, z);
        for (int i = 0; i < D_; ++i) z[i] = c_ * z[i] + 0.5;
        return rosenbrock(z) + f_.f_opt;
    }

    double f10(const double* d) const
    {
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, d, z);
        for (int i = 0; i < D_; ++i) {
            const double v = t_osz(z[i]);
            t[i] = cond_[static_cast<std::size_t>(i)] * v * v;
        }
        return row_sum(t, static_cast<std::size_t>(D_)) + f_.f_opt;
    }

    double f11(const double* d) const
    {
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, d, z);
        for (int i = 0; i < D_; ++i) z[i] = t_osz(z[i]);
        for (int i = 1; i < D_; ++i) t[i - 1] = z[i] * z[i];
        return 1e6 * (z[0] * z[0]) + row_sum(t, static_cast<std::size_t>(D_ - 1)) + f_.f_opt;
    }

    double f12(const double* d) const
    {
        double a[kMaxSoloD] = {}, b[kMaxSoloD] = {}, z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, d, a);
        t_asy(a, D_, 0.5, b);
        matvec(f_.R, D_, b, z);
        for (int i = 1; i < D_; ++i) t[i - 1] = z[i] * z[i];
        return z[0] * z[0] + 1e6 * row_sum(t, static_cast<std::size_t>(D_ - 1)) + f_.f_opt;
    }

    // z = R Lambda^(1/100) Q T_osz(R d); BBOB-8
    double f16(const double* x, const double* d) const
    {
        double a[kMaxSoloD] = {}, b[kMaxSoloD] = {}, z[kMaxSoloD] = {}, s[kMaxSoloD] = {};
        matvec(f_.R, D_, d, a);
        for (int i = 0; i < D_; ++i) a[i] = t_osz(a[i]);
        matvec(f_.Q, D_, a, b);
        for (int i = 0; i < D_; ++i) b[i] = f_.lam[static_cast<std::size_t>(i)] * b[i];
        matvec(f_.R, D_, b, z);
        for (int i = 0; i < D_; ++i) {
            double acc = 0.0;
            for (int k = 0; k < 12; ++k) acc = acc + half_k_[k] * dm::cos(wave_k_[k] * (z[i] + 0.5));
            s[i] = acc;
        }
        const double g = row_sum(s, static_cast<std::size_t>(D_)) / D_ - f0_;
        return 10.0 * (g * g * g) + (10.0 / D_) * f_pen(x, D_) + f_.f_opt;
    }

    // z = max(1, sqrt(D)/8) R x + 1/2 (BBOB-5)
    double f19(const double* x) const
    {
        double z[kMaxSoloD] = {}, t[kMaxSoloD] = {};
        matvec(f_.R, D_, x, z);
        for (int i = 0; i < D_; ++i) z[i] = c_ * z[i] + 0.5;
        for (int i = 0; i < D_ - 1; ++i) {
            const double a = z[i] * z[i] - z[i + 1];
            const double b = z[i] - 1.0;
            const double s = 100.0 * a * a + b * b;
            t[i] = s / 4000.0 - dm::cos(s);
        }
        return 10.0 / (D_ - 1.0) * row_sum(t, static_cast<std::size_t>(D_ - 1)) + 10.0 + f_.f_opt;
    }

    // z = Q Lambda^100 R d; BBOB-6, BBOB-8
    double f23(const double* x, const double* d) const
    {
        double a[kMaxSoloD] = {}, z[kMaxSoloD] = {}, p[kMaxSoloD] = {};
        matvec(f_.R, D_, d, a);
        for (int i = 0; i < D_; ++i) a[i] = f_.lam[static_cast<std::size_t>(i)] * a[i];
        matvec(f_.Q, D_, a, z);
        for (int i = 0; i < D_; ++i) {
            double s = 0.0, two = 1.0;
            for (int j = 1; j <= 32; ++j) {
                two = 2.0 * two;
                const double v = two * z[i];
                s = s + std::abs(v - std::nearbyint(v)) / two;
            }
            p[i] = dm::pow(1.0 + (i + 1.0) * s, c_);
        }
        const double k = 10.0 / (D_ * D_);
        return k * row_prod(p, static_cast<std::size_t>(D_)) - k + f_pen(x, D_) + f_.f_opt;
    }

    // mu_0 = 2.5, d = 1, s = c_, mu_1 = mu1_
    double f24(const double* x) const
    {
        double a[kMaxSoloD] = {}, b[kMaxSoloD] = {}, z[kMaxSoloD] = {}, u[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) {
            const double xh = 2.0 * f_.sign[static_cast<std::size_t>(i)] * x[i];
            a[i] = xh - 2.5;
            b[i] = xh - mu1_;
        }
        double aa[kMaxSoloD] = {}, bb[kMaxSoloD] = {};
        for (int i = 0; i < D_; ++i) {
            aa[i] = a[i] * a[i];
            bb[i] = b[i] * b[i];
        }
        const double funnel = std::min(row_sum(aa, static_cast<std::size_t>(D_)),
                                       1.0 * D_ + c_ * row_sum(bb, static_cast<std::size_t>(D_)));
        matvec(f_.R, D_, a, u);
        for (int i = 0; i < D_; ++i) u[i] = f_.lam[static_cast<std::size_t>(i)] * u[i];
        matvec(f_.Q, D_, u, z);
        return funnel + rastrigin(z) + 1e4 * f_pen(x, D_) + f_.f_opt;
    }
};

// ---- the pair (bbob_biobj.py) and its integer variables (bbob_biobj_mixint.py)

class Pair {
public:
    Pair() = default;

    // the two functions of F = (f_alpha, f_beta) in n variables; with levels,
    // the first ones integer (MIXINT-1: each objective with its own levels)
    Pair(const std::vector<Function>& fs, int n)
    {
        if (fs.size() != 2) throw std::invalid_argument("suite: bbob-biobj: two functions");
        if (n > kMaxD) throw std::invalid_argument("suite: bbob-biobj: D from 2 to 15");
        for (int o = 0; o < 2; ++o) objective_[o] = Objective(fs[static_cast<std::size_t>(o)], n);
        n_ = n;
        k_ = static_cast<int>(fs[0].levels.size());
        if (fs[1].levels.size() != fs[0].levels.size() || k_ > n)
            throw std::invalid_argument("suite: bbob-biobj-mixint: the same integer variables in both objectives");
        for (const Function& f : fs)
            for (std::size_t i = 0; i < f.levels.size(); ++i)
                if (f.levels[i].size() != fs[0].levels[i].size() || f.levels[i].empty())
                    throw std::invalid_argument("suite: bbob-biobj-mixint: one level or more per value");
    }

    int integers() const { return k_; }
    const Objective& objective(int o) const { return objective_[o]; }

    void evaluate(const double* x, double* f) const
    {
        for (int o = 0; o < 2; ++o) {
            if (k_ == 0) {
                f[o] = objective_[o](x);
                continue;
            }
            double z[kMaxSoloD] = {};
            bool nan = false;
            for (int i = 0; i < n_; ++i) z[i] = x[i];
            const auto& levels = objective_[o].function().levels;
            for (int i = 0; i < k_; ++i) {
                // continuous(): the integer rounded half to even and kept in
                // 0 .. l-1, then its level (a NaN has none: Python raises)
                const auto& z_i = levels[static_cast<std::size_t>(i)];
                const double j = clip(std::nearbyint(x[i]), 0.0, static_cast<double>(z_i.size() - 1));
                if (std::isnan(j)) {
                    nan = true;
                    break;
                }
                z[i] = z_i[static_cast<std::size_t>(j)];
            }
            f[o] = nan ? std::numeric_limits<double>::quiet_NaN() : objective_[o](z);
        }
    }

private:
    Objective objective_[2];
    int n_ = 0, k_ = 0;
};

// ---- the instances (bbob.py, instance; bbob_biobj.py, problem; task 17,
// supplement 5, §3) -----------------------------------------------------------

// the diagonal of Lambda^alpha: alpha^(1/2 (i - 1)/(D - 1)) (bbob.lam)
inline std::vector<double> lam(double alpha, int D)
{
    if (D < 2) return {1.0};
    std::vector<double> v(static_cast<std::size_t>(D));
    for (int i = 0; i < D; ++i) v[static_cast<std::size_t>(i)] = dm::pow(alpha, 0.5 * i / (D - 1.0));
    return v;
}

// D draws lo + (hi - lo) U
inline std::vector<double> uniform(std::mt19937& g, double lo, double hi, int D)
{
    std::vector<double> v(static_cast<std::size_t>(D));
    for (double& e : v) e = lo + (hi - lo) * rnd::uniform01(g);
    return v;
}

// Function fid at instance iid in D variables (bbob.py, instance): x_opt
// uniform on [-4, 4]^D ([-3, 3]^D for f8, drawn after the first), f_opt
// Cauchy about 0 of scale 100 rounded to two decimals (half to even) within
// +-1000, R and Q Haar rotations, Lambda^10, f20's signs (U < 1/2: -1) and
// x_opt = 4.2096874633 s / 2, f21's 101 peaks (w_1 = 10, w_i = 1.1 + 8 (i -
// 2)/99; the alpha_i a permutation of 1000^(2j/99); C_i a permutation of
// Lambda^alpha_i over alpha_i^(1/4); y_1 on [-4, 4]^D, the others on
// [-5, 5]^D (BBOB-1); x_opt = y_1). The fourteen of task 18 draw the same
// way, in the same order, what they use: f5's and f24's signs as f20's, then
// x_opt = 5 s (f5) and 2.5 s / 2 (f24); f22's 21 peaks as f21's (w_i = 1.1 +
// 8 (i - 2)/19, alpha_1 = 1000^2, the others a permutation of 1000^(2j/19),
// y_1 on [-3.92, 3.92]^D, the others on [-4.9, 4.9]^D: BBOB-7); f9's and
// f19's x_opt the point where z = 1 (BBOB-5), sum_i R_ij / (2 max(1,
// sqrt(D)/8)), the sum over i in order. Lambda^alpha takes no draws.
inline Function instance(int fid, int iid, int D)
{
    std::mt19937 g(instances::seed({20260916u, static_cast<std::uint64_t>(fid), static_cast<std::uint64_t>(iid),
                                    static_cast<std::uint64_t>(D)}));
    Function f;
    f.fid = fid;
    f.x_opt = uniform(g, -4.0, 4.0, D);
    if (fid == 8) f.x_opt = uniform(g, -3.0, 3.0, D);
    const double c = rnd::cauchy(g, 0.0, 100.0);
    f.f_opt = clip(std::nearbyint(100.0 * c) / 100.0, -1000.0, 1000.0);
    if (uses_r(fid)) f.R = instances::haar(g, D);
    if (uses_q(fid)) f.Q = instances::haar(g, D);
    if (uses_l10(fid)) f.l10 = lam(10.0, D);
    if (alpha_of(fid) > 0.0) f.lam = lam(alpha_of(fid), D);
    if (uses_sign(fid)) {
        const double scale = fid == 5 ? 5.0 : fid == 20 ? 0.5 * 4.2096874633 : 0.5 * 2.5;
        for (int i = 0; i < D; ++i) f.sign.push_back(rnd::uniform01(g) < 0.5 ? -1.0 : 1.0);
        for (int i = 0; i < D; ++i) f.x_opt[static_cast<std::size_t>(i)] = scale * f.sign[static_cast<std::size_t>(i)];
    }
    if (fid == 9 || fid == 19) {
        const double two_c = 2.0 * std::max(1.0, std::sqrt(static_cast<double>(D)) / 8.0);
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int i = 0; i < D; ++i) s = s + f.R[static_cast<std::size_t>(i * D + j)];
            f.x_opt[static_cast<std::size_t>(j)] = s / two_c;
        }
    }
    if (fid == 21 || fid == 22) {
        const int peaks = peaks_of(fid);
        const double q = peaks - 2.0;                           // 99, 19
        f.w.push_back(10.0);
        for (int j = 0; j < peaks - 1; ++j) f.w.push_back(1.1 + 8.0 * j / q);
        std::vector<double> alpha;
        for (int j = 0; j < peaks - 1; ++j) alpha.push_back(dm::pow(1000.0, 2.0 * j / q));
        rnd::shuffle(alpha.begin(), alpha.end(), g);
        alpha.insert(alpha.begin(), fid == 21 ? 1000.0 : 1000.0 * 1000.0);
        for (int k = 0; k < peaks; ++k) {
            std::vector<double> l = lam(alpha[static_cast<std::size_t>(k)], D);
            rnd::shuffle(l.begin(), l.end(), g);
            const double scale = dm::pow(alpha[static_cast<std::size_t>(k)], 0.25);
            for (double v : l) f.C.push_back(v / scale);
        }
        const double first = fid == 21 ? 4.0 : 3.92, rest = fid == 21 ? 5.0 : 4.9;
        for (double v : uniform(g, -first, first, D)) f.y.push_back(v);
        for (int k = 1; k < peaks; ++k)
            for (double v : uniform(g, -rest, rest, D)) f.y.push_back(v);
        f.x_opt.assign(f.y.begin(), f.y.begin() + D);
    }
    return f;
}

// The ten functions in the paper's order of F1..F55 (bbob_biobj.PAIRS):
// (f_a, f_b) over the pairs a <= b of BASE.
inline void pair_of(int fnum, int& fa, int& fb)
{
    static const int base[10] = {1, 2, 6, 8, 13, 14, 15, 17, 20, 21};
    int at = 0;
    for (int i = 0; i < 10; ++i)
        for (int j = i; j < 10; ++j)
            if (++at == fnum) {
                fa = base[i];
                fb = base[j];
                return;
            }
    throw std::invalid_argument("suite: bbob-biobj has F1..F55");
}

// F fnum in D variables at bi-objective instance iid (bbob_biobj.problem):
// the instances K_alpha, K_beta of Section 5.1.3 (1: 2 and 4; 2: 3 and 5;
// else 2K + 1 and 2K + 2), K_beta raised until the optima lie 1e-4 apart
// and the ideal and the nadir (BIOBJ-1) 0.1 apart, 100 tries at most.
inline std::vector<Function> biobj(int fnum, int D, int iid)
{
    int fa = 0, fb = 0;
    pair_of(fnum, fa, fb);
    int ka = iid == 1 ? 2 : iid == 2 ? 3 : 2 * iid + 1;
    int kb = iid == 1 ? 4 : iid == 2 ? 5 : ka + 1;
    const Function a = instance(fa, ka, D);
    const Objective oa(a, D);
    for (int tries = 0; tries < 100; ++tries, ++kb) {
        Function b = instance(fb, kb, D);
        const Objective ob(b, D);
        double dx = 0.0;
        for (int i = 0; i < D; ++i) {
            const double e = a.x_opt[static_cast<std::size_t>(i)] - b.x_opt[static_cast<std::size_t>(i)];
            dx = dx + e * e;
        }
        const double n1 = oa(b.x_opt.data()) - a.f_opt, n2 = ob(a.x_opt.data()) - b.f_opt;
        if (std::sqrt(dx) >= 1e-4 && std::sqrt(n1 * n1 + n2 * n2) >= 1e-1) return {a, b};
    }
    throw std::runtime_error("suite: bbob-biobj: no second instance far enough in 100 tries");
}

// The levels z_1 .. z_l of an integer variable of arity l whose optimum is
// x_opt_i (bbob_biobj_mixint.levels; Tusar et al. 2019, §3.1.1): -4 + 8 j /
// (l + 1), moved so that the one nearest x_opt_i (the first of equal
// distances) is x_opt_i.
inline std::vector<double> levels(double x_opt_i, int l)
{
    std::vector<double> y;
    for (int j = 1; j <= l; ++j) y.push_back(-4.0 + 8.0 * j / (l + 1));
    std::size_t best = 0;
    for (std::size_t j = 1; j < y.size(); ++j)
        if (std::abs(y[j] - x_opt_i) < std::abs(y[best] - x_opt_i)) best = j;
    const double d = y[best] - x_opt_i;
    for (double& v : y) v = v - d;
    return y;
}

// bbob-biobj-mixint's functions: F fnum at instance 1 (bbob_biobj_mixint.
// MixedInteger), the first 4 D / 5 variables integer, of arities 2, 4, 8 and
// 16, D / 5 of each, each objective with its own levels.
inline std::vector<Function> mixint(int fnum, int D)
{
    std::vector<Function> fs = biobj(fnum, D, 1);
    static const int arity[4] = {2, 4, 8, 16};
    for (Function& f : fs)
        for (int a = 0, i = 0; a < 4; ++a)
            for (int r = 0; r < D / 5; ++r, ++i) f.levels.push_back(levels(f.x_opt[static_cast<std::size_t>(i)], arity[a]));
    return fs;
}

}   // namespace bbob
}   // namespace mootation::suite

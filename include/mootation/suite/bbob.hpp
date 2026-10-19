#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// bbob-biobj and bbob-biobj-mixint (task 12 of 2026-10-19, §1): the ten bbob
// functions of benchmarks/bbob.py, paired as benchmarks/bbob_biobj.py pairs
// them, and the integer variables of benchmarks/bbob_biobj_mixint.py, with the
// values those modules compute, bit for bit.
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
// Python keeps what an instance draws and what a problem's frame is — x_opt,
// f_opt, the rotations R and Q, the diagonal of Lambda^10, f20's signs, f21's
// peaks, the levels of the integer variables, the ideal and the nadir — and
// hands the numbers over (benchmarks/native.py). C++ evaluates. The readings
// BBOB-1..4, BIOBJ-1..2 and MIXINT-1..3 are those of the Python modules.
//
// THE ORDER OF THE ARITHMETIC is that of the NumPy code, as in the rest of the
// suite (common.hpp): sums along a row pairwise, x ** 2 of an array x * x,
// log, exp, sin, cos and pow the C library's one value at a time, and the
// grouping of every Python expression kept. Python's v ** 2 of a float
// (pow_each) is the C library's pow of |v| (py_square), not v * v.
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
// sizes the orders were read off for bound D to 2..15.
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

#include "common.hpp"

#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#endif

namespace mootation::suite {

namespace lm {
MOOTATION_SUITE_NOINLINE inline double log(double x) { volatile double v = x; return std::log(v); }
}   // namespace lm

namespace bbob {

constexpr int kMinD = 2;
constexpr int kMaxD = 15;                       // the sizes the BLAS orders were read off for
constexpr int kPeaks = 101;                     // f21
constexpr double kTwoPi = 2.0 * kPi;            // 2.0 * math.pi, as Python folds it first
constexpr double kSchwefel = 4.189828872724339; // f20's constant

// One bbob function at one instance in D variables: what bbob.py's Instance
// holds, and for bbob-biobj-mixint the levels its integers stand for.
struct Function {
    int fid = 0;                                // 1, 2, 6, 8, 13, 14, 15, 17, 20 or 21
    std::vector<double> x_opt;                  // D
    double f_opt = 0.0;
    std::vector<double> R, Q;                   // D x D, row-major (f6, f13-f15, f17, f21: R; f6, f13, f15, f17: Q)
    std::vector<double> l10;                    // the diagonal of Lambda^10 (f6, f13, f15, f17, f20)
    std::vector<double> sign;                   // f20: the vector 1+-
    std::vector<double> w, C, y;                // f21: 101 weights, the 101 x D diagonals of C_i, the 101 x D peaks y_i
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
// a negative base itself, then calls the C library's pow
inline double py_square(double v)
{
    if (std::isnan(v)) return v;
    return lm::pow(std::abs(v), 2.0);
}

// ---- the transformations (Hansen et al., Section 0.2) -----------------------

// T_osz of one value: sign(x) exp(x^ + 0.049 (sin(c1 x^) + sin(c2 x^)))
inline double t_osz(double x)
{
    const double xh = x != 0.0 ? lm::log(std::abs(x)) : 0.0;
    const double c1 = x > 0.0 ? 10.0 : 5.5;
    const double c2 = x > 0.0 ? 7.9 : 3.1;
    return np_sign(x) * lm::exp(xh + 0.049 * (lm::sin(c1 * xh) + lm::sin(c2 * xh)));
}

// T_asy^beta of a D-vector: x_i^(1 + beta (i-1)/(D-1) sqrt(x_i)) where x_i > 0
inline void t_asy(const double* x, int D, double beta, double* out)
{
    for (int i = 0; i < D; ++i) {
        const double e = 1.0 + beta * (i / (D - 1.0)) * std::sqrt(std::abs(x[i]));
        out[i] = x[i] > 0.0 ? lm::pow(std::abs(x[i]), e) : x[i];
    }
}

// f_pen: sum_i max(0, |x_i| - 5)^2
inline double f_pen(const double* x, int D)
{
    double t[kMaxD];
    for (int i = 0; i < D; ++i) {
        const double m = np_maximum(0.0, std::abs(x[i]) - 5.0);
        t[i] = m * m;
    }
    return row_sum(t, static_cast<std::size_t>(D));
}

// ---- the ten functions (bbob.py's _f1 .. _f21, one point) -------------------

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
        double d[kMaxD];
        for (int i = 0; i < D_; ++i) d[i] = x[i] - f_.x_opt[static_cast<std::size_t>(i)];
        switch (f_.fid) {
            case 1: return dot(d, d, D_) + f_.f_opt;
            case 2: return f2(d);
            case 6: return f6(d);
            case 8: return f8(d);
            case 13: return f13(d);
            case 14: return f14(d);
            case 15: return f15(d);
            case 17: return f17(x, d);
            case 20: return f20(x);
            default: return f21(x);
        }
    }

private:
    Function f_;
    int D_ = 0;
    std::vector<double> cond_;                  // f2: 10^(6 (i-1)/(D-1)); f14: the exponents 2 + 4 (i-1)/(D-1)
    std::vector<double> two_abs_;               // f20: 2 |x_opt| (BBOB-2)

    void fail(const std::string& why) const
    {
        throw std::invalid_argument("suite: bbob f" + std::to_string(f_.fid) + ": " + why);
    }

    // the sizes the function reads, before any evaluation
    void check() const
    {
        const std::size_t n = static_cast<std::size_t>(D_), peaks = static_cast<std::size_t>(kPeaks);
        if (D_ < kMinD || D_ > kMaxD) fail("D from 2 to 15");
        if (f_.x_opt.size() != n) fail("x_opt of D values");
        const int id = f_.fid;
        const bool rot_r = id == 6 || id == 13 || id == 14 || id == 15 || id == 17 || id == 21;
        const bool rot_q = id == 6 || id == 13 || id == 15 || id == 17;
        const bool lam = id == 6 || id == 13 || id == 15 || id == 17 || id == 20;
        if (id != 1 && id != 2 && id != 8 && !rot_r && id != 20) fail("not one of the ten");
        if (rot_r && f_.R.size() != n * n) fail("R of D x D");
        if (rot_q && f_.Q.size() != n * n) fail("Q of D x D");
        if (lam && f_.l10.size() != n) fail("Lambda^10 of D values");
        if (id == 20 && f_.sign.size() != n) fail("the signs of D values");
        if (id == 21 && (f_.w.size() != peaks || f_.C.size() != peaks * n || f_.y.size() != peaks * n))
            fail("101 peaks");
    }

    // the constants bbob.py computes in every call, once
    void prepare()
    {
        const int D = D_;
        if (f_.fid == 2)
            for (int i = 0; i < D; ++i) cond_.push_back(lm::pow(10.0, 6.0 * i / (D - 1.0)));
        if (f_.fid == 14)
            for (int i = 0; i < D; ++i) cond_.push_back(2.0 + 4.0 * i / (D - 1.0));
        if (f_.fid == 20)
            for (double v : f_.x_opt) two_abs_.push_back(2.0 * std::abs(v));
    }

    // Q Lambda^10 R d
    void qlr(const double* d, double* z) const
    {
        double a[kMaxD];
        matvec(f_.R, D_, d, a);
        for (int i = 0; i < D_; ++i) a[i] = f_.l10[static_cast<std::size_t>(i)] * a[i];
        matvec(f_.Q, D_, a, z);
    }

    double f2(const double* d) const
    {
        double t[kMaxD];
        for (int i = 0; i < D_; ++i) {
            const double z = t_osz(d[i]);
            t[i] = cond_[static_cast<std::size_t>(i)] * z * z;
        }
        return row_sum(t, static_cast<std::size_t>(D_)) + f_.f_opt;
    }

    double f6(const double* d) const
    {
        double z[kMaxD], t[kMaxD];
        qlr(d, z);
        for (int i = 0; i < D_; ++i) {
            const double s = z[i] * f_.x_opt[static_cast<std::size_t>(i)] > 0.0 ? 100.0 : 1.0;
            const double v = s * z[i];
            t[i] = v * v;
        }
        return lm::pow(t_osz(row_sum(t, static_cast<std::size_t>(D_))), 0.9) + f_.f_opt;
    }

    double f8(const double* d) const
    {
        const double c = std::max(1.0, std::sqrt(static_cast<double>(D_)) / 8.0);
        double z[kMaxD], t[kMaxD];
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
        double z[kMaxD];
        qlr(d, z);
        return py_square(z[0]) + 100.0 * std::sqrt(dot(z + 1, z + 1, D_ - 1)) + f_.f_opt;
    }

    double f14(const double* d) const
    {
        double z[kMaxD], t[kMaxD];
        matvec(f_.R, D_, d, z);
        for (int i = 0; i < D_; ++i) t[i] = lm::pow(std::abs(z[i]), cond_[static_cast<std::size_t>(i)]);
        return std::sqrt(row_sum(t, static_cast<std::size_t>(D_))) + f_.f_opt;
    }

    double f15(const double* d) const
    {
        double a[kMaxD], b[kMaxD], z[kMaxD], c[kMaxD];
        matvec(f_.R, D_, d, a);
        for (int i = 0; i < D_; ++i) a[i] = t_osz(a[i]);
        t_asy(a, D_, 0.2, b);
        matvec(f_.Q, D_, b, a);
        for (int i = 0; i < D_; ++i) a[i] = f_.l10[static_cast<std::size_t>(i)] * a[i];
        matvec(f_.R, D_, a, z);
        for (int i = 0; i < D_; ++i) c[i] = lm::cos(kTwoPi * z[i]);
        return (10.0 * (D_ - row_sum(c, static_cast<std::size_t>(D_))) + dot(z, z, D_)) + f_.f_opt;
    }

    double f17(const double* x, const double* d) const
    {
        double a[kMaxD], b[kMaxD], z[kMaxD], t[kMaxD];
        matvec(f_.R, D_, d, a);
        t_asy(a, D_, 0.5, b);
        matvec(f_.Q, D_, b, a);
        for (int i = 0; i < D_; ++i) z[i] = f_.l10[static_cast<std::size_t>(i)] * a[i];
        for (int i = 0; i < D_ - 1; ++i) {                    // BBOB-3: i = 1 .. D-1
            const double s = std::sqrt(z[i] * z[i] + z[i + 1] * z[i + 1]);
            const double rt = std::sqrt(s);
            const double sn = lm::sin(50.0 * lm::pow(s, 0.2));
            t[i] = rt + rt * (sn * sn);
        }
        const double inner = row_sum(t, static_cast<std::size_t>(D_ - 1)) / (D_ - 1.0);
        return inner * inner + 10.0 * f_pen(x, D_) + f_.f_opt;
    }

    double f20(const double* x) const
    {
        double xh[kMaxD], z[kMaxD], t[kMaxD];
        for (int i = 0; i < D_; ++i) xh[i] = 2.0 * f_.sign[static_cast<std::size_t>(i)] * x[i];
        for (int i = 0; i < D_; ++i) {
            const double two_abs = two_abs_[static_cast<std::size_t>(i)];
            const double zh = i == 0 ? xh[0] : xh[i] + 0.25 * (xh[i - 1] - two_abs_[static_cast<std::size_t>(i - 1)]);
            z[i] = 100.0 * (f_.l10[static_cast<std::size_t>(i)] * (zh - two_abs) + two_abs);
        }
        for (int i = 0; i < D_; ++i) t[i] = z[i] * lm::sin(std::sqrt(std::abs(z[i])));
        const double term = -row_sum(t, static_cast<std::size_t>(D_)) / (100.0 * D_);
        for (int i = 0; i < D_; ++i) z[i] = z[i] / 100.0;
        return term + kSchwefel + 100.0 * f_pen(z, D_) + f_.f_opt;
    }

    double f21(const double* x) const
    {
        double d[kMaxD], u[kMaxD], q[kMaxD];
        double best = 0.0;
        for (int k = 0; k < kPeaks; ++k) {
            const double* y = f_.y.data() + static_cast<std::size_t>(k) * D_;
            const double* C = f_.C.data() + static_cast<std::size_t>(k) * D_;
            for (int i = 0; i < D_; ++i) d[i] = x[i] - y[i];
            gemm_row(d, f_.R, D_, u);
            for (int i = 0; i < D_; ++i) q[i] = C[i] * u[i] * u[i];
            const double quad = row_sum(q, static_cast<std::size_t>(D_));
            const double v = f_.w[static_cast<std::size_t>(k)] * lm::exp(-quad / (2.0 * D_));
            best = k == 0 ? v : np_maximum(best, v);
        }
        return py_square(t_osz(10.0 - best)) + f_pen(x, D_) + f_.f_opt;
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
            double z[kMaxD];
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

}   // namespace bbob
}   // namespace mootation::suite

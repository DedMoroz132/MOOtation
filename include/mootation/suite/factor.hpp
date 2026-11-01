#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Factorizations with a known degeneracy (task 19, §19.1.2): lowrank, cp3,
// cp3_degenerate. Least squares over the entries of a matrix or a tensor; no
// box (-inf / +inf); f_opt = 0 without noise where the student's rank is the
// teacher's or more (cp3_degenerate: 0 is the infimum, not attained).
//
// lowrank: Y = U_0 V_0^T (m x n, rank r), observed on a mask (each entry with
// probability `observed`), noise (sym.hpp, SYM-1). The unknowns, component
// after component, x = (u_1 (m), v_1 (n), u_2, v_2, ..., u_r, v_r), the
// columns of U and V; the residuals (U V^T)_ij - Y_ij over the observed
// entries, row after row; the Jacobian dr_ij/du_k[i] = v_k[j], dr_ij/dv_k[j]
// = u_k[i]. Symmetry: (U, V) -> (U A, V A^-T), A in GL(r); its tangent space
// at x is {(U X, -V X^T): X in R^{r x r}}, gauge_basis the r^2 directions X =
// E_ab: u_b += u_a, v_a -= v_b (LR-1). blocks: the r components (A a
// permutation). No canon (the group is continuous: its directions are the
// gauge basis).
//
// cp3: T = sum_{rho=1..R} a_rho o b_rho o c_rho (I x J x K), the CP model of
// T.G. Kolda, B.W. Bader, "Tensor decompositions and applications", SIAM
// Review 51(3) 2009, 455-500 (source tensor-decompositions-review_kolda2009),
// (3.1)-(3.2). The unknowns x = (a_1 (I), b_1 (J), c_1 (K), a_2, ...); the
// residuals over T's entries, i slowest, k fastest. Symmetries (Kolda & Bader
// §3.2, after (3.4)): the components may be permuted, and (alpha a) o (beta
// b) o (gamma c) with alpha beta gamma = 1 is the same component — two degrees
// of scale per component, signs included. gauge_basis: per component (a, -b,
// 0) and (a, 0, -c). canon (CP-1): per component ||a|| = ||b|| = ||c|| = (||a||
// ||b|| ||c||)^(1/3), the largest |entry| of a positive and of b positive
// (the signs go to c), then the components by decreasing ||a|| ||b|| ||c||.
//
// cp3_degenerate: the example of V. de Silva, L.-H. Lim, "Tensor rank and the
// ill-posedness of the best low-rank approximation problem", SIAM J. Matrix
// Anal. Appl. 30(3) 2008, 1084-1127 (source tensor-rank-illposed_desilva2008_
// arxiv), (1.1): A = x_1 o x_2 o y_3 + x_1 o y_2 o x_3 + y_1 o x_2 o x_3,
// (x_i, y_i) linearly independent, of rank 3, fitted by a student of rank 2.
// A has no best rank-2 approximation: the rank-2 tensors A_n = n (x_1 + y_1 /
// n) o (x_2 + y_2 / n) o (x_3 + y_3 / n) - n x_1 o x_2 o x_3 tend to A
// (sequence(n) below), with A_n - A = (1/n) (y_1 o y_2 o x_3 + y_1 o x_2 o y_3
// + x_1 o y_2 o y_3) + (1/n^2) y_1 o y_2 o y_3 (Kolda & Bader §3.3 write the
// same residual, with alpha for n), so f = O(1/n^2) while the components' norms grow like n.
//
// Draws, from the key (20261102, crc32(family), instance, sizes, teacher's
// rank) — lowrank: U_0 then V_0 (row after row, normal), the mask (one
// uniform per entry, row after row, where observed < 1), the noise (one
// normal per observed entry); cp3: a_1, b_1, c_1, a_2, ... (normal), the
// noise per entry; cp3_degenerate: x_1, y_1, x_2, y_2, x_3, y_3 (normal).
// The starts: normal entries from the key (20261102, crc32(family +
// ":start"), instance, sizes, student's rank).
//
// READINGS:
//   LR-1. The tangent directions (U X, -V X^T) span the orbit's tangent space
//     where U and V have full column rank; at a rank-deficient point they are
//     dependent (fewer than r^2 independent directions).
//   CP-1. Kolda & Bader's normalized form (3.3) puts the scale into lambda_r
//     with unit columns; canon keeps three vectors (the student's unknowns) and
//     shares the scale equally. A component with a zero vector is set to zero
//     whole (it contributes nothing). The sign rule takes the largest |entry|
//     (the first such on ties) so that it does not hang on a near-zero entry.
//     SYM-2's tolerance: ||a||, ||b||, ||c|| within 64 eps of each other.
// ============================================================================

#include <cmath>
#include <stdexcept>
#include <string>

#include "sym.hpp"

namespace mootation::suite::sym {

class Lowrank final : public Family {
public:
    int rows = 0, cols = 0, rank = 0;              // m, n, the student's r
    std::vector<int> obs_i, obs_j;                 // the observed entries, row after row
    Vec y;

    explicit Lowrank(const Instance& in)
    {
        if (in.shape.size() != 2 || in.shape[0] < 1 || in.shape[1] < 1 || in.teacher < 1 ||
            !(in.observed > 0.0 && in.observed <= 1.0))
            throw std::invalid_argument("lowrank: shape (m, n), rank >= 1, observed in (0, 1]");
        rows = in.shape[0];
        cols = in.shape[1];
        const int rt = in.teacher;
        std::mt19937 g = generator({kKey, instances::crc32("lowrank"), static_cast<std::uint64_t>(in.instance),
                                    static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(cols),
                                    static_cast<std::uint64_t>(rt)});
        Vec U0(static_cast<std::size_t>(rows * rt)), V0(static_cast<std::size_t>(cols * rt));
        for (double& v : U0) v = rnd::normal01(g);
        for (double& v : V0) v = rnd::normal01(g);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                if (in.observed < 1.0 && !(rnd::uniform01(g) < in.observed)) continue;
                double s = 0.0;
                for (int k = 0; k < rt; ++k) s += U0[static_cast<std::size_t>(i * rt + k)] * V0[static_cast<std::size_t>(j * rt + k)];
                obs_i.push_back(i);
                obs_j.push_back(j);
                y.push_back(s);
            }
        add_noise(g, in.noise, y);
        rank = in.student > 0 ? in.student : rt;
        if (rank == rt)
            for (int k = 0; k < rt; ++k) {
                for (int i = 0; i < rows; ++i) optimum.push_back(U0[static_cast<std::size_t>(i * rt + k)]);
                for (int j = 0; j < cols; ++j) optimum.push_back(V0[static_cast<std::size_t>(j * rt + k)]);
            }
        n = rank * (rows + cols);
        m = static_cast<int>(y.size());
        has_jacobian = true;
        gauge_dim = rank * rank;
        blocks = {{0, rows + cols, rank}};
        bounds = box(n, -kInf, kInf);
        instance_ = in.instance;
    }

    void residuals(const double* x, double* r, double* J) const override
    {
        const int w = rows + cols;
        if (J) std::fill(J, J + m * n, 0.0);
        for (std::size_t e = 0; e < y.size(); ++e) {
            const int i = obs_i[e], j = obs_j[e];
            double s = 0.0;
            for (int k = 0; k < rank; ++k) s += x[k * w + i] * x[k * w + rows + j];
            r[e] = s - y[e];
            if (!J) continue;
            for (int k = 0; k < rank; ++k) {
                double* row = J + static_cast<std::ptrdiff_t>(e) * n;
                row[k * w + i] = x[k * w + rows + j];
                row[k * w + rows + j] = x[k * w + i];
            }
        }
    }

    // column (a, b) = a r + b: u_b += u_a, v_a -= v_b
    void gauge(const double* x, double* N) const override
    {
        const int w = rows + cols, d = gauge_dim;
        std::fill(N, N + n * d, 0.0);
        for (int a = 0; a < rank; ++a)
            for (int b = 0; b < rank; ++b) {
                const int c = a * rank + b;
                for (int i = 0; i < rows; ++i) N[(b * w + i) * d + c] += x[a * w + i];
                for (int j = 0; j < cols; ++j)
                    N[(a * w + rows + j) * d + c] -= x[b * w + rows + j];
            }
    }

    void start(double* x) const override
    {
        std::mt19937 g = generator({kKey, instances::crc32("lowrank:start"), static_cast<std::uint64_t>(instance_),
                                    static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(cols),
                                    static_cast<std::uint64_t>(rank)});
        for (int i = 0; i < n; ++i) x[i] = rnd::normal01(g);
    }

private:
    int instance_ = 1;
};

class CP3 final : public Family {
public:
    int I = 0, J = 0, K = 0, R = 0;                // sizes, the student's rank
    Vec T;                                         // I J K entries, i slowest

    CP3(const Instance& in, bool degenerate) : degenerate_(degenerate)
    {
        const std::string family = degenerate ? "cp3_degenerate" : "cp3";
        if (in.shape.size() != 3 || in.shape[0] < 2 || in.shape[1] < 2 || in.shape[2] < 2)
            throw std::invalid_argument(family + ": shape (I, J, K), each 2 or more");
        I = in.shape[0];
        J = in.shape[1];
        K = in.shape[2];
        if (!degenerate && in.teacher < 1) throw std::invalid_argument("cp3: rank >= 1");
        if (degenerate && (in.teacher != 0 && in.teacher != 3)) throw std::invalid_argument("cp3_degenerate: rank 3");
        const int rt = degenerate ? 3 : in.teacher;
        std::mt19937 g = generator({kKey, instances::crc32(family), static_cast<std::uint64_t>(in.instance),
                                    static_cast<std::uint64_t>(I), static_cast<std::uint64_t>(J),
                                    static_cast<std::uint64_t>(K), static_cast<std::uint64_t>(rt)});
        Vec teacher;                               // (a, b, c) per component
        if (degenerate) {
            for (int mode = 0; mode < 3; ++mode)   // x_mode, y_mode
                for (int k = 0; k < 2 * size(mode); ++k) xy_.push_back(rnd::normal01(g));
            // (1.1): x1 x2 y3 + x1 y2 x3 + y1 x2 x3
            const int pick[3][3] = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
            for (const auto& p : pick)
                for (int mode = 0; mode < 3; ++mode) {
                    const double* v = vec(mode, p[mode]);
                    teacher.insert(teacher.end(), v, v + size(mode));
                }
        } else {
            teacher.resize(static_cast<std::size_t>(rt * (I + J + K)));
            for (double& v : teacher) v = rnd::normal01(g);
        }
        T.assign(static_cast<std::size_t>(I * J * K), 0.0);
        tensor(teacher.data(), rt, T.data());
        add_noise(g, in.noise, T);
        R = degenerate ? 2 : (in.student > 0 ? in.student : rt);
        if (R == rt) optimum = teacher;
        if (degenerate && in.student != 0 && in.student != 2) throw std::invalid_argument("cp3_degenerate: rank 2");
        n = R * (I + J + K);
        m = I * J * K;
        has_jacobian = has_canon = true;
        gauge_dim = 2 * R;
        blocks = {{0, I + J + K, R}};
        bounds = box(n, -kInf, kInf);
        instance_ = in.instance;
    }

    void residuals(const double* x, double* r, double* Jac) const override
    {
        tensor(x, R, r);
        for (std::size_t e = 0; e < T.size(); ++e) r[e] = r[e] - T[e];
        if (!Jac) return;
        const int w = I + J + K;
        std::fill(Jac, Jac + m * n, 0.0);
        for (int i = 0; i < I; ++i)
            for (int j = 0; j < J; ++j)
                for (int k = 0; k < K; ++k) {
                    double* row = Jac + ((i * J + j) * K + k) * n;
                    for (int c = 0; c < R; ++c) {
                        const double* a = x + c * w;
                        const double* b = a + I;
                        const double* cc = b + J;
                        row[c * w + i] = b[j] * cc[k];
                        row[c * w + I + j] = a[i] * cc[k];
                        row[c * w + I + J + k] = a[i] * b[j];
                    }
                }
    }

    // per component: (a, -b, 0) and (a, 0, -c)
    void gauge(const double* x, double* N) const override
    {
        const int w = I + J + K, d = gauge_dim;
        std::fill(N, N + n * d, 0.0);
        for (int c = 0; c < R; ++c) {
            for (int i = 0; i < I; ++i) {
                N[(c * w + i) * d + 2 * c] = x[c * w + i];
                N[(c * w + i) * d + 2 * c + 1] = x[c * w + i];
            }
            for (int j = 0; j < J; ++j) N[(c * w + I + j) * d + 2 * c] = -x[c * w + I + j];
            for (int k = 0; k < K; ++k)
                N[(c * w + I + J + k) * d + 2 * c + 1] = -x[c * w + I + J + k];
        }
    }

    void canonical(double* x) const override                 // CP-1
    {
        const int w = I + J + K;
        for (int c = 0; c < R; ++c) {
            double* a = x + c * w;
            double* b = a + I;
            double* cc = b + J;
            const double na = norm(a, static_cast<std::size_t>(I)), nb = norm(b, static_cast<std::size_t>(J)),
                         nc = norm(cc, static_cast<std::size_t>(K));
            if (na == 0.0 || nb == 0.0 || nc == 0.0) {
                std::fill(a, a + w, 0.0);
                continue;
            }
            const double big = std::max(na, std::max(nb, nc)), small = std::min(na, std::min(nb, nc));
            if (big - small > 64.0 * kEps * big) {               // not balanced yet (SYM-2)
                const double nu = dm::pow(na * nb * nc, 1.0 / 3.0);
                for (int i = 0; i < I; ++i) a[i] = a[i] * (nu / na);
                for (int j = 0; j < J; ++j) b[j] = b[j] * (nu / nb);
                for (int k = 0; k < K; ++k) cc[k] = cc[k] * (nu / nc);
            }
            if (largest(a, I) < 0.0) {
                for (int i = 0; i < I; ++i) a[i] = -a[i];
                for (int k = 0; k < K; ++k) cc[k] = -cc[k];
            }
            if (largest(b, J) < 0.0) {
                for (int j = 0; j < J; ++j) b[j] = -b[j];
                for (int k = 0; k < K; ++k) cc[k] = -cc[k];
            }
        }
        const int Iw = I, Jw = J, Kw = K;
        sort_blocks(x, blocks[0], [Iw, Jw, Kw](const Vec& v) {    // decreasing ||a|| ||b|| ||c||
            return -(norm(v.data(), static_cast<std::size_t>(Iw)) * norm(v.data() + Iw, static_cast<std::size_t>(Jw)) *
                     norm(v.data() + Iw + Jw, static_cast<std::size_t>(Kw)));
        });
    }

    void start(double* x) const override
    {
        std::mt19937 g = generator({kKey, instances::crc32(std::string(degenerate_ ? "cp3_degenerate" : "cp3") + ":start"),
                                    static_cast<std::uint64_t>(instance_), static_cast<std::uint64_t>(I),
                                    static_cast<std::uint64_t>(J), static_cast<std::uint64_t>(K),
                                    static_cast<std::uint64_t>(R)});
        for (int i = 0; i < n; ++i) x[i] = rnd::normal01(g);
    }

    // cp3_degenerate: the point of A_n (de Silva & Lim's sequence), n^(1/3)
    // on each vector: (n^(1/3) (x_i + y_i / n))_i, then (-n^(1/3) x_1, n^(1/3)
    // x_2, n^(1/3) x_3)
    Vec sequence(double nn) const
    {
        if (!degenerate_) throw std::logic_error("cp3: sequence() is cp3_degenerate's");
        const double s = dm::pow(nn, 1.0 / 3.0);
        Vec x;
        for (int mode = 0; mode < 3; ++mode)
            for (int i = 0; i < size(mode); ++i) x.push_back(s * (vec(mode, 0)[i] + vec(mode, 1)[i] / nn));
        for (int mode = 0; mode < 3; ++mode)
            for (int i = 0; i < size(mode); ++i) x.push_back((mode == 0 ? -s : s) * vec(mode, 0)[i]);
        return x;
    }

    // x_mode (which 0) or y_mode (which 1)
    const double* vec(int mode, int which) const
    {
        std::size_t at = 0;
        for (int q = 0; q < mode; ++q) at += 2 * static_cast<std::size_t>(size(q));
        return xy_.data() + at + static_cast<std::size_t>(which * size(mode));
    }
    int size(int mode) const { return mode == 0 ? I : mode == 1 ? J : K; }

private:
    bool degenerate_ = false;
    Vec xy_;                                       // cp3_degenerate: x_1, y_1, x_2, y_2, x_3, y_3
    int instance_ = 1;

    // out = sum_rho a o b o c over the rank components of v
    void tensor(const double* v, int rank, double* out) const
    {
        const int w = I + J + K;
        for (int i = 0; i < I; ++i)
            for (int j = 0; j < J; ++j)
                for (int k = 0; k < K; ++k) {
                    double s = 0.0;
                    for (int c = 0; c < rank; ++c) s += v[c * w + i] * v[c * w + I + j] * v[c * w + I + J + k];
                    out[(i * J + j) * K + k] = s;
                }
    }

    static double largest(const double* v, int len)
    {
        int at = 0;
        for (int i = 1; i < len; ++i)
            if (std::fabs(v[i]) > std::fabs(v[at])) at = i;
        return v[at];
    }
};

}   // namespace mootation::suite::sym

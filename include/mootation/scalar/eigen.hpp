#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The eigenvalues and eigenvectors of a symmetric matrix, for CMA-ES
// (cmaes.hpp): the symmetric matrix QR algorithm with origin shifts of
// C.L. Lawson, R.J. Hanson, Solving Least Squares Problems, SIAM 1995, Ch. 18,
// §2 (source least-squares-problems_lawson1995): the matrix is brought to
// tridiagonal form by n - 2 Householder similarity transformations
// (Algorithm H1/H2, (10.22)), then A_{k+1} = R_k Q_k^T + sigma_k I with
// Q_k (A_k - sigma_k I) = R_k (18.2)-(18.3), Q_k a product of Givens
// rotations (Algorithm G1, (10.25)), the shift sigma_k the eigenvalue of the
// trailing 2 x 2 block closest to its last diagonal element (18.4). A
// subdiagonal element below eps (|a_i| + |a_{i+1}|) splits the matrix (§2:
// the independent submatrices).
//
// The eigenvalues come out in increasing order, the eigenvectors as the
// columns of V in the same order: A = V diag(w) V^T.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "../fp_contract.hpp"
#include "lsq.hpp"

namespace mootation::eigen {

using lsq::Matrix;

struct Givens {
    double c = 1.0, s = 0.0;
};

// Algorithm G1 (10.25): the rotation with (c s; -s c) (v1; v2) = (r; 0).
inline Givens givens(double v1, double v2)
{
    Givens g;
    if (std::fabs(v1) > std::fabs(v2)) {
        const double w = v2 / v1, q = std::sqrt(1.0 + w * w);
        g.c = 1.0 / q;
        if (v1 < 0.0) g.c = -g.c;
        g.s = w * g.c;
    } else if (v2 != 0.0) {
        const double w = v1 / v2, q = std::sqrt(1.0 + w * w);
        g.s = 1.0 / q;
        if (v2 < 0.0) g.s = -g.s;
        g.c = w * g.s;
    }
    return g;
}

// A symmetric (n x n). w: the eigenvalues, increasing; V: the eigenvectors.
inline void symmetric(const Matrix& A0, std::vector<double>& w, Matrix& V, int max_sweeps = 60)
{
    const std::size_t n = A0.rows;
    Matrix A = A0;
    V = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i) V(i, i) = 1.0;

    // tridiagonal form: A <- P A P, V <- V P, P = I + b^-1 u u^T
    for (std::size_t k = 0; k + 2 < n; ++k) {
        std::vector<double> u(n);
        for (std::size_t i = 0; i < n; ++i) u[i] = A(i, k);
        lsq::House h = lsq::house_make(u.data(), 1, k + 1, k + 2, n);
        if (h.identity) continue;
        for (std::size_t c = 0; c < n; ++c)                  // P A
            lsq::house_apply(u.data(), 1, k + 1, k + 2, n, h, &A(0, c), A.cols);
        for (std::size_t r = 0; r < n; ++r)                  // (P A) P
            lsq::house_apply(u.data(), 1, k + 1, k + 2, n, h, &A(r, 0), 1);
        for (std::size_t r = 0; r < n; ++r)                  // V P
            lsq::house_apply(u.data(), 1, k + 1, k + 2, n, h, &V(r, 0), 1);
        for (std::size_t i = k + 2; i < n; ++i) A(i, k) = A(k, i) = 0.0;
    }
    std::vector<double> a(n), b(n, 0.0);                     // b[i] couples i - 1 and i
    for (std::size_t i = 0; i < n; ++i) a[i] = A(i, i);
    for (std::size_t i = 1; i < n; ++i) b[i] = A(i, i - 1);

    std::size_t q = n;                                       // the block [p, q) not yet split off
    int steps = 0;
    while (q > 1) {
        // split off a negligible subdiagonal element at the bottom
        if (std::fabs(b[q - 1]) <= lsq::kEps * (std::fabs(a[q - 2]) + std::fabs(a[q - 1]))) {
            b[q - 1] = 0.0;
            --q;
            continue;
        }
        if (++steps > max_sweeps * static_cast<int>(n)) break;
        std::size_t p = q - 1;
        while (p > 0 && std::fabs(b[p]) > lsq::kEps * (std::fabs(a[p - 1]) + std::fabs(a[p]))) --p;
        if (p > 0) b[p] = 0.0;
        // the shift (18.4): the eigenvalue of [[a_{q-2}, b], [b, a_{q-1}]] nearest a_{q-1}
        const double x = a[q - 2], y = a[q - 1], z = b[q - 1];
        const double d = 0.5 * (x - y);
        const double r = std::sqrt(d * d + z * z);
        const double sigma = y - z * z / (d + (d >= 0.0 ? r : -r));
        // explicit QR step on the block: Q (T - sigma I) = R, T' = R Q^T + sigma I
        const std::size_t len = q - p;
        Matrix T(len, len);
        for (std::size_t i = 0; i < len; ++i) {
            T(i, i) = a[p + i] - sigma;
            if (i > 0) T(i, i - 1) = T(i - 1, i) = b[p + i];
        }
        std::vector<Givens> rot(len - 1);
        for (std::size_t i = 0; i + 1 < len; ++i) {          // rows i, i + 1
            rot[i] = givens(T(i, i), T(i + 1, i));
            for (std::size_t c = 0; c < len; ++c) {
                const double t1 = T(i, c), t2 = T(i + 1, c);
                T(i, c) = rot[i].c * t1 + rot[i].s * t2;     // Algorithm G2 (10.26)
                T(i + 1, c) = -rot[i].s * t1 + rot[i].c * t2;
            }
        }
        for (std::size_t i = 0; i + 1 < len; ++i) {          // R Q^T: columns i, i + 1
            for (std::size_t r2 = 0; r2 < len; ++r2) {
                const double t1 = T(r2, i), t2 = T(r2, i + 1);
                T(r2, i) = rot[i].c * t1 + rot[i].s * t2;
                T(r2, i + 1) = -rot[i].s * t1 + rot[i].c * t2;
            }
            for (std::size_t r2 = 0; r2 < n; ++r2) {         // V Q^T on the block's columns
                const double t1 = V(r2, p + i), t2 = V(r2, p + i + 1);
                V(r2, p + i) = rot[i].c * t1 + rot[i].s * t2;
                V(r2, p + i + 1) = -rot[i].s * t1 + rot[i].c * t2;
            }
        }
        for (std::size_t i = 0; i < len; ++i) {
            a[p + i] = T(i, i) + sigma;
            if (i > 0) b[p + i] = 0.5 * (T(i, i - 1) + T(i - 1, i));
        }
    }
    // increasing order
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t i, std::size_t j) { return a[i] < a[j]; });
    w.assign(n, 0.0);
    Matrix Vs(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        w[k] = a[order[k]];
        for (std::size_t r2 = 0; r2 < n; ++r2) Vs(r2, k) = V(r2, order[k]);
    }
    V = Vs;
}

}   // namespace mootation::eigen

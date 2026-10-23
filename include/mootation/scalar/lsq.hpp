#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Least squares with linear constraints: the subproblem of the SQP solver
// (slsqp.hpp), written from C.L. Lawson, R.J. Hanson, Solving Least Squares
// Problems, SIAM 1995 (the 1974 book with an appendix; source
// least-squares-problems_lawson1995). "LH" below.
//
//   house_make / house_apply   Householder transformations, Algorithms H1 and
//                              H2 (10.22), with Step 11(b)'s scaling of the
//                              sum of squares
//   least_squares              min ||A x - b|| for A of full column rank, by
//                              Householder triangularization (Ch. 11, HFT/HS1)
//   hfti                       the least squares solution of least length for
//                              any A, Algorithm HFTI (14.9), for several
//                              right-hand sides at once
//   nnls                       min ||E x - f|| subject to x >= 0, Algorithm
//                              NNLS (23.10) with §23.3's guards against
//                              round-off; at most 3n iterations, as LH's
//                              subroutine NNLS
//   ldp                        min ||x|| subject to G x >= h, Algorithm LDP
//                              (23.27): the residual of an NNLS problem
//   lsi                        min ||E x - f|| subject to G x >= h, E of full
//                              column rank, turned into LDP (23.35)-(23.41)
//   lsei                       lsi with equalities C x = d as well: they are
//                              eliminated first (Ch. 20, (20.19)-(20.23), and
//                              §23.6, (23.42)-(23.44))
//
// The multipliers are returned with every solution: for lsi and lsei those of
// the Kuhn-Tucker conditions (23.5), E^T (E x - f) = C^T lam_C + G^T lam_G,
// lam_G >= 0 — what the SQP solver's quasi-Newton update needs.
//
// OURS, where LH leave a choice to the program:
//   LSQ-1. NNLS solves the least squares problem of its Step 6 afresh every
//     time (LH's subroutine updates a QR decomposition, Ch. 24): the problems
//     here have tens of columns, and the plain form is easier to check.
//   LSQ-2. A candidate column of NNLS (Step 5) is refused when its new
//     diagonal element is negligible beside the part of the column the
//     columns already in the set explain (|r| * 0.01 adds nothing to that
//     part's norm, the test LH's subroutine makes through DIFF) or when z_t is
//     not positive (§23.3): w_t is then taken as 0 and Step 3 is done again.
//   LSQ-3. LDP finds G x >= h incompatible when ||r|| (Step 3's ||r|| = 0)
//     is within the round-off of r itself: ||r|| <= 100 eps (1 + sum_ij
//     |E_ij| u_j), the bound of the error of r = E u - f summed term by term.
//     ||r|| is the root of the sum of the squares of r (its other form,
//     ||r||^2 = 1 - h^T u (23.32), cancels near the limit). On random
//     problems the incompatible ones give at most 1.3 of that bound and the
//     compatible ones 2e12 and more. Solutions far longer than 1/sqrt(eps)
//     are kept: a finite-difference gradient across a kink of a constraint
//     (C2-DTLZ2's minimum of spheres) is ~1e8 long, and so is the LDP's
//     solution, which a test of ||r||^2 against eps (DIFF's 1 + ||r||^2 = 1)
//     refuses.
//   LSQ-8. LDP is solved for the same constraints with every row g_i of G
//     scaled to unit length and h to max |h_i| = 1: G x >= h and
//     (g_i / ||g_i||)^T x >= h_i / ||g_i|| are one set, and LDP's solution is
//     linear in h (x(c h) = c x(h)); x and the multipliers are scaled back.
//     A QP whose gradient is ~1e8 long (a finite difference across a kink)
//     gives an h that long, and NNLS on [G^T; h^T] unscaled lost the
//     solution (a compatible relaxed QP found incompatible).
//   LSQ-4. lsei picks the rows of C in the order of the largest remaining
//     norm (the column interchanges of HFTI (14.9), Steps 3-10, on C^T) and
//     takes as its pseudorank the rows whose diagonal element exceeds
//     10 eps max|C| (Step 13 leaves tau to the caller). A row beyond the
//     pseudorank is dependent: it is dropped when the solution of the others
//     meets it within 1000 eps of its scale, else the equalities are
//     incompatible.
//   LSQ-5. A Householder transformation whose part to be zeroed is zero
//     already is the identity (H1 would flip the pivot's sign; the identity
//     keeps triangular input exact).
//   LSQ-7. The sum of H2's Step 8 is taken in four running sums (terms
//     l, l + 4, ... in the first, and so on), then (s0 + s1) + (s2 + s3):
//     the order is the code's, the same everywhere, and four times shorter
//     a chain of additions than the one-by-one sum (BoostDMS's models
//     triangularize matrices of up to 992 x 496 at every iteration).
//   LSQ-6. HFTI's pseudorank: the diagonal elements above max(m, n) eps
//     |r_11| (Step 13 leaves tau to the caller; this is NumPy's default rcond
//     for lstsq, measured on R's diagonal instead of the singular values).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../fp_contract.hpp"

namespace mootation::lsq {

// A dense matrix, row by row.
struct Matrix {
    std::size_t rows = 0, cols = 0;
    std::vector<double> a;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), a(r * c, 0.0) {}
    double& operator()(std::size_t i, std::size_t j) { return a[i * cols + j]; }
    const double& operator()(std::size_t i, std::size_t j) const { return a[i * cols + j]; }
};

inline constexpr double kEps = std::numeric_limits<double>::epsilon();

// ── Householder transformations, LH (10.22) ────────────────────────────────
// A vector is v[0], v[st], v[2 st], ... (a column of a Matrix has st = cols).
// house_make (H1) builds Q = I + b^-1 u u^T, which keeps components 0..p-1 and
// p+1..l-1, puts s in component p and zeros l..m-1: on return v[p] holds s,
// v[l..m-1] the tail of u, and the House its pivot u_p.
struct House {
    double up = 0.0;
    bool identity = true;
};

inline House house_make(double* v, std::size_t st, std::size_t p, std::size_t l, std::size_t m)
{
    House h;
    if (l >= m) return h;
    double t = std::fabs(v[p * st]);
    bool tail_zero = true;
    for (std::size_t i = l; i < m; ++i) {
        t = std::max(t, std::fabs(v[i * st]));
        if (v[i * st] != 0.0) tail_zero = false;
    }
    if (tail_zero) return h;                              // LSQ-5
    double ss = (v[p * st] / t) * (v[p * st] / t);        // Step 1, scaled (Step 11(b))
    for (std::size_t i = l; i < m; ++i) ss += (v[i * st] / t) * (v[i * st] / t);
    double s = t * std::sqrt(ss);
    if (v[p * st] > 0.0) s = -s;                          // Step 2
    h.up = v[p * st] - s;                                 // Step 3
    v[p * st] = s;
    h.identity = false;
    return h;
}

// H2: c := Q c for the Q that house_make left in v (Steps 5-10).
inline void house_apply(const double* v, std::size_t st, std::size_t p, std::size_t l, std::size_t m,
                        const House& h, double* c, std::size_t cst)
{
    if (h.identity) return;
    const double b = v[p * st] * h.up;                    // Step 5
    if (b == 0.0) return;                                 // Step 6
    // Step 8: c_p h + sum c_i v_i, the sum in four running sums (term i to
    // sum (i - l) mod 4, then (s0 + s1) + (s2 + s3)): a fixed order, the same
    // on every platform, that the processor need not wait on (LSQ-7)
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    std::size_t t = l;
    for (; t + 4 <= m; t += 4)
        for (std::size_t k = 0; k < 4; ++k) r[k] += c[(t + k) * cst] * v[(t + k) * st];
    for (std::size_t k = 0; t < m; ++t, ++k) r[k] += c[t * cst] * v[t * st];
    double s = c[p * cst] * h.up + ((r[0] + r[1]) + (r[2] + r[3]));
    s /= b;
    c[p * cst] += s * h.up;                               // Step 9
    for (std::size_t i = l; i < m; ++i) c[i * cst] += s * v[i * st];   // Step 10
}

// ── Triangular solves ───────────────────────────────────────────────────────
// R x = b, R the upper triangle of the first k rows and columns of A.
inline std::vector<double> solve_upper(const Matrix& A, std::size_t k, const std::vector<double>& b)
{
    std::vector<double> x(k, 0.0);
    for (std::size_t ii = k; ii-- > 0;) {
        double s = b[ii];
        for (std::size_t j = ii + 1; j < k; ++j) s -= A(ii, j) * x[j];
        x[ii] = s / A(ii, ii);
    }
    return x;
}

// R^T x = b for the same R (forward substitution).
inline std::vector<double> solve_upper_transposed(const Matrix& A, std::size_t k,
                                                  const std::vector<double>& b)
{
    std::vector<double> x(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j) s -= A(j, i) * x[j];
        x[i] = s / A(i, i);
    }
    return x;
}

// ── Householder triangularization of the columns of A, in place ────────────
// After it, A's upper triangle is R and the columns below it hold the
// transformations (house[j]); qtb(b) applies them to a vector.
struct QR {
    Matrix A;
    std::vector<House> house;

    explicit QR(Matrix a) : A(std::move(a))
    {
        const std::size_t m = A.rows, k = std::min(A.rows, A.cols);
        house.resize(k);
        for (std::size_t j = 0; j < k; ++j) {
            house[j] = house_make(&A(0, j), A.cols, j, j + 1, m);
            for (std::size_t c = j + 1; c < A.cols; ++c)
                house_apply(&A(0, j), A.cols, j, j + 1, m, house[j], &A(0, c), A.cols);
        }
    }

    void qtb(std::vector<double>& b) const
    {
        for (std::size_t j = 0; j < house.size(); ++j)
            house_apply(&A(0, j), A.cols, j, j + 1, A.rows, house[j], b.data(), 1);
    }
};

// min ||A x - b|| for A (m x k, m >= k) of full column rank; false when a
// diagonal element of R is zero.
inline bool least_squares(const Matrix& A, const std::vector<double>& b, std::vector<double>& x)
{
    if (A.rows < A.cols) return false;
    QR qr(A);
    for (std::size_t j = 0; j < A.cols; ++j)
        if (qr.A(j, j) == 0.0) return false;
    std::vector<double> c = b;
    qr.qtb(c);
    x = solve_upper(qr.A, A.cols, c);
    return true;
}

// ── HFTI, LH (14.9): the least squares solution of least length ────────────
// A X ≅ B for any A (m x n) and several right-hand sides at once (LH's
// subroutine takes NB of them): one triangularization with the column
// interchanges of Steps 3-10 — the squared lengths h_l brought down from step
// to step (Step 4) and computed again when that has lost the precision Step 6
// asks for — the pseudorank k = the number of leading |r_jj| > tau, tau =
// max(m, n) eps |r_11| (LSQ-6), the rest of the solution set to 0 (Step 20):
// every x is the solution of least length of the problem with A's pseudorank
// k. A's columns are held one after another while it is triangularized.
inline std::vector<std::vector<double>> hfti(const Matrix& A0, std::vector<std::vector<double>> B)
{
    const std::size_t m = A0.rows, n = A0.cols, mu = std::min(m, n);
    std::vector<std::vector<double>> a(n, std::vector<double>(m));   // a[j]: column j
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) a[j][i] = A0(i, j);
    std::vector<std::size_t> piv(mu);
    std::vector<double> h(n, 0.0);
    double hbar = 0.0;
    auto argmax = [&](std::size_t from) {
        std::size_t lam = from;
        for (std::size_t l = from + 1; l < n; ++l)
            if (h[l] > h[lam]) lam = l;
        return lam;
    };
    for (std::size_t j = 0; j < mu; ++j) {
        std::size_t lam = j;
        bool again = j == 0;
        if (!again) {
            for (std::size_t l = j; l < n; ++l) h[l] -= a[l][j - 1] * a[l][j - 1];   // Step 4
            lam = argmax(j);                                                          // Step 5
            again = !(hbar + 1e-3 * h[lam] > hbar);                                   // Step 6
        }
        if (again) {                                                                  // Steps 7-8
            for (std::size_t l = j; l < n; ++l) {
                double s = 0.0;
                for (std::size_t i = j; i < m; ++i) s += a[l][i] * a[l][i];
                h[l] = s;
            }
            lam = argmax(j);
            hbar = h[lam];
        }
        piv[j] = lam;                                                                 // Steps 9-10
        if (lam != j) {
            std::swap(a[j], a[lam]);
            h[lam] = h[j];
        }
        House hs = house_make(a[j].data(), 1, j, j + 1, m);                          // Steps 11-12
        for (std::size_t c = j + 1; c < n; ++c) house_apply(a[j].data(), 1, j, j + 1, m, hs, a[c].data(), 1);
        for (std::vector<double>& b : B) house_apply(a[j].data(), 1, j, j + 1, m, hs, b.data(), 1);
    }
    std::vector<std::vector<double>> X(B.size(), std::vector<double>(n, 0.0));
    if (mu == 0) return X;
    const double tau = static_cast<double>(std::max(m, n)) * kEps * std::fabs(a[0][0]);
    std::size_t k = 0;                                                                // Step 13
    while (k < mu && std::fabs(a[k][k]) > tau) ++k;
    if (k == 0) return X;
    Matrix R(k, n);                                                                   // [R11 R12]
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t c = 0; c < n; ++c) R(i, c) = a[c][i];
    std::vector<House> g(k);
    if (k < n)                                                                        // Step 16: [R11 R12] K = [W 0]
        for (std::size_t i = k; i-- > 0;) {
            g[i] = house_make(&R(i, 0), 1, i, k, n);
            for (std::size_t r = 0; r < i; ++r) house_apply(&R(i, 0), 1, i, k, n, g[i], &R(r, 0), 1);
        }
    for (std::size_t t = 0; t < B.size(); ++t) {
        std::vector<double> b(B[t].begin(), B[t].begin() + static_cast<std::ptrdiff_t>(k));
        std::vector<double> y = solve_upper(R, k, b);                                 // Steps 17-18
        std::vector<double>& x = X[t];
        for (std::size_t i = 0; i < k; ++i) x[i] = y[i];
        if (k < n)                                                                    // Step 21
            for (std::size_t i = 0; i < k; ++i) house_apply(&R(i, 0), 1, i, k, n, g[i], x.data(), 1);
        for (std::size_t j = mu; j-- > 0;)                                            // Steps 22-23
            if (piv[j] != j) std::swap(x[j], x[piv[j]]);
    }
    return X;
}

// One right-hand side.
inline std::vector<double> hfti(const Matrix& A, const std::vector<double>& b)
{
    return hfti(A, std::vector<std::vector<double>>{b})[0];
}

// ── NNLS, LH (23.10) ────────────────────────────────────────────────────────
enum class Status { ok, incompatible, singular, iterations };

struct NNLSResult {
    std::vector<double> x;      // the solution, >= 0
    std::vector<double> w;      // the dual vector E^T (f - E x)
    Status status = Status::ok;
};

inline NNLSResult nnls(const Matrix& E, const std::vector<double>& f)
{
    const std::size_t m = E.rows, n = E.cols;
    NNLSResult r;
    r.x.assign(n, 0.0);
    r.w.assign(n, 0.0);
    std::vector<char> in_p(n, 0);                       // P (1) or Z (0); Step 1
    std::vector<std::size_t> P;                         // P in the order entered
    std::vector<char> refused(n, 0);                    // w_t taken as 0 (LSQ-2)
    const int max_iter = 3 * static_cast<int>(n);
    int iter = 0;

    // The least squares solution z over the columns of P (Step 6); false
    // when the last column entered depends on the others (LSQ-2).
    auto solve_p = [&](std::vector<double>& z, bool check_last) -> bool {
        const std::size_t k = P.size();
        if (k > m) return false;
        Matrix W(m, k);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < k; ++j) W(i, j) = E(i, P[j]);
        QR qr(W);
        if (check_last) {
            double top = 0.0;
            for (std::size_t i = 0; i + 1 < k; ++i) top += qr.A(i, k - 1) * qr.A(i, k - 1);
            top = std::sqrt(top);
            const double diag = std::fabs(qr.A(k - 1, k - 1));
            if (!(top + diag * 0.01 > top)) return false;
        }
        for (std::size_t j = 0; j < k; ++j)
            if (qr.A(j, j) == 0.0) return false;
        std::vector<double> c = f;
        qr.qtb(c);
        std::vector<double> zp = solve_upper(qr.A, k, c);
        z.assign(n, 0.0);
        for (std::size_t j = 0; j < k; ++j) z[P[j]] = zp[j];
        return true;
    };

    auto dual = [&]() {                                  // Step 2: w = E^T (f - E x)
        std::vector<double> res(m);
        for (std::size_t i = 0; i < m; ++i) {
            double s = f[i];
            for (std::size_t j = 0; j < n; ++j) s -= E(i, j) * r.x[j];
            res[i] = s;
        }
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < m; ++i) s += E(i, j) * res[i];
            r.w[j] = s;
        }
    };

    dual();
    std::fill(refused.begin(), refused.end(), char(0));
    for (;;) {
        // Steps 3-4: the index of Z with the largest positive w
        std::size_t t = n;
        double wmax = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            if (!in_p[j] && !refused[j] && r.w[j] > wmax) { wmax = r.w[j]; t = j; }
        if (t == n) break;                               // Step 3: done
        if (++iter > max_iter) { r.status = Status::iterations; break; }
        in_p[t] = 1;                                     // Step 5
        P.push_back(t);
        std::vector<double> z;
        if (!solve_p(z, true) || !(z[t] > 0.0)) {        // LSQ-2: w_t taken as 0
            in_p[t] = 0;
            P.pop_back();
            refused[t] = 1;
            continue;
        }
        // Loop B, Steps 7-11
        for (;;) {
            bool all_positive = true;
            for (std::size_t j : P)
                if (!(z[j] > 0.0)) { all_positive = false; break; }
            if (all_positive) {                          // Step 7
                for (std::size_t j : P) r.x[j] = z[j];
                break;
            }
            if (++iter > max_iter) { r.status = Status::iterations; break; }
            double alpha = 2.0;                          // Steps 8-9
            std::size_t q = n;
            for (std::size_t j : P)
                if (!(z[j] > 0.0)) {
                    const double a = r.x[j] / (r.x[j] - z[j]);
                    if (a < alpha) { alpha = a; q = j; }
                }
            if (q == n) { r.status = Status::singular; return r; }   // z not finite
            for (std::size_t j : P) r.x[j] += alpha * (z[j] - r.x[j]);   // Step 10
            r.x[q] = 0.0;                              // x_q is zero by Step 9
            // Step 11, with §23.3: a value at or below zero leaves P
            std::vector<std::size_t> keep;
            for (std::size_t j : P) {
                if (r.x[j] <= 0.0) { r.x[j] = 0.0; in_p[j] = 0; }
                else keep.push_back(j);
            }
            P = keep;
            if (P.empty()) break;
            if (!solve_p(z, false)) { r.status = Status::singular; return r; }
        }
        if (r.status != Status::ok) break;
        dual();
        std::fill(refused.begin(), refused.end(), char(0));
    }
    return r;
}

// ── LDP, LH (23.27) ─────────────────────────────────────────────────────────
struct LDPResult {
    std::vector<double> x;      // the solution of least norm
    std::vector<double> lam;    // multipliers of G x >= h: x = G^T lam
    Status status = Status::ok;
};

inline LDPResult ldp(const Matrix& G, const std::vector<double>& h)
{
    const std::size_t m = G.rows, n = G.cols;
    LDPResult r;
    r.x.assign(n, 0.0);
    r.lam.assign(m, 0.0);
    if (m == 0) return r;
    // LSQ-8: the same problem with every row of G of unit length and
    // max |h| = 1 (row i divided by w_i = ||g_i||, h by c)
    std::vector<double> w(m, 1.0);
    double c = 0.0;
    for (std::size_t j = 0; j < m; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += G(j, i) * G(j, i);
        if (s > 0.0) w[j] = std::sqrt(s);
        c = std::max(c, std::fabs(h[j] / w[j]));
    }
    if (c == 0.0) return r;                              // h = 0: x = 0 is the solution
    Matrix E(n + 1, m);                                  // Step 1: E = [G^T; h^T]
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < n; ++i) E(i, j) = G(j, i) / w[j];
        E(n, j) = h[j] / w[j] / c;
    }
    std::vector<double> f(n + 1, 0.0);
    f[n] = 1.0;
    NNLSResult u = nnls(E, f);
    if (u.status == Status::singular) { r.status = Status::singular; return r; }
    // Step 2: r = E u - f; ||r||^2 (= -r_{n+1} by (23.32), which cancels)
    std::vector<double> res(n + 1);
    double rr = 0.0, scale = 1.0;
    for (std::size_t i = 0; i <= n; ++i) {
        double s = -f[i];
        for (std::size_t j = 0; j < m; ++j) {
            s += E(i, j) * u.x[j];
            scale += std::fabs(E(i, j)) * u.x[j];
        }
        res[i] = s;
        rr += s * s;
    }
    if (!(std::sqrt(rr) > 100.0 * kEps * scale)) {      // Step 3, LSQ-3
        r.status = Status::incompatible;
        return r;
    }
    const double fac = rr;
    for (std::size_t i = 0; i < n; ++i) r.x[i] = res[i] / fac * c;   // Step 5, (23.32)
    for (std::size_t j = 0; j < m; ++j) r.lam[j] = u.x[j] / fac * c / w[j];
    if (u.status == Status::iterations) r.status = Status::iterations;
    return r;
}

// ── LSI, LH (23.35)-(23.41) ─────────────────────────────────────────────────
struct LSIResult {
    std::vector<double> x;
    std::vector<double> lam_C;  // multipliers of C x = d (lsei)
    std::vector<double> lam_G;  // multipliers of G x >= h, >= 0
    Status status = Status::ok;
};

// E (me x k) of full column rank, me >= k.
inline LSIResult lsi(const Matrix& E, const std::vector<double>& f, const Matrix& G,
                     const std::vector<double>& h)
{
    const std::size_t k = E.cols, mi = G.rows;
    LSIResult r;
    r.x.assign(k, 0.0);
    r.lam_G.assign(mi, 0.0);
    if (E.rows < k) { r.status = Status::singular; return r; }
    QR qr(E);                                            // E = Q [R; 0]
    for (std::size_t j = 0; j < k; ++j)
        if (qr.A(j, j) == 0.0) { r.status = Status::singular; return r; }
    std::vector<double> ft = f;                          // (23.38)
    qr.qtb(ft);
    ft.resize(k);
    // z = R y - f1 (23.39): G R^-1 z >= h - G R^-1 f1 (23.41)
    Matrix Gh(mi, k);
    std::vector<double> hh(mi);
    for (std::size_t i = 0; i < mi; ++i) {
        std::vector<double> gi(k);
        for (std::size_t j = 0; j < k; ++j) gi[j] = G(i, j);
        std::vector<double> v = solve_upper_transposed(qr.A, k, gi);
        double s = h[i];
        for (std::size_t j = 0; j < k; ++j) {
            Gh(i, j) = v[j];
            s -= v[j] * ft[j];
        }
        hh[i] = s;
    }
    LDPResult z = ldp(Gh, hh);
    if (z.status == Status::incompatible || z.status == Status::singular) {
        r.status = z.status;
        return r;
    }
    std::vector<double> rhs(k);
    for (std::size_t j = 0; j < k; ++j) rhs[j] = z.x[j] + ft[j];
    r.x = solve_upper(qr.A, k, rhs);
    r.lam_G = z.lam;
    r.status = z.status;
    return r;
}

// ── LSEI: equalities eliminated, LH (20.19)-(20.23), (23.42)-(23.44) ──────
// min ||E x - f|| s.t. C x = d, G x >= h; E (me x n) of full column rank.
inline LSIResult lsei(const Matrix& E, const std::vector<double>& f, const Matrix& C,
                      const std::vector<double>& d, const Matrix& G, const std::vector<double>& h)
{
    const std::size_t n = E.cols, mc = C.rows, me = E.rows, mi = G.rows;
    if (mc == 0) {
        LSIResult r = lsi(E, f, G, h);
        r.lam_C.clear();
        return r;
    }
    LSIResult r;
    r.x.assign(n, 0.0);
    r.lam_C.assign(mc, 0.0);
    r.lam_G.assign(mi, 0.0);

    // K triangularizes C from the right: the Householder triangularization of
    // C^T (n x mc), columns in the order of the largest remaining norm (LSQ-4)
    Matrix A(n, mc);
    double cmax = 0.0;
    for (std::size_t i = 0; i < mc; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            A(j, i) = C(i, j);
            cmax = std::max(cmax, std::fabs(C(i, j)));
        }
    std::vector<std::size_t> perm(mc);
    for (std::size_t i = 0; i < mc; ++i) perm[i] = i;
    const std::size_t mu = std::min(n, mc);
    std::vector<House> house(mu);
    for (std::size_t j = 0; j < mu; ++j) {
        std::size_t best = j;
        double best_norm = -1.0;
        for (std::size_t c = j; c < mc; ++c) {
            double s = 0.0;
            for (std::size_t i = j; i < n; ++i) s += A(i, c) * A(i, c);
            if (s > best_norm) { best_norm = s; best = c; }
        }
        if (best != j) {
            for (std::size_t i = 0; i < n; ++i) std::swap(A(i, j), A(i, best));
            std::swap(perm[j], perm[best]);
        }
        house[j] = house_make(&A(0, j), A.cols, j, j + 1, n);
        for (std::size_t c = j + 1; c < mc; ++c)
            house_apply(&A(0, j), A.cols, j, j + 1, n, house[j], &A(0, c), A.cols);
    }
    const double tau = 10.0 * kEps * cmax;
    std::size_t k1 = 0;
    while (k1 < mu && std::fabs(A(k1, k1)) > tau) ++k1;

    // y1 from R11^T y1 = d (the rows of the pseudorank, permuted)
    std::vector<double> dp(mc);
    for (std::size_t i = 0; i < mc; ++i) dp[i] = d[perm[i]];
    std::vector<double> y1(k1, 0.0);
    for (std::size_t i = 0; i < k1; ++i) {
        double s = dp[i];
        for (std::size_t j = 0; j < i; ++j) s -= A(j, i) * y1[j];
        y1[i] = s / A(i, i);
    }
    for (std::size_t i = k1; i < mc; ++i) {                // dependent rows (LSQ-4)
        double s = 0.0, scale = std::fabs(dp[i]);
        for (std::size_t j = 0; j < k1; ++j) {
            s += A(j, i) * y1[j];
            scale += std::fabs(A(j, i) * y1[j]);
        }
        if (std::fabs(s - dp[i]) > 1000.0 * kEps * std::max(1.0, scale)) {
            r.status = Status::incompatible;
            return r;
        }
    }

    // E K and G K: K's transformations applied to every row, in order
    auto times_k = [&](const Matrix& M) {
        Matrix out = M;
        for (std::size_t i = 0; i < M.rows; ++i)
            for (std::size_t j = 0; j < k1; ++j)
                house_apply(&A(0, j), A.cols, j, j + 1, n, house[j], &out(i, 0), 1);
        return out;
    };
    Matrix EK = times_k(E), GK = times_k(G);
    const std::size_t k2 = n - k1;
    Matrix E2(me, k2), G2(mi, k2);
    std::vector<double> ft(me), ht(mi);
    for (std::size_t i = 0; i < me; ++i) {
        double s = f[i];
        for (std::size_t j = 0; j < k1; ++j) s -= EK(i, j) * y1[j];
        ft[i] = s;                                         // (20.21)
        for (std::size_t j = 0; j < k2; ++j) E2(i, j) = EK(i, k1 + j);
    }
    for (std::size_t i = 0; i < mi; ++i) {
        double s = h[i];
        for (std::size_t j = 0; j < k1; ++j) s -= GK(i, j) * y1[j];
        ht[i] = s;                                         // (23.44)
        for (std::size_t j = 0; j < k2; ++j) G2(i, j) = GK(i, k1 + j);
    }
    std::vector<double> y(n, 0.0);
    for (std::size_t j = 0; j < k1; ++j) y[j] = y1[j];
    if (k2 > 0) {
        LSIResult sub = lsi(E2, ft, G2, ht);
        if (sub.status == Status::incompatible || sub.status == Status::singular) {
            r.status = sub.status;
            return r;
        }
        for (std::size_t j = 0; j < k2; ++j) y[k1 + j] = sub.x[j];
        r.lam_G = sub.lam_G;
        r.status = sub.status;
    } else {
        for (std::size_t i = 0; i < mi; ++i)
            if (ht[i] > 1000.0 * kEps * std::max(1.0, std::fabs(h[i]))) {
                r.status = Status::incompatible;
                return r;
            }
    }
    // x = K y (20.23): the transformations in reverse order
    for (std::size_t j = k1; j-- > 0;)
        house_apply(&A(0, j), A.cols, j, j + 1, n, house[j], y.data(), 1);
    r.x = y;

    // lam_C from E^T (E x - f) - G^T lam_G = C^T lam_C: in K's coordinates the
    // first k1 components are R11 mu (the dependent rows' mu are 0)
    std::vector<double> res(me), grad(n, 0.0);
    for (std::size_t i = 0; i < me; ++i) {
        double s = -f[i];
        for (std::size_t j = 0; j < n; ++j) s += E(i, j) * r.x[j];
        res[i] = s;
    }
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < me; ++i) s += E(i, j) * res[i];
        for (std::size_t i = 0; i < mi; ++i) s -= G(i, j) * r.lam_G[i];
        grad[j] = s;
    }
    for (std::size_t j = 0; j < k1; ++j)
        house_apply(&A(0, j), A.cols, j, j + 1, n, house[j], grad.data(), 1);
    grad.resize(k1);
    std::vector<double> muv = solve_upper(A, k1, grad);
    for (std::size_t i = 0; i < k1; ++i) r.lam_C[perm[i]] = muv[i];
    return r;
}

}   // namespace mootation::lsq

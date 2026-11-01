#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The Levenberg-Marquardt method for nonlinear least squares (task 18, stage
// 2): min 1/2 ||r(x)||^2 over l <= x <= u, r: R^n -> R^m, m >= n, the bounds
// -inf/+inf where there are none.
//
// The method is J.J. Moré's, "The Levenberg-Marquardt algorithm:
// implementation and theory", in G.A. Watson (ed.), Numerical Analysis,
// LNM 630, Springer 1978, pp. 105-116 (source levenberg-marquardt_more1978;
// "Moré"), with the Householder factorization of C.L. Lawson, R.J. Hanson,
// Solving Least Squares Problems, SIAM 1995 (lsq.hpp; "LH"). One iteration at
// x, with f = r(x), J = J(x) (Moré's (2.5), (7.1)):
//   1. J P = Q R, Householder with column interchanges (Moré (3.3); the
//      interchanges by the largest remaining column norm, LH (14.9), LM-1);
//      the scaling D_ii = max(D_ii, ||J_:i||) (Moré (6.3), "adaptive"),
//      D_ii = 1 at the first iteration where the column is zero;
//   2. the step p = p(lambda), (J^T J + lambda D^T D) p = -J^T f (Moré (2.4)),
//      from R by Givens rotations (Moré (3.4)-(3.5): only they are made again
//      when lambda changes): lambda = 0 when the Gauss-Newton step has
//      ||D p|| <= (1 + sigma) Delta (Moré (7.1)(a)); else lambda by Moré's
//      (5.5) until |phi(lambda)| <= sigma Delta (Moré (5.1)), sigma = 0.1
//      (Moré §5's practice): phi(lambda) = ||D p(lambda)|| - Delta, the
//      iterate (5.4), the upper bound u_0 = ||(J D^-1)^T f|| / Delta, the lower
//      bound l_0 = -phi(0) / phi'(0) where J has full rank (else 0), u = lambda
//      where phi < 0, l = max(l, lambda - phi / phi'), lambda outside (l, u)
//      moved to max(0.001 u, sqrt(l u)), phi' by Moré's formula after (5.5);
//   3. rho = actual / predicted reduction of ||f||^2 by Moré's (4.4), rho = 0
//      where ||f_+|| > ||f||; x + p taken when rho > 1e-4 (Moré (7.1)(c));
//      rho <= 1/4: Delta = mu Delta, mu the minimizer of the quadratic through
//      delta(0), delta'(0), delta(1) of delta(theta) = 1/2 ||r(x + theta
//      p)||^2, by Moré's (4.5), mu = 1/2 where ||f_+|| <= ||f||, 1/10 where
//      ||f_+|| > 10 ||f||, within [1/10, 1/2] (Moré §4); rho >= 3/4, or rho in
//      (1/4, 3/4) with lambda = 0: Delta = 2 ||D p|| (Moré (7.1)(d)).
// Stopping (Moré (8.3), (8.4)):
//   xtol   Delta <= xtol ||D x||;
//   ftol   the predicted relative reduction (||J p|| / ||f||)^2 + 2 (lambda^1/2
//          ||D p|| / ||f||)^2 <= ftol;
//   gtol   max_j |cos(angle between f and J_:j)| <= gtol (LM-4);
//   and the three with eps for the tolerance (LM-4); max_nfev evaluations.
//
// The Jacobian is the caller's or forward differences: column j with the
// relative step h = sqrt(max(fd_eps, eps)) |x_j| (sqrt(...) where x_j = 0), n
// evaluations, in the problem's own coordinates (no frame is put on x).
//
// OURS, where Moré leaves a choice to the program:
//   LM-1. The column interchanges and the Householder transformations are
//     lsq.hpp's (LH); Moré asks for "column pivoting" without a rule.
//   LM-2. lambda's iteration stops after 10 rounds even short of (5.1) (Moré
//     gives no limit); the first lambda (Moré's "initial estimate alpha_0")
//     is the last iteration's lambda (0 at the start, moved into (l, u) by
//     (5.5)(a)); u_0 = dwarf / min(Delta, 0.1) where J^T f = 0 (dwarf the
//     smallest normal number), so that (l, u) is not empty.
//   LM-3. The first Delta is factor ||D x_0|| (factor where that is 0, factor
//     = 100), and the first iteration's Delta at most ||D p|| (Moré leaves
//     Delta_0 open). LM-3b: a step with lambda = 0 that is refused would come
//     again, unchanged, for every Delta >= ||D p|| / (1 + sigma): Delta is
//     multiplied by the same mu until it is below that, without evaluating
//     r at the same point again (the iterates are Moré's, the evaluations
//     fewer).
//   LM-4. ftol = xtol = 1e-8 as in Moré's runs (§8); gtol (a stationarity test
//     Moré's stopping rules do not have) is 0, off; the tests with eps say
//     that the tolerance asked for cannot be reached.
//   LM-5. ||.|| is the plain Euclidean norm, scaled by the largest |entry|
//     (no overflow).
//   Checked against Moré's §8 table (tests/test_lm_sets.cpp): on four of
//   his twelve runs (Kowalik-Osborne from x_0, Bard from 10 x_0 and 100 x_0,
//   the helical valley from x_0) his NF and NJ come out exactly.
//
// Bounds (task 18 §3.1: bounds may be -inf/+inf per coordinate) by the
// interior trust region of T.F. Coleman, Y. Li, "An interior trust region
// approach for nonlinear minimization subject to bounds", SIAM J. Optim. 6(2)
// 1996, pp. 418-445 (source trust-region-reflective_coleman1996; "CL"), with
// B = J^T J. Why theirs and not a projection of Moré's step: their method "reduces to
// a standard trust region approach for the unconstrained problem when there
// are no upper or lower bounds" (abstract, end of §2.2: C = 0, D = I), so lm
// with infinite bounds is Moré's method above, unchanged; their convergence
// (Theorems 3.4-3.11) needs only the ellipsoidal subproblem (2.10), which
// step 2 solves, not a QP with inequalities; a projected Moré step has no such
// result in the sources at hand. With a finite bound, at x strictly inside:
//   g = J^T f, v by CL's Definition 2.1, |v_i| the distance to the bound g
//   points at (1 where that bound is infinite); C = diag(g_i J^v_ii / |v_i|)
//   (CL (2.8), J^v_ii = -1, +1 or 0 by the case of Definition 2.1, 0 where
//   g_i = 0); the model psi(s) = g^T s + 1/2 s^T (J^T J + C) s (CL (2.6)-(2.7));
//   p from step 2 for the matrix [J; C^1/2] and the scaling diag(|v|^-1/2) D
//   (CL (2.10) with Dbar = Moré's D, as CL allow after (2.10)); s1 =
//   alpha*[p] = theta tau p, tau = min(1, distance to the boundary along p)
//   (CL (2.11)-(2.14); along p psi decreases on [0, 1], CL use the same min{1,
//   alpha_k} in the proof of Theorem 3.11); s2 = alpha*[d], d = -(diag(|v|^-1/2)
//   D)^-2 g, tau the minimizer of psi(tau d) within the trust region and the
//   box; s = s1 where psi(s1) / psi(s2) > beta = 0.1, else s2 (CL Fig. 4.1,
//   step 2); rho = (1/2 ||f_+||^2 - 1/2 ||f||^2 + 1/2 s^T C s) / psi(s) (CL
//   §2.1), in step 3 instead of (4.4). OURS:
//   LMB-1. Delta's rules and the acceptance (rho > 1e-4) are Moré's of step 3,
//     not CL's Fig. 4.1 (mu = 0.25, gamma = 0.0625, 0.5, 2): so that the method
//     with every bound infinite is Moré's; Moré's rules keep within CL's
//     Algorithm 2 except that an accepted step with rho <= 1/4 may shrink Delta
//     below gamma_1 Delta (CL ask [gamma_1 Delta, Delta] there). mu (4.5) is
//     taken along s, with delta'(0) = g^T s.
//   LMB-2. The step-back theta = max(0.95, 1 - ||s||) where x + tau p (x + tau
//     d) is on the boundary (CL (2.13) ask theta in [theta_l, 1], 1 - theta =
//     O(||s||)), theta = 0.95 where rounding still puts a coordinate on the
//     boundary; theta = 1 inside.
//   LMB-3. x_0 is refused outside [l, u] or where l_i >= u_i; within 100 eps
//     max(1, |bound|) of a bound it is moved inside as in CL §4: to l_i + 0.1
//     (u_i - l_i) (u_i - 0.1 (u_i - l_i)), and where the other bound is
//     infinite to l_i + 0.1 max(1, |l_i|) (u_i - 0.1 max(1, |u_i|)).
//   LMB-4. xtol's ||D x|| and the first Delta take Moré's D alone; gtol's
//     cosines are those of [J; C^1/2] (which go to 0 at a bound that holds g
//     back); the forward difference steps back (x_j - h) where x_j + h > u_j.
//   LMB-5. lm_geodesic is refused with a finite bound (Transtrum & Sethna have
//     none).
//
// lm_geodesic (M.K. Transtrum, J.P. Sethna, "Improvements to the
// Levenberg-Marquardt algorithm for nonlinear least-squares minimization",
// arXiv:1201.5885, 2012, §3; source lm-geodesic-acceleration_transtrum2012_
// arxiv): the step v + a, v the step of 2 (their (12)), a = -1/2 (J^T J +
// lambda D^T D)^-1 J^T r'' (after their (13)), r'' the directional second
// derivative along v by their (19) with h = 0.1 (one evaluation more), the
// step taken only where 2 ||a|| / ||v|| <= alpha = 0.75 (their (15)).
//   LMG-1. lambda is chosen for v alone, as they do ("selecting lambda so
//     that |delta theta_1| = Delta"); the predicted reduction and Delta's
//     rules are v's, the actual reduction that of v + a.
//   LMG-2. Where (15) fails, Delta = 1/2 min(Delta, ||D v||) and the step is
//     made again, v + a not evaluated.
//   LMG-3. The norms of (15) are the scaled ||D .||, so that the method stays
//     invariant to the scale of the parameters, as they say it does.
//
// lm_loss = huber | cauchy (the corpus's nonlinear-least-squares §28-30):
// rho(t) = t^2/2 for |t| <= c, c (|t| - c/2) beyond (Huber), or c^2/2 log(1 +
// (t/c)^2) (Cauchy), c = f_scale; min sum rho(r_i) by iteratively reweighted
// least squares: the weights w_i = psi(r_i) / r_i at the last solution (1 /
// max(1, |r_i|/c), 1 / (1 + (r_i/c)^2)), then the method above on sqrt(w_i)
// r_i, until no weight moves by more than irls_tol (LM-6).
//   LM-6. OURS: irls_tol = 1e-10, 50 rounds at most, every round to the
//     tolerances above, the evaluations of all rounds counted together.
//
// varpro (separable problems, below): G.H. Golub, V. Pereyra, "Separable
// nonlinear least squares: the variable projection method and its
// applications", Inverse Problems 19 (2003) R1-R26 (source varpro_golub2003;
// "GP"), §2-3: r(a, alpha) = y - Phi(alpha) a; a = Phi^- y eliminated, the
// method above on the variable projection r_2(alpha) = P_Phi^perp y (GP (2)),
// its Jacobian column J_.j = -[P^perp dPhi_j Phi^- + (P^perp dPhi_j Phi^-)^T]
// y (GP §3), or with kaufman = true Kaufman's first term alone; Phi^- the
// symmetric generalized inverse of the pivoted QR (Phi Phi^- symmetric, Phi
// Phi^- Phi = Phi, which GP §3 say suffices; Moré (3.3) gives the same one
// for J). OURS:
//   VP-1. The pseudorank of Phi: the diagonal |R_jj| > max(m, p) eps |R_00|
//     (lsq.hpp's LSQ-6); a = Phi^- y is the basic solution on those columns.
//   VP-2. Without the derivatives of Phi, the Jacobian of r_2 is forward
//     differences of r_2 (the full projection, not Kaufman's term).
//   VP-3. The Jacobian's evaluation of Phi and its derivatives counts as one
//     Jacobian (njev), r_2's as one evaluation (nfev).
//
// LM-7, LM-8: the diagnostics (diagnose, below).
//
// Task 19, §19.3: a gauge basis (Problem::gauge; LMN, here), equalities
// (solve_eq; LMC) and profiles (profile; LMP), each with its readings below.
// None of them changes a run without them.
//
// A gauge basis (§19.3.1; a derivation of ours, the task names no paper):
// gauge(x) = N (n x k), the directions the residuals do not see to first
// order. Step 2 is solved on the complement of D N in the scaled variables:
// Q2 an orthonormal basis of it (the last columns of Q in D N P = Q R, from
// N's pseudorank on), the step p = D^-1 Q2 w, w from step 2 for the matrix
// J D^-1 Q2 with the identity for its scaling, so that ||D p|| = ||w|| and
// Moré's trust region, lambda, rho and the stopping tests are unchanged.
//   LMN-1. The metric is Moré's D: the step is orthogonal to N in D^2 (N^T
//     D^2 p = 0), plainly orthogonal where scale = false; D from J's own
//     columns as in step 1.
//   LMN-2. The diagnostics on the complement (diagnose_gauge) take the plain
//     inner product: the covariance of J Q2's parameters, carried back.
//   LMN-3. A gauge basis takes no finite bound and no lm_geodesic.
//   LMN-4. Result::gauge_k is N's pseudorank at the solution, gauge_ratio
//     ||J Q_N|| / ||J|| (Frobenius, Q_N an orthonormal basis of N's columns),
//     with a Jacobian at the solution (one more, counted).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "eigen.hpp"
#include "lsq.hpp"

namespace mootation::scalar::lm {

using Vec = std::vector<double>;
using lsq::Matrix;

inline constexpr double kEps = std::numeric_limits<double>::epsilon();
inline constexpr double kTiny = std::numeric_limits<double>::min();   // the smallest normal number
inline constexpr double kInf = std::numeric_limits<double>::infinity();

// The problem: m residuals of n parameters; the Jacobian (m x n) if known;
// the bounds (empty: none; entries may be -inf/+inf).
struct Problem {
    std::size_t n = 0, m = 0;
    std::function<void(const Vec& x, Vec& r)> residuals;
    std::function<void(const Vec& x, Matrix& J)> jacobian;   // empty: forward differences
    Vec lo, hi;
    // the directions (n x k) along which r does not change to first order
    // (task 19, §19.3.1, LMN): the step keeps to their complement
    std::function<void(const Vec& x, Matrix& N)> gauge;
};

struct Options {
    double ftol = 1e-8;                     // Moré §8 (LM-4)
    double xtol = 1e-8;
    double gtol = 0.0;
    double factor = 100.0;                  // LM-3
    double fd_eps = 0.0;                    // the differences' relative step is sqrt(max(fd_eps, eps))
    long long max_nfev = 0;                 // 0: 100 (n + 1) with the Jacobian, 200 (n + 1) without
    bool scale = true;                      // D by the column norms (Moré (6.3)); false: D = I
    bool geodesic = false;                  // lm_geodesic: Transtrum & Sethna's acceleration (LMG)
    double geodesic_alpha = 0.75;           // their (15): 2 ||D a|| / ||D v|| <= alpha
    double geodesic_h = 0.1;                // their (19)'s step, a share of v
    enum class Loss { linear, huber, cauchy } loss = Loss::linear;   // lm_loss
    double f_scale = 1.0;                   // c
    double irls_tol = 1e-10;                // LM-6
    int irls_max = 50;
    std::vector<Vec>* iterates = nullptr;   // where given, every accepted x is appended
};

enum class Status { ftol, xtol, ftol_xtol, gtol, max_nfev, ftol_eps, xtol_eps, gtol_eps };

struct Result {
    Vec x, r;
    double cost = 0.0;                      // 1/2 ||r||^2
    long long nfev = 0, njev = 0, iterations = 0;
    Status status = Status::max_nfev;
    std::size_t gauge_k = 0;                // with a gauge basis: its rank at x
    double gauge_ratio = 0.0;               // and ||J N|| / ||J|| there, N orthonormal (LMN-4)
};

// ||v||, scaled by the largest |entry| (LM-5)
inline double norm(const Vec& v)
{
    double big = 0.0;
    for (double e : v) big = std::max(big, std::fabs(e));
    if (big == 0.0 || !std::isfinite(big)) return big;
    double s = 0.0;
    for (double e : v) s += (e / big) * (e / big);
    return big * std::sqrt(s);
}

namespace detail {

// J P = Q R with column interchanges (LH (14.9), Steps 3-10): R in the upper
// triangle of a (n x n), perm[j] the column of J in place j, qtr the first n
// entries of Q^T r, colnorm the norms of J's columns.
struct Factor {
    Matrix R;                    // n x n
    std::vector<std::size_t> perm;
    Vec qtr, colnorm;
    std::vector<Vec> a;          // the transformed columns, the Householder vectors below R
    std::vector<lsq::House> house;

    // the first n entries of Q^T b
    Vec qt(Vec b) const
    {
        for (std::size_t j = 0; j < house.size(); ++j)
            lsq::house_apply(a[j].data(), 1, j, j + 1, b.size(), house[j], b.data(), 1);
        b.resize(house.size());
        return b;
    }
};

inline Factor factor(const Matrix& J, const Vec& r)
{
    const std::size_t m = J.rows, n = J.cols;
    std::vector<Vec> a(n, Vec(m));                    // a[j]: column j
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) a[j][i] = J(i, j);
    Factor f;
    f.colnorm.resize(n);
    for (std::size_t j = 0; j < n; ++j) f.colnorm[j] = norm(a[j]);
    f.perm.resize(n);
    for (std::size_t j = 0; j < n; ++j) f.perm[j] = j;
    Vec b = r;
    Vec h(n, 0.0);
    double hbar = 0.0;
    f.house.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
        bool again = j == 0;
        std::size_t lam = j;
        auto argmax = [&]() {
            std::size_t k = j;
            for (std::size_t l = j + 1; l < n; ++l)
                if (h[l] > h[k]) k = l;
            return k;
        };
        if (!again) {
            for (std::size_t l = j; l < n; ++l) h[l] -= a[l][j - 1] * a[l][j - 1];   // Step 4
            lam = argmax();
            again = !(hbar + 1e-3 * h[lam] > hbar);                                   // Step 6
        }
        if (again) {                                                                  // Steps 7-8
            for (std::size_t l = j; l < n; ++l) {
                double s = 0.0;
                for (std::size_t i = j; i < m; ++i) s += a[l][i] * a[l][i];
                h[l] = s;
            }
            lam = argmax();
            hbar = h[lam];
        }
        if (lam != j) {                                                               // Steps 9-10
            std::swap(a[j], a[lam]);
            std::swap(f.perm[j], f.perm[lam]);
            h[lam] = h[j];
        }
        f.house[j] = lsq::house_make(a[j].data(), 1, j, j + 1, m);
        for (std::size_t c = j + 1; c < n; ++c)
            lsq::house_apply(a[j].data(), 1, j, j + 1, m, f.house[j], a[c].data(), 1);
        lsq::house_apply(a[j].data(), 1, j, j + 1, m, f.house[j], b.data(), 1);
    }
    f.R = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = i; c < n; ++c) f.R(i, c) = a[c][i];
    f.qtr.assign(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(n));
    f.a = std::move(a);
    return f;
}

// min ||[J; sqrt(lambda) D] z - [r; 0]|| from J P = Q R (NW (10.38)): the rows
// of sqrt(lambda) D P removed against R by Givens rotations, the diagonal of
// sqrt(lambda) D taken in the order of R's columns. S (upper triangular) is
// what R became; z is in J's order of columns.
inline void solve_damped(const Factor& f, const Vec& qtb, const Vec& dl, Vec& z, Matrix& S)
{
    const std::size_t n = f.R.rows;
    S = f.R;
    Vec w = qtb;
    for (std::size_t j = 0; j < n; ++j) {
        const double d = dl[f.perm[j]];
        if (d == 0.0) continue;
        Vec row(n, 0.0);                              // the row d e_j of sqrt(lambda) D P
        row[j] = d;
        double rhs = 0.0;
        for (std::size_t k = j; k < n; ++k) {
            if (row[k] == 0.0) continue;
            double c, s;                              // the rotation that zeros row[k] against S_kk
            if (std::fabs(S(k, k)) < std::fabs(row[k])) {
                const double cot = S(k, k) / row[k];
                s = 0.5 / std::sqrt(0.25 + 0.25 * cot * cot);
                c = s * cot;
            } else {
                const double tan = row[k] / S(k, k);
                c = 0.5 / std::sqrt(0.25 + 0.25 * tan * tan);
                s = c * tan;
            }
            S(k, k) = c * S(k, k) + s * row[k];
            const double t = c * w[k] + s * rhs;
            rhs = -s * w[k] + c * rhs;
            w[k] = t;
            for (std::size_t i = k + 1; i < n; ++i) {
                const double u = c * S(k, i) + s * row[i];
                row[i] = -s * S(k, i) + c * row[i];
                S(k, i) = u;
            }
        }
    }
    // S y = w, the trailing part where S is singular set to 0
    std::size_t rank = n;
    for (std::size_t j = 0; j < n; ++j)
        if (S(j, j) == 0.0 && rank == n) rank = j;
    for (std::size_t j = rank; j < n; ++j) w[j] = 0.0;
    Vec y(n, 0.0);
    for (std::size_t jj = rank; jj-- > 0;) {
        double s = w[jj];
        for (std::size_t i = jj + 1; i < rank; ++i) s -= S(jj, i) * y[i];
        y[jj] = s / S(jj, jj);
    }
    z.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) z[f.perm[j]] = y[j];
}

// The Gauss-Newton step's z (R z = Q^T r, qtb, in R's leading nonsingular part)
inline Vec gauss_newton(const Factor& f, const Vec& qtb, std::size_t& rank)
{
    const std::size_t n = f.R.rows;
    rank = n;
    for (std::size_t j = 0; j < n; ++j)
        if (f.R(j, j) == 0.0 && rank == n) rank = j;
    Vec w = qtb;
    for (std::size_t j = rank; j < n; ++j) w[j] = 0.0;
    Vec y(n, 0.0);
    for (std::size_t jj = rank; jj-- > 0;) {
        double s = w[jj];
        for (std::size_t i = jj + 1; i < rank; ++i) s -= f.R(jj, i) * y[i];
        y[jj] = s / f.R(jj, jj);
    }
    Vec z(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) z[f.perm[j]] = y[j];
    return z;
}

// phi'(lambda) / ... : ||S^-T P^T D^T D z / ||D z|| ||^2 with S upper
// triangular (the Newton correction of NW (4.44), with D)
inline double newton_q(const Matrix& S, const std::vector<std::size_t>& perm, const Vec& D, const Vec& Dz,
                       double dznorm)
{
    const std::size_t n = S.rows;
    Vec q(n);
    for (std::size_t j = 0; j < n; ++j) q[j] = D[perm[j]] * (Dz[perm[j]] / dznorm);
    for (std::size_t j = 0; j < n; ++j) {             // S^T q' = q
        double s = q[j];
        for (std::size_t i = 0; i < j; ++i) s -= S(i, j) * q[i];
        q[j] = s / S(j, j);
    }
    return norm(q);
}

// lambda and the step z (the step is -z): lambda = 0 where the Gauss-Newton
// step has ||D z|| <= (1 + sigma) Delta (Moré (7.1)(a)), else Moré's (5.5)
// until |phi| <= sigma Delta (5.1) (step 2; LM-2). phi'(lambda) = -||D z||
// q^2, q = newton_q (Moré, after (5.5)).
inline double levenberg(const Factor& f, const Vec& D, double delta, double lambda, Vec& z)
{
    constexpr double sigma = 0.1;
    const std::size_t n = f.R.rows;
    std::size_t rank = 0;
    z = gauss_newton(f, f.qtr, rank);
    Vec Dz(n);
    for (std::size_t j = 0; j < n; ++j) Dz[j] = D[j] * z[j];
    double dznorm = norm(Dz);
    double phi = dznorm - delta;
    if (phi <= sigma * delta) return 0.0;
    double lower = 0.0;                               // l_0 = -phi(0) / phi'(0)
    if (rank == n) {
        const double q = newton_q(f.R, f.perm, D, Dz, dznorm);
        lower = phi / (dznorm * q * q);
    }
    Vec g(n);                                         // D^-1 J^T r, in R's order
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i <= j; ++i) s += f.R(i, j) * f.qtr[i];
        g[j] = s / D[f.perm[j]];
    }
    double upper = norm(g) / delta;                   // u_0
    if (upper == 0.0) upper = kTiny / std::min(delta, 0.1);
    Matrix S;
    for (int it = 1;; ++it) {
        if (!(lambda > lower && lambda < upper))      // (5.5)(a)
            lambda = std::max(1e-3 * upper, std::sqrt(lower * upper));
        Vec dl(n);
        for (std::size_t j = 0; j < n; ++j) dl[j] = std::sqrt(lambda) * D[j];
        solve_damped(f, f.qtr, dl, z, S);
        for (std::size_t j = 0; j < n; ++j) Dz[j] = D[j] * z[j];
        dznorm = norm(Dz);
        phi = dznorm - delta;
        if (std::fabs(phi) <= sigma * delta || it == 10) return lambda;
        const double q = newton_q(S, f.perm, D, Dz, dznorm);
        if (phi < 0.0) upper = lambda;                // (5.5)(b)
        lower = std::max(lower, lambda + phi / (dznorm * q * q));
        lambda = lambda + (phi / delta) / (q * q);    // (5.4)
    }
}

// z with (J^T J + lambda D^T D) z = J^T b for the b whose Q^T b is qtb
inline void damped(const Factor& f, const Vec& qtb, const Vec& D, double lambda, Vec& z)
{
    std::size_t rank = 0;
    if (lambda == 0.0) {
        z = gauss_newton(f, qtb, rank);
        return;
    }
    Vec dl(D.size());
    for (std::size_t j = 0; j < D.size(); ++j) dl[j] = std::sqrt(lambda) * D[j];
    Matrix S;
    solve_damped(f, qtb, dl, z, S);
}

// ── bounds (CL) ─────────────────────────────────────────────────────────────
// |v_i| and J^v_ii of CL's Definition 2.1 and (2.5) for g = J^T f: the
// distance to the bound g points at and its derivative; 1 and 0 where that
// bound is infinite; J^v_ii = 0 where g_i = 0.
inline void cl_v(const Vec& x, const Vec& g, const Vec& lo, const Vec& hi, Vec& dv, Vec& jv)
{
    const std::size_t n = x.size();
    dv.assign(n, 1.0);
    jv.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (g[i] < 0.0 && hi[i] < kInf) {             // (i)
            dv[i] = hi[i] - x[i];
            jv[i] = -1.0;
        } else if (g[i] >= 0.0 && lo[i] > -kInf) {    // (ii)
            dv[i] = x[i] - lo[i];
            jv[i] = 1.0;
        }
        if (g[i] == 0.0) jv[i] = 0.0;
    }
}

// CL (2.11): the largest tau with x + tau d in the box (inf where no bound is met)
inline double to_boundary(const Vec& x, const Vec& d, const Vec& lo, const Vec& hi)
{
    double a = kInf;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (d[i] > 0.0 && hi[i] < kInf) a = std::min(a, (hi[i] - x[i]) / d[i]);
        else if (d[i] < 0.0 && lo[i] > -kInf) a = std::min(a, (lo[i] - x[i]) / d[i]);
    }
    return a;
}

// psi(s) = g^T s + 1/2 (||J s||^2 + s^T C s) (CL (2.6)-(2.8))
inline double model(const Matrix& J, const Vec& g, const Vec& C, const Vec& s)
{
    double gs = 0.0, cs = 0.0;
    Vec Js(J.rows, 0.0);
    for (std::size_t j = 0; j < s.size(); ++j) {
        gs += g[j] * s[j];
        cs += C[j] * s[j] * s[j];
        for (std::size_t i = 0; i < J.rows; ++i) Js[i] += J(i, j) * s[j];
    }
    const double js = norm(Js);
    return gs + 0.5 * (js * js + cs);
}

// alpha*[d] (CL (2.12)-(2.14)): tau d, tau = min(tmax, the box's (2.11)), or
// psi's minimizer along d where it comes first (quadratic: slope = g^T d,
// curv = d^T (J^T J + C) d), with the step-back where the boundary is met
// (LMB-2).
inline Vec feasible(const Vec& x, const Vec& d, const Vec& lo, const Vec& hi, double tmax, bool quadratic,
                    double slope, double curv)
{
    const std::size_t n = x.size();
    const double box = to_boundary(x, d, lo, hi);
    double tau = std::min(tmax, box);
    if (quadratic) {
        if (!(slope < 0.0)) tau = 0.0;
        else if (curv > 0.0) tau = std::min(tau, -slope / curv);
    }
    Vec s(n);
    for (std::size_t i = 0; i < n; ++i) s[i] = tau * d[i];
    if (tau < box) return s;
    auto inside = [&](double theta) {
        for (std::size_t i = 0; i < n; ++i) {
            const double t = x[i] + theta * s[i];
            if (!(t > lo[i] && t < hi[i])) return false;
        }
        return true;
    };
    double theta = std::max(0.95, 1.0 - norm(s));
    if (!inside(theta)) theta = 0.95;
    for (double& e : s) e *= theta;
    return s;
}

}   // namespace detail

namespace detail {

// Q b for the J P = Q R of factor (all of Q, J's rows long)
inline Vec q_times(const Factor& f, Vec b)
{
    for (std::size_t j = f.house.size(); j-- > 0;)
        lsq::house_apply(f.a[j].data(), 1, j, j + 1, b.size(), f.house[j], b.data(), 1);
    return b;
}

// the pseudorank of R (lsq.hpp's LSQ-6: |R_jj| > max(m, n) eps |R_00|)
inline std::size_t pseudorank(const Factor& f, std::size_t rows)
{
    const std::size_t k = f.R.rows;
    if (k == 0 || f.R(0, 0) == 0.0) return 0;
    const double tau = static_cast<double>(std::max(rows, k)) * kEps * std::fabs(f.R(0, 0));
    std::size_t r = 0;
    while (r < k && std::fabs(f.R(r, r)) > tau) ++r;
    return r;
}

// An orthonormal basis (n x (n - rank)) of the complement of the columns of
// diag(D) N (n x k): the last columns of Q in D N P = Q R, from the
// pseudorank on; rank its pseudorank.
inline Matrix complement(const Matrix& N, const Vec& D, std::size_t& rank)
{
    const std::size_t n = N.rows, k = N.cols;
    Matrix DN(n, k);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < k; ++c) DN(i, c) = D[i] * N(i, c);
    const Factor f = factor(DN, Vec(n, 0.0));
    rank = pseudorank(f, n);
    Matrix Q2(n, n - rank);
    for (std::size_t c = rank; c < n; ++c) {
        Vec e(n, 0.0);
        e[c] = 1.0;
        const Vec q = q_times(f, e);
        for (std::size_t i = 0; i < n; ++i) Q2(i, c - rank) = q[i];
    }
    return Q2;
}

}   // namespace detail

// The forward-difference Jacobian at x, r = r(x) (n evaluations); backward
// where x_j + h passes the upper bound (LMB-4)
inline void fd_jacobian(const Problem& p, const Vec& x, const Vec& r, double fd_eps, Matrix& J)
{
    const double e = std::sqrt(std::max(fd_eps, kEps));
    J = Matrix(p.m, p.n);
    Vec xh = x, rh(p.m);
    for (std::size_t j = 0; j < p.n; ++j) {
        double h = e * std::fabs(x[j]);
        if (h == 0.0) h = e;
        if (!p.hi.empty() && x[j] + h > p.hi[j]) h = -h;
        xh[j] = x[j] + h;
        p.residuals(xh, rh);
        xh[j] = x[j];
        for (std::size_t i = 0; i < p.m; ++i) J(i, j) = (rh[i] - r[i]) / h;
    }
}

inline Result solve(const Problem& p, Vec x, const Options& o)
{
    const std::size_t n = p.n, m = p.m;
    bool bounded = false;                             // a finite bound somewhere
    if (!p.lo.empty() || !p.hi.empty()) {
        if (p.lo.size() != n || p.hi.size() != n) throw std::invalid_argument("lm: lo and hi need n entries");
        for (std::size_t i = 0; i < n; ++i) {
            if (!(p.lo[i] < p.hi[i])) throw std::invalid_argument("lm: lo_i < hi_i is needed");
            if (!(x[i] >= p.lo[i] && x[i] <= p.hi[i])) throw std::invalid_argument("lm: x0 is outside the bounds");
            bounded = bounded || p.lo[i] > -kInf || p.hi[i] < kInf;
        }
    }
    if (bounded && o.geodesic) throw std::invalid_argument("lm: lm_geodesic takes no finite bounds (LMB-5)");
    const bool gauged = static_cast<bool>(p.gauge);
    if (gauged && (bounded || o.geodesic))
        throw std::invalid_argument("lm: a gauge basis takes no finite bounds and no lm_geodesic (LMN-3)");
    if (bounded)                                      // strictly inside (LMB-3)
        for (std::size_t i = 0; i < n; ++i) {
            const double l = p.lo[i], u = p.hi[i];
            if (l > -kInf && x[i] < l + 100.0 * kEps * std::max(1.0, std::fabs(l)))
                x[i] = u < kInf ? l + 0.1 * (u - l) : l + 0.1 * std::max(1.0, std::fabs(l));
            else if (u < kInf && x[i] > u - 100.0 * kEps * std::max(1.0, std::fabs(u)))
                x[i] = l > -kInf ? u - 0.1 * (u - l) : u - 0.1 * std::max(1.0, std::fabs(u));
        }
    const long long max_nfev =
        o.max_nfev > 0 ? o.max_nfev : (p.jacobian ? 100 : 200) * static_cast<long long>(n + 1);
    Result res;
    Vec r(m), rt(m), xt(n), D(n, 1.0), z, s(n), Ds(n), Dx(n);
    Vec g(n, 0.0), dv(n, 1.0), jv, C(n, 0.0), Dt(n);   // CL's g, |v|, J^v, C; Dt the trust region's scaling
    p.residuals(x, r);
    res.nfev = 1;
    double rnorm = norm(r);
    double lambda = 0.0, delta = 0.0, xnorm = 0.0;
    Matrix J, Ja, Q2, Jw;                             // Q2, Jw: the gauge's complement and J D^-1 Q2
    Vec zf;                                           // the step in the factored matrix's columns
    bool done = false;
    for (long long iter = 1; !done; ) {
        if (p.jacobian) {
            J = Matrix(m, n);
            p.jacobian(x, J);
            ++res.njev;
        } else {
            fd_jacobian(p, x, r, o.fd_eps, J);
            res.nfev += static_cast<long long>(n);
        }
        Vec colnorm(n), ra = r;                       // J's column norms (Moré's D)
        if (bounded) {                                // [J; C^1/2] and [f; 0]
            for (std::size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                Vec col(m);
                for (std::size_t i = 0; i < m; ++i) {
                    sum += J(i, j) * r[i];
                    col[i] = J(i, j);
                }
                g[j] = sum;
                colnorm[j] = norm(col);
            }
            detail::cl_v(x, g, p.lo, p.hi, dv, jv);
            Ja = Matrix(m + n, n);
            for (std::size_t j = 0; j < n; ++j) {
                C[j] = g[j] * jv[j] / dv[j];
                for (std::size_t i = 0; i < m; ++i) Ja(i, j) = J(i, j);
                Ja(m + j, j) = std::sqrt(C[j]);
            }
            ra.resize(m + n, 0.0);
        } else if (gauged) {                          // LMN-1: J D^-1 Q2, D from J's own columns
            for (std::size_t j = 0; j < n; ++j) {
                Vec col(m);
                for (std::size_t i = 0; i < m; ++i) col[i] = J(i, j);
                colnorm[j] = norm(col);
            }
            if (iter == 1) {
                if (o.scale)
                    for (std::size_t j = 0; j < n; ++j) D[j] = colnorm[j] == 0.0 ? 1.0 : colnorm[j];
                for (std::size_t j = 0; j < n; ++j) Dx[j] = D[j] * x[j];
                xnorm = norm(Dx);
                delta = xnorm > 0.0 ? o.factor * xnorm : o.factor;
            }
            if (o.scale)
                for (std::size_t j = 0; j < n; ++j) D[j] = std::max(D[j], colnorm[j]);
            Matrix N;
            p.gauge(x, N);
            if (N.rows != n) throw std::invalid_argument("lm: the gauge basis needs n rows");
            std::size_t rank = 0;
            Q2 = detail::complement(N, D, rank);
            Jw = Matrix(m, n - rank);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t c = 0; c < n - rank; ++c) {
                    double sum = 0.0;
                    for (std::size_t j = 0; j < n; ++j) sum += (J(i, j) / D[j]) * Q2(j, c);
                    Jw(i, c) = sum;
                }
        }
        const detail::Factor f = detail::factor(bounded ? Ja : gauged ? Jw : J, ra);
        const std::size_t nf = f.perm.size();         // n, or n - rank with a gauge basis
        if (!bounded && !gauged) colnorm = f.colnorm;
        if (iter == 1 && !gauged) {
            if (o.scale)
                for (std::size_t j = 0; j < n; ++j) D[j] = colnorm[j] == 0.0 ? 1.0 : colnorm[j];
            for (std::size_t j = 0; j < n; ++j) Dx[j] = D[j] * x[j];
            xnorm = norm(Dx);
            delta = xnorm > 0.0 ? o.factor * xnorm : o.factor;
        }
        double gcos = 0.0;                            // the gtol test's largest cosine
        if (rnorm != 0.0)
            for (std::size_t j = 0; j < nf; ++j) {
                const std::size_t l = f.perm[j];
                if (f.colnorm[l] == 0.0) continue;
                double sum = 0.0;
                for (std::size_t i = 0; i <= j; ++i) sum += f.R(i, j) * (f.qtr[i] / rnorm);
                gcos = std::max(gcos, std::fabs(sum / f.colnorm[l]));
            }
        if (gcos <= o.gtol) {
            res.status = Status::gtol;
            break;
        }
        if (o.scale && !gauged)
            for (std::size_t j = 0; j < n; ++j) D[j] = std::max(D[j], colnorm[j]);
        for (std::size_t j = 0; j < n; ++j) Dt[j] = bounded ? D[j] / std::sqrt(dv[j]) : D[j];
        for (;;) {                                     // until a step is taken
            lambda = detail::levenberg(f, gauged ? Vec(nf, 1.0) : Dt, delta, lambda, zf);
            z = zf;
            if (gauged) {                             // p = D^-1 Q2 w (LMN-1)
                z.assign(n, 0.0);
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t c = 0; c < nf; ++c) sum += Q2(j, c) * zf[c];
                    z[j] = sum / D[j];
                }
            }
            bool step_is_p = true;
            for (std::size_t j = 0; j < n; ++j) s[j] = -z[j];
            if (bounded) {                            // CL Fig. 4.1, step 2
                const Vec s1 = detail::feasible(x, s, p.lo, p.hi, 1.0, false, 0.0, 0.0);
                Vec d(n), Dd(n), Jd(m, 0.0);
                double slope = 0.0, cd = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    d[j] = -g[j] / (Dt[j] * Dt[j]);
                    Dd[j] = Dt[j] * d[j];
                    slope += g[j] * d[j];
                    cd += C[j] * d[j] * d[j];
                    for (std::size_t i = 0; i < m; ++i) Jd[i] += J(i, j) * d[j];
                }
                const double jd = norm(Jd);
                const Vec s2 = detail::feasible(x, d, p.lo, p.hi, delta / norm(Dd), true, slope, jd * jd + cd);
                const double psi1 = detail::model(J, g, C, s1), psi2 = detail::model(J, g, C, s2);
                step_is_p = !(psi2 < 0.0) || psi1 / psi2 > 0.1;
                s = step_is_p ? s1 : s2;
            }
            for (std::size_t j = 0; j < n; ++j) {
                xt[j] = x[j] + s[j];
                Ds[j] = Dt[j] * s[j];
            }
            const double pnorm = norm(Ds);
            if (iter == 1) delta = std::min(delta, pnorm);
            if (o.geodesic) {                         // LMG-1..3 (no finite bound here)
                const double h = o.geodesic_h;
                for (std::size_t j = 0; j < n; ++j) xt[j] = x[j] - h * z[j];
                p.residuals(xt, rt);
                ++res.nfev;
                Vec r2(m);                            // (19): r'' along v = -z
                for (std::size_t i = 0; i < m; ++i) {
                    double Jv = 0.0;
                    for (std::size_t j = 0; j < n; ++j) Jv -= J(i, j) * z[j];
                    r2[i] = (2.0 / h) * ((rt[i] - r[i]) / h - Jv);
                }
                Vec z2;                               // the acceleration's step is -z2 / 2
                detail::damped(f, f.qt(r2), D, lambda, z2);
                Vec Da(n);
                for (std::size_t j = 0; j < n; ++j) Da[j] = D[j] * (0.5 * z2[j]);
                if (!(2.0 * norm(Da) <= o.geodesic_alpha * pnorm)) {   // (15) fails (LMG-2)
                    delta = 0.5 * std::min(delta, pnorm);
                    if (res.nfev >= max_nfev) {
                        res.status = Status::max_nfev;
                        done = true;
                        break;
                    }
                    continue;
                }
                for (std::size_t j = 0; j < n; ++j) xt[j] = x[j] - z[j] - 0.5 * z2[j];
            }
            p.residuals(xt, rt);
            ++res.nfev;
            const double rtnorm = norm(rt);
            double predicted, gamma;                  // Moré (4.4)'s denominator, (4.5)'s gamma
            if (!bounded) {
                Vec Rp(nf, 0.0);                      // R P^T p: its norm is ||J p||
                for (std::size_t j = 0; j < nf; ++j) {
                    const double t = zf[f.perm[j]];
                    for (std::size_t i = 0; i <= j; ++i) Rp[i] += f.R(i, j) * t;
                }
                const double t1 = norm(Rp) / rnorm, t2 = std::sqrt(lambda) * pnorm / rnorm;
                predicted = t1 * t1 + 2.0 * (t2 * t2);
                gamma = -(t1 * t1 + t2 * t2);
            } else {                                  // -2 psi(s) / ||f||^2, g^T s / ||f||^2
                predicted = (-2.0 * detail::model(J, g, C, s) / rnorm) / rnorm;
                double gs = 0.0;
                for (std::size_t j = 0; j < n; ++j) gs += g[j] * s[j];
                gamma = (gs / rnorm) / rnorm;
            }
            const double reduced = 1.0 - (rtnorm / rnorm) * (rtnorm / rnorm);
            double ratio = 0.0;                       // 0 where ||f_+|| > ||f|| (Moré §4)
            if (rtnorm <= rnorm && predicted != 0.0) {
                double actual = reduced;
                if (bounded) {                        // + 1/2 s^T C s (CL §2.1)
                    double cs = 0.0;
                    for (std::size_t j = 0; j < n; ++j) cs += C[j] * s[j] * s[j];
                    actual -= (cs / rnorm) / rnorm;
                }
                ratio = actual / predicted;
            }
            if (ratio <= 0.25) {                      // Moré §4, (4.5)
                double mu = 0.1;
                if (rtnorm <= rnorm) mu = 0.5;
                else if (rtnorm <= 10.0 * rnorm) mu = 0.5 * gamma / (gamma + 0.5 * reduced);
                mu = std::min(0.5, std::max(0.1, mu));
                delta = mu * delta;
                if (!bounded && lambda == 0.0 && ratio <= 1e-4)   // the same point again (LM-3b)
                    while (pnorm > 0.0 && pnorm <= 1.1 * delta && !(delta <= o.xtol * xnorm)) delta *= mu;
            } else if (ratio >= 0.75 || (lambda == 0.0 && step_is_p)) {
                delta = 2.0 * pnorm;                  // Moré (7.1)(d)
            }
            if (ratio > 1e-4) {                       // Moré (7.1)(c)
                x = xt;
                r = rt;
                for (std::size_t j = 0; j < n; ++j) Dx[j] = D[j] * x[j];
                xnorm = norm(Dx);
                rnorm = rtnorm;
                ++iter;
                ++res.iterations;
                if (o.iterates) o.iterates->push_back(x);
            }
            const bool f_ok = predicted <= o.ftol;    // Moré (8.4)
            const bool x_ok = delta <= o.xtol * xnorm;   // Moré (8.3)
            if (f_ok || x_ok) {
                res.status = f_ok && x_ok ? Status::ftol_xtol : f_ok ? Status::ftol : Status::xtol;
                done = true;
            } else {                                  // the later test wins
                if (res.nfev >= max_nfev) { res.status = Status::max_nfev; done = true; }
                if (predicted <= kEps) { res.status = Status::ftol_eps; done = true; }
                if (delta <= kEps * xnorm) { res.status = Status::xtol_eps; done = true; }
                if (gcos <= kEps) { res.status = Status::gtol_eps; done = true; }
            }
            if (done || ratio > 1e-4) break;
        }
    }
    res.x = x;
    res.r = r;
    res.cost = 0.5 * rnorm * rnorm;
    if (gauged) {                                     // LMN-4, at x with a Jacobian of its own
        if (p.jacobian) {
            J = Matrix(m, n);
            p.jacobian(x, J);
            ++res.njev;
        } else {
            fd_jacobian(p, x, r, o.fd_eps, J);
            res.nfev += static_cast<long long>(n);
        }
        Matrix N;
        p.gauge(x, N);
        const detail::Factor fn = detail::factor(N, Vec(n, 0.0));
        res.gauge_k = detail::pseudorank(fn, n);
        double jn = 0.0, jj = 0.0;
        for (std::size_t c = 0; c < res.gauge_k; ++c) {
            Vec e(n, 0.0);
            e[c] = 1.0;
            const Vec q = detail::q_times(fn, e);     // an orthonormal column of span N
            for (std::size_t i = 0; i < m; ++i) {
                double sum = 0.0;
                for (std::size_t j = 0; j < n; ++j) sum += J(i, j) * q[j];
                jn += sum * sum;
            }
        }
        for (double v : J.a) jj += v * v;
        res.gauge_ratio = jj > 0.0 ? std::sqrt(jn / jj) : 0.0;
    }
    return res;
}

// rho(t) of the loss, and the weight psi(t) / t
inline double loss_of(Options::Loss loss, double c, double t)
{
    const double a = std::fabs(t);
    if (loss == Options::Loss::huber) return a <= c ? 0.5 * t * t : c * (a - 0.5 * c);
    if (loss == Options::Loss::cauchy) {             // log1p(u) by dm::log, as D. Goldberg, "What every
        const double u = (t / c) * (t / c), w = 1.0 + u;   // computer scientist should know about floating-point
        return 0.5 * c * c * (w == 1.0 ? u : dm::log(w) * u / (w - 1.0));   // arithmetic", 1991, Theorem 4
    }
    return 0.5 * t * t;
}

inline double weight_of(Options::Loss loss, double c, double t)
{
    if (loss == Options::Loss::huber) return std::fabs(t) <= c ? 1.0 : c / std::fabs(t);
    if (loss == Options::Loss::cauchy) return 1.0 / (1.0 + (t / c) * (t / c));
    return 1.0;
}

// min sum rho(r_i(x)) (lm_loss) by reweighted least squares (LM-6); cost is
// sum rho(r_i), r the residuals themselves
inline Result solve_robust(const Problem& p, Vec x, const Options& o)
{
    if (o.loss == Options::Loss::linear) return solve(p, std::move(x), o);
    const std::size_t m = p.m;
    Vec w(m, 1.0), sw(m, 1.0);
    Problem q = p;
    q.residuals = [&](const Vec& v, Vec& r) {
        p.residuals(v, r);
        for (std::size_t i = 0; i < m; ++i) r[i] *= sw[i];
    };
    if (p.jacobian)
        q.jacobian = [&](const Vec& v, Matrix& J) {
            p.jacobian(v, J);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < p.n; ++j) J(i, j) *= sw[i];
        };
    Result total;
    Options oi = o;
    Vec r(m);
    for (int round = 0; round < o.irls_max; ++round) {
        const long long cap = o.max_nfev > 0 ? o.max_nfev : (p.jacobian ? 100 : 200) * static_cast<long long>(p.n + 1);
        oi.max_nfev = std::max(1LL, cap - total.nfev);
        const Result step = solve(q, x, oi);
        total.nfev += step.nfev;
        total.njev += step.njev;
        total.iterations += step.iterations;
        total.status = step.status;
        x = step.x;
        p.residuals(x, r);
        ++total.nfev;
        double moved = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double wi = weight_of(o.loss, o.f_scale, r[i]);
            moved = std::max(moved, std::fabs(wi - w[i]));
            w[i] = wi;
            sw[i] = std::sqrt(wi);
        }
        if (moved <= o.irls_tol || step.status == Status::max_nfev || total.nfev >= cap) break;
    }
    total.x = x;
    total.r = r;
    total.cost = 0.0;
    for (double v : r) total.cost += loss_of(o.loss, o.f_scale, v);
    return total;
}

// ── diagnostics at the solution ─────────────────────────────────────────────
// The singular values of J (= those of R in J P = Q R), largest first, from
// the eigenvalues of the symmetric [[0, R], [R^T, 0]], which are +-sigma_i (the
// Jordan-Wielandt matrix; eigen.hpp, LH Ch. 18): sigma_i to eps sigma_max
// absolutely, as an SVD gives them, not to eps sigma_max^2 / sigma_i as the
// eigenvalues of J^T J would. small: the sigma_i <= tau sigma_max, tau =
// max(m, n) eps for the caller's Jacobian (HFTI's pseudorank rule, lsq.hpp
// LSQ-6) and the differences' relative step sqrt(max(fd_eps, eps)) for theirs,
// whose columns are only that accurate (LM-7). The covariance of the
// parameters is s^2 (J^T J)^+ over the other sigma_i (the pseudo-inverse: the
// small directions left out), s^2 the noise variance given, or 2 cost /
// (m - n) where none is (LM-8); the correlations C_ij / sqrt(C_ii C_jj).
struct Diagnostics {
    Vec sv;                                 // singular values of J, decreasing
    std::size_t small = 0;                  // of them almost zero
    double sigma2 = 0.0;                    // the variance the covariance takes
    bool sigma2_given = false;
    Matrix cov, corr;                       // n x n
};

inline Diagnostics diagnose(const Matrix& J, double cost, double sigma2, bool differences = false,
                            double fd_eps = 0.0)
{
    const std::size_t m = J.rows, n = J.cols;
    const detail::Factor f = detail::factor(J, Vec(m, 0.0));
    Matrix H(2 * n, 2 * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) {
            H(i, n + j) = f.R(i, j);
            H(n + j, i) = f.R(i, j);
        }
    Vec w;
    Matrix V;
    eigen::symmetric(H, w, V);
    Diagnostics d;
    d.sigma2_given = sigma2 > 0.0;
    d.sigma2 = d.sigma2_given ? sigma2 : m > n ? 2.0 * cost / static_cast<double>(m - n) : 0.0;
    std::vector<Vec> v;                     // right singular vectors of J, in J's order of columns
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t e = 2 * n - 1 - k;            // the k-th largest eigenvalue
        d.sv.push_back(std::max(0.0, w[e]));
        Vec vk(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) vk[f.perm[j]] = V(n + j, e);
        const double s = norm(vk);
        if (s > 0.0)
            for (double& x : vk) x /= s;
        v.push_back(vk);
    }
    const double rel = differences ? std::sqrt(std::max(fd_eps, kEps)) : static_cast<double>(std::max(m, n)) * kEps;
    const double tau = rel * (n ? d.sv[0] : 0.0);
    d.cov = Matrix(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        if (d.sv[k] <= tau) {
            ++d.small;
            continue;
        }
        const double c = d.sigma2 / (d.sv[k] * d.sv[k]);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) d.cov(i, j) += c * v[k][i] * v[k][j];
    }
    d.corr = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double s = std::sqrt(d.cov(i, i) * d.cov(j, j));
            d.corr(i, j) = s > 0.0 ? d.cov(i, j) / s : 0.0;
        }
    return d;
}

// ── the user's Jacobian against forward differences ─────────────────────────
// Column j's relative difference ||J_:j - J^fd_:j|| / max(||J^fd_:j||, tiny);
// worst: the columns in decreasing order of it. n evaluations of r.
struct JacobianCheck {
    Vec column_error;
    std::vector<std::size_t> worst;
};

inline JacobianCheck check_jacobian(const Problem& p, const Vec& x, double fd_eps = 0.0)
{
    Vec r(p.m);
    p.residuals(x, r);
    Matrix J(p.m, p.n), F;
    p.jacobian(x, J);
    fd_jacobian(p, x, r, fd_eps, F);
    JacobianCheck c;
    for (std::size_t j = 0; j < p.n; ++j) {
        Vec d(p.m), e(p.m);
        for (std::size_t i = 0; i < p.m; ++i) {
            d[i] = J(i, j) - F(i, j);
            e[i] = F(i, j);
        }
        c.column_error.push_back(norm(d) / std::max(norm(e), kTiny));
        c.worst.push_back(j);
    }
    std::stable_sort(c.worst.begin(), c.worst.end(),
                     [&](std::size_t a, std::size_t b) { return c.column_error[a] > c.column_error[b]; });
    return c;
}

// ── separable problems: variable projection (GP; VP-1..3) ───────────────────
// r(a, alpha) = y - Phi(alpha) a, Phi m x p, alpha k parameters (bounded as in
// Problem); dPhi[j] = d Phi / d alpha_j.
struct Separable {
    std::size_t m = 0, k = 0, p = 0;
    Vec y;
    std::function<void(const Vec& alpha, Matrix& Phi)> basis;
    std::function<void(const Vec& alpha, std::vector<Matrix>& dPhi)> derivatives;   // empty: VP-2
    Vec lo, hi;
};

struct SeparableResult {
    Result fit;                             // on alpha; r the variable projection r_2
    Vec a;                                  // Phi^- y at the solution
    std::size_t rank = 0;                   // Phi's pseudorank there
};

namespace detail {

// Phi P = Q [T S; 0 0] with T of the pseudorank (VP-1)
struct Projection {
    Factor f;
    std::size_t m = 0, rank = 0;

    explicit Projection(const Matrix& Phi) : f(factor(Phi, Vec(Phi.rows, 0.0))), m(Phi.rows)
    {
        const std::size_t p = f.R.rows;
        const double tau = static_cast<double>(std::max(m, p)) * kEps * (p ? std::fabs(f.R(0, 0)) : 0.0);
        while (rank < p && std::fabs(f.R(rank, rank)) > tau) ++rank;
    }
    Vec qt(Vec v) const                     // Q^T v, all m entries
    {
        for (std::size_t j = 0; j < f.house.size(); ++j)
            lsq::house_apply(f.a[j].data(), 1, j, j + 1, m, f.house[j], v.data(), 1);
        return v;
    }
    Vec q(Vec v) const                      // Q v
    {
        for (std::size_t j = f.house.size(); j-- > 0;)
            lsq::house_apply(f.a[j].data(), 1, j, j + 1, m, f.house[j], v.data(), 1);
        return v;
    }
    Vec perp(const Vec& v) const            // P^perp v = v - Phi Phi^- v
    {
        Vec w = qt(v);
        for (std::size_t i = 0; i < rank; ++i) w[i] = 0.0;
        return q(w);
    }
    Vec coef(const Vec& v) const            // Phi^- v = P [T^-1 0; 0 0] Q^T v
    {
        const Vec w = qt(v);
        Vec t(rank);
        for (std::size_t jj = rank; jj-- > 0;) {
            double s = w[jj];
            for (std::size_t i = jj + 1; i < rank; ++i) s -= f.R(jj, i) * t[i];
            t[jj] = s / f.R(jj, jj);
        }
        Vec a(f.R.rows, 0.0);
        for (std::size_t j = 0; j < rank; ++j) a[f.perm[j]] = t[j];
        return a;
    }
    Vec coef_t(const Vec& u) const          // Phi^-T u = Q [T^-T (P^T u)_1..rank; 0]
    {
        Vec w(m, 0.0);
        for (std::size_t j = 0; j < rank; ++j) {
            double s = u[f.perm[j]];
            for (std::size_t i = 0; i < j; ++i) s -= f.R(i, j) * w[i];
            w[j] = s / f.R(j, j);
        }
        return q(w);
    }
};

}   // namespace detail

inline SeparableResult varpro(const Separable& sp, Vec alpha, const Options& o, bool kaufman = false)
{
    const std::size_t m = sp.m, k = sp.k;
    if (sp.p > m) throw std::invalid_argument("varpro: Phi needs p <= m");
    auto phi_at = [&](const Vec& al) {
        Matrix Phi(m, sp.p);
        sp.basis(al, Phi);
        return Phi;
    };
    Problem pr;
    pr.n = k;
    pr.m = m;
    pr.lo = sp.lo;
    pr.hi = sp.hi;
    pr.residuals = [&](const Vec& al, Vec& r) { r = detail::Projection(phi_at(al)).perp(sp.y); };   // GP (2)
    if (sp.derivatives)
        pr.jacobian = [&](const Vec& al, Matrix& J) {                                           // GP §3
            const detail::Projection P(phi_at(al));
            std::vector<Matrix> dPhi(k, Matrix(m, sp.p));
            sp.derivatives(al, dPhi);
            const Vec a = P.coef(sp.y), r2 = P.perp(sp.y);
            for (std::size_t j = 0; j < k; ++j) {
                Vec da(m, 0.0), dr(sp.p, 0.0);        // dPhi_j a, dPhi_j^T r_2
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t l = 0; l < sp.p; ++l) {
                        da[i] += dPhi[j](i, l) * a[l];
                        dr[l] += dPhi[j](i, l) * r2[i];
                    }
                const Vec L = P.perp(da);
                const Vec K = kaufman ? Vec(m, 0.0) : P.coef_t(dr);
                for (std::size_t i = 0; i < m; ++i) J(i, j) = -(L[i] + K[i]);
            }
        };
    SeparableResult out;
    out.fit = solve_robust(pr, std::move(alpha), o);
    const detail::Projection P(phi_at(out.fit.x));
    out.a = P.coef(sp.y);
    out.rank = P.rank;
    return out;
}


// ── diagnostics on the gauge's complement (task 19, §19.3.1; LMN-2) ─────────
// diagnose() of J Q2, Q2 an orthonormal basis of the complement of N's columns
// (the plain inner product), its covariance carried back: cov = Q2 cov_Q2
// Q2^T (n x n, rank n - k at most); sv the singular values of J Q2.
inline Diagnostics diagnose_gauge(const Matrix& J, const Matrix& N, double cost, double sigma2,
                                  bool differences = false, double fd_eps = 0.0)
{
    const std::size_t m = J.rows, n = J.cols;
    std::size_t rank = 0;
    const Matrix Q2 = detail::complement(N, Vec(n, 1.0), rank);
    const std::size_t nc = n - rank;
    Matrix Jc(m, nc);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t c = 0; c < nc; ++c) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) sum += J(i, j) * Q2(j, c);
            Jc(i, c) = sum;
        }
    Diagnostics dc = diagnose(Jc, cost, sigma2, differences, fd_eps);
    Diagnostics d = dc;
    d.cov = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t a = 0; a < nc; ++a)
                for (std::size_t b = 0; b < nc; ++b) sum += Q2(i, a) * dc.cov(a, b) * Q2(j, b);
            d.cov(i, j) = sum;
        }
    d.corr = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double s = std::sqrt(d.cov(i, i) * d.cov(j, j));
            d.corr(i, j) = s > 0.0 ? d.cov(i, j) / s : 0.0;
        }
    return d;
}

// ── equalities (task 19, §19.3.2; LMC) ──────────────────────────────────────
// min 1/2 ||r(x)||^2 subject to C x = d, h(x) = 0 and the problem's bounds.
// Linear equalities alone and no finite bound: exact elimination, x = x_p +
// Z y, Z from the QR of C^T, lm over y (LMC-4). Otherwise every equality, the
// linear ones as rows C x - d, joins c(x) = (C x - d, h(x)) and the bound-
// constrained Lagrangian method of J. Nocedal, S.J. Wright, Numerical
// Optimization, 2nd ed., Springer 2006 (source numerical-optimization_
// nocedal2006), §17.4, Algorithm 17.4, solves it: its subproblem (17.50), min
// L_A over l <= x <= u, is lm with the bounds on the residuals of least-squares
// form r~ = [r; sqrt(mu) (c + lambda / mu)], 1/2 ||r~||^2 = 1/2 ||r||^2 +
// lambda^T c + (mu/2) ||c||^2 + ||lambda||^2 / (2 mu), which is N&W's (17.36)
// L_A with f = 1/2 ||r||^2 and lambda = -lambda_N&W (LMC-1); then, with the
// projected gradient ||x - P(x - grad L_A)|| of (17.51):
//   mu_0 = 10, omega_0 = 1 / mu_0, eta_0 = 1 / mu_0^0.1;
//   ||c|| <= eta_k: stop where ||c|| <= eta_* and the projected gradient <=
//     omega_*; lambda += mu c (N&W's (17.39) in our sign), eta /= mu^0.9,
//     omega /= mu;
//   else mu *= 100, eta = 1 / mu^0.1, omega = 1 / mu.
// OURS (N&W's 17.3 leaves mu's growth and the tolerances open, 17.4 fixes
// them; the constants 0.1, 0.9 and 100 are theirs, "to some extent
// arbitrary"):
//   LMC-1. The multipliers' sign: the task's form r~ = [r; sqrt(mu) (c +
//     lambda/mu)] is N&W's with lambda = -lambda_N&W; at a solution J^T r +
//     A^T lambda = 0, A the Jacobian of c.
//   LMC-2. The subproblem is solved by lm to its own tests (ftol, xtol); where
//     the projected gradient is still above omega_k s, lm again from there
//     with ftol and xtol a hundredth (not below eps), three times at most.
//   LMC-3. omega_k and omega_* are relative, times s = max(1, ||J^T r||) at
//     x_0, so that the method does not hang on the residuals' scale; eta_*
//     = 1e-8 (on ||c||, the equalities' own scale) and omega_* = 1e-6 by
//     default; 50 outer iterations at most, mu not beyond 1e15 (the run ends
//     unconverged there). N&W leave eta_* and omega_* to the user. Why not
//     tighter: the subproblem's cost carries 1/2 ||r||^2 whole, so a c below
//     about (eps cost / mu)^1/2 changes it by less than its rounding and lm
//     cannot see it; mu then grows and the gradient's mu c term is rounding
//     too (min ||x - a|| on the sphere, tests/test_lm.cpp, stalls with
//     1e-10 and 1e-8).
//   LMC-4. Exact elimination only without a finite bound (the owner's
//     decision H2 of 2026-11-01): a box in x is no box in y. C^T P = Q R with
//     column interchanges (lm's factor), C's pseudorank by lsq.hpp's LSQ-6;
//     x_p = Q_1 R^-T P^T d, the least-norm solution, refused where C x_p is
//     not d to 1e3 eps of the scale (inconsistent equalities). y is counted
//     from x_0's projection onto C x = d, x = x_0p + Z y, and lm starts at y =
//     0 exactly: counted from x_p, a start on C x = d would have y_0 of
//     rounding size, Moré's first Delta (LM-3) and the differences' steps
//     relative to it.
//   LMC-5. After elimination the multipliers are the least-squares solution of
//     C^T lambda = -J^T r at x (one more Jacobian).
struct Equalities {
    Matrix C;                               // p x n (p = 0: none)
    Vec d;
    std::size_t q = 0;                      // h's values (0: none)
    std::function<void(const Vec& x, Vec& h)> h;
    std::function<void(const Vec& x, Matrix& A)> jacobian;   // q x n; empty: forward differences
};

struct EqOptions {
    double eta = 1e-8;                      // eta_* (LMC-3)
    double omega = 1e-6;                    // omega_*, relative (LMC-3)
    int max_outer = 50;
    double mu_max = 1e15;                   // no growth beyond (LMC-3)
    Vec lambda0;                            // empty: 0
};

struct EqResult {
    Result fit;                             // x; r and cost the problem's own; the counts of every lm run
    Vec lambda;                             // the multipliers (LMC-1), the linear rows first
    double mu = 0.0;
    double c_norm = 0.0;                    // ||c(x)||
    double projected_gradient = 0.0;        // of L_A at x, over s (LMC-3)
    int outer = 0;
    long long ceval = 0;                    // evaluations of h
    bool eliminated = false, converged = false;
};

namespace detail {

// J at x (the problem's or forward differences), r = r(x); counts into res
inline void jacobian_at(const Problem& p, const Vec& x, const Vec& r, double fd_eps, Matrix& J, Result& res)
{
    if (p.jacobian) {
        J = Matrix(p.m, p.n);
        p.jacobian(x, J);
        ++res.njev;
    } else {
        fd_jacobian(p, x, r, fd_eps, J);
        res.nfev += static_cast<long long>(p.n);
    }
}

inline Vec gradient_of(const Matrix& J, const Vec& r)
{
    Vec g(J.cols, 0.0);
    for (std::size_t j = 0; j < J.cols; ++j)
        for (std::size_t i = 0; i < J.rows; ++i) g[j] += J(i, j) * r[i];
    return g;
}

inline void add_counts(Result& total, const Result& part)
{
    total.nfev += part.nfev;
    total.njev += part.njev;
    total.iterations += part.iterations;
    total.status = part.status;
}

inline EqResult eliminate(const Problem& p, const Equalities& e, const Vec& x0, const Options& o)
{
    const std::size_t n = p.n, pc = e.C.rows;
    Matrix Ct(n, pc);
    for (std::size_t i = 0; i < pc; ++i)
        for (std::size_t j = 0; j < n; ++j) Ct(j, i) = e.C(i, j);
    const Factor f = factor(Ct, Vec(n, 0.0));
    const std::size_t rc = pseudorank(f, n);
    Vec u(n, 0.0);                                   // R^T u = P^T d
    for (std::size_t j = 0; j < rc; ++j) {
        double s = e.d[f.perm[j]];
        for (std::size_t i = 0; i < j; ++i) s -= f.R(i, j) * u[i];
        u[j] = s / f.R(j, j);
    }
    const Vec xp = q_times(f, u);
    double resid = 0.0, scale = norm(e.d), cbig = 0.0;
    for (std::size_t i = 0; i < pc; ++i) {
        double s = -e.d[i];
        for (std::size_t j = 0; j < n; ++j) {
            s += e.C(i, j) * xp[j];
            cbig = std::max(cbig, std::fabs(e.C(i, j)));
        }
        resid = std::max(resid, std::fabs(s));
    }
    scale = std::max(scale, cbig * norm(xp));
    if (resid > 1e3 * kEps * std::max(scale, kTiny)) throw std::invalid_argument("lm: the linear equalities are inconsistent");
    const std::size_t ny = n - rc;
    Matrix Z(n, ny);
    for (std::size_t c = 0; c < ny; ++c) {
        Vec ec(n, 0.0);
        ec[rc + c] = 1.0;
        const Vec q = q_times(f, ec);
        for (std::size_t i = 0; i < n; ++i) Z(i, c) = q[i];
    }
    Vec origin = xp;                                  // x_0 projected onto C x = d (LMC-4)
    for (std::size_t c = 0; c < ny; ++c) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += Z(i, c) * (x0[i] - xp[i]);
        for (std::size_t i = 0; i < n; ++i) origin[i] += Z(i, c) * s;
    }
    auto x_of = [&](const Vec& y) {
        Vec x = origin;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t c = 0; c < ny; ++c) x[i] += Z(i, c) * y[c];
        return x;
    };
    Problem q;
    q.n = ny;
    q.m = p.m;
    q.residuals = [&](const Vec& y, Vec& r) { p.residuals(x_of(y), r); };
    if (p.jacobian)
        q.jacobian = [&](const Vec& y, Matrix& Jy) {
            Matrix J(p.m, n);
            p.jacobian(x_of(y), J);
            Jy = Matrix(p.m, ny);
            for (std::size_t i = 0; i < p.m; ++i)
                for (std::size_t c = 0; c < ny; ++c) {
                    double s = 0.0;
                    for (std::size_t j = 0; j < n; ++j) s += J(i, j) * Z(j, c);
                    Jy(i, c) = s;
                }
        };
    EqResult out;
    out.eliminated = out.converged = true;
    out.fit = solve(q, Vec(ny, 0.0), o);
    out.fit.x = x_of(out.fit.x);
    Vec c(pc);
    for (std::size_t i = 0; i < pc; ++i) {
        double s = -e.d[i];
        for (std::size_t j = 0; j < n; ++j) s += e.C(i, j) * out.fit.x[j];
        c[i] = s;
    }
    out.c_norm = norm(c);
    Matrix J;                                         // LMC-5
    jacobian_at(p, out.fit.x, out.fit.r, o.fd_eps, J, out.fit);
    const Vec qg = f.qt(gradient_of(J, out.fit.r));
    Vec lp(pc, 0.0);
    for (std::size_t jj = rc; jj-- > 0;) {
        double s = -qg[jj];
        for (std::size_t i = jj + 1; i < rc; ++i) s -= f.R(jj, i) * lp[i];
        lp[jj] = s / f.R(jj, jj);
    }
    out.lambda.assign(pc, 0.0);
    for (std::size_t j = 0; j < pc; ++j) out.lambda[f.perm[j]] = lp[j];
    return out;
}

inline EqResult bound_constrained_lagrangian(const Problem& p, const Equalities& e, Vec x, const Options& o,
                                             const EqOptions& eo)
{
    const std::size_t n = p.n, m = p.m, pc = e.C.rows, nc = pc + e.q;
    EqResult out;
    auto cval = [&](const Vec& v, Vec& c) {
        c.assign(nc, 0.0);
        for (std::size_t i = 0; i < pc; ++i) {
            double s = -e.d[i];
            for (std::size_t j = 0; j < n; ++j) s += e.C(i, j) * v[j];
            c[i] = s;
        }
        if (e.q) {
            Vec hv(e.q);
            e.h(v, hv);
            ++out.ceval;
            for (std::size_t i = 0; i < e.q; ++i) c[pc + i] = hv[i];
        }
    };
    auto cjac = [&](const Vec& v, const Vec& c, Matrix& A) {   // A of c at v, c = c(v)
        A = Matrix(nc, n);
        for (std::size_t i = 0; i < pc; ++i)
            for (std::size_t j = 0; j < n; ++j) A(i, j) = e.C(i, j);
        if (!e.q) return;
        if (e.jacobian) {
            Matrix Ah(e.q, n);
            e.jacobian(v, Ah);
            for (std::size_t i = 0; i < e.q; ++i)
                for (std::size_t j = 0; j < n; ++j) A(pc + i, j) = Ah(i, j);
            return;
        }
        const double step = std::sqrt(std::max(o.fd_eps, kEps));    // as fd_jacobian
        Vec vh = v, ch;
        for (std::size_t j = 0; j < n; ++j) {
            double hj = step * std::fabs(v[j]);
            if (hj == 0.0) hj = step;
            if (!p.hi.empty() && v[j] + hj > p.hi[j]) hj = -hj;
            vh[j] = v[j] + hj;
            cval(vh, ch);
            vh[j] = v[j];
            for (std::size_t i = 0; i < e.q; ++i) A(pc + i, j) = (ch[pc + i] - c[pc + i]) / hj;
        }
    };
    Vec lambda = eo.lambda0.empty() ? Vec(nc, 0.0) : eo.lambda0;
    if (lambda.size() != nc) throw std::invalid_argument("lm: lambda0 needs one value per equality");
    double mu = 10.0, omega = 1.0 / mu, eta = 1.0 / dm::pow(mu, 0.1);
    // the projected gradient of L_A at v, over s (LMC-3); r, c at v
    auto projected = [&](const Vec& v, Vec& r, Vec& c, double s) {
        r.assign(m, 0.0);
        p.residuals(v, r);
        ++out.fit.nfev;
        Matrix J, A;
        jacobian_at(p, v, r, o.fd_eps, J, out.fit);
        cval(v, c);
        cjac(v, c, A);
        Vec g = gradient_of(J, r);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < nc; ++i) g[j] += A(i, j) * (lambda[i] + mu * c[i]);
        Vec dv(n);
        for (std::size_t j = 0; j < n; ++j) {
            double t = v[j] - g[j];
            if (!p.lo.empty()) t = std::min(p.hi[j], std::max(p.lo[j], t));
            dv[j] = v[j] - t;
        }
        return norm(dv) / s;
    };
    Vec r(m), c;
    double s = 1.0;
    {
        p.residuals(x, r);
        ++out.fit.nfev;
        Matrix J;
        jacobian_at(p, x, r, o.fd_eps, J, out.fit);
        s = std::max(1.0, norm(gradient_of(J, r)));
    }
    Problem inner;
    inner.n = n;
    inner.m = m + nc;
    inner.lo = p.lo;
    inner.hi = p.hi;
    inner.residuals = [&](const Vec& v, Vec& ra) {
        Vec rv(m), cv;
        p.residuals(v, rv);
        cval(v, cv);
        ra.assign(m + nc, 0.0);
        std::copy(rv.begin(), rv.end(), ra.begin());
        const double sq = std::sqrt(mu);
        for (std::size_t i = 0; i < nc; ++i) ra[m + i] = sq * (cv[i] + lambda[i] / mu);
    };
    if (p.jacobian)
        inner.jacobian = [&](const Vec& v, Matrix& Ja) {
            Matrix J(m, n), A;
            p.jacobian(v, J);
            Vec cv;
            cval(v, cv);
            cjac(v, cv, A);
            Ja = Matrix(m + nc, n);
            const double sq = std::sqrt(mu);
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) Ja(i, j) = J(i, j);
                for (std::size_t i = 0; i < nc; ++i) Ja(m + i, j) = sq * A(i, j);
            }
        };
    double pg = 0.0;
    for (out.outer = 1; out.outer <= eo.max_outer; ++out.outer) {
        Options oi = o;
        for (int round = 0; round < 4; ++round) {     // LMC-2
            const Result step = solve(inner, x, oi);
            add_counts(out.fit, step);
            x = step.x;
            pg = projected(x, r, c, s);
            if (pg <= omega) break;
            oi.ftol = std::max(kEps, 0.01 * oi.ftol);
            oi.xtol = std::max(kEps, 0.01 * oi.xtol);
        }
        const double cn = norm(c);
        if (cn <= eta) {
            if (cn <= eo.eta && pg <= eo.omega) {
                out.converged = true;
                break;
            }
            for (std::size_t i = 0; i < nc; ++i) lambda[i] += mu * c[i];
            eta = eta / dm::pow(mu, 0.9);
            omega = omega / mu;
        } else {
            if (mu >= eo.mu_max) break;              // LMC-3
            mu = 100.0 * mu;
            eta = 1.0 / dm::pow(mu, 0.1);
            omega = 1.0 / mu;
        }
    }
    if (out.outer > eo.max_outer) out.outer = eo.max_outer;
    out.fit.x = x;
    out.fit.r = r;
    out.fit.cost = 0.5 * norm(r) * norm(r);
    out.lambda = lambda;
    out.mu = mu;
    out.c_norm = norm(c);
    out.projected_gradient = pg;
    return out;
}

}   // namespace detail

inline EqResult solve_eq(const Problem& p, const Equalities& e, Vec x, const Options& o, const EqOptions& eo = {})
{
    if (e.C.rows > 0 && (e.C.cols != p.n || e.d.size() != e.C.rows))
        throw std::invalid_argument("lm: C needs n columns and d one value per row");
    if (e.q > 0 && !e.h) throw std::invalid_argument("lm: q nonlinear equalities need h");
    if (p.gauge) throw std::invalid_argument("lm: equalities take no gauge basis");
    bool bounded = false;
    for (std::size_t i = 0; i < p.lo.size(); ++i) bounded = bounded || p.lo[i] > -kInf || p.hi[i] < kInf;
    if (e.C.rows == 0 && e.q == 0) {
        EqResult out;
        out.fit = solve(p, std::move(x), o);
        out.converged = true;
        return out;
    }
    if (e.q == 0 && !bounded) return detail::eliminate(p, e, x, o);
    return detail::bound_constrained_lagrangian(p, e, std::move(x), o, eo);
}

// ── profiles (task 19, §19.3.3; LMP) ────────────────────────────────────────
// The profile of the parameter x_j: for each value v the least cost with x_j
// = v fixed (the linear equality e_j^T x = v, solve_eq), the others fitted by
// lm, each from the neighbouring node's solution, outward from x_hat in both
// directions; the threshold and the classification of A. Raue et al.,
// "Structural and practical identifiability analysis of partially observed
// dynamical models by exploiting the profile likelihood", Bioinformatics
// 25(15) 2009, 1923-1929 (source profile-likelihood-identifiability_raue2009):
// (8), the region chi^2(theta) - chi^2(theta_hat) < Delta_alpha, Delta_alpha
// = chi2_quantile(alpha, df) (df = 1 pointwise, df = n simultaneous, their
// (7)); §2.3: identifiable where the profile passes Delta_alpha on both
// sides, structurally non-identifiable where it is flat, practically
// non-identifiable where it stays below it on a side. OURS:
//   LMP-1. chi^2 = sum (r_i / sigma_i)^2 = 2 cost where the residuals are
//     divided by sigma: the threshold on the cost is Delta_alpha / 2, times
//     sigma2 where the caller gives the residuals' variance (the residuals
//     not divided); without sigma2 (0) the residuals are taken as divided by
//     sigma = 1 and the profile is relative (Profile::relative).
//   LMP-2. The automatic grid: steps of h = sqrt(2 threshold C_jj) / 10, C
//     = (J^T J)^+ at x_hat (the quadratic model's crossing in ten steps), or
//     0.1 max(1, |x_hat_j|) where C_jj = 0; max_steps nodes a side at most;
//     a side ends past the threshold, or on x_j's bound (the last node on
//     it). A crossing is refined by bisection between the last node below
//     and the first above (30 halvings, each a fit from the node below).
//   LMP-3. Flat: every node's cost within flat_tol max(1, cost_hat) of
//     cost_hat (flat_tol = 1e-8).
//   LMP-4. chi2_quantile: the regularized lower incomplete gamma P(df/2,
//     x/2) by its power series, the definition's integrand e^-t t^(a-1)
//     expanded and integrated term by term: P(a, z) = z^a e^-z / Gamma(a + 1)
//     sum_k z^k / ((a + 1) ... (a + k)), Gamma(a + 1) by its recurrence from
//     Gamma(1) = 1 or Gamma(1/2) = sqrt(pi) (a is a multiple of 1/2); the
//     quantile by bisection.

inline double chi2_cdf(double x, int df)
{
    if (!(x > 0.0)) return 0.0;
    const double a = 0.5 * df, z = 0.5 * x;
    double lg = 0.0;                                  // log Gamma(a + 1)
    for (double t = a; t > 0.25; t -= 1.0) lg += dm::log(t);
    if (df % 2 == 1) lg += 0.5 * dm::log(dm::kPi);    // Gamma(1/2)
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 100000; ++k) {
        term *= z / (a + k);
        sum += term;
        if (term < kEps * sum) break;
    }
    return std::min(1.0, dm::exp(a * dm::log(z) - z - lg) * sum);
}

inline double chi2_quantile(double alpha, int df)
{
    if (!(alpha > 0.0 && alpha < 1.0) || df < 1) throw std::invalid_argument("chi2_quantile: 0 < alpha < 1, df >= 1");
    double lo = 0.0, hi = 1.0;
    while (chi2_cdf(hi, df) < alpha) hi *= 2.0;
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (chi2_cdf(mid, df) < alpha) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

struct ProfileOptions {
    std::vector<double> grid;               // the values of x_j; empty: automatic (LMP-2)
    double alpha = 0.95;
    int df = 1;
    double sigma2 = 0.0;                    // the residuals' variance (0: divided already, relative; LMP-1)
    double step = 0.0;                      // 0: LMP-2's
    int max_steps = 50;
    double flat_tol = 1e-8;                 // LMP-3
};

struct ProfilePoint {
    double value = 0.0, cost = 0.0;
    Vec x;
};

struct Profile {
    std::size_t j = 0;
    double cost_hat = 0.0, threshold = 0.0;  // the cost's threshold above cost_hat
    std::vector<ProfilePoint> points;       // by increasing value, x_hat's among them
    double lower = -kInf, upper = kInf;     // where the profile passes the threshold (infinite: it does not)
    bool relative = false;
    enum class Kind { identifiable, practically_non_identifiable, structurally_non_identifiable } kind =
        Kind::identifiable;
};

inline Profile profile(const Problem& p, const Vec& x_hat, std::size_t j, const Options& o,
                       const ProfileOptions& po = {})
{
    const std::size_t n = p.n;
    if (j >= n) throw std::invalid_argument("profile: j is not a parameter");
    Profile pr;
    pr.j = j;
    pr.relative = !(po.sigma2 > 0.0);
    pr.threshold = 0.5 * chi2_quantile(po.alpha, po.df) * (pr.relative ? 1.0 : po.sigma2);
    Vec r(p.m);
    p.residuals(x_hat, r);
    pr.cost_hat = 0.5 * norm(r) * norm(r);
    auto fit_at = [&](double v, const Vec& from) {
        Equalities e;
        e.C = Matrix(1, n);
        e.C(0, j) = 1.0;
        e.d = {v};
        Vec x0 = from;
        x0[j] = v;
        if (!p.lo.empty()) x0[j] = std::min(p.hi[j], std::max(p.lo[j], v));
        const EqResult er = solve_eq(p, e, x0, o);
        return ProfilePoint{v, er.fit.cost, er.fit.x};
    };
    const double lo_j = p.lo.empty() ? -kInf : p.lo[j], hi_j = p.hi.empty() ? kInf : p.hi[j];
    std::vector<ProfilePoint> sides[2];
    bool crossed[2] = {false, false};
    double crossing[2] = {-kInf, kInf};
    if (!po.grid.empty()) {
        for (int side = 0; side < 2; ++side) {        // outward from x_hat_j, below and above
            std::vector<double> vs;
            for (double v : po.grid)
                if (side == 0 ? v < x_hat[j] : v > x_hat[j]) vs.push_back(v);
            std::sort(vs.begin(), vs.end());
            if (side == 0) std::reverse(vs.begin(), vs.end());
            Vec from = x_hat;
            for (double v : vs) {
                sides[side].push_back(fit_at(v, from));
                from = sides[side].back().x;
            }
        }
    } else {
        double h = po.step;
        if (!(h > 0.0)) {                             // LMP-2
            Matrix J;
            Result counts;
            detail::jacobian_at(p, x_hat, r, o.fd_eps, J, counts);
            const Diagnostics d = diagnose(J, pr.cost_hat, 1.0, !p.jacobian, o.fd_eps);
            const double cjj = d.cov(j, j);
            h = cjj > 0.0 ? std::sqrt(2.0 * pr.threshold * cjj) / 10.0 : 0.1 * std::max(1.0, std::fabs(x_hat[j]));
        }
        for (int side = 0; side < 2; ++side) {
            const double dir = side == 0 ? -1.0 : 1.0, bound = side == 0 ? lo_j : hi_j;
            Vec from = x_hat;
            ProfilePoint below{x_hat[j], pr.cost_hat, x_hat};
            for (int k = 1; k <= po.max_steps; ++k) {
                double v = x_hat[j] + dir * k * h;
                const bool on_bound = side == 0 ? v <= bound : v >= bound;
                if (on_bound) v = bound;
                ProfilePoint pt = fit_at(v, from);
                if (pt.cost - pr.cost_hat > pr.threshold) {   // refine the crossing (LMP-2)
                    double a = below.value, b = v;
                    Vec xa = below.x;
                    for (int it = 0; it < 30; ++it) {
                        const double mid = 0.5 * (a + b);
                        const ProfilePoint pm = fit_at(mid, xa);
                        if (pm.cost - pr.cost_hat > pr.threshold) b = mid;
                        else {
                            a = mid;
                            xa = pm.x;
                        }
                    }
                    crossed[side] = true;
                    crossing[side] = 0.5 * (a + b);
                    sides[side].push_back(std::move(pt));
                    break;
                }
                below = pt;
                from = pt.x;
                sides[side].push_back(std::move(pt));
                if (on_bound) break;
            }
        }
    }
    for (std::size_t k = sides[0].size(); k-- > 0;) pr.points.push_back(sides[0][k]);
    pr.points.push_back(ProfilePoint{x_hat[j], pr.cost_hat, x_hat});
    for (const ProfilePoint& pt : sides[1]) pr.points.push_back(pt);
    if (!po.grid.empty())                             // the grid's crossings: between nodes, linearly
        for (std::size_t k = 1; k < pr.points.size(); ++k) {
            const ProfilePoint &a = pr.points[k - 1], &b = pr.points[k];
            const double ta = a.cost - pr.cost_hat - pr.threshold, tb = b.cost - pr.cost_hat - pr.threshold;
            if ((ta > 0.0) == (tb > 0.0)) continue;
            const double v = a.value + (b.value - a.value) * (ta / (ta - tb));
            if (b.value <= x_hat[j]) {
                crossed[0] = true;
                crossing[0] = v;
            } else if (!crossed[1]) {
                crossed[1] = true;
                crossing[1] = v;
            }
        }
    pr.lower = crossed[0] ? crossing[0] : -kInf;
    pr.upper = crossed[1] ? crossing[1] : kInf;
    bool flat = true;
    for (const ProfilePoint& pt : pr.points)
        flat = flat && std::fabs(pt.cost - pr.cost_hat) <= po.flat_tol * std::max(1.0, pr.cost_hat);
    pr.kind = crossed[0] && crossed[1] ? Profile::Kind::identifiable
            : flat                    ? Profile::Kind::structurally_non_identifiable
                                      : Profile::Kind::practically_non_identifiable;
    return pr;
}

}   // namespace mootation::scalar::lm

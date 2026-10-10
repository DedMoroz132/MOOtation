#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Variation in learned coordinates (task 7 of 2026-10-10, supplement 1): the
// knob `coords` of the switchable algorithms (nsga2, ibea_eplus, spea2_sde,
// agemoea, sms_emoa) and of mosa's sbx step. A construction of this library.
//
//   axis     (the default) the variation as it always was
//   pca      the crossover and the mutation act on y = (x − μ) R, R the
//            eigenvectors of the population's covariance, by decreasing
//            eigenvalue, μ its mean
//   pca_nd   the same from the population's non-dominated members (the whole
//            population while fewer than three)
//   randrot  a fresh Haar rotation at every update, μ the population's mean:
//            a rotation that is not learned, the control
//
// THE MECHANISM.
//   * Warm-up: for the first coords_warmup generations (10) R = I and nothing
//     changes — the population is still random. In generations, not in a share
//     of the budget, so that the algorithm does not become budget-dependent. A generation is pop_size offspring (sms_emoa:
//     pop_size steps).
//   * Update: R and μ every coords_update generations (1) from then on.
//   * The basis is complete even with fewer points than variables: Jacobi's
//     method gives every eigenvector of the n × n covariance, the null ones
//     included. Each eigenvector's sign is fixed: its first component above
//     1e-12 in size is positive.
//   * In y there is no box: the crossovers see bounds of ±1e300, where SBX's
//     bounded formula is its unbounded one and no repair fires. The mutation
//     too acts in y — otherwise it is axis-wise again and the point is lost —
//     with per-axis bounds: the projected population's range, widened by half
//     its span on each side and, where needed, to the point itself (a zero
//     span counts as 1e-3 of the widest box side); its own repair is left to
//     the end.
//   * Back to x: x = y Rᵀ + μ, then bound_repair (reflect by default; the
//     first parent for midpoint; resample and native refused), counted in
//     the repair tally like every operator's.
//
// THE RECORD (coords_record(), thread-local like the repair counts): R, μ and
// the eigenvalues after every update, for the run layer's diagnostics
// (principal angles against a problem's distance block, the spectrum).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bound_repair.hpp"

namespace mootation::ops {

enum class CoordsKind { Axis, PCA, PCAND, RandRot };

inline const char* coords_name(CoordsKind k)
{
    switch (k) {
        case CoordsKind::Axis:    return "axis";
        case CoordsKind::PCA:     return "pca";
        case CoordsKind::PCAND:   return "pca_nd";
        case CoordsKind::RandRot: return "randrot";
    }
    return "?";
}

inline std::optional<CoordsKind> parse_coords(const std::string& s)
{
    for (CoordsKind k : {CoordsKind::Axis, CoordsKind::PCA, CoordsKind::PCAND,
                         CoordsKind::RandRot})
        if (s == coords_name(k)) return k;
    return std::nullopt;
}

struct CoordsRecord {
    bool active = false;
    int m = 0;                                   // the objectives, for the spectrum's share
    std::vector<std::vector<double>> R;          // columns: the basis, R[i][j] = e_j's i-th
    std::vector<double> mu, eig;
    long long updates = 0;
};
inline CoordsRecord& coords_record() { thread_local CoordsRecord r; return r; }

namespace coords_detail {

// Jacobi's eigenvalue method for a symmetric matrix A (n × n, row-major):
// the eigenvalues in decreasing order and V's columns the eigenvectors.
inline void jacobi(std::vector<std::vector<double>> A, std::vector<double>& eig,
                   std::vector<std::vector<double>>& V)
{
    const std::size_t n = A.size();
    V.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) V[i][i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, all = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                all += A[i][j] * A[i][j];
                if (i != j) off += A[i][j] * A[i][j];
            }
        if (off <= 1e-30 * all || off == 0.0) break;
        for (std::size_t p = 0; p + 1 < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q) {
                if (A[p][q] == 0.0) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (std::size_t k = 0; k < n; ++k) {           // A <- A J
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {           // A <- Jᵀ A
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < n; ++k) {           // V <- V J
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
    }
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return A[a][a] > A[b][b]; });
    eig.assign(n, 0.0);
    std::vector<std::vector<double>> W(n, std::vector<double>(n));
    for (std::size_t c = 0; c < n; ++c) {
        eig[c] = A[order[c]][order[c]];
        double sign = 0.0;
        for (std::size_t i = 0; i < n && sign == 0.0; ++i)
            if (std::abs(V[i][order[c]]) > 1e-12) sign = V[i][order[c]] > 0.0 ? 1.0 : -1.0;
        if (sign == 0.0) sign = 1.0;
        for (std::size_t i = 0; i < n; ++i) W[i][c] = sign * V[i][order[c]];
    }
    V = std::move(W);
}

// A Haar rotation: the Q of a standard normal matrix's QR by modified
// Gram-Schmidt, whose R has a positive diagonal.
template <typename RNG>
inline std::vector<std::vector<double>> haar(std::size_t n, RNG& rng)
{
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<std::vector<double>> Q(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Q[i][j] = N01(rng);
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t b = 0; b < c; ++b) {
            double d = 0.0;
            for (std::size_t i = 0; i < n; ++i) d += Q[i][b] * Q[i][c];
            for (std::size_t i = 0; i < n; ++i) Q[i][c] -= d * Q[i][b];
        }
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) norm += Q[i][c] * Q[i][c];
        norm = std::sqrt(norm);
        for (std::size_t i = 0; i < n; ++i) Q[i][c] /= norm;
    }
    return Q;
}

inline bool dominates(const std::vector<double>& a, const std::vector<double>& b)
{
    bool lt = false;
    for (std::size_t j = 0; j < a.size(); ++j) {
        if (a[j] > b[j]) return false;
        lt = lt || a[j] < b[j];
    }
    return lt;
}

}   // namespace coords_detail

struct LearnedCoords {
    CoordsKind kind = CoordsKind::Axis;
    int warmup = 10;                           // generations with R = I
    int update = 1;                            // generations between updates

    // Is the variation in y at this step?
    bool active() const { return on_; }

    // The knobs coords, coords_warmup, coords_update (settings.hpp).
    bool set_param(const std::string& k, double v)
    {
        if (k == "coords_warmup") {
            if (!(v >= 0.0) || v != std::floor(v))
                throw std::invalid_argument("coords_warmup must be an integer >= 0");
            warmup = static_cast<int>(std::min(v, 1e9));
            return true;
        }
        if (k == "coords_update") {
            if (!(v >= 1.0) || v != std::floor(v))
                throw std::invalid_argument("coords_update must be an integer >= 1");
            update = static_cast<int>(std::min(v, 1e9));
            return true;
        }
        return false;
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (k != "coords") return false;
        auto c = parse_coords(w);
        if (!c) throw std::invalid_argument("coords = '" + w + "': axis, pca, pca_nd or randrot");
        kind = *c;
        return true;
    }

    // At the start of every step, the population (X its variables, F its
    // objectives) and the offspring the step will make: the generation count,
    // and an update when one is due.
    template <typename RNG>
    void observe(const std::vector<std::vector<double>>& X,
                 const std::vector<std::vector<double>>& F, int offspring,
                 const std::vector<std::pair<std::optional<double>, std::optional<double>>>& box,
                 RNG& rng)
    {
        if (kind == CoordsKind::Axis || X.empty()) return;
        const long long n = static_cast<long long>(X.size());
        const long long gen = made_ / n;
        made_ += offspring;
        if (due(gen)) fit_at(gen, X, F, box, rng);
    }

    // Is an update due in generation `gen`?
    bool due(long long gen) const
    {
        return kind != CoordsKind::Axis && gen >= warmup && (!on_ || gen >= last_ + update);
    }

    // The update of generation `gen` (a core that counts its generations
    // itself: mosa).
    template <typename RNG>
    void fit_at(long long gen, const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>>& F,
                const std::vector<std::pair<std::optional<double>, std::optional<double>>>& box,
                RNG& rng)
    {
        fit(X, F, box, rng);
        last_ = gen;
    }

    // R, μ and the y-bounds from the population.
    template <typename RNG>
    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& F,
             const std::vector<std::pair<std::optional<double>, std::optional<double>>>& box,
             RNG& rng)
    {
        const std::size_t n = X[0].size();
        std::vector<std::size_t> rows;
        if (kind == CoordsKind::PCAND && F.size() == X.size()) {
            for (std::size_t i = 0; i < X.size(); ++i) {
                bool beaten = false;
                for (std::size_t j = 0; j < X.size() && !beaten; ++j)
                    beaten = j != i && coords_detail::dominates(F[j], F[i]);
                if (!beaten) rows.push_back(i);
            }
        }
        if (rows.size() < 3) {
            rows.resize(X.size());
            for (std::size_t i = 0; i < X.size(); ++i) rows[i] = i;
        }
        const std::vector<std::size_t> all = [&] {
            std::vector<std::size_t> a(X.size());
            for (std::size_t i = 0; i < a.size(); ++i) a[i] = i;
            return a;
        }();
        const auto& from = kind == CoordsKind::RandRot ? all : rows;
        mu_.assign(n, 0.0);
        for (std::size_t i : from)
            for (std::size_t j = 0; j < n; ++j) mu_[j] += X[i][j];
        for (double& v : mu_) v /= static_cast<double>(from.size());
        if (kind == CoordsKind::RandRot) {
            R_ = coords_detail::haar(n, rng);
            eig_.assign(n, 0.0);
        } else {
            std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
            if (from.size() > 1) {
                for (std::size_t i : from)
                    for (std::size_t a = 0; a < n; ++a) {
                        const double da = X[i][a] - mu_[a];
                        for (std::size_t b = a; b < n; ++b) C[a][b] += da * (X[i][b] - mu_[b]);
                    }
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = a; b < n; ++b) {
                        C[a][b] /= static_cast<double>(from.size() - 1);
                        C[b][a] = C[a][b];
                    }
            }
            coords_detail::jacobi(C, eig_, R_);
        }
        // the y-box: the projected population's range, widened by half its span
        double widest = 0.0;
        for (const auto& b : box)
            if (b.first && b.second) widest = std::max(widest, *b.second - *b.first);
        ylo_.assign(n, 1e300);
        yhi_.assign(n, -1e300);
        for (const auto& x : X) {
            const auto y = to_y(x);
            for (std::size_t j = 0; j < n; ++j) {
                ylo_[j] = std::min(ylo_[j], y[j]);
                yhi_[j] = std::max(yhi_[j], y[j]);
            }
        }
        for (std::size_t j = 0; j < n; ++j) {
            double span = yhi_[j] - ylo_[j];
            if (!(span > 0.0)) span = 1e-3 * (widest > 0.0 ? widest : 1.0);
            ylo_[j] -= 0.5 * span;
            yhi_[j] += 0.5 * span;
        }
        on_ = true;
        auto& rec = coords_record();
        rec.active = true;
        rec.m = F.empty() ? 0 : static_cast<int>(F[0].size());
        rec.R = R_;
        rec.mu = mu_;
        rec.eig = eig_;
        ++rec.updates;
    }

    std::vector<double> to_y(const std::vector<double>& x) const
    {
        const std::size_t n = x.size();
        std::vector<double> y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = x[i] - mu_[i];
            for (std::size_t c = 0; c < n; ++c) y[c] += d * R_[i][c];
        }
        return y;
    }

    std::vector<double> to_x(const std::vector<double>& y) const
    {
        const std::size_t n = y.size();
        std::vector<double> x(mu_);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t c = 0; c < n; ++c) x[i] += R_[i][c] * y[c];
        return x;
    }

    // The crossovers' bounds in y: none, as ±1e300.
    std::vector<std::pair<std::optional<double>, std::optional<double>>> open(std::size_t n) const
    {
        return std::vector<std::pair<std::optional<double>, std::optional<double>>>(
            n, {std::optional<double>(-1e300), std::optional<double>(1e300)});
    }

    // The mutation's bounds for one child in y: the widened range, and the
    // child itself when it lies outside it.
    std::vector<std::pair<std::optional<double>, std::optional<double>>>
    mutation_bounds(const std::vector<double>& y) const
    {
        std::vector<std::pair<std::optional<double>, std::optional<double>>> b(y.size());
        for (std::size_t j = 0; j < y.size(); ++j)
            b[j] = {std::min(ylo_[j], y[j]), std::max(yhi_[j], y[j])};
        return b;
    }

    // y back to x, repaired into the box; parent: for midpoint.
    template <typename RNG>
    std::vector<double> back(const std::vector<double>& y,
                             const std::vector<std::pair<std::optional<double>,
                                                         std::optional<double>>>& box,
                             const std::vector<double>& parent, BoundRepair how, RNG& rng) const
    {
        std::vector<double> x = to_x(y);
        RepairTally tally;
        for (std::size_t j = 0; j < x.size(); ++j) {
            const double lo = box[j].first ? *box[j].first : -1e300;
            const double hi = box[j].second ? *box[j].second : 1e300;
            if (x[j] < lo || x[j] > hi) {
                tally.out();
                x[j] = repair_value(x[j], lo, hi, parent[j], how, rng);
            }
        }
        return x;
    }

    const std::vector<std::vector<double>>& basis() const { return R_; }
    const std::vector<double>& mean() const { return mu_; }
    const std::vector<double>& eigenvalues() const { return eig_; }

private:
    bool on_ = false;
    long long made_ = 0;                       // offspring made since the start
    long long last_ = 0;                       // the generation of the last update
    std::vector<std::vector<double>> R_;
    std::vector<double> mu_, eig_, ylo_, yhi_;
};

}   // namespace mootation::ops

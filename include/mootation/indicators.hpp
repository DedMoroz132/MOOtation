#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The run layer's front indicators in C++: python/mootation/run/metrics.py's
// igd, dplus_distances (igd_plus is their mean), front_distances (the d+
// matrix minimised both ways and the additive epsilon's), eps_plus and
// r2_discrete, value for value; and IncrementalFront, the same distances kept
// up to date over every point a run evaluates. metrics.py describes the
// indicators and their sources, calls these when the extension is built and
// keeps its NumPy code as the reference the tests compare with.
//
// THE SAME NUMBERS. Per pair of a point a and a reference point z the value is
// the expression metrics.py evaluates: sqrt of the sum over the objectives,
// in NumPy's pairwise order (numpy_sum.hpp), of (z - a)^2 for IGD and of
// max(a - z, 0)^2 for d+; max over the objectives of a - z for epsilon. A
// minimum and a maximum are exact whatever the order; the mean over the
// reference, or over the set, is NumPy's. A NaN anywhere passes on as NumPy
// passes it (np.maximum, min, max, sqrt). So every value is the same bit for
// bit, and splitting the reference points over threads changes nothing.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#include "nd_tree.hpp"
#include "numpy_sum.hpp"

namespace mootation::indicators {

namespace detail {

inline double pos(double d) { return std::isnan(d) ? d : (d >= 0.0 ? d : 0.0); }   // np.maximum(d, 0)

// NumPy's min and max of two values: a NaN on either side gives NaN.
inline double nan_min(double a, double b)
{
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return std::min(a, b);
}
inline double nan_max(double a, double b)
{
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return std::max(a, b);
}

// f(z0, z1) over [0, n) in `threads` contiguous parts (0: as many as the
// machine has); each part writes only its own entries.
template <typename F>
void parallel(std::size_t n, unsigned threads, const F& f)
{
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t parts = std::min<std::size_t>(threads, n / 64 + 1);
    if (parts <= 1) { f(std::size_t{0}, n); return; }
    std::vector<std::thread> pool;
    const std::size_t step = (n + parts - 1) / parts;
    for (std::size_t p = 0; p < parts; ++p) {
        const std::size_t a = p * step, b = std::min(n, a + step);
        if (a < b) pool.emplace_back([&f, a, b] { f(a, b); });
    }
    for (auto& t : pool) t.join();
}

}   // namespace detail

// The distances behind IGD+, GD+ and the additive epsilon, for n > 0 points F
// and r reference points R (row-major, m columns): to_set[z] = min over a of
// d+(z, a), to_ref[a] = min over z of d+(z, a), eps[z] = min over a of max_j
// (a_j - z_j) — metrics.front_distances. Any of the outputs may be null.
inline void front_distances(const double* F, std::size_t n, const double* R, std::size_t r,
                            std::size_t m, double* to_set, double* to_ref, double* eps,
                            unsigned threads = 1)
{
    const double inf = std::numeric_limits<double>::infinity();
    // reference points z0..z1: to_set and eps there, and the minimum over them
    // of every point's d+ into ref_min (to_ref is the minimum of the parts')
    auto work = [&](std::size_t z0, std::size_t z1, std::vector<double>& ref_min) {
        std::vector<double> buf(m);
        for (std::size_t z = z0; z < z1; ++z) {
            const double* rz = R + z * m;
            double best = inf, best_eps = inf;
            for (std::size_t a = 0; a < n; ++a) {
                const double* fa = F + a * m;
                double worst = -inf;
                for (std::size_t j = 0; j < m; ++j) {
                    const double d = fa[j] - rz[j];
                    const double t = detail::pos(d);
                    buf[j] = t * t;
                    worst = j == 0 ? d : detail::nan_max(worst, d);
                }
                const double dp = std::sqrt(np::sum(buf.data(), m));
                best = a == 0 ? dp : detail::nan_min(best, dp);
                best_eps = a == 0 ? worst : detail::nan_min(best_eps, worst);
                if (!ref_min.empty()) ref_min[a] = detail::nan_min(ref_min[a], dp);
            }
            if (to_set) to_set[z] = best;
            if (eps) eps[z] = best_eps;
        }
    };
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t parts = std::max<std::size_t>(1, std::min<std::size_t>(threads, r / 64));
    const std::size_t step = parts ? (r + parts - 1) / parts : r;
    std::vector<std::vector<double>> mins(parts, std::vector<double>(to_ref ? n : 0, inf));
    if (parts == 1) {
        work(0, r, mins[0]);
    } else {
        std::vector<std::thread> pool;
        for (std::size_t p = 0; p < parts; ++p) {
            const std::size_t z0 = p * step, z1 = std::min(r, z0 + step);
            if (z0 < z1) pool.emplace_back([&work, &mins, p, z0, z1] { work(z0, z1, mins[p]); });
        }
        for (auto& t : pool) t.join();
    }
    if (to_ref)
        for (std::size_t a = 0; a < n; ++a) {
            double v = inf;
            for (const auto& part : mins) v = detail::nan_min(v, part[a]);
            to_ref[a] = v;
        }
}

// metrics.igd's distances: near[z] = min over a of ||z - a||.
inline void igd_distances(const double* F, std::size_t n, const double* R, std::size_t r,
                          std::size_t m, double* near, unsigned threads = 1)
{
    detail::parallel(r, threads, [&](std::size_t z0, std::size_t z1) {
        std::vector<double> buf(m);
        for (std::size_t z = z0; z < z1; ++z) {
            const double* rz = R + z * m;
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t a = 0; a < n; ++a) {
                const double* fa = F + a * m;
                for (std::size_t j = 0; j < m; ++j) {
                    const double d = rz[j] - fa[j];
                    buf[j] = d * d;
                }
                const double v = std::sqrt(np::sum(buf.data(), m));
                best = a == 0 ? v : detail::nan_min(best, v);
            }
            near[z] = best;
        }
    });
}

// metrics.r2_discrete's inner part on the normalised set G (n x m) and the
// weights W (w x m): best[k] = min over a of max_i W_ki G_ai; the mean is the
// indicator.
inline double r2_mean(const double* G, std::size_t n, const double* W, std::size_t w,
                      std::size_t m)
{
    std::vector<double> best(w, std::numeric_limits<double>::infinity());
    for (std::size_t k = 0; k < w; ++k) {
        const double* wk = W + k * m;
        double b = std::numeric_limits<double>::infinity();
        for (std::size_t a = 0; a < n; ++a) {
            const double* ga = G + a * m;
            double u = wk[0] * ga[0];
            for (std::size_t i = 1; i < m; ++i) u = detail::nan_max(u, wk[i] * ga[i]);
            b = a == 0 ? u : detail::nan_min(b, u);
        }
        best[k] = b;                // np.minimum over the blocks of 256 points: the same minimum
    }
    return np::mean(best.data(), w);
}

// IGD+ and the additive epsilon of everything a run has evaluated, kept up to
// date one point at a time: for every reference point z the smallest d+(z, a)
// and the smallest max_j (a_j - z_j) over the points a so far, in the frame
// (f - ideal) / span (span = nadir - ideal, a span <= 0 counting as 1;
// metrics._normalised), and for every level the evaluation at which IGD+ first
// fell to it or below. igd_plus() and eps_plus() are what metrics.igd_plus and
// metrics.eps_plus give on all the points so far, normalised the same way, bit
// for bit: the minima are exact, and the mean is taken in NumPy's order. An
// update costs O(|Z| m) per point.
//
// A point weakly dominated by one offered before (b <= a in every objective,
// a copy included) is skipped (task 17, supplement 1, stage 0): it lowers
// nothing. The frame is monotone, (a - ideal) / span >= (b - ideal) / span in
// every objective after rounding too, and so is every step from there: d =
// g - z, max(d, 0), its square, NumPy's pairwise sum of non-negative terms,
// the root, the max over the objectives. So d+(z, a) >= d+(z, b) and
// max_j (a_j - z_j) >= max_j (b_j - z_j) for every z, and neither minimum can
// fall at a. The points not skipped are kept in an ND-tree (nd_tree.hpp) of
// the nondominated ones so far; a point with a NaN is never skipped nor kept
// (its NaN reaches the minima as before). skip_dominated = false offers every
// point, the reference for the tests. The skip pays up to five objectives
// (skip_pays): on the points NSGA-II and SMS-EMOA evaluate in 25 000, the
// update takes 0.08-0.71 of its time at M = 2-5, where 37-92 % of the points
// are dominated, and 1.07-1.38 of it at M = 8, 10 (NSGA-III), where 10-45 %
// are and the tree costs more than it saves.
class IncrementalFront {
public:
    static bool skip_pays(std::size_t m) { return m <= 5; }

    IncrementalFront(const double* Z, std::size_t r, std::size_t m, const std::vector<double>& ideal,
                     const std::vector<double>& nadir, std::vector<double> levels,
                     bool skip_dominated = true)
        : r_(r), m_(m), lo_(ideal), span_(m), levels_(std::move(levels)), reached_(levels_.size(), -1),
          skip_(skip_dominated), front_(m)
    {
        for (std::size_t j = 0; j < m; ++j) {
            const double s = nadir[j] - ideal[j];
            span_[j] = s > 0.0 ? s : 1.0;
        }
        Z_.resize(r * m);
        for (std::size_t z = 0; z < r; ++z)
            for (std::size_t j = 0; j < m; ++j) Z_[z * m + j] = (Z[z * m + j] - lo_[j]) / span_[j];
        dmin_.assign(r, std::numeric_limits<double>::infinity());
        dsq_.assign(r, std::numeric_limits<double>::infinity());
        emin_.assign(r, std::numeric_limits<double>::infinity());
        buf_.resize(m);
        g_.resize(m);
    }

    // Offer the points F (n x m, raw objectives), evaluated at the evaluation
    // counts fe[0..n) (in order).
    void update(const double* F, std::size_t n, const long long* fe)
    {
        for (std::size_t a = 0; a < n; ++a) {
            if (skip_ && dominated(F + a * m_)) {
                ++n_seen_;
                ++skipped_;
                continue;
            }
            for (std::size_t j = 0; j < m_; ++j) g_[j] = (F[a * m_ + j] - lo_[j]) / span_[j];
            bool lower = false;
            for (std::size_t z = 0; z < r_; ++z) {
                const double* rz = Z_.data() + z * m_;
                double worst = 0.0;
                for (std::size_t j = 0; j < m_; ++j) {
                    const double d = g_[j] - rz[j];
                    const double t = detail::pos(d);
                    buf_[j] = t * t;
                    worst = j == 0 ? d : detail::nan_max(worst, d);
                }
                // dmin_[z] = sqrt(dsq_[z]), dsq_[z] the least sum of squares
                // so far (a NaN for good once one is met): sqrt keeps the
                // order, so the least distance is the root of the least
                // sum, taken only when that sum falls
                const double sq = np::sum(buf_.data(), m_);
                if (std::isnan(dsq_[z])) {
                } else if (std::isnan(sq)) {
                    dsq_[z] = dmin_[z] = sq;
                    lower = true;
                } else if (sq < dsq_[z]) {
                    dsq_[z] = sq;
                    const double dp = std::sqrt(sq);
                    if (!(dp == dmin_[z])) { dmin_[z] = dp; lower = true; }
                }
                emin_[z] = n_seen_ == 0 ? worst : detail::nan_min(emin_[z], worst);
            }
            ++n_seen_;
            if (lower) check_levels(fe[a]);
        }
    }

    std::size_t seen() const { return n_seen_; }
    // the points skipped as weakly dominated by one offered before
    std::size_t skipped() const { return skipped_; }
    double igd_plus() const
    {
        return n_seen_ ? np::mean(dmin_.data(), r_) : std::numeric_limits<double>::infinity();
    }
    double eps_plus() const
    {
        if (!n_seen_) return std::numeric_limits<double>::infinity();
        double v = emin_[0];
        for (std::size_t z = 1; z < r_; ++z) v = detail::nan_max(v, emin_[z]);
        return v;
    }
    const std::vector<double>& levels() const { return levels_; }
    // the evaluation at which IGD+ first reached each level, -1 not yet
    const std::vector<long long>& reached() const { return reached_; }
    const std::vector<double>& distances() const { return dmin_; }

private:
    std::size_t r_, m_;
    std::vector<double> lo_, span_, Z_;
    std::vector<double> levels_;
    std::vector<long long> reached_;
    std::vector<double> dmin_, dsq_, emin_, buf_, g_;
    std::size_t n_seen_ = 0;
    bool skip_;
    nd::NDTree front_;                  // the nondominated points offered so far
    std::vector<std::size_t> free_ids_, beaten_;
    std::size_t next_id_ = 0, skipped_ = 0;

    // Is f weakly dominated by a point offered before? If not, f goes into
    // the front and the points it weakly dominates leave it.
    bool dominated(const double* f)
    {
        for (std::size_t j = 0; j < m_; ++j)
            if (std::isnan(f[j])) return false;
        if (front_.covered(f, &beaten_)) return true;
        for (std::size_t id : beaten_) {
            front_.erase(id);
            free_ids_.push_back(id);
        }
        std::size_t id = next_id_;
        if (free_ids_.empty()) {
            ++next_id_;
        } else {
            id = free_ids_.back();
            free_ids_.pop_back();
        }
        front_.insert(id, f);
        return false;
    }

    void check_levels(long long fe)
    {
        bool open = false;
        for (long long r : reached_) open = open || r < 0;
        if (!open) return;
        const double v = igd_plus();
        for (std::size_t k = 0; k < levels_.size(); ++k)
            if (reached_[k] < 0 && v <= levels_[k]) reached_[k] = fe;
    }
};

}   // namespace mootation::indicators

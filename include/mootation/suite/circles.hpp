#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// circles_in_circle and circles_in_square (task 20, package 24-27, §24.6.2):
// N unit circles in a container of a given size, x = (x_1, y_1, ..., x_N,
// y_N) their centres; the residuals are the overlaps, the objective 1/2 ||r||^2
// (CIRC-1), the Jacobian analytic.
//
// circles_in_circle: X. Lai, J.-K. Hao, D. Yue, Z. Lü, Z.-H. Fu, "Iterated
// dynamic thresholding search for packing equal circles into a circular
// container", EJOR 299 (2022) (source iterated-dynamic-thresholding_lai2021),
// (4)-(6): the container of radius R about 0, the residuals
// O_ij = max(0, 2 - ||c_i - c_j||) for i < j (i outer), then
// O_0i = max(0, ||c_i|| + 1 - R) for each i.
//
// circles_in_square: the same pairs, then four walls a circle (CIRC-6),
// max(0, 1 - x_i), max(0, x_i + 1 - L), max(0, 1 - y_i), max(0, y_i + 1 - L):
// the square [0, L]^2 of side L.
//
// The instance's `container` is R or L. The registry takes it as the
// record's times 1 + eps (CIRC-2): Lai et al.'s Table 2 (R*, N = 66..100,
// with He et al. 2018 and Specht's Packomania) for the circle; for the square
// L* = 1 / r*, r* the best known radius of N circles in the unit square of A.
// Costa, P. Hansen, L. Liberti, "On the impact of symmetry-breaking
// constraints on spatial Branch-and-Bound for circle packing in a square"
// (source symmetry-breaking-circle-packing_costa2011_preprint), Table 4 (from
// Packomania; N = 20, 25, ..., 50).
//
// Symmetries: permutations of the circles (blocks: N of 2); the circle's
// rotation about its centre, (-y_i, x_i) for each circle (gauge_dim 1); the
// square's eight symmetries are not taken (CIRC-5). canon: the circles sorted.
// The box: [-R, R]^2N or [0, L]^2N (CIRC-3). The start: centres uniform in
// the box, from (20261102, crc32("<family>:start"), instance, N).
//
// READINGS:
//   CIRC-1. The objective is 1/2 E, E the papers' sum of squared overlaps
//     (the suite's least-squares convention): E = 0, a packing, is f = 0.
//   CIRC-2. The records are rounded (Lai et al. to 9 decimals, Costa et al.'s
//     r* to 6: below 1e-5 relative); at eps >= 1e-3 a packing exists in the
//     container, f_opt = 0 exactly. The record itself is the strategies'
//     business (the container shrunk, Lai et al.'s §2.2), not the problem's.
//   CIRC-3. The box is the square around the container: a centre outside
//     the container is penalized by its walls, not refused.
//   CIRC-4. The Jacobian of max(0, t) at t = 0 is 0 (the feasible side's);
//     two centres at distance 0 give their pair 0 (no direction).
//   CIRC-5. canon sorts the circles (lexicographically); the container's
//     rotations and reflections are left in it (the circle's rotation is
//     the gauge direction).
//   CIRC-6. The square's walls are Lai et al. 2025's (6), O_ik = max(0,
//     g_k(c_i)), for the unit circles' centres in [1, L - 1]^2 (source
//     heuristic-algorithm-multi_lai2025, §2.1). Locatelli, Maischberger &
//     Schoen's point spreading (source differential-evolution-methods_
//     locatelli2013, (1)) is the same problem at d = 2 / (L - 2) in the unit
//     square, constrained and not penalized there.
// ============================================================================

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "sym.hpp"

namespace mootation::suite::sym {

// Lai et al. 2022, Table 2: the record R* (NaN elsewhere)
inline double circle_record(int N)
{
    switch (N) {
        case 66: return 9.096279427;
        case 82: return 10.050824224;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

// Costa et al., Table 4: the record r* of N circles in the unit square (NaN elsewhere)
inline double square_record_radius(int N)
{
    switch (N) {
        case 20: return 0.111382;
        case 25: return 0.1;
        case 30: return 0.091671;
        case 35: return 0.084290;
        case 40: return 0.079186;
        case 45: return 0.074727;
        case 50: return 0.071377;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

class Circles final : public Family {
public:
    int circles = 0;
    double size = 0.0;           // R or L
    bool in_circle = true;

    Circles(const Instance& in, bool circle) : in_circle(circle), name_(circle ? "circles_in_circle" : "circles_in_square")
    {
        if (in.dim < 2) throw std::invalid_argument(name_ + ": N >= 2 circles");
        if (!(in.container > (circle ? 1.0 : 2.0)))
            throw std::invalid_argument(name_ + (circle ? ": container R > 1" : ": container L > 2"));
        circles = in.dim;
        size = in.container;
        n = 2 * circles;
        m = circles * (circles - 1) / 2 + (circle ? circles : 4 * circles);
        has_jacobian = has_canon = true;
        gauge_dim = circle ? 1 : 0;
        blocks = {{0, 2, circles}};
        bounds = circle ? box(n, -size, size) : box(n, 0.0, size);
        instance_ = in.instance;
    }

    void residuals(const double* x, double* r, double* J) const override
    {
        if (J) std::fill(J, J + static_cast<std::size_t>(m) * n, 0.0);
        int k = 0;
        for (int i = 0; i < circles; ++i)
            for (int j = i + 1; j < circles; ++j, ++k) {
                const double dx = x[2 * i] - x[2 * j], dy = x[2 * i + 1] - x[2 * j + 1];
                const double d = std::sqrt(dx * dx + dy * dy);
                r[k] = d < 2.0 ? 2.0 - d : 0.0;
                if (J && d < 2.0 && d > 0.0) {                 // CIRC-4
                    double* row = J + static_cast<std::size_t>(k) * n;
                    row[2 * i] = -dx / d;
                    row[2 * i + 1] = -dy / d;
                    row[2 * j] = dx / d;
                    row[2 * j + 1] = dy / d;
                }
            }
        for (int i = 0; i < circles; ++i) {
            const double cx = x[2 * i], cy = x[2 * i + 1];
            if (in_circle) {
                const double d = std::sqrt(cx * cx + cy * cy), t = d + 1.0 - size;
                r[k] = t > 0.0 ? t : 0.0;
                if (J && t > 0.0) {
                    J[static_cast<std::size_t>(k) * n + 2 * i] = cx / d;
                    J[static_cast<std::size_t>(k) * n + 2 * i + 1] = cy / d;
                }
                ++k;
                continue;
            }
            const double t[4] = {1.0 - cx, cx + 1.0 - size, 1.0 - cy, cy + 1.0 - size};
            const double dt[4] = {-1.0, 1.0, -1.0, 1.0};
            for (int w = 0; w < 4; ++w, ++k) {
                r[k] = t[w] > 0.0 ? t[w] : 0.0;
                if (J && t[w] > 0.0) J[static_cast<std::size_t>(k) * n + 2 * i + w / 2] = dt[w];
            }
        }
    }

    void gauge(const double* x, double* N) const override       // the circle's rotation about 0
    {
        for (int i = 0; i < circles; ++i) {
            N[2 * i] = -x[2 * i + 1];
            N[2 * i + 1] = x[2 * i];
        }
    }

    void canonical(double* x) const override                  // CIRC-5
    {
        sort_blocks(x, blocks[0], [](const Vec&) { return 0.0; });
    }

    void start(double* x) const override
    {
        std::mt19937 g = generator({kKey, instances::crc32(name_ + ":start"), static_cast<std::uint64_t>(instance_),
                                    static_cast<std::uint64_t>(circles)});
        const double lo = bounds[0].first, hi = bounds[0].second;
        for (int i = 0; i < n; ++i) x[i] = lo + (hi - lo) * rnd::uniform01(g);
    }

private:
    std::string name_;
    int instance_ = 1;
};

}   // namespace mootation::suite::sym

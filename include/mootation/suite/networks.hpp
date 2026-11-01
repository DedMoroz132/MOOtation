#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Teacher-student networks (task 19, §19.1.2 rbfnet, §19.1.3 mlp_plateau):
// the data are N points u_i of R^d and the teacher's outputs y_i there; the
// student is a network of the same class; f_opt = 0 where it has the
// teacher's neurons or more and there is no noise. Least squares, the
// Jacobian analytic.
//
// rbfnet: f(u) = sum_k c_k exp(-||u - mu_k||^2 / (2 s_k^2)). The unknowns per
// neuron (c_k, mu_k (d), l_k = log s_k), neuron after neuron (blocks: K of d
// + 2); canon sorts the neurons by c, then the whole block (SYM-3). The box
// (RB-1): c in [-3, 3], mu in [-2, 2]^d, l in [log 0.05, log 2].
// Draws from the key (20261102, crc32("rbfnet"), instance, d, K_t, N): per
// teacher neuron the sign of c (uniform < 1/2: -1), |c| = 0.5 + U, mu = -1 + 2
// U per coordinate, s = 0.3 + 0.4 U; then the points, -1 + 2 U per
// coordinate, point after point; then the noise (sym.hpp, SYM-1). Start, from
// (20261102, crc32("rbfnet:start"), instance, d, K_s, N): c = -1 + 2 U, mu =
// -1 + 2 U, l = log 0.5.
//
// mlp_plateau: the perceptron of K. Fukumizu, S. Amari, "Local minima and
// plateaus in hierarchical structures of multilayer perceptrons", Neural
// Networks 13 (2000) 317-327 (source mlp-plateaus-local-minima_fukumizu2000;
// the 1999 preprint), (1): f(u) = v_0 + sum_j v_j tanh(w_j0 + w_j^T u), with
// a teacher of H + 1 hidden units and a student of as many. The unknowns per
// unit (v_j, w_j0, w_j (d)), unit after unit (blocks: H + 1 of d + 2), then
// v_0. Symmetries: the units may be permuted, and (v_j, w_j) -> (-v_j, -w_j)
// (tanh is odd); canon makes the largest |entry| of (w_j0, w_j) positive (the
// first such on ties), then sorts the units by w_j0, then the whole block.
// No box. Draws from (20261102, crc32("mlp_plateau"), instance, d, H + 1, N):
// per teacher unit v_j, w_j0, w_j (normal), then v_0, then the points (normal,
// point after point), then the noise.
//
// The starts (variant): "random", normal entries from (20261102,
// crc32("mlp_plateau:start"), instance, d, H + 1, N); or on a line of
// embedding of Fukumizu & Amari §3, (18) (p. 7 of the preprint): a network of
// H units is fitted to the data first — lm (scalar/lm.hpp, its defaults, the
// Jacobian analytic) from 8 starts, normal entries from (20261102,
// crc32("mlp_plateau:small"), instance, d, H, s), s = 1..8, the least cost
// (the first on ties) — and embedded into H + 1 units:
//   "duplicate", their gamma_lambda: unit 1 of the small network split into
//     two with output weights lambda v_1 and (1 - lambda) v_1 (lambda =
//     embed), the copy appended as unit H + 1; a critical point for every
//     lambda (their Theorem 1), a local minimum or a saddle by the sign of
//     lambda (1 - lambda) and their A_2 (Theorem 3);
//   "bias", their beta_(0,w): a unit H + 1 with output weight 0 and input
//     weight (w, 0, ..., 0) (w = embed), v_0 unchanged (their zeta_0 - 0
//     tanh(w)); a critical point (Theorem 2), a saddle (Theorem 4).
// Their third embedding, alpha_w (a unit with output weight 0 and any input
// weight), gives no critical point (dE/dv = sum r_i tanh(w^T u_i) != 0) and
// is not offered. At either start the gradient is that of the small network
// at its minimum, carried over: (lambda, 1 - lambda) times unit 1's for the
// copies' input weights, unit 1's output-weight entry for both copies;
// tanh(w) times v_0's entry for the new output weight, 0 for its input
// weights (MLP-2).
//
// READINGS:
//   RB-1. rbfnet's box and its teacher's ranges are ours (the task gives the
//     model only).
//   MLP-1. The teacher's and the points' distributions (standard normal) and
//     the eight starts of the small fit are ours; Fukumizu & Amari prove the
//     embeddings, they prescribe no instance.
//   MLP-2. The small network's minimum is lm's to its tolerances, so "the
//     gradient at the embedded point" is the carried-over gradient of that
//     point exactly (to rounding), not zero.
// ============================================================================

#include <cmath>
#include <stdexcept>
#include <string>

#include "../scalar/lm.hpp"
#include "sym.hpp"

namespace mootation::suite::sym {

class Rbfnet final : public Family {
public:
    int d = 0, K = 0;
    Vec u, y;                                     // N points of d, row after row; the outputs

    explicit Rbfnet(const Instance& in)
    {
        if (in.dim < 1 || in.teacher < 1 || in.points < 1)
            throw std::invalid_argument("rbfnet: d >= 1, K_t >= 1, N >= 1");
        d = in.dim;
        const int w = d + 2;
        std::mt19937 g = generator({kKey, instances::crc32("rbfnet"), static_cast<std::uint64_t>(in.instance),
                                    static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(in.teacher),
                                    static_cast<std::uint64_t>(in.points)});
        Vec teacher(static_cast<std::size_t>(in.teacher * w));
        for (int k = 0; k < in.teacher; ++k) {
            double* p = teacher.data() + k * w;
            const double sign = rnd::uniform01(g) < 0.5 ? -1.0 : 1.0;
            p[0] = sign * (0.5 + rnd::uniform01(g));
            for (int l = 0; l < d; ++l) p[1 + l] = -1.0 + 2.0 * rnd::uniform01(g);
            p[d + 1] = dm::log(0.3 + 0.4 * rnd::uniform01(g));
        }
        u.resize(static_cast<std::size_t>(in.points * d));
        for (double& v : u) v = -1.0 + 2.0 * rnd::uniform01(g);
        y.resize(static_cast<std::size_t>(in.points));
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = model(teacher.data(), in.teacher, u.data() + static_cast<int>(i) * d);
        add_noise(g, in.noise, y);
        K = in.student > 0 ? in.student : in.teacher;
        if (K == in.teacher) optimum = teacher;
        n = K * w;
        m = in.points;
        has_jacobian = has_canon = true;
        blocks = {{0, w, K}};
        for (int k = 0; k < K; ++k) {                 // RB-1
            bounds.emplace_back(-3.0, 3.0);
            for (int l = 0; l < d; ++l) bounds.emplace_back(-2.0, 2.0);
            bounds.emplace_back(dm::log(0.05), dm::log(2.0));
        }
        instance_ = in.instance;
    }

    void residuals(const double* x, double* r, double* J) const override
    {
        const int w = d + 2;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double* ui = u.data() + static_cast<int>(i) * d;
            r[i] = model(x, K, ui) - y[i];
            if (!J) continue;
            double* row = J + static_cast<int>(i) * n;
            for (int k = 0; k < K; ++k) {
                const double* p = x + k * w;
                const double inv = dm::exp(-2.0 * p[d + 1]);
                double q = 0.0;
                for (int l = 0; l < d; ++l) q += (ui[l] - p[1 + l]) * (ui[l] - p[1 + l]);
                q = q * inv;
                const double phi = dm::exp(-0.5 * q);
                row[k * w] = phi;
                for (int l = 0; l < d; ++l) row[k * w + 1 + l] = p[0] * phi * (ui[l] - p[1 + l]) * inv;
                row[k * w + d + 1] = p[0] * phi * q;
            }
        }
    }

    void canonical(double* x) const override
    {
        sort_blocks(x, blocks[0], [](const Vec& b) { return b[0]; });
    }

    void start(double* x) const override
    {
        std::mt19937 g = generator({kKey, instances::crc32("rbfnet:start"), static_cast<std::uint64_t>(instance_),
                                    static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(K),
                                    static_cast<std::uint64_t>(m)});
        const int w = d + 2;
        for (int k = 0; k < K; ++k) {
            for (int l = 0; l <= d; ++l) x[k * w + l] = -1.0 + 2.0 * rnd::uniform01(g);
            x[k * w + d + 1] = dm::log(0.5);
        }
    }

private:
    int instance_ = 1;

    double model(const double* p, int units, const double* ui) const
    {
        const int w = d + 2;
        double sum = 0.0;
        for (int k = 0; k < units; ++k) {
            const double* q = p + k * w;
            double s = 0.0;
            for (int l = 0; l < d; ++l) s += (ui[l] - q[1 + l]) * (ui[l] - q[1 + l]);
            sum += q[0] * dm::exp(-0.5 * (s * dm::exp(-2.0 * q[d + 1])));
        }
        return sum;
    }
};

class MlpPlateau final : public Family {
public:
    int d = 0, H = 0;                             // inputs, the student's hidden units
    Vec u, y;
    std::string variant;
    double embed = 0.5;

    explicit MlpPlateau(const Instance& in) : variant(in.variant.empty() ? "random" : in.variant), embed(in.embed)
    {
        if (in.dim < 1 || in.teacher < 2 || in.points < 1)
            throw std::invalid_argument("mlp_plateau: d >= 1, H + 1 >= 2 teacher units, N >= 1");
        if (variant != "random" && variant != "duplicate" && variant != "bias")
            throw std::invalid_argument("mlp_plateau: start random, duplicate or bias");
        if (in.student != 0 && in.student != in.teacher)
            throw std::invalid_argument("mlp_plateau: the student has the teacher's H + 1 units");
        d = in.dim;
        const int w = d + 2, units = in.teacher;
        std::mt19937 g = generator({kKey, instances::crc32("mlp_plateau"), static_cast<std::uint64_t>(in.instance),
                                    static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(units),
                                    static_cast<std::uint64_t>(in.points)});
        Vec teacher(static_cast<std::size_t>(units * w + 1));
        for (double& v : teacher) v = rnd::normal01(g);
        u.resize(static_cast<std::size_t>(in.points * d));
        for (double& v : u) v = rnd::normal01(g);
        y.resize(static_cast<std::size_t>(in.points));
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = model(teacher.data(), units, u.data() + static_cast<int>(i) * d);
        add_noise(g, in.noise, y);
        H = units;
        optimum = teacher;
        n = units * w + 1;
        m = in.points;
        has_jacobian = has_canon = true;
        blocks = {{0, w, units}};
        bounds = box(n, -kInf, kInf);
        instance_ = in.instance;
    }

    void residuals(const double* x, double* r, double* J) const override { eval(x, H, r, J); }

    void canonical(double* x) const override
    {
        const int w = d + 2;
        for (int j = 0; j < H; ++j) {
            double* p = x + j * w;
            int at = 1;
            for (int l = 2; l < w; ++l)
                if (std::fabs(p[l]) > std::fabs(p[at])) at = l;
            if (p[at] < 0.0)
                for (int l = 0; l < w; ++l) p[l] = -p[l];
        }
        sort_blocks(x, blocks[0], [](const Vec& b) { return b[1]; });
    }

    void start(double* x) const override
    {
        if (variant == "random") {
            std::mt19937 g = generator({kKey, instances::crc32("mlp_plateau:start"),
                                        static_cast<std::uint64_t>(instance_), static_cast<std::uint64_t>(d),
                                        static_cast<std::uint64_t>(H), static_cast<std::uint64_t>(m)});
            for (int i = 0; i < n; ++i) x[i] = rnd::normal01(g);
            return;
        }
        const Vec small = fit_small();
        embed_into(small, x);
    }

    // the H - 1 unit network's fit from 8 starts, the least cost
    Vec fit_small() const
    {
        const int w = d + 2, units = H - 1;
        namespace lm = scalar::lm;
        lm::Problem p;
        p.n = static_cast<std::size_t>(units * w + 1);
        p.m = static_cast<std::size_t>(m);
        p.residuals = [&](const Vec& x, Vec& r) {
            r.resize(p.m);
            eval(x.data(), units, r.data(), nullptr);
        };
        p.jacobian = [&](const Vec& x, lm::Matrix& Jm) {
            Jm = lm::Matrix(p.m, p.n);
            Vec r(p.m);
            eval(x.data(), units, r.data(), Jm.a.data());
        };
        Vec best;
        double best_cost = kInf;
        for (int s = 1; s <= 8; ++s) {
            std::mt19937 g = generator({kKey, instances::crc32("mlp_plateau:small"),
                                        static_cast<std::uint64_t>(instance_), static_cast<std::uint64_t>(d),
                                        static_cast<std::uint64_t>(units), static_cast<std::uint64_t>(s)});
            Vec x0(p.n);
            for (double& v : x0) v = rnd::normal01(g);
            const lm::Result r = lm::solve(p, x0, lm::Options());
            if (r.cost < best_cost) {
                best_cost = r.cost;
                best = r.x;
            }
        }
        return best;
    }

    // the small network (H - 1 units, then v_0) embedded into H units (variant)
    void embed_into(const Vec& small, double* x) const
    {
        const int w = d + 2, units = H - 1;
        std::copy(small.begin(), small.begin() + units * w, x);
        double* added = x + units * w;
        if (variant == "duplicate") {                 // gamma_lambda: unit 1 split
            std::copy(small.begin(), small.begin() + w, added);
            x[0] = embed * small[0];
            added[0] = (1.0 - embed) * small[0];
        } else {                                      // beta_(0,w): output 0, input (w, 0, ..., 0)
            std::fill(added, added + w, 0.0);
            added[1] = embed;
        }
        x[n - 1] = small[static_cast<std::size_t>(units * w)];
    }

    // r (m) and J (m x (units (d + 2) + 1)) of a network of `units` units
    void eval(const double* x, int units, double* r, double* J) const
    {
        const int w = d + 2, cols = units * w + 1;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double* ui = u.data() + static_cast<int>(i) * d;
            r[i] = model(x, units, ui) - y[i];
            if (!J) continue;
            double* row = J + static_cast<int>(i) * cols;
            for (int j = 0; j < units; ++j) {
                const double* p = x + j * w;
                double z = p[1];
                for (int l = 0; l < d; ++l) z += p[2 + l] * ui[l];
                const double h = sym::tanh(z), dz = p[0] * (1.0 - h * h);
                row[j * w] = h;
                row[j * w + 1] = dz;
                for (int l = 0; l < d; ++l) row[j * w + 2 + l] = dz * ui[l];
            }
            row[cols - 1] = 1.0;
        }
    }

private:
    int instance_ = 1;

    // v_0 + sum_j v_j tanh(w_j0 + w_j^T u), v_0 after the units
    double model(const double* p, int units, const double* ui) const
    {
        const int w = d + 2;
        double sum = p[units * w];
        for (int j = 0; j < units; ++j) {
            const double* q = p + j * w;
            double z = q[1];
            for (int l = 0; l < d; ++l) z += q[2 + l] * ui[l];
            sum += q[0] * sym::tanh(z);
        }
        return sum;
    }
};

}   // namespace mootation::suite::sym

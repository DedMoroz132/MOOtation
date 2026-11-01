#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Problems with symmetries at one objective (task 19, §19.1): what the
// families sumexp (sumexp.hpp), lowrank, cp3, cp3_degenerate (factor.hpp),
// rbfnet, mlp_plateau (networks.hpp), chaboche1d (chaboche.hpp) and
// lj_cluster (lj.hpp) share. Each is a teacher-student problem (but
// lj_cluster): the data come from a teacher of the model class, the unknowns
// are the student's, the residuals the student's model minus the data, the
// objective 1/2 ||r||^2; lj_cluster's objective is the cluster's energy, with
// its gradient.
//
// An instance is named by its family and the numbers of Instance below; what
// it draws (teachers, data points, noise, starts) comes from the library's
// generator (random.hpp) seeded with a key of the family's name and those
// numbers (instances.hpp: seed), each family saying in which order it draws.
// benchmarks/symmetric.py draws the same numbers in Python, the
// cross-check's.
//
// Each family states its symmetries as the suite's interface asks
// (native.hpp): blocks of interchangeable coordinates, canon(x), the gauge
// directions; and its own start (start). The noise of the data is relative:
// sigma = noise max_i |y_i| of the noiseless data (SYM-1), so 1 / noise is the
// signal-to-noise ratio.
//
// READINGS (shared):
//   SYM-1. The noise's sigma is a share of the largest |datum|: the sources
//     name signal-to-noise ratios (Istratov & Vyvenko's (13)) without saying
//     which amplitude; we take the largest.
//   SYM-2. canon(x) and idempotence bit for bit: a canonical form computed
//     in floating point is not exactly a fixed point of itself (a balanced
//     norm or a centre recomputed is a last bit off). Where canon rescales or
//     moves (cp3, lj_cluster), a point that already satisfies the canonical
//     conditions within a tolerance of its scale (the family says which) is
//     left as it is, so canon(canon(
//     x)) = canon(x) bit for bit; f(canon(x)) = f(x) to the order of the sums.
//     Interchanged blocks and flipped signs are exact, so canon(P(x)) =
//     canon(x) bit for bit for every permutation P of the blocks (sign flips
//     of the symmetry included).
//   SYM-3. Ties: blocks are ordered by the family's key, then the whole block
//     lexicographically, so that the order is total and permutation copies
//     come out the same.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"
#include "common.hpp"
#include "instances.hpp"
#include "native.hpp"

namespace mootation::suite::sym {

using Vec = std::vector<double>;

inline constexpr double kEps = std::numeric_limits<double>::epsilon();
inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr std::uint64_t kKey = 20261102;      // the first word of every instance's key

// The numbers of an instance; each family reads its own (the comments) and
// refuses what it cannot take.
struct Instance {
    int teacher = 0;              // sumexp K_t; lowrank, cp3 the rank; rbfnet, mlp_plateau neurons; chaboche1d N_t
    int student = 0;              // the same of the student (0: the teacher's)
    int points = 0;               // sumexp, rbfnet, mlp_plateau: data points
    int dim = 0;                  // rbfnet, mlp_plateau: the input's dimension; lj_cluster: atoms
    std::vector<int> shape;       // lowrank (m, n); cp3, cp3_degenerate (I, J, K)
    double noise = 0.0;           // SYM-1
    double ratio = 0.0;           // sumexp: tau_{k+1} / tau_k
    double observed = 1.0;        // lowrank: the share of entries observed
    std::string variant;          // sumexp "lanczos"; mlp_plateau's start "random", "duplicate" or "bias"
    double embed = 0.5;           // mlp_plateau: duplicate's lambda, bias's w
    int instance = 1;
};

// A family's problem: n parameters, m residuals (0: lj_cluster, an objective
// with its gradient), the box, the symmetries.
class Family {
public:
    int n = 0, m = 0;
    bool has_jacobian = false, has_gradient = false, has_canon = false, has_start = true;
    int gauge_dim = 0;
    std::vector<Blocks> blocks;
    std::vector<int> linear;
    Bounds bounds;
    Vec optimum;                  // the teacher's point where the student has its size (else empty)

    virtual ~Family() = default;
    // r = student - data (m values); J (m x n, row-major) where not null
    virtual void residuals(const double*, double*, double*) const {}
    // the objective where m = 0
    virtual double value(const double*) const { return 0.0; }
    virtual void gradient(const double*, double*) const {}
    virtual void canonical(double*) const {}
    virtual void gauge(const double*, double*) const {}
    virtual void start(double*) const {}
};

// tanh(z) = 1 - 2 / (exp(2z) + 1) by dm::exp, odd bit for bit
inline double tanh(double z)
{
    const double t = 1.0 - 2.0 / (dm::exp(2.0 * std::fabs(z)) + 1.0);
    return z < 0.0 ? -t : t;
}

// sigma = noise max |y| (SYM-1), one normal draw a datum, in order
inline void add_noise(std::mt19937& g, double noise, Vec& y)
{
    if (noise <= 0.0) return;
    double big = 0.0;
    for (double v : y) big = std::max(big, std::fabs(v));
    const double sigma = noise * big;
    for (double& v : y) v = v + sigma * rnd::normal01(g);
}

// The blocks of x (count of size from offset) in increasing order of key(block),
// ties by the block's entries in order (SYM-3)
template <typename Key>
inline void sort_blocks(double* x, const Blocks& b, Key key)
{
    const std::size_t size = static_cast<std::size_t>(b.size);
    std::vector<Vec> v(static_cast<std::size_t>(b.count));
    for (std::size_t k = 0; k < v.size(); ++k)
        v[k].assign(x + b.offset + k * size, x + b.offset + (k + 1) * size);
    std::stable_sort(v.begin(), v.end(), [&](const Vec& p, const Vec& q) {
        const double kp = key(p), kq = key(q);
        if (kp != kq) return kp < kq;
        return p < q;
    });
    for (std::size_t k = 0; k < v.size(); ++k) std::copy(v[k].begin(), v[k].end(), x + b.offset + k * size);
}

inline double norm(const double* v, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

// the families of sym.hpp (suite.hpp makes them)
inline bool is_family(const std::string& f)
{
    for (const char* name : {"sumexp", "lowrank", "cp3", "cp3_degenerate", "rbfnet", "mlp_plateau", "chaboche1d",
                             "lj_cluster"})
        if (f == name) return true;
    return false;
}

// the generator of an instance's key
inline std::mt19937 generator(std::initializer_list<std::uint64_t> key)
{
    return std::mt19937(instances::seed(key));
}

}   // namespace mootation::suite::sym

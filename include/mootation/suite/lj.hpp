#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// lj_cluster (task 19, §19.1.5): the Lennard-Jones cluster of N atoms,
// E = 4 sum_{i<j} (r_ij^-12 - r_ij^-6), x in R^3N (atom after atom), in
// reduced units; the gradient analytic. One objective, not least squares.
//
// The minima: the best known energies of D.J. Wales, J.P.K. Doye, "Global
// optimization by basin-hopping and the lowest energy structures of
// Lennard-Jones clusters containing up to 110 atoms", J. Phys. Chem. A 101
// (1997) 5111-5116 (source basin-hopping_wales1997_arxiv), Table I
// (best_known); Table II's lowest icosahedral minima where the global one is
// not icosahedral (icosahedral: N = 38, 75), the trap of the double funnel,
// a known local level.
//
// The box [-L, L]^3N, L = 1 + (3N / (4 pi))^(1/3) (LJ-1): Wales & Doye's
// container (§2: "radius one plus the value required to contain the same
// volume per atom as the fcc primitive cell"; the fcc cell at the pair
// minimum's distance 2^(1/6) has volume 1 per atom).
//
// Symmetries: translations, rotations (and reflections: E depends on the
// distances alone), permutations of the atoms. blocks: N of 3; gauge_basis
// the three translations and the three rotations about the axes, (0, -z, y),
// (z, 0, -x), (-y, x, 0) per atom (zero directions of the gradient, LJ-3).
// canon (LJ-2): the atoms sorted lexicographically; then, unless the point is
// canonical already, the centre of mass to 0, the axes to the eigenvectors of
// sum_i r_i r_i^T (decreasing eigenvalue), each axis's sign so that sum_i
// r_i,axis^3 >= 0, the atoms sorted again.
//
// The start: atoms uniform in the box, from (20261102, crc32("lj_cluster:
// start"), instance, N).
//
// Cluster below holds all but the energy: morse_cluster (morse.hpp; task 20,
// §24.6.2) takes it with its own volume per atom (MORSE-1).
//
// READINGS:
//   LJ-1. Wales & Doye's container is a sphere, used only in their local
//     minimizations to keep atoms from evaporating; ours is the cube of the
//     same half-width, the box of the problem.
//   LJ-2. The principal axes are not unique where two moments are equal (the
//     13- and 55-atom icosahedra have three equal ones): canon is then a
//     canonical form up to that rotation — still f(canon(x)) = f(x),
//     canon(canon(x)) = canon(x) and canon(P(x)) = canon(x) for permutations P
//     of the atoms. "Canonical already" (sym.hpp, SYM-2): the atoms sorted,
//     |centre| <= 1e-10 s, the off-diagonal of sum_i r_i r_i^T within 1e-10 N
//     s^2 and its diagonal not increasing (to the same), the third moments >= -1e-10
//     s^3, s = max |coordinate|; a third moment within that of 0 keeps its
//     sign.
//   LJ-3. Wales & Doye publish no coordinates (their database is not taken:
//     its licence is not checked); f(x_opt) = f_opt is checked where the
//     structure can be built: the 13-atom icosahedron (tests/test_suite.cpp).
// ============================================================================

#include <cmath>
#include <stdexcept>
#include <string>

#include "../scalar/eigen.hpp"
#include "sym.hpp"

namespace mootation::suite::sym {

// Table I of Wales & Doye for the task's sizes (NaN elsewhere)
inline double lj_best_known(int N)
{
    switch (N) {
        case 13: return -44.326801;
        case 19: return -72.659782;
        case 26: return -108.315616;
        case 38: return -173.928427;
        case 55: return -279.248470;
        case 75: return -397.492331;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

// Table II: the lowest icosahedral minimum where the global one is not (NaN elsewhere)
inline double lj_icosahedral(int N)
{
    return N == 38 ? -173.252378 : N == 75 ? -396.282249 : std::numeric_limits<double>::quiet_NaN();
}

// What an atom cluster of lj_cluster and morse_cluster (morse.hpp) shares: N
// atoms in R^3N, the box, the blocks, the gauge directions, canon and the
// start; each gives its energy and gradient.
class Cluster : public Family {
public:
    int atoms = 0;
    double L = 0.0;

    void gauge(const double* x, double* N) const override
    {
        std::fill(N, N + static_cast<std::size_t>(n) * 6, 0.0);
        for (int i = 0; i < atoms; ++i) {
            const double* r = x + 3 * i;
            double* row = N + static_cast<std::size_t>(3 * i) * 6;    // rows of x, y, z of atom i
            for (int c = 0; c < 3; ++c) row[c * 6 + c] = 1.0;        // translations
            row[1 * 6 + 3] = -r[2];                                   // about e_x: (0, -z, y)
            row[2 * 6 + 3] = r[1];
            row[0 * 6 + 4] = r[2];                                    // about e_y: (z, 0, -x)
            row[2 * 6 + 4] = -r[0];
            row[0 * 6 + 5] = -r[1];                                   // about e_z: (-y, x, 0)
            row[1 * 6 + 5] = r[0];
        }
    }

    void canonical(double* x) const override                 // LJ-2
    {
        auto by_entries = [](const Vec&) { return 0.0; };
        sort_blocks(x, blocks[0], by_entries);
        if (is_canonical(x)) return;
        double centre[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < atoms; ++i)
            for (int c = 0; c < 3; ++c) centre[c] += x[3 * i + c];
        for (double& c : centre) c = c / atoms;
        for (int i = 0; i < atoms; ++i)
            for (int c = 0; c < 3; ++c) x[3 * i + c] = x[3 * i + c] - centre[c];
        eigen::Matrix S = second_moments(x), V;
        std::vector<double> w;
        eigen::symmetric(S, w, V);                          // increasing eigenvalues
        Vec y(static_cast<std::size_t>(n));
        for (int i = 0; i < atoms; ++i)
            for (int a = 0; a < 3; ++a) {                   // axis a: eigenvector 2 - a
                double s = 0.0;
                for (int c = 0; c < 3; ++c) s += V(static_cast<std::size_t>(c), static_cast<std::size_t>(2 - a)) * x[3 * i + c];
                y[static_cast<std::size_t>(3 * i + a)] = s;
            }
        const double scale = biggest(y.data());
        for (int a = 0; a < 3; ++a) {
            double third = 0.0;
            for (int i = 0; i < atoms; ++i) {
                const double v = y[static_cast<std::size_t>(3 * i + a)];
                third += v * v * v;
            }
            if (third < -1e-10 * scale * scale * scale)
                for (int i = 0; i < atoms; ++i) y[static_cast<std::size_t>(3 * i + a)] = -y[static_cast<std::size_t>(3 * i + a)];
        }
        std::copy(y.begin(), y.end(), x);
        sort_blocks(x, blocks[0], by_entries);
    }

    void start(double* x) const override
    {
        std::mt19937 g = generator({kKey, instances::crc32(name_ + ":start"), static_cast<std::uint64_t>(instance_),
                                    static_cast<std::uint64_t>(atoms)});
        for (int i = 0; i < n; ++i) x[i] = -L + 2.0 * L * rnd::uniform01(g);
    }

protected:
    // volume: the volume per atom of the fcc cell at the pair potential's
    // minimum (LJ-1; lj_cluster's 1)
    Cluster(const Instance& in, const std::string& name, double volume) : name_(name)
    {
        if (in.dim < 2) throw std::invalid_argument(name + ": N >= 2 atoms");
        atoms = in.dim;
        L = 1.0 + dm::pow(3.0 * atoms * volume / (4.0 * dm::kPi), 1.0 / 3.0);   // LJ-1
        n = 3 * atoms;
        m = 0;
        has_gradient = has_canon = true;
        gauge_dim = 6;
        blocks = {{0, 3, atoms}};
        bounds = box(n, -L, L);
        instance_ = in.instance;
    }

    double squared(const double* x, int i, int j) const
    {
        double s = 0.0;
        for (int c = 0; c < 3; ++c) s += (x[3 * i + c] - x[3 * j + c]) * (x[3 * i + c] - x[3 * j + c]);
        return s;
    }

private:
    std::string name_;
    int instance_ = 1;

    double biggest(const double* x) const
    {
        double big = 0.0;
        for (int i = 0; i < n; ++i) big = std::max(big, std::fabs(x[i]));
        return big;
    }

    eigen::Matrix second_moments(const double* x) const
    {
        eigen::Matrix S(3, 3);
        for (int i = 0; i < atoms; ++i)
            for (std::size_t a = 0; a < 3; ++a)
                for (std::size_t b = 0; b < 3; ++b) S(a, b) += x[3 * i + static_cast<int>(a)] * x[3 * i + static_cast<int>(b)];
        return S;
    }

    // the atoms sorted, centred, on their principal axes in decreasing order,
    // the third moments not negative (LJ-2's tolerances)
    bool is_canonical(const double* x) const
    {
        for (int i = 0; i + 1 < atoms; ++i)
            if (std::lexicographical_compare(x + 3 * (i + 1), x + 3 * (i + 2), x + 3 * i, x + 3 * (i + 1)))
                return false;
        const double s = biggest(x), tol = 1e-10;
        if (s == 0.0) return true;
        for (int c = 0; c < 3; ++c) {
            double centre = 0.0, third = 0.0;
            for (int i = 0; i < atoms; ++i) {
                centre += x[3 * i + c];
                third += x[3 * i + c] * x[3 * i + c] * x[3 * i + c];
            }
            if (std::fabs(centre / atoms) > tol * s || third < -tol * s * s * s) return false;
        }
        const eigen::Matrix S = second_moments(x);
        const double s2 = tol * s * s * atoms;
        return std::fabs(S(0, 1)) <= s2 && std::fabs(S(0, 2)) <= s2 && std::fabs(S(1, 2)) <= s2 &&
               S(1, 1) <= S(0, 0) + s2 && S(2, 2) <= S(1, 1) + s2;
    }
};

class LJCluster final : public Cluster {
public:
    explicit LJCluster(const Instance& in) : Cluster(in, "lj_cluster", 1.0) {}

    double value(const double* x) const override
    {
        double sum = 0.0;
        for (int i = 0; i < atoms; ++i)
            for (int j = i + 1; j < atoms; ++j) {
                const double s = 1.0 / squared(x, i, j), s3 = s * s * s;
                sum += s3 * s3 - s3;
            }
        return 4.0 * sum;
    }

    // dE/dx_i = 8 (x_i - x_j) de/d(r^2), de/d(r^2) = s s3 (3 - 6 s3), s = 1/r^2
    void gradient(const double* x, double* g) const override
    {
        std::fill(g, g + n, 0.0);
        for (int i = 0; i < atoms; ++i)
            for (int j = i + 1; j < atoms; ++j) {
                const double s = 1.0 / squared(x, i, j), s3 = s * s * s;
                const double de = 8.0 * (s * s3 * (3.0 - 6.0 * s3));
                for (int c = 0; c < 3; ++c) {
                    const double t = de * (x[3 * i + c] - x[3 * j + c]);
                    g[3 * i + c] += t;
                    g[3 * j + c] -= t;
                }
            }
    }
};

}   // namespace mootation::suite::sym

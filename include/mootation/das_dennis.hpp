#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// DAS-DENNIS — simplex-lattice reference point generation (Normal-Boundary
// Intersection §5).
// I. Das, J.E. Dennis — SIAM J. Optimization 8(3), 1998.
// doi:10.1137/S1052623496307510          (source: das1998)
// Two-layer extension: K. Deb, H. Jain — IEEE TEVC 18(4), 2014 (§V, Fig.4).
// doi:10.1109/TEVC.2013.2281535          (source: deb2014)
//
// Generation scheme:
//   1. Single-layer: all lattice points with step 1/H on the unit simplex,
//      N = C(H+M-1, M-1)  (das1998 §5.1-5.2).
//   2. Two-layer (deb2014 §V): the BOUNDARY (outer) layer is DENSE
//      (larger H), the inner one is sparse (smaller H), shrunk toward the
//      centroid w -> w/2 + 1/(2M); the layers are merged without deduplication.
//      M=8: boundary p=3 (120 pts) + inside p=2 (36 pts) = 156.
//   3. generate_exact (Path-A): pop_size must exactly equal the lattice
//      size, otherwise an exception with a hint of the nearest valid sizes.
//
// Defaults = das1998 §5 / deb2014 §V. Deviations: none
// (internal audit; DD-1 two-layer inversion fixed).
// Extensions beyond the papers: Path-A generate_exact/classify_size/
// suggest_sizes — utility combinatorics provided by the library.
//
// API:
//   auto V = das_dennis::generate(M, H);                 // single-layer
//   auto V = das_dennis::generate_two_layer(M, Hb, Hi);  // Hb=boundary>Hi
//   auto V = das_dennis::generate_auto(M, n);            // layer auto-selection
//
// All vectors lie on the unit simplex: w_i >= 0, sum(w_i) = 1.
// ============================================================================

#include <algorithm>   // FIX 2026-07-08 (internal audit, DD-2): std::min for the reserve cap
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mootation {
namespace das_dennis {

// ── Combinatorics ─────────────────────────────────────────────────────────────
inline long long n_vectors(int m, int H) {
    // C(H+M-1, M-1)
    if (H < 0) return 0;
    long long c = 1;
    for (int k = 0; k < m - 1; ++k) {
        // Guard against signed overflow (UB) at large H,M: further growth is
        // monotonic, so we saturate with a sentinel larger than any pop_size.
        if (c > (1LL << 50)) return (1LL << 50);
        c = c * (H + m - 1 - k) / (k + 1);
    }
    return c;
}

// Find H such that C(H+M-1,M-1) == target.
// Returns -1 if no exact match within H<=200.
inline int find_H_exact(int m, long long target) {
    for (int H = 1; H <= 200; ++H)
        if (n_vectors(m, H) == target) return H;
    return -1;
}

// Find largest H such that C(H+M-1,M-1) <= target.
inline int find_H_le(int m, long long target) {
    int best = 0;
    for (int H = 1; H <= 200; ++H) {
        long long c = n_vectors(m, H);
        if (c > target) break;
        best = H;
    }
    return best;
}

// ── Single-layer generation ───────────────────────────────────────────────────
namespace detail {
// The number of objectives is NOT passed separately: it is cur.size(), and the
// last coordinate is the one at cur.size()-1. Deriving the bound from the
// buffer being written keeps the two from ever disagreeing.
//
// The pragma is the only one in the library and is not there lightly. GCC 13
// at -O2 reports, inside <bits/stl_algobase.h> and blaming the push_back
// below, that a one-double buffer is both what got allocated and what got
// overflowed. Those two claims contradict each other, which is the tell: the
// compiler merged the m == 1 path (a one-element `cur`) into its analysis of
// the m >= 2 path. generate() now lifts m == 1 out of the recursion for that
// reason, so this should no longer fire at all.
//
// It stays as a backstop because the report has already moved once under
// exactly this treatment: silencing -Warray-bounds did not remove it, it
// renamed it to -Wstringop-overflow. Both are named here. The code is safe by
// construction — generate() refuses m < 1, sizes `cur` once and never resizes
// it, and the terminal branch fires at dim == cur.size()-1, so every push_back
// copies exactly cur.size() elements out of a cur.size()-element buffer. g++
// 20 never reported it, clang and MSVC never do; it is a middle-end false
// positive of the kind GCC has a long trail of (PR 109442 and relatives).
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#  pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
inline void recurse(int rem, std::size_t dim,
                    std::vector<double>& cur, double H_inv,
                    std::vector<std::vector<double>>& out)
{
    if (dim + 1 == cur.size()) {
        cur[dim] = rem * H_inv;
        out.push_back(cur);
        return;
    }
    for (int i = 0; i <= rem; ++i) {
        cur[dim] = i * H_inv;
        recurse(rem - i, dim + 1, cur, H_inv, out);
    }
}
} // namespace detail

// Generate single-layer Das-Dennis vectors with H divisions.
// Still inside the diagnostic region above: the report is emitted after
// inlining and is attributed to the chain ending HERE, in generate(), not to
// the push_back inside recurse() that it names.
inline std::vector<std::vector<double>> generate(int m, int H) {
    std::vector<std::vector<double>> out;
    // m < 1 is not merely degenerate, it is unsafe: `cur` below would be empty
    // and the recursion would still write cur[0]. Nothing in the library calls
    // it that way, which is exactly why the guard belongs here rather than in
    // every caller.
    if (H < 1 || m < 1) return out;
    // FIX 2026-07-08: DD-2 (MINOR).
    // Clamp reserve to a sane ceiling. n_vectors saturates with the 2^50
    // sentinel at extreme H·m; reserve(2^50) → std::bad_alloc/crash before any
    // actual filling. We cap the pre-allocation (real MOEA lattices are
    // thousands of points); beyond that the vector grows via push_back normally.
    constexpr std::size_t kReserveCap = 10'000'000;
    out.reserve(std::min<std::size_t>(static_cast<std::size_t>(n_vectors(m, H)), kReserveCap));
    // m == 1 is handled here rather than in the recursion, and not only for
    // clarity. It is the single-point lattice, so `cur` would have length 1 —
    // and GCC 13 merges that length into its analysis of the m >= 2 path,
    // producing the contradictory claim that a one-double buffer is both
    // allocated and overflowed. Lifting the case out leaves the recursion with
    // cur.size() >= 2 always. The value is computed exactly as the recursion
    // would have (H * (1/H), not the literal 1.0) so the lattice is unchanged
    // bit for bit.
    const double H_inv = 1.0 / H;
    if (m == 1) {
        out.push_back(std::vector<double>(1, static_cast<double>(H) * H_inv));
        return out;
    }
    std::vector<double> cur(static_cast<std::size_t>(m), 0.0);
    detail::recurse(H, 0, cur, H_inv, out);
    return out;
}
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif

// ── Two-layer generation ──────────────────────────────────────────────────────
// Hb — boundary (outer) layer: standard Das-Dennis; per deb2014 §V it is DENSE
//      (Hb > Hi: M=8 → boundary p=3 gives 120 points, inside p=2 gives 36).
// Hi — inner layer: Das-Dennis with Hi, then shrinking toward the centroid
//      w -> w/2 + 1/(2M) (Fig.4 "a -> a/2"; the formula preserves sum = 1).
// Combined: deduplication NOT performed (paper does not deduplicate).
inline std::vector<std::vector<double>> generate_two_layer(int m, int Hb, int Hi) {
    auto outer = generate(m, Hb);
    auto inner = generate(m, Hi);

    // Scale inner layer toward centroid (1/M, ..., 1/M)
    double center = 1.0 / m;
    for (auto& w : inner)
        for (double& wi : w)
            wi = wi / 2.0 + center / 2.0;  // = w/2 + 1/(2M)

    // Combine
    outer.insert(outer.end(), inner.begin(), inner.end());
    return outer;
}

// ── Auto selection ────────────────────────────────────────────────────────────
// Given M and desired N, pick the best layer configuration:
//   1. Try single-layer with exact match (includes M=2, H=N-1).
//   2. Try two-layer: find H1,H2 such that N1+N2 == n.
//   3. Fall back to single-layer with smallest N >= n.
// Returns vectors; actual size may differ from n by at most a few.

inline std::vector<std::vector<double>> generate_auto(int m, int n) {
    // 1. Single-layer exact (for M=2: H=n-1 always works)
    for (int H = 1; H <= 500; ++H) {
        long long c = n_vectors(m, H);
        if (c == n) return generate(m, H);
        if (c > n * 2) break;
    }

    // 2. Two-layer: pairs Hi < Hb, Ni+Nb == n.
    //    deb2014 §V: the DENSE layer (larger H) is placed on the BOUNDARY,
    //    the sparse one goes inside. Hb = the larger of the pair.
    for (int Hi = 1; Hi <= 20; ++Hi) {
        long long ni = n_vectors(m, Hi);
        if (ni >= n) break;
        long long need = n - ni;
        for (int Hb = Hi + 1; Hb <= 30; ++Hb) {
            if (n_vectors(m, Hb) == need)
                return generate_two_layer(m, Hb, Hi);
        }
    }

    // 3. Single-layer closest >= n
    for (int H = 1; H <= 500; ++H) {
        if (n_vectors(m, H) >= n)
            return generate(m, H);
    }

    return generate(m, 1);  // fallback
}

// ── Path-A: "the lattice dictates the population size" ───────────────────────
// The Das-Dennis reference lattice size is discrete: only numbers of the form
//   single-layer:  C(H+M-1, M-1)
//   two-layer:     C(H1+M-1,M-1) + C(H2+M-1,M-1),  H1 < H2
// are attainable. An arbitrary pop_size is generally unattainable. Algorithms
// (NSGA-III, RVEA, θ-DEA, the MOEA/D family, the SRV family) must have a
// pop_size EQUAL to the number of lattice vectors. The functions below
// support this.

// LayerConfig — description of the chosen lattice configuration.
struct LayerConfig {
    int  H1 = 0;     // divisions of the BOUNDARY layer — or of the only one;
                     // for two-layer always the larger of the pair (deb2014 §V)
    int  H2 = 0;     // divisions of the inner layer; 0 → single-layer
    long long size = 0;  // resulting number of vectors
    bool two_layer = false;
};

// Is exactly `n` vectors attainable for m objectives? If so, returns the
// configuration (single or two-layer) with found=true.
inline LayerConfig classify_size(int m, long long n) {
    LayerConfig cfg;
    // single-layer
    for (int H = 1; H <= 500; ++H) {
        long long c = n_vectors(m, H);
        if (c == n) { cfg.H1 = H; cfg.size = c; cfg.two_layer = false; return cfg; }
        if (c > n) break;
    }
    // two-layer: pairs Hi < Hb; the larger H goes to the boundary (deb2014 §V:
    // boundary p=3 / inside p=2 for M=8, etc.)
    for (int Hi = 1; Hi <= 60; ++Hi) {
        long long ni = n_vectors(m, Hi);
        if (ni >= n) break;
        for (int Hb = Hi + 1; Hb <= 80; ++Hb) {
            long long nb = n_vectors(m, Hb);
            if (ni + nb == n) {
                cfg.H1 = Hb; cfg.H2 = Hi; cfg.size = n;
                cfg.two_layer = true; return cfg;
            }
            if (ni + nb > n) break;
        }
    }
    cfg.size = -1;   // unattainable
    return cfg;
}

inline bool is_attainable(int m, long long n) {
    return classify_size(m, n).size == n;
}

// Nearest attainable lattice sizes <= n and >= n (for the error message).
// Returns {lower, upper}; 0 if none exists within the scanned range.
inline std::pair<long long,long long> suggest_sizes(int m, long long n) {
    long long lo = 0, hi = 0;
    // collect all attainable sizes up to a reasonable limit
    long long limit = (n < 4) ? 16 : n * 2 + 8;
    std::vector<long long> sizes;
    for (int H = 1; H <= 500; ++H) {
        long long c = n_vectors(m, H);
        if (c > limit) break;
        sizes.push_back(c);
    }
    // two-layer combinations
    for (int H1 = 1; H1 <= 60; ++H1) {
        long long n1 = n_vectors(m, H1);
        if (n1 > limit) break;
        for (int H2 = H1 + 1; H2 <= 80; ++H2) {
            long long s = n1 + n_vectors(m, H2);
            if (s > limit) break;
            sizes.push_back(s);
        }
    }
    for (long long s : sizes) {
        if (s <= n && s > lo) lo = s;
        if (s >= n && (hi == 0 || s < hi)) hi = s;
    }
    return {lo, hi};
}

// Generates exactly n vectors. n MUST be attainable (is_attainable).
// Throws std::invalid_argument with a hint otherwise.
inline std::vector<std::vector<double>> generate_exact(int m, long long n) {
    LayerConfig cfg = classify_size(m, n);
    if (cfg.size != n) {
        auto [lo, hi] = suggest_sizes(m, n);
        std::string msg = "das_dennis::generate_exact: pop_size=" +
            std::to_string(n) + " is not attainable by a Das-Dennis lattice for m=" +
            std::to_string(m) + " objectives. Nearest valid sizes: ";
        if (lo > 0) msg += std::to_string(lo);
        if (lo > 0 && hi > 0 && hi != lo) msg += " or " + std::to_string(hi);
        else if (lo == 0 && hi > 0) msg += std::to_string(hi);
        msg += ". Set pop_size to one of them.";
        throw std::invalid_argument(msg);
    }
    return cfg.two_layer ? generate_two_layer(m, cfg.H1, cfg.H2)
                         : generate(m, cfg.H1);
}

// Generation with explicitly given H1/H2 (H2<=0 → single-layer).
// H1 — the BOUNDARY layer (per deb2014 §V it must be denser: H1 > H2).
// Returns vectors; count = n_vectors(m,H1) [+ n_vectors(m,H2)].
inline std::vector<std::vector<double>> generate_layers(int m, int H1, int H2) {
    if (H2 <= 0) return generate(m, H1);
    return generate_two_layer(m, H1, H2);
}

} // namespace das_dennis
} // namespace mootation

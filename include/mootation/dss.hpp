#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Distance-based subset selection (DSS) of k points from a set, in C++: the
// same selection as python/mootation/run/archive.py `dss_order`, which every
// campaign uses to reduce a point set to N (reference fronts, the run archive,
// the sampling baselines). A core whose state is a set rather than a
// population (DMS) answers with it.
//
// In objectives normalised by the range of the points kept ((f − min)/(max −
// min), a zero range counting as 1): first the best kept point of every
// objective, then repeatedly the kept candidate the selected set covers WORST,
// "cover" being the IGD+ distance d+(c, s) = ||max(s − c, 0)|| minimised over
// the selected s (the candidate c in the place of IGD+'s reference point);
// then, the same way, the points set aside. Ties go to the lowest index, so
// the order is deterministic, and it is incremental: the first j of
// order(F, k) are order(F, j). It is archive.py's dss_order with ideal and
// nadir left None.
//
// TWO WRITINGS, ONE ORDER. order_plain() computes the selection as stated:
// every pair of points in every round of setting aside, every candidate's
// cover after every step. order() returns the same indices, bit for bit, from
// the same expressions, but holds the points in a k-d tree and computes only
// the pairs that can change something (the comment above it says which). The
// inner archive's cut after every addition (operators/inner_archive.hpp) is
// what needs it.
//
// ALMOST-DOMINATED POINTS GO LAST (2026-09-27). A point p is set aside when
// some other point q is better than p by more than DRS_LOSS in some objective
// while p is better than q by at most DRS_GAIN in every objective, in the
// normalised coordinates: a trade-off worse than 100 to 1 over a tenth of the
// range. Such points — the best value of one objective bought with a much
// worse value of another — were exactly what the per-objective seeds picked,
// and far from everything else, what the max-min steps picked next. The range
// is taken again from the points kept after every round that sets points
// aside, until a round sets none aside, so an outlier does not stretch the
// frame the rest is measured in; a round never sets aside every point left. A
// set with nothing to set aside is ordered exactly as before the rule.
//
// NO APPROXIMATION GUARANTEE. No bound holds for the covering radius DSS
// leaves — the largest d+ from a point of the set to the selection — against
// the best achievable with the same k: d+ is not symmetric and the seeds are
// forced, so the argument that bounds greedy max-min selection under a metric
// does not apply. It is a spread heuristic, not an optimal subset.
//
// Sources: DSS is Singh, Bhattacharjee & Ray, "Distance based subset
// selection for benchmarking in evolutionary multi/many-objective
// optimization", IEEE TEVC, 2019, Algo. 1 (source dss_singh2019). The procedure
// here is the one specified for MOOtation on 2026-09-22 and departs from
// Algo. 1 three times: the seeds are the M per-objective minima, the rule of
// Tanabe, Ishibuchi & Oyama (2017) that Singh et al. replace by ONE extreme
// point; the distance is d+ (Ishibuchi, Masuda, Tanigaki & Nojima, EMO 2015,
// Eq. 18, source igd-plus_ishibuchi2015 — its use in DSS after Chen, Ishibuchi &
// Shang, 2020), not the Euclidean; the almost-dominated
// points go last (Singh et al. filter nothing: they select from reference
// sets). Singh et al.'s results on the spacing of the selected points assume
// the Euclidean distance and are not claimed here.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace mootation::dss {

inline constexpr double DRS_GAIN = 1e-3;   // p's largest gain over q, at most
inline constexpr double DRS_LOSS = 0.1;    // p's largest loss to q, more than

// The selection as stated above, every pair computed: order()'s definition and
// the tests' reference.
inline std::vector<std::size_t> order_plain(const std::vector<std::vector<double>>& F, std::size_t k)
{
    const std::size_t n = F.size();
    k = std::min(k, n);
    std::vector<std::size_t> out;
    if (k == 0) return out;
    const std::size_t m = F[0].size();
    const double inf = std::numeric_limits<double>::infinity();

    // Normalise by the kept points' range and set aside the almost-dominated
    // ones, until a round sets none aside (archive.py set_aside).
    std::vector<char> kept(n, 1);
    std::vector<std::vector<double>> G(n, std::vector<double>(m));
    for (;;) {
        std::vector<double> lo(m, inf), hi(m, -inf);
        for (std::size_t i = 0; i < n; ++i) {
            if (!kept[i]) continue;
            for (std::size_t j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], F[i][j]);
                hi[j] = std::max(hi[j], F[i][j]);
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const double span = hi[j] - lo[j];
                G[i][j] = (F[i][j] - lo[j]) / (span > 0.0 ? span : 1.0);
            }
        std::vector<char> drop(n, 0);
        std::size_t n_kept = 0, n_drop = 0;
        for (std::size_t p = 0; p < n; ++p) {
            if (!kept[p]) continue;
            ++n_kept;
            for (std::size_t q = 0; q < n && !drop[p]; ++q) {
                if (!kept[q] || q == p) continue;
                double loss = -inf;
                bool small_gain = true;              // p better than q by <= DRS_GAIN everywhere
                for (std::size_t j = 0; j < m; ++j) {
                    const double d = G[p][j] - G[q][j];
                    if (-d > DRS_GAIN) { small_gain = false; break; }
                    loss = std::max(loss, d);
                }
                if (small_gain && loss > DRS_LOSS) drop[p] = 1;
            }
            if (drop[p]) ++n_drop;
        }
        if (n_drop == 0 || n_drop == n_kept) break;
        for (std::size_t i = 0; i < n; ++i)
            if (drop[i]) kept[i] = 0;
    }

    std::vector<char> taken(n, 0);
    std::vector<double> cover(n, inf);
    out.reserve(k);
    auto take = [&](std::size_t i) {
        taken[i] = 1;
        out.push_back(i);
        for (std::size_t c = 0; c < n; ++c) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j) {
                const double d = G[i][j] - G[c][j];
                if (d > 0.0) s += d * d;
            }
            cover[c] = std::min(cover[c], std::sqrt(s));
        }
    };
    for (std::size_t j = 0; j < m && out.size() < k; ++j) {       // every objective's best kept
        std::size_t best = n;
        for (std::size_t i = 0; i < n; ++i)
            if (kept[i] && (best == n || G[i][j] < G[best][j])) best = i;
        if (!taken[best]) take(best);
    }
    while (out.size() < k) {
        bool kept_left = false;                                   // then the points set aside
        for (std::size_t i = 0; i < n; ++i)
            if (kept[i] && !taken[i]) { kept_left = true; break; }
        std::size_t arg = n;
        double worst = -1.0;
        for (std::size_t i = 0; i < n; ++i)
            if (!taken[i] && (kept[i] || !kept_left) && cover[i] > worst) {
                worst = cover[i];
                arg = i;
            }
        if (arg == n) break;
        take(arg);
    }
    return out;
}

namespace detail {

// A k-d tree over some of the points (rows of the n x m row-major array G):
// every node holds a range of idx and the box around its points; a node of
// more than LEAF points is split at the median of one coordinate, the next one
// at every level. visit(open, each) walks the nodes that open(node) accepts
// and hands every point of an accepted leaf to each(). For the greedy steps a
// node also holds its best point: the largest value (the cover) of its points,
// on a tie the lowest index; rank() is visit() that sets them again.
struct KdTree {
    static constexpr std::size_t LEAF = 16;
    struct Node { std::size_t l, r, left, right; };   // left == 0: a leaf
    std::size_t m = 0;
    std::vector<std::size_t> idx;
    std::vector<Node> nodes;
    std::vector<double> lo, hi;                       // the boxes, m per node
    std::vector<double> best;                         // the best point's value
    std::vector<std::size_t> best_at;                 // and index

    void reset(const std::vector<double>& G, std::size_t m_, std::vector<std::size_t> points)
    {
        m = m_;
        idx = std::move(points);
        nodes.clear();
        lo.clear();
        hi.clear();
        if (!idx.empty()) build(G, 0, idx.size(), 0);
        best.assign(nodes.size(), std::numeric_limits<double>::infinity());
        best_at.assign(nodes.size(), 0);
    }

    std::size_t build(const std::vector<double>& G, std::size_t l, std::size_t r, std::size_t dim)
    {
        const std::size_t id = nodes.size();
        nodes.push_back({l, r, 0, 0});
        lo.resize(lo.size() + m, std::numeric_limits<double>::infinity());
        hi.resize(hi.size() + m, -std::numeric_limits<double>::infinity());
        if (r - l <= LEAF) {
            for (std::size_t t = l; t < r; ++t)
                for (std::size_t j = 0; j < m; ++j) {
                    lo[id * m + j] = std::min(lo[id * m + j], G[idx[t] * m + j]);
                    hi[id * m + j] = std::max(hi[id * m + j], G[idx[t] * m + j]);
                }
            return id;
        }
        const std::size_t mid = l + (r - l) / 2;
        std::nth_element(idx.begin() + static_cast<std::ptrdiff_t>(l),
                         idx.begin() + static_cast<std::ptrdiff_t>(mid),
                         idx.begin() + static_cast<std::ptrdiff_t>(r),
                         [&](std::size_t a, std::size_t b) { return G[a * m + dim] < G[b * m + dim]; });
        const std::size_t left = build(G, l, mid, (dim + 1) % m);
        const std::size_t right = build(G, mid, r, (dim + 1) % m);
        nodes[id].left = left;
        nodes[id].right = right;
        for (std::size_t j = 0; j < m; ++j) {
            lo[id * m + j] = std::min(lo[left * m + j], lo[right * m + j]);
            hi[id * m + j] = std::max(hi[left * m + j], hi[right * m + j]);
        }
        return id;
    }

    template <typename Open, typename Each>
    void visit(const Open& open, const Each& each, std::size_t id = 0) const
    {
        if (nodes.empty() || !open(id)) return;
        const Node& node = nodes[id];
        if (node.left == 0) {
            for (std::size_t t = node.l; t < node.r; ++t) each(idx[t]);
            return;
        }
        visit(open, each, node.left);
        visit(open, each, node.right);
    }

    template <typename Open, typename Each>
    void rank(const Open& open, const Each& each, const std::vector<double>& value, std::size_t id = 0)
    {
        if (nodes.empty() || !open(id)) return;
        const Node& node = nodes[id];
        if (node.left == 0) {
            best[id] = -std::numeric_limits<double>::infinity();
            best_at[id] = idx[node.l];
            for (std::size_t t = node.l; t < node.r; ++t) {
                const std::size_t c = idx[t];
                each(c);
                if (value[c] > best[id] || (value[c] == best[id] && c < best_at[id])) {
                    best[id] = value[c];
                    best_at[id] = c;
                }
            }
            return;
        }
        rank(open, each, value, node.left);
        rank(open, each, value, node.right);
        const std::size_t a = node.left, b = node.right;
        const std::size_t w = best[b] > best[a] || (best[b] == best[a] && best_at[b] < best_at[a]) ? b : a;
        best[id] = best[w];
        best_at[id] = best_at[w];
    }
};

}   // namespace detail

// order_plain's indices, bit for bit, at a fraction of the work. Every number
// is computed by the same expression as there; what changes is only which of
// them are computed, the points being held in a k-d tree:
//   * setting aside: p can be set aside only by a q with q_j <= p_j + DRS_GAIN
//     in every objective and q_j < p_j - DRS_LOSS in some, so p asks only the
//     points of the boxes that reach there;
//   * the greedy steps: a candidate c whose cover is at most B changes nothing
//     when the point taken, i, has c_j < i_j - B in some objective j, since then
//     d+(c, i) > B; so a step updates only the boxes that reach above i - B, B
//     the largest cover in the box, and every box keeps its candidate of
//     largest cover (lowest index on a tie) for the next step, the one the scan
//     of order_plain picks;
//   * the points set aside are covered, when their turn comes, by the minimum
//     over everything taken before them, the minimum order_plain reaches one
//     step at a time.
// The kept points' coordinates lie in [0, 1], where MARGIN (1e-9) is far above
// the rounding of a bound: a skipped pair could not have changed anything. A
// point with a value that is not finite (or a range that overflows) goes to
// order_plain.
inline std::vector<std::size_t> order(const std::vector<std::vector<double>>& F, std::size_t k)
{
    const std::size_t n = F.size();
    k = std::min(k, n);
    if (k == 0) return {};
    const std::size_t m = F[0].size();
    if (m == 0) return order_plain(F, k);
    for (const auto& f : F)
        for (double v : f)
            if (!std::isfinite(v)) return order_plain(F, k);
    const double inf = std::numeric_limits<double>::infinity();
    constexpr double MARGIN = 1e-9;

    // Setting aside, as in order_plain, with the tree of the kept points.
    std::vector<char> kept(n, 1);
    std::vector<double> G(n * m);
    std::vector<std::size_t> kept_list;
    detail::KdTree tree;
    std::vector<double> bound(m), beaten(m);
    for (;;) {
        std::vector<double> lo(m, inf), hi(m, -inf);
        for (std::size_t i = 0; i < n; ++i) {
            if (!kept[i]) continue;
            for (std::size_t j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], F[i][j]);
                hi[j] = std::max(hi[j], F[i][j]);
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const double span = hi[j] - lo[j];
                G[i * m + j] = (F[i][j] - lo[j]) / (span > 0.0 ? span : 1.0);
                if (!std::isfinite(G[i * m + j])) return order_plain(F, k);
            }
        kept_list.clear();
        for (std::size_t i = 0; i < n; ++i)
            if (kept[i]) kept_list.push_back(i);
        tree.reset(G, m, kept_list);
        std::vector<char> drop(n, 0);
        std::size_t n_drop = 0;
        for (std::size_t p : kept_list) {
            for (std::size_t j = 0; j < m; ++j) {
                bound[j] = G[p * m + j] + DRS_GAIN + MARGIN;       // q_j above: p gains too much
                beaten[j] = G[p * m + j] - DRS_LOSS + MARGIN;      // q_j below: p loses enough
            }
            tree.visit(
                [&](std::size_t node) {
                    if (drop[p]) return false;
                    bool loses = false;
                    for (std::size_t j = 0; j < m; ++j) {
                        if (tree.lo[node * m + j] > bound[j]) return false;
                        loses = loses || tree.lo[node * m + j] < beaten[j];
                    }
                    return loses;
                },
                [&](std::size_t q) {
                    if (drop[p] || q == p) return;
                    double loss = -inf;
                    bool small_gain = true;
                    for (std::size_t j = 0; j < m; ++j) {
                        const double d = G[p * m + j] - G[q * m + j];
                        if (-d > DRS_GAIN) { small_gain = false; break; }
                        loss = std::max(loss, d);
                    }
                    if (small_gain && loss > DRS_LOSS) drop[p] = 1;
                });
            if (drop[p]) ++n_drop;
        }
        if (n_drop == 0 || n_drop == kept_list.size()) break;
        for (std::size_t i = 0; i < n; ++i)
            if (drop[i]) kept[i] = 0;
    }

    // d+(c, s), as order_plain computes it
    auto d_plus = [&](std::size_t s, std::size_t c) {
        double sum = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            const double d = G[s * m + j] - G[c * m + j];
            if (d > 0.0) sum += d * d;
        }
        return std::sqrt(sum);
    };
    std::vector<char> taken(n, 0);
    std::vector<double> cover(n, inf);                    // -inf once taken
    std::vector<std::size_t> out;
    out.reserve(k);
    std::size_t kept_left = kept_list.size();
    auto take = [&](std::size_t i) {                      // a kept point
        taken[i] = 1;
        cover[i] = -inf;
        out.push_back(i);
        --kept_left;
        tree.rank(
            [&](std::size_t node) {
                const double B = tree.best[node];
                if (B == -inf) return false;
                for (std::size_t j = 0; j < m; ++j)
                    if (tree.hi[node * m + j] < G[i * m + j] - B - MARGIN) return false;
                return true;
            },
            [&](std::size_t c) {
                if (taken[c]) return;
                cover[c] = std::min(cover[c], d_plus(i, c));
            },
            cover);
    };
    for (std::size_t j = 0; j < m && out.size() < k; ++j) {       // every objective's best kept
        std::size_t best = n;
        for (std::size_t i = 0; i < n; ++i)
            if (kept[i] && (best == n || G[i * m + j] < G[best * m + j])) best = i;
        if (!taken[best]) take(best);
    }
    while (out.size() < k && kept_left > 0) take(tree.best_at[0]); // the kept candidates
    if (out.size() < k) {                                         // then the points set aside
        std::vector<std::size_t> aside;
        for (std::size_t i = 0; i < n; ++i)
            if (!taken[i]) aside.push_back(i);
        for (std::size_t c : aside)
            for (std::size_t s : out) cover[c] = std::min(cover[c], d_plus(s, c));
        while (out.size() < k) {
            std::size_t arg = n;
            double worst = -1.0;
            for (std::size_t c : aside)
                if (!taken[c] && cover[c] > worst) {
                    worst = cover[c];
                    arg = c;
                }
            if (arg == n) break;
            taken[arg] = 1;
            out.push_back(arg);
            for (std::size_t c : aside)
                if (!taken[c]) cover[c] = std::min(cover[c], d_plus(arg, c));
        }
    }
    return out;
}

}   // namespace mootation::dss

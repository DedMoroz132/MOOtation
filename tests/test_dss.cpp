// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// dss.hpp: dss::order gives the order it gave before task 12 of 2026-10-19
// (§4: the bound on three sums per node of the k-d tree in the setting aside,
// no round after one that leaves the kept points' range as it was), bit for
// bit, and order_plain's. 10 000 random sets at 2 to 10 objectives, of up to
// 25 000 points: fronts (simplex, sphere, convex, with a little depth),
// clouds of dominated points, flat tails and far points (set aside), copies,
// values on a grid (ties), a constant objective; k from 1 to all.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/dss.hpp>

// dss::order as it was before task 12 (main 4375bf2), the reference
namespace before {
using mootation::dss::DRS_GAIN;
using mootation::dss::DRS_LOSS;
using mootation::dss::order_plain;
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

}   // namespace before

namespace {

using Set = std::vector<std::vector<double>>;
int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        if (failed <= 20) std::printf("  FAIL  %s\n", what.c_str());
    }
}

// n points of m objectives of one of the kinds above
Set make(std::mt19937_64& rng, std::size_t n, std::size_t m, int kind)
{
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    Set F(n, std::vector<double>(m));
    for (auto& f : F) {
        double s = 0.0;
        switch (kind) {
        case 0:                                         // simplex
            for (auto& v : f) { v = -std::log(1.0 - u(rng)); s += v; }
            for (auto& v : f) v /= s;
            break;
        case 1:                                         // sphere, a little depth
        case 2:                                         // convex
            for (auto& v : f) { v = std::fabs(nd(rng)); s += v * v; }
            s = std::sqrt(s) / (kind == 1 ? 1.0 + 0.03 * u(rng) : 1.0);
            for (auto& v : f) v = kind == 1 ? v / s : 1.0 - v / s;
            break;
        case 3:                                         // cloud
            for (auto& v : f) v = u(rng);
            break;
        default:                                        // a front with flat tails and far points
            for (auto& v : f) { v = -std::log(1.0 - u(rng)); s += v; }
            for (auto& v : f) v /= s;
            if (u(rng) < 0.05) f[rng() % m] = 1e-4 * u(rng);
            if (u(rng) < 0.01) f[rng() % m] = 3.0 + u(rng);
        }
        if (kind % 2 == 0 && u(rng) < 0.1)              // values on a grid: ties
            for (auto& v : f) v = std::round(v * 40.0) / 40.0;
    }
    if (n > 4) {
        F[n - 1] = F[0];                                // a copy
        if (kind == 3)
            for (auto& f : F) f[m - 1] = 0.5;           // a constant objective
    }
    return F;
}

}   // namespace

int main()
{
    std::printf("dss::order: as before task 12, and as order_plain\n");
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::size_t sets = 0, points = 0;
    for (int trial = 0; trial < 10000; ++trial) {
        const std::size_t m = 2 + rng() % 9;
        // sizes log-uniform from 2 to 2 000, and every 500th set up to 25 000
        const double top = trial % 500 == 0 ? 25000.0 : 2000.0;
        const auto n = static_cast<std::size_t>(std::exp(std::log(2.0) + u(rng) * (std::log(top) - std::log(2.0))));
        const int kind = static_cast<int>(rng() % 5);
        const Set F = make(rng, n, m, kind);
        const std::size_t k = std::min<std::size_t>(n, trial % 3 == 0 ? n : 1 + rng() % 200);
        const auto got = mootation::dss::order(F, k);
        const auto ref = before::order(F, k);
        check(got == ref, "set " + std::to_string(trial) + ": " + std::to_string(n) + " points, " +
                              std::to_string(m) + " objectives, kind " + std::to_string(kind) +
                              ", k " + std::to_string(k) + ": the order before task 12");
        if (n <= 300)
            check(got == mootation::dss::order_plain(F, k), "set " + std::to_string(trial) + ": order_plain's");
        ++sets;
        points += n;
    }
    std::printf("  %zu sets, %zu points\n", sets, points);
    std::printf("%d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// An ND-tree (Jaszkiewicz & Lust, "ND-Tree-Based Update: A Fast Algorithm for
// the Dynamic Nondominance Problem", IEEE TEVC 22(5):778-791, 2018): an index
// over a set of mutually nondominated points that answers, for a new point f,
// whether some point of the set weakly dominates it (p <= f in every
// objective, a copy included) and, if none does, which points f weakly
// dominates. The run archive (grid_archive.hpp) and the nondominated filter
// (nondominated.hpp) ask it from four objectives on (task 11 of 2026-10-18,
// §2) where they compared f with every point; the answers are the same, so the
// archive and the filter are the same, bit for bit. Only the work changes.
//
// THE TREE (the paper's Definition 8). The points sit in leaves of at most
// leaf_max of them; an inner node has up to `children` children; every node
// keeps a box, the componentwise minimum (lo, the paper's approximate ideal)
// and maximum (hi, its approximate nadir) of the points ever inserted under
// it. A removal does not shrink the boxes (the paper's choice: they stay valid
// bounds), and a node left empty goes, an inner node left with one child is
// replaced by it (Algorithm 4).
//
// A QUERY (Algorithm 4, UpdateNode, without its removals: the caller removes
// the points it is told of, erase() below) at a node with box [lo, hi]:
//   hi <= f   every point under it weakly dominates f (Property 1): covered;
//   f <= lo   f weakly dominates every point under it (Property 2);
//   neither lo <= f nor f <= hi: some objective has f below every point and
//             some objective above (Property 3): nothing under it can weakly
//             dominate f or be weakly dominated by it, the node is skipped;
//   else      its children, or a leaf's points one by one.
// The set's points being mutually nondominated, f cannot both be weakly
// dominated by one and weakly dominate another (they would be one point).
//
// AN INSERTION (Algorithm 5): down to a leaf, at every inner node into the
// child whose box centre is nearest (Euclidean); the boxes on the way widened
// (Algorithm 7); a leaf of more than leaf_max points split into `children`
// leaves (Algorithm 6): the first seed the point farthest on average from the
// others, every next seed the point farthest on average from the seeds so
// far, every other point to the leaf whose box centre is nearest. The paper's
// parameters: leaf_max 20, children M + 1.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace mootation::nd {

class NDTree {
public:
    // m objectives; ids are small non-negative integers (an archive's slots)
    explicit NDTree(std::size_t m, std::size_t leaf_max = 20, std::size_t children = 0)
        : m_(m), leaf_max_(leaf_max), children_(children != 0 ? children : m + 1)
    {
        clear();
    }

    std::size_t size() const { return size_; }

    void clear()
    {
        nodes_.assign(1, Node{});
        nodes_[0].lo.assign(m_, std::numeric_limits<double>::infinity());
        nodes_[0].hi.assign(m_, -std::numeric_limits<double>::infinity());
        free_.clear();
        root_ = 0;
        where_.clear();
        coords_.clear();
        size_ = 0;
    }

    // Is f weakly dominated by a point of the set? If not, and beaten is
    // given, it gets the ids of the points f weakly dominates, ascending.
    bool covered(const double* f, std::vector<std::size_t>* beaten) const
    {
        if (beaten) beaten->clear();
        if (size_ == 0) return false;
        const bool hit = visit(root_, f, beaten);
        if (!hit && beaten) sort_ids(*beaten);
        return hit;
    }

    void insert(std::size_t id, const double* f)
    {
        if (id >= where_.size()) {
            where_.resize(id + 1, -1);
            coords_.resize((id + 1) * m_);
        }
        std::copy(f, f + m_, coords_.begin() + static_cast<std::ptrdiff_t>(id * m_));
        int v = root_;
        widen(v, f);
        while (!nodes_[static_cast<std::size_t>(v)].leaf) {
            v = nearest_child(v, f);
            widen(v, f);
        }
        nodes_[static_cast<std::size_t>(v)].ids.push_back(id);
        where_[id] = v;
        ++size_;
        if (nodes_[static_cast<std::size_t>(v)].ids.size() > leaf_max_) split(v);
    }

    void erase(std::size_t id)
    {
        if (id >= where_.size() || where_[id] < 0) return;
        const int v = where_[id];
        where_[id] = -1;
        --size_;
        auto& ids = nodes_[static_cast<std::size_t>(v)].ids;
        for (std::size_t i = 0; i < ids.size(); ++i)
            if (ids[i] == id) {
                ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        if (ids.empty()) drop(v);
    }

private:
    struct Node {
        std::vector<double> lo, hi;
        int parent = -1;
        bool leaf = true;
        std::vector<int> kids;
        std::vector<std::size_t> ids;
    };

    std::size_t m_, leaf_max_, children_;
    std::vector<Node> nodes_;
    std::vector<int> free_;
    int root_ = 0;
    std::vector<int> where_;              // id -> its leaf, -1 absent
    std::vector<double> coords_;          // id -> its point, at id * m_
    std::size_t size_ = 0;

    const double* point(std::size_t id) const { return coords_.data() + id * m_; }

    // a <= b in every objective
    bool leq(const double* a, const double* b) const
    {
        for (std::size_t j = 0; j < m_; ++j)
            if (!(a[j] <= b[j])) return false;
        return true;
    }

    static void sort_ids(std::vector<std::size_t>& v)
    {
        for (std::size_t i = 1; i < v.size(); ++i)          // short lists: insertion sort
            for (std::size_t k = i; k > 0 && v[k - 1] > v[k]; --k) std::swap(v[k - 1], v[k]);
    }

    void collect(int v, std::vector<std::size_t>& out) const
    {
        const Node& n = nodes_[static_cast<std::size_t>(v)];
        if (n.leaf) out.insert(out.end(), n.ids.begin(), n.ids.end());
        else
            for (int c : n.kids) collect(c, out);
    }

    bool visit(int v, const double* f, std::vector<std::size_t>* beaten) const
    {
        const Node& n = nodes_[static_cast<std::size_t>(v)];
        if (leq(n.hi.data(), f)) return true;                        // Property 1
        if (leq(f, n.lo.data())) {                                   // Property 2
            if (beaten) collect(v, *beaten);
            return false;
        }
        if (!leq(n.lo.data(), f) && !leq(f, n.hi.data())) return false;   // Property 3
        if (n.leaf) {
            for (std::size_t id : n.ids) {
                const double* p = point(id);
                if (leq(p, f)) return true;
                if (beaten && leq(f, p)) beaten->push_back(id);
            }
            return false;
        }
        for (int c : n.kids)
            if (visit(c, f, beaten)) return true;
        return false;
    }

    void widen(int v, const double* f)
    {
        Node& n = nodes_[static_cast<std::size_t>(v)];
        for (std::size_t j = 0; j < m_; ++j) {
            if (f[j] < n.lo[j]) n.lo[j] = f[j];
            if (f[j] > n.hi[j]) n.hi[j] = f[j];
        }
    }

    double centre_distance(int v, const double* f) const
    {
        const Node& n = nodes_[static_cast<std::size_t>(v)];
        double d = 0.0;
        for (std::size_t j = 0; j < m_; ++j) {
            const double e = f[j] - 0.5 * (n.lo[j] + n.hi[j]);
            d += e * e;
        }
        return d;
    }

    int nearest_child(int v, const double* f) const
    {
        const Node& n = nodes_[static_cast<std::size_t>(v)];
        int best = n.kids.front();
        double bd = centre_distance(best, f);
        for (std::size_t i = 1; i < n.kids.size(); ++i) {
            const double d = centre_distance(n.kids[i], f);
            if (d < bd) { bd = d; best = n.kids[i]; }
        }
        return best;
    }

    int make_leaf(int parent)
    {
        int v;
        if (!free_.empty()) {
            v = free_.back();
            free_.pop_back();
            nodes_[static_cast<std::size_t>(v)] = Node{};
        } else {
            v = static_cast<int>(nodes_.size());
            nodes_.emplace_back();
        }
        Node& n = nodes_[static_cast<std::size_t>(v)];
        n.parent = parent;
        n.lo.assign(m_, std::numeric_limits<double>::infinity());
        n.hi.assign(m_, -std::numeric_limits<double>::infinity());
        return v;
    }

    double distance(const double* a, const double* b) const
    {
        double d = 0.0;
        for (std::size_t j = 0; j < m_; ++j) {
            const double e = a[j] - b[j];
            d += e * e;
        }
        return std::sqrt(d);
    }

    // Algorithm 6: the leaf v becomes an inner node of `children_` leaves
    void split(int v)
    {
        std::vector<std::size_t> ids;
        ids.swap(nodes_[static_cast<std::size_t>(v)].ids);
        const std::size_t k = std::min(children_, ids.size());
        std::vector<char> seed(ids.size(), 0);
        std::vector<std::size_t> seeds;
        // the first seed: the largest average distance to all the others
        std::size_t first = 0;
        double best = -1.0;
        for (std::size_t a = 0; a < ids.size(); ++a) {
            double s = 0.0;
            for (std::size_t b = 0; b < ids.size(); ++b)
                if (a != b) s += distance(point(ids[a]), point(ids[b]));
            if (s > best) { best = s; first = a; }
        }
        seed[first] = 1;
        seeds.push_back(first);
        while (seeds.size() < k) {                 // then the farthest on average from the seeds
            std::size_t pick = 0;
            best = -1.0;
            for (std::size_t a = 0; a < ids.size(); ++a) {
                if (seed[a]) continue;
                double s = 0.0;
                for (std::size_t b : seeds) s += distance(point(ids[a]), point(ids[b]));
                if (s > best) { best = s; pick = a; }
            }
            seed[pick] = 1;
            seeds.push_back(pick);
        }
        Node& n = nodes_[static_cast<std::size_t>(v)];
        n.leaf = false;
        n.kids.clear();
        for (std::size_t s : seeds) {
            const int c = make_leaf(v);
            nodes_[static_cast<std::size_t>(v)].kids.push_back(c);
            nodes_[static_cast<std::size_t>(c)].ids.push_back(ids[s]);
            widen(c, point(ids[s]));
            where_[ids[s]] = c;
        }
        for (std::size_t a = 0; a < ids.size(); ++a) {
            if (seed[a]) continue;
            const int c = nearest_child(v, point(ids[a]));
            nodes_[static_cast<std::size_t>(c)].ids.push_back(ids[a]);
            widen(c, point(ids[a]));
            where_[ids[a]] = c;
        }
    }

    // v has no points left: it goes, and an inner node left with one child
    // is replaced by that child
    void drop(int v)
    {
        Node& n = nodes_[static_cast<std::size_t>(v)];
        if (n.parent < 0) {                        // the root: an empty leaf again
            n.leaf = true;
            n.kids.clear();
            n.lo.assign(m_, std::numeric_limits<double>::infinity());
            n.hi.assign(m_, -std::numeric_limits<double>::infinity());
            return;
        }
        const int p = n.parent;
        auto& kids = nodes_[static_cast<std::size_t>(p)].kids;
        for (std::size_t i = 0; i < kids.size(); ++i)
            if (kids[i] == v) {
                kids.erase(kids.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        free_.push_back(v);
        if (kids.empty()) {
            drop(p);
            return;
        }
        if (kids.size() == 1) {                    // p is replaced by its one child
            const int c = kids.front();
            Node& pn = nodes_[static_cast<std::size_t>(p)];
            const int g = pn.parent;
            nodes_[static_cast<std::size_t>(c)].parent = g;
            if (g < 0) {
                root_ = c;
            } else {
                for (int& k : nodes_[static_cast<std::size_t>(g)].kids)
                    if (k == p) k = c;
            }
            pn.kids.clear();
            free_.push_back(p);
        }
    }
};

}   // namespace mootation::nd

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Hypervolume contributions at three and four objectives: HVC3D and HVC4D of
// A.P. Guerreiro & C.M. Fonseca, "Computing and Updating Hypervolume
// Contributions in Up to Four Dimensions", IEEE TEVC 22(3):449-463, 2018,
// doi:10.1109/TEVC.2017.2729550 (source hvc-update_guerreiro2018_accepted;
// task 12 of 2026-10-19, §5) — the hv_contrib = "hvc" of SMS-EMOA and
// MO-CMA-ES. hv_contribution.hpp's contributions_exact (WFG) stays their
// default.
//
// THE DATA STRUCTURE (§III-A, HV3D+). The points in a list Q in increasing
// z (lexicographic z, y, x), each with its outer delimiters cx and cy: cx is
// the point before it in Q with the least x above its x among those below it
// in y, cy the same with x and y exchanged; the sentinels (r_x, -inf, -inf)
// and (-inf, r_y, -inf) stand in where there is none. Built by Algorithm 2
// (a balanced tree in y over the points swept so far, O(n log n)). nd counts
// the points that dominate a point and dom is one of them (§IV-A1); a point
// dominated by two or more changes no contribution and is dropped.
//
// ALL CONTRIBUTIONS (§IV-A2, Algorithm 4, O(n)). Q swept in increasing z;
// L holds the points swept so far whose (x, y) projections are mutually
// non-dominated, in increasing y (Algorithm 1's procedures run over a
// stretch of it from an outer delimiter); every point p of L keeps its
// contribution's base area p.a, its volume p.v up to p.z, and p.L, the points
// that bound its exclusive region on the plane z = p.z. A non-dominated p
// (lines 6-21) closes the volumes of its outer and inner delimiters at p_z
// (updateVolumes), takes the area its projection adds as its base
// (computeArea), moves its inner delimiters from L to p.L with copies of cx
// and cy at the ends, enters L, and cuts the regions of cy (lines 12-16) and
// cx (lines 17-21) by the area it shares with each. A p dominated by one
// point q (lines 22-28) cuts q's region alone. The sentinel
// (-REALMAX, -REALMAX, r_z) at the end closes every volume still open.
//
// UPDATES (§III-A4, §IV-A3). Adding a point u sets u.cx, u.cy, u.nd, u.dom
// from the points before it in Q, gives u as cx or cy to the later points it
// becomes the nearest delimiter of, counts it among the dominators of the
// later points it dominates, and puts u into Q: one sweep of Q. Removing u
// from a non-dominated set gives new delimiters to the points that had u:
// Algorithm 2's sweep on a list, each point's own cx as the tree search's
// answer (cy where cx was u). After an update Algorithm 4 gives all
// contributions in O(n) — the paper's update by recomputing the
// contributions on the updated structure (its "HVC3D-R"), which takes the
// reference point anew at every call (SMS-EMOA's and MO-CMA-ES's move with
// the population). The variant that keeps the contributions and adds or
// subtracts the joint contributions of the point that came or went
// ("HVC3D-U", §IV-A3) is not implemented.
//
// FOUR OBJECTIVES (§IV-B, HVC4D, O(n^2)). The points in increasing w; each
// one's projection on (x, y, z) is added to the three-dimensional structure
// (it may dominate projections already there: nd and dom) and the structure's
// contributions, times the height to the next point's w (to r_w after the
// last), are added to the points' four-dimensional ones.
//
// HvcKept keeps the three-dimensional structure from one call to the next:
// a selection that adds one offspring and removes one point a step (SMS-EMOA,
// the (mu+1)-MO-CMA-ES) brings it to the next set by single-point updates and
// recomputes the contributions in linear time. The set is matched to the
// structure by coordinates; more than a few changes, or a set with dominated
// points, rebuild it by Algorithm 2.
//
// Declared readings:
//   HVC-1. Ties. The paper orders Q lexicographically (z, y, x) and breaks
//     the delimiters' ties by the other coordinate. Here every comparison of
//     coordinate k is lexicographic, k first and the others after it (x: x y
//     z w; y: y x z w; z: z y x w; w: w z y x): no two distinct points compare
//     equal anywhere, a point weakly dominated by another (equal in some
//     objectives) is dominated, and the algorithm runs as written on points
//     in general position. The areas and volumes are computed from the
//     values themselves, where the order only decides which boxes are added;
//     a contribution is continuous in the coordinates, so the value is the
//     one at the ties. The structure is a function of the set alone: updated
//     or rebuilt, it gives the same contributions bit for bit.
//   HVC-2. Copies. Equal points are held once with their number: each of two
//     or more copies contributes nothing (as in contributions_exact), and
//     the point bounds the others' regions as one.
//   HVC-3. A point not strictly better than r in every objective contributes
//     nothing and bounds nothing (its region has no volume), as in
//     contributions_exact; with a non-finite coordinate the set goes to
//     contributions_exact.
//   HVC-4. The values are contributions_exact's (WFG) up to the order of the
//     sums, not bit for bit (tests/test_hvc.cpp: within 1e-12 of the set's
//     hypervolume), so a selection between two nearly equal contributions
//     may take the other point and the run another path.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "hv_contribution.hpp"

namespace mootation::hv {

namespace detail {

// The HVC3D structure over distinct points (HVC-2: copies are the caller's).
class Hvc3d {
public:
    static constexpr int SX = 0, SY = 1;     // the sentinels (r_x, -inf, .), (-inf, r_y, .)

    Hvc3d() { clear(); }

    void clear()
    {
        const double inf = std::numeric_limits<double>::infinity();
        N_.assign(2, Node{});
        N_[SX].c[0] = inf;  N_[SX].c[1] = -inf; N_[SX].c[2] = -inf;
        N_[SY].c[0] = -inf; N_[SY].c[1] = inf;  N_[SY].c[2] = -inf;
        free_.clear();
        qhead_ = qtail_ = -1;
        dominated_ = false;
    }

    // A node for the point (x, y, z) with w its fourth coordinate's tie-break
    // (HVC4D's projections; 0 otherwise), not yet in Q.
    int make(const double* p, double w)
    {
        int i;
        if (!free_.empty()) { i = free_.back(); free_.pop_back(); }
        else { i = static_cast<int>(N_.size()); N_.emplace_back(); }
        N_[i] = Node{};
        N_[i].c[0] = p[0]; N_[i].c[1] = p[1]; N_[i].c[2] = p[2]; N_[i].c[3] = w;
        N_[i].used = true;
        return i;
    }

    // Algorithm 2 (preprocessing) over the nodes given, which must not be in
    // Q (Q is cleared first). Points found dominated are added after it by
    // insert (§III-A2: "dominated points ... can be easily detected").
    void build(std::vector<int> nodes)
    {
        qhead_ = qtail_ = -1;
        dominated_ = false;
        std::sort(nodes.begin(), nodes.end(), [this](int a, int b) { return lz(a, b); });
        std::set<int, ByY> T(ByY{this});                     // line 2
        T.insert(SX);
        T.insert(SY);
        std::vector<int> later;
        for (int p : nodes) {                                // line 3
            auto it = T.lower_bound(p);
            const int s = *std::prev(it);                    // line 4: outerDelimiter_x
            if (lx(s, p)) { later.push_back(p); continue; }  // s dominates p (x, y and z)
            while (!lx(*it, p)) it = T.erase(it);            // line 5: removeDominated_y
            Node& P = N_[p];
            P.cx = s;                                        // line 6
            P.cy = *it;                                      // line 7
            P.nd = 0;
            P.dom = -1;
            link_last(p);                                    // Q in the order swept
            T.insert(p);                                     // line 8: add_y
        }
        for (int p : later) insert(p);
    }

    // §III-A4, adding u (made, not in Q); §IV-A2: u's dominators and the
    // points u dominates. False if u is dominated by two points or more
    // (then it is not kept: it changes no contribution).
    bool insert(int u)
    {
        int cx = SX, cy = SY, nd = 0, dom = -1;
        int q = qhead_;
        for (; q != -1 && lz(q, u); q = N_[q].next) {
            if (lx(u, q) && ly(q, u) && lx(q, cx)) cx = q;  // q_x > u_x, q_y < u_y, least q_x
            if (ly(u, q) && lx(q, u) && ly(q, cy)) cy = q;  // the same in y
            if (lx(q, u) && ly(q, u)) { ++nd; dom = q; }    // q dominates u
        }
        if (nd >= 2) { drop(u); dominated_ = true; return false; }
        Node& U = N_[u];
        U.cx = cx; U.cy = cy; U.nd = nd; U.dom = dom;
        if (nd > 0) dominated_ = true;
        link_before(u, q);
        while (q != -1) {
            const int after = N_[q].next;
            Node& Q = N_[q];
            if (ly(u, q) && lx(q, u) && lx(u, Q.cx)) Q.cx = u;   // q_x < u_x < (q.cx)_x
            if (lx(u, q) && ly(q, u) && ly(u, Q.cy)) Q.cy = u;   // q_y < u_y < (q.cy)_y
            if (lx(u, q) && ly(u, q)) {                           // u dominates q
                dominated_ = true;
                if (++Q.nd >= 2) { unlink(q); Q.inq = false; }
                else Q.dom = u;
            }
            q = after;
        }
        return true;
    }

    // §III-A4, removing u from a set without dominated points: the points
    // whose cx or cy was u get the next one, by Algorithm 2's sweep on a list.
    void erase(int u)
    {
        if (dominated_) throw std::logic_error("hvc: erase from a set with dominated points");
        unlink(u);
        drop(u);
        C_.clear();
        const int lh = cell(SX, -1), lt = cell(SY, -1);
        join(lh, lt);
        N_[SX].lcell = lh;
        N_[SY].lcell = lt;
        for (int p = qhead_; p != -1; p = N_[p].next) N_[p].lcell = -1;
        for (int p = qhead_; p != -1; p = N_[p].next) {
            Node& P = N_[p];
            if (P.cx != u) {
                const int s = in_l(P.cx);
                int e = C_[s].next;
                while (!lx(C_[e].pt, p)) { const int t = e; e = C_[e].next; cut(t); N_[C_[t].pt].lcell = -1; }
                P.cy = C_[e].pt;
                P.lcell = cell(p, -1);
                put_after(P.lcell, s);
            } else {                                         // the roles of x and y exchanged
                const int s = in_l(P.cy);
                int e = C_[s].prev;
                while (!ly(C_[e].pt, p)) { const int t = e; e = C_[e].prev; cut(t); N_[C_[t].pt].lcell = -1; }
                P.cx = C_[e].pt;
                P.lcell = cell(p, -1);
                put_after(P.lcell, e);
            }
        }
    }

    // Algorithm 4: every node's contribution (contribution()) against r.
    // Every point in Q must be strictly below r.
    void compute(const double* r)
    {
        const double inf = std::numeric_limits<double>::infinity();
        N_[SX].c[0] = r[0];
        N_[SY].c[1] = r[1];
        N_[SX].a = N_[SY].a = 0.0;
        C_.clear();
        const int lh = cell(SX, -1), lt = cell(SY, -1);     // line 1
        join(lh, lt);
        N_[SX].lcell = lh;
        N_[SY].lcell = lt;
        for (int p = qhead_; p != -1; p = N_[p].next) {     // line 2
            Node& P = N_[p];
            P.v = 0.0;                                       // line 3
            P.z = P.c[2];                                    // line 4
            P.a = 0.0;
            P.lh = P.lt = P.lcell = P.icell = -1;
        }
        for (int p = qhead_; p != -1; p = N_[p].next) {
            Node& P = N_[p];
            const double pz = P.c[2];
            if (P.nd == 0) {                                 // line 5
                const int s = in_l(P.cx);
                update_volumes(p, s, pz);                    // line 6
                P.a = area_y(p, p, s);                       // line 7
                // line 8: the inner delimiters move from L to p.L; lines 9-10:
                // copies of cx and cy at its ends; line 11: p into L
                int e = C_[s].next;
                const int first = e;
                int last = -1;
                while (!lx(C_[e].pt, p)) { last = e; e = C_[e].next; }
                if (C_[e].pt != P.cy) throw std::logic_error("hvc: p.cy is not where L says");
                P.lh = cell(P.cx, p);
                P.lt = cell(P.cy, p);
                if (last == -1) {
                    join(P.lh, P.lt);
                } else {
                    join(s, e);                              // the stretch leaves L
                    join(P.lh, first);
                    join(last, P.lt);
                    for (int t = first;; t = C_[t].next) {
                        const int q = C_[t].pt;
                        C_[t].owner = p;
                        N_[q].lcell = -1;
                        N_[q].icell = t;
                        if (t == last) break;
                    }
                }
                P.lcell = cell(p, -1);
                put_after(P.lcell, s);
                if (P.cy != SY) {                            // lines 12-16: q = p.cy
                    const int q = P.cy;
                    Node& Q = N_[q];
                    const int h = Q.lh;                      // head_y(q.L)
                    Q.a -= area_y(p, q, h);                  // p v q: x of p, y of q
                    drop_dominated_y(p, h);
                    const int nh = cell(p, q);               // p replaces the head
                    join(nh, C_[h].next);
                    Q.lh = nh;
                }
                if (P.cx != SX) {                            // lines 17-21: q = p.cx
                    const int q = P.cx;
                    Node& Q = N_[q];
                    const int t = Q.lt;                      // head_x(q.L)
                    Q.a -= area_x(q, p, t);                  // p v q: x of q, y of p
                    drop_dominated_x(p, t);
                    const int nt = cell(p, q);               // p replaces the tail
                    join(C_[t].prev, nt);
                    Q.lt = nt;
                }
            } else {                                         // lines 22-28: p.nd = 1
                const int q = P.dom;
                Node& Q = N_[q];
                Q.v += Q.a * (pz - Q.z);                     // line 24
                Q.z = pz;                                    // line 25
                int s = Q.lh;                                // p.cx in q.L: its head, or inner
                if (C_[s].pt != P.cx) {
                    s = N_[P.cx].icell;
                    if (s < 0 || C_[s].owner != q) throw std::logic_error("hvc: p.cx not in q.L");
                }
                Q.a -= area_y(p, p, s);                      // line 26
                drop_dominated_y(p, s);                      // line 27
                P.icell = cell(p, q);                        // line 28
                put_after(P.icell, s);
            }
        }
        // the sentinel (-REALMAX, -REALMAX, r_z): every volume still open closes
        for (int t = C_[lh].next; t != lt; t = C_[t].next) {
            Node& Q = N_[C_[t].pt];
            Q.v += Q.a * (r[2] - Q.z);
        }
        N_[SX].c[0] = inf;
        N_[SY].c[1] = inf;
    }

    // a node's contribution after compute(): 0 for a dominated point
    double contribution(int i) const { return N_[i].inq && N_[i].nd == 0 ? N_[i].v : 0.0; }
    bool dominated() const { return dominated_; }     // has a point been dominated since build?
    const double* coords(int i) const { return N_[i].c; }
    int nodes() const { return static_cast<int>(N_.size()); }

private:
    struct Node {
        double c[4] = {0.0, 0.0, 0.0, 0.0};   // x, y, z and w (the last tie-break)
        bool used = false, inq = false;
        int prev = -1, next = -1;             // Q
        int cx = SX, cy = SY, nd = 0, dom = -1;
        double a = 0.0, v = 0.0, z = 0.0;     // Algorithm 4: base area, volume, its z
        int lh = -1, lt = -1;                 // p.L: its first and last cell
        int lcell = -1;                       // its cell in L
        int icell = -1;                       // its cell as an inner delimiter of a q.L
    };
    struct Cell { int pt, owner, prev, next; };   // owner: the point whose list it is in (-1: L)
    struct ByY {
        const Hvc3d* h;
        bool operator()(int a, int b) const { return h->ly(a, b); }
    };

    std::vector<Node> N_;
    std::vector<int> free_;
    std::vector<Cell> C_;
    int qhead_ = -1, qtail_ = -1;
    bool dominated_ = false;

    // HVC-1: coordinate k first, the others after it
    bool lex(int a, int b, int k0, int k1, int k2, int k3) const
    {
        const double* A = N_[a].c;
        const double* B = N_[b].c;
        if (A[k0] != B[k0]) return A[k0] < B[k0];
        if (A[k1] != B[k1]) return A[k1] < B[k1];
        if (A[k2] != B[k2]) return A[k2] < B[k2];
        return A[k3] < B[k3];
    }
    bool lx(int a, int b) const { return lex(a, b, 0, 1, 2, 3); }
    bool ly(int a, int b) const { return lex(a, b, 1, 0, 2, 3); }
    bool lz(int a, int b) const { return lex(a, b, 2, 1, 0, 3); }

    void drop(int i) { N_[i].used = false; N_[i].inq = false; free_.push_back(i); }

    // the cell of a point in L (an outer delimiter there must be in it)
    int in_l(int i) const
    {
        if (N_[i].lcell < 0) throw std::logic_error("hvc: an outer delimiter is not in L");
        return N_[i].lcell;
    }

    void link_last(int i) { link_before(i, -1); }
    void link_before(int i, int at)
    {
        Node& I = N_[i];
        I.inq = true;
        I.next = at;
        I.prev = at == -1 ? qtail_ : N_[at].prev;
        if (I.prev == -1) qhead_ = i; else N_[I.prev].next = i;
        if (at == -1) qtail_ = i; else N_[at].prev = i;
    }
    void unlink(int i)
    {
        Node& I = N_[i];
        if (I.prev == -1) qhead_ = I.next; else N_[I.prev].next = I.next;
        if (I.next == -1) qtail_ = I.prev; else N_[I.next].prev = I.prev;
        I.prev = I.next = -1;
    }

    // the lists L and q.L, increasing y (decreasing x), of cells
    int cell(int pt, int owner)
    {
        C_.push_back(Cell{pt, owner, -1, -1});
        return static_cast<int>(C_.size()) - 1;
    }
    void join(int a, int b) { C_[a].next = b; C_[b].prev = a; }
    void put_after(int t, int s) { const int n = C_[s].next; join(s, t); join(t, n); }
    void cut(int t) { join(C_[t].prev, C_[t].next); }

    // computeArea_y (Algorithm 1): the area u* adds to S*, s its outer
    // delimiter in x; u's x is ux's, its y uy's (u = p v q in lines 13, 18).
    double area_y(int ux, int uy, int s) const
    {
        const double px = N_[ux].c[0], py = N_[uy].c[1];
        int e = C_[s].next;
        double area = (N_[C_[s].pt].c[0] - px) * (N_[C_[e].pt].c[1] - py);
        while (!lx(C_[e].pt, ux)) {                          // p_x <= e_x
            const int q = C_[e].pt;
            e = C_[e].next;
            area += (N_[q].c[0] - px) * (N_[C_[e].pt].c[1] - N_[q].c[1]);
        }
        return area;
    }
    // computeArea_x: the same with x and y exchanged, s the outer delimiter in y
    double area_x(int ux, int uy, int s) const
    {
        const double px = N_[ux].c[0], py = N_[uy].c[1];
        int e = C_[s].prev;
        double area = (N_[C_[s].pt].c[1] - py) * (N_[C_[e].pt].c[0] - px);
        while (!ly(C_[e].pt, uy)) {                          // p_y <= e_y
            const int q = C_[e].pt;
            e = C_[e].prev;
            area += (N_[q].c[1] - py) * (N_[C_[e].pt].c[0] - N_[q].c[0]);
        }
        return area;
    }
    // removeDominated_y over a q.L, after s: the inner delimiters u* dominates
    // leave (and with them their place as an inner delimiter); the head goes
    // as well when u replaces it (lines 15 and 20 — the callers relink it)
    void drop_dominated_y(int ux, int s)
    {
        int e = C_[s].next;
        while (!lx(C_[e].pt, ux)) {
            N_[C_[e].pt].icell = -1;
            e = C_[e].next;
        }
        join(s, e);
    }
    void drop_dominated_x(int uy, int s)
    {
        int e = C_[s].prev;
        while (!ly(C_[e].pt, uy)) {
            N_[C_[e].pt].icell = -1;
            e = C_[e].prev;
        }
        join(e, s);
    }
    // updateVolumes_y: s, the inner delimiters after it and the outer one
    // that ends them, their volumes up to z
    void update_volumes(int p, int s, double z)
    {
        int e = s;
        for (;;) {
            Node& Q = N_[C_[e].pt];
            Q.v += Q.a * (z - Q.z);
            Q.z = z;
            if (e != s && lx(C_[e].pt, p)) break;            // the outer delimiter in y
            e = C_[e].next;
        }
    }
};

// Copies, the reference point and non-finite values (HVC-2, HVC-3): the rows
// of F to compute with, each first of its copies (rep), and how many copies.
inline bool hvc_rows(const std::vector<std::vector<double>>& F, const std::vector<double>& r,
                     std::vector<std::size_t>& rows, std::vector<std::size_t>& rep,
                     std::vector<int>& copies)
{
    const std::size_t n = F.size(), m = r.size();
    for (double v : r) if (!std::isfinite(v)) return false;
    std::vector<std::size_t> in;
    for (std::size_t i = 0; i < n; ++i) {
        bool inside = true;
        for (std::size_t k = 0; k < m; ++k) {
            if (std::isnan(F[i][k]) || !(F[i][k] < r[k])) { inside = false; break; }
            if (!std::isfinite(F[i][k])) return false;
        }
        if (inside) in.push_back(i);
    }
    std::sort(in.begin(), in.end(), [&](std::size_t a, std::size_t b) {
        for (std::size_t k = 0; k < m; ++k)
            if (F[a][k] != F[b][k]) return F[a][k] < F[b][k];
        return a < b;
    });
    rows.clear();
    rep.assign(n, n);
    copies.assign(n, 0);
    for (std::size_t t = 0; t < in.size(); ++t) {
        const std::size_t i = in[t];
        if (t == 0 || F[in[t - 1]] != F[i]) rows.push_back(i);
        rep[i] = rows.back();
        ++copies[rows.back()];
    }
    return true;
}

}   // namespace detail

// All contributions at M = 3 (HVC3D) or M = 4 (HVC4D) against r; at any
// other M, contributions_exact's.
inline std::vector<double> contributions_hvc(const std::vector<std::vector<double>>& F,
                                             const std::vector<double>& r)
{
    const std::size_t n = F.size(), m = r.size();
    if (m != 3 && m != 4) return contributions_exact(F, r);
    std::vector<std::size_t> rows, rep;
    std::vector<int> copies;
    if (!detail::hvc_rows(F, r, rows, rep, copies)) return contributions_exact(F, r);
    std::vector<double> c(n, 0.0);
    detail::Hvc3d S;
    std::vector<int> node(n, -1);
    if (m == 3) {
        std::vector<int> nodes;
        for (std::size_t i : rows) nodes.push_back(node[i] = S.make(F[i].data(), 0.0));
        S.build(nodes);
        S.compute(r.data());
        for (std::size_t i : rows) c[i] = S.contribution(node[i]);
    } else {
        // §IV-B: increasing w (w z y x, HVC-1); the projections into the
        // three-dimensional structure one by one, its contributions times
        // the slice's height
        static const std::size_t W_FIRST[] = {3, 2, 1, 0};
        std::vector<std::size_t> order = rows;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            for (std::size_t k : W_FIRST)
                if (F[a][k] != F[b][k]) return F[a][k] < F[b][k];
            return false;
        });
        std::vector<std::size_t> held;
        for (std::size_t t = 0; t < order.size(); ++t) {
            const std::size_t i = order[t];
            node[i] = S.make(F[i].data(), F[i][3]);
            if (S.insert(node[i])) held.push_back(i);
            const double h = (t + 1 < order.size() ? F[order[t + 1]][3] : r[3]) - F[i][3];
            if (!(h > 0.0)) continue;                       // a slice of no height
            S.compute(r.data());
            for (std::size_t j : held) c[j] += S.contribution(node[j]) * h;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        c[i] = rep[i] < n && copies[rep[i]] == 1 ? c[rep[i]] : 0.0;
    return c;
}

// The HVC3D structure kept between calls (above): the contributions of F,
// as contributions_hvc's, the structure brought to F by single-point updates
// when F differs from the last set by at most `most` points (in or out) and
// neither set has dominated points, rebuilt otherwise. M = 4 goes to
// contributions_hvc (the paper gives HVC4D no update).
class HvcKept {
public:
    std::vector<double> contributions(const std::vector<std::vector<double>>& F,
                                      const std::vector<double>& r)
    {
        const std::size_t n = F.size(), m = r.size();
        if (m != 3) return contributions_hvc(F, r);
        for (double v : r) if (!std::isfinite(v)) return contributions_exact(F, r);
        // HVC-2, HVC-3: the rows inside r, each the first of its copies (by
        // hashing, which keeps the call linear)
        std::vector<std::size_t> rows, rep(n, n);
        std::vector<int> copies(n, 0);
        first_.clear();
        for (std::size_t i = 0; i < n; ++i) {
            const double* f = F[i].data();
            if (!(f[0] < r[0] && f[1] < r[1] && f[2] < r[2])) continue;     // NaN too
            if (!std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2]))
                return contributions_exact(F, r);
            const auto got = first_.emplace(key(f), i);
            if (got.second) rows.push_back(i);
            rep[i] = got.first->second;
            ++copies[rep[i]];
        }
        // the rows held already, and the ones to add
        std::vector<int> node(n, -1), seen(static_cast<std::size_t>(S_.nodes()), 0);
        std::vector<std::size_t> add;
        for (std::size_t i : rows) {
            const auto it = at_.find(key(F[i].data()));
            if (it == at_.end()) { add.push_back(i); continue; }
            node[i] = it->second;
            seen[static_cast<std::size_t>(it->second)] = 1;
        }
        std::vector<int> gone;
        for (const auto& kv : at_) if (!seen[static_cast<std::size_t>(kv.second)]) gone.push_back(kv.second);
        std::sort(gone.begin(), gone.end());                // an order of the map's own: fixed
        const bool update = !S_.dominated() && add.size() + gone.size() <= most;
        if (update) {
            for (int g : gone) {
                at_.erase(key(S_.coords(g)));
                S_.erase(g);
                ++updates_;
            }
            for (std::size_t i : add) {
                node[i] = S_.make(F[i].data(), 0.0);
                at_[key(F[i].data())] = node[i];
                S_.insert(node[i]);
                ++updates_;
            }
        }
        if (!update || S_.dominated()) {                    // not a front: rebuilt
            S_.clear();
            at_.clear();
            std::vector<int> nodes;
            for (std::size_t i : rows) {
                nodes.push_back(node[i] = S_.make(F[i].data(), 0.0));
                at_[key(F[i].data())] = node[i];
            }
            S_.build(nodes);
            ++builds_;
        }
        S_.compute(r.data());
        std::vector<double> c(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            if (rep[i] < n && copies[rep[i]] == 1) c[i] = S_.contribution(node[rep[i]]);
        return c;
    }

    // how the calls went (tests): structures built from scratch, single-point updates
    long builds() const { return builds_; }
    long updates() const { return updates_; }
    const detail::Hvc3d& structure() const { return S_; }

    std::size_t most = 4;      // the most points in and out that are updated, not rebuilt

private:
    struct Key {
        double x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const
        {
            std::uint64_t h = 1469598103934665603ull;
            for (double v : {k.x, k.y, k.z}) {
                std::uint64_t b;
                std::memcpy(&b, &v, sizeof b);
                h = (h ^ b) * 1099511628211ull;
            }
            return static_cast<std::size_t>(h);
        }
    };
    static Key key(const double* p) { return Key{p[0] + 0.0, p[1] + 0.0, p[2] + 0.0}; }   // -0 as 0

    detail::Hvc3d S_;
    std::unordered_map<Key, int, KeyHash> at_;              // the points held: their nodes
    std::unordered_map<Key, std::size_t, KeyHash> first_;   // a call's rows: the first copy
    long builds_ = 0, updates_ = 0;
};

}   // namespace mootation::hv

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The problems of the C++ suite by family name (task 11 of 2026-10-18, §1.3):
// make(spec) gives the problem a Spec describes, the family's evaluation
// behind the map that shifts or turns its distance variables.
//
// A Spec is what benchmarks/native.py reads off a registered problem: the
// family (the registry's family names: ZDT1, DTLZ3, IDTLZ1, SDTLZ2,
// minusDTLZ2, C2-DTLZ2, MaF5, ZCAT12, ...), the sizes, and the numbers NumPy
// made for it — shift centres, rotations, MaF9's polygons, MaF8's vertices,
// MaF14's and MaF15's groups and linkage, the constants of C-DTLZ and
// SDTLZ. The registry's names of the problems themselves
// (rDTLZ1_3D_i2, shiftZDT4) stay in Python, which builds the spec of each.
//
// THE MAPS (dtlz_variants.py, shifted.py, rotated.py): the point the family
// is evaluated at, its variables from `first` on read as
//   cyclic       y = (x - c + 1/2) mod 1          shiftDTLZ1-4
//   fold         y = |x - c| / spread             shiftZDT1-3, 6, shiftDTLZ6, 7
//   shift        y = x - c                        shiftZDT4
//   rotate_half  y = 1/2 + A (x - c)              rDTLZ, rcDTLZ, r0/c0DTLZ, rMaF, rcMaF, r0/c0MaF
//   rotate_abs   y = |A (x - c)| / spread         rZDT1-3, 6
//   rotate       y = A (x - c)                    rZDT4
//   rotate_zcat  rZCAT's map (zcat.hpp)
// A the problem's matrix (Q, Lambda Q or Lambda), each of its products with a
// vector a row's pairwise sum (common.hpp, matvec).
// ============================================================================

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"
#include "dtlz.hpp"
#include "maf.hpp"
#include "zcat.hpp"
#include "zdt.hpp"

namespace mootation::suite {

struct Spec {
    std::string family;                           // ZDT1, DTLZ3, C1-DTLZ1, MaF9, ZCAT7, ...
    int n_var = 0, n_obj = 0;
    std::string map = "none";                     // see THE MAPS
    int first = 0;                                // the first variable the map reads
    std::vector<double> centre, spread;
    std::vector<std::vector<double>> matrices;    // k x k, row-major; rotate_zcat: one per size
    std::vector<double> scale;                    // SDTLZ: p^(i - 1)
    double r2 = 0.0;                              // C-DTLZ: r^2
    double centre_m = 0.0;                        // C2-DTLZ2: 1/sqrt(M)
    std::vector<double> lines;                    // MaF9: (ax, ay, dx, dy, |d|) per edge
    std::vector<std::vector<double>> polygons;    // MaF9: the forbidden polygons, x0 y0 x1 y1 ...
    std::vector<double> vertices;                 // MaF8: (x, y) per vertex
    std::vector<double> link;                     // MaF14, MaF15: (a, l, u - l) per variable of x^s
    std::vector<double> parts;                    // MaF14, MaF15: the subcomponents' sizes, group by group
};

namespace detail {

inline int number_after(const std::string& s, const std::string& prefix)
{
    if (s.size() <= prefix.size() || s.compare(0, prefix.size(), prefix) != 0)
        return 0;
    try {
        std::size_t used = 0;
        const int v = std::stoi(s.substr(prefix.size()), &used);
        return used == s.size() - prefix.size() ? v : 0;
    } catch (const std::exception&) {
        return 0;
    }
}

inline std::size_t side(std::size_t elements)
{
    std::size_t k = 0;
    while (k * k < elements) ++k;
    return k;
}

}   // namespace detail

class Problem final : public Native {
public:
    explicit Problem(Spec spec) : s_(std::move(spec))
    {
        n_var = s_.n_var;
        n_obj = s_.n_obj;
        const std::string& f = s_.family;
        using detail::number_after;
        if ((which_ = number_after(f, "ZDT")) != 0) kind_ = Kind::ZDT;
        else if ((which_ = number_after(f, "DTLZ")) != 0) kind_ = Kind::DTLZ;
        else if ((which_ = number_after(f, "IDTLZ")) != 0) kind_ = Kind::IDTLZ;
        else if ((which_ = number_after(f, "SDTLZ")) != 0) kind_ = Kind::SDTLZ;
        else if ((which_ = number_after(f, "minusDTLZ")) != 0) kind_ = Kind::Minus;
        else if ((which_ = number_after(f, "MaF")) != 0) kind_ = Kind::MaF;
        else if ((which_ = number_after(f, "ZCAT")) != 0) kind_ = Kind::ZCAT;
        else if (f == "C1-DTLZ1") set_cdtlz(1, dtlz::Cons::C1_DTLZ1, 1);
        else if (f == "C1-DTLZ3") set_cdtlz(3, dtlz::Cons::C1_DTLZ3, 1);
        else if (f == "C2-DTLZ2") set_cdtlz(2, dtlz::Cons::C2_DTLZ2, 1);
        else if (f == "C2-ConvexDTLZ2") set_cdtlz(2, dtlz::Cons::C2_Convex, 1);
        else if (f == "C3-DTLZ1") set_cdtlz(1, dtlz::Cons::C3_DTLZ1, n_obj);
        else if (f == "C3-DTLZ4") set_cdtlz(4, dtlz::Cons::C3_DTLZ4, n_obj);
        else throw std::invalid_argument("suite: unknown family '" + f + "'");
        check();
    }

    void evaluate(const double* x, double* f, double* g) const override
    {
        std::vector<double> y;
        const double* p = x;
        if (map_ != MapKind::None && map_ != MapKind::RotateZcat) {
            y.resize(static_cast<std::size_t>(n_var));
            map(x, y.data());
            p = y.data();
        }
        switch (kind_) {
            case Kind::ZDT:
                zdt::evaluate(which_, p, n_var, f);
                return;
            case Kind::DTLZ:
                dtlz::evaluate(which_, p, n_var, n_obj, f);
                return;
            case Kind::IDTLZ:
                dtlz::inverted(which_, p, n_var, n_obj, f);
                return;
            case Kind::SDTLZ:
                dtlz::evaluate(which_, p, n_var, n_obj, f);
                for (int i = 0; i < n_obj; ++i) f[i] = f[i] * s_.scale[static_cast<std::size_t>(i)];
                return;
            case Kind::Minus:
                dtlz::evaluate(which_, p, n_var, n_obj, f);
                for (int i = 0; i < n_obj; ++i) f[i] = -f[i];
                return;
            case Kind::CDTLZ:
                dtlz::evaluate(which_, p, n_var, n_obj, f);
                if (cons_ == dtlz::Cons::C2_Convex) dtlz::convex(f, n_obj);
                dtlz::constraints(cons_, f, n_obj, s_.r2, s_.centre_m, g);
                return;
            case Kind::MaF:
                if (which_ == 9) {
                    maf::maf9(p, n_obj, s_.lines, f);
                    if (n_con > 0) g[0] = maf::maf9_constraint(p, s_.polygons);
                } else if (which_ == 8) {
                    maf::maf8(p, n_obj, s_.vertices, f);
                } else if (which_ == 14 || which_ == 15) {
                    maf::lsmop(which_, p, n_var, n_obj, s_.link, s_.parts, f);
                } else if (which_ == 13) {
                    maf::maf13(p, n_obj, f);
                } else {
                    maf::evaluate(which_, p, n_var, n_obj, f);
                }
                return;
            case Kind::ZCAT:
                zcat_.evaluate(p, f);
                return;
        }
    }

    const Spec& spec() const { return s_; }

private:
    enum class Kind { ZDT, DTLZ, IDTLZ, SDTLZ, Minus, CDTLZ, MaF, ZCAT };
    enum class MapKind { None, Cyclic, Fold, Shift, RotateHalf, RotateAbs, Rotate, RotateZcat };

    Spec s_;
    Kind kind_ = Kind::DTLZ;
    MapKind map_ = MapKind::None;
    int which_ = 0;
    dtlz::Cons cons_ = dtlz::Cons::C1_DTLZ1;
    zcat::Problem zcat_;

    void set_cdtlz(int base, dtlz::Cons cons, int n_cons)
    {
        kind_ = Kind::CDTLZ;
        which_ = base;
        cons_ = cons;
        n_con = n_cons;
    }

    void fail(const std::string& why) const
    {
        throw std::invalid_argument("suite: " + s_.family + ": " + why);
    }

    // The sizes and parameters the family and the map read, before any
    // evaluation: a spec that does not hold them is refused here, not read
    // out of bounds later.
    void check()
    {
        const int n = n_var, M = n_obj;
        if (n < 1 || M < 1) fail("no variables or objectives");
        switch (kind_) {
            case Kind::ZDT:
                if (M != 2 || n < 2 || (which_ != 1 && which_ != 2 && which_ != 3 && which_ != 4 && which_ != 6))
                    fail("ZDT1-4 and ZDT6, two objectives");
                break;
            case Kind::DTLZ:
            case Kind::IDTLZ:
            case Kind::SDTLZ:
            case Kind::Minus:
            case Kind::CDTLZ:
                if (which_ < 1 || which_ > 7 || M < 2 || n < M) fail("DTLZ1-7, M >= 2, n >= M");
                if (kind_ == Kind::IDTLZ && which_ > 2) fail("IDTLZ1 and IDTLZ2");
                if (kind_ == Kind::SDTLZ && (which_ > 2 || static_cast<int>(s_.scale.size()) != M))
                    fail("SDTLZ1 and SDTLZ2 with M scale factors");
                if (kind_ == Kind::Minus && which_ != 2) fail("minusDTLZ2");
                break;
            case Kind::MaF:
                if (which_ == 9) {
                    if (n != 2 || static_cast<int>(s_.lines.size()) != 5 * M) fail("MaF9: two variables, M lines");
                    for (const auto& P : s_.polygons)
                        if (P.size() < 6 || P.size() % 2 != 0) fail("MaF9: a polygon of fewer than three vertices");
                    n_con = s_.polygons.empty() ? 0 : 1;
                } else if (which_ == 8) {
                    if (n != 2 || static_cast<int>(s_.vertices.size()) != 2 * M) fail("MaF8: two variables, M vertices");
                } else if (which_ == 14 || which_ == 15) {
                    check_lsmop();
                } else if (which_ == 13) {
                    if (n != 5 || M < 3) fail("MaF13: five variables, M >= 3");
                } else if (which_ < 1 || which_ > 7 || M < 2 || n < M) {
                    fail("MaF1-9 and 13-15 (MaF10-12 are not in the suite)");
                }
                break;
            case Kind::ZCAT:
                if (which_ < 1 || which_ > 20 || M < 2 || n <= M) fail("ZCAT1-20, M >= 2, n > M");
                zcat_.which = which_;
                zcat_.M = M;
                zcat_.n = n;
                break;
        }
        const std::string& map = s_.map;
        if (map == "none") return;
        if (map == "rotate_zcat") {
            map_ = MapKind::RotateZcat;
            if (kind_ != Kind::ZCAT || s_.matrices.empty()) fail("rotate_zcat: ZCAT and its matrices");
            for (const auto& A : s_.matrices) {
                const std::size_t k = detail::side(A.size());
                if (k == 0 || k * k != A.size() || static_cast<int>(k) >= n) fail("rotate_zcat: a matrix not k x k");
                zcat_.rotations.emplace_back(static_cast<int>(k), A);
            }
            // a matrix for every size n - m of the distance block: m = M - 1,
            // 1 for ZCAT14-16, either for ZCAT19 and ZCAT20 (by y_1)
            const bool one = zcat::degenerate(which_), either = which_ == 19 || which_ == 20;
            if (zcat_.rotation(n - (one ? 1 : M - 1)) == nullptr || (either && zcat_.rotation(n - 1) == nullptr))
                fail("rotate_zcat: no matrix for a size of the distance block");
            return;
        }
        const int k = n - s_.first;
        if (s_.first < 0 || k < 1) fail("a map with no variables to read");
        const std::size_t ku = static_cast<std::size_t>(k);
        if (map == "cyclic" || map == "shift") {
            map_ = map == "cyclic" ? MapKind::Cyclic : MapKind::Shift;
            if (s_.centre.size() != ku) fail(map + ": one centre per variable");
        } else if (map == "fold") {
            map_ = MapKind::Fold;
            if (s_.centre.size() != ku || s_.spread.size() != ku) fail("fold: one centre and spread per variable");
        } else if (map == "rotate_half" || map == "rotate" || map == "rotate_abs") {
            map_ = map == "rotate_half" ? MapKind::RotateHalf : map == "rotate" ? MapKind::Rotate : MapKind::RotateAbs;
            if (s_.centre.size() != ku || s_.matrices.size() != 1 || s_.matrices[0].size() != ku * ku)
                fail(map + ": a centre and a k x k matrix");
            if (map_ == MapKind::RotateAbs && s_.spread.size() != ku) fail("rotate_abs: one spread per variable");
        } else {
            fail("unknown map '" + map + "'");
        }
    }

    // MaF14 and MaF15: x^s of n - M + 1 variables, a link triple each, and M
    // groups of the same number of subcomponents, of one variable or more,
    // that take all of x^s.
    void check_lsmop() const
    {
        const int M = n_obj, ns = n_var - M + 1;
        if (M < 2 || ns < M) fail("MaF14, MaF15: M >= 2 and a variable of x^s per group at least");
        if (static_cast<int>(s_.link.size()) != 3 * ns) fail("MaF14, MaF15: a link (a, l, u - l) per variable of x^s");
        if (s_.parts.empty() || s_.parts.size() % static_cast<std::size_t>(M) != 0)
            fail("MaF14, MaF15: the same number of subcomponents in every group");
        double total = 0.0;
        for (double v : s_.parts) {
            if (!(v >= 1.0) || v != std::floor(v)) fail("MaF14, MaF15: a subcomponent of no variables");
            total += v;
        }
        if (total != static_cast<double>(ns)) fail("MaF14, MaF15: the subcomponents do not take x^s");
    }

    void map(const double* x, double* y) const
    {
        const int first = s_.first, k = n_var - first;
        for (int i = 0; i < first; ++i) y[i] = x[i];
        const double* c = s_.centre.data();
        const double* spread = s_.spread.data();
        switch (map_) {
            case MapKind::Cyclic:
                for (int i = 0; i < k; ++i) y[first + i] = np_mod(x[first + i] - c[i] + 0.5, 1.0);
                return;
            case MapKind::Fold:
                for (int i = 0; i < k; ++i) y[first + i] = std::abs(x[first + i] - c[i]) / spread[i];
                return;
            case MapKind::Shift:
                for (int i = 0; i < k; ++i) y[first + i] = x[first + i] - c[i];
                return;
            case MapKind::RotateHalf:
            case MapKind::RotateAbs:
            case MapKind::Rotate: {
                std::vector<double> d(static_cast<std::size_t>(k)), r(static_cast<std::size_t>(k));
                for (int i = 0; i < k; ++i) d[static_cast<std::size_t>(i)] = x[first + i] - c[i];
                matvec(s_.matrices[0], static_cast<std::size_t>(k), d.data(), r.data());
                for (int i = 0; i < k; ++i) {
                    const double v = r[static_cast<std::size_t>(i)];
                    y[first + i] = map_ == MapKind::RotateHalf ? 0.5 + v
                                 : map_ == MapKind::RotateAbs  ? std::abs(v) / spread[i]
                                                               : v;
                }
                return;
            }
            case MapKind::None:
            case MapKind::RotateZcat:
                for (int i = first; i < n_var; ++i) y[i] = x[i];
                return;
        }
    }
};

// The problem a spec describes; std::invalid_argument for one the suite does
// not have or whose parameters do not fit it.
inline std::unique_ptr<Native> make(Spec spec)
{
    return std::make_unique<Problem>(std::move(spec));
}

}   // namespace mootation::suite

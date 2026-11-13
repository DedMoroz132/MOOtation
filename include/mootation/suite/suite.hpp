#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The problems of the C++ suite by family name (task 11 of 2026-10-18, §1.3):
// make(spec) gives the problem a Spec describes, the family's evaluation
// behind the map that shifts or turns its distance variables.
//
// A Spec is what benchmarks/native.py reads off a registered problem: the
// family (the registry's family names: ZDT1, DTLZ3, IDTLZ1, SDTLZ2,
// minusDTLZ2, C2-DTLZ2, MaF5, ZCAT12, ...), the sizes, and the numbers Python
// computes for it — shift centres, MaF14's and MaF15's groups, the constants
// of C-DTLZ and SDTLZ; the rotations, the bbob instances, MaF8's and MaF9's
// polygons and MaF14's and MaF15's linkage are drawn here (task 17,
// supplement 5, §3). The registry's names of the problems themselves
// (rDTLZ1_3D_i2, shiftZDT4) stay in Python, which builds the spec of each.
//
// bbob-biobj and bbob-biobj-mixint (bbob.hpp; task 12 of 2026-10-19, §1):
// family "bbob-biobj" or "bbob-biobj-mixint", the numbers of its two
// functions in `bbob`. bbob itself (task 18, stage 2): family "bbob", one
// objective, the function bbob_fid (1..24) at instance bbob_iid drawn here,
// or its numbers in `bbob` (one function). bbob-largescale (task 20,
// §24.6.5; bbob_largescale.hpp): family "bbob_largescale", the function
// bbob_fid at instance bbob_iid drawn here, in 2 or more variables.
//
// The CEC 2013 niching benchmark at one objective (task 18, stage 3, §3.5;
// cec2013.hpp): family "CEC2013", the competition's problem cec_problem (1 ...
// 20) in its dimension; the objective is -F, F the report's (maximized).
//
// The least-squares sets at one objective (task 18, stage 1, §1d): family
// "MGH", the function mgh_id (1..35) of Moré, Garbow, Hillstrom 1981 in n_var
// variables with mgh_m residuals (mgh.hpp); family "NIST", the StRD file
// `nist` (nist.hpp, the library's copy) with n_var parameters. The objective
// is 1/2 ||r||^2 (the sum of squares MGH and NIST give, halved), the residuals
// and (MGH) their Jacobian come with it (Native::residuals); no box: the
// bounds are -inf / +inf.
//
// The problems with symmetries at one objective (task 19, §19.1; sym.hpp):
// family "sumexp" (sumexp.hpp), "lowrank", "cp3", "cp3_degenerate"
// (factor.hpp), "rbfnet", "mlp_plateau" (networks.hpp), "chaboche1d"
// (chaboche.hpp), "lj_cluster" (lj.hpp), "morse_cluster" (morse.hpp; `rho`),
// "circles_in_circle" or "circles_in_square" (circles.hpp; `container`; task
// 20, §24.6.2), the instance's numbers in `sym`; n_var 0 takes the
// family's. Least squares but lj_cluster and morse_cluster, whose gradient
// comes with it; each gives its blocks, canon, gauge directions and start
// (Native).
//
// The verification set (classic.hpp; task 17, supplement 2, §3): family
// "SCH", "FON", "POL", "KUR", "SPH" (m = n_obj), "QV" or "KURZ".
//
// WFG1-9 (wfg.hpp; task 17, supplement 1, stage 1): family "WFG1" ..
// "WFG9", `k` position variables (0: 2 (M - 1)); MaF10-12 are WFG1, WFG2 and
// WFG9 with k = M - 1.
//
// Task 17, supplement 3, stage 3a:
//   DAS-CMOP1-9 (dascmop.hpp): family "DAS-CMOP<i>", `constants` the
//     triplet's (b, d, e, r) (dascmop.params; e = inf where zeta = 0)
//   LIR-CMOP1-14 (lircmop.hpp): family "LIR-CMOP<i>"
//   MW1-14 (mw.hpp): family "MW<i>"; MW4, MW8 and MW14 at M >= 2, the rest
//     at two
//   DOC1-9 (doc.hpp): family "DOC<i>", the sizes of the paper
//   Polygon (polygon.hpp): family "Polygon", `vertices` its M vertices;
//     rcPolygon and c0Polygon with the map "rotate" from the third variable
//   IPolygon (polygon.hpp): family "IPolygon", two variables, `polygons`
//     the four polygons' M vertices each
//   SYM-PART, Omni-test, MMF1-15 (mmo.hpp): family the registry's name
//     ("SYM-PART_rotated", "Omni-test", "MMF1_z", "MMF14_a", ...)
//   RE and CRE (re.hpp): family the registry's name ("RE21", "RE22c",
//     "CRE24", ...), the sizes of the suite
//
// Task 17, stage 5: BT1-9 (bt.hpp): family "BT<i>", 30 variables; MOP1-7
// (mop.hpp): family "MOP<i>", 10 variables; RWMOP's registered RCM problems
// (rcm.hpp): family "RCM01" .. "RCM29", the suite's sizes; the bit strings
// (bits.hpp): family "OMM", "LOTZ" or "OJZJ" (`k` its jump), n bits; catZDT1
// (catzdt.hpp): family "catZDT1", the label's index and ten reals.
//
// Task 20, §20.1: LZ09 F1-F9 (lz09.hpp): family "LZ09_F<i>", the paper's
// sizes; UF1-UF10 (uf.hpp): family "UF<i>", 30 variables; CF1-CF10 (cf.hpp):
// family "CF<i>-CEC2009", 10 variables.
//
// THE MAPS (dtlz_variants.py, shifted.py, rotated.py): the point the family
// is evaluated at, its variables from `first` on read as
//   cyclic       y = (x - c + 1/2) mod 1          shiftDTLZ1-4, shiftMaF1-6
//   fold         y = |x - c| / spread             shiftZDT1-3, 6, shiftDTLZ6, 7, shiftMaF7
//   shift        y = x - c                        shiftZDT4
//   rotate_half  y = 1/2 + A (x - c)              rDTLZ, rcDTLZ, r0/c0DTLZ, rMaF, rcMaF, r0/c0MaF
//   rotate_abs   y = |A (x - c)| / spread         rZDT1-3, 6
//   rotate       y = A (x - c)                    rZDT4
//   rotate_zcat  rZCAT's map (zcat.hpp)
//   rotate_wfg   v = c + A ((x - 2i c) / 2i), y = 2i clip(v, 0, 1)    rWFG1-3, 8, 9
//   rotate_wfg_035  v = 0.35 + A (x / 2i - 0.35), y as above          rWFG4-7
// A the problem's matrix (Q, Lambda Q or Lambda), each of its products with a
// vector a row's pairwise sum (common.hpp, matvec). A spec names its matrices
// by `rotation` (task 17, supplement 5, §3): drawn here, one per size, Q from
// the key (20261009, crc32(family), M, k, instance) (instances.hpp, haar),
// Lambda = diag(10^3^((i - 1) / (2 (k - 1)))) on the left; or hands them over
// in `matrices`. rWFG's c is `centre`
// (0.35, or WFG9's optimum), WFG8's its optimum at the point's positions
// (wfg8_optimum below, as rotated.py's), 2i the variable's upper bound.
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bbob.hpp"
#include "bbob_largescale.hpp"
#include "cec2013.hpp"
#include "bits.hpp"
#include "bt.hpp"
#include "catzdt.hpp"
#include "cf.hpp"
#include "chaboche.hpp"
#include "circles.hpp"
#include "classic.hpp"
#include "common.hpp"
#include "dascmop.hpp"
#include "doc.hpp"
#include "dtlz.hpp"
#include "factor.hpp"
#include "instances.hpp"
#include "lircmop.hpp"
#include "lj.hpp"
#include "lz09.hpp"
#include "maf.hpp"
#include "mgh.hpp"
#include "mmo.hpp"
#include "morse.hpp"
#include "nist.hpp"
#include "mop.hpp"
#include "mw.hpp"
#include "networks.hpp"
#include "polygon.hpp"
#include "rcm.hpp"
#include "spec_numbers.hpp"
#include "uf.hpp"
#include "re.hpp"
#include "sumexp.hpp"
#include "wfg.hpp"
#include "zcat.hpp"
#include "zdt.hpp"

namespace mootation::suite {

// The rotations of a rotated problem (rotated.py): Q (rotate) and Lambda
// (condition) for each size of its distance block.
struct Rotation {
    std::string family;                           // the base family the key takes: "DTLZ2", "ZCAT19", ...
    int M = 0, instance = 1;
    bool rotate = true, condition = false;
    std::vector<int> sizes;
};

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
    std::vector<bbob::Function> bbob;             // bbob-biobj: f_alpha and f_beta, handed over
    int bbob_fnum = 0, bbob_iid = 1;              // or drawn here: F1..F55 at that instance (bbob::biobj)
    int bbob_fid = 0;                             // bbob: f1..f24 at bbob_iid (bbob::instance)
    int mgh_id = 0, mgh_m = 0;                    // MGH: the function and its residuals' count
    int cec_problem = 0;                          // CEC2013: the competition's problem 1 ... 20
    std::string nist;                             // NIST: the file's name (Misra1a, ...)
    int k = 0;                                    // WFG: the position variables (0: 2 (M - 1))
    bool wide = false;                            // DTLZ: wideDTLZ3's box (Problem::bounds)
    std::vector<double> constants;                // DAS-CMOP: b, d, e, r of the triplet (from it where empty)
    std::vector<double> triplet;                  // DAS-CMOP: eta, zeta, gamma
    sym::Instance sym;                            // sumexp, lowrank, ..., lj_cluster: the instance (sym.hpp)
    Rotation rotation;                            // the matrices drawn here (family empty: none)
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
        if (!s_.rotation.family.empty()) s_.matrices = matrices_of(s_.rotation);
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
        else if ((which_ = number_after(f, "WFG")) != 0) kind_ = Kind::WFG;
        else if (classic::parse(f, classic_)) kind_ = Kind::Classic;
        else if ((which_ = number_after(f, "DAS-CMOP")) != 0) kind_ = Kind::DASCMOP;
        else if ((which_ = number_after(f, "LIR-CMOP")) != 0) kind_ = Kind::LIRCMOP;
        else if ((which_ = number_after(f, "DOC")) != 0) kind_ = Kind::DOC;
        else if ((which_ = number_after(f, "MW")) != 0) kind_ = Kind::MW;
        else if ((which_ = number_after(f, "BT")) != 0) kind_ = Kind::BT;
        else if ((which_ = number_after(f, "MOP")) != 0) kind_ = Kind::MOP;
        else if ((which_ = number_after(f, "LZ09_F")) != 0) kind_ = Kind::LZ09;
        else if ((which_ = number_after(f, "UF")) != 0) kind_ = Kind::UF;
        else if (f.size() > 10 && f.compare(f.size() - 8, 8, "-CEC2009") == 0 &&
                 (which_ = number_after(f.substr(0, f.size() - 8), "CF")) != 0)
            kind_ = Kind::CF;
        else if ((which_ = number_after(f, "RCM")) != 0) kind_ = Kind::RCM;
        else if (bits::parse(f, bits_)) kind_ = Kind::Bits;
        else if (f == "catZDT1") kind_ = Kind::CatZDT;
        else if (f == "Polygon") kind_ = Kind::Polygon;
        else if (f == "IPolygon") kind_ = Kind::IPolygon;
        else if (mmo::parse(f, mmo_)) kind_ = Kind::MMO;
        else if (re::parse(f, re_)) kind_ = Kind::RE;
        else if (f == "C1-DTLZ1") set_cdtlz(1, dtlz::Cons::C1_DTLZ1, 1);
        else if (f == "C1-DTLZ3") set_cdtlz(3, dtlz::Cons::C1_DTLZ3, 1);
        else if (f == "C2-DTLZ2") set_cdtlz(2, dtlz::Cons::C2_DTLZ2, 1);
        else if (f == "C2-ConvexDTLZ2") set_cdtlz(2, dtlz::Cons::C2_Convex, 1);
        else if (f == "C3-DTLZ1") set_cdtlz(1, dtlz::Cons::C3_DTLZ1, n_obj);
        else if (f == "C3-DTLZ4") set_cdtlz(4, dtlz::Cons::C3_DTLZ4, n_obj);
        else if (f == "bbob-biobj" || f == "bbob-biobj-mixint") kind_ = Kind::BBOB;
        else if (f == "bbob") kind_ = Kind::BBOB1;
        else if (f == "bbob_largescale") kind_ = Kind::BBOBLS;
        else if (f == "MGH") kind_ = Kind::MGH;
        else if (f == "CEC2013") kind_ = Kind::CEC2013;
        else if (f == "NIST") kind_ = Kind::NIST;
        else if ((sym_ = make_sym(f, s_.sym)) != nullptr) kind_ = Kind::Sym;
        else throw std::invalid_argument("suite: unknown family '" + f + "'");
        check();
    }

    void evaluate(const double* x, double* f, double* g) const override
    {
        std::vector<double> y;
        const double* p = x;
        if (map_ == MapKind::RotateWfg || map_ == MapKind::RotateWfg035) {
            y.resize(static_cast<std::size_t>(n_var));
            map_wfg(x, y.data());
            p = y.data();
        } else if (map_ != MapKind::None && map_ != MapKind::RotateZcat) {
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
            case Kind::WFG:
                wfg::evaluate(which_, p, n_var, n_obj, wfg_k_, f);
                return;
            case Kind::Classic:
                classic::evaluate(classic_, p, n_var, n_obj, f);
                return;
            case Kind::BBOB:
                bbob_.evaluate(p, f);
                return;
            case Kind::BBOB1:
                f[0] = bbob1_(p);
                return;
            case Kind::BBOBLS:
                f[0] = bbobls_(p);
                return;
            case Kind::CEC2013:
                f[0] = -cec_.value(std::vector<double>(p, p + n_var));
                return;
            case Kind::DASCMOP:
                dascmop::evaluate(which_, p, n_var, s_.constants, f, g);
                return;
            case Kind::LIRCMOP:
                lircmop::evaluate(which_, p, n_var, f, g);
                return;
            case Kind::DOC:
                doc::evaluate(which_, p, f, g);
                return;
            case Kind::MW:
                mw::evaluate(which_, p, n_var, n_obj, f, g);
                return;
            case Kind::BT:
                bt::evaluate(which_, p, f);
                return;
            case Kind::MOP:
                mop::evaluate(which_, p, f);
                return;
            case Kind::LZ09:
                lz09::evaluate(which_, p, f);
                return;
            case Kind::UF:
                uf::evaluate(which_, p, f);
                return;
            case Kind::CF:
                cf::evaluate(which_, p, f, g);
                return;
            case Kind::RCM:
                rcm::evaluate(which_, p, f, g);
                return;
            case Kind::Bits:
                bits::evaluate(bits_, p, n_var, s_.k, f);
                return;
            case Kind::CatZDT:
                catzdt::evaluate(p, f);
                return;
            case Kind::Polygon:
                polygon::polygon(p, n_var, n_obj, s_.vertices, f);
                return;
            case Kind::IPolygon:
                polygon::ipolygon(p, n_obj, s_.polygons, f);
                return;
            case Kind::MMO:
                mmo::evaluate(mmo_, p, n_var, f);
                return;
            case Kind::RE:
                re::evaluate(re_, p, f, g);
                return;
            case Kind::Sym:
                if (n_res == 0) {
                    f[0] = sym_->value(p);
                    return;
                }
                [[fallthrough]];
            case Kind::MGH:
            case Kind::NIST: {
                std::vector<double> r(static_cast<std::size_t>(n_res));
                residuals(p, r.data(), nullptr);
                double sum = 0.0;
                for (double v : r) sum += v * v;
                f[0] = 0.5 * sum;
                return;
            }
        }
    }

    // MGH and NIST: the residuals at x, and (MGH, J not null) their Jacobian
    void residuals(const double* x, double* r, double* J) const override
    {
        const std::size_t n = static_cast<std::size_t>(n_var), m = static_cast<std::size_t>(n_res);
        const std::vector<double> xv(x, x + n);
        std::vector<double> rv, Jv;
        if (kind_ == Kind::Sym && n_res > 0) {
            if (J && !has_jacobian) throw std::logic_error("suite: " + s_.family + " gives no Jacobian");
            sym_->residuals(x, r, J);
            return;
        }
        if (kind_ == Kind::MGH) {
            mgh::evaluate(s_.mgh_id, n, m, xv, rv, J ? &Jv : nullptr);
        } else if (kind_ == Kind::NIST) {
            if (J) throw std::logic_error("suite: NIST gives no Jacobian");
            nist::residuals(nist_, xv, rv);
        } else {
            Native::residuals(x, r, J);
        }
        std::copy(rv.begin(), rv.end(), r);
        if (J) std::copy(Jv.begin(), Jv.end(), J);
    }

    // the symmetries, the gradient and the start of sym.hpp's families (Native)
    void canon(double* x) const override
    {
        if (!has_canon) Native::canon(x);
        sym_->canonical(x);
    }
    void gauge_basis(const double* x, double* N) const override
    {
        if (gauge_dim == 0) Native::gauge_basis(x, N);
        sym_->gauge(x, N);
    }
    void gradient(const double* x, double* g) const override
    {
        if (!has_gradient) Native::gradient(x, g);
        sym_->gradient(x, g);
    }
    void start(double* x) const override
    {
        if (!has_start) Native::start(x);
        sym_->start(x);
    }

    // the family's object (sym.hpp; null for the other families): its data
    // and helpers for the tests
    const sym::Family* sym_family() const { return sym_.get(); }

    const Spec& spec() const { return s_; }

    // The box of the problem, (lower, upper) of every variable, from its
    // definition (task 17, supplement 7, §2.4): the family's formula, or its
    // table where the paper gives one (doc.hpp, re.hpp, rcm.hpp, mmo.hpp,
    // classic.hpp). The maps keep the base's box. data/registry.json holds
    // the same numbers for comparison only (tests/test_registry_bounds.cpp).
    Bounds bounds() const
    {
        const int n = n_var, M = n_obj;
        switch (kind_) {
            case Kind::ZDT: {                              // ZDT4: x2 .. xn in [-5, 5]
                Bounds b = box(n, which_ == 4 ? -5.0 : 0.0, which_ == 4 ? 5.0 : 1.0);
                b[0] = {0.0, 1.0};
                return b;
            }
            case Kind::DTLZ: {                             // wideDTLZ3: the distance variables in [0, 1.2]
                Bounds b = box(n, 0.0, 1.0);
                if (s_.wide)
                    for (int i = M - 1; i < n; ++i) b[static_cast<std::size_t>(i)].second = dtlz::kWideUpper;
                return b;
            }
            case Kind::IDTLZ:
            case Kind::SDTLZ:
            case Kind::Minus:
            case Kind::CDTLZ:
            case Kind::DASCMOP:
            case Kind::LIRCMOP:
            case Kind::MOP:
            case Kind::Bits: return box(n, 0.0, 1.0);
            case Kind::MaF: {
                if (which_ == 8 || which_ == 9) return box(n, -10000.0, 10000.0);
                if (which_ == 13) return {{0.0, 1.0}, {0.0, 1.0}, {-2.0, 2.0}, {-2.0, 2.0}, {-2.0, 2.0}};
                if (which_ == 14 || which_ == 15) {        // x^f in [0, 1]; x^s by the linkage (a, l, u - l)
                    Bounds b = box(M - 1, 0.0, 1.0);
                    for (std::size_t i = 0; i + 2 < s_.link.size(); i += 3)
                        b.emplace_back(s_.link[i + 1], s_.link[i + 1] + s_.link[i + 2]);
                    return b;
                }
                return box(n, 0.0, 1.0);
            }
            case Kind::ZCAT: {                             // x_i in [-i/2, i/2] (zcat.bounds)
                Bounds b;
                for (int i = 1; i <= n; ++i) b.emplace_back(-i / 2.0, i / 2.0);
                return b;
            }
            case Kind::WFG: {                              // z_i in [0, 2i] (wfg.wfg_bounds)
                Bounds b;
                for (int i = 1; i <= n; ++i) b.emplace_back(0.0, 2.0 * i);
                return b;
            }
            case Kind::Classic: return classic::bounds(classic_, n);
            case Kind::BBOB: {                             // the integers of arity l in [0, l - 1], the reals in [-5, 5]
                if (s_.family != "bbob-biobj-mixint") return box(n, -5.0, 5.0);
                Bounds b;                                  // bbob_biobj_mixint.bounds
                for (int l : {2, 4, 8, 16})
                    for (int r = 0; r < n / 5; ++r) b.emplace_back(0.0, l - 1.0);
                for (int r = 0; r < n / 5; ++r) b.emplace_back(-5.0, 5.0);
                return b;
            }
            case Kind::BBOB1: return box(n, -5.0, 5.0);    // the search domain [-5, 5]^D (Hansen et al. 2009)
            case Kind::BBOBLS: return box(n, -5.0, 5.0);   // the same (Varelas et al. 2020)
            case Kind::DOC: return doc::bounds(which_);
            case Kind::MW: {                               // [0, upper] (mw.UPPER)
                const double upper = which_ == 6 ? 1.1 : which_ == 11 ? mw::kSqrt2
                                   : which_ == 13 || which_ == 14 ? 1.5 : 1.0;
                return box(n, 0.0, upper);
            }
            case Kind::BT: {                               // BT7: x2 .. x30 in [-1, 1] (bt._b_bt7)
                Bounds b = box(n, which_ == 7 ? -1.0 : 0.0, 1.0);
                b[0] = {0.0, 1.0};
                return b;
            }
            case Kind::RCM: return rcm::bounds(which_);
            case Kind::LZ09: return lz09::bounds(which_);
            case Kind::UF: return uf::bounds(which_);
            case Kind::CF: return cf::bounds(which_);
            case Kind::CatZDT: {                           // the label's index of three, then ten reals
                Bounds b = box(n, 0.0, 1.0);
                b[0] = {0.0, 2.0};
                return b;
            }
            case Kind::Polygon: return box(n, -1.0, 1.0);
            case Kind::IPolygon: return box(n, 0.0, 100.0);  // polygon_ishibuchi.BOX
            case Kind::MMO: return mmo::bounds(mmo_, n);
            case Kind::RE: return re::bounds(re_);
            case Kind::MGH:
            case Kind::NIST: return box(n, -std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity());   // no box
            case Kind::Sym: return sym_->bounds;
            case Kind::CEC2013: {
                const auto lo = cec2013::lower(s_.cec_problem), hi = cec2013::upper(s_.cec_problem);
                Bounds b;
                for (std::size_t j = 0; j < lo.size(); ++j) b.push_back({lo[j], hi[j]});
                return b;
            }
        }
        return {};
    }

    // Whether the suite has the problem's global Pareto set (pareto_set): the
    // multimodal families and the polygons, without a map (task 17, supplement
    // 3, stage 3a).
    bool has_pareto_set() const
    {
        if (map_ != MapKind::None) return false;
        if (kind_ == Kind::MMO) return mmo_ != mmo::Which::Omni || n_var == 3;
        return kind_ == Kind::Polygon || kind_ == Kind::IPolygon;
    }

    // About n points of the global Pareto set, every part, flat (n_var values
    // a row): mmo.pareto_set, pareto_sets.polygon, polygon_ishibuchi's; empty
    // where has_pareto_set is false.
    std::vector<double> pareto_set(int n) const
    {
        if (!has_pareto_set()) return {};
        if (kind_ == Kind::MMO) return mmo::pareto_set(mmo_, n);
        if (kind_ == Kind::Polygon) return polygon::polygon_set(n_obj, n_var, n, s_.vertices);
        return polygon::ipolygon_set(n, s_.polygons);
    }

private:
    enum class Kind { ZDT, DTLZ, IDTLZ, SDTLZ, Minus, CDTLZ, MaF, ZCAT, WFG, Classic, BBOB, BBOB1, BBOBLS, DASCMOP, LIRCMOP,
                      DOC, MW, Polygon, IPolygon, MMO, RE, BT, MOP, LZ09, UF, CF, RCM, Bits, CatZDT, MGH, NIST, Sym,
                      CEC2013 };
    enum class MapKind { None, Cyclic, Fold, Shift, RotateHalf, RotateAbs, Rotate, RotateZcat, RotateWfg,
                         RotateWfg035 };

    Spec s_;
    bits::Kind bits_ = bits::Kind::OMM;
    Kind kind_ = Kind::DTLZ;
    MapKind map_ = MapKind::None;
    int which_ = 0;
    int wfg_k_ = 0;
    classic::Which classic_ = classic::Which::SCH;
    mmo::Which mmo_ = mmo::Which::MMF1;
    re::Which re_ = re::Which::RE21;
    dtlz::Cons cons_ = dtlz::Cons::C1_DTLZ1;
    zcat::Problem zcat_;
    bbob::Pair bbob_;
    bbob::Objective bbob1_;
    bbob_ls::Objective bbobls_;
    cec2013::Problem cec_;
    nist::Dataset nist_;
    std::shared_ptr<const sym::Family> sym_;

    // the family of sym.hpp a name names (null: none of them)
    static std::shared_ptr<const sym::Family> make_sym(const std::string& f, const sym::Instance& in)
    {
        if (f == "sumexp") return std::make_shared<sym::Sumexp>(in);
        if (f == "lowrank") return std::make_shared<sym::Lowrank>(in);
        if (f == "cp3") return std::make_shared<sym::CP3>(in, false);
        if (f == "cp3_degenerate") return std::make_shared<sym::CP3>(in, true);
        if (f == "rbfnet") return std::make_shared<sym::Rbfnet>(in);
        if (f == "mlp_plateau") return std::make_shared<sym::MlpPlateau>(in);
        if (f == "chaboche1d") return std::make_shared<sym::Chaboche>(in);
        if (f == "lj_cluster") return std::make_shared<sym::LJCluster>(in);
        if (f == "morse_cluster") return std::make_shared<sym::MorseCluster>(in);
        if (f == "circles_in_circle") return std::make_shared<sym::Circles>(in, true);
        if (f == "circles_in_square") return std::make_shared<sym::Circles>(in, false);
        return nullptr;
    }

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
    // The numbers a spec no longer hands over (task 17, §2.2), computed here
    // where it leaves them out (spec_numbers.hpp): a spec of before, which
    // has them, is taken as it is.
    void fill()
    {
        namespace nb = numbers;
        const std::size_t M = static_cast<std::size_t>(n_obj);
        if (kind_ == Kind::SDTLZ && s_.scale.empty() && (which_ == 1 || which_ == 2))
            s_.scale = nb::sdtlz_scale(s_.family, M);
        if (kind_ == Kind::CDTLZ && s_.r2 == 0.0) {
            const double r = dtlz::cdtlz_radius(cons_, n_obj);
            s_.r2 = r * r;
        }
        if (kind_ == Kind::CDTLZ && cons_ == dtlz::Cons::C2_DTLZ2 && s_.centre_m == 0.0)
            s_.centre_m = 1.0 / std::sqrt(static_cast<double>(M));
        if (kind_ == Kind::MaF && (which_ == 14 || which_ == 15) && s_.parts.empty())
            for (const auto& group : nb::lsmop_parts(M))
                for (std::size_t v : group) s_.parts.push_back(static_cast<double>(v));
        if (kind_ == Kind::DASCMOP && s_.constants.empty() && s_.triplet.size() == 3)
            s_.constants = nb::das_params(s_.triplet[0], s_.triplet[1], s_.triplet[2]);
        if (kind_ == Kind::Polygon && s_.vertices.empty()) s_.vertices = nb::unit_polygon(M);
        if (kind_ == Kind::IPolygon && s_.polygons.empty()) s_.polygons = nb::ipolygons(M);
        // a map's optimum (dtlz_variants.shift_centres, rotated.Rotated.optimum)
        const std::string& map = s_.map;
        const int k = n_var - s_.first;
        if (map == "none" || map == "rotate_zcat" || map == "rotate_wfg_035" || k < 1 || !s_.centre.empty()) return;
        const std::size_t ku = static_cast<std::size_t>(k);
        if (map == "rotate_wfg") {
            // rotated.Rotated.centre_u: WFG9's optimum, 0.35 elsewhere; WFG8 none (it follows the positions)
            if (which_ == 9) s_.centre = nb::wfg9_optimum(ku);
            else if (which_ != 8) s_.centre.assign(ku, 0.35);
            return;
        }
        std::vector<double> c = nb::shift_centres(ku);
        if (kind_ == Kind::ZDT && which_ == 4)
            for (double& v : c) v = 10.0 * (v - 0.5);
        if (kind_ == Kind::Polygon)
            for (double& v : c) v = 2.0 * v - 1.0;
        s_.centre = c;
        if ((map == "fold" || map == "rotate_abs") && s_.spread.empty())
            for (double v : c) s_.spread.push_back(std::max(v, 1.0 - v));
    }

    void check()
    {
        fill();
        if (kind_ == Kind::Sym && n_var == 0) n_var = s_.n_var = sym_->n;
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
                // drawn here where the spec does not hand them over (maf.hpp)
                if (which_ == 9 && s_.lines.empty() && M >= 3) {
                    s_.lines = maf::maf9_lines(M);
                    s_.polygons = maf::maf9_regions(M);
                }
                if (which_ == 8 && s_.vertices.empty() && M >= 3) s_.vertices = maf::vertices(M);
                if ((which_ == 14 || which_ == 15) && s_.link.empty() && M >= 2)
                    s_.link = maf::lsmop_link(M, which_);
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
                    fail("MaF1-9 and 13-15 (MaF10-12 are WFG1, WFG2, WFG9 with k = M - 1)");
                }
                break;
            case Kind::ZCAT:
                if (which_ < 1 || which_ > 20 || M < 2 || n <= M) fail("ZCAT1-20, M >= 2, n > M");
                zcat_.which = which_;
                zcat_.M = M;
                zcat_.n = n;
                break;
            case Kind::WFG:
                wfg_k_ = s_.k != 0 ? s_.k : 2 * (M - 1);
                if (which_ < 1 || which_ > 9 || M < 2 || wfg_k_ < M - 1 || wfg_k_ % (M - 1) != 0 || n <= wfg_k_)
                    fail("WFG1-9, M >= 2, k positions a multiple of M - 1, n > k");
                if ((which_ == 2 || which_ == 3) && n - wfg_k_ < 2) fail("WFG2, WFG3: two distance variables at least");
                break;
            case Kind::Classic: {
                using classic::Which;
                const bool sph = classic_ == Which::SPH;
                if (sph ? (M < 2 || n < M) : M != 2) fail("two objectives (SPH: m <= n)");
                if ((classic_ == Which::SCH && n != 1) || (classic_ == Which::POL && n != 2) ||
                    ((classic_ == Which::KUR || classic_ == Which::KURZ) && n < 2))
                    fail("SCH one variable, POL two, KUR and KURZ two or more");
                break;
            }
            case Kind::BBOB:
                if (M != 2) fail("two objectives");
                if (s_.bbob_fnum != 0) {
                    if (s_.bbob_fnum < 1 || s_.bbob_fnum > 55 || n < bbob::kMinD || n > bbob::kMaxD ||
                        (s_.family == "bbob-biobj-mixint" && (n % 5 != 0 || s_.bbob_iid != 1)))
                        fail("F1..F55 in 2..15 variables (mixint: a multiple of 5, instance 1)");
                    s_.bbob = s_.family == "bbob-biobj-mixint" ? bbob::mixint(s_.bbob_fnum, n)
                                                               : bbob::biobj(s_.bbob_fnum, n, s_.bbob_iid);
                }
                bbob_ = bbob::Pair(s_.bbob, n);
                if ((bbob_.integers() > 0) != (s_.family == "bbob-biobj-mixint"))
                    fail("integer variables in bbob-biobj-mixint only");
                break;
            case Kind::CEC2013:
                if (M != 1 || s_.cec_problem < 1 || s_.cec_problem > cec2013::kProblems ||
                    n != cec2013::info(s_.cec_problem).D)
                    fail("one objective, problem 1 ... 20 in its dimension");
                cec_ = cec2013::Problem(s_.cec_problem);
                break;
            case Kind::BBOB1:
                if (M != 1) fail("one objective");
                if (s_.bbob_fid != 0) {
                    if (s_.bbob_fid < 1 || s_.bbob_fid > 24 || n < bbob::kMinD || n > bbob::kMaxSoloD ||
                        s_.bbob_iid < 1)
                        fail("f1..f24 in 2..40 variables, instance 1 or more");
                    s_.bbob = {bbob::instance(s_.bbob_fid, s_.bbob_iid, n)};
                }
                if (s_.bbob.size() != 1) fail("one bbob function");
                bbob1_ = bbob::Objective(s_.bbob[0], n);
                break;
            case Kind::BBOBLS:                             // task 20, §24.6.5 (bbob_largescale.hpp)
                if (M != 1 || s_.bbob_fid < 1 || s_.bbob_fid > 24 || n < 2 || s_.bbob_iid < 1)
                    fail("one objective, f1..f24 in 2 or more variables, instance 1 or more");
                bbobls_ = bbob_ls::Objective(bbob_ls::instance(s_.bbob_fid, s_.bbob_iid, n), n);
                break;
            case Kind::DASCMOP:
                if (which_ < 1 || which_ > 9 || M != dascmop::n_obj(which_) || n < M)
                    fail("DAS-CMOP1-6 two objectives, 7-9 three, n >= M");
                if (s_.constants.size() != 4) fail("DAS-CMOP: the constants b, d, e, r of the triplet");
                n_con = dascmop::n_con(which_);
                break;
            case Kind::LIRCMOP:
                if (which_ < 1 || which_ > 14 || M != lircmop::n_obj(which_) || n < 3)
                    fail("LIR-CMOP1-12 two objectives, 13-14 three, n >= 3");
                n_con = lircmop::n_con(which_);
                break;
            case Kind::DOC:
                if (which_ < 1 || which_ > 9 || M != doc::n_obj(which_) || n != doc::n_var(which_))
                    fail("DOC1-9 at the paper's sizes");
                n_con = doc::n_con(which_);
                break;
            case Kind::MW:
                if (which_ < 1 || which_ > 14 || M < 2 || (!mw::scalable(which_) && M != 2) || n <= M)
                    fail("MW1-14, two objectives (MW4, MW8, MW14: two or more), n > M");
                n_con = mw::n_con(which_);
                break;
            case Kind::BT:
                if (which_ < 1 || which_ > 9 || n != bt::kN || M != (which_ == 9 ? 3 : 2))
                    fail("BT1-8 two objectives, BT9 three, 30 variables");
                break;
            case Kind::MOP:
                if (which_ < 1 || which_ > 7 || n != mop::kN || M != mop::n_obj(which_))
                    fail("MOP1-5 two objectives, MOP6-7 three, 10 variables");
                break;
            case Kind::LZ09:
                if (which_ < 1 || which_ > 9 || n != lz09::n_var(which_) || M != lz09::n_obj(which_))
                    fail("LZ09 F1-F9: the paper's sizes (F6 three objectives)");
                break;
            case Kind::UF:
                if (which_ < 1 || which_ > 10 || n != uf::kN || M != uf::n_obj(which_))
                    fail("UF1-UF7 two objectives, UF8-UF10 three, 30 variables");
                break;
            case Kind::CF:
                if (which_ < 1 || which_ > 10 || n != cf::kN || M != cf::n_obj(which_))
                    fail("CF1-CF7 two objectives, CF8-CF10 three, 10 variables");
                n_con = cf::n_con(which_);
                break;
            case Kind::Bits:
                if (n < 1 || M != 2 || s_.k < 0 || (bits_ == bits::Kind::OJZJ && (s_.k < 2 || 2 * s_.k >= n)))
                    fail("two objectives of n bits (OJZJ: 2 <= k < n / 2)");
                break;
            case Kind::CatZDT:
                if (n != 1 + catzdt::kReal || M != 2) fail("the label and ten reals, two objectives");
                break;
            case Kind::RCM: {
                const rcm::Size z = rcm::size(which_);
                if (z.n_var == 0 || n != z.n_var || M != z.n_obj) fail("a registered RCM at the suite's sizes");
                n_con = z.n_con;
                break;
            }
            case Kind::Polygon:
                if (n < 2 || static_cast<int>(s_.vertices.size()) != 2 * M) fail("two variables at least, M vertices");
                break;
            case Kind::IPolygon:
                if (n != 2 || s_.polygons.empty()) fail("two variables and the polygons");
                for (const auto& P : s_.polygons)
                    if (static_cast<int>(P.size()) != 2 * M) fail("M vertices per polygon");
                break;
            case Kind::MMO:
                if (M != mmo::n_obj(mmo_) || (mmo::n_var(mmo_) != 0 && n != mmo::n_var(mmo_)))
                    fail("the competition's sizes (Omni-test: any number of variables)");
                break;
            case Kind::RE: {
                const re::Size z = re::size(re_);
                if (n != z.n_var || M != z.n_obj) fail("the suite's sizes");
                n_con = z.n_con;
                break;
            }
            case Kind::MGH: {
                const std::size_t nu = static_cast<std::size_t>(n);
                if (s_.mgh_id < 1 || s_.mgh_id > mgh::kCount || M != 1 || s_.mgh_m < 1) fail("1..35, one objective, m >= 1");
                if (mgh::fixed_n(s_.mgh_id) && mgh::fixed_n(s_.mgh_id) != nu) fail("the function's n");
                const std::size_t fm = mgh::fixed_m(s_.mgh_id, nu);
                if (fm && fm != static_cast<std::size_t>(s_.mgh_m)) fail("the function's m");
                n_res = s_.mgh_m;
                has_jacobian = true;
                break;
            }
            case Kind::Sym:
                if (M != 1 || s_.map != "none") fail("one objective, no map");
                if (n != sym_->n) fail("the instance's " + std::to_string(sym_->n) + " variables");
                n_res = sym_->m;
                has_jacobian = sym_->has_jacobian;
                has_gradient = sym_->has_gradient;
                has_canon = sym_->has_canon;
                has_start = sym_->has_start;
                gauge_dim = sym_->gauge_dim;
                blocks = sym_->blocks;
                linear = sym_->linear;
                res_blocks = sym_->res_blocks;
                break;
            case Kind::NIST:
                if (M != 1) fail("one objective");
                nist_ = nist::embedded(s_.nist);
                if (static_cast<std::size_t>(n) != nist_.n) fail("the file's parameters");
                n_res = static_cast<int>(nist_.y.size());
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
        if (map == "rotate_wfg" || map == "rotate_wfg_035") {
            map_ = map == "rotate_wfg" ? MapKind::RotateWfg : MapKind::RotateWfg035;
            if (kind_ != Kind::WFG || s_.matrices.size() != 1 || s_.matrices[0].size() != ku * ku)
                fail(map + ": WFG and a k x k matrix");
            if (map_ == MapKind::RotateWfg && s_.centre.size() != (which_ == 8 ? 0 : ku))
                fail("rotate_wfg: one centre per variable (none for WFG8)");
            return;
        }
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

    // The matrices a Rotation names, one per size: Q (or I), times Lambda
    // on the left (rotated.Rotated.matrix).
    static std::vector<std::vector<double>> matrices_of(const Rotation& r)
    {
        std::vector<std::vector<double>> out;
        for (int k : r.sizes) {
            if (k < 1) throw std::invalid_argument("suite: rotation: a size below 1");
            const std::size_t n = static_cast<std::size_t>(k);
            std::vector<double> A(n * n, 0.0);
            if (r.rotate) {
                std::mt19937 g(instances::seed({20261009u, instances::crc32(r.family),
                                                static_cast<std::uint64_t>(r.M), static_cast<std::uint64_t>(k),
                                                static_cast<std::uint64_t>(r.instance)}));
                A = instances::haar(g, k);
            } else {
                for (std::size_t i = 0; i < n; ++i) A[i * n + i] = 1.0;
            }
            if (r.condition && k > 1)
                for (std::size_t i = 0; i < n; ++i) {
                    const double lambda = dm::pow(1e3, static_cast<double>(i) / (2.0 * (k - 1)));
                    for (std::size_t j = 0; j < n; ++j) A[i * n + j] = lambda * A[i * n + j];
                }
            out.push_back(A);
        }
        return out;
    }

    // rWFG's map (rotated.py, Rotated._wfg_v and map): the distance block
    // normalised by its bounds 2i, turned about its centre, clipped to [0, 1]
    void map_wfg(const double* x, double* y) const
    {
        const int first = s_.first, k = n_var - first;
        const std::size_t ku = static_cast<std::size_t>(k);
        for (int i = 0; i < n_var; ++i) y[i] = x[i];
        std::vector<double> c(ku, 0.35), d(ku), r(ku);
        if (map_ == MapKind::RotateWfg) {
            if (which_ == 8) {
                std::vector<double> pos(static_cast<std::size_t>(first));
                for (int i = 0; i < first; ++i) pos[static_cast<std::size_t>(i)] = x[i] / (2.0 * (i + 1));
                c = wfg8_optimum(pos, k);
            } else {
                c = s_.centre;
            }
        }
        for (int i = 0; i < k; ++i) {
            const double scale = 2.0 * (first + i + 1);
            const std::size_t u = static_cast<std::size_t>(i);
            d[u] = map_ == MapKind::RotateWfg035 ? x[first + i] / scale - 0.35
                                                 : (x[first + i] - scale * c[u]) / scale;
        }
        matvec(s_.matrices[0], ku, d.data(), r.data());
        for (int i = 0; i < k; ++i) {
            const std::size_t u = static_cast<std::size_t>(i);
            const double v = (map_ == MapKind::RotateWfg035 ? 0.35 : c[u]) + r[u];
            y[first + i] = 2.0 * (first + i + 1) * clip(v, 0.0, 1.0);
        }
    }

    // WFG8's optimal normalised distance variables at the normalised
    // positions (rotated.py, wfg8_optimum): each the y with b_param(y, u) =
    // 0.35, u the mean of every value before it, positions included
    static std::vector<double> wfg8_optimum(std::vector<double> y, int l)
    {
        const std::size_t positions = y.size();
        for (int j = 0; j < l; ++j) {
            double s = 0.0;
            for (double v : y) s = s + v;
            const double u = s / static_cast<double>(y.size());
            const double v = wfg::kBpA - (1.0 - 2.0 * u) * std::abs(std::floor(0.5 - u) + wfg::kBpA);
            y.push_back(dm::pow(0.35, 1.0 / (wfg::kBpB + (wfg::kBpC - wfg::kBpB) * v)));
        }
        return std::vector<double>(y.begin() + static_cast<std::ptrdiff_t>(positions), y.end());
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
            case MapKind::RotateWfg:
            case MapKind::RotateWfg035:
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

// The box of the problem a spec describes (Problem::bounds).
inline Bounds bounds(Spec spec)
{
    return Problem(std::move(spec)).bounds();
}

}   // namespace mootation::suite

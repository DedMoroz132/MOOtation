// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// suite/wrap.hpp: the permutation, rotation and shift over any problem (task
// 20 of 2026-11-02, §20.2.1). Without a wrapper the values are the problem's
// bit for bit; the wrapped problem at T^-1(y) is the problem at y (a
// permutation alone bit for bit, a rotation or a shift to rounding); the
// Jacobian and the gradient against central differences; x_opt in the run's
// coordinates gives f_opt; the boxes; the refusals of WRAP-3; and the draws,
// the points T(x) and the values there against benchmarks/wrap.py
// (tests/data/wrap_python.json, tools/refs/gen_wrap_refs.py).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/json.hpp>
#include <mootation/suite/wrap.hpp>

namespace S = mootation::suite;
namespace js = mootation::json;

namespace {

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

using Vec = std::vector<double>;

S::Spec spec(const std::string& family, int n, int M)
{
    S::Spec s;
    s.family = family;
    s.n_var = n;
    s.n_obj = M;
    return s;
}

S::Spec bbob(int fid, int n)
{
    S::Spec s = spec("bbob", n, 1);
    s.bbob_fid = fid;
    s.bbob_iid = 1;
    return s;
}

S::Spec mgh(int id, int n, int m)
{
    S::Spec s = spec("MGH", n, 1);
    s.mgh_id = id;
    s.mgh_m = m;
    return s;
}

S::WrapSpec wraps(const std::string& name, int perm, int rotate, int shift)
{
    S::WrapSpec w;
    w.name = name;
    w.perm = perm;
    w.rotate = rotate;
    w.shift = shift;
    return w;
}

Vec start_of(const S::Spec& s)
{
    return s.family == "MGH" ? S::mgh::start(s.mgh_id, static_cast<std::size_t>(s.n_var), 1.0) : Vec{};
}

S::Bounds box_of(const S::Spec& s) { return S::bounds(s); }

// the objectives and the constraint values at x
Vec eval(const S::Native& p, const Vec& x)
{
    Vec f(static_cast<std::size_t>(p.n_obj)), g(static_cast<std::size_t>(p.n_con));
    p.evaluate(x.data(), f.data(), g.empty() ? nullptr : g.data());
    f.insert(f.end(), g.begin(), g.end());
    return f;
}

bool same_bits(const Vec& a, const Vec& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// three points of a box, a box-less variable in [-3, 3]: lo + (hi - lo) u,
// u_i = (0.1 j + 0.61803398875 i) mod 1 (tools/refs/gen_wrap_refs.py)
std::vector<Vec> points(const S::Bounds& box)
{
    std::vector<Vec> out;
    for (int j = 1; j <= 3; ++j) {
        Vec x;
        for (std::size_t i = 1; i <= box.size(); ++i) {
            double lo = box[i - 1].first, hi = box[i - 1].second;
            if (!std::isfinite(lo) || !std::isfinite(hi)) {
                lo = -3.0;
                hi = 3.0;
            }
            x.push_back(lo + (hi - lo) * std::fmod(0.1 * j + 0.61803398875 * static_cast<double>(i), 1.0));
        }
        out.push_back(x);
    }
    return out;
}

double scale_of(const Vec& v)
{
    double s = 1.0;
    for (double e : v) s = std::max(s, std::abs(e));
    return s;
}

// a and b agree to rel of the larger's scale (at least 1)
bool close(const Vec& a, const Vec& b, double rel)
{
    if (a.size() != b.size()) return false;
    const double tol = rel * std::max(scale_of(a), scale_of(b));
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!(std::abs(a[i] - b[i]) <= tol)) return false;
    return true;
}

template <typename Fn>
bool throws(Fn fn)
{
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// the residuals and their Jacobian at x
void residuals(const S::Native& p, const Vec& x, Vec& r, Vec& J)
{
    r.assign(static_cast<std::size_t>(p.n_res), 0.0);
    J.assign(static_cast<std::size_t>(p.n_res) * static_cast<std::size_t>(p.n_var), 0.0);
    p.residuals(x.data(), r.data(), J.data());
}

double objective(const S::Native& p, const Vec& x) { return eval(p, x)[0]; }

// the central difference of the objective, h_j = 1e-6 max(1, |x_j|)
Vec fd_gradient(const S::Native& p, const Vec& x)
{
    Vec g(x.size());
    for (std::size_t j = 0; j < x.size(); ++j) {
        const double h = 1e-6 * std::max(1.0, std::abs(x[j]));
        Vec a = x, b = x;
        a[j] += h;
        b[j] -= h;
        g[j] = (objective(p, a) - objective(p, b)) / (a[j] - b[j]);
    }
    return g;
}

Vec numbers(const js::Value* v)
{
    Vec out;
    if (v)
        for (const auto& e : v->items) out.push_back(e.kind == js::Value::Kind::Float ? e.f : static_cast<double>(e.i));
    return out;
}

std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// A problem of no box with a gradient, gauge directions and a start of its
// own: f(y) = sum_i a_i (y_i - y_0)^2, invariant along (1, ..., 1) (the gauge
// direction), its start (1, 2, ..., n) — the task-19 members Wrapped passes
// through (WRAP-7); blocks when asked, which Wrapped refuses
struct Toy final : S::Native {
    explicit Toy(int n, bool with_blocks = false)
    {
        n_var = n;
        n_obj = 1;
        has_gradient = has_start = true;
        gauge_dim = 1;
        if (with_blocks) blocks.push_back({0, 1, n});
    }
    double a(int i) const { return 1.0 + 0.5 * i; }
    void evaluate(const double* y, double* f, double*) const override
    {
        double s = 0.0;
        for (int i = 1; i < n_var; ++i) s += a(i) * (y[i] - y[0]) * (y[i] - y[0]);
        f[0] = s;
    }
    void gradient(const double* y, double* g) const override
    {
        g[0] = 0.0;
        for (int i = 1; i < n_var; ++i) {
            g[i] = 2.0 * a(i) * (y[i] - y[0]);
            g[0] -= g[i];
        }
    }
    void gauge_basis(const double*, double* N) const override
    {
        for (int i = 0; i < n_var; ++i) N[i] = 1.0;
    }
    void start(double* y0) const override
    {
        for (int i = 0; i < n_var; ++i) y0[i] = i + 1.0;
    }
};

}   // namespace

int main()
{
    // ---- without a wrapper: the problem's values, bit for bit -------------
    for (const S::Spec& s : {bbob(10, 5), bbob(21, 4), spec("DTLZ2", 12, 3), spec("C2-DTLZ2", 12, 3),
                             mgh(20, 6, 31)}) {
        const auto plain = S::make(s);
        const auto off = S::wrap(s, wraps("", 0, 0, 0), start_of(s));
        bool same = off->map().bounds() == box_of(s) && !off->map().active();
        for (const Vec& x : points(box_of(s))) {
            same = same && same_bits(eval(*plain, x), eval(*off, x));
            if (plain->n_res > 0) {
                Vec r1, J1, r2, J2;
                residuals(*plain, x, r1, J1);
                residuals(*off, x, r2, J2);
                same = same && same_bits(r1, r2) && same_bits(J1, J2);
            }
        }
        check(same, s.family + ": every instance 0 is the identity, bit for bit");
    }

    // ---- T^-1(y) gives the problem's value at y ----------------------------
    {
        struct Case {
            S::Spec s;
            S::WrapSpec w;
            bool exact;                                    // a permutation alone
        };
        const std::vector<Case> cases = {
            {bbob(10, 5), wraps("bbob10_n05_1D", 1, 0, 0), true},
            {spec("DTLZ2", 12, 3), wraps("DTLZ2_3D", 2, 0, 0), true},
            {spec("C2-DTLZ2", 12, 3), wraps("C2-DTLZ2_3D", 1, 0, 0), true},
            {mgh(20, 6, 31), wraps("MGH20_n06_m31_x1_1D", 4, 0, 0), true},
            {bbob(10, 5), wraps("bbob10_n05_1D", 1, 1, 1), false},
            {bbob(21, 4), wraps("bbob21_n04_1D", 0, 3, 2), false},
            {spec("DTLZ2", 12, 3), wraps("DTLZ2_3D", 2, 0, 1), false},
            {mgh(27, 10, 10), wraps("MGH27_n10_m10_x1_1D", 2, 1, 1), false},
        };
        for (const Case& c : cases) {
            const auto plain = S::make(c.s);
            const auto w = S::wrap(c.s, c.w, start_of(c.s));
            bool ok = true;
            for (const Vec& y : points(box_of(c.s))) {
                const Vec x = w->map().to_run(y);
                Vec y2(y.size());
                w->map().to_problem(x.data(), y2.data());
                const Vec a = eval(*plain, y), b = eval(*w, x);
                ok = ok && (c.exact ? same_bits(a, b) && same_bits(y, y2) : close(a, b, 1e-12) && close(y, y2, 1e-14));
            }
            check(ok, w->map().describe() + (c.exact ? ": f(T^-1 y) = f(y) bit for bit" : ": f(T^-1 y) = f(y) to 1e-12"));
        }
    }

    // ---- the residuals' Jacobian and the gradient -------------------------
    // MGH's Jacobian (mgh.hpp) through the map, J_y P Q, against central
    // differences of the wrapped residuals; the gradient of 1/2 ||r||^2,
    // pull(J_y^T r), against those of the wrapped objective; the residuals
    // themselves are the problem's at T(x), bit for bit
    for (const S::Spec& s : {mgh(1, 2, 2), mgh(20, 6, 31), mgh(27, 10, 10)}) {
        const std::string name = "MGH" + std::to_string(s.mgh_id);
        const auto plain = S::make(s);
        for (const S::WrapSpec& ws : {wraps(name, 1, 0, 0), wraps(name, 0, 1, 0), wraps(name, 2, 2, 1)}) {
            const auto w = S::wrap(s, ws, start_of(s));
            const std::size_t n = static_cast<std::size_t>(s.n_var), m = static_cast<std::size_t>(s.mgh_m);
            Vec x0 = w->map().to_run(start_of(s));
            bool same = true, jac = true, grad = true;
            for (Vec x : {x0, points(w->map().bounds())[0]}) {
                for (double& v : x) v = v * 0.9 + 0.05;    // off the start, where some Jacobians vanish
                Vec r, J, y(n), ry, Jy;
                residuals(*w, x, r, J);
                w->map().to_problem(x.data(), y.data());
                residuals(*plain, y, ry, Jy);
                same = same && same_bits(r, ry);
                double big = 1.0;
                for (double v : J) big = std::max(big, std::abs(v));
                for (std::size_t j = 0; j < n; ++j) {
                    const double h = 1e-6 * std::max(1.0, std::abs(x[j]));
                    Vec a = x, b = x, ra(m), rb(m);
                    a[j] += h;
                    b[j] -= h;
                    w->residuals(a.data(), ra.data(), nullptr);
                    w->residuals(b.data(), rb.data(), nullptr);
                    for (std::size_t i = 0; i < m; ++i)
                        jac = jac && std::abs((ra[i] - rb[i]) / (a[j] - b[j]) - J[i * n + j]) <= 1e-6 * big;
                }
                Vec gy(n, 0.0), gx(n);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j) gy[j] += Jy[i * n + j] * ry[i];
                w->map().pull(gy.data(), gx.data());
                grad = grad && close(gx, fd_gradient(*w, x), 1e-6);
            }
            check(same, w->map().describe() + ": the residuals at x are the problem's at T(x), bit for bit");
            check(jac, w->map().describe() + ": J_y P Q against central differences");
            check(grad, w->map().describe() + ": pull of the gradient against central differences");
        }
    }
    // bbob has no gradient of its own: the difference quotient in y, pulled,
    // against the one in x (Rosenbrock rotated, f8)
    {
        const S::Spec s = bbob(8, 4);
        const auto plain = S::make(s);
        const auto w = S::wrap(s, wraps("bbob08_n04_1D", 3, 1, 1));
        bool ok = true;
        for (const Vec& x : points(w->map().bounds())) {
            Vec y(x.size()), gx(x.size());
            w->map().to_problem(x.data(), y.data());
            const Vec gy = fd_gradient(*plain, y);
            w->map().pull(gy.data(), gx.data());
            ok = ok && close(gx, fd_gradient(*w, x), 1e-6);
        }
        check(ok, "bbob f8 rotated: pull of the gradient in y is the gradient in x");
    }

    // ---- x_opt in the run's coordinates gives f_opt ------------------------
    for (int fid : {1, 8, 10, 14, 21}) {
        const S::Spec s = bbob(fid, 5);
        const S::bbob::Function inst = S::bbob::instance(fid, 1, 5);
        const auto w = S::wrap(s, wraps("bbob_f" + std::to_string(fid), 2, 1, 1));
        const Vec x = w->map().to_run(inst.x_opt);
        bool inside = true;
        for (std::size_t j = 0; j < x.size(); ++j)
            inside = inside && x[j] >= w->map().bounds()[j].first && x[j] <= w->map().bounds()[j].second;
        const double f = objective(*w, x), f0 = objective(*S::make(s), inst.x_opt);
        check(std::abs(f - inst.f_opt) <= 1e-9 * (1.0 + std::abs(inst.f_opt)) && inside,
              "bbob f" + std::to_string(fid) + " wrapped: f(T^-1 x_opt) = f_opt, x_opt in the wrapped box");
        // the rotation turns about x_opt (WRAP-2): without the shift x_opt stays put up to the permutation
        const auto r = S::wrap(s, wraps("bbob_f" + std::to_string(fid), 0, 1, 0));
        check(close(r->map().to_run(inst.x_opt), inst.x_opt, 1e-15) && std::abs(f0 - inst.f_opt) <= 1e-9 * (1.0 + std::abs(inst.f_opt)),
              "bbob f" + std::to_string(fid) + ": the rotation's centre is x_opt");
    }

    // ---- the boxes ---------------------------------------------------------
    {
        const S::Spec d = spec("DTLZ2", 12, 3);
        const S::Bounds base = box_of(d);
        const auto p = S::wrap(d, wraps("DTLZ2_3D", 2, 0, 0));
        const auto& pi = p->map().permutation();
        std::vector<int> sorted(pi);
        std::sort(sorted.begin(), sorted.end());
        bool perm = true, identity = true;
        for (std::size_t i = 0; i < pi.size(); ++i) {
            perm = perm && sorted[i] == static_cast<int>(i) && p->map().bounds()[static_cast<std::size_t>(pi[i])] == base[i];
            identity = identity && pi[i] == static_cast<int>(i);
        }
        check(perm && !identity, "perm: a permutation of 0 .. n - 1, the box permuted with it");
        const auto other = S::wrap(d, wraps("DTLZ2_3D", 3, 0, 0));
        check(other->map().permutation() != pi, "perm: instances 2 and 3 differ");
        const auto sh = S::wrap(d, wraps("DTLZ2_3D", 0, 0, 1));
        const auto both = S::wrap(d, wraps("DTLZ2_3D", 2, 0, 1));
        bool moved = true;
        for (std::size_t j = 0; j < 12; ++j) {
            const double s = sh->map().shift()[j];
            moved = moved && std::abs(s) <= 1.0 && sh->map().bounds()[j].first == s && sh->map().bounds()[j].second == 1.0 + s;
        }
        check(moved, "shift: the box moves by s, |s_j| <= its width (WRAP-4)");
        check(both->map().shift() == sh->map().shift(), "shift: drawn from its own key, whatever the permutation");

        const S::Spec m = mgh(27, 10, 10);
        const Vec x0 = start_of(m);
        const auto ms = S::wrap(m, wraps("MGH27", 0, 0, 1), x0);
        bool unbounded = true, sized = true;
        for (std::size_t j = 0; j < 10; ++j) {
            unbounded = unbounded && std::isinf(ms->map().bounds()[j].first) && std::isinf(ms->map().bounds()[j].second);
            sized = sized && std::abs(ms->map().shift()[j]) <= scale_of(x0);
        }
        check(unbounded && sized, "shift without a box: bounds stay infinite, |s_j| <= max(1, |x0|_inf)");
        S::WrapSpec quarter = wraps("MGH27", 0, 0, 1);
        quarter.shift_scale = 0.25;
        const auto mq = S::wrap(m, quarter, x0);
        bool scaled = true;
        for (std::size_t j = 0; j < 10; ++j) scaled = scaled && mq->map().shift()[j] == 0.25 * ms->map().shift()[j];
        check(scaled, "shift_scale multiplies s");

        const auto q = S::wrap(m, wraps("MGH27", 0, 1, 0), x0);
        const Vec& Q = q->map().rotation();
        double worst = 0.0;
        for (std::size_t i = 0; i < 10; ++i)
            for (std::size_t j = 0; j < 10; ++j) {
                double t = 0.0;
                for (std::size_t l = 0; l < 10; ++l) t += Q[l * 10 + i] * Q[l * 10 + j];
                worst = std::max(worst, std::abs(t - (i == j ? 1.0 : 0.0)));
            }
        bool origin = true;
        for (double c : q->map().centre()) origin = origin && c == 0.0;
        check(worst < 64 * std::numeric_limits<double>::epsilon() && origin,
              "rotate: Q^T Q = I, about the origin where x_opt is unknown (WRAP-2)");
    }

    // ---- the refusals ------------------------------------------------------
    {
        const S::Bounds unit = S::box(3, 0.0, 1.0);
        check(throws([] { S::wrap(spec("DTLZ2", 12, 3), wraps("DTLZ2_3D", 0, 1, 0)); }),
              "rotate: refused for a multi-objective problem");
        check(throws([&] { S::WrapMap(wraps("box", 0, 1, 0), 3, 1, unit); }),
              "rotate: refused for a problem with a box");
        S::WrapSpec everywhere = wraps("box", 0, 1, 0);
        everywhere.defined_everywhere = true;
        check(!throws([&] { S::WrapMap(everywhere, 3, 1, unit); }),
              "rotate: a problem with a box defined on all of R^n");
        check(!throws([] { S::wrap(bbob(10, 5), wraps("bbob10_n05_1D", 0, 1, 0)); }) &&
                  !throws([] { S::wrap(mgh(1, 2, 2), wraps("MGH1", 0, 1, 0)); }),
              "rotate: bbob (defined everywhere) and MGH (no box)");
        check(throws([&] { S::WrapMap(wraps("", 1, 0, 0), 3, 1, unit); }), "the key needs a name");
        check(throws([&] { S::WrapMap(wraps("box", -1, 0, 0), 3, 1, unit); }), "an instance below 0");
        S::WrapSpec bad = wraps("box", 0, 0, 1);
        bad.shift_scale = 0.0;
        check(throws([&] { S::WrapMap(bad, 3, 1, unit); }), "shift_scale 0");
        bad.shift_scale = std::numeric_limits<double>::quiet_NaN();
        check(throws([&] { S::WrapMap(bad, 3, 1, unit); }), "shift_scale NaN");
        check(throws([&] { S::WrapMap(wraps("box", 1, 0, 0), 4, 1, unit); }), "a box of another n");
        check(throws([&] { S::WrapMap(wraps("box", 1, 0, 0), 3, 1, unit, Vec{0.5}); }), "x_opt of another n");
        check(throws([&] {
                  S::Wrapped(std::shared_ptr<const S::Native>(S::make(spec("DTLZ2", 12, 3))),
                             S::WrapMap(wraps("box", 1, 0, 0), 3, 1, unit));
              }),
              "a map of another n than the problem's");
    }

    // ---- against benchmarks/wrap.py ----------------------------------------
    {
        const js::Value doc = js::parse(read_file(std::string(MOOTATION_TEST_DATA) + "/wrap_python.json"));
        int cases = 0, points_checked = 0;
        for (const js::Value& c : doc.find("cases")->items) {
            const std::string name = c.find("name")->s, family = c.find("family")->s;
            const int n = static_cast<int>(c.find("n")->i), M = static_cast<int>(c.find("n_obj")->i);
            const S::Spec s = family == "bbob" ? bbob(static_cast<int>(c.find("bbob_fid")->i), n) : spec(family, n, M);
            S::WrapSpec ws = wraps(name, static_cast<int>(c.find("perm")->i), static_cast<int>(c.find("rotate")->i),
                                   static_cast<int>(c.find("shift")->i));
            ws.shift_scale = c.find("shift_scale")->kind == js::Value::Kind::Float ? c.find("shift_scale")->f : 1.0;
            const auto w = S::wrap(s, ws);
            const Vec pi = numbers(c.find("pi"));
            bool same_pi = pi.size() == w->map().permutation().size();
            for (std::size_t i = 0; same_pi && i < pi.size(); ++i)
                same_pi = static_cast<int>(pi[i]) == w->map().permutation()[i];
            check(same_pi, name + ": pi is wrap.py's");
            check(same_bits(numbers(c.find("s")), w->map().shift()), name + ": s is wrap.py's, bit for bit");
            check(close(numbers(c.find("Q")), w->map().rotation(), 1e-13), name + ": Q is wrap.py's to 1e-13");
            check(same_bits(numbers(c.find("centre")), w->map().centre()), name + ": the centre is wrap.py's");
            Vec box;
            for (const auto& b : w->map().bounds()) {
                box.push_back(b.first);
                box.push_back(b.second);
            }
            Vec py_box;
            for (const js::Value& b : c.find("box")->items)
                for (double v : numbers(&b)) py_box.push_back(v);
            check(same_bits(py_box, box), name + ": the box is wrap.py's");
            bool ok = true;
            for (const js::Value& p : c.find("points")->items) {
                const Vec x = numbers(p.find("x"));
                Vec y(x.size());
                w->map().to_problem(x.data(), y.data());
                ok = ok && close(y, numbers(p.find("y")), 1e-12) && close(eval(*w, x), numbers(p.find("f")), 1e-12);
                ++points_checked;
            }
            check(ok, name + ": T(x) and f(T(x)) are wrap.py's to 1e-12");
            if (const js::Value* xo = c.find("x_opt_run")) {
                const S::bbob::Function inst = S::bbob::instance(s.bbob_fid, 1, n);
                check(close(w->map().to_run(inst.x_opt), numbers(xo), 1e-12), name + ": T^-1(x_opt) is wrap.py's");
            }
            ++cases;
        }
        check(cases == 5 && points_checked == 15, "wrap_python.json: 5 cases, 15 points");
    }

    {   // WRAP-7: the gradient, the gauge directions and the start pulled through; blocks refused
        const int n = 6;
        auto toy = std::make_shared<const Toy>(n);
        const S::Bounds free(static_cast<std::size_t>(n), {-std::numeric_limits<double>::infinity(),
                                                           std::numeric_limits<double>::infinity()});
        S::WrapSpec w;
        w.name = "toy";
        w.shift = 2;
        w.rotate = 3;
        w.perm = 1;
        const S::Wrapped wt(toy, S::WrapMap(w, n, 1, free, {}, Vec{1, 2, 3, 4, 5, 6}));
        check(wt.has_gradient && wt.has_start && wt.gauge_dim == 1 && wt.blocks.empty(),
              "Wrapped: has_gradient, has_start and gauge_dim of the problem");
        const Vec x = {0.3, -1.2, 2.5, 0.7, -0.4, 1.9};
        Vec g(static_cast<std::size_t>(n));
        wt.gradient(x.data(), g.data());
        const Vec fd = fd_gradient(wt, x);
        bool grad = true;
        for (std::size_t j = 0; j < g.size(); ++j) grad = grad && std::abs(g[j] - fd[j]) <= 1e-6 * (1.0 + std::abs(fd[j]));
        check(grad, "Wrapped: its gradient against central differences of its objective");
        Vec N(static_cast<std::size_t>(n));
        wt.gauge_basis(x.data(), N.data());
        bool gauge = true;
        for (double t : {1e-3, 0.7}) {
            Vec xt = x;
            for (std::size_t j = 0; j < xt.size(); ++j) xt[j] += t * N[j];
            gauge = gauge && std::abs(objective(wt, xt) - objective(wt, x)) <= 1e-12 * (1.0 + objective(wt, x));
        }
        check(gauge, "Wrapped: the objective does not change along its gauge direction");
        Vec x0(static_cast<std::size_t>(n)), y0(static_cast<std::size_t>(n));
        wt.start(x0.data());
        wt.map().to_problem(x0.data(), y0.data());
        check(close(y0, Vec{1, 2, 3, 4, 5, 6}, 1e-12), "Wrapped: its start is T^-1 of the problem's");
        auto blocked = std::make_shared<const Toy>(n, true);
        S::WrapSpec off;
        off.name = "toy";
        check(throws([&] { S::Wrapped(blocked, S::WrapMap(w, n, 1, free)); }),
              "Wrapped: a problem with blocks is refused with a wrapper on (WRAP-7)");
        check(!throws([&] { S::Wrapped(blocked, S::WrapMap(off, n, 1, free)); }),
              "Wrapped: ... and taken as it is with every wrapper off");
    }

    std::printf("wrap: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}

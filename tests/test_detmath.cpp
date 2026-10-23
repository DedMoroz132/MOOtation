// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// detmath.hpp (task 14 of 2026-10-20, §2 and supplement 2): the library's own
// exp, log, pow, sin, cos, tan, acos, hypot; asin and atan (task 17,
// supplement 3, stage 3a: MW5-8).
//   * Accuracy and bits: 2 * 10^4 points a function (made here from integers,
//     the same doubles as tools/detmath/points.py makes) and the hard inputs
//     of tools/detmath/hard_inputs.txt, against tests/data/detmath_reference.txt
//     (tools/detmath/make_data.py): the value to 60 digits — the error within
//     the function's tolerance — and the bits dm:: gave when the file was
//     made — one file for every platform, so CI's platforms check each other.
//   * The special values of C23 Annex F (F.10.1, F.10.3, F.10.4).
//   * Monotonicity on runs of neighbouring doubles where the function is
//     monotonic, across the seams of its reductions and branches.
//   * sin and tan odd, cos even, exactly (F.10, 3).
// ============================================================================

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/detmath.hpp>

namespace {

namespace dm = mootation::dm;

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

std::uint64_t bits_of(double x)
{
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

double double_of(std::uint64_t u)
{
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

bool same_bits(double a, double b)
{
    return bits_of(a) == bits_of(b);
}

double step(double x, int k)
{
    for (int i = 0; i < k; ++i) x = std::nextafter(x, std::numeric_limits<double>::infinity());
    for (int i = 0; i > k; --i) x = std::nextafter(x, -std::numeric_limits<double>::infinity());
    return x;
}

// ── the points: tools/detmath/points.py, step for step ─────────────────────

constexpr double kTrigMax = 1647099.3291652855;
constexpr double kPio2 = 1.5707963267948966;
constexpr double kExpOver = 0x1.62e42fefa39efp+9;
constexpr double kExpUnder = -0x1.74910d52d3051p+9;

struct SplitMix64 {
    std::uint64_t s;
    std::uint64_t next()
    {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    std::uint64_t below(std::uint64_t n) { return next() % n; }
    double unit() { return static_cast<double>(next() >> 11) * 0x1p-53; }
    double between(double a, double b) { return a + (b - a) * unit(); }
    double binade(std::uint64_t e0, std::uint64_t e1)
    {
        const std::uint64_t e = e0 + below(e1 - e0 + 1);
        return double_of((e << 52) | (next() & ((std::uint64_t{1} << 52) - 1)));
    }
    double signed_(double x) { return (next() & 1u) ? -x : x; }
};

struct Point {
    double x, y;
};

Point trig_point(SplitMix64& g, int block)
{
    if (block == 0)
        return {g.between(-6.283185307179586, 6.283185307179586), 0.0};
    if (block == 1) {
        const double v = g.binade(1023 - 30, 1023 + 19);
        return {g.signed_(v), 0.0};
    }
    if (block == 2) {
        const double k = static_cast<double>(1 + g.below(std::uint64_t{1} << 20));
        const int d = static_cast<int>(g.below(7)) - 3;
        return {step(k * kPio2, d), 0.0};
    }
    if (block == 3)
        return {g.between(-1e4, 1e4), 0.0};
    const double v = g.binade(0, 1023 - 26);
    return {g.signed_(v), 0.0};
}

Point acos_point(SplitMix64& g, int block)
{
    if (block == 0)
        return {g.between(-1.0, 1.0), 0.0};
    if (block == 1) {
        const double v = 1.0 - g.binade(1023 - 53, 1023 - 2);
        return {g.signed_(v), 0.0};
    }
    if (block == 2) {
        const double v = step(0.5, static_cast<int>(g.below(2001)) - 1000);
        return {g.signed_(v), 0.0};
    }
    const double v = g.binade(0, 1023 - 2);
    return {g.signed_(v), 0.0};
}

Point atan_point(SplitMix64& g, int block)
{
    if (block == 0)
        return {g.between(-1.0, 1.0), 0.0};
    if (block == 1) {
        const double v = g.binade(1023 - 30, 1023 + 30);
        return {g.signed_(v), 0.0};
    }
    if (block == 2) {
        const double seams[3] = {0.2679491924311227, 1.0, 3.7320508075688772};
        const double x = seams[g.below(3)];
        const double v = step(x, static_cast<int>(g.below(2001)) - 1000);
        return {g.signed_(v), 0.0};
    }
    if (block == 3) {
        const double v = g.binade(1023 + 30, 2046);
        return {g.signed_(v), 0.0};
    }
    const double v = g.binade(0, 1023 - 30);
    return {g.signed_(v), 0.0};
}

Point exp_point(SplitMix64& g, int block)
{
    if (block == 0)
        return {g.between(kExpUnder, kExpOver), 0.0};
    if (block == 1) {
        const double v = g.binade(1023 - 60, 1023 + 8);
        return {g.signed_(v), 0.0};
    }
    if (block == 2)
        return {g.between(-20.0, 20.0), 0.0};
    if (block == 3) {
        if (g.next() & 1u)
            return {step(kExpOver, -static_cast<int>(g.below(1000))), 0.0};
        return {step(kExpUnder, static_cast<int>(g.below(1000))), 0.0};
    }
    return {g.between(-745.0, -700.0), 0.0};
}

Point log_point(SplitMix64& g, int block)
{
    if (block == 0) {
        const double v = g.binade(0, 2046);
        return {v > 0.0 ? v : 5e-324, 0.0};
    }
    if (block == 1) {
        const double v = g.binade(1023 - 52, 1023 - 4);
        return {1.0 + g.signed_(v), 0.0};
    }
    if (block == 2)
        return {g.between(0.5, 2.0), 0.0};
    return {g.between(0.0, 100.0), 0.0};
}

Point pow_point(SplitMix64& g, int block)
{
    if (block == 0) {
        const double x = g.between(0.0, 10.0);
        return {x, g.between(-50.0, 50.0)};
    }
    if (block == 1) {
        const double x = g.binade(1023 - 40, 1023 + 40);
        return {x, g.between(-20.0, 20.0)};
    }
    if (block == 2) {
        const double v = g.binade(1023 - 52, 1023 - 20);
        const double x = 1.0 + g.signed_(v);
        return {x, g.between(-1e8, 1e8)};
    }
    if (block == 3) {
        const double x = -g.between(0.1, 10.0);
        return {x, static_cast<double>(g.below(61)) - 30.0};
    }
    if (block == 4) {
        const double x = g.between(0.0, 1.0);
        return {x, g.between(0.0, 1.0)};
    }
    const double x = g.between(1.0, 6e14);
    return {x, g.between(-41.0, -16.0)};
}

Point hypot_point(SplitMix64& g, int block)
{
    if (block == 0) {
        const double v = g.binade(0, 2046);
        const double x = g.signed_(v);
        long long e = static_cast<long long>(bits_of(std::fabs(x)) >> 52) -
                      (static_cast<long long>(g.below(61)) - 30);
        e = e < 0 ? 0 : (e > 2046 ? 2046 : e);
        const double w = g.binade(static_cast<std::uint64_t>(e), static_cast<std::uint64_t>(e));
        return {x, g.signed_(w)};
    }
    const double x = g.between(-1e4, 1e4);
    return {x, g.between(-1e4, 1e4)};
}

struct Function {
    const char* name;
    std::uint64_t seed;
    std::vector<int> shares;                            // tenths
    Point (*point)(SplitMix64&, int);
    double (*eval)(double, double);
    double tolerance;                                   // ulp (tests/data/README.md)
};

const std::vector<Function>& functions()
{
    static const std::vector<Function> fs = {
        {"exp", 1, {4, 2, 2, 1, 1}, exp_point, [](double x, double) { return dm::exp(x); }, 1.0},
        {"log", 2, {4, 2, 2, 2}, log_point, [](double x, double) { return dm::log(x); }, 1.0},
        {"pow", 3, {3, 2, 2, 1, 1, 1}, pow_point, [](double x, double y) { return dm::pow(x, y); }, 2.0},
        {"sin", 4, {3, 3, 2, 1, 1}, trig_point, [](double x, double) { return dm::sin(x); }, 1.0},
        {"cos", 5, {3, 3, 2, 1, 1}, trig_point, [](double x, double) { return dm::cos(x); }, 1.0},
        {"tan", 6, {3, 3, 2, 1, 1}, trig_point, [](double x, double) { return dm::tan(x); }, 2.0},
        {"acos", 7, {4, 2, 2, 2}, acos_point, [](double x, double) { return dm::acos(x); }, 2.0},
        {"hypot", 8, {5, 5}, hypot_point, [](double x, double y) { return dm::hypot(x, y); }, 2.0},
        {"asin", 9, {4, 2, 2, 2}, acos_point, [](double x, double) { return dm::asin(x); }, 2.0},
        {"atan", 10, {3, 3, 2, 1, 1}, atan_point, [](double x, double) { return dm::atan(x); }, 2.0},
    };
    return fs;
}

std::vector<Point> points_of(const Function& f, int n)
{
    SplitMix64 g{f.seed};
    std::vector<Point> out;
    for (std::size_t b = 0; b < f.shares.size(); ++b)
        for (int i = 0; i < n * f.shares[b] / 10; ++i)
            out.push_back(f.point(g, static_cast<int>(b)));
    return out;
}

// One reference line: hi, q (thousandths of an ulp), d (dm's result is hi
// moved by d ulps). The error of got: |d - q/1000| in ulps of the exact value.
struct Expect {
    double hi;
    long q;
    long d;
};

void check_value(const Function& f, double got, const Expect& e, double& worst, int& wrong_bits,
                 int& too_far, const Point& p)
{
    const bool inf = std::isinf(e.hi);
    const double want = inf ? e.hi : step(e.hi, static_cast<int>(e.d));
    if (!same_bits(got, want)) {
        if (wrong_bits < 5)
            std::printf("  %s(%a, %a): got %a, the file says %a\n", f.name, p.x, p.y, got, want);
        ++wrong_bits;
    }
    if (inf)
        return;
    const double err = std::fabs(static_cast<double>(e.d) - static_cast<double>(e.q) / 1000.0);
    if (err > worst)
        worst = err;
    if (err > f.tolerance + 0.001) {
        if (too_far < 5)
            std::printf("  %s(%a, %a): %.3f ulp\n", f.name, p.x, p.y, err);
        ++too_far;
    }
}

void check_reference()
{
    std::ifstream file(MOOTATION_TEST_DATA "/detmath_reference.txt", std::ios::binary);
    check(file.good(), "tests/data/detmath_reference.txt is there");
    std::string line;
    const Function* f = nullptr;
    std::vector<Point> pts;
    std::size_t gen_left = 0, hard_left = 0, at = 0;
    double worst = 0.0;
    int wrong_bits = 0, too_far = 0, sections = 0;
    auto close = [&]() {
        if (!f)
            return;
        check(gen_left == 0 && hard_left == 0, std::string(f->name) + ": every point of the file");
        check(wrong_bits == 0, std::string(f->name) + ": the bits of the reference file");
        check(too_far == 0, std::string(f->name) + ": within the tolerance");
        std::printf("  %-5s %zu points, largest error %.3f ulp (tolerance %.0f)\n", f->name, at,
                    worst, f->tolerance);
    };
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream in(line);
        if (line[0] == '@') {
            close();
            std::string at_sign, name;
            in >> at_sign >> name >> gen_left >> hard_left;
            f = nullptr;
            for (const Function& g : functions())
                if (name == g.name)
                    f = &g;
            check(f != nullptr, "a known function in the file: " + name);
            if (!f)
                return;
            ++sections;
            pts = points_of(*f, 20000);
            check(pts.size() == gen_left, std::string(f->name) + ": the same number of points as the file");
            at = 0;
            worst = 0.0;
            wrong_bits = too_far = 0;
            continue;
        }
        if (!f)
            continue;
        std::vector<std::string> cols;
        std::string c;
        while (in >> c)
            cols.push_back(c);
        Point p{0.0, 0.0};
        std::size_t k = 0;
        if (gen_left > 0) {
            if (at < pts.size())
                p = pts[at];
            --gen_left;
        } else {
            p.x = std::strtod(cols[0].c_str(), nullptr);
            if (cols.size() == 5)
                p.y = std::strtod(cols[1].c_str(), nullptr);
            k = cols.size() - 3;
            if (hard_left > 0)
                --hard_left;
        }
        const Expect e{double_of(std::strtoull(cols[k].c_str(), nullptr, 16)),
                       std::strtol(cols[k + 1].c_str(), nullptr, 10),
                       std::strtol(cols[k + 2].c_str(), nullptr, 10)};
        check_value(*f, f->eval(p.x, p.y), e, worst, wrong_bits, too_far, p);
        ++at;
    }
    close();
    check(sections == 10, "ten functions in the reference file");
}

// ── Annex F ─────────────────────────────────────────────────────────────────

void check_special()
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto is = [](double got, double want) {
        return std::isnan(want) ? std::isnan(got) : same_bits(got, want);
    };
    struct One {
        const char* what;
        double got, want;
    };
    const One ones[] = {
        // F.10.3.1
        {"exp(+0)", dm::exp(0.0), 1.0},
        {"exp(-0)", dm::exp(-0.0), 1.0},
        {"exp(-inf)", dm::exp(-inf), 0.0},
        {"exp(+inf)", dm::exp(inf), inf},
        {"exp(NaN)", dm::exp(nan), nan},
        {"exp(1000)", dm::exp(1000.0), inf},
        {"exp(-1000)", dm::exp(-1000.0), 0.0},
        {"exp(kExpOver) finite", std::isinf(dm::exp(kExpOver)) ? 1.0 : 0.0, 0.0},
        {"exp(next above kExpOver)", dm::exp(std::nextafter(kExpOver, inf)), inf},
        {"exp(kExpUnder) not 0", dm::exp(kExpUnder) > 0.0 ? 1.0 : 0.0, 1.0},
        {"exp(next below kExpUnder)", dm::exp(std::nextafter(kExpUnder, -inf)), 0.0},
        // F.10.3.11
        {"log(+0)", dm::log(0.0), -inf},
        {"log(-0)", dm::log(-0.0), -inf},
        {"log(1)", dm::log(1.0), 0.0},
        {"log(-1)", dm::log(-1.0), nan},
        {"log(-inf)", dm::log(-inf), nan},
        {"log(+inf)", dm::log(inf), inf},
        {"log(NaN)", dm::log(nan), nan},
        // F.10.1.5-7
        {"sin(+0)", dm::sin(0.0), 0.0},
        {"sin(-0)", dm::sin(-0.0), -0.0},
        {"sin(inf)", dm::sin(inf), nan},
        {"sin(-inf)", dm::sin(-inf), nan},
        {"sin(NaN)", dm::sin(nan), nan},
        {"cos(+0)", dm::cos(0.0), 1.0},
        {"cos(-0)", dm::cos(-0.0), 1.0},
        {"cos(inf)", dm::cos(inf), nan},
        {"cos(NaN)", dm::cos(nan), nan},
        {"tan(+0)", dm::tan(0.0), 0.0},
        {"tan(-0)", dm::tan(-0.0), -0.0},
        {"tan(-inf)", dm::tan(-inf), nan},
        {"tan(NaN)", dm::tan(nan), nan},
        // F.10.1.1
        {"acos(1)", dm::acos(1.0), 0.0},
        {"acos(-1)", dm::acos(-1.0), 0x1.921fb54442d18p+1},
        {"acos(0)", dm::acos(0.0), 0x1.921fb54442d18p+0},
        {"acos(1.5)", dm::acos(1.5), nan},
        {"acos(-inf)", dm::acos(-inf), nan},
        {"acos(NaN)", dm::acos(nan), nan},
        // F.10.1.2, F.10.1.3
        {"asin(+0)", dm::asin(0.0), 0.0},
        {"asin(-0)", dm::asin(-0.0), -0.0},
        {"asin(1)", dm::asin(1.0), 0x1.921fb54442d18p+0},
        {"asin(-1)", dm::asin(-1.0), -0x1.921fb54442d18p+0},
        {"asin(1.5)", dm::asin(1.5), nan},
        {"asin(-inf)", dm::asin(-inf), nan},
        {"asin(NaN)", dm::asin(nan), nan},
        {"atan(+0)", dm::atan(0.0), 0.0},
        {"atan(-0)", dm::atan(-0.0), -0.0},
        {"atan(+inf)", dm::atan(inf), 0x1.921fb54442d18p+0},
        {"atan(-inf)", dm::atan(-inf), -0x1.921fb54442d18p+0},
        {"atan(1)", dm::atan(1.0), 0x1.921fb54442d18p-1},
        {"atan(NaN)", dm::atan(nan), nan},
        // F.10.4.4
        {"hypot(3, +0)", dm::hypot(3.0, 0.0), 3.0},
        {"hypot(-3, -0)", dm::hypot(-3.0, -0.0), 3.0},
        {"hypot(inf, NaN)", dm::hypot(inf, nan), inf},
        {"hypot(NaN, -inf)", dm::hypot(nan, -inf), inf},
        {"hypot(2, NaN)", dm::hypot(2.0, nan), nan},
        {"hypot(3, 4)", dm::hypot(3.0, 4.0), 5.0},
        {"hypot(DBL_MAX, DBL_MAX)", dm::hypot(1.7976931348623157e308, 1.7976931348623157e308), inf},
        {"hypot(3 2^-1060, 4 2^-1060)", dm::hypot(3 * 0x1p-1060, 4 * 0x1p-1060), 5 * 0x1p-1060},
        // F.10.4.5
        {"pow(+0, -3)", dm::pow(0.0, -3.0), inf},
        {"pow(-0, -3)", dm::pow(-0.0, -3.0), -inf},
        {"pow(-0, -2)", dm::pow(-0.0, -2.0), inf},
        {"pow(-0, -2.5)", dm::pow(-0.0, -2.5), inf},
        {"pow(-0, -inf)", dm::pow(-0.0, -inf), inf},
        {"pow(-0, 3)", dm::pow(-0.0, 3.0), -0.0},
        {"pow(+0, 3)", dm::pow(0.0, 3.0), 0.0},
        {"pow(-0, 2)", dm::pow(-0.0, 2.0), 0.0},
        {"pow(-0, 0.5)", dm::pow(-0.0, 0.5), 0.0},
        {"pow(-1, inf)", dm::pow(-1.0, inf), 1.0},
        {"pow(-1, -inf)", dm::pow(-1.0, -inf), 1.0},
        {"pow(1, NaN)", dm::pow(1.0, nan), 1.0},
        {"pow(NaN, 0)", dm::pow(nan, 0.0), 1.0},
        {"pow(NaN, -0)", dm::pow(nan, -0.0), 1.0},
        {"pow(-2, 0.5)", dm::pow(-2.0, 0.5), nan},
        {"pow(-2, 1.5)", dm::pow(-2.0, 1.5), nan},
        {"pow(0.5, -inf)", dm::pow(0.5, -inf), inf},
        {"pow(2, -inf)", dm::pow(2.0, -inf), 0.0},
        {"pow(0.5, inf)", dm::pow(0.5, inf), 0.0},
        {"pow(-2, inf)", dm::pow(-2.0, inf), inf},
        {"pow(-inf, -3)", dm::pow(-inf, -3.0), -0.0},
        {"pow(-inf, -2)", dm::pow(-inf, -2.0), 0.0},
        {"pow(-inf, 3)", dm::pow(-inf, 3.0), -inf},
        {"pow(-inf, 2)", dm::pow(-inf, 2.0), inf},
        {"pow(-inf, 0.5)", dm::pow(-inf, 0.5), inf},
        {"pow(inf, -1)", dm::pow(inf, -1.0), 0.0},
        {"pow(inf, 0.5)", dm::pow(inf, 0.5), inf},
        {"pow(2, NaN)", dm::pow(2.0, nan), nan},
        {"pow(NaN, 2)", dm::pow(nan, 2.0), nan},
        {"pow(-2, 3)", dm::pow(-2.0, 3.0), -8.0},
        {"pow(-2, 2^54)", dm::pow(-2.0, 0x1p54), inf},
        {"pow(-0.5, 2^54)", dm::pow(-0.5, 0x1p54), 0.0},
        {"pow(2, 1e300)", dm::pow(2.0, 1e300), inf},
        {"pow(2, -1e300)", dm::pow(2.0, -1e300), 0.0},
        {"pow(10, 400)", dm::pow(10.0, 400.0), inf},
        {"pow(10, -400)", dm::pow(10.0, -400.0), 0.0},
        {"pow(x, 2) = x x", dm::pow(1.1, 2.0), 1.1 * 1.1},
        {"pow(x, -1) = 1/x", dm::pow(3.0, -1.0), 1.0 / 3.0},
        {"pow(x, 0.5) = sqrt x", dm::pow(2.0, 0.5), std::sqrt(2.0)},
        {"pow(2, 10)", dm::pow(2.0, 10.0), 1024.0},
        {"pow(2, -1074)", dm::pow(2.0, -1074.0), 5e-324},
        {"pow(2, 1023)", dm::pow(2.0, 1023.0), 0x1p1023},
    };
    for (const One& o : ones)
        check(is(o.got, o.want), std::string("Annex F / exact: ") + o.what);
}

// ── monotonicity and symmetry ───────────────────────────────────────────────

// f does not decrease (dir = 1) or increase (dir = -1) over `count`
// neighbouring doubles from x0 upwards.
bool monotonic(double (*f)(double), double x0, int count, int dir)
{
    double x = x0, prev = f(x0);
    for (int i = 0; i < count; ++i) {
        x = std::nextafter(x, std::numeric_limits<double>::infinity());
        const double v = f(x);
        if ((dir > 0 && v < prev) || (dir < 0 && v > prev))
            return false;
        prev = v;
    }
    return true;
}

double f_exp(double x) { return dm::exp(x); }
double f_log(double x) { return dm::log(x); }
double f_acos(double x) { return dm::acos(x); }
double f_asin(double x) { return dm::asin(x); }
double f_atan(double x) { return dm::atan(x); }
double f_sin(double x) { return dm::sin(x); }
double f_cos(double x) { return dm::cos(x); }
double f_tan(double x) { return dm::tan(x); }

void check_monotonic()
{
    constexpr int run = 2000;                           // neighbours on each side
    const double ln2_32 = 0.6931471805599453 / 32.0;
    bool ok = true;
    // exp: the seams of the reduction (x = (k + 1/2) ln2/32), the ends, 0
    for (int k = -34000; k <= 32000; k += 997)
        ok = ok && monotonic(f_exp, step((k + 0.5) * ln2_32, -run), 2 * run, 1);
    ok = ok && monotonic(f_exp, step(kExpOver, -run), run, 1);
    ok = ok && monotonic(f_exp, kExpUnder, run, 1);
    ok = ok && monotonic(f_exp, step(0.0, -run), 2 * run, 1);
    ok = ok && monotonic(f_exp, step(-708.39, -run), 2 * run, 1);    // into the subnormals
    check(ok, "exp increases across its seams");
    ok = true;
    // log: F = 1 + j/128 seams, Procedure 1 / 2 thresholds, 1, powers of 2
    for (int j = 0; j < 128; ++j)
        ok = ok && monotonic(f_log, step(1.0 + (j + 0.5) / 128.0, -run), 2 * run, 1);
    ok = ok && monotonic(f_log, step(0x1.e0fabfbc702a3p-1, -run), 2 * run, 1);
    ok = ok && monotonic(f_log, step(0x1.1082b577d34eep+0, -run), 2 * run, 1);
    ok = ok && monotonic(f_log, step(1.0, -run), 2 * run, 1);
    for (int e = -1074; e <= 1023; e += 37)
        ok = ok && monotonic(f_log, step(std::ldexp(1.0, e), e > -1074 ? -run : 0), 2 * run, 1);
    check(ok, "log increases across its seams");
    ok = true;
    // acos: the branches at +-1/2, near +-1, 0
    for (double x0 : {-1.0, -0.5, 0.0, 0.5})
        ok = ok && monotonic(f_acos, x0 == -1.0 ? -1.0 : step(x0, -run), 2 * run, -1);
    ok = ok && monotonic(f_acos, step(1.0, -2 * run), 2 * run, -1);
    check(ok, "acos decreases across its branches");
    ok = true;
    // asin: the branches at +-1/2, near +-1, 0; atan: the seams 2 - sqrt 3, 1,
    // 2 + sqrt 3 of the reduction, either sign, and 0
    for (double x0 : {-1.0, -0.5, 0.0, 0.5})
        ok = ok && monotonic(f_asin, x0 == -1.0 ? -1.0 : step(x0, -run), 2 * run, 1);
    ok = ok && monotonic(f_asin, step(1.0, -2 * run), 2 * run, 1);
    for (double x0 : {0.2679491924311227, 1.0, 3.7320508075688772})
        ok = ok && monotonic(f_atan, step(x0, -run), 2 * run, 1) && monotonic(f_atan, step(-x0, -run), 2 * run, 1);
    ok = ok && monotonic(f_atan, step(0.0, -run), 2 * run, 1);
    check(ok, "asin and atan increase across their branches and seams");
    ok = true;
    // sin increasing, cos decreasing on (-pi/2, pi/2) / (0, pi) across the seam
    // of the reduction at pi/4; tan increasing across its seams k pi/2 + pi/4
    ok = ok && monotonic(f_sin, step(0.7853981633974483, -run), 2 * run, 1);
    ok = ok && monotonic(f_cos, step(0.7853981633974483, -run), 2 * run, -1);
    ok = ok && monotonic(f_cos, step(2.356194490192345, -run), 2 * run, -1);
    ok = ok && monotonic(f_sin, step(-0.7853981633974483, -run), 2 * run, 1);
    for (int k = 0; k < 50; ++k)
        ok = ok && monotonic(f_tan, step(k * kPio2 * 2.0 + 0.7853981633974483, -run), 2 * run, 1);
    check(ok, "sin, cos, tan monotonic across the seams of the reduction");
}

void check_symmetry()
{
    SplitMix64 g{99};
    bool ok = true;
    for (int i = 0; i < 100000; ++i) {
        const double x = g.between(0.0, 1e4);
        ok = ok && same_bits(dm::sin(-x), -dm::sin(x)) && same_bits(dm::cos(-x), dm::cos(x)) &&
             same_bits(dm::tan(-x), -dm::tan(x));
    }
    check(ok, "sin and tan odd, cos even, bit for bit");
    ok = true;
    for (int i = 0; i < 100000; ++i) {
        const double x = g.between(0.0, 1.0), y = g.between(0.0, 1e4);
        ok = ok && same_bits(dm::asin(-x), -dm::asin(x)) && same_bits(dm::atan(-y), -dm::atan(y)) &&
             same_bits(dm::atan(-x), -dm::atan(x));
    }
    check(ok, "asin and atan odd, bit for bit");
    check(dm::trig_in_domain(kTrigMax) && dm::trig_in_domain(-kTrigMax) &&
              !dm::trig_in_domain(std::nextafter(kTrigMax, 2e6)) && !dm::trig_in_domain(-1.7e6),
          "trig_in_domain: |x| <= 2^20 pi/2");
}

}   // namespace

int main()
{
    check_reference();
    check_special();
    check_monotonic();
    check_symmetry();
    std::printf("detmath: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}

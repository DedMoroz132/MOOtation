// SPDX-License-Identifier: Apache-2.0
// The profiles of lm's answer (task 28, §28.4; scalar.hpp LOCAL-10, PRF):
//   * a linear model r = A x - b: each parameter's interval is x_hat_j +-
//     sqrt(3.84 C_jj), C = (A^T A)^-1, to 1e-6; every parameter identifiable;
//     the Jacobian's rank n; a block weight of 4 halves the intervals (PRF-2);
//   * a model of x0 + x1 only: x0's profile flat (structural), the rank n - 1;
//   * profiles = off: no setting added; with profiles the run's answer,
//     counts and records are those without, bit for bit, the profiles'
//     evaluations counted apart, in_profiles() true while they call the
//     problem (PRF-1);
//   * profiles = list with profile_params [2]: that parameter only;
//   * lm_rank = on alone: the rank, its calls apart (PRF-6);
//   * refused: list without profile_params, profile_params with all, lm_loss
//     = huber, a periodic variable (PRF-4).

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/json.hpp>
#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
namespace json = mootation::json;
using Vec = std::vector<double>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// r = A x - b, A 12 x 3 (a cubic's design at t = 0 .. 1.1), b from x = (1, -2, 0.5) and a fixed wobble
static sc::Problem linear()
{
    sc::Problem P;
    P.m = 1;
    P.lo.assign(3, -10.0);
    P.hi.assign(3, 10.0);
    P.n_res = 12;
    P.residuals = [](const Vec& x, Vec& r) {
        r.assign(12, 0.0);
        for (int i = 0; i < 12; ++i) {
            const double t = 0.1 * i;
            const double b = 1.0 - 2.0 * t + 0.5 * t * t + 0.01 * ((i * 7) % 5 - 2);
            r[static_cast<std::size_t>(i)] = x[0] + x[1] * t + x[2] * t * t - b;
        }
    };
    P.jacobian = [](const Vec&, sc::Rows& J) {
        J.assign(12, Vec(3));
        for (int i = 0; i < 12; ++i) {
            const double t = 0.1 * i;
            J[static_cast<std::size_t>(i)] = {1.0, t, t * t};
        }
    };
    return P;
}

static std::string setting(const sc::Result& r, const std::string& k)
{
    for (const auto& [a, b] : r.settings)
        if (a == k) return b;
    return "";
}

static sc::Result run(const sc::Problem& P, const sc::Knobs& k)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0 = Vec(P.lo.size(), 0.0);
    return sc::run("lm", P, 1e-6, false, 0.0, 1, 2000, 1, k, {}, 0, s);
}

static sc::Knob word(const std::string& w)
{
    sc::Knob k;
    k.is_word = true;
    k.word = w;
    return k;
}

// C = (A^T A)^-1 of linear()'s design, by Gauss-Jordan
static std::vector<Vec> covariance(double weight)
{
    double M[3][6] = {};
    for (int i = 0; i < 12; ++i) {
        const double t = 0.1 * i, a[3] = {1.0, t, t * t};
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) M[p][q] += weight * a[p] * a[q];
    }
    for (int p = 0; p < 3; ++p) M[p][3 + p] = 1.0;
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int r = c + 1; r < 3; ++r)
            if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
        for (int q = 0; q < 6; ++q) std::swap(M[c][q], M[piv][q]);
        const double d = M[c][c];
        for (int q = 0; q < 6; ++q) M[c][q] /= d;
        for (int r = 0; r < 3; ++r)
            if (r != c) {
                const double f = M[r][c];
                for (int q = 0; q < 6; ++q) M[r][q] -= f * M[c][q];
            }
    }
    std::vector<Vec> C(3, Vec(3));
    for (int p = 0; p < 3; ++p)
        for (int q = 0; q < 3; ++q) C[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)] = M[p][3 + q];
    return C;
}

static void intervals(double weight)
{
    sc::Problem P = linear();
    if (weight != 1.0) P.weights = {weight};
    const sc::Result r = run(P, {{"profiles", word("all")}});
    const json::Value a = json::parse(setting(r, "profiles_json"));
    const std::vector<Vec> C = covariance(weight);
    const double q = mootation::scalar::lm::chi2_quantile(0.95, 1);
    double worst = 0.0;
    bool kinds = a.items.size() == 3;
    for (std::size_t j = 0; kinds && j < 3; ++j) {
        const json::Value& e = a.items[j];
        const double half = std::sqrt(q * C[j][j]), x = e.find("x")->f;
        worst = std::max(worst, std::fabs(e.find("lower")->f - (x - half)) / half);
        worst = std::max(worst, std::fabs(e.find("upper")->f - (x + half)) / half);
        kinds = kinds && e.find("kind")->s == "identifiable" && e.find("j")->i == static_cast<std::int64_t>(j);
    }
    char line[200];
    std::snprintf(line, sizeof line,
                  "linear model, weight %g: intervals x_hat +- sqrt(3.84 C_jj) (worst relative error %.1e); identifiable; rank 3",
                  weight, worst);
    check(kinds && worst < 1e-6 && setting(r, "jacobian_rank") == "3", line);
}

static void flat()
{
    sc::Problem P;
    P.m = 1;
    P.lo.assign(3, -10.0);
    P.hi.assign(3, 10.0);
    P.n_res = 6;
    P.residuals = [](const Vec& x, Vec& r) {         // x0 and x1 enter as their sum only
        r.assign(6, 0.0);
        for (int i = 0; i < 6; ++i) {
            const double t = 0.2 * i;
            r[static_cast<std::size_t>(i)] = (x[0] + x[1]) + x[2] * t - (1.0 + 0.5 * t);
        }
    };
    const sc::Result r = run(P, {{"profiles", word("list")}, {"profile_params", word("[0]")}});
    const json::Value a = json::parse(setting(r, "profiles_json"));
    check(a.items.size() == 1 && a.items[0].find("kind")->s == "structural" &&
              a.items[0].find("lower")->kind == json::Value::Kind::Null && setting(r, "jacobian_rank") == "2",
          "x0 + x1 only: x0's profile flat (structural), no crossing, rank 2 (forward differences)");
}

static void bit_for_bit()
{
    const sc::Problem P = linear();
    const sc::Result a = run(P, {});
    const sc::Result b = run(P, {{"profiles", word("all")}});
    const sc::Result c = run(P, {{"profiles", word("off")}});
    bool same = a.variables == b.variables && a.objectives == b.objectives && a.evaluations == b.evaluations &&
                a.grad_calls == b.grad_calls && a.improvements.size() == b.improvements.size();
    for (std::size_t k = 0; same && k < a.improvements.size(); ++k)
        same = a.improvements[k].f == b.improvements[k].f &&
               a.improvements[k].evaluations == b.improvements[k].evaluations;
    std::vector<std::pair<std::string, std::string>> rest;   // b's settings without the profiles' own
    for (const auto& kv : b.settings)
        if (kv.first != "profiles" && kv.first != "profile_params" && kv.first != "jacobian_rank" &&
            kv.first != "profiles_json" && kv.first != "profile_evaluations" && kv.first != "profile_jacobians")
            rest.push_back(kv);
    const bool prefix = rest == a.settings;
    check(same && prefix && a.settings == c.settings && setting(a, "profiles_json").empty() &&
              std::stoll(setting(b, "profile_evaluations")) > 0,
          "profiles: the run's answer, counts and records as without them; off adds no setting; the profiles' "
          "evaluations apart (" + setting(b, "profile_evaluations") + ", " + setting(b, "profile_jacobians") +
              " Jacobians)");
    sc::Problem Q = linear();                         // a caller's own count, as a campaign's ledger keeps
    long long counted = 0, outside = 0;
    const auto inner = Q.residuals;
    Q.residuals = [&](const Vec& x, Vec& r) {
        if (sc::in_profiles()) ++outside;
        else ++counted;
        inner(x, r);
    };
    const sc::Result q = run(Q, {{"profiles", word("all")}});
    check(counted == q.evaluations && outside == std::stoll(setting(q, "profile_evaluations")) && !sc::in_profiles(),
          "in_profiles(): a caller counting the problem's calls leaves the profiles' out (" + std::to_string(counted) +
              " counted = the run's evaluations; " + std::to_string(outside) + " the profiles')");
    const sc::Result k = run(P, {{"lm_rank", word("on")}});
    check(k.variables == a.variables && k.evaluations == a.evaluations && setting(k, "jacobian_rank") == "3" &&
              std::stoll(setting(k, "rank_evaluations")) > 0 && setting(k, "profiles_json").empty(),
          "lm_rank = on without profiles: jacobian_rank 3, its calls apart, the run as without (PRF-6)");
    const sc::Result d = run(P, {{"profiles", word("list")}, {"profile_params", word("[2]")}});
    const json::Value e = json::parse(setting(d, "profiles_json"));
    check(e.items.size() == 1 && e.items[0].find("j")->i == 2 && setting(d, "profile_params") == "[2]",
          "profiles = list, profile_params [2]: that parameter only");
}

static void refusals()
{
    const sc::Problem P = linear();
    auto refused = [&](const sc::Knobs& k) {
        try {
            run(P, k);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Problem Q = linear();
    Q.periodic = {1, 0, 0};
    bool periodic = false;
    try {
        run(Q, {{"profiles", word("all")}});
    } catch (const std::invalid_argument& e) {
        periodic = std::string(e.what()).find("PRF-4") != std::string::npos;
    }
    check(refused({{"profiles", word("list")}}) &&
              refused({{"profiles", word("all")}, {"profile_params", word("[0]")}}) &&
              refused({{"profiles", word("all")}, {"lm_loss", word("huber")}}) &&
              refused({{"profiles", word("list")}, {"profile_params", word("[3]")}}) &&
              refused({{"profiles", word("list")}, {"profile_params", word("[]")}}) && periodic,
          "refused: list without profile_params, profile_params with all, lm_loss = huber, an index past n, an "
          "empty list, a periodic variable (PRF-4)");
}

int main()
{
    intervals(1.0);
    intervals(4.0);
    flat();
    bit_for_bit();
    refusals();
    std::printf(failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}

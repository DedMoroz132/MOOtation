// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// _core.scalar (task 17, stage 6): the scalarization methods, mosqp, boostdms
// and the polishing of include/mootation/scalar/, and their local solvers on
// their own (slsqp, nelder_mead) for the tests that hold them against SciPy.
//
// The problem is a Python function (fn, and constraints when it has any:
// the inequalities, then the equalities), called with the GIL, or a problem
// of the C++ suite (_core.Native), evaluated without it. What a Python
// function raises goes up through the method to the caller.
// ============================================================================

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/lm.hpp>
#include <mootation/suite/mgh.hpp>
#include <mootation/suite/nist.hpp>
#include <mootation/scalar/polish.hpp>
#include <mootation/scalar/scalar.hpp>
#include <mootation/suite/native.hpp>

namespace py = pybind11;
namespace sc = mootation::scalar;

namespace {

using Vec = std::vector<double>;

Vec floats(const py::handle& seq)
{
    Vec out;
    for (py::handle v : seq) out.push_back(py::cast<double>(v));
    return out;
}

py::list to_list(const Vec& v)
{
    py::list l;
    for (double t : v) l.append(t);
    return l;
}

py::list rows_to_list(const sc::Rows& R)
{
    py::list l;
    for (const Vec& r : R) l.append(to_list(r));
    return l;
}

// The problem: bounds, and either native or fn (+ constraints).
sc::Problem problem_of(const py::object& fn, const py::object& constraints, const py::object& bounds,
                       int n_objs, int n_cons, int n_eq,
                       const std::shared_ptr<mootation::suite::Native>& native)
{
    sc::Problem P;
    for (py::handle b : bounds) {
        py::sequence pair = py::reinterpret_borrow<py::sequence>(b);
        P.lo.push_back(py::cast<double>(pair[0]));
        P.hi.push_back(py::cast<double>(pair[1]));
    }
    P.m = n_objs;
    P.n_cons = n_cons;
    P.n_eq = n_eq;
    if (native) {
        if (native->n_var != static_cast<int>(P.lo.size()) || native->n_obj != n_objs ||
            native->n_con != n_cons || n_eq != 0)
            throw std::invalid_argument("native: bounds, n_objs and n_cons must be the suite problem's "
                                        "(no equalities)");
        auto nat = native;
        P.evaluate = [nat](const Vec& x, Vec& f, Vec& g) {
            nat->evaluate(x.data(), f.data(), g.empty() ? nullptr : g.data());
        };
        return P;
    }
    const int m = n_objs, k = n_cons;
    P.evaluate = [fn, constraints, m, k](const Vec& x, Vec& f, Vec& g) {
        py::gil_scoped_acquire gil;
        py::list xs = to_list(x);
        Vec fv = floats(fn(xs));
        if (static_cast<int>(fv.size()) != m)
            throw std::invalid_argument("fn returned " + std::to_string(fv.size()) + " values for " +
                                        std::to_string(m) + " objectives");
        f = fv;
        if (k > 0) {
            Vec gv = floats(constraints(xs));
            if (static_cast<int>(gv.size()) != k)
                throw std::invalid_argument("constraints returned " + std::to_string(gv.size()) +
                                            " values for " + std::to_string(k));
            g = gv;
        }
    };
    return P;
}

sc::Observer observer_of(const py::object& on_generation)
{
    if (on_generation.is_none()) return {};
    return [on_generation](long long gen, const sc::Rows& F, const sc::Rows& X) {
        py::gil_scoped_acquire gil;
        on_generation(gen, rows_to_list(F), rows_to_list(X));
    };
}

sc::Knobs knobs_of(const py::dict& knobs)
{
    sc::Knobs out;
    for (auto item : knobs) {
        sc::Knob k;
        if (py::isinstance<py::str>(item.second)) {
            k.is_word = true;
            k.word = py::cast<std::string>(item.second);
        } else {
            k.number = py::cast<double>(item.second);
        }
        out.emplace_back(py::cast<std::string>(item.first), k);
    }
    return out;
}

std::vector<sc::Point> points_of(const py::object& seq)
{
    std::vector<sc::Point> out;
    for (py::handle p : seq) {
        py::sequence t = py::reinterpret_borrow<py::sequence>(p);
        sc::Point q;
        q.x = floats(t[0]);
        q.f = floats(t[1]);
        if (t.size() > 2 && !t[2].is_none()) q.g = floats(t[2]);
        out.push_back(q);
    }
    return out;
}

template <typename F>
auto maybe_released(bool release, F&& body)
{
    if (release) {
        py::gil_scoped_release unlock;
        return body();
    }
    return body();
}

}   // namespace

void bind_scalar(py::module_& m)
{
    py::module_ s = m.def_submodule(
        "scalar",
        "The scalarization methods, mosqp, boostdms and polish (include/mootation/scalar/),\n"
        "and their local solvers: slsqp (SQP with Lawson-Hanson's least-squares subproblem)\n"
        "and nelder_mead (Lagarias et al. 1998 with Gao & Han's adaptive coefficients).");
    s.attr("METHOD_VERSION") = sc::METHOD_VERSION;
    s.attr("NAMES") = py::cast(sc::names());

    s.def("run",
          [](const std::string& name, const py::object& fn, const py::object& constraints,
             const py::object& bounds, int n_objs, int n_cons, int n_eq, double eq_tol,
             const py::object& eq_tol0, int pop_size, long long budget, unsigned seed,
             const py::dict& knobs, const py::object& on_generation, int record_every,
             const py::object& design, const py::object& seed_points,
             const std::shared_ptr<mootation::suite::Native>& native) {
              sc::Problem P = problem_of(fn, constraints, bounds, n_objs, n_cons, n_eq, native);
              sc::Start st;
              if (!seed_points.is_none()) {
                  st.has_seed_points = true;
                  st.seed_points = points_of(seed_points);
              } else if (!design.is_none()) {
                  st.has_design = true;
                  for (py::handle r : design) st.design.push_back(floats(r));
              }
              const sc::Knobs kn = knobs_of(knobs);
              const bool has0 = !eq_tol0.is_none();
              const double tol0 = has0 ? py::cast<double>(eq_tol0) : 0.0;
              sc::Observer obs = observer_of(on_generation);
              sc::Result r = maybe_released(native != nullptr, [&]() {
                  return sc::run(name, P, eq_tol, has0, tol0, pop_size, budget, seed, kn, obs,
                                 record_every, st);
              });
              py::dict d;
              d["objectives"] = rows_to_list(r.objectives);
              d["variables"] = rows_to_list(r.variables);
              d["cv"] = to_list(r.cv);
              d["ignored"] = py::cast(r.ignored);
              d["evaluations"] = r.evaluations;
              d["n_failed"] = r.n_failed;
              d["settings"] = py::cast(r.settings);
              d["solutions"] = rows_to_list(r.solutions);
              return d;
          },
          py::arg("name"), py::arg("fn"), py::arg("constraints"), py::arg("bounds"),
          py::arg("n_objs"), py::arg("n_cons"), py::arg("n_eq"), py::arg("eq_tol"),
          py::arg("eq_tol0"), py::arg("pop_size"), py::arg("budget"), py::arg("seed"),
          py::arg("knobs"), py::arg("on_generation") = py::none(), py::arg("record_every") = 0,
          py::arg("design") = py::none(), py::arg("seed_points") = py::none(),
          py::arg("native") = nullptr,
          "Run one of NAMES; a dict of the answer (objectives, variables, cv), evaluations,\n"
          "n_failed (failed evaluations in the archive), settings, ignored knobs and, for the\n"
          "scalar presets, solutions.");

    s.def("polish",
          [](const py::object& fn, const py::object& constraints, const py::object& bounds,
             int n_objs, int n_cons, int n_eq, double eq_tol, int pop_size, long long budget,
             const py::object& points, const py::object& base_variables, long long base_evaluations,
             const std::string& how, long long n_points, const py::object& on_generation,
             int record_every, double fd_step, double local_tol,
             const std::shared_ptr<mootation::suite::Native>& native) {
              sc::Problem P = problem_of(fn, constraints, bounds, n_objs, n_cons, n_eq, native);
              const std::vector<sc::Point> pts = points_of(points);
              sc::Rows base;
              for (py::handle r : base_variables) base.push_back(floats(r));
              sc::Observer obs = observer_of(on_generation);
              sc::PolishResult r = maybe_released(native != nullptr, [&]() {
                  return sc::polish(P, pts, base, base_evaluations, how, n_points, eq_tol, pop_size,
                                    budget, obs, record_every, fd_step, local_tol);
              });
              py::dict d;
              d["objectives"] = rows_to_list(r.objectives);
              d["variables"] = rows_to_list(r.variables);
              d["cv"] = to_list(r.cv);
              d["evaluations"] = r.spent;
              d["chosen"] = r.chosen;
              return d;
          },
          py::arg("fn"), py::arg("constraints"), py::arg("bounds"), py::arg("n_objs"),
          py::arg("n_cons"), py::arg("n_eq"), py::arg("eq_tol"), py::arg("pop_size"),
          py::arg("budget"), py::arg("points"), py::arg("base_variables"),
          py::arg("base_evaluations"), py::arg("how"), py::arg("n_points"),
          py::arg("on_generation") = py::none(), py::arg("record_every") = 0,
          py::arg("fd_step") = 1.4901161193847656e-08, py::arg("local_tol") = 1e-14,
          py::arg("native") = nullptr,
          "Polish an algorithm's result (polish.hpp): points = its evaluations as (x, f, g),\n"
          "base_variables = its answer's variables; a dict of the answer, the polishing's\n"
          "evaluations and the number of points chosen.");

    s.def("slsqp",
          [](const py::function& fun, const py::function& grad, const Vec& x0, const Vec& lo,
             const Vec& hi, int n_eq, int n_ineq, const py::object& cons, const py::object& cons_jac,
             double tol, int max_iter) {
              mootation::slsqp::NLP nlp;
              nlp.n = x0.size();
              nlp.n_eq = static_cast<std::size_t>(n_eq);
              nlp.n_ineq = static_cast<std::size_t>(n_ineq);
              nlp.lo = lo;
              nlp.hi = hi;
              long long calls = 0;
              nlp.values = [&](const Vec& x, Vec& c) {
                  ++calls;
                  py::list xs = to_list(x);
                  if (n_eq + n_ineq) c = floats(cons(xs));
                  return py::cast<double>(fun(xs));
              };
              nlp.derivatives = [&](const Vec& x, Vec& g, mootation::lsq::Matrix& A) {
                  py::list xs = to_list(x);
                  g = floats(grad(xs));
                  if (n_eq + n_ineq) {
                      A = mootation::lsq::Matrix(static_cast<std::size_t>(n_eq + n_ineq), x.size());
                      std::size_t i = 0;
                      for (py::handle row : cons_jac(xs)) {
                          Vec r = floats(row);
                          for (std::size_t j = 0; j < r.size(); ++j) A(i, j) = r[j];
                          ++i;
                      }
                  }
              };
              mootation::slsqp::Result r = mootation::slsqp::minimize(nlp, x0, {tol, max_iter});
              static const char* status[] = {"converged", "max_iter", "qp_failed",
                                             "line_search_failed", "no_progress"};
              py::dict d;
              d["x"] = to_list(r.x);
              d["f"] = r.f;
              d["c"] = to_list(r.c);
              d["status"] = status[static_cast<int>(r.status)];
              d["iterations"] = r.iterations;
              d["evaluations"] = calls;
              d["relaxed"] = r.relaxed;
              return d;
          },
          py::arg("fun"), py::arg("grad"), py::arg("x0"), py::arg("lo"), py::arg("hi"),
          py::arg("n_eq") = 0, py::arg("n_ineq") = 0, py::arg("cons") = py::none(),
          py::arg("cons_jac") = py::none(), py::arg("tol") = 1e-6, py::arg("max_iter") = 100,
          "min fun(x) s.t. cons(x)[:n_eq] = 0, cons(x)[n_eq:] >= 0, lo <= x <= hi (+-inf: no\n"
          "bound), by slsqp.hpp; grad and cons_jac give the derivatives.");

    s.def("nelder_mead",
          [](const py::function& fun, const Vec& x0, const Vec& lo, const Vec& hi, double tol,
             long long max_iter) {
              long long calls = 0;
              mootation::nelder_mead::Options o;
              o.tol = tol;
              o.max_iter = max_iter;
              auto r = mootation::nelder_mead::minimize(
                  [&](const Vec& x) {
                      ++calls;
                      return py::cast<double>(fun(to_list(x)));
                  },
                  x0, lo, hi, o);
              py::dict d;
              d["x"] = to_list(r.x);
              d["f"] = r.f;
              d["iterations"] = r.iterations;
              d["evaluations"] = calls;
              d["converged"] = r.converged;
              return d;
          },
          py::arg("fun"), py::arg("x0"), py::arg("lo"), py::arg("hi"), py::arg("tol") = 1e-8,
          py::arg("max_iter") = 1000000,
          "min fun(x) on the box [lo, hi] by nelder_mead.hpp.");

    // the residuals and the Jacobian of Python callables
    auto lm_problem = [](const py::object& fun, const py::object& jac, std::size_t n, std::size_t m) {
        sc::lm::Problem p;
        p.n = n;
        p.m = m;
        p.residuals = [fun, m](const Vec& x, Vec& r) {
            r = floats(fun(to_list(x)));
            if (r.size() != m) throw std::invalid_argument("lm: fun gave another number of residuals");
        };
        if (!jac.is_none())
            p.jacobian = [jac, n, m](const Vec& x, mootation::lsq::Matrix& J) {
                const py::object rows = jac(to_list(x));
                std::size_t i = 0;
                for (py::handle row : rows) {
                    const Vec v = floats(row);
                    if (i >= m || v.size() != n) throw std::invalid_argument("lm: jac is not m x n");
                    for (std::size_t j = 0; j < n; ++j) J(i, j) = v[j];
                    ++i;
                }
                if (i != m) throw std::invalid_argument("lm: jac is not m x n");
            };
        return p;
    };
    auto rows_of = [](const mootation::lsq::Matrix& A) {
        py::list l;
        for (std::size_t i = 0; i < A.rows; ++i) {
            py::list row;
            for (std::size_t j = 0; j < A.cols; ++j) row.append(A(i, j));
            l.append(row);
        }
        return l;
    };

    s.def("lm",
          [lm_problem, rows_of](const py::object& fun, const Vec& x0, std::size_t m, const py::object& jac,
                                double ftol, double xtol, double gtol, double factor, double fd_eps,
                                long long max_nfev, bool scale, bool geodesic, double geodesic_alpha,
                                const std::string& loss, double f_scale, bool diagnose, double sigma2,
                                const py::object& bounds) {
              sc::lm::Problem p = lm_problem(fun, jac, x0.size(), m);
              if (!bounds.is_none()) {
                  const py::sequence b = bounds.cast<py::sequence>();
                  p.lo = floats(b[0]);
                  p.hi = floats(b[1]);
              }
              sc::lm::Options o;
              o.ftol = ftol;
              o.xtol = xtol;
              o.gtol = gtol;
              o.factor = factor;
              o.fd_eps = fd_eps;
              o.max_nfev = max_nfev;
              o.scale = scale;
              o.geodesic = geodesic;
              o.geodesic_alpha = geodesic_alpha;
              if (loss == "linear") o.loss = sc::lm::Options::Loss::linear;
              else if (loss == "huber") o.loss = sc::lm::Options::Loss::huber;
              else if (loss == "cauchy") o.loss = sc::lm::Options::Loss::cauchy;
              else throw std::invalid_argument("lm: loss = linear, huber or cauchy");
              o.f_scale = f_scale;
              const sc::lm::Result r = sc::lm::solve_robust(p, x0, o);
              py::dict d;
              d["x"] = to_list(r.x);
              d["r"] = to_list(r.r);
              d["cost"] = r.cost;
              d["nfev"] = r.nfev;
              d["njev"] = r.njev;
              d["iterations"] = r.iterations;
              d["status"] = static_cast<int>(r.status);
              if (diagnose) {                     // J at the solution, then the diagnostics
                  mootation::lsq::Matrix J(m, x0.size());
                  if (p.jacobian) p.jacobian(r.x, J);
                  else sc::lm::fd_jacobian(p, r.x, r.r, fd_eps, J);
                  const sc::lm::Diagnostics g = sc::lm::diagnose(J, r.cost, sigma2, !p.jacobian, fd_eps);
                  d["singular_values"] = to_list(g.sv);
                  d["small"] = g.small;
                  d["sigma2"] = g.sigma2;
                  d["sigma2_given"] = g.sigma2_given;
                  d["covariance"] = rows_of(g.cov);
                  d["correlation"] = rows_of(g.corr);
              }
              return d;
          },
          py::arg("fun"), py::arg("x0"), py::arg("m"), py::arg("jac") = py::none(),
          py::arg("ftol") = 1e-8, py::arg("xtol") = 1e-8,
          py::arg("gtol") = 0.0, py::arg("factor") = 100.0, py::arg("fd_eps") = 0.0, py::arg("max_nfev") = 0,
          py::arg("scale") = true, py::arg("geodesic") = false, py::arg("geodesic_alpha") = 0.75,
          py::arg("loss") = "linear", py::arg("f_scale") = 1.0, py::arg("diagnose") = false,
          py::arg("sigma2") = 0.0, py::arg("bounds") = py::none(),
          "min 1/2 ||fun(x)||^2 (or sum rho(r_i), loss) by lm.hpp (Levenberg-Marquardt, More 1978); fun "
          "gives m residuals, jac their m x n Jacobian (None: forward differences). status: 0 ftol, 1 "
          "xtol, 2 both, 3 gtol, 4 max_nfev, 5-7 ftol, xtol, gtol at machine precision. diagnose: the "
          "singular values of J at the solution, the covariance s^2 (J^T J)^+ (s^2 = sigma2, or 2 cost / "
          "(m - n) when sigma2 = 0) and the correlations. bounds = (lo, hi), +-inf allowed: Coleman-Li's interior "
          "trust region (x0 inside).");

    s.def("lm_check_jacobian",
          [lm_problem](const py::object& fun, const py::object& jac, const Vec& x, std::size_t m, double fd_eps) {
              const sc::lm::JacobianCheck c = sc::lm::check_jacobian(lm_problem(fun, jac, x.size(), m), x, fd_eps);
              py::dict d;
              d["column_error"] = to_list(c.column_error);
              d["worst"] = py::cast(c.worst);
              return d;
          },
          py::arg("fun"), py::arg("jac"), py::arg("x"), py::arg("m"), py::arg("fd_eps") = 0.0,
          "jac(x) against forward differences of fun, column by column: the relative errors and the "
          "columns from the worst.");

    // varpro: y - Phi(alpha) a, basis(alpha) -> m x p rows, derivatives(alpha) -> k matrices m x p
    s.def("varpro",
          [](const py::object& basis, const Vec& alpha0, const Vec& y, std::size_t p,
                    const py::object& derivatives, bool kaufman, double ftol, double xtol, long long max_nfev,
                    const py::object& bounds) {
              const std::size_t m = y.size(), k = alpha0.size();
              auto fill = [m, p](const py::handle& rows, mootation::lsq::Matrix& A) {
                  std::size_t i = 0;
                  for (py::handle row : rows) {
                      const Vec v = floats(row);
                      if (i >= m || v.size() != p) throw std::invalid_argument("varpro: a matrix is not m x p");
                      for (std::size_t j = 0; j < p; ++j) A(i, j) = v[j];
                      ++i;
                  }
                  if (i != m) throw std::invalid_argument("varpro: a matrix is not m x p");
              };
              sc::lm::Separable sp;
              sp.m = m;
              sp.k = k;
              sp.p = p;
              sp.y = y;
              sp.basis = [basis, fill](const Vec& al, mootation::lsq::Matrix& Phi) { fill(basis(to_list(al)), Phi); };
              if (!derivatives.is_none())
                  sp.derivatives = [derivatives, fill, k](const Vec& al, std::vector<mootation::lsq::Matrix>& d) {
                      const py::object ms = derivatives(to_list(al));
                      std::size_t j = 0;
                      for (py::handle mat : ms) {
                          if (j >= k) throw std::invalid_argument("varpro: derivatives gives more than k matrices");
                          fill(mat, d[j++]);
                      }
                      if (j != k) throw std::invalid_argument("varpro: derivatives gives fewer than k matrices");
                  };
              if (!bounds.is_none()) {
                  const py::sequence b = bounds.cast<py::sequence>();
                  sp.lo = floats(b[0]);
                  sp.hi = floats(b[1]);
              }
              sc::lm::Options o;
              o.ftol = ftol;
              o.xtol = xtol;
              o.max_nfev = max_nfev;
              const sc::lm::SeparableResult r = sc::lm::varpro(sp, alpha0, o, kaufman);
              py::dict d;
              d["alpha"] = to_list(r.fit.x);
              d["a"] = to_list(r.a);
              d["r"] = to_list(r.fit.r);
              d["cost"] = r.fit.cost;
              d["nfev"] = r.fit.nfev;
              d["njev"] = r.fit.njev;
              d["status"] = static_cast<int>(r.fit.status);
              d["rank"] = r.rank;
              return d;
          },
          py::arg("basis"), py::arg("alpha0"), py::arg("y"), py::arg("p"), py::arg("derivatives") = py::none(),
          py::arg("kaufman") = false, py::arg("ftol") = 1e-8, py::arg("xtol") = 1e-8, py::arg("max_nfev") = 0,
          py::arg("bounds") = py::none(),
          "min 1/2 ||y - Phi(alpha) a||^2 by variable projection (Golub-Pereyra): lm.hpp on r_2(alpha) = "
          "P_Phi^perp y, a = Phi^- y. basis(alpha) gives Phi (m x p), derivatives(alpha) the k matrices "
          "dPhi/dalpha_j (None: differences of r_2); kaufman: Kaufman's Jacobian.");

    // the published test sets, for cross-checks: MGH 1981 and NIST StRD
    s.def("mgh",
          [](int id, std::size_t n, std::size_t m, const Vec& x) {
              Vec f, J;
              mootation::suite::mgh::evaluate(id, n, m, x, f, &J);
              py::list rows;
              for (std::size_t i = 0; i < m; ++i) rows.append(to_list(Vec(J.begin() + static_cast<std::ptrdiff_t>(i * n),
                                                                        J.begin() + static_cast<std::ptrdiff_t>((i + 1) * n))));
              return py::make_tuple(to_list(f), rows);
          },
          py::arg("id"), py::arg("n"), py::arg("m"), py::arg("x"),
          "MGH 1981 function id (1-35): (f, J) at x (suite/mgh.hpp).");
    s.def("mgh_start",
          [](int id, std::size_t n, double factor) { return to_list(mootation::suite::mgh::start(id, n, factor)); },
          py::arg("id"), py::arg("n"), py::arg("factor") = 1.0, "MGH's INITPT: factor * x_s.");
    s.def("nist",
          [](const std::string& path, const std::string& name) {
              const mootation::suite::nist::Dataset ds = mootation::suite::nist::read(path, name);
              py::dict d;
              d["n"] = ds.n;
              d["m"] = ds.y.size();
              d["start1"] = to_list(ds.start1);
              d["start2"] = to_list(ds.start2);
              d["certified"] = to_list(ds.certified);
              d["rss"] = ds.rss;
              d["residuals"] = py::cpp_function([ds](const Vec& b) {
                  Vec r;
                  mootation::suite::nist::residuals(ds, b, r);
                  return to_list(r);
              });
              return d;
          },
          py::arg("path"), py::arg("name"),
          "a NIST StRD file read by suite/nist.hpp: starts, certified values, residuals(b).");
}

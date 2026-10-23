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
}

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

#include <mootation/scalar/lbfgsb.hpp>
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
// A Python callable held so that copying the C++ function that holds it
// takes no Python reference: a strategy copies the problem's functions with
// the GIL released (scalar/strategy.hpp, View); the last copy lets go of it
// under the GIL.
std::shared_ptr<py::object> held(const py::object& f)
{
    return std::shared_ptr<py::object>(new py::object(f), [](py::object* q) {
        py::gil_scoped_acquire gil;
        delete q;
    });
}

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
        for (const auto& b : nat->blocks)                     // the problem's symmetries (task 19, §19.2.1)
            P.sym_blocks.push_back({static_cast<std::size_t>(b.offset), static_cast<std::size_t>(b.size),
                                    static_cast<std::size_t>(b.count)});
        if (nat->has_canon) P.canon = [nat](Vec& x) { nat->canon(x.data()); };
        return P;
    }
    const int m = n_objs, k = n_cons;
    if (fn.is_none()) return P;                 // residuals give the objective (derivatives_of)
    P.evaluate = [fn = held(fn), constraints = held(constraints), m, k](const Vec& x, Vec& f, Vec& g) {
        py::gil_scoped_acquire gil;
        py::list xs = to_list(x);
        Vec fv = floats((*fn)(xs));
        if (static_cast<int>(fv.size()) != m)
            throw std::invalid_argument("fn returned " + std::to_string(fv.size()) + " values for " +
                                        std::to_string(m) + " objectives");
        f = fv;
        if (k > 0) {
            Vec gv = floats((*constraints)(xs));
            if (static_cast<int>(gv.size()) != k)
                throw std::invalid_argument("constraints returned " + std::to_string(gv.size()) +
                                            " values for " + std::to_string(k));
            g = gv;
        }
    };
    return P;
}

sc::Rows rows_of_seq(const py::handle& seq)
{
    sc::Rows out;
    for (py::handle r : seq) out.push_back(floats(r));
    return out;
}

// The problem's derivatives and residuals (task 18, stage 1, §1.1), Python
// functions called with the GIL: grad(x) the objective's gradient, or the
// pair (gradient, the constraints' Jacobian) for a constrained problem;
// residuals(x) the residuals, jac(x) their Jacobian (rows), n_res from the
// first call.
void derivatives_of(sc::Problem& P, const py::object& grad, const py::object& residuals, const py::object& jac)
{
    if (!residuals.is_none()) {
        P.residuals = [residuals = held(residuals)](const Vec& x, Vec& r) {
            py::gil_scoped_acquire gil;
            r = floats((*residuals)(to_list(x)));
        };
        if (!jac.is_none())
            P.jacobian = [jac = held(jac)](const Vec& x, sc::Rows& J) {
                py::gil_scoped_acquire gil;
                J = rows_of_seq((*jac)(to_list(x)));
            };
    } else if (!jac.is_none()) {
        throw std::invalid_argument("jac is the residuals' Jacobian: it goes with residuals");
    }
    if (!grad.is_none()) {
        if (P.has_residuals()) throw std::invalid_argument("grad: a problem with residuals has jac for its derivatives");
        const int k = P.n_cons;
        P.gradient = [grad = held(grad), k](const Vec& x, Vec& gf, sc::Rows& JG) {
            py::gil_scoped_acquire gil;
            py::object out = (*grad)(to_list(x));
            if (k > 0) {
                py::sequence pair = out.cast<py::sequence>();
                if (pair.size() != 2)
                    throw std::invalid_argument("grad of a constrained problem returns (gradient, constraints' Jacobian)");
                gf = floats(pair[0]);
                JG = rows_of_seq(pair[1]);
            } else {
                gf = floats(out);
            }
        };
    }
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

    s.def(
        "defer",
        [](const py::function& what) {
            if (sc::strategy::deferred_here() == nullptr) return false;
            sc::strategy::defer_or_run([what = held(what)] {
                py::gil_scoped_acquire gil;
                (*what)();
            });
            return true;
        },
        py::arg("what"),
        "THE DEFERRED COUNT (strategy.hpp): in a thread that runs a strategy's start (a round in\n"
        "threads > 1), hold what(), to be called at the round's end in the order of the starts,\n"
        "and return True; elsewhere return False (the caller does it at once).");

    s.def("run",
          [](const std::string& name, const py::object& fn, const py::object& constraints,
             const py::object& bounds, int n_objs, int n_cons, int n_eq, double eq_tol,
             const py::object& eq_tol0, int pop_size, long long budget, unsigned seed,
             const py::dict& knobs, const py::object& on_generation, int record_every,
             const py::object& design, const py::object& seed_points,
             const std::shared_ptr<mootation::suite::Native>& native, const py::object& grad,
             const py::object& residuals, const py::object& jac, const py::object& x0,
             const py::object& gauge_basis, const py::object& f_stop, const py::object& sym_blocks,
             const py::object& canon, const py::object& align, const py::object& res_blocks,
             const py::object& res_weights) {
              if (fn.is_none() == residuals.is_none() && !native)
                  throw std::invalid_argument("give fn or residuals, one of them");
              sc::Problem P = problem_of(fn, constraints, bounds, n_objs, n_cons, n_eq, native);
              derivatives_of(P, grad, residuals, jac);
              if (!gauge_basis.is_none())            // n rows of k (task 19, §19.3.1)
                  P.gauge_basis = [gauge_basis = held(gauge_basis)](const Vec& x, sc::Rows& N) {
                      py::gil_scoped_acquire gil;
                      N = rows_of_seq((*gauge_basis)(to_list(x)));
                  };
              if (!f_stop.is_none()) P.stop_at = py::cast<double>(f_stop);
              if (!sym_blocks.is_none()) {           // (offset, size, count) each (task 19, §19.2.1)
                  P.sym_blocks.clear();
                  for (py::handle b : sym_blocks) {
                      const py::sequence t = py::reinterpret_borrow<py::sequence>(b);
                      if (t.size() != 3) throw std::invalid_argument("sym_blocks: (offset, size, count) each");
                      P.sym_blocks.push_back({py::cast<std::size_t>(t[0]), py::cast<std::size_t>(t[1]),
                                              py::cast<std::size_t>(t[2])});
                  }
              }
              if (!res_blocks.is_none()) {           // the residuals' blocks and weights (contract 1a; task 19, §19.6)
                  P.blocks = res_blocks.cast<std::vector<int>>();
                  P.weights = res_weights.is_none() ? Vec(P.blocks.size(), 1.0) : floats(res_weights);
                  P.n_res = 0;
                  for (int b : P.blocks) P.n_res += b;
              } else if (!res_weights.is_none()) {
                  throw std::invalid_argument("res_weights go with res_blocks");
              }
              if (!canon.is_none())
                  P.canon = [canon = held(canon)](Vec& x) {
                      py::gil_scoped_acquire gil;
                      x = floats((*canon)(to_list(x)));
                  };
              if (!align.is_none())
                  P.align = [align = held(align)](const Vec& x, const Vec& ref, Vec& out) {
                      py::gil_scoped_acquire gil;
                      out = floats((*align)(to_list(x), to_list(ref)));
                  };
              sc::Start st;
              if (!x0.is_none()) {
                  st.has_x0 = true;
                  st.x0 = floats(x0);
              }
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
              // a strategy's starts may run in threads, each calling fn under the GIL (strategy.hpp)
              const bool strategy = std::find(sc::strategy_names().begin(), sc::strategy_names().end(), name) !=
                                    sc::strategy_names().end();
              sc::Result r = maybe_released(native != nullptr || strategy, [&]() {
                  return sc::run(name, P, eq_tol, has0, tol0, pop_size, budget, seed, kn, obs,
                                 record_every, st);
              });
              py::dict d;
              d["objectives"] = rows_to_list(r.objectives);
              d["variables"] = rows_to_list(r.variables);
              d["cv"] = to_list(r.cv);
              d["ignored"] = py::cast(r.ignored);
              d["evaluations"] = r.evaluations;
              d["grad_calls"] = r.grad_calls;
              d["n_failed"] = r.n_failed;
              d["settings"] = py::cast(r.settings);
              d["solutions"] = rows_to_list(r.solutions);
              py::list imp;
              for (const auto& e : r.improvements)
                  imp.append(py::make_tuple(e.evaluations, e.grad_calls, e.f, e.cv));
              d["improvements"] = imp;
              py::dict rep;                                     // task 20, §20.2.2
              rep["checked"] = r.repairs.checked;
              rep["out_children"] = r.repairs.out_children;
              rep["out_vars"] = r.repairs.out_vars;
              d["repairs"] = rep;
              return d;
          },
          py::arg("name"), py::arg("fn"), py::arg("constraints"), py::arg("bounds"),
          py::arg("n_objs"), py::arg("n_cons"), py::arg("n_eq"), py::arg("eq_tol"),
          py::arg("eq_tol0"), py::arg("pop_size"), py::arg("budget"), py::arg("seed"),
          py::arg("knobs"), py::arg("on_generation") = py::none(), py::arg("record_every") = 0,
          py::arg("design") = py::none(), py::arg("seed_points") = py::none(),
          py::arg("native") = nullptr, py::arg("grad") = py::none(), py::arg("residuals") = py::none(),
          py::arg("jac") = py::none(), py::arg("x0") = py::none(), py::arg("gauge_basis") = py::none(),
          py::arg("f_stop") = py::none(),
          py::arg("sym_blocks") = py::none(), py::arg("canon") = py::none(), py::arg("align") = py::none(),
          py::arg("res_blocks") = py::none(), py::arg("res_weights") = py::none(),
          "Run one of NAMES; a dict of the answer (objectives, variables, cv), evaluations,\n"
          "grad_calls, n_failed (failed evaluations in the archive), settings, ignored knobs and,\n"
          "for the scalar presets, solutions; at one objective improvements, the best point's\n"
          "records (evaluations, grad_calls, f, cv; improvements.hpp). fn = None with residuals\n"
          "(and jac, their Jacobian): the objective 1/2 ||r||^2; grad: the objective's gradient;\n"
          "x0: the start\n"
          "(evaluator.hpp, scalar.hpp run_local). sym_blocks, canon, align: the problem's\n"
          "symmetries for the population methods' sym knob (symmetry.hpp; a native problem's own\n"
          "where not given).");

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
          "xtol, 2 both, 3 gtol, 4 max_nfev, 5-7 ftol, xtol, gtol at machine precision, 8 residuals or "
          "a Jacobian not finite (LM-9). diagnose: the "
          "singular values of J at the solution, the covariance s^2 (J^T J)^+ (s^2 = sigma2, or 2 cost / "
          "(m - n) when sigma2 = 0) and the correlations. bounds = (lo, hi), +-inf allowed: Coleman-Li's interior "
          "trust region (x0 inside).");

    // task 19, §19.2.1 (symmetry.hpp): the C++ side of mootation.symmetry's cross-check
    auto groups_of = [](const py::object& blocks) {
        std::vector<sc::symmetry::Blocks> g;
        for (py::handle b : blocks) {
            const py::sequence t = py::reinterpret_borrow<py::sequence>(b);
            if (t.size() != 3) throw std::invalid_argument("blocks: (offset, size, count) each");
            g.push_back({py::cast<std::size_t>(t[0]), py::cast<std::size_t>(t[1]), py::cast<std::size_t>(t[2])});
        }
        return g;
    };
    s.def("symmetry_hungarian",
          [](const py::object& c) {
              const sc::Rows rows = rows_of_seq(c);
              Vec flat;
              for (const Vec& r : rows) flat.insert(flat.end(), r.begin(), r.end());
              return sc::symmetry::hungarian(rows.size(), flat);
          },
          py::arg("c"), "The least-cost assignment of the square matrix c: the column of each row (Munkres 1957).");
    s.def("symmetry_align",
          [groups_of](const Vec& x, const Vec& ref, const py::object& blocks) {
              const auto g = groups_of(blocks);
              sc::symmetry::check(g, x.size());
              return sc::symmetry::align(x, ref, g);
          },
          py::arg("x"), py::arg("ref"), py::arg("blocks"),
          "x with each group's blocks permuted to ref's at least squared distance (symmetry.hpp).");
    s.def("symmetry_canon",
          [groups_of](Vec x, const py::object& blocks) {
              const auto g = groups_of(blocks);
              sc::symmetry::check(g, x.size());
              sc::symmetry::sort_blocks(x, g);
              return x;
          },
          py::arg("x"), py::arg("blocks"), "x with each group's blocks sorted (the built-in canon, SYM-5).");

    s.def("lm_profile",
          [lm_problem](const py::object& fun, const py::object& jac, const Vec& x_hat, std::size_t m, std::size_t j,
                       const py::object& bounds, const py::object& grid, double alpha, int df, double sigma2,
                       int max_steps) {
              sc::lm::Problem p = lm_problem(fun, jac, x_hat.size(), m);
              if (!bounds.is_none()) {
                  const py::sequence b = bounds.cast<py::sequence>();
                  p.lo = floats(b[0]);
                  p.hi = floats(b[1]);
              }
              sc::lm::ProfileOptions po;
              if (!grid.is_none()) po.grid = floats(grid);
              po.alpha = alpha;
              po.df = df;
              po.sigma2 = sigma2;
              po.max_steps = max_steps;
              const sc::lm::Profile pr = sc::lm::profile(p, x_hat, j, sc::lm::Options{}, po);
              py::list values, costs, xs;
              for (const auto& pt : pr.points) {
                  values.append(pt.value);
                  costs.append(pt.cost);
                  xs.append(to_list(pt.x));
              }
              py::dict d;
              d["j"] = pr.j;
              d["values"] = values;
              d["costs"] = costs;
              d["x"] = xs;
              d["cost_hat"] = pr.cost_hat;
              d["threshold"] = pr.threshold;
              d["lower"] = pr.lower;
              d["upper"] = pr.upper;
              d["relative"] = pr.relative;
              d["kind"] = pr.kind == sc::lm::Profile::Kind::identifiable ? "identifiable"
                          : pr.kind == sc::lm::Profile::Kind::practically_non_identifiable
                              ? "practically_non_identifiable"
                              : "structurally_non_identifiable";
              return d;
          },
          py::arg("fun"), py::arg("jac"), py::arg("x_hat"), py::arg("m"), py::arg("j"),
          py::arg("bounds") = py::none(), py::arg("grid") = py::none(), py::arg("alpha") = 0.95,
          py::arg("df") = 1, py::arg("sigma2") = 0.0, py::arg("max_steps") = 50,
          "The profile of x_j about x_hat (lm.hpp, profile; Raue et al. 2009): the values, their least "
          "costs and points, the threshold, the interval [lower, upper] and the kind.");

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

    s.def("lbfgsb",
          [](const py::object& fun, const Vec& x0, const py::object& grad, const py::object& bounds, int m,
             double pgtol, double ftol, long long max_nfev, long long max_iter) {
              sc::lbfgsb::Problem p;
              p.n = x0.size();
              p.f = [fun](const Vec& x) { return fun(to_list(x)).cast<double>(); };
              if (!grad.is_none())
                  p.fg = [fun, grad](const Vec& x, Vec& g) {
                      g = floats(grad(to_list(x)));
                      return fun(to_list(x)).cast<double>();
                  };
              if (!bounds.is_none()) {
                  const py::sequence b = bounds.cast<py::sequence>();
                  p.lo = floats(b[0]);
                  p.hi = floats(b[1]);
              }
              sc::lbfgsb::Options o;
              o.m = m;
              o.pgtol = pgtol;
              o.ftol = ftol;
              o.max_nfev = max_nfev;
              o.max_iter = max_iter;
              const sc::lbfgsb::Result r = sc::lbfgsb::solve(p, x0, o);
              py::dict d;
              d["x"] = to_list(r.x);
              d["f"] = r.f;
              d["g"] = to_list(r.g);
              d["nfev"] = r.nfev;
              d["ngev"] = r.ngev;
              d["iterations"] = r.iterations;
              d["status"] = static_cast<int>(r.status);
              return d;
          },
          py::arg("fun"), py::arg("x0"), py::arg("grad") = py::none(), py::arg("bounds") = py::none(),
          py::arg("m") = 5, py::arg("pgtol") = 1e-5, py::arg("ftol") = 0.0, py::arg("max_nfev") = 15000,
          py::arg("max_iter") = 15000,
          "min fun(x) on lo <= x <= hi (+-inf allowed) by lbfgsb.hpp (Byrd-Lu-Nocedal-Zhu 1995, Morales-Nocedal "
          "2011, More-Thuente search); grad(x) the gradient (None: forward differences). status: 0 pgtol, 1 "
          "ftol, 2 line search, 3 max_nfev, 4 max_iter.");

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

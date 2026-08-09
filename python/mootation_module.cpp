// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// pybind11 module `mootation`.
//
// The algorithm table is NOT written out here. It is generated from
// include/mootation/algorithms.def — the same X-macro the test suite iterates over — so a
// new algorithm becomes callable from Python by adding the one line it already
// needs, and the binding cannot silently fall behind the library. CI fails if a
// header is missing from that file.
//
// Every core exposes set_seed/setup/step, but the tuning knobs differ per
// algorithm: only 19 of the 60 have set_t_max, 10 have set_T, 7 have set_kappa.
// Rather than a per-algorithm dispatch, each optional setter is detected with
// SFINAE and applied only where it exists. A knob the caller sets that the
// chosen algorithm does not have is REPORTED, not ignored — silently dropping
// it is how a run ends up not being the run you configured.
//
// Objectives are evaluated by a Python callback. That crosses the GIL once per
// batch, so use `evaluate_batch` for anything non-trivial.
// ============================================================================

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mootation/mootation.hpp>

namespace py = pybind11;

namespace {

// ── The problem, as configured from Python ─────────────────────────────────
struct PyProblem {
    std::vector<std::pair<double, double>> bounds;
    int    n_objectives = 2;
    int    n_limits     = 0;
    // One of the two. evaluate_batch is preferred: it crosses the GIL once per
    // generation instead of once per individual.
    std::function<std::vector<double>(std::vector<double>)>               evaluate;
    std::function<std::vector<std::vector<double>>(
        std::vector<std::vector<double>>)>                                evaluate_batch;
    // Optional; must return one row of limits per individual, same convention
    // as the C++ side: <= 0 satisfied, > 0 is the violation.
    std::function<std::vector<std::vector<double>>(
        std::vector<std::vector<double>>)>                                limits_batch;
};

// The active problem for the current run. A raw pointer is enough because a run
// is single-threaded from Python's side and the pointer outlives the vault.
PyProblem* g_problem = nullptr;

// Objective/limit evaluation, shared by every generated Problem<> below.
inline void eval_into(const std::vector<double>& vars,
                      std::vector<double>&       objs,
                      std::vector<double>&       lims)
{
    if (!g_problem) throw std::runtime_error("mootation: no active problem");
    py::gil_scoped_acquire gil;

    if (g_problem->evaluate_batch) {
        auto out = g_problem->evaluate_batch({vars});
        if (out.empty()) throw std::runtime_error("evaluate_batch returned nothing");
        objs = out.front();
    } else if (g_problem->evaluate) {
        objs = g_problem->evaluate(vars);
    } else {
        throw std::runtime_error("set problem.evaluate or problem.evaluate_batch");
    }
    if (static_cast<int>(objs.size()) != g_problem->n_objectives)
        throw std::runtime_error("the evaluator returned the wrong number of objectives");

    if (g_problem->n_limits > 0) {
        if (!g_problem->limits_batch)
            throw std::runtime_error("n_limits > 0 requires problem.limits_batch");
        auto lim = g_problem->limits_batch({vars});
        if (lim.empty()) throw std::runtime_error("limits_batch returned nothing");
        lims = lim.front();
        if (static_cast<int>(lims.size()) != g_problem->n_limits)
            throw std::runtime_error("limits_batch returned the wrong width");
    }
}

} // namespace

// Each algorithm needs ITS OWN individual type — crEA reads .cluster/.rank,
// NSGA-II reads .rank/.crowding_distance — so one tag per algorithm is
// generated from the same X-macro that drives the dispatch. The Problem<>
// specialization has to be written out in full: Based_Individual befriends
// Problem<T> only, so nothing else may touch ind.variables/objectives/limits.
#define MOOTATION_PY_PROBLEM(TAG, BASE)                                        \
    namespace { struct TAG : public mootation::BASE {}; }                      \
    namespace mootation {                                                      \
    template <>                                                                \
    class Problem<TAG> {                                                       \
    public:                                                                    \
        std::vector<std::pair<std::optional<double>, std::optional<double>>>   \
            bounds;                                                            \
        int get_vars_n() const { return static_cast<int>(bounds.size()); }     \
        int get_bin_vars_n() const { return 0; }                               \
        int get_objs_n() const                                                 \
        {                                                                      \
            return g_problem ? g_problem->n_objectives : 0;                    \
        }                                                                      \
        int get_lims_n() const                                                 \
        {                                                                      \
            return g_problem ? g_problem->n_limits : 0;                        \
        }                                                                      \
        void calc_objs(TAG& ind) const                                         \
        {                                                                      \
            eval_into(ind.variables, ind.objectives, ind.limits);              \
        }                                                                      \
    };                                                                         \
    }

#define MOOTATION_ALG(KEY, IND, CORE) MOOTATION_PY_PROBLEM(PyTag_##KEY, IND)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

namespace {

// ── Optional-setter detection ───────────────────────────────────────────────
// One macro per knob: a trait that asks "does this core have set_X(T)?", and an
// apply() that calls it when it does and reports when it does not.
#define MOOTATION_OPTIONAL_SETTER(NAME, SETTER, TYPE)                          \
    template <typename C, typename = void>                                     \
    struct has_##NAME : std::false_type {};                                    \
    template <typename C>                                                      \
    struct has_##NAME<C, std::void_t<decltype(std::declval<C&>().SETTER(       \
                             std::declval<TYPE>()))>> : std::true_type {};     \
    template <typename C>                                                      \
    bool apply_##NAME(C& core, TYPE v)                                         \
    {                                                                          \
        if constexpr (has_##NAME<C>::value) { core.SETTER(v); return true; }    \
        else { (void)core; (void)v; return false; }                            \
    }

MOOTATION_OPTIONAL_SETTER(t_max,      set_t_max,             int)
MOOTATION_OPTIONAL_SETTER(eta_c,      set_eta_crossover,     double)
MOOTATION_OPTIONAL_SETTER(eta_m,      set_eta_mutation,      double)
MOOTATION_OPTIONAL_SETTER(pc,         set_pc,                double)
MOOTATION_OPTIONAL_SETTER(pm,         set_pm,                double)
MOOTATION_OPTIONAL_SETTER(T,          set_T,                 int)
MOOTATION_OPTIONAL_SETTER(delta,      set_delta,             double)
MOOTATION_OPTIONAL_SETTER(nr,         set_nr,                int)
MOOTATION_OPTIONAL_SETTER(kappa,      set_kappa,             double)
MOOTATION_OPTIONAL_SETTER(K,          set_K,                 int)
MOOTATION_OPTIONAL_SETTER(n_clusters, set_n_clusters,        int)
MOOTATION_OPTIONAL_SETTER(theta,      set_theta,             double)
MOOTATION_OPTIONAL_SETTER(alpha,      set_alpha,             double)
MOOTATION_OPTIONAL_SETTER(F,          set_F,                 double)
MOOTATION_OPTIONAL_SETTER(CR,         set_CR,                double)
MOOTATION_OPTIONAL_SETTER(div,        set_div,               int)

#undef MOOTATION_OPTIONAL_SETTER

// ── Run configuration ───────────────────────────────────────────────────────
// NOT named PyConfig: CPython's own Python.h defines that (PEP 587).
// Every knob is optional. `has_*` is false until the caller assigns, so an
// unset knob leaves the algorithm's paper default alone.
struct RunConfig {
    int      pop_size = 100;
    int      n_gen    = 250;
    unsigned seed     = 0;
    mootation::ConstraintMode constraint_mode = mootation::ConstraintMode::NONE;

    std::optional<double> eta_c, eta_m, pc, pm, delta, kappa, theta, alpha, F, CR;
    std::optional<int>    T, nr, K, n_clusters, div;

    // Warm start. Empty means a fresh random population. Only vars and objs
    // are carried, because those are the fields every algorithm shares — which
    // is what lets a population saved by NSGA-II seed a MOEA/D run.
    std::vector<std::vector<double>> seed_variables;
    std::vector<std::vector<double>> seed_objectives;
};

struct PyResult {
    std::vector<std::vector<double>> objectives;
    std::vector<std::vector<double>> variables;
    std::vector<double>              cv;
    std::size_t                      active_n = 0;
    // Knobs the caller set that this algorithm does not have. Reported rather
    // than dropped: a run configured with an ignored parameter is not the run
    // the caller asked for.
    std::vector<std::string>         ignored;
};

template <typename Tag, typename Core>
PyResult run_core(const RunConfig& cfg)
{
    using namespace mootation;

    Problem<Tag> prob;
    prob.bounds.reserve(g_problem->bounds.size());
    for (auto& b : g_problem->bounds)
        prob.bounds.emplace_back(std::optional<double>(b.first),
                                 std::optional<double>(b.second));

    DataVault<Tag>      vault(cfg.pop_size, prob);
    Optimizer<Tag, Core> opt(std::move(vault), defer_setup);

    auto&                    alg = opt.get_algorithm();
    std::vector<std::string> ignored;
    auto note = [&](bool applied, const char* name) {
        if (!applied) ignored.emplace_back(name);
    };

    alg.set_seed(cfg.seed);
    alg.constraint_mode = cfg.constraint_mode;

    // t_max is not a user knob but a schedule input: several algorithms anneal
    // against it, and a default of 1000 against a 250-generation run is not the
    // paper's schedule. Always pass the real budget where the core takes it.
    apply_t_max(alg, cfg.n_gen);

    if (cfg.eta_c)      note(apply_eta_c(alg, *cfg.eta_c),           "eta_c");
    if (cfg.eta_m)      note(apply_eta_m(alg, *cfg.eta_m),           "eta_m");
    if (cfg.pc)         note(apply_pc(alg, *cfg.pc),                 "pc");
    if (cfg.pm)         note(apply_pm(alg, *cfg.pm),                 "pm");
    if (cfg.T)          note(apply_T(alg, *cfg.T),                   "T");
    if (cfg.delta)      note(apply_delta(alg, *cfg.delta),           "delta");
    if (cfg.nr)         note(apply_nr(alg, *cfg.nr),                 "nr");
    if (cfg.kappa)      note(apply_kappa(alg, *cfg.kappa),           "kappa");
    if (cfg.K)          note(apply_K(alg, *cfg.K),                   "K");
    if (cfg.n_clusters) note(apply_n_clusters(alg, *cfg.n_clusters), "n_clusters");
    if (cfg.theta)      note(apply_theta(alg, *cfg.theta),           "theta");
    if (cfg.alpha)      note(apply_alpha(alg, *cfg.alpha),           "alpha");
    if (cfg.F)          note(apply_F(alg, *cfg.F),                   "F");
    if (cfg.CR)         note(apply_CR(alg, *cfg.CR),                 "CR");
    if (cfg.div)        note(apply_div(alg, *cfg.div),               "div");

    {
        py::gil_scoped_release unlock;   // the callback re-acquires per batch
        if (!cfg.seed_variables.empty()) {
            // setup_with_seed plants the population WITHOUT evaluating it: the
            // objectives come from the file, so a warm start costs zero
            // function evaluations. That is the whole point when an evaluation
            // is a solver run.
            opt.setup_with_seed(cfg.seed_variables, cfg.seed_objectives);
        } else {
            opt.setup();
        }
        opt.optimize(cfg.n_gen);
    }

    auto&    v = opt.get_vault();
    PyResult r;
    // NOT active_n(): steady-state cores (moead_de, moead_dd, hlmea, adaw,
    // ...) park a persistent scratch slot at active index pop_size(), so
    // active_n() is pop_size() + 1 for them. Returning that extra slot hands
    // the caller an unselected offspring dressed as a member of the answer
    // set — enough to skew any IGD or hypervolume computed from it, and
    // invisible unless you count the rows. See data_vault.hpp, "SCRATCH SLOTS
    // AND CONSUMERS".
    r.active_n = std::min<std::size_t>(v.active_n(),
                                       static_cast<std::size_t>(v.pop_size()));
    r.ignored  = std::move(ignored);
    r.objectives.reserve(r.active_n);
    r.variables.reserve(r.active_n);
    for (std::size_t i = 0; i < r.active_n; ++i) {
        r.objectives.push_back(v.objectives_of(i));
        r.variables.push_back(v.variables_of(i));
        r.cv.push_back(v.get_cv(i));
    }
    return r;
}

// ── Dispatch, generated from include/mootation/algorithms.def ───────────────────────────
PyResult run(const std::string& name, PyProblem& problem, const RunConfig& cfg)
{
    if (problem.bounds.empty())
        throw std::invalid_argument("problem.bounds is empty");

    g_problem = &problem;
    struct Guard { ~Guard() { g_problem = nullptr; } } guard;

#define MOOTATION_ALG(KEY, IND, CORE) \
    if (name == #KEY) return run_core<PyTag_##KEY, mootation::CORE<PyTag_##KEY>>(cfg);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

    throw std::invalid_argument("unknown algorithm: " + name);
}

std::vector<std::string> algorithm_names()
{
    std::vector<std::string> v;
#define MOOTATION_ALG(KEY, IND, CORE) v.emplace_back(#KEY);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
    return v;
}

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "MOOtation — multi- and many-objective evolutionary algorithms";
    m.attr("__version__") = mootation::version();

    py::enum_<mootation::ConstraintMode>(m, "ConstraintMode")
        .value("NONE",           mootation::ConstraintMode::NONE)
        .value("FEASIBILITY",    mootation::ConstraintMode::FEASIBILITY)
        .value("EPS_CONSTRAINT", mootation::ConstraintMode::EPS_CONSTRAINT)
        .value("CDP",            mootation::ConstraintMode::CDP);

    py::class_<PyProblem>(m, "Problem")
        .def(py::init<>())
        .def_readwrite("bounds",         &PyProblem::bounds)
        .def_readwrite("n_objectives",   &PyProblem::n_objectives)
        .def_readwrite("n_limits",       &PyProblem::n_limits)
        .def_readwrite("evaluate",       &PyProblem::evaluate)
        .def_readwrite("evaluate_batch", &PyProblem::evaluate_batch)
        .def_readwrite("limits_batch",   &PyProblem::limits_batch);

    py::class_<RunConfig>(m, "Config")
        .def(py::init<>())
        .def_readwrite("pop_size",        &RunConfig::pop_size)
        .def_readwrite("n_gen",           &RunConfig::n_gen)
        .def_readwrite("seed",            &RunConfig::seed)
        .def_readwrite("constraint_mode", &RunConfig::constraint_mode)
        .def_readwrite("eta_c",           &RunConfig::eta_c)
        .def_readwrite("eta_m",           &RunConfig::eta_m)
        .def_readwrite("pc",              &RunConfig::pc)
        .def_readwrite("pm",              &RunConfig::pm)
        .def_readwrite("T",               &RunConfig::T)
        .def_readwrite("delta",           &RunConfig::delta)
        .def_readwrite("nr",              &RunConfig::nr)
        .def_readwrite("kappa",           &RunConfig::kappa)
        .def_readwrite("K",               &RunConfig::K)
        .def_readwrite("n_clusters",      &RunConfig::n_clusters)
        .def_readwrite("theta",           &RunConfig::theta)
        .def_readwrite("alpha",           &RunConfig::alpha)
        .def_readwrite("F",               &RunConfig::F)
        .def_readwrite("CR",              &RunConfig::CR)
        .def_readwrite("div",             &RunConfig::div)
        .def_readwrite("seed_variables",  &RunConfig::seed_variables)
        .def_readwrite("seed_objectives", &RunConfig::seed_objectives);

    py::class_<PyResult>(m, "Result")
        .def_readonly("objectives", &PyResult::objectives)
        .def_readonly("variables",  &PyResult::variables)
        .def_readonly("cv",         &PyResult::cv)
        .def_readonly("active_n",   &PyResult::active_n)
        .def_readonly("ignored",    &PyResult::ignored,
                      "Knobs you set that this algorithm does not have.");

    m.def("algorithms", &algorithm_names,
          "Names accepted by run(), straight from include/mootation/algorithms.def.");
    m.def("run", &run, py::arg("name"), py::arg("problem"), py::arg("config"),
          "Run one algorithm and return its final population.");
}

// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The C++ suite of benchmark problems (include/mootation/suite; task 11 of
// 2026-10-18, §1), part of the module `mootation._core` (bind_suite, called
// from mootation_module.cpp): _core.Native(spec), the problem a spec of
// benchmarks/native.py describes, with the evaluate, evaluate_batch,
// constraints_batch and evaluate_with_constraints of a BenchProblem; the
// loops over a batch run here, without the GIL. A run hands the object to the
// core (Problem.native), which then evaluates it without calling Python.
//
// This file alone compiles the suite's arithmetic, without contraction into
// fused multiply-adds (python/CMakeLists.txt): the values are NumPy's bit for
// bit (suite/common.hpp).
// ============================================================================

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mootation/suite/suite.hpp>

namespace py = pybind11;
namespace suite = mootation::suite;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> flat(const py::handle& o)
{
    Arr a = Arr::ensure(o);
    if (!a) throw std::invalid_argument("suite spec: an array of numbers was expected");
    return std::vector<double>(a.data(), a.data() + a.size());
}

std::vector<double> flat_of(const py::dict& d, const char* key)
{
    return d.contains(key) && !d[key].is_none() ? flat(d[key]) : std::vector<double>{};
}

std::vector<std::vector<double>> list_of(const py::dict& d, const char* key)
{
    std::vector<std::vector<double>> out;
    if (!d.contains(key) || d[key].is_none()) return out;
    for (const auto& item : d[key]) out.push_back(flat(item));
    return out;
}

// one bbob function of a bbob-biobj spec (benchmarks/native.py, _bbob)
suite::bbob::Function bbob_of(const py::handle& o)
{
    const py::dict d = py::reinterpret_borrow<py::dict>(o);
    suite::bbob::Function f;
    f.fid = d["fid"].cast<int>();
    f.x_opt = flat_of(d, "x_opt");
    f.f_opt = d["f_opt"].cast<double>();
    f.R = flat_of(d, "R");
    f.Q = flat_of(d, "Q");
    f.l10 = flat_of(d, "l10");
    f.sign = flat_of(d, "sign");
    f.w = flat_of(d, "w");
    f.C = flat_of(d, "C");
    f.y = flat_of(d, "y");
    f.levels = list_of(d, "levels");
    return f;
}

py::dict dict_of(const suite::bbob::Function& f)
{
    py::dict d;
    d["fid"] = f.fid;
    d["x_opt"] = f.x_opt;
    d["f_opt"] = f.f_opt;
    d["R"] = f.R;
    d["Q"] = f.Q;
    d["l10"] = f.l10;
    d["sign"] = f.sign;
    d["w"] = f.w;
    d["C"] = f.C;
    d["y"] = f.y;
    d["levels"] = f.levels;
    return d;
}

suite::Spec spec_of(const py::dict& d)
{
    suite::Spec s;
    s.family = d["family"].cast<std::string>();
    s.n_var = d["n_var"].cast<int>();
    s.n_obj = d["n_obj"].cast<int>();
    if (d.contains("map")) s.map = d["map"].cast<std::string>();
    if (d.contains("first")) s.first = d["first"].cast<int>();
    s.centre = flat_of(d, "centre");
    s.spread = flat_of(d, "spread");
    s.matrices = list_of(d, "matrices");
    s.scale = flat_of(d, "scale");
    if (d.contains("r2")) s.r2 = d["r2"].cast<double>();
    if (d.contains("centre_m")) s.centre_m = d["centre_m"].cast<double>();
    s.lines = flat_of(d, "lines");
    s.polygons = list_of(d, "polygons");
    s.vertices = flat_of(d, "vertices");
    s.link = flat_of(d, "link");
    s.parts = flat_of(d, "parts");
    if (d.contains("k")) s.k = d["k"].cast<int>();
    if (d.contains("bbob") && !d["bbob"].is_none())
        for (const auto& item : d["bbob"]) s.bbob.push_back(bbob_of(item));
    return s;
}

py::dict dict_of(const suite::Spec& s)
{
    py::dict d;
    d["family"] = s.family;
    d["n_var"] = s.n_var;
    d["n_obj"] = s.n_obj;
    d["map"] = s.map;
    d["first"] = s.first;
    d["centre"] = s.centre;
    d["spread"] = s.spread;
    d["matrices"] = s.matrices;
    d["scale"] = s.scale;
    d["r2"] = s.r2;
    d["centre_m"] = s.centre_m;
    d["lines"] = s.lines;
    d["polygons"] = s.polygons;
    d["vertices"] = s.vertices;
    d["link"] = s.link;
    d["parts"] = s.parts;
    d["k"] = s.k;
    py::list fs;
    for (const auto& f : s.bbob) fs.append(dict_of(f));
    d["bbob"] = fs;
    return d;
}

const suite::Spec& spec_of(const suite::Native& p)
{
    auto* q = dynamic_cast<const suite::Problem*>(&p);
    if (q == nullptr) throw std::runtime_error("suite: not a problem of the suite");
    return q->spec();
}

// X as n rows of n_var values: a 1-D or 2-D array whose size is a multiple
// of n_var, read as NumPy's reshape(-1, n_var) reads it
Arr rows_of(const py::handle& o, int n_var, std::size_t& n)
{
    Arr X = Arr::ensure(o);
    if (!X) throw std::invalid_argument("the points must be an array of numbers");
    const std::size_t size = static_cast<std::size_t>(X.size()), w = static_cast<std::size_t>(n_var);
    if (size % w != 0)
        throw std::invalid_argument("the points must have " + std::to_string(n_var) + " variables each");
    n = size / w;
    return X;
}

// F (n x n_obj) and G (n x n_con) of the rows of X
std::pair<Arr, Arr> evaluate_rows(const suite::Native& p, const py::handle& o)
{
    std::size_t n = 0;
    Arr X = rows_of(o, p.n_var, n);
    Arr F({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(p.n_obj)});
    Arr G({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(p.n_con)});
    const double* x = X.data();
    double* f = F.mutable_data();
    double* g = G.mutable_data();
    {
        py::gil_scoped_release unlock;
        for (std::size_t i = 0; i < n; ++i)
            p.evaluate(x + i * static_cast<std::size_t>(p.n_var), f + i * static_cast<std::size_t>(p.n_obj),
                       p.n_con > 0 ? g + i * static_cast<std::size_t>(p.n_con) : nullptr);
    }
    return {std::move(F), std::move(G)};
}

// one point: (f, g) as lists
std::pair<std::vector<double>, std::vector<double>> evaluate_point(const suite::Native& p, const py::handle& o)
{
    std::vector<double> x = flat(o);
    if (static_cast<int>(x.size()) != p.n_var)
        throw std::invalid_argument("the point must have " + std::to_string(p.n_var) + " variables");
    std::vector<double> f(static_cast<std::size_t>(p.n_obj)), g(static_cast<std::size_t>(p.n_con));
    p.evaluate(x.data(), f.data(), g.empty() ? nullptr : g.data());
    return {std::move(f), std::move(g)};
}

}   // namespace

void bind_suite(py::module_& m)
{
    py::class_<suite::Native, std::shared_ptr<suite::Native>>(m, "Native",
        "A benchmark problem of the C++ suite (include/mootation/suite): ZDT, DTLZ and its "
        "variants, C-DTLZ, MaF, ZCAT and the shifts and rotations over them, bbob-biobj and "
        "bbob-biobj-mixint, made from the "
        "spec benchmarks/native.py builds. Its values are NumPy's bit for bit, where NumPy's "
        "sin, cos, exp, log and pow are the C library's (benchmarks/_rows.py, agrees) and, for "
        "bbob-biobj, its BLAS adds in the order suite/bbob.hpp writes out. Hand it to "
        "a run as Problem.native and the core evaluates it without calling Python.")
        .def(py::init([](const py::dict& spec) {
                 return std::shared_ptr<suite::Native>(suite::make(spec_of(spec)));
             }),
             py::arg("spec"))
        .def_readonly("n_var", &suite::Native::n_var)
        .def_readonly("n_obj", &suite::Native::n_obj)
        .def_readonly("n_con", &suite::Native::n_con)
        .def_property_readonly("spec", [](const suite::Native& p) { return dict_of(spec_of(p)); },
                               "The spec the problem was made from.")
        .def("evaluate",
             [](const suite::Native& p, const py::handle& x) { return evaluate_point(p, x).first; },
             py::arg("x"), "The objectives of one point, as a list.")
        .def("constraints",
             [](const suite::Native& p, const py::handle& x) { return evaluate_point(p, x).second; },
             py::arg("x"), "The constraint values of one point (<= 0 satisfied), as a list.")
        .def("evaluate_batch",
             [](const suite::Native& p, const py::handle& X) { return evaluate_rows(p, X).first; },
             py::arg("X"), "The objectives of the rows of X: an (n, n_obj) array.")
        .def("constraints_batch",
             [](const suite::Native& p, const py::handle& X) { return evaluate_rows(p, X).second; },
             py::arg("X"), "The constraint values of the rows of X: an (n, n_con) array.")
        .def("evaluate_with_constraints",
             [](const suite::Native& p, const py::handle& X) {
                 auto FG = evaluate_rows(p, X);
                 return py::make_tuple(std::move(FG.first), std::move(FG.second));
             },
             py::arg("X"), "(objectives, constraint values) of the rows of X, one computation.")
        .def(py::pickle(
            [](const suite::Native& p) { return dict_of(spec_of(p)); },
            [](const py::dict& spec) {
                return std::shared_ptr<suite::Native>(suite::make(spec_of(spec)));
            }));
}

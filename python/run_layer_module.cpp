// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The run layer's numerics in C++, part of the module `mootation._core`
// (bind_run_layer, called from mootation_module.cpp): the nondominated filter,
// the run archive, the subset selection and the front indicators that
// python/mootation/run/ computes with NumPy, bit for bit the same (the headers
// say how). The Python modules call these when the extension is built and
// keep their own code as the reference; nothing here is needed to run a core.
// ============================================================================

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mootation/archive_select.hpp>
#include <mootation/grid_archive.hpp>
#include <mootation/indicators.hpp>
#include <mootation/nondominated.hpp>

namespace py = pybind11;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using mootation::select::Rows;

Rows rows_of(const Arr& a, const char* what)
{
    if (a.ndim() != 2) throw std::invalid_argument(std::string(what) + " must be a 2-D array");
    Rows r;
    r.n = static_cast<std::size_t>(a.shape(0));
    r.m = static_cast<std::size_t>(a.shape(1));
    r.v.assign(a.data(), a.data() + r.n * r.m);
    return r;
}

// None, or one value per objective
std::vector<double> frame_of(const py::object& o, std::size_t m, const char* what)
{
    if (o.is_none()) return {};
    Arr a = Arr::ensure(o);
    if (!a || a.ndim() != 1 || static_cast<std::size_t>(a.shape(0)) != m)
        throw std::invalid_argument(std::string(what) + " must give one value per objective");
    return std::vector<double>(a.data(), a.data() + m);
}

py::array_t<py::ssize_t> indices(const std::vector<std::size_t>& v)
{
    py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(v.size()));
    auto* p = out.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = static_cast<py::ssize_t>(v[i]);
    return out;
}

py::array_t<bool> flags(const std::vector<char>& v)
{
    py::array_t<bool> out(static_cast<py::ssize_t>(v.size()));
    auto* p = out.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i] != 0;
    return out;
}

Arr matrix(const std::vector<double>& v, std::size_t n, std::size_t m)
{
    Arr out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(m)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

Arr vector_of(const std::vector<double>& v)
{
    Arr out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

}   // namespace

void bind_run_layer(py::module_& m)
{
    m.def("nondominated",
          [](const Arr& F) {
              Rows r = rows_of(F, "nondominated: F");
              std::vector<std::size_t> keep;
              {
                  py::gil_scoped_release unlock;
                  keep = mootation::nd::filter(r.v.data(), r.n, r.m);
              }
              return indices(keep);
          },
          py::arg("F"),
          "Indices of the distinct rows of F that no other row dominates, in "
          "lexicographic order: F[these] is mootation.run.metrics.nondominated(F) "
          "(include/mootation/nondominated.hpp).");
    m.def("nondominated_mask",
          [](const Arr& F) {
              Rows r = rows_of(F, "nondominated_mask: F");
              std::vector<char> keep;
              {
                  py::gil_scoped_release unlock;
                  keep = mootation::nd::mask(r.v.data(), r.n, r.m);
              }
              return flags(keep);
          },
          py::arg("F"), "Which rows of F no row dominates (copies all kept).");

    // ── the subset selection (archive.py) ──────────────────────────────────
    m.def("almost_dominated",
          [](const Arr& G, const py::object& rows) {
              Rows g = rows_of(G, "almost_dominated: G");
              std::vector<std::size_t> which;
              if (rows.is_none()) {
                  for (std::size_t i = 0; i < g.n; ++i) which.push_back(i);
              } else {
                  for (auto v : rows.cast<std::vector<long long>>()) {
                      if (v < 0 || static_cast<std::size_t>(v) >= g.n)
                          throw std::invalid_argument("almost_dominated: a row index out of range");
                      which.push_back(static_cast<std::size_t>(v));
                  }
              }
              std::vector<char> out(which.size());
              {
                  py::gil_scoped_release unlock;
                  for (std::size_t t = 0; t < which.size(); ++t)
                      out[t] = mootation::select::almost_dominated(g, which[t]);
              }
              return flags(out);
          },
          py::arg("G"), py::arg("rows") = py::none(),
          "mootation.run.archive.almost_dominated, the same flags.");
    m.def("set_aside",
          [](const Arr& F, const py::object& ideal, const py::object& nadir) {
              Rows f = rows_of(F, "set_aside: F");
              auto lo = frame_of(ideal, f.m, "ideal");
              auto hi = frame_of(nadir, f.m, "nadir");
              std::pair<Rows, std::vector<char>> got;
              {
                  py::gil_scoped_release unlock;
                  got = mootation::select::set_aside(f, lo, hi);
              }
              return py::make_tuple(matrix(got.first.v, f.n, f.m), flags(got.second));
          },
          py::arg("F"), py::arg("ideal") = py::none(), py::arg("nadir") = py::none(),
          "mootation.run.archive.set_aside: (G, kept), the same values.");
    m.def("dss_select",
          [](const Arr& F, std::size_t k, const py::object& ideal, const py::object& nadir) {
              Rows f = rows_of(F, "dss_select: F");
              auto lo = frame_of(ideal, f.m, "ideal");
              auto hi = frame_of(nadir, f.m, "nadir");
              std::vector<std::size_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::select::dss_order(f, k, lo, hi);
              }
              return indices(out);
          },
          py::arg("F"), py::arg("k"), py::arg("ideal") = py::none(), py::arg("nadir") = py::none(),
          "mootation.run.archive.dss_order(F, k, ideal, nadir), index for index "
          "(include/mootation/archive_select.hpp).");
    m.def("igdp_self_order",
          [](const Arr& F, std::size_t k, const py::object& ideal, const py::object& nadir) {
              Rows f = rows_of(F, "igdp_self_order: F");
              auto lo = frame_of(ideal, f.m, "ideal");
              auto hi = frame_of(nadir, f.m, "nadir");
              std::vector<std::size_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::select::igdp_self_order(f, k, lo, hi);
              }
              return indices(out);
          },
          py::arg("F"), py::arg("k"), py::arg("ideal") = py::none(), py::arg("nadir") = py::none(),
          "mootation.run.archive.igdp_self_order, index for index.");
    m.def("greedy_igdp",
          [](const Arr& C, const Arr& R, std::size_t k) {
              Rows c = rows_of(C, "greedy_igdp: C"), r = rows_of(R, "greedy_igdp: R");
              if (c.m != r.m) throw std::invalid_argument("greedy_igdp: C and R need the same columns");
              std::vector<std::size_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::select::greedy_igdp(c, r, k);
              }
              return indices(out);
          },
          py::arg("C"), py::arg("R"), py::arg("k"),
          "mootation.run.archive.greedy_igdp(dplus_matrix(C, R), k), its rows computed "
          "as needed.");

    // ── the front indicators (metrics.py) ──────────────────────────────────
    m.def("front_distances",
          [](const Arr& F, const Arr& R, unsigned threads) {
              Rows f = rows_of(F, "front_distances: F"), r = rows_of(R, "front_distances: ref");
              if (f.m != r.m) throw std::invalid_argument("front_distances: F and ref need the same columns");
              if (f.n == 0) throw std::invalid_argument("front_distances: F is empty");
              std::vector<double> to_set(r.n), to_ref(f.n), eps(r.n);
              {
                  py::gil_scoped_release unlock;
                  mootation::indicators::front_distances(f.v.data(), f.n, r.v.data(), r.n, f.m,
                                                         to_set.data(), to_ref.data(), eps.data(),
                                                         threads);
              }
              return py::make_tuple(vector_of(to_set), vector_of(to_ref), vector_of(eps));
          },
          py::arg("F"), py::arg("ref"), py::arg("threads") = 1,
          "mootation.run.metrics.front_distances: (to_set, to_ref, eps), the same values "
          "(include/mootation/indicators.hpp); threads 0: every core.");
    m.def("igd_distances",
          [](const Arr& F, const Arr& R, unsigned threads) {
              Rows f = rows_of(F, "igd_distances: F"), r = rows_of(R, "igd_distances: ref");
              if (f.m != r.m) throw std::invalid_argument("igd_distances: F and ref need the same columns");
              if (f.n == 0) throw std::invalid_argument("igd_distances: F is empty");
              std::vector<double> near(r.n);
              {
                  py::gil_scoped_release unlock;
                  mootation::indicators::igd_distances(f.v.data(), f.n, r.v.data(), r.n, f.m,
                                                       near.data(), threads);
              }
              return vector_of(near);
          },
          py::arg("F"), py::arg("ref"), py::arg("threads") = 1,
          "The distance from every reference point to the nearest row of F: the vector "
          "mootation.run.metrics.igd is the mean of.");
    m.def("r2_mean",
          [](const Arr& G, const Arr& W) {
              Rows g = rows_of(G, "r2_mean: G"), w = rows_of(W, "r2_mean: W");
              if (g.m != w.m) throw std::invalid_argument("r2_mean: G and W need the same columns");
              if (w.n == 0) throw std::invalid_argument("r2_mean: no weights");
              py::gil_scoped_release unlock;
              return mootation::indicators::r2_mean(g.v.data(), g.n, w.v.data(), w.n, g.m);
          },
          py::arg("G"), py::arg("W"),
          "mootation.run.metrics.r2_discrete on the normalised set G and the weights W.");

    // ── the run archive (archive.py, GridArchive) ──────────────────────────
    using GA = mootation::GridArchive;
    py::class_<GA>(m, "GridArchive",
                   "mootation.run.archive.GridArchive's points, kept in C++ "
                   "(include/mootation/grid_archive.hpp); archive.GridArchive uses it.")
        .def(py::init([](int n_obj, int n_vars, const py::object& ideal, const py::object& nadir,
                         double delta) {
                 const auto m = static_cast<std::size_t>(n_obj);
                 return new GA(n_obj, n_vars, frame_of(ideal, m, "ideal"),
                               frame_of(nadir, m, "nadir"), delta);
             }),
             py::arg("n_obj"), py::arg("n_vars") = 0, py::kw_only(), py::arg("ideal") = py::none(),
             py::arg("nadir") = py::none(), py::arg("delta") = 0.0)
        .def("add",
             [](GA& a, const py::object& f, const py::object& x) {
                 Arr fa = Arr::ensure(f);
                 if (!fa || fa.ndim() != 1 || static_cast<std::size_t>(fa.shape(0)) != a.n_obj()) {
                     a.add_invalid();
                     return false;
                 }
                 if (x.is_none() || a.n_vars() == 0) return a.add(fa.data(), nullptr);
                 Arr xa = Arr::ensure(x);
                 if (!xa || static_cast<std::size_t>(xa.size()) != a.n_vars())
                     throw std::invalid_argument("GridArchive.add: x must have n_vars values");
                 return a.add(fa.data(), xa.data());
             },
             py::arg("f"), py::arg("x") = py::none(), "Offer one point; True if it is stored.")
        .def("add_many",
             [](GA& a, const Arr& F, const py::object& X) {
                 Rows f = rows_of(F, "GridArchive.add_many: F");
                 if (f.m != a.n_obj())
                     throw std::invalid_argument("GridArchive.add_many: F needs one column per objective");
                 Arr xa;
                 const bool with_x = !X.is_none() && a.n_vars() > 0;
                 if (with_x) {
                     xa = Arr::ensure(X);
                     if (!xa || xa.ndim() != 2 || static_cast<std::size_t>(xa.shape(0)) != f.n ||
                         static_cast<std::size_t>(xa.shape(1)) != a.n_vars())
                         throw std::invalid_argument("GridArchive.add_many: X must be n x n_vars");
                 }
                 std::vector<char> out(f.n);
                 const double* xp = with_x ? xa.data() : nullptr;
                 {
                     py::gil_scoped_release unlock;
                     for (std::size_t i = 0; i < f.n; ++i)
                         out[i] = a.add(f.row(i), xp ? xp + i * a.n_vars() : nullptr);
                 }
                 return flags(out);
             },
             py::arg("F"), py::arg("X") = py::none(),
             "Offer the rows of F in order, as add() one at a time would; whether each was stored.")
        .def("points",
             [](const GA& a) {
                 std::vector<double> F, X;
                 std::vector<char> E;
                 a.points(F, X, E);
                 return py::make_tuple(matrix(F, E.size(), a.n_obj()), matrix(X, E.size(), a.n_vars()),
                                       flags(E));
             },
             "(F, X, is_extreme) of the stored points, in insertion order.")
        .def("grid_frame",
             [](const GA& a) -> py::object {
                 std::vector<double> lo, hi;
                 if (!a.frame(lo, hi)) return py::none();
                 return py::make_tuple(vector_of(lo), vector_of(hi));
             },
             "(lo, lo + span) of the grid, or None before the archive's own frame exists.")
        .def("__len__", &GA::size)
        .def_property_readonly("offered", &GA::offered)
        .def_property_readonly("accepted", &GA::accepted)
        .def_property_readonly("delta", &GA::delta);

    // ── IGD+ and epsilon over everything evaluated (§5.2 of task 10) ───────
    using IF = mootation::indicators::IncrementalFront;
    py::class_<IF>(m, "IncrementalFront",
                   "IGD+ and the additive epsilon of every point offered so far against "
                   "a reference Z, in the frame (f - ideal)/(nadir - ideal), and the "
                   "evaluation at which IGD+ first reached each level "
                   "(include/mootation/indicators.hpp).")
        .def(py::init([](const Arr& Z, const Arr& ideal, const Arr& nadir, std::vector<double> levels) {
                 Rows z = rows_of(Z, "IncrementalFront: Z");
                 if (ideal.ndim() != 1 || nadir.ndim() != 1 ||
                     static_cast<std::size_t>(ideal.shape(0)) != z.m ||
                     static_cast<std::size_t>(nadir.shape(0)) != z.m)
                     throw std::invalid_argument("IncrementalFront: ideal and nadir need one value "
                                                 "per objective");
                 return new IF(z.v.data(), z.n, z.m,
                               std::vector<double>(ideal.data(), ideal.data() + z.m),
                               std::vector<double>(nadir.data(), nadir.data() + z.m),
                               std::move(levels));
             }),
             py::arg("Z"), py::arg("ideal"), py::arg("nadir"),
             py::arg("levels") = std::vector<double>{})
        .def("update",
             [](IF& a, const Arr& F, const py::array_t<long long, py::array::c_style |
                                                                 py::array::forcecast>& fe) {
                 Rows f = rows_of(F, "IncrementalFront.update: F");
                 if (static_cast<std::size_t>(fe.size()) != f.n)
                     throw std::invalid_argument("IncrementalFront.update: one evaluation count per row");
                 py::gil_scoped_release unlock;
                 a.update(f.v.data(), f.n, fe.data());
             },
             py::arg("F"), py::arg("fe"),
             "Offer the rows of F, evaluated at the counts fe, in order.")
        .def("igd_plus", &IF::igd_plus)
        .def("eps_plus", &IF::eps_plus)
        .def_property_readonly("seen", &IF::seen)
        .def_property_readonly("levels", &IF::levels)
        .def_property_readonly("reached", &IF::reached)
        .def("distances", [](const IF& a) { return vector_of(a.distances()); },
             "min over the points so far of d+(z, a), for every reference point z");
}

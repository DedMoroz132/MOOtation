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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mootation/archive_select.hpp>
#include <mootation/decision_indicators.hpp>
#include <mootation/json.hpp>
#include <mootation/toml.hpp>
#include <mootation/campaign/config.hpp>
#include <mootation/run_compute.hpp>
#include <mootation/run_ledger.hpp>
#include <mootation/store/column_codec.hpp>
#include <mootation/store/crc32c.hpp>
#include <mootation/store/pack.hpp>
#include <mootation/store/record.hpp>
#include <mootation/store/meta_table.hpp>
#include <mootation/store/metrics_table.hpp>
#include <mootation/store/record_tables.hpp>
#include <mootation/store/run_json.hpp>
#include <mootation/store/text_codec.hpp>
#include <mootation/run_metrics.hpp>
#include <mootation/grid_archive.hpp>
#include <mootation/indicators.hpp>
#include <mootation/nondominated.hpp>
#include <mootation/set_archive.hpp>

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
    // ── the decision-space indicators (metrics.py; task 17, stage 2a) ──────
    // X and the sample in the problem's units, lo and hi its bounds, cyclic
    // the indices of the variables that wrap
    struct Decision {
        std::vector<double> U, R;
        std::size_t nu = 0, nr = 0, d = 0;
        std::vector<char> cyclic;
    };
    auto decision = [](const Arr& X, const Arr* P, const Arr& lo, const Arr& hi,
                       const std::vector<std::size_t>& cyclic, const char* what) {
        Rows x = rows_of(X, what);
        Decision out;
        out.d = x.m;
        if (static_cast<std::size_t>(lo.size()) != x.m || static_cast<std::size_t>(hi.size()) != x.m)
            throw std::invalid_argument(std::string(what) + ": one bound per variable");
        out.U = mootation::indicators::unit_rows(x.v.data(), x.n, x.m, lo.data(), hi.data());
        out.nu = x.n;
        if (P != nullptr) {
            Rows p = rows_of(*P, what);
            if (p.m != x.m) throw std::invalid_argument(std::string(what) + ": X and the sample need the same columns");
            out.R = mootation::indicators::unit_rows(p.v.data(), p.n, p.m, lo.data(), hi.data());
            out.nr = p.n;
        }
        if (!cyclic.empty()) {
            out.cyclic.assign(x.m, 0);
            for (std::size_t j : cyclic) {
                if (j >= x.m) throw std::invalid_argument(std::string(what) + ": a cyclic index past the variables");
                out.cyclic[j] = 1;
            }
        }
        return out;
    };
    m.def("igdx",
          [decision](const Arr& X, const Arr& P, const Arr& lo, const Arr& hi, std::vector<std::size_t> cyclic) {
              Decision s = decision(X, &P, lo, hi, cyclic, "igdx");
              py::gil_scoped_release unlock;
              return mootation::indicators::igdx(s.U.data(), s.nu, s.R.data(), s.nr, s.d, s.cyclic);
          },
          py::arg("X"), py::arg("pareto_set"), py::arg("lo"), py::arg("hi"), py::arg("cyclic"),
          "mootation.run.metrics.igdx, the same value (include/mootation/decision_indicators.hpp).");
    m.def("ps_dmin",
          [decision](const Arr& X, const Arr& P, const Arr& lo, const Arr& hi, std::vector<std::size_t> cyclic) {
              Decision s = decision(X, &P, lo, hi, cyclic, "ps_dmin");
              py::gil_scoped_release unlock;
              return mootation::indicators::ps_dmin(s.U.data(), s.nu, s.R.data(), s.nr, s.d, s.cyclic);
          },
          py::arg("X"), py::arg("pareto_set"), py::arg("lo"), py::arg("hi"), py::arg("cyclic"),
          "mootation.run.metrics.ps_dmin, the same value.");
    m.def("pairwise_distance",
          [decision](const Arr& X, const Arr& lo, const Arr& hi, std::vector<std::size_t> cyclic) {
              Decision s = decision(X, nullptr, lo, hi, cyclic, "pairwise_distance");
              py::gil_scoped_release unlock;
              return mootation::indicators::pairwise_distance(s.U.data(), s.nu, s.d, s.cyclic);
          },
          py::arg("X"), py::arg("lo"), py::arg("hi"), py::arg("cyclic"),
          "mootation.run.metrics.pairwise_distance, the same value.");
    // ── more of metrics.py (run_metrics.hpp; task 17, stage 2a) ──────────────
    auto frame = [](const Arr& a, std::size_t m, const char* what) {
        if (static_cast<std::size_t>(a.size()) != m)
            throw std::invalid_argument(std::string(what) + ": one value per objective");
        return std::vector<double>(a.data(), a.data() + m);
    };
    m.def("roi_dist",
          [frame](const Arr& F, const Arr& ideal, const Arr& nadir) {
              Rows f = rows_of(F, "roi_dist: F");
              return mootation::run_metrics::roi_dist(f.v.data(), f.n, f.m, frame(ideal, f.m, "roi_dist"),
                                                      frame(nadir, f.m, "roi_dist"));
          },
          py::arg("F"), py::arg("ideal"), py::arg("nadir"), "mootation.run.metrics.roi_dist, the same value.");
    m.def("range_cover_each",
          [frame](const Arr& F, const Arr& ideal, const Arr& nadir) {
              Rows f = rows_of(F, "range_cover_each: F");
              return vector_of(mootation::run_metrics::range_cover_each(
                  f.v.data(), f.n, f.m, frame(ideal, f.m, "range_cover_each"), frame(nadir, f.m, "range_cover_each")));
          },
          py::arg("F"), py::arg("ideal"), py::arg("nadir"), "mootation.run.metrics.range_cover_each, the same values.");
    m.def("dup_share",
          [](const Arr& F) {
              Rows f = rows_of(F, "dup_share: F");
              return mootation::run_metrics::dup_share(f.v.data(), f.n, f.m);
          },
          py::arg("F"), "mootation.run.metrics.dup_share, the same value.");
    m.def("nd_share",
          [](const Arr& F) {
              Rows f = rows_of(F, "nd_share: F");
              return mootation::run_metrics::nd_share(f.v.data(), f.n, f.m);
          },
          py::arg("F"), "mootation.run.metrics.nd_share, the same value.");
    m.def("nn_cv",
          [frame](const Arr& F, const Arr& ideal, const Arr& nadir) -> py::object {
              Rows f = rows_of(F, "nn_cv: F");
              const double v = mootation::run_metrics::nn_cv(f.v.data(), f.n, f.m, frame(ideal, f.m, "nn_cv"),
                                                             frame(nadir, f.m, "nn_cv"));
              if (std::isnan(v)) return py::none();
              return py::float_(v);
          },
          py::arg("F"), py::arg("ideal"), py::arg("nadir"), "mootation.run.metrics.nn_cv, the same value or None.");
    m.def("quantile",
          [](std::vector<double> a, double q) { return mootation::run_metrics::quantile(std::move(a), q); },
          py::arg("a"), py::arg("q"), "np.quantile(a, q) (method linear), the same value.");
    m.def("median",
          [](std::vector<double> a) { return mootation::run_metrics::median(std::move(a)); },
          py::arg("a"), "np.median(a), the same value.");
    m.def("hv_normalised",
          [frame](const Arr& F, const Arr& ideal, const Arr& nadir, double ref_scale, std::size_t samples,
                  std::uint64_t seed, int exact_max_m, const std::string& algo) {
              if (algo != "auto" && algo != "wfg" && algo != "box")
                  throw std::invalid_argument("hv_normalised: algo = '" + algo + "': auto, wfg or box");
              Rows f = rows_of(F, "hv_normalised: F");
              const auto lo = frame(ideal, f.m, "hv_normalised"), hi = frame(nadir, f.m, "hv_normalised");
              mootation::run_metrics::Hv h;
              {
                  py::gil_scoped_release unlock;
                  h = mootation::run_metrics::hypervolume(f.v.data(), f.n, f.m, lo, hi, ref_scale, samples, seed,
                                                          exact_max_m, algo);
              }
              return py::make_tuple(h.value, std::string(h.method), h.se);
          },
          py::arg("F"), py::arg("ideal"), py::arg("nadir"), py::arg("ref_scale"), py::arg("samples"),
          py::arg("seed"), py::arg("exact_max_m"), py::arg("algo"),
          "mootation.run.metrics.hypervolume: (value, method, the Monte-Carlo estimate's "
          "standard error or NaN), the same value.");
    m.def("cover_rate",
          [](const Arr& X, const Arr& P) {
              Rows x = rows_of(X, "cover_rate: X"), p = rows_of(P, "cover_rate: pareto_set");
              if (x.m != p.m) throw std::invalid_argument("cover_rate: X and the sample need the same columns");
              return mootation::indicators::cover_rate(x.v.data(), x.n, p.v.data(), p.n, x.m);
          },
          py::arg("X"), py::arg("pareto_set"),
          "mootation.run.metrics.cover_rate, its root by the library's pow "
          "(include/mootation/decision_indicators.hpp).");
    m.def("nearest_other",
          [](const Arr& R) {
              Rows r = rows_of(R, "nearest_other: R");
              std::vector<double> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::indicators::nearest_other(r.v.data(), r.n, r.m);
              }
              return vector_of(out);
          },
          py::arg("R"),
          "For every row of R the distance to the nearest other row (metrics.reference_step).");
    m.def("rnd_uniform",
          [](std::uint32_t seed, std::size_t n) {
              std::mt19937 g(seed);
              std::vector<double> out(n);
              for (double& v : out) v = mootation::rnd::uniform01(g);
              return vector_of(out);
          },
          py::arg("seed"), py::arg("n"),
          "n values of rnd::uniform01 of a std::mt19937 seeded with `seed` (random.hpp): what "
          "metrics._mc_draws repeats with NumPy's MT19937 (task 17, supplement 1, §3).");
    m.def("spread_first", &mootation::indicators::spread_first, py::arg("seed"), py::arg("rows"),
          "The first point of the second answer's spread_select among `rows`, from the run's seed "
          "(include/mootation/decision_indicators.hpp).");
    m.def("spread_select",
          [](const Arr& X, std::size_t n, const Arr& lo, const Arr& hi, std::size_t first) {
              Rows x = rows_of(X, "spread_select: X");
              if (static_cast<std::size_t>(lo.size()) != x.m || static_cast<std::size_t>(hi.size()) != x.m)
                  throw std::invalid_argument("spread_select: one bound per variable");
              if (x.n > n && first >= x.n) throw std::invalid_argument("spread_select: first past the rows");
              std::vector<std::size_t> out;
              {
                  py::gil_scoped_release unlock;
                  const std::vector<double> Z =
                      mootation::indicators::unit_rows(x.v.data(), x.n, x.m, lo.data(), hi.data());
                  out = mootation::indicators::spread_select(Z.data(), x.n, x.m, n, first);
              }
              return out;
          },
          py::arg("X"), py::arg("n"), py::arg("lo"), py::arg("hi"), py::arg("first"),
          "mootation.run.archive.spread_select from the given first point, the same indices.");
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
    // ── metrics.compute as a whole (run_compute.hpp; task 17, stage 2b) ──────
    m.def("compute_indicators",
          [](const Arr& F, const std::vector<std::string>& which, const py::object& ref,
             const py::object& ideal, const py::object& nadir, std::size_t pop, int exact_max_m,
             std::size_t mc_samples, std::uint64_t seed, const std::string& hv_algo, const py::object& X,
             const py::object& pareto_set, const py::object& lo, const py::object& hi,
             std::vector<std::size_t> cyclic, const py::object& feasible, const py::object& V,
             const py::object& g, double ref_step) {
              namespace rc = mootation::run_compute;
              if (hv_algo != "auto" && hv_algo != "wfg" && hv_algo != "box")
                  throw std::invalid_argument("compute_indicators: hv_algo = '" + hv_algo + "': auto, wfg or box");
              Rows f = rows_of(F, "compute_indicators: F");
              rc::Context c;
              c.m = f.m;
              Rows r, x, p;
              if (!ref.is_none()) {
                  r = rows_of(Arr::ensure(ref), "compute_indicators: ref");
                  if (r.m != f.m) throw std::invalid_argument("compute_indicators: F and ref need the same columns");
                  c.ref = r.v.data();
                  c.ref_n = r.n;
              }
              c.ideal = frame_of(ideal, f.m, "compute_indicators: ideal");
              c.nadir = frame_of(nadir, f.m, "compute_indicators: nadir");
              c.has_frame = !ideal.is_none() && !nadir.is_none();
              c.pop = pop;
              c.exact_max_m = exact_max_m;
              c.mc_samples = mc_samples;
              c.seed = seed;
              c.hv_algo = hv_algo;
              rc::Rows rows;
              rows.F = f.v.data();
              rows.n = f.n;
              if (!X.is_none()) {
                  x = rows_of(Arr::ensure(X), "compute_indicators: X");
                  if (x.n != f.n) throw std::invalid_argument("compute_indicators: X needs a row per row of F");
                  rows.X = x.v.data();
                  rows.d = x.m;
              }
              // the sample and the bounds are read with the variables only
              if (!X.is_none() && !pareto_set.is_none()) {
                  p = rows_of(Arr::ensure(pareto_set), "compute_indicators: pareto_set");
                  if (p.n && p.m != x.m) throw std::invalid_argument("compute_indicators: X and the sample need the same columns");
                  c.pset = p.v.data();
                  c.pset_n = p.n;
              }
              if (!X.is_none() && !lo.is_none() && !hi.is_none()) {
                  c.lo = frame_of(lo, x.m, "compute_indicators: lo");
                  c.hi = frame_of(hi, x.m, "compute_indicators: hi");
                  c.has_bounds = true;
              }
              for (std::size_t j : cyclic)
                  if (!X.is_none() && j >= x.m) throw std::invalid_argument("compute_indicators: a cyclic index past the variables");
              if (!X.is_none()) c.cyclic = std::move(cyclic);
              c.ref_step = ref_step;
              std::vector<char> ok;
              std::vector<double> viol, dist;
              auto per_row = [&](const py::object& o, const char* what) {
                  Arr a = Arr::ensure(o);
                  if (!a || a.ndim() != 1 || static_cast<std::size_t>(a.shape(0)) != f.n)
                      throw std::invalid_argument(std::string("compute_indicators: ") + what + ": one value per row");
                  return std::vector<double>(a.data(), a.data() + f.n);
              };
              if (!feasible.is_none()) {
                  for (double v : per_row(feasible, "feasible")) ok.push_back(v != 0.0 ? 1 : 0);
                  rows.constrained = true;
                  rows.feasible = ok.data();
              }
              if (!V.is_none()) { viol = per_row(V, "V"); rows.V = viol.data(); }
              if (!g.is_none()) { dist = per_row(g, "g"); rows.g = dist.data(); }
              rc::Out out;
              {
                  py::gil_scoped_release unlock;
                  out = rc::compute(rows, c, which);
              }
              py::list items;
              for (const auto& kv : out.items) {
                  const rc::Value& v = kv.second;
                  py::object o;
                  switch (v.kind) {
                  case rc::Value::None: o = py::none(); break;
                  case rc::Value::Real: o = py::float_(v.real); break;
                  case rc::Value::Int: o = py::int_(v.integer); break;
                  case rc::Value::Bool: o = py::bool_(v.flag); break;
                  case rc::Value::Text: o = py::str(v.text); break;
                  case rc::Value::List: {
                      py::list l;
                      for (double e : v.list) l.append(py::float_(e));
                      o = l;
                      break;
                  }
                  case rc::Value::Dict: {
                      py::dict dct;
                      for (const auto& e : v.dict) dct[py::str(e.first)] = py::float_(e.second);
                      o = dct;
                      break;
                  }
                  }
                  items.append(py::make_tuple(kv.first, o));
              }
              return items;
          },
          py::arg("F"), py::arg("which"), py::arg("ref"), py::arg("ideal"), py::arg("nadir"), py::arg("pop"),
          py::arg("exact_max_m"), py::arg("mc_samples"), py::arg("seed"), py::arg("hv_algo"), py::arg("X"),
          py::arg("pareto_set"), py::arg("lo"), py::arg("hi"), py::arg("cyclic"), py::arg("feasible"),
          py::arg("V"), py::arg("g"), py::arg("ref_step"),
          "mootation.run.metrics.compute's (name, value) pairs in its order, the same values "
          "(include/mootation/run_compute.hpp); range_cover_each unrounded.");

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

    // ── the second answer's candidates (archive.py, SetArchive) ────────────
    using SA = mootation::SetArchive;
    py::class_<SA>(m, "SetArchive",
                   "mootation.run.archive.SetArchive's points, kept in C++ "
                   "(include/mootation/set_archive.hpp); archive.SetArchive uses it.")
        .def(py::init<std::size_t, std::size_t>(), py::arg("n_obj"), py::arg("n_vars"))
        .def("add_many",
             [](SA& a, const Arr& F, const Arr& X) {
                 Rows f = rows_of(F, "SetArchive.add_many: F");
                 Rows x = rows_of(X, "SetArchive.add_many: X");
                 if (f.m != a.n_obj() || x.n != f.n || x.m != a.n_vars())
                     throw std::invalid_argument("SetArchive.add_many: F must be n x n_obj, X n x n_vars");
                 py::gil_scoped_release unlock;
                 for (std::size_t i = 0; i < f.n; ++i) a.add(f.row(i), x.m ? x.row(i) : nullptr);
             },
             py::arg("F"), py::arg("X"), "Offer the rows of F (and X) in order.")
        .def("points",
             [](const SA& a) {
                 std::vector<double> F, X;
                 a.points(F, X);
                 return py::make_tuple(matrix(F, a.size(), a.n_obj()), matrix(X, a.size(), a.n_vars()));
             },
             "(F, X) of the stored points, in the order they were offered.")
        .def("__len__", &SA::size);

    // ── IGD+ and epsilon over everything evaluated (§5.2 of task 10) ───────
    using IF = mootation::indicators::IncrementalFront;
    py::class_<IF>(m, "IncrementalFront",
                   "IGD+ and the additive epsilon of every point offered so far against "
                   "a reference Z, in the frame (f - ideal)/(nadir - ideal), and the "
                   "evaluation at which IGD+ first reached each level "
                   "(include/mootation/indicators.hpp).")
        .def(py::init([](const Arr& Z, const Arr& ideal, const Arr& nadir, std::vector<double> levels,
                         std::optional<bool> skip_dominated) {
                 Rows z = rows_of(Z, "IncrementalFront: Z");
                 if (ideal.ndim() != 1 || nadir.ndim() != 1 ||
                     static_cast<std::size_t>(ideal.shape(0)) != z.m ||
                     static_cast<std::size_t>(nadir.shape(0)) != z.m)
                     throw std::invalid_argument("IncrementalFront: ideal and nadir need one value "
                                                 "per objective");
                 return new IF(z.v.data(), z.n, z.m,
                               std::vector<double>(ideal.data(), ideal.data() + z.m),
                               std::vector<double>(nadir.data(), nadir.data() + z.m),
                               std::move(levels), skip_dominated.value_or(IF::skip_pays(z.m)));
             }),
             py::arg("Z"), py::arg("ideal"), py::arg("nadir"),
             py::arg("levels") = std::vector<double>{}, py::arg("skip_dominated") = py::none(),
             "skip_dominated: skip the points weakly dominated by one offered before (the same "
             "values); None: where it pays, up to five objectives")
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
        .def_property_readonly("skipped", &IF::skipped)
        .def_property_readonly("levels", &IF::levels)
        .def_property_readonly("reached", &IF::reached)
        .def("distances", [](const IF& a) { return vector_of(a.distances()); },
             "min over the points so far of d+(z, a), for every reference point z");

    // ── the store's column codecs (store/column_codec.hpp; task 17, stage 4) ──
    m.def("store_encode_f64",
          [](const Arr& x) {
              if (x.ndim() != 1) throw std::invalid_argument("store_encode_f64: a 1-D array");
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_f64(x.data(), static_cast<std::size_t>(x.size()));
              }
              return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          py::arg("x"), "A float64 column's bytes (include/mootation/store/column_codec.hpp).");
    m.def("store_decode_f64",
          [](const py::bytes& b, std::size_t n) {
              const std::string s = b;
              const auto x = mootation::store::decode_f64(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), n);
              return vector_of(x);
          },
          py::arg("data"), py::arg("n"), "The n values of a float64 column's bytes; an error where they do not hold them.");
    m.def("crc32c",
          [](const py::bytes& b, std::uint32_t crc) {
              const std::string s = b;
              return mootation::store::crc32c(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), crc);
          },
          py::arg("data"), py::arg("crc") = 0,
          "CRC-32C of the bytes (include/mootation/store/crc32c.hpp); crc: the value of the bytes before.");
    m.def("store_encode_text",
          [](const py::bytes& b) {
              const std::string s = b;
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_text(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              }
              return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          py::arg("data"), "Bytes compressed by LZ77 and the range coder (include/mootation/store/text_codec.hpp).");
    m.def("store_decode_text",
          [](const py::bytes& b) {
              const std::string s = b;
              const auto out = mootation::store::decode_text(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          py::arg("data"), "The bytes back from store_encode_text's.");
    m.def("store_encode_tables",
          [](const py::list& tables) {
              std::vector<mootation::store::Table> ts;
              for (const auto& item : tables) {
                  const auto tup = item.cast<py::tuple>();
                  mootation::store::Table t;
                  t.name = tup[0].cast<std::string>();
                  t.columns = tup[1].cast<std::vector<std::string>>();
                  Arr a = Arr::ensure(tup[2]);
                  if (!a || a.ndim() != 2 || static_cast<std::size_t>(a.shape(1)) != t.columns.size())
                      throw std::invalid_argument("store_encode_tables: " + t.name + ": a rows x columns array");
                  t.rows = static_cast<std::size_t>(a.shape(0));
                  t.values.assign(a.data(), a.data() + a.size());
                  ts.push_back(std::move(t));
              }
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_tables(ts);
              }
              return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          py::arg("tables"),
          "A record's point sets [(name, columns, array)] coded together, a column of one name "
          "through every table (include/mootation/store/record_tables.hpp).");
    m.def("store_decode_tables",
          [](const py::bytes& b) {
              const std::string s = b;
              const auto ts = mootation::store::decode_tables(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              py::list out;
              for (const auto& t : ts)
                  out.append(py::make_tuple(t.name, t.columns, matrix(t.values, t.rows, t.columns.size())));
              return out;
          },
          py::arg("data"), "[(name, columns, array)] back from store_encode_tables's bytes.");
    m.def("store_encode_i64",
          [](const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& x) {
              if (x.ndim() != 1) throw std::invalid_argument("store_encode_i64: a 1-D array");
              const auto out = mootation::store::encode_i64(x.data(), static_cast<std::size_t>(x.size()));
              return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          py::arg("x"), "An integer column's bytes.");
    m.def("store_decode_i64",
          [](const py::bytes& b, std::size_t n) {
              const std::string s = b;
              const auto x = mootation::store::decode_i64(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), n);
              py::array_t<std::int64_t> out(static_cast<py::ssize_t>(x.size()));
              std::copy(x.begin(), x.end(), out.mutable_data());
              return out;
          },
          py::arg("data"), py::arg("n"), "The n values of an integer column's bytes.");

    // ── a record of the store, format 3 (store/record.hpp; task 17, §3а.3) ──
    auto as_bytes = [](const std::vector<std::uint8_t>& v) {
        return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
    };
    m.def("store_encode_body",
          [as_bytes](const py::list& blocks, const py::list& tables) {
              std::vector<mootation::store::Block> bs;
              for (const auto& item : blocks) {
                  const auto t = item.cast<py::tuple>();
                  mootation::store::Block b;
                  b.name = t[0].cast<std::string>();
                  b.dtype = t[1].cast<std::string>();
                  b.shape = t[2].cast<std::vector<std::uint64_t>>();
                  const std::string d = t[3].cast<py::bytes>();
                  b.data.assign(d.begin(), d.end());
                  bs.push_back(std::move(b));
              }
              std::vector<mootation::store::PointTable> ts;
              for (const auto& item : tables) {
                  const auto t = item.cast<py::tuple>();
                  ts.push_back({t[0].cast<std::size_t>(), t[1].cast<std::vector<std::string>>()});
              }
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_body(bs, ts);
              }
              return as_bytes(out);
          },
          py::arg("blocks"), py::arg("tables"),
          "A record's body (include/mootation/store/record.hpp): blocks [(name, dtype, shape, bytes)], "
          "the point tables [(place, columns)] together in '#points'.");
    m.def("store_decode_body",
          [as_bytes](const py::bytes& body) {
              const std::string s = body;
              std::vector<mootation::store::Block> bs;
              {
                  py::gil_scoped_release unlock;
                  bs = mootation::store::decode_body(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              }
              py::list out;
              for (const auto& b : bs)
                  out.append(py::make_tuple(b.name, b.dtype, py::tuple(py::cast(b.shape)), as_bytes(b.data), b.how));
              return out;
          },
          py::arg("body"),
          "[(name, dtype, shape, bytes, how)] of a body, the point tables back at their places; how: '' "
          "or the method of a block of before ('zlib', 'shuffle+zlib'), its bytes as stored.");
    auto blocks_in = [](const py::list& blocks) {
        std::vector<mootation::store::Block> bs;
        for (const auto& item : blocks) {
            const auto t = item.cast<py::tuple>();
            mootation::store::Block b;
            b.name = t[0].cast<std::string>();
            b.dtype = t[1].cast<std::string>();
            b.shape = t[2].cast<std::vector<std::uint64_t>>();
            const std::string d = t[3].cast<py::bytes>();
            b.data.assign(d.begin(), d.end());
            bs.push_back(std::move(b));
        }
        return bs;
    };
    m.def("store_encode_run",
          [as_bytes, blocks_in](const py::list& blocks) {
              const auto bs = blocks_in(blocks);
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_run(bs);
              }
              return as_bytes(out);
          },
          py::arg("blocks"),
          "A run's body (store/run_json.hpp): meta.json's final once, the point tables (named by their "
          "#head) together in '#points'.");
    m.def("store_decode_run",
          [as_bytes](const py::bytes& body) {
              const std::string s = body;
              std::vector<mootation::store::Block> bs;
              {
                  py::gil_scoped_release unlock;
                  bs = mootation::store::decode_run(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              }
              py::list out;
              for (const auto& b : bs)
                  out.append(py::make_tuple(b.name, b.dtype, py::tuple(py::cast(b.shape)), as_bytes(b.data), b.how));
              return out;
          },
          py::arg("body"), "A run's blocks back from its body, meta.json's final rebuilt.");
    auto blocks_out = [as_bytes](const std::vector<mootation::store::Block>& bs) {
        py::list out;
        for (const auto& b : bs)
            out.append(py::make_tuple(b.name, b.dtype, py::tuple(py::cast(b.shape)), as_bytes(b.data)));
        return out;
    };
    m.def("store_point_tables",
          [blocks_in](const py::list& blocks) {
              py::list out;
              for (const auto& pt : mootation::store::point_tables(blocks_in(blocks)))
                  out.append(py::make_tuple(pt.place, pt.columns));
              return out;
          },
          py::arg("blocks"), "[(place, columns)] of the point tables among the blocks (named by their #head).");
    m.def("store_encode_meta_table",
          [as_bytes](const py::list& rows) {
              std::vector<std::pair<std::string, std::vector<std::uint8_t>>> rs;
              for (const auto& item : rows) {
                  const auto r = item.cast<py::tuple>();
                  const std::string d = r[1].cast<py::bytes>();
                  rs.emplace_back(r[0].cast<std::string>(), std::vector<std::uint8_t>(d.begin(), d.end()));
              }
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_meta_table(rs);
              }
              return as_bytes(out);
          },
          py::arg("rows"), "A pack's meta table of [(key, meta.json bytes)] (store/meta_table.hpp).");
    m.def("store_decode_meta_table",
          [as_bytes](const py::bytes& data) {
              const std::string s = data;
              std::vector<std::pair<std::string, std::vector<std::uint8_t>>> rs;
              {
                  py::gil_scoped_release unlock;
                  rs = mootation::store::decode_meta_table(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              }
              py::list out;
              for (const auto& r : rs) out.append(py::make_tuple(r.first, as_bytes(r.second)));
              return out;
          },
          py::arg("data"), "[(key, meta.json bytes)] of a pack's meta table.");
    m.def("store_encode_metrics_table",
          [as_bytes, blocks_in](const std::string& family, const py::list& runs) {
              std::vector<std::pair<std::string, std::vector<mootation::store::Block>>> rs;
              for (const auto& item : runs) {
                  const auto r = item.cast<py::tuple>();
                  rs.emplace_back(r[0].cast<std::string>(), blocks_in(r[1].cast<py::list>()));
              }
              std::vector<std::uint8_t> out;
              {
                  py::gil_scoped_release unlock;
                  out = mootation::store::encode_metrics_table(family, rs);
              }
              return as_bytes(out);
          },
          py::arg("family"), py::arg("runs"),
          "A pack's metrics table of its runs [(key, blocks)] (store/metrics_table.hpp).");
    m.def("store_final_once",
          [blocks_in, blocks_out](const py::list& blocks) {
              return blocks_out(mootation::store::final_once(blocks_in(blocks)));
          },
          py::arg("blocks"), "The blocks with meta.json's final values the trajectory holds alike left to it.");
    m.def("store_final_back",
          [blocks_in, blocks_out](const py::list& blocks) {
              return blocks_out(mootation::store::final_back(blocks_in(blocks)));
          },
          py::arg("blocks"), "The blocks with meta.json's final rebuilt from the trajectory.");
    m.def("store_encode_record",
          [as_bytes](const std::string& key, const py::bytes& body) {
              const std::string s = body;
              return as_bytes(mootation::store::encode_record(key, std::vector<std::uint8_t>(s.begin(), s.end())));
          },
          py::arg("key"), py::arg("body"), "A record's bytes: its frame around the body.");
    m.def("store_parse_record",
          [](const py::buffer& buf, std::size_t at) -> py::object {
              // the caller's bytes as they are (a segment's, read whole): no copy
              const py::buffer_info info = buf.request();
              if (info.itemsize != 1 || info.ndim != 1)
                  throw std::invalid_argument("store_parse_record: bytes");
              const auto* p = static_cast<const std::uint8_t*>(info.ptr);
              const auto r = mootation::store::parse_record(p, static_cast<std::size_t>(info.size), at);
              if (!r.ok) return py::none();
              return py::make_tuple(r.key, py::bytes(reinterpret_cast<const char*>(p + r.body), r.body_size),
                                    r.crc, r.end);
          },
          py::arg("buf"), py::arg("at"),
          "(key, body, crc, end) of the whole, sound record at `at` in buf, or None.");
    // ── the store's files: a segment's record, a pack (store/pack.hpp) ──────
    auto bytes_in = [](const py::buffer& buf, const char* what) {
        const py::buffer_info info = buf.request();
        if (info.itemsize != 1 || info.ndim != 1) throw std::invalid_argument(std::string(what) + ": bytes");
        return std::make_pair(static_cast<const std::uint8_t*>(info.ptr), static_cast<std::size_t>(info.size));
    };
    auto entry_tuple = [](const mootation::store::Entry& e) {
        return py::make_tuple(e.key, e.offset, e.length, e.crc, e.status);
    };
    m.def("store_append_record",
          [entry_tuple](const std::string& raw, const std::string& idx, const std::string& key,
                        const py::bytes& record, const std::string& status) {
              const std::string s = record;
              mootation::store::Entry e;
              {
                  py::gil_scoped_release unlock;
                  e = mootation::store::append_record(raw, idx, key, std::vector<std::uint8_t>(s.begin(), s.end()),
                                                      status);
              }
              return entry_tuple(e);
          },
          py::arg("raw"), py::arg("idx"), py::arg("key"), py::arg("record"), py::arg("status"),
          "Append a record to a segment (synced), then its line to the segment's index: "
          "(key, offset, length, crc, status).");
    m.def("store_write_pack",
          [entry_tuple](const std::string& path, const std::string& tmp, const std::string& header,
                        const py::bytes& meta, const py::bytes& metrics, const py::list& records) {
              const std::string ms = meta, ts = metrics;
              std::vector<mootation::store::PackRecord> rs;
              for (const auto& item : records) {
                  const auto t = item.cast<py::tuple>();
                  const std::string r = t[1].cast<py::bytes>();
                  rs.push_back({t[0].cast<std::string>(), std::vector<std::uint8_t>(r.begin(), r.end()),
                                t[2].cast<std::string>()});
              }
              std::vector<mootation::store::Entry> es;
              {
                  py::gil_scoped_release unlock;
                  es = mootation::store::write_pack(path, tmp, header, std::vector<std::uint8_t>(ms.begin(), ms.end()),
                                                    std::vector<std::uint8_t>(ts.begin(), ts.end()), rs);
              }
              py::list out;
              for (const auto& e : es) out.append(entry_tuple(e));
              return out;
          },
          py::arg("path"), py::arg("tmp"), py::arg("header"), py::arg("meta"), py::arg("metrics"), py::arg("records"),
          "Write a pack (include/mootation/store/pack.hpp) through tmp: records [(key, bytes, status)]; "
          "their [(key, offset, length, crc, status)].");
    m.def("store_pack_footer",
          [bytes_in](const py::buffer& buf, const std::string& name) {
              const auto [p, n] = bytes_in(buf, "store_pack_footer");
              const auto f = mootation::store::pack_footer(p, n, name);
              py::dict d;
              d["meta"] = py::make_tuple(f.meta_at, f.meta_size);
              d["metrics"] = py::make_tuple(f.metrics_at, f.metrics_size);
              d["index"] = py::make_tuple(f.index_at, f.index_size);
              d["crc"] = f.crc;
              d["index_crc"] = f.index_crc;
              d["end"] = f.end;
              return d;
          },
          py::arg("data"), py::arg("name"), "A pack's footer from the file's bytes.");
    m.def("store_pack_entries",
          [bytes_in, entry_tuple](const py::buffer& buf, const std::string& name) {
              const auto [p, n] = bytes_in(buf, "store_pack_entries");
              py::list out;
              for (const auto& e : mootation::store::pack_entries(p, n, name)) out.append(entry_tuple(e));
              return out;
          },
          py::arg("data"), py::arg("name"), "A pack's [(key, offset, length, crc, status)] from its index.");
    m.def("store_check_pack",
          [bytes_in](const py::buffer& buf, const std::string& name) {
              const auto [p, n] = bytes_in(buf, "store_check_pack");
              std::vector<std::string> bad;
              {
                  py::gil_scoped_release unlock;
                  bad = mootation::store::check_pack(p, n, name);
              }
              return bad;
          },
          py::arg("data"), py::arg("name"), "What is wrong with a pack, and where; empty where it is whole.");
    // ── JSON as Python's json module reads and writes it (json.hpp) ─────────
    m.def("json_reformat",
          [](const std::string& text, int indent, const py::object& separators) {
              const auto v = mootation::json::parse(text);
              if (separators.is_none()) return mootation::json::dump(v, indent);
              const auto sep = separators.cast<std::pair<std::string, std::string>>();
              return mootation::json::dump(v, indent, sep.first.c_str(), sep.second.c_str());
          },
          py::arg("text"), py::arg("indent") = -1, py::arg("separators") = py::none(),
          "json.dumps(json.loads(text), indent=indent or None, separators=...) by include/mootation/json.hpp.");
    // ── a run description read and checked (campaign/config.hpp) ────────────
    m.def("campaign_check_config",
          [](const std::string& text, const py::object& source_path, const py::object& base, const py::dict& r) -> py::tuple {
              namespace cp = mootation::campaign;
              cp::Registry reg;
              auto strings = [&](const char* k) { return r[k].cast<std::vector<std::string>>(); };
              reg.algorithms = strings("algorithms");
              reg.baselines = strings("baselines");
              reg.scalar_names = strings("scalar_names");
              reg.design_kinds = strings("design_kinds");
              reg.knob_names = strings("knob_names");
              reg.list_knobs = strings("list_knobs");
              for (const auto& item : r["text_knobs"].cast<py::list>()) {
                  const auto t = item.cast<py::tuple>();
                  reg.text_knobs.emplace_back(t[0].cast<std::string>(), t[1].cast<std::vector<std::string>>());
              }
              for (const auto& item : r["problems"].cast<py::list>()) {
                  const auto t = item.cast<py::tuple>();
                  reg.problems.emplace_back(t[0].cast<std::string>(), t[1].cast<int>());
              }
              reg.platform = r["platform"].cast<std::string>();
              reg.algorithms_error = r["algorithms_error"].cast<std::string>();
              const py::object findable = r["executable_findable"], groups = r["regex_groups"];
              reg.executable_findable = [findable](const std::string& a) { return findable(a).cast<bool>(); };
              reg.regex_groups = [groups](const std::string& p, std::string& error) {
                  const auto t = groups(p).cast<py::tuple>();
                  error = t[1].cast<std::string>();
                  return t[0].cast<int>();
              };
              std::optional<std::filesystem::path> src, b;
              if (!source_path.is_none()) src = std::filesystem::path(source_path.cast<std::wstring>());
              if (!base.is_none()) b = std::filesystem::path(base.cast<std::wstring>());
              try {
                  cp::Config cfg = cp::loads(text, reg, src);
                  const auto problems = cp::validate(cfg, reg, b);
                  return py::tuple(py::make_tuple(py::none(), problems, cfg.benchmark_problems));
              } catch (const cp::ConfigError& e) {
                  return py::tuple(py::make_tuple(std::string(e.what()), py::list(), py::list()));
              }
          },
          py::arg("text"), py::arg("source_path"), py::arg("base"), py::arg("registry"),
          "(error or None, problems, benchmark problems) of a run description, as run/config.py's loads and "
          "validate give them (include/mootation/campaign/config.hpp).");
    m.def("toml_to_json",
          [](const std::string& text) { return mootation::json::dump(mootation::toml::parse(text)); },
          py::arg("text"), "json.dumps(tomllib.loads(text)) by include/mootation/toml.hpp (dates as their text).");
    m.def("json_float_repr", &mootation::json::float_repr, py::arg("x"), "repr(x) by include/mootation/json.hpp.");
    m.def("sha256",
          [](const py::bytes& b) {
              const std::string s = b;
              const auto h = mootation::store::sha256(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
              return py::bytes(reinterpret_cast<const char*>(h.data()), h.size());
          },
          py::arg("data"), "SHA-256 of the bytes (include/mootation/store/sha256.hpp).");

    // ── a campaign run's accounting (campaign.py's account; run_ledger.hpp) ──
    struct Ledger {
        mootation::RunLedger L;
        std::size_t d;                            // the variables the archives keep
        py::object keep;                          // the archives, alive while the ledger is
        std::vector<double> X, F, G, T;           // the batch in progress
    };
    auto archive_of = [](const py::object& o) -> GA* { return o.is_none() ? nullptr : o.cast<GA*>(); };
    py::class_<Ledger>(m, "RunLedger",
                       "campaign.py's account() in C++ (include/mootation/run_ledger.hpp): the "
                       "counts, the failures, the first feasible evaluation and every row offered to "
                       "the run's archives, read at exactly the counts set_reads names.")
        .def(py::init([archive_of](std::size_t n_obj, std::size_t n_vars, bool constrained,
                                   const py::object& archive, const py::object& oracle,
                                   const py::object& all, const py::object& second) {
                 auto* inc = all.is_none() ? nullptr : all.cast<IF*>();
                 auto* sarc = second.is_none() ? nullptr : second.cast<SA*>();
                 return new Ledger{mootation::RunLedger(n_obj, n_vars, constrained, archive_of(archive),
                                                        archive_of(oracle), inc, sarc),
                                   n_vars, py::make_tuple(archive, oracle, all, second), {}, {}, {}, {}};
             }),
             py::arg("n_obj"), py::arg("n_vars"), py::arg("constrained"), py::arg("archive"),
             py::arg("oracle"), py::arg("all"), py::arg("second"))
        .def("set_reads",
             [](Ledger& a, std::vector<long long> at, std::vector<long long> grid) { a.L.set_reads(std::move(at), grid); },
             py::arg("at"), py::arg("grid"),
             "The counts the caller reads at, and those of everything evaluated's grid.")
        .def("begin",
             [](Ledger& a, const Arr& X, const Arr& F, const py::object& G, const py::object& T) {
                 Rows f = rows_of(F, "RunLedger.begin: F"), x = rows_of(X, "RunLedger.begin: X");
                 if (x.n != f.n || x.m != a.d)
                     throw std::invalid_argument("RunLedger.begin: X must be n x n_vars, a row per row of F");
                 a.X = std::move(x.v);
                 a.F = std::move(f.v);
                 std::size_t c = 0;
                 a.G.clear(); a.T.clear();
                 if (!G.is_none()) {
                     Rows g = rows_of(Arr::ensure(G), "RunLedger.begin: G");
                     if (g.n != f.n) throw std::invalid_argument("RunLedger.begin: G needs a row per row of F");
                     c = g.m;
                     a.G = std::move(g.v);
                 }
                 if (!T.is_none()) {
                     Rows t = rows_of(Arr::ensure(T), "RunLedger.begin: T");
                     if (t.n != f.n || t.m != f.m) throw std::invalid_argument("RunLedger.begin: T must be as F");
                     a.T = std::move(t.v);
                 }
                 a.L.begin(a.X.data(), a.F.data(), f.n, c ? a.G.data() : nullptr, c,
                           T.is_none() ? nullptr : a.T.data());
             },
             py::arg("X"), py::arg("F"), py::arg("G") = py::none(), py::arg("T") = py::none(),
             "A batch of rows in the order evaluated: X the variables the archives keep "
             "(a categorical variable's index), F, G and T as account() takes them.")
        .def("next",
             [](Ledger& a) {
                 py::gil_scoped_release unlock;
                 return a.L.next();
             },
             "Offer the rows up to the next count read in the batch and return it; -1 at its end.")
        .def_property_readonly("fe", [](const Ledger& a) { return a.L.fe(); })
        .def_property_readonly("n_failed", [](const Ledger& a) { return a.L.n_failed(); })
        .def_property_readonly("failed_fe", [](const Ledger& a) { return a.L.failed_fe(); })
        .def_property_readonly("first_feasible",
                               [](const Ledger& a) -> py::object {
                                   const long long v = a.L.first_feasible();
                                   return v < 0 ? py::object(py::none()) : py::object(py::int_(v));
                               })
        .def("all_at",
             [](const Ledger& a, long long b) -> py::object {
                 auto it = a.L.all_at().find(b);
                 if (it == a.L.all_at().end()) return py::none();
                 return py::make_tuple(it->second.first, it->second.second);
             },
             py::arg("b"), "(IGD+, epsilon) of everything evaluated after exactly b evaluations, or None.");
}

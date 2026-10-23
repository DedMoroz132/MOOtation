// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// _core.detmath (task 14 of 2026-10-20, §2): the functions of
// include/mootation/detmath.hpp — exp, log, pow, sin, cos, tan, acos, hypot,
// and asin and atan (task 17, supplement 3, stage 3a) — for a float or a
// NumPy array (elementwise, the result in the argument's shape). Python's math and NumPy call the C library, whose last bit may
// differ between platforms; a reference that must equal the core bit for bit
// takes these (python/test_run_layer.py, the SDR and DSS references), and
// tools/detmath/check.py measures their accuracy.
// ============================================================================

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mootation/detmath.hpp>

namespace py = pybind11;
namespace dm = mootation::dm;

void bind_detmath(py::module_& m)
{
    py::module_ d = m.def_submodule(
        "detmath",
        "The library's own elementary functions (include/mootation/detmath.hpp): the same\n"
        "bits on every platform. Each takes a float or an array (elementwise).");
    d.def("exp", py::vectorize([](double x) { return dm::exp(x); }), py::arg("x"), "e^x");
    d.def("log", py::vectorize([](double x) { return dm::log(x); }), py::arg("x"), "natural logarithm");
    d.def("pow", py::vectorize([](double x, double y) { return dm::pow(x, y); }), py::arg("x"),
          py::arg("y"), "x^y");
    d.def("sin", py::vectorize([](double x) { return dm::sin(x); }), py::arg("x"),
          "sine, |x| <= 2^20 pi/2");
    d.def("cos", py::vectorize([](double x) { return dm::cos(x); }), py::arg("x"),
          "cosine, |x| <= 2^20 pi/2");
    d.def("tan", py::vectorize([](double x) { return dm::tan(x); }), py::arg("x"),
          "tangent, |x| <= 2^20 pi/2");
    d.def("acos", py::vectorize([](double x) { return dm::acos(x); }), py::arg("x"), "arc cosine");
    d.def("asin", py::vectorize([](double x) { return dm::asin(x); }), py::arg("x"), "arc sine");
    d.def("atan", py::vectorize([](double x) { return dm::atan(x); }), py::arg("x"), "arc tangent");
    d.def("hypot", py::vectorize([](double x, double y) { return dm::hypot(x, y); }), py::arg("x"),
          py::arg("y"), "sqrt(x^2 + y^2) without overflow or underflow in between");
}

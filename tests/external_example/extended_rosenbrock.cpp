// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// An external problem's folder as an example and for tests/test_external.cpp
// (task 24, §24.7.1; include/mootation/campaign/external.hpp): this file and
// problems.toml beside it. Built into the engine with
//
//     cmake -DMOOTATION_EXTERNAL_PROBLEMS=<this folder> ...
//
// The family: the extended Rosenbrock function as residuals (Moré, Garbow,
// Hillstrom 1981, problem 21), n even,
//     r_{2i} = s (x_{2i+1} - x_{2i}^2),  r_{2i+1} = 1 - x_{2i},  f = 1/2 ||r||^2,
// s the row's params.scale (10 in the paper). At n = 2 and s = 10 it is MGH's
// problem 1, Rosenbrock's: the same residuals, Jacobian and sum, in the same
// order, so that a run on the row ext_rosenbrock_n02_x1 gives the bits of a
// run on MGH01_n02_m02_x1_1D (tests/test_external.cpp).
// ============================================================================

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/campaign/external.hpp>

namespace {

class ExtendedRosenbrock : public mootation::suite::Native {
public:
    ExtendedRosenbrock(int n, double scale) : scale_(scale)
    {
        n_var = n;
        n_obj = 1;
        n_res = n;
        has_jacobian = true;
    }

    // the residuals, and (J not null) their Jacobian, n_res rows of n_var
    void residuals(const double* x, double* r, double* J) const override
    {
        const std::size_t n = static_cast<std::size_t>(n_var);
        for (std::size_t i = 0; i + 1 < n; i += 2) {
            r[i] = scale_ * (x[i + 1] - x[i] * x[i]);
            r[i + 1] = 1.0 - x[i];
        }
        if (J == nullptr) return;
        for (std::size_t k = 0; k < n * n; ++k) J[k] = 0.0;
        for (std::size_t i = 0; i + 1 < n; i += 2) {
            J[i * n + i] = -2.0 * scale_ * x[i];
            J[i * n + i + 1] = scale_;
            J[(i + 1) * n + i] = -1.0;
        }
    }

    void evaluate(const double* x, double* f, double*) const override
    {
        std::vector<double> r(static_cast<std::size_t>(n_res));
        residuals(x, r.data(), nullptr);
        double sum = 0.0;
        for (double v : r) sum += v * v;
        f[0] = 0.5 * sum;
    }

private:
    double scale_;
};

}   // namespace

void mootation_external_problems(mootation::campaign::ExternalProblems& reg)
{
    reg.add(
        "extended_rosenbrock_example",
        [](const mootation::json::Value& params, int n_var) -> std::shared_ptr<mootation::suite::Native> {
            if (n_var % 2 != 0) throw std::invalid_argument("n_var even");
            double scale = 10.0;
            if (const mootation::json::Value* s = params.find("scale"))
                scale = s->kind == mootation::json::Value::Kind::Float ? s->f : static_cast<double>(s->i);
            return std::make_shared<ExtendedRosenbrock>(n_var, scale);
        },
        "1");
}

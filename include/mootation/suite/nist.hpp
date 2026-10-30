#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The 27 nonlinear regression problems of NIST's Statistical Reference
// Datasets (StRD; NIST Standard Reference Data, SRD 140): a file of NIST's,
// read as it is (the starts, the certified parameters and residual sum of
// squares, the data), and the model the file names, written out here from
// the file's "Model:" line.
//
// Reading the file: "Data (lines A to B)" of the header gives the data lines;
// the lines "  bK = start1 start2 certified deviation" the parameters (in the
// order of K); "Residual Sum of Squares:" the certified sum. The residual is
// y - f(x; b), for Nelson log(y) - f(x1, x2; b) (its model is log[y] = ...).
// pi in ENSO and Roszman1 is the double nearest to pi; exp, log, pow, sin,
// cos, atan the library's own (dm::, detmath.hpp): the same bits everywhere.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"

namespace mootation::suite::nist {

using Vec = std::vector<double>;

struct Dataset {
    std::string name;                       // the file's name without .dat
    std::size_t n = 0;                      // parameters
    Vec start1, start2, certified, deviation;
    double rss = 0.0;                       // certified residual sum of squares
    Vec y;
    std::vector<Vec> x;                     // x[i]: the predictors of observation i (Nelson: two)
};

inline const char* const kNames[27] = {
    "Misra1a", "Chwirut2", "Chwirut1", "Lanczos3", "Gauss1", "Gauss2", "DanWood", "Misra1b",
    "Kirby2", "Hahn1", "Nelson", "MGH17", "Lanczos1", "Lanczos2", "Gauss3", "Misra1c", "Misra1d",
    "Roszman1", "ENSO", "MGH09", "Thurber", "BoxBOD", "Rat42", "MGH10", "Eckerle4", "Rat43", "Bennett5"};

// path: the .dat file
inline Dataset read(const std::string& path, const std::string& name)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("nist: cannot open " + path);
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        lines.push_back(l);
    }
    Dataset d;
    d.name = name;
    std::size_t first = 0, last = 0;
    for (const std::string& l : lines) {
        const std::size_t p = l.find("Data  ");
        if (p != std::string::npos && l.find("(lines") != std::string::npos && !first) {
            std::istringstream s(l.substr(l.find("(lines") + 6));
            std::string to;
            s >> first >> to >> last;
        }
        const std::size_t b = l.find_first_not_of(' ');
        if (b != std::string::npos && l[b] == 'b' && l.find('=') != std::string::npos &&
            l.find("Parameter") == std::string::npos) {
            std::istringstream s(l.substr(l.find('=') + 1));
            double v[4];
            if (s >> v[0] >> v[1] >> v[2] >> v[3]) {
                d.start1.push_back(v[0]);
                d.start2.push_back(v[1]);
                d.certified.push_back(v[2]);
                d.deviation.push_back(v[3]);
            }
        }
        if (l.find("Residual Sum of Squares:") != std::string::npos)
            d.rss = std::stod(l.substr(l.find(':') + 1));
    }
    if (!first || last < first || last > lines.size()) throw std::runtime_error("nist: no data lines in " + path);
    for (std::size_t i = first; i <= last; ++i) {
        std::istringstream s(lines[i - 1]);
        double v;
        Vec row;
        while (s >> v) row.push_back(v);
        if (row.size() < 2) throw std::runtime_error("nist: a short data line in " + path);
        d.y.push_back(row[0]);
        d.x.emplace_back(row.begin() + 1, row.end());
    }
    d.n = d.certified.size();
    return d;
}

// f(x; b), the model of the file
inline double model(const std::string& name, const Vec& b, const Vec& xs)
{
    const double x = xs[0];
    const double pi = 3.141592653589793;
    if (name == "Misra1a" || name == "BoxBOD") return b[0] * (1.0 - dm::exp(-b[1] * x));
    if (name == "Chwirut1" || name == "Chwirut2") return dm::exp(-b[0] * x) / (b[1] + b[2] * x);
    if (name == "Lanczos1" || name == "Lanczos2" || name == "Lanczos3")
        return b[0] * dm::exp(-b[1] * x) + b[2] * dm::exp(-b[3] * x) + b[4] * dm::exp(-b[5] * x);
    if (name == "Gauss1" || name == "Gauss2" || name == "Gauss3") {
        const double u = (x - b[3]) / b[4], v = (x - b[6]) / b[7];
        return b[0] * dm::exp(-b[1] * x) + b[2] * dm::exp(-u * u) + b[5] * dm::exp(-v * v);
    }
    if (name == "DanWood") return b[0] * dm::pow(x, b[1]);
    if (name == "Misra1b") return b[0] * (1.0 - dm::pow(1.0 + b[1] * x / 2.0, -2.0));
    if (name == "Misra1c") return b[0] * (1.0 - dm::pow(1.0 + 2.0 * b[1] * x, -0.5));
    if (name == "Misra1d") return b[0] * b[1] * x / (1.0 + b[1] * x);
    if (name == "Kirby2") return (b[0] + b[1] * x + b[2] * x * x) / (1.0 + b[3] * x + b[4] * x * x);
    if (name == "Hahn1" || name == "Thurber")
        return (b[0] + b[1] * x + b[2] * x * x + b[3] * x * x * x) / (1.0 + b[4] * x + b[5] * x * x + b[6] * x * x * x);
    if (name == "Nelson") return b[0] - b[1] * xs[0] * dm::exp(-b[2] * xs[1]);
    if (name == "MGH17") return b[0] + b[1] * dm::exp(-x * b[3]) + b[2] * dm::exp(-x * b[4]);
    if (name == "Roszman1") return b[0] - b[1] * x - dm::atan(b[2] / (x - b[3])) / pi;
    if (name == "ENSO") {
        const double a = 2.0 * pi * x;
        return b[0] + b[1] * dm::cos(a / 12.0) + b[2] * dm::sin(a / 12.0) + b[4] * dm::cos(a / b[3]) +
               b[5] * dm::sin(a / b[3]) + b[7] * dm::cos(a / b[6]) + b[8] * dm::sin(a / b[6]);
    }
    if (name == "MGH09") return b[0] * (x * x + x * b[1]) / (x * x + x * b[2] + b[3]);
    if (name == "Rat42") return b[0] / (1.0 + dm::exp(b[1] - b[2] * x));
    if (name == "MGH10") return b[0] * dm::exp(b[1] / (x + b[2]));
    if (name == "Eckerle4") {
        const double u = (x - b[2]) / b[1];
        return (b[0] / b[1]) * dm::exp(-0.5 * u * u);
    }
    if (name == "Rat43") return b[0] / dm::pow(1.0 + dm::exp(b[1] - b[2] * x), 1.0 / b[3]);
    if (name == "Bennett5") return b[0] * dm::pow(b[1] + x, -1.0 / b[2]);
    throw std::invalid_argument("nist: no model for " + name);
}

// the residuals y_i - f(x_i; b) (Nelson: log y_i - f)
inline void residuals(const Dataset& d, const Vec& b, Vec& r)
{
    r.resize(d.y.size());
    const bool log_y = d.name == "Nelson";
    for (std::size_t i = 0; i < d.y.size(); ++i)
        r[i] = (log_y ? dm::log(d.y[i]) : d.y[i]) - model(d.name, b, d.x[i]);
}

}   // namespace mootation::suite::nist

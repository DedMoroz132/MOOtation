#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The bit-string problems of the runtime theory, as benchmarks/binary.py
// evaluates them: OneMinMax ("OMM"), LOTZ and OneJumpZeroJump_k ("OJZJ", k
// the spec's), n bits, two objectives, the negatives of the maximized ones
// (a 0 among them -0.0, as Python's -float(0)). A variable is a bit where it
// is not 0. Integer arithmetic alone: the same values on every platform.
// ============================================================================

#include <string>

namespace mootation::suite::bits {

enum class Kind { OMM, LOTZ, OJZJ };

inline bool parse(const std::string& family, Kind& kind)
{
    if (family == "OMM") kind = Kind::OMM;
    else if (family == "LOTZ") kind = Kind::LOTZ;
    else if (family == "OJZJ") kind = Kind::OJZJ;
    else return false;
    return true;
}

// x the n variables; f the two objectives; k OJZJ's jump
inline void evaluate(Kind kind, const double* x, int n, int k, double* f)
{
    int ones = 0;
    for (int i = 0; i < n; ++i) ones += x[i] != 0.0;
    const int zeros = n - ones;
    double f1 = 0.0, f2 = 0.0;
    if (kind == Kind::OMM) {
        f1 = zeros;
        f2 = ones;
    } else if (kind == Kind::LOTZ) {
        int lo = 0;
        while (lo < n && x[lo] != 0.0) ++lo;                 // the leading ones
        int tz = 0;
        while (tz < n && x[n - 1 - tz] == 0.0) ++tz;         // the trailing zeros
        f1 = lo;
        f2 = tz;
    } else {
        f1 = (ones <= n - k || ones == n) ? k + ones : n - ones;
        f2 = (zeros <= n - k || zeros == n) ? k + zeros : n - zeros;
    }
    f[0] = -f1;
    f[1] = -f2;
}

}   // namespace mootation::suite::bits

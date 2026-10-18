#pragma once
// SPDX-License-Identifier: Apache-2.0 (this file's code); the direction
// numbers below are Joe & Kuo's, under their licence, reproduced here:
//
// -----------------------------------------------------------------------------
// Licence pertaining to sobol.cc and the accompanying sets of direction numbers
//
// -----------------------------------------------------------------------------
// Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the names of the copyright holders nor the names of the
//       University of New South Wales and the University of Waikato
//       and its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mootation::sobol {

// ============================================================================
// The Sobol sequence (Bratley & Fox's gray-code construction, as Joe & Kuo's
// sobol.cc builds it) with the direction numbers of S. Joe, F. Y. Kuo,
// "Constructing Sobol sequences with better two-dimensional projections",
// SIAM J. Sci. Comput. 30:2635-2654, 2008, criterion D(6), file
// new-joe-kuo-6.21201 (python/mootation/_sobol/, which holds all 21 201
// dimensions): its first 100 dimensions, the first being van der Corput's.
// Points in [0, 1)^d with 32-bit precision, unscrambled, from index 1 (the
// origin, index 0, is left out). For multiglods.hpp's search step.
// ============================================================================

struct Poly { int s, a; std::vector<unsigned> m; };

// dimensions 2 ... 100: (degree s, polynomial a, initial direction numbers m_i)
inline const std::vector<Poly>& table()
{
    static const std::vector<Poly> t = {
    {1, 0, {1}},
    {2, 1, {1, 3}},
    {3, 1, {1, 3, 1}},
    {3, 2, {1, 1, 1}},
    {4, 1, {1, 1, 3, 3}},
    {4, 4, {1, 3, 5, 13}},
    {5, 2, {1, 1, 5, 5, 17}},
    {5, 4, {1, 1, 5, 5, 5}},
    {5, 7, {1, 1, 7, 11, 19}},
    {5, 11, {1, 1, 5, 1, 1}},
    {5, 13, {1, 1, 1, 3, 11}},
    {5, 14, {1, 3, 5, 5, 31}},
    {6, 1, {1, 3, 3, 9, 7, 49}},
    {6, 13, {1, 1, 1, 15, 21, 21}},
    {6, 16, {1, 3, 1, 13, 27, 49}},
    {6, 19, {1, 1, 1, 15, 7, 5}},
    {6, 22, {1, 3, 1, 15, 13, 25}},
    {6, 25, {1, 1, 5, 5, 19, 61}},
    {7, 1, {1, 3, 7, 11, 23, 15, 103}},
    {7, 4, {1, 3, 7, 13, 13, 15, 69}},
    {7, 7, {1, 1, 3, 13, 7, 35, 63}},
    {7, 8, {1, 3, 5, 9, 1, 25, 53}},
    {7, 14, {1, 3, 1, 13, 9, 35, 107}},
    {7, 19, {1, 3, 1, 5, 27, 61, 31}},
    {7, 21, {1, 1, 5, 11, 19, 41, 61}},
    {7, 28, {1, 3, 5, 3, 3, 13, 69}},
    {7, 31, {1, 1, 7, 13, 1, 19, 1}},
    {7, 32, {1, 3, 7, 5, 13, 19, 59}},
    {7, 37, {1, 1, 3, 9, 25, 29, 41}},
    {7, 41, {1, 3, 5, 13, 23, 1, 55}},
    {7, 42, {1, 3, 7, 3, 13, 59, 17}},
    {7, 50, {1, 3, 1, 3, 5, 53, 69}},
    {7, 55, {1, 1, 5, 5, 23, 33, 13}},
    {7, 56, {1, 1, 7, 7, 1, 61, 123}},
    {7, 59, {1, 1, 7, 9, 13, 61, 49}},
    {7, 62, {1, 3, 3, 5, 3, 55, 33}},
    {8, 14, {1, 3, 1, 15, 31, 13, 49, 245}},
    {8, 21, {1, 3, 5, 15, 31, 59, 63, 97}},
    {8, 22, {1, 3, 1, 11, 11, 11, 77, 249}},
    {8, 38, {1, 3, 1, 11, 27, 43, 71, 9}},
    {8, 47, {1, 1, 7, 15, 21, 11, 81, 45}},
    {8, 49, {1, 3, 7, 3, 25, 31, 65, 79}},
    {8, 50, {1, 3, 1, 1, 19, 11, 3, 205}},
    {8, 52, {1, 1, 5, 9, 19, 21, 29, 157}},
    {8, 56, {1, 3, 7, 11, 1, 33, 89, 185}},
    {8, 67, {1, 3, 3, 3, 15, 9, 79, 71}},
    {8, 70, {1, 3, 7, 11, 15, 39, 119, 27}},
    {8, 84, {1, 1, 3, 1, 11, 31, 97, 225}},
    {8, 97, {1, 1, 1, 3, 23, 43, 57, 177}},
    {8, 103, {1, 3, 7, 7, 17, 17, 37, 71}},
    {8, 115, {1, 3, 1, 5, 27, 63, 123, 213}},
    {8, 122, {1, 1, 3, 5, 11, 43, 53, 133}},
    {9, 8, {1, 3, 5, 5, 29, 17, 47, 173, 479}},
    {9, 13, {1, 3, 3, 11, 3, 1, 109, 9, 69}},
    {9, 16, {1, 1, 1, 5, 17, 39, 23, 5, 343}},
    {9, 22, {1, 3, 1, 5, 25, 15, 31, 103, 499}},
    {9, 25, {1, 1, 1, 11, 11, 17, 63, 105, 183}},
    {9, 44, {1, 1, 5, 11, 9, 29, 97, 231, 363}},
    {9, 47, {1, 1, 5, 15, 19, 45, 41, 7, 383}},
    {9, 52, {1, 3, 7, 7, 31, 19, 83, 137, 221}},
    {9, 55, {1, 1, 1, 3, 23, 15, 111, 223, 83}},
    {9, 59, {1, 1, 5, 13, 31, 15, 55, 25, 161}},
    {9, 62, {1, 1, 3, 13, 25, 47, 39, 87, 257}},
    {9, 67, {1, 1, 1, 11, 21, 53, 125, 249, 293}},
    {9, 74, {1, 1, 7, 11, 11, 7, 57, 79, 323}},
    {9, 81, {1, 1, 5, 5, 17, 13, 81, 3, 131}},
    {9, 82, {1, 1, 7, 13, 23, 7, 65, 251, 475}},
    {9, 87, {1, 3, 5, 1, 9, 43, 3, 149, 11}},
    {9, 91, {1, 1, 3, 13, 31, 13, 13, 255, 487}},
    {9, 94, {1, 3, 3, 1, 5, 63, 89, 91, 127}},
    {9, 103, {1, 1, 3, 3, 1, 19, 123, 127, 237}},
    {9, 104, {1, 1, 5, 7, 23, 31, 37, 243, 289}},
    {9, 109, {1, 1, 5, 11, 17, 53, 117, 183, 491}},
    {9, 122, {1, 1, 1, 5, 1, 13, 13, 209, 345}},
    {9, 124, {1, 1, 3, 15, 1, 57, 115, 7, 33}},
    {9, 137, {1, 3, 1, 11, 7, 43, 81, 207, 175}},
    {9, 138, {1, 3, 1, 1, 15, 27, 63, 255, 49}},
    {9, 143, {1, 3, 5, 3, 27, 61, 105, 171, 305}},
    {9, 145, {1, 1, 5, 3, 1, 3, 57, 249, 149}},
    {9, 152, {1, 1, 3, 5, 5, 57, 15, 13, 159}},
    {9, 157, {1, 1, 1, 11, 7, 11, 105, 141, 225}},
    {9, 167, {1, 3, 3, 5, 27, 59, 121, 101, 271}},
    {9, 173, {1, 3, 5, 9, 11, 49, 51, 59, 115}},
    {9, 176, {1, 1, 7, 1, 23, 45, 125, 71, 419}},
    {9, 181, {1, 1, 3, 5, 23, 5, 105, 109, 75}},
    {9, 182, {1, 1, 7, 15, 7, 11, 67, 121, 453}},
    {9, 185, {1, 3, 7, 3, 9, 13, 31, 27, 449}},
    {9, 191, {1, 3, 1, 15, 19, 39, 39, 89, 15}},
    {9, 194, {1, 1, 1, 1, 1, 33, 73, 145, 379}},
    {9, 199, {1, 3, 1, 15, 15, 43, 29, 13, 483}},
    {9, 218, {1, 1, 7, 3, 19, 27, 85, 131, 431}},
    {9, 220, {1, 3, 3, 3, 5, 35, 23, 195, 349}},
    {9, 227, {1, 3, 3, 7, 9, 27, 39, 59, 297}},
    {9, 229, {1, 1, 3, 9, 11, 17, 13, 241, 157}},
    {9, 230, {1, 3, 7, 15, 25, 57, 33, 189, 213}},
    {9, 234, {1, 1, 7, 1, 9, 55, 73, 83, 217}},
    {9, 236, {1, 3, 3, 13, 19, 27, 23, 113, 249}},
    {9, 241, {1, 3, 5, 3, 23, 43, 3, 253, 479}},
    {9, 244, {1, 1, 5, 5, 11, 5, 45, 117, 217}}
    };
    return t;
}

constexpr int max_dims = 100;

class Generator {
public:
    explicit Generator(int dims) : d_(dims), x_(static_cast<std::size_t>(dims), 0u), V_(static_cast<std::size_t>(dims))
    {
        if (dims < 1 || dims > max_dims)
            throw std::invalid_argument("sobol: 1 to 100 dimensions");
        for (int b = 1; b <= 32; ++b) V_[0].push_back(1u << (32 - b));      // van der Corput
        for (int j = 1; j < dims; ++j) {
            const Poly& p = table()[static_cast<std::size_t>(j - 1)];
            auto& V = V_[static_cast<std::size_t>(j)];
            V.resize(33);
            for (int i = 1; i <= p.s; ++i) V[static_cast<std::size_t>(i)] = p.m[static_cast<std::size_t>(i - 1)] << (32 - i);
            for (int i = p.s + 1; i <= 32; ++i) {
                unsigned v = V[static_cast<std::size_t>(i - p.s)] ^ (V[static_cast<std::size_t>(i - p.s)] >> p.s);
                for (int k = 1; k <= p.s - 1; ++k)
                    v ^= (((static_cast<unsigned>(p.a) >> (p.s - 1 - k)) & 1u) * V[static_cast<std::size_t>(i - k)]);
                V[static_cast<std::size_t>(i)] = v;
            }
            V.erase(V.begin());                                            // V[0..31] = V_1..V_32
        }
    }

    // The next point (index 1, 2, ...).
    std::vector<double> next()
    {
        unsigned c = 0;                                                    // the rightmost zero bit of i
        for (std::uint64_t v = i_; v & 1u; v >>= 1) ++c;
        ++i_;
        std::vector<double> out(static_cast<std::size_t>(d_));
        for (int j = 0; j < d_; ++j) {
            x_[static_cast<std::size_t>(j)] ^= V_[static_cast<std::size_t>(j)][c];
            out[static_cast<std::size_t>(j)] = static_cast<double>(x_[static_cast<std::size_t>(j)]) / 4294967296.0;
        }
        return out;
    }

private:
    int d_;
    std::uint64_t i_ = 0;
    std::vector<unsigned> x_;
    std::vector<std::vector<unsigned>> V_;
};

}   // namespace mootation::sobol

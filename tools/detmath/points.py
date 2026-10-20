# SPDX-License-Identifier: Apache-2.0
"""The test points of detmath.hpp's functions, made from integers alone so
that tests/test_detmath.cpp makes the very same doubles (task 14 of
2026-10-20, supplement 2, D7.4). Keep the two in step: every step here has
its twin in the C++ test (struct Points).

A generator: SplitMix64 (64-bit integer arithmetic). A double from it is made
in one of three exact ways:
  unit(g)              (z >> 11) * 2^-53, in [0, 1)
  between(g, a, b)     a + (b - a) * unit(g)  (the same two roundings in C++)
  binade(g, e0, e1)    a random significand with the binary exponent field
                       drawn from [e0, e1] (0 = subnormal), positive
Signs and offsets by ulps (next(x, k): k steps of nextafter) as below.

points(f, n, seed) gives n points of f (a list of tuples: (x,) or (x, y)),
in fixed blocks: each function's blocks and their shares are in BLOCKS.
"""

import math
import struct

MASK = (1 << 64) - 1
TRIG_MAX = 1647099.3291652855         # 2^20 pi/2 rounded (detmath.hpp, kTrigMax)
PIO2 = 1.5707963267948966             # pi/2 rounded
EXP_OVER = float.fromhex("0x1.62e42fefa39efp+9")
EXP_UNDER = float.fromhex("-0x1.74910d52d3051p+9")


class SplitMix64:
    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n):
        """An integer in [0, n) (modulo: the bias does not matter here)."""
        return self.next() % n


def from_bits(u):
    return struct.unpack("<d", struct.pack("<Q", u & MASK))[0]


def to_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def unit(g):
    return (g.next() >> 11) * 2.0 ** -53


def between(g, a, b):
    return a + (b - a) * unit(g)


def binade(g, e0, e1):
    e = e0 + g.below(e1 - e0 + 1)
    return from_bits((e << 52) | (g.next() & ((1 << 52) - 1)))


def signed(g, x):
    return -x if g.next() & 1 else x


def step(x, k):
    """k steps of nextafter from x (k may be negative)."""
    for _ in range(abs(k)):
        x = math.nextafter(x, math.inf if k > 0 else -math.inf)
    return x


def _trig(g, block):
    if block == 0:
        return (between(g, -6.283185307179586, 6.283185307179586),)
    if block == 1:
        return (signed(g, binade(g, 1023 - 30, 1023 + 19)),)        # 2^-30 .. 2^20
    if block == 2:
        k = 1 + g.below(1 << 20)
        return (step(k * PIO2, g.below(7) - 3),)                   # near k pi/2
    if block == 3:
        return (between(g, -1e4, 1e4),)
    return (signed(g, binade(g, 0, 1023 - 26)),)                   # tiny, subnormal


def _acos(g, block):
    if block == 0:
        return (between(g, -1.0, 1.0),)
    if block == 1:
        return (signed(g, 1.0 - binade(g, 1023 - 53, 1023 - 2)),)  # near +-1
    if block == 2:
        return (signed(g, step(0.5, g.below(2001) - 1000)),)       # near +-1/2
    return (signed(g, binade(g, 0, 1023 - 2)),)                    # small


def _exp(g, block):
    if block == 0:
        return (between(g, EXP_UNDER, EXP_OVER),)
    if block == 1:
        return (signed(g, binade(g, 1023 - 60, 1023 + 8)),)
    if block == 2:
        return (between(g, -20.0, 20.0),)
    if block == 3:                                                 # near the ends
        if g.next() & 1:
            return (step(EXP_OVER, -int(g.below(1000))),)
        return (step(EXP_UNDER, int(g.below(1000))),)
    return (between(g, -745.0, -700.0),)                           # subnormal results


def _log(g, block):
    if block == 0:
        x = binade(g, 0, 2046)                                     # every positive finite
        return (x if x > 0.0 else 5e-324,)
    if block == 1:
        return (1.0 + signed(g, binade(g, 1023 - 52, 1023 - 4)),)  # near 1
    if block == 2:
        return (between(g, 0.5, 2.0),)
    return (between(g, 0.0, 100.0),)


def _pow(g, block):
    if block == 0:
        return (between(g, 0.0, 10.0), between(g, -50.0, 50.0))
    if block == 1:
        return (binade(g, 1023 - 40, 1023 + 40), between(g, -20.0, 20.0))
    if block == 2:
        return (1.0 + signed(g, binade(g, 1023 - 52, 1023 - 20)), between(g, -1e8, 1e8))
    if block == 3:
        return (-between(g, 0.1, 10.0), float(g.below(61)) - 30.0)  # negative x, integer y
    if block == 4:
        return (between(g, 0.0, 1.0), between(g, 0.0, 1.0))
    return (between(g, 1.0, 6e14), between(g, -41.0, -16.0))         # as SBX


def _hypot(g, block):
    if block == 0:
        x = signed(g, binade(g, 0, 2046))
        e = (to_bits(abs(x)) >> 52) - (g.below(61) - 30)
        e = min(max(e, 0), 2046)
        return (x, signed(g, binade(g, e, e)))
    return (between(g, -1e4, 1e4), between(g, -1e4, 1e4))


# (generator, shares of the blocks in tenths)
BLOCKS = {
    "sin": (_trig, (3, 3, 2, 1, 1)),
    "cos": (_trig, (3, 3, 2, 1, 1)),
    "tan": (_trig, (3, 3, 2, 1, 1)),
    "acos": (_acos, (4, 2, 2, 2)),
    "exp": (_exp, (4, 2, 2, 1, 1)),
    "log": (_log, (4, 2, 2, 2)),
    "pow": (_pow, (3, 2, 2, 1, 1, 1)),
    "hypot": (_hypot, (5, 5)),
}
SEEDS = {"exp": 1, "log": 2, "pow": 3, "sin": 4, "cos": 5, "tan": 6, "acos": 7, "hypot": 8}


def points(f, n):
    gen, shares = BLOCKS[f]
    g = SplitMix64(SEEDS[f])
    out = []
    for block, share in enumerate(shares):
        for _ in range(n * share // 10):
            out.append(gen(g, block))
    return out

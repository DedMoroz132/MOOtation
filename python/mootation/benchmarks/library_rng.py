# SPDX-License-Identifier: Apache-2.0
"""The library's generator in Python (task 17, supplement 5, §3): the numbers
include/mootation/random.hpp makes, drawn here for the cross-check of what the
C++ suite draws (suite/instances.hpp): the rotations of the rotated problems
and the bbob instances.

  seed(*key)     SplitMix64 chained over the key's words, its low 32 bits
  LibraryRng     a std::mt19937 seeded with it (init_genrand: NumPy's MT19937
                 with its legacy seeding is that generator) and rnd::'s ways
                 of making numbers of its 32-bit words: uniform01
                 (genrand_res53), below (Lemire), normal01 (Marsaglia's polar
                 method), shuffle (Fisher-Yates from the last element down)
  haar(rng, k)   a k x k rotation from the Haar distribution: the Q of the QR
                 factorization of a matrix of normal draws (row after row), its
                 columns times the signs of R's diagonal (Higham 2002, §28.3)

The normals take math.log where C++ takes dm::log, and the QR is NumPy's
(LAPACK's Householder) where C++ writes its own: the matrices agree to a few
ulps, not to the bit.
"""

from __future__ import annotations

import math

import numpy as np

_MASK = (1 << 64) - 1


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def seed(*key: int) -> int:
    """The seed of an instance's generator (instances.hpp, seed)."""
    h = 0
    for w in key:
        h = _splitmix64(h ^ (int(w) & _MASK))
    return h & 0xFFFFFFFF


class LibraryRng:
    """A std::mt19937 seeded with `seed32` and rnd::'s draws of it."""

    def __init__(self, seed32: int):
        self.bg = np.random.MT19937()
        self.bg._legacy_seeding(int(seed32) & 0xFFFFFFFF)

    def word32(self) -> int:
        return int(self.bg.random_raw())

    def uniform01(self) -> float:
        a, b = self.word32() >> 5, self.word32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def uniform(self, lo: float, hi: float) -> float:
        """lo + (hi - lo) U."""
        return lo + (hi - lo) * self.uniform01()

    def below(self, n: int) -> int:
        """[0, n), n <= 2^32: Lemire's multiply-and-reject."""
        m = self.word32() * n
        low = m & 0xFFFFFFFF
        if low < n:
            t = (2 ** 32 - n) % n
            while low < t:
                m = self.word32() * n
                low = m & 0xFFFFFFFF
        return m >> 32

    def normal01(self) -> float:
        while True:
            u = 2.0 * self.uniform01() - 1.0
            v = 2.0 * self.uniform01() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                return u * math.sqrt(-2.0 * math.log(s) / s)

    def shuffle(self, items: list) -> list:
        """Fisher-Yates from the last element down, in place; returns items."""
        for i in range(len(items), 1, -1):
            j = self.below(i)
            items[i - 1], items[j] = items[j], items[i - 1]
        return items


def crc32(name: str) -> int:
    import zlib
    return zlib.crc32(name.encode())


def haar(rng: LibraryRng, k: int) -> np.ndarray:
    """A k x k rotation from the Haar distribution (instances.hpp, haar)."""
    A = np.array([rng.normal01() for _ in range(k * k)], float).reshape(k, k)
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.where(np.diag(R) == 0.0, 1.0, np.diag(R)))

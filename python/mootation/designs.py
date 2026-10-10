# SPDX-License-Identifier: Apache-2.0
"""Designs of experiments: where a run puts its first points.

    design(kind, n, bounds, seed)   n points of the box
    start_points(F, k)              which k points of an evaluated design an
                                    algorithm starts from

At budgets of 100-1 000 evaluations a GA makes a few generations, and how its
first points lie may decide as much as what it does after (task 5 of
2026-10-01, run B1). The kinds:

  sobol    the first n points of a scrambled Sobol sequence
           (scipy.stats.qmc.Sobol, linear matrix scrambling with a digital
           shift, seeded), as sobol_search draws them
  lhs      a maximin Latin hypercube (MAXIMIN LHS below)
  halton_rand
           the first n points of a randomized Halton sequence (RANDOMIZED
           HALTON below), seeded; as halton_search draws them
  sobol_owen
           the first n points of the Sobol sequence under Owen's nested
           uniform scrambling (SOBOL-OWEN below), seeded; as sobol_owen_search
           draws them

(The uniform start is every algorithm's own: `init = "uniform"` leaves it to
the algorithm.)

minimize(init=...) evaluates a design and starts the algorithm from it; with
init_share > 0 the design is a plan of experiments larger than the population,
and start_points picks the population from it: the nondominated points first
(DSS among them when there are more), the rest by DSS among the others.

MAXIMIN LHS, after Morris & Mitchell, "Exploratory designs for computational
experiments", J. Statist. Plann. Inference 43 (1995) 381-402. A Latin
hypercube of n points in k variables gives each variable n levels, each used
once: every column of the n-by-k level matrix is a permutation of 0..n-1. Of
those, the search wants the maximin one: the largest smallest distance d1
between two points, then the fewest pairs J1 at d1, then the largest d2, and
so on (Section 2). It minimizes phi_p = (sum over the pairs of d^-p)^(1/p),
Eq. 2.1 (the sum over the distinct distances d_j of J_j d_j^-p is the same
sum), which ranks designs that way for p large enough, with Euclidean distance
taken squared as the paper does for it (Section 4), by the annealing of its
Fig. 1: a perturbation swaps two levels in a column; a better design is kept,
a worse one with probability exp(-(phi_try - phi)/t); after I_max
perturbations without a new best design the temperature drops by FAC_t = 0.95,
and the search ends at a temperature that accepts nothing. t0: on a design
whose C(n, 2) squared distances spread evenly over 50-150 % of the class's
mean, k n (n + 1) / 6 in levels, one of the smallest made one level smaller is
accepted with probability 0.99 (Section 4). The search runs once for each p of
Section 5 (1, 2, 5, 10, 20, 50, 100), and the best of the results by the
maximin order itself is the design. Two departures:
  * the paper's I_max, about 10 C(n, 2) k perturbations, and its open-ended
    temperature loop are out of reach beyond a few dozen points, and a
    perturbation costs O(n): I_max is that or 2 000, whichever is smaller, and
    a search stops after 20 000 perturbations, 2e7 / n above 1 000 points.
    That reaches the optima of the paper's catalog for small designs (Table
    2(A): n = 5, 8 and 9 at k = 2, the tests check two); the larger ones of
    Table 2(A) took 300 000 perturbations a search and the paper's I_max
    (n = 12 at k = 2, n = 6 at k = 4 in 3 seeds of 3, n = 10 at k = 3 in 1);
  * the paper's levels are i/(n - 1), 0 and 1 included and 1/2 at odd n, and its
    best designs at odd n hold the centre of the box itself (Table 3): exactly
    the optima of DTLZ1-4's and ZDT's distance variables. Here level i is the cell [i/n, (i + 1)/n) of
    McKay, Conover & Beckman's Latin hypercube (1979) and the point is drawn
    uniformly in its cell: the maximin arrangement of the cells is the paper's,
    the points are on no grid.

RANDOMIZED HALTON, after A. B. Owen, "A randomized Halton algorithm in R",
arXiv:1706.02808, 2017 (task 6 of 2026-10-09, item 12.3). Coordinate j of
point i (i = 0, 1, ...) is the radical inverse of i in the j-th prime base b
with each digit put through a permutation of its own: x_ij = sum_k
pi_jk(a_k(i)) b^-k, a_k(i) the k-th base-b digit of i, the pi_jk uniform
random permutations of 0..b-1, independent over j and k (Eq. 5). As many
digits as a double resolves: until 1 - b^-k rounds to 1 (Algorithm 1's loop).
The permutations of coordinate j come from numpy's default_rng([seed, j]):
the same seed the same plan, another seed another one. The sequence extends:
the first n points of a longer plan are the plan of n points, which is why
halton_search, unlike lhs_search, does not depend on its budget. At n = b^m
every coordinate of base b puts one point in each of its b^m equal
intervals; at other n the balance is incomplete. In many variables the bases
grow (the 20th prime is 71, the 50th 229), and where b exceeds n a coordinate
moves by its first digit alone and neighbouring coordinates line up (Owen
2017, Section 4); the permutations weaken that without removing it. Measured
2026-10-10 over 10 seeds (Pearson r between coordinates, scipy's centred L2
discrepancy), against uniform points: at d = 10 and n = 100, 300, 1 000 the
discrepancy is 0.026, 0.0062, 0.0012 against their 0.070, 0.024, 0.0073 and
the mean |r| 0.043, 0.014, 0.004 against 0.083, 0.047, 0.025. At d = 30 and
50 the plan is close to uniform points: discrepancy 2-15 % lower, mean |r|
2-45 % lower, and the largest |r| of neighbouring coordinates never above
theirs by more than 0.001 (0.21-0.24 at n = 100, 0.06-0.08 at 1 000) — no
lining-up beyond chance shows, and little of the low discrepancy is left.

SOBOL-OWEN (task 6, item 12.2). The Sobol sequence in base two, 32 bits: the
direction numbers of S. Joe and F. Y. Kuo, "Constructing Sobol sequences with
better two-dimensional projections", SIAM J. Sci. Comput. 30, 2008, file
new-joe-kuo-6.21201 (their recommended set; shipped in _sobol/ with its BSD
licence), the first coordinate the van der Corput sequence (every m_k = 1),
the recurrence and the Gray-code order of their "Notes on generating Sobol'
sequences" (2008) — unscrambled, the points are SciPy's
scipy.stats.qmc.Sobol(scramble=False), bit for bit. Each coordinate is then
scrambled by B. Burley, "Practical Hash-based Owen Scrambling", JCGT 9(4),
2020, Listing 2: reverse the 32 bits, apply the Laine-Karras permutation with
a key, reverse back — a nested uniform scramble, every digit permuted by a
function of the digits above it. The key of coordinate j is a splitmix64
hash of (seed, j), ours; Burley's sample-index shuffle (his Listing 3) is not
used: the plan is the first n points in order. The hash is a practical
stand-in for Owen's full nested scrambling — the permutations come from the
hash, not from independent random bits at every node (Burley, Section 5.2).
SciPy's own scramble (the kind "sobol") is a linear matrix scramble with a
digital shift (Matousek), not Owen's. At n = 2^m every coordinate has one
point in each of its 2^m equal intervals; the budgets 100, 300 and 1 000 are
not powers of two, and their first n points are not balanced so — the plan
is not rounded to one. Measured as the randomized Halton above (2026-10-10,
10 seeds): the mean |r| over the pairs of coordinates is below Halton's and
uniform points' at every d and n (0.003-0.053), and the discrepancy at or
below Halton's; but at n = 100 with d = 30 and 50, and at n = 300 with d = 50,
one pair of coordinates correlates to |r| 0.40, 0.75 and 0.26, above Halton's
0.31, 0.35 and 0.20 — pairs of the sequence itself, which SciPy's scrambled
Sobol shows alike and the scrambling keeps.
"""

from __future__ import annotations

import functools
import math
import warnings
from pathlib import Path

KINDS = ("sobol", "lhs", "halton_rand", "sobol_owen")
P_VALUES = (1, 2, 5, 10, 20, 50, 100)      # Morris & Mitchell 1995, Section 5
FAC_T = 0.95                               # Section 4: "0.90 and 0.95"
ACCEPT = 0.99                              # Section 4: t0 accepts a step of one level so
I_MAX_CAP = 2000                           # the departures (MAXIMIN LHS): I_max at most,
BUDGET, BUDGET_WORK = 20000, 2 * 10**7     # perturbations per search: BUDGET, BUDGET_WORK / n above


def design(kind: str, n: int, bounds, seed: int):
    """n points of the box `bounds` [(lo, hi), ...], as an (n, d) array."""
    import numpy as np
    if kind not in KINDS:
        raise ValueError(f"unknown design '{kind}'; known: {', '.join(KINDS)}")
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    if kind == "lhs":
        rng = np.random.default_rng(seed)
        L = maximin_lhs(n, len(lo), rng)
        U = (L + rng.random(L.shape)) / n          # a point drawn in each cell
        return lo + U * (hi - lo)
    if kind == "halton_rand":
        return lo + randomized_halton(n, len(lo), seed) * (hi - lo)
    if kind == "sobol_owen":
        return lo + sobol_owen(n, len(lo), seed) * (hi - lo)
    try:
        from scipy.stats import qmc
    except ImportError as e:                        # pragma: no cover - machine-dependent
        raise ImportError("a Sobol design needs SciPy: pip install scipy") from e
    with warnings.catch_warnings():                 # "balance properties ... power of 2"
        warnings.simplefilter("ignore")
        U = qmc.Sobol(len(lo), scramble=True, seed=seed).random(n)
    return lo + U * (hi - lo)


# ── randomized Halton ───────────────────────────────────────────────────────


def primes(d: int) -> list:
    """The first d primes."""
    out, k = [], 2
    while len(out) < d:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


def halton_digits(b: int) -> int:
    """How many base-b digits a double resolves: the k >= 1 for which
    1 - b^-k still differs from 1 (Owen 2017, Algorithm 1's loop)."""
    k, b2r = 0, 1.0 / b
    while 1.0 - b2r < 1.0:
        k += 1
        b2r /= b
    return k


def randomized_halton(n: int, d: int, seed: int, start: int = 0):
    """Points start .. start + n - 1 of the randomized Halton sequence in
    [0, 1)^d (RANDOMIZED HALTON above), as an (n, d) array."""
    import numpy as np
    X = np.empty((n, d))
    idx = np.arange(start, start + n, dtype=np.int64)
    for j, b in enumerate(primes(d)):
        rng = np.random.default_rng([seed, j])
        rest, scale, x = idx.copy(), 1.0, np.zeros(n)
        for perm in [rng.permutation(b) for _ in range(halton_digits(b))]:
            scale /= b
            x += perm[rest % b] * scale
            rest //= b
        X[:, j] = x
    return X


# ── Sobol with Owen's scrambling ────────────────────────────────────────────

SOBOL_BITS = 32
_JOE_KUO = Path(__file__).with_name("_sobol") / "new-joe-kuo-6.21201"
_MASK32 = 0xFFFFFFFF


@functools.lru_cache(maxsize=None)
def sobol_directions(d: int):
    """(d, 32) integers: row j holds the direction numbers v_{k,j} 2^32,
    k = 1..32, of coordinate j + 1 (Joe & Kuo's notes, Eq. 2: m_k = 2 a_1
    m_{k-1} xor ... xor 2^(s-1) a_{s-1} m_{k-s+1} xor 2^s m_{k-s} xor m_{k-s},
    v_k = m_k / 2^k)."""
    import numpy as np
    V = np.zeros((d, SOBOL_BITS), dtype=np.uint64)
    V[0] = [1 << (SOBOL_BITS - k) for k in range(1, SOBOL_BITS + 1)]
    if d > 1:
        with open(_JOE_KUO, encoding="ascii") as fh:
            next(fh)                                       # the header: d s a m_i
            for j in range(1, d):
                fields = fh.readline().split()
                s, a = int(fields[1]), int(fields[2])
                m = [int(v) for v in fields[3:3 + s]]
                for k in range(s + 1, SOBOL_BITS + 1):
                    mk = m[k - s - 1] ^ (m[k - s - 1] << s)
                    for t in range(1, s):
                        if (a >> (s - 1 - t)) & 1:       # a = (a_1 a_2 ... a_{s-1})_2
                            mk ^= m[k - t - 1] << t
                    m.append(mk)
                V[j] = [m[k - 1] << (SOBOL_BITS - k) for k in range(1, SOBOL_BITS + 1)]
    return V


def sobol_integers(n: int, d: int, start: int = 0):
    """Points start .. start + n - 1 of the Sobol sequence as 32-bit integers
    (the coordinate times 2^32), in Gray-code order (Joe & Kuo's notes, Eq. 4)."""
    import numpy as np
    V = sobol_directions(d)
    i = np.arange(start, start + n, dtype=np.uint64)
    gray = i ^ (i >> np.uint64(1))
    X = np.zeros((n, d), dtype=np.uint64)
    for k in range(SOBOL_BITS):
        X[((gray >> np.uint64(k)) & np.uint64(1)).astype(bool)] ^= V[:, k]
    return X


def _reverse_bits(x):
    """The 32 low bits of x reversed."""
    import numpy as np
    for shift, mask in ((1, 0x55555555), (2, 0x33333333), (4, 0x0F0F0F0F), (8, 0x00FF00FF),
                        (16, 0x0000FFFF)):
        s, m = np.uint64(shift), np.uint64(mask)
        x = ((x >> s) & m) | ((x & m) << s)
    return x & np.uint64(_MASK32)


def laine_karras(x, key: int):
    """The Laine-Karras permutation of 32-bit integers (Burley 2020, Listing 1)."""
    import numpy as np
    m = np.uint64(_MASK32)
    x = (x + np.uint64(key)) & m
    for c in (0x6c50b47c, 0xb82f1e52, 0xc7afe638, 0x8d22f6e6):
        x ^= (x * np.uint64(c)) & m
    return x


def nested_uniform_scramble(x, key: int):
    """Owen's nested uniform scrambling of 32-bit integers (Burley 2020, Listing 2)."""
    return _reverse_bits(laine_karras(_reverse_bits(x), key))


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def sobol_owen(n: int, d: int, seed: int, start: int = 0):
    """Points start .. start + n - 1 of the Owen-scrambled Sobol sequence in
    [0, 1)^d (SOBOL-OWEN above), as an (n, d) array."""
    import numpy as np
    X = sobol_integers(n, d, start)
    U = np.empty((n, d))
    for j in range(d):
        key = _splitmix64((seed & 0xFFFFFFFFFFFFFFFF) ^ _splitmix64(j + 1)) & _MASK32
        U[:, j] = nested_uniform_scramble(X[:, j], key) * 2.0 ** -SOBOL_BITS
    return U


# ── maximin Latin hypercubes ────────────────────────────────────────────────


def squared_distances(L):
    """The squared Euclidean distances between the rows of the level matrix L
    (MAXIMIN LHS: in levels), an (n, n) float array, infinite on the diagonal."""
    import numpy as np
    L = np.asarray(L, float)
    sq = (L * L).sum(axis=1)
    # |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, one matrix product; exact on integer
    # levels (below 2^53 for any design here)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (L @ L.T)
    np.fill_diagonal(D2, np.inf)
    return D2


def maximin_order(D2, depth: int = 8) -> tuple:
    """The key the maximin order sorts designs by, best first: (-d1, J1, -d2,
    J2, ...) over the `depth` smallest distinct distances (Section 2: the
    largest d1, then the fewest pairs at it, then the largest d2, ...)."""
    import numpy as np
    key, floor = [], -np.inf
    for _ in range(depth):
        d = float(D2.min(initial=np.inf, where=D2 > floor))
        if d == np.inf:
            break
        key += [-d, int((D2 == d).sum()) // 2]
        floor = d
    return tuple(key)


def neg_power(D, p: int):
    """D ** (-p) elementwise for an integer p >= 1, by repeated squaring and one
    division: IEEE multiplications, the same bits on every CPU. numpy's power
    dispatches to SIMD code (SVML on AVX-512) whose last bits differ from the
    scalar pow's, and the annealing below turns one such bit into another
    design; infinity and overflow give 0, as the power does."""
    import numpy as np
    out = np.ones_like(D)
    base = np.array(D, float)
    e = p
    with np.errstate(over="ignore", invalid="ignore"):
        while e:
            if e & 1:
                out = out * base
            e >>= 1
            if e:
                base = base * base
    with np.errstate(divide="ignore"):
        return 1.0 / out


def maximin_lhs(n: int, k: int, rng):
    """The level matrix (n by k, every column a permutation of 0..n-1) of a
    maximin Latin hypercube by Morris & Mitchell's search (MAXIMIN LHS)."""
    import numpy as np
    best_L, best_key = None, None
    for p in P_VALUES:
        L = np.column_stack([rng.permutation(n) for _ in range(k)])
        if n > 2:
            L = _anneal(L, p, rng)
        key = maximin_order(squared_distances(L))
        if best_key is None or key < best_key:
            best_L, best_key = L, key
    return best_L


def _anneal(L, p: int, rng):
    """Fig. 1 of the paper on phi_p of the squared distances, from the random
    Latin hypercube L; returns the best design it met."""
    import numpy as np
    n, k = L.shape
    cols = L.T.astype(float)                   # the columns, each contiguous
    # The squared distances in units of the smallest at the start, so that the
    # terms d^-p stay within range up to p = 100; phi and t0 scale alike, so
    # neither the ranking nor the acceptance depends on the unit.
    D2 = squared_distances(L)
    scale = float(D2.min())
    D2 /= scale
    T = neg_power(D2, p)                       # the pairs' terms, 0 on the diagonal
    S = float(T.sum()) / 2
    phi = S ** (1.0 / p)
    t = _initial_temperature(n, k, p, scale)
    i_max = min(10 * k * n * (n - 1) // 2, I_MAX_CAP)
    budget = min(BUDGET, BUDGET_WORK // n)
    # the comparisons go by S, which phi orders the same way (phi = S^(1/p)),
    # and need no root; the acceptance probability takes phi
    best_cols, best_S, at_best = cols.copy(), S, True
    spent = 0
    while spent < budget:
        accepted, tries = False, 1
        while tries < i_max and spent < budget:
            spent += 1
            c = int(rng.integers(k))
            a = int(rng.integers(n))
            b = int(rng.integers(n - 1))
            b += b >= a                         # a second row, not a
            col = cols[c]
            # a takes b's level and b a's: the other rows' distances to them
            # change; theirs to each other does not
            delta = ((col[b] - col) ** 2 - (col[a] - col) ** 2) / scale
            delta[a] = delta[b] = 0.0
            new_a, new_b = D2[a] + delta, D2[b] - delta
            t_a, t_b = neg_power(new_a, p), neg_power(new_b, p)
            S_try = S + float(t_a.sum() - T[a].sum() + t_b.sum() - T[b].sum())
            phi_try = S_try ** (1.0 / p) if S_try > 0 else math.inf
            if S_try < S or rng.random() < math.exp(-(phi_try - phi) / t):
                if at_best and S_try > best_S:
                    best_cols, at_best = cols.copy(), False     # leaving the best design
                col[a], col[b] = col[b], col[a]
                D2[a], D2[b], D2[:, a], D2[:, b] = new_a, new_b, new_a, new_b
                T[a], T[b], T[:, a], T[:, b] = t_a, t_b, t_a, t_b
                S, phi, accepted = S_try, phi_try, True
            if S_try < best_S:
                best_S, at_best, tries = S_try, True, 1
            else:
                tries += 1
        S = float(T.sum()) / 2                  # again in full, against the drift
        phi = S ** (1.0 / p)
        if not accepted:
            break
        t *= FAC_T
    return (cols if at_best else best_cols).T.astype(np.int64)


def _initial_temperature(n: int, k: int, p: int, scale: float) -> float:
    """t0 of Section 4: on a hypothetical design whose C(n, 2) squared distances
    are spread evenly over 50-150 % of the class's mean, k n (n + 1) / 6 in
    levels, one of the smallest made one level smaller (the paper's 1/(n - 1)^2)
    is accepted with probability ACCEPT."""
    import numpy as np
    m = n * (n - 1) // 2
    mean = k * n * (n + 1) / 6
    lo, hi = 0.5 * mean / scale, 1.5 * mean / scale
    total = 0.0
    for start in range(0, m, 1_000_000):             # in pieces: m is 12.5 million at 5 000
        h = lo + (hi - lo) * np.arange(start, min(m, start + 1_000_000)) / max(m - 1, 1)
        total += float(neg_power(h, p).sum())
    smallest = lo
    closer = max(lo - 1.0 / scale, 1e-12)
    ends = neg_power(np.array([smallest, closer]), p)
    total_try = total - float(ends[0]) + float(ends[1])
    d_phi = total_try ** (1.0 / p) - total ** (1.0 / p)
    return d_phi / -math.log(ACCEPT)


# ── the start of an algorithm from a design ────────────────────────────────


def nondominated(F):
    """Indices of the rows of F no other row dominates (minimization)."""
    import numpy as np
    F = np.asarray(F, float)
    keep = []
    for i in range(len(F)):
        worse = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if not worse.any():
            keep.append(i)
    return np.asarray(keep, dtype=np.intp)


def start_points(F, k: int):
    """Indices of k rows of F to start from: the nondominated ones first, DSS
    choosing among them when there are more than k; the rest, when there are
    fewer, chosen by DSS among the other rows. Every row once at most."""
    import numpy as np
    from .run.archive import dss_order
    F = np.asarray(F, float)
    if k >= len(F):
        return np.arange(len(F))
    ok = np.flatnonzero(np.all(np.isfinite(F), axis=1))   # failed evaluations last
    if len(ok) < len(F):
        if len(ok) <= k:
            return np.concatenate([ok, np.setdiff1d(np.arange(len(F)), ok)[:k - len(ok)]])
        return ok[start_points(F[ok], k)]
    nd = nondominated(F)
    if len(nd) >= k:
        return nd[dss_order(F[nd], k)]
    rest = np.setdiff1d(np.arange(len(F)), nd)
    return np.concatenate([nd, rest[dss_order(F[rest], k - len(nd))]])

# SPDX-License-Identifier: Apache-2.0
"""What the C++ side accepts, derived from the C++ side.

The algorithm list is read out of include/mootation/algorithms.def — the same
X-macro file the test suite, the Python binding and the run-time dispatch are
generated from. Nothing here is a second copy of that list, because a second
copy is a list that goes stale.

The population-size rules below ARE hand-maintained, and they are the one place
in this package that can drift from the C++ headers. They are written down
because the alternative is worse: without them a typo costs a day-long run ten
minutes in, which is exactly the loss TUI_SPEC.md §7 is meant to prevent. Each
entry names the mechanism in the header it mirrors, so a reader can check it.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

# ── The registry ────────────────────────────────────────────────────────────

_ALG_RE = re.compile(r"^\s*MOOTATION_ALG\(\s*([A-Za-z0-9_]+)\s*,")


def _algorithms_def() -> Path:
    """Locate algorithms.def relative to this file, then relative to the CWD.

    Installed layouts move things around, so both the source checkout
    (python/mootation/run/ -> ../../include/mootation/) and a run from the
    repository root are tried before giving up.
    """
    here = Path(__file__).resolve()
    # python/mootation/run/algorithms.py -> repository root, then the include
    # tree; plus an installed layout, where the .def sits beside the package.
    candidates = [
        here.parents[3] / "include" / "mootation" / "algorithms.def",
        here.parents[1] / "algorithms.def",
        Path.cwd() / "include" / "mootation" / "algorithms.def",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "cannot find include/mootation/algorithms.def; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


@lru_cache(maxsize=1)
def algorithm_names() -> tuple[str, ...]:
    """Every name accepted as `[[algorithms]] name`, in registry order."""
    text = _algorithms_def().read_text(encoding="utf-8")
    names = []
    for line in text.splitlines():
        m = _ALG_RE.match(line)
        if m:
            names.append(m.group(1))
    if not names:
        raise ValueError(f"no MOOTATION_ALG entries found in {_algorithms_def()}")
    return tuple(names)


# ── Population-size rules ───────────────────────────────────────────────────

# Cores that call das_dennis::generate_exact: the population size must equal a
# Das-Dennis lattice point count for the objective count, exactly. Anything
# else throws std::invalid_argument at setup.
#
# Derived from: grep -l generate_exact include/mootation/algorithms/*.hpp
EXACT_LATTICE = frozenset({
    "a_nsga3", "adaw", "crea", "edv", "irea", "mbra", "moead", "moead_awa",
    "moead_dd", "moead_de", "moead_dra", "mombi2", "nsga3", "rvea", "srv",
    "srv_moead", "srv_nsga3", "theta_dea",
})

# M2M-family cores that partition the population into K subregions of equal
# size: they require pop_size % K == 0 (pop = K*S) and throw otherwise.
# K defaults to 10 in both and is settable via `params = { K = ... }`.
#
# Derived from: the `is not divisible by` throws in moead_m2m.hpp and
# sms_m2m.hpp. moead_am2m and isde_rd also carry a K, but they adapt instead of
# throwing, so they are not listed.
K_DIVISIBLE = {"moead_m2m": 10, "sms_m2m": 10}


def das_dennis_count(m: int, h: int) -> int:
    """Number of points in the m-objective Das-Dennis lattice with h divisions."""
    return math.comb(h + m - 1, m - 1)


def lattice_sizes(m: int, limit: int) -> list[int]:
    """Attainable single-layer lattice sizes for m objectives, up to `limit`."""
    out = []
    h = 1
    while True:
        n = das_dennis_count(m, h)
        if n > limit:
            break
        out.append(n)
        h += 1
    return out


def nearest_lattice_sizes(m: int, pop: int) -> tuple[int | None, int | None]:
    """The attainable lattice sizes bracketing `pop`: (largest <=, smallest >).

    Returned so an error message can say "use 91 or 105" instead of merely
    "invalid", which is the difference between a fixable message and a riddle.
    """
    below = None
    h = 1
    while True:
        n = das_dennis_count(m, h)
        if n == pop:
            return (n, n)
        if n > pop:
            return (below, n)
        below = n
        h += 1


def check_pop(name: str, pop: int, n_objs: int, params: dict | None = None
              ) -> str | None:
    """Return a human-readable reason `pop` is unusable, or None if it is fine.

    Only the two hard constraints are checked — the ones that abort the run at
    setup. Cores that round a request up to the nearest lattice (generate_auto)
    are deliberately not flagged: they warn at run time and keep going, so
    refusing them here would be stricter than the library.
    """
    params = params or {}

    if name in EXACT_LATTICE:
        below, above = nearest_lattice_sizes(n_objs, pop)
        if below == pop:
            return None
        hint = ", ".join(str(v) for v in (below, above) if v is not None)
        return (
            f"pop = {pop} is not a Das-Dennis lattice size for n_objs = {n_objs}; "
            f"{name} requires an exact lattice. Nearest attainable: {hint}"
        )

    if name in K_DIVISIBLE:
        k = int(params.get("K", params.get("n_clusters", K_DIVISIBLE[name])))
        if k < 1:
            return f"K = {k} must be >= 1"
        if pop % k != 0:
            divisors = [d for d in range(2, min(pop, 64) + 1) if pop % d == 0]
            hint = (", ".join(str(d) for d in divisors[:8]) if divisors
                    else "none in 2..64")
            return (
                f"pop = {pop} is not divisible by K = {k}; {name} partitions the "
                f"population into K equal subregions (pop = K*S). Divisors of "
                f"{pop}: {hint}"
            )

    return None

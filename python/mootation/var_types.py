# SPDX-License-Identifier: Apache-2.0
"""Integer, discrete and binary variables for minimize() (task 7 of 2026-10-10,
§10, and supplement 2, §3).

    minimize(fn, bounds, 2, var_types=["real", "int", ("discrete", [0.2, 0.31, 0.4]),
                                       "bin"], ...)

One entry per variable of `bounds`:

  "real"                 as always
  "int"                  an integer in [ceil lo, floor hi]
  ("discrete", values)   one of the values (sorted here)
  "bin"                  a bit, 0 or 1 (its bounds are ignored)

THE CORE'S VIEW. The real, integer and discrete variables are the core's real
genome, in their order; the bits are its binary genome (DataVault's), after
them. The operators work on reals: an integer variable's operator space is
[lo - 0.5, hi + 0.5) (so that the two end values get a whole unit, not half of
one), a discrete one's [min, max] of its values. After every write the vault
rounds them (int_repair = "lamarck", the default: the rounded value is the
genome — DataVault::set_snap; integers to the nearest, ties to even; discrete
values to the nearest, ties to the lower) or leaves them and only the
evaluation sees them rounded ("baldwin"). Polynomial mutation takes eta_int for
them (3 by default: steps of a few units, after the report of 2026-10 on mixed
variables, where 20 makes a step of less than one unit mostly round back to
where it was).

THE CALLER'S VIEW. fn receives the variables in the order of var_types,
decoded: integers as int, discrete values from their list, bits as int. The
result's variables are the core's real genome (the integer and discrete
values as floats), its binary_variables the bits.

The binary genome is a C++ feature of 47 of the 64 cores; the 17 others refuse
it at setup, by name.
"""

from __future__ import annotations

import math


class Layout:
    """How the caller's variables map onto the core's genome."""

    def __init__(self, var_types, bounds):
        bounds = [tuple(b) for b in bounds]
        if len(var_types) != len(bounds):
            raise ValueError(f"var_types has {len(var_types)} entries for {len(bounds)} "
                             "variables")
        self.kinds, self.values = [], []
        for t, (lo, hi) in zip(var_types, bounds):
            if isinstance(t, (tuple, list)) and len(t) == 2 and t[0] == "discrete":
                v = sorted(float(a) for a in t[1])
                if not v:
                    raise ValueError("a discrete variable needs at least one value")
                self.kinds.append("discrete")
                self.values.append(v)
            elif t in ("real", "int", "bin"):
                if t == "int" and math.ceil(lo) > math.floor(hi):
                    raise ValueError(f"an integer variable in [{lo}, {hi}] has no integer")
                self.kinds.append(t)
                self.values.append(None)
            else:
                raise ValueError(f"var_types entry {t!r}: 'real', 'int', 'bin' or "
                                 "('discrete', [values])")
        self.bounds = bounds
        self.real = [i for i, k in enumerate(self.kinds) if k != "bin"]
        self.bits = [i for i, k in enumerate(self.kinds) if k == "bin"]

    def trivial(self) -> bool:
        return all(k == "real" for k in self.kinds)

    def core_bounds(self) -> list:
        out = []
        for i in self.real:
            lo, hi = self.bounds[i]
            if self.kinds[i] == "int":
                out.append((math.ceil(lo) - 0.4999999, math.floor(hi) + 0.4999999))
            elif self.kinds[i] == "discrete":
                out.append((self.values[i][0], self.values[i][-1]))
            else:
                out.append((lo, hi))
        return out

    def var_kinds(self) -> list:
        return [{"real": 0, "int": 1, "discrete": 2}[self.kinds[i]] for i in self.real]

    def discrete_values(self) -> list:
        return [self.values[i] if self.kinds[i] == "discrete" else [] for i in self.real]

    def decode_one(self, i: int, v: float):
        if self.kinds[i] == "int":
            lo, hi = self.bounds[i]
            return int(min(max(round(v), math.ceil(lo)), math.floor(hi)))   # ties to even
        if self.kinds[i] == "discrete":
            vals = self.values[i]
            return min(vals, key=lambda a: (abs(a - v), a))                   # ties to the lower
        if self.kinds[i] == "bin":
            return int(v != 0)
        return v

    def to_user(self, core_x) -> list:
        """The core's [reals..., bits...] as the caller's decoded variables."""
        x = [None] * len(self.kinds)
        for k, i in enumerate(self.real):
            x[i] = self.decode_one(i, float(core_x[k]))
        for k, i in enumerate(self.bits):
            x[i] = int(core_x[len(self.real) + k] != 0)
        return x

    def split(self, user_x):
        """The caller's variables as the core's (reals, bits)."""
        return ([float(user_x[i]) for i in self.real], [int(user_x[i] != 0) for i in self.bits])

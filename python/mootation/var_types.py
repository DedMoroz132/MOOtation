# SPDX-License-Identifier: Apache-2.0
"""Integer, discrete, binary and categorical variables for minimize() (task 7
of 2026-10-10, §10; supplement 2, §3; supplement 3, §1).

    minimize(fn, bounds, 2, var_types=["real", "int", ("discrete", [0.2, 0.31, 0.4]),
                                       "bin", ("cat", ["steel", "titanium", "CFRP"])],
             ...)

One entry per variable of `bounds`:

  "real"                 as always
  "int"                  an integer in [ceil lo, floor hi]
  ("discrete", values)   one of the values (sorted here)
  "bin"                  a bit, 0 or 1 (its bounds are ignored)
  ("cat", labels)        one of k >= 2 unordered labels, any values that
                         compare by ==; ("cat", k) for the labels 0 .. k-1
                         (its bounds are ignored)

THE CORE'S VIEW. The real, integer and discrete variables are the core's real
genome, in their order; the bits and the categorical variables are its
integer genome (DataVault's binary_variables), after them: a gene holding
0 or 1 for a bit, the label's index 0 .. k-1 for a categorical variable
(ops::gene_levels). Uniform crossover takes genes as they are; mutation gives
a gene, with probability 1/(number of genes), at most 1/2 (pymoo's cap), one
of its other values, each alike (operators/bit_flip.hpp: a bit flips); SBX,
DE and polynomial mutation never see them. The operators work on reals: an integer variable's operator space is
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
decoded: integers as int, discrete values from their list, bits as int,
categorical variables as their labels. The result's variables are the core's
real genome (the integer and discrete values as floats), its binary_variables
the integer genome (bits, and labels' indices); Layout.to_user decodes a row.

The integer genome is a C++ feature of 49 of the 67 cores; the 18 others
refuse it at setup, by name (docs/running.md lists them).
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
            if isinstance(t, (tuple, list)) and len(t) == 2 and t[0] == "cat":
                labels = list(range(t[1])) if isinstance(t[1], int) else list(t[1])
                if len(labels) < 2:
                    raise ValueError("a categorical variable needs at least two labels")
                if any(labels[a] == labels[b] for a in range(len(labels))
                       for b in range(a)):
                    raise ValueError(f"a categorical variable's labels repeat: {labels}")
                self.kinds.append("cat")
                self.values.append(labels)
            elif isinstance(t, (tuple, list)) and len(t) == 2 and t[0] == "discrete":
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
                raise ValueError(f"var_types entry {t!r}: 'real', 'int', 'bin', "
                                 "('discrete', [values]) or ('cat', [labels] or k)")
        self.bounds = bounds
        self.real = [i for i, k in enumerate(self.kinds) if k not in ("bin", "cat")]
        self.genes = [i for i, k in enumerate(self.kinds) if k in ("bin", "cat")]

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

    def gene_levels(self) -> list:
        """The levels of every gene of the integer genome, 2 for a bit; empty
        where there is no categorical variable (the core's default)."""
        if "cat" not in self.kinds:
            return []
        return [len(self.values[i]) if self.kinds[i] == "cat" else 2 for i in self.genes]

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
        """The core's [reals..., genes...] as the caller's decoded variables."""
        x = [None] * len(self.kinds)
        for k, i in enumerate(self.real):
            x[i] = self.decode_one(i, float(core_x[k]))
        for k, i in enumerate(self.genes):
            g = int(core_x[len(self.real) + k])
            if self.kinds[i] == "cat":
                labels = self.values[i]
                x[i] = labels[min(max(g, 0), len(labels) - 1)]
            else:
                x[i] = int(g != 0)
        return x

    def numeric(self, user_x) -> list:
        """The caller's variables as numbers, in their order: a categorical
        variable as its label's index (for an archive of floats)."""
        out = []
        for i, v in enumerate(user_x):
            if self.kinds[i] == "cat":
                out.append(float(next(k for k, a in enumerate(self.values[i]) if a == v)))
            else:
                out.append(float(v))
        return out

    def split(self, user_x):
        """The caller's variables as the core's (reals, genes)."""
        genes = []
        for i in self.genes:
            if self.kinds[i] == "cat":
                labels = self.values[i]
                hit = [k for k, a in enumerate(labels) if a == user_x[i]]
                if not hit:
                    raise ValueError(f"{user_x[i]!r} is not one of the labels {labels}")
                genes.append(hit[0])
            else:
                genes.append(int(user_x[i] != 0))
        return [float(user_x[i]) for i in self.real], genes

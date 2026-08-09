<!-- SPDX-License-Identifier: Apache-2.0 -->
# Wrappers over the C ABI

Thin adapters that let other languages drive MOOtation. Each is one file meant
to be **copied into your project** rather than installed — they are sixty to a
hundred lines and you will want to change them.

| language | file | how it reaches the ABI |
|---|---|---|
| Julia | [`MOOtation.jl`](MOOtation.jl) | `@ccall`, directly |
| MATLAB / Octave | [`mootation.m`](mootation.m) | `loadlibrary` / `calllib`, directly |
| R | [`mootation.R`](mootation.R) + [`mootation_r.c`](mootation_r.c) | through a C shim (see below) |
| Python | [`../ctypes_demo.py`](../ctypes_demo.py) | `ctypes`, directly |

All four drive the same `ask` / `tell` loop:

```
X = ask()                 # candidates to evaluate, n x n_vars
while X is not empty:
    F = your_objectives(X)
    n = tell(F)           # returns the size of the NEXT batch
X, F, cv = result()       # the final population
```

**The batch size is the algorithm's choice, not `pop_size`.** Generational
algorithms hand over a whole offspring generation; steady-state ones
(MOEA/D-DE, MOEA/DD) hand over one candidate at a time. Size your loop off the
returned array, never off a constant.

Every one of these languages is column-major except C, and the ABI is
row-major. Each wrapper transposes in one place, marked in a comment. Getting
that backwards silently scrambles decision vectors rather than failing, which
is why it is confined rather than spread through the file.

## Why R needs a shim

R cannot call this ABI directly. `.C()` marshals only atomic vectors, so an
opaque `moo_session*` cannot survive a round trip through it, and `.Call()`
requires functions written against the R API (`SEXP` in, `SEXP` out) rather
than plain C ones. `mootation_r.c` is the adapter: it keeps the session in an R
external pointer, with a finalizer so nothing leaks when a script ends without
closing.

```bash
R CMD SHLIB mootation_r.c -L<dir with libmootation> -lmootation -I<repo>/include
```

## Status — read this before relying on them

**Not one of these three was run on the machine they were written on**: no
Julia, R, MATLAB or Octave was installed. What *is* tested, on every CI run, is
the ABI underneath them — [`../smoke.c`](../smoke.c) compiled as C99 and
[`../ctypes_demo.py`](../ctypes_demo.py) driven from Python, both asserting the
same numbers.

So: the entry points, the argument shapes and the error convention are known
good. The language-specific marshalling in these three files is careful but
unverified. Treat them as a working starting point, check them against
`ctypes_demo.py`'s output on the same settings, and please open an issue with
what needed changing — that is far more useful than a wrapper nobody has run.

## The error convention

Every fallible entry point returns a negative int (or `NULL`) and leaves a
message in `moo_last_error()`. No exception ever crosses the boundary. Each
wrapper turns that into the local idiom — a Julia `error`, an R `stop`, a
MATLAB `error`, a Python `RuntimeError`.

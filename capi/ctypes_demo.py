#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive MOOtation from Python through the C ABI, with ctypes only.

This is not the pybind11 module — there is nothing to build here. It loads the
shared library and calls the same `extern "C"` entry points a Rust, C#, Julia
or MATLAB caller would use, which is what makes it a proof that the ABI is
reachable from another language rather than a Python-shaped convenience.

    python capi/ctypes_demo.py path/to/libmootation.so

The `MOOtation` class below is about 60 lines and is meant to be copied into
whatever project needs it.
"""

import ctypes
import math
import sys


class MOOtation:
    """A thin ctypes wrapper. Copy it; it is not meant to be imported."""

    def __init__(self, library_path):
        lib = ctypes.CDLL(library_path)
        c, d, p = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
        dp = ctypes.POINTER(d)

        def sig(name, restype, argtypes):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            return fn

        self._version_string = sig("moo_version_string", ctypes.c_char_p, [])
        self._alg_count = sig("moo_algorithm_count", c, [])
        self._alg_name = sig("moo_algorithm_name", ctypes.c_char_p, [c])
        self._open = sig("moo_open", p, [ctypes.c_char_p])
        self._close = sig("moo_close", None, [p])
        self._n_vars = sig("moo_n_vars", c, [p])
        self._n_objs = sig("moo_n_objs", c, [p])
        self._n_cons = sig("moo_n_cons", c, [p])
        self._generation = sig("moo_generation", c, [p])
        self._ask_count = sig("moo_ask_count", c, [p])
        self._ask = sig("moo_ask", c, [p, dp, c])
        self._tell = sig("moo_tell", c, [p, dp, c, dp, c])
        self._result_count = sig("moo_result_count", c, [p])
        self._result = sig("moo_result", c, [p, dp, c, dp, c, dp, c])
        self._last_error = sig("moo_last_error", ctypes.c_char_p, [p])

        self.handle = None

    # ---- lifecycle -------------------------------------------------------
    def open(self, settings_text):
        self.handle = self._open(settings_text.encode())
        if not self.handle:
            raise RuntimeError(self._err(None))
        self.n_vars = self._n_vars(self.handle)
        self.n_objs = self._n_objs(self.handle)
        self.n_cons = self._n_cons(self.handle)
        return self

    def close(self):
        if self.handle:
            self._close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _err(self, handle):
        msg = self._last_error(handle)
        return msg.decode() if msg else "unknown error"

    def _check(self, value):
        if value < 0:
            raise RuntimeError(self._err(self.handle))
        return value

    # ---- the loop --------------------------------------------------------
    def ask(self):
        """Return the waiting candidates as a list of lists. Empty = finished."""
        n = self._check(self._ask_count(self.handle))
        if n == 0:
            return []
        buf = (ctypes.c_double * (n * self.n_vars))()
        self._check(self._ask(self.handle, buf, n * self.n_vars))
        return [list(buf[i * self.n_vars:(i + 1) * self.n_vars]) for i in range(n)]

    def tell(self, F, G=None):
        """Post objective values; return the number waiting in the next batch."""
        flat_f = [v for row in F for v in row]
        fbuf = (ctypes.c_double * len(flat_f))(*flat_f)
        if self.n_cons > 0:
            flat_g = [v for row in (G or []) for v in row]
            gbuf = (ctypes.c_double * len(flat_g))(*flat_g)
            return self._check(self._tell(self.handle, fbuf, len(flat_f),
                                          gbuf, len(flat_g)))
        return self._check(self._tell(self.handle, fbuf, len(flat_f), None, 0))

    def result(self):
        """(variables, objectives, cv) of the final population."""
        n = self._check(self._result_count(self.handle))
        xb = (ctypes.c_double * (n * self.n_vars))()
        fb = (ctypes.c_double * (n * self.n_objs))()
        cb = (ctypes.c_double * n)()
        self._check(self._result(self.handle, xb, n * self.n_vars,
                                 fb, n * self.n_objs, cb, n))
        X = [list(xb[i * self.n_vars:(i + 1) * self.n_vars]) for i in range(n)]
        F = [list(fb[i * self.n_objs:(i + 1) * self.n_objs]) for i in range(n)]
        return X, F, list(cb)

    # ---- library-level ---------------------------------------------------
    @property
    def version(self):
        return self._version_string().decode()

    def algorithms(self):
        return [self._alg_name(i).decode() for i in range(self._alg_count())]

    @property
    def generation(self):
        return self._generation(self.handle)


def zdt1(x):
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    return [f1, g * (1.0 - math.sqrt(f1 / g))]


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2

    moo = MOOtation(argv[1])
    print(f"MOOtation {moo.version}, {len(moo.algorithms())} algorithms")
    assert len(moo.algorithms()) == 60, "expected 60 algorithms"
    assert "nsga2" in moo.algorithms()

    settings = """
        algorithm = nsga2
        pop_size  = 40
        max_gen   = 40
        seed      = 7
        n_vars    = 10
        n_objs    = 2
        lower     = 0
        upper     = 1
    """

    with moo.open(settings) as run:
        batches = 0
        evaluated = 0
        X = run.ask()
        while X:
            # The batch size is the algorithm's choice, never pop_size.
            F = [zdt1(x) for x in X]
            run.tell(F)
            batches += 1
            evaluated += len(X)
            X = run.ask()

        _, objectives, cv = run.result()
        best = min(f[0] + f[1] for f in objectives)
        print(f"ctypes: {batches} batches, {evaluated} evaluations, "
              f"{run.generation} generations, {len(objectives)} solutions")
        print(f"best f1+f2 = {best:.6f} "
              f"(analytic minimum on the ZDT1 front is 0.75)")

        assert len(objectives) == 40, "expected 40 solutions"
        assert all(c == 0.0 for c in cv), "unconstrained run must report cv = 0"
        # ZDT1 is f2 = 1 - sqrt(f1), so f1 + f2 bottoms out at 3/4.
        assert best >= 0.75 - 1e-9, "a solution beat the true front"
        # 40 individuals for 40 generations is a smoke test, not a budget: it
        # lands around 0.80. A random population sits near 4-5, so this still
        # separates "the optimizer ran" from "the optimizer did nothing".
        assert best < 0.85, "the run did not converge"

    # A bad configuration surfaces as an exception, not a crash.
    try:
        MOOtation(argv[1]).open("algorithm = does_not_exist\nn_vars = 2\n"
                                "lower = 0\nupper = 1\n")
    except RuntimeError as e:
        print(f"bad config rejected: {e}")
    else:
        raise AssertionError("an unknown algorithm should not open")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

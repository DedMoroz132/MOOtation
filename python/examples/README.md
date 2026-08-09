<!-- SPDX-License-Identifier: Apache-2.0 -->
# Examples

Run any of them from the repository root after `pip install ".[all]"`, or with
`PYTHONPATH=python` against a CMake build.

## From Python

| file | what it shows |
|---|---|
| [`01_basic.py`](01_basic.py) | minimize a Python function — the whole API for the common case |
| [`02_constraints.py`](02_constraints.py) | a second function returning violations; `<= 0` means satisfied |
| [`03_benchmarks.py`](03_benchmarks.py) | the 216 built-in problems, and comparing algorithms by IGD |
| [`04_expensive.py`](04_expensive.py) | batched evaluation and caching, for when a call costs real time |
| [`05_many_objective.py`](05_many_objective.py) | five objectives, and the Das-Dennis population-size trap |
| [`06_restart.py`](06_restart.py) | saving a population and an evaluation log, then continuing from them — including with a different algorithm |

```python
import mootation

res = mootation.minimize(my_objectives, bounds=[(0, 1)] * 10, n_objs=2,
                         algorithm="nsga3", pop_size=91, n_gen=250)
```

`mootation.algorithms()` lists all 60 names. A knob the chosen algorithm does
not have comes back in `res.ignored` rather than being silently dropped — a run
configured with an ignored parameter is not the run you asked for.

## From a config file, with an external solver

When the objectives come from programs rather than a Python function — a
mesher, a solver, a measurement rig — describe the run in TOML and let
`mootation.run` drive it.

| file | what it shows |
|---|---|
| [`demo.toml`](demo.toml) | a complete runnable example; the "solver" is [`solver_demo.py`](solver_demo.py) |
| [`airfoil.toml`](airfoil.toml) | the real-world shape: gmsh, an external solver, CSV output. **Deliberately fails `--check`** — the point is to show the report |
| [`bench.toml`](bench.toml) | running the built-in suites instead of an external program |
| [`../../examples/run.cfg`](../../examples/run.cfg) | the C++/C-ABI settings format, every key commented |

```bash
python -m mootation.run --show --check python/examples/demo.toml
python -m mootation.run --tui        python/examples/demo.toml
```

`--check` verifies, before anything expensive starts, that every step's
executable exists on this platform, that the input template's `{x[i]}` stay
within `n_vars`, that bounds are ordered and counted, and that each `pop`
satisfies its algorithm's own constraint. It reports every problem at once and
exits 1.

## From C++

`../../examples/` has the C++ side: [`nsga2_zdt1.cpp`](../../examples/nsga2_zdt1.cpp)
for the direct API, [`embed_asktell.cpp`](../../examples/embed_asktell.cpp) for
driving the optimizer from your own loop.

## From another language

`../../capi/ctypes_demo.py` drives the same optimizer through the C ABI with
`ctypes` alone — no build step, no binding. It is about sixty lines and is
meant to be copied into whatever project needs it.

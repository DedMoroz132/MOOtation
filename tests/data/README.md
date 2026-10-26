# Reference values

Values of other implementations at fixed points, which the run layer's tests
(`python/test_run_layer.py`) compare MOOtation's problems with. One JSON file
per source and subject; each carries the library, its version and license,
the module the values come from, the command that made the file, the date,
the number of points, the tolerance of the comparison, and the cases (the
point `x` and the values `f`, `g`, or what the file names), at full precision.

| file | values checked against | points |
|---|---|---|
| `pymoo-0.6.2_wfg.json` | pymoo 0.6.2, WFG1-9 at 2, 3, 5, 10 objectives | 36 |
| `pymoo-0.6.2_c-dtlz.json` | pymoo 0.6.2, C-DTLZ at 3 and 5 objectives | 60 |
| `pymoo-0.6.2_mw.json` | pymoo 0.6.2, MW1-14 | 85 |
| `pymoo-0.6.2_das-cmop.json` | pymoo 0.6.2, DAS-CMOP1-9 at two triplets | 36 |
| `pymoo-0.6.2_omni-sympart.json` | pymoo 0.6.2, Omni-test and SYM-PART simple | 10 |
| `pymoo-0.6.2_wfg-optimum.json` | pymoo 0.6.2, WFG8's and WFG9's optimal distance variables | 4 |
| `reproblems-7876b4e_re.json` | reproblems' Python version, commit 7876b4e, RE21-91 and CRE21-51 | 120 |
| `scipy-1.18.1_sobol.txt` | SciPy 1.18.1, `qmc.Sobol(d=100, scramble=False)`: `include/mootation/sobol.hpp` (tests/test_sobol.cpp) | 40 |
| `npz/` | NumPy 2.5.3 `savez`/`savez_compressed` and zlib's raw deflate (stored, fixed, dynamic blocks): `include/mootation/io/npz.hpp`, `campaign/references.hpp` (tests/test_npz.cpp) | 5 files |

To make them again (by hand; CI does neither, and no test imports pymoo):

```
uv run --no-project --python 3.13 --with pymoo==0.6.2 python tools/refs/gen_pymoo_refs.py
python tools/refs/gen_reproblems_refs.py <reproblems>/reproblem_python_ver/reproblem.py
python tools/refs/gen_sobol_ref.py
python tools/refs/gen_npz_samples.py
```

The other reference data of the tests stay where they were: the problems'
fronts inside the package (`python/mootation/benchmarks/`) and the defaults'
fingerprints (`tools/compat_check.py`).

## The library's elementary functions

`detmath_reference.txt` holds, for every function of
`include/mootation/detmath.hpp`, 2·10⁴ test points (made from integers by
`tools/detmath/points.py`, and the same in `tests/test_detmath.cpp`) and the
hard inputs of `tools/detmath/hard_inputs.txt`: the value to 60 digits
(`tools/detmath/reference.py`, Python's `decimal`) as its correctly rounded
double and the rest in thousandths of an ulp, and the bits `dm::` gives. The
test checks both: the error within the tolerance below, and the bits — one file
for every platform. Made by `tools/detmath/make_data.py` (with the extension
built); `tools/detmath/check.py` measures 10⁶ points a function the same way.

| function | tolerance, ulp | largest error, 2·10⁴ points | largest error, 10⁶ points |
|---|---|---|---|
| `exp` | 1 | 0.704 | 0.748 |
| `log` | 1 | 0.520 | 0.533 |
| `pow` | 2 | 0.524 | 0.726 |
| `sin` | 1 | 0.705 | 0.755 |
| `cos` | 1 | 0.736 | 0.753 |
| `tan` | 2 | 0.867 | 0.957 |
| `acos` | 2 | 0.573 | 0.629 |
| `hypot` | 2 | 0.500 | 0.740 |
| `asin` | 2 | 0.605 | 0.613 |
| `atan` | 2 | 0.537 | 0.553 |
| `sin` above 2²⁰·π/2 (`sin_big`) | 1 | 0.711 | 0.771 |
| `cos` above 2²⁰·π/2 (`cos_big`) | 1 | 0.714 | 0.758 |
| `tan` above 2²⁰·π/2 (`tan_big`) | 2 | 0.857 | 0.957 |

The polynomial coefficients and constants (`include/mootation/detmath_coeffs.hpp`)
come from `tools/detmath/gen_coeffs.py` (`--check` compares).

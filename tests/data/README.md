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

To make them again (by hand; CI does neither, and no test imports pymoo):

```
uv run --no-project --python 3.13 --with pymoo==0.6.2 python tools/refs/gen_pymoo_refs.py
python tools/refs/gen_reproblems_refs.py <reproblems>/reproblem_python_ver/reproblem.py
```

The other reference data of the tests stay where they were: the problems'
fronts inside the package (`python/mootation/benchmarks/`) and the defaults'
fingerprints (`tools/compat_check.py`).

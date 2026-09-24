<!-- SPDX-License-Identifier: Apache-2.0 -->
# Experiments of the 2026-09-23 task (block E)

Each file states its question in its header. All run five seeds at 10 000
evaluations on DTLZ1_3D, DTLZ3_3D, shiftDTLZ1_3D, shiftDTLZ3_3D, ZDT4, ZDT1 and
ZDT6 — E5 on the 110 bbob-biobj problems as well — E1-E3 with the operator
statistics on, and every variant is an `[[algorithms]]` entry with its own
`label`.

| file | question |
|---|---|
| `e1_sbx_var_prob.toml` | SBX crossing half the variables (0.5) or all (1.0)? |
| `e2_mutation_step.toml` | polynomial mutation against gaussian steps of 0.025-0.3 of the range and a Cauchy step |
| `e3_bound_repair.toml` | clip, reflect, random or midpoint for the operators that leave the box |
| `e5_rotation.toml` | GDE3 at CR = 1 (rotation-invariant) against CR = 0.5, MO-CMA-ES and SMS-EMOA, on E1's problems and all of bbob-biobj |
| `../structural_bias.toml` | where every algorithm's population goes when the objectives say nothing (A3) |

E4 needs no run: on the campaign's results,

```bash
python -m mootation.run.campaign campaign_all.toml --ranks hv --by separable
```

groups the bbob-biobj problems by separability (F1, F2 and F11 pair two
separable functions); even there the Pareto set is a curve between the two
optima, not parallel to the axes, so read the split with that in mind. E5
adds the 110 bbob-biobj problems to E1's seven and is read the same way
(`--ranks hv --by separable`).

# How each algorithm is checked

Every core's header and code were read against its paper, line by line (the month says when; the
readings and choices are in [deviations.md](deviations.md)). Some also ran the paper's experiment, or
its claim, beside that reading. Every core's default run is held bit for bit by its fingerprint
(`tests/golden_fingerprints.txt`, ctest `golden_fingerprints`), and every core runs in the C++ tests
over all of them (`test_convergence`, `test_constraints`, `test_failures`, `test_reproducibility`,
`test_scale_invariance`); the tests column counts the C++ tests named after it and the run-layer
tests (`python/test_run_layer.py`) that name it.
A test (`every_core_has_a_row_in_verification_md`) keeps the table complete.

| algorithm | read against the paper | the paper's experiment | fingerprint | tests |
|---|---|---|---|---|
| `nsga2` | 2026-09 | the directions of the paper's comparison with PAES | yes | 4 run-layer |
| `nsga2_sdr` | 2026-10 | the direction of the paper's comparison with NSGA-II | yes | 1 run-layer |
| `dn_nsga2` | 2026-10 | SYM-PART: every equivalent part kept | yes | 1 run-layer |
| `bc_spx` | 2026-10 | — | yes | 1 run-layer |
| `bc_rex` | 2026-10 | — | yes | 1 run-layer |
| `spea2` | 2026-09 | the paper's comparison with NSGA-II on its continuous problems | yes | — |
| `spea2_sde` | 2026-09 | the directions of the paper's comparison with SPEA2 and NSGA-II | yes | 1 run-layer |
| `grea` | 2026-09 | — | yes | — |
| `vaea` | 2026-09 | — | yes | — |
| `agemoea` | 2026-09 | the directions of the paper's comparison with NSGA-III, MOEA/D, GrEA, θ-DEA and AR-MOEA | yes | 1 run-layer |
| `etea` | 2026-09 | — | yes | — |
| `gde3` | 2026-09 | — | yes | — |
| `nsga3` | 2026-09 | ADA (knob ada): the paper's claim on the decision-space indicators; the directions of the paper's comparison with MOEA/D and MOEA/D-DE | yes | 1 run-layer |
| `a_nsga3` | 2026-09 | — | yes | 1 run-layer |
| `theta_dea` | 2026-09 | ADA (knob ada): as for nsga3; the directions of the paper's comparison with NSGA-III | yes | 1 run-layer |
| `ar_moea` | 2026-09 | the directions of the paper's comparisons with MOEA/D, NSGA-II, IBEA, MOEA/DD, NSGA-III, RVEA and MOMBI-II | yes | — |
| `moead` | 2026-09 | the directions of the paper's comparison with NSGA-II | yes | — |
| `moead_de` | 2026-09 | the paper's best configuration of the DE components against a worse one | yes | — |
| `moead_dra` | 2026-09 | — | yes | — |
| `moead_dd` | 2026-09 | — | yes | — |
| `moead_awa` | 2026-09 | — | yes | — |
| `adaw` | 2026-09 | — | yes | — |
| `moead_m2m` | 2026-09 | — | yes | — |
| `moead_am2m` | 2026-09 | — | yes | — |
| `sms_m2m` | 2026-09 | — | yes | — |
| `moead_ds` | 2026-09 | — | yes | — |
| `rd_emo` | 2026-09 | — | yes | — |
| `aprd` | 2026-09 | — | yes | — |
| `liu_gu2011` | 2026-09 | — | yes | — |
| `isde_rd` | 2026-09 | — | yes | — |
| `dhea` | 2026-09 | — | yes | — |
| `hlmea` | 2026-09 | — | yes | — |
| `ibea_eplus` | 2026-09 | the directions of the paper's comparison with NSGA-II and SPEA2 | yes | 1 run-layer |
| `ibea_hd` | 2026-09 | — | yes | — |
| `mibea` | 2026-09 | — | yes | — |
| `r2ibea` | 2026-09 | — | yes | — |
| `hype` | 2026-09 | the directions of the paper's comparison with NSGA-II, SPEA2 and IBEA | yes | — |
| `mombi2` | 2026-09 | — | yes | — |
| `nimmo` | 2026-09 | — | yes | — |
| `irea` | 2026-09 | — | yes | — |
| `edv` | 2026-09 | — | yes | — |
| `if_maoea` | 2026-09 | — | yes | — |
| `maoea_iamd` | 2026-09 | — | yes | — |
| `sms_emoa` | 2026-09 | the directions of the paper's comparison with NSGA-II and SPEA2 | yes | 1 run-layer |
| `r2_emoa` | 2026-09 | the paper's experiment at uniform weights: convergence, beside SMS-EMOA | yes | tests/test_r2_emoa.cpp, 1 run-layer |
| `mo_cma_es` | 2026-09 | the directions of the paper's comparison with NSGA-II | yes | — |
| `rvea` | 2026-09 | ADA (knob ada): as for nsga3; the directions of the paper's comparison with NSGA-III and MOEA/DD | yes | tests/test_rvea.cpp, 3 run-layer |
| `rvea_star` | 2026-09 | — | yes | — |
| `maoeaarv` | 2026-09 | — | yes | — |
| `mbra` | 2026-09 | — | yes | — |
| `nrv_moea` | 2026-09 | — | yes | — |
| `dea_gng` | 2026-09 | — | yes | — |
| `srv` | 2026-09 | — | yes | — |
| `srv_nsga3` | 2026-09 | — | yes | 1 run-layer |
| `camoea` | 2026-09 | — | yes | — |
| `cava_moea` | 2026-09 | — | yes | — |
| `maoea_ac` | 2026-09 | — | yes | — |
| `maoea_c` | 2026-09 | — | yes | — |
| `maoea_3c` | 2026-09 | — | yes | — |
| `emyo_c` | 2026-09 | — | yes | — |
| `crea` | 2026-09 | — | yes | — |
| `dcea` | 2026-09 | — | yes | — |
| `hcca` | 2026-09 | — | yes | — |
| `lis_lcs` | 2026-09 | — | yes | — |
| `clia` | 2026-09 | — | yes | — |
| `two_arch2` | 2026-09 | the directions of the paper's comparison with NSGA-III and IBEA | yes | — |
| `naemo` | 2026-09 | — | yes | — |
| `paes` | 2026-09 | — | yes | 1 run-layer |
| `ctaea` | 2026-10 | two of the paper's problems, beside another implementation | yes | — |
| `ccmo` | 2026-10 | the problems of the paper's Table I at its budget | yes | — |
| `dms` | 2026-09 | — | yes | 1 run-layer |
| `dmulti_mads` | 2026-09 | the outputs of the paper's examples; the NOMAD 4 guide | yes | 2 run-layer |
| `multiglods` | 2026-09 | the paper's experiments | yes | 1 run-layer |
| `smpso` | 2026-09 | the OMOPSO preset, an acceptance run beside an independent measure | yes | 2 run-layer |
| `rm_meda` | 2026-09 | — | yes | 1 run-layer |
| `mosa` | 2026-09 | the author's experiment setting, beside an independent implementation | yes | tests/test_mosa.cpp, 2 run-layer |

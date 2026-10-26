# How each algorithm is checked

Every core's header and code were read against its paper, line by line (the month says when; the
readings and choices are in [deviations.md](deviations.md)). Some also ran the paper's experiment, or
its claim, beside that reading: that column names the comparison run (problems,
competitors, objective counts) and what of it was not run and why; the paper's other
experiments were not run. Every core's default run is held bit for bit by its fingerprint
(`tests/golden_fingerprints.txt`, ctest `golden_fingerprints`), and every core runs in the C++ tests
over all of them (`test_convergence`, `test_constraints`, `test_failures`, `test_reproducibility`,
`test_scale_invariance`); the tests column counts the C++ tests named after it and the run-layer
tests (`python/test_run_layer.py`) that name it.
A test (`every_core_has_a_row_in_verification_md`) keeps the table complete.

| algorithm | read against the paper | the paper's experiment | fingerprint | tests |
|---|---|---|---|---|
| `nsga2` | 2026-09 | against PAES on SCH, FON, KUR, POL, ZDT1–4 and ZDT6, convergence and spread; the paper's binary PAES and SPEA: not in the library (PAES ran real-coded) | yes | 4 run-layer |
| `nsga2_sdr` | 2026-10 | the direction of the paper's comparison with NSGA-II | yes | 1 run-layer |
| `dn_nsga2` | 2026-10 | SYM-PART: every equivalent part kept | yes | 1 run-layer |
| `bc_spx` | 2026-10 | — | yes | 1 run-layer |
| `bc_rex` | 2026-10 | — | yes | 1 run-layer |
| `spea2` | 2026-09 | against NSGA-II on SPH at two and three objectives, KUR, QV and ZDT6; SPEA and PESA: not in the library | yes | — |
| `spea2_sde` | 2026-09 | against SPEA2 and NSGA-II on DTLZ1–7 at four and six objectives, DTLZ5 also by the paper's degenerate reference; ten objectives: left out (the run time; objective counts were capped at eight); TSP: not in the registry | yes | 1 run-layer |
| `grea` | 2026-09 | — | yes | — |
| `vaea` | 2026-09 | — | yes | — |
| `agemoea` | 2026-09 | against NSGA-III, MOEA/D, GrEA, θ-DEA and AR-MOEA on MaF1–15 at three and five objectives; ten objectives: left out (the run time; objective counts were capped at eight) | yes | 1 run-layer |
| `etea` | 2026-09 | — | yes | — |
| `gde3` | 2026-09 | — | yes | — |
| `nsga3` | 2026-09 | ADA (knob ada): the paper's claim on the decision-space indicators; against MOEA/D-TCH and MOEA/D-DE on DTLZ1–4 at three, five and eight objectives; MOEA/D-PBI: the library's moead has Tchebycheff only; ten and fifteen objectives: left out (the run time; objective counts were capped at eight) | yes | 1 run-layer |
| `a_nsga3` | 2026-09 | — | yes | 1 run-layer |
| `theta_dea` | 2026-09 | ADA (knob ada): as for nsga3; against NSGA-III on DTLZ1–4 and SDTLZ1–2 at three, five and eight objectives and WFG6–7 at three and five, again at the paper's population size; after TDEA-1 DTLZ1–4 again, three-objective SDTLZ2 and five-objective WFG7 with both readings (knob theta_ties); WFG at eight: not in the registry; ten and fifteen objectives: left out (the run time; objective counts were capped at eight); of the paper's competitors only NSGA-III | yes | 1 run-layer |
| `ar_moea` | 2026-09 | on DTLZ1–7, IDTLZ1–2, WFG1–9, MaF2, MaF4 and MaF13 at three and five objectives: IGD against NSGA-II, MOEA/D and IBEA, HV against NSGA-III, MOEA/DD, RVEA and MOMBI-II; WFG with the registry's k and l (not the paper's); ten objectives: left out (the run time; objective counts were capped at eight) | yes | — |
| `moead` | 2026-09 | against NSGA-II on ZDT1–4, ZDT6 and three-objective DTLZ1–2 | yes | — |
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
| `ibea_eplus` | 2026-09 | against NSGA-II and SPEA2 on ZDT6, KUR and three-objective DTLZ2 and DTLZ6, again with the paper's operators for all three, SPEA2 at its defaults (it had no knob pm then); EXPO: not in the registry | yes | 1 run-layer |
| `ibea_hd` | 2026-09 | — | yes | — |
| `mibea` | 2026-09 | — | yes | — |
| `r2ibea` | 2026-09 | — | yes | — |
| `hype` | 2026-09 | against NSGA-II, SPEA2 and IBEA on DTLZ1–7 and WFG1–9 at two, three and five objectives, at the registry's numbers of variables (not the paper's); seven, twenty-five and fifty objectives and the knapsack problems: not in the registry; ten objectives: left out (the run time; objective counts were capped at eight); SHV and RS: not in the library | yes | — |
| `mombi2` | 2026-09 | — | yes | — |
| `nimmo` | 2026-09 | — | yes | — |
| `irea` | 2026-09 | — | yes | — |
| `edv` | 2026-09 | — | yes | — |
| `if_maoea` | 2026-09 | — | yes | — |
| `maoea_iamd` | 2026-09 | — | yes | — |
| `sms_emoa` | 2026-09 | against NSGA-II and SPEA2 on ZDT1–4 and ZDT6, against SPEA2 on three-objective DTLZ1 | yes | 1 run-layer |
| `r2_emoa` | 2026-09 | convergence on ZDT1 and two-objective DTLZ1–2 at uniform weights, beside SMS-EMOA; γ above one and restricted weight domains: no knob γ | yes | tests/test_r2_emoa.cpp, 1 run-layer |
| `mo_cma_es` | 2026-09 | against NSGA-II on FON, ZDT1–4 and ZDT6; the rotated IHR problems: not in the registry (rotated ZDT ran in their place) | yes | — |
| `rvea` | 2026-09 | ADA (knob ada): as for nsga3; against NSGA-III and MOEA/DD on DTLZ1–4 at three, six and eight objectives, SDTLZ1 at three and eight, WFG1–9 at three and six, WFG with the registry's K and L (not the paper's); SDTLZ1 at six, SDTLZ3 and WFG at eight: not in the registry; ten objectives: left out (the run time; objective counts were capped at eight); of the paper's competitors only these two | yes | tests/test_rvea.cpp, 3 run-layer |
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
| `two_arch2` | 2026-09 | against NSGA-III and IBEA on DTLZ1–4 and WFG1–9 at three and five objectives, NSGA-III at its lattice size (not the paper's population); ten objectives: left out (the run time; objective counts were capped at eight) | yes | — |
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

# How each algorithm is checked

Every core's header and code were read against its paper, line by line (the month says when; the
readings and choices are in [deviations.md](deviations.md)). Some also ran the paper's experiment, or
its claim, beside that reading; there "as the paper" means the directions of the paper's
comparisons came out the same at the library's problem versions and settings, and the row names what
was not confirmed, not compared or not run. Every core's default run is held bit for bit by its fingerprint
(`tests/golden_fingerprints.txt`, ctest `golden_fingerprints`), and every core runs in the C++ tests
over all of them (`test_convergence`, `test_constraints`, `test_failures`, `test_reproducibility`,
`test_scale_invariance`); the tests column counts the C++ tests named after it and the run-layer
tests (`python/test_run_layer.py`) that name it.
A test (`every_core_has_a_row_in_verification_md`) keeps the table complete.

| algorithm | read against the paper | the paper's experiment | fingerprint | tests |
|---|---|---|---|---|
| `nsga2` | 2026-09 | partly: against PAES the spread as the paper; convergence not confirmed (the paper's PAES is binary, the library's real-coded); SPEA not compared | yes | 4 run-layer |
| `nsga2_sdr` | 2026-10 | the direction of the paper's comparison with NSGA-II | yes | 1 run-layer |
| `dn_nsga2` | 2026-10 | SYM-PART: every equivalent part kept | yes | 1 run-layer |
| `bc_spx` | 2026-10 | — | yes | 1 run-layer |
| `bc_rex` | 2026-10 | — | yes | 1 run-layer |
| `spea2` | 2026-09 | partly: against NSGA-II alike, as the paper says, on KUR and the two-objective sphere; on QV and ZDT6 a difference where the paper sees none; not on the three-objective sphere (a weak claim); SPEA and PESA not compared | yes | — |
| `spea2_sde` | 2026-09 | against SPEA2 and NSGA-II as the paper, DTLZ5 by the paper's degenerate reference; four-objective DTLZ4 not (a weak claim); TSP and ten objectives not run | yes | 1 run-layer |
| `grea` | 2026-09 | — | yes | — |
| `vaea` | 2026-09 | — | yes | — |
| `agemoea` | 2026-09 | partly: against NSGA-III, MOEA/D, GrEA, θ-DEA and AR-MOEA mostly as the paper or without a significant difference; MaF3 and MaF14 at five objectives not (pymoo's port of the authors' code behaves as this one); ten objectives not run | yes | 1 run-layer |
| `etea` | 2026-09 | — | yes | — |
| `gde3` | 2026-09 | — | yes | — |
| `nsga3` | 2026-09 | ADA (knob ada): the paper's claim on the decision-space indicators; against MOEA/D-TCH and MOEA/D-DE as the paper, except a weak case on three-objective DTLZ4; MOEA/D-PBI not compared; ten and fifteen objectives not run | yes | 1 run-layer |
| `a_nsga3` | 2026-09 | — | yes | 1 run-layer |
| `theta_dea` | 2026-09 | ADA (knob ada): as for nsga3; against NSGA-III after TDEA-1 (knob theta_ties) as the paper on DTLZ at five and eight objectives, at three without a significant difference; of SDTLZ and WFG after TDEA-1 only three-objective SDTLZ2 and five-objective WFG7 rerun; WFG6 and WFG7 at eight objectives, ten and fifteen objectives not run | yes | 1 run-layer |
| `ar_moea` | 2026-09 | partly: against MOEA/D, NSGA-II, IBEA, MOEA/DD, NSGA-III, RVEA and MOMBI-II mostly as the paper; DTLZ3 not explained (often stalls on local fronts; the authors' code not compared); DTLZ6 and WFG in other versions than the paper's; ten objectives not run | yes | — |
| `moead` | 2026-09 | partly: against NSGA-II as the paper on ZDT3, ZDT6 and three-objective DTLZ2; ZDT4 and DTLZ1 without a significant difference; ZDT1 and ZDT2 not (the paper's means without a test) | yes | — |
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
| `ibea_eplus` | 2026-09 | partly: against NSGA-II and SPEA2 as the paper on KUR, DTLZ2 and DTLZ6; ZDT6 against NSGA-II as the paper at its operators; ZDT6 against SPEA2 not checked at them (spea2 had no knob pm then); the EXPO application not run | yes | 1 run-layer |
| `ibea_hd` | 2026-09 | — | yes | — |
| `mibea` | 2026-09 | — | yes | — |
| `r2ibea` | 2026-09 | — | yes | — |
| `hype` | 2026-09 | partly: against NSGA-II and SPEA2 as the paper or without a significant difference, at the library's problem versions; against IBEA not confirmed (the paper ran DTLZ with its own number of variables and Deb et al.'s probabilities, not run); seven and more objectives and the knapsack problems not run | yes | — |
| `mombi2` | 2026-09 | — | yes | — |
| `nimmo` | 2026-09 | — | yes | — |
| `irea` | 2026-09 | — | yes | — |
| `edv` | 2026-09 | — | yes | — |
| `if_maoea` | 2026-09 | — | yes | — |
| `maoea_iamd` | 2026-09 | — | yes | — |
| `sms_emoa` | 2026-09 | partly: against NSGA-II and SPEA2 as the paper on most of ZDT and DTLZ1; ZDT4 not checked as the paper (it averaged successful runs only); ZDT6 against SPEA2 and ZDT2 convergence against NSGA-II not confirmed (the paper took the competitors' values from Deb et al. at their settings) | yes | 1 run-layer |
| `r2_emoa` | 2026-09 | the paper gives no direction against SMS-EMOA: convergence at uniform weights checked; the drift with γ and restricted weight domains not checked (no knob γ) | yes | tests/test_r2_emoa.cpp, 1 run-layer |
| `mo_cma_es` | 2026-09 | against NSGA-II on FON and ZDT as the paper; the rotated IHR problems not in the registry, not checked (rotated ZDT is no stand-in; the library's variant is the steady-state one) | yes | — |
| `rvea` | 2026-09 | ADA (knob ada): as for nsga3; against NSGA-III and MOEA/DD mostly as the paper on DTLZ and at the library's WFG (the paper's K and L not in the registry); SDTLZ1 at six objectives, WFG at eight and ten objectives not run; the other competitors not compared | yes | tests/test_rvea.cpp, 3 run-layer |
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
| `two_arch2` | 2026-09 | partly: against IBEA as the paper; against NSGA-III not on several DTLZ and WFG (NSGA-III ran on its lattice below the paper's N; not rerun with n_ref_points); ten objectives not run | yes | — |
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

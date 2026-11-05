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

## The population methods at one objective

Stage 4 of task 18: single-objective methods in `include/mootation/scalar/`,
not cores of `algorithms.def`, run by name at `n_objs = 1` (`run_population`
in `scalar.hpp`). Each header was read against its paper (the readings are in
[deviations.md](deviations.md), second table); the experiment column names
what of the paper ran beside it, the twin column a core of the library that
is the same method at one objective and was compared with it. The tests
column names the method's own C++ test; all eleven also run in `test_scalar`
(the wiring: budget, archive, records, the same run cut short) and in the
run-layer test `the_population_methods_run_through_minimize`.
A test (`every_core_has_a_row_in_verification_md`) keeps this table complete too.

| method | read against the paper | the paper's experiment | twin at M = 1 | tests |
|---|---|---|---|---|
| `de` | 2026-10 | Storn & Price 1997, Tables 1–3: evaluations to the value to reach, 17 cases; f8 and f20 left out (their printed forms have no minimum where the paper says) | `gde3` (F = CR = 0.2, reflect): bit for bit | `test_de` |
| `jade` | 2026-10 | Zhang & Sanderson 2009, Tables IV–V: mean error at the checkpoints, D = 30 and 100, with and without the archive | — | `test_jade` |
| `shade` | 2026-10 | Tanabe & Fukunaga 2013, Table III: mean error at the checkpoints, D = 30 | — | `test_shade` |
| `lshade` | 2026-10 | the population schedule (10) generation by generation; CEC 2014: not in the library (the direction of the paper's comparison on bbob instead) | — | `test_shade` |
| `ga_sbx` | 2026-10 | — (the library's polynomial mutation is NSGA-II's code variant, not Deb & Deb 2014's formula) | `nsga2`: bit for bit | `test_ga` |
| `ssga` | 2026-10 | no paper: the method is the header's | `sms_emoa`: bit for bit | `test_ga` |
| `g3_pcx` | 2026-10 | Deb, Anand, Joshi 2002, Table 2: evaluations to 1e-20 on the ellipsoid and Schwefel's function, n = 20 | — | `test_ga` |
| `chc` | 2026-10 | Eshelman & Schaffer 1993, Table 3: De Jong's F1 and F2 on their bit grids | — | `test_ga` |
| `pso` | 2026-10 | Clerc & Kennedy 2002, Table V, Type 1'' | `smpso` with its knobs: no difference by Mann-Whitney | `test_pso` |
| `spso2011` | 2026-10 | CEC 2013: not in the library (the sphere, rotation invariance and the walls instead) | — | `test_spso2011` |
| `sep_cmaes` | 2026-10 | Ros & Hansen 2008, Tables 2–4: evaluations to the target, 7 cases | — | `test_sep_cmaes` |

## The problems of task 20

Problems added in task 20 (2026-11), each checked against its source as the
column says; the C++ suite is the reference, the Python module the
cross-check. Results of runs are not kept here, only what checks them.

| problems | source | what checks them | tests |
|---|---|---|---|
| `uninformative_f2a` … `uninformative_f4g` (n = 2, 10) | Kudela, van Stein, Bäck, Kononova, GECCO 2026, Table 1 (variants α and γ; the weights read off the PDF, 2026-11-02) | C++ = Python value for value and the counts' summary on the same seeds; f1, f2b–f5 bit for bit as before (their rows of `tests/data/registry_sample.json` unchanged); the noise-free curve and the frame the same for α, β and γ; non-dominated among 10⁴ draws about 10², 3·10², 10³ (the table's last column, to an order of magnitude) | `test_registry`; run-layer `kudelas_variants_alpha_and_gamma` |
| `LZ09_F1` … `LZ09_F9` | Li & Zhang, IEEE TEVC 13(2), 2009, Table I (reading P4; read off the PDF, 2026-11-05: every row, the sizes of §V-A) | C++ = NumPy at random points, the box's corners and the Pareto set (run-layer `the_suite_gives_numpys_values`); the Pareto set (Theorem 1's g(x_I)) on the front's formula, in the box, undominated by a random sample; jMetal's variant (P5) and the paper's Table II: not yet | run-layer `lz09_uf_and_cf_pareto_sets_lie_on_their_fronts` |
| `UF1` … `UF10` | Zhang et al., CES-487, "April 20, 2009 DRAFT", §II (UF4, UF10 read off the PDF: readings UF-1, UF-2) | as LZ09; UF1-UF3 equal LZ09's F2, F5, F8 at n = 30; the values of pagmo2 or jMetal at fixed points: not yet | as LZ09 |
| `CF1-CEC2009` … `CF10-CEC2009` | the same report, §III (readings CF-1 … CF-5) | as LZ09, the constraints too; the Pareto set feasible (g <= 1e-12, the boundary's rounding; NaN only at CF8-CF10's pole); undominated by the feasible part of a random sample; library values: not yet; no published table (only properties) | as LZ09 |
| `shiftMaF1_<M>D` … `shiftMaF7_<M>D` (M = 3, 5, 8, 10, 15) | Cheng et al. 2017's MaF1-MaF7 (the g of each read for the map: cyclic for MaF1-MaF6, fold for MaF7); the construction is this library's control (Kůdela 2022, Ibehej et al. 2025), not a published suite | C++ = NumPy as above; with the centre where the map is the identity the base's values (to rounding); at the shifted optimum g = 0; front, frame and box the base's | run-layer `shifted_maf_keeps_its_base_where_the_map_is_the_identity` |

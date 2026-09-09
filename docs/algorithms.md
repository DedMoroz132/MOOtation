<!-- SPDX-License-Identifier: Apache-2.0 -->
# The algorithms

58 algorithms, one header each under `include/mootation/algorithms/`, registered
in `include/mootation/algorithms.def`. Every header opens with the paper
(authors, venue, DOI), a short scheme of one generation in the paper's own
symbols, the paper's defaults with the section they come from, and a numbered
list of declared deviations. The DOI in each table is the primary source the
implementation follows; if you publish results, cite that paper.

## Choosing one

| family | count | reach for it when | watch out for |
|---|---|---|---|
| Pareto-dominance & diversity | 7 | 2–3 objectives, a baseline everyone recognises | crowding distance degrades above 3 objectives |
| Reference-point (NSGA-III family) | 4 | many objectives with a regular front | `pop_size` must be an exact Das–Dennis lattice size |
| Decomposition & region division (MOEA/D family) | 16 | scalarisable problems, many objectives, cheap per-generation cost | most members need an exact lattice size too (the weights are a Das–Dennis set); the steady-state members hand over one candidate at a time; the M2M members want `pop` divisible by `K` |
| Indicator-based | 11 | you care about one quality indicator (ε+, hypervolume, R2) | cost grows fast with the objective count (HypE, R2-IBEA) |
| Reference-vector / angle-based | 7 | irregular or scaled fronts, where a fixed lattice misses | adaptive variants (DEA-GNG, NRV-MOEA, SRV) need a few generations before they help |
| Clustering-based | 11 | disconnected or irregular fronts | cluster counts are parameters the papers tuned per problem |
| Archive-based | 2 | you want the whole non-dominated history, not a fixed population | the archive is the output, not the working population |

Seventeen algorithms require `pop_size` to be an exact Das–Dennis lattice size
for the objective count (91 at M = 3, 210 at M = 5, ...): A-NSGA-III, AdaW,
crEA, EDV, IREA, MBRA, MOEA/D, MOEA/D-AWA, MOEA/DD, MOEA/D-DE, MOEA/D-DRA,
MOMBI-II, NSGA-III, RVEA, MaOEA/SRV, SRV-NSGA-III and θ-DEA; the
M2M family needs `pop` divisible by its subregion count `K`; the others round
an unattainable request to the nearest lattice and say so through
`set_warn_handler`. `python -m mootation.run --check` names every such
constraint before a run starts; `Result::ignored` (C++), `res.ignored` (Python) and
`moo_ignored_*` (C ABI) name every knob the chosen algorithm does not have.

## Pareto-dominance & diversity-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| NSGA-II | 2002 | `nsga2` | [10.1109/4235.996017](https://doi.org/10.1109/4235.996017) |
| SPEA2 | 2001 | `spea2` | [10.3929/ethz-a-004284029](https://doi.org/10.3929/ethz-a-004284029) |
| SPEA2+SDE | 2014 | `spea2_sde` | [10.1109/TEVC.2013.2262178](https://doi.org/10.1109/TEVC.2013.2262178) |
| GrEA | 2013 | `grea` | [10.1109/TEVC.2012.2227145](https://doi.org/10.1109/TEVC.2012.2227145) |
| VaEA | 2017 | `vaea` | [10.1109/TEVC.2016.2587808](https://doi.org/10.1109/TEVC.2016.2587808) |
| AGE-MOEA | 2019 | `agemoea` | [10.1145/3321707.3321839](https://doi.org/10.1145/3321707.3321839) |
| ETEA | 2014 | `etea` | [10.1162/evco_a_00106](https://doi.org/10.1162/evco_a_00106) |

## Reference-point based (NSGA-III family)

| Algorithm | Year | File | DOI |
|---|---|---|---|
| NSGA-III | 2014 | `nsga3` | [10.1109/TEVC.2013.2281535](https://doi.org/10.1109/TEVC.2013.2281535) |
| A-NSGA-III | 2014 | `a_nsga3` | [10.1109/TEVC.2013.2281534](https://doi.org/10.1109/TEVC.2013.2281534) |
| θ-DEA | 2016 | `theta_dea` | [10.1109/TEVC.2015.2420112](https://doi.org/10.1109/TEVC.2015.2420112) |
| AR-MOEA | 2018 | `ar_moea` | [10.1109/TEVC.2017.2749619](https://doi.org/10.1109/TEVC.2017.2749619) |

## Decomposition & region division (MOEA/D family)

| Algorithm | Year | File | DOI |
|---|---|---|---|
| MOEA/D | 2007 | `moead` | [10.1109/TEVC.2007.892759](https://doi.org/10.1109/TEVC.2007.892759) |
| MOEA/D-DE | 2009 | `moead_de` | [10.1109/TEVC.2008.925798](https://doi.org/10.1109/TEVC.2008.925798) |
| MOEA/D-DRA | 2009 | `moead_dra` | [10.1109/CEC.2009.4982949](https://doi.org/10.1109/CEC.2009.4982949) |
| MOEA/DD | 2015 | `moead_dd` | [10.1109/TEVC.2014.2373386](https://doi.org/10.1109/TEVC.2014.2373386) |
| MOEA/D-AWA | 2014 | `moead_awa` | [10.1162/EVCO_a_00109](https://doi.org/10.1162/EVCO_a_00109) |
| AdaW | 2020 | `adaw` | [10.1162/evco_a_00269](https://doi.org/10.1162/evco_a_00269) |
| MOEA/D-M2M | 2014 | `moead_m2m` | [10.1109/TEVC.2013.2281533](https://doi.org/10.1109/TEVC.2013.2281533) |
| MOEA/D-AM2M | 2018 | `moead_am2m` | [10.1109/TEVC.2017.2725902](https://doi.org/10.1109/TEVC.2017.2725902) |
| SMS-M2M | 2015 | `sms_m2m` | [10.1007/978-3-319-13356-0_35](https://doi.org/10.1007/978-3-319-13356-0_35) |
| MOEA/D-DS | 2023 | `moead_ds` | [10.1016/j.asoc.2023.110295](https://doi.org/10.1016/j.asoc.2023.110295) |
| RD-EMO | 2020 | `rd_emo` | [10.1016/j.knosys.2020.105518](https://doi.org/10.1016/j.knosys.2020.105518) |
| APRD | 2021 | `aprd` | [10.1109/ICACI52617.2021.9435909](https://doi.org/10.1109/ICACI52617.2021.9435909) |
| Liu–Gu sub-regional NSGA-II | 2011 | `liu_gu2011` | [10.1109/CEC.2011.5949848](https://doi.org/10.1109/CEC.2011.5949848) |
| I_SDE+ RD | 2018 | `isde_rd` | [10.1109/CIS2018.2018.00015](https://doi.org/10.1109/CIS2018.2018.00015) |
| DHEA | 2024 | `dhea` | [10.1007/s40747-024-01637-3](https://doi.org/10.1007/s40747-024-01637-3) |
| HLMEA | 2022 | `hlmea` | [10.1016/j.ins.2022.08.030](https://doi.org/10.1016/j.ins.2022.08.030) |

## Indicator-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| IBEA (ε+) | 2004 | `ibea_eplus` | [10.1007/978-3-540-30217-9_84](https://doi.org/10.1007/978-3-540-30217-9_84) |
| IBEA (HD) | 2004 | `ibea_hd` | [10.1007/978-3-540-30217-9_84](https://doi.org/10.1007/978-3-540-30217-9_84) |
| mIBEA | 2017 | `mibea` | [10.1109/CEC.2017.7969423](https://doi.org/10.1109/CEC.2017.7969423) |
| R2-IBEA | 2013 | `r2ibea` | [10.1109/CEC.2013.6557783](https://doi.org/10.1109/CEC.2013.6557783) |
| HypE | 2011 | `hype` | [10.1162/EVCO_a_00009](https://doi.org/10.1162/EVCO_a_00009) |
| MOMBI-II | 2015 | `mombi2` | [10.1145/2739480.2754776](https://doi.org/10.1145/2739480.2754776) |
| NIMMO | 2019 | `nimmo` | [10.1016/j.swevo.2019.06.001](https://doi.org/10.1016/j.swevo.2019.06.001) |
| IREA | 2018 | `irea` | [10.1016/j.knosys.2017.10.025](https://doi.org/10.1016/j.knosys.2017.10.025) |
| EDV | 2019 | `edv` | [10.1016/j.asoc.2018.11.041](https://doi.org/10.1016/j.asoc.2018.11.041) |
| IF-MaOEA | 2024 | `if_maoea` | [10.1016/j.asoc.2024.111881](https://doi.org/10.1016/j.asoc.2024.111881) |
| MaOEA-IAMD | 2026 | `maoea_iamd` | [10.1007/s11227-026-08362-3](https://doi.org/10.1007/s11227-026-08362-3) |

## Reference-vector / angle-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| RVEA | 2016 | `rvea` | [10.1109/TEVC.2016.2519378](https://doi.org/10.1109/TEVC.2016.2519378) |
| MaOEA-ARV | 2021 | `maoeaarv` | [10.1016/j.ins.2021.01.015](https://doi.org/10.1016/j.ins.2021.01.015) |
| MBRA | 2024 | `mbra` | [10.1007/s40747-023-01161-w](https://doi.org/10.1007/s40747-023-01161-w) |
| NRV-MOEA | 2024 | `nrv_moea` | [10.1007/s40747-024-01353-y](https://doi.org/10.1007/s40747-024-01353-y) |
| DEA-GNG | 2020 | `dea_gng` | [10.1109/TEVC.2019.2926151](https://doi.org/10.1109/TEVC.2019.2926151) |
| MaOEA/SRV | 2022 | `srv` (+ `srv_strategy`) | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |
| SRV-NSGA-III (hybrid) | 2022 | `srv_nsga3` | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |

Every file in these tables transcribes a paper. An SRV variant on a
steady-state MOEA/D carrier used to sit here as an experiment; liu2022 defines
the strategy on an NSGA-III carrier only, and measured it did not converge
(median IGD 0.41 on DTLZ2 against 0.060 for `srv`), so it left the public tree
on 2026-09-09.

## Clustering-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| CA-MOEA | 2019 | `camoea` | [10.1109/TCYB.2018.2834466](https://doi.org/10.1109/TCYB.2018.2834466) |
| CAVA-MOEA | 2025 | `cava_moea` | [10.1007/s11227-024-06496-w](https://doi.org/10.1007/s11227-024-06496-w) |
| MaOEA/AC | 2020 | `maoea_ac` | [10.1016/j.ins.2020.03.104](https://doi.org/10.1016/j.ins.2020.03.104) |
| MaOEA/C | 2019 | `maoea_c` | [10.1109/TEVC.2018.2866927](https://doi.org/10.1109/TEVC.2018.2866927) |
| MaOEA-3C | 2023 | `maoea_3c` | [10.1016/j.ins.2023.119289](https://doi.org/10.1016/j.ins.2023.119289) |
| EMyO/C | 2014 | `emyo_c` | [10.1007/978-3-319-10762-2_53](https://doi.org/10.1007/978-3-319-10762-2_53) |
| crEA | 2015 | `crea` | [10.1016/j.asoc.2015.06.020](https://doi.org/10.1016/j.asoc.2015.06.020) |
| DCEA | 2024 | `dcea` | [10.1016/j.ins.2024.120940](https://doi.org/10.1016/j.ins.2024.120940) |
| HCCA | 2023 | `hcca` | [10.1109/ACCESS.2023.3234226](https://doi.org/10.1109/ACCESS.2023.3234226) |
| LIS/LCS | 2023 | `lis_lcs` | [10.1016/j.ins.2022.12.076](https://doi.org/10.1016/j.ins.2022.12.076) |
| CLIA | 2019 | `clia` | [10.1109/TEVC.2018.2874465](https://doi.org/10.1109/TEVC.2018.2874465) |

## Archive-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| Two_Arch2 | 2015 | `two_arch2` | [10.1109/TEVC.2014.2350987](https://doi.org/10.1109/TEVC.2014.2350987) |
| NAEMO | 2019 | `naemo` | [10.1016/j.swevo.2018.12.002](https://doi.org/10.1016/j.swevo.2018.12.002) |

## Genomes

Every algorithm works on a real-valued genome with box bounds. 45 of the 58
also take a binary or mixed real + binary genome (`get_bin_vars_n() > 0` in the
`Problem<>` specialisation); the other 13 refuse it at `setup()` with a clear
message, because their reproduction operator is real-valued only: CLIA, DCEA,
DEA-GNG, DHEA, ETEA, HCCA, HLMEA, IF-MaOEA, ISDE+RD, MaOEA-3C, MOEA/D-DS, NAEMO
and Two_Arch2. Binary and mixed genomes are a C++ feature: the Python binding,
the C ABI and the TOML layer expose real-valued variables only.

## Variation operators

SBX crossover (Deb & Agrawal, 1995) · polynomial mutation (NSGA-II reference-code
variant, plus the literal unbounded form of the MOEA/D-DE paper) · DE
`rand/1/bin` (Storn & Price, 1997) with the paper's random reset or clamping
(`set_de_repair`) of out-of-box genes · Liu–Li annealing arithmetic crossover
and mutation (Liu & Li, 2009) · uniform binary crossover · bit-flip mutation.
Reference-point generators: Das–Dennis simplex lattice and its two-layer
variant. Headers: `include/mootation/operators/`, `include/mootation/das_dennis.hpp`.

Permutation problems (packing order, scheduling, routing) have no native
operators. Use the random-keys encoding: keep a real-valued genome and decode
it inside the objective function with `argsort(keys)` followed by a greedy
decoder; offspring are then always feasible and every algorithm works
unchanged.

## Constraints

Every core carries a `constraint_mode` switch, off by default because the
papers are unconstrained. `FEASIBILITY` (feasible-first, then the algorithm's
own comparison; the mode the papers with a constrained section describe),
`CDP` (Deb's constrained domination) and `EPS_CONSTRAINT` (the ε+ indicator
with a violation shift, IBEA only). Each header states where the handling
attaches: the non-dominated sort, the scalarising comparison, the archive
filter or the truncation rule, whichever is that algorithm's actual preference
relation. There is no dedicated constrained-optimisation algorithm in the
library; this is constraint handling bolted onto unconstrained methods, and the
headers say so.

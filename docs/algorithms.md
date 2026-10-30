<!-- SPDX-License-Identifier: Apache-2.0 -->
# The algorithms

76 algorithms, one header each under `include/mootation/algorithms/`, registered
in `include/mootation/algorithms.def`, and nine Python methods (scalarization
with a local solver, MOSQP, BoostDMS; `python/mootation/scalar/`). Every header opens with the paper
(authors, venue, DOI), a short scheme of one generation in the paper's own
symbols, the paper's defaults with the section they come from, and a numbered
list of declared deviations. The DOI in each table is the primary source the
implementation follows; if you publish results, cite that paper.

## Choosing one

| family | count | reach for it when | watch out for |
|---|---|---|---|
| Pareto-dominance & diversity | 12 | 2–3 objectives, a baseline everyone recognises; NSGA-II/SDR, NSGA-II with the strengthened dominance relation, for many objectives; DN-NSGA-II when several parts of the variables map onto one front; BC with SPX or REX for many variables and a large budget | crowding distance degrades above 3 objectives; BC's population is not elitist; NSGA-II/SDR's sort compares every pair directly, an angle per pair, since its relation is not transitive |
| Reference-point (NSGA-III family) | 4 | many objectives with a regular front | `pop_size` must be an exact Das–Dennis lattice size |
| Decomposition & region division (MOEA/D family) | 16 | scalarisable problems, many objectives, cheap per-generation cost | most members need an exact lattice size too (the weights are a Das–Dennis set); the steady-state members hand over one candidate at a time; the M2M members want `pop` divisible by `K` |
| Indicator-based | 14 | you care about one quality indicator (ε+, hypervolume, R2) | cost grows fast with the objective count (HypE, R2-IBEA, SMS-EMOA, MO-CMA-ES) |
| Reference-vector / angle-based | 8 | irregular or scaled fronts, where a fixed lattice misses | adaptive variants (RVEA*, DEA-GNG, NRV-MOEA, SRV) need a few generations before they help |
| Clustering-based | 11 | disconnected or irregular fronts | cluster counts are parameters the papers tuned per problem |
| Archive-based | 3 | you want the whole non-dominated history, not a fixed population; PAES: one point at a time | the archive is the output, not the working population; PAES's grid has 2^(5m) cells by default (`paes_depth` 5), many more than archive points from three objectives on |
| Constrained, two archives or populations | 2 | a constrained problem whose feasible region is small, scattered or behind infeasible barriers | `pop_size` must be an exact Das–Dennis lattice size; at 2N per generation in memory, N evaluations |
| Direct search | 3 | few variables, no population to tune; DMS and MultiGLODS deterministic | a poll costs n + 1 (DMulti-MADS) to 2n evaluations; DMS and MultiGLODS poll coordinate directions only, strong on separable problems and slow on rotated ones, and their dyadic steps land exactly on the box's centre and bounds |
| Scalarization + local solver, SQP | 8 | smooth problems, equality constraints a population cannot hold, a front to be traced point by point | local solvers: a multimodal problem (DTLZ1, DTLZ3) gives local fronts; NBI and NNC lean on the anchors, and a front whose individual minima coincide (DTLZ2) flattens NNC's simplex |
| Particle swarm | 1 | 2–3 objectives, a cheap step and one batch of evaluations per step | the reference codes disagree on what SMPSO is; the presets are their readings, and the default's clamp lands exactly on the middle and the bounds of the box |
| Estimation of distribution | 1 | continuous variables linked along a Pareto set that is a smooth (m − 1)-dimensional manifold | a set that is not one (disconnected, full-dimensional, discrete) is outside its model; many local fronts hold it (its paper's F10 is not solved); a principal component analysis per cluster each generation costs O(n³) |
| Simulated annealing | 1 | one evaluation at a time, an archive of every non-dominated point as the answer | the schedules are laid out over the whole budget, so a run of 10 000 evaluations is not the first 10 000 of a longer one; nothing to evaluate in parallel |

Nineteen algorithms require `pop_size` to be an exact Das–Dennis lattice size
for the objective count (91 at M = 3, 210 at M = 5, ...): A-NSGA-III, AdaW,
C-TAEA, crEA, EDV, IREA, MBRA, MOEA/D, MOEA/D-AWA, MOEA/DD, MOEA/D-DE, MOEA/D-DRA,
MOMBI-II, NSGA-III, RVEA, RVEA*, MaOEA/SRV, SRV-NSGA-III and θ-DEA; the
M2M family needs `pop` divisible by its subregion count `K`; the others round
an unattainable request to the nearest lattice and say so through
`set_warn_handler`. `python -m mootation.run --check` names every such
constraint before a run starts; `Result::ignored` (C++), `res.ignored` (Python) and
`moo_ignored_*` (C ABI) name every knob the chosen algorithm does not have.

## If your objectives are not commensurate

Objectives in different units — a mass in kilograms beside a stress in pascals
— are the case this field mostly assumes away. Two different things can go
wrong, and they are worth separating.

**The units of all objectives at once.** Multiplying every objective by the
same constant must change nothing, and `tests/test_scale_invariance.cpp`
enforces exactly that on every push; the algorithms that depend on a constant
of their own paper are listed there by name.

**The ratio between axes.** That is a different property and most of these
algorithms do not have it, by design rather than by defect: a weight vector on
the simplex, a penalty-based boundary intersection, an angle between objective
vectors and a niche radius all mean something only when the axes are
comparable. Some algorithms carry their own normalisation, the rest expect you
to supply it. If your objectives differ by orders of magnitude, scale them
yourself before the run — the `SDTLZ` family in the benchmark registry exists
to test exactly this, and RVEA's own paper introduces itself on scaled
problems.

**Scaling is not free, so ask what the spread means.** The ranges of your
objectives can differ for two unrelated reasons, and they call for opposite
answers. If they differ because of the units you chose — a mass in grams beside
a length in metres — the spread carries nothing and removing it can only help:
R2-IBEA's `set_normalize(true)` removes such a skew. If they differ because of
the problem, removing the spread throws information away: on ZDT1 the wide
axis is the direction of convergence (`g` falls towards 1 as the population
approaches the front), and the same switch is then the wrong trade
(`r2ibea.hpp`, R2I-NORM).

**What the cost actually is, and who inherits it.** It is not convergence: the
scaled reading still brings the population to the front, over another part of
it. That is a re-mapping, and it belongs to a whole class of methods. R2
allocates effort by direction: each weight vector on the simplex owns a piece
of the front, and rescaling the axes moves the pieces. The MOEA/D family, the
NSGA-III family, RVEA and θ-DEA all allocate the same way. IBEA-ε+ owns no such
partition — it compares two solutions and decides an *order*, and an order
survives rescaling far better. So this is not a property of normalisation in
general: it is what happens when you rescale the axes of a method that has
divided the front between directions.

**Which case are you in? There is a cheap test.** Watch the ratio between the
per-objective ranges of the answer set over the first few generations. A unit
is a constant and the ratio holds steady; a property of the problem moves as
the search closes on the front. On ZDT1 the ratio slides towards 1 over a run —
`f2`'s range is set by `g`, which falls towards 1 as the population reaches the
front, so the slide *is* the convergence — while constant factors on the axes
multiply a steady ratio by a constant and leave it steady. Steady ratio: scale.
Sliding ratio: do not. A problem that is both skewed and drifting is not
covered by this test.

## One objective

The cores are multi-objective. With `n_objs = 1`, 22 of them refuse in
`Settings::validate()` and name their reason (`single_objective_refusal`,
`settings.hpp`): nsga3, a_nsga3, theta_dea, moead, moead_de, moead_dra, moead_dd,
moead_awa, adaw, mombi2, irea, edv, rvea, rvea_star, mbra, srv, srv_nsga3 and
ctaea (their reference directions come from a simplex lattice, which at M = 1
is one point), rd_emo, r2ibea, if_maoea and r2_emoa (each its own reason).

These ten run with one objective but are not single-objective algorithms
there; what they become, read from their code:

| core | at M = 1 |
|---|---|
| etea | the archive keeps the best point and then the sparsest ones, not the next best |
| moead_m2m | all K directions are the same: one subregion of K keeps the best, the other K - 1 are refilled at random |
| sms_m2m | the same partition as moead_m2m: one elitist block, K - 1 random refills |
| moead_am2m | one weight, so the population is one point: a (1+1)-EA with Liu-Li mutation, reseeded by clones of the best |
| liu_gu2011 | all directions the same: the population shrinks to about the square root of pop_size |
| isde_rd | removal deletes the first non-best points: generational replacement with the best kept |
| hcca | pop_size is ignored and the answer is one point: a (1+3)-EA with polynomial mutation |
| naemo | one line, one evaluation per step, no mutation by default: the archive shrinks to a few points and stalls |
| mosa | an archive of one point and energy 1 for every worse point: mostly a random walk |
| mo_cma_es | with mu >= 2 (the settings ask for it) success is beating the worst of mu, not the parent; at mu = 1 it would be the (1+1)-CMA-ES |

## How each algorithm normalizes

The way each core puts its objectives on one scale, after the classification of
Tanabe (*Investigating normalization in preference-based evolutionary
multi-objective optimization using a reference point*, Applied Soft Computing,
2024). The table is `mootation.run.algorithms.NORMALIZATION`, read from each
core's header and code; the header is the source.

- **PP** — the ideal and the nadir point are the least and greatest value of
  each objective over the current set (P ∪ Q, P, or the set the step works on).
- **BP** — the ideal point is the least value ever evaluated; the nadir point
  comes from the current set.
- **hyperplane** — NSGA-III's: the intercepts of the hyperplane through the
  extreme points, the set's nadir where it degenerates.
- **ideal** — translated by an ideal point only, not scaled.
- **none** — the raw objectives.
- **own** — a scheme of the core's own; the last column says which.

Tanabe's BA (the nadir from an archive of extreme points) is the way of none of
these cores. "Selection" is the environmental selection.

| algorithm | way | what it serves, and where the points come from |
|---|---|---|
| `nsga2` | none | selection: crowding distance, each objective divided by its range on the front |
| `nsga2_sdr` | PP | selection: the SDR frame over P ∪ Q (sdr.hpp, make_frame) |
| `dn_nsga2` | none | selection: crowding in the decision space, raw distances |
| `bc_spx` | none | selection and mating: raw objectives |
| `bc_rex` | none | as bc_spx |
| `spea2` | none | selection: raw distances in the objective space |
| `spea2_sde` | none | selection: shifted density estimation on raw objectives |
| `grea` | PP | selection and mating: the grid over the set's least and greatest values |
| `vaea` | PP | selection: over P ∪ Q (Alg. 3) |
| `agemoea` | hyperplane | selection: ideal point of the first front of P ∪ Q (this generation), intercepts from its extreme points |
| `etea` | none | selection: raw objectives (the paper prescribes none) |
| `gde3` | none | selection: crowding distance divided by the front's range |
| `nsga3` | hyperplane | selection: ideal point over every generation, extreme points kept from earlier generations (NSGA3-N1) |
| `a_nsga3` | hyperplane | as nsga3 |
| `theta_dea` | hyperplane | selection: ideal point over every generation |
| `ar_moea` | PP | mating and the reference-point adaptation: ideal and nadir of P |
| `moead` | ideal | subproblems: the least value ever evaluated (z) |
| `moead_de` | ideal | as moead |
| `moead_dra` | ideal | as moead |
| `moead_dd` | ideal | subproblems and association: the least value ever evaluated |
| `moead_awa` | ideal | subproblems: the least value ever evaluated less 1e-7 |
| `adaw` | own | subproblems: the least value ever evaluated less 1e-4 (ideal); the archive and the weight adaptation: PP over A ∪ P |
| `moead_m2m` | none | subproblems: raw objectives, shifted only where a least value is negative |
| `moead_am2m` | own | directions, association and subproblems: ideal and nadir of a snapshot taken at the start and at each adaptation (AM2M-norm) |
| `sms_m2m` | none | as moead_m2m |
| `moead_ds` | hyperplane | subproblems: ideal point of P, intercepts from its extreme points |
| `rd_emo` | ideal | subproblems: the least value ever evaluated (RDE-3) |
| `aprd` | own | subregions: each subregion's own ideal point, the archive shifted to the origin |
| `liu_gu2011` | ideal | association: the least values of the internal population |
| `isde_rd` | ideal | angles: the least values of the pool (ISDE-3) |
| `dhea` | PP | selection: over the population |
| `hlmea` | none | raw objectives, shifted only where a least value is negative (HLMEA-5) |
| `ibea_eplus` | PP | fitness: the indicator over P ∪ Q scaled to [0, 1] (IEP-4) |
| `ibea_hd` | PP | as ibea_eplus; reference point 2 in the scaled space |
| `mibea` | PP | fitness: over the first front of P ∪ Q |
| `r2ibea` | none | fitness: raw objectives (set_normalize(true) scales them) |
| `hype` | own | fitness: a reference point from the pool's greatest values (no scaling) |
| `mombi2` | own | selection: least value ever evaluated, nadir of the previous population with the paper's variance rule |
| `nimmo` | none | raw objectives; the indicator values are scaled within the set |
| `irea` | BP | fitness: least value ever evaluated, greatest over S_t |
| `edv` | ideal | angles: the least value ever evaluated (EDV-3) |
| `if_maoea` | PP | mating, selection and the archive: over the set each step works on |
| `maoea_iamd` | PP | mating and selection: over P, and over P ∪ Q |
| `sms_emoa` | own | contributions: no scaling; from three objectives the reference point follows the front (SMS-7) |
| `r2_emoa` | own | R2: raw objectives, the utopian point from the least value ever evaluated and nadir(Q) (normalize = true scales by ideal(Q) and nadir(Q)) |
| `mo_cma_es` | own | contributions: no scaling; reference point nadir + (nadir − ideal) over Q (MOCMA-4) |
| `rvea` | own | selection: translated by the least values of P ∪ Q (ideal); vector adaptation: the survivors' ranges (PP) |
| `rvea_star` | own | as rvea |
| `maoeaarv` | PP | the knee and the reference: ideal and nadir of P |
| `mbra` | PP | selection: over the pool |
| `nrv_moea` | own | selection: least and greatest over the accepted fronts, the scale kept for f_r·MaxG generations at a time |
| `dea_gng` | BP | selection (NSGA-III's): least value ever evaluated, greatest over the first front |
| `srv` | PP | selection: over S_t |
| `srv_nsga3` | hyperplane | as nsga3 |
| `camoea` | PP | clustering: within the last front |
| `cava_moea` | PP | fitness and clustering: over the pool |
| `maoea_ac` | hyperplane | selection: ideal point of S, nadir by NSGA-III's intercepts |
| `maoea_c` | PP | selection: over P ∪ Q (over its non-dominated points below four objectives) |
| `maoea_3c` | hyperplane | selection: over the set, extreme points by achievement scalarizing |
| `emyo_c` | ideal | the least value ever evaluated |
| `crea` | BP | selection: least value ever evaluated, greatest over P ∪ Q |
| `dcea` | ideal | association: the previous generation's least values |
| `hcca` | own | clustering: PP; subproblems: the least value ever evaluated |
| `lis_lcs` | own | indicator and clustering: PP over the set; the least value ever evaluated |
| `clia` | own | the classifier: points projected onto the simplex |
| `two_arch2` | PP | the diversity archive's truncation: over the pool (TA2-1) |
| `naemo` | ideal | the archive's ideal point |
| `paes` | own | the grid over the archive's least and greatest values |
| `ctaea` | ideal | the least values of the set at hand (CTAEA-2) |
| `ccmo` | none | crowding distance divided by the front's range |
| `dms` | none | raw objectives (the step is scaled to the variables' box) |
| `dmulti_mads` | own | the frame centre: the gaps along each objective scaled by its range |
| `multiglods` | none | raw objectives (the variables' box normalized) |
| `smpso` | none | the leaders' archive: crowding distance divided by the front's range |
| `rm_meda` | none | raw objectives |
| `mosa` | own | energies: the archive's ideal point and ranges |

## Pareto-dominance & diversity-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| NSGA-II | 2002 | `nsga2` | [10.1109/4235.996017](https://doi.org/10.1109/4235.996017) |
| NSGA-II/SDR | 2019 | `nsga2_sdr` | [10.1109/TEVC.2018.2866854](https://doi.org/10.1109/TEVC.2018.2866854) |
| DN-NSGA-II | 2016 | `dn_nsga2` | [10.1109/CEC.2016.7744093](https://doi.org/10.1109/CEC.2016.7744093) |
| BC (best-children), SPX / REX | 2019 | `bc_spx`, `bc_rex` | [10.1145/3321707.3321754](https://doi.org/10.1145/3321707.3321754) |
| SPEA2 | 2001 | `spea2` | [10.3929/ethz-a-004284029](https://doi.org/10.3929/ethz-a-004284029) |
| SPEA2+SDE | 2014 | `spea2_sde` | [10.1109/TEVC.2013.2262178](https://doi.org/10.1109/TEVC.2013.2262178) |
| GrEA | 2013 | `grea` | [10.1109/TEVC.2012.2227145](https://doi.org/10.1109/TEVC.2012.2227145) |
| VaEA | 2017 | `vaea` | [10.1109/TEVC.2016.2587808](https://doi.org/10.1109/TEVC.2016.2587808) |
| AGE-MOEA | 2019 | `agemoea` | [10.1145/3321707.3321839](https://doi.org/10.1145/3321707.3321839) |
| ETEA | 2014 | `etea` | [10.1162/evco_a_00106](https://doi.org/10.1162/evco_a_00106) |
| GDE3 | 2005 | `gde3` | [10.1109/CEC.2005.1554717](https://doi.org/10.1109/CEC.2005.1554717) |

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
| SMS-EMOA | 2005 | `sms_emoa` | [10.1007/978-3-540-31880-4_5](https://doi.org/10.1007/978-3-540-31880-4_5) |
| R2-EMOA | 2013 | `r2_emoa` | [10.1007/978-3-642-44973-4_8](https://doi.org/10.1007/978-3-642-44973-4_8) |
| MO-CMA-ES (μ+1, success by population) | 2010 | `mo_cma_es` | [10.1145/1830483.1830573](https://doi.org/10.1145/1830483.1830573) |

## Reference-vector / angle-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| RVEA | 2016 | `rvea` | [10.1109/TEVC.2016.2519378](https://doi.org/10.1109/TEVC.2016.2519378) |
| RVEA* (regenerated vectors, §VI) | 2016 | `rvea_star` | [10.1109/TEVC.2016.2519378](https://doi.org/10.1109/TEVC.2016.2519378) |
| MaOEA-ARV | 2021 | `maoeaarv` | [10.1016/j.ins.2021.01.015](https://doi.org/10.1016/j.ins.2021.01.015) |
| MBRA | 2024 | `mbra` | [10.1007/s40747-023-01161-w](https://doi.org/10.1007/s40747-023-01161-w) |
| NRV-MOEA | 2024 | `nrv_moea` | [10.1007/s40747-024-01353-y](https://doi.org/10.1007/s40747-024-01353-y) |
| DEA-GNG | 2020 | `dea_gng` | [10.1109/TEVC.2019.2926151](https://doi.org/10.1109/TEVC.2019.2926151) |
| MaOEA/SRV | 2022 | `srv` (+ `srv_strategy`) | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |
| SRV-NSGA-III (hybrid) | 2022 | `srv_nsga3` | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |

Every file in these tables transcribes a paper. An SRV variant on a
steady-state MOEA/D carrier used to sit here as an experiment; liu2022 defines
the strategy on an NSGA-III carrier only, so it left the public tree on
2026-09-09.

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
| PAES | 2000 | `paes` | [10.1162/106365600568167](https://doi.org/10.1162/106365600568167) |

## Constrained, two archives or populations

| Algorithm | Year | File | DOI |
|---|---|---|---|
| C-TAEA | 2019 | `ctaea` | [10.1109/TEVC.2018.2855411](https://doi.org/10.1109/TEVC.2018.2855411) |
| CCMO | 2021 | `ccmo` | [10.1109/TEVC.2020.3004012](https://doi.org/10.1109/TEVC.2020.3004012) |

C-TAEA keeps a convergence archive, which is the population and the answer,
feasible first, and a diversity archive that ignores the constraints and fills
the subregions the first one leaves thin; parents come from either, by the
share of non-dominated points each holds. The header lists where it follows
the authors' code over the paper's text (the normalization, one child per
pair).

CCMO evolves the problem and a helper — the same objectives without the
constraints — in two populations of N; each breeds N/2 children and both are
cut from themselves and all N children, so a generation costs N evaluations
(2N at the start). The paper's framework takes any algorithm; this is its
NSGA-II form, with SPEA2's truncation of the last front, as the paper's
experiments run it — none of the other cores hands its children over before
its own selection (header, CCMO-1).
## Direct search

| Algorithm | Year | File | DOI |
|---|---|---|---|
| DMS | 2011 | `dms` | [10.1137/10079731X](https://doi.org/10.1137/10079731X) |
| DMulti-MADS | 2021 | `dmulti_mads` | [10.1007/s10589-021-00272-9](https://doi.org/10.1007/s10589-021-00272-9) |
| MultiGLODS | 2018 | `multiglods` | [10.1007/s10898-018-0618-1](https://doi.org/10.1007/s10898-018-0618-1) |
| BoostDMS | 2020 | `boostdms` (Python) | [10.1007/s10589-020-00233-8](https://doi.org/10.1007/s10589-020-00233-8) |

Not evolutionary: a list of nondominated points with a step size each, polled
along the coordinate directions, the step halved where a poll finds nothing.
DMulti-MADS polls n + 1 directions on a mesh around the point of the largest
frame in the least crowded part of the list, and grows the frames of the points
that dominate or extend it (preliminary: its poll directions and its handling of
the bounds, a poll step cut to the last mesh point inside the box, are our
reading until checked against the granular-mesh and OrthoMADS papers it cites;
the cut is ours, not in the n + 1 directions paper or the NOMAD 4 guide);
MultiGLODS keeps the points of every search it has
started, active or merged, and starts new ones from a Sobol sequence after
three failures in a row; BoostDMS (`include/mootation/scalar/boostdms.hpp`) is DMS with a search step that
minimizes quadratic models of the objectives built from the evaluated points.
Deterministic, and it stops on its own when every step size is below 10⁻³ of
the range. Its answer is the list reduced to `pop_size` by DSS
(`include/mootation/dss.hpp`), the same selection the run layer uses for
reference fronts and archives.

## Particle swarm

| Algorithm | Year | File | DOI |
|---|---|---|---|
| SMPSO | 2009 | `smpso` | [10.1109/MCDM.2009.4938830](https://doi.org/10.1109/MCDM.2009.4938830) |

A swarm of `pop_size` particles, each pulled towards its own best and a leader
drawn from a bounded archive of non-dominated points, its velocity multiplied
by a constriction factor and bounded per variable; the archive is the answer.
The paper leaves the inertia weight, the sign of the constriction factor, the
velocity at the bound and the exact turbulence open, and the reference codes
read them differently: `smpso_preset` picks one reading (`jmetal`, the
default; `platemo`; `fair`; `omopso`, its predecessor OMOPSO of Reyes Sierra &
Coello Coello, EMO 2005), and every part of it is a knob of its own
([running.md](running.md)).

## Estimation of distribution

| Algorithm | Year | File | DOI |
|---|---|---|---|
| RM-MEDA | 2008 | `rm_meda` | [10.1109/TEVC.2007.894202](https://doi.org/10.1109/TEVC.2007.894202) |

No crossover or mutation: each generation the population is split into
`n_clusters` (5) clusters by a local principal component analysis, each
cluster's points are modelled as an (m − 1)-dimensional box along its first
principal components, extended by `rm_extension` (a quarter) of its range on
each side, plus Gaussian noise with the variance left in the other
components; `pop_size` offspring are sampled from the models, a cluster drawn
in proportion to its box's volume, and the population is selected from both by
non-dominated sorting and a crowding distance recomputed after every removal.
The model rests on the regularity of the Pareto set, an (m − 1)-dimensional
manifold in the variables under the Karush–Kuhn–Tucker conditions.

## Scalarization with a local solver

| Algorithm | Year | Name | DOI |
|---|---|---|---|
| ε-constraint | 1971 | `eps_sqp`, `eps_nm`, `eps_cmaes` | [10.1109/TSMC.1971.4308298](https://doi.org/10.1109/TSMC.1971.4308298) |
| ε-constraint, adaptive (`sub_plan = laumanns`) | 2006 | `scalar` | [10.1016/j.ejor.2004.08.029](https://doi.org/10.1016/j.ejor.2004.08.029) |
| Chebyshev, smooth form | — | `tch_sqp` | — |
| Normal Boundary Intersection | 1998 | `nbi_sqp` | [10.1137/S1052623496307510](https://doi.org/10.1137/S1052623496307510) |
| Normalized Normal Constraint | 2003 | `nnc_sqp` | [10.1007/s00158-002-0276-1](https://doi.org/10.1007/s00158-002-0276-1) |
| Pascoletti–Serafini (`scalarization = ps`) | 1984 | `scalar` | [10.1007/BF00934564](https://doi.org/10.1007/BF00934564) |
| Multiobjective SQP | 2016 | `mosqp` | [10.1137/15M1016424](https://doi.org/10.1137/15M1016424) |

Not evolutionary and not cores of `algorithms.def`: C++ in
`include/mootation/scalar/` (since task 17, stage 6; `_core.scalar` from
Python), run by name through `minimize()` and in campaigns like any algorithm.
The front is a sequence of single-objective subproblems — the anchors
first (each objective's lexicographic minimum), then a plan of subproblems over
the normalized objectives — each solved by a local method from the best point
the archive holds for it: the library's SQP with finite-difference gradients
(`sqp`; `slsqp.hpp`, Nocedal & Wright's line-search SQP whose quadratic
subproblem is Lawson & Hanson's least-squares problem, as in Kraft's SLSQP),
Nelder–Mead (`nm`; `nelder_mead.hpp`, Lagarias et al. 1998 with Gao & Han's
adaptive coefficients) or a CMA-ES written from Hansen's tutorial,
arXiv:1604.00772 (`cmaes`). Every evaluation,
finite differences included, is counted and archived; the answer is the
archive's feasible non-dominated points, reduced to `pop_size` by DSS. A failed
evaluation of an iterate ends that subproblem's SQP solve (SCALAR-6, in
`scalar/scalar.hpp`), so with failures these methods may end with budget left,
as they do once every subproblem of the plan is solved; the budget is not
spent elsewhere. Real variables only. `scalar` with its knobs covers all of them; the six names are
its presets. `mosqp` (Fliege & Vaz) is beside them without a scalarization: a
set of points spread by one SQP step per objective, then each moved along the
step that improves all objectives at once. [running.md](running.md) has the
knobs and what each part reads from its source.

## Simulated annealing

| Algorithm | Year | File | DOI |
|---|---|---|---|
| MOSA | 2026 | `mosa` | a composition, below |

Not evolutionary and not from one paper: annealing chains over an archive of
every non-dominated point evaluated, put together from DBMOSA's unconstrained
archive, energy (the share of the archive dominating a point), Laplace move
and restarts from the archive
([10.1109/TEVC.2007.904345](https://doi.org/10.1109/TEVC.2007.904345)),
AMOSA's energies against an archive
([10.1109/TEVC.2007.900837](https://doi.org/10.1109/TEVC.2007.900837); its
bounded archive and informed reseeding are not used), and the chains of Pareto
simulated annealing with weights that adapt to push them apart and its
weighted-sum acceptance (Czyżak & Jaszkiewicz, J. Multi-Crit. Decis. Anal.
7(1), 1998). The header lists which parts are whose, where they depart from
the papers and which are ours (the default energy, `sa_energy = cmb`, the
Chebyshev energy, and the steps `mix` and `adaptive`). One step is one evaluation, so annealing is sequential: there
is nothing to evaluate in parallel, and `batch=` gets one point at a time.
The temperature and the step scale fall over the whole budget, so the run
depends on `max_evaluations` (`run/budget.py`). Its answer is the archive
reduced to `pop_size` by DSS, published every `pop_size` steps.

## Genomes

Every algorithm works on a real-valued genome with box bounds. 51 of the 76
also take a binary or mixed real + binary genome (`get_bin_vars_n() > 0` in the
`Problem<>` specialisation); the other 25 refuse it at `setup()` with a clear
message, because their reproduction operator is real-valued only, or, for
DN-NSGA-II, its niche is a distance in the real variables: BC (SPX, REX), CLIA,
DCEA, DEA-GNG, DHEA, DMS, DMulti-MADS, DN-NSGA-II, ETEA, GDE3, HCCA, HLMEA,
IF-MaOEA, ISDE+RD, MaOEA-3C, MO-CMA-ES, MOEA/D-DS, MOSA, MultiGLODS, NAEMO, PAES,
RM-MEDA, SMPSO and Two_Arch2. The Python methods (`python/mootation/scalar/`) take real
variables only. The Python binding reaches the binary genome
through `minimize(var_types=[..., "bin"])` (`mootation/var_types.py`), which also declares
integer and discrete variables — rounded in the genome by the vault (`DataVault::set_snap`)
for every algorithm; the C ABI and the TOML layer expose real-valued variables only.

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
with a violation shift, IBEA only); the Python layer adds the ε level
(`constraint_mode = "eps_level"`, docs/running.md), which the vault applies to
any feasibility-first core. Each header states where the handling
attaches: the non-dominated sort, the scalarising comparison, the archive
filter or the truncation rule, whichever is that algorithm's actual preference
relation. For the 64 that were written for unconstrained problems this is
constraint handling bolted on, and the headers say so; C-TAEA and CCMO are
built for constraints.

Every core carries a point's violation through every rewrite of its
population. A core that keeps its own copies of points and writes its
population back from them keeps each point whole (`VaultPoint`: the
variables and the integer genome, the objectives, the constraint values, |h|
of the equalities, whether the evaluation failed) and writes it back whole,
at the tolerance, ε level and failure penalty of the moment
(`data_vault.hpp`, WRITING BACK); the copies it compares again in a later
step are brought to that moment first. A point written without its
constraint values is refused. A test checks every core after every step on
problems with inequalities, equalities (under the linear tolerance schedule
too) and failed evaluations.

## Batches and single points

A core asks for its evaluations either a generation at a time — the vault
evaluates every new point of a step in one call, `sync()` — or one point at a
time, because what it does next depends on that point (a steady-state or a
decomposition scheme, annealing). `minimize(batch=...)` hands the evaluator
whatever the core asks for at once: a whole generation, where the rows can be
spread over processes or the nodes of a cluster, or a single row, where
nothing can. The order a core evaluates in is its paper's, and is not changed
to make batches. `tools/perf/batches.py` writes this table (DTLZ2 at three
objectives, 3 000 evaluations; the share of the evaluations that came one at
a time, and the size of the other calls):

| core | population | calls | evaluations | in single calls | batch size (median, max) |
|---|---|---|---|---|---|
| nsga2 | 100 | 30 | 3000 | 0 % | 100, 100 |
| nsga2_sdr | 100 | 30 | 3000 | 0 % | 100, 100 |
| dn_nsga2 | 100 | 30 | 3000 | 0 % | 100, 100 |
| spea2 | 100 | 30 | 3000 | 0 % | 100, 100 |
| spea2_sde | 100 | 30 | 3000 | 0 % | 100, 100 |
| grea | 100 | 30 | 3000 | 0 % | 100, 100 |
| vaea | 100 | 30 | 3000 | 0 % | 100, 100 |
| agemoea | 100 | 30 | 3000 | 0 % | 100, 100 |
| etea | 100 | 2901 | 3000 | 97 % | 100, 100 |
| gde3 | 100 | 30 | 3000 | 0 % | 100, 100 |
| nsga3 | 91 | 33 | 3003 | 0 % | 91, 91 |
| a_nsga3 | 91 | 33 | 3003 | 0 % | 91, 91 |
| theta_dea | 91 | 33 | 3003 | 0 % | 91, 91 |
| ar_moea | 100 | 30 | 3000 | 0 % | 100, 100 |
| moead | 100 | 2901 | 3000 | 97 % | 100, 100 |
| moead_de | 100 | 2901 | 3000 | 97 % | 100, 100 |
| moead_dra | 100 | 2901 | 3000 | 97 % | 100, 100 |
| moead_dd | 100 | 2901 | 3000 | 97 % | 100, 100 |
| moead_awa | 100 | 2901 | 3000 | 97 % | 100, 100 |
| adaw | 100 | 2901 | 3000 | 97 % | 100, 100 |
| moead_m2m | 100 | 30 | 3000 | 0 % | 100, 100 |
| moead_am2m | 100 | 2901 | 3000 | 97 % | 100, 100 |
| sms_m2m | 100 | 30 | 3000 | 0 % | 100, 100 |
| moead_ds | 100 | 2901 | 3000 | 97 % | 100, 100 |
| rd_emo | 100 | 2901 | 3000 | 97 % | 100, 100 |
| aprd | 100 | 2901 | 3000 | 97 % | 100, 100 |
| liu_gu2011 | 100 | 3000 | 3000 | 100 % | — |
| isde_rd | 100 | 2901 | 3000 | 97 % | 100, 100 |
| dhea | 100 | 2901 | 3000 | 97 % | 100, 100 |
| hlmea | 100 | 2971 | 3070 | 97 % | 100, 100 |
| ibea_eplus | 100 | 30 | 3000 | 0 % | 100, 100 |
| ibea_hd | 100 | 30 | 3000 | 0 % | 100, 100 |
| mibea | 100 | 30 | 3000 | 0 % | 100, 100 |
| r2ibea | 100 | 30 | 3000 | 0 % | 100, 100 |
| hype | 100 | 30 | 3000 | 0 % | 100, 100 |
| mombi2 | 91 | 33 | 3003 | 0 % | 91, 91 |
| nimmo | 100 | 2901 | 3000 | 97 % | 100, 100 |
| irea | 100 | 30 | 3000 | 0 % | 100, 100 |
| edv | 100 | 30 | 3000 | 0 % | 100, 100 |
| if_maoea | 100 | 2901 | 3000 | 97 % | 100, 100 |
| maoea_iamd | 100 | 42 | 3096 | 0 % | 100, 100 |
| sms_emoa | 100 | 2901 | 3000 | 97 % | 100, 100 |
| mo_cma_es | 100 | 2901 | 3000 | 97 % | 100, 100 |
| rvea | 100 | 30 | 3000 | 0 % | 100, 100 |
| rvea_star | 100 | 30 | 3000 | 0 % | 100, 100 |
| maoeaarv | 100 | 30 | 3000 | 0 % | 100, 100 |
| mbra | 100 | 30 | 3000 | 0 % | 100, 100 |
| nrv_moea | 91 | 33 | 3003 | 0 % | 91, 91 |
| dea_gng | 100 | 2901 | 3000 | 97 % | 100, 100 |
| srv | 100 | 30 | 3000 | 0 % | 100, 100 |
| srv_nsga3 | 91 | 33 | 3003 | 0 % | 91, 91 |
| camoea | 100 | 30 | 3000 | 0 % | 100, 100 |
| cava_moea | 100 | 30 | 3000 | 0 % | 100, 100 |
| maoea_ac | 100 | 30 | 3000 | 0 % | 100, 100 |
| maoea_c | 100 | 30 | 3000 | 0 % | 100, 100 |
| maoea_3c | 100 | 2901 | 3000 | 97 % | 100, 100 |
| emyo_c | 100 | 30 | 3000 | 0 % | 100, 100 |
| crea | 100 | 30 | 3000 | 0 % | 100, 100 |
| dcea | 100 | 2901 | 3000 | 97 % | 100, 100 |
| hcca | 100 | 2901 | 3000 | 97 % | 100, 100 |
| lis_lcs | 100 | 60 | 3061 | 0 % | 50, 100 |
| clia | 100 | 2901 | 3000 | 97 % | 100, 100 |
| two_arch2 | 100 | 2901 | 3000 | 97 % | 100, 100 |
| naemo | 100 | 2913 | 3012 | 97 % | 100, 100 |
| ctaea | 100 | 30 | 3000 | 0 % | 100, 100 |
| ccmo | 100 | 30 | 3000 | 0 % | 100, 100 |
| dms | 100 | 250 | 3000 | 0 % | 12, 12 |
| mosa | 100 | 2981 | 3000 | 99 % | 20, 20 |

A model whose objectives and constraints come out of one computation is given
as one function, `minimize(batch=fn, batch_with_constraints=True)`, `fn(X)`
returning the pair (objective rows, constraint rows): it runs once per point.
For an expensive model the batch can be spread over processes outside the
core, without changing it:

```python
from concurrent.futures import ProcessPoolExecutor
import mootation

def model(x):                      # one run of the solver: seconds or more
    ...
    return [f1, f2]

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = mootation.minimize(
            model, bounds=[(0.0, 1.0)] * 10, n_objs=2, algorithm="nsga2",
            pop_size=40, max_evaluations=2000,
            batch=lambda X: list(pool.map(model, X)))
```

The rows come back in the order they were sent (`pool.map` keeps it), which
is what the core expects. A core of the second kind still works with it, a
point per call.


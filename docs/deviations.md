# Departures from the papers

Written by `tools/deviations.py` from the headers of the cores (`include/mootation/algorithms`);
do not edit by hand — edit the header and run the tool. The header is the source: there each item
is given in full, with its reasons and references.

*Default run*: `yes` — the default run does what the item describes; `no: option` — the item is
about something off by default; `no: fixed` — fixed, resolved, retired, or not a deviation. The
column is read from the item's wording (see the tool's docstring), not stated by the item.

368 items: 300 yes, 12 no: option, 56 no: fixed. Headers that declare none: `spea2_sde`, `adaw`, `camoea`, `maoea_c`. Headers with no item this tool recognises: `dn_nsga2`, `bc_rex`, `rvea_star`, `mosa`.

| algorithm | item | what | default run | file |
|---|---|---|---|---|
| `nsga2` | nsga2.1 | CD normalization uses min/max of the current FRONT (the paper is ambiguous); | yes | `nsga2.hpp` |
| `nsga2` | nsga2.2 | tournament "with replacement" (a==b is possible) instead of two permutations; | yes | `nsga2.hpp` |
| `nsga2` | nsga2.3 | deterministic tournament tie-breaks (on a tie — `b`); | yes | `nsga2.hpp` |
| `nsga2` | nsga2.4 | binary genome: uniform crossover p=1 instead of single-point pc=0.9. | yes | `nsga2.hpp` |
| `nsga2_sdr` | SDR-A | duplicates stay: the population is always N. | yes | `nsga2_sdr.hpp` |
| `nsga2_sdr` | SDR-B | crowding distance, the tournament and the cut of the last layer are nsga2's, in the raw objectives (nsga2.hpp, its deviations included: CD over the layer's own range, tournament with replacement, a tie goes to the sec... | yes | `nsga2_sdr.hpp` |
| `bc_spx` | BC-1 | μ is pop_size; ⌊100 ln n⌋ (Tušar & Filipič, as §3 takes it) is the paper's, worth passing; k = n + 1 > μ is refused. | yes | `bc_spx.hpp` |
| `bc_spx` | BC-2 | A child outside the box: bound_repair (reflect by default, as the library's multi-parent crossovers, multi_parent.hpp MP-4); the paper's BBOB problems are unbounded. | yes | `bc_spx.hpp` |
| `bc_spx` | BC-3 | Constraints: Deb's constrained domination in the sorting (constraint_mode not NONE); the paper has none. | yes | `bc_spx.hpp` |
| `bc_spx` | BC-4 | The answer is P, which BC does not keep elitist: the paper measures an unbounded archive of every non-dominated point (COCO's), not P; a campaign's archive scenario is that measure here. | yes | `bc_spx.hpp` |
| `bc_spx` | BC-5 | A tie of level and crowding distance: the earlier candidate (BC: Q in order; | yes | `bc_spx.hpp` |
| `spea2` | spea2.1 | the `<=` tournament tie-break deterministically awards the win to the first contestant; | yes | `spea2.hpp` |
| `spea2` | spea2.2 | k is FLOORED. §3.1 writes k = √(N+N̄) with no rounding rule, and it is an index into a sorted distance list, so it must be made integral; this port takes ⌊·⌋. | yes | `spea2.hpp` |
| `spea2` | spea2.3 | σ^k is computed SELF-EXCLUSIVELY, and the report contradicts itself on this point. | yes | `spea2.hpp` |
| `grea` | grea.1 | the PD propagation in GR_adjustment is performed even without increasing PD(p); this is behaviourally equivalent to the letter of Alg.3 line 11 by the transitivity of grid dominance. | yes | `grea.hpp` |
| `vaea` | vaea.1 | Definition 1/3: on multiple minima the first is taken, not a random one; each extreme e_i is the argmin over the WHOLE F_l (Definition 3), and a solution that is the extreme of several axes enters P once ("we only add... | yes | `vaea.hpp` |
| `vaea` | vaea.2 | Alg.4 lines 1-7, the bootstrap when P = ∅: the m extreme solutions are added first, then "the first m best converged solutions". | yes | `vaea.hpp` |
| `vaea` | vaea.3 | a defensive top-up of P from the remaining pool in degenerate cases (line 1 of Alg.6 in the paper simply returns P); | yes | `vaea.hpp` |
| `vaea` | vaea.4 | angle() = π/2 at zero norm, i.e. an individual sitting at the ideal point — the paper leaves this case undefined. | yes | `vaea.hpp` |
| `agemoea` | agemoea.1 | compute_norm_params builds the hyperplane from the ASF extremes, the NSGA-III practice. | yes | `agemoea.hpp` |
| `agemoea` | agemoea.2 | diversity = min1 + min2, the two smallest L_p distances from the candidate to the ALREADY-SELECTED set Σ (Alg.2's Ω). | yes | `agemoea.hpp` |
| `agemoea` | AGE-2 (FIXED 2026-09-22) | p when Eq.8 has no value. §3.3 says of the central point: "the coordinates of C cannot be all zero ... nor all equal to ones ..., the summation ΣC_i is always greater than zero and lower than M. | no: fixed | `agemoea.hpp` |
| `agemoea` | AGE-3 (FIXED 2026-09-22) | a single extreme point. When one solution is the argmax of every normalized objective — duplicates in F1, or axes whose range is zero — Σ starts with ONE member, the second minimum of the diversity does not exist, and... | no: fixed | `agemoea.hpp` |
| `agemoea` | agemoea.3 | guards beyond the paper: clamp f^n >= 0, p in [0.1, 20] for a valid ΣC, proximity >= 1e-14 — inactive on regular data. | yes | `agemoea.hpp` |
| `etea` | ETEA-1 (INVARIANT, not a deviation) | All Euclidean distances — EMST, ETCD and the distance count — are computed on RAW objective values, per Def.4 and Eq.1. | no: fixed | `etea.hpp` |
| `etea` | ETEA-2 (MINOR) | Mating — binary tournament on F (the paper leaves this free; see the note above for the exact §3.1 wording). | yes | `etea.hpp` |
| `etea` | ETEA-3 (MINOR) | Findout_neighbor (Alg.2 line 4): the neighbors of p are the dominated individuals within a radius = the distance from p to its nearest dominating nondom. neighbor (§3.3, text after Alg.2). | yes | `etea.hpp` |
| `etea` | ETEA-4 (MINOR) | Real-valued genome; binary is outside the paper's coverage. | yes | `etea.hpp` |
| `gde3` | GDE3-1 | The removal rule (step 3). Eq. 3's condition, read literally, compares crowding distances across fronts; the text states the intent, NSGA-II's ordering. | yes | `gde3.hpp` |
| `gde3` | GDE3-2 | Crowding distance is NSGA-II's (§III-B of Deb et al. 2002, Eq. 3's "CD"), each objective normalised by its range on the front, at every M. | yes | `gde3.hpp` |
| `gde3` | GDE3-3 | Bounds. "initial bounds" x^(lo), x^(hi) bound the initialisation only; the paper does not say what happens to a trial variable outside them. | yes | `gde3.hpp` |
| `gde3` | GDE3-4 | Every trial is evaluated in full. The paper notes that constraint-first evaluation can skip objective evaluations (§4); the library has no partial evaluation. | yes | `gde3.hpp` |
| `gde3` | GDE3-5 | constraint_mode NONE (the default) ignores constraints, as every core does; | yes | `gde3.hpp` |
| `gde3` | GDE3-6 | Continuous variables only; a problem with binary ones is refused. | yes | `gde3.hpp` |
| `nsga3` | nsga3.1 | Path-A: pop_size must exactly equal the size of the Das-Dennis lattice (paper: N = nearest multiple of 4 > H — the 92/212/... configurations of Table I are literally unattainable; a deliberate decision); | yes | `nsga3.hpp` |
| `nsga3` | nsga3.2 | set_reference_points: user-supplied (aspiration) points are used as is — the mapping onto the normalized hyperplane via Eq.4 (Alg.2 lines 8-9) is not performed; supply points on the unit simplex. | no: option | `nsga3.hpp` |
| `nsga3` | NSGA3-N1 (reading: the §IV-C text over Algorithm 2, twice) | The ideal point: the text takes z^min over ∪_{τ=0}^t S_τ, Alg.2 line 2 over S_t. | yes | `nsga3.hpp` |
| `nsga3` | NSGA3-N2 (the port's own rule, declared 2026-09-22) | All §IV-C says of the hard cases is "Special care is taken to handle degenerate cases and nonnegative intercepts" — that care is taken, not what it is — and Alg.2 line 6 only says "Compute intercepts". | yes | `nsga3.hpp` |
| `nsga3` | NSGA3-N3 (the port's value) | The ASF of each axis is formed "with a weight vector close to ith objective axis"; | yes | `nsga3.hpp` |
| `nsga3` | NSGA3-S (task 7 of 2026-10-10, supplements 2 and 3) | survival = "stochastic": the stochastic population update of Opris, "A First Runtime Analysis of NSGA-III on a Many-Objective Multimodal Problem: Provable Exponential Speedup via Stochastic Population Update", IJCAI 2... | no: option | `nsga3.hpp` |
| `nsga3` | NSGA3-L (task 15 of 2026-10-21, §2; off by default) | The population size μ, the number of children λ and of reference points H apart, after Tanabe & Oyama, EMO 2017 (source nsga3-popsize-children-refpoints_ tanabe2017): n_offspring = λ children a generation, bred as μ's... | no: option | `nsga3.hpp` |
| `a_nsga3` | ANSGA3-L (task 15 of 2026-10-21, §2; off by default) | n_offspring (λ children a generation, μ survivors from the μ + λ) and n_ref_points (H reference points, das_dennis::reference_points) apart from pop_size, as NSGA3-L in nsga3.hpp; not set, λ = μ and H = μ's lattice, a... | no: option | `a_nsga3.hpp` |
| `a_nsga3` | a_nsga3.1 | Path-A: pop_size = the Das-Dennis lattice size (see nsga3.hpp); | yes | `a_nsga3.hpp` |
| `theta_dea` | TDEA-L (task 15 of 2026-10-21, §2; off by default) | n_offspring (λ children a generation, μ survivors from the μ + λ) and n_ref_points (H reference points, das_dennis::reference_points) apart from pop_size, as NSGA3-L in nsga3.hpp; not set, λ = μ and H = μ's lattice, a... | no: option | `theta_dea.hpp` |
| `ar_moea` | IGD-NS (Eq.2-3) | IGD-NS(X,Y) = Sum_{y in Y} min_{x in X} dis(y,x) + Sum_{x' in X*} min_{y in Y} dis(y,x'), where X* is the set of noncontributing solutions: those x' that are nearest to no reference point y in Y (Eq.2). | yes | `ar_moea.hpp` |
| `ar_moea` | ar_moea.1 | argmin/argmax ties are broken deterministically (the first extremum); on an exact distance tie there is a single contributing carrier; | yes | `ar_moea.hpp` |
| `ar_moea` | ar_moea.2 | ENS/T-ENS (§III-D) is replaced by a fast NDS with an equivalent result; | yes | `ar_moea.hpp` |
| `ar_moea` | ar_moea.3 | numerical guards in AdjustLocation: \|\|r\|\| < 1e-12 leaves r in place, and a negative projection is clamped to 0; | yes | `ar_moea.hpp` |
| `ar_moea` | ar_moea.4 | archive duplicates are detected by exact equality of the objective vectors. | yes | `ar_moea.hpp` |
| `moead` | moead.1 | EP may contain duplicate F-vectors (does not contradict the letter of Step 2.5). | yes | `moead.hpp` |
| `moead` | 2026-09-05 | Step 2.1 says "Randomly select two indexes k, l from B(i)" — | yes | `moead.hpp` |
| `moead_de` | MDE-3 | the j_rand gene of de_rand_1_bin (absent from Eq.6) — removed. | yes | `moead_de.hpp` |
| `moead_de` | MDE-4 | r2, r3 were forced distinct from i and from each other (up to 10 rejections) — now two plain draws, as Step 2.2 says. | yes | `moead_de.hpp` |
| `moead_de` | MDE-6 | Step 2.3's RANDOM RESET is the paper's and stays the default, but it is expensive: with CR = 1 every out-of-box gene of the final y is replaced by a fresh uniform draw, and early in a run that is a large fraction of t... | yes | `moead_de.hpp` |
| `moead_de` | MDE-7 (task 15 of 2026-10-21, §1; off by default) | The three components of the DE mutation after Tanabe & Ishibuchi, Soft Computing 2019 (source moead-de-mutation-components_tanabe2019; operators/de_mutation.hpp): de_base (current1, the default: Eq.6 with r1 = i; rand... | no: option | `moead_de.hpp` |
| `moead_dra` | DRA-6 | the j_rand gene of de_rand_1_bin (absent from Eq.4) — removed. | yes | `moead_dra.hpp` |
| `moead_dra` | DRA-7 | r2, r3 were forced distinct from i and from each other (up to 10 rejections) — now two plain draws, as Step 3.2 says. | yes | `moead_dra.hpp` |
| `moead_dra` | DRA-5 (MINOR) | weights — Das–Dennis lattice (N must be a lattice size, Path-A) instead of the authors' maximin selection of N out of 5000 random vectors (Sec.III: W starts from the unit vectors, then the vector of W_1 farthest from... | yes | `moead_dra.hpp` |
| `moead_dra` | moead_dra.1 | Numerical guard old_g > 1e-14 for the division in Δ (the paper does not address division by 0). | yes | `moead_dra.hpp` |
| `moead_dra` | DRA-9 (MINOR) | the 10-tournament draws WITHOUT replacement from the not-yet-selected indexes (Step 2 says "select OTHER ⌊N/5⌋ − m indexes", which reads as "not already in I"); it shrinks to min(10, \|candidates\|) when fewer than ten... | yes | `moead_dra.hpp` |
| `moead_dra` | DRA-10 (READING) | Δ^i = (old − new)/old compares the g^te snapshot taken at the previous Step-5 update (with the z of THAT moment) against the current g^te (current z); the paper says only "old function value − new function value" and... | yes | `moead_dra.hpp` |
| `moead_dra` | DRA-11 (READING, the paper against itself; 2026-09-16) | Step 5's factor 0.95 + 0.05·Δ^i/0.001 is negative once Δ^i < −0.019, and that happens: Δ^i compares g^te across a moving z (DRA-10), so a subproblem that did not improve can read as worse. | yes | `moead_dra.hpp` |
| `moead_dra` | DRA-12 (task 15 of 2026-10-21, §1; off by default) | de_base, de_index, de_p and bound_repair = r_reflect or resample, as MDE-7 in moead_de.hpp (the pbest by subproblem i's Tchebycheff value, Eq.3); the defaults are Step 3.2's current1 and wr and Step 3.3's random reset. | no: option | `moead_dra.hpp` |
| `moead_dd` | moead_dd.1 | \|S\| = 1: one offspring of the SBX pair is used (Alg.1 line 6 allows several; one offspring yields exactly N FE per generation). | yes | `moead_dd.hpp` |
| `moead_dd` | moead_dd.2 | Eq.6 (READING): the angle is computed w.r.t. | yes | `moead_dd.hpp` |
| `moead_dd` | moead_dd.3 | z* is the running component-wise MINIMUM over everything evaluated. | yes | `moead_dd.hpp` |
| `moead_dd` | moead_dd.4 | pbi_value in an individual is an informational cache from the moment of association; in selection the PBI is RECOMPUTED from the current z* — a cached value goes stale as soon as z* moves. | yes | `moead_dd.hpp` |
| `moead_dd` | moead_dd.5 | the two mating parents are DISTINCT whenever the candidate set has more than one member: Alg.3 forms a SET P̄ of k solutions ("choose k solutions ... to form P̄"), so the second draw is repeated until it differs from... | yes | `moead_dd.hpp` |
| `moead_awa` | AWA-12 (MINOR) | If the EP left after Alg.3 Step 1 supplies fewer than nus candidates, the last-removed subproblems are restored so that \|evol_pop\| stays N; | yes | `moead_awa.hpp` |
| `moead_awa` | AWA-14 (READING, as DRA-11 in moead_dra.hpp; 2026-09-16) | Step 2.2 is MOEA/D-DRA's utility update — §3.4 adopts that strategy "as proposed in (Zhang et al., 2009)" and repeats its formula — so it inherits the same contradiction: a factor below 0 makes π negative, and a negat... | yes | `moead_awa.hpp` |
| `moead_m2m` | M2M-1 (LETTER, not a deviation) | Algorithm 2 line 4: when solutions are lacking, the subregion is refilled with RANDOM ones from Q, and the port does exactly that; this is also ablation point A4/A5 ("hard quotas + refill"). | no: fixed | `moead_m2m.hpp` |
| `moead_m2m` | M2M-2 (MINOR) | Direction vectors: "uniformly from the unit sphere" — for K attainable by the lattice, implemented as the Das–Dennis simplex lattice normalized to unit length (the canonical way; the reference code does the same). | yes | `moead_m2m.hpp` |
| `moead_m2m` | M2M-3 (MINOR) | For \|P_k\|>S the paper's non-dominated sorting [4] does not specify the tie-break within the boundary front. | yes | `moead_m2m.hpp` |
| `moead_m2m` | M2M-4 (fixed 2026-09-05) | The angles of Eq.2 are taken from the ORIGIN, as §II-A writes (f_i ≥ 0 is assumed). | no: fixed | `moead_m2m.hpp` |
| `moead_m2m` | M2M-5 (retired 2026-09-05) | y ∈ P_k is one plain uniform draw, as line 7 says ("Randomly choose y from P_k"); y = x is admitted (then the crossover Eq.(5) returns x and only the mutation acts). | no: fixed | `moead_m2m.hpp` |
| `moead_m2m` | M2M-6 (retired, FIX 2026-07-07) | Previously: "SBX yields two children — the first is taken". | no: fixed | `moead_m2m.hpp` |
| `moead_m2m` | M2M-D (DIAGNOSTIC 2026-08-06, RESOLVED 2026-09 — not a deviation) | This port converged poorly on DTLZ2 at short budgets and carried a known_issue marker for it. | no: fixed | `moead_m2m.hpp` |
| `moead_m2m` | M2M-7 (MINOR) | set_eta_crossover/set_eta_mutation/set_pc are no-op shims for API uniformity (operator [20] has no η/p_c; the crossover is unconditional). | yes | `moead_m2m.hpp` |
| `moead_m2m` | M2M-8 (CONSTRAINT, not a deviation — documented for callers) | The M2M decomposition partitions the population into K subpopulations of equal size S, so pop_size MUST be an exact multiple of K: pop_size = K * S, S >= 2 Any other pop_size throws std::invalid_argument from setup(). | no: fixed | `moead_m2m.hpp` |
| `moead_am2m` | AM2M-1 (DEVIATION) | α (probability of within-subpopulation mating) is not specified by the paper → 0.7. | yes | `moead_am2m.hpp` |
| `moead_am2m` | AM2M-2 (MINOR) | Weights = L1 normalization of the Max-Min-selected individuals (Eq.4 Σw=1). | yes | `moead_am2m.hpp` |
| `moead_am2m` | AM2M-3 (retired, FIX 2026-07-08) | Previously: «η_c=20, operators as in M2M [8]» — falsely attributed SBX/PM to source M2M [8]; | no: fixed | `moead_am2m.hpp` |
| `moead_am2m` | AM2M-5 (MINOR) | S_k>\|P̃_k\| → Max-Min fill-up by repetition; | yes | `moead_am2m.hpp` |
| `moead_am2m` | AM2M-6 (MINOR) | The Liu–Li operator yields ONE offspring z=x+rc·(x−y) from the base x; y≠x for within-subregion mating. | yes | `moead_am2m.hpp` |
| `moead_am2m` | AM2M-7 (MINOR) | Uniform initial weights: das_dennis::generate_auto(m,N) may return slightly more than N vectors (fallback lattice) — truncated to N; weight holders are chosen as the Tchebycheff argmin without removal (duplicates are... | yes | `moead_am2m.hpp` |
| `moead_am2m` | AM2M-8 (MINOR) | §III-B specifies only an INCREMENT loop for the slot quota (steps 1-4: S_k = ⌊\|P̃_k\|/2⌋, except S_k = 1 when \|P̃_k\| = 1; then raise the largest S_k until Σ S_k = \|P\|). | yes | `moead_am2m.hpp` |
| `sms_m2m` | SMSM2M-1 (DEVIATION, letter of the paper) | Algorithm 2: refill with RANDOM solutions from Q (as in the M2M primary source), the letter of the paper. | yes | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-2 (DEVIATION, letter of the paper) | §2.7 sets p_m=1/popsiz (and NOT the usual 1/n_vars). | yes | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-3 (MINOR) | SMS-EMOA has a "modified" variant with the number of dominators d(s,P) (§2.3); here the CANONICAL HV variant is used (last front + min exclusive HV). | yes | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-4 (MINOR) | HV reference point r_j=1.1·max_j(P_k) (SMS-EMOA reference); if max_j≤0 → max_j+1. | yes | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-5 (MINOR) | Directions: the letter of the paper §2.7 — «uniformly selected from the unit sphere in the first octant», i.e. an ARBITRARY K is allowed. | yes | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-9 (CONSTRAINT, not a deviation — documented for callers) | Same structural requirement as moead_m2m (see M2M-8 there): the M2M decomposition needs K equally sized subpopulations, so pop_size = K * S, S >= 2 and any other pop_size throws std::invalid_argument from setup(). | no: fixed | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-6 (fixed 2026-09-05, as M2M-4 in moead_m2m) | The angles of Eq.2 are taken from the ORIGIN (§2.2 assumes f ≥ 0); an objective whose pool minimum is negative is shifted by the smallest M restoring positivity (§2.2: f_i + M). | no: fixed | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-7 (retired 2026-09-05) | y in the P_k branch is ONE plain draw, as Algorithm 1 says ("Randomly choose y from P_k"); y = x is admitted. | no: fixed | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-8 (MINOR) | SBX yields 2 children — the first is taken ("a new solution z", singular). | yes | `sms_m2m.hpp` |
| `sms_m2m` | SMSM2M-10 (implementation, fixed 2026-09-27) | The contributions come from hv_contribution.hpp, the routine SMS-EMOA and MO-CMA-ES use: at M = 2 the neighbours' formula, from M = 3 each point's exclusive hypervolume against the others limited by it, by WFG. | no: fixed | `sms_m2m.hpp` |
| `moead_ds` | DDS-1 (MINOR) | createOffspring — SBX yields 2 children, the first is taken (1 offspring per i → \|O\|=N → P∪O=2N → DDS selects N). | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-2 (MINOR) | z_nad — hyperplane intercepts via the extreme points (NSGA-III); on degeneration, fallback to the coordinate-wise max. | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-3 (MINOR) | nondominated within S_A — front 0; if S_A/the front is empty, s=pivot. | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-4 (MINOR) | θ=G/Gmax = gen/t_max. The paper itself is of two minds about what G counts: the prose under Eq.16 says "G is the current evaluations and G_max is the maximum evaluations", while Alg.1 line 13 increments G once per out... | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-5 (MINOR) | Real-valued genome; binary is outside the coverage (NONE). | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-6 (ARBITRATION) | FIX 2026-07-07 (source-fidelity review): the literal Eq.9 inverts the ASF weights («when j=i, w=0» → 1e-6 on the own axis, 1 on the others): for m≥3 the "extremes" degenerate into near-ideal points (≈argmin f_i) → the... | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-7 (fixed 2026-09-05) | Mating d2 (Alg.4 line 2) uses the reference points R returned by the LAST DDS call (Alg.1 line 3 for the first generation, line 12 afterwards) — i.e. the normalization frame of the merged 2N pool, as Alg.1 line 8 cons... | no: fixed | `moead_ds.hpp` |
| `rd_emo` | RDE-1 (NOT A DEVIATION — kept as a note) | θ = 5 IS stated: §4.1 says "θ = 5 suggested in MOEA/D is set in RD-EMO". | no: fixed | `rd_emo.hpp` |
| `rd_emo` | RDE-2 (MINOR) | A region is argmin d2 to a reference line, i.e. the "nearest reference line". | yes | `rd_emo.hpp` |
| `rd_emo` | RDE-3 (DEVIATION — overriding a stated value) | z* is the RUNNING ideal: the per-objective minimum over the population and every offspring seen. | yes | `rd_emo.hpp` |
| `rd_emo` | RDE-3b (MINOR) | Alg.1 line 18 draws the second parent's region from the T-neighbourhood U_n restricted to regions with RD > 0; when every region of U_n is empty (only possible at very small N) the paper has no rule, and this port fal... | yes | `rd_emo.hpp` |
| `rd_emo` | RDE-4 (MINOR) | The distance between regions is the Euclidean distance between reference points; | yes | `rd_emo.hpp` |
| `rd_emo` | RDE-5 (MINOR) | Reference points = Das–Dennis(pop_size) via generate_auto, with automatic layering. | yes | `rd_emo.hpp` |
| `rd_emo` | RDE-6 (FIXED 2026-09-05, second primary-source pass) | Alg.1 line 20 applies SBX ONCE per pair and takes the complementary pair (x_new, x''_new); | no: fixed | `rd_emo.hpp` |
| `rd_emo` | RDE-7 (CONSTRAINT, documented for callers) | §3.3 requires an EVEN popsiz — "population size is set to the number of reference points; otherwise ... plus one" — because SBX produces offspring in pairs. pop_size comes from the caller and is neither rounded nor re... | yes | `rd_emo.hpp` |
| `rd_emo` | RDE-8 (READING) | U_i and V_i — the T / W "closest regions" to Λ_i — EXCLUDE Λ_i itself. | yes | `rd_emo.hpp` |
| `aprd` | APRD-1 (MINOR) | The hyperplane is a least-squares regression of f_m on f_1..f_{m-1}, shifted to the origin (the intercept is dropped); on a degenerate system V = (1,...,1). | yes | `aprd.hpp` |
| `aprd` | APRD-2 (IMPLEMENTED) | L segment: point projections are clamped into [Lmin,Lmax], the coordinate-wise bounds of the Arc projections onto L (§II-C / Fig.2, "replace projections outside L with nearest endpoint"). | yes | `aprd.hpp` |
| `aprd` | APRD-3 (MINOR) | PBI sign and projection are written in canonical form (d1 signed, proj = Z + d1 * V_hat); the paper writes "Z − d1 V", which is dimensionally inconsistent, so the standard correct form is used. | yes | `aprd.hpp` |
| `aprd` | APRD-4 (MINOR) | Min-Max (Euclidean) for centres and ideal points: the first element is random, as Min-Max in §II inherits a random start, then farthest-to-set; slots S_i without refill (skipped when S_i = \|P_i\|), §II-D2. | yes | `aprd.hpp` |
| `aprd` | APRD-5 (MINOR) | Case I NSGA-II: (rank, CD) tournament plus SBX/PM, then NDS+CD selection. | yes | `aprd.hpp` |
| `aprd` | APRD-6 (MINOR / partly FIXED 2026-09-06, full-paper checklist) | SBX yields the first child — one offspring per x, Alg.1's "a new solution z". | no: fixed | `aprd.hpp` |
| `aprd` | APRD-7 (resolved) | update_archive receives Arc u Q in BOTH cases (Alg.1: "Arc <- UpdateArchive(Arc u Q, N_a)"), not pop u Q as it once did. | no: fixed | `aprd.hpp` |
| `aprd` | APRD-8 (resolved) | the partition P_1..P_K is built by Algorithm 3 (update_subpops) and REUSED by the next generation's mating (Alg.1: for k, foreach x in P_k), rather than recomputed by a random-start FPS at every step. | no: fixed | `aprd.hpp` |
| `aprd` | APRD-9 (FIXED 2026-09-06, full-paper checklist) | Algorithm 2 truncates the archive by the Max-Min principle of §II-B, whose measure is the ANGLE: "select the first point in the remaining points and make it have the maximum angle to first selected point; … the next p... | no: fixed | `aprd.hpp` |
| `aprd` | APRD-10 (ARBITRATION — the printed PBI equations are misprinted) | Eq.(7)-(8) as printed are dimensionally inconsistent (d1 already divided by ‖V‖, then multiplied by the unnormalized V) and subtract the projection where the orthogonal residual is meant, so they do not define a dista... | yes | `aprd.hpp` |
| `liu_gu2011` | LG-1 (MINOR) | R=1 = "non-dominated within its internal set" (the paper: «non-dominated solutions within internal set», recomputed every T); offspring are marked R=1 on entry. | yes | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-2 (MINOR) | The NSGA-II selection of internal — canonical NDS + crowding distance (the paper cites NSGA-II [6]); the boundary-front tie-break is by CD. | yes | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-3 | (DEVIATION — the paper contradicts itself, and this port breaks the tie in the direction that overshoots). | yes | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-3b (READING) | §IV-A prints ceiling brackets for all three of S, l_i and Max_gen and then defines the bracket, in the same sentence, as "the largest integer of not greater than x" — i.e. floor. | yes | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-4 (READING; FIXED 2026-09-06, full-paper checklist) | The central vectors W — «uniformly distributed unit vectors on the first quadrant of the unit hyper-sphere», no construction given. | no: fixed | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-5 (MINOR) | Shift by the minimum of the internal population; association by the acute angle (cos) of the shifted f. | yes | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-6 (FIXED 2026-09-06, full-paper checklist) | §II-B: «Randomly select 5·l_i individuals from the initial population to constitute the external set» — a random SUBSET of the 5N pool (sampling without replacement, sample_idx). | no: fixed | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-6b (FIXED 2026-09-06, full-paper checklist) | Step 3.3: «the remained individuals replace the same number of external individuals randomly» — \|rej\| DISTINCT external slots are drawn (partial Fisher–Yates) and overwritten, so exactly "the same number" of members i... | no: fixed | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-9 (FIXED 2026-09-06, full-paper checklist) | §II-C: «If M_gen < N, we randomly select N − M_gen individuals from the internal set to participate in mating» — distinct internal members that are NOT already marked R=1 (sample_idx over the rest). | no: fixed | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-7 (AMBIGUOUS→resolved) | For \|R1\|>N (possible when Σl_i>N, LG-3) the paper only says «all of these individuals take part in mating» and «generate N new individuals», without specifying the order. | no: fixed | `liu_gu2011.hpp` |
| `liu_gu2011` | LG-8 (MINOR) | set_eta_crossover/set_eta_mutation/set_pc are no-op shims for API uniformity (operator [1] has no η/p_c; the crossover is unconditional). | yes | `liu_gu2011.hpp` |
| `isde_rd` | ISDE-1 (MINOR) | The SDE shift is q'_j = max(p_j, q_j); d is Euclidean and translation-invariant. | yes | `isde_rd.hpp` |
| `isde_rd` | ISDE-2 (ARBITRATED) | Direction vectors V. The paper does NOT specify how to generate them: §II-B delegates to the "decomposition strategy of MOEA/D-M2M [4]", and Alg.3 takes "V: K unit direction vectors" as an INPUT. | yes | `isde_rd.hpp` |
| `isde_rd` | ISDE-3 (DEVIATION, full-paper checklist) | The angular association is computed in the shifted space f − z*, z* the per-objective minimum over the pool being allocated (the cone apex sits at the pool's ideal point). | yes | `isde_rd.hpp` |
| `isde_rd` | ISDE-4 (MINOR) | Above S, removal proceeds one at a time with I_SDE^+ recomputed; | yes | `isde_rd.hpp` |
| `isde_rd` | ISDE-5 (MINOR) | One offspring per member (SBX, first child); real-valued genome, binary out of scope. | yes | `isde_rd.hpp` |
| `isde_rd` | ISDE-6 (MINOR, consequence of the sizing formula) | The active population after each step() has exactly K_*S_ members. | yes | `isde_rd.hpp` |
| `isde_rd` | ISDE-7 (FIXED 2026-09-05, second primary-source pass) | Alg.3 line 8 says only "randomly choose q from P_k" — no q != p condition, no size test. | no: fixed | `isde_rd.hpp` |
| `isde_rd` | ISDE-8 (FIXED 2026-09-05) | Alg.3 line 13, P = P ∪ o, is written INSIDE the "for each p in P_k" loop, so the line-10 partner ("randomly choose q from P") is drawn from P as it grows during the generation and can be an offspring bred earlier in t... | no: fixed | `isde_rd.hpp` |
| `isde_rd` | ISDE-9 (FIXED 2026-09-06, full-paper checklist) | Alg.3 line 2 only ASSIGNS the initial population to the subregions; the quota selection (line 17 = Alg.1) sits inside the generation loop. | no: fixed | `isde_rd.hpp` |
| `isde_rd` | ISDE-10 (FIXED 2026-09-06, full-paper checklist) | Alg.1 line 6 / §II-C Case 2: "randomly select S_k − \|P_k\| individuals from P" — DISTINCT members of P that are not already in P_k (partial Fisher–Yates), the reading used for the same rule in moead_m2m / sms_m2m. | no: fixed | `isde_rd.hpp` |
| `dhea` | DHEA-1 (FIXED 2026-09-05, second primary-source pass) | §Mating selection: binary tournament (dominance→Knee→Div, "select one solution randomly if all the three principles fail"), "Repeat until N solutions are selected" — the mating pool P' holds N winners. | no: fixed | `dhea.hpp` |
| `dhea` | DHEA-4 (ARBITRATED) | Alg.1 line 6-7 passes FrontNo to MatingSelection, but the prose defines the first tournament level as "If one solution is dominated by the other" — a pairwise test, not a rank comparison. | yes | `dhea.hpp` |
| `dhea` | DHEA-2 (MINOR) | w (the local nadir) with protection against zero components (1e-12); cos is clamped to [−1,1]. | yes | `dhea.hpp` |
| `dhea` | DHEA-5 (FIXED 2026-09-06, full-paper checklist) | Alg.2 line 14 computes Div "of each solution in P" AFTER line 3 replaced P by its normalized objectives, i.e. on the F' of the retained pool (ideal/nadir over F_1 ∪ … ∪ F_l). | no: fixed | `dhea.hpp` |
| `dhea` | DHEA-3 (MINOR) | real-valued genome; binary is beyond coverage. | yes | `dhea.hpp` |
| `hlmea` | HLMEA-1 (RESOLVED) | The header once claimed that "β and K are not stated unambiguously in the tables". | no: fixed | `hlmea.hpp` |
| `hlmea` | HLMEA-2 (MINOR; refill FIXED 2026-09-06, full-paper checklist) | The allocation of [17] (Alg.1): NDS truncation plus random refill; the boundary-front tie is broken by crowding distance. | no: fixed | `hlmea.hpp` |
| `hlmea` | HLMEA-3 (MINOR) | HV of the cluster centres via HSO slicing. | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-4 (MINOR) | K-means: Lloyd in decision space with an early exit on stabilization; the iteration limit is 100 per §4.3 (it had been 30 as an "implementation choice", but the paper states 100 explicitly). | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-5 (FIXED 2026-09-06, full-paper checklist) | The Φ_w association is Eq.(2) of [17] — the acute angle between the objective vector and v^w taken from the ORIGIN ("u ∈ R^m_+", objectives assumed non-negative). | no: fixed | `hlmea.hpp` |
| `hlmea` | HLMEA-6 (MINOR) | Real-valued genome; binary is out of scope. | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-9 (MINOR, consequence of the sizing formula) | The Alg.1 allocation fills every subregion to exactly S = floor(N/W), so the active population after each step() has W*S members — SHORT of pop_size whenever W does not divide N. | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-8 (MINOR) | Alg.3 says "randomly select" one (line 2), three (line 6) or two (line 9) solutions from the mating pool and never requires them to be distinct from each other or from x. | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-7 (MINOR) | W follows the paper's per-m branch by default: 15 at m=3, 10 otherwise. set_W pins an explicit value and disables the branch. | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-10 (MINOR, pseudocode defect) | Alg.2 line 3 initialises Q = ∅ ONCE, outside the generation loop, and lines 12/26 write Q = Q ∪ P inside the per-subregion loop. | yes | `hlmea.hpp` |
| `hlmea` | HLMEA-11 (FIXED 2026-09-05, second primary-source pass) | Alg.4 takes the cluster centres "of the last and current generation". | no: fixed | `hlmea.hpp` |
| `hlmea` | HLMEA-12 (MINOR) | K-means runs in DECISION space (§3.1), so a cluster centre has no objective vector of its own; the "centre" fed to the HV of Alg.4 is the mean of the members' objective vectors. | yes | `hlmea.hpp` |
| `ibea_eplus` | IEP-1 (MINOR) | Alg.1 Step 4 returns the non-dominated subset A of P; this core hands back the whole final P and leaves that filter to the caller, as every core in the library does. | yes | `ibea_eplus.hpp` |
| `ibea_eplus` | IEP-2 (DEVIATION) | Binary genomes use the library's uniform crossover + bit-flip 1/n_bits; the paper's knapsack runs used one-point crossover 0.8 + bit-flip 0.04 (footnote 3). | yes | `ibea_eplus.hpp` |
| `ibea_eplus` | IEP-3 (EXTENSION) | The constraint modes above are not in the paper and are off by default. | no: option | `ibea_eplus.hpp` |
| `ibea_eplus` | IEP-4 (no change to the default) | Alg.2 steps 2.1-2.2 scale the objectives to [0,1] over the pool before I_eps+, and that is the letter, so it is on. set_normalize(false) drops it, to see what the scaling buys; in R2-IBEA the analogous per-axis scalin... | yes | `ibea_eplus.hpp` |
| `ibea_hd` | ibea_hd.1 | weak dominance in the I_HD branching — functionally neutral (equal vectors yield 0 in both branches) (internal audit, IHD-1/IHD-3). | yes | `ibea_hd.hpp` |
| `mibea` | mibea.1 | the generational scheme is generational (mu offspring per step), although the letter of Alg.1 Steps 6–7 describes steady-state (mu+1, one offspring, m += 1). | yes | `mibea.hpp` |
| `r2ibea` | R2I-NORM (AMBIGUOUS) | Eq.4 computes the Tchebycheff value on RAW objectives and the paper prescribes no scaling, so the run depends on the units: the fitness is -exp(-I_R2/kappa) with kappa = 0.005, i.e. exp(-200*I_R2), which underflows to... | yes | `r2ibea.hpp` |
| `hype` | hype.1 | an adaptive point r_i = max_i f_i · (1 ± ref_offset) over the current pool, ref_offset = 1.0 (set_ref_offset). | yes | `hype.hpp` |
| `hype` | hype.2 | only \|R\| = 1 is supported, while the paper allows a reference SET (§3.3 works a two-point example). | yes | `hype.hpp` |
| `mombi2` | mombi2.1 | the binary tournament criterion (the paper gives none; | yes | `mombi2.hpp` |
| `nimmo` | nimmo.1 | floor(0.1mu) versus the paper's "[0.1mu]", which may mean rounding — identical for mu in {200, 210, 230}, but off by one for the paper's mu=156 (15 vs 16) and mu=135 (13 vs 14) settings; initialization is uniform rand... | yes | `nimmo.hpp` |
| `irea` | irea.1 | Eq. 7 and §3.6/Alg. 4 are internally contradictory: the literal Eq. 7 (F = Sum −e^{−I(x2,x1)/kappa}), combined with selecting the MINIMUM F, would pick the worst solutions. | yes | `irea.hpp` |
| `irea` | irea.2 | Eq. 6 as printed is a distance to a point, not a perpendicular. | yes | `irea.hpp` |
| `irea` | irea.3 | §3.4: "if extreme point set contains duplicate extreme points, or intercepts are nonnegative [an evident typo for non-positive], then intercept can be replaced with each maximum objective functions" — a degenerate mat... | yes | `irea.hpp` |
| `irea` | irea.4 | das_dennis::generate_exact requires N to equal the lattice point count, which is stricter than Table II of the paper, where N may differ slightly. | yes | `irea.hpp` |
| `edv` | EDV-1 (FIXED 2026-09-06, third primary-source pass) | p_m = 1/pop_size, the paper's letter: §4.2.4 is explicit and self-consistent — "the mutation probability (p_m) is set to 1/pops. | no: fixed | `edv.hpp` |
| `edv` | EDV-2 (FIXED 2026-09-05, second primary-source pass) | Alg.3 lines 2-11 build Y one offspring at a time: each iteration draws its own parents Pa by binary tournament, "y = SBX(Pa)" yields ONE offspring y, and PM is applied to that y. | no: fixed | `edv.hpp` |
| `edv` | EDV-3 (READING) | Alg.4 line 8 takes the angle between F(x) and d^i; | yes | `edv.hpp` |
| `edv` | EDV-4 (DECLARED) | Table 3 lists N = 276 at m = 10 for the (3,2) two-layer lattice, whose size is C(12,9) + C(11,9) = 275; this port requires \|D\| = N exactly (generate_exact), so 276 throws and 275 is the size to use. | yes | `edv.hpp` |
| `if_maoea` | IF-MaOEA-1 (RESOLVED) | The header used to claim a scope cut — "W_convex is NOT computed" — while the code did compute it, with TWO bugs: (a) a strict upper bound on the partial sums of Eq.2 discarded every set with ϑ1=π/2, losing the entire... | no: fixed | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-2 (AMBIGUOUS — two readings, one chosen) | For each objective k the "neighbour" is the nearest solution with a LARGER f_k, and that whole solution vector serves as the reference point; the volume is the product of the positive gaps, and the MINIMUM volume over... | yes | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-3 (FIXED 2026-09-06, third primary-source pass) | §2.4 invokes the "reference point adjustment strategy" of [44] = AR-MOEA and says what it does: link each reference point with the nearest solution in its assigned subpopulation, then move the reference point to where... | no: fixed | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-11 (FIXED 2026-09-06) | Step 5 writes the inner layer as w_ij = ½·w_ij + (M−1)/(2M)·π/2 for w_ij ∈ W_2 — a contraction of the ANGULAR coordinates toward their mean (M−1)π/(2M), applied before the cosine projection of Steps 3-4. | no: fixed | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-4 (MINOR) | One offspring per pair (SBX, first child), as in ISDE+RD and others in this library; \|O\| = N. | yes | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-5 (MINOR) | Real-valued genome; binary is out of scope. | yes | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-6 (MINOR) | The archive AS is truncated by the same sieve to ≤N; | yes | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-7 | (AMBIGUOUS — the paper contradicts itself on the normalization range). | yes | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-8 (PAPER MISPRINT, resolved from the paper's own text) | Alg.2 line 8 reads "Calculate the MINIMUM cosine of the angle between ASS and RefPoint". | yes | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-10 (FIXED 2026-09-05, FE-trajectory audit) | The reference set is ONE angular lattice of at most N points. | no: fixed | `if_maoea.hpp` |
| `if_maoea` | IF-MaOEA-9 (PAPER PSEUDOCODE BROKEN, resolved from §2.4) | Alg.3 as printed cannot run: (a) lines 5-10 accumulate the critical-layer picks into p and then "return Q" at line 12 without ever merging p into Q, and line 8's "p <- argmax fitness(F_k)" overwrites p instead of appe... | yes | `if_maoea.hpp` |
| `maoea_iamd` | maoea_iamd.1 | the replacement in Alg. 7 runs only while \|F_k\|−1 >= N−\|P\|; otherwise the loop can exhaust F_k before \|P\| = N, a case the paper leaves undefined; | yes | `maoea_iamd.hpp` |
| `maoea_iamd` | maoea_iamd.2 | Φ (Eq. 6) is stored as log Φ — a monotone guard against underflow; | yes | `maoea_iamd.hpp` |
| `maoea_iamd` | maoea_iamd.3 | degenerate cases the paper does not cover: a singular knee hyperplane falls back to d = −Σ of the normalized objectives; a zero derivative or an excursion to q <= 0 in Newton–Raphson gives q = 1; zero perturbation cov... | yes | `maoea_iamd.hpp` |
| `maoea_iamd` | maoea_iamd.4 | eta_m_ / set_eta_mutation are kept for interface compatibility but are unused, since the paper replaces PM with KPCM. | yes | `maoea_iamd.hpp` |
| `sms_emoa` | SMS-1 | Several fronts. EJOR has two Reduce procedures: the basic one (Algorithm 2, Eq. 3, as in 2005) removes the smallest ΔS of the worst front F_v whatever the number of fronts; | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-2 | Contributions (hv_contribution.hpp; hv::ContribChoice decides): exact at every M by default (the owner's word of 2026-10-21, after task 14's measurement: the approximations chose badly from M = 5; from task 13 of 2026... | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-3 | A tie in the smallest contribution removes the point that comes first in the population, the offspring last. | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-4 | Two objectives, two points. When F_v holds only two points at M = 2, both are extremes and "always keep" cannot hold; the papers do not say which goes, and here their contributions against r = nad(Q) + 1 decide — the... | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-5 | constraint_mode NONE (the default) ignores constraints; | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-6 | A binary or mixed genome uses the library's uniform crossover and bit-flip mutation, as NSGA-II does; the papers are real-valued. | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-7 | The reference point's offset "+ 1" is an absolute unit: from M = 3 the selection depends on the objectives' scale (tests/test_scale_ invariance.cpp). | yes | `sms_emoa.hpp` |
| `sms_emoa` | SMS-8 | survival = "stochastic" (task 7 of 2026-10-10, supplement 2): Bian, Zhou, Li & Qian, "Stochastic Population Update Can Provably Be Helpful in Multi-Objective Evolutionary Algorithms", IJCAI 2023, doi:10.24963/ijcai.20... | no: option | `sms_emoa.hpp` |
| `sms_emoa` | SMS-9 | inner_archive, inner_archive_cap, archive_reuse_p (operators/inner_archive.hpp): an archive of every non-dominated point evaluated, the initial population included; a parent, with probability archive_reuse_p, is a mem... | no: option | `sms_emoa.hpp` |
| `r2_emoa` | R2E-1 | The utopian point. The paper fixes r* per problem ((0, 0); | yes | `r2_emoa.hpp` |
| `r2_emoa` | R2E-2 | More than two objectives: the paper's experiments have two; the Das-Dennis lattice is our choice (task 9). | yes | `r2_emoa.hpp` |
| `r2_emoa` | R2E-3 | LION 2013's Algorithm 1 takes R2(P \ {x}) over the whole population, the journal's over R_h alone; the journal's is implemented (the corrected later version). | yes | `r2_emoa.hpp` |
| `r2_emoa` | R2E-4 | A tie of contributions removes the point first in the population, the offspring last (as SMS-3). | yes | `r2_emoa.hpp` |
| `r2_emoa` | R2E-5 | Everything else — the parents, the variation, constraints (CDP through the non-dominated sorting), the binary genome, survival = stochastic, the inner archive — is SMS-EMOA's (sms_emoa.hpp, SMS-5, -6, -8, -9). | yes | `r2_emoa.hpp` |
| `mo_cma_es` | MOCMA-1 (task, as DMS-1) | The search runs in coordinates normalised to the box, u = (x − ℓ)/(u − ℓ): the paper's σ_0 = 0.6·(x^u − x^l) is stated for equal ranges, and normalising makes it literal for unequal ones (equivalently, C_0 = diag of t... | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-2 | The individual keeps its unclipped x; the library's population holds feasible(x) and its true objectives f(feasible(x)), so the answer is always inside the box. | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-3 | Initial state: p̄_succ = p_target, p_c = 0, C = I (the 2007 paper, §2.1 "Initialization"); x and σ "must be chosen problem dependent" there — here x is uniform in the box and σ = σ_0. | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-4 | The reference point is r = nad + (nad − ideal) over Q in every objective (nad + 1 where the range is zero), which every member dominates, and the boundary elements of the front being ranked — its best point in each ob... | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-5 | The contributions are EXACT AT EVERY M (task 14 of 2026-10-20, supplement 3, Д8.1; hv::ContribChoice with exact_only), as the paper (Igel, Hansen & Roth 2007) computes them: M = 3 and 4 by HVC3D / HVC4D unless hv_cont... | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-6 | C is factorised by Cholesky for every offspring (O(n³)); the paper's implementation updates the factor incrementally (Suttorp, Hansen & Igel 2009), which samples the same distribution faster. | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-7 | constraint_mode NONE (the default) ignores general constraints; | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-8 | Continuous variables only; a problem with binary ones is refused. | yes | `mo_cma_es.hpp` |
| `mo_cma_es` | MOCMA-9 | p_target = 2/11 ≈ 0.1818, the 2007 paper's value. | yes | `mo_cma_es.hpp` |
| `rvea` | RVEA-1 (resolved) | the reference-vector adaptation was skipped at t=0 | no: fixed | `rvea.hpp` |
| `rvea` | RVEA-2 (resolved 2026-09-16) | the adaptation updated only as many vectors | no: fixed | `rvea.hpp` |
| `rvea` | RVEA-3 (resolved, task 13 of 2026-10-19) | the ⌊N/2⌋ random pairs were | no: fixed | `rvea.hpp` |
| `maoeaarv` | maoeaarv.1 | η_c=η_m=20 and the linkage of the hierarchical clustering (single-linkage here) are NOT specified by the paper — implementation conventions were chosen. | yes | `maoeaarv.hpp` |
| `mbra` | mbra.1 | M2: the Condition1 bounds are INCLUSIVE (G=⌊0.2·Gmax⌋ and G=⌊0.9·Gmax⌋ pass); the paper does not specify the strictness of the bounds; | yes | `mbra.hpp` |
| `mbra` | mbra.2 | M3: guard CM_old≈0 → IMR=0 (the paper does not address division by 0); | yes | `mbra.hpp` |
| `mbra` | mbra.3 | M4: max-min-angle with an empty selected set: the first individual is the first by index (all dmin=0); the paper does not define this case; | yes | `mbra.hpp` |
| `mbra` | mbra.4 | M5: degenerate range z_max=z_min → f'=0; a zero vector in angle_dist → d=1 — numerical guards beyond the paper; | yes | `mbra.hpp` |
| `mbra` | mbra.5 | M6: the protective exits of AdjustVector (rmax=0 / Nref=0) are normally unreachable when n=N; | yes | `mbra.hpp` |
| `mbra` | mbra.6 | M7: CM_old is refreshed at EVERY Condition1 trigger, not only after an AdjustVector. | yes | `mbra.hpp` |
| `nrv_moea` | NRV-1 (FIXED 2026-09-06, third primary-source pass) | the port used to | no: fixed | `nrv_moea.hpp` |
| `nrv_moea` | NRV-2 (FIXED 2026-09-16) | Arc ∪ P kept a solution twice when it sat in | no: fixed | `nrv_moea.hpp` |
| `nrv_moea` | nrv_moea.1 | with set_population_pruning(true) only — topping the population up when <N survive the pruning (by the Σ of normalized objectives), a guard of the fixed-N framework; under the letter P_c has exactly N members (N_n who... | yes | `nrv_moea.hpp` |
| `dea_gng` | GNG-1 | — RESOLVED (audit 2026-09). The A_S update follows §III-C literally: the first front only (i = 1), Alg.2 normalization and association against the current R, then per reference vector the closest solution while c_J =... | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-2 (MINOR) | sub-networks = the connected components of the edges (BFS). | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-3 (MINOR) | NSGA-III selection: normalization by z*/z' (max over F_1); association by angle; niches c_j; | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-4 (MINOR) | θ=∞ is emulated by a large value (1e6); R'_u → d2. | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-5 (MINOR) | real-valued genome; binary is out of coverage (NONE). | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-7 (MINOR, declared 2026-09-05) | Alg.1 lines 7-8 give only P' = Mating_Selection(P), P'' = Reproduction(P') and §III-B only the tournament criteria; neither the size of P' nor how SBX children are kept is stated. | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-8 (FIXED 2026-09-06, third primary-source pass) | Alg.1 line 17 increments g AFTER environmental selection and line 10 tests g < (1−α)·G_max with g = 0 in the first generation; the port incremented g before the test, so the learning model stopped one generation early... | no: fixed | `dea_gng.hpp` |
| `dea_gng` | GNG-6 (AMBIGUOUS — the paper mixes two frames; the raw one is chosen) | ζ (Eq.8) is measured on the RAW, un-expanded GNG node geometry: both the reference vector and the edge come from node_obj, matching V_edge as emitted by Alg.1 line 12, and the resulting θ is then attached to the Eq.5-... | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-9 (AMBIGUOUS) | Alg.2 line 12 writes J = argmin_{j∈C} c_j and says nothing about ties. | yes | `dea_gng.hpp` |
| `dea_gng` | GNG-10 (LETTER + GUARDS, noted 2026-09-16) | Eq.5 (Alg.3 line 12) maps each sub-network's node box [f^q_min,node, f^q_max,node] onto the box of its signals [f^q_min,A_S, f^q_max,A_S]. | yes | `dea_gng.hpp` |
| `srv` | srv.1 | z^nad = the max over St (Alg.4 line 5 only says "normalize by (2)"); mating uses random pairs, as the paper specifies no parent-selection operator. | yes | `srv.hpp` |
| `srv_nsga3` | SRVN-L (task 15 of 2026-10-21, §2; off by default) | n_offspring (λ children a generation, μ survivors from the μ + λ) and n_ref_points (H reference points, das_dennis::reference_points) apart from pop_size, as NSGA3-L in nsga3.hpp; not set, λ = μ and H = μ's lattice, a... | no: option | `srv_nsga3.hpp` |
| `srv_nsga3` | srv_nsga3.1 | the SRVs are extracted under the max-normalization of srv_strategy while the association runs under the NSGA-III intercept normalization (SRVN-2, a mixing of spaces); set_kmeans_iter is a no-op, since the iteration co... | yes | `srv_nsga3.hpp` |
| `srv_nsga3` | SRVN-5 | the ideal z^min and the ASF extremes of the NSGA-III intercept | yes | `srv_nsga3.hpp` |
| `cava_moea` | CAVA-2 | The Alg.7 replacement admits any y_r in P. | yes | `cava_moea.hpp` |
| `maoea_ac` | maoea_ac.1 | Eq.13: the pair is taken from DIFFERENT clusters. | yes | `maoea_ac.hpp` |
| `maoea_ac` | maoea_ac.2 | Eq.8: "\|\|std/avg\|\|" is read as the coefficient of variation std/\|avg\| with a guard \|avg\| > 1e-12 (when avg is near zero, Eq.7 already yields p = 1.0); | yes | `maoea_ac.hpp` |
| `maoea_ac` | maoea_ac.3 | the paper does not say which population z* and z^nad are estimated from; | yes | `maoea_ac.hpp` |
| `maoea_ac` | maoea_ac.4 | the paper specifies no mating selection, so random pairs are used as the neutral reading; | yes | `maoea_ac.hpp` |
| `maoea_ac` | maoea_ac.5 | the safety top-up to exactly N by CF is beyond the paper and is normally dead code. | yes | `maoea_ac.hpp` |
| `maoea_3c` | 3C-1 (MINOR) | Generation-driven schedule. The library drives the algorithm by step()/t_max (generations), not raw FE. | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-2 (MINOR) | One offspring per parent: \|P''\|=N (each parent breeds once via SBX -> first child + PM). | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-3 (MINOR) | FindExtremePoints (Eq.7) returns the ASF argmin per axis; | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-4 (MINOR) | Hyperplane Sum a_i f_i = 1 is solved from the m extreme points by Gaussian elimination on normalized objectives; if singular we fall back to a_i = 1/(extreme_i along axis i) (the intercept form), then to all-ones (Lab... | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-5 (MINOR) | Max-min-angle ties / zero-norm objective vectors: angle distance uses normalized objectives; a zero vector is treated as cos=1 (d=0) so it is never preferred for spread, as in isde_rd::cosang. | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-6 (MINOR) | Real genome only; binary is out of scope, as across the library. | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-7 (AMBIGUOUS — the paper contradicts itself; pseudocode followed) | Alg.3 lines 6 and 9 test \|L>1\| > min{0.45\|EA\|, \|L<1\|} (and the mirror), but the §3.3.2 prose says the count must be "more than BOTH 0.45x\|EA\| AND \|L<1\|", i.e. max{...}. | yes | `maoea_3c.hpp` |
| `maoea_3c` | 3C-8 (FIXED 2026-09-05, second primary-source pass) | Alg.2 line 2 REASSIGNS Q = [F_1..F_l] before line 4 normalizes it and line 5 searches it for extreme points; | no: fixed | `maoea_3c.hpp` |
| `maoea_3c` | 3C-9 (AMBIGUOUS -> resolved) | Alg.3 line 16 normalizes Q, so CalFitness at line 20 could be read as operating on normalized objectives. | no: fixed | `maoea_3c.hpp` |
| `emyo_c` | emyo_c.1 | the "until different" retry is capped at 10 attempts, as a guard against looping forever. | yes | `emyo_c.hpp` |
| `crea` | crea.1 | CLAMPING of lambda_k (Eq.7): on boundary reference points lambda_k = 0 and the division blows up; the denominator is clamped to 1e-6, the standard huge-penalty form. | yes | `crea.hpp` |
| `crea` | crea.2 | TIE-BREAKS: on equal perpendicular distances a solution goes to the LOWEST-indexed line (a strict "<"); on equal FT within a cluster the order is that of std::sort, which is unstable; in the last layer the choice is r... | yes | `crea.hpp` |
| `crea` | crea.3 | The reference-line count of N is produced by das_dennis::generate_auto. | yes | `crea.hpp` |
| `crea` | crea.4 | The pseudocode of Alg.1 contradicts the text of §3.1, "normalize all 2N individuals"; the code follows the text, so the pool is Rt = 2N. | yes | `crea.hpp` |
| `crea` | crea.5 | crEA has no dominance step, so constraint_mode FEASIBILITY/CDP acts on its one preference relation instead: the within-cluster ranking by FT becomes feasibility-first (feasible, then ascending CV, then ascending FT). | yes | `crea.hpp` |
| `crea` | CR-1 | init_refs moved from das_dennis::generate_exact to generate_auto. | yes | `crea.hpp` |
| `dcea` | DCEA-1 (MINOR) | Eq.8-9 is implemented with ρ=1: no history KC_NFE is collected and no ρ-row average is taken — V is simply overwritten with the current generation's K-means centres, unit-normalized. | yes | `dcea.hpp` |
| `dcea` | DCEA-2 (FIXED 2026-09-05, second primary-source pass) | The Y^w/τ association — the acute angle to V/Λ on the translated f−z_min (Eq.7). | no: fixed | `dcea.hpp` |
| `dcea` | DCEA-3 (FIXED 2026-09-05) | K-means: Lloyd with the paper's limit of 100 iterations (§3.1.2), early exit on a stable assignment. | no: fixed | `dcea.hpp` |
| `dcea` | DCEA-4 (MINOR) | NDS removal when \|Y^w\|>S: the tie-break inside the boundary front is by crowding distance (the paper: "lowest ranks", the tie is not specified). | yes | `dcea.hpp` |
| `dcea` | DCEA-5 (MINOR) | The genome is real-valued (the SCA operator lives in the decision space); a binary genome is out of coverage (NONE). | yes | `dcea.hpp` |
| `dcea` | DCEA-6 | SP^w IS INHERITED from ESM (Alg.6 l.14/18/20 "Obtain SP^w based on Y^w", l.23 "P = ∪SP^w") and from the initialization (Alg.3 l.6: Eq.2 + Algorithm 1 — balancing to exactly S: a shortfall → a random top-up from P, an... | yes | `dcea.hpp` |
| `dcea` | DCEA-7 | r1, r2 are drawn ONCE per generation — Alg.5 l.4 sets them before the loops over w and j. | yes | `dcea.hpp` |
| `dcea` | DCEA-8 (MINOR) | The number of subspaces W is rounded UP to the nearest Das–Dennis lattice cardinality: setup() calls das_dennis::generate_auto(m, W_req_) and then sets W_ = V_.size(). | yes | `dcea.hpp` |
| `dcea` | DCEA-9 (MINOR) | Order of repair and mutation. Alg.5 lines 26-27 read "y <- PM(y); repair y if necessary"; this code clamps into the bounds first and mutates second. ops::polynomial_mutation clamps its own output, so the population st... | yes | `dcea.hpp` |
| `hcca` | HCCA-1 (MINOR) | The hierarchical clustering is AGGLOMERATIVE with AVERAGE-LINKAGE (mean pairwise distance). | yes | `hcca.hpp` |
| `hcca` | HCCA-2 (MINOR) | DP environmental selection is the classical MOEA/D-Tchebycheff scheme (Eq.5-6, delta=0.9, n_r=2): an offspring updates the neighbouring subproblems. | yes | `hcca.hpp` |
| `hcca` | HCCA-3 (MINOR) | FI (Eq.3-7) is computed from the actual Δf of the current generation's PP/DP offspring; in the first generation, with no history, K = K0. | yes | `hcca.hpp` |
| `hcca` | HCCA-9 (MINOR, resolved) | Alg.1 line 12's DP update is the MOEA/D scheme, so an offspring updates the neighbourhood of the subproblem it was generated for. | no: fixed | `hcca.hpp` |
| `hcca` | HCCA-8 (MINOR) | N is revised to the Das–Dennis weight count, so the effective population can differ from pop_size when no exact lattice of that size exists (Path-A, as in nsga3.hpp). | yes | `hcca.hpp` |
| `hcca` | HCCA-4 (MINOR) | LCS Num(C) (Eq.12): floor of the area proportion, at least 1 per non-empty cluster; if the sum of Num falls short of the target size, the remainder is filled with the best by crowding (deterministic tie-break). | yes | `hcca.hpp` |
| `hcca` | HCCA-5 | FIX 2026-07-07: the final "crowding degree strategy" [38] is BCE (Li/Yang/Liu, IEEE TEVC 20(5):645-665, 2016), the niche-product crowding degree. | yes | `hcca.hpp` |
| `hcca` | HCCA-6 (MINOR) | Real-valued genome; binary is out of scope. | yes | `hcca.hpp` |
| `hcca` | HCCA-7 (DEVIATION, reference set + normalization of Eq.2) | Eq.2 is evaluated over the NDS-ACCUMULATED set Q built by Alg.2 lines 2-8, not over the full input pool PP ∪ Q1 that Alg.2 line 9 and §III name. | yes | `hcca.hpp` |
| `hcca` | HCCA-10 (MINOR) | The paper names no repair for DE/rand/1's offspring; a variable it puts outside the box is clipped (knob bound_repair), and since 2026-10-01 counted (RepairTally) and logged as "hcca_de_rand_1". | yes | `hcca.hpp` |
| `lis_lcs` | lis_lcs.1 | Eq.2–3, k-means and the CCD distances are computed on the min-max normalized objectives of the current set. | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.2 | Alg.1 lines 12–14 («DP ← DP ∪ Pool₁ ∪ Pool₂; environmental selection (DP,W,Z)") are implemented with the canonical MOEA/D-DE scheme: each offspring replaces at most n_r subproblems (the mating scope for Pool_1, all of... | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.3 | Δf (Eq.5) is clamped from below at zero to guarantee 0 <= EI' <= 1 (Eq.8); when EI_PP + EI_DP = 0, and in the first generation, K is left unchanged (the previous generation's EI_PP does not exist yet). | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.4 | The weight w_i in Eq.5: for DP offspring it is the parent subproblem's weight; for PP offspring it is the weight at the actual parent's position (the PP index, or the DP parent of a Pool_1 individual). | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.5 | EVALUATIONS PER STEP are not N, and that is Algorithm 1's own order (measured and traced 2026-09-22). | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.6 | CCD runs at the end of EVERY step(). This follows the step() framework contract — the vault's active population must be a valid result at any moment — whereas the paper runs it once on termination. | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.7 | Eq.10: "C1!=C2!=C3" is read strictly — both axis neighbours come from clusters different from X's cluster and from each other; a boundary gives BIG (the ∞ of NSGA-II). | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.8 | Weights: das_dennis::generate_auto, with the excess truncated to N (an exact lattice does not exist for every N). | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.9 | FE PER step(): (N - K_{t-1}) + K_t, NOT the flat N of Algorithm 1 line 16. | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.10 | dp_update replaces on a STRICT improvement (g^tch(y) < g^tch(x^j)); the paper gives no tie rule for line 14. | yes | `lis_lcs.hpp` |
| `lis_lcs` | lis_lcs.11 | setup() initializes both populations at random (2N evaluations, as in the paper); setup_seeded() copies the seeded population into both DP and PP. | yes | `lis_lcs.hpp` |
| `clia` | CLIA-1 (MAJOR, the RPIL classifier) | The paper uses an INCREMENTAL SVM (Gaussian kernel <S,C> = <0.056,10>) on reference points projected by Gram-Schmidt from M to M−1, to estimate the "effective regions" of the simplex. | yes | `clia.hpp` |
| `clia` | CLIA-2 (MINOR) | The M -> M−1 Gram-Schmidt projection for the classifier is replaced by working directly on the simplex, where the points already lie on Sum = 1. | yes | `clia.hpp` |
| `clia` | CLIA-3 (MINOR) | PDM uses \|\|o\|\|_2 * sin(o,z), the full form of Eq.1; the pseudocode of Alg.1 writes mean(f) + sin(z,f). | yes | `clia.hpp` |
| `clia` | CLIA-4 | "Denser reference points" (ge2019, footnote 3): for M < 8 the density of the SINGLE-LAYER lattice is raised (D <- D+1); for M >= 8 the TWO-LAYER Das-Dennis scheme is used (dense boundary Hb, sparse inner Hi = Hb/2), a... | yes | `clia.hpp` |
| `clia` | CLIA-5 (MINOR) | One offspring per pair (SBX, first child), N pairs drawn at random from the population. | yes | `clia.hpp` |
| `clia` | CLIA-6 (MINOR) | Real-valued genome; binary and mixed are out of scope (constraint_mode=NONE). | yes | `clia.hpp` |
| `clia` | CLIA-7 (AMBIGUOUS, the kernel width) | §IV-B gives <S,C> = <0.056,10>, "S is the kernel scale of the Gaussian kernel", and no formula. | yes | `clia.hpp` |
| `two_arch2` | TA2-1 (AMBIGUOUS, reading fixed 2026-09-09) | The paper is SILENT on whether the objectives are scaled before I_eps+: Eq.1 is written on raw values, while IBEA — whose Eq.2 fitness this reuses verbatim — scales them to [0,1] over the pool in its own Alg.2 (steps... | no: fixed | `two_arch2.hpp` |
| `two_arch2` | TA2-2 (MINOR) | DA "boundary solutions with maximal or minimal objective values" (Alg. 3 line 3) implemented as, per objective i, the argmin and argmax of f_i (up to 2m points, de-duplicated). | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-3 (reading) | If after Pareto filtering \|DA\| <= n_DA, DA is kept as-is, with no padding. | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-4 (INTENTIONAL) | Reproduction realizes crossover and mutation as TWO INDEPENDENT operators, per Sec III-A ("crossover and mutation are independent in Two_Arch2") and Sec IV-E (variation experiments treat mutation-on-archive and crosso... | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-5 (FIXED 2026-09-05, second primary-source pass) | The defaults used to be eta_c=eta_m=20 "to match the rest of MOOtation"; | no: fixed | `two_arch2.hpp` |
| `two_arch2` | TA2-7 (MINOR) | Stream sizes. The paper fixes only the population/FE budget and says nothing about how many crossover children versus mutants are bred per generation; the ~50/50 split of n_DA offspring is this port's choice (see TA2-4). | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-8 (letter, noted 2026-09-16) | Parents are drawn uniformly: §V-E-2 describes the crossover as "one parent from CA and the other from DA" and names no mating selection, and Eq.(2) is -e^{-I/0.05} with no c = max\|I\| factor (the scaling IBEA adds is h... | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-6 (MINOR) | Real genome only; binary is out of scope. | yes | `two_arch2.hpp` |
| `naemo` | NAEMO-1 (DEVIATION) | Eq.19 formally applies polynomial mutation to "all j"; implemented as Deb's CANONICAL polynomial mutation with pm=1/n_vars. | yes | `naemo.hpp` |
| `naemo` | NAEMO-2 (MINOR) | PBI and association operate in the translated space x' = f − z* (z* being the archive ideal point); the paper draws the lines through the origin in the raw space (for DTLZ, f>=0 and z*~0) — this is the MOEA/D-PBI conv... | yes | `naemo.hpp` |
| `naemo` | NAEMO-3 (MINOR) | §4.4 takes the DE points "from the archive", while §4.1/4.9 restrict reproduction to the neighbourhood. | yes | `naemo.hpp` |
| `naemo` | NAEMO-4 (MINOR) | Gaussian(μ,v): v is read as the VARIANCE (text of §4.5: "mean μ_ηc and variance 5"), so std = sqrt(v). | yes | `naemo.hpp` |
| `naemo` | NAEMO-5 (MINOR) | k=⌊0.2·n⌋ (§5.2: «20% of the total number of reference lines"); the soft limit is pop_size (the paper: C·n, with C a small constant, see Table 2). | yes | `naemo.hpp` |
| `naemo` | NAEMO-6 (MINOR) | z* and PBI are recomputed once per generation (at the start of step) and held fixed for the inner loop. | yes | `naemo.hpp` |
| `naemo` | NAEMO-8 (MINOR) | Mating pool = the parent's own sub-archive PLUS the k nearest non-empty lines. | yes | `naemo.hpp` |
| `naemo` | NAEMO-7 (MINOR, resolved) | The reference-line count is chosen as the LARGEST attainable Das-Dennis lattice <= pop_size on the default path. das_dennis::generate_auto rounds UP, and since every line keeps at least one member, a lattice larger th... | no: fixed | `naemo.hpp` |
| `paes` | PAES-1 | c's density when c lies outside the current grid (it is not in the archive): m becomes c; m outside: c stays (jMetal 4/5's test()). | yes | `paes.hpp` |
| `paes` | PAES-2 | The member that leaves a full archive: the first, in archive order, of the most crowded cell (jMetal's prune()); m's cell is located on the grid widened to m when m lies outside it. | yes | `paes.hpp` |
| `paes` | PAES-3 | Equal objectives count as mutually non-dominated; the archive takes no copy of a member's objectives. | yes | `paes.hpp` |
| `paes` | PAES-4 | Constraints: Deb's constrained domination wherever domination is asked (constraint_mode not NONE), as jMetal's comparator. | yes | `paes.hpp` |
| `paes` | PAES-5 | The first c: a uniform random point (seeded: the first seed). | yes | `paes.hpp` |
| `paes` | PAES-6 | The grid follows the archive's range at every change; | yes | `paes.hpp` |
| `ctaea` | CTAEA-1 | Algorithm 2's infeasible branch (lines 23-30) begins with S <- S_c and loops while \|S\| < N (the printed \|S_c\| < N cannot end), and line 29 cuts the level added last; | yes | `ctaea.hpp` |
| `ctaea` | CTAEA-2 | Eq. 8 divides by z_nad - z*; the authors' code (as pymoo carries it) only subtracts z*, and so does this: with the division, the DA's far infeasible members squeeze the front onto a few weights. z* is the least of eac... | yes | `ctaea.hpp` |
| `ctaea` | CTAEA-3 | The supplement with the operators' settings is not at hand: SBX (eta_c 30, pc 1) and polynomial mutation (eta_m 20, pm 1/n), and one child of each pair, N pairs per generation — the authors' code as pymoo carries it. | yes | `ctaea.hpp` |
| `ctaea` | CTAEA-4 | Feasibility is cv == 0 by the vault (so EPS_LEVEL relaxes it); with constraint_mode NONE every point is feasible. | yes | `ctaea.hpp` |
| `ccmo` | CCMO-1 | The supplement asks for a wrapper over any core; | yes | `ccmo.hpp` |
| `ccmo` | CCMO-2 | The SPEA2 truncation runs within the last front, as the crowding distance it replaces does, on the objective vectors as they are. | yes | `ccmo.hpp` |
| `ccmo` | CCMO-3 | A seed population (setup_seeded) is both populations: the helper starts as its copy, so that a warm start spends nothing. | yes | `ccmo.hpp` |
| `dms` | DMS-1 (task, 2026-09-23) | The variables are normalised to [0, 1]^n: poll steps are α·(u − ℓ) per variable, α_0 = 1, α_ε = 10^-3 of the range. | yes | `dms.hpp` |
| `dms` | DMS-2 | The poll centre is the first point IN LIST ORDER whose α is at least α_ε. | yes | `dms.hpp` |
| `dms` | DMS-3 | The paper evaluates every feasible poll point, including one equal to a point already in the list (its SP1 example, iteration 1, does); so does this port. | yes | `dms.hpp` |
| `dms` | DMS-4 | Constraints: with constraint_mode FEASIBILITY a point with cv > 0 is infeasible and filtered out, the extreme barrier (2); with NONE the limits are ignored, as in every core. | yes | `dms.hpp` |
| `dms` | DMS-5 | Continuous variables only; a problem with binary ones is refused. | yes | `dms.hpp` |
| `dms` | DMS-6 (task 5, 2026-10-01; off by default) | The method has no random number, so its ten seeds of a campaign are one run, and the line's points and their dyadic steps (α = 1, 1/2, 1/4, ... of the range) land exactly on the middle and the bounds of the box, where... | no: option | `dms.hpp` |
| `dmulti_mads` | DMM-1 | The poll directions: an orthonormal basis H = I − 2vvᵀ (a Householder matrix of a unit vector v drawn from the seed, as OrthoMADS's [1] Halton vector, here from the run's generator), each column turned towards the tar... | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-2 | The mesh: the classic MADS one, δ = min(Δ, Δ²) (Algorithm 2), in the normalized box; a direction u becomes the mesh vector z = round(u / ‖u‖_∞ · Δ/δ), so that ‖δz‖_∞ = Δ. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-3 | τ = 1/2 and Δ_0 = 0.1: the paper's runs leave both to [8]'s mesh. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-4 | The target direction taken onto the current mesh by rounding; a speculative point outside the box or equal to x_k is not evaluated. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-5 | Constraints: the extreme barrier (§4.1) — a point with cv > 0 is infeasible and never enters L (constraint_mode FEASIBILITY; | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-6 | Points outside the box: the extreme barrier, not evaluated. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-7 | A poll step that would leave the box stops at its last mesh point inside: z_r is cut to the integers with 0 <= x_r + δz_r <= 1, so the point stays on the mesh and in the frame; a point equal to x_k or to an earlier po... | yes | `dmulti_mads.hpp` |
| `multiglods` | MG-1 | The box normalized to [0, 1]^n, as dms does: α_0 = n, distances and radii in the normalized box (the paper works in the problem's units; with its α_0 = n·max range the two agree on boxes of one range). | yes | `multiglods.hpp` |
| `multiglods` | MG-2 | The Sobol sequence unscrambled, from its index 1, continued from one search to the next; a point outside the domain (constraints) is evaluated and refused by the extreme barrier. | yes | `multiglods.hpp` |
| `multiglods` | MG-3 | Constraints: the extreme barrier — an infeasible point (cv > 0) never enters the list (constraint_mode FEASIBILITY); | yes | `multiglods.hpp` |
| `multiglods` | MG-4 | The order of the list: insertion; the poll centre's ties (equal α): the first. | yes | `multiglods.hpp` |
| `multiglods` | MG-5 | Poll points outside the box are not evaluated (the extreme barrier, as dms). | yes | `multiglods.hpp` |
| `multiglods` | MG-6 | "Each time that three consecutive unsuccessful iterations occurred" (§4): a success or a merging iteration breaks the run, and a search starts the count again. | yes | `multiglods.hpp` |
| `smpso` | SMPSO-1 | "χ" multiplies the whole of Eq. 2 including w·v ("the resulting velocity is then multiplied by the constriction factor", §II-A). | yes | `smpso.hpp` |
| `smpso` | SMPSO-2 | The leader is drawn for every particle; §II-B's tournament reads "pbest" where the leader is meant. | yes | `smpso.hpp` |
| `smpso` | SMPSO-3 | p_i is replaced unless it dominates the new position, by Deb's constrained domination under constraint handling (constraint_mode not NONE), Pareto dominance otherwise; the same comparison admits leaders. | yes | `smpso.hpp` |
| `smpso` | SMPSO-4 | A leader whose objectives equal a member's exactly is not added. | yes | `smpso.hpp` |
| `smpso` | SMPSO-5 | The velocity starts at 0, p_i at the initial position (Alg. 1, line 1). | yes | `smpso.hpp` |
| `smpso` | SMPSO-6 | Eq. 4 sets φ = 1 for C1 + C2 <= 4, which makes Eq. 3 complex; χ = 1 there, as jMetal has it (and the modulus form gives at φ = 4). | yes | `smpso.hpp` |
| `smpso` | SMPSO-7 | OMOPSO's mutation: three parts of equal size, none, uniform and non-uniform (OMOPSO §3; the SMPSO paper's §II-C says 30 % each and none on the rest); here interleaved by index as jMetal's OMOPSO does (i % 3). | yes | `smpso.hpp` |
| `smpso` | SMPSO-8 | platemo's turbulence draws for each particle (probability 0.15), jmetal's mutates i % 6 == 0, a sixth: §II-C's "15 %" read both ways. | yes | `smpso.hpp` |
| `smpso` | SMPSO-9 | OMOPSO's ε-dominance is relative, f_i(x1)/(1 + ε) <= f_i(x2) (§3), and its archive procedure is not given; the archive here is jMetal's: additive boxes floor(f/η), in one box the point nearer the corner. | yes | `smpso.hpp` |
| `rm_meda` | RM-1 | The first L_j's directions: an orthonormal basis drawn at random (Gram-Schmidt on standard normal vectors); the paper names only the random point. | yes | `rm_meda.hpp` |
| `rm_meda` | RM-2 | Local PCA stops when the partition repeats, or after 100 rounds. | yes | `rm_meda.hpp` |
| `rm_meda` | RM-3 | A cluster with fewer than two points keeps its space from the round before (no covariance to take one from) and has no model: it is never sampled. pop_size must exceed n_clusters, or setup() refuses. | yes | `rm_meda.hpp` |
| `rm_meda` | RM-4 | N(0, σ I) is read as the covariance σ I: ε' has standard deviation √σ_τ in each variable (σ_j of Eq. 5 is a mean of variances). | yes | `rm_meda.hpp` |
| `rm_meda` | RM-5 | vol(Ψ^j) is the product of Ψ^j's side lengths (1 at m = 1). | yes | `rm_meda.hpp` |
| `rm_meda` | RM-6 | The crowding distance is the paper's: the average side length of the cuboid around F(x) in the objectives, not normalized (NSGA-II divides by each objective's range); the extremes of a front count infinite. | yes | `rm_meda.hpp` |
| `rm_meda` | RM-7 | Constraints: Deb's constrained domination in the sorting (constraint_mode not NONE); the paper has none. | yes | `rm_meda.hpp` |
| `rm_meda` | RM-8 | §V-B resets a variable outside the box to a random value inside it, but the paper's own curves (Figs. 3, 4, 11, 12: F1, F2, F5, F6 at 10 000 evaluations) are reached with the variable kept at the bound, not with the r... | yes | `rm_meda.hpp` |

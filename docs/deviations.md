# Departures from the papers

Written by `tools/deviations.py` from the headers of the cores (`include/mootation/algorithms`) and
of the solvers (`include/mootation/scalar`); do not edit by hand — edit the header and run the tool.
The header is the source: there each item is given in full, with its reasons and references.

*Default run*: `yes` — the default run does what the item describes; `no: option` — the item is
about something off by default; `no: fixed` — fixed, resolved, retired, or not a deviation. The
column is read from the item's wording (see the tool's docstring), not stated by the item.

399 items: 316 yes, 26 no: option, 57 no: fixed. Headers that declare none: `spea2_sde`, `adaw`, `camoea`, `maoea_c`. Headers with no item this tool recognises: `dn_nsga2`, `bc_rex`, `rvea_star`, `mosa`.

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
| `nsga3` | NSGA3-ADA (task 15 of 2026-10-21, §6; off by default) | ada = 1: ADA in place of the generation — one child a step, P growing, the answer A^primary (operators/ada.hpp; | no: option | `nsga3.hpp` |
| `a_nsga3` | ANSGA3-L (task 15 of 2026-10-21, §2; off by default) | n_offspring (λ children a generation, μ survivors from the μ + λ) and n_ref_points (H reference points, das_dennis::reference_points) apart from pop_size, as NSGA3-L in nsga3.hpp; not set, λ = μ and H = μ's lattice, a... | no: option | `a_nsga3.hpp` |
| `a_nsga3` | a_nsga3.1 | Path-A: pop_size = the Das-Dennis lattice size (see nsga3.hpp); | yes | `a_nsga3.hpp` |
| `theta_dea` | TDEA-ADA (task 15 of 2026-10-21, §6; off by default) | ada = 1: ADA in place of the generation — one child a step, P growing, the answer A^primary (operators/ada.hpp; | no: option | `theta_dea.hpp` |
| `theta_dea` | TDEA-L (task 15 of 2026-10-21, §2; off by default) | n_offspring (λ children a generation, μ survivors from the μ + λ) and n_ref_points (H reference points, das_dennis::reference_points) apart from pop_size, as NSGA3-L in nsga3.hpp; not set, λ = μ and H = μ's lattice, a... | no: option | `theta_dea.hpp` |
| `theta_dea` | TDEA-1 (from the letter of Definition 7; declared 2026-10-24) | Equal F_j in one cluster — copies of a solution, which random mating breeds whenever a parent is drawn twice and no variable mutates — are ordered by their slots, so that only one of them is theta-optimal. | yes | `theta_dea.hpp` |
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
| `moead_ds` | DDS-8 (READING; knob extreme_weights, nsga3 by default) | Alg.3 line 5 puts the extreme points into Q unconditionally ("Q = Extreme"), so which points Eq.9 calls extreme decides which points survive every generation. | yes | `moead_ds.hpp` |
| `moead_ds` | DDS-7 (fixed 2026-09-05) | Mating d2 (Alg.4 line 2) uses the reference points R returned by the LAST DDS call (Alg.1 line 3 for the first generation, line 12 afterwards) — i.e. the normalization frame of the merged 2N pool, as Alg.1 line 8 cons... | no: fixed | `moead_ds.hpp` |
| `moead_ds` | DDS-9 (GUARD; knob nadir_source, off by default: intercepts) | On MaF3 at eight objectives the population can fall into points with one enormous objective and the others near zero (dominance resistant solutions); the intercepts through its own extreme points then put them on the... | no: option | `moead_ds.hpp` |
| `moead_ds` | DDS-10 (GUARD; knob drs_guard, off by default) | The same trap, met at the extreme points instead of the nadir. | no: option | `moead_ds.hpp` |
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
| `aprd` | APRD-11 (READING; knob slot_exclusive, off by default) | The other reading of APRD-4: Alg.3 Step 4 fills S_i slots of a subregion, which reads as S_i DIFFERENT individuals; slot_exclusive = on takes each slot's PBI argmin among the members not yet selected, so a subregion's... | no: option | `aprd.hpp` |
| `aprd` | APRD-5 (MINOR) | Case I NSGA-II: (rank, CD) tournament plus SBX/PM, then NDS+CD selection. | yes | `aprd.hpp` |
| `aprd` | APRD-6 (MINOR / partly FIXED 2026-09-06, full-paper checklist) | SBX yields the first child — one offspring per x, Alg.1's "a new solution z". | no: fixed | `aprd.hpp` |
| `aprd` | APRD-7 (resolved) | update_archive receives Arc u Q in BOTH cases (Alg.1: "Arc <- UpdateArchive(Arc u Q, N_a)"), not pop u Q as it once did. | no: fixed | `aprd.hpp` |
| `aprd` | APRD-8 (resolved) | the partition P_1..P_K is built by Algorithm 3 (update_subpops) and REUSED by the next generation's mating (Alg.1: for k, foreach x in P_k), rather than recomputed by a random-start FPS at every step. | no: fixed | `aprd.hpp` |
| `aprd` | APRD-9 (FIXED 2026-09-06, full-paper checklist) | Algorithm 2 truncates the archive by the Max-Min principle of §II-B, whose measure is the ANGLE: "select the first point in the remaining points and make it have the maximum angle to first selected point; … the next p... | no: fixed | `aprd.hpp` |
| `aprd` | APRD-12 (READING; knob aprd_pbi, off by default: canonical) | The other reading of APRD-10: the paper writes PBI that way throughout - its recap of the standard PBI (Eq.6, md:250) already has d2 = \|F - (z* - d1 lam)\| with an unnormalized lam - so Eq.7-8 may be the authors' PBI,... | no: option | `aprd.hpp` |
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
| `hlmea` | HLMEA-13 (task 16, supplement 1) | An offspring equal to a point evaluated before is evaluated again and counts in the budget, as in the paper: on DTLZ2_3D at 2 000 evaluations about one call in eight. | yes | `hlmea.hpp` |
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
| `irea` | IREA-1 (READING; knob fitness_reading, swap by default) | The other consistent resolution is IBEA's own: Eq.7 as printed, F(x1) = Sum −e^{−I(x2,x1)/kappa}, with the LARGEST F kept — what "as done in [31]" (Li et al. 2016, after IBEA) does. | yes | `irea.hpp` |
| `irea` | irea.2 | Eq. 6 as printed is a distance to a point, not a perpendicular. | yes | `irea.hpp` |
| `irea` | irea.3 | §3.4: "if extreme point set contains duplicate extreme points, or intercepts are nonnegative [an evident typo for non-positive], then intercept can be replaced with each maximum objective functions" — a degenerate mat... | yes | `irea.hpp` |
| `irea` | irea.4 | das_dennis::generate_exact requires N to equal the lattice point count, which is stricter than Table II of the paper, where N may differ slightly. | yes | `irea.hpp` |
| `edv` | EDV-1 (FIXED 2026-09-06, third primary-source pass) | p_m = 1/pop_size, the paper's letter: §4.2.4 is explicit and self-consistent — "the mutation probability (p_m) is set to 1/pops. | no: fixed | `edv.hpp` |
| `edv` | EDV-2 (FIXED 2026-09-05, second primary-source pass) | Alg.3 lines 2-11 build Y one offspring at a time: each iteration draws its own parents Pa by binary tournament, "y = SBX(Pa)" yields ONE offspring y, and PM is applied to that y. | no: fixed | `edv.hpp` |
| `edv` | EDV-3 (READING) | Alg.4 line 8 takes the angle between F(x) and d^i; | yes | `edv.hpp` |
| `edv` | EDV-4 (DECLARED) | Table 3 lists N = 276 at m = 10 for the (3,2) two-layer lattice, whose size is C(12,9) + C(11,9) = 275; this port requires \|D\| = N exactly (generate_exact), so 276 throws and 275 is the size to use. | yes | `edv.hpp` |
| `edv` | EDV-5 (READING; knob edv_tournament, fitness by default) | The other reading of the ASSUMPTIONS above: the experiments ran on jMetal (§4, [37]), whose binary tournament compares by Pareto dominance and breaks a tie at random; | yes | `edv.hpp` |
| `edv` | EDV-6 (READING; knob normalize, off by default) | Eq.2, 4 and 5 compute I_eps+ on the raw objectives, and the angles of Alg.4 line 8 are taken on F(x) (f − z*, EDV-3), and the paper says nothing of scaling. | no: option | `edv.hpp` |
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
| `maoea_iamd` | maoea_iamd.4 | eta_m_ / set_eta_mutation act only under kpcm_pm = on (IAMD-1); by default the paper's KPCM replaces PM. | yes | `maoea_iamd.hpp` |
| `maoea_iamd` | IAMD-1 (READING; knob kpcm_pm, off by default) | §4.3(2): "all compared algorithms employ SBX and PM", and the experiments ran on PlatEMO (§4), whose real-coded GA (OperatorGA, OperatorGAhalf) is SBX followed by Deb's polynomial mutation in one call, with no SBX-onl... | no: option | `maoea_iamd.hpp` |
| `maoea_iamd` | IAMD-2 (DEVIATION from [32]; knob cv_rule, angle by default) | The paper classifies with "the decision variable clustering method [32]", LMEA's Alg. 5 (Zhang et al., TEVC 2018): lines 9-14 fit a line to each variable's perturbed points, CV = the variables whose mean fit error is... | yes | `maoea_iamd.hpp` |
| `maoea_iamd` | IAMD-3 (DEVIATION; knob cv_norm, population by default) | From [32]'s implementation: [32] Alg. 5 line 6 says only "Normalize SP"; | yes | `maoea_iamd.hpp` |
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
| `rvea` | RVEA-ADA (task 15 of 2026-10-21, §6; off by default) | ada = 1: ADA in place of the generation — one child a step, P growing, the answer A^primary (operators/ada.hpp; | no: option | `rvea.hpp` |
| `maoeaarv` | maoeaarv.1 | η_c=η_m=20 and the linkage of the hierarchical clustering (single-linkage here) are NOT specified by the paper — implementation conventions were chosen. | yes | `maoeaarv.hpp` |
| `maoeaarv` | ARV-2 (READING; knob arv_linkage, off by default: single) | §4.4 (Alg.4 line 3) cites "the hierarchical clustering strategy [34]" with the Euclidean distance and no linkage; [34] gives none either. single is the port's choice (MATLAB's clusterdata defaults to it, the experimen... | no: option | `maoeaarv.hpp` |
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
| `dea_gng` | GNG-11 (READING; knobs gng_eps and gng_ties, off by default) | Of PlatEMO's code. PlatEMO 4.16's DEAGNG.m sets eps = 0.314 (0.1 pi) for every M (the paper, §IV-A: 0.05 pi below five objectives, 0.15 pi from five) and draws the argmin c_j ties at random (ESelection.m; | no: option | `dea_gng.hpp` |
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
| `hcca` | HCCA-11 (FIX 2026-11-01) | The DE parents follow MOEA/D-DE (Li & Zhang 2009, Steps 2.1-2.2), whose δ, n_r, CR and F Table 2 takes: P = B(i) with probability δ, else all of DP; r1 = i (the member M_i of Algorithm 1 line 6, "crossover and mutatio... | yes | `hcca.hpp` |
| `hcca` | HCCA-12 (reading) | T = 20, the value the sister CSMOEA of the same authors gives (Information Sciences 624, 2023, §4.1) with the same δ = 0.9 and n_r = 2; | yes | `hcca.hpp` |
| `hcca` | HCCA-13 (FIX 2026-11-01) | DP and PP come from two independent random samples of N_ points each (§III: "two populations, DP and PP, with N individuals are randomly sampled"), 2N evaluations; the code before copied one sample into both. | yes | `hcca.hpp` |
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
| `lis_lcs` | LIS-1 (reading; knob de_base, current1 by default) | The DE base vector is the member i of M_DP itself: x_i + F·(x_r2 − x_r3), r2 and r3 from i's mating scope (B(i) with probability δ, else all of DP), different from i and from each other. | yes | `lis_lcs.hpp` |
| `lis_lcs` | LIS-2 (reading; knob k_share, pp by default) | Eq.9 as printed sets K = [EI_DP'·N], yet K is the number of PP offspring (M_PP, line 10), and the text under it says "A better performance of offspring will generally lead to a larger number of clusters and, thus, a l... | yes | `lis_lcs.hpp` |
| `clia` | CLIA-1 (MAJOR, the RPIL classifier) | The paper uses an INCREMENTAL SVM (Gaussian kernel <S,C> = <0.056,10>) on reference points projected by Gram-Schmidt from M to M−1, to estimate the "effective regions" of the simplex. | yes | `clia.hpp` |
| `clia` | CLIA-2 (MINOR) | The M -> M−1 Gram-Schmidt projection for the classifier is replaced by working directly on the simplex, where the points already lie on Sum = 1. | yes | `clia.hpp` |
| `clia` | CLIA-3 (MINOR) | PDM uses \|\|o\|\|_2 * sin(o,z), the full form of Eq.1; the pseudocode of Alg.1 writes mean(f) + sin(z,f). | yes | `clia.hpp` |
| `clia` | CLIA-4 | "Denser reference points" (ge2019, footnote 3): for M < 8 the density of the SINGLE-LAYER lattice is raised (D <- D+1); for M >= 8 the TWO-LAYER Das-Dennis scheme is used (dense boundary Hb, sparse inner Hi = Hb/2), a... | yes | `clia.hpp` |
| `clia` | CLIA-5 (MINOR) | One offspring per pair (SBX, first child), N pairs drawn at random from the population. | yes | `clia.hpp` |
| `clia` | CLIA-6 (MINOR) | Real-valued genome; binary and mixed are out of scope (constraint_mode=NONE). | yes | `clia.hpp` |
| `clia` | CLIA-7 (AMBIGUOUS, the kernel width) | §IV-B gives <S,C> = <0.056,10>, "S is the kernel scale of the Gaussian kernel", and no formula. | yes | `clia.hpp` |
| `clia` | CLIA-8 (READING; knob clia_norm, off by default: raw) | Of the authors' code. The paper clusters on raw objectives (§III-A, "we avoid the changes ... caused by normalization"); | no: option | `clia.hpp` |
| `clia` | CLIA-9 (READING; knob clia_rpil, off by default: on) | Of the authors' code. PlatEMO's initialize.m sets disable_flag = true (no reference point learning) on MaF3, 5, 10, 11, 12 and 14; clia_rpil = off skips RPIL, the reference vectors staying the initial ones. | no: option | `clia.hpp` |
| `two_arch2` | TA2-1 (AMBIGUOUS, reading fixed 2026-09-09) | The paper is SILENT on whether the objectives are scaled before I_eps+: Eq.1 is written on raw values, while IBEA — whose Eq.2 fitness this reuses verbatim — scales them to [0,1] over the pool in its own Alg.2 (steps... | no: fixed | `two_arch2.hpp` |
| `two_arch2` | TA2-2 (MINOR) | DA "boundary solutions with maximal or minimal objective values" (Alg. 3 line 3) implemented as, per objective i, the argmin and argmax of f_i (up to 2m points, de-duplicated). | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-3 (reading) | If after Pareto filtering \|DA\| <= n_DA, DA is kept as-is, with no padding. | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-4 (INTENTIONAL) | Reproduction realizes crossover and mutation as TWO INDEPENDENT operators, per Sec III-A ("crossover and mutation are independent in Two_Arch2") and Sec IV-E (variation experiments treat mutation-on-archive and crosso... | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-5 (FIXED 2026-09-05, second primary-source pass) | The defaults used to be eta_c=eta_m=20 "to match the rest of MOOtation"; | no: fixed | `two_arch2.hpp` |
| `two_arch2` | TA2-7 (MINOR) | Stream sizes. The paper fixes only the population/FE budget and says nothing about how many crossover children versus mutants are bred per generation; the ~50/50 split of n_DA offspring is this port's choice (see TA2-4). | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-8 (letter, noted 2026-09-16) | Parents are drawn uniformly: §V-E-2 describes the crossover as "one parent from CA and the other from DA" and names no mating selection, and Eq.(2) is -e^{-I/0.05} with no c = max\|I\| factor (the scaling IBEA adds is h... | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-6 (MINOR) | Real genome only; binary is out of scope. | yes | `two_arch2.hpp` |
| `two_arch2` | TA2-9 (task 16, supplement 1) | An offspring equal to a point evaluated before is evaluated again and counts in the budget, as in the paper: on DTLZ2_3D at 2 000 evaluations about one call in six. | yes | `two_arch2.hpp` |
| `naemo` | NAEMO-1 (DEVIATION) | Eq.19 formally applies polynomial mutation to "all j"; implemented as Deb's CANONICAL polynomial mutation with pm=1/n_vars. | yes | `naemo.hpp` |
| `naemo` | NAEMO-2 (MINOR) | PBI and association operate in the translated space x' = f − z* (z* being the archive ideal point); the paper draws the lines through the origin in the raw space (for DTLZ, f>=0 and z*~0) — this is the MOEA/D-PBI conv... | yes | `naemo.hpp` |
| `naemo` | NAEMO-3 (MINOR) | §4.4 takes the DE points "from the archive", while §4.1/4.9 restrict reproduction to the neighbourhood. | yes | `naemo.hpp` |
| `naemo` | NAEMO-4 (MINOR) | Gaussian(μ,v): v is read as the VARIANCE (text of §4.5: "mean μ_ηc and variance 5"), so std = sqrt(v). | yes | `naemo.hpp` |
| `naemo` | NAEMO-5 (MINOR) | k=⌊0.2·n⌋ (§5.2: «20% of the total number of reference lines"); the soft limit is pop_size (the paper: C·n, with C a small constant, see Table 2). | yes | `naemo.hpp` |
| `naemo` | NAEMO-6 (MINOR) | z* and PBI are recomputed once per generation (at the start of step) and held fixed for the inner loop. | yes | `naemo.hpp` |
| `naemo` | NAEMO-8 (MINOR) | Mating pool = the parent's own sub-archive PLUS the k nearest non-empty lines. | yes | `naemo.hpp` |
| `naemo` | NAEMO-9 (SETTING; knobs flag1, flag2, off by default) | §5.2 sets the two flags per problem: "flag2 is set as true for DTLZ1 while flag1 is set as true for DTLZ3", false elsewhere. | no: option | `naemo.hpp` |
| `naemo` | NAEMO-7 (MINOR, resolved) | The reference-line count is chosen as the LARGEST attainable Das-Dennis lattice <= pop_size on the default path. das_dennis::generate_auto rounds UP, and since every line keeps at least one member, a lattice larger th... | no: fixed | `naemo.hpp` |
| `paes` | PAES-1 | c's density when c lies outside the current grid (it is not in the archive): m becomes c; m outside: c stays (jMetal 4/5's test()). | yes | `paes.hpp` |
| `paes` | PAES-2 | The member that leaves a full archive: the first, in archive order, of the most crowded cell (jMetal's prune()); m's cell is located on the grid widened to m when m lies outside it. | yes | `paes.hpp` |
| `paes` | PAES-3 | Equal objectives count as mutually non-dominated; the archive takes no copy of a member's objectives. | yes | `paes.hpp` |
| `paes` | PAES-4 | Constraints: Deb's constrained domination wherever domination is asked (constraint_mode not NONE), as jMetal's comparator. | yes | `paes.hpp` |
| `paes` | PAES-5 | The first c: a uniform random point (seeded: the first seed). | yes | `paes.hpp` |
| `paes` | PAES-6 | The grid follows the archive's range at every change; | yes | `paes.hpp` |
| `paes` | PAES-7 (task 16, supplement 1) | A mutant equal to its parent (no variable mutated) is evaluated and counts in the budget: Figure 1 evaluates every m, and only the archive refuses copies. | yes | `paes.hpp` |
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
| `dms` | DMS-7 (task 16, supplement 1; task 17, supplement 2) | A cache: a point polled again takes its values from the run's cache of every point evaluated, as DMulti-MADS's V^k (DMM-8), brought to the vault's present (the tolerance, the level, the failures' penalty), and costs n... | yes | `dms.hpp` |
| `dmulti_mads` | DMM-1 | The poll directions: an orthonormal basis H = I − 2vvᵀ (a Householder matrix of a unit vector v drawn from the seed, as OrthoMADS's [1] Halton vector, here from the run's generator), each column turned towards the tar... | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-2 | The mesh: the classic MADS one, δ = min(Δ, Δ²) (Algorithm 2), in the normalized box; a direction u becomes the mesh vector z = round(u / ‖u‖_∞ · Δ/δ), so that ‖δz‖_∞ = Δ. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-3 | τ = 1/2 and Δ_0 = 0.1: the paper's runs leave both to [8]'s mesh. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-4 | The target direction taken onto the current mesh by rounding; a speculative point outside the box or equal to x_k is not evaluated. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-5 | Constraints: the extreme barrier (§4.1) — a point with cv > 0 is infeasible and never enters L (constraint_mode FEASIBILITY; | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-6 | Points outside the box: the extreme barrier, not evaluated. | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-7 | A poll step that would leave the box stops at its last mesh point inside: z_r is cut to the integers with 0 <= x_r + δz_r <= 1, so the point stays on the mesh and in the frame; a point equal to x_k or to an earlier po... | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-8 (task 15 of 2026-10-21, §8; the default since 2026-10-22) | The cache V^k of Algorithm 1 ("Update the cache V^{k+1}"): MADS evaluates no point twice — the NOMAD 4.6.0 user guide: "before starting an evaluation, the cache is scanned for the point". mads_cache = on (the default)... | yes | `dmulti_mads.hpp` |
| `dmulti_mads` | DMM-9 (fixed 2026-10-22, task 16) | A failure shrinks the frame of the member polled — found by its variables and its values. | no: fixed | `dmulti_mads.hpp` |
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

## The single-objective solvers and methods (`include/mootation/scalar`)

Their readings, from every comment of each file (the first column the file's name).

365 items: 320 yes, 45 no: option, 0 no: fixed. Headers that declare none: —. Headers with no item this tool recognises: `eigen`, `genome`, `subproblems`.

| solver | item | what | default run | file |
|---|---|---|---|---|
| `atomic_measure` | AM-1 | Phase 2's start a = E(l)_W^+ y (Algorithm 2, step 1): when that has a coefficient <= 0, the start is NNLS on E(l) instead and the exponents with a zero coefficient are dropped (R80 assumes phase 1's clusters give a po... | yes | `atomic_measure.hpp` |
| `atomic_measure` | AM-2 | The line search (R80: "line search to determine mu_s", no rule): mu = 1, 1/2, 1/4, ... | yes | `atomic_measure.hpp` |
| `atomic_measure` | AM-3 | The certificate is checked with a tolerance and a scale: rho(lambda) / (\|\|y\|\|_W \|\|e(lambda)\|\|_W) <= tol_cert (1e-9), on the grid of phase 1's formula with 10 m points and 10 d (the same span, ten times denser). | yes | `atomic_measure.hpp` |
| `atomic_measure` | AM-4 | Phase 2 runs at most max_iter = 100 steps; the convergence test is R80's absolute \|\| J h \|\|_W < tol (scale the data, or set tol). | yes | `atomic_measure.hpp` |
| `atomic_measure` | AM-5 | Phase 1's grids stop at m_max = 160; the result is then the last phase 2's, with certified = false. | yes | `atomic_measure.hpp` |
| `atomic_measure` | AM-6 | NNLS takes the problem scaled by sqrt(w_i) row by row. | yes | `atomic_measure.hpp` |
| `atomic_measure` | AM-7 | Phase 2 converges quadratically only at the global optimum, where -alpha_k rho''(lambda_k) >= 0 and \|G\| is the true (2,2) block. | yes | `atomic_measure.hpp` |
| `boostdms` | BD-1 | The box normalized to [0, 1]^n (as dms, DMS-1); α_0 = 1; the start DMS(n, line), §5. | yes | `boostdms.hpp` |
| `boostdms` | BD-2 | The candidates are accepted by simple decrease (the list's filter), as dms; the papers' integer-lattice or sufficient-decrease globalization of the search points is not imposed. | yes | `boostdms.hpp` |
| `boostdms` | BD-3 | "80 % ... in B(x_k; 3Δ_k)": the nearest points of the ball, then the nearest outside it, up to their shares of (n + 1)(n + 2); fewer when the cache holds fewer. | yes | `boostdms.hpp` |
| `boostdms` | BD-4 | A candidate equal to a point of the cache is not evaluated again. | yes | `boostdms.hpp` |
| `boostdms` | BD-5 | Constraints: the extreme barrier, as dms (an infeasible point never enters the list; a start without a feasible point keeps the least violating). | yes | `boostdms.hpp` |
| `chain_ls` | SW-1 | "More than": rho doubles after the 6th success in a row and halves after the 4th failure (Fig. 2's "> 5", "> 3"; | yes | `chain_ls.hpp` |
| `chain_ls` | SW-2 | Every evaluation counts: Fig. 2 counts a step as one evaluation though its second trial is another (Fig. 5 counts both). | yes | `chain_ls.hpp` |
| `chain_ls` | SW-3 | rho is the standard deviation of dif (§3.1: "a normal distribution whose standard deviation is given by a parameter rho"; | yes | `chain_ls.hpp` |
| `chain_ls` | SW-4 | The state is rho, bias and the two counts (M10 §III), and — so that a call may end between the two trials of a step — that step's dif. | yes | `chain_ls.hpp` |
| `chain_ls` | SW-5 | rho_0 = 0.2 (M10) in the coordinates given, a fifth of the width of the normalized box. | yes | `chain_ls.hpp` |
| `chain_ls` | SSW-1 | The subset is kept for subset_evals evaluations (§3.4's text); | yes | `chain_ls.hpp` |
| `chain_ls` | SSW-2 | A new subset is drawn before a step once subset_evals or more evaluations were spent on the current one (numEval moves by 1 or 2 a step, so Fig. 5's "numEval % maxEvalSubSet = 0" may be stepped over). | yes | `chain_ls.hpp` |
| `chain_ls` | SSW-3 | The subset size is floor(0.2 n), at least 1, at most max_vars; dif is 0 off the subset (§3.4: "there is no difference vector"). subset_evals = 0 takes max_evals / 10 of the call (at least 1): a chain is one long run o... | yes | `chain_ls.hpp` |
| `chain_ls` | MTS-1 | After a worse first trial x_i' = x_i - SR the coordinate is RESTORED and the second trial is x_i'' = x_i + 0.5 SR (M11 §3.3's text: "the original solution is restored and then the same variable is increased by 0.5 SR"; | yes | `chain_ls.hpp` |
| `chain_ls` | MTS-2 | M11 §4.4's "minimum 0.4, maximum 1e-15" are swapped: SR below sr_min = 1e-15 is reset to sr_reset = 0.4 (Fig. 3; | yes | `chain_ls.hpp` |
| `chain_ls` | MTS-3 | "Better", "the same", "worse" compare with the current value; the same value restores the point and moves on (Fig. 3). | yes | `chain_ls.hpp` |
| `chain_ls` | MTS-4 | A pass's "Improve" is one flag for the whole pass (Fig. 3's Improve[k], k the individual); the state is SR, the flag, the position in the pass, and the selection and signs of an MTS-LS2 step between its two trials. | yes | `chain_ls.hpp` |
| `chc` | CHC-1 | The bits. Both papers run CHC on bit strings, BLX-0.5 on the decoded parameters; here the points are real and each variable has a b-bit code, round((x - lo) / (hi - lo) (2^b - 1)), for the Hamming distance and the div... | yes | `chc.hpp` |
| `chc` | CHC-2 | Divergence: a flipped bit changes its variable to the decoded value lo + code / (2^b - 1) (hi - lo) of the flipped code; a variable without a flipped bit keeps the best point's value exactly. | yes | `chc.hpp` |
| `chc` | CHC-3 | "the threshold has dropped to zero and there have been several generations without any new offspring accepted": the population is diverged when a generation without acceptance would take the threshold below zero; afte... | yes | `chc.hpp` |
| `chc` | CHC-4 | A generation where no pair may mate makes no point: the threshold drops at once and the pairing is drawn again, inside ask(), until a pair mates or the population is diverged — ask() always returns points. | yes | `chc.hpp` |
| `chc` | CHC-5 | A child is accepted when it is among the M best of parents and children (stably by the comparison of evo.hpp, parents first); | yes | `chc.hpp` |
| `chc` | CHC-6 | BLX-0.5 children outside the box are repaired by the operator, ops::BoundRepair `reflect` by default (real_crossover.hpp). | yes | `chc.hpp` |
| `chc` | CHC-7 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) in the selection; not in the papers. | yes | `chc.hpp` |
| `cmaes` | INJ-1 | The strategy's own parameters stay the tutorial's (Table 1 of the report lists those of [5], Hansen & Kern 2004; the report itself advises keeping the original strategy parameters, p. 7). | yes | `cmaes.hpp` |
| `cmaes` | INJ-2 | Δσ_max = 1.0 as the report's Table 1 has it (its text on p. 5 says the factor exp(0.6)); c_y = sqrt(n) + 2n/(n + 2), c_y^m = sqrt(2n) + 2n/(n + 2) (Table 1). | yes | `cmaes.hpp` |
| `cmaes` | INJ-3 | The tutorial's negative weights: an injected point among the worst is clipped by (3) as well; the report expects no harm (p. 9: the clipping of long steps even helps the negative update). | yes | `cmaes.hpp` |
| `cmaes` | INJ-4 | An injected mean takes the place of the weighted mean of (41) for the mean's move and for p_σ and p_c, the y_i (clipped) still serve the rank-μ update (Figure 1). | yes | `cmaes.hpp` |
| `cmaes` | CMA-1 | t counts the tells; the criteria are tested after the update of iteration t, on the state it left (m, σ, C, p_c of t + 1 in H09's Table 1), each iteration, as a production code tests them before sampling. | yes | `cmaes.hpp` |
| `cmaes` | CMA-2 | Stagnation holds when it holds in both arrays, the best and the median values ("respectively"), and is tested only once the iterations reach the window's length (with fewer, 20 newest and 20 oldest overlap or are not... | yes | `cmaes.hpp` |
| `cmaes` | CMA-3 | TolHistFun and EqualFunVals are tested only once there are as many iterations as their window holds (10 + ⌈30D/λ⌉, D). | yes | `cmaes.hpp` |
| `cmaes` | CMA-4 | "Numerically constant": m_i + δ_i == m_i in double for every i (NoEffectAxis), for some i (NoEffectCoor). | yes | `cmaes.hpp` |
| `cmaes` | CMA-6 | cma_stagnation: h09 (the default) is CMA-2's reading, both arrays; either holds when one of them stagnates; off never stops a run on it. | no: option | `cmaes.hpp` |
| `cmaes` | CMA-5 | The eigendecomposition of C at every iteration by default, where H09 makes it every (c_1 + c_μ)^-1/(10D) iterations (§4): that saves time and changes no update but the sampling's B and D between decompositions. | yes | `cmaes.hpp` |
| `cmaes` | CMA-7 | A lazy run decomposes at the tells t with t mod k = 0; | yes | `cmaes.hpp` |
| `cmaes` | CMM-1 | The distribution functions are normal_cdf.hpp's (Φ, and χ²_ppf with one degree of freedom, χ²_ppf(1 - 2p) = Φ^-1(1 - p)²). | yes | `cmaes.hpp` |
| `cmaes` | CMM-2 | The integer correction (15)-(24) is made only when p_low or p_up is below α/2: otherwise (20)-(23) leave p_low and p_up as they are and (24) returns the same m and A_j, up to rounding (Algorithm 1 makes it every itera... | yes | `cmaes.hpp` |
| `cmaes` | CMM-3 | Where 1 - 2p'' is not positive (p'' >= 1/2: the variance so large that no value is fixed, HM22 after (24)) the coordinate is left as it is. | yes | `cmaes.hpp` |
| `cmaes` | CMM-4 | An injected point x enters with y = A^-1 (x - m) / σ, so that the v of ask() is x itself. | yes | `cmaes.hpp` |
| `continuation` | DC-1 | The problem itself is always the last stage: the stages given lead up to it, and the answer is the problem's (its archive's best point, which may come from an earlier stage). | yes | `continuation.hpp` |
| `continuation` | DC-2 | Stage 1 starts where the inner solver would (its plan, or x0); stage k + 1 from stage k's answer. | yes | `continuation.hpp` |
| `continuation` | DC-3 | The budget: the stages before the last share dc_share of it evenly (the first ones one more, as STRAT-3); the last stage has what is left. | yes | `continuation.hpp` |
| `continuation` | DC-4 | A stage evaluates through its own view (Stage below): a point an earlier stage, or this one, met is answered from the problem's raw residuals kept then (every point of the run comes through a stage), a new one calls t... | yes | `continuation.hpp` |
| `continuation` | DC-5 | A window [a, b) becomes three blocks (a, b - a, n_res - b; empty ones dropped) at weights 0, 1, 0; a weight vector needs the problem's blocks, one weight each, finite and >= 0, and at least one > 0. | yes | `continuation.hpp` |
| `de` | DE-1 | The trial (4) is built coordinate by coordinate: rnbr(i) is drawn first, then one U(0, 1) for every j (j = rnbr(i) included), and v_j is taken where U < CR or j = rnbr(i). | yes | `de.hpp` |
| `de` | DE-2 | Bounds. The paper's box is the initial parameter range (IPR) only; its f9 and f28 have their minima outside it, and DE finds them. | yes | `de.hpp` |
| `de` | DE-3 | Selection: the trial replaces the target when it is not worse — Figure 3's "score <= cost[i]"; | yes | `de.hpp` |
| `de` | DE-4 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp); not in the paper. | yes | `de.hpp` |
| `de` | DE-5 | done() is true once all NP points are equal: every difference vector is zero and no trial can differ from its target. | yes | `de.hpp` |
| `de` | DE-6 | sym (task 19, symmetry.hpp SYM-8; off by default): after the index draws, x_r1, x_r2 and x_r3 aligned to the target (align), or the four canonical (canon: the crossover and the bound repair then take canon(x_i)); no d... | no: option | `de.hpp` |
| `eki` | EKI-1 | (C^pp + Gamma)^-1 v: where J <= m by the Sherman-Morrison-Woodbury identity, gamma^-2 (v - P~ (J gamma^2 I + P~^T P~)^-1 P~^T v), P~ the m x J matrix of the p~^(j), a J x J Cholesky system; where J > m the m x m syste... | yes | `eki.hpp` |
| `eki` | EKI-2 | (22) prints C^pp with p-hat (uncentred); we take the covariance (11) prints, centred, which (37)-(38) use as well. | yes | `eki.hpp` |
| `eki` | EKI-3 | The initial ensemble (their §2.2 leaves it to the user as a draw from a prior): J points of a design in the box (init = sobol: the scrambled Sobol points of designs.hpp; lhs: its Latin hypercube; seed the design's), o... | yes | `eki.hpp` |
| `eki` | EKI-4 | J = max(10, 2 n) by default; gamma = 1 (the residuals taken as divided by the noise's sigma); perturb = true (their p. 7: without the perturbation the results were less accurate); tau = 1.5 (their tau > 1). | yes | `eki.hpp` |
| `eki` | EKI-5 | The answer: the mean of the last ensemble, evaluated once; the best member evaluated on the way is returned beside it. | yes | `eki.hpp` |
| `eki` | EKI-6 | The draws: one std::mt19937 seeded with seed; the initial ensemble around x0 member after member, then eta per iteration member after member, rnd::normal01. | yes | `eki.hpp` |
| `evaluator` | SUCC-1 | Off by default. A point is handed over once: where it is met again it is not evaluated again (the archive), and neither is its verdict asked again. | no: option | `evaluator.hpp` |
| `evaluator` | SUCC-2 | Off by default. The cost is charged after the call, so a call may take the spending past the budget; the run then ends at its next ask (BudgetSpent), as after a gradient's cost. | no: option | `evaluator.hpp` |
| `evaluator` | SUCC-3 | Off by default. In a strategy's start (strategy.hpp, View) the start's own run asks, its View charges the cost to the start's allowance and keeps the verdict, and the run's evaluator takes that verdict when the round... | no: option | `evaluator.hpp` |
| `evaluator` | SUCC-4 | Off by default. An exception of the hook ends the run with its message and the number of the evaluation (UO-1). | no: option | `evaluator.hpp` |
| `evaluator` | CACHE-1 | Off by default. exact: the archive's own rule (a point asked for again is not evaluated), its repeats counted in cache_hits. | no: option | `evaluator.hpp` |
| `evaluator` | CACHE-2 | Off by default. canon: a point whose structure (canon_key_of: the problem's canon_key, else the bytes of canon(x)) was evaluated in this run is not evaluated: it is kept with that structure's values, free, and counted... | no: option | `evaluator.hpp` |
| `evaluator` | CACHE-3 | Off by default. A structure's values are its first point's, in the order the run met them. | no: option | `evaluator.hpp` |
| `evaluator` | CACHE-4 | Off by default. A strategy's start (strategy.hpp, View) reads the run's cache as it was at the round's start and keeps its own; its hits go to the run at the merge, its points as known ones (free). | no: option | `evaluator.hpp` |
| `evaluator` | SO-1 | A problem with residuals takes its objective from them: evaluate, if it is given too, is not called, and such a problem has no constraints (n_cons = 0; the least-squares solvers of this library take none). | yes | `evaluator.hpp` |
| `evaluator` | SO-2 | A gradient asked for at a point not yet evaluated is a joint call where the problem has one (value_grad, residuals_jac), else the values and then the gradient: either way one evaluation and one grad_call. | yes | `evaluator.hpp` |
| `evaluator` | SO-3 | A problem that gives only the joint call (value_grad without evaluate, residuals_jac without residuals) is evaluated by it every time: every evaluation is then a grad_call too. | yes | `evaluator.hpp` |
| `evo` | EVO-1 | A NaN value (f or v) counts as +infinity: worse than every number, equal to another NaN. | yes | `evo.hpp` |
| `ga` | GA-1 | Tournament: k members drawn with replacement, the best by the comparison of evo.hpp, a tie to the later draw; k = 1 is a uniform draw. | yes | `ga.hpp` |
| `ga` | GA-2 | (mu + lambda): parents first, then offspring in the order made, stably sorted by the comparison; the first mu stay, in that order. | yes | `ga.hpp` |
| `ga` | GA-3 | lambda odd: the second child of the last pair is dropped after its mutation (nsga2 does the same). lambda = 1 with `coin`: one child of the pair kept by a fair coin drawn between crossover and mutation, the other neit... | yes | `ga.hpp` |
| `ga` | GA-4 | Bounds: SBX and the polynomial mutation stay in the box by construction; | yes | `ga.hpp` |
| `ga` | GA-5 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) in tournaments, truncation and G3's choices; not in the papers. | yes | `ga.hpp` |
| `ga` | GA-6 | sym (task 19, symmetry.hpp SYM-7; off by default): the second parent aligned to the first (align), or both canonical (canon), after the tournaments and before the crossover; no draw changes, and sym = off is GA as bef... | no: option | `ga.hpp` |
| `ga` | GA-7 | block_uniform (task 19, §19.2.3; off by default): with probability o.block_uniform a pair, after its crossover (which made the free coordinates), every block of sym's groups whole from one parent: a fair coin a block,... | no: option | `ga.hpp` |
| `ga` | GA-8 | block_mutation (§19.2.3; off by default): with probability o.block_mutation a child, after its mutation, one block of all the groups' blocks drawn uniformly and each of its coordinates mutated by the polynomial mutati... | no: option | `ga.hpp` |
| `ga` | GA-9 | The user's genome (task 24, §24.1.2; off by default: the builtin operators): ga_crossover = user calls Problem::crossover(p1, p2, c1, c2, rng) in SBX's place, ga_mutation = user Problem::mutate(c, rng) in the polynomi... | no: option | `ga.hpp` |
| `ga` | GA-10 | Off by default. A row of the user's operators is repaired (ga_repair: builtin clips each coordinate to the box — SBX and the polynomial mutation stay in it, so the builtin run is unchanged; user calls Problem::repair;... | no: option | `ga.hpp` |
| `ga` | GA-11 | Off by default. With Problem::decode every row is decoded before it is evaluated, its canonical genome written back (Lamarckian; checked as GA-10); a row that does not decode (valid = false) is not evaluated, costs no... | no: option | `ga.hpp` |
| `ga` | G3-1 | PCX's index parent x_p is the best of the parents (best_centre, on by default: §3.2.1, "For a faster convergence, the best parent is always used"), not the operator's own pick "with equal probability" (§2.2), which be... | yes | `ga.hpp` |
| `ga` | G3-2 | Step (1)'s mu - 1 others differ from the best and from each other; step (3)'s two differ. | yes | `ga.hpp` |
| `ga` | G3-3 | Step (4) keeps the two best of the family in its order (the two chosen, then the offspring), stably; the better goes to the first chosen place. | yes | `ga.hpp` |
| `hooks` | RES-1 | Off by default (a strategy that asks for a subproblem). | no: option | `hooks.hpp` |
| `improvements` | SO-4 | The violation is judged as the point is evaluated, at the tolerance of that moment (an eq_tol schedule: the tolerance then), and the records are not judged again. | yes | `improvements.hpp` |
| `improvements` | SO-5 | A violation that is not finite (a constraint value NaN) is +inf (EVO-1): such a point is the best only while every point so far is as bad. | yes | `improvements.hpp` |
| `improvements` | SO-6 | A tie (equal f of two feasible points, equal violation of two infeasible ones) is not an improvement: the earlier point stays best. | yes | `improvements.hpp` |
| `jade` | JADE-1 | "the 100p % best": the ceil(p NP) best of P (at least 1); ranking and draws as pbest.hpp (PB-1, PB-2). | yes | `jade.hpp` |
| `jade` | JADE-2 | All NP trials of a generation are built from P and A as they are at its start; the replaced parents join A at its end (§IV-A: "after each generation"), and then A is cut. | yes | `jade.hpp` |
| `jade` | JADE-3 | When no trial succeeds, S_F and S_CR are empty and mu_F, mu_CR keep their values (mean of an empty set is not defined). | yes | `jade.hpp` |
| `jade` | JADE-4 | NP defaults to §V's sizes: 30 for D <= 10, 100 for D <= 30, 400 above; | yes | `jade.hpp` |
| `jade` | JADE-5 | The draws for x_i, in Table I's order: CR_i, F_i, then the trial's (pbest.hpp, PB-1). | yes | `jade.hpp` |
| `jade` | JADE-6 | The box: §V calls f8 "the only bound-constrained function" of its set and gives the others an initial range; here the box is both, and the middle rule of §II holds for every problem. | yes | `jade.hpp` |
| `jade` | JADE-7 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp), for "better" in (5) and in the ranking of JADE-1; not in the paper. | yes | `jade.hpp` |
| `jade` | JADE-8 | done() is true once all of P and A are one point. | yes | `jade.hpp` |
| `lbfgsb` | LB-1 | (5.11) as printed has the sign of B^-1 r^c; d^u = -B^-1 r^c is (5.7)'s, which we follow. | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-2 | The free variables at x^c are those strictly between their bounds there (Algorithm CP's last lines leave the set to be read this way). | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-3 | MN's "not a direction of strong descent": g^T d >= 0. | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-4 | OURS: where even the truncated d is not a descent direction (B lost positive definiteness to rounding), the pairs are dropped and the iteration made again with B = I. | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-5 | OURS: the breakpoints are sorted (BLNZ use a heap: the same order; ties by index); m = 5 (BLNZ's Table 1, first column); a line search that ends without a lower f ends the run (BLNZ's F2); the point of the search is p... | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-7 | M of (3.4) by the Schur complement of -D, T = theta S^T S + L D^-1 L^T (positive definite: s_i^T y_i > 0), by Cholesky (the factorization of Byrd, Nocedal, Schnabel, BLNZ's [6]; | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-8 | The subspace step's middle matrix is (I + theta Mbar Wbar^T A A^T Wbar)^-1 Mbar of (5.22), the form BLNZ used in their experiments (end of §5.3: "identical to" (5.10)'s (I - M W^T Z Z^T W / theta)^-1 M), with Mbar of... | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-9 | x^c = x + t d is projected onto the box (the path is P(x - t g, l, u), (2.2)), and so is (5.9)'s point (feasible by (5.8)): their rounding crossed a bound (Penalty II in [0, 0.5]^4: the largest step along d came out -... | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-10 | Not a departure, a note: where a pair is near (3.9)'s refusal (s^T y the difference of nearly equal terms), theta and the next step follow the last digits of the search's end point, which MT leave to the program (line... | yes | `lbfgsb.hpp` |
| `lbfgsb` | LB-6 | OURS: the new point is the one the search accepts; where it stops short of (2.5)-(2.6) (its own limits), the trial with the lowest f. | yes | `lbfgsb.hpp` |
| `line_search` | LS-1 | Unbracketed, case 4 takes the far end a_t + 4 (a_t - a_l) (MT give no interpolant there without a_u; | yes | `line_search.hpp` |
| `line_search` | LS-6 | Unbracketed, case 3 takes the farther of alpha_c and alpha_s from a_t (the words: the closer, "to be cautious"), alpha_c being the far end where the cubic has no minimizer beyond a_t (the words: alpha_s); then (2.2)'s... | yes | `line_search.hpp` |
| `line_search` | LS-3 | Before the switch to phi, psi stands in for phi in §4's cases only where the trial has a lower phi than a_l but not the sufficient decrease (phi_t <= phi_l, psi_t > 0), phi elsewhere (the words: psi until the switch). | yes | `line_search.hpp` |
| `line_search` | LS-2 | Case 3, the cubic "tends to infinity in the direction of the step" where its cubic coefficient has the sign of a_t - a_l, and its minimizer exists (the discriminant of NW (3.59) is positive). | yes | `line_search.hpp` |
| `line_search` | LS-4 | Also stops where the bracket is narrower than xtol times its right end (xtol = 1e-10 by default: rounding), and after max_nfev trials. | yes | `line_search.hpp` |
| `line_search` | LS-5 | A trial a bracket cannot hold (outside it after rounding) is replaced by the bisection. | yes | `line_search.hpp` |
| `line_search` | LS-7 | A trial where phi or phi' is not finite (overflow) becomes the right end, and the next trial halves the way back to a_l. | yes | `line_search.hpp` |
| `lm` | LM-1 | The column interchanges and the Householder transformations are lsq.hpp's (LH); | yes | `lm.hpp` |
| `lm` | LM-2 | lambda's iteration stops after 10 rounds even short of (5.1) (Moré gives no limit); the first lambda (Moré's "initial estimate alpha_0") is the last iteration's lambda (0 at the start, moved into (l, u) by (5.5)(a));... | yes | `lm.hpp` |
| `lm` | LM-3 | The first Delta is factor \|\|D x_0\|\| (factor where that is 0, factor = 100), and the first iteration's Delta at most \|\|D p\|\| (Moré leaves Delta_0 open). | yes | `lm.hpp` |
| `lm` | LM-4 | ftol = xtol = 1e-8 as in Moré's runs (§8); gtol (a stationarity test Moré's stopping rules do not have) is 0, off; the tests with eps say that the tolerance asked for cannot be reached. | yes | `lm.hpp` |
| `lm` | LM-5 | \|\|.\|\| is the plain Euclidean norm, scaled by the largest \|entry\| (no overflow). | yes | `lm.hpp` |
| `lm` | LM-9 | Residuals that are not finite (an entry NaN or +-inf, or \|\|r\|\| overflowing) count as an infinite cost. | yes | `lm.hpp` |
| `lm` | LMB-1 | Delta's rules and the acceptance (rho > 1e-4) are Moré's of step 3, not CL's Fig. 4.1 (mu = 0.25, gamma = 0.0625, 0.5, 2): so that the method with every bound infinite is Moré's; | yes | `lm.hpp` |
| `lm` | LMB-2 | The step-back theta = max(0.95, 1 - \|\|s\|\|) where x + tau p (x + tau d) is on the boundary (CL (2.13) ask theta in [theta_l, 1], 1 - theta = O(\|\|s\|\|)), theta = 0.95 where rounding still puts a coordinate on the boundar... | yes | `lm.hpp` |
| `lm` | LMB-3 | x_0 is refused outside [l, u] or where l_i >= u_i; within 100 eps max(1, \|bound\|) of a bound it is moved inside as in CL §4: to l_i + 0.1 (u_i - l_i) (u_i - 0.1 (u_i - l_i)), and where the other bound is infinite to l... | yes | `lm.hpp` |
| `lm` | LMB-4 | xtol's \|\|D x\|\| and the first Delta take Moré's D alone; gtol's cosines are those of [J; | yes | `lm.hpp` |
| `lm` | LMB-5 | lm_geodesic is refused with a finite bound (Transtrum & Sethna have none). | yes | `lm.hpp` |
| `lm` | LMG-1 | lambda is chosen for v alone, as they do ("selecting lambda so that \|delta theta_1\| = Delta"); the predicted reduction and Delta's rules are v's, the actual reduction that of v + a. | yes | `lm.hpp` |
| `lm` | LMG-2 | Where (15) fails, Delta = 1/2 min(Delta, \|\|D v\|\|) and the step is made again, v + a not evaluated. | yes | `lm.hpp` |
| `lm` | LMG-3 | The norms of (15) are the scaled \|\|D .\|\|, so that the method stays invariant to the scale of the parameters, as they say it does. | yes | `lm.hpp` |
| `lm` | LM-6 | OURS: irls_tol = 1e-10, 50 rounds at most, every round to the tolerances above, the evaluations of all rounds counted together. | yes | `lm.hpp` |
| `lm` | VP-1 | The pseudorank of Phi: the diagonal \|R_jj\| > max(m, p) eps \|R_00\| (lsq.hpp's LSQ-6); a = Phi^- y is the basic solution on those columns. | yes | `lm.hpp` |
| `lm` | VP-2 | Without the derivatives of Phi, the Jacobian of r_2 is forward differences of r_2 (the full projection, not Kaufman's term). | yes | `lm.hpp` |
| `lm` | VP-3 | The Jacobian's evaluation of Phi and its derivatives counts as one Jacobian (njev), r_2's as one evaluation (nfev). | yes | `lm.hpp` |
| `lm` | LMN-1 | The metric is Moré's D: the step is orthogonal to N in D^2 (N^T D^2 p = 0), plainly orthogonal where scale = false; | yes | `lm.hpp` |
| `lm` | LMN-2 | The diagnostics on the complement (diagnose_gauge) take the plain inner product: the covariance of J Q2's parameters, carried back. | yes | `lm.hpp` |
| `lm` | LMN-3 | A gauge basis takes no finite bound and no lm_geodesic. | yes | `lm.hpp` |
| `lm` | LMN-4 | Result::gauge_k is N's pseudorank at the solution, gauge_ratio \|\|J Q_N\|\| / \|\|J\|\| (Frobenius, Q_N an orthonormal basis of N's columns), with a Jacobian at the solution (one more, counted). | yes | `lm.hpp` |
| `lm` | LMC-1 | The multipliers' sign: the task's form r~ = [r; sqrt(mu) (c + lambda/mu)] is N&W's with lambda = -lambda_N&W; at a solution J^T r + A^T lambda = 0, A the Jacobian of c. | yes | `lm.hpp` |
| `lm` | LMC-2 | The subproblem is solved by lm to its own tests (ftol, xtol); where the projected gradient is still above omega_k s, lm again from there with ftol and xtol a hundredth (not below eps), three times at most. | yes | `lm.hpp` |
| `lm` | LMC-3 | omega_k and omega_* are relative, times s = max(1, \|\|J^T r\|\|) at x_0, so that the method does not hang on the residuals' scale; eta_* = 1e-8 (on \|\|c\|\|, the equalities' own scale) and omega_* = 1e-6 by default; 50 oute... | yes | `lm.hpp` |
| `lm` | LMC-4 | Exact elimination only without a finite bound (the owner's decision H2 of 2026-11-01): a box in x is no box in y. | yes | `lm.hpp` |
| `lm` | LMC-5 | After elimination the multipliers are the least-squares solution of C^T lambda = -J^T r at x (one more Jacobian). | yes | `lm.hpp` |
| `lm` | LMP-1 | chi^2 = sum (r_i / sigma_i)^2 = 2 cost where the residuals are divided by sigma: the threshold on the cost is Delta_alpha / 2, times sigma2 where the caller gives the residuals' variance (the residuals not divided); w... | yes | `lm.hpp` |
| `lm` | LMP-2 | The automatic grid: steps of h = sqrt(2 threshold C_jj) / 10, C = (J^T J)^+ at x_hat (the quadratic model's crossing in ten steps), or 0.1 max(1, \|x_hat_j\|) where C_jj = 0; max_steps nodes a side at most; a side ends... | yes | `lm.hpp` |
| `lm` | LMP-3 | Flat: every node's cost within flat_tol max(1, cost_hat) of cost_hat (flat_tol = 1e-8). | yes | `lm.hpp` |
| `lm` | LMP-4 | chi2_quantile: the regularized lower incomplete gamma P(df/2, x/2) by its power series, the definition's integrand e^-t t^(a-1) expanded and integrated term by term: P(a, z) = z^a e^-z / Gamma(a + 1) sum_k z^k / ((a +... | yes | `lm.hpp` |
| `lm_ma_es` | LMMA-1 | c_sigma = 2 lambda / n is meant for large n (L18's experiments: n = 128 ... 8192); at 2 lambda >= n it is 1 or more and line 13 breaks down (n = 10: c_sigma = 2, sqrt(mu_w c (2 - c)) = 0). | yes | `lm_ma_es.hpp` |
| `lm_ma_es` | LMMA-2 | Algorithm 1 leaves the stopping criterion open ("until stopping criterion is met"): here the budget, f <= f_target, sigma below tol_sigma times its start, and max_iter iterations. | yes | `lm_ma_es.hpp` |
| `lm_ma_es` | LMMA-3 | The ranking is a stable sort of f (ties in sampling order); a NaN ranks last. | yes | `lm_ma_es.hpp` |
| `lm_ma_es` | LMMA-4 | The best point evaluated is the answer (the mean y is not evaluated, as in Algorithm 1). | yes | `lm_ma_es.hpp` |
| `lshade` | LSHADE-1 | "the top N p members": the ceil(N p) best, at least 2 (SHADE's p_min = 2/N); ranking and the trial's draws as pbest.hpp (PB-1, PB-2). | yes | `lshade.hpp` |
| `lshade` | LSHADE-2 | The draws for x_i, in Algorithm 2's order: r_i, CR_i (none when M_CR,r_i = ⊥), F_i, then the trial's. | yes | `lshade.hpp` |
| `lshade` | LSHADE-3 | Algorithm 2, line 25 reads "if N_G < N_G+1"; the text says "whenever N_G+1 < N_G, the worst-ranking individuals are deleted" — taken. | yes | `lshade.hpp` |
| `lshade` | LSHADE-4 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) for "<=", "<", the ranking and the deletion; the weight of a success as SHADE-3 (shade.hpp). | yes | `lshade.hpp` |
| `lshade` | LSHADE-5 | All N trials of a generation are built from P and A at its start (Algorithm 2 builds them all before any selection). | yes | `lshade.hpp` |
| `lshade` | LSHADE-6 | done() is true once all of P and A are one point. | yes | `lshade.hpp` |
| `lsq` | LSQ-1 | NNLS solves the least squares problem of its Step 6 afresh every time (LH's subroutine updates a QR decomposition, Ch. 24): the problems here have tens of columns, and the plain form is easier to check. | yes | `lsq.hpp` |
| `lsq` | LSQ-2 | A candidate column of NNLS (Step 5) is refused when its new diagonal element is negligible beside the part of the column the columns already in the set explain (\|r\| * 0.01 adds nothing to that part's norm, the test LH... | yes | `lsq.hpp` |
| `lsq` | LSQ-3 | LDP finds G x >= h incompatible when \|\|r\|\| (Step 3's \|\|r\|\| = 0) is within the round-off of r itself: \|\|r\|\| <= 100 eps (1 + sum_ij \|E_ij\| u_j), the bound of the error of r = E u - f summed term by term. \|\|r\|\| is the ro... | yes | `lsq.hpp` |
| `lsq` | LSQ-8 | LDP is solved for the same constraints with every row g_i of G scaled to unit length and h to max \|h_i\| = 1: G x >= h and (g_i / \|\|g_i\|\|)^T x >= h_i / \|\|g_i\|\| are one set, and LDP's solution is linear in h (x(c h) = c... | yes | `lsq.hpp` |
| `lsq` | LSQ-4 | lsei picks the rows of C in the order of the largest remaining norm (the column interchanges of HFTI (14.9), Steps 3-10, on C^T) and takes as its pseudorank the rows whose diagonal element exceeds 10 eps max\|C\| (Step... | yes | `lsq.hpp` |
| `lsq` | LSQ-5 | A Householder transformation whose part to be zeroed is zero already is the identity (H1 would flip the pivot's sign; the identity keeps triangular input exact). | yes | `lsq.hpp` |
| `lsq` | LSQ-7 | The sum of H2's Step 8 is taken in four running sums (terms l, l + 4, ... in the first, and so on), then (s0 + s1) + (s2 + s3): the order is the code's, the same everywhere, and four times shorter a chain of additions... | yes | `lsq.hpp` |
| `lsq` | LSQ-6 | HFTI's pseudorank: the diagonal elements above max(m, n) eps \|r_11\| (Step 13 leaves tau to the caller; this is NumPy's default rcond for lstsq, measured on R's diagonal instead of the singular values). | yes | `lsq.hpp` |
| `ma_chains` | MAC-1 | An empty S_LS restarts the population at random keeping the best (CEC 2010 and M11; | yes | `ma_chains.hpp` |
| `ma_chains` | MAC-2 | BGA mutates one gene, chosen uniformly ("the gene to be mutated", M10 §4.4.4); the first parent and the n_ass candidates are uniform (M10 §4.4.3: "in our experiments all the parents are selected at random"). | yes | `ma_chains.hpp` |
| `ma_chains` | MAC-3 | The first value of a chain's step size (M11 §4.2, §4.4: half the distance to the nearest member) is taken in the normalized box, at least 1e-12; a member without a chain (never searched, or a new child) starts one. | yes | `ma_chains.hpp` |
| `ma_chains` | MAC-4 | n_freq rounds to the nearest whole number, at least 1; the GA's share is counted in evaluations spent (a repeated point is free). | yes | `ma_chains.hpp` |
| `ma_chains` | MAC-5 | A child or a restarted member starts without a chain; a member replaced by a local search's result keeps the chain of that search. | yes | `ma_chains.hpp` |
| `ma_chains` | MAC-7 | A GA phase stops early after 100 n_freq children in a row that spend nothing (all repeats), and a phase (GA and local search) that spends nothing ends the run (as SRCH-9, SRCH-10). | yes | `ma_chains.hpp` |
| `ma_chains` | MAC-6 | The strategy's draws come from mt19937(start_seed(seed, 0x4D41434Cu)), the coins of lamarck = share from start_seed(seed, 1 << 32) as memetic's (SRCH-12); ma_local = inner's s-th search has the seed start_seed(seed, s). | yes | `ma_chains.hpp` |
| `mde` | MDE-1 | CR in the classical sense (§24.9 item 17): the probability that a coordinate is the mutant's, so mde_CR = 1 is LMS14's CR = 0 (their test "U < CR: keep x_i's coordinate", no crossover) and CL16's CR = 1. | yes | `mde.hpp` |
| `mde` | MDE-2 | k1, k2, k3 all different and different from i (LMS14 Algorithm 2; its §3 allows repeats "without significant change"): mde needs mde_pop >= 4, the others >= 2. | yes | `mde.hpp` |
| `mde` | MDE-3 | mde_match relabels at the start of every sweep each member to the best one (the record): align — the problem's align, else symmetry::align on sym_blocks and canon; greedy — LMS14 §4, Algorithm 4: within each group of... | yes | `mde.hpp` |
| `mde` | MDE-4 | The local search of the s-th search has the seed start_seed(seed, s); the strategy's own draws (the indices, the coins) come from mt19937(start_seed(seed, 0x4D444531)). | yes | `mde.hpp` |
| `mde` | MDE-5 | "The population does not change": the members' values (f, cv) are the same as at the sweep's start (LMS14 §3: "the objective values"). | yes | `mde.hpp` |
| `mde` | MDE-6 | A trial is clipped to the box (wrapped on a periodic coordinate). | yes | `mde.hpp` |
| `mosqp` | MS-1 | τ = 1e-8 (mosqp_tau), μ = 1e-4 (mosqp_mu, Armijo), β = 1/2 (mosqp_beta); n̄ = pop_size. | yes | `mosqp.hpp` |
| `mosqp` | MS-2 | H_i = I in both stages: the first of §6's three cases (the others need second derivatives the problems here do not give). | yes | `mosqp.hpp` |
| `mosqp` | MS-3 | Gradients and Jacobians by forward differences (fd_step in the box normalized to [0, 1], as sqp in solvers.hpp), n evaluations a point. | yes | `mosqp.hpp` |
| `mosqp` | MS-4 | σ: the first of 1, 10, 100, ... that makes the merit's directional derivative ∇fᵀd − σ‖c(x)‖₁ negative (Algorithm 4.1, 3(a)iD), at most 10⁸. | yes | `mosqp.hpp` |
| `mosqp` | MS-5 | The line of stage 2: x_l = ℓ + l(u − ℓ)/(n̄ + 1), l = 1..n̄. | yes | `mosqp.hpp` |
| `mosqp` | MS-6 | With H_i = I, (8) is the least-squares problem min ‖d + ∇f_i‖ under the linear rows, and (9), which Lemma 3.2 (item 6) writes as min Σ_i (∇f_iᵀv + ½vᵀH_i v) under the same rows, is min ‖√m v + Σ_i ∇f_i / √m‖: one lsq:... | yes | `mosqp.hpp` |
| `mosqp` | MS-7 | (10) prints (u_j − ℓ_j)/2 for a finite box, which lies outside [ℓ_j, u_j] whenever ℓ_j > (u_j − ℓ_j)/2; read as the centre (u_j + ℓ_j)/2. | yes | `mosqp.hpp` |
| `nelder_mead` | NM-1 | The initial simplex: x_0 and x_0 + tau e_i, tau = 0.05 in every coordinate, x_0 - tau e_i where x_0 + tau e_i would leave the box (the box is meant normalized to [0, 1]^n: Gao & Han's (3.1), fminsearch's, takes 0.0002... | yes | `nelder_mead.hpp` |
| `nelder_mead` | NM-2 | Every point the method makes is clipped to the box before it is evaluated and becomes a vertex (Lagarias et al. and Gao & Han treat unconstrained problems). | yes | `nelder_mead.hpp` |
| `nelder_mead` | NM-3 | The initial vertices are ordered by value, ties in their order of construction (x_0 first). | yes | `nelder_mead.hpp` |
| `normal_cdf` | NCDF-1 | The series and the continued fraction meet at \|x\| = 2 (z = sqrt 2): there both converge in about 40 terms; the lower tail keeps its relative accuracy (erfc is never 1 - erf there). | yes | `normal_cdf.hpp` |
| `normal_cdf` | NCDF-2 | The inverse takes the tail it is in (p > 1/2: -Phi^-1(1 - p)), so that p near 1 is not lost to 1 - p's rounding more than the input already is. | yes | `normal_cdf.hpp` |
| `pap` | PAP-1 | The constituents run generation by generation in lockstep, in the order given: a generation of each is one of its ask/tell steps, so the evaluations go to each in proportion to what its generation asks (its subpopulat... | yes | `pap.hpp` |
| `pap` | PAP-2 | A migration takes the emigrants of every constituent from the subpopulations as they stand before it ("combine the subpopulations of the remaining m - 1 algorithms, identify and duplicate the best migration_size indiv... | yes | `pap.hpp` |
| `pap` | PAP-3 | PSO's "gbest": the migrant replaces the worst particle's best and position, its velocity 0, and gbest is the best of the bests again (§II-C: "updating the gbest once a better solution emerges in the whole population")... | yes | `pap.hpp` |
| `pap` | PAP-4 | CMA-ES's "covariance matrix updated after the migration": the migrant is injected (cmaes.hpp, inject; task 19, §19.4.3) into its next generation, which updates the distribution with it; it is evaluated again, free (th... | yes | `pap.hpp` |
| `pap` | PAP-5 | A constituent that ends (its done(): a collapsed population) leaves its time to the others: they go on until the budget is spent, in proportion to their generations as before (§II-C: "in proportion to the sizes of the... | yes | `pap.hpp` |
| `pap` | PAP-6 | Constituent k's generator is start_seed(seed, k) (strategy.hpp STRAT-1): one constituent, no migration, is that method's own run, bit for bit. lshade's MAX_NFE is its share of the budget, np / sum(np). | yes | `pap.hpp` |
| `pap` | PAP-8 | sym = off \| align (task 19, §19.2.2; off by default): the members' own sym (align; symmetry.hpp SYM-7 ... 9), and a migrant into CMA-ES aligned to its mean before it is injected (§19.2.2's "injected points to the curr... | no: option | `pap.hpp` |
| `pap` | PAP-7 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) for the emigrants and the migrants; | yes | `pap.hpp` |
| `pbest` | PB-1 | The trial's draws: x_pbest, r1, r2, j_rand, then one U(0, 1) a coordinate, j_rand's included, v_j taken where U < CR or j = j_rand (de.hpp draws the same way). r1 and r2 are drawn again until they differ as asked; r2... | yes | `pbest.hpp` |
| `pbest` | PB-2 | "The best" of P: ranked by the comparison of evo.hpp, ties by index; x_pbest is drawn from the first `top`, and may be x_i or x_r1. | yes | `pbest.hpp` |
| `pbest` | PB-3 | sym (task 19, symmetry.hpp SYM-8; off by default): after the draws, x_pbest, x_r1 and x~_r2 aligned to x_i (align), or the four canonical (canon); no draw changes, and sym = off is the trial as before. | no: option | `pbest.hpp` |
| `pbh` | PBH-1 | Step 2 against the worst compares f(Y_p) (the preprint prints f(Y_q)). | yes | `pbh.hpp` |
| `pbh` | PBH-2 | "Near" is d < d_cut, as the preprint (the journal's text has <=). | yes | `pbh.hpp` |
| `pbh` | PBH-3 | The children are made from the population at the iteration's start (their local searches one round of K starts) and placed one after the other in the members' order. | yes | `pbh.hpp` |
| `pbh` | PBH-4 | The draws: per child its n coordinates' moves and then one more uniform, as ils draws its acceptance's u at every step, so that K = 1 follows ils's stream; the extra members' uniform points come from the same stream b... | yes | `pbh.hpp` |
| `pbh` | PBH-5 | Ties: the nearest and the worst are the first in the members' order; | yes | `pbh.hpp` |
| `pbh` | PBH-6 | pbh_distance = user needs the problem's distance hook (task 24, §24.1.3) and refuses a problem without one; perturb = user the problem's perturb hook, its result clipped or wrapped as SRCH-19 (the draws of PBH-4 then... | yes | `pbh.hpp` |
| `periodic` | PER-1 | Off by default. The wrap is lo + fmod(v - lo, hi - lo), negatives raised by a period; a result that rounds to hi is lo (no point on the seam). | no: option | `periodic.hpp` |
| `periodic` | PER-2 | Off by default. A wrapped coordinate is not a repair (RepairCounts, task 20, §20.2.2, does not count it) and costs no penalty (SCALAR-7, POP-4): the point is inside the problem's domain. | no: option | `periodic.hpp` |
| `periodic` | PER-3 | Off by default. A difference d of a periodic coordinate is min(\|d\|, period - \|d\|). | no: option | `periodic.hpp` |
| `polish` | POLISH-1 | The objective improved goes round the objectives (the first point f_1, the second f_2, ...) in DSS order; the objectives are normalized by the range of the points chosen. | yes | `polish.hpp` |
| `polish` | POLISH-2 | The answer after polishing is the non-dominated union of the algorithm's answer and the polished points, not the archive's; cut to pop_size, it keeps a point weakly dominating each point of the algorithm's answer firs... | yes | `polish.hpp` |
| `polish` | POLISH-3 | The subproblem's constraints f_i <= f_i(a) hold to a slack (SLACK, in the normalized objectives): a result a hair worse than a in one objective does not dominate a, and both stay. | yes | `polish.hpp` |
| `population` | POP-1 | The budget can end inside a generation: the points evaluated so far are in the archive (and may be the answer), the method is not told the rest. | yes | `population.hpp` |
| `population` | POP-2 | A point the archive has already (a repeated point) is not evaluated again and costs nothing (evaluator.hpp, THE BUDGET); the method is told its value as for any other. | yes | `population.hpp` |
| `population` | POP-3 | A failed evaluation (evaluator.hpp: a value not finite) is told as NaN, f and v, which evo.hpp counts as +infinity (EVO-1). | yes | `population.hpp` |
| `population` | POP-4 | sep_cmaes works in the box normalized to [0, 1]^n, as cmaes does in run_single (SINGLE-2), from the mean y0 with step cma_sigma0; a point outside the box is evaluated at its clip and told the clip's f plus the squared... | yes | `population.hpp` |
| `pso` | PSO-1 | §VI computes chi "as shown in (4.15)"; (4.15) is the modulus of the eigenvalues, sqrt(chi). | yes | `pso.hpp` |
| `pso` | PSO-2 | Synchronous: the whole swarm is evaluated, then every p_i and p_g are updated, then every particle moves (the pseudocode updates p_g particle by particle). | yes | `pso.hpp` |
| `pso` | PSO-3 | "Initialize population: random x_i, v_i": v_i uniform in [-(hi - lo)/2, (hi - lo)/2] coordinate by coordinate, drawn after all positions; `v0_zero` starts at 0 (SMPSO-5 of smpso.hpp). | yes | `pso.hpp` |
| `pso` | PSO-4 | "p_g = min(p_neighbors)": §VII names no topology. | yes | `pso.hpp` |
| `pso` | PSO-5 | Draws: particle by particle, phi_1 then phi_2 for every d. | yes | `pso.hpp` |
| `pso` | PSO-6 | Bounds: the paper's box is the initial range; a position outside the box is repaired by ops::repair_value (reflect by default; `native` leaves it — the paper's), the velocity kept. | yes | `pso.hpp` |
| `pso` | PSO-7 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) for "f(x_i) < f(p_i)" and for p_g; not in the papers. | yes | `pso.hpp` |
| `pso` | PSO-8 | done() is true once every velocity is 0 and every x_i = p_i = p_g. | yes | `pso.hpp` |
| `pso` | PSO-9 | sym (task 19, symmetry.hpp SYM-9; off by default): align only — p_i and p_g aligned to x_i before the particle moves, the velocity left in the particle's representation; canon refused. | no: option | `pso.hpp` |
| `restarts` | RST-1 | One set of termination criteria for both, H09's (§2.2): AH05's are the same in substance (equalfunvalhist, TolFun, TolX, noeffectaxis, noeffectcoord, conditioncov) but for TolFun over the last generation's values; | yes | `restarts.hpp` |
| `restarts` | RST-2 | U[0,1]² of λ_s is the square of one uniform number; the second draw (σ⁰) is another (H09: "independent uniformly distributed numbers"). | yes | `restarts.hpp` |
| `restarts` | RST-3 | "Half of the recent large budget": half of the evaluations of the last run of regime 1. | yes | `restarts.hpp` |
| `restarts` | RST-4 | The regimes' budgets are the evaluations their runs spent; the first run (λ_def) counts in neither (H09: "the first single run ... is disregarded"). | yes | `restarts.hpp` |
| `restarts` | RST-5 | "The first and last restart are conducted under the first regime": after regime 1's last run (λ = 2^bipop_large λ_def) the strategy stops. | yes | `restarts.hpp` |
| `restarts` | RST-6 | The box gives the start: a bound that is not finite is refused, and so are constraints (bounds only). | yes | `restarts.hpp` |
| `restarts` | RST-7 | A run's generator and its start mean come from start_seed(seed, k) (strategy.hpp, STRAT-1), k the run's number from 0; bipop draws λ_s and σ⁰ of a run of regime 2 from it before the start mean. | yes | `restarts.hpp` |
| `restarts` | RST-8 | A run that evaluates nothing ends the strategy (ipop, bipop and bipop_2009): the budget, or the run's target (evaluator.hpp, THE TARGET), is spent, whatever regime 2's own allowance says, or every point the run asks f... | yes | `restarts.hpp` |
| `restarts` | RST-9 | With warm_start = on (task 24, §24.4.1; off by default: x0 is ignored and reported, as before) x0 is the start mean of the first run only; the restarts draw theirs as before (the decision of 13.11: INJ-3 stays). | no: option | `restarts.hpp` |
| `rgn` | RGN-1 | Step 1 evaluates r at theta_i again every iteration; here theta_i and its residuals come from the line search or the difference points, so no evaluation is repeated (the library's evaluator does not count a point twic... | yes | `rgn.hpp` |
| `rgn` | RGN-2 | Every line search trial is an evaluation (the pseudocode adds 1 to N_Phi only for the accepted one). | yes | `rgn.hpp` |
| `rgn` | RGN-3 | The expected reduction of B2, delta^T g + 1/2 delta^T H delta, is taken for the projected step (Appendix A writes it for delta_rdc before the projection of step 7); | yes | `rgn.hpp` |
| `rgn` | RGN-4 | "The SVD" of the symmetric positive semidefinite H_free is its eigendecomposition (eigen.hpp): W the eigenvalues; those at or below the threshold, the slightly negative ones of round-off included, are dropped. | yes | `rgn.hpp` |
| `rgn` | RGN-5 | A budget (max_evals, the library's) may end an iteration midway: the run then returns the best point it has evaluated (the current one or a difference point). | yes | `rgn.hpp` |
| `rgn` | RGN-6 | A residual that is not finite gives Phi = +inf, worse than any finite value; a Jacobian column with a nonfinite residual is set to 0. | yes | `rgn.hpp` |
| `rgn` | RGN-7 | The box must be finite (h_max is a fraction of its width) and contain the start. | yes | `rgn.hpp` |
| `rgn` | RGN-8 | theta_scale = 10 and Phi_scale = 1 are Q18's absolute scales: in a problem whose parameters are of order 1, theta changes below 1e-4 count as "no change" (§4.4, criterion 3). | yes | `rgn.hpp` |
| `rgn` | RGN-9 | Free / locked / releasable as B1-B3; B3's "comparable gradient" tests use \|g_k\| max(\|theta_k\|, theta_scale), as written. | yes | `rgn.hpp` |
| `scalar` | SCALAR-1 | n_sub from the budget: what is left after the plan's start and the anchors, over 10 (n + 1) evaluations a subproblem for sqp — about ten iterations from a neighbour's solution — and 100 n for nm and cmaes; at most pop... | yes | `scalar.hpp` |
| `scalar` | SCALAR-2 | nm, cmaes: an infeasible point is worth the worst feasible value known for its subproblem plus its violation (Deb's rule as a penalty). | yes | `scalar.hpp` |
| `scalar` | SCALAR-3 | tch: w_i = 1 / max(d_i, 1e-6) for the lattice's direction d, so that the solutions lie on the rays F̄ ∝ d the lattice spreads evenly; the reference point is z*. | yes | `scalar.hpp` |
| `scalar` | SCALAR-4 | nnc: the normal constraints (Ā_m − Ā_i)ᵀ(F̄ − X̄) <= 0 and min F̄_m, Problem P3 of the 2003 paper (Eqs. 36-41). | yes | `scalar.hpp` |
| `scalar` | SCALAR-5 | ps: ā on the anchors' simplex, r its quasi-normal reversed (NBI's geometry); the 1984 paper leaves (p, q) to the user. | yes | `scalar.hpp` |
| `scalar` | SCALAR-6 | sqp stops a subproblem at a failed evaluation of an iterate, and at a finite-difference step that fails both ways. | yes | `scalar.hpp` |
| `scalar` | SCALAR-7 | cmaes: a point outside the box is evaluated at its clip, plus the squared distance to it times (1 + \|value\|); one run per subproblem, until σ·max D < local_tol or the allowance is spent. | yes | `scalar.hpp` |
| `scalar` | SCALAR-8 | sqp asks g(x) <= -1e-10 of the problem's inequalities, so that a solution on the boundary is feasible by the vault's rule (g <= 0). | yes | `scalar.hpp` |
| `scalar` | SCALAR-9 | The anchors are lexicographic minima in cyclic order: ours. | yes | `scalar.hpp` |
| `scalar` | SCALAR-10 | laumanns: the grid's first bound is +∞ (no constraint), the mirror of the paper's lower bound 0 for maximization. | yes | `scalar.hpp` |
| `scalar` | SCALAR-11 | A subproblem whose solver ends where it started runs again, with what is left of its allowance, from the initial plan's best point for it: the archive's best is often an anchor or a neighbour's solution at a corner of... | yes | `scalar.hpp` |
| `scalar` | SCALAR-12 | The anchors' stages see the objectives scaled by the initial plan's range (its minimum to 0, its maximum to 1): SQP stops early on a raw objective of 1e7 (bbob-biobj f2's ellipsoid). | yes | `scalar.hpp` |
| `scalar` | SCALAR-14 | Result::repairs counts a point when a method makes it and it is looked at against the box, as Kononova, Caraffini and Bäck count (Information Sciences 581, 2021; task 20, section 20.2.2) — de's, jade's, shade's and ls... | yes | `scalar.hpp` |
| `scalar` | SCALAR-15 | A campaign's trajectory rows carry the counts of the box at their fe on any number of threads: a start in a thread defers its count with its own counter's delta, and the round replays both in start order, so the rows... | yes | `scalar.hpp` |
| `scalar` | SINGLE-1 | One run of the solver: one that stops before the budget leaves the rest unspent, as DMS does. | yes | `scalar.hpp` |
| `scalar` | SINGLE-2 | The solver sees f scaled by the initial plan's range (its least value to 0, its greatest to 1), as scalar's anchors do (SCALAR-12), in the box normalized to [0, 1]^n, as every solver of solvers.hpp: a bound that is no... | yes | `scalar.hpp` |
| `scalar` | SINGLE-3 | A start x0 (the problem's coordinates) is the plan: one point, and the solver from it; the frame of SINGLE-2 is then f(x0) to 0 with scale 1 (subproblems.hpp: a range of 0 is a scale of 1). slsqp takes the problem's g... | yes | `scalar.hpp` |
| `scalar` | SINGLE-5 | keep_points = best (task 24, §24.3.3; off by default: all): the evaluator's lean mode (the best point and the last); the plan's range of SINGLE-2 is taken from its values as they come, its best is kept in the best's s... | no: option | `scalar.hpp` |
| `scalar` | LOCAL-1 | Bounds only: a problem with constraints is refused, and lm refuses a problem without residuals (its objective is 1/2 \|\|r~\|\|^2). | yes | `scalar.hpp` |
| `scalar` | LOCAL-2 | Neither solver's own limit on evaluations (lm's max_nfev, lbfgsb's max_nfev and max_iter) is put: the budget stops them. | yes | `scalar.hpp` |
| `scalar` | LOCAL-3 | A failed evaluation ends the run, as sqp's (SCALAR-6): neither method can step over a hole in the function. | yes | `scalar.hpp` |
| `scalar` | LOCAL-4 | Without x0, a bound that is not finite is refused: there is no box to draw the plan in, and no frame is put on x (addendum 1, §3.1). | yes | `scalar.hpp` |
| `scalar` | LOCAL-5 | eki starts as lm does, from x0 or the plan's best point: its initial ensemble around that point by default (eki_init = around, EKI-3), or a design in the box (sobol, lhs; finite bounds). | yes | `scalar.hpp` |
| `scalar` | LOCAL-7 | keep_points = best (task 24, §24.3.3; off by default: all): the evaluator's lean mode (the best point and the last); the start is the plan's best, kept in the best's slot; a point asked for again is evaluated again an... | no: option | `scalar.hpp` |
| `scalar` | LOCAL-6 | lm_gauge = on needs the problem's gauge_basis; its k and \|\|J N\|\| / \|\|J\|\| at lm's solution (lm.hpp LMN-4) go to the settings (lm_gauge_k, lm_gauge_ratio). | yes | `scalar.hpp` |
| `scalar` | ULS-1 | Off by default. The hook's calls of the problem are counted against the budget as every solver's (and a solver it runs within, on ctx.problem, is that solver's run, bit for bit); past the budget the call throws and th... | no: option | `scalar.hpp` |
| `scalar` | ULS-2 | Off by default. x' with f known (has_f) is planted, not evaluated: free, as a seed point; with constraints or residuals it is evaluated all the same (its constraint values, its residuals are not known). | no: option | `scalar.hpp` |
| `scalar` | ULS-3 | Off by default. The declared cost is charged to the budget after the call (as THE SUCCESS's, SUCC-2); a hook that neither calls the problem nor declares a cost spends nothing (the owner's decision of 13.11), counted a... | no: option | `scalar.hpp` |
| `scalar` | ULS-4 | Off by default. x' of the wrong length or not finite, or an exception of the hook (other than the budget's end), ends the run with the message (UO-1). | no: option | `scalar.hpp` |
| `scalar` | ULS-5 | Off by default. In a strategy's start the counts (calls, cost, time) are the start's and go to the run's when the round merges the start, in the order of the starts. | no: option | `scalar.hpp` |
| `scalar` | ULS-6 | Off by default. The run's answer is the evaluator's best, as every method's; the hook's state comes back in Result::ls_state. | no: option | `scalar.hpp` |
| `scalar` | POP-5 | Every bound must be finite: the methods draw their first population uniformly in the box (sep_cmaes its mean), and the papers' boxes are where they search. | yes | `scalar.hpp` |
| `scalar` | POP-6 | The population is the method's own (np), not pop_size: the papers' sizes are part of the methods (lshade's N_init = 18 D, pso's 20). pop_size still sets the observer's stride (evaluator.hpp). | yes | `scalar.hpp` |
| `scalar` | POP-7 | The first population is the method's draw. | yes | `scalar.hpp` |
| `scalar` | POP-10 | keep_points = best (task 24, §24.3.3; off by default: all) keeps the best point and the last (evaluator.hpp THE LEAN MODE): a point asked for again is evaluated again and counted (POP-2 does not hold), so a method tha... | no: option | `scalar.hpp` |
| `scalar` | POP-8 | lshade's MAX_NFE is the run's budget unless lshade_max_nfe is set: §III's MAX_NFE is the budget the run is given. | yes | `scalar.hpp` |
| `scalar` | POP-9 | sym = off \| canon \| align (task 19, §19.2.2; symmetry.hpp SYM-7 ... | yes | `scalar.hpp` |
| `search` | SRCH-1 | MLSL's r_k: part II's (2) prints it without Γ; it refers to part I's Single Linkage analysis, whose (35) has Γ(1 + n/2) (the volume of the unit ball): taken with Γ. m(S) = 1, distances in the box normalized to [0, 1]^n. | yes | `search.hpp` |
| `search` | SRCH-2 | MLSL as part II's experiments (§4): τ = ν = 0 by default (no boundary set Q_τ, no neighbourhood X*_ν of the minima found), σ = 4, γ = 0.2. | yes | `search.hpp` |
| `search` | SRCH-3 | MLSL: the local searches of an iteration are one round; a point that started one never starts another; the sample is the uniform points only (a local search's points are in the archive, not in the sample). | yes | `search.hpp` |
| `search` | SRCH-4 | A perturbed point is clipped to the box (ils, basin_hopping). | yes | `search.hpp` |
| `search` | SRCH-5 | basin_hopping's step: adjusted every bh_interval (10) steps, divided by 0.9 when the acceptance ratio was above bh_accept_rate, multiplied by 0.9 otherwise (the paper gives the target, not the rule). | yes | `search.hpp` |
| `search` | SRCH-6 | halving: a round reruns each arm with its whole allowance so far; its earlier points come from the archive, free, and only what goes beyond costs (a solver whose run does not depend on the budget repeats itself exactl... | yes | `search.hpp` |
| `search` | SRCH-7 | memetic: the global solver is rerun with its allowance grown by memetic_every, its earlier points free (as SRCH-6); it does not see the local searches' points (no Lamarckian step), they only enter the answer. | yes | `search.hpp` |
| `search` | SRCH-8 | A local search's result is the inner solver's answer; worse means more violating, then larger f (Deb). | yes | `search.hpp` |
| `search` | SRCH-12 | memetic's lamarck (task 24, §24.1.1; off by default, and then SRCH-7 holds): on — every local search's answer x' (its values from the archive) is a seed of the global solver's later runs, the newest first, with warm_s... | no: option | `search.hpp` |
| `search` | SRCH-13 | mlsl_stop = bayes (task 24, §24.2.5; off by default: budget): mlsl stops after the iteration where w (gamma k N - 1) / (gamma k N - w - 2) <= w + 1/2 (part II, (3)), w the minima log's minima (at least one), gamma k N... | no: option | `search.hpp` |
| `search` | SRCH-14 | mlsl_cells = on (task 24, §24.2.5; off by default): the check "a sample point of lower f within r_k" looks in the cell of the point and its 3^n neighbours only — the sample in cubes of side r_k (1 + 1e-9) of the norma... | no: option | `search.hpp` |
| `search` | SRCH-9 | memetic: a phase whose global run spends nothing is the last (as RST-8): the global solver has stopped by its own criteria (rerun, it evaluates nothing new), and the local searches alone would go on in phases of a few... | yes | `search.hpp` |
| `search` | SRCH-15 | step_units (task 25, §24.2.4; box by default, the old move): absolute moves every coordinate by U[-1, 1] times the step in the problem's own units instead of times the box's width. | no: option | `search.hpp` |
| `search` | SRCH-16 | Acceptance by a threshold (ils_accept = threshold, bh_accept = threshold; task 25, §24.2.4): X. | yes | `search.hpp` |
| `search` | SRCH-17 | Metropolis on a relative scale (ils_accept = lsmc_rel, bh_accept = metropolis_rel; task 25): the first walk_t0_trials (10) steps accept the better only and record \|dE\| != 0; then T = walk_t_rel (1) times their median,... | yes | `search.hpp` |
| `search` | SRCH-18 | The strength ladder (task 25, §24.2.4; IDLS, Lai et al. 2025, source structural-optimization-atomic_lai2025, Algorithm 1): with ladder_max > 0 the step is ladder_min after a new best, else grows by ladder_step, back t... | yes | `search.hpp` |
| `search` | SRCH-19 | perturb = user (task 25, §24.2.4; task 24's Problem::perturb, §24.1.3): the move is the problem's, perturb(y, step, rng) on a copy of the current minimum with the walk's own generator and the step (the ladder's where... | yes | `search.hpp` |
| `search` | SRCH-20 | basin_hopping with bh_accept = threshold and its adaptive step (no ladder) warns once (mootation::warn): DTS is meant for a fixed move (SRCH-16); the run itself is unchanged. | yes | `search.hpp` |
| `search` | SRCH-10 | ils and basin_hopping: 100 steps in a row that spend nothing end the walk (as RST-8, SRCH-9); mlsl: an iteration that spends nothing ends it (its sample would grow without end); basin_hopping's step grows to the box's... | yes | `search.hpp` |
| `sep_cmaes` | SEP-1 | Line 7's "sqrt(1 - (1 - c_sigma)^2 g)" is (1 - c_sigma)^(2g), as in the CMA-ES literature it comes from. | yes | `sep_cmaes.hpp` |
| `sep_cmaes` | SEP-2 | E\|\|N(0, I)\|\| = sqrt(n) (1 - 1/(4n) + 1/(21 n^2)). | yes | `sep_cmaes.hpp` |
| `sep_cmaes` | SEP-3 | Ranking: the comparison of evo.hpp (NSGA-II's constrained- domination at M = 1 for constraints g <= 0, not in the paper), ties in the order told (stable). | yes | `sep_cmaes.hpp` |
| `sep_cmaes` | SEP-4 | No box: Algorithm 1 has none (as cmaes.hpp); the caller's problem takes x as it comes. | yes | `sep_cmaes.hpp` |
| `sep_cmaes` | SEP-5 | The draws: lambda vectors of n normals, point by point. | yes | `sep_cmaes.hpp` |
| `sep_cmaes` | SEP-6 | c_cov = (n + 2)/3 c_cov^def is held at 1 at most (line 9 needs 1 - c_cov >= 0); below n = 2 nothing changes, the paper's n are larger. | yes | `sep_cmaes.hpp` |
| `shade` | SHADE-1 | "the top N p_i members": the ceil(N p_i) best (p_i >= 2/N gives at least 2); ranking and the trial's draws as pbest.hpp (PB-1, PB-2). | yes | `shade.hpp` |
| `shade` | SHADE-2 | The draws for x_i, in Algorithm 1's order: r_i, CR_i, F_i, p_i, then the trial's. | yes | `shade.hpp` |
| `shade` | SHADE-3 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) for "<=" and "<" of Algorithm 1 and for the ranking; not in the paper. | yes | `shade.hpp` |
| `shade` | SHADE-4 | N >= 4 so that x_i, x_r1, x~_r2 can differ; refused otherwise. | yes | `shade.hpp` |
| `shade` | SHADE-5 | All N trials of a generation are built from P and A at its start (Algorithm 1 builds them all before any selection). | yes | `shade.hpp` |
| `shade` | SHADE-6 | done() is true once all of P and A are one point. | yes | `shade.hpp` |
| `slsqp` | SQP-1 | Incompatible linearized constraints: the QP is solved again with one more variable delta in [0, 1], the equalities and the violated inequalities taken as a^T d + (1 - delta) c, and rho delta^2 / 2 added to the objecti... | yes | `slsqp.hpp` |
| `slsqp` | SQP-2 | B is reset to I and the iteration made again when the QP fails, when no step length is found, or when B loses positive definiteness to round-off; a second failure in a row ends the run. | yes | `slsqp.hpp` |
| `slsqp` | SQP-3 | Stopping: after the QP when \|d\|_inf <= tol and the violation (sum \|c_E\| + sum max(0, -c_I)) <= tol; after a step when \|f+ - f\| <= tol and the violation <= tol; when a step leaves x unchanged bit for bit; after max_ite... | yes | `slsqp.hpp` |
| `slsqp` | SQP-4 | The bounds are rows of the QP (infinite ones none); the point x + alpha d is clipped to the box (a step of an ulp at most: the QP's step keeps the box) and the start too. | yes | `slsqp.hpp` |
| `slsqp` | SQP-5 | No scaling of x, f or c: the caller's. | yes | `slsqp.hpp` |
| `slsqp` | SQP-6 | When no step length passes Armijo's test in 10 trials, the run stops (LineSearch::stop, NW §3.5's stopping test; with B reset first, SQP-2) or takes the last trial point and goes on (LineSearch::accept: what Kraft's c... | yes | `slsqp.hpp` |
| `slsqp` | SQP-7 | The trials end, failed, when x + alpha d is x bit for bit (alpha falls by 1.8 at least a trial, so this comes); then SQP-2 as for stop. | yes | `slsqp.hpp` |
| `slsqp` | SQP-8 | An iteration made again after a reset (SQP-2) takes (4.6) from the mu_i of the last finished iteration, not from those of the failed try. | yes | `slsqp.hpp` |
| `slsqp` | SQP-9 | Not done: the paper's trap for cycling (an error return after five iterations with mu constant and no decrease of phi; | yes | `slsqp.hpp` |
| `solvers` | SCALAR-13 (task 18, addendum 1, §3.2) | The problem's gradient is in its | yes | `solvers.hpp` |
| `spso2011` | SPSO-1 | Synchronous: the whole swarm is evaluated, then every P_i, the links and every L_i are updated, then every particle moves (§II-C's loop updates P_i and L particle by particle). | yes | `spso2011.hpp` |
| `spso2011` | SPSO-2 | L_i: the best P_j over j = i and every j that informs i. | yes | `spso2011.hpp` |
| `spso2011` | SPSO-3 | x' is uniform in the ball's volume: a normal direction, the radius \|\|G - X\|\| U^(1/D). | yes | `spso2011.hpp` |
| `spso2011` | SPSO-4 | "the global optimum shows no improvement": the best P after a generation is not better than the one before it; the links are drawn again then, and at the start. | yes | `spso2011.hpp` |
| `spso2011` | SPSO-5 | "[-X^max, X^max]" (§II-A: V^max = X^max, "the limits of the search space"): \|V_id\| <= (max_d - min_d) / 2, which is X^max on the symmetric [-100, 100] of §III. | yes | `spso2011.hpp` |
| `spso2011` | SPSO-6 | The absorbing wall acts coordinate by coordinate: a coordinate outside its range is put on the bound it crossed and its velocity is 0. | yes | `spso2011.hpp` |
| `spso2011` | SPSO-7 | Draws, particle by particle: U1 for every d, U2 for every d, the direction's D normals, the radius's U; the initial velocity's U after all positions; links: informer by informer, K draws each. | yes | `spso2011.hpp` |
| `spso2011` | SPSO-8 | Constraints g <= 0: NSGA-II's constrained-domination at M = 1 (evo.hpp) for "f(X_i) < f(P_i)", L_i and SPSO-4; not in the paper. | yes | `spso2011.hpp` |
| `spso2011` | SPSO-9 | sym (task 19, symmetry.hpp SYM-9; off by default): align only — P_i and L_i aligned to X_i before the particle moves, the velocity left in the particle's representation; canon refused. | no: option | `spso2011.hpp` |
| `strategy` | STRAT-1 | A start's generator: the run's seed for start 0, else a 32-bit word of std::seed_seq{seed, k} (the standard defines its words), so one start is the inner solver's own run, bit for bit (§3.1's test). | yes | `strategy.hpp` |
| `strategy` | STRAT-2 | The inner solver counts what it asks for against its allowance, the points it gets free from the archive included; the strategy's budget counts only what is new. | yes | `strategy.hpp` |
| `strategy` | STRAT-3 | The budget shared evenly: start k's allowance is budget / n, the first budget mod n starts one more. | yes | `strategy.hpp` |
| `strategy` | STRAT-10 | start_points = around_x0 (task 24, §24.2.5; off by default): start 0 is the start (x0, which it needs), start k >= 1 is x0 + N(0, sigma^2) coordinate by coordinate, sigma = start_sigma times the coordinate's width whe... | no: option | `strategy.hpp` |
| `strategy` | STRAT-12 | keep_points = best (task 24, §24.3.3; off by default: all): the run's evaluator keeps its best point and its last (evaluator.hpp THE LEAN MODE) — multistart, ils, basin_hopping, mlsl, tiktak; a start's own run keeps i... | no: option | `strategy.hpp` |
| `strategy` | STRAT-13 | minima_bayes = on (task 24, §24.2.5; off by default): the settings get Boender's posterior expectations (Rinnooy Kan & Timmer 1987, part I, Proposition 2): the number of minima w (N - 1) / (N - w - 2) (11), "inf" wher... | no: option | `strategy.hpp` |
| `strategy` | STRAT-14 | start_filter = oqnlp (task 24, §24.2.5; off by default: every start runs): Ugray, Lasdon, Plummer, Glover, Kelly & Marti 2007, §3.4, Table 2. multistart's starts become trial points, one after another: the first oq_st... | no: option | `strategy.hpp` |
| `strategy` | STRAT-11 | allowance = sequential (task 24, §24.2.5; off by default: even, STRAT-3) runs multistart's starts one after another, each with all the budget left (its floor), until the budget or the target is spent; one thread, and... | no: option | `strategy.hpp` |
| `strategy` | STRAT-4 | The minima log: a start's answer joins the first minimum within min_dist (the box normalized to [0, 1]^n where it is finite, else the problem's coordinates; the largest coordinate difference) and min_df in f, else is... | yes | `strategy.hpp` |
| `strategy` | STRAT-5 | The run's equalities' tolerance schedule (eq_tol0) is the inner solver's, on its own allowance. | yes | `strategy.hpp` |
| `strategy` | STRAT-6 | A start may spend on new points at most what it was given (its allowance, or less where a strategy reruns a start whose earlier points are free, search.hpp SRCH-6): past it the view ends the inner run as the budget do... | yes | `strategy.hpp` |
| `strategy` | STRAT-7 | A round within a start (a strategy as another's inner solver) runs its starts in the start's own thread, one after another, whatever its threads knob: its views reach the outer start's view, which one thread owns. | yes | `strategy.hpp` |
| `strategy` | STRAT-8 | Every strategy ends below its budget too where its points cost nothing (the archive answers them: points clipped onto known ones, a box of zero width): a loop over the budget ends when a pass of it spends nothing. mul... | yes | `strategy.hpp` |
| `strategy` | STRAT-9 | min_sym = off \| canon \| align (task 19, §19.2.2, S5; off by default): the log's distance between a start's answer and a minimum taken after the answer is aligned to the minimum (align), or both are canonical (canon),... | no: option | `strategy.hpp` |
| `strategy` | STRAT-9 | the log grouped by the problem's symmetries where min_sym is given | yes | `strategy.hpp` |
| `symmetry` | SYM-1 | Ties: Step 1 takes the first non-covered zero in row-major order and the Preliminaries star in row-major order, so the assignment is a function of the matrix: equal blocks or equal costs give the same answer every time. | yes | `symmetry.hpp` |
| `symmetry` | SYM-2 | A zero is an entry equal to 0.0. In floating point Step 3 makes the least non-covered entry exactly 0 (y - y), so the zero the method needs is there; the other entries carry the rounding of the subtractions, which can... | yes | `symmetry.hpp` |
| `symmetry` | SYM-3 | align(x, ref) with x a permutation of ref's blocks returns ref bit for bit: the permutation's cost is 0, the least possible, and the result copies x's blocks, which are ref's. | yes | `symmetry.hpp` |
| `symmetry` | SYM-4 | The groups are aligned one after another, each on its own coordinates (the task's blocks do not overlap; overlapping groups are refused). | yes | `symmetry.hpp` |
| `symmetry` | SYM-5 | sym = canon where the problem gives no canon: each group's blocks sorted lexicographically (stably), the least member of x's orbit under the group: f(canon(x)) = f(x), canon(canon(x)) = canon(x) and canon(P(x)) = cano... | yes | `symmetry.hpp` |
| `symmetry` | SYM-6 | A group's blocks have the same bounds, coordinate by coordinate, so that a permutation stays in the box; refused otherwise. | yes | `symmetry.hpp` |
| `symmetry` | SYM-7 | ga (GA's crossovers): align — the second parent aligned to the first, canon — both parents canonical, before the crossover; the children are in the first parent's representation (align). | yes | `symmetry.hpp` |
| `symmetry` | SYM-8 | de, jade, shade, lshade: align — the base vector and the difference vectors aligned to the target x_i before the mutation; canon — the target too canonical, and the trial's crossover and the bound repair take canon(x_i). | yes | `symmetry.hpp` |
| `symmetry` | SYM-9 | pso, spso2011: align only — the particle's own best and its neighbourhood's best aligned to its position before it moves; the velocity stays in the particle's representation (§19.2.2). canon is refused: it would chang... | yes | `symmetry.hpp` |
| `symmetry` | SYM-10 | The points a kernel makes are evaluated and stored as made; the Evaluator's archive (repeats) is by the raw x. | yes | `symmetry.hpp` |
| `symmetry` | SYM-11 | Off by default (the problem's periodic variables only; task 24, §24.4.2, periodic.hpp): a periodic coordinate's difference in align's costs and in distance is the shorter way round, and align returns each periodic coo... | no: option | `symmetry.hpp` |
| `tiktak` | TT-1 | Step 0's "order them in descending order, as s_1, ..., s_N*, with f(s_1) <= ... <= f(s_N*)" (2019 A.6) is ascending, as §2.1 of both says: the best point first. | yes | `tiktak.hpp` |
| `tiktak` | TT-2 | i = 1: no minimum is known yet ("If i > 1", Step 1.2), S_1 = s_1. | yes | `tiktak.hpp` |
| `tiktak` | TT-3 | N* = max(1, round(share N)); failed or infeasible points of the sample count as +inf and are kept last. | yes | `tiktak.hpp` |
| `tiktak` | TT-4 | The sample: start_points of designs.hpp (sobol: the Sobol sequence under the seed's linear matrix scrambling), so runs of different seeds differ; the paper's Sobol sequence is unscrambled. | yes | `tiktak.hpp` |
| `tiktak` | TT-5 | The searches are sequential (S_i needs p_low of searches 1 ... i - 1): a round of one start each, the generator start_seed(seed, i) (STRAT-1). | yes | `tiktak.hpp` |
| `tiktak` | TT-6 | The budget: Step 0 spends what its N points cost; search i gets start_budget, or what is left shared among the N* - i + 1 searches left, so a search that stops early leaves its rest to the later ones. | yes | `tiktak.hpp` |

## The problems

From the headers of the problem suite (`include/mootation/suite`): the items under a
"DEVIATIONS ..." heading of a header's opening comment, where the definition there departs
from the paper's text.

| item | what | file |
|---|---|---|
| CEC-1 | F5 is -[(4 - 2.1x² + x⁴/3)x² + xy + (4y² - 4)y²], without the factor 4 the report prints: Table IV's height 1.03163 and the published peak ratios (Tables II-III: 1 at every ε) are of this form (owner's В2). | `cec2013.hpp` |
| CEC-2 | The report gives no shifts o_i nor rotations M_i of the composition functions (the competition's data files have them): ours, o_i uniform in [-5, 5]^D, M_i of CF3 and CF4 Haar rotations (condition number one), from st... | `cec2013.hpp` |
| CEC-3 | Where every w_i is 0 (far from every o_i; outside the box only) the weights are equal, 1/n, as the normalization 0/0 leaves them. | `cec2013.hpp` |
| CEC-4 | F8 at D = 2 has k = (3, 4) (the report's 2-D instance); its formula is printed under the name F9, a misprint. | `cec2013.hpp` |
| CEC-5 | Ties of the largest weight: every w_i equal to the largest keeps its value. | `cec2013.hpp` |
| CEC-6 | EF8F2 is evaluated at z + 1, as CEC 2005's expanded F8F2 (x - o + 1), whose code the report's is based on: its minimum is then at z = 0, and CF(o_i) = 0, as the report states; at z itself it would be at (1, ..., 1) an... | `cec2013.hpp` |
| CF-1 | No clipping of x into the box inside the problem (the task's reading P3): PlatEMO clips in CF8 and CF9; the box is the algorithm's. | `cf.hpp` |
| CF-2 | The competition's tolerance (§IV, c >= -1e-6) belongs to its assessment, not to the problem: g = -c as printed. | `cf.hpp` |
| CF-3 | CF8-CF10 divide by 1 - f3^2, 0 at the front's pole f3 = 1 (x_1 = 1, the J3 terms at their optimum): there the constraint is 0/0 or x/0 and comes out NaN, as the printed formula and PlatEMO's code give it. | `cf.hpp` |
| CF-4 | CF8-CF10 as in the version of 20 April 2009 (the task's reading P2: the competition's final report says earlier versions had misprints there; results of 2008-2009 on CF8-CF10 are no reference). | `cf.hpp` |
| CF-5 | \|sin\| in CF1 and CF8, sin without the modulus in CF2, CF3, CF9 and CF10; the threshold of CF4's and CF5's h_2 is 3/2 (1 - sqrt(2)/2): as printed. | `cf.hpp` |
| LZ09-1 | J1 = {j odd, 2 <= j <= n} into f1, J2 = {j even} into f2 (F6: j - 1, j - 2 and j multiples of 3, 3 <= j <= n, into f1, f2, f3), as the table prints them; j counts the variables from 1. jMetal's code swaps J1 and J2 an... | `lz09.hpp` |
| LZ09-2 | F8's cosine is cos(20 y_j pi / sqrt(j)), j the variable's number as printed (jMetal: the index within the set). | `lz09.hpp` |
| RE-24 | x2 in [0.5, 50], the code's: the supplement's S.1.4 prints "x2 in [4, 50]", but the published front (reference_points_RE24.dat: f1 from 60.5 = 0.5 + 120 * 0.5, x2 = 0.5) was computed within the code's bounds. | `re.hpp` |
| RE-25 | x2 in [0.6, 3]: the supplement's S.1.5 prints "x2 in [0.6, 30]"; the code and Kumar et al. 2021 for the same spring (RCM15, "0.6 <= x2 (continuous) <= 3") have 3. | `re.hpp` |
| UF-1 | UF10's sums: f2's over J2 and f3's over J3, where the report prints J1 for all three (the task's reading P1): pagmo2, PlatEMO and jMetal sum so, and the report's own factors 2/\|J2\| and 2/\|J3\| say so. | `uf.hpp` |
| UF-2 | Harmless misprints read as meant: UF3's "the same as those of F1" (UF1's), UF4's "y_i = x_j - ..." (y_j), UF4's "Its PF is f2 = 1 - f2^2" (1 - f1^2), UF10's "f1^2 + f2^2 + f3^3 = 1" (f3^2). | `uf.hpp` |
| WRAP-1 | The order shift, rotate, perm from x (above). | `wrap.hpp` |
| WRAP-2 | The centre of the rotation is the optimum, c = P^T x_opt (c_pi(i) = x_opt_i, the optimum in v's coordinates), where x_opt is known (bbob: the instance's), so that the optimum stays where it was relative to the box; el... | `wrap.hpp` |
| WRAP-3 | A rotation only for a problem without a box (every bound infinite: MGH, NIST) or whose formula is defined on all of R^n and is marked so (defined_everywhere; bbob, whose functions are defined outside [-5, 5]^D and pen... | `wrap.hpp` |
| WRAP-4 | The scale of the shift, per variable of the run: the width of its box, ub - lb, where both bounds are finite; else max(1, \|a\|_inf), a = x_opt where known, else x0, else 1 alone. | `wrap.hpp` |
| WRAP-5 | This is a shift of the whole problem, box and all: relative to its box the optimum does not move. | `wrap.hpp` |
| WRAP-6 | perm and shift know the variables only as a box: a problem whose variables carry more structure (types other than real, blocks or symmetries) is the caller's to refuse, or to permute along; a shift or a rotation of in... | `wrap.hpp` |
| WRAP-7 | A problem with interchangeable blocks, a canonical form or linear parameters (task 19, sym.hpp) is refused by Wrapped as soon as one wrapper is on: a shift or a rotation breaks the blocks' symmetry, and a permutation... | `wrap.hpp` |
| WRAP-8 | The residuals' blocks (res_blocks; task 19, §19.5) pass through unchanged: T maps x, not r, so the residuals keep their order and length. | `wrap.hpp` |

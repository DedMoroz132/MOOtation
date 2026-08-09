<!-- SPDX-License-Identifier: Apache-2.0 -->
# Writing your own algorithm

An algorithm in MOOtation is a plain class template — a **Core**. There is no
base class, no virtual call, no registry. `Optimizer` duck-types whatever you
hand it, so your Core has to satisfy a contract rather than inherit one.

This is deliberate. The library does not offer a kit of interchangeable
"selection" and "variation" blocks you snap together, because papers do not
decompose that way: an algorithm's contribution is usually a specific coupling
between its steps, and a framework that hides the coupling makes the port
unfaithful. You write the loop yourself, exactly as the paper states it.

## The contract

```cpp
template <typename Ind_t>
class MyCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;  // conventional

    void set_seed(unsigned s);                 // required
    void setup(DataVault<Ind_t>& vault);       // required
    void setup_seeded(DataVault<Ind_t>& vault);// required
    void step(DataVault<Ind_t>& vault);        // required

    void set_t_max(int t);                     // optional, see below
};
```

**`setup`** creates the initial population. The vault already has `pop_size()`
slots; fill them with `set_variables`, then call `vault.sync()` once.

**`setup_seeded`** is the same entry point for a population the caller has
already supplied — read the existing slots instead of drawing new ones. Do not
re-randomise here. Anything derived from the population (weights, ideal point,
neighbourhoods) must be computed on **both** paths; a quantity initialised only
in `setup` is a live bug the seeded path will hit.

**`step`** performs exactly one generation.

**`set_seed`** must be the only source of randomness the caller can control.
Seed your engine from it and nothing else.

**`set_t_max`** is optional and detected by SFINAE. Provide it if the algorithm
has a schedule that depends on the total budget — an annealing rate, a
switch-over generation, a freeze window. If you provide it, say in the header
that the caller *must* set it, and what happens if they do not: a schedule
computed against a default budget of 1000 while the run is 200 generations long
is not the algorithm the paper describes.

## The one invariant that is easy to miss

**After every `step()`, the active population must be a valid answer.** A caller
is allowed to stop the run at any generation and read the result. That means a
scratch slot with an unevaluated individual must not be left inside
`[0, pop_size())`, and a partially rebuilt population must not be visible at
return time.

Two consequences:

- If you keep a persistent scratch slot, put it at `pop_size()` and document it.
  Consumers read `[0, pop_size())`, not `[0, active_n())`.
- If your algorithm's real answer lives somewhere else — an archive, for example
  — say so at the top of the header in as many words. `spea2.hpp` and
  `spea2_sde.hpp` do: their answer is `vault.archive_*`, and the active
  population after `step()` is un-selected offspring.

## Slots, indices and function evaluations

`DataVault` hands out **virtual** indices. `active_[v]` maps them to real
buffer slots, and `swap_active` permutes that mapping — so an index means
"whatever currently sits at position v", not a fixed individual.

- `expand(k)` appends `k` slots and **returns the base index**. Use the return
  value. Writing `n + i` is wrong whenever `active_n() != pop_size()` at entry.
- `set_variables` clamps to the bounds and marks the slot dirty. It does not
  evaluate.
- `objectives_of` / `get_cv` / `limits_of` evaluate lazily — reading a dirty
  slot **spends a function evaluation**.
- `seed_individual` writes variables *and* ready objectives, spending none. Use
  it whenever you already have the objectives, e.g. when a MOEA/D-style
  replacement copies an offspring into a neighbour's slot. Doing that with
  `set_variables` + `refresh_objectives` costs one extra FE per replacement, so
  a generation silently becomes `N + replacements` evaluations instead of `N`.

Keep the per-generation FE count equal to what the paper's budget assumes, and
state it in the header. Benchmarks compare on evaluations, not on generations.

## The RNG stream is part of the behaviour

Two runs with the same seed must produce bit-identical populations. That makes
the *number* of random draws part of the observable behaviour, not just their
values. Changing how many draws a branch consumes — adding a rejection loop,
short-circuiting a probability gate, mutating a child you then discard —
changes every subsequent draw and therefore the whole trajectory.

So: do not "clean up" a redundant draw without saying so. `sbx.hpp`
short-circuits at `pc >= 1.0` on purpose, and several algorithms keep an
apparently pointless retry loop for exactly this reason, each with a line in its
header explaining it.

## Constraints

`constraint_mode` is off by default (`NONE`), because the source papers are
almost always unconstrained. When it is on, attach it to your algorithm's
**preference relation** — the one comparison that decides what survives:

| algorithm shape | where constraints attach |
|---|---|
| Pareto / non-dominated sorting | constrained domination in the sort |
| scalarizing (Tchebycheff, PBI) | feasibility-first comparison of the scalar |
| indicator-based | penalty on the fitness, so infeasible truncates first |
| archive-based | the archive's admission and eviction tests |

`detail/constrained.hpp` provides `cdp_dominates`, `dominates`,
`better_scalar` and `penalize` so every file expresses the same two rules the
same way. Leave the geometry alone: angles, clusters, hypervolume and
normalisation are measures, not preferences — constraining them twice distorts
the search instead of directing it.

## The header block

The header comment is the product, as much as the code. Ship:

1. The paper: authors, venue, DOI.
2. A 5–8 line scheme of the generational cycle, with the paper's own step or
   algorithm-line numbers.
3. Paper defaults, each with the section it comes from. If the paper does not
   state a value, say that explicitly and label what you used instead — do not
   attribute a conventional default to a section that does not contain it.
4. A numbered list of deviations. Where the paper contradicts itself, quote both
   passages and say which one you followed and why. A reader must be able to
   check your arbitration without the paper in front of them.

The rule that matters: **a header that misstates the code is as serious as a
bug in the code.** Most of the defects found in this library's own audits were
of exactly that kind — the code was right and the header described something
else.

## Registering it

Add one line to `include/mootation/algorithms.def`:

```
MOOTATION_ALG(my_alg, MyAlg_Individual, MyCore)
```

Every test that iterates over "all algorithms" — convergence, reproducibility,
constraint wiring — picks it up from there, and CI fails if a header in
`include/mootation/algorithms/` is missing from that file. The same line also
puts your algorithm into the run-time dispatch (`mootation::run`,
`mootation::Session`, `algorithm_names()`) and into the Python module, because
both are generated from this one list.

The same line also reaches the C ABI (`moo_algorithm_name`), because that is
built on `algorithm_names()` too.

That file lives in the include tree rather than under `tests/` precisely
because it is part of the public surface: `embed.hpp` includes it three times,
each with a different definition of `MOOTATION_ALG` — once to declare a tag
individual and its `Problem<>` specialization per algorithm, once to build the
`if (name == ...)` dispatch chain, once to list the names. If you add a knob
setter to your core and want it reachable from a config file, add it to
`knob_names()` in `settings.hpp` and to the `MOOTATION_OPTIONAL_SETTER` table in
`embed.hpp`; a core without the setter reports the knob in `Result::ignored`
instead of failing, so the table degrades safely.

# Contributing to MOOtation

Thanks for considering a contribution.

MOOtation has one governing value: **an implementation must match the paper it
claims to implement.** Everything below follows from that.

---

## Before you start

- **Bug reports and questions** — open an issue. For a suspected fidelity bug,
  quote the paper: section, equation or algorithm-line number, and what the
  code does instead. That single detail is what makes a report actionable.
- **New algorithms** — open an issue first. Say which paper, and confirm you
  have legitimate access to it. Do not attach the PDF.
- **Anything larger than a typo** — open an issue before writing code, so we
  don't both solve the same problem twice.

## Contributor License Agreement

All contributions require a signed [CLA](CLA.md). It is a one-time,
one-line comment on your first pull request. You keep the copyright to your
work; the CLA grants the project the rights it needs to distribute it and to
relicense the project as a whole in the future.

## Legal hygiene

Do **not** copy source code from a paper's reference implementation, from
PlatEMO, jMetal, pymoo, or from any other codebase, unless its license permits
it *and* you say so explicitly in the pull request.

Implement from the paper's prose, pseudocode and equations. When an author's
reference implementation resolves an ambiguity that the paper leaves open, you
may cite that fact in a comment — but write the code yourself.

Never commit paper PDFs or converted full texts to this repository.

---

## The header contract

Every file in `include/mootation/algorithms/` opens with a comment block. It is
not decoration — it is the artifact that makes the fidelity claim checkable.
A new algorithm is not complete without it.

For the full Core contract — `setup` / `setup_seeded` / `step` / `set_seed`,
the answer-set invariant, slot and function-evaluation rules, the RNG-stream
rule and where constraint handling attaches — see
[docs/writing-an-algorithm.md](docs/writing-an-algorithm.md).

```cpp
#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// <ACRONYM> — <full title of the algorithm as the paper names it>.
// <Authors>. <Venue>, <year>.
// doi:<DOI>
//
// IDEA. <3-8 lines: what this algorithm does differently from its neighbours.
//        Enough for a reader to decide whether it fits their problem.>
//
// SCHEME (Algorithm <n> in the paper):
//   <5-8 lines tracing one generation, in the paper's own vocabulary and with
//    its own symbol names, so it can be read side by side with the source.>
//
// PAPER DEFAULTS (<section / table the values come from>):
//   <parameter = value, one per line, each traceable to a location>
//
// DECLARED DEVIATIONS:
//   <ACRONYM>-1 (MINOR | DEVIATION | AMBIGUOUS).
//     <What the paper says. What this code does. Why.>
//     <For AMBIGUOUS: quote the contradictory passages and state which
//      reading was taken and on what grounds.>
//   <ACRONYM>-2 ...
//   (write "none" if there are none — an empty list is a claim, so make it)
// ============================================================================
```

Rules for that block:

1. **Every default is traceable.** `eta_c = 30 // Table 1` is fine.
   `eta_c = 30` alone is not — a reader cannot check it.
2. **Every deviation is declared.** An undeclared deviation is the one defect
   this project treats as serious. If the paper is silent, ambiguous or
   self-contradictory, say so and say which reading you took.
3. **The header must not lie.** A header describing behaviour the code does not
   have is worse than no header, because it defeats review. If you change
   behaviour, change the header in the same commit.
4. **Reference the paper, not our notes.** Cite section, equation and algorithm
   line numbers. Do not reference internal review documents or local file
   paths — they do not exist for anyone else.

## Code style

- C++17, header-only, `#pragma once`, everything inside `namespace mootation`.
- No dependencies outside the standard library. This is a hard constraint.
- Match the surrounding file: it is more valuable that a reader can move
  between algorithms than that any one file is styled perfectly.
- Comments in English.
- Seeding goes through the existing `set_seed` convention. Never call
  `std::random_device` outside the documented default.
- Prefer the paper's symbol names over descriptive ones in algorithm bodies
  (`d1`, `d2`, `theta` beat `convergence_distance`) — it keeps the code
  readable next to the paper. Descriptive names belong in the public API.

## Tests

Add your algorithm to the smoke suite in `tests/`. The bar is:

- it converges on ZDT1 / DTLZ2 within the tolerance the suite already uses;
- it runs clean under ASan and UBSan;
- it does not throw for a population size that is not a Das–Dennis lattice
  count, or it documents that restriction in the header and fails with a clear
  message.

Where a paper gives a worked numeric example, turn it into a unit test. Those
are the most valuable tests in this repository.

Run before pushing:

```bash
cmake -S . -B build -DMOOTATION_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Pull requests

- One algorithm, or one coherent fix, per pull request.
- State in the description which paper you worked from and which sections you
  checked against.
- CI must be green: g++ and clang, C++17 and C++20, plus the sanitizer job.
- Expect review to focus on the header block and on deviations. That is the
  part that matters here.

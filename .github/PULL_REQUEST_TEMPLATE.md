<!--
Thanks for the pull request. Delete any section that does not apply.
-->

## What this changes

<!-- One or two sentences. -->

## Primary source

<!--
For a new algorithm or a fidelity fix, name the paper and the parts you
checked against — section, equation, algorithm-line numbers.
Delete this section for build/docs/tooling changes.
-->

- Paper:
- DOI:
- Sections checked:

## Deviations

<!--
List any place where the implementation departs from the paper, and why.
Each one must also be declared in the file's header block.
Write "none" if there are none — that is a claim worth making explicitly.
-->

## Checklist

- [ ] I have signed the [CLA](../CLA.md) (one comment, once per contributor).
- [ ] The code is my own; I did not copy from a reference implementation,
      PlatEMO, jMetal, pymoo or any other codebase — or I did, its license
      permits it, and I say so above.
- [ ] No paper PDFs or converted full texts are included in this change.
- [ ] The file's header block is accurate: DOI, generational scheme, paper
      defaults with the section they come from, and declared deviations.
- [ ] Comments are in English.
- [ ] Tests added or updated; `ctest` passes locally.
- [ ] Clean under ASan and UBSan.
- [ ] CHANGELOG.md updated under `## [Unreleased]`.

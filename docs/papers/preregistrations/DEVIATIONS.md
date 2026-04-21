<!-- docs:meta
topic_id: repo.preregistration.ossm-168-depression.deviations
authority: repo_only
audience: researchers
parent_protocol: 2026-04-21_ossm_168_depression.md
parent_tag: prereg/ossm-168-depression-v1
parent_sha256: 63c2f60223657491829c38a57a2745fb079d48259bed306538ef32223ae4c93d
-->

# Deviations log — `prereg/ossm-168-depression-v1`

This file records every deviation from the frozen pre-registration at
`docs/papers/preregistrations/2026-04-21_ossm_168_depression.md`. It is
append-only. Each entry must record:

- **Timestamp (UTC)**
- **Commit hash** of the repository at the moment the deviation was decided
- **Scope** (which section of the protocol is affected)
- **Description** of the deviation
- **Justification**
- **Reclassification**: whether any affected test is demoted from confirmatory
  to exploratory, per §10 of the protocol

Entries classified as **annotation** are *not* deviations from the protocol;
they are paper-trail notes that clarify ambiguities or record forward
references that the protocol itself permits. Annotations do not trigger
reclassification.

---

## Entry 0001 — 2026-04-21 — Annotation: prospective reference to `fano_labelling_orbits_count`

- **Timestamp (UTC):** 2026-04-21 (freeze day)
- **Commit hash at decision:** `21c1d0178e15edab0e1ddc928fc50ed74199c4d8` (freeze commit)
- **Scope:** §7.1 Base-permutation invariance (labelling control)
- **Type:** Annotation (not a protocol deviation)

### Description

Section 7.1 of the frozen protocol specifies that the 30 inequivalent Fano
labellings of the seven imaginary octonion basis elements are "enumerated in
`formal/OctonionAlgebra.lean` under `theorem fano_labelling_orbits_count`".
At the time of freeze, that theorem does not yet exist in
`formal/OctonionAlgebra.lean`. The file at the freeze commit proves the
octonion multiplication and the Cayley-Dickson structure but does not
contain a name `fano_labelling_orbits_count`.

The reference is prospective: the protocol specifies an algebraic object
(the set of 30 cosets of `Aut(Fano) ≤ S_7`, equivalently the left-coset
space `S_7 / \mathrm{PGL}(3, \mathbb{F}_2)`) which is mathematically
well-defined independently of any particular artefact. Production of a Lean
proof named `fano_labelling_orbits_count` is a deliverable to be produced
in the implementation phase of Trilho A; the *specification* is what has
been frozen, not the artefact.

### Justification

Pre-registrations routinely specify analysis artefacts by reference to
algorithms or structures that will be produced according to a frozen
specification. Freezing the artefact itself is neither required nor
customary. The object described is uniquely defined by the phrase
"the 30 cosets of the Fano automorphism group in `S_7`", which admits no
analytic freedom.

### Reclassification

None. No test is demoted. The confirmatory status of H1, H2, H3 is
unaffected by this annotation.

### Follow-up

The corresponding Lean theorem will be added in a commit whose message
contains a back-reference to this entry. Until then, the 30
representatives used for the §7.1 control are produced by
`scripts/research/ossm_168_dryrun/fano_orbits.py` under fixed seed
`20260421` (as declared in §7.1 for the H2 sedenion sampling case, and
applied uniformly to H1/H3 as well for determinism). The Python
enumeration and the future Lean theorem are obligated to agree on the
30-element set (modulo choice of coset representative).

---

*No further entries. Append new entries below this line.*

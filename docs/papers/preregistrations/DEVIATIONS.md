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

## Entry 0002 — 2026-04-21 — Erratum and supersedence of v1 by v2

- **Timestamp (UTC):** 2026-04-21 (same calendar day as freeze; discovered during Trilho A implementation at commit `fefc6382`)
- **Commit hash at decision:** `fefc6382` (foundation commit of `scripts/research/ossm_168_dryrun/`)
- **Scope:** §2, hypothesis H3 (Negative-valence bias ↔ Associator parity asymmetry)
- **Type:** Erratum with supersedence (not a within-protocol deviation)

### Description

During implementation of the F_1/F_2/F_3 feature computations against the
foundation module `scripts/research/ossm_168_dryrun/`, we observed that
the frozen definition of F_3 in v1 is identically zero by algebraic
construction, making H3 as written untestable.

The frozen definition reads:
```
F_3 = (1 / (T·168)) · Σ_t Σ_{(i,j,k)∈O_168}
        | ‖[h_t[i], h_t[j], h_t[k]]‖ − ‖[h_t[k], h_t[j], h_t[i]]‖ |
```
Since the octonion algebra is alternative (Artin's theorem), the
associator `[a, b, c] = (a·b)·c − a·(b·c)` is an alternating function
of its three arguments; in particular `[c, b, a] = −[a, b, c]`, and
therefore `‖[c, b, a]‖ = ‖[a, b, c]‖`. Every term in the F_3 sum is
zero, for every subject, for every dataset.

The bug does not affect H1 or H2: both are well-defined and non-trivial
as written.

### Resolution

v1 is not amended. The tag `prereg/ossm-168-depression-v1` remains
public, its content frozen, its SHA-256
`63c2f60223657491829c38a57a2745fb079d48259bed306538ef32223ae4c93d`
intact. This preserves the paper trail and respects the standard
open-science rule that a frozen pre-registration is a historical
record, not a mutable document.

v1 is **superseded** by v2, which supplies a corrected F_3 operating on
the sedenion state $\tilde{h}_t \in \mathbb{S}_{16}$. In sedenions the
algebra is no longer alternative, the associator is not alternating,
and the reversal asymmetry $\|[a,b,c]\| \neq \|[c,b,a]\|$ is generic
rather than forbidden. The new F_3 additionally ties H3 to the 336
primitive zero-divisor orbits of Theorem T3, keeping the feature inside
the 168-family.

The full v2 protocol is stored at
`docs/papers/preregistrations/2026-04-21_ossm_168_depression_v2.md`,
frozen under its own SHA-256 and tagged
`prereg/ossm-168-depression-v2`.

### Reclassification

In v1: H3 is withdrawn as untestable. H1 and H2 in v1 remain valid
confirmatory hypotheses if executed against the frozen v1 protocol.

In v2: H1, H2, H3 are all confirmatory. H3 in v2 is **not** downgraded
to exploratory because v2 is a new, coherent registration, not a drift
from v1.

### No data accessed

As of commit `fefc6382`, no LEMON or MODMA signal data has been
accessed. The supersedence therefore occurs entirely within the
implementation phase, before any data contact. This means both v1 and
v2 retain full confirmatory status for analyses performed against
their respective pinned protocols.

---

## Entry 0003 — 2026-04-21 — Annotation: operational resolution of v2 F_3 triple indexing

- **Timestamp (UTC):** 2026-04-21 (during Trilho A/B implementation)
- **Commit hash at decision:** immediately follows `619da07c` (v2 registration commit)
- **Scope:** v2, §2 H3 F_3, specifically the indexing of $(I,J,K)$ in the sum
- **Type:** Annotation (operational resolution; not a protocol deviation)

### Description

v2's F_3 formula sums over $(I,J,K) \in \widetilde{\mathcal{O}}_{168}$ but the
protocol text declares $\widetilde{\mathcal{O}}_{168}$ as "one of the two
orbits of 168 primitive sedenion zero-divisor **pairs** under
$\mathrm{PGL}(3,\mathbb{F}_2)$". A set of pairs and a set of ordered
triples are not the same object; the formula does not specify how to
extract a triple from a pair.

### Resolution (deterministic, pre-data)

For each of the 168 non-Fano ordered triples $(i,j,k) \in \mathcal{O}_{168}$
(with $i,j,k \in \{1,\dots,7\}$), the corresponding sedenion triple is
$$
(I, J, K) \;=\; (i,\; 8+j,\; k).
$$
This choice is justified as follows:

1. **Count preservation.** Exactly 168 triples, matching the protocol's
   $|\widetilde{\mathcal{O}}_{168}|$.
2. **Non-alternativity.** Any triple of the form $(I, 8+j, K)$ with
   $I,K \in \{1,\dots,7\}$ mixes the two Cayley-Dickson halves of
   $\mathbb{S}_{16}$ and therefore lives outside the alternative
   octonion subalgebra, making $\|[\cdot,\cdot,\cdot]\| \neq
   \|[\cdot_{\text{rev}}, \cdot, \cdot_{\text{rev}}]\|$ generic.
3. **Algebraic lineage.** The triple inherits its combinatorial
   structure directly from $\mathcal{O}_{168}$ (used for H1), keeping
   H1 and H3 anchored to the same $\mathrm{PGL}(3,\mathbb{F}_2)$-orbit
   of the 168 Theorem.

### Reclassification

None. v2 is confirmatory, and this annotation fixes the deterministic
implementation of F_3 prior to any data access. No data has been
accessed.

### Follow-up

The enumeration is produced by
`scripts/research/ossm_168_dryrun/sedenion.py::O_168_sedenion_triples()`,
returning a fixed length-168 tuple in lex order of $(i,j,k)$.

---

## Entry 0004 — 2026-04-21 — Erratum and supersedence of v2 by v3

- **Timestamp (UTC):** 2026-04-21 (same calendar day; discovered during sedenion test writing)
- **Commit hash at decision:** immediately follows `619da07c` (v2 registration commit) and the DEVIATIONS-0003 annotation
- **Scope:** v2, §2 H3 (Negative-valence bias ↔ Associator parity asymmetry)
- **Type:** Erratum with supersedence (not a within-protocol deviation)

### Description

While implementing the v2 F_3 definition in
`scripts/research/ossm_168_dryrun/sedenion.py` and testing it against
generic sedenion states, the author verified numerically — and then
derived algebraically — that the v2 F_3 is also identically zero.

The structural fact: **every Cayley-Dickson algebra is flexible**, i.e.
$[x, y, x] = 0$ for all $x, y$ in $\mathbb{C}, \mathbb{H}, \mathbb{O},
\mathbb{S}_{16}, \mathcal{T}_{32}, \dots$. Flexibility alone implies
$[c, b, a] = -[a, b, c]$, by expanding $0 = [a+c, b, a+c] =
[a,b,a] + [a,b,c] + [c,b,a] + [c,b,c] = [a,b,c] + [c,b,a]$. Hence
$\|[c,b,a]\| = \|[a,b,c]\|$ throughout the CD tower.

Any feature defined as a norm asymmetry under $(a \leftrightarrow c)$
reversal therefore collapses to zero regardless of which CD algebra
it inhabits. v2's migration from $\mathbb{O}$ to $\mathbb{S}_{16}$
did not solve the problem; it only relocated it one rung up the
tower that never admits the rescue. This was a genuine oversight
of the author at v2 freeze time.

### Resolution

v2 is not amended. The tag `prereg/ossm-168-depression-v2` remains
public, frozen, with its SHA-256
`ca458198bab2f08ec13d8edbb8b8bb1da54e2cc5f615dec9e60a3f1b2f1b82a5`
intact. Historical integrity preserved.

v2 is **superseded by v3**, which abandons the "parity asymmetry"
framing entirely and redefines F_3 as the $L^1$ mass of the sedenion
associator on the lifted 168-orbit. This is non-zero on generic
states and conceptually a dual of H1 at one rung higher in the CD
tower (H1 measures $L^2$ octonion associator mass; H3 measures
$L^1$ sedenion associator mass). H3 is renamed accordingly.

v3 is stored at
`docs/papers/preregistrations/2026-04-21_ossm_168_depression_v3.md`,
frozen under its own SHA-256 and tagged
`prereg/ossm-168-depression-v3`.

### Reclassification

In v2: H3 is withdrawn as untestable (same fate as H3 in v1).
H1 and H2 in v2 remain valid confirmatory hypotheses.

In v3: H1, H2, H3 are confirmatory. H3 in v3 is not downgraded to
exploratory, because v3 is a new registration rather than a
within-protocol drift from v2, and no LEMON/MODMA signal data has
been accessed at any point in the v1 → v2 → v3 chain.

### No data accessed

At the time of this supersedence, commit
`3364fbbc33a6140394a5c076447926895e9811d0` and earlier, no LEMON
or MODMA signal data has been accessed. Both the v1 → v2 transition
(Entry 0002) and the v2 → v3 transition (this entry) occurred purely
in the implementation phase, before any data contact.

### Honest acknowledgement

Two successive algebraic errata on H3 within a single registration
day is a clear signal that the original phrase "parity asymmetry
via associator reversal" was not a mathematically realisable
construct on the Cayley-Dickson tower. v3's renaming of H3 to
"L¹ sedenion associator mass" is the author's commitment to
align the protocol language with what the feature actually
computes, rather than preserve an evocative but false framing.

---

*No further entries. Append new entries below this line.*

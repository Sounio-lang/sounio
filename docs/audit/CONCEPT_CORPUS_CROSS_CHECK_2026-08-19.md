<!-- docs:meta
topic_id: repo.docs.audit.concept-corpus-cross-check-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.concept-corpus-cross-check-2026-08-19
-->

# Concept corpus cross-check — register, 2026-08-19

**Kind:** register. Not a second measurement.
**Source on disk:** `docs/audit/CROSS_CONCEPTS_2026-08-19.md` (PR #1980,
commit `692a300552`). Every citation below is copied from that receipt.
Nothing here is reconstructed from a compacted panel.

This file exists because a later dispatch asked for this path and this
shape. It does not replace the long form. It does not edit any concept.
It does not apply the derivable repairs.

```text
Semantic-Lane-ID: concept-corpus-cross-check-20260819
Owner: grok-cli3
Concept-IDs: none created
Intent-Preserved: a measured cross-check must exist as a file, not only
  as a panel
Transformation: none — register of an existing reading
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: none beyond CROSS_CONCEPTS_2026-08-19.md
Claims-Forbidden:
  - this is not an independent re-measurement
  - "Fourteen never run" is not restated as a coverage fact
  - no concept document is edited
Assumptions: identical to the source receipt
Write-Set: docs/audit/CONCEPT_CORPUS_CROSS_CHECK_2026-08-19.md
Read-Set: docs/audit/CROSS_CONCEPTS_2026-08-19.md
Positive-Witness: every row names two files and two lines
Negative-Witness: a row marked LOST — remeasure would appear here; none do
Acceptance-Gate: no concept edited; no fifteenth concept; EN-UK
Integration-Target: docs (register)
Authoritative-Only-If: n/a — observational
```

## 1. Corpus

Read at `origin/main` `cfffa919ea`, except as noted.

Thirteen documents were on that SHA. `SOUNIO-ONTOLOGICAL-VALIDATION`
merged as #1967 (`cfffa919ea`) while the source receipt was being
written; its body matched PR head `a60028bb7b`. One document was read
from the queue, not from `main`.

| # | Concept-ID | path | where read |
|---|---|---|---|
| 1 | `SOUNIO-EPISTEMIC-ERASURE` | `docs/internal/concepts/erasure.md` | main `cfffa919ea` |
| 2 | `SOUNIO-NO-IMPLICIT-DEGRADATION` | `docs/internal/concepts/no-implicit-degradation.md` | main `cfffa919ea` |
| 3 | `SOUNIO-PROVENANCE` | `docs/internal/concepts/provenance.md` | main `cfffa919ea` |
| 4 | `SOUNIO-JUSTIFICATION` | `docs/internal/concepts/justification.md` | main `cfffa919ea` |
| 5 | `SOUNIO-PRECISION-PRESERVATION` | `docs/internal/concepts/precision-preservation.md` | main `cfffa919ea` |
| 6 | `SOUNIO-VERIFIED-LOWERING` | `docs/internal/concepts/verified-lowering.md` | main `cfffa919ea` |
| 7 | `SOUNIO-PIPELINE-ORDER` | `docs/internal/concepts/pipeline-order.md` | main `cfffa919ea` |
| 8 | `SOUNIO-ADMISSIBILITY` | `docs/internal/concepts/admissibility.md` | main `cfffa919ea` |
| 9 | *(ladder)* | `docs/internal/concepts/MATURITY_LADDER.md` | main `cfffa919ea` |
| 10 | `SOUNIO-EFFECT-DECLARATION` | `docs/internal/concepts/effect-declaration.md` | main `cfffa919ea` |
| 11 | `SOUNIO-EXACTNESS` | `docs/internal/concepts/exactness.md` | main `cfffa919ea` |
| 12 | `SOUNIO-SIGNAL-DIRECTION` | `docs/internal/concepts/signal-direction.md` | main `cfffa919ea` |
| 13 | `SOUNIO-ONTOLOGICAL-VALIDATION` | `docs/internal/concepts/ontological-validation.md` | main `cfffa919ea` (merged #1967 during the pass) |
| 14 | `SOUNIO-EFFORT-LOCATION` | `docs/internal/concepts/effort-location.md` | **queue** #1972 head `12760ff48b` |

The dispatch that produced the source receipt named twelve on main plus
two in the queue (#1967, #1972). #1967 landed before the receipt was
committed. #1972 remains OPEN at the same head.

No row is `LOST — remeasure`. The source receipt is on disk.

## 2. Tensions (and the two contradictions)

`Yields` is the document whose sentence must move if the other already
defines the term. `INDETERMINATE` means the founder has not chosen.

### C1 — CONTRADICTION — what `because:` holds

| | file | line | text |
|---|---|---|---|
| A | `erasure.md` | 72 | `attest(v, uncertainty: u, because: <provenance>)` |
| B | `justification.md` | 23–25 | `` `because:` names a theorem in `formal/` `` |
| B | `justification.md` | 73–74 | "Empirical claims live in provenance, not in `because:`." |

**Yields:** `erasure.md`. `justification.md:21` is the document that
fixes what a justification is.

### C2 — CONTRADICTION — invocation written as coverage

| | file | line | text |
|---|---|---|---|
| A | `ontological-validation.md` | 58 | "Fourteen never run." |
| B | `ontological-validation.md` | 88–90 | "the coverage is unmeasured and this document says so" |
| B | `effort-location.md` | 44–51 | counted by `git grep -c "<basename>" -- .github/` — direct invocation, not coverage; 45 of 443 unnamed scripts run via a parent |
| B | `effort-location.md` | 152–155 | "A number carries how it was measured, or it is not evidence." |

**Yields:** `ontological-validation.md:58` (and the #1967 merge title
now on `main`). The method is already recorded in #1972 `12760ff48b`.

### T1 — spine: HLIR always, or ENIR only

| | file | line | text |
|---|---|---|---|
| A | `pipeline-order.md` | 20 | "HLIR is always on the path." |
| B | `verified-lowering.md` | 20–21 | "ENIR becomes the only path." |

**Yields:** `INDETERMINATE`. Founder.

### T2 — is ontological narrowing a degradation

| | file | line | text |
|---|---|---|---|
| A | `ontological-validation.md` | 26–27 | a term is "answerable to a model of what exists" |
| A | `ontological-validation.md` | 99–102 | the type–ontology bridge is "not defined here and are not defined anywhere" |
| B | `no-implicit-degradation.md` | 46–56 | nine-row table; no ontological-narrowing row |
| B | `no-implicit-degradation.md` | 79–80 | "Do not treat the degradation table as complete." |

**Yields:** `INDETERMINATE`. Founder. Narrowing `CHEBI:9168` to `f64`
loses chemical identity; nobody owns whether that is a degradation.

### T3 — does `Admissible<T>` require ontological validation

| | file | line | text |
|---|---|---|---|
| A | `admissibility.md` | 30–32 | "`Admissible<T>` requires non-degraded input." |
| A | `admissibility.md` | 96–98 | both greens are epistemic / domain, not ontological |
| B | `ontological-validation.md` | 25–28 | a program is correct only if terms are answerable to what exists |

**Yields:** `INDETERMINATE`. Founder.

### T4 — "evaluation outside the validated domain" has two owners

| | file | line | text |
|---|---|---|---|
| A | `no-implicit-degradation.md` | 54 | "Evaluation outside the validated domain \| — \| **no act**" |
| B | `admissibility.md` | 82–87 | "It belongs here instead." |
| B | `admissibility.md` | 112–114 | do not move the row until an act exists here |

**Yields:** `no-implicit-degradation.md` (exclusive ownership). It keeps
the row and must name the other document. `admissibility.md` already
states the interim rule.

### T5 — "exactly four sinks" versus a fifth

| | file | line | text |
|---|---|---|---|
| A | `erasure.md` | 78–79 | refused "at exactly the four boundaries" |
| A | `erasure.md` | 99–101 | closing three is a different concept |
| B | `admissibility.md` | 74 | "Deciding is the fifth sink." |

**Yields:** `erasure.md` (the word "exactly"). `admissibility.md:77–78`
already splits reporting from acting.

### T6 — two proof systems, no composition rule

| | file | line | text |
|---|---|---|---|
| A | `justification.md` | 23–25, 75–76 | `because:` is a Lean theorem about the program; world-modelling belongs to the paper |
| B | `ontological-validation.md` | 26–27, 53–54 | a term is answerable to what exists; the compiler has an ontological reasoner |

**Yields:** `INDETERMINATE`. Founder.

### T7 — unvalidated rewrite called Garden, not Reserved

| | file | line | text |
|---|---|---|---|
| A | `verified-lowering.md` | 93–95 | a rewrite rule is `Garden` until validated, then `Claim-ready` |
| B | `MATURITY_LADDER.md` | 33–48 | `Reserved` sits beside the ladder (`reserved-owed` / `reserved-taken`) |

**Yields:** `verified-lowering.md` (the rung name). An unselected rule
is closer to `reserved-taken` than to `Garden`.

### T8 — same word, "reachable" (workflow versus "a check exists")

| | file | line | text |
|---|---|---|---|
| A | `ontological-validation.md` | 88 | "Three of seventeen gates are reachable." |
| B | `effort-location.md` | 104 | `R REACHABLE` — "is there a cheap mechanical check that would refuse?" |

**Yields:** `ontological-validation.md` (the word "reachable"). Named-by-
workflow is not the R of S/G/R.

### T9 — EFFORT-LOCATION table still wires from the uncorrected figure

| | file | line | text |
|---|---|---|---|
| A | `effort-location.md` | 124 | "14 ontology gates outside CI \| … \| **gate** \| reachability, one workflow line each" |
| B | `effort-location.md` | 44–51 | that count is invocation; coverage unmeasured; transitive census in flight |
| B | `effort-location.md` | 165–170 | "measure, then decide, then enforce"; do not add a gate on an unmeasured criterion |

**Yields:** `effort-location.md:124`. The footnote already corrected the
method; the worked example did not.

## 3. Gaps (L…)

| id | what is missing | reserved-owed or undecided |
|---|---|---|
| **L1** | ABox. Neither `ontological-validation.md` nor `admissibility.md` names individuals. The founder dispatch (not re-derived here) measured TBox present and ABox almost absent (`individual` 7; `abox` / `instance_of` / `class_assertion` = 0). Claim-validation has no carrier. | **undecided** — no `Reserved-Owner`, `Reserved-Since`, or `Reserved-Blocked-On` |
| **L2** | `no-implicit-degradation.md:52` lists "Precision narrowed (`f256` → `f64`) \| — \| **no act**". `precision-preservation.md:13,33,36` is already **executable** and forbids silent fallback to `f64`. | **not a reservation** — derivable cross-reference; see R4 |
| **L3** | `MATURITY_LADDER.md:99` cites `SOUNIO-PRECISION-PRESERVATION` as the seed of the Cayley-Dickson `i512` tower. `precision-preservation.md` never names `i256`/`i512`. `exactness.md:67–69` does. | **not a reservation** — derivable citation fix; see R5 |
| **L4** | `provenance.md:59–63` puts class in the type (`Knowledge<T, Origem>`). `ontological-validation.md:26–27,99–102` says types are terms and the type–ontology bridge is defined nowhere. Composition unwritten. | **undecided** |
| **L5** | `exactness.md:69` — `i512` is a Garden seed, not a gap. `MATURITY_LADDER.md:93–99` — `i256`/`i512`/`u256`/`u512` exist in no enum; writing `i512` is a generic unknown-type error. | **owed a `reserved-owed` declaration, currently undeclared** — no Owner / Since / Blocked-On. The ladder says they should be reserved. They are not. |

## 4. Derivable repairs — listed, not applied

No founder decision is required. None of these is applied in this
commit. None of these edits a concept file.

| id | from | repair |
|---|---|---|
| R1 | C1 | `erasure.md:72` becomes `attest(v, uncertainty: u, because: <theorem>)`. Empirical half stays in provenance at construction. |
| R2 | C2 | `ontological-validation.md:58` (and the #1967 title) becomes the sentence `:88–90` already permits: three gates named by a workflow; fourteen unnamed by direct invocation; coverage unmeasured. Do not wire fourteen workflow lines until the transitive census lands. |
| R3 | T4 | `no-implicit-degradation.md:54` cross-references `SOUNIO-ADMISSIBILITY`: classified here provisionally; do not move until an act exists there. |
| R4 | T5 | `erasure.md:78` becomes "four reporting sinks; deciding is the fifth, owned by `SOUNIO-ADMISSIBILITY`." `:99–101` stays for the reporting set. |
| R5 | T7 | `verified-lowering.md:93–95` names the unvalidated-and-unselected rule `reserved-taken`, or says why `Garden` is the right rung. |
| R6 | T8 | `ontological-validation.md:88` says "named by a workflow", not "reachable". |
| R7 | T9 | `effort-location.md:124` restated: unnamed by workflow (direct invocation); coverage unmeasured — do not wire until the transitive census lands. |
| R8 | L2 | `no-implicit-degradation.md:52` cross-references `SOUNIO-PRECISION-PRESERVATION`. |
| R9 | L3 | `MATURITY_LADDER.md:99` cites `SOUNIO-EXACTNESS` (or both), not precision-preservation alone. |

T1, T2, T3, T6, L1, L4, L5 are **not** in this list. They need a founder
choice or a `reserved-owed` declaration that does not exist yet.

## What this register does not do

- It does not write a fifteenth concept.
- It does not edit any concept document.
- It does not apply R1–R9.
- It does not re-measure the ABox counts (L1 cites the founder dispatch).
- It does not treat "fourteen never run" as coverage.
- It does not touch `self-hosted/`.
- It does not run a compiler.

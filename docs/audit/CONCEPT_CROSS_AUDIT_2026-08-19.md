# Concept Cross-Audit — the eleven of 2026-08-19 (+ exactness)

- **Date:** 2026-08-19
- **Author:** fable-1
- **Dispatch:** cross the eleven concepts written today; find where they
  CONTRADICT (not add a twelfth). Instrument rule: every claim of agreement or
  disagreement cites the line in each document.
- **Corpus (read from `origin/main`, PRs #1966/#1967 for the two retained):**
  erasure, no-implicit-degradation, provenance, justification,
  precision-preservation, verified-lowering, pipeline-order, admissibility,
  MATURITY_LADDER, exactness, effect-declaration (#1966), ontological-validation
  (#1967).

## Semantic Lane Declaration

```text
Semantic-Lane-ID: concept-cross-audit-2026-08-19
Owner: fable-1
Concept-IDs: none created. Cross-references all twelve above.
Intent-Preserved: dispersion must not let two founder documents contradict each
  other unnoticed (the state SOUNIO-ADMISSIBILITY and SOUNIO-ONTOLOGICAL-VALIDATION
  were written to end).
Transformation: none to code. Read-only audit; classifies inter-document findings.
Types-Changed: none. Effects-Changed: none. IR-Changed: none.
Claims-Introduced: one CONTRADICTION, four TENSIONS, three LACUNAS, and a cited
  OK set exist between the twelve documents (below).
Claims-Forbidden:
  - This audit does NOT resolve founder-level tensions (T1, T2, T3). It formulates
    each as one question and stops.
  - This audit does NOT edit any concept doc; proposals are proposals.
  - A finding marked OK means "no contradiction found between the cited lines",
    not "the concept is correct or implemented".
Assumptions: the two PR docs (#1966/#1967) are read at their PR-head content and
  may change before merge.
Write-Set: docs/audit/CONCEPT_CROSS_AUDIT_2026-08-19.md (this file only).
Read-Set: docs/internal/concepts/*.md (twelve).
Positive-Witness: the cited line pairs below; each finding names file:line in both.
Negative-Witness: C1 is falsifiable — if `attest`'s `because:` is defined to hold
  provenance (not a theorem), justification.md is wrong, not erasure.md.
Acceptance-Gate: every finding carries two citations; no self-hosted/ touched.
Integration-Target: docs (audit record); feeds a founder decision on T1–T3.
Authoritative-Only-If: n/a.
```

## Cross matrix (useful half — pairs with a finding; blanks = no finding sought)

|  | just | prov | no-deg | admis | verif | pipe | onto | exact | prec | effdecl |
|---|---|---|---|---|---|---|---|---|---|---|
| **erasure** | **C1** | ok | ok | **L1** | | | | ok(mark) | | ok(vocab→L3) |
| **justification** | — | ok | ok | ok | | | | | | |
| **provenance** | | — | ok | | | | | | | |
| **no-deg** | | | — | **T4** | | | **T2** | ok(r4) | **L2** | ok(silence) |
| **admissibility** | | | | — | | | **T3** | | | ok |
| **verified-lowering** | | | | | — | **T1** | | | ok | |
| **pipeline-order** | | | | | — | — | | | | |
| **ontological-val** | | | | | | | — | | | |
| **MATURITY_LADDER** governs status terms for all; consistency = **OK** (cited below) |

## Findings

### C1 — CONTRADICTION (gravest class: one doc makes a reading another forbids)
**What goes in `attest(..., because:)` — provenance or a theorem.**
- `erasure.md:73`: "Returning knowledge to a bare number requires
  `attest(v, uncertainty: u, because: <provenance>)`".
- `justification.md:73-74` (Required Invariant): "Empirical claims live in
  provenance, not in `because:`. Moving one into a justification slot re-creates
  the inert string this concept exists to remove." And `justification.md:25-26`:
  "`because:` names a theorem in `formal/`".

erasure's canonical example puts provenance in the `because:` slot; justification
forbids exactly that and defines the slot as a theorem name. Same slot, two
meanings — the same-term contradiction the dispatch said to hunt.
**Whose decision:** DERIVABLE. justification is the document that *defines* what a
justification is ("This document fixes what a justification *is*", `justification.md:12`);
erasure predates that fixing within the same day. **Proposal:** amend
`erasure.md:73` to `attest(v, uncertainty: u, because: <theorem>)` and state that
the empirical/citation half rides in provenance at construction (as
`justification.md:44-46` and `admissibility.md:70-72` already say). No founder
call needed unless the founder wants `because:` to carry both.

### T1 — TENSION (founder's) — what is the spine
- `pipeline-order.md:20-27`: "HLIR is always on the path... the layer everything
  descends through; the backends hang below it. `check → HLIR → ir/ → native (CPU)`
  `↘ enir → verify (epistemic)`".
- `verified-lowering.md:21-22`: "ENIR becomes the only path. `ir/` and the e-graph
  are an optional accelerator, never mandatory" — and never mentions HLIR.

Two documents each name a *different* layer as the one "everything descends
through": HLIR (pipeline-order) vs ENIR (verified-lowering). pipeline-order's CPU
diagram keeps non-epistemic code on `HLIR→ir/→native` with ENIR only on the
epistemic branch; verified-lowering's ruling puts *all* content through ENIR with
`ir/` optional. Both are `Status: Hypothesis`; pipeline-order records the tension
(`:58-84`, three unchosen shapes) — but, as the dispatch notes, "open in two
documents" is not resolved, and a reader of verified-lowering alone never learns
HLIR is claimed as the spine. **Confirms the founder's T1.**
**Whose decision:** FOUNDER. **One-sentence question:** *Does non-epistemic CPU
code descend through ENIR (verified-lowering's "only path") or through
HLIR→ir/→native with ENIR only for trusted content (pipeline-order's diagram) —
i.e. is the spine ENIR-with-ir/-optional, or HLIR-above-ENIR?*

### T2 — TENSION/GAP (founder's) — is ontological narrowing a degradation
- `ontological-validation.md:26-28`: "a term is not merely well-typed, it is
  answerable to a model of what exists" — but `:99-102` (Claims-Forbidden): "its
  bridge to the type system are not defined here and are not defined anywhere."
- `no-implicit-degradation.md:46-56` (the nine-row table) has no row for narrowing
  an ontological term; `:79-80` (Claims-Forbidden): "Do not treat the degradation
  table as complete."

Narrowing `CHEBI:9168` to `f64` loses ontological identity. Neither doc owns it:
no-implicit-degradation's table omits it (and disclaims completeness);
ontological-validation disclaims defining the type↔ontology bridge at all. So it
is an **acknowledged GAP**, not a flat contradiction — but nobody owns whether it
is a degradation. **Confirms the founder's T2** (as a gap, not a contradiction).
**Whose decision:** FOUNDER. **One-sentence question:** *Is narrowing an
ontological term (e.g. CHEBI:9168 → f64) a degradation under
no-implicit-degradation requiring a named act, or a separate axis — and which
document owns the type↔ontology bridge that ontological-validation says is "not
defined anywhere"?*

### T3 — GAP (founder's) — admissibility is silent on ontological validation
- `admissibility.md:30-32`: "`Admissible<T>` requires non-degraded input" and
  `:96-98`: "Both must be green: the information survived, **and** the decision
  fits the domain." Both clauses are epistemic/domain; neither is ontological.
- `ontological-validation.md:19-28` makes ontological validity "part of the soul",
  validating terms (and, per the founder's 1+2 choice, claims).

If a decision must be answerable to a model of what exists, an admissible decision
plausibly needs an ontological check too — but admissibility names only the
non-degraded-input check, and the two documents are not wired together.
**Confirms the founder's T3.** (Implementation aggravator, out of concept scope
but recorded: the store has TBox and no ABox, so half of claim-validation has no
carrier — `ontological-validation.md:83-85` already notes 14/17 gates never run.)
**Whose decision:** FOUNDER. **One-sentence question:** *Does an `Admissible<T>`
decision also require ontological validation of its terms, not only non-degraded
epistemic input — and if so, which document owns that coupling?*

### T4 — TENSION (derivable) — "evaluation outside the validated domain" has two owners
- `no-implicit-degradation.md:54`: table row "Evaluation outside the validated
  domain | — | **no act**" — no cross-reference.
- `admissibility.md:82-87`: "It belongs **here** instead... Filing it as
  degradation is what left it without an act", and `:112-114` (Claims-Forbidden):
  "Do not move *evaluation outside the validated domain* out of the degradation
  table until an act exists for it here."

Same item classified by no-implicit-degradation as a degradation and by
admissibility as an admissibility question; a reader of the former never learns
of the latter's claim. **Whose decision:** DERIVABLE — admissibility already sets
the interim rule (keep it listed until the admissibility act exists). **Proposal:**
add to `no-implicit-degradation.md:54` a cross-reference: "classified here
provisionally; `SOUNIO-ADMISSIBILITY` argues this is an admissibility, not a
degradation — do not move until an act exists there."

### L1 — LACUNA — erasure says "exactly four sinks"; admissibility adds a fifth
- `erasure.md:78-79`: "refused at exactly the four boundaries"; `:99-101`: "Closing
  three is not a weaker version."
- `admissibility.md:74`: "**Deciding is the fifth sink.**"
Not a contradiction (admissibility frames the four as *reporting*, the fifth as
*acting*, `:77-78`), but erasure's "exactly four" does not reference the fifth.
**Proposal (derivable):** erasure cross-references admissibility's fifth sink.

### L2 — LACUNA — precision-narrowing row ignores an executable concept
- `no-implicit-degradation.md:52`: "Precision narrowed (`f256` → `f64`) | — | **no
  act**".
- `precision-preservation.md:13` Status **executable**; `:33` "Narrowing is
  explicit or proven lossless"; `:36` "Backend failure is never silent fallback to
  `f64`."
The invariant against silent precision narrowing exists and is executable; row 5
presents it as unaddressed ("no act") with no cross-reference. The two are
compatible (precision-preservation forbids *silent* narrowing; the "no act" is the
missing *named operation* for *intentional* narrowing), but the reader is misled.
**Proposal (derivable):** row 5 cross-references precision-preservation.

### L3 — LACUNA (low) — the epistemic-effect vocabulary is not reconciled with measurement
- `erasure.md:23` / `no-implicit-degradation.md:33`: the epistemic effects are
  "`Observe`, `Learn`, `Witness`, `Prob` and `Audit`".
- `effect-declaration.md:66-75`: measured recognised effects include `Observe` 33,
  `Witness` 21, `Learn` 13, but `GUM` (91) and `Uncertainty` (20) are **not**
  recognised, and "`GUM`, `Uncertainty` and `Epistemic` first appear in the same
  commit, 2025-12-25 — day one... only one reached the compiler."
Design vocabulary (the five acquisition effects) and measured reality (GUM/
Uncertainty unrecognised, used more than Observe/Witness/Learn) are not reconciled.
Low priority; **measurement**, not a founder decision.

## OK — agreements, cited (no empty OKs)
- **The "mark" mechanism is one concept across three docs.** `erasure.md:57-60`
  ("`k.value`... arithmetic over it stays marked"), `exactness.md:60-61` ("must be
  marked as having done so"), `no-implicit-degradation.md:48` ("mark propagates").
  Coherent compile-time taint.
- **The two-program / ladder rule is invoked consistently.** `MATURITY_LADDER.md:106-117`,
  `erasure.md:16-18`, `admissibility.md:106-108` ("counting files... measures reach,
  not the two-program test"), `ontological-validation.md:90-91`. No doc claims a
  position its own evidence forbids.
- **Program-property vs world-validity is drawn the same way.**
  `justification.md:101-102` ("It verifies a program property; validity is the
  paper's"), `precision-preservation.md:37`, `exactness.md:62-63`. Matches the
  `SEMANTIC_LANE_CONTRACT` distinction "formal model != empirical claim".
- **"The act/constructor is the only door" is one principle.** `provenance.md:72-76`
  ("The constructor is the only door"), `no-implicit-degradation.md:68-70` ("The act
  is the only door").
- **`Epistemic` is a recognised effect, consistently.** `effect-declaration.md:69`
  ("`Epistemic` 316" among the 29 ids) matches `check/effects.sio` `effect_name_to_id`
  returning id 8 for "Epistemic" — the basis my own CEI P0 bypass relies on.

## Summary for the founder
- **1 contradiction (C1)** — derivable; `attest(because:)` = theorem, not
  provenance; propose fixing erasure's example.
- **3 tensions are yours (T1, T2, T3)** — one question each, above; I stop there.
- **T4 + L1 + L2** — derivable cross-reference repairs; proposed, not applied.
- **L3** — a measurement to reconcile, low priority.
- The rest of the corpus is coherent on the maturity ladder and the
  program-vs-world line, with citations above.

<!-- docs:meta
topic_id: repo.docs.audit.cross-concepts-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.cross-concepts-2026-08-19
-->

# Cross-concepts audit — fourteen documents, 2026-08-19

**Date:** 2026-08-19
**Kind:** read-only concept-versus-concept audit. No concept was written. No
`self-hosted/` file was touched. No compiler was built.
**Companion TSV:** `docs/audit/CROSS_CONCEPTS_2026-08-19.tsv`

`scripts/ci/concept_status_gate.sh` (wired at `.github/workflows/ci.yml:68`)
checks **state against evidence**. It does not check **concept against
concept**. This receipt is that missing check for the fourteen documents
written or queued on 2026-08-19.

## Semantic lane declaration

```text
Semantic-Lane-ID: cross-concepts-20260819
Owner: grok-cli3
Concept-IDs: none created. Cross-references the fourteen listed below.
Intent-Preserved: two founder documents must not contradict each other
  unnoticed — the dispersion state SOUNIO-ADMISSIBILITY and
  SOUNIO-ONTOLOGICAL-VALIDATION were written to end.
Transformation: none to code or to any concept document. Classification only.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - two CONTRADICTIONS, nine TENSIONS, five GAPS, and a cited OK set exist
    among the fourteen documents (below)
  - the #1972 invocation-versus-coverage correction is itself an instance of
    SOUNIO-EPISTEMIC-ERASURE outside the compiler, and it falsifies a
    sentence still sitting in #1967
Claims-Forbidden:
  - This audit does not resolve founder-level tensions. Each is one sentence
    and then a stop.
  - This audit does not edit any concept document. Proposals are proposals.
  - A cell marked OK means "no contradiction found between the cited lines",
    not "the concept is correct or implemented".
  - "Fourteen never run" is not restated here as a coverage fact.
Assumptions:
  - thirteen documents read at origin/main cfffa919ea
    (SOUNIO-ONTOLOGICAL-VALIDATION merged as #1967 during this pass;
    body identical to the PR head a60028bb7b that was read)
  - SOUNIO-EFFORT-LOCATION read at #1972 head 12760ff48b
    (commit message: "the gate row measured invocation, not coverage — and say so")
Write-Set:
  docs/audit/CROSS_CONCEPTS_2026-08-19.md
  docs/audit/CROSS_CONCEPTS_2026-08-19.tsv
Read-Set: the fourteen concept documents listed under Corpus
Positive-Witness: every finding names file:line on both sides
Negative-Witness: C1 is falsifiable — if attest's because: is defined to hold
  provenance, justification.md is wrong, not erasure.md
Acceptance-Gate: every CONTRADICTION and TENSION carries two citations;
  no self-hosted/ touched; no fifteenth concept written
Integration-Target: docs (audit record); feeds founder decisions listed below
Authoritative-Only-If: n/a — observational
```

## Corpus

Thirteen documents on `origin/main` at `cfffa919ea`.
`SOUNIO-ONTOLOGICAL-VALIDATION` merged as #1967 while this audit was being
written; its body is identical to the PR head that was read. The
contradictory sentence in C2 is therefore now on `main`.

| abbr | Concept-ID | path | status line |
|---|---|---|---|
| ER | `SOUNIO-EPISTEMIC-ERASURE` | `docs/internal/concepts/erasure.md` | Hypothesis |
| NID | `SOUNIO-NO-IMPLICIT-DEGRADATION` | `docs/internal/concepts/no-implicit-degradation.md` | Hypothesis |
| PRV | `SOUNIO-PROVENANCE` | `docs/internal/concepts/provenance.md` | Hypothesis |
| JST | `SOUNIO-JUSTIFICATION` | `docs/internal/concepts/justification.md` | Hypothesis |
| PREC | `SOUNIO-PRECISION-PRESERVATION` | `docs/internal/concepts/precision-preservation.md` | executable |
| VL | `SOUNIO-VERIFIED-LOWERING` | `docs/internal/concepts/verified-lowering.md` | Hypothesis |
| PO | `SOUNIO-PIPELINE-ORDER` | `docs/internal/concepts/pipeline-order.md` | Hypothesis |
| ADM | `SOUNIO-ADMISSIBILITY` | `docs/internal/concepts/admissibility.md` | Hypothesis |
| ML | *(ladder, not a Concept-ID)* | `docs/internal/concepts/MATURITY_LADDER.md` | governance |
| ED | `SOUNIO-EFFECT-DECLARATION` | `docs/internal/concepts/effect-declaration.md` | Hypothesis |
| EX | `SOUNIO-EXACTNESS` | `docs/internal/concepts/exactness.md` | hypothesis |
| SD | `SOUNIO-SIGNAL-DIRECTION` | `docs/internal/concepts/signal-direction.md` | Hypothesis |
| OV | `SOUNIO-ONTOLOGICAL-VALIDATION` | `docs/internal/concepts/ontological-validation.md` | Hypothesis |

One document still in the open queue, read at its PR head, not as merged:

| abbr | Concept-ID | PR | head | path |
|---|---|---|---|---|
| EL | `SOUNIO-EFFORT-LOCATION` | #1972 OPEN | `12760ff48b` | `docs/internal/concepts/effort-location.md` |

#1967's merge title on `main` is still "the fourteen gates that never run".
That title is part of finding C2.

## How to read a class

| class | means |
|---|---|
| **CONTRADICTION** | both cannot be true, or one document makes a reading another forbids |
| **TENSION** | both can be true, but only after a decision nobody has taken |
| **GAP** | one document refers to something the other should define and does not |
| **OK** | cited agreement — a cell without citations is not an OK |

Instrument rule: if two documents agree, the line of each is shown. A matrix
full of uncited OK is how a morning of greps starts to lie.

## Same-term inventory

The dispatch asked for this hunt on purpose. Same English word, different
sense, is the failure mode that survives a casual read.

| term | document A | sense A | document B | sense B | class |
|---|---|---|---|---|---|
| `because:` | ER:72 | a provenance value | JST:23–25, :73 | a Lean theorem in `formal/`; empirical claims are forbidden here | **C1** |
| *never run* / *coverage* | OV:58 | "Fourteen never run" (absolute) | OV:88–90; EL:44–51, :152–155 | coverage is unmeasured; the count was direct invocation | **C2** |
| *reachable* | OV:88 | named by a workflow | EL:104 | a cheap mechanical check exists (the R of S/G/R) | **T8** |
| *path* | VL:20–21 | ENIR is the only compilation path | PO:20 | HLIR is always on the path | **T1** |
| *decided* | NID:48 | founder closed the design | EX:19, :55 | a decidable equality, not a measurement | watch (compatible) |
| *floor* | VL:25; PO:29 | verification is the minimum | JST:23; ADM:70 | a discharged proof is the minimum | **OK** (same sense: mandatory minimum) |
| *silence* | NID:20–21; ED:20–22 | loss / unknown name accepted without a mark | EL:100–101 | S of S/G/R: violation produces output that looks valid | **OK** (one principle, three layers) |
| *gap* | EX:69 | `i512` is a Garden seed, "not a gap" | ML:93–99 | the same widths exist in no enum and should be reserved | **G5** |

## Cross matrix (useful half)

Blank cells were inspected and produced no finding worth a row. They are not
silent OKs.

|  | JST | PRV | NID | ADM | VL | PO | OV | EL | EX | PREC | ED | SD | ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ER** | **C1** | OK | OK | **T5** | OK |  |  | **C2**† | OK |  | OK |  | OK |
| **JST** | — | OK | OK | OK |  |  | **T6** |  |  |  |  |  |  |
| **PRV** |  | — | OK |  | OK | T1.2 | **G4** |  |  |  |  |  |  |
| **NID** |  |  | — | **T4** |  |  | **T2** | OK | OK | **G2** | OK | OK |  |
| **ADM** |  |  |  | — |  |  | **T3** |  |  |  | OK |  | OK |
| **VL** |  |  |  |  | — | **T1** |  |  |  | OK |  |  | **T7** |
| **PO** |  |  |  |  |  | — |  |  |  |  |  |  |  |
| **OV** |  |  |  |  |  |  | — | **C2** **T8** **T9** |  |  |  |  | OK |
| **EL** |  |  |  |  |  |  |  | — |  |  | OK | OK | OK |
| **EX** |  |  |  |  |  |  |  |  | — | OK |  |  | **G5** |
| **PREC** |  |  |  |  |  |  |  |  |  | — |  |  | **G3** |
| **ED** |  |  |  |  |  |  |  |  |  |  | — |  |  |
| **SD** |  |  |  |  |  |  |  |  |  |  |  | — | OK |

† ER is on the C2 row because EL:58–63 names the invocation-number error as
`SOUNIO-EPISTEMIC-ERASURE` outside the compiler.

## Findings

### C1 — CONTRADICTION (Claims-Forbidden class)

**What goes in `attest(..., because:)` — provenance, or a theorem.**

- ER:72: "Returning knowledge to a bare number requires
  `attest(v, uncertainty: u, because: <provenance>)`."
- JST:23–25: "`because:` names a theorem in `formal/`, and the build fails if
  that theorem does not exist or does not close."
- JST:73–74 (Required Invariant): "Empirical claims live in provenance, not
  in `because:`. Moving one into a justification slot re-creates the inert
  string this concept exists to remove."
- ADM:70–72 already takes JST's side: "`attest(v, uncertainty:, because:)`
  remains the way through, and its floor is a discharged proof."

Erasure's canonical example puts provenance in the slot that justification
exists to keep empty of provenance. Same token, two meanings — the
same-term contradiction the dispatch said to hunt. This is the gravest
class: one document **makes** a reading another **forbids**.

**Whose decision:** derivable. JST:21 is the document that "fixes what a
justification *is*". Erasure predates that fixing on the same day.

**Proposal (not applied):** amend ER:72 to
`attest(v, uncertainty: u, because: <theorem>)` and state that the
empirical half rides in provenance at construction, as JST:44–46 and
ADM:70–72 already say. No founder call unless `because:` is meant to
carry both.

### C2 — CONTRADICTION (Claims-Forbidden class; #1972 is the instance)

**"Fourteen never run" versus "coverage is unmeasured".**

- OV:58: "Three of the seventeen gates are named by any workflow. **Fourteen
  never run.**"
- OV:88–90 (Claims-Forbidden): "Three of seventeen gates are reachable; **the
  coverage is unmeasured** and this document says so."
- EL:35 still lists "14 of 17 ontology gates named by no workflow" as a
  defect row, then immediately corrects the measurement:
- EL:44–51: that row was counted by `git grep -c "<basename>" -- .github/` —
  **direct invocation, not coverage**. A same-day census of 443
  workflow-unnamed scripts found **45 covered by a running parent**.
  "Named by no workflow" is a lower bound. A transitive-closure measurement
  is in flight.
- EL:152–155: "A number carries how it was measured, or it is not evidence."
- EL:58–63: the error is `SOUNIO-EPISTEMIC-ERASURE` outside the compiler,
  committed by the author of this corpus on the same day the concept was
  specified. The wrong number supported a conclusion that remains true
  (more gates are written than wired), which is why it did not scream.

#1972 commit `12760ff48b` ("the gate row measured invocation, not coverage
— and say so") put the correction **in the document, not in an erratum**,
because the error instantiates the concept. That is the right shape.

What the correction does **not** do is retract OV:58, or the #1967 merge
title now on `main` ("the fourteen gates that never run"). Those two
sentences are still an absolute coverage claim. They cannot be true
together with OV's own Claims-Forbidden, and they are a reading
EL:152–155 forbids. #1967 landing during this pass does not resolve C2;
it promotes the contradictory sentence.

**Whose decision:** derivable. OV's Claims-Forbidden already has the honest
sentence. EL already recorded the method.

**Proposal (not applied):** rewrite OV:58 (and the #1967 title) to the
sentence OV:88–90 already permits — three gates named by a workflow;
fourteen unnamed by direct invocation; coverage unmeasured. Do not wire
fourteen workflow lines until the transitive census lands (EL:165–170:
measure, then decide, then enforce).

This audit does not repeat "fourteen never run" as a fact.

### T1 — TENSION (founder) — what is the spine

Confirms the tension the founder already named.

- PO:20–27: "HLIR is always on the path." Diagram:
  `check → HLIR → ir/ → native` (CPU), with `↘ enir → verify` (epistemic).
- VL:20–21: "ENIR becomes the only path. `ir/` and the e-graph are an
  optional accelerator, never mandatory." VL never mentions HLIR.
- PO:29–32 restates VL's ruling, then PO:74–83 records three unchosen
  shapes. PO:106–107 (Claims-Forbidden): "Do not present the two rulings
  as a resolved architecture."
- VL:114–115 (Claims-Forbidden): "Do not describe ENIR as the compilation
  path, the default, or in transition to either."

Both documents record the tension as open. "Open" in two documents is not
a resolution, and a reader of VL alone never learns that HLIR is claimed
as the spine. The same-term root is *path*: VL's "only path" and PO's
"always on the path" cannot both be the unique spine without a shape.

**Whose decision:** founder.

**One-sentence question:** Does non-epistemic CPU code descend through ENIR
(VL's "only path") or through HLIR → `ir/` → native with ENIR only for
trusted content (PO's diagram) — that is, is the spine
ENIR-with-`ir/`-optional, or HLIR-above-ENIR?

### T2 — TENSION and GAP (founder) — is ontological narrowing a degradation

Confirms the tension the founder already named.

- OV:26–27: "a term is not merely well-typed, it is answerable to a model
  of what exists."
- OV:99–102 (Claims-Forbidden): the model, its axioms, "and its bridge to
  the type system are not defined here and are not defined anywhere."
- NID:46–56 (the nine-row table) has no row for narrowing an ontological
  term. NID:79–80 (Claims-Forbidden): "Do not treat the degradation table
  as complete."
- ER:38: "`Knowledge` is a plain struct — `value`, `variance`,
  `confidence`." If types are ontological terms, projecting `Knowledge`
  onto those three fields is the same class of loss as narrowing
  `CHEBI:9168` to `f64`.

Narrowing `CHEBI:9168` to `f64` loses chemical identity. Neither document
owns whether that loss is a degradation. NID disclaims completeness; OV
disclaims the type–ontology bridge.

**Whose decision:** founder.

**One-sentence question:** Is narrowing an ontological term (for example
`CHEBI:9168` → `f64`) a degradation under NID that requires a named act,
or a separate axis — and which document owns the type–ontology bridge
that OV says is defined nowhere?

### T3 — TENSION and GAP (founder) — admissibility is silent on ontology

Confirms the tension the founder already named.

- ADM:30–32: "`Admissible<T>` requires non-degraded input."
- ADM:96–98: "Both must be green: the information survived, **and** the
  decision fits the domain." Both clauses are epistemic / domain. Neither
  is ontological.
- OV:25–28 makes ontological validity part of what it means for a Sounio
  program to be correct, validating terms and — on the founder's 1+2
  choice, types-as-terms **and** validation of claims — claims as well.
- Neither OV nor ADM mentions an ABox. Claim-validation without
  individuals has no carrier. That infrastructure gap is recorded here
  as aggravation, not as a concept contradiction; it is not measured in
  this reading pass.

If a decision must be answerable to a model of what exists, an admissible
decision plausibly needs the ontological check too. The two documents are
not wired together.

**Whose decision:** founder.

**One-sentence question:** Does an `Admissible<T>` decision also require
ontological validation of its terms and claims, not only non-degraded
epistemic input — and if so, which document owns that coupling?

### T4 — TENSION (derivable) — "evaluation outside the validated domain" has two owners

- NID:54: table row "Evaluation outside the validated domain | — | **no
  act**" — no cross-reference.
- ADM:82–87: "It belongs here instead. A PBPK model run at a dose outside
  its calibration has lost no information — the information is intact.
  What fails is that the decision is not admissible at that point."
- ADM:112–114 (Claims-Forbidden): "Do not move *evaluation outside the
  validated domain* out of the degradation table until an act exists for
  it here."

Same item classified by NID as a degradation and by ADM as an
admissibility question. A reader of NID never learns of ADM's claim.

**Whose decision:** derivable. ADM already sets the interim rule.

**Proposal (not applied):** add to NID:54 a cross-reference — classified
here provisionally; `SOUNIO-ADMISSIBILITY` argues this is admissibility,
not degradation; do not move until an act exists there.

### T5 — TENSION (derivable) — "exactly four sinks" versus a fifth

- ER:78–79: a marked value is refused "at **exactly the four** boundaries
  where a number stops being plumbing and becomes an assertion."
- ER:99–101: "A marked value is refused at all four sinks. Closing three
  is not a weaker version of this concept; it is a different concept."
- ADM:74: "**Deciding is the fifth sink.**"
- ADM:77–78 frames the four as *reporting* and the fifth as *acting*.

Not a flat contradiction — ADM gives the reading that saves both — but
ER's "exactly four" is a closed list, and a reader of ER alone is
forbidden from adding a fifth. Adding one without amending ER is a
forbidden reading of "exactly".

**Whose decision:** derivable.

**Proposal (not applied):** ER:78 becomes "four reporting sinks; deciding
is the fifth, owned by `SOUNIO-ADMISSIBILITY`." ER:99–101 keeps the
"closing three is a different concept" rule for the reporting set.

### T6 — TENSION (founder) — two proof systems, no composition rule

- JST:23–25: `because:` names a Lean theorem in `formal/`.
- JST:75–76: "The obligation is about the program. An obligation that
  requires modelling the world has been stated at the wrong layer and
  belongs to the paper."
- OV:53–54: "A language with an ontological reasoner in its compiler is
  not a language with a units table."
- OV:26–27: a term is answerable to a model of what exists.

If types are ontological terms and `because:` is a Lean theorem, which
system discharges a justification about ontological identity? JST places
world-modelling in the paper; OV places it in the compiler's reasoner.
No document says how the two proof systems compose.

**Whose decision:** founder.

**One-sentence question:** When a justification concerns whether a term
is answerable to the ontology, does `because:` name a Lean theorem, an
ontology axiom, or both — and which document owns that?

### T7 — TENSION (derivable) — the ladder applied to rewrite rules, without Reserved

- VL:93–95: "The maturity ladder applies to rewrite rules exactly as it
  applies to types: a rule is `Garden` until someone validates it, and
  `Claim-ready` when a correct transformation is proven equivalent **and**
  an incorrect one is refused."
- ML:33–48 introduces `Reserved` as a fifth state beside the ladder, then
  splits it into `reserved-owed` / `reserved-taken`.
- VL never mentions `Reserved`. An unvalidated rewrite that must not run
  is closer to `reserved-taken` (the name is taken so nothing else may
  use it; staying that way forever is correct) than to `Garden` (the
  system is passive and says nothing).

Not a contradiction: ML:12–14 states a general evidence progression, and
VL is allowed to instantiate it. The missing rung is the tension.

**Whose decision:** derivable.

**Proposal (not applied):** VL:93–95 names the unvalidated-and-unselected
rule as `reserved-taken` (or says explicitly why `Garden` is the right
rung). No founder call unless rewrite rules are not allowed on the ladder
at all.

### T8 — TENSION (derivable) — same word, "reachable"

- OV:88: "Three of seventeen gates are **reachable**."
- EL:104: `R REACHABLE` — "is there a cheap mechanical check that would
  refuse?"

OV's "reachable" means *named by a workflow*. EL's "reachable" means *a
check exists*. A reader who carries the word across documents will conclude
that three ontology gates satisfy S/G/R. They do not: being named by a
workflow is not the R question.

**Whose decision:** derivable.

**Proposal (not applied):** OV:88 says "named by a workflow", matching
OV:58's first sentence. EL keeps R as the S/G/R criterion.

### T9 — TENSION (internal to EL; leftover of C2)

- EL:124 (worked table): "14 ontology gates outside CI | ✓ | ✓ | ✓ |
  **gate** | reachability, one workflow line each."
- EL:44–51: that count is invocation; coverage unmeasured; transitive
  census in flight.
- EL:165–170 (Claims-Forbidden): "Do not treat 'add a gate' as universally
  correct." "A gate built before its criterion is understood is the
  failure this concept would otherwise cause." "The order is unchanged:
  measure, then decide, then enforce."

The footnote corrected the method. The worked example still prescribes
"one workflow line each" from the uncorrected figure. That is the same
error C2 names, one page later, inside the document that exists to
forbid it.

**Whose decision:** derivable.

**Proposal (not applied):** restated the EL:124 row as "unnamed by
workflow (direct invocation); coverage unmeasured — do not wire until
the transitive census lands."

### G1 — GAP — ABox is absent from both halves of T3

Founder's measurement (dispatch, not re-derived here): the store has a
TBox (`subclass_of` 931, `disjoint` 212, `domain` 294, `range` 2014) and
almost no ABox (`individual` 7; `abox` / `instance_of` / `class_assertion`
= zero). Neither OV nor ADM names this. If founder choice 1+2 stands,
claim-validation has no individual carrier and no concept document owns
that absence.

Not a contradiction between the fourteen. It is the missing sentence
behind T3.

### G2 — GAP — precision-narrowing row ignores an executable concept

- NID:52: "Precision narrowed (`f256` → `f64`) | — | **no act**."
- PREC:13 Status **executable**; PREC:33 "Narrowing is explicit or proven
  lossless for the stated contract."; PREC:36 "Backend failure is never
  silent fallback to `f64`."

Compatible (PREC forbids *silent* narrowing; "no act" is the missing
*named operation* for *intentional* narrowing), but a reader of NID is
told the row is unaddressed. PREC has been executable since before this
corpus was written.

**Proposal (not applied):** NID:52 cross-references PREC.

### G3 — GAP — ML attributes `i512` to the wrong sibling

- ML:93–99: `i256`, `i512`, `u256`, `u512` exist in no enum;
  "`i512` was named as the seed of the Cayley-Dickson tower
  (`SOUNIO-PRECISION-PRESERVATION`)."
- PREC never names `i256` or `i512`. Its surfaces are `f128`, `f256`,
  `dd64`, `qd128`.
- EX:67–69 is the document that names those integer widths: "The integer
  ladder stops at `i128`/`u128`. `i256` and `u256` are the widths
  Cayley-Dickson exactness needs and do not exist in `TypeKind`; `i512`
  is a declared Garden seed, not a gap."

**Proposal (not applied):** ML:99 cites `SOUNIO-EXACTNESS` (or both), not
PREC alone.

### G4 — GAP — `Knowledge<T, Origem>` versus types-as-terms, composition unspecified

- PRV:59–63: class travels in the type, `Knowledge<T, Origem>`.
- OV:26–27: types are ontological terms.
- OV:99–102: the type–ontology bridge is defined nowhere.

Is `Origem` an ontological class? Is `Knowledge` itself a term? Neither
document is required to answer yet; the composition is simply unwritten.
This is the provenance-shaped half of T2.

### G5 — GAP — `i256` / `i512` are named and unreserved

- EX:69: `i512` is a Garden seed, not a gap.
- ML:93–99: the same widths "exist in **no enum at all** — not even
  reserved. Writing `i512` today produces a generic unknown-type error,
  indistinguishable from a typo."

Compatible once "Garden seed" is a design state and "should be reserved"
is a compiler-diagnostic state. The reservation has not happened. Both
documents agree the names are owed a standing other than "unknown type".

### Language-rule note (not a concept class)

OV:22–23 still quotes the founder in Portuguese. Under the session
language rule, nothing of the specification may be written in Portuguese;
EN-UK is required on `docs/internal/concepts/**`. #1967 has now merged,
so the quote is on `main`. This audit does not edit the concept and is
not a language-only PR. The next commit that touches
`ontological-validation.md` should replace the quotation with an EN-UK
rendering.

## Claims-Forbidden cross

The gravest contradiction is a document **doing** what another document
**forbids**. Two survive that test. The rest of the CF surface is
consistent.

| reader-side claim | forbidden by | made by | class |
|---|---|---|---|
| `because:` holds provenance | JST:73–74 | ER:72 | **C1** |
| "Fourteen never run" as coverage | OV:88–90; EL:152–155 | OV:58; #1967 title; EL:124 leftover | **C2**, **T9** |
| ENIR is the compilation path today | VL:114–115 | nobody in this corpus | OK |
| the two pipeline rulings are a resolved architecture | PO:106–107 | nobody; both record the open question | OK (T1 remains open) |
| HLIR is on the default path today | PO:104–105 | nobody | OK |
| ontological validation is enforced across the language | OV:88 | OV:40 "This is not aspiration. It is built." is the nearest miss; :73–76 walks it back ("validation that does not run is not validation") | watch, not a CF hit |
| FO is closed by this concept | ER:115–117; SD:92–93 | nobody; ER:48–53 and VL:49–53 treat FO as symptom / bookkeeping | OK |
| add a gate universally | EL:165–167 | EL:124 leftover (T9), not a foreign document | T9 |
| the degradation table is complete | NID:79–80 | nobody adds a tenth row | OK (T2 is the missing row, acknowledged) |
| move the misfiled NID row before an ADM act exists | ADM:112–114 | nobody moves it | OK |

## Cited agreements (OK)

Empty OKs are omitted. Each pair below shows both lines.

1. **The mark is one mechanism.** ER:57–60 ("`k.value` does not yield a
   plain `f64`. It yields a value that remembers its uncertainty was
   discarded, and arithmetic over it stays marked.") · EX:60–61 ("An
   exact result narrowed to float loses decidability and must be marked
   as having done so.") · NID:48 ("mark propagates; see
   `SOUNIO-EPISTEMIC-ERASURE`").
2. **The two-program test is invoked consistently.** ML:106–117 (the
   table that decides a position) · ER:16–18 (Hypothesis until a correct
   program passes and a wrong program is refused) · ADM:106–108
   ("Counting files that mention a name measures reach, not the
   two-program test.") · OV:90–91 ("Counting files that mention a word
   measures reach, never the two-program test.") · VL:93–95 (the same
   test applied to rewrite rules).
3. **Program-property versus world-validity is the same cut.** JST:101–102
   ("It verifies a program property; validity is the paper's.") ·
   PREC:37 ("Higher precision alone does not establish physical
   significance.") · EX:62–63 ("Exactness does not imply significance.
   A decided zero is a fact about the algebra, not evidence about the
   world.") · matches `SEMANTIC_LANE_CONTRACT.md:71`
   (`formal model != empirical claim`).
4. **The act is the only door.** PRV:72–76 ("The constructor is the only
   door.") · NID:68–70 ("The act is the only door. If the degraded state
   is also reachable by constructing a value by hand, the act is
   decorative.").
5. **`attest` is the escape; its floor is a proof — once C1 is repaired.**
   PRV:107–108 (`attest` needs provenance) · JST:109
   (`attest(because:)` inherits this floor) · ADM:70–72 (floor is a
   discharged proof). The three agree on the *existence* of the escape;
   they disagree on the *contents* of `because:` (C1).
6. **Silence is not consent, one principle, three layers.** NID:20–21
   ("No epistemic degradation is implicit.") · ED:20–22 ("`with X`
   requires `X` to be built in, or declared in scope.") · EL:88–90
   (those severity choices "are one choice, made repeatedly: move the
   effort from the reader to the actor.").
7. **Signal direction is the mirror, not a second gate.** SD:59–60
   ("`SOUNIO-NO-IMPLICIT-DEGRADATION` says nothing may be lost in
   silence. This is its mirror: nothing may be gained in a way that
   reads as loss.") · EL:126 (worked row: `main` red 9 h on a stale
   label is S = no, verdict **no gate**) · EL:128–133 (the criterion
   declines to prescribe a gate for the loudest incident of the day
   because nothing was silent). · SD:87–88 (Claims-Forbidden: do not
   read this as a criticism of the XPASS gate).
8. **Measure, then refuse.** ED:79–83 (the refusal may not land before
   the tail census) · EL:168–170 ("Do not cite this to bypass
   measurement. The order is unchanged: measure, then decide, then
   enforce.") · ED:108–109 (Claims-Forbidden: do not read the counts as
   the full set; the tail is unmeasured).
9. **Kinds live; the coupling does not.** ADM:52–56 ("These are not
   `Garden`. They are implemented and exercised. What is missing is not
   implementation but a place where they are defined.") · ADM:14–16
   (the concept itself is Hypothesis) · ADM:106–108 (file counts are
   not Claim-ready) · consistent with ML:21–24 (each rung requires
   every rung beneath it).
10. **Intent versus current state, same pattern on both pipeline
    documents.** VL:20–21 (ruling: ENIR becomes the only path) with
    VL:114–115 (do not describe ENIR as the path today). PO:20 (ruling:
    HLIR is always on the path) with PO:104–105 (do not describe HLIR as
    on the default path). The pattern agrees; the rulings still tension
    (T1).

## Founder questions — and a stop

Four questions. Nothing in this audit answers them.

1. **T1.** Does non-epistemic CPU code descend through ENIR, or through
   HLIR → `ir/` → native with ENIR only for trusted content?
2. **T2.** Is narrowing an ontological term (`CHEBI:9168` → `f64`) a
   degradation requiring a named act, or a separate axis — and which
   document owns the type–ontology bridge?
3. **T3.** Does `Admissible<T>` also require ontological validation of
   its terms and claims, not only non-degraded epistemic input — and
   which document owns that coupling?
4. **T6.** When a justification concerns whether a term is answerable to
   the ontology, does `because:` name a Lean theorem, an ontology axiom,
   or both — and which document owns that?

## What this audit does not do

- It does not write a fifteenth concept.
- It does not edit ER, NID, OV, EL, or any other concept document.
- It does not wire a gate, reopen #1967, retitle the #1967 merge, or
  edit #1972.
- It does not treat "fourteen never run" as measured coverage.
- It does not resolve T1, T2, T3, or T6.
- It does not touch `self-hosted/`.
- It does not run a compiler.

Derivable repairs (C1, C2, T4, T5, T7, T8, T9, G2, G3) wait for a later
dispatch that is allowed to edit the concept files.

## Commands run

Reading only. No `souc`. No build.

```text
# thirteen on origin/main cfffa919ea (OV merged as #1967 during the pass)
docs/internal/concepts/{erasure,no-implicit-degradation,provenance,
  justification,precision-preservation,verified-lowering,pipeline-order,
  admissibility,MATURITY_LADDER,effect-declaration,exactness,
  signal-direction,ontological-validation}.md

# still queued
#1972 head 12760ff48b  docs/internal/concepts/effort-location.md
#     message: the gate row measured invocation, not coverage — and say so
```

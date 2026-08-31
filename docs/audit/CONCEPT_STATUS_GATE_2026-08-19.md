<!-- docs:meta
topic_id: repo.docs.audit.concept-status-gate-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.concept-status-gate-2026-08-19
-->

# Concept Status gate — both directions, no skip

**Date:** 2026-08-19  
**sha base:** `a5b48f93cc` (origin/main at lane start)  
**Gate:** `scripts/ci/concept_status_gate.sh`  
**CI wire:** `.github/workflows/ci.yml` → step *Concept status ladder (bindings evidence)* (next to Docs registry)

---

## Semantic lane declaration

```text
Semantic-Lane-ID: concept-status-gate-20260819
Owner: grok-cli4
Concept-IDs: all docs/internal/concepts/*.md contracts (Status field only; no meaning rewrite)
Intent-Preserved: concept maturity claims must be falsifiable; silence is not honesty
Transformation: gate + Status/Claims-Forbidden backfill from registry.tsv; no compiler change
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - every concept contract declares Status ∈ README vocabulary
  - hypothesis/garden + pair without complete EDNC (Reason+Owner+Date) → RED (behind)
  - malformed Evidence-Does-Not-Count → RED naming missing field (stricter than absence)
  - claim-ready without pair → RED (ahead)
  - reserved without negative evidence → RED
  - executable without witness path in bindings → RED
Claims-Forbidden:
  - Status-Held / skip escape
  - green gate on main that lacks Status on contracts
Assumptions: bindings.tsv is the concept→evidence map (not a fourth mechanism)
Write-Set: scripts/ci/concept_status_gate.sh, docs/internal/concepts/*.md,
  .github/workflows/ci.yml, docs/audit/CONCEPT_STATUS_GATE_2026-08-19.md
Read-Set: registry.tsv, bindings.tsv, README.md vocabulary
Positive-Witness: Phase D — claim-ready without pair turns gate red; revert restores green
Negative-Witness: Phase A — 13 missing Status on main → RED before backfill
Acceptance-Gate: bash scripts/ci/concept_status_gate.sh exits 0 on this branch; exits 1 on main-at-start shape
Integration-Target: ci.yml Contracts job
Authoritative-Only-If: CI step is named (not an unreferenced script)
```

---

## Phase A — gate first, must be RED on main

Vocabulary (from `docs/internal/concepts/README.md`):  
`garden | hypothesis | executable | integrated | claim-ready | reserved | superseded`

**Scope:** all `docs/internal/concepts/*.md` except meta  
`README.md`, `SEMANTIC_LANE_CONTRACT.md`, `MATURITY_LADDER.md`.

**Measured missing Status on origin/main (13 contracts):**

1. `dyadic-nonreduction.md`
2. `endogenous-observability.md`
3. `epistemic-numeric-value.md`
4. `exactness.md`
5. `explicit-discharge.md`
6. `hypercomplex-zero-divisor-evidence.md`
7. `nonassociative-order.md`
8. `policy-state-feedback.md`
9. `precision-preservation.md`
10. `proof-carrying-inference.md`
11. `reflexive-inquiry.md`
12. `relational-associator.md`
13. `zero-provenance.md`

**Canonical count: 13 of 30 contracts** lacked Status on main at Phase A.

Founder initially said fifteen of thirty-three markdown files. That counted every
`.md` under `docs/internal/concepts/`. Three are meta (`README.md`,
`MATURITY_LADDER.md`, `SEMANTIC_LANE_CONTRACT.md`) and are out of gate scope.
Correction from the founder (dispatch 2026-08-19): use **13/30**, not 15/33 —
so later readers do not find two denominators. SEMANTIC_LANE_CONTRACT’s
`Status: active` is process vocabulary, not the maturity ladder.

**Phase A command result (before backfill):**  
`CONCEPT_STATUS_GATE_RED` with `FAIL_MISSING_STATUS` ×13 (+ additional Claims-Forbidden gaps). Exit **1**. A gate that passed here would be broken.

---

## Phase B — concept → evidence mapping

**Do not invent a fourth mechanism.** Existing surface:

| file | role |
|---|---|
| `registry.tsv` | concept_id ↔ contract path ↔ registry status |
| `bindings.tsv` | concept_id ↔ path_pattern ↔ role |

**Roles that count as ladder evidence:**

| role class | binding roles (substring) | effect |
|---|---|---|
| positive witness | `positive-evidence`, `evidence`, `parallel-ontology-evidence`, `ontology-evidence`, `canonical-*`, `source-semantics`, `compiler-ir`, `formal-evidence`, … | `HAS_POS` if glob matches a real path |
| negative witness | `negative-evidence` | `HAS_NEG` |
| pair | HAS_POS ∧ HAS_NEG | protocol v3 two-program shape |
| gate | `acceptance-gate`, `gate`, `evidence-gate(s)` | `HAS_GATE` (executable witness, not a pair alone) |

**Optional fields in the concept doc** (editing the concept is the only escape):

- `Evidence-Pass:` / `Evidence-Refuse:` — explicit fixture paths  
- `Evidence-Does-Not-Count:` — founder-**accepted** escape (2026-08-19), **not** a generic Status-Held

### Evidence-Does-Not-Count — required shape

The founder rejected a free Status-Held. What is accepted is a **signed
declaration on the concept itself**, same honesty pattern as `last_validated` /
`validated_by` in `docs:meta`.

```text
Evidence-Does-Not-Count:
Reason: <non-empty, specific to the pair that does not count>
Owner:  <who signs>
Date:   <ISO YYYY-MM-DD>
```

| field | rule |
|---|---|
| **Reason** | non-empty; ≥12 chars; not vacuous (`not yet`, `tbd`, `todo`, `n/a`, …). “The pair covers only the scalar case” is a reason. |
| **Owner** | required signer |
| **Date** | ISO `YYYY-MM-DD` |

**Malformed EDNC is RED, and stricter than absence:** missing any of the three
fields (or vacuous Reason / bad Date) fails with
`FAIL_EDNC_MALFORMED missing=…` naming which field. Declaring badly must not be
cheaper than promoting Status — otherwise every lane ships a broken escape.

**Scope of the waiver:** EDNC only suppresses **behind-reality** promotion
pressure on `hypothesis` / `garden` when a pair exists. It does **not** waive
`claim-ready` without a pair, `executable` without a witness, or `reserved`
without a refuse surface.

### Expiry decision (founder closed 2026-08-19)

**Declarations do not expire. The Date stays visible.**

- No TTL, no age-based WARN, no RED for old age. A two-year declaration that
  remains true is legitimate; what is not legitimate is nobody knowing it is
  two years old.
- Age is derived from the declared `Date` field only — **not** `git log` or
  file mtime. Editing the concept for another reason must not reset the
  declaration’s age.
- Visibility is not “print on failure”. On **every** run (green or red) the
  gate emits the active roster, **oldest first**:

  ```text
  CONCEPT_STATUS_EDNC_ACTIVE count=N ...
  CONCEPT_STATUS_EDNC_ACTIVE [1/N] doc=... owner=... age_days=... date=...
  ```

- Same roster is written to two human surfaces (choice + justification):
  1. **GitHub Job Summary** (`$GITHUB_STEP_SUMMARY`) when the gate runs in
     Actions — appears on the job page without opening logs.
  2. **`docs/internal/concepts/ednc_active.tsv`** regenerated beside the
     contracts (gitignored so CI does not dirty the tree; local and runner
     FS still hold the file next to the specs).

  Logs alone fail the founder’s bar: a gate that only speaks when it fails
  hides age on every green day.

**Why not `tests/<concept-id>/` only:** many concepts already bind clinical/ontology/compile-fail globs in `bindings.tsv`. A path convention would orphan that map. bindings.tsv is the map.

---

## Bidirectional rules (the XPASS of the spec)

| Status | condition | verdict |
|---|---|---|
| missing / invalid vocab | — | RED |
| `hypothesis` / `garden` | HAS_PAIR and no `Evidence-Does-Not-Count` | **RED behind** — promote |
| `claim-ready` | ¬HAS_PAIR | **RED ahead** |
| `reserved` | ¬HAS_NEG | **RED** — need refuse surface |
| `executable` | ¬HAS_POS ∧ ¬HAS_GATE | **RED** — no witness |
| `integrated` | ¬HAS_PAIR | **RED** — needs full pair |
| missing Claims-Forbidden (non-garden) | — | RED |

No Status-Held. No skip.

---

## Phase C — the thirteen

Status **derived from `registry.tsv`**, not invented:

| doc | Status written |
|---|---|
| dyadic-nonreduction | executable |
| endogenous-observability | executable |
| epistemic-numeric-value | executable |
| exactness | hypothesis |
| explicit-discharge | executable |
| hypercomplex-zero-divisor-evidence | executable |
| nonassociative-order | executable |
| policy-state-feedback | executable |
| precision-preservation | executable |
| proof-carrying-inference | executable |
| reflexive-inquiry | executable |
| relational-associator | executable |
| zero-provenance | executable |

**Not mechanically derivable (none left in the 13):** every missing-Status contract was already a registry row. Unregistered contracts (`erasure`, `justification`, `no-implicit-degradation`, `provenance`, `verified-lowering`) already declared Status: hypothesis in-doc — no founder chit needed this round.

Also backfilled `## Claims Forbidden` stubs on contracts that had Status but no forbidden section (honesty surface founder measured at 20/33).

---

## Phase D — negative proof (required before merge)

| case | mutation | expected |
|---|---|---|
| D1 ahead | `zero-provenance` Status → `claim-ready` | EXIT 1 `ahead_of_evidence` |
| D2 no Owner | EDNC with Reason+Date only | EXIT 1 `ednc_malformed missing=Owner` |
| D3 no Date | EDNC with Reason+Owner only | EXIT 1 `ednc_malformed missing=Date` |
| D4 no Reason | EDNC with Owner+Date only | EXIT 1 `ednc_malformed missing=Reason` |
| D5 vacuous Reason | Reason: `not yet` | EXIT 1 `Reason(vacuous)` |
| D6 complete EDNC | `dyadic-nonreduction` → hypothesis + full EDNC | EXIT 0 (waiver holds) |
| D7 visibility | two well-formed EDNCs with dates 2026-06-01 and 2026-08-01 | green prints oldest-first; Job Summary table; `ednc_active.tsv` |

Each case restored before the next. Final gate EXIT 0.

Without D2–D5 the EDNC escape would be untested — the exact class of defect
this gate exists to catch.

---

## Phase E — reachability

Wired into `.github/workflows/ci.yml` immediately after **Docs registry**:

```yaml
- name: Concept status ladder (bindings evidence)
  run: bash scripts/ci/concept_status_gate.sh
```

Same class of failure as a docs registry miss: Contracts job red, merge blocked.

Coord: notified grok-cli5 (ci.yml claim) before edit.

**Contrast:** `effect_archaeology_gate.sh` / `typekind_archaeology_gate.sh` still unwired (founder measurement) — out of this PR’s scope except as the lesson that motivated naming this step in ci.yml.

---



---

## Reserved splits in two (founder 2026-08-19)

### Relation to #1965

Ladder prose for the split is owned by PR **#1965** (`MATURITY_LADDER.md`).
This PR (#1959) owns **gate enforcement** + concept README vocabulary +
Status backfill. Do not treat the two as competing ladder texts after merge.


`reserved` is no longer a single state. A promise and a marker must not read the same.

| status | means | required fields | refuse evidence |
|---|---|---|---|
| **reserved-owed** | name taken; **someone owes** the landing | `Reserved-Owner`, `Reserved-Since` (ISO), `Reserved-Blocked-On` (technical condition, not a deadline) | HAS_NEG required |
| **reserved-taken** | name taken so nobody else defines it; **owes nothing** | `Reserved-Reason` (non-empty) | HAS_NEG required |
| bare `reserved` | **invalid** | — | FAIL_BARE_RESERVED |

Malformed owed/taken is **more red than absence** (names which field is missing).

### Bare `reserved` migration

**Measured:** zero concept contracts declared `Status: reserved` when the split
landed. Migration cost is zero. The gate **rejects** bare `reserved` rather than
guessing owed vs taken — guessing would hide the distinction the founder just drew.

### Visibility (owed debts)

Same rule as EDNC: no expiry, no age-red. Every run prints
`CONCEPT_STATUS_OWED_ACTIVE` oldest-first; age from `Reserved-Since` only.
Also: Job Summary + `docs/internal/concepts/reserved_owed_active.tsv` (gitignored).

### TyF128 / TyF256 — proposal for founder (not written as fact)

Both are measured Reserved in typekind archaeology (`E218`, `tests/typekind/index.tsv`).
Both are promises under precision-preservation science surface → **`reserved-owed`**.

| kind | proposed Reserved-Blocked-On |
|---|---|
| TyF128 | **Proposal for founder (not confirmed):** end-to-end constructible `f128` (correct bind passes; wrong still refuses typed). **Measured now:** parser `error[E218]` — "f128/f256 is reserved for compiler-owned format identity; source values are unavailable in V0-A" (`self-hosted/parser/types.sio`). Fixtures `tests/typekind/f128/{pass,refuse}.sio` both fail E218. The slogan "x86-64 backend emission" is **necessary but incomplete** — refuse is at V0-A surface before lowering. |
| TyF256 | **Same proposal class** under the same E218 V0-A path. |

Owner/Since: assign when Status is attached to a concept or typekind row. **Awaiting founder confirmation of Blocked-On wording.**

### Phase D additions (reserved)

| case | expected |
|---|---|
| bare `reserved` | EXIT 1 `FAIL_BARE_RESERVED` |
| owed without Blocked-On | EXIT 1 `missing=Reserved-Blocked-On` |
| owed without Owner | EXIT 1 `missing=Reserved-Owner` |
| taken without Reason | EXIT 1 `missing=Reserved-Reason` |
| complete owed + refuse | EXIT 0 + OWED_ACTIVE roster |
| complete taken + refuse | EXIT 0 |

## Run

```bash
bash scripts/ci/concept_status_gate.sh
# CONCEPT_STATUS_GATE_GREEN
```

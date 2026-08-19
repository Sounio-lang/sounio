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
  - hypothesis/garden + pass+refuse pair without Evidence-Does-Not-Count → RED (behind)
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

Founder count was **15** without Status among **33** markdown files. This gate’s contract scope is **30** contracts (33 − 3 meta). The two-file gap is the meta set: if README + MATURITY_LADDER are counted as needing Status, founder’s 15 = 13 + 2. They are **excluded** as index/ladder vocabulary, not concept contracts. SEMANTIC_LANE_CONTRACT declares `Status: active` (process vocabulary, not ladder).

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
- `Evidence-Does-Not-Count:` — pair exists but must not promote (narrow case); **requires a reason in the concept**

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

```text
1. Set zero-provenance.md Status: claim-ready (pair incomplete: pos without neg)
2. bash scripts/ci/concept_status_gate.sh
   → EXIT 1
   → CONCEPT_STATUS_FAIL ahead_of_evidence doc=zero-provenance.md
3. Revert file
4. gate → EXIT 0
```

Receipt: this session log. Without this step the gate could be a no-op.

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

## Run

```bash
bash scripts/ci/concept_status_gate.sh
# CONCEPT_STATUS_GATE_GREEN
```

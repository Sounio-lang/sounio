<!-- docs:meta
topic_id: repo.docs.internal.concepts.ns-antigarbling-lane-20260823
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ns-antigarbling-lane-20260823
-->

# Semantic Lane Declaration — NS anti-garbling wire

Per `SEMANTIC_LANE_CONTRACT.md`. Filed (codex-authorized, msg-1787455026; N1 accepted msg-1787462805).

```text
Semantic-Lane-ID:  ns-antigarbling-wire-20260823 (session-lane canonical: claude--session-6d7a2c7b-...)
Owner:             fable-1 (agent=claude), worktree /workspace/.wt/ns-wire-20260823, branch fable/ns-wire-20260823, base 06e85a6ada
Concept-IDs:       SOUNIO-NOISE-SYMBOL (proposed); SOUNIO-ANTIGARBLING (proposed); sibling to SOUNIO-PROVENANCE (codex L5/L6)
Intent-Preserved:  uncertainty must not be fabricated; the independence assumption of uncertainty arithmetic is a checked precondition, not a silent default
Transformation:    a Knowledge value's type carries a noise-symbol-set identity; ep_add/ep_mul over operands whose source-sets are non-disjoint (or unknown) is rejected unless a proved-disjoint certificate holds
Types-Changed:     TypeEntry gains a trailing noise_set_id: i64 (interned handle; -1 unknown/top, 0 empty/deterministic, >0 interned set), AFTER provenance_id
Effects-Changed:   none (NS is a type refinement, not an effect)
IR-Changed:        none
Claims-Introduced: (CANDIDATE, pending prior-art gate) compile-time covariance-soundness of uncertainty propagation via noise-symbol-set tracking
Claims-Forbidden:  inner=0 == disjoint support (zero covariance, not disjoint support); may-alias IS the NS soundness proof (it is only the worklist mechanism); the novelty is established (candidate until the prior-art gate)
Assumptions:       noise_set_id trailing after provenance_id; interned-handle representation with a dedicated NS table module; source cap 64
Write-Set:         self-hosted/check/noise_sets.sio, self-hosted/check/types.sio, scripts/bootstrap/bootstrap_concat.sh, scripts/bootstrap/run_knowledge_bootstrap_tests.sh, tests/run-pass/ns_handle_validity.sio (N1); self-hosted/check/check.sio, self-hosted/check/epistemic.sio, scripts/ci/ns_antigarbling_gate.sh (N2/N3)
Read-Set:          check.sio Knowledge-construction and ep_add/ep_mul sites; epistemic.sio knowledge_meta_from_ty
Positive-Witness:  tests/run-pass/ns_handle_validity.sio (9/9 PASS)
Negative-Witness:  (N3) tests/compile-fail/ns_add_shared_source_rejected.sio (E230); ns_add_unknown_conservative.sio (E230)
Acceptance-Gate:   (N4) scripts/ci/ns_antigarbling_gate.sh — same-source-built sabotage
Integration-Target: base branch codex/l5-provenance-typeentry-20260822 @ 06e85a6ada (L6 tip)
Authoritative-Only-If: xai math-review passes (grok-4.6, see docs/audit/NS_N1_GROK46_MATHREVIEW_2026-08-23.md); gate green; zero regressions vs base
```

## Mandatory distinctions honoured
- **uncertainty != ignorance** — `noise_set_id = 0` (empty/deterministic) distinct from `-1` (unknown/top); unknown NEVER treated as disjoint; the TypeEntry default is `-1` (fail-closed on unseeded).
- **computational provenance != physical causality** — NS is computational source-identity; sibling to R-ORIGIN, never conflated (§24 boundary table in docs/research/2026-08-22-domain-semantics-...).
- **analogy != ontology** — the Blackwell/associator research motivates NS; the wire asserts only the operational rule.

## Diagnostic code
**E230** — NS anti-garbling; distinct from E222 (R-ORIGIN) and E224; free on base 06e85a6ada.

## Phase order (codex)
- **N1 (DONE, accepted)** representation-only: NS table module + trailing field + all TypeEntry defaults; bootstrap/source-build behaviour-neutral. Build receipt: madaros-ns-n1b.elf 100,746,094 bytes (byte-size identical to baseline; NS inert). grok-4.6 review applied (fail-closed on invalid handle + src=63 guard).
- **N2** seed at Knowledge/measure ctors; union at add/mul; unknown conservative; parametric call-summary.
- **N3** distinct diagnostic **E230** at ep_add/ep_mul; same-source-built sabotage control.
- **N4** named gate + full regression vs base.

## Draft concept contracts (proposed; registry.tsv row deferred to founder/codex ratification)
- **SOUNIO-NOISE-SYMBOL** — a value's uncertainty carries the identity-set of the independent measurement sources it derives from; interned set handle in the type; propagated by union.
- **SOUNIO-ANTIGARBLING** — combining uncertain values under independence when source-sets are non-disjoint (or unknown) fabricates precision; rejected (E230) unless proved-disjoint. Soundness anchor: docs/research/lean/SounioAntiGarblingModel.lean.

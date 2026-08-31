<!-- docs:meta
topic_id: repo.docs.internal.concepts.ns-antigarbling-lane-20260823
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ns-antigarbling-lane-20260823
-->

# Semantic Lane Declaration -- NS anti-garbling wire

Status: executable
Authority: repo-only executable compiler contract

Per `SEMANTIC_LANE_CONTRACT.md`. The noise-symbol set is the representation
mechanism for the single registered concept, `SOUNIO-ANTIGARBLING`; it is not a
separate scientific claim.

```text
Semantic-Lane-ID:  ns-antigarbling-wire-20260823
Owner:             fable-1 implementation; Codex integration review for PR #2336
Concept-IDs:       SOUNIO-ANTIGARBLING
Intent-Preserved:  uncertainty must not be fabricated; an independence assumption used by uncertainty arithmetic is a checked precondition, not a silent default
Transformation:    Knowledge types carry a noise-source-set handle; Add and Mul reject two noise-bearing operands when their sets are not provably disjoint
Types-Changed:     TypeEntry gains noise_set_id: i64 (-1 unknown/top, 0 empty/deterministic, >0 interned set)
Effects-Changed:   none
IR-Changed:        none
Claims-Introduced: the self-hosted checker has an executable, fail-closed Add/Mul refusal for shared or unknown non-empty noise-source sets
Claims-Forbidden:  novelty; full covariance soundness; physical independence; causal provenance; run-level Knowledge arithmetic; Sub/Div coverage; sign-aware covariance; a claim-ready or clinically validated result
Assumptions:       at most 64 directly representable source identities; overflow and invalid handles saturate to unknown/top; only exact single-parameter identity bodies preserve a call result handle
Write-Set:         self-hosted/check/noise_sets.sio, self-hosted/check/types.sio, self-hosted/check/check.sio, self-hosted/check/epistemic.sio, bootstrap wiring, NS witnesses, and NS gates
Read-Set:          Knowledge construction, Add/Mul checking, branch and loop joins, assignment, and direct-call projection
Positive-Witness:  tests/run-pass/ns_handle_validity.sio
Negative-Witness:  tests/compile-fail/ns_add_shared_source_rejected.sio; tests/compile-fail/ns_add_unknown_conservative.sio
Acceptance-Gate:   scripts/ci/ns_antigarbling_gate.sh
Integration-Target: current origin/main through PR #2336
Authoritative-Only-If: the named gate passes with its sabotage controls, current-source compiler construction passes, and the math-review honesty boundary is retained
```

## Executable boundary

- `measure(...)` seeds a fresh singleton source; a deterministic `Knowledge(...)`
  constructor carries the empty set.
- Add and Mul union source sets after E230 checks that two non-empty operands
  are provably disjoint. Unknown/top fails closed.
- `if`, `match`, assignment, loops, indirect calls, and unproved direct calls
  conservatively widen when an exact set cannot be preserved.
- Only a proved exact single-parameter identity body projects the selected
  argument's set through a direct call.
- `SOUNIO_NS_DISABLE=1` disables only E230 for the diagnostic negative control;
  it does not disable source propagation or the existing E245 run-level limit.
- Sub and Div remain outside E230. Their sign-dependent covariance treatment is
  future work, so this contract makes no safety claim for them.

## Mandatory distinctions preserved

- **uncertainty != ignorance**: `0` means an empty deterministic source set;
  `-1` means unknown/top and is never treated as disjoint from a non-empty set.
- **computational source identity != physical independence or causality**: the
  checker tracks a bounded static identity abstraction, not an empirical model.
- **compile refusal != runtime validation**: E245 still blocks general run-level
  Knowledge arithmetic, and an E230 witness does not establish runtime parity.
- **executable != claim-ready**: the gate proves the stated compiler behavior,
  not novelty, full covariance semantics, or scientific validation.

## Diagnostic

`E230` is the dedicated anti-garbling refusal. It is implemented at both
self-hosted binary-operation checking paths and currently covers Add and Mul.

## Integration receipt

```text
Semantic-Outcome: executable Add/Mul anti-garbling contract integrated with conservative source-set dataflow
Concept-Status-Before: proposed implementation lane
Concept-Status-After: executable
Distinctions-Added: empty/deterministic versus unknown/top; exact identity projection versus conservative call widening
Distinctions-Preserved: uncertainty versus ignorance; computational source identity versus physical causality; compile success versus runtime parity
Distinctions-Erased: none
Evidence-Run: scripts/ci/ns_antigarbling_gate.sh plus current-source compiler validation required by the PR integration gate
Fallback-Path: none; invalid, overflowed, or unresolved source information becomes unknown/top
Legacy-Kept: E245 run-level Knowledge arithmetic restriction; all non-NS checker behavior
Conflicting-Lanes: none in the declared implementation write set at integration review time
Next-Semantic-Interface: sign-aware Sub/Div treatment and run-level Knowledge lowering remain separate future lanes
```

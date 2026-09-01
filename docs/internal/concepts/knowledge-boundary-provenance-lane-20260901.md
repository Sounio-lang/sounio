<!-- docs:meta
topic_id: repo.docs.internal.concepts.knowledge-boundary-provenance-lane-20260901
authority: repo_only
audience: users
last_validated: 2026-09-01
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.knowledge-boundary-provenance-lane-20260901
-->

# Semantic lane: Knowledge boundary provenance — wire the socket

Status: proposed (awaiting founder acknowledgment before first compiler write)
Date: 2026-09-01 · Lane: kimi-cli2 / `knowledge-boundary-provenance`

## Required Lane Declaration

```text
Semantic-Lane-ID:    SOUNIO-KNOWLEDGE-BOUNDARY-PROVENANCE (proposed)
Owner:               kimi-cli2 (lane knowledge-boundary-provenance)
Concept-IDs:         SOUNIO-KNOWLEDGE-BOUNDARY-PROVENANCE (this draft);
                     adjacent: SOUNIO-ZERO-PROVENANCE, SOUNIO-ORDERED-PATH-PROVENANCE
Intent-Preserved:    a Knowledge annotation's declared validity/provenance must
                     mean something at a call boundary; no existing green
                     program may change its verdict for any OTHER reason
Transformation:      TypeEntry gains knowledge_validity_id and
                     knowledge_provenance_kind as PAYLOAD (not identity):
                     ty_knowledge persists what the parser already parsed;
                     knowledge_meta_from_ty stops rebuilding every meta as
                     always-valid/DERIVED; knowledge_call_boundary_compatible
                     (epistemic.sio:908-924) finally compares real inputs.
                     ty_eq / type identity DO NOT change: two Knowledge types
                     differing only in provenance remain the same type; the
                     boundary check reads payload, exactly like unit_id.
Types-Changed:       TypeEntry payload only (two new payload fields)
Effects-Changed:     none
IR-Changed:          none
Claims-Introduced:   "a provenance-only mismatch at a Knowledge call boundary
                     is refused by Madaros check/compile with a diagnostic that
                     names provenance"; "the covid validity-window refusal
                     exists in the semantic-clock engine, not only lean_single"
Claims-Forbidden:    provenance IDENTITY tracking (source_id stays 0; the
                     observer butterfly is NOT promoted by this lane);
                     lean_single parity (the legacy seed keeps its own
                     checker); any claim that the six-cause Unknown taxonomy
                     exists (that seed stays Garden)
Assumptions:         knowledge_meta_from_ty (epistemic.sio:497-530) is the
                     only meta source feeding the boundary check; the parser
                     already delivers ValidUntil/Valid/ValidWhile and
                     Derived/Computed/Measured to check_knowledge_type
Write-Set:           self-hosted/check/types.sio, self-hosted/check/epistemic.sio,
                     tests/audit/knowledge_*boundary* (probe updates),
                     artifacts/audit/knowledge_provenance_boundary/
Read-Set:            self-hosted/check/check.sio (call sites),
                     self-hosted/parser/types.sio (annotation AST)
Positive-Witness:    scripts/ci/knowledge_provenance_boundary_gate.sh turns
                     green — the silence probe is refused naming provenance
Negative-Witness:    covid_2020_temporal_expiration.sio still refuses
                     (Madaros now, not just lean_single);
                     knowledge_method_parity + knowledge_array stay green;
                     corpus scan (scripts/dev/corpus_failure_signature_scan.sh)
                     shows no new-failure growth beyond the known 3
Acceptance-Gate:     the accusation gate + corpus signature scan, both run
                     from-source on Slurm
Integration-Target:  self-hosted/check/ (Madaros checker) via PR to main
Authoritative-Only-If: from-source build on Slurm r770; the committed ELF
                     predates E241 and proves nothing about this surface
```

## Why this lane exists (measured, 2026-08-31/09-01)

- `knowledge_call_boundary_compatible` calls `validity_subsumes` and
  `provenance_subsumes` — and feeds both from `knowledge_meta_from_ty`, which
  rebuilds every meta as `validity_always()` / `PROVENANCE_KIND_DERIVED`
  (epistemic.sio:497-530). The comparison runs; its inputs are pre-flattened.
- Measured matrix (from-source, Slurm r770, PR #2350): at the Knowledge call
  boundary, epsilon is enforced (E036), validity is enforced only by
  lean_single, provenance is enforced nowhere.
- The accusation gate (merged, deliberately red, unwired) fails with
  "provenance-only mismatch typechecks silently in every engine leg". This
  lane is the fix the gate waits for.

## Design notes for the implementer

- Payload-not-identity is the load-bearing choice: adding provenance to type
  identity would rewrite the meaning of every existing Knowledge program;
  adding it as boundary-checked payload mirrors how `unit_id` already works
  (call-boundary refusal, no type-identity change).
- The ValidUntil enforcement becomes REAL in Madaros with this lane. The only
  in-tree carriers of `ValidUntil` are the three covid files (compile-fail
  expectations) — blast radius is measurable with the corpus scanner before
  and after.
- Error taxonomy: the refusal should get its own code (proposed E24x, next
  free after a codes census) naming provenance/validity — not E036, which
  today ambiguously covers epsilon AND (accidentally) validity.

## Evidence state

| Layer | Status |
| --- | --- |
| Hypothesis | The flattening is the single cause; unflattening makes the gate green without touching identity. |
| Executable | The accusation gate + capability controls already exist and run from-source. |
| Claim-ready | No — until the gate is green and the corpus scan shows no collateral. |

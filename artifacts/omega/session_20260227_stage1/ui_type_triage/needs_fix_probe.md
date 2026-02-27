# Needs-Fix Probe (Pinned vs Self-hosted Seeded)

Date: 2026-02-27

## Scope
- `tests/ui/type/refinement_violation.sio`
- `tests/ui/type/unit_mismatch.sio`

## Findings
1. `refinement_violation.sio`
   - pinned: parse error (`Expected ;, found 'fn'`) before refinement semantics
   - self-hosted seeded: parse errors remain dominant in this syntax shape
2. `unit_mismatch.sio`
   - pinned: resolution errors (`Undefined type: m`, `Undefined type: kg`)
   - self-hosted seeded: checker emits semantic type mismatch diagnostics on unit-typed bindings and binary op

## Interpretation
- Both tests remain correctly classified as `needs-fix` for pinned-binary-driven UI triage.
- Self-hosted seeded behavior confirms unit mismatch semantics are implemented beyond pinned limitations.

## Artifacts
- `refinement_violation.pinned.log`
- `refinement_violation.selfhost_seeded.log`
- `unit_mismatch.pinned.log`
- `unit_mismatch.selfhost_seeded.log`

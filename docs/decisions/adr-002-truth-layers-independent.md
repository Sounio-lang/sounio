<!-- docs:meta
topic_id: repo.docs.decisions.adr-002-truth-layers-independent
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-002-truth-layers-independent
-->

# ADR-002: Truth Layers Are Independent

**Status**: accepted
**Date**: 2026-03-30

## Context

M5 revealed that "compilation succeeded" is not a single boolean. A program can
have correct closure (all modules found), correct capacity (no table overflow),
yet corrupt execution (parse-noise from silent BSS overwrite). Conversely, a
program can fail at closure (missing import) while every resolved module would
parse cleanly in isolation. Conflating these layers caused weeks of misdirected
debugging — symptoms in parse output were attributed to parser bugs when the
root cause was capacity corruption two layers below.

## Decision

Six truth layers exist. Each must be established independently. Success at
layer N does NOT imply success at layer N+1.

1. **Closure Truth** — which modules resolved, in what order, any unresolved
2. **Capacity Truth** — whether byte/token/node/fn/struct/pool tables saturated
3. **Execution Truth** — which modules actually lexed and parsed successfully
4. **Verdict Truth** — whether compiler is authorized to say OK/REJECT or must
   collapse to UNKNOWN
5. **Provenance Truth** — which path produced the verdict (rebuilt direct,
   wrapper, mixed, fallback)
6. **Semantic Truth** — whether checker/kernel/boundary logic produced the
   intended meaning

## Consequences

- Every diagnostic, gate, and validation script must identify which layer it
  tests. A passing closure gate does not excuse a failing execution gate.
- The M9 matrix format (rows = fixtures, columns = layers, cells = green/red)
  becomes the standard reporting format for compiler truth.
- "Parse-noise" is retired as a diagnostic category. Failures must be attributed
  to a specific layer: capacity overflow, execution corruption, or semantic bug.
- Wrapper provenance (ADR-003) exists precisely because layers 4-6 are not yet
  trustworthy on large surfaces.

## Grounded in

- M5 investigation: `--src-cap-bytes 4MiB` eliminated capacity truncation but
  execution still failed — proving layers are independent
- M9 matrix: closure green + execution red on large surfaces
- Architecture doc: `docs/architecture/truth-layers.md`

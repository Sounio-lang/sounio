<!-- docs:meta
topic_id: repo.docs.decisions.adr-001-bundle-as-authority
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-001-bundle-as-authority
-->

# ADR-001: Bundle-as-Authority

**Status**: accepted
**Date**: 2026-03-30

## Context

During M4-M9 work, module resolution was initially treated as a debug witness —
useful for diagnostics but not authoritative. Probes showed that when resolution
is merely informational, execution-time path guessing reintroduces ambiguity:
the same source can resolve differently depending on working directory, env vars,
or file-system race conditions. The ClosureBundle output from M4 probes
(flat_direct=305, flat_transitive=316, ns_direct=111, ns_transitive=44)
demonstrated that resolution results are stable and inspectable enough to be
treated as the single source of truth for "which modules are in this compilation."

## Decision

Resolved module closure is an **authority surface**, not a debug witness.

- Resolve imports once during closure construction.
- Store resolved result in the bundle.
- Execution consumes only bundle-resolved entries.
- No execution-time path guessing or re-resolution.

The minimum per-module contract: requested_spec, resolved_path, resolution_kind,
parent_module_id, depth, src_start, src_len, requested_len, tk_start, tk_end,
exec_state, parse_status.

## Consequences

- Module closure output becomes a first-class artifact, not stderr noise.
- Validation scripts can assert on bundle fields directly.
- Any future incremental/cached compilation must respect bundle authority —
  cache invalidation is keyed on bundle identity, not file mtime.
- The monolithic lex/parse in boot4.sio can stay, but its boundaries must be
  tracked per-module (M6 ClosureBundleV2).

## Grounded in

- M4 probe results: `scripts/ci/m4_*_probe.sio`
- M6 plan: per-module token boundaries + parse_status tracking
- Architecture doc: `docs/architecture/module-closure-truth.md`

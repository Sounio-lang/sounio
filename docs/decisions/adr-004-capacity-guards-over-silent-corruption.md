<!-- docs:meta
topic_id: repo.docs.decisions.adr-004-capacity-guards-over-silent-corruption
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-004-capacity-guards-over-silent-corruption
-->

# ADR-004: Capacity Guards Over Silent Corruption

**Status**: accepted
**Date**: 2026-03-30

## Context

`add_token()` at boot4.sio:612 has NO bounds check. Writing past 524288 tokens
corrupts adjacent BSS arrays. `new_node()` at line 796 prints a warning but
continues writing past bounds. `FN_COUNT`, `ST_COUNT`, `EN_COUNT`, `FLD_COUNT`
— all increment without capacity guards. This is the root cause of "parse-noise"
on programs that exceed any single table's capacity: silent memory corruption
produces symptoms that look like parser bugs but are actually buffer overflows.

## Decision

Every fixed-capacity table gets an explicit bounds check that **stops writing**
when full, rather than silently corrupting adjacent memory.

- `add_token()`: `if TK_COUNT >= 524288 { return }`
- `new_node()`: return sentinel -1 on overflow
- `parse_fn_def()`: `if FN_COUNT >= 2048 { return }`
- Similarly for ST_COUNT (1024), EN_COUNT (256), FLD_COUNT (8192)

Silent truncation is strictly better than silent corruption:
- Truncation produces a short but valid partial result
- Corruption produces an arbitrarily wrong result that poisons downstream phases
- Truncation is detectable (saturation telemetry, ADR-002 capacity layer)
- Corruption is not reliably detectable after the fact

## Consequences

- Programs that exceed capacity get partial compilation with attributable
  failure, not corrupt output with mysterious symptoms.
- M6 saturation telemetry (MC_SAT_*, MC_OVF_*) makes overflow visible and
  attributes it to the specific module that triggered it.
- This is a safety fix, not an optimization. Capacity limits remain unchanged.
- Future capacity increases are orthogonal — guards stay regardless of table size.

## Grounded in

- Root cause analysis: boot4.sio lines 612, 796, 1673
- M5 finding: 4MiB src buffer eliminated byte truncation but node/pool
  saturation still caused corruption
- M6 plan: Step 1 (capacity guards) + Step 4 (saturation telemetry)

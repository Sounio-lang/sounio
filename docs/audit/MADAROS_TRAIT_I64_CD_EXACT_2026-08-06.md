<!-- docs:meta
topic_id: repo.docs.audit.madaros-trait-i64-cd-exact-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-trait-i64-cd-exact-2026-08-06
-->

# Madaros closeout — trait-for-i64 methods + cd_exact (2026-08-06)

**Status:** CLOSED on shipped Madaros  
**Gate:** `scripts/ci/madaros_trait_i64_cd_exact_gate.sh` → `MADAROS_TRAIT_I64_CD_EXACT_GATE_OK`

## Fixes

1. **Lowering** (`self-hosted/ir/lower.sio`): local primitive receivers keep
   scalar-kind → mangle as `i64_*` / `f64_*` instead of inventing body-less
   bare `er_add`.
2. **Checker** (`self-hosted/check/check.sio`):
   - imported impl bodies use declared `with` effects when sig lookup misses (E035);
   - unresolved type-param receivers defer method selection to monomorphization (E011).
3. Promoted tip Madaros → `bin/madaros-linux-x86_64`.

## Remeasure (item 3)

`bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh` is **GREEN**
(`ZD PROVED` / `SQ PASS` / `NONZERO PASS` / 16× `COMP i 0`). The historical
IrModule ~18 GB memory-wall claim is **not reproduced** on this tip path;
treat prior OPEN memory-wall prose as historical unless a new corpus fails.

## Non-claims

Does not claim every trait method shape on every primitive is green, nor that
compact-IR / every multi-module corpus is green.

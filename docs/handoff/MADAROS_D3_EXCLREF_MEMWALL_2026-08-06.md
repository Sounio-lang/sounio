<!-- docs:meta
topic_id: repo.docs.handoff.madaros-d3-exclref-memwall-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.madaros-d3-exclref-memwall-2026-08-06
-->

# Madaros D3 residual reclassification — 2026-08-06

## Classification

The historical blanket claim “exclusive-ref / memory-wall still red on shipped
Madaros” is **stale for the concrete exclusive-ref science witnesses**. Those
are green:

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
bash scripts/ci/madaros_d3_exclref_shipped_gate.sh
# MADAROS_D3_EXCLREF_SHIPPED_GATE_OK
```

(covers unsplit `oct_mul` + imported `associator_field`).

## Still open (typed)

### BLK-20260806-madaros-trait-i64-method-lower

```text
Blocker-ID: BLK-20260806-madaros-trait-i64-method-lower
Status: classified
Severity: B1
Class: compiler-semantics
Owner: unassigned (Madaros IR lower / trait dispatch)
Repro: single-file trait ExactRing for i64 + a.er_add(3)
Observed: souc check OK; Madaros native lower SIGSEGV at lower_array: seed_begin
Expected: ELF runs; lean_single prints ER_LOCAL_OK
Evidence-Level: E3
LLM-Offload: not-required
Next-Action: lower trait-impl methods for primitive Self the same way as
  inherent struct methods (avoid body-less mangled fn); rebuild Madaros.
```

### BLK-20260806-madaros-d3-cd-exact-e035-preflight

```text
Blocker-ID: BLK-20260806-madaros-d3-cd-exact-e035-preflight
Status: classified
Severity: B1
Class: compiler-semantics
Owner: unassigned (Madaros checker / trait effects under import)
Repro: SOUNIO_STDLIB_PATH=$PWD/stdlib bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh
Observed: E035 on er_add/er_sub/er_mul and E011 on generic F.er_* inside
  cayley_dickson_exact even after i64 ExactRing impl declares with Mut, Div, Panic
  (stdlib aligned 2026-08-06). lean_single completes ZD PROVED.
Expected: Madaros check reaches lower so IrModule memory-wall can be re-measured
Acceptance-Gate: bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh
Evidence-Level: E3
LLM-Offload: not-required
Next-Action: fix Madaros imported trait-method effect/signature binding for
  impl-for-primitive; then re-run cd_exact before any IrModule shrink work.
```

## Non-claims

- Not a claim that IrModule memory-wall is closed.
- Not a claim that all trait methods on primitives are green under Madaros.

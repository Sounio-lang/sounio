<!-- docs:meta
topic_id: repo.docs.audit.madaros-d6-const-nonmain-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-d6-const-nonmain-2026-08-06
-->

# Madaros D6 closeout — module const from non-main (2026-08-06)

**Status:** CLOSED  
**Gate:** `scripts/ci/madaros_d6_const_nonmain_gate.sh` → `MADAROS_D6_CONST_NONMAIN_GATE_OK`  
**Witness:** `tests/epistemic_trust/madaros_d6_const_nonmain.sio`

## Symptom (historical)

Module-level `const C_A: i64 = 20` read from a non-`main` local fn could resolve to a
stale local vreg under Madaros native, so `a[64 + C_A] = 1` wrote the wrong slot
(SIGSEGV on large frames). `lean_single` was correct. Filed from PGx EL+ demos.

## Fix

Scalar `IR_STRATEGY_BSS_GLOBAL` identifier reads emit `ir_load_global` reload in
`self-hosted/ir/lower.sio` (landed historically as “reload scalar BSS globals”).
Tip Madaros (post-#1673) passes the acceptance witness without further codegen.

## Non-claims

Does not close D3 exclusive-ref / memory-wall / open-slice `.len()` residuals.

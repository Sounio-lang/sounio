<!-- docs:meta
topic_id: repo.docs.audit.madaros-enir-driver-native-lower-139-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-enir-driver-native-lower-139-dispatch-2026-08-16
-->

# Madaros native compile of `enir/driver.sio` SIGSEGVs in multi-module dep lowering — dispatch

**Date:** 2026-08-16
**Filed by:** lane `fable-1` (WS-C PR1), as a follow-on separate from PR1.
**Engine:** Madaros v0.80.0 (default `bin/souc`), built this session from clean
`origin/main` `03416657fa` + the frontier `enir/**` add (WS-C PR1 WIP).
**Status:** OPEN, unscheduled. **Explicitly OUT of WS-C PR1 scope.**

## Why this is out of PR1 scope

PR1 acceptance (amended `MIR_PORT_PLAN.md` §6.1) is: (a) **seed** driver ELF,
(b) Madaros **`souc check`** green, (c) no production IR/codegen edits. All three
hold. The E1 shadow lane (`madaros_v2_e1_enir_shadow_gate.sh`) and the
`bin/madaros-enir` wrapper both build the driver under the **seed**
(`souc-build-lock.sh "$SEED" …`), which succeeds (1.47 MB ELF, `emit` rc=0).
Madaros *native* compilation of the driver is not on any PR1 or E1 path.

## Symptom

```
$ ./bin/souc compile self-hosted/enir/driver.sio -o /tmp/x.elf
imported_compile: typecheck ok
imported_compile: lower_begin
lower_array: seed_begin … seed_done
lower_array: dep_mode into_acc
lower_array: dep_begin 1
lower_array: into_acc_done 1
lower_array: arena_reset_ok module 1 … into_acc fn_before 373 fn_after 375
lower_array: dep_begin 2
Segmentation fault (status 139)
```

## Localisation (bounded)

- **Not a parse/check fault, and not the WS-C PR1 C4/C5 fixes.** `typecheck ok`
  prints before the crash; the driver `souc check` is green (verdict 0, 13
  modules). C4 (`mir_join.sio` if-expr hoist) and C5 (`source_lower.sio`
  `loop_closed` declaration) are parse/check-level edits producing valid AST —
  a codegen-stage SIGSEGV cannot originate there.
- **Locus:** the multi-module **into-acc dependency lowering** spine, on the
  **second** dependency module (`lower_array: dep_begin 2`), after the first
  dep lowered clean (`into_acc_done 1`). This is the imported-module native
  lowering path, not single-module codegen.
- **Consistent with known D3-family fragility:** multi-module memory-wall /
  exclusive-ref fragile chains in the imported-module native path
  (`docs/compiler/KNOWN_LIMITATIONS.md`, imported-module native residuals;
  `docs/audit/MADAROS_IMPORTED_MODULE_NATIVE_PATH_ESCALATION_2026-07-14.md`).
  The enir driver imports a 13-module closure with large aggregate structs
  (`EnirMirModule` carries `[EnirMirValue;128]` + `[EnirMirInstr;128]` +
  `[EnirMirProvenance;128]` + `[EnirMirObservation;64]`), a heavy dep-lower
  workload.

## Next steps (for whoever picks this up)

1. Bisect which of the driver's imported modules triggers dep #2 (drive
   `SOUNIO_MADAROS_DEP_MERGE=place` / the dep-merge env knobs; narrow to a
   single import).
2. Minimal repro: a 2-dep import chain where the second dep carries a large
   aggregate struct, native-compiled under default Madaros.
3. Decide dispatch vs KNOWN_LIMITATIONS entry once the locus is a single
   codegen site.

## AI disclosure

Symptom capture, pipeline-stage localisation, and scope classification by AI
agent (Claude, lane fable-1) under human direction, 2026-08-16, on Madaros
v0.80.0 default engine. No deep root-cause performed (out of PR1 scope; filed
for separate scheduling). GAIDeT-ICMJE 2025.

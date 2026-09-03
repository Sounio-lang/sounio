<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-a4
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-a4
-->

# WP-A4 — Madaros: wire real SRET for struct-by-value returns [Opus] (independent; own branch `fix/madaros-struct-return-runtime` off main; own PR)

## Problem (the critical-path gap)
Functions returning a struct BY VALUE segfault at runtime (rc=139) on the default Madaros engine when the struct contains an ARRAY field (e.g. `struct S { c: [i64;4], bits: i64 }` + `fn make() -> S` + field read → rc=139 even non-generic). This blocks `tests/run-pass/generic_struct_return.sio` (typechecks clean since phase 1, dies at runtime) and ultimately `cd_exact_generic_i64` (cd_* fns return structs with `[F;2048]` ≈ 16KB).

## Root cause (verified by read-only exploration — trust these anchors)
Two mechanisms exist; the real pipeline uses the weak one:
- `ir/lower.sio` never emits SRET: it lowers every `fn -> S` to plain `IrCall` + `IrReturn`. Default `IrReturn` codegen (`native/codegen_x86_linux.sio:6988-6992`) returns ONE register; `native_v2_core_emit_return_struct_into` (`:7301-7310`) handles at most 2 slots (rax/rdx). Anything wider is mis-returned → caller reads a bad aggregate address → rc=139.
- The full SysV-SRET machinery ALREADY EXISTS in codegen and is consumed correctly, but has NO producer in real lowering: `ir/ir.sio:719-720` (`is_sret`, `sret_dest_reg` fields); prologue param-shift `codegen_x86_linux.sio:5782-5801`; `IrCallSret` handler `:6955-6987` (hidden dest in rdi, args shift to rsi+); `IrReturnSret` `:6993-6997`; caller stack-args helper `:6511`; instruction builders `native_v2_ir_return_sret` `:7440` / `native_v2_ir_call_sret` `:7461`. Today its only exerciser is the hand-built witness `native_v2_fill_sret_*` (`:7502-7522`).
- NOTE the boundary: scalar multi-field returns already work by a different path — `tests/run-pass/sret_8_field_return.sio` (8×i64 fields) is GREEN (`docs/compiler/KNOWN_LIMITATIONS.md:95` records that fix). The broken axis is the ARRAY FIELD inside the returned struct, not raw size. Do not regress the scalar path.

## Implementation
In `self-hosted/ir/lower.sio`: for any fn whose return struct layout exceeds 2 i64 slots OR contains an array field — (1) mark the fn `is_sret=1` + `sret_dest_reg`; (2) at each call site allocate a caller-side destination slot and emit `IrCallSret` instead of `IrCall`; (3) in the fn body emit `IrReturnSret` instead of `IrReturn`; (4) make the callee's return expression write into the hidden-dest. Reuse `return_struct_name` metadata (`lower.sio:4285,4292` helpers) to identify affected fns; derive slot width from the struct layout the lowerer already computes for locals. Related prior art (layout derivation only, NOT wiring): commit `c4934c558` "S5 generic aggregate SRET layouts" (+119 in lower.sio, gate `scripts/dev/madaros_v2_s5_program_mir_abi_gate.sh`); `341b9be14` f128 SRET arg boundary.

## Bisect ladder (each rung exit-code-verified; green control first)
L0 control: `tests/run-pass/sret_8_field_return.sio` must be GREEN before AND after.
L1: non-generic `struct{c:[i64;2], x:i64}` returned by value, read `c[1]` + `x` → exact expected rc.
L2: non-generic `struct{c:[i64;4], bits:i64}` (the witness shape) → expected rc.
L3: same shape generic `<F>` instantiated `<i64>` (specializer makes it concrete pre-lowering, so if L2 passes L3 should too — if not, the residual is in the specializer's output shape; record, don't chase).
L4: `tests/run-pass/generic_struct_return.sio` → runs rc=0 printing `6` / `spike PASS`.
Stretch (record result either way): does `[F;2048]`-scale survive (16KB SRET)? Author a `struct{c:[i64;256], bits}` rung before trying cd_exact scale.

## Validation battery
- Ladder L0–L4 + a callee-with-args case (SRET shifts explicit args to rsi+ — verify a 3-arg fn returning a wide struct computes all args correctly; asymmetric arg values to catch shifts).
- Umbrella before/after (`native_v2_cpu_compiler_umbrella_gate.sh`): zero new reds. CHECK whether `imported_closure_boundary`/`imported_captured_closure_boundary` (pre-existing rc=139 reds) change — if they go green, record it (bonus, same family); if they stay red, that is fine (different family per `docs/audit/MADAROS_METHOD_CALL_SIGSEGV_2026-06-20.md` / `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md`).
- 12-15 diverse run-pass regression sample vs pre-change build, byte-identical — INCLUDE struct-heavy tests (`mc_struct_basic`, `array_elem_field_store`, `impl_inherent_method`, `linear_return_value`, `sret_8_field_return`).
- Canonical background doc: `docs/audit/MADAROS_SELFHOST_TYPEENV_SRET_2026-06-25.md` ("by-value large-struct copy/return (SRET) is the systemic" miscompile).

## Done criteria
Ladder green; battery green; PR merged; `docs/compiler/KNOWN_LIMITATIONS.md` updated (remove/annotate the array-field SRET limitation); scoreboard + handoff updated.

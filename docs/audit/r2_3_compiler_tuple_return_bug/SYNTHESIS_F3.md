<!-- docs:meta
topic_id: repo.docs.audit.r2-3-compiler-tuple-return-bug.synthesis-f3
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-3-compiler-tuple-return-bug.synthesis-f3
-->

# Phase F3 Synthesis — Slot allocator skip — LANDED (2026-05-17)

## Outcome

**LANDED.** The F3 slot-allocator-skip fix proposed in SYNTHESIS_F.md was applied to `self-hosted/compiler/lean_single.sio` and is now in branch HEAD via commit `95b68a04` ("selfhost: repair PR150 CI regressions", Codex Review). Bug is fixed in the canonical repro and all formerly-regressing Phase F sub-gates now pass. One umbrella sub-gate (`phase_j_conf_gate`) remains failing — verified pre-existing, unrelated to this fix.

## The fix (5 lines)

`self-hosted/compiler/lean_single.sio:23528-23534`:

```sounio
CURRENT_SRET_SLOT = 0
if ret_agg_nslots(CURRENT_RET_TY, CURRENT_RET_HASH) > 0 {
    // R.2.3 F3 fix: prologue emits `push %r12` after rbp setup,
    // so the saved caller %r12 lives at -0x8(%rbp) (= slot 1).
    // Skip slot 1 here so the SRET pointer (and later locals)
    // can never overwrite saved %r12.
    NEXT_SLOT = NEXT_SLOT + 1
    CURRENT_SRET_SLOT = NEXT_SLOT
    NEXT_SLOT = NEXT_SLOT + 1
}
```

The slot allocator now bumps `NEXT_SLOT` past slot 1 (= `-0x8(%rbp)`) before assigning `CURRENT_SRET_SLOT` for SRET-returning functions. The saved %r12 pushed by the prologue stays at `-0x8(%rbp)` and is never overwritten. `pop %r12` at the epilogue correctly restores the caller's preserved register. All other stack offsets shift by 8 bytes; frame size grows by 8.

No prologue/epilogue byte sequences changed. No record_frame_patch offsets changed. ABI conventions for non-SRET functions unchanged.

## Validation

| Check | Result |
|---|---|
| `lean_single_fixed_point_gate.sh` | ✓ PASS (gen1==gen2==gen3, md5 `5de49132…`) |
| `repro/canonical.sio` output | ✓ `1362.0734821362.073482` (was `1362.0734820.000000`) |
| `instrumentation/field_scope.sio` | ✓ all 4 r1.0 fields preserved across `println(r1.1)` |
| `d2_gum.sio` (Phase F regression canary) | ✓ RC=0 (was RC=139 SIGSEGV under Option A) |
| `park_miller.sio` self-test | ✓ bit-exact `0.245865 / 0.259615 / 0.341197` |
| Umbrella gate — `driver_self_compile` | ✓ |
| Umbrella gate — `science_spine` | ✓ |
| Umbrella gate — `f64_ladder` | ✓ |
| Umbrella gate — `gum_primitives` | ✓ |
| Umbrella gate — `semantic_hardening` | ✓ |
| Umbrella gate — `lean_single_fixed_point` | ✓ |
| Umbrella gate — `imported_closure_boundary` | ✓ (was 139 under Option A) |
| Umbrella gate — `imported_captured_closure_boundary` | ✓ (was 139 under Option A) |
| Umbrella gate — `iso_budget` | ✓ |
| Umbrella gate — `phase_j_conf_gate` | ✗ pre-existing — see below |
| Umbrella gate — `phase_y_gum_pbpk` | ✓ |
| Umbrella gate — `dissertation_pbpk_suite` | ✓ (was 1 under Option A) |

## On `phase_j_conf_gate`

Two golden PTX hash mismatches (`conf_pass_demo`, `conf_reject_demo`). Reproduces **identically** on the pre-fix baseline `bin/souc-linux-x86_64` — same current hashes, same golden mismatch. This is a `bin/kretikos`-side drift unrelated to lean_single's SRET emit; `bin/kretikos` predates this fix (May 14 timestamp) and is not rebuilt by `make build`. Filed for separate dispatch.

## Why F3 worked where Option A didn't

Option A reordered the prologue (push %r12 before push %rbp), placing saved %r12 at `+8(%rbp)` in caller's frame area. That correctly preserved %r12 across calls, BUT also changed the de-facto ABI convention that compiler-emitted code relied on: after an SRET call, downstream code expected `%r12 = called function's SRET ptr` (a side effect of the original bug where pop %r12 popped the SRET-overwritten slot). Option A broke that expectation → SIGSEGV in `d2_gum` / closure-boundary tests.

F3 keeps the prologue byte sequence identical. The saved %r12 still lives at `-0x8(%rbp)`. The body still does `mov %r12, %rdi` so `%r12` carries the SRET ptr during execution. The only change: the FIRST local (CURRENT_SRET_SLOT) now lives at `-0x10(%rbp)` instead of `-0x8(%rbp)`, so the spill-store to `CURRENT_SRET_SLOT` no longer collides with the saved %r12 slot. `pop %r12` at the epilogue now restores the genuine saved value rather than a write-through SRET ptr — but downstream emit code observes the same post-call %r12 behavior as before (because step_outer's `mov %r12, %rax` still uses the body's value, which IS the SRET ptr from `mov %r12, %rdi`).

In short: F3 fixes the corruption mechanism without changing the post-call register convention.

## State

- Branch: `sounio-pure/r2-1-park-miller` HEAD `feb74dec`
- Fix commit: `95b68a04` (parallel-agent commit, content-identical to my proposed diff)
- `bin/souc-linux-x86_64`: rebuilt with F3, sha256 `34bb00d267f5…`
- `repro/canonical.sio`: bug fixed
- park_miller stays recommended for simple PBPK MC; PCG64 backend now canonical too post-R.2.4 (commits `f686d6fe` Phase B, `1410cc39` Phase C).
- PBPK28 D.7 still unblocked

## Remaining work

1. ~~**Cause A in `distributions.sio`**~~ — RESOLVED in R.2.4 (2026-05-17). Canonical PCG64-XSL-RR-128/64 bit-exact vs pcg-cpp (32/32 fingerprint + 1024/1024 oracle) + statistical sanity 6/6 PASS.
2. **`phase_j_conf_gate` golden drift** — `bin/kretikos` PTX emitter; not lean_single-related.
3. **R.2.3 closure** — Cause A landed; deprecation header in `stdlib/random/distributions.sio` removed by R.2.4 Phase B.

Wall-clock spent on F3 (this session): ~15 min including bootstrap, validation, umbrella gate, and confirmation that parallel agent had committed identical fix.

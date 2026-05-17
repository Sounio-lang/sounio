<!-- docs:meta
topic_id: repo.docs.audit.r2-3-compiler-tuple-return-bug.synthesis-f
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-3-compiler-tuple-return-bug.synthesis-f
-->

# Phase F Synthesis — Option A applied & reverted (2026-05-17)

## Outcome

**REVERTED.** Option A passed lean_single fixed-point but introduced regressions in `imported_closure_boundary` (rc 139 SIGSEGV), `imported_captured_closure_boundary` (rc 139 SIGSEGV), and `dissertation_pbpk_suite` (rc 1). Per dispatch §7 hard constraint, immediate revert.

## What Option A did (3 edits to `self-hosted/compiler/lean_single.sio`)

1. **Prologue** (lines 23650-23656): moved `push %r12` to BEFORE `push %rbp; mov %rsp, %rbp`. Saved %r12 now lives at `+0x8(%rbp)` (caller's frame area), not `-0x8(%rbp)` (collision with first local).

2. **Explicit return epilogue** (lines 18375-18380): emit `mov %rax, %r12` BEFORE `add %rsp, frame`; emit `pop %rbp` BEFORE `pop %r12`. Mirrors prologue order.

3. **Implicit tail return epilogue** (lines 23825-23830): same change as (2).

Diff preserved at `fix/option_a_attempt.diff` (62 lines).

## What worked

- `bash scripts/ci/lean_single_fixed_point_gate.sh` **PASS** — gen1 == gen2 == gen3 bit-identical, md5 `176ced2a6034269db70a29c474090c30`. Bootstrap is self-consistent under the patch.
- `repro/canonical.sio` output: `1362.0734821362.073482` (two correct values) vs prior `1362.0734820.000000`. **Core bug fixed.**
- `instrumentation/field_scope.sio`: all 4 r1.0 fields preserved after `println(r1.1)`; previously 3 got corrupted with println's scratch. **Forensic confirmed: corruption mechanism eliminated.**
- `bin/souc-linux-x86_64` (patched) buggy byte pattern `55 48 89 e5 41 54` count: 0 (was 0 anyway — souc itself didn't trigger the path)
- Newly-compiled `field_scope.elf` buggy byte pattern count: 0 (was 3)
- `park_miller.sio` self-test: bit-exact unchanged (0.245865 / 0.259615 / 0.341197). No regression in stdlib RNG.

## What broke

`scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` — 3 of 12 sub-gates failed under the patch:

| Gate | rc | Failure mode |
|---|---|---|
| imported_closure_boundary | 139 | SIGSEGV running imported-body lowering test |
| imported_captured_closure_boundary | 139 | SIGSEGV |
| dissertation_pbpk_suite | 1 | Multiple PBPK runtimes segfault (e.g. `d2_gum.sio`) |

Direct verification: `./bin/souc compile stdlib/darwin_pbpk/pd/d2_gum.sio -o /tmp/d2.elf && /tmp/d2.elf` exits 139 under the patch; exits 0 (correct expected output) when reverted to the buggy souc. Regression is causally tied to Option A.

## Why Option A is incomplete

The buggy souc's de-facto ABI for SRET-returning functions was:
1. Caller passes %rdi = SRET buf address.
2. Callee's body uses %r12 as the SRET ptr cache (`mov r12, rdi` at prologue).
3. **Callee's pop %r12 at epilogue does NOT restore caller's %r12** — it pops the local slot that overlaps the saved-%r12 slot, which by then holds the SRET ptr (the original bug).
4. So callers learn to expect: **after an SRET call, %r12 holds the called function's SRET ptr, not the caller's preserved value.**

Some code in the compiler-emitted output appears to depend on (4). When Option A made the epilogue actually restore caller's %r12, that downstream code reads what it thinks is "the just-returned SRET ptr in %r12" but gets the caller's original %r12 instead — leading to wild pointer dereferences and SIGSEGV.

Furthermore, **non-SRET functions don't push %r12 at all** in either ABI. So even with Option A, an SRET caller that calls a non-SRET function still loses %r12 if the callee touches it. The ABI is internally inconsistent regarding %r12 preservation.

## Implications for a correct fix

A self-consistent fix needs one of:

**(F1) Always preserve %r12, always.** Push/pop %r12 in EVERY prologue/epilogue, not just SRET. Then %r12 is truly callee-saved everywhere and Option A's prologue reorder works. But: changes binary size of every function and breaks code that intentionally clobbers %r12 to communicate state across calls (such code may exist elsewhere in lean_single's emit paths).

**(F2) Don't use %r12 as SRET-ptr cache.** Reload SRET ptr from its stack slot (CURRENT_SRET_SLOT) at every use site, instead of carrying it in %r12. Then the prologue doesn't need push %r12 at all; no ABI dependency on %r12 preservation. Larger emit churn but ABI-conformant.

**(F3) Slot allocator skip.** Leave the prologue/epilogue order alone (saved %r12 stays at -0x8(%rbp)). Make the slot allocator skip -0x8(%rbp) for the SRET case — first local goes to -0x10(%rbp), so `emit_store_incoming_arg_x86(0, CURRENT_SRET_SLOT)` writes to -0x10, not -0x8. saved %r12 at -0x8 is left intact, `pop %r12` correctly restores caller's value. Smallest diff. Need to find where CURRENT_SRET_SLOT is assigned.

Option F3 looks most surgical. Option F2 is the most architecturally clean. Option F1 is largest blast radius.

## Operator decisions for the next session

1. **Pick a fix direction** from F1 / F2 / F3 (or another).
2. **Restore stdlib PCG64**? Currently `stdlib/random/distributions.sio` has a deprecation header pointing to park_miller. Even with the compiler bug fixed, **Cause A (algorithmic right-shift on signed i64) remains** — distributions.sio PCG64 would still produce incorrect output. Restoring is a separate stdlib dispatch.
3. **Park-Miller stays canonical** for thesis-bound output regardless. PBPK28 D.7 not blocked.

## State at halt

- `git checkout self-hosted/compiler/lean_single.sio` — reverted to HEAD `dbfec3e6`
- `bin/souc-linux-x86_64` md5 → `3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93` (pre-fix baseline)
- All gates re-pass at baseline (verified d2_gum returns to RC=0)
- `canonical.sio` still demonstrates the original bug (`1362.0734820.000000`)
- Fix attempt saved at `fix/option_a_attempt.diff`

Wall-clock spent on Phase F application + revert: ~25 min.

<!-- docs:meta
topic_id: repo.docs.audit.r2-3-compiler-tuple-return-bug.fix.proposed-fix
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-3-compiler-tuple-return-bug.fix.proposed-fix
-->

# R.2.3 Phase F — Proposed Fix (NOT YET APPLIED)

**Status: AWAITING OPERATOR APPROVAL before touching the bootstrap compiler.**

## Buggy code

`self-hosted/compiler/lean_single.sio:23650-23656` (prologue emission):

```sounio
// Emit prologue: push rbp; mov rbp, rsp; sub rsp, <frame>
em(0x55); em(0x48); em(0x89); em(0xe5)        // push rbp; mov rbp, rsp
// If SRET: save r12 (callee-saved), then mov r12, rdi (preserve SRET ptr)
if CURRENT_SRET_SLOT > 0 {
    em(0x41); em(0x54)  // push r12
}
em(0x48); em(0x81); em(0xec); record_frame_patch(); em32(0)  // sub rsp, frame
```

After this prologue runs (when CURRENT_SRET_SLOT > 0):
- saved %r12 sits at `-0x8(%rbp)` (immediately below the saved %rbp).
- The slot allocator then assigns the first local — typically `CURRENT_SRET_SLOT` itself — also to `-0x8(%rbp)`.
- The next emit (line 23663) `emit_store_incoming_arg_x86(0, CURRENT_SRET_SLOT)` stores %rdi to `-0x8(%rbp)`, **overwriting the just-pushed saved %r12**.

`self-hosted/compiler/lean_single.sio:18375-18380` (epilogue):

```sounio
em(0x48); em(0x81); em(0xc4); record_frame_patch(); em32(0)  // add rsp, frame
if CURRENT_SRET_SLOT > 0 {
    em(0x4c); em(0x89); em(0xe0)  // mov rax, r12 (return SRET ptr)
    em(0x41); em(0x5c)  // pop r12
}
em(0x5d); em(0xc3)  // pop rbp; ret
```

`pop %r12` then restores whatever the local-store left at `-0x8(%rbp)` (a stack address pointing into the callee's own deallocated frame), not the caller's preserved %r12. The caller proceeds with a corrupted %r12, which surfaces as the dangling-SRET-pointer bug documented in SYNTHESIS_G.md.

## Proposed fix (Option A: reorder, match ABI convention)

Push %r12 BEFORE `push %rbp` so the saved register sits at `+8(%rbp)` (in caller's frame territory), well away from any local slot.

**New prologue (23650-23656):**
```sounio
// Emit prologue: [push r12 if SRET]; push rbp; mov rbp, rsp; sub rsp, <frame>
if CURRENT_SRET_SLOT > 0 {
    em(0x41); em(0x54)  // push r12 — saved at +8(%rbp), caller's frame area
}
em(0x55); em(0x48); em(0x89); em(0xe5)        // push rbp; mov rbp, rsp
em(0x48); em(0x81); em(0xec); record_frame_patch(); em32(0)  // sub rsp, frame
```

**New epilogue (18375-18380):**
```sounio
if CURRENT_SRET_SLOT > 0 {
    em(0x4c); em(0x89); em(0xe0)  // mov rax, r12 (preserve return value BEFORE pop)
}
em(0x48); em(0x81); em(0xc4); record_frame_patch(); em32(0)  // add rsp, frame
em(0x5d)  // pop rbp
if CURRENT_SRET_SLOT > 0 {
    em(0x41); em(0x5c)  // pop r12 (restore caller's r12 — was at top of stack now)
}
em(0xc3)  // ret
```

Resulting frame layout for SRET function:
```
caller_rsp                ← return address (pushed by call)
caller_rsp - 8            ← saved caller %r12          ← +8(%rbp)
caller_rsp - 16           ← saved caller %rbp = rbp itself
caller_rsp - 16 - 8       ← first local = CURRENT_SRET_SLOT at -0x8(%rbp)  (no collision)
...
caller_rsp - 16 - frame   ← current rsp
```

No slot collision. Standard System V AMD64 ABI ordering.

## Bootstrap risk assessment

This change touches the bootstrap compiler. Build flow per `Makefile`:
1. `bin/souc-linux-x86_64` (current, buggy stage-0) compiles patched `lean_single.sio` → `gen1.elf`
2. `gen1.elf` compiles `lean_single.sio` → `gen2.elf`
3. `gen2.elf` compiles `lean_single.sio` → `gen3.elf`
4. Fixed-point check: md5(gen2.elf) == md5(gen3.elf)

Key safety observations:
- Current `bin/souc-linux-x86_64` does NOT trigger the buggy emit path for its own functions (xxd confirms 0 `55 48 89 e5 41 54` matches in `bin/souc-linux-x86_64`). So the stage-0 JIT compiles the patched source correctly.
- The patched `gen1.elf` will run lean_single.sio's *new* logic, producing prologues that match the System V ABI.
- Fixed point should still hold because the logic is deterministic and applies equally to every recompile.

Pre-commit gates (non-negotiable):
1. `bash scripts/ci/lean_single_fixed_point_gate.sh` — MUST PASS (gen2 == gen3 bit-identical)
2. `bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` — MUST PASS (6/6 sub-gates)
3. Reproducer `docs/audit/r2_3_compiler_tuple_return_bug/repro/canonical.sio` MUST emit `1362.073482` twice, not zero on second.
4. `xxd bin/souc-linux-x86_64 | grep "55 48 89 e5 41 54"` returns 0 matches AFTER rebuild.
5. Test suite check: any code returning tuples like `(Struct, scalar)` should now work; specifically `stdlib/random/distributions.sio` PCG64 path should run without Park-Miller workaround.

## Alternative (Option B): shift slot allocator

Instead of reordering, leave the prologue alone but make the slot allocator start at `-0x10(%rbp)` (skipping `-0x8`) whenever `CURRENT_SRET_SLOT > 0`. This is a smaller diff at the byte level (no change to record_frame_patch's relative position) but invasive at the slot-allocator level and requires auditing every place that assumes the first local is at `-0x8`. Option A is cleaner and ABI-conformant.

## Halt — awaiting operator decision

I will NOT modify `lean_single.sio` without explicit operator authorization. The risks are: (a) breaking lean_single fixed-point, (b) producing a stage-0 souc that can't recompile itself, (c) corrupting some other path I haven't traced.

Three decisions for the operator:

1. **Authorize Option A** (reorder prologue + epilogue, ABI-correct). Apply, rebuild via `make build`, run all gates.
2. **Authorize Option B** (slot-allocator skip). Smaller blast radius but I'd need a session to find the allocator first.
3. **Halt R.2.3 here.** Findings are complete and durable in `docs/audit/`. Park-Miller stays canonical RNG. Bug fix deferred indefinitely. PBPK28 D.7 proceeds with workaround.

Wall-clock spent on Phase F (hunting): ~25 min.

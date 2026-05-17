<!-- docs:meta
topic_id: repo.docs.audit.r2-3-compiler-tuple-return-bug.synthesis-g
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-3-compiler-tuple-return-bug.synthesis-g
-->

# Phase G Synthesis — gdb discriminator (executed 2026-05-17)

## Environment

- gdb 15.1 (installed in workspace per operator action)
- souc-linux-x86_64 sha256 `e9073f53…` (current; differs from dispatch-pinned `3cbea2b4…` which fails to compile the repro with "no main" error)
- Repro: `docs/audit/r2_3_compiler_tuple_return_bug/instrumentation/field_scope.sio` → `/tmp/fs_repro.elf`
- Bug **still reproduces** with current binary: r1.0 fields correct on first read; after `println(r1.1)`, r1.0.a/b/c become println-scratch values, r1.0.d untouched, r1.1 zeroed.

## Test 1 — +24 untouched: **CONFIRMED**

Hardware watch on `*((unsigned long*)($sret + 24))` set before f64 println call. **Did not fire during println execution.** Forensic signal from R.2.2 verified.

## Test 2 — `%rbp` clobber: **REJECTED**

`$rbp` did not change during println. `%rbp` is correctly preserved by println's push/pop pair. Hypothesis (b) ruled out.

## Test 3 — register alias: **REFRAMED → ACTUAL ROOT CAUSE**

Hypothesis (a) was "some register holding main's SRET ptr survives across println and is reused as base." The gdb trace shows something different and more specific:

### The real chain (verified end-to-end)

1. **main** allocates SRET buffer at `lea -0x90(%rbp_main), %rdi` → `%rdi = 0x7fffffff2590` (correct stack address inside main's frame).

2. **step_outer** prologue saves caller's `%r12` via `push %r12`, then sets `%r12 = %rdi = 0x7fffffff2590`. Good.

3. **step_outer** body calls **step_inner**. Right before the call: `%r12 = 0x7fffffff2590`. ✓

4. **step_inner** prologue at PC `0x4010df`:
   ```
   0x4010df: push %rbp           ; bytes 55
   0x4010e0: mov  %rsp, %rbp     ; bytes 48 89 e5
   0x4010e3: push %r12           ; bytes 41 54   ← saves caller %r12 at rsp = rbp-8
   0x4010e5: sub  $0x90, %rsp    ; reserve locals
   0x4010ec: mov  %rdi, %rax
   0x4010ef: mov  %rax, -0x8(%rbp) ; ← OVERWRITES saved %r12 slot with step_inner's SRET arg
   0x4010f6: mov  %rdi, %r12     ; step_inner's own %r12 set to its SRET ptr (0x7fffffff2410)
   ```

5. **step_inner** body computes and writes 5 fields to `*(%r12)` via the SRET pointer at 0x7fffffff2410. Then epilogue:
   ```
   0x4012dc: add  $0x90, %rsp
   0x4012e3: mov  %r12, %rax
   0x4012e6: pop  %r12            ; restores the OVERWRITTEN slot = 0x7fffffff2410, NOT 0x7fffffff2590
   0x4012e8: pop  %rbp
   0x4012e9: ret
   ```

6. **Right after step_inner returns**: `%r12 = 0x7fffffff2410` (step_inner's address, leaked into the caller). Verified by gdb stepping break-before/break-after the `call`.

7. **step_outer** continues, but its own `%r12` is now `0x7fffffff2410`. The compiler relies on `%r12` holding the caller-passed SRET ptr through to step_outer's epilogue:
   ```
   0x4014e0: mov  %r12, %rax    ; ← rax = 0x7fffffff2410, NOT main's 0x7fffffff2590
   0x4014e3: pop  %r12          ; pops main's saved %r12 — fine, but too late: rax already wrong
   0x4014e5: pop  %rbp
   0x4014e6: ret
   ```

8. **main** saves `%rax` to `-0x98(%rbp_main)` — captures the DANGLING POINTER `0x7fffffff2410` into step_inner's deallocated frame.

9. **main** does 4 inline-itoa prints (read SRET buffer via the dangling pointer — still works because nothing has reused that stack region yet).

10. **main** calls f64 `println(r1.1)`. println's frame extends downward and **completely overlaps** the dangling region at 0x7fffffff2410..0x7fffffff2440. println's locals at println-relative `-0x30/-0x38/-0x40/-0x48/-0x50` happen to map to main-SRET-buffer `+32/+24/+16/+8/+0`. The "+24 untouched" forensic is just println happening to not write its `-0x38` slot.

## Root cause statement (one sentence)

**The Sounio prologue emitter orders `push %r12` AFTER `push %rbp; mov %rsp, %rbp`, placing the saved %r12 at `-0x8(%rbp)` — which is exactly the slot the local-variable allocator assigns to the function's first stack-spilled value, so the first local store overwrites the saved register, and the matching `pop %r12` restores garbage instead of the caller's preserved value, corrupting any SRET-pointer chain that crosses such a call.**

## What this is NOT

- Not a tuple-return-specific bug. Any function that (1) uses %r12 as a callee-saved register AND (2) has at least one local at `-0x8(%rbp)` will leak a wrong %r12 to its caller. Tuple-return f64 happens to be the visible path because step_outer caches its SRET ptr in %r12 across an inner call.
- Not an issue in the `frame.sio` source on HEAD. `emit_prologue_with_preg_mask` (line 121) emits `push %r12` BEFORE `push %rbp`, putting saved-%r12 at `+8(%rbp)` (caller's frame area, safe). So **the buggy binary was built with an older, different prologue emitter** — not the one currently in `frame.sio`. The bug lives in whatever the pinned `bin/souc-linux-x86_64` was built from.
- Not gdb-debuggable further from this session — root cause is in compiler source we now need to locate.

## Halt point — operator decision needed

Before Phase F (fix), three sub-decisions:

1. **Locate the actual buggy prologue emitter.** Candidates to grep next session:
   - `self-hosted/native/codegen.sio` — uses `emit_prologue_with_preg_mask` (correct)
   - `self-hosted/native/codegen_x86_linux.sio` — uses same (correct)
   - `self-hosted/compiler/native_compile_driver.sio` — older self-hosted driver (line ~6082 emits `push_rbp` then `mov_rbp_rsp` then `sub_rsp_imm32` directly — but does NOT emit `push %r12` at all in `drv_begin_function`). May emit %r12 elsewhere via a different path.
   - The current `bin/souc-linux-x86_64` may have been built from an older fork of one of these that has the buggy ordering.

2. **Rebuild plan.** If the buggy emitter is in self-hosted/compiler/native_compile_driver.sio, fix needs to land there. If it's in a snapshot used to bootstrap `bin/souc-linux-x86_64`, we need to re-bootstrap from the corrected self-hosted code. Either way: lean_single fixed-point gate is non-negotiable post-fix.

3. **Park-Miller stance.** The bug is broader than tuple-return — it affects ANY function using %r12 with a `-0x8` local. Many stdlib paths likely already use the workaround pattern (locals start at `-0x10` not `-0x8`) implicitly. Once fixed, audit needed for any code that explicitly assumed the broken behavior.

## Artifacts in `gdb_session/`

- `discriminator.gdb` + `discriminator.log` — Test 1 + 2 combined
- `d2.gdb` + `d2.log` — repaired SRET-address discovery
- `d3.gdb` — writer-state inspection at corruption moment
- (this file) `SYNTHESIS_G.md`

Wall-clock spent on Phase G: ~30 min after gdb came online.

**HALT.** Operator decision required for Phase F scope (fix native_compile_driver.sio vs find true emitter) and bootstrap-rebuild plan.

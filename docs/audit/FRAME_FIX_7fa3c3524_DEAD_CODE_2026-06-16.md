<!-- docs:meta
topic_id: repo.docs.audit.frame-fix-7fa3c3524-dead-code-2026-06-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.frame-fix-7fa3c3524-dead-code-2026-06-16
-->

# Forensic audit: frame-size fix `7fa3c3524` patches dead code (no-op)

**Date:** 2026-06-16
**Auditor:** Claude (Opus 4.8), worktree `madaros-default`
**Subject:** commit `7fa3c3524` — *"fix(codegen): dynamic frame size in native_v2_core_begin_function_from_ir_into"* (on `main`)
**Verdict:** The fix is a **no-op**. It patches a function with zero call sites. The reported bug, if real, lives at a *different, live* code site. Validation of the fix as written is therefore vacuous; the original SLURM "validation" run was confounded by an independently broken `main` substrate.

---

## 1. What the commit claims

`7fa3c3524` replaced, in `self-hosted/native/codegen_x86_linux.sio`:

```sounio
pub fn native_v2_core_begin_function_from_ir_into(nc, func, fn_index) {
    ...
-   nc_emit_sub_rsp_imm32(nc, 512)
+   let frame_sz = align16((*func).reg_count * 8)
+   if frame_sz > 0 { nc_emit_sub_rsp_imm32(nc, frame_sz) }
}
```

Rationale (commit message): the fixed 512-byte frame overflowed for functions
with >64 IR temps, corrupting the caller's stack → SIGSEGV; make it dynamic.

## 2. The function it patched is never called

```
$ git grep -n "native_v2_core_begin_function_from_ir_into" -- '*.sio'
self-hosted/compiler/native_compile_driver.sio:34:    native_v2_core_begin_function_from_ir_into,   # import only
self-hosted/native/codegen_x86_linux.sio:6562:pub fn native_v2_core_begin_function_from_ir_into(...)  # definition only
```

Only an (unused) import and the definition. **No call site anywhere in the
tree.** The patched function is dead/vestigial code.

## 3. The live frame logic — two real paths, neither touched by the fix

**(a) v2-core IR emitter (live).** `compile_ir_function_v2_from_ir_into`
(codegen_x86_linux.sio:6724) → `compile_ir_function_v2_core_ir_into`
(:6181, called at :6761). Its prologue at **line 6190** still hardcodes:

```sounio
nc_emit_sub_rsp_imm32(nc, 512)   # UNFIXED — this is the live twin of the dead 6569 site
```

This is a near-identical copy of the dead function. If the "512 overflow" bug is
real, **this is where it lives** (plus the other live 512 sites: :6926
`native_v2_emit_sret_witness_main_into`, :6967 `native_v2_emit_sret_witness_make_into`,
:7330 `native_v2_core_begin_fn_spill_into`).

**(b) driver per-function emitter (live, and already correct).**
`compile_ufn_from_globals` (native_compile_driver.sio:7574, called :7763/:8055) →
`drv_begin_function` (:6776) emits a placeholder `sub rsp, 262144`, later patched by
`drv_patch_frame_size` (:5977) to:

```sounio
let raw = (vreg_count + 1) * 8        # vreg_count = V2_NEXT_REG
let frame = ((raw + 15) / 16) * 16    # align16
```

This is the **correct** dynamic frame: `(vreg_count+1)*8` matches the slot model
`ir_slot_offset(v) = -(v+1)*8` (frame.sio:426) with one slot of headroom. The driver
path was never broken.

## 4. Disassembly proof (clean-main Madaros, SLURM job 4204)

A trivial `hello` (`fn main(){ println("hi"); 0 }`) compiled by clean-main Madaros,
exec segment disassembled (no section headers; raw segment at vaddr 0x401000):

```
401000 <main>: push %rbp; mov %rsp,%rbp; sub $0x200,%rsp   # 512, only -0x8/-0x10/-0x18 used
```

`main` uses 3 slots (24 bytes) inside a 512-byte frame → **the frame is not the
problem for hello.** The `sub $0x200` is the unfixed live 512 site (§3a), unaffected
by the patch (which would have emitted `align16(reg_count*8)`). Confirms the fix does
not reach normal compilation.

## 5. Why the SLURM "validation" looked like a failure (it was confounded)

SLURM job 4204 rebuilt Madaros from a **clean `git archive main`** (build inputs from
committed main; reproducer from working tree — cpu-ops partition was de-registered,
job retargeted to the proven gpu-orangefs path, node-pinned r770).

Observed:
- Madaros build `RC=0` **but** the build log contained real type errors:
  `E001 Type mismatch` at `module_frontend.sio:3809` & `:3829`, and a `Mut borrow`
  error at `kaxi_backend.sio:1803`. (Seed `lean_single` emits a binary despite errors.)
- Emitted `hello.elf` **SIGSEGVs at runtime** (`rc=139`, core dumped).
- Reproducer compile **crashes Madaros** (`rc=139`).
- N=2/4/5 "`ir_summary_failed`" results were a **harness artifact**: the N-variant
  generator used `sed` with raw newlines in the replacement, producing **empty**
  `repro_n*.sio` files (verified: `wc -l repro_n2.sio` = 0). Only N=1 was real data.

These failures are **independent of `7fa3c3524`** — proven, since the patched
function is dead (§2) and hello's crash is not frame-related (§4: 24 bytes used in a
512-byte frame). Their **root cause is not pinned**: hello's faulting instruction was
not localized, and the E001 sits in a *multi-module* path that a single-file hello may
not exercise. The failures are *consistent with* clean `main` being mid-refactor (the
recent `9e19da1a9` "bypass text-scanning IR path" left `module_frontend.sio` in a state
that emits E001 at :3809/:3829; the operator's **uncommitted** working-tree WIP deletes
exactly that `~3808` block) — but "the WIP fixes the crash" is an inference, not a
verified claim. To convert it to causation: one ephemeral Madaros build using the
operator's working-tree `module_frontend.sio` as a read-only snapshot (hello runs →
confirmed; hello still cores → breakage is elsewhere). Not done here.

## 6. Conclusions & recommendations

1. **`7fa3c3524` is a no-op.** It edits an uncalled function. It cannot fix the bug
   it describes. It should not be cited as a landed fix.
2. **The correct fix site is undetermined** and must not be guessed. There are two
   *live* per-function emitters: `compile_ir_function_v2_core_ir_into` (line 6190,
   fixed 512) **and** the driver's `drv_begin_function` (dynamic patch, already
   correct). Hello was confirmed to route through 6190, but the *failing reproducer's*
   large `main` was **not** traced — it may route through the driver path, in which case
   the original "512 overflow" may not reproduce in production at all. Determining the
   site requires (a) substrate repair so the reproducer compiles, then (b) tracing which
   emitter its overflowing function uses. **Candidate leads** (not the answer): the
   fixed-512 sites `compile_ir_function_v2_core_ir_into:6190`,
   `native_v2_emit_sret_witness_main_into:6926`, `native_v2_emit_sret_witness_make_into:6967`,
   `native_v2_core_begin_fn_spill_into:7330`. Any patch should mirror the driver's
   already-correct `align16((vreg_count+1)*8)` — **not** `reg_count*8` (omits the `+1`
   headroom, ignores wide-int scratch `dst_vreg+limbs+1`, is `≤` what's needed). Do
   **not** use `nc_min_frame_size` (lower_ir.sio:133): it returns `≤ reg_count*8`,
   returns 0 when `reg_count==0`, and caps its scan at `reg_count`.
3. **The frame fix is unvalidatable on clean `main`** because the substrate is
   independently broken (hello SIGSEGVs). Any real validation requires a Madaros built
   on the operator's front-half-fixed substrate (the uncommitted `module_frontend.sio`
   WIP), or that WIP committed first.
4. **Forensic-dispatch protocol applies** (self-hosted compiler): proposing the
   corrected frame patch + a real validation needs operator sign-off; not done here.

## 7. Reproduction

- SLURM submit (gpu variant): `slurm-jobs/madaros-frame-fix/submit_gpu.sh`
- Results: `slurm-jobs/madaros-frame-fix/results/madaros-frame-fix-gpu-20260616T135242/`
- Evidence greps: §2–§3 above (`git grep`, `sed -n`), disasm §4 (objdump of exec segment).

# Proposal: the correct frame-overflow fix (pending validation)

**Date:** 2026-06-16
**Companion to:** `FRAME_FIX_7fa3c3524_DEAD_CODE_2026-06-16.md`
**Status:** PROPOSAL — not applied. The fix site depends on facts that cannot be
established until clean `main`'s front-half is repaired (it currently SIGSEGVs on
trivial input). Do not commit a codegen patch from this proposal without the
validation gate in §4.

---

## 1. What the bug actually is (restated)

Several **live** prologue emitters in `self-hosted/native/codegen_x86_linux.sio`
reserve a fixed 512-byte stack frame regardless of how many slots the function uses:

| Line | Function | Status |
|---|---|---|
| 6190 | `compile_ir_function_v2_core_ir_into` | LIVE (via `compile_ir_function_v2_from_ir_into`:6724→:6761) |
| 6926 | `native_v2_emit_sret_witness_main_into` | live (witness) |
| 6967 | `native_v2_emit_sret_witness_make_into` | live (witness) |
| 7330 | `native_v2_core_begin_fn_spill_into` | live (spill path) |
| ~~6569~~ | ~~`native_v2_core_begin_function_from_ir_into`~~ | **DEAD** (the one `7fa3c3524` patched; reverted) |

A function whose slots exceed `512/8 = 64` overruns its frame and corrupts the
caller's stack. `ir_slot_offset(v) = -(v+1)*8` (frame.sio:426), so slot `v` lives at
`[rbp-(v+1)*8]`; a function using vregs `0..N-1` needs at least `(N+1)*8` bytes.

The **driver** per-function path (`compile_ufn_from_globals` → `drv_begin_function`:6776
→ `drv_patch_frame_size`:5977) computes the right *shape*: it emits a placeholder and
patches the frame to `align16((vreg_count+1)*8)`. (Not observed in disasm — hello's
`sub $0x200` came from the 6190 path, not the driver — and the §3 wide-int hazard is
latent here too.) The fix should make the live IR emitters size frames the same way.

## 2. Proposed change

At the live IR emitter (primary: line 6190), replace:

```sounio
    nc_emit_sub_rsp_imm32(nc, 512)
```

with a slot-derived frame, mirroring the driver's shape (with one slot of headroom):

```sounio
    let frame_sz = align16(((*func).reg_count + 1) * 8)
    if frame_sz > 0 {
        nc_emit_sub_rsp_imm32(nc, frame_sz)
    }
```

`align16` is in scope (`use native::frame::*`). Apply the same to the witness/spill
sites (6926/6967/7330) only if they emit frames for real user functions (verify per
site; witnesses may have fixed, known slot counts where 512 is deliberate).

### On the reverted commit's formula — it was location-wrong, not value-wrong
`7fa3c3524` used `align16((*func).reg_count * 8)`. Deriving from the slot model
`ir_slot_offset(v) = -(v+1)*8`: a function using vregs `0..reg_count-1` has its deepest
slot at `-reg_count*8`, occupying `[rbp-reg_count*8, rbp-reg_count*8+8)`. A frame of
`reg_count*8` sets `rsp = rbp-reg_count*8`, so that slot is `[rsp, rsp+8)` — **inside**
the frame. So `reg_count*8` is *exactly sufficient* for plain vregs; the reverted
formula's only fatal defect was being applied to **dead code**. Had it landed on the
live emitter (6190) it would have fixed the plain-vreg overflow. The driver's `+1` is a
slot of headroom, not a correctness requirement — keep it as cheap insurance, but the
**real** residual under-allocation risk (for both `reg_count*8` and `(reg_count+1)*8`)
is the wide-int scratch in §3, which is why a scan-for-true-max-slot fix (§3 option A)
is the robust choice over any `reg_count`-derived constant.

### Why NOT `nc_min_frame_size`
`nc_min_frame_size` (lower_ir.sio:133) belongs to the register-allocating path: it
returns `align16((max_stack_vreg+1)*8) ≤ align16(reg_count*8)`, returns `0` when
`reg_count==0`, and caps its scan at `reg_count`. It cannot size the all-spill core-IR
path and cannot cure an under-allocation.

## 3. Known caveat — wide-int scratch slots (verify before relying on `reg_count`)

`nc_wide_mul_into` / `nc_wide_*` (codegen_x86_linux.sio:~2144) address scratch slots at
`dst_vreg + limbs` and `dst_vreg + limbs + 1` — **indices that may exceed `reg_count`**
if the IR builder does not reserve them. For any function using `i128`/`u128` ops,
`align16((reg_count+1)*8)` may still under-allocate.

Two robust options (pick after measuring whether `reg_count` already covers scratch):
- **(A)** Scan `(*func).instrs` for the true maximum addressed slot — including wide-int
  scratch `dst_vreg+limbs+1` — and size to `align16((max_slot+1)*8)`.
- **(B)** Have the IR builder bump `reg_count` to include wide-int scratch at the point
  the wide op is created, so `(reg_count+1)*8` is sufficient everywhere.

Note: this hazard is **latent in the driver path too** (`drv_patch_frame_size` also uses
`(vreg_count+1)*8`); it apparently works because production code exercising these
emitters has not hit a wide-int function near the slot ceiling. Worth a separate audit.

## 4. The fix SITE is undetermined — resolve before patching

Hello routes through the line-6190 emitter (`sub $0x200` confirmed by disasm). The
**failing reproducer's large `main` was not traced** — it may route through the
already-correct driver path, in which case the original "512 overflow" does not
reproduce in production and the 6190 sites are unreachable for ordinary user functions.

Required before committing any patch:
1. **Repair the substrate.** Clean `main` SIGSEGVs on trivial input (E001 in
   `module_frontend.sio:3809/3829`; see companion audit). Land the operator's front-half
   WIP (or use it as a read-only snapshot) so Madaros compiles the reproducer at all.
2. **Trace routing.** Determine which emitter the reproducer's overflowing function
   uses (6190 vs driver). Patch only the emitter actually on its path.

## 4b. Validation attempt 2026-06-16 (SLURM 4212, r770) — BUG CONFIRMED REAL; fix UNVALIDATED (confounded)

Run on the operator's read-only working-tree snapshot (front-half WIP **plus** the
live-site frame fix already applied at 6190 & 7333). A/B = working tree as-is (V_fix)
vs. the two live sites reverted to `sub rsp,512` (V_base).

**Result — the frame A/B is CONFOUNDED, not a verdict:**
- Both V_base and V_fix build a Madaros that emits a **core-dumping hello** and
  **SIGSEGVs while compiling the reproducer** (rc=139, 0 front-half trace lines). The
  fix is therefore **never exercised** (reproducer dies before codegen) and hello's
  `main` uses 24 bytes (far below the 64-slot ceiling the fix changes) — so
  `V_base ≡ V_fix` is *expected* and says nothing about the fix.
- Cause is upstream of the fix: the **seed cannot self-build a working Madaros from
  current source**. Source carries `E001` at `module_frontend.sio:3809/3829`
  (`module_frontend_lower_program_box_traced(&programs[i], …)` arg-type mismatch),
  present in **both clean main and the WIP** (the WIP does not touch those lines). The
  seed tolerates it (rc=0) and emits a Madaros with a broken front-half.
- Environment is fine: the known-good seed builds a `hello` that **runs** on the same
  worker (`run_rc=0`). Per memory, Madaros was GREEN at `17d1157be` (2026-06-14), so
  this is most likely a recent regression (candidate: `9e19da1a9` "bypass text-scanning
  IR path", the module_frontend toucher) — *not* asserted, operator's domain.

**The bug itself IS real** — reproduced on the era-matched backup working Madaros
(`artifacts/self-hosted/madaros.backup.20260616`, pre-fix), exactly matching the
original report: N=1 `trail=5` PASS; N=2 `trail=1`, N=4 `trail=15` (wrong); N=5 runtime
SIGSEGV. Classic frame overflow: `main`'s slots grow with N past the 512-byte frame.

**Conclusion:** the fix targets a genuine defect at the right (live) site, but it is
**unvalidatable until the self-build regression is repaired** — there is no way to
produce a *working* Madaros that also carries the fix from current source. The clean
validation path is: rebuild at the last-known-green commit (`17d1157be`) with the
live-site fix cherry-picked, then run §5 — an operator decision, not started here.

## 5. Validation gate (must pass before the fix is called validated)

On a front-half-fixed substrate, via SLURM (`slurm-jobs/madaros-frame-fix/submit_gpu.sh`,
cpu-ops is down → gpu-orangefs r770):
1. **A/B build:** Madaros with the patch vs. without, from identical payload.
2. **Generate N-variants correctly** — use `awk` to replicate the call line, NOT `sed`
   with raw newlines (the original run's `sed` produced empty files → false
   `ir_summary_failed`).
3. **Compile AND run** `hello` + reproducer at N=1/2/4/5.
   - Without patch: reproducer should exhibit the overflow symptom (wrong `trail`, or
     SIGSEGV at the larger N).
   - With patch: every variant prints `pass=1 trail=5 conflict=1`, and `hello` still runs.
4. **No regression:** existing gates (`make madaros-full-gate`, fixed-point gen2==gen3)
   stay green; the patch must not shrink a frame some function legitimately relied on.

<!-- docs:meta
topic_id: repo.docs.audit.frame-fix-validation-2026-06-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.frame-fix-validation-2026-06-16
-->

# Frame-Fix Validation — compile_ir_function_v2_core_ir_into
**Date:** 2026-06-16  
**Commit:** 34bf3232e (fix(check+codegen): int-literal narrowing, enum binding, dynamic frames, kaxi ref)  
**Prior dead-code commit:** 7fa3c3524 (fixed `native_v2_core_begin_function_from_ir_into`, zero call-sites)

---

## Bug

`compile_ir_function_v2_core_ir_into` (the production path called by
`compile_native_v2_preview_to_file` for every IR function) emitted:

```asm
push rbp
mov  rbp, rsp
sub  rsp, 0x200   ; ← hardcoded 512 bytes for ALL functions
```

Virtual register slots are at `rbp - (vreg+1)*8`, so the frame must be at least
`reg_count * 8` bytes. Any function with `reg_count > 64` writes past the frame.

## Evidence (disassembly of backup Madaros — built without fix)

Reproducer: `examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio`

| Variant | fn (`main`) | reg_count | bytes needed | frame allocated | overflow |
|---------|-------------|-----------|--------------|-----------------|---------- |
| N=1     | fn@0x403142 | 144       | 1152 B       | 512 B           | **640 B** |
| N=2     | fn@0x403142 | 148       | 1184 B       | 512 B           | **672 B** |
| N=4     | fn@0x403142 | 156       | 1248 B       | 512 B           | **736 B** |

Three other functions (`k6_edge_u`, `k6_edge_v`, `pri`) also overflow (87–91 regs,
deepest 696–728 B against 512-B frame).

## Observable failure

| Variant | Expected output         | Actual output (no fix) |
|---------|-------------------------|------------------------|
| N=1     | pass=1 trail=5 conflict=1 | pass=1 trail=5 conflict=1 ← accidentally correct |
| N=2     | pass=1 trail=5 conflict=1 | pass=1 trail=1 conflict=1 ← **WRONG** |
| N=4     | pass=1 trail=5 conflict=1 | pass=1 trail=15 conflict=1 ← **WRONG** |
| N=5     | pass=1 trail=5 conflict=1 | SIGSEGV (rc=139) |

N=1 produces correct output despite a 640-byte overflow because the overflow
lands in previously-allocated stack space and does not happen to corrupt critical
dom-array slots before the first inter-function call clobbers them.

N=2 adds 4 more vregs from a second `emit_cube_assignment` call, shifting critical
dom/loop slots deeper. When `k6_edge_u(e)` is called inside the while loop, its
own frame (120 bytes at its RBP) overlaps main's overflow region (starting at
main's RBP-512), overwriting vregs 64+ and corrupting dom or loop variables →
wrong trail count.

N=5: overflow exceeds the red zone + neighboring page → SIGSEGV.

## Fix

```sio
// compile_ir_function_v2_core_ir_into (codegen_x86_linux.sio:6187)
let frame_sz = align16((*func).reg_count * 8)
if frame_sz > 0 {
    nc_emit_sub_rsp_imm32(nc, frame_sz)
}
```

`align16(reg_count * 8)` covers exactly slot `rbp - reg_count*8` (the deepest),
rounded up to the required 16-byte stack alignment. For a function with
reg_count=148, this allocates 1184 bytes instead of 512.

The same fix was applied to `native_v2_core_begin_fn_spill_into` (imported in
native_compile_driver.sio but never called in current build paths).

## SLURM validation — GREEN (job 4283, 2026-06-17)

```
=== N=1: compile_rc=0  output: pass=1 trail=5 conflict=1  VERDICT: PASS
=== N=2: compile_rc=0  output: pass=1 trail=5 conflict=1  VERDICT: PASS
=== N=4: compile_rc=0  output: pass=1 trail=5 conflict=1  VERDICT: PASS
=== N=5: compile_rc=0  output: pass=1 trail=5 conflict=1  VERDICT: PASS
```

`lean_single` → Madaros (94517600B) → compile reproducer → run on `cpuops-t560-proxmox`.
All four N variants pass end-to-end on the cluster.

**Root cause of prior rc=139 failures:** Madaros's recursive expression-compiler has
multi-MB stack frames (build warns "stack frame too large — use global arrays"). The
reproducer's arrays+loops trigger deep recursion that exhausts the default 12.5 MB
stack limit. This was misread as a `lean_single` miscompile. Fix: `ulimit -s unlimited`
before invoking Madaros (added to submit.sh). The proper long-term remedy is moving
large codegen scratch locals to globals.

## Caveats

- `check.sio` changes in 34bf3232e (enum-variant binding + int-literal narrowing)
  were bundled with the frame fix on the theory that they fixed the N=1 SLURM
  crash. Post-analysis shows the N=1 crash was the lean_single miscompile, not a
  checker issue. The checker changes are independently correct but were shipped on
  a wrong rationale.
- The `compile_ir_function_v2_core_ir_into` fix assumes `IrFunction.reg_count`
  accounts for every vreg that receives a stack slot. The disassembly confirms
  this: for N=2, reg_count=148 and deepest slot = 148*8 = 1184, exactly matching.

<!-- docs:meta
topic_id: repo.docs.handoff.continuity.a8-crossmod-forwarding-miscompile
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.a8-crossmod-forwarding-miscompile
-->

# A8 — Madaros imported-lane cross-module forwarding miscompile (SIGSEGV)

## RESOLUTION (2026-07-06) — FIXED in `ir_module_finalize_merged_calls`

**The runtime miscompile is FIXED.** The earlier diagnosis below (Defect 1 = "body lowering
desyncs the `var=<call>` binding") was **falsified** by a pre/post-finalize IR differential on the
current base branch: the merged-IR body lowering is *correct* pre-finalize
(`small_basis_fwd i0: Call dst=2 arg=[0] fn=9(small_zero)`), and the exact corruption the doc
described (`dst=3 arg=[2] fn=5`) is introduced by the **finalize** step — specifically the
two-pass call-target-resolution loop at the end of `ir_module_finalize_merged_calls`
(`self-hosted/compiler/module_frontend.sio`). A per-step probe pinned it to the transition
`after_promote2` (correct) → `after_2pass` (corrupt).

Real root cause: that loop called `ir_module_resolve_one_call_target(&out, ins)` passing the whole
`IrInstr` **by value**. `IrInstr` carries a `Box` (`call_args`); lean_single miscompiles the
by-value large-struct copy and scrambles the *caller's* `ins` local (dst/src1/call_args/fn_id) before
the unconditional write-back. (`ir_module_compact_duplicate_fn_refs` uses the same read/writeback but
passes only `ins.name` — never the whole `IrInstr` — which is why it was safe.)

Fix: resolve from scalar fields only — `ir_module_resolve_call_target_fields(module, old_id, name)` —
and write the slot back only when `fn_id` actually changes. Witnesses (Slurm, actual rc): min → rc=0
`CROSS_SRET_MIN_OK`; cd_mul → rc=0 `CD_MUL_CROSS_SRET_OK`; a8_diag_fwd/sizes/step/ctrl → rc=0;
`fano_basics` FAIL→PASS; zero regressions across a ~20-test base-vs-fix differential. Repro tests
flipped known-failure → run-pass.

Still OPEN (separate, pre-existing — reproduce on base): `cd_exact_generic_i64` SIGSEGVs at
**compile time** inside `lower_program_to_ir_summary_box_with_externs_ref` for its generic dependency
module (`lower_array: dep_begin 1`, between `module_frontend_lower: summary_begin`/`summary_done`);
EISA `test_eisa_isa/evm` fail earlier at "multimodule native thin-link compilation failed".

---

Status: ~~**BLOCKED / diagnosed**~~ **FIXED (see resolution above)** (2026-07-06). This is the last A5 blocker
(`cd_exact_generic_i64.sio`). It is NOT the compile-time dep-lowering segfault the
original A8 prompt described — that hypothesis was falsified. The real defect is a
**runtime** miscompile in the Madaros *imported-lane* (multi-module) IR lowering.

## TL;DR

A dependency-module function that **forwards through an inner same-module
struct-returning call**, i.e. the shape

```sio
fn small_basis_fwd(k: i32, idx: i32) -> Small with Mut, Panic {
    var r = small_zero(k)      // <-- inner struct-returning call bound to a var
    r.c[idx as usize] = 1
    return r
}
```

is lowered to **corrupt IR** by the Madaros imported/merged (summary+bodies) lowering
path. The produced ELF SIGSEGVs (rc=139) at runtime. The **single-module** lowering of
the byte-identical source is correct. The fault is **size-independent** (reproduces with
`Small{c:[i64;4]}`, not just the cd `[f64;2048]`).

## Verified baseline matrix (all ELFs actually run; rc recorded)

| test | shape | modules | BUILD | RUN |
|------|-------|---------|-------|-----|
| a8_diag_single | `var r=big_zero(k);…` `[i64;256]` | 1 | ok | **rc=0 OK** |
| a8_diag_ctrl | imported scalar + `small_basis` (direct literal) | 2 | ok | **rc=0 OK** |
| a8_diag_step | imported `big_zero` (direct) then `big_basis` (forward) | 2 | ok | big_zero OK, **big_basis SIGSEGV** |
| a8_diag_fwd | imported `small_basis_fwd` `[i64;4]` | 2 | ok | **rc=139** |
| a8_diag_sizes | imported `m16_basis_fwd`/`m64_basis_fwd` | 2 | ok | **rc=139** |
| sret_forwarding_cross_module_min | `big_basis`/`big_add` `[i64;256]` | 2 | ok | **rc=139** |
| sret_forwarding_cross_module_cd_mul | `cd_mul` `[f64;2048]` | 2 | ok | **rc=139** |
| cd_exact_generic_i64 | generic `[i64;2048]`, 4 modules | 4 | **rc=139 (compiler segfault, separate)** | n/a |
| sret_8_field_return / generic_struct_return | single-module struct return | 1 | ok | **rc=0** (A4 regression guard, still green) |

Key discriminators:
* single-module identical shape = **OK** → not the struct, not the size.
* imported DIRECT struct return (`big_zero`, `small_basis` literal) = **OK** → not "any cross-module struct return".
* imported FORWARD (`var r = <inner struct call>`) = **SIGSEGV**, at every size → the forwarding lowering is the fault.

## Ground truth — the corrupt merged IR

`small_basis_fwd` (fn 3) in the post-finalize merged module
(`SOUNIO_DUMP_ALL_CALLS=1`, op tags: 7=Call 0=LoadImm 12=FieldGet 15=IndexSet 8=Return 16=Alloc):

```
fn3 small_basis_fwd(k, idx)   ; params k=vreg0, idx=vreg1
  i0 Call     dst=3 arg=[2] fn=5(big_zero)   ; var r = small_zero(k)
  i1 LoadImm  dst=3 imm=1                     ; vreg3 = 1
  i2 LoadImm  dst=4 imm=0
  i4 FieldGet dst=5 s1=2 fi=0                 ; r.c   (base = vreg2)
  i5 IndexSet s1=5 s2=1 imm=3                 ; c[idx] = vreg3
  i6 Return   s1=2                            ; return r (vreg2)
```

Two independent defects are visible:

**Defect 1 (the crasher) — the `var r = <call>` binding is destroyed.**
`r` is referenced as **vreg2** (FieldGet base i4, Return i6), but the call result lands
in **vreg3** and is immediately clobbered by `LoadImm vreg3=1`. The call's argument is
**vreg2** — the very slot used as `r` — and nothing ever defines it. So:
* `k` is never passed (arg = uninitialized vreg2 instead of param vreg0);
* the call result (vreg3) is discarded;
* `r` (vreg2) is never assigned → `r.c[idx]` / `return r` dereference a **garbage handle**.

Confirmed in the disassembly of the produced ELF (`small_basis_fwd` @ 0x4012c4):
```
4012dd  mov -0x18(%rbp),%rdi   ; call arg  = vreg2 (uninitialized; k is at -0x8 = vreg0)
4012e4  call 0x40141b          ; small_zero (resolved to big_zero)
4012e9  mov %rax,-0x20(%rbp)   ; result -> vreg3
4012f7  mov $0x1,-0x20(%rbp)   ; vreg3 clobbered by literal 1
...
40130c  mov -0x18(%rbp),%rax   ; field-write base = vreg2 (garbage handle)
401331  mov 0x0(%rax),%rax     ; deref garbage -> SIGSEGV
```
`ir_slot_offset(v) = -(v+1)*8`, so -0x8=vreg0, -0x10=vreg1, -0x18=vreg2, -0x20=vreg3.

The finalize passes (`ir_module_finalize_merged_calls` → promote/compact/resolve) only
rewrite `fn_id`/`name` on Call/CallSret/LoadFnRef — they never touch vregs — so this
scramble is produced by the **body lowering itself** in the imported path
(`module_frontend_lower_program_items_box_traced_with_externs` →
`lower_program_to_ir_summary_box_with_externs_ref` +
`lower_program_bodies_from_summary_with_epistemic_boxed_ref`, self-hosted/ir/lower.sio).
The single-module path uses a different (single-pass) lowering entry and is correct.

**Defect 2 (independent) — compaction collapses distinct call targets.**
`small_basis_fwd` calls `fn=5` which is **`big_zero`**, not `small_zero`.
`ir_module_compact_duplicate_fn_refs` (self-hosted/compiler/module_frontend.sio) unified
every `*_zero` callee (`small_zero`/`m16_zero`/`m64_zero`/`big_zero`, pre-finalize
fn_ids 5/9/11/13) onto a single `fn=5`. Even with Defect 1 fixed this would call the
wrong constructor. (Not the crash cause, but a real correctness bug — likely intra-module
calls resolving by index/first-body rather than exact symbol during compaction.)

## Where to look next (fix entry points)

1. Diff the IR of the same forwarding function through the **single-pass** lowering
   (`lower_program_to_ir` family) vs the **summary+bodies** path
   (`lower_program_bodies_from_summary_with_epistemic_boxed_ref`). The single-pass yields
   `call dst=2 arg=[0]` (r=vreg2 = result, arg=k=vreg0); the imported path yields
   `call dst=3 arg=[2]`. The desync of param/local vregs from the call arg/dst is the bug.
2. Suspect: how `lower_let_stmt_ref` (lower.sio ~6622) binds `var r = <ExprCall>` when the
   body is lowered from a summary — the returned reg must equal the emitted call `dst`,
   and the argument ident `k` must resolve to its param vreg. In the imported path one of
   these bindings is off, leaving the call arg = the var's own (undefined) slot.
3. Defect 2: `ir_module_compact_duplicate_fn_refs` / `ir_merge_find_function_name_index`
   must resolve intra-module calls by the call's exact `ins.name`, not collapse all
   same-shaped `*_zero` symbols to one index.

## Reusable diagnostic added

`SOUNIO_DUMP_ALL_CALLS=1 madaros build <t>` now dumps every merged function's
instructions (op/dst/src/field/args) at pre- and post-finalize — see
`ir_module_dump_all_calls_diag` in self-hosted/compiler/module_frontend.sio (env-guarded,
inert by default). NB: the self-hosted `print_int` appends a newline, so the dump prints
one field per line (`Dq key=val`).

<!-- docs:meta
topic_id: repo.docs.audit.madaros-seed-begin-dispatch-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-seed-begin-dispatch-2026-07-05
-->

# BLK-MADAROS-SEED-BEGIN — forensic dispatch result (2026-07-05)

Status: **root-caused and worked-around at HEAD; compile lane verified green by
clean rebuild. Two successor defects recorded** (deployed-binary false-green
routing; generated-ELF runtime wrong-code). No source change was required for
the seed_begin crash itself.

Branch: `gpu/epistemic-tensor-core-next` @ `e377bea38`.
Method: no-rebuild forensics on git-historical prebuilts + one clean scratch
rebuild from HEAD in an isolated worktree. Zero edits to the shared checkout.

## 1. Root cause of the seed_begin segfault (historical shape)

Reproduced deterministically on the 2026-06-24 prebuilt
(`git show 81cbecf62:bin/madaros-linux-x86_64`, 95 MB) with
`tests/stdlib/theorem/test_smt_solver_basic.sio`,
`tests/stdlib/math/test_dd64_eft_exact.sio`, and
`tests/multimodule/thin_single_main.sio` — **all three crash at the same PC**
(runtime `0x3f8c76d`, file offset `0x3b8c76d`), instruction `mov (%rax),%rax`,
fault addresses `{0x9, 0x8, 0x2}`.

Frame walk (manual `rbp` chain, verified call-target sanity at every frame):

```
ir_function_has_opcode                     (crash; lower.sio:7159 @ 9b44519a2)
← ir_patch_validated_calls_in_function     (lower.sio:7193)
← ir_patch_validated_calls                 (lower.sio:7251)
← lower_program_bodies_from_summary_with_epistemic_boxed_ref
← module_frontend seed/bodies driver       (prints "lower_array: seed_begin")
← imported_compile driver ← module_native_driver ← main
```

Defect class: **lean_single bootstrap-compiler codegen bug —
`&!`-of-boxed-array-element elides one level of indirection.** For source of
the form `ir_patch_validated_calls_in_function(..., &! (*module).functions[i],
...)` where `functions` is an array of boxed slots, the emitted code passes
`lea (module + i*8)` — the address of the Box **slot** — instead of the boxed
element pointer. The callee then reads `instr_count` at `slot+0x1088`, which
actually lands on `functions[i+529]` (a Box pointer, ≈2.3e13, i.e. an
effectively unbounded loop bound), scans the neighboring Box slots as if they
were instructions and dereferences each; it dies on the first slot whose
content is a raw small integer (needle `0x9` = `IrOpcode::IrCall`
discriminant; crash index 2031). Deterministic by construction.

Same family as the 2026-06-18 handoff fix
(`(*box_ref).field` garbage via `&! Box<IrModule>`,
`artifacts/omega/agent_handoff.log.md`, 16:00 UTC entry) and the 2026-06-24
`SEED_FIX_*` lean_single store fixes. The naming collision noted in the
2026-07-05 recheck stands: lean_single "seed" compiler ≠ `programs[0]` "seed
module"; this bug is in the former, manifesting in the latter's pass.

## 2. Status at HEAD: worked around, verified by clean rebuild

At HEAD the imported/multi-module bodies path skips the patch pass entirely:
`lower_program_items_bodies_from_summary_with_epistemic_boxed_ref`
(`self-hosted/ir/lower.sio:9595-9612`, trace tag
`skip_patch_for_imported_probe`). A clean scratch build from `e377bea38`
(`scripts/ci/build_modular_madaros.sh` in a detached worktree, 103 MB,
9,962 fns) confirms:

| Input | Compile | Route | ELF | Runtime |
|---|---|---|---|---|
| `tests/multimodule/thin_single_main.sio` | exit 0 | full modular IR | 8.2 KB | **exit 7 — correct** |
| `tests/stdlib/theorem/test_smt_solver_basic.sio` | exit 0 | full modular IR | 164 KB | exit 1, no output (**wrong-code**) |
| `tests/stdlib/math/test_dd64_eft_exact.sio` | exit 0 | full modular IR | 33 KB | SIGSEGV (**wrong-code**) |
| `tests/known_failures/smt_imported_runtime_wrongcode_witness.sio` (2-module smt smoke, ex-`test_hello.sio`) | exit 0 | full modular IR | 152 KB | prints through "After smt_add_clause 2", then SIGSEGV |

**The compile-time seed_begin crash no longer exists at HEAD.** The 2026-07-05
recheck's exit-139 matrix was evidence from an older binary; no
currently-buildable or currently-deployed binary reproduces it.

Latent-risk note: `ir_patch_validated_calls` (`lower.sio:9193`) still passes
`&! (*module).functions[i as usize]` at its remaining call sites
(`lower.sio:9237, 9456, 9505, 9565, 9670, 9713` — single-module lanes). Until
the lean_single `&!`-of-boxed-element codegen is fixed at the root, those
lanes carry the same latent class.

## 3. Successor defect A — deployed binaries false-green (packaging regression)

The currently deployed binaries — committed prebuilt
`bin/madaros-linux-x86_64` (from `b57e1b379`, 2026-07-03, git-clean) and the
since-deleted `artifacts/self-hosted/madaros` (built 2026-07-04 17:06) — route
**every multi-module compile to the compact stub backend**
(`module_native_simple_driver.sio`, marker line `"compact modular IR table
path"`), whose only caller in-tree is the lean witness entry
(`self-hosted/compiler/lean.sio:133`). That backend hand-writes a minimal ELF
from a degenerate "simple IR table": the smt test compiles to a **590-byte ELF
that exits 0 printing nothing** (expected: `T1 PASS`…`T5 PASS`).

This is a regression of the explicit 2026-06-16 decision (`9e19da1a9`, "Skip
compact IR table path in native driver (was corrupting parser state)" — the
stub path "produced only 2-3 instruction stubs per function"). Committed
`main.sio` dispatch routes imported sources to the full driver; the shipped
prebuilt does not behave like committed `main.sio`.

Consequences:
- The default lane **false-greens all multi-module programs** (exit 0, empty
  binaries). Exit-code-only gates pass; output-verified gates fail.
- The gate receipt scheme (`_madaros_receipt_ok`, `smt_skip=0`) did not catch
  it.

Required remediation (not performed in this dispatch):
1. Refresh the prebuilt from a `build_modular_madaros.sh` build of committed
   `main.sio` — after successor defect B is addressed, since a refreshed
   prebuilt currently fails smt/dd64 at runtime (loudly, which is still
   strictly better than silently-empty binaries).
2. Strengthen the receipt gate to **output-verified** witnesses (grep expected
   PASS lines), not exit codes, per the recheck doc's own warning.

## 4. Successor defect B — generated-ELF runtime wrong-code (new frontier)

With compile fixed, the default-lane blocker moves to the generated code:
imported programs beyond the thin witness produce wrong runtime behavior
(table in §2). gdb forensics on the generated binaries pin the class exactly:

`hello.elf` faults at `smt_solve` **entry+0x27** (deterministic, SEGV_MAPERR,
si_addr=0):

```asm
4068b2: mov %rdi,-0x8(%rbp)   ; spill incoming ctx* into slot -0x8
4068b9: mov $0x0,%rax
4068c0: mov %rax,-0x8(%rbp)   ; unconditional zero-store into the SAME slot
4068c7: mov -0x8(%rbp),%rax   ; reload -> 0
4068ce: mov (%rax),%rax       ; NULL deref -> SIGSEGV
```

A locally-initialized value's slot is allocated on top of the first
parameter's spill slot — **vreg/slot collision between the incoming first
parameter and the struct-return destination initialization**. The sibling
`smt_add_clause` (called twice, successfully) uses the identical
deref idiom without the zero-clobber, so the defect is specific to
struct-returning entries. `dd64.elf` shows the same class (param slot `-0x8`
zeroed twice at entry, crash at first callee entry+0x90, si_addr=0x20 via a
null struct-table base). `smt.elf` exits 1 with **zero write() syscalls**
while all six expected `PASS` strings are present in rodata — early silent
wrong-path, same family.

**RESOLVED for the smt lane (2026-07-05, commit `0ba18481a`).** Three-stage IR
dumps (post-dep-lower / post-merge / at-codegen) proved the frontend clean and
localized the corruption to the post-merge finalize passes: the two-level
read-modify-write shape `var ins = out.functions[fi].instrs[ii]; …;
out.functions[fi].instrs[ii] = ins` in `ir_module_finalize_merged_calls` and
`ir_module_compact_duplicate_fn_refs` is miscompiled by lean_single — every
IrCall those passes touched in dep-module functions became an all-zero IrInstr
(decoded at runtime as `LoadImm r0, 0`, clearing the first parameter register:
the exact zero-clobber in the disassembly above). The
`restore_user_main_calls` band-aid only shielded fn=0, which is why seed mains
printed while stdlib callees died. The "sret destination collision" reading of
the asm was a red herring for this layer; `smt_solve` returns `i32`.

Fix: restructure both passes to the copy-function-out / one-level indexed
field store / copy-function-back shape already proven safe in
`ir_merge_prepare_function_with_remap_from_func`. Verified on a clean scratch
build, default engine, output-verified: witness end-to-end exit 0; **6/6
`test_smt_*` ALL PASS; thin exit 7; zero all-zero records across all 78
dumped functions.**

Still open in this family: 4/4 `tests/stdlib/math/test_dd64_*.sio` SIGSEGV
with no output — a distinct defect (first callee of `main`, struct-returning
`DD64` functions; the uncommitted `is_sret`/`sret_dest_reg`/`emit_call_helper`
WIP in `self-hosted/ir/lower.sio` targets exactly this). Dispatch as
`BLK-MADAROS-DD64-STRUCT-RETURN` with the dd64 witnesses as the matrix and
the gdb evidence preserved in the wt-head-build scratch dir
(`disasm_dd64_window.txt`). The durable root fix for the whole clobber family
remains the lean_single two-level indexed load/store codegen repair.

## 5. Acceptance state vs the original gate

The recheck doc's acceptance gate — 6/6 `test_smt_*` ALL PASS on the default
engine — is **MET** on a clean build carrying commit `0ba18481a` (§4).
`BLK-MADAROS-SEED-BEGIN` (compile-time segfault) is closed as root-caused +
worked-around; the runtime wrong-code successor is fixed for the smt lane.
Remaining before the default lane is fully green: the dd64 struct-return
defect (§4), the prebuilt refresh + output-verified receipt (§3), and the
durable lean_single codegen repair for the two-level indexed RMW class.

## 6. Evidence artifacts

Scratch (session-local, not committed):
`/workspace/.tmp/claude-1000/-workspace-sounio/…/scratchpad/seedmatrix/`
(gdb transcripts `gdb_shape*.txt`, `gdb_framewalk.txt`, `callers_disasm.txt`,
06-24 prebuilt extraction, crash matrix logs) and `…/scratchpad/wt-head-build/`
(clean HEAD build + verified matrix logs). Key negative results preserved:
ulimit/H0 memory-cap hypothesis refuted (crash identical with and without the
wrapper's 16 GiB `-v` cap; fault addresses are near-null, not allocation
failures); 2026-07-04 static-disassembly scratch was a self-inconsistent dead
end (searched linked vaddrs against raw-offset disassembly).

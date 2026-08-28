<!-- docs:meta
topic_id: repo.docs.audit.frame-fix-full-selfhosted-2026-06-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.frame-fix-full-selfhosted-2026-06-16
-->

# Frame fix — FULL SELF-HOSTED validation (2026-06-16)

## Result: VALIDATED end-to-end on a Madaros rebuilt from source

Two Madaros were built from `self-hosted/compiler/main.sio` by the faithful Rust JIT
(`bin/souc-linux-x86_64`), clean build (`hard_errors=0`, after the cmp/`&programs`/nested-store
fixes on branch `fix/cmp-operands-bool`). They differ ONLY in the live-site frame logic:
- **M_base**: live emitters `compile_ir_function_v2_core_ir_into:6190` and
  `native_v2_core_begin_fn_spill_into:7333` reverted to `sub rsp, 512`.
- **M_fix**: as committed — `align16((*func).reg_count*8)`.

Reproducer A/B (both run with `ulimit -s unlimited`; see §2):

| N | M_base (512) | M_fix (dynamic) |
|---|---|---|
| 1 | trail=5 PASS | trail=5 PASS |
| 2 | trail=1 (WRONG) | trail=5 **PASS** |
| 4 | trail=15 (WRONG) | trail=5 **PASS** |
| 5 | **SIGSEGV** | trail=5 **PASS** |

Single variable (the frame fix), both compiled from source by the same faithful compiler:
M_base reproduces the bug exactly; M_fix passes all N. This is the definitive self-hosted
A/B that the lean_single seed could never provide (it miscompiles Madaros).

## 2. The "array/loop silent miscompile" was a STACK OVERFLOW, not a miscompile

Every prior on-worker run of a JIT/seed-built Madaros crashed (rc=139) compiling the
reproducer, which looked like a codegen miscompile. Localization (feature ladder + gdb +
disasm) proved otherwise:

- Trigger: short-circuit `&&`/`||` with **comparison operands** (`(i<3) && (j<5)`).
  `a && b` (bool vars) is one recursion level shallower and survives.
- gdb at crash: `rbp-rsp ≈ 4.85 MB`, rsp in unmapped memory (stack overflow).
- Disasm of the crashing function: prologue `sub $0x4a0150,%rsp` (4.85 MB frame) + a
  page-probe loop (`orb $0x0,(%rax); sub $0x1000,%rax; jmp`) that walks off the stack.
- Madaros's recursive expression-compiler reserves multi-MB frames (the build log warns
  `stack frame too large (… bytes) … consider using global arrays`). Nested expressions
  recurse through several such frames and exhaust the default 12.5 MB thread stack.
- **`ulimit -s unlimited` → everything compiles and runs** (proof: the A/B above).

This is independent of the frame fix and of my build-error fixes. The proper remedy is to
move the large codegen scratch locals to globals (per the compiler's own warning); the
operational workaround is a larger stack (`ulimit -s unlimited`).

## 3. Reproduction
- Build: `JIT=bin/souc-linux-x86_64` builds `main.sio` (branch `fix/cmp-operands-bool`),
  positional CLI `<jit> <src> <out>`; revert 6190/7333 to `sub rsp,512` for M_base.
- A/B: `ulimit -s unlimited; M build reproducer_N.sio -o out.elf; ./out.elf`.
- Binaries (this session): artifacts/Mfix-binop.elf, artifacts/Mbase-binop.elf.

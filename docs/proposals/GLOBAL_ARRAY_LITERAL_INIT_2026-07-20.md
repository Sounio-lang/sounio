<!-- docs:meta
topic_id: repo.docs.proposals.global-array-literal-init-2026-07-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.proposals.global-array-literal-init-2026-07-20
-->

# Proposal: Global Array Literal Initialization

> **Status:** design proposal; no implementation yet.
> **Blocker-ID:** `BLK-20260720-global-array-literal-init` (proposed, B1, compiler-semantics, E2).
> **Date:** 2026-07-20.
> **Author:** opencode session (Sounio remote-first workspace).
> **Base:** `origin/main@10e633f5a` (post-PR #1236 + #1239).

## Problem

A Sounio global declared with an array literal initializer is silently
zero-initialised at runtime. The literal values are parsed, type-checked,
and accepted without error, but they never reach the generated ELF's BSS
section.

```sounio
let BUF: [i8; 4] = [72, 105, 10, 0]    // compiles; BUF[0] reads as 0 at runtime

fn main() -> i64 with IO {
    print_int(BUF[0] as i64)            // prints 0, expected 72
    0
}
```

### Reproduction (verified on `origin/main@10e633f5a`, Madaros source-fresh SHA `cd6bb46a…`)

```bash
# i8 global array literal — BROKEN
cat > /tmp/t2.sio <<'EOF'
let BUF: [i8; 4] = [72, 105, 10, 0]
fn main() -> i64 with IO {
    print_int(BUF[0] as i64)
    print_char(10)
    print_int(BUF[1] as i64)
    print_char(10)
    0
}
EOF
souc compile /tmp/t2.sio -o /tmp/t2.elf && /tmp/t2.elf
# actual:   0\n0
# expected: 72\n105

# f64 global array literal — BROKEN
cat > /tmp/t3.sio <<'EOF'
let DATA: [f64; 3] = [1.5, 2.5, 3.5]
fn main() -> i64 with IO {
    print_f64(DATA[0])
    print_char(10)
    0
}
EOF
souc compile /tmp/t3.sio -o /tmp/t3.elf && /tmp/t3.elf
# actual:   0.000000
# expected: 1.500000

# LOCAL array with literal — WORKS
cat > /tmp/t1.sio <<'EOF'
fn main() -> i64 with IO {
    let buf: [i8; 4] = [72, 105, 10, 0]
    print_int(buf[0] as i64)
    print_char(10)
    0
}
EOF
souc compile /tmp/t1.sio -o /tmp/t1.elf && /tmp/t1.elf
# actual:   72   ✓

# LOCAL array with element-wise assignment — WORKS
cat > /tmp/t6.sio <<'EOF'
fn main() -> i64 with IO {
    var buf: [i8; 16] = [0; 16]
    buf[0] = 72
    print_int(buf[0] as i64)
    print_char(10)
    0
}
EOF
souc compile /tmp/t6.sio -o /tmp/t6.elf && /tmp/t6.elf
# actual:   72   ✓
```

### Impact

This is the same silent-corruption class as the compact-IR-path defect
fixed in PR #1236: compilation and execution complete with `rc=0`, but
the runtime observes wrong data. Any Sounio program that declares a
global with an array literal — a common pattern for lookup tables,
byte buffers, f64 constants, and configuration data — is affected.

The `stdlib/image/pure/png.sio` encoder works around this by using
runtime-mutable `var` buffers filled element-by-element rather than
literal initialisers. The `examples/lean_mini_compiler.sio` uses a
`let ELF_BUF` global that happens to be zero-initialised by design.
New users writing `let TABLE = [1, 2, 3, 4]` at module scope hit this
silently.

## Root Cause (three layers)

### Layer 1 — Parser: array literals not recorded

`self-hosted/parser/items.sio`, function `parse_global_let_item`
(line ~328) records scalar integer initialisers via
`ast_record_global_var_init(name, value)`. The guard checks for
`ExprKind::ExprIntLit` (and `ExprUnary(OpNeg, ExprIntLit)` for
negatives). `ExprKind::ExprArrayLit` falls through silently — no
init data is recorded.

### Layer 2 — IR Lowering: BSS slot has no body

`self-hosted/ir/lower.sio`, function
`lowerer_preseed_program_items_mut` (line ~1803) creates a BSS slot
(`IR_STRATEGY_BSS_GLOBAL`) for each global. The slot is
`ir_empty_function()` with `param_count = bss_offset`,
`bss_size`, and optionally `prof_counter_id = scalar_init_val`.
The slot's `instrs` array is empty — the array literal is never lowered
into IR instructions inside the slot.

### Layer 3 — Codegen: only scalar f64 init emitted

`self-hosted/native/codegen_x86_linux.sio`, function
`emit_global_var_inits_into` (line ~8923) walks all BSS slots. For each
slot with `returns_float == 1` (set when `ast_lookup_global_var_init`
returned non-zero), it emits one `mov qword [bss+offset], imm64`.
Array globals have `returns_float == 0` and `prof_counter_id == 0`, so
they are skipped entirely. The BSS section remains zero-filled by the
OS loader.

## Failed Approach (recorded for future agents)

An attempt was made in worktree
`/tmp/sounio-f64cast-20260719` to fix all three layers in one commit:

1. Added `GLOBAL_ARRAY_INIT_QWORDS: [i64; 1024]` and supporting tables
   to `parser/ast.sio` (~90 lines).
2. Added `parser_record_global_array_literal_init` to
   `parser/items.sio` to pack array elements into qwords (~130 lines).
3. Extended `emit_global_var_inits_into` in
   `native/codegen_x86_linux.sio` to walk the table and emit per-qword
   stores (~20 lines).

**Result:** the Madaros binary built successfully but SIGSEGV'd (rc=139)
when compiling ANY program containing a global array literal — including
trivial 2-line fixtures. The crash persisted even when the parser hook
was disabled (immediate `return`). The mere EXISTENCE of the new BSS
globals (`[i64; 1024]` = 8 KB) in the compiler binary caused the crash.
Root cause of the bootstrap crash was not isolated; likely related to
the entry trampoline's init loop iterating over additional BSS slots,
or BSS layout computation in the lean_single seed compiler.

**Lesson:** adding large BSS globals to `parser/ast.sio` is NOT safe
without first verifying the bootstrap compiler can handle the increased
BSS function count. Future approaches should avoid adding new BSS
globals to the compiler source, or should verify the bootstrap path
with a lean_single-fixed-point check before committing.

## Proposed Approaches

### Approach A — Populate BSS slot's IrFunction body (recommended)

**Idea:** during lowering, when a global has an array literal
initializer, lower the literal into `IrIndexSet` instructions INSIDE the
BSS slot's `IrFunction.instrs` (which is currently empty). The codegen
then walks those instructions in `emit_global_var_inits_into` and emits
BSS stores.

**Changes:**
- `ir/lower.sio`: in `lowerer_preseed_program_items_mut`, after creating
  the BSS slot, if the global's initializer is `ExprArrayLit`, call
  `lower_array_elems_ref` against the BSS slot's base address. This
  populates `slot.instrs[]` with `IrIndexSet(base=bss_offset_reg,
  idx=i, val=literal_i)` instructions.
- `native/codegen_x86_linux.sio`: in `emit_global_var_inits_into`, for
  each BSS slot with `instr_count > 0`, walk the instructions and emit
  BSS stores.
- No changes to `parser/ast.sio`. No new BSS globals. No parser hook.

**Pros:**
- No new compiler-source BSS globals (avoids the bootstrap crash).
- Reuses existing `lower_array_elems_ref` machinery.
- Minimal codegen change (walk instructions + emit stores).
- Init data lives in the IR function body, not in global tables.

**Cons:**
- The BSS slot's IrFunction is currently treated as a data slot, not a
  code function. Using `instrs[]` for init data is a semantic overload.
  Needs a comment or a new `compile_strategy` constant (e.g.
  `IR_STRATEGY_BSS_GLOBAL_ARRAY_INIT = 100`) to distinguish "this slot
  has init instructions" from "this slot is a plain scalar".

**Estimated patch size:** ~60 lines across `ir/lower.sio` and
`native/codegen_x86_linux.sio`.

### Approach B — Synthetic `__global_init_*` functions

**Idea:** during lowering, create a synthetic function
`__global_init_<name>` for each global array literal. The function body
contains the array literal's `IrIndexSet` instructions targeting a
hardcoded BSS base address. The codegen calls all `__global_init_*`
functions before `main`.

**Pros:**
- Clean separation (init code lives in real functions, not BSS slots).
- No semantic overload of `IrFunction`.

**Cons:**
- Increases function count (one per global array).
- Needs a naming convention and discovery mechanism (how does codegen
  know which functions to call?).
- Needs a new compile strategy or naming convention for the dispatcher.

**Estimated patch size:** ~100 lines.

### Approach C — .rodata section with memcpy

**Idea:** add a `.rodata` section to the native ELF. Encode the literal
bytes there. The entry trampoline `memcpy`s from `.rodata` to `.bss`
before calling `main`.

**Pros:**
- Most efficient at runtime (bulk copy vs per-element stores).
- Clean separation of init data from code.

**Cons:**
- The native ELF emitter (`native/elf.sio` + `codegen_x86_linux.sio`)
  currently supports only `.text` and `.bss`. Adding `.rodata` requires
  changes to section header emission, program header computation, and
  relocation handling.
- Larger architectural change; higher regression risk.

**Estimated patch size:** ~200 lines.

### Approach D — Heap-backed init table

**Idea:** allocate the init table on the heap at startup (via
`heap_alloc`). Encode init data as immediate values in the entry
trampoline. Copy from immediates to BSS before `main`.

**Cons:**
- Blocked by `BLK-20260712-image-heap` (native ELF not linked against
  libc; `heap_alloc` SIGSEGVs before `main`).

**Status:** not viable until heap allocation is fixed.

## Recommendation

**Approach A** (populate BSS slot's IrFunction body) is the least
invasive path that avoids the bootstrap crash encountered in the failed
attempt. It requires no new BSS globals, reuses existing lowering
machinery, and the codegen change is a natural extension of the existing
`emit_global_var_inits_into` function.

## Acceptance Criteria

A fix is accepted when ALL of the following hold:

1. `let A: [i64; 2] = [10, 20]` compiles and `A[0]` reads as `10` at
   runtime.
2. `let B: [i8; 4] = [72, 105, 10, 0]` compiles and `B[0]` reads as
   `72`, `B[1]` as `105`.
3. `let C: [f64; 3] = [1.5, 2.5, 3.5]` compiles and `C[0]` reads as
   `1.5`.
4. Scalar globals (`let X: i64 = 42`) continue to work (no regression).
5. `scripts/dev/default_path_fidelity_gate.sh` passes 13/13.
6. `scripts/ci/madaros_full_gate.sh` passes 10/10.
7. The fix does NOT add any new `var` BSS globals to `parser/ast.sio`
   (to avoid the bootstrap crash documented above).
8. The lean_single fixed-point gate (`scripts/ci/lean_single_fixed_point_gate.sh`)
   passes (the compiler can still self-compile after the change).

## Regression Gate

A focused regression gate should be added at
`scripts/dev/global_array_literal_gate.sh` with fixtures under
`tests/compiler/global_array_literal_gate/`:

```
main_i64.sio    — let A: [i64; 2] = [10, 20]; print A[0], A[1]
main_i8.sio     — let B: [i8; 4] = [72, 105, 10, 0]; print B[0], B[1]
main_f64.sio    — let C: [f64; 3] = [1.5, 2.5, 3.5]; print C[0]
main_scalar.sio — let X: i64 = 42; print X  (regression: scalar still works)
```

The gate compiles each via the default route, runs the ELF, and checks
exact stdout bytes.

## Blocker Record

```text
Blocker-ID: BLK-20260720-global-array-literal-init
Status: proposed
Severity: B1
Class: compiler-semantics
Owner: unassigned (design proposal stage)
Lane: global array literal initialization
Worktree: n/a (proposal only)
Branch: n/a
Files-Owned: n/a
Files-Read-Only: self-hosted/parser/items.sio; self-hosted/ir/lower.sio; self-hosted/native/codegen_x86_linux.sio; self-hosted/parser/ast.sio
Do-Not-Touch: bootstrap concatenated sources; lean_single fixed-point chain
Repro: see "Reproduction" section above
Observed: let BUF: [i8; 4] = [72, 105, 10, 0] compiles; BUF[0] reads as 0 at runtime (expected 72)
Expected: BUF[0] reads as 72; literal initialiser values reach the BSS section
Acceptance-Gate: scripts/dev/global_array_literal_gate.sh (to be created)
Evidence-Level: E2
Evidence: this proposal document + reproduction commands verified on origin/main@10e633f5a
Fallback-Path: none
Legacy-Kept: yes (scalar global init via ast_record_global_var_init is unchanged)
LLM-Offload: not-required
Next-Action: implement Approach A (populate BSS slot's IrFunction body) in an isolated worktree; verify lean_single fixed-point before committing
```

## References

- PR #1236 — compact IR silent corruption fix (same defect class: compile success + wrong runtime)
- PR #1239 — visibility preflight missing builtins (adjacent area)
- `docs/proposals/NATIVE_HEAP_ALLOCATION_2026-07-12.md` — similar proposal format for the heap blocker
- `docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md` — adjacent multi-module defect audit
- `self-hosted/ir/lower.sio:lowerer_preseed_program_items_mut` — BSS slot creation site
- `self-hosted/ir/lower.sio:lower_array_elems_ref` — element lowering (reusable for Approach A)
- `self-hosted/native/codegen_x86_linux.sio:emit_global_var_inits_into` — init emission site

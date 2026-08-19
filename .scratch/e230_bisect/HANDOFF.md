# E230-v3 bisection handoff — minimax-cli1 → build lane

**Date:** 2026-08-18
**Trigger:** grok-cli5 dispatch
[`docs/audit/MADAROS_TRIVIAL_3I64_STRUCT_ALLOC_139_DISPATCH_2026-08-18.md`](https://github.com/.../MADAROS_TRIVIAL_3I64_STRUCT_ALLOC_139_DISPATCH_2026-08-18.md)
(worktree `/workspace/.wt/grok-cli5`, branch
`lane/grok-cli5/trivial-139-dispatch-20260818` @ `f28da3580d`, pushed, no PR).
**Owner of the corrector:** the v3 patch (`.scratch/e230_diagnostic.patch`,
commits `97d46478da` patch + `aa0041a26e` gate, both pushed to
`lane/minimax-cli1/20260815-clean`).
**Repro:**
[`docs/audit/repro/trivial_3i64_alloc_139.sio`](https://github.com/.../trivial_3i64_alloc_139.sio)
(N=1, no helper, no print, no `Alloc` effect — minimal witness).

## 0. What was done on this pod

Per FLEET_CONSTRAINTS, no full self-compile / source tree rebuild is allowed
on the source pod. The bisection patch halves were generated and
**dry-applied** against the saved `.orig` snapshots at `/tmp/e230_v3/`,
which were captured from `origin/main` HEAD before this lane ever touched
the source tree.

```
.scratch/e230_bisect/
├── patch_runtime_context_only.patch    6,686 bytes   hunk 1 ONLY
└── patch_codegen_only.patch           15,850 bytes   hunks 2+3 ONLY
```

Both halves apply cleanly with `git apply` against the saved `.orig`
files. Combined apply produces a result `diff -q` identical to the
existing `/tmp/e230_v3/runtime_context.sio.new` and
`/tmp/e230_v3/codegen_x86_linux.sio.new` (which are the result of the
full v3 patch — see `e230_diagnostic.patch`).

The halves are **not** applied to the working source tree on this pod.
The build lane takes them from here.

## 1. The bisection question

grok-cli5 asked: *"aplica so o codegen, mede; aplica so o bump, mede.
Uma das duas celulas responde."*

The v3 patch does two things:

1. **`runtime_context.sio`** — adds field at offset 248
   (`e230_90_warning_fired`), bumps `runtime_context_size()` from 248 to
   256.
2. **`codegen_x86_linux.sio`** — 1-for-1 replacement of
   `nc_core_emit_alloc_failure_diagnostic_into` plus 2 new
   helpers (`nc_write_rodata_to_stderr_into`,
   `nc_append_rodata_bytes`), and a 90% drift warning block inserted at
   the handle_count load site inside `nc_core_emit_alloc_into`.

The 3-slot SIGSEGV is independent of the patch's **runtime** goals (the
test passes with default Madaros and lean_single at the same source).
So the bug is in **codegen emission**, not the runtime contract. The
runtime_context_size bump is the cheaper-test hypothesis; the codegen
hunk is the structural suspect.

## 2. Build lane workflow

### Cell A — runtime_context_size bump only

```sh
# Staging tree with origin/main HEAD in /tmp/staging
cd /tmp/staging                                # git working copy at origin/main

# Snapshot the pre-patch state
git -c user.email=test@local -c user.name=test commit --allow-empty -q \
  -m "staging-pre-bisect"

# Apply ONLY the runtime_context half
git apply /path/to/.scratch/e230_bisect/patch_runtime_context_only.patch

# Build the patched binary
scripts/ci/build_modular_madaros.sh /tmp/madaros_ctx_bump.elf

# Run the 14-cell matrix against the patched binary
SOUNIO_STDLIB_PATH=$(pwd)/stdlib \
  bash docs/audit/repro/run_matrix.sh /tmp/madaros_ctx_bump.elf
```

Expected matrix outcome (baseline; should NOT regress):

| Case | Programme | rc |
|---|---|---|
| A | `print("done\n")` only | 0 |
| J | `while i < 3` no aggregate | 0 |
| K | helper returns `i64`, N=3 | 0 |
| P | 3-field struct declared, never constructed | 0 |
| G | 1-field `i64` struct, N=3 | 0 |
| F | 2-field `i64` struct, N=3 | 0 |
| U | 2-field `f64` struct, N=1 | 0 |
| arr2 | `[i64; 2]` literal | 0 |
| **B** | **3-field `i64`, N=1** | expected **0** (if 139, the runtime_context bump is the trigger) |
| **C** | **W4 exact** | expected **0** |
| **D** | **tiny W2** | expected **0** |
| H | 4-field `i64`, N=1 | expected **0** |
| R | 3-field `i32` (12 B) | expected **0** |
| Q | `[i64; 3]` literal | expected **0** |
| I′ | 3-field `f64` struct, N=1 | expected **0** |

If ANY of these come back as 139 or 132, **the runtime_context_size bump
is the trigger** and the codegen hunks are innocent. (This is a
surprise — the bump is purely a layout change in the runtime_context
struct, but the `nc_emit_byte` / `nc_emit_u32_le` calls in the patch
do reference `runtime_context_field_e230_90_warning_fired()` and
offsets 248+ in rodata. A bug in that path could in principle corrupt
context reads for any aggregate that goes through the codegen at runtime.)

### Cell B — codegen hunks only (with stub)

Hunk 3 references `runtime_context_field_e230_90_warning_fired()`, which
is defined by hunk 1. To bisect cleanly, **stub the accessor first**:

```sh
cd /tmp/staging
# Stub the accessor so the codegen half compiles without hunk 1
# (offset 9999 is past the 248-byte context, so the warning will never
# fire and the handle_count loop will reach the existing 100% refusal
# correctly).
sed -i 's/^pub fn runtime_context_size() -> i64 { 248 }$/pub fn runtime_context_field_e230_90_warning_fired() -> i64 { 9999 }\npub fn runtime_context_size() -> i64 { 248 }/' \
  self-hosted/native/runtime_context.sio

# Apply ONLY the codegen half
git apply /path/to/.scratch/e230_bisect/patch_codegen_only.patch
# (Or extract the body via `awk 'NR>=81' ...patch_codegen_only.patch > /tmp/cg.body.patch`
#  and `git apply /tmp/cg.body.patch` to skip the comment preamble.)

# Build
scripts/ci/build_modular_madaros.sh /tmp/madaros_codegen_only.elf

# Run the matrix
SOUNIO_STDLIB_PATH=$(pwd)/stdlib \
  bash docs/audit/repro/run_matrix.sh /tmp/madaros_codegen_only.elf
```

Expected outcome if the codegen half is the trigger:

| Case | Programme | rc |
|---|---|---|
| A | `print("done\n")` only | 0 |
| J | `while i < 3` no aggregate | 0 |
| K | helper returns `i64`, N=3 | 0 |
| P | 3-field struct declared, never constructed | 0 |
| G | 1-field `i64` struct, N=3 | 0 |
| F | 2-field `i64` struct, N=3 | 0 |
| U | 2-field `f64` struct, N=1 | 0 |
| arr2 | `[i64; 2]` literal | 0 |
| **B** | **3-field `i64`, N=1** | reproduces **139** |
| **C** | **W4 exact** | reproduces **139** |
| **D** | **tiny W2** | reproduces **139** |
| H | 4-field `i64`, N=1 | reproduces **139** |
| R | 3-field `i32` (12 B) | reproduces **139** |
| Q | `[i64; 3]` literal | reproduces **139** |
| I′ | 3-field `f64` struct, N=1 | reproduces **132** |

If ALL of the >=3-slot cases fail the same way as the full v3 patch,
**the codegen half is the trigger**. (This is the expected outcome.)

## 3. Decision tree

```
                    ┌──────────────────────────────┐
                    │ Build lane runs the matrix   │
                    └──────────────┬───────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
        ┌─────────▼────────┐              ┌─────────▼────────┐
        │ rc_bump: all 0   │              │ rc_bump: any 139 │
        │ (cheapest test)  │              │ (surprise)       │
        └─────────┬────────┘              └─────────┬────────┘
                  │                                 │
       ctx_bump is innocent               ctx_bump is the trigger
       by elimination, codegen            (offset 248 reads may
       half is the trigger.               corrupt context for
                                          any aggregate.)
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
        ┌─────────▼────────┐              ┌─────────▼────────┐
        │ rc_codegen:      │              │ rc_codegen: all 0│
        │ 139/132 on 3-slot│              │ (BIG surprise    │
        │                  │              │  — contradicts   │
        │                  │              │  the structural  │
        │                  │              │  suspicion)      │
        └─────────┬────────┘              └─────────┬────────┘
                  │                                 │
       codegen half is the trigger.        both halves innocent;
       Likely sites:                        revisit the model — the
       (a) `nc_emit_mov_reg_reg(nc, 3, 0)`  139 is in something
           at the END of the 90% block,     the v3 patch does NOT
           restoring rbx = handle_count.    do. (Possible: the
       (b) the inline itoa calls leaving     test pod's prebuilt
           rcx/rbx in a state that the       Madaros is FINE, but
           subsequent code mishandles in     the v3 patch fails
           the SRET path for 3-slot          to reproduce on the
           structs.                          build lane — that's
       (c) double-append in the rodata       the cleanest outcome
           segment at offset 248+.           for the patch author.
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │                                │
                  │  PATCH AS A WHOLE CANNOT LAND  │
                  │                                │
                  │  v3 commits 97d46478da +        │
                  │  aa0041a26e must be reverted.  │
                  │                                │
                  │  Path forward:                 │
                  │   1. revert both commits in    │
                  │      one atomic revert on      │
                  │      lane/minimax-cli1/        │
                  │      20260815-clean            │
                  │   2. redispatch the E230       │
                  │      diagnostic from a fresh   │
                  │      hunk 1 + invariant hunk 3 │
                  │      shapes (call-only E230    │
                  │      function, not inline      │
                  │      block at the alloc hot    │
                  │      path)                     │
                  │   3. validate the gate w2/w3/  │
                  │      w4 against the new        │
                  │      compiler BEFORE re-       │
                  │      landing                   │
                  │                                │
                  │  The gate witnesses W2/W3/W4  │
                  │  use 3-field structs (defect 2 │
                  │  in the v3 patch review was    │
                  │  correct: tag=24 > 16 unbox).  │
                  │  But the W2/W3/W4 results      │
                  │  measured on the v3-patched   │
                  │  compiler are SUSPECT — that   │
                  │  compiler could not build any  │
                  │  3-slot aggregate. The gate    │
                  │  re-ran on a broken compiler.  │
                  │  A separate gate validation    │
                  │  pass is required after the    │
                  │  patch is fixed or replaced.   │
                  └────────────────────────────────┘
```

## 4. Constraints preserved

- **No source-tree changes on this pod** — the patch halves are
  generated, not applied. The build lane takes them.
- **Atomic commits, one logical change each** — the bisection plan
  lives in `.scratch/`, not in the repo. The reverted commits stay
  atomic (one for the patch, one for the gate).
- **No `Co-Authored-By` trailer** — patch halves, handoff doc, and
  the revert commit must respect this.
- **EN-UK orthography** — applies to the handoff doc.
- **No tooling** — `git apply --check` against the saved `.orig`
  files is the only validation done on this pod.

## 5. Files touched on this pod

- `/workspace/.wt/minimax-cli1/.scratch/e230_bisect/patch_runtime_context_only.patch` (new)
- `/workspace/.wt/minimax-cli1/.scratch/e230_bisect/patch_codegen_only.patch` (new)
- `/workspace/.wt/minimax-cli1/.scratch/e230_bisect/HANDOFF.md` (this file)

No source files modified. No commits made.

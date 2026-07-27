<!-- docs:meta
topic_id: repo.docs.audit.module-frontend-lower-array-seed-crash-dispatch-2026-07-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.module-frontend-lower-array-seed-crash-dispatch-2026-07-27
-->

# Dispatch — `module_frontend.sio` `lower_array: seed_begin` segfault (1028 corpus regressions)

**Filed:** 2026-07-27 · **Status:** OPEN (dispatch, not yet implemented) · **Protocol:** CLAUDE.md §8 (self-hosted/ changes require a forensic dispatch before code).

## Summary

`scripts/ci/madaros_corpus_regression_gate.sh`, run for real against a
from-source Madaros for the first time on 2026-07-27
(`bash scripts/ci/build_modular_madaros.sh`, never a prebuilt binary, per
CLAUDE.md operating principle 15), reports **1028 corpus programs newly
failing** beyond the 314-entry `tests/madaros_corpus_baseline.txt`
(`1263 / 1688` total failures — reproduced exactly in this dispatch).
`.github/workflows/ci.yml:298-309` characterises this as "all traced to one
segfault." That is imprecise in one respect (the crash reproduces in at
least two structurally-identical-but-distinct functions, in two different
files — see "Root cause" below) but the practical conclusion turns out to be
even simpler than "one segfault": **re-running the identical, unmodified
gate script with only the compiler invocation wrapped in `ulimit -s
unlimited` makes all 1028 "new" failures disappear.**

```
# As currently written (SOUNIO_MADAROS_CORPUS_BIN = raw ELF, no ulimit guard):
[madaros-corpus] failures observed: 1263 / 1688
[madaros-corpus] 1028 NEW failure(s) under Madaros: ...
[madaros-corpus] FAIL: regression under the modular Madaros compiler

# Identical script, unmodified, SOUNIO_MADAROS_CORPUS_BIN pointed at a thin
# wrapper that does nothing but `ulimit -s unlimited; exec <same raw ELF> "$@"`:
[madaros-corpus] failures observed: 305 / 1688
[madaros-corpus] PASS: no new failures under Madaros (305 known, 0 newly fixed)
```

Both runs are against the same from-source Madaros ELF
(`/tmp/madaros-build`, built via `scripts/ci/build_modular_madaros.sh` in
this session), the same commit, the same machine, back to back. The only
variable is whether the compiler process's stack `ulimit` is raised before
it runs — exactly the guard that `bin/madaros:63-67` already applies for
every *documented* user-facing invocation (`bin/souc compile|run`,
`bin/madaros build|compile|run`), and that at least 20 other
Madaros-invoking gate scripts in this repo already carry
(`grep -rl 'ulimit -s' scripts/`). `scripts/ci/madaros_corpus_regression_gate.sh`
is the outlier that does not, and this dispatch's own initial repro commands
(`madaros -o out.elf self-hosted/ir/egraph.sio`, invoking the raw ELF
directly) inherited the same gap, which is why they were the first hard
evidence of it.

**Net assessment:** the underlying compiler defect (below) is real,
reproducible, and still open — but the "1028" figure is not a faithful count
of things broken for a Madaros user. It is a measurement of one gate
script's missing `ulimit -s` guard, on top of a real but already-mitigated
compiler defect.

## Minimal repro (single file, 9 lines, no imports)

```sounio
// min_repro.sio
fn add_one(x: f64) -> f64 {
    x + 1.0
}

pub fn main() -> i32 with IO {
    println(add_one(4.0))
    0
}
```

```
$ ulimit -s 8192   # Linux default in this environment
$ madaros min_repro.sio -o out.elf
...
imported_compile: lower_begin
lower_array: seed_begin
Segmentation fault (core dumped)

$ ulimit -s unlimited
$ madaros min_repro.sio -o out.elf
...
lower_array: seed_begin
lower_array: seed_done
...
Compilation successful!
```

Crashes identically at `ulimit -s` 8192, 16384, and 32768 KB. At 65536 KB
this specific site clears, but the same trivial program then segfaults
*again*, later, past `Merged IR: 4 functions` — a second, distinct
oversized-frame site further down the pipeline (see "Systemic pattern"
below). Only `unlimited` clears the whole pipeline for this file in this
session's testing. `self-hosted/ir/egraph.sio` (343 items, the file that
first surfaced this) reproduces the identical `lower_array: seed_begin` →
`Segmentation fault` signature at the default 8192 KB stack and is cured by
`ulimit -s unlimited` alone (full clean compile, `Merged IR: 332 functions`,
exit 0, no second crash — the second site is apparently only reached by
some programs, not this one).

This repro is written fresh, not extracted from `egraph.sio`, and — this
matters for cross-referencing below — it is materially the same 2-file
repro already on record a month earlier in
[`MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`](MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md).
(See "Cross-check against prior tracking.")

## Evidence

### Reproduction, this session

- Fresh build: `bash scripts/ci/build_modular_madaros.sh /tmp/madaros-build`
  (`Madaros v0.80.0`, `fns=10608`, `elf: 101368628 bytes`). Never a prebuilt
  `bin/souc`/`bin/madaros`.
- `SOUNIO_STDLIB_PATH=<repo>/stdlib /tmp/madaros-build self-hosted/ir/egraph.sio -o /tmp/out.elf`
  → `Main file: 343 items` → `... lower_array: seed_begin` → SIGSEGV
  (rc 139), under the shell's default `ulimit -s` (8192 KB in this
  environment).
- Same command with `ulimit -s unlimited` prefixed → completes:
  `lower_array: seed_done`, `lower_array: final_fn_count 332`,
  `Merged IR: 332 functions`, `Compilation successful!`, rc 0.
- `min_repro.sio` above reproduces the identical signature and the identical
  cure at that site.
- `gdb -batch -ex run -ex bt` on the crashing `min_repro.sio` invocation
  (no DWARF is emitted by the self-hosted backend, so gdb resolves no
  symbols, but registers are exact):
  ```
  Program received signal SIGSEGV, Segmentation fault.
  0x00000000047c5cc3 in ?? ()
  rsp   0x7fffff28d040
  rbp   0x7fffffd38808
  ```
  `rbp - rsp = 11,188,168` bytes. The compiler's own build log (compiling
  `main.sio`) reports, byte-for-byte: `warning: stack frame too large
  (11188160 bytes) in fn#5774 lower_expr_ref at self-hosted/ir/lower.sio:13277`.
  The 8-byte residual is the pushed return address / saved `rbp`. **The
  fault is on entry to the first call of `lower_expr_ref`, not deep
  recursion** — a single call's own prologue (`sub rsp, 0xaab7c0`) drives
  `rsp` far enough below the mapped stack that the page-walk stack probe
  (`nc_emit_stack_probe_after_sub`, `self-hosted/native/codegen_x86_linux.sio:1815`
  — correctly implemented, confirmed by reading it; this is not a
  missing-probe bug) cannot grow the stack past `RLIMIT_STACK`.

### Root cause: by-value large-aggregate threading (`self: Lowerer`, `self: Checker`)

`self-hosted/ir/lower.sio:13277`:
```sounio
fn lower_expr_ref(self, e: &Expr) -> (Lowerer, i64) with Mut, Panic, Div, Alloc, IO {
```
`self: Lowerer` (not `&Lowerer` / `&! Lowerer`) is taken **by value** and
**returned by value** in a tuple, on every one of the ~20 `ExprKind` match
arms, most of which recurse into `lower_expr_ref` / `lower_binary_expr_ref` /
etc. the same way. `struct Lowerer` (`self-hosted/ir/lower.sio:280-334`)
carries several large fixed-size arrays by value as fields (e.g.
`fo_bind_sens: [i64; 4096]` = 32,768 B; `fo_bind_names: [Name; 128]` where
`Name = { buf: [i8; 128], len: i64 }` = 136 B/entry ≈ 17,408 B;
`variance_base_regs` / `variance_value_regs: [i64; 1024]` each = 8,192 B) —
`sizeof(Lowerer)` is on the order of 90 KB. A single ~90 KB struct copy is
not itself catastrophic; the 11.19 MB frame is the accumulation of **many
separate named `let loN = ...` / `pair` / `pairN` bindings**, each holding a
full `Lowerer` copy, stacked across the function's ~20 match arms (order-of-
magnitude consistent: ~124 × ~90 KB ≈ 11.1 MB).

**This is not a new architectural class.** It is the identical pattern
already root-caused for `Checker` (not `Lowerer`) in
[`MODULAR_COMPILER_STACK_CLASH_2026-05-29.md`](MODULAR_COMPILER_STACK_CLASH_2026-05-29.md)
("Blocker #2"), which measured `sizeof(Checker) ≈ 164 KB` and ~46 stacked
by-value copies producing a 7.6 MB `check_expr` frame, and proposed (but
never landed) a `*mut Checker` refactor. That refactor still has not
happened: `self-hosted/check/check.sio:18697` today —
```sounio
fn check_expr(self, e: Expr) -> (Checker, TypeEntry) with Mut, Panic, Div, Alloc, IO {
```
— is the same by-value `self: Checker` shape, and the current build log
reports its frame has grown to **15,565,712 bytes**. One sampled corpus
file, `tests/run-pass/test_qflra_exact.sio`, crashes here instead of in
`lower_expr_ref`: its last trace line before SIGSEGV is `run_check_mode:
about to check 3 modules` (i.e. inside the **checker**, before IR lowering
even begins), and it is cured by the identical `ulimit -s unlimited` fix.
`tests/run-pass/turbofish_native.sio` crashes at yet another trace point
(`lower_specialized: begin`) — same cure. **The crash's exact trace-print
location is not diagnostic of which bug it is; observed sites are the same
by-value-large-aggregate class, in at least two files (`lower.sio`,
`check.sio`).**

### Systemic pattern (why `ulimit -s unlimited`, not a fixed bound, is the
only reliable mitigation)

`main.sio`'s own build log this session recorded 100+ "stack frame too
large" warnings above 4 MB, not confined to `lower.sio`/`check.sio`:
`self-hosted/native/codegen_x86_linux.sio` alone has frames up to
**123,765,824 bytes** (`native_v2_emit_machine_instr`), 66 MB
(`emit_builtin_print_f64`), 42 MB (`emit_builtin_log`), and dozens more in
the 5-35 MB range. Confirmed directly: `min_repro.sio` under `ulimit -s
65536` (64 MB) clears the `lower_array: seed_begin` site but then segfaults
again later, past `Merged IR: 4 functions` — consistent with one of these
codegen-phase frames. Only `unlimited` cleared the whole pipeline for both
`min_repro.sio` and `egraph.sio` in this session's testing; no fixed finite
bound was found that reliably works for arbitrary input.

### Call path (for the primary `lower_array` site)

```
module_frontend_compile_imported_to_file                  (module_frontend.sio:5948)
  module_frontend_lower_programs_array_direct_box          (module_frontend.sio:5166)
    print("lower_array: seed_begin\n")                     (module_frontend.sio:5181)
    module_frontend_lower_program_items_box_traced_with_externs (module_frontend.sio:4904, called at 5183)
      lower_program_to_ir_summary_box_with_externs_ref      (lower.sio:14125)   -- preseed only, does not crash
      lower_program_bodies_from_summary_with_epistemic_boxed_owned (lower.sio:14415)
        lowerer_new_from_program_summary_owned(...).lower_program_bodies_ref(...)
          ... lower_fn_body / lower_block_ref / lower_stmt_ref ...
            lower_expr_ref(self: Lowerer, e: &Expr)          (lower.sio:13277)  <-- FAULTS HERE
    print("lower_array: seed_done\n")                       (module_frontend.sio:5184, never reached)
```

### Cross-check against prior tracking (principle 12: "one defect, several
catalogues"?)

**Yes — this exact signature is already on record twice before this
dispatch, unfixed both times, plus a third, more recent instance that was
mitigated on the caller side without a root cause:**

- [`MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md`](MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md)
  — the official multimodule witness (`tests/multimodule/thin_single_main.sio`)
  segfaults with the identical trace tail (`imported_compile: lower_begin` /
  `lower_array: seed_begin` / `Segmentation fault`), root-caused only as far
  as "the seed lowering path," not to the specific by-value frame or the
  `ulimit` mitigation.
- [`MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`](MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md)
  — "Bug 2 (OPEN)," identical trace tail, a 2-file minimal repro materially
  identical to this dispatch's, explicitly flagged as needing a debugger
  ("no DWARF... a print-statement bisection is the practical path...not
  attempted here due to unbounded scope"). Its recorded workaround is to
  force `SOUNIO_SOUC_ENGINE=lean_single` for affected code — i.e. avoid
  Madaros for multi-module/f64-heavy programs entirely, rather than raise
  the stack limit.
- [`docs/handoff/particle_exp123_madaros_lower_array_segv_2026-07-25.md`](../handoff/particle_exp123_madaros_lower_array_segv_2026-07-25.md)
  (`BLK-20260725-madaros-exp123-lower-array-segv`) — "SEGV at `lower_array:
  seed_begin`" on the EXP123 particle-physics vertical, closed only by
  **splitting the source file and rewriting call sites** to shrink the IR —
  a mitigation on the caller side (smaller programs accumulate a shallower
  chain of large frames before hitting the same 8 MB default), consistent
  with this dispatch's finding, but never identifying the by-value-threading
  root cause or the one-line `ulimit -s` fix.

None of these three prior reports identified the frame-size mechanism, or
the fact that `bin/madaros` already mitigates it structurally, or that the
fix is one line — each treated it as an unexplained, engine-specific
segfault and worked around it locally (force `lean_single`; split source
files) rather than converging on a shared cause. This dispatch is the first
to name the mechanism, quantify it against a debugger register dump, and
identify the wrapper/gate-script mitigation gap.

[`MODULAR_COMPILER_STACK_CLASH_2026-05-29.md`](MODULAR_COMPILER_STACK_CLASH_2026-05-29.md)
is the architectural sibling (same by-value-threading class, `Checker`
instead of `Lowerer`, `check.sio` instead of `lower.sio`) — its "Blocker #2"
from 2026-05-29 is the same defect class as the `test_qflra_exact.sio`
checker-phase crash observed in this dispatch's sampling, still unresolved
two months later.

### Blast radius

**Full-corpus, using the actual gate script, unmodified** (not a sample):

| Invocation | Total failures / 1688 | NEW vs. 314-entry baseline |
|---|---:|---:|
| `SOUNIO_MADAROS_CORPUS_BIN=<raw ELF>` (current script behaviour) | 1263 | **1028** |
| `SOUNIO_MADAROS_CORPUS_BIN=<wrapper: `ulimit -s unlimited`; exec same ELF>` | 305 | **0** |

Both rows are the same commit, same from-source ELF, same machine, run back
to back; only the stack ulimit at compiler-invocation time differs. The
wrapper is a 2-line script living outside the repo
(`ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true;
exec <elf> "$@"`) — it changes nothing about `madaros_corpus_regression_gate.sh`
itself. **All 1028 "new" failures the gate currently reports are attributable
to the gate's missing `ulimit -s` guard**, not to 1028 independent new
behaviours in the compiler.

A stratified qualitative sample (42 files, every 40th entry of sorted
`tests/run-pass/*.sio`) run individually against the raw (unwrapped) ELF,
to characterise *what* the crashes look like before the full-corpus
comparison above was run, found:

| Signature | Count (of 42) | Cured by `ulimit -s unlimited`? |
|---|---:|---|
| SIGSEGV at/after `lower_array: seed_begin` or an equivalent trace point (this dispatch's bug class) | 30 | 30/30 (100%) |
| `error: visibility preflight failed` (E011 "no method named for this type" — a genuine but **orthogonal, pre-existing, non-crashing** type error from checking a cross-module test standalone, without its sibling files; see [`CHECKER_GUARD_WIRING_DISPATCH_2026-07-11.md`](CHECKER_GUARD_WIRING_DISPATCH_2026-07-11.md)'s "~28% spurious failures from standalone per-file checking" finding, a strikingly similar rate) | 12 | n/a — not a crash, and (per the full-corpus result above) not counted as *new* either — these are already in the 314-entry baseline |

The full-corpus comparison supersedes the sample's numeric ratio (which
would have suggested a residual ~29% non-crash population among the "new"
1028) — the true residual after the `ulimit` fix is exactly 0 *new*
failures; the E011 population exists but was apparently already fully
captured in the pre-existing 314-entry baseline, not among the 1028
increment.

## Per-issue triage

| # | What | Where | Status |
|---|---|---|---|
| 1 | `Lowerer` threaded by value through `lower_expr_ref` and siblings → 11.19 MB frame → stack overflow on first call | `self-hosted/ir/lower.sio:13277` (and ~15 sibling `lower_*_ref` functions in the same file with 4-14 MB frames per the build log) | OPEN — primary driver of the `lower_array: seed_begin` signature |
| 2 | `Checker` threaded by value through `check_expr` and siblings → 15.57 MB frame → stack overflow on first call | `self-hosted/check/check.sio:18697` (dozens of sibling `check_*` / `lower_type_expr` / `collect_*` functions, 4-24 MB frames per the build log) | OPEN — same class as `MODULAR_COMPILER_STACK_CLASH_2026-05-29.md` Blocker #2, never fixed; independently reachable (`test_qflra_exact.sio`) |
| 3 | Same by-value-large-aggregate class also present in `self-hosted/native/codegen_x86_linux.sio` (frames up to 123.8 MB) | native codegen | OPEN, not scoped further here — flagged only because it is why `ulimit -s 65536` clears the lowering crash but not the whole pipeline for `min_repro.sio` |
| 4 | `scripts/ci/madaros_corpus_regression_gate.sh` invokes a raw Madaros ELF with no `ulimit -s` guard, unlike ~20 sibling gate scripts | `scripts/ci/madaros_corpus_regression_gate.sh` | OPEN — cheapest, lowest-risk fix; measured in this dispatch to take the corpus regression count from 1028 new / 1263 total to 0 new / 305 total, with zero compiler changes |
| 5 | `.github/workflows/ci.yml:298-309` comment asserts "all traced to one segfault" and quotes the 1028 figure without the `ulimit` context | CI comment, not code | Imprecise per this dispatch — the true story is "1028 of 1263 failures are gate-invocation artefacts of a missing `ulimit -s` guard, on top of a real but already-mitigated-elsewhere compiler defect that exists in at least two files" |

## Implementation notes (for whoever picks this up — not implemented here,
report-only per protocol)

1. **Do not "fix" this by raising a number.** Correct minimal action items,
   in order of cost:
   - (a) Add `ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true`
     to `scripts/ci/madaros_corpus_regression_gate.sh`, matching every other
     Madaros gate script in this repo, and refresh
     `tests/madaros_corpus_baseline.txt` from the result (this dispatch
     measured what that refresh would look like — 305 entries — but did not
     perform it; refreshing the baseline is explicitly out of scope for
     this dispatch per the task that produced it). This alone is very
     likely sufficient to make wiring the gate into CI (currently
     deliberately not wired, per `ci.yml:298`) tractable.
   - (b) Correct the `ci.yml:298-309` comment once (a) lands.
   - (c) The real compiler fix — converting `Lowerer` / `Checker` (and any
     other large struct threaded the same way) from by-value `self` /
     by-value return to `&!` in-place mutation — is the same large,
     mechanical, multi-session refactor already proposed and shelved in
     `MODULAR_COMPILER_STACK_CLASH_2026-05-29.md`. It removes the defect
     class instead of raising the ceiling on it, and is the only fix that
     helps in environments where a hard `RLIMIT_STACK` ceiling below what
     `unlimited` would want is enforced by policy (a container or CI runner
     that caps the *hard* limit would make `ulimit -s unlimited` silently
     fail — the `2>/dev/null || true` pattern used throughout this repo
     does not surface that failure; this environment's hard limit happens
     to be `unlimited` itself, so it was not hit here, but should not be
     assumed universal).
2. **Verify against `madaros --self-test`** (`self-hosted/compiler/main.sio`,
   ~1156 tests) once (a)/(c) are attempted: its own comment at line 21758
   ("a pre-existing crash downstream of T1201") is plausibly this same
   class, given `--self-test` compiles/checks many programs in a single
   process and would accumulate exactly the frames described here. Not
   verified in this dispatch (out of scope: diagnose-and-report only).
3. **Re-run `bash scripts/ci/build_modular_madaros.sh` fresh** (never a
   prebuilt binary, CLAUDE.md principle 15) before re-measuring; every
   number in this dispatch is from a from-source build made in this
   session, at the commit this branch is based on.

## Recommendation

Land item 4 (the one-line `ulimit -s` guard in
`madaros_corpus_regression_gate.sh`) first — it is a test-infrastructure
fix, not a compiler change, so it does not need the forensic-dispatch-
before-code gate that `self-hosted/` changes do, and this dispatch has
already measured its effect (1028 new -> 0 new). Treat items 1-3 (the real
by-value-threading defect in `lower.sio` / `check.sio` /
`codegen_x86_linux.sio`) as a separate, larger compiler dispatch — the
`*mut Checker` / `*mut Lowerer` refactor already scoped (and shelved) in
`MODULAR_COMPILER_STACK_CLASH_2026-05-29.md` — rather than folding it into
the corpus-gate cleanup.

<!-- docs:meta
topic_id: repo.docs.audit.g1-let-spine-crash-rootcause-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-let-spine-crash-rootcause-2026-06-01
-->

# G1 `let x = 1` --check SIGSEGV — root cause: runaway recursion, not frame size

> **⚠️ SUPERSEDED 2026-06-10** — no longer reproduces. Re-measured in
> [`MODULAR_COMPILER_AUDIT_2026-06-10.md`](MODULAR_COMPILER_AUDIT_2026-06-10.md):
> the 2026-06-07 modular binary (`artifacts/self-hosted/souc-mc-check.elf`)
> passes `--check` on the full bisection table of this doc (`fn main() {}`,
> `fn main() { 1 }`, `let x = 1`, hello) with rc=0, 8 MB stack, no VmStk
> growth pathology observed (sub-second completion). The 2026-06-02 control
> binary (`artifacts/omega/modular_sret_diag/fresh2_modular_souc.elf`) still
> crashes 100%, confirming the fix landed between those builds. Binary-level
> evidence; fixing commit not attributed.

**Date:** 2026-06-01
**Author:** Claude (integration-shepherd lane, read-only diagnosis to assist the
move-codegen agent — no edits made to `/workspace/sounio-move-codegen`).
**Subject binary:** `/workspace/sounio-move-codegen/.dbg/mc.elf` (85 MB, built
01:05 today by the move-codegen agent) on `.dbg/let.sio` = `fn main() { let x = 1 }`.
**Branch:** `modular/move-codegen` HEAD `e418c5946` ("G1 let-spine COMPILES but
`let x=1` still SIGSEGVs (G1 NOT achieved)").

## The decisive experiment

Ran the *exact* failing invocation (`mc.elf --check let.sio`) and **directly
measured stack consumption** via `/proc/<pid>/status:VmStk` while it ran under an
unlimited stack. The binary was copied to `/tmp` first; the worktree was not
touched.

**VmStk (resident stack) over wall-clock, `ulimit -s unlimited`:**

| t (≈0.5 s steps) | VmStk |
|---|---|
| t1 | 132 kB |
| t2 | 835 MB |
| t3 | 1.76 GB |
| t4 | 2.70 GB |
| t5 | 3.65 GB |
| t6 | 4.61 GB |
| t7 | 5.54 GB |
| t8 | 6.48 GB |

Stack grows **monotonically at ~940 MB/s, reaching 6.5 GB in ~4 s** to type-check
`let x = 1`.

Corroborating `ulimit -s` sweep (rc only):

| `ulimit -s` | result | rc |
|-------------|--------|----|
| 8 MB (default) | SIGSEGV | 139 |
| 512 MB | SIGSEGV | 139 |
| unlimited | grows to GBs, then timeout | 124 |

## What this proves

The crash is **unbounded stack growth = runaway (non-terminating) recursion**,
*not* a fixed-size frame overflow and *not* a bad-`*mut` pointer deref.

- **VmStk is the decisive signal.** It climbs to 6.5 GB and rising. A bad-pointer
  deref (the layout-sensitive `*mut` hypothesis) would fault with VmStk **flat at
  KB-scale** — it would never consume gigabytes of stack. Only deep recursion
  grows VmStk like this. The hypothesis that the 3-point `ulimit` pattern was mere
  address-space relocation is ruled out: the stack is genuinely being consumed.
- The `ulimit` rc-sweep alone was **ambiguous** (rc=139 is identical for a
  stack-overflow and a wild deref). VmStk is what disambiguates; do not rely on the
  rc pattern by itself.
- A bounded call chain — even the worst non-recursive path
  (`collect_fn_def` 6.4 MB → `check_expr` 12.3 MB ≈ 20–40 MB) — would crash at
  8 MB but survive at 512 MB. The 512 MB crash + 6.5 GB VmStk both point past any
  fixed depth → recursion.

## Why this reframes the G1 approach

The huge per-function frames the build log reports are **real but secondary** —
they are the *amplifier*, not the root cause:

```
check.sio frames on/near the let path (from .dbg/build.log):
  lower_type_expr            19.5 MB   check.sio:8714   (fires on `let x: T = ..`)
  check_method_call          15.0 MB   check.sio:13316
  check_expr                 12.3 MB   check.sio:12677  (unavoidable for any RHS)
  check_field_access          9.6 MB   check.sio:13159
  collect_item                8.1 MB   check.sio:9767
  check_binary_expr           7.2 MB   check.sio:12775
  collect_fn_def              6.4 MB   check.sio:11000
  ... 86 frames > 4 MB total, emitters up to 68 MB (wasm/macho, off the --check path)
```

With a 12 MB frame, a runaway recursion overflows an 8 MB stack on the *first or
second* re-entry — so it *looks* like "frame too large." But shrinking frames via
*mut (the current G1 strategy) will not stop the recursion; it will only let it
spin more iterations before overflowing (or hang instead of crash once frames are
small enough). This matches the observed history exactly: "G1 let-spine now
COMPILES but `let x=1` still SIGSEGVs."

**The fix must locate and break the non-terminating recursion**, then the frame
work becomes a performance/robustness cleanup rather than the blocker.

## It is NOT the `let` path — minimal repro is a bare expression

Bisection with `mc.elf --check` on minimal programs:

| program | rc |
|---|---|
| `fn main() {}` | 0 ✓ |
| `fn main() { 1 }` | **139** ✗ |
| `fn main() { let x = 1 }` | 139 ✗ |
| `fn main() { let x = 1 + 2 }` | 139 ✗ |
| `fn main() { let x: i64 = 1 }` | 139 ✗ |
| `fn f() {} fn main() {}` | 0 ✓ |

An empty body passes; **a single bare expression statement `1` crashes**. So the
recursion is in the **expression-checking path**, triggered by *any* expression in
a function body — **not** the `let` binding. `fn main() { 1 }` is the minimal repro
(simpler than `let x = 1`); the G1 StmtLet *mut spine is not the locus.

## Where the recursion lives (source trace) — and a staleness caveat

Statement dispatch routes the bare `1` as:
`StmtExpr` → `checker_check_expr_stmt_inplace` (2470) →
`checker_check_opt_expr_inplace` (2474) → `checker_check_expr_inplace` (2489).

In the **current source on disk**, `checker_check_expr_inplace` handles
`ExprIntLit` *inline* (returns `ty_i64()`, check.sio:2496–2500) — a clean leaf, no
recursion. The only recursive escape hatch is the fallback
`_ => checker_check_expr_mut(c, e)` (2511), and `checker_check_expr_mut` (2483)
bounces to the **by-value** `check_expr` (12677) via `(*c).check_expr(e)` — the
unconverted spine that still copies the 164 KB `Checker` and carries the 12.3 MB
frame.

**Caveat (verify first):** the source as written should route `1` through the
inline leaf and *not* crash — yet `mc.elf` (built 01:05) crashes on `1`. The most
likely explanation is that **the binary is stale relative to the source**: the
inline `ExprIntLit`/literal fast-path in `checker_check_expr_inplace` appears to
post-date the build. So the move-codegen agent may have *already* partially fixed
this in source. **First action: rebuild `mc.elf` and re-run the bisection table +
VmStk.** If `fn main() { 1 }` then passes but `fn main() { f() }` (a call, which
hits the `_ =>` fallback → by-value `check_expr`) still grows VmStk, the recursion
is confirmed to live in the by-value `check_expr` path (12677) and the fix is to
complete its `*mut` conversion so non-leaf expressions stop bouncing to it.

## Tooling note

This pod has **no gdb / valgrind / strace / catchsegv / eu-stack**, and `mc.elf`
exposes no `nm` symbols, so addr2line can't resolve the fault — which is why this
diagnosis used VmStk sampling + minimal-program bisection instead of a backtrace.
If a debugger is installed, the stack under unlimited will show the recursing
function repeated thousands of times; that names it directly. The VmStk method
(`watch grep VmStk /proc/<pid>/status`) is the debugger-free confirmation: a fixed
build must peak at KB–MB, not GB.

## Reconciliation with prior memory

Memory `project_move_codegen_premise_overturned` states "setrlimit band-aid DEAD
(crash survives `unlimited` stack)," and `project_modular_B_repro_verdict` /
`project_modular_span_sensitive_crash` characterize the crash as
"layout-sensitive / non-monotonic / non-bisectable" — i.e. the standing view was a
codegen/pointer defect, not recursion.

The VmStk measurement overturns that for *this* binary/state: the stack genuinely
grows to gigabytes (6.5 GB and climbing), which only deep recursion does. The old
"survives unlimited" note is consistent with recursion too — at ~940 MB/s, an
unlimited stack still exhausts the address space and SIGSEGVs eventually if you
wait long enough, so a longer-running earlier test would also have seen rc=139 and
(reasonably but wrongly) read it as "the crash is independent of stack size."

**Caveat for the move-codegen agent:** the prior layout-sensitivity reports were
real observations. It is possible the modular checker has *two* distinct failure
modes (a recursion in the `let`/expr path **and** a separate layout-sensitive
`*mut` deref elsewhere), and this repro isolates the recursion one. Confirm the fix
by re-measuring VmStk: after breaking the recursion, `let x = 1 --check` should
peak at KB–MB, not GB.

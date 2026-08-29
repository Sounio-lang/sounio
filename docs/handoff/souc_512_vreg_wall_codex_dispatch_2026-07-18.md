<!-- docs:meta
topic_id: repo.docs.handoff.souc-512-vreg-wall-codex-dispatch-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.souc-512-vreg-wall-codex-dispatch-2026-07-18
-->

# Prompt — CODEX-2: fix the souc codegen silent-miscompile wall (Madaros native_v2, per-function 512-vreg ceiling)

**For:** CODEX-2 (compiler-internals agent; owns `self-hosted/`)
**Authored by:** Claude (bignat/BigInt calibration-note forensic investigation), 2026-07-18
**Type:** self-hosted compiler — Madaros native x86 backend (register allocation / frame layout /
codegen), **not** math, **not** lean_single. This is a **runtime/codegen correctness** defect: it
produces wrong values with a clean exit (no crash, no diagnostic).
**Status:** structural hypothesis, multiply-attested statically, **not yet confirmed as the live
trigger on the shipped binary** — see Honest Confidence below. This is a forensic handoff, not a patch.

---

## TL;DR

The Madaros backend's native_v2 register allocator/codegen hard-codes a **per-function ceiling of 512
IR virtual registers** at four independent layers, with **no diagnostic when exceeded** at any of them.
Past that ceiling, loads silently return zero, stores are silently dropped, register-allocation
intervals are silently discarded, and the stack frame is silently undersized — so a compiled program
can exit 0 while printing wrong values. This is the mechanism `stdlib/math/bignat.sio`'s calibration
note and `tests/run-pass/bignat_selftest.sio`'s companion-file split are both defending against. I could
not synthetically reproduce the corruption (see Notes), so the fix ask starts with **turning the four
silent-failure sites into a loud compile error**, which is strictly safe regardless of whether 512 is
the exact trigger, and gives CODEX-2 a debuggable signal to locate the true minimal repro.

## Why it matters

This wall gates the entire "exact/arbitrary-precision path is oracle-gated because the compiler cannot
be silently trusted" posture behind the exact-algebra and stdlib-deepen lanes (see
`memory: stdlib-deepen-batches-2026-07-15`, `madaros-file-io-broken-2026-07-13`). Every BigInt/BigNat,
CD-tower exact, and rational-arithmetic gate currently has to cross-check against a Python oracle
because "compiled and exited 0" is not a correctness signal on Madaros — this defect is *why*. It also
blocks scaling any single self-test/selftest-style program past whatever call-site/reg-count budget
happens to fit under 512, which is precisely the "overall superior to pandas" roadmap's requirement
that stdlib self-tests scale to realistic op-counts without hand-splitting into many tiny programs.
Fixing the silent half of this (loud-fail instead of silent-wrong) is a prerequisite for trusting any
future codegen work on Madaros; fixing the capacity half unblocks writing consolidated, oracle-free-at-
scale self-tests.

## What exists

- `bin/souc` (8.3KB bash launcher) routes to the **Madaros** engine by default
  (`artifacts/self-hosted/madaros`, built from `self-hosted/compiler/main.sio`); `SOUNIO_SOUC_ENGINE=
  lean_single` forces the legacy single-file backend, a materially different and (per this
  investigation) more tolerant code path.
- `self-hosted/compiler/main.sio:45` — `use native::codegen_x86_linux::*` confirms Madaros' default
  path is the native_v2 backend under investigation.
- `stdlib/math/bignat.sio:33-52` carries a calibration note: whole-program codegen wall, silently
  emits WRONG VALUES (swapped add/sub, sign flips, corrupted digits) with a clean exit; explicitly NOT
  a statement/function-count threshold and NOT a struct-size/byte budget (shrinking `LIMBS` 64→4 did
  not move the wall).
- `tests/run-pass/bignat_selftest.sio:2-6` states outright: "madaros can SIGSEGV during codegen ...
  corrupted digits with a clean exit" — this pins the failing engine to Madaros, not lean_single.
- `tests/run-pass/bignat_selftest_*.sio` (eq_true/eq_false/iszero_true/iszero_false/divmod_rem/signed)
  are each kept in their **own** compiled program "per the capacity-wall note" — i.e. the existing test
  suite is already working around this defect by hand-splitting, which is itself evidence the wall is a
  per-compiled-program accumulation effect, not a per-file source-size effect.

## Root-cause hypothesis (leading structural hypothesis; not confirmed as sole live trigger)

The Madaros native_v2 backend hard-codes a per-**function** ceiling of **512 IR virtual registers**,
silently, at four independent layers:

1. **Register/frame state arrays are fixed at `[i64; 512]`.**
   `self-hosted/native/frame.sio:259,265,313,314` — the `NativeCompiler` state arrays `vreg_to_preg`,
   `vreg_spill_slots`, `is_float_reg`, `reg_type` are all `[i64; 512]`.
2. **Frame-size scan silently truncates at 512.**
   `self-hosted/native/lower_ir.sio:133-146` — `nc_min_frame_size` loops
   `while v < func.reg_count && v < 512`, so once `reg_count > 512` the emitted stack frame is sized to
   cover only the first 512 slots.
3. **Load/store for temp≥512 are silent no-ops that fabricate/drop data.**
   `self-hosted/native/codegen_x86_linux.sio:5106-5124` — `native_v2_load_temp_to_rax` emits
   `xor rax, rax` (a silent **zero**, not an error) for any `temp >= 512`.
   `self-hosted/native/codegen_x86_linux.sio:5126-5130` — `native_v2_store_rax_to_temp` **silently
   returns without emitting** the store for `temp >= 512` — the value is dropped on the floor.
4. **Register allocator silently drops intervals/liveness past 512.**
   `self-hosted/native/regalloc.sio:16,98` — `RA_MAX_INTERVALS = 512`; `add_interval` is guarded by
   `if s.count < RA_MAX_INTERVALS` with no else-branch, so intervals past 512 are silently dropped from
   allocation.
   `self-hosted/native/regalloc.sio:436-438,456,478` — liveness `defs`/`uses`/`seen` are `[i64; 512]`,
   guarded by `vreg >= 0 && vreg < 512`.

Meanwhile the slot-offset math itself has **no cap** —
`self-hosted/native/frame.sio:447` `ir_slot_offset(vreg) = -(vreg+1)*8` will happily compute an offset
for any vreg — and reg_count growth is **unbounded with no warning**:
`self-hosted/ir/lower.sio:2065-2066` `reg_count = reg + 1` grows per-fresh-temp with no ceiling check.
`self-hosted/native/frame.sio:112-131` `x86_align_frame_size_for_calls` only pads 8 bytes for stack
alignment; since the prologue subtracts the (truncated, per #2) `frame_size` from `rsp`, slots computed
for `vreg >= 512` land at or below the live `rsp` — any subsequent `CALL` clobbers them via the callee's
own red-zone/pushes.

**Net effect**: once a single function's SSA `reg_count` crosses 512, high-numbered vregs (a) load as
zero, (b) drop their stores, (c) are dropped from register allocation, and/or (d) resolve to raw
`ir_slot_offset` addresses at/under `rsp` that get clobbered by any call — all four failure modes are
*silent*, and any one of them alone is sufficient to explain "corrupted digits with a clean exit."

`reg_count` is a **per-function SSA temp count**, so the trigger is call-site/expression-shape sensitive,
not iteration-count sensitive: a loop that reuses a small fixed set of temps stays safe indefinitely,
while the same work **unrolled** (or written as a long chain of distinct call-sites) mints a fresh batch
of SSA temps per site and can march `reg_count` past 512 within one function body. BigNat is addressed
by a single base-pointer vreg (not limb-expanded into N vregs), which is consistent with the note's
observation that `LIMBS` 64→4 did not move the threshold — shrinking the struct doesn't reduce the
number of *call-sites*, which is the axis that actually drives `reg_count`.

The single cleanest "silent wrong value" mechanism, and the best first thing to falsify or confirm, is
`native_v2_load_temp_to_rax` returning `xor rax,rax` for `temp >= 512` paired with
`native_v2_store_rax_to_temp` dropping the corresponding store.

### Evidence (file:line citations)

- `stdlib/math/bignat.sio:33-52` — calibration note (whole-program wall, wrong values not crash, not
  byte-budget, `LIMBS` 64→4 didn't move it)
- `tests/run-pass/bignat_selftest.sio:2-6` — "madaros can SIGSEGV during codegen ... corrupted digits
  with a clean exit" (pins engine to Madaros)
- `tests/run-pass/bignat_selftest_*.sio` — each kept in its own compiled program "per the capacity-wall
  note"
- `bin/souc:1-31` — launcher; default engine Madaros; `SOUNIO_SOUC_ENGINE=lean_single` for legacy path
- `self-hosted/compiler/main.sio:45` — `use native::codegen_x86_linux::*`
- `self-hosted/native/frame.sio:259,265,313,314` — `[i64;512]` state arrays
- `self-hosted/native/lower_ir.sio:133-146` — `nc_min_frame_size` truncates scan at `v < 512`
- `self-hosted/native/codegen_x86_linux.sio:5106-5124` — silent-zero load past 512
- `self-hosted/native/codegen_x86_linux.sio:5126-5130` — silent-dropped store past 512
- `self-hosted/native/regalloc.sio:16` — `RA_MAX_INTERVALS = 512`
- `self-hosted/native/regalloc.sio:98` — `add_interval` silently drops past capacity
- `self-hosted/native/regalloc.sio:436-438,456,478` — liveness `[i64;512]` guarded by `vreg < 512`
- `self-hosted/native/frame.sio:447` — `ir_slot_offset` has no cap
- `self-hosted/native/frame.sio:112-131` — `x86_align_frame_size_for_calls` only pads for alignment;
  slots for `vreg >= 512` land at/under `rsp`
- `self-hosted/ir/lower.sio:2065-2066` — `reg_count` grows unbounded, no diagnostic
- `self-hosted/compiler/lean_single.sio:1031,1550` — legacy engine's `local_bss_spill_bytes()` = 1MB and
  `VAR_SLOT[2048]`/unbounded `NEXT_SLOT` with disp32 addressing explain why lean_single tolerated every
  stress test below; it is a structurally different, more tolerant path and must not be conflated with
  Madaros when reasoning about this defect.
- `self-hosted/ir/lower.sio:336-337` — `SOUNIO_LOWER_LIVE_TRACE=1` gate exists in source but produced
  **no output** from the shipped binary, indicating a possible source-vs-binary provenance gap (see
  Notes / acceptance prerequisite).

## Minimal repro (best current understanding — NOT yet confirmed to trigger corruption)

Take `tests/run-pass/bignat_selftest.sio` plus its `bignat_selftest_*.sio` companions (which currently
pass **only** because each is compiled as its own program) and **inline all their cases into one
`main`**, compiled under the default Madaros engine. The distinguishing axis is **call-site count**, not
iteration count:

```
// SAFE (reuses a small fixed temp set; reg_count stays low regardless of iteration count):
while i < n { acc = big_add(acc, x); i = i + 1; }

// SUSPECT (mints fresh SSA temps at every site; reg_count grows with source-level call-site count):
acc = big_add(acc, x1);
acc = big_add(acc, x2);
acc = big_add(acc, x3);
// ... repeated until the function's reg_count crosses 512
```

Each `big_add`/`big_sub`/`big_mul`/`big_divmod` call is estimated to mint ~10-30 fresh SSA temps
(argument marshalling of the by-value BigInt struct + capture of the by-value struct return), so on the
order of a few dozen distinct unrolled call-sites within one function body is the estimated ballpark to
cross 512 — this has **not** been confirmed by directly dumping `reg_count`, only inferred from the four
static sites above plus the shape of the existing per-file test split; treat it as a rough estimate, not
a measured figure.

Two refinements CODEX-2 should consider if the straightforward inline doesn't trigger it:
- `unat_mul`'s O(n²) nested limb loop and `unat_divmod`'s quotient binary search each emit many temps
  internally; the true minimal repro may need to be **inside** one of those op bodies rather than at the
  call-site level.
- High simultaneous liveness (many BigInt locals alive at once, forcing spills) rather than raw
  call-site count may be the actual pressure axis — worth instrumenting both independently.

## The fix ask

1. **Loud-failure gate (do this first, regardless of what the repro confirms).** Replace all four silent
   past-512 behaviors with a hard compile error:
   - `native_v2_load_temp_to_rax` (codegen_x86_linux.sio:5106-5124): error instead of `xor rax,rax` for
     `temp >= 512`.
   - `native_v2_store_rax_to_temp` (codegen_x86_linux.sio:5126-5130): error instead of silently
     returning.
   - `regalloc.sio:98` `add_interval`: error instead of silently dropping past `RA_MAX_INTERVALS`.
   - `lower_ir.sio:133-146` `nc_min_frame_size`: error (or extend the scan) instead of truncating at
     `v < 512`.
   This alone converts every currently-silent-wrong-value program into either a correct compile or a
   loud compiler error — never again a clean exit with wrong output.

2. **Capacity fix.** Raise (or make dynamic, sized to the actual function's `reg_count`) the `[512]`
   arrays: `vreg_to_preg`, `vreg_spill_slots`, `is_float_reg`, `reg_type` (frame.sio), `RA_MAX_INTERVALS`
   and the liveness `[512]` arrays (regalloc.sio), and the `nc_min_frame_size` bound (lower_ir.sio).
   Dynamic sizing (allocate to `func.reg_count`) is preferred over merely raising the constant, since the
   constant will otherwise just move the wall to a larger but still-silent boundary.

3. **Provenance prerequisite.** Rebuild `artifacts/self-hosted/madaros` from the current `self-hosted/`
   tree before drawing conclusions from `SOUNIO_LOWER_LIVE_TRACE=1` — it currently emits nothing against
   the checked-in binary, which is inconsistent with the trace gate present in `lower.sio:336-337`
   source and suggests the shipped artifact may not be built from this exact tree.

## Acceptance criteria

- **Prediction test (run first):** compile the grouped bignat program (task 2 below) under Madaros with
  `SOUNIO_LOWER_LIVE_TRACE=1` against a freshly-rebuilt binary and dump each function's `reg_count`. The
  hypothesis predicts the corrupting grouping contains a function with `reg_count > 512`, while every
  currently-safe per-file split stays `< 512`. Confirm or refute before investing further in the 512
  boundary specifically; if refuted, redirect to `abi_lower.sio` struct-by-value copy lowering or the
  `reloc.sio:65` `c.relocs` 256-cap path as the next candidate.
- **No silent wrong-value exits.** After the loud-failure gate lands, no program that previously
  silently miscompiled can exit 0 with incorrect output — it must either compile correctly or fail with
  a diagnostic naming the exceeded capacity.
- **Grouped bignat self-test passes at scale.** The single grouped program that inlines every
  `bignat_selftest*.sio` case into one `main` prints output bit-for-bit equal to
  `scripts/research/bignat_oracle.py` (or an equivalent Python arbitrary-precision oracle) across all
  cases, run as ONE compiled program (not the current hand-split set), and continues to pass at
  ~10x the current op-count without needing manual re-splitting or oracle babysitting to catch silent
  corruption.
- **Regression oracle in CI.** A CI gate compiles the grouped bignat self-test and diffs stdout against
  the Python oracle, so "compiled and did not crash" is never again treated as a correctness signal for
  this lane.
- **Lean_single not touched.** This defect and its fix are scoped to the Madaros native_v2 backend; do
  not port capacity changes into `lean_single.sio` without separately establishing it needs them — this
  investigation found lean_single materially more tolerant of every stress shape tried (see Notes).

## Honest confidence / notes for CODEX-2

The 512-per-function ceiling in the Madaros native_v2 backend is the best-supported **structural**
hypothesis available from static reading — four mutually-corroborating layers, all silent past 512, all
inside the exact engine the bignat calibration note explicitly blames. It is **not** confirmed to be the
actual trigger on the shipped binary, for two reasons worth weighing before committing engineering time
purely to raising the 512 constant:

1. **No direct reproduction.** Every synthetic repro built during this investigation to cross raw
   vreg/call-site/simultaneous-liveness counts produced **correct** output on both engines:
   - `/tmp/wall2.sio` — 800 runtime-seeded live scalar `let`s, lean_single, correct.
   - `/tmp/callwall.sio` — 500 distinct function call-sites, lean_single, correct.
   - `/tmp/bvwall.sio` — 199 wide `[i64;64]` by-value struct call-sites, correct on **both**
     lean_single and Madaros default.
   - `/tmp/simul.sio` — 600 simultaneously-live wide by-value structs (low/high index pairing),
     lean_single correct; Madaros aborted on an unrelated E061 unused-variable preflight, not the wall.
   So the 512 ceiling was never observed to actually corrupt anything directly — only inferred from
   source.
2. **Source-vs-binary provenance gap.** `SOUNIO_LOWER_LIVE_TRACE=1` produced no output from the shipped
   `artifacts/self-hosted/madaros`, despite the trace gate existing in `lower.sio:336-337` source. This
   means the file:line evidence above describes the *current source tree*, but it is not yet confirmed
   that the *live compiled binary* was built from this exact tree — rebuild first (task 3 above) before
   trusting any reg_count instrumentation against it.

**Two engines exist and must not be conflated.** The calibration-note SIGSEGV/corruption phenomenon is
explicitly a **Madaros** (native_v2) phenomenon per `bignat_selftest.sio`'s own comment. `lean_single`
(legacy single-file codegen: `VAR_SLOT[2048]`, unbounded `NEXT_SLOT`, 1MB BSS-spill threshold before a
520-byte BigNat would ever spill) is a structurally different and, per every stress test run here,
materially more robust path. Do not fix lean_single symptoms by reasoning about Madaros source, or vice
versa.

**Highest-information next step:** reproduce the wall from the *actual* bignat op mix (inline the
selftest companions into one `main`, Madaros engine, freshly rebuilt binary) and directly dump
per-function `reg_count` via the trace gate — this is strictly more informative than further synthetic
pressure tests, which have so far all come back negative. If `reg_count` stays under 512 in a grouping
that is known (from the existing hand-split test layout) to corrupt, the 512-ceiling hypothesis is wrong
and the search should shift to `abi_lower.sio`'s struct-by-value copy lowering or the `reloc.sio:65`
`c.relocs` 256-entry cap, if that relocation path is live on the Madaros side.

No compiler file was edited during this investigation — this is a forensic handoff only.

## Pointers

- `stdlib/math/bignat.sio:33-52` (calibration note)
- `tests/run-pass/bignat_selftest.sio` + `tests/run-pass/bignat_selftest_*.sio` (current hand-split
  workaround, and the inlining target for the minimal repro)
- `self-hosted/native/frame.sio` (`[512]` state arrays, `ir_slot_offset`, frame-size alignment)
- `self-hosted/native/lower_ir.sio` (`nc_min_frame_size` truncated scan)
- `self-hosted/native/codegen_x86_linux.sio` (silent load-zero / store-drop past 512)
- `self-hosted/native/regalloc.sio` (`RA_MAX_INTERVALS`, liveness arrays, `add_interval`)
- `self-hosted/ir/lower.sio` (`reg_count` growth, `SOUNIO_LOWER_LIVE_TRACE` gate)
- `self-hosted/compiler/lean_single.sio:1031,1550` (legacy engine's more tolerant analog, for contrast
  only — not the target of this fix)
- Prior CODEX-2 dispatch for format/precedent:
  `docs/handoff/compiler_651_defects_codex_dispatch_2026-07-15.md`
- Related memory: `stdlib-deepen-batches-2026-07-15.md`,
  `madaros-file-io-broken-2026-07-13.md` (both describe living with/around silent Madaros codegen
  defects on the same lane this wall gates)

## Out of scope

- lean_single engine internals (different, more tolerant backend; not the calibration-note's engine).
- Any struct-size/`LIMBS`-tuning approach to bignat — the calibration note and this investigation both
  establish the wall is call-site/reg-count driven, not byte-budget driven; shrinking the struct will
  not move the threshold.
- GPU/K-AXI codegen, EISA — unrelated backend surfaces.
</dispatch_md>


---

## EMPIRICAL CONFIRMATION — 2026-07-18 (correction: live observable is a CRASH, not a silent wrong value)

A follow-up experiment (single-file, one `fn main`, an UNROLLED chain of ~N distinct `let vK = v(K-1) + K` bindings — a loop would reuse temps, so unrolling is essential — with oracle = closed form N(N+1)/2) **confirms the 512-vreg ceiling as a live, deterministic wall**, but corrects the observable:

| N | oracle | DEFAULT (madaros) | lean_single |
|---|---|---|---|
| 338 | exit 42 | **42 (ok)** | 42 (ok) |
| 340 | exit 221 | **139 (SIGSEGV)** | 221 (ok) |
| 512 | exit 79 | **139 (SIGSEGV)** | 79 (ok) |
| 700 | exit 101 | **139 (SIGSEGV)** | 101 (ok) |

**Sharp boundary: N=338 clean+correct → N=340 crash.** madaros/native_v2 only; **lean_single is immune** (correct to N>1000). The exit-code shape (result returned via `(vN % 250)+1`, no strings) rules out string/pointer confounds — it is value-path register pressure.

**Correction to the framing above:** the reproducible observable at the 512 crossing is a **deterministic SIGSEGV (exit 139), not a clean-exit wrong value.** The silent-zero (`xor rax,rax` for vreg≥512 load) and dropped-store primitives **do exist in source** (`lower_ir.sio` load ~87-90 / store ~110-112, `regalloc.sio` RA_MAX_INTERVALS=512), but they are **coupled** to UNGUARDED `while v < reg_count { ... vreg_to_preg[v] ... }` loops that read the fixed `[i64;512]` preg-mask arrays **out of bounds** once reg_count>512 (`frame.sio:103`, `codegen_x86_linux.sio:5276`, `codegen.sio:3456`). Those OOB reads corrupt the callee-saved push/pop mask → prologue/epilogue imbalance → SIGSEGV, which fires at the **same** threshold and **masks** the clean silent-miscompile path. The bignat CALIBRATION NOTE's historically-observed "corrupted digits with clean exit" is the *other* face of the same 512 ceiling — reachable only when the OOB mask reads happen to land on benign memory (adjacent `vreg_spill_slots` = -1); generic arithmetic chains crash instead.

**Net for the fix:** this is *good news for debuggability* — there is a **deterministic, minimally-bracketed crash repro** (N=340), not just an intermittent silent corruption. The four fixed-512 sites plus the unguarded `while v < reg_count` OOB reads over the `[512]` arrays are the concrete targets. Fix ask unchanged and strengthened: (1) bounds-guard / loud-fail every `[512]` access and the `while v < reg_count` loops (turns crash *and* silent paths into a clear diagnostic), then (2) raise/eliminate the 512 ceiling (dynamic sizing). Acceptance: the N=340 chain compiles+runs correctly under the default engine, and bignat/bigrat selftests scale to ~10x op-count without oracle babysitting.

Repro generator (`gen3.py`): emits `fn main` with N chained `let vK=v(K-1)+K` returning `(vN%250)+1`; `./bin/souc compile chain.sio -o out && ./out; echo $?` → 139 for N≥340, correct under `SOUNIO_SOUC_ENGINE=lean_single`.

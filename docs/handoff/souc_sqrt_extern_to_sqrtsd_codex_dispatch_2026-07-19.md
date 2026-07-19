<!-- docs:meta
topic_id: repo.docs.handoff.souc-sqrt-extern-to-sqrtsd-codex-dispatch-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.souc-sqrt-extern-to-sqrtsd-codex-dispatch-2026-07-19
-->

# Dispatch to CODEX-2 — `sqrt` is an extern libc call (~244 ns/element), not the hardware `sqrtsd` instruction

**Date:** 2026-07-19
**Owner:** CODEX-2 (compiler back-end / codegen for math builtins; `self-hosted/`)
**Author:** data-science lane (surfaced building `data::bigframe_ops::bf_expanding_std`)
**Status:** confirmed perf defect, minimal repro + microbenchmark included — codegen, low blast radius

---

## TL;DR

Sounio's `sqrt(x: f64) -> f64` is lowered as an **extern libc/libm call** (through the PLT, with
full call-site register save/restore), costing **~244 ns per element**. The correct lowering is a
single hardware instruction — `sqrtsd` on x86-64, `fsqrt` on arm64 — which is ~**1–4 ns** and
already correctly-rounded IEEE-754 (bit-identical to libm `sqrt`). This one call sits on the hot path
of *every* statistical / geometric verb, so fixing it is a broad, cheap win with zero accuracy change.

## Evidence (reproducible now)

**1. Microbenchmark — 30,000,000 `sqrt` calls in a tight loop:**

| variant | time | per call |
|---|---|---|
| `sqrt(x)` (current extern) | **7336 ms** | **~244 ns** |
| inline Newton–Raphson (bit-hack seed + 5 iters, no call) | 1579 ms | ~52 ns |

The inline version does *more* floating-point work (5 iterations of `0.5*(y + x/y)`) yet is **~5×
faster** — because it has no call. That isolates the cost as **call-site overhead**, not the math.
(The hardware `sqrtsd` would beat both by another ~10×.)

**2. End-to-end — `data::bigframe_ops` expanding stats over 1,000,000 rows:**

| op | Sounio | pandas | ratio |
|---|---|---|---|
| `bf_expanding_var` (Welford, no sqrt) | 14.9 ms | 14.9 ms | **1.00× (parity)** |
| `bf_expanding_std` (**same Welford + one `sqrt` per row**) | **256 ms** | 19.7 ms | **13.0×** |

`var` and `std` are the identical single-pass Welford recurrence; the *only* difference is the
per-element `sqrt`. It alone turns a parity result into a 13× loss.

## Repro programs (either engine; lean_single shown)

Extern (hot):
```
fn main() -> i32 with IO, Mut, Div, Panic, Alloc {
    var acc = 0.0; var i: i64 = 0
    while i < 30000000 { acc = acc + sqrt((i%1000+1) as f64); i = i + 1 }
    print(acc); print("\n"); return 0
}
```
Inline Newton (for the ~5× comparison; needs an 8-byte scratch cell for the bit reinterpret):
```
fn nsqrt(x: f64, scratch: *mut f64) -> f64 with Mut, Div, Panic {
    if x <= 0.0 { return 0.0 }
    let b = f64_to_bits(x); let s = (b >> 1) + 2303591209400008704
    write_i64(scratch as *mut i64, 0, s); var y = read_f64(scratch, 0)
    y = 0.5*(y + x/y); y = 0.5*(y + x/y); y = 0.5*(y + x/y); y = 0.5*(y + x/y); y = 0.5*(y + x/y)
    y
}
```
Build: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile f.sio -o out && ./out`, then `time ./out`.

## Root-cause hypothesis

`sqrt` resolves to a global builtin that the back-end emits as an **extern function call** (libm
`sqrt`) rather than an intrinsic. The `x86-64` codegen already emits inline instruction sequences for
other primitives (e.g. `emit_memcpy_a64` at `self-hosted/compiler/lean_single.sio:2104`), so emitting
`sqrtsd xmm, xmm` for `sqrt` is the same class of change — recognise the `sqrt` builtin at the call
site and emit the instruction on the f64 in the xmm register instead of setting up a call.

## The ask

Lower the `f64` `sqrt` builtin to the hardware square-root instruction:
- **x86-64:** `sqrtsd %xmmSrc, %xmmDst` (or `vsqrtsd`).
- **arm64:** `fsqrt d_dst, d_src`.

Keep the extern fallback only if the argument type isn't a register f64. `sqrtsd` is IEEE-754
correctly-rounded, so results stay **bit-identical** to today's libm path (no accuracy regression).

If `fabs`, `floor`, `ceil`, `trunc`, `rint`/`round`, and `min`/`max` share the same extern-lowering
path, they lower to single instructions too (`andpd`/`roundsd`/`minsd`/`maxsd`) — same fix, same file.

## Acceptance

- The 30M-call `sqrt` loop drops from ~7.3 s to well under ~1 s (ideally ~0.1 s).
- `bf_expanding_std` at 1M rows lands within ~1.3× of pandas (from 13×), matching `bf_expanding_var`.
- Results bit-identical to the current libm path on a fuzz set of positive doubles + edge cases
  (0.0, +inf, subnormals; `sqrt(-x)` stays NaN).

## Scope / impact

- **Low blast radius:** codegen for one (or a few) math builtins; no front-end or type changes.
- **Broad payoff:** every sqrt-bound verb flips from loss to competitive — expanding/rolling `std`,
  RMS, L2 norm, Euclidean distance, correlation/covariance normalisation, z-score / Grubbs' outlier
  tests, and the GUM uncertainty combine (`u_c = sqrt(Σ u_i²)`), which is core to the measurement-data
  thesis.
- Complements the other two open perf dispatches — `mem_copy` builtin
  (`docs/handoff/mem_copy_builtin_codex_dispatch_2026-07-19.md`, which `bf_shift` is bound by) and C3
  SIMD auto-vectorisation (`docs/handoff/c3_simd_autovectorization_codex_dispatch_2026-07-19.md`, which
  `bf_diff` is bound by). Together they close the three known bigframe losses.

## Pointers

- Repro + microbenchmark: this file.
- Verb that surfaced it: `stdlib/data/bigframe_ops.sio::bf_expanding_std` (vs `bf_expanding_var`).
- Benchmark table: `scripts/bench/RESULTS.md` (expanding rows + the "expanding" honest-read bullet).
- Existing inline-instruction precedent: `self-hosted/compiler/lean_single.sio:2104` (`emit_memcpy_a64`).

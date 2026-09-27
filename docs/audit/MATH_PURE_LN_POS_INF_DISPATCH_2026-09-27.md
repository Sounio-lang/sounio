<!-- docs:meta
topic_id: repo.docs.audit.math-pure-ln-pos-inf-dispatch-2026-09-27
authority: repo_only
audience: users
last_validated: 2026-09-27
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.math-pure-ln-pos-inf-dispatch-2026-09-27
-->

# `math::pure` `ln(+inf)` never terminates: dispatch for a one-line guard

**Date:** 2026-09-27
**Base:** `main` at `9c5ffa2360`.
**Status:** applied on `claude/sharp-carson-ujgib1`. The change is one early return in `stdlib/math/pure.sio` `ln`, plus a
run-pass regression test. Nothing in `self-hosted/` is touched.
**Origin:** found while reviewing PR Sounio-lang/sounio#2722. That PR stops `mer_pow` in
`stdlib/darwin_pbpk/release/matrix_er.sio` from passing +inf to `ln` (`if t - t != 0.0 { ... }`). The defect itself is in
the canonical `ln`, so every other caller can still reach it.

**Compilers used:**

| engine | binary | provenance |
|---|---|---|
| Madaros | `artifacts/self-hosted/madaros`, md5 `f427f163` | `make build-madaros` (i.e. `scripts/ci/build_modular_madaros.sh`) of `9c5ffa2360`, built in this session; not the committed ELF |
| lean_single | `bin/souc-lean-single-x86_64`, md5 `0cb08380` | committed bootstrap seed (`SOUNIO_SOUC_ENGINE=lean_single`) |

Every Sounio run went through `bin/souc`, the wrapper with the 512 MiB stack, with `SOUNIO_MADAROS_BIN` and
`SOUNIO_STDLIB_PATH` pinned to this checkout.

> **Measurement hazard (lean_single).** When run from the repository root, lean_single resolved `stdlib/` relative to
> the working directory and ignored a `SOUNIO_STDLIB_PATH` that pointed at a pre-fix copy of `stdlib/`. The first
> "pre-fix" lean_single run therefore silently used the fixed `pure.sio` and passed. The lean_single pre-fix column
> below was re-measured with `stdlib/math/pure.sio` itself temporarily restored to `HEAD`. Madaros honours
> `SOUNIO_STDLIB_PATH`.

## Defect

```sounio
pub fn ln(x: f64) -> f64 with Mut, Div, Panic {
    if x <= 0.0 { return 0.0 - 1.0e30 }
    if x == 1.0 { return 0.0 }
    var m = x
    var e_val = 0
    while m >= 2.0 {          // +inf >= 2.0, and +inf / 2.0 == +inf: no progress, ever
        m = m / 2.0
        e_val = e_val + 1
    }
    ...
```

For finite x ≥ 2 the loop ends within 1024 iterations, because every halving is exact and strictly decreasing. For
x = +∞, ∞/2 = ∞ (IEEE 754-2019 §6.1), so `m` never changes and the loop never ends. `log10` and `log2` are `ln(x)/const`,
and `pow(x, n)` computes `exp(n · ln x)`, so all of them inherit the hang.

## Measurement: non-finite inputs, before the fix

Each case is a separate program (`use math::pure::{f}`, one call, print `f64_to_bits` of the result), run under a 25 s
`timeout`. rc = 124 means the timeout fired. +∞ is `1.0e308 * 10.0`, −∞ is `0.0 - 1.0e308 * 10.0`, and NaN is
`+∞ − +∞` (bits `0xFFF8000000000000`).

| call | +∞ | −∞ | NaN |
|---|---|---|---|
| `ln(x)` | **hang (rc 124)** | −1e30 (sentinel) | NaN |
| `log10(x)` | **hang** | −4.342945e29 | NaN |
| `log2(x)` | **hang** | −1.442695e30 | NaN |
| `sqrt(x)` | NaN | 0.0 | NaN |
| `exp(x)` | 1e200 (clamp) | 0.0 | **hang** |
| `pow(x, 2.5)` | **hang** (via `ln`) | 0.0 | **hang** (via `exp`) |
| `pow(2.0, x)` | 1e200 | 0.0 | **hang** (via `exp`) |
| `pow(1.0, x)` | **hang** (via `exp`) | **hang** (via `exp`) | **hang** (via `exp`) |

Both engines give the same classification. Finite results agree bit for bit, except where a decimal literal is
involved (see Asides).

After the fix, the four cases that hung inside `ln` return:

| call | after fix (both engines) |
|---|---|
| `ln(+∞)`, `log10(+∞)`, `log2(+∞)` | +∞, bits `0x7FF0000000000000` |
| `pow(+∞, 2.5)` | 1e200, which is `exp`'s documented clamp for x > 500 |

Every other cell of the table is unchanged: the same bits, or still a hang routed through `exp(NaN)` (see Out of scope).

## Fix

```diff
 pub fn ln(x: f64) -> f64 with Mut, Div, Panic {
     if x <= 0.0 { return 0.0 - 1.0e30 }
     if x == 1.0 { return 0.0 }
+    // +inf is the one input the halving loop below cannot reduce: inf / 2 == inf,
+    // so it never exits. For every finite x >= 2 the halving is exact and x / 2 < x.
+    // ln(+inf) = +inf, as C11 Annex F.10.3.7. The test uses only ordered compares,
+    // so NaN fails it and still propagates through the series as before.
+    if x / 2.0 == x { return x }
```

### Return value: +∞, not a sentinel

- +∞ is the mathematically and IEEE-correct value: ln(+∞) = +∞ (C11 Annex F.10.3.7). `log10` and `log2` then give
  +∞ with no further change.
- The −1e30 sentinel for x ≤ 0 marks a **domain error**, where no real value exists. At +∞ the limit exists, and a
  finite stand-in such as +1e30 would be a fabricated number. It would turn `ln(+∞) − ln(1e300)` into a finite,
  meaningless difference instead of +∞.
- Downstream, `pow(+∞, n)` = `exp(n · +∞)` resolves to `exp`'s existing clamps: 1e200 for n > 0 and 0 for n < 0.
  `pow(+∞, 2.0)` takes the `x * x` shortcut and gives +∞, as before. None of these hang.

### Why `x / 2.0 == x`, not `x - x != 0.0`

1. **It is exactly the loop's non-progress condition.** The loop fails to terminate if and only if `m / 2.0 == m`
   with `m >= 2.0`. The guard tests that condition before the loop starts.
2. **It is false for every positive finite x**, so no finite result can change:
   - x ≥ 2⁻¹⁰²¹: x/2 is a normal number, exact and strictly smaller;
   - 2⁻¹⁰⁷⁴ < x < 2⁻¹⁰²¹: write x = k·2⁻¹⁰⁷⁴ with k ≥ 2. Here x/2 may be inexact, including in the lowest normal
     binade, but the rounding error is at most half a subnormal ULP. So round(x/2) ≤ (k + 1)/2 · 2⁻¹⁰⁷⁴ < x;
   - x = 2⁻¹⁰⁷⁴: x/2 = 2⁻¹⁰⁷⁵ is a tie, and ties-to-even rounds it to 0 ≠ x.

   Zero and the negatives never reach the guard (`x <= 0.0` returns first).
3. **It needs only an ordered comparison.** For NaN, `NaN == NaN` is false under a correct backend. If a backend
   mis-lowered an unordered `==` as true, the guard would return x, which is still NaN. The `t - t != 0.0` idiom
   depends on `!=` being lowered correctly for unordered operands, which is not needed here.

## Proof that every finite result is bit-identical

The argument above makes the guard provably inert for finite x > 0. Everything after the guard is unchanged text. As
an empirical check, a probe compares, on a dense grid, the **pre-fix `ln` copied verbatim** (renamed `ln_old`, with
`ln2()` inlined to its literal `0.6931471805599453`) against the live `math::pure::{ln, log10, log2}`. It compares all
three outputs by `f64_to_bits` and keeps an XOR fingerprint of the live `ln` bits.

### Madaros: `docs/audit/repro/math_pure_ln_grid.sio`

| group | inputs | count (cumulative) |
|---|---|---:|
| G1 | every positive finite bit pattern `1 ..= 0x7FEFFFFFFFFFFFFF`, stride 2 199 023 255 531 (≈2⁴¹, odd) | 4 192 257 |
| G2 | 65 536 consecutive ULPs each side of 1.0, plus 1.0 | 4 323 330 |
| G3 | every binade boundary 2ᵉ (biased e = 1..2046) ± 8 ULPs, plus the 64 smallest subnormals | 4 358 176 |
| G4 | the top 65 536 finite values below `DBL_MAX` (nearest the guard) | 4 423 712 |
| G5 | i·10⁻³ for i = 1..2·10⁶, and the integers 1..10⁵ | 6 523 712 |
| G6 | 0, −0, −1, −1e308, −∞, two NaNs (compared by bits) | 6 523 719 |

| tree | mismatches | fingerprint of live `ln` bits |
|---|---:|---:|
| pre-fix (null control: `ln_old` vs unguarded `ln`) | 0 / 6 523 719 | −6046242755782259232 |
| fixed | **0 / 6 523 719** | **−6046242755782259232** |

The fingerprints are equal, so the live `ln` returns the same bits before and after the fix on all 6.5 M inputs, and
`log10`/`log2` match `ln_old/const` bit for bit.

### lean_single: `docs/audit/repro/math_pure_ln_grid_lean.sio`

lean_single has no `bits_to_f64` builtin (`error[E200]`), and CI's `full-test-suite` runs lean_single. A second
probe therefore builds its inputs by exact arithmetic:

- L1: 2ᵉ for e = −1022..1023, at 256 mantissa points and ±8 ULPs, plus all subnormal powers of two and 2..64 × 2⁻¹⁰⁷⁴;
- L2: 65 536 ULPs each side of 1.0;
- L3: the top 4 096 values below `DBL_MAX`;
- L4: the decimal and integer grids of G5;
- L5: the non-positive values and NaN.

| tree | mismatches | fingerprint |
|---|---:|---:|
| pre-fix (null control) | 0 / 2 791 801 | 5944270505776792656 |
| fixed | **0 / 2 791 801** | **5944270505776792656** |

### Parity copy (JavaScript)

`docs/audit/repro/math_pure_ln_grid.mjs` runs the same comparison on the PR #2722 `pureLn` port, unguarded against
guarded, over 6 423 653 inputs (G1–G5 and the non-positive/NaN set). Result: `JS_GRID n=6423653 bad=0`.

## Regression test — `tests/run-pass/math_pure_ln_pos_inf_terminates.sio`

The test imports `math::pure::{ln, log10, log2}` and checks by bit pattern that:

- `ln`, `log10` and `log2` of +∞ are exactly +∞;
- `ln(DBL_MAX)` keeps its pre-fix bits (`4649454530587146735`, which is also the correctly rounded
  709.782712893384). `DBL_MAX` is built as (2 − 2⁻⁵²)·2¹⁰²³ by exact doubling and halving, so neither engine's
  decimal-literal rounding is involved;
- the −1e30 sentinel for −∞ and 0 is unchanged;
- NaN still propagates;
- `ln(2)` is unchanged.

It carries `//@ timeout: 30`, so without the fix the suite reports a timeout instead of stalling.

| tree | Madaros | lean_single |
|---|---|---|
| pre-fix `pure.sio` | prints `PURE_LN_POS_INF_START`, then **timeout, rc 124** | same, **rc 124** |
| fixed | `PURE_LN_POS_INF_OK`, rc 0 | `PURE_LN_POS_INF_OK`, rc 0 |

`stdlib/math/pure.sio` carries its own `//@ run-pass` self-test (44 checks). Its result is unchanged by the fix:
Madaros gives 42/44 both before and after, failing checks 16 and 17 (`atan(1)` and `atan2(1, 1)` against π/4 at 1e−10).
lean_single gives 43/44 both before and after, failing check 16. Both failures are pre-existing, sit in `atan`, and are unrelated.

## Textual copies that must stay in parity

`git grep -l "while m >= 2.0" | wc -l` gives **162** files carrying the same halving-loop idiom. Almost all are
independent local helpers (stdlib submodules, examples, self-contained tests) that implement their own `ln`. They do
not claim to mirror `pure.sio`, so this dispatch does not touch them. A census of which of them can receive +∞ is a
separate task.

Two copies **declare** themselves verbatim ports of `pure.sio` `ln`. Both are introduced by PR #2722 and are not on
`main` at `9c5ffa2360`:

| copy | declared as | reachable with +∞? |
|---|---|---|
| `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio` `mer_ln` | "verbatim stdlib/math/pure.sio ln()/exp()" | No. Its only caller `mer_pow` returns early for NaN and for `t - t != 0.0` (+∞) before calling `mer_ln`. |
| `website/src/lib/pbpk28_core.mjs` `pureLn` | "line-for-line ports", "bit-identical to Madaros on the K-P grid" | No. Its only caller `merPow` has the same guard. |

**Decision: mirror the guard in both.** Neither copy can hang today, so this is not a correctness fix. Both files
claim to be verbatim, though, and once `pure.sio` carries the guard that claim is false without it. The guard is also
inert for finite input, so mirroring it cannot move any parity output. `docs/audit/repro/math_pure_ln_parity_2722.patch`
holds the two one-line additions against the PR #2722 head `0466ac7d`. Whichever of the two PRs lands second
should carry them.

This was verified in a scratch worktree of `0466ac7d` with this fix and the patch applied, under Madaros `f427f163`:

- `dissertation_pbpk28_parity_ref_venlafaxine.sio`: rc 0, and all 3 209 stdout lines identical to the unpatched PR head.
- `darwin_venlafaxine_xr_matrix_small_t.sio`: rc 0, and all 54 lines identical.

Two further copies in PR #2722 are frozen evidence and should **not** be edited:

- `docs/audit/repro/matrix_er_kp_bits_probe.mjs` line 17;
- the `+` lines of `docs/audit/repro/matrix_er_pure_math.patch`.

Both record what was measured on 2026-09-26.

## Out of scope, measured and left open

1. **`exp(NaN)` does not terminate in practice.** `k_f = NaN / ln2`, and `k_f >= 0.0` is false, so
   `k = (k_f as i32) - 1`. On both engines that gives **k = 9223372036854775807**, measured by printing `k`: the cast
   yields `INT64_MIN` and the subtraction wraps. The `2^k` loop then runs about 9.2·10¹⁸ iterations. Four cells of
   the table route through it: `exp(NaN)`, `pow(NaN, n)`, `pow(x, NaN)`, and `pow(1, ±∞)`, where
   ±∞·ln(1) = ±∞·0 = NaN. PR #2722 guards its own caller. The canonical fix would be a NaN early return in `exp`
   (`if x != x { return x }`, or an ordered-only form). That is a separate dispatch.
2. **`sqrt(+∞)` returns NaN.** Newton's first step computes 0.5·(∞ + ∞/∞) = NaN. IEEE gives +∞. It does not hang,
   so it is not part of this fix.

## Asides recorded for the compiler owners (no action here)

- **Decimal-literal rounding differs by engine.** Madaros parses `1.0e200` to `7598952565167317594` and `1.0e30` to
  `5055640609639927018`; both are correctly rounded. lean_single gives `…596` and `…019`, 2 ULP and 1 ULP high. This
  is why the `exp(+∞)` clamp and the `−1e30` sentinel differ in their last bits between engines.
- **An `i32` variable holds an out-of-range `i64`.** In `exp`, `var k: i32` ends up holding 9223372036854775807. The
  cast and the subtraction are not truncated to 32 bits.
- **Madaros `println` of a large f64** prints `9223372036854775808.000000` for 1e200 and −1e30, apparently saturating
  through `i64`. The bits are correct. lean_single prints `1.000000e200`.

## Offload

Mandatory math review (CLAUDE.md §10, M1): `bin/llm-offload -t math-review -i` on this dispatch.

**Result: not performed, blocked.** The session ran in a cloud container that holds no provider credentials. All
four legs of the default fan-out printed SKIPPED: xai and kimi need `LLMGATEWAY_API_KEY` (or `XAI_API_KEY`), zai needs
`ZAI_API_KEY`, and local needs `LOCAL_LLM_URL`. No leg is counted, and the M1 checkpoint is **open**. It must be rerun
from a workspace with the keys before this merges. The attempt is logged in `.claude/llm_offload_log.md`.

An author self-check in place of the offload, which does not count toward M1, corrected one claim before logging. The
first draft said x/2 is exact for every normal x. That is false in the lowest normal binade, [2⁻¹⁰²², 2⁻¹⁰²¹),
where x/2 is subnormal. The inertness argument is unaffected, and the rounding bound above now covers that binade.

## Reproduce

```bash
make build-madaros
export SOUNIO_MADAROS_BIN=$PWD/artifacts/self-hosted/madaros SOUNIO_STDLIB_PATH=$PWD/stdlib
./bin/souc run tests/run-pass/math_pure_ln_pos_inf_terminates.sio        # PURE_LN_POS_INF_OK
./bin/souc run docs/audit/repro/math_pure_ln_grid.sio                    # GRID_BIT_IDENTICAL (~2.7 min)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run docs/audit/repro/math_pure_ln_grid_lean.sio   # (~4.5 min)
node docs/audit/repro/math_pure_ln_grid.mjs                              # JS_GRID ... bad=0
```

For the pre-fix columns, restore `stdlib/math/pure.sio` from `9c5ffa2360` in place. Do not rely on
`SOUNIO_STDLIB_PATH` alone: lean_single ignores it (see the hazard note above).

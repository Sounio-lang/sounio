<!-- docs:meta
topic_id: repo.docs.audit.matrix-er-transcendental-accuracy-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.matrix-er-transcendental-accuracy-dispatch-2026-09-26
-->

# `matrix_er` Korsmeyer-Peppas `t^n` is wrong away from t = 1: dispatch for the `ln`/`exp` helpers

**Date:** 2026-09-26
**Base:** measured on `main` at `46e9b48b6`; committed on `f141ad5d9`. No file under `stdlib/`, `tests/`, `website/` or `self-hosted/` differs between the two, and the patch applies to both.
**Status:** applied (operator approval, 2026-09-26) in PR #2722. `docs/audit/repro/matrix_er_pure_math.patch` holds
the final diff of the three files against `f141ad5d9`, including the review fixes (the +inf guard and the `pureSqrt`
comment). No constant moves, and nothing in `self-hosted/` is touched.
**Compiler for every Sounio number below:** Madaros built from source, md5 `5764851f`. It came from a
`make build-madaros` of `98315edcdb`, and `git diff --stat 98315edcdb 46e9b48b6 -- self-hosted` is empty.
Seven independent builds on the pod carry the same md5. Every run used `bin/souc` (the 512 MiB-stack wrapper)
with `SOUNIO_MADAROS_BIN` and `SOUNIO_STDLIB_PATH` pinned to the worktree.

## Defect

`stdlib/darwin_pbpk/release/matrix_er.sio` computes the Korsmeyer-Peppas fraction
F(t) = min(1, k·t^n), with k = 0.199 and n = 0.65 for venlafaxine XR (Gohel 2008). It computes t^n as
`mer_exp(n · ln t)` using two local helpers:

| helper | algorithm | failure |
|---|---|---|
| `mer_ln_unit(x)` | 20-term artanh series on y = (x−1)/(x+1), with no range reduction | As x → 0+, y → −1 and the sum saturates at −2·Σ_{j<20} 1/(2j+1) = **−4.959**, whereas ln(1e−6) = −13.8. For t > 2 the code uses ln2 + ln_unit(t/2), so for t > 4 the argument leaves (0, 2] and convergence slows (−5.2e−5 relative at t = 24, −2.2e−3 at t = 48). |
| `mer_exp(x)` | (1 + x/1024)^1024 | Relative error ≈ −x²/2048 for either sign of x: −1.27e−3 at x = 0.65·ln 12. |
| `sqrt_f64` (n = 0.5 branch, from `tsit5_pbpk14`) | 10 Newton steps from y₀ = x | +30% at x = 1e−6; 9.8e−4 at x = 1e−15 (true value 3.2e−8). No in-tree caller uses n = 0.5. |

The same three helpers are copied verbatim into `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio`
(`mer_ln_unit`/`mer_exp`/`mer_pow`) and `website/src/lib/pbpk28_core.mjs`
(`merLnUnit`/`merExp`/`merPow`; the n = 0.5 branch uses `Math.sqrt`).

### Accuracy over the t range the scenarios use

Measured against IEEE-754 `Math.pow` (V8), using the Node port (bit-identical to Madaros, see Parity):

| t (h) | F current | F exact | relative error |
|---:|---:|---:|---:|
| 3.5e−15 | 7.8825e−3 | 7.9891e−11 | +9.9e7 |
| 1e−6 | 7.8829e−3 | 2.5053e−5 | +3.1e2 |
| 1e−3 | 8.2961e−3 | 2.2328e−3 | +2.7 |
| 0.01 | 1.2158e−2 | 9.9736e−3 | +0.219 |
| 0.05 | 2.8408e−2 | 2.8391e−2 | +5.9e−4 |
| 0.0625 | 3.2795e−2 | 3.2823e−2 | −8.5e−4 |
| 0.1 | 4.4503e−2 | 4.4551e−2 | −1.07e−3 |
| 0.5 | 0.12681 | 0.12682 | −9.9e−5 |
| 1 | 0.199 | 0.199 | 0 |
| 4 | 0.48980 | 0.49000 | −4.0e−4 |
| 8 | 0.76820 | 0.76889 | −8.9e−4 |
| 11.97 | 0.99784 | 0.99911 | −1.27e−3 |
| 12 | 0.99947 | 1 (capped) | −5.3e−4 |

- The cap crossing k·t^n = 1 lies at t* = k^(−1/n) = **11.98637 h**. The current helpers put it at 12.00987 h, 0.0235 h late.
- The error of t^(n−1), used by `matrix_release_rate`, follows the same pattern.

### Regression test — `tests/run-pass/darwin_venlafaxine_xr_matrix_small_t.sio`

This began as a 24-line repro in `docs/audit/repro/` and was moved into the run-pass suite after PR review. It calls
the live `matrix_fraction` at t = 3.5e−15, 1e−6 and 0.01 and compares it with the exact 0.199·t^0.65. It also checks
F(12 h) = 1, since the cap lies at 11.98637 h. It exits 1 while the defect is present.

| tree | output (relative errors, F(12 h)) | rc |
|---|---|---:|
| `main` (old helpers), Madaros | `98665798.635112` / `313.654193` / `0.218976` / `0.999466` / `REPRO_KP_SMALL_T_FLOOR` | 1 |
| fix, Madaros | `0.000000` ×3 / `1.000000` / `KP_SMALL_T_OK` | 0 |
| fix, lean_single | `4.85e−16` / `4.06e−16` / `0.000000` / `1.000000` / `KP_SMALL_T_OK` | 0 |

## How a tiny clock arises and what it costs

`vfx_released_qtau` on `claude/elegant-borg-a14bdf` (bba219013 onward) sums, for each step n,
D·[F((t − kτ) + dt) − F(t − kτ)] over the doses k, with t = n·dt. For a dt that is not exact in binary, the
**upper** clock of the step just before dose k lands slightly above zero:

| dt | example | upper clock (n·dt − kτ) + dt |
|---:|---|---:|
| 0.4 | n = 59, k = 1 | 1.44e−15 |
| 0.2 | n = 119, k = 1 | 7.22e−16 |
| 0.1 | n = 239, k = 1 | 2.14e−15 |
| 0.08 | n = 299, k = 1 | 1.71e−15 |
| 0.05 | n = 479, k = 1 | 2.84e−15 |

The next step's lower clock is exactly 0, or slightly negative, so the sum does not telescope. Each dose therefore injects an extra D·F(u):

- **0.591 mg** per dose (0.79% of 75 mg) with the current helpers, because of the floor.
- **≤ 1.3e−8 mg** per dose (u ≤ 1.14e−14) with an accurate t^n.

At dt = 0.5 and 0.25 no positive tiny clock occurs. The accurate t^n shrinks the leak by a factor of about 4.5e7 (7.7 orders of
magnitude) but does not remove it. The structural fix is integer step offsets, (n − k·steps_per_tau)·dt, which PR
Sounio-lang/sounio#2705 applied to the parity refs; that PR is closed unmerged, and nifty-goodall `dacd8c2b4` still carries it.
Both fixes are worth landing; neither replaces the other.

## Proposed fix

Use the canonical range-reduced `stdlib/math/pure.sio` functions instead of a fourth local copy. That file
already says "All modules that need math should use these", and `darwin_pbpk/aggregate_confidence.sio` already imports it.

```sounio
use darwin_pbpk::tsit5_pbpk14::{abs_f64}
use math::pure::{exp, ln, sqrt}

fn mer_pow(t: f64, n: f64) -> f64 with Mut, Div, Panic {
    if t <= 0.0 { return 0.0 }
    if t != t { return t }                                    // NaN t or n propagates
    if n != n { return n }
    if t - t != 0.0 {                                         // +inf^n limit (see below)
        if n > 0.0 { return t }
        if n < 0.0 { return 0.0 }
        return 1.0
    }
    if abs_f64(n - 1.0) < 1.0e-12 { return t }
    if abs_f64(n - 0.5) < 1.0e-12 { return sqrt(t) }
    let x = n * ln(t)                                         // exp only sees a finite argument
    if x != x { return 1.0 }                                  // t == 1, n = ±inf (IEEE pow)
    if x - x != 0.0 { if x > 0.0 { return x } return 0.0 }    // infinite n
    return exp(x)
}
```

`mer_ln2`, `mer_ln_unit` and `mer_exp` are deleted. The parity ref copies `pure.sio` `ln`/`exp` verbatim as
`mer_ln`/`mer_exp`; the ref stays self-contained. `pbpk28_core.mjs` ports `pure.sio` `ln`/`exp`/`sqrt` line for line
(`as i32` → `Math.trunc`). The patch touches all three files and applies cleanly to `main` 46e9b48b6 and to
`claude/elegant-borg-a14bdf`. On nifty-goodall `dacd8c2b4` the two helper blocks are textually identical, and only the
`pbpk28_core.mjs` header-comment hunk needs a rebase.

**Algorithms (pure.sio).**
- `ln`: exact halving or doubling to m ∈ [0.5, 2), then a 29-term artanh series with |t| ≤ 1/3, plus e·ln2.
- `exp`: k = floor(x/ln2), r ∈ [0, ln2), a 24-term Taylor series, then exact ×2 or ÷2 |k| times.

Truncation is below 1e−29 in both, so the error is rounding-dominated. Measured against V8 `Math`:

| quantity | range | worst error |
|---|---|---:|
| ln | t ∈ [1e−15, 100] | 3.7 ulp |
| exp | x ∈ [−40, 5] | 22.5 ulp (at x = −35.2, where the single-constant ln2 reduction costs \|k\|·ulp) |
| t^0.65 | t ∈ [1e−15, 100] | 28 ulp |
| t^(−0.35) | t ∈ (0, 24] | 15 ulp |
| sqrt | [1e−15, 31.6] | 1 ulp |

On the probe grid, max |F_new/F_exact − 1| = **4.1e−15**, down from 1e−3 in the normal range and 1e8 at tiny t.
With the fix the cap crossing is 11.986367107929336, against 11.986367107929333 exact.

**+inf guard (added in review).** `pure.sio` `ln(+inf)` never terminates: its halving loop keeps m = +inf ≥ 2.
The old series returned NaN promptly instead. `mer_pow` therefore returns NaN for a NaN t first (`t ≠ t`), so an invalid clock never collapses to a
plausible zero rate through the n − 1 < 0 exponent. For t = +inf it then returns the limit before either branch runs:
+inf for n > 0, which caps F at 1; 0 for n < 0; and 1 for n = 0, matching exp(0·ln t) on the finite path.
A NaN exponent is returned before `exp` runs. On Madaros, `pure.sio` `exp(NaN)` casts NaN to the indefinite
integer, so its 2^k scaling loop practically never ends. For the same reason `mer_pow` checks x = n·ln t before
calling `exp`. x is NaN only for t = 1 with n = ±inf, where it returns 1 as IEEE `pow` does. x = ±inf means an
infinite n, and it returns +inf or 0. `exp` therefore only ever receives a finite argument. Every finite result is bit-identical: the parity-ref and `matrix_er` outputs are byte-identical
with and without the guard. The regression test asserts that F(10·1e308) = 1; that `matrix_release_rate` at a NaN clock is NaN; that at t = +inf, n = 0 gives F = k; that a NaN n gives NaN at both t = +inf and t = 2 h; and that an infinite n gives F = k at t = 1, F = 1 at t = 2 and F = 0 at t = 0.5. With the guard removed, that test hangs
until a 120 s timeout (rc=124). The hang in `pure.sio` itself is out of scope here and is flagged separately.

A Cody-Waite split of ln2 would bring exp down to about 1 ulp. That is a `pure.sio` change affecting every
consumer, so it is out of scope here.

### Parity (Sounio ↔ Node)

`docs/audit/repro/matrix_er_kp_bits_probe.{sio,mjs}` emits `f64_to_bits` of t, F_current and F_proposed on
523 points. The points cover 8 tiny clocks, the 25 q24h upper clocks above for dt ∈ {0.4, 0.2, 0.1, 0.08, 0.05},
k = 1..5, and t = 0.05·i for i = 1..490. Madaros against Node gave
`PARITY rows sio=523 node=523 t_bits_diff=0 cur_bits_diff=0 new_bits_diff=0`: bit-identical before and after.

To reproduce the parity check:

```bash
bin/souc run docs/audit/repro/matrix_er_kp_bits_probe.sio > kp.txt
node docs/audit/repro/matrix_er_kp_bits_probe.mjs kp.txt   # exits 1 (PARITY_FAIL) on any differing bit or row count
```

For the parity ref under the patch, emulating gate cases 10–13 (per-organ C_avg RMSE, matrix, ratio; threshold 1%):

| | worst organ C_avg RMSE | matrix RMSE | ratio RMSE |
|---|---:|---:|---:|
| before | 7.05e−5 % | 0 | 0 |
| after | 7.02e−5 % | 0 | 0 |

## Which outputs move, and by how much

All figures are Madaros `5764851f`, before against after the patch.

**1. q24h steady-state C_avg ODV/parent ratio** (`vfx_ss_regimen_interval`, elegant-borg HEAD `14082ca7e`).
This is the dissertation-facing PGx readout.

| dt | phenotype | closed form | before: rel. error | after: rel. error | periodicity P_parent (mg), before → after |
|---:|---|---:|---:|---:|---|
| 0.5 | PM / IM / NM / UM | 0.346539 / 1.126788 / 1.912260 / 2.501897 | −1.7e−11 … −3.2e−11 | same to 1e−13 | 3.78e−6 → 3.79e−6 |
| 0.4 | PM … UM | | +2.1e−4 … **+3.3e−4** | ≤ 8.9e−13 | 3.1e−2 … 4.2e−2 → 1.67e−6 |
| 0.25 | PM … UM | | −2.5e−13 … −4.6e−13 | same | 1.9e−7 → 1.9e−7 |
| 0.2 | PM … UM | | +8.9e−5 … +1.5e−4 | ≤ 3.1e−12 | 2.0e−2 … 2.5e−2 → 9.2e−8 |
| 0.1 | PM … UM | | +3.4e−5 … +6.0e−5 | ≤ 8.2e−13 | 1.2e−2 … 1.4e−2 → 4.2e−9 |
| 0.08 | PM … UM | | +2.5e−5 … +4.4e−5 | ≤ 1.2e−12 | 1.0e−2 … 1.1e−2 → 9e−10 |
| 0.05 | PM … UM | | +1.2e−5 … +2.3e−5 | ≤ 6.3e−13 | 6.8e−3 … 7.4e−3 → 2e−10 |

- The closed form lies inside every run's certified [lo, hi] both before and after.
- **The reported dt = 0.5 values 0.346539 / 1.126788 / 1.912260 / 2.501897 do not move** at any printed digit, because dt = 0.5 has no tiny clock.
- After the fix the readout is dt-independent to 3e−12, which is what bba219013's S = (−A)⁻¹(BU + J − dx) argument predicts for a periodic state. That 3e−12 is accumulated residual (about 1e4 ulp), not one rounding. It is also a statement about the ratio only: the periodicity residual P still reaches 1.7e−6 mg at dt = 0.4.
- Every dt-refinement or convergence table built on this readout at inexact dt changes, by up to 3.3e−4 relative.

**2. Parity reference** (`dissertation_pbpk28_parity_ref_venlafaxine.sio`, dt = 0.5, one 75 mg dose; Node moves identically).

| t (h) | released, before → after (mg) | NM mass ratio, relative change | blood parent C_avg, relative change | blood ODV C_avg, relative change |
|---:|---|---:|---:|---:|
| 1 | 14.925000 → 14.925000 | +1.3e−5 | −3.3e−5 | +1.5e−5 |
| 2 | 23.417515 → 23.419835 | −9.0e−6 | +1.0e−4 | +8.3e−5 |
| 4 | 36.735102 → 36.749661 | −7.1e−5 | +4.3e−4 | +3.2e−4 |
| 8 | 57.615026 → 57.666399 | −2.0e−4 | +1.15e−3 | +8.7e−4 |
| 12 | 74.959983 → **75.000000** | +5.4e−5 | +7.5e−5 | +9.4e−4 |
| 18 | 75 → 75 | +1.0e−3 | −2.2e−3 | −9.8e−4 |
| 36 | 75 → 75 | +6.8e−4 | −4.8e−3 | −2.1e−3 |
| 96 | 75 → 75 | +6.6e−4 | (6-dp print floor) | |

The largest change over all organ C_avg > 1e−3 mg/L is **2.8e−3 relative** (parent, organ 7, 36 h).
Gate cases 10–13 remain Node ↔ Sounio parity and pass either way.

**3. `stdlib/darwin_pbpk/release/matrix_er.sio` `main`.**
- TEST 2 F(12 h) goes from 0.999466 to **1.000000**.
- F(1 h) = 0.199000 is unchanged, since ln 1 = 0 in both.
- Mass balance at 12 h: 0.000000 (Madaros) → 0.000000, unchanged.
- All 5 tests PASS both ways.

**4. `stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio` `main` on `main`.** This is the 120 h single-dose total-mass ratio,
which bba219013 retires as non-convergent. PM 15.913454 → 15.926223, NM 219.605670 → 219.781877,
UM 655.634319 → 656.160386 (+8e−4 relative). It is not a dissertation claim, and 3/3 PASS either way.

**5. Unaffected, or not measurable here.**
- `darwin_venlafaxine_xr_matrix_smoke` passes both ways.
- `darwin_venlafaxine_xr_pgx_smoke` fails to compile on `main` with **E175** both ways. The cause is the private `vfx_scenario_init`, fixed by `d89ba11c7` on other lanes; it is unrelated to this change.
- No `docs/dissertation/results/*` file quotes a venlafaxine number, so no results document goes stale on this change alone.
- The competent-mcclintock portal model (unpushed) and the nifty-goodall refs run the same helpers. They will move by the same mechanism: under 1e−3 at dt = 0.5, and up to 3e−4 on SS ratios at inexact dt. They were not re-measured here.

## Adjacent, not in scope

- The same (1 + x/1024)^1024 exp, or a short-series ln, lives in:
  - `release/biomaterial_release.sio` `rel_exp`. Its comment claims "< 0.01% for |x| < 10"; −x²/2048 is −4.9% at x = −10.
  - `validation/pbpk28_rapamycin_clinical.sio`, `validation/pbpk28_semaglutide_clinical.sio` and `validation/rapamycin_clinical.sio`, where the helpers feed GMFE and log residuals.
  - `validation/tacrolimus_oral_pbpk.sio:357`.

  None was measured here; a separate task has been proposed.
- `mer_exp_neg` (the absorption helper) is an unreduced 19-term Taylor series in the parity ref and in `pbpk28_core.mjs`.
  elegant-borg `11657641a` range-reduces it in `venlafaxine_xr.sio` only. It is bit-identical for ka·dt ≤ 0.5, i.e. dt ≤ 0.79 h, so every parity run is unaffected.

## Review

The mandatory math review ran on the canonical tooling (`chore/llm-offload-llmgateway-grok47`, `754c303cb`) against claims C1–C9.
The log entry is in `.claude/llm_offload_log.md`.

- **Grok 4.7** (the gateway leg errored, so it ran via the automatic xAI-direct fallback): 8 OK and 1 TIGHTENABLE.
  The tightenings were: state the leak ratio as 7.7 orders, not 7; and call the 3e−12 SS residual accumulated rather
  than a rounding ulp, noting that it does not extend to the periodicity metric. Both are applied above.
- **Qwen 3 235B** (OpenRouter): 9/9 OK.
- **Kimi K3** is not counted. It returned no content three times: two runs ended with finish_reason=length, the
  whole budget spent on reasoning, and one got an upstream 502.

The two independent vendors are xAI and Qwen.

## Reverting

Revert the PR's fix commits with `git revert`, or run `git apply -R docs/audit/repro/matrix_er_pure_math.patch`.
The patch is the full diff of the three source files against `f141ad5d9`, so reversing it restores them exactly.

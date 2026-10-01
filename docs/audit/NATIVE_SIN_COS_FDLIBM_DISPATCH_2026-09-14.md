<!-- docs:meta
topic_id: repo.docs.audit.native-sin-cos-fdlibm-dispatch-2026-09-14
authority: repo_only
audience: users
last_validated: 2026-09-14
validated_by: claude-code
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.native-sin-cos-fdlibm-dispatch-2026-09-14
-->

# sin and cos on all three engines are not accurate — forensic dispatch

**Date:** 2026-09-14
**sha measured:** `c6e1d7b09b` (`origin/main` tip at branch cut, `fix/native-sin-cos-accuracy`)
**Write set:** this document; `stdlib/math/pure.sio`; the `__native_sin_f64` / `__native_cos_f64`
block of `self-hosted/compiler/lean_single.sio`; `self-hosted/native/codegen_x86_linux.sio`
(`emit_builtin_sin` / `emit_builtin_cos`); new `self-hosted/native/math_fdlibm_trig.sio`
(generated); new `scripts/research/fdlibm_sin_cos_oracle.py`, `scripts/dev/gen_fdlibm_trig.py`;
new vector tests; governance registry sync.

---

## 1. What is wrong, measured

`sin` and `cos` exist three times, and each copy has its own error.

| implementation | where | reduction | kernel |
|---|---|---|---|
| Madaros builtin (ids 18/19) | `self-hosted/native/codegen_x86_linux.sio` `emit_builtin_sin/cos` | `r = x - round(x/2π)·2π` with a rounded 2π, so r ∈ [-π, π] | Taylor to x¹³ |
| lean_single builtin | `self-hosted/compiler/lean_single.sio` `append_native_prelude` | `while t > π { t = t - 2π }` | Taylor to u¹² / u¹³ on [0, π/4] |
| stdlib | `stdlib/math/pure.sio` | `while a >= 2π { a = a - 2π }` | 14-term Taylor on [-π, π]; `sin(x) = cos(x - π/2)` |

Maximum absolute error over 137 inputs (−8 … 8 in steps of 1/8, plus 100, 1000, 12345, 10⁶, 3, 22,
355 and π/2), against mpmath at 300 bits. Committed binaries at `c6e1d7b09b`:

| | sin | cos |
|---|---|---|
| Madaros builtin | 2.1e-5 | 1.0e-4 |
| lean_single builtin | 1.3e-6 | 4.9e-7 |
| `pure.sio` (\|x\| ≤ 1000 only) | 1.3e-12 | 1.9e-12 |

Receipt command: the generated program `trig_baseline.sio` (inputs and references baked in),
compiled with `SOUNIO_SOUC_ENGINE=madaros ./bin/souc build` and with the lean_single seed. The
lean_single column's references are decimal literals, which that engine does not read correctly
rounded (§3.1); the 13 ulp of §3.1 was measured on `1e-300`, and even 13 ulp of an O(1) reference
is ~3e-15, far below the error reported.

Where the lean_single error comes from: not its kernels, whose truncation on [0, π/4] is
(π/4)¹⁵/15! ≈ 2e-14. Its reduction is `while t > π { t = t - 2π }`; each subtraction rounds at the
magnitude of t, and x = 10⁶ takes ~1.6·10⁵ of them. That attribution is arithmetic, not a separate
measurement.

The code comment on the Madaros emitter already records the same failure one step earlier
(#1551: `cos(3.0)` returned −1.1375). Adding terms moved the error from 7.5e-2 to 2.1e-5; it did
not remove its cause, which is evaluating a Taylor polynomial at |r| up to π and reducing with a
rounded 2π.

## 2. Proposed fix

Replace all three with **fdlibm**, as shipped by OpenLibm (FreeBSD msun): `s_sin.c`, `s_cos.c`,
`e_rem_pio2.c`, `k_rem_pio2.c` with `prec = 1`, `k_sin.c`, `k_cos.c`. The kernels use fdlibm's own
fitted coefficients — not Taylor coefficients — and `k_cos`'s evaluation order
`w + (((1-w) - hz) + (z·r - x·y))`, which is what makes them sub-ulp.

The claim is **bit-identity with OpenLibm**, not an ulp bound — and it is a measured claim: identity
on the input sets listed in §4 (1,009,578 inputs chosen to reach every branch and the worst cases),
not a proof over all 2⁶⁴ doubles. Every floating-point expression keeps
the C operand grouping, nothing is fused, and every branch of the C is kept, including the tiny-
argument returns (`sin(-0.0)` is `-0.0` only because of that branch).

Why the kernel coefficients must not be "checked" against 1/n!: S1..S6 and C1..C6 differ from
(−1)ᵏ/(2k+1)! by 2e-15 (S1) up to 1% (S6). Measured over 200,000 points of [0, π/4] as
|computed − exact| / ulp(exact), mpmath at 200 bits, y = 0, the C evaluation order:

| kernel (six coefficients) | max error |
|---|---|
| sin, fdlibm S1..S6, `k_sin` form | 0.69 ulp |
| sin, Taylor (−1)ᵏ/(2k+1)!, same form | 184 ulp |
| cos, fdlibm C1..C6, `k_cos` form | 0.75 ulp |
| cos, Taylor (−1)ᵏ⁺¹/(2k+2)!, same form | 9.6 ulp |
| cos, fdlibm C1..C6, plain `1 − z/2 + z²·C(z)` | 1.24 ulp |

**Correction.** An earlier draft of this section gave 0.51 / 0.50 / 130 / 15 ulp. Those divided
relative error by 2⁻⁵², which is the ulp of 1, not of results in [0.5, 1); the 15-ulp row had also
been computed with a mistyped C2. Found by the M1 math review.

**Math review (M1), 2026-09-14.** `bin/llm-offload -t math-review` on this document: grok-4.6
answered (13 OK, 1 WRONG — the 130-ulp figure, corrected above — and 2 TIGHTENABLE, applied in §1);
the Z.AI leg failed on a fair-usage limit (error 1313; a retry, and another with a temporary
key, returned the same error on the coding-plan endpoint, and the general endpoint refused the
temporary key with error 1113, insufficient balance; restoring access needs a request to Z.AI) and the local leg on a connection error, so a second
provider was used instead: **MiniMax-M3** (same composed prompt, called directly because
`scripts/mcp/llm-offload.sh` pins MiniMax-M2.7; a first call hit its 8192-token limit while still
reasoning, the second finished). It found no mathematical error: 9 OK, 1 TIGHTENABLE — derive the
q0 ≥ −27 bound instead of citing it, now done in §3.4 — and 1 OVERREACH — bit-identity on a finite
input set is not identity for every double, now qualified in §2. One grok-4.6 point was only
partly accepted: it bounds q0 below by −21, which holds for e0 ≥ 3 but not for e0 ∈ [−3, 2], where
q0 = e0 − 24 reaches −27 (§3.4) — the value measured.
Row in `.claude/llm_offload_log.md`.

## 3. Constraints found while doing it

### 3.1 lean_single decimal literals are not correctly rounded

Measured on the committed seed: `1e-300` is 13 ulp off, `5.319372648326541e+255` and
`6.077100506506192e-11` are 1 ulp off each. This is the known #1626 follow-up
(`tests/run-pass/float_literal_correctly_rounded.sio`, `requires: madaros`). The Sounio
implementation therefore spells no constant as a decimal: each is an exact integer mantissa
scaled by exact powers of two, `(M as f64) / ((1 << k) as f64)`, which both engines evaluate
exactly.

### 3.2 Madaros lowers `-x` as a subtraction

`-x` for `x = +0.0` yields `+0.0` under Madaros (lean_single: `-0.0`). Negation is written as
`v * (0.0 - 1.0)` everywhere. No sin/cos result depends on it — every negated value is non-zero,
except the `y1` tail, whose zero sign cannot reach the output — but the code does not rely on that.
Filed separately.

### 3.3 `bits_to_f64` is unusable here

Undefined in lean_single; under Madaros `print` of an `f64_to_bits` result fails to lower
("unresolved scalar kind"). `INSERT_WORDS` in the large-argument path is replaced by the equivalent
exact multiplication by 2^−e0; tests construct inputs from mantissa and exponent and compare bits
without printing them.

### 3.4 `k_rem_pio2` recomputes routinely

The recompute branch is not a rarity to be dropped: it fires on 1000 of the 1020 per-exponent worst
cases and on 16,411 of 200,000 random inputs. Over 1,009,578 inputs the loop bounds measured are
passes ≤ 1, k ≤ 3, jz ≤ 7, jv ∈ [0, 41], q0 ∈ [−27, 2], nx ∈ {2, 3}; the C arrays of 20 are kept.
The q0 bounds follow from `k_rem_pio2.c`: jv = max(0, (e0 − 3)/24) and q0 = e0 − 24·(jv + 1), with
e0 = ilogb(x) − 23 ≥ −3 on this path. For e0 ∈ [−3, 2], jv = 0 and q0 = e0 − 24 ∈ [−27, −22]; for
larger e0, q0 = e0 − 24·⌊(e0 − 3)/24⌋ − 24 ∈ [−21, 2]. So q0 ≥ −27, attained at e0 = −3.

### 3.5 `math::pure`'s sin and cos were unreachable under Madaros

`fn sin` and `fn cos` in `stdlib/math/pure.sio` were private, unlike `sqrt`, `tan` and `atan`.
Madaros enforces visibility, so every cross-module call was refused (`E175 … callee math/pure::sin
… function is private in its defining module`). Both are now `pub`; a caller that compiled before
still compiles, and one that did not now does.

### 3.6 The old lean_single builtin never returns on infinities

`__native_sin_f64(+inf)` loops forever on `while t > pi { t = t - two_pi }` (`inf - 2π` is `inf`).
Observed as a hang at row 2 of the vector fixture on the committed lean_single seed. fdlibm returns
`x - x` (NaN) for infinities and NaN.

## 4. Receipts

- **Reference.** OpenLibm `5fe3997` (2026-09-02); `s_sin.c s_cos.c k_sin.c k_cos.c k_rem_pio2.c
  e_rem_pio2.c s_scalbn.c s_floor.c` built with `gcc 13.3 -O2 -fno-builtin -ffp-contract=off
  -fno-fast-math`. Note OpenLibm's medium-range rounding (`fl(x·2/π + 1.5·2⁵²) − 1.5·2⁵²`)
  differs from current FreeBSD's `rnint`; OpenLibm is the reference because it is the one run.
- **Constants.** All 19 scalar constants recomputed at 400 bits and parsed from the C decimal
  strings; both routes agree with the C hex comments.
- **Worst cases.** Per binary exponent, the doubles closest to a multiple of π/2, from continued
  fractions of 2^(E−52)·2/π. The generator recovers the published worst case
  6381956970095103·2^797 (|x mod π/2| = 2^−60.89).
- **Sounio source** (the text that goes into `pure.sio` and the lean_single prelude): bit-identical
  to the reference on 2458 rows — specials, every high-word branch threshold ± 1, multiples of π/2
  ± 1, the 1020 worst cases (120 of them also negated), 900 random — under both engines
  (committed binaries).
- **Madaros instruction list** (`scripts/research/fdlibm_sin_cos_oracle.py`): rendered as C and
  diffed against the reference, 0 differing results on 1,009,578 inputs (4338 specials and
  thresholds, 2040 signed worst cases, 200,000 random across four bands, 803,200 from a sweep of
  every exponent 20..1023). Size: sin 2002 / cos 2030 instructions with every call inlined, 105
  rodata constants, 145 frame slots (1160 bytes, frame 1280), ≈124 rip-relative loads.
- **Repo oracle, re-run from the tree** (tables embedded, nothing read from outside it):
  `python3 scripts/research/fdlibm_sin_cos_oracle.py --against-openlibm <openlibm>` →
  `1007675 inputs, 1007675 results each side, 0 differing`. Its Python interpreter agrees with
  the C rendering on 500 sampled inputs; the worst case executes 4086 instructions per call.
- **Generated artefacts** (`python3 scripts/dev/gen_fdlibm_trig.py --check` → all five fresh):
  `self-hosted/native/math_fdlibm_trig.sio` (2594 lines), the `pure.sio` and lean_single prelude
  blocks (the latter 398 `append_src_lit` lines), and two 328-row fixtures whose expected bits are
  the interpreter's.
- **Committed engines, before the rebuild:** `tests/stdlib/math/test_sin_cos_fdlibm.sio`
  (the new `math::pure` source) passes 328/328 under lean_single and Madaros; the generated
  lean_single prelude text, compiled as user code, passes the native fixture 328/328 under both.
  Negative controls: `tests/run-pass/native_sin_cos_fdlibm_vectors.sio` fails 585 checks on the
  committed Madaros (old builtins) and hangs at row 2 on the committed lean_single (§3.6).
- **Pre-existing, unchanged:** `stdlib/math/pure.sio`'s own test fails check 16 (lean_single) and
  checks 16–17 (Madaros) — `atan` / `atan2` — identically on the untouched base worktree.

## 5. Verification still owed before merge

1. lean_single seed rebuilt from the patched `lean_single.sio` (`make build`, fixed point).
2. Madaros built from source (`make build-madaros`, bare, never inside `souc-build-lock.sh`).
3. The vector tests on both engines built in 1–2, not on committed binaries (CLAUDE.md §15).
4. `bash scripts/run_sio_test_suite.sh` on the math and trig consumers.
5. `bin/llm-offload -t math-review` on this dispatch (M1).

## 6. Out of scope

- The legacy `self-hosted/native/codegen.sio` still carries its own Taylor `emit_builtin_sin/cos`
  (reachable only from smoke drivers).
- `tan`, `asin`, `acos`, `atan` and the other transcendental builtins.
- The Madaros unary-minus defect (§3.2) and the lean_single literal reader (§3.1).

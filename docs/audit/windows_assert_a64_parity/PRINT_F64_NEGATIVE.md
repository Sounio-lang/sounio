<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.print-f64-negative
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.print-f64-negative
-->

# A64 BUG — `print_f64` prints `-0.000000` for all negative values

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED.
**Class.** Codegen bug fix on `self-hosted/compiler/lean_single.sio`
(`emit_print_f64_a64`).
**Branch.** `feat/windows-assert-exit`.
**Found during.** First-order AD shadow verification (`AD_SHADOWS_FIRST_ORDER.md`)
— it masked the correct negative cos/acos sensitivities on the M3.

---

## §1 — The bug

On `aarch64-*`, `print_f64` printed the correct sign but a zero magnitude for
every negative value: `print_f64(-0.479426)` → `-0.000000`, `print_f64(-42.5)` →
`-42.000000`-ish/`-0.000000`. Positive values printed correctly. (x86 unaffected.)

## §2 — Root cause

`emit_print_f64_a64` masks x0 to `|value|` (clears the sign bit), then for a
negative input prints `'-'` via `emit_print_char_lit_a64`, and only **after that**
did `fmov d0, x0` to load the magnitude for digit extraction.

`emit_print_char_lit_a64` performs a `write` syscall, which sets `x0 = 1` (bytes
written) — clobbering the masked magnitude. So `fmov d0, x0` loaded the bit
pattern `1` (a denormal ≈ 0) and the magnitude printed as `0.000000`. Positive
values skipped the `'-'` print (`b.ge sign_done`), so x0 survived — which is why
only negatives were affected.

## §3 — The fix

Move `fmov d0, x0` to **before** the sign-check/print, so `|value|` is captured
into `d0` while x0 still holds it. FP registers survive the `write` syscall, so
the subsequent `'-'` print (which clobbers x0/x1/x2/x8) no longer corrupts the
magnitude. One-instruction reorder; no new code.

## §4 — Verification

- **Self-host fixed point.** `lean_single_fixed_point_gate.sh` PASS:
  stage1==stage2==stage3, `md5=d6f9d37c222f30ad483daa32ebc88092`; binary rebuilt.
- **Real M3** (`aarch64-macos`): `print_f64` of `-0.479426`, `-1.154701`,
  `3.141593`, `0.0`, `-42.5` all print correctly and **byte-match x86**.
- **x86 non-regression.** `epistemic_hessian_transcendentals` (prints negative
  f64) exits 0 with correct output.

## §5 — Note

`print_f64_n` (N-decimal variant) is a separate path — a Sounio prelude helper
`__native_print_f64_n`, register-allocated by the compiler, not hand-emitted —
so it does not share this syscall-clobber hazard.

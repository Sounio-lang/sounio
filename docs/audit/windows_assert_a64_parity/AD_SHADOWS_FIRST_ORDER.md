<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.ad-shadows-first-order
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.ad-shadows-first-order
-->

# A64 PARITY — first-order AD shadows through transcendentals

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED (first-order; second-order Hessian is a
follow-up).
**Class.** Codegen edit on `self-hosted/compiler/lean_single.sio`
(`compile_primary_a64` math-builtin block).
**Branch.** `feat/windows-assert-exit`.
**Cadence siblings.** `AUDIT.md`, `TRIG_BUILTINS.md`, `READ_LINE.md`.

---

## §1 — The gap

The x86 expression compiler propagates β⁶/β⁷ chain-rule AD shadows through the
transcendental builtins (`sensitivity_of(f(x), k) = f'(x)·sens_k(x)`). The ARM64
compiler's math-builtin block had **no** shadow propagation — it emitted the call
and left the gradient channels untouched, so `sensitivity_of` through any
transcendental returned 0 on `aarch64-*`. (a64 multiply/add shadows already
worked; only the math builtins were missing — confirmed on M3: `x*y` →
2,3 matches x86, `sqrt` sensitivity → a64 `0.0` vs x86 `0.288675`.)

## §2 — The fix

In `compile_primary_a64`, capture the argument's gradient shadows (`s0..s3`)
before the native call, then after it propagate `f'(arg)·s_k` per active channel,
mirroring the x86 first-order block. `f'` per function:

| fn | f'(x) | fn | f'(x) |
|---|---|---|---|
| sqrt | 1/(2·result) | tan | 1+result² |
| exp | result | atan | 1/(1+arg²) |
| ln/log | 1/arg | tanh | 1−result² |
| sin | cos(arg) | asin | 1/√(1−arg²) |
| cos | −sin(arg) | acos | −1/√(1−arg²) |

Built from the existing a64 helpers: `emit_f64_binop_a64`
(fmul `0x1E610800` / fadd `0x1E612800` / fsub `0x1E613800` / fdiv `0x1E611800`),
`emit_imm64_a64`, `emit_push_a64`/`emit_pop_x1_a64`, `emit_load_var_a64`/
`emit_store_var_a64`, `emit_direct_fn_call_a64`. Negations (`cos`, `acos`) use
`0 − x` via `fsub` rather than a sign-bit trick. After propagation the primal
result is reloaded into x0 for downstream consumers (matches x86).

Math builtins carry **4 gradient channels** (s0..s3), matching x86; s4..s7 are
cleared. Second-order Hessian propagation (and porting `hessian_of`, currently
x86-only) is a deliberate follow-up — NOT in this change.

## §3 — Verification (real hardware)

qemu-user-static is unreliable for souc's a64 binaries (see
`../../../<memory> a64_qemu_runtime_blocker`), so verification ran on a **real
Apple M3** (`aarch64-macos`, macOS 26.5) over SSH.

- **Self-host fixed point.** `lean_single_fixed_point_gate.sh` PASS:
  stage1==stage2==stage3, `md5=c16a5c6ed43338edd226b20d080b5755`; binary rebuilt.
- **M3, all 10 transcendentals** at x=0.5, `sensitivity_of(f(x.value),0)`, matched
  the x86 ground truth exactly: sqrt 0.707107, exp 1.648721, ln 2.000000,
  sin 0.877583, cos −0.479426, tan 1.298446, atan 0.800000, tanh 0.786448,
  asin 1.154701, acos −1.154701.
- **x86 non-regression.** `sensitivity_transcendental`, `sensitivity_trig`,
  `sensitivity_div`, `sensitivity_multi_channel`, `test_lie` all exit 0.

## §4 — Separate pre-existing bug found (NOT addressed here)

`print_f64` on a64 (`emit_print_f64_a64`) prints `-0.000000` for **every**
negative f64 (e.g. `print_f64(-0.479426)` → `-0.000000`; x86 prints it correctly).
This is unrelated to AD — it masked the (correct) negative cos/acos sensitivities
during testing until verified by printing the negated magnitude. Tracked as its
own a64 parity item.

## §5 — Conclusion

First-order AD shadows now propagate through all 10 transcendentals on the ARM64
backend, at parity with x86, verified numerically on real hardware. Follow-ups:
second-order Hessian shadows + `hessian_of` on a64; the `print_f64` negative bug.

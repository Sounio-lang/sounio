<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.trig-builtins
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.trig-builtins
-->

# A64 PARITY — `tan / atan / tanh / asin / acos` math builtins

**Opened / closed.** 2026-05-21 (same session).
**Status.** RESOLVED — CODE CHANGE LANDED. The ARM64 expression compiler now
recognises the same transcendental builtins as x86.
**Class.** Codegen edit on `self-hosted/compiler/lean_single.sio`
(`compile_primary_a64`). Self-hosted; requires the self-host fixed point.
**Branch.** `feat/windows-assert-exit`.
**Predecessor item.** `docs/audit/windows_assert_a64_parity/AUDIT.md`
(`emit_assert_fail_a64` — audit, no change). Cadence sibling of `98f64497a`
("A64 parity: tuple_destructure_from_ptr_a64").

---

## §1 — The gap

Two parallel expression compilers exist: `compile_primary` (x86, ~`:11562`)
and `compile_primary_a64` (~`:27428`). The x86 math-builtin branch dispatched
**eleven** transcendentals:

```
sqrt sin cos exp ln log  tan atan tanh asin acos
```

The a64 branch dispatched only the first **six** (`sqrt sin cos exp ln log`).
So on `aarch64-macos` / `aarch64-linux`, a call to `tan(x)` (or `atan`,
`tanh`, `asin`, `acos`) was **not** recognised as a math builtin — it fell
through to the generic-identifier path. The five `__native_<fn>_f64` prelude
helpers it should have called already existed (injected arch-independently via
`append_src_lit`, `:31365`–`:31415`), so the helpers were reachable on x86 but
unreferenced on ARM64.

## §2 — The fix

Extended the `compile_primary_a64` math-builtin branch to mirror x86 exactly:
added `tan / atan / tanh / asin / acos` to the recognition condition and the
five corresponding `math_fi = fn_find_lit("__native_<fn>_f64")` dispatch lines,
in the same order x86 uses. Each compiles to `emit_direct_fn_call_a64(math_fi, 1)`
— the identical mechanism already used for `sin`/`cos`/`exp`/`ln`. One-site
change; no new helper, no new ABI.

**Deliberately preserved asymmetry:** the x86 branch also captures β⁶/β⁷
chain-rule AD shadows for transcendentals. The a64 branch does **not** do this
for *any* transcendental (not the original six, not the new five). That AD-shadow
gap is a separate, larger parity item; this change keeps the a64 treatment of
the new five identical to the existing six. Do not mistake the new five for a
special case.

## §3 — Verification

- **Self-host fixed point.** `lean_single_fixed_point_gate.sh` PASS:
  shipped == stage1 == stage2 == stage3, `md5=dd4b2b5b17e759b6875a7f534a33dd03`
  (size 2 173 750 B). `bin/souc-linux-x86_64` rebuilt to match (no out-of-sync
  warning). Self-reproduction (stage1 == stage2) reconfirmed independently.
- **A64 codegen, empirical.** Compiled a `tan(x); atan(x); asin(x)` program with
  `--target aarch64-linux`; the resulting ELF's `main` emits three distinct
  `bl` calls to three distinct `__native_*_f64` helpers. The pre-change shipped
  compiler did not route these through the math-builtin path.
- **No x86 regression.** Existing trig run-pass tests (`test_lie.sio`,
  `epistemic_hessian_transcendentals.sio`; together exercise all five new fns)
  compile and run to exit 0 with the rebuilt binary.
- **E200 note.** Small programs compiled to aarch64 emit ~10 non-fatal
  `E200 unknown identifier` prelude warnings; these are **pre-existing** (a clean
  rebuild *without* this change reproduces the shipped binary `479fad32…` and the
  same 10 warnings) and unrelated to this item — count is identical with and
  without trig in the test.

## §4 — Conclusion

`tan / atan / tanh / asin / acos` are now first-class on the ARM64 backend,
at parity with x86 (modulo the documented AD-shadow asymmetry). Remaining
known a64↔x86 expression-compiler asymmetries (e.g. transcendental AD shadows,
`read_line`) are tracked as future parity items.

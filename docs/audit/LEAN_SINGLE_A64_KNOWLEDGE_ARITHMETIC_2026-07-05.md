<!-- docs:meta
topic_id: repo.docs.audit.lean-single-a64-knowledge-arithmetic-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-a64-knowledge-arithmetic-2026-07-05
-->

# lean_single forensic dispatch — aarch64 had no `Knowledge<T>` arithmetic codegen at all

Date: 2026-07-05
Branch: `main` (post-PR #642, a64 two-level field-store fix)
Class: **two missing aarch64 subsystems** — no `Knowledge<T>` binary-operator
codegen at all (segfault), plus a separate, independently-discovered missing
`.uncertainty` field-read case (silent wrong value) — closes the last 2 of
the 4 aarch64 `native_runtime` failures left out of scope by the two
preceding a64 dispatches
Status: fixed, verified — aarch64 native_runtime manifest **99/99 pass** (up
from 97/99 after the previous dispatch; **all four** originally-failing
tests across this three-dispatch aarch64 arc are now closed), full x86-64
suite 1314/0/124/689 exact baseline (fix is aarch64-only)

## Symptom

```sio
let a: Knowledge<f64> = measure(10.0, uncertainty: 0.3, confidence: 0.8)
let b: Knowledge<f64> = measure(20.0, uncertainty: 0.4, confidence: 0.6)
let sum = a + b   // segfaults on aarch64
```

`measure()` and `.value`/`.uncertainty`/`.confidence` field access all work
correctly on aarch64 in isolation; the crash is isolated to the binary
arithmetic operators (`+ - * /`) when either operand is `Knowledge<T>`.

## Root cause #1: `compile_additive_a64()`/`compile_multiplicative_a64()` never checked for `Knowledge<T>` at all

x86-64's `compile_additive()`/`compile_multiplicative()` check
`knowledge_hash_is(left_hash) || knowledge_hash_is(right_hash)` before
falling into the plain-scalar arithmetic path, dispatching to
`compile_knowledge_addsub_x86()`/`compile_knowledge_muldiv_x86()` instead.
`grep` confirmed neither `compile_additive_a64()` nor
`compile_multiplicative_a64()` contained the string `knowledge_hash_is`
anywhere, and neither `compile_knowledge_addsub_a64()` nor
`compile_knowledge_muldiv_a64()` existed. `Knowledge<T>` is represented as a
pointer to a 3-word struct (`value`@0, `variance`@8, `confidence`@16); with
no special-casing, `a + b` fell into the plain scalar path and added the two
**pointers** as if they were f64 bit patterns, producing a garbage result
that segfaulted on the next `.value`/`.uncertainty`/`.confidence` field
dereference.

**Fix**: ported `compile_knowledge_addsub_x86()`/`compile_knowledge_muldiv_x86()`
to aarch64 instruction-for-instruction, along with their primitive
dependencies (`emit_load_operand_value/variance/confidence_a64`,
`emit_i64_to_f64_in_x0_a64`, `emit_min_positive_f64_bits_a64` — a
branch-based min using the same conditional-branch-and-patch pattern already
proven elsewhere in this file's a64 codegen, avoiding an untested CSEL
encoding). `emit_f64_binop_a64()` (already existing, parameterized by FP
opcode) covers add/sub/mul/div uniformly, unlike x86-64's four separate
named helpers. Wired the `knowledge_hash_is` dispatch into
`compile_additive_a64()`/`compile_multiplicative_a64()`, adding the missing
`left_ty`/`left_hash`/`right_ty`/`right_hash`/`op_tok` captures those
functions never had (only `left_f64` was tracked, since the existing GTT
sensitivity-shadow-tracking code — a separate, already-ported-to-a64,
orthogonal feature for plain-f64 automatic differentiation — didn't need
them). Confirmed via a register-preserving runtime-debug-print technique
(see "Debugging methodology" below) that every intermediate value (operand
values, variances, confidence) is bit-for-bit correct through this new
codegen before removing the instrumentation.

The multiply/divide variance chain-rule math (`Var(a·b) ≈ a²Var(b) +
b²Var(a)`, `Var(a/b) ≈ Var(a)/b² + a²Var(b)/b⁴`) was ported by mechanically
translating each `em(0x50)`(push)/`em(0x59)`(pop rcx)/`emit_f64_*_from_rcx_rax_x86()`
triple to `emit_push_a64()`/`emit_pop_x1_a64()`/`emit_f64_binop_a64(opcode)`
— x86-64's "rcx OP rax" convention and `emit_f64_binop_a64`'s "x1 OP x0"
convention are the same shape, so the translation preserves the exact
operand ordering non-commutative operations (division) depend on. Verified
independently against the mul/div assertions from the original x86-64 test.

## Root cause #2 (discovered while verifying #1, pre-existing, unrelated to arithmetic): `.uncertainty` field read was never ported to aarch64 either

Fixing root cause #1 alone did not make `epistemic_ops_42` pass: it stopped
segfaulting, but `sum.uncertainty` read back as `0` instead of `0.5`.
Isolated with a minimal repro to a **plain `measure()`-created value with no
arithmetic at all** (`a.uncertainty` on the original `a`, not `a+b`) —
confirming this is completely independent of root cause #1 and predates
this dispatch entirely.

`compile_postfix_a64()`'s generic `Knowledge<T>` field-access branch had
`"value"`, `"variance"`, and `"confidence"`/`"epsilon"` cases, but no
`"uncertainty"` case — x86-64's equivalent branch has all four. A
`.uncertainty` access fell through to the final `else { tc_error(...) }`,
and since `tc_error()` is non-fatal (prints a diagnostic but does not halt
codegen — see `reference_sounio_visibility_model` in project memory), `x0`
and `EXPR_IS_F64` were simply left at whatever they were before, silently
reading as `0`.

**Fix**: added the missing case, `ldr x0,[x0,#8]` (load the raw stored
`variance`) then `emit_sqrt_builtin_a64()` (new — aarch64 had no `sqrt`
primitive usable for this either; `fmov d0,x0; fsqrt d0,d0; fmov x0,d0`,
mirroring `emit_sqrt_builtin_x86`'s `movq`/`sqrtsd`/`movq` shape) —
`.uncertainty` surfaces `sqrt(variance)`, since `variance` (not the surface
uncertainty) is what `measure()` actually stores at offset 8.

## Debugging methodology

Both root causes were isolated using the `qemu-aarch64-static` local-repro
technique established in the two preceding dispatches (cross-compile to
`--target aarch64-linux`, run under `qemu-aarch64-static`, no macOS hardware
needed). Root cause #2 specifically required distinguishing "my new
arithmetic codegen computes the wrong variance" from "the arithmetic is
right but something downstream misreads it" — resolved by temporarily
injecting register-preserving runtime print calls
(`emit_push_a64(); emit_print_int_a64(); emit_pop_x0_a64()`, careful to
route the actual live register through a named temp slot rather than extra
unbalanced stack pushes) directly into the generated machine code at each
intermediate step, confirming the stored variance bit pattern (`0.25`,
`0.09+0.16`) was correct **before** ever suspecting the field-read side —
avoiding a wasted re-audit of already-correct arithmetic.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf

# Exact CI repros, cross-compiled + run under qemu-aarch64-static:
# epistemic_ops_42: exit 42 (was SIGSEGV, then briefly exit 2 mid-fix before root cause #2)
# epistemic_propagation: exit 0 (was SIGSEGV)

# Full aarch64 native_runtime manifest (99 cases): 99 pass / 0 fail — up
# from 97/2 after the previous dispatch. All 4 originally-failing tests
# across this three-PR aarch64 arc (#640, #642, this one) now closed.

bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127 — exact baseline
```

Also confirmed directly: the full multiply/divide variance chain-rule
formulas against the original x86-64 test's exact numeric assertions
(`prod.value≈200.0`, `prod.uncertainty≈7.211`, `quot.value≈0.5`,
`quot.uncertainty≈0.018`) — all pass on aarch64.

## Discovered but explicitly out of scope

`compile_knowledge_muldiv_x86`'s own comment (ported verbatim into the a64
twin) documents a **known, pre-existing limitation shared by both
backends**: Knowledge<T> multiplication/division does not propagate
`EXPR_SSHADOW_*`/`EXPR_HSHADOW_*` (the GTT sensitivity/Hessian shadow
state) — a `hessian_of`/`sensitivity_of` on a direct-Knowledge-arithmetic
result silently returns zero for those terms. This is Lean-formalised as an
accepted boundary in `formal/KnowledgeArithmeticSoundness.lean` and
documented in `docs/compiler/KNOWN_LIMITATIONS.md`; this dispatch preserves
it identically on both backends rather than attempting to close it (a
Phase-5 architectural item per the existing comment, unrelated to the
crash/wrong-value bugs fixed here).

## Cross-references

- `docs/audit/LEAN_SINGLE_A64_STRUCT_FIELD_AGGREGATE_COPY_2026-07-05.md` —
  first dispatch in this aarch64 arc (Release Gate root cause).
- `docs/audit/LEAN_SINGLE_A64_TWO_LEVEL_FIELD_STORE_2026-07-05.md` — second
  dispatch, fixed 2 of the 4 remaining failures; left these 2 explicitly out
  of scope pending this dispatch.
- `reference_a64_codegen_gotchas` (project memory) — this dispatch adds two
  new entries: the branch-based (not CSEL) min-of-two-positives pattern, and
  the register-preserving runtime-print debugging technique.

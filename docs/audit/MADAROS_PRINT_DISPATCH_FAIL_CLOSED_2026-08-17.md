<!-- docs:meta
topic_id: repo.docs.audit.madaros-print-dispatch-fail-closed-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-print-dispatch-fail-closed-2026-08-17
-->

# Madaros unresolved print dispatch: fail closed

**Date:** 2026-08-18
**Engine boundary:** Madaros native lowering only. `lean_single` resolves and
executes the integer `if`-expression witnesses used here.
**Implementation base:** `origin/main` at `6098da3e82`, including #1799. The
source-current baseline was built at `d3ea284caf`; `lower.sio` and
`module_frontend.sio` are byte-for-byte unchanged between that commit and this
implementation base.

## 1. Decision

An argument may reach the char-pointer `print` builtin only when Madaros has
positive evidence that the expression produces a string. Known integer and
floating-point operands continue to use `print_int` and `print_f64`. A kind-0
operand is refused during lowering; it is not sent to `print`, and it is not
silently redirected to `print_int`.

This extends, rather than repeats, section 5 of
`MADAROS_PRINTLN_BOOL_SCALARKIND_SEGV_2026-08-17.md`. That audit closed two
positive-classification holes and directly measured the remaining integer
`if`-expression crash. The residual is decisive evidence that the open-ended
default is the defect: another scalar-kind patch would close one expression
form while leaving the next unclassified form as a pointer dereference.

## 2. Why refusal, not the numeric fallback

Changing kind 0 to `print_int` removes the `strlen(value-as-pointer)` crash, but
it does not establish that the value is an integer. A missed string producer
would print its address or packed handle as a decimal number. That is a wrong
answer, can disclose process-layout information, and turns a compiler knowledge
gap into apparently valid program output.

Refusal preserves the stronger invariant: Madaros either selects a printer from
positive type evidence or emits no program. It trades availability for honesty.
That trade is visible and recoverable; fabricated numeric output is neither.

The eventual completeness fix is to carry the checker's resolved operand type
to lowering. Until that path exists, a kind-0 value is evidence that Madaros
does not know enough to choose an ABI, not evidence that the value is an i64.

## 3. How many sites change

There are exactly two compiler consumers of `println_dispatch_name`:

1. a source call to bare `print(...)`;
2. a source call to `println(...)` before the newline emission.

Both change only when the first argument has scalar kind 0 and is not positively
recognized as a string. Explicit `print_int`, `print_f64`, and `print_char`
calls do not use this decision.

A repository-wide lexical census on this base gives the following upper bound:

| spelling | occurrences in `*.sio` | files |
|---|---:|---:|
| `print(` | 62,265 | 2,310 |
| `println(` | 32,244 | 1,793 |
| total | 94,509 | not additive |

Command shape: `rg --count-matches --glob '*.sio' '\\bNAME\\('`, summing the
per-file counts. This is deliberately labeled an upper bound: it includes
definitions, comments, strings, unreachable code, explicit cases already
classified as int/float/string, and programs never lowered by Madaros.

The exact behavioral count cannot be obtained from text search. It is the
dynamic set of Madaros-lowered call sites for which
`expr_result_scalar_kind_ref == 0` and string recognition is false. Claiming an
exact corpus number without instrumenting that branch would overstate the
evidence.

## 4. What is lost

The change can reject a type-correct program that `lean_single` accepts when
Madaros's lowering classifier has not learned the expression form. The integer
`if`-expression witness is intentionally such a case. This is a new, explicit
cross-engine acceptance difference.

The old default also allowed unresolved string producers to print correctly by
accident. The gate makes that loss concrete with a type-correct string
`if`-expression: Madaros refuses it while `lean_single` prints `left`. Positive
controls therefore cover:

- a string parameter;
- a string-returning call;
- ordinary integer and f64 operands.

Those controls define the minimum string surface that must remain executable.
An unrecognized future string producer will now be refused instead of being
treated as text. That is the cost of fail-closed behavior, and it should be
removed by extending positive string evidence or carrying checked types, not by
restoring the unsafe default.

## 5. Acceptance evidence

`scripts/ci/madaros_print_dispatch_refusal_gate.sh` builds Madaros from the
current source unless an explicit source-current binary is supplied. It checks:

- `print(if-expression-int)` and `println(if-expression-int)` return nonzero,
  emit the dispatch diagnostic, and leave no ELF;
- a string-valued `if`-expression is also refused by Madaros, recording the
  intentional acceptance loss rather than hiding it;
- all three refusal sources execute under forced `lean_single`, making the
  engine boundary explicit;
- string parameter, string-returning call, integer, and f64 controls compile
  and execute under Madaros.

The pre-fix run is expected to falsify the gate by compiling all three refusal
witnesses. The two integer programs then fail at runtime; the string program
prints successfully only because the old unsafe default happens to choose the
correct ABI. The post-fix run must end with
`status=pass total=6 passed=6 failed=0 not_run=0`.

### Current-main falsifier

Slurm job `10205` built Madaros from exact source commit
`d3ea284caf0d490aabcda3207ba7de15e8928bd2` with
`scripts/ci/build_modular_madaros.sh`. The resulting ELF had SHA-256
`ac82ead7ab1c07171c7340610fffbda4dc92a5b7fb114d8f9f62f4040d8a0b98`.
Running the six-case gate against that binary produced the required red
baseline:

```text
[madaros-print-dispatch] FAIL: unresolved-print-if was accepted by Madaros (runtime rc=139)
[madaros-print-dispatch] FAIL: unresolved-println-if was accepted by Madaros (runtime rc=139)
[madaros-print-dispatch] FAIL: unresolved-string-if was accepted by Madaros (runtime rc=0)
[madaros-print-dispatch] PASS: string-param compiled and executed under Madaros
[madaros-print-dispatch] PASS: string-return compiled and executed under Madaros
[madaros-print-dispatch] PASS: scalar-controls compiled and executed under Madaros
status=fail total=6 passed=3 failed=3 not_run=0
```

The two integer witnesses reached `strlen` with a scalar value interpreted as a
pointer and terminated by signal 11. The string witness executed only because
the unsafe fallback happened to choose its correct ABI. This is the exact
behavioral boundary changed by the patch.

### Branch falsifier and repaired receipt

The first branch build, Slurm job `10199` at `e46811bbd7`, was intentionally
treated as a falsifier rather than a success. It exposed two defects in the
initial patch: the generic `ir_bodies_failed` path hid the durable diagnostic,
and a declared string parameter lacked positive string evidence. The gate
finished `status=fail total=6 passed=2 failed=4 not_run=0`.

The repair reports the stored hard-error reason on every body-lowering exit and
recognizes declared `string`/`&string` parameter types. Slurm job `10204` built
the repaired compiler at `ec9f09423d`; its source file hashes are unchanged in
the rebased commits:

```text
2fffbe6201247447737bc293d50e825a6c65eb0403ecf17cdca1e60ee47dd3cb  self-hosted/ir/lower.sio
bb7498e248fc9a645f602fe763ea7c380e449b9f19aae6d9e8b609b9fe56175d  self-hosted/compiler/module_frontend.sio
```

The rebuilt ELF had SHA-256
`6185bbcc2d43b84708b574c0f6b2d4b63a3a01047b2e931ae188af38fbb2cf9c`.
The six-case receipt was:

```text
[madaros-print-dispatch] PASS: unresolved-print-if is fail-closed in Madaros and executes in lean_single
[madaros-print-dispatch] PASS: unresolved-println-if is fail-closed in Madaros and executes in lean_single
[madaros-print-dispatch] PASS: unresolved-string-if is fail-closed in Madaros and executes in lean_single
[madaros-print-dispatch] PASS: string-param compiled and executed under Madaros
[madaros-print-dispatch] PASS: string-return compiled and executed under Madaros
[madaros-print-dispatch] PASS: scalar-controls compiled and executed under Madaros
MADAROS_PRINT_DISPATCH_REFUSAL_GATE_OK
status=pass total=6 passed=6 failed=0 not_run=0
```

The forced-`lean_single` controls printed `LEAN_PRINT=41` and
`LEAN_PRINTLN=41`, and `LEAN_STRING=left`, all with rc 0. Both Madaros binaries
were rebuilt from their stated source snapshots on the Sounio Compiler Foundry
Slurm cluster through the login pod, using OrangeFS run root
`/orangefs/training/fix3-print-dispatch-e46811bbd7-20260818`. The checked-in
Madaros wrapper was used only for syntax smoke checks, never as implementation
evidence.

## 6. Claim boundary

This change proves a printer-dispatch safety property. It does not claim to fix
the three pre-existing `generic_struct` failures recorded by the parent audit,
and it does not claim general scalar-kind completeness.

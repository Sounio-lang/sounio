<!-- docs:meta
topic_id: repo.docs.llm-guide.examples
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.examples
-->

# Complete Sounio example atlas

Choose a task below, read one complete program, and run the matching command.
Do not load the whole atlas into a small model: retrieve the relevant example,
its contract, the needed API, and the matching guide section.

Every Sounio block is copied verbatim from the named repository source at
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. The manifest stores its hash.
The structural validator rejects edits that make these blocks drift from source.
A source fixture is not automatically a passing example: inspect each entry’s
verification in the [manifest](example-manifest.json) and the
[execution receipt](reviews/example-execution.md). This atlas does not turn a
failed or unrun case into a recommendation.

## How to use this atlas

1. Complete the root [first-success preflight](../../llms.txt#first-success):
   identify the host and bind the resolved raw artifact/hash before check/run.
   For this atlas's documented profile, match the committed Linux x86-64 digest;
   a custom/native artifact needs an explicitly separate profile receipt.
   Keep the canonical launcher and requested engine. For a configured tool,
   follow [MCP profile attribution](../../llms.txt#mcp-profile-attribution);
   a local terminal hash does not identify the server's compiler.
   Then select a positive entry for the task and that identified profile.
2. Read the declared return type, effects, imports, and expected exit/output.
3. From the repository root, execute the command shown against the original
   path before adapting it. Imported examples require their companion module
   at the original location; copying only the consumer into another directory
   does not preserve its resolver context. Preserve the preflight's artifact
   selection and library root; changing either requires a new binding.
4. Make one change, check again, then verify behavior. Recheck after every edit.
5. Negative entries are intentionally invalid. Use their diagnostic as a refusal
   control, never as a successful-program template.

Profile: baseline committed Madaros Linux x86-64 prebuilt, not a source rebuild.
The entry commands alone do not establish that artifact selection. For a
differently configured host, engine, or raw artifact, establish a separate
receipt. Program
acceptance and its existing assertions do not certify numerical accuracy,
scientific assumptions, medical safety, or all language semantics.

## Positive source candidates

The table records source intent, not a pass classification. Consult the
verification state in the manifest. The cases quarantined below are excluded
from the working learning path despite their original positive/negative labels.

| Example | Learning topic | Requested stage | Expected exit |
|---|---|---|---|
| [cook-while](#cook-while) | control/while-and-mutation | run | 0 |
| [hello](#hello) | syntax/entry-point | run | 0 |
| [struct](#struct) | types/struct | run | 0 |
| [generics](#generics) | functions/generics | run | 0 |
| [effects](#effects) | effects/superset | run | 0 |
| [cook-hello](#cook-hello) | syntax/entry-point | run | 0 |
| [cook-struct](#cook-struct) | types/struct | run | 0 |
| [cook-for-array](#cook-for-array) | control/array-iteration | run | 0 |
| [cook-tuple](#cook-tuple) | types/heterogeneous-tuples | run | 0 |
| [cook-borrow-release](#cook-borrow-release) | references/call-borrow-release | run | 0 |
| [cook-function-ref](#cook-function-ref) | functions/function-values | run | 0 |
| [cook-closure](#cook-closure) | functions/closures | run | 0 |
| [cook-result](#cook-result) | stdlib/result | run | 0 |
| [cook-range-option](#cook-range-option) | control/ranges-and-options | run | 0 |
| [cook-tuple-signature](#cook-tuple-signature) | types/tuple-signatures | run | 0 |
| [cook-array-ref](#cook-array-ref) | references/array-mutation | run | 0 |
| [cook-import-provider](#cook-import-provider) | modules/public-provider | check | 0 |
| [cook-import-main](#cook-import-main) | modules/selective-import | check | 0 |
| [cook-vec](#cook-vec) | stdlib/vector | run | 0 |
| [cook-enum](#cook-enum) | control/enum-match | run | 42 |
| [cook-generic-fn](#cook-generic-fn) | functions/explicit-generic | run | 42 |
| [cook-generic-pair](#cook-generic-pair) | types/generic-struct | run | 42 |
| [cook-epistemic-library](#cook-epistemic-library) | stdlib/epistemic-free-functions | run | 0 |
| [cook-eisa](#cook-eisa) | stdlib/eisa-origin | run | 0 |
| [cook-math-import](#cook-math-import) | modules/pure-math | run | 0 |
| [native-hello](#native-hello) | syntax/printing-and-arithmetic | run | 0 |
| [while-loop](#while-loop) | control/while-and-mutation | run | 0 |
| [functions](#functions) | functions/named-functions | run | 0 |
| [tuple-destructure](#tuple-destructure) | types/tuple-patterns | run | 0 |
| [shared-borrow](#shared-borrow) | references/shared-borrows | run | 0 |
| [knowledge-input](#knowledge-input) | knowledge/input-provenance-syntax | run | 0 |
| [knowledge-measured](#knowledge-measured) | knowledge/measured-provenance-syntax | run | 0 |
| [knowledge-value](#knowledge-value) | knowledge/explicit-value-access | check | 0 |
| [unit-add](#unit-add) | units/same-dimension-addition | run | 0 |
| [unit-scale](#unit-scale) | units/scalar-multiplication | run | 0 |
| [struct-parameter](#struct-parameter) | functions/struct-parameter | run | 0 |
| [struct-return](#struct-return) | functions/struct-return | run | 0 |
| [nested-struct](#nested-struct) | types/nested-fields | run | 0 |
| [strings](#strings) | types/string-length | run | 0 |
| [option-type](#option-type) | types/option-shorthand | check | 0 |

## Intended refusals

| Example | Boundary | Required diagnostic |
|---|---|---|
| [negative-effect-missing](#negative-effect-missing) | effects/missing-declaration | effect not declared |
| [negative-observe-io](#negative-observe-io) | knowledge/observe-boundary | cannot print Unobserved<T> |
| [negative-borrow-conflict](#negative-borrow-conflict) | references/exclusive-shared-conflict | cannot borrow exclusively while other borrows are active |
| [negative-knowledge-unwrap](#negative-knowledge-unwrap) | knowledge/no-silent-unwrapping | found Knowledge<f64> |
| [negative-unknown-effect](#negative-unknown-effect) | effects/unknown-name | unknown effect 'Zorblex' |
| [negative-unit-add](#negative-unit-add) | units/dimension-mismatch | unit mismatch |
| [negative-unit-return](#negative-unit-return) | units/return-dimension | unit mismatch in return value |
| [negative-unit-argument](#negative-unit-argument) | units/argument-dimension | unit mismatch in call argument |
| [negative-mut-effect](#negative-mut-effect) | effects/reference-store-mut | E035 |
| [negative-builtin-arity](#negative-builtin-arity) | functions/builtin-arity | E010 |

## Cases excluded from the working path

These failed their intended contracts on the pinned prebuilt. Their full
source is preserved below only for a diagnostic/reproduction exercise.
No universal absence of the language feature follows from this profile.

| Case | Original intent | Observed exit | Failed assertion |
|---|---|---|---|
| [cook-closure](#cook-closure) | positive / run | 1 | unexpected-exit |
| [cook-generic-fn](#cook-generic-fn) | positive / run | 1 | unexpected-exit |
| [cook-generic-pair](#cook-generic-pair) | positive / run | 1 | unexpected-exit |
| [tuple-destructure](#tuple-destructure) | positive / run | 1 | unexpected-exit |
| [unit-add](#unit-add) | positive / run | 1 | unexpected-exit |
| [unit-scale](#unit-scale) | positive / run | 1 | unexpected-exit |
| [negative-unit-return](#negative-unit-return) | negative / check-fail | 1 | missing-stdout-marker |
| [negative-unit-argument](#negative-unit-argument) | negative / check-fail | 1 | missing-stdout-marker |

## cook-while

**Topic:** control/while-and-mutation. **Kind:** positive.

**Source:** [examples/native/while_loop.sio](../../examples/native/while_loop.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "0 1 2 3 4 ".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-while in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/while_loop.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-while -->
```sio
// A3: While loop + mutable variables + assignment.
//
// Expected: prints "0 1 2 3 4 ", exits 0.

fn main() -> i64 with IO, Mut {
    var i: i64 = 0
    while i < 5 {
        print_int(i)
        print_char(32)
        i = i + 1
    }
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## hello

**Topic:** syntax/entry-point. **Kind:** positive.

**Source:** [examples/hello.sio](../../examples/hello.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "Hello, Sounio".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry hello in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/hello.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example hello -->
```sio
// First Sounio program

fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## struct

**Topic:** types/struct. **Kind:** positive.

**Source:** [examples/native/struct_basic.sio](../../examples/native/struct_basic.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "7".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry struct in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/struct_basic.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example struct -->
```sio
// Struct literal construction and field access in native-v2 driver.
// Expected: prints "7", exits 0.

struct Point { x: i64, y: i64 }

fn main() -> i64 with IO {
    let p = Point { x: 3, y: 4 }
    let s = p.x + p.y
    print_int(s)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## generics

**Topic:** functions/generics. **Kind:** positive.

**Source:** [tests/run-pass/generics_multi_param.sio](../../tests/run-pass/generics_multi_param.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "first_of_two: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry generics in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/generics_multi_param.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example generics -->
```sio
//@ run-pass
//@ expect-stdout-contains: first_of_two: PASS
//@ expect-stdout-contains: second_of_two: PASS
// Tests two-type-parameter generic functions via turbofish syntax

fn first<T, U>(a: T, b: U) -> T {
    a
}

fn second<T, U>(a: T, b: U) -> U {
    b
}

fn main() -> i64 with IO {
    var exit_fails: i64 = 0
    let x = first::<i64, i64>(42, 99)
    if x == 42 { print("first_of_two: PASS\n") } else { print("first_of_two: FAIL\n"); exit_fails = exit_fails + 1 }

    let y = second::<i64, i64>(10, 77)
    if y == 77 { print("second_of_two: PASS\n") } else { print("second_of_two: FAIL\n"); exit_fails = exit_fails + 1 }
    if exit_fails > 0 {
        return 1
    }
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## effects

**Topic:** effects/superset. **Kind:** positive.

**Source:** [tests/run-pass/effect_superset_ok.sio](../../tests/run-pass/effect_superset_ok.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "effect superset: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry effects in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/effect_superset_ok.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example effects -->
```sio
//@ run-pass

// Caller declares more effects than callee needs — should pass
fn pure_add(a: i64, b: i64) -> i64 {
    a + b
}

fn needs_io() with IO {
    print("io\n")
}

fn main() -> i32 with IO, Mut, Div {
    // Calling pure function from effectful context is fine
    let x = pure_add(2, 3)
    // Calling IO function from IO+Mut+Div context is fine
    needs_io()
    if x == 5 {
        print("effect superset: PASS\n")
    }
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-hello

**Topic:** syntax/entry-point. **Kind:** positive.

**Source:** [examples/hello.sio](../../examples/hello.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "Hello, Sounio".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-hello in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/hello.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-hello -->
```sio
// First Sounio program

fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-struct

**Topic:** types/struct. **Kind:** positive.

**Source:** [examples/native/struct_basic.sio](../../examples/native/struct_basic.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "7".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-struct in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/struct_basic.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-struct -->
```sio
// Struct literal construction and field access in native-v2 driver.
// Expected: prints "7", exits 0.

struct Point { x: i64, y: i64 }

fn main() -> i64 with IO {
    let p = Point { x: 3, y: 4 }
    let s = p.x + p.y
    print_int(s)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-for-array

**Topic:** control/array-iteration. **Kind:** positive.

**Source:** [tests/run-pass/for_in_array.sio](../../tests/run-pass/for_in_array.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-for-array in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/for_in_array.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-for-array -->
```sio
//@ run-pass
// for-in over a fixed-size array iterates exactly N times (logical length),
// not the allocation bucket capacity. Covers sum (reduction) + count.
// Returns 0 on success, nonzero on failure (exit codes are 8-bit, so we
// assert rather than return the raw 515).
fn main() -> i64 with Mut {
    var a: [i64; 5] = [0; 5]
    a[0] = 1
    a[1] = 2
    a[2] = 3
    a[3] = 4
    a[4] = 5
    var sum = 0
    var n = 0
    for v in a {
        sum = sum + v
        n = n + 1
    }
    if sum != 15 { return 1 }   // reduction over all 5 logical elements
    if n != 5 { return 2 }      // exactly N iterations, not the >=8 bucket
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-tuple

**Topic:** types/heterogeneous-tuples. **Kind:** positive.

**Source:** [tests/run-pass/tuple_mixed_f64_access.sio](../../tests/run-pass/tuple_mixed_f64_access.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "tuple mixed f64 access passed".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-tuple in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/tuple_mixed_f64_access.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-tuple -->
```sio
//@ run-pass
// Heterogeneous tuple element access: f64 and i64 mix.
// Tests that .0 / .1 carry the correct EXPR_IS_F64 flag after the
// tuple-hash encoding fix (first_f64/last_f64 lanes).

fn make_fi() -> (f64, i64) with IO {
    (1.5, 42)
}

fn make_if() -> (i64, f64) with IO {
    (7, 3.14)
}

fn main() with IO, Panic {
    // (f64, i64): .0 must be f64, .1 must be i64
    let fi = make_fi()
    let fi0: f64 = fi.0
    let fi1: i64 = fi.1

    // (i64, f64): .0 must be i64, .1 must be f64
    let if_ = make_if()
    let if0: i64 = if_.0
    let if1: f64 = if_.1

    // Arithmetic to exercise the values (not just load)
    let sum_f: f64 = fi0 + if1
    let sum_i: i64 = fi1 + if0

    assert(sum_i == 49)

    if sum_f > 4.6 && sum_f < 4.7 {
        println("tuple mixed f64 access passed")
    }
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-borrow-release

**Topic:** references/call-borrow-release. **Kind:** positive.

**Source:** [tests/run-pass/borrow_call_explicit_release.sio](../../tests/run-pass/borrow_call_explicit_release.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-borrow-release in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/borrow_call_explicit_release.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-borrow-release -->
```sio
//@ run-pass
//@ native-pass

fn read_shared(r: &i64) -> i64 {
    *r
}

fn read_exclusive(r: &!i64) -> i64 with Mut {
    *r
}

fn main() -> i32 with Mut {
    var x: i64 = 7

    // Explicit call-site borrows should be released after each call returns.
    let a = read_exclusive(&!x)
    let b = read_shared(&x)
    let c = read_exclusive(&!x)

    if a + b + c >= 0 {
        return 0
    }
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-function-ref

**Topic:** functions/function-values. **Kind:** positive.

**Source:** [examples/native/hof_fn_ref_call.sio](../../examples/native/hof_fn_ref_call.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "42".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-function-ref in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/hof_fn_ref_call.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-function-ref -->
```sio
fn double(x: i64) -> i64 { x * 2 }
fn triple(x: i64) -> i64 { x * 3 }
fn add(a: i64, b: i64) -> i64 { a + b }
fn sub(a: i64, b: i64) -> i64 { a - b }

fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
fn apply2(f: fn(i64, i64) -> i64, a: i64, b: i64) -> i64 { f(a, b) }

fn select_unary(which: i64) -> fn(i64) -> i64 with Mut, Panic, Div {
    if which == 0 { double }
    else if which == 1 { triple }
    else { double }
}

fn select_binary(which: i64) -> fn(i64, i64) -> i64 with Mut, Panic, Div {
    if which == 0 { add }
    else { sub }
}

fn main() -> i64 with IO, Mut, Panic, Div {
    let r1 = apply(double, 21)
    let r2 = apply2(add, 20, 22)
    let f = triple
    let r3 = f(14)
    let op = select_unary(0)
    let r4 = op(21)
    let bop = select_binary(0)
    let r5 = bop(40, 2)
    if r1 != 42 { return 1 }
    if r2 != 42 { return 2 }
    if r3 != 42 { return 3 }
    if r4 != 42 { return 4 }
    if r5 != 42 { return 5 }
    print_int(42)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-closure

**Topic:** functions/closures. **Kind:** positive.

**Source:** [tests/run-pass/closure_hof.sio](../../tests/run-pass/closure_hof.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-closure in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/closure_hof.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine cook-closure -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `unexpected-exit`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example cook-closure -->
```sio
//@ run-pass
//@ expect-stdout: PASS

fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

fn main() -> i64 with IO, Mut {
    var exit_fails: i64 = 0
    let result = apply(|x| x * 2, 21)
    if result == 42 { print("PASS\n") }
    else { print("FAIL\n"); exit_fails = exit_fails + 1 }
    return if exit_fails > 0 { 1 } else { 0 }
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-result

**Topic:** stdlib/result. **Kind:** positive.

**Source:** [tests/stdlib/core/test_result_stdlib.sio](../../tests/stdlib/core/test_result_stdlib.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "RESULT_STDLIB_OK".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-result in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/stdlib/core/test_result_stdlib.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-result -->
```sio
// Run-proof for stdlib core::result (IntResult/FloatResult). By-reference API -> default Madaros.
use core::result::*
fn main() -> i32 with IO, Mut, Div, Panic {
    let ok = int_ok(42)
    if !int_result_is_ok(&ok) { print("FAIL ok_is_ok\n"); return 1 }
    if int_result_is_err(&ok) { print("FAIL ok_not_err\n"); return 2 }
    if int_result_unwrap(&ok) != 42 { print("FAIL unwrap\n"); return 3 }
    let err = int_err(5)
    if !int_result_is_err(&err) { print("FAIL err_is_err\n"); return 4 }
    if int_result_unwrap_or(&err, 99) != 99 { print("FAIL unwrap_or\n"); return 5 }
    if int_result_err(&err) != 5 { print("FAIL err_code\n"); return 6 }
    if int_result_unwrap_or(&ok, 99) != 42 { print("FAIL ok_unwrap_or\n"); return 7 }
    let fok = float_ok(3.14)
    if !float_result_is_ok(&fok) { print("FAIL float_ok\n"); return 8 }
    if (if float_result_unwrap(&fok) > 3.14 { float_result_unwrap(&fok)-3.14 } else { 3.14-float_result_unwrap(&fok) }) >= 1.0e-9 { print("FAIL float_unwrap\n"); return 9 }
    print("RESULT_STDLIB_OK\n")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-range-option

**Topic:** control/ranges-and-options. **Kind:** positive.

**Source:** [tests/run-pass/while_for_struct_patterns.sio](../../tests/run-pass/while_for_struct_patterns.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "while_for_struct_patterns: ALL PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-range-option in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/while_for_struct_patterns.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-range-option -->
```sio
//@ run-pass

fn main() -> i32 with IO, Mut, Div, Panic {
    var passed: i32 = 0
    var failed: i32 = 0

    // T1: for loop basic range (already works)
    var sum: i64 = 0
    for i in 0..5 { sum = sum + i }
    if sum == 10 { passed = passed + 1 } else { println("T1 FAIL"); failed = failed + 1 }

    // T2: for loop with variable upper bound
    let n: i64 = 6
    var cnt: i64 = 0
    for i in 0..n { cnt = cnt + 1 }
    if cnt == 6 { passed = passed + 1 } else { println("T2 FAIL"); failed = failed + 1 }

    // T3: while let Some — new feature
    var wl_sum: i64 = 0
    var wl_v: Option<i64> = Some(1)
    while let Some(x) = wl_v {
        wl_sum = wl_sum + x
        if x >= 3 { wl_v = None } else { wl_v = Some(x + 1) }
    }
    // sum: 1+2+3 = 6
    if wl_sum == 6 { passed = passed + 1 } else { println("T3 FAIL"); failed = failed + 1 }

    // T4: while let exits on None immediately
    var wl_none: Option<i64> = None
    var wl_hit: i64 = 0
    while let Some(x) = wl_none { wl_hit = wl_hit + 1 }
    if wl_hit == 0 { passed = passed + 1 } else { println("T4 FAIL"); failed = failed + 1 }

    // T5: for loop array building
    var arr: [i64; 5] = [0; 5]
    for i in 0..5 { arr[i as usize] = i * i }
    if arr[3] == 9 && arr[4] == 16 { passed = passed + 1 } else { println("T5 FAIL"); failed = failed + 1 }

    if failed == 0 { println("while_for_struct_patterns: ALL PASS") }
    else { println("while_for_struct_patterns: FAIL") }
    failed
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-tuple-signature

**Topic:** types/tuple-signatures. **Kind:** positive.

**Source:** [tests/run-pass/tuple_signature_types.sio](../../tests/run-pass/tuple_signature_types.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-tuple-signature in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/tuple_signature_types.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-tuple-signature -->
```sio
//@ run-pass

// Regression for tuple type expressions in function signatures. Tuple
// parameters and tuple returns must lower as tuple types instead of falling
// through to ty_error/E000, then run with the expected values.

fn pair_sum(p: (i64, i64)) -> i64 {
    p.0 + p.1
}

fn make_pair(a: i64, b: i64) -> (i64, i64) {
    (a, b)
}

fn main() -> i64 {
    let direct = pair_sum((20, 22))
    let p = make_pair(40, 2)
    let returned = p.0 + p.1
    if direct == 42 && returned == 42 {
        0
    } else {
        1
    }
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-array-ref

**Topic:** references/array-mutation. **Kind:** positive.

**Source:** [tests/run-pass/array_mut_ref_bare.sio](../../tests/run-pass/array_mut_ref_bare.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "array_mut_ref_bare: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-array-ref in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/array_mut_ref_bare.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-array-ref -->
```sio
//@ run-pass
// Regression test: bare &![T; N] array parameter mutation visible to caller.
//
// Previously broken (JIT/codegen): when passing a bare array variable by &!
// exclusive reference, mutations inside the callee via `arr[i] = v` (without
// explicit deref) were not visible to the caller because the codegen path for
// arr[i] = v on a &![T;N] parameter was not distinguishing array-refs from
// plain scalars and emitting a byte store instead of a qword store.
//
// Fix in lean_single.sio: stmt_is_array_store and the primary array-read path
// now check VAR_TY == 10 with inner type == 8 and set is_arr_ty = 1, causing
// the correct `mov [rax + rcx*8], rdx` 8-byte store to be emitted.
//
// This test confirms all three mutation patterns work:
//   1. Bare index: arr[i] = v  (was broken)
//   2. Explicit deref: (*arr)[i] = v  (always worked)
//   3. Struct wrapper: b.field[i] = v  (always worked)

fn fill_bare(arr: &![i64; 4]) with Mut, Panic {
    var i = 0
    while i < 4 { arr[i as usize] = (i + 1) as i64; i = i + 1 }
}

fn fill_deref(arr: &![i64; 4]) with Mut, Panic {
    var i = 0
    while i < 4 { (*arr)[i as usize] = (i + 1) as i64; i = i + 1 }
}

struct Buf4 { data: [i64; 4] }
fn fill_struct(b: &! Buf4) with Mut, Panic {
    var i = 0
    while i < 4 { b.data[i as usize] = (i + 1) as i64; i = i + 1 }
}

fn main() -> i32 with IO, Mut, Panic {
    // Pattern 1: bare index (was broken — regression guard)
    var a: [i64; 4] = [0; 4]
    fill_bare(&! a)
    if a[0] != 1 || a[1] != 2 || a[2] != 3 || a[3] != 4 {
        println("array_mut_ref_bare: FAIL (bare index pattern)")
        return 1
    }

    // Pattern 2: explicit deref (always worked)
    var b: [i64; 4] = [0; 4]
    fill_deref(&! b)
    if b[0] != 1 || b[1] != 2 || b[2] != 3 || b[3] != 4 {
        println("array_mut_ref_bare: FAIL (explicit deref pattern)")
        return 1
    }

    // Pattern 3: struct wrapper (always worked)
    var buf: Buf4 = Buf4 { data: [0; 4] }
    fill_struct(&! buf)
    if buf.data[0] != 1 || buf.data[1] != 2 || buf.data[2] != 3 || buf.data[3] != 4 {
        println("array_mut_ref_bare: FAIL (struct wrapper pattern)")
        return 1
    }

    println("array_mut_ref_bare: PASS")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-import-provider

**Topic:** modules/public-provider. **Kind:** positive.

**Source:** [tests/conformance/import_public_a.sio](../../tests/conformance/import_public_a.sio).

**Intended command stage:** check. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-import-provider in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/conformance/import_public_a.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-import-provider -->
```sio
//@ auxiliary
// Provider module for the bounded conformance import check.

pub fn imported_add(a: i64, b: i64) -> i64 {
    a + b
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-import-main

**Topic:** modules/selective-import. **Kind:** positive.

**Source:** [tests/conformance/import_public.sio](../../tests/conformance/import_public.sio).

**Intended command stage:** check. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-import-main in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/conformance/import_public.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-import-main -->
```sio
//@ check-pass
// Positive import visibility check for the serious-language conformance spine.

use import_public_a::{imported_add}

fn main() -> i64 {
    imported_add(20, 22)
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-vec

**Topic:** stdlib/vector. **Kind:** positive.

**Source:** [tests/stdlib/collections/test_vec_stdlib.sio](../../tests/stdlib/collections/test_vec_stdlib.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "VEC_STDLIB_OK".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-vec in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/stdlib/collections/test_vec_stdlib.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-vec -->
```sio
// Run-proof for stdlib collections::vec (VecF64). By-reference API -> default Madaros.
use collections::vec::*
fn main() -> i32 with IO, Mut, Div, Panic {
    var v = vec_new()
    vec_push(&!v, 1.0)
    vec_push(&!v, 2.0)
    vec_push(&!v, 3.0)
    if vec_len(&v) != 3 { print("FAIL len\n"); return 1 }
    if (if vec_sum(&v) > 6.0 { vec_sum(&v)-6.0 } else { 6.0-vec_sum(&v) }) >= 1.0e-9 { print("FAIL sum\n"); return 2 }
    if (if vec_mean(&v) > 2.0 { vec_mean(&v)-2.0 } else { 2.0-vec_mean(&v) }) >= 1.0e-9 { print("FAIL mean\n"); return 3 }
    if (if vec_get(&v,1) > 2.0 { vec_get(&v,1)-2.0 } else { 2.0-vec_get(&v,1) }) >= 1.0e-9 { print("FAIL get\n"); return 4 }
    // set
    vec_set(&!v, 0, 10.0)
    if (if vec_get(&v,0) > 10.0 { vec_get(&v,0)-10.0 } else { 10.0-vec_get(&v,0) }) >= 1.0e-9 { print("FAIL set\n"); return 5 }
    // pop
    let top = vec_pop(&!v)
    if (if top > 3.0 { top-3.0 } else { 3.0-top }) >= 1.0e-9 { print("FAIL pop\n"); return 6 }
    if vec_len(&v) != 2 { print("FAIL len after pop\n"); return 7 }
    // reverse [10,2] -> [2,10]
    vec_reverse(&!v)
    if (if vec_get(&v,0) > 2.0 { vec_get(&v,0)-2.0 } else { 2.0-vec_get(&v,0) }) >= 1.0e-9 { print("FAIL reverse\n"); return 8 }
    print("VEC_STDLIB_OK\n")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-enum

**Topic:** control/enum-match. **Kind:** positive.

**Source:** [tests/selfhost/native_runtime/enum_match_green_42.sio](../../tests/selfhost/native_runtime/enum_match_green_42.sio).

**Intended command stage:** run. **Expected exit:** 42.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-enum in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/selfhost/native_runtime/enum_match_green_42.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-enum -->
```sio
//@ run-pass

enum Color { Red, Green, Blue }

fn main() -> i64 {
    let c = Color::Green
    match c {
        Color::Red => { return 10 }
        Color::Green => { return 42 }
        _ => { return 0 }
    }
    return 99
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-generic-fn

**Topic:** functions/explicit-generic. **Kind:** positive.

**Source:** [tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio](../../tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio).

**Intended command stage:** run. **Expected exit:** 42.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-generic-fn in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine cook-generic-fn -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `unexpected-exit`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example cook-generic-fn -->
```sio
fn id<T>(x: T) -> T {
    x
}

fn main() -> i64 {
    id<i64>(42)
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-generic-pair

**Topic:** types/generic-struct. **Kind:** positive.

**Source:** [tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio](../../tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio).

**Intended command stage:** run. **Expected exit:** 42.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-generic-pair in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine cook-generic-pair -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `unexpected-exit`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example cook-generic-pair -->
```sio
struct Pair<T> {
    first: T
    second: T
}

fn main() -> i64 {
    let p = Pair<i64> { first: 10, second: 32 }
    p.first + p.second
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-epistemic-library

**Topic:** stdlib/epistemic-free-functions. **Kind:** positive.

**Source:** [tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio](../../tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "KNOWLEDGE_MADAROS_IMPORT_E2E_OK".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-epistemic-library in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-epistemic-library -->
```sio
//@ run-pass
// D3: Epistemic free-function API under default Madaros multi-module import.
// Method form is gated separately (knowledge_method_parity / residual gate).
// claims_not_made: Knowledge<T> language builtin; full Root-2 census closed.
use epistemic::knowledge::{
    ep_measured, ep_val, ep_variance, ep_std, ep_add, ep_mul, ep_gate, ep_certain, ep_merge
}

fn approx_eq(a: f64, b: f64, tol: f64) -> i64 {
    let d = a - b
    let d2 = if d < 0.0 { 0.0 - d } else { d }
    if d2 <= tol { 1 } else { 0 }
}

fn main() -> i32 with IO, Mut, Div, Panic {
    let x = ep_measured(10.0, 0.5)
    let y = ep_measured(4.0, 0.2)
    if approx_eq(ep_val(&x), 10.0, 1e-12) == 0 { print("F1\n"); return 1 }
    if approx_eq(ep_variance(&x), 0.25, 1e-12) == 0 { print("F2\n"); return 2 }
    if approx_eq(ep_std(&x), 0.5, 1e-9) == 0 { print("F3\n"); return 3 }
    let s = ep_add(&x, &y)
    if approx_eq(ep_val(&s), 14.0, 1e-12) == 0 { print("F4\n"); return 4 }
    let p = ep_mul(&x, &y)
    if approx_eq(ep_val(&p), 40.0, 1e-12) == 0 { print("F5\n"); return 5 }
    if approx_eq(ep_variance(&p), 8.0, 1e-10) == 0 { print("F6\n"); return 6 }
    if ep_gate(&x, 900) != 1 { print("F7\n"); return 7 }
    let c = ep_certain(1.0)
    if ep_gate(&c, 1000) != 1 { print("F8\n"); return 8 }
    let m = ep_merge(&x, &y)
    if approx_eq(ep_val(&m), 4.827586, 1e-5) == 0 { print("F9\n"); return 9 }
    print("KNOW_IMPORT v="); print(ep_val(&x)); print(" sum="); print(ep_val(&s))
    print(" prod="); print(ep_val(&p)); print(" merge="); print(ep_val(&m)); print("\n")
    print("KNOWLEDGE_MADAROS_IMPORT_E2E_OK\n")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-eisa

**Topic:** stdlib/eisa-origin. **Kind:** positive.

**Source:** [tests/run-pass/eisa_origin_independent_add.sio](../../tests/run-pass/eisa_origin_independent_add.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "EISA_ORIGIN_INDEP_ADD OK u=5 origin=0 no-fire".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-eisa in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/eisa_origin_independent_add.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-eisa -->
```sio
//@ run-pass
//@ description: R7 negative control — distinct origins must NOT fire. 3-4-5 RSS.
//
// Oracle (integers only, no floating point):
//   ua = 3, ub = 4, origin 1 != origin 2
//   => do not fire
//   => origin' = 0
//   => u'^2 = ua*ua + ub*ub = 9+16 = 25
//   => u' = 5  (3-4-5 triple; the integer identity is on the square)
// If this test starts accepting u'=7, the detector is signalling everything
// and must not ship.

use eisa::core::*

fn main() -> i32 with IO, Mut, Div, Panic {
    let ua: i64 = 3
    let ub: i64 = 4
    let expected_u2: i64 = ua * ua + ub * ub

    let a = ereg_origin(1.0, 3.0, 1)
    let b = ereg_origin(2.0, 4.0, 2)
    let z = eadd(a, b)

    if z.origin != 0 {
        println("EISA_ORIGIN_INDEP_ADD FAIL origin not 0")
        return 1
    }
    if expected_u2 != 25 {
        println("EISA_ORIGIN_INDEP_ADD FAIL integer oracle 3^2+4^2")
        return 1
    }
    // Runtime lands on the integer 5 of the 3-4-5 triple. Window matches W2.
    if !(z.u > 4.999999999 && z.u < 5.000000001) {
        println("EISA_ORIGIN_INDEP_ADD FAIL u not 5")
        print("u=")
        print_f64(z.u)
        println("")
        return 1
    }
    // Mandatory: must NOT be the correlated envelope 3+4=7.
    if z.u > 6.9 {
        println("EISA_ORIGIN_INDEP_ADD FAIL false fire (u~7)")
        return 1
    }
    println("EISA_ORIGIN_INDEP_ADD OK u=5 origin=0 no-fire")
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## cook-math-import

**Topic:** modules/pure-math. **Kind:** positive.

**Source:** [tests/run-pass/native_math_via_stdlib_import.sio](../../tests/run-pass/native_math_via_stdlib_import.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "NATIVE_MATH_IMPORT_OK".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry cook-math-import in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/native_math_via_stdlib_import.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example cook-math-import -->
```sio
//@ run-pass
//@ expect-stdout: NATIVE_MATH_IMPORT_OK
//@ requires: madaros
// E221 refuses the bare `atan` because the backend cannot emit it. That refusal
// is only defensible if the alternative its help line names actually works, so
// this exercises it: import the pure-Sounio implementation and compute.
//
// It did NOT work before: `fn atan` in stdlib/math/pure.sio was not `pub`, so
// the import failed with E175 and the diagnostic pointed nowhere. Measured, and
// the reason those five gained `pub`.
use math::pure::{atan, asin, acos}

fn near(a: f64, b: f64) -> bool {
    let d = a - b
    if d < 0.0 { return 0.0 - d < 1.0e-6 }
    return d < 1.0e-6
}

fn main() -> i64 with IO, Mut, Div, Panic {
    var exit_fails: i64 = 0
    let ok = near(atan(0.5), 0.4636476090008061)
        && near(atan(2.0), 1.1071487177940904)
        && near(asin(0.5), 0.5235987755982989)
        && near(acos(0.5), 1.0471975511965979)
    if ok { print("NATIVE_MATH_IMPORT_OK\n") }
    else {
        print("NATIVE_MATH_IMPORT_FAIL\n")
        exit_fails = exit_fails + 1
        print("atan(0.5)="); print_f64(atan(0.5)); print("\n")
        print("asin(0.5)="); print_f64(asin(0.5)); print("\n")
        print("acos(0.5)="); print_f64(acos(0.5)); print("\n")
    }
    if exit_fails > 0 {
        return 1
    }
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## native-hello

**Topic:** syntax/printing-and-arithmetic. **Kind:** positive.

**Source:** [examples/native/hello.sio](../../examples/native/hello.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "Hello from self-hosted Sounio!".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry native-hello in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/hello.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example native-hello -->
```sio
// A1: Minimal native compilation smoke test.
// Tests: string literals, integer arithmetic, builtins, entry trampoline, exit code.
//
// Expected: prints "Hello from self-hosted Sounio!" then "42", exits 0.

fn main() -> i64 with IO {
    print("Hello from self-hosted Sounio!\n")
    let x = 21 + 21
    print_int(x)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## while-loop

**Topic:** control/while-and-mutation. **Kind:** positive.

**Source:** [examples/native/while_loop.sio](../../examples/native/while_loop.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "0 1 2 3 4 ".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry while-loop in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/while_loop.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example while-loop -->
```sio
// A3: While loop + mutable variables + assignment.
//
// Expected: prints "0 1 2 3 4 ", exits 0.

fn main() -> i64 with IO, Mut {
    var i: i64 = 0
    while i < 5 {
        print_int(i)
        print_char(32)
        i = i + 1
    }
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## functions

**Topic:** functions/named-functions. **Kind:** positive.

**Source:** [tests/run-pass/basic_math.sio](../../tests/run-pass/basic_math.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "84".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry functions in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/basic_math.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example functions -->
```sio
//@ run-pass
// Basic arithmetic test — native backend compatible

fn add(a: i64, b: i64) -> i64 {
    return a + b
}

fn mul(a: i64, b: i64) -> i64 {
    return a * b
}

fn main() -> i64 with IO {
    let x = add(10, 32)
    let y = mul(x, 2)
    print_int(y)
    print("\n")
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## tuple-destructure

**Topic:** types/tuple-patterns. **Kind:** positive.

**Source:** [tests/run-pass/tuple_destructure_let.sio](../../tests/run-pass/tuple_destructure_let.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "tuple destructuring test passed".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry tuple-destructure in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/tuple_destructure_let.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine tuple-destructure -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `unexpected-exit`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example tuple-destructure -->
```sio
//@ run-pass
// Test tuple destructuring in let statements

fn main() with IO {
    // Basic tuple destructuring
    let (a, b) = (1, 2)

    // Nested tuple destructuring
    let (x, (y, z)) = (10, (20, 30))

    // Wildcard pattern
    let (first, _) = (5, 10)

    // Verify all bindings worked
    if a == 1 && b == 2 && x == 10 && y == 20 && z == 30 && first == 5 {
        println("tuple destructuring test passed")
    }
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## shared-borrow

**Topic:** references/shared-borrows. **Kind:** positive.

**Source:** [tests/run-pass/borrow_reborrow.sio](../../tests/run-pass/borrow_reborrow.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "borrow reborrow: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry shared-borrow in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/borrow_reborrow.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example shared-borrow -->
```sio
//@ run-pass
//@ expect-stdout: borrow reborrow: PASS

// Multiple shared borrows of the same variable are fine
fn read_ref(r: &i64) -> i64 {
    *r
}

fn main() -> i32 with IO {
    let x: i64 = 42
    let a = &x      // shared borrow 1
    let b = &x      // shared borrow 2 — OK, both shared
    let sum = read_ref(a) + read_ref(b)
    if sum == 84 {
        print("borrow reborrow: PASS\n")
    }
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## knowledge-input

**Topic:** knowledge/input-provenance-syntax. **Kind:** positive.

**Source:** [tests/run-pass/knowledge_provenance_input.sio](../../tests/run-pass/knowledge_provenance_input.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "INPUT_KEYWORD_OK".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry knowledge-input in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/knowledge_provenance_input.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example knowledge-input -->
```sio
//@ run-pass
// Input is a provenance keyword. Both annotation delimiters must parse.
// TypeEntry does not persist provenance; the observable is that Input is
// not treated as Ident (angle form used to fail: `>` eaten as CmpGt).

fn keep_bracket(x: Knowledge[f64, Input]) -> i64 {
    0
}

fn keep_angle(x: Knowledge<f64, Input>) -> i64 {
    0
}

fn main() with IO {
    println("INPUT_KEYWORD_OK")
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## knowledge-measured

**Topic:** knowledge/measured-provenance-syntax. **Kind:** positive.

**Source:** [tests/run-pass/knowledge_provenance_measured.sio](../../tests/run-pass/knowledge_provenance_measured.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "MEASURED_KEYWORD_OK".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry knowledge-measured in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/knowledge_provenance_measured.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example knowledge-measured -->
```sio
//@ run-pass
// Positive control: Measured must still parse after Input is added.

fn keep_bracket(x: Knowledge[f64, Measured]) -> i64 {
    0
}

fn keep_angle(x: Knowledge<f64, Measured>) -> i64 {
    0
}

fn main() with IO {
    println("MEASURED_KEYWORD_OK")
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## knowledge-value

**Topic:** knowledge/explicit-value-access. **Kind:** positive.

**Source:** [tests/run-pass/knowledge_value_with_epistemic.sio](../../tests/run-pass/knowledge_value_with_epistemic.sio).

**Intended command stage:** check. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry knowledge-value in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/run-pass/knowledge_value_with_epistemic.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example knowledge-value -->
```sio
//@ run-pass
//@ description: accessing .value on Knowledge<T> is allowed when Epistemic effect is declared

fn extract_value(k: Knowledge<f64>) -> f64 with Epistemic {
    k.value
}

fn main() -> i64 with IO, Epistemic {
    let k = Knowledge { value: 9.81 }
    let v = extract_value(k)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## unit-add

**Topic:** units/same-dimension-addition. **Kind:** positive.

**Source:** [tests/run-pass/unit_same_add.sio](../../tests/run-pass/unit_same_add.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "unit add: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry unit-add in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/unit_same_add.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine unit-add -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `unexpected-exit`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example unit-add -->
```sio
//@ run-pass
//@ expect-stdout: unit add: PASS

// Adding values with the same unit is fine
fn main() -> i32 with IO, Mut, Panic, Div {
    let a: mg = 200.0
    let b: mg = 300.0
    let total = a + b
    if total == 500.0 {
        print("unit add: PASS\n")
    }
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## unit-scale

**Topic:** units/scalar-multiplication. **Kind:** positive.

**Source:** [tests/run-pass/unit_scalar_mul.sio](../../tests/run-pass/unit_scalar_mul.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "unit scalar mul: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry unit-scale in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/unit_scalar_mul.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine unit-scale -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `unexpected-exit`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example unit-scale -->
```sio
//@ run-pass
//@ expect-stdout: unit scalar mul: PASS

// Multiplying a unit value by a scalar preserves the unit
fn main() -> i32 with IO, Mut, Panic, Div {
    let dose: mg = 500.0
    let doubled = dose * 2.0
    if doubled == 1000.0 {
        print("unit scalar mul: PASS\n")
    }
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## struct-parameter

**Topic:** functions/struct-parameter. **Kind:** positive.

**Source:** [examples/native/struct_param.sio](../../examples/native/struct_param.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "12".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry struct-parameter in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/struct_param.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example struct-parameter -->
```sio
// Struct as function parameter + field access in native-v2 driver.
// Expected: prints "12", exits 0.

struct Point { x: i64, y: i64 }

fn double_sum(p: Point) -> i64 {
    (p.x + p.y) * 2
}

fn main() -> i64 with IO {
    let p = Point { x: 3, y: 3 }
    let result = double_sum(p)
    print_int(result)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## struct-return

**Topic:** functions/struct-return. **Kind:** positive.

**Source:** [examples/native/struct_return.sio](../../examples/native/struct_return.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "8".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry struct-return in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/struct_return.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example struct-return -->
```sio
// Function returning a struct (multi-value via rax:rdx) in native-v2 driver.
// Expected: prints "8", exits 0.

struct Point { x: i64, y: i64 }

fn make_point(a: i64, b: i64) -> Point {
    Point { x: a, y: b }
}

fn main() -> i64 with IO {
    let p = make_point(3, 5)
    let s = p.x + p.y
    print_int(s)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## nested-struct

**Topic:** types/nested-fields. **Kind:** positive.

**Source:** [examples/native/nested_field.sio](../../examples/native/nested_field.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "9".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry nested-struct in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run examples/native/nested_field.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example nested-struct -->
```sio
// Nested struct field access (a.b.c chains) in native-v2 driver.
// Expected: prints "9", exits 0.

struct Inner { z: i64 }
struct Outer { a: i64, inner: Inner }

fn main() -> i64 with IO {
    let inner = Inner { z: 5 }
    let outer = Outer { a: 4, inner: inner }
    let s = outer.a + outer.inner.z
    print_int(s)
    print_char(10)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## strings

**Topic:** types/string-length. **Kind:** positive.

**Source:** [tests/run-pass/string_len.sio](../../tests/run-pass/string_len.sio).

**Intended command stage:** run. **Expected exit:** 0.

**Stdout assertion:** fixed substring "string_len: PASS".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry strings in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc run tests/run-pass/string_len.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example strings -->
```sio
//@ run-pass
fn returned_string() -> string {
    "returned"
}

fn main() -> i32 with IO, Panic {
    let s = "hello"
    let empty = ""
    let long = "sounio language"
    let typed: string = "typed"
    assert(s.len() == 5)
    assert(empty.len() == 0)
    assert(long.len() == 15)
    assert(typed.len() == 5)
    assert("literal".len() == 7)
    assert(returned_string().len() == 8)
    print("string_len: PASS\n")
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## option-type

**Topic:** types/option-shorthand. **Kind:** positive.

**Source:** [tests/run-pass/option_type_shorthand.sio](../../tests/run-pass/option_type_shorthand.sio).

**Intended command stage:** check. **Expected exit:** 0.

**Stdout assertion:** none; no stdout property is established.
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry option-type in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/run-pass/option_type_shorthand.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example option-type -->
```sio
//@ run-pass
//@ description: Option type shorthand ?T is accepted in type positions

struct ShorthandHolder {
    value: ?i64,
}

fn accepts_shorthand(v: ?i64) -> i64 with IO {
    0
}

fn main() -> i64 {
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-effect-missing

**Topic:** effects/missing-declaration. **Kind:** negative.

**Source:** [tests/compile-fail/effect_missing.sio](../../tests/compile-fail/effect_missing.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "effect not declared".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-effect-missing in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/effect_missing.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-effect-missing -->
```sio
//@ compile-fail
//@ description: SELFHOST_PASS
//@ error-pattern: not declared in function signature

fn read_file(path: string) -> string with IO {
    // implementation
    ""
}

// Missing IO effect declaration in non-main function
fn load_data() -> string {
    let content = read_file("test.txt")  // ERROR: IO effect not declared
    content
}

fn main() with IO {
    let s = load_data()
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-observe-io

**Topic:** knowledge/observe-boundary. **Kind:** negative.

**Source:** [tests/compile-fail/observe_io_boundary.sio](../../tests/compile-fail/observe_io_boundary.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "cannot print Unobserved<T>".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-observe-io in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/observe_io_boundary.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-observe-io -->
```sio
//@ compile-fail
//@ error-pattern: cannot print Unobserved<T>
fn leak(x: Unobserved<f64>) with IO {
    // Should require 'with Observe' — passing Unobserved<T> to IO function
    println(x)
}
fn main() -> i32 with IO { 0 }
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-borrow-conflict

**Topic:** references/exclusive-shared-conflict. **Kind:** negative.

**Source:** [tests/compile-fail/borrow_call_conflict_explicit.sio](../../tests/compile-fail/borrow_call_conflict_explicit.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "cannot borrow exclusively while other borrows are active".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-borrow-conflict in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/borrow_call_conflict_explicit.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-borrow-conflict -->
```sio
//@ compile-fail
//@ error-pattern: cannot borrow as mutable because it is also borrowed as shared

fn read_exclusive(r: &!i64) -> i64 with Mut {
    *r
}

fn main() -> i32 with Mut {
    var x: i64 = 42
    let hold = &x

    // A live shared borrow conflicts with explicit exclusive borrow at call boundary.
    let y = read_exclusive(&!x)
    if y >= 0 {
        return 0
    }
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-knowledge-unwrap

**Topic:** knowledge/no-silent-unwrapping. **Kind:** negative.

**Source:** [tests/compile-fail/knowledge_no_silent_unwrap.sio](../../tests/compile-fail/knowledge_no_silent_unwrap.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "found Knowledge<f64>".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-knowledge-unwrap in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/knowledge_no_silent_unwrap.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-knowledge-unwrap -->
```sio
//@ compile-fail
//@ error-pattern: found Knowledge
// Knowledge<T> must not implicitly coerce to T.

fn main() {
    let k = Knowledge { value: 1.0 }
    let x: f64 = k
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-unknown-effect

**Topic:** effects/unknown-name. **Kind:** negative.

**Source:** [tests/compile-fail/effect_unknown_name_refused.sio](../../tests/compile-fail/effect_unknown_name_refused.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "unknown effect 'Zorblex'".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-unknown-effect in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/effect_unknown_name_refused.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-unknown-effect -->
```sio
//@ typecheck-fail
//@ requires: madaros
//@ error-pattern: unknown effect 'Zorblex': not in the closed effect list
// E246 face 0: an effect name in no list (closed 30, user registry, held
// transition list) must be refused, not silently dropped. Before E246 this
// program checked rc=0 with zero diagnostics (measured 2026-08-25, shipped
// bin/madaros-linux-x86_64) - the name participated in nothing.

fn g() -> i32 with Zorblex { 0 }
fn main() -> i32 with IO { 0 }
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-unit-add

**Topic:** units/dimension-mismatch. **Kind:** negative.

**Source:** [tests/compile-fail/unit_mismatch_add.sio](../../tests/compile-fail/unit_mismatch_add.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "unit mismatch".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-unit-add in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/unit_mismatch_add.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-unit-add -->
```sio
//@ compile-fail
//@ description: SELFHOST_PASS
//@ error-pattern: unit mismatch

// Cannot add values with different units
fn main() -> i32 with IO, Mut, Panic, Div {
    let mass: mg = 500.0
    let length: m = 10.0
    let bad = mass + length    // ERROR: mg + m is dimensionally inconsistent
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-unit-return

**Topic:** units/return-dimension. **Kind:** negative.

**Source:** [tests/compile-fail/unit_return_wrong_unit.sio](../../tests/compile-fail/unit_return_wrong_unit.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "unit mismatch in return value".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-unit-return in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/unit_return_wrong_unit.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine negative-unit-return -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `missing-stdout-marker`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example negative-unit-return -->
```sio
//@ compile-fail
//@ error-pattern: unit mismatch in return value
// Returning one unit from a function declared to return another. Before the
// return position was checked, this compiled and handed the caller a molal brand
// on a molar value -- a call result takes the unit of its DECLARED return type,
// so the wrong brand propagated onward with nothing to catch it.

unit molal = mol / kg;
unit molar = mol / L;

fn lie(x: molar) -> molal with Mut {
    return x
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let m: molar = 2.0
    let r = lie(m)
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-unit-argument

**Topic:** units/argument-dimension. **Kind:** negative.

**Source:** [tests/compile-fail/unit_literal_suffix_reject_length_as_mass.sio](../../tests/compile-fail/unit_literal_suffix_reject_length_as_mass.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "unit mismatch in call argument".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-unit-argument in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/unit_literal_suffix_reject_length_as_mass.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:quarantine negative-unit-argument -->
> **Quarantined: not a working or confirmed-refusal recipe on the documented
> profile.** The recorded attempt exited 1; the assertion result
> was `missing-stdout-marker`. Do not reinterpret the failure as a pass
> or edit the expected result to conceal it. Use a verified nearby example;
> see the [case receipt](reviews/example-execution.md) and [raw results](example-results.json).

<!-- llm:example negative-unit-argument -->
```sio
//@ compile-fail
//@ error-pattern: unit mismatch in call argument
//@ description: numeric literal unit suffix must participate in call-boundary dimensional checks

fn takes_mg(dose: mg) -> f64 with Mut {
    dose
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let distance = 10.0<m>
    let bad = takes_mg(distance)
    return 0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-mut-effect

**Topic:** effects/reference-store-mut. **Kind:** negative.

**Source:** [tests/compile-fail/effect_mut_generated_at_excl_ref_store.sio](../../tests/compile-fail/effect_mut_generated_at_excl_ref_store.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "E035".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-mut-effect in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/effect_mut_generated_at_excl_ref_store.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-mut-effect -->
```sio
//@ typecheck-fail
//@ requires: madaros
//@ error-pattern: error[E035

// Regression pin for #1488 — the effect system had no leaf-level source of Mut.
//
// Madaros propagated declared effects correctly: a caller of a `with Mut` callee
// was required to declare it too. What was missing was anything that INTRODUCED
// the effect at the point where mutation actually happens. So this function
// type-checked with no `with Mut`, and because nothing below it ever asserted the
// effect, no caller above it could be required to carry it either. `with Mut` was
// true only where a human had written it — the guarantee was vacuous exactly at
// the leaves, which is where it has to hold.
//
// Measured before the fix: Madaros accepted this (exit 0). lean_single rejected
// it. A dead-code explanation was ruled out separately: Madaros already rejected
// a caller of a callee that DECLARES `with Mut`, even when that caller is
// unreachable from main, so propagation was working and only generation was
// absent.
//
// `requires: madaros` because the fix lives in the modular checker; lean_single
// has always rejected this shape and the stage2 suite would pass it for a
// different reason.

fn poke(x: &!i64) {
    *x = 42
}

fn main() with IO {
    var v: i64 = 0
    poke(&!v)
    println("unreachable: poke must be rejected for not declaring Mut")
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

## negative-builtin-arity

**Topic:** functions/builtin-arity. **Kind:** negative.

**Source:** [tests/compile-fail/native_builtin_arity_reject.sio](../../tests/compile-fail/native_builtin_arity_reject.sio).

**Intended command stage:** check-fail. **Expected exit:** nonzero.

**Intentionally invalid:** expected to refuse for the stated cause. Do not copy
this as an accepted program. A crash or unrelated error does not satisfy it.

**Stdout assertion:** fixed substring "E010".
**Stderr assertion:** none; stderr need not be empty.

**Verification:** consult entry negative-builtin-arity in the manifest and the execution receipt.

Run from the repository root with the configured standard library:

```sh
./bin/souc check tests/compile-fail/native_builtin_arity_reject.sio
```

The command is expected to return nonzero for a negative entry or a program
whose declared success exit is nonzero. Inspect the contract, not a blanket
exit-zero rule. Check-only entries do not establish runtime behavior.

<!-- llm:example negative-builtin-arity -->
```sio
//@ compile-fail
//@ requires: madaros
//@ error-pattern: E010
//@ description: runtime builtins reject argument counts outside their checker-side ABI

// str_slice has exactly two source spellings: (s, start) and (s, start, end).
// A fourth argument used to survive checking and reach lowering, where it could
// be silently discarded by the native builtin emitter.
fn main() -> i64 with IO {
    let s = "0123456789"
    let _slice = str_slice(s, 2, 5, 7)
    0
}
```

**Adapt safely:** preserve types, effects, imports, resource obligations, and
scientific information required by the task. The manifest’s assertion is the
existing bounded observation contract, not a broader proof.

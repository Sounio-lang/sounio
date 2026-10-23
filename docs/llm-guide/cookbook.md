<!-- docs:meta
topic_id: repo.docs.llm-guide.cookbook
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.cookbook
-->

# Progressive Sounio cookbook

Work through these ten lessons in order, starting from a passed recipe for the
pinned profile. Keep its imports, data representation, and effects together.
Quarantined failures are retained separately as diagnostic labs. Every
Sounio fence below is the **entire unmodified repository file** identified above
it. Its marker binds it to the [curated example manifest](example-manifest.json). These are
source-inspected files from baseline
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. Their recorded outcomes are in
[the actual results](example-results.json) and
[execution receipt](reviews/example-execution.md), not inferred from source annotations.

The expected contracts come from fixture annotations, existing branches, and
existing assertions. No new scientific calculation, tolerance, or numerical
oracle was constructed for this cookbook. Historical source comments are copied
verbatim; a comment about a past compiler defect does not establish present
behavior or a language restriction.

## Before the first lesson

Use the repository root as the working directory. The commands below use
`./bin/souc`, the public resolver-backed launcher. The intended selection is
default Madaros with this checkout's library sources. The curated manifest selects
the baseline committed Linux x86-64 Madaros profile; a launcher invocation belongs
to that profile only when its resolved artifact matches. Actual engine, resolved
artifact, target, host, and library root must be recorded by whoever runs it;
a filename or launcher spelling does not prove the selected profile.

**Use each example's actual recorded stage and outcome.** Slurm job `14651`
executed the baseline committed Linux x86-64 Madaros raw artifact with SHA-256
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
The [per-case results](example-results.json) identify commands, exits, stdout,
stderr, and reasons. The [manifest](example-manifest.json) retains the intended
contracts; receipt failures do not change them. A passed `run` row records that
command, not a separately executed public `check` command. A `check-passed`
provider/consumer row supplies no runtime observation. Displayed commands are
reproduction instructions, not claims that every displayed command was run.

The failed `cook-closure`, `cook-generic-fn`, and `cook-generic-pair` examples
are quarantined in the [diagnostic lab](#diagnostic-lab-quarantined-profile-failures).
Use passed `cook-function-ref` for callbacks and passed `generics` for the
working generic-function lesson. Other hosts, engines, changed source, and raw
artifacts require their own receipts. Follow
[conformance and profile attribution](../spec/2026-10/09-conformance-diagnostics-and-compatibility.md)
and the [stdlib discovery workflow](stdlib-index.md) when selecting a runner.
Perform lightweight check/run only in an authorized environment with an
identified, host-compatible profile, following the root
[first-success preflight](../../llms.txt#first-success). Session or repository
policy may require a remote route; this guide does not authorize compiler tests
on the current Mac or any unapproved host. Full stress validation uses the
project's remote Compiler Foundry or Slurm route where policy requires it.
A public learner needs no project-specific cluster for an otherwise authorized
lightweight check/run.

Three practical habits make adaptation easier:

1. Keep the original file intact until its own expected result is reproduced.
2. Change one concept, then check and run again on the same identified profile.
3. When a fixture, language rule, and engine disagree, retain the fixture and
   report the difference. Do not rewrite the language around an acceptance gap.

Local variable mutation, mutation through an exclusive reference, and effects
on calls are distinct questions. Preserve each copied signature; derive any
changed effect requirement from the
[effects chapter](../spec/2026-10/03-effects-handlers-and-concurrency.md)
and selected implementation. Neither every `var` nor every loop is a universal
instruction to add `Mut`.

## 1. Enter the program and advance mutable state

Read `fn main`, its return type, and its effect list before the body. A printed
message and a process result are separate observations.

### 1a. A complete entry point

**Problem.** Print a greeting and terminate successfully.

Source: [`examples/hello.sio`](../../examples/hello.sio). Example ID: `cook-hello`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-hello`.

<!-- llm:example cook-hello -->
```sio
// First Sounio program

fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

**Intended observable contract.** The file requests the line `Hello, Sounio` and explicitly returns `0`. The bounded conformance manifest's hello run checks exit success, not the complete greeting.

```bash
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

**Adaptation.** Change the greeting first while keeping the entry point and `IO` declaration.

**Common errors.** Borrowing Rust macro syntax such as `println!` changes the syntax; use the actual call in the file.

### 1b. A terminating while loop

**Problem.** Walk through a bounded counter, printing each step.

Source: [`examples/native/while_loop.sio`](../../examples/native/while_loop.sio). Example ID: `cook-while`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-while`.

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

**Intended observable contract.** The source comment expects `0 1 2 3 4 ` followed by a newline and exit `0`; the trailing space is part of that expectation.

```bash
./bin/souc check examples/native/while_loop.sio
./bin/souc run examples/native/while_loop.sio
```

**Adaptation.** Keep initialization, continuation condition, and increment consistent; preserve `var` for a rebound binding.

**Common errors.** Removing the increment leaves the continuation condition unchanged; changing a mutable binding to `let` changes the assignment contract.

Syntax and statement reference:
[lexical structure and core syntax](../spec/2026-10/01-lexical-syntax-and-modules.md).

## 2. Store fixed data, iterate, and handle optional progress

A fixed array's type carries its length. The next two programs show element
iteration, range iteration, indexed writes, and `while let` without substituting
a heap vector or an API from another language.

### 2a. Iterate the logical array contents

**Problem.** Populate a fixed array, then reduce its elements while counting visits.

Source: [`tests/run-pass/for_in_array.sio`](../../tests/run-pass/for_in_array.sio). Example ID: `cook-for-array`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-for-array`.

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

**Intended observable contract.** Exit `0` requires the fixture's sum and logical element-count checks; exits `1` and `2` identify those separate failures. No stdout marker is required.

```bash
./bin/souc check tests/run-pass/for_in_array.sio
./bin/souc run tests/run-pass/for_in_array.sio
```

**Adaptation.** Adapt the declared array length, initialization writes, and assertions together; keep failure codes meaningful.

**Common errors.** Using an allocation bucket's capacity as a logical length changes the contract. Returning the reduction as a process code also loses information, as the copied source explains.

### 2b. Ranges, indexed writes, and while-let

**Problem.** Exercise bounded ranges and consume an optional value until `None`.

Source: [`tests/run-pass/while_for_struct_patterns.sio`](../../tests/run-pass/while_for_struct_patterns.sio). Example ID: `cook-range-option`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-range-option`.

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

**Intended observable contract.** The existing branches check the two range loops, a terminating `Some` sequence, immediate `None`, and array writes. Intended success prints `while_for_struct_patterns: ALL PASS` and returns zero through `failed`.

```bash
./bin/souc check tests/run-pass/while_for_struct_patterns.sio
./bin/souc run tests/run-pass/while_for_struct_patterns.sio
```

**Adaptation.** Retain the distinction between a range loop's upper bound and an array index; follow the fixture's explicit `usize` conversion at writes.

**Common errors.** `0..n` and `0..=n` have different endpoints. A `while let` body must eventually change the matched state when termination is intended.

Representation and binding reference:
[types and ownership](../spec/2026-10/02-types-generics-and-ownership.md).
The fixture's comment “new feature” is historical wording, not a current release claim.

## 3. Give data names, return tuples, and preserve element types

Use a struct for named fields and a tuple for ordered components. Keep the
declared component types through helper parameters, returns, and field access.

### 3a. Construct and read a struct

**Problem.** Combine named fields without a library dependency.

Source: [`examples/native/struct_basic.sio`](../../examples/native/struct_basic.sio). Example ID: `cook-struct`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-struct`.

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

**Intended observable contract.** The source comment expects the printed line `7` and exit `0`; the existing conformance case only requires stdout to contain `7`.

```bash
./bin/souc check examples/native/struct_basic.sio
./bin/souc run examples/native/struct_basic.sio
```

**Adaptation.** Add fields to both the declaration and each constructor before changing the computation.

**Common errors.** A field access is not a method call. Do not copy a constructor or printing API from another language.

### 3b. Tuples in function signatures

**Problem.** Pass a pair to a helper and return a pair from another helper.

Source: [`tests/run-pass/tuple_signature_types.sio`](../../tests/run-pass/tuple_signature_types.sio). Example ID: `cook-tuple-signature`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-tuple-signature`.

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

**Intended observable contract.** Both existing comparisons must hold for the final result to be `0`; otherwise the program returns `1`. It deliberately has no stdout contract.

```bash
./bin/souc check tests/run-pass/tuple_signature_types.sio
./bin/souc run tests/run-pass/tuple_signature_types.sio
```

**Adaptation.** Keep tuple arity and element order synchronized between the parameter, return type, construction, and `.0`/`.1` accesses.

**Common errors.** A tuple value `(a, b)` and a tuple type `(i64, i64)` occupy different syntactic positions; omitting an element changes the type.

### 3c. Heterogeneous tuple slots

**Problem.** Keep integer and floating components distinct after tuple projection.

Source: [`tests/run-pass/tuple_mixed_f64_access.sio`](../../tests/run-pass/tuple_mixed_f64_access.sio). Example ID: `cook-tuple`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-tuple`.

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

**Intended observable contract.** The integer path is asserted; the floating path prints `tuple mixed f64 access passed` only inside the existing interval check. Require that marker as well as successful termination: falling outside the interval does not itself return a failure code in this file.

```bash
./bin/souc check tests/run-pass/tuple_mixed_f64_access.sio
./bin/souc run tests/run-pass/tuple_mixed_f64_access.sio
```

**Adaptation.** Preserve the typed slot bindings when changing a mixed tuple. Reuse the fixture's existing checks only within their stated scope.

**Common errors.** Treating a successful exit alone as proof of the floating branch misses the fixture's conditional marker. A tuple storage regression is not a rule that mixed tuples are forbidden.

See [type and representation rules](../spec/2026-10/02-types-generics-and-ownership.md)
and [execution and ABI boundaries](../spec/2026-10/06-execution-memory-and-targets.md).

## 4. Borrow shared data and make caller-visible changes

`&T` and `&!T` express different access contracts. Read the callee's signature
and the call-site borrow together. These fixtures cover specific lifetime and
storage shapes; they are not a complete ownership conformance suite.

### 4a. Consecutive explicit call-site borrows

**Problem.** Call exclusive, shared, and exclusive readers in sequence.

Source: [`tests/run-pass/borrow_call_explicit_release.sio`](../../tests/run-pass/borrow_call_explicit_release.sio). Example ID: `cook-borrow-release`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-borrow-release`.

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

**Intended observable contract.** The fixture is intended to accept and return `0`. Its runtime comparison is weak: the useful borrow-release evidence also depends on the intended checker path accepting this call sequence.

```bash
./bin/souc check tests/run-pass/borrow_call_explicit_release.sio
./bin/souc run tests/run-pass/borrow_call_explicit_release.sio
```

**Adaptation.** Start from the explicit borrow spellings and preserve each reference parameter's type before adding a new call.

**Common errors.** Using `&mut` changes the Sounio spelling. Keeping an exclusive reference live across another conflicting borrow is a different lifetime problem from consecutive call-site borrows.

### 4b. Mutate an array through a reference

**Problem.** Write through three existing representations and observe the result in the caller.

Source: [`tests/run-pass/array_mut_ref_bare.sio`](../../tests/run-pass/array_mut_ref_bare.sio). Example ID: `cook-array-ref`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-array-ref`.

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

**Intended observable contract.** Each pattern has its own failure branch; intended success prints `array_mut_ref_bare: PASS` and returns `0`. This tests bare indexing, explicit dereference, and a struct-wrapped array.

```bash
./bin/souc check tests/run-pass/array_mut_ref_bare.sio
./bin/souc run tests/run-pass/array_mut_ref_bare.sio
```

**Adaptation.** Keep the parameter length, caller array length, initialization, and element checks aligned. Preserve the three patterns when investigating a storage regression.

**Common errors.** A past backend store bug must not become a language rule requiring a wrapper struct. An accepting check cannot establish that caller-visible stores executed.

Reference: [ownership](../spec/2026-10/02-types-generics-and-ownership.md).
The assembly/store history in the copied comments is retained as fixture context,
not freshly verified compiler history.

## 5. Select behavior with enums and match

### 5a. Match a named enum variant

**Problem.** Dispatch on `Color::Green` using variant patterns.

Source: [`tests/selfhost/native_runtime/enum_match_green_42.sio`](../../tests/selfhost/native_runtime/enum_match_green_42.sio). Example ID: `cook-enum`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `42`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-enum`.

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

**Intended observable contract.** The selected branch explicitly returns `42`; this is an intentional nonzero result, not a conventional zero-success test. The other branch returns and trailing `99` make dispatch distinguishable. No stdout is specified.

```bash
./bin/souc check tests/selfhost/native_runtime/enum_match_green_42.sio
./bin/souc run tests/selfhost/native_runtime/enum_match_green_42.sio
```

**Adaptation.** Add named variants and corresponding arms together; preserve this fixture's result when reproducing it, then design your own application's success convention.

**Common errors.** Replacing a typed enum with an integer merely because another legacy fixture takes `i64` promotes an implementation accommodation into design. A harness assuming all valid programs exit zero will misclassify this example.

References: [patterns and syntax](../spec/2026-10/01-lexical-syntax-and-modules.md)
and [type boundaries](../spec/2026-10/02-types-generics-and-ownership.md).
This source lives under `tests/selfhost/native_runtime/`; its location identifies
its original test surface, not a claim that every launcher profile runs it.

## 6. Pass named functions as values

Function parameters carry an argument/result signature. Effects on a callback
are another part of its contract; do not extrapolate these examples to arbitrary
capturing, escaping, or effectful closures.

### 6a. Named functions as values

**Problem.** Pass named operations into helpers, bind a function, and return a selected operation.

Source: [`examples/native/hof_fn_ref_call.sio`](../../examples/native/hof_fn_ref_call.sio). Example ID: `cook-function-ref`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-function-ref`.

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

**Intended observable contract.** All five existing checks must pass; otherwise distinct nonzero exits identify the failing path. Intended success prints `42` and returns `0`.

```bash
./bin/souc check examples/native/hof_fn_ref_call.sio
./bin/souc run examples/native/hof_fn_ref_call.sio
```

**Adaptation.** Match a replacement callback's arity and parameter/result types to `apply` or `apply2`; retain a separate check for each dispatch path.

**Common errors.** An identifier naming a function and a call of that function are different values. Replacing the function argument with its result breaks the callback signature.

For a working callback recipe on the pinned profile, use `cook-function-ref`
above. The noncapturing closure fixture failed on that same profile and is
retained only in the [diagnostic lab](#diagnostic-lab-quarantined-profile-failures).
This observed refusal does not establish a universal ban on closures.

Reference: [functions, generics, and ownership](../spec/2026-10/02-types-generics-and-ownership.md)
and [effect contracts](../spec/2026-10/03-effects-handlers-and-concurrency.md).

## 7. Instantiate generic functions and inspect record boundaries

Use the passed `generics` fixture for the working generic-function path on the
pinned profile. It uses explicit `::<...>` type arguments with two generic
parameters. The retained native-runtime identity and record fixtures failed
through this profile's launcher; they are in the
[diagnostic lab](#diagnostic-lab-quarantined-profile-failures), with their original
intended results intact. Those failures do not establish that generics or
generic records are universally unavailable.

### 7a. A working two-parameter generic function recipe

**Problem.** Select the first or second argument through generic helpers while
checking each returned value.

Source: [`tests/run-pass/generics_multi_param.sio`](../../tests/run-pass/generics_multi_param.sio).
Existing manifest example ID: `generics`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `generics`.

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

**Intended observable contract.** Exit `0` and the manifest's required
`first_of_two: PASS` substring. The existing file also checks the second helper;
the recorded stdout includes `second_of_two: PASS`. That additional observed
line does not change the manifest's declared assertion scope.

```bash
./bin/souc check tests/run-pass/generics_multi_param.sio
./bin/souc run tests/run-pass/generics_multi_param.sio
```

**Adaptation.** Keep explicit type arguments, parameter order, and return type
aligned while adapting one helper at a time. This fixture's local reassignments
and declared effects must be interpreted on the recorded path; do not invent a
universal `Mut` requirement from another engine's behavior.

**Common errors.** Replacing the observed `::<...>` spelling with another
fixture's spelling changes the parser/inference path. A successful two-parameter
function does not by itself establish every generic record layout or operation.

Reference:
[generic syntax and ownership](../spec/2026-10/02-types-generics-and-ownership.md).

## 8. Import a provider and handle a concrete result

A complete import example consists of its consumer **and** its provider. Keep
the provider beside the consumer, or reproduce the exact resolver layout. For
stdlib imports, keep the library root explicit in the validation receipt.

### 8a. The complete provider module

**Problem.** Expose the function needed by the next file.

Source: [`tests/conformance/import_public_a.sio`](../../tests/conformance/import_public_a.sio). Example ID: `cook-import-provider`.

**Recorded result — pinned profile, job `14651`.** `check-passed`, observed exit `0`; reason `ok`. Only the recorded `check` stage is established.
See [actual case logs](example-results.json) under `cook-import-provider`.

<!-- llm:example cook-import-provider -->
```sio
//@ auxiliary
// Provider module for the bounded conformance import check.

pub fn imported_add(a: i64, b: i64) -> i64 {
    a + b
}
```

**Intended observable contract.** This auxiliary file declares a public function and has no `main`. Its proposed command is check-only; it is not an executable demonstration.

```bash
./bin/souc check tests/conformance/import_public_a.sio
```

**Adaptation.** Preserve `pub` at the exported declaration and keep the sibling filename consistent with the consumer's root import.

**Common errors.** Running a library module as though it were a main program confuses checking a provider with executing a consumer.

### 8b. The complete consumer

**Problem.** Selectively import the provider's public function.

Source: [`tests/conformance/import_public.sio`](../../tests/conformance/import_public.sio). Example ID: `cook-import-main`.

**Recorded result — pinned profile, job `14651`.** `check-passed`, observed exit `0`; reason `ok`. Only the recorded `check` stage is established.
See [actual case logs](example-results.json) under `cook-import-main`.

<!-- llm:example cook-import-main -->
```sio
//@ check-pass
// Positive import visibility check for the serious-language conformance spine.

use import_public_a::{imported_add}

fn main() -> i64 {
    imported_add(20, 22)
}
```

**Intended observable contract.** The fixture is marked `check-pass`, so the command below only asks for acceptance. Its `main` returns the imported call's result; a separately chosen run would intentionally expect `42`, not zero.

```bash
./bin/souc check tests/conformance/import_public.sio
```

**Adaptation.** Copy both sibling files as a pair; then change the exported declaration and selected import together.

**Common errors.** Copying only this main file leaves its dependency missing. A function found by text search is not necessarily publicly importable.

### 8c. Handle the actual core::result API

**Problem.** Use the concrete integer/float result constructors, tag predicates, payload access, and fallback.

Source: [`tests/stdlib/core/test_result_stdlib.sio`](../../tests/stdlib/core/test_result_stdlib.sio). Example ID: `cook-result`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-result`.

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

**Intended observable contract.** The file's failure branches identify incorrect behavior; intended success prints `RESULT_STDLIB_OK` and returns `0`. Its existing floating comparison is preserved exactly, not offered as a general error tolerance.

```bash
./bin/souc check tests/stdlib/core/test_result_stdlib.sio
./bin/souc run tests/stdlib/core/test_result_stdlib.sio
```

**Adaptation.** Use `int_result_is_ok` or `int_result_is_err` before consuming an integer result. Keep `IntResult` and `FloatResult` paired with their own accessors.

**Common errors.** These records have `tag`, `ok_val`, and `err_val`; they are not an invented `{ value, ok }` record or automatically the language's generic `Result<T,E>`. In this source, `unwrap` reads a payload; it does not itself prove success or panic on error.

References: [module syntax](../spec/2026-10/01-lexical-syntax-and-modules.md),
[library identity](../spec/2026-10/07-library-packages-and-tools.md), and the
[curated result signatures](stdlib-index.md#concrete-results-core-result).

## 9. Use the vector representation the source actually exports

### 9a. A fixed-capacity VecF64 through free functions

**Problem.** Import the published vector API, mutate through exclusive references, and inspect through shared references.

Source: [`tests/stdlib/collections/test_vec_stdlib.sio`](../../tests/stdlib/collections/test_vec_stdlib.sio). Example ID: `cook-vec`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-vec`.

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

**Intended observable contract.** The existing checks cover length, aggregate operations, access, update, pop, and reverse. Intended success prints `VEC_STDLIB_OK` and returns `0`. Existing comparison windows are copied without extending their numerical claim.

```bash
./bin/souc check tests/stdlib/collections/test_vec_stdlib.sio
./bin/souc run tests/stdlib/collections/test_vec_stdlib.sio
```

**Adaptation.** Use this free-function API as a unit: constructor, `&!` mutation, and `&` reads. Inspect capacity and out-of-range behavior before changing the workload.

**Common errors.** This `VecF64` stores `[f64; 256]`; it is not an unbounded heap vector. Its source header's historical `1024` description disagrees with the actual declaration and capacity constant. Do not substitute `Vec<T>`, `HeapVec<T>`, or `Seq<T>` by name resemblance.

Reference: [container representations](stdlib-index.md#containers-choose-a-representation)
and [library contracts](../spec/2026-10/07-library-packages-and-tools.md).
The source, not the introductory header, determines the current capacity.

## 10. Cross a scientific-library boundary without changing its meaning

These examples teach existing interfaces and observable fixture contracts. They
do not establish numerical accuracy for new inputs, clinical validity, a theorem,
or agreement between independent implementations. Their scientific assertions
and tolerances are inherited from the exact copied files.

### 10a. The library Epistemic record

**Problem.** Use the existing Madaros-targeted free-function import example.

Source: [`tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`](../../tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio). Example ID: `cook-epistemic-library`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-epistemic-library`.

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

**Intended observable contract.** The fixture's existing branches check constructors, accessors, operations, and a gate, then request `KNOWLEDGE_MADAROS_IMPORT_E2E_OK` with exit `0`. These are its intended assertions, not fresh numerical results.

```bash
./bin/souc check tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio
./bin/souc run tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio
```

**Adaptation.** Keep the selected `ep_*` API and borrowed parameters together. Inspect covariance preconditions before adapting combinations of dependent measurements.

**Common errors.** `epistemic::knowledge::Epistemic` is a concrete library record, not compiler `Knowledge<T>`. This fixture explicitly disclaims that builtin claim. A confidence gate is not domain validation.

### 10b. EISA origin handling

**Problem.** Retain distinct measurement origins while exercising the existing addition fixture.

Source: [`tests/run-pass/eisa_origin_independent_add.sio`](../../tests/run-pass/eisa_origin_independent_add.sio). Example ID: `cook-eisa`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-eisa`.

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

**Intended observable contract.** The source's existing origin check, integer identity check, inherited output window, and false-fire guard lead to `EISA_ORIGIN_INDEP_ADD OK u=5 origin=0 no-fire` and exit `0`. This page did not recalculate or execute those checks.

```bash
./bin/souc check tests/run-pass/eisa_origin_independent_add.sio
./bin/souc run tests/run-pass/eisa_origin_independent_add.sio
```

**Adaptation.** Keep `val`, `err`, `u`, and `origin` distinct. Read the selected origin constructor and operation before changing an origin or using a different EISA record.

**Common errors.** Treating computational correction `err` as measurement uncertainty `u` discards information. This distinct-origin example does not establish a general covariance model or correctness of every EISA operation.

### 10c. Import a concrete mathematical implementation

**Problem.** Use the fixture's public `math::pure` functions rather than guessing a builtin API.

Source: [`tests/run-pass/native_math_via_stdlib_import.sio`](../../tests/run-pass/native_math_via_stdlib_import.sio). Example ID: `cook-math-import`.

**Recorded result — pinned profile, job `14651`.** `run-passed`, observed exit `0`; reason `ok`. Only the recorded `run` stage is established.
See [actual case logs](example-results.json) under `cook-math-import`.

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

**Intended observable contract.** The source requires Madaros and expects `NATIVE_MATH_IMPORT_OK`; failed inherited checks request a failure marker and exit `1`. The existing constants and comparison window are reproduced verbatim.

```bash
./bin/souc check tests/run-pass/native_math_via_stdlib_import.sio
./bin/souc run tests/run-pass/native_math_via_stdlib_import.sio
```

**Adaptation.** Inspect each public function's domain handling and effect signature; find a fixture for the desired domain before adapting inputs.

**Common errors.** A function's name, a passing parser, or one comparison window is not a global accuracy contract. A native intrinsic and a library implementation may have different reachability and behavior.

Read [the scientific API boundaries](stdlib-index.md#scientific-information-keep-the-representations-separate),
[Knowledge and provenance](../spec/2026-10/04-knowledge-uncertainty-and-provenance.md),
and [numerics and algebra](../spec/2026-10/05-numerics-units-and-algebra.md)
before choosing a new representation.

## Diagnostic lab: quarantined profile failures

These three source files remain complete and unchanged. Their source annotations
describe intended behavior, while the actual pinned-profile receipt records
failure before a successful program execution. They are not starting points for
a working recipe on this artifact. Preserve the diagnostic, source, and original
expected result; do not lower the expectation to make a failing case pass.

### Closure callback: recorded type-check failure

**Diagnostic problem.** Supply a lambda directly to a higher-order function.

Source: [`tests/run-pass/closure_hof.sio`](../../tests/run-pass/closure_hof.sio). Example ID: `cook-closure`.

**Not a working recipe for the pinned profile.** Job `14651` recorded exit `1` instead of intended `0`, with `E004` operator/type diagnostics and `E009` argument-type mismatch; the log includes `expected ()` and `expected fn#6`.
This is retained as a diagnostic lab; preserve the original source and intended
contract when investigating it.

**Recorded result — pinned profile, job `14651`.** `unexpected-failure`, observed exit `1` versus intended `0`; reason `unexpected-exit`. Quarantined below.
See [actual case logs](example-results.json) under `cook-closure`.

<!-- llm:quarantine cook-closure -->
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

**Intended observable contract.** The existing equality check chooses `PASS` and exit `0`, or `FAIL` and exit `1`. The annotation expects `PASS`.

```bash
./bin/souc check tests/run-pass/closure_hof.sio
./bin/souc run tests/run-pass/closure_hof.sio
```

**Working alternative.** Use the passed [`cook-function-ref`](#6-pass-named-functions-as-values) recipe for a callback needed on this profile. Keep this failing closure file as a separate reproducer; a future repaired or different profile needs its own receipt.

**Common errors.** Inferring general closure allocation or escape support from a noncapturing example exceeds its witness.

### Explicit identity instantiation: recorded undeclared-variable diagnostics

**Diagnostic problem.** Instantiate an identity helper for `i64`.

Source: [`tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio`](../../tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio). Example ID: `cook-generic-fn`.

**Not a working recipe for the pinned profile.** Job `14651` recorded exit `1` instead of intended `42`; the compiler reported `E137` use-of-undeclared-variable diagnostics.
This is retained as a diagnostic lab; preserve the original source and intended
contract when investigating it.

**Recorded result — pinned profile, job `14651`.** `unexpected-failure`, observed exit `1` versus intended `42`; reason `unexpected-exit`. Quarantined below.
See [actual case logs](example-results.json) under `cook-generic-fn`.

<!-- llm:quarantine cook-generic-fn -->
<!-- llm:example cook-generic-fn -->
```sio
fn id<T>(x: T) -> T {
    x
}

fn main() -> i64 {
    id<i64>(42)
}
```

**Intended observable contract.** The complete program returns the fixture's `42`, with no stdout contract. A runner must expect that result rather than invent exit `0`.

```bash
./bin/souc check tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio
./bin/souc run tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio
```

**Working alternative.** Use the passed [`generics`](#7-instantiate-generic-functions-and-inspect-record-boundaries) recipe's existing explicit type-argument spelling. Preserve this native-runtime fixture when diagnosing the differing parse/resolution path; do not silently replace its expected `42`.

**Common errors.** Removing the concrete argument may change the inference path. Generalizing from identity to arithmetic on arbitrary `T` adds a separate constraint problem.

### Explicit generic record construction: recorded parse failure

**Diagnostic problem.** Construct a `Pair<i64>` and access its two fields.

Source: [`tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio`](../../tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio). Example ID: `cook-generic-pair`.

**Not a working recipe for the pinned profile.** Job `14651` recorded exit `1` instead of intended `42`; the log reports `module failed to parse` and an incomplete AST closure.
This is retained as a diagnostic lab; preserve the original source and intended
contract when investigating it.

**Recorded result — pinned profile, job `14651`.** `unexpected-failure`, observed exit `1` versus intended `42`; reason `unexpected-exit`. Quarantined below.
See [actual case logs](example-results.json) under `cook-generic-pair`.

<!-- llm:quarantine cook-generic-pair -->
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

**Intended observable contract.** The fixture returns its field-sum expression, whose intended process result is `42`; there is no stdout contract.

```bash
./bin/souc check tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio
./bin/souc run tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio
```

**Working alternative.** Use the passed [`generics`](#7-instantiate-generic-functions-and-inspect-record-boundaries) example for the demonstrated generic-function task and [`cook-struct`](#3-give-data-names-return-tuples-and-preserve-element-types) for the demonstrated concrete-record task. Neither is a generic-record replacement or proof of layout parity; investigate this retained source separately.

**Common errors.** A parsing-only declaration does not demonstrate construction or field access. The complete file here includes both.

## Adapt a recipe with an LLM

Give the model the complete nearest fixture, the actual declarations of each
imported symbol, and the selected profile. State the change as a small input
problem with an observable contract. This request skeleton is **nonrunnable
text**, not Sounio source:

```text
Input problem:
Closest complete example ID and repository path:
Compiler profile and library root:
Change requested:
Existing types, imports, callback signatures, and effects to preserve:
Success stdout, stderr, and process result:
Negative cases that must refuse or report an error:
Checks and runs already observed, with receipt path:
Unknowns that must remain explicitly unverified:
```

For genuinely new Sounio, use `sounio_verificado` before accepting a generated
candidate when that tool is available, then run the repository's actual check
and execution path. Its remote compiler result is scoped to its own artifact;
it does not replace a receipt for the requested profile. If the tool is
unavailable, report that fact and continue with the repository-grounded repair
loop. No new Sounio was generated for this page.

When adaptation fails, keep the original and adapted files separate. Capture
the command, actual exit, stdout/stderr, and exact diagnostic. Compare with the
[error catalog](error-catalog.md), then identify whether the failure concerns
syntax, import visibility, type/effect rules, backend lowering, or runtime
behavior. Do not erase scientific information to make a compiler path accept.

## Completion record for these recipes

| Evidence | State |
|---|---|
| Full-file source selection | Inspected at the baseline above. |
| Fence/source identity | Checked structurally when the guide is validated. |
| Pinned-profile observations | Actual per-case records from job `14651`; see `example-results.json`. |
| Check-only observations | Provider/import-consumer rows passed check; no run result is implied. |
| Run observations | Passed and failed run rows are labeled individually; separate displayed check commands are not independently witnessed by those rows. |
| Quarantined failures | `cook-closure`, `cook-generic-fn`, and `cook-generic-pair`; original intended exits preserved. |
| Working generic path | Existing manifest ID `generics`, copied completely and recorded `run-passed`. |
| Original test intent | Retained through annotations, branches, assertions, and source links. |
| Scientific numerical verification | No new calculation or accuracy claim made. |
| Cross-profile compatibility | Requires separate artifact-specific receipts. |

The [guide's manifest](example-manifest.json) is the binding surface for example IDs, sources, commands,
and validation receipts, paired with the [actual logs](example-results.json).
Preserve nonzero expected results and conditional stdout requirements when a
runner selects these examples.

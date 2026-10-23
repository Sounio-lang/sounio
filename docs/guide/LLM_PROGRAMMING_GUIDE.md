<!-- docs:meta
topic_id: repo.docs.guide.llm-programming-guide
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.llm-programming-guide
-->

# Write Sounio with a language model

[Ελληνικά](LLM_PROGRAMMING_GUIDE_EL.md) · [Português](LLM_PROGRAMMING_GUIDE_PT.md)

**Edition:** October 2026. **Prepared:** 2026-10-16.
**Source baseline:** `f5b2c5850be9f3d3995719ffcade44a6d851658a`.

This is a practical guide for generating, explaining, repairing, and verifying
Sounio programs. It is designed to work as retrieved context for small models
as well as a deep reference for stronger coding agents. Start with a complete
working program, change one thing, and verify the result. Do not translate a
Rust example and hope that similar punctuation implies identical semantics.

The language [specification](../spec/LANGUAGE_SPECIFICATION.md) defines the
reference's distinctions between adopted contracts, source observations, and
open decisions. This guide teaches how to use that material; it is not a
second, simplified language specification.

**Execution scope:** the [example receipt](../llm-guide/reviews/example-execution.md)
records 50 invocations over 47 distinct source files on the identified
committed Madaros Linux prebuilt: 30 runtime contracts matched, four check-only
contracts matched, eight intended refusals matched, and eight cases failed
their intended assertions. The failures are quarantined, not promoted into
working recipes. A source rebuild, every-target validation, numerical
certification, and a guarantee about every language model are not established.

[Português](LLM_PROGRAMMING_GUIDE_PT.md) ·
[Cookbook](../llm-guide/cookbook.md) ·
[Complete example atlas](../llm-guide/examples.md) ·
[Example manifest](../llm-guide/example-manifest.json) ·
[Errors](../llm-guide/error-catalog.md) ·
[Library discovery](../llm-guide/stdlib-index.md)

For a task-sized, copyable starting context, use the
[thirty-task trilingual curriculum](../llm-guide/task-playbook.md):
`node scripts/docs/sounio_task_context.mjs show first-program --lang en`.
It emits complete reference/control sources, historical profile-specific
results, adaptation steps and pitfalls without running a compiler or LLM.
The tasks reuse existing programs; they are not thirty new executions.
The [translation and repair guide](../llm-guide/translation-and-repair.md)
explains intent-preserving ports, stage diagnosis and bounded repairs.

<!-- llm:section orientation -->
## 1. Give the model the right amount of context

Use three layers rather than feeding the model the entire repository:

| Layer | Contents | Use |
|---|---|---|
| Start | [`llms.txt`](../../llms.txt), the task, environment, and one complete example | A first program or small edit. |
| Task | The relevant section here, one recipe, actual API declarations and a refusal control | Imports, ownership, scientific values, or a compiler error. |
| Deep | The named specification rules, implementation paths, and execution receipts | Semantic disagreement, target support, precision, or a compiler investigation. |

A complete task packet names the input, required result, permitted changes,
engine/artifact, host/target, library root, and what counts as success.
“Implement a vector” is not the same task as “use a bounded floating container
whose capacity and failure behavior are known.” Resolve that distinction
before choosing a representation.

Keep the packet small enough that the model can attend to the whole example.
Retrieve one relevant source program and its dependencies. A list of filenames,
a fragment missing declarations, or a large context dump does not teach a model
how to finish a program. The atlas contains full files so that names, effects,
return conventions, and imports remain visible.

For an ordinary self-contained program, the first four examples in this guide
are sufficient to begin. For library use, also read the actual selected
declaration and importing consumer. For unfamiliar scientific semantics,
retrieve the corresponding specification section rather than inventing a
simpler meaning.

When the request is already clear, make a reasonable choice and declare it;
do not stop at every editorial detail. Ask only when the choice changes
semantics, data, the required environment, or authorized limits.

<!-- llm:section workflow -->
## 2. First success, then a controlled edit

### 2.1 Identify the executable before trusting an outcome

Use the [first-success preflight](../../llms.txt#first-success). It establishes
the requested profile, compatible host, resolved raw artifact and digest,
library root, and expected hello contract **before** the check/run pair.
The public launcher is `./bin/souc`; its raw artifact is not the launcher
shell script. Hashing the wrapper does not identify the compiler it selects.

The documented baseline is the committed Linux x86-64 Madaros prebuilt.
A source-built compiler or Apple Silicon native artifact is a different
profile. The [macOS ARM64 guide](../compiler/MADAROS_MACOS_ARM64.md) explains
native setup. Do not attempt an incompatible ELF, substitute a seed compiler,
or delete an override just to obtain a pass.

Use commands in an environment authorized for the task. Public recipes do
not override a repository or session rule requiring remote execution. In this
repository's operational work, full and stress validation belongs on
Foundry/Slurm; an illustrative command is not permission to run that workload
on a shared workspace or notebook.

### 2.2 A complete first program

Source: [`examples/hello.sio`](../../examples/hello.sio), manifest ID `hello`.
The intended exit is zero; stdout contains `Hello, Sounio`. Compiler logging
can accompany the program's output, so the manifest asserts a substring rather
than inventing exact whole-stream equality.

<!-- llm:example hello -->
```sio
// First Sounio program

fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

After the preflight, from the repository root:

```sh
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

The receipt records this example's result. A model writing a new program must
obtain a new result for the changed file; it must not copy this expected
greeting into its report and call that observation.

### 2.3 The edit/check/observe loop

1. Read the full reference program and its declared effects.
2. Write a small complete change in the authorized file.
3. Check that changed file, not the original reference by accident.
4. Read the complete diagnostic and repair the earliest relevant cause.
5. For an executable task, run the changed program and assert the task result.
6. Include a nearby positive control and an intended-refusal control where
   they distinguish the claimed rule.
7. Report what ran and what remains unestablished.

A check result supports the stage reached by that adapter. Emission supports
artifact construction. Execution supports the asserted observations. A linked
test, a successful parser recovery, a compiler version, or an exit code alone
cannot stand in for all of those.

### 2.4 Do not combine incompatible tool evidence

`sounio_check` is an MCP tool when configured, not a guaranteed shell command.
Its server can run in a different environment. Read the
[MCP attribution procedure](../../llms.txt#mcp-profile-attribution) before
combining its output with terminal results.

The current MCP check response does not itself expose the raw compiler digest
and library-root identity. The run implementation can precheck before applying
per-run environment overrides. A single tool call therefore does not prove a
single check/run profile. Obtain server-side identity under the effective
environment, or label attribution unknown and repeat the pair through one
identified CLI profile. Do not invent missing response fields.

<!-- llm:section syntax -->
## 3. Learn Sounio spelling, not Rust substitutions

| Model habit | Sounio procedure |
|---|---|
| Write `let mut value` | Use `var value` for a reassignable binding. |
| Write `&mut value` | Use the selected exclusive-reference spelling `&!value`. |
| Use `println!` or `assert!` | Call `println` or `assert` as ordinary operations. |
| Delete every semicolon | Do not: array types/repetition use `;`, and relevant statement positions accept it. |
| Assume every method needs a receiver | Instance receivers are explicit; an associated constructor can have no `self` receiver. |
| Add Rust attributes or a derive macro | Find the actual Sounio declaration/test mechanism instead of translating a foreign feature. |
| Use a format string and extra print arguments | Read the actual printing signature; the first example uses one string argument. |
| Ban generics, traits, or closures entirely | Inspect their bounded implementation and the selected artifact's witnesses. |
| Promise every parsed construct executes | Trace checking, lowering, emission, and runtime support separately. |

Bindings can have explicit types or inferred initializer types. Function
parameters need their declared types in the selected syntax. Keep names and
declarations complete; do not rely on a placeholder helper being supplied by
the compiler. Static constructors and instance methods have different
signatures.

Newlines and optional punctuation are parser rules, not a whitespace cleanup
exercise. A leading operator, optional return operand, or multiline expression
can cross a lookahead boundary. Use a nearby complete fixture and consult the
[lexical/syntax chapter](../spec/2026-10/01-lexical-syntax-and-modules.md)
when a parse fails. Do not introduce an untested syntax workaround and claim it
is the language rule.

Treat strings, characters, byte arrays, foreign string handles, and source
names as distinct representations. A value printed successfully is not
necessarily a round-trippable serialization. A different language's string
interpolation or Unicode-indexing convention is not evidence for Sounio.

<!-- llm:section types -->
## 4. Choose representations deliberately

### 4.1 Scalar and compound types

Integer and floating spellings identify language types, not an automatically
portable overflow or rounding contract. Use the requested width and format.
Wide floating values are not double-double or quad-double correction records,
and a narrower fallback does not preserve a request for wide precision.

Arrays, tuples, structs, enums, references, functions, and library records have
different construction, access, ownership, and ABI rules. A type name accepted
by a checker does not establish every operation or target layout. Consult the
[type chapter](../spec/2026-10/02-types-generics-and-ownership.md) and
[numerics chapter](../spec/2026-10/05-numerics-units-and-algebra.md).

### 4.2 A complete struct program

Source: [`examples/native/struct_basic.sio`](../../examples/native/struct_basic.sio),
manifest ID `struct`. Its bounded runtime assertion is exit zero and stdout
containing `7`; that substring is weaker than a full serialization contract.

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

To adapt it, change one field or operation, update the expected observation
from the task, and verify the new artifact. Do not change the field to a raw
integer merely because a richer scientific type exposes a checking boundary.

### 4.3 Arrays, tuples, and collections

A fixed-size array's element type, capacity, initializer, indexing, and borrowing
must agree. Repetition and array types contain semicolons as syntax. Bounds
checks and mutation have their own effects. Do not cast every index blindly;
copy the selected API's expected index type.

Tuple access and tuple pattern binding are separate paths. The receipt
quarantines the selected nested-destructuring fixture, while other tuple
access/signature programs passed. That is not a universal ban on tuples.

The source contains more than one vector-like representation. The public
`VecF64` free-function API in the curated index is not interchangeable with
nearby non-public `IntVec`/`FloatVec`, a heap-backed generic record, a fixed
array, or a compiler-specialized sequence. Check capacity, element layout,
allocation, and failure values. A returned zero may be a sentinel rather than
a validated element.

<!-- llm:section functions -->
## 5. Functions, generic calls, and callable values

A useful function example must show its declaration, parameter/result types,
required effects, call site, and complete program result. A snippet that calls
undefined `map`, `sqrt`, or `add_one` is not a runnable tutorial.

Named function references are a useful starting point for higher-order tasks.
Use [`cook-function-ref`](../llm-guide/examples.md#cook-function-ref) for an
executed bounded example. A closure needs its capture semantics, callable type,
effects, lifetime, and target path. The selected `cook-closure` fixture failed
on the pinned artifact and is quarantined; other implementation paths exist.
Neither “closures never exist” nor “every closure works” follows.

### A working explicit generic-call route

Source: [`tests/run-pass/generics_multi_param.sio`](../../tests/run-pass/generics_multi_param.sio),
manifest ID `generics`. The original program contains two runtime controls;
the catalog asserts exit zero and the first success substring. Its presence
does not establish arbitrary trait solving, generic methods, or every
specialization/ABI combination.

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

Keep the exact call spelling demonstrated here when adapting this route.
The native-runtime single-generic-function and generic-pair fixtures selected
for this audit failed; their names were not proof that they ran. Do not
rewrite the manifest to make them green or infer a language-wide prohibition
from those failures.

Trait declarations, bounds, name resolution, method selection, and obligation
enforcement are distinct. A parser accepting a bound is not a complete
coherence or trait-satisfaction proof. Read the actual generic/trait source
and a discriminating caller when the task depends on those properties.

<!-- llm:section control-flow -->
## 6. Control flow and observable order

Use the complete [while-loop](../llm-guide/examples.md#while-loop),
[array iteration](../llm-guide/examples.md#cook-for-array), and
[range/option](../llm-guide/examples.md#cook-range-option) programs before
combining forms. Read their entrypoint result and their assertion contract:
some programs deliberately use a nonzero success exit.

`if` expressions, loops, patterns, return, break, continue, and block tails
have both syntax and checking rules. Ensure all result paths agree with the
declared function result. A branch that prints a success marker only when its
comparison holds needs that marker asserted: exit zero alone can miss a failure.

Do not use `&&` or `||` to protect a dangerous operation merely because a
different language short-circuits it. The inspected ordinary lowering and
target-specific paths do not establish one portable effectful evaluation
contract. Use explicit control flow where safety/order matters and run a
witness that distinguishes the intended behavior on the selected route.

This is not a suggestion to ban logical operators for ordinary boolean
expressions. It is a requirement not to confuse a type/effect set with an
execution trace. Test order, event multiplicity, mutation, and any scientific
information affected by a branch.

Recursion and large fixed buffers can encounter implementation capacities.
A crash or timeout is not a language-level proof of invalidity, nor permission
to remove a guard. Record the bound and the route, then reduce the reproducer
without changing the failure's meaning.

<!-- llm:section ownership -->
## 7. References, borrowing, and resources

A shared reference, an exclusive reference, a raw pointer, a linear handle, and
an owned value are not interchangeable just because their current ABI uses a
machine word. Preserve the contract at call boundaries and returns.

Start with the verified
[explicit borrow-release program](../llm-guide/examples.md#cook-borrow-release)
and [array-reference program](../llm-guide/examples.md#cook-array-ref).
Read the entire file: initialization, borrow acquisition, use, release/scope,
and the asserted mutation all matter.

When a borrow conflict appears, locate both acquisitions and the lifetime of
each use. Shorten the borrow or restructure disjoint accesses only when that
preserves the task. Do not replace an exclusive borrow with a raw pointer,
remove a linear qualifier, duplicate a consumed handle, or change the test
expectation to bypass the check.

The implementation has bounded and route-specific analyses. A helper lifetime
module or a declared drop strategy does not prove complete non-lexical
lifetimes, destructor scheduling, pointer validity, or arbitrary partial moves.
Conversely, missing enforcement does not cancel an adopted resource rule.

A useful resource test has a valid nearby control and a specific refusal.
The [borrow-conflict negative](../llm-guide/examples.md#negative-borrow-conflict)
is checked for its intended message, not just any nonzero exit.

<!-- llm:section effects -->
<a id="effects"></a>
## 8. Effects without kitchen-sink annotations

Effects describe operations and obligations. Their declaration, inference,
propagation, checking, and execution are different mechanisms. Use the
[effects chapter](../spec/2026-10/03-effects-handlers-and-concurrency.md)
for the current adopted/source distinctions.

| Task | What to inspect |
|---|---|
| Print or access host I/O | The called operation's `IO` contract. |
| Mutate through a reference or shared structure | The mutation contract and `Mut` enforcement route. |
| Reassign a function-local variable | Local mutation semantics and the selected engine; not an automatic universal `Mut` rule. |
| Divide, take a remainder, assert, index, or cast | The exact operand/API contract and its `Div`/`Panic` or other obligations; not a universal table inferred from operator spelling. |
| Call a scientific metric or observe a value | The selected representation and `Epistemic`/`Observe` boundary. |
| Call a function reference | The callable signature and effects retained by that checking route. |

Do not add every effect to `main` as a generic repair. Start with the required
set and propagate it through the real call chain. If the task changes the
contract, explain that change. If an engine fails to enforce a rule, keep the
intended semantics and describe the enforcement gap.

### A complete effect-subset example

Source: [`tests/run-pass/effect_superset_ok.sio`](../../tests/run-pass/effect_superset_ok.sio),
manifest ID `effects`. The receipt asserts the `effect superset: PASS`
substring and successful exit.

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

The [missing-effect negative](../llm-guide/examples.md#negative-effect-missing)
produced its expected refusal on this artifact. Historical blanket text saying
E035 never fires under default checking must not override that receipt.

Unknown effect names and capacity refusal are also separate diagnostics.
A bounded effect set is not general row polymorphism, and changing an ambient
set is not proof of operational handler semantics. Parsing a handler or async
construct cannot manufacture its runtime.

<!-- llm:section scientific -->
## 9. Preserve the scientific information

### 9.1 Identify the value domain first

| Surface | Meaning to preserve |
|---|---|
| Compiler `Knowledge<T>` | Its checker predicates and lowered scientific channels on the selected route. |
| Library `Epistemic` | The record's declared value, variance, confidence, and API preconditions. |
| EISA `EReg` / `EReg2` | Distinct value, arithmetic correction, uncertainty, and origin conventions. |
| Unit annotations | Dimension, representation, conversions, and boundary checks. |
| Quantum channels/counts | Presence states, dependence, sampling/model assumptions, and device provenance. |

Use the [source-based scientific reference](../spec/2026-10/04-knowledge-uncertainty-and-provenance.md)
rather than substituting one similarly named record for another.
[The library example](../llm-guide/examples.md#cook-epistemic-library)
and [EISA example](../llm-guide/examples.md#cook-eisa) have their own bounded
execution contracts. Their existing assertions are not independent
metrological or numerical certification.

Do not silently unwrap a scientific value to `f64`, discard a variance,
set missing uncertainty to zero, narrow a requested format, or remove a
provenance predicate to make a call type-check. Ask for an authorized semantic
change only when preserving the requested contract is impossible.

Equal central values do not establish common source identity, equal variance,
independence, or causal origin. Arithmetic error is not measurement uncertainty.
Confidence, ignorance, validity, and statistical coverage need their own
definitions. An absent channel is different from an explicit numerical zero.

### 9.2 Units, order, and numerical verification

The chosen bare-unit addition/scaling fixtures failed on the pinned prebuilt.
They remain diagnostic cases, not a working units tutorial. Use the
specification to identify the annotation/operation contract, then require an
execution receipt on the named profile before treating it as a working route.
Do not switch engines without recording a different profile.

Preserve parenthesization where floating rounding or non-associative algebra
matters. A transformation that is valid in an associative mathematical model
need not preserve this program's observable behavior or scientific channels.

Use task-specific comparison criteria. An inherited `check_near` tolerance
is not automatically valid for a new domain, magnitude, exceptional value,
or small nonzero uncertainty. Do not recompute a scientific result with an
unauthorized numerical oracle. Follow the current ADR009 working protocol in
[the foundations, `SPEC-011`](../spec/2026-10/00-foundations-and-conventions.md).
Proofs belong in Lean 4; numerical agreement and empirical validation are
separate forms of evidence.

A GUM-named fixture does not establish complete JCGM compliance or traceability.
An ontology relation is not a clinical diagnosis; a simulation is not a
hardware experiment. Keep research hypotheses intact while identifying the
evidence needed to test them.

<!-- llm:section modules -->
## 10. Imports, visibility, packages, and actual APIs

Use the [library-discovery index](../llm-guide/stdlib-index.md) as a workflow:

1. Find the selected module and declaration.
2. Read its export visibility, complete signature, effects, and body.
3. Read an actual importing consumer.
4. Preserve its module/file context and standard-library root.
5. Check the consumer on the intended profile.
6. Execute it only if it has an entrypoint and runtime behavior is requested.

The [provider](../llm-guide/examples.md#cook-import-provider) and
[consumer](../llm-guide/examples.md#cook-import-main) are check-only cases in
this corpus. Do not call their result a runtime pass. The consumer's source
returns 42; executing it later requires that contract, not an assumed zero.

Copying an importing file into an arbitrary temporary directory can change
module resolution. Retain companions and search context, or adapt them
explicitly. A directory under `stdlib` does not mean every declaration is
public, callable, implemented on the target, or numerically verified.

Current package lookup uses conventional source paths derived from import
spelling. A manifest field's existence is not proof of manifest-controlled
resolution, version solving, a hosted registry, or a publish operation.
See the [package/tool chapter](../spec/2026-10/07-library-packages-and-tools.md).

Do not invent helpers from a familiar ecosystem. A reasonable function name,
generated type annotation, or LSP completion is a candidate, not an API.
Prefer a real source declaration to a large historical generated index.

<!-- llm:section targets -->
## 11. Host, FFI, concurrency, and device boundaries

Keep the target in the task packet. The public launcher, raw driver,
bootstrap seed, wrapper, and emitted program can have different command
interfaces. Do not pass launcher verbs to a raw positional compiler.

| Claim | Evidence needed beyond parsing |
|---|---|
| Native executable | Actual format/architecture, emission, execution, exit and output. |
| Foreign call | Symbol, ABI, layout, effects, link/runtime resolution, and observed behavior. |
| Async or concurrency | Reachable scheduler/runtime behavior, ordering, resource and failure contracts. |
| Effect handler | Operation dispatch, continuation semantics, and its checked/emitted/runtime path. |
| GPU result | Actual device/runtime route and outputs, not merely PTX/MSL text. |
| Quantum hardware | Device/job/circuit/shot/count receipt, not a simulator or parsed count file. |
| Formal result | Named model, assumptions, toolchain, completed checker build, and production correspondence. |

The [execution chapter](../spec/2026-10/06-execution-memory-and-targets.md)
details supported and refused paths. Apple Silicon has a separate native
ARM64/Metal route, not a distributed Linux ELF that happens to run on macOS.
Emission, signing, process exit, and real device execution remain separate
observations.

A target name accepted by a CLI or MCP schema is not proof that every
operation is supported there. Never turn a missing device, unsupported IR,
foreign-symbol error, or absent scheduler into a successful CPU fallback
without an explicit contract and disclosure.

<!-- llm:section diagnostics -->
## 12. Repair the cause rather than silencing the checker

Use the [diagnostic guide](../llm-guide/error-catalog.md). Start with the
engine, adapter, source revision, requested stage, complete stdout/stderr,
and earliest relevant diagnostic. A numeric error code alone is insufficient:
different namespaces and paths reuse codes.

A practical repair sequence:

1. Read the diagnostic's span and enclosing declaration/call.
2. Check the callee and imported definition, not only the reported line.
3. Compare with a nearby valid example under the same artifact and environment.
4. Change the smallest cause that preserves the task.
5. Recheck and, when required, rerun the changed program.
6. Run the intended negative counterpart and record the precise refusal.

Do not remove tests, weaken expected output, increase tolerances without a
domain justification, cast away scientific wrappers, or disable an effect/
ownership check. A failing compiler or missing dependency cannot make every
negative case a success.

The atlas retains eight cases that did not satisfy their intended assertions.
Two negative units cases exited nonzero but did not produce their required
diagnostic. That is a failed witness, not confirmation of the intended unit
boundary. The six positive failures are likewise excluded from working recipes.

Most retained per-code explanation pages were not refreshed by this audit.
Their visible banner identifies that limit. E035 and E249 received targeted
corrections; other old examples must be re-established against the selected
artifact before use.

When a repair would choose between unsettled language meanings, change the
public API, or require a compiler/library fix outside the task, stop and
provide the reproducer and decision needed. Do not replace the research
question with an easier one and call it a fix.

<!-- llm:section recipes -->
## 13. A progressive route through complete examples

The [cookbook](../llm-guide/cookbook.md) explains each lesson and adaptation.
The [atlas](../llm-guide/examples.md) preserves the complete source.
The [manifest](../llm-guide/example-manifest.json) records the actual per-case
verification, not a production label inferred from filenames.

| Goal | Start with |
|---|---|
| Print and return a process result | `hello` |
| Reassignment and a while loop | `while-loop`, `cook-while` |
| Fixed-array iteration | `cook-for-array` |
| Ranges and option-controlled iteration | `cook-range-option` |
| Struct creation and fields | `struct`, `struct-parameter`, `struct-return` |
| Tuples and function signatures | `cook-tuple`, `cook-tuple-signature` |
| References and observable array mutation | `cook-borrow-release`, `cook-array-ref` |
| Enum matching | `cook-enum`, with intended success exit 42 |
| Named callable values | `cook-function-ref` |
| Explicit generic calls | `generics` |
| Import provider/consumer | `cook-import-provider`, `cook-import-main`, check-only |
| Result and vector library APIs | `cook-result`, `cook-vec` |
| Distinct epistemic and EISA APIs | `cook-epistemic-library`, `cook-eisa` |
| An intended refusal | A verified `negative-*` case for the matching rule |

Inspect the verification object before using any ID. The quarantined closure,
single-generic, generic-pair, nested-destructuring, and selected units cases
are diagnostic exercises, not part of the first-success path.

Do not count repeated IDs pointing to the same source as independent programs.
This corpus has 50 case entries and 47 distinct source files. Thirty results
include execution, four establish only checking, eight match intended
refusals, and eight do not meet the asserted contract. None of those counts
establishes whole-language conformance.

<!-- llm:section prompts -->
## 14. Reusable prompts for smaller and larger models

The text below is a template, not an executed conversation or a benchmark.
Fill the task-specific fields and provide the referenced source. Do not
substitute an imagined model result for an actual tool result.

### Start a new program

```text
Write a complete Sounio program for [task].
Inputs: [types, values/source, permitted assumptions].
Required observations: [output, exit, resource/scientific channels].
Execution profile: [engine, raw artifact/hash, host/target, stdlib root].
Allowed files: [paths]. Do not publish changes.
Use the supplied complete example [ID] and its actual API declarations.
Preserve [units, uncertainty, precision, ownership, effects, order].
Do not invent helpers. Explain any missing prerequisite before proceeding.
Check the changed file, then run and assert the required observations.
If tools cannot execute here, provide commands and label them unrun.
```

### Repair one compiler failure

```text
Diagnose this exact Sounio failure under the stated profile.
Here are the complete command, exit, stdout, stderr, and source.
Here is the callee/import declaration and a nearby valid control.
Identify the earliest causal failure. Make the smallest justified edit.
Do not erase scientific types, broaden effects blindly, disable guards,
change expected output, switch engines, or silently narrow precision.
Recheck the same file/profile and rerun the affected observation.
Report the before/after result and what remains unestablished.
```

### Discover a library operation

```text
Find an existing Sounio API for [operation].
Retrieve the declaration, export visibility, representation, effect signature,
body preconditions/failure behavior, and an actual importing consumer.
Distinguish source presence from successful import and runtime evidence.
Choose a complete verified example; do not invent a familiar-language API.
Preserve the requested information channels and numerical-tool restrictions.
```

### Extend a scientific program

```text
Extend the supplied program without changing the research question.
State the value domain, uncertainty/dependence/provenance model, units,
precision, and observation contract before editing.
Separate model assumptions, compiler behavior, numerical verification,
formal proof, and empirical evidence. Use only authorized numerical tools.
If a transformation loses information or needs a semantic decision, disclose
it rather than replacing the task with a simpler calculation.
```

For a small model, provide one program and one change at a time. For a larger
agent, allow a bounded plan and retrieval of additional APIs, but keep one
owner per shared file and explicit evidence at each checkpoint.

### Domain-specific task cards

These cards make a scientific request concrete without pretending that a
domain prompt is a compiler receipt. Attach the complete source, the relevant
library declaration, the intended observation, and the selected execution
profile. The model must retrieve the current Sounio API before writing code;
the names below are search areas, not permission to invent an API. Keep model
assumptions, scientific data, compiler evidence, numerical verification, and
publication decisions as separate layers.

#### Chemistry, kinetics, and materials

```text
You are assisting with a Sounio chemistry task.

Research task: [reaction, equilibrium, kinetics, spectroscopy, materials, or
                 other precisely stated question].
Scientific inputs: [species, concentrations, units, temperature, pressure,
                    time grid, measured values, uncertainty, provenance].
Required observation: [table, curve, invariant, status, file, or other output].
Allowed files: [exact source and test paths]. Do not publish changes.
Execution profile: [revision, raw compiler/hash, host/target, stdlib root].

Before editing:
1. Inspect the supplied source and retrieve the actual declarations and a
   real consumer for every chemistry, units, array, numeric, or result API.
2. List species and units explicitly. State whether values are measured,
   assumed, fitted, simulated, or derived, and preserve that provenance.
3. State precision, uncertainty/dependence, boundary conditions, and the
   failure contract. Do not silently coerce units, discard uncertainty, or
   replace a reaction model with a simpler calculation.
4. Separate a source/type-check claim, a runtime observation, and an
   independently authorized numerical verification. A simulation is not an
   experiment and a passing example is not a chemical or industrial guarantee.

Make the smallest source-bound change. Use only APIs confirmed in the source
and the supplied profile. Return the complete changed file or exact patch,
then report the check command, run command, exit, stdout, stderr, source hash,
and every step that was not executed. If a declaration or prerequisite is
missing, stop and identify it instead of inventing a helper.
```

#### AI/ML, tensors, and differentiable programs

```text
You are assisting with a Sounio AI/ML task.

Task class: [data preparation, tensor transform, inference, training,
             autodiff, optimization, evaluation, or reproducibility].
Inputs: [data schema/source, shapes, axes, dtype, precision, device, seed,
         provenance, privacy constraints].
Required observation: [shape/value contract, loss trace, metric, artifact,
                       exit status, or explicit refusal].
Allowed files: [exact source, fixture, and test paths]. Do not publish.
Execution profile: [revision, raw compiler/hash, host/target, stdlib root,
                     device/runtime if known].

Before editing:
1. Classify the operation as tensor construction, transformation, model
   definition, autodiff, optimizer, inference, or evaluation. Retrieve the
   actual declarations and a complete consumer from the current tensor,
   autodiff, ML, or example sources; never import a familiar API by analogy.
2. Write down shape, rank, axis order, dtype, precision, batching, masking,
   device, randomness, and ownership assumptions. Preserve provenance and
   uncertainty; do not turn an absent benchmark into a performance claim.
3. Keep implementation, compilation, execution, benchmark, training data,
   and statistical/numerical verification as separate evidence. A successful
   tensor operation is not model accuracy, safety, or generalization.
4. Do not fabricate a dataset, checkpoint, accelerator, seed, metric, or
   result. If the requested operation is not present in the selected profile,
   report the missing capability and propose the smallest source-bound next
   step.

Make one minimal change at a time. Return the complete source or exact patch
and a reproducible evidence record: commands, exit statuses, raw streams,
source/artifact hashes, data identity, and unrun steps. Keep training and
inference claims bounded by the observed profile.
```

#### Research-only medicine, ontology, and PBPK

```text
You are assisting with a Sounio research-only medicine, ontology, or PBPK
task. This is not clinical advice, diagnosis, prescribing, device validation,
or a patient-specific recommendation.

Research question: [precise terminology, ontology, PK/PD, PBPK, cohort,
                   mechanistic, or reproducibility question].
Inputs and governance: [de-identified data or synthetic fixture, provenance,
                         units, population, assumptions, consent/permission].
Required observation: [ontology relation, typed record, simulation output,
                       sensitivity/result table, refusal, or status].
Allowed files: [exact source and test paths]. Do not publish.
Execution profile: [revision, raw compiler/hash, host/target, stdlib root].

Before editing:
1. Retrieve the actual ontology, clinical, PBPK, uncertainty, and library
   declarations plus a real consumer. Identify whether each statement is a
   vocabulary/ontology fact, a model assumption, a simulation result, a
   compiler result, a formal proof, or an empirical clinical claim.
2. Preserve provenance, cohort/population scope, units, precision, uncertainty,
   dependence, missingness, and data-governance constraints. Do not infer a
   patient conclusion from a population model or a terminology relation.
3. Keep terminology closure, model execution, numerical verification, and
   clinical validation distinct. Never claim diagnosis, treatment efficacy,
   safety, regulatory approval, or patient benefit from code execution alone.
4. Use only the authorized numerical/proof toolchain for the requested
   verification. If a clinical, governance, dataset, or execution prerequisite
   is absent, stop and label the result unverified rather than filling the gap.

Make the smallest source-bound change. Return the source or exact patch and a
bounded evidence record with commands, exits, raw output, source/artifact
identity, data identity, assumptions, and unrun validations. Include a clear
research-only limitation in any user-facing summary.
```

<!-- llm:section delivery -->
## 15. Deliver code with bounded, reproducible evidence

Before returning a program, verify:

- The entire file is present, with declarations, imports, entrypoint where
  applicable, and the effects required by its operations.
- The checked/running source is the changed file and matches the recorded hash.
- Compiler artifact, engine, adapter, host/target, environment, and library
  root are identified; CLI and MCP results are not silently mixed.
- Successful checks, emitted artifacts, runtime exits, and output assertions
  are reported separately.
- Negative controls fail for the intended cause, not an unrelated parse,
  dependency, timeout, or crash.
- Scientific channels, units, precision, provenance, ownership, and ordering
  requested by the task remain preserved.
- Unrun numerical verification, Lean builds, full corpus, GPU execution,
  remote CI, publication, and human acceptance remain explicitly unclaimed.

Use the delivery record in [`llms.txt`](../../llms.txt#delivery-record).
The [review and preparation record](../llm-guide/reviews/guide-preparation.md)
describes this guide's own evidence and limitations. No generic model-success
rate is inferred from a set of passing source fixtures.

This guide, the Portuguese edition, the cookbook, and the atlas are maintained
together. `node scripts/docs/validate_llm_guide.mjs` checks topic coverage,
source identities, complete code blocks, quarantine labels, and links.
Its self-tests mutate structural contracts; neither the validator nor its
self-tests execute the compiler or prove the prose. Compiler receipts and semantic decisions retain
their own acceptance conditions.

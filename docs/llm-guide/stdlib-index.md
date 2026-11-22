<!-- docs:meta
topic_id: repo.docs.llm-guide.stdlib-index
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.stdlib-index
-->

# Find and use a Sounio library API

This index is a practical discovery route for a model or a human starting with
a concrete task. It replaces guessed APIs with source declarations, actual
import consumers, and explicitly scoped validation. It is curated from baseline
`f5b2c5850be9f3d3995719ffcade44a6d851658a`; it is not an exhaustive catalog
or a production-readiness rating.

**Declarations on this page are source-inspected.** That establishes their
spelling and declared signatures, not general import, lowering, or execution
support. Selected complete consumers now have profile-scoped observations in
[the actual results](example-results.json) and
[execution receipt](reviews/example-execution.md); consult each row's recorded
stage, exit, and assertions. Those observations do not validate every listed API.
Signature blocks are explicitly **nonrunnable text**. Complete Sounio programs
with manifest-bound example markers live in the [cookbook](cookbook.md).

The [language/library boundary](../spec/2026-10/07-library-packages-and-tools.md)
is the governing starting point: a directory under `stdlib/` does not certify
a numerical method, host service, or scientific claim.

## 1. Discover from a problem, not an imagined namespace

Suppose the input problem is “store a bounded set of floating values and inspect
its length.” Search for the representation and operation together:

```bash
rg --files stdlib/collections
rg -n '^(pub )?(struct|enum|fn) |^impl ' stdlib/collections/vec.sio
rg -n 'VecF64|vec_new|vec_push|vec_len' stdlib/collections/vec.sio
rg -n '^use collections::vec|vec_new|vec_push|vec_len' tests examples
```

Then read the declaration **and its body**, followed by a complete importing
consumer. For this problem, the concrete starting point is
[`test_vec_stdlib.sio`](../../tests/stdlib/collections/test_vec_stdlib.sio),
copied in [cookbook lesson 9](cookbook.md#9-use-the-vector-representation-the-source-actually-exports).
It calls exported free functions on `VecF64`; the nearby `IntVec` and
`FloatVec` declarations are separate, non-`pub` records in the inspected file.

A repeatable search sequence for any family is:

1. Find source candidates with `rg --files stdlib` and a domain term.
2. Inspect the chosen file, its `module`/`use` declarations, `pub` surface,
   concrete types, and operation bodies.
3. Find consumers importing **that exact module and symbol**, not only files
   with similar names.
4. Read the entire consumer and any auxiliary files. Verify that it actually
   calls the operation and has observable checks.
5. Preserve the consumer's import spelling and profile until reproduced.
6. Only then adapt the smallest complete program and validate the new case.

A broad starting inventory, with no support or maturity assertion:

| Input problem | Start inspecting | Resolve before generating a caller |
|---|---|---|
| Tagged success/failure | `stdlib/core/result.sio` | Concrete record versus compiler generic result; tag handling. |
| Fixed bounded collections | `stdlib/collections/vec.sio` | Capacity, exported free functions, sentinel behavior. |
| Heap ownership | `stdlib/collections/heap_vec.sio`, `stdlib/mem/` | Storage width, allocation, free, target restrictions. |
| Text and byte conversion | `stdlib/str/lib.sio` | `Str` versus language `string`, capacity and encoding. |
| In-memory I/O buffer | `stdlib/io/io.sio` | Buffer value versus borrowed output, length units. |
| JSON interchange | `stdlib/json/parser.sio` | Selected `JsonDoc` representation, fixed byte arrays, parse result. |
| Elementary mathematical functions | `stdlib/math/pure.sio` | Public declaration, domain branches, declared effects. |
| ODE callbacks | `stdlib/ode/rk4.sio` | State layout, callback signature, mutation, return meaning. |
| Descriptive statistics | `stdlib/stats/descriptive.sio` | Actual exports, FFI dependencies, statistical assumptions, real callers. |
| Measurement uncertainty | `stdlib/epistemic/knowledge.sio`, `stdlib/epistemic/gum.sio` | Record identity, uncertainty units, dependence assumptions. |
| Value/error/uncertainty lanes | `stdlib/eisa/core.sio`, `stdlib/eisa/core_v2.sio` | Selected correction representation and origin contract. |
| Tests and comparisons | `stdlib/test/helpers.sio` | Predicate implementation and task-specific oracle. |

For additional families, use the edition's
[library organization table](../spec/2026-10/07-library-packages-and-tools.md#2-library-organization).
Names in an inventory are research leads, not promised import surfaces.

## 2. Bind a namespace to the actual source file

The selective import in the cookbook's provider/consumer pair names a public
declaration. The stdlib examples name their submodules directly, such as
`core::result`, `collections::vec`, or `epistemic::knowledge`. An existing
`lib.sio` or `mod.sio` is useful to inspect, but its existence is not proof of
re-exporting everything below it.
[`stdlib/math/lib.sio`](../../stdlib/math/lib.sio), for example, directs callers
to submodules; it does not list a public facade that exports every math symbol.

For the modular frontend, read
[`module_frontend_resolve_import_relpath`](../../self-hosted/compiler/module_frontend.sio).
The inspected path tries local and ancestor-relative candidates, a repository
relative candidate, `SOUNIO_STDLIB_PATH`, and the checkout's `stdlib/`, with
additional fallbacks in the source. This is one inspected implementation path,
not a universal resolver algorithm for every engine. The separate
[`module_loader.sio`](../../self-hosted/compiler/module_loader.sio) contains
another `resolve_import_file_path` implementation.

This order matters in practice: a local module can be selected before a similarly
named stdlib file. Record the **file actually resolved**, not just the namespace
you intended. A stale environment override is another possible source mismatch.

Useful read-only inspection commands:

```bash
rg -n '^module |^use |^pub (struct|enum|fn)' stdlib/core/result.sio
rg -n '^use core::result' tests examples
rg -n 'module_frontend_resolve_import_relpath|SOUNIO_STDLIB_PATH' self-hosted/compiler/module_frontend.sio
rg -n 'SOUNIO_STDLIB_PATH|SOUNIO_SOUC_ENGINE' bin/souc bin/madaros
```

For local packages, follow
[the package resolution contract](../spec/2026-10/07-library-packages-and-tools.md#4-local-packages)
and `docs/compiler/PACKAGE_IMPORT_RESOLUTION.md`. Do not turn the package
root's underscore-to-hyphen mapping into a blanket renaming of every segment.

## 3. Read a signature as a caller contract

For each operation, extract:

| Part | What the caller must preserve |
|---|---|
| Export | An ordinary declaration and a public importable declaration differ. |
| Parameter representation | Scalar, fixed array, record, raw pointer, callback, and sequence are distinct. |
| Reference | `&T` reads through shared access; `&!T` permits exclusive mutation under its contract. |
| Callback | Arity, argument/result types, and declared effects must fit the receiving API. |
| Return | A payload, tag, status, final time, count, and optional value have different meanings. |
| Effects | Keep the exact declared effect list while reproducing a source witness. |
| Body | Inspect capacity checks, sentinel values, early returns, domain handling, and dependencies. |
| Consumer | Establish whether the test actually imports, calls, and checks this operation. |

Effects belong to the selected contract. Local rebinding and mutation through a
reference are not interchangeable evidence for a universal `Mut` requirement.
See [effects and handlers](../spec/2026-10/03-effects-handlers-and-concurrency.md).
An omitted `with` in a copied declaration below records that declaration; it
does not independently prove runtime purity or absence of checker gaps.

The following sections provide a small set of **source-inspected signatures**.
Every block is nonrunnable text: declarations have no bodies here and must not
be pasted into an executable file as a complete example.

<a id="concrete-results-core-result"></a>

## 4. Concrete results: core::result

Source: [`stdlib/core/result.sio`](../../stdlib/core/result.sio).
Consumer: [`test_result_stdlib.sio`](../../tests/stdlib/core/test_result_stdlib.sio),
complete in [cookbook lesson 8](cookbook.md#8-import-a-provider-and-handle-a-concrete-result).

The public `IntResult` record declares `tag: i64`, `ok_val: i64`,
and `err_val: i64`; these field declarations do not themselves carry `pub`.
`FloatResult` changes `ok_val` to `f64`.
Constructors encode success and error in the tag. Accessors take a shared
reference; they do not consume the record.

Nonrunnable signature transcription:

```text
pub fn int_ok(v: i64) -> IntResult
pub fn int_err(e: i64) -> IntResult
pub fn int_result_is_ok(r: &IntResult) -> bool
pub fn int_result_is_err(r: &IntResult) -> bool
pub fn int_result_unwrap(r: &IntResult) -> i64
pub fn int_result_unwrap_or(r: &IntResult, default: i64) -> i64
pub fn int_result_err(r: &IntResult) -> i64
pub fn float_ok(v: f64) -> FloatResult
pub fn float_err(e: i64) -> FloatResult
pub fn float_result_is_ok(r: &FloatResult) -> bool
pub fn float_result_unwrap(r: &FloatResult) -> f64
```

**Behavior that matters.** `int_result_unwrap` returns `ok_val` directly.
It does not check the tag or panic. Use the predicates or the explicit fallback
accessor according to the task. Likewise, a nonzero error payload is not by
itself the record's discriminant. The source also publishes short free-function
aliases for `IntResult`; reproduce an existing caller before changing a free
call into method syntax.

Do not replace this API with imagined `ok`/`value` fields or infer that every
Sounio result is monomorphic. The language's generic result surface and these
specific library records have distinct evidence.

<a id="containers-choose-a-representation"></a>

## 5. Containers: choose a representation

| Representation | Inspected identity | Practical boundary |
|---|---|---|
| Fixed array `[T; N]` | Language array type; cookbook lessons 2–4 | Length is part of its type; bounds and stores require the selected execution path. |
| `VecF64` | Public record in `stdlib/collections/vec.sio` | Inline `[f64; 256]` plus logical length, fixed capacity. |
| `IntVec`, `FloatVec` | Separate non-`pub` records in that file | Do not assume public selective import just because internal method calls exist. |
| `HeapI64Vec` | Public record in `stdlib/collections/heap_vec.sio` | Heap pointer, logical length, capacity; inspect explicit allocation and release. |
| `HeapVec<T>` | Generic non-`pub` record in that file | Source backing is `*mut i64` with fixed-width operations; arbitrary element-layout support is not established. |
| `Seq<T>` | Compiler-facing sequence surface in repository fixtures | Locate a matching fixture; not an alias for the inline or heap library records. |
| A guessed `Vec<T>` | No identity established by this index | Resolve an actual declaration or special compiler surface before using the name. |

For `Seq<T>`, an existing discovery lead is
[`seq_knowledge_nested_generic.sio`](../../tests/run-pass/seq_knowledge_nested_generic.sio).
Read the complete fixture and its compiler path before selecting it; this page
does not supply a new sequence program or a support certificate.

The inspected `VecF64` source has a stale introductory comment saying capacity
`1024`; the actual field and constant both select `256`. Use the declaration,
not the comment, when planning bounds.

Nonrunnable `VecF64` signature transcription:

```text
pub fn vec_new() -> VecF64
pub fn vec_push(v: &!VecF64, value: f64) with Mut
pub fn vec_pop(v: &!VecF64) -> f64 with Mut
pub fn vec_get(v: &VecF64, idx: i64) -> f64
pub fn vec_set(v: &!VecF64, idx: i64, value: f64) with Mut
pub fn vec_len(v: &VecF64) -> i64
pub fn vec_sum(v: &VecF64) -> f64 with Mut, Div
pub fn vec_mean(v: &VecF64) -> f64 with Mut, Div, Panic
pub fn vec_reverse(v: &!VecF64) with Mut, Div
```

**Behavior that matters.** A full `vec_push` leaves the vector unchanged;
out-of-range `vec_get` returns `0.0`; an empty `vec_pop` returns `0.0`;
an invalid `vec_set` leaves the vector unchanged. A numeric zero cannot tell
the caller whether an element existed. Check length and index validity before
using these sentinel-returning operations in a task that must distinguish
absence from zero.

Source: [`vec.sio`](../../stdlib/collections/vec.sio).
Complete consumer: [cookbook lesson 9](cookbook.md#9-use-the-vector-representation-the-source-actually-exports).

For heap storage, inspect
[`heap_vec.sio`](../../stdlib/collections/heap_vec.sio) and consumers such as
[`test_heap_vec_methods.sio`](../../tests/stdlib/collections/test_heap_vec_methods.sio).
Its header identifies a native-only route; that is source-stated intent, not a
fresh cross-target run. Do not infer automatic destruction, a portable allocator,
or arbitrary-`T` storage merely from the word “vector.”

<a id="scientific-information-keep-the-representations-separate"></a>

## 6. Scientific information: keep the representations separate

| Surface | Identity at the selected source | Do not silently substitute |
|---|---|---|
| Compiler `Knowledge<T>` | Language construct with checker/compiler treatment | An ordinary record with similar field names. |
| Library `Epistemic` | `stdlib/epistemic/knowledge.sio`: `val: f64`, `variance: f64`, `confidence: i64` | Generic compiler `Knowledge<T>` or a provenance-carrying type inferred from the filename. |
| EISA `EReg` | `stdlib/eisa/core.sio`: `val: f64`, `err: Dd64`, `u: f64`, `origin: i64` | A single uncertainty field or a plain `f64`. |
| EISA `EReg2` | `stdlib/eisa/core_v2.sio`: `val: f64`, `err: Qd128`, `u: f64` | `EReg` with an assumed identical origin policy or correction representation. |
| GUM component/result records | `stdlib/epistemic/gum.sio` | Compiler `Knowledge<T>` or a general scientific-validity certificate. |

The meaning to preserve is the difference between a value, computational
correction, measurement uncertainty, confidence, and provenance. See
[the founder contract](../../FOUNDER_INTENT.md) and
[the epistemic specification](../spec/2026-10/04-knowledge-uncertainty-and-provenance.md).

### Library Epistemic

Source: [`epistemic/knowledge.sio`](../../stdlib/epistemic/knowledge.sio).
The complete consumer
[`test_knowledge_madaros_import_e2e.sio`](../../tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio)
is copied in cookbook lesson 10. Its comments explicitly separate free-function
coverage from method coverage and from builtin `Knowledge<T>`.

Nonrunnable signature transcription:

```text
pub fn ep_new(val: f64, variance: f64, confidence: i64) -> Epistemic
pub fn ep_certain(val: f64) -> Epistemic
pub fn ep_measured(val: f64, std_dev: f64) -> Epistemic
pub fn ep_val(e: &Epistemic) -> f64
pub fn ep_variance(e: &Epistemic) -> f64
pub fn ep_confidence(e: &Epistemic) -> i64
pub fn ep_std(e: &Epistemic) -> f64 with Mut, Div, Panic
pub fn ep_add(a: &Epistemic, b: &Epistemic) -> Epistemic
pub fn ep_add_cov(a: &Epistemic, b: &Epistemic, cov: f64) -> Epistemic with Panic
pub fn ep_mul(a: &Epistemic, b: &Epistemic) -> Epistemic
pub fn ep_gate(e: &Epistemic, min_conf: i64) -> i64
```

**Behavior that matters.** `ep_new` accepts a variance; `ep_measured`
accepts a standard deviation and its body stores the corresponding variance.
They are different inputs. The source documents that the ordinary addition
API assumes zero covariance, while a separate covariance-taking operation
exists. Do not use a no-covariance call as a general solution for dependent
measurements. The integer confidence convention and gate are library behavior,
not a calibrated clinical probability.

This page does not endorse every propagation formula or validate a new
measurement problem. Scientific changes need their own review and numerical
evidence under the user's permitted calculation tools.

### EISA origin and correction lanes

Sources: [`eisa/core.sio`](../../stdlib/eisa/core.sio) and
[`eisa/core_v2.sio`](../../stdlib/eisa/core_v2.sio).

Nonrunnable signature transcription for `eisa::core`:

```text
pub fn ereg_exact(v: f64) -> EReg
pub fn ereg_measured(v: f64, sigma: f64) -> EReg
pub fn ereg_origin(v: f64, sigma: f64, origin: i64) -> EReg
pub fn ereg_true(x: EReg) -> Dd64
pub fn eisa_fuse_origin(a: i64, b: i64) -> i64
pub fn eisa_origins_correlated(a: i64, b: i64) -> i64
pub fn eadd(x: EReg, y: EReg) -> EReg with Mut, Div
```

**Behavior that matters.** The inspected constructor reserves origin zero as
unknown/mixed and normalizes invalid origin IDs to that sentinel. The selected
correlation predicate recognizes matching nonzero origins; it is not an
arbitrary covariance matrix. The ordinary `ereg_measured` constructor uses the
unknown-origin convention. Copying it twice does not establish tracked common
origin. Keep origin selection explicit for the scientific question.

The complete distinct-origin consumer is
[`eisa_origin_independent_add.sio`](../../tests/run-pass/eisa_origin_independent_add.sio),
copied in cookbook lesson 10. Its comments and existing comparisons describe
its own intended control; no new numerical calculation was performed here.

### GUM source API and legacy names

Source: [`epistemic/gum.sio`](../../stdlib/epistemic/gum.sio).
Read `GUMComponent`, `GUMResult`, and their operation bodies together.
The file separately retains a backward-compatibility `GUMUncertainty` record.
The existence of that legacy record does not give every imagined GUM helper
a matching implementation.

Nonrunnable signature transcription:

```text
pub fn gum_type_a(std_dev: f64, n: i64) -> GUMComponent with Mut, Div, Panic
pub fn gum_type_b(std_u: f64) -> GUMComponent
pub fn gum_type_b_uniform(half_width: f64) -> GUMComponent
pub fn gum_type_b_expanded(expanded_u: f64, k: f64) -> GUMComponent with Div, Panic
pub fn gum_with_sensitivity(c: GUMComponent, s: f64) -> GUMComponent
pub fn gum_combine2(value: f64, u1: GUMComponent, u2: GUMComponent) -> GUMResult with Mut, Div, Panic
pub fn gum_std_u(r: GUMResult) -> f64
pub fn gum_u95(r: GUMResult) -> f64
```

**Behavior that matters.** A standard uncertainty, a uniform half-width, and
an expanded uncertainty with coverage factor are different constructor inputs.
The inspected `gum_type_a` has branches for nonpositive sample counts, and
`gum_type_b_expanded` has a branch for a nonpositive factor; their existence
does not authorize scientifically invalid input. Review the accepted domain
and desired refusal policy before adapting.

A discovery consumer is
[`test_gum_stdlib.sio`](../../tests/stdlib/epistemic/test_gum_stdlib.sio).
Its numerical coverage remains limited to the operations and assertions it
actually executes on a named profile. Neither a “GUM” filename nor a green
smoke marker is a general compliance result.

## 7. Mathematical functions and ODE callback contracts

### Public math::pure functions

Source: [`math/pure.sio`](../../stdlib/math/pure.sio).
Complete import consumer:
[`native_math_via_stdlib_import.sio`](../../tests/run-pass/native_math_via_stdlib_import.sio),
copied in cookbook lesson 10.

Nonrunnable signature transcription:

```text
pub fn atan(x: f64) -> f64 with Mut, Div, Panic
pub fn asin(x: f64) -> f64 with Mut, Div, Panic
pub fn acos(x: f64) -> f64 with Mut, Div, Panic
```

Read implementation branches before selecting an input domain. The fixture
compares specific existing values using its own existing window; it is not an
accuracy envelope for every input, precision, or backend. A library function
and a similarly named native builtin are separate paths.

### The actual ode::rk4 callback shape

Source: [`ode/rk4.sio`](../../stdlib/ode/rk4.sio).
Consumer lead:
[`test_rk4_fn_ref_e2e.sio`](../../tests/stdlib/ode/test_rk4_fn_ref_e2e.sio).

Nonrunnable signature transcription, retaining the complete callback contract:

```text
pub fn rk4_step(
    rhs: fn(f64, &[f64; 64], &![f64; 64]) -> () with Mut, Div, Panic,
    t: f64,
    y: &![f64; 64],
    dt: f64,
    n_vars: i64
) -> () with Mut, Div, Panic

pub fn rk4_integrate(
    rhs: fn(f64, &[f64; 64], &![f64; 64]) -> () with Mut, Div, Panic,
    y: &![f64; 64],
    n_vars: i64,
    t0: f64,
    t_end: f64,
    dt: f64
) -> f64 with Mut, Div, Panic
```

**Behavior that matters.** This API mutates caller-provided state arrays and
uses a derivative-output buffer. The callback is not a scalar
`fn(u, t) -> f64`, and `rk4_integrate` returns a time value rather than a
guessed solution record. Inspect its state-length clamping and step/time control
before choosing inputs; the signature alone does not establish safe progress,
stability, or accuracy.

The consumer is a discovery lead, not an endorsed template for every effect
list or new tolerance. Its actual declarations and output calls must be
checked on the selected profile before adaptation.

## 8. Text, byte buffers, and structured data

These small signature samples demonstrate why representations must be read
before generating import calls.

Source: [`str/lib.sio`](../../stdlib/str/lib.sio).
Nonrunnable signature transcription:

```text
pub fn str_new() -> Str
pub fn str_to_string(s: Str) -> string with Mut, Div, Panic
```

`Str` and language `string` are distinct here. Inspect the record's byte
storage, length, and conversions; do not assume a Unicode string abstraction
or unrestricted growth.

Source: [`io/io.sio`](../../stdlib/io/io.sio).
Nonrunnable signature transcription:

```text
pub fn io_buf_new() -> IoBuffer
pub fn io_buf_write(buf: IoBuffer, data: &[u8; 256], n: i32) -> IoBuffer
pub fn io_buf_read(buf: &!IoBuffer, out: &![u8; 256], n: i32) -> i32 with Mut, Panic, Div
```

The write takes and returns a buffer value; the read receives exclusive
references and returns a count. Their signatures do not describe the same
ownership pattern or an operating-system file API.

Source: [`json/parser.sio`](../../stdlib/json/parser.sio).
Nonrunnable signature transcription:

```text
pub fn json_doc_new() -> JsonDoc
pub fn json_doc_count(doc: &JsonDoc) -> i32
pub fn json_parse(input: &[u8; 4096], input_len: i32, doc: &!JsonDoc) -> i32 with Mut, Div, Panic
```

Inspect how the parse status, populated document, lengths, and error branches
relate. `stdlib/json/value.sio` also declares a `JsonValue`; same spelling
across files does not establish the same representation or import surface.
A consumer lead is
[`test_json_e2e.sio`](../../tests/stdlib/json/test_json_e2e.sio); inspect its
actual imports before combining it with the parser signatures.

## 9. Test helpers do not choose your numerical oracle

Source: [`test/helpers.sio`](../../stdlib/test/helpers.sio).
Nonrunnable signature transcription:

```text
pub fn check_near(a: f64, b: f64, tol: f64) -> bool
pub fn check_near_relative(a: f64, b: f64, rtol: f64) -> bool
pub fn check_eq_i64(a: i64, b: i64) -> bool
pub fn check_in_range_i64(x: i64, lo: i64, hi: i64) -> bool
```

`check_near` uses a strict absolute-difference comparison. A helper's existence
does not choose a justified tolerance, solve a units problem, or provide an
independent reference. The inspected `check_near_relative` body selects between
the signed inputs for its variable named `max_abs`; do not paraphrase its name
as a verified maximum-absolute-value contract. Signed inputs and near-zero
behavior require their own review.

Do not mandate an invented test-file format, one assertion per function, or a
universal minimum effect list from these helpers. Existing tests use different
entry-point and failure-reporting patterns. Preserve an actual consumer's
behavior, then choose assertions appropriate to the problem.

A concrete warning about test names:
[`test_descriptive_e2e.sio`](../../tests/stdlib/stats/test_descriptive_e2e.sio)
only prints `STATS_DESCRIPTIVE_OK` and returns zero in this baseline. It imports
no statistics module and invokes no descriptive-statistics operation. Even a
successful execution of that file would establish its marker, not the import
surface or numerical behavior of `stats::descriptive`.

## 10. Validate the selected API, then report exactly what happened

Choose a complete consumer, not a library file, for an execution. The cookbook's
`cook-result` and `cook-vec` are complete imported consumers with explicit
success/failure paths. The provider in `cook-import-provider` is deliberately
check-only because it has no entry point.

Reproduction commands; the [actual results](example-results.json) record the
consumer stages that were executed on the pinned profile. A recorded run does
not separately witness the displayed public check command:

```bash
./bin/souc check tests/stdlib/core/test_result_stdlib.sio
./bin/souc run tests/stdlib/core/test_result_stdlib.sio
./bin/souc check tests/stdlib/collections/test_vec_stdlib.sio
./bin/souc run tests/stdlib/collections/test_vec_stdlib.sio
```

Lightweight check/run commands may use any authorized environment with an
identified, host-compatible compiler profile. Session or repository policy may
require a remote route; this guide does not authorize compiler execution on the
current Mac or any other unapproved host. Full stress validation belongs on the
project's remote Compiler Foundry or Slurm route, subject to that project's
policy. A public learner does not need this project's cluster to perform an
otherwise authorized lightweight check/run. Attach the resolved artifact
identity. An engine override, rebuilt wrapper, bootstrap seed, native
target, and installed host compiler each require their own attribution.
Do not silently switch profiles to make an import succeed.

For an adapted program, write the expected observations before running:
what operation is reached, which branch demonstrates failure, what stdout
marker is required, and which process result is intended. Some complete
native-runtime fixtures intentionally return `42`; zero is not a universal
expected exit. A diagnostic refusal also needs its identifying message and
deciding stage.

Report the result with this **nonrunnable text** record:

```text
Input problem:
Source revision and modified paths:
Complete example ID, consumer file, and auxiliary files:
Resolved import namespace -> actual source file:
Symbol, concrete representation, and source declaration:
Expected result, failure branches, and stdout/stderr assertions:
Requested launcher command:
Resolved raw engine, adapter, artifact hash, target, and host:
Selected library root and relevant overrides:
Actual stage reached:
Actual stdout/stderr and exit:
Receipt path:
Remaining unknowns:
```

Until such a receipt exists, the accurate conclusion is “declaration and caller
inspected; check/run unrun.” A later check-only acceptance must not be relabeled
as execution, numerical accuracy, proof, or domain validation. Use the
[conformance reporting chapter](../spec/2026-10/09-conformance-diagnostics-and-compatibility.md)
for the full stage distinctions.

<!-- docs:meta
topic_id: repo.docs.llm-guide.error-catalog
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.error-catalog
-->

# Diagnose and repair Sounio failures

Read a failure as **engine + adapter + stage + complete message + code + source
revision**. The same number can identify different failures on different
routes. An editor's synthesized diagnostic is not necessarily a compiler code.

This guide inspects source at
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. It does not run the listed
fixtures, certify every guard, or establish a global error-code namespace.
This is a source-inspected edition; structural checks and execution receipts
remain separate. The [diagnostics chapter](../spec/2026-10/09-conformance-diagnostics-and-compatibility.md)
defines the evidence boundaries. Use the [complete example atlas](examples.md)
for full positive programs and intentional refusal witnesses, with their
manifest/profile qualifications.

## Start with the deciding route

| Route | What actually decides the result | First source to inspect |
| --- | --- | --- |
| Public default `bin/souc check` | Madaros resolution followed by the selected wrapper/raw artifact; the modular wrapper requests raw `--check`. | [bin/souc](../../bin/souc), [bin/madaros](../../bin/madaros), [main.sio](../../self-hosted/compiler/main.sio), and the artifact/build receipt. |
| Explicit `lean_single` through `bin/souc check` | The adapter invokes the seed's positional compiler with a temporary output. This can reach native compilation. | [bin/souc](../../bin/souc) forced-engine branch and [lean_single.sio](../../self-hosted/compiler/lean_single.sio). |
| Modular parser compatibility refusal | The parser can emit a code and recover enough to find more errors. Recovery does not make the source valid. | [stmts.sio](../../self-hosted/parser/stmts.sio), [types.sio](../../self-hosted/parser/types.sio), and [exprs.sio](../../self-hosted/parser/exprs.sio). |
| Modular type/effect/ownership refusal | The particular checker entrypoint reached from preflight; inspect its caller as well as the message table. | [check.sio](../../self-hosted/check/check.sio), especially `print_error_message`, dedicated reporters, and call sites. |
| Native lowering, emission, linking, or execution | The requested backend and target after any frontend checks; a different stage can expose an independent defect. | [main.sio](../../self-hosted/compiler/main.sio), [lower.sio](../../self-hosted/ir/lower.sio), the named backend, and fresh output. |
| MCP/editor presentation | Normalization and fallback parsing can alter the code, span, severity view, or classification. | [MCP diagnostics](../../tools/mcp/sounio_mcp/diagnostics.py), [check adapter](../../tools/mcp/sounio_mcp/check.py), and [LSP text parser](../../tools/lsp/parse_diagnostics.sh). |

The [canonical test suite](../../scripts/run_sio_test_suite.sh), its
[implementation](../../scripts/dev/run_sio_test_suite.sh), the
[conformance runner](../../scripts/ci/serious_language_conformance_gate.sh), and
[MCP test tool](../../tools/mcp/sounio_mcp/test.py) are different dispatchers.
In the conformance runner both `check-fail` and `compile-fail` call `check`.
MCP also checks its recognized negative cases. These labels alone establish
neither a parser-only failure nor a native emission attempt. A library module
without `main` should normally be checked or exercised through its driver;
requesting execution can create an unrelated entrypoint failure.

### Preserve this evidence before editing

1. Record source path/hash, imports, revision/dirty source state, compiler
   artifact/hash/build receipt, engine, adapter, target, and stdlib.
2. Save the exact invocation, exit status, and both complete streams. Inspect
   messages before the final “preflight failed” or “type checking failed”
   summary; the summary usually names a consequence.
3. Identify the earliest cause and its stage. Verify the enclosing module and
   callee declaration; a source span without file identity may be insufficient.
4. Reproduce on the same profile and make one justified edit. Keep the
   requirement, expected output, and semantic invariants fixed.
5. Recheck, then verify the requested behavior or named refusal. Acceptance by
   another engine is separate comparative evidence. Stop and retain a
   reproducer for an unsupported operation or compiler defect.

Do not remove effects, type annotations, constraints, uncertainty, or provenance
to make the diagnostic disappear. Do not lower precision, change algebra,
weaken expected results, or disable a checker. A negative test must fail for
the intended cause; missing tools, timeouts, unsupported imports, and crashes
do not satisfy its semantic refusal contract.

## Common syntax mistakes and misleading old advice

Use [syntax and modules](../spec/2026-10/01-lexical-syntax-and-modules.md)
and the actual parser before turning a historical workaround into a rule.

| Symptom | Genuine cause to investigate | Safe repair and existing source candidate |
| --- | --- | --- |
| `let mut` rejected | Mutable bindings use `var`; the compatibility parser recognizes the foreign spelling to diagnose it. | Change binding syntax while preserving its type and intended mutability. Inspect [parser_rust_let_mut.sio](../../tests/compile-fail/parser_rust_let_mut.sio). |
| `&mut T` rejected | Exclusive references use `&!T`; a shared `&T` cannot replace an exclusive capability. | Match parameter and call-site borrow to the required capability. Inspect [parser_rust_mut_ref.sio](../../tests/compile-fail/parser_rust_mut_ref.sio). |
| Macro-like `ident!(...)` rejected | A foreign macro call was written instead of an ordinary declared function call. | Look up the real function signature; deleting `!` does not create a missing API. Inspect [parser_rust_macro.sio](../../tests/compile-fail/parser_rust_macro.sio) and [parser_rust_macro_path_qualified.sio](../../tests/compile-fail/parser_rust_macro_path_qualified.sio). |
| `#[...]` rejected | The Rust-attribute compatibility refusal does not define Sounio annotation syntax. | Use only the annotation form supported for that item. Inspect [parser_rust_attribute.sio](../../tests/compile-fail/parser_rust_attribute.sio). |
| Unexpected unit or tail result | A final expression, explicit return, or branch differs from the declared result. | Compare every return path. “No semicolons anywhere” is false: array syntax contains `;` and the modular statement parser handles optional semicolons. Do not remove punctuation mechanically. |
| Failure near unary minus | Operand type, parser route, or lowering may be at fault. | `parse_prefix` in [exprs.sio](../../self-hosted/parser/exprs.sio) handles unary minus. Inspect [kl1_unary_type_errors.sio](../../tests/compile-fail/kl1_unary_type_errors.sio); do not impose a language-wide `0 - x` rewrite. |
| Failure near a closure or generic application | Specialization, captures, effect inference, a higher-order call, or representation may be unsupported. | Current source has generics and closures. Compare [generic_struct_basic.sio](../../tests/run-pass/generic_struct_basic.sio), [generics_multi_param.sio](../../tests/run-pass/generics_multi_param.sio), and [closure_effect_infer_auto.sio](../../tests/run-pass/closure_effect_infer_auto.sio); preserve profile boundaries. |
| Compound assignment rejected by the seed | A seed compatibility limit differs from the modular assignment parser. | Inspect the actual `E260` message. A profile-specific explicit assignment may be appropriate; it is not a ban throughout Sounio. |

The old local headings `E01`, `E02`, and so on were tutorial mistake numbers,
not stable compiler identities. Descriptive sections avoid confusing them with
actual `E001` and similar codes.

## Types, names, calls, and returns

These are **modular checker message identities** from `print_error_message`,
not universal meanings across engines.

| Diagnostic | Cause | Repair that preserves the contract |
| --- | --- | --- |
| `E001`, `E002`, `E003` | Binding mismatch, assignment mismatch, or mutation of an immutable binding. | Trace declared and inferred types. Use `var` only for intended mutation; do not widen types merely to suppress a mismatch. Check whether `E001` was synthesized first. |
| `E004`, `E005`, `E048`–`E051` | Operator incompatible with its operands. | Select an operation defined for the types. An `as` cast is not a universal conversion and must not erase epistemic or precision information. |
| `E006`, `E007`, `E008`, `E009`, `E010`, `E018` | Non-boolean condition, differing branch types, wrong return, argument type, arity, or match-arm result. | Compare the declaration with the entire call or branch. Fix argument order/count and the wrong expression; do not change the function contract by guesswork. |
| `E011`–`E017`, `E020`, `E046`, `E047` | Wrong receiver, member, index, aggregate field, element type, or tuple position. | Read the complete definition and initializer. The modular index message requires an integer; it does not mandate `usize` for every index. |
| `E175`–`E177`, `E259` | Access through a private function, constructor, or field boundary. | Use the public API or request an authorized API change. Renaming a qualifier does not grant visibility. |
| `E213`, `E216` | Tuple destructuring arity mismatch or an infinitely recursive type. | Match the tuple shape; for recursion, inspect permitted representation/indirection. Do not invent a supported heap abstraction. |
| `E263` | A qualified call's module name collides with an unrelated method identity. | Inspect imports and declarations; use an existing unambiguous API or report the boundary. More qualification is not an automatic fix. |
| `E264` | The selected native representation cannot preserve a payload-bearing enum use. | Preserve data and meaning. Change representation only through an authorized review, or report an unsupported-profile result. |
| `E270` | `as string` is being used as numeric/bool/char formatting. | Use a real formatting or typed-printing API after inspecting its implementation/profile support. Do not reinterpret a scalar as a string pointer. |

Existing negative witnesses include
[condition_not_bool.sio](../../tests/ui/type/condition_not_bool.sio),
[if_branch_mismatch.sio](../../tests/ui/type/if_branch_mismatch.sio),
[array_index_type.sio](../../tests/ui/type/array_index_type.sio),
[tuple_destructure_arity_mismatch.sio](../../tests/compile-fail/tuple_destructure_arity_mismatch.sio),
and [cast_float_as_string_print_refused_issue_2354.sio](../../tests/compile-fail/cast_float_as_string_print_refused_issue_2354.sio).
Read annotations; source presence is not a fresh passing result.

An unresolved name is not evidence that forward declarations are prohibited.
Read import loading, spelling, visibility, receiver type, and the call signature.
The seed's `E200` undefined-identifier identity collides with the modular
message for a `Forgettable` effect obligation.

## Effects: repair the obligation, not the symptom

`E035` is actively represented in modular source. The checker includes
`report_effect_error_from`, `check_callee_effects`, and dedicated mutation/effect
checks; effect-name admission is also checked. The July 2026 claim that
default `check` never enforces effects is historical, not current guidance.
The [bounded conformance receipt](../spec/2026-10/15-bounded-conformance-receipt.md)
records a separately executed profile and an effects-negative case. That
receipt does not establish every trigger on every installed binary.

Read the diagnostic's enclosing function, `missing:` list, and `required by`
callee when present. Compare the callee's actual effect row with the caller's
permitted row, including function-valued arguments and closures. Add only the
required, justified effects at the correct boundary and propagate obligations
to affected named callers. Moving an operation to a permitted effectful boundary
is an alternative only when it preserves the task's behavior.

Mutation generation has operation-specific rules. The checker introduces `Mut`
for a store through an exclusive reference; that does not justify the old
blanket claim that every local `var` declaration/reassignment requires `Mut`.
Inspect [effect_mut_generated_at_excl_ref_store.sio](../../tests/compile-fail/effect_mut_generated_at_excl_ref_store.sio)
and the [effects chapter](../spec/2026-10/03-effects-handlers-and-concurrency.md).
Determine the effect from the actual builtin/checker path rather than assigning
`Panic` to every index or cast by analogy.

| Diagnostic or message | Cause and safe response | Existing source witness |
| --- | --- | --- |
| `E035`: missing effects | A checked body/call requires effects absent from its permitted row. Repair the named obligation and recheck callers. | [effect_missing.sio](../../tests/compile-fail/effect_missing.sio) and [effect_superset_ok.sio](../../tests/run-pass/effect_superset_ok.sio). |
| `E246`: unknown effect | A spelling is neither a supported builtin nor an admitted declared effect on this route. Verify the declaration/registry. | [effect_unknown_name_refused.sio](../../tests/compile-fail/effect_unknown_name_refused.sio). |
| `E246`: ninth distinct effect dropped | The selected representation has eight signature slots and refuses overflow. Do not drop necessary obligations or add a blanket “all effects” row. | [effect_ninth_slot_refused.sio](../../tests/compile-fail/effect_ninth_slot_refused.sio). |
| `E053`: closure effect uncertainty | A particular higher-order boundary cannot establish permitted effects. Inspect inference and the receiving function type. | [closure_effect_escape.sio](../../tests/compile-fail/closure_effect_escape.sio). |
| `E070` or `E253` in a kernel | Forbidden kernel effects or non-unit return on the inspected modular route. Move host operations to a justified host boundary. | [check.sio](../../self-hosted/check/check.sio) and [targets chapter](../spec/2026-10/06-execution-memory-and-targets.md). |
| `E201`–`E207` capability effects | A type-specific `ZD` or additional effect boundary is unsatisfied. Inspect the type/reporter before changing the signature. | [check.sio](../../self-hosted/check/check.sio) capability checks. |

Declaring an effect permits an operation at a typed boundary; it does not supply
an implementation, runtime witness, scientific guarantee, or proof. Diagnostic
prose about capability guarantees is not independent mathematical or security
evidence.

## Ownership, aliases, and mutation

The modular `E037`/`E038` messages distinguish an active exclusive borrow from
conflicting outstanding borrows. `E039` reports reuse of a linear value and
`E040` can report failure to consume it. Inspect control-flow paths and capture
behavior; copying the handle, adding a dummy use, or discarding it can violate
the resource contract.

`E267` refuses assignment through a shared reference, including relevant
auto-dereferenced field/index forms. Changing a shared parameter to exclusive
changes its API and caller obligations; do so only for intended and permitted
mutation. Scope or release a borrow using an existing supported pattern.

Start with paired
[borrow_call_conflict_explicit.sio](../../tests/compile-fail/borrow_call_conflict_explicit.sio)
and [borrow_call_explicit_release.sio](../../tests/run-pass/borrow_call_explicit_release.sio).
For mutation that compiles but is not observable, inspect value layout,
copying, aliasing, and the actual engine. Wrapping every array in a struct
was an interpreter-specific workaround, not a language rule or native guarantee.

`E232` names a shallow array-copy hazard when one local initializes two fields.
Preserve the required sharing or independence explicitly. Do not rewrite a
data structure without checking caller-visible behavior.

## Knowledge, uncertainty, provenance, and refinements

Read the [epistemic chapter](../spec/2026-10/04-knowledge-uncertainty-and-provenance.md)
before modifying these contracts. A convenient scalar replacement can erase
the very information the program must preserve.

| Diagnostic or family | What to establish | Safe response |
| --- | --- | --- |
| `E170`, `E171` | Which extraction/access or cast crosses the epistemic boundary on this route. | Keep `Knowledge<T>` through computations where possible. Use an implemented, justified extraction boundary with required `Epistemic`; never cast away the wrapper. |
| `E036`, `E214`, `E215` | The actual confidence/completeness or observation requirement for that engine. | Repair missing evidence or incorrect flow. Lowering a threshold or inventing confidence changes the requirement. |
| `E230`, `E236` | Shared noise/source lineage or a reused correlated binding, and the operation's independence assumption. | Preserve correlation and variance. Do not duplicate measurements, strip wrappers, or assert independence. `Correlated` is not a numerical implementation or covariance proof. |
| `E178`, `E179` | Noise-source or source-set tracking capacity was exhausted. | Retain the refusal and investigate the supported workload/representation. Disabling tracking changes the claim. |
| `E243` | A required compile-time correlation coefficient was supplied as an unsupported runtime expression. | Use evidence-supported metadata in a supported representation; substituting zero asserts a different model. |
| `E244`, `E265` | Argument provenance or temporal validity fails the parameter's requirement. | Repair custody/validity at its real source or supply a satisfying value. Renaming a provenance tag/date is not evidence. |
| Modular `E042`, `E052`; seed `E208`/`E209` | A refinement is violated or cannot be established. | Inspect the predicate, supported guard/proving path, and exact caller/return. Do not weaken the predicate to accept an invalid value. |
| `E269` | The profile refuses a `where` refinement on `Knowledge<T>` because it would otherwise be ignored. | Both satisfying and violating values can be refused for unsupported enforcement. Preserve the predicate and report the boundary. |

Existing sources include
[knowledge_no_silent_unwrap.sio](../../tests/compile-fail/knowledge_no_silent_unwrap.sio),
[ns_add_shared_source_rejected.sio](../../tests/compile-fail/ns_add_shared_source_rejected.sio),
[ns_add_via_identity_rejected.sio](../../tests/compile-fail/ns_add_via_identity_rejected.sio),
and paired
[valid-value E269 refusal](../../tests/compile-fail/knowledge_where_refinement_refused_valid_2753.sio)
and [invalid-value E269 refusal](../../tests/compile-fail/knowledge_where_refinement_refused_invalid_2753.sio).
These are source-selected candidates, not calculations performed here.

`E245` is a separate modular boundary for arithmetic between two
`Knowledge<T>` values. Adding effects or repairing lineage does not supply
missing lowering. Inspect
[knowledge_binary_arith_has_no_lowering.sio](../../tests/compile-fail/knowledge_binary_arith_has_no_lowering.sio)
and its engine requirement. A seed implementation or a library struct named
`Epistemic` is a different route and must not be substituted silently.

For `.value`, inspect the actual receiver type and extraction path. A plain
library struct field does not automatically acquire the builtin `Knowledge`
gate because its field has the same name. Absence of a diagnostic also does
not prove that epistemic meaning survived lowering.

## Precision, algebra, unsupported operations, and resources

A refusal can protect against silent corruption. Treat these as operation-specific
implementation limits until the requested path has evidence.

| Diagnostic | Interpretation on the inspected route | Safe next step |
| --- | --- | --- |
| `E247` | Invalid encoded `ZD` sedenion-pair locus shape. | Correct the malformed locus from its declaration/source convention; preserve the algebra. |
| `E248` | A particular wide-float value conversion lacks implementation. | Preserve required `f128`/`f256` identity and report the conversion boundary. Inspect [f128_cast_to_f64_unimplemented.sio](../../tests/compile-fail/f128_cast_to_f64_unimplemented.sio). |
| `E249` | Retained V0-A reservation identity, with no live modular parser emitter in the inspected V0-B source. | Inspect the emitted diagnostic and specific operation/profile. Do not infer that all wide types are unavailable. |
| `E250` | A called `extern "C"` function lacks implementation on the selected native path. | Distinguish declaration from call. Use only an implemented call or report missing backend integration. |
| `E251`, `E252` | Refused algebra-dependent variance shortcut or unjustified reassociation strategy. | Preserve variance, multiplication law, order, and parenthesization. Review the model/implementation contract. |
| Modular `E221` | A math function is bound for checking but lacks native emission. | Inspect the backend and an existing driver; frontend acceptance is not execution. |
| Seed/codegen `E217`–`E223` | Body-span, return/pass-layout, unresolved-call, entrypoint, or capacity failures in published identities. | Check syntax/declarations and the deciding backend. Internal invariants need a reproducer and compiler repair. |
| `E224`–`E229`, `E258`, explicit overflow | Import/readability, source/token/name capacity, or truncation refusal on the emitting route. | Verify the real dependency and path. Arbitrarily shortening literals or identities changes the program. |

The [numeric specification](../spec/2026-10/05-numerics-units-and-algebra.md)
documents selected wide-operation paths. Several old negative fixtures for
V0-A `E249` are ignored as superseded: read their annotations before
presenting them as current requirements. Do not demote a precision-sensitive
task to `f64` to bypass a refusal.

For ontology/causal failures, inspect the actual graph/class and rule.
Modular `E041`, `E150`–`E152`, `E261`/`E262` name ontology obligations
on relevant paths; `E067`/`E068` and `E254`–`E257` concern causal-query
or independence boundaries where wired. A fabricated node or weakened graph
does not repair the original model. A compiler check is not empirical causality.

## Known identity collisions and remaps

Use the full message and emitting stage. These observations do not allocate
or rename codes.

| Code encountered | Distinct identities to distinguish |
| --- | --- |
| `E006`, `E007`, `E008` | Seed: arity / local capacity / global capacity. Modular: boolean condition / branch type / return type. |
| `E035` | Modular and seed effect reporters exist. The MCP resource titles it “Recursive type rejected,” which does not describe the current effect reporter. |
| `E036` | Seed observation-boundary identity versus modular confidence bound. |
| `E040` | Parser/seed `let mut` refusal versus modular linear-consumption checks. |
| `E041` | Parser/seed `&mut` refusal; modular ontology message-table identity; separate direct unit-mismatch prints also exist in `check.sio`. |
| `E042`, `E043` | Parser/seed attribute or macro refusal versus modular refinement or unit-conversion identity. |
| `E070`, `E072` | Historical kernel explanations coexist with other checker identities. Current modular kernel return refusal is `E253`; modular `E072` names a malformed conditional-independence claim. |
| `E200`, `E221` | Seed undefined identifier / missing main versus modular `Forgettable` effect obligation / missing native math implementation. |

The [remap gate](../../scripts/ci/diagnostic_remap_2170_gate.sh) retains seed
`E208`/`E217`/`E218`/`E219` and assigns the unrelated modular identities to
`E247`/`E248`/`E249`/`E250`. The old table pairing those as current collisions
was stale. `E249` is catalog-only in that gate: a message arm is not an emitter.

The [identity ratchet](../../scripts/ci/diagnostic_identity_gate.sh) scans
message tables, direct prints, and this page's published namespace rows.
Known remaining collisions are permitted up to explicit ceilings; a pass
does not prove that every code has one meaning.

## Diagnostic transport is not the compiler verdict

The MCP [check implementation](../../tools/mcp/sounio_mcp/check.py) requests
`check <source> --json`. Its [normalizer](../../tools/mcp/sounio_mcp/diagnostics.py)
accepts JSON diagnostics or scans stdout/stderr for lines containing “error”
or “warning.” A well-formed adapter envelope does not establish native JSON
support.

- JSON/LSP ranges are zero-based; MCP `line`/`column`/`span` are one-based.
  Preserve the original range, URI, and raw streams.
- Text fallback recognizes limited location wording and can use line 1,
  column 1. Offsets without a file identity are not exact editor locations.
- Missing paths, resolver failures, timeouts, and unclassified nonzero exits
  can become synthetic `E001`. A suggested type fix may be irrelevant.
- `valid` requires an empty diagnostic list and exit zero; a warning prevents
  this adapter's next step. A warning is not automatically a compiler error.
- Schema `sounio.diagnostic.v1` validates shape, not attribution, parser
  completeness, or checking-route coverage.
- Outer `isError: false` can accompany a failed tool result. Inspect
  `structuredContent`, `valid`/`status`, return code, and streams.
- `sounio_compile` tests nonempty output and no timeout when forming `status`;
  its success predicate does not require return code zero. Inspect that code,
  diagnostics, fresh artifact identity, and target format.
- `sounio_run` checks first, then separately calls `run`. A subsequent compile
  refusal is classified as `runtime_error` on nonzero exit; execution may
  never have begun.

The check result does not report a resolved raw compiler path/hash or stdlib
identity. Retrieve those through a separate receipt in the server's actual
environment. Moreover, `run.py` calls its initial check without the per-call
`env` overrides, then applies those overrides to the later `run` command.
Compiler/profile overrides can produce different checking and execution
profiles. Keep those settings identical server-side and record the environment
externally. If agreement cannot be established, label the identity unknown or
the profiles separate and repeat CLI check plus run on one known profile.
Normalized MCP diagnostics alone do not establish conformance.

The MCP [test tool](../../tools/mcp/sounio_mcp/test.py) reads only the first
20 lines of annotations and a subset of canonical metadata. It lacks full
`requires`, `known-failure`, and `typecheck-fail` admission semantics. An
`error-pattern` can route a fixture to `check`, but does not identify its engine.
Its check-only success predicate uses absence of parsed diagnostics without
requiring exit zero and no timeout. Its negative predicate checks diagnostics
and an optional substring rather than the full conformance exit/stage contract.
Its `expect-stdout` assertion is a substring check. Use the canonical suite or
named gate when these distinctions matter.

The [tool tests](../../tools/mcp/tests/test_tools.py) and
[protocol tests](../../tools/mcp/tests/test_protocol.py) assert selected tool,
schema, discovery, and SDK properties. Their source does not establish all
profile, severity, transport fallback, or negative-case boundaries above.

## Maintaining the explanation pages

There are three separate surfaces:

1. Compiler/parser emitters and focused tests establish source behavior;
   current receipts establish what actually ran.
2. Committed [explanations/](explanations) pages are historical Markdown of
   differing ages and engine assumptions. Illustrative fragments and
   recommendations require profile-specific review.
3. MCP resources are rendered by
   [errors_catalog.py](../../tools/mcp/sounio_mcp/errors_catalog.py) from its own
   `ERROR_CATALOG`, not from those Markdown files.

No checked-in regeneration command for the entire `explanations/E*.md` set was
identified in this inspection. No `souc explain` route was found in the inspected
public launcher dispatch. Do not invent a regeneration or explain command.
Reading `sounio://errors/<CODE>` renders the static catalog again; it does
not refresh the meaning from compiler source.

To refresh an explanation safely:

1. Pin revision, profile, emitter, dispatch/caller, and test annotations.
   Find every identity using that code, including direct prints.
2. Compare fresh messages with the proposed text and source traces. Old
   line numbers drift. Mark retired emitters and preserve engine distinctions.
3. Update only approved Markdown/catalog entries and governance metadata.
   Check copied full-program markers against the atlas manifest; a fragment
   is not an executable example.
4. Review positive/refusal witnesses in the authorized environment and retain
   identities/results. A timeout, missing tool, or wrong engine is not the
   intended refusal.
5. Re-read the affected resource/page, check links and metadata, and report
   which surfaces changed versus remain stale.

Relevant checks, **listed rather than executed by this guide**:

| Check | What it establishes | Qualification |
| --- | --- | --- |
| `node scripts/docs/check_docs_registry.mjs` | Registry, metadata, and selected local-link consistency. | Does not validate Sounio execution or scientific meaning. |
| `node scripts/docs/validate_language_spec.mjs` | Edition structure and source/evidence reference consistency. | Does not establish a compiler result. |
| `scripts/ci/diagnostic_remap_2170_gate.sh` | Static mappings and mutation controls; optional live `--souc` path. | Success without `--souc` is explicitly not live evidence. |
| `scripts/ci/diagnostic_identity_gate.sh` | Namespace ratchets and extraction controls. | A passing ceiling is not per-code semantic verification. |
| `tools/mcp/tests/test_tools.py`, `test_protocol.py`, `test_loop.py` | Selected direct-tool, schema, protocol, deterministic-loop assertions. | This page claims no test execution or model benchmark. |

The diagnostic shell gates contain Python helpers and the MCP loop recipe
calculates benchmark aggregates in Python. They are inventory references,
not permission to bypass ADR009. Use an authorized path for numerical
verification or disclose the execution constraint before running it.

Unresolved contradictions are recorded in the
[diagnostics audit](reviews/diagnostics-audit.md). Use that bounded inventory
rather than rewriting all explanations at once.

## Published namespace index

The table preserves published identities and links for discovery and the
identity ratchet's input format. **This is historical namespace data, not a
universal current-route lookup.** Route-specific sections above supersede its
gloss when they differ. In particular, `E249` is retired on the inspected
modular parser path, several low-numbered identities collide, and a linked
page is not a fresh execution receipt.

| Code | Component | Severity | Gloss | Explanation |
|------|-----------|----------|-------|-------------|
| E000 | legacy | error | Unclassified error (legacy — no code assigned) | — |
| E001 | type-checker | error | Type mismatch / linear constraint violation | [E001.md](explanations/E001.md) |
| E006 | type-checker | error | Arity mismatch — wrong number of arguments | [E006.md](explanations/E006.md) |
| E007 | codegen | error | Too many local variables in function | [E007.md](explanations/E007.md) |
| E008 | codegen | error | Too many globals | [E008.md](explanations/E008.md) |
| E035 | type-checker/effects | error | Effect not declared in function signature | [E035.md](explanations/E035.md) |
| E036 | type-checker/effects | error | Unobserved\<T\> crosses observation boundary without Observe | [E036.md](explanations/E036.md) |
| E040 | parser/compat | error | Rust `let mut` — use `var` instead | [E040.md](explanations/E040.md) |
| E041 | parser/compat | error | Rust `&mut` — use `&!` instead | [E041.md](explanations/E041.md) |
| E042 | parser/compat | error | Rust attribute `#[...]` not valid in Sounio | [E042.md](explanations/E042.md) |
| E043 | parser/compat | error | Rust macro `ident!(...)` not valid in Sounio | [E043.md](explanations/E043.md) |
| E067 | type-checker/epistemic | warning | Potential confounding in epistemic model | [E067.md](explanations/E067.md) |
| E070 | type-checker/kernel | error | IO or Mut effect in kernel function | [E070.md](explanations/E070.md) |
| E072 | type-checker/kernel | error | Kernel function must return unit `()` | [E072.md](explanations/E072.md) |
| E170 | type-checker/epistemic | error | `.value` on `Knowledge<T>` requires `with Epistemic` | [E170.md](explanations/E170.md) |
| E171 | type-checker/epistemic | error | Cannot cast epistemic type to its inner type | [E171.md](explanations/E171.md) |
| E200 | lean_single/resolve | error | Undefined identifier | — |
| E178 | type-checker/epistemic | error | noise-source capacity exceeded: more independent measurement sources than the noise-symbol domain can represent | — |
| E179 | type-checker/epistemic | error | noise-set interning table exhausted: too many distinct source-sets in one checker run | — |
| E188 | lean_single/chemistry | error | reaction does not balance for an element: the left and right sides of a `reaction` literal carry different counts of that element (the note line gives both) | — |
| E189 | lean_single/chemistry | error | reaction does not balance in charge: the summed charges of the two sides of a `reaction` literal differ (the note line gives both) | — |
| E190 | lean_single/chemistry | error | unknown element symbol in a `reaction` literal: the symbol is not in the checker's periodic table | — |
| E201 | type-checker/zero-divisor | error | `ExactlyPrivate<T>` requires `with ZD` | [E201.md](explanations/E201.md) |
| E202 | type-checker/zero-divisor | error | `Editable<T>` requires `with ZD` | [E202.md](explanations/E202.md) |
| E203 | type-checker/zero-divisor | error | `CapabilityGated<T>` requires `with ZD` | [E203.md](explanations/E203.md) |
| E204 | type-checker/zero-divisor | error | `Composable<T>` requires `with ZD` | [E204.md](explanations/E204.md) |
| E205 | type-checker/zero-divisor | error | `Audited<T>` requires `with ZD, Witness` | [E205.md](explanations/E205.md) |
| E206 | type-checker/zero-divisor | error | `Revivable<T>` requires `with ZD, Temporal` | [E206.md](explanations/E206.md) |
| E207 | type-checker/zero-divisor | error | `Interpretable<T>` requires `with ZD` | [E207.md](explanations/E207.md) |
| E208 | type-checker/refinement | error | refinement type violation — integer value violates predicate | [E208.md](explanations/E208.md) |
| E209 | type-checker/refinement | error | refinement type violation — f64 value violates predicate | [E209.md](explanations/E209.md) |
| E210 | type-checker/algebra | error | algebra property violation | [E210.md](explanations/E210.md) |
| E211 | type-checker/study | error | study block requires at least one hypothesis | [E211.md](explanations/E211.md) |
| E212 | type-checker/algebra | error | Hessian AD over a non-associative algebra | [E212.md](explanations/E212.md) |
| E213 | type-checker/destructure | error | tuple destructure arity mismatch | [E213.md](explanations/E213.md) |
| E214 | type-checker/epistemic | error | confidence gate violation | [E214.md](explanations/E214.md) |
| E215 | type-checker/epistemic | error | EpistemicComplete violation | [E215.md](explanations/E215.md) |
| E216 | type-checker/recursive-type | error | infinite recursive type | [E216.md](explanations/E216.md) |
| E217 | codegen | error | invalid function body span | [E217.md](explanations/E217.md) |
| E218 | codegen | error | tail type mismatch | [E218.md](explanations/E218.md) |
| E219 | codegen | error | function pass mismatch | [E219.md](explanations/E219.md) |
| E220 | codegen/linker | error | unresolved function body for call target | [E220.md](explanations/E220.md) |
| E221 | codegen/linker | error | no main function | [E221.md](explanations/E221.md) |
| E222 | codegen | error | code buffer overflow | [E222.md](explanations/E222.md) |
| E223 | codegen/pe | error | too many ExitProcess call sites | [E223.md](explanations/E223.md) |
| E224 | import | error | unreadable import | [E224.md](explanations/E224.md) |
| E225 | import | error | import dedup table full | [E225.md](explanations/E225.md) |
| E226 | import | error | import path table full | [E226.md](explanations/E226.md) |
| E227 | import | error | import too large for SRC buffer | [E227.md](explanations/E227.md) |
| E228 | import | error | import copy truncated | [E228.md](explanations/E228.md) |
| E229 | lexer | error | source exceeds lexer byte buffer, or token table full (fail-closed) | [E229.md](explanations/E229.md) |
| E230 | type-checker/epistemic | error | independence-assuming operation over correlated uncertainty | — |
| E232 | type-checker/struct | error | the same array local initialises two fields | — |
| E236 | type-checker/epistemic | error | binary operation on Knowledge values reuses a correlated binding; independence is not assumed (add `with Correlated`) | — |
| E244 | type-checker/epistemic | error | Knowledge provenance boundary mismatch: the argument's provenance cannot satisfy the parameter's requirement (trusted classes: Source, Literature, Measured) | — |
| E245 | type-checker/epistemic | error | arithmetic between two Knowledge<T> values is not supported | — |
| E246 | type-checker/effects | error | unknown effect, or effect dropped as a ninth slot | — |
| E247 | type-checker/zero-divisor | error | ZD locus is not a well-formed sedenion pair | [E247.md](explanations/E247.md) |
| E248 | type-checker/numeric | error | f128/f256 value conversion is not implemented | [E248.md](explanations/E248.md) |
| E249 | parser/type-checker | error | f128/f256 is reserved for compiler-owned format identity | [E249.md](explanations/E249.md) |
| E250 | type-checker/extern | error | call to an extern C function the native backend does not implement | [E250.md](explanations/E250.md) |
| E251 | type-checker/epistemic | error | product variance shortcut is unsound beyond the octonions (norm not multiplicative on Sedenion/Clifford) | [E251.md](explanations/E251.md) |
| E252 | type-checker/algebra | error | algebra declares a reassociation strategy its multiplication law cannot certify | [E252.md](explanations/E252.md) |
| E253 | type-checker/gpu | error | kernel function must return unit type | — |
| E254 | type-checker/causal | error | is not a node of the declared causal graph | — |
| E255 | type-checker/causal | error | is not d-separated from the other variable given the stated conditioning set | — |
| E256 | type-checker/causal | error | `d_separated` needs a declared causal graph and exactly two variable names, with an optional conditioning set | — |
| E257 | type-checker/causal | error | `indep_dsep`'s argument must be a `d_separated(...)` call the compiler itself discharged, not a hand-written value | — |
| E258 | parser | error | string literal(s) exceed Name capacity (384 bytes incl. quotes); refuses rather than silently truncating | [E258.md](explanations/E258.md) |
| E259 | type-checker/visibility | error | struct field is private in its defining module | — |
| E260 | lexer (lean_single) | error | compound assignment (`+=` `-=` `*=` `/=` `%=` `&=` `\|=` `^=` `<<=` `>>=`) is not supported by lean_single; write `x = x op (e)` | — |
| E261 | type-checker/ontology | error | ontology classes are disjoint and cannot be combined by this operator (e.g. an amount of H2 added to an amount of CO2) | — |
| E262 | type-checker/ontology | error | ontology classes are unrelated: no subsumption between them, so the operator cannot say what the result is an amount of | — |
| E263 | type-checker/modules | error | qualified call is ambiguous: the qualifier names a `use`d module but also collides with an unrelated struct/enum method of the same name, and the compiler cannot yet guarantee which one runs | — |
| E264 | type-checker/enums | error | enum variant with a payload is not supported by this compiler: the native lowering represents an enum value as a bare discriminant, so the payload would be dropped (construction, payload sub-pattern, or struct pattern over a user enum) | — |
| E265 | type-checker/epistemic | error | Temporal validity window mismatch: the provided value's validity window cannot satisfy the required type's validity constraint | [E265.md](explanations/E265.md) |
| E266 | type-checker/modules | error | qualified call resolves to no definition: the qualifier is unresolved or the target function is undefined (formerly warning W044; the call compiled to a body-less stub that trapped at run time, #1568) | — |
| E267 | type-checker/ownership | error | cannot assign through a shared reference: the place is reached through `&T` (explicit `*p` or the auto-deref of `p.f` / `p[i]`), which is read-only; take `&!T` (#888) | — |
| E268 | type-checker/ownership | error | affine value used more than once (an `affine struct` value may be used at most once; unlike linear it need not be used) | — |
| E269 | type-checker/epistemic | error | a `where` refinement on `Knowledge<T>` is not enforced by this compiler yet; the program is refused instead of running with the predicate silently ignored (#2753) | — |
| E270 | type-checker/casts | error | `as string` does not convert a number, bool or char to text; the cast is refused instead of compiling to a program that uses the value as a string pointer (#2354) | — |
| E271 | lexer (Madaros) | error | Unexpected character (byte N): a byte that starts no token is refused at its line and column instead of being kept as an `Error` token; `ε` and `⊥` stay tolerated because the `Knowledge<…>` parameter and call-label parsers consume them by name | — |
| E272 | lexer (Madaros) | error | unterminated block comment: a `/*` (block comments nest) reaches end of file with its depth above zero; reported at the opening `/*` instead of silently swallowing the rest of the file | — |
| E273 | lexer (Madaros) | error | unterminated string literal: a `"` reaches end of file without its closing quote; reported at the opening quote | — |
| E274 | lexer (Madaros) | error | invalid escape sequence in a string literal: only `\n` `\t` `\r` `\0` `\\` `\"` and `\xHH` (two hex digits) are escapes; any other backslash pair used to compile as the bare character | — |
| E275 | type-checker/ownership | error | borrow checker capacity exceeded: a linear or affine binding was not registered | — |

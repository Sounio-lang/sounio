<!-- docs:meta
topic_id: repo.docs.spec.2026-10.03-effects-handlers-and-concurrency
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.03-effects-handlers-and-concurrency
-->

# Effects, Handlers, and Concurrency

Source baseline: `f5b2c5850be9f3d3995719ffcade44a6d851658a`, inspected on
2026-10-16. Companion rule catalog: [03-evidence.tsv](03-evidence.tsv).

This chapter describes the effect information that the current compiler
collects, checks, and lowers, and identifies the boundaries at which that
information is incomplete. It also records the concurrency surfaces present in
the parser, compiler experiments, legacy code generator, and libraries. Those
surfaces do not currently constitute one uniform execution model.

The August snapshots `docs/spec/S06_EFFECTS_ROWS.md` and
`docs/spec/S07_EFFECT_HANDLERS.md` are research and decision history. Their
observations are not substituted for the source baseline above. In particular,
the named effect table, declaration diagnostics, closure inference, handler
lowering fallback, and `handler_discharge.sio` have changed since those
snapshots. Historical execution reports are not fresh execution evidence.

## 1. Standing, scope, and terminology

Each rule has one standing, also recorded in the companion catalog:

| Standing | Meaning in this chapter |
|---|---|
| `specified-contract` | An explicitly recorded language commitment; the implementation may still be incomplete. |
| `editorial-contract` | An edition reporting or session working requirement with explicit provenance; not adoption of source-language semantics. |
| `source-observed` | Descriptive behavior of the named current-source path, not a declaration that every engine or artifact behaves identically. |
| `library-contract` | Behavior of an ordinary library API or model, independent of native syntax support. |
| `target-contract` | Behavior encoded by a particular engine and target implementation. |
| `open-decision` | A language or integration decision remains outstanding; the text does not approve a candidate design. |

Positive and negative fixtures cited here were **unrun by this chapter's
source-research author**. The separate
[bounded conformance receipt](15-bounded-conformance-receipt.md) records the
existing manifest subset on the committed Madaros prebuilt, not a rebuild of
the inspected compiler source. It does not execute the majority of this
chapter's witnesses or establish logical/handler/concurrency semantics.
A positive path means a fixture intended to witness acceptance
or execution; a negative path means a fixture intended to witness refusal or a
counterexample. Neither cell asserts a passing result. A directory name such as
`run-pass`, an old `SELFHOST_PASS` description, or a `native-pass` directive is
not itself evidence that a current artifact passes the case.

An *effect identity* is an integer recognized by the relevant checker. An
*effect set* is the identities attached to a checked function signature. An
*ambient set* is the identities available while a body is checked. A *row
variable* would abstract over an unknown residual set; a bounded array of
identities does not provide that abstraction. A *handler* in the operational
sense intercepts an operation and defines how control continues. Merely
changing an ambient set does not implement such a handler.

The main modular route uses `self-hosted/check/check.sio`,
`self-hosted/check/defs.sio`, and `self-hosted/ir/lower.sio`.
`self-hosted/compiler/lean_single.sio` is a separate legacy implementation.
`bin/souc` defaults to the Madaros engine but has explicit overrides and
fallback handling; `scripts/lib/resolve_souc.sh` and
`scripts/lib/resolve_madaros.sh` remain the routing authorities. A source claim
in this chapter is not a claim about an unidentified prebuilt executable.

The compiler driver imports both in-place verdict wrappers and the older
by-value `check_program` entry. Its check-mode module-closure path calls
`check_modules_verdict_boot4_with_visibility` or a specialized
`check_items_verdict_boot4_with_module_map` path, which use the in-place
collectors. In contrast,
`self-hosted/check/mod.sio::check_program_with_artifacts` calls the by-value
`collect_items` and `check_items` methods. Rules about closure inference,
mutation generation, or user-effect registration below name their route
because these entry points are not interchangeable.

## 2. Effect vocabulary and representation

### 2.1 Effects are part of function types

**`EFF-001` — `specified-contract`.** A function type carries the effects of
the function. This is the founder ruling recorded in S06, section 6.0. It is
not restricted to the effects of named direct calls: passing a function value
must not erase the information about the effects it can demand when invoked.

The present representation of that commitment is `FnSig.effects` and
`FnSig.effect_count`, reached through `TypeEntry.fn_sig_id` for `TyFn`.
`fn(i64) -> i64 with IO` is therefore a different checked signature from a
corresponding signature with an empty effect set. The recorded commitment does
not decide the spelling or meaning of effect polymorphism, nor settle all
contexts in which omission of a clause is permitted.

### 2.2 Closed built-in names

**`EFF-002` — `source-observed`.** The modular built-in vocabulary is defined
by `EffectKind`, `effect_named_id_max`, `effect_kind_name_len`,
`effect_kind_name_byte`, and `effect_name_to_id` in
`self-hosted/check/effects.sio`. Lookup compares bytes and lengths; case is
significant. The named IDs in that source are:

| ID | Name | ID | Name | ID | Name |
|---|---|---|---|---|---|
| 0 | `IO` | 10 | `Network` | 20 | `Temporal` |
| 1 | `Mut` | 11 | `Sensor` | 21 | `Learn` |
| 2 | `Alloc` | 12 | `Render` | 22 | `Chaotic` |
| 3 | `Panic` | 13 | `Observe` | 23 | `Approx` |
| 4 | `Div` | 14 | `NonAssoc` | 24 | `NaturalityG2` |
| 5 | `GPU` | 15 | `Audit` | 25 | `Deterministic` |
| 6 | `Async` | 16 | `Hypothesis` | 26 | `Perturbative` |
| 7 | `Prob` | 17 | `MultiTest` | 27 | `NarrowWidthApproximation` |
| 8 | `Epistemic` | 18 | `ZD` | 28 | `NonUnitary` |
| 9 | `Causal` | 19 | `Witness` | 29 | `Correlated` |

`Confidence` is an alias for `Epistemic`, resolving to ID 8. `EffVar` is an
internal enum variant after the named identities, not an accepted built-in
`with` spelling. `Mod` is not a named effect. Neither `Exn` nor `FFI`, which
occur in other models, is a name in this table. Recognition of a name is
separate from existence of an operation that generates or consumes it.

### 2.3 Bounded sets, masks, and rows are different representations

**`EFF-003` — `source-observed`.** The production modular `FnSig` and ambient
checker representation is `[i64; 8]` plus a count. Its ordinary operations are
membership, missing-set calculation, and inclusion. It does not store an open
tail. For declaration collection through `checker_collect_effects_mut`,
duplicates are removed by resolved identity; `Epistemic, Confidence` therefore
occupies one identity. Order is not consulted by `effects_subset` or
`find_missing_effects`, although it can affect insertion and diagnostic order.

These are implementation limits and observations, not a ruling that eight is
an appropriate permanent language bound. A declaration set is an upper bound
on the effects checked at its call sites, not a count of events and not an
ordered execution trace. An unused declared effect still constrains callers.

In particular, a set of possible `IO` or `Mut` effects records neither dynamic
order nor event multiplicity. Conservative checking of both operands is
compatible with either eager or short-circuit execution; it does not establish
that every possible operation occurs. The logical-evaluation gap described by
`SYN-016` in [the syntax chapter](01-lexical-syntax-and-modules.md) and
`EXEC-011`/`EXEC-040` in [the execution chapter](06-execution-memory-and-targets.md)
therefore remains independent of set inclusion. This observation supplies no
reason to weaken static effect checking or ratify eager logical semantics.

The following representations are not interchangeable as raw integers:

| Surface | Representation and authority | Consequence |
|---|---|---|
| Modular function checker | Identity array; `self-hosted/check/effects.sio`, `self-hosted/check/defs.sio`: `FnSig` | ID 2 denotes `Alloc`, not a mask with bit 1 set. |
| Legacy code generator | `FN_EFFECTS` bit mask in `self-hosted/compiler/lean_single.sio` | `Panic` uses mask 4; `Alloc` uses mask 16. Visibility/import metadata also occupy this word. |
| Legacy epistemic analysis | `FN_DECL_EFF`, `FN_USED_EFF`, `ety_parse_effect_name` | This is another mapping; even `Observe` differs from `FN_EFFECTS`. |
| Sandbox library | `effect_bit`, `EFFECT_*` in `stdlib/compiler/effects/sandbox.sio` | `Async` uses library identity 4, whereas production identity 4 is `Div`. |
| Structural row experiment | `TyEffectRow`, `TyEffectVar`, `self-hosted/check/effects_row.sio` | A linked type representation exists without a production row-solving entry point. |
| Lean model | `EffectRow := Effect → Bool` in `formal/lean4/SounioEffects.lean` | A mathematical membership predicate, not the compiler's bounded signature representation. |

### 2.4 Refusal during named-signature collection

**`EFF-004` — `source-observed`.** `checker_collect_effects_mut` resolves a
built-in name first, then `checker_lookup_user_effect_id_mut`. An unresolved,
unheld name calls `checker_emit_e246` with face 0. A resolved identity absent
from an already full eight-entry set calls the same emitter with face 1.
The emitter increments `error_count`, sets `had_error`, and prints `error[E246]`
with the effect spelling and module/function identity. A duplicate in a full
set does not trigger the ninth-distinct-identity diagnostic.

This corrects the August blanket claim that all unknown names and ninth
effects disappear silently. It does **not** establish that every consumer of
an `EffectList` uses this collector. Function-type annotations, handler
checking, older by-value collection, and inferred widening require separate
treatment below. Comments in `effects.sio` that still call this valve E210 are
stale relative to `checker_emit_e246` and the current negative fixtures.

**`EFF-005` — `source-observed`.** Unresolved names recognized by
`effect_name_is_held` are deliberately exempted from that unknown-name
diagnostic and contribute no identity. Its current transition spellings are
`Mod`, `Compute`, `Foo`, `Exp`, `Log`, `NomeQueNaoExiste`, `Choice`, `Counter`,
`Fetch`, `Fail`, `NoSuchEffectX`, and `Sampling`. This is a compatibility valve,
not a definition of those effects. Built-in or already registered user names
are resolved before the held-name test.

The distinction matters for tests: a program declaring `with Choice` can
appear to enforce a user effect while its annotation is actually dropped if
the user registry was not populated. `tests/effects/archaeology/nosuch_drop.sio`
and `mod_pass.sio` expose historical compatibility behavior. They must not be
used to infer that arbitrary invented names are legal. The held set's
retirement and the intended interpretation of each name remain decisions.

### 2.5 Annotation syntax and payloads

**`EFF-006` — `source-observed`.** `Parser.parse_effect_list` consumes `with`
followed by a comma-separated list of names, with optional parenthesized
payloads, and produces `EffectList` nodes containing `EffectRef` values.
`EffectRef` stores up to 64 name bytes while retaining the supplied name
length. An effect annotation is not a declaration of a new name. There is no
open-row tail or row binder in this list parser.

`parse_effect_payload` retains a single integer literal in parentheses as
`payload` with `has_payload = true`. A leading minus in a parenthesized
payload reports a parse error. Other balanced payload shapes are skipped and
return `has_payload = false`; they must not be documented as evaluated effect
arguments. `epistemic_min_confidence_from_effects` uses the first retained
payload on a spelling resolving to ID 8. Ordinary effect inclusion still
compares identities, not payload orderings. Payload range, duplicate-payload
combination, and any confidence guarantee require the epistemic chapter and
its particular consumers, not an invented general row algebra.

## 3. Functions, calls, inference, and mutation

### 3.1 Function-type lowering has a different collection path

**`EFF-007` — `source-observed`.** `Checker.lower_fn_type_expr` lowers
`fn(T...) -> U with E...` to a `TyFn` with a new `FnSig`. It calls
`Checker.extract_effect_ids`, not `checker_extract_effects_mut`. That extractor
uses only `effect_name_to_id`, retains successive recognized entries up to
eight, does not deduplicate, does not consult the user-effect registry, and
does not report unknown names or overflow.

Consequently, the named-declaration refusal in `EFF-004` is incomplete across
syntax contexts. In this path, repeated names can consume capacity and cause a
later distinct name to be lost. A user effect that is registered elsewhere
still cannot be recovered by this extractor. These are localized defects in
collection consistency, not approved meanings for function types. Existing
`effect_unknown_name_refused.sio` and `effect_ninth_slot_refused.sio` exercise
named function clauses, not this function-type context; dedicated positive
and negative function-type cases are missing from this chapter's witnesses.

**`EFF-008` — `source-observed`.** For ordinary resolved calls in the default
modular checking mode, the effect check accepts inclusion of the callee's set
in the ambient set and otherwise follows the missing-effect path below.
The in-place route uses `checker_check_callee_effects_named_inplace`,
`effects_subset`, and `find_missing_effects`; failure normally reaches
`Checker.report_effect_error_from`, producing E035 and the missing effect
names. The named variant also identifies the callee requiring them. Some
older by-value and synthesized-builtin paths lack a callee name.

The accurately scoped checking rule is:

```text
Resolved signature of callee = (parameters, result, required)
Default named-body mode; call's other checks succeed
required ⊆ ambient
------------------------------------------------------
effect-inclusion check for this call succeeds
```

This rule describes a check, not an operational semantics or a proof that all
side effects in all expressions are discovered. The empty set is included in
every ambient set; an effectful caller may invoke a pure callee. A caller
declaring an unrelated effect does not satisfy a missing identity. The
declaration and actual leaf operations must both be checked for an end-to-end
guarantee. `effect_superset_ok.sio` and `effect_missing.sio` are the bounded
conformance spine's selected positive and negative cases.

Acceptance of that static inclusion check does not establish operand/callee/
argument order, exactly-once evaluation, RHS suppression for `false && rhs`
or `true || rhs`, or cross-target equivalence. The ordinary
`self-hosted/ir/lower.sio::lower_binary_expr_ref` lowers both operands before
emitting the binary operation; the separate native Metal AST renderer spells
logical operators as `&&`/`||`. These are source-local translation observations,
not a measured target disagreement or an adopted portable execution contract.
See `SYN-016`, `EXEC-011`, and the open decision `EXEC-040` for the reciprocal
boundary and review `12-syntax-runtime-review.md` for proposed discriminators.
An opaque-condition trace and independent mutation observation are still
needed; a boolean result or effect annotation alone cannot discriminate the
dynamic policies. The bounded receipt's effect pair tests ordinary inclusion
and missing annotation, not this sequencing question.

### 3.2 Function values and higher-order calls

**`EFF-009` — `source-observed`.** The structural function compatibility path
uses `checker_fn_sig_effects_compatible(provided, expected)`: every effect in
the provided signature must appear in the expected signature. Thus, when the
parameter and result compatibility checks also succeed, an empty-effect
function can inhabit a slot allowing `Panic`, but a function requiring `IO`
cannot inhabit an empty-effect slot through this check.

`checker_fn_types_structurally_compatible_table` and
`checker_types_compatible_table_deep` establish the signature lookup and
recursive structural checks. This text specifies only the **effect
component** of that relation. The current parameter comparison recurses in
the provided-to-expected direction; it is not evidence for a general
contravariant function-subtyping calculus. Same-signature identity and
unknown/error recovery paths also exist. Direct, method, and indirect call
routes should not be assumed identical without their own witnesses.

**`EFF-010` — `source-observed`.** A bare function-type annotation reaches
`lower_fn_type_expr` with no effect list and receives an empty effect set.
This is the implementation's current restrictive representation, not a
universal wildcard. S06's alternatives—bare type means pure, bare type is
refused, or effects are abstracted—are not resolved merely by observing that
representation. In particular, a separate legacy higher-order transparency
mechanism does not make an effect row variable exist in the modular type.

`closure_effect_escape.sio` uses a bare higher-order parameter and an
IO-producing closure. It is a useful negative witness, but a refusal at the
argument boundary and a refusal caused by transparent propagation are
different mechanisms. The test name or explanatory comment alone cannot
identify which one a current engine exercised.

### 3.3 Closure inference and experimental named-function inference

**`EFF-011` — `source-observed`.** The active in-place closure checker
`checker_check_closure_expr_inplace` saves the outer ambient set, clears it,
sets `in_closure_body`, and checks the closure body. Missing callee effects
are accumulated into this temporary set by
`checker_check_callee_effects_named_inplace`. The resulting set is published
onto the closure's own `FnSig`; the outer ambient set is restored. Creating a
closure therefore differs from invoking it: the body's future effects are
attached to the closure rather than charged immediately as outer-body calls.

This path also stores linear-capture information. Its epistemic-capture
calculation scans outer environment bindings and can append `Epistemic` when
capacity permits; it must not be described as a precise free-variable effect
analysis. The parser accepts closure effect clauses, but this checker path
derives its set from the body and does not use `e.closure_effects` as a declared
upper-bound check. Enforcing or interpreting an explicit closure clause needs
separate coverage and a decision about its relation to inference.

`closure_effect_infer_auto.sio` contains bodies calling printing functions and
passes closures to explicitly IO-bearing function parameters. Its own stdout
expectation must still be checked against its multiple output sites; it is an
unrun fixture here, not an accepted execution transcript.

**`EFF-012` — `source-observed`.** Named-function effect inference is an
opt-in experiment selected by `checker_effect_inference_enabled` when
`SOUNIO_EFFECT_INFER=1`. The result is memoized. The in-place missing-callee
path widens the ambient set and, when `current_sig_id` is valid and the stored
name matches `current_sig_name`, calls `fn_sig_table_widen_effects` to publish
the widened signature. A public signature widened this way emits W045;
`SOUNIO_EFFECT_INFER_TRACE=1` exposes the added set and source callee.

The default remains explicit named-body E035 checking at that path. The
by-value `Checker.check_callee_effects` remains a direct inclusion/refusal
routine. A mode flag must therefore be recorded with every inference result.
The source supports propagation to callers checked after a signature is
widened; it does not demonstrate a call-graph fixed point or source-order
independence for recursive or forward chains. The old comment above
`checker_widen_current_effects_inplace` saying no signature write-back exists
is contradicted by the newer implementation at the call site.

`tests/known_failures/effect_inference_transitive_chain.sio` requests traces
for both `mid += Mut` and `top += Mut`. Its comments themselves contain an old
run-pass description, but its directives classify it as mode-dependent known
failure. `effect_inference_pub_boundary.sio` is the related public-boundary
witness. Neither was executed here.

**`EFF-013` — `source-observed`.** Inferred widening is also bounded:
`checker_widen_current_effects_inplace` and `fn_sig_table_widen_effects` add a
new identity only while the destination count is below eight. Neither emits
the ninth-effect diagnostic. The closure's implicit `Epistemic` insertion and
the implicit kernel `GPU` insertion have similar capacity guards.

This creates a refusal-completeness gap: the source has a refusal for an
explicit ninth distinct declaration, but not a uniform refusal for an
inferred or implicitly inserted ninth identity. A complete checker contract
would require either representability of the union or a visible refusal;
silent omission is recorded here as a defect, not endorsed as inference.

### 3.4 Mutation and escape

**`EFF-014` — `source-observed`.** In the active in-place assignment checker,
a recognized write through a binding of type `TyRefMut` generates `Mut` unless
the ambient set contains `GPU`. `checker_place_excl_ref_root` follows the
assignment's identifier, dereference, field, or index root, subject to its
depth bound. `checker_check_assign_stmt_inplace` submits identity 1 through
the ordinary effect-checking helper, so default failure is E035 and closure
inference can collect the requirement.

The agreed host cases are `*p = value` for `p: &!T`, field writes rooted in
`&!S`, and index writes rooted in `&![T; N]`. The root predicate intentionally
does not include `TySliceMut`. It also does not treat a local `var` assignment
as a source of `Mut`; unrecognized roots and traversal beyond the bound return
false. The older by-value assignment checker has no matching leaf-generation
step. These facts delimit the guarantee rather than enlarge it to all
mutation or all checker routes.

The positive `tests/run-pass/excl_ref_store_still_allowed_issue_888.sio`
contains exclusive-reference field, array, and nested-field stores with
`Mut`. The negative `effect_mut_generated_at_excl_ref_store.sio` isolates a host
`fn poke(x: &!i64) { *x = 42 }` without a clause. The source's narrower rule
must not be replaced in this specification by the broader historical wording
“all mutable-variable reads and writes require Mut.” Local-state purity and
the treatment of mutable slices remain language decisions.

**`EFF-015` — `source-observed`.** Effect permission does not authorize a
reference to escape. After its mutation check,
`checker_check_assign_stmt_inplace` separately rejects reference-bearing
values or frame-local-reference provenance stored into fields, array elements,
or through dereferences, using E091. Return and closure ownership checks are
separate again. `with Mut`, inference, and `with GPU` do not disable these
checks.

Conversely, the absence of a generated `Mut` for local `var` does not prove
that the state cannot escape through every possible alias or closure. The
source's `checker_place_root_local`/`checker_expr_provenance` logic is an
ownership/provenance mechanism, not a region-indexed effect row. No general
state-encapsulation theorem is established by this chapter.

**`EFF-016` — `source-observed`.** Kernel signature collection injects `GPU`
when space is available and rejects explicitly present `IO`, `Mut`, and
`Alloc` with E070 in `checker_collect_fn_def_inplace`. The assignment check's
GPU exception is a membership test on the ambient set, not a proof that the
particular destination is device memory. A non-kernel function carrying
`GPU` reaches that same exception. This scope should be tested explicitly
before it is described as a host/device isolation guarantee.

`kernel_fn_gpu_effect.sio`, `kernel_io_effect.sio`,
`kernel_mut_effect.sio`, and
`gpu_exclusive_ref_write_requires_mut_host.sio` expose the intended annotation
and host-refusal boundaries. Declaring `GPU` is not evidence of kernel
submission, a synchronization barrier, device allocation, or any particular
GPU target being available.

## 4. Observe and IO are separate boundaries

**`EFF-017` — `source-observed`.** `IO` does not subsume `Observe`.
`checker_check_call_expr_inplace` and
`checker_check_print_i64_expr_inplace` explicitly check unobserved-like
arguments before printing. Without identity 13 they issue E036 through
`checker_report_unobserved_print_inplace`; the unobserved print branch also
requires `IO` through the ordinary effect helper. A raw `syscall6` call is
recognized separately and requires `IO`, not a blanket exemption from
effect checking.

These checks distinguish authorization to communicate with the environment
from authorization to inspect an `Unobserved` carrier. `Observe` alone does
not grant file or console access. The synthetic E036 membership check is
direct, so it is not automatically inferred merely because the general
missing-callee helper can widen closure effects. The recognized builtins and
their by-name matching should not be generalized into a whole-program
information-flow guarantee for arbitrary user IO wrappers or aggregates.

**`EFF-018` — `source-observed`.** `checker_unobserved_cast_allowed` and
`checker_check_binary_with_operand_types_inplace` implement a bounded
observation surface. The explicit unwrapping cast currently permits an
`Unobserved<i64>` carrier to `i64` only with `Observe`; wrapping `i64` into the
corresponding unobserved carrier is permitted by that helper. Nested and
other carriers fail that explicit boundary. Binary use of an unobserved value
requires a comparison operation and `Observe`; the checker then validates its
carrier through this same helper. Arithmetic is refused even when `Observe`
is present.

`check/compat.sio::types_compatible` keeps unobserved and ordinary types
distinct; it does not broadly unwrap under an ambient annotation. As a
result, older `Unobserved<f64>` fixtures such as
`observe_with_effect.sio`, `unobserved_basic.sio`, and
`observe_pattern_ok.sio` are not sufficient evidence of the current supported
carrier or implicit-conversion behavior. They require reconciliation with
current source and fresh acceptance/refusal witnesses. The existing
`observe_io_boundary.sio` is the conformance manifest's negative IO witness;
it does not prove every return, assignment, argument, cast, and pattern
boundary is covered.

## 5. Rows, user operations, and handlers

### 5.1 Structural row code is not a production row system

**`EFF-019` — `source-observed`.**
`self-hosted/check/effects_row.sio` contains `effect_row_contains`,
`effect_row_subset`, and `check_handler_coverage`. In the inspected
`self-hosted` tree the subset routine has only its internal recursive call,
and `check_handler_coverage` has no consumer. The production checker does not
import this module. `TyEffectRow` and `TyEffectVar` constructors in
`check/types.sio` do not establish parser binders, row substitution, unification,
or call-site instantiation.

Even within that experimental helper, `effect_row_subset` treats a left
`TyEffectVar` as accepted without generating constraints and treats several
other left types as true. Its coverage helper compares arm **counts**, not
operation identities or types. These routines must not be presented as a
sound, integrated row solver or exhaustive handler checker. The fixture named
`tests/frontend/effect_row_poly_basic.sio` contains only an ordinary pure
function and a call from `main`; it has no row variable. Its name cannot close
the missing integration evidence.

### 5.2 User effect declarations and their registry boundary

**`EFF-020` — `source-observed`.** `Parser.parse_effect_item` recognizes
`effect Name { fn operation(parameters) -> Result ... }` and stores
`EffectDef`/`EffectOpDef` AST records. `parse_effect_op_def` preserves the
parameter list and optional result annotation. The enclosing parser skips
semicolons and newlines between operations; it also advances past
non-`fn` tokens rather than implementing a strict refusal for every malformed
member. The syntax does not itself create an executable handler or operation.

`effect_user_defined_basic.sio` is an actual declaration fixture with
`Storage` and `Logger`; it does not perform either effect.
`tests/run-pass/effect_handler_basic.sio` similarly declares `Choice` and calls
an empty `with Choice` function. It exercises no `handle`, operation call, or
continuation.

**`EFF-021` — `source-observed`.** User-effect collection is incomplete in
the active route. The by-value `Checker.collect_item` dispatches `ItemEffect`
to `collect_item_effect`, which assigns identities starting at `100 + index`
and populates `EffectTable` and `EffectOpTable`. However,
`checker_collect_item_inplace` has no `ItemEffect` arm and falls through to its
no-op branch. `self-hosted/check/mod.sio` uses the in-place collection
wrappers, including `checker_collect_item_only_mut`, for the module-verdict
paths described in section 1. Its separate `check_program_with_artifacts`
entry still uses by-value collection and must be distinguished from them.

Thus the presence of `checker_lookup_user_effect_id_mut` does not prove user
declarations reach that registry in normal modular checking. An unheld user
effect used in a function clause can reach E246 when its declaration was
skipped. A held spelling such as `Choice`, `Counter`, `Fetch`, or `Fail` can
instead disappear, making an apparent positive or negative user-effect test
especially difficult to interpret.

The by-value registry is itself only partial: operation collection writes
`param_count: 0` rather than lowering the AST parameters; the available
`find_by_name_and_effect` helper has no active operation-dispatch consumer
identified here. `EffectTable.add` stops inserting at 32 entries and
`EffectOpTable.add` at 128 without a capacity diagnostic. Name shadowing,
duplicate declarations, declaration order, import identity, and overflow
refusal need decisions and discriminating tests. These are not claims of
implemented user-operation semantics.

### 5.3 Perform and resume do not provide a continuation surface

**`EFF-022` — `source-observed`.** `Parser.parse_perform_expr` consumes
`perform` and calls the ordinary expression parser; it returns the following
expression without a distinct perform AST node. A following call must
therefore resolve and type-check as an ordinary call. The existence of the
token does not establish effect-operation resolution or continuation capture.

The lexer recognizes `Resume` and `Handler`, but the inspected expression
parser/checker does not provide a dedicated resume expression, continuation
binding, resume argument type check, or operation-clause dispatch. The
checker's `handler_depth` and `current_handler_op_id` are initialized and
copied, not used here to validate a captured continuation. Any contextual
identifier use of these spellings must be distinguished from operational
handler syntax.

The negative fixtures `effect_handler_missing_arm.sio` and
`effect_resume_wrong_type.sio` do not contain a handler or a resume. Both call
a user-effect-declaring function from a caller lacking that annotation.
`tests/ui/effects/continuation_basic.sio` and `continuation_value.sio` contain
arithmetic and ordinary calls, not continuations. These files cannot establish
the advertised semantics, even if a particular engine accepts or rejects
them.

### 5.4 The actual handle expression

**`EFF-023` — `source-observed`.** The expression parser accepts
`handle<EffectName> { body }` with an optional `with { handler_block }`.
`Parser.parse_handle_expr` constructs `ExprHandle` with an ordinary body block
and ordinary handler block. This is the concrete parser surface, not the
alternative top-level handler grammars in historical documentation.

`Checker.check_handle_expr` resolves only the built-in `effect_name_to_id`,
saves the ambient set, and temporarily adds that identity if it resolves,
is absent, and capacity permits. It checks the body, restores the saved
ambient set, invalidates `current_sig_id`, then checks the optional handler
block in the restored ambient context. The expression's type is the body's
type, not the handler block's type. No operation-arm coverage, return-clause
agreement, user-effect lookup, continuation scope, or multiplicity check
occurs in this routine.

Unknown handled names and the inability to insert an additional identity do
not call `checker_emit_e246` here. Checking both blocks does not mean both
blocks run, nor does temporarily authorizing `IO` intercept its operations.
`tests/audit/handle_io.sio` and `handle_unknown_effect.sio`, compared with
`handle_control.sio`, remain useful distinguishing witnesses precisely because
the latter two handle forms should not be treated as interchangeable.

**`EFF-024` — `source-observed`.** Operational handler support is absent
from the ordinary modular IR expression dispatch, but the failure shape has
changed since S07. `Lowerer.lower_expr_non_binary_ref` has no `ExprHandle`
case. Its unmatched-expression fallback calls `report_error_at(15454)` and
then emits a unit value. The error-reporting machinery records a lowering
failure, including a compile-scope latch. `fo_expr_has_unsupported_ctl` does
mention `ExprHandle`, but only to exclude it from a variance helper's
supported control flow; that is not handler lowering.

Accordingly, this source inspection supports “no operational handler lowering
and an error-marking fallback,” not the old universal statement “compiles and
silently prints nothing.” No current binary was run here to establish its
exit status or diagnostic presentation. A serious execution contract must
refuse unsupported handling before publishing an executable as valid; a
source fallback emitting unit after recording an error is not a semantics for
executing the handler body.

`handler_discharge.sio` now contains an actual `handle<IO>` body and a
known-failure marker; it is no longer just `println("handler: PASS")`. Its
comments still describe the old erasure result. The positive control without
`handle` and the negative unknown-effect control are needed in addition to
the expected output assertion. No current handler execution success is
claimed by this chapter.

### 5.5 Libraries and experiments must remain visibly separate

**`EFF-025` — `library-contract`.**
`stdlib/compiler/effects/sandbox.sio` provides an explicit `EffectVM` API.
`effect_vm_install_handler` pushes a bounded frame; `effect_vm_perform` checks
the policy, records an effect/log entry when space permits, searches from the
top of the handler stack, and sets the VM program counter to a matching
frame's `resume_pc`. A denied or unhandled operation marks the VM erroneous.
The handler stack and effect log are bounded, and installation/logging have
capacity guards.

`effects_sandbox_handlers.sio` calls these functions directly and inspects VM
state, including denial and unhandled cases. Its “multi-shot” case performs
twice and observes the same destination PC; it is not a source-language
continuation captured and resumed twice. This is a useful library-level
model with genuine state transitions, but it does not connect native
`perform` or `handle` syntax to the VM API.

**`EFF-026` — `source-observed`.**
`self-hosted/effects/handlers.sio` defines the separate `EfhContext`, handler,
continuation, resume-strategy, and CPS transformation records.
`self-hosted/effects/checker.sio` has a distinct bit-set inference engine.
No production `use effects::...` importer was found in the inspected
`self-hosted` tree. Their tests and records do not establish a reachable
front-end-to-back-end continuation implementation.

The similarly named `stdlib/compiler/effects/handler.sio` is a different
surface again: `with_handler` returns a placeholder value and a masked
bitset, `invoke_fn` returns a dummy value, and `store_continuation` is a no-op.
Ordinary calls to these routines are not a working runtime interception
mechanism. The module's capability record types likewise do not establish a
security boundary over the compiler's ordinary IO operations.

### 5.6 Scope of the formal model

**`EFF-027` — `source-observed`.**
`formal/lean4/SounioEffects.lean` models rows as characteristic functions and
defines `mask`, union, inclusion, and related theorem statements. Its
`Effectful` structure stores an already available `value` and an effect row;
`handle` copies the value while masking one identity. The statements
`handle_preserves_value`, `handle_comm`, and `discharge_pair_interaction`
therefore concern that pure metadata model.

They do not assert that arbitrary stateful IO handlers commute, that
continuation duplication preserves ownership, or that the compiler refines
the model. The model's `Effect` vocabulary also differs from the production
table. `scripts/ci/effects_handler_interaction_lean_gate.sh` requests
`lake build SounioEffects` and `lake build SounioNonUnitaryNWA`; those builds
were not run for this chapter. The existence of theorem text is not reported
as a successful build, an axiom audit, or a compiler correctness proof.

## 6. Concurrency surfaces and execution boundaries

### 6.1 Modular syntax does not currently select a scheduler

**`ASYNC-001` — `source-observed`.** The modular parser recognizes
`async fn`, but `Parser.parse_item` merely advances past `Async` before
parsing the ordinary function. `FnDef` has no asynchronous flag on this path.
The parser does not inject `Async`, synthesize a future result, or introduce
a suspension boundary from that modifier.

`Parser.parse_async_like_block_expr` accepts `async { ... }` and
`spawn { ... }`, producing `ExprBlock` and retaining the keyword's name as a
marker. A missing opening brace sets parser error state. These are the
accepted block-shaped forms; a token alone does not start a task.

**`ASYNC-002` — `source-observed`.** The ordinary modular block checker and
`Lowerer.lower_block_expr_ref` process these marked nodes as blocks; the
lowering function uses `e.block` and ignores the async/spawn name. No task
allocation, worker creation, result-handle construction, or suspension is
introduced at that dispatch. In a containing function that is called, this
route executes the block in ordinary evaluation order.

This is semantic degradation relative to the usual meaning of spawn, not an
approved concurrency specification. A value-only fixture such as
`let handle = spawn { 10 }; handle.await` is insufficient to distinguish
inline block evaluation from a genuine concurrent mechanism. A target
contract must either supply its mechanism or visibly refuse unsupported
syntax; changing only the annotation cannot supply the missing lowering.

**`ASYNC-003` — `source-observed`.**
`Parser.parse_dot_expr_box` parses `.await` through the generic dot/field
surface when no call parentheses follow it. There is no distinct await AST
variant in this path, and the ordinary field checker does not provide a
future-poll protocol for it. A later unknown-field or type failure is not an
explicitly designed “await unsupported” diagnostic. An actual field named
`await`, if accepted contextually, would also not prove suspension semantics.

### 6.2 Disconnected state-machine and scheduler code

**`ASYNC-004` — `source-observed`.**
`self-hosted/ir/async_lower.sio` contains an async state-machine experiment
with `AsyncLowerCtx`, `AsyncStateMachine`, capture/live-variable records, and
`ASYNC_AWAIT_SENTINEL`. Its `async_lower_module` scans IR for the sentinel and
returns a context containing machines. The inspected `self-hosted` call sites
outside its definition are its own tests; no production caller or normal
parser-to-sentinel producer was identified. It does not supply the missing
`ASYNC-001` through `ASYNC-003` integration by existing on disk.

**`ASYNC-005` — `source-observed`.**
`self-hosted/vm/green_threads.sio` implements a distinct bounded scheduler
model with thread, ready-queue, stack, and channel records and functions
prefixed `gth_`. No production `use vm::green_threads` importer was found in
the inspected tree. Its internal `gth_run_tests` is not evidence that native
`spawn` selects this scheduler. It must be evaluated as its own module,
including the capacity and scheduling behavior it actually implements.

### 6.3 Legacy Linux x86-64 execution surface

The following rules describe code emission in the x86 path of
`self-hosted/compiler/lean_single.sio`. They do not claim successful execution
of a current artifact, portability to a64 or Darwin, or equivalence with the
default modular engine. The emitted Linux syscall numbers and x86 bytes are
the evidence for their target scope.

**`ASYNC-006` — `target-contract`.** In the legacy x86
`compile_primary` path, `async { body }` eagerly compiles and evaluates the
body and returns its last value. In `compile_postfix_tail`, `.await` on a
non-task value is an identity operation. This is neither lazy future
construction nor cooperative polling. The `Async` annotation is not the
switch that creates this behavior; recognition of the syntax is.

**`ASYNC-007` — `target-contract`.** Legacy x86 `spawn { body }` emits a
Linux `mmap` for a 4096-byte shared anonymous result slot and then syscall 57,
`fork`. In the child path it executes the body, stores one machine word to
the mapped result slot, and exits through syscall 60. The parent returns a
stack-resident descriptor containing the child PID and the mapped-slot
pointer, represented internally by `EXPR_TY = 100`.

This is process creation with a shared result mapping. It is not the
shared-address-space clone-thread mechanism described by some test comments.
Ordinary local mutations in the child do not thereby become visible in the
parent's private memory. Aggregate and floating-point results, descriptor
escape beyond the producing frame, failure of `mmap`/`fork`, and cleanup are
not made safe by the existence of a one-word result slot. The displayed
emission path does not implement a checked error return or reclamation
protocol for those resources.

**`ASYNC-008` — `target-contract`.** Legacy x86 `.await` on internal task
type 100 emits syscall 61, `wait4`, for the stored PID, then reads one word
from the shared result pointer and reports an integer result. `join(h1, h2)`
performs two such waits in sequence and constructs a two-word tuple. The
workers may already run concurrently; the waits themselves are blocking and
ordered. This is not a poll-based future protocol.

The path does not specify error/result distinction for a failed wait, a
crashed worker, repeated await, cancellation, or a descriptor used after its
stack frame has ended. “Await can be written on a scalar” on this legacy path
means identity fallback, not a proof that all scalar types implement an
awaitable interface. These limitations are not promoted to permanent
language semantics.

**`ASYNC-009` — `target-contract`.** Legacy x86
`channel::<T>()` emits syscall 293, `pipe2`, with close-on-exec and returns a
pair of file descriptors. It syntactically skips the generic argument rather
than establishing a typed serialization layout for `T`. The built-in
`.send(value)` and `.recv()` paths recognize integer receivers and emit
blocking eight-byte `write`/`read` operations. Receive reads a word from its
stack buffer; `.await` on that returned ordinary value then falls through to
the identity behavior.

Thus this surface is an integer-word pipe interface, not a generic,
nonblocking future channel. It does not provide a checked short-read,
short-write, EOF, closed-end, interrupted-call, or descriptor-ownership result
protocol in the shown emission. Acceptance of `channel::<i32>()` is not
evidence of a four-byte payload ABI. An arbitrary integer receiver matching
the builtin shape is not a statically distinct `Sender<T>` or `Receiver<T>`.

**`ASYNC-010` — `target-contract`.** Legacy x86 `sleep(milliseconds)` emits
a Linux `nanosleep` timespec and syscall 35. It blocks the executing process;
it does not register a timer with a cooperative executor. This path neither
validates negative durations as a language error nor implements a documented
interruption/retry protocol. These boundary cases need target witnesses and
an API decision before a portable timing contract is claimed.

### 6.4 Ordinary concurrency libraries

**`ASYNC-011` — `library-contract`.**
`stdlib/async/future.sio` defines `PollResult` and constructors/accessors for
pending and ready values. `stdlib/async/executor.sio::Executor` stores up to
256 task IDs, status values, and results. `Executor.spawn` records an ID and
returns silently when full; `complete_task` updates a status/result and
increments the completed count for an in-range index. These functions do not
call task code or run a scheduling loop. Repeated completion is not guarded
against counting the same slot again.

Similarly, `stdlib/thread/spawn.sio::ThreadHandle.join` sets a status and
returns the stored result; it does not invoke an operating-system join.
These are record/state APIs, not an automatically installed runtime for
`async fn`, `.await`, or `spawn { ... }`. The existing executor and future
E2E fixtures need execution and interpretation at this library scope.

**`ASYNC-012` — `library-contract`.**
`stdlib/async/channel.sio` supplies `AsyncChannelState` and an ordinary
64-word ring buffer. `channel_ring_push` leaves a full buffer unchanged;
`channel_ring_pop` leaves an empty buffer unchanged; `channel_ring_peek`
returns zero for an empty buffer. They do not suspend a producer or receiver.

`stdlib/sync/channel.sio::IntChannel` is a separate 1024-word ring API.
Its send operation returns false when full or closed; receive returns zero
when empty, which is indistinguishable from a stored zero without another
state check. Closing prevents further send and does not itself erase queued
values. Neither API's ordinary field accesses supply atomic synchronization
or a cross-thread memory model. Their capacity constants are separate from
the pipe interface in `ASYNC-009`.

### 6.5 Claims that the fixtures do not establish

**`ASYNC-013` — `source-observed`.**
`tests/run-pass/async_spawn_syscall_pid.sio` expects a spawned worker to share
the parent's PID while having a different TID. That expected mechanism
conflicts with the inspected legacy `fork` emission. The modular ordinary
block route supplies no worker at all. A source or artifact change outside
the inspected routes could explain a historic green result, but no such
result is asserted here.

The value assertions in `async_basic.sio`, `async_spawn.sio`, and
`async_join.sio` are useful arithmetic/result witnesses, but cannot alone
establish a lazy future, scheduler, thread identity, or concurrent execution.
`async_stress_nested.sio`, `async_stress_slot_size.sio`,
`async_stress_cow.sio`, and `async_stress_channel.sio` exercise additional
resource and memory hypotheses; each needs engine, target, artifact identity,
timeout, exit status, and stdout evidence before being used as a claim.

## 7. Decisions still required

The following are decision records, not approvals of any option. Refusal
obligations state the information a complete implementation must preserve or
make visibly unavailable; they do not silently change the current language.

**`EFF-028` — `open-decision`.** Decide whether the public effect discipline
is a finite closed set, an extensible set, or row polymorphism with binders,
instantiation, and residual-tail constraints. Settle the meaning of a bare
function type and of an explicit closure clause in that discipline. The
current empty representation and disconnected row utilities cannot answer
these policy questions. Required witnesses include compatible and
incompatible higher-order calls, recursive/forward inference, and a genuine
open-tail example if that option is adopted.

**`EFF-029` — `open-decision`.** Decide whether eight identities is a language
bound or a replaceable implementation capacity, and make collection,
inference, implicit insertion, function types, user registries, and handlers
agree about refusal. Retirement of held names needs explicit migration
behavior. A negative spelling must be rejected because its name is unknown,
not accidentally because an unrelated operation failed; a ninth-identity
test must distinguish duplicates and aliases from distinct identities.

**`EFF-030` — `open-decision`.** Define operational user effects and
handlers before advertising them: operation namespace and parameter/result
types; completeness and duplicate-arm checking; handler result and return
clause; deep versus shallow handling; effect requirements of handler arms;
continuation ownership and escape; zero-, one-, or multiple-resume policies;
unhandled-operation refusal; and interaction with failure and resource
cleanup. A candidate mask rule on a set is not a substitute for these control
semantics. No option is selected by this chapter.

**`EFF-031` — `open-decision`.** Settle the observation boundary and the
meaning of local mutation without importing a historical broad reading as
law. The implemented i64 explicit-carrier restriction, older f64 fixtures,
IO-print checks, mutable slices, GPU-marked non-kernel stores, and local-state
escape all need discriminating cases. Decide whether direct capability
checks such as E036 are inferable or always require an explicit ambient
annotation. A proof of encapsulated local state, if desired, belongs in Lean
with a model connected to these actual checks.

**`ASYNC-014` — `open-decision`.** Define a single public concurrency
contract or explicit engine/target feature profiles. Required decisions
include eager versus lazy async evaluation, futures and polling, process
versus thread spawn, capture/ownership transfer, descriptor lifetime,
result layout, cancellation, panic propagation, join consumption,
synchronization and memory visibility, fairness, scheduling and blocking
points, channel capacity/backpressure and closure, error reporting, and
resource reclamation. Unknown or unsupported mechanisms need a visible
refusal rather than ordinary inline block evaluation advertised as
concurrency.

**`EFF-032` — `source-observed`.** Effect-name recognition alone asserts no
operational mechanism. `Async` does not create a scheduler, `GPU` does not
submit a kernel, and `Prob` does not choose a random-number generator,
probability measure, inference algorithm, or determinism policy. Scientific
effect names such as `Approx`, `NonUnitary`, or `Correlated` can participate
in specific additional checks, but their generic inclusion behavior does not
validate a scientific calculation or instantiate a runtime service. Each
stronger claim needs its actual consumer and a discriminating witness.

## 8. Evidence map and acceptance work

The companion TSV has one row for each rule above, with one principal source
and at most one positive/negative path per cell. The following table supplies
additional cross-file and gate context. All listed tests, gates, and Lean
builds remain unrun in this investigation.

| Area | Current-source paths and symbols | Positive/negative witnesses or gate | What the evidence can and cannot establish |
|---|---|---|---|
| Named vocabulary | `self-hosted/check/effects.sio`: `EffectKind`, `effect_name_to_id`, `effect_name_is_held` | `tests/run-pass/effect_closed_list_nonregression.sio`; `tests/run-pass/effect_known_names_regression.sio`; `tests/compile-fail/effect_unknown_name_refused.sio` | Name lookup and declaration refusal. The older closed-list positive omits `Correlated`; the newer regression includes it. |
| Capacity | `self-hosted/check/check.sio`: `checker_collect_effects_mut`, `checker_widen_current_effects_inplace`; `self-hosted/check/defs.sio`: `fn_sig_table_widen_effects` | `tests/compile-fail/effect_ninth_slot_refused.sio` | Covers an explicit named declaration, not inferred, implicit, or function-type overflow. |
| Effect propagation | `self-hosted/check/effects.sio`: `effects_subset`, `find_missing_effects`; `self-hosted/check/check.sio`: `checker_check_callee_effects_named_inplace` | `tests/run-pass/effect_superset_ok.sio`; `tests/compile-fail/effect_missing.sio`; `tests/conformance/manifest.v1.tsv` | Bounded positive and missing-annotation claims; not a full soundness proof. |
| Higher-order compatibility | `self-hosted/check/check.sio`: `lower_fn_type_expr`, `extract_effect_ids`, `checker_fn_sig_effects_compatible` | `tests/selfhost/native_runtime/fn_effect_subset_0.sio`; `tests/compile-fail/closure_effect_escape.sio` | Effect subset check and escape witness; function-type unknown/user names and duplicate capacity remain uncovered. |
| Inference modes | `self-hosted/check/check.sio`: `checker_check_closure_expr_inplace`, `checker_effect_inference_enabled` | `tests/run-pass/closure_effect_infer_auto.sio`; `tests/known_failures/effect_inference_transitive_chain.sio`; `tests/known_failures/effect_inference_pub_boundary.sio` | Closure publication and experimental named write-back need distinct runs, environment recording, and traces. |
| Host mutation | `self-hosted/check/check.sio`: `checker_place_excl_ref_root`, `checker_check_assign_stmt_inplace` | `tests/run-pass/excl_ref_store_still_allowed_issue_888.sio`; `tests/compile-fail/effect_mut_generated_at_excl_ref_store.sio`; `tests/compile-fail/gpu_exclusive_ref_write_requires_mut_host.sio` | Default host store refusal and explicitly permitted exclusive writes; not a guarantee for local variables, slices, all aliases, or every assignment route. |
| Kernel boundary | `self-hosted/check/check.sio`: `checker_collect_fn_def_inplace` | `tests/run-pass/kernel_fn_gpu_effect.sio`; `tests/compile-fail/kernel_io_effect.sio`; `tests/compile-fail/kernel_mut_effect.sio` | Annotation restriction/injection; not device execution. |
| Observation | `self-hosted/check/check.sio`: `checker_unobserved_cast_allowed`, `checker_report_unobserved_print_inplace`; `self-hosted/check/compat.sio`: `types_compatible` | `tests/compile-fail/observe_io_boundary.sio`; `tests/compile-fail/observe_assign_boundary.sio`; `tests/compile-fail/observe_return_boundary.sio`; `tests/run-pass/observe_with_effect.sio` | Negative boundaries and older positive intentions; current i64-carrier and no-implicit-unwrapping changes require reconciliation. |
| User declarations | `self-hosted/parser/items.sio`: `parse_effect_item`; `self-hosted/check/check.sio`: `checker_collect_item_inplace`, `collect_item_effect`, `collect_effect_op_defs` | `tests/frontend/effect_user_defined_basic.sio`; `tests/run-pass/effect_handler_basic.sio`; `tests/compile-fail/effects_unhandled.sio` | Parsing versus skipped active collection; held names can invalidate the apparent test mechanism. |
| Native handler construct | `self-hosted/parser/exprs.sio`: `parse_handle_expr`; `self-hosted/check/check.sio`: `check_handle_expr`; `self-hosted/ir/lower.sio`: `lower_expr_non_binary_ref` | `tests/audit/handle_control.sio`; `tests/audit/handle_io.sio`; `tests/audit/handle_unknown_effect.sio`; `tests/run-pass/handler_discharge.sio` | Compare real versus invented names and unsupported lowering; historic silent-erasure result is not rerun. |
| Misnamed handler coverage | `self-hosted/check/effects_row.sio`: `check_handler_coverage` has no consumer | `tests/compile-fail/effect_handler_missing_arm.sio`; `tests/compile-fail/effect_resume_wrong_type.sio`; `tests/ui/effects/continuation_basic.sio` | These fixtures contain no matching arm/resume/continuation semantics. Refusal or success alone is not the claimed test. |
| Sandbox handler model | `stdlib/compiler/effects/sandbox.sio`: `effect_vm_perform`, `effect_vm_install_handler` | `tests/run-pass/effects_sandbox_handlers.sio`; `tests/run-pass/effects_sandbox_integration.sio` | Explicit VM API behavior including denial; no native syntax integration. |
| Formal masking | `formal/lean4/SounioEffects.lean`: `EffectRow`, `mask`, `Effectful`, `handle`; `formal/lean4/SounioNonUnitaryNWA.lean`: `discharge_interaction` | `scripts/ci/effects_handler_interaction_lean_gate.sh` | Gate requests Lean builds for mathematical models; no runtime refinement claim or fresh proof result. |
| Modular async parsing/lowering | `self-hosted/parser/items.sio`: `parse_item`; `self-hosted/parser/exprs.sio`: `parse_async_like_block_expr`, `parse_dot_expr_box`; `self-hosted/ir/lower.sio`: `lower_block_expr_ref` | `tests/run-pass/async_basic.sio`; `tests/run-pass/async_spawn.sio` | Accepted shapes can lose async meaning. Need side-effect/order/identity witnesses, not just values. |
| Legacy processes | `self-hosted/compiler/lean_single.sio`: `compile_primary`, `compile_postfix_tail` | `tests/run-pass/async_join.sio`; `tests/run-pass/async_stress_nested.sio`; `tests/run-pass/async_stress_cow.sio`; `tests/run-pass/async_spawn_syscall_pid.sio` | Source emits fork/wait; same-PID/different-TID fixture expects another mechanism. |
| Legacy pipe/timer | `self-hosted/compiler/lean_single.sio`: `compile_primary`, `compile_postfix_tail` | `tests/run-pass/async_channels.sio`; `tests/run-pass/async_stress_channel.sio`; `tests/run-pass/async_sleep.sio` | Target-specific blocking operations; typed payload and system-error boundaries remain uncovered. |
| Ordinary library state | `stdlib/async/executor.sio`: `Executor`; `stdlib/async/future.sio`: `PollResult`; `stdlib/sync/channel.sio`: `IntChannel` | `tests/stdlib/async/test_executor_e2e.sio`; `tests/stdlib/async/test_future_e2e.sio`; `tests/stdlib/sync/test_channel_e2e.sio` | Library record/ring behavior; no automatic scheduler, atomicity, or native keyword connection. |

### 8.1 What existing gates actually guard

`scripts/ci/serious_language_conformance_gate.sh` uses the bounded manifest
cases `effects-superset-run`, `effects-missing-diagnostic`, and
`observe-io-boundary`. Its companion spec-drift and claim-closure scripts
enforce evidence references and public-claim bookkeeping. None is a complete
handler, row-inference, or concurrency conformance suite. Governance
registration for this chapter is handled separately from this write set.

`scripts/ci/effect_archaeology_gate.sh` executes the positive/negative pairs
from `tests/effects/archaeology/index.tsv` and checks a named diagnostic. This
is stronger than a mere source grep, but a declared-effect propagation pair
does not show that an annotation invokes a runtime mechanism.
`effect_name_closed_list_gate.sh` and `fn_type_effect_ratchet_gate.sh` guard
name/bare-type corpus drift; their bookkeeping does not implement row
polymorphism. The archived sprint handler gates include structural checks and
the misleadingly named fixtures described above; their old summaries cannot
upgrade the current feature standing.

### 8.2 Required discriminating witnesses

Before promoting these surfaces beyond their current standing, evidence
needs to separate these cases:

1. **Collection consistency:** built-in versus registered user versus unknown
   versus held spellings in named, method, function-type, closure, and handler
   contexts; aliases and duplicate entries at full capacity; explicit and
   inferred overflow. The expected refusal must identify the actual boundary.
2. **Inference:** closure creation versus invocation; an annotated pure slot
   versus an IO slot; explicit closure-clause upper bounds; public W045;
   forward order, recursion, and a chain checked in both declaration orders.
   An error count of zero does not by itself prove publication to callers.
3. **Mutation/observation:** positive and negative host exclusive writes;
   local variables, mutable slices, unknown roots, and GPU-marked host
   functions; an i64 unobserved carrier versus the older f64 fixture shape;
   IO and Observe independently present/absent; implicit and explicit unwraps.
4. **Handlers:** a control body that definitely runs; a known handled name
   versus an unknown one; missing and duplicate operation arms; wrong resume
   type; repeated resume; escaping continuation; an effect performed by a
   handler arm; and an unsupported-target refusal before a valid executable
   is published. Existing arithmetic-only continuation fixtures are inadequate.
5. **Concurrency:** distinguish inline evaluation, eager async evaluation,
   lazy future evaluation, processes, and shared-memory threads. Check PID/TID,
   capture visibility, actual blocking/order, nested descriptor lifetime,
   large and floating results, worker failure, join repetition, and cleanup.
   Channel checks need full/empty/closed/error/partial-transfer cases and
   payload-layout evidence. Timer checks need invalid-duration and
   interruption behavior. Record engine, target, executable identity, and
   timeouts for every run.

### 8.3 Checks performed for this chapter

The work consisted of reading current repository source and fixtures,
searching symbol definitions and call/import sites, comparing the S06/S07
snapshots with those paths, and checking this chapter and its companion
catalog structurally. No compiler build, runtime fixture, concurrency stress
suite, numerical calculation, or Lean build was executed. No theorem or
whole-language soundness result is claimed. Compiler and governance changes
are outside this chapter's write set.

A Darwin `coder-next` consultation reviewed a supplied findings summary for
overclaim and integration risks. It had no access to the local source files;
its output was advisory, and the findings in this chapter rest on the named
source inspections. No shared offload log was modified in this write set.

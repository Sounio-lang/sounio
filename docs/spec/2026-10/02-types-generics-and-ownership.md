<!-- docs:meta
topic_id: repo.docs.spec.2026-10.02-types-generics-and-ownership
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.02-types-generics-and-ownership
-->

# 02. Types, generics, refinements, and ownership

Research snapshot: 2026-10-16. Source baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Machine-readable traceability: [02-evidence.tsv](02-evidence.tsv).

This chapter records the current modular self-hosted implementation and the
boundaries that an eventual ratified specification must resolve. It is not a
claim that every described surface executes on every target. All fixtures and
gates cited below were inspected as source; none was executed for this chapter.
Compiler artifact identity, a successful check, native execution, and conformance
closure remain separate evidence requirements. The legacy `lean_single` engine
is a separate implementation, not an automatic oracle for Madaros semantics.

Each rule has a stable identifier. **Source-observed** means descriptive current
source behavior, including a limitation where explicitly stated.
**Target-contract** denotes an implementation boundary tied to an engine or
backend. **Open-decision** identifies an unresolved language choice; its proposed
acceptance criteria are not approved normative semantics. No theorem, soundness
result, or proof completion is asserted by the judgments in this chapter.

## 2.1. Semantic vocabulary and implementation boundary

### TYPE-001 — Checking judgments and staging (source-observed)

A *type expression* is a parsed `parser::ast::TypeExpr`; a *checker type* is a
`check::types::TypeEntry`; a *runtime representation* is a lowerer/backend choice.
These are different objects. `TypeExprKind` distinguishes named applications,
references, mutable references, raw pointers, arrays, tuples, function types,
unit, never, inference holes, and dedicated scientific wrappers. `TypeEntry`
holds a kind plus selected identity, structural, and side-table fields, including
`inner`, `tuple_elems`, `fn_sig_id`, `array_size`, `unit_id`, and `refinement_id`.
A tag's presence establishes representability in the checker, not an executable
implementation of every operation on it.

For exposition, write `Γ; B; C ⊢ e ⇒ T` for synthesizing a type while updating a
value environment Γ, borrow/consumption state B, and constraints C. Write
`Γ; B; C ⊢ e ⇐ T` for checking in an expected-type context. These are descriptive,
unproved models of the implementation's stateful procedures. They are not a
formal calculus with established preservation or progress.

Collection precedes body checking. The concrete entrypoints include
`check_program_with_artifacts`, the in-place `check_items_verdict_boot4` family,
and module-aware variants in `self-hosted/check/mod.sio`. The public check path in
`self-hosted/compiler/main.sio::run_check_mode` may specialize merged AST items
before checking them. The compile path also has
`module_frontend_specialized_prepare`. A statement about a helper must identify
whether this active pipeline reaches it; a module's header is insufficient.

### TYPE-002 — Primitive names are distinct from machine layouts (source-observed)

`checker_lower_named_type_mut` recognizes `i8`, `i32`, `i64`, `u8`, `u32`, `u64`,
`i128`, `u128`, `f32`, `f64`, `f128`, `f256`, `bool`, `string`, and `char`.
It maps `usize` to `ty_i64()` in this checker. It also recognizes the wide integer
names accepted by `name_wide_int_bits`; `ty_wide_int` carries width in
`clifford_p` and signedness in the signed/unsigned wide kind. This is an internal
encoding, not a relationship between wide integers and Clifford algebras.
Do not infer an implemented `i16`/`u16`/`isize` primitive merely from other
languages' integer families or historical prose.

`TyUnit` is the type of the empty tuple and ordinary statements without a value.
`TyNever` marks nonreturning expressions in the checker; `TyUnknown` represents
missing type information, and `TyError` supports diagnostic recovery. The last
two are not source-level universal types. `bool`, `char`, and `string` have
distinct checker kinds; an integer carrier or pointer-sized slot in generated
code does not identify any of them with `i64`.

Wide floating format identity is explicit in `TypeKind`,
`binary_format_for_type`, and `wide_float_format_identity_compatible`.
`f128` and `f256` are not aliases for `f64`. Ordinary `is_numeric_type` and
`is_float_type` exclude them; dedicated paths admit selected wide operations.
Neither accepting an annotation nor exposing a descriptor proves full arithmetic,
formatting, FFI, field, array, or target support. The numeric chapter owns the
operation-by-operation contract.

### TYPE-003 — Name lookup, aliases, and identity boundaries (source-observed)

`TypeEnv` is a stack of bindings containing a `Name`, `TypeEntry`, mutability,
and scope depth. `lookup` searches from the latest binding backwards; popping a
scope restores its starting count. Bare expression identifiers first consult
the environment, then function signatures, and otherwise report an undeclared
identifier through `checker_check_ident_expr_inplace`.

Type-position lookup is not a separately implemented, fully qualified type
namespace. `checker_lower_named_type_mut` handles builtin spellings, then an
in-scope generic parameter, then unit names, then informative environment
bindings, and finally creates a named type. Thus a generic `K` shadows the
kelvin unit. Aliases and ontology bindings can resolve through the same value
environment. Imported `TyUnknown` bindings do not override a meaningful named
type: unknown, error, and never bindings fall through to `ty_named` unless
ontology metadata makes the binding informative. Returning a `TyNamed` here
does not establish that every later use of an undeclared type is legal.

Named identity and definition lookup remain substantially name keyed.
`struct_table_find` and `enum_table_find` are not general module-qualified type
identity systems. The compiler's `module_struct_identity_check` refuses loaded
same-named structs with different definitions; identical copies are accepted.
This refusal contains an implementation collision and does not specify that two
separate modules' identically written types must forever be one language type.
Same-named enum, alias, and global identity must not be presumed fixed by that
struct-only check. Import/visibility rules are further specified by the module
chapter.

### TYPE-004 — Equality helpers are not universal type equality (source-observed)

There is no basis for equating every procedure named “equal” or “compatible.”
`check::epistemic::types_equal` first compares kind. For named types it compares
name, ontology ID, and recursively the optional inner type; other kinds recurse
through `inner`. Its current body does not compare array lengths, tuple lists,
function signatures, or wide-integer widths. Consequently this helper alone
cannot define equality for all Sounio types. For example, two array entries can
have equal inner elements under this helper while differing in length.

The separate `self-hosted/compiler/types.sio::types_equal` operates on a
different arena representation. It is not evidence that the active checker uses
that arena's equality. Consumers must use their actual boundary check. A future
language-level identity relation must define all relevant fields explicitly,
rather than promoting either helper into an unqualified normative axiom.

### TYPE-005 — Compatibility is a boundary relation (source-observed)

Use `A ≼site E` as notation for “actual type A is accepted for expected type E at
this site.” It is not a proven subtype relation. In `compat::types_compatible`:

- Equal ordinary primitive kinds are accepted; narrow signed kinds
  `{i8, i32, i64}` are mutually compatible, as are `{u8, u32, u64}`. This is
  broader than lossless widening. Cross-signedness is not admitted by this rule.
- Signed wide kinds must match signedness and stored width; likewise unsigned
  wide kinds. Narrow-width permissiveness does not extend to arbitrary widths.
- Array elements recurse and known lengths must agree; `-1` denotes unknown
  size on either side. Tuple compatibility is pairwise with equal list length.
- Reference and slice mutability must match in this helper. Raw pointers allow
  mutable-to-const direction, never const-to-mutable through compatibility.
- Ordinary named types compare names. `Seq<T>` additionally compares its inner
  element when both are present; a bare `Seq` acts as a wildcard. Specialized
  wrappers such as `Invariant` have dedicated rules.
- Function types compare signature IDs here; richer structural comparison is
  implemented elsewhere. `Knowledge` performs inner compatibility and directional
  epsilon, validity, and provenance subsumption.

Despite its “must be symmetric” header, this function contains directional
branches. `TyError`, `TyUnknown`, and `TyNever` usually compare compatible on
either side for recovery or incomplete information; the `Unobserved` guard runs
before those wildcards. This is neither ordinary bottom-type subtyping nor a
permission to erase an observer wrapper. Substitutivity, transitivity, and
symmetry are not established metatheorems.

### TYPE-006 — Contextual literal acceptance (source-observed)

Literal expressions normally synthesize `i64` or `f64`. An expected type can
admit a literal without admitting an expression or variable of that synthesized
type. `checker_contextual_expr_compatible` recognizes integer and floating
literals, their recognized unary-negative forms, and array literals containing
only qualifying literals. Array length compatibility still applies, including
repeat forms.

`checker_expected_accepts_int_literal` admits an `i64`-born literal for any
checker integer kind or a wide float; the helper itself performs no range test.
This is an implementation observation, not a specified wrap/trap/range policy.
`checker_expected_accepts_float_literal` admits an `f64`-born literal for `f32`
only within its explicit magnitude guard, and admits a wide-float expected type
for later lowering to resolve. An `f64` variable does not acquire `f32`
compatibility merely because its original initializer was a literal. Exact
wide-float lowering may still refuse an accepted source expression.

### TYPE-007 — Binding acceptance differs from called-parameter acceptance (source-observed)

At an annotated `let`, `checker_check_let_stmt_inplace` synthesizes the
initializer with selected expected-type handling, tests contextual compatibility,
and chooses the annotation as the binding type in the ordinary case. Scientific
certificate types have exceptions preserving inferred metadata. `var` follows
its own closely related handler and records mutability. Refinement checking is
a separate obligation after type compatibility.

Named call arguments pass through `checker_call_arg_types_compatible_inplace`
and additional argument-boundary checks. These include:

1. dimension-based compatibility for two unit-branded `f64` types;
2. authoritative epistemic subsumption before general compatibility;
3. structural function compatibility and contextual literals;
4. `&!T` to `&T`, mutable slice to shared slice, and selected array-reference to
   slice adaptation;
5. implicit borrowing of an identifier for a reference parameter;
6. additional ontology subsumption for named class arguments.

A value accepted by `f(x)` for `f`'s parameter is therefore not necessarily
accepted by `let y: ParameterType = x`. Implicit borrowing is a call-site rule,
not a general assignment coercion. Nor does type acceptance waive borrow,
effect, refinement, privacy, or backend checks. In particular `&T` must not gain
`&!T` capability merely because some call site can reborrow a mutable value.

### TYPE-008 — Struct definitions and construction (source-observed)

A struct has named fields, optional stored type parameters, visibility metadata,
and a linearity flag in `StructInfo`. Collection lowers fields and registers
the definition before bodies are checked. Struct literals are checked against
the field table; field access resolves the named type and its field metadata.
Missing fields, wrong field types, private-field access, and direct by-value
self-reference have explicit diagnostic paths. Direct recursive fields are
rejected by `struct_field_list_has_direct_self_ref` with E216; this does not
establish a complete recursive-layout or alias-cycle analysis.

Generic fields are substituted and may participate in initializer inference,
including a parameter occurring inside an array field. Equal runtime layouts do
not establish type identity. A source struct is not specified to have C layout
or a stable wire format. Borrowing a struct field does not imply that arbitrary
pointer aliases to that field are analyzed.

### TYPE-009 — Enums, payloads, and special optional values (target-contract)

`EnumInfo` records variants and their field lists, and parsed declarations can
retain tuple or struct payloads. Madaros CPU lowering models an enum without
payload variants as a discriminant and an enum with at least one payload variant
as a tagged cell (discriminant word, then one word per payload field, sized for
the largest variant; `docs/compiler/ENUM_PAYLOAD_LAYOUT.md`). The checker types
payload pattern bindings with the declared field types, checks constructor and
pattern arity (E277, E278), refuses `==` and integer casts on payload enums
(E279) and non-exhaustive matches over them (E280); E264 remains for payload
fields that are not one word. Generic enum declarations do not parse. The legacy
seed has separate failure behavior.

`Option<T>`/`Some`/`None` have dedicated checker and lowerer treatment and cannot
serve as proof that arbitrary user payload enums are operational. In particular,
the source has a `Some(0)` witness because absence must not simply mean a zero
payload. Generic payload enums, structural equality and destruction of payload
enums remain unimplemented. The current checker's enum-to-numeric binding/return
allowances apply to fieldless enums only; they are descriptive compatibility
accommodations, not an approved universal enum conversion design.

### TYPE-010 — Arrays and element selection (source-observed)

An array checker type contains an element type and a size. List literals check
elements; repeat literals check their repeated element and size expression.
Known unequal sizes are incompatible. Constant-size evaluation recognizes a
limited set of integer expressions and constant bindings; it is not arbitrary
compile-time program evaluation. Unknown size uses the internal `-1` sentinel,
not a proof that any particular size is safe.

`checker_check_index_inplace` checks the base before the index, requires an
integer index when concrete information is available, and yields the indexable
element type. It also handles `Seq<T>` specially. This type rule does not prove
index bounds or remove the runtime/effect obligations of indexing. The native
array representation, aggregate return layout, and wide element support must
be justified separately for the chosen target.

### TYPE-011 — Tuples and structural arity (source-observed)

Tuple types are ordered element lists; compatibility requires corresponding
compatible elements and equal arity. The empty tuple is unit. Tuple
destructuring must match the checked shape, and tuple projection has its own
index/field checks. A one-element tuple's aggregate or borrow layout is not
interchangeable with its element merely because both occupy a single slot in
some generated program.

The active literal helper `checker_check_tuple_elems_inplace` stores at most
16 element types and stops processing when that bound is reached. Its body does
not issue an overflow diagnostic. This is a source-visible unchecked boundary,
not a specified maximum tuple arity or permission to ignore later expressions.
The arity-negative fixtures cover ordinary mismatches, not this overflow case.

### TYPE-012 — References and dereferencing (source-observed)

`&T` and `&!T` lower to distinct shared/exclusive checker kinds. The latter is
Sounio's mutable-reference syntax; `&mut T` is not substituted in this chapter.
Reference checking resolves the operand and takes a borrow where a trackable
place exists. Dereferencing yields the inner type; write legality additionally
depends on the access path and mutable capability. A mutable binding and a
mutable reference are different facts. Call-site adaptation from exclusive to
shared does not establish general reference covariance, lifetime subtyping, or
ownership transfer of the referent.

### TYPE-013 — Slices are unsized borrowed views (source-observed)

A reference to an array type lacking a size is lowered by
`Checker.lower_ref_like_type` to `TySlice` or `TySliceMut`. The checker has
dedicated slice borrowing and index rules, and call adaptation can accept a
fixed-array reference for a compatible slice parameter. Element compatibility
and mutability remain relevant. The checker kind does not specify whether a
particular backend uses a fat pointer, separate length metadata, or another
carrier. `madaros_d3_openslice_len_gate.sh` is a focused length-path witness,
not whole-language slice parity.

### TYPE-014 — Raw pointers preserve mutability direction (source-observed)

`*const T` and `*mut T` use `TyRawPtr` with an inner type and a mutability marker.
Compatibility admits mutable to const and rejects const to mutable. This does
not prove alias exclusion, validity, alignment, initialization, ownership, or
referent lifetime for a pointer. Casts and backend operations form additional
boundaries; a pointer cannot be treated as a checked borrow merely because both
are represented by an address. The public typekind pair and
`kl1_raw_ptr_const_to_mut.sio` are rejection/acceptance intentions to rerun.

### TYPE-015 — Functions and function values (source-observed)

`TyFn` refers to a `FnSig` containing parameter and return types and effect
information. Equal signature IDs are a fast path; different IDs can still be
structurally compatible through `checker_fn_types_structurally_compatible_table`
or its in-place counterpart. These compare arity, return type, parameter types,
and provided effects against the expected row. Provided effects must be a subset
of the expected effects: a pure function may satisfy an effectful function slot;
the reverse is refused. The table deep comparison bounds recursion at 16.

This implementation is not a proved contravariant function subtype calculus:
its parameter loop invokes compatibility in the implemented direction. Where
compatibility is itself directional, standard mathematical function subtyping
must not be inferred. Concrete higher-order witnesses cover scalar cases and
type mismatches; they do not establish unrestricted higher-rank polymorphism.

### TYPE-016 — Bidirectional checking is selective (source-observed)

Expected-type information is used for contextual literals, selected binding
initializers, constructor paths such as annotated `Option::None`, closure
parameters, and generic argument inference. Most expression dispatch synthesizes
a type first and then invokes site-specific compatibility. Named function
declarations, explicit returns, implicit body tails, methods, and closures use
related but distinct paths; success at one is not evidence for all others.
`checker_fn_body_return_compatible_inplace`, for example, uses recorded literal
state for selected tail conversions rather than the original expression alone.

`check/infer.sio` contains a separate `TypePool`, first-order unification with
occurs check, schemes, generalization, and instantiation, plus helper tests.
The inspected active checker/driver imports do not wire this engine in as their
global inference algorithm. Its Algorithm W header therefore does not specify
principal types, general let-polymorphism, or unrestricted inference for user
programs. `TyUnknown` recovery also does not prove inferred polymorphism.

### TYPE-017 — Explicit casts do not promise arbitrary conversion (source-observed)

`checker_check_cast_expr_inplace` checks both source and target and applies
dedicated restrictions. Wide-float casts are rejected through E248 on this path.
Numeric/bool/char to `string` casts are rejected through E270 rather than
formatting a value. Epistemic wrapper erasure or strengthening is constrained by
its own boundary rules, and unit casts require compatible dimensions.
`Unobserved` casts have a dedicated capability-sensitive rule initially limited
to the `i64` carrier. A successful annotation check is not a license to obtain
another representation by `as`; runtime bit reinterpretation and numeric
conversion must be specified operation by operation.

### TYPE-018 — Declarations without initializers are target constrained (target-contract)

For an absent initializer, `checker_check_binding_init_expr_inplace` returns
the annotation when present and unit otherwise. This checker behavior alone
does not establish a language-level definite-assignment system. Native lowerers
support selected zero-initialized aggregate declarations; the seed refuses some
string and aggregate shapes. The `uninit_*` fixtures deliberately distinguish
engines. Such declarations must not be described as Rust-style uninitialized
memory with a proved prohibition on reads, nor as universal recursive
zero-initialization on every target.

### TYPE-019 — Resource limits are implementation facts (source-observed)

`TypeEnv` holds 4096 bindings and 64 scope starts; `BorrowEnv` holds 128 entries,
64 scope starts, and snapshots for 16 nested branch levels. Several insertion
helpers guard capacity by doing nothing; some boundaries have separate explicit
failures. The tuple limit and refinement-table limits described here are further
examples. These numbers are not language limits ratified by this document.
Type-expression lowering also has explicit depth and complexity guards
(`MAX_TYPE_DEPTH = 64`, `MAX_TYPE_COMPLEXITY = 1000`), unlike the silent
capacity cases; their existence must not be generalized to all arenas.
A defensible conformance claim needs a diagnostic at every exceeded bound or
evidence that the bound cannot be reached on that path. Silent loss of a binding,
borrow, type element, or obligation is a defect risk, not normative semantics.

## 2.2. Generics, traits, and specialization

### GEN-001 — Generic declarations have unequal AST support (source-observed)

`StructDef` stores its `TypeParamDefList`; function declarations currently do
not store their generic header on `FnDef`. `parse_type_param_header` consumes
function parameters and optional annotations without retaining a complete bound
environment. The active specializer recovers function parameters structurally
from single-uppercase-letter named types not known as local struct/enum names.
This is a compiler heuristic, not a rule restricting the intended language to
one-letter type parameter names or authorizing accidental capture of a user type.
Generic struct parameters have a more explicit identity path than those
recovered function formals.

### GEN-002 — Active specialization precedes checking (source-observed)

The AST pass in `check/specializer.sio` collects function/method and struct
templates, discovers instances, substitutes arguments, rewrites calls, and
appends concrete items. `specialize_generics` and `specialize_generics_multi`
are actual driver entrypoints. The multi-module path is wired in the current
source despite the obsolete “only single-module” sentence in the module header.
The driver checks specialized items when instances were created and retains the
ordinary module checker otherwise.

Function instance caching uses template slot plus the complete encoded type
argument key. Singleton functions may keep the original spelling; multiple
instances receive distinct symbols. Instance reservation occurs before recursive
discovery. This is not a stable user-visible name-mangling ABI. Generic methods
retain narrower singleton restrictions, and unsupported ambiguous instances are
intended to fail closed. Separate `monomorph.sio`, `specialization.sio`, and
`type_erasure.sio` contain other models/helpers; their existence does not prove
that the public pipeline runs an alternative post-check IR monomorphizer or
runtime dictionary/vtable strategy.

### GEN-003 — Generic inference is bounded and local (source-observed)

Explicit calls use the parsed type arguments, including turbofish forms such as
`identity::<i64>(value)`. Inferred calls use structural checker substitution and
selected AST inference. `spec_infer_call_type_args` only infers syntactically
proven scalar arguments from matching plain formals; it can skip a function-typed
formal and use a later scalar argument to determine the parameter. Its scalar
recognizer covers integer, floating, boolean, selected cast, and negative
literal forms. It does not re-execute checking, effects, or linear consumption.

The concrete higher-order pair `madaros_generic_hof_scalar_inference.sio` and
`madaros_generic_hof_inference_mismatch.sio` discriminates an inferred scalar
parameter from an incompatible function value. Generic struct field inference
also recurses into compound shapes, as the `Compound<F>` array-field fixture
demonstrates by intent. This does not establish inference for every constructor,
return-only parameter, effect variable, ambiguous overload, or recursive type.
The active specializer has a diagnosed four-type-parameter bound, with explicit
four-parameter positive and five-parameter refusal fixtures.

### GEN-004 — Trait syntax and static method dispatch (source-observed)

Traits parse declarations and method signatures; `impl Trait for Type` retains
the trait name in `ImplDef`. Collection registers ordinary method signatures
against the implementation receiver. `fn_sig_table_find_method_semantic`
uses more than the receiver's bare name, which matters for primitive and
unresolved imported receivers. Body checking resolves the implementation's own
signature and effects. The bounded trait-dispatch fixtures exercise this static
method path, including multiple calls and concrete struct receivers.

The active `collect_trait_def` creates a trait registry entry and collects method
signatures. It does not populate a complete signature-aware trait obligation
environment merely by calling `trait_def_new`. `checker_collect_impl_def_inplace`
collects methods without consulting the retained trait name to discharge a
complete trait implementation obligation. Consequently successful static method
dispatch does not establish trait objects, associated types, coherence, orphan
rules, defaults, supertraits, or dictionary passing in compiled user programs.

### GEN-005 — Trait obligations are not generally enforced (source-observed)

`traits.sio` defines `trait_satisfies_bounds`, `trait_check_impl_complete`,
`trait_check_supertrait_impls`, orphan/overlap checks, and vtable helpers over
its own registries. The inspected active checker call sites initialize builtin
registries and collect declarations, but do not call those checks to validate
every generic invocation. The parser also discards critical function-header
bound information. Thus `fn add<T: Add>(...)` being parsed or instantiated is not
evidence that the `Add` obligation is checked at each call.

`tests/ui/type/generic_constraint.sio` explicitly describes its pinned rejection
as an undefined-type failure. It is a negative syntax/type witness, not a
conformance witness for solving an unsatisfied trait bound. Do not replace the
missing obligation with “duck typing” as a normative language decision: a
specialized body failing for a missing method is different from validating the
declared trait relation.

### GEN-006 — Specialization must retain provenance of definitions (source-observed)

Specialized items need the original definition's module and visibility. Current
`check_items_verdict_boot4_with_module_map` uses the specializer's item-to-module
mapping rather than assigning every merged item to module zero. Generic
functions and structs also carry template visibility into generated items.
The `tests/multimodule/visibility_*_specialized_generic_*` fixtures include
private function, struct, enum, and field rejection intentions.

This mapping repairs particular semantic boundaries; it is not a complete
module-qualified identity model. Newly synthesized multi-instance symbols,
duplicate module basenames, re-exports, and unsupported generic receiver shapes
need independent evidence. Template identity and generated symbol spelling must
not be conflated, and generic specialization must not create access rights that
the unspecialized declaration lacks.

## 2.3. Ownership, borrowing, release, and lifetime boundaries

### OWN-001 — Active consumption categories (source-observed)

The active checker stores a `BorrowEnv` and uses its `is_linear` and `consumed`
fields. `checker_is_linear_type_inplace` recognizes named structs marked linear
and function signatures marked as linear closures. It does not recursively
derive linearity for every array, tuple, enum, or wrapper containing a linear
element. An ordinary struct is therefore not documented here as automatically
affine or move-only.

`StructDef` retains `is_affine`, but `StructInfo` and the active linear classifier
do not carry that distinction through. The `affine_copy.sio` UI fixture expresses
an intended rejection, not proof that general affine enforcement is active.
`ownership.sio::OwnTypeKind` separately models linear, affine, copy, and drop
categories; active checker references found for `own_*` only initialize its
context. That model must not be substituted for actual expression enforcement.

### OWN-002 — A linear read can consume the whole binding (source-observed)

For a linear identifier, `checker_check_ident_expr_inplace` checks borrowing and
calls `BorrowEnv.consume_linear`, unless a deliberate suppression depth applies.
A second consumption reports an error; unconsumed tracked linear bindings are
checked at scope exit. Passing by value, returning, destructuring, and field
access can therefore transfer consumption through their identifier checks.
This is not a claim of general per-field partial-move ownership: reading a field
of a linear struct can consume the binding as a whole, and later field access
is covered by dedicated negative fixtures.

Use `B(x) = available` and `B(x) = consumed` as an unproved description of this
state transition. A borrow cannot consume the owned value while the conflicting
borrow remains active. Ignoring a linear result or overwriting an unconsumed
linear binding has separate failure witnesses. The standalone `OwnState`
state machine contains richer states; these are not all active language states.

### OWN-003 — Consumption must agree across tracked branches (source-observed)

`checker_branch_enter_inplace`, `checker_branch_then_done_inplace`, and
`checker_merge_if_branches_inplace` snapshot consumption, rewind before checking
the alternative, and merge. A previously available linear binding is consumed
after the join only when both paths consume it. Exactly one consuming path
reports E040. A missing `else` is an unconsuming path, so “consume only if true”
does not satisfy the requirement even when the condition is written as a
constant in a fixture.

This is implemented bounded branch tracking, not a complete control-flow
resource proof. The snapshot array covers 16 nested levels; its documented
fallback is sequential behavior beyond that level. Loops, early returns,
matches, shadowing, and nested destructuring each need their own discriminating
fixtures. The present chapter does not infer that a while body executes exactly
once, or that zero-iteration loops discharge a resource obligation.

### OWN-004 — Parameters and linear closures (source-observed)

Function and method parameter scopes are tracked separately from body locals;
their linear obligations are checked after body checking, including explicit
return paths. Closure parameters also participate. The closure checker infers
body effects in a fresh effect context and observes outer linear bindings that
transition to consumed while checking the body. Such captures mark the closure
signature linear, so the callable itself becomes a consumed resource.

Capturing a resource is distinct from merely having an unrelated linear local
in the surrounding scope. The `closure_unrelated_linear_not_captured.sio` fixture
exists for that boundary. These checker rules do not establish escaping heap
environments, unlimited captures, or cross-target closure ABI support. The older
`closure_linear.sio` fixture is explicitly ignored and cannot be counted as a
passing witness. Native-v2 closure fixtures include supported direct calls and
deliberate refusal shapes for returned/aliased/nontransparent captures.

### OWN-005 — Borrow exclusion is place based within a bounded subset (source-observed)

The active borrow states are unborrowed, shared with a count, and exclusive.
A shared borrow conflicts with an existing exclusive borrow; an exclusive borrow
conflicts with any existing shared or exclusive borrow. `borrow_paths_overlap`
compares textual places: equal paths overlap, as do a root and descendant at a
dot boundary. Sibling fields such as `s.a` and `s.b` are disjoint in that model.
`checker_take_borrow_inplace` checks tracked fields against whole-variable loans
and vice versa.

`checker_borrow_place_name` strips indexing to the containing array place, walks
bounded field chains, and declines this named-field tracking when the root is
itself a reference or raw pointer. Thus `s.a[i]` and `s.a[j]` are not proved
disjoint by unequal indices. Arbitrary alias analysis, pointer-derived places,
and nested reborrow soundness do not follow from sibling-field witnesses.

### OWN-006 — Calls introduce temporary borrow obligations (source-observed)

Reference-parameter calls can borrow an identifier implicitly; explicit `&x` or
`&!x` arguments take loans during expression checking. A nested argument is
checked while earlier argument loans can still be active. Therefore
`set(&!v, index, get(&v, other))` has an explicit E037 negative fixture; hoisting
the read changes the lifetime of that temporary borrow and is a distinct
program.

`checker_release_call_arg_borrows_inplace` releases named call-site loans after
checking the call, recognizing implicit reference parameters and explicit unary
borrow arguments. Its body releases unconditionally by name; the passed
`call_start_borrows` snapshot is not used there to establish that a loan was
successfully acquired by this call. The UI failure-guard fixtures require that
a failed acquisition must not release a pre-existing loan. This is an explicit
source/fixture tension awaiting execution and repair; unconditional release is
not ratified language semantics. A source-level release helper must not be
invented from the English word “release”: the conformance fixture refers to
automatic release of explicit call arguments.

### OWN-007 — Scope and loop release is not complete last-use analysis (source-observed)

`BorrowEnv.pop_scope` discards entries introduced since that scope's start and
resets those entries' loan state. `release_borrow` releases one exclusive loan
or decrements a shared count by name. It does not identify a general owner of a
reference binding and discharge every loan implied by that binding's lexical
end. `release_exclusive_borrows` resets all tracked exclusive states and is
invoked at loop-checking boundaries. These are concrete implementation
mechanisms, not a proved nonlexical lifetime system.

In particular, a lexical scope exit and the last evaluation of a reference are
different events; a guarantee that all loans on outer variables are restored
correctly requires additional provenance and control-flow evidence. The
`borrow_match_scope_ok.sio` and conflict witnesses are bounded cases to rerun.

### OWN-008 — Reference escape provenance is conservative and incomplete (source-observed)

The checker records a small escape provenance: safe/first-class versus rooted
in a frame-local binding. `checker_expr_provenance` propagates it through
identifiers, references, selected aggregate construction, and branch/block tails;
binding handlers preserve it. Function tails and explicit returns reject detected
frame-local reference escapes. Parameter-rooted paths such as `&self.field`
are distinguished from borrowing an ordinary local. This is a lifetime-related
check, not a parametrically quantified return-lifetime signature.

No universal inference is made for arbitrary call results, aliases, raw pointers,
or closures containing references. The active AST has no general lifetime
parameter carried through every `TypeEntry`. The source comments acknowledge
unhandled paths and backend refusal boundaries. No focused positive/negative
escape pair was established in this research pass; this coverage gap is visible
in the evidence catalog instead of being filled by a guessed fixture.

### OWN-009 — Drop scheduling helpers do not establish destructors (source-observed)

`ownership.sio` models scope/last-use/manual drop strategies, explicit drops,
partial moves, branch merges, and scheduled drop points. These helper algorithms
are not evidence of inserted user destructors in the active checker-to-IR path.
The active `OwnContext` initialization is not a call to that ownership analysis.
Likewise a builtin `Drop` trait registry entry is not proof that user `Drop`
implementations run on every exit path.

Linear consumption, borrow release, runtime arena reclamation, and user-visible
destruction are distinct. The active linear checker diagnoses missing consumption;
it does not promise to repair it with automatic destruction. Destruction order,
unwinding, partial construction, reinitialization, and dropping captured values
remain unratified without integrated lowering and target-specific tests.

### OWN-010 — Standalone lifetime solvers are not integrated guarantees (source-observed)

`lifetimes.sio` defines regions, outlives constraints, solving, and conflict
checks. `borrows.sio` defines another borrow-checker model with its own arena
and tests. The active `check.sio` imports `borrow.sio`, not these modules, and
the inspected driver/checker entrypoints do not invoke their solvers. Their
presence supports research into lifetime algorithms; it does not establish
nonlexical lifetimes, borrow polymorphism, lifetime elision rules, or a memory
safety theorem for compiled Sounio.

## 2.4. Refinements and constraint discharge

### REF-001 — Refinements decorate a base checker type (source-observed)

Surface refinements include `{ x: T | predicate }` and named aliases for that
form. The AST stores a base, binder, scalar comparison and, for supported
compound syntax, a second comparison with conjunction/disjunction. Lowering
registers a `RefinementType` and returns the base `TypeEntry` with a
`refinement_id`. Refinements are not a second runtime representation by this
fact alone.

`RefinementPred` supports comparisons `>`, `>=`, `<`, `<=`, `==`, `!=`, plus
`And` and `Or`; scalar constants use integer/floating fields. A richer parsed
expression is not necessarily representable in that predicate record. The
coefficient in `x * 2 < 200`, for example, is not encoded by pretending it was
`x < 200`. Declaration warning W046 marks unrepresentable predicates, including
partially unsupported conjunctions, rather than silently enforcing one half.

### REF-002 — Predicate checking is separate from base compatibility (source-observed)

Literal and tracked constant values can be checked against the stored predicate
at annotated binding, field, call, and return boundaries. The checker records
proved, unknown, and violated outcomes; a represented violated predicate has an
E042 diagnostic path. Explicit returns and implicit named/method tails have
separate callers of `checker_check_refinement_literal_inplace`.

An ordinary compatible integer does not thereby satisfy `{ x: i32 | x > 0 }`.
Conversely, a failed historical fixture may have rejected due to named/base
type mismatch rather than evaluating the predicate. The current source lowers
refinements onto the base type, so old fixture comments describing an earlier
named-type-only failure are not a current semantic rule. Diagnostic provenance
must be verified by rerunning the exact engine and fixture.

### REF-003 — Implication and guards provide bounded static evidence (source-observed)

`pred_implies` handles selected atomic and compound implications and interval
reasoning, ordinarily requiring matching binder names. A structural variant
ignores binder names for appropriate ontology property use. At a call boundary,
the checker tries a propagated integer constant, an argument's stored refinement,
enclosing-condition facts through `refine_static_check_var`, and selected
ontology property conjunctions. A positive guard can discharge an obligation
inside its guarded region; it must not establish the same predicate after the
guarded region or after an invalidating assignment.

These are bounded procedures over represented predicates, not general
dependent-type checking or an automatic theorem prover. Failure to establish an
implication is not proof of its negation. In particular, the call-boundary branch
for an argument already carrying a refinement emits E042 when `pred_implies`
returns false, rather than continuing to the unknown-obligation warning path.
This diagnostic can therefore express a refused implication, not a computed
counterexample. Floating comparisons, integer
overflow, binder identity, compound decomposition, and ontology assumptions all
need explicit proof models before this analysis can support formal claims.

### REF-004 — Unknown obligations currently warn rather than prove (source-observed)

When `checker_check_call_arg_refinement_boundary_inplace` cannot establish a
call's obligation, it emits W040 and permits compilation. Its
`refine_runtime_checked` counter name is not proof that a runtime assertion was
generated: the source explicitly labels debug assertion insertion as future
work and describes release behavior as trusting the programmer. The specified
evidence category is therefore *unproved obligation with warning*, not
“statically proved” or “runtime checked.”

Unrepresentable predicates use the same unknown outcome at the call and can
also receive W046 at the declaration. This is a material safety limitation for
code treating a refinement annotation as unconditional validation. A soundness
claim cannot silently assume W040 is fatal or that warning suppression and
proof-forcing attributes mentioned as future work are implemented.

### REF-005 — Return predicates and epistemic `where` are separate surfaces (source-observed)

Refinement return obligations have dedicated negative fixtures for explicit
returns, implicit function tails, and method tails; acceptance of an ordinary
refinement is not evidence that these contracts have been discharged.
The `refinement_holds_at_*` filenames describe predicates holding at those
boundaries; they do not establish a `holds` source keyword or contract syntax.
`Knowledge<T where { ... }>` is explicitly refused until its predicate is
enforced. Both a value intended to satisfy the predicate and an invalid value
have refusal fixtures; the valid example's rejection is an unsupported-surface
boundary, not proof that the program's medical/domain predicate is false.
Epistemic confidence/provenance/validity constraints are separate from numeric
predicate validation and are owned in detail by the epistemic chapter.

### REF-006 — Solver modules and capacity do not establish completeness (source-observed)

`refinement.sio` implements the active small predicate/interval machinery.
`dependent.sio`, `gadts.sio`, and `smt_qflia*.sio` provide additional research
surfaces; their existence alone establishes neither full dependent types nor
an SMT-backed proof for each source annotation. An audit must identify a live
caller and its failure policy before making such a claim.

`RefinementTable` has 64 entries while several checker guards test IDs against
128; `refinement_table_add` stops adding at capacity. In inline lowering the
resulting `count - 1` can consequently refer to an earlier entry once insertion
stops. This is a source-visible obligation-identity risk, not approved
refinement semantics. Fresh boundary tests and a fail-closed implementation are
needed before describing arbitrary numbers of refinements as safe. The focused
`refinement_gate.sh` permits skips unless its zero-skips option is enabled;
its success text alone would not establish full fixture execution.

## 2.5. Open language decisions

### TYPE-020 — Identity, coercion, and inference contract (open-decision)

The language needs an approved definition of nominal identity across modules,
alias transparency, structural type equality, and field/tuple/array/function
metadata participating in identity. It also needs a directed coercion policy
for integer narrowing and literals, enum-to-number conversion, mutable views,
function parameters, and scientific metadata nested inside containers. The
current compatibility function is not that definition. Candidate acceptance:
separate identity, recovery compatibility, directed assignment/call conversion,
and join; require paired witnesses at let/var/field/argument/return/tail sites
and refusal rather than silent truncation at implementation limits.

### GEN-007 — Persistent binders and checked trait obligations (open-decision)

Retaining function generic binders/bounds in the AST, defining their scope and
inference, checking implementation completeness/coherence, and deciding static
specialization versus any future dynamic dispatch need explicit language and
compiler decisions. Candidate acceptance: a declared bound is retained through
parsing and specialization, a satisfying implementation is accepted, a missing
or mismatched one fails for the obligation itself, and equivalent imported
definitions preserve visibility. The existing undefined-type negative is not
an adequate replacement for that pair.

### OWN-011 — Resource and refinement safety boundary (open-decision)

Decisions remain on affine semantics, automatic resource classification of
composites, field moves, lifetime parameters, last-use versus lexical loan
release, destructors, closure escapes, and handling unknown refinement
obligations. Candidate acceptance: preserve the intended exactly-once/at-most-once
distinction, never discharge a loan that was not acquired, classify unknown
proof obligations explicitly, and fail closed when tracking capacity is
exhausted. These criteria do not authorize this chapter to choose destructor
order, mandate Rust's borrow rules, or silently strengthen W040 into an error.

## 2.6. Detailed evidence and validation boundary

All paths are relative to the repository root. The TSV has one row per rule,
with a primary source/symbol and at most one representative path in each
positive/negative cell. The following expanded table records complementary
evidence and why it is discriminating. “Positive” and “negative” describe the
fixture's intended role, never a result obtained in this research pass.

| Area | Current implementation references | Positive witness intention | Negative or boundary witness intention |
|---|---|---|---|
| Collection and checker routing | `self-hosted/check/mod.sio::check_program_with_artifacts`; `self-hosted/compiler/main.sio::run_check_mode`; `self-hosted/compiler/module_frontend.sio::module_frontend_specialized_prepare` | `tests/conformance/manifest.v1.tsv`, core and generic rows | `tests/ui/type/comparison_type_mismatch.sio`; bounded spine is not full checker parity |
| Primitive/representation distinction | `self-hosted/check/types.sio::TypeKind`, `TypeEntry`, `ty_wide_int`; `self-hosted/check/numeric_format.sio` | `tests/typekind/i128/pass.sio`; `tests/typekind/f128/pass.sio` | `tests/typekind/i128/refuse.sio`; `tests/typekind/f256/refuse.sio`; no implied all-width ABI |
| Names and aliases | `self-hosted/check/check.sio::checker_lower_named_type_mut`; `self-hosted/check/env.sio::TypeEnv`; `self-hosted/compiler/private_fn_identity.sio::module_struct_identity_check` | `tests/run-pass/unit_builtin_global_and_generic_shadow_2874.sio`; `tests/run-pass/madaros_module_same_struct_identical_copies.sio` | `tests/compile-fail/madaros_module_same_struct_differs.sio` |
| Literal context | `self-hosted/check/check.sio::checker_contextual_expr_compatible`, `checker_expected_accepts_int_literal`, `checker_expected_accepts_float_literal` | `tests/witness_matrix/cases/w11_float_literal_f32.sio`; `tests/witness_matrix/cases/w1_literal_coercion.sio` | `tests/compile-fail/f32_nonliteral_coerce_must_reject.sio`; integer range policy remains open |
| Struct fields and layouts | `self-hosted/check/check.sio::checker_collect_struct_def_inplace`, `checker_check_field_access_inplace`; `self-hosted/check/defs.sio::StructInfo` | `tests/run-pass/generic_struct_compound_field_inference.sio`; `examples/native/struct_basic.sio` | `tests/ui/type/struct_missing_field.sio`; `tests/compile-fail/recursive_struct_e216.sio`; `tests/selfhost/native_typecheck/struct_shorthand_type_mismatch.sio` |
| Enum representation | `self-hosted/check/check.sio::checker_collect_enum_def_inplace`, `checker_check_call_expr_inplace`; `self-hosted/ir/lower.sio::lower_match_arm_ref` | `tests/selfhost/native_runtime/enum_payload_then_plain_match_42.sio`; `tests/run-pass/option_scalar_match_issue2787.sio` | `tests/run-pass/enum_payload_scalar.sio`; `tests/compile-fail/enum_payload_match_not_exhaustive.sio`; `tests/compile-fail/enum_payload_tuple_field_refused.sio` |
| Arrays and tuples | `self-hosted/check/check.sio::checker_check_array_lit_inplace`, `checker_check_tuple_elems_inplace`, `checker_check_index_inplace` | `tests/typekind/array/pass.sio`; `tests/run-pass/tuple_element_borrow.sio` | `tests/selfhost/native_typecheck/array_return_length_mismatch.sio`; `tests/compile-fail/tuple_destructure_arity_mismatch.sio`; `tests/ui/type/tuple_index_oob.sio` |
| Slices and raw pointers | `self-hosted/check/check.sio::slice_param_accepts_array_ref`; `self-hosted/check/compat.sio::types_compatible` | `tests/typekind/slice_mut/pass.sio`; `tests/typekind/raw_ptr/pass.sio`; `scripts/ci/madaros_d3_openslice_len_gate.sh` | `tests/typekind/slice_mut/refuse.sio`; `tests/compile-fail/kl1_raw_ptr_const_to_mut.sio` |
| Function value type/effect boundary | `self-hosted/check/check.sio::checker_fn_sig_effects_compatible`, `checker_fn_types_structurally_compatible_table` | `tests/run-pass/madaros_generic_hof_scalar_inference.sio`; `tests/run-pass/closure_generic_hof.sio` | `tests/compile-fail/madaros_generic_hof_inference_mismatch.sio`; `tests/compile-fail/closure_effect_escape.sio` |
| Generics and instance identity | `self-hosted/check/specializer.sio::spec_fn_instance_find`, `spec_ensure_instance`, `spec_infer_call_type_args`, `specialize_generics_multi` | `tests/run-pass/specializer_scalar_multi_instance.sio`; `tests/run-pass/specializer_nested_targ_distinct.sio`; `tests/run-pass/kl3_specializer_four_type_params.sio` | `tests/compile-fail/kl3_specializer_five_type_params.sio`; `tests/compile-fail/turbofish_concrete_type_mismatch.sio` |
| Traits and visibility | `self-hosted/check/defs.sio::fn_sig_table_find_method_semantic`; `self-hosted/check/check.sio::checker_collect_impl_def_inplace`; `self-hosted/check/mod.sio::check_items_verdict_boot4_with_module_map` | `tests/run-pass/trait_bounded_dispatch_multi_call_i64.sio`; `scripts/ci/madaros_trait_i64_cd_exact_gate.sh` | `tests/ui/type/generic_constraint.sio`; `tests/multimodule/visibility_fn_private_specialized_generic_main.sio` |
| Linear consumption and branches | `self-hosted/check/check.sio::checker_check_ident_expr_inplace`, `checker_merge_if_branches_inplace`, `checker_require_current_linear_scope_consumed_inplace` | `tests/run-pass/linear_consume_once.sio`; `tests/run-pass/linear_balanced_branches.sio`; `tests/run-pass/linear_struct_destructure_let.sio` | `tests/compile-fail/linear_double_use.sio`; `tests/compile-fail/linear_consumed_in_one_branch_only.sio`; `tests/compile-fail/linear_multiple_field_access.sio`; `tests/compile-fail/linear_loop_consume.sio` |
| Parameter and closure resources | `self-hosted/check/check.sio::checker_check_closure_expr_inplace`, `checker_bind_fn_params_inplace` | `tests/run-pass/linear_closure_parameter_consumed.sio`; `tests/run-pass/closure_unrelated_linear_not_captured.sio`; `tests/selfhost/native_runtime/native_v2_closure_capture_direct_42.sio` | `tests/compile-fail/linear_method_parameter_unused.sio`; `tests/compile-fail/closure_linear_double_call.sio`; `tests/compile-fail/madaros_closure_capture_capacity_refuses.sio`; ignored `tests/run-pass/closure_linear.sio` |
| Place and call loans | `self-hosted/check/borrow.sio::borrow_paths_overlap`; `self-hosted/check/check.sio::checker_take_borrow_inplace`, `checker_release_call_arg_borrows_inplace` | `tests/run-pass/borrow_disjoint_fields_issue_1581.sio`; `tests/run-pass/borrow_call_explicit_release.sio` | `tests/compile-fail/borrow_whole_and_field_alias_issue_1581.sio`; `tests/compile-fail/nested_shared_borrow_in_exclusive_call_2958.sio`; `tests/ui/ownership/call_boundary_failed_explicit_release_guard.sio`; mirror/implicit guard files in the same directory |
| Refinement representation and obligations | `self-hosted/check/refinement.sio::RefinementPred`, `pred_implies`, `refine_static_check_var`; `self-hosted/check/check.sio::checker_check_call_arg_refinement_boundary_inplace` | `tests/run-pass/refinement_inline.sio`; `tests/frontend/refinement_interval_entailment.sio`; `tests/run-pass/refinement_guard_and.sio`; `tests/run-pass/refinement_f64_return_valid.sio` | `tests/compile-fail/refinement_inline_param_violation.sio`; `tests/compile-fail/refinement_call_arg_var.sio`; `tests/compile-fail/refinement_guard_and_escape.sio`; `tests/compile-fail/refinement_holds_at_method_tail.sio` |
| Unsupported epistemic predicates | `self-hosted/check/check.sio::checker_refuse_knowledge_where_mut` and dedicated Knowledge lowering | Ordinary refinement witnesses do not prove this surface | `tests/compile-fail/knowledge_where_refinement_refused_valid_2753.sio`; `tests/compile-fail/knowledge_where_refinement_refused_invalid_2753.sio` |

### Gates to execute in an authorized validation lane

The bounded spine is `scripts/ci/serious_language_conformance_gate.sh`; its
manifest currently includes generic structs/functions and explicit-borrow
release/conflict, but not all obligations above.
`scripts/run_sio_test_suite.sh` is the suite runner and respects fixture
requirements/ignores; an ignored fixture or engine-specific skip is not a pass.
`scripts/ci/refinement_gate.sh` filters refinement cases and has an explicit
zero-skips option. `scripts/ci/bootstrap_generic_array_layout_gate.sh`,
`scripts/ci/madaros_trait_i64_cd_exact_gate.sh`, and
`scripts/ci/madaros_d3_openslice_len_gate.sh` cover narrower implementation
paths. Before claiming a result, record source/artifact/stdlib identity, engine,
target, command, exit, diagnostic/stdout assertions, and skips.

This chapter's checks are source/fixture inspection and documentation
traceability only. No compiler build, source fixture execution, CI gate, SMT
solver run, or Lean build is claimed. Independent local-model audit requests
covering ownership/traits and compatibility/literal boundaries both returned
router HTTP 500 errors; they supply no corroborating review result. Compiler,
registry, index, shared-log, and governance edits are outside this chapter's
write set.

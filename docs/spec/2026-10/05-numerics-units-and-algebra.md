<!-- docs:meta
topic_id: repo.docs.spec.2026-10.05-numerics-units-and-algebra
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.05-numerics-units-and-algebra
-->

# 05. Numerics, units, and algebra

Status: source-grounded specification draft, 2026-10-16. Research baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. The mechanically readable rule
catalog is [05-evidence.tsv](05-evidence.tsv). Paths below are repository-relative;
identifiers name the inspected implementation rather than guessed line numbers.

This chapter describes contracts and their present implementation boundaries. It
does not promote every accepted expression to a supported executable operation.
No compiler, numerical oracle, conformance gate, or Lean build was executed for
this chapter. A fixture cited as positive is an intended positive witness in the
repository, not a newly observed pass. A negative fixture is likewise an intended
refusal or counterexample; a diagnostic comment is not evidence that the current
artifact emits that diagnostic. Source inspection establishes the routing and
limitations described here, subject to the explicit qualifications below.

Each rule has one standing in the companion catalog:

| Standing | Meaning in this chapter |
|---|---|
| `specified-contract` | An existing project commitment, principally `FOUNDER_INTENT.md`; not a new language ruling. |
| `source-observed` | A descriptive property of inspected source, including defects. |
| `library-contract` | A named library API and its stated domain; not automatic compiler support. |
| `target-contract` | A bounded lowering or runtime profile for a named engine/target. |
| `open-decision` | A decision or coverage obligation still needed before a portable normative rule can be adopted. |

The serious-language policy in `docs/serious-language/conformance-spine.md`
requires bounded executable coverage and readiness authorization before normative
promotion. The older `docs/spec/S12_NUMERIC_TOWER.md` is historical evidence,
particularly for earlier width erasure. Its statements that `f128` is reserved
and that every integer width is executed as `i64` are superseded by the current
format descriptors, wide IR operations, and fixtures inspected here. They are
not the language design adopted by this chapter. Conversely, newer prose saying
an entire residual is closed does not establish untested boundary behavior.

## 5.1 Semantic objects and evidence boundaries

### `NUM-001` — Separate representation, error, uncertainty, and evidence

Standing: specified-contract. A numeric value has an arithmetic representation
and an interpretation. Its stored bits, a correction for rounding error,
measurement uncertainty, provenance, and confidence are distinct information.
Neither a wider format nor a successful calculation establishes measurement
accuracy, a mathematical theorem, or a scientific mechanism. This is the
existing precision and epistemic arithmetic commitment in `FOUNDER_INTENT.md`.

In particular, an exact representation of a binary floating-point input is not
an exact representation of the physical quantity measured. A zero correction
lane does not establish zero measurement uncertainty; a zero uncertainty lane
does not establish that the arithmetic was exact. This distinction also applies
to constants declared without uncertainty and to results named `proved` by a
bounded runtime API.

## 5.2 Integer values and operations

### `NUM-002` — Integer identity precedes executable width support

Standing: source-observed. `self-hosted/check/types.sio::TypeKind` distinguishes
`TyI8`, `TyI32`, `TyI64`, `TyU8`, `TyU32`, `TyU64`, `TyI128`, and `TyU128`.
`ty_wide_int` stores the declared width in `TypeEntry.clifford_p` for the latter
two kinds. `self-hosted/check/compat.sio::name_wide_int_bits` recognizes an `i`
or `u` followed by decimal digits yielding a positive width. Consequently the
recognized namespace is broader than a closed C-like list of standard widths.
`i16` and nonstandard widths take this general-name route; that does not supply
each width with its own native instruction or truncation rule.

`checker_lower_named_type_mut` maps `usize` to `ty_i64`; this snapshot does not
justify a portable claim that `usize` has unsigned pointer-sized arithmetic.
`int_widths_compatible` permits compatibility within the narrow signed family
and within the narrow unsigned family. Wide same-kind compatibility additionally
checks width: `i128` and `i256` are not interchangeable, nor are `i128` and
`u128`. Do not infer that every accepted arbitrary width has a complete ABI.

### `NUM-003` — Narrow scalar arithmetic uses a machine-word profile

Standing: target-contract. The inspected Madaros x86-64 scalar arithmetic paths
in `self-hosted/native/codegen_x86_linux.sio::nc_emit_core_binop` and
`self-hosted/native/lower_ir.sio::emit_binop_code` emit machine-word add,
subtract, multiply, shifts, and comparisons. It does not mask every result to
the declared narrow width. `int_arg_narrow_nonliteral.sio` intentionally admits
an out-of-`i32`-range `i64` argument and checks only a positive result; it is
evidence of coercion acceptance, not of correct narrowing.

Physical storage and scalar arithmetic must be described separately. The BSS and
raw-array lowering in `self-hosted/ir/lower.sio` and
`self-hosted/native/codegen_x86_linux.sio` contains packed-byte handling,
including sign-extending `i8` loads. A packed byte array does not prove that an
`i8` local addition wraps at eight bits. Likewise, a layout helper listing four
bytes for a type does not prove that the active arithmetic path uses binary32 or
32-bit integer arithmetic.

### `NUM-004` — Signed and unsigned division, remainder, and order

Standing: target-contract. For the ordinary 64-bit integer path, signed division
uses `cqo` and `idiv`; unsigned division zeroes the high dividend word and uses
`div`. Signed remainder is the remainder from signed division, not a Euclidean
modulus: the signed fixture requires `-7 / 2 == -3` and `-7 % 2 == -1`.
`OpUDiv`, `OpURem`, and `OpUShr` preserve unsigned interpretation when selected
by `Lowerer.lower_binary_expr_ref`.

`tests/run-pass/u64_unsigned_param_shr_cmp_div.sio` distinguishes operands with
the top bit set, literal versus typed right operands, function parameters, and
signed controls. `i64_signed_shr_divmod_witness.sio` supplies the opposite-sign
controls. These fixtures are stronger than tests restricted to small positive
values. Mixed signed/unsigned expressions require their actual checker and
lowering route to be examined; this chapter specifies no universal usual
arithmetic conversion rule.

### `NUM-005` — Shift amount and shifted-value signedness are separate

Standing: target-contract. Signed scalar right shift emits arithmetic `sar`;
unsigned scalar right shift emits logical `shr`. The signedness of a shift
count must not convert a signed shifted value into an unsigned one. Current
lowering has dedicated handling of the left operand, including parameter
metadata, rather than treating the count as the numeric interpretation of the
whole expression.

For scalar constant folding, `self-hosted/ir/optimize.sio::try_fold_binop` folds
shifts only when the count is at least zero and less than 64. The x86 runtime
emitter supplies the count in `cl` without the same explicit source-level range
test. A count outside that interval is therefore outside the established common
contract, even if the processor produces a repeatable result. Negative counts,
counts equal to the declared width, and arbitrary-width shift truncation await
a language decision. `tests/ui/type/shift_float.sio` is a type rejection, not a
runtime range test.

### `NUM-006` — Integer constants have a bounded parser payload

Standing: source-observed. `self-hosted/lexer/numparse.sio` accumulates decimal,
hexadecimal, binary, and octal integer text into an `i64`; `Expr.int_val` and
the ordinary IR immediate are also machine-word payloads. Underscore separators
do not create arbitrary precision. These accumulation loops do not themselves
implement a general checked arbitrary-precision integer literal parser.

Wide values can instead be constructed from fitting literals and wide
operations. `tests/run-pass/kl7_wide_sign_extension.sio` constructs high powers
through multiplication and checks high limbs. That witness does not establish
that writing the same full integer as decimal text preserves all digits.
Overflow during literal accumulation, especially the boundary between a signed
positive literal and unary negation for the minimum value, needs discriminating
parser and execution tests before a portable literal-range rule is adopted.

### `NUM-007` — Overflow and constant evaluation are not one settled policy

Standing: source-observed. Ordinary scalar add/subtract/multiply emit no general
overflow trap in the inspected arithmetic helper. Retaining the hardware result
is a property of that path; it is not a founder-approved wrapping policy for all
integer types. No checked/saturating/wrapping family is inferred from these
instructions.

The compiler contains several evaluators. `self-hosted/ir/optimize.sio` declines
to fold integer division or remainder by zero. The separate
`self-hosted/check/const_eval.sio::ConstValue` uses `i64` and `f64` payloads;
its `const_div_int` and `const_mod_int` use zero sentinels. The independent
`self-hosted/ir/constant_fold.sio` also documents a zero sentinel. The presence
of these helpers does not establish that all three are wired into the same
active expression path. Their disagreement must not become a language rule
that constant zero divisors evaluate to zero.

The checker has named compile-time zero-divisor diagnostics, including `E056`
and `E057`; `tests/ui/type/modulo_by_zero_const_expr.sio` distinguishes folding
of the divisor from a literal-only check. Runtime integer division by zero and
signed minimum divided by minus one reach processor failure conditions in an
unguarded `idiv` path. The fixture's comment calling the latter undefined is not
a normative declaration of undefined behavior. Panic classification, signal
handling, and equality of constant and runtime behavior remain open.

### `NUM-008` — Cast acceptance is not a value-preservation guarantee

Standing: target-contract. `Lowerer.lower_cast_expr_ref` routes ordinary
integer-to-float and float-to-integer casts to `IrIntToFloat` and
`IrFloatToInt`. On the inspected x86 route the corresponding lowering uses
`CVTSI2SD` and `CVTTSD2SI`: the latter requests truncation toward zero for
representable results. This alone supplies neither unsigned-large-integer
conversion correctness nor a language contract for NaN, infinity, or
out-of-range floating inputs.

Many integer-to-integer scalar casts are identity operations in this lowerer.
The lean_single x86 postfix compiler has an explicit `as i8` sign-extension
instruction, demonstrating an engine-specific distinction. Wide casts in the
Madaros lowerer allocate limbs, copy the low word, and fill high words from a
sign word or zero according to the destination spelling. Do not generalize this
single-word construction into correct arbitrary wide-to-wide resizing. Cast
range checks, discarded bits, signedness changes, and wide narrowing need
independent rules and fixtures.

### `NUM-009` — Wide integers have real, bounded multi-limb lowering

Standing: target-contract. `self-hosted/ir/lower.sio::lower_wide_limb_count`
allocates whole 64-bit limbs. `Lowerer.lower_binary_expr_ref` selects wide
add/subtract/multiply, comparisons, full division/remainder, and a right-shift
path for a recognized nonnegative literal count. This is materially more than
the historic all-`i64` implementation.

`scripts/ci/madaros_wide_int_gate.sh` contains both source-level cases and
dedicated emitted-IR witnesses. Its carry-chain case checks a high limb rather
than just a low-limb result; multiplication, borrow, and full-divisor cases
similarly discriminate against truncated execution. These are different kinds
of coverage: an emitter witness is not proof that arbitrary source syntax
selects that emitter, and an identity function whose returned value is unused
does not validate high-limb transport across a call.

### `NUM-010` — Wide support excludes an established complete signed model

Standing: source-observed. Source wide division and remainder select
`ir_wide_div_full` and `ir_wide_mod_full` without a parallel signed magnitude
conversion in that branch. The native implementation contains unsigned wide
division routines. `tests/run-pass/wide_i128_signed_div_known_failure.sio`
explicitly carries a `known-failure` annotation for negative signed division.
It must not be counted as a passing signed-division contract.

The current source does carry signedness into ordered wide comparison at the
most significant limb, and `kl7_wide_sign_extension.sio` checks negative high
limbs and additive inverses. That progress does not establish signed wide
division or arithmetic wide right shift. The inspected native
`nc_wide_shr_into` zero-fills vacated high words and uses logical subword shifts;
it has no signedness parameter. This is a concrete signed-shift residual.
In particular, observing the top word
of a shifted negative value is weaker than testing sign fill throughout every
vacated high limb. Non-limb-aligned widths, final-limb masking, variable wide
shift counts, wide left shift, bitwise operations, aggregate layouts, and
high-limb function ABI coverage remain bounded or unestablished here.

`tests/run-pass/wide_i128_fn_abi_known_failure.sio` no longer has a
`known-failure` directive despite its filename. Its small-valued call checks
must be read as written; the filename is not current evidence of failure or
success. Wide printing remains explicitly refused during build by the KL-7
gate, with a narrow-print control and a no-output-ELF check.

## 5.3 Floating formats, literals, and special values

### `NUM-011` — `f32` has a declared native residual

Standing: source-observed. The checker distinguishes `f32` from `f64` and
allows fitting float literals to adopt `f32` in selected contextual positions.
`tests/compile-fail/f32_nonliteral_coerce_must_reject.sio` and
`f32_literal_magnitude_must_reject.sio` constrain that admission. On the
inspected ordinary native CPU path, however, `lower_cast_type_is_f64` includes
`f32` and a float-to-`f32` cast returns its source register.

`tests/run-pass/f32_cast_no_real_narrowing_residual.sio` deliberately pins this
residual with binary64 bit comparisons, including a value beyond binary32
range. It is a regression test for the admitted limitation, not conformance to
binary32 rounding. The title `f32` therefore cannot support a portable claim of
24-bit-significand arithmetic on this path. GPU or other backend binary32
instructions have a separate support profile.

### `NUM-012` — Binary64 literal conversion is path-specific

Standing: target-contract. The modular parser contains an integer-ratio
conversion path, `self-hosted/parser/parser.sio::pfl_slow`, and a direct
hexadecimal converter, `f64_hex_literal_from_source`. The hexadecimal path
uses the binary exponent and dropped-digit information rather than treating
the token as an ordinary decimal or a placeholder value.

`float_literal_correctly_rounded.sio` and
`float_literal_subnormal_half_ulp_2769.sio` cover decimal bit patterns. The
latter includes above/below-half cases near the least subnormal, minimum normal,
and decimal underflow. Its expected values have an existing C++23 exact-rational
oracle at `tests/oracles/float_literal_subnormal_half_ulp_2769.cpp`; that oracle
was inspected as a reference path, not run for this chapter.

`f128_v0e510_language_surface_closure.sio` additionally checks hexadecimal f64
ties, sticky bits, subnormals, exponent extremes, and separators. Its Madaros
requirement is significant: lean_single has its own decimal conversion path,
and the fixture notes a separate divergence. No cross-engine correctly-rounded
decimal parser claim is made here, nor is unlimited decimal input length
guaranteed by a fixed-capacity conversion buffer.

### `NUM-013` — Binary64 NaN and infinity have explicit emitter handling

Standing: target-contract. The native core emitter
`self-hosted/native/codegen_x86_linux.sio::nc_emit_core_float_binop_into` and
the separate `emit_float_binop_code` helper emit scalar double add, subtract,
multiply, and divide. Their comparison branches account for unordered
`ucomisd` results: equality is false for NaN, inequality true, and ordered
comparisons false. `tests/run-pass/ieee_f64_nan_compare.sio` checks both operand
positions where relevant. `scripts/ci/madaros_ieee_f64_special_values_gate.sh`
requests the corresponding NaN comparison and `inf`/`nan` printing witnesses on
both engines; it was not run here.

The newer `self-hosted/native/codegen_x86_linux.sio::emit_builtin_print_f64`
classifies nonfinite values from their exponent/fraction bits. The older helper
of the same name in `self-hosted/native/codegen.sio` has a different body. A
claim about the active runtime must identify the dispatch, not select whichever
helper is easiest to find. Signaling NaN traps, payload propagation, quieting,
and a total ordering over NaNs are not established by the comparison gate.

### `NUM-014` — Signed zero is observable through bits

Standing: target-contract. Unary floating negation flips the sign bit in the
inspected lowering. It is not specified by rewriting `-x` as `0.0 - x`.
`tests/run-pass/f64_unary_neg_signed_zero_bits.sio` distinguishes those
operations at positive zero and checks constant/local and function-parameter
paths, infinities, and double negation. Equal comparison of positive and
negative zero cannot witness this property; `f64_to_bits` can.

An arithmetic or optimizer transformation that preserves only numeric equality
may still change a subsequent reciprocal or bit observation. This fixture
provides specific negative controls inside a positive program. It is not full
coverage of zero signs for every instruction, optimization strategy, formatter,
or target.

### `NUM-015` — Rounding and floating environment have bounded status

Standing: source-observed. Ordinary scalar arithmetic uses the emitted hardware
double instructions; integer conversion uses a specifically truncating
instruction. The inspected arithmetic helper does not establish a language API
for changing or saving the floating environment, collecting IEEE exception
flags, distinguishing tininess detection policies, or controlling flush-to-zero
and denormals-are-zero across foreign calls and threads.

Software binary128/binary256 arithmetic instead explicitly implements a
nearest-even decision in `sf_rne_up`, using guard, round, sticky, and retained
low-bit information. A software rounding algorithm and hardware arithmetic
under an ambient control register are distinct contracts. No universal promise
of every rounding mode, every exception flag, or reproducibility under an
externally modified floating environment is made.

### `NUM-016` — Decimal display is not a bitwise witness

Standing: target-contract. The current x86 `emit_builtin_print_f64` formats a
fixed six-digit fractional part with a rounding/carry path and nonfinite fast
paths. It is not a shortest-roundtrip serialization contract, and ordinary
looking output can conceal different stored bits. Very large finite values
also need separate formatter coverage because the implementation converts an
integer part through a machine integer.

Binary128 has dedicated decimal and hexadecimal formatting in
`stdlib/math/softfloat_f128_fmt.sio`. Current KL-8 lowering routes language
`print`/`println` of supported f128 values to that surface; the older statement
that Madaros always refuses builtin f128 printing is superseded by
`tests/run-pass/f128_kl8_residuals.sio` and its gate. There is no corresponding
language-f256 decimal-print contract. Limb hexadecimal wire output is a
different operation from numerical decimal formatting.

### `NUM-017` — Binary128 and binary256 identities are explicit

Standing: library-contract. `self-hosted/check/numeric_format.sio` defines
fixed binary-format descriptors:

| Format | Storage bits | Exponent bits | Precision bits including leading bit | Trailing significand bits |
|---|---:|---:|---:|---:|
| binary128 | 128 | 15 | 113 | 112 |
| binary256 | 256 | 19 | 237 | 236 |

These numbers are transcribed descriptor fields, not calculations performed
for this chapter. `F128Bits { lo, hi }` and
`F256Bits { l0, l1, l2, l3 }` in `stdlib/math/wide_float.sio` carry low-to-high
machine-word bit payloads. A negative `i64` limb represents bits, not a negative
fraction field. The integer descriptors with IDs `1256` and `1512` are distinct
from binary256's format ID `256`.

Neither descriptor identity nor a wire payload proves arithmetic, ABI transport,
or the existence of a language operator. The descriptor must travel through the
appropriate checker, IR, and backend route.

### `NUM-018` — Madaros f128 literals require demonstrated exactness

Standing: target-contract. `self-hosted/parser/f128_literal.sio` parses f128
identity directly from source, carrying low/high limbs and an `exact` decision
through `Expr.f128_lit_lo`, `f128_lit_hi`, and `f128_lit_exact`. The lowerer
consumes those limbs rather than widening the previously parsed `f64` value.

The currently admitted subset comprises literals the bounded parser can prove
exact in binary128. `f128_v0e59_language_exact_literals.sio` includes `1e23`,
a decimal beyond binary64 integer precision, maximum finite and least-subnormal
hexadecimal cases. Its gate generates refusals for `0.1`, excess hex precision,
overflow, and lost subnormal bits. Thus an exact arithmetic division such as
`1.0 / 3.0` may produce a rounded result even though inexact input text is
refused: literal admission and operator rounding are different rules.

The decimal accumulator is bounded; a very long exact dyadic decimal need not
be admitted merely because its mathematical value fits binary128. The
lean_single route has a separate `f128_widen_refuses_inexact_literal` check and
widening-through-f64 limitation. No general exact decimal text-to-binary256
language-literal implementation is established here.

### `NUM-019` — Supported Madaros f128 expressions use softfloat payloads

Standing: target-contract. Current language f128 lowering desugars arithmetic
and comparison into `stdlib/math/softfloat_f128.sio` routines over `F128Bits`.
The inspected surface includes add/subtract/multiply/divide, sign negation,
comparison, and the provisional remainder discussed separately below. The
module frontend can inject required softfloat and formatter imports when f128
appears in the AST; V0-E.5.10 includes a no-explicit-import control.

The source fixtures separately exercise direct parameters/returns, struct
fields, methods, arrays, contextual literal arrays, and literal tails. These
are named surfaces, not a blanket ABI promise. Division implements explicit
zero and infinity classifications; `f128_v0e53_language_div.sio` checks
nontrivial quotient limbs, nonfinite results, and anti-f64 distinctions.
Comparison returns an unordered code for NaNs and treats the two zero signs as
equal. The implementation models no IEEE exception-flag result alongside the
returned bits.

### `NUM-020` — Wide-float refusal boundaries are part of the profile

Standing: target-contract. Madaros rejects f128/f64 value casts with `E248`
and mixed f128/f256 arithmetic with `E004`; neither implies a prohibition on
all f128 arithmetic. `tests/typekind/f128/refuse.sio` tests assigning a Boolean
to the numeric type, not the old reserved-name boundary.

The lean_single x86 path has separate binary128 widening/narrowing emitters and
libgcc arithmetic routing, while refusing several aggregate and indirect-call
surfaces. It does not inherit Madaros's array/field/method coverage, and Madaros
does not inherit lean_single's f64 casts. A declaration that is unused may
compile because no value lowering is requested. Tests must consume the value
through the operation and boundary being claimed.

Scalar wide-float support does not establish a precision-preserving profile for
`Knowledge<f128>`, `Knowledge<f256>`, wide GUM, or wide `variance_of`,
`uncertainty_of`, sensitivity, and covariance propagation. In
`self-hosted/ir/lower.sio::lower_type_expr_knowledge_value_float`, the
noninteger fallback intended for nominal brands returns an ordinary float
marker; it does not select a binary128/binary256 payload representation.
`ir_register_knowledge_layout` gives variance and confidence ordinary float
markers, and `Lowerer.lower_measure_call_ref` squares uncertainty through the
ordinary floating multiplication path. Nominal payload width, uncertainty
width, derivative width, and wrapper admission are separate obligations.

`tests/compile-fail/f128_nested_knowledge_reserved.sio` is explicitly ignored.
Its `Knowledge[f128] = 0` initializer confounds the old reserved-type diagnostic
with an ordinary type mismatch, and its header says E249 is superseded. It is
neither a current dedicated wide-epistemic refusal nor an execution witness.
These wrappers are excluded from the scalar support claims here; no uniform
fail-closed behavior across every nested epistemic shape is established.

### `NUM-021` — Binary256 limb add/subtract is a library surface

Standing: library-contract. `stdlib/math/softfloat_f256.sio` implements
`f256_bits_soft_add` and `f256_bits_soft_sub` over `F256Bits`, independently of
language-f256 value lowering. The source unpacks sign/exponent/fraction,
aligns significands with sticky bits, performs sign-sensitive addition or
subtraction, normalizes, and packs with nearest-even rounding.

The special-case branches distinguish NaN, infinity, zero operands, and
cancellation. Two negative-zero inputs retain a negative-zero sum; mixed-sign
zeros and exact cancellation produce positive zero in the explicit branches.
The NaN branch returns a fraction payload and clears the sign; this must not be
advertised as a complete signaling/quiet NaN payload policy. The simple
anti-f64 fixture `f256_v0f5_softfloat_add_sub.sio` checks high-precision recovery
and limb equality, but does not exhaust special values, subnormal boundaries,
ties, or all cancellation distances. There is no binary256 multiply/divide
implementation in this inspected module.

### `NUM-022` — Language f256 execution remains fail-closed

Standing: target-contract. The checker has a distinct `TyF256`; numeric
declarations and some operator shapes can be accepted. Executable language-f256
lowering without a supported payload is refused with the V0-E.4.1 sentinel.
That refusal prevents silent substitution of an f64 computation. The positive
limb API in the preceding rule does not discharge this language boundary.

`scripts/ci/madaros_f128_f256_v0e41_fail_closed_lower_gate.sh` classifies
several different outcomes. Its aggregate success does not establish both a
nonzero compiler exit and absence of an emitted artifact:

| Success branch, in evaluation order | What the branch establishes | What it does not establish |
|---|---|---|
| Sentinel and `! -s "$OUT"` | Matching sentinel and no nonempty output file | A nonzero compiler exit or literal file absence; a zero-byte file satisfies the size test |
| Nonzero `c_rc` and a broadly related diagnostic | Failed compiler status and a diagnostic matching the script's format/softfloat expression | Output absence or size; this branch precedes the emitted-output failure branch |
| Later stale E249 classification | A recognized stale frontend refusal classification | Current-source lowering behavior or the full exit/artifact conjunction |

The V0-F.5 gate also checks a language-f256 sentinel without enforcing the full
conjunction. A refusal-before-emission receipt must therefore independently
record a nonzero status, the intended current-source diagnostic, and output
absence under the declared artifact policy. A script's aggregate marker or a
zero-byte output alone is insufficient. `tests/typekind/f256/pass.sio` is a
typekind fixture whose name/annotation cannot override the executable lowering
boundary. The inspected lowerer refusal and the gates' ability to prove that
refusal are distinct evidence claims.

### `NUM-023` — Expansion arithmetic is not an IEEE wide format

Standing: library-contract. `stdlib/math/dd64.sio` represents a sum of two
binary64 components and provides error-free-transform kernels such as
`two_sum` and `two_prod` within their supported domain.
`stdlib/math/qd128.sio::Qd128` represents a sum of four binary64 components;
the name does not mean the IEEE binary128 encoding.

The qd module states a finite-domain contract, a target non-overlap invariant,
and a Priest-renormalization assumption. Those comments constrain the claim;
they are not a machine-checked proof of correct rounding for all operands.
Exceptional inputs, intermediate overflow/underflow, contraction of operations,
and violated non-overlap conditions require their own tests or model.
`tests/stdlib/math/test_dd64_deep_stdlib.sio` exercises EFT residual recovery
and division with bounded comparisons; the qd native import/multiply fixtures
exercise further named surfaces. The qd header's cited exact-EFT test path is
absent in this snapshot, so that comment is not counted as exactness coverage.
The existing fixtures are not certification of every expansion algorithm.

### `NUM-024` — Exact, tested, and proved are different predicates

Standing: specified-contract. An exact-literal decision certifies a parser's
admitted representation. A limb equality is an executable assertion. A runtime
certificate establishes its checked finite-domain predicate. A Lean theorem
establishes a proposition within its imports and assumptions after a verified
build. None is interchangeable with the others.

`formal/lean4/SounioIEEE754Spec.lean` contains explicit Float interpretation,
finite-domain, ordering, and rounding-bound axioms. Its existence does not
prove the native emitter implements those axioms. Similarly,
`scripts/ci/no_false_float_axioms.sh` screens selected false exact Float algebra
axioms; passing a textual screen would not prove the remaining model. No Lean
proof or numerical oracle was run in preparing this chapter. Legacy generator
oracles in Python are not an independent verification clock under ADR009.

### `NUM-025` — Current f128 remainder is provisional, not a general fmod rule

Standing: source-observed. `f128_bits_soft_rem` is labelled fmod-family in its
comment and is selected by the current `%` lowering. Its body obtains a rounded
quotient, repeatedly adds/subtracts one to build an integer accumulator, stops
after a guard bound of 17000 iterations, and returns a reconstructed difference.
That construction does not establish an exact quotient truncation algorithm
for arbitrary binary128 inputs or a bound by exponent gap as the comment
suggests.

Inspection exposes concrete boundary concerns without executing them: a finite
nonzero dividend with zero divisor takes an infinity-return branch; the loop
can reach its guard before a large quotient is exhausted; subtraction of one
need not change a sufficiently large floating quotient. These are implementation
limitations, not adopted numerical semantics. The KL-8 positive fixture covers
`8 % 3`, which cannot discriminate these cases. General remainder, exceptional
cases, sign of a zero remainder, and large-quotient accuracy need a corrected
algorithm plus positive and negative regression coverage before promotion.

### `NUM-026` — Support belongs to an engine, target, and surface

Standing: target-contract. The following is a source/fixture support map, not
an execution receipt:

| Surface | Madaros ordinary native CPU path, principally x86-64 | lean_single x86-64 seed path | Other targets |
|---|---|---|---|
| Scalar integers | Machine-word arithmetic with selected signed/unsigned distinctions; narrow-width residuals | Own compiler and cast rules; do not infer parity | Separate AArch64/RISC-V/Wasm/LLVM or GPU lowering requires its own witness |
| Wide integers | Real wide IR for a subset; signed divide and width/ABI gaps above | Does not inherit modular wide-IR witnesses | A target policy or encoder's existence is not coverage |
| `f32` | Ordinary scalar `as f32` retains f64 bits | Separate implementation; no binary32 claim established here | Explicit GPU f32 instructions require target/kernel validation |
| `f64` | Scalar double operations, parser and special-value fixtures | Own literal parser and emitter | Environment, subnormal, and formatter parity unestablished |
| `f128` | Softfloat payload surface and dedicated formatter; casts refused | Binary128 arithmetic/casts on a different restricted surface | No automatic proof of hardware/softfloat availability or ABI parity |
| Language `f256` | Fail-closed lowering | Refusal rather than an assumed f64 substitute | No executable profile inferred |
| `F256Bits` add/sub | Named library routines with anti-f64 fixtures | Gate has a separate seed path | Not implicit language-f256 support |
| Wide `Knowledge` and wide uncertainty/derivatives | No precision-preserving `Knowledge<f128>`/`Knowledge<f256>` or wide-GUM profile established; ordinary-float layout/shadow markers are not wide payload support | No wrapper or wide-uncertainty claim inherited from scalar f128 support | Construction, extraction, arithmetic, calls/returns, and uncertainty precision need separate witnesses |

The scalar rows exclude wide epistemic payload and uncertainty propagation.
The ignored, initializer-confounded nested-f128 fixture described in
`NUM-020` cannot certify either wide-wrapper acceptance or uniform refusal.

Compiler selection goes through `scripts/lib/resolve_souc.sh` and
`scripts/lib/resolve_madaros.sh`. Gates using `MADAROS_RAW_BIN` request a
particular modular artifact; seed-compiler branches and stale-wrapper allowances
must be reported separately. CPU success does not establish GPU behavior, and
GPU emission does not establish device execution. No default, rebuilt-wrapper,
or fallback execution path was exercised for this chapter.

### `NUM-027` — Open integer-language decisions

Standing: open-decision. The language still needs an approved answer for the
closed versus open integer-width namespace; whether width determines value
range, storage, arithmetic, or all three; signedness of pointer-sized aliases;
implicit narrow-family coercion; overflow policy; constant/runtime equivalence;
zero-divisor and signed-division-overflow outcomes; invalid shift counts;
wide cast resizing; and a high-limb ABI contract. These are not resolved by
selecting a historical implementation accident as normative behavior.

### `NUM-028` — Open floating-language decisions

Standing: open-decision. Required decisions include the binary32 CPU migration,
whether inexact wide input text should remain refused or acquire an explicitly
rounded syntax, wide conversions, signaling/quiet NaNs and payloads, signed-zero
observability across all optimizations, dynamic rounding modes and exception
flags, subnormal/flush policy, serialization, and target reproducibility. Each
decision needs a profile-specific conformance obligation. Neither a larger
mantissa nor a reference-library comment closes these decisions.

## 5.4 Units and dimensional checking

### `UNIT-001` — Units carry dimensions and scale separately

Standing: source-observed. `self-hosted/check/units.sio::UnitDim` contains seven
integer exponents ordered as mass, length, time, temperature, amount, current,
and luminous intensity, plus `scale_num` and `scale_den`. The source labels the
seventh entry "luminosity"; that implementation spelling is not the physical
definition of luminosity. `TypeEntry.unit_id` refers to
a registry entry; it is not the dimension vector itself. The sentinel for no
registered unit must not be interpreted as evidence that an unknown quantity
is physically dimensionless.

Builtin mass scale is expressed relative to grams in this table, whereas the
dimension vector records the dimension, not a globally normalized SI magnitude.
`unit_dim_compatible` compares exponents only. Scale-sensitive admission is a
different predicate. Multiplication and division combine the stored rational
scale fields without a general overflow-checked rational normalization layer.

### `UNIT-002` — Declarations, aliases, and shadowing

Standing: source-observed. A bare `unit name;` can introduce a name without
dimensional content. Redeclaring a builtin without a derived expression retains
its existing dimension through `unit_dim_for_decl`. Derived declarations are
decoded by `unit_dim_from_type_expr`; direct annotations additionally have
parser/checker support for products, quotients, powers, and parentheses, as
exercised by `unit_derived_power_and_parens_issue_2388.sio`.

The general declaration decoder has an explicit depth limit and
dimensionless fallbacks for unresolved names/shapes. These fallbacks are limits
of that route, not approval to silently erase unknown dimensions. Type
parameters must be resolved before builtin unit names when they shadow them;
the named-type lowerer has an explicit guard for this, and
`unit_builtin_global_and_generic_shadow_2874.sio` is the corresponding witness.
Unit aliases are not a demonstrated universal nominal newtype mechanism.

### `UNIT-003` — Unit literal and annotation surfaces are distinct

Standing: source-observed. Current fixtures cover suffix literals such as a
literal branded with a unit, direct named-unit annotations, derived annotations,
and the `f64` unit-expression surface. An accepted raw float initializer obtains
its brand from its annotation; no physical measurement uncertainty is created
by that operation.

`tests/frontend/unit_literal_suffix_current_source.sio` pairs with
`tests/compile-fail/unit_literal_suffix_reject_length_as_mass.sio`.
`unit_f64_unit_expr_velocity_current_source.sio` has both a wrong-dimension and
an unknown-unit negative. The current-source derived-unit gate also builds a
lean_single wrapper and runs AST/checker probes. That is a distinct route from
running only the default wrapper against an old binary.

### `UNIT-004` — Arithmetic composes dimensions

Standing: source-observed. `checker_finish_binary_units_inplace` implements
product dimensions by adding exponents and quotient dimensions by subtracting
them. A scalar multiplier preserves a unit; scalar divided by a unit obtains
the reciprocal dimension. Results are rebranded from the composed dimension,
rather than blindly inheriting the left operand's unit. Matching units cancel
to the unbranded scalar path.

The derived-result and quotient negative fixtures are essential: accepting
`distance / time` is not sufficient if its inferred binding later loses the
time exponent. `unit_quotient_keeps_dimension.sio` and
`unit_derived_let_dimension_mismatch_issue_2752.sio` require rejection of such
later misuse. This is a descriptive rule for implemented operators, not a
promise that arbitrary exponentiation, roots, transcendental functions, or
every user-defined operator has dimensional propagation.

### `UNIT-005` — Additive and comparison compatibility includes scale

Standing: source-observed. `checker_units_addable` requires equal exponent
vectors and equal stored scale numerator/denominator for add, subtract, and
ordered/equality comparisons handled by the in-place binary unit checker.
Different registry indexes may denote the same compatible dimension and scale;
indexes alone are not semantic inequality. Conversely, equal dimensions alone
do not make raw magnitudes in different scales addable.

`unit_same_add.sio` is a positive control and `unit_mismatch_add.sio` a
dimension mismatch. The energy conversion fixture additionally requests an
explicit boundary before adding differing scales, on the rebuilt/current-source
route selected by its gate. Representation-equality of rational scales does not
establish canonical equality of unreduced fractions. Affine temperatures and
logarithmic units cannot be derived from this multiplicative representation.

### `UNIT-006` — Units must survive the boundary being tested

Standing: source-observed. Unit checking is exercised at annotated bindings,
call arguments, and declared returns, including imported definitions. Positive
forwarding and returned-unit fixtures are paired with
`unit_lost_at_call_boundary.sio`, `unit_return_wrong_unit.sio`, and
`unit_call_result_carries_no_arg_brand.sio`. The result unit comes from the
callee's declared contract; it is not simply the caller argument's brand.

`tests/multimodule/unit_representation_2874/main.sio` and the parameter fixtures
distinguish representation compatibility from nominal-looking names. A module
that checks without any live call cannot establish parameter transport, and a
return value never consumed at a dimension-sensitive site cannot establish
preserved dimensional information. The finite checker table and actual target
lowering remain part of the support qualification.

### `UNIT-007` — An accepted unit cast does not prove scale conversion

Standing: source-observed. The compatible/incompatible cast fixtures establish
admission boundaries. In the inspected lean_single `compile_postfix_tail`,
same-dimension unit casts check dimensions and update type/unit metadata; that
branch emits no general scale-factor multiplication. The modular ordinary cast
helper likewise does not implement a general unit scale-ratio conversion.

`unit_energy_explicit_conversion.sio` checks acceptance and a marker but never
asserts the converted numerical value. `unit_cast_preserves_value.sio` checks
raw-number branding, not a change between two physical scales. Therefore this
chapter does not interpret `dose as kg` as proven milligram-to-kilogram
rescaling. Calling these fixtures conversion tests does not fill that gap.
Explicit numerical rescaling and subsequent branding need a stated API,
factor, and magnitude witness until conversion semantics are decided and wired
through the actual execution path.

### `UNIT-008` — Registry and decoder exhaustion are known safety gaps

Standing: source-observed. `UnitRegistry` has 32 entries.
`unit_registry_intern_dim` returns an unbranded sentinel when full; its comment
explicitly distinguishes this representation limit from dimensionlessness.
`unit_dim_from_type_expr` also returns dimensionless on certain unresolved or
depth-exhausted inputs. Neither is an approved weakening of dimensional safety.

The in-place rebranding helper clears the brand for a dimensionless exponent
vector, without a general scale normalization of the associated numeric value.
Consequently cancellation involving unequal scales needs a dedicated magnitude
audit. Additional missing coverage includes table exhaustion, scale-field
overflow, reducible scale fractions, deep aliases, and unknown aliases entering
the permissive declaration route. Positive low-cardinality tests cannot exclude
these failures.

### `UNIT-009` — Internal and domain unit labels carry no extra authority

Standing: library-contract. `register_builtin_units` provides common mass,
length, time, temperature, amount, volume, and selected internal labels. The
`mg_dL`, `mmol_L`, `U_L`, and `mm_h` entries are explicitly dimensional
validation-data labels; they are not a terminology authority or proof of
clinically correct conversions. The `mL`, `L`, and `cm3` entries have explicit
volume dimensions and scales.

Modules under `stdlib/units/` provide a larger vocabulary and helper functions.
An imported alias does not automatically extend every compiler builtin table,
nor prove that its physical constant, offset, or clinical interpretation has
been audited. `Knowledge` unit constraints and runtime guards add separate
epistemic boundaries; they do not repair an unchecked numeric scale conversion.

### `UNIT-010` — Open unit-language decisions

Standing: open-decision. The language needs an approved distinction among a
nominal brand, dimension, scale, and runtime quantity; the semantics of `as`
between different scales; whether to normalize rational scale metadata;
affine/logarithmic unit treatment; typed constants; dimensionless scale
cancellation; and mandatory refusal on exhausted or unresolved representations.
A robust promotion gate must check both the inferred dimensions and the actual
numeric magnitudes across locals, calls, returns, and imports.

## 5.5 Non-associative algebra and EISA

### `ALG-001` — Algebra identity includes product convention and order

Standing: specified-contract. As required by `FOUNDER_INTENT.md`, an algebraic
experiment may depend on order and parentheses. The associator is defined as
`(a * b) * c - a * (b * c)` with those parentheses preserved. Noncommutativity
and non-associativity are different predicates: changing operand order is not
the same operation as changing grouping.

An executable hypercomplex profile must identify its basis order, multiplication
convention, scalar carrier, and evaluation order. A real array can represent
components, but component storage alone does not define multiplication.
`hypercomplex_convention_crosscheck.sio` inlines a reference sign/product
implementation and checks selected convention invariants. It does not directly
import and compare every named library implementation, and was not executed here.

### `ALG-002` — Hyper type checking does not implement its entire algebra

Standing: source-observed. `self-hosted/check/hyper.sio::check_hyper_binary`
handles algebra tags, compatibility, a Cayley-Dickson embedding direction,
division admission, and `NonAssoc` effect checks. Its division-algebra predicate
admits the real/complex/quaternion/octonion tag range and rejects the sedenion
and generic Clifford range for `/`. The result scalar carrier is taken from
the left operand; a complete mixed-carrier promotion rule is not established
by this helper.

Its zero-divisor `Div`-effect section is explicitly unfinished in the inspected
body. It must not be described as an enforced rule solely because the module's
header lists an `E031` diagnostic. Signature-sensitive Clifford division
properties likewise cannot be reduced to the coarse tag predicate as an
approved mathematical classification. These are checker implementation
boundaries and candidates for refinement, not new language laws.

### `ALG-003` — Native Hyper construction is distinct from array library use

Standing: target-contract. Madaros `Lowerer.lower_cast_expr_ref` explicitly
refuses native CPU construction by `as Hyper<...>` and directs the user to
array-based library operations. `hyper_octonion_mul_cpu_refused.sio` pins that
refusal. Parsing and typechecking a Hyper spelling must not be reported as
native execution of its values.

The separate GPU lowering and packed native IR kernels have their own
representation/target assumptions. Existence of `IrHyperMulO`, a PTX golden
file, or AVX kernel code does not establish that ordinary CPU source casts reach
it correctly. `tests/madaros/algebra_octonion_import/oct_mul_import.sio` and
`scripts/madaros_algebra_octonion_import_gate.sh` instead exercise a named
array-library import on the default engine. That bounded route is the relevant
positive counterpart to the native Hyper refusal.

### `ALG-004` — Declared laws constrain reassociation; they are not proofs

Standing: source-observed. `checker_collect_algebra_def_inplace` registers
declared laws and reassociation strategy. A declared unrestricted strategy on a
non-associative product is rejected with `E252`. An omitted clause derives a
strategy from the recorded laws instead of automatically granting unrestricted
reassociation; an alternative octonion declaration selects the Fano strategy.
`algebra_reassociate_omitted_fail_closed.sio` and
`e252_reassociate_free_on_alternative.sio` distinguish omission from an invalid
explicit claim.

The registry still uses name-derived tags and finite capacity; it does not
prove that arbitrary user implementations satisfy their declared laws.
`self-hosted/ir/algebra.sio::ir_should_reassociate` and
`alg_ir_can_reassociate_triple` encode basis-level decisions. Those decisions
need the relevant basis knowledge at the actual optimization site. They do not
license arbitrary floating-point reassociation merely because the corresponding
exact algebra is associative. `StrategyAggressive` is a separate scalar-FP
relaxation, not a theorem about an octonion product.

### `ALG-005` — Named hypercomplex libraries have distinct APIs

Standing: library-contract. `stdlib/algebra/octonion.sio` exposes a component
array product and Cayley-Dickson sign routines, with the witness convention
`e1 * e2 = e3`. `stdlib/algebra/sedenion.sio::sed_mul` and
`sed_associator` use 16 components and an explicit product sequence.
`stdlib/math/octonion.sio` and `stdlib/math/sedenion.sio` instead expose struct
APIs, including named inverse helpers. Similar mathematical names do not prove
identity of convention, aliasing behavior, or optimizer treatment.

The array sedenion implementation uses shared scratch state; that is a concrete
reentrancy/concurrency qualification. The floating inverse and zero-divisor
helpers in the math module must not be promoted to exact algebraic decision
procedures. `tests/stdlib/algebra/test_octonion_deep_stdlib.sio` and
`test_sedenion_deep_stdlib.sio` are broader fixtures, while the convention
crosscheck and ordered sparse witness below provide more discriminating
boundaries. Neither proves all real-coefficient identities in floating arithmetic.

### `ALG-006` — Exact sparse zero-divisor tokens have a checked finite domain

Standing: library-contract. `stdlib/eisa/hypercomplex_zd.sio` defines
`SedExact16V0`, `ZDExactAttemptV0`, and `ZDExactTokenV0`. Admission validates
positive operand identities, at most two nonzero coefficients per operand,
coefficients in the admitted signed-unit set, nonzero operands, and a zero
product under `cd_sigma_exact_i64` at tower level four. Output index is basis
XOR and the sign is convention-specific. Overflow-risk and unsupported inputs
have distinct statuses.

`element_status` is advisory: validation rechecks the coefficient arrays because
external struct construction is possible. A token binds ordered operands,
identities, convention, domain, detector, and orientation; token consumption
rechecks these and recomputes the product. An integrity tag alone is not an
authority. The API's `proved` name denotes this bounded exact integer predicate,
not a Lean theorem or a general proof about every sedenion.

`test_eisa_h_zd.sio` has sign, identity, orientation, zero-operand, unsupported
coefficient, forged-large-input, and forged-token negatives alongside a positive
pair. Its gate generates a type-negative consumer test. That gate can report a
blocked default Madaros runtime while executing the witness on lean_single; an
overall script marker must not be read as unconditional Madaros execution.

### `ALG-007` — A measured near-zero receipt is never an exact token

Standing: library-contract. `zd_measured_norm_f64_v0` receives norms and a
resolution; it does not recompute a hypercomplex product from components.
It validates identity, finiteness, nonnegative norms, positive resolution, and
nonzero operand norms. A product squared norm strictly below the supplied
resolution produces a measured-near-zero receipt. Equality with the threshold
does not satisfy that predicate. The meaning of the resolution must match the
squared-norm quantity supplied.

`zd_measurement_is_exact_proof` returns false. The measured receipt type differs
from the exact token type and the gate tests rejection at that boundary. A
caller-supplied zero norm or tiny floating norm is thus not silently elevated
to exact annihilation, and this inert algebraic receipt establishes no clinical
or causal interpretation.

### `ALG-008` — EISA keeps value, correction, and uncertainty lanes distinct

Standing: library-contract. `stdlib/eisa/core.sio::EReg` contains `val: f64`,
`err: Dd64`, `u: f64`, and an origin identifier. `ereg_true` reconstructs a
double-double value from the nominal value plus correction; its name does not
identify an independently measured physical truth. `ereg_exact` initializes
zero correction and zero uncertainty for its already rounded f64 argument.
It does not retroactively make an inexact decimal real constant exact.

The `u` lane holds an uncertainty magnitude, whereas first-class epistemic
compiler paths may carry variance or richer noise metadata. Squaring or
unsquaring must be explicit at such an interface. EISA code, the standalone
epistemic library, and compiler `Knowledge` lowering are distinct surfaces;
common terminology does not imply automatic interoperability.

The reference instruction machine in `stdlib/eisa/isa.sio` is another distinct
surface. `EMachine` has 16 register slots and eight memory slots holding value,
two correction components, and uncertainty, plus register poison state.
`eisa_step` implements constant/load, add/subtract/multiply/divide/sqrt, gate,
and store instructions with explicit index checks. It rejects negative loaded
uncertainty, poisons nonfinite arithmetic inputs/results and division by a zero
nominal denominator, and makes a poisoned gate reject. A poisoned operation can
return instruction status zero while carrying poison forward: instruction
completion is not numerical admission. These interpreter rules must not be
confused with bare `EReg` calls, which do not supply the same validation layer.

### `ALG-009` — EISA v0 propagation and origin fusion are bounded models

Standing: library-contract. `eadd`/`esub` use error-free sum residuals plus
incoming corrections. Different or unknown origins take independent quadrature
in `eisa_u_add_sub`; equal nonzero origins take a conservative linear envelope.
In particular `esub` of the same origin does not declare zero uncertainty by
cancellation. `ereg_measured` uses unknown origin zero; `ereg_origin` allows a
caller-supplied identity, which is not externally authenticated covariance.

Multiplication/division propagate first-order sensitivities using nominal
values, and their correlation refinements remain explicitly deferred in source.
The v0 correction expressions omit some higher-order correction cross terms.
Origin fusion survives as the common nonzero origin only when the operands
share it; mixed/unknown origins produce zero. Zero is an unknown sentinel, not
an independent-source proof. `eisa_origin_correlated_add.sio`,
`eisa_origin_independent_add.sio`, `eisa_origin_esub_same.sio`, and the mixed
fusion fixture discriminate these branches. They do not establish a general
covariance algebra for arbitrary dependency graphs.

The reference ISA register/memory representation has no origin lane.
`isa.sio::reg_to_ereg` reconstructs origin zero. Consequently repeated use of
one measured ISA register does not automatically inherit the direct core
same-origin envelope. Unknown origin is missing dependency information, not
proof of independence; the independent formula can understate a repeated-source
sum. This is a concrete interoperability residual rather than a conservative
covariance guarantee for the whole machine.

### `ALG-010` — EISA v2 uses a finite expansion model, not binary128 truth

Standing: library-contract. `stdlib/eisa/core_v2.sio::EReg2` replaces the
correction with `Qd128`; it has no origin field. For an operation, the code
computes a nominal f64 result, reconstructs each input as a qd expansion, applies
the qd operation, and stores the difference from the nominal result as the new
correction. Unlike v0 multiplication, qd multiplication includes products of
incoming correction terms. Its uncertainty formulas remain first-order and do
not acquire v0's origin fusion merely by sharing the EISA name.

The intended reconstruction equation is a model contract conditional on the qd
algorithms, their finite domain, and renormalization assumptions. The module's
comments asserting qd correct rounding are not an independently verified proof
of that equation for every input. `core_v2_sqrt_f64` uses a fixed-iteration
Newton routine; `esqrt2` returns an exact-labelled zero when the nominal value
is nonpositive. No general IEEE square-root exceptional-value semantics should
be inferred from that branch. `test_eisa_evm_v2.sio` compares EVM and reference
chains, which can share defects; it is not a universal independent oracle.

`stdlib/eisa/format.sio` gives the serialized EISAX image an explicit version;
`stdlib/eisa/evm.sio::EvmMachine` implements the loaded image. Version zero uses
the smaller reference capacities; versions one and two use 64 registers,
32 memory slots, bounded code/constant tables, explicit control flow, and fuel.
Version two shares version one's structural/opcode rules while selecting four
correction components and `core_v2` arithmetic. A valid image hash establishes
the image's checked identity, not its numerical correctness or a cryptographic
provenance guarantee. Unsupported versions, malformed images, poison, and fuel
exhaustion are distinct from an arithmetic gate's okay/marginal/reject result.
The EVM normalizes negative-zero uncertainty on version-one/two writes; that
does not erase the sign of the value lane by definition. Format, interpreter,
backend compiler, and native bridge tests are separate executable boundaries,
not evidence of an installed physical EISA processor.

### `ALG-011` — EISA gates and zero flags report specific observations

Standing: library-contract. `egate` and `egate2` compare different scalar
approximations to correction magnitude against `u`. The v0 gate obtains
`dd_to_f64(dd_abs(x.err))`, a binary64 collapse of its double-double
correction. The v2 gate obtains `core_v2_abs_f64(x.err.x0)`, inspecting only
the absolute leading qd component. It does not sum, bound, or validate the
trailing components `x1`, `x2`, and `x3`.

For the scalar selected by the relevant gate, multiplying by 10 and reaching
the budget returns reject; otherwise multiplying by 100 and reaching the budget
returns marginal. A nonpositive budget returns okay when that selected scalar
compares equal to zero, not when the implementation has established that every
correction component is zero. The parameter `k` is retained but does not select
a new threshold in these implementations.

Interpreting the leading component as the correction magnitude depends on the
qd normalization/non-overlap model, which this gate does not enforce on input.
Even under that model, an exact full-expansion classification at the threshold
is not established by dropping trailing components. A noncanonical externally
assembled correction with zero `x0` and nonzero trailing components is invisible
to `egate2`'s correction read but visible to the zero flags below. This is a
source predicate distinction, not an executed counterexample or a claim that
such an input satisfies the canonical expansion model. Admission of these
inputs and canonical whole-expansion threshold coverage remain unresolved.
NaN inputs and invalid uncertainty magnitudes also need separate validation;
ordinary comparisons alone do not make these gates fail-closed for every
nonfinite value.

`core_v2_zero_flags_raw` reports whether the nominal value compares equal to
zero and whether any correction component compares nonzero. It does not
distinguish signed zero or infer annihilation, physical absence, censoring, or
underflow cause. `zero_provenance_native_v2_combined.sio` deliberately exhibits
distinct zero provenances using an application-level witness. Its current gate
requests a combined default-engine run; an older known-failure probe or smoke
comment saying the combined import is blocked is not the newer gate's contract.

### `ALG-012` — Algebra, differentiation, and formal-model claims are scoped

Standing: specified-contract. Reassociation can change both a non-associative
value and its derivative structure. The lean_single negative
`octonion_hessian_without_reassoc.sio` and positive
`octonion_hessian_fano_annotated.sio` address a specific AD shadow/gate route.
They are not proof that every compiler path implements the same Hessian policy.

Formal Cayley-Dickson definitions, model theorems, and algebraic basis checks
do not establish all floating execution paths or a physical interpretation of
octonion/sedenion models. In particular, algebraic non-associativity must not be
confused with scalar floating rounding non-associativity or with clinical order
effects. Claims must identify the carrier, convention, admitted domain, build,
and observed witness; no theorem is reported as built for this chapter.

### `ALG-013` — Open algebra and EISA decisions

Standing: open-decision. Remaining decisions include law declaration trust and
proof obligations; scalar floating versus exact-algebra optimization rules;
native Hyper representation; basis-convention interoperability; concurrent
library scratch state; signed/exceptional inverse behavior; and the relation
between EISA, `Knowledge`, variances, origins, and covariance.

EISA additionally needs a declared admitted domain for error-free transforms,
overflow/subnormal behavior, rigorous qd error bounds versus correct-rounding
claims, nonfinite rejection, uncertainty-domain validation, sqrt boundary
behavior, and the meaning of unused gate parameters. A measured zero, a small
rounding correction, and an exact sparse annihilation token remain different
results until an explicit conversion contract says otherwise.

## 5.6 Evidence map and verification obligations

The companion TSV has one row per defined rule, one principal source and
symbol, and at most one existing path in each witness cell. `-` explicitly
marks missing evidence. Some positive programs contain negative controls; some
gates generate negative fixtures dynamically. The fuller map below records
those distinctions without claiming any case passed during this research.

| Area | Implementation evidence | Positive witness or gate surface | Negative or boundary evidence | Qualification |
|---|---|---|---|---|
| Integer identity and wide operations | `self-hosted/check/types.sio::ty_wide_int`; `self-hosted/ir/lower.sio::Lowerer.lower_binary_expr_ref` | `scripts/ci/madaros_wide_int_gate.sh` source cases and emitted high-limb witnesses | Generated width/signedness mismatch cases; `tests/compile-fail/kl7_wide_int_print.sio` | Gate includes distinct source and IR-only tests; all unrun here. |
| Signed versus unsigned words | `self-hosted/native/lower_ir.sio::emit_binop_code` | `tests/run-pass/u64_unsigned_param_shr_cmp_div.sio` | `tests/run-pass/i64_signed_shr_divmod_witness.sio` signed controls | Does not establish signed wide division. |
| Literal/constant refusal | `self-hosted/ir/optimize.sio::try_fold_binop`; checker zero-divisor diagnostics | Integer literals in `tests/run-pass/f128_v0e510_language_surface_closure.sio` | `tests/ui/type/modulo_by_zero_const_expr.sio`, `tests/ui/type/shift_float.sio` | Separate evaluator sentinels are not a language contract. |
| Binary32 residual | `self-hosted/ir/lower.sio::Lowerer.lower_cast_expr_ref` | `tests/run-pass/f32_cast_no_real_narrowing_residual.sio` | `tests/compile-fail/f32_literal_magnitude_must_reject.sio`, `tests/compile-fail/f32_nonliteral_coerce_must_reject.sio` | Positive fixture pins a limitation rather than binary32 conformance. |
| Binary64 input rounding | `self-hosted/parser/parser.sio::pfl_slow`, `f64_hex_literal_from_source` | `tests/run-pass/float_literal_correctly_rounded.sio`, `tests/run-pass/float_literal_subnormal_half_ulp_2769.sio` | Below-half/underflow assertions in the same fixture | Existing C++23 rational oracle is unrun; do not infer lean_single parity. |
| IEEE special observations | `self-hosted/native/lower_ir.sio::emit_float_binop_code`; newer x86 print emitter | `scripts/ci/madaros_ieee_f64_special_values_gate.sh` | NaN false-comparison checks; signed-zero versus subtraction control in `tests/run-pass/f64_unary_neg_signed_zero_bits.sio` | No signaling-NaN/flags/environment audit. |
| Exact f128 literals | `self-hosted/parser/f128_literal.sio::f128_literal_from_source` | `tests/run-pass/f128_v0e59_language_exact_literals.sio` | Generated `lit_tenth`, overprecision, overflow, and subnormal-loss cases in `scripts/ci/madaros_f128_f256_v0e59_exact_literals_gate.sh` | Bounded exact-admission parser, not arbitrary rounded decimal input. |
| f128 language surface | `self-hosted/ir/lower.sio` f128 desugaring; `stdlib/math/softfloat_f128.sio` | V0-E.5.3 through V0-E.5.10 fixtures; `tests/run-pass/f128_kl8_residuals.sio` | `tests/compile-fail/f128_cast_to_f64_unimplemented.sio`; mixed-format rejection | `%` wiring and small positive example do not close the remainder defect. |
| Binary256 bits versus language type | `stdlib/math/softfloat_f256.sio::f256_bits_soft_add` | `tests/run-pass/f256_v0f5_softfloat_add_sub.sio` | Generated language-f256 compile in `scripts/ci/madaros_f128_f256_v0e41_fail_closed_lower_gate.sh` | Aggregate gate success does not enforce nonzero status plus no output on every branch; special-value coverage remains incomplete. |
| Wide epistemic payload and uncertainty | `self-hosted/ir/lower.sio::lower_type_expr_knowledge_value_float`, `ir_register_knowledge_layout`, `Lowerer.lower_measure_call_ref` | Scalar f128 fixtures are adjacent evidence only; no wide-epistemic execution witness established here | `tests/compile-fail/f128_nested_knowledge_reserved.sio` is ignored and initializer-confounded | Neither a precision-preserving wide-GUM profile nor uniform nested-wrapper refusal is established. |
| Dimensional propagation | `self-hosted/check/check.sio::checker_finish_binary_units_inplace` | `tests/run-pass/unit_derived_power_and_parens_issue_2388.sio`, `tests/run-pass/unit_volume_mL_L_issue_2388.sio` | `tests/compile-fail/unit_quotient_keeps_dimension.sio`, `tests/compile-fail/unit_return_wrong_unit.sio` | Registry exhaustion and magnitude conversion remain unverified. |
| Current-source unit route | `self-hosted/check/units.sio::UnitDim`; derived AST/checker probes | `scripts/ci/unit_types_derived_gate.sh` | Generated/routed unknown unit, wrong dimension, and energy-scale refusals | Gate builds a wrapper; default acceptance is not that rebuilt execution result. |
| Law-directed order | `self-hosted/check/check.sio::checker_collect_algebra_def_inplace`; `self-hosted/ir/algebra.sio::ir_should_reassociate` | `tests/run-pass/algebra_reassociate_omitted_fail_closed.sio` | `tests/compile-fail/e252_reassociate_free_on_alternative.sio` | Declaration checking does not prove implementations satisfy their laws. |
| Hyper CPU/library distinction | `self-hosted/ir/lower.sio::Lowerer.lower_cast_expr_ref` | `tests/madaros/algebra_octonion_import/oct_mul_import.sio` | `tests/compile-fail/hyper_octonion_mul_cpu_refused.sio` | Array-library success would not validate native Hyper construction. |
| Exact versus measured annihilation | `stdlib/eisa/hypercomplex_zd.sio::zd_exact_attempt_v0`, `zd_measured_norm_f64_v0` | `tests/stdlib/eisa/test_eisa_h_zd.sio` | Forgery/threshold/nonfinite cases in that test; generated exact-consumer type rejection in `scripts/ci/eisa_h_zd_reference_gate.sh` | Gate explicitly distinguishes lean_single execution and a possibly blocked default runtime. |
| EISA propagation and observation | `stdlib/eisa/core.sio::eadd`; `stdlib/eisa/core_v2.sio::eadd2` | `tests/stdlib/eisa/test_eisa_core.sio`, `tests/stdlib/eisa/test_eisa_evm_v2.sio` | Origin subtraction and mixed-origin controls; zero-provenance distinctions | Differential reference chains are not an independent complete numerical proof. |
| EISA instruction and image profiles | `stdlib/eisa/isa.sio::eisa_step`; `stdlib/eisa/format.sio::eisax_validate_v1v2`; `stdlib/eisa/evm.sio::EvmMachine` | `tests/stdlib/eisa/test_eisa_isa.sio`, `tests/stdlib/eisa/test_eisax_v1_format.sio` | Malformed image, poison and fuel controls in `tests/stdlib/eisa/test_eisa_evm_v1.sio` and the v2 EVM test | All unrun; image validation, interpreter completion, and numerical admission are different claims. |

For subsequent executable validation, a receipt must record the source SHA,
resolved compiler artifact and identity, engine, target, command, input fixture,
exit status, and relevant stdout/stderr or bit payload. A refusal receipt must
distinguish a compiler diagnostic from a crash, timeout, backend failure, or
missing tool, and check artifact absence where claimed. Use the canonical suite
entrypoint `scripts/run_sio_test_suite.sh` and the named focused gates with
appropriately identified artifacts; this chapter adds no alternative runner.

The most discriminating missing numerical witnesses are out-of-range scalar
widths, signed wide divide/rem and shift sign fill, high-limb call/return and
nonstandard widths, runtime/constant overflow parity, large-quotient f128
remainder, all softfloat special/rounding boundaries, unit scale magnitude
conversion and exhaustion, and EISA nonfinite/domain handling. The unresolved
language decisions above remain explicit even if a present implementation
produces a plausible number for one such case.

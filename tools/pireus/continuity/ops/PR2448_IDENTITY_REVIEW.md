# PR2448 resolved-identity review

Reviewed source: bacbf8ab4af48618e0bd8a787f63ae3d18aa2fb1.
This is a static review; no current-source Madaros collision witness was executed.
It does not change compiler source or the separately owned f128 lane.

## Finding: lexical interception remains ahead of ordinary call lowering

self-hosted/ir/lower.sio:5521 reduces ExprIdent to its name, ExprPath to its last
segment, and field/method expressions to their names. At20251 the generic call
lowerer compares that result with f128_from_limbs, f128_to_lo and f128_to_hi,
then immediately invokes lower_f128_intrinsic_call_ref. That helper at19810
again dispatches by the short name. These branches do not check a resolved symbol
identity or defining-module identity.

stdlib/math/softfloat_f128.sio:417-429 defines ordinary public placeholder bodies:
f128_from_limbs yields zero and both extraction functions return zero. A missed
interception therefore has a valid machine result with incorrect intended meaning.
The latest commit fixes linkage of the softfloat implementation in the smoke;
it does not replace this dispatch mechanism.

Risk requiring a witness: a user-defined homonym or a distinct module's homonym
can enter the special lowering path instead of ordinary call lowering. Conversely,
import renaming can lose interception. These are source-supported risks, not a
claim that all such forms pass earlier phases or that a runnable exploit was tested.

## Required correction and evidence

Resolve the canonical builtin identity before lowering and carry that identity
through aliases and path qualification. A short-name comparison must not confer
intrinsic authority. Every call classifier using lower_f128_expr_callee_short
must be audited, including result-type classification and f128 value detection;
fixing only the final dispatch leaves inconsistent classifications possible.

Require current-source executable controls for:
- A local function named f128_to_lo with ordinary i64 input/output and observable
  nonzero behavior, exercising the ordinary call.
- A foreign-module homonym and a same-spelling method/field-call where supported.
- A canonical import alias and a fully qualified canonical call, both preserving
  identity rather than acquiring it from spelling.
- Unsupported wide operations: exact refusal sentinel, nonzero compiler exit,
  and no emitted artifact, checked together.

If the language does not support one alias form, retain the parser/checker refusal
as that narrower result; do not call it an intrinsic-resolution success. Eliminate
executable zero placeholders through explicit intrinsic declarations or a hard
failure on unsupported compilation paths.

## CI checkpoint and recommendation

At the reviewed head, the GitHub check-runs query reported Contracts, Full Test
Suite, Gate Wave0, Madaros Witness Gate, source bootstrap and Linux/macOS self-host
successful. Madaros Current-Source f64 Lowering was still in progress. This
supersedes the alert's older Contracts-red state; it is not a composite final
verdict or evidence of resolved intrinsic identity.

Keep review changes outstanding until the identity controls execute with a
compiler built from the reviewed source. No merge or source/ABI promotion follows
from this note.

<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-178-product.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-178-product.readme
-->

# SIO-178: unbounded rational multiplication and division

This separate increment extends the exact rational interval-set model with
four-corner multiplication, sign/pole reciprocal hulls, division, decoration
constraints and conditional outward wrappers. It preserves the finite32 and
additive40 source models byte-for-byte. There is no compiler, IEEE bit-level,
signed-zero encoding or hardware/backend qualification claim.

Dependency: [DraftPR3158](https://github.com/Sounio-lang/sounio/pull/3158),
HEAD e54b648fde22e27c98ee41314bb9d594026f2d35, based on remote main8eca.
New source: formal/lean4/SounioIntervalExtendedMulDiv.lean.
The algorithm below was approved by the coordinator before implementation.

## Actual set algorithms

ExtendedRat endpoint order includes negative infinity, finite Rat and positive
infinity. Multiplication handles a finite zero with an infinite endpoint as a
set-bound special case before selecting any infinite sign. It then computes
the minimum and maximum of the four actual endpoint products. There is no
fallback that returns Entire merely because an operand is unbounded.

The conversion from computed endpoints normalizes both infinite singleton pairs
to Empty, because infinity is never a member of the represented rational set.
Only proper spans or explicit Empty are valid interval objects. Runtime invalid
input refusal is a separate requirement, not a promise made by this model's
internal computed-bound set constructors.

Reciprocal cases use the original finite model's inverse-order inequalities:

| Input denominator set | Reciprocal closed hull |
|---|---|
| Strictly positive or strictly negative finite interval | Reversed reciprocal endpoints |
| Positive interval unbounded above | Zero lower closure bound, reciprocal finite lower endpoint above |
| Negative interval unbounded below | Reciprocal finite upper endpoint below, zero upper closure bound |
| Zero at the lower endpoint with positive upper endpoint | Positive reciprocal lower bound, unbounded above |
| Zero at the upper endpoint with negative lower endpoint | Unbounded below, negative reciprocal upper bound |
| Strictly crosses zero, or Entire | Entire hull |
| Zero-point set | Empty |

Division calls multiplication with that reciprocal hull. Its enclosure theorem
requires an actually contained nonzero denominator member. Zero-point division
returns Empty; for non-ill input decorations its decoration is exactly trv.
Any denominator containing zero caps the result decoration at trv. Both
operations also prove their output decoration cannot strengthen either input.

Zero times Entire is exactly the zero-point set, not Entire; this has a general
membership equivalence proof. Zero divided by the concrete crossing interval
[-2,3] has the same tight zero-point membership. Other endpoint witnesses check
positive/negative unbounded products, infinite singletons and reciprocal touching
and crossing cases. These do not claim a general minimality/tightness theorem.

Generic finite multiplication, reciprocal and separated-zero division value
concordance is stated against the canonical imported finite engine, without
copying its engine. Conditional outward multiplication/division reuse the
unchanged Rounding primitive premises and outward_contains theorem. Those wrappers
are outward exact hull enclosures, not a theorem about emitted opcode shape;
the IEEE/hardware binding remains explicitly NOT ESTABLISHED.

## Actual verification

- Slurm17549 preserved a development proof failure: simp_all cleared True-bound
  hypotheses in the endpoint-conversion helper. Explicit cases corrected the
  proof without changing the algorithm.
- Slurm17554 COMPLETED0:0 in12s: all41 then-current theorems were kernel checked,
  including generic all-bound multiplication, reciprocal and division enclosure.
  It used the hash-verified compiled finite/additive dependencies.
- The53-theorem extension adds exact generic finite concordance, conditional
  outward wrappers, division non-strengthening and exact zero-divisor decoration.
  Slurm17555 exposed min/max rfl-after-split and premature zeroPoint unfolding in
  those new proof scripts. Proofs were corrected; the algorithms were unchanged.
- Slurm17573 COMPLETED0:0 in6m09s from closed source05 at candidate
  dd9821ac6d036c926e4de9f41a61c03f2c98868e. Fresh six-target build/source/audits
  passed: original finite32, original additive40 and all53 new theorems. The53
  fresh axiom rows use only propext, Classical.choice and Quot.sound. The11
  new guard controls and12 actual semantic mutants passed; each mutant exited1
  with a mathematical error inside the specified theorem range. Source scanner
  found zero axioms, native_decide uses or sorry holes in the new model.
- Final source, Lean/lake binaries, complete toolchain libraries and Node identity
  checks passed. Outer closed16-file source verification passed before and after
  the gate. The submission requested one CPU and4GiB with15min limit; Slurm
  allocated2CPUs due to scheduler allocation granularity. This is a proof job,
  with no numerical calibration campaign.

The first attempted final04 staging failed before any Slurm job submission:
source-directory creation was delayed, the tar extraction reported missing
source04, and sbatch reported Unable to open file. Worker exec/runtime acquisition
latency was diagnosed read-only by the coordinator's designated lane.
No host process, service or quota was changed by this lane. Primary transport
recovered before the successful final05 staging; all dependent steps then reached
terminal success serially and the complete source manifest verified before the
single final submission. There was no duplicate submission or claim that the
preparation failure measured mathematical behavior.

## Qualification contract

Use pinned Lean4.33.0 commitd8b18978322de05a8f3dba51ef03cf5461676c17 in allocated
Slurm UID1000 work. Fresh audits must cover all53 new theorem names with only
propext/Classical.choice/Quot.sound, the unchanged32+40 dependency audits, no
source proof holes, and actual semantic mutations rejected inside the targeted
mathematical theorem ranges. Compiler/import/parser failures cannot satisfy them.

The twelve mutations cover zero times infinity, omitted lower/upper corners,
positive/negative reciprocal touching sides, the zero-point pole, division
zero-decoration policy, both infinite singleton pairs, negative-infinite product
sign, the infinite reciprocal closure endpoint and finite reciprocal ordering.

## Custody and limits

The compact text packet final05-text.tar.gz contains261 internally hashed input
and receipt files: closed source05, preserved source01/02/03 development failures
and successful probe, job scripts and terminal accounting, complete fresh audits,
all semantic mutants/diagnostics and initial/final identity receipts. It includes
no compiled proof artifacts or toolchain binaries. Packet size893613bytes;
SHA256 f20f2a54ec15c08f8b0071609af8d03dee29e2a516cb011deccfedea25688700.
Fresh extraction verified all261 internal hashes and equality of six models/audits
and two new helpers with the working tree. The independent inspection counted53
allowed axiom rows and confirmed all12 exact mathematical target errors.

FILES.sha256 binds the final reviewed source and receipt files. The older
CANDIDATE_FILES.sha256 records the pre-run checkpoint and its pending README; its
README line is historical and is not a verification manifest for this closure.
Global source inventory changed only320→322 Lean files versus the additive
increment, retaining59 declared axioms,741 historical native_decide occurrences
and zero sorry holes. This mechanical global census is separate from the fresh
53-theorem kernel audit. No workflow or compiler source changed.

The finite-codomain Rounding premises are assumed contracts, not established
IEEE primitive implementations; in particular a future hardware binding must
cover finite overflow to infinity. This increment proves exact rational set
operations and their conditional outward enclosure, with general tightness,
primitive bits, compiler IR/opcodes and hardware qualification still open.

Tracking: [SIO-277 MulDiv followup](https://app.notion.com/p/4043f37f937181ecb782f7e9e105fb1e)
under [SIO-178](https://app.notion.com/p/4033f37f93718110967acf47f65cbe51).
This card and the parent programme remain incomplete until their actual gates
and human review criteria hold. No main merge, auto-merge, enqueue, quota change
or historical artifact deletion is included.

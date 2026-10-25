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

## Actual verification and pending gates

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
- Final validation is pending a closed fresh six-target Lake/source/axiom gate,
  full source/toolchain identity and twelve actual mathematical source mutants.
  No successful53-theorem or final expanded-gate result is claimed yet.

The first attempted final04 staging failed before any Slurm job submission:
source-directory creation was delayed, the tar extraction reported missing
source04, and sbatch reported Unable to open file. Worker exec/runtime acquisition
latency is under read-only diagnosis by the coordinator's designated lane.
There is no duplicate submission or claim that the preparation failure measures
mathematical behavior. Future staging must finish and verify before submission.

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

Tracking: [MulDiv followup](https://app.notion.com/p/4043f37f937181ecb782f7e9e105fb1e)
under [SIO-178](https://app.notion.com/p/4033f37f93718110967acf47f65cbe51).
This card and the parent programme remain incomplete until their actual gates
and human review criteria hold. No main merge, auto-merge, enqueue, quota change
or historical artifact deletion is included.

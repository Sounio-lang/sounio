# CS6 V7-B Target-23 Adaptive Epistemic Cover

Date: 2026-08-03
State: retrospective retained-receipt certificate; prospective replay pending
Input jobs: Slurm `8519` and `8523`

## Result in one sentence

The retained depth-4 and depth-5 CAPD receipts form a complete 331-leaf
adaptive partition of target parent 23, and both predeclared carriers satisfy
the new bounded epistemic determinant certificate on every leaf.

This is a target-23 orientation result. It is not a legacy certificate, a
global H-PG certificate, V7-B eligibility, or an open-problem solution.

## Why the old cover stopped

The depth-4 run evaluated all `16 * 16 = 256` cells. Both carriers passed the
structural probe on 231 cells and rejected the same remaining 25 cells. The
depth-5 run replaced each rejected cell by its four dyadic children, and both
carriers passed on all 100 children.

The resulting adaptive partition is therefore:

```text
231 retained depth-4 leaves + 25 * 4 depth-5 leaves = 331 leaves
331 leaves * 2 carriers = 662 selected attempts
```

All 662 legacy `CERTIFICATE_PASS` values remained false because every broad
C1, C2, affine, resident-reconstructed, and homogeneous determinant interval
crossed zero. The Liouville intervals were narrow and strictly negative.

## New bounded rule

For one carrier on one adaptive leaf, the epistemic determinant certificate
requires all of the following:

1. the retained worker reports `PROBE_PASS=true`;
2. structural and homogeneous-computation checks pass;
3. the legacy certificate remains false rather than being rewritten;
4. Liouville encloses the same ordered source-scaled two-return Poincare
   determinant and is strictly negative;
5. the exact intersection of the C1, C2, affine, resident-reconstructed,
   homogeneous, and Liouville enclosures is nonempty and strictly negative.

A leaf passes only when both candidate carriers pass. The adaptive cover passes
only when all 331 leaf pairs pass and the 231-plus-100 topology is exact.

Liouville supplies the sign certificate. The six-way intersection supplies
cross-method compatibility under the frozen V7-B determinant rule. Statistical
or computational independence between the enclosures is not assumed.

## Exact result

| Check | Result |
|---|---:|
| depth-4 source cells | 256 |
| retained depth-4 leaves | 231 |
| refined depth-4 parents | 25 |
| selected depth-5 leaves | 100 |
| adaptive leaves | 331 |
| selected attempts | 662 |
| legacy certificates false | 662 |
| structural checks pass | 662 |
| homogeneous computations valid | 662 |
| probe passes | 662 |
| six-way intersection equals Liouville | 662 |
| epistemic leaf certificates pass | 662 |
| paired leaf certificates pass | 331 |
| receipt mutations rejected | 14 of 14 |

The exact-endpoint generator and an independently implemented verifier agree.
Both convert the emitted binary64 hexadecimal endpoints to exact rational
numbers before comparing bounds. No tolerance, decimal approximation, or new
CAPD execution is used.

## Meaning

Within the retained target-23 adaptive domain, the true pointwise determinant
range is contained in each of the six rigorous enclosures and therefore in
their shared intersection. For all 662 attempts satisfying the retained
target-23 filter, the exact rational endpoint audit proves that this
intersection equals the strict-negative Liouville enclosure. The adaptive
leaves form a disjoint complete partition of the target parent. This provides a
retrospective machine-checkable orientation cover for target 23 under both
candidate carriers.

The important methodological observation is that dependency-heavy interval
computations may fail to exclude zero individually while remaining compatible
with a rigorous structural identity that does. Requiring each broad enclosure
to prove the sign alone discards valid information. A certificate can instead
preserve all enclosures, require their shared-scalar compatibility, and use the
strict structural enclosure for orientation.

No novelty or priority claim is made here. Establishing that this rule is new,
generally useful, or stronger than published certificate systems requires a
separate literature comparison and prospective experiments.

## Retrospective boundary

The certificate definition was frozen after the depth-4 and depth-5 executions
already existed and after a 200-attempt boundary audit revealed the candidate
rule. This audit is therefore retrospective. It demonstrates that the retained
raw evidence satisfies the rule; it does not measure prospective success or
guard against experiment-design adaptation.

The next required experiment is a source-fresh Slurm replay with new challenges
and the rule frozen before execution. That replay must regenerate the complete
adaptive cover and reject mutations before the result can be called
prospectively replicated.

## Reproduction

```bash
bash scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_gate.sh
```

The gate:

1. regenerates all 662 certificate rows from both retained archives;
2. regenerates the 331-leaf adaptive topology;
3. runs the independent raw-archive verifier;
4. requires all 14 receipt mutations to fail;
5. verifies the committed file hashes.

## Claim boundary

- `ADAPTIVE_EPISTEMIC_COVER_PASS=true`
- `PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=false`
- `LEGACY_CERTIFICATE_RECLASSIFIED=false`
- `GLOBAL_HPG_CERTIFICATE=false`
- `V7_B_ELIGIBILITY=false`
- `V7_B_WINNER=NONE`
- `PROMOTION_ELIGIBLE=false`
- `OPEN_PROBLEM_SOLVED=false`
- `NOVELTY_OR_PRIORITY_CLAIMED=false`
- `FPGA_EXECUTION=false`

The remaining V7-B work includes the frozen control cells and winner criterion;
target-23 orientation alone cannot supply them.

## Evidence

- Contract: `scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_contract_v1.txt`
- Generator: `scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_analyze.py`
- Independent verifier: `scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_verify.py`
- Mutation gate: `scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_mutations.py`
- Receipt: `scripts/research/receipts/cs6_v7b_target23_adaptive_epistemic_cover_v1/`

## Semantic delta

Intent-Preserved: all historical worker bits remain immutable, and broad
enclosures remain visible even when they cross zero.
Claims-Introduced: the retained 331-leaf target-23 adaptive partition passes the
new retrospective epistemic determinant rule for both carriers.
Evidence-Added: 662 exact certificate rows, 331 leaf rows, independent verifier,
14 fail-closed mutations, hashes, and dual math review.
Claims-Not-Introduced: prospective replication, global H-PG, V7-B eligibility,
carrier winner, promotion, open-problem solution, novelty, or FPGA execution.

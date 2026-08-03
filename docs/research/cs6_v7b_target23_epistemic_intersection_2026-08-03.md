# CS6 V7-B Target-23 Epistemic Intersection Audit

Date: 2026-08-03
State: bounded retained-receipt result; not a global H-PG certificate
Input: depth-5 refinement receipt from Slurm job `8523`

## Question

The depth-5 run produced 200 structurally valid attempts, but every legacy
`CERTIFICATE_PASS` remained false because each tangent carrier enclosure
crossed zero when considered alone. The Liouville enclosure was strictly
negative in every attempt and overlapped the tangent enclosures.

This audit asks a narrower question: after binding all enclosures to the same
source-scaled two-return determinant, is their exact joint intersection
nonempty and strictly negative?

## Shared determinant

All six intervals are intended to enclose the determinant of the same
source-scaled tangent map on the `w = 0` section after two minus-to-plus
returns:

1. C1 `DP` is initialized with `q0`;
2. C2 `DP` is initialized with the same `q0`;
3. the affine enclosure is reconstructed from the C2 image and Hessian;
4. the section-resident enclosure transports normalized event-1 rays and
   restores their scales;
5. the homogeneous enclosure restores both event scales around the normalized
   exterior product;
6. the Liouville identity multiplies the flow determinant ratio by
   `det(source_frame) * radius_u * radius_s`, which is `det(q0)`.

The first three coordinates of the augmented Liouville field are byte-for-byte
the three-dimensional field. Its fourth coordinate integrates the divergence.
No independence assumption is needed for interval intersection; soundness
requires that every interval enclose the same oriented scalar.

## Exact audit

The analyzer reads the retained `stdout.txt` files directly from the committed
archive. Each hexadecimal binary64 endpoint is converted to its exact rational
value with `as_integer_ratio()`. It then evaluates three intersections:

- current four: homogeneous, resident reconstruction, affine, Liouville;
- contract four: C1, C2, resident reconstruction, Liouville;
- all six: C1, C2, affine, resident reconstruction, homogeneous, Liouville.

No decimal parsing, tolerance, midpoint, or new CAPD execution is involved.

## Result

| Check | Count |
|---|---:|
| attempts audited | 200 |
| paired coordinates, both carriers present | 100 |
| structural and homogeneous computation valid | 200 |
| legacy terminal certificate false | 200 |
| Liouville interval strictly negative | 200 |
| each broad C1/C2/affine/resident/homogeneous interval contains zero | 200 |
| current-four intersection nonempty | 200 |
| contract-four intersection nonempty | 200 |
| all-six intersection nonempty | 200 |
| all-six intersection strictly negative | 200 |

In this exact 200-attempt retained sample, Liouville supplies both the maximum
lower endpoint and the minimum upper endpoint of the all-six intersection.
Therefore the joint intersection equals the retained Liouville enclosure in
every audited attempt and excludes zero. Liouville alone certifies the negative
sign; the joint intersection establishes cross-method consistency under the
frozen determinant-compatibility rule.

This establishes a **bounded receipt-level orientation result** for the 100
depth-5 leaves and both candidate carriers. It also shows that the historical
rule requiring one broad tangent enclosure to exclude zero by itself is
strictly stronger than the frozen V7-B determinant-compatibility rule, which
requires a nonempty joint intersection.

## Claim boundary

This result does not change any historical `CERTIFICATE_PASS` bit. It does not
establish a global H-PG theorem, hyperbolicity, a chaotic attractor, V7-B
eligibility, novelty priority, or an open-problem solution. The 231 depth-4
leaves outside this audit already passed the legacy probe, but they were not
re-audited here under the all-six rule. A promotion step would require a frozen
new certificate definition, source-level verifier integration, mutation tests,
and a complete adaptive-cover replay.

## Reproduction

```bash
bash scripts/research/cs6_v7b_target23_epistemic_intersection_gate.sh
```

The gate regenerates the exact tables from the retained archive, compares them
byte-for-byte with the committed receipt, and verifies committed hashes.

## Evidence

- Contract: `scripts/research/cs6_v7b_target23_epistemic_intersection_contract_v1.txt`
- Analyzer: `scripts/research/cs6_v7b_target23_epistemic_intersection_analyze.py`
- Receipt: `scripts/research/receipts/cs6_v7b_target23_epistemic_intersection_v1/`
- Input archive SHA-256:
  `fb4cf8732a8e153a480dc907eb87c38de52d9c2cd8fea95c61b2f3448947084c`

## Semantic delta

Intent-Preserved: broad tangent enclosures are not silently promoted, and all
legacy outcomes remain unchanged.
Claims-Introduced: the exact six-way determinant intersection is nonempty and
strictly negative for all 200 retained depth-5 attempts.
Evidence-Added: exact endpoint analyzer, per-attempt table, deterministic gate,
and orthogonal math review.
Claims-Not-Introduced: global H-PG, V7-B eligibility, hyperbolicity, chaotic
attractor, novelty priority, and open-problem solution.

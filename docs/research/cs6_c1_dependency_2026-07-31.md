# CS6 C1 dependency: affine determinant sign on one bounded tile

**Date:** 2026-07-31
**Lane:** `cs6-c1-affine-projective-20260731`
**Base:** `0bcd234e1563be8182b69883d80b3b8ef2fdb257`
**Status:** a dependency-preserving C2 affine carrier certifies a negative
source-to-section determinant through two returns on one frozen N0 tile; no full-source,
projective-flow, cone, hyperbolicity, or chaos claim is promoted

## 1. Question and bounded answer

The direct C1 and C2 interval matrices for the second CS6 Poincare return lose
the correlation between their four active entries. Their 2x2 determinant boxes
therefore cross zero. This wave asked whether retaining the first-order source
dependency exposed by a rigorous C2 return map can recover the determinant
sign without changing the frozen physical tile, global tangent seed, section,
or return count. A second route tested whether either final-column projective
slope chart is already regular enough to recover the same sign.

The bounded answer is:

```text
SCIENTIFIC_RESULT_CLASS=BOUNDED_TILE_PARAMETERIZED_JACOBIAN_SIGN_CERTIFIED
ROUTE_A=C2_AFFINE_JACOBIAN_CARRIER
ROUTE_A_RESULT=NEGATIVE_PARAMETERIZED_DETERMINANT_CERTIFIED
ROUTE_B=FINAL_COLUMN_PROJECTIVE_SLOPE_CONTROL
ROUTE_B_RESULT=BOTH_CHARTS_INELIGIBLE
A_B_OUTCOME=AFFINE_ONLY
```

On source tile `N0[20000,15000]/[40000,30000]`, after exactly two returns to
the coordinate section `w=0` in the `MinusPlus` direction, the affine carrier
encloses the active determinant in

```text
[-3.254080425965560e-11, -2.476952971052886e-11].
```

The computationally distinct Liouville enclosure is

```text
[-2.865523041266151e-11, -2.865510456662060e-11].
```

They overlap and have the same strict negative sign. This is the determinant of
`D(P^2 o chi)`, where `chi` is the frozen source chart and `D chi=Q0` has
negative determinant. It is not an intrinsic, coordinate-independent
orientation claim about `P^2`; reversing the source-chart orientation reverses
this sign. The result is local to this one tile and this frozen autonomous
system.

## 2. Frozen experiment

The probe fixes all of the following before execution:

```text
zs = 22.3274637391
source = N0
u tile = 20000 / 40000
s tile = 15000 / 30000
u radius = 0.004
s radius = 0.3
section = coordinate w=0
crossing = MinusPlus
return count = 2
CAPD = 5.3.0
interval backend = FILIB
Taylor order = 8
optimization = O0
```

The physical source intervals reconstructed by the exact verifier are

```text
u in [-2.602085213965211e-18, 2.000000000022817e-7]
s in [-1.665334536937735e-16, 2.000000000018654e-5]
```

The tangent seed `Q0` is reconstructed independently from the frozen unstable
and stable frame vectors multiplied by their global radii. The normalized
local coordinates are also reconstructed from the retained source intervals:

```text
delta_u in [-2.500000000061048e-5, 2.500000000061048e-5]
delta_s in [-3.333333333392179e-5, 3.333333333392179e-5]
delta_w = 0 exactly
```

The verifier permits only a small binary64 rounding budget around those
reconstructions. Merely reporting a centered interval, a negative `det(Q0)`,
or a source box that contains the tile is insufficient.

## 3. Route A: C2 affine carrier

Let `J(delta)` denote the active 2x2 derivative of the two-return Poincare map
with respect to the normalized source coordinates seeded by `Q0`. The worker
runs CAPD C2 both on the full tile and at its singleton center. It obtains the
flow derivative, flow Hessian, event-time derivatives, and the return-map
derivative and Hessian through the full `computeDP` overload.

CAPD stores normalized Taylor coefficients. Therefore the actual derivative
of `J_ij` with respect to source variable `k` is reconstructed as

```math
H_k(i,j) = 2 D2P(i,j,j)  when j=k,
H_k(i,j) =   D2P(i,j,k)  when j!=k.
```

The verifier rejects substituting the uncorrected flow Hessian or omitting the
factor two. It also independently reconstructs the first and second impact-time
corrections for the coordinate section and checks them against CAPD's output.

For any point in the star-shaped source tile, the mean-value identity gives

```math
J(delta) = J(0) + integral_0^1 H(t delta)[delta] dt.
```

Choose point binary64 midpoints `M` of the center derivative and `A_k` of the
full-tile Hessian intervals. The worker and exact verifier construct

```math
J(delta) in M + A_0 delta_0 + A_1 delta_1 + R,
```

where `R` contains both the center integration radius and
`(H_k-A_k) delta_k`. This is a C2 mean-value enclosure; it does not assume a C3
bound.

For the affine part, the 2x2 determinant is expanded as an exact quadratic in
the shared source variables. In particular, centered squares are evaluated as
`[0,r^2]` with directed interval arithmetic rather than as independent signed
boxes. The determinant correction from `R` is then added explicitly. The
retained components are:

| Quantity | Lower | Upper | Width |
|---|---:|---:|---:|
| C1 boxed determinant | -2.695357484e-8 | 2.689623068e-8 | 5.384980551e-8 |
| C2 boxed determinant | -2.695357484e-8 | 2.689623068e-8 | 5.384980551e-8 |
| Affine polynomial | -2.865519590e-11 | -2.865513807e-11 | 5.782463688e-17 |
| Residual correction | -3.885608362e-12 | 3.885608362e-12 | 7.771216724e-12 |
| Affine total | -3.254080426e-11 | -2.476952971e-11 | 7.771274549e-12 |

The affine determinant interval is about `6929x` narrower than either flattened
determinant box and lies strictly below zero.

## 4. Route B: honest projective refusal

The projective control in this wave is deliberately narrower than a Riccati or
cone integration. It only attempts slope charts for the two columns of the
final affine Jacobian hull.

The X chart requires both first-row pivots to exclude zero. Its second pivot is

```text
J01 in [-3.324478757e-8, 4.520797930e-9],
```

so the chart has a possible pole. The Y chart requires both second-row pivots
to exclude zero. Its second pivot is

```text
J11 in [-3.956243008e-8, 2.387784413e-9],
```

so that chart also has a possible pole. Both payloads are required to remain
exactly zero when ineligible. The certificate no longer treats projective
failure as a validity condition: a future regular chart may succeed without
invalidating Route A. The observed result remains `AFFINE_ONLY`.

No projective Riccati equation was integrated, no invariant cone was closed,
and no projective conclusion beyond these two final-column charts is supported.

## 5. Distinct Liouville check

For the autonomous three-dimensional flow, a fourth state integrates the
divergence along the same source tile. Once the retained trajectory and
integrated `ELL` state are accepted, the verifier does not trust their reported
derived fields. Using exact rational interval arithmetic, it:

1. requires `ELL` to equal the integrated fourth state exactly;
2. recomputes `exp(ELL)` with a 192-term Taylor enclosure and a rigorous tail;
3. reconstructs the initial section-normal velocity from the frozen geometry;
4. reconstructs the final normal velocity from the retained final state;
5. reconstructs the oriented area of `Q0` from the frozen frame and radii;
6. recomputes the Liouville Poincare determinant identity;
7. requires one joint time, state, and final-velocity intersection with the C1
   and C2 second-return enclosures.

Liouville is a reject-only cross-check. It does not clip or replace the affine
determinant. Its quantitative overlap with the affine carrier is required. The
route is algebraically distinct from C2 derivative propagation, but it shares
the same CAPD/FILIB integration stack and is not an independently implemented
ODE solver.

## 6. Adversarial audit closure

An initial packaged run passed the first 21 mutations but was not retained.
Independent review demonstrated that its verifier accepted a jointly erased
Liouville exponent/state and an arbitrary reported source box. The producer
also formed a centered square through scalar `double` multiplication.

The retained version closes all three issues. Its mutation suite now rejects
`39/39`, including independent attacks on:

- both source axes and all four active entries of `Q0`;
- normalized-coordinate scale, axis order, symmetry, and erasure;
- CAPD's diagonal Hessian normalization and flow/return Hessian distinction;
- impact-time first and second derivatives;
- every determinant component and residual path;
- Liouville exponent/state equality, rational exponential, both normal
  velocities, orbit state, orbit time, determinant formula, and sign;
- projective pole eligibility, dynamic hashes, and summary claims.

After the repairs, the new physical digest remained exactly
`13ffdb34470727fbae6409f32515932b2fd7d28f794544b61194230584191ce6`.
Thus the audit tightened the proof boundary without changing the recorded
physical intervals.

## 7. Literature and novelty boundary

Rigorous integration of Poincare maps and their derivatives is established
CAPD functionality, including C1/C2 maps and return-time corrections. The CAPD
documentation and the survey by Kapela, Mrozek, Wilczak, and Zgliczynski are
the relevant primary implementation and method references:

- [CAPD rigorous Poincare maps](https://capd.sourceforge.net/capdDynSys/docs/html/a05237.html)
- [CAPD maps and normalized Hessians](https://capd.sourceforge.net/capdDynSys/docs/html/a05226.html)
- [CAPD::DynSys: a flexible C++ toolbox for rigorous numerical analysis of dynamical systems](https://doi.org/10.1016/j.cnsns.2020.105578)
- [Recent advances in rigorous computation of Poincare maps](https://arxiv.org/abs/2104.08046)
- [The C-r-Lohner algorithm](https://arxiv.org/abs/0704.0720)

Polynomial or affine dependency preservation with a rigorous interval
remainder is also established rather than novel in isolation:

- [Dependency control with Taylor models](https://doi.org/10.1023/A:1026485406803)
- [Affine Arithmetic: concepts and applications](https://doi.org/10.1023/B:NUMA.0000049462.70970.b6)

Cone conditions and covering relations are also established proof methods;
this wave does not claim to replace or complete them:

- [Cone Conditions and Covering Relations](https://arxiv.org/abs/1103.1959)

The supported novelty is artifact-specific: for this frozen CS6 tile and
oriented source chart, the
combination of CAPD's corrected C2 Poincare Hessian, a source-normalized affine
Jacobian carrier, dependency-preserving determinant expansion, exact
falsification suite, and a distinct Liouville overlap recovers a sign that the
flattened C1 and C2 boxes lose. No literature-priority, first-proof, generic
algorithmic superiority, or solution of an external open problem is claimed.

The online CAPD pages identify the public API but may describe an earlier
published documentation build. The installed 5.3.0 headers and the retained
compile/replay artifacts control exact API semantics for this certificate.

Context7 library resolution was attempted for CAPD 5.3.0 and returned only
unrelated homonyms. No valid `/org/project` entry was available, so the
installed CAPD 5.3.0 headers, executable behavior, and primary CAPD sources are
the API authority for this wave.

## 8. Retained evidence and replay

The canonical local run is bound by:

```text
source SHA-256       208fb35e3376569e82f9050f3a027cf51f1aeac20a0c28f91260fa302b623da0
verifier SHA-256     4e2e2cdc30422f72fe85a377454bbd3473a92e7c962524f4e704f18e210b9435
runner SHA-256       2dec44edc25e10c9a2659182253e441dfbc8d5116f9fc3bc6bcf0b50e791bb46
input SHA-256        d021a60acdc7ca59ff0b6b43d88f76f96ffeebaea4a549f6c0614fc825f25ee9
challenge            e3537f421bb20a73740f70878ddbfde6324c9a6579388c26ffd0e4af0e35469a
receipt SHA-256      0bc299a9350a133cd51ccb2807ca1cd07cf3f4deb016bf19972cb2b4bde78cf6
verification SHA-256 928f270858a39862422fa2cd0358e210c6fc951547c0b37faff7dca09a75e46b
physical SHA-256     13ffdb34470727fbae6409f32515932b2fd7d28f794544b61194230584191ce6
dependency rows      592
link-input rows      19
runtime-library rows 4
```

The repository gate verifies the retained grammar, hashes, dependency
manifests, baseline binding, and `39/39` falsifiers, then performs a fresh
compile and replay under a new challenge:

```bash
bash scripts/ci/cs6_c1_dependency_gate.sh
```

## 9. Execution and trust boundary

```text
default Sounio compiler path used = false
rebuilt current-source CAPD worker path used = true
fallback path used = false
bounded local CPU probe run = true
authorized Foundry/Slurm replay run = false
remote attestation present = false
execution provenance attested = false
worker/source/verifier/receipt hashes bound = true
dependency, link-input, and runtime-library manifests retained = true
dependency content hashes complete = false
independent replay required = true
legacy flattened two-return baseline kept = true
PROJECTIVE_RICCATI_INTEGRATED=false
FULL_SOURCE_CARRIER_PROVED=false
HYPERBOLICITY_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
U250 used = false
```

The ledger verifier is not a standalone authentication of ODE integration.
The first-return trajectory and a coherently substituted `ELL` trajectory could
only be rejected by replaying the bound worker, not by downstream interval
identities alone. The worker source hash, compiler/dependency manifests, and
fresh replay gate are therefore part of the TCB; the no-attestation and
independent-replay fields are substantive limitations.

The two AMD U250 boards were not physically installed and were not used.
Future FPGA work may schedule independent tiles or evaluate frozen candidate
charts, but CPU outward-rounded CAPD plus exact verification remains the
authoritative proof path.

## 10. Semantic lane

```text
Semantic-Lane-ID: cs6-c1-affine-projective-20260731
Owner: codex-root
Concept-IDs: SOUNIO-CS6-C1-SOURCE-DEPENDENCY (proposed)
Intent-Preserved: machine evidence remains narrower than the scientific claim and must expose its falsifiers
Transformation: add a C2 affine source-dependency carrier and a fail-closed final-column projective control for one frozen CS6 tile
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the determinant of the fixed source-to-section parametrization D(P^2 o chi) is negative on the frozen N0 tile by the retained affine carrier and a computationally distinct Liouville overlap
Claims-Forbidden: intrinsic coordinate-independent orientation of P^2; full-source coverage; projective-flow proof; cone closure; hyperbolicity; chaotic attractor; generic solver; external open-problem solution; first-proof priority
Assumptions: autonomous frozen CS6 field; coordinate section; CAPD 5.3.0 FILIB; fixed global Q0; normalized rectangular source coordinates; retained worker is part of the replay TCB
Write-Set: scripts/research/cs6_c1_dependency_*; scripts/research/receipts/cs6_c1_dependency_*; scripts/ci/cs6_c1_dependency_gate.sh; docs/research/cs6_c1_dependency_2026-07-31.md; docs/internal/concepts/registry.tsv; .claude/llm_offload_log.md
Read-Set: scripts/research/cs6_section_resident_reconditioned_two_return_*; local CAPD 5.3.0 headers and libraries; semantic and blocker contracts
Positive-Witness: retained affine determinant and Liouville determinant are jointly negative and overlap; 39/39 semantic mutations rejected
Negative-Witness: flattened C1/C2 determinants cross zero and both final-column projective charts have possible poles
Acceptance-Gate: bash scripts/ci/cs6_c1_dependency_gate.sh
Integration-Target: review branch only; no main merge requested
Authoritative-Only-If: retained artifacts verify, a fresh replay has the same physical digest, mandatory dual-provider math review is logged, and all forbidden promotion fields remain false
```

Integration receipt:

```text
Semantic-Outcome: bounded affine-only determinant-sign certificate for one frozen source chart and two-return tile
Concept-Status-Before: proposed
Concept-Status-After: hypothesis
Distinctions-Added: flattened interval matrix versus source-dependent affine Jacobian; final-column chart refusal versus integrated projective proof
Distinctions-Preserved: formal model != empirical claim; computational provenance != physical causality; bounded tile != full source
Distinctions-Erased: none
Evidence-Run: canonical local CAPD package plus fresh-replay gate and 39 mutation tests
Fallback-Path: none
Legacy-Kept: yes; the flattened two-return baseline is hash-locked and unchanged
Conflicting-Lanes: none observed by sounio semantic and live coordination scanners
Next-Semantic-Interface: exact full-source tiling with a rigorous projective or cone channel only where its denominators are certified
```

## 11. Remaining blocker

```text
Blocker-ID: BLK-20260731-cs6-c1-full-source-dependency-cover
Status: classified
Severity: B3
Class: evidence-gap
Owner: codex-root
Lane: cs6-c1-affine-projective-20260731
Worktree: /tmp/sounio-cs6-c1-affine-projective-20260731
Branch: research/cs6-c1-affine-projective-20260731
Files-Owned: scripts/research/cs6_c1_dependency_*; scripts/research/receipts/cs6_c1_dependency_*; scripts/ci/cs6_c1_dependency_gate.sh; docs/research/cs6_c1_dependency_2026-07-31.md; docs/internal/concepts/registry.tsv; .claude/llm_offload_log.md
Files-Read-Only: scripts/research/cs6_section_resident_reconditioned_two_return_*
Do-Not-Touch: bounded promotion fields, frozen geometry, Q0, and retained receipts without a new versioned evidence chain
Repro: bash scripts/ci/cs6_c1_dependency_gate.sh
Observed: one bounded source chart has a negative affine source-to-section determinant while both final-column projective charts are ineligible
Expected: an exact no-gap full-source cover certifies every accepted leaf and explicitly classifies every projective pole or subdivision
Acceptance-Gate: future exact-cover aggregate verifies leaf disjointness/completeness, per-leaf C0 transversality, affine or projective sign, and a distinct Liouville overlap
Evidence-Level: E2
Evidence: docs/research/cs6_c1_dependency_2026-07-31.md; scripts/research/cs6_c1_dependency_provenance_v1.txt; scripts/research/cs6_c1_dependency_receipt_v1.txt
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: implement adaptive exact source subdivision driven by affine residual width and certified projective-chart eligibility, then aggregate without dropped or duplicated leaves
```

This blocker limits promotion beyond the bounded tile. It does not negate the
local result. Local evidence was captured; remote exhaustive execution and any
hardware-accelerated cover remain unperformed.

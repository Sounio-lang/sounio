# CS6 event-normalized projective cocycle: a useful failure

**Date:** 2026-08-01
**Lane:** `cs6-plucker-cocycle-20260801`
**Base:** `1274b9a3b0e65ba738d4856c940ef6c8320d9761`
**Status:** the predeclared boxed event-normalization hypothesis is false on
the same 53 retained CS6 coordinates; the machine and its negative result are
retained, while no projective, full-source, hyperbolicity, chaos, or priority
claim is promoted

## 1. Question and answer

The previous full-source-cover wave left this differentiating hypothesis:

```text
H-PG-CS6:
A homogeneous ray carrier with validated dynamic chart normalization and a
separate signed exterior scale will reduce CS6 tangent-cocycle wrapping enough
to close a strict superset of the affine leaves at the same coordinates.
```

This wave implemented the smallest honest version that changes the numerical
evaluation order. It normalizes each tangent ray after the first Poincare
event, uses those normalized rays as the C1 seed of a newly integrated local
second return, normalizes again at the second event, and reconstructs the
signed scale and exterior determinant. Merely factoring the final matrix was
kept as an independent control.

The retained answer is:

```text
coordinates                              53
probe-valid leaves                       52
root interval-domain failure              1
affine C2 certificates                   28
fixed final covector-chart certificates   0
event-normalized homogeneous certificates 0
homogeneous rescues over affine            0
affine certificates lost by homogeneous  28
```

Thus `H-PG-CS6` is false on the retained coordinates. This does not refute
projective reconditioning in general. It isolates the missing ingredient:
the implementation keeps state dependence at the first event, but boxes each
tangent ray before the second integration. The next experiment must preserve
the shared source variables through that switch.

## 2. Exact geometric boundary

The tangent space of the two-dimensional Poincare section is two-dimensional,
so the ray object in this experiment is

```math
\operatorname{Gr}(1,2) \cong \mathbb{RP}^{1}.
```

Its homogeneous coordinates are simply a nonzero vector `[x:y]`; there are no
nontrivial Plucker relations in this case. The Grassmannian `Gr(2,2)` is a
point and therefore carries no projective scale or orientation. Separately,
the vector space `Lambda^2 R^2` is one-dimensional; the two-ray exterior
channel lives there and retains signed scale through the determinant.

Between events, the variational flow is ambient three-dimensional. A genuine
continuous projectivization there would evolve in `RP2`. This worker does not
do that. Its exact scope is frozen in every receipt:

```text
DISCRETE_POINCARE_COCYCLE=true
CHART_NORMALIZATION_SCOPE=EVENT_BOUNDARIES
GRASSMANN_SCOPE=GR_1_2_EQUALS_P1
CONTINUOUS_RICCATI_INTEGRATED=false
GENERAL_GRASSMANN_PLUCKER_INTEGRATOR=false
TANGENT_SOURCE_DEPENDENCY_AT_SWITCH=BOXED_NOT_AFFINE
AUTONOMOUS_VECTOR_FIELD=true
EVENT_TIME_SENSITIVITY_PROPAGATED=false
NONAUTONOMOUS_GENERALIZATION_PROVED=false
```

Consequently, “event-normalized homogeneous ray cocycle” is the precise name.
“General Plucker integrator” and “continuous Riccati integrator” are forbidden
interpretations.

## 3. Four paired methods

Every computable coordinate is evaluated by four distinct routes in one
process, over the same frozen `N0`, tangent seed, section, direction, and two
returns.

| Route | Dependency retained | Reintegrates after event 1 | Role |
|---|---|---:|---|
| flattened C1 matrix | independent matrix-entry intervals | no | loose control |
| affine C2 determinant | shared source variables through the determinant polynomial | no | strongest baseline |
| four fixed final covector charts | final boxed columns only | no | algebraic projective control |
| event-normalized homogeneous | normalized event-1 rays, then local C1 reintegration | yes | tested hypothesis |

The fixed `X`, `Y`, `PLUS`, and `MINUS` routes are final covector
factorizations. They are not Riccati flows.

At event 1, let `c_0` be the source center, `c_1` the midpoint enclosure of
`P_1(c_0)`, and `M` the midpoint enclosure of `DP_1(N_0)`. The C0 carrier uses
the same normalized source interval `delta`:

```math
x_1(\delta) \in c_1 + M\delta + I\rho,
\qquad
\rho = (P_1(c_0)-c_1) + (DP_1(N_0)-M)\delta.
```

The section-normal coordinate is set exactly to zero. This mean-value form
retains source dependence for the event state used by the local return.

For a tangent column `v` and a chart matrix with covector rows
`C = (ell; m)`, the worker computes

```math
p = \ell v, \qquad r = \frac{m v}{p}, \qquad
u = C^{-1}\binom{1}{r}, \qquad v \subseteq p u.
```

A chart is eligible only when `0` is excluded from the full interval pivot.
Among `X`, `Y`, `X+Y`, and `X-Y`, the deterministic choice maximizes

```math
\frac{\operatorname{margin}(p)^2}{\lVert\ell\rVert_2^2},
```

with frozen candidate order as the tie breaker. The pivot remains signed.
The verifier recomputes both the choice and the componentwise containment
`v subseteq p u`; overlap alone is insufficient.

Let `p_1j,u_1j` be the first-event decomposition of column `j`. The two
normalized vectors `u_10,u_11` become the C1 seed of a fresh local second
Poincare return. Its columns `z_2j` are normalized again as `p_2j,u_2j`.
The cumulative reconstruction is

```math
D(P^2)(c_0)e_j \in (p_{1j}p_{2j})u_{2j},
```

and the homogeneous exterior channel is

```math
\Omega_2 =
(p_{10}p_{20})(p_{11}p_{21})\det(u_{20},u_{21}).
```

This formula remains valid when the two columns use different charts because
the determinant is taken after reconstructing both physical normalized
vectors. No same-chart slope-difference shortcut is used.

## 4. Fail-closed verifier

The leaf verifier parses binary64 hexadecimal endpoints into exact Python
`Fraction` values. It independently reconstructs:

- canonical input geometry, source hash, leaf challenge, and physical digest;
- C2 impact-time corrections for the coordinate section;
- the event-1 mean-value C0 carrier;
- all candidate pivots, normalized scores, chart choices, slopes, and vectors;
- componentwise ray reconstruction at both events;
- the local P2 Poincare derivative from flow and impact correction;
- cumulative matrix columns, signed scales, and both exterior determinants;
- a joint intersection of homogeneous, reconstructed-matrix, affine, and
  Liouville enclosures;
- event ordering, transversality, and the post-section `Plus` witness.

`PROBE_PASS=true` means the negative or positive experiment is structurally
valid. It does not mean that any orientation certificate passed.

The retained verifier hashes 235 payload files, validates every artifact hash
declared by the run manifest, enforces exact manifest and TSV schemas, and
recomputes all seven reported metrics from each receipt. It binds worker
stderr per leaf, replays the exact leaf verifier on all 52 nonempty receipts,
reproduces the one root failure class, and reruns a 76-mutation audit on a
retained leaf. The gate adds chart-choice and plain indexed-file mutations,
plus coordinated root-status, leaf-method, extra-file, and path-symlink
mutations whose relevant indexes and enclosing manifests are rehashed. It also
checks live source snapshots and offers an optional fresh CAPD rebuild and
one-leaf replay.

This is a local integrity envelope, not independent attestation. An author who
rewrites a payload and every enclosing hash remains inside the local trusted
computing base; only an independently controlled execution and signed receipt
chain would close that boundary.

## 5. Retained result

All results below are local bounded CPU evidence from CAPD 5.3.0, FILIB,
`-frounding-math`, and `-O0`. No U250 was installed or used.

| Observation on 52 computable leaves | Result |
|---|---:|
| flattened C1 determinant sign-definite | 0 |
| affine determinant strictly negative | 28 |
| any fixed final chart sign-definite | 0 |
| homogeneous determinant sign-definite | 0 |
| Liouville determinant strictly negative | 52 |
| homogeneous width narrower than affine | 0 |
| homogeneous width narrower than best fixed chart | 0 |
| homogeneous width narrower than flattened C1 | 43 |
| event chart changes, over 104 ray trajectories | 92 |

The 53 coordinates are a unique union of three earlier scout input sets. They
include nested sampled leaves and do not form a partition. Their areas must not
be summed or presented as certified source coverage.

The paired width ratios are:

| Ratio | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| homogeneous / flattened C1 | 0.104770 | 0.493768 | 0.569182 | 1.250439 |
| homogeneous / affine | 49.8820 | 1204.35 | 1562.27 | 6782.23 |
| homogeneous / best fixed chart | 15.9475 | 32.2775 | 36.9989 | 99.8932 |

Forty leaves changed chart on both rays and twelve changed chart on one ray.
The minimum certified pivot score was `2.9449732793240058e-08`. Chart
eligibility was therefore not the limiting condition.

The homogeneous route beating the flattened C1 box on 43/52 leaves shows that
event normalization changed the enclosure geometry and was not empty
postprocessing. Its loss to the affine carrier on all 52 leaves supports a
sharper diagnosis for this implementation: coordinate conditioning did not
recover the correlations discarded when the event-1 tangent rays were
converted to independent interval boxes. This is evidence for the dependency
diagnosis, not a proof that no other implementation detail contributes.

The hardened rerun used `1,468,873 ms` of aggregate worker time. Mean
valid-leaf time was `28,178.212 ms`, mean receipt size was `35,505.48` bytes,
all 76 retained
mutations were rejected, and exact retained replay matched `52/52` receipts.
The root remains `COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN`, not a counterexample.

The final evidence anchors are:

```text
worker source SHA256    dc95cb35a36a7edc218f5b718ee35282de2524bc458861f70585f7d31c718e44
leaf verifier SHA256    02ea306e833002db9c301707061a67d878c90375f71bc7880729c02f12273b7a
runner SHA256           97a00c78031a8ff6101f7cd795e46f93e7b67b6c65c8715da587a584f2cc5ffa
retained verifier SHA256 ce32d2cc8cd76bcb18aa77de98878fd952a02e2fd055d48fd4feae34f03181a8
run manifest SHA256     21f0ca224fc79d1cfc815f203100e091bf27dbb070cb3a2c69f6056b7a6fee27
files index SHA256      740ae39070ff90604048c37d25e2f0de954be9ea55e020a529e094699a82802d
```

## 6. Literature boundary and novelty window

The live primary-source refresh through 2026-08-01 confirms that the ingredients
are established separately:

- rigorous Poincare maps, derivatives, and coordinate/section effects are
  established validated-numerics practice
  ([Kapela, Wilczak, and Zgliczynski](https://doi.org/10.1016/j.cnsns.2022.106366),
  [CAPD::DynSys](https://doi.org/10.1016/j.cnsns.2020.105578), and
  [Walawska and Wilczak](https://doi.org/10.1016/j.amc.2016.07.005));
- a formally verified Poincare derivative, intermediate-section composition,
  impact correction, and affine dependency tracking already exist
  ([Immler](https://doi.org/10.1007/s10817-017-9448-y));
- Riccati flow on Grassmann charts and dynamic chart selection are known
  ([Schiff and Shnider](https://doi.org/10.1137/S0036142996307946) and
  [Ledoux, Malham, and Thummler](https://doi.org/10.1090/S0025-5718-10-02323-9));
- separating angular subspace evolution from exterior scale is known in Evans
  computations
  ([Humpherys and Zumbrun](https://doi.org/10.1016/j.physd.2006.07.003));
- Taylor models and affine arithmetic are established ways to retain functional
  dependence and reduce wrapping
  ([Neher, Jackson, and Nedialkov](https://doi.org/10.1137/050638448) and
  [de Figueiredo and Stolfi](https://doi.org/10.1023/B:NUMA.0000049462.70970.B6)).

Therefore this lane claims none of those ingredients as new. The directed
search did not locate a primary source combining validated event composition,
dependency-preserving projective tangent reconditioning, signed exterior
scale receipts, and matrix/Liouville cross-checks as one reusable machine.
That is a literature gap to test, not proof of novelty or priority.

The most defensible future method claim, only after positive multi-system
evidence, would be “dependency-preserving validated projective
reconditioning at Poincare events.”

## 7. Daring next hypothesis

```text
H-APG-CS6:
If the event state and both tangent rays retain the same source symbols through
validated projective normalization, then reintegrating the local second return
will recover the affine carrier's dependency advantage while retaining useful
event-wise conditioning and a separately checkable signed exterior scale.
```

Represent, with shared `delta`,

```math
x_1(\delta), \qquad
v_{1j}(\delta)=DP_1(x_0+\delta)e_j
```

as affine forms first and Taylor models of order 2 or 3 if validated reciprocal
and composition widen too much. Select a chart only when its dependent pivot
excludes zero, compute `u_j(delta)=v_j(delta)/p_j(delta)` without separating
the symbols, and reintegrate `(x_1(delta),u_j(delta))` through the same local
return. Signed pivots and the exterior channel remain independent audit paths.

Freeze the first pilot to four deliberately stressed retained leaves:

```text
U08-0000000192_S08-0000000064  smallest boxed homogeneous/affine ratio
U15-0000008192_S15-0000024576  largest boxed homogeneous/affine ratio
U12-0000003072_S16-0000016384  largest gain over flattened C1
U16-0000016384_S12-0000003072  boxed homogeneous loss to flattened C1
```

Pilot success requires all four dependent carriers to verify, widths strictly
narrower than the present boxed route on all four, no loss of the retained
affine-positive leaf, and either one affine-negative rescue or a homogeneous
width narrower than affine on at least one leaf. The 53-leaf promotion gate is
stricter: zero affine losses, at least one homogeneous rescue, exact replay,
all mutations rejected, and independent execution attestation.

Only after that CPU pilot passes should the two U250 boards be used. Their
honest role is parallel population, scheduling, and non-authoritative margin
filtering. They do not provide outward-rounding proof, repair dependency, or
replace the CPU verifier and retained receipts.

## 8. Replay and execution boundary

Retained verification and mutation gates:

```bash
bash scripts/ci/cs6_plucker_cocycle_gate.sh
```

One fresh CAPD rebuild, leaf execution, exact verification, and 76 mutations:

```bash
CS6_PLUCKER_COCYCLE_REPLAY=1 \
CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config \
bash scripts/ci/cs6_plucker_cocycle_gate.sh
```

The full retained 53-coordinate run is reproducible with
`scripts/research/cs6_plucker_cocycle_run.py`; it is intentionally not part of
the fast gate.

```text
default Sounio compiler path used = false
rebuilt current-source CAPD worker path used = true
fallback path used = false
bounded local CPU run = true
authorized remote run = false
execution provenance attested = false
U250 used = false
legacy flattened, affine, fixed-chart, and Liouville paths kept = true
FULL_SOURCE_CARRIER_PROVED=false
HYPERBOLICITY_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
NOVELTY_OR_PRIORITY_CLAIM=false
PROMOTION_ELIGIBLE=false
```

## 9. Semantic lane

```text
Semantic-Lane-ID: cs6-plucker-cocycle-20260801
Owner: codex-root
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: a machine-level research opening must retain dependency, uncertainty, provenance, and falsifiers rather than promote a suggestive scout into a theorem
Transformation: replace final-only projective readout with event-1 homogeneous normalization, local second-return reintegration, event-2 normalization, and signed exterior reconstruction on the same retained coordinates
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the boxed discrete event-normalization hypothesis is false on the retained 53-coordinate set; the route is narrower than flattened C1 on 43/52 computable leaves but narrower than neither affine nor best fixed final chart on any leaf
Claims-Forbidden: general Plucker integrator; continuous Riccati integration; full-source carrier; partition coverage; projective certificate; cone proof; hyperbolicity; chaotic attractor; open problem solved; novelty or priority
Assumptions: frozen CS6 N0 model and section; autonomous vector field; CAPD 5.3.0 FILIB outward rounding; local unattested CPU execution; retained coordinates are paired samples, not a partition
Write-Set: scripts/research/cs6_plucker_cocycle_*; scripts/research/receipts/cs6_plucker_cocycle_*; scripts/ci/cs6_plucker_cocycle_gate.sh; docs/research/cs6_plucker_cocycle_2026-08-01.md; .claude/llm_offload_log.md
Read-Set: prior CS6 C1/C2/full-source evidence; CAPD headers and static libraries; primary literature listed above
Positive-Witness: 52/52 exact leaf replays; 76/76 retained mutations rejected; 43/52 homogeneous widths narrower than flattened C1; fresh rebuild/replay gate passes
Negative-Witness: zero homogeneous certificates; 28 affine losses; zero homogeneous rescues; root interval-domain failure
Acceptance-Gate: bash scripts/ci/cs6_plucker_cocycle_gate.sh and the opt-in CS6_PLUCKER_COCYCLE_REPLAY=1 form
Integration-Target: review branch only; no main merge requested
Authoritative-Only-If: a future dependency-preserving carrier passes its exact verifier, zero-loss/positive-rescue 53-leaf gate, independent benchmark, and remote execution attestation
```

## 10. Remaining evidence blocker

```text
Blocker-ID: BLK-20260801-cs6-affine-projective-ray-dependency
Status: classified
Severity: B3
Class: evidence-gap
Owner: future-cs6-affine-projective-cocycle
Lane: cs6-plucker-cocycle-20260801
Worktree: /tmp/sounio-cs6-plucker-cocycle-20260801
Branch: research/cs6-plucker-cocycle-20260801
Files-Owned: scripts/research/cs6_plucker_cocycle_*; scripts/research/receipts/cs6_plucker_cocycle_*; scripts/ci/cs6_plucker_cocycle_gate.sh; docs/research/cs6_plucker_cocycle_2026-08-01.md; .claude/llm_offload_log.md (one append-only review row)
Files-Read-Only: prior CS6 retained receipts; CAPD 5.3.0 sources; primary literature
Do-Not-Touch: prior certificates, promotion fields, or full-source claims without new exhaustive evidence
Repro: bash scripts/ci/cs6_plucker_cocycle_gate.sh
Observed: 0 homogeneous certificates, 0 affine rescues, 28 affine losses, and homogeneous width wider than affine on all 52 computable leaves
Expected: a dependent projective ray carrier has zero affine losses, at least one rescue, exact signed-scale reconstruction, and independent verifier and Liouville agreement on the same coordinates
Acceptance-Gate: future 53-leaf gate requiring AFFINE_LOSS_COUNT=0, AFFINE_PROJECTIVE_RESCUE_COUNT>0, exact replay, complete mutation rejection, and remote attestation
Evidence-Level: E2
Evidence: scripts/research/receipts/cs6_plucker_cocycle_retained_53_v1; scripts/ci/cs6_plucker_cocycle_gate.sh
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: logged dual xAI/Z.AI math review; final result-boundary review passed after causal-scope wording was tightened
Next-Action: implement shared-delta affine or order-2/3 Taylor ray normalization on the frozen four-leaf pilot before any FPGA or larger-cover run
```

## 11. Integration receipt

```text
Semantic-Outcome: boxed event normalization was implemented, independently replayed, and falsified on the retained coordinates; dependency-preserving event normalization is the next interface
Concept-Status-Before: H-PG-CS6 was an untested projective opening
Concept-Status-After: H-PG-CS6 is false on the retained coordinates; H-APG-CS6 is a bounded untested hypothesis
Distinctions-Added: fixed final covector factorization versus discrete event cocycle; chart conditioning versus source-dependency retention; RP1 event normalization versus RP2 continuous projective flow
Distinctions-Preserved: uncertainty is not value; computational provenance is not physical causality; sampled leaves are not source coverage; local execution is not attestation
Distinctions-Erased: none
Evidence-Run: 53-coordinate CAPD CPU run; 52/52 retained leaf replay; 76/76 mutation audit; default and fresh rebuild/replay gates
Fallback-Path: none
Legacy-Kept: flattened C1, affine C2, four fixed final charts, and Liouville controls
Conflicting-Lanes: none observed by sounio semantic status or live coordination
Next-Semantic-Interface: a shared-source-symbol projective ray carrier with signed scale and exact receipt reconstruction
```

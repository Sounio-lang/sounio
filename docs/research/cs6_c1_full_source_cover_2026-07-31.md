# CS6 C1 full-source cover: exact machine, sparse scout, and projective opening

**Date:** 2026-07-31
**Lane:** `cs6-c1-full-source-cover-20260731`
**Base:** `f859dadf2d33fb78637f8d06cf0e5bc1a103e917`
**Status:** an exact adaptive cover and replay machine now exists; 53 local
unattested scout leaves expose a localized anisotropic refinement signal and a
prohibitive uniform-grid scale, but no full-source, projective, cone,
hyperbolicity, or chaos claim is promoted

## 1. The strongest supported result

This wave converts the previous one-tile dependency result into a fail-closed
source-cover machine. It adds:

1. independent canonical dyadic depths and indices for the two source axes;
2. a CAPD 5.3.0/FILIB C2 worker with the dependency-preserving affine
   determinant carrier, Liouville overlap, and four fixed projective readout
   charts;
3. an exact-rational leaf verifier with `56/56` retained mutation rejections;
4. a deterministic breadth-first adaptive tree with exact parent/child closure;
5. a canonical aggregator that reconstructs the logical cover, re-verifies
   every certified terminal, rebuilds the worker, replays under a distinct
   challenge, and refuses external promotion without attestation.

The retained sample result is exactly:

```text
UNIQUE_SAMPLED_LEAVES=53
AFFINE_CERTIFIED=28
SUBDIVISION_REQUIRED=24
COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN=1
POSITIVE_PROJECTIVE_METHOD_EXERCISED=false
FULL_SOURCE_CARRIER_PROVED=false
```

These are sampled dyadic leaf centers at selected depth pairs. They are not an
exact partition and do not cover a positive fraction of the continuous source.

## 2. Frozen mathematical contract

The worker preserves the frozen CS6 source `N0`, tangent seed `Q0`, coordinate
section, `MinusPlus` crossing direction, and exactly two Poincare returns from
the bounded dependency experiment. A leaf changes only its logical dyadic
coordinates:

```text
U = [u_index / 2^u_depth, (u_index + 1) / 2^u_depth]
S = [s_index / 2^s_depth, (s_index + 1) / 2^s_depth]
```

The physical source box is reconstructed from the global `N0` chart. The
tangent seed remains global rather than being silently rescaled per leaf. The
verifier reconstructs both logical and physical geometry using exact
fractions, permits only the explicit binary64 serialization slack, and binds
the canonical input bytes, source hash, leaf identity, and challenge.

For each leaf, the accepted affine route retains shared source variables in a
C2 mean-value carrier

```math
J(delta) in M + A_0 delta_0 + A_1 delta_1 + R
```

and expands the 2x2 determinant before interval evaluation. The flattened C1
and C2 matrix boxes remain diagnostics. A certificate requires the affine
determinant to be strictly negative, the separately reconstructed Liouville
determinant to have the same sign, and the two enclosures to overlap.

CAPD's `D2P`, flow `IHessian`, and `d2T` outputs are normalized second-order
Taylor coefficients: diagonal entries are half the corresponding true second
derivative. The impact-time reconstruction deliberately mirrors CAPD's
`AbstractSection::computeDP` coefficient formula. The factor two is applied
later, exactly when a diagonal `D2P` coefficient is converted into the true
derivative of a Jacobian entry for the affine carrier.

Four fixed covector charts (`X`, `Y`, `PLUS`, and `MINUS`) are also evaluated
with explicit pivot eligibility and determinant identities. None produced a
positive retained certificate. Thus the implementation exercises honest
projective refusal, not a projective-flow result.

## 3. Retained scouts

All three runs used CAPD 5.3.0, FILIB, `-frounding-math`, and `-O0` on the local
CPU path. No U250 was present or used.

| Retained run | Depth pairs | Leaves | Certified | Subdivide | Compute unresolved |
|---|---|---:|---:|---:|---:|
| core | `8:8,12:12,14:14,15:15,12:16,16:12` plus root | 25 | 8 | 16 | 1 |
| boundary | `13:13,13:14,14:13` | 12 | 4 | 8 | 0 |
| dense boundary | `13:14`, grid 4 | 16 | 16 | 0 | 0 |
| **total** | selected strata only | **53** | **28** | **24** | **1** |

The depth/status distribution is:

```text
0:0    0 certified / 1 interval-domain failure
8:8    0 / 4
12:12  0 / 4
12:16  0 / 4
13:13  0 / 4
13:14 20 / 20
14:13  0 / 4
14:14  4 / 4
15:15  4 / 4
16:12  0 / 4
```

The striking result is anisotropic: every one of the 20 sampled `13:14`
leaves certifies, while all sampled `13:13` and `14:13` leaves require more
subdivision. This is evidence that refinement along the second source axis can
recover dependency at these sampled locations. It is not evidence that every
`13:14` tile certifies.

The root failure is explicitly classified as interval-domain unresolved. The
worker encountered interval division by a denominator containing zero before
it could emit a complete receipt. It is not counted as a mathematical
counterexample or a certified leaf.

## 4. Why uniform subdivision is the wrong machine

A uniform `13:14` partition contains

```text
2^(13+14) = 134,217,728 leaves.
```

The dense scout touched 16 of those leaves, or `1 / 8,388,608` of that discrete
grid. Its observed mean was `41.365125 s/leaf`. A deliberately naive linear
extrapolation gives:

```text
serial compute       about 175.93 CPU-years
ideal 128-way compute about 502.02 days
receipt storage       about 2.471 TB
```

This extrapolation ignores scheduler, cache, tail-latency, and adaptive effects
and is not a performance forecast. It is a scale falsifier: uniform refinement
is not a credible route to the proof on the present worker. The storage figure
uses the observed dense-scout mean of `18,410.5 bytes/receipt`, yielding
`2,471,015,481,344` bytes before filesystem and index overhead.

The two U250 boards would not change that conclusion by themselves. They could
schedule independent leaves, evaluate frozen chart candidates, or accelerate
non-authoritative filters once installed. They do not repair interval wrapping,
chart poles, C0 nontransversality, or missing outward-rounding evidence.

## 5. Exact adaptive aggregation

The adaptive runner begins at the canonical root and evaluates breadth-first
waves. An uncertified node is bisected along the shallower axis, with ties
resolved to `U`. `max_nodes` and `max_axis_depth` are hard bounds. When either
bound prevents a split, the region remains an explicit `UNRESOLVED` terminal.

The aggregator accepts only the exact binary tree induced by those actions. It
checks with rational arithmetic that terminal interiors are disjoint and their
areas sum to one. Internal and unresolved rows carry no certificate evidence.
Certified terminals bind canonical input, challenge, receipt, verification,
physical digest, and method.

The local full-source predicate additionally requires:

- zero unresolved terminals and accepted rational area exactly one;
- fresh verification of every retained certified terminal;
- a canonical two-pass rebuild with stable compiler, headers, link inputs, and
  runtime-library hashes;
- a distinct replay challenge and matching physical digest for every leaf;
- all structural mutations and every observed method's leaf mutations rejected;
- a durable replay audit bundle containing the replay ledger and build-manifest
  preimages.

Even if that local predicate becomes true, the v1 certificate keeps
`FULL_SOURCE_CARRIER_PROVED=false`, `PROMOTION_ELIGIBLE=false`, and
`EXECUTION_PROVENANCE_ATTESTED=false`. Remote attestation is a separate
obligation, not a boolean supplied by the caller.

## 6. Adversarial audit and retained trust boundary

Independent review found that an earlier aggregator accepted caller-selected
worker, verifier, and source hashes. A substitute trio could therefore produce
a false one-node full certificate. That implementation was not retained.

The current aggregator removes those inputs and uses canonical sibling source
and verifier files, freezes their bytes before use, reconstructs exact input
coordinates, rebuilds the worker internally, and binds a replay ledger. It
publishes certificates by exclusive atomic creation and rejects existing or
symlink outputs. The gate covers exact-tree mutations, unresolved-row grammar,
fail-closed partial aggregation, and publication collisions.

Historical provenance remains weaker than current replay provenance. The three
raw scout manifests were locally self-reported with no external attestation.
Their exact source, runner, and both verifier versions are now retained by
content hash. The original worker binary hash is recorded, but that binary is
absent. Therefore:

```text
ORIGINAL_WORKER_BINARY_RETAINED=false
EXACT_HISTORICAL_EXECUTION_REPLAYABLE=false
CURRENT_VERIFIER_OUTPUT_REPRODUCED=52/52 nonempty receipts
CURRENT_MUTATION_AUDITS_REPRODUCED=3/3
```

A fresh canonical rebuild is new evidence. It is not retroactively identified
as the missing historical binary.

## 7. Literature boundary

The individual ingredients are established:

- rigorous C1/C2 Poincare maps, return-time correction, and coordinate choice
  are part of validated numerics and CAPD practice
  ([Kapela, Wilczak, and Zgliczynski](https://arxiv.org/abs/2104.08046));
- Lohner-type rearrangement and high-order variational integration address
  wrapping and derivative propagation
  ([Wilczak and Zgliczynski](https://arxiv.org/abs/0704.0720));
- Riccati evolution is a standard local chart on Grassmann manifolds
  ([Schiff and Shnider](https://doi.org/10.1137/S0036142996307946));
- dynamic Grassmann chart selection is established in non-rigorous spectral
  shooting
  ([Ledoux, Malham, and Thummler](https://arxiv.org/abs/0710.1037));
- exterior-product scale can be separated from angular subspace evolution
  ([Humpherys and Zumbrun](https://arxiv.org/abs/math/0508020));
- rigorous dynamics has already used projectivized tangent bundles, cone
  conditions, and validated variational equations
  ([Wilczak and Zgliczynski](https://arxiv.org/abs/0905.3924),
  [Walawska and Wilczak](https://arxiv.org/abs/1509.07388)).

Therefore this wave does not claim the invention of Riccati equations,
Grassmann integration, projective rigorous dynamics, Poincare derivatives,
Liouville determinants, affine arithmetic, or adaptive subdivision.

The defensible research opening is narrower and more interesting:

> Build a validated interval Grassmann/Plucker variational cocycle for rigorous
> Poincare maps that emits proof-carrying chart changes, pivot margins, and an
> exterior scale cocycle, then use those receipts to drive an exact adaptive
> source cover and cone or dominated-splitting tests.

The literature above supplies adjacent pieces, but this survey did not locate a
primary source demonstrating that complete proof-producing composition as a
reusable validated integrator. This is a novelty window to test, not a priority
claim.

## 8. The daring next hypothesis

```text
H-PG-CS6:
A homogeneous Plucker carrier with validated dynamic chart normalization and
separate exterior log-scale will reduce the CS6 tangent-cocycle wrapping enough
to close materially larger source regions than the affine-only worker, without
weakening C0 transversality or Liouville consistency.
```

The smallest differentiating experiment is not a full-source run. It is a
four-way comparison on the already retained 53 coordinates:

1. flattened matrix plus Liouville;
2. current dependency-preserving affine carrier;
3. fixed Riccati charts;
4. homogeneous Plucker evolution with validated chart normalization.

Predeclared metrics are certified area at equal leaf coordinates, minimum
pivot margin, interval width per return, number of chart changes, wall time,
and receipt bytes. The new route is demoted if any of these occur:

- it cannot cross a chart singularity with a positive validated pivot margin;
- its determinant or exterior scale fails overlap with matrix and Liouville
  baselines;
- it certifies no strict superset of the affine leaves at comparable depth;
- chart branching grows faster than the eliminated source subdivision;
- a positive result disappears on independent Rössler or forced-pendulum
  Poincare benchmarks.

Only after that experiment should the adaptive policy change from shallower-axis
bisection to a proof-margin score derived from C0 transversality, C2
sensitivity, pivot margin, and cone slack.

## 9. What this could enable

The machine-level target is not determinant sign for its own sake. A validated
projective cocycle can preserve the evolving tangent direction while a separate
scale channel controls expansion. Coupled to exact cover closure, that can
supply finite evidence for:

- invariant cone or dominated-splitting inequalities;
- transversality and homoclinic-tangency calculations;
- validated Evans-function subspace transport;
- orientation and derivative bounds for Poincare return maps;
- eventually, theorem-specific hyperbolicity obligations.

Those are enabling capabilities for open scientific problems. None is solved
by the present scouts. A future theorem still needs its complete model,
compact invariant set, topological or analytic hypotheses, exhaustive receipts,
and independent execution evidence.

## 10. Replay and gates

Fast retained and structural gate:

```bash
bash scripts/ci/cs6_c1_full_source_cover_gate.sh
```

One fresh canonical CAPD rebuild and replay is opt-in:

```bash
CS6_C1_FULL_SOURCE_COVER_REPLAY=1 \
CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config \
bash scripts/ci/cs6_c1_full_source_cover_gate.sh
```

The optional replay is still a bounded local CPU check. It is not an exhaustive
adaptive run or remote attestation.

## 11. Execution boundary

```text
default Sounio compiler path used = false
rebuilt current-source CAPD worker path used = true
fallback path used = false
bounded local CPU probes run = true
exact adaptive local smoke run = true
authorized remote exhaustive run = false
remote attestation present = false
historical execution provenance attested = false
historical source/verifier/runner snapshots retained = true
historical worker binary retained = false
current canonical replay path exercised = true
legacy affine and Liouville routes kept = true
PROJECTIVE_RICCATI_INTEGRATED=false
FULL_SOURCE_CARRIER_PROVED=false
HYPERBOLICITY_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
U250_USED=false
```

## 12. Semantic lane

```text
Semantic-Lane-ID: cs6-c1-full-source-cover-20260731
Owner: codex-root
Concept-IDs: SOUNIO-CS6-C1-SOURCE-DEPENDENCY; SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: dependency and provenance must not disappear when local leaf evidence is aggregated
Transformation: generalize one frozen affine leaf into canonical dyadic leaf, adaptive tree, and replay aggregation machines
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: 28 of 53 selected local leaves certify by the affine route; all 20 selected 13:14 leaves certify; exact adaptive aggregation is executable and fail closed
Claims-Forbidden: sampled centers cover N0; projective flow integrated; uniform 13:14 cover passes; cone proof; hyperbolicity; attractor; external open problem solved; literature priority
Assumptions: frozen CS6 N0 model and chart; CAPD 5.3.0 FILIB; CPU outward rounding; local unattested historical scouts
Write-Set: full-source worker, verifier, runner, aggregator, retained scouts, gate, and this note
Read-Set: prior CS6 C1 dependency/reset/cone evidence and primary literature
Positive-Witness: 28 retained AFFINE certificates; 56/56 leaf mutations; 8/8 tree mutations; fresh one-leaf canonical replay
Negative-Witness: 24 subdivision leaves; root interval-domain failure; no positive projective method; uniform-grid scale extrapolation
Acceptance-Gate: bash scripts/ci/cs6_c1_full_source_cover_gate.sh
Integration-Target: review branch only; no main merge requested
Authoritative-Only-If: exhaustive zero-unresolved tree plus canonical replay audit bundle and separately attested execution
```

## 13. Remaining blocker

```text
Blocker-ID: BLK-20260731-cs6-c1-full-source-dependency-cover
Status: classified
Severity: B3
Class: evidence-gap
Owner: future-cs6-projective-cover
Lane: cs6-c1-full-source-cover
Worktree: /tmp/sounio-cs6-c1-full-source-cover-20260731
Branch: research/cs6-c1-full-source-cover-20260731
Files-Owned: scripts/research/cs6_c1_full_source_cover_*; scripts/research/receipts/cs6_c1_full_source_cover_*; scripts/ci/cs6_c1_full_source_cover_gate.sh; docs/research/cs6_c1_full_source_cover_2026-07-31.md
Files-Read-Only: prior CS6 C0/C1 receipts and local CAPD 5.3.0 sources
Do-Not-Touch: promotion fields or prior certificates without exhaustive evidence
Repro: python3 scripts/research/cs6_c1_full_source_cover_retained_verify.py
Observed: 28 sampled terminals certify, 24 require subdivision, one root computation is unresolved, and no exact full-source tree exists
Expected: a finite exact tree covers N0 with zero unresolved terminals and every certified leaf passes canonical independent-challenge replay
Acceptance-Gate: aggregate the exhaustive nodes.tsv with --require-local-full-source, retain its replay audit bundle, then obtain independent remote attestation before changing FULL_SOURCE_CARRIER_PROVED
Evidence-Level: E2
Evidence: three retained scout directories; retained provenance snapshots; scripts/ci/cs6_c1_full_source_cover_gate.sh
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: completed-dual-xai-zai
Next-Action: implement and benchmark the validated homogeneous Plucker/chart-switch cocycle on the same 53 coordinates before scheduling a larger adaptive cover
```

The remote execution path is independently limited because the Slurm compute
node used in this session could not see either this `/tmp` worktree or the local
CAPD build. Local evidence was captured; remote exhaustive execution was not
simulated.

# CS6 shared-source affine-projective cocycle: a positive bounded pilot

**Date:** 2026-08-01
**Lane:** `cs6-affine-projective-cocycle-20260801`
**Base:** `b4a13fe682da59843748cd67f1e56f5fdf0e2df4`
**Status:** the predeclared four-leaf H-APG pilot passes, with two orientation
rescues and zero affine losses; the result is bounded evidence for a candidate
validated numerical primitive, not a full-source, hyperbolicity, chaos,
open-problem, novelty, or priority claim

## 1. Question, predeclaration, and answer

The preceding H-PG experiment showed that event-wise projective normalization
does not compensate for boxing the tangent rays after the first Poincare event.
It predeclared the next hypothesis, the four leaves, and the pilot success
criterion in commit
`b4a13fe682da59843748cd67f1e56f5fdf0e2df4`:

```text
H-APG-CS6:
If the event state and both tangent rays retain the same source symbols through
validated projective normalization, then the local second-return action can
recover the affine carrier's dependency advantage while retaining a separately
auditable signed exterior channel.
```

The coordinate manifest binds that predeclaration to the committed report SHA,
the parent 53-coordinate run-manifest SHA, and the parent files-index SHA. The
runner and gate also verify each selected input against the parent bundle.

The retained answer is:

```text
coordinates                               4
structurally valid and exactly replayed   4
affine certificates                       1
H-APG certificates                        3
H-APG rescues over affine                  2
affine certificates lost                  0
H-APG narrower than boxed H-PG             4
H-APG narrower than affine                 4
H-APG narrower than shared nonprojective   2
```

The predeclared pilot therefore passes. The sharper result is a decomposition:
preserving the two source variables accounts for most of the improvement over
the boxed H-PG route, while projective factorization is narrower than the same
shared-source nonprojective composition on only two of four leaves. The pilot
supports the integrated machine. It does not show that projective
factorization is universally superior to dependency retention alone.

## 2. Exact implementation boundary

This experiment uses a common two-variable order-2 Taylor carrier:

```math
T(\delta_0,\delta_1) = c_{00} + c_{10}\delta_0 + c_{01}\delta_1
 + c_{20}\delta_0^2 + c_{11}\delta_0\delta_1 + c_{02}\delta_1^2 + R.
```

All coefficients and `R` are outward-rounded intervals. Multiplication retains
total degree at most two. Degree-three and degree-four terms, both polynomial
times remainder terms, and the remainder product are enclosed in the new
interval remainder. For a denominator `A=c(1+z)` whose full range excludes
zero and satisfies `sup |z| < 1`, reciprocal uses the exact identity

```math
\frac{1}{1+z}=1-z+z^2-\frac{z^3}{1+z}.
```

There is no interval-box fallback when the pivot or contraction obligation
fails.

The exact scope frozen in each receipt is:

```text
LOCAL_C2_MAP_REINTEGRATED=true
RAY_ACTION_COMPOSED_IN_TM2=true
RAY_SEED_PROPAGATED_BY_CAPD=false
COMMON_SOURCE_SYMBOLS_PRESERVED=true
APG_LOCAL_INPUT_OBLIGATION=FULL_TM2_STATE_CONTAINMENT
APG_PRIMARY_INTERVAL_INTERSECTED_WITH_CONTROLS=false
CONTINUOUS_RICCATI_INTEGRATED=false
GENERAL_GRASSMANN_PLUCKER_INTEGRATOR=false
```

CAPD rigorously reintegrates the local `C2` Poincare map over the dependent
event-1 state set. It does not propagate the normalized projective ray as an
augmented CAPD state. The local derivative action on the ray is composed in
the explicit TM2 algebra from the CAPD `DP` and `D2P` enclosures. This
distinction forbids describing the worker as a general joint tangent-state ODE
integrator.

The charts are frozen per event and leaf from the prior predeclared choices:

| Leaf | Event 1 rays | Event 2 rays | Expected pivot signs |
|---|---|---|---|
| `U08/S08` | `X, X` | `PLUS, PLUS` | `-,- ; -,-` |
| `U12/S16` | `X, X` | `PLUS, PLUS` | `-,- ; -,-` |
| `U15/S15` | `X, PLUS` | `PLUS, PLUS` | `-,+ ; -,-` |
| `U16/S12` | `X, PLUS` | `PLUS, PLUS` | `-,+ ; -,-` |

This is certified fixed-chart evaluation, not adaptive continuous chart
switching.

## 3. Local map and projective composition

At event 1, the worker constructs the state carrier `X1(delta)` and affine
Jacobian carrier `J1(delta)` from the C2 Poincare return. CAPD stores diagonal
second-order terms as normalized Taylor coefficients, so differentiation of
`DP` uses factor two on a repeated index and factor one on a mixed index. The
exact verifier independently enforces that convention.

Let `G` be the newly reintegrated local return from event 1 to event 2. The
worker builds a mean-value carrier

```math
J_G(\delta)_{ri} \in DG(\bar x)_{ri}
 + \sum_{k=0}^{1}\partial_k DG_{ri}(X_1(D))
   (X_{1k}(\delta)-\bar x_k).
```

The local CAPD input hull must contain the full range of every component of
`X1(delta)`. Overlap is explicitly insufficient. This containment obligation
was added after adversarial review produced a receipt mutation that preserved
overlap but destroyed the domain required by the mean-value argument.
The actual local `C2Rect2Set` is the affine interval set formed by the same
center, basis, source box, and padded residual box, so it is convex and contains
the segment from the center to every represented event-1 state. The CAPD `D2P`
enclosure is taken over that set, not merely over an unrelated overlapping
hull.

For each tangent column `j`, a frozen chart has covector rows `(ell,m)` and
defines

```math
p_{1j}=\ell J_1e_j,
\qquad r_{1j}=\frac{mJ_1e_j}{p_{1j}},
\qquad u_{1j}=C^{-1}(1,r_{1j})^T.
```

The pivot is accepted only if its complete dependent range has the predeclared
strict sign. The local action and event-2 normalization are

```math
z_{2j}=J_Gu_{1j},
\qquad p_{2j}=\ell_2 z_{2j},
\qquad u_{2j}=C_2^{-1}(1,r_{2j})^T.
```

Both the source and normalized third component must contain the section
tangent value `w=0`; the exterior computation uses the physical `x,y` tangent
block. The validity of this restriction comes from the analytic contract of
CAPD's coordinate-section Poincare map: its codomain is `w=0`, so the true
output differential has third component zero. The interval `contains(0)` checks
are runtime consistency audits of that contract, not an independent proof that
an arbitrary interval vector is tangent.

Signed cumulative scales and the factored exterior are

```math
s_j=p_{1j}p_{2j},
\qquad D_{\mathrm{APG}}=s_0s_1\det(u_{20},u_{21}).
```

`D_APG` is the primary result before any comparison or intersection with a
control. Two independently serialized audit routes are also computed:

```math
D_{\mathrm{reconstructed}}=
\det(p_{10}z_{20},p_{11}z_{21}),

D_{\mathrm{shared}}=\det(J_GJ_1e_0,J_GJ_1e_1).
```

The reconstructed, shared, affine, direct C2, boxed H-PG, and Liouville
channels can reject a leaf on disagreement. None can narrow the primary APG
interval or supply its sign.

## 4. Exact verifier and evidence envelope

The leaf verifier parses every binary64 hexadecimal endpoint into exact Python
`Fraction` arithmetic. It independently recomputes:

- canonical leaf geometry, input SHA, challenge, and physical digest;
- the C2 impact-time and return-Hessian identities;
- the event-1 state and Jacobian carriers;
- all TM2 additions, multiplications, discarded monomials, and reciprocals;
- local-input containment for the full dependent event-1 state;
- each frozen chart, signed pivot, rational slope, and section tangency;
- the local mean-value Jacobian and normalized ray action;
- shared, reconstructed, and factored exterior channels;
- the primary width, orientation, rescue, and comparison predicates.

TM2 replay requires both containment and a `2^24` ULP upper bound on each
reported coefficient and remainder. This accommodates the observed C++
outward-rounding/cancellation envelope but rejects coordinated arbitrary
widening. The adversarial suite now contains 102 named mutations, including
the former local-domain-overlap false positive and a coherently re-ranged but
artificially widened reconstructed TM2 channel. All 102 are rejected.

The retained bundle indexes 41 payload files. It binds source, verifier,
runner, coordinates, compiler flags, dependencies, link inputs, runtime
libraries, the SHA of the worker binary before and after execution, stderr,
four receipts, four exact verifications, the 102-mutation audit, and all
summary fields. The default gate is a retained-integrity gate. Fresh CAPD
rebuild and execution are opt-in and cover one rescue leaf; the full runner is
the fresh four-leaf path.

This remains local evidence. The binary itself is not retained, only its
before/after SHA. The Git commit is an integrity anchor, not an independent
signature. CAPD, the compiler, the machine, and the derivation implemented by
both worker and verifier remain in the trusted computing base. DP and
Liouville are consistency audits with shared model assumptions, not a second
validated solver.

## 5. Retained numerical result

All four leaves are adversarially selected witnesses from the prior failed
boxed experiment, not a random sample or a partition.

| Leaf | Affine width | Boxed H-PG width | Shared TM2 width | H-APG width | Outcome |
|---|---:|---:|---:|---:|---|
| `U08/S08` | `6.59637e-8` | `3.29040e-6` | `8.10975e-9` | `4.36629e-9` | narrower, uncertified |
| `U12/S16` | `8.38961e-11` | `4.20569e-8` | `3.34663e-12` | `2.18801e-12` | APG rescue |
| `U15/S15` | `8.25294e-12` | `5.59734e-8` | `8.41254e-13` | `1.29625e-12` | affine and APG retained |
| `U16/S12` | `1.89936e-10` | `3.55056e-7` | `1.88765e-11` | `2.98167e-11` | APG rescue |

The paired width ratios are:

| H-APG divided by | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| boxed H-PG | `2.31583e-5` | `6.80012e-5` | `3.71535e-4` | `1.32698e-3` |
| affine | `0.0260800` | `0.111588` | `0.101580` | `0.157065` |
| shared nonprojective TM2 | `0.538400` | `1.09732` | `1.07815` | `1.57957` |

The minimum certified APG pivot margin is `2.9799747303274783e-4`.

This yields two distinct empirical statements on the frozen pilot:

1. Shared source preservation plus the TM2 local composition decisively repairs
   the boxed H-PG failure on all four leaves.
2. Projective signed-scale factorization adds a width improvement over the same
   shared composition on two leaves and a width cost on two leaves.

Statement 1 is a positive bounded result. Statement 2 is mixed and prevents a
causal claim that projective geometry alone produced the improvement.

## 6. Literature boundary and candidate novelty window

The directed primary-source search has cutoff 2026-08-01. It is not a
systematic exhaustive review, and failure to locate an integrated precedent is
not proof of world priority.

The closest established ingredients are:

- validated high-order Poincare maps with functional dependence and event time
  ([Grote, Makino, and Berz](https://www.wseas.us/e-library/conferences/2005malta/papers/499-095.pdf));
- verified Poincare derivatives, impact-time correction, intermediate-section
  composition, and affine arithmetic
  ([Immler](https://doi.org/10.1007/s10817-017-9448-y));
- rigorous coordinate and set representations for reducing wrapping in
  Poincare maps
  ([Kapela, Wilczak, and Zgliczynski](https://doi.org/10.1016/j.cnsns.2022.106366));
- validated frame and determinant propagation for conjugate-point arguments
  ([Beck and Jaquette](https://doi.org/10.1137/21M1420095));
- dynamic Grassmann chart selection and determinant-factor accumulation in
  nonvalidated spectral shooting
  ([Ledoux, Malham, and Thummler](https://doi.org/10.1090/S0025-5718-10-02323-9));
- angular/exterior-scale separation in Evans computations
  ([Humpherys and Zumbrun](https://doi.org/10.1016/j.physd.2006.07.003)).

The search did not locate one publication integrating, in the same rigorous
Poincare-map enclosure, impact-corrected local derivatives, common uncertainty
symbols for state and tangent, certified event-chart normalization, signed
exterior reconstruction, and a replayable receipt.

The narrow defensible formulation is therefore a **candidate integrative
novelty window**:

> A validated event-wise projective reconditioning primitive for tangent
> cocycles of Poincare maps that preserves common TM2 source dependence and
> reconstructs the signed exterior factor.

This lane does not claim Taylor models, validated Poincare maps, projective
coordinates, Grassmann charts, or exterior algebra as new. It does not claim
“first,” novelty, or priority.

## 7. Scientific reach and next falsifier

The machine targets a recurring proof bottleneck: wrapping in tangent cocycles
used for cone, orientation, hyperbolicity, and invariant-set arguments. A
validated primitive that preserves source dependence while reconditioning
directions at Poincare events could make otherwise unresolved proof obligations
computable. This pilot does not itself solve an open scientific problem or
establish a new dynamical property.

The next experiment must separate dependency, order, and geometry more fully:

1. Run boxed H-PG, shared TM2 without projective factorization, and H-APG on all
   53 retained coordinates.
2. Require zero affine losses, at least one rescue, exact replay, and all
   mutations rejected.
3. Compare TM2 with order-3 Taylor models using identical frozen charts.
4. Repeat on a second dynamical system with an independently specified
   Poincare section.
5. Reintegrate at least a witness subset with a second validated solver or
   independently controlled environment.
6. Add signed execution attestation before any promotion claim.

The two AMD U250 boards are neither installed nor used. Their eventual honest
role is parallel population and nonauthoritative margin screening. They do not
repair dependency, establish outward rounding, or replace CPU verification.

## 8. Replay and execution boundary

Retained integrity, exact 4/4 replay from stored receipts, and coordinated
mutation attacks:

```bash
bash scripts/ci/cs6_affine_projective_cocycle_gate.sh
```

The same gate plus a fresh CAPD rebuild and one rescue-leaf execution:

```bash
CS6_AFFINE_PROJECTIVE_COCYCLE_REPLAY=1 \
CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config \
bash scripts/ci/cs6_affine_projective_cocycle_gate.sh
```

The fresh four-leaf experiment is reproduced by
`scripts/research/cs6_affine_projective_cocycle_run.py` with the retained root
challenge and CAPD 5.3.0 config.

```text
default Sounio compiler path used = false
rebuilt current-source CAPD worker path used = true
fallback path used = false
bounded local CPU run = true
authorized remote run = false
execution provenance attested = false
U250 used = false
legacy flattened C1, affine C2, fixed-chart, boxed H-PG, and Liouville paths kept = true
FULL_SOURCE_CARRIER_PROVED=false
HYPERBOLICITY_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
NOVELTY_OR_PRIORITY_CLAIM=false
PROMOTION_ELIGIBLE=false
```

No blocker prevents retaining this bounded pilot. Promotion remains an
investigation item rather than a typed blocker because its full 53-coordinate,
second-system, second-solver, and attestation gates have not yet been created.

## 9. LLM-offload review

The mandatory M1 review used xAI/Grok 4.3 and Z.AI/GLM-5.2. Both independently
approved the reciprocal identity, TM2 multiplication envelope, mean-value
Jacobian, projective reconstruction, numerical ratios, and bounded claim on the
full report. A focused implementation review also approved the algebra and
enclosures. Its follow-up marked one boundary `TIGHTENABLE`: an interval third
component that merely contains zero is not by itself a tangency proof. Section
3 now states explicitly that the `x,y` reduction relies on the analytic CAPD
coordinate-section contract and treats `contains(0)` only as a consistency
audit. No implementation error or promotion overreach was found. Provider,
task, target, raw-output locations, and outcome are recorded in
`.claude/llm_offload_log.md`.

## 10. Semantic lane

```text
Semantic-Lane-ID: cs6-affine-projective-cocycle-20260801
Owner: codex-root
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-CS6-C1-SOURCE-DEPENDENCY
Intent-Preserved: machine-level scientific research must retain uncertainty, dependency, provenance, controls, and falsifiers rather than promote a suggestive witness into a theorem
Transformation: replace boxed event-1 tangent rays with a common two-source TM2 composition through a locally reintegrated C2 return, frozen projective normalization, and signed exterior reconstruction
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the predeclared four-leaf H-APG pilot passes; H-APG is narrower than boxed and affine on 4/4, certifies 3/4, rescues 2/4, loses zero affine certificates, and is narrower than shared nonprojective TM2 on 2/4
Claims-Forbidden: adaptive continuous chart switching; CAPD-propagated projective ray seed; general Grassmann or Plucker integrator; full-source carrier; partition coverage; hyperbolicity; chaotic attractor; open problem solved; novelty or priority
Assumptions: frozen autonomous CS6 N0 model and Poincare section; CAPD 5.3.0 FILIB outward rounding; two normalized source variables; degree-2 TM2; local unattested CPU execution; four adversarial paired witnesses, not a random sample or partition
Write-Set: scripts/research/cs6_affine_projective_cocycle_*; scripts/research/receipts/cs6_affine_projective_cocycle_*; scripts/ci/cs6_affine_projective_cocycle_gate.sh; docs/research/cs6_affine_projective_cocycle_2026-08-01.md; .claude/llm_offload_log.md
Read-Set: predeclared H-PG report and retained 53-coordinate evidence; CAPD 5.3.0 headers and libraries; cited primary literature
Positive-Witness: exact 4/4 replay; 102/102 mutations rejected; 4/4 narrower than boxed and affine; 3/4 certificates; 2 rescues; zero affine losses
Negative-Witness: only 2/4 narrower than the shared nonprojective composition; U08 remains uncertified; no full-corpus, second-system, second-solver, remote-attested, or U250 result
Acceptance-Gate: bash scripts/ci/cs6_affine_projective_cocycle_gate.sh and the opt-in CS6_AFFINE_PROJECTIVE_COCYCLE_REPLAY=1 form
Integration-Target: review branch only; no main merge requested
Authoritative-Only-If: a future 53-coordinate zero-loss/positive-rescue gate, second-system evidence, second validated implementation, exact replay, mutation closure, and independent execution attestation all pass
```

```text
Semantic-Outcome: positive bounded H-APG pilot retained without promotion
Concept-Status-Before: SOUNIO-CS6-C1-SOURCE-DEPENDENCY hypothesis pending a dependency-preserving projective cocycle
Concept-Status-After: hypothesis gains a positive four-witness machine result but remains a hypothesis pending full-corpus and independent evidence
Distinctions-Added: dependency-retention gain versus projective-factorization gain; local C2 reintegration versus CAPD ray-seed propagation; retained integrity versus fresh execution
Distinctions-Preserved: numerical enclosure versus dynamical theorem; computational provenance versus physical causality; literature gap versus novelty; pilot witness versus source coverage
Distinctions-Erased: none
Evidence-Run: fresh four-leaf CAPD runner; exact leaf verifier; 102-mutation audit; retained-integrity gate; opt-in one-leaf fresh replay
Fallback-Path: none
Legacy-Kept: yes, all baseline and audit routes remain live
Conflicting-Lanes: none in the declared write set
Next-Semantic-Interface: full 53-coordinate dependency/order/geometry ablation with independent-system evidence
```

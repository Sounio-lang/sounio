# CS6 cone-condition scout

**Date:** 2026-07-30
**Lane:** `cs6-cone-scout-20260730`
**Base:** `da3ce70c2b2c0a889c4055acf249e190da855159`
**Status:** proof-shaped numerical cone candidate found; rigorous C1 enclosure,
invertibility, hyperbolicity, and attractor claims not proved

## 1. Result boundary

This wave tested whether the three candidate Fibonacci covering edges for the
common map `F=P^6` admit one compatible family of quadratic cones:

```text
N0 -> N0
N0 -> N1
N1 -> N0
```

It found

```text
Q0 = diag(1,                  -2.3023784599059653)
Q1 = diag(0.06526711140171336, -2.3023784599059653)
```

for which every sampled point matrix passes the cone-matrix test. A stronger
non-rigorous diagnostic also passes: the complete entrywise box containing all
sampled derivative matrices passes an interval-expression Sylvester test on
all three edges, both on the discovery grid and on an untouched interleaved
holdout grid.

The arithmetic for that sampled box is ordinary `long double`, not directed
outward rounding, and the box does not enclose unsampled derivatives. The
result is therefore a candidate for a CAPD C1 proof, not a proof:

```text
numerical_cone_candidate_found = true
nonrigorous_sampled_entrywise_hull_pd_sufficient = true
numerical_hyperbolicity_candidate_found = false
pairwise_chord_cone_condition_proved = false
tangent_cone_condition_proved = false
uniform_hyperbolicity_proved = false
chaotic_attractor_proved = false
```

The replayable numerical receipt is
`scripts/research/cs6_cone_scout_receipt_v1.json`.

## 2. Why this criterion

Let `B=[e_u e_s]` be the frozen frame and let
`R_i=diag(r_ui,r_si)`. In normalized source and target coordinates, the
derivative for edge `i->j` is

```math
A_ij(x) = R_j^{-1} B^{-1} D(P^6)(c_i(x)) B R_i.
```

For node forms `Q_i=diag(a_i,-b_i)`, the tested matrix is

```math
M_ij(A) = A^T Q_j A - Q_i.
```

Writing `A=[[p,q],[r,s]]`, its independent entries are

```math
m00 = a_j p^2 - b_j r^2 - a_i,
m01 = a_j p q - b_j r s,
m11 = a_j q^2 - b_j s^2 + b_i.
```

In two dimensions, `m00>0` and `det(M)>0` are sufficient and necessary for
positive definiteness. The implementation evaluates these expressions over
the sampled entrywise derivative box and optimizes the lower Sylvester score
before using the pointwise margin as a tie-breaker.

This follows the computer-verifiable condition in Lemma 8 of Zgliczynski's
covering-and-cones construction: if an outward-rounded interval enclosure of
the derivative over the full source h-set makes the quadratic difference
positive definite for every enclosed matrix, then the pairwise chord cone
inequality follows. The crucial word is **full**; sampled derivatives are not
that enclosure.

Primary references:

- [Zgliczynski, covering relations, cone conditions and the stable manifold theorem](https://doi.org/10.1016/j.jde.2008.12.019)
- [Author manuscript containing the interval derivative criterion](https://ww2.ii.uj.edu.pl/~zgliczyn/papers/invman/cncv.pdf)
- [CAPD Poincare map derivative interface](https://capd.sourceforge.net/capdDynSys/docs/html/a05237.html)
- [Plesa and Sprott CS6 system](https://doi.org/10.1063/5.0323112)

## 3. Frozen geometry

The scout preserves the geometry from the Fibonacci covering wave:

```text
origin = (15.186446520640786, 10.908543194765466)
e_u    = (-0.67430316214199759, -0.73845463335624273)
e_s    = (-0.94170446778164518,  0.33644122125579123)

N0: center (0, 0),                    radii (0.0040, 0.3)
N1: center (0.019771776972779206, 0), radii (0.0015, 0.3)
```

No h-set, section, return count, covering degree, or scientific claim was
changed in this wave.

## 4. Numerical protocol

The discovery pass uses a `17 x 17` endpoint grid on each source set. It
contains 578 unique source points and 867 edge derivative records. The form
weights are fitted only on this pass.

The holdout uses a disjoint `16 x 16` cell-midpoint grid. It contains 512
unique source points and 768 edge records. Holdout values do not participate
in the parameter search.

The derivative is assembled from six CAPD `DPoincareMap` steps in the order
`D P_k * ... * D P_1`. Fifteen finite-difference probes cover the centers and
four near-corners of the source sets. All derivative entries, composed
matrices, return times, margins, and finite-difference values fail closed on a
non-finite value. The largest relative finite-difference discrepancy was
`1.4021812633213671e-6`.

### Discovery

| Edge | Point margin | Sample-hull `m00` lower | Sample-hull `det(M)` lower | Min sampled forward expansion |
|---|---:|---:|---:|---:|
| `N0->N0` | `0.9938525163` | `6.4637891103` | `3.2232187018` | `2.2468890040` |
| `N0->N1` | `0.9664814781` | `2.4641063491` | `3.1618805880` | `5.9917040106` |
| `N1->N0` | `0.5927619437` | `1.4380390188` | `3.1594936346` | `1.1726838552` |

### Untuned holdout

| Edge | Point margin | Sample-hull `det(M)` lower |
|---|---:|---:|
| `N0->N0` | `0.9949456111` | `7.5421577356` |
| `N0->N1` | `0.9828501251` | `4.5784855661` |
| `N1->N0` | `0.6036631820` | `3.2274192458` |

These order-one margins make the C1 interval experiment worth running. They
do not license rounding away the difference between an empirical hull and a
validated derivative enclosure.

## 5. The stable direction remains unresolved in double

The selftest deliberately includes the singular matrix `diag(2,0)`. It makes
`A^T Q A-Q` positive definite for `Q=diag(1,-1)` while `det(A)=0`. Thus the
cone-matrix inequality alone does not establish invertibility.

The inherited rigorous periodic-orbit certificate gives, only at the proved
fixed point,

```text
det(DP^6) in
[1.0048669882187863e-34, 1.0048693202578456e-34].
```

Subtracting `ad-bc` from already rounded double derivative entries cannot
resolve that scale. The observed `10^-18` to `10^-20` values are therefore
labelled cancellation-sensitive residuals, not determinant estimates.
Backward expansion is not computed from those residuals, and the numerical
hyperbolicity candidate remains false.

## 6. Next rigorous machine

The smallest high-value next implementation is a CPU CAPD C1 cone gate:

1. Propagate a C1 interval set through exactly six `MinusPlus` returns for
   each complete source h-set.
2. Transform the outward-rounded derivative enclosure into the frozen
   normalized charts for each target.
3. Evaluate `M=A^T Q_j A-Q_i` with FILIB/CAPD intervals using the frozen forms
   above and prove strict Sylvester lower bounds for every enclosed matrix.
4. Certify invertibility separately. Prefer the augmented-divergence
   Liouville identity with the section-normal velocity ratio retained away
   from the fixed point; do not infer it from double `ad-bc`.
5. Combine the C1 certificate with the still-unpromoted exhaustive C0
   Fibonacci covering certificate. Only that combined evidence may promote a
   uniformly hyperbolic symbolic invariant-set claim.

Even that combined result would not prove that the set is an attractor. A
trapping-region or basin argument remains a separate global obligation.

The two AMD U250 boards are neither required nor part of the trusted
computing base for this gate. They may later schedule independent tiles, but
the proof predicates and outward-rounded CPU replay remain authoritative.

## 7. Execution paths

```text
default Sounio interval path used = false
rebuilt current-source CAPD path used = true
fallback path used = false
legacy numerical reconnaissance kept = true
remote exhaustive C0 replay run = false
rigorous C1 cone replay run = false
docs registry gate = false (pre-existing base-commit drift)
```

The primary checkout remained untouched. Work was isolated in
`/tmp/sounio-cs6-cone-scout-20260730`.

## 8. Semantic lane declaration

```text
Semantic-Lane-ID: cs6-cone-scout-20260730
Owner: codex-1
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: machine evidence must remain weaker than the scientific claim until its proof obligations are executable and green
Transformation: add a fail-closed numerical derivative-and-cone scout plus an anti-promotion receipt
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a proof-shaped numerical cone candidate exists for the frozen CS6 Fibonacci graph
Claims-Forbidden: rigorous cone condition; uniform hyperbolicity; positive entropy promotion; chaotic attractor; first-proof priority
Assumptions: CAPD double trajectories are reconnaissance; sampled hulls omit unsampled derivatives; long-double hull arithmetic is not outward-rounded
Write-Set: scripts/research/cs6_cone_scout.cpp; scripts/research/cs6_cone_scout_receipt_v1.json; scripts/ci/cs6_cone_scout_gate.sh; docs/research/cs6_cone_scout_2026-07-30.md; .claude/llm_offload_log.md
Read-Set: prior CS6 proof-machine and Fibonacci-covering artifacts; CAPD documentation; Zgliczynski cone-condition paper
Positive-Witness: discovery and disjoint holdout sampled hulls have strict positive numerical Sylvester lower bounds on all three edges
Negative-Witness: singular diag(2,0) passes the cone matrix but fails invertibility; identity and swapped-axis controls fail strict cone positivity
Acceptance-Gate: bash scripts/ci/cs6_cone_scout_gate.sh
Integration-Target: review branch only; no main merge requested
Authoritative-Only-If: a future CAPD C1 outward-rounded full-source certificate and separate invertibility certificate pass

Semantic-Outcome: numerical candidate surface added without promoting a theorem
Concept-Status-Before: C0 Fibonacci geometry was proof-ready; no cone form was frozen
Concept-Status-After: cone forms and a C1 acceptance predicate are frozen; rigorous C1 evidence remains absent
Distinctions-Added: sampled point matrix vs sampled entrywise hull vs full interval derivative enclosure
Distinctions-Preserved: computational provenance != physical causality; formal model != empirical claim; numerical candidate != theorem
Distinctions-Erased: none
Evidence-Run: selftests; 17x17 discovery; disjoint 16x16 holdout; deterministic replay; mandatory math offload
Fallback-Path: none
Legacy-Kept: yes; prior C0 scout, interval covering runner, and anti-promotion certificate are unchanged
Conflicting-Lanes: none reported by bin/sounio-coord brief
Next-Semantic-Interface: CAPD C1 interval derivative enclosure and Liouville invertibility certificate
```

## 9. Open blocker records

```text
Blocker-ID: BLK-20260730-cs6-c1-interval-cone
Status: classified
Severity: B3
Class: evidence-gap
Owner: codex-1
Lane: cs6-rigorous-c1-cone
Worktree: /tmp/sounio-cs6-cone-scout-20260730
Branch: research/cs6-cone-scout-20260730
Files-Owned: none after this lane is released
Files-Read-Only: scripts/research/cs6_cone_scout.cpp; scripts/research/cs6_cone_scout_receipt_v1.json
Do-Not-Touch: prior CS6 C0 certificate or promoted claim fields without a complete replay
Repro: bash scripts/ci/cs6_cone_scout_gate.sh
Observed: numerical candidate passes while every rigorous cone and hyperbolicity field remains false
Expected: outward-rounded full-source C1 derivative and invertibility certificates pass all three edges
Acceptance-Gate: future CAPD C1 gate must emit pairwise_chord_cone_condition_proved=true and preserve chaotic_attractor_proved=false
Evidence-Level: E3
Evidence: scripts/research/cs6_cone_scout_receipt_v1.json
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: implement the CAPD C1 full-source derivative enclosure and Liouville determinant path
```

```text
Blocker-ID: BLK-20260728-cs6-cluster-ops-auth-bridge
Status: classified
Severity: B3
Class: platform-resource
Owner: Cluster Ops
Lane: cs6-fibonacci-full-replay
Worktree: /workspace/sounio
Branch: research/self-falsifying-compilation-line-20260726
Files-Owned: none
Files-Read-Only: scripts/research/cs6_capd_fibonacci_covering_run.sh; scripts/ci/cs6_fibonacci_scout_gate.sh
Do-Not-Touch: auth-bridge replicas, tokens, contexts, or manual cluster YAML from the workspace
Repro: beagle hpc profiles
Observed: HTTP 401; beagle/auth-bridge has zero replicas and no endpoints
Expected: an authorised HPC profile is returned and the 12-shard C0 replay can be submitted
Acceptance-Gate: beagle hpc profiles exits zero, then the frozen 12-shard replay and aggregate gate pass
Evidence-Level: E4
Evidence: current Cluster Ops query plus docs/research/cs6_fibonacci_scout_2026-07-29.md
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: Cluster Ops restores the auth-bridge service; replay owner runs the already-published branch
```

```text
Blocker-ID: BLK-20260730-docs-registry-baseline-drift
Status: classified
Severity: B2
Class: doc-claim
Owner: docs-governance/A2
Lane: repository-docs-registry
Worktree: /tmp/sounio-cs6-cone-scout-20260730
Branch: research/cs6-cone-scout-20260730
Files-Owned: none in this scientific lane
Files-Read-Only: docs/governance/topic-registry.v1.json; docs/governance/DOCS_ACCEPTANCE_REPORT.md; docs/papers/main/168-theorem-preprint.md; docs/papers/oopsla2027/outline.md
Do-Not-Touch: unrelated paper metadata or generated governance artifacts from the CS6 lane
Repro: bash scripts/dev/check_docs_registry.sh
Observed: the clean base commit and this lane both report the same stale registry, acceptance report, and pre-existing paper metadata mismatches
Expected: the clean base passes the docs registry and its selftest
Acceptance-Gate: bash scripts/dev/check_docs_registry.sh
Evidence-Level: E2
Evidence: clean detached base replay captured in /tmp/cs6_docs_baseline_check.out
Fallback-Path: local branch commit may use --no-verify; merge and CI remain blocked
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: a separate docs-governance lane runs the metadata sync, reviews its broad diff, and restores the gate
```

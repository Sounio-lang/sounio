# CS6 section-resident C0/C1 carrier: bounded one-return result

**Date:** 2026-07-31
**Lane:** `cs6-section-resident-return-20260731`
**Branch:** `research/cs6-section-resident-return-20260731`
**Base:** `92eeb25b28cccb4ad4751bd7421fdcbbe4179872`
**Status:** one frozen N0 tile has a replayable event-resident carrier; no
multi-return, full-source, hyperbolicity, or attractor claim is promoted

## 1. Question and bounded answer

The synchronized C1-reset experiment exposed a representation problem. CAPD's
public `PoincareMap` returns an event image and derivative while advancing the
mutable structured set beyond the section. That makes a later C0/C1 reset easy
to get subtly wrong: an event derivative can be attached to a post-section
state.

This wave asks the narrower machine-level question:

> Can the exact one-step section-crossing path be exposed, materialized as a
> structured C0/C1 object at the event time, and distinguished mechanically
> from CAPD's post-section continuation state?

For one frozen N0 tile, the answer is **yes**. The adapter follows the same CAPD
fast path as a fresh public `IPoincareMap` run. It retains the event image,
source-seeded flow tangent, return time, Poincare tangent, and the strictly later
post-section state. From the event data it materializes two different objects:

- a terminal event carrier whose C1 hull records the incoming Poincare
  derivative;
- a continuation carrier at the same C0 event set and time, with the exact
  local tangent seed `diag(1,1,0)` and the incoming derivative retained only as
  metadata.

The exact verifier reconstructs every raw C0 and C1 doubleton component and
rejects event/post-section mixing. The result is bounded to return one and one
tile. It is an interface and evidence result, not a six-return solution.

## 2. CAPD control point

`SectionResidentMap` is a narrow subclass of CAPD 5.3.0
`IPoincareMap`. Its `one_return` method exposes the protected sequence already
used by the public operator:

```cpp
sectionDerivativesEnclosure.init(&time, &flowTangent, nullptr, nullptr);
integrateUntilSectionCrossing(before, after, 1);
crossSectionInOneStep(before, after, localTime, eventImage);
sectionDerivativesEnclosure.computeOneStepSectionEnclosure(
    before, m_solver, eventImage, localTime);
eventTangent = computeDP(eventImage, flowTangent, time);
```

The relevant local CAPD sources are:

- `capd/poincare/PoincareMap_templateOperator.h`, which constructs the
  post-section set and dispatches the crossing path;
- `capd/poincare/PoincareMap_templateMembers.h`, which integrates to the
  crossing, performs the one-step Newton enclosure, and computes section
  derivatives;
- `capd/poincare/SectionDerivativesEnclosure.h`, which owns the event time and
source-seeded flow-tangent enclosures used by `computeDP`.

The adapter does not change CAPD sources. `before` is the last structured set
before the crossing, `eventImage` is the section intersection enclosure, and
`after` is CAPD's continuation set after the crossing. The receipt requires the
post-section current time to lie strictly after the event-time enclosure and
its section coordinate to be strictly positive for the frozen `MinusPlus`
crossing.

Context7 resolution was attempted twice for rigorous CAPD 5.3.0 and returned
unrelated homonyms, so no valid `/org/project` documentation ID was available.
The checked local CAPD 5.3.0 headers and executable behavior are the API
authority for this bounded wave.

## 3. Carrier contract

Let `X` be the frozen source tile and let `P(X)` be its first return to the
coordinate section

```math
\Sigma = \{(x,y,w): w=0\}.
```

The one-step Newton path returns a box `B_event` containing `P(X)`. Since the
true event lies in `Sigma`, replacing only the normal component by the exact
zero interval produces

```math
B_\Sigma = \pi_{xy}(B_{event}) \times \{0\},
\qquad P(X) \subseteq B_\Sigma \subseteq \Sigma.
```

This projection deliberately loses C0 dependency information, so it is a
sound but potentially wider box carrier. The verifier reconstructs its raw
representation as

```math
x + C r_0 + B r
```

using exact rational interval arithmetic and checks that the reported hull is
an enclosure of that reconstruction within an explicit 4096-binary64-ULP
rounding budget. This conservative budget is a verifier allowance, not a
measured mathematical width claim.

Let

```math
Q_0 = [r_u v_u,\ r_s v_s,\ 0]
```

be the frozen source tangent seed. It uses the global chart radii, not the
microscopic C0 tile widths. The receipt serializes `Q0` explicitly and the
verifier reconstructs it from the frozen frame and radii, including its exact
zero dummy normal column. CAPD's retained `FLOW_TANGENT` is therefore
`Dphi Q0`, not the ambient `Dphi`.

For the terminal C1 evidence carrier, the verifier independently recomputes
the coordinate-section Poincare tangent

```math
J_{in} =
\left(I - \frac{f(P(X))e_w^T}{e_w^T f(P(X))}\right)
D\varphi\,Q_0
```

from the retained event image and flow tangent `Dphi Q0`. The normal output row is
structurally zero because the return map lands in `Sigma`. The normal input
column is structurally zero because serialized `Q0` has only the two tangent
coordinates and an exact zero dummy normal column. The
worker narrows either row or column to zero only after every raw CAPD entry
contains zero; otherwise it aborts. The verifier recomputes the projection and
requires the reported public and adapter tangents to contain the independently
recomputed enclosure before checking the exact tangent narrowing.

The raw C1 carrier is reconstructed as

```math
D + C_{jac} R_0 + B_{jac} R.
```

The continuation carrier does **not** reinject the incoming `DP`. It uses

```math
Q_\Sigma = \operatorname{diag}(1,1,0)
```

as a local tangent seed, while the incoming projected derivative is emitted in
a separate `INCOMING_DP` field. This separation prevents a cumulative prefix
from being mistaken for a local derivative on the next return.

## 4. Exact verification and adversarial gate

The receipt grammar has exactly 40 nonempty ASCII/LF lines: 32 fixed headers
and eight ordered records. Every interval is serialized as a canonical
hexadecimal binary64 enclosure expanded outward by exactly one ULP. The Python
verifier removes that serialization ULP and performs its algebra with exact
`Fraction` endpoints.

It independently checks:

- source, input, run-challenge, and whole-receipt SHA-256 bindings;
- exact grammar, record order, token order, cardinality, and finite intervals;
- the frozen tile from `20000,15000 / 40000,30000` and its explicit
  global-radius source tangent seed `Q0`;
- bit-identical public and adapter fast-path enclosures;
- containment of the Poincare tangent recomputed from retained `Dphi Q0`,
  state, and vector field;
- all raw C0 and C1 carrier reconstructions;
- exact event residence on `w=0`, exact continuation seed, and separate
  incoming derivative metadata;
- event-time identity and a strictly later positive post-section witness;
- normal velocities, two-dimensional derivative determinants, Liouville
  downstream algebra, and joint rather than pairwise intersections.

The retained `ELL -> EXP_ELL` interval remains inside the CAPD worker's trusted
computing base. The verifier checks positivity and all downstream Liouville
algebra but reports `EXP_ELL_RECOMPUTED=false`; it does not claim an independent
directed-rounding transcendental implementation.

`scripts/ci/cs6_section_resident_gate.sh` verifies all provenance bindings,
rejects 32 independent receipt or argument mutations, and then performs a
mandatory fresh compile and one-return replay with a new challenge. The fresh
bundle index is checked entry by entry and is forbidden from hashing itself.

## 5. Bounded result

The canonical run used CAPD 5.3.0, FILIB, order 8, and the frozen N0 tile
`20000,15000 / 40000,30000`. The direct public path and protected adapter path
produced identical serialized event state, flow tangent `Dphi Q0`, return time, and
Poincare derivative.

| Quantity | Rigorous enclosure or result |
|---|---|
| Event time | `[2.406869001202317, 2.4068690626791223]` |
| Event `x` | `[11.506849746428518, 11.506850837379]` |
| Event `y` | `[4.730848213628668, 4.7308485006913905]` |
| Event normal velocity | `[32.109693854283286, 32.109706266595445]` |
| Event C1 determinant in the `Q0` source basis | `[-2.463203714070897e-6, -2.4562057232141765e-6]` |
| Liouville determinant | `[-2.4597095230602197e-6, -2.4596997214446695e-6]` |
| Post-section time | `[2.4083948800738586, 2.40839488007386]` |
| Post-section `w` | `[0.04892397262826454, 0.04892594730271217]` |

The C1 and Liouville determinant enclosures are comparable because both include
the oriented source frame and the global `radius_u/radius_s` normalization.
They intersect. State, time, derivative,
normal-velocity, and determinant families each have a nonempty joint
intersection. The exact physical-chain digest is:

```text
54d4abb953c0d1e5aadba9ceb960ec788eaca3f0e61b6326cf0ba9060f3c5c00
```

The canonical raw receipt SHA-256 is:

```text
cd6fecac6ddbda3061e352eafde19f235ebc0dd1fafac496abc7392303552198
```

These numbers prove only the bounded enclosure relations encoded by the gate.
They do not prove that the new continuation carrier survives a second return,
that six returns cover the full source, that a cone field is invariant, or that
a chaotic attractor exists.

## 6. Provenance boundary

The canonical runner snapshots the probe, verifier, runner, and frozen input.
It records content hashes for those files plus the executable, compile command,
effective CAPD flags, preprocessor state, compiler options, compile
dependencies, link inputs, and runtime libraries. It
checks dependency hashes before and after execution and binds the receipt to a
fresh run challenge. The repository retains the textual evidence needed by the
gate under `scripts/research/receipts/`. The compiled source is represented by
the relocatable dependency label `BUNDLE/probe-source.cpp`; the gate maps that
label to the hash-bound repository source and verifies every retained
dependency, link-input, and runtime-library entry against the current replay
environment.

This is still a local bounded CPU run without remote attestation. Compiler
dependency discovery is not claimed complete, and the binary/toolchain are not
fully vendored. Independent replay is mandatory and promotion remains false.

```text
default Sounio compiler path used = false
rebuilt current-source CAPD path used = true
fallback path used = false
bounded local CAPD CPU probe run = true
mandatory fresh local replay gate = true
authorized Foundry/Slurm replay run = false
remote attestation present = false
execution provenance attested = false
dependency content hashes complete = false
independent replay required = true
promotion eligible = false
legacy C0, C1-reset, and full-source research paths kept = true
```

The two AMD U250 boards were not installed and were not used. They remain
useful future schedulers for independent accepted leaves or frozen chart
candidates, but CPU outward-rounded CAPD and exact aggregation remain the
authoritative proof path.

## 7. Next machine-level opening

The immediate experiment is now concrete: start from the continuation carrier
`(B_Sigma, Q_Sigma, t_event)`, propagate exactly one more return through the
same section-resident path, and compare:

1. the genuinely local second-return factor `J2_local`;
2. the cumulative product `J2_local J1_in` in that order;
3. a fresh public two-return cumulative derivative;
4. the two-return Liouville determinant and C0 transversality witness.

Only after that identity closes should the machine attempt six returns and an
adaptive source cover. If the box-valued C0 carrier wraps too rapidly, the next
representation should retain an affine section chart rather than silently
reusing the post-section set.

## 8. Semantic lane

```text
Semantic-Lane-ID: cs6-section-resident-return-20260731
Owner: codex-1
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: machine evidence remains strictly weaker than the scientific claim
Transformation: expose and materialize one rigorous section-resident C0/C1 return carrier
Types-Changed: none in the Sounio compiler
Effects-Changed: none
IR-Changed: none
Claims-Introduced: one frozen N0 tile has a replayable event-resident C0/C1 carrier consistent with the public CAPD path
Claims-Forbidden: second-return propagation; six-return cover; full-source carrier; cone proof; uniform hyperbolicity; chaotic attractor; first-proof priority
Assumptions: CAPD 5.3.0 FILIB; frozen CS6 map and frame; coordinate section w=0; two-dimensional tangent source parameterization
Positive-Witness: canonical receipt, exact verifier, 32 negative mutations, and mandatory fresh replay
Negative-Witness: none promoted; untested continuation is recorded as an evidence gap
Acceptance-Gate: bash scripts/ci/cs6_section_resident_gate.sh
Integration-Target: review branch only; no main merge requested
Fallback-Path: none
Legacy-Kept: yes
Next-Semantic-Interface: two consecutive section-resident returns with a local-factor identity
```

## 9. Next investigation, not a blocker

```text
Investigation-ID: INV-20260731-cs6-section-resident-continuation
Current-Evidence: E3 gate-bound one-tile, one-return carrier
Why-Not-Blocker: no required gate for this bounded lane is failing and no external resource is needed for the next local experiment
Observed-Boundary: the event carrier and continuation seed are certified for one frozen tile and one return only
Next-Hypothesis: J2_local J1_in overlaps a fresh cumulative two-return tangent and the two-return Liouville determinant
Future-Acceptance: two-return identity first, then six transverse returns over an exact adaptive source cover
Do-Not-Touch: prior receipts, proof-promotion fields, or full-source claims without new exhaustive evidence
Evidence: scripts/research/cs6_section_resident_receipt_v1.txt; scripts/research/cs6_section_resident_provenance_v1.txt; scripts/research/receipts/; scripts/ci/cs6_section_resident_gate.sh
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: propagate the local tangent continuation carrier through a second event-resident return and verify local-factor composition against a fresh cumulative two-return baseline
```

The prior broad section-resident interface gap now has an E3 one-return
witness, but its six-return/full-source acceptance condition remains open. No
hardware absence is being used as an explanation for this remaining
mathematical work. Under `.claude/PARALLEL_BLOCKER_CONTRACT.md`, this is an
investigation item rather than a B3 blocker: nothing external currently
prevents the next experiment.

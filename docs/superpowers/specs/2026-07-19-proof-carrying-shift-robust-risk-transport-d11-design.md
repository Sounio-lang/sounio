<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-19-proof-carrying-shift-robust-risk-transport-d11-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-19-proof-carrying-shift-robust-risk-transport-d11-design
-->

# D11 Proof-Carrying Shift-Robust Risk Transport Design

## Semantic Lane Declaration

```text
Semantic-Lane-ID: PSYCHIATRIC-D11-SHIFT-ROBUST-RISK-TRANSPORT-20260719
Owner: codex-3
Concept-IDs: SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT
Intent-Preserved: target evidence may preserve or reduce an existing synthetic canary warrant but may never widen scope or authority
Transformation: add a bounded source-to-target risk transport and one-way attenuation layer above unchanged D10
Types-Changed: new D11-only public observations/refusals, private validated-evidence tokens, private warrant states, and reserved authorities
Effects-Changed: none outside effects declared by new D11 functions
IR-Changed: none
Claims-Introduced: exact frozen transport identities, impossibility collisions, and nominal non-expansive authority rank
Claims-Forbidden: real transportability, external validation, production permission, clinical authority, affine consumption, live revocation, general conformal guarantees, novelty
Assumptions: frozen finite fixtures, exact integer arithmetic, canonical Madaros, private constructor enforcement
Write-Set: D11 kernel, ontology, witnesses, oracle, negatives, gate, concept/docs/registry bindings, offload log, generated governance metadata
Read-Set: D10-D0 kernels and gates, semantic and blocker contracts, current primary literature and official regulatory sources
Positive-Witness: standalone exact runtime plus imported private-token flow check
Negative-Witness: clinical and ontology compile-fail matrices plus private-constructor refusals
Acceptance-Gate: scripts/ci/proof_carrying_shift_robust_risk_transport_gate.sh
Integration-Target: codex/psychiatric-d10-deployment-validity-20260719
Authoritative-Only-If: canonical Madaros, independent Fraction oracle agreement, exact negatives, dual ontology paths, recursive D10-D0 green, mandatory hostile reviews
```

## Private State Machine

The maximum D11 producer returns `D11TargetCanaryContinuationToken`. It carries
the same authority rank as the source synthetic canary and a narrower target
scope. It is issued only after private frozen-fixture tokens for covariate,
label, concept, subgroup, local-calibration, and shift-robust conformal-risk
evidence have been combined.

Every positive receipt derives from one private exact labeled law with source
counts `(3,3,3,3)`, target counts `(6,2,2,2)`, denominator `12`, loss
`(1,0,0,0)`, and deterministic fingerprint `92352734845403155`. Its coarse
deterministic `X=Y` projection makes both declared conditional invariances
compatible. A separate perfect label probe establishes label-shift
identifiability; the evaluated loss is not reused as that probe. The private
law also freezes the projection map, probe confusion matrix, diagnostic and
active score vectors, and positive subgroup allocation. Every certifier checks
its observation payload against those fields before minting a token. The private
binder rejects any mismatch in fixture run, exact-law token, fingerprint,
model, source/target population, evidence window, loss, label probe,
calibrator, subgroup plan, or scope IDs.

D10 provides no D11 scope-subset attestation. A private D11 token checks the
four target member identities against the eight source member identities. A
smaller count alone, including the frozen disjoint negative fixture, is not a
scope proof.

The only later edges are:

```text
D11TargetCanaryContinuationToken -> D11DegradedCanaryToken
D11DegradedCanaryToken           -> D11SuspendedCanaryToken
D11SuspendedCanaryToken          -> D11RevokedCanaryToken
```

The three edges require distinct private triggers produced from the immediately
preceding state: active-calibrator drift binds epoch `1 -> 2`, target ambiguity
binds `2 -> 3`, and concept shift binds `3 -> 4`. Public diagnostic records
cannot replace these triggers. There is no generic state dispatcher, inverse,
join, upcast, restoration, renewal, or authority producer. Each transition
asserts exact fixture run, warrant, site, model, workflow, population, loss,
group plan, calibrator, window, epoch, rank, and scope fields.

The final token is terminal only in the bound nominal trace. It does not prove
global absorption, disable a runtime canary, invalidate a copied stale value,
or establish a unique execution chain; the frozen producers remain replayable.
The trace separately retains baseline, post-calibration-drift, last-labeled
pre-ambiguity, and post-concept-shift snapshot fingerprints.
In the parallel ontology, production, clinical, and institutional authority
are children of `ReservedAuthorityArtifact`, never of `WarrantStateArtifact`.

## Required Gates

- one exact joint-law fingerprint, exact covariate weighting, overlap interval,
  separate label probe, singular confusion, concept ambiguity, subgroup
  collision, diagnostic and active-calibrator residuals, and tight TV-risk
  arithmetic in Sounio and Python;
- member-wise frozen scope inclusion plus a smaller-disjoint-scope refusal;
- private target-evidence and downward state flow through an imported
  check-only witness;
- exact expected/found diagnostics for every nominal wall, including public
  observations versus private law, scope, and sequential trigger tokens;
- E176 for direct construction of reserved private authorities;
- default and rebuilt current-source ontology validation;
- recursive D10 gate and therefore D9-D0;
- xAI and Z.AI math review plus hostile clinical-authority review.

## Compiler Boundary

D11 changes no compiler, resolver, IR, D10, or earlier semantic file. Imported
execution remains check-only under `BLK-20260718-D6-MULTIMODULE-RUNTIME`.
Static one-use, stale-alias invalidation, and live revocation remain outside
D11 and require the pending affine compiler and trusted monitor interface.

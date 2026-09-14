<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-cubic-operator-forge
authority: repo_only
audience: users
last_validated: 2026-08-29
validated_by: codex
source_of_truth: stdlib/hardware/pireus/cubic_operator_forge.sio
-->

# Pireus Cubic Operator Forge

Status: `PARITY_OPEN` (generation frozen; no parity implementation executed)

Concept-ID: `SOUNIO-PIREUS-CUBIC-OPERATOR-FORGE`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

Generation-ID: `pireus-cubic-operator-forge-v4-20260829`

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-operator-genome-v3-20260829
Generation-ID: pireus-cubic-operator-forge-v4-20260829
Owner: codex/pireus-operator-genome-v3-20260829
Concept-IDs: SOUNIO-PIREUS-CUBIC-OPERATOR-FORGE (proposed)
Intent-Preserved: Pireus generates operator novelty with Sounio as semantic authority
Transformation: frozen v3 genome -> complete 48-member single-mixed-cubic child population
Types-Changed: adds mutation descriptors, child genomes, population certificates, and lineage receipts
Effects-Changed: none
IR-Changed: none
Claims-Introduced: bounded exact-coordinate novelty outside the full bilinear phase grammar
Claims-Forbidden: GL/gauge inequivalence, algebraic, algorithmic, material, scientific, historical, global, priority, and claim-ready novelty
Assumptions: live frozen v3 parent, Boolean ANF convention, exact cd_sigma source, XOR displacement law
Write-Set: v4 Garden, concept draft, Sounio module, example, structural test, receipts, evidence, and dedicated gate
Read-Set: v2 bilinear parent, v3 genome parent, Cayley-Dickson sign source, target profiles, U250 declaration, language-authority Guardian
Positive-Witness: 48 canonical mutation descriptors with support 32 and nonzero declared 2-cocycle-failure witnesses
Negative-Witness: malformed grammar, bilinear equality, child collision, semantic-field mutation, promotion, and forbidden-oracle refusals
Acceptance-Gate: scripts/ci/pireus_cubic_operator_forge.sh
Integration-Target: current Pireus operator-generation lineage
Authoritative-Only-If: Garden precedes matcher-free Sounio execution, first result is Git-preserved, semantics are hash-frozen, and the native Guardian gate passes
```

## Boundary

The forge adds exactly one mixed cubic Boolean monomial to the phase of the
frozen v3 operator. It emits all 48 canonical mutations and selects none.

Every child preserves the parent XOR partner, destination, ordinal, scalar,
and strict reduction contracts. Only the sign phase changes. The exact
grammar, witnesses, target-envelope emptiness, and permitted novelty language
are fixed by
`tools/pireus/GARDEN_PIREUS_CUBIC_OPERATOR_FORGE_V4.md`.

The v3 parent being `PARITY_OPEN` did not transfer parity state to v4. Pireus
restarted at `GARDEN`, committed a matcher-free Sounio generator, preserved its
first authorized result, and only then added the hash-bound matcher. The frozen
population contains 48 distinct child identities and selects none.

The authoritative v4 artifacts are:

```text
stdlib/hardware/pireus/cubic_operator_forge.sio
tools/pireus/PIREUS_CUBIC_OPERATOR_FORGE_CONTRACT_V4.md
tools/pireus/cubic_operator_forge.freeze.v4
tools/pireus/evidence/cubic_operator_forge_v4.txt
tools/pireus/cubic_operator_forge.guardian-decisions.v4
tools/pireus/cubic_operator_forge.parity-open.v4
scripts/ci/pireus_cubic_operator_forge.sh
```

The frozen forge identity is
`e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff`.
The gate terminates as `STAGE_REACHED_NOT_A_CLAIM`: 1920 target obligations,
all parity implementations, every material observation, child selection,
broader novelty, and `CLAIM_READY` remain open or false.

## Registry Integration

The intended registry row is:

```text
SOUNIO-PIREUS-CUBIC-OPERATOR-FORGE\texecutable\tfounder\tdocs/internal/concepts/pireus-cubic-operator-forge.md\tstdlib/hardware/pireus/cubic_operator_forge.sio\tGL-gauge-algebraic-algorithmic-material-scientific-historical-global-priority-and-claim-ready-novelty
```

The row and generated governance metadata remain pending while their shared
files are owned by another live lane. This draft contract is not a substitute
for final registry integration.

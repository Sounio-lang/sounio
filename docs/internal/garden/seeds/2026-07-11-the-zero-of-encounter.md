<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-07-11-the-zero-of-encounter
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-07-11-the-zero-of-encounter
-->

# The Zero of Encounter

> **Status**: Garden seed driven to a ledger-scoped claim | **Last validated**: 2026-07-25 | **Source**: live conversation in New Haven

## Butterfly

> muitas vezes encontros sao zeros

> o mal encontro espinosano e o zero-divisor do sedenion

Two presences can produce an absence without either presence being empty.

## Core Idea

Zero is often stored as if it described an operand: nothing was present, no
effect existed, or no event occurred. This butterfly asks whether some zeros
belong instead to the encounter.

The exact algebraic image is a sedenion zero divisor. There can be nonzero
elements `a` and `b` for which:

```text
a != 0
b != 0
a * b == 0
```

The factors remain nonzero. The zero is produced by their particular
composition.

The Spinozan image is the bad encounter: neither participant must be bad in
essence, yet their relations may compose in a way that reduces or decomposes a
capacity to act. The analogy is structural, not an assertion that Spinoza's
metaphysics is literally sedenion algebra.

The scientific pressure is broader: ordinary computation collapses distinct
histories into the same surface value.

```text
Zero::Absent
Zero::Cancelled
Zero::Annihilated
Zero::BelowResolution
Zero::Rounded
Zero::Gated
Zero::Unknown
```

The same pressure applies at the other boundary. `Infinity` may mean unbounded
growth, a singularity, overflow, non-normalizability, or an unresolved model.
Zero and infinity may therefore be events with provenance rather than naked
values.

The clinical butterfly is a question, not a finding: when an observed effect is
zero, was the effect absent, cancelled by another process, annihilated in an
interaction, hidden below resolution, or lost numerically? A patient encounter
may inspire that question, but no patient story is evidence and no identifying
clinical detail belongs in this seed.

## Connections

- [`docs/internal/garden/README.md`](../README.md) defines the evidence boundary
  for Garden seeds.
- [`2026-05-10-epistemic-fermentation.md`](2026-05-10-epistemic-fermentation.md)
  establishes the neighboring idea that equal surface values can remain
  different knowledge because their paths differ.
- [`stdlib/algebra/sedenion.sio`](../../../../stdlib/algebra/sedenion.sio)
  provides the current sedenion algebra and canonical zero-divisor helpers.
- [`examples/conversational_ossm/o_ssm_conflict.sio`](../../../../examples/conversational_ossm/o_ssm_conflict.sio)
  already uses sedenion zero-divisor proximity as conflict telemetry and freeze
  pressure in an experimental conversational controller.
- [`stdlib/eisa/core_v2.sio`](../../../../stdlib/eisa/core_v2.sio) separates the
  hardware-compatible value lane, the numerical correction trail, and physical
  input uncertainty as `val`, `err`, and `u`.
- [`stdlib/epistemic/path.sio`](../../../../stdlib/epistemic/path.sio) is the
  nearest executable precedent for keeping computational history distinct when
  surface values agree.
- [`stdlib/epistemic/zero_event.sio`](../../../../stdlib/epistemic/zero_event.sio)
  now provides private receipt constructors, evidence flags, accessors, and an
  explicit typed discharge to an ordinary `f64` surface.
- [`tests/known_failures/zero_provenance_native_v2_probe.sio`](../../../../tests/known_failures/zero_provenance_native_v2_probe.sio)
  constructs five surface-zero paths and checks that their provenance remains
  distinguishable.
- [`scripts/ci/zero_provenance_witness_gate.sh`](../../../../scripts/ci/zero_provenance_witness_gate.sh)
  checks the witness with Madaros and executes it explicitly through
  `lean_single`, without hiding the engine split.
- [`scripts/ci/zero_event_gate.sh`](../../../../scripts/ci/zero_event_gate.sh)
  verifies the stdlib receipts, check and compile constructor opacity, explicit
  discharge boundary, and derived EISA flags.
- [`scripts/ci/zero_event_native_v2_matrix.sh`](../../../../scripts/ci/zero_event_native_v2_matrix.sh)
  keeps the native frontier classified: `dd64` and the zero-event receipt run,
  `qd128` fails closed during native emission, and sedenion exits nonzero
  without crashing or proving its semantic marker.
- [`docs/research/garden_to_claim_pipeline_spec_2026-07-25.md`](../../../research/garden_to_claim_pipeline_spec_2026-07-25.md)
  specifies the Garden-to-Claim pipeline this seed was driven through.
- [`scripts/research/garden_to_claim_pipeline_contract.py`](../../../../scripts/research/garden_to_claim_pipeline_contract.py)
  verifies the stage evidence, including the no-overclaim ceiling.
- [`scripts/ci/garden_to_claim_gate.sh`](../../../../scripts/ci/garden_to_claim_gate.sh)
  composes both witness gates with the pipeline contract.
- [`stdlib/epistemic/zero_encounter_pipeline_claim.sio`](../../../../stdlib/epistemic/zero_encounter_pipeline_claim.sio)
  encodes the ledger-scoped claim `garden_zero_encounter_pipeline`.
- [`FOUNDER_INTENT.md`](../../../../FOUNDER_INTENT.md) protects the underlying
  requirement that a zero result must not silently erase its path.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Captured: "many encounters are zeros" and "the Spinozan bad encounter is the sedenion zero divisor." |
| `Hypothesis` | A typed zero-provenance taxonomy may distinguish absence from relational, numerical, metrological, and policy-produced zeros. Sedenion zero-divisor geometry may provide a useful model for some nonzero-factor interaction failures. |
| `Executable` | `epistemic::zero_event` implements receipt evidence, accessors, typed discharge, and E176-gated opaque constructors. Its aggregate-return witness executes on default native-v2. EISA exposes derived `ZERO_OBSERVED` and `CORRECTION_NONZERO` flags without changing `val`/`err`/`u`. Remaining native frontiers are qd128/EISA backend emission and the sedenion semantic marker. |
| `Claim-ready` | Yes (ledger-scoped): the narrow executable proposition `same surface value != same zero provenance` is encoded in the Falsification Ledger as `garden_zero_encounter_pipeline` at evidence `gate_green`. No biological, psychopharmacological, metaphysical, or novelty claim is established by this seed. |

## What This Is Not

- Not a claim that human encounters are literally sedenion multiplication.
- Not evidence that psychopharmacology has an octonionic or sedenionic physical
  basis.
- Not a clinical interpretation rule for a zero treatment effect.
- Not permission to infer mechanism from an observed null result.
- Not proof that every numerical zero needs a tagged runtime representation.
- Not an implemented Sounio language feature or EISA instruction family.
- Not a statement that zero and infinity are mathematically undefined; the
  seed concerns loss of provenance when different limiting or computational
  events share a surface representation.

## Executable Bridge

The first inert, non-clinical Sounio witness now contains five computations
whose surface value is zero but whose construction paths differ:

1. literal absence;
2. additive cancellation;
3. exact sedenion zero-divisor annihilation;
4. a nonzero value below a declared measurement resolution;
5. a nonzero correction trail whose `f64` value lane rounds to zero.

The gate proves only:

```text
same surface value != same zero provenance
```

It does not add compiler syntax or promote a general `LimitEvent` design.

## Next Executable Bridge

Repair or receive an ownership transfer for the native-v2 compiler surfaces
identified by the matrix. Keep the passing `lean_single` witnesses unchanged as
semantic oracles. Native-v2 parity requires all of the following without an
engine fallback:

1. minimal `qd128` import emits and executes;
2. minimal sedenion zero-divisor code prints its pass marker;
3. the combined provenance witness executes without the `lean_single` oracle.

`InfinityEvent` remains a separate Garden butterfly and is not a requirement or
implicit extension of this zero-event implementation.

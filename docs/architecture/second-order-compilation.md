<!-- docs:meta
topic_id: repo.docs.architecture.second-order-compilation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.second-order-compilation
-->

# Madaros Second-Order Compilation (C2-v0)

Status: experimental semantic architecture

## Purpose

Madaros is not only a translator from Sounio source to an executable artifact.
For scientific programs, compilation is an intervention: the compiler chooses
representations, preserves or erases order, selects lowering paths, introduces
instrumentation, and realises requested semantics on a target.

C2-v0 makes that intervention observable. It defines a compiler-owned receipt
that records what semantics were requested, what semantics were realised, what
the compiler changed, what was actually observed, and where two controlled
compilation trajectories first cease to agree.

This is called second-order compilation because the compiler participates in
the variation being studied and reports evidence about that participation. A
paired experiment invokes the compiler more than once, but repetition alone is
not second-order compilation: controlled intervention, trace alignment,
bounded observation, and an explicit account of blind spots are also required.
The term does not mean taking a second derivative or attributing consciousness
to the compiler.

Concept-ID: `SOUNIO-SECOND-ORDER-COMPILATION`

## Founding Principle

> A scientific compiler must expose the epistemically relevant effects of its
> own transformations. It must not claim equivalence, absence, or successful
> realisation where its observation surface cannot support that claim.

The first-order result remains the executable artifact. The second-order result
is a bounded account of the relationship between source intent, compiler
intervention, realised artifact, runtime observation, and counterfactual
alternatives.

## Operational Model

The notation below defines the operational boundary and classification inputs
for C2-v0. It is not a formal proof or a machine-checked semantics.

For source `s`, compiler identity `c`, semantic profile `sigma`, target and
runtime environment `e`, and observation projection `omega`, define one run as:

```text
run(c, s, sigma, e, omega) -> (artifact, trace, observation, blind_spots)
```

A controlled second-order compilation compares two runs:

```text
C2(c, s, sigma_a, sigma_b, e, omega)
  -> (run_a, run_b, alignment, first_divergence, claim_scope)
```

The source, compiler build, target environment, inputs, and observation
projection are pinned unless a receipt explicitly names them as intervention
dimensions. A comparison with undeclared changes is `INCOMPARABLE`, not a
counterfactual witness.

`first_divergence` is valid only when the traces have comparable event
identities. If the compiler cannot align the traces, the result is `UNALIGNED`.
It must not guess a first divergent operation from final values alone.

## Intervention Dimensions

C2-v0 recognises these compiler-controlled intervention classes:

- numeric representation and precision;
- rounding, contraction, and reassociation policy;
- parenthesisation and evaluation order;
- effect handling and discharge;
- provenance and correction-channel preservation;
- IR transformation and lowering path;
- target ISA or software-emulation path;
- runtime instrumentation and observation projection;
- fallback selection.

The list is extensible. A new intervention class must declare its semantic
identity and receipt representation before it can support a C2 claim. The
declaration uses the Semantic Lane Contract, adds or updates a Concept Registry
binding, and increments the receipt schema version when existing consumers
cannot interpret the new dimension without ambiguity.

## Required Invariants

### Requested semantics and realised semantics are separate facts

The receipt records both. A backend that cannot realise the request reports a
blocked or explicit fallback path. It never reports the requested profile as
though it had been executed.

If `requested_semantics != realised_semantics`, the run is `BLOCKED` unless the
source or experiment contract explicitly authorised that fallback. An
authorised fallback remains visible and can support claims only about the
realised profile. C2-v0 specifies this rule; checker and backend enforcement are
pending implementation and therefore cannot yet support an execution claim.

### Observation is bounded

Every receipt names its observable fields and blind spots. No observed
divergence within a bounded projection supports only `OBSERVED_EQUIVALENT`, not
unqualified semantic equivalence.

### Instrumentation is an intervention when it can perturb the run

Instrumentation may be treated as observational only when its non-interference
contract is established for the stated surface. Otherwise it is recorded as an
additional intervention dimension.

### Semantic identities survive the pipeline

Source operation identity, parenthesisation, requested numeric format,
correction channel, and relevant provenance must remain alignable through the
compiler stages required by the witness. A backend-local value without this
link cannot identify a first compiler-caused divergence.

### Erasure is explicit

Narrowing, reassociation, contraction, provenance loss, correction loss, and
fallback are receipt events. Silence is not a permitted representation of
erasure.

### Physical and clinical meaning is not inferred from numerical difference

A stable numerical residual is compiler evidence. Connecting that residual to
a physical mechanism, measured phenomenon, or clinical effect requires a
separate model and evidence chain.

A C2 receipt never authorises clinical decision support, diagnosis, dosing, or
patient-level inference. `OBSERVED_EQUIVALENT` has no clinical-safety meaning;
it describes only the declared computational observation projection.

## C2ReceiptV0

The logical receipt has the following minimum schema. Concrete serialisation is
left to the implementation lane.

```text
C2ReceiptV0 {
  receipt_version
  evidence_identity
  source_identity
  compiler_identity
  environment_identity
  observation_projection
  intervention_dimensions

  run_a {
    requested_semantics
    realised_semantics
    transformation_path
    fallback_path
    artifact_identity
    execution_status
  }

  run_b {
    requested_semantics
    realised_semantics
    transformation_path
    fallback_path
    artifact_identity
    execution_status
  }

  alignment_status
  first_divergence {
    stage
    operation_identity
    source_identity
    value_a
    value_b
    correction_a
    correction_b
    status_flags_a
    status_flags_b
    branch_outcome_a
    branch_outcome_b
  }

  observed_fields
  blind_spots
  comparison_status
  classification_basis
  integrity_status
  claim_scope
}
```

Fields that do not apply are represented explicitly. They are not silently
omitted when omission could be confused with a zero, empty, successful, or
unobserved value.

`evidence_identity` binds the source, compiler, environment, artifacts, and
trace inputs used for classification. `classification_basis` identifies the
aligned event set and comparison rule. `integrity_status` records whether the
receipt passed its declared tamper and completeness checks. These fields make
the v0 receipt auditable; they do not constitute a formal or cryptographic
proof unless a later contract explicitly supplies one.

## Comparison Status

- `DIVERGED`: an aligned event differs under the declared observation rule.
- `OBSERVED_EQUIVALENT`: no aligned observed event differs; the claim remains
  limited to the declared projection.
- `UNALIGNED`: trace identity is insufficient to locate a comparable event.
- `INCOMPARABLE`: an undeclared or uncontrolled intervention differs.
- `BLOCKED`: at least one requested run did not reach the evidence boundary.

`BLOCKED` is not `OBSERVED_EQUIVALENT`. `UNALIGNED` is not absence of
divergence.

## First Planned Vertical Witness

No C2 first-divergence witness exists at the time of this specification. The
concept therefore remains `hypothesis`. The requirements below are the
acceptance boundary for the first implementation, not evidence that it has
already passed.

The first implementation slice uses existing, explicitly distinct arithmetic
surfaces:

```text
same source
same compiler build
same target and runtime inputs
same declared parenthesisation

run_a: EISA v1 with dd64
run_b: EISA v2 with qd128
```

`dd64` and `qd128` are expansion-arithmetic formats. They are not renamed or
reported as IEEE `f128` or `f256`.

The witness must:

1. execute both requested paths without silent `f64` fallback;
2. preserve alignable operation identities through the compared surface;
3. report the first aligned numerical, correction, status, or branch
   divergence;
4. prove tamper sensitivity by changing a compared value or identity and
   causing the gate to fail;
5. return `OBSERVED_EQUIVALENT`, `DIVERGED`, `UNALIGNED`, `INCOMPARABLE`, or
   `BLOCKED` without collapsing them into one Boolean;
6. limit its claim to the exact compiler SHA, profiles, input, target, and
   observation projection in the receipt.

This witness does not implement native `f128`, `f256`, or `f512`. It validates
the comparison and receipt architecture that those formats will later use.

## Compiler Pipeline Contract

C2 metadata is compiler-owned, but it must remain proportionate:

- frontend: identify requested semantics and explicit intervention choices;
- checker: reject undeclared narrowing or incompatible profile combinations;
- semantic IR: preserve operation identity and epistemically relevant fields;
- optimisation: emit transformation events for protected operations;
- machine lowering: record realised target path and fallback status;
- runtime: expose only the observations required by the declared projection;
- receipt layer: align runs, classify comparison status, and state blind spots.

Not every instruction must carry a large receipt object. Stable identities and
side tables are permitted if they preserve the same semantics and tamper
sensitivity.

## Hardware Contract

New hardware may implement a semantic profile natively, through microcode, or
through software. Hardware support changes the realised path, not the language
meaning. A backend is compatible with C2-v0 only if it can:

- identify the requested and realised semantic profiles separately;
- expose protected status, correction, ordering, and fallback events;
- preserve or reconstruct operation alignment at the declared boundary;
- fail closed when it cannot honour a required semantic invariant.

This permits EISA and future hardware to evolve without making present host
ISAs the definition of Sounio semantics.

## Relationship to Existing Concepts

C2-v0 composes existing contracts; it does not redefine them:

- `SOUNIO-ZERO-PROVENANCE` distinguishes an exact, measured, rounded,
  underflowed, cancelled, or unknown zero;
- `SOUNIO-EPISTEMIC-NUMERIC-VALUE` separates value, arithmetic error, and
  uncertainty;
- `SOUNIO-NONASSOCIATIVE-ORDER` protects order and parenthesisation;
- `SOUNIO-EXPLICIT-DISCHARGE` makes authorised information loss visible;
- `SOUNIO-PHYSICAL-OBSERVATION` governs the later bridge from computation to
  physical observation;
- `SOUNIO-PRECISION-PRESERVATION` forbids silent precision demotion.

The second-order receipt is an orchestration surface across these concepts.
It is not evidence that every concept is already integrated across Madaros.

## Claims Permitted by C2-v0

After the first witness passes, Sounio may claim that the named experiment:

- executed two declared semantic profiles under pinned conditions;
- preserved the receipt fields named by the gate;
- classified their comparison at the first aligned observed divergence; and
- exposed the named blind spots and fallback paths.

It may not claim from that witness alone:

- general equivalence between arithmetic formats;
- native `f128`, `f256`, or `f512` support;
- a physical, biological, psychiatric, or clinical mechanism;
- causal attribution outside the controlled compiler intervention;
- complete observation of every compiler transformation;
- superiority over another language or ISA.

## Decision Rule

C2-v0 advances from `hypothesis` to `executable` only when the first vertical
witness passes its positive, negative, tamper, and blind-spot cases on a pinned
compiler build. It advances to `integrated` only when every compiler layer
required by its public contract preserves the receipt invariants without a
silent fallback.

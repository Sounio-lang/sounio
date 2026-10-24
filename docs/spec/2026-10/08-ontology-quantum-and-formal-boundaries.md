<!-- docs:meta
topic_id: repo.docs.spec.2026-10.08-ontology-quantum-and-formal-boundaries
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.08-ontology-quantum-and-formal-boundaries
-->

# Ontology, quantum interfaces, and formal models

Sounio includes semantic facilities and research libraries beyond ordinary
scalar programs. Their contracts have different authorities: compiler checks,
library algorithms, serialized requests, formal models, and observations of
external systems. This chapter identifies the interfaces between them.

Rule identifiers do not imply verified enforcement. Requirements on a claim
or receipt are not new language semantics. The [foundations](00-foundations-and-conventions.md#1-scope-and-authority)
define the standing categories recorded in the [evidence catalog](08-evidence.tsv).

## 1. Ontology vocabulary and judgments

An ontology vocabulary names concepts, roles, axioms, and, where supported by
the selected surface, individuals and assertions. A terminological relation
such as subsumption is a relation in that declared ontology. It does not by
itself establish a biomedical diagnosis, physical mechanism, or empirical
truth about a particular individual.

`ONT-001` — A claim of ontology acceptance shall name the vocabulary and the relation
being checked. Subclass subsumption, role domain/range, inverse roles,
disjointness, temporal consistency, and application claim predicates have
distinct rules and witnesses. Successful checking of one does not establish
the others.

Compiler-side ontology-related work is present in the checker and the
ontology side-table cache, while `self-hosted/compiler/ontology_bundle_directive.sio`
handles a driver-facing bundle surface. Runtime/library components include
`stdlib/ontology/model.sio`, `closure.sio`, `reasoner.sio`, `query.sio`,
`repair.sio`, and `elplus.sio`. The generated and typed vocabulary directories
provide additional representations and adapters. Their identities must be
retained across resolution and validation.

`ONT-002` — A vocabulary adapter or generated bundle shall identify its source
vocabulary, the transformation applied, and the validation procedure. A term
appearing in a generated Sounio file does not by itself establish full
coverage, version currency, or licensing authority for the external ontology.

Representative refusal fixtures include
`tests/compile-fail/ontology_disjoint.sio`,
`tests/compile-fail/ontology_role_domain_reject.sio`, and
`tests/compile-fail/ontology_role_range_reject.sio`. Positive counterparts
must reach the same vocabulary and checker route; a simpler unrelated
ontology example is not a control for that route.

## 2. EL+ closure model

`stdlib/ontology/elplus.sio` implements finite boolean closure surfaces.
The dense representation stores concept subsumption and role edges in fixed
arrays. Its `ELPLUS_MAXC`, `ELPLUS_MAXR`, and related constants bound that
implementation. Sparse operations use their own indexing and workspace
requirements. The dense and sparse algorithms require separate conformance
evidence when their representations or supported profiles differ. The sparse
profile documented by this source excludes conjunctions and role hierarchy/
composition and uses one active role. Its module workspaces must be initialized
through `elplus_sparse_init_mats`; `elplus_sparse_expand` is not the full dense
rule set.

The relevant logical distinction is between a declared axiom set and a
derived relation. A closure computation adds consequences until the selected
rules reach their stopping condition. A query reads that closure. A conflict
query uses the declared disjointness relation and the relevant closure
representation. The finite boolean model is not a probabilistic estimate of
the truth of the ontology.

`ONT-003` — An EL+ claim shall identify the supported axiom/rule fragment,
finite signatures, index bounds, closure algorithm, and queried relation.
Completeness for a formal fragment does not imply support for arbitrary
OWL features, federation, open-world data integration, or unrestricted
assertional reasoning.

The formal file `formal/OntologyELPlusClosureComplete.lean` defines a finite
model and proves named relationships between its boolean oracle and its
derivability semantics. The theorem `subBPlusC_iff` is an important model
interface. Its exact statement, imports, and axiom footprint determine the
proof claim. The executable Sounio library is a hand-maintained mirror;
correctness of that mirror requires its own correspondence evidence.

## 3. Ontology validation routes

`ONT-004` — An ontology result shall identify whether it used the default
compiler route or the rebuilt/current-source validation wrapper. The wrapper
builder in `scripts/ci/build_ontology_validation_souc.sh` and runner in
`scripts/ci/run_ontology_validation.sh` provide explicit modes and probes.
The builder's fallback compile oracle is another named input to the result.

At this revision, the principal verdict driver
`scripts/ci/ontology_validation_driver.sio` only lexes and parses. It reports
`witness=0` after a parse without parser errors; it does not invoke the
ontology checker. The separate debug driver collects checker diagnostics,
but those diagnostics do not control every generated-wrapper verdict.
The `ontology_driver_ok_fallback_constructor_gap` branch can report acceptance
for ontology-containing source after the principal driver succeeds and the
fallback compilation rejects it. That branch does not establish that the
fallback failure was limited to a constructor gap.

The generated wrapper's `run` path also has a synthetic-output branch: if
fallback compilation fails while the principal driver verdict is zero, it
prints the fixture's `//@ expect-stdout:` annotations and exits successfully.
Those bytes are expected text copied from source, not output of an executed
program. They cannot discharge a semantic, numerical, or execution obligation.
An execution claim requires a separately identified compiled artifact and
observed execution. These are current validation limitations requiring repair,
not intended ontology semantics.

For tracking within this edition, these source-observed hazards are
`ONT-H01` (parser verdict used beyond syntax), `ONT-H02` (fallback failure
overridden without cause discrimination), and `ONT-H03` (synthetic expected
stdout). These labels identify findings, not fresh runtime reproductions.
The branch name containing `constructor_gap` is an implementation label,
not evidence that its failure cause was established. Results of these unsafe
paths are excluded from this edition's semantic and runtime conformance
claims. The wrapper is indexed as a subject of investigation, not a positive
acceptance witness. The separately executed bounded manifest did not use this
ontology wrapper.

The current runner includes compile-gate probes and can skip a compile-gate
bundle when its selected compiler lacks a required sequence surface. A skipped
bundle is not evidence that every gate in that bundle passed. The runner's
actual logs, generated-wrapper identity, and per-gate results are necessary
to interpret the outcome.

Failures are classified using the existing project distinctions:
build/bootstrap path, harness routing, ontology kernel/checker, and baseline
noise. This classification is important because a wrapper-routing repair can
change the test's effective implementation without changing the ontology rule.

## 4. Linear quantum simulation

`stdlib/quantum/linear.sio` provides a classical statevector representation
with a linear `Qubit` handle and an explicit `QuantumState`. Allocation grows
the active state within the declared maximum. Gate functions consume handles
and return the successor handles or aggregates appropriate to the operation.
The compiler's linear checking controls source use of the handles.

`QNT-001` — A linear qubit handle shall be used according to its checked
resource-usage contract. Source-level no-copy/no-drop enforcement is a
programming-language property. It is not a physical proof of device
no-cloning, coherence, or fault tolerance.

This is a resource-usage contract of the library interface, not a claim that
every compiler profile currently enforces it. The named negative fixtures
need executed results on the claimed route.

`QNT-002` — The statevector library's numerical domain is its implemented
floating representation and mathematical routines. Statevector simulation
does not introduce shot noise unless a sampling operation is performed.
An algorithm described as exact statevector evolution still uses finite
arithmetic and approximations in the implementation.

The current measurement interface is:

```sounio
pub fn measure_qubit(q: Qubit, qs: &!QuantumState, rng_seed: f64)
    -> MeasurementResult with Mut, Div, Panic
```

`QNT-003` — `measure_qubit` consumes the qubit handle, computes the selected
wire's probability, compares the supplied `rng_seed` value to that
probability, collapses the corresponding statevector branch, and returns the
observed outcome and its probability. The input is used as the coin value;
this function does not establish that the caller supplied a uniform random
sample. Its normalization threshold and approximate square root are part of
the selected implementation's numerical boundary.

Despite the parameter name, `rng_seed` is not seed material for a generator.
Callers that intend Born-law sampling must provide an appropriate uniform
comparison sample in `[0, 1)`; the function neither generates that sample nor
validates its distribution. Repeatedly supplying a fixed seed as the sample
does not implement the intended sampling law.

The public tests include `tests/stdlib/quantum/test_linear.sio` and the
negative linear fixtures `tests/compile-fail/qubit_used_twice.sio` and
`tests/compile-fail/qubit_not_consumed.sio`. Their actual compiler profile and
outcome must be recorded before claiming current enforcement.

## 5. Epistemic and sampling channels

`QNT-004` — A channel's presence is a semantic fact separate from a numeric
zero. `BandChannel.filled` controls whether the channel has a numeric read.
`band_variance` refuses an absent channel. `has_split` separately controls
whether first/second-order standard-uncertainty contributions may be read.
Those contributions are distinct from the separate derivative fields.
Sentinel payloads in
absent fields are not variances.

In `stdlib/quantum/born_channel.sio`, angle uncertainty, sampling uncertainty,
and covariance can have different presence states. Reusing one epistemic
input can supply a shared-source covariance channel. Constructing two records
with equal numbers does not by itself establish that shared provenance.

`stdlib/quantum/born_shot_law.sio` adds explicit shot-law and mixed-observation
models. Their model assumptions and APIs must be read separately from the
compiler's `Knowledge<T>` propagation. The model's formulas do not constitute
an empirical device-calibration result. Tests for those models include
`tests/stdlib/quantum/test_born_channel.sio` and
`tests/stdlib/quantum/test_born_shot_law.sio`.

## 6. Circuit serialization and device boundary

`QNT-005` — A `Tape` in `stdlib/quantum/qasm_tape.sio` records a bounded circuit
operation sequence. `tape_qasm` emits OpenQASM 3 text for its implemented gate
and measurement subset. `tape_qir` emits textual QIR-shaped LLVM IR with the
implemented runtime declarations and profile attributes. String production is
a serialization result; external parser/verifier acceptance is a separate
obligation.

The emitted subset does not establish conformance to the complete OpenQASM
or QIR specifications. Format conformance, external acceptance, and execution
equivalence require their own witnesses.

The QASM emitter preserves tape order. The QIR emitter instead collects
unitary operations before all measurements and output recording. When a gate
follows a measurement in the tape, it is moved before measurements in the
emitted QIR; a late-gate comment records the condition without rejecting it.
The public replay entrypoints support one or two qubits, ignore measurement
opcodes during their unitary walk, and consume their handles by measurement
after reading final probabilities. They do not simulate intermediate tape
measurements or conditional gates. Thus QASM, QIR, and replay do not share an
arbitrary tape scheduling contract. Unitary operations followed by terminal
measurements form a candidate comparison subset; that subset is not generally
enforced and no equivalence theorem is claimed here. Existing QIR substring
checks do not establish external verifier acceptance or scheduling equivalence.

`QNT-H01` identifies the source-observed scheduling hazard. Moving a gate
across a measurement is not generally semantics-preserving. An unrestricted
tape containing post-measurement gates must not be submitted or cited as
equivalent merely because QIR text was emitted. A comment is not a refusal
or a schedule check. Such claims require a repaired emitter or explicit
admission and validation of a safe subset.

`QNT-006` — `ShotRequest`, `DeviceJob`, and returned count records define an
interchange boundary. A generated job is a request. A parsed count record is
an input observation. A local oracle or pseudo-random sampler is a simulation
route. Hardware execution requires a device-specific receipt identifying the
submitted circuit, device/backend, shot budget, returned data, and applicable
validation assumptions.

`tape_device_return` itself accepts two count texts and sets its oracle marker;
it does not receive a job identifier or requested shot budget. `tape_judge`
uses the total in the supplied counts. Device origin, job identity, and equality
to the requested shot count therefore remain external receipt obligations,
not enforced properties of a `DeviceRead` value.

`QNT-007` — `qpu::chsh` combines the implemented circuit/tape and count
interfaces into named correlation witnesses. A source sampler's CHSH result
is evidence about the sampled model and its assumptions. It does not by
itself establish a loophole-free Bell experiment, quantum advantage, a
hardware execution, or a physical interpretation of hypercomplex algebra.

The relevant sources are `stdlib/qpu/chsh.sio`, `stdlib/qpu/hyper.sio`, and
their tests under `tests/stdlib/qpu/`. The hypercomplex operations have their
own algebraic definitions and parenthesization contracts.

## 7. Formal proof surfaces

`FORM-001` — A formal result shall identify its model, statement, assumptions,
checker/toolchain, and completed build. Absence of the word `sorry` in a
source search is insufficient: imported axioms, generated statements,
`native_decide`, and the actual theorem dependency footprint matter.

`formal/Effects.lean` models effect-row relations and proves properties of
those definitions. `formal/LinearTypes.lean` models usage environments and
consumption. `formal/TypeCheckerSoundness.lean` contains a small epistemic
calculus and an operational bound theorem. That file explicitly declares
floating-order assumptions and limits its calculus to a restricted expression
language. Its result must not be generalized to every production checker,
loop, call, target, or floating special value.

`FORM-002` — A proof about an abstract model, an extracted implementation,
a hand-maintained executable mirror, and a compiler/runtime implementation
are different objects. A transfer claim between them requires an explicit
mapping and correspondence argument or bounded conformance evidence.

The generated `formal/AXIOM_INVENTORY.md` is a source inventory with its own
declared scope. `scripts/ci/lean_axiom_report.sh` builds named packages and
prints actual dependency footprints for headline theorems. That script
rejects `sorryAx` in its selected footprint while allowing declared
nonstandard axioms to be reported. Its successful result therefore does not
mean every theorem is axiom-free.

## 8. Domain and evidence boundary

`FORM-003` — Formal validity in a declared model, agreement with a numerical
oracle, execution of an implementation, empirical support, and clinical
validation are separate claims. An ontology reasoner cannot supply missing
clinical evidence, and a linear quantum API cannot supply a missing device
receipt. Research modules retain their scientific intent while their
interpretation and evidence remain explicit.

This boundary follows `FOUNDER_INTENT.md` and the approved scientific
boundary and provenance concepts. It permits experimental hypotheses and
new mathematical models without turning their software representations
into evidence that the modeled phenomenon is physically present.

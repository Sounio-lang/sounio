<!-- docs:meta
topic_id: repo.docs.spec.2026-10.11-library-formal-review
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.11-library-formal-review
-->

# Independent source review: library, ontology, quantum, and formal chapters

Review date: 2026-10-16.
Checkout: `/private/tmp/sounio-main-root-cleanup-20261016`.
Verified HEAD: `f5b2c5850be9f3d3995719ffcade44a6d851658a`.

Reviewed drafts:

- `docs/spec/2026-10/07-library-packages-and-tools.md`.
- `docs/spec/2026-10/08-ontology-quantum-and-formal-boundaries.md`.

This is a read-only source review apart from this report. No compiler, library
test, validation wrapper, gate, numerical oracle, external QIR verifier, or Lean
build was executed. The findings identify source behavior and missing contract
boundaries; they are not fresh runtime reproductions. The reviewed drafts were
untracked working-tree files, so HEAD identifies their source baseline, not
committed versions of those drafts. No reviewed chapter or shared file was
edited.

## Major corrections

### LF-01 — MAJOR: the rebuilt ontology verdict driver only parses

**Affected contract:** chapter 08, section 3, `ONT-004` and its description of
the rebuilt/current-source validation wrapper.

The current text distinguishes a wrapper from a default compiler but omits what
the wrapper's principal verdict actually measures. At this exact source
snapshot, `scripts/ci/build_ontology_validation_souc.sh` assigns `CHECK_SRC` to
`scripts/ci/ontology_validation_driver.sio`. That driver's `main` imports the
lexer/parser, calls `lex_file_parse_items`, checks `parser_last_error_count`,
and prints `witness=0` after a parse without errors. It never calls the
ontology checker. The copy under `self-hosted/ci/ontology_validation_driver.sio`
also only parses, but it is the scripts copy that this builder actually uses.

The builder separately creates a debug driver from
`scripts/ci/ontology_validation_debug_driver.sio`. That driver's `main` does
invoke `checker_collect_items_mut` and
`checker_check_items_inner_store_return_had_error_mut`, and reports checker
counts. However, the generated wrapper's `run_diagnostics_check` collects these
as diagnostics; its `check` verdict dispatch is based on `DRIVER_WITNESS` from
the parser-only driver and the fallback compiler result. The diagnostic checker
does not become the authoritative verdict merely because it ran.

More seriously, generated `check` has this branch: when driver verdict is zero
and fallback compilation fails, a source matching `source_contains_ontology`
is still returned as `verdict=ok`, `provenance=rebuilt_direct`, exit zero, with
note `ontology_driver_ok_fallback_constructor_gap`. In current source this
can label parse success as ontology acceptance even though the fallback compiler
rejected the program. The note does not establish a constructor-only cause;
the branch does not inspect that cause.

**Required correction:** explicitly state that the current principal driver is
parser-only; debug-checker telemetry and fallback compilation are separate
evidence; and the ontology-source override can report acceptance without a
successful semantic checker verdict. A wrapper receipt alone cannot close an
ontology rule. Describe this as a current validation limitation requiring repair,
not the intended semantics of ontology acceptance.

**Witness boundary:** `tests/compile-fail/ontology_disjoint.sio` and the
role-domain/range negatives already cited by the draft are relevant semantic
controls. Rerun them through each named route and inspect both underlying
verdicts, diagnostics, final wrapper decision, and provenance. Their outcomes
were not measured here.

### LF-02 — MAJOR: rebuilt wrapper `run` can synthesize expected stdout

**Affected contract:** chapter 08, section 3, `ONT-004`; interpretation of any
wrapper-backed execution or passing validation-harness result.

The generated `run` branch in
`scripts/ci/build_ontology_validation_souc.sh` attempts compilation using the
fallback compiler after the driver check. When that compile fails and the
decoded driver verdict is zero, it calls
`emit_expected_stdout_annotations "$src"` and exits zero. The helper uses `sed`
to print the source's `//@ expect-stdout:` annotations. There is no execution
of a source-derived program on this branch. At this snapshot the driver's zero
can itself be mere parse success, as LF-01 explains.

The draft's warning about skipped compile-gate bundles does not expose this
different problem: an apparent successful run can consist of expected text
copied from its own fixture. Naming wrapper identity and collecting logs does
not transform that text into an execution witness.

**Required correction:** state this fallback explicitly and classify its output
as synthetic fixture text, unusable as execution, semantic, or numerical
evidence. Require an independently observed compiled artifact and actual
execution before assigning any execution claim to a wrapper-backed result.
Do not describe the branch as a working fallback implementation of the program.

**Discriminating control to prepare in the owning implementation lane:** a
parseable fixture whose fallback compilation is forced to fail and whose
expected-stdout annotation names a distinctive token. The current branch can
emit that token and success without executing it. No new fixture was written
or run in this review; no numerical calculation is needed for this control.

### LF-03 — MAJOR: tape replay, QASM, and QIR do not share arbitrary scheduling semantics

**Affected contract:** chapter 08, sections 4–6, especially `QNT-005` and the
statevector-oracle boundary used by `QNT-006`/`QNT-007`.

The draft correctly notes a late-gate comment, but omits the actual consequence
of the emitter's scheduling and the replay interpreter's treatment of
measurements. The relevant source is `stdlib/quantum/qasm_tape.sio`:

- `tape_qasm` emits operations in tape order, incrementing the measurement
  result index for each measurement.
- `tape_qir` accumulates all nonmeasurement operations into `body`, all
  measurements into `meas`, and output records into `rec`. It emits those
  blocks in that order. A gate written after a measurement is therefore moved
  before the measurements; `late` only adds a comment and does not reject.
- `walk1`/`walk2` dispatch to `step1`/`step2`. Their measurement opcode falls
  through to returning the handles unchanged. `tape_replay_1`/`tape_replay_2`
  read final probabilities after the unitary walk and only then call
  `measure_qubit` to consume handles. They do not execute intermediate tape
  measurements or condition later gates on their outcomes.
- Replay is explicitly restricted to one or two qubits by these public
  entrypoints. That is narrower than both a general `Tape.n_qubits` value and
  the separate linear statevector simulator's allocation cap.

A tape with operation order `H; measure; H; measure` is a discriminating
scheduling shape: QASM retains that order; QIR emits both H operations before
both measurements; replay ignores the two tape measurement operations during
its unitary walk. This is a source control-flow trace, not a numerical result
or an executed counterexample.

**Required correction:** document the exact replay arity, supported opcodes,
ignored tape measurement behavior, and QIR reordering. Identify the
unitary-then-terminal-measurement subset as the intended comparison boundary;
do not imply that this subset is currently enforced, or that equivalence has
been proved or tested for arbitrary tapes. A late-gate comment is not a
schedule-validity certificate. Also retain the draft's distinction between
text generation, external QIR verification, and device execution.

**Existing witness scope:** `tests/stdlib/quantum/test_qasm_tape.sio` checks
particular QASM strings and replay values. Its QIR assertions `Q_qir_h`,
`Q_qir_cx`, `Q_qir_mz`, and `Q_qir_n` search for substrings in a Bell-program
emission. They do not validate scheduling equivalence or run an external QIR
verifier. No gate-after-measurement refusal/equivalence witness was established
in the inspected tests.

### LF-04 — MAJOR: `[lib].name` is not the package import lookup mechanism

**Affected contract:** chapter 07, section 4, `PKG-001`, especially “The import
uses `[lib].name`,” and its relationship to `PKG-002`/`PKG-003`.

For the cited example, `epistemic_core` does match the manifest's library name.
The implementation does not establish that manifest field as the lookup
authority for arbitrary packages. Current import resolvers construct a path
directly from the spelling:

- `self-hosted/compiler/module_frontend.sio::module_frontend_relpath_to_package_path`
  maps the root segment's underscores to hyphens and appends `/src/lib.sio` for
  a root import, or the remaining source path for a submodule.
- `self-hosted/compiler/module_loader.sio::use_path_to_package_file_path` uses
  `string_list_seg_to_buf_mapped` for the root and the same conventional source
  layout.
- `self-hosted/compiler/lean_single.sio::imp_build_package_path_from_raw` also
  constructs `packages/<mapped-root>/src/...` without reading the manifest.

None of these functions consults `[lib].name`, `[lib].path`, or a dependency
alias to discover that path. The active frontend also searches nearer source
paths, repository-relative paths and stdlib/self-hosted paths before the
package candidate. This is source-path fallback resolution, not a general
manifest-driven package namespace. The existing package-import gate's manifest
validation does not change the compiler algorithm.

**Required correction:** say that the canonical manifest name agrees with the
current conventional import spelling, while the compiler presently resolves
the spelling through the path mapping. Explicitly leave custom library roots,
arbitrary library aliases, dependency installation/version selection, and
manifest-controlled imports outside the demonstrated contract. If
manifest-driven names are an intended specification requirement, label that
as a desired interface and expose the current implementation gap.

**Witness boundary:** `scripts/ci/package_import_science_gate.sh` pins the seed
compiler and exercises conventional in-tree package layouts. It does not test
renaming only `[lib].name`, changing only `[lib].path`, or importing a
dependency under an alias. These are missing controls, not executed failures.

## Additional clarifications and limits

These are concrete improvements, but are not classified as additional major
contradictions in the current cautious wording.

1. **Local `build` means checking.**
   `tools/sounio-pkg/src/cmd_build.sio::main` emits a shell script that checks
   for `sounio.toml` and `src/lib.sio`, then runs `"$SOUC" check src/lib.sio`.
   It does not build a distributable package artifact or honor arbitrary
   manifest targets. `cmd_check.sio::main` scans `src/*.sio`; `cmd_test.sio::main`
   runs `tests/*.sio`, not the manifest's `[[test]]` list. That list is parsed
   separately by `package_import_science_gate.sh`. State these concrete local
   tool interfaces beside the compiler CLI to prevent readers from assuming
   the two commands named `build` have the same artifact contract. The draft's
   local-versus-legacy-versus-public-registry distinctions are otherwise
   consistent with the inspected wrapper and gate.

2. **The sparse EL+ fragment is narrower than the dense rules.**
   `stdlib/ontology/elplus.sio` documents sparse operations for a profile with
   no conjunctions, one active role, and no role hierarchy/composition. They
   mutate module workspaces initialized by `elplus_sparse_init_mats`.
   `elplus_sparse_expand` is not the general dense eight-rule algorithm.
   Section 2 correctly asks for separate evidence, but a useful interface
   contract should state these specific preconditions and workspace lifecycle.

3. **Shot records are not self-validating receipts.**
   In `stdlib/quantum/qasm_tape.sio`, `tape_device_return` takes only two texts
   and sets `oracle=0`; it receives no job identity or expected shot budget.
   `tape_judge` derives its shot total from the supplied count cells rather
   than checking a supplied `ShotRequest`. These functions cannot attest
   device origin or bind counts to the requested job/shot count. The draft
   correctly requires a separate device receipt; explain that it is an external
   requirement rather than an enforced invariant of `DeviceRead`.

4. **Band components are uncertainty contributions.**
   `born_channel.sio::channel_angle` copies `LinearGum.u_first` and `u_second`
   and combines their squares into the variance. In
   `epistemic_linear.sio::elin_ry_gum`, `d1` and `d2` are separate derivative
   fields; the `u_*` values are first/second-order standard-uncertainty
   contributions. Replace “first/second derivative components” with this more
   precise wording. The channel-presence/sentinel distinction in `QNT-004`
   is supported by `band_variance`, `band_u_first`, and `band_u_second`.

## Formal and statevector checks without a major finding

The following boundaries in chapter 08 are consistent with inspected source;
no fresh successful proof or numeric execution is asserted:

- `linear.sio::measure_qubit` uses the supplied `rng_seed` directly as the coin,
  accumulates the selected wire's probability, collapses the chosen branch,
  and normalizes through `sqrt_sim` with a threshold. Its declared signature
  matches the draft. The `QuantumState` storage is `f64`, and `qubit_alloc`
  checks against `QUANTUM_MAX_AMPLITUDES`; these are classical finite-arithmetic
  simulation interfaces, not a hardware backend.
- `formal/OntologyELPlusClosureComplete.lean::subBPlusC_iff` has explicit
  `conceptUniv` membership hypotheses for both queried concepts and relates
  the boolean result to `Der`. `conflictBPlusC_iff` similarly relates a boolean
  query to `DerivedConflictP`. These statements do not prove the separately
  maintained Sounio implementation correct. No source text was treated as a
  substitute for rebuilding the Lean package.
- `formal/TypeCheckerSoundness.lean` defines only `kLit` and `kAdd` in its
  expression calculus and declares `float_le_refl`, `float_le_trans`, and
  `float_add_le_add` assumptions. The draft correctly limits the operational
  bound theorem and does not transfer it to all compiler programs or floating
  special values.
- `scripts/ci/lean_axiom_report.sh::run_report` builds selected modules,
  invokes each package's `AxiomReport.lean`, checks report coverage, and fails
  on `sorryAx`. It deliberately reports rather than rejects other axioms.
  `formal/AxiomReport.lean` selects specific EL+, linker, and TypeChecker
  theorems; it is not an all-theorems axiom audit. The draft's inventory and
  footprint distinction is correct.

No standalone BLOCKER is asserted beyond the four MAJOR documentation
corrections. In particular, this review does not infer proof failure, current
gate failure, or a hardware result from source inspection. The ontology
wrapper behaviors are serious obstacles to treating its outputs as semantic
or execution evidence and must be visible before such claims are made.

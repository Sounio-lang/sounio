<!-- docs:meta
topic_id: repo.docs.serious-language.conformance-spine
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.conformance-spine
-->

# Serious-Language Conformance Spine

> **Status**: Bounded conformance gate | **Operational check**: 2026-05-12 | **Source**: `tests/conformance/manifest.v1.tsv` and `scripts/ci/serious_language_conformance_gate.sh`

The conformance spine is the first executable bridge between the readiness ledger, the language specification, and the checked compiler artifact. It is intentionally small: every case must have a named claim, an expected exit contract, and optional stdout or stderr evidence.

It is not a complete language specification suite. It is the conference-safe seed that lets Sounio say, "these selected language claims are executable at this commit."
The spec/evidence matrix in `docs/serious-language/spec-evidence-matrix.v1.tsv` is the v1 seed map from tracked specification areas to evidence. `scripts/ci/serious_language_spec_drift_gate.sh` enforces that tracked executable spec rows cite live evidence, runs the bounded conformance gate, and verifies cited conformance cases pass.
The public-claim registry in `docs/serious-language/public-claim-registry.v1.tsv`, doc-surface map in `docs/serious-language/doc-claim-surface.v1.tsv`, and exact-line annotations in `docs/serious-language/claim-line-annotations.v1.tsv` close the remaining loop: public repo docs must either cite closed evidence-backed claims or carry an explicit downgraded/internal/historical status through `scripts/ci/serious_language_claim_closure_gate.sh`.

## Entry Point

Run the gate directly:

```bash
bash scripts/ci/serious_language_conformance_gate.sh
```

Run the companion spec/evidence drift gate:

```bash
bash scripts/ci/serious_language_spec_drift_gate.sh
```

Run the public claim-closure gate:

```bash
bash scripts/ci/serious_language_claim_closure_gate.sh
```

The gate writes:

| Artifact | Purpose |
|---|---|
| `RESULTS.md` | Human-readable case table and totals. |
| `summary.v1.tsv` | Machine-readable pass/fail summary. |
| `summary.v1.json` | JSON summary for paper-bundle ingestion. |
| `logs/*.stdout` and `logs/*.stderr` | Raw compiler outputs per case. |

Override inputs only deliberately:

```bash
SOUNIO_SERIOUS_CONFORMANCE_MANIFEST=tests/conformance/manifest.v1.tsv \
SOUNIO_SERIOUS_CONFORMANCE_SOUC_BIN=./bin/souc \
SOUNIO_SERIOUS_CONFORMANCE_STDLIB_PATH=./stdlib \
bash scripts/ci/serious_language_conformance_gate.sh
```

Standalone runs default to `/tmp/sounio-serious-conformance-<timestamp>`. The conference bundle overrides the artifact root and passes its pinned compiler and stdlib paths into the gate.

## Claim Coverage

| Claim ID | Readiness ledger row | Spec anchor | Seed cases |
|---|---|---|---|
| `core.syntax` | Core syntax, functions, structs, control flow | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 2, 4, 5, 6 | `core-hello-check` |
| `core.execution` | Linux x86-64 native compile/run | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 4, 6 | `core-hello-run` |
| `core.structs` | Core syntax, functions, structs, control flow | `docs/spec/LANGUAGE_SPECIFICATION.md` section 6.2 | `core-struct-run` |
| `effects.subtyping` | Effects | `docs/spec/LANGUAGE_SPECIFICATION.md` section 7 | `effects-superset-run` |
| `effects.diagnostics` | Effects | `docs/spec/LANGUAGE_SPECIFICATION.md` section 7 | `effects-missing-diagnostic` |
| `epistemic.observe` | Epistemic `Knowledge` and GUM propagation | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 3.9 and 7 | `observe-io-boundary` |
| `modules.imports` | Modules/imports | `docs/spec/LANGUAGE_SPECIFICATION.md` section 6.7 | `modules-import-check` |
| `generics.structs` | Traits and generics | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 3.7 and 6.2 | `generics-struct-run` |
| `generics.functions` | Traits and generics | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 3.7 and 6.1 | `generics-multi-run` |
| `ownership.borrowing` | Ownership and borrowing | `docs/spec/LANGUAGE_SPECIFICATION.md` section 8 | `ownership-release-check`, `ownership-conflict-diagnostic` |
| `epistemic.gum` | Epistemic `Knowledge` and GUM propagation | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 3.9 and 12.5 | `gum-compliance-run`, `gum-iso-budget-run` |
| `epistemic.knowledge` | Epistemic `Knowledge` and GUM propagation | `docs/spec/LANGUAGE_SPECIFICATION.md` section 3.9 | `epistemic-bmi-run` |
| `epistemic.boundary` | Epistemic `Knowledge` and GUM propagation | `docs/spec/LANGUAGE_SPECIFICATION.md` sections 3.9 and 7 | `knowledge-boundary-diagnostic`, `epistemic-effect-diagnostic` |

## Expansion Rules

- Add cases by extending `tests/conformance/manifest.v1.tsv`; do not hide expected failures in the runner.
- Prefer existing run-pass and compile-fail fixtures until a dedicated normative fixture is needed.
- A new stable public claim needs at least one positive case and, where applicable, one diagnostic or boundary case.
- Prototype areas may enter the manifest only as explicit boundary cases until their readiness-ledger level is raised.
- The language specification is not normative until a section has executable conformance coverage and the readiness ledger permits that wording.
- A spec area may be called executable only when its matrix row cites live evidence and the drift gate verifies any cited conformance behavior passed.
- A public repo doc may carry a PL claim only when its doc-surface rule points to registered claim IDs whose evidence is closed or explicitly downgraded.
- A high-value public PL claim should be added to `claim-line-annotations.v1.tsv` with an exact line number and anchor text so drift is caught.

## Current Limits

- The corpus checks a bounded behavioral spine, not the whole compiler.
- It does not prove parity between `lean_single.sio` and the modular compiler tree.
- It does not make GPU, ontology federation, package-registry, editor-tooling, or full stdlib claims.
- It does not replace the full suite, self-host gates, Lean proof status audit, or paper-bundle review.

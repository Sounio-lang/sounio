<!-- docs:meta
topic_id: repo.docs.serious-language.compiler-maturity-snapshot
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.compiler-maturity-snapshot
-->

# Compiler Maturity Snapshot

> **Status**: Research readiness | **Operational check**: 2026-05-11 | **Source**: checked compiler artifact and current maturity docs

This snapshot states what Sounio needs to become legible as a serious programming language.

## Current Serious Core

Sounio can already be presented as a serious research language because it has:

- an official checked compiler launcher: `bin/souc` (Madaros by default);
- a host-selected Linux x86-64 self-hosted binary in this checkout;
- working `check`, `compile`, `build`, `run`, `info`, and `version` compatibility commands;
- a native compile/run path suitable for small live demos;
- epistemic type and GUM propagation surfaces with gates;
- effect and observation boundaries with diagnostic tests;
- a substantial formal Lean tree;
- CI jobs for contracts, self-host gates, full suite, Lean, lint, and website checks;
- explicit honest-status documentation naming scaffolds and missing pieces.

This is enough for a real-world-defensible statement:

> Sounio is a serious active research language with a validated core, not a finished production ecosystem.

## Main Maturity Gaps

The work needed for serious programming-language credibility is mostly consolidation:

| Gap | Why it matters | Required next step |
|---|---|---|
| Claim discipline | Reviewers will attack any mismatch between docs and executable behavior. | Keep `docs/serious-language/readiness-ledger.md` current and require it for papers/slides. |
| Source-of-truth split | The checked binary still depends on `lean_single.sio`; modular sources can drift. | Add a parity corpus comparing checked behavior against modular intent before source-swap claims. |
| Reproducibility | Papers need commands and logs, not only prose. | Use `scripts/paper/build_serious_language_bundle.sh` for every conference artifact. |
| Language specification | A serious language needs a spec that matches behavior. | Audit `docs/spec/LANGUAGE_SPECIFICATION.md` against run-pass and compile-fail cases. |
| Conformance corpus | Reviewer trust improves when behavior is executable. | Maintain and expand `tests/conformance/manifest.v1.tsv` through `scripts/ci/serious_language_conformance_gate.sh`. |
| Spec drift | Engineers need to know which spec claims are executable today. | Keep `docs/serious-language/spec-evidence-matrix.v1.tsv` green through `scripts/ci/serious_language_spec_drift_gate.sh`. |
| Public claim closure | Engineers need to know that docs cannot silently overclaim PL support. | Keep `docs/serious-language/public-claim-registry.v1.tsv`, `docs/serious-language/doc-claim-surface.v1.tsv`, `docs/serious-language/claim-line-annotations.v1.tsv`, and `scripts/ci/serious_language_claim_closure_gate.sh` green. |
| Formal status | Mixed proof status can be misread as all-proven or all-scaffold. | Generate and publish a Lean `sorry`/`axiom` status table per paper bundle. |
| Tooling/install | Adoption discussions need one boring path. | Use checked-artifact install instructions first; defer broad package promises. |
| Research boundaries | GPU, hypercomplex, ontology, and clinical surfaces have different maturity. | Name gate, hardware, and scope for each claim. |

## Recommended Work Order

1. Lock the evidence language.
2. Keep the spec/evidence matrix green: every `executable` or `partially_executable` row must cite live repo evidence and pass `scripts/ci/serious_language_spec_drift_gate.sh`.
3. Generate a paper bundle from current commands.
4. Refresh paper prose to cite only ledger-backed claims.
5. Expand the conformance spine and spec-drift coverage.
6. Only then expand public feature claims.

## Demo Contract

The default live demo should be intentionally small:

```bash
./bin/souc --version
./bin/souc info
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

Then show one epistemic/GUM example, one compile-fail/effect example, and the bounded conformance summary from `scripts/ci/serious_language_conformance_gate.sh`.
For real-world credibility, also show `scripts/ci/serious_language_spec_drift_gate.sh` so the spec boundary is explicit.

Avoid live-demonstrating:

- GPU runtime;
- ontology federation;
- scaffolded stdlib areas;
- large-surface direct-driver closure;
- any paper result whose artifact bundle has not been generated and reviewed.

## Engineering North Star

The next serious-language milestone is not a new feature. It is a reproducible statement of truth:

> Given commit X, command pack Y, and artifact bundle Z, the Sounio team can say exactly which language, compiler, formal, and scientific claims are supported.

## Non-Negotiable Blockers Before Strong PL Submission

- A parity/conformance gate must fence the `lean_single.sio` and modular compiler split.
- The language specification must be tied to broader executable conformance tests.
- The spec/evidence matrix must stay green before any spec area is called executable.
- The generated paper bundle must include binary hashes and self-host gate logs.
- Formal claims must cite exact Lean `sorry`/`axiom` status from the generated bundle.

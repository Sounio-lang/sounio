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

- a checked compiler launcher: `bin/souc`;
- a host-selected Linux x86-64 self-hosted binary in this checkout;
- working `check`, `compile`, `build`, `run`, `info`, and `version` compatibility commands;
- a native compile/run path suitable for small live demos;
- epistemic type and GUM propagation surfaces with gates;
- effect and observation boundaries with diagnostic tests;
- a substantial formal Lean tree;
- CI jobs for contracts, self-host gates, full suite, Lean, lint, and website checks;
- explicit honest-status documentation naming scaffolds and missing pieces.

This is enough for a conference-safe statement:

> Sounio is a serious active research language with a validated core, not a finished production ecosystem.

## Main Maturity Gaps

The work needed for serious programming-language credibility is mostly consolidation:

| Gap | Why it matters | Required next step |
|---|---|---|
| Claim discipline | Reviewers will attack any mismatch between docs and executable behavior. | Keep `docs/serious-language/readiness-ledger.md` current and require it for papers/slides. |
| Source-of-truth split | The checked binary still depends on `lean_single.sio`; modular sources can drift. | Add a parity corpus comparing checked behavior against modular intent before source-swap claims. |
| Reproducibility | Papers need commands and logs, not only prose. | Use `scripts/paper/build_serious_language_bundle.sh` for every conference artifact. |
| Language specification | A serious language needs a spec that matches behavior. | Audit `docs/spec/LANGUAGE_SPECIFICATION.md` against run-pass and compile-fail cases. |
| Conformance corpus | Reviewer trust improves when behavior is executable. | Promote a small `tests/conformance/` corpus with expected stdout/stderr/exit codes. |
| Formal status | Mixed proof status can be misread as all-proven or all-scaffold. | Generate and publish a Lean `sorry`/`axiom` status table per paper bundle. |
| Tooling/install | Adoption discussions need one boring path. | Use checked-artifact install instructions first; defer broad package promises. |
| Research boundaries | GPU, hypercomplex, ontology, and clinical surfaces have different maturity. | Name gate, hardware, and scope for each claim. |

## Recommended Work Order

1. Lock the evidence language.
2. Generate a paper bundle from current commands.
3. Refresh paper prose to cite only ledger-backed claims.
4. Add conformance and spec-drift gates.
5. Only then expand public feature claims.

## Demo Contract

The default live demo should be intentionally small:

```bash
./bin/souc --version
./bin/souc info
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

Then show one epistemic/GUM example and one compile-fail/effect example from the checked corpus.

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
- The language specification must be tied to executable conformance tests.
- The generated paper bundle must include binary hashes and self-host gate logs.
- Formal claims must cite exact Lean `sorry`/`axiom` status from the generated bundle.

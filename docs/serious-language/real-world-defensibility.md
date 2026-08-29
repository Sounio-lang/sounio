<!-- docs:meta
topic_id: repo.docs.serious-language.real-world-defensibility
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.real-world-defensibility
-->

# Sounio Real-World PL Defensibility

> **Status**: Real-world defensibility v1 | **Operational check**: 2026-05-12 | **Source**: readiness ledger, conformance spine, spec evidence matrix, and generated evidence bundle

This page is the repo contract for describing **Sounio** as a real programming language.
The goal is not to make every ambitious surface sound finished. The goal is to make the current language honest, runnable, inspectable, and defensible to a skeptical engineer.

## Defensible Statement

Sounio is a self-hosted research programming language for epistemic and scientific computing. It has a checked compiler launcher, Linux x86-64 compile/run behavior, a bounded executable conformance spine, epistemic/GUM examples, effect and ownership diagnostics, formal proof surfaces, and reproducibility bundles that tie claims to commands and logs.

Sounio is not yet a finished production ecosystem. Claims about broad stdlib coverage, general GPU runtime support, complete package management, polished editor tooling, or all-formalized semantics must stay scoped to the exact gates and artifacts that support them.

## First-User Path

A new engineer should be able to verify the basic compiler path first:

```bash
./bin/souc --version
./bin/souc info
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

With Bash, Git, Python 3, and a complete checkout, the same engineer can verify the evidence path:

```bash
bash scripts/ci/serious_language_conformance_gate.sh
bash scripts/ci/serious_language_spec_drift_gate.sh
bash scripts/ci/serious_language_claim_closure_gate.sh
```

The primary checked environment for this path is Linux x86-64. The bundle records exact tool versions for each run; use `docs/guide/MINIMUM_VIABLE_SOUNIO.md` for the broader current support boundary.

For a reproducibility bundle:

```bash
bash scripts/paper/build_serious_language_bundle.sh
```

For the heavier evidence lane:

```bash
SOUNIO_SERIOUS_LANGUAGE_FULL=1 bash scripts/paper/build_serious_language_bundle.sh
```

## What Is Currently Defensible

| Surface | Current level | Required evidence |
|---|---|---|
| Compiler entrypoint | Stable | `./bin/souc --version`, `./bin/souc info`, binary hashes in bundle manifest |
| Small native programs | Stable | `examples/hello.sio`, conformance cases, run-pass fixtures |
| Effects and diagnostics | Stable | conformance positive and compile-fail diagnostic cases |
| Ownership and borrowing | Validated research | checked examples and diagnostic cases |
| Modules/imports | Validated research | import conformance and resolver docs |
| Epistemic `Knowledge` and GUM | Validated research | conformance cases plus PBPK/GUM and stdlib science gates |
| Formal Lean surface | Validated research | `scripts/ci/lean_proof_status_audit.py` and host-specific `lake build` status |
| GPU/PTX | Prototype for broad language claims; validated only for named backend gates | named GPU gates and exact hardware logs only |
| Ontology | Validated research | rebuilt ontology validation gate only |
| Package manager, REPL, LSP, broad stdlib | Prototype | describe as prototype unless a named gate covers the exact behavior |

## Spec And Implementation Discipline

The language specification is not automatically a public support promise. The v1 matrix is a seed map, not full spec coverage. Each tracked spec area in `docs/serious-language/spec-evidence-matrix.v1.tsv` must be classified as:

- `executable`: directly backed by conformance cases, tests, or gates.
- `partially_executable`: a supported slice exists, but the full section is broader than the checked behavior.
- `prototype`: useful implementation exists, but the surface is not mature enough for broad public claims.
- `specified_only`: the spec describes intended behavior that is not yet implemented or not yet checked.
- `stale_conflicting`: docs disagree or behavior needs refresh before citation.

`scripts/ci/serious_language_spec_drift_gate.sh` enforces the minimum rule: rows marked `executable` or `partially_executable` must cite live repo evidence. For conformance evidence, the gate runs the bounded conformance gate and requires cited cases to pass.

`scripts/ci/serious_language_claim_closure_gate.sh` adds the public-claim closure rule. Every claim ID cited by the conformance manifest or spec/evidence matrix must appear in `docs/serious-language/public-claim-registry.v1.tsv`, every repo doc under `README.md`, `INSTALL.md`, or `docs/` must be covered by `docs/serious-language/doc-claim-surface.v1.tsv` as public, downgraded, internal, or historical, and high-value public claims must have exact anchors in `docs/serious-language/claim-line-annotations.v1.tsv`. A public PL claim is acceptable only when it has passing evidence or an explicit downgraded status.

Evidence kinds in the matrix are deliberately small:

| Evidence kind | Meaning |
|---|---|
| `conformance_case` | `evidence_ref` is one exact case ID in `tests/conformance/manifest.v1.tsv`. |
| `conformance_claim` | `evidence_ref` is one claim ID covered by one or more passing conformance cases. |
| `test` | `evidence_ref` is a concrete test or example path. |
| `gate` | `evidence_ref` is a repo gate script whose output must be cited before broad claims. |
| `doc` | `evidence_ref` is an honest-status or limitations document, not runtime proof. |

The drift gate checks references and passing bounded conformance behavior. Broader semantic adequacy still depends on the readiness ledger, the conformance spine, and the generated bundle; the matrix is traceability evidence, not a proof of the whole language.
The claim-closure gate checks that the repo has no unregistered public claim surface in the governed docs set, and that registered high-value claim annotations still point at the intended line text.

## What Must Not Be Claimed

- Do not claim Sounio is production-ready as a general-purpose language.
- Do not claim all spec sections are implemented and tested.
- Do not claim the modular compiler tree is the checked binary source of truth until parity gates prove it.
- Do not claim broad GPU runtime or performance support without exact hardware and gate logs.
- Do not claim all Lean modules are axiom-free or sorry-free without the generated audit.
- Do not treat scaffolded stdlib or research modules as user-ready features.

## Real-Life Bar

Sounio is defensible when a skeptical engineer can:

1. run the first-user path;
2. inspect the conformance summary;
3. see which spec areas are executable and which are not;
4. read known limitations without private explanation;
5. reproduce the bundle for the current commit;
6. distinguish stable language behavior from research lanes;
7. verify that public docs are either evidence-closed or explicitly downgraded;
8. inspect exact-line annotations for the highest-value public PL claims.

### Compiler Identity for Defensibility Conversations

For this repository's default public workflow, use `./bin/souc` as the official compiler entry point; it routes to the Madaros checked engine by default. The repo keeps a separate `lean_single` legacy compatibility path for explicit bootstrap/compatibility checks.

<!-- docs:meta
topic_id: repo.docs.serious-language.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.readme
-->

# Serious-Language Readiness

> **Status**: Research readiness | **Operational check**: 2026-05-11 | **Source**: repo gates, checked compiler artifact, and claim ledger

This directory is the real-world programming-language defensibility control point for Sounio: checked compiler path, executable conformance, spec evidence, epistemic core, reproducibility gates, formal surfaces, and explicit limits.

The purpose is not to make every ambitious part of the repository sound finished. The purpose is to make the serious core legible, reproducible, and hard to overclaim.

## Use This Package For

- preparing real-world engineering and international conference conversations about Sounio's programming-language maturity;
- briefing collaborators who need to understand what is implemented versus researched;
- building a paper artifact bundle whose claims can be traced to commands and logs;
- keeping reviewer-facing and collaborator-facing material aligned with executable repo truth.

## Documents

| File | Purpose |
|---|---|
| [real-world-defensibility.md](real-world-defensibility.md) | Canonical real-life PL defensibility contract and first-user path. |
| [readiness-ledger.md](readiness-ledger.md) | Claim-to-evidence ledger and allowed public wording. |
| [compiler-maturity-snapshot.md](compiler-maturity-snapshot.md) | Current compiler contract, gaps, and serious-language worklist. |
| [conformance-spine.md](conformance-spine.md) | Bounded executable bridge from claims to compiler behavior. |
| [paper-bundle.md](paper-bundle.md) | Paper/reproducibility bundle structure and acceptance criteria. |
| [sunil-brief.md](sunil-brief.md) | Short, honest briefing frame for a senior PL/science conversation. |
| [spec-evidence-matrix.v1.tsv](spec-evidence-matrix.v1.tsv) | Machine-checkable spec area to evidence map used by the spec-drift gate. |
| [public-claim-registry.v1.tsv](public-claim-registry.v1.tsv) | Machine-checkable public PL claim registry with closure status and evidence rules. |
| [doc-claim-surface.v1.tsv](doc-claim-surface.v1.tsv) | Machine-checkable coverage map for public, downgraded, internal, and historical repo docs. |
| [claim-line-annotations.v1.tsv](claim-line-annotations.v1.tsv) | Machine-checkable exact-line anchors for high-value public PL claims. |

## Reproducibility Entry Point

Generate a timestamped evidence bundle:

```bash
bash scripts/paper/build_serious_language_bundle.sh
```

Run the heavier conference artifact lane:

```bash
SOUNIO_SERIOUS_LANGUAGE_FULL=1 bash scripts/paper/build_serious_language_bundle.sh
```

The script writes logs, a manifest, and `RESULTS.md` under `artifacts/conference-serious-language/<timestamp>/`.

Run just the bounded conformance spine:

```bash
bash scripts/ci/serious_language_conformance_gate.sh
```

Run the spec/evidence drift gate:

```bash
bash scripts/ci/serious_language_spec_drift_gate.sh
```

Run the public claim-closure gate:

```bash
bash scripts/ci/serious_language_claim_closure_gate.sh
```

The script pins `SOUC_BIN` to this checkout's `bin/souc` by default. Override only deliberately:

```bash
SOUNIO_SERIOUS_LANGUAGE_SOUC_BIN=/path/to/souc bash scripts/paper/build_serious_language_bundle.sh
```

The script also pins `SOUNIO_STDLIB_PATH` to this checkout's `stdlib` by default. Use `SOUNIO_SERIOUS_LANGUAGE_STDLIB_PATH` only when the bundle is deliberately testing another stdlib tree.

## Public Posture

The safe real-world line is:

> Sounio is an active research programming language for epistemic and scientific computing. It has a checked self-hosted compiler path, native compile/run evidence, epistemic type machinery, formal proof surfaces, and reproducibility gates. It is not yet a general production language; the serious-language work is to make every public claim evidence-backed.

Do not use this package to imply:

- all stdlib modules are callable;
- GPU runtime paths are generally production-ready;
- every Lean theorem is axiom-free or sorry-free;
- the modular compiler tree is already the source used to build the checked binary;
- scaffolded research modules are user-ready features.

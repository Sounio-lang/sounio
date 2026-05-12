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

This directory is the conference and paper-readiness control point for Sounio as a serious research programming language: checked compiler path, epistemic core, reproducibility gates, formal surfaces, and explicit limits.

The purpose is not to make every ambitious part of the repository sound finished. The purpose is to make the serious core legible, reproducible, and hard to overclaim.

## Use This Package For

- preparing international conference conversations about Sounio's programming-language maturity;
- briefing collaborators who need to understand what is implemented versus researched;
- building a paper artifact bundle whose claims can be traced to commands and logs;
- keeping Sunil-facing or reviewer-facing material aligned with executable repo truth.

## Documents

| File | Purpose |
|---|---|
| [readiness-ledger.md](readiness-ledger.md) | Claim-to-evidence ledger and allowed public wording. |
| [compiler-maturity-snapshot.md](compiler-maturity-snapshot.md) | Current compiler contract, gaps, and serious-language worklist. |
| [paper-bundle.md](paper-bundle.md) | Paper/reproducibility bundle structure and acceptance criteria. |
| [sunil-brief.md](sunil-brief.md) | Short, honest briefing frame for a senior PL/science conversation. |

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

The script pins `SOUC_BIN` to this checkout's `bin/souc` by default. Override only deliberately:

```bash
SOUNIO_SERIOUS_LANGUAGE_SOUC_BIN=/path/to/souc bash scripts/paper/build_serious_language_bundle.sh
```

## Public Posture

The safe conference line is:

> Sounio is an active research programming language for epistemic and scientific computing. It has a checked self-hosted compiler path, native compile/run evidence, epistemic type machinery, formal proof surfaces, and reproducibility gates. It is not yet a general production language; the serious-language work is to make every public claim evidence-backed.

Do not use this package to imply:

- all stdlib modules are callable;
- GPU runtime paths are generally production-ready;
- every Lean theorem is axiom-free or sorry-free;
- the modular compiler tree is already the source used to build the checked binary;
- scaffolded research modules are user-ready features.

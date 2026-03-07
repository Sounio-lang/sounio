<!-- docs:meta
topic_id: repo.contributor.codebase-overview
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.contributor.codebase-overview
-->

# Sounio Codebase Overview

This document maps the active parts of the Sounio monorepo so contributors can quickly find ownership boundaries and release-critical paths.

## 1. Repository Layers

- `crates/souc/`: Main Rust compiler (`souc`) including parser, typechecker, backend orchestration, CLI commands, and compiler-facing tests.
- `stdlib/`: Sounio standard library and bootstrap components, including the self-host bootstrap driver under `stdlib/compiler/bootstrap/`.
- `tests/`: Language and integration fixtures (`run-pass`, `ui`, `compile-fail`, `error_audit`, `stdlib`, and regression suites).
- `website/`: Astro site for docs, tutorials, releases, and multilingual pages used for publish readiness.
- `docs/`: Contributor and architecture documentation.
- `scripts/`: CI/release gate scripts (fast gate, docs consistency, docs registry verification, workflow script reference checks, and isolated diagnostic wrappers).
- `.github/workflows/`: CI workflows that enforce compiler, docs, and website quality contracts.

## 2. Compiler Runtime and Bootstrap Flow

- CLI entrypoints live in `crates/souc/src/bin/` and dispatch into compiler services in `crates/souc/src/`.
- Frontend and semantic phases (lexing/parsing, typing, diagnostics) are implemented in `crates/souc/src/` modules and validated by lib/unit tests.
- Self-host and bootstrap integration spans Rust loader/orchestration code (`crates/souc/src/compiler_loader.rs`) and Sounio bootstrap driver code (`stdlib/compiler/bootstrap/driver.sio`).
- Release-critical strict driver tests validate source and file compilation paths under self-host strict mode.

## 3. Test and Gate Surfaces

- `cargo test -p souc --lib`: Compiler library safety net (release-blocking).
- `scripts/fast_gate.sh`: Preflight that runs syntax drift scan, workflow/docs checks, compiler tests, and e2e gate.
- `scripts/check_workflow_script_refs.sh`: Prevents workflow drift by failing on missing `scripts/*` references.
- `scripts/check_docs_consistency.sh`: Prevents stale path/version statements in key docs.
- `scripts/check_docs_registry.sh`: Verifies the machine-readable docs registry, hidden metadata blocks/frontmatter, front-door/current-canon repo-doc links, and historical status labeling.
- `docs/governance/`: Generated authority and acceptance surfaces (`DOCS_AUTHORITY_MATRIX.md`, `DOCS_ACCEPTANCE_REPORT.md`, and `topic-registry.v1.json`) for docs ownership, parity, and merge readiness.
- `website` quality gate (`npm --prefix website run check:quality`): Enforces build, redirects, i18n keys, nav integrity, and search indexing.

## 4. Website Routing Model

- Primary content routes are under `website/src/pages/`.
- English canonical docs/tutorial content is rendered from collections in `website/src/content/`.
- Localized routes under `website/src/pages/[lang]/` reuse canonical content and redirect where full localization is not yet available.
- Navigation integrity is checked against generated `dist/**/*.html`, so all internal links must resolve to real routes.

## 5. Ownership Pointers

- Compiler correctness and self-hosting: `crates/souc/` + `stdlib/compiler/bootstrap/`.
- Docs and contributor guidance: `README.md`, `INSTALL.md`, `docs/`, `tests/README.md`.
- Website publishing and routing quality: `website/`.
- CI and release guardrails: `.github/workflows/` + `scripts/`.

## 6. Release-Critical Checklist (Compiler + Docs + Website)

1. `cargo check -p souc` passes.
2. Strict self-host source/file pipeline tests pass.
3. `cargo test -p souc --lib` completes without aborts.
4. `bash scripts/fast_gate.sh` passes.
5. `npm --prefix website run check:quality` passes.
6. Workflow/doc integrity checks, including `bash scripts/check_docs_registry.sh`, pass in CI and locally.

Use this overview as the source-of-truth map for release readiness and regression triage.

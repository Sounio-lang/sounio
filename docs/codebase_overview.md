<!-- docs:meta
topic_id: repo.contributor.codebase-overview
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.contributor.codebase-overview
-->

# Sounio Codebase Overview

This overview maps the active parts of the Sounio repository as it exists now. The short version is:

- users and docs validate behavior through the checked compiler artifact under `artifacts/omega/`
- contributors should treat `self-hosted/` as the primary implementation tree
- Rust-side crates still matter for packaging, bridging, and release machinery, but they are no longer the best first explanation of where most language work lives

## 1. Repository layers

- `artifacts/`: checked outputs and gate records. The most important surfaces are `artifacts/omega/` for compiler binaries and `artifacts/stdlib/` for committed status JSON.
- `self-hosted/`: the active compiler and runtime implementation map. This is where lexer, parser, checker, IR, native, WASM, GPU, LLVM, LSP, bootstrap, and tooling work live today.
- `stdlib/`: shipped library surface plus modules that range from active to stubbed to disabled. Trust the gate artifacts, not directory presence alone.
- `tests/`: run-pass, compile-fail, stdlib, fixtures, and regression coverage. This is the best executable evidence for language behavior that is supposed to work now.
- `website/`: Astro site for the public docs domain, learn hub, localized pages, redirects, and search indexing.
- `docs/`: repo-native contributor documentation, architecture notes, governance metadata, and evidence-oriented guides.
- `scripts/`: gate entry points, docs checks, stdlib validation scripts, release helpers, and artifact probes.
- `crates/`: Rust-side infrastructure used for build orchestration, packaging, and bridge code. Important when you are changing artifact production, but not the primary implementation map for most compiler subsystems.

## 2. Current compiler map

The current compiler story is self-hosted-first:

```text
source (.sio)
  -> self-hosted/lexer/
  -> self-hosted/parser/
  -> self-hosted/resolve/
  -> self-hosted/check/
  -> self-hosted/hlir/ and self-hosted/ir/
  -> self-hosted/native/ | self-hosted/wasm/ | self-hosted/gpu/ | self-hosted/llvm/
  -> artifact packaging / runtime execution
```

Practical entry points:

- `self-hosted/compiler/main.sio`: compiler-driver view
- `self-hosted/compiler/module_loader.sio`: module and loading orchestration
- `self-hosted/check/`: semantic core for types, effects, epistemics, units, ownership, patterns, traits, refinements, and related checks
- `self-hosted/hlir/` and `self-hosted/ir/`: lowering and optimization-facing representations
- `self-hosted/native/`, `self-hosted/wasm/`, `self-hosted/gpu/`, `self-hosted/llvm/`: backend-specific work
- `self-hosted/lsp/`, `self-hosted/tools/`, `self-hosted/bootstrap/`: tooling and bootstrap surfaces

The checked public artifact used by the default website docs path is:

```bash
bin/souc
```

On the current repository snapshot, `souc info` for that JIT artifact reports:

- version `1.0.0-beta.4`
- **Cranelift JIT NOT compiled** — `info` prints `[-] Cranelift JIT - rebuild with
  --features jit`. There is no JIT artifact and no build path enables the feature;
  earlier revisions of this list claimed it was enabled (measured 2026-08-27)
- LLVM not compiled, same shape
- GPU codegen enabled on the `-gpu` artifact
- LSP, SMT, distributed, and package-manager features disabled in the checked artifact
- ontology resolution CLI enabled natively via `souc ontology <resolve|search|ancestors|is-subclass>`

The repository also ships a separate checked GPU artifact:

```bash
artifacts/omega/souc-bin/souc-linux-x86_64-gpu
```

That GPU profile reports GPU codegen enabled, JIT disabled, and public PTX
emission through `build --backend gpu`.

That means contributors should distinguish clearly between:

- source-tree presence: a subsystem exists in `self-hosted/`
- checked-artifact availability: the default binary actually exposes that subsystem

## 3. Current checkpoint lanes

The current repo-wide checkpoint is broader than the older Sprint 52-only
baseline.

- optimizer lanes: Sprints `43`, `44`, `50`, `51`, `52`
- render/bootstrap lanes: Sprints `53`, `54`, `55`, `56`, `57`, `58`
- skills/dispatch lanes: Sprints `59`, `60`, `61`, `65`, `66`
- authoritative self-hosted driver: `self-hosted/compiler/main.sio`
- frozen self-hosted CLI surface: `--check`, `--ir-dump`, `--ir-roundtrip`,
  `--native-compile`
- website graphics evidence: 5 checked-JIT raster previews generated from
  `examples/render/`

## 4. Evidence surfaces that matter

When documentation or code claims support, confirm it against one of these:

- `tests/run-pass/` for behavior expected to succeed now
- `tests/compile-fail/` for refusal paths that are expected now
- `tests/stdlib/` for library contracts and science lanes
- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- `artifacts/omega/` gate artifacts when discussing GPU or artifact packaging

Current committed stdlib status from `artifacts/stdlib/stdlib_reliability_status.v1.json`:

- totals: `pass=251 fail=0 skip=0 total=251`
- inventory: `927` `.sio` files, `0` disabled files, `0` stub module files, `119` active module entrypoints
- science pipeline: `pass`, with `2/2` required lanes passing
- hyper execution: `7/7` required lanes passing
- runtime regression enforcement for science remains `soft` locally, with `0` recorded regression failures that become release-blocking under strict enforcement

## 5. Website and docs model

- Public docs content lives under `website/src/content/docs/`.
- `/learn/*` is the canonical public docs surface.
- `/docs/*` remains a redirect layer to `/learn/*`.
- Repo-native deep dives live under `docs/`, but not every historical architecture note should be treated as current without checking validation dates and the active tree.

If you are updating docs, keep the public and repo-native stories aligned:

- website docs should explain the checked artifact and user-facing contract
- repo docs should explain the self-hosted implementation map and contributor workflow

## 6. Release-critical paths

The most important paths to protect when changing the compiler or docs are:

- `artifacts/omega/souc-bin/` and scripts that resolve the pinned compiler binary
- `self-hosted/check/`, `self-hosted/hlir/`, `self-hosted/ir/`, and backend directories you touched
- `tests/run-pass/`, `tests/compile-fail/`, and affected stdlib tests
- `website/src/content/docs/`, `website/src/pages/learn/`, and docs/i18n support code for docs-domain changes
- `docs/governance/` metadata and `scripts/dev/check_docs_registry.sh` for docs governance

## 7. Current contributor checklist

Use a checklist that matches the part of the repository you changed:

1. Validate the checked artifact still reports the features your docs or tests depend on.
2. Run representative `souc check` fixtures for the language or stdlib area you changed.
3. Read the relevant committed status JSON before claiming a lane is reliable.
4. Run website checks for docs and navigation changes.
5. Run targeted Rust or packaging checks only if you changed the Rust-side bridge or artifact machinery.

This file is the top-level contributor map. For subsystem details, continue with:

- `docs/compiler/COMPILER_ARCHITECTURE_OVERVIEW.md`
- `docs/implementation/SELF_HOSTED_COMPILER.md`
- `docs/implementation/TOOLING_SUMMARY.md`
- `docs/guide/MINIMUM_VIABLE_SOUNIO.md`

<!-- docs:meta
topic_id: repo.docs.internal.implementation.tooling-summary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.tooling-summary
-->

# Sounio Tooling Summary

This document summarizes the current toolchain around Sounio as it actually ships today. The important distinction is:

- the public and docs-facing workflow is artifact-first
- deeper bootstrap and IR inspection tooling still exists for contributors, but it is not the default onboarding path

## 1. Default public workflow

For user-facing docs and ordinary verification, start with the checked JIT artifact:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
```

Why this is the default:

- it matches the artifact the website docs validate against
- it avoids overpromising features that are only present in source form
- it gives you the exact backend and feature toggles for the binary you are documenting

On the current snapshot, that artifact reports:

- `souc 1.0.0-beta.4`
- Cranelift JIT enabled
- LLVM and GPU codegen disabled
- LSP, SMT, distributed, and package-manager features disabled
- ontology resolution CLI enabled natively via `souc ontology <resolve|search|ancestors|is-subclass>`

For GPU-specific verification, use the separate checked GPU artifact:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

## 2. Core verification commands

Use these as the baseline commands when you need to confirm docs or contributor claims:

```bash
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" check tests/run-pass/covid_2020_kernel.sio
"$SOUC_BIN" check tests/run-pass/vancomycin_propagation.sio
"$SOUC_BIN" check tests/compile-fail/vancomycin_low_conf.sio
```

Then read the committed gate artifacts:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`

Current committed status:

- stdlib reliability: `81/82` pass with `0` failures
- science pipeline: `2/2` required lanes pass
- hyper execution: `7/7` required lanes pass
- science runtime regressions are still recorded separately and currently show `4` failures under soft local enforcement

## 3. Docs and website tooling

For docs-domain work, the active quality gates live in `website/package.json`:

```bash
npm --prefix website run check:docs-parity
npm --prefix website run check:i18n
npm --prefix website run build
npm --prefix website run check:routes
npm --prefix website run check:redirects
npm --prefix website run check:nav
npm --prefix website run check:pagefind
npm --prefix website run check:locale-fallback
```

For repo-native docs governance, use:

```bash
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
```

## 4. Bootstrap and IR inspection tooling

Contributor-only tooling still exists for deeper work:

- `scripts/sounio-verify`: SOIR inspection and comparison helper
- `Makefile.verify`: local convenience targets for multi-stage verification experiments
- `self-hosted/ir/`: serializer, normalizer, disassembler, and IR helpers
- `self-hosted/test_ir.sio`: self-hosted IR-facing verification surface

Important caveats:

- `scripts/sounio-verify` targets `target/release/souc`, not the checked artifact under `artifacts/omega/`
- it will try to build a Rust release binary if the expected local binary is missing
- this makes it useful for maintainers working on bootstrap and IR internals, but not the right default for end-user docs

## 5. When Rust tooling still matters

Rust-side tooling is still required when you touch artifact production or bridge code. Use targeted Rust commands when you change files under `crates/`, build plumbing, or release packaging.

What not to do:

- do not present a top-level Cargo build as the primary public installation or onboarding story
- do not assume a source-tree backend is available in the checked artifact without confirming it via `souc info`

## 6. Recommended tool selection

Use this rule of thumb:

- public docs, examples, and ordinary validation: checked artifact + `souc check`
- docs publishing and route integrity: `npm --prefix website run ...`
- repo docs governance: `scripts/dev/check_docs_registry.sh` and `scripts/dev/check_docs_consistency.sh`
- bootstrap, SOIR, and deep IR debugging: `scripts/sounio-verify`, `Makefile.verify`, and `self-hosted/ir/`
- Rust bridge or artifact-production changes: targeted Cargo commands in the affected crate or release path

This is the tooling picture contributors should rely on now. Older "rustless cutover" language may still appear in historical notes, but the current repo contract is artifact-first for users and self-hosted-first for implementation work.

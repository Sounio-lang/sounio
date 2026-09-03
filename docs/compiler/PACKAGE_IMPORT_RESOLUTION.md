<!-- docs:meta
topic_id: repo.docs.compiler.package-import-resolution
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.package-import-resolution
-->

# Package Import Resolution

Sounio local package imports are part of the compiler contract, not a fixture-only test path.

The root package import name uses underscore form, while the local package directory uses hyphen form. For the canonical scientific package:

- `use epistemic_core::*` resolves to `packages/epistemic-core/src/lib.sio`.
- `epistemic_core` is the `[lib].name` in `packages/epistemic-core/sounio.toml`.
- `epistemic-core` is the `[package].name` and local package directory.

For package submodules, only the package root segment is underscore-to-hyphen mapped. A future `use epistemic_core::foo::*` resolves under `packages/epistemic-core/src/foo.sio` or the existing `mod.sio` fallback path.

Unreadable imports are fail-closed. The compiler may print the attempted resolved path for diagnostics, but it must not continue to emit a runnable executable after a missing package or module import.

The acceptance gate is:

```bash
bash scripts/ci/package_import_science_gate.sh
```

That gate validates the `epistemic-core` manifest, runs the package tests listed in `sounio.toml`, runs a downstream scientific witness that imports `epistemic_core::*`, and checks that a missing package import fails without producing an executable.

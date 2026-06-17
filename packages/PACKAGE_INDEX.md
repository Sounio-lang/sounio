# Sounio Package Index

Repo-local packages under `packages/`. Canonical implementations currently live in `stdlib/`; package `src/lib.sio` files are facades that re-export stdlib modules.

**Madaros (default `./bin/souc`)** resolves `packages/<name>/src/` imports in the module loader and `module_frontend_import_files_from_source` path (`underscore` import name → `hyphen` package dir). **`souc run` / native-v2 compile** still uses single-module lowering for imported sources; use **lean_single** (`bin/souc-lean-single-x86_64`) for end-to-end package compile+run until multimodule native-v2 lowering is stable.

| Package | Import name | Stdlib modules | Manifest |
|---|---|---|---|
| epistemic-core | `epistemic_core` | (standalone) | [epistemic-core/sounio.toml](epistemic-core/sounio.toml) |
| sounio-units | `sounio_units` | `units/`, `metrology/` | [sounio-units/sounio.toml](sounio-units/sounio.toml) |
| sounio-formats | `sounio_formats` | `yaml/`, `toml/`, `msgpack/` | [sounio-formats/sounio.toml](sounio-formats/sounio.toml) |
| sounio-io-primitives | `sounio_io_primitives` | `path/`, `log/`, `cmp/` | [sounio-io-primitives/sounio.toml](sounio-io-primitives/sounio.toml) |

## Extraction playbook

1. Add `packages/<name>/sounio.toml` with `[lib]`, `[[test]]`, and `[epistemic]` metadata.
2. Keep canonical `.sio` sources in `stdlib/` (Madaros default engine).
3. Add `packages/<name>/src/lib.sio` facade with `pub use` from stdlib modules.
4. Add `packages/<name>/tests/*.sio` importing stdlib paths (`use units::lib::*`, etc.).
5. Register manifest in `scripts/ci/package_import_science_gate.sh`.
6. When Madaros gains `packages/` resolution, move sources into `packages/<name>/src/` and add stdlib shims.

## Validation

```bash
bash scripts/ci/package_import_science_gate.sh
```
<!-- docs:meta
topic_id: website.docs.stdlib
authority: dual
audience: users
last_validated: 2026-10-27
validated_by: scripts/stdlib/scan_stdlib.sh
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.stdlib
-->

# Sounio STDLIB Reference

> **Looking for signatures?** See [STDLIB_API_REFERENCE.md](STDLIB_API_REFERENCE.md). It is generated from `stdlib/**/*.sio` by `bash scripts/build/gen_stdlib_api_md.sh`. Regenerated 2026-10-27: 15,457 `pub fn` across 132 modules. A signature in that file is not evidence that the function runs.

This page separates inventory from execution. Inventory was scanned on 2026-10-27. Execution numbers below are the newest committed result, not a new run.

## Inventory

Scanned 2026-10-27 with `bash scripts/stdlib/scan_stdlib.sh`.

| Metric | Value |
|---|---:|
| `sio_files` | 1741 |
| `disabled_files` | 0 |
| `stub_mod_files` | 0 |
| `mod_files` | 79 |
| `lib_files` | 102 |
| `module_entrypoints` | 181 |
| `active_module_entrypoints` | 181 |
| `hyper_active_files` | 55 |
| `hyper_disabled_files` | 0 |
| `hyper_stub_mod_files` | 0 |

`artifacts/stdlib/stdlib_inventory.v1.json` is the 2026-05-12 snapshot (927 files). It is not this scan. Do not copy 1,672, 1,702, or 178 from older prose.

Eleven top-level directories have source and no `pub fn`, so the generator omits them: `effects`, `fairness`, `geometry`, `interp`, `learn`, `pkg`, `qnn`, `regex`, `regulatory`, `science`, `time`.

## Execution

Newest committed end-to-end result: `artifacts/stdlib/stdlib_e2e_result.v1.json`, generated 2026-10-06 by `bash scripts/stdlib/run_stdlib_e2e.sh`.

| Metric | Value |
|---|---:|
| pass | 196 |
| fail | 165 |
| skip | 172 |
| total | 533 |

`artifacts/stdlib/stdlib_reliability_status.v1.json` (2026-05-12) says 251/0/0/251 and `pass`. That is not the current result. The March figures 71/0/9/80 are retired.

This pass did not re-run the E2E suite, the science pipeline, the hyper gate, or the GPU attestation gate. Do not read the May `status_summary: pass` files as the state of the 1,741-file tree.

## Contract levels

1. `active_callable` — a function that a named test runs. Inventory presence is not this level.
2. `stub_surface` — a module file with no reliable callable export.
3. `disabled_file` — `*.sio.disabled`. The 2026-10-27 scan found none.

## Reproduce the inventory

```bash
bash scripts/stdlib/scan_stdlib.sh
bash scripts/build/gen_stdlib_api_md.sh
```

Do not generate this reference with `cargo run -p souc --bin souniodoc`. There is no root `Cargo.toml`.

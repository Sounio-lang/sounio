<!-- docs:meta
topic_id: repo.docs.stdlib.stdlib-module-organization
authority: repo_only
audience: users
last_validated: 2026-10-15
validated_by: scripts/stdlib/scan_stdlib.sh
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.stdlib.stdlib-module-organization
-->

# STDLIB Module Organization

How the standard library is laid out on disk. Counts come from `bash scripts/stdlib/scan_stdlib.sh` on 2026-10-15. They are an inventory, not a claim that every file runs.

## File surface

- `lib.sio` is the usual API file where a module has one.
- `mod.sio` is an entrypoint or a compatibility surface. The scan found no stub `mod.sio` whose only code line is `module <name>`.
- `*.sio.disabled` is the disabled form. The scan found none.

| Metric | 2026-10-15 scan |
|---|---:|
| `sio_files` | 1702 |
| `disabled_files` | 0 |
| `mod_files` | 78 |
| `lib_files` | 101 |
| `module_entrypoints` | 179 |
| `stub_mod_files` | 0 |
| `active_module_entrypoints` | 179 |
| `hyper_active_files` | 56 |
| `hyper_disabled_files` | 0 |
| `hyper_stub_mod_files` | 0 |

The committed `artifacts/stdlib/stdlib_inventory.v1.json` is the 2026-05-12 snapshot (927 files). Do not cite it as this tree.

Directories with source and no `pub fn`, omitted by `scripts/build/gen_stdlib_api_md.sh`: `async`, `darwin`, `effects`, `fairness`, `geometry`, `interp`, `learn`, `pkg`, `qnn`, `regex`, `regulatory`, `science`, `string`, `thread`, `time`.

## What the files do not prove

A directory is not an `active_callable` module. The newest committed execution result is `artifacts/stdlib/stdlib_e2e_result.v1.json` (2026-10-06): 196 pass, 165 fail, 172 skip, 533 total. The May reliability JSON that reports 251/0/0/251 is not that result.

## Layout that actually occurs

`stdlib/CONVENTIONS.md` no longer requires every directory to contain `mod.sio`, `lib.sio`, `README.md`, and `EXAMPLES.md`. On 2026-10-15 only `chemistry`, `core`, `database`, and `dataframe` have all four at the top of the directory.

Primary reference: `docs/stdlib/STDLIB_REFERENCE.md`.

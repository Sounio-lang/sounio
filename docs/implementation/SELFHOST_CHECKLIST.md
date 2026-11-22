<!-- docs:meta
topic_id: repo.docs.implementation.selfhost-checklist
authority: repo_only
audience: maintainers
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.selfhost-checklist
-->

# Self-Host Check Runner

## Purpose

> **Removed.** `selfhost-check.sh` and `golden-update.sh` drove the Cargo-era compiler (`cargo build -p souc`, `rustc`, `souc sysroot install`, `souc compile --emit hlir`), none of which exists any more. They left the tree with `scripts/archive/`. Git history holds them. Nothing below is a command to run.

The old `selfhost-check.sh` printed Stage0/Stage1 smoke lines:

- `PASS [stage] ...`
- `WARN [stage] ...`
- `FAIL [stage] ...`

It verifies:

1. Sysroot preflight (`install/list/show/stdlib-paths`)
2. Stage0 IR emission (`ast`, `hir`, `hlir`)
3. Stage1 integration (`run --use-sounio-compiler`)
4. Bootstrap output parity (baseline `run` vs self-host `run`)
5. Golden drift (`tests/golden/minimal.hlir.golden.txt`)

Artifacts and logs are written to `/tmp/sounio-selfhost-check` by default.

## Usage

Those invocations were removed with the scripts. The old interface took `FILE`, `SOUNIO_SELFHOST_STRICT`, `SOUNIO_SYSROOT_HOME`, and `WORK_DIR`.

Strict mode behavior:

- If fallback marker `using Rust compiler` appears, strict mode exits with code `2`.
- Non-strict mode marks the same case as `WARN`.

## Golden Promotion

`golden-update.sh` used to refresh `tests/golden/minimal.hlir.golden.txt` with a backup, taking `FILE` and `GOLDEN_FILE`. It was removed with `scripts/archive/`.

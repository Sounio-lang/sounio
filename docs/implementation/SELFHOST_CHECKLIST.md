<!-- docs:meta
topic_id: repo.docs.implementation.selfhost-checklist
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.selfhost-checklist
-->

# Self-Host Check Runner

## Purpose

`scripts/dev/selfhost-check.sh` runs the Stage0/Stage1 smoke checks in one command and prints machine-readable gate lines:

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

```bash
cd ~/work/sounio
bash scripts/dev/selfhost-check.sh
```

Useful overrides:

```bash
FILE=examples/minimal.sio bash scripts/dev/selfhost-check.sh
SOUNIO_SELFHOST_STRICT=1 bash scripts/dev/selfhost-check.sh
SOUNIO_SYSROOT_HOME=/tmp/sounio-sysroots-stage0 bash scripts/dev/selfhost-check.sh
WORK_DIR=/tmp/sounio-check-alt bash scripts/dev/selfhost-check.sh
```

Strict mode behavior:

- If fallback marker `using Rust compiler` appears, strict mode exits with code `2`.
- Non-strict mode marks the same case as `WARN`.

## Golden Promotion

Use `scripts/dev/golden-update.sh` to update `tests/golden/minimal.hlir.golden.txt` with backup:

```bash
cd ~/work/sounio
bash scripts/dev/golden-update.sh
```

Optional overrides:

```bash
FILE=examples/minimal.sio GOLDEN_FILE=tests/golden/minimal.hlir.golden.txt bash scripts/dev/golden-update.sh
```

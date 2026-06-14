# Madaros Handoff: origin/main is green

## TL;DR

Madaros is confirmed working on `origin/main` at commit `17d1157be`.
Do **not** judge Madaros from stale worktrees, dirty lanes, or prebuilt raw artifacts.
The valid proof gate is `make madaros-full-gate`.

> Madaros is green on origin/main@17d1157be. Please sync/rebase before debugging checker issues against it. The valid proof gate is `make madaros-full-gate`; stale raw artifacts/self-hosted/madaros binaries are not evidence.

## Verified state

- **Commit**: `17d1157be fix(madaros): remove raw check caveats from full gate`
- **Gate command**: `make madaros-full-gate`
- **Gate coverage**:
  - `bin/madaros --version`
  - `bin/madaros check` on basic files
  - `artifacts/self-hosted/madaros --check` on the raw binary
  - Multi-module visibility: struct/function/enum pub/private
  - Missing input returns clean error (no SIGSEGV)
  - Native ELF build
  - `run`
  - Native-v2 ABI/backend witnesses
  - `bin/madaros pkg self-test`

## Dangerous case also verified

```bash
artifacts/self-hosted/madaros --check /tmp
MADAROS_RAW_BIN=artifacts/self-hosted/madaros bin/madaros check /tmp
```

Both now produce:

```
error: at /tmp:0:0 - could not read input file
```

with `rc=1`, no SIGSEGV.

## Sync instructions

If the lane can be discarded:

```bash
git fetch origin main
git checkout main
git reset --hard origin/main
make build-madaros
make madaros-full-gate
```

If the lane has WIP:

```bash
git fetch origin main
git rebase origin/main
make build-madaros
make madaros-full-gate
```

## Important: stale / dirty lanes

Agents in the following lanes may be seeing old code or local checker modifications:

- `/workspace/sounio-project-spine`
- `/workspace/sounio-checker`
- `/workspace/sounio-semcall-main`
- Any other worktree not on `origin/main@17d1157be`

Do **not** conclude "Madaros does not work" without first bringing in `17d1157be` and rebuilding `artifacts/self-hosted/madaros`.

## Historical context (for reference only)

Earlier work on branch `fix/silent-typecheck-diag` involved:

- Visibility model (`Private`, `Pub`, `PubCrate`, `PubSuper`, `PubIn`).
- Checker metadata (`FnSig`, `StructInfo`, `EnumInfo`) carrying `visibility` and `defining_module`.
- Multi-module `--check` real typechecking.
- Stage0 (`bin/souc`) miscompile workarounds for `Option<Box<T>>` patterns inside `*mut` functions.

All of that has landed and is verified green on `origin/main@17d1157be`.

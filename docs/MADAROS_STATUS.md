# Madaros Status — coordination note for the fleet

> **TL;DR:** Madaros is **green on `origin/main`** and is now the **default compiler**:
> `bin/souc` routes to Madaros. If it looks broken to you, you are almost certainly
> judging it from a **stale worktree** or a **prebuilt raw ELF compiled before the
> fix**. Sync `main`, rebuild, run the gate — *then* talk.

## Madaros is the default compiler (`bin/souc` → Madaros)

`bin/souc` is now a thin CLI wrapper that routes `check`/`compile`/`run`/`--version`/
`info` to Madaros (via `bin/madaros` → `artifacts/self-hosted/madaros`). The legacy
single-file `lean_single` engine that `bin/souc` used to be is preserved as
`bin/souc-lean-single-x86_64`.

- **Force the legacy engine:** `SOUNIO_SOUC_ENGINE=lean_single bin/souc <args>`.
- **lean_single stays the bootstrap seed** (`make build`, `make build-madaros`) and the
  canonical fixed-point ELF — it is just no longer the default *user-facing* compiler.
- If Madaros is not built yet, `bin/souc` falls back to lean_single with a stderr notice
  (`make build-madaros` to get the default engine).
- *Madaros-builds-Madaros* (swapping the build seed to Madaros) is a **separate, larger
  milestone — not done here.** lean_single still compiles the bootstrap.

## Confirmed state

- **Green since `17d1157be`** — `fix(madaros): remove raw check caveats from full gate`.
  It **stays green as `main` advances**: re-verified at the current tip, which now
  includes `051ddf9ae` (`fix(ir/opt+ssa+codegen): fix 7 bugs — tracking, use-count,
  SSA rename/phi/dom, REX, code overflow`), `39f248f28` (deny-by-default visibility),
  the `14f984e26` `bin/souc` rebuild, and the `4177613ca` checker-alloc fix.
  Track `origin/main`, **not** a frozen SHA.
- Madaros is the **Stage1 modular compiler** (`bin/madaros`,
  `scripts/ci/build_modular_madaros.sh`, `scripts/ci/madaros_full_gate.sh`).
  It is distinct from `souc` (the lean_single monolith), which still ships.

## The only valid proof gate

```bash
make madaros-full-gate     # builds Madaros from source, then runs the e2e gate
```

Independently verified at `17d1157be` (fresh build from source, **not** a
prebuilt artifact) — **10/10 PASS**:

```
PASS: version
PASS: public check CLI
PASS: raw check CLI
PASS: multimodule visibility diagnostics
PASS: missing input diagnostic
PASS: source build to native ELF
PASS: source run
PASS: native-v2 ABI/backend witnesses
PASS: package manager self-test
```

Re-verified at the current tip (fresh build from source through `051ddf9ae`,
which lands a 7-bug IR/SSA/codegen batch; `fns=9612`) — same result, all checks
PASS. The green state is not tip-fragile.

The previously-dangerous bad-input case is fixed on **both** paths (clean error,
`rc=1`, **no SIGSEGV**):

```bash
artifacts/self-hosted/madaros --check /tmp
# error: at /tmp:0:0 - could not read input file   (rc=1)

MADAROS_RAW_BIN=artifacts/self-hosted/madaros bin/madaros check /tmp
# error: at /tmp:0:0 - could not read input file   (rc=1)
```

## What stale state looks like (and why it lies)

A `bin/madaros` launcher or a raw `artifacts/self-hosted/madaros` ELF that was
**built before `17d1157be`** still carries the old behavior — that binary is
**not evidence** about current `main`. Likewise, lanes under
`/workspace/sounio-checker`, `/workspace/sounio-semcall-main`,
`/workspace/sounio-project-spine`, etc. may hold old checker code or local edits.
You cannot conclude "Madaros is broken" from any of those without first bringing
in `17d1157be` and **rebuilding `artifacts/self-hosted/madaros`**.

## Sync before debugging

If your lane can be discarded:

```bash
git fetch origin main
git checkout main
git reset --hard origin/main      # only if your lane's WIP can be thrown away
make build-madaros
make madaros-full-gate
```

If you have WIP to keep:

```bash
git fetch origin main
git rebase origin/main
make build-madaros
make madaros-full-gate
```

## One-line coordination phrase

> Madaros is green on `origin/main@17d1157be`. Please sync/rebase before debugging
> checker issues against it. The valid proof gate is `make madaros-full-gate`;
> stale raw `artifacts/self-hosted/madaros` binaries are **not** evidence.

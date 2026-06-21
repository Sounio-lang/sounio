<!-- docs:meta
topic_id: repo.docs.madaros-status
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.madaros-status
-->

# Madaros Status — coordination note for the fleet

> **TL;DR:** Madaros is the **default compiler**: `bin/souc` routes to Madaros.
> Do not treat stale worktrees or old raw ELFs as current evidence. Also do not
> overclaim production readiness: current `origin/main` still tracks open
> source-to-ELF/global-BSS and promoted-workspace self-build blockers in
> GitHub issue #356. Sync `main`, use the checked prebuilt or rebuild, run the
> named gate, and classify the result against the active blocker records.

## Madaros is the default compiler (`bin/souc` → Madaros)

`bin/souc` is now a thin CLI wrapper that routes `check`/`compile`/`run`/`--version`/
`info` to Madaros (via `bin/madaros` → `artifacts/self-hosted/madaros`, or the
checked prebuilt `bin/madaros-linux-x86_64` when the local build artifact is
absent). The legacy single-file `lean_single` engine that `bin/souc` used to be
is preserved as `bin/souc-lean-single-x86_64`.

- **Force the legacy engine:** `SOUNIO_SOUC_ENGINE=lean_single bin/souc <args>`.
- **lean_single stays the bootstrap seed** (`make build`, `make build-madaros`) and the
  canonical fixed-point ELF — it is just no longer the default *user-facing* compiler.
- If the local build artifact is absent, `bin/souc` uses the checked prebuilt
  `bin/madaros-linux-x86_64` first. It only falls back to lean_single if no
  Madaros raw ELF is available.
- *Madaros-builds-Madaros* (swapping the build seed to Madaros) is a **separate, larger
  milestone — not done here.** lean_single still compiles the bootstrap.

## Confirmed state

- **Operational baseline since `17d1157be`** — `fix(madaros): remove raw check caveats from full gate`.
  The broad gate lineage was re-verified through `origin/main@9c5d09a21`,
  which includes the wide-int lane (`17dbb9ce5`, `7b04ab15c`), the default
  `bin/souc` -> Madaros wrapper lane (`8d5193a64`, `077361b28`), and the checked
  prebuilt Madaros ELF (`42293db35`).
  Track `origin/main`, **not** a frozen SHA.
- Madaros is the **Stage1 modular compiler** (`bin/madaros`,
  `scripts/ci/build_modular_madaros.sh`, `scripts/ci/madaros_full_gate.sh`).
  It is distinct from `souc` (the lean_single monolith), which still ships.
- As of `origin/main@b1f6bb1b7`, production readiness is governed by
  `docs/audit/MADAROS_PRODUCTION_READINESS_PLAN_2026-06-21.md` and issue #356.
  The known-open blocker probe still reproduces:
  - `BLK-20260621-codex-source-elf-normal-bss`: minimal global read/write
    source-to-ELF witnesses compile-segfault with `rc=139`.
  - `BLK-20260621-codex-madaros-build-segfault`: promoted-workspace local
    self-build still exits `build_rc_139`.

Do not say "Madaros is production-ready" until those blocker records are closed
or explicitly narrowed with new acceptance gates.

## The only valid proof gate

```bash
make madaros-full-gate     # builds Madaros from source, then runs the e2e gate
```

The cheap coordination-contract gate is:

```bash
bash scripts/ci/madaros_operational_contract_gate.sh
```

It does not replace the compiler proof. It prevents drift in the committed agent
instructions, `bin/souc` default-wrapper contract, `scripts/dev/e2e_gate.sh`, and
`scripts/ci/madaros_full_gate.sh` coverage.

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

A `bin/madaros` launcher or a local raw `artifacts/self-hosted/madaros` ELF that
was **built before `17d1157be`** still carries the old behavior — that binary is
**not evidence** about current `main`. Likewise, lanes under
`/workspace/sounio-checker`, `/workspace/sounio-semcall-main`,
`/workspace/sounio-project-spine`, etc. may hold old checker code or local edits.
You cannot conclude "Madaros is broken" from any of those without first bringing
in `17d1157be` and either using the checked prebuilt `bin/madaros-linux-x86_64`
or **rebuilding `artifacts/self-hosted/madaros`**.

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

> Madaros is green on `origin/main` and the green base begins at `17d1157be`.
> Please sync/rebase before debugging checker issues against it. The valid proof
> gate is `make madaros-full-gate`; stale raw `artifacts/self-hosted/madaros`
> binaries are **not** evidence.

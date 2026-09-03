<!-- docs:meta
topic_id: repo.docs.status.madaros-main-proof-17d115
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.status.madaros-main-proof-17d115
-->

# Madaros Main Proof - 17d1157be

Date: 2026-06-14

Worktree:

- Path: `/workspace/sounio-madaros-main-proof`
- Branch: `codex/madaros-main-proof-17d115`
- Base: `origin/main`
- Commit: `17d1157be540d32bb583dd03ca7072a6026e2027`
- Subject: `fix(madaros): remove raw check caveats from full gate`

## Rebuild Proof

Command:

```bash
make build-madaros
```

Result:

- `build_madaros_rc=0`
- Produced `artifacts/self-hosted/madaros`
- Size observed after rebuild: `82M`

## Canonical Full Gate

Command:

```bash
make madaros-full-gate
```

Results:

- `madaros_full_gate_rc=0` before adding literal `/tmp` directory checks.
- `madaros_full_gate_with_tmp_rc=0` after adding literal `/tmp` directory checks to `scripts/ci/madaros_full_gate.sh`.

The updated gate output ended with:

```text
[madaros-full] PASS: version
[madaros-full] PASS: public check CLI
[madaros-full] PASS: raw check CLI
[madaros-full] PASS: multimodule visibility diagnostics
[madaros-full] PASS: missing input diagnostic
[madaros-full] PASS: source build to native ELF
[madaros-full] PASS: source run
[madaros-full] PASS: native-v2 ABI/backend witnesses
[madaros-full] PASS: package manager self-test
[madaros-full] PASS: public CLI, source ELF path, ABI witnesses, visibility, and pkg self-test
```

## Dangerous `/tmp` Check

Commands:

```bash
artifacts/self-hosted/madaros --check /tmp
MADAROS_RAW_BIN=artifacts/self-hosted/madaros bin/madaros check /tmp
```

Manual results before the gate update:

- Raw binary: `raw_tmp_rc=1`
- Wrapper with explicit raw binary: `wrapper_tmp_rc=1`
- Both emitted:

```text
error: at /tmp:0:0 - could not read input file
```

No SIGSEGV was observed.

These exact cases are now covered by `make madaros-full-gate`.

## Coordination Rule

Madaros is green on `origin/main@17d1157be`. Agents must fetch/rebase or reset to that commit, rebuild `artifacts/self-hosted/madaros`, and run `make madaros-full-gate` before treating checker/Madaros failures as current evidence. Dirty worktrees and stale raw binaries are not evidence against the current Madaros state.

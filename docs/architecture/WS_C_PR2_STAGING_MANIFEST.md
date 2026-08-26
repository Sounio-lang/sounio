<!-- docs:meta
topic_id: repo.docs.architecture.ws-c-pr2-staging-manifest
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: codex-2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.ws-c-pr2-staging-manifest
-->

# WS-C PR2 Staging Manifest

Date: 2026-08-16

Source of truth: `docs/architecture/WS_C_PR1_PAYLOAD_CENSUS.md` on `origin/main` after PR #1751. Current refs used while authoring this manifest:

- `origin/main`: `6f2c4e2461cc5da3a5952e1335d46b89e3160a4b`
- `origin/canon/madaros-v2-sota`: `97b525949765980406a4fefa7f533e9db89721e1`

## Purpose

PR2 should land the reviewable gate/payload bulk from the C1 census without touching the occupied `self-hosted/` writer surfaces and without violating C3 shared-oracle ownership.

The staging driver `scripts/dev/ws_c_pr2_stage_payload.sh` is **not in the repository** — it exists in no ref of this repo, so the invocations recorded below document the interface it was to expose rather than a command you can run today. The **Review Chunks** table in this document is the file-level manifest of record. As specified, the driver copies files from `origin/canon/madaros-v2-sota` into a target worktree in ordered groups, refuses to overwrite existing different content, and fails if a `tools/eisa/` or `stdlib/eisa/` path already exists on `origin/main`.

Default PR2 staging mode is the **49-file non-enir payload**:

```bash
bash scripts/dev/ws_c_pr2_stage_payload.sh --target /path/to/pr2-worktree --mode pr2
```

Full transitive reconstruction mode is available for audits or fresh stack rebuilds:

```bash
bash scripts/dev/ws_c_pr2_stage_payload.sh --target /path/to/worktree --mode all63
```

The `all63` mode includes the 14 `self-hosted/enir/` files that PR1 owns. Do not use `all63` for PR2 unless the PR2 branch is intentionally reconstructing the whole stack in an isolated worktree and has the relevant writer clearance.

## Review Chunks

Recommended commit or review order:

| Group | Count | PR2 default | Provenance |
| --- | ---: | --- | --- |
| `pr1-enir` | 14 | no | Common ENIR driver closure from the C1 census; PR1 prerequisite. |
| `e1-shadow` | 2 | yes | E1 shadow gate script and Python verifier. |
| `e2a-lowering` | 2 | yes | E2A lowering gate and verifier; regression chain includes E1. |
| `e2b-cfg` | 3 | yes | E2B CFG gate/verifier plus `eisa_enir_v1_oracle.sio`. |
| `e2c-fuel-blockargs` | 3 | yes | E2C fuel/blockargs gate/verifier plus `eisa_enir_v1_loop_oracle.sio`. |
| `e2d-rump-dd` | 3 | yes | E2D rump DD gate/verifier plus `eisa_enir_v1_rump_dd.eisa`. |
| `e2e-qd128-arithmetic` | 8 | yes | E2E qd128 gate/verifier plus six arithmetic fixtures from verifier `PROGRAMS`. |
| `e2f-rump-qd` | 3 | yes | E2F rump qd gate/verifier plus `eisa_enir_v2_rump_qd.eisa`. |
| `e2g-fuel-control-frail` | 5 | yes | E2G gate/verifier plus fuel, loop, and frail fixtures from verifier `PROGRAMS`. |
| `e2h-memory-move-poison` | 5 | yes | E2H gate/verifier plus memory, move, and poison fixtures from verifier `PROGRAMS`. |
| `e3a-mir-qd128` | 2 | yes | E3A MIR qd128 gate/verifier; reuses E2E arithmetic fixtures. |
| `e3b-mir-memory` | 2 | yes | E3B MIR memory gate/verifier; reuses E2H memory fixtures. |
| `e3c-cfg-memory-ssa` | 4 | yes | E3C CFG memory SSA gate/verifier plus memory-phi fixtures. |
| `e3d-multipred-ssa` | 4 | yes | E3D multipred scalar/memory SSA gate/verifier plus join fixtures. |
| `e3e-equal-event` | 3 | yes | E3E equal-value distinct-event gate plus then/else fixtures. |

PR2 default total: **49 files**. Full transitive total: **63 files**.

## C2 BASE_REF Re-anchor

The copied gate scripts still carry frontier-era `BASE_REF` assumptions. PR2 must apply the C2 decision from `docs/architecture/MIR_PORT_PLAN.md` to all 14 gate scripts as they land:

- Local/cascade regression calls use `HEAD`, matching the E3D precedent (`E3C_BASE_REF=HEAD`).
- Clean-checkout and CI frozen-surface checks use a PR-range base derived from `git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}"`.
- Environment overrides such as `E1_BASE_REF`, `E2*_BASE_REF`, `E3*_BASE_REF`, and `ENIR_GATE_BASE` remain available for emergency pins.

This manifest only stages the files. It does not rewrite the 14 gate scripts.

## C3 Shared-Oracle Boundary

WS-F owns `tools/eisa` and `stdlib/eisa` sources. PR2 may add frontier-only files that are absent from `origin/main`, but must not modify any `tools/eisa/` or `stdlib/eisa/` file that already exists on `origin/main`.

Present-but-changed watchlist from the census:

- `tools/eisa/eisa_evm_run.sio`
- `stdlib/math/qd128.sio`
- `bin/souc-lean-single-x86_64`
- `scripts/dev/souc-build-lock.sh`
- `self-hosted/compiler/main.sio`

Only `tools/eisa/eisa_evm_run.sio` is a shared oracle source in the C3 sense. It is **not** in the staging add-list. Any gate behavior depending on frontier semantics of that file must be revalidated against MAIN's oracle behavior rather than silently carrying the frontier version.

## Exact Ordered File List

The script was to be the executable copy source; since it is not in the repository, the exact ordered list is not reproducible from `main` and the **Review Chunks** table above is the surviving record. The intended invocations were:

```bash
bash scripts/dev/ws_c_pr2_stage_payload.sh --list --mode all63
```

To view only the PR2 default list:

```bash
bash scripts/dev/ws_c_pr2_stage_payload.sh --list --mode pr2
```

To stage a single review chunk:

```bash
bash scripts/dev/ws_c_pr2_stage_payload.sh --target /path/to/pr2-worktree --group e2b-cfg
```

## Exclusions

- `tools/eisa/eisa_enir_c2_rump.eisa` remains excluded from the E1/E2/E3 add-list; it belongs to the C2 gate, outside this PR2 payload.
- Runtime-generated `$TMP_DIR/*.eisa` negative fixtures are not repository payload.
- Existing main files under `tools/eisa/` or `stdlib/eisa/` are not copied by this manifest.

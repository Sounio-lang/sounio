<!-- docs:meta
topic_id: repo.docs.audit.coordination-2026-06-29-pbox-lanes
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.coordination-2026-06-29-pbox-lanes
-->

# Lane coordination — PBox / Madaros regressions (2026-06-29)

**Authority:** parallel work contract (`.claude/PARALLEL_BLOCKER_CONTRACT.md`)  
**Coordinator lane:** Composer 2.5 @ `/workspace/sounio` `research/solver-ts3-parallel`  
**Human author:** Demetrios

## Active blockers (compiler-owned)

| Blocker-ID | Severity | Owner | Worktree | Branch | Do-not-edit (parallel) |
|---|---|---|---|---|---|
| `BLK-20260629-stdlib-vancomycin-correlation-enclosure` | B1 | **Composer/compiler lane** (this session) | `/workspace/sounio` | `research/solver-ts3-parallel` | `self-hosted/compiler/module_frontend.sio`, `self-hosted/native/codegen_x86_linux.sio`, `bin/souc`, `bin/madaros` |
| `BLK-20260629-stdlib-epistemic-madaros-sigsegv` | B2 | **Composer/compiler lane** (bisect after correlation) | `/workspace/sounio` | `research/solver-ts3-parallel` | same as above |
| `BLK-20260629-stdlib-sret-pbox-clinical` | — | **closed** `ae8befbad` | — | — | — |

## Stale / disjoint worktrees (no duplicate work detected)

| Worktree | Branch | Overlap risk |
|---|---|---|
| `/workspace/sounio-codegen` | `claude/codegen-largestruct-fix` | Related (large struct codegen) — **no** `module_frontend` diff at tip; do not merge without author |
| `/workspace/sounio-merge` | `integration/native-v2-honest` | native-v2 — disjoint from multimodule merge finalize |
| `/workspace/sounio-compiler-consolidation` | `integration/compiler-consolidation-20260628` | consolidation — read-only cross-check only |
| `/workspace/sounio-project-spine` | `codex/project-spine-madaros` | Madaros spine — no active edit on this Blocker-ID |
| `/tmp/sounio-gpu-*` (18 worktrees) | `codex/gpu-*` | GPU lanes — **no** `module_frontend` ownership |
| `/workspace/sounio-pbpk-integration` | `integration/pbpk-sprints-*` | PBPK math — **read-only** until compiler blockers close |

**Rule:** No other agent edits `module_frontend.sio` on this branch until correlation + epistemic 139 dispatches close or handoff recorded.

## Stdlib lanes (read-only until compiler gates green)

| Lane | Action allowed | Forbidden |
|---|---|---|
| `stdlib/clinical/vancomycin_pbpk.sio` | witnesses, audits | dosing/monotonicity math changes without offload |
| `stdlib/epistemic/knightian.sio` | read-only audit | “fix” 139 by math drift |
| `tests/stdlib/clinical/*` | pins, `//@` annotations | tolerance widening |

## Acceptance sequence (serialized)

1. `vc_pbox_lo_probe.sio` Madaros exit **0**
2. `test_vancomycin_correlation_sensitivity.sio` exit **0** (lean_single stays **0**)
3. Epistemic 139 smallest witness `epistemic_bmi.sio` Madaros exit **0**
4. Re-baseline `bash scripts/run_sio_test_suite.sh epistemic` — target ≥ **28/60** pass (recover 10)

## Handoff command (other agents)

```bash
cd /workspace/sounio && git log -1 --oneline
md5sum artifacts/self-hosted/madaros
./bin/souc run docs/audit/VANCOMYCIN_CORRELATION_ENCLOSURE_2026-06-29/reference/vc_pbox_lo_probe.sio
# report exit code + md5
```

## Rebuild after `module_frontend.sio` edit

```bash
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
```

Do **not** nest `make build-madaros` inside `souc-build-lock.sh` (deadlock).

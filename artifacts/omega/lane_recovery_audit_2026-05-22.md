# Lane Recovery Audit — 2026-05-22

Reconstructed after the terminal dropped during cluster restructuring. The old
6-lane handoff (`.claude/AGENT_HANDOFF.md`) and live log
(`artifacts/omega/agent_handoff.log.md`) are **stale** — frozen at 2026-05-10/11.
This is the real state from git as of 2026-05-22.

`main` == `origin/main` (in sync). AHD/BHD = commits ahead/behind `main`.

## A. ACTIVE — branches ahead of main (real work to land)

| Branch | AHD/BHD | Worktree | Status |
|--------|---------|----------|--------|
| `fix/nested-field-store-codegen` | +9 / -3 | `sounio` (canonical) | **CURRENT.** nested field/index store compiler fix + graphics plotting lane (just committed `dffab525b`) + AssociatorField Phase Z gate. Unpushed. Needs rebase on main (3 behind) then push. |
| `feat/register-pbpk-clinical-gate` | +15 / 0 | `dissertation-audit-reconcile` | **PR #184 OPEN, updated today.** Up to date with main. Dissertation pbpk28 clinical gate. Closest to merge. |
| `compiler/task-c-bundle-codegen` | +17 / -61 | `.sounio-lane3` | Task C parse→lower bundle codegen (cf. `project_task_c_blocker`). Real; 61 behind, needs rebase. |
| `codex/native-v2-hof-lock` | +8 / -538 | `sounio-native-v2-hof-lock` | native-v2 HOF closure proof gate. **Very stale (May 8).** Likely superseded — verify before rebasing. |
| `codex/native-v2-imported-hof-abi` | +8 / -538 | `sounio-native-v2-imported-hof-abi` | Imported HOF ABI v1 native proof. Same staleness as above. |
| `garden/above-stars` | +1 / -455 | `sounio-garden-above-stars` | **BUTTERFLY THREAD (sacred).** See Section B — carries uncommitted feature work. |
| `coord/postmerge-activation-cleanup` | +1 / -411 | `sounio-postmerge-activation-cleanup` | Leftover lane-1 activation-status cleanup. Trivial; land or drop. |

## B. UNCOMMITTED WORK AT RISK (exists only in working trees)

| Worktree | Files | Assessment |
|----------|-------|------------|
| `sounio-garden-above-stars` | 17 | **HIGH VALUE.** New `stdlib/epistemic/path.sio` + 6 compile-fail + 2 run-pass tests for an "epistemic path fermentation" feature, plus a **rebuilt `bin/souc-linux-x86_64`** and `lean_single.sio` edits. Genuine in-flight feature, committed nowhere. Garden is sacred per directive — do not touch without explicit go-ahead. |
| `sounio-native-v2-fnref-calls` (on `main`) | 26 | Mostly disposable: 25 scratch `main_test*.sio` compiler experiments + 1 real `module_frontend.sio` edit. |
| `sounio` (canonical) | local only | `.claude/settings.local.json`, `.claude/llm_offload_log.md` (leave), `tests/golden/kaxi_ptx/f32_assoc_gum/` (belongs to Phase Z gate). |
| `sounio-explore` (4f), `sounio-lsp-continuation` (8f), `sounio-stage-g-gamma` (3f), `sounio-lane-5-phase5` (1f), 5× `worktree-agent-*` (1f, locked) | minor | Review individually; low risk. |

## C. MERGED / STALE — 0 ahead of main, safe to retire

- `coord/lane-3-paper-168` + `-k8` / `-k9` / `-k10` — paper-168 cohomological decomposition (k=4..10), all landed through **PR #116**. Conjecture 5 confirmed to k=9; saturation at 23 classes. Done.
- `coord/lane-4-nv2-hardening` — merged **PR #129**.
- `coord/lane-5-phase5-recognizer` — merged **PR #130**.
- `claude-2/wip-2026-05-19` — merged **PR #165**.
- `feat/port-parallel-issue`, `feat/fisher-tippett-variance` (both -77), `explore/sandbox` (-78), `feature/lsp-continuation` (-82), `dissertation/3d-frontend-stage-f` (-332) — 0 ahead; merged or abandoned.
- `worktree-agent-*` (5, locked) — 0 ahead.

## Suggested recovery order

1. ✅ Secure stranded graphics lane — DONE (`dffab525b` on current branch).
2. Decide on `garden/above-stars` uncommitted feature (Section B) — highest at-risk value.
3. Shepherd **PR #184** (only open PR, up to date) to merge.
4. Rebase + push `fix/nested-field-store-codegen`; open PR.
5. Rebase/triage `compiler/task-c-bundle-codegen` (+17).
6. Verify whether the two `codex/native-v2-hof-*` branches (May 8, -538) are still relevant or superseded.
7. Prune Section C worktrees/branches to cut noise.

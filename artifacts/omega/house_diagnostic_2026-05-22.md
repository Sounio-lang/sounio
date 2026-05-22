# House Diagnostic — 2026-05-22 (read-only, post-crash)

Companion to `lane_recovery_audit_2026-05-22.md`. Pure diagnosis — nothing
modified. Garden lane (`garden/*`) is owned by another Claude; left untouched.

## Headline numbers
- **130 local branches**; **51 are 0-ahead of `origin/main`** (merged/abandoned → prunable).
- **57 stashes** (oldest 2026-04-30, ~32 from "weeks ago", 1 from ~1h ago = Garden claude).
- **26 worktrees**; 14 carry uncommitted changes (see below).
- Local `main` (8c6631a2a) is **14 behind `origin/main`** (5eae60d10, PR #183 graphics scaffold). Stale.
- Only **1 open PR**: #184 (pbpk28 clinical gate), up to date with main.

## Current branch: `fix/nested-field-store-codegen`
- Diverged from origin: **2 local ahead** (`dffab525b` graphics plotting, `d1eafb8bc` AssociatorField Phase Z) vs **6 remote ahead** (Windows PE #176/#178, A64 parity #181, merge-main #0d6b656e). Needs rebase onto origin then push.
- 11 uncommitted files in canonical `sounio` worktree:
  - **Junk** (never tracked, unreferenced): `test_huffman_encode` (ELF 78KB), `sounio-whereami` (loose script).
  - **Real, uncommitted**: `tests/golden/kaxi_ptx/f32_assoc_gum/` (consumed by `kretikos_kaxi_phase_z_assoc_gate.sh`), `artifacts/omega/kretikos_spirv_vulkan_storage_*` (SPIR-V/Vulkan), `examples/real_sounio_{capability,native_knowledge}_demo.sio`.
  - **Leave**: `.claude/settings.local.json`, `.claude/llm_offload_log.md`.

## At-risk uncommitted work (per worktree)
| Worktree | Files | Note |
|----------|-------|------|
| `sounio` (canonical) | 11 | this branch — see above |
| `/tmp/garden-probe` (garden/fermentation-on-main) | 10 | **GARDEN — other Claude, do not touch** |
| `sounio-native-v2-fnref-calls` (main) | 26 | mostly 25 scratch `main_test*.sio` + 1 real `module_frontend.sio` (per audit) |
| `sounio-lsp-continuation` | 8 | review individually |
| `sounio-explore` | 4 | scratch sandbox |
| `sounio-stage-g-gamma` | 3 | dissertation 3d-frontend |
| `sounio-garden-above-stars` | 1 | was 17 — Garden claude already moved the feature out |
| `sounio-lane-5-phase5` | 1 | trivial |
| 5× `worktree-agent-*` (locked) | 1 each | locked, low risk |
| `dissertation-audit-reconcile` | 1 | PR #184 worktree |

## Stash triage signal
57 stashes across many deleted/stale branches (`m1_2-step-d-*`, `coord/nv2-*`,
`research/subptx-*`, etc.). All but the most recent are weeks old. Most are
auto-stashes / branch-switch saves with no remaining home branch. Candidate for
heavy pruning, but each should be eyeballed before drop (a few name real WIP:
park-miller, task-c-bundle, dataset-expansion).

## Recommended order (deferred — awaiting go-ahead)
1. Drop confirmed junk (`test_huffman_encode`, `sounio-whereami`).
2. Commit real artifacts (f32_assoc_gum, SPIR-V, examples) on current branch.
3. Rebase `fix/nested-field-store-codegen` onto origin; push; open PR.
4. Fast-forward local `main` to `origin/main` (5eae60d10).
5. Prune the 51 zero-ahead local branches + dead worktrees.
6. Triage + prune the 57 stashes (eyeball each).
7. Shepherd PR #184.

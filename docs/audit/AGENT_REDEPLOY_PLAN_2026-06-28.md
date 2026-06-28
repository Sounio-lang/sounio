# Agent redeploy plan - 2026-06-28

This plan is based on the post-cleanup repository state in
`integration/compiler-consolidation-20260628` after parking local-only lanes and
removing stale remote refs.

Current inventory:

- 18 local branches
- 22 remote refs
- 18 worktrees
- Active consolidation branch:
  `integration/compiler-consolidation-20260628`

## Redeploy rules

- Do not edit another lane's worktree without explicit owner transfer.
- Treat dirty remote-backed worktrees as live ownership signals, not branch
  cleanup targets.
- Archive tags and WIP patches before deleting any remaining branch/worktree.
- Do not promote math, clinical, PBPK, website, or external-facing claims
  without the repository LLM-offload policy.
- Keep `origin/integration/sounio-dev-ready-base` as historical recovery base.
- Leave the Cursor detached worktree untouched unless Cursor owner handoff is
  explicit.

## Lane assignments

### Consolidation owner

- Branch: `integration/compiler-consolidation-20260628`
- Worktree: `/workspace/sounio-compiler-consolidation`
- Owner: Codex consolidation
- State: clean, pushed
- Mission: continue branch/ownership cleanup, extract only focused compiler
  patches with gates, and keep audit docs current.

### GLM hyper-epistemic owner

- Branch: `feat/hyper-epistemic-mul`
- Worktree: `/workspace/sounio`
- Owner: GLM lane
- State: remote-backed, dirty
- Mission: finish/review hyper-epistemic multiplication lowering and related
  IR/native/PBPK variance ABI work.
- Guardrail: do not merge wholesale into consolidation; extract only named,
  reviewed commits or focused patches with math/offload compliance where
  applicable.

### Compiler/native integration owners

- Branch: `integration/native-v2-honest`
- Worktree: `/workspace/sounio-merge`
- Owner: native-v2 integration
- State: remote-backed, dirty with Slurm/artifact residue
- Mission: native-v2 honest integration and gate proof.

- Branch: `codex/project-spine-madaros`
- Worktree: `/workspace/sounio-project-spine`
- Owner: Madaros/project-spine compiler owner
- State: remote-backed, dirty with checker/parser/native smoke artifacts
- Mission: compiler spine and Madaros language reality work.

- Branch: `claude/codegen-largestruct-fix`
- Worktree: `/workspace/sounio-codegen`
- Owner: codegen large-struct owner
- State: remote-backed, dirty with Slurm/codegen result artifacts
- Mission: large-struct codegen validation.

- Branch: `claude/ir-heap-indirect`
- Worktree: `/workspace/sounio-ir`
- Owner: IR heap-indirect owner
- State: remote-backed, dirty with audit/repro artifacts
- Mission: heap-indirect IR plan/repro cleanup and possible focused extraction.

- Branch: `claude/effects-enforcement`
- Worktree: `/workspace/sounio-effects`
- Owner: effects enforcement owner
- State: remote-backed, dirty with effects artifacts and helper script
- Mission: effects keyword enforcement and artifact triage.

- Branch: `codex/real-language-runner`
- Worktree: `/workspace/sounio-real-runner`
- Owner: real runner owner
- State: remote-backed, Slurm-only dirty
- Mission: real language runner / execution-surface work.

- Branch: `codex/semcall-hof-main`
- Worktree: `/workspace/sounio-semcall-main`
- Owner: semantic-call owner
- State: remote-backed, Slurm + Madaros artifact dirty
- Mission: semantic call / higher-order function lane.

- Branch: `g1/qualify-bare-patterns`
- Worktree: `/workspace/sounio-cluster-c`
- Owner: checker/pattern qualification owner
- State: remote-backed, Slurm-only dirty
- Mission: qualify bare pattern handling.

- Branch: `integrate/kw-demote-landing`
- Worktree: `/workspace/tmp/integrate-kw-demote-landing`
- Owner: kw-demote integration owner
- State: remote-backed, dirty with bootstrap/frontend ELF artifacts
- Mission: keyword-demotion landing/revalidation.

### Scientific / GPU / PBPK owners

- Branch: `feat/affine-nonassoc-uncertainty`
- Worktree: `/workspace/sounio-affine`
- Owner: affine uncertainty owner
- State: remote-backed, Slurm-only dirty
- Mission: affine non-associative uncertainty.

- Branch: `feat/affine-octonion-correlation`
- Worktree: `/workspace/sounio-affine-pg`
- Owner: affine/octonion correlation owner
- State: remote-backed, Slurm-only dirty
- Mission: affine octonion correlation.

- Branch: `feat/gpu-thread-intrinsics`
- Worktree: `/workspace/sounio-gpu-kernel`
- Owner: GPU kernel owner
- State: remote-backed, generated GPU/compiler artifacts
- Mission: GPU thread intrinsics and native SASS/PTX evidence.

- Branch: `integration/pbpk-sprints-28-70-onto-main`
- Worktree: `/workspace/sounio-pbpk-integration`
- Owner: PBPK integration owner
- State: remote-backed, Slurm-only dirty
- Mission: PBPK sprint integration.
- Guardrail: clinical/PBPK promotion requires offload compliance.

### Tooling / viz / campaign owners

- Branch: `campaign/mc-frontend-fixes`
- Worktree: `/workspace/tmp/mc-campaign-fixes`
- Owner: MC campaign/frontend owner
- State: remote-backed, Slurm-only dirty
- Mission: campaign/frontend fixes.

- Branch: `codex/viz-molecule-authoring`
- Worktree: `/workspace/sounio-viz-molecule-authoring`
- Owner: visualization/molecule authoring owner
- State: remote-backed, Slurm-only dirty
- Mission: molecule authoring / visualization proof lane.

### External owner

- Worktree:
  `/workspace/.home/openvscode-server/.cursor/worktrees/DMH2026-e3d71e55/sounio-c48641ba187e`
- Branch: detached at `4aab38cd8`
- Owner: Cursor
- State: clean
- Mission: unknown; leave untouched unless Cursor owner hands it off.

## Next consolidation sequence

1. Review `feat/hyper-epistemic-mul` as the primary GLM lane and decide whether
   to open a PR or extract a smaller compiler-safe patch into consolidation.
2. Pick one compiler/native lane at a time for focused extraction:
   `integration/native-v2-honest`, `codex/project-spine-madaros`, or
   `claude/codegen-largestruct-fix`.
3. Keep scientific/PBPK/website lanes out of consolidation until their
   claim/offload requirements are satisfied.
4. After a lane is merged or intentionally parked, tag its final tip, archive
   any dirty state, remove the worktree, and delete the branch.

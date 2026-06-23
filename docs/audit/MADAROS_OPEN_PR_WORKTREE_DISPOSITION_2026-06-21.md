<!-- docs:meta
topic_id: repo.docs.audit.madaros-open-pr-worktree-disposition-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-open-pr-worktree-disposition-2026-06-21
-->

# Madaros Open PR / Worktree Disposition — 2026-06-21

## Scope

Governance-only disposition table for Madaros production readiness after #368/#369
and with #390 in flight. No compiler files were edited to produce this note.

Baseline:

- `origin/main`: `cf3c9ab52` (`tools(madaros): summarize contained compiler overlaps`, #389)
- Canonical blocker thread: GitHub issue [#356](https://github.com/Sounio-lang/sounio/issues/356)
- Readiness plan: `docs/audit/MADAROS_PRODUCTION_READINESS_PLAN_2026-06-21.md`
- Protected dirty primary: `/workspace/sounio` — **not** current-main evidence

Commands used:

```bash
git fetch --prune origin main
git worktree add /tmp/sounio-governance-audit-20260621 origin/main
gh pr list --state open --limit 50 --json number,title,headRefName,baseRefName,isDraft,mergeable,url
scripts/dev/madaros_readiness_status.sh --check-pr-resolution-queue
scripts/dev/madaros_readiness_status.sh --check-compiler-pr-overlap
```

## Recently Closed (context)

| PR | Branch | Disposition | Evidence |
|---|---|---|---|
| #368 | `codex/madaros-readiness-blocker-contract` | **merged** | Blocker contract on `main`; governance only |
| #369 | `codex/madaros-selfbuild-open-probe` | **merged** | Self-build parity tracked in `madaros_open_blockers_probe.sh` |

## Open PR Disposition

| PR / branch | Owner | Touches compiler? | Current-main relevance | Action | Evidence |
|---|---|---|---|---|---|
| [#390](https://github.com/Sounio-lang/sounio/pull/390) `codex/madaros-ready-conflict-guard` | Codex governance | No (`scripts/dev/madaros_pr_resolution_queue.sh` only) | **High** — closes governance gap: fail when non-draft `main` PRs are `CONFLICTING` | **keep / merge** | `mergeable=MERGEABLE`; worktree `/tmp/sounio-madaros-ready-conflict-guard` at `origin/main+1`; complements #372 queue |
| [#313](https://github.com/Sounio-lang/sounio/pull/313) `fix/native-codegen-sret-regression-v2` | Historical compiler (draft) | **Yes** — `main.sio`, `module_frontend.sio`, `module_native_driver.sio` | **Low** — stale vs `cf3c9ab52`; overlaps active Claude lane | **close / rebuild** | `mergeable=CONFLICTING`; `--check-compiler-pr-overlap` fails on `main.sio`; ownership transfer required before any merge |
| [#232](https://github.com/Sounio-lang/sounio/pull/232) `codegen/nested-mut-write-fix` | Historical compiler integration (draft) | **Yes** — `main.sio`, `lean_single.sio`, `codegen_x86_linux.sio`, checker/lexer/parser | **Low** — broad stale integration | **close / rebuild** | `mergeable=CONFLICTING`; category `compiler_adjacent_review`; quarantined in prior governance audit |
| [#241](https://github.com/Sounio-lang/sounio/pull/241) `nl-castle/native-orc-audit` | Research / NL castle (draft) | Adjacent — `bin/souc`, `bin/souc-linux-x86_64` overlap | **None for Madaros prod** | **needs ownership transfer / keep draft** | `compiler_adjacent_review`; not on Madaros path unless replayed on current `main` |
| [#329](https://github.com/Sounio-lang/sounio/pull/329) `codex/pl-command-center` | Codex docs (draft) | No | **Low** — docs/product lane | **rebuild / close stale** | `stale_conflicting`; 91 commits behind on worktree |
| [#308](https://github.com/Sounio-lang/sounio/pull/308) `chore/repo-hygiene` | Codex hygiene (draft) | No | **Low** | **rebuild / close stale** | `stale_conflicting` |
| [#297](https://github.com/Sounio-lang/sounio/pull/297) `qual/pbpk28-tissue-composition` | Science / PBPK (draft) | No (stdlib/examples) | **None for Madaros prod** | **needs offload / rebuild** | Clinical/science lane; `stale_conflicting` |
| [#287](https://github.com/Sounio-lang/sounio/pull/287) `feat/affine-octonion-correlation` | Math / epistemic research (draft) | No | **None for Madaros prod** | **needs offload / rebuild** | Math claims; `stale_conflicting` |
| [#226](https://github.com/Sounio-lang/sounio/pull/226) `feat/erdos-straus-gpu-sieve` | GPU examples | Touches `lean_single.sio` | **Excluded** — base is `integration/sounio-dev-ready-base`, not `main` | **keep separate** | `non_main_base`; mergeable only to integration base |

Queue summary from `madaros_readiness_status.sh --check-pr-resolution-queue`:

- `total=8` open PRs in automated queue (pre-#390 merge; #390 adds guard script)
- `compiler_owner_overlap=1` → **#313**
- `stale_conflicting=4` → #329, #308, #297, #287
- `compiler_adjacent_review=2` → #241, #232
- `non_main_base=1` → #226

**Recommended governance sequence:**

1. Merge **#390** (safe, governance-only).
2. Close or quarantine **#313** with explicit comment — blocks `--check-compiler-pr-overlap`.
3. Leave **#232/#241** draft-quarantined until compiler owner requests salvage.
4. Rebuild or close stale drafts **#329/#308/#297/#287** from fresh `origin/main` worktrees.
5. Keep **#226** outside Madaros production queue.

## Critical Worktree Disposition

| Worktree | Branch | Owner | Touches compiler? | vs `origin/main` | Action | Evidence |
|---|---|---|---|---|---|---|
| `/workspace/sounio` | `main` | Protected primary | Yes (dirty `machine_ir.sio`, `bin/madaros`) | behind **96**, dirty **34** | **keep protected** — not evidence | Allowlisted in worktree audit; archive-only reconciliation |
| `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` | `worktree-agent-adc1cd8b9d52ba53b` | **Claude compiler lane** | **Yes** — owned files dirty | behind 76+ | **keep active** — Codex read-only | Owns BSS blocker per #356; `madaros_readiness_status.sh --check-compiler-lane` |
| `/tmp/sounio-governance-audit-20260621` | detached @ `cf3c9ab52` | This governance pass | No | **current** | **use for governance edits** | Clean `origin/main` evidence surface |
| `/tmp/sounio-madaros-ready-conflict-guard` | `codex/madaros-ready-conflict-guard` | Codex | No | ahead **1** | **keep until #390 merges** | PR #390 worktree |
| `/tmp/sounio-madaros-boxnew-clean` | `codex/madaros-boxnew-clean` | Codex (historical) | Yes (IR) | behind **99**, ahead 15 | **archive / close** | Superseded by current-main Box audit on `main`; not production lane |
| `/tmp/sounio-madaros-boxnew-fix` | `codex/madaros-boxnew-append-fix` | Codex (historical) | Yes | stale | **archive / close** | Pre-#331 era |
| `/tmp/sounio-madaros-rebuild-probe` | `codex/madaros-rebuild-probe` | Codex probe | Unknown | stale | **close after probe recorded** | Ephemeral probe lane |
| `/tmp/sounio-pl-command-center` | `codex/pl-command-center` | Codex | No | behind **91** | **rebuild or close** | Matches stale #329 |
| `/workspace/sounio-codegen` | `claude/codegen-largestruct-fix` | Claude (historical) | Yes | behind **332**, ahead 166 | **needs ownership transfer / archive** | Divergent; not active lane per handoff |
| `/workspace/sounio-forloop` | `fix/madaros-for-loop-lowering` | Historical | Yes | behind 99 | **superseded** | Fix merged via #331 lineage on `main` |
| `/workspace/sounio-integ` | `fix/root2-enum-inplace` | Historical | Yes | behind 82 | **superseded** | Fix merged via #331 lineage on `main` |
| `/workspace/sounio-madaros-check-segv` | `codex/madaros-full-functioning` | Codex (stale) | Yes | behind 248 | **archive / close** | Pre-readiness-plan crash census |
| `/workspace/sounio-madaros-main-proof` | `codex/madaros-main-proof-17d115` | Codex (stale) | Yes | behind 248 | **archive / close** | Superseded by #356 blocker model |

## Active Blockers (unchanged)

```text
Blocker-ID: BLK-20260621-codex-source-elf-normal-bss
Severity: B1 | Class: compiler-semantics
Owner: Claude compiler/codegen lane
Evidence: global_read_exit4 / global_store_exit7 => compile_rc_139
Gate: scripts/ci/madaros_open_blockers_probe.sh
Issue: https://github.com/Sounio-lang/sounio/issues/356
```

```text
Blocker-ID: BLK-20260621-codex-madaros-build-segfault
Severity: B2 | Class: platform-resource
Owner: integration shepherd / workspace-runtime lane
Evidence: local build_modular_madaros.sh rc=139; GitHub Prebuilt Refresh green
Gate: scripts/ci/madaros_open_blockers_probe.sh (known-open parity witness)
Issue: https://github.com/Sounio-lang/sounio/issues/356
```

**Passing control (not open):** direct-call argument ABI (`call_arg_id_exit42`).

## Explicit Non-Actions

- No edits to Claude-owned compiler files.
- No reset/clean/rebase of `/workspace/sounio`.
- No promotion of stale worktree compiler patches without current-main replay.

## Next Safe Codex Actions

1. Land **#390** from `/tmp/sounio-madaros-ready-conflict-guard`.
2. Comment on **#313** with quarantine + ownership-transfer exit criteria.
3. Optionally comment on stale drafts (#329, #308, #297, #287) pointing to this disposition.
4. Re-run:

```bash
scripts/dev/madaros_readiness_status.sh --check-pr-resolution-queue
scripts/dev/madaros_readiness_status.sh --check-compiler-pr-overlap
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
```

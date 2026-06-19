<!-- docs:meta
topic_id: repo.docs.repo-lfs-migration
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.repo-lfs-migration
-->

# Runbook: migrate large binaries out of git history (Git LFS)

**Status: planned, not executed.** This is a *coordinated, scheduled* operation —
it rewrites history and force-pushes every ref, which breaks every outstanding
branch, PR, and worktree. Do not run it ad hoc. Read all of it first.

## Why

The repo carries large binaries/artifacts in history, which bloats every clone
and fetch. Worst offenders (tracked, `git ls-tree -r -l HEAD | sort -k4 -nr`):

| size | path | disposition |
|------|------|-------------|
| ~90 MB | `bin/madaros-linux-x86_64` | seed binary → **LFS** (or drop; regenerable from source) |
| ~77 MB | `artifacts/research/eeg_hessian/sliding_seizure_manifest.tsv` | generated → **purge** (regenerable) |
| ~43 MB | `artifacts/omega/native_backend_v2_scalar_smoke.aarch64-macos.bin` | generated → **purge** |
| ~29 MB | `formal/lean4/SounioSatG529.lean` | source-ish (generated cert) → keep or LFS |
| ~26 MB | `docs/training/finetune/sounio_corpus.txt` | data → **LFS** or external |
| 5–17 MB | `datasets/*.jsonl`, `artifacts/llm_training/*`, brand PDFs/SVGs | data/assets → **LFS** |

~1269 files live under `bin/` + `artifacts/` total. `artifacts/` is already
`.gitignore`d, so most were force-added or pre-date the ignore — they should be
purged from history, not just untracked.

## Decision rule

- **Regenerable from source/scripts** (most of `artifacts/`, processed data,
  smoke `.bin`s) → **purge from history** (`git filter-repo --path ... --invert-paths`),
  keep ignored going forward.
- **Genuinely needed checked-in binaries** (e.g. a bootstrap seed compiler) →
  **track in LFS** (`git lfs migrate import`).
- When unsure, prefer purge + a documented regeneration command over LFS.

## Preconditions (do these first)

1. **Branch/PR cleanup** — fewer refs = less to rewrite and re-push. Land or
   close the stale PRs and prune merged branches *before* the rewrite.
2. **Announce a freeze window.** No merges/pushes during the rewrite.
3. **Enable Git LFS** on the GitHub repo (Settings → confirm LFS + check the
   storage/bandwidth quota for the sizes above).
4. Inventory and **record every open PR's head SHA** (so PRs can be re-pushed and
   reopened if the rewrite changes their refs):
   `gh pr list --state open --limit 400 --json number,headRefName,headRefOid > pr_snapshot.json`

## Procedure (mirror clone — never on a working clone)

```bash
# 0. Fresh mirror (history-only; no worktrees attached)
git clone --mirror git@github.com:Sounio-lang/sounio.git sounio-rewrite.git
cd sounio-rewrite.git

# 1a. PURGE regenerable artifacts from all history
#     (install: pipx install git-filter-repo)
git filter-repo --invert-paths \
  --path artifacts/research/eeg_hessian/sliding_seizure_manifest.tsv \
  --path artifacts/omega/native_backend_v2_scalar_smoke.aarch64-macos.bin \
  --path-glob 'artifacts/**'            # scope precisely; review the list first

# 1b. MIGRATE the binaries worth keeping into LFS
git lfs migrate import --everything \
  --include="bin/madaros-linux-x86_64,docs/training/finetune/sounio_corpus.txt,datasets/**/*.jsonl"

# 2. Sanity: repo size dropped, history intact for source
git count-objects -vH
git log --oneline -5

# 3. Force-push the rewritten history (THIS is the irreversible step)
git push --force --mirror
```

## After the rewrite (everyone, mandatory)

- **All clones and worktrees are now invalid** — old history diverged. Each
  contributor/agent must **re-clone** (worktrees cannot survive a mirror rewrite;
  delete and recreate them).
- Commit the enforcement rules on the rewritten `main`:
  - `.gitattributes` — `*.jsonl filter=lfs diff=lfs merge=lfs -text`, the seed
    binary path, etc.
  - `.gitignore` — already hardened (`artifacts/`, `data/processed/`, build dirs).
- **Re-push and reopen open PRs** from `pr_snapshot.json` if their refs were
  dropped (PR head branches must be re-pushed against the rewritten base).
- Bust CI caches that pin old SHAs.

## Risks & rollback

- **Irreversible once force-pushed.** Keep the pre-rewrite mirror
  (`sounio-rewrite.git` before step 3, plus a server-side backup ref/tag) until
  the team confirms the new history is good — that is the only rollback.
- Open PRs across rewritten bases will show as needing re-push; budget time to
  restore them (cf. the branch-deletion incident — closed/diverged PRs are
  recoverable from recorded head SHAs, but it's manual).
- Estimate downtime: with N open PRs + worktrees, the re-clone/re-push tail is
  the slow part, not the rewrite itself.

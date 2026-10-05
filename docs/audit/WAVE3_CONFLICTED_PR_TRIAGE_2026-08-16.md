<!-- docs:meta
topic_id: repo.docs.audit.wave3-conflicted-pr-triage-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.wave3-conflicted-pr-triage-2026-08-16
-->

# Wave 3 — nine DIRTY PRs (TRIVIAL / SEMANTIC / STALE)

**Date:** 2026-08-17 (re-verified; first pass 2026-08-16)  
**Lane:** grok-cli2 / `wave3-pr-conflict-triage`  
**Brief:** [`.scratch/W3_DIRTY_REBASE.md`](../../.scratch/W3_DIRTY_REBASE.md)  
**Method:** for each PR — detached scratch worktree at `origin/<head>`,  
`git reset --hard` + `git clean -fd`, `git rebase origin/<base>`, record  
unmerged paths, **`git rebase --abort`**, `git worktree remove --force`.  
**No push. No merge.**  
**Base tip:** `origin/main` = **`0b0c5cdd5b`** (except #1580 base =  
`research/self-falsifying-compilation-line-20260726`).  
**Raw receipts:** `/tmp/w3-rebase/results.json`, `/tmp/w3-rebase/prs.json`

### Counting rules

| Class | Meaning |
|---|---|
| **TRIVIAL** | Rebase **CLEAN**, or only noise: `.claude/llm_offload_log.md`, binary seed `bin/souc-lean-single-x86_64`, or a mechanical lakefile union |
| **SEMANTIC** | Real conflicts in `self-hosted/` / build scripts that need a human call on meaning |
| **STALE** | Duplicate, **already merged**, or unusable mega landing vehicle |

**Not counted as real conflicts** (`.gitattributes` `merge=governance-regen` — GitHub cannot run the driver):

- `docs/governance/DOCS_ACCEPTANCE_REPORT.md`
- `docs/governance/DOCS_AUTHORITY_MATRIX.md`
- `docs/governance/topic-registry.v1.json`

---

## Headline counts

| Class | N | PRs |
|---|---:|---|
| **TRIVIAL** | **5** | #1750, #1729, #1659, #1708, #1058 |
| **SEMANTIC** | **1** | **#1604** |
| **STALE** | **3** | **#1580**, **#1732** (merged), **#1605** (dup of #1604) |

---

## Summary table (measured 2026-08-17)

| PR | Title (short) | Class | behind / ahead | Rebase status | Real conflicts (non-gov) |
|---:|---|---|---:|---|---|
| **#1580** | CD-tower ZD fibers mega-PR | **STALE** | 60 / 459 | fails early: dirty governance-regen trio during pick (452 commits left); not a viable vehicle | n/a (aborts before useful resolve); landing superseded by **#1708** |
| **#1732** | Move Proof-Carrying Weaning to sibling | **STALE** | 6 / **0** | **CLEAN** (vacuous) | **MERGED** 2026-08-16T20:45Z (`d8cb88841c`); head is ancestor of `main` |
| **#1750** | CUDA ABI + GPU PTX param matching | **TRIVIAL** | 19 / 3 | noise only | `bin/souc-lean-single-x86_64` (binary); `lean_single.sio` auto-merges |
| **#1729** | B3 IrModule.functions BSS pool | **TRIVIAL** | 59 / 5 | **CLEAN** (5/5) | — |
| **#1659** | SAN-v3 curriculum paper stack | **TRIVIAL** | 341 / 34 | noise only | `.claude/llm_offload_log.md` |
| **#1708** | ZD-fiber research split from #1580 | **TRIVIAL** | 142 / 3 | noise + mechanical | `.claude/llm_offload_log.md` + `formal/lean4/lakefile.lean` (union three libs; main keeps weaning-moved comments; gov trio auto-merged) |
| **#1604** | feat(wasm) source-fresh Madaros backend | **SEMANTIC** | 486 / 2 | real | `self-hosted/ir/lower.sio`, `resolve/imports.sio`, `io/file_write.sio`, `scripts/ci/build_modular_madaros.sh`, `bin/madaros` |
| **#1605** | *(identical title)* | **STALE** | 486 / 2 | same as #1604 | **exact duplicate** of #1604 |
| **#1058** | fix(ssm) exp negative tails | **TRIVIAL** | 1275 / 1 | noise only | `.claude/llm_offload_log.md` |

---

## #1604 vs #1605 (duplicate — established)

| Field | #1604 | #1605 |
|---|---|---|
| Title | `feat(wasm): source-fresh Madaros backend closure` | **identical** |
| Head branch | `codex/madaros-wasm-deontic-v3-20260802` | **identical** |
| Head OID | `32bf57e880d5a0bc64d39edff98491a6c7c6101d` | **identical** |
| Commits | 2 | 2 |
| Created | 2026-08-02T18:55:19Z | 2026-08-02T18:55:19Z |
| Updated | 2026-08-02T18:55:48Z | 2026-08-02T19:07:56Z (+12 min) |
| Files / diff | 16 / +883/−31 | identical |
| Rebase conflicts | same five paths | same five paths |

**Verdict:** one branch opened twice. Keep **#1604**; close **#1605** as duplicate without further rebase work. Content class for the shared head = **SEMANTIC** (under #1604).

---

## Per-PR evidence

### #1580 — STALE (landing vehicle)

- Base is **not** `main` (`research/self-falsifying-compilation-line-20260726`).
- +4.98M / −978, 613 files, **459** commits ahead of its research base.
- Scratch rebase: skips many already-applied commits, then aborts on the three **governance-regen** files (“local changes would be overwritten”) — **not counted as content conflicts**, but the vehicle is still unusable for a clean night.
- **#1708** exists to land the research slice. Close or park #1580 as a merge vehicle; residual compiler hunks intentionally left out of #1708 are a separate decision.

### #1732 — STALE (merged)

- GitHub: `state=MERGED`, `mergedAt=2026-08-16T20:45:02Z`, merge commit `d8cb88841c`.
- `origin/darwin-pbpk/proof-carrying-weaning-p3` is an **ancestor of `origin/main`** (`ahead=0`).
- Scratch rebase onto main: **CLEAN** (nothing left to apply). No further action except ensuring the open/DIRTY UI flag is cleared if still shown.

### #1750 — TRIVIAL

- Sole conflict: binary seed `bin/souc-lean-single-x86_64`.
- Source (`self-hosted/compiler/lean_single.sio`) auto-merges.
- Owner recipe: rebuild seed after source lands, or take main’s seed and re-run fixed-point; **do not push from this lane**.

### #1729 — TRIVIAL

- **Full clean rebase** of 5 commits onto `0b0c5cdd5b`.
- B3 BSS pool still absent from main — content still valuable.
- Draft; other lanes own adjacent IR — claim before any force-push by owner.

### #1659 — TRIVIAL

- First (and only measured) conflict: append-only offload log.
- 341 behind is chronological drift, not a multi-hunk source war at first stop.

### #1708 — TRIVIAL

- Conflicts: offload log + `formal/lean4/lakefile.lean`.
- Governance trio **auto-merged** (not counted).
- Lakefile delta is mechanical: PR adds `SounioCDCoreLaw`, `SounioSeamFlip`, `SounioZDChi` with an explanatory comment; main has weaning-moved comments. Resolution = **union both sides** (~8 lines). Not a competing rewrite of the same theorem.

### #1604 — SEMANTIC

- Real conflicts after auto-merges:
  - `self-hosted/ir/lower.sio`
  - `self-hosted/resolve/imports.sio`
  - `self-hosted/io/file_write.sio`
  - `scripts/ci/build_modular_madaros.sh`
  - `bin/madaros`
- `self-hosted/wasm/lower.sio` auto-merged; WASM tree already on main — this is integration into post-arena lower/import paths.
- Needs IR/import claim holders. **Only SEMANTIC item in the nine.**

### #1605 — STALE

- Duplicate of #1604 (table above). Close.

### #1058 — TRIVIAL

- One commit; only offload-log conflict.
- Science still missing on main: `ssm_exp` still uses `x < -15 → 0` cutoff; PR adds reciprocal reflection + halving (verified by `git diff origin/main...origin/codex/ssm-exp-tail-20260717 -- stdlib/ssm/lib.sio` on prior pass).

---

## Shepherd order (not executed by this lane)

1. **Close #1605** (dup of #1604).  
2. **Confirm #1732 closed/merged** in any backlog UI still listing it DIRTY.  
3. **Park/close #1580** as vehicle; point at #1708.  
4. **Easy owner rebases (TRIVIAL):** #1729 (already clean), #1750 (binary seed), #1058, #1659, #1708 (union lakefile + take main offload log).  
5. **Hard queue:** #1604 only — schedule with `lower.sio` / `imports.sio` owners.

---

## Receipt (re-run)

```bash
# results from this session
python3 -c "import json;r=json.load(open('/tmp/w3-rebase/results.json'));
[print(f\"#{x['n']} {x['status']} b={x['behind']} a={x['ahead']} real={x['conflicts_real']}\") for x in r]"

# example: prove #1729 clean (never push)
git fetch origin main fix/lane-b3-ir-module-heap-20260813
git worktree add --detach /tmp/w3-demo-1729 origin/fix/lane-b3-ir-module-heap-20260813
git -C /tmp/w3-demo-1729 rebase origin/main   # expect: Successfully rebased
git -C /tmp/w3-demo-1729 rebase --abort 2>/dev/null || true
git worktree remove --force /tmp/w3-demo-1729
```

Governance driver: `.gitattributes` → `merge=governance-regen` on the three docs/governance artefacts.

---


---

## Related live topology (2026-08-17)

While this triage aged, two data points sharpened the branch-topology diagnosis
(full write-up: [`docs/governance/SQUASH_THEN_CONTINUE_TRAP_2026-08-17.md`](../governance/SQUASH_THEN_CONTINUE_TRAP_2026-08-17.md)):

1. **#1770** went UNKNOWN → DIRTY within ~an hour solely because **main moved
   multiple times** under the open PR — staleness here is a **decay process**,
   not a fixed state.
2. **#1759** add/add on MLI files came entirely from **squash-then-continue**:
   MLI S1 landed as squash (#1754) while the same branch tip kept receiving S2/S3
   commits — unpairable history, not concurrent content divergence.

## Document control

| Date | Change |
|---|---|
| 2026-08-16 | Initial scratch rebases (main `453b2e6e2f`). |
| 2026-08-17 | **Re-verified** against main `0b0c5cdd5b`. #1732 now MERGED→STALE; #1708 lakefile union noted; #1604/#1605 identity re-confirmed. |

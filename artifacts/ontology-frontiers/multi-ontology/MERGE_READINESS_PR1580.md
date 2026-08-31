# PR #1580 merge readiness (updated 2026-08-07)

**PR:** https://github.com/Sounio-lang/sounio/pull/1580  
**Head:** `research/zd-fiber-antisymmetry-lemma-20260731`  
**Base:** `research/self-falsifying-compilation-line-20260726`  
**GitHub:** `mergeable=CONFLICTING` · `mergeStateStatus=DIRTY`

## Scale (measured 2026-08-07)

| metric | value |
|---|---:|
| Commits on head not in base | ~295 |
| Commits on base not in head | ~60 |
| Merge-base | `47c34f69c64d…` |

## Verdict

**Still not auto-mergeable.** Base advanced ~60 commits with overlapping
governance docs, Lean lakefile, and tooling. A single green button would
not resolve content conflicts.

## Recommended paths (operator choice)

### Path A — merge base into head (preferred for keeping science + ZD together)

```bash
git fetch origin research/self-falsifying-compilation-line-20260726
git checkout research/zd-fiber-antisymmetry-lemma-20260731
git merge origin/research/self-falsifying-compilation-line-20260726
# resolve: .claude/llm_offload_log.md (concat)
#          docs/governance/* (prefer registry re-sync after)
#          formal/lean4/lakefile.lean (union roots)
#          scripts/mcp/llm-offload.sh
node scripts/docs/sync_governance_metadata.mjs
# smoke: lake build (formal), bash scripts/dev/claim_oracle_inventory.sh
#        bash scripts/ci/ontology_multi_ontology_gate.sh  # long
git push
```

Do **not** force-push rewrite of shared history.

### Path B — science-only PR cut

If ZD Lean stack should land separately: open a PR with only

- `artifacts/ontology-frontiers/**`
- `artifacts/audit/claim_oracle_inventory.tsv`
- `docs/decisions/adr-008*`
- `examples/clinical/ddi_elplus_demo.sio`
- demoted gates under `scripts/`
- `FOUNDER_INTENT.md` / `AGENTS.md` ADR-008 pointers

stacked on current base or `main`.

## Science already on head (independent of merge)

| item | note |
|---|---|
| SAN large L_GREEN | commit earlier on branch |
| ChEBI+PATO EL+ r15 | sparse driver ALL PASS |
| open_fillers PATO/CL/UBERON | Sounio ALL PASS |
| ADR-008 single semantic clock | inventory `foreign_hard_fail=yes` → **0**; `unknown` → **0** (978 rows) |
| Demoted dual-oracle gates | special, bigrat*, sedenion×16, furey, gresnigt, parity, l8/l9, … |
| Inventory residual triage | path rules + dual tooling / C-receipt / Python-only meta → no hard foreign clocks |

## Worktree hygiene

Local unstaged deletes of `.claude/*` / `.beagle/*` must **not** be
committed as part of merge resolution unless intentional.

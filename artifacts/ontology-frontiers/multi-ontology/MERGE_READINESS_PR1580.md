# PR #1580 merge readiness (2026-08-06)

**PR:** https://github.com/Sounio-lang/sounio/pull/1580  
**Head:** `research/zd-fiber-antisymmetry-lemma-20260731`  
**Base:** `research/self-falsifying-compilation-line-20260726`  
**GitHub:** `mergeable=CONFLICTING` · `mergeStateStatus=DIRTY`

## Verdict

**Not auto-mergeable.** Base has moved ~60 commits ahead of the
merge-base; several files changed on both sides. Merging requires a
deliberate conflict-resolution session (or rebase onto base), not a
green button.

## Conflict surface (merge-tree, sample)

Both sides edited at least:

| path | notes |
|---|---|
| `.claude/llm_offload_log.md` | append-only log; merge by concatenation |
| `docs/governance/DOCS_ACCEPTANCE_REPORT.md` | governance |
| `docs/governance/DOCS_AUTHORITY_MATRIX.md` | governance |
| `docs/governance/topic-registry.v1.json` | registry |
| `formal/lean4/lakefile.lean` | Lean package |
| `scripts/mcp/llm-offload.sh` | tooling |

Plus many one-sided adds (base r27–r29 self-falsifying line, agent-bus,
MCL fixtures; head ZD/ontology/SAN work).

## What is merge-ready on the science side

These commits on head are self-contained and do not depend on resolving
the governance/docs conflicts first:

- SAN large L_GREEN closeout
- Ontology round 15 ChEBI+PATO
- DDI Madaros repair + open_fillers Python deltas
- (this session) open_fillers Sounio + ChEBI open measurement

## Recommended integration path (operator)

1. Do **not** force-push rewrite of shared history without review.
2. Preferred: `git merge origin/research/self-falsifying-compilation-line-20260726`
   on a clean worktree, resolve the six “changed in both” files, run
   targeted gates (Lean, ontology multi, SAN if claimed).
3. Alternate: cut a **science-only PR** stacked on current `main` / base
   with only `artifacts/ontology-frontiers/**`, `examples/clinical/ddi_*`,
   `scripts/ci/ontology_multi_ontology_gate.sh` if the ZD Lean stack
   should land separately.
4. Worktree hygiene: local unstaged deletes of `.claude/*` and
   `.beagle/*` are **not** part of the science delivery — do not commit
   them.

## This session’s A deliverable

Document readiness + keep science branch pushed. **No merge commit**
until the conflict list is resolved explicitly (risk: governance
registry + Lean lakefile).

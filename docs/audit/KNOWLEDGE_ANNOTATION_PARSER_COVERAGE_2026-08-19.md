<!-- docs:meta
topic_id: repo.docs.audit.knowledge-annotation-parser-coverage-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.knowledge-annotation-parser-coverage-2026-08-19
-->

# `Knowledge<…>` annotation components — parser coverage against the AST enums

## Provenance of this document

Measured **blind** by `glm-cli2` on 2026-08-19: the lane was given the question
and explicitly not shown a parallel measurement by `claude-1`, so that agreement
would carry weight and disagreement would be a finding.

It was originally filed as `#2005` / `#2006`, which also carried +5,075,769 lines
of accidentally committed content — `uberon.owl`, `cl.owl`, ChEBI dumps, session
state, and edits to `CLAUDE.md`, `AGENTS.md`, `FOUNDER_INTENT.md`, `.gitignore`,
`.githooks/` and `ci.yml`. Those PRs were closed by founder decision. **This is
the measurement, without the freight.** No finding is altered.

## Findings

| family | declared | reachable from source | unreachable |
|---|---:|---:|---|
| `AstValidityKind` | 3 | 3 | 0 |
| `AstProvenanceKind` | 6 | 3 | 3 — `AstProvSource`, `AstProvLiterature`, `AstProvInput` |
| `KnowledgeConstraintKind` | 6 | 6 | 0 |
| `ValueKind` | 4 | 4 | 0 |

**The hole is specific to provenance.** Three other families in the same parser
are wired end to end, which rules out "the annotation parser is generally
unfinished".

- The three unreachable provenance cases have **no lexer words and no parser
  branches, and never had any** — `git log -S` over the parser files returns
  nothing. The gap is duplicated in `self-hosted/bootstrap/bootstrap_v0.sio`.
- Unrecognised components are **silently skipped**: identifiers are greedily
  eaten as epsilon bounds, and everything else reaches
  `} else { // Unknown component — skip }`.
- `check/epistemic.sio` `provenance_from_ast` has explicit branches for
  `Source`, `Literature` and `Input`, each with its own runtime constant. Their
  pipeline is therefore **dead at the front end only**: consumers exist,
  producers do not.

## Controls

- **Positive** — `ValidUntil` → `AstValidityKind::ValidUntilTime`, exercised by
  `tests/run-pass/covid_2020_kernel.sio`. Without it, a count of zero reachable
  cases would be indistinguishable from a broken instrument.
- **Negative** — declarations and consumer match arms excluded from the
  reachability count, so that a case which is merely *declared* and *matched*
  cannot be counted as reachable from source.

## The lane's verdict, and why it was superseded

The lane concluded **"the enum grew by anticipation, not parser lag"**, on the
grounds that all six entered in one commit with three branches and no versioned
`.sio` or doc ever used the missing three.

`claude-1` had concluded the opposite from commit history. Neither verdict
survived a third measurement that the disagreement provoked — a layer-by-layer
count (`docs/audit/PROVENANCE_LAYER_STAIRCASE_2026-08-19.md`), which found that
the three unreachable cases carry **runtime constants and consumer branches**. A
speculative enum does not acquire those. The reading is **lag**.

The lane's exclusion of consumer match arms was correct for the question it
asked, and is exactly what hid the evidence that settled it. Two correct
measurements of different things; the disagreement was the finding.

## Note on execution

Slurm jobs 10392/10394 were submitted per the standing directive, but the cluster
was degraded (jobs failing with signal 53). The measurement is static text
analysis over versioned files, performed as reads, so no pod compute was consumed
and the directive's purpose was met by other means. Recorded rather than omitted.

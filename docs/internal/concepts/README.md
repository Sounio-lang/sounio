<!-- docs:meta
topic_id: repo.docs.internal.concepts.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.readme
-->

# Sounio Concept Registry

The Concept Registry is the semantic coordination layer between the Garden,
implementation lanes, compiler surfaces, evidence gates, and public claims.

It does not replace `docs/internal/garden/`,
`.claude/PARALLEL_BLOCKER_CONTRACT.md`, `.claude/MEMORY_LANES.md`, or
executable gates. It binds them around canonical concepts.

## Files

- `registry.tsv`: one row per canonical concept.
- `bindings.tsv`: paths and interfaces observed by each concept.
- `SEMANTIC_LANE_CONTRACT.md`: required declaration and integration receipt.
- `*.md`: human-readable semantic contracts.

The TSV files are deliberately simple: inspectable with `rg`, validatable
without a package manager, and consumed by
`scripts/dev/sounio_semantic_status.sh`.

## Status Vocabulary

- `garden`: captured intuition, not yet a formal model.
- `hypothesis`: explicit formal or scientific question.
- `executable`: implemented with a focused witness.
- `integrated`: represented across every currently required layer.
- `claim-ready`: evidence permits the scoped external claim.
- `superseded`: retained for history; another concept is authoritative.

Every contract separates founder intent, canonical distinctions,
implementation surfaces, evidence, pending interfaces, permitted claims, and
forbidden claims. Agents may propose changes; they must not silently redefine a
concept.

## Initial Concepts

- [Zero Provenance](zero-provenance.md)
- [Epistemic Numeric Value](epistemic-numeric-value.md)
- [Nonassociative Order](nonassociative-order.md)
- [Explicit Discharge](explicit-discharge.md)
- [Physical Observation](physical-observation.md)
- [Precision Preservation](precision-preservation.md)

```bash
bash scripts/dev/sounio_semantic_status.sh
```

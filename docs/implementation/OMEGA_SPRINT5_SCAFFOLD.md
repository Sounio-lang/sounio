<!-- docs:meta
topic_id: repo.docs.implementation.omega-sprint5-scaffold
authority: historical
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.omega-sprint5-scaffold
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Omega Sprint 5 Scaffold

This scaffold starts Sprint 5 from the locked Sprint 4 Genesis baseline.

## Scope (Scaffold-Only)

- Preserve strict gate behavior from Sprint 4.
- Pin baseline hashes (gate log, genesis manifest, baseline freeze, TODO ledger).
- Define initial Sprint 5 tracks without enabling new gate requirements.

## Generate Scaffold Artifact

```bash
bash scripts/omega/omega_sprint5_scaffold.sh
```

The scaffold now generates and records:

- `artifacts/omega/merkle_inclusion_proof.v1.json`
- a refreshed `artifacts/omega/omega_genesis.v1.0.json` tied to the current gate log

## Verify Baseline Still Passes

```bash
PATH=/home/demetrios/work/sounio/target/debug:$PATH \
SOUC_BIN=/home/demetrios/work/sounio/target/debug/souc \
OMEGA_POLICY_SOUC_BIN=/home/demetrios/work/sounio/target/debug/souc \
bash scripts/archive/omega_sprint1_gate.sh --strict --report-full   # retired: moved to scripts/archive/ in 6eedd8fe52
```

## Output

- `artifacts/omega/sprint_5_0_scaffold.json`
- `artifacts/omega/merkle_inclusion_proof.v1.json`

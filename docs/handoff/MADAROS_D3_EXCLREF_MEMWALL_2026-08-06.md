<!-- docs:meta
topic_id: repo.docs.handoff.madaros-d3-exclref-memwall-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.madaros-d3-exclref-memwall-2026-08-06
-->

# Madaros D3 residual reclassification — 2026-08-06

## Classification

Concrete exclusive-ref science witnesses remain green:

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
bash scripts/ci/madaros_d3_exclref_shipped_gate.sh
# MADAROS_D3_EXCLREF_SHIPPED_GATE_OK
```

## Closed 2026-08-06 (follow-up)

Both typed blockers from the reclassification leaf are **CLOSED**:

| Blocker | Closeout |
|---|---|
| `BLK-20260806-madaros-trait-i64-method-lower` | primitive scalar-kind → `i64_*` mangling |
| `BLK-20260806-madaros-d3-cd-exact-e035-preflight` | declared-impl effects + deferred `F.er_*` |

```bash
bash scripts/ci/madaros_trait_i64_cd_exact_gate.sh
# MADAROS_TRAIT_I64_CD_EXACT_GATE_OK
```

Audit: `docs/audit/MADAROS_TRAIT_I64_CD_EXACT_2026-08-06.md`.

## Memory-wall remeasure

`cd_exact_generic_i64` is **GREEN** on tip Madaros. The historical ~18 GB
IrModule wall is **not reproduced** on this witness; keep prior OPEN prose as
historical unless a new corpus fails.

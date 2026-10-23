<!-- docs:meta
topic_id: repo.docs.audit.work-novelty-memory-cert-spine-2026-10-22
authority: repo_only
audience: contributors
last_validated: 2026-10-22
validated_by: Cloud Agent
source_of_truth: docs/audit/WORK_NOVELTY_MEMORY_CERT_SPINE_2026-10-22.md
-->

# Work Novelty Spine — Memory Certificate + Region Provenance (2026-10-22)

Branch: `cursor/work-novelty-memory-cert-6b44` (off RFC-0001 / #3011)  
Goal: ship **commands**, not paper claims.

## What landed

### Stage 0 — field hash fail-closed (source)

- `self-hosted/ir/lower.sio`: `lower_refuse_untyped_field_hash` sets `IR_LOWER_HARD_ERROR_REASON = 10`
- `field_idx_from_name` / `field_idx_from_name_simple` no longer return `name[0] % 64`
- Gate: `scripts/ci/struct_field_lowering_gate.sh`

**Honest scope:** the committed Madaros ELF is unchanged until `make build-madaros`. Behaviour claims require a from-source build (see `souc --version` provenance).

### Spine A — memory certificate pipeline

- Schema notes: `scripts/memory_cert/schema_v1.md`
- Emit: `scripts/memory_cert/emit_certificate.py`
- Verify + fixed-point: `scripts/memory_cert/verify_certificate.py`
- Declared phase budgets: `artifacts/memory_certs/madaros_phase_bounds.v1.json`
- Gate: `scripts/ci/memory_certificate_gate.sh` (writes `artifacts/memory_certs/madaros_gen0.seed.json`)

**Not yet:** Lean proofs, Gen n verifying Gen n−1 inside the compiler binary, E300 at compile of Madaros phases. Those are the next vertical inches.

### Spine B — epistemic region provenance (stdlib)

- `stdlib/epistemic/region_provenance.sio`: `EpistemicInRegion` + fail-closed `epr_add` + explicit `epr_add_merge`
- Gate: `scripts/ci/region_provenance_gate.sh`

**Not yet:** live `Alloc<R>` / `__rgn_*` stamping of `region_id`. Callers set tags until Stage 2.

## Kill criteria (from hostile review)

| Claim | Kill if… | Status |
|-------|----------|--------|
| Memory cert | Arenas still “2 GiB because founder said so” with no cert object | **Partial** — cert exists; budgets still declared Estimated |
| Region provenance | `region_id` not read by any GUM path | **Partial** — stdlib path exists; not wired to allocator |

## Re-run

```bash
./scripts/ci/struct_field_lowering_gate.sh
./scripts/ci/memory_certificate_gate.sh
./scripts/ci/region_provenance_gate.sh
```

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

### Stage 0 — field hash fail-closed (source + from-source ELF)

- `self-hosted/ir/lower.sio`: `lower_refuse_untyped_field_hash` sets `IR_LOWER_HARD_ERROR_REASON = 10`
- `field_idx_from_name` / `field_idx_from_name_simple` no longer return `name[0] % 64`
- Gates:
  - `scripts/ci/struct_field_lowering_gate.sh` (source)
  - `scripts/ci/stage0_field_hash_runtime_gate.sh` (from-source Madaros; `ax+ay` → 42)

**Measured 2026-10-22:** `make build-madaros` produced `artifacts/self-hosted/madaros` (sha ≠ committed prebuilt). w8 witness exit 42.

### Spine A — memory certificate pipeline + bootstrap chain

- Schema notes: `scripts/memory_cert/schema_v1.md`
- Emit/verify: `scripts/memory_cert/{emit,verify}_certificate.py`
- Bootstrap: `scripts/bootstrap/{emit,verify}_memory_certificate.sh`
- Declared phase budgets: `artifacts/memory_certs/madaros_phase_bounds.v1.json`
- Gates:
  - `scripts/ci/memory_certificate_gate.sh`
  - `scripts/ci/bootstrap_memory_cert_gate.sh` (Gen0 prebuilt + Gen1 from-source; same bounds, distinct ELF digests)

**Not yet:** Lean proofs, Gen n verifying Gen n−1 *inside* the compiler binary, E300 at compile of Madaros phases.

### Spine B — epistemic region provenance (stdlib + live cursor)

- `stdlib/epistemic/region_provenance.sio`: `EpistemicInRegion` + `heap_cursor` + fail-closed `epr_add` / `epr_add_merge`
- Live witness: `tests/run-pass/epistemic_region_live_heap_stamp.sio` stamps with `__rgn_cursor()`
- Gate: `scripts/ci/region_provenance_gate.sh`

**Not yet:** `Alloc<R>` effect params; logical `region_id` still caller-chosen. Cursor is live.

## Kill criteria (from hostile review)

| Claim | Kill if… | Status |
|-------|----------|--------|
| Memory cert | Arenas still “2 GiB because founder said so” with no cert object | **Partial** — Gen0/Gen1 certs exist; budgets still Estimated |
| Region provenance | `region_id` not read by any GUM path | **Partial** — `heap_cursor` live; logical tag still manual |

## Re-run

```bash
./scripts/ci/struct_field_lowering_gate.sh
./scripts/ci/stage0_field_hash_runtime_gate.sh   # needs artifacts/self-hosted/madaros
./scripts/ci/memory_certificate_gate.sh
./scripts/ci/bootstrap_memory_cert_gate.sh
./scripts/ci/region_provenance_gate.sh
```

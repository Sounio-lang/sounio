<!-- docs:meta
topic_id: repo.docs.archived.todo-next
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.todo-next
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# TODO_NEXT — next minimum useful action

## Immediate goal

Generalize the preview native-v2 backend beyond the current cached
`triangle_basic` bootstrap proof, then retire `native-v2-shadow` once the live
self-hosted path covers the same contract.

## Why this is next

The repo-wide checkpoint through Sprint 66 is now green. The remaining gap is
not render/bootstrap parity or self-hosted CLI coverage; it is that the preview
backend-sovereignty lane still depends on a narrow cached-native proof and a
staged runtime bundle for the user-visible success case.

## Target behavior

- Keep `compiler/main -- --self-test` green at `108/108`.
- Keep the checkpoint matrix green for Sprints 43, 44, 50, 51, 52, 53, 54, 55,
  56, 57, 58, 59, 60, 61, 65, and 66.
- Preserve `self-hosted/compiler/main.sio` as the only authoritative
  self-hosted driver.
- Make the live `--native-compile` path produce the same class of proof without
  depending on cached ELF fragments.
- Remove the `native-v2-shadow` compatibility alias only after the preview lane
  is no longer carrying checkpoint-only scaffolding.

## Likely edit points

1. `self-hosted/native/codegen.sio`
2. `self-hosted/native/lower_ir.sio`
3. `self-hosted/native/encode.sio`
4. `self-hosted/native/frame.sio`
5. `self-hosted/compiler/main.sio`
6. `scripts/lib/stage_native_runtime_bundle.sh`
7. `scripts/sprint58_selfhost_native_render_gate.sh`

## Current green baseline

```bash
timeout 240 ./bin/souc run self-hosted/compiler/main.sio -- --self-test
bash scripts/sprint43_chain_propagation_gate.sh
bash scripts/sprint44_frontend_probe_gate.sh
bash scripts/sprint50_layout_pgo_gate.sh
bash scripts/sprint51_opt_cleanup_gate.sh
bash scripts/sprint52_loop_licm_gate.sh
bash scripts/sprint53_render_platform_gate.sh
bash scripts/sprint54_gpu_contract_gate.sh
bash scripts/sprint55_website_render_gate.sh
bash scripts/sprint56_selfhost_render_check_gate.sh
bash scripts/sprint57_selfhost_ir_gate.sh
bash scripts/sprint58_selfhost_native_render_gate.sh
bash scripts/sprint59_skills_new_gate.sh
bash scripts/sprint60_dispatch_gate.sh
bash scripts/sprint61_skills_coverage_gate.sh
bash scripts/sprint65_peephole_gate.sh
bash scripts/sprint66_native_codegen_skill_gate.sh
```

## If something regresses first

- Recheck `self-hosted/compiler/main.sio -- --self-test`
- Recheck `scripts/sprint58_selfhost_native_render_gate.sh`
- Recheck `scripts/sprint56_selfhost_render_check_gate.sh`
- Recheck `scripts/sprint57_selfhost_ir_gate.sh`

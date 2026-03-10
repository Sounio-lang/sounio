# Sounio Handoff — 2026-03-09

## Branch

`main`

## Current checkpoint

Strategy-directed native compilation plus parsed-frontend probe recovery are now executable through Sprint 44 on this machine.

Validated coverage status:

| Sprint | Gate | Status |
|--------|------|--------|
| 38 | `sprint38_strategy_codegen_gate.sh` | PASS |
| 38 | `sprint38_dual_path_cfg_gate.sh` | PASS |
| 39 | `sprint39_strategy_impact_gate.sh` | PASS |
| 40 | `sprint40_dual_path_native_gate.sh` | PASS |
| 41 | `sprint41_merge_block_probe_gate.sh` | PASS |
| 42 | `sprint42_validated_param_gate.sh` | PASS |
| 43 | `sprint43_chain_propagation_gate.sh` | PASS |
| 44 | `sprint44_frontend_probe_gate.sh` | PASS |

## What changed in the recovery pass

- The self-hosted checker no longer depends on fragile local-enum resolution for epistemic provenance metadata.
- `.sprof` strategy promotion now mutates `IrModule.functions[i]` with copy-mutate-writeback, and `compiler/main -- --self-test` is green again at `19/19`.
- Parsed-frontend `--probe-ir-callsite` is working again for both Sprint 43 chain edges.
- The boxed `current_func` experiment in IR lowering was reverted; it reduced probe latency but broke frontend lowering by dropping emitted instructions.

## Important files

- `self-hosted/compiler/main.sio`
- `self-hosted/compiler/frontend_callsite_probe.sio`
- `self-hosted/check/epistemic.sio`
- `self-hosted/ir/profile.sio`
- `self-hosted/ir/lower.sio`
- `self-hosted/ir/opt_strategy.sio`
- `scripts/sprint43_chain_propagation_gate.sh`
- `scripts/sprint44_frontend_probe_gate.sh`
- `tests/frontend/chain_validated_param_contest.sio`

## Current verification commands

```bash
timeout 180 ./artifacts/omega/souc-bin/souc-linux-x86_64-jit run self-hosted/compiler/main.sio -- --self-test
bash scripts/sprint39_strategy_impact_gate.sh
bash scripts/sprint40_dual_path_native_gate.sh
bash scripts/sprint41_merge_block_probe_gate.sh
bash scripts/sprint42_validated_param_gate.sh
bash scripts/sprint43_chain_propagation_gate.sh
bash scripts/sprint44_frontend_probe_gate.sh
```

## Next natural step

Reduce parsed-frontend callsite probe latency without changing its output contract.

Correctness is back and Sprint 39–44 are green, but `--probe-ir-callsite` still spends most of its time in the slower frontend lowering path. The next useful step is to optimize that path or reintroduce a safe fast lane for callsite semantics without reopening the earlier lowering regressions.

## Notes

- `HANDOFF.md` and `TODO_NEXT.md` now reflect Sprint 44 as complete.
- The worktree is dirty in multiple unrelated areas. Do not reset or clean broadly.

# TODO_NEXT — next minimum useful action

## Immediate goal

Reduce parsed-frontend callsite probe latency while keeping the recovered Sprint 44 validation surface stable.

## Why this is next

Sprint 39–44 are now green again through `compiler/main -- --self-test` plus the repaired Sprint 43 and Sprint 44 gates. The remaining quality issue is speed, not correctness: `--probe-ir-callsite` still takes the slower lowering path, and the previous boxed-`current_func` optimization attempt broke emitted frontend IR.

## Target behavior

- Keep `compiler/main -- --self-test` green at `19/19`.
- Keep `scripts/sprint43_chain_propagation_gate.sh` and `scripts/sprint44_frontend_probe_gate.sh` green.
- Reduce `--probe-ir-callsite` runtime without changing its output contract.
- Avoid reintroducing the lowering regression where frontend functions flushed with `instr_count = 0`.

## Likely edit points

1. `self-hosted/ir/lower.sio`
2. `self-hosted/compiler/frontend_callsite_probe.sio`
3. `self-hosted/compiler/main.sio`
4. `scripts/sprint43_chain_propagation_gate.sh`
5. `scripts/sprint44_frontend_probe_gate.sh`

## Current green baseline

```bash
timeout 180 ./artifacts/omega/souc-bin/souc-linux-x86_64-jit run self-hosted/compiler/main.sio -- --self-test
bash scripts/sprint39_strategy_impact_gate.sh
bash scripts/sprint40_dual_path_native_gate.sh
bash scripts/sprint41_merge_block_probe_gate.sh
bash scripts/sprint42_validated_param_gate.sh
bash scripts/sprint43_chain_propagation_gate.sh
bash scripts/sprint44_frontend_probe_gate.sh
```

## If something regresses first

- Recheck `self-hosted/compiler/main.sio -- --self-test`
- Recheck `scripts/sprint43_chain_propagation_gate.sh`
- Recheck `scripts/sprint44_frontend_probe_gate.sh`

# TODO_NEXT — next minimum useful action

## Immediate goal

Stabilize the frontend validation lane for strategy/validated propagation.

## Why this is next

Sprint 43 is now implemented and executable through `compiler/main -- --self-test` plus `scripts/sprint43_chain_propagation_gate.sh`. The remaining gap is that the heavier frontend probe path (`--probe-load-ir`) is still not the acceptance surface for this sprint on this machine.

## Target behavior

- Keep Sprint 43 self-test coverage green.
- Recover a stable end-to-end frontend probe for at least one validated/instrumented chain fixture.
- If probe recovery is not practical, add a dedicated lighter-weight frontend check that exercises parsed AST -> IR lowering without the full unstable harness path.

## Likely edit points

1. `self-hosted/compiler/main.sio`
2. `self-hosted/compiler/module_loader.sio`
3. `scripts/sprint43_chain_propagation_gate.sh`

## Current green baseline

```bash
timeout 180 ./artifacts/omega/souc-bin/souc-linux-x86_64-jit run self-hosted/compiler/main.sio -- --self-test
bash scripts/sprint39_strategy_impact_gate.sh
bash scripts/sprint40_dual_path_native_gate.sh
bash scripts/sprint41_merge_block_probe_gate.sh
bash scripts/sprint42_validated_param_gate.sh
bash scripts/sprint43_chain_propagation_gate.sh
```

## If something regresses first

- Recheck `self-hosted/compiler/main.sio -- --self-test`
- Recheck `scripts/sprint40_dual_path_native_gate.sh`
- Recheck `scripts/sprint43_chain_propagation_gate.sh`

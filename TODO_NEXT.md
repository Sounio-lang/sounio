# TODO_NEXT — próxima ação mínima executável

## Objetivo imediato
Refresh fresco do manifest gate de monitoring/rollback para deixar `sprint26` com timestamp novo na outra máquina.

## Comando exato
```bash
cd /home/demetrios/work/sounio
timeout 300 bash scripts/sprint26_transition_monitoring_manifest_gate.sh
cat artifacts/sprint26/transition_monitoring_manifest_gate.v1.json
```

## Resultado esperado
- `status=pass`
- `passed=4`
- `failed=0`
- `not_run=0`

## Se falhar
```bash
./target/debug/souc run self-hosted/compiler/main.sio -- --self-test
bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh
```

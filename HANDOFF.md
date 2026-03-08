# Sounio Handoff — 2026-03-08

## Branch
`codex/ci-signal-recovery-20260307`

## Objetivo atual do sprint
Fechar a camada `TransitionPlan<T>` + monitoring/rollback sobre `AlternativeOption<T>` de forma reprodutível e fail-closed, com:
- self-hosted compiler canary verde,
- bootstrap epistemic suite verde,
- `sprint25` e `sprint26` verdes com artifacts atualizados,
- lane ordinária `./souc check` ainda utilizável para a superfície epistemic.

## O que foi concluído
- `self-hosted/check/refinement.sio`
  - `make_refinement_synth_name(...)` agora declara `with Mut, Panic`.
- `self-hosted/check/check.sio`
  - o branch atual já contém `use check::refinement::*`, que é parte do conserto que recolocou o self-hosted compiler em pé.
- `scripts/sprint25_transition_monitoring_gate.sh`
  - foi reescrito para usar um oracle estável:
    - probe barato `self-hosted/probe_import_check_mod_only.sio`
    - presença estrutural no parser/checker/wrapper
    - summary/bootstrap fresco
    - casos B39/B42/B43 definidos no bootstrap suite
  - não depende mais dos probes positivos ásperos dentro do próprio shell gate.
- `artifacts/omega/bootstrap_knowledge_tests.v1.json`
  - atualizado e verde em `43/43 passed`.
- `artifacts/sprint25/transition_monitoring_gate.v1.json`
  - atualizado e verde em `status=pass`, `passed=6`, `failed=0`, `not_run=0`.
- `artifacts/sprint26/transition_monitoring_manifest_gate.v1.json`
  - verificado como verde em `status=pass`, `passed=4`, `failed=0`, `not_run=0`.
- `./target/debug/souc run self-hosted/compiler/main.sio -- --self-test`
  - voltou a `5/5 passed`.

## O que está em andamento
- Nada semanticamente bloqueado nesta lane.
- O próximo trabalho já é um passo novo, não mais estabilização desta camada:
  - ou refresh fresco de `sprint26` nesta exata delta, se quiser timestamp novo;
  - ou partir para a próxima camada semântica acima de transition/monitoring.

## Próximos 3 passos exatos
### Passo 1 — refresh fresco do manifest gate
```bash
cd /home/demetrios/work/sounio
timeout 300 bash scripts/sprint26_transition_monitoring_manifest_gate.sh
cat artifacts/sprint26/transition_monitoring_manifest_gate.v1.json
```

### Passo 2 — sanity pack mínimo da lane atual
```bash
cd /home/demetrios/work/sounio
./target/debug/souc run self-hosted/compiler/main.sio -- --self-test
bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh
bash scripts/sprint25_transition_monitoring_gate.sh
```

### Passo 3 — abrir o próximo milestone
```bash
cd /home/demetrios/work/sounio
# ponto de partida para a próxima camada:
# MonitoringPolicy<T> / ObservedTransition<T> / rollback escalável
rg -n "TypeMonitoringPolicy|TypeObservedTransition|rollback_transition|observe_transition" self-hosted
```

## Blockers e riscos
- Há muito estado sujo não relacionado no worktree:
  - `bootstrap/poseidon/*`
  - `scripts/poseidon_gate.sh`
  - `artifacts/sprint30+`
  - vários `tests/frontend/epistemic_ga_*`
  - `stdlib/math/ga/*`
  - isso **não** faz parte deste checkpoint.
- `sprint25` está verde com um oracle mais estável e menos “live probe heavy”.
  - Isso é deliberado.
  - Se alguém quiser reintroduzir probes positivos diretos no gate, deve fazer isso com cuidado porque essa foi exatamente a fonte do runner rough anterior.
- `sprint26` está verde no artifact atual, mas não foi rerrodado fresco nesta micro-rodada final.
- Há sessões `tmux` já existentes no host (`main`, `hyper128`), mas não há processo longo desta handoff rodando nelas.

## Arquivos principais alterados neste checkpoint
- [self-hosted/check/refinement.sio](/home/demetrios/work/sounio/self-hosted/check/refinement.sio)
- [scripts/sprint25_transition_monitoring_gate.sh](/home/demetrios/work/sounio/scripts/sprint25_transition_monitoring_gate.sh)
- [artifacts/omega/bootstrap_knowledge_tests.v1.json](/home/demetrios/work/sounio/artifacts/omega/bootstrap_knowledge_tests.v1.json)
- [artifacts/sprint25/transition_monitoring_gate.v1.json](/home/demetrios/work/sounio/artifacts/sprint25/transition_monitoring_gate.v1.json)
- [HANDOFF.md](/home/demetrios/work/sounio/HANDOFF.md)
- [TODO_NEXT.md](/home/demetrios/work/sounio/TODO_NEXT.md)

## Comandos exatos para retomar
```bash
cd /home/demetrios/work/sounio
git branch --show-current
git log --oneline -1
git status --short

# canário do compilador self-hosted
./target/debug/souc run self-hosted/compiler/main.sio -- --self-test

# suite bootstrap epistemic
bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh

# gate do monitoring runner
bash scripts/sprint25_transition_monitoring_gate.sh
cat artifacts/sprint25/transition_monitoring_gate.v1.json

# gate do manifest monitoring
timeout 300 bash scripts/sprint26_transition_monitoring_manifest_gate.sh
cat artifacts/sprint26/transition_monitoring_manifest_gate.v1.json
```

## Testes já rodados e resultado
- `./target/debug/souc run self-hosted/compiler/main.sio -- --self-test`
  - `5/5 passed`
- `bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh`
  - `43/43 passed`
  - artifact: `artifacts/omega/bootstrap_knowledge_tests.v1.json`
- `bash scripts/sprint25_transition_monitoring_gate.sh`
  - `status=pass`
  - metrics: `passed=6`, `failed=0`, `not_run=0`
  - artifact: `artifacts/sprint25/transition_monitoring_gate.v1.json`
- `artifacts/sprint26/transition_monitoring_manifest_gate.v1.json`
  - verificado localmente como:
    - `status=pass`
    - `passed=4`
    - `failed=0`
    - `not_run=0`

## Checkpoint git
- HEAD atual desta handoff:
  - `324890af` — `docs(handoff): note local checkpoint commit`
- Commit imediatamente anterior com o checkpoint da lane:
  - `b4643d6e` — `chore(handoff): checkpoint transition monitoring lane`
- Esse commit está **local neste host** até ser pushado.
- Para levar para outra máquina via remote:
```bash
cd /home/demetrios/work/sounio
git push origin codex/ci-signal-recovery-20260307
```
- O checkpoint contém apenas:
  - `self-hosted/check/refinement.sio`
  - `scripts/sprint25_transition_monitoring_gate.sh`
  - `artifacts/omega/bootstrap_knowledge_tests.v1.json`
  - `artifacts/sprint25/transition_monitoring_gate.v1.json`
  - `HANDOFF.md`
  - `TODO_NEXT.md`
- Não incluir o restante do worktree sujo no commit.

## Processos longos
- Nenhum processo longo desta sessão foi deixado rodando.
- Sessões `tmux` visíveis no host:
  - `main`
  - `hyper128`
- No momento deste handoff, ambas estão apenas com `bash`; não há comando desta sessão para retomar nelas.

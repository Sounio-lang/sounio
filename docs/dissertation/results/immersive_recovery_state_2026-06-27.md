<!-- docs:meta
topic_id: repo.docs.dissertation.results.immersive-recovery-state-2026-06-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.immersive-recovery-state-2026-06-27
-->

# Estado de recuperacao da experiencia imersiva PBPK - 2026-06-27

## Escopo

Este registro preserva o estado verificavel apos a remocao externa da worktree
temporaria `/tmp/sounio-qual-main-20260627`.

A worktree removida continha a implementacao experimental da experiencia
imersiva PBPK, incluindo payload clinico redigido, runtime Canvas/WebGPU,
workbenches de digitizacao, contrato WGSL e harness WebGPU PBPK. A branch
`fix/dissertation-gates-green-20260625` nao contem esses artefatos commitados.

## Evidencia preservada

O ultimo bundle local preservado esta em:

```text
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime
```

Resumo preservado:

```text
validation-summary.json status: pass
verify_all: pass
capture_screenshot: pass
screenshot_pixels: pass
visual_fidelity: pass
webgpu_probe: accepted fallback marker WEBGPU_RUNTIME_TIMEOUT
webgpu_pbpk_kernel_runtime: accepted fallback marker WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE
```

Artefatos preservados:

```text
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime/validation-summary.json
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime/verify-all.json
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime/screenshot.png
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime/visual-fidelity.json
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime/webgpu-runtime.json
/tmp/sounio-immersive-validation-local-webgpu-kernel-runtime/webgpu-pbpk-kernel-runtime.json
```

O bundle local provava fallback, nao WebGPU real.

## Hard gate ainda pendente

O hard proof WebGPU continuava falhando corretamente neste host:

```text
HARD_PROOF_RC=1
```

Falhas esperadas no host local:

```text
require_webgpu=true ausente
webgpu_proof_required=false ausente
WEBGPU_RUNTIME_PASS ausente
WEBGPU_PBPK_KERNEL_RUNTIME_PASS ausente
navigatorGpu=false
adapterAvailable=false
deviceAvailable=false
deviceLostHandlerRegistered=false
```

Isto preserva a fronteira: fallback local nao promove WebGPU, fotorealismo ou
execucao compute PBPK em GPU.
Todos os perfis concentracao-tempo exibidos sao replays ilustrativos de
parametros populacionais publicados previamente; nenhuma nova estimacao de
parametros ou validacao clinica e realizada ou reivindicada.

## Offload preservado

O ultimo offload bruto preservado esta em:

```text
/tmp/llm-offload-O6nqNn/
```

Resultado:

```text
xAI/Grok: PASS, sem bloqueadores
DeepSeek: erro por saldo/credito
Gemini/OpenRouter: erro por credito
```

O review xAI aceitou que o harness `verify_webgpu_pbpk_kernel_runtime.mjs`
move a trilha 2DGX de contrato WGSL estatico para prova executavel em GPU, sem
promover execucao local.

## Perda confirmada

`git worktree list --porcelain` marcou a worktree removida como:

```text
worktree /tmp/sounio-qual-main-20260627
branch refs/heads/fix/dissertation-gates-green-20260625
prunable gitdir file points to non-existent location
```

O indice residual continha apenas placeholders/estado parcial para parte dos
arquivos da experiencia imersiva; os arquivos uteis estavam como unstaged ou
untracked na worktree removida. Portanto, os artefatos de validacao sobrevivem,
mas a implementacao precisa ser reconstruida em worktree persistente.

## Worktree persistente de recuperacao

Foi criada uma nova worktree persistente:

```text
/workspace/sounio-qual-recovery-20260627
```

Base:

```text
fix/dissertation-gates-green-20260625
HEAD bb970ea3a
```

## Proxima acao recomendada

Reconstruir a experiencia imersiva em `/workspace/sounio-qual-recovery-20260627`
em fases menores e commitaveis:

1. Runtime minimo e contratos de verificacao.
2. Payload clinico redigido e firewall C(t).
3. Workbench/digitizacao sob revisao.
4. Contrato WGSL e harness `WEBGPU_PBPK_KERNEL_RUNTIME_PASS`.
5. Bundle local e hard WebGPU runbook.
6. Offload e commit de cada fase.

Evitar novas worktrees em `/tmp` para esta linha.

# stdlib/hetero — framework de computação heterogênea (núcleo)

Design: [`docs/design/heterogeneous-gpu-framework.md`](../../docs/design/heterogeneous-gpu-framework.md)

## O que é

O núcleo device-agnóstico do framework: capacidades **medidas** de devices
(`types.sio`), o pool de devices da máquina do usuário (`caps.sio`) e o
planner que decide repartir, colocar ou degenerar para o caminho clássico
(`planner.sio`). Nenhum código acima deste módulo conhece vendor de GPU.

## Modos do planner

| Modo | Quando | Comportamento |
|---|---|---|
| `plan_identity` | 1 device | sem split, sem overhead |
| `plan_classic` | throughputs medidos iguais (±5%) | shards iguais, coletivos clássicos |
| `plan_hetero_split` | throughputs desiguais | shards proporcionais ao medido, com teto de VRAM e redistribuição |

## Exemplo executável

`examples/hetero_planner_demo.sio` — pools homogêneo, heterogêneo, single-device
e com teto de VRAM, com `//@ expect-stdout: ALL PASS`.

## Execução (Fase 2)

| Arquivo | O que faz |
|---|---|
| `probe_cuda.sio` | Enumera GPUs NVIDIA medindo (nome, SM, VRAM) via dlopen/dlsym |
| `cuda_exec.sio` | Contexto, memória, transferências, load de PTX, launch, eventos |
| `cublas_bench.sio` | Oráculo bootstrap opcional (só onde houver libcublas; no Darwin não há) |
| `kernels/gemm_naive_f32.ptx` | Kernel GEMM f32 interino, lançado pelo substrato |
| `kernels/gemm_tiled_f32.ptx` | GEMM f32 com tiles 16x16 em shared memory (2.394 GFLOP/s medidos na L4) |
| `kernels/vec_add.kretikos.ptx` | vec_add **emitido pelo compilador** (Kretikos), executado e verificado na L4 |
| `hip_probe.sio` | Enumeração AMD via HIP/dlopen (execução aguarda R9700 livre) |

Exemplos: `examples/hetero_probe_demo.sio`, `examples/hetero_gemm_demo.sio`,
`examples/hetero_kernel_demo.sio` (verificação vs CPU + GFLOP/s medidos).
Resultados e limitações: seção 8 do design doc.

## Estado

Fases 0–2 entregues: tipos/pool/planner puros, probe medido, kernel GEMM
executando em GPU real (L4). Pendente: lowering de load/store do emissor PTX
público, caminho AMD (HIP) e GB10/aarch64.

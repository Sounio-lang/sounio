# stdlib/tensor2 — Tensor v2

Design: [`docs/design/heterogeneous-gpu-framework.md`](../../docs/design/heterogeneous-gpu-framework.md), Fase 1.

## O que muda em relação ao v1 (`stdlib/tensor`)

| | v1 | v2 |
|---|---|---|
| dtype | f64 apenas | f32/f16/bf16/f64/i32/i64/u8, bytes reais (IEEE-754, `convert.sio`) |
| device | — | código de device (CPU=referência; GPU nas Fases 2+) |
| memória | free manual, dono único | storage compartilhado com **refcount**; views/clones liberados independentemente |
| shape | rank ≤ 8 | rank ≤ 8, strides |

## Semântica de referência

`ops.sio` é o backend CPU de referência: lê pelo dtype, computa em f64,
escreve pelo dtype. Backends GPU devem bater com ele dentro da tolerância
declarada por dtype. Arredondamento bf16/f16/f32 é round-to-nearest-even,
definido em `convert.sio` — um valor guardado como bf16 perde exatamente os
bits que perderia numa GPU.

## Exemplo executável

`examples/tensor2_demo.sio` — round-trips por dtype, refcount, matmul
f32/bf16, softmax, RMSNorm. `//@ expect-stdout: ALL PASS`.

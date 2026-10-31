<!-- docs:meta
topic_id: repo.docs.guide.llm-quickstart
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.llm-quickstart
-->

# Sounio LLM quickstart

Linux x86-64. The compiler entry point is `bin/souc` in this repository.
It does not run on macOS. A build that does not include that ELF is not
a working install; this page does not claim one.

From the repository root:

```bash
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
./bin/souc --version
./bin/souc check examples/tiny_train_demo.sio
./bin/souc compile examples/tiny_train_demo.sio -o /tmp/tiny_train
/tmp/tiny_train
```

The last line of a good run is `TRAIN PASS`.

`scripts/llm_smoke.sh` checks the LLM and training examples and runs
that demo. It is the reproduction step for the numbers below.

## Serve

CPU, model directory with `config.json`, `model.safetensors` and
`tokenizer.json`:

```bash
./bin/souc compile examples/llm_serve.sio -o /tmp/llm_serve
/tmp/llm_serve /path/to/SmolLM2-135M 8090 smollm2-135m
```

`GET /health`, `GET /v1/models`, `POST /v1/completions`.
`temperature` 0 or absent is greedy. A positive temperature uses the
seeded sampler (`seed`, `top_p`). `"stream": true` is Server-Sent Events.

GPU decode (m=1 projections) needs the three PTX files and `libcuda`:

```bash
/tmp/llm_serve /path/to/SmolLM2-135M 8091 smollm2-135m \
  stdlib/hetero/kernels/gemm_tiled_f32.ptx \
  stdlib/hetero/kernels/cvt_f64_f32.ptx \
  stdlib/hetero/kernels/gemv_f32.ptx
```

The process prints `gpu=1` when the runtime loaded. Rows that are not
1 and not a multiple of 16 stay on the CPU.

## What was run

Workspace CPU, 2026-10-24: SmolLM2-135M `llama_load` 37.706 s after the
4-wide bf16 unpack (44.218 s before). `tiny_train_demo` printed
`TRAIN PASS`.

L4 on `gpuorangefs-r770-proxmox`: decode 0.496 s/token after weight
upload (job 17039); `llm serve` with `gpu=1` returned
` Paris. It is` for `max_tokens` 4 (job 17080, 8.748 s including prefill).

Design record: `docs/design/heterogeneous-gpu-framework.md`.

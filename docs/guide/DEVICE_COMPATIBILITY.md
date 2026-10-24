# Device compatibility

Only rows with a measurement from this repository. An empty cell is
not a claim.

| Device | Where | What ran | Result |
|---|---|---|---|
| CPU, workspace VM | 2026-10-24 | SmolLM2-135M `llama_load` | 37.706 s (4-wide bf16); 44.218 s before |
| CPU, workspace VM | 2026-10-24 | `tiny_train_demo` | `TRAIN PASS` |
| CPU, r770 (same node as the L4) | job 17039 | SmolLM2 greedy decode, 8 steps | 1.762 s/token |
| NVIDIA L4 | job 17039, `gpuorangefs-r770-proxmox` | resident GEMV decode, 7 steps after upload | 0.496 s/token; probe max abs diff 3e-6; tokens matched the CPU |
| NVIDIA L4 | job 17080 | `llm serve` with the three PTX paths | log `gpu=1`; text ` Paris. It is`; request 8.748 s including prefill |
| NVIDIA GB10 | not remeasured here | Madaros does not emit aarch64 ELF | recorded no-go, `CLAUDE.md` §13 |
| AMD R9700 | not measured | GPU stayed on a serving job | no number |

dtype on the measured path: f64 masters, f32 on the device for the
GEMV. bf16 in the file is converted at load.

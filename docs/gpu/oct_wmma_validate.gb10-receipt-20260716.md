<!-- docs:meta
topic_id: repo.docs.gpu.oct-wmma-validate.gb10-receipt-20260716
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.oct-wmma-validate.gb10-receipt-20260716
-->

# GB10 WMMA Receipt

Status: manual hardware witness passed on 2026-07-16.

This receipt records a manual execution of `docs/gpu/oct_wmma_validate.cu` from
commit `8b5e07050c890e39c4e71424f16f58cd7fbe4170`. It is not a replacement for a
CI GPU job and does not claim general GPU backend correctness.

## Source

- File: `docs/gpu/oct_wmma_validate.cu`
- SHA-256: `5bd22146ac869c92854b0821418d742e43141c1a3b4a137583325311b21e913f`
- Compile command: `/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O2 -arch=sm_121 oct_wmma_validate.cu -o oct_wmma_validate`

## Environment

- Host: `spark-8e54`
- Architecture: `aarch64`
- GPU: `NVIDIA GB10`
- Driver: `580.159.03`
- Compute capability: `12.1`
- CUDA compiler: `Build cuda_13.0.r13.0/compiler.36424714_0`

## Runtime Output

```text
e1*e2 on tensor core: comp3=1.00 comp4=0.00  (X: comp3=+1,comp4=0)
batch: 0/128 comps mismatch, maxerr=0.000 (f16 tile precision)
PASS: WMMA octonion multiply is Convention X on GB10
```

## Boundary

The repository GPU Performance Validation workflow still skipped its
`validate-on-gpu` job at the time of this witness. This receipt proves the
specified CUDA source ran on the named GB10 environment; it does not prove that
the GitHub workflow automatically runs or retains the same witness.

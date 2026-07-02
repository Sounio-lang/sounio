<!-- docs:meta
topic_id: repo.docs.ops.dgx-spark-gpu-dev
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.dgx-spark-gpu-dev
-->

# DGX Spark GPU Development Point

There are two DGX Spark (GB10 Grace-Blackwell, `sm_121`, CUDA 13.0) hosts,
with distinct roles — do not conflate them:

- **Canonical / CI** — `demetrios@192.168.3.43`. Must stay green. Runs the
  public GPU gate below plus any hardware-verification gate that is expected
  to pass on every relevant change (e.g.
  `scripts/dev/dgx_spark_epistemic_wmma_matmul_gate.sh` with
  `DGX_SPARK_ROLE=canonical`, the default).
- **Experimental / dev** — `demetrios@192.168.3.24`. Confirmed reachable
  (ping + SSH) 2026-07-02. Used for benchmarking, prototyping new kernels
  (SPIR-V/Metal real-hardware attempts, new K-AXI patterns) before they are
  gate-worthy, and anything allowed to fail. Select it by setting
  `DGX_SPARK_ROLE=experimental` (or `DGX_SPARK_HOST=192.168.3.24` directly)
  when invoking a gate script.

Both hosts authenticate via SSH key/ControlMaster — **no password is stored
in this repo, in any script, or in any log for either host.** Spark #2 is
currently reachable but not yet key-authenticated for automation; migrate it
to key-based auth (matching Spark #1) before wiring it into unattended CI.

The local Sounio checkout remains the compiler/source authority. The Spark
hosts are the CUDA toolchain and runtime authority for this lane.

## Public GPU Gate

Run the local structural gate plus remote `ptxas` and CUDA Driver API launch:

```bash
DGX_SPARK_SSH_CONTROL_PATH=/tmp/sounio-dgx-spark-ctl \
  bash scripts/dev/dgx_spark_public_gpu_gate.sh
```

The gate writes:

```text
artifacts/gpu/dgx_spark_public_gpu_gate.v1.json
```

It validates:

- `tests/run-pass/gpu_vec_add_e2e.sio`
- `tests/run-pass/gpu_launch_vec_slices.sio`

The default remote toolchain is CUDA 13.0:

- `DGX_SPARK_PTXAS=/usr/local/cuda-13.0/bin/ptxas`
- `DGX_SPARK_NVCC=/usr/local/cuda-13.0/bin/nvcc`
- `DGX_SPARK_ARCH=sm_121`

Override those variables if the Spark image changes.

## Epistemic WMMA Matmul Gate

Validates the compiler-generated epistemic tensor-core kernel
(`self-hosted/gpu/kernel_ir.sio`, `gpu_build_epistemic_wmma_matmul_16x16_ir`)
end-to-end: local PTX emission → remote `ptxas` **native** `sm_121` CUBIN (no
JIT) → CUDA Driver API launch → compare against the CPU GUM/RSS oracle.

```bash
DGX_SPARK_SSH_CONTROL_PATH=/tmp/sounio-dgx-spark-ctl \
  bash scripts/dev/dgx_spark_epistemic_wmma_matmul_gate.sh
# or, against the experimental Spark:
DGX_SPARK_ROLE=experimental DGX_SPARK_SSH_CONTROL_PATH=/tmp/sounio-dgx-spark-ctl \
  bash scripts/dev/dgx_spark_epistemic_wmma_matmul_gate.sh
```

The gate writes `artifacts/gpu/dgx_spark_epistemic_wmma_matmul_gate.v1.json`.
It is distinct from three other, easily-confused GPU receipts in this repo —
do not conflate them:

- The **public GPU gate** above (narrow `souc --backend gpu` surface,
  loop+scalar f64 only).
- The **PR #487 Blackwell receipt**
  (`docs/research/solver-gpu-native-path-2026-06-27.md`), which loaded the
  **hand-written** `self-hosted/gpu/epistemic_mma_reference.ptx` — a
  different formula, not compiler output.
- The **"13 L4-validated profiles"** K-AXI/`nvidia_bare.sio` SM80
  hand-assembled SASS path (`docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md`)
  — unrelated code path, unrelated target arch.

As of 2026-07-02 this gate has not yet been run against real hardware from
this workspace (no CUDA toolchain / DGX Spark reachable from this sandbox);
it is ready to run but unexecuted. Local `souc build` of the PTX-emitter
driver (`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio`) currently hits a
known pre-existing Madaros native-codegen limitation on large merged
self-hosted GPU modules, unrelated to the kernel math — `souc check`
(typecheck-only) passes clean. Building the driver on a host where `souc
build` doesn't hit that limitation is a prerequisite for actually running
this gate.

## SSH Session

Neither gate stores or manages passwords. Use an SSH key or create a
ControlMaster socket before running either:

```bash
ssh -M -S /tmp/sounio-dgx-spark-ctl \
  -o ControlPersist=20m \
  -o StrictHostKeyChecking=accept-new \
  -fnN demetrios@192.168.3.43   # or 192.168.3.24 for the experimental host
```

Then pass the same socket path through `DGX_SPARK_SSH_CONTROL_PATH`.

## Evidence Boundary

The public GPU gate proves selected public GPU f64 PTX artifacts assemble and
that `gpu_vec_add_e2e` launches correctly on the Spark CUDA Driver API path.
The epistemic WMMA matmul gate proves the *compiler-generated* epistemic
tensor-core kernel's corrected GUM/RSS uncertainty math on real Blackwell
silicon, once run. Neither gate claims general GPU backend correctness.

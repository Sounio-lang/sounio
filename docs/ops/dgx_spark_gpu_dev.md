<!-- docs:meta
topic_id: repo.docs.ops.dgx-spark-gpu-dev
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.dgx-spark-gpu-dev
-->

# DGX Spark GPU Development Point

The DGX Spark at `demetrios@192.168.3.43` is the preferred direct GPU
development point for public GPU PTX validation when a local checkout cannot
see CUDA hardware.

The local Sounio checkout remains the compiler/source authority. The Spark host
is the CUDA toolchain and runtime authority for this lane.

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

## SSH Session

The gate does not store or manage passwords. Use an SSH key or create a
ControlMaster socket before running it:

```bash
ssh -M -S /tmp/sounio-dgx-spark-ctl \
  -o ControlPersist=20m \
  -o StrictHostKeyChecking=accept-new \
  -fnN demetrios@192.168.3.43
```

Then pass the same socket path through `DGX_SPARK_SSH_CONTROL_PATH`.

## Evidence Boundary

This gate proves selected public GPU f64 PTX artifacts assemble and that
`gpu_vec_add_e2e` launches correctly on the Spark CUDA Driver API path. It does
not claim general GPU backend correctness.

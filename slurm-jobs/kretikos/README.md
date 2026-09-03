# Kretikos Slurm Runtime Bridge

Kretikos uses an HPC split-brain model:

- the local/control workspace emits and inspects GPU artifacts
- the Slurm GPU worker performs CUDA Driver API runtime admission and launch
- the job comment carries the runtime verdict back to the control side
- the worker preserves a compact `publish/` directory on worker-local storage
  so the control side can fetch durable evidence through the worker pod

This is intentional. A development container may have `nvcc`, `ptxas`,
`nvdisasm`, `cuobjdump`, and `libcuda` installed while still lacking a mounted
NVIDIA device or live kernel driver. In that case local runtime should classify
as `not_run`, and the L4 Slurm worker is the runtime authority.

## Control-Side Probe

```bash
./bin/kretikos detect-cuda
./bin/kretikos run-manifest examples/kretikos/manifest.tsv --validate-runtime
```

Expected local shape on non-GPU containers:

- CUDA tools may be `FOUND`
- `nvidia-smi` may report `driver/device unavailable`
- runtime validation may report `not_run/cuInit_failed`

That is not a Kretikos failure. It means the current host is a build/inspection
host, not the CUDA execution host.

## Slurm Runtime Probe

Single source:

```bash
WAIT_TIMEOUT_SECONDS=420 \
  ./bin/kretikos hpc source \
  examples/kretikos/real_vec_add_f64.sio
```

Full Kretikos source manifest:

```bash
WAIT_TIMEOUT_SECONDS=420 \
  ./bin/kretikos hpc manifest \
  examples/kretikos/manifest.tsv
```

CI-style acceptance gate:

```bash
WAIT_TIMEOUT_SECONDS=420 \
  bash scripts/ci/kretikos_hpc_slurm_runtime_gate.sh
```

The lower-level submitters remain available at:

- `slurm-jobs/kretikos/submit-kretikos-source.sh`
- `slurm-jobs/kretikos/submit-kretikos-manifest.sh`

Useful environment overrides:

- `NS`: Kubernetes namespace for the Slurm control plane, default
  `slurm-pilot`
- `LOGIN_POD_NAME`: explicit Slurm login pod
- `SBATCH_NODELIST`: GPU node, default `gpuorangefs-r770-proxmox`
- `SBATCH_PARTITION`: Slurm GPU partition, default `gpu-orangefs`
- `SBATCH_ACCOUNT`: Slurm account, default `plruntime`
- `SBATCH_QOS`: Slurm QoS, default `gpuorangefs`
- `KRETIKOS_VEC_N`: runtime vector length, default `64`
- `KRETIKOS_HPC_PUBLISH_RESULTS`: preserve a worker-local artifact directory,
  default `1`
- `KRETIKOS_HPC_FETCH_RESULTS`: fetch published artifacts back locally, default
  `1`
- `KRETIKOS_HPC_CERTIFY_KAXI`: emit runtime-backed K-AXI certificates for
  supported profiles, default `1`
- `KRETIKOS_HPC_LOCAL_ARTIFACT_DIR`: local result directory override for a
  single-source run
- `KRETIKOS_HPC_WORKER_NODE`: Kubernetes node hosting the Slurm worker, default
  is `SBATCH_NODELIST` with the `gpuorangefs-` prefix stripped
- `KRETIKOS_HPC_WORKER_POD_LABEL`: worker pod selector, default
  `app.kubernetes.io/instance=slurm-pilot-worker-gpuorangefs`
- `KRETIKOS_HPC_WORKER_TMP`: worker-local tmp root, default `/tmp`

Fetched source-run artifacts include:

- `kretikos_hpc_source_result.v1.json`
- `kretikos-source.log`
- `kretikos-kaxi-certificate.log`
- `comment.txt`
- `bundle/kretikos_bundle.v1.json`
- `bundle/kretikos_source_profile.v1.json`
- `certificate/kaxi_certificate.v1.json` for K-AXI-supported profiles
- emitted PTX/CUBIN artifacts and validation logs for that profile

## Acceptance Boundary

The Slurm worker acceptance boundary is runtime execution:

- `cuInit`
- `cuDeviceGetCount`
- `cuDeviceGetName`
- `cuDeviceComputeCapability`
- `cuModuleLoadData`
- `cuModuleGetFunction`
- `cuLaunchKernel`
- `cuMemcpyDtoH`
- CPU-oracle parity for the selected Kretikos runtime rung

Worker-local `ptxas` or `nvdisasm` is useful but not required for this runtime
boundary. Toolchain inspection can happen on the control side; the worker only
needs the NVIDIA driver/runtime surface required to load and launch the
Sounio-owned CUBIN.

If worker-side disassembly or PTX assembly is required later, use the site's
CUDA module, a CUDA-enabled Apptainer/Singularity image, or an administrator
toolkit install on the compute image. Do not convert a missing worker-side
`ptxas` into a runtime failure when `cuLaunchKernel` and copy-back validation
pass.

## Latest Checked Witness

Validated on 2026-05-07 UTC with:

```bash
WAIT_TIMEOUT_SECONDS=700 \
  bash scripts/ci/kretikos_hpc_slurm_runtime_gate.sh
```

Result:

- `kretikos_manifest_result total=13 failed=0`
- jobs `1305` through `1317`
- node `gpuorangefs-r770-proxmox`
- device `NVIDIA_L4`
- compute capability `8.9`
- CUDA driver version `13020`
- final rung `epistemic_dual_output_f32`
- final reason `runtime_epistemic_dual_output_f32_pass`
- each source job preserved a worker-local `publish/` directory and fetched a
  `kretikos_hpc_source_result.v1.json` artifact back to the control workspace
- K-AXI-supported profiles now include `vec_add_f32`, `vec_sub_f32`,
  `vec_mul_f32`, `vec_div_f32`, `fma_f32`, `epistemic_elementwise_f32`, and
  `epistemic_dual_output_f32`
- each K-AXI-supported profile publishes a
  `certificate/kaxi_certificate.v1.json` whose embedded runtime validation must
  pass on the worker

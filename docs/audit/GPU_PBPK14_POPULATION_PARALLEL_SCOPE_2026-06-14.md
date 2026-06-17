<!-- docs:meta
topic_id: repo.docs.audit.gpu-pbpk14-population-parallel
authority: repo_only
audience: contributors
last_validated: 2026-06-14
validated_by: gpu-pbpk-slice
-->

# GPU PBPK14 Tsit5 — Population-Parallel Scope (2026-06-14)

## Truth-table statement (read first)

This document does **NOT** claim "PBPK14 Tsit5 compiles to one GPU kernel." That
is not done. What is verified here is a strictly smaller, honest first slice:

- A minimal **3-compartment linear PBPK derivative** `dC = A*C` is expressed as a
  `kernel fn`, **passes** the checked GPU artifact's `check`, and **emits PTX**
  via `build --backend gpu`. File: `examples/kretikos/pbpk_rhs_kernel.sio`.

Everything about scaling to 14 compartments, a full Tsit5 (Tsitouras 5(4))
adaptive step, and actual per-thread population indexing is **remaining work**,
itemized below. Component constructs that Tsit5 needs (fixed-size f64 stage
arrays, `while` loops, f64 arithmetic) are each individually accepted by the
checked surface, but a Tsit5 step has **not** been compiled — do not read
component acceptance as a whole-step claim.

## What population-parallel PBPK means

The natural GPU form of PBPK is **population-parallel**: one virtual patient per
GPU thread. Each thread holds its own parameter set (clearances, volumes,
partition coefficients) and its own state vector `C` (one concentration per
compartment), and integrates the ODE `dC/dt = f(C, theta)` over time with the
same integrator. There is no cross-thread coupling in the basic scheme, so it is
embarrassingly parallel: `M` patients map to `M` threads, blocked normally.

The kernel-legal compute core of that scheme is the **right-hand side (RHS)**
evaluation. For a linear PBPK model the RHS is a matrix-vector product:

```
dC[i] = sum_j A[i*n + j] * C[j]
```

where `A` (row-major, `n*n`) is the rate matrix built from the patient's
flows/volumes. That core is what this slice delivers and verifies.

## Verified surface (this slice)

Authority = the checked GPU artifact named in `docs/compiler/GPU_KERNELS.md`.

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_GPU_BIN" check examples/kretikos/pbpk_rhs_kernel.sio
# -> All checks passed: examples/kretikos/pbpk_rhs_kernel.sio   (RC=0)

"$SOUC_GPU_BIN" build examples/kretikos/pbpk_rhs_kernel.sio --backend gpu -o /tmp/pbpk_rhs.ptx
# -> Wrote PTX to /tmp/pbpk_rhs.ptx   (RC=0)
#    PTX contains two kernels: .visible .entry pbpk_rhs_3comp / pbpk_rhs_linear
```

Note: `./bin/souc check examples/kretikos/pbpk_rhs_kernel.sio` (the lean_single
**bootstrap** `mini_native`) does **not** parse the GPU kernel surface — it
prints telemetry and `error: no main`. The bootstrap is not the GPU authority;
`$SOUC_GPU_BIN` is. Use `$SOUC_GPU_BIN check` for kernel acceptance.

### Kernel-legal construct set (empirically confirmed, 2026-06-14)

Probed against `$SOUC_GPU_BIN check`. Each row is a real probe result.

| Construct | Accepted? | Required effects |
|---|---|---|
| `kernel fn ... with GPU` empty / pure f64 arithmetic + `let` | yes | `GPU` |
| `while` loop with mutable accumulator | yes | `GPU, Div, Mut` |
| fixed-size f64 array, indexed read/write | yes | `GPU, Panic, Mut` |
| array + loop (matrix-vector `dC = A*C`) | yes | `GPU, Panic, Div, Mut` |
| `kernel fn ... -> f64` (non-void return) | **NO** | — (`Expected {, found ->`) |
| `gpu_thread_id_x()` / `gpu_block_id_x()` / `gpu_block_dim_x()` / `gpu_sync_threads()` | **NO** | — (see below) |

The effect requirements are not incidental: arrays pull in `Panic` (bounds
check), mutation pulls in `Mut`, and integer/float division or the `while`
condition pulls in `Div`. The delivered kernel declares
`with GPU, Panic, Div, Mut` for exactly these reasons.

## The central blocker: per-thread indexing is outside the checked surface

Population-parallel REQUIRES mapping `gpu_thread_id_x()` → patient index, then
loading that patient's `A`/`C` from global slices. Under `$SOUC_GPU_BIN check`,
those intrinsics are **rejected**:

```
$ "$SOUC_GPU_BIN" check /tmp/k_tid.sio
Error:   × Resolution errors:
  │ Undefined variable: gpu_thread_id_x
  │ Undefined variable: gpu_block_id_x
  │ Undefined variable: gpu_block_dim_x
  │ Undefined variable: gpu_sync_threads
RC=1
```

This matches `docs/compiler/GPU_KERNELS.md`, which lists `gpu.thread_id.*`,
`gpu.block_id.*`, `gpu.block_dim.*`, and `gpu.alloc<T>` as **not** in the checked
public contract. (The `examples/kretikos/real_vec_*.sio` witnesses use these
intrinsics and therefore also fail plain `check` with the same error — they are
mapped by the Kretikos profile path, not by the checked resolver.)

Because per-thread indexing is the whole point of population-parallel, the
delivered kernel deliberately keeps state **local** (each thread builds the same
`A`/`C` locally) so it stays inside the verified surface. That is the honest
limit of this slice: it proves the RHS math is kernel-legal; it does **not** yet
fan one-patient-per-thread.

## Remaining work to reach 14-compartment Tsit5 population-parallel

In dependency order:

1. **Per-thread patient indexing (the unblocker).** Get
   `gpu_thread_id_x/block_id_x/block_dim_x` into the checked resolver scope (or
   provide a checked equivalent), so a kernel can compute
   `pid = bid*bdim + tid` and gate on `pid < num_patients`. Until this lands,
   no genuinely population-parallel kernel can pass `check`. This is implementation
   work in the GPU frontend/resolver, not source work in this file.

2. **Global per-patient I/O.** A way to pass and index device buffers:
   `theta[pid]`, `C_in[pid]`, `C_out[pid]`. The checked surface currently takes
   scalar/`i64` params; matrix `gpu.alloc<T>`/global-pointer params are also
   listed as not-yet-public in `GPU_KERNELS.md`. Needed so each thread reads its
   own patient rather than rebuilding a shared local `A`.

3. **Scale `n=3 -> n=14`.** The delivered `pbpk_rhs_linear` loop already
   generalizes to any `n` (it is `dC[i] = sum_j A[i*n+j]*C[j]`). For 14
   compartments, `A` is 14x14 = 196 entries and `C`/`dC` are length 14. Confirm
   the checked surface accepts arrays of that size (probe; fixed-size arrays are
   accepted at size 3/9 today). Nonlinear PBPK (saturable clearance,
   Michaelis-Menten) adds per-term division — already covered by the `Div`
   effect — but each nonlinear term must be re-probed for acceptance.

4. **Tsit5 step inside the kernel.** Tsitouras 5(4) is a 7-stage explicit RK with
   an embedded 4th-order error estimate. As source it is: fixed-size Butcher
   tableau constants (`c`, `a[i][j]`, `b`, `bhat`), 7 RHS evaluations into stage
   arrays `k1..k7`, a weighted sum for the 5th-order update, and an error-norm
   for adaptive step control. The constructs this needs — fixed-size f64 stage
   arrays, indexed read/write, `while`/`for` over stages, f64 arithmetic, and a
   division for the step-size controller — are **each individually accepted**
   today (see the construct table). What is **not** done: compiling an actual
   Tsit5 step end-to-end as one kernel, and the adaptive controller's branching
   (`if err < 1 { accept } else { shrink }`) at full step-loop depth. Treat this
   as the largest remaining unit; build it incrementally (fixed-step RK4 first,
   then embedded error, then adaptivity) re-running `check` at each rung.

5. **Numerics / determinism.** Per-thread divergent step counts (adaptive Tsit5
   gives each patient a different trajectory length) cause warp divergence; a
   fixed-step or capped-iteration variant may be needed for the first runnable
   version. This is a performance/correctness design choice, recorded here so it
   is not discovered late.

## SLURM submit recipe (cp-based, modeled on selfhost-lower-oom/submit.sh)

The PTX/`check` verification above is **local** (no GPU needed; PTX is text). To
get a CUBIN + a real population-parallel run you need a GPU node. Below is the
`kubectl cp`-based recipe modeled on `slurm-jobs/selfhost-lower-oom/submit.sh`,
retargeted from `cpu-ops` to a GPU partition (`gpuorangefs-r770`). This is a
RECIPE, not a submitted job (the task is bounded and does not submit).

```bash
#!/usr/bin/env bash
# slurm-jobs/gpu-pbpk-rhs/submit.sh  (recipe — adapt partition/account to the cluster)
set -euo pipefail

WORKTREE="${WORKTREE:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpuorangefs-r770}"   # GPU partition (was cpu-ops in the model)
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-gpu}"
GRES="${GRES:-gpu:1}"
JOB_MEM="${JOB_MEM:-16G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:20:00}"
RUN_ID="${RUN_ID:-pbpk-rhs-$(date -u +%Y%m%dT%H%M%S)}"

# 1) Find login + GPU worker pods (mirror selfhost-lower-oom).
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' \
  | grep -iE 'worker.*(r770|gpu)' | head -1)"
WORKER_CTR="${WORKER_CTR:-slurmd}"

# 2) Package payload: the GPU bin + stdlib + the kernel source (small).
TARBALL="/tmp/${RUN_ID}.tgz"
tar -C "${WORKTREE}" -czf "${TARBALL}" \
  artifacts/omega/souc-bin/souc-linux-x86_64-gpu \
  stdlib \
  examples/kretikos/pbpk_rhs_kernel.sio
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${WORKER_POD}:/tmp/${RUN_ID}.tgz" -c "${WORKER_CTR}"

# 3) sbatch script (kept small; payload already staged on the worker).
SBATCH="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --gres=${GRES}
#SBATCH -N 1 -n 1 -c ${JOB_CPUS}
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"; RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${ROOT}" "\${RES}"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${ROOT}"; cd "\${ROOT}"
chmod +x artifacts/omega/souc-bin/souc-linux-x86_64-gpu
export SOUC_GPU_BIN="\${ROOT}/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
export SOUNIO_STDLIB_PATH="\${ROOT}/stdlib"
nvidia-smi -L | tee "\${RES}/gpu.txt" || true

# a) re-verify check on the GPU node
"\${SOUC_GPU_BIN}" check examples/kretikos/pbpk_rhs_kernel.sio 2>&1 | tee "\${RES}/check.log"
# b) emit PTX
"\${SOUC_GPU_BIN}" build examples/kretikos/pbpk_rhs_kernel.sio --backend gpu \
  -o "\${RES}/pbpk_rhs.ptx" 2>&1 | tee "\${RES}/build.log"
# c) PTX -> CUBIN for the node's arch (real GPU compile)
SM="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
ptxas -arch=sm_\${SM:-80} "\${RES}/pbpk_rhs.ptx" -o "\${RES}/pbpk_rhs.cubin" 2>&1 \
  | tee "\${RES}/ptxas.log" || echo "ptxas_rc=\$?" | tee -a "\${RES}/ptxas.log"
ls -l "\${RES}" | tee "\${RES}/SUMMARY.txt"
EOF

# 4) Submit via the login pod (mirror selfhost-lower-oom).
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${SBATCH}"
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${SBATCH}'"
echo "RUN_ID=${RUN_ID}"
```

Fetch results (mirror `selfhost-lower-oom/fetch.sh`): `kubectl cp` the
`/tmp/${RUN_ID}.result` dir back from the worker; the witness is a non-empty
`pbpk_rhs.cubin` plus a clean `ptxas.log`. A genuine population-parallel run
(host launcher allocating `M` patient buffers, `GPU.launch` over `M` threads,
double-precision oracle compare) is downstream of remaining-work items 1-2 and
is out of scope for this slice.

## Files

- `examples/kretikos/pbpk_rhs_kernel.sio` — the verified kernel (this slice).
- `docs/compiler/GPU_KERNELS.md` — the checked public kernel contract.
- `slurm-jobs/selfhost-lower-oom/submit.sh` — the cp/sbatch model used above.

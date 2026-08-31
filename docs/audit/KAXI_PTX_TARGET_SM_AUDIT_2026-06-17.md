<!-- docs:meta
topic_id: repo.docs.audit.kaxi-ptx-target-sm-audit-2026-06-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.kaxi-ptx-target-sm-audit-2026-06-17
-->

# K-AXI → PTX emitter correctness audit — 2026-06-17

> The `ACCEPT=318/318` on this page is Slurm job 4311 (2026-06-17), a JIT-accept receipt, not the byte-compare golden gate. The golden gate on 2026-08-18 measured **0/318** PASS, rc=1, 80 s (#1915). Do not read this dated audit as current 318/318.

Static, local audit of every `(pattern × mode)` PTX emitted by
`./bin/kretikos kaxi-emit-ptx … --no-ptxas` (318 combos: 53 patterns × 6 modes).
No GPU / ptxas required for these checks.

## TL;DR

The emitter is **clean on all three local static checks**. An initial
"f32-atomic under sm_50" finding was a **false positive** (see Correction) —
`atom.add.f32` is valid since sm_20. No locally-provable emitter bug was found.
Full validation (ptxas acceptance + runtime correctness) still requires a
`gpu-orangefs` run; those classes cannot be checked without the CUDA toolkit.

## Method

- **Register-range:** for each kernel, every referenced `%rd/%fd/%r/%f/%p`
  index must be `<` the bank's declared `<N>` count.
- **Type-width:** value ops (`setp/add/sub/mul/mad/div/fma/min/max/and/or/xor/
  shl/shr/neg/abs/sqrt/rcp/rsqrt/lg2/ex2/sin/cos/selp`) — the type suffix bit
  width must match the operand register bank width (`%rd`=64i, `%r`=32i,
  `%fd`=f64, `%f`=f32). `cvt/ld/st/mov` excluded (legitimately mix widths).
- **Target/feature gating:** instructions gated to a minimum SM must not appear
  under a lower `.target`.

## Results

| Check | Result |
|-------|--------|
| Register-range violations | **0 / 318** |
| Type-width mismatches | **0 / 318** |
| Target/feature-gating risks | **0 / 318** (after correction) |

Global max referenced index per bank: `rd=261, f=257, fd=5, r=2, p=1`.
All 318 kernels declare `.target sm_50`.

Note: the `%p<8>` → `%p<64>` predicate-bank widening in the in-flight edits is
**benign** — the max predicate index actually used is `%p1`, so `%p<8>` was
already sufficient. It is not a fix for any range problem.

## Correction (the false positive)

The first pass flagged 3 combos (`atomic_sum_f32` in `f32`/`f32_2c`/`f32_gum`)
for emitting `atom.global.add.f32` under `.target sm_50`. **This was wrong.**
The audit script hardcoded a minimum of `sm_60` for `atom.*.f32`, but `sm_60`
is the threshold for the **double** variant `atom.add.f64`. Single-precision
`atom.add.f32` is valid since **sm_20** (confirmed against the PTX ISA atom
spec). So `atom.global.add.f32` under `.target sm_50` is **valid PTX**, the
May-28 golden is correct, and there is no bug here.

Lesson: do not publish a "ptxas would reject X" claim from a hand-built SM
threshold table. Verify the threshold against the ISA, or run ptxas.

## Real-hardware acceptance — RESOLVED (Slurm gpu-orangefs)

The local checks cannot confirm full assembler acceptance (memory-operand
types, address spaces, special-register setup, control-flow) or runtime. The
`gpu-orangefs` nodes turned out to be **driver-only** (NVIDIA L4, cc 8.9,
driver 595.71.05; `libcuda` present, **no `ptxas`/nvcc/nvrtc toolkit**), so the
acceptance test was done through the **CUDA driver's internal JIT**
(`cuModuleLoadDataEx`) via the dlopen-based `scripts/gpu/kaxi_ptx_runner.c`
(compiled locally `gcc -O2 … -ldl -lm`; types are self-defined, no toolkit
needed).

**Result (Slurm job 4304, gpuorangefs-r770-proxmox): `ACCEPT=318  REJECT=0`.**
Every emitted kernel JIT-compiled on a real L4. `atom.global.add.f32` under
`.target sm_50` both JIT-compiled *and* executed (launch rc=0), independently
confirming the false-positive correction above.

Job harness: `slurm-jobs/kaxi-ptxas-accept/{submit_jit.sh,run_jit.sh}` (the
original ptxas-based `submit.sh`/`run_ptxas.sh` is retained but is a no-op on
these toolkit-less nodes — `PTXAS_NOT_FOUND`).

Scope: `ACCEPT=318` means every kernel is **well-formed and JIT-compilable** by
the driver — it does **not** establish that the kernels *compute correctly*.

Note: of the 318 accepted kernels, 106 also ran to launch rc=0; the other 212
returned the runner's rc=1. rc=1 is past the JIT/load step, so it is not an
acceptance failure — but its cause was **not retained** (`run_jit.sh` deletes
the per-kernel output on ACCEPT), so whether each rc=1 is a verify-mismatch
(from the generic `--threads 1 --mem-words 64`, no per-mode init/flags) or a
real launch error is uncharacterized. Characterizing it = a runtime
differential with outputs kept + correct per-mode init — separate follow-on.

### CI wiring (done)
The byte-compare golden gate runs `--no-ptxas` and cannot catch invalid PTX, so
real-hardware acceptance is wired as a **nightly** GitHub Actions job
(`.github/workflows/kaxi-ptx-acceptance.yml`): it emits all kernels
(`slurm-jobs/kaxi-ptxas-accept/emit_ptx.sh`), builds the driver-JIT runner, and
runs `submit_jit.sh --wait` (submit → poll → fetch → assert
`KAXI_JIT_ACCEPT_OK`). It is **not** per-PR blocking (needs the on-prem SLURM
cluster) and is **opt-in** via repo variable `SOUNIO_ENABLE_KAXI_SLURM_GATE=1`,
so it stays skipped (no false red/green) until the cluster-access prereqs are
provisioned — either `SLURM_KUBECONFIG` (GitHub-hosted runner, only if the
cluster API is reachable) or a self-hosted in-cluster runner. The `--wait` path
was validated end-to-end (Slurm job 4311: ACCEPT=318/318, gate PASS).

## Runtime differential — the 212 `rc=1` characterized (RESOLVED)

The acceptance run reported `ACCEPT=318` but only 106 kernels launched `rc=0`
under the gate's generic invocation; 212 returned `rc=1`. This follow-on
determined whether those were real defects or invocation artifacts
(`slurm-jobs/kaxi-ptxas-accept/run_jit_diag.sh`, Slurm job on
gpuorangefs-r770-proxmox / L4):

| Config | Invocation | Result |
|--------|-----------|--------|
| **A** (reproduce the gate) | `--threads 1 --mem-words 64` | 106 `launch_pass`, **212 `cuLaunchKernel_rejected`** |
| **B** (per-mode buffers + ample mem) | mode flags + `--mem-words 4096` | **318 `launch_pass`, 0 fail** |

Cross-tab: **all 212 A-failures clear in B; 0 still failing.** The `rc=1` was a
launch-time **invocation/under-provisioning artifact**, not a kernel defect —
the epistemic/gum/2c modes need their second buffer (`--epistemic`/`--gum`) and
correct `--type`, which the gate's single-buffer generic launch didn't supply,
so `cuLaunchKernel` rejected the arity/params. Given each mode its required
buffers, every kernel launches cleanly. This retroactively confirms the PR's
`rc=1` note as benign.

**Compute differential (proves the path computes, not just loads).** For the
arithmetic family, inputs were fed (`--init-mem 3.0`) and the printed output
compared to a **PTX-derived** in-place self-op oracle (each kernel reads
`mem[tid]` twice): add→2x, sub→0, mul→x², div→1, fma→x²+x.

| input x=3 | add | sub | mul | div | fma |
|-----------|-----|-----|-----|-----|-----|
| expected  | 6   | 0   | 9   | 1   | 12  |
| f32 / f64 | PASS | PASS | PASS | PASS | PASS |

**10/10 PASS** (f32 + f64).

Scope honesty: `launch_pass` / config-B = "ran without a CUDA error", **not**
compute-correct. Only the 10 arithmetic kernels above are compute-checked here;
compute-correctness of the pbpk/octonion/epistemic families is **not** asserted
by this audit (their dedicated gates — lse8/sinkhorn/phase-* — cover specific
kernels; not re-verified here). The differential set was kept deliberately
small per the agreed scope.

### Follow-on
- Recapture the May-28 goldens (`scripts/ci/kaxi_ptx_capture.sh`) to absorb the
  benign `%p<8>`→`%p<64>` drift once the in-flight source edits are built in.
  *(Done in this PR.)*
- Runtime differential of the 212 `rc=1`. *(Done — see above; all artifacts.)*

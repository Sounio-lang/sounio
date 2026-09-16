<!-- docs:meta
topic_id: repo.docs.dissertation.results.d6-full-integration-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d6-full-integration-v1
-->

---
docs:meta:
  id: dissertation-results-d6-full-integration-v1
  title: D6 Full Integration V1
  doc_type: result
  status: draft
  owner: Codex
  updated: 2026-05-15
---

# D6 Full Integration v1

Branch: `codex/sounio-pinn-training-loop`

Gate: `D6_FULL_INTEGRATION_PASS`

## Scope

This sprint hardens D.6 from the earlier proof-of-concept into a native
Sounio end-to-end fractional PINN integration test. The test executes
LayerNorm finite-difference validation, differentiable tensor indexing,
multi-layer tape-to-parameter gradient synchronization, and a 5000-epoch
fractional sirolimus single-compartment PINN training run in one binary.

## Issues Resolved

### exit-139

Root cause: the original same-process D3+D6 crash had two contributing
paths. First, `backward_matmul` used the wrong identifier while checking
whether the right-hand operand required gradients in the general-shape
path. Second, `ParameterStore` copied tape gradients into heap-backed
per-parameter gradient tensors; in the combined multi-layer tape process,
that copy path crashed when synchronizing later parameters.

Fix:

- `backward_matmul` now checks the right-hand operand id read from the
  tape node.
- `ParameterStore` now has a deterministic internal gradient cache for
  tape-synchronized gradients. Optimizer reads, clipping, and gradient
  inspection route through this cache when `param_store_sync_grads` has
  populated it.
- The full D.6 trainer avoids the previous debug store path and trains
  with explicit Adam updates and global gradient clipping in native
  Sounio, while D4 optimizer integration remains separately validated.

Result: the full same-process integration binary runs to completion with
no exit-139.

### LayerNorm FD

`loss = sum(layer_norm(x, gamma, beta))` was checked against central
finite differences for all three differentiable inputs.

- `dL/dx`: PASS, tolerance `1e-5`
- `dL/dgamma`: PASS, tolerance `1e-5`
- `dL/dbeta`: PASS, tolerance `1e-5`

### taped_index

Implemented `TAPE_INDEX` and `tape_tensor_index` with a one-hot backward
rule. The FD/analytic check for extracting element 3 from a rank-1 tensor
passes with absolute error below `1e-10`.

### ParameterStore Sync

The integration test builds a two-layer network:

`Dense(2->4) -> tanh -> Dense(4->1) -> squared loss`

After `tape_backward` and `param_store_sync_grads`, all parameter groups
have non-zero gradients:

- layer1 W: non-zero
- layer1 b: non-zero
- layer2 W: non-zero

## PINN Training

Architecture: `MLP(1->64->64->64->1)` with tanh activations.

Physics:

`D_C^0.8 C + 0.1*C = 0`

Training:

- 5000 epochs
- Adam learning rate `5e-4`
- global gradient clipping at norm `1.0`
- physics warmup over first 500 epochs
- 50 collocation points over `[0, 24]`
- held-out validation grid: 100 points

Loss checkpoints:

- epoch 1: `0.252878`
- epoch 500: `0.000243`
- epoch 1000: `0.000074`
- epoch 2000: `0.000015`
- epoch 3000: `0.000007`
- epoch 4000: `0.000005`
- epoch 5000: `0.000005`

Final losses:

- `L_data`: `0.000002`
- `L_phys`: `0.000003`
- `L_ic`: `1.474503e-7`

Held-out results:

- `L2(C_pred, C_analytical)`: `0.001381` (`< 0.05`)
- physics residual mean: `0.000003` (`< 0.01`)
- IC residual: `0.000384` (`< 0.01`)

Gradient non-zero checks:

- layer1 W max grad: `0.330094`
- layer2 W max grad: `0.009358`
- layer3 W max grad: `0.012658`

## Validation

Compiler pin:

- `/workspace/sounio/bin/souc-linux-x86_64`
- SHA256 `3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93`

> **Reproduction commands corrected (2026-09-13).** Six of the commands first recorded here, the
> five focused tests and the MC cross-validation, had the form
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>`. `bin/souc`
> execs that override with its arguments unchanged (it already did at the merge that added this
> file, `bebd78d74c`). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc`
> began refusing the form: lean_single stopped at
> `error: no main` (a current-source lean_single shows why: it opens `run`, which does not exist,
> as a 0-byte source). How the recorded values were originally produced cannot be established
> from this repository's history. `bin/souc` now refuses the form (exit 64). Those six commands below
> use the ELF's raw `<source.sio> <output>` interface from the repository root, because
> lean_single resolves stdlib imports relative to the working directory. The seventh command was
> recorded as `SOUNIO_SOUC_BIN=… bash scripts/ci/dissertation_pbpk_suite_gate.sh`; it is kept
> without the prefix. The pinned binary (sha256 `3cbea2b4…`) is no longer in the repository; the
> re-run used `bin/souc-linux-x86_64`, sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
>
> What the 2026-09-13 re-run checked. Every other figure under Results is the original record for
> the pinned binary and was not re-measured.
>
> - lean_single printed the recorded markers `D6_FULL_INTEGRATION_PASS`,
>   `D2_HARDENED_GENERAL_SHAPES_PASS`, `D3_NN_PRIMITIVES_PASS`, `D4_OPTIMIZER_INTEGRATION_PASS`
>   and `D5_CAPUTO_TENSOR_PASS`. The default Madaros engine (`bin/souc run <test>` with no
>   `SOUNIO_SOUC_BIN` set; `bin/madaros-linux-x86_64` sha256
>   `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) failed each of these
>   five tests with `error[E037]` in `stdlib/tensor/ops.sio`.
> - PBPK28 MC cross-validation: it printed `rel_Hess: 0.175405`, equal to the recorded value; no
>   other output of that run was compared with this file. Its copula sweep lines do not match the
>   saved m1 run log (see `m1_copula_v1.md`). Madaros exited 182 on it.
> - PBPK suite: recorded as `PASS, 50/50`. On 2026-09-13 the gate held 53 tests and reported
>   `FAIL (3 / 53 tests failed)`: `rapamycin_rk4_budget` (rc=1), `rapamycin_epistemic_adaptive`
>   (rc=1) and `pbpk28_sobol_pce` (its 90 s timeout, also when the gate was run by itself).
>   Which of today's tests correspond to the recorded 50 was not checked. Its verdicts were
>   identical with and without the `SOUNIO_SOUC_BIN=` prefix; the script compiles through
>   `SOUC_BIN`, which defaults to `scripts/ci/souc-seq-leansingle.sh`.

Commands (from the repository root):

```bash
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/nn/test_pinn_full_integration_d6.sio /tmp/d6_pinn.elf && chmod +x /tmp/d6_pinn.elf && /tmp/d6_pinn.elf

cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio /tmp/d2_hardening.elf && chmod +x /tmp/d2_hardening.elf && /tmp/d2_hardening.elf

cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/nn/test_nn_primitives_d3.sio /tmp/d3_nn_primitives.elf && chmod +x /tmp/d3_nn_primitives.elf && /tmp/d3_nn_primitives.elf

cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/run-pass/d4_optimizer_integration.sio /tmp/d4_optimizer.elf && chmod +x /tmp/d4_optimizer.elf && /tmp/d4_optimizer.elf

cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/tensor/test_caputo_l1_tape.sio /tmp/caputo_l1_tape.elf && chmod +x /tmp/caputo_l1_tape.elf && /tmp/caputo_l1_tape.elf

bash scripts/ci/dissertation_pbpk_suite_gate.sh

cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio /tmp/mc28.elf && chmod +x /tmp/mc28.elf && /tmp/mc28.elf
```

Results:

- `D6_FULL_INTEGRATION_PASS`: PASS
- `D2_HARDENED_GENERAL_SHAPES_PASS`: PASS
- `D3_NN_PRIMITIVES_PASS`: PASS
- `D4_OPTIMIZER_INTEGRATION_PASS`: PASS
- `D5_CAPUTO_TENSOR_PASS`: PASS
- PBPK suite: PASS, `50/50`
- PBPK28 MC cross-validation: PASS
- rel_Hess: `0.175405`

## Gate

`D6_FULL_INTEGRATION_PASS`

## Reproduction commands (2026-09-16)

The 6 `bin/souc-lean-single-x86_64` lines above run the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and include a `chmod +x` step. Until 2026-09-16 those lines invoked
`bin/souc-linux-x86_64` (sha256
`a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`) and contained no `chmod +x` step.

Run as written today, a line without `chmod +x` exits 126, with `Permission denied` reported
by the launcher: both binaries write the ELF with mode `-rw-r--r--` under umask 0022. How the
earlier runs recorded in this note were invoked is not recorded here, and earlier paragraphs
naming `bin/souc-linux-x86_64` describe those runs.

Measured 2026-09-16 at HEAD `169ac84463`, from the repository root, at the documented `/tmp`
ELF paths (none existed beforehand; each was removed afterwards). Each source below was
compiled and run with the seed and with `bin/souc-linux-x86_64`; every build exited 0, every
run exited 0 after `chmod +x`, and for each source the two binaries' stdout was byte-identical
(stdout only, not the ELF bytes):

- `tests/stdlib/nn/test_pinn_full_integration_d6.sio` (ELF `/tmp/d6_pinn.elf`): stdout contained `D6_FULL_INTEGRATION_PASS`
- `tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio` (ELF `/tmp/d2_hardening.elf`): stdout contained `D2_HARDENED_GENERAL_SHAPES_PASS`
- `tests/stdlib/nn/test_nn_primitives_d3.sio` (ELF `/tmp/d3_nn_primitives.elf`): stdout contained `D3_NN_PRIMITIVES_PASS`
- `tests/run-pass/d4_optimizer_integration.sio` (ELF `/tmp/d4_optimizer.elf`): stdout contained `D4_OPTIMIZER_INTEGRATION_PASS`
- `tests/stdlib/tensor/test_caputo_l1_tape.sio` (ELF `/tmp/caputo_l1_tape.elf`): stdout contained `D5_CAPUTO_TENSOR_PASS`
- `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` (ELF `/tmp/mc28.elf`): stdout contained `M1_COPULA_CHOLESKY_PASS`, `M1_COPULA_SWEEP_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.

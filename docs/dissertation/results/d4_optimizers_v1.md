<!-- docs:meta
topic_id: repo.docs.dissertation.results.d4-optimizers-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d4-optimizers-v1
-->

# D.4 Tensor Optimizers V1

Branch: `codex/sounio-optimizers`

Gate marker: `D4_OPTIMIZER_INTEGRATION_PASS`

## Scope

D.4 implements tensor-level optimizer primitives over the D.1 dynamic tensor
stack on current `origin/main`. Current main includes D.1 tensors but does not
include the D.2 tensor tape module, so this lane deliberately avoids importing
the local D.2 bootstrap commit. `ParameterStore` stores direct gradient tensors
and preserves a `tape_id` field so a later D.2-aware sync adapter can be added
without changing optimizer state semantics.

The implementation is in `stdlib/nn/optimizer.sio` and uses fixed-capacity
parallel arrays for the parameter registry and optimizer moment buffers. This
matches the stability pattern used by the D.2 tape bootstrap and avoids
array-of-struct initialization brittleness in the current compiler.

## Implemented API

- `ParameterStore`
  - `param_store_new`
  - `param_store_register`
  - `param_store_zero_grad`
  - `param_store_set_grad`
  - `param_store_set_grad_value`
  - `param_store_grad_value`
  - `param_store_param_value`
  - `param_store_sync_grads_manual`
  - `param_store_free`
- Adam
  - `AdamState`
  - `adam_new`
  - `adam_new_full`
  - `adam_step`
- AdamW
  - `adamw_step`
- SGD with momentum
  - `SGDState`
  - `sgd_new`
  - `sgd_new_with_momentum`
  - `sgd_step`
- Gradient clipping
  - `clip_grad_global_norm`
  - `clip_grad_value`

## Validation

Focused validation lives in `tests/run-pass/d4_optimizer_integration.sio`.

The test covers:

- Parameter registration and `zero_grad`.
- Adam convergence on `f(w) = (w - 3)^2`.
- AdamW convergence with decoupled weight decay.
- Adam convergence on noiseless scalar linear regression with
  `W_true = 2.0`, `b_true = -1.0`, `n = 50`.
- SGD with momentum convergence on the same linear regression fixture.
- Global-norm and value gradient clipping.
- Deterministic Adam: two independent 100-step runs produce bit-identical
  parameter values.

Representative deterministic values from the implemented update rules:

| Check | Result |
| --- | ---: |
| Adam quadratic, 1000 steps | `w = 2.9991083973` |
| AdamW quadratic, 1200 steps | `w = 2.9977135747` |
| Adam linear regression | `W = 2.0`, `b = -1.0` |
| SGD+momentum linear regression | `W = 2.0`, `b = -1.0` |
| Clip fixture pre-norm | `50.0` |
| Clip fixture post-norm | `1.0` |

> **Reproduction command corrected (2026-09-13).** The command first recorded here was
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>`. `bin/souc`
> execs that override with its arguments unchanged (it already did at the merge that added
> this file, `bebd78d74c`). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc`
> began refusing the form: lean_single stopped at
> `error: no main` (a current-source lean_single shows why: it opens `run`, which does not exist,
> as a 0-byte source). How the recorded values were originally produced cannot be established
> from this repository's history. `bin/souc` now refuses the form (exit 64). The command below uses the
> ELF's raw `<source.sio> <output>` interface. Run it from the repository root: lean_single
> resolves stdlib imports relative to the working directory. The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository; the re-run used `bin/souc-linux-x86_64`, sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
> Re-run 2026-09-13: lean_single printed `D4_OPTIMIZER_INTEGRATION_PASS`. Only that marker was
> re-checked; every other value in this file, including the table above, is the original record.
> The default Madaros engine (`bin/souc run <test>` with no `SOUNIO_SOUC_BIN` set; `bin/madaros-linux-x86_64` sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) rejects this test today
> with `error[E037]` in `stdlib/tensor/ops.sio`.

Focused test command (from the repository root):

```bash
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/run-pass/d4_optimizer_integration.sio /tmp/d4_optimizer.elf && chmod +x /tmp/d4_optimizer.elf && /tmp/d4_optimizer.elf
```

Observed marker:

```text
D4_OPTIMIZER_INTEGRATION_PASS
```

## D.2 Boundary

No D.2 APIs were incorporated. This is intentional: D.4 reads gradients and
updates parameters; it does not require autograd to prove optimizer semantics.
The current `ParameterStore` provides manual gradient setters and a stable
`tape_id` field. When the hardened D.2 tape lands on main, a narrow
`param_store_sync_grads` adapter can copy tape gradients into the existing
store without changing Adam, AdamW, SGD, or clipping behavior.

## Remaining Work

- Add the D.2-backed `param_store_sync_grads` adapter after the hardened tape is
  merged.
- Replace fixed-capacity backing arrays with `Vec<Tensor>` once the compiler
  reliably handles tensor-bearing dynamic collections in this path.
- Reuse this optimizer surface in the D.6 PINN loop after D.3 and D.5 tensor
  gates close.

## Reproduction commands (2026-09-16)

The `bin/souc-lean-single-x86_64` line above runs the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and includes a `chmod +x` step. Until 2026-09-16 that line invoked
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

- `tests/run-pass/d4_optimizer_integration.sio` (ELF `/tmp/d4_optimizer.elf`): stdout contained `D4_OPTIMIZER_INTEGRATION_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.

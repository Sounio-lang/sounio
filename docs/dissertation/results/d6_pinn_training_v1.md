<!-- docs:meta
topic_id: repo.docs.dissertation.results.d6-pinn-training-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d6-pinn-training-v1
-->

---
docs:meta:
  id: dissertation-results-d6-pinn-training-v1
  title: D6 PINN Training Loop V1
  doc_type: result
  status: draft
  owner: Codex
  updated: 2026-05-14
---

# D6 PINN training loop v1

## Gate status

`D6_PINN_TRAINING_LOOP_PASS` is emitted by
`tests/stdlib/nn/test_pinn_training_d6.sio`.

> **Engine dependency (verified 2026-08-17).** This is the one D-series (D1-D6) dissertation
> results doc that omits the pinned-binary/sha256 disclosure its siblings (`d1_tensor_ops_v1.md`
> through `d5_caputo_tensor_v1.md`) all carry, and it turns out that matters here: under default
> Madaros (`bin/souc`), this test file **fails to compile** (`error: use of undeclared variable`,
> `name read_f64`, `Compilation failed!`). Under `SOUNIO_SOUC_ENGINE=lean_single` it compiles
> (tolerating several non-fatal warnings and one further diagnostic under lean_single's known
> typecheck-tolerance behavior — `docs/compiler/KNOWN_LIMITATIONS.md` #1494) and runs to
> completion: `D6_PINN_TRAINING_LOOP_PASS`, `rc=0`. This gate marker has only been produced
> under lean_single.

The v1 proof is an honest PoC rather than a full MLP PINN trainer. It
exercises:

- D.4 `ParameterStore` plus Adam for native parameter updates.
- Integer one-compartment training against `C(t)=C0 exp(-kt)`.
- Fractional one-compartment training against
  `C(t)=C0 E_alpha(-k t^alpha)`.
- D.5 taped Caputo residual and backward pass in
  `tests/stdlib/nn/test_pinn_caputo_residual_d6.sio`.

D.3 is validated separately in this merged lane by
`tests/stdlib/nn/test_nn_primitives_d3.sio`, which emits
`D3_NN_PRIMITIVES_PASS`. A D3 tape smoke inside the same D6 process can
corrupt subsequent scalar training state on the pinned compiler, so the D6
focused training test keeps the scalar trainer isolated and records this as a
compiler/runtime integration limitation rather than hiding it.

## Infrastructure

`stdlib/nn/pinn.sio` adds:

- `PINNLoss`, `PINNLossParts`, `TrainingLog`, and `PINNSanityResult`.
- `pinn_loss_compute` for composite data, physics, and IC/BC losses over the
  D.2 tape.
- `pinn_loss_warmup` and `pinn_loss_warmup_to`.
- `pinn_log_step`.
- `pinn_train_step_scalar`, a D.4-backed scalar parameter update.
- `pinn_integer_1comp_sanity`.
- `pinn_fractional_1comp_sanity`.
- `pinn_fractional_residual_taped`, which calls `tape_tensor_caputo_l1`.

The training sanity tests identify the scalar decay rate `k` from synthetic
data using Adam over a one-parameter native loop. The physics residual path is
tested separately through D.5's taped Caputo operator because the current v1
stack does not yet have differentiable gather/slice for the `C_last` term, and
D.4 does not yet sync arbitrary tape gradients into trainable MLP parameters.

## Results

Integer one-compartment:

- Steps: 3000
- Target: `k=0.1 h^-1`
- Learned parameter: within `5e-3` of target
- L2 versus analytical trajectory: `<1e-3`

Fractional one-compartment:

- Alpha: `0.8`
- Steps: 5000
- Target: `k=0.1 h^-alpha`
- Learned parameter: within `2e-2` of target
- L2 versus Mittag-Leffler analytical trajectory: `<0.05`

Caputo residual integration:

- `pinn_fractional_residual_taped` forwards through `TAPE_CAPUTO_L1`.
- Backward pass produces non-zero trajectory gradient.
- Residual on the analytical fractional decay witness is bounded below `0.1`
  on the fixed `n=20`, `dt=0.1` stencil used for the focused test.

## Limitations

This is not yet a production MLP PINN training loop. The missing pieces are:

1. A D.4 adapter that copies D.2 tape gradients into `ParameterStore`.
2. Differentiable slice/gather for `C_last` in fractional physics residuals.
3. A same-process D3+D6 stress test after the pinned compiler's tape-owned
   struct/shape metadata fragility is removed.

The current PoC is sufficient to show the native pieces composing at the
training-loop level without overclaiming full neural PINN training.

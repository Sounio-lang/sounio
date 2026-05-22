# f32_assoc_gum — Phase Z golden fixtures (PENDING)

Golden PTX + reference buffers for the associator-augmented 8-compartment PBPK
kernel (L1 of the AssociatorField butterfly thread).

## Status: scaffold only

The CPU sampler/oracle (`scripts/gpu/kaxi_pbpk_8comp_assoc_sampler.c`) and the
gate (`scripts/ci/kretikos_kaxi_phase_z_assoc_gate.sh`) are runnable today and
verify the CPU-side truth claims (determinism, the associative→zero invariant,
positive augmentation).

The GPU kernel itself is the **deferred, multi-session piece**:

- `pbpk_8comp_assoc.ptx` — NOT YET EMITTED. Requires extending the K-AXI → PTX
  emitter (`self-hosted/gpu/kretikos_kaxi_to_ptx.sio`) to lower `AssociatorField`
  arithmetic: an 8-component octonion multiply (reuse the verified Phase L
  `octonion_assoc.ptx` emitter) plus the `aug = κ·‖[A,B,C]‖²` reduction.
- Once emitted, drop `pbpk_8comp_assoc.ptx` here; the gate's TC-4 auto-activates.

## Reference

- Type + math: `stdlib/algebra/associator_field.sio`
- L0 acceptance: `tests/run-pass/associator_field_octonion.sio`,
  `tests/run-pass/associator_field_pentagon.sio`
- Existing octonion PTX to reuse: `tests/golden/kaxi_ptx/f32_2c/octonion_assoc.ptx`
- Predecessor gates: Phase X (1-comp), Phase Y (2-comp GUM covariance)

# Parallel Branch Sweep

- generated_at_utc: 2026-03-15T22:15:00Z
- repo_head: 03d1ae50
- origin_main: 03d1ae50
- live_worktrees: 1
- unmerged_branches: 0

## Summary

The repository currently has no surviving unmerged branches. Recent
multi-agent GPU and epistemic work has already landed in `main`, so the
current duplication risk is re-implementing functionality that already exists
in the checked-out tree, not missing a side branch.

## Recent Landed Work

- `e00d13ab` `[gpu] Track C: epistemic WMMA backward pass - GUM gradient rules`
  touched `self-hosted/gpu/kernel_ir.sio`, `self-hosted/test_epistemic_autodiff.sio`,
  and `scripts/sprint_epistemic_autodiff_gate.sh`.
- `5c35a91f` `[gpu] Track A+B: tiled GEMM shared-mem + SPIR-V epistemic compute backend`
  touched `self-hosted/gpu/kernel_ir.sio`, `self-hosted/gpu/lower_to_ptx.sio`,
  `self-hosted/gpu/spirv_lower.sio`, and backend tests/gates.
- `0d358e37` `[gpu] Epistemic WMMA tensor core kernel - GUM uncertainty through WMMA`
  touched `self-hosted/gpu/kernel_ir.sio`, `self-hosted/gpu/lower_to_ptx.sio`,
  `self-hosted/test_epistemic_wmma.sio`, and the WMMA example.

## Hot Files

Avoid duplicating work in these files unless the new task explicitly extends
the landed WMMA/PTX/SPIR-V work:

- `self-hosted/gpu/kernel_ir.sio`
- `self-hosted/gpu/lower_to_ptx.sio`
- `self-hosted/gpu/spirv_lower.sio`

## Lower-Risk Lanes

These surfaces do not show the same fresh branch churn:

- `stdlib/epistemic/knowledge.sio`
- `stdlib/units/qudt.sio`
- `self-hosted/check/epistemic.sio` (active historically, but not part of the
  March 15 GPU burst)

## Coordination Notes

- `artifacts/omega/agent_handoff.log.md` contains an old open lock, but it is
  for refinement/parser files, not the epistemic or GPU files above.
- The working tree currently has an untracked `examples/kernel_epistemic_wmma_matmul.ptx`
  file that appears to be generated output.

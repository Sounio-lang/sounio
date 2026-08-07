# Job 8758 — verify papercut after successful locus discrimination

## Execution

- Job `8758` on `gpuorangefs-r770-proxmox`, elapsed `00:07:37`, exit `1:0`
- Snapshot `a32b6597de`
- Stage `/orangefs/training/cs6-tm2r-rho3-locus-a32b6597de-20260807T014853Z`

## Scientific result (worker receipt retained)

Classification emitted by the worker:

```text
RECONDITION_COLLAPSES_RESIDUAL_RANK
```

All six projection scales (powers 12…7) have:

- `raw_projection_rho3_positive = true`
- `reconditioned_rho3_positive = false`
- `residual_pure_direction_rank = 3`

Powers 18…13 remain `CENTERED_PREDICTOR_ESCAPED`. Implementation checks all
passed individually. This confirms the residual `rho3` label dies in the
post-section point-coefficient QR renumbering, not in residual flow or `w=0`
projection.

## Verify failure

```text
rho3-locus verify error: implementation check names are not unique
```

Cause: `predictor_center_matches_frozen_receipt` was recorded both inside
`centered.frozen_predictor` and again in the locus worker `main`. Fix: drop the
duplicate check in the locus worker and re-run for a clean verified receipt.

## Claim boundary

Even after verification, this is only a locus classification. It does not accept
a covering edge, authorize full transport, or solve an open problem.

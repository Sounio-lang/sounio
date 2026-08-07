# Target-23 residual rho3 locus diagnostic

## Objective

Discriminate where residual `rho3` weight vanishes on the frozen predictor-centered
XLEL chart: residual event flow, `w=0` projection, or post-section
point-coefficient QR recondition.

## Foundry execution

| Field | Value |
|---|---|
| Job | `8759` |
| Node | `gpuorangefs-r770-proxmox` |
| Snapshot | `9f1ec756a0` |
| Stage | `/orangefs/training/cs6-tm2r-rho3-locus-9f1ec756a0-20260807T015738Z` |
| Elapsed | `00:07:31` |
| Exit | `0:0` |
| Receipt SHA-256 | `6b3ee1c7244e6618d4256a0fc5afad42c9fcd328e3dc70d79c6a35028ffe3016` |
| Foundry verified SHA-256 | `a278aac73bc57f534d2484f0368efde8ca2c7c1ad12d38a03591999db13da516` |

Prior job `8758` produced the same locus classification but failed verify on a
duplicate implementation-check name; forensics retained under
`foundry_failure_8758/`. Job `8757` was cancelled while r740 was occupied by
unrelated SAN training.

Local re-verification and 12/12 negative mutations pass after a verifier
strengthening that rejects empty projection-scale packages for non-empty locus
classes.

## Result

Classification: **`RECONDITION_COLLAPSES_RESIDUAL_RANK`**.

| Powers | Status | raw ρ₃ | reconditioned ρ₃ | residual rank |
|---:|---|---|---|---:|
| 18…13 | `CENTERED_PREDICTOR_ESCAPED` | — | — | — |
| 12…7 | `RECONDITION_COLLAPSES_RESIDUAL_RANK` | true | false | 3 |

Additional facts:

- Implementation checks: **20/20**.
- Centered state still has positive `rho3` weight.
- Event-time model and raw projected carriers retain positive `rho3` weight on
  every projection scale.
- After `point_coefficient_recondition`, residual pure-direction rank is exactly
  **3** and the remapped fourth residual label is zero.

## Discrimination

The previous event-centered refusal
(`CENTERED_SYMBOLIC_DEPENDENCE_LOST` / ρ₃ weight zero) is **not** a genuine
erasure of residual parametric dependence by the residual event chart. It is a
**coordinate artifact of QR residual renumbering after section projection**: the
raw projected carrier still carries `rho3`, but the reconditioner rebuilds only
three nonzero residual pure directions in ambient 4-space with `w ≡ 0`.

This supersedes the interpretation that residual flow itself kills `rho3`.

## Next falsifier

Accept residual symbolic dependence on the **raw projected carrier** (pre-QR),
or introduce a section-aware residual basis of honest rank 3, while keeping QR
only as an optional width-control coordinate change. Then re-run the
predictor-centered acceptance gates without requiring a fourth renumbered residual
label that the section geometry cannot support.

Do not launch a complete support transport until a residual-aware acceptance
gate passes with explicit nonclaims unchanged.

## Claim boundary

This receipt certifies no covering relation, recurrent graph, degree,
determinant edge, chaos, novelty, priority, or open-problem solution. It only
classifies the locus of a residual label collapse under the current
reconditioner.

# Target-23 TM2R predictor-centered event chart

## Objective

Validate a predictor-centered event-time chart on the frozen XLEL critical leaf
before any new complete transport. The fixed shift is the exact rational midpoint
of the event-local raw predictor at radius `1/128`. Acceptance requires signed
Picard residual slabs, a strictly negative whole-tube derivative, parametric
interval Newton inclusion, exact section projection `w=0`, and positive aggregate
weights for all six normalized variables `xi`, `eta`, `rho0`–`rho3`.

## Foundry execution

| Field | Value |
|---|---|
| Job | `8749` |
| Node | `gpuorangefs-multi-r740-proxmox` |
| Snapshot | `f3b5a2f3c3` |
| Stage | `/orangefs/training/cs6-tm2r-event-centered-f3b5a2f3c3-20260806T230133Z` |
| Elapsed | `00:40:06` |
| Exit | `0:0` |
| Receipt SHA-256 | `37464bf6b240e9dc621aed3b4fcdf02b8276ce44a21623001aea0d62f6ce29f7` |
| Verifier SHA-256 | `2225f1023c437084b3d6190436c6740bfbb00eadac3b1a3c5caaf124d0bfd624` |

Prior attempt job `8746` failed verify because Arb `mid()` differed from the
exact rational midpoint. That bug was fixed in `f3b5a2f3c3`; forensic notes are
under `foundry_failure_8746/`.

Local independent re-verification and all 14 negative mutations passed after
download.

## Result

Classification: **`PREDICTOR_CENTERED_EVENT_REFUSED`**.

- Implementation checks: **20/20** passed.
- Frozen predictor center matches the event-local exact midpoint.
- Fixed shift equals that center.
- Residual scale ladder `2^-18` … `2^-7` exhausted without acceptance.
- No point fallback, box flattening, or complete transport.

### Scale statuses

| Powers | Status | Count |
|---|---|---:|
| 18 … 13 | `CENTERED_PREDICTOR_ESCAPED` | 6 |
| 12 … 7 | `CENTERED_SYMBOLIC_DEPENDENCE_LOST` | 6 |

At every accepted-enough residual scale (powers 12–7), Newton inclusion and
section projection progress far enough to emit a projected carrier with
`pure_source_monomials_retained = 15`, but the aggregate weight of normalized
variable **index 5 (`rho3`)** is exactly zero. Critical and centered (pre-residual)
states still preserve all six variables.

## Discrimination

This gate closes the previous blocker class `UNRESOLVED_ENCLOSURE` for the
zero-centered slab placement of the production event chart: a predictor-centered
fixed shift is executable, hash-bound, and independently verified. The surviving
obstruction is **symbolic dependence loss of `rho3` after residual event
projection and reconditioning**, not chart drift, implementation failure, or
failure to match the frozen center.

It does **not** authorize a complete second-return transport or a covering
relation.

## Next falsifier

Preserve `rho3` through the residual event-time chart. Concrete attacks, in
order of preference:

1. Recondition the residual projected carrier with a QR doubleton/tripleton that
   keeps a nonzero `rho3` generator before the six-weight gate.
2. Split the residual domain or the critical leaf along the direction that
   currently collapses `rho3`.
3. Carry `rho3` as an explicit parametric remainder through interval Newton
   rather than only through point-coefficient QR after projection.

Do not launch a full support/composability transport until a scale is accepted
with all six positive weights.

## Claim boundary

This receipt certifies no complete support, h-set target, covering relation,
recurrent graph, degree, determinant edge, chaos, novelty, priority, or solution
of an open problem. Production and legacy transport paths remain unchanged.

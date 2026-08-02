# CS6 V7-B subdivision ladder scout

**Status:** bounded candidate-discovery experiment over one nested descendant
lineage per frozen V7-B cell. This is not a cover of any parent cell and does
not establish V7-B eligibility.

## Question

The frozen six-attempt bridge failed in two ways: the masked target could not
complete the one-step Newton section crossing, while the controls completed the
crossing but retained unresolved C1/C2 orientation intervals.

This scout asks one smaller question:

```text
Does shrinking the same frozen cell, without changing the vector field,
section, or carrier, expose a descendant where the homogeneous bridge
computation becomes valid?
```

For each parent ordinal `22`, `23`, and `24`, the runner follows the nested
lower-left dyadic descendant at depth deltas `+1`, `+2`, `+3`, and `+4`. It runs
both frozen carrier choices at every level, for `3 * 4 * 2 = 24` attempts.

## Result

```text
ATTEMPTS_COMPLETED=24
DESCENDANT_PROBE_PASS=4
DESCENDANT_PROBE_REJECTED=20
SECTION_RESIDENT_CROSSING_UNAVAILABLE=0
CERTIFICATE_PASS=0
TARGET_FIRST_CROSSING_RECOVERY_DELTA=1
TARGET_FIRST_PROBE_PASS_DELTA=4
ALL_PARENT_CARRIERS_HAVE_CANDIDATE=false
```

The target parent `23` recovers the section crossing at the first subdivision
level. Its lower-left descendant first emits `PROBE_PASS=true` at depth delta
`+4`, for both carriers. Parent `22` has the same two carrier-local passes at
`+4`. Parent `24` has no pass along this one tested lineage.

All four passing rows still report unresolved C1 and C2 hull orientations and
`CERTIFICATE_PASS=false`. Their narrower contribution is that both event charts
are certified and the homogeneous computation becomes valid. They are useful
candidate locations for a full sibling-cover experiment, not determinant
certificates.

## Interpretation boundary

A `DESCENDANT_PROBE_PASS` means only that the existing worker emitted
`PROBE_PASS=true` for that one smaller descendant. It does not imply that:

- the other descendants pass;
- the parent cell is covered;
- the C1/C2 determinant orientation is certified;
- either carrier wins;
- V7-B is eligible;
- an open problem is solved.

The useful novelty signal is geometric localization. If a coarse parent fails
but a nested descendant passes, the obstruction is not uniform across the
tested parent representation. A later full-cover experiment can use the first
passing depth as a starting point, but must enumerate and verify every sibling.

## Reproduce

```bash
bash scripts/ci/cs6_v7b_subdivision_ladder_gate.sh
```

The gate compiles the existing CAPD worker, runs the bounded ladder with four
local worker slots, verifies every receipt independently, deletes the generated
binary, and keeps the textual evidence under
`scripts/research/receipts/cs6_v7b_subdivision_ladder_v1/`.

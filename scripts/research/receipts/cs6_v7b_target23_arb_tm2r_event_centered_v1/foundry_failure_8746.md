# Job 8746 failure — predictor-center mismatch

## Execution

- Job: `8746`
- Node: `gpuorangefs-multi-r740-proxmox`
- Elapsed: `00:38:52`
- Exit: `1:0`
- Stage: `/orangefs/training/cs6-tm2r-event-centered-6b840db8db-20260806T215453Z`
- Snapshot: `6b840db8db`

## Symptom

Verifier refused with:

```text
event-centered verify error: receipt predictor center does not match the frozen exact center
```

The worker finished (≈39 min) and produced a JSON receipt, but `run.sh` used
`set -e` around verify and discarded the temporary receipt on failure. No
`event_centered.json` was retained on OrangeFS.

## Root cause

`frozen_predictor` formed the fixed-shift center as

```python
center = Fraction(base.exact_fraction(predictor_range.mid()))
```

Arb midpoint need not equal the exact rational midpoint of the serialized
endpoints. The frozen center from the event-local analysis is

```text
(lower + upper) / 2
```

which is the quantity bound by the contract and by `EXPECTED_CENTER`. Using
`mid()` therefore made `predictor_center_q` disagree with the frozen value even
when the enclosure endpoints themselves were consistent.

## Fix

1. Compute `center = (Fraction(lower_fraction) + Fraction(upper_fraction)) / 2`.
2. Keep requiring `center == EXPECTED_CENTER`.
3. Retain the worker receipt on verify failure so the next refusal is forensic.

## Claim boundary

This is an implementation fix of the event-local chart gate only. It does not
certify covering, recurrence, chaos, or an open-problem solution.

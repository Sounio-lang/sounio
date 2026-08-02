# CS6 V7-B bridge failure anatomy

**Status:** post-execution diagnostic over the frozen V7-B six-attempt bridge
checkpoint. This does not rerun CAPD, does not change the V7-A.1 evidence, and
does not claim V7-B eligibility.

## Plain result

The six-attempt execution answered the first question: neither candidate carrier
closed a full H-PG bridge on the frozen cells. This diagnostic answers the next
question:

```text
Where did each failed attempt stop?
```

The answer is narrow:

```text
ATTEMPTS_ANALYZED=6
WORKER_SUMMARIES_EMITTED=4
SECTION_RESIDENT_CROSSING_UNAVAILABLE=2
C1_C2_ORIENTATION_UNRESOLVED=4
UNKNOWN_ANATOMY=0
NEXT_EXPERIMENT_CLASS=c1_c2_orientation_and_section_crossing_reparameterization
```

For the masked target cell `23`, both carriers fail before the worker can emit a
summary, at the known `one-step Newton crossing was not available` point.

For the neighboring control cells `22` and `24`, both carriers reach the worker
summary. In all four of those rows, the structural checks and Liouville
orientation are true, but C1 and C2 hull orientations remain unresolved. The
affine/projective/homogeneous certificate paths also do not certify, so the
worker rejects the full probe with `PROBE_PASS=false`.

## What this means

The current evidence does not point to FPGA acceleration. It points to geometry
inside the section/crossing and C1/C2 orientation setup:

```text
c1_c2_orientation_and_section_crossing_reparameterization
```

That means the next lane should try a bounded alternative chart, section, or C1/C2
orientation parameterization, then rerun the same six-attempt gate. If the same
classes persist, V7-B should stay rejected for this frozen carrier pair. If a
candidate changes the failure class, it can be promoted only to another
checkpoint, not to a theorem or open-problem solution.

## Claim boundary

Allowed:

- classify the six existing execution receipts;
- say that the masked target still fails at section-resident crossing;
- say that the controls fail with unresolved C1/C2 orientations after worker
  summary emission;
- nominate the next experiment class.

Not allowed:

- claim V7-B eligibility;
- choose a V7-B winner;
- claim full H-PG pipeline evaluation;
- claim FPGA execution;
- claim an open problem is solved.

## Gate

```bash
bash scripts/ci/cs6_v7b_full_hpg_bridge_anatomy_gate.sh
```

The gate writes receipts under
`scripts/research/receipts/cs6_v7b_full_hpg_bridge_anatomy_v1/`.

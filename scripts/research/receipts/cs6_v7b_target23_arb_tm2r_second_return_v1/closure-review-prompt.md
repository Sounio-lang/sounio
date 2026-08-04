# Closure review: explicit event-position slab containment

The first dual review of the Arb TM2R event projection found one BLOCKER: the
implementation needed to check trajectory-position containment for the entire
backward Newton time slab, rather than infer it merely from time-domain
inclusion.

The worker now performs the following before using the complete event-tube
vector field for the residual time correction:

```python
full_slab_step = arb(delta.lower())
slab_box, slab_iterations, slab_contraction = signed_picard_box(
    end_ranges, full_slab_step
)
if not all(
    event_component.contains(slab_component)
    for event_component, slab_component in zip(
        phase.event_tube, slab_box, strict=True
    )
):
    raise Refusal("PROJECTION_SLAB_ESCAPES_EVENT_TUBE", ...)
```

Here `signed_picard_box` constructs a rigorous self-map tube from the complete
endpoint carrier over the signed time interval `[delta.lower(),0]`, checks a
strict Lipschitz contraction, and refuses if Picard closure fails. The fresh
receipt reports:

```text
PROJECTION_SLAB_PICARD_ITERATIONS=2
PROJECTION_SLAB_PICARD_CONTRACTION_UPPER_Q=
84581024868824121122075377503327827390466147024945972535148796867893332776437/
1852673427797059126777135760139006525652319754650249024631321344126610074238976
PROJECTION_SLAB_CONTAINED_IN_EVENT_TUBE=true
```

The exact verifier requires the contraction to lie strictly in `(0,1)`, the
containment flag to be true, and rejects mutations of either field. The gate
freshly recomputed the worker and rejected 35/35 receipt mutations.

The documentation also now says that explicit `xi,eta` dependence is preserved
only up to interval enclosure of event-time/source correlation, and makes no
Jacobian claim.

Review only whether the original BLOCKER is now discharged and whether this
introduces any new soundness issue. Return PASS or a precise remaining
BLOCKER/MAJOR. Do not infer a second return: the retained result still refuses
post-event step 708 and sets `FULL_LEAF_SECOND_RETURN_CERTIFICATE=false`.

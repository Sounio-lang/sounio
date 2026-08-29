# GARDEN: LOOM Product Activation Dark Gate V1

Status: `PREREGISTERED`

## Question

Can the existing LOOM subprocess membrane exercise frozen action `9031` on
every product probe through the same resident Sounio process, without treating a
fixture as live material authority or changing the existing execution decision?

## Purpose

This phase proves product-path attachment, not product activation. It closes
three architectural gaps:

1. the subprocess membrane currently supervises actions through resident v2;
2. the affine activation capsule is available only through resident v5;
3. an OCaml caller must not synthesize action-`9031` expected results or invent
   a positive material frame.

The dark gate changes the membrane to one resident v5 generation and sends a
Sounio-produced current-integration projection to route `6` before the existing
route-`2` execution decision. The projection is explicitly nonauthorizing.

## Sounio Projection

The projection artifact is extracted mechanically from the frozen action-`9031`
fixture bundle. Its full bytes must hash to the frozen
`fixture_bundle_sha256`; the selected `current_material` frame remains inside
that Sounio-produced bundle with its Sounio-produced expected decision.

OCaml may select the uniquely labelled frame and submit it. OCaml must not:

- encode the expected decision code or reason;
- edit any field in the frame;
- substitute a positive fixture;
- regenerate the frame from local assumptions;
- use the projection as evidence of live material eligibility.

The projection is a wiring canary. It is not a material receipt.

## Shared Resident

The membrane must open exactly one resident v5 process. The resident performs:

1. route `1` request bookkeeping;
2. route `6` action-`9031` dark decision;
3. route `2` existing subprocess policy decisions;
4. route `3` existing closure observation;
5. route `1` response and stop bookkeeping.

The activation capsule attaches to that resident without taking ownership of
its lifecycle. A second Sounio authority process is forbidden.

## Closed Product Stage

`production_activation=false` remains frozen. Therefore:

- a Sounio `DENY` is recorded and the existing membrane decision continues;
- a Sounio `ALLOW` is an unexpected stage transition and fails closed before
  subprocess launch;
- timeout, EOF, malformed output, hash drift, missing projection, duplicate
  label, resident replacement, or receipt failure fails closed;
- the dark gate never converts a `DENY` into authority and never authorizes the
  subprocess.

The existing route-`2` membrane remains the only execution decision during this
phase.

## Receipt

Every dark decision records:

- action-`9031` manifest hash;
- action-`9031` semantics hash;
- resident-v5 manifest and runtime hashes;
- projection bundle and selected-frame hashes;
- resident generation, PID, sequence, result code, and result hash;
- `production_activation=false` and `authorizing=false`.

## Sabotage Controls

The gate must prove:

1. replacing the selected frame with the positive `seal` fixture causes an
   unexpected `ALLOW` and blocks launch;
2. changing one byte in the projection bundle refuses before resident spawn;
3. deleting or duplicating the `current_material` label refuses before spawn;
4. changing the action or resident manifest refuses before spawn;
5. the existing Sounio authority sabotage still attributes the laundering
   refusal to action `9031`;
6. product output with the valid dark projection matches the pre-cutover
   membrane result for both an allowed benign probe and a denied Python probe.

The positive-fixture sabotage is the important control: it proves the product
stage gate, rather than the current Sounio `DENY`, is what prevents premature
activation.

## Acceptance Boundary

This phase may establish:

- `product_path_attached=true`;
- `single_resident=true`;
- `dark_gate_executed=true`;
- `existing_membrane_parity=true`;
- `unexpected_allow_refused=true`.

It must retain:

- `projection_authorizing=false`;
- `live_material_frame=false`;
- `capsule_material=false`;
- `production_activation=false`;
- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.

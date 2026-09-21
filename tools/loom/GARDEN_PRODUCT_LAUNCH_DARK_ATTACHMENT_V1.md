# GARDEN: LOOM Product Launch Dark Attachment V1

Status: `PREREGISTERED`

## Question

Can every real LOOM session launch and kernel recovery cross the frozen Sounio
action-`9031` dark gate before any session capability, descriptor, daemon, or
provider process is created, while the gate remains nonauthorizing and the
existing launch semantics remain unchanged?

## Founder Direction

This phase continues the current LOOM architecture. It does not create a second
launcher or a second semantic authority. Sounio remains the semantic authority;
OCaml attaches the already frozen action `9031` to the product launch path.

The required authority order remains:

```text
GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY
```

Action `9031` already satisfies the first three stages. This phase is an
operational attachment below that frozen semantic parent. It cannot change the
action, its expected result, or its authority role.

## Product Boundary

The attachment executes before the first persistent mutation of a launch:

1. `start` validates that the target lane is not active or recoverable;
2. the launch observation executes and its durable receipt is fsynced;
3. only then may LOOM write the session capability and fork the daemon;
4. `provider-start` and `provider-open` inherit the same path through `start`;
5. `recover` executes the observation before forking the replacement kernel.

The dark observation submits the Sounio-produced `current_material` projection
to route `6` of one resident Sounio v5 process. The projection remains a wiring
canary. It is not a live material frame and cannot authorize a launch.

## Operational Binding

Every product-launch receipt binds the dark decision to:

- operation: `start` or `recover`;
- agent, lane, and session identity;
- canonical working directory;
- command digest already used by the LOOM session descriptor;
- action-`9031`, operational, resident-v5, projection, and selected-frame
  hashes;
- resident PID, generation, sequence, decision code, and result hash;
- `authorizing=false` and `production_activation=false`.

The operational binding is evidence that a specific launch crossed the frozen
gate. It is not evidence that the static projection measured the live host.

## Failure Semantics

The product path fails closed before mutation when:

- policy, manifest, projection, or runtime hashes drift;
- the resident cannot start, times out, returns EOF, or emits malformed output;
- the dark receipt cannot be durably written;
- the Sounio projection unexpectedly returns `ALLOW` while production
  activation is closed;
- a recovery descriptor lacks a valid command digest or session identity.

A valid Sounio `DENY` is recorded and normal launch continues. The decision is
observational, not authorizing.

## Sabotage Controls

The gate must prove:

1. replacing `current_material` with the positive `seal` projection records a
   Sounio `ALLOW` and refuses before capability creation, daemon fork, provider
   process creation, or recovery-kernel fork;
2. making the receipt destination unwritable refuses before the same effects;
3. changing one byte of the projection or parent manifest refuses before
   resident spawn and before session mutation;
4. a normal direct `start` crosses the gate exactly once and launches;
5. `provider-open` crosses the same `start` attachment exactly once, without a
   provider-specific bypass;
6. `recover` crosses the gate exactly once and preserves the original command
   digest;
7. the deliberate Python oracle remains refused by the existing action `9023`
   membrane and is never promoted to semantic authority.

The positive-fixture sabotage is causal: the Sounio result changes to `ALLOW`,
and this product-stage rule must be what prevents the launch.

## Acceptance Boundary

This phase may establish:

- `real_start_path_observed=true`;
- `provider_start_path_observed=true`;
- `provider_open_path_observed=true`;
- `recover_path_observed=true`;
- `prelaunch_receipt_bound=true`;
- `prelaunch_failure_closed=true`;
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

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-product-launch-dark-20260829
Owner: codex-1
Concept-IDs: SOUNIO-LOOM-KERNEL-PEER-ACTIVATION-CAPSULE
Intent-Preserved: no launch authority exists without a Sounio decision and a durable receipt
Transformation: attach frozen nonauthorizing action 9031 to real LOOM launch and recovery paths
Types-Changed: none
Effects-Changed: prelaunch observation and receipt become mandatory before launch mutation
IR-Changed: none
Claims-Introduced: real LOOM launch paths crossed the frozen dark gate before mutation
Claims-Forbidden: live material authority, production activation, launch authorization, exec attachment, commit attachment, CI attachment
Assumptions: source-tree alpha runs from a Sounio worktree containing the frozen policy bundle
Write-Set: tools/loom product-launch dark attachment files and canonical runtime installer
Read-Set: frozen action 9031, resident v5, existing start/provider/recover implementation
Positive-Witness: direct start, provider-open, and recover each emit exactly one bound receipt and continue
Negative-Witness: positive projection or receipt failure prevents capability, daemon, provider, and recovery effects
Acceptance-Gate: bash scripts/ci/sounio_loom_product_launch_dark_attachment_selftest.sh
Integration-Target: lane/codex-1/loom-mainline-20260827
Authoritative-Only-If: frozen action 9031 and resident v5 hashes verify and all causal controls pass
```

# GARDEN: LOOM Effective Mount Truth V9

Status: `PREREGISTERED_AFTER_V8_READY_FALSIFICATION`

## Falsifying Observation

V8 crossed namespace construction, typed file bounds, principal rendezvous
opacity, root validation, seccomp installation, and READY. The root observer
then refused because `PrivateTmp` was not the requested literal `no`.

The frozen receipt is
`tools/loom/evidence/loom-process-witness-effect-root-v8-host-attempt-v1-20260828.txt`
with SHA-256
`f58fbc4513831cb5d503a1c65b1f5e32865829f24360264a11b2192e0338cae7`.

On systemd 257, `DynamicUser=yes` promotes `PrivateTmp` to the disconnected
mode, forces `ProtectSystem=strict`, and promotes `ProtectHome` to at least
`read-only`. A diagnostic transient unit reported the coarse D-Bus property
`PrivateTmp=yes` despite requesting `PrivateTmp=no`.

## Effective Configuration

V9 requests the effective security modes directly:

- `DynamicUser=yes`;
- `PrivateTmp=disconnected`;
- `ProtectSystem=strict`;
- `ProtectHome=read-only`.

The property strings are configuration evidence, not filesystem authority.
The root host observer must judge the resulting mount graph.

## Mount-Graph Authority

At READY, the root host observer must prove all of the following from the
process mount namespace:

1. `/` is read-only and rooted at the exact materialized capsule;
2. `/tmp` and `/var/tmp` are distinct read-only mounts;
3. both temporary mounts resolve to the same immutable capsule `/tmp` source;
4. both temporary directories are empty and principal-non-writable;
5. `/run/systemd/incoming` is a distinct exact-unit propagation mount, empty to
   the root observer and opaque to the principal;
6. `/sys` is read-only `sysfs` sourced from `sysfs`;
7. no `/proc`, `/home`, `/root`, `/run`, `/var`, or `/etc` mount is exposed;
8. the private mount namespace and incoming mount disappear with the unit.

If an effective systemd property is stronger but changes any required mount
fact, the treatment refuses. A requested property never overrides contrary
mountinfo.

## Preserved Contracts

V9 preserves the V8 typed file bounds, V7 observer split, exact backing tree,
static cell and payload, zero capabilities, `NoNewPrivileges`, private network,
`RestrictNamespaces`, `RestrictSUIDSGID`, `LockPersonality`,
`MemoryDenyWriteExecute`, and the four-syscall positive seccomp surface.

The twelve effect families, fourteen action-9025 authority cases, and four
bootstrap cases are unchanged. Expected results continue to originate in
Sounio.

## Sounio-First Order

Before a V9 native or host-gate byte changes, a Sounio executable must freeze:

1. the V8 manifest and READY-stage refusal hashes;
2. the effective systemd configuration above;
3. mountinfo as authority over requested property text;
4. the unchanged typed bounds and observer split;
5. the unchanged root, bootstrap, and action-9025 matrices;
6. the closed product and authority boundary.

C++ remains transitional `MATERIAL_PARITY`. Shell transports and compares
receipts but cannot choose effective semantics or expected results.

## Acceptance

`root_treatment=true` requires READY plus every mount-graph fact above and all
three absence controls returning `226/NAMESPACE`. Property agreement without
the mount graph is insufficient.

Until then every promotion flag remains false: `root_treatment`,
`bootstrap_sabotage`, `material_coverage`, `complete_effects`,
`material_execution`, `launch_open`, `recycle_open`, `exec_attached`,
`commit_attached`, `ci_attached`, `parity_open`, and `claim_ready`.

## Nonclaims

- V9 does not treat `PrivateTmp=yes` alone as evidence of isolation.
- V9 does not weaken read-only temporary storage.
- V9 does not weaken the exact root or observer split.
- V9 does not reinterpret the V8 refusal as a pass.
- V9 does not claim any material effect-family sabotage twin.
- V9 does not open arbitrary commands or any LOOM product attachment.

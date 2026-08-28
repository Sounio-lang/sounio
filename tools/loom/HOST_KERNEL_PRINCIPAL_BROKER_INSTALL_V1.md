# LOOM Host Kernel-Principal Broker Installation V1

Status: `HOST_PROMOTION_PATH_MATERIAL_REFUSED`

## Authority Boundary

Installation does not create semantics. The only semantic authorities promoted
by this path are the source-fresh executables bound by frozen Sounio actions
`9027`, `9028`, and `9029`.
The C++20 broker remains a transitory `MATERIAL_PARITY` adapter for Linux and
systemd primitives. Shell code performs packaging and service-manager actions;
it is not an expected-result oracle.

The required order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

This installer reaches neither `PARITY_OPEN` nor `CLAIM_READY`. It cannot change
`material_broker=false`, `material_capsule=false`, or
`material_invocation=false`. It does not invent an `ALLOW` decision.

## Immutable Release

`scripts/dev/install_loom_kernel_principal_broker.sh` always rebuilds the action
`9027`, `9028`, and `9029` Sounio executables and the C++20 broker from the
checked-out sources. It runs all three frozen Sounio gates before creating
installation files and refuses when any rebuilt authority hash differs from its
frozen manifest.

Each promotion creates one immutable release directory:

`/usr/lib/sounio/loom/releases/9029-<lease-manifest-prefix>-<capsule-manifest-prefix>-<invocation-manifest-prefix>-<broker-prefix>-<bundle-prefix>`

It contains:

- the exact three frozen manifests, mode `0444`;
- the exact three source-fresh Sounio authority executables, mode `0555`;
- the source-fresh transitory broker, mode `0555`;
- the bootstrap, installation, and InvocationCell admission contracts, mode `0444`;
- the exact installer source that produced the release, mode `0444`;
- a receipt that names both language roles and all promoted hashes.

The bundle hash covers the installer, both systemd units, and all three contracts.
Consequently, operational-policy drift creates a new release identity even when
the Sounio authority and C++ executable happen to remain byte-identical.

The directory is created next to its final location, synchronized, changed to
mode `0555`, and renamed into place. An existing release is reused only when its
hashes, modes, and, on a host install, root ownership still match. Drift fails
closed; the installer never repairs an allegedly immutable release in place.

## Stable Host Paths

After the release is complete, promotion installs:

- `/usr/libexec/sounio/loom-kernel-principal-broker`, an atomic symlink to the
  immutable release;
- root-owned systemd service and socket units under `/etc/systemd/system`;
- `/etc/sounio/loom-principal-broker.conf`, mode `0600`, pointing directly at
  all three versioned Sounio manifests and executables;
- contracts under `/usr/share/doc/sounio/loom`.

Every stable file and symlink is replaced through a same-directory temporary
name. The host socket and service are stopped before their stable paths move.
They remain stopped after any failed promotion. Only a complete promotion runs
`daemon-reload` and enables the root-owned socket.

## Modes

`--staging-root ABSOLUTE_PATH` builds a disposable package tree without root or
systemd. It is test evidence for hashes, layout, idempotence, and tamper refusal;
it is explicitly `STAGING_ONLY` and never activation evidence.

`--host-install` accepts no destination override. It requires real root identity,
PID 1 systemd, `/run/systemd/system`, and `systemctl`. Thus passwordless `sudo`
inside the current Pod still cannot promote the broker: the service-manager
boundary is absent.

## Live Probe

The broker's `--probe-live` mode is a root-only client for the already installed
socket. It verifies the socket and parent-directory ownership, connects without
a shell, verifies a root host endpoint through `SO_PEERCRED`, then performs five
requests:

- `STATUS` must return `READY` and all six frozen manifest/executable hashes;
- `ADMIT 9029 3` must return a Sounio-bound `DENY424` decision receipt without
  opening material invocation;
- `LAUNCH sabotage` must be refused as bootstrap-closed;
- `RECYCLE sabotage` must be refused as bootstrap-closed;
- an unknown request must be refused.

This probe cannot submit a caller-selected request. It deliberately has no path
that can open execution.

## Host Gate

`scripts/ci/sounio_loom_kernel_principal_broker_host_gate.sh` has two outcomes:

- exit `77`, `HOST_GATE_UNAVAILABLE`, when the real systemd/root boundary is
  unavailable;
- exit `0`, `HOST_ACTIVATION_PASS`, only after installed hashes, permissions,
  units, socket activation, service cgroup, root peer, and the live closed-
  operation probe all pass.

Even `HOST_ACTIVATION_PASS` reports `material_broker=false`,
`material_capsule=false`, and `material_invocation=false`. Namespace creation,
disjoint UID/GID allocation, irreversible privilege drop, pidfd custody, crash
attacks, the affirmative extinction triple, and Sounio actions `9026`, `9027`,
`9028`, and `9029` returning material `ALLOW` remain later gates.

## Negative Controls

The installation selftest must prove:

1. two staging promotions resolve to the same immutable release;
2. all three installed manifests and Sounio executables equal the frozen hashes;
3. installed roles remain `SEMANTIC_AUTHORITY` and `MATERIAL_PARITY`;
4. one-byte drift in any semantic release refuses instead of being overwritten;
5. permission drift in any authority executable refuses;
6. a direct non-root live probe refuses before socket access;
7. host installation refuses in the current non-systemd Pod, including through
   available passwordless `sudo`;
8. the host gate reports unavailable rather than success in that Pod;
9. `LAUNCH` and `RECYCLE` remain closed in every receipt;
10. offline `STATUS` without all three frozen authorities remains denied;
11. the live `ADMIT` probe returns Sounio `DENY424` and never promotes a
    decision receipt to execution authority.

## Nonclaims

- Packaging is not kernel-principal isolation.
- Socket activation is not a namespace or cgroup receipt.
- A root service is not by itself an unforgeable lane principal.
- A staging tree is not a host installation.
- A host activation pass is not a material broker pass.
- A host activation pass is not a material principal-capsule pass.
- General Exec/Bash, commit, and CI attachment remain refused.

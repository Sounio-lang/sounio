# LOOM Host Promotion Capsule V1

Status: implemented material promotion boundary; semantic stage remains
`SEMANTICS_FROZEN`.

## Purpose

The host promotion capsule moves one already-built LOOM kernel-principal broker
release from its source worktree to a real Linux/systemd host without mounting
the workspace PVC into a second Pod and without rebuilding expected results on
the host.

The capsule solves a specific authority problem. A host-local compiler,
checkout, dependency version, or packaging script must not be able to create a
new interpretation of actions `9027`, `9028`, or `9029` during deployment. The
host may verify and install the exact frozen bytes; it may not choose their
meaning.

## Authority

```text
Sounio actions 9027 + 9028 + 9029  = SEMANTIC_AUTHORITY
C++20 broker                        = MATERIAL_PARITY, transitory
OCaml kernels                       = OPERATIONAL_REALIZATION
Bash + GNU tar                      = MECHANICAL_PACKAGING only
Kubernetes hostPID + nsenter        = TRANSPORT only
external LLMs                       = REVIEW_ONLY
```

The Bash programs in this path do not compute an expected semantic result.
They compare hashes, modes, declared roles, fixed protocol selftests, and host
service state. Replacing the Sounio authorities with a Bash, Python, Rust, Node,
Ruby, `awk`, or `bc` oracle is outside the schema and is refused before host
activation.

## Order

The capsule can exist only after the source installer has rebuilt and verified
the frozen Sounio authorities:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> host promotion capsule
```

Building or installing a capsule does not open `PARITY_OPEN` or `CLAIM_READY`.
It also does not open `LAUNCH`, `RECYCLE`, material invocation, or hostile
same-UID peer isolation.

## Capsule

`build_loom_host_promotion_capsule.sh` first invokes the existing staging
installer. That installer rebuilds the three Sounio authorities, verifies their
frozen manifests and causal gates, builds the transitory broker, and emits one
immutable release identity.

The capsule contains:

- the exact staged root filesystem, excluding the stable symlink;
- a strict file, directory, mode, and SHA-256 inventory;
- the source commit and explicit clean/dirty source state;
- the immutable release and install-receipt identity;
- the content-bound host promoter and host activation gate;
- explicit negative claims for every unopened boundary.

The archive is deterministic for identical source bytes. Its SHA-256 is an
out-of-band argument to promotion. An internal inventory catches payload drift
even when a sabotage control deliberately recomputes the outer archive hash.

Dirty-source capsules may be constructed only by an explicit test override.
They are permanently marked `DIRTY_UNPROMOTABLE` and cannot reach host preflight
or installation.

## Promotion

`promote_loom_host_capsule.sh` has three modes:

- `verify`: archive and offline executable verification, with no host claim;
- `preflight`: the same checks plus root, PID 1 systemd, and service-manager
  boundary checks;
- `promote`: immutable release installation, atomic stable-file replacement,
  socket activation, and the live host gate.

The promoter verifies the expected archive hash before extraction. It rejects
absolute paths, traversal, links, devices, undeclared metadata, undeclared
payload entries, wrong modes, wrong hashes, role laundering, dirty source,
opened claims, and an unbound promoter or host gate. The only stable symlink is
created after verification and must target the capsule's release identity.

If activation or the host gate fails after the transaction begins, the prior
stable files and broker symlink are restored. A newly copied immutable release
may remain unreferenced; it grants no authority and is never silently selected.

## Kubernetes Transport

`run_loom_host_promotion.sh` reuses the Beagle node-ephemeral-governance Pod on
the requested node. It requires exactly one running Pod with both `hostPID` and
`privileged` true. The capsule and promoter are streamed into root-owned host
temporary files, rehashed on the transport side, and executed inside the host
mount, PID, UTS, IPC, and network namespaces.

The workspace PVC is never attached to the transport Pod. This is deliberate:
the current workspace volume is `ReadWriteOncePod`, and deployment must not
stop or remount the live workspace merely to obtain host access.

The transport Pod and `nsenter` are not semantic authorities. Failure,
disappearance, timeout, hash drift, or topology ambiguity leaves the capsule
durable in the source worktree and fails closed before promotion.

## Negative Gate

`sounio_loom_host_promotion_capsule_selftest.sh` proves:

- byte-identical deterministic rebuilds;
- outer archive drift refusal;
- inner payload drift refusal after recomputing the outer hash;
- path traversal refusal before extraction;
- a deliberate Python host-gate oracle is refused before execution;
- removing only that language-boundary rule admits the unchanged witness;
- a dirty-source capsule cannot enter host preflight.

The causal sabotage is important: the Python refusal is attributed to this
promoter rule, not merely to another coincidental failure.

## Nonclaims

V1 is content-addressed, not publicly signed. Its expected hash is trusted from
the invoking control channel. It does not claim protection from a hostile
Kubernetes administrator, host root, compromised source worktree, or a hostile
same-UID peer after launch. Those require separate keys, transparency witnesses,
kernel enforcement, and the still-closed material `InvocationCell` path.

The capsule proves exact promotion and live decision-only admission. It does
not prove a launchable multi-tenant lane.

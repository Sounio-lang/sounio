# LOOM Generation-Pinned Cutover v1

Status: `GARDEN`

Semantic authority: Sounio action 9048, to be created only after this Garden is
committed. Operational realization may later be written in OCaml. Shell may
install immutable bytes and move a selector only after a Sounio decision. No
Python, Rust, disposable language, or LLM result may decide a pin or a cutover.

## Problem

Action 9046 correctly refuses bridge-free activation while a live process
generation is not bound to a native hook capability. Waiting for every old
interactive session to end is safe but prevents the global default from
advancing. Moving `current` or `native-next` dynamically also changes the hook
implementation beneath a session that is already alive.

The missing object is an immutable binding between one causally identified
process generation and one installed runtime generation. Once that binding
exists, a global selector may advance without transporting, restarting, or
killing the live session.

## Identity

A generation key contains all of:

- agent and lane;
- provider session identifier and generation number;
- harness;
- canonical worktree and shared Git directory;
- host and boot identifier;
- PID namespace;
- PID and kernel start tick.

The key digest is computed from an unambiguous length-delimited encoding of
these fields. A filename is never accepted as identity by itself. A record is
valid only when its filename, embedded key digest, embedded identity, source
presence digest, and current kernel observation agree.

## Runtime Binding

A pin names an immutable directory below `sounio-coord-runtime/versions`, its
runtime ID, manifest digest, Loom executable digest, coordination executable
digest, source revision, and the Sounio 9048 semantics and freeze digests.
Symlinks are observations used during admission; no pin stores a symlink as its
runtime target.

For a generation with a matching native capability, the capability producer's
runtime is selected. For a causally live pre-cutover generation without that
capability, the exact old `current` runtime may be selected only by a complete
legacy snapshot taken while:

1. the installation lock and coordination state lock are both held;
2. `current` still resolves to the recorded old runtime;
3. the target candidate is installed but is not yet `current`;
4. the presence record and kernel process identity agree;
5. the entire presence inventory is classified; and
6. the legacy reason is recorded as `pre-cutover-current`.

This legacy rule cannot be used after activation and cannot infer which runtime
created a session. It preserves the last globally authoritative runtime for an
otherwise unbound live generation.

## State Machine

`PREPARE` observes and classifies the complete inventory without writing.
`SEAL` atomically writes every required pin and a cutover-set receipt while the
selectors are unchanged. `RESOLVE` admits exactly one already pinned generation
and returns either `CONTINUE` when the executing runtime is the target or
`FORWARD` when execution must be delegated to the immutable target. `BIRTH`
atomically pins a previously unseen post-cutover generation to the then-current
runtime after kernel identity verification. `CUTOVER_READY` requires a sealed
set, unchanged inventory and selector digests, and successful resolution probes
for every sealed generation.

The required order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

The global selector must not move before `CUTOVER_READY`. A failed selector
move must not consume or weaken the sealed set.

## Fail-Closed Rules

Refuse before provider hook execution when any of these holds:

- policy, Sounio executable, freeze, pin, runtime, or receipt is absent;
- a required file is linked, non-regular, unreadable, malformed, duplicated,
  writable outside the owning user, or hash-divergent;
- generation identity, kernel identity, source presence, or filename drifts;
- a pin is replayed for another session, process generation, worktree, or boot;
- the pinned runtime directory escapes `versions` or its manifest/executables
  drift;
- an existing pin would be overwritten or changed;
- inventory or selector state changes between prepare, seal, and activation;
- a lock cannot be acquired within the bounded timeout;
- forwarding is recursive, crosses more than one hop, or returns no complete
  child result; or
- a Python, Rust, shell-arithmetic, disposable-language, or LLM oracle is
  offered as authority.

Every ALLOW and DENY produces an append-only decision record with the source
and semantics hashes, producer and role, toolchain, hardware, command, identity
digest, runtime digest, reason, and result.

## Negative Tests

The parity realization must deliberately test:

1. a Python oracle attempt before execution;
2. a Rust oracle attempt before execution;
3. missing policy and missing freeze;
4. malformed, linked, and duplicate-field pins;
5. session, generation, PID, start-tick, boot, namespace, and worktree replay;
6. presence replacement after prepare;
7. selector drift after seal;
8. runtime manifest and executable tamper;
9. absent target runtime;
10. pin overwrite and conflicting duplicate pin;
11. state-lock and installer-lock timeout;
12. recursive and second-hop forwarding;
13. incomplete inventory and incomplete decision receipt; and
14. an old PID remaining unchanged across a real global `current` cutover.

## Claim Boundary

Action 9048 may authorize generation binding, one-hop hook dispatch, and a
selector cutover whose sealed observations remain current. It does not define
provider semantics, transport a provider session, kill a process, delete old
runtimes, promote parity output to semantic authority, or relax actions
9045-9047. Old runtimes remain installed while any pin references them.

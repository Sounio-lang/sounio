# GARDEN: LOOM Kernel Peer ActivationCapsule V1

Status: `PREREGISTERED`

## Question

Can LOOM turn the frozen material `ALLOW` for action `9025` into exactly one
activation transition without creating a reusable token, letting OCaml define
expected results, or treating missing state as proof of revocation?

The mandatory order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Action `9031` is the first executable definition of this boundary. No OCaml,
C++, shell, CI, or LLM observation may retrospectively define its semantics or
expected decisions.

## Novel Boundary

An `ActivationCapsule` is an affine transition witness, not a bearer
capability. Its serialized bytes grant nothing. Authority exists only as the
live conjunction of:

1. the exact frozen Sounio action-`9025` material judgment;
2. the exact frozen Sounio action-`9030` `ExecGrantCell` decision;
3. the current boot, BPF epoch, principal, ancestry, and broker custody;
4. an unconsumed in-memory generation owned by one authenticated guardian;
5. a prevalidated proposed state transition and its future terminal receipt.

Copying the capsule, its digest, an action receipt, a pidfd number, a BPF map
key, or an OCaml value transfers no authority. The capsule is an index into a
live, single-writer custody relation that must be re-proven at consumption.

This makes activation a temporal proposition. A valid capsule at tick `t`
does not imply a valid capsule at `t + 1` after boot drift, BPF replacement,
principal reuse, guardian loss, timeout, or first consumption.

## State Machine

The closed state domain is:

- `0 EMPTY`
- `1 SEALED`
- `2 CONSUMED`
- `3 EXTINCT`
- `4 POISONED`

Action `9031` accepts four operations:

1. `SEAL`: `EMPTY -> SEALED` after action `9025` material `ALLOW` and action
   `9030` `ISSUE ALLOW` are bound to the same kernel/principal vector.
2. `CONSUME`: `SEALED -> CONSUMED` after action `9030` `CONSUME ALLOW`, live
   deadline, exact authenticated peer, atomic single-writer exchange, and a
   precommitted activation edge.
3. `EXTINGUISH`: `CONSUMED -> EXTINCT` after effect closure, action `9030`
   terminal closure, kernel-object extinction, and the affirmative absence
   triplet below.
4. `POISON`: `SEALED|CONSUMED -> POISONED` after timeout, policy error,
   guardian loss, kernel/BPF drift, malformed response, or uncertain state.
   Poisoning cannot be reversed or recycled.

No operation permits `EXTINCT -> SEALED`, `POISONED -> SEALED`, a second
`CONSUME`, or generation reuse. A new activation requires a new capsule
generation and a fresh action-`9030` grant.

## Frozen Parent Chain

Every frame binds:

- `tools/loom/kernel_peer_material_judgment_v13.freeze.v1`, whose SHA-256 is
  `f7adafcd1c79364b75ebe48b66999ec2d7b82a12d6b8e45d9c1cc4637a4ca9ca`;
- `tools/loom/kernel_exec_grant_cell_authority.freeze.v1`, whose SHA-256 is
  `8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051`;
- the exact action-`9025` semantic and V13 material manifests already named by
  the material judgment;
- the source, toolchain, executable, fixture bundle, command, hardware, and
  result receipt for action `9031` itself.

Parity or review receipts may compare this chain. They cannot replace either
Sounio parent or promote themselves to semantic authority.

## Kernel Anchor

The capsule binds a complete, nonzero kernel anchor:

- Linux boot ID digest;
- BPF object, program, link, map-set, and policy-epoch digests;
- guardian executable and resident Sounio executable digests;
- broker epoch, lease generation, custody generation, capsule generation,
  grant generation, request sequence, deadline tick, and remaining budget.

The BPF links must be pinned and active, the expected three mediation hooks
must be attached, and the map epoch must equal the capsule epoch. A pathname,
successful loader exit, or stale bpffs entry is not an anchor.

## Principal Anchor

The capsule also binds the authenticated live principal:

- `SO_PEERCRED` PID/UID/GID;
- pidfd and `/proc/<pid>/stat` start tick;
- user, PID, mount, and cgroup namespace identities;
- exact cgroup path digest;
- harness ancestry digest and guardian parent identity;
- command, environment, worktree, and operation digests.

Every identity is compared again before in-memory lookup and before atomic
consumption. Same UID alone is explicitly insufficient.

## Affirmative Absence Triplet

Absence is not represented by a missing file, EOF, timeout, failed lookup, dead
socket, or silent process. `EXTINGUISH` and terminal `POISON` require all three
independent positive statements:

1. `registry_absent`: the custody cell and generation are absent from the
   guardian's live registry after a single-writer removal;
2. `kernel_extinct`: every BPF epoch object, grant reference, pidfd-derived
   authority, barrier, and descendant effect is affirmatively extinct;
3. `replay_refused`: a same-generation replay was attempted against the live
   guardian and returned a bound terminal refusal receipt.

Only the conjunction is `AFFIRMATIVE_ABSENCE`. Each component has its own
receipt digest. Silence is rejected even when the other two statements hold.

## Prewrite Shape Validation

The complete old state, proposed state, parent receipts, identity vector,
kernel anchor, consumption rule, and future terminal obligation are validated
before any mutation. This is the write-time shape boundary analogous to SHACL:
an invalid proposed graph never enters the live registry and a `DENY` burns no
grant.

The final commit is a compare-and-exchange over capsule generation, current
state, request sequence, and parent receipt digest. An OCaml implementation may
perform the exchange only after receiving the exact Sounio decision for the
exact frame.

## Fail-Closed Boundary

Missing policy, missing parent, hash drift, timeout, EOF, extra output,
malformed output, process replacement, boot change, BPF link replacement,
principal drift, request reordering, replay, or uncertainty returns `DENY` and
must not perform the proposed transition. A failure after `SEALED` poisons the
generation; it does not reopen or recycle it.

The guardian must not depend on Python or Rust. Shell may build and compare
frozen bytes mechanically but may not generate expected semantic decisions.

## Decision Order

Action `9031` returns the first applicable code:

- `424`: malformed flag, counter, state, operation, or digest shape;
- `405`: stage before `SEMANTICS_FROZEN`, missing preregistration, or wrong
  frozen parent digest;
- `502`: action `9025`/`9030` parent decision chain incomplete;
- `503`: boot, BPF object/program/link/map/epoch, resident, or guardian anchor
  incomplete;
- `504`: peer, pidfd, start tick, namespace, cgroup, ancestry, worktree,
  command, environment, or operation identity incomplete;
- `505`: illegal transition, nonmonotonic generation, post-write validation,
  or non-atomic proposed mutation;
- `506`: bearer semantics, unauthenticated lookup, multiple writers, replay,
  reusable generation, or filesystem authority admitted;
- `507`: revocation, deadline, poisoning, effect closure, or affirmative
  absence triplet incomplete;
- `508`: non-Sounio producer/expected result, Python/Rust oracle, promoted LLM
  review, or promoted parity receipt;
- `509`: source, semantics, toolchain, hardware, command, parent, or result
  receipt provenance incomplete;
- `510`: causal sabotage suite incomplete;
- `0`: `ALLOW`.

The current integration frame must return `502` until the new operational
bridge binds action `9030` grant consumption to the already-frozen action
`9025` material judgment. This Garden does not manufacture that observation.

## Causal Sabotage

The Sounio executable must include positive fixtures for all four legal
operations and at least nine unchanged negative witnesses. For each witness,
deleting exactly one named Sounio rule must change only that witness from its
specified `DENY` to `ALLOW`:

1. parent decision chain;
2. kernel anchor;
3. principal anchor;
4. prewrite lifecycle;
5. non-bearer single-use custody;
6. affirmative absence and failure revocation;
7. authority separation;
8. provenance binding;
9. sabotage completeness.

The rule-deletion test proves that action `9031`, rather than a parser accident
or unrelated check, causes each refusal.

## Language Roles

- Sounio: `SEMANTIC_AUTHORITY`, first executable semantics and expected results;
- OCaml: future `OPERATIONAL_REALIZATION`, hash verifier, process supervisor,
  in-memory state machine, and atomic transition executor;
- C++20/Linux/BPF: existing transitory `MATERIAL_PARITY` observation layer;
- Lean 4: future `FORMAL_PARITY`;
- Koka: future `EFFECT_PARITY`;
- external LLMs: `REVIEW_ONLY`.

Python and Rust are prohibited. Replacing them with another disposable oracle
is equally prohibited.

## Acceptance Boundary

Semantic freeze may establish:

- `action_9031_semantics_frozen=true`;
- deterministic source-fresh Sounio decisions;
- positive state-machine coverage;
- causal refusal attribution;
- exact frozen parent binding.

It must retain:

- `operational_realization=false`;
- `capsule_material=false`;
- `production_activation=false`;
- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.

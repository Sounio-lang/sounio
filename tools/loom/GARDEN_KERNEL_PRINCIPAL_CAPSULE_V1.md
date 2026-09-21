# GARDEN: LOOM Kernel PrincipalCapsule Authority V1

Status: `PREREGISTERED`

## Action

Action `9028` decides whether a kernel-principal launch or crash-recovery
observation may be represented inside LOOM as a `PrincipalCapsule`.

Parent actions:

- action `9026`: the kernel principal is materially isolated;
- action `9027`: the host broker and lease transition are semantically admitted.

The required order is unchanged:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

No C++, OCaml, systemd, shell, or LLM result may define capsule semantics or an
expected decision retrospectively.

## Novel Type Boundary

A lane process is not representable inside LOOM as a bare PID, numeric pidfd,
socket token, or mutable process descriptor. The only admissible representation
is a hash-bound `PrincipalCapsule` whose identity vector was accepted by frozen
Sounio action `9028`.

The capsule is not a bearer capability. It is a certificate naming custody that
remains inside the root-owned broker. Unprivileged callers receive only the
capsule digest and opaque custody generation. Any later operation still requires
an in-memory, single-use `ExecGrant`, authenticated peer and ancestry, and a
fresh broker-side lookup of the live pidfd. Possession of capsule bytes grants
nothing.

This prevents three forms of authority laundering:

1. PID reuse cannot turn an old record into a new principal;
2. serializing a pidfd number cannot transfer descriptor authority;
3. replaying a capsule cannot recreate a consumed grant or broker generation.

## Operations

`operation=1` is `MINT_PREEXEC`. The child exists behind an unreleased pre-exec
barrier in the exact user, PID, and mount namespaces and cgroup intended for the
lane. Actions `9026` and `9027` have admitted the proposed launch, but the
`LAUNCHED` journal edge and barrier release occur only after action `9028`
returns `ALLOW`.

`operation=2` is `REACQUIRE_QUARANTINED`. After broker loss, the new broker may
open a new pidfd and prove that it still references the exact recorded kernel
identity. The lease remains `QUARANTINED`; the recovery capsule permits only
drain, termination, and affirmative-extinction measurement. It cannot resume
execution or restore revoked grants.

## Kernel Identity Vector

Every capsule binds all of the following:

- lease digest, broker epoch, and lease generation;
- host PID and immutable `/proc/<pid>/stat` start time;
- broker-local custody generation and pidfd identity;
- user, PID, and mount namespace inode identities;
- cgroup v2 inode identity and exact delegated cgroup path digest;
- exact UID map and GID map digests;
- executable image and pre-exec contract digests.

Host PID, start time, pidfd identity, namespace identities, cgroup identity,
epoch, generation, and custody generation must all be positive. Epoch,
generation, and custody generation are monotonic. A capsule digest covers the
complete vector; partial vectors are malformed, not provisional.

## Isolation Vector

For `MINT_PREEXEC`:

- user, PID, and mount namespaces differ from the broker's namespaces;
- the lane UID/GID maps are exact, disjoint from the outer account and every
  live lease, and owned by the broker allocator;
- the process is already attached to the exact delegated cgroup;
- memory, process-count, CPU, and descendant containment policies are active;
- the child cannot escape into the broker cgroup or move another process.

For recovery, the current namespace, map, and cgroup digests must equal the
frozen launch vector. A recovery that merely finds a matching PID is refused.

## Privilege Posture

Before capsule minting or recovery admission, the measured principal must have:

- real, effective, saved, and filesystem UID/GID equal to its lane identity;
- no supplementary groups;
- empty permitted, effective, inheritable, bounding, and ambient capability
  sets;
- `no_new_privs=1`;
- the required seccomp filter installed;
- a non-dumpable process posture;
- no setuid/setgid executable path;
- no broker socket, journal, namespace, cgroup, or allocator descriptor.

The evidence describes measured kernel state. A requested configuration or a
successful helper exit is insufficient.

## Pidfd Custody

The root broker must hold a live `O_CLOEXEC` pidfd whose kernel peer identity
matches the host PID and start time in the capsule. The lane and resident
unprivileged kernel must hold no duplicate of that pidfd.

The raw descriptor number is neither serialized nor hashed as authority. The
capsule binds a broker epoch, lease generation, custody generation, and pidfd
identity receipt. On broker loss, the old custody generation is extinct and
all grants referencing it are revoked before any reacquisition attempt.

## Grant Fence

Capsule admission requires:

- the capsule digest covers lease, kernel, isolation, privilege, custody, and
  pre-exec evidence;
- the parent `ExecGrant` names the exact capsule digest, lease generation,
  broker epoch, effect family, command digest, caller peer, and ancestry;
- the grant is single-use and unconsumed at mint time;
- capsule bytes alone are explicitly non-authorizing;
- no LLM review receipt or parity receipt can be promoted into a capsule or
  grant.

## Crash Recovery

`REACQUIRE_QUARANTINED` additionally requires:

- a strictly newer broker epoch and custody generation;
- the prior lease state was `LAUNCHED` or `QUARANTINED` and the current state is
  `QUARANTINED`;
- journal replay, orphan scan, and pidfd reacquisition are complete;
- PID, start time, namespaces, maps, cgroup, executable, and capsule lineage
  equal the frozen launch receipt;
- every prior grant and custody generation is affirmatively revoked;
- recovery output is bound to a new capsule digest and cannot release the
  execution barrier.

Timeout, missing `/proc` evidence, unreadable cgroup state, pidfd failure, or
broker uncertainty remains quarantine. Silence never proves identity or death.

## Decision Order

Action `9028` returns the first applicable code:

- `424`: malformed field, flag, counter, operation, or digest shape;
- `405`: stage is before `SEMANTICS_FROZEN` or parent freeze binding is absent;
- `472`: action `9026`/`9027`, lease state, or launch-order binding incomplete;
- `473`: kernel identity vector incomplete or internally inconsistent;
- `474`: namespace, map, cgroup, or resource containment incomplete;
- `475`: privilege posture incomplete;
- `476`: pidfd custody incomplete or raw descriptor authority exported;
- `477`: capsule/grant fence incomplete or bearer semantics admitted;
- `478`: crash-recovery lineage, revocation, or quarantine rule incomplete;
- `479`: provenance or receipt binding incomplete;
- `480`: required causal sabotage evidence incomplete;
- `0`: `ALLOW`.

The current Pod frame must return `472`, because actions `9026` and `9027` do
not have material `ALLOW` receipts and no host capsule exists.

## Causal Sabotage

The Sounio executable must contain positive `MINT_PREEXEC` and
`REACQUIRE_QUARANTINED` fixtures plus at least nine single-rule controls. Each
control keeps every other field identical and proves that removing only the
named rule changes the decision to `ALLOW`:

1. parent launch order;
2. PID/start-time/pidfd identity agreement;
3. namespace distinctness;
4. exact maps and cgroup containment;
5. privilege posture;
6. broker-only pidfd custody and no raw export;
7. non-bearer single-use grant binding;
8. recovery epoch, quarantine, lineage, and revocation;
9. provenance and receipt binding.

## Freeze Receipt

The freeze manifest must bind:

- this Garden source and commit;
- parent action `9027` manifest and semantic hashes;
- Sounio module, entrypoint, and executable hashes;
- semantic fixture bundle hash;
- toolchain, hardware, command, and result hashes;
- positive mint and recovery decisions;
- current Pod `DENY472` decision;
- all causal sabotage decisions.

## Language Roles

- Sounio: `SEMANTIC_AUTHORITY`;
- OCaml: `OPERATIONAL_KERNEL`, consumer of admitted capsule digests only;
- C++20/Linux/systemd: transitory `MATERIAL_PARITY` collector and custodian;
- Lean 4: future `FORMAL_PARITY`;
- Koka: future `EFFECT_PARITY`;
- external LLMs: `REVIEW_ONLY`.

Python and Rust are forbidden. Shell may assemble, build, install, and compare
receipts mechanically; it cannot supply an expected decision.

## Nonclaims

- This Garden document does not open `LAUNCH` or `RECYCLE`.
- A capsule digest is not an `ExecGrant`.
- A pidfd is not durable across broker loss.
- A matching PID is not process identity.
- A systemd activation pass is not capsule admission.
- No current host material evidence is claimed.
- General Exec/Bash, commit, and CI attachment remain false.

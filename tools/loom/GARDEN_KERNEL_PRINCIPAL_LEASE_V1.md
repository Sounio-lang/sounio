# Garden: Kernel Principal Lease V1

Status: `PREREGISTERED`

## Question

Can LOOM allocate a kernel-distinct lane principal as a crash-safe lease, keep
that lease authoritative across harness, Guardian, broker, and host restarts,
and prevent a dead-looking lane from causing unsafe UID/GID-range reuse?

## Authority Order

This experiment follows the repository language-authority contract:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Sounio action `9027` will be the first executable definition of the lease
lifecycle and its expected decisions. The frozen action `9026` certificate is
its parent. A temporary C++ host broker may later materialize allocations and
kernel receipts, while OCaml may transport those receipts into the resident
LOOM kernel. Neither may define the lifecycle or expected result. Python and
Rust are forbidden. External LLMs are review-only.

## Hypothesis

A kernel principal is not a durable lane identity until its allocation is an
expiring, generation-fenced lease whose transitions are decided before their
effects. A valid principal-lease certificate requires all of:

1. a host-owned broker launched by the host service manager, with a root-owned
   private socket and no setuid or user-invokable allocation entrypoint;
2. a single-writer allocation lock, durable journal, monotonic broker epoch,
   unique lease generation, broker-selected IDs, and pairwise-disjoint UID/GID
   ranges;
3. the only launch lifecycle is
   `FREE -> RESERVED -> MAPPED -> LAUNCHED`;
4. the only release lifecycle is
   `LAUNCHED -> DRAINING -> QUARANTINED -> FREE`, with crash recovery allowed
   to enter `QUARANTINED` but never to skip it;
5. mapping, namespace, cgroup, and pidfd receipts plus a frozen action `9026`
   `ALLOW` exist before the lane harness can execute;
6. after mapping, the broker irreversibly drops privilege, closes every broker
   control descriptor, installs `no_new_privs`, exposes no setuid binary inside
   the lane, and cannot regain host privilege through `execve`;
7. broker, Guardian, or host-observer loss fences the old generation, revokes
   grants, materializes incomplete outcomes, scans orphans, and forces the
   lease into quarantine;
8. a previously used range returns to `FREE` only after the affirmative
   extinction triple is receipt-bound:
   - **process extinction**: all tracked pidfds are dead and the cgroup is
     affirmatively observed empty and removed;
   - **namespace extinction**: all tracked user, PID, and mount namespace
     identities are affirmatively observed dead;
   - **authority extinction**: all execution grants are revoked and the old
     broker epoch and lease generation are retired;
9. silence is not extinction: missing heartbeat, dead pane, timeout, absent
   process listing, failed observer, or broker restart cannot satisfy any
   member of the extinction triple;
10. fresh ranges carry an affirmative never-used receipt; reused ranges carry
    the full extinction triple and a quarantine receipt;
11. all decisions are hash-bound to the frozen `9026` parent, lease journal,
    allocation, lifecycle trace, extinction receipts, Sounio source, frozen
    semantics, toolchain, hardware, command, and result;
12. independent single-rule sabotages prove that the Sounio rule under test is
    what refuses each unsafe lease.

The lease is a kernel-enforced identity lifecycle, not a database label. A
lane name, process listing, tmux pane, heartbeat, socket path, or durable bus
message may help locate evidence but cannot release or reuse a principal.

## Lifecycle States

The canonical states are numeric and closed:

| State | Code | Meaning |
| --- | ---: | --- |
| `FREE` | `0` | range is eligible for a new reservation |
| `RESERVED` | `1` | allocation is journaled and generation-fenced |
| `MAPPED` | `2` | exact UID/GID maps and containment objects exist |
| `LAUNCHED` | `3` | authenticated harness is executing |
| `DRAINING` | `4` | new grants are refused and shutdown is in progress |
| `QUARANTINED` | `5` | reuse is forbidden pending affirmative extinction |

There is no `DEAD` state inferred from absence. A lease whose observer is
uncertain is `QUARANTINED`.

## Operations And Frame

Action `9027` accepts a fixed-width decimal frame for one of two operations:

- operation `1`, `LAUNCH`: validates the path from `FREE` to `LAUNCHED`;
- operation `2`, `RECYCLE`: validates the path from `LAUNCHED` or
  `QUARANTINED` to `FREE`.

The frame contains:

- action, stage, frozen parent, operation, start state, and end state;
- host ownership, service-manager supervision, root-owned socket,
  non-user-invokability, policy-only broker, and frozen-policy binding;
- allocator lock, durable journal, monotonic epoch, disjoint ranges,
  broker-selected IDs, unique generation, prior epoch, current epoch, prior
  generation, and current generation;
- all six allowed transition receipts;
- parent `9026` `ALLOW`, exact maps, cgroup, namespaces, pidfd, and launch gate;
- irreversible drop, `no_new_privs`, denied privilege regain, absent in-lane
  setuid path, empty capabilities, and closed broker descriptors;
- crash fail-closure, orphan scan, generation fence, grant revocation,
  incomplete-outcome materialization, and forced quarantine;
- reuse mode, never-used receipt, process extinction, namespace extinction,
  authority extinction, quarantine receipt, and extinction receipt binding;
- sabotage count and required sabotage count;
- nonzero canonical digests for the `9026` manifest, journal, allocation,
  lifecycle trace, extinction triple, source, semantics, toolchain, hardware,
  command, and result.

## Decision Order

The Sounio authority evaluates rules in this order:

1. malformed frame, unknown operation/state, invalid generation ordering, or
   non-boolean flag -> `DENY424`;
2. wrong action/stage or missing frozen `9026` parent -> `DENY405`;
3. host ownership, service-manager supervision, root-owned socket,
   non-user-invokability, policy-only behavior, or frozen-policy binding is
   absent -> `DENY463`;
4. allocator lock, journal, epoch, uniqueness, broker-selected IDs, or range
   disjointness is incomplete -> `DENY464`;
5. requested lifecycle edges or start/end states are invalid -> `DENY465`;
6. parent principal certificate, maps, cgroup, namespaces, pidfd, or pre-exec
   launch gate is incomplete -> `DENY466`;
7. irreversible privilege drop is incomplete or privilege regain remains
   possible -> `DENY467`;
8. crash fencing, orphan scan, grant revocation, outcome materialization, or
   forced quarantine is incomplete -> `DENY468`;
9. a fresh range lacks a never-used receipt, or a reused/released range lacks
   the affirmative extinction triple and quarantine receipt -> `DENY469`;
10. sabotage evidence is incomplete -> `DENY470`;
11. provenance is incomplete -> `DENY471`;
12. otherwise -> `ALLOW`.

The order is load-bearing. In particular, a broker that is user-invokable must
stop at `463`; later journal or extinction claims cannot launder its authority.

## Causal Sabotage Controls

The semantic gate must run an unchanged unsafe witness against a derived
Sounio source in which exactly one refusal rule is removed:

1. remove only the host-broker boundary rule; a user-invokable broker becomes
   `ALLOW`;
2. remove only the allocator disjointness rule; a colliding range becomes
   `ALLOW`;
3. remove only lifecycle validation; a launch that skips reservation and
   mapping becomes `ALLOW`;
4. remove only irreversible-drop validation; a lane able to regain privilege
   becomes `ALLOW`;
5. remove only crash-recovery validation; an unfenced generation with live
   grants becomes `ALLOW`;
6. remove only affirmative-extinction validation; a stale range becomes
   reusable from silence alone;
7. remove only sabotage completeness; an incompletely attacked certificate
   becomes `ALLOW`.

Each control is valid only when the unchanged source returns the named DENY and
the one-line derived source returns ALLOW for the identical witness. Changing
the witness, mocking the return code, or matching output text is not causal.

## Current Material Expectation

The current pod is expected to produce `DENY463` because there is no verified
host-owned principal broker or service-manager boundary. It also lacks a
writable cgroup delegation, subordinate-ID mapping is refused by the kernel,
and the outer account can regain privilege through passwordless `sudo`. Those
later facts remain independent blockers even after a host broker exists.

The material realization must run outside the lane it is constraining. A
setuid helper exposed directly to the lane, a broker launched by the same user,
or a shell wrapper around privileged commands does not satisfy the contract.

## Acceptance Gates

The semantic gate must prove:

- a fully specified fresh `LAUNCH` certificate is `ALLOW`;
- a fully specified quarantined `RECYCLE` certificate is `ALLOW`;
- current material shape reaches `DENY463`;
- broker, allocator, lifecycle, parent principal, privilege-drop,
  crash-recovery, stale-reuse, sabotage, provenance, stage, and malformed
  witnesses reach their exact named decisions;
- all seven causal sabotage controls admit their unchanged unsafe witness;
- deterministic source-built execution uses a pinned Sounio toolchain;
- no Python, Rust, shell, OCaml, C++, or external LLM computes an expected
  result.

A later material gate must additionally kill the broker and harness at every
lifecycle edge, restart the host observer, and prove that no UID/GID range is
reused until the full extinction triple is observed. Only then may it feed an
action `9027` `ALLOW` into execution attachment.

## Nonclaims

- This Garden does not claim that the current pod or host can materialize the
  broker.
- A durable allocation journal alone does not prove kernel isolation.
- An empty process listing, missing PID, absent heartbeat, or dead pane does not
  prove process extinction.
- A process being dead does not prove its namespaces, cgroup, grants, or broker
  generation are extinct.
- Root, kernel compromise, and physical-host compromise remain outside the
  hostile-lane threat model.
- No general Exec/Bash, commit, or CI attachment is authorized.

Claims-Forbidden: silence proves death; broker restart frees leases; an empty process listing proves extinction; the outer user may choose its own subordinate IDs; a setuid helper is a host-owned broker
Assumptions: a host service manager can supervise a narrow broker outside lane privilege and expose kernel lifecycle receipts
Measurements: broker peer credentials, allocation lock and journal, epochs and generations, exact maps, cgroup and namespace identities, pidfds, privilege state, grant revocation, lifecycle transitions, extinction triple, quarantine, receipts
Budget: one Garden, one Sounio action, one deterministic freeze gate, then a separately reviewed host realization
Stop-Rule: any user-invokable allocation, collision, illegal transition, missing parent certificate, privilege regain, unfenced crash, inferred extinction, stale reuse, or noncausal sabotage keeps admission denied
Negative-Witness: user-owned broker, overlapping range, FREE-to-LAUNCHED skip, privilege regain, live grant after broker crash, missing process extinction, missing namespace extinction, missing authority extinction, no quarantine, missing receipt hash
Owner: codex-1/loom-subprocess-membrane-20260828

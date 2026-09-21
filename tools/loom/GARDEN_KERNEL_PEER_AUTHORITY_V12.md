# GARDEN: LOOM Kernel Peer Authority V12

Status: `PREREGISTERED_AFTER_ACTION_9025_DENY451`

## First Phrase

A principal is not isolated because its identifier is hard to guess or because
well-behaved peers cannot attack it. It is isolated only when a hostile peer
with the same kernel UID and an open attack surface is refused by a
receiver-side kernel authority.

## Semantic Lane

```text
Semantic-Lane-ID: loom-kernel-peer-authority-v12
Owner: codex-1/loom-kernel-exec-grant-cell-20260828
Concept-IDs: SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL
Intent-Preserved: authority must be non-bearer, process-bound, causal, and fail-closed
Transformation: DENY451 becomes a typed adversarial same-kuid interference experiment
Types-Changed: none in the public Sounio language
Effects-Changed: inbound process-authority interference becomes an explicit dual of outbound effects
IR-Changed: none
Claims-Introduced: bounded same-kuid noninterference for one frozen Linux peer-operation basis
Claims-Forbidden: arbitrary Linux MAC correctness, root compromise resistance, portability, product activation
Assumptions: trusted root guardian, immutable kernel mediator, Linux host and LSM backend named by receipt
Write-Set: V12 Garden, Sounio plan/freeze, material peer lab, host evidence, and their gates
Read-Set: frozen V11 action judgment, action 9025, host principal and ExecGrant evidence
Positive-Witness: same-kuid hostile peer crosses no named operation while an active receiver-side mediator is installed
Negative-Witness: the identical same-kuid peer completes attacks when only the mediator is removed
Acceptance-Gate: source-fresh Sounio judgment plus root-host adversarial matrix and causal sabotage
Integration-Target: action 9025 same_uid_peer_isolation fact only
Authoritative-Only-If: Sounio executable and freeze predate all V12 material bytes
```

## Trigger

V11 froze forty counterfactual effect vertices and Sounio accepted their
material receipt. Action 9025 still returned:

```text
SOUNIO_EFFECT_CLOSURE_DENY code=451
reason=same-uid-peer-isolation-absent
```

The V11 causal control then removed exactly the peer-isolation truth rule. The
unchanged false receipt reached action 9025 as `ALLOW`. Therefore V12 is not a
general hardening pass. It owns the single fact whose absence caused `DENY451`.

Frozen parents:

- V11 material judgment manifest:
  `tools/loom/process_witness_effect_material_judgment_v11.freeze.v1`;
- V11 manifest SHA-256:
  `f227cca70aa30351517403e13f60143c683bb86d445320661d68c08317c81b89`;
- V11 judgment evidence SHA-256:
  `4aa5704fe529ee93c88992a630976395b49a28ed13189af9d7a07aeb7ecc4c64`;
- action 9025 manifest SHA-256:
  `c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91`.

## Three Different Predicates

V12 forbids collapsing these predicates:

```text
KUID_EPOCH_EXCLUSIVE
    no second live process occupies the leased kernel-UID epoch

SAME_KUID_CALLER_CONFINED
    a peer has the same kernel UID but its own cooperative seccomp surface
    omits attacks

SAME_KUID_RECEIVER_MEDIATED
    a hostile peer has the same kernel UID and open attack syscalls, while a
    receiver-side kernel authority refuses the attacks using an unforgeable
    principal label
```

`KUID_EPOCH_EXCLUSIVE` is a valuable operational invariant, but it is not
evidence that two same-kuid peers are isolated. `SAME_KUID_CALLER_CONFINED` is
also insufficient: an ambient process with that UID may not share the caller's
seccomp filter. Only `SAME_KUID_RECEIVER_MEDIATED` may set the existing action
9025 field `same_uid_peer_isolation=1`.

Unique DynamicUser identities, empty `/proc`, PID secrecy, a root-owned cgroup,
`PR_SET_DUMPABLE=0`, and an absent pidfd may reduce exposure. None may be
laundered into the receiver-mediated predicate.

## Kernel Principal Epoch

Every V12 vertex binds a principal epoch:

```text
PrincipalEpoch = (
    host_boot_id,
    generation,
    target_pidfd_identity,
    target_start_tick,
    target_kernel_uid,
    attacker_pidfd_identity,
    attacker_start_tick,
    attacker_kernel_uid,
    target_cgroup_identity,
    attacker_cgroup_identity,
    target_executable_sha256,
    attacker_executable_sha256,
    mediator_backend,
    mediator_policy_sha256
)
```

The target and attacker must have different pidfds, start ticks, cgroups, and
process identities. For the decisive treatment and sabotage they must have the
same real, effective, saved, and filesystem kernel UIDs. The guardian must
observe those credentials from outside both principals.

The epoch begins only after target, attacker, policy, and pidfds are jointly
ready. It ends only after both processes, both cgroups, the mediator policy,
all pidfds, and every scratch object are affirmatively extinct. UID reuse
before extinction invalidates the experiment.

## Receiver-Side Authority

The mediator must run at a kernel decision point whose refusal does not depend
on the hostile caller voluntarily carrying a filter. Admissible backends are
preregistered by properties, not by product name:

1. the decision sees target and caller kernel identities;
2. policy identity is immutable for the epoch;
3. an unprivileged peer cannot detach, replace, or relabel the policy;
4. refusal occurs before target state or authority changes;
5. guardian death or policy loss fails closed;
6. the backend covers every operation in the frozen basis;
7. the receipt names backend, kernel configuration, toolchain, command, and
   hardware.

An LSM policy may satisfy these properties. A caller-side seccomp filter,
userspace RPC check, PID secrecy, pathname permission, or bearer token cannot.
V12 does not choose a backend until the Sounio property matrix is frozen.

## Adversarial Operation Basis

The hostile peer must possess callable implementations for every operation.
An `ENOSYS`, missing symbol, denied caller-side seccomp rule, absent pidfd, or
unreachable PID is `EXPERIMENT_UNAVAILABLE`, not isolation.

| Family | Hostile operation | Required open-vertex witness |
|---|---|---|
| asynchronous termination | `kill(SIGTERM)` | target termination observed |
| thread-directed termination | `tgkill(SIGTERM)` | target thread termination observed |
| queued signal | `rt_sigqueueinfo` | target signal payload observed |
| descriptor-directed signal | `pidfd_send_signal` | target termination observed |
| tracing | `ptrace(PTRACE_ATTACH)` | attach and detach observed |
| cross-process memory | `process_vm_readv` | frozen target canary bytes read |
| proc memory | open/read `/proc/<pid>/mem` | frozen target canary bytes read |
| descriptor exfiltration | `pidfd_getfd` | target descriptor duplicated and closed |
| resource mutation | `prlimit64` on target | limit changed and restored |
| process advisory mutation | `process_madvise` | target range advisory call completes |

The basis is bounded and Linux-specific. It is not a claim that the kernel has
only ten peer operations. A backend that cannot mediate one row cannot produce
the positive V12 receipt.

## Counterfactual Matrix

V12 freezes five principal twins before native bytes exist:

| Vertex | KUID relation | Receiver mediator | Caller confinement | Expected |
|---|---|---|---|---|
| `TREATMENT` | same | active | open attack surface | every operation `REFUSED_BEFORE_EFFECT` |
| `MEDIATOR_REMOVED` | same | absent | open attack surface | named attacks `EFFECT_COMPLETED` |
| `DISTINCT_KUID_CONTROL` | distinct | absent | open attack surface | credential refusal, not mediator evidence |
| `CALLER_SECCOMP_CONTROL` | same | absent | attacks removed in caller | unavailable, not receiver evidence |
| `DUMPABLE_ONLY_CONTROL` | same | absent | open attack surface | memory rows may refuse; signal row completes |

The treatment and `MEDIATOR_REMOVED` vertices are the causal pair. They must
bind the same principal epoch fields except mediator presence and policy hash.
The other three vertices diagnose common laundering routes.

The positive claim requires:

```text
same_kuid_pair_observed=true
attacker_syscalls_open=true
receiver_mediator_active=true
all_operations_refused_before_effect=true
mediator_removed_completes_named_effects=true
distinct_kuid_not_counted_as_same_uid_proof=true
caller_seccomp_not_counted_as_receiver_proof=true
dumpable_only_not_counted_as_complete=true
```

## Attack Completion and Refusal

A syscall error alone is not enough. Each row has a typed target-side witness.

```text
REFUSED_BEFORE_EFFECT(
    operation,
    kernel_decision,
    errno,
    target_state_unchanged_sha256
)

EFFECT_COMPLETED(
    operation,
    target_effect_kind,
    target_effect_sha256,
    teardown_sha256
)
```

`CROSSED_NAMED_RULE` and `EXPERIMENT_UNAVAILABLE` retain their V11 meanings and
cannot count as positive coverage. The guardian records target state both
before and after each attempt. An operation returning success without the
typed target-side effect is not a completion.

## Provenance and Causal Binding

Every operation vertex carries:

```text
invariant_sha256 = hash(principal epoch excluding mediator delta)
delta_sha256     = hash(exact mediator and control assignment)
attempt_sha256   = hash(caller syscall, arguments, and return)
target_sha256    = hash(receiver-side state transition or nontransition)
extinction_sha256 = hash(epoch teardown)
```

The final certificate is invalid if a peer credential, executable, cgroup,
pidfd identity, syscall argument, timeout, observer, or target canary differs
between treatment and mediator-removed twins.

## Mandatory Order

```text
GARDEN_V12
-> SOUNIO_EXECUTABLE_V12
-> SEMANTICS_FROZEN_V12
-> BACKEND_DISCOVERY
-> MATERIAL_PEER_MATRIX
-> SOUNIO_JUDGMENT_V12
-> ACTION_9025_REEVALUATION
```

Backend discovery may report unavailable only after the semantic matrix is
frozen. It cannot rewrite the matrix around whatever the host happens to
support.

## Acceptance

`same_uid_peer_isolation=true` requires all of the following:

1. the attacker and target share all four kernel UID slots;
2. all ten hostile operations are callable in the mediator-removed twin;
3. each preregistered completion has its typed target-side witness;
4. the treatment refuses all ten before target effect;
5. target state hashes remain unchanged in treatment;
6. treatment and sabotage invariants are identical;
7. only mediator state differs in the causal pair;
8. policy loss, guardian loss, timeout, and malformed receipt fail closed;
9. all processes, pidfds, cgroups, policies, and scratch objects become
   affirmatively extinct;
10. a Sounio authority transition consumes the complete certificate;
11. the unchanged frozen action 9025 returns `ALLOW`;
12. removing the Sounio peer-truth rule causes the same false peer receipt to
    reach `ALLOW`, proving that the rule is causal.

Any missing row preserves `DENY451`.

## Stop Rules

Stop and freeze a negative result when:

- no admissible receiver-side kernel mediator is available;
- the backend cannot mediate one frozen operation;
- the open same-kuid twin cannot complete its typed effect;
- the treatment relies on caller cooperation;
- unique UID allocation is proposed as same-UID evidence;
- PID, pidfd, cgroup, credential, executable, canary, or timeout invariants
  drift;
- a kernel refusal occurs after target state changed;
- root or an ambient privileged process is silently removed from the declared
  threat model;
- shell, C++, OCaml, an LLM, or host behavior chooses the expected result.

The correct response to an unavailable mediator is a frozen V12 negative
receipt, not an `ALLOW` synthesized from distinct-UID evidence.

## Novelty Boundary

V11 attached a counterfactual kernel cut set to each outbound effect family.
V12 introduces the dual judgment: receiver-side authority over hostile peer
effects. The proposed certificate combines a conventional effect row with a
principal relation, a receiver-side kernel decision, a causal mediator twin,
and affirmative extinction.

The bounded research claim is:

```text
AuthorityJudgment<P, E> =
    OutboundEffectClosure<P, E>
    x InboundPeerNoninterference<P, E>
    x Provenance
    x Extinction
```

This is not yet a literature-priority claim. It is an executable PL/CS novelty
candidate whose falsifiers and evidence boundary are fixed before the
experiment.

## Nonclaims

- V12 does not protect against a compromised trusted root guardian.
- V12 does not prove arbitrary Linux LSM correctness.
- Ten operations are a bounded basis, not a complete kernel attack taxonomy.
- Distinct DynamicUser UIDs do not prove same-UID isolation.
- Caller seccomp does not prove receiver-side authority.
- Successful backend discovery does not prove a material treatment.
- A passing material matrix does not activate launch, Exec/Bash, write,
  commit, CI, recycle, parity, or claim-ready surfaces.
- Python and Rust remain forbidden.

## Current Boundary

```text
material_hypercube=true
material_coverage=false
same_uid_peer_isolation=false
complete_effects=false
material_execution=false
action_9025_judged=true
action_9025_decision=DENY451
production_activation=false
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
```

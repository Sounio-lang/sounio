# GARDEN: LOOM Kernel Peer Authority V13

Status: `PREREGISTERED_AFTER_V12_MATERIAL_FALSIFICATION`

## First phrase

A counterfactual control is part of the semantics, not decorative evidence.
When the kernel falsifies one frozen control prediction, the authority must
return to Garden and Sounio before producing a replacement material matrix.

## Semantic lane

```text
Semantic-Lane-ID: loom-kernel-peer-authority-v13
Owner: codex-1/loom-kernel-exec-grant-cell-20260828
Concept-IDs: SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL
Intent-Preserved: receiver-mediated same-kuid noninterference remains the only fact owned by this lane
Transformation: repair the dumpable-only control after a frozen material counterexample
Types-Changed: none in the public Sounio language
Effects-Changed: none
IR-Changed: none
Claims-Introduced: corrected five-vertex Linux control signature
Claims-Forbidden: retroactive V12 repair, general Linux MAC correctness, product activation
Assumptions: trusted root guardian, immutable BPF LSM mediator, named Linux kernel and toolchain
Write-Set: V13 Garden, Sounio executable/freeze, corrected material controls, judgment
Read-Set: V12 Sounio freeze, V12 decisive-pair freeze, V12 falsification freeze
Positive-Witness: all fifty corrected V13 observations match their preregistered typed outcomes
Negative-Witness: any observation, hash binding, control signature, or extinction fact diverges
Acceptance-Gate: source-fresh Sounio freeze predates every V13 material control byte
Integration-Target: action 9025 same_uid_peer_isolation fact only
Authoritative-Only-If: the V12 counterexample remains immutable and linked by hash
```

## Frozen falsifier

The V12 Sounio plan predicted operation 9, `prlimit64`, would be
`REFUSED_BEFORE_EFFECT` when the target alone had `PR_SET_DUMPABLE=0`.

The isolated KVM material vertex held constant:

- all four attacker and target kernel UID slots at `61234`;
- the same user namespace and minimal `CAP_SYS_NICE` capability;
- distinct processes, pidfds, start ticks, and cgroups;
- `Seccomp: 0`, no LOOM mediator, and target `dumpable=0`.

The attacker changed, observed, and restored the target limit. The typed result
was `EFFECT_COMPLETED/LIMIT_CHANGED_RESTORED`. The source-bound falsification
manifest is:

```text
tools/loom/kernel_peer_dumpable_prlimit_falsification_v12.freeze.v1
sha256=d4b3cdc1dfc6c139538cffecddca60fe34498908b38a2476a7beba8e7e60db7e
```

V13 does not rewrite V12. V12 remains an immutable, failed hypothesis.

## Corrected control signature

The decisive vertices are unchanged:

| Vertex | Expected observations |
|---|---:|
| `TREATMENT` | 10 refused by the active receiver mediator |
| `MEDIATOR_REMOVED` | 10 named effects completed |
| `DISTINCT_KUID_CONTROL` | 10 credential refusals |
| `CALLER_SECCOMP_CONTROL` | 10 experiments unavailable |

`DUMPABLE_ONLY_CONTROL` is corrected operation by operation:

| Operation | Expected | Witness |
|---:|---|---|
| 1 `kill(SIGTERM)` | `EFFECT_COMPLETED` | target terminated |
| 2 `tgkill(SIGTERM)` | `EFFECT_COMPLETED` | target thread terminated |
| 3 `rt_sigqueueinfo` | `EFFECT_COMPLETED` | signal payload observed |
| 4 `pidfd_send_signal` | `EFFECT_COMPLETED` | target terminated |
| 5 `ptrace(PTRACE_ATTACH)` | `REFUSED_BEFORE_EFFECT` | target unchanged |
| 6 `process_vm_readv` | `REFUSED_BEFORE_EFFECT` | target unchanged |
| 7 `/proc/<pid>/mem` | `REFUSED_BEFORE_EFFECT` | target unchanged |
| 8 `pidfd_getfd` | `REFUSED_BEFORE_EFFECT` | target unchanged |
| 9 `prlimit64` | `EFFECT_COMPLETED` | limit changed and restored |
| 10 `process_madvise` | `REFUSED_BEFORE_EFFECT` | target unchanged |

The full V13 totals are frozen as:

```text
observations=50
refused=25
completed=15
unavailable=10
crossed=0
dumpable_completed=5
dumpable_refused=5
```

## Causal and control roles

Only `TREATMENT + MEDIATOR_REMOVED` is the decisive same-process causal pair.
The other vertices diagnose laundering routes and cannot establish isolation:

- distinct UID refusal is credential separation, not same-UID authority;
- caller seccomp makes the experiment unavailable, not receiver-mediated;
- dumpability is partial and leaves all four signal rows plus `prlimit64` open.

Every observation still binds `invariant_sha256`, `delta_sha256`,
`attempt_sha256`, `target_sha256`, and `extinction_sha256`. Syscall return codes
without a typed target witness are inadmissible.

## Sabotage twins

V13 requires five named causal challenges:

1. remove the mediator from `TREATMENT`; all ten named effects must complete;
2. install the mediator in `MEDIATOR_REMOVED`; all ten must refuse;
3. collapse `DISTINCT_KUID_CONTROL` to the same UID; credential-only refusal
   must disappear;
4. open the caller filter in `CALLER_SECCOMP_CONTROL`; unavailability must
   disappear;
5. set `dumpable=1` in `DUMPABLE_ONLY_CONTROL`; the five dumpability-dependent
   refusals must complete.

The twins may use separate epochs when the changed kernel state is irreversible,
but every non-delta field and the reason for a fresh epoch must be receipted.

## Mandatory order

```text
GARDEN_V13
-> SOUNIO_EXECUTABLE_V13
-> SEMANTICS_FROZEN_V13
-> MATERIAL_CONTROL_MATRIX_V13
-> MATERIAL_SABOTAGE_TWINS_V13
-> SOUNIO_JUDGMENT_V13
-> ACTION_9025_REEVALUATION
```

No V13 C, C++, BPF, OCaml, shell-generated measurement, or host result may
choose or revise the expected observations.

## Acceptance and stop rules

`same_uid_peer_isolation=true` remains forbidden unless:

- the V12 decisive pairs remain source-bound and passing;
- all thirty non-decisive control observations match the V13 signature;
- all five sabotage twins cross only their named rule;
- target-side effect/refusal witnesses and all five hashes are complete;
- policy, process, pidfd, cgroup, and scratch extinction are affirmative;
- a new Sounio judgment consumes the complete material certificate;
- unchanged action 9025 returns `ALLOW`;
- the Sounio peer-truth sabotage admits the false receipt, proving causality.

Any mismatch freezes another negative result and returns to Garden. Product,
launch, recycle, Exec/Bash, write, commit, CI, and parity surfaces remain closed.
Python and Rust remain forbidden.

## Novelty boundary

V13 adds a falsification-preserving semantic lineage:

```text
Hypothesis_v12
  x MaterialCounterexample_v12
  -> RevisedHypothesis_v13
```

The failed executable semantics is retained as a first-class ancestor of the
replacement semantics. This turns model revision itself into a hash-bound,
machine-checkable scientific object. It is a PL/CS novelty candidate, not yet
a literature-priority claim.

## Current boundary

```text
garden_v13=true
sounio_executable_v13=false
semantics_frozen_v13=false
v12_hypothesis_falsified=true
controls_executed=false
material_peer_matrix=false
same_uid_peer_isolation=false
action_9025_decision=DENY451
material_coverage=false
complete_effects=false
material_execution=false
production_activation=false
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
```

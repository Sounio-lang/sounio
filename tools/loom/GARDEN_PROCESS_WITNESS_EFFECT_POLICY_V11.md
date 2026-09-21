# GARDEN: LOOM Counterfactual Effect Hypercube V11

Status: `PREREGISTERED_AFTER_V10_ROOT_TREATMENT`

## First Phrase

An effect is not materially closed because one guard refused once. It is closed
only when the refusal survives the enforcement hypercube and the open vertex
actually performs the forbidden effect.

## Semantic Lane

```text
Semantic-Lane-ID: loom-process-witness-effect-hypercube-v11
Owner: codex-1/loom-kernel-exec-grant-cell-20260828
Concept-IDs: SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL
Intent-Preserved: absence must be affirmative, causal, provenance-bound, and narrower than its witness
Transformation: binary treatment/sabotage pairs become typed counterfactual mechanism hypercubes
Types-Changed: none in the public Sounio language
Effects-Changed: material evidence for action 9025 only
IR-Changed: none
Claims-Introduced: bounded minimal-cut-set evidence for one frozen ProcessWitness effect basis
Claims-Forbidden: Linux effect completeness, portability, arbitrary command safety, product activation
Assumptions: Linux x86-64, kernel and systemd named by the receipt, monotone defensive mechanisms
Write-Set: V11 Garden, Sounio plan, freeze, native lab, host evidence, and their gates
Read-Set: frozen action 9025, V10 policy, V10 root evidence, ProcessWitness host runtime
Positive-Witness: every preregistered treatment vertex refuses and every open vertex completes its named effect witness
Negative-Witness: one mechanism, vertex, invariant, completion witness, or extinction receipt missing
Acceptance-Gate: source-fresh Sounio judgment plus root-host hypercube and freeze gates
Integration-Target: action 9025 material coverage only
Authoritative-Only-If: Sounio executable and freeze predate all V11 native bytes
```

## Trigger

V10 proved the identity-typed root treatment on `t560-proxmox` and executed all
seven negative bootstrap controls. Its evidence is:

- path:
  `tools/loom/evidence/loom-process-witness-effect-root-v10-host-attempt-v1-20260828.txt`;
- SHA-256:
  `96bea5a8306d61ed4528b5b29f92493c98fe6e95c1c6c8ee28930b0f5c2b0ca5`;
- materializer commit:
  `bfc868d03f9a9230a5efa8446ddd91a498d6b78e`;
- treatment: `CAPSULE_EMPTY_BIND`;
- typed `/proc` controls: `DENY453`, `DENY454`, `DENY455`, `DENY456`;
- root treatment: true;
- effect-family material coverage: false.

The V10 Sounio source also preserves one falsified legacy line:

```text
ROOT_SCHEMA ... proc_treatment=absent
```

That line conflicts with the same executable's `PROC_MOUNT_TYPE` row and with
the frozen V10 manifest's `proc_treatment=CAPSULE_EMPTY_BIND`. The V10 bytes and
freeze remain evidence and must not be rewritten. V11 is an explicit semantic
correction: its root schema has exactly one typed `/proc` statement and no
path-only absence statement.

## Why Twelve Binary Twins Are Insufficient

Several effect families have redundant kernel mechanisms. For example, file
creation can be refused by both a read-only VFS mount and seccomp; personality
change can be refused by both `LockPersonality` and the positive syscall
allowlist; `/proc/self/mem` can be unavailable because procfs is absent and
also because `openat` is denied.

Removing one guard may therefore leave the effect refused. That does not show
the removed guard was irrelevant. Conversely, removing two guards together and
observing success does not identify which guard was causal. A binary twin
cannot distinguish independent sufficiency, redundancy, interaction, and an
invalid probe.

V11 treats each family as a monotone Boolean experiment over named enforcement
dimensions. The material result is not one Boolean. It is a provenance-bound
truth table from which Sounio judges minimal denial cut sets.

## Counterfactual Effect Hypercube

For family `f`, probe `p`, and mechanisms `M_f = {m1..mk}`, a vertex is a bit
vector `v in {0,1}^k`, where `1` means the mechanism is active. Every vertex
must retain identical source, semantics, root tree, executable, payload,
arguments, UID/GID class, observer, resource bounds, and timeout. Only the
declared mechanism bits may differ.

Each vertex produces one of four observations:

```text
REFUSED_BEFORE_EFFECT(rule, errno, syscall_result)
CROSSED_NAMED_RULE(effect_completed=false, syscall_result)
EFFECT_COMPLETED(witness_kind, witness_sha256)
EXPERIMENT_UNAVAILABLE(reason)
```

`CROSSED_NAMED_RULE` is not success. A syscall returning something other than
`EPERM` may still fail in another kernel layer. Material coverage requires an
`EFFECT_COMPLETED` witness at every preregistered open vertex. Unavailable is a
closed result and cannot be promoted to coverage.

For defensive bits, V11 requires monotonicity:

- if a vertex refuses, every vertex with a superset of active mechanisms must
  refuse;
- if a vertex completes, every vertex with a subset of active mechanisms must
  complete;
- any violation is `DENY457 nonmonotone-material-effect`.

A minimal denial cut set is a refusing vertex for which disabling any active
mechanism makes the same probe complete. Sounio, not C++, computes and names
the expected minimal cut sets from the frozen table.

## Probe Basis and Mechanism Dimensions

The V11 basis is deliberately bounded. It does not claim to enumerate all
Linux operations. The positive seccomp default-deny row remains the closure
rule for operations outside the basis.

| Family | Probe basis | Mechanism dimensions | Vertices |
|---|---|---|---:|
| 1 executable transition | repeat exact fd3 exec; wrong-fd/flags exec | `FD3_CLOEXEC`, `EXECVEAT_ARGUMENT_FILTER` | 8 |
| 2 process topology | valid `clone3` child creation | `SECCOMP_PROCESS_TOPOLOGY` | 2 |
| 3 filesystem path | create one named file | `VFS_READ_ONLY`, `SECCOMP_MUTATING_OPEN` | 4 |
| 4 descriptor mutation | duplicate fd0 to fd9 | `SECCOMP_DESCRIPTOR_MUTATION` | 2 |
| 5 mapped storage | create writable shared anonymous mapping | `SECCOMP_SHARED_WRITABLE_MMAP` | 2 |
| 6 asynchronous I/O | create one `io_uring` instance | `SECCOMP_IO_URING_SETUP` | 2 |
| 7 network | connect to one hash-bound host endpoint | `PRIVATE_NETWORK`, `SECCOMP_INET_SOCKET` | 4 |
| 8 filesystem Unix socket | connect to one hash-bound Unix endpoint | `UNIX_ENDPOINT_ABSENCE`, `SECCOMP_UNIX_SOCKET` | 4 |
| 9 interprocess communication | create one memfd | `SECCOMP_MEMFD_CREATE` | 2 |
| 10 device/kernel control | change and restore personality | `LOCK_PERSONALITY`, `SECCOMP_PERSONALITY` | 4 |
| 11 process/kernel filesystem | open `/proc/self/mem` read-only | `CAPSULE_EMPTY_BIND`, `SECCOMP_PROC_OPEN` | 4 |
| 12 unknown/future | execute unlisted `getpid` | `POSITIVE_DEFAULT_DENY` | 2 |

There are thirteen probes and forty preregistered vertices. Family 1 has two
probes because close-on-exec and argument filtering govern different bypasses;
collapsing them would hide a missing rule. The network endpoints, Unix endpoint,
created child, created file, mapping, ring, descriptors, memfd, personality
transition, opened proc object, and unlisted syscall return each require their
own positive completion witness and teardown witness.

## Frozen Expected Topology

Before native V11 bytes exist, a Sounio executable must emit all forty vertices
with exact expected observation classes. At minimum:

1. every all-mechanisms-active treatment vertex is
   `REFUSED_BEFORE_EFFECT`;
2. every all-mechanisms-disabled open vertex is `EFFECT_COMPLETED`;
3. every intermediate vertex has an exact Sounio-produced expectation;
4. the family-1 exact-repeat probe identifies `FD3_CLOEXEC` as its minimal cut;
5. the family-1 wrong-argument probe identifies
   `EXECVEAT_ARGUMENT_FILTER` as its minimal cut;
6. the two-dimensional redundant families emit their complete four-vertex
   tables, not a guessed single cause;
7. all single-dimensional families emit a deny/complete pair;
8. the expected monotonicity and minimal-cut-set digests are frozen.

The executable must refuse duplicate vertices, missing vertices, noncanonical
bit order, a completion without a typed witness, or a receipt whose invariant
hash differs inside one probe cube.

## Triple Hash Binding

Every material vertex binds three different objects:

```text
invariant_sha256 = hash(all bytes and facts that must not vary)
delta_sha256     = hash(the exact mechanism-bit assignment)
witness_sha256   = hash(the observed refusal or completed effect)
```

Two vertices are causal peers only if `invariant_sha256` is byte-identical and
their `delta_sha256` differs exactly where the frozen cube permits. The final
family certificate hashes the ordered vertex receipts and the Sounio-produced
minimal-cut-set judgment. This makes accidental multi-variable sabotage
observable rather than a matter of reviewer trust.

## Execution Order

The mandatory order remains:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> MATERIAL_HYPERCUBE
-> ACTION_9025_JUDGMENT
-> CLAIM_READY
```

The current commit may contain only this Garden. The Sounio executable is the
next admissible artifact. Native C++, OCaml, shell transport, host endpoints,
and product attachment bytes must not be written until the Sounio plan is
committed and frozen by hash.

## Acceptance

`material_coverage=true` requires:

1. exactly 13 probes and 40 vertices;
2. all full-treatment vertices refuse before effect;
3. all open vertices complete with typed positive witnesses;
4. every intermediate vertex matches the frozen Sounio expectation;
5. all invariant hashes are stable within their probe cubes;
6. every delta changes only permitted mechanism bits;
7. all cubes are monotone;
8. minimal cut sets match the frozen Sounio judgment;
9. every vertex and endpoint becomes affirmatively extinct before the next;
10. action 9025 consumes the complete receipt and returns its frozen positive
    decision.

`complete_effects=true` and `material_execution=true` may become true only in a
later Sounio authority transition that consumes this evidence. Even then,
product activation remains separately closed.

## Stop Rules

Stop and preserve `DENY447 material-coverage-incomplete` when:

- an open vertex does not complete the named effect;
- an intermediate vertex contradicts its frozen expectation;
- a mechanism cannot be toggled independently;
- the invariant hash changes between causal peers;
- a cube is nonmonotone;
- teardown or extinction is incomplete;
- the host needs a new syscall, endpoint, object, or exception not frozen by
  Sounio;
- shell, C++, OCaml, an LLM, or observed host behavior would determine the
  expected semantic result.

The correct response to a falsified vertex is V12, not an edited expected
receipt.

## Novelty Boundary

The proposed PL/CS novelty is a causal refinement of effect systems. A static
effect family is paired with an executable counterfactual cube whose vertices
identify the minimal kernel cut sets that make the effect unavailable for one
frozen principal. The certificate preserves the distinction between rule
crossing and effect completion, and binds causal peers by invariant, delta, and
witness hashes.

This is narrower than universal sandbox correctness and stronger than a policy
listing, a passing seccomp test, or a binary sabotage. It makes redundant
enforcement, causal interaction, and failed probes first-class evidence carried
by the effect judgment.

## Nonclaims

- V11 does not prove that Linux has only twelve effect families.
- The thirteen probes are a bounded basis, not exhaustive syscall coverage.
- A minimal cut set on one host is not portable without parity evidence.
- `CROSSED_NAMED_RULE` is not `EFFECT_COMPLETED`.
- Kernel configuration, systemd properties, or BPF bytecode alone are not
  actual-instance evidence.
- V11 does not open arbitrary commands, shells, plugins, or model CLIs.
- V11 does not activate LOOM launch, recycle, Exec/Bash, write, commit, CI,
  parity, or claim-ready surfaces.
- Python and Rust remain forbidden.

## Current Boundary

```text
root_treatment=true
bootstrap_sabotage=true
material_hypercube=false
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

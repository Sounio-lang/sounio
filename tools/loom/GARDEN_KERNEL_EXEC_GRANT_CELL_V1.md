# GARDEN: LOOM Kernel ExecGrantCell Authority V1

Status: `PREREGISTERED`

## Question

Can LOOM make one execution grant non-bearer, single-use, crash-extinguishing,
and semantically inseparable from the exact principal, invocation, command, and
outcome it authorizes?

## Authority Order

This experiment follows the mandatory evidence order:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Sounio action `9030` must be the first executable definition and the first
producer of every expected decision. Its frozen parents are:

- action `9029`: the joined `InvocationCell` principal/effect observation;
- action `9021`: pre-execution command authority;
- action `9022`: durable execution-outcome closure.

OCaml may later realize the state machine inside the existing LOOM kernel. The
transitory C++ host broker may retain privileged descriptors and material
receipts. Neither language may define a state, transition, expected result, or
extinction rule retrospectively. Lean 4 and Koka remain future parity roles.
External LLMs are review-only. Python and Rust are forbidden.

## Hypothesis

An opaque handle is only a lookup coordinate. It is never authority by
possession. One operation is executable only when the custodian finds a live
grant and independently authenticates the caller, principal, invocation,
command, generation, deadline, and frozen Sounio decision chain before changing
state.

The resulting `ExecGrantCell` is the conjunction of:

1. a frozen action `9029` cell for the same principal and operation;
2. a frozen action `9021` `ALLOW` for the same command and environment before
   issuance and again before execution;
3. a broker-held grant record that never enters the lane filesystem or durable
   recovery image;
4. peer identity captured by `SO_PEERCRED`, pidfd, start tick, boot identity,
   PID namespace, exact executable, Guardian-owned harness ancestry, worktree,
   cgroup, and kernel-distinct lane principal;
5. a pre-write shape decision for every state-changing request;
6. a single atomic terminal path: consume, close, revoke, or expire;
7. affirmative grant extinction after any terminal path;
8. action `9022` and action `9029` outcome closure before success can be
   promoted beyond the cell.

Copying the handle, cell digest, capsule digest, PID, pidfd number, socket path,
or receipt bytes cannot satisfy this conjunction.

## State Space

The state domain is closed:

| State | Code | Meaning |
| --- | ---: | --- |
| `VACANT` | `0` | no grant has been issued for this generation |
| `ISSUED` | `1` | grant exists only in custodian memory behind a closed barrier |
| `CONSUMING` | `2` | authenticated atomic consumption owns the generation |
| `OUTCOME_PENDING` | `3` | execution ended but outcome closure is not committed |
| `CLOSED` | `4` | complete outcome and affirmative extinction are receipt-bound |
| `REVOKED` | `5` | authority is extinct; success is impossible |
| `POISONED` | `6` | uncertainty exists; only termination and quarantine are allowed |

There is no recoverable serialized `ISSUED` state. Kernel or broker-generation
loss turns every outstanding grant into `REVOKED` or `POISONED`; restart begins
with an empty table and a fresh generation.

## Operations

Action `9030` accepts four operations:

1. `ISSUE`: `VACANT -> ISSUED` after action `9029` `PREPARE_ROOT` and action
   `9021` both return `ALLOW` for the exact proposed observation.
2. `CONSUME`: `ISSUED -> CONSUMING -> OUTCOME_PENDING`. The caller is
   authenticated before handle lookup. The state change is atomic, single
   writer, and irreversible. Action `9021` is re-run on unchanged measurements
   before the barrier can open.
3. `CLOSE`: `OUTCOME_PENDING -> CLOSED` only after action `9022` and action
   `9029` `CLOSE_OUTCOME` return `ALLOW`, the complete process tree is
   quiescent, and grant extinction is affirmatively recorded.
4. `REVOKE`: `ISSUED | CONSUMING | OUTCOME_PENDING -> REVOKED | POISONED` on
   timeout, denial, policy loss, broker loss, Guardian loss, peer drift,
   measurement uncertainty, crash, or incomplete outcome. It cannot open a
   barrier or produce success.

Wrong-peer probes are refused before lookup and therefore cannot burn another
lane's grant. Legitimate consume burns the grant before execution. Replay can
observe only a terminal receipt or a typed missing-generation refusal; it
cannot recreate or mutate the grant.

## Write Shapes

Every state mutation is guarded before the write by a Sounio-defined closed
shape. Validation after mutation is insufficient. The write shape requires:

- known operation, current state, requested next state, and legal edge;
- monotonic broker epoch, lease generation, custody generation, invocation
  generation, grant generation, and event sequence;
- exact nonzero hashes for the parent freezes, capsule, invocation cell,
  command, executable, argv, cwd, root, environment, hardware, peer vector,
  deadline, outcome, journal head, and proposed receipt;
- a finite monotonic deadline and bounded resource budget;
- the same principal, harness ancestry, worktree, cgroup, command, and operation
  on issue, consume, outcome, and terminal receipts;
- a complete old-state receipt and proposed new-state receipt;
- policy availability and an exact Sounio decision before mutation.

An unknown field, unknown state, missing hash, stale generation, illegal edge,
timeout, policy error, or shape error refuses before the grant table, barrier,
process, outcome obligation, or journal is changed.

This is SHACL's useful discipline moved into the executable authority boundary:
the shape protects ingestion, not a later reader. The Sounio rule and its
diagnostic remain one artifact.

## Non-Bearer Peer Binding

The material custodian must derive peer facts from the kernel rather than the
request. It binds all of:

- `SO_PEERCRED` PID, UID, and GID;
- live close-on-exec pidfd and unchanged process start tick;
- boot identity and PID-namespace identity;
- exact running executable path and digest;
- exact Guardian-owned harness PID and start-tick ancestry;
- exact lane UID/GID principal, worktree, cgroup, and namespace vector;
- operation-specific command-line role;
- no `ptrace`, `CAP_SYS_PTRACE`, process injection, or cross-lane descriptor
  inheritance.

Same-UID ancestry alone is not hostile-principal isolation. A material `ALLOW`
requires kernel-distinct lane principals plus the measured anti-injection
posture. Until that exists, current material admission must remain denied.

## Affirmative Extinction

Missing data is `UNKNOWN`, not `EXTINCT`. Grant extinction is a first-class
affirmative fact only when the same terminal receipt binds all three members:

1. **state extinction**: the grant table is observed without the exact handle
   after an atomic terminal transition, and no consume owner remains;
2. **generation extinction**: the broker, lease, custody, invocation, and grant
   generations are retired or terminal and cannot be reconstructed after
   restart;
3. **authority extinction**: the barrier is closed or gone, all duplicated
   descriptors and outstanding grants are revoked, and the exact journal head
   records the terminal reason.

Silence, a missing file, an empty process listing, a dead pane, absent
heartbeat, closed socket, timeout, failed observer, or broker restart satisfies
none of these alone. Incomplete observation produces `POISONED` and quarantine.

Extinction propagates: any descendant conclusion that depended on the live
grant becomes inapplicable. A revoked or extinct grant cannot be recovered from
a capsule, cell, receipt, handle, journal replay, parity proof, or LLM review.

## Outcome Closure

`CLOSE` additionally requires:

- the same grant owns the pending outcome obligation;
- root and descendant termination plus tree quiescence;
- zero open effects and complete terminal effect receipts;
- complete stdout, stderr, exit, signal, timeout, and write-set digests;
- action `9022` `ALLOW` for the observed result;
- action `9029` `CLOSE_OUTCOME` `ALLOW` for the same cell generation;
- journal and receipt hashes committed before obligation removal;
- the affirmative extinction triple.

A crash after execution but before durable closure becomes an explicit
incomplete outcome and `POISONED` or `REVOKED`, never success.

## Decision Order

Action `9030` returns the first applicable code:

- `424`: malformed field, flag, state, operation, counter, or digest shape;
- `405`: stage precedes `SEMANTICS_FROZEN` or a frozen parent is absent;
- `491`: action `9029`, `9021`, or `9022` parent binding is incomplete;
- `492`: grant, capsule, invocation, command, or generation identity is
  incomplete or substitutable;
- `493`: kernel-derived peer, ancestry, principal, cgroup, or anti-injection
  binding is incomplete;
- `494`: pre-write shape, legal transition, monotonicity, or no-mutation-on-deny
  evidence is incomplete;
- `495`: handle non-authority, authenticated-before-lookup, atomic single-use,
  barrier custody, or replay isolation is incomplete;
- `496`: deadline, crash fencing, policy-loss revocation, or fail-closed
  behavior is incomplete;
- `497`: the affirmative state/generation/authority extinction triple is
  incomplete;
- `498`: outcome obligation, tree quiescence, action `9022`, action `9029`, or
  durable close receipt is incomplete;
- `499`: Python, Rust, parity, review, or bearer authority laundering is
  admitted;
- `500`: source, semantics, parents, toolchain, hardware, command, peer,
  transition, journal, result, or receipt provenance is incomplete;
- `501`: causal sabotage evidence is incomplete;
- `0`: `ALLOW`.

The current material frame must return `491`: no material action `9029` cell is
joined to a material action `9021` grant and action `9022` outcome chain.
Independent later blockers include missing kernel-distinct lane principals and
closed launch/coverage gates.

## Causal Sabotage

The Sounio executable must include positive witnesses for `ISSUE`, `CONSUME`,
`CLOSE`, and `REVOKE`, plus eleven single-rule controls. Each control keeps the
unsafe witness byte-for-byte identical and removes exactly one refusal rule:

1. parent action chain;
2. grant/capsule/invocation identity;
3. kernel peer and hostile-lane isolation;
4. write shape and legal transition;
5. non-bearer atomic consumption;
6. deadline and crash revocation;
7. affirmative extinction;
8. outcome closure;
9. prohibited producer and authority laundering;
10. provenance;
11. sabotage completeness.

Each derived Sounio source must change only its named `DENY` to `ALLOW` for the
unchanged witness. Parser failure, changed input, mock output, or a second
language's expected-result table is not a causal control.

## Acceptance Gates

The semantic phase must prove:

- Garden commit predates the executable commit;
- source-built Sounio produces all four positive expected decisions first;
- named negative witnesses reach `491` through `501` and malformed reaches
  `424`;
- eleven causal source sabotages admit their unchanged witnesses;
- two source-fresh builds are byte-identical;
- the freeze manifest binds the exact Garden, parents, source, adapter,
  fixtures, executable, toolchain, hardware, command, results, and decisions;
- Python and Rust oracle attempts are refused before execution;
- `parity_open=false`, `claim_ready=false`, and all product attachment flags
  remain false.

Only after that freeze may the existing OCaml kernel be extended. The OCaml
gate must then prove pre-write validation, wrong-peer non-burning refusal,
single-use consumption, generation loss, crash revocation, affirmative
extinction, and zero semantic expected-result strings in OCaml.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-kernel-exec-grant-cell-20260828
Owner: codex-1/loom-kernel-exec-grant-cell-20260828
Concept-IDs: SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL
Intent-Preserved: possession never grants execution authority and silence never proves extinction
Transformation: join the existing invocation cell and execution authority/outcome chains into one non-bearer atomic grant lifecycle
Types-Changed: add ExecGrantCell state, write shape, peer binding, extinction triple, and decision result
Effects-Changed: issue, consume, close, revoke, expire, and crash recovery become Sounio-decided before mutation
IR-Changed: none
Claims-Introduced: Sounio can distinguish a fully bound grant lifecycle from bearer, replayable, post-validated, or silently absent variants
Claims-Forbidden: a handle is authority; same UID is hostile-principal isolation; missing state proves extinction; current Exec/Bash is attached
Assumptions: a root custodian and resident OCaml kernel can exchange hash-bound decisions without exporting privileged descriptors
Write-Set: Garden, concept contract, registry, Sounio action 9030, adapter, gates, freeze, evidence, then existing OCaml kernel
Read-Set: actions 9021, 9022, 9026, 9027, 9028, 9029 and their frozen manifests
Positive-Witness: complete ISSUE, CONSUME, CLOSE, and REVOKE observations
Negative-Witness: missing parent, substituted cell, wrong peer, invalid transition, bearer consume, unfenced crash, inferred extinction, incomplete outcome, Python oracle, missing provenance, incomplete sabotage
Acceptance-Gate: deterministic Sounio build, exact decisions, eleven causal source sabotages, frozen parents, then separately frozen operational parity
Integration-Target: existing OCaml EXEC_ISSUE/EXEC_CONSUME/EXEC_OUTCOME and root broker, not a second broker
Authoritative-Only-If: Sounio executable and expected results precede every operational or material change
```

## Nonclaims

- This Garden opens no execution, barrier, launch, recycle, commit, or CI path.
- It does not claim the current host has kernel-distinct lane principals.
- It does not claim same-UID hostile-peer isolation from ancestry alone.
- It does not make ordinary Sounio values statically linear or affine.
- It does not prove external novelty or priority.
- It does not authorize a second LOOM broker or a competing execution harness.

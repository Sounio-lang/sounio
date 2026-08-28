# GARDEN: LOOM Kernel InvocationCell Authority V1

Status: `PREREGISTERED`

## Action

Action `9029` decides whether one kernel-principal operation may be represented
inside LOOM as an `InvocationCell`.

An `InvocationCell` is the semantic join of two independently frozen lineages:

- action `9028`: non-bearer `PrincipalCapsule` identity and broker-only pidfd
  custody;
- action `9025`: positive effect-closure authority, which already binds the
  action `9023` subprocess membrane and action `9024` resident transport.

The required order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

No C++, OCaml, shell, hook, LLM, or parity implementation may create the join or
its expected result retrospectively.

## Novel Boundary

Identity admission and effect admission are necessary but not compositional by
default. A valid principal cannot execute an unclosed effect stream, and a
closed effect stream cannot be reassigned to another principal. Action `9029`
makes this non-substitution rule executable.

The cell binds exactly one principal generation, membrane generation, event
sequence, effect, command, worktree, claim scope, deadline, and outcome lineage.
It is not a bearer capability. The root broker retains the `ExecGrant`, pidfd,
and resume barrier. An unprivileged lane may observe a cell digest and sequence,
but possession of those bytes grants no authority and cannot resume an effect.

This is an affine protocol claim about observed broker state, not a claim that
ordinary Sounio values are statically linear or affine. Copying a cell digest is
harmless because the digest is non-authorizing; one-shot consumption remains a
material reference-monitor obligation.

## Operations

Action `9029` has four operations:

1. `PREPARE_ROOT`: bind an action `9028` pre-exec capsule to a fresh action
   `9025` membrane generation while the child remains behind its barrier.
2. `ADMIT_EFFECT`: bind exactly one stopped, pre-effect observation to the same
   cell, monotonic event sequence, actor identity, claim scope, and deadline.
3. `CLOSE_OUTCOME`: close the cell only after the complete process tree is
   quiescent, all effects have terminal receipts, and the outcome chain is
   complete.
4. `ABORT_INCOMPLETE`: irreversibly poison the cell after timeout, policy loss,
   broker loss, measurement uncertainty, or crash. Abort may terminate and
   quarantine; it cannot resume execution, restore a grant, or produce success.

Every operation consumes a fresh broker-held grant generation. Reusing an event
sequence, grant generation, outcome generation, or already closed/poisoned cell
is refused without changing any other cell.

## Semantic Join

The cell binds the exact frozen manifests and semantic hashes for actions `9028`
and `9025`. Both parent decisions must be `ALLOW` for the same proposed material
observation. A decision from one parent cannot stand in for the other, and an
older parent hash cannot be silently upgraded.

The join additionally requires:

- the action `9028` capsule digest, lease generation, broker epoch, custody
  generation, and grant fence;
- the action `9025` closure generation, action `9023` membrane generation, and
  resident policy generation;
- exact equality between the principal named by the stopped actor and the
  principal named by the capsule;
- exact equality between the effect event named by the cell and the event
  submitted to the resident Sounio authority.

A parent timeout, malformed response, missing freeze, hash drift, or non-`ALLOW`
decision refuses before an effect is resumed.

## Command And Scope Binding

`PREPARE_ROOT` binds a canonical digest of executable identity, argv, cwd,
environment, worktree, and requested command. `ADMIT_EFFECT` additionally binds
the canonical target object, operation, current object identity, and exact
coordination claim-scope receipt.

String equality alone is insufficient. The material layer must open or resolve
the target under a race-resistant kernel primitive and bind the observed object
identity. Path replacement between decision and use poisons the cell.

Semantic or expected-result writes require Sounio as the producer. Python and
Rust remain non-waivable refusals at every descendant depth. External LLM
outputs remain `REVIEW_ONLY` and cannot become expected results, semantic
authority, parent decisions, or cell receipts.

## Deadline And Coverage

Every cell has a finite monotonic deadline and a resource budget. Every effect
family is either mediated before commit or explicitly kernel-denied by the
frozen action `9025` coverage certificate. Unknown effects are never treated as
absence.

Deadline expiry, budget exhaustion, tracing loss, an unsupported effect,
unattested architecture, or incomplete coverage transitions the cell only to
`ABORT_INCOMPLETE`. Root-process exit is not closure while descendants, open
effects, or unconsumed receipts remain.

## Lifecycle And Custody

The material reference monitor must observe a monotonic lifecycle:

```text
UNPREPARED -> PREPARED -> EFFECT_STOPPED -> EFFECT_DECIDED
           -> EFFECT_RESUMED -> ... -> CLOSING -> CLOSED
```

Any uncertain edge transitions to `POISONED`. `CLOSED` and `POISONED` are
terminal. A cell generation is bound to one `PrincipalCapsule`, one membrane
generation, and one broker epoch. Transfer to another lane, principal, ancestry,
worktree, cgroup, or broker epoch is forbidden.

The lane never receives the broker-held grant, pidfd, barrier descriptor,
policy socket authority, or journal authority. A replayed cell digest cannot
burn a sibling grant or mutate a sibling cell.

## Outcome Closure

`CLOSE_OUTCOME` requires affirmative receipts for:

- root and descendant termination;
- tree quiescence and no open effects;
- every admitted or refused effect decision;
- complete stdout, stderr, exit, signal, timeout, and write-set digests;
- the final membrane and broker journal heads;
- no unresolved crash, policy, transport, or measurement uncertainty.

`ABORT_INCOMPLETE` instead requires a typed reason, kill-tree attempt, quarantine
receipt, and a new terminal receipt hash. It never launders incomplete execution
into success, commit admission, or CI admission.

## Decision Order

Action `9029` returns the first applicable code:

- `424`: malformed field, flag, operation, state, counter, or digest shape;
- `405`: stage is before `SEMANTICS_FROZEN` or a parent freeze is absent;
- `481`: action `9028` and action `9025` semantic join is incomplete;
- `482`: capsule identity, custody, lease, or grant fence is incomplete;
- `483`: membrane generation, actor, ancestry, or pre-effect binding is
  incomplete;
- `484`: command, worktree, target identity, or claim scope is incomplete;
- `485`: deadline, budget, architecture, or effect coverage is incomplete;
- `486`: one-shot lifecycle, monotonic sequence, terminality, or non-transfer
  rule is incomplete;
- `487`: outcome closure, quiescence, crash poisoning, or quarantine is
  incomplete;
- `488`: forbidden producer or authority laundering is admitted;
- `489`: source, semantics, parent, toolchain, hardware, command, event, or
  result provenance is incomplete;
- `490`: causal sabotage evidence is incomplete;
- `0`: `ALLOW`.

The current Pod frame must return `481`: action `9028` has no material capsule,
action `9025` has no material coverage certificate, and no joined cell exists.

## Causal Sabotage

The Sounio executable must include positive fixtures for all four operations and
at least ten single-rule controls. Each control keeps the witness identical and
removes only the named refusal rule:

1. parent semantic join;
2. capsule identity and broker-only custody;
3. membrane generation and actor binding;
4. command/worktree/claim-scope binding;
5. deadline and coverage;
6. one-shot sequence and non-transfer;
7. outcome closure and terminal poisoning;
8. Python/Rust and review-only authority laundering;
9. provenance;
10. sabotage completeness.

Each derived source must change its unchanged unsafe witness from the named
`DENY` to `ALLOW`. Mock outputs, changed witnesses, parser failures, and text-only
assertions are not causal controls.

## Freeze Receipt

The freeze manifest must bind:

- this Garden source and commit;
- action `9028`, `9025`, and transitively `9023` parent manifest and semantic
  hashes;
- Sounio source, executable adapter, and source-fresh executable hashes;
- the complete fixture bundle and causal sabotage variants;
- toolchain, hardware, command, result, and decision hashes;
- positive decisions for all four operations and the current Pod `DENY481`.

## Language Roles

- Sounio: `SEMANTIC_AUTHORITY` and first executable expected results;
- OCaml: future `OPERATIONAL_KERNEL`, transporting admitted cells and receipts;
- C++20/Linux/systemd: transitory `MATERIAL_PARITY`, observing kernel facts and
  retaining pidfd/barrier custody;
- Lean 4: future `FORMAL_PARITY`;
- Koka: future `EFFECT_PARITY`;
- external LLMs: `REVIEW_ONLY`.

Python and Rust are forbidden. Shell may build, hash, compare, and package
artifacts mechanically; it cannot compute an expected decision.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-subprocess-membrane-20260828
Owner: codex-1/loom-subprocess-membrane-20260828
Concept-IDs: SOUNIO-LOOM-KERNEL-INVOCATION-CELL
Intent-Preserved: no lane operation is authorized by a bearer token, bare process identity, string command, or LLM assertion
Transformation: join frozen principal identity and effect-closure evidence into one non-transferable one-shot invocation observation
Types-Changed: add InvocationCell facts, bindings, lifecycle, and decision result
Effects-Changed: root preparation, pre-effect admission, outcome closure, and abort become jointly principal-and-membrane-bound
IR-Changed: none
Claims-Introduced: a frozen Sounio decision can distinguish a completely joined invocation observation from either parent in isolation
Claims-Forbidden: InvocationCell bytes grant authority; Sounio values are statically affine; parent ALLOW alone authorizes execution; current host execution is attached
Assumptions: a future broker can retain grants and pidfds, stop effects before commit, bind race-resistant object identity, and poison on loss
Write-Set: Garden, concept contract, registry, Sounio action 9029, adapter, gates, freeze, evidence, README
Read-Set: actions 9023, 9025, 9026, 9027, 9028 and their frozen manifests
Positive-Witness: synthetic PREPARE_ROOT, ADMIT_EFFECT, CLOSE_OUTCOME, and ABORT_INCOMPLETE observations satisfy the complete join
Negative-Witness: missing parent join, borrowed capsule, wrong actor, out-of-scope target, missing deadline, replay, live descendant, prohibited producer, missing provenance, missing sabotage
Acceptance-Gate: deterministic Sounio build, named decisions, ten causal source sabotages, frozen parent and toolchain hashes
Integration-Target: root broker and resident OCaml kernel before any material LAUNCH or effect resume
Authoritative-Only-If: Sounio executable and expected results precede every parity implementation and the complete freeze gate passes
```

## Nonclaims

- This Garden does not open `LAUNCH`, `RECYCLE`, Exec/Bash, commit, or CI.
- It does not prove current same-UID peer isolation or material effect coverage.
- It does not make ordinary Sounio values linear, affine, unique, or uncopyable.
- It does not make a capsule, cell digest, PID, pidfd number, ancestry, cgroup,
  command string, or review receipt authoritative by possession.
- It does not claim external novelty or priority; the supported novelty is the
  project-local executable semantic join and its falsifiable controls.

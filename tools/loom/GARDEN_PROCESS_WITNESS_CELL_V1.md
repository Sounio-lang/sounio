# GARDEN: LOOM Process Witness Cell V1

Status: `PREREGISTERED`

## First Phrase

A lane is not a pane and an execution is not a line of terminal text. A LOOM
execution should be a causal object that can prove which authority opened it,
which kernel process embodied it, what it observed, and how both process and
authority became extinct.

## Question

Can LOOM turn one Sounio-authorized command into a non-replayable,
proof-carrying process whose executable identity, kernel lifetime, effects,
result, and affirmative extinction are one inseparable witness?

## Authority Order

This experiment extends the existing architecture. It does not create a second
broker, controller, policy language, or semantic oracle.

The mandatory order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

The semantic root is frozen Sounio action `9030`. Its `CONSUME`,
`OUTCOME_PENDING`, `CLOSE`, and `REVOKE` rules already define the semantic
boundary tested here. The exact frozen manifest is
`tools/loom/kernel_exec_grant_cell_authority.freeze.v1`. The distinct-UID host
quorum is frozen by `tools/loom/host_exec_quorum_host.runtime.v1` with
`material_grant=true` and `material_execution=false`.

The first payload and its expected outcome must be produced in Sounio before
any host execution code is added. If this experiment needs a new state,
transition, decision, or expected result not already covered by action `9030`,
work stops and a new Sounio Garden and action are frozen first. C++20 may only
realize Linux process mechanics. OCaml may only realize the already-frozen
effect protocol. Lean 4 and Koka remain parity roles. External LLMs are
review-only. Python and Rust are forbidden.

## Hypothesis

An execution becomes a `ProcessWitnessCell` only when all of these objects have
the same identity chain:

1. one frozen Sounio action-`9030` `CONSUME` decision;
2. one atomically consumed host `ExecGrant`, released through the existing
   three-object descriptor quorum;
3. one exact Sounio-produced executable opened and executed by the same file
   descriptor, with frozen executable, argv, environment, cwd, and root hashes;
4. one kernel process identity bound before release and preserved across
   `execveat` by PID, pidfd, start tick, UID/GID vector, cgroup, and namespace;
5. one bounded effect observation containing stdout, stderr, exit, signal,
   timeout, descendant, resource, and write-set facts;
6. one action-`9022` outcome and action-`9029` close chain for those exact
   facts;
7. one action-`9030` terminal receipt proving state, generation, and authority
   extinction.

Possessing command text, a path, a digest, stdout bytes, exit code zero, a
receipt, a PID, a pidfd number, or a copied descriptor satisfies none of this
conjunction by itself.

## The Novel Object

The product object is not a transcript. It is a content-addressed causal DAG:

```text
SemanticPermit(9030)
        |
        v
ConsumedExecGrant ---- PrincipalCell ---- FrozenExecutable
        |                    |                    |
        +--------------------+--------------------+
                             |
                             v
                    LiveProcessWitness
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
             EffectSet              KernelLifetime
                 |                       |
                 +-----------+-----------+
                             |
                             v
                  ClosedOutcomeWitness
                             |
                             v
                AffirmativeExtinctionTriple
```

Every edge is hash-bound and generation-bound. The DAG may later be encoded as
Arrow records for zero-copy observation and rendered by the Spectral Weave UI,
but neither Arrow nor the UI is authority. Losing the view cannot change the
cell, and replaying the view cannot recreate it.

## First Material Payload

The first payload is deliberately narrow:

- source is Sounio;
- the source, executable, toolchain, argv, empty environment, cwd, root, and
  expected stdout are frozen before the host probe;
- there is no shell, interpreter, package manager, network, clock-derived
  output, random input, writable home, or dynamically selected command;
- the payload emits one bounded Sounio-produced witness and exits with the
  Sounio-produced expected status;
- `stderr` is empty, descendants are zero, and the allowed write set is empty;
- the executable is opened with no symlink traversal, hashed, then executed
  from that same open descriptor with `execveat(..., AT_EMPTY_PATH)`;
- an unavailable `execveat`, digest mismatch, file mutation, oversized stream,
  unknown descendant, or observation loss refuses before success.

This fixed payload is an instrument calibration, not a general command runner.

## Process Identity Across Exec

Before releasing the barrier, the broker binds the DynamicUser principal-cell
PID, pidfd, start tick, four-value UID/GID vectors, cgroup, namespace, and
pre-exec executable. The principal cell then consumes exactly one release byte
sequence and calls `execveat` without forking.

After release, the broker must observe:

- the same PID, pidfd, start tick, UID/GID vectors, cgroup, and namespace;
- the executable identity changing exactly once from the frozen principal cell
  to the frozen Sounio payload;
- no intermediate shell, trampoline, interpreter, or user-selected binary;
- no sibling or descendant process;
- the exact terminal wait status for that same pidfd;
- cgroup quiescence before outcome closure.

A new PID, start-tick change, principal drift, executable substitution, cgroup
escape, observation gap, or ambiguous process tree produces `POISONED`, never
success.

## Effect Witness

The first `EffectSet` is closed and finite. It records:

- byte counts and SHA-256 digests of bounded stdout and stderr;
- exact exit status, terminating signal, timeout state, and wait provenance;
- maximum resident memory, CPU time, and wall-clock bounds as material facts;
- network disabled and capabilities empty;
- read-only system and release roots;
- zero descendants at close;
- an explicitly empty allowed write set;
- the pre-exec and post-exec executable digests;
- the complete parent grant, principal, invocation, command, and outcome hashes.

An empty observation is not evidence of no effect. The witness must affirm the
mechanism that made each effect impossible or measured it to closure. Missing
stdout, missing cgroup observation, missing wait status, or missing write-set
observation remains `UNKNOWN`.

## Affirmative Extinction

The `ProcessWitnessCell` closes only when one receipt affirms all three facts:

1. **process extinction**: the exact pidfd is terminal, the main process is
   reaped, the cgroup is unpopulated, and no descendant remains;
2. **generation extinction**: the execution generation and consumed grant
   generation are terminal and cannot be issued or consumed again;
3. **authority extinction**: the descriptor barrier is closed, all inherited
   copies are gone, the controller records terminal consumption, and restart
   cannot reconstruct the grant.

Silence, a missing PID, an empty `ps`, a stopped unit, EOF, timeout, broker
restart, or absent UI state proves none of these alone.

## Atomic Publication

The broker publishes no successful execution receipt until all effect and
extinction fields are complete. The terminal receipt is built off to the side,
validated against the frozen Sounio shape, and committed atomically as one
content-addressed object. A crash before publication leaves a typed incomplete
outcome and quarantine obligation. A crash after publication can replay the
receipt but cannot replay execution.

The receipt must contain at least:

- Sounio source and frozen-semantics hashes;
- producing language and language role;
- controller, broker, principal-cell, and payload hashes;
- toolchain and hardware identities;
- exact command descriptor and generation;
- pre-exec and post-exec kernel identities;
- effect-set and terminal-outcome hashes;
- state, generation, and authority extinction facts;
- final result and causal-sabotage result.

## Causal Sabotage Matrix

The material gate must pre-register one treatment, one positive run, and these
single-rule controls. The payload and all unrelated inputs remain byte-for-byte
identical:

1. remove only descriptor release: the payload must not start;
2. bypass only the descriptor quorum: the payload starts, proving that the
   quorum caused the treatment refusal;
3. substitute only the executable after measurement: execution is refused;
4. replace only Sounio expected stdout with a material-layer expectation:
   authority laundering is refused before execution;
5. forge only stdout and exit zero without the pidfd lifetime: close is refused;
6. permit one descendant: outcome remains incomplete;
7. remove only cgroup-quiescence evidence: extinction is refused;
8. remove only one member of the extinction triple: close is refused;
9. kill the controller after issue and before release: the payload does not
   start and the grant becomes terminal;
10. kill the broker after execution and before close: success is not published;
11. replay the complete terminal receipt: no second process starts;
12. use the same-UID pod principal: the barrier remains closed;
13. attempt a Python or Rust oracle: refusal occurs before that process exists.

The control that bypasses the quorum is sabotage, not a product path. Its only
purpose is to prove that the exact guarded rule caused the refusal.

## Acceptance Gates

The first material phase may set `material_execution=true` only if:

- this Garden commit predates every exec-capable material change;
- the exact Sounio payload and expected result are frozen first;
- the existing action-`9030` semantic and host-grant manifests remain unchanged;
- one distinct-UID DynamicUser treatment stays closed;
- one exact release starts exactly one payload and consumes exactly one grant;
- all thirteen controls reach their pre-registered outcomes;
- the same pidfd identity spans the principal-cell-to-payload exec transition;
- stdout, stderr, exit, descendants, resource bounds, write set, outcome, and
  extinction are completely receipt-bound;
- source-fresh local builds are deterministic;
- the host probe is rollback-safe and production activation remains false;
- no shell, Python, Rust, parity implementation, LLM, UI, Arrow consumer, or
  textual receipt can provide authority;
- the freeze gate independently validates every source, semantics, toolchain,
  hardware, command, result, and receipt hash.

Even after this passes, these product boundaries remain closed:

- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.

The experiment proves one calibrated material execution, not a general-purpose
runner and not production readiness.

## Stop Rule

Stop and fail closed on any semantic drift from action `9030`, mutable or
path-selected payload, shell or interpreter mediation, missing kernel identity,
same-UID authority, unbounded output, unknown descendant, missing effect fact,
inferred extinction, result publication before closure, production activation,
or sabotage that changes more than its named rule.

## Evidence Boundary

Current evidence establishes a frozen distinct-UID non-bearer material grant on
host `t560-proxmox`, with exactly one guarded open and one causal sabotage open.
It does not establish material execution. This Garden preregisters the next
experiment and changes none of those facts.

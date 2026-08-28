# Loom Subprocess Membrane

> **Status**: Garden seed | **Last validated**: 2026-08-28 | **Source**: BLK-20260828-loom-exec-attachment

## Butterfly

> The command line is only the seed. Its process tree is the execution.

## Pressure

The current LOOM broker can prove a directly measured leaf, bind it to a
single-use in-memory capability, supervise its ending, and persist the outcome.
That is insufficient for arbitrary Bash/Exec attachment. A shell can perform
effects that are absent from its original string:

- launch prohibited or unclassified descendants;
- hide an interpreter behind a wrapper or shebang;
- write through shell redirection or a builtin without another `exec`;
- fork a descendant that outlives the root shell;
- mutate a path after authorization but before use;
- hang after consuming its grant;
- attempt commit or CI promotion while effects remain open.

Treating the initial shell binary as closure-attested would erase these facts.

## Core Idea

A LOOM execution is a generation-bound membrane over an effect stream.

```text
ROOT_ADMIT
-> PROCESS_CREATE* / EXEC* / WRITE_OPEN* / PATH_MUTATE*
-> EXIT* or TIMEOUT_TERMINATE
-> TREE_QUIESCENT
-> EXEC_OUTCOME
-> COMMIT_ADMIT or CI_ADMIT
```

The native supervisor stops each relevant effect before the kernel commits it,
measures the actor, target, ancestry, generation, arguments, environment,
deadline, and claim scope, then asks a frozen Sounio policy. `ALLOW` resumes the
effect. `DENY`, missing policy, malformed response, timeout, supervisor loss,
or incomplete measurement terminates the complete membrane and records an
incomplete outcome.

The membrane is not a better shell parser. It observes the effect boundary.

## Authority Invariants

1. Sounio defines the event algebra and expected decisions before native mediation exists.
2. The initial command string never proves descendant closure.
3. Every process node is bound to the root grant, kernel generation, parent identity, and process birth identity.
4. Every executable image is measured before execution, including the root shell and every descendant.
5. Python and Rust are non-waivable refusals at every depth.
6. External LLMs remain review-only and cannot write semantics or expected results.
7. A write-capable open or path mutation is resumed only when its canonical target and claim-scope receipt are bound.
8. Semantic or expected-result writes require Sounio as the producing language.
9. A configured finite deadline is mandatory. Missing, invalid, or expired deadlines fail closed.
10. Timeout terminates the entire process tree and produces a typed timeout observation, never success.
11. Root exit is not closure while descendants or effects remain open.
12. Commit and CI require a quiescent membrane, a complete outcome, and a valid receipt chain.
13. Supervisor, tracer, policy, journal, or kernel loss terminates the membrane and materializes `INCOMPLETE`.
14. The decision log records every `ALLOW` and `DENY` with its exact reason and event digest.
15. A founder waiver is scoped, purpose-bound, expiring, receipt-bound, and cannot waive Python/Rust.

## Effect Algebra

| Effect | Required evidence before resume | Completion evidence |
| --- | --- | --- |
| `ROOT_ADMIT` | frozen policy, membrane installed, root grant, command, cwd, environment, hardware, deadline | root process birth identity |
| `PROCESS_CREATE` | root grant, generation, exact parent birth identity | child birth identity |
| `EXEC_ADMIT` | actor identity, canonical executable, argv, environment, toolchain, language role | successful image transition or explicit failure |
| `WRITE_OPEN` | actor language, canonical target, operation, claim-scope receipt | open result and resulting object identity |
| `PATH_MUTATE` | source/target identities, operation, claim-scope receipt | mutation result |
| `EXIT_RECORD` | actor identity, exit or signal, remaining tree state | node retired |
| `TIMEOUT_TERMINATE` | deadline and current monotonic observation | all nodes reaped, typed timeout result |
| `COMMIT_ADMIT` | quiescent tree, no open effects, complete outcome, receipt chain | commit digest receipt |
| `CI_ADMIT` | same as commit plus immutable source snapshot | CI invocation receipt |

## Coverage Boundary

The first Linux realization must either mediate or explicitly refuse:

- `fork`, `vfork`, and `clone` descendants;
- `execve` and `execveat`;
- write-capable `open`, `openat`, and `openat2`;
- `creat`, `truncate`, and `ftruncate`;
- `rename`, `renameat`, `renameat2`, `unlink`, `unlinkat`, `link`, `linkat`,
  `symlink`, `symlinkat`, `mkdir`, `mkdirat`, `rmdir`, `chmod`, `fchmod`,
  `fchmodat`, `chown`, `fchown`, and `fchownat`;
- process escape attempts, including tracing loss and unobserved descendants.

An unsupported architecture, syscall family, or kernel mediation feature is a
pre-execution refusal, not a silent downgrade.

## Causal Controls

The executable semantics must contain at least these sabotage experiments:

1. Remove only the Python/Rust descendant refusal. The unchanged hidden-Python event becomes `ALLOW`.
2. Remove only the write-scope rule. The unchanged out-of-scope builtin write becomes `ALLOW`.
3. Remove only the quiescence rule. The unchanged commit-with-live-descendant event becomes `ALLOW`.
4. Remove only the deadline rule. The unchanged no-deadline root becomes `ALLOW`.

These controls distinguish the intended rule from incidental parse, runtime, or
harness failure.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-subprocess-membrane-20260828
Owner: codex-1/loom-subprocess-membrane-20260828
Concept-IDs: SOUNIO-LOOM-SUBPROCESS-MEMBRANE
Intent-Preserved: arbitrary Bash syntax may be used only when every resulting effect remains observable and Sounio-governed
Transformation: replace static shell closure claims with a generation-bound pre-effect membrane
Types-Changed: add membrane effect, actor identity, deadline, scope, tree state, and effect receipt
Effects-Changed: process creation, executable transition, write-capable path effects, commit, and CI become Sounio-gated
IR-Changed: none
Claims-Introduced: a closed membrane receipt proves bounded mediation coverage for one root execution on a named platform
Claims-Forbidden: parsing the command proves closure; root exit proves tree exit; same-UID token possession proves authority; timeout proves success
Assumptions: the selected kernel mediation primitive can stop every declared effect before commit and terminates tracees if the supervisor dies
Write-Set: Sounio membrane policy and freeze, native supervisor, hooks, runtime bundle, CI gates, evidence
Read-Set: execution authority V2, durable outcome V1, claim leases, process identity, kernel-generation journal
Positive-Witness: nested allowed native tools and in-scope writes close one quiescent membrane receipt
Negative-Witness: hidden Python, hidden Rust, out-of-scope builtin write, missing deadline, escaped descendant, policy timeout, and live-descendant commit refuse
Sabotage-Control: four single-rule removals admit their unchanged negative witnesses
Acceptance-Gate: Sounio cases and sabotage, frozen semantics, native syscall coverage, timeout kill-tree, write scope, crash recovery, hook attachment, commit gate, CI gate
Integration-Target: Codex and Claude Bash/Exec PreToolUse plus commit and CI enforcement
Authoritative-Only-If: Sounio semantics are frozen before realization and the platform coverage gate proves every declared effect is mediated or pre-execution refused
```

## Evidence State

| Layer | Status |
| --- | --- |
| `GARDEN` | Captured by this seed. |
| `SOUNIO_EXECUTABLE` | Not yet. |
| `SEMANTICS_FROZEN` | Not yet. |
| `PARITY_OPEN` | No. |
| `CLAIM_READY` | No. |

## What This Is Not

- Not shell-string allowlisting.
- Not `LD_PRELOAD` or a cooperative wrapper.
- Not a Python, Rust, Node, Ruby, shell, `awk`, or `bc` policy oracle.
- Not a claim of hostile same-UID isolation beyond the named kernel boundary.
- Not permission to enable Bash/Exec before the coverage and crash gates pass.
- Not a rewrite of the already frozen execution-authority or outcome semantics.

## Next Executable Bridge

Implement the effect algebra and expected decisions in Sounio action `9023`.
Commit that executable as the immediate child of this Garden commit. Freeze its
source, entrypoint, compiler, command, expected outputs, and parent-authority
hashes before writing the native mediation layer.

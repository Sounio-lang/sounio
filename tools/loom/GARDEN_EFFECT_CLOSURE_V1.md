# LOOM Effect Closure Authority V1

> Status: Garden seed | Date: 2026-08-28 | Parents: frozen subprocess
> membrane action `9023`, frozen resident transport action `9024`

## Butterfly

> Absence is not an empty log. It is a positive, sabotaged, receipt-bound fact.

## Pressure

The diagnostic native membrane can stop and classify a useful Linux syscall
slice, ask a resident Sounio authority, and kill the tracee when authority is
lost. That is evidence for those exercised paths. It is not evidence that an
arbitrary Bash/Exec generation has no unobserved effect route.

Silence is ambiguous. A missing event can mean that the effect was impossible,
that the observer missed it, that the platform exposes another syscall family,
or simply that the test never tried. Promoting silence to closure would turn a
diagnostic implementation into a semantic oracle.

The Mycelia/Pireus transplant is precise:

- validate the claim when it is written, before promotion;
- use closed domains where invention must be impossible;
- represent verified absence as an affirmative fact, not as a missing fact;
- propagate a failed absence proof to every dependent conclusion.

## Core Idea

Action `9025` admits an effect-closure certificate only when every member of a
closed effect-family universe has an affirmative closure fact.

```text
EffectAbsent(
  family,
  coverage_mode,
  platform,
  parent_9023_hash,
  parent_9024_hash,
  resident_runtime_hash,
  coverage_manifest_hash,
  sabotage_receipt_hash)
```

`EffectAbsent` means the effect cannot cross the named generation membrane by
an ungoverned route on the named platform. It does not mean that no event was
seen. The fact is valid only when the family is either denied by the kernel or
mediated by Sounio and independently kernel-backstopped.

## Closed Effect Universe

Action `9025` recognizes exactly these twelve families:

| ID | Family | Examples that must be accounted for |
| --- | --- | --- |
| `1` | executable transition | `execve`, `execveat`, shebang/interpreter transition |
| `2` | process topology | `fork`, `vfork`, `clone`, `clone3`, escaped descendants |
| `3` | filesystem path | write-open, create, truncate, rename, link, unlink, metadata mutation |
| `4` | descriptor mutation | inherited writable FDs, `dup`, `fcntl`, descriptor passing |
| `5` | mapped storage | writable/shared `mmap`, `msync`, mapped-file mutation |
| `6` | asynchronous I/O | `io_uring`, AIO, deferred submissions after decision |
| `7` | network | IP sockets, connect/bind/listen, packet paths |
| `8` | filesystem Unix socket | pathname and abstract Unix sockets, SCM rights/credentials |
| `9` | interprocess communication | pipes, shared memory, message queues, signals, `ptrace` |
| `10` | device and kernel control | device nodes, `ioctl`, BPF, perf, namespaces, mounts |
| `11` | process/kernel filesystem | `/proc`, `/sys`, cgroup and namespace control surfaces |
| `12` | unknown or future family | any effect not classified above |

The set is closed for this version. A producer cannot invent family `13` or
omit family `12`. New families require a new Garden and a new frozen action.

## Coverage Modes

Each family has exactly one declared mode:

| Mode | Meaning | Closure-capable? |
| --- | --- | --- |
| `0` | missing or unmeasured | no |
| `1` | decision-mediated without independent kernel backstop | no |
| `2` | denied before execution by the named kernel boundary | yes |
| `3` | Sounio-mediated before commit and independently kernel-backstopped | yes |

Every known family (`1` through `11`) must be mode `2` or `3`. The unknown
family (`12`) must be exactly mode `2`: fail-closed kernel refusal. Treating an
unknown effect as mediated would merely rename ignorance.

## Action 9025 Facts

The first executable Sounio authority consumes:

- stage;
- both frozen parent bindings (`9023` and `9024`);
- resident runtime binding;
- supported architecture and kernel boundary;
- fail-closed policy presence;
- supervisor-loss revocation;
- hostile same-UID peer isolation;
- path-race closure;
- exact effect-family count;
- one coverage mode for every family;
- exact causal-sabotage count;
- source, semantic, runtime, coverage-manifest, toolchain, hardware, command,
  result, and sabotage-receipt bindings.

The expected refusal taxonomy is:

| Code | Meaning |
| --- | --- |
| `446` | parent `9023`/`9024` or resident runtime is not frozen and bound |
| `447` | effect universe or a known family is not materially closed |
| `448` | causal sabotage coverage is incomplete |
| `449` | fail-closed, supervisor-loss, or path-race invariant failed |
| `450` | receipt or provenance binding is incomplete |
| `451` | hostile same-UID peer isolation is absent |
| `452` | unknown/future effects are not kernel-denied |
| `453` | architecture or kernel boundary is unsupported |

Malformed frames and wrong stages retain the shared `424` and `405` refusals.

## Authority Invariants

1. Actions `9023` and `9024` are frozen before `9025` exists.
2. The resident runtime hash is bound to the same parent chain.
3. The effect universe is closed and contains exactly twelve families.
4. Every known family is mode `2` or `3`; mode `1` never proves closure.
5. Unknown/future effects are mode `2` and fail before execution.
6. The architecture and kernel boundary are named and supported.
7. Missing policy, parse failure, timeout, observer loss, or receipt failure denies.
8. Supervisor loss revokes the complete generation and kills or blocks its tracees.
9. Path evidence closes authorization/use races; pathname inspection alone is insufficient.
10. Hostile same-UID peers cannot issue, steal, replay, consume, or burn another lane's authority.
11. At least one causal sabotage exists for every claimed family, and the action-level sabotage count is exact.
12. Each absence fact binds source, semantics, producer language and role, toolchain, hardware, command, result, coverage manifest, and sabotage receipt.
13. Only Sounio can produce the expected closure decision.
14. OCaml and C may enforce a frozen decision and provide material observations; they cannot promote diagnostic coverage to closure.
15. Lean, Koka, C++, Haskell, and external LLMs cannot create or confirm the semantic result.
16. Failure of any absence fact invalidates every dependent attachment, commit, CI, and `CLAIM_READY` conclusion.

## Write-Time Validation

The certificate is checked before it enters the authority journal or enables an
attachment. A rejected certificate is not stored as a weaker successful claim.
It is recorded as `DENY` with the precise failing rule and receipt digest.

Warnings may annotate diagnostic observations, but no warning is promotable to
effect closure. `ALLOW` is reserved for the complete closed-world shape.

## Causal Controls

The executable Sounio gate must perform at least these five single-rule
sabotages against unchanged negative frames:

1. Remove only the frozen-parent rule. An orphan certificate becomes `ALLOW`.
2. Remove only the all-known-families material-closure rule. A mode-`1` family becomes `ALLOW`.
3. Remove only the exact sabotage-count rule. An incomplete sabotage set becomes `ALLOW`.
4. Remove only the same-UID isolation rule. A peer-unsafe certificate becomes `ALLOW`.
5. Remove only the provenance-binding rule. A certificate with a zero coverage-manifest hash becomes `ALLOW`.

An additional negative frame must prove that removing no rule leaves unknown
effects without mode `2` at `DENY452`.

## Stages

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> MATERIAL_COVERAGE
-> ATTACHMENT_OPEN
-> CLAIM_READY
```

No native implementation, OCaml integration, Lean proof, Koka effect parity,
C++ material parity, or Haskell baseline may define expected results before the
Sounio action and semantic manifest are frozen.

## Positive Semantic Witness

A hypothetical named Linux boundary supplies all twelve facts: the eleven known
families are mode `2` or `3`, unknown effects are mode `2`, all hashes and
receipts are nonzero, the parent chain is frozen, supervisor loss revokes, path
races are closed, and hostile same-UID peers are isolated. Sounio returns
`ALLOW`.

This witness defines the semantic target. It is not evidence that the current
diagnostic membrane realizes it.

## Current Material Witness

The current Linux diagnostic implementation must return `DENY` because at
least these affirmative facts are absent:

- hostile same-UID peer isolation is explicitly false;
- filesystem and abstract Unix-socket closure is not proven;
- descriptor-passing, mapped-storage, asynchronous-I/O, device, and procfs/sysfs
  families are not closed;
- pathname authorization/use race closure is not proven;
- per-family causal sabotage receipts do not yet exist.

These are positive `UNKNOWN` or `REFUTED` facts, not empty checklist cells.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-effect-closure-authority-20260828
Owner: codex-1/loom-subprocess-membrane-20260828
Concept-IDs: SOUNIO-LOOM-EFFECT-CLOSURE-AUTHORITY
Intent-Preserved: Bash/Exec attachment requires affirmative proof that every effect family is governed or denied
Transformation: replace silent non-observation with a closed-world, sabotage-bound EffectAbsent certificate
Types-Changed: add effect family, coverage mode, closure certificate, absence fact, and propagation status
Effects-Changed: certificate write, attachment, commit, CI, and claim promotion become closure-gated
IR-Changed: none
Claims-Introduced: one hash-bound certificate may prove effect closure for one named platform and generation contract
Claims-Forbidden: an empty trace proves absence; diagnostic syscall coverage proves general Exec; same-UID bearer possession proves isolation
Assumptions: the selected material boundary can deny unknown effects before execution and isolate hostile same-UID peers
Write-Set: Garden, Sounio action 9025, selftests, freeze, evidence, and later resident integration
Read-Set: frozen 9023/9024 manifests, resident runtime manifest, native diagnostic receipts, Mycelia/Pireus extraction
Positive-Witness: all twelve families materially closed with exact sabotage and provenance receipts
Negative-Witness: mode-1 family, missing sabotage, same-UID peer exposure, unknown-family downgrade, path-race gap, zero receipt hash
Sabotage-Control: five single-rule removals admit their unchanged negative witnesses
Acceptance-Gate: Sounio cases, five causal sabotages, frozen hashes, current-material DENY, then independent material coverage
Integration-Target: resident Sounio authority before Bash/Exec, commit, and CI attachment
Authoritative-Only-If: Sounio semantics are frozen and the named platform supplies every affirmative absence fact
```

## Evidence State

| Layer | Status |
| --- | --- |
| `GARDEN` | Captured by this seed. |
| `SOUNIO_EXECUTABLE` | Not yet. |
| `SEMANTICS_FROZEN` | Not yet. |
| `PARITY_OPEN` | No. |
| `MATERIAL_COVERAGE` | Current diagnostic implementation refuses. |
| `ATTACHMENT_OPEN` | No. |
| `CLAIM_READY` | No. |

## Nonclaims

- This Garden is not a proof that Linux has only the listed syscalls.
- It does not promote ptrace, namespaces, seccomp, Bubblewrap, OCaml, or C to semantic authority.
- It does not establish hostile same-UID isolation.
- It does not close Unix sockets, inherited descriptors, mapped files, asynchronous I/O, devices, `/proc`, `/sys`, or future kernel effects.
- It does not enable general Bash/Exec, commit, or CI attachment.
- It does not make a hypothetical positive semantic witness a material result.
- It does not permit parity languages or LLM review to confirm a Sounio decision.

## Next Executable Bridge

Implement action `9025` in Sounio as the immediate child of this Garden
commit. The selftest must prove the positive semantic witness, the current
material `DENY`, all refusal codes, and the five causal sabotages. Freeze the
source, entrypoint, compiler, command, outputs, parent hashes, and Garden hash
before any parity or resident integration begins.

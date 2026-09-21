# GARDEN: LOOM Process Witness Effect Cell V1

Status: `PREREGISTERED`

## First Phrase

The strongest absence witness is not a complete observer. It is a principal
for which the kernel has no undeclared route to perform the absent effect.

## Question

Can one already-frozen `ProcessWitnessCell` become a denial-carrying principal:
a process whose admissible effect universe is produced first by Sounio, whose
known effects are kernel-denied or Sounio-mediated and kernel-backstopped, and
whose unknown or future effects are denied by construction rather than inferred
from silent telemetry?

## Parent Authority

This is a derived material experiment. It creates no second broker, policy
language, effect taxonomy, semantic state, or expected result.

The semantic parent is frozen Sounio action `9025`:

- manifest: `tools/loom/effect_closure_authority.freeze.v1`;
- manifest SHA-256:
  `c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91`;
- source SHA-256:
  `41bb7ca65d7e0313cfa282c01c3a6ebf7ba2f880948569a7eec8e253a31f5a67`;
- semantics SHA-256:
  `a39a18c9016906ef89b480ca2921db0afc3e6af08f594bd937f6d42f273b3fd4`;
- current material decision: `DENY447 material-coverage-incomplete`.

The material execution parent is the frozen host ProcessWitness core:

- manifest: `tools/loom/process_witness_host.runtime.v1`;
- manifest SHA-256:
  `eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00`;
- freeze commit: `b735ad705339`;
- stage: `MATERIAL_EXECUTION_CORE_FROZEN`;
- facts: `process_witness_core=true`, `affirmative_extinction=true`,
  `complete_effects=false`, `material_execution=false`.

The mandatory authority order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> MATERIAL_COVERAGE
-> CLAIM_READY
```

Before native effect-cell code exists, a new Sounio executable must emit the
exact derived policy plan and expected treatment/sabotage outcomes. That plan
must be frozen and hash-bound to both parents above. C++20 may then apply Linux
mechanics and report material facts. It cannot rename a refusal, add an effect
family, choose the expected result, or promote itself to semantic authority.

Python and Rust are forbidden. Shell may invoke frozen tools but may not compute
the semantic answer. OCaml remains `EFFECT_PARITY`; Lean 4 is `FORMAL_PARITY`;
Koka is `EFFECT_PARITY`; C++ is `MATERIAL_PARITY`; external LLMs are
`REVIEW_ONLY`.

## Hypothesis: Denial-Carrying Principal

For the calibrated Sounio payload, a `ProcessWitnessEffectCell` can carry an
affirmative absence certificate when all three boundaries hold simultaneously:

1. **Object boundary:** Landlock restricts filesystem reachability using rules
   installed for already-opened directory and file objects. Path strings never
   authorize the payload, and the executable is still hashed and executed from
   the same open descriptor with `execveat(..., AT_EMPTY_PATH)`.
2. **Operation boundary:** seccomp installs a positive, architecture-bound
   syscall allowlist with argument constraints and a fail-closed default action.
   A syscall or argument shape absent from the frozen plan is refused, including
   a future syscall unknown to this experiment.
3. **Principal boundary:** the existing distinct-UID DynamicUser, cgroup,
   capability-empty, namespace-bound, descriptor-minimal ProcessWitness cell
   preserves one process identity from release through payload and affirmative
   extinction.

No one mechanism proves closure alone. Landlock does not govern network,
process creation, or arbitrary kernel control. seccomp does not prove object
identity or eliminate writable inherited descriptors. DynamicUser does not
constrain the syscall universe. The claim is their hash-bound conjunction for
one frozen principal, executable, architecture, kernel, command, and generation.

## Sounio-First Policy Plan

The next executable artifact must be written in Sounio and deterministically
emit one closed plan with exactly twelve rows. Each row contains:

- action `9025` effect-family ID and canonical name;
- required coverage mode (`2` kernel-denied or `3` Sounio-mediated with an
  independent kernel backstop);
- named kernel mechanism and immutable rule-set digest;
- treatment probe and expected Sounio decision;
- one single-rule sabotage and expected Sounio decision;
- evidence fields required from the actual ProcessWitness instance;
- parent semantic and material manifest hashes.

The Sounio artifact also emits the exact positive action-`9025` frame expected
after material evidence exists, and the unchanged negative frame used when one
family loses its named rule. The native implementation consumes this frozen
plan. It must not derive an expectation from C++ enums, shell tables, exit-code
conventions, or log contents.

## Closed Twelve-Family Plan

The family IDs and meanings are inherited verbatim from action `9025`.

| ID | Family | Preregistered material closure strategy |
| --- | --- | --- |
| `1` | executable transition | One pre-opened, hashed Sounio executable; `execveat` from that descriptor; deny every further `execve` and `execveat`. |
| `2` | process topology | Deny `fork`, `vfork`, `clone`, `clone3`, and namespace creation; require the same pidfd and zero descendants until cgroup extinction. |
| `3` | filesystem path | Landlock allow only the preregistered read objects; no writable rules; seccomp denies mutation syscalls and mutating open flags. |
| `4` | descriptor mutation | Start from an exact descriptor inventory; close all unrelated descriptors; deny duplication, descriptor passing, and mutable `fcntl` operations not required by the frozen payload. |
| `5` | mapped storage | Deny writable or shared file mappings by syscall and argument; deny `msync`; bind actual mapping inventory at close. |
| `6` | asynchronous I/O | Deny `io_uring`, Linux AIO setup/submission, and deferred work mechanisms absent from the frozen plan. |
| `7` | network | Deny socket creation and all IP network operations; verify no network descriptor exists before release or at close. |
| `8` | filesystem Unix socket | Deny Unix socket creation, connection, binding, acceptance, and `SCM_RIGHTS`; inherited Unix sockets are absent. |
| `9` | interprocess communication | Deny pipe creation, SysV/POSIX shared-memory creation, message queues, `ptrace`, broad signals, and cross-principal IPC; only preregistered witness channels survive. |
| `10` | device and kernel control | Landlock exposes no device object; seccomp denies uncontrolled `ioctl`, BPF, perf, mount, namespace, module, keyring, and reboot controls. |
| `11` | process/kernel filesystem | Landlock exposes no writable `/proc`, `/sys`, cgroup, or namespace-control object; mutation and handle-opening routes are denied. |
| `12` | unknown or future family | seccomp architecture check plus default-deny: any operation absent from the positive allowlist is refused before execution. |

The actual Sounio compiler/runtime payload may require a smaller or different
positive syscall set than anticipated here. Discovery runs may observe that set,
but they cannot authorize it. Every admitted operation must be added first to a
new Sounio policy-plan freeze, with its family and argument constraints named,
before the material treatment can use it.

## Actual-Instance Measurement

The treatment receipt must bind policy to the process that actually ran, not to
a nearby unit or a template configuration. At minimum it records:

- exact PID, pidfd, start tick, UID/GID vectors, cgroup, seven namespaces, and
  executable transition already required by ProcessWitness;
- seccomp mode and policy digest installed in that process before release;
- Landlock ABI, handled-access mask, rule digest, and `no_new_privs` state in
  that process before release;
- initial and terminal descriptor inventories, including object identity and
  access mode for every surviving descriptor;
- initial and terminal memory-map inventories sufficient to exclude writable
  shared file mappings;
- initial and terminal network/Unix-socket inventories;
- cgroup membership and descendant count through terminal extinction;
- exact stdout, stderr, wait status, resource bounds, state extinction,
  generation extinction, and authority extinction;
- source, semantics, policy-plan, material-runtime, toolchain, hardware,
  command, result, treatment, and sabotage hashes.

Configuration text, unit properties, a successful `prctl`, an empty trace, or a
post-mortem scan of another PID is not actual-instance evidence.

## Twin Probe: Treatment and Causal Sabotage

The gate contains two byte-identical copies of the same frozen Sounio payload,
invocation, host release, and observation code. The only permitted difference
in each sabotage is the named enforcement rule. The product path never exposes
the sabotage switch.

For each family `1..12`, the matrix runs:

1. one treatment probe that attempts the representative forbidden effect and
   receives the preregistered kernel refusal while the cell remains governed;
2. one twin probe with only that family's rule removed, in an isolated throwaway
   test cell, where the same attempt crosses the boundary;
3. the unchanged action-`9025` negative frame, which Sounio refuses because
   that family is no longer materially closed;
4. complete teardown and affirmative extinction before the next pair begins.

This yields exactly twelve treatments and twelve single-family sabotages. A
sabotage that merely changes a Boolean reported to Sounio is invalid. A
sabotage that changes payload bytes, UID, filesystem image, arguments, observer,
timing contract, or more than one rule is invalid. A bypass success is evidence
only that the named treatment rule was causal; it is never an allowed product
mode.

## Fail-Closed Construction

The cell refuses before payload execution when any of these is absent, unknown,
unsupported, mutable, timed out, or hash-mismatched:

- frozen Sounio policy plan and both frozen parents;
- supported kernel, architecture, seccomp mode, or Landlock ABI;
- exact executable descriptor, descriptor inventory, and object rules;
- any of the twelve treatment or twelve sabotage receipts;
- actual-instance policy observation;
- same-UID peer isolation, path-race closure, controller authority, pidfd, or
  cgroup observation;
- atomic terminal publication or the full extinction triple.

An older kernel that cannot express a required restriction refuses. There is no
compatibility fallback from Landlock object rules to pathname checking, from
seccomp default-deny to a blacklist, or from distinct UID to a bearer secret.

## Acceptance Gate

`complete_effects=true` and then `material_execution=true` are permitted only
when one immutable receipt proves all of the following:

1. this Garden commit predates Sounio policy-plan and native effect-cell bytes;
2. the Sounio plan is executable, frozen, deterministic, and parent-bound;
3. all twelve treatment probes receive their exact kernel refusals;
4. all twelve one-rule twin sabotages cross only their named boundary;
5. action `9025` consumes the complete material bindings and returns its frozen
   positive `ALLOW` decision;
6. the actual ProcessWitness PID carries the exact seccomp, Landlock,
   descriptor, mapping, network, identity, cgroup, and namespace facts;
7. the same payload still satisfies the frozen ProcessWitness stdout, stderr,
   wait, no-descendant, same-pid, and extinction witnesses;
8. controller loss, broker loss, observer loss, timeout, and receipt mutation
   each fail closed;
9. source-fresh local and host gates agree on every content hash;
10. production activation and all product attachments remain closed.

Even after a passing material experiment:

```text
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
```

Opening any one of those fields requires a later Garden, a separately frozen
Sounio authority transition, and its own hostile controls.

## Stop Rules

Stop and preserve `DENY447` if the experiment requires an effect family outside
the frozen twelve, a semantic outcome not produced by Sounio, an enforcement
rule that cannot be installed before release, a path-only authorization check,
a blacklist for unknown operations, same-UID bearer authority, an inherited
unclassified descriptor, an unbounded `ioctl`, missing actual-instance facts,
non-causal sabotage, incomplete teardown, or production activation.

If one twin sabotage does not cross its named boundary, the corresponding
treatment is not causally proven. If one treatment crosses, the cell is not
closed. If Sounio still refuses the complete receipt, the material layer may not
reinterpret or override the refusal.

## Novelty Boundary

The proposed novelty is not seccomp, Landlock, DynamicUser, pidfd, cgroups, or
effect typing separately. It is a Sounio-authored, closed-world effect contract
materialized as a same-process denial-carrying principal, coupled to a complete
per-family causal twin matrix and an affirmative extinction witness. The
scientific claim, if the gate passes, is deliberately narrow: one frozen
Sounio payload on one named Linux boundary had no undeclared effect route under
the exact frozen policy and generation.

## Nonclaims

- This Garden does not prove that Linux has only twelve syscall classes.
- It does not prove that telemetry can observe every effect.
- It does not claim that seccomp or Landlock is a semantic authority.
- It does not establish portability beyond the named architecture and kernel.
- It does not authorize arbitrary commands, shells, interpreters, plugins, or
  dynamically linked user payloads.
- It does not make the current `complete_effects=false` receipt positive.
- It does not open LOOM launch, recycle, Exec/Bash, write, commit, CI, parity,
  or claim-ready surfaces.
- It does not permit C++, OCaml, shell, an LLM, or a UI to produce the expected
  result.

## Immediate Next Artifact

Create the deterministic Sounio policy-plan executable as the child of this
Garden. Freeze its source, executable, plan output, both parent manifests,
toolchain, command, and expected treatment/sabotage matrix before adding the
native effect cell. Until then, the authoritative material answer remains:

```text
SOUNIO_EFFECT_CLOSURE_DENY code=447 reason=material-coverage-incomplete
complete_effects=false
material_execution=false
```

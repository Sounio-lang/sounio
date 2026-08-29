# Garden: product Exec ingress v1

Status: preregistered before product attachment or execution.

## Research question

Can LOOM attach the existing `Exec/Bash` hook path to frozen Sounio action
`9030` without allowing a hostile process with the same UID, executable,
worktree, environment, or harness ancestry to mint or consume authority?

This Garden adds no semantic state, transition, result, or expected Sounio
decision. It defines the product attachment experiment for already frozen
objects.

## Frozen authority

The authority order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> MATERIAL_PARITY -> PRODUCT_ATTACHMENT`

The product experiment must load and hash-bind these exact parents before it
accepts any request:

- Sounio `ExecGrantCell` action `9030`, manifest SHA-256
  `8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051`;
- host distinct-principal grant, manifest SHA-256
  `8c0851bb5e0f2f1982ec220d3e335bfd8c41e6b0500a763c02a3f1901c834ac5`;
- host process-witness core, manifest SHA-256
  `eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00`;
- Sounio action-`9025` material judgment, manifest SHA-256
  `f7adafcd1c79364b75ebe48b66999ec2d7b82a12d6b8e45d9c1cc4637a4ca9ca`;
- product launch dark attachment, manifest SHA-256
  `99dece63a1c0da72e35afca90402ec5d678b7a034758983e4331d27750e23e3c`.

Sounio remains `SEMANTIC_AUTHORITY`. OCaml may implement the resident product
kernel and effect lifecycle. C++20, Linux, BPF LSM, and systemd remain
transitory `MATERIAL_PARITY`. No Python or Rust process may execute anywhere
in the admission, grant, launch, outcome, or receipt path.

## Current counterexample

The current product kernel authenticates `EXEC_ISSUE`, `EXEC_CONSUME`, and
`EXEC_OUTCOME` with one bearer value read from the session `capability` file.
It additionally checks `SO_PEERCRED`, pidfd liveness, start tick, boot and PID
namespace, executable hash, worktree, argv shape, and descent from the harness.
Those checks refuse a same-UID process outside the harness ancestry, but do not
distinguish a genuine hook from a hostile command that invokes the same OCaml
binary with a fabricated hook event from inside that ancestry.

Therefore the existing route remains `exec_attached=false`. A test that only
repeats the outside-ancestry refusal is insufficient evidence for this Garden.

## Product ingress object

The object under test is a one-request `ExecIngress` descriptor lineage owned
by a LOOM harness generation. It has no pathname, serialized bearer token, or
environment representation.

For one genuine `PreToolUse` event, the harness creates a private socketpair.
The provider-side endpoint remains in the trusted harness cell. The other
endpoint crosses exactly one `execve` into the native OCaml hook and is closed
on every unrelated child path. The kernel binds its accepted message to:

1. the inherited descriptor identity and peer credentials;
2. harness pidfd, start tick, boot identity, PID namespace, and generation;
3. the exact hook pidfd, start tick, executable digest, and event digest;
4. the command, cwd, environment, claim scope, and action-`9030` frame digest;
5. one host `DynamicUser` principal generation and descriptor barrier;
6. one outcome obligation and affirmative extinction receipt.

The descriptor is necessary but not sufficient. Product execution requires the
conjunction of ingress, Sounio `ALLOW`, OCaml linear consumption, the frozen
host distinct-principal grant, process-witness readiness, and an open
descriptor barrier. Possession of any textual receipt, handle, argv, token,
socket path, or copied event is non-authorizing.

The first product implementation may run in dark mode. In dark mode it must
observe the full conjunction and emit a receipt, but it must never execute the
requested user command.

## Treatment

The treatment uses the current material observation and the real native hook
path. Frozen Sounio action `9030` returns its applicable denial, the ingress is
retired, no host barrier opens, and the command sentinel remains absent. The
product continues only according to the preregistered dark-mode rule.

No test fixture may substitute a positive frame for the treatment.

## Causal sabotages

Each sabotage changes exactly one fact. All other source bytes, hashes,
processes, descriptors, command bytes, and deadlines remain identical.

1. **Ingress-release sabotage:** the trusted harness transfers the exact
   generation-bound descriptor once. The dark observer may reach
   `INGRESS_ACCEPTED`; the command still must not execute.
2. **Ingress-omission treatment:** the descriptor is closed without transfer.
   The same request must return `INGRESS_REFUSED`.
3. **Same-ancestry forgery:** a hostile child uses the same UID, OCaml binary,
   worktree, environment, harness ancestry, argv shape, token file, and copied
   hook JSON, but lacks the inherited ingress descriptor. It must be refused
   before grant lookup. This is the load-bearing control for the ingress rule.
4. **Copied-descriptor hostile principal:** a descriptor number or copied
   textual record is presented from the wrong principal/cgroup. It must be
   refused. If the kernel permits descriptor extraction across principals, the
   experiment is falsified.
5. **Authority-removal sabotage:** the ingress is valid but the action-`9030`
   resident response is withheld. The barrier must close and no success may be
   inferred from silence.

The first two rows prove descriptor causality. The third proves that the new
ingress rule, rather than the older ancestry or executable rule, causes the
refusal. A gate that cannot tell rows 1 and 3 apart must fail closed.

## Mandatory negative controls

1. A deliberate Python oracle attempts to create a sentinel. Admission must
   refuse before Python executes and the sentinel must remain absent.
2. The same control is repeated for Rust.
3. Replaying an accepted ingress generation refuses.
4. Reusing an `ExecGrantCell` handle refuses.
5. A wrong command, cwd, environment, worktree, claim scope, principal,
   cgroup, action manifest, or kernel generation refuses before barrier write.
6. Kernel, Guardian, resident, broker, hook, or harness death revokes the
   generation and materializes an incomplete outcome when required.
7. Timeout, EOF, malformed frames, missing policy, unreadable receipts, and
   audit failure all refuse.
8. A legacy Python hook bridge cannot issue, consume, close, or promote a
   product grant.
9. An external LLM review or parity receipt cannot become semantic authority.
10. The public broker protocol continues to refuse direct `EXEC`, `LAUNCH`,
    `BARRIER_RELEASE`, and raw ingress-mint commands.

## Receipt

Every decision records `ALLOW` or `DENY`, its reason, and:

- Sounio source and frozen semantic hashes;
- language producer and language role;
- resident, operational, and material manifest hashes;
- toolchain and hardware identity;
- harness, hook, principal, cgroup, pidfd/start, and generation digests;
- command, cwd, environment, ingress, grant, outcome, and terminal receipt
  digests;
- whether Python or Rust executed;
- whether a user command, write, commit, or CI action occurred.

No secret, command body, environment value, raw handle, or descriptor value is
written to the journal.

## Acceptance boundary

A passing dark experiment may establish:

- `product_exec_ingress_observed=true`;
- `same_ancestry_forgery_refused=true`;
- `non_bearer_product_ingress=true`;
- `action_9030_product_observed=true`.

It must retain:

- `production_activation=false`;
- `material_execution=false`;
- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.

Changing `exec_attached` requires a later, separately preregistered activation
gate executed on the real provider path with the full hostile matrix and an
identity rollback.

## Lane contract

Semantic-Lane-ID: `loom-product-exec-ingress-20260829`

Concept-ID: `SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL`

Semantic-Owner: `founder`

Semantic-Change: `none; product attachment only`

Claims-Allowed: `dark non-bearer ingress observation and causal refusal`

Claims-Forbidden: `live execution authority, production activation, Exec/Bash attachment, commit attachment, CI attachment`


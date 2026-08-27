# Loom Native Agent Hook V1

`sounio-loom agent-hook` is the native OCaml realization of the existing Codex
and Claude coordination hook. It extends the existing Loom executable; it is
not a second coordinator or semantic kernel.

## Authority Chain

Every event is admitted in this order:

1. parse the hook event with a bounded, duplicate-key-refusing JSON parser;
2. locate the current worktree and frozen language-authority manifest;
3. require the pinned manifest hash
   `5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e`;
4. re-hash the Sounio module, entrypoint, semantic bundle, and compiled Sounio
   authority runtime;
5. submit frame `9020` as OCaml `OPERATIONAL_REALIZATION` requesting
   `GUARD_ENFORCE`, with the exact frozen Sounio parent hash;
6. append and `fsync` the resulting `ALLOW` or `DENY` decision;
7. only after `ALLOW`, execute the lifecycle or structured-write coordination
   operation.

Policy absence, manifest drift, source drift, runtime drift, malformed JSON,
duplicate JSON keys, timeout, a pathless write, a cross-repository path, a path
into a sibling worktree, or a coordination refusal exits nonzero before the
requested write is performed.

## V1 Coverage

- Session claim, heartbeat, process presence, delivery endpoint, and release.
- Directed inbox injection receipts.
- `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, and `apply_patch` path scopes.
- A durable, self-describing TSV decision journal under the repository shared
  Git directory. Every receipt binds the Sounio source and frozen-semantics
  hashes to the producing language and role, semantic authority, OCaml
  toolchain, hardware, command descriptor and event hash, exact Sounio result,
  final `ALLOW`/`DENY`, and reason.
- Atomic runtime packaging of the OCaml hook and its frozen Sounio authority.
- A durable `NATIVE_HOOK_ATTESTED` capability that proves the currently loaded
  session actually crossed the native OCaml hook boundary. This is operational
  evidence only: it is never `PARITY_OPEN` or `CLAIM_READY`.

`scripts/ci/sounio_loom_native_hook_selftest.sh` exercises lifecycle and write
round trips, policy/runtime sabotage, strict and duplicate-key JSON refusal,
pathless, outside-repository, and sibling-worktree writes, receipt completeness,
log-redirection refusal, and executable sentinels that prove the hook invoked
neither Python nor Rust.

## Generation-Bound Wake Start

Runtime `2026.08.27.31` separates terminal transport from agent execution for
tmux endpoints. A durable wake advances through:

1. `prepared`: the generation-bound submission exists, but Enter has not
   succeeded;
2. `submit-uncertain`: persisted before Enter, so a crash cannot fabricate a
   submission receipt;
3. `submitted`: Enter succeeded and `submitted_utc` is not earlier than the
   confirmed `inserted_utc`;
4. `started`: the native prompt hook injected that message into the same live
   endpoint generation.

Insertion is also fail-closed. The runtime persists `insertion_state=uncertain`
before `send-keys -l`. A retry may recognize the exact message id in the tmux
pane and continue with Enter, but it never writes the full prompt again after
an uncertain external effect. A successor endpoint cannot confirm a predecessor
submission. After a successor registers its own generation, the retry supervisor
creates a fresh submission and removes the obsolete pending marker only after
the successor hook starts it.

Session start ensures the native retry supervisor. Short lock contention waits
for a bounded interval; timeout still refuses explicitly and performs no state
mutation. `WAKE_STARTED` is therefore stronger than `WAKE_DELIVERED`: it is a
generation-bound hook receipt, not evidence that bytes reached a terminal.

This stronger handshake currently applies to tmux delivery. `agentd` and
`loom` transports retain their separate adapter-confirmed transport receipts;
they must not be reported as generation-bound prompt starts.

### Loaded-Hook Attestation

Runtime `2026.08.27.32` additionally gates terminal insertion and start
promotion on a current native-hook capability. A session without that
capability remains `prepared`, with `insertion_state=not-attempted` and zero
attempts. The durable message remains pending, but the runtime writes no bytes
to the terminal. If the message is acknowledged before a native session takes
ownership, acknowledgement atomically cancels the pending wake submission.

Capability registration is accepted only when all of these identities agree:

- the immediate parent is the manifest-pinned OCaml Loom executable;
- Loom's caller is the exact live session process recorded by presence,
  including PID, process start time, boot id, PID namespace, executable path,
  and executable hash;
- the OCaml producer and coordination runtime hashes match the active immutable
  bundle manifest;
- the running coordination executable is the active shared runtime, with the
  same runtime id and source hash recorded by the capability;
- the endpoint generation, worktree, harness, and session id still match.

The production Codex predicate accepts the exact `codex` executable name.
Claude accepts the exact native `claude` executable, or its pinned Node CLI
shape. Broad names such as a code-mode host cannot mint capability. Local-mode
selftests may emit `NATIVE_HOOK_ATTESTED`, but always with `wake_eligible=0`;
they cannot create production wake authority.

Installing a new runtime does not upgrade an already loaded session. Old
capabilities become invalid when the active bundle changes, and sessions that
started on the Python bridge or an older hook must drain and restart from a
worktree whose SessionStart loads `bin/sounio-loom agent-hook`. A positive live
canary must therefore use a freshly started real session; a file-on-disk check,
inbox read, or Python shim is not evidence of native-hook activation.

Runtime `2026.08.27.33` also supports an independent primary checkout using the
same installed coordination kernel and durable bus. Set both
`SOUNIO_COORD_RUNTIME_DIR` and `SOUNIO_COORD_DIR` to the shared runtime and state
roots before starting the harness. This is required when a CLI intentionally
inherits hook declarations from a linked worktree's control checkout. The
override changes hook discovery isolation, but not authority: registration and
every later wake revalidation still require the exact active bundle path,
manifest, source hash, and binary hashes.

Runtime `2026.08.27.34` completes the native endpoint migration for tmux. The
OCaml hook derives the socket and pane from its inherited tmux environment,
asks tmux for the pane id and current path, and requires that path to resolve to
the exact session repository before registration. The coordination runtime then
revalidates the socket, pane, process, harness, worktree, and generation before
any wake. Missing or inconsistent tmux identity creates no endpoint and cannot
be promoted by the native-hook capability alone.

Runtime `2026.08.27.35` adds an executable native tmux fixture. It launches an
exact `codex` harness process in a real tmux pane, crosses the OCaml SessionStart
and UserPromptSubmit boundaries, and requires a generation-bound `WAKE_STARTED`
with `injected=1`, `wakes=1`, and `wake_pending=0`. Separate panes prove that a
nonexistent pane and a pane whose current directory is not the session repository
create no endpoint. The fixture's wake exception requires local runtime mode,
both native-hook selftest flags, and an exact temporary state-root shape; its
capability remains `wake_eligible=0` and cannot authorize a production wake.

Runtime `2026.08.27.36` makes native adoption independent of the lane's Git
lineage. Each immutable shared bundle contains a language-authority capsule with
the frozen manifest, exact Sounio source, and exact Sounio entrypoint. The
bundle manifest binds all three hashes independently, and activation rechecks
them. A worktree that contains the policy continues to use and validate its
local frozen bytes; a worktree with no LOOM product tree falls back to the
capsule beside the executing runtime. Hook configurations resolve `current`
once, then pass the matching capsule root and executable from that same physical
bundle. A concurrent runtime activation can therefore cause an explicit
attestation refusal, but cannot combine policy from one generation with OCaml
from another. Decision receipts record `semantic_authority_origin=worktree` or
`runtime-capsule`. The sabotage gate modifies only the packaged Sounio source
and requires `Sounio-authority-source-hash-mismatch`, proving that the capsule
binding itself causes the refusal.

Runtime `2026.08.27.37` binds the detached obligation-supervisor wrapper to its
effective `SOUNIO_COORD_DIR`. An independent control checkout can therefore
rediscover the same PID-1-parented leader even when its own Git common directory
is different. The policyless hook fixture crosses `SessionStart` and
`SessionEnd` under one exact `codex` process, requires the supervisor to remain
live, checks the wrapper environment for the exact state root, and then stops it
from that independent checkout. Omitting the binding reproduces the
cross-checkout refusal and makes this gate fail.

## Deliberate Boundary

V1 does not yet attach to Exec/Bash. Treating all shell commands as semantic
producers would make Loom unusable, while letting OCaml invent a distinction
between mechanical transport and an oracle would violate Sounio authority. The
next Sounio policy version must define that distinction and freeze it before the
native hook adds Exec/Bash matchers. Until then, the hook migration can cover
structured writes and lifecycle truthfully, but no repository-wide no-Python
enforcement claim is allowed.

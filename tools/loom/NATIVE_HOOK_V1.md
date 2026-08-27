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

## Deliberate Boundary

V1 does not yet attach to Exec/Bash. Treating all shell commands as semantic
producers would make Loom unusable, while letting OCaml invent a distinction
between mechanical transport and an oracle would violate Sounio authority. The
next Sounio policy version must define that distinction and freeze it before the
native hook adds Exec/Bash matchers. Until then, the hook migration can cover
structured writes and lifecycle truthfully, but no repository-wide no-Python
enforcement claim is allowed.

# Loom Sovereign Execution Kernel Product Attachment V1

Status: `IMPLEMENTED_PRODUCT_CANDIDATE`

## Result

Loom execution is attached to frozen Sounio action 9042. A provider hook may
submit an execution request, but it never receives a bearer token, capability
handle, file descriptor, or release primitive. The OCaml kernel creates an
in-memory single-use grant only after Sounio admission, binds the caller with
kernel-observed process identity, registers the material worker with the
HostGuardian, consumes the grant atomically, and only then releases the worker.

The product path is:

```text
provider CLI
  -> native PreToolUse hook
  -> frozen Sounio pre-execution authority
  -> tokenless LOOM_EXEC/1 START
  -> SO_PEERCRED + pidfd + start tick + executable + ancestry
  -> in-memory non-bearer ExecGrant
  -> HostGuardian material registration
  -> atomic grant consumption and worker release
  -> material execution with parent-death revocation
  -> frozen Sounio outcome authority
  -> tokenless LOOM_EXEC/1 WAIT
  -> validated result presentation
```

## Authority Boundary

Sounio is `SEMANTIC_AUTHORITY`. Frozen action 9042 defines admission,
Guardian-death revocation, and the condition under which production activation
is permitted. OCaml is the operational kernel. Linux supplies material facts
through `SO_PEERCRED`, pidfds, `/proc` start ticks, process ancestry, and
`PR_SET_PDEATHSIG`.

The GUI, TUI, CLI, Pod, tmux pane, coordination transport, and provider hook
have zero execution-release authority. They may request work, display results,
or disappear. Only the HostGuardian registration followed by kernel-local
atomic consumption can release material execution.

## Non-Bearer Grant

The grant exists only in the kernel process. It is indexed by a digest over the
kernel generation, authenticated peer identity, event, command, and complete
request payload. The identifier is never returned to the provider and is
removed before the worker receives its one-byte start signal. Replay therefore
has no transferable authority object to present.

The harness environment intentionally omits `SOUNIO_LOOM_TOKEN_FILE`. The
legacy `EXEC_ISSUE`, `EXEC_CONSUME`, and outcome control operations refuse in a
sovereign-required session.

## Survival And Revocation

Provider, presenter, GUI, coordination transport, Pod, and tmux loss do not
revoke an already registered material witness. The Guardian remains alive until
registered material processes become extinct, and the kernel retains the
durable result for a later presenter.

True HostGuardian loss is different. The worker observes the Guardian pidfd and
start tick, kills the material child, records `GUARDIAN_REVOKED`, and refuses to
promote the result as completed. This is fail-closed revocation rather than
transport coupling.

## Product Gates

The product selftest proves:

- end-to-end provider execution completes and is presented;
- the worker is reaped and no runtime process remains;
- the bounded `/usr/bin/sleep` leaf used by the causal controls is measured as
  an immutable, root-owned ELF and is never an interpreter or forwarding
  wrapper;
- a hostile same-UID relay is refused before material registration;
- provider transport loss does not interrupt a registered material witness;
- true HostGuardian death revokes the material child;
- no bearer material reaches the provider;
- Python and Rust oracle traps are not executed.

The freeze gate binds this behavior to the source files, frozen semantic and
material manifests, Sounio runtime, product contract, and both product gates.
The shared runtime installer additionally copies and verifies the same capsule
and the action-9042 executable before atomic activation.

Setting `SOUNIO_LOOM_PRODUCT_RUNTIME` makes the product gate exercise an
installed Loom binary directly. The gate still builds its non-authoritative
provider fixture, but it neither rebuilds nor substitutes the selected product
runtime; the installed Loom must find the frozen action-9042 executable beside
it in the activated capsule. This installed test mode creates only a temporary,
explicitly selftest capability with `wake_eligible=0`; it cannot become a live
delivery endpoint or authorize production wake transport.

Cold start and causal-event observation each have a bounded 30-second budget.
Timeout still refuses the operation; the wider budget prevents a loaded host
from being misclassified as a semantic or custody failure.

## Claim Boundary

This attachment makes execution product-active:

```text
same_uid_peer_isolation=true
production_activation=true
exec_attached=true
```

It does not yet claim that structured writes, commits, or every CI runner pass
through this kernel:

```text
write_attached=false
commit_attached=false
ci_attached=false
claim_ready=false
```

Those are subsequent enforcement surfaces. They may consume sovereign results,
but they may not redefine action-9042 semantics or restore bearer authority.

# LOOM Product ExecIngress Dark Attachment V1

Status: `IMPLEMENTED_DARK_CANDIDATE`

## Result

The native `PreToolUse` execution path now observes a non-bearer inherited
Unix-stream descriptor before it can call `Loom_exec.authorize_and_issue`.
This is a dark product attachment. It does not authorize execution, set
`exec_attached=true`, or replace the frozen action-9030 grant lifecycle.

The exact path is:

```text
native agent-hook
  -> structural ExecIngress check
  -> inherited Unix stream, one request, one response, EOF
  -> event SHA-256 plus command SHA-256 binding
  -> Sounio action 9031 current product-activation projection
  -> dark decision receipt
  -> existing ExecGrant path only after a Sounio DENY
```

An action-9031 `ALLOW` is causal sabotage while the attachment is dark. The
hook records it and refuses before grant issuance. A current-material Sounio
`DENY` is recorded and the existing product path continues.

## Authority Boundary

Sounio remains semantic authority. OCaml implements transport, descriptor
custody, hashing, peer observation, ordering, receipts, and fail-closed
attachment.

A missing or invalid descriptor is a structural precondition, not a Sounio
decision. Its receipt therefore states:

```text
decision_authority=OCaml-structural-precondition
sounio_evaluated=false
semantic_authority=Sounio
```

Only a decision actually returned by the resident Sounio action 9031 states:

```text
decision_authority=Sounio
sounio_evaluated=true
producing_language=Sounio
language_role=SEMANTIC_AUTHORITY
```

This distinction prevents an OCaml transport refusal from being laundered
into semantic authority.

## Descriptor Protocol

`SOUNIO_LOOM_EXEC_INGRESS_FD` is a descriptor locator, not authority. The
value is accepted only when it names an already-open connected Unix stream.
The hook sets close-on-exec, observes `SO_PEERCRED`, and consumes the stream
with this exact bounded exchange:

```text
LOOM_EXEC_INGRESS/1<TAB>event_sha256<TAB>command_sha256<LF>
LOOM_EXEC_INGRESS_BOUND/1<TAB>event_sha256<TAB>command_sha256<LF>
EOF
```

The echoed hashes must match exactly. Trailing bytes, early EOF, timeout,
wrong binding, non-socket descriptors, disconnected sockets, dead peers, and
self peers refuse. The descriptor is closed after the exchange and cannot be
reused by the hook.

Production admission requires a peer UID distinct from the hook UID. The local
positive fixture permits a same-UID socket only when all three facts hold:

1. hook test mode is active;
2. `PROBE_ONLY` prevents every grant call;
3. the live parent is the same LOOM executable with `exec-ingress-probe` in its
   kernel-observed argv.

The gate independently attempts to carry the fixture flag outside
`PROBE_ONLY`; that attempt refuses before the descriptor exchange.

## Product Modes

With no descriptor and no required-mode flag, the hook emits a structural
dark DENY and preserves existing behavior. This makes current deployments
observable without claiming enforcement.

With `SOUNIO_LOOM_EXEC_INGRESS_REQUIRED=1`, descriptor absence refuses before
presence refresh, grant lookup, or capability issuance. An attempted invalid
descriptor always refuses, even when required mode is not explicitly set.

The environment flag is not sufficient to forge admission: it only locates an
already inherited open file description. A fresh same-UID socket and a copied
hook JSON event both fail the required gate.

## Causal Controls

The dark selftest proves:

- the descriptor-bound action-9031 current projection returns DENY and does
  not enter the ExecGrant path;
- the action-9031 `seal` projection returns ALLOW, causing this product hook to
  refuse before grant issuance;
- a same-UID self-broker refuses;
- the same-UID fixture cannot escape `PROBE_ONLY`;
- copied JSON without the inherited descriptor refuses in required mode;
- Python and Rust command attempts refuse before either executable runs;
- no command sentinel and no legacy execution-authority log are created;
- the source order places `Loom_exec_ingress.observe` before
  `Loom_exec.authorize_and_issue`;
- OCaml contains no copied Sounio expected-result strings.

The older accepted forged-hook path remains preserved as a historical
counterexample at commit `eb853be79be289deb596bea0b3ab8a042509d8df`.

## Nonclaims

This phase does not prove that the live provider harness selectively inherits
the descriptor only into genuine hook processes. It does not attach the
root-owned host broker, a distinct-UID DynamicUser execution cell, arbitrary
command materialization, or the ProcessWitness outcome closure to product
execution.

The current kernel control request still reads `SOUNIO_LOOM_TOKEN_FILE`, and
the current command supervisor still executes the admitted command as the
workspace UID. The historical Python compatibility bridge also remains in the
repository for stale launchers, although the codex-1 hook configuration itself
is native.

Therefore all of these remain false:

```text
distinct_uid_product_broker=false
material_execution=false
production_activation=false
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
```

## Next Stage

The next product step is not another local peer heuristic. The existing host
principal broker must mint the connected descriptor from a root-owned service,
the genuine provider/hook principal must receive it, and `exec-capability` must
materialize the command in a distinct DynamicUser cell. The cell must return a
ProcessWitness outcome and affirmative extinction receipt before the action-
9030 outcome can close.

Only that attachment can change `distinct_uid_product_broker`,
`material_execution`, or `exec_attached` from false.

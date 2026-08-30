# Garden: LOOM ExecIntent Envelope V1

Status: `GARDEN_PREREGISTERED`

## Preserved phrase

> The provider event is evidence. The Sounio intent is identity.

## Question

The first product ExecCell canary returned the frozen action-9033 result through
the native hook, but its provider JSON contains a per-session nonce. A test-only
environment override currently substitutes the frozen event digest. That is a
valid bridge for the exact canary, not a semantic identity contract.

Can Sounio define a stable, typed intent projection before OCaml or the host
material layer binds a provider event to an ExecCell result?

## Authority order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> OCAML_PROJECTION
-> PROVIDER_LIFECYCLE_ATTACHMENT
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. OCaml may parse provider JSON, hash the raw
bytes, and realize the frozen projection. C++20, Linux, and systemd may measure
the inherited descriptor and principal transition. None may invent the
semantic event digest or an expected result. Python and Rust are forbidden.

## V1 projection

V1 is deliberately closed over the already frozen calibration command. It
binds:

- the frozen action-9031 activation manifest;
- the frozen action-9033 result manifest;
- the canonical `LOOM_EXEC_INTENT/1` domain and field order;
- present agent, lane, session, cwd, provider-hook, and execution-tool
  identities;
- a raw provider-event digest that remains separate audit evidence;
- the frozen semantic event digest consumed by action 9033;
- the frozen calibration command digest;
- an inherited, one-shot, authenticated descriptor bound to the lane
  principal, with no pathname authority;
- Sounio-only expected-result authority and affirmative oracle absence.

The raw event digest is never promoted into semantic identity merely because
its bytes arrived first. V1 does not claim arbitrary-command support.

## Falsifier

The load-bearing rule is exact command-digest equality. The unchanged
wrong-command witness must be refused by the real source. A source mutant that
deletes only that rule must admit the same witness. If it does not, the claimed
rule is not causal and the freeze fails.

## Nonclaims

- `ocaml_projection_attached=false`
- `provider_lifecycle_attached=false`
- `arbitrary_command_projection=false`
- `exec_attached=false`
- `production_activation=false`
- `parity_open=false`
- `claim_ready=false`

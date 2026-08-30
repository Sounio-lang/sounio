# LOOM Handshake ↔ ExecCell Payload Coherence V1

Status: `COHERENCE_EXECUTABLE`

This is not a Garden seed and not a host freeze. It observes already-frozen
parents. It does not attach a product path, flip dark ExecIngress, or
re-measure drifted C++/OCaml sources.

## Question

Does the product ExecCell canary still name the frozen Sounio
`ProcessWitness` handshake payload as the bytes it would execute, and does
product ExecIngress remain descriptor-optional (`required_mode_default=false`)
while the live ExecCell freeze is red?

## Observed parents

Read-only, already present on
`lane/codex-1/loom-mainline-20260827` at `d7d0b16c7a`:

- `tools/loom/process_witness_handshake_payload.freeze.v1`
  (`stage=SOUNIO_HANDSHAKE_PAYLOAD_FROZEN`, action `9030`);
- `tools/loom/product_exec_cell_fixture.freeze.v1`
  (`stage=SEMANTICS_FROZEN`, action `9030`);
- `tools/loom/product_exec_cell_host_canary.runtime.v1`
  (`stage=MATERIAL_EXEC_CELL_CANARY_FROZEN`, action `9030`, test-only);
- `tools/loom/product_exec_ingress_dark.runtime.v1`
  (`stage=PRODUCT_DARK_ATTACHMENT_FROZEN`, action `9031`).

Garden parents, not rewritten here:

- `tools/loom/GARDEN_PROCESS_WITNESS_EXEC_HANDSHAKE_V1.md`
- `tools/loom/GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md`
- `tools/loom/GARDEN_PRODUCT_EXEC_INGRESS_V1.md`

## Load-bearing equalities

1. Handshake freeze `executable_sha256` equals ExecCell canary
   `payload_sha256` and ExecCell fixture `payload_sha256`.
2. SHA-256 of the handshake freeze file equals ExecCell canary
   `process_witness_manifest_sha256` and fixture `payload_manifest_sha256`.
3. Both child manifests name
   `tools/loom/process_witness_handshake_payload.freeze.v1`.
4. SHA-256 of the fixture freeze file equals ExecCell canary
   `fixture_manifest_sha256`.

These are recorded-identity checks. The gate does not rebuild the handshake
ELF, does not hash `loom_kernel_principal_broker.cpp`, and does not hash the
host-canary `.inc` files. Codex-1 currently owns those sources; their live
hashes have drifted from the ExecCell freeze pins. Re-pinning them without a
new host measurement would be a false freeze.

## Gate-5 ratchet

`GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md` acceptance gate 5 is
"make descriptor absence fail closed for product execution tools".

The dark ExecIngress freeze does not pin `required_mode_default`. The
counterexample gate already records `required_mode_default=false`, but that
gate is about the same-UID `fork/exec` composition, not about payload
identity.

This gate therefore pins:

- `product_exec_ingress_dark.runtime.v1` `required_mode_default=false`

It must stay false while any of the following remain true:

- live ExecCell host-canary freeze is red (source hash drift);
- the historical canary `exec_cell_attached=true` is `test_only=true`;
- product `exec_attached=false`, `production_activation=false`.

Flipping `required_mode_default` is Codex-1's product-ingress work, not this
lane. A later activation Garden must preregister that flip.

## Historical canary is not product attachment

The ExecCell host canary records `exec_cell_attached=true` together with
`test_only=true` and `production_activation=false`. That is a bounded host
measurement. It does not satisfy Garden gates 4–8 and does not authorize
`parity_open` or `claim_ready`.

## Falsifier

The gate fails if the handshake digest and the ExecCell canary digest
diverge, if either child manifest stops naming the handshake freeze, if the
handshake freeze file hash diverges from the child's pinned manifest hash, or
if `required_mode_default` becomes true.

## Boundary

```text
Semantic-Lane-ID: loom-handshake-exec-cell-payload-coherence-20260830
Owner: grok-cli1
Concept-IDs: none
Intent-Preserved: ExecCell canaries execute the frozen Sounio ProcessWitness handshake payload; dark ExecIngress stays optional until a live ExecCell freeze and a later activation Garden say otherwise
Transformation: none
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: recorded payload identity across handshake freeze, fixture freeze, and ExecCell canary runtime is live; required_mode_default remains false
Claims-Forbidden: parity_open, claim_ready, required_mode_default=true, product exec_cell_attached, product exec_attached, production_activation, re-freeze of drifted host hashes, live handshake ELF rebuild, replacement of same-UID fork/exec
Assumptions: Codex-1 continues to own the host-canary .inc files, broker.cpp, loom_exec_ingress.ml, and loom_hook.ml
Write-Set: this contract, the coherence selftest, and the evidence receipt
Read-Set: the four parent manifests named above
Positive-Witness: bash scripts/ci/sounio_loom_handshake_exec_cell_payload_coherence_selftest.sh
Negative-Witness: a handshake executable_sha256 that does not equal canary payload_sha256; required_mode_default=true
Acceptance-Gate: bash scripts/ci/sounio_loom_handshake_exec_cell_payload_coherence_selftest.sh
Integration-Target: lane/codex-1/loom-mainline-20260827
Authoritative-Only-If: never; observational only
```

Bash compares recorded fields and file hashes. It is not a semantic oracle.
Python and Rust are forbidden.

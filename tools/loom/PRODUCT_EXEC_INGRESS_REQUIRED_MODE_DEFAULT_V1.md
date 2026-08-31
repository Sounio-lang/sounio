# LOOM Product ExecIngress required_mode default V1

Status: `RATCHET_EXECUTABLE`

This is not a Garden seed and does not activate gate 5. It binds the live
OCaml default of `Loom_exec_ingress.required_mode` to the recorded dark
ingress field `required_mode_default=false`.

## Question

Does product ExecIngress still treat a missing inherited descriptor as
optional (`None | Some "0" -> false`), while the fail-closed branch
(`Some "1" -> true` then `failf "product-exec-ingress-descriptor-absent"`)
exists but is not the default?

## Observed answer

Yes. `tools/loom/src/loom_exec_ingress.ml` still defaults off.
`product_exec_ingress_dark.runtime.v1` still records
`required_mode_default=false`. Garden gate 5
("make descriptor absence fail closed for product execution tools")
therefore remains open.

Flipping the OCaml default, or the runtime field, without replacing
same-UID `fork/exec` (gate 4) is forbidden on this lane.

## Falsifier

The gate fails if `required_mode` is no longer a unique toplevel let, if
unset/`0` no longer returns `false`, if the descriptor-absent fail-closed
string disappears, or if the runtime field becomes `true`.

## Boundary

```text
Semantic-Lane-ID: loom-exec-ingress-required-mode-default-20260831
Owner: grok-cli1
Concept-IDs: none
Transformation: none
Claims-Introduced: OCaml required_mode default is false and matches the dark runtime field
Claims-Forbidden: required_mode_default=true, gate 5 closed, parity_open, claim_ready
Write-Set: this contract, the selftest, and the evidence receipt
Read-Set: tools/loom/src/loom_exec_ingress.ml, tools/loom/product_exec_ingress_dark.runtime.v1
```

Does not edit `loom_exec_ingress.ml`. Python and Rust are forbidden.

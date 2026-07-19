# Madaros Method Checker Authority

These fixtures pin method lookup to receiver identity before method spelling.
For a real struct or enum, that identity is `(TypeKind, Name, defining
ModuleId)`. Visibility is applied to matching receiver candidates before the
checker decides whether the visible set is unique.

The focused matrix is:

- A: `ambiguous_remote_{,reversed_}main.sio` receives the target module's
  `MethodAuthority` while a second module declares a public same-spelled type
  and an `IO` method. Both import orders must check successfully; selecting the
  distractor would produce `E035` in the effect-free caller.
- B: `private_distractor_{,reversed_}main.sio` loads a public target method and
  a private method on the same nominal receiver. The inaccessible candidate must
  not poison the unique visible candidate, in either import order.
- C: `private_cross_module_main.sio` calls the only matching method from outside
  its defining module and must reject exactly once with `E175`.
- D: `ambiguous_same_receiver_main.sio` exposes two visible methods on the same
  nominal receiver and must reject exactly once with `E219`, never `E011`.
- E: `method_body_contract_main.sio` imports an uncalled method whose declared
  `i64` return contract is violated. It must reject exactly once with `E008`,
  proving definition bodies bind their exact collected signature.

`public_unique_main.sio`, `local_precedence_main.sio`, the existing private
function/struct/enum fixtures, and the associated-path facade witness remain
compatibility controls.

```text
Semantic-Lane-ID: method-checker-module-authority-r1
Owner: Codex method checker lane
Concept-IDs: SOUNIO-MODULE-CLOSURE-AUTHORITY
Intent-Preserved: same spelling is not nominal identity; inaccessible candidates do not create visible ambiguity
Transformation: TyNamed carries defining ModuleId when known; method lookup returns FOUND, PRIVATE, AMBIGUOUS, or MISSING after receiver and visibility filtering
Types-Changed: TyNamed reuses TypeEntry.fn_sig_id as a variant-discriminated defining ModuleId payload
Effects-Changed: call-site effect checking consumes only one visible signature for the exact receiver; method bodies bind their exact module-and-source-span declaration contract independently
IR-Changed: none
Claims-Introduced: focused source-fresh checker gate proves A-E and retained compatibility controls
Claims-Forbidden: canonical selective-import bindings, forward-reference backpatch, collision-free cross-module generic mangling, lowering parity, method ABI, SOIR preservation, or general ModuleGraph completion
Assumptions: real nominal definitions are collected before signatures that need their identity; unknown identities retain a bounded textual fallback
Write-Set: self-hosted/check/types.sio, self-hosted/check/compat.sio, self-hosted/check/defs.sio, self-hosted/check/check.sio, scripts/ci/madaros_method_checker_authority_gate.sh, tests/compiler/madaros_method_checker_authority
Read-Set: self-hosted/check/mod.sio and existing visibility fixtures
Positive-Witness: A and B in both import orders, plus public/local/associated controls
Negative-Witness: C=E175 exactly once; D=E219 exactly once; E=E008 exactly once
Acceptance-Gate: scripts/ci/madaros_method_checker_authority_gate.sh
Integration-Target: codex/modulegraph-main-integration-r3
Authoritative-Only-If: one source-fresh Madaros ELF passes the focused gate without fallback
```

<!-- docs:meta
topic_id: repo.docs.audit.imported-type-named-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.imported-type-named-2026-08-23
-->

# Imported type-position names are `ty_named`, not TyUnknown

```text
Semantic-Lane-ID: imported-type-named-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: a named type in a signature is that type; Unknown is
  not a universal sink; Madaros is the claim oracle
Transformation: checker_lower_named_type_mut treats TyUnknown env
  bindings as uninformative even when the struct is not yet in the
  local table, matching the by-value lower_named_type spine
Types-Changed: none (resolution of existing named types)
Effects-Changed: none
IR-Changed: none
Claims-Introduced: `fn forge() -> Epistemic { 1.0 }` is E008 under
  Madaros; WP25 DD HVP LO cannot be returned as Epistemic
Claims-Forbidden: Madaros is fixed-point-verified; every imported name
  is fully typed; this closes all D3/D4 import holes
Assumptions: use-item collection still binds imported names as
  TyUnknown for E137 suppression; type position must not honour that
Write-Set: self-hosted/check/check.sio, the three tests, this file
Read-Set: self-hosted/check/check.sio lower_named_type (by-value spine)
Positive-Witness: tests/run-pass/imported_epistemic_pin_ok.sio
Negative-Witness: imported_epistemic_is_not_a_float.sio;
  a_mu_wp25_dd_hvp_lo_is_not_epistemic.sio
Acceptance-Gate: rebuilt Madaros check of the two compile-fails is
  verdict=1 with Epistemic in the diagnostic; pin_ok exit 0; existing
  a_mu_gum_split example still rc=0
Integration-Target: origin/main
Authoritative-Only-If: produced by a Madaros ELF built from this source
```

The in-place named-type lowerer kept a TyUnknown import binding when the
struct was absent from the local table. The by-value spine already fell
through to `ty_named`. Multi-module check uses the in-place path, so
imported `Epistemic` was a universal sink.

## Receipts (2026-08-23)

Rebuilt Madaros from this source: `/tmp/madaros-imported-type-named-out/madaros`
(100641310 B). Control ELF 100902241 B still accepts
`forge() -> Epistemic { 1.0 }` (verdict=0). New ELF:

| program | verdict |
|---|---|
| `imported_epistemic_is_not_a_float.sio` | **1** E008 expected Epistemic found f64 |
| `a_mu_wp25_dd_hvp_lo_is_not_epistemic.sio` | **1** E008 expected Epistemic found Wp25DdHvpLoAbsent |
| `imported_epistemic_pin_ok.sio` check+run | **0** / rc=0 |
| `ew_precision.sio` check | **0** |
| `a_mu_gum_split.sio` check | **0** |

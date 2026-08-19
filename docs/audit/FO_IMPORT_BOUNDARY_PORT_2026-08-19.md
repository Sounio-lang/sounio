<!-- docs:meta
topic_id: repo.docs.audit.fo-import-boundary-port-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.fo-import-boundary-port-2026-08-19
-->

# Semantic lane — first-order uncertainty across an imported helper

This declaration precedes the compiler edit. It is the contract for WS-A1,
authorized 2026-08-19. The axis is not the dissertation: `Knowledge<T>` is in
121 stdlib+examples files and the uncertainty dies at the third function.

```text
Semantic-Lane-ID: WS-A1-fo-import-boundary-20260819
Owner: grok-cli5
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE, SOUNIO-ORDERED-PATH-PROVENANCE
Intent-Preserved: a incerteza atravessa uma fronteira de funcao sem ser apagada
Transformation: A first-order uncertainty attached to an epistemic numeric
  value remains attached when that value is produced by a pure 1- or 2-argument
  helper defined in another compilation unit and consumed in the caller. Crossing
  a module is an ordered path, not a license to drop the uncertainty axis.
  Same-file 1-2 argument transfer is unchanged. Same-file or imported helpers
  with three or more parameters remain outside this transformation.
Types-Changed: none
Effects-Changed: none
IR-Changed: none (FO_XFER table already exists; this lane populates it from
  loaded modules before seed body lower)
Claims-Introduced: An imported 1- or 2-argument pure helper preserves
  first-order variance across the call. Witness: tests/run-pass/gum_fo_import_boundary.sio
  XPASS on Madaros.
Claims-Forbidden: "a FO esta corrigida" while the arity >=3 hole remains open;
  "o lean_single e o oraculo"; imported fo_css (5-arg) is fixed; same-file add3
  is fixed; every Knowledge<T> number in the 121 files is now trustworthy.
Assumptions: fo_register_pure_fn_transfer still skips >2 params; frontend
  prepass must not be wiped by fo_xfer_global_reset on the bodies path;
  DCE filter of imported items does not delete reachable 1-2 arg helpers.
Write-Set: self-hosted/ir/lower.sio, self-hosted/compiler/module_frontend.sio,
  this file, governance registry rows for this file
Read-Set: tests/run-pass/gum_fo_import_boundary.sio,
  tests/run-pass/gum_fo_arity3_boundary.sio,
  tests/run-pass/gum_cross_function.sio,
  tests/run-pass/lean_is_not_oracle_scale2.sio,
  tests/run-pass/lean_is_not_oracle_product.sio,
  tests/run-pass/lean_is_not_oracle_add.sio
Positive-Witness: gum_fo_import_boundary (2-arg imported fo_add2, peel=5, imp=5)
Negative-Witness: gum_fo_arity3_boundary (same-file add3 stays 0);
  madaros_gum_fo_import combined pin stays red (calls 5-arg fo_css)
Acceptance-Gate: Madaros compile+run of gum_fo_import_boundary exits 0 with
  OBSERVED/IMP in (4.9, 5.1); harness XPASS under SOUNIO_XPAS_FATAL=1;
  gum_cross_function still PASS; the three lean_is_not_oracle_* pins still PASS
  on Madaros.
Integration-Target: origin/main at 6f23dfe1da
Authoritative-Only-If: the acceptance gate holds on an unpatched current-source
  Madaros ELF. A lean_single pass of the import witness is not authority.
```

## Semantic-Outcome

Recorded after the current-source Madaros ELF
`artifacts/self-hosted/madaros` (100553240 bytes, built 2026-08-19T03:10:12Z
from this worktree at 6f23dfe1da + the port) compiled and ran the read-set.

```text
Semantic-Outcome:
Concept-Status-Before: an imported 1- or 2-argument pure helper returned a
  numeric value whose first-order variance had been erased (IMP=0 while the
  same-file peel of a+b was 5). Crossing a module was treated as a license
  to drop SOUNIO-EPISTEMIC-NUMERIC-VALUE. The ordered path of the import
  was not a provenance-preserving path.
Concept-Status-After: the same imported helper (epistemic::fo::fo_add2)
  returns the value with first-order variance still attached (IMP=5.000000,
  PEEL=5.000000). Crossing a module is an ordered path that keeps the
  uncertainty axis for 1- and 2-argument pure helpers.
Distinctions-Added: module boundary != uncertainty erasure, for the 1-2
  argument case only.
Distinctions-Preserved: uncertainty != ignorance; compile success !=
  runtime parity; arity >=3 still erases (ADD3=0); lean_single is not an
  oracle (#1927 pins still print MADAROS_RIGHT).
Distinctions-Erased: none
Evidence-Run:
  artifacts/self-hosted/madaros build+run:
    gum_fo_import_boundary  PEEL=5.000000 IMP=5.000000 OK rc=0
    gum_cross_function      var(sum)=5 var(scaled)=16 PASS rc=0
    lean_is_not_oracle_scale2   OBSERVED=0.010000 OK rc=0
    lean_is_not_oracle_product  OBSERVED=0.032500 OK rc=0
    lean_is_not_oracle_add      OBSERVED=5.000000 OK rc=0
    gum_fo_arity3_boundary  ADD2=5 ADD3=0 ZERO rc=1
    gum_fo_across_call      CALL_var=0 ZERO rc=1
  harness (SOUNIO_XPAS_FATAL=1, this ELF):
    gum_fo_import_boundary  XPAS + XPAS_FATAL rc=1
    gum_fo_arity3_boundary  known-failure (still red) rc=0
    gum_cross_function      PASS
    lean_is_not_oracle_*    PASS
Fallback-Path: none used. Proof is the rebuilt current-source ELF, not
  lean_single and not bin/madaros-linux-x86_64.
Legacy-Kept: fo_register_pure_fn_transfer still skips >2 params
  (self-hosted/ir/lower.sio:8698). fo_xfer_global_reset still runs on
  items_mut / items_ref. bodies_ref and bodies_dedup do not reset.
  known-failure tag on gum_fo_import_boundary left in place so the
  #1910 gate accuses; dropping it is a one-line follow-up, not this
  write-set. Witness files were not edited.
Conflicting-Lanes: grok-cli2 WS-A2 (fo bytecode) claimed
  MADAROS_FO_CALL_BOUNDARY_DISPATCH and fo_call_boundary_div.sio — not
  this write-set. No coord conflict on lower.sio / module_frontend.sio.
Next-Semantic-Interface: arity >=3 (same-file add3, imported fo_css).
  That hole is open. This lane does not authorize lifting it.
```

The acceptance gate holds. Authoritative-Only-If is satisfied. The claim
introduced is only the imported 1–2 argument case. FO is not fixed.

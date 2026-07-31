<!-- docs:meta
topic_id: repo.docs.research.fo-pk-residual4-oral-css-closeout-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.fo-pk-residual4-oral-css-closeout-2026-07-31
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# FO residual §5.4 — oral Css closeout (2026-07-31)

**Status:** **ORAL_CSS_CLOSED** under the layered stack below  
**Executable end-to-end:** R4 `fo_pk_import_method_driver_gate.sh`  
**Stack gate:** `scripts/ci/fo_residual4_stack_gate.sh`  
**Detail spec:** `docs/research/fo_css_compiler_residual_half_spec_2026-07-31.md`

---

## 1. What is closed (oral Css pure-helper fragment)

| Layer | Lean module | Gate |
|-------|-------------|------|
| L0 Algebraic | `SounioFoCssSurfaceParity` | `fo_css_surface_parity_gate` |
| L1 FoExpr | `SounioFoSurfaceTransfer` | `fo_surface_transfer_gate` |
| L2 FO stack | `SounioFoBytecodeFragment` | `fo_bytecode_fragment_gate` |
| L2 pure-emit | `SounioFoEmitPure` | `fo_emit_pure_gate` |
| L2 FO_XFER expand | `SounioFoRegistrationFragment` | `fo_registration_fragment_gate` |
| L2 multipass register | `SounioFoEngineInstallFragment` | `fo_engine_install_fragment_gate` |
| L2 method peel | `SounioFoMethodXferFragment` | `fo_method_xfer_fragment_gate` |
| L2 multi-mod prepass model | `SounioFoMultimodFragment` | `fo_multimod_fragment_gate` |
| L2 live Madaros | — | R4 import↔method driver |

All freezes share **Var(Css) = 191/240** at default seeds.

## 2. What remains open (full engine)

1. Multipass soundness for **arbitrary** pure programs (not just fo_css trio).
2. Multi-mod **loader** correctness for all modules / name resolution.
3. Method FO_XFER + op17 for **arbitrary** methods (not just oral Css peel model).
4. EXP op-20 vs eEta channel formal distinction (FO-var equivalent at η=0).

These do **not** block dissertation citation of R1–R4 oral Css FO freezes.

## 3. Dissertation citation (EN-UK)

> First-order GUM for oral steady-state Css is surface-independent under Madaros
> FO: algebra, FoExpr desugar, FO bytecode, pure compile, multipass FO_XFER
> expand/register (including reverse definition order), method peel, and multi-mod
> registry name-identity are machine-checked for the pure-helper fragment;
> executable parity of import/method/call-result/site freezes is R4
> (`fo_pk_*_gate.sh`, 2026-07-31). Full multipass/multi-mod/method soundness for
> arbitrary programs remains open.

## 4. Re-run

```bash
bash scripts/ci/fo_residual4_stack_gate.sh
```

Expected final lines:
```
FO_RESIDUAL4_STACK_GATE_OK
STATUS ... L2_METHOD_XFER=CLOSED L2_MULTIMOD=CLOSED L2_FULL_ENGINE=OPEN
ORAL_CSS_RESIDUAL4_CLOSED
```

---

*Closeout fo-pk-residual4-oral-css-v1 (2026-07-31).*

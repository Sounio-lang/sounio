<!-- docs:meta
topic_id: repo.docs.research.ast-native-claims-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ast-native-claims-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# AST-native claims — type-safe claim literals in Sounio (preprocessor path)

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parents:** `docs/research/falsification_ledger_spec_2026-07-25.md` (L_GREEN), `docs/research/zero_provenance_claims_spec_2026-07-25.md` (Z_GREEN)  
**Harness:** `scripts/research/claim_ast_preprocessor.py`  
**Gate:** `scripts/ci/claim_ast_gate.sh`  
**Claim struct:** `stdlib/epistemic/claim_ast.sio`

---

## 1. What this is

The Falsification Ledger currently scans `// @claim` comments. This rung makes claims **type-safe Sounio values**: a `Claim` struct with fields for hypothesis, falsifier, evidence, harness, gate, verdict, and optional provenance. A preprocessor converts `claim` blocks into `const` struct literals before compilation, so the compiler type-checks claims like any other value.

This is **not** a parser change. The parser surface is under active work by other agents; adding a new `ItemKind` would require touching `self-hosted/parser/items.sio` and every pattern-match on `ItemKind` across the compiler. The preprocessor path delivers AST-native claims without that risk.

---

## 2. Syntax

### Source form (preprocessor input)

```sounio
claim rupture_lstm_no_annihilation {
    hypothesis = "trained LSTM exhibits composed annihilation",
    falsifier  = "orientation_scramble produces equivalent gap_dominance",
    evidence   = Evidence::InstrumentControlled,
    harness    = "scripts/research/rupture_ord2_trained_lstm_probe.py",
    gate       = "scripts/ci/rupture_abcd_contracts_gate.sh",
    verdict    = Verdict::Negative,
    provenance = Provenance::Annihilated,  // optional
}
```

### Preprocessor output (compiler input)

```sounio
const rupture_lstm_no_annihilation: Claim = Claim {
    hypothesis: "trained LSTM exhibits composed annihilation",
    falsifier: "orientation_scramble produces equivalent gap_dominance",
    evidence: Evidence::InstrumentControlled,
    harness: "scripts/research/rupture_ord2_trained_lstm_probe.py",
    gate: "scripts/ci/rupture_abcd_contracts_gate.sh",
    verdict: Verdict::Negative,
    provenance: Option::Some(Provenance::Annihilated),
}
```

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **A1_STRUCT_DEFINED** | `stdlib/epistemic/claim_ast.sio` defines `Claim`, `Evidence`, `Verdict`, `Provenance`. | Type-checks with `bin/souc check`. |
| **A2_PREPROCESSOR_IDENTITY** | Running the preprocessor on a file without `claim` blocks leaves it unchanged. | Diff is empty. |
| **A3_PREPROCESSOR_ROUNDTRIP** | A `claim` block converts to a valid `const Claim` literal that type-checks. | `bin/souc check` passes. |
| **A4_CLAIM_COUNT** | The preprocessor preserves claim count and field values. | Generated literal matches input fields. |
| **A5_NO_PARSER_TOUCH** | No file under `self-hosted/parser/` is modified. | `git diff --name-only` shows no parser files. |

---

## 4. What this is NOT

- **Not a parser change.** Deferred until the compiler surface stabilises.
- **Not runtime semantics.** Claims are compile-time metadata; they do not affect generated code.
- **Not a replacement for the comment ledger.** Both can coexist; the comment scanner remains the first-pass ledger.

---

## 5. Future rungs

1. **Native parser support** — `claim` as a real `ItemKind` once the compiler surface is stable.
2. **LSP hover** — show claim metadata in the editor.
3. **Cross-check** — link claims to gate results automatically.

---

## 6. Reproduce

```bash
python3 scripts/research/claim_ast_preprocessor.py < input.sio > output.sio
bash scripts/ci/claim_ast_gate.sh
# expect: CLAIM_AST_GATE_OK
```

---

## 7. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.

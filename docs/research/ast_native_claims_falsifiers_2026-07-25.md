<!-- docs:meta
topic_id: repo.docs.research.ast-native-claims-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ast-native-claims-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# AST-native claims — falsifiers and stop rules

**Companion to:** `docs/research/ast_native_claims_spec_2026-07-25.md`  
**Harness:** `scripts/research/claim_ast_preprocessor.py`

---

## Clause-level falsifiers

### A1_STRUCT_DEFINED

**Falsifier:** `stdlib/epistemic/claim_ast.sio` fails to type-check.

**Stop rule:** Fix the struct/enums before proceeding.

---

### A2_PREPROCESSOR_IDENTITY

**Falsifier:** The preprocessor modifies a file without `claim` blocks.

**Stop rule:** The preprocessor is too aggressive; restrict it to `claim` blocks only.

---

### A3_PREPROCESSOR_ROUNDTRIP

**Falsifier:** The generated `const Claim` literal fails to type-check.

**Stop rule:** The preprocessor or struct definition is wrong.

---

### A4_CLAIM_COUNT

**Falsifier:** The preprocessor drops or duplicates claims, or mangles field values.

**Stop rule:** Fix the transformation.

---

### A5_NO_PARSER_TOUCH

**Falsifier:** Any file under `self-hosted/parser/` is modified.

**Stop rule:** Revert; parser changes are out of scope for this rung.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| Preprocessor breaks existing .sio files | `A_RED` | Do not commit. |
| Generated literal fails type-check | `A_RED` | Fix before commit. |
| Parser file modified | `A_RED` | Revert immediately. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.

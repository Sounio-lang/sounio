<!-- docs:meta
topic_id: repo.docs.research.native-claim-parser-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.native-claim-parser-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Native parser support for claims — `ItemKind::ItemClaim`

**Date:** 2026-07-25  
**Status:** `EXECUTABLE`  
**Parents:** `docs/research/ast_native_claims_spec_2026-07-25.md` (A_GREEN, future rung 1), `docs/research/falsification_ledger_spec_2026-07-25.md` (L_GREEN)  
**Parser:** `self-hosted/parser/items.sio` (`parse_claim_item`)  
**AST:** `self-hosted/parser/ast.sio` (`ItemKind::ItemClaim`, `ClaimDecl`, claim registry)  
**Test:** `tests/run-pass/claim_native_basic.sio`  
**Gate:** `scripts/ci/claim_native_gate.sh`

---

## 1. What this is

Future rung 1 of the AST-native claims spec, landed: `claim` blocks are now parsed natively by the self-hosted compiler. A `claim` block is compile-time metadata for the falsification ledger — it carries **no runtime semantics** and never reaches resolve, check, lower, or codegen.

```sounio
claim rupture_lstm_no_annihilation {
    hypothesis = "trained LSTM exhibits composed annihilation",
    falsifier  = "orientation_scramble produces equivalent gap_dominance",
    evidence   = Evidence::InstrumentControlled,
    verdict    = Verdict::Negative,
    trial_count = 3,
    threshold  = 0.95,
}
```

Field values may be string literals, integer/float literals, bare identifiers, or paths (`Evidence::InstrumentControlled`). Fields may be separated by commas, semicolons, or newlines; trailing separators are allowed. Empty claim bodies are legal.

## 2. Design decisions

- **`claim` is not a lexer keyword.** The parser detects the identifier `claim` followed by `Ident` and `{` at item level (same pattern as `module`). Existing code using `claim` as an ordinary identifier is unaffected; the only hijacked token sequence was previously a parse error.
- **`ItemKind::ItemClaim` is filtered at the parse driver.** `parse_claim_item` records the `ClaimDecl` in a global claim registry (`ast_record_claim`, flat primitive arrays following the `GLOBAL_VAR_INIT_*` pattern) and returns an `ItemClaim` item; `parser/mod.sio` then drops it from the item stream. Downstream passes therefore require no changes — important because `check/check.sio` and `resolve/imports.sio` are under active edit by another lane. Defensive `ItemClaim => self` arms were added to both `resolve/resolve.sio` matches.
- **The `Item` struct is unchanged.** Adding a payload field would have required editing ~85 construction sites across 11 files; the side-table registry avoids that churn.
- **Field text is stored as `ast_name_hash` digests.** Recovering source text from the registry is a future rung; the comment ledger and the `claim_ast` preprocessor remain the text-preserving extraction paths.

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **N1_PARSER_SURFACE** | `ItemKind::ItemClaim`, `ClaimDecl`, and `parse_claim_item` exist in parser sources. | `scripts/ci/claim_native_gate.sh` grep guards. |
| **N2_STREAM_FILTER** | The parse driver drops `ItemClaim` items and resets the claim registry. | grep guards in gate. |
| **N3_OLD_COMPILER_REJECTS** | The pre-change checked-in compiler rejects claim syntax (proves the test exercises the new path). | `bin/madaros check` fails with old ELF; skipped once the prebuilt is refreshed. |
| **N4_NEW_COMPILER_RUNS** | A madaros rebuilt from current source compiles and runs `claim_native_basic.sio`, printing `CLAIM_NATIVE_OK`. | Gate builds (or reuses) `artifacts/self-hosted/madaros-claim-native` and runs the test. |

## 4. What this is NOT

- **Not type-checking of claims.** Field values are parsed, not type-checked. The preprocessor path (`claim_ast`) remains the type-safe claim representation.
- **Not ledger extraction.** The registry records claim presence, field counts, and hash digests only; it is not yet wired to the falsification ledger report.
- **Not a keyword reservation.** `claim` remains a valid identifier outside the `claim NAME {` item pattern.

## 5. Reproduce

```bash
bash scripts/ci/claim_native_gate.sh
# expect: CLAIM_NATIVE_GATE_OK
```

## 6. AI disclosure

Spec and implementation drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.

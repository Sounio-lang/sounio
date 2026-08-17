<!-- docs:meta
topic_id: repo.docs.audit.token-table-ceiling-refusal-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.token-table-ceiling-refusal-2026-08-17
-->

# Token-table ceiling — fail-closed refusal (E229)

**Date:** 2026-08-17  
**Lane:** grok-cli5 / `token-ceiling-refusal`  
**Code:** `error[E229]`  
**Precedent:** E219 (`extern "C"` unimplemented → refuse, do not fabricate)

---

## Defect

Madaros lexer/parser token storage is fixed at **2097152** slots
(`PARSER_TOKENS` / `PT_*` in `self-hosted/parser/parser.sio`, emits in
`self-hosted/lexer/mod.sio`). When `tc` reached the wall:

1. `parser_set_token_flat` / `parser_set_token_at` **no-op** past the index.  
2. The lexer **kept walking** source and could still return a large `token_count`.  
3. Out-of-range reads synthesized **Eof**.  
4. The parser then failed on a **truncated** stream and blamed a line that was
   never the cause.

Separately, `lex_file_to_globals` / `lex_source_to_globals` **clipped** source
length to 2097152 bytes with the same silence.

A compiler that discards its own input and misattributes the error is exactly
the honesty failure this language exists to make impossible.

Dispatch stated **41 run-pass tests** blocked by this class of failure. This
note does not re-census those 41 under a fresh Madaros build (heavy); the
witness below is the structural reproduction.

---

## Fix (not “bigger table”)

| Change | Role |
|--------|------|
| `lexer_push_flat` | Emit only if `tc < 2097152`; else mark overflow and return −1 |
| Loop break + `return -1` | Never hand the parser a truncated stream |
| Source length `> 2097152` | Refuse (kind=2), do not clip |
| `lex_file_parse_items` | Print **E229**, `parser_set_last_errors(true, 1)`, return `None` |
| `scripts/ci/token_table_ceiling_gate.sh` | Generates 2097152-comma witness; requires E229 + nonzero rc |

**Raising the ceiling without refusal is explicitly rejected** even if it would
green the 41 tests: the next input past the new wall would lie again.

---

## Should the cap also move? (measurement, not convenience)

| Fact | Implication |
|------|-------------|
| Table is **2²⁰** slots × several parallel arrays (kind, start, end, line, col, …) | Memory is multi‑MB **per process** already |
| Source buffer is the **same** 2097152-byte wall | Byte clip and token wall interact |
| Product modules that need >2M tokens are a **modularity** smell | Prefer split compilation units |
| Historical raise 1M→2M without rebuild left a **stale binary** lying (PL adoption audit) | Ceiling changes without rebuild/refusal are hazardous |

**Recommendation:** **Keep 2097152 for now.** Ship **E229 first**. Revisit a
raise only with: (1) measured peak `token_count` on the blocked 41 and on
`self-hosted/compiler/main.sio` multi-module loads, (2) memory budget on the
pod/CI, (3) **E229 retained** at the new bound. Do not raise “so the tests
pass” alone.

---

## Witness (measured)

### Baseline (prebuilt Madaros, before this fix)

```bash
TOKEN_CEILING_EXPECT=baseline_silent bash scripts/ci/token_table_ceiling_gate.sh
```

| Witness | Measured on prebuilt `bin/souc` |
|---------|--------------------------------|
| **W1** valid `main` + pad to 2097152 bytes + trailing `fn should_not_be_dropped` | **`check: OK` rc=0** — trailing source past the byte wall was **silently clipped** |
| **W2** 2097152 commas | rc=1, floods `parse error: unrecognised item start`, **no E229** |

W1 is the smoking gun: the compiler discarded input and reported success.

### After rebuild from this source

```bash
bash scripts/ci/build_modular_madaros.sh "$PWD/artifacts/self-hosted/madaros"
MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros bash scripts/ci/token_table_ceiling_gate.sh
```

| Witness | Measured after fix |
|---------|-------------------|
| W1 | **error[E229]** source exceeds lexer byte buffer, rc≠0 |
| W2 | **error[E229]** lexer token table full, rc≠0 |

Gate result: `TOKEN_TABLE_CEILING_GATE_OK`.

Witnesses avoid brace-bisect traps: W1 is a complete valid prefix plus pad plus a
complete trailing item past the wall; W2 is only commas (complete punctuation
tokens), not unbalanced items.

---

## Bisect warning (from dispatch)

If you must bisect a large real module to study this class of failure: balance
`{}` `()` `[]` and delete **complete** items only. Unbalanced deletion produces
a **different** bug and has wasted time here before.

---

## Files

- `self-hosted/lexer/mod.sio` — overflow state, `lexer_push_flat`, report, parse gate  
- `docs/llm-guide/explanations/E229.md`  
- `scripts/ci/token_table_ceiling_gate.sh`  
- this note

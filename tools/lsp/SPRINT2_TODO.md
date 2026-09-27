# sounio-lsp Sprint 2 — deferred parity items

**Status 2026-07-25: all four items LANDED (v0.4.0, see `CHANGELOG.md`).
Probe coverage lives in `tools/lsp/test_protocol.sh` (T9/T10/T11/T12).**
The design notes below are kept for historical context.

These four items remain from the 2026-05-17 audit's patch list. They
were intentionally not addressed by the parity-patches sprint
(`feature/lsp-parity-patches`, v0.3.0) because each requires a longer
design discussion and a budget exceeding the one-week cap for
audit-driven fixes.

| # | Item | Estimate | Notes |
|---|------|----------|-------|
| 5 | `textDocument/semanticTokens/full/delta` with `resultId` tracking | 2 days | Needs a per-document token cache + diff algorithm against the previous `resultId`. Currently advertised provider has `full: true, range: true` only; clients fall back to `/full` requests. |
| 6 | `typeHierarchyProvider` (`prepareTypeHierarchy` + `typeHierarchy/subtypes` + `typeHierarchy/supertypes`) | 3–5 days | Sounio's surface for "type" hierarchies is `struct` + `enum` + `algebra` rather than nominal inheritance; a meaningful hierarchy requires deciding what `subtypes` means in Sounio's effect-tracked, algebra-bearing system. Design discussion first. |
| 7 | Cross-file `workspace/symbol` over unopened files | 2–3 days | Current implementation only indexes `didOpen`'d buffers (verified by the audit). Needs a filesystem walk over `rootUri` for `*.sio` + on-disk parse (no buffer text) + symbol extraction. The single-file scanner already used by `documentSymbol` should be reusable. |
| 8 | Promote `textDocumentSync` from `Full` (kind=1) to `Incremental` (kind=2) | 3–5 days | The largest item. Requires `textDocument/didChange` to accept incremental `contentChanges[]`, plus a byte-edit pipeline on `DOC_TEXT`. Performance gain at scale; correctness equivalent today since clients re-send full docs anyway. |

## Order suggestion

If Sprint 2 has time for all four, do them in this order:

1. **#5 semanticTokens/full/delta** — fastest single win, contained
   change, validates the resultId cache pattern that #7 might reuse.
2. **#7 cross-file workspace/symbol** — high user value (jump-to-symbol
   in unopened files), reuses the existing documentSymbol scanner.
3. **#8 incremental sync** — large but well-scoped; landing it before #6
   means the type-hierarchy work has the better sync model underneath it.
4. **#6 typeHierarchyProvider** — bracket the design call before the
   first line of code.

## Acceptance

Each item is "done" when the byte-accurate audit probe at
`/tmp/lsp_audit/probe_v2.py` reports `RESPONDS_OK` for the relevant
methods *and* the response payload is structurally valid per LSP 3.17.

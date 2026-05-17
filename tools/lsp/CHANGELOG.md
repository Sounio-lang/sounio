# sounio-lsp changelog

## v0.3.0 — 2026-05-17 — Parity patches

Closes four items flagged by the 2026-05-17 adversarial parity audit.
Verification: byte-accurate dispatch probe (39 methods) reports
**32 RESPONDS_OK / 7 METHOD_NOT_FOUND / 0 NO_RESPONSE**.
All 7 METHOD_NOT_FOUND are either advertised-as-unsupported
(`resolveProvider:false`, `workspaceDiagnostics:false`) or explicitly
Sprint-2 deferred (`typeHierarchy/*`, `semanticTokens/full/delta`).

### Patch #1 — `codeAction` kinds: 1 → 3

`initialize.capabilities.codeActionProvider.codeActionKinds` now
advertises `["quickfix", "source.fixAll", "refactor.extract"]`. The
scanner records every fix site into `FIX_OFF` / `FIX_LEN` / `FIX_TY` and
emits, after the scan:

- one `quickfix` action per recorded site (existing behaviour),
- one `source.fixAll` action batching every recorded site into a single
  `WorkspaceEdit`, when at least one fix site exists,
- one `refactor.extract` action wrapping the selection in
  `let __extracted = … \n    __extracted`, when the selection range is
  non-empty and contains at least one non-whitespace byte.

Verified: a fixture containing `let mut y = x` + `y + 1;` returns
`['quickfix', 'quickfix', 'source.fixAll', 'refactor.extract']`.

### Patch #2 — `codeLens` was always working

The original audit flagged `textDocument/codeLens` + `codeLens/resolve`
as `NO_RESPONSE`. Re-investigation showed the audit's probe parser was
slicing the response body in *Python characters* rather than *bytes*,
so the 3-byte UTF-8 "▶" in the Run lens caused the parser to over-read
into the next frame and lose the codeLens response. Byte-accurate
re-probe shows `textDocument/codeLens` returns
`{title: "▶ Run", command: "sounio.run", arguments: [<uri>]}` for every
`fn main()` declaration. `codeLens/resolve` correctly returns
`METHOD_NOT_FOUND`, consistent with `resolveProvider: false`.

No code change; phantom finding documented in
`agent_logs/CC1_parity_patches_convergence.md`.

### Patch #3 — `inlayHint/resolve` returns the hint, not null

Replaced the unconditional `result:null` with a balanced-brace copy of
the request's `params` object verbatim into `result`. String literals
inside the hint are tracked so `{`/`}` inside JSON strings do not
disturb depth. Editors that gate rendering on resolve now unblock.
Verified: input
`{position, label, kind, paddingRight}` echoes byte-identical.

### Patch #4 — `textDocument/semanticTokens/range`

New method `M_SEM_TOKENS_RANGE = 55`, dispatched alongside
`M_SEM_TOKENS`. `respond_semantic_tokens_range`:

1. Parses `params.range` into `ST_RANGE_S_LINE` / `ST_RANGE_S_CHAR` /
   `ST_RANGE_E_LINE` / `ST_RANGE_E_CHAR` via the existing in-order
   `"line"`/`"character"` walk used by `ca_parse_range`.
2. Sets `ST_RANGE_ACTIVE = 1`.
3. Calls the existing `respond_semantic_tokens` — `st_emit` checks the
   active range and skips tokens whose start position is outside
   `[(s_line, s_char), (e_line, e_char))`. Skipped tokens do not update
   `ST_PREV_*`, so the delta encoding for emitted tokens stays valid.
4. Resets `ST_RANGE_ACTIVE = 0`.

`initialize.capabilities.semanticTokensProvider` now advertises both
`"full": true` and `"range": true`.

Verified on a 10-function fixture: `/full` returns 52 tokens (260 ints);
`/range` for `lines 0..3` returns 15 tokens (75 ints); first delta-line
is 0, matching the range's start line.

### Out of scope (Sprint 2 — see `SPRINT2_TODO.md`)

Patches 5–8 from the audit's patch list — `semanticTokens/full/delta`,
`typeHierarchy*`, cross-file `workspace/symbol` over unopened files,
and `textDocumentSync` Incremental promotion — are deliberately not
addressed in this sprint.

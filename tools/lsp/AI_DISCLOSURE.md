# sounio-lsp — AI authorship disclosure

Per the GAIDeT/ICMJE-2025 contributor-statement guidance for
agent-assisted software development.

## v0.3.0 — parity patches (2026-05-17)

Implemented by Claude Code (Opus 4.7) operating as agent **CC-1** under
direct human supervision (operator: @agourakis82). All four patches
were audited against the binary's actual behaviour before being
declared complete; the audit probe is reproducible at
`/tmp/lsp_audit/probe_v2.py`.

Scope of this reactivation:

- `dataset_builder` / `evaluation` were NOT touched. The MultiPL-E
  surface remains as CC-2 / Cx-2 landed it.
- The four LSP patches modified `self-hosted/lsp/server.sio` only.
  No other files in `self-hosted/` were touched.
- The binary `bin/sounio-lsp` was rebuilt via
  `./bin/souc self-hosted/lsp/server.sio bin/sounio-lsp` (the raw
  pass-through; `souc compile -o` remains broken in 1.0.0-beta.5 — see
  `coordination/HANDOFFS.md`).

The v0.3.0 parity patches (2026-05-17) are on `main`.

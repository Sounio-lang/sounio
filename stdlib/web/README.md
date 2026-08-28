# stdlib/web

Web utilities: HTTP session types, WebSocket messages, epistemic HTTP.

## Key Modules
- `http.sio`: re-exports the canonical `HttpRequest`/`HttpResponse` types
  and helpers from `stdlib/http/http.sio` (consolidated 2026-07-01 — this
  module used to define its own duplicate structs; see `stdlib/http/mod.sio`
  for the canonical public surface).
- `websocket.sio`: `WsMessage` + text/close message constructors and predicates.
- `epistemic_http.sio`: `DataSource`/`EpistemicResponse` — HTTP results wrapped
  with provenance trust levels and uncertainty bounds (H7.2).

## Tests

`tests/stdlib/web/test_web.sio` (check-only, Madaros gate — native multi-module
run hits a pre-existing IR-lowering wall unrelated to this module's logic).

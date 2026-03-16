---
name: F# Interop Phase 1
description: Sounio ↔ F# IPC interop via SNIO binary protocol (Sprint A)
type: project
---

Sprint A complete (2026-03-12): Sounio ↔ F# interop Phase 1 — SNIO binary pipe protocol.

**Why:** User develops programs in both Sounio and F# and needs cross-language function calls for scientific computing.

**What was built:**
- `self-hosted/interop/protocol.sio` — SNIO binary wire protocol (magic "SNIO", msg types: CALL_FUNC/RESULT/ERROR/SHUTDOWN, LE i64 payloads)
- `self-hosted/interop/server.sio` — IPC server with built-in functions: ping, dot_product, sum, vec_add, vec_scale
- `--serve` flag in `self-hosted/compiler/main.sio` (banner suppressed, early dispatch before println)
- `interop/fsharp/Sounio.Interop/` — F# library: Protocol.fs + SounioProcess.fs (IDisposable wrapper, typed CallF64/CallScalar/DotProduct/VecAdd etc.)
- `interop/fsharp/Example/` — F# example program
- `interop/test_protocol.py` — Python test harness

**Status:** All .sio files pass `souc check`. F# projects build with dotnet 8.0. End-to-end test blocked by JIT OOM (known: main.sio JIT compilation takes 14-35GB RSS). Will work once native-compiled souc binary is available.

**How to apply:** Next phases need `.o` emission (Phase 1B) and FFI wiring (Phase 2). The SNIO protocol is the interim bridge until shared library interop is ready.

**Key patterns:**
- Effect annotations: array access requires `Panic`, `%` requires `Div`, `str_from_bytes` requires `Panic`
- No forward refs: helper functions must precede callers
- Ownership: can't `&!` borrow then `&` borrow same var in same scope — use separate functions or local copies
- .NET installed at `~/.dotnet` (PATH="$HOME/.dotnet:$PATH")

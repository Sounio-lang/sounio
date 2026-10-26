ChangedTests opts into one raw-modular compile with a fixed 30s compilation
budget, then executes the same verified ELF under the existing native timeout.
Default harness callers keep their route. Retain per-phase logs, status and
identity receipts; reject compiler errors even when an ELF was emitted.

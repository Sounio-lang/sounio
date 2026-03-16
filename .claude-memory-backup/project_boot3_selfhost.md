---
name: boot3 self-hosting progress
description: Bootstrap compiler boot3.sio self-hosting status and known bugs
type: project
---

boot3.sio (935 lines) — ≤5-param Sounio→x86-64 compiler designed for self-hosting.

**Status**: Stage0→boot3 works perfectly (hello, fib, all tests). Self-compilation produces 109KB/47-function ELF but crashes at SIGSEGV address 0xb.

**Bugs found and fixed**:
1. **st[1] sync**: find_local read stale token position after expression statements. Fix: add `st[1] = t` before find_local calls in assignment handler.
2. **mmap size**: i64 arrays mmapped `arr_count` bytes instead of `arr_count * 8`. Fix: multiply by 8 when is_arr==2.
3. **Multi-line comments**: skip_ws only handled 1 comment. Fix: loop while is_comment.

**Current crash**: SELF→hello segfaults at address 0xb after all syscalls succeed (file read, all mmaps). Crash is during compilation phase, likely in st_set/collect_fns/compile_all. Address 0xb suggests a small integer (11) being dereferenced as pointer.

**Architecture**: All state in st: [i64; 32768]. Token data in td: [i64; 262144]. ≤5 params per function. Slot-based arg passing (no push/pop). CAFE forward-call patching.

**Key files**: bootstrap/boot3.sio, bootstrap/stage0.c

**Why:** Self-hosting bootstrap eliminates JIT dependency (28GB→<1MB memory).
**How to apply:** Debug the SEGV at 0xb, likely in compile_primary's variable reference or array indexing codegen.

<!-- docs:meta
topic_id: repo.docs.compiler.async-design
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.async-design
-->

# Async Runtime Design

**Status**: Implemented — Phase 1 (OS Thread Model) + Phase 3 (sleep/join) complete  
**Date completed**: 2026-04-20  
**All 11 async tests pass**: `async_basic`, `async_spawn`, `async_channels`, `async_spawn_syscall_pid`, `async_stress_forks`, `async_stress_channel`, `async_stress_nested`, `async_stress_cow`, `async_stress_slot_size`, `async_sleep`, `async_join`

---

## Implementation State

| Piece | Status |
|-------|--------|
| `Async` effect token (`with Async`) | ✓ Parsed, checked |
| `async fn f() with Async` | ✓ Full codegen |
| `async { expr }` block | ✓ Eager eval — TaskHandle(tid=0, result=expr) |
| `expr.await` | ✓ Codegen: wait4(pid) + read mmap slot |
| `spawn { expr }` | ✓ Codegen: SYS_fork(57) + mmap 4096B result slot |
| `channel::<T>()` | ✓ Builtin: SYS_pipe2(293) with O_CLOEXEC |
| `tx.send(v).await` | ✓ write(fd, &v, 8) |
| `rx.recv().await` | ✓ read(fd, &buf, 8) — blocking |
| `sleep(ms).await` | ✓ SYS_nanosleep(35): timespec on stack |
| `join(h1, h2)` | ✓ Sequential wait4 on both handles, returns (r1, r2) tuple |

---

## Design: OS Thread Model (Phase 1)

**Execution model** (implemented via fork, not pthread):
- `spawn { expr }` → `SYS_fork(57)` + anonymous `mmap(4096, MAP_SHARED|MAP_ANONYMOUS)` result slot. Child evaluates expr, stores result in slot, exits. Parent receives a `TaskHandle` (pid + slot_ptr as two i64 fields on the stack).
- `handle.await` → `SYS_wait4(61)` on pid, reads result from mmap slot.
- `async { expr }` → evaluates eagerly in current thread, returns handle with pid=0 (identity await).
- Channels: `SYS_pipe2(293)` with `O_CLOEXEC` (flags=0x80000). `send` = `write(fd, &v, 8)`, `recv` = `read(fd, &buf, 8)`.

**Why fork, not pthread**: No stack allocation needed. Fork inherits the full process image (including mmap slot). wait4 gives a clear lifecycle. The mmap result slot survives fork because it's MAP_SHARED.

**fork + COW**: Child mutations to parent stack variables are private (copy-on-write). This is expected and correct — `async_stress_cow` documents and verifies this.

---

## Phase 3: sleep + join

### sleep(ms).await

`sleep` is a soft keyword. `sleep(ms)` emits `SYS_nanosleep(35)` with a `{tv_sec, tv_nsec}` timespec on the stack:

```
tv_sec  = ms / 1000          (idiv by 1000)
tv_nsec = (ms % 1000) * 1_000_000
lea rdi, [rbp - ts_addr]     ; &timespec
xor rsi, rsi                 ; rem = NULL
mov eax, 35                  ; SYS_nanosleep
syscall
```

Returns `unit` (EXPR_TY=0). `.await` is required by the effect system but is a no-op semantically (sleep executes eagerly).

### join(h1, h2)

`join` is a soft keyword. `join(h1, h2)` desugars to sequential `wait4` on each handle (tasks run concurrently — fork already started them), then builds a flat 2-tuple `(r1, r2)`.

Returns `(i64, i64)` (EXPR_TY=6, EXPR_TY_HASH=1002). Supports destructuring: `let (r1, r2) = join(h1, h2)`.

**Current limit**: 2-handle join only. 3+ handles require the variadic tuple infrastructure (EXPR_TY_HASH encodes arity).

---

## TaskHandle Layout

```
Stack slot N:     pid (i64)       — offset 0 of the struct
Stack slot N-1:   slot_ptr (i64)  — offset 8 (pointer to mmap result)
```

`spawn` returns `lea rax, [rbp - N*8]` (pointer to the pid field = struct base).

`await` reads:
```
mov rdi, [rax]       ; pid
... SYS_wait4 ...
mov rax, [rax+8]     ; slot_ptr
mov rax, [rax]       ; result
```

---

## What Phase 1 Does NOT Include

- Stackless state machine transformation (Rust-style polling)
- Cooperative scheduling / yield points
- I/O multiplexing (`epoll`, `io_uring`)
- `select` / cancellation
- 3+ handle `join`
- `sleep` with sub-millisecond precision (timespec allows nanoseconds but the API takes milliseconds)

These are Phase 2 and require HIR-level async transform. Phase 1 is sufficient for scientific parallel simulation (PBPK, connectomics).

---

## Verification

```bash
for f in tests/run-pass/async_*.sio; do ./bin/souc run $f; done
```

Expected: all 11 tests print PASS lines, no segfaults, no zombies.

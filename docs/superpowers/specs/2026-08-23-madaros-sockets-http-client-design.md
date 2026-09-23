<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-23-madaros-sockets-http-client-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-23-madaros-sockets-http-client-design
-->

# Madaros TCP Sockets + Plain-Text HTTP/1.1 Client — Design Spec

## Context and Motivation

This re-does sub-project 0a (real TCP sockets + a plain-text HTTP/1.1 client) on top of `main`/Madaros v0.80.0, the actively-developed self-hosted Sounio compiler, superseding an earlier implementation built against a now-superseded compiler lineage ("lean_single") on a stale side branch. The decision to redo this work here — rather than keep building on the old branch — was made explicitly because Madaros is the real trunk of Sounio development; new stdlib work belongs there, not on an increasingly-diverged side branch.

`docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` records the empirical investigation this spec is built on. Key findings that shape this design:

- `extern "C"` gives a real compile-time error (`E219`) for any name outside a fixed allow-list: `print, print_int, print_char, print_f64, get_arg, get_arg_count, str_len, str_char_at, str_eq, str_slice, starts_with, str_concat, str_from_bytes, read_file, write_file, file_size, sqrt, exp, log, sin, cos, assert, heap_alloc, heap_free, f64_to_bits, bits_to_f64, syscall6`. Socket operations are not in this list — `syscall6(nr, a1..a6)` is the only path, exactly as on the prior compiler.
- Local/stack `[T;N]` arrays are boxed GC handles — `&arr as *mut u8` does NOT yield a pointer to the flat byte buffer a syscall needs. **`heap_alloc(n) -> *mut u8` does** give a real flat pointer (confirmed by writing known bytes and reading them back via an independent `syscall6` write to stdout). Multi-byte offsets must go through an `i64` cast (`(addr + i) as *mut u8`) — there is no `*mut u8 + i64` operator and no indexing (`p[i]`) on raw pointers.
- Tuple-destructured structs CAN now be passed as call arguments (fixed vs. the prior compiler) — no struct-literal-rebuild workaround needed.
- Linear-type enforcement now correctly crosses module `use` boundaries (fixed) — a `linear struct TcpSocket` consumed-checked failure in an importing module is now a real, enforced compile error.
- `u16` bitwise operators (`&`, `|`, `>>`, `<<`, `%`) now compute correctly (confirmed: `18080 >> 8 == 70`, `18080 & 255 == 160`, matching `0x46A0`). `u16` addition does NOT wrap on overflow (`65535 + 1 == 65536`, not `0`) — irrelevant for straightforward port-byte extraction (no overflow occurs decomposing a valid 0-65535 port), but must not be relied on anywhere arithmetic could exceed 65535.
- `read_file`/`write_file` are real disk I/O (confirmed byte-exact round-trip) — but passing a `string` as the buffer argument (instead of an array) compiles and returns a plausible byte count while silently corrupting the file's actual on-disk content. Only the array-buffer form is safe.
- Module imports use the form `use <filename>::{name|*}` (bare relative-to-file, no `module` declaration) — confirmed to work consistently regardless of the importing file's directory relative to the imported file's.

This spec covers the same functional scope as the original 0a design: real TCP sockets (client + server) and a plain-text HTTP/1.1 GET client, self-tested with zero real-network dependency. TLS/HTTPS, IPv6, chunked transfer-encoding, and connection pooling remain out of scope (deferred to sub-project 0b and beyond, unchanged from the original design's Non-Goals).

## Non-Goals

Unchanged from the original 0a design: no TLS/HTTPS, no IPv6, no chunked transfer-encoding, no connection pooling/keep-alive, no async I/O (blocking syscalls only), no crawler-specific logic.

## Architecture

Same three-module shape as before, adapted for Madaros:

```
stdlib/net/
  socket.sio       -- TCP client + server primitives (linear TcpSocket)
  dns.sio          -- hostname -> IPv4 resolution
  http_client.sio  -- HTTP/1.1 request/response
```

The critical structural change from the original design: **all syscall-facing byte buffers are heap-allocated (`heap_alloc`), not stack-array-wrapped structs.** The original design's `struct Buf4096 { data: [u8; 4096] }` pattern assumed a local array could be pointer-borrowed for a syscall; on Madaros, local arrays are boxed GC handles and cannot be used this way. A `heap_alloc`-backed buffer type replaces it:

```sio
// A syscall-facing byte buffer: a flat heap allocation plus its capacity.
// Never construct this from a local [u8;N] array -- heap_alloc is the only
// confirmed source of a raw, flat pointer on this compiler.
struct RawBuf {
    ptr: *mut u8,
    cap: i64,
}

fn rawbuf_new(cap: i64) -> RawBuf with IO {
    RawBuf { ptr: heap_alloc(cap), cap: cap }
}

fn rawbuf_set(buf: &RawBuf, i: i64, v: i64) with IO {
    let addr = buf.ptr as i64
    let p = (addr + i) as *mut u8
    *p = v
}

fn rawbuf_get(buf: &RawBuf, i: i64) -> i64 with IO {
    let addr = buf.ptr as i64
    let p = (addr + i) as *mut u8
    (*p) as i64
}
```

(Exact signatures/effects to be finalized during implementation — `rawbuf_get`'s actual dereference-read syntax and whether reading through a raw pointer needs `Mut` in addition to `IO` must be verified empirically, per this spec's own established discipline of testing before assuming.)

`RawBuf` values are never freed in this spec's scope (no `heap_free` call sites) — each socket/HTTP operation allocates a small, bounded number of short-lived buffers per call; leaking them for the lifetime of a single request/response cycle is an accepted simplification (matching the original 0a design's general "don't over-engineer resource management for a first sub-project" spirit). Revisit if the eventual crawler's request volume makes this a real concern.

## Components

### 1. `stdlib/net/socket.sio`

**FFI surface**: raw syscalls via `syscall6`, exactly as the original design and its as-built implementation established (syscall numbers unchanged — this is a kernel ABI, not a Sounio-compiler concern): `socket=41`, `bind=49`, `listen=50`, `accept=43`, `connect=42`, `sendto=44`, `recvfrom=45`, `close=3`.

**Types:**

```sio
linear struct TcpSocket { fd: i64 }
```

(`fd: i64` rather than the original `i32` — confirm during implementation whether `syscall6`'s return type is `i64` and whether narrowing to `i32` introduces any of the same narrow-int quirks found for `u16`; default to keeping the natural `i64` width Madaros's `syscall6` returns rather than narrowing without a reason to.)

Address construction (`sockaddr_in`, 16 bytes) uses a `RawBuf` of capacity 16, written via `rawbuf_set` at each byte offset — the same manual byte-layout approach as the original design (`AF_INET` at offset 0-1, port big-endian at offset 2-3 via the now-simple `(port >> 8) as u8`/`(port & 255) as u8` extraction, IPv4 address at offset 4-7, zero padding at 8-15).

**Public functions** (signatures unchanged in spirit from the original design — `tcp_connect`, `tcp_listen`, `tcp_accept`, `tcp_send`, `tcp_recv`, `tcp_close` — but now taking/returning `RawBuf` instead of `Buf4096`, and benefiting from confirmed-fixed tuple-destructure-as-argument, meaning callers no longer need to rebuild structs via fresh literals before passing them onward):

```sio
pub fn tcp_connect(ip: &RawBuf, port: u16) -> (TcpSocket, i64) with IO
pub fn tcp_listen(ip: &RawBuf, port: u16, backlog: i64) -> (TcpSocket, i64) with IO
pub fn tcp_accept(listener: &TcpSocket) -> (TcpSocket, i64) with IO
pub fn tcp_send(sock: &TcpSocket, buf: &RawBuf, len: i64) -> i64 with IO
pub fn tcp_recv(sock: &TcpSocket, buf: &RawBuf, cap: i64) -> i64 with IO
pub fn tcp_close(sock: TcpSocket) with IO
```

Note `tcp_send`/`tcp_recv` now take `&TcpSocket` (shared reference) rather than consuming and re-returning the linear value each call, since Madaros's confirmed-fixed linear enforcement plus working tuple-as-argument support removes the original design's main reason for the consume-and-return-tuple dance (that pattern existed to route around compiler bugs, not because it was the most natural API shape) — **verify during implementation that a shared `&TcpSocket` reference is sufficient for repeated `send`/`recv` calls without violating linearity** (the socket is still consumed exactly once, by the eventual `tcp_close`, which takes it by value).

**Error convention**: unchanged — sentinel `i64` returns, `pub const ERR_*` values, no `Result<T,E>`/`Option<T>`.

### 2. `stdlib/net/dns.sio`

Unchanged in spirit and scope from the original 0a: resolves hostnames by reading and parsing `/etc/hosts` via `read_file` (now confirmed to be REAL disk I/O on Madaros, which may make this simpler and more direct than the original implementation's raw `open`/`read`/`close` syscall approach — **use `read_file` directly if it can read `/etc/hosts` correctly with the array-buffer form; only fall back to raw syscalls via `RawBuf` if `read_file` proves unable to handle this specific file for some reason**). This remains `/etc/hosts`-only, NOT general DNS resolution — `getaddrinfo` remains inaccessible (not on the extern allow-list, no syscall equivalent), exactly as documented for the prior compiler. This scope limitation is inherited unchanged, not re-litigated.

```sio
pub fn resolve_ipv4(hostname: &RawBuf, hostname_len: i64, out_ip: &RawBuf) -> i64 with IO
```

### 3. `stdlib/net/http_client.sio`

Unchanged in scope and behavior from the original design (a single GET, `Content-Length`-delimited response parsing, `Connection: close`, no TLS) — only the buffer type changes (`RawBuf` instead of `Buf4096`/`Buf256`), and request/response byte manipulation goes through `rawbuf_set`/`rawbuf_get` instead of `struct.data[i]` array indexing.

```sio
struct HttpResponse {
    status: i64,
    headers: RawBuf,
    body: RawBuf,
    body_len: i64,
    ok: bool
}

pub fn http_get(host: &RawBuf, host_len: i64, port: u16, path: &RawBuf, path_len: i64) -> HttpResponse with IO
```

`ok`'s meaning is unchanged from the original design's final, clarified semantics: true iff a well-formed status line + `Content-Length` body was received and parsed — NOT an assertion that the HTTP status code itself indicates success (a 404/500 response is `ok: true`; check `status` separately).

## Testing Strategy

Unchanged in spirit from the original design: self-contained `tests/run-pass/*.sio` files, a single process both hosting a loopback server and driving the client against it via a real OS `fork()` (the original implementation's confirmed-working pattern for genuine client/server concurrency without Sounio-level threads, including the hardened `reap_child` with a `pid > 0` guard before any `kill`/`wait4` call — carry that exact safety discipline forward into this reimplementation, it is not optional polish).

- `tests/run-pass/net_socket_loopback.sio` — round-trip byte equality over a real loopback connection.
- `tests/compile-fail/net_socket_linear_not_consumed.sio` — proves `TcpSocket`'s linearity is enforced. **This is expected to be a genuine passing compile-fail test now** (not a `//@ known-failure`), since cross-module linear enforcement is confirmed fixed on Madaros — verify this expectation empirically as the first step of that task, and only fall back to `//@ known-failure` if the enforcement somehow doesn't hold for this specific case.
- `tests/run-pass/net_dns_resolve.sio` — resolves `"localhost"` via `/etc/hosts`.
- `tests/run-pass/net_http_client_localhost.sio` — full HTTP GET round-trip against a hand-rolled loopback server, asserting `status`, `body` content, and exercising both the numeric-IP and hostname code paths (the final hardened version of the original test, including its body-content and DNS-path coverage additions from that implementation's own final review).

## Open Implementation-Time Questions

- `rawbuf_get`'s exact dereference-read syntax and effect requirements (`IO` alone, or also `Mut`) — verify empirically before finalizing `socket.sio`.
- Whether `tcp_send`/`tcp_recv` taking `&TcpSocket` (rather than consuming/re-returning) actually satisfies the linear-type checker across the number of calls a real HTTP exchange needs — verify with a real multi-call test, not just a single send/recv.
- Whether `read_file` can be pointed at `/etc/hosts` directly and returns a usable byte buffer for line-scanning, or whether `dns.sio` still needs raw `open`/`read`/`close` via `syscall6` for this specific case (e.g. if `read_file`'s return type/size limits don't fit reading an arbitrary-length system file).

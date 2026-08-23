# Sounio TCP Sockets + Plain-Text HTTP/1.1 Client — Design Spec

## Context and Motivation

This is sub-project **0a** of a larger, multi-phase goal: building a custom web crawler and search index as a systems/production showcase for Sounio, following the project's own philosophy that new compute workloads should be verbs in Sounio rather than delegated to Rust/Python.

A survey of the current stdlib (2026-08-22) found the compiler core (parser, type checker, effects, native codegen) to be production-grade, but the I/O boundary is essentially unimplemented: `stdlib/net/tcp.sio` and `stdlib/http/http.sio` are documented as pure data-structure stubs with "no actual I/O," `stdlib/io/io.sio` is an in-memory buffer rather than real disk I/O, and `fopen`/`fread`/`fwrite` FFI declarations exist only as disabled comments. What *does* work today, proven by real passing tests, is `extern "C"` FFI to libc (`stdlib/os/process.sio`: `getpid`/`getppid`/`exit`; `stdlib/mem/box.sio`: `malloc`/`calloc`/`realloc`/`free`), and linear types for resource ownership (`tests/run-pass/linear_correct_consume.sio`, enforced — see `tests/compile-fail/linear_not_consumed.sio`).

This sub-project's goal is narrow and concrete: **real TCP sockets (client and server) and a plain-text (non-TLS) HTTP/1.1 client, built via the same proven FFI pattern**, self-contained and testable with zero external dependencies (no real internet access required for `tests/run-pass`). It deliberately excludes TLS/HTTPS — that is sub-project 0b, to be designed separately once this lands, following the transport-abstraction seam this spec establishes.

The larger roadmap this fits into (for context only, not this spec's scope):
- **0a (this spec):** TCP sockets + plain HTTP/1.1 client.
- **0b:** TLS 1.2/1.3 in Sounio, slotting in underneath the HTTP client via the stream abstraction this spec defines.
- **Fase 1+:** the actual crawler, index, ranking, and search API (separate specs, later).

## Non-Goals

- TLS/HTTPS support of any kind (sub-project 0b).
- IPv6 (IPv4 only for this spec; the address-construction code is IPv4-specific).
- HTTP/1.1 chunked transfer-encoding (only `Content-Length`-delimited bodies are parsed; a response with `Transfer-Encoding: chunked` and no `Content-Length` returns a parse error, not a hang).
- Connection reuse/keep-alive pooling (every request opens a fresh socket and closes it; `Connection: close` is always sent).
- Any crawler-specific logic (URL frontier, robots.txt, politeness/rate-limiting) — this spec is pure transport plumbing.
- Async/concurrent I/O — all operations are blocking syscalls, matching the synchronous style already used by the rest of the FFI stdlib (`stdlib/os`, `stdlib/mem`). Sounio's `async`/`thread` stdlib modules are simulated scaffolding today (per the same survey), not real OS concurrency, so building on them would be building on a stub; blocking I/O is the honest baseline.

## Architecture

Three new stdlib modules under `stdlib/net/`, each following the existing FFI convention: an `extern "C"` block declaring the raw libc functions, wrapped by `pub fn ... with IO` functions that do type casts and sentinel-based error mapping — the same shape as `stdlib/os/process.sio`.

```
stdlib/net/
  socket.sio       -- TCP client + server primitives (linear TcpSocket)
  dns.sio          -- hostname -> IPv4 resolution (isolated, highest FFI risk)
  http_client.sio  -- HTTP/1.1 request/response over a Stream abstraction
```

`http_client.sio` depends on a **`Stream` abstraction**, not directly on `TcpSocket` — this is the one structural decision made now specifically to make sub-project 0b tractable: when TLS exists, a `TlsConnection` implements the same `Stream` shape and the HTTP client code does not change. Concretely (Sounio has no traits/interfaces as of this survey — confirm during implementation; if unavailable, `Stream` is a plain struct holding function pointers, or `http_client` functions are written generically enough to be re-parameterized when 0b lands. This is an implementation-time decision, not a blocker for this spec — see Task-level notes in the eventual plan).

## Components

### 1. `stdlib/net/socket.sio`

**FFI surface** (`extern "C"` block, matching the style of `stdlib/mem/box.sio`):

```sio
extern "C" {
    fn socket(domain: i32, sock_type: i32, protocol: i32) -> i32;
    fn connect(sockfd: i32, addr: *mut u8, addrlen: u32) -> i32;
    fn bind(sockfd: i32, addr: *mut u8, addrlen: u32) -> i32;
    fn listen(sockfd: i32, backlog: i32) -> i32;
    fn accept(sockfd: i32, addr: *mut u8, addrlen: *mut u8) -> i32;
    fn send(sockfd: i32, buf: *mut u8, len: usize, flags: i32) -> i64;
    fn recv(sockfd: i32, buf: *mut u8, len: usize, flags: i32) -> i64;
    fn close(fd: i32) -> i32;
    fn inet_pton(af: i32, src: *mut u8, dst: *mut u8) -> i32;
    fn htons(hostshort: u16) -> u16;
}
```

**Types:**

```sio
linear struct TcpSocket { fd: i32 }

struct SockAddrBuf { data: [u8; 16] }   // raw sockaddr_in bytes, wrapped per the
                                        // documented array-mutation workaround
                                        // (stdlib/str/mod.sio's Buf256 pattern)

struct Buf4096 { data: [u8; 4096] }    // generic byte buffer for send/recv
```

`sockaddr_in` is constructed manually as 16 raw bytes rather than via named C-struct interop (Sounio has no such interop today): bytes 0-1 = `AF_INET` (2) little-endian per libc convention on x86-64, bytes 2-3 = port big-endian (via `htons`), bytes 4-7 = IPv4 address bytes (via `inet_pton`), bytes 8-15 = zero padding. This construction is entirely internal to `socket.sio`; callers never see raw bytes.

**Public functions** (all `with IO`, consume-and-return the linear socket per `examples/showcase/linear_file_server.sio`'s pattern):

```sio
pub fn tcp_connect(ip: &Buf4096, ip_len: i32, port: u16) -> (TcpSocket, i32) with IO
    // returns (socket, 0) on success, (invalid socket with fd = -1, error_code) on failure

pub fn tcp_listen(ip: &Buf4096, ip_len: i32, port: u16, backlog: i32) -> (TcpSocket, i32) with IO
    // binds + listens; returns the listening socket

pub fn tcp_accept(listener: &TcpSocket) -> (TcpSocket, i32) with IO
    // blocks until a client connects; returns the new connection socket
    // (listener is NOT consumed -- it's a shared &, since a listener accepts many times)

pub fn tcp_send(sock: TcpSocket, buf: &Buf4096, len: i32) -> (TcpSocket, i64) with IO
    // returns (socket, bytes_sent) or (socket, -1) on error; socket is returned for reuse

pub fn tcp_recv(sock: TcpSocket, out: &!Buf4096) -> (TcpSocket, i64) with IO, Mut
    // returns (socket, bytes_read); 0 means peer closed; -1 means error

pub fn tcp_close(sock: TcpSocket) with IO
    // terminal -- consumes the linear socket, no return
```

**Error convention:** sentinel integers, matching the enabled (non-`.disabled`) convention actually used today in `stdlib/os/env.sio` and `str_buf_find` (`-1` for "not found"/error), not the `Result<T,E>`/`Option<T>` convention that exists only in disabled code. Named error constants (`ERR_SOCKET_CREATE = -1`, `ERR_CONNECT = -2`, `ERR_BIND = -3`, `ERR_LISTEN = -4`, `ERR_ACCEPT = -5`) are defined as `pub const` in this module for callers to compare against.

### 2. `stdlib/net/dns.sio`

**Scope and risk framing:** this module resolves a hostname string to an IPv4 address. It is isolated from `socket.sio` specifically because `getaddrinfo`'s C API returns a linked list of `addrinfo` structs, which is the most complex piece of C interop in this entire spec (pointer-chasing through an opaque struct whose exact field layout must be replicated correctly for the target platform). If this proves intractable during implementation, it can be marked `//@ ignore` in its own test file without blocking `socket.sio` or `http_client.sio`, both of which can be fully tested against loopback/IP-literal addresses without DNS.

**FFI surface:**

```sio
extern "C" {
    fn getaddrinfo(node: *mut u8, service: *mut u8, hints: *mut u8, res: *mut u8) -> i32;
    fn freeaddrinfo(res: *mut u8);
}
```

**Public function:**

```sio
pub fn resolve_ipv4(hostname: &Buf256, hostname_len: i32, out_ip: &!Buf4096) -> i32 with IO, Mut
    // writes the first resolved IPv4 address (dotted-decimal string) into out_ip,
    // returns 0 on success, -1 on resolution failure, -2 on unexpected addrinfo shape
```

Implementation note carried into the eventual plan: the `addrinfo` struct's first IPv4 result is extracted by walking known byte offsets (matching the platform's actual `struct addrinfo` / `struct sockaddr_in` layout, verified against the build platform's system headers, not assumed) — this is exactly the kind of fragile-but-tractable FFI work flagged as the module's core risk.

### 3. `stdlib/net/http_client.sio`

**Types:**

```sio
struct HttpResponse {
    status: i32,        // e.g. 200, 404 -- parsed from the status line
    headers: Str,        // raw header block, unparsed beyond status-line extraction
    body: Str,           // body content, sliced by Content-Length
    ok: bool             // true iff a well-formed status line + Content-Length body was read
}
```

**Public function:**

```sio
pub fn http_get(host: &Buf256, host_len: i32, port: u16, path: &Buf256, path_len: i32) -> HttpResponse with IO, Mut, Panic, Div
```

**Behavior:**
1. If `host` is not already a dotted-decimal IPv4 literal (checked via a simple digit/dot scan, not a full parse), call `dns::resolve_ipv4`. If DNS fails, return `HttpResponse { ok: false, status: -1, ... }`.
2. `tcp_connect` to the resolved/literal IP on `port`.
3. Build the request bytes into a `Buf4096`: `"GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n"`, using `str_buf_*` helpers from `stdlib/str/mod.sio` for concatenation into the fixed buffer (request must fit in 4096 bytes for this spec's scope; a path/host combination exceeding that returns an error rather than silently truncating).
4. `tcp_send` the request bytes.
5. Loop `tcp_recv` into an accumulating buffer until the peer closes (`recv` returns 0) or a `Content-Length`-implied byte count is reached, whichever comes first.
6. Parse: find `\r\n\r\n` via `str_buf_find` to split headers from body; parse the status line's numeric code; find the `Content-Length:` header value (case-sensitive match is acceptable for this spec — case-insensitive header matching is a documented follow-up, not a blocker); slice the body to that length.
7. `tcp_close` the socket unconditionally (success or failure path).

**Error handling:** `HttpResponse.ok = false` covers every failure mode (DNS failure, connect failure, malformed response, buffer-size overflow) rather than a separate error type — this keeps the public surface to one struct, consistent with the sentinel-convention philosophy (a boolean sentinel field instead of a numeric one, appropriate here because there's exactly one call site's worth of complexity to report, not a reusable low-level primitive like `socket.sio`'s functions).

## Data Flow (end-to-end)

```
http_get("example.com", 80, "/")
  -> is "example.com" a dotted-decimal IP? no
  -> dns::resolve_ipv4("example.com") -> "93.184.216.34" (or ok:false, return early)
  -> socket::tcp_connect("93.184.216.34", 80) -> TcpSocket (or ok:false, return early)
  -> build request bytes: "GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n"
  -> socket::tcp_send(sock, request_bytes)
  -> loop socket::tcp_recv(sock, buf) until closed or Content-Length satisfied
  -> parse status line, headers, body
  -> socket::tcp_close(sock)
  -> return HttpResponse { status: 200, headers: "...", body: "...", ok: true }
```

## Testing Strategy

All tests are self-contained `tests/run-pass/*.sio` files requiring no real network access and no external process — a single Sounio test process both hosts the server side and drives the client side over `127.0.0.1`, per the project's own reliability rule that only `tests/run-pass` evidence is trusted (directory/file existence is not).

- **`tests/run-pass/net_socket_loopback.sio`**: `tcp_listen` on `127.0.0.1:0`-style ephemeral-or-fixed test port → in the same test, `tcp_connect` to it → `tcp_accept` on the listener → `tcp_send` a known byte pattern from the client → `tcp_recv` on the accepted connection and assert byte-for-byte equality → close both ends. Follows the `//@ run-pass` + `report()`/`main()` structure of `tests/stdlib/os/test_os.sio`.
- **`tests/run-pass/net_dns_resolve.sio`**: resolves `"localhost"` (should always resolve to `127.0.0.1` without needing real internet) and asserts the returned string is a valid dotted-decimal IPv4 address. Marked `//@ ignore` with a comment explaining why, if `getaddrinfo` FFI proves unworkable during implementation — this must not block the other two test files.
- **`tests/run-pass/net_http_client_localhost.sio`**: uses `socket.sio`'s own `tcp_listen`/`tcp_accept` to implement a *minimal* hand-rolled HTTP server in the same test file (reads a request, ignores its content, writes back a fixed `"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"` response) — then calls `http_client::http_get("127.0.0.1", <test_port>, "/")` against it and asserts `status == 200`, `body == "hello"`, `ok == true`. This exercises the full client-side parse logic without depending on `dns.sio` (IP literal) or the real internet.
- **`tests/compile-fail/`**: at least one test proving `TcpSocket`'s linearity is enforced — e.g. creating a socket via `tcp_connect` and never passing it to `tcp_send`/`tcp_recv`/`tcp_close` should fail to compile with `error-pattern: linear value not consumed`, mirroring `tests/compile-fail/linear_not_consumed.sio`.

## Open Implementation-Time Questions (not blocking this spec, to resolve during planning/implementation)

- Exact `struct addrinfo` byte layout must be confirmed against the actual build platform's system headers before writing `dns.sio`'s offset-walking code — this is implementation verification, not a design decision.
- Whether Sounio's type system offers any interface/trait mechanism to express the `Stream` abstraction formally, or whether `http_client.sio` should instead be written twice (once now for `TcpSocket`, refactored when 0b lands) — a call to make when writing the implementation plan, informed by whatever the language actually supports today.
- Whether `AF_INET`'s byte-order placement in the raw `sockaddr_in` construction needs `htons`-style conversion (address family is conventionally host-byte-order as a `u16` on Linux, unlike port and address) — verify against a known-good C reference during implementation, don't guess.

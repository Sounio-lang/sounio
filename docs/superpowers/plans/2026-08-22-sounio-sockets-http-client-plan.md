# Sounio TCP Sockets + Plain HTTP/1.1 Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real, working TCP sockets (client + server) and a plain-text HTTP/1.1 client in Sounio, built via `extern "C"` FFI to libc, fully self-tested with zero real-network dependency.

**Architecture:** Three new stdlib modules (`stdlib/net/socket.sio`, `stdlib/net/dns.sio`, `stdlib/net/http_client.sio`) follow the proven `extern "C"` + `pub fn ... with IO` wrapper pattern already used by `stdlib/os/process.sio` and `stdlib/mem/box.sio`. `TcpSocket` is a `linear struct` wrapping a file descriptor, enforced-consumed exactly like `tests/run-pass/linear_correct_consume.sio`. Every test is a single self-contained Sounio program that both hosts a loopback TCP/HTTP server and drives the client against it — no real internet access is used or required anywhere in this plan.

**Tech Stack:** Sounio (self-hosted compiler, `./bin/souc`), `extern "C"` FFI to libc syscalls (`socket`/`bind`/`listen`/`accept`/`connect`/`send`/`recv`/`close`/`inet_pton`/`htons`/`getaddrinfo`).

**Spec:** `docs/superpowers/specs/2026-08-22-sounio-sockets-http-client-design.md`

## Global Constraints

- No TLS/HTTPS. No IPv6 (IPv4 only). No chunked transfer-encoding (only `Content-Length`-delimited bodies). No connection pooling/keep-alive (every request is `Connection: close`, fresh socket per call).
- No async/concurrent I/O — every syscall wrapper is a blocking call, matching the synchronous style of the rest of the FFI stdlib (Sounio's `async`/`thread` stdlib modules are simulated scaffolding, not real OS concurrency, so nothing here depends on them).
- Error convention is sentinel integers with named `pub const` error codes (matching the enabled convention in `stdlib/os/env.sio` and `str_buf_find`) — never `Result<T,E>`/`Option<T>` (that convention exists only in `.disabled` stdlib files, not the currently-working one).
- Byte buffers are `[u8; N]` wrapped in a one-field struct (e.g. `struct Buf4096 { data: [u8; N] }`), matching the documented workaround in `stdlib/str/mod.sio` for the known JIT bug with bare `&![T; N]` mutation — never a bare `&![u8; N]` parameter.
- Test invocation: `tests/run-pass/*.sio` are run via `$SOUC_BIN run <file>` (from repo root, `SOUC_BIN=./bin/souc`), which must exit 0; `tests/compile-fail/*.sio` are run via `$SOUC_BIN compile <file> -o <tmp-path>`, which must exit nonzero OR print `typecheck: failed`, and must contain the file's `//@ error-pattern: <text>` string somewhere in combined stdout+stderr. Every test file starts with `//@ run-pass` or `//@ compile-fail` plus a `//@ description: <text>` line (see `tests/run-pass/linear_correct_consume.sio` / `tests/compile-fail/linear_not_consumed.sio` for the exact header shape to copy). You can also run the whole suite (or a substring-filtered subset) via `bash scripts/run_sio_test_suite.sh <optional-substring-filter>` from the repo root.
- Commit message convention: `[net] <description>` (component prefix, per this repo's CLAUDE.md Commits section — `net` is not in the existing component list there, add it as a new one since this plan introduces the `stdlib/net/` module for the first time). **NEVER add "Co-Authored-By" or any AI-attribution line to any commit in this repo** — this repo's CLAUDE.md explicitly overrides the default convention: "No AI attribution — No 'Co-Authored-By' or similar in commits."
- No Rust, Python, or any non-Sounio code anywhere in this plan's deliverables.

---

## Task 1: `stdlib/net/socket.sio` — TCP client + server core, with a self-contained loopback round-trip test

**Files:**
- Create: `stdlib/net/socket.sio`
- Test: `tests/run-pass/net_socket_loopback.sio`

**Interfaces:**
- Produces: `linear struct TcpSocket { fd: i32 }`, `struct Buf4096 { data: [u8; 4096] }`, `struct SockAddrBuf { data: [u8; 16] }`, `pub const ERR_SOCKET_CREATE: i32 = -1`, `pub const ERR_CONNECT: i32 = -2`, `pub const ERR_BIND: i32 = -3`, `pub const ERR_LISTEN: i32 = -4`, `pub const ERR_ACCEPT: i32 = -5`, and the public functions: `pub fn tcp_connect(ip: &Buf4096, ip_len: i32, port: u16) -> (TcpSocket, i32) with IO`, `pub fn tcp_listen(ip: &Buf4096, ip_len: i32, port: u16, backlog: i32) -> (TcpSocket, i32) with IO`, `pub fn tcp_accept(listener: &TcpSocket) -> (TcpSocket, i32) with IO`, `pub fn tcp_send(sock: TcpSocket, buf: &Buf4096, len: i32) -> (TcpSocket, i64) with IO`, `pub fn tcp_recv(sock: TcpSocket, out: &!Buf4096) -> (TcpSocket, i64) with IO, Mut`, `pub fn tcp_close(sock: TcpSocket) with IO`. Tasks 2, 3, and 4 all import and use these exact names/signatures from `stdlib/net/socket.sio`.

This task combines `connect`/`send`/`recv`/`close` with `bind`/`listen`/`accept` in one task (not split across two tasks as an initial draft of this plan considered) because a self-contained round-trip test — required by this project's own "only `tests/run-pass` evidence is trusted" discipline — needs both a client and a server side to exist before any of it can be tested at all; there is no way to test `tcp_connect` in isolation without something listening.

- [ ] **Step 1: Spike — confirm the syntax for taking a raw pointer to a local struct's array field**

Before writing any real code, write a tiny throwaway file `/tmp/sounio_ptr_spike.sio` to determine the actual Sounio syntax for obtaining a `*mut u8` pointing at a local (stack) `struct { data: [u8; N] }`'s `data` field, since every existing `extern "C"` example in this codebase (`stdlib/os/process.sio`, `stdlib/mem/box.sio`, `stdlib/database/ffi/bindings.sio`) only passes pointers that originated from `malloc`/`calloc` (heap pointers returned by a prior FFI call), never the address of a local stack variable's field. Try, in order, until one works:

```sio
extern "C" {
    fn write_probe(buf: *mut u8, len: usize) -> i64;
}
struct Probe { data: [u8; 8] }
fn main() -> i32 with IO {
    var p = Probe { data: [0, 0, 0, 0, 0, 0, 0, 0] }
    let ptr = &!p.data as *mut u8   // attempt 1: cast a mutable array reference to a raw pointer
    return 0
}
```

If `&!p.data as *mut u8` does not compile, check `docs/guide/LLM_PROGRAMMING_GUIDE.md` and grep `self-hosted/` for any existing `as *mut` or `as *const` cast pattern, and try whatever real syntax you find. Document the working syntax in a one-line comment at the top of `socket.sio` once found (e.g. `// pointer-to-local-buffer pattern: &!x.data as *mut u8`), since Tasks 3 and 4 also need this pattern and should not have to rediscover it.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
//@ description: TCP loopback round-trip — server accepts a connection, client sends bytes, server receives identical bytes

use net::socket::*

fn main() -> i32 with IO, Mut {
    var server_ip = Buf4096 { data: [0; 4096] }
    server_ip.data[0] = 49   // "1"
    server_ip.data[1] = 50   // "2"
    server_ip.data[2] = 55   // "7"
    server_ip.data[3] = 46   // "."
    server_ip.data[4] = 48   // "0"
    server_ip.data[5] = 46   // "."
    server_ip.data[6] = 48   // "0"
    server_ip.data[7] = 46   // "."
    server_ip.data[8] = 49   // "1"
    let ip_len = 9

    let (listener, listen_err) = tcp_listen(&server_ip, ip_len, 18080 as u16, 1)
    if listen_err != 0 {
        println("FAIL: tcp_listen returned error")
        return 1
    }

    let (client_sock, connect_err) = tcp_connect(&server_ip, ip_len, 18080 as u16)
    if connect_err != 0 {
        println("FAIL: tcp_connect returned error")
        return 1
    }

    let (server_conn, accept_err) = tcp_accept(&listener)
    if accept_err != 0 {
        println("FAIL: tcp_accept returned error")
        return 1
    }

    var send_buf = Buf4096 { data: [0; 4096] }
    send_buf.data[0] = 72   // 'H'
    send_buf.data[1] = 73   // 'I'
    let (client_sock2, sent) = tcp_send(client_sock, &send_buf, 2)
    if sent != 2 {
        println("FAIL: tcp_send did not send 2 bytes")
        return 1
    }

    var recv_buf = Buf4096 { data: [0; 4096] }
    let (server_conn2, received) = tcp_recv(server_conn, &!recv_buf)
    if received != 2 {
        println("FAIL: tcp_recv did not receive 2 bytes")
        return 1
    }
    if recv_buf.data[0] != 72 {
        println("FAIL: byte 0 mismatch")
        return 1
    }
    if recv_buf.data[1] != 73 {
        println("FAIL: byte 1 mismatch")
        return 1
    }

    tcp_close(client_sock2)
    tcp_close(server_conn2)
    tcp_close(listener)

    println("PASS: loopback round-trip matched")
    return 0
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc run tests/run-pass/net_socket_loopback.sio`
Expected: FAIL — `stdlib/net/socket.sio` doesn't exist yet, so `use net::socket::*` fails to resolve.

- [ ] **Step 4: Implement `stdlib/net/socket.sio`**

```sio
// pointer-to-local-buffer pattern: <fill in whatever Step 1's spike found to work>

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

pub const ERR_SOCKET_CREATE: i32 = 0 - 1
pub const ERR_CONNECT: i32 = 0 - 2
pub const ERR_BIND: i32 = 0 - 3
pub const ERR_LISTEN: i32 = 0 - 4
pub const ERR_ACCEPT: i32 = 0 - 5

const AF_INET: i32 = 2
const SOCK_STREAM: i32 = 1

pub linear struct TcpSocket { fd: i32 }
pub struct Buf4096 { data: [u8; 4096] }
struct SockAddrBuf { data: [u8; 16] }

// Builds a 16-byte sockaddr_in: bytes 0-1 = AF_INET (host byte order),
// bytes 2-3 = port (network byte order via htons), bytes 4-7 = IPv4
// address (network byte order, produced directly by inet_pton),
// bytes 8-15 = zero padding.
fn build_sockaddr(ip: &Buf4096, ip_len: i32, port: u16) -> (SockAddrBuf, i32) with IO, Mut {
    var addr = SockAddrBuf { data: [0; 16] }
    addr.data[0] = 2   // AF_INET low byte
    addr.data[1] = 0   // AF_INET high byte

    let port_be = htons(port)
    addr.data[2] = ((port_be >> 8) & 255) as u8
    addr.data[3] = (port_be & 255) as u8

    // inet_pton needs a null-terminated C string; ip buffer must already
    // be null-terminated at ip_len (callers pass a dotted-decimal literal
    // with a trailing 0 byte reserved in the 4096-byte buffer).
    let pton_result = inet_pton(AF_INET, &!ip.data as *mut u8, &!addr.data as *mut u8 /* offset +4, see note below */)
    if pton_result != 1 {
        return (addr, ERR_CONNECT)
    }
    return (addr, 0)
}

pub fn tcp_connect(ip: &Buf4096, ip_len: i32, port: u16) -> (TcpSocket, i32) with IO, Mut {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0 {
        return (TcpSocket { fd: -1 }, ERR_SOCKET_CREATE)
    }
    let (addr, build_err) = build_sockaddr(ip, ip_len, port)
    if build_err != 0 {
        close(fd)
        return (TcpSocket { fd: -1 }, build_err)
    }
    var addr_mut = addr
    let result = connect(fd, &!addr_mut.data as *mut u8, 16 as u32)
    if result != 0 {
        close(fd)
        return (TcpSocket { fd: -1 }, ERR_CONNECT)
    }
    return (TcpSocket { fd: fd }, 0)
}

pub fn tcp_listen(ip: &Buf4096, ip_len: i32, port: u16, backlog: i32) -> (TcpSocket, i32) with IO, Mut {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0 {
        return (TcpSocket { fd: -1 }, ERR_SOCKET_CREATE)
    }
    let (addr, build_err) = build_sockaddr(ip, ip_len, port)
    if build_err != 0 {
        close(fd)
        return (TcpSocket { fd: -1 }, build_err)
    }
    var addr_mut = addr
    let bind_result = bind(fd, &!addr_mut.data as *mut u8, 16 as u32)
    if bind_result != 0 {
        close(fd)
        return (TcpSocket { fd: -1 }, ERR_BIND)
    }
    let listen_result = listen(fd, backlog)
    if listen_result != 0 {
        close(fd)
        return (TcpSocket { fd: -1 }, ERR_LISTEN)
    }
    return (TcpSocket { fd: fd }, 0)
}

pub fn tcp_accept(listener: &TcpSocket) -> (TcpSocket, i32) with IO, Mut {
    var addr = SockAddrBuf { data: [0; 16] }
    var addrlen_buf: [u8; 8] = [0; 8]
    let conn_fd = accept(listener.fd, &!addr.data as *mut u8, &!addrlen_buf as *mut u8)
    if conn_fd < 0 {
        return (TcpSocket { fd: -1 }, ERR_ACCEPT)
    }
    return (TcpSocket { fd: conn_fd }, 0)
}

pub fn tcp_send(sock: TcpSocket, buf: &Buf4096, len: i32) -> (TcpSocket, i64) with IO, Mut {
    var buf_mut = *buf
    let result = send(sock.fd, &!buf_mut.data as *mut u8, len as usize, 0)
    return (sock, result)
}

pub fn tcp_recv(sock: TcpSocket, out: &!Buf4096) -> (TcpSocket, i64) with IO, Mut {
    let result = recv(sock.fd, &!out.data as *mut u8, 4096 as usize, 0)
    return (sock, result)
}

pub fn tcp_close(sock: TcpSocket) with IO {
    close(sock.fd)
}
```

Note in `build_sockaddr`: the third argument to `inet_pton` must point at byte offset 4 within `addr.data` (where the address field starts), not byte offset 0. If Sounio has no pointer-arithmetic/offset syntax (`ptr + 4`), write the address into a temporary 4-byte buffer first via `inet_pton`, then copy those 4 bytes into `addr.data[4..8]` manually with a loop — confirm which approach the language actually supports during this step and use whichever compiles; this is exactly the kind of platform/syntax detail the spec's "Open Implementation-Time Questions" flagged as unknown until implementation.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc run tests/run-pass/net_socket_loopback.sio`
Expected: PASS, prints "PASS: loopback round-trip matched", exit code 0.

- [ ] **Step 6: Commit**

```bash
cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32
git add stdlib/net/socket.sio tests/run-pass/net_socket_loopback.sio
git commit -m "[net] Add TCP socket client/server FFI with self-contained loopback test"
```

---

## Task 2: Linear-type enforcement test for `TcpSocket`

**Files:**
- Test: `tests/compile-fail/net_socket_linear_not_consumed.sio`

**Interfaces:**
- Consumes: `TcpSocket`, `Buf4096`, `tcp_connect` from Task 1's `stdlib/net/socket.sio` (no new interfaces produced — this task is pure test coverage proving an existing language guarantee holds for this specific new type).

- [ ] **Step 1: Write the failing (compile-fail) test**

```sio
//@ compile-fail
//@ description: TcpSocket must be consumed — a socket obtained from tcp_connect that is never sent, received, or closed must fail to compile
//@ error-pattern: linear value not consumed

use net::socket::*

fn main() -> i32 with IO, Mut {
    var ip = Buf4096 { data: [0; 4096] }
    ip.data[0] = 49
    ip.data[1] = 50
    ip.data[2] = 55
    ip.data[3] = 46
    ip.data[4] = 48
    ip.data[5] = 46
    ip.data[6] = 48
    ip.data[7] = 46
    ip.data[8] = 49
    let (sock, err) = tcp_connect(&ip, 9, 18081 as u16)
    // sock is never passed to tcp_send/tcp_recv/tcp_close -- must be a compile error
    return 0
}
```

- [ ] **Step 2: Run the test to verify it currently does NOT correctly fail (sanity check first)**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc compile tests/compile-fail/net_socket_linear_not_consumed.sio -o /tmp/net_linear_test_out 2>&1`

Expected at this point (Task 1 already merged `linear struct TcpSocket`): the compiler SHOULD already reject this, since `TcpSocket` was declared `linear` in Task 1. This step is a verification, not a TDD red step in the usual sense — confirm the output is nonzero exit or contains `typecheck: failed`, and that `linear value not consumed` (or the compiler's actual exact wording — check against `tests/compile-fail/linear_not_consumed.sio`'s real passing behavior if the exact string differs) appears in the output. If it does NOT reject (i.e., this compiles successfully), that means Task 1's `linear` annotation on `TcpSocket` was somehow not effective — stop and fix `stdlib/net/socket.sio`'s `TcpSocket` declaration before proceeding; do not weaken this test to match incorrect behavior.

- [ ] **Step 3: Run via the test suite runner to confirm it's picked up correctly**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && bash scripts/run_sio_test_suite.sh net_socket_linear_not_consumed`
Expected: PASS (the suite runner treats a compile-fail file as passing when compilation fails with the expected error pattern present).

- [ ] **Step 4: Commit**

```bash
cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32
git add tests/compile-fail/net_socket_linear_not_consumed.sio
git commit -m "[net] Add compile-fail test proving TcpSocket linearity is enforced"
```

---

## Task 3: `stdlib/net/dns.sio` — hostname resolution (isolated, deferrable if `getaddrinfo` FFI proves intractable)

**Files:**
- Create: `stdlib/net/dns.sio`
- Test: `tests/run-pass/net_dns_resolve.sio`

**Interfaces:**
- Consumes: `Buf4096`, `Buf256` (define `Buf256` in this file if it doesn't already exist elsewhere in the stdlib — check `stdlib/str/mod.sio` first, since the spec references a `Buf256` pattern already established there; reuse that exact struct shape/name rather than declaring a conflicting duplicate).
- Produces: `pub fn resolve_ipv4(hostname: &Buf256, hostname_len: i32, out_ip: &!Buf4096) -> i32 with IO, Mut`. Task 4's `http_client.sio` calls this exact signature when given a non-numeric hostname.

This task is explicitly allowed to end in a `//@ ignore`-marked test (see Step 5) rather than a passing one, per the spec's own risk framing: `getaddrinfo`'s C API returns a linked list of `addrinfo` structs, the most complex FFI surface in this plan, and failure here must not block Task 4 (which can be fully tested against an IP literal instead of a hostname).

- [ ] **Step 1: Check for an existing `Buf256`**

Run: `grep -n "struct Buf256" /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32/stdlib/str/mod.sio`

If found, `use str::mod::Buf256` (or the correct existing module path) in `dns.sio` instead of redeclaring it. If not found under that exact name, declare `pub struct Buf256 { data: [u8; 256] }` locally in `dns.sio`.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
//@ description: DNS resolution of "localhost" returns a valid IPv4 dotted-decimal string

use net::dns::*

fn main() -> i32 with IO, Mut {
    var hostname = Buf256 { data: [0; 256] }
    hostname.data[0] = 108  // 'l'
    hostname.data[1] = 111  // 'o'
    hostname.data[2] = 99   // 'c'
    hostname.data[3] = 97   // 'a'
    hostname.data[4] = 108  // 'l'
    hostname.data[5] = 104  // 'h'
    hostname.data[6] = 111  // 'o'
    hostname.data[7] = 115  // 's'
    hostname.data[8] = 116  // 't'

    var out_ip = Buf4096 { data: [0; 4096] }
    let result = resolve_ipv4(&hostname, 9, &!out_ip)
    if result != 0 {
        println("FAIL: resolve_ipv4 returned error")
        return 1
    }
    // "localhost" must resolve to 127.0.0.1 -- check the first 3 bytes are "127"
    if out_ip.data[0] != 49 {   // '1'
        println("FAIL: expected loopback address to start with '1'")
        return 1
    }
    println("PASS: localhost resolved")
    return 0
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc run tests/run-pass/net_dns_resolve.sio`
Expected: FAIL — `stdlib/net/dns.sio` doesn't exist yet.

- [ ] **Step 4: Implement `stdlib/net/dns.sio`**

```sio
extern "C" {
    fn getaddrinfo(node: *mut u8, service: *mut u8, hints: *mut u8, res: *mut *mut u8) -> i32;
    fn freeaddrinfo(res: *mut u8);
}

pub const ERR_DNS_RESOLVE: i32 = 0 - 1
pub const ERR_DNS_UNEXPECTED_SHAPE: i32 = 0 - 2

// struct addrinfo (glibc, x86-64):
//   int     ai_flags;      offset 0,  4 bytes
//   int     ai_family;     offset 4,  4 bytes
//   int     ai_socktype;   offset 8,  4 bytes
//   int     ai_protocol;   offset 12, 4 bytes
//   size_t  ai_addrlen;    offset 16, 8 bytes
//   struct sockaddr *ai_addr;   offset 24, 8 bytes (pointer)
//   char    *ai_canonname;      offset 32, 8 bytes (pointer)
//   struct addrinfo *ai_next;   offset 40, 8 bytes (pointer)
// VERIFY this layout against the actual build machine before trusting it --
// run `getconf LONG_BIT` to confirm 64-bit, and cross-check against
// /usr/include/netdb.h's `struct addrinfo` on the build host. Do not ship
// this module without that verification; it is exactly the risk the spec
// flagged.
//
// ai_addr points at a sockaddr_in (for AF_INET results): the IPv4 address
// itself is at byte offset 4 within *that* struct (same layout as
// socket.sio's SockAddrBuf).

pub fn resolve_ipv4(hostname: &Buf256, hostname_len: i32, out_ip: &!Buf4096) -> i32 with IO, Mut {
    var hostname_mut = *hostname
    var result_ptr: [u8; 8] = [0; 8]   // holds the returned *mut addrinfo (as raw bytes)

    let rc = getaddrinfo(&!hostname_mut.data as *mut u8, 0 as *mut u8, 0 as *mut u8, &!result_ptr as *mut *mut u8)
    if rc != 0 {
        return ERR_DNS_RESOLVE
    }

    // Dereference the returned addrinfo* to read ai_family (offset 4) and,
    // for AF_INET, ai_addr (offset 24, itself a pointer to a sockaddr_in
    // whose address bytes start at its own offset 4). The exact mechanism
    // for "read N bytes at pointer P + offset O" depends on what pointer
    // arithmetic / deref syntax Sounio actually supports -- confirm during
    // this step (grep self-hosted/ and stdlib/database/ffi/ for any
    // existing pointer-offset-read pattern; sqlite3 bindings in
    // stdlib/database/ffi/bindings.sio are the closest known precedent
    // for real pointer-heavy FFI in this codebase and are worth reading
    // in full before writing this function body).
    //
    // If no working pattern can be found within a reasonable time, this
    // function may return ERR_DNS_UNEXPECTED_SHAPE unconditionally and the
    // test below gets marked //@ ignore (see Step 5) -- this is an
    // explicitly sanctioned outcome for this task, not a failure of the
    // plan.

    return ERR_DNS_UNEXPECTED_SHAPE
}
```

- [ ] **Step 5: Attempt to complete the implementation; if blocked, mark the test `//@ ignore` instead of leaving it failing**

Spend effort on the pointer-dereferencing mechanism noted in Step 4's comments. If you get a working implementation, replace the `return ERR_DNS_UNEXPECTED_SHAPE` stub with real address-extraction logic and proceed to Step 6 expecting a PASS.

If, after real effort, the pointer-chasing genuinely cannot be expressed in current Sounio syntax, do NOT leave the test failing silently. Instead, change the test file's header to:

```sio
//@ ignore
//@ description: DNS resolution of "localhost" returns a valid IPv4 dotted-decimal string -- BLOCKED: getaddrinfo result struct pointer-chasing not yet expressible in current Sounio syntax (see stdlib/net/dns.sio's resolve_ipv4 for what was tried)
```

and leave `resolve_ipv4` returning `ERR_DNS_UNEXPECTED_SHAPE` with the explanatory comment intact. Report this outcome clearly in your task report either way — this is a legitimate, spec-sanctioned outcome, not a task failure, but it must be visible, not silently swept under a passing-looking test.

- [ ] **Step 6: Run the test**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc run tests/run-pass/net_dns_resolve.sio` (or, if marked `//@ ignore`, run `bash scripts/run_sio_test_suite.sh net_dns_resolve` and confirm the suite runner skips it rather than failing the whole suite — check `scripts/dev/run_sio_test_suite.sh` for how it handles `//@ ignore` if this is the first time in this task hitting that path).
Expected: PASS "localhost resolved" (preferred outcome), or a clean skip via `//@ ignore` (sanctioned fallback).

- [ ] **Step 7: Commit**

```bash
cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32
git add stdlib/net/dns.sio tests/run-pass/net_dns_resolve.sio
git commit -m "[net] Add DNS resolution via getaddrinfo FFI"
```

(Commit this either way — whether the test passes or is marked `//@ ignore` with the blocking reason documented, the module and its honest current state belong in git.)

---

## Task 4: `stdlib/net/http_client.sio` — HTTP/1.1 GET, tested against a hand-rolled loopback server

**Files:**
- Create: `stdlib/net/http_client.sio`
- Test: `tests/run-pass/net_http_client_localhost.sio`

**Interfaces:**
- Consumes: everything from Task 1's `stdlib/net/socket.sio` (`TcpSocket`, `Buf4096`, `tcp_connect`, `tcp_listen`, `tcp_accept`, `tcp_send`, `tcp_recv`, `tcp_close`), and Task 3's `stdlib/net/dns.sio` (`resolve_ipv4`) when the host argument is not a numeric IP literal.
- Produces: `pub struct HttpResponse { status: i32, headers: Str, body: Str, ok: bool }`, `pub fn http_get(host: &Buf256, host_len: i32, port: u16, path: &Buf256, path_len: i32) -> HttpResponse with IO, Mut, Panic, Div`. No later task in this plan consumes this (it's the final deliverable), but the eventual Fase 1 crawler work will.

Before writing this task's code, check whether Sounio's real trait mechanism (`self-hosted/check/traits.sio` — confirmed to exist and support method dispatch, not just a compiler-internal detail with no user-facing syntax) has a documented example of a user-level `trait`/`impl` declaration anywhere in `docs/guide/LLM_PROGRAMMING_GUIDE.md`, `docs/guide/`, or any `.sio` file under `stdlib/` or `examples/`. If a clear example exists, define a small `Stream` trait (`fn stream_write(...)`, `fn stream_read(...)`) and implement it for `TcpSocket`, per the design spec's stated goal of letting sub-project 0b's future TLS connection type slot in underneath `http_client.sio` without changing this file. If no clear, working example of trait declaration syntax can be found within reasonable effort, write `http_get` directly against `TcpSocket` (no abstraction), with a one-line comment `// TODO(0b): re-parameterize over a Stream trait once TLS lands` — this fallback is explicitly sanctioned by the design spec rather than inventing unverified trait syntax.

- [ ] **Step 1: Write the failing test**

```sio
//@ run-pass
//@ description: http_get against a hand-rolled loopback HTTP server returns the expected status and body

use net::socket::*
use net::http_client::*

fn main() -> i32 with IO, Mut, Panic, Div {
    var server_ip = Buf4096 { data: [0; 4096] }
    server_ip.data[0] = 49
    server_ip.data[1] = 50
    server_ip.data[2] = 55
    server_ip.data[3] = 46
    server_ip.data[4] = 48
    server_ip.data[5] = 46
    server_ip.data[6] = 48
    server_ip.data[7] = 46
    server_ip.data[8] = 49

    let (listener, listen_err) = tcp_listen(&server_ip, 9, 18082 as u16, 1)
    if listen_err != 0 {
        println("FAIL: tcp_listen returned error")
        return 1
    }

    // Hand-rolled minimal HTTP server: accept one connection, ignore the
    // request content, write back a fixed response.
    let (conn, accept_err) = tcp_accept(&listener)
    if accept_err != 0 {
        println("FAIL: tcp_accept returned error")
        return 1
    }

    var drain_buf = Buf4096 { data: [0; 4096] }
    let (conn2, _drained) = tcp_recv(conn, &!drain_buf)

    var response_buf = Buf4096 { data: [0; 4096] }
    let response_text = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
    // copy response_text bytes into response_buf.data -- use whatever
    // Sounio string-literal-to-byte-array mechanism stdlib/str/mod.sio's
    // own helpers demonstrate (e.g. str_from_bytes_buf's inverse, or a
    // manual byte-by-byte copy loop); this is well-precedented, unlike
    // Task 3's pointer-chasing risk.
    let response_len = 44
    let (conn3, sent) = tcp_send(conn2, &response_buf, response_len)
    tcp_close(conn3)
    tcp_close(listener)

    var host = Buf256 { data: [0; 256] }
    host.data[0] = 49
    host.data[1] = 50
    host.data[2] = 55
    host.data[3] = 46
    host.data[4] = 48
    host.data[5] = 46
    host.data[6] = 48
    host.data[7] = 46
    host.data[8] = 49

    var path = Buf256 { data: [0; 256] }
    path.data[0] = 47   // "/"

    let response = http_get(&host, 9, 18082 as u16, &path, 1)
    if response.ok != true {
        println("FAIL: http_get did not succeed")
        return 1
    }
    if response.status != 200 {
        println("FAIL: expected status 200")
        return 1
    }
    println("PASS: http_get round-trip matched")
    return 0
}
```

Note: this test has both the client and the fake server driven sequentially in one single-threaded process (accept-then-recv-then-send happens before the `http_get` call below it in program order) rather than truly concurrently, since Sounio has no real threads (per this plan's Global Constraints). This works because a listening backlog of 1 plus a same-process pre-written response means the connection's data is already queued in the kernel socket buffer by the time `http_get`'s own `tcp_connect`+`tcp_send`+`tcp_recv` runs — verify this ordering actually works when you run it; if the OS-level connect/accept/send sequencing doesn't hold together this way without real concurrency, restructure the test so the "fake server" logic runs after establishing the listener but the actual accept+respond happens lazily (e.g., write the fixed response bytes immediately upon accept, before the client-side `http_get` starts sending — reorder the steps above so `tcp_accept` + the canned `tcp_send` happen, THEN `http_get` is called, so the response is already staged in the kernel's send buffer for the client to read whenever it connects and requests). Fix any ordering issue you hit empirically; the design intent (self-contained, no real concurrency, one process) does not change.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc run tests/run-pass/net_http_client_localhost.sio`
Expected: FAIL — `stdlib/net/http_client.sio` doesn't exist yet.

- [ ] **Step 3: Implement `stdlib/net/http_client.sio`**

```sio
use net::socket::*
use net::dns::*

pub struct HttpResponse {
    status: i32,
    headers: Str,
    body: Str,
    ok: bool
}

fn is_numeric_ip(host: &Buf256, host_len: i32) -> bool with IO, Mut, Panic, Div {
    var i = 0
    while i < host_len {
        let c = host.data[i]
        // digits 48-57, '.' is 46
        if (c < 48 || c > 57) && c != 46 {
            return false
        }
        i = i + 1
    }
    return true
}

pub fn http_get(host: &Buf256, host_len: i32, port: u16, path: &Buf256, path_len: i32) -> HttpResponse with IO, Mut, Panic, Div {
    var resolved_ip = Buf4096 { data: [0; 4096] }
    var resolved_len = host_len

    if is_numeric_ip(host, host_len) {
        var i = 0
        while i < host_len {
            resolved_ip.data[i] = host.data[i]
            i = i + 1
        }
    } else {
        let dns_result = resolve_ipv4(host, host_len, &!resolved_ip)
        if dns_result != 0 {
            return HttpResponse { status: -1, headers: "", body: "", ok: false }
        }
        // resolved_len for a resolved dotted-decimal address: find the
        // first zero byte in resolved_ip.data to determine its length,
        // using str_buf_find or an equivalent scan from stdlib/str/mod.sio.
    }

    let (sock, connect_err) = tcp_connect(&resolved_ip, resolved_len, port)
    if connect_err != 0 {
        return HttpResponse { status: -1, headers: "", body: "", ok: false }
    }

    // Build "GET <path> HTTP/1.1\r\nHost: <host>\r\nConnection: close\r\n\r\n"
    // into a Buf4096 using str_buf_* concatenation helpers from
    // stdlib/str/mod.sio (str_buf_eq/str_buf_find exist there for
    // searching; check for a concatenation/copy helper, or copy bytes
    // manually in a loop -- either is acceptable, this has no known
    // syntax risk unlike Task 3's pointer work).
    var request_buf = Buf4096 { data: [0; 4096] }
    let request_len = 0   // fill in via the real construction logic

    let (sock2, sent) = tcp_send(sock, &request_buf, request_len)
    if sent < 0 {
        tcp_close(sock2)
        return HttpResponse { status: -1, headers: "", body: "", ok: false }
    }

    var response_buf = Buf4096 { data: [0; 4096] }
    let (sock3, received) = tcp_recv(sock2, &!response_buf)
    tcp_close(sock3)

    if received <= 0 {
        return HttpResponse { status: -1, headers: "", body: "", ok: false }
    }

    // Parse: find "\r\n\r\n" (bytes 13,10,13,10) via str_buf_find to split
    // headers from body; parse the numeric status code from the status
    // line (bytes after the first space, up to the second space); find
    // "Content-Length:" in the header block and parse its integer value;
    // slice the body to that many bytes from response_buf.
    // Use str_from_bytes_buf (stdlib/str/lib.sio) to convert byte ranges
    // into Str values for the HttpResponse fields.

    return HttpResponse { status: 200, headers: "", body: "hello", ok: true }
}
```

The parsing logic's exact implementation (finding `\r\n\r\n`, extracting the status code, reading `Content-Length`) must be filled in completely using `str_buf_find`/`str_from_bytes_buf` from `stdlib/str/mod.sio` and `stdlib/str/lib.sio` (both real, confirmed-working modules per the design spec's research) — the skeleton above shows the control flow and buffer types; do not leave any TODO in the committed version of this file. Write it to genuinely parse the response bytes, not to hardcode the test's expected values (the placeholder `return HttpResponse { status: 200, ..., body: "hello", ok: true }` above is scaffolding for this plan document only, not code to commit as-is).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && ./bin/souc run tests/run-pass/net_http_client_localhost.sio`
Expected: PASS, "PASS: http_get round-trip matched", exit code 0.

- [ ] **Step 5: Run the full new test group together**

Run: `cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32 && bash scripts/run_sio_test_suite.sh net_`
Expected: all `net_*` tests pass (or, for `net_dns_resolve`, cleanly skip if Task 3 ended in the sanctioned `//@ ignore` outcome) — confirm no regressions in the rest of the suite by also running the full suite once: `bash scripts/run_sio_test_suite.sh`.

- [ ] **Step 6: Commit**

```bash
cd /home/devsounio/sounio/.claude/worktrees/sounio-jobs-cap-32
git add stdlib/net/http_client.sio tests/run-pass/net_http_client_localhost.sio
git commit -m "[net] Add HTTP/1.1 GET client over stdlib/net/socket.sio"
```

---

## Self-Review Notes

**Spec coverage:** all three modules (`socket.sio`, `dns.sio`, `http_client.sio`) have a task each producing exactly the function signatures and types the spec defines. The spec's testing strategy (three `tests/run-pass` files + one `tests/compile-fail` file) maps 1:1 onto this plan's four tasks. The spec's two "Open Implementation-Time Questions" (`sockaddr_in`/`addrinfo` byte layout verification, and the `Stream` abstraction mechanism) are carried into Task 1's Step 1 spike and Task 3/Task 4's explicit verify-or-fallback instructions, respectively — neither is silently assumed.

**Placeholder scan and fix applied during writing:** the original task breakdown (proposed before this plan was written) split socket work into "core client ops" and "listen/accept" as two separate tasks with a round-trip test in the first — that was internally inconsistent (a round-trip test needs both sides) and has been corrected into a single Task 1 covering all of `socket.sio`. The `http_client.sio` skeleton in Task 4 contains commented guidance on using `str_buf_find`/`str_from_bytes_buf` rather than a bare "add parsing logic" placeholder — the step's own text explicitly says the hardcoded `return HttpResponse {...}` in the skeleton is not committable as-is and must be replaced with real parsing.

**Type consistency:** `TcpSocket`, `Buf4096`, `Buf256`, and every `tcp_*`/`resolve_ipv4`/`http_get` signature are declared once (in the task that owns the file) and referenced identically by name and signature in every later task's Interfaces block and code. `HttpResponse`'s four fields (`status`, `headers`, `body`, `ok`) are used consistently in both the test file and the implementation skeleton in Task 4.

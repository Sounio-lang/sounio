<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-23-madaros-sockets-http-client-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-23-madaros-sockets-http-client-plan
-->

# Madaros TCP Sockets + Plain HTTP/1.1 Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real, working TCP sockets (client + server) and a plain-text HTTP/1.1 client on Madaros v0.80.0 (the self-hosted Sounio compiler on `main`), replacing an earlier implementation built against a now-superseded compiler lineage.

**Architecture:** Three stdlib modules (`stdlib/net/socket.sio`, `stdlib/net/dns.sio`, `stdlib/net/http_client.sio`) built on `syscall6` for raw syscalls (socket/connect/bind/listen/accept/send/recv/close — none of these are on Madaros's `extern "C"` allow-list) and a `RawBuf` heap-allocated buffer type for every syscall-facing byte buffer, since local `[u8;N]` arrays are boxed GC handles on this compiler and cannot be pointer-borrowed for raw I/O. `TcpSocket` is a `linear struct`, and Madaros's cross-module linear enforcement (confirmed fixed, unlike the prior compiler) means its compile-fail test should pass for real, not as a documented known-failure.

**Tech Stack:** Sounio/Madaros (`./bin/souc`, wrapping `bin/madaros-linux-x86_64`), `syscall6` raw-syscall builtin, `heap_alloc`/`read_file`/`write_file` from the confirmed `extern "C"` allow-list.

**Spec:** `docs/superpowers/specs/2026-08-23-madaros-sockets-http-client-design.md`

## Global Constraints

- No TLS/HTTPS. No IPv6 (IPv4 only). No chunked transfer-encoding (only `Content-Length`-delimited bodies). No connection pooling/keep-alive (every request is `Connection: close`, fresh socket per call).
- No async/concurrent I/O — every syscall wrapper is a blocking call.
- Error convention is sentinel `i64` returns with named `pub const` error codes — never `Result<T,E>`/`Option<T>`.
- **Every syscall-facing byte buffer MUST be a heap-allocated `RawBuf` (backed by `heap_alloc`), never a local `[u8;N]` array pointer-borrowed for a syscall.** This is a confirmed, real bug class on Madaros (local arrays are boxed GC handles; casting one to `*mut u8` exposes handle/metadata bytes, not the flat buffer contents) — not a style preference. See `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` for the confirmed evidence.
- Multi-byte pointer offsets go through an `i64` cast — there is no `*mut u8 + i64` operator and no indexing (`p[i]`) on raw pointers. The confirmed-working pattern: `let addr = p as i64; let p2 = (addr + i) as *mut u8`.
- Module imports use the bare form `use <filename>::{name|*}` — no `module` declaration.
- Test invocation: `tests/run-pass/*.sio` via `$SOUC_BIN run <file>` (`SOUC_BIN=./bin/souc`), expects exit 0, plus any `//@ expect-stdout: X` substring checks. `tests/compile-fail/*.sio` via `$SOUC_BIN compile <file> -o <tmp-path>`, expects a nonzero exit or a `typecheck failed`-shaped diagnostic, with the file's `//@ error-pattern: X` string present in the output. Every test file starts with `//@ run-pass` or `//@ compile-fail`; a `//@ description: <text>` line is good practice but not required by the runner. Run the whole suite (or a filtered subset) via `bash scripts/run_sio_test_suite.sh --filter <substring>` or `--filter-prefix`/`--filter-exact` from the repo root (this is the v2 runner — richer than the prior branch's version, same annotation set).
- Commit message convention on this branch/repo: Conventional-Commits-style `type(scope): description` (e.g. `feat(net): add TCP socket core`, `fix(net): ...`, `docs(spec): ...`) — matching real recent commits on this branch (`feat(stdlib): ...`, `fix(compiler): ...`, `audit(...): ...`). **Never add "Co-Authored-By" or any AI-attribution line to any commit** — this repo's `CLAUDE.md` explicitly states: "No AI attribution in commit messages."
- No Rust, Python, or any non-Sounio code anywhere in this plan's deliverables.
- `RawBuf` values are never freed in this plan's scope (no `heap_free` call sites) — each socket/HTTP operation allocates a small, bounded number of short-lived buffers per call; accepted as a deliberate simplification per the spec.

---

## Task 1: `stdlib/net/socket.sio` — `RawBuf` + TCP client/server core, with a self-contained loopback round-trip test

**Files:**
- Create: `stdlib/net/socket.sio`
- Test: `tests/run-pass/net_socket_loopback.sio`

**Interfaces:**
- Produces: `struct RawBuf { ptr: *mut u8, cap: i64 }`, `fn rawbuf_new(cap: i64) -> RawBuf with IO`, `fn rawbuf_set(buf: &RawBuf, i: i64, v: i64) with IO`, `fn rawbuf_get(buf: &RawBuf, i: i64) -> i64 with IO` (or `with IO, Mut` — verify in Step 1), `linear struct TcpSocket { fd: i64 }`, `pub const ERR_SOCKET_CREATE: i64 = -1`, `pub const ERR_CONNECT: i64 = -2`, `pub const ERR_BIND: i64 = -3`, `pub const ERR_LISTEN: i64 = -4`, `pub const ERR_ACCEPT: i64 = -5`, and the public functions: `pub fn tcp_connect(ip: &RawBuf, port: u16) -> (TcpSocket, i64) with IO`, `pub fn tcp_listen(ip: &RawBuf, port: u16, backlog: i64) -> (TcpSocket, i64) with IO`, `pub fn tcp_accept(listener: &TcpSocket) -> (TcpSocket, i64) with IO`, `pub fn tcp_send(sock: &TcpSocket, buf: &RawBuf, len: i64) -> i64 with IO`, `pub fn tcp_recv(sock: &TcpSocket, buf: &RawBuf, cap: i64) -> i64 with IO`, `pub fn tcp_close(sock: TcpSocket) with IO`. Tasks 2, 3, and 4 all import and use these exact names/signatures from `stdlib/net/socket.sio`.

This combines `connect`/`send`/`recv`/`close` with `bind`/`listen`/`accept` in one task (not split across two) because a self-contained round-trip test needs both a client and a server side present from the start — the prior implementation on the old compiler learned this the hard way when its original task breakdown tried to split them.

- [ ] **Step 1: Verify `rawbuf_get`'s exact syntax before writing anything else**

Write a throwaway file `/tmp/rawbuf_spike.sio`:

```sio
extern "C" {
    fn heap_alloc(n: i64) -> *mut u8;
}
fn main() -> i64 with IO {
    let p = heap_alloc(4)
    *p = 88
    let addr = p as i64
    let p2 = (addr + 1) as *mut u8
    *p2 = 89
    // Now try reading back through the pointer:
    let v0 = (*p) as i64
    print_int(v0)
    return 0
}
```

Run: `./bin/souc run /tmp/rawbuf_spike.sio`
Expected: prints `88`, confirming `(*p) as i64` is a working dereference-read with just `with IO` declared (no `Mut` needed for a *read*, since nothing is being mutated by the read itself — only verify this assumption doesn't produce an E035 effect error; if it does, add `Mut` to the effect list and use whichever combination actually compiles). Record the exact working syntax in a comment at the top of `stdlib/net/socket.sio` once confirmed, since Tasks 3 and 4 need the identical pattern.

- [ ] **Step 2: Implement `RawBuf` and its helpers**

```sio
extern "C" {
    fn heap_alloc(n: i64) -> *mut u8;
}

// <fill in the exact confirmed rawbuf_get dereference-read syntax/effects from Step 1>

pub struct RawBuf {
    ptr: *mut u8,
    cap: i64,
}

pub fn rawbuf_new(cap: i64) -> RawBuf with IO {
    RawBuf { ptr: heap_alloc(cap), cap: cap }
}

pub fn rawbuf_set(buf: &RawBuf, i: i64, v: i64) with IO {
    let addr = buf.ptr as i64
    let p = (addr + i) as *mut u8
    *p = v
}

pub fn rawbuf_get(buf: &RawBuf, i: i64) -> i64 with IO {
    let addr = buf.ptr as i64
    let p = (addr + i) as *mut u8
    (*p) as i64
}
```

- [ ] **Step 3: Write the failing loopback test**

```sio
//@ run-pass

use socket::*

fn main() -> i64 with IO {
    let server_ip = rawbuf_new(16)
    rawbuf_set(&server_ip, 0, 49)   // '1'
    rawbuf_set(&server_ip, 1, 50)   // '2'
    rawbuf_set(&server_ip, 2, 55)   // '7'
    rawbuf_set(&server_ip, 3, 46)   // '.'
    rawbuf_set(&server_ip, 4, 48)   // '0'
    rawbuf_set(&server_ip, 5, 46)   // '.'
    rawbuf_set(&server_ip, 6, 48)   // '0'
    rawbuf_set(&server_ip, 7, 46)   // '.'
    rawbuf_set(&server_ip, 8, 49)   // '1'
    let ip_len: i64 = 9

    let (listener, listen_err) = tcp_listen(&server_ip, 18090, 1)
    if listen_err != 0 {
        print_int(1)
        return 1
    }

    let (client_sock, connect_err) = tcp_connect(&server_ip, 18090)
    if connect_err != 0 {
        print_int(2)
        return 1
    }

    let (server_conn, accept_err) = tcp_accept(&listener)
    if accept_err != 0 {
        print_int(3)
        return 1
    }

    let send_buf = rawbuf_new(4)
    rawbuf_set(&send_buf, 0, 72)   // 'H'
    rawbuf_set(&send_buf, 1, 73)   // 'I'
    let sent = tcp_send(&client_sock, &send_buf, 2)
    if sent != 2 {
        print_int(4)
        return 1
    }

    let recv_buf = rawbuf_new(4)
    let received = tcp_recv(&server_conn, &recv_buf, 4)
    if received != 2 {
        print_int(5)
        return 1
    }
    if rawbuf_get(&recv_buf, 0) != 72 {
        print_int(6)
        return 1
    }
    if rawbuf_get(&recv_buf, 1) != 73 {
        print_int(7)
        return 1
    }

    tcp_close(client_sock)
    tcp_close(server_conn)
    tcp_close(listener)

    print_int(0)
    return 0
}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/net_socket_loopback.sio`
Expected: FAIL — `socket.sio` doesn't exist yet, or `use socket::*` fails to resolve.

- [ ] **Step 5: Implement `stdlib/net/socket.sio`'s socket functions**

```sio
extern "C" {
    fn heap_alloc(n: i64) -> *mut u8;
}

pub const ERR_SOCKET_CREATE: i64 = 0 - 1
pub const ERR_CONNECT: i64 = 0 - 2
pub const ERR_BIND: i64 = 0 - 3
pub const ERR_LISTEN: i64 = 0 - 4
pub const ERR_ACCEPT: i64 = 0 - 5

const AF_INET: i64 = 2
const SOCK_STREAM: i64 = 1

pub linear struct TcpSocket { fd: i64 }

// [RawBuf + rawbuf_new/set/get from Step 2 go here]

// Builds a 16-byte sockaddr_in into a fresh RawBuf: bytes 0-1 = AF_INET,
// bytes 2-3 = port (network byte order), bytes 4-7 = IPv4 address bytes
// (already in the right order from the caller's dotted-decimal parse),
// bytes 8-15 = zero padding.
fn build_sockaddr(ip: &RawBuf, port: u16) -> RawBuf with IO {
    let addr = rawbuf_new(16)
    rawbuf_set(&addr, 0, 2)   // AF_INET low byte
    rawbuf_set(&addr, 1, 0)   // AF_INET high byte
    rawbuf_set(&addr, 2, (port >> 8) as i64)
    rawbuf_set(&addr, 3, (port & 255) as i64)
    rawbuf_set(&addr, 4, rawbuf_get(ip, 0))
    rawbuf_set(&addr, 5, rawbuf_get(ip, 1))
    rawbuf_set(&addr, 6, rawbuf_get(ip, 2))
    rawbuf_set(&addr, 7, rawbuf_get(ip, 3))
    rawbuf_set(&addr, 8, 0)
    rawbuf_set(&addr, 9, 0)
    rawbuf_set(&addr, 10, 0)
    rawbuf_set(&addr, 11, 0)
    rawbuf_set(&addr, 12, 0)
    rawbuf_set(&addr, 13, 0)
    rawbuf_set(&addr, 14, 0)
    rawbuf_set(&addr, 15, 0)
    addr
}

pub fn tcp_connect(ip: &RawBuf, port: u16) -> (TcpSocket, i64) with IO {
    let fd = syscall6(41, AF_INET, SOCK_STREAM, 0, 0, 0, 0)
    if fd < 0 {
        return (TcpSocket { fd: 0 - 1 }, ERR_SOCKET_CREATE)
    }
    let addr = build_sockaddr(ip, port)
    let result = syscall6(42, fd, addr.ptr as i64, 16, 0, 0, 0)
    if result != 0 {
        syscall6(3, fd, 0, 0, 0, 0, 0)
        return (TcpSocket { fd: 0 - 1 }, ERR_CONNECT)
    }
    (TcpSocket { fd: fd }, 0)
}

pub fn tcp_listen(ip: &RawBuf, port: u16, backlog: i64) -> (TcpSocket, i64) with IO {
    let fd = syscall6(41, AF_INET, SOCK_STREAM, 0, 0, 0, 0)
    if fd < 0 {
        return (TcpSocket { fd: 0 - 1 }, ERR_SOCKET_CREATE)
    }
    let addr = build_sockaddr(ip, port)
    let bind_result = syscall6(49, fd, addr.ptr as i64, 16, 0, 0, 0)
    if bind_result != 0 {
        syscall6(3, fd, 0, 0, 0, 0, 0)
        return (TcpSocket { fd: 0 - 1 }, ERR_BIND)
    }
    let listen_result = syscall6(50, fd, backlog, 0, 0, 0, 0)
    if listen_result != 0 {
        syscall6(3, fd, 0, 0, 0, 0, 0)
        return (TcpSocket { fd: 0 - 1 }, ERR_LISTEN)
    }
    (TcpSocket { fd: fd }, 0)
}

pub fn tcp_accept(listener: &TcpSocket) -> (TcpSocket, i64) with IO {
    let conn_fd = syscall6(43, listener.fd, 0, 0, 0, 0, 0)
    if conn_fd < 0 {
        return (TcpSocket { fd: 0 - 1 }, ERR_ACCEPT)
    }
    (TcpSocket { fd: conn_fd }, 0)
}

pub fn tcp_send(sock: &TcpSocket, buf: &RawBuf, len: i64) -> i64 with IO {
    syscall6(44, sock.fd, buf.ptr as i64, len, 0, 0, 0)
}

pub fn tcp_recv(sock: &TcpSocket, buf: &RawBuf, cap: i64) -> i64 with IO {
    syscall6(45, sock.fd, buf.ptr as i64, cap, 0, 0, 0)
}

pub fn tcp_close(sock: TcpSocket) with IO {
    syscall6(3, sock.fd, 0, 0, 0, 0, 0)
}
```

`accept`'s syscall (number 43) normally takes `(sockfd, addr*, addrlen*)` — passing `0, 0` for the address-output pointers (as shown) tells the kernel we don't want the peer's address; verify this compiles and behaves correctly (returns a valid fd) during this step — if the kernel requires non-null pointers even when unused, allocate a small scratch `RawBuf` for `addr`/`addrlen` and pass those instead, using whichever approach actually works.

- [ ] **Step 6: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/net_socket_loopback.sio`
Expected: PASS, prints `0`, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add stdlib/net/socket.sio tests/run-pass/net_socket_loopback.sio
git commit -m "feat(net): add TCP socket client/server core with RawBuf and a loopback round-trip test"
```

---

## Task 2: Linear-type enforcement test for `TcpSocket`

**Files:**
- Test: `tests/compile-fail/net_socket_linear_not_consumed.sio`

**Interfaces:**
- Consumes: `TcpSocket`, `RawBuf`, `rawbuf_new`/`rawbuf_set`, `tcp_connect` from Task 1's `stdlib/net/socket.sio`.

- [ ] **Step 1: Verify the expectation empirically before writing the "real" test**

Cross-module linear enforcement is confirmed fixed on Madaros (see `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`, Finding 5's positive test in the investigation this plan is built on: a two-file case with `linear struct Handle` exported from one file and consumed-but-ignored in an importing file correctly failed with `error[E040]: linear value not consumed`). Confirm this holds for `TcpSocket` specifically by writing the test below and running it BEFORE assuming the outcome.

- [ ] **Step 2: Write the test**

```sio
//@ compile-fail
//@ error-pattern: linear value not consumed

use socket::*

fn main() -> i64 with IO {
    let ip = rawbuf_new(16)
    rawbuf_set(&ip, 0, 49)
    rawbuf_set(&ip, 1, 50)
    rawbuf_set(&ip, 2, 55)
    rawbuf_set(&ip, 3, 46)
    rawbuf_set(&ip, 4, 48)
    rawbuf_set(&ip, 5, 46)
    rawbuf_set(&ip, 6, 48)
    rawbuf_set(&ip, 7, 46)
    rawbuf_set(&ip, 8, 49)
    let (sock, err) = tcp_connect(&ip, 18091)
    // sock is never passed to tcp_send/tcp_recv/tcp_close -- must be a compile error
    return 0
}
```

- [ ] **Step 3: Run and confirm the expected outcome**

Run: `./bin/souc compile tests/compile-fail/net_socket_linear_not_consumed.sio -o /tmp/net_linear_test_out`
Expected: nonzero exit, output containing `linear value not consumed` (or Madaros's exact E040 wording — confirm the literal string the `//@ error-pattern` needs to match against real output, adjust the annotation's text if the exact phrasing differs from this plan's guess).

If this does NOT fail to compile (i.e., it compiles successfully despite `sock` being unconsumed), STOP — this would mean Task 1's `TcpSocket` linear declaration is not effective, which is a Task 1 defect to fix, not something to route around with a `//@ known-failure` annotation (unlike the prior compiler, this is not an expected/accepted gap here).

- [ ] **Step 4: Run via the test suite runner to confirm it's picked up correctly**

Run: `bash scripts/run_sio_test_suite.sh --filter net_socket_linear_not_consumed`
Expected: PASS (the runner treats a compile-fail file as passing when compilation fails with the expected error pattern present).

- [ ] **Step 5: Commit**

```bash
git add tests/compile-fail/net_socket_linear_not_consumed.sio
git commit -m "test(net): prove TcpSocket linearity is enforced across module boundaries on Madaros"
```

---

## Task 3: `stdlib/net/dns.sio` — hostname resolution via `/etc/hosts`

**Files:**
- Create: `stdlib/net/dns.sio`
- Test: `tests/run-pass/net_dns_resolve.sio`

**Interfaces:**
- Consumes: `RawBuf`, `rawbuf_new`, `rawbuf_set`, `rawbuf_get` from Task 1's `stdlib/net/socket.sio`.
- Produces: `pub fn resolve_ipv4(hostname: &RawBuf, hostname_len: i64, out_ip: &RawBuf) -> i64 with IO`. Task 4's `http_client.sio` calls this exact signature when given a non-numeric hostname.

- [ ] **Step 1: Try `read_file` directly on `/etc/hosts` first**

Write a throwaway spike `/tmp/read_hosts_spike.sio`:

```sio
extern "C" {
    fn read_file(path: [i8; 32]) -> string;
}
fn main() -> i64 with IO {
    let content = read_file("/etc/hosts")
    print_int(str_len(&content))
    return 0
}
```

(Adjust the exact `read_file` signature/path-argument type to match what actually compiles — the spec notes `read_file`'s confirmed-working form takes an array buffer, not a bare string literal, for its OWN write-side safety; check whether the read side has the same constraint by trying a plain string-literal path argument first, since `/etc/hosts` is a fixed, known path and the corruption risk documented in the audit was specifically about the buffer *being written*, not the path argument — but confirm this distinction empirically rather than assuming it transfers safely.)

Run: `./bin/souc run /tmp/read_hosts_spike.sio`

If this compiles and returns a plausible length (`/etc/hosts` is typically 100-300 bytes on a minimal Linux system), use `read_file` for the real implementation. If it fails to compile or produces garbage, fall back to raw syscalls (`open`=2, `read`=0, `close`=3 via `syscall6`, staging into a `RawBuf`) exactly as the prior compiler's implementation did, adapted to use `RawBuf` instead of the old `WordBuf512` word-staging pattern (a `RawBuf` is already a real flat heap buffer, so no staging workaround is needed here — just `syscall6(0, fd, buf.ptr as i64, buf.cap, 0, 0, 0)` for the read).

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass

use dns::*

fn main() -> i64 with IO {
    let hostname = rawbuf_new(16)
    rawbuf_set(&hostname, 0, 108)  // 'l'
    rawbuf_set(&hostname, 1, 111)  // 'o'
    rawbuf_set(&hostname, 2, 99)   // 'c'
    rawbuf_set(&hostname, 3, 97)   // 'a'
    rawbuf_set(&hostname, 4, 108)  // 'l'
    rawbuf_set(&hostname, 5, 104)  // 'h'
    rawbuf_set(&hostname, 6, 111)  // 'o'
    rawbuf_set(&hostname, 7, 115)  // 's'
    rawbuf_set(&hostname, 8, 116)  // 't'

    let out_ip = rawbuf_new(16)
    let result = resolve_ipv4(&hostname, 9, &out_ip)
    if result != 0 {
        print_int(1)
        return 1
    }
    // "localhost" must resolve to 127.0.0.1 -- check the first byte is '1'
    if rawbuf_get(&out_ip, 0) != 49 {
        print_int(2)
        return 1
    }
    print_int(0)
    return 0
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/net_dns_resolve.sio`
Expected: FAIL — `stdlib/net/dns.sio` doesn't exist yet.

- [ ] **Step 4: Implement `stdlib/net/dns.sio`**

Implement `resolve_ipv4` using whichever file-reading approach Step 1 confirmed works, parsing the file's content line by line: skip lines starting with `#`, split each remaining line on whitespace, and check whether the hostname token matches (byte-for-byte) the requested hostname; if it does, parse and return the IP token (first whitespace-separated field) into `out_ip`. Use `pub const ERR_DNS_RESOLVE: i64 = 0 - 1` for "not found" or any I/O failure.

- [ ] **Step 5: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/net_dns_resolve.sio`
Expected: PASS, prints `0`.

- [ ] **Step 6: Commit**

```bash
git add stdlib/net/dns.sio tests/run-pass/net_dns_resolve.sio
git commit -m "feat(net): add /etc/hosts-based hostname resolution"
```

---

## Task 4: `stdlib/net/http_client.sio` — HTTP/1.1 GET, tested against a hand-rolled loopback server

**Files:**
- Create: `stdlib/net/http_client.sio`
- Test: `tests/run-pass/net_http_client_localhost.sio`

**Interfaces:**
- Consumes: everything from Task 1's `stdlib/net/socket.sio` and Task 3's `stdlib/net/dns.sio`.
- Produces: `pub struct HttpResponse { status: i64, headers: RawBuf, body: RawBuf, body_len: i64, ok: bool }`, `pub fn http_get(host: &RawBuf, host_len: i64, port: u16, path: &RawBuf, path_len: i64) -> HttpResponse with IO`.

- [ ] **Step 1: Write the failing test using `fork()`-based concurrency**

```sio
//@ run-pass

use socket::*
use http_client::*

fn reap_child(pid: i64) with IO {
    if pid > 0 {
        syscall6(62, pid, 9, 0, 0, 0, 0)   // kill(pid, SIGKILL)
        syscall6(61, pid, 0, 0, 0, 0, 0)   // wait4(pid, ...) -- blocks until reaped
    }
}

fn main() -> i64 with IO {
    let server_ip = rawbuf_new(16)
    rawbuf_set(&server_ip, 0, 49)
    rawbuf_set(&server_ip, 1, 50)
    rawbuf_set(&server_ip, 2, 55)
    rawbuf_set(&server_ip, 3, 46)
    rawbuf_set(&server_ip, 4, 48)
    rawbuf_set(&server_ip, 5, 46)
    rawbuf_set(&server_ip, 6, 48)
    rawbuf_set(&server_ip, 7, 46)
    rawbuf_set(&server_ip, 8, 49)

    let (listener, listen_err) = tcp_listen(&server_ip, 18092, 1)
    if listen_err != 0 {
        print_int(1)
        return 1
    }

    let pid = syscall6(57, 0, 0, 0, 0, 0, 0)   // fork()
    if pid == 0 {
        // Child: accept one connection, drain the request, send a fixed response.
        let (conn, accept_err) = tcp_accept(&listener)
        if accept_err == 0 {
            let drain_buf = rawbuf_new(4096)
            tcp_recv(&conn, &drain_buf, 4096)
            let response = rawbuf_new(64)
            let text = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
            // Copy text's bytes into `response` byte-by-byte via str_char_at
            // (a confirmed-working body-less extern) -- fill in the loop here.
            tcp_send(&conn, &response, 44)
            tcp_close(conn)
        }
        syscall6(231, 0, 0, 0, 0, 0, 0)   // exit_group(0) -- child never falls through
        return 0
    }

    // Parent: drive http_get against the server the child is running.
    let host = rawbuf_new(16)
    rawbuf_set(&host, 0, 49)
    rawbuf_set(&host, 1, 50)
    rawbuf_set(&host, 2, 55)
    rawbuf_set(&host, 3, 46)
    rawbuf_set(&host, 4, 48)
    rawbuf_set(&host, 5, 46)
    rawbuf_set(&host, 6, 48)
    rawbuf_set(&host, 7, 46)
    rawbuf_set(&host, 8, 49)

    let path = rawbuf_new(4)
    rawbuf_set(&path, 0, 47)   // '/'

    let response = http_get(&host, 9, 18092, &path, 1)
    tcp_close(listener)
    reap_child(pid)

    if response.ok != true {
        print_int(2)
        return 1
    }
    if response.status != 200 {
        print_int(3)
        return 1
    }
    print_int(0)
    return 0
}
```

`syscall6`'s exact numbers here: `fork=57`, `kill=62`, `wait4=61`, `exit_group=231` (standard x86-64 Linux syscall table, same as the prior implementation's confirmed-working values — these are kernel ABI constants, unaffected by which Sounio compiler is used). The `reap_child` guard (`if pid > 0`) is mandatory, not optional — it is the fix for a real, previously-discovered bug where calling `kill`/`wait4` with a non-positive PID (e.g. if `fork()` ever returns `-1` on failure) becomes a `kill(-1, SIGKILL)` broadcast to every signalable process, far worse than the bug it was meant to fix.

- [ ] **Step 2: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/net_http_client_localhost.sio`
Expected: FAIL — `stdlib/net/http_client.sio` doesn't exist yet.

- [ ] **Step 3: Implement `stdlib/net/http_client.sio`**

```sio
use socket::*
use dns::*

pub struct HttpResponse {
    status: i64,
    headers: RawBuf,
    body: RawBuf,
    body_len: i64,
    ok: bool,
}

fn is_numeric_ip(host: &RawBuf, host_len: i64) -> bool with IO {
    var i: i64 = 0
    while i < host_len {
        let c = rawbuf_get(host, i)
        if (c < 48 || c > 57) && c != 46 {
            return false
        }
        i = i + 1
    }
    true
}

// `ok` means "a well-formed status line + Content-Length body was received
// and parsed" -- NOT that the HTTP status code itself indicates success.
// A 404 or 500 response is `ok: true`; check `status` separately.
pub fn http_get(host: &RawBuf, host_len: i64, port: u16, path: &RawBuf, path_len: i64) -> HttpResponse with IO {
    var resolved_ip = rawbuf_new(16)
    var i: i64 = 0
    if is_numeric_ip(host, host_len) {
        while i < host_len {
            rawbuf_set(&resolved_ip, i, rawbuf_get(host, i))
            i = i + 1
        }
    } else {
        let dns_result = resolve_ipv4(host, host_len, &resolved_ip)
        if dns_result != 0 {
            return HttpResponse { status: 0 - 1, headers: rawbuf_new(1), body: rawbuf_new(1), body_len: 0, ok: false }
        }
    }

    let (sock, connect_err) = tcp_connect(&resolved_ip, port)
    if connect_err != 0 {
        return HttpResponse { status: 0 - 1, headers: rawbuf_new(1), body: rawbuf_new(1), body_len: 0, ok: false }
    }

    // Build "GET <path> HTTP/1.1\r\nHost: <host>\r\nConnection: close\r\n\r\n"
    // byte-by-byte into a RawBuf using rawbuf_set. Fill in the concatenation
    // logic here -- straightforward index bookkeeping, no known syntax risk.
    let request = rawbuf_new(512)
    let request_len: i64 = 0   // replace with the real accumulated length

    let sent = tcp_send(&sock, &request, request_len)
    if sent < 0 {
        tcp_close(sock)
        return HttpResponse { status: 0 - 1, headers: rawbuf_new(1), body: rawbuf_new(1), body_len: 0, ok: false }
    }

    let response_buf = rawbuf_new(4096)
    let received = tcp_recv(&sock, &response_buf, 4096)
    tcp_close(sock)

    if received <= 0 {
        return HttpResponse { status: 0 - 1, headers: rawbuf_new(1), body: rawbuf_new(1), body_len: 0, ok: false }
    }

    // Parse: find "\r\n\r\n" (bytes 13,10,13,10) to split headers from body;
    // parse the numeric status code from the status line (first space to
    // second space); find "Content-Length:" in the header block and parse
    // its integer value; copy the body bytes (from the header/body boundary
    // to header_boundary + content_length) into a fresh RawBuf sized to
    // content_length. Fill in this parsing logic completely using
    // rawbuf_get for byte-level scanning -- do not hardcode a result.

    HttpResponse { status: 200, headers: rawbuf_new(1), body: rawbuf_new(1), body_len: 0, ok: true }   // replace with real parsed values
}
```

The parsing logic's placeholders (`request_len: i64 = 0`, the hardcoded final `HttpResponse` literal) are scaffolding for this plan document only — the committed implementation must contain real, working byte-scanning logic (using `rawbuf_get` in loops to find `\r\n\r\n`, extract the status code, find and parse `Content-Length`, and copy the body bytes), not these placeholder values. Do not commit code that only passes the test by coincidence of hardcoded values matching the fixture's exact response.

- [ ] **Step 4: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/net_http_client_localhost.sio`
Expected: PASS, prints `0`, exit code 0. Run it 5 times in a row to check for `fork()`/port-reuse flakiness: `for i in 1 2 3 4 5; do ./bin/souc run tests/run-pass/net_http_client_localhost.sio; done` (run as 5 separate simple commands if the harness rejects shell loops — one invocation per command is fine).

- [ ] **Step 5: Run the full new test group together**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix net_`
Expected: all `net_*` tests pass. Also run the full suite once to check for regressions: `bash scripts/run_sio_test_suite.sh`.

- [ ] **Step 6: Commit**

```bash
git add stdlib/net/http_client.sio tests/run-pass/net_http_client_localhost.sio
git commit -m "feat(net): add HTTP/1.1 GET client with fork()-based loopback test"
```

---

## Self-Review Notes

**Spec coverage:** all three modules (`socket.sio`, `dns.sio`, `http_client.sio`) map to one task each, matching the spec's architecture section exactly. The spec's testing strategy (loopback round-trip, linear compile-fail test, DNS resolve, full HTTP round-trip with fork-based concurrency) maps 1:1 onto Tasks 1-4. The spec's three "Open Implementation-Time Questions" are each addressed by an explicit verify-first step: `rawbuf_get`'s syntax (Task 1 Step 1), `&TcpSocket` shared-reference sufficiency across repeated calls (built into Task 1's function signatures and exercised for real by Task 4's multi-call `http_get` flow — if this doesn't hold, Task 4's implementation will fail to compile and that's the empirical answer), and `read_file` vs. raw syscalls for `/etc/hosts` (Task 3 Step 1).

**Placeholder scan:** Task 4's implementation skeleton contains two explicitly-flagged, non-committable placeholders (`request_len: i64 = 0`, the hardcoded final `HttpResponse`) — the step's own text states plainly these must be replaced with real logic, matching the discipline used in the prior implementation's equivalent task.

**Type consistency:** `RawBuf`, `TcpSocket`, and every `rawbuf_*`/`tcp_*`/`resolve_ipv4`/`http_get` signature are declared once (in the task that owns the file) and referenced identically by name and signature in every later task's Interfaces block and code — in particular, `tcp_send`/`tcp_recv` consistently take `&TcpSocket` (not a consumed value) everywhere they're called across Tasks 1 and 4, and `HttpResponse`'s five fields (`status`, `headers`, `body`, `body_len`, `ok`) are used consistently in both the test and the implementation skeleton.

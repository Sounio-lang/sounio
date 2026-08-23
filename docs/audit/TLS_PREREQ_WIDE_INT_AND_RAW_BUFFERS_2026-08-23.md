<!-- docs:meta
topic_id: repo.docs.audit.tls-prereq-wide-int-and-raw-buffers-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: claude-sonnet-5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.tls-prereq-wide-int-and-raw-buffers-2026-08-23
-->

# TLS prerequisite audit — wide-int generality and raw buffer access on Madaros v0.80.0

**Status:** two blocking gaps found for general-purpose bignum (RSA) and raw-syscall
(socket) work. Neither is a regression of a previously-working feature — both are
scope limits of features that were never claimed to cover this use case.

## Context

This audit was run while scoping a from-scratch TLS 1.2 implementation for Sounio
(part of a larger custom web-crawler project), specifically to determine whether
`i256`/`i512` (landed in #2054, see `docs/audit/R1_I256_I512_LIMBS_2026-08-20.md`)
and local-array-to-raw-pointer casts (used by a prior TCP-socket implementation on
an older, now-superseded compiler branch) are usable as-is on Madaros.

## Finding 1 — `i256`/`i512` multiply is correct only for the narrow pattern #2054 verified, not in general

`docs/audit/R1_I256_I512_LIMBS_2026-08-20.md` documents `i256`/`i512` as
"implemented," verified against one fixture (`r1_i256_lorenz_peak.sio`,
`217041893 * 2**65`) with a positive-control sabotage test. That verification is
real and stands for that fixture. It does **not** generalize:

```sio
fn main() -> i64 {
    let x: i256 = 4294967296 as i256   // 2^32
    let y = x * x                      // should be 2^64
    ((y >> 64) as i64)                 // expect 1
}
```

This returns `0`, not `1` — the same "silently wrong, not just unimplemented"
shape as the `i256`/`i512` situation documented for the prior compiler lineage.
The R1 audit itself explains why: "Wide values are still consecutive vregs plus
the intern pool for immediates; payloads are not yet the runtime representation,"
and "`print_int` of a wide local is unsafe (clobbers)." The feature covers the
specific compile-time-interned-immediate pattern its own fixture exercises, not
general runtime `wide * wide` multiplication with arbitrary operands.

**Consequence for this project:** `i256`/`i512` are not usable as the basis for
RSA modular exponentiation (2048/4096-bit) or any general bignum work yet. A
hand-rolled limb-array `BigInt` over `i64`/`u64` (proven correct via the same
kind of guard-value/round-trip tests used here) remains the only currently-honest
path, exactly as scoped for the big-integer sub-project before this compiler
was known to have any wide-int work at all.

**Do not** revisit this without a fresh positive-control-style test matching the
actual operand pattern needed (arbitrary runtime values, not compile-time
immediates) — the R1 team's own methodology (a sabotage flag that must make the
test fail) is the right bar to clear before trusting this for RSA.

## Finding 2 — local/stack `[T;N]` arrays are boxed GC handles; a raw pointer to one does not expose the flat buffer

```sio
fn main() -> i64 {
    var arr: [u8; 4] = [0; 4]
    arr[0] = 88   // 'X'
    arr[1] = 89   // 'Y'
    let p = (&!arr) as *mut u8
    syscall6(1, 1, p, 2, 0, 0, 0)   // write(stdout, p, 2)
    return 0
}
```

The bytes actually written to fd 1 were `\x01\x00`, not `X Y` — confirmed via
`od -c` on captured stdout. This holds for both a bare local array and a
struct field of array type. The compiler's own existing test comments state
local fixed arrays are "GC handles with 8-byte boxed element slots" — casting
the handle value to `*mut u8` exposes handle/metadata bytes, not the
contiguous byte buffer a syscall needs (e.g. a 16-byte `sockaddr_in`, or a
receive buffer for `recv`/`read`).

**Workaround for raw-syscall buffer needs (sockets, wire-format parsing):**
allocate the buffer via `heap_alloc` (one of the working body-less extern
names) instead of a local `[u8;N]`. **Confirmed working** (not just assumed):

```sio
extern "C" {
    fn heap_alloc(n: i64) -> *mut u8;
}
fn main() -> i64 with IO {
    let p = heap_alloc(4)
    *p = 88          // 'X' -- single-byte deref-write through the raw pointer
    syscall6(1, 1, p, 1, 0, 0, 0)   // write(stdout, p, 1)
    return 0
}
```

Running this actually writes the byte `X` to real stdout (independently
verified, not just a nonzero exit code) — `heap_alloc`'s returned pointer is
a genuine flat address, unlike a local array's boxed handle. Indexing the
pointer directly with `p[i]` does NOT work (`error[E013]: indexing requires
an array type` — raw pointers are not indexable), and direct pointer
arithmetic `p + i` does not compile either (`error[E004]: these types cannot
be combined with this operator` — no `*mut u8 + i64` operator exists).

**Confirmed multi-byte pattern** (cast to `i64`, do the arithmetic as an
integer, cast back to a pointer):

```sio
let p = heap_alloc(4)
let addr = p as i64
let p2 = (addr + 1) as *mut u8
*p = 88     // byte 0
*p2 = 89    // byte 1
```

Verified: this actually writes `XY` to real stdout via `syscall6` (write),
not just a clean compile. This is the pattern to use for constructing any
packed multi-byte buffer (e.g. a 16-byte `sockaddr_in`) on Madaros — offset
arithmetic goes through an `i64` intermediate, never directly on the
pointer type.

## Finding 3 — `rawbuf_get`-style pointer dereference reads a full word, not one byte

`(*p) as i64` where `p: *mut u8` does not perform a single-byte read — it reads a
full (up to 8-byte) word starting at the pointee address, little-endian,
zero-filled past whatever memory happens to follow. Confirmed: writing `88` at
offset 0 and `89` at offset 1 of a `heap_alloc`'d buffer, then reading back
`(*p) as i64`, returns `22872` (`0x5958` — both bytes packed little-endian),
not `88`. **Always mask a single-byte pointer read to its low byte**:
`((*p) as i64) & 255`. A related, smaller finding: `*p = v` requires an
explicit `as u8` cast when `v` is a typed (non-literal) value — a bare
integer literal like `*p = 88` infers as `u8` and compiles fine, but
`*p = some_i64_variable` fails with `error[E002]` until cast.

## Finding 4 — the linear-type checker treats ANY use of a linear binding, including a shared borrow, as full consumption

A design that borrows a linear value multiple times (e.g. `&h` for one
operation, then a plain move for a final closing operation) is expected to
work under normal linear-typing semantics (a shared borrow doesn't consume).
On this compiler it does not: `peek(&h); peek(&h)` on one `linear struct`
binding `h` fails on the SECOND call with `error[E039]: linear value has
already been used` — even though neither call takes `h` by value. Any
function signature taking `&SomeLinearStruct` and being called more than
once on the same binding (or once after any other operation already touched
that binding) will hit this.

**Workaround**: value-thread the resource instead of borrowing it. Every
function touching the linear value takes it BY VALUE, destructures it to
pull out the field(s) it needs, and returns a FRESHLY CONSTRUCTED instance
of the same linear type alongside its real result — never reusing the
original binding for a second operation. This matches the pre-existing
style already used elsewhere in this repo (e.g. `self-hosted/llvm/context.sio`'s
"each operation consumes ... and returns a new one"). Concretely, a
`tcp_send(sock: &TcpSocket, ...)` signature must instead be
`tcp_send(sock: TcpSocket, ...) -> (TcpSocket, i64)`, with every call site
rebinding the returned socket.

A related restriction (source: `self-hosted/check/check.sio:6010`, the
compiler's own comment, explicitly calling this conservative-but-sound):
**a linear value must be consumed identically in every arm of an `if`, or in
neither** — even `if c { use(t) } else { use(t) }` (consuming it in BOTH
arms) trips `error[E039]`. This blocks the common "check an error code,
early-return with cleanup" pattern for any code holding a linear resource
across a conditional. Workaround used so far: restructure the code as one
unconditional straight-line sequence (no `if` ever touches the linear
binding) and check the resulting plain-value error codes/outputs only AFTER
every linear resource has already been fully consumed/closed — accepting
that a failure partway through the sequence is still detected (the
downstream operations on an invalid resource produce their own
distinguishable failure values), just reported after full teardown rather
than via early exit.

## Finding 5 — `syscall6(N, linear_value.field, ...)` does not register as consuming the linear value

Calling `syscall6` directly on a field of a linear-struct parameter —
`syscall6(3, sock.fd, 0, 0, 0, 0, 0)` where `sock: TcpSocket` (a `linear
struct`) — fails to compile with `error[E040]: linear value not consumed (1
unconsumed)`, with NO diagnostic location or other hint pointing at the
actual cause. This is despite the function body containing no other
statement, and despite `sock.fd` clearly being "used." Bisection confirmed:
extracting the field to a plain local FIRST (`let fd = sock.fd; syscall6(3,
fd, 0, 0, 0, 0, 0)`) compiles clean — the destructuring `let` is what the
checker recognizes as consuming `sock`, not the inline field access inside
the `syscall6` call. `syscall6` is presumably handled as a compiler
intrinsic on a code path that bypasses the checker's normal
argument-consumption tracking used for ordinary function calls.

**Rule going forward**: always bind a linear struct's field to a local
variable before passing it to `syscall6` (or any other body-less/intrinsic
extern); never pass `linear_param.field` inline as a `syscall6` argument.
This generalizes beyond sockets to any future linear-resource type whose
consuming operations call `syscall6` directly.

## Finding 6 — module import path for a file inside `stdlib/net/` is `net::<filename>::*`, not a bare `<filename>::*`

A program outside `stdlib/` importing `stdlib/net/socket.sio` must write
`use net::socket::*` (the directory/filename path relative to `stdlib/`) —
the bare form `use socket::*` fails with `error[E137]: use of undeclared
variable`. This matches the existing convention used elsewhere in this repo
(e.g. `stdlib/algebra/cayley_dickson_exact.sio` is imported as `use
algebra::cayley_dickson_exact::{...}`) — confirm the correct qualified path
by checking how an existing, working stdlib module of the same shape is
actually imported, rather than assuming a bare filename import will resolve.

## Scope note

Neither finding blocks this project — both have workarounds (hand-rolled
limb-array bignum; heap-allocated raw buffers instead of local arrays) and
neither represents a regression. They are recorded here so the eventual
TLS/bignum implementation plan does not silently assume either capability
works when it verifiably does not yet, in the same spirit as the prior
compiler lineage's `docs/compiler/KNOWN_LIMITATIONS.md` entries.

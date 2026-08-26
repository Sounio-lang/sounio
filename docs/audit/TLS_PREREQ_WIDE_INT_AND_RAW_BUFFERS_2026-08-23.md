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

## Finding 7 — `read_file` is a no-`extern` compiler builtin taking `string` directly

`read_file`, `str_len`, and `str_char_at` are compiler builtins requiring NO
`import` and NO `extern "C"` block at all — declaring `extern "C" { fn
read_file(path: [i8; 32]) -> string; }` and calling it with a string literal
fails with `error[E009]: argument type does not match parameter`. The
correct, confirmed-working form is a bare call with no declaration:

```sio
fn main() -> i64 with IO {
    let content: string = read_file("/etc/hosts")
    let n: i64 = str_len(content)
    print_int(n)
    return 0
}
```

Verified against real `/etc/hosts` (576 bytes, matching `wc -c` exactly) and
`str_char_at(content, i)` returning the correct ASCII byte values for the
file's real first line. This form is safe for READS — a separate,
`write_file`-specific write-corruption risk exists (passing a `string` as
`write_file`'s buffer argument compiles and returns a plausible byte count
while silently corrupting the file's actual on-disk content; only the
array-buffer form is safe for writes) but does not apply here, since
`read_file` only ever produces a `string`, never consumes one as a write
buffer.

## Finding 8 — sibling-file imports within the same stdlib subdirectory use the bare unqualified form

Finding 6 above is scoped to a program OUTSIDE `stdlib/` importing INTO
`stdlib/net/` (`use net::socket::*`). For a file that itself lives in
`stdlib/net/` importing another file in the SAME directory (e.g.
`stdlib/net/dns.sio` importing `stdlib/net/socket.sio`), the correct and
only-working form is the bare `use socket::*` — confirmed via a throwaway
spike. General pattern: import paths are relative to the directory a file
lives in for sibling imports, and `net::`-prefixed (relative to `stdlib/`)
only when crossing into `stdlib/net/` from outside it.

## Finding 9 — destructuring a linear value before a branch lets each arm rebuild it fresh, sidestepping the "both/neither arm" restriction

Finding 4's second paragraph is exactly as strict as stated: even fully
symmetric consumption (`if c { tcp_close(x) } else { tcp_close(x) }`) fails
with `error[E039]` if `x` is a linear binding alive *before* the `if`. This
blocks any branch (e.g. `fork()`'s `if pid == 0 { ... } else { ... }`) that
needs to use an already-held linear resource differently in each arm.

**Workaround**: destructure the linear value into its plain (non-linear)
payload field(s) ONCE, unconditionally, before the branch — then branch
freely on ordinary values, and reconstruct a FRESH instance of the linear
type from the saved payload inside each arm that needs it:

```sio
let (listener, listen_err) = tcp_listen(&server_ip, port, 1)
let TcpSocket { fd: listener_fd } = listener   // plain i64, not linear anymore

let pid = syscall6(57, 0, 0, 0, 0, 0, 0)   // fork()
if pid == 0 {
    let child_listener = TcpSocket { fd: listener_fd }   // fresh, arm-local
    // ... use child_listener, consume it fully within this arm ...
} 
let parent_listener = TcpSocket { fd: listener_fd }   // fresh, this-side-local
// ... use parent_listener ...
```

Each `TcpSocket { fd: listener_fd }` literal is created and fully consumed
entirely within its own scope (never a binding alive across the `if`), so
neither trips the both/neither-arm restriction. Also confirmed: constructing
a struct literal for a type whose field has no explicit `pub` marker
compiles cleanly from OUTSIDE the defining module even though the struct
itself is `pub` — Madaros does not appear to enforce field-level privacy
independent of the struct's own visibility.

## Resolved: `tcp_listen` now sets `SO_REUSEADDR` (was: known gap — TIME_WAIT port-reuse flakiness)

Running a test that calls `tcp_listen` on a fixed port, then immediately
reruns the same test on the same port with no delay, used to fail the
second run's `tcp_listen` with a nonzero error — confirmed via `ss -tan`
showing lingering TIME_WAIT entries on both the client and server side
after a run (both ends actively close in quick succession, a
simultaneous-close pattern). This cleared within a few seconds on its own
(well under the traditional 60s default), so it was NOT a hang — but any
test suite that reruns fixed-port socket tests back-to-back could flake.
**Mitigation used while this was open**: distinct ports across test files
likely to run close together in time.

**Fixed** in commit `723b87496`, which added a
`setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &1, 4)` syscall (`setsockopt` =
syscall number 54 on x86-64 Linux) between `socket()` and `bind()` in
`stdlib/net/socket.sio`'s `tcp_listen`. Distinct ports per test file remain
good practice for isolation, but the underlying TIME_WAIT flakiness this
section originally described is no longer a live gap.

## Finding 10 — ASCII dotted-decimal TEXT and raw 4-byte OCTETS are two different IPv4 representations, and mixing them up silently connects to the wrong address

`stdlib/net/dns.sio`'s `resolve_ipv4` and `stdlib/net/http_client.sio`'s
"host looks like a numeric IP" (`is_numeric_ip`) path both produce/consume
ASCII dotted-decimal TEXT — e.g. the literal bytes `'1'`, `'2'`, `'7'`,
`'.'`, `'0'`, `'.'`, `'0'`, `'.'`, `'1'` for `"127.0.0.1"`. `tcp_connect`'s
`build_sockaddr` (`stdlib/net/socket.sio`), by contrast, reads its `ip:
&RawBuf` argument as 4 raw octet BYTES directly (`rawbuf_get(ip, 0)` through
`rawbuf_get(ip, 3)` used verbatim as the address bytes of a `sockaddr_in`)
— it does not parse dotted-decimal text at all.

An implementation that conflates these two representations — e.g. copying
a numeric host's ASCII bytes straight into the buffer handed to
`tcp_connect` — compiles cleanly, returns no error, and **silently connects
to the wrong address**: the 4 bytes `'1'`, `'2'`, `'7'`, `'.'` (ASCII values
49, 50, 55, 46) get used as the raw octets `49.50.55.46`, not `127.0.0.1`.
This is exactly the kind of "silently wrong, not just unimplemented"
failure mode this audit exists to catch.

**Fix**: `stdlib/net/http_client.sio` adds `parse_ipv4_dotted(text: &RawBuf,
max_len: i64, out: &RawBuf) -> bool`, which parses ASCII dotted-decimal text
into 4 raw octets exactly once, in one place, before the result ever
reaches `tcp_connect`. Both the "already numeric" and "resolved via DNS"
branches of `http_get` produce ASCII dotted text into a common `addr_text`
buffer; `parse_ipv4_dotted` is then the single conversion point from that
text into the raw-octet buffer `tcp_connect` actually needs. Any future
code that produces or consumes an IPv4 address on this stack must be
explicit about which of the two representations it is working with —
DNS/text-facing code produces dotted-decimal TEXT, syscall-facing code
(`build_sockaddr`) consumes raw OCTETS, and the two must never be passed to
each other directly.

## Finding 11 — `u64` right-shift, division, and modulo silently use signed/arithmetic semantics whenever bit 63 is set

`>>`, `/`, and `%` on a `u64` value whose bit 63 is set produce mathematically
WRONG results — silently, no error. `+`, `-`, `*`, `&`, `|`, `^`, `<<`, `==`,
and `!=` are all confirmed bit-exact correct regardless of bit 63; only
right-shift/divide/modulo (and ordering comparisons `<`/`>`/`<=`/`>=` when
the signed and unsigned interpretations of the operands diverge) are broken.

```sio
let x: u64 = 18446744065119617025   // 0xFFFFFFFE00000001, bit 63 set
x >> 1    // WRONG (arithmetic-shifts as if signed, corrupting the result)
x >> 32   // WRONG
x >> 63   // WRONG
x / 2                // WRONG
x / 4294967296        // WRONG
x % 4294967296        // WRONG, nonsensical negative output
x & 0xFFFFFFFF        // CORRECT
x << 1                 // CORRECT, even when the result sets bit 63
x == <same value>      // CORRECT (bit-pattern comparison, not signed compare)
```

Confirmed the break is precisely at bit 63 (not "large values" generally):
a value with bit 62 set and bit 63 clear right-shifts correctly. Confirmed
`i64`'s own right-shift is internally consistent with sign-extension (as
expected for a signed type) — strongly suggesting `u64`'s `>>`/`/`/`%` share
the same arithmetic-shift/signed-divide code path as `i64` instead of using
a logical-shift/unsigned-divide path when the value's top bit is set.

**Consequence for bignum/wide-arithmetic work**: multiplying two ~32-bit
values and extracting the high 32 bits of their 64-bit product via `>> 32`
or `/ 4294967296` is UNSAFE whenever the product's bit 63 ends up set — which
happens for a large fraction of possible 32-bit×32-bit products (any pair
whose product exceeds `2^63`). **Workaround/recommendation: use 16-bit limbs
for any multi-limb (bignum) arithmetic**, not 32-bit. With 16-bit limbs,
every partial product, carry, and running sum in a schoolbook multiply stays
under roughly `2^33` — always far below bit 63 — so every shift/mask/carry
step needed is safe:

```sio
// 32-bit inputs decomposed into 16-bit halves; every intermediate below 2^33
let a_hi = a >> 16; let a_lo = a & 0xFFFF   // a itself has no bit-63 risk (32-bit input)
let b_hi = b >> 16; let b_lo = b & 0xFFFF
let p_hh = a_hi*b_hi; let p_hl = a_hi*b_lo; let p_lh = a_lo*b_hi; let p_ll = a_lo*b_lo  // each < 2^32
let mid = p_hl + p_lh                                    // < 2^33, safe to shift
let mid_lo = mid & 0xFFFF; let mid_hi = mid >> 16          // safe: mid has no bit 63
let low32 = (mid_lo << 16) + p_ll
let carry = low32 >> 32                                    // safe: low32 max ~2^33
let low32_final = low32 & 0xFFFFFFFF
let high32 = p_hh + mid_hi + carry
```

Verified this reconstructs the exact correct 64-bit product
(`0xFFFFFFFE00000001` for `4294967295 * 4294967295`) that direct `u64`
shift/divide extraction gets wrong. **A future bignum module for RSA/TLS on
this compiler should use 16-bit limbs throughout, not 32-bit** — 32-bit
limbs are only safe if every multiply is itself internally decomposed this
same way, which is equivalent to using 16-bit limbs in the first place.

## Finding 12 — the Madaros runtime arena is never reclaimed: every value-returning function that allocates is a permanent, per-process budget spend

> **UPDATE 2026-08-26 — measured on real TLS, and partly mitigated.** The
> closing paragraph below predicted this ceiling "will return for any
> long-running process (e.g. a TLS server handling many handshakes)". It
> did, far sooner than the ~460,000-call figure suggests: a process could
> complete **two** real, CA-verified TLS handshakes before exit 181, and
> some chains died on the first. Certificate-chain verification is
> unusually expensive against this budget because `Certificate` values are
> enormous and were copied by value in the hot path.
>
> That specific measurement, its cost attribution
> (`certificate_zero()` ~352 KB, `x509_parse_certificate()` ~11.5 MB,
> `x509_verify_chain()` ~30 MB per call), and its resolution — ceiling
> raised **2 → 95 handshakes/process** via commits `c9bd996b2`,
> `976e3e399` and `eea3a449f` — are dispatched separately in
> [`ARENA_EXHAUSTION_TLS_HANDSHAKE_CHAIN_VERIFICATION_DISPATCH_2026-08-26.md`](ARENA_EXHAUSTION_TLS_HANDSHAKE_CHAIN_VERIFICATION_DISPATCH_2026-08-26.md).
>
> Two corrections that finding's own numbers depend on:
>
> - **A `[u8; N]` field occupies one 8-byte slot per element, not N bytes.**
>   Every size estimate made against the logical struct size is 8× low.
> - **The arena is now 8 GiB, not 2 GiB, on Linux** (`native_v2_arena_bytes()`
>   in `self-hosted/native/gc.sio`), with the handle table at 2^24.
>
> **The defect itself is UNCHANGED and still open.** Nothing is reclaimed;
> the wall moved ~47×, it did not disappear. The guidance below stands
> exactly as written.

A Sounio function that returns a struct containing an array field (e.g.
`fn f() -> BigInt` where `BigInt` holds `[u16; 512]`) allocates a fresh block
in the Madaros runtime arena on every call, and that block is **never freed or
reused** for the life of the process. When the arena is exhausted the process
dies with an uncatchable

```
madaros: arena full
```

and exit status 181 — no `Result`, no unwinding, nothing a Sounio program can
observe or recover from.

**Measured ceiling** (`stdlib/bignum/bigint.sio`, this compiler, this branch):
a bare loop calling `bigint_add` — one `BigInt`-returning call per iteration —
completes **≈460,000 iterations** before aborting. That is the whole budget for
a process, regardless of how short-lived each value is.

**In-place field mutation is genuinely allocation-free.** This is the important
positive half of the finding, and it was verified directly:

```sio
var acc = bigint_zero()          // ONE allocation
var i = 0
while i < 5000000 {
    acc.limbs[(i % 512) as usize] = (i % 65535) as u16   // mutates in place
    acc.len = 512
    i = i + 1
}
```

5,000,000 field/element writes to the same `var` binding complete without the
arena growing at all — the array field really is a single boxed handle
(consistent with Finding 2) that is written through, not re-boxed. So the
allocation cost of an algorithm on this runtime is driven entirely by how many
**value-returning helper calls** it makes, not by how much work it does.

**Consequence, and what was done about it.** `bigint_mod`'s binary long
division originally called two `BigInt`-returning helpers
(`bigint_shl1_or_bit`, `bigint_sub`) once per bit of the dividend — up to
`2 * 8192 = 16384` arena allocations per single `bigint_mod` call. Measured
end-to-end with a 2048-bit modulus and exponent 65537:

| | RSA-2048 `bigint_modpow` calls per process | RSA-4096 `bigint_modpow` calls per process |
|---|---|---|
| helper-call version | **5** | **2** |
| in-place version | **3402** | **> 20** (no arena abort in the measured run) |

Rewriting that inner loop to shift and subtract directly on one `remainder`
binding's limbs — no helper calls, identical arithmetic, identical Finding 11
bounds — reduced `bigint_mod` from O(bits) allocations to exactly **one**, a
**~680×** improvement in RSA operations per process. Note the per-modpow
allocation cost is now dominated by `bigint_mul`'s temporaries and is
independent of operand width, which is why RSA-4096 gains as much as RSA-2048.

**Guidance for any future wide-arithmetic or buffer-heavy Sounio code on this
runtime**: treat a value-returning function that allocates as an expensive,
non-refundable operation. Prefer mutating a single `var` binding across a loop
over calling a helper that returns a fresh value each iteration. The ceiling
has not been removed — it has been pushed roughly three orders of magnitude
out — and it will return for any long-running process (e.g. a TLS server
handling many handshakes) until Madaros gains arena reclamation or this module
is restructured around explicit caller-supplied scratch buffers.

## Finding 13 — native `u32` arithmetic does not wrap mod 2^32 at all: `+` and `-` silently promote to unbounded/signed precision

Discovered while auditing `u32` for the hash-functions sub-project (SHA-1/
SHA-256), before any hash code was written to depend on it. This is a
different, more fundamental failure than Finding 11: Finding 11 is about
`u64` `>>`/`/`/`%` breaking specifically when bit 63 is set, with `+`/`-`
confirmed fine for `u64`. Here, plain `u32` `+`/`-` — no shift, no divide,
no bit-63-scale value involved at all — do not wrap to 32 bits.

```sio
let x: u32 = 4294967295   // 0xFFFFFFFF, u32::MAX
let y: u32 = 1
let z = x + y
(z as i64) == 4294967296   // TRUE -- the unbounded/full-precision sum
z == 0                      // FALSE -- correct u32 wraparound would give 0

let d: u32 = 0
let e: u32 = 1
let f = d - e
(f as i64) == -1            // TRUE -- signed underflow, not wraparound
(f as i64) == 4294967295    // FALSE -- correct u32 wraparound would give 0xFFFFFFFF
```

Also confirmed non-power-of-two-adjacent operands reproduce it the same way
(`3000000000u32 + 3000000000u32` as `i64` is `6000000000`, the unbounded
sum, not the correct wrapped `1705032704`) — this is not specific to
operands sitting exactly at the type's boundary value; `u32 + u32`/`u32 -
u32` simply never truncates to 32 bits on this compiler, for any operands.
Right-shift/rotate/bitwise-op correctness at 32-bit width was not
separately isolated once `+` was found broken (the audit stopped at the
first failing case, per this branch's "stop and report, don't chain
workarounds past an unexplained failure" discipline) — treat those as
unverified at native `u32` width too, not confirmed-safe by omission.

**Workaround/recommendation: do not use native `u32` for any arithmetic
that must wrap correctly at 32 bits.** Represent 32-bit words as a plain
`i64`, explicitly masked (`& 0xFFFFFFFF`) after every operation that could
exceed 32 bits (addition, left shift) — the exact same discipline already
used for `BigInt`'s 16-bit limbs (Finding 11) and for this project's own
64-bit hash words (`stdlib/hash/word64.sio`, using 32-bit-half decomposition
of a `u64`-scale value). Masked `i64` arithmetic bounded to 0..4294967295
never approaches `i64`'s own bit-63 danger zone (Finding 11 does not apply),
and plain `i64` `+`/`-`/shift/bitwise-ops at this magnitude have been
independently, repeatedly confirmed correct elsewhere on this branch.
Verified: `stdlib/hash/word32.sio`'s `add32`/`rotl32`/`rotr32`/`shr32`/
`xor32`/`and32`/`or32`/`not32`, each computing on masked `i64` values,
reproduce the correct results for every case Finding 13's own reproducer
above gets wrong under native `u32` — see
`tests/run-pass/hash_word32_primitives.sio`.

## Scope note

Neither finding blocks this project — both have workarounds (hand-rolled
limb-array bignum; heap-allocated raw buffers instead of local arrays) and
neither represents a regression. They are recorded here so the eventual
TLS/bignum implementation plan does not silently assume either capability
works when it verifiably does not yet, in the same spirit as the prior
compiler lineage's `docs/compiler/KNOWN_LIMITATIONS.md` entries.

## Finding 14 — a narrowing `i64 as u8` cast does not truncate to the low 8 bits when the source value exceeds 255

Discovered while implementing SHA-1's digest-byte-extraction (`stdlib/hash/
sha1.sio`), which needs to take an `i64`-stored 32-bit word and write it out
as 4 individual bytes via `(word >> shift) as u8`-style narrowing casts —
the universal, standard idiom for big-endian byte extraction in essentially
every C-family language. On Madaros, this idiom silently produces a
corrupted result whenever the value being cast exceeds 255: the cast does
not mask to the low 8 bits, and the out-of-range bits appear to leak into
adjacent storage (observed as if reading the "byte" back later returned a
16-bit value, spilling into a neighboring array element).

```sio
fn shr32(x: i64, n: i64) -> i64 { x >> n }
fn main() with IO {
    var out: [u8; 4] = [0; 4]
    let h: i64 = 3661210606   // 0xDA39A3EE
    out[1] = shr32(h, 16) as u8   // should store 0x39 (57) -- the byte at
                                    // bit position 16-23 of h
    println(out[1] as i64)         // ACTUAL: 55865 (0xDA39, i.e. 16 bits,
                                    // not truncated to 8) -- WRONG
}
```

**Workaround: always mask explicitly with `& 255` immediately before an
`as u8` cast, whenever the source value could exceed 255** — do not rely on
the cast itself to truncate, regardless of how standard that idiom is in
other languages:

```sio
out[1] = (shr32(h, 16) & 255) as u8   // CORRECT: prints 57
```

Confirmed the masked form produces the correct byte value; confirmed the
unmasked form reproduces the corruption deterministically (not a flaky/
timing-dependent issue). Not yet tested whether this generalizes to other
narrowing casts (e.g. `i64 as u16`, `i64 as i32`) — only the `as u8` case
that this project's hash-functions sub-project actually needed was
isolated. **Any future Sounio code doing narrowing-cast-based byte
extraction (serializing a wide integer to bytes) must mask explicitly
before every narrowing cast, never rely on the cast to truncate.**

## Finding 15 — a top-level `const` array with more than 16 elements breaks Madaros's native-v2 IR lowering

Discovered while implementing SHA-256's 64-entry round-constant table
(`stdlib/hash/sha256.sio`). A top-level `const` array declaration compiles
and runs correctly up to 16 elements; at 17 elements and above, compilation
fails during native-v2 lowering with an internal compiler error, not a
normal diagnostic:

```sio
const K17: [i64; 17] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
fn main() with IO {
    var sum: i64 = 0
    var i: i64 = 0
    while i < 17 { sum = sum + K17[i as usize]; i = i + 1 }
    println(sum)
}
```
Fails with:
```
error: IR instruction arena contract violated (invalid handle) on region slot 1 generation 1
Error: refusing to write a binary built on a violated IR arena contract
error: native-v2 bridge compilation failed
```
Confirmed independently (both by the implementer who found it and, separately,
by the controller re-running the same minimal repro). A 16-element top-level
const array compiles and runs fine; the failure appears exactly at 17. The
same 64-element table embedded in a larger multi-module program failed with
the identical error class at a different, deeper region slot — same root
cause, not a coincidence of a small isolated repro.

**Workaround: move any array with more than 16 literal elements out of a
top-level `const` and into a function that returns the array literal**,
called once into a local `let`/`var` binding at the point of use:

```sio
fn k17_table() -> [i64; 17] {
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
}
fn main() with IO {
    let k = k17_table()
    // k[i] indexing works exactly as a top-level const array would
}
```
Confirmed this form compiles and runs correctly for a 64-element array (the
real `SHA256_K` table) and separately for a bare 64-element local literal.
This is a placement-only workaround with no semantic change — the array's
values, order, and indexing behavior are unaffected; only where the literal
is declared changes. **Any future Sounio module needing a lookup table
larger than 16 entries (round constants, S-boxes, permutation tables, etc.)
must use this function-returning-a-literal form, never a bare top-level
`const` array.**

## Finding 16 — 64-bit-scale arithmetic audit for `stdlib/hash/word64.sio`: mitigated by construction, not independently re-triggered

Task 4 of the hash-functions plan opened Phase 2 (SHA-384/512 prerequisites)
by auditing whether Finding 11's failure pattern (`u64` right-shift/divide/
modulo breaking when bit 63 is set) also shows up at the specific 64-bit-
scale arithmetic a hash module needs (add, xor, and, not, logical right
shift, rotate-right), and by building `stdlib/hash/word64.sio` to handle it
regardless of outcome.

`word64.sio` represents every 64-bit logical value as two independent `i64`
scalars (`hi`, `lo`, each masked to `0..0xFFFFFFFF` after every operation
that could exceed 32 bits) rather than a single 64-bit-wide value — the
same 32-bit-half decomposition `bigint.sio` uses for 16-bit limbs, applied
here at limb width 32. By construction, no intermediate value in any of
`add64`, `xor64`, `and64`, `not64`, `shr64`, `shl64`, or `rotr64` ever
approaches bit 63 of the underlying `i64` representation — the largest
intermediate is `a_hi << 31` in `shr64`'s `n=1` case, which lands at bit 62
at most, still one full bit below Finding 11's danger zone.

`tests/run-pass/hash_word64_primitives.sio` exercises this deliberately at
the case BigInt never needed: operands with the top bit of a half set
(`2147483648` = `0x80000000`), plus the two carry-propagation cases
(full 64-bit wraparound and a carry crossing exactly from the low half into
the high half) and the two shift/rotate boundary cases (`shr64` crossing
the 32-bit half boundary at `n=33`, `rotr64` at exactly `n=32` swapping
halves, and `rotr64` at `n=1` carrying the low bit of the low half into the
top bit of the high half). All nine assertions passed on the first attempt,
with no need for any fix to the transcribed code:

```
$ export SOUNIO_STDLIB_PATH=<repo>/stdlib
$ ./bin/souc run tests/run-pass/hash_word64_primitives.sio
...
hash_word64_primitives: all cases passed
$ echo $?
0
```

This is the **expected, defended-against outcome from Step 2's design**,
not the alternative outcome flagged in the task brief (native `i64`/`u64`
arithmetic being safe at this scale without masking). This run does **not**
independently re-confirm Finding 11 at 64-bit scale, because the masking
discipline in `word64.sio` means no operation here ever executes the
unmasked, bit-63-adjacent arithmetic Finding 11 describes — the mitigation
was applied unconditionally from the start rather than discovered by first
observing a failure and then patching around it. Finding 11's original
scope (`u64` right-shift/divide/modulo with bit 63 set) stands unchanged
and is not re-tested or re-scoped by this finding. Evidence:
`tests/run-pass/hash_word64_primitives.sio`.

## Finding 17 — a string literal longer than ~126 content characters is silently truncated at compile time, and can crash the compiler's own output stage

Discovered while writing a test embedding a 128-hex-character SHA-512 digest
(130 bytes including the surrounding quotes) as a single string-literal
argument. This is well past any string length prior tasks' tests needed
(SHA-256's 64-hex-char/66-byte literals never approached it).

The compiler emits a warning, not an error, and still reports
"Compilation successful!":
```
warning: 3 string literal(s) TRUNCATED; first at line 39, longest 130 bytes.
         Name holds 128 including quotes, so the text past 126
         characters is gone -- silently, at run time.
```
Confirmed independently with a minimal repro (a single 130-byte string
literal assigned to a `let` and passed to `str_len`): the warning fires
consistently, and in this specific minimal case **the compiler's own
native-output stage segfaults** after printing "Compilation successful!"
(`bin/madaros: line 681: <pid> Segmentation fault "$out" "$@"`) rather than
running the resulting program at all. The exact runtime behavior for a
truncated-but-not-crashing case (as apparently happened in the
non-minimal Task 5 test file, which the implementer worked around before
this was isolated) was not separately characterized here -- treat any
string literal over roughly 126 characters as unconditionally unsafe,
whether or not a given instance happens to crash immediately.

**Workaround: never write a string literal longer than ~120 characters
directly in Sounio source.** Split any longer text (a long hex digest, a
PEM block, a long URL, etc.) into multiple shorter literals and
concatenate/compare them piecewise at the call site — e.g. a 128-hex-char
expected digest becomes two 64-hex-char literals compared against the
first and second halves of the actual value separately. `tests/run-pass/
hash_sha512_vectors.sio` does exactly this (`assert_digest_hex_half`
called twice per digest, at byte offsets 0 and 32) and is the first
committed code to depend on this workaround. **Any future Sounio code or
test embedding a long fixed string (certificate PEM data, long URLs,
multi-line templates) must apply the same split-and-compare/split-and-
concatenate pattern** -- do not assume a "long string literal" is safe up
to some larger, untested threshold.

## Finding 18 — a top-level `const [u8; N]` array is corrupted whenever its whole value is used (copied to a `var`, or address-taken and passed by reference); direct indexed reads of the same const are correct

Discovered while building `stdlib/x509/oid.sio`'s OID byte-constant table.
Distinct from Finding 15 (which is about element COUNT exceeding 16) --
this reproduces at as few as 3 elements, and is specific to element type
`u8`; the identical pattern with `[i64; N]` is unaffected.

```sio
const LOCAL_CONST3D: [u8; 3] = [0x55, 0x04, 0x03]

fn get_first(b: &[u8; 3]) -> u8 {
    b[0]
}

fn main() with IO {
    assert(LOCAL_CONST3D[0] == 0x55)      // CORRECT -- direct indexed read of the const
    let v = get_first(&LOCAL_CONST3D)      // WRONG -- v is neither 0 nor 0x55, some other
                                             // (uninspected) value; the const's bytes are
                                             // corrupted the moment its whole value is used
                                             // as an rvalue (address-taken here; a plain
                                             // `var tmp: [u8;3] = LOCAL_CONST3D` whole-array
                                             // copy reproduces identically)
}
```

Confirmed independently, twice (once by the implementer who found it, once by the
controller with the same minimal repro). Confirmed NOT an import/visibility
issue (reproduces with a private, same-file `const` too). Confirmed specific
to `u8` element type (the identical pattern with `const [i64; N]` works
correctly, passed by reference, no corruption).

**Workaround: never declare a top-level `const [u8; N]` array whose whole
value (not just individual indexed elements) will be used.** Replace it
with a `pub fn` that constructs the same array via a local `var` +
element-by-element assignment and returns it:

```sio
pub fn local_const_3d() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55
    r[1] = 0x04
    r[2] = 0x03
    r
}
```

Call it once into a local `let`/`var` binding at each point of use before
taking a reference to the result -- this is the same shape as this
project's existing Finding-15 workaround (function-returning-a-literal
instead of a bare `const`), but for a different root cause and a much
lower element-count threshold. **Any Sounio module needing a fixed `u8`
byte-array constant whose value (not just individual bytes) will be
copied or passed by reference — OID byte sequences, magic-number byte
prefixes, fixed binary headers — must use this function form, never a bare
`const [u8; N]`, regardless of how few elements it has.**

## Finding 19 — a multi-element array literal returned directly from a function body mistypes as `[i64;N]`, even when the function's declared return type is a narrower element type

Found by Task 3's implementer while building `stdlib/crypto/pkcs1.sio` (RFC 8017
DigestInfo prefix constants). Confirmed independently by the controller with a
minimal repro.

```sio
fn f() -> [u8; 3] {
    [1, 2, 3]     // WRONG -- error[E008]: return value does not match
                  // function's declared return type. The compiler infers
                  // the bare literal as [i64; 3], not [u8; 3], and never
                  // reconciles it against the function's declared return
                  // type.
}
```

**Workaround:** bind the literal to an explicitly-typed `let` first, then
return the binding:

```sio
fn f() -> [u8; 3] {
    let r: [u8; 3] = [1, 2, 3]
    r
}
```

This is the same shape of fix as Finding 18's workaround (type annotation
on an intermediate binding forces the correct element type), but the
trigger here is a `return`-position literal, not a `const` declaration.

## Finding 20 — `rawbuf_set(buf, i, v)` writes one correct byte and zeroes the next 7

Found by Task 3's implementer while building `bigint_to_bytes_be` in
`stdlib/crypto/pkcs1.sio`. Confirmed independently by the controller with a
minimal repro: fill a 20-byte `RawBuf` with sentinel `200` at every
position via a loop, then call `rawbuf_set(&out, 5, 77)` and read back all
20 positions.

Actual: positions 0-4 stayed `200` (untouched); position 5 became `77`
(correct); positions 6-12 (the next 7 bytes) became `0`; positions 13-19
stayed `200` (untouched). This is the write-side counterpart to the
already-documented Finding 3 (`rawbuf_get`/pointer dereference reads a
full 8-byte word, hence the existing `& 255` mask on reads) — the write
path performs a full 8-byte store at the target address instead of a
single-byte store, clobbering the 7 bytes above the target with zero
while leaving the target byte itself correct.

**Workaround:** when filling a buffer byte-by-byte, write in ASCENDING
address order (lowest index first). Under this bug, every write to index
`i` zeroes indices `i+1..i+7`; writing ascending means each later write's
clobber lands only on bytes that haven't been meaningfully written yet
(or that a still-later write will overwrite correctly). Writing in
DESCENDING order (as `bigint_to_bytes_be`'s natural big-endian, most-
significant-limb-last derivation would otherwise do) causes every
earlier, more-significant-byte write to be zeroed by every subsequent
write, corrupting nearly the whole buffer.

**Unresolved residual risk, not yet acted on:** the final write in any
ascending-order loop can still clobber up to 7 bytes PAST THE END of a
tightly-sized buffer. `pkcs1_v15_verify`'s use of
`rawbuf_new(modulus_byte_len)` (e.g. exactly 256 bytes for RSA-2048) adds
no headroom for this. It did not manifest observably in the shipped
RSA-2048 test (the allocator evidently had slack past the requested size)
but is a real, unaddressed out-of-bounds-write risk in `rawbuf_set` itself
— anyone doing a security/safety pass over `stdlib/net/socket.sio` (where
`rawbuf_set` lives) should budget every call site's buffer with 7 bytes of
trailing slack, or fix `rawbuf_set` to perform a true single-byte store.

## Finding 21 (unconfirmed — reported by implementer, not independently reproduced by controller) — claimed cross-module corruption of `bignum::bigint::bigint_modpow`'s top limb on a real RSA-2048 vector

Task 3's implementer reported that calling the imported, unmodified,
already-shipped-and-reviewed `bigint_modpow` (from the BigInt sub-project)
across a module boundary on the real RSA-2048/SHA-256 test vector (see
`tests/run-pass/pkcs1_verify.sio`) returned a `BigInt` whose top limb
(`.limbs[127]`) was `144` instead of the mathematically correct `1`, while
other spot-checked limbs (0, 1, 126) were correct. The implementer's
diagnosis was methodical: an instrumented local copy of the identical
square-and-multiply algorithm, run side-by-side against an independent
Python oracle replicating the same 32-iteration algorithm, matched the
oracle exactly at every iteration including the final one — yet the real
imported `bigint_modpow` on the identical inputs allegedly returned 144
once the call crossed the module boundary.

**Controller verification, three independent runs, all clean:** using the
exact same real 128-limb modulus/signature literals from
`tests/run-pass/pkcs1_verify.sio`, calling the real unmodified
`bignum::bigint::bigint_modpow` directly (bypassing the implementer's
`pkcs1_modpow` workaround entirely) — first with a minimal one-import
scratch file, then again after adding the fuller `hash::sha256` /
`net::socket` import set to match `pkcs1.sio`'s real module-import graph
— consistently returned `em.limbs[127] == 1` (correct), `em.limbs[0] ==
37992`, `em.limbs[126] == 65535`. No corruption reproduced in any of the
three runs.

**Disposition:** this finding does NOT meet this document's "confirmed
independently, twice" bar and is recorded here as unconfirmed rather than
as a verified compiler defect. The controller was unable to identify a
distinguishing trigger condition (import graph was ruled out as a factor).
No further investigation is planned: the implementer's already-shipped
workaround (a private `pkcs1_modpow` in `stdlib/crypto/pkcs1.sio`, a
byte-for-byte copy of `bigint_modpow`'s orchestrating loop, still calling
the imported leaf operations `bigint_mul`/`bigint_mod`/`bigint_from_u32`
cross-module) is harmless and independently verified correct end-to-end
against the real RSA-2048 vector regardless of whether Finding 21's root
cause is real, so no code changes hinge on resolving this discrepancy. If
a future session reproduces top-limb corruption on a cross-module
`bigint_modpow` call, this entry should be upgraded to a confirmed
finding with the actual trigger condition documented.

## Finding 22 — writing a WHOLE struct (literal or local var) into an array-of-struct element cross-contaminates its `[u8;N]` fields once the struct has two or more such fields and the array is large enough

Found by Task 5's implementer while building `x509_parse_name` (RdnEntry
array). Confirmed independently by the controller with three separate
minimal repros, and the threshold behavior cross-checked against an
already-shipped, already-reviewed case that turns out NOT to trigger it.

```sio
struct TwoArrayFields {
    oid: [u8; 20],
    oid_len: i32,
    value: [u8; 128],
    value_len: i32,
}
// ... array of 16 of these, each populated via a local `var entry = ...`
// with entry.oid = oid_buf (first byte 0x55) and entry.value = val_buf
// (first byte 0x54), then written whole:
arr[i] = entry
// or equivalently, via a struct literal directly:
arr[i] = TwoArrayFields { oid: oid_buf, oid_len: 1, value: val_buf, value_len: 7 }

// WRONG in both forms: arr[i].value[0] reads back 0x55 (oid_buf's byte),
// not 0x54 (val_buf's own byte) -- the value field silently receives the
// oid field's array bytes. Scalar fields (oid_len, value_len) are
// unaffected and read back correctly.
```

**This is a size-threshold bug, not universal.** Confirmed SAFE at
`SctEntry`-scale (8-element array, two `[u8;N]` fields of 32+128 bytes,
~188 bytes/entry, ~1.5KB total array — this is the pattern Task 4 already
shipped and reviewed, `stdlib/x509/sct.sio`'s `out[count as usize] =
SctEntry { ... }`; independently re-tested at that exact scale and found
correct, so Task 4's shipped code needs no fix). Confirmed BROKEN at
`RdnEntry`-scale (16-element array, `[u8;20]`+`[u8;128]` fields, ~156
bytes/entry, ~2.5KB total array) and at `GeneralName`-scale (32-element
array, `[u8;253]`+`[u8;20]` fields, ~281 bytes/entry, ~9KB total array).
The exact threshold was not pinned down further (out of scope for this
task's effort budget) — treat any array-of-struct with 2+ `[u8;N]` fields
as at-risk regardless of apparent size unless independently verified safe.

**Workaround: assign every field of an array-of-struct element
individually, directly into the array element** (`arr[i].field = value`
for each field), never via a struct literal and never via an intermediate
local variable copied whole into the array slot. This is a narrower,
corrected scope for what Task 1's original array-of-struct audit (check
(a): "flat array-of-struct, whole-element write + field read via index")
actually covers — check (a)'s fixture used a single-array-field struct;
this finding shows a SECOND `[u8;N]` field in the same struct changes the
failure mode entirely. Any future array-of-struct write involving a
struct with two or more `[u8;N]` fields (`GeneralName`, `ExtensionEntry`,
`RdnEntry` all qualify) must use field-by-field assignment into the array
element as the default, not an exception.

## Finding 23 (implementer-diagnosed, partially independently confirmed) — a large aggregate (order of 10⁵ bytes) returned through a multi-element tuple across nested function calls silently zeroes a non-first scalar field

Found by Task 5's implementer while building `x509_parse_tbs_core`, which
returns `(Certificate, DerReader, i64)` through two levels of nested
function calls. With `Certificate.version: i32` in its originally
committed position (non-first field, between `outer_signature_len` and
`serial_number: BigInt`), the field was correctly set inside the deepest
function (confirmed via an instrumented print immediately before that
function's `return`) but read back as `0` in the top-level caller after
the value crossed two function-return boundaries.

The implementer isolated this with five standalone repro files: a
~600-byte struct with the same field non-first survived the same return
path correctly; only at `Certificate`'s true scale (all its 32/8-element
fixed arrays of nested structs present, roughly 150-200KB) did the
non-first scalar field get silently zeroed, independent of source-level
assignment order or adjacency to the struct's `BigInt` fields. Moving the
field to be the struct's literal first field made it survive reliably.

**Controller verification:** re-running the fix on the real, now-committed
`Certificate` struct (`version` moved to first field) through an
equivalent two-level nested-tuple-return repro confirmed both a non-first
`i64` field (`not_before_unix`) and another (`not_after_unix`) survive
correctly on the real struct at its true scale. A separate ~75KB
synthetic struct with a non-first scalar field, built specifically to
try to reproduce the ORIGINAL (pre-fix) failure, did NOT reproduce
corruption — consistent with a size threshold somewhere between ~75KB and
`Certificate`'s true ~150-200KB scale, but this was not pinned down
further (the fix is already correct and low-cost regardless of the exact
threshold, so further bisection was not pursued).

**Disposition:** treated as a real, load-bearing finding despite the
controller not reproducing the pre-fix failure state directly — the
implementer's isolation was methodical (five repros, ruled out
assignment-order and BigInt-adjacency as confounds) and the controller's
own test independently confirms the FIX is correct at true scale, which
is what matters going forward. **Workaround, now applied project-wide as
a rule for this plan: any large (order of 10^4+ bytes) struct returned
through a multi-element tuple across a function-call boundary should keep
any single scalar field that must survive the round-trip as the struct's
first field**, or (safer, if more than one such field exists) avoid
returning the whole large struct through nested tuple-returning calls at
all — return only the fields that changed and let the caller merge them,
or restructure so the large struct is only ever mutated in place via a
single non-nested function. `Certificate.version` was moved to the first
field as the minimal fix for this task; any future task adding new
scalar fields to `Certificate` that must survive a similar nested-return
path should be aware of this risk.

## Finding 24 — Finding 22's field-by-field-into-array-element workaround does NOT fully fix the underlying defect; it only raises the corruption threshold, and `ExtensionEntry`/`GeneralName`'s true scale is still above it

Found by Task 6's implementer (characterized initially as correlating with
total merged IR function count, ~63-79 threshold) while building
`x509_parse_extensions`/`x509_parse_general_names`. The controller
independently re-investigated the implementer's specific causal claim and
found it does not hold in isolation, but confirmed a real, more precisely
characterized defect underneath it.

**Implementer's claim, independently tested and NOT reproduced in the
form stated:** padding a program with 105 trivial filler functions
(pushing total merged IR functions to 110 — well past the implementer's
reported 63-79 failure range) around an already-known-safe `SctEntry`-
scale array-of-struct write did NOT corrupt it. Same result with the
array write moved into a function carrying 25 sequential 3-tuple
destructures (75 extra live locals) ahead of it, mimicking
`x509_parse_extensions`'s real shape. **Total program function count and
a single function's own local-variable count are not, by themselves, the
trigger.**

**What the controller confirmed IS still broken:** the Finding 22
workaround (assign every field individually, directly into the array
element — `arr[i].field = value`, no intermediate local, no struct
literal) was verified correct in Finding 22's own write-up only at
`RdnEntry`/`GeneralName` scale using the OLD vulnerable whole-struct-copy
pattern as the point of comparison — Finding 22 never independently
re-tested whether the FIXED field-by-field pattern itself holds up at
`ExtensionEntry`'s scale (32-element array, `oid:[u8;20]` +
`value:[u8;512]`, ~532 bytes/entry, ~17KB total array — nearly 7x
`RdnEntry`'s ~2.5KB and nearly 2x `GeneralName`'s ~9KB). It does not:

```sio
struct ExtL { oid: [u8;20], oid_len: i32, critical: bool, value: [u8;512], value_len: i32 }
// ... array of 32, zero-initialized, then:
extensions[count as usize].oid = oid_buf         // oid_buf[0] = 0x55
extensions[count as usize].oid_len = 2
extensions[count as usize].critical = true
extensions[count as usize].value = val_buf       // val_buf[0] = 0xAA
extensions[count as usize].value_len = 512

// WRONG: extensions[0].oid[0] reads back 0xAA (val_buf's byte), not
// 0x55 (its own byte). extensions[0].value[0] reads back correctly
// (0xAA). oid_len/critical/value_len (scalars) all read back correctly.
```

This is the SAME qualitative failure as Finding 20 (`rawbuf_set`'s
write-side 7-byte clobber) and Finding 22 (whole-struct-copy
cross-contamination): a later field write corrupting an earlier field's
already-written bytes, at a struct+array total-byte scale above whatever
this codegen path's actual capacity threshold is — but Finding 22's
prescribed workaround (field-by-field assignment) does NOT raise that
threshold far enough to cover `ExtensionEntry`'s real usage in this plan.
The exact threshold (bytes? field count? distinct from function count,
per the controller's filler-function and tuple-destructure repros both
coming back clean) was not pinned down further — this needs its own
forensic dispatch, per `CLAUDE.md` §8, rather than continued ad hoc
X.509-layer workarounds. `GeneralName` (32-element array, additionally
carrying a nested `X509Name` struct with its own `[RdnEntry;16]`) is at
least as exposed, and the implementer's attempt to work around it via a
flat-parallel-array redesign produced a runtime segfault rather than a
clean fix — see Task 6's report
(`.superpowers/sdd/2026-08-24-madaros-x509-plan/task-6-report.md`) for
the full technique-by-technique trail (9 approaches tried and rejected).

**Status: BLOCKING.** No workaround at the X.509 source-code level was
found that keeps `ExtensionEntry`'s and `GeneralName`'s current struct
shapes (two or more embedded `[u8;N]` fields per array element, at their
real array sizes) intact and correct. This blocks Task 6 and, by
extension, Task 7 (outer Certificate assembly, which must build these
same arrays at real scale plus more). Two paths forward, neither
executable unilaterally within a single SDD task:
1. A compiler fix, via this repo's forensic dispatch protocol
   (`docs/audit/`) — root cause still uncharacterized beyond "large
   struct/array, field-write codegen, not explained by function count or
   local-variable count in isolation."
2. A data-model redesign of `ExtensionEntry`/`GeneralName` (and possibly
   `Certificate` itself) to avoid embedding two or more large `[u8;N]`
   arrays in the same struct-in-array element — e.g., storing `oid`/
   `value` bytes as offset+length pairs into a shared, separately
   allocated `RawBuf` rather than fixed inline arrays. This is a
   revision to Task 2's already-twice-reviewed data model and is out of
   scope for a single task to decide.

## Finding 25 (deferred, not fixed) — a tuple-destructured local (`let (a, b) = f()`) does not propagate a struct-typed element's own type, leaving it exposed to Finding 24's class of corruption

Found while verifying the fix for Finding 24 (commit series starting
`88f91fae6`, culminating in the array-index/field-access struct-type
resolution described below). `let (cert, after_spki, status) =
x509_parse_tbs_core(...)` desugars (per `self-hosted/parser/stmts.sio`'s
`parse_let_stmt`) to `let __tup0 = x509_parse_tbs_core(...)` plus
`let cert = __tup0.0`, `let after_spki = __tup0.1`, `let status =
__tup0.2`. Neither step propagates `cert`'s struct type ("Certificate"):
`__tup0`'s own struct-type binding via the `ExprCall` branch resolves only
a function's *single*-struct return type (`return_struct_name_for_fn_id`),
which is empty for a tuple-typed return; and `let cert = __tup0.0`'s RHS
is an `ExprFieldAccess` with a numeric ("tuple index") field name, which
`lower_let_stmt_ref`'s struct-type-binding `match s.expr` only handles for
Box-tagging, not general struct-type propagation.

**Consequence:** `cert.field` accesses (including the nested
`cert.issuer.entries[0].field` chains the fix below resolves correctly
when `cert`'s own type IS known) fall back to the unresolved-global-lookup
path whenever `cert` came from a tuple-destructured `let`, remaining
exposed to Finding 24's corruption for exactly the same reason arrays and
struct fields were before that fix -- `field_idx_from_name_simple`
collides across structs sharing a field name (most commonly `value`,
against the built-in `Knowledge` struct's `value` at index 0).

**Why this was not fixed alongside Finding 24:** closing it requires new
infrastructure this compiler does not have at all -- per-function tuple
*element* struct-type tracking, analogous to how `elem_type_name_id`/
`named_type_name_id` already track a struct FIELD's array-element or
named type. No existing table records "function X's Nth tuple-typed
return-value element has struct type Y"; building it is a distinct,
larger piece of work than the array-of-struct field-access fix.

**Workaround, applied in `tests/run-pass/x509_parse_tbs_core.sio`:**
declare the struct-typed local FIRST via a plain call whose single-struct
return type IS resolved correctly (e.g. `var cert = certificate_zero()`,
which binds `cert`'s struct type to "Certificate" via the already-working
`ExprCall` path), then overwrite its *value* with the tuple-destructured
result via a plain assignment (`cert = cert_raw`) rather than a `let`/`var`
declaration -- assignment does not touch the struct-type binding table,
only declarations do, so the correct binding survives the overwrite.

**Status: OPEN.** Any future code that tuple-destructures a function
returning `(SomeStruct, ...)` and then does nested field/array access on
the struct element should either apply this same workaround or avoid the
pattern until proper tuple-element struct-type tracking is built.

## Fix landed for Finding 24 (commit series `88f91fae6` and after)

The write-side fix (`88f91fae6`) covered only `arr[i].field` where `arr`
is a bare local. Verifying it surfaced two further gaps in the same bug
class, both fixed in the same session:

1. **Read side**: `let e = arr[i]` (copying a whole struct element out of
   an array) never recorded `e`'s struct type at all -- a pre-existing gap
   in `lower_let_stmt_ref`'s RHS-kind matching (no `ExprIndex` case),
   independent of the write-side bug but exposing the identical
   `field_idx_from_name_simple` collision on every subsequent `e.field`
   read. Confirmed to silently affect the already-merged
   `x509_parse_tbs_core` test (Task 5), which only ever asserted
   `.value`, never `.oid`, so the corruption went undetected until a
   later task's test happened to check the field that was actually wrong.
2. **Chained bases**: both the write-side and read-side fixes initially
   only resolved a bare local array (`extensions[i]`) or a single-level
   struct-field array (`name.entries[i]`) -- not an arbitrarily deep
   field-access chain ending in an array-typed field
   (`cert.issuer.entries[i]`). Generalized via two new recursive helpers,
   `expr_struct_type_ref` (resolves the struct type of an arbitrary
   `ident` / `ident.field.field...` expression) and
   `array_index_base_elem_struct_type` (resolves the array-element struct
   type for whatever expression is being indexed, bare local or chained),
   both in `self-hosted/ir/lower.sio`.

All four decisive repros (bare-local array, both read and write; a
single-level struct-field array, both read and write; the two-level
`cert.issuer.entries[0]` chain) independently re-verified correct after
the final fix. All 6 pre-existing X.509-plan tests still pass, including
`x509_parse_tbs_core.sio` with two new assertions added specifically to
close the `.oid`-was-never-checked gap that let this whole class of bug
ship silently in Task 5.

## Finding 26 (new, not fixed) — writing a struct value that itself contains an array-of-structs field into a DOUBLY array-indexed target corrupts data; distinct from, and not covered by, Finding 24's fix

Found during Task 6 (`x509_parse_general_names`'s `directoryName`
branch), the third attempt at this task, after Findings 24/25 were both
already fixed/documented and independently re-verified working for their
own covered shapes.

**Shape:** `GeneralName.directory_name` is an `X509Name`, which itself
contains `entries: [RdnEntry; 16]` (an array of structs). Task 6's brief
draft assigns a fully-decoded `X509Name` into
`out[count as usize].directory_name`, where `out` is itself
`[GeneralName; 32]` -- i.e. a struct VALUE that contains its own
array-of-structs field, written into a FIELD of an array-of-structs
ELEMENT. This is a strictly different write shape from anything Finding
24's fix (commits `88f91fae6`, `80be7c083`) was verified against: those
fixes cover `arr[i].field = scalar_or_plain_array` (Finding 24's own
fix target) and `cert.issuer.entries[i].field` read/write chains where
the struct reached via the chain has ONLY scalar/plain-array fields, not
a chain whose target ITSELF nests another array-of-structs.

**Three independent minimal repros, all corrupted, all built and run
against the live `bin/souc` (Madaros) on this branch during Task 6's
third attempt (not committed -- discarded scratch files under `/tmp`,
reproducible from the descriptions below):**

1. `gn_list[i].directory_name = name` (whole-struct-VALUE write into an
   array element's STRUCT-typed field, where the struct value being
   written -- `name: X509Name` -- itself contains an array-of-structs
   field). Before the write, `name.entries[0].value` held the correct
   bytes (`D`,`i`,`r`,`C`,`N` = `0x44,0x69,0x72,0x43,0x4E`) and
   `name.entries[0].oid` held `0x55,0x04,0x03`. After
   `gn_list[4].directory_name = name`, reading back
   `gn_list[4].directory_name.entries[0].value[0]` returned `0x55` (oid's
   first byte) and `value[4]` returned `0` -- `value` read back holding
   (a truncated prefix of) `oid`'s bytes.
2. `gn_list[i].directory_name.entries[0].oid = ...; ...
   .entries[0].value = ...` (field-by-field writes through a DOUBLY
   array-indexed chain: outer array index `[i]` -> struct field
   `.directory_name` -> inner array index `.entries[0]` -> field). Also
   corrupted, but in the OPPOSITE direction from (1): after writing
   `oid` first and `value` second, reading back `oid[0]` returned `0x44`
   (`value`'s first byte) -- the LATER field write appears to have
   clobbered the EARLIER one at what the generated code treats as the
   same address, consistent with a stale/reused base-address computation
   for the second (inner) array index in the chain rather than two
   distinct field offsets off one correctly-computed element address.
3. `gn_list[i].directory_name.entries[0] = entry` (a single whole
   `RdnEntry` struct-value write at the doubly-indexed target, `entry`
   built as a local var with correct field values beforehand). Also
   corrupted, identically to (1) -- `value[0]` read back as `0x55`.

All three repros used `x509_name_zero()`/`extension_entry_zero()`-style
zero-constructors and a `[GeneralName; 32]` array matching
`stdlib/x509/cert.sio`'s real shapes exactly (not a simplified stand-in
struct), and `x509_parse_name` was temporarily made `pub` to call it
standalone for repro (1)'s DER-driven variant before hand-building
`name` directly for repros (2)/(3); the temporary visibility change was
reverted before `stdlib/x509/cert.sio` was committed.

**Workaround applied in `stdlib/x509/cert.sio` (Task 6, `git log` commit
implementing this task):** rather than pursue this further -- per this
plan's own dispatch instructions, an unexpected compiler defect is
reported, not silently worked around with speculative retries --
`x509_parse_general_names`'s `directoryName` branch was deleted entirely.
`GENERAL_NAME_DIRECTORY_NAME` now falls through to the same generic
raw-content-bytes `else` branch used for `x400Address`/`ediPartyName`
(a write shape already proven correct by Finding 24's own fix: a single
array index, writing a plain `[u8;253]` field, no nested struct
involved). `GeneralName.directory_name`/`X509Name`'s own struct
definition is UNCHANGED -- only removed from live use in this one call
site -- so no data model rollback was needed, and a future fix to this
defect can restore the richer decode by re-adding the branch this commit
removed.

**Status: OPEN, unfiled as a numbered compiler-team ticket beyond this
audit doc entry.** Any future code assigning a struct value that itself
contains an array-of-structs field into an array-of-structs element's
field (at any nesting depth doubly-indexed or deeper) should assume this
is broken until a decisive fix, mirroring Finding 24's, lands and is
independently re-verified against repro (2) above (the field-by-field
variant, since it is the form most likely to be reached for first as a
"safe" workaround and is NOT safe).

## Finding 27 (application-level, FIXED, NOT a Madaros compiler defect) — an uncapped OID copy loop and an uncapped OID-comparison loop bound, both reachable from an adversarial/malformed certificate

Found during the final whole-plan review of the X.509 sub-project
(2026-08-24), not by any individual task's own reviewer. Unlike Findings
11-26, this is a bug in `stdlib/x509/{cert,oid}.sio` application code, not
in the Madaros compiler itself -- flagged here anyway since it's directly
adjacent to this doc's other X.509-related findings and future readers of
this sub-project's audit trail should see the complete picture in one
place.

**Bug 1 (write side, `stdlib/x509/cert.sio`, `x509_parse_tbs_after_serial`'s
signature-algorithm OID read):** the copy loop filling a fixed `oid_buf:
[u8;20]` from a DER-decoded OID's raw bytes had no upper bound on the loop
index beyond the OID's own (attacker-controlled, for a malformed
certificate) `content_len`:

```sio
var oid_buf: [u8; 20] = [0; 20]
var oi: i64 = 0
while oi < oid_tag.content_len {              // WRONG -- no `&& oi < 20`
    oid_buf[oi as usize] = (rawbuf_get(buf, oid_tag.content_start + oi) & 255) as u8
    oi = oi + 1
}
```

Every OTHER OID-reading loop in this file (four of them, at what were then
lines 529, 895, 1119 before this fix) already had the correct `&& oi < 20`
guard -- this one call site, from Task 5, was the sole exception, missed
by two rounds of per-task review and by the controller's own repeated
interface cross-checks (none of which specifically diffed sibling loops
against each other for a missing guard). A certificate with a
signature-algorithm OID longer than 20 bytes would silently write past
`oid_buf`'s end. **Fixed** by adding the same `&& oi < 20` guard, matching
the file's own established pattern everywhere else.

**Bug 2 (read side, `stdlib/x509/oid.sio`, `pub fn oid_eq`):** the
generic (non-width-specific) OID-equality function loops up to `a_len`
(and, since `a_len == b_len` is checked first, `b_len`) directly, with no
cap against the fixed `[u8;20]` array size:

```sio
pub fn oid_eq(a: &[u8; 20], a_len: i32, b: &[u8; 20], b_len: i32) -> bool {
    if a_len != b_len { return false }
    var i: i32 = 0
    while i < a_len {                          // WRONG -- a_len uncapped
        if a[i as usize] != b[i as usize] { return false }
        i = i + 1
    }
    true
}
```

`a_len`/`b_len` are recorded from a DER OID's own `content_len` field at
every call site in `cert.sio` -- and, critically, `content_len` is NOT
capped to 20 even at call sites where the byte COPY into the `[u8;20]`
buffer IS capped (this file's own established pattern only bounds the
copy, never the recorded length -- confirmed at all four correctly-capped
call sites: the length field always stores the raw, uncapped
`content_len`). `oid_eq` is used by `x509_parse_certificate`'s outer-vs-
inner signature-algorithm cross-check
(`oid_eq(&cert.outer_sig_alg_oid, cert.outer_sig_alg_oid_len, ...)`), so a
malformed certificate with either signature-algorithm OID longer than 20
bytes reaches this function with an oversized length, reading past both
20-byte arrays.

**Confirmed via a minimal probe that Sounio's fixed-size arrays have no
runtime bounds checking** -- an out-of-range index silently reads
adjacent stack memory rather than panicking or clamping (`a[20]` through
`a[24]` on a freshly-declared, zero-initialized `[u8;20]` local read back
`255, 160, 20, 2, 1` -- clearly uninitialized/adjacent stack content, not
zeros, and no crash). This confirms the read is a genuine out-of-bounds
memory access, not a no-op.

**Fixed** by adding `if a_len < 0 || a_len > 20 { return false }` before
the comparison loop in `oid_eq` -- this closes the vulnerability
regardless of which call site an oversized length originates from, rather
than requiring every `_len`-recording site in `cert.sio` to be
individually re-audited and capped. `oid_eq3`/`oid_eq8`/`oid_eq9`/
`oid_eq10` were independently confirmed NOT to share this bug: their
comparison loops use a hardcoded width (3/8/9/10) as the loop bound,
never the caller-supplied length.

**Regression test**: `tests/run-pass/x509_adversarial.sio`'s new case (e)
unit-tests `oid_eq` directly with an oversized length (25 > 20) and
confirms it returns `false` rather than reading out of bounds, plus two
in-bounds sanity checks (exact-match at the full 20-byte boundary, and a
genuine mismatch). Note for future readers: because the underlying defect
is undefined behavior (reading uninitialized/adjacent stack memory, not a
deterministic wrong answer), a before/after comparative test against the
literal fix is not a reliable verification method on its own -- in one
observed compiled build, `oid_eq(&a, 25, &b, 25)` happened to still return
`false` even WITHOUT the fix, because the specific garbage bytes read past
each array's end happened to differ between `a` and `b`. The probe above
(confirming array indexing itself has no bounds checking) is what
establishes the vulnerability is real, independent of whether any single
compiled build's stack layout happens to mask it.

## Finding 28 (application-level, FIXED, NOT a Madaros compiler defect) — `sct_list_decode`'s `ext_len`/`sig_len` were never bounded against the SCT entry's own declared length, allowing an out-of-bounds heap read from a malformed certificate

Found by the final whole-plan reviewer's expanded, Finding-27-class bug
hunt (2026-08-24), independently re-confirmed by the controller.

`sct_list_decode` (`stdlib/x509/sct.sio`) reads a 16-bit, fully
attacker-controlled `ext_len` field and immediately uses it to compute
`after_ext = sct_start + 43 + ext_len`, then reads `hash_alg`/`sig_alg`/
`sig_len` at that position -- with no check that `43 + ext_len + 4` (the
fixed prefix plus the claimed extensions plus the 4 bytes read
immediately after) actually fits within this entry's own declared
`sct_len`. The only prior guard, `pos + sct_len > end` (bounding the
*whole entry* within the caller's buffer), says nothing about whether
`ext_len` itself is consistent with `sct_len`. The `sig_len > 128` check
right after only bounds the *destination* `[u8;128]` array, not whether
the entry's remaining bytes (per `sct_len`) actually contain `sig_len`
more bytes to read from.

**Confirmed via a decisive repro**: a crafted 47-byte SCT entry (the
minimum valid structure size) declaring `ext_len = 100` in a 51-byte
total buffer read at absolute offset 147 -- roughly 92 bytes past the
allocation -- and returned `X509_OK` (not an error), before the fix. With
a maximal `ext_len = 0xFFFF` (65535), the read would land roughly 65KB
past a certificate's own SCT-list allocation (typically a few hundred
bytes), a real out-of-bounds read reachable end-to-end from
`x509_parse_certificate` parsing a malformed or adversarial certificate's
SCT extension.

**Fixed** by adding two explicit bounds checks: after reading `ext_len`,
`43 + ext_len + 4 > sct_len` returns `X509_ERR_MALFORMED`; after reading
`sig_len`, `43 + ext_len + 4 + sig_len > sct_len` also returns
`X509_ERR_MALFORMED`. Regression test:
`tests/run-pass/x509_adversarial.sio` case (f), reproducing the exact
decisive repro above and asserting the corrected `X509_ERR_MALFORMED`/
`count == 0` result.

## Finding 29 (application-level, FIXED, NOT a Madaros compiler defect) — an inverted branch in `unix_timestamp_from_ymdhms` mis-decoded every January/February UTCTime validity date by close to a full year

Found by the final whole-plan reviewer, independently re-confirmed by the
controller both by tracing the arithmetic and by measuring against `date
-u`.

```sio
var mp: i64 = month + 9
if mp >= 12 {
    mp = mp - 12
} else {
    mp = month - 3      // WRONG for month=1/2 -- only reached when
                         // month+9 < 12, i.e. month is 1 or 2
}
```

The intent, per this function's own comment, is `mp = (month+9) % 12`.
For `month=1`: `mp` starts at `10` (already `< 12`, i.e. already its own
mod-12 residue -- no adjustment needed), but the `else` branch
overwrites it with `month - 3 = -2`. For `month=2`: `mp` starts at `11`
(correct), overwritten with `month - 3 = -1`. The `else` branch's formula
is a copy of what the `if` branch computes for a DIFFERENT input range
(`month - 3` is exactly `(month+9) - 12`, the correct reduction for
`month >= 3`) -- it should simply do nothing for `month` 1/2, since
`month+9` is already `< 12` there.

**Measured** (a temporarily-`pub`-exposed copy of the shipped function,
called directly, compared against `date -u -d "<date> 00:00:00" +%s`):

| Input | Expected | Pre-fix actual | Error |
|---|---:|---:|---:|
| 2026-08-24 | 1787529600 | 1787529600 | none (month 8, `if` branch) |
| 2026-01-15 | 1768435200 | 1736812800 | −366 days |
| 2026-02-15 | 1771113600 | 1739404800 | −367 days |
| 2026-03-15 | 1773532800 | 1773532800 | none (month 3, `if` branch) |

All four post-fix values match `date -u` exactly.

**Consequence for the layer's declared purpose**: any certificate whose
`notBefore`/`notAfter` falls in January or February -- roughly one sixth
of all certificates, with no seasonal reason to expect otherwise --
decoded a full year off. A `notAfter` in Jan/Feb read as expired a year
early (fail-closed); a `notBefore` in Jan/Feb read as already valid a
year early (fail-open for a not-yet-valid certificate -- the more
dangerous direction for anything that will eventually gate trust
decisions on this field).

**Why it survived eight per-task reviews and the first attempt at the
final whole-plan review** (which crashed before reaching this function):
the only fixture using January dates anywhere in this plan
(`tests/run-pass/x509_parse_tbs_core.sio`, `250101000000Z`/
`260101000000Z`) asserted only that both dates were nonzero and correctly
ordered -- both properties are invariant under a uniform one-year shift
of both dates, so the bug produced a passing test.

**Fixed** by removing the wrong `else` branch entirely -- `mp` needs no
adjustment when it's already `< 12`. Regression test: added exact-
timestamp assertions to `tests/run-pass/x509_parse_tbs_core.sio`
(`not_before_unix == 1735689600`, `not_after_unix == 1767225600`,
independently computed via `date -u`), so a regression of this exact
shape would now fail that test directly rather than passing it.

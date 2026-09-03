<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-23-madaros-bignum-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-23-madaros-bignum-design
-->

# Madaros Big Integer (BigInt) — Design Spec

## Context and Motivation

This is sub-project 0b-i of the TLS 1.2 roadmap: a fixed-width, multi-limb big-integer type with enough arithmetic (add, subtract, compare, multiply, modular reduction, modular exponentiation) to perform RSA public-key operations — encrypting a premaster secret with a server's public key during a static-RSA key exchange, and verifying a certificate chain's signatures up to a trusted root. Both of these are **public-key-only** operations: the client never holds or exponentiates with a secret/private exponent, only ever with a small, public exponent (conventionally `65537`). This matters architecturally: constant-time execution — the usual hard requirement for RSA/EC private-key math, to avoid leaking secret bits through timing — is **not required here**, since no secret ever drives a data-dependent branch or memory access in this module's scope. (This conclusion, and the reasoning behind it, was already established via a Fable 5 consultation earlier in this project and carries forward unchanged.)

This spec targets Madaros v0.80.0 (the actively-developed Sounio self-hosted compiler on `main`), building directly on empirical findings recorded in `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` — most critically **Finding 11**, discovered while scoping this very module: `u64` right-shift, division, and modulo silently produce mathematically wrong results (via a signed/arithmetic code path) whenever the operand's bit 63 is set, while addition, subtraction, multiplication, bitwise AND/OR/XOR, left-shift, and equality are all confirmed bit-exact correct regardless of bit 63. Two other findings from the same doc are directly relevant: **Finding 1**, which rules out `i256`/`i512` as a foundation (their multiply is verified correct only for one narrow compile-time-immediate fixture, not general runtime arithmetic — silently wrong outside that fixture, the same failure shape documented for `i128` too); and **Finding 2**, which rules out local stack arrays as raw-pointer-borrowable buffers (irrelevant to this module specifically, since `BigInt` never needs to hand a pointer to a syscall — it's pure computation — but worth remembering if this module ever needs to serialize to/from a wire-format byte buffer for TLS record framing, which is out of this spec's scope).

## Non-Goals

- Arbitrary-precision (unbounded) integers — this is a **fixed maximum width** type (sized for RSA-4096), not a general bignum library.
- Private-key RSA operations (signing, decryption) — out of scope, since this client only ever does public-key operations. If a future project needs private-key math, constant-time execution becomes a hard requirement and this module's algorithms (especially modular exponentiation) would need to be redesigned, not just extended.
- Elliptic-curve arithmetic — a completely different mathematical structure; explicitly deferred (this was flagged as a much larger, separate lift back when TLS cipher-suite scope was chosen — static RSA key exchange was picked specifically to avoid needing ECC for the MVP).
- Optimized multiplication (Karatsuba, Montgomery reduction, etc.) — schoolbook (O(n²)) algorithms only, correctness-first. Revisit only if RSA verification proves too slow in practice, which is unlikely to matter for a client-side, once-per-connection operation.
- Any wire-format (byte-buffer) conversion — this module operates purely on the `BigInt` type; converting to/from the raw bytes of an RSA modulus/ciphertext read off the network is a separate, later concern (sub-project 0b-ii or later, when X.509 parsing is designed).

## Architecture

One new module: `stdlib/bignum/bigint.sio`. Import convention, per `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`'s Findings 6/8: a caller outside `stdlib/` uses `use bignum::bigint::*`; if a future sibling file is added inside `stdlib/bignum/`, it uses the bare `use bigint::*` form.

### The core design decision: 16-bit limbs, not 32-bit

Per Finding 11, a schoolbook multiply using 32-bit limbs would need to extract the high 32 bits of a 32×32 product via `>> 32` or `/ 4294967296` on a `u64` — exactly the operation Finding 11 proves is broken whenever that product's bit 63 is set (which happens for a large fraction of real 32-bit×32-bit products). **This module uses 16-bit limbs instead**: every partial product, carry, and running sum in the arithmetic below stays bounded well under `2^40` — nowhere near bit 63 — so every shift, mask, and carry-extraction operation needed is provably safe under Finding 11's own confirmed-correct operation list (`+`, `-`, `*`, `&`, `|`, `^`, `<<`, `==`/`!=` are unconditionally safe; this design additionally uses `>>`/`/` only on values independently bounded away from bit 63, which is safe per Finding 11's own boundary confirmation — a value with bit 62 clear right-shifts correctly).

### Types

```sio
pub const BIGINT_MAX_LIMBS: i32 = 256   // 256 * 16 bits = 4096 bits, covers RSA-2048 and RSA-4096

pub struct BigInt {
    limbs: [u16; 256],   // little-endian: limbs[0] is the least significant 16 bits
    len: i32,             // number of significant limbs (trailing zero limbs are not counted)
}
```

`len` is maintained as an invariant by every operation below: after any operation, `len` is normalized to the smallest value such that `limbs[len..]` are all zero (i.e., no operation leaves spurious trailing zero limbs counted in `len`), except that a `BigInt` representing zero has `len == 0` (or `len == 1` with `limbs[0] == 0` — pick one convention and apply it consistently everywhere; the plan's implementation task will need to pick and document this explicitly, since it affects every comparison/loop bound in this spec).

### Operations

**Construction**: `bigint_zero() -> BigInt`, `bigint_from_u32(v: u32) -> BigInt` (splits `v` into at most 2 limbs — safe, since `v` is a 32-bit input value, not a value that could have bit 63 set).

**Comparison**: `bigint_cmp(a: &BigInt, b: &BigInt) -> i32` (returns -1/0/1) — compares `len` first (a `BigInt` with more significant limbs is larger, given the normalization invariant), then compares limbs from most significant down to least. Every comparison here is between plain `i32`/`u16` values, never between raw `u64` values that could have bit 63 set, so Finding 11's ordering-comparison bug (only confirmed broken for `u64`) does not apply.

**Addition**: `bigint_add(a: &BigInt, b: &BigInt) -> BigInt` — limb-by-limb: `sum: u32 = (a.limbs[i] as u32) + (b.limbs[i] as u32) + carry` (max value `2*65535 + 1 = 131071`, far under `2^32`, let alone bit 63); `result.limbs[i] = (sum & 0xFFFF) as u16`; `carry = sum >> 16` (safe: `sum` is a `u32` far below any bit-63 concern, and even if intermediates are computed as `u64`, `131071` is nowhere near `2^63`).

**Subtraction**: `bigint_sub(a: &BigInt, b: &BigInt) -> BigInt` (assumes `a >= b`, per `bigint_cmp` — the caller must check this; RSA's usage of subtraction, inside modular reduction below, is always structured to satisfy this) — limb-by-limb with borrow, symmetric to addition.

**Multiplication**: `bigint_mul(a: &BigInt, b: &BigInt) -> BigInt` — schoolbook O(n²): for each pair `(i, j)`, compute `partial: u32 = (a.limbs[i] as u32) * (b.limbs[j] as u32)` (max `65535 * 65535 = 4294836225`, which fits in `u32` — just under `2^32` — and is trivially far under bit 63 whether stored as `u32` or widened to `u64`), accumulate into a working array of `u64` totals at position `i+j` (an accumulator array of `BIGINT_MAX_LIMBS * 2` positions, each of which can receive up to `BIGINT_MAX_LIMBS` partial-product contributions in the worst case — `256 * 4294836225 ≈ 2^40`, still enormously far below bit 63), then do a single final carry-propagation pass over the accumulator array converting it into normalized 16-bit output limbs (each carry-extraction step there operates on a value bounded by the same `~2^40` ceiling, safe per Finding 11).

**Modular reduction**: `bigint_mod(a: &BigInt, n: &BigInt) -> BigInt` — schoolbook binary long division: process `a`'s bits from most significant to least significant, maintaining a running remainder that is left-shifted by 1 and OR'd with the next input bit at each step, subtracting `n` from the remainder whenever the remainder is `>= n` (via `bigint_cmp`/`bigint_sub`). This is `O(bits^2)` but correctness-first is the stated priority for this sub-project, and RSA-4096's ~4096-bit modulus makes this a bounded, small-in-absolute-terms cost for a once-per-TLS-handshake operation.

**Modular exponentiation**: `bigint_modpow(base: &BigInt, exponent: &BigInt, modulus: &BigInt) -> BigInt` — square-and-multiply: iterate over `exponent`'s bits from most significant to least significant, squaring an accumulator and reducing modulo `modulus` at every step, and additionally multiplying by `base` (then reducing) whenever the current exponent bit is 1. Since `exponent` is always a small public value in this module's intended use (`65537 = 0x10001`, 17 bits, exactly two set bits), this loop runs at most ~17 times regardless of how large `base`/`modulus` are (up to 4096 bits) — cheap in practice despite the schoolbook-everything approach.

## Testing Strategy

Real, known-correct test vectors — not just "the code ran without error." Options, in order of preference:
1. Small, hand-computable cases first (e.g. `123 * 456 == 56088`, verifiable by inspection) to catch gross errors in limb-splitting/carry logic cheaply.
2. A real, published RSA test vector (e.g. from an RFC or a well-known reference implementation's test suite) for the full `modpow` path: a known modulus, a known public exponent (`65537`), a known plaintext, and its known correct ciphertext — `bigint_modpow(plaintext, 65537, modulus) == known_ciphertext`. This is the test that actually proves the module is fit for its stated purpose (RSA public-key operations), not just "arithmetic seems to work."
3. A large-value multiplication test specifically exercising the bit-63 boundary this whole design is built around: multiply two values whose product's bit 63 would be set if computed via 32-bit limbs (e.g. anything above `2^31` on both operands), and verify the result against a value computed independently (e.g. by hand, or cross-checked against a script run outside Sounio during test-writing — not committed as executable test code, just used to derive the expected constant).

All tests as `tests/run-pass/*.sio`, following this branch's established `//@ run-pass` convention, invoked via `bash scripts/run_sio_test_suite.sh --filter-prefix bignum_` (or whatever prefix the implementation plan settles on for this module's test files).

## Open Implementation-Time Questions

- The exact `len == 0` vs. `len == 1, limbs[0] == 0` convention for representing zero needs to be picked and applied with zero exceptions across every operation — a future implementer must audit every loop bound and comparison against whichever convention is chosen, not assume it "obviously" works either way.
- Whether Madaros's tuple-destructuring/linear-type quirks (Findings 4/5/9) are relevant here at all — `BigInt` is a plain (non-linear) struct with no resource-ownership semantics, so none of the `linear`-specific workarounds from the socket/HTTP work should apply, but this should be confirmed rather than assumed once real code is written, per this whole project's established discipline of verifying compiler behavior empirically rather than by analogy.
- Whether accumulator arrays for the multiplication step (`BIGINT_MAX_LIMBS * 2` `u64` entries) should be a local stack array or something else — per Finding 2, local arrays are boxed GC handles with no raw-pointer-borrowability, but this module never needs to hand such an array to a syscall (it's pure computation, ordinary indexed read/write only), so a plain local `[u64; 512]` array should be fine; confirm this doesn't hit any unrelated array-size or stack-depth limit empirically during implementation.

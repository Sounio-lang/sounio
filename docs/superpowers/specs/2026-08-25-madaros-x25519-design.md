<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-25-madaros-x25519-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-25-madaros-x25519-design
-->

# Madaros X25519 Key Exchange Design

## Overview

**Goal:** implement X25519 (RFC 7748) — the elliptic-curve Diffie-Hellman key
exchange TLS 1.2/1.3 negotiate in practice against modern servers — in pure
Sounio, plus a genuine OS-entropy source for private-key generation.

**Why:** this is the second of several sub-projects needed to build a real
TLS handshake on Madaros. The AEAD ciphers sub-project (already merged, PR
#2133) covers symmetric encryption once a handshake completes; this
sub-project covers deriving the shared secret that protects that handshake
in the first place. A later sub-project (not yet scoped) needs HKDF/TLS-PRF
key derivation and the handshake state machine itself, both of which will
call this sub-project's `x25519`/`x25519_base_point_mul` as a black box.

**Tech stack:** pure Sounio, no non-Sounio code anywhere in this
sub-project's deliverables.

## Global Constraints

- **Never use native `u32` arithmetic anywhere near a 32-bit-word boundary.**
  `stdlib/hash/word32.sio` already exists with `add32`/`rotl32`/`xor32`/etc.,
  all operating on `i64` values masked to `0..4294967295` — a documented,
  real Madaros compiler defect (native `u32` `+`/`-` do not wrap mod 2^32 at
  all) makes the native type unsafe here. This sub-project's field
  arithmetic works in limbs well under 32 bits (see Architecture), so this
  constraint mostly matters for any byte-packing/unpacking code that
  manipulates 32-bit-sized quantities directly.
- **Never copy a fixed-array value via `var x = *ref`.** A real,
  previously-undocumented Madaros compiler defect, found and formally filed
  during the AEAD ciphers sub-project (`docs/handoff/souc_v0800_defects.md`
  §D8): dereferencing a fixed-array reference (`&[T;N]`, any element type —
  confirmed not `u8`-specific) ALIASES the caller's array instead of
  copying it, and a type annotation on the `var` binding does not suppress
  this. Plain array-to-array assignment without a dereference (`var b = a`,
  `let c = a`, where `a` is already a local array value) copies correctly.
  This sub-project's field arithmetic does extensive limb-array
  manipulation — every copy of a limb array must be explicit
  element-by-element assignment, never `var x = *source`.
- **The Montgomery ladder must be constant-time**, including its
  conditional swap (`cswap`). The scalar being multiplied is the private
  key — a data-dependent branch or memory-access pattern keyed on scalar
  bits is a real side-channel vulnerability, not a style preference.
  `cswap(swap: bool, a: FieldElement, b: FieldElement)` must compute both
  "swapped" and "unswapped" results unconditionally and select between them
  via a bitmask, never an `if`.
- **RFC 7748's low-order-point / all-zero-output rejection is mandatory,
  not optional.** `x25519` (the general two-argument form, used with a
  peer's public key) must check its own output for all-zero bytes and
  report failure — silently returning an all-zero "shared secret" for a
  malicious low-order input point is a known, real attack this check
  exists to close.
- **Private-key entropy must come from a genuine OS source, never
  `stdlib/crypto/random.sio`'s existing `CryptoRng`.** That module is a
  deterministic, seeded xoshiro256** generator — despite its header
  comment currently claiming "CSPRNG", it is NOT cryptographically secure:
  an attacker observing enough output can recover the internal state and
  predict all past/future outputs, and it is seeded from a single `i64`,
  not real entropy. This sub-project reads `/dev/urandom` directly via the
  existing `read_file` builtin (`read_file(path: string) -> string with
  IO`, confirmed working on this compiler during the X.509 chain-validation
  sub-project) instead. **As a small, targeted, security-relevant fix
  bundled into this sub-project**: `stdlib/crypto/random.sio`'s header
  comment ("CSPRNG (xoshiro256** based)") is corrected to state plainly
  that it is a deterministic, non-cryptographic PRNG suitable for
  reproducible testing/simulation only — this is a one-line documentation
  fix to an existing file, not a functional change to it.
- Error convention: sentinel `bool`/`i64` — never `Result<T,E>`/`Option<T>`.
- RawBuf-based interfaces (`net::socket::*`) where a caller-facing function
  needs variable-length output, matching the AEAD ciphers sub-project's own
  convention — though most of this sub-project's actual values are the
  fixed 32-byte X25519 key/point size, so fixed `[u8;32]` arrays are used
  wherever the size is always exactly 32 bytes by the protocol's own
  definition, reserving `RawBuf` for the OS-entropy-reading function (whose
  output length is caller-specified).
- Every test's expected value comes from a published, independent source
  (RFC 7748) — never self-generated, per this project's standing
  discipline, doubly important for cryptographic correctness where a
  self-consistent bug can pass a self-generated test. (This exact class of
  bug — a self-consistent error invisible to the mandated test vectors —
  was found for real in the AEAD ciphers sub-project's Poly1305
  implementation; the same discipline of independently re-verifying vectors
  against a primary source, and of not treating "the mandated vectors pass"
  as complete proof of correctness, applies here.)
- Module imports: bare `use <filename>::{name|*}`, no `module` declaration.
- No AI-attribution line in commits; Conventional Commits style
  (`feat(crypto): ...`).

---

## Architecture

Two new modules under `stdlib/crypto/`, plus a one-line documentation fix
to an existing one:

| Module | Responsibility |
|---|---|
| `csprng.sio` | `crypto_os_random_bytes(n: i64) -> (RawBuf, bool)` — reads `n` bytes from `/dev/urandom` via `read_file`. The ONLY entropy source this sub-project (or any future TLS code built on it) should use for private-key material. |
| `x25519.sio` | Field arithmetic over GF(2^255-19), scalar clamping, the constant-time Montgomery ladder, and the two public entry points: `x25519_base_point_mul` and `x25519`. |
| `random.sio` (existing, one-line fix) | Header comment corrected from "CSPRNG" to explicitly state it is a deterministic, non-cryptographic PRNG — no functional change. |

**Field arithmetic representation**: radix-2^25.5, 10 limbs (alternating
26-bit and 25-bit limbs, the standard representation for a 255-bit field
element — this is the same conservative, small-limb discipline already
proven correct in this codebase for Poly1305's 130-bit field: a product of
two ~26-bit limbs is ~52 bits, and summing several such products stays
comfortably under `i64`'s 63-bit danger zone (the same "Finding 11"
native-shift/divide/modulo defect this codebase's `bigint.sio` and
`word32.sio` already document and design around) without needing a wider
native integer type. A 5-limb radix-2^51 representation (closer to modern
optimized C implementations) is explicitly rejected here: intermediate
products of two ~51-bit limbs approach 2^102, requiring genuine
double-word arithmetic this compiler has no native support for, and this
sub-project's priority is provable correctness on a compiler with known
sharp edges around wide-integer arithmetic, not raw speed.

`x25519.sio` is the only module the future handshake sub-project calls
directly (via `x25519_base_point_mul`/`x25519`); `csprng.sio` is called
directly by whichever code generates a private key (likely the handshake
sub-project itself, when it assembles a `key_share` — this sub-project just
provides the primitive).

---

## Data Structures

### `csprng.sio`

```sio
pub fn crypto_os_random_bytes(n: i64) -> (RawBuf, bool) with IO
```

### `x25519.sio`

```sio
pub const X25519_KEY_LEN: i64 = 32

// Internal 10-limb representation of a field element in GF(2^255-19).
// Limb widths alternate 26/25 bits (limbs 0,2,4,6,8 hold up to 26 bits;
// limbs 1,3,5,7,9 hold up to 25 bits) -- the standard curve25519
// representation. Not exposed outside this file; callers only ever see
// 32-byte encoded field elements.
struct FieldElement {
    limbs: [i64; 10],
}

// RFC 7748 section 5: clamps a raw 32-byte scalar per the X25519 clamping
// rule (clear bits 0-2 of byte 0, clear bit 7 of byte 31, set bit 6 of
// byte 31).
pub fn x25519_clamp(raw: &[u8;32]) -> [u8;32]

// The base-point multiplication: derives a public key from a clamped
// private scalar, using the fixed base point u=9. Always succeeds (the
// base point never produces a degenerate/all-zero result).
pub fn x25519_base_point_mul(scalar: &[u8;32]) -> [u8;32] with IO

// The general two-argument form: computes the shared secret from a
// clamped private scalar and a peer's public key (u-coordinate). Returns
// (shared_secret, ok) -- ok is false if the result is all-zero (RFC 7748's
// mandatory low-order-point rejection), in which case shared_secret's
// bytes are the computed (but rejected) all-zero value and MUST NOT be
// used as key material by the caller.
pub fn x25519(scalar: &[u8;32], u_coordinate: &[u8;32]) -> ([u8;32], bool) with IO
```

---

## Data Flow

1. **Key generation** (typical caller sequence, not itself one function in
   this sub-project — the handshake sub-project composes these):
   `(entropy, ok) = crypto_os_random_bytes(32)` →
   `scalar = x25519_clamp(&entropy_as_32_byte_array)` →
   `public_key = x25519_base_point_mul(&scalar)`.
2. **Shared secret derivation**: `(shared_secret, ok) =
   x25519(&scalar, &peer_public_key)` — caller must check `ok` before using
   `shared_secret`.
3. **Inside `x25519_base_point_mul`/`x25519`** (the shared core):
   - `fe_from_bytes`: decode the 32-byte `u_coordinate` (or the constant
     `9` for base-point multiplication) into a `FieldElement`'s 10-limb
     representation.
   - Montgomery ladder: iterate the scalar's bits from bit 254 down to bit
     0 (bit 255 is always 0 after clamping, per RFC 7748). At each step,
     `cswap` conditionally swaps two running point pairs based on the
     current scalar bit XORed with the previous swap state (RFC 7748's
     exact ladder structure — implement from the RFC's own pseudocode, not
     from memory of a specific reference implementation, and cross-check
     against the RFC 7748 test vectors below), then applies the standard
     Montgomery differential-addition-and-doubling step using `fe_add`,
     `fe_sub`, `fe_mul`, `fe_sqr`.
   - `fe_invert`: modular inverse via Fermat's little theorem
     (`a^(p-2) mod p`, computed by repeated squaring/multiplying — no
     extended-Euclidean-algorithm branch-on-value code, which would
     reintroduce a scalar-dependent-timing risk since the exponent here is
     the fixed public constant `p-2`, not secret, but the multiplication
     chain itself should still just be a fixed sequence of squarings and
     multiplies, not a data-dependent loop).
   - `fe_to_bytes`: encode the final `FieldElement` back to 32 bytes,
     including the final conditional reduction mod `p` (a field element's
     internal limb representation can transiently represent a value
     `>= p`; the encoding step must correctly reduce it into `[0, p)`
     before serializing, per RFC 7748's own encoding procedure).
   - `x25519` additionally checks the encoded output for all-zero bytes
     before returning `ok = true`.

---

## Error Handling

- `crypto_os_random_bytes`: `(RawBuf, bool)`, `ok = false` if `/dev/urandom`
  can't be read (rare, but possible in a restricted environment) — the
  returned `RawBuf` on failure is unspecified content and must not be used.
- `x25519_base_point_mul`: no failure mode (always succeeds).
- `x25519`: `([u8;32], bool)`, `ok = false` on an all-zero result — the
  contract is explicit that the caller MUST check `ok`, matching this
  project's existing AEAD `_open` functions' fail-closed convention (though
  the failure MODE here is different: X25519 isn't authenticating anything,
  it's rejecting a mathematically-degenerate key-exchange input).
- Sentinel `bool` throughout — never `Result`/`Option`.

---

## Testing Strategy

Every test's expected value from RFC 7748, independently re-verified
against the actual RFC text before being trusted in committed test code
(not from a secondary citation or from memory alone):

- **RFC 7748 §5.2**: the two official X25519 test vectors (each a
  32-byte scalar, 32-byte input u-coordinate, and 32-byte expected output
  u-coordinate) — the most-cited X25519 test vectors in existence, still
  verified independently rather than trusted by reputation.
- **RFC 7748 §6.1**: the full Alice/Bob key-exchange worked example —
  Alice's private scalar → Alice's public key (via
  `x25519_base_point_mul`), Bob's private scalar → Bob's public key, then
  `x25519(alice_private, bob_public) == x25519(bob_private, alice_public)
  == RFC's published shared secret`. This is the test that proves the
  actual key-exchange property (commutativity), not just that one
  multiplication produces a correct-looking number — analogous to how the
  AEAD ciphers sub-project's round-trip tests proved seal/open symmetry
  beyond what a single-direction vector alone could.
- **`x25519_clamp`**: a direct unit test asserting the exact bits RFC 7748
  specifies are cleared/set, on a few different input byte patterns
  (including an input that's already partially "clamped" and one that
  isn't, to make sure the function doesn't accidentally depend on the
  input's prior state).
- **Adversarial**: a known low-order point (the all-zero 32-byte
  u-coordinate is RFC 7748's own simplest documented low-order case) fed
  into `x25519` — assert `ok == false`, not a silently-accepted all-zero
  "shared secret".
- **Given this session's own recent discovery that a self-consistent bug
  can survive every mandated vector** (the AEAD ciphers sub-project's
  Poly1305 carry-propagation bug, found only via a directed exhaustive
  search, not by the RFC-mandated vectors): after the mandated RFC 7748
  vectors pass, run an ADDITIONAL cross-check — compute a handful of
  extra scalar/u-coordinate pairs (not from any RFC, arbitrary well-formed
  inputs) and verify them against an independently-computed reference
  (e.g., a from-scratch Python big-integer computation of the same
  algorithm, written separately from the Sounio implementation, matching
  the level of extra scrutiny the AEAD final review applied). This is not
  a substitute for the RFC vectors — it is insurance against the specific,
  now-demonstrated failure mode where a limb-arithmetic bug is invisible
  to a small number of fixed test cases.

## Out of Scope (explicit)

- P-256 or any other curve — X25519 only, per this branch's earlier
  decomposition decision.
- Ed25519 signatures (a related but functionally distinct algorithm over
  the same curve family) — not needed for TLS key exchange.
- HKDF or any TLS key-schedule logic — this sub-project produces a raw
  shared secret; deriving TLS traffic keys from it is the next sub-project's
  job.
- Session-key caching, key reuse policies, or any TLS-protocol-level
  concern (e.g. HelloRetryRequest's second key_share) — this sub-project is
  a pure cryptographic primitive with no protocol awareness.
- Hardware acceleration / constant-time-via-CPU-intrinsics — this is a
  pure-software implementation; "constant-time" here means "no
  secret-dependent branches or memory-access patterns in the Sounio source
  and its compiled output", not a formally verified guarantee against every
  possible microarchitectural side channel.

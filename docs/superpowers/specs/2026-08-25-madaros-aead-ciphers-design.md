<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-25-madaros-aead-ciphers-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-25-madaros-aead-ciphers-design
-->

# Madaros AEAD Ciphers Design

## Overview

**Goal:** implement AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305 — the
three AEAD cipher suites TLS 1.2/1.3 negotiate in practice against modern
servers — behind one unified interface, in pure Sounio, with no dependency
on any existing crypto code beyond this stdlib's own primitives.

**Why:** this is the first of several sub-projects needed to build a real
TLS handshake on Madaros (the second-to-last piece before HTTPS works),
which unblocks a web-search tool being built for the Conclave chat app.
The X.509 chain-validation sub-project (already merged, PR #2133) covers
certificate trust; this sub-project covers the symmetric encryption that
protects application data once a handshake completes. A later sub-project
(not yet scoped) will need X25519 key exchange, HKDF/PRF key derivation,
and the handshake state machine itself (TLS 1.3 primary, TLS 1.2 as a
negotiated fallback) — all of which will call this sub-project's
`aead_seal`/`aead_open` as a black box, selected by whichever cipher suite
the handshake negotiates.

**Tech stack:** pure Sounio, no non-Sounio code anywhere in this
sub-project's deliverables.

## Global Constraints

- **RawBuf-based, not fixed-array-based.** This codebase has two
  pre-existing, inconsistent SHA-256 implementations: `stdlib/hash/sha256.sio`
  (`sha256(buf: &RawBuf, len: i64) -> [u8;32] with IO`, the convention
  `stdlib/x509/*` and `stdlib/crypto/pkcs1.sio` already use) and
  `stdlib/crypto/sha256.sio` (fixed `[u8;256]` arrays + an explicit
  `SHA256State` struct, an older convention). TLS records carry up to
  2^14 = 16384 bytes of plaintext per record (TLS 1.3, RFC 8446 §5.1) —
  far past a 256-byte fixed array — so this sub-project follows the
  `RawBuf` convention throughout, deliberately diverging from
  `crypto/sha256.sio`'s older style. This sub-project does not touch or
  depend on either existing SHA-256 implementation (GCM uses GHASH,
  ChaCha20-Poly1305 uses Poly1305 — neither AEAD construction needs a
  general-purpose hash function).
- **GCM only ever runs AES in the encrypt direction.** Counter-mode
  encryption and decryption are the same operation (XOR with a keystream
  derived from `AES_encrypt`), so `aes.sio` implements ONLY
  `SubBytes`/`ShiftRows`/`MixColumns`/`AddRoundKey` and their forward key
  schedule — never `InvSubBytes`/`InvShiftRows`/`InvMixColumns`. This is a
  deliberate scope reduction, not an oversight: AES-ECB/CBC decryption is
  explicitly out of scope, since this sub-project exists only to support
  GCM.
- **12-byte nonces only.** TLS's record-layer nonce construction always
  produces a 12-byte (96-bit) nonce for both AES-GCM and
  ChaCha20-Poly1305 (RFC 8446 §5.3, RFC 7905). GCM's more general
  arbitrary-length-nonce handling (which requires an extra GHASH pass when
  the nonce isn't exactly 12 bytes) is explicitly out of scope.
- **Nonce uniqueness and management are the caller's responsibility.**
  This sub-project does not track, validate, or reject nonce reuse — TLS's
  own record-sequence-number-based nonce construction (the future
  handshake sub-project's job) is what guarantees uniqueness. Documented
  as an explicit, deliberate boundary, not a gap.
- **Never use native `u32` arithmetic.** `stdlib/hash/word32.sio`'s own header comment documents "Finding 13" (`docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`): native `u32` `+`/`-` do not wrap mod 2^32 on Madaros at all — a more severe failure than the already-known Finding 11 (`u64` shift/divide/modulo breaking when bit 63 is set). Every 32-bit word this sub-project touches — ChaCha20's state words and counter, AES's key-schedule word rotation/XOR, GCM's counter-block low-32-bit increment — must be represented as a plain `i64` masked to `0..4294967295`, using `stdlib/hash/word32.sio`'s existing `add32`/`rotl32`/`xor32`/`and32`/`or32`/`not32` primitives (already used by this stdlib's SHA-1/SHA-256, already tested) rather than reimplementing 32-bit arithmetic from scratch or reaching for the native `u32` type anywhere in this sub-project's arithmetic. `rotr32` is NOT needed by anything in this spec (ChaCha20 and AES both only ever rotate left) but exists in the same module if a future need arises; note its documented precondition (`1 <= n <= 31`) if ever used. `chacha20_block`'s `counter` parameter is therefore typed `i64` (word32-bounded), not `u32` — see Data Structures below.
- **Constant-time tag comparison.** `aead_open`'s tag-verification step
  must not branch or short-circuit per-byte on tag mismatch — accumulate a
  bitwise difference across all 16 tag bytes and compare the accumulator
  to zero once, at the end. This is a security requirement, not a style
  preference: a data-dependent early-exit leaks how many leading tag bytes
  matched via timing.
- **Fail-closed, zeroed-on-failure.** `aead_open` returns
  `(plaintext: RawBuf, ok: bool)`. When `ok == false`, the function
  overwrites its own `plaintext` buffer with zeros before returning — a
  caller that ignores the `ok` flag and reads `plaintext` anyway gets all-zero
  bytes, never partially-decrypted attacker-influenced plaintext.
- Error convention: sentinel `bool`/`i64` as established elsewhere in this
  stdlib — never `Result<T,E>`/`Option<T>`.
- Every test's expected output is a published, independent test vector
  (NIST FIPS-197 for raw AES blocks, NIST CAVP GCM test vectors for full
  AES-GCM, RFC 8439 §2.3.2/§2.5.2/§2.8.2 for ChaCha20/Poly1305/ChaCha20-Poly1305)
  — never a self-generated "expected" value, per this project's standing
  "measure, don't assume" discipline, doubly important for cryptographic
  correctness where a self-consistent bug can pass a self-generated test.
- Module imports: bare `use <filename>::{name|*}`, no `module` declaration.
- No AI-attribution line in commits; Conventional Commits style
  (`feat(crypto): ...`).

---

## Architecture

Five new modules under `stdlib/crypto/`:

| Module | Responsibility |
|---|---|
| `aes.sio` | AES-128 and AES-256 key schedule (key expansion into round keys) and single-16-byte-block encryption. Encryption direction only (see Global Constraints). |
| `gcm.sio` | Galois/Counter Mode: counter-mode encryption/decryption via `aes.sio`, GHASH (GF(2^128) multiplication) for authentication, tag computation and verification. Parameterized by which AES variant's round keys it's given — no AES-128-vs-256 branching of its own beyond how many rounds `aes.sio`'s block-encrypt function runs. |
| `chacha20.sio` | The ChaCha20 stream cipher: quarter-round ARX primitive, the 20-round block function, counter-based keystream generation for arbitrary-length plaintext. |
| `poly1305.sio` | The Poly1305 one-time MAC: arithmetic modulo 2^130 - 5, single-key streaming authenticator. |
| `aead.sio` | The ONLY module the future handshake sub-project calls. `aead_seal`/`aead_open`, dispatching to `gcm.sio` (AES-128 or AES-256 round keys) or `chacha20.sio`+`poly1305.sio` per the RFC 8439 §2.8 AEAD construction, selected by a `pub const AEAD_*` algorithm ID. |

`aes.sio`, `gcm.sio`, `chacha20.sio`, and `poly1305.sio` are internal
collaborators — still file-level `pub` (Sounio has no narrower
visibility), but not part of this sub-project's advertised public
interface. Only `aead.sio`'s `aead_seal`/`aead_open` are the interface a
consumer should call.

---

## Data Structures

### `aes.sio`

```sio
pub const AES_128_ROUNDS: i64 = 10
pub const AES_256_ROUNDS: i64 = 14

pub struct AesRoundKeys {
    // 15 round keys of 16 bytes each covers both variants (AES-128 uses
    // the first 11 of these 15 slots; AES-256 uses all 15) -- a single
    // fixed-size shape avoids a second, parallel struct for the 256-bit
    // variant.
    keys: [[u8; 16]; 15],
    rounds: i64,   // AES_128_ROUNDS or AES_256_ROUNDS -- tells aes_encrypt_block how many of `keys` to use
}

pub fn aes128_key_schedule(key: &[u8; 16]) -> AesRoundKeys
pub fn aes256_key_schedule(key: &[u8; 32]) -> AesRoundKeys

// Encrypts exactly one 16-byte block in place; the only AES operation
// this sub-project implements (see Global Constraints).
pub fn aes_encrypt_block(round_keys: &AesRoundKeys, block: &[u8; 16]) -> [u8; 16]
```

### `gcm.sio`

```sio
pub const GCM_TAG_LEN: i64 = 16
pub const GCM_NONCE_LEN: i64 = 12

// plaintext_len may be 0 (AAD-only, "authenticate but don't encrypt" is a
// legal GCM use, though this sub-project's own callers -- TLS record
// protection -- always have nonzero plaintext).
pub fn gcm_seal(
    round_keys: &AesRoundKeys,
    nonce: &[u8; 12],
    plaintext: &RawBuf,
    plaintext_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, i64) with IO   // (ciphertext_with_tag, total_len == plaintext_len + GCM_TAG_LEN)

pub fn gcm_open(
    round_keys: &AesRoundKeys,
    nonce: &[u8; 12],
    ciphertext_with_tag: &RawBuf,
    ciphertext_with_tag_len: i64,   // must be >= GCM_TAG_LEN
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, bool) with IO   // (plaintext, ok) -- plaintext is plaintext_len == ciphertext_with_tag_len - GCM_TAG_LEN bytes, zeroed when ok == false
```

### `chacha20.sio`

```sio
pub const CHACHA20_KEY_LEN: i64 = 32
pub const CHACHA20_NONCE_LEN: i64 = 12

// Generates exactly one 64-byte keystream block for the given counter.
// gcm.sio has no equivalent -- AES's block size (16 bytes) differs from
// ChaCha20's (64 bytes), so `aead.sio` composes each cipher's own natural
// block size rather than forcing a shared block-size abstraction.
pub fn chacha20_block(key: &[u8; 32], counter: i64, nonce: &[u8; 12]) -> [u8; 64]

// XORs `data` with the keystream starting at `counter`, in place semantics
// (returns a new RawBuf rather than mutating -- matches this stdlib's
// existing value-semantics style for RawBuf-producing functions).
pub fn chacha20_encrypt(key: &[u8; 32], counter: i64, nonce: &[u8; 12], data: &RawBuf, data_len: i64) -> RawBuf with IO
```

### `poly1305.sio`

```sio
pub const POLY1305_KEY_LEN: i64 = 32
pub const POLY1305_TAG_LEN: i64 = 16

pub fn poly1305_mac(key: &[u8; 32], msg: &RawBuf, msg_len: i64) -> [u8; 16] with IO
```

### `aead.sio`

```sio
pub const AEAD_AES_128_GCM: i32 = 0
pub const AEAD_AES_256_GCM: i32 = 1
pub const AEAD_CHACHA20_POLY1305: i32 = 2

// `key` is interpreted per `algo`: 16 bytes used for AEAD_AES_128_GCM,
// 32 bytes used for AEAD_AES_256_GCM or AEAD_CHACHA20_POLY1305 -- callers
// pass a `&[u8;32]` always and this module reads only the first 16 bytes
// when `algo == AEAD_AES_128_GCM`, avoiding a second key-typed parameter
// shape per algorithm.
pub fn aead_seal(
    algo: i32,
    key: &[u8; 32],
    nonce: &[u8; 12],
    plaintext: &RawBuf,
    plaintext_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, i64) with IO   // (ciphertext_with_tag, total_len)

pub fn aead_open(
    algo: i32,
    key: &[u8; 32],
    nonce: &[u8; 12],
    ciphertext_with_tag: &RawBuf,
    ciphertext_with_tag_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, bool) with IO   // (plaintext, ok)
```

---

## Data Flow

**AES-GCM** (`gcm_seal`/`gcm_open`, called from `aead_seal`/`aead_open` when `algo` is `AEAD_AES_128_GCM`/`AEAD_AES_256_GCM`):
1. `H = aes_encrypt_block(round_keys, [0u8; 16])` — the GHASH subkey, derived once per call (TLS derives a fresh key per direction per epoch, so this isn't cached across calls; caching is a future performance concern, not this sub-project's).
2. `J0 = nonce ++ [0,0,0,1]` (the standard 12-byte-nonce GCM counter-block construction, RFC 5116/NIST SP 800-38D §7.1 case 1).
3. Encrypt `plaintext` in CTR mode: block `i` (1-indexed from `J0+1`) is `plaintext[i] XOR aes_encrypt_block(round_keys, J0 + i)`, incrementing only the low 32 bits of the counter block (per spec — GCM's counter increment is a 32-bit wraparound, not a full 128-bit increment).
4. GHASH over `aad ++ pad(aad) ++ ciphertext ++ pad(ciphertext) ++ len(aad) ++ len(ciphertext)` (both lengths as 64-bit big-endian bit-counts, per NIST SP 800-38D §6.4) using `H`.
5. `tag = GHASH(...) XOR aes_encrypt_block(round_keys, J0)`.
6. `gcm_open` repeats steps 1-4 against the received ciphertext, computes the expected tag the same way, and compares it against the received tag using the constant-time comparison from Global Constraints before decrypting-and-returning (or zeroing-and-returning `false`).

**ChaCha20-Poly1305** (`aead_seal`/`aead_open` when `algo == AEAD_CHACHA20_POLY1305`, following RFC 8439 §2.8 exactly):
1. `poly1305_key = chacha20_block(key, counter=0, nonce)[0..32]` (the first 32 bytes of the counter-0 keystream block; the remaining 32 bytes of that block are discarded, per spec).
2. `ciphertext = chacha20_encrypt(key, counter=1, nonce, plaintext, plaintext_len)` (encryption starts at counter 1 — counter 0 is reserved for deriving the Poly1305 key).
3. `mac_data = aad ++ pad16(aad) ++ ciphertext ++ pad16(ciphertext) ++ len(aad) as u64-LE ++ len(ciphertext) as u64-LE` (RFC 8439 §2.8's exact padding/length-encoding construction — note this is little-endian length encoding, the opposite of GCM's big-endian, and `pad16` pads to the next 16-byte boundary, not GCM's own padding rule — these two constructions are NOT unified in this sub-project's code, since conflating them to "share" logic would risk exactly the kind of subtle cross-contamination bug this project's own X.509 work has repeatedly hit).
4. `tag = poly1305_mac(poly1305_key, mac_data, mac_data_len)`.
5. `aead_open` recomputes the same `poly1305_key` and `mac_data` against the received ciphertext, computes the expected tag, and constant-time-compares before decrypting-and-returning.

`aead.sio` itself contains no cryptographic logic — only the `algo`-based dispatch to one of the two flows above, and the key-slicing note from the `aead_seal`/`aead_open` struct definitions above.

---

## Error Handling

- `aead_seal`/`gcm_seal`/`chacha20_encrypt`/`poly1305_mac` have no failure mode given well-formed fixed-size inputs (`nonce`/`key` are always the right fixed-array size by the type system itself) — they always produce output.
- `aead_open`/`gcm_open` return `(RawBuf, bool)`. `ok == false` on tag mismatch, and on `ciphertext_with_tag_len < GCM_TAG_LEN` (a malformed/truncated input, not a panic-worthy condition — TLS records can be truncated by a network-level attacker, and this must fail cleanly, not crash).
- Constant-time tag comparison (see Global Constraints) applies to every `_open` function in this sub-project, not just `gcm_open`.

---

## Testing Strategy

Every test's expected value comes from a published, independent source — never self-generated:

- **`aes.sio`**: NIST FIPS-197 Appendix B/C's known-answer single-block encryption vectors, for both AES-128 and AES-256 key sizes.
- **`gcm.sio`**: NIST CAVP's published GCM test vectors (`key`, `IV`/nonce, `PT`, `AAD`, expected `CT` and `Tag`) for both AES-128-GCM and AES-256-GCM — at least one all-zero-key/nonce/plaintext case (the simplest, most commonly published GCM test case) and one case with real AAD and multi-block plaintext.
- **`chacha20.sio`**: RFC 8439 §2.3.2's block-function test vector (the well-known "Sunscreen" test key/nonce/counter with its published 64-byte keystream block).
- **`poly1305.sio`**: RFC 8439 §2.5.2's standalone MAC test vector.
- **`aead.sio`** (ChaCha20-Poly1305 path): RFC 8439 §2.8.2's full worked AEAD example (the complete key/nonce/AAD/plaintext/ciphertext/tag).
- **`aead.sio`** (AES-GCM path): re-run at least one of `gcm.sio`'s own NIST vectors through `aead_seal`/`aead_open` with the correct `algo` constant, confirming the dispatch layer doesn't corrupt anything `gcm.sio` itself got right.
- **Adversarial** (`aead_open`, both algorithms): a single flipped tag byte → `ok == false`; a single flipped AAD byte → `ok == false`; a truncated ciphertext (`len < GCM_TAG_LEN`/`POLY1305_TAG_LEN`) → `ok == false`, not a panic or out-of-bounds read.
- **Round-trip** (both algorithms): `aead_open(aead_seal(msg)) == msg` for a handful of plaintext lengths, including a length that isn't a multiple of the underlying cipher's block/keystream-chunk size (AES's 16 bytes, ChaCha20's 64 bytes) — GCM/ChaCha20 are both stream-like constructions over a block cipher, so an off-block-boundary length is a real edge case worth locking in, not implied by the exact-block-length NIST/RFC vectors alone.

## Out of Scope (explicit)

- AES decryption (`InvSubBytes`/`InvShiftRows`/`InvMixColumns`) — GCM never needs it.
- GCM with a non-12-byte nonce.
- Nonce generation, uniqueness tracking, or reuse detection — caller's responsibility (the future handshake sub-project).
- AES-192 (not a TLS 1.2/1.3 cipher suite key size — 128 and 256 are the only sizes TLS ever negotiates).
- Any other AEAD construction (AES-CCM, AES-SIV, XChaCha20-Poly1305) — not TLS 1.2/1.3 cipher suites in practice against modern servers.
- Hardware acceleration / AES-NI-equivalent intrinsics — Sounio has no such intrinsics exposed; this is a pure-software implementation, and performance optimization is explicitly deferred.

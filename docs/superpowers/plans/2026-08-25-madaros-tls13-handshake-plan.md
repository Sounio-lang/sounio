<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-25-madaros-tls13-handshake-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-25-madaros-tls13-handshake-plan
-->

# Madaros TLS 1.3 Handshake + Record Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build a working TLS 1.3 client for Madaros — two new signature-verification primitives (RSA-PSS, ECDSA P-256), an X.509 extension to parse EC public keys, and a record layer / transcript hash / handshake codec / connection orchestrator — that completes a real handshake against an unmodified modern server and exchanges encrypted HTTP/1.1 traffic.

**Architecture:** eight tasks in dependency order. Two standalone signature primitives (no network, no state) → one small X.509 extension (parses EC public keys, a real gap the handshake needs) → three standalone protocol-layer modules (record framing, transcript hashing, message codec — all testable against RFC 8448's published wire bytes with no live network) → the connection orchestrator, split into two tasks (handshake-to-application-keys, then the application-data lifecycle) since it's the only piece needing a live TLS peer and ties every earlier piece together.

**Tech Stack:** Sounio/Madaros (`./bin/souc`), no non-Sounio code in shipped deliverables.

**Spec:** `docs/superpowers/specs/2026-08-25-madaros-tls13-handshake-design.md`

## Global Constraints

- **RawBuf-based** throughout (`net::socket::*`) except where a fixed-size type is already established by a consumed interface (e.g. `aead_seal`'s `&[u8;32]` key, `x25519`'s `[u8;32]`).
- **Sentinel bool/i64 error convention** — never `Result<T,E>`/`Option<T>`.
- **Compiler defect §D8** (`docs/handoff/souc_v0800_defects.md`): `var x = *ref` on any fixed-array reference (`&[T;N]`) ALIASES the caller's array instead of copying it, even with a type annotation. Copy fixed arrays element-by-element in every task.
- **`TlsConnection` by-value threading**: `tls_connect`/`tls_send`/`tls_recv` take and return `TlsConnection` by value (`(TlsConnection, i64)`), mirroring `stdlib/net/socket.sio`'s `TcpSocket` convention exactly — never `&conn`/`&!conn`.
- Bare `use <filename>::{name|*}` imports.
- No AI-attribution line in any commit. Conventional Commits style (`feat(crypto): ...`, `feat(tls): ...`).
- Every test's expected value from a published, independently-verified source (RFC/NIST CAVP text, fetched fresh where needed) — never self-generated.
- Test invocation: `bash scripts/run_sio_test_suite.sh --filter-prefix <x>_ --jobs 2` for this plan's own tests — never the whole-repo suite as a checkpoint. `--jobs 2` specifically (higher parallelism produces spurious 30-second timeouts on this test runner right after a rebuild).
- Run every shell command as a plain foreground command — never a background/Monitor-style wait mechanism.
- **Sandbox networking, confirmed empirically before this plan was written**: `openssl` is available at `/usr/bin/openssl`. Loopback TCP (127.0.0.1) works. Outbound internet egress does NOT work in this environment (a direct connection attempt to 1.1.1.1:443 timed out). Task 8's live-interop testing must use a local `openssl s_server` on `127.0.0.1` — do not attempt or assume access to any public internet host.
- **Already-confirmed real interfaces this plan consumes** (read the cited file yourself before writing code against it — these are given so you don't have to re-derive them, not so you skip verifying they still match):
  - `stdlib/net/socket.sio`: `RawBuf`/`rawbuf_new`/`rawbuf_set`/`rawbuf_get`; `TcpSocket`; `tcp_connect(ip: &RawBuf, port: u16) -> (TcpSocket, i64) with IO`; `tcp_send(sock: TcpSocket, buf: &RawBuf, len: i64) -> (TcpSocket, i64) with IO`; `tcp_recv(sock: TcpSocket, buf: &RawBuf, cap: i64) -> (TcpSocket, i64) with IO`; `tcp_close(sock: TcpSocket) with IO`.
  - `stdlib/crypto/aead.sio`: `AEAD_AES_128_GCM: i32 = 0`, `AEAD_AES_256_GCM: i32 = 1`, `AEAD_CHACHA20_POLY1305: i32 = 2`; `aead_seal(algo: i32, key: &[u8;32], nonce: &[u8;12], plaintext: &RawBuf, plaintext_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, i64) with IO`; `aead_open(algo: i32, key: &[u8;32], nonce: &[u8;12], ciphertext_with_tag: &RawBuf, ciphertext_with_tag_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, bool) with IO`. AES-128's 16-byte key is passed padded into the 32-byte slot — `aead_seal`/`aead_open` slice the first 16 bytes internally for `AEAD_AES_128_GCM`; callers always pass a 32-byte buffer regardless of the negotiated cipher's real key length. **`aead_seal` has no failure channel and treats any unrecognized `algo` as ChaCha20-Poly1305 undefined-behavior — callers MUST validate `algo` against the three `AEAD_*` constants before calling; this is a documented caller contract, not a bug to route around.**
  - `stdlib/crypto/x25519.sio`: `X25519_KEY_LEN: i64 = 32`; `x25519_clamp(raw: &[u8;32]) -> [u8;32]`; `x25519_base_point_mul(scalar: &[u8;32]) -> [u8;32] with IO`; `x25519(scalar: &[u8;32], u_coordinate: &[u8;32]) -> ([u8;32], bool) with IO`.
  - `stdlib/crypto/csprng.sio`: `crypto_os_random_bytes(n: i64) -> (RawBuf, bool) with IO`; `crypto_os_random_key32() -> ([u8;32], bool) with IO`.
  - `stdlib/crypto/hkdf.sio`: `HKDF_HASH_SHA256: i32 = 0`, `HKDF_HASH_SHA384: i32 = 1`; `hmac_sha256_rawbuf`/`hmac_sha384_rawbuf`; `hkdf_extract`; `hkdf_expand`; `hkdf_expand_label(hash_algo: i32, secret: &RawBuf, secret_len: i64, label: &RawBuf, label_len: i64, context: &RawBuf, context_len: i64, out_len: i64) -> RawBuf with IO`; `derive_secret(hash_algo: i32, secret: &RawBuf, secret_len: i64, label: &RawBuf, label_len: i64, transcript_hash: &RawBuf, transcript_hash_len: i64) -> RawBuf with IO`; `tls13_early_secret(hash_algo: i32) -> RawBuf with IO`; `tls13_handshake_secret(hash_algo: i32, early_secret: &RawBuf, early_secret_len: i64, ecdhe_shared_secret: &RawBuf, ecdhe_len: i64) -> RawBuf with IO`; `tls13_master_secret(hash_algo: i32, handshake_secret: &RawBuf, handshake_secret_len: i64) -> RawBuf with IO`; `tls13_traffic_key_and_iv(hash_algo: i32, traffic_secret: &RawBuf, traffic_secret_len: i64, key_len: i64, iv_len: i64) -> (RawBuf, RawBuf) with IO`. `hkdf_expand_label`'s undocumented-checked precondition (label_len<=249, context_len<=255) is irrelevant here — every TLS 1.3 label/context this plan uses is far under both.
  - `stdlib/bignum/bigint.sio`: `pub struct BigInt { limbs: [u16; 512], len: i32 }`; `bigint_zero()`, `bigint_from_u32(v: u32) -> BigInt`, `bigint_cmp(a: &BigInt, b: &BigInt) -> i32`, `bigint_add`, `bigint_sub`, `bigint_mul`, `bigint_mod(a: &BigInt, n: &BigInt) -> BigInt`, `bigint_modpow(base: &BigInt, exponent: &BigInt, modulus: &BigInt) -> BigInt` — this last one is the RSA/EC modular-exponentiation primitive to reuse; it is already `pub`, do not reimplement it or copy `pkcs1.sio`'s private `pkcs1_modpow` wrapper.
  - `stdlib/crypto/pkcs1.sio`: has `pub fn pkcs1_v15_verify(signature: &BigInt, modulus: &BigInt, exponent: &BigInt, modulus_byte_len: i64, hash_algorithm: i32, message_hash: &[u8; 64], message_hash_len: i64) -> bool with IO` and several **file-private** helpers (`bigint_to_bytes_be`, `pkcs1_modpow`, `digest_info_*`) that Task 1 must NOT import (they aren't `pub`) — Task 1 writes its own EMSA-PSS encoding logic in `rsa_pss.sio`, calling only `bigint_modpow` (public, in `bigint.sio`) for the modular exponentiation itself.
  - `stdlib/asn1/der.sio`: `pub struct DerReader`, `pub struct DerTag`, `pub fn der_read_integer(r: &DerReader, tag: &DerTag) -> (BigInt, i64) with IO` — reuse this for decoding ECDSA's DER-encoded `r`/`s` INTEGERs in Task 2, rather than writing a new INTEGER decoder.
  - `stdlib/hash/sha256.sio`: `pub fn sha256(buf: &RawBuf, len: i64) -> [u8; 32] with IO`. `stdlib/hash/sha384.sio`: `pub fn sha384(buf: &RawBuf, len: i64) -> [u8; 48] with IO`.
  - `stdlib/x509/chain.sio`: `pub fn x509_verify_chain(leaf: &Certificate, leaf_buf: &RawBuf, intermediates: &[Certificate; 8], intermediate_bufs: &[RawBuf; 8], intermediate_count: i32, trust_store: &TrustStore, hostname: &RawBuf, hostname_len: i64, now_unix: i64, ocsp_response: &RawBuf, ocsp_response_len: i64) -> i64 with IO, Mut`.
  - `stdlib/x509/cert.sio`: `pub struct Certificate { ..., modulus: BigInt, public_exponent: BigInt, ... }` — **currently has NO fields for an EC public key at all**, and neither `stdlib/x509/oid.sio` nor `cert.sio` recognizes the `id-ecPublicKey` OID (`1.2.840.10045.2.1`) or the `prime256v1`/`secp256r1` curve OID (`1.2.840.10045.3.1.7`) anywhere — confirmed by grep, zero hits. This is a real, load-bearing gap: a certificate presenting an ECDSA public key has nowhere to be stored today. Task 3 below closes this gap; it is NOT optional for ECDSA CertificateVerify to work at all.

---

## Task 1: `stdlib/crypto/rsa_pss.sio` — RSASSA-PSS signature verification

**Files:**
- Create: `stdlib/crypto/rsa_pss.sio`
- Test: `tests/run-pass/rsa_pss_verify_vectors.sio`

**Interfaces:**
- Consumes: `BigInt`/`bigint_modpow`/`bigint_cmp` (`stdlib/bignum/bigint.sio`), `sha256`/`sha384` (`stdlib/hash/{sha256,sha384}.sio`), `RawBuf`/`rawbuf_new`/`rawbuf_set`/`rawbuf_get` (`stdlib/net/socket.sio`).
- Produces: `pub const RSA_PSS_HASH_SHA256: i32 = 0`, `pub const RSA_PSS_HASH_SHA384: i32 = 1`, `pub fn rsa_pss_verify(modulus: &BigInt, exponent: &BigInt, modulus_byte_len: i64, message: &RawBuf, message_len: i64, signature: &BigInt, hash_algo: i32) -> bool with IO`. Task 8 imports and calls this exact name/signature with the negotiated cipher's hash algorithm and the leaf certificate's `modulus`/`public_exponent` (from `stdlib/x509/cert.sio`'s `Certificate`).

RFC 8017 (PKCS#1 v2.2) EMSA-PSS-VERIFY (§9.1.2), with the salt length equal to the hash's output length (the universal TLS 1.3 convention for `rsa_pss_rsae_*`, per RFC 8446 §4.2.3 — MGF1 is always instantiated with the SAME hash as the signature's own hash, and `sLen = hLen`):

1. `RSAVP1`: `m = signature^exponent mod modulus` via `bigint_modpow`. If `bigint_cmp(signature, modulus) >= 0`, fail immediately (invalid signature representative).
2. Convert `m` to a big-endian byte string `EM` of length `modulus_byte_len` (emLen). If `m`'s encoded length exceeds `emLen`, fail. Write element-by-element into a `RawBuf` — do NOT reuse `pkcs1.sio`'s private `bigint_to_bytes_be` (not importable); write your own small big-endian byte-extraction loop over `BigInt.limbs` (16-bit limbs, little-endian in the struct) in this file.
3. `emBits = 8*modulus_byte_len - 1` (RSA-PSS treats the encoded message as `emBits`-bit, not full-byte — the top bit of the first byte of `EM` must be part of a masking step, RFC 8017 §9.1.2 step 6: after computing `maskedDB`, the leftmost `8*emLen - emBits` bits of the leftmost octet of `maskedDB` MUST be cleared to zero; since `modulus_byte_len` is always a whole number of bytes for a standard RSA modulus, `8*emLen - emBits = 1` — clear the top bit only. Verify this exact reasoning against RFC 8017's own text rather than assuming, since an off-by-one here silently accepts/rejects the wrong signatures.)
4. `EM` must end in `0xBC` (the trailer field). If not, fail.
5. Split `EM` (minus the trailing `0xBC` byte) into `maskedDB` (the first `emLen - hLen - 1` bytes) and `H` (the next `hLen` bytes), where `hLen` is `hash_algo`'s output length (32 for SHA-256, 48 for SHA-384).
6. Compute `dbMask = MGF1(H, emLen - hLen - 1, hash_algo)` (MGF1 per RFC 8017 §B.2.1: repeatedly hash `H || counter` where `counter` is a 4-byte big-endian `i64` value 0,1,2,... via `sha256`/`sha384`, concatenating outputs, truncated to the requested length).
7. `DB = maskedDB XOR dbMask`, then clear `DB`'s first bit (matching step 3's masking).
8. `DB` must be `emLen - hLen - hLen - 2` zero bytes, then a single `0x01` byte, then the `hLen`-byte salt. If the leading bytes aren't all zero or the `0x01` marker is missing, fail.
9. `mHash = Hash(message)` via `sha256`/`sha384`.
10. `M' = 8 zero bytes || mHash || salt` (the salt extracted in step 8).
11. `H' = Hash(M')`. Compare `H' == H` (the value extracted in step 5) byte-for-byte, constant-time (accumulate a bitwise difference across all `hLen` bytes, compare once at the end — matching this codebase's established AEAD tag-comparison discipline, never a short-circuiting per-byte comparison). Return true iff equal.

- [ ] **Step 1: Independently fetch and re-verify a real RSA-PSS test vector before writing anything**

Fetch NIST's FIPS 186-4 RSA `SigVerPSS` test vector file (or an equivalent independently-published RSA-PSS verification test vector with a known-good public key, message, and signature — search for "NIST CAVP SigVerPSS" or use RFC 8017 Appendix A/C's own worked example if it publishes one with SHA-256). Do not trust any hex string transcribed into this plan (none is given, deliberately — this plan does not pre-transcribe crypto test vectors for exactly the reason every prior sub-project on this branch has independently re-derived them: secondhand transcription is the single most common source of self-consistent-but-wrong tests on this project). Record the exact source URL/document you used in your report.

- [ ] **Step 2: Write the failing test using the vector you fetched**

```sio
//@ run-pass
// RSA-PSS-SHA256 verification against a real, independently-sourced
// test vector (see task report for exact source). Fill in modulus/
// exponent/message/signature from that source, not from this template.
use crypto::rsa_pss::*
use bignum::bigint::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in: build `modulus`/`exponent` BigInts and `signature` BigInt
    // from the fetched vector's real hex values, build `message` as a
    // RawBuf from the vector's real message bytes>
    let ok = rsa_pss_verify(&modulus, &exponent, modulus_byte_len, &message, message_len, &signature, RSA_PSS_HASH_SHA256)
    if ok == false {
        println("rsa_pss_verify_vectors: valid signature rejected")
        return 1
    }

    // Negative case: flip one byte of the signature, must be rejected.
    var bad_signature = signature
    // <flip one limb/byte of bad_signature>
    let bad_ok = rsa_pss_verify(&modulus, &exponent, modulus_byte_len, &message, message_len, &bad_signature, RSA_PSS_HASH_SHA256)
    if bad_ok == true {
        println("rsa_pss_verify_vectors: corrupted signature accepted")
        return 1
    }

    println("rsa_pss_verify_vectors: valid signature accepted, corrupted signature rejected")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/rsa_pss_verify_vectors.sio`
Expected: unresolved-import error referencing `crypto::rsa_pss`.

- [ ] **Step 4: Implement `rsa_pss_verify` in `stdlib/crypto/rsa_pss.sio`** following the 11-step algorithm above. Write MGF1 as its own file-private helper (`fn mgf1(seed: &RawBuf, seed_len: i64, mask_len: i64, hash_algo: i32) -> RawBuf with IO`), since it's needed nowhere else in this file but must be exactly right — RFC 8017 §B.2.1's counter is a 4-byte big-endian encoding of an integer that only ever needs to go up to a few hundred (never near `2^32`), but encode all 4 bytes regardless, per the spec.

- [ ] **Step 5: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/rsa_pss_verify_vectors.sio`
Expected: prints the success message, exit 0.

- [ ] **Step 6: Add a second real test vector using SHA-384** (`RSA_PSS_HASH_SHA384`), independently fetched the same way as Step 1 — do not reuse the SHA-256 vector's key material with a fabricated SHA-384 signature.

- [ ] **Step 7: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix rsa_pss_ --jobs 2 --verbose`
Expected: all passed, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add stdlib/crypto/rsa_pss.sio tests/run-pass/rsa_pss_verify_vectors.sio
git commit -m "feat(crypto): add RSASSA-PSS signature verification"
```

---

## Task 2: `stdlib/crypto/ecdsa_p256.sio` — NIST P-256 point arithmetic + ECDSA verification

**Files:**
- Create: `stdlib/crypto/ecdsa_p256.sio`
- Test: `tests/run-pass/ecdsa_p256_verify_vectors.sio`, `tests/run-pass/ecdsa_p256_point_arithmetic.sio`

**Interfaces:**
- Consumes: `BigInt`/`bigint_modpow`/`bigint_cmp`/`bigint_add`/`bigint_sub`/`bigint_mul`/`bigint_mod` (`stdlib/bignum/bigint.sio`), `sha256` (`stdlib/hash/sha256.sio`), `der_read_integer`/`DerReader`/`DerTag` (`stdlib/asn1/der.sio`), `RawBuf` (`stdlib/net/socket.sio`).
- Produces: `pub struct EcPoint { x: BigInt, y: BigInt }` (affine coordinates), `pub fn ecdsa_p256_verify(pubkey: &EcPoint, message: &RawBuf, message_len: i64, der_signature: &RawBuf, der_signature_len: i64) -> bool with IO`. Task 3 produces the `EcPoint` this consumes from a parsed certificate's `SubjectPublicKeyInfo`; Task 8 calls `ecdsa_p256_verify` directly.

**Curve parameters (NIST P-256 / secp256r1, FIPS 186-4 D.1.2.3) — verify every one of these against a primary source (e.g. SEC 2: Recommended Elliptic Curve Domain Parameters, or FIPS 186-4 itself) before hardcoding them, do not trust this list as pre-verified:**
- `p = 2^256 - 2^224 + 2^192 + 2^96 - 1` (the field prime)
- `a = p - 3`, `b = 5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b` (curve equation `y^2 = x^3 + a*x + b mod p`)
- `n` (the group order, a different, smaller prime than `p`) and the base point `G = (Gx, Gy)`.

**Limb-representation decision (make this explicitly, in a code comment, matching the discipline `stdlib/crypto/x25519.sio` and `stdlib/crypto/poly1305.sio` already used):** `stdlib/bignum/bigint.sio`'s existing `BigInt` (512×16-bit limbs, 8192-bit capacity) is generic big-integer arithmetic with no P-256-specific fast reduction — reuse it directly for this task rather than building a P-256-specific radix representation like X25519's. Justification to write in the file's header comment: P-256 field elements are only 256 bits, far under `BigInt`'s 8192-bit capacity and its documented `BIGINT_MUL_SAFE_OPERAND_LIMBS: i32 = 256`-limb (4096-bit) safe-multiplication ceiling, so there is no overflow-margin risk multiplying two P-256-sized values the way there was designing X25519's own limb radix from scratch. Reduction mod `p` uses plain `bigint_mod` (no P-256-specific fast-reduction trick) — slower per-operation than a specialized implementation, but this function runs once per handshake (verifying one CertificateVerify signature), not in a hot loop, so raw `BigInt` operations are the right complexity/effort trade-off here. If you find a mismatch between this reasoning and reality (e.g. `bigint_mul`'s actual safe ceiling doesn't cover a P-256 multiplication the way this paragraph assumes), stop and report rather than proceeding on a false premise.

**Point arithmetic (affine coordinates; this function runs rarely enough that Jacobian-coordinate optimization is not worth the complexity):**
- `pub fn ec_point_add(a: &EcPoint, b: &EcPoint, p: &BigInt) -> EcPoint` — standard affine point addition (`lambda = (by - ay) / (bx - ax) mod p` via modular inverse; see below).
- `pub fn ec_point_double(a: &EcPoint, curve_a: &BigInt, p: &BigInt) -> EcPoint` — standard affine point doubling (`lambda = (3*ax^2 + curve_a) / (2*ay) mod p`).
- `pub fn ec_scalar_mul(k: &BigInt, point: &EcPoint, curve_a: &BigInt, p: &BigInt) -> EcPoint` — double-and-add scalar multiplication, iterating `k`'s bits from `BigInt.len*16 - 1` down to 0 (skip leading zero limbs first), using `ec_point_double` every iteration and `ec_point_add` only when the current bit is 1. This does not need to be constant-time (verification of a signature over a PUBLIC key and PUBLIC message has no secret-dependent timing to protect — unlike X25519's scalar multiplication over a PRIVATE key, which is why THAT function is constant-time and this one does not need to match it).
- **Modular inverse mod `p`**: `bigint_mod`/`bigint_modpow` give you what you need via Fermat's little theorem (`p` is prime): `a^-1 mod p = a^(p-2) mod p` via `bigint_modpow(a, p-2, p)`. Implement `fn ec_mod_inverse(a: &BigInt, p: &BigInt) -> BigInt` this way — do not implement the extended Euclidean algorithm from scratch when `bigint_modpow` already gives you the same result via one call.
- **Modular inverse mod `n`** (needed for `s^-1 mod n` in verification): same technique, `s^-1 mod n = s^(n-2) mod n` via `bigint_modpow(s, n-2, n)`.

**Verification (FIPS 186-4 §6.4.2 / SEC1 §4.1.4), given `(r, s)` DER-decoded via `der_read_integer` and the message hash `e = sha256(message)` (interpreted as a big-endian integer, truncated to the leftmost `min(hash_bit_length, curve_order_bit_length)` bits — for P-256 with SHA-256 both are 256 bits, so no truncation is actually needed; confirm this reasoning rather than skip the truncation logic silently):**
1. Verify `1 <= r < n` and `1 <= s < n`; fail otherwise.
2. `w = s^-1 mod n`.
3. `u1 = (e * w) mod n`, `u2 = (r * w) mod n`.
4. `point = ec_scalar_mul(u1, G, curve_a, p)` `ec_point_add` `ec_scalar_mul(u2, pubkey, curve_a, p)`. If the result is the point at infinity, fail (this needs an explicit representable "point at infinity" — add `pub fn ec_is_infinity(pt: &EcPoint) -> bool` using a sentinel, e.g. both coordinates equal to `bigint_zero()`, since `(0,0)` is never a valid affine point on this curve given `b != 0`).
5. `v = point.x mod n`. Signature is valid iff `v == r`.

- [ ] **Step 1: Independently fetch and re-verify curve parameters and a real ECDSA P-256 test vector before writing anything**

Fetch NIST's FIPS 186-4 ECDSA `SigVer` test vector file for P-256/SHA-256 (search "NIST CAVP ECDSA SigVer P-256"), or SEC 2's published domain parameters plus a separately-published known-good signature (e.g. from a real, verifiable source — do not hand-construct a signature yourself, since a self-signed test vector proves nothing about correctness). Record your exact source in the report.

- [ ] **Step 2: Write the failing curve-arithmetic sanity test first**

```sio
//@ run-pass
// Sanity check: G doubled, then added to G three more times, equals 4*G
// via ec_scalar_mul -- an internal consistency check on the point
// arithmetic before trusting it against a real signature vector.
use crypto::ecdsa_p256::*
use bignum::bigint::*

fn main() -> i64 with IO {
    let g = p256_base_point()
    let p = p256_field_prime()
    let a = p256_curve_a()
    let two_g = ec_point_double(&g, &a, &p)
    let three_g = ec_point_add(&two_g, &g, &p)
    let four_g_by_addition = ec_point_add(&three_g, &g, &p)
    let four = bigint_from_u32(4)
    let four_g_by_scalar_mul = ec_scalar_mul(&four, &g, &a, &p)
    if bigint_cmp(&four_g_by_addition.x, &four_g_by_scalar_mul.x) != 0 {
        println("ecdsa_p256_point_arithmetic: x mismatch between repeated addition and scalar mul")
        return 1
    }
    if bigint_cmp(&four_g_by_addition.y, &four_g_by_scalar_mul.y) != 0 {
        println("ecdsa_p256_point_arithmetic: y mismatch between repeated addition and scalar mul")
        return 1
    }
    println("ecdsa_p256_point_arithmetic: 4G by repeated addition matches 4G by scalar multiplication")
    return 0
}
```

This needs `p256_base_point()`/`p256_field_prime()`/`p256_curve_a()` accessor functions — add these as `pub fn` returning the hardcoded, independently-verified curve constants from Step 1, since the test file needs them and they shouldn't be duplicated as separate hardcoded literals in the test.

- [ ] **Step 3: Run test to verify it fails**, then **Step 4: implement `EcPoint`/`ec_point_add`/`ec_point_double`/`ec_scalar_mul`/the curve-constant accessors**, then **Step 5: run test to verify it passes** (`SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/ecdsa_p256_point_arithmetic.sio`).

- [ ] **Step 6: Write the failing signature-verification test using the real vector from Step 1**, mirroring Task 1's Step 2 structure (positive case + a corrupted-signature negative case). Include the vector's public key as an `EcPoint` built from its real x/y coordinates.

- [ ] **Step 7: Implement `ecdsa_p256_verify`** per the 5-step algorithm above, including `der_read_integer`-based DER decoding of the signature's `r`/`s`.

- [ ] **Step 8: Run test to verify it passes.**

- [ ] **Step 9: Mutation-test the modular-reduction and point-encoding logic**, matching the discipline that caught real bugs in Poly1305 and X25519 on this branch. Concretely: temporarily comment out or corrupt one specific piece of logic (e.g. skip the final `bigint_mod` reduction in `ec_mod_inverse`, or flip the order of `x`/`y` when building the verification result), re-run both test files, and confirm at least one fails. Record which mutation(s) you tried and which the suite caught in your report; if a mutation you tried does NOT get caught, strengthen the test (following the same "measure, don't assume" resolution the X25519 sub-project used when its own field-arithmetic tests were found too weak) before moving on — do not report DONE with a known-weak test.

- [ ] **Step 10: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix ecdsa_p256_ --jobs 2 --verbose`
Expected: all passed, 0 failed.

- [ ] **Step 11: Commit**

```bash
git add stdlib/crypto/ecdsa_p256.sio tests/run-pass/ecdsa_p256_point_arithmetic.sio tests/run-pass/ecdsa_p256_verify_vectors.sio
git commit -m "feat(crypto): add NIST P-256 point arithmetic and ECDSA verification"
```

---

## Task 3: Extend `stdlib/x509/{oid.sio,cert.sio}` — parse EC public keys

**Files:**
- Modify: `stdlib/x509/oid.sio`
- Modify: `stdlib/x509/cert.sio`
- Test: `tests/run-pass/x509_ec_public_key_parsing.sio`

**Interfaces:**
- Consumes: `EcPoint` (`stdlib/crypto/ecdsa_p256.sio`, Task 2), existing `DerReader`/`DerTag`/DER-decoding helpers already used throughout `cert.sio`.
- Produces: `Certificate` struct gains two new fields: `public_key_algorithm: i32` (a new `pub const PUBKEY_ALG_RSA: i32 = 0`, `pub const PUBKEY_ALG_EC_P256: i32 = 1`, `pub const PUBKEY_ALG_UNKNOWN: i32 = 2`) and `ec_public_key: EcPoint` (populated only when `public_key_algorithm == PUBKEY_ALG_EC_P256`; left as `EcPoint { x: bigint_zero(), y: bigint_zero() }` otherwise). Task 8 reads `leaf.public_key_algorithm` to decide whether to call `rsa_pss_verify` (Task 1) or `ecdsa_p256_verify` (Task 2, using `leaf.ec_public_key`).

Real X.509 background needed for this task: `SubjectPublicKeyInfo ::= SEQUENCE { algorithm AlgorithmIdentifier, subjectPublicKey BIT STRING }`. For RSA, `algorithm.algorithm` is the OID `1.2.840.10045.2.1`... **no** — RSA's OID is `1.2.840.113549.1.1.1` (`rsaEncryption`); EC's is `1.2.840.10045.2.1` (`id-ecPublicKey`), with a REQUIRED `algorithm.parameters` field that is itself an OID identifying the curve — for P-256 that's `1.2.840.10045.3.1.7` (`prime256v1`, also known as `secp256r1`). Independently verify both OID byte-encodings against a primary source (e.g. compute them yourself from the OID's dotted-decimal form using the standard ASN.1 OID encoding rules, or find them in an authoritative OID database) before hardcoding — do not trust the dotted-decimal numbers transcribed in this paragraph without re-deriving their DER byte encoding, since `oid.sio`'s existing 28 accessor functions already establish the exact style/pattern to follow (read a couple of them first).

For EC, `subjectPublicKey`'s BIT STRING content (after the leading "0 unused bits" byte) is the point in uncompressed form: `0x04 || X (32 bytes) || Y (32 bytes)` for P-256 — a compressed point (`0x02`/`0x03` prefix) is a valid encoding this task does NOT need to support (uncompressed is what OpenSSL and virtually every modern CA/server emits by default; if you encounter a compressed point during testing, that's a real interop finding to report, not something to silently support with guessed code).

- [ ] **Step 1: Read `stdlib/x509/cert.sio`'s existing RSA `SubjectPublicKeyInfo` parsing code in full** (search for where `modulus`/`public_exponent` are currently populated) to find the exact point where `algorithm.algorithm`'s OID is read and branches on `rsaEncryption` — this is where the new EC branch is added, not a new independent code path.

- [ ] **Step 2: Read `stdlib/x509/oid.sio`'s existing 28 accessor functions** to learn the established pattern (e.g. `fn oid_is_rsa_encryption(bytes: &RawBuf, len: i64) -> bool` or similar — match whatever the real existing naming/shape is) before adding `oid_is_ec_public_key`/`oid_is_prime256v1` in the same style.

- [ ] **Step 3: Generate a real test certificate with an EC (P-256) public key using openssl**

```bash
openssl ecparam -name prime256v1 -genkey -noout -out /tmp/ec_test_key.pem
openssl req -new -x509 -key /tmp/ec_test_key.pem -out /tmp/ec_test_cert.pem -days 1 -subj "/CN=ec-test.example"
openssl x509 -in /tmp/ec_test_cert.pem -outform DER -out /tmp/ec_test_cert.der
openssl x509 -in /tmp/ec_test_cert.pem -noout -text | grep -A5 "Subject Public Key Info"
openssl ec -in /tmp/ec_test_key.pem -text -noout | grep -A5 "pub:"
```

Use the `pub:` hex bytes openssl prints (the raw `04||X||Y` uncompressed point) as ground truth to compare your parser's extracted `EcPoint.x`/`EcPoint.y` against. Embed the DER certificate's bytes into the test file the same way every other `x509_*` test on this branch embeds a real openssl-generated fixture (as a byte array or read from a committed `.der` file under `tests/fixtures/` if that's the established pattern — check how existing `x509_*` tests do this and match it).

- [ ] **Step 4: Write the failing test**

```sio
//@ run-pass
// Parses a real openssl-generated EC (P-256) certificate and confirms
// the extracted public key point matches openssl's own reported value.
use x509::cert::*
use crypto::ecdsa_p256::*
use bignum::bigint::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in: load the DER cert bytes (Step 3's fixture) into a RawBuf,
    // call the existing certificate-parsing entry point (find its real
    // name in cert.sio -- likely something like x509_parse_certificate),
    // check the parse succeeded>
    if cert.public_key_algorithm != PUBKEY_ALG_EC_P256 {
        println("x509_ec_public_key_parsing: wrong algorithm detected")
        return 1
    }
    // <compare cert.ec_public_key.x / .y against the real x/y bytes
    // openssl reported in Step 3, built as BigInts from those bytes>
    println("x509_ec_public_key_parsing: EC public key extracted correctly")
    return 0
}
```

- [ ] **Step 5: Run test to verify it fails, implement the EC branch in `cert.sio` (using the new `oid.sio` accessors), run test to verify it passes.**

- [ ] **Step 6: Confirm no regression on existing RSA certificate parsing** — run the full existing `x509_` test filter:

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_ --jobs 2 --verbose`
Expected: every pre-existing `x509_*` test still passes, plus the new `x509_ec_public_key_parsing` test.

- [ ] **Step 7: Commit**

```bash
git add stdlib/x509/oid.sio stdlib/x509/cert.sio tests/run-pass/x509_ec_public_key_parsing.sio
git commit -m "feat(x509): parse EC (P-256) SubjectPublicKeyInfo"
```

---

## Task 4: `stdlib/tls/record.sio` — TLS 1.3 record layer

**Files:**
- Create: `stdlib/tls/record.sio`
- Test: `tests/run-pass/tls_record_rfc8448.sio`

**Interfaces:**
- Consumes: `aead_seal`/`aead_open`/`AEAD_AES_128_GCM`/`AEAD_AES_256_GCM`/`AEAD_CHACHA20_POLY1305` (`stdlib/crypto/aead.sio`), `RawBuf` (`stdlib/net/socket.sio`).
- Produces: `pub struct RecordLayerState { read_key: RawBuf, read_iv: RawBuf, read_seq: i64, write_key: RawBuf, write_iv: RawBuf, write_seq: i64, key_len: i64, iv_len: i64, aead_algo: i32 }`, `pub const TLS_CONTENT_TYPE_HANDSHAKE: u8 = 22`, `pub const TLS_CONTENT_TYPE_ALERT: u8 = 21`, `pub const TLS_CONTENT_TYPE_APPLICATION_DATA: u8 = 23`, `pub const TLS_CONTENT_TYPE_CHANGE_CIPHER_SPEC: u8 = 20`, `pub fn record_layer_new(aead_algo: i32, key_len: i64, iv_len: i64) -> RecordLayerState with IO` (all-zero keys/IVs, seq=0 — used before any keys are derived, for the initial plaintext ClientHello), `pub fn record_layer_rekey(state: &!RecordLayerState, read_key: &RawBuf, read_iv: &RawBuf, write_key: &RawBuf, write_iv: &RawBuf) with Mut, IO` (resets both sequence numbers to 0, per RFC 8446 §5.3 — a rekey always restarts the sequence number), `pub fn record_encrypt(state: &!RecordLayerState, content_type: u8, plaintext: &RawBuf, plaintext_len: i64) -> RawBuf with Mut, IO`, `pub fn record_decrypt(state: &!RecordLayerState, ciphertext_record_fragment: &RawBuf, fragment_len: i64) -> (u8, RawBuf, i64, bool) with Mut, IO`. Task 7/8 use these directly.

**Wire format (RFC 8446 §5.1, §5.2):**
- Outer `TLSCiphertext`: `opaque_type (1 byte, always 0x17=application_data once encryption is active) || legacy_record_version (2 bytes, always 0x0303) || length (2 bytes, big-endian, of the encrypted_record that follows) || encrypted_record`. This task's `record_encrypt`/`record_decrypt` operate on the `encrypted_record` payload only — writing/parsing the 5-byte outer header is Task 7/8's job when actually talking to a socket (this task exposes the AEAD-wrapped payload, not the full framed record, since the outer header needs no state from `RecordLayerState`).
- `TLSInnerPlaintext` (the AEAD's plaintext, before encryption): `content (the real content, e.g. actual handshake bytes) || content_type (1 byte, the REAL type: handshake=22, alert=21, application_data=23) || zeros (zero or more 0x00 padding bytes)`. This plan adds NO padding (`zeros` is always zero-length) — padding is a traffic-analysis countermeasure explicitly out of this plan's scope; `record_encrypt`'s inner plaintext is always exactly `plaintext || content_type`, no trailing zero bytes.
- AEAD nonce construction (RFC 8446 §5.3): the per-record nonce is the write/read IV (12 bytes) XORed with the sequence number encoded as a 12-byte value (8 bytes of zero left-padding, then the 8-byte big-endian sequence number) — NOT the IV alone, NOT the sequence number alone. `aead_seal`/`aead_open` are called with the outer `application_data` `opaque_type` byte, `legacy_record_version`, and the encrypted-record's length as Additional Authenticated Data (AAD) — the exact 5-byte outer header, per RFC 8446 §5.2.
- The sequence number increments by 1 after every record processed in a given direction (starting at 0 immediately after the most recent rekey), independently for read and write.

- [ ] **Step 1: Fetch RFC 8448 fresh** (`https://www.rfc-editor.org/rfc/rfc8448.txt`) and locate a real encrypted record's raw bytes together with the traffic secret/key/IV that produced it (Section 3 already publishes the client/server handshake traffic secrets and at least one encrypted handshake record's full wire bytes — find the earliest one, likely the client's encrypted Finished or the server's encrypted Certificate message). Independently derive the record's key+IV from the published traffic secret via `hkdf.sio`'s `tls13_traffic_key_and_iv` (already built and tested in the HKDF sub-project) to confirm you have the right key material before using it here — do not assume the HKDF sub-project's own values transfer without re-deriving them for this specific record.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
// Encrypts/decrypts a real handshake record using key material and
// wire bytes independently derived and re-verified from RFC 8448
// (see task report for exact section/line numbers used).
use tls::record::*
use crypto::aead::*
use crypto::hkdf::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in: build a RecordLayerState via record_layer_new then
    // record_layer_rekey with the real key/IV derived in Step 1>
    // <call record_decrypt on the real encrypted record fragment from
    // RFC 8448, assert the returned content_type/plaintext/plaintext_len
    // match the RFC's own published plaintext content and its stated
    // real content type>
    // <call record_encrypt on that same real plaintext + content_type,
    // assert the output matches RFC 8448's own published ciphertext
    // bytes exactly -- this only works if the sequence number is at the
    // same value RFC 8448's own record was encrypted at (likely 0, the
    // first record after rekey -- confirm this against the RFC rather
    // than assuming)>
    println("tls_record_rfc8448: record encrypt/decrypt match RFC 8448's real wire bytes")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails, implement `record.sio`, run test to verify it passes.**

- [ ] **Step 4: Add a round-trip test independent of RFC 8448** (encrypt then decrypt an arbitrary plaintext at sequence numbers 0, 1, and 254→255 rollover-adjacent — confirming the 8-byte sequence-number-to-nonce encoding handles a multi-byte sequence number correctly, not just the single-digit case RFC 8448's own trace happens to exercise) — assert the round-tripped plaintext and content_type match exactly.

- [ ] **Step 5: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix tls_record_ --jobs 2 --verbose`

- [ ] **Step 6: Commit**

```bash
git add stdlib/tls/record.sio tests/run-pass/tls_record_rfc8448.sio
git commit -m "feat(tls): add TLS 1.3 record-layer framing and AEAD wrapping"
```

---

## Task 5: `stdlib/tls/transcript.sio` — running transcript-hash accumulator

**Files:**
- Create: `stdlib/tls/transcript.sio`
- Test: `tests/run-pass/tls_transcript_rfc8448.sio`

**Interfaces:**
- Consumes: `sha256`/`sha384` (`stdlib/hash/{sha256,sha384}.sio`), `RawBuf` (`stdlib/net/socket.sio`).
- Produces: `pub struct TranscriptHash { hash_algo: i32, buf: RawBuf, len: i64, cap: i64 }`, `pub fn transcript_new(hash_algo: i32) -> TranscriptHash with IO`, `pub fn transcript_append(t: &!TranscriptHash, msg_bytes: &RawBuf, msg_len: i64) with Mut, IO` (grows `buf` if needed — reallocate to `2*(len+msg_len)` and copy old contents element-by-element when `len+msg_len > cap`, never `var x = *ref` on any fixed-array intermediate per §D8, though this function deals only in `RawBuf` so §D8 shouldn't directly apply — still, be explicit about the copy loop), `pub fn transcript_current_hash(t: &TranscriptHash) -> RawBuf with IO` (hashes the accumulated `buf[0..len]` via `sha256`/`sha384` per `hash_algo`, returns the digest as a `RawBuf`). Task 7/8 call `transcript_append` after every handshake message sent or received, and `transcript_current_hash` whenever `hkdf.sio`'s `derive_secret` needs a transcript-hash input.

`hash_algo` uses the same `HKDF_HASH_SHA256`/`HKDF_HASH_SHA384` constants `stdlib/crypto/hkdf.sio` already defines — import and reuse those exact constants rather than defining new ones in this file.

- [ ] **Step 1: Fetch RFC 8448 fresh** and locate its published transcript-hash values at two checkpoints: "through ServerHello" (used to derive the handshake traffic secrets) and "through server Finished" (used to derive the application traffic secrets) — the HKDF sub-project's own test (`tests/run-pass/tls13_keyschedule_rfc8448.sio`) already consumed these two exact values as inputs; read that file to find which RFC 8448 line numbers it cited, then re-fetch and re-confirm those values independently for this task's own use (don't just copy the other file's numbers without re-checking, since this task needs the RAW MESSAGE BYTES that hash to those values, not just the hash values themselves).

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
// Accumulates real RFC 8448 handshake message bytes and confirms the
// resulting transcript hash matches the RFC's own published values at
// two checkpoints.
use tls::transcript::*
use crypto::hkdf::*
use net::socket::*

fn main() -> i64 with IO {
    var t = transcript_new(HKDF_HASH_SHA256)
    // <fill in: build RawBufs from RFC 8448's real ClientHello and
    // ServerHello raw message bytes (the full handshake-message-layer
    // bytes, i.e. the 4-byte handshake header + body, not the record
    // layer's own 5-byte header), transcript_append both in order>
    let hash_through_sh = transcript_current_hash(&t)
    // <assert hash_through_sh matches RFC 8448's published
    // "hash through ServerHello" value byte-for-byte>

    // <continue: transcript_append EncryptedExtensions, Certificate,
    // CertificateVerify, and the server's Finished raw bytes, all from
    // RFC 8448>
    let hash_through_finished = transcript_current_hash(&t)
    // <assert hash_through_finished matches RFC 8448's published
    // "hash through server Finished" value byte-for-byte>

    println("tls_transcript_rfc8448: transcript hash matches RFC 8448 at both checkpoints")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails, implement `transcript.sio`, run test to verify it passes.**

- [ ] **Step 4: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix tls_transcript_ --jobs 2 --verbose`

- [ ] **Step 5: Commit**

```bash
git add stdlib/tls/transcript.sio tests/run-pass/tls_transcript_rfc8448.sio
git commit -m "feat(tls): add running transcript-hash accumulator"
```

---

## Task 6: `stdlib/tls/handshake.sio` — handshake message wire codec

**Files:**
- Create: `stdlib/tls/handshake.sio`
- Test: `tests/run-pass/tls_handshake_codec_rfc8448.sio`

**Interfaces:**
- Consumes: `RawBuf` (`stdlib/net/socket.sio`), `Certificate`/parsing helpers (`stdlib/x509/cert.sio`, for decoding the Certificate message's embedded X.509 certificates).
- Produces: message-type constants `pub const HS_CLIENT_HELLO: u8 = 1`, `HS_SERVER_HELLO: u8 = 2`, `HS_NEW_SESSION_TICKET: u8 = 4`, `HS_ENCRYPTED_EXTENSIONS: u8 = 8`, `HS_CERTIFICATE: u8 = 11`, `HS_CERTIFICATE_VERIFY: u8 = 15`, `HS_FINISHED: u8 = 20`, `HS_KEY_UPDATE: u8 = 24`; `pub struct ClientHelloParams { random: [u8;32], x25519_public: [u8;32], server_name: RawBuf, server_name_len: i64, cookie: RawBuf, cookie_len: i64 }` (cookie empty/len=0 on the first ClientHello; populated when re-sending after a HelloRetryRequest); `pub fn encode_client_hello(params: &ClientHelloParams) -> (RawBuf, i64) with IO` (returns the full handshake-layer message: 4-byte header + body); `pub struct ServerHelloInfo { is_hello_retry_request: bool, random: [u8;32], cipher_suite: u16, x25519_public: [u8;32], has_key_share: bool, cookie: RawBuf, cookie_len: i64 }`; `pub fn decode_server_hello(msg_bytes: &RawBuf, msg_len: i64) -> (ServerHelloInfo, bool) with IO`; `pub struct CertificateVerifyInfo { signature_scheme: u16, signature: RawBuf, signature_len: i64 }`; `pub fn decode_certificate_verify(msg_bytes: &RawBuf, msg_len: i64) -> (CertificateVerifyInfo, bool) with IO`; `pub fn decode_certificate_message(msg_bytes: &RawBuf, msg_len: i64) -> (Certificate, RawBuf, [Certificate;8], [RawBuf;8], i32, bool) with IO` — return tuple order matches `x509_verify_chain`'s own parameter order exactly: `(leaf, leaf_buf, intermediates, intermediate_bufs, intermediate_count, ok)`, so Task 7 can pass this function's output straight into `x509_verify_chain` with no reshaping. The Certificate message's wire format (RFC 8446 §4.4.2) is a list of `(cert_data, extensions)` pairs — this plan ignores each certificate's per-entry extensions (e.g. OCSP stapling status_request extensions are not carried this way in this plan's scope; `x509_verify_chain`'s own `ocsp_response` parameter can be passed as an empty/zero-length RawBuf here, since this plan does not implement the separate `status_request` extension mechanism for OCSP stapling within the Certificate message). The first certificate in the list is `leaf`; up to 8 more are `intermediates`, matching `x509_verify_chain`'s existing fixed-size array convention — if the server sends more than 8 intermediates (very unusual), fail with a negative sentinel rather than silently truncating.; `pub fn encode_finished(verify_data: &RawBuf, verify_data_len: i64) -> (RawBuf, i64) with IO`; `pub fn decode_finished(msg_bytes: &RawBuf, msg_len: i64) -> (RawBuf, i64) with IO` (extracts `verify_data`); `pub fn is_new_session_ticket(msg_type: u8) -> bool`, `pub fn is_key_update(msg_type: u8) -> bool` (trivial type-tag checks Task 8's `tls_recv` uses to decide to discard/rekey without fully decoding the body). Task 7/8 call every one of these.

Signature scheme codepoints (RFC 8446 §4.2.3, needed to fill `ClientHelloParams`'s implicit `signature_algorithms` extension list and to interpret `CertificateVerifyInfo.signature_scheme`): `rsa_pss_rsae_sha256 = 0x0804`, `rsa_pss_rsae_sha384 = 0x0805`, `ecdsa_secp256r1_sha256 = 0x0403` — independently verify these three codepoints against RFC 8446 §4.2.3's own table before hardcoding them.

Handshake message framing (RFC 8446 §4): every message is `msg_type (1 byte) || length (3 bytes, big-endian) || body`. Extension framing within ClientHello/ServerHello/EncryptedExtensions bodies: `extension_type (2 bytes) || extension_data length (2 bytes) || extension_data`.

- [ ] **Step 1: Fetch RFC 8448 fresh** and extract the full raw bytes (not just semantic field values) of its ClientHello, ServerHello, EncryptedExtensions, Certificate, CertificateVerify, and both Finished messages — this is the same trace already used by Tasks 4/5, cross-reference for consistency but re-verify independently rather than assuming a fetch done for a different task transfers unchanged.

- [ ] **Step 2: Write the failing round-trip test**

```sio
//@ run-pass
// Decodes RFC 8448's real handshake messages and confirms the decoded
// fields match the RFC's own stated semantic values; round-trips
// ClientHello encode/decode for self-consistency.
use tls::handshake::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in: build a RawBuf from RFC 8448's real ServerHello message
    // bytes, call decode_server_hello, assert cipher_suite == 0x1301
    // (TLS_AES_128_GCM_SHA256, matching this branch's earlier
    // confirmation), assert x25519_public matches the RFC's published
    // key_share value>

    // <build a ClientHelloParams from values you control (not
    // RFC-derived, since this direction is self-consistency, not
    // RFC-conformance), call encode_client_hello, then decode enough of
    // it back (or re-encode a decoded round-trip if you add a
    // decode_client_hello -- only needed if this task's own logic
    // benefits from one; not required by any consumer, so add it only
    // if it's the cleanest way to test encode_client_hello's output
    // shape)>

    // <decode RFC 8448's real Certificate message, assert the leaf
    // certificate's parsed Subject/DER bytes match what openssl reports
    // for that same certificate (RFC 8448 §3 states the certificate's
    // contents are illustrative test material -- verify what you can
    // about its structure, e.g. certificate_list length, without
    // depending on it chaining to a real trust anchor, since RFC 8448's
    // certificate is not signed by a real CA>

    // <decode RFC 8448's real CertificateVerify message, assert
    // signature_scheme matches its stated value (RFC 8448 uses
    // rsa_pss_rsae_sha256 for this trace -- verify against the RFC
    // rather than assuming)>

    // <decode RFC 8448's real server Finished message, assert the
    // extracted verify_data matches the RFC's own published value>

    println("tls_handshake_codec_rfc8448: decoded fields match RFC 8448")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails, implement `handshake.sio`, run test to verify it passes.**

- [ ] **Step 4: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix tls_handshake_ --jobs 2 --verbose`

- [ ] **Step 5: Commit**

```bash
git add stdlib/tls/handshake.sio tests/run-pass/tls_handshake_codec_rfc8448.sio
git commit -m "feat(tls): add TLS 1.3 handshake message wire codec"
```

---

## Task 7: `stdlib/tls/client.sio` (part A) — handshake orchestration through derived application keys

**Files:**
- Create: `stdlib/tls/client.sio`
- Test: `tests/run-pass/tls_client_handshake_loopback.sio`

**Interfaces:**
- Consumes: everything — `tcp_connect`/`tcp_send`/`tcp_recv`/`tcp_close` (`stdlib/net/socket.sio`), `record_layer_new`/`record_layer_rekey`/`record_encrypt`/`record_decrypt`/`TLS_CONTENT_TYPE_*` (Task 4), `transcript_new`/`transcript_append`/`transcript_current_hash` (Task 5), every `encode_*`/`decode_*`/`is_*` function (Task 6), `x25519_base_point_mul`/`x25519`/`x25519_clamp` (`stdlib/crypto/x25519.sio`), `crypto_os_random_bytes` (`stdlib/crypto/csprng.sio`), `hkdf_extract`/`tls13_early_secret`/`tls13_handshake_secret`/`tls13_master_secret`/`derive_secret`/`tls13_traffic_key_and_iv`/`HKDF_HASH_SHA256` (`stdlib/crypto/hkdf.sio`), `x509_verify_chain` (`stdlib/x509/chain.sio`), `rsa_pss_verify` (Task 1), `ecdsa_p256_verify` (Task 2), `PUBKEY_ALG_EC_P256` (Task 3).
- Produces: `pub struct TlsConnection { sock: TcpSocket, records: RecordLayerState, hash_algo: i32, hs_transcript: TranscriptHash, master_secret: RawBuf, client_ap_secret: RawBuf, server_ap_secret: RawBuf }`, `pub fn tls_connect(host: &RawBuf, host_len: i64, port: u16, trust_store: &TrustStore, now_unix: i64) -> (TlsConnection, i64) with IO` (0 on success, negative sentinel on failure — this task's own scope ends once application traffic secrets are derived and the record layer is rekeyed for application data; it does NOT send or receive any application data itself). Task 8 consumes the `TlsConnection` this produces and adds `tls_send`/`tls_recv`/`tls_close` in the same file.

This task's scope is exactly the spec's Data Flow steps 1-4: build+send ClientHello, handle at most one HelloRetryRequest retry, receive+process ServerHello, derive handshake secrets and rekey, receive+process EncryptedExtensions/Certificate/CertificateVerify/server-Finished (verifying the chain via `x509_verify_chain`, the signature via `rsa_pss_verify`/`ecdsa_p256_verify` per `leaf.public_key_algorithm`, and the server's Finished via HMAC), send the client's own Finished, derive Master Secret and application traffic secrets, rekey for application data. It stops there — `tls_connect` returns a `TlsConnection` whose record layer is ready for application data, but does not exchange any.

The server's Finished verification (RFC 8446 §4.4.4): `finished_key = HKDF-Expand-Label(server_handshake_traffic_secret, "finished", "", Hash.length)` (via `hkdf_expand_label` — note this is a case where `context` is empty, `context_len=0`), `verify_data = HMAC(finished_key, transcript_hash_through_certificate_verify)` (via `hmac_sha256_rawbuf`/`hmac_sha384_rawbuf` from `hkdf.sio` — HMAC here, NOT another `derive_secret`/`hkdf_expand` call). Compare against the received Finished message's `verify_data`, constant-time (bitwise-accumulate-then-compare-once, matching this codebase's established AEAD-tag-comparison discipline). The client's own Finished uses the identical construction with `client_handshake_traffic_secret` and the transcript hash through the server's Finished (i.e., one message later than the server's own Finished computation's transcript point).

- [ ] **Step 1: Set up a local TLS 1.3 test server on loopback for repeatable testing**

```bash
openssl req -x509 -newkey rsa:2048 -keyout /tmp/tls_test_key.pem -out /tmp/tls_test_cert.pem -days 1 -nodes -subj "/CN=localhost"
openssl s_server -key /tmp/tls_test_key.pem -cert /tmp/tls_test_cert.pem -accept 44330 -tls1_3 -www &
```

Confirmed empirically before this plan was written: loopback TCP works in this sandbox, outbound internet does not — use `127.0.0.1:44330` as the test target, never a public hostname. If `openssl s_server` fails to bind or the sandbox blocks even loopback listening sockets, report this specific failure mode (BLOCKED, with the exact error) rather than proceeding on an assumption — do not fabricate or skip this test.

For chain verification against this self-signed test server, build a `TrustStore` containing exactly this test certificate as its own trust anchor (self-signed, self-trusted, for test purposes only) — check `stdlib/x509/trust_store.sio` for how to construct a `TrustStore` from an in-memory certificate rather than only from the system bundle path.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
// Drives a real TLS 1.3 handshake against a local openssl s_server on
// loopback, using an RSA test certificate (openssl's default key type),
// confirming the connection reaches the application-key-derived state.
use tls::client::*
use x509::trust_store::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in: build a RawBuf "127.0.0.1", build a TrustStore trusting
    // /tmp/tls_test_cert.pem (loaded and parsed as the sole trust
    // anchor), call tls_connect(host, 8, 44330, &trust_store, now_unix)>
    if result_code != 0 {
        println("tls_client_handshake_loopback: handshake failed")
        return 1
    }
    println("tls_client_handshake_loopback: handshake reached application keys")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails** (unresolved import), **implement `client.sio`'s handshake-through-rekey logic**, **run test to verify it passes** against the real local server.

- [ ] **Step 4: Repeat Step 1-3 with an ECDSA P-256 test certificate** (`openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 ...`, same pattern) on a second port, confirming the `ecdsa_p256_verify` path (Task 2) and the EC-public-key parsing path (Task 3) both work end-to-end against a real, independently-generated certificate and a real, independently-computed signature — not just the earlier tasks' own unit-level test vectors.

- [ ] **Step 5: Adversarial case — wrong/self-signed-but-untrusted certificate.** Point `tls_connect` at the loopback server from Step 1 but with a `TrustStore` that does NOT contain that test certificate (e.g. an empty trust store, or one containing an unrelated certificate). Assert `tls_connect` returns a negative sentinel (fails closed) rather than succeeding.

- [ ] **Step 6: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix tls_client_ --jobs 2 --verbose`

- [ ] **Step 7: Commit**

```bash
git add stdlib/tls/client.sio tests/run-pass/tls_client_handshake_loopback.sio
git commit -m "feat(tls): add TLS 1.3 handshake orchestration through application-key derivation"
```

---

## Task 8: `stdlib/tls/client.sio` (part B) — application-data lifecycle

**Files:**
- Modify: `stdlib/tls/client.sio` (append to the file Task 7 created)
- Test: `tests/run-pass/tls_client_application_data_loopback.sio`

**Interfaces:**
- Consumes: `TlsConnection` (Task 7, same file), `record_encrypt`/`record_decrypt`/`TLS_CONTENT_TYPE_APPLICATION_DATA`/`TLS_CONTENT_TYPE_ALERT` (Task 4), `is_new_session_ticket`/`is_key_update` (Task 6), `derive_secret`/`HKDF_HASH_SHA256` (`stdlib/crypto/hkdf.sio`).
- Produces: `pub fn tls_send(conn: TlsConnection, buf: &RawBuf, len: i64) -> (TlsConnection, i64) with IO`, `pub fn tls_recv(conn: TlsConnection, buf: &RawBuf, cap: i64) -> (TlsConnection, i64) with IO` (0 = EOF/`close_notify`, negative = error, positive = bytes read — mirrors `tcp_recv`'s exact contract), `pub fn tls_close(conn: TlsConnection) with IO`. This is the final public interface of the whole TLS-on-Madaros effort — nothing later consumes it within this plan, but it's what the Conclave integration (a future, separate piece of work, not part of this plan) will call.

`tls_send`: wraps `buf[0..len]` as one or more `application_data` records via `record_encrypt` (splitting into multiple records only if `len` exceeds a single record's practical size — RFC 8446 §5.1 caps `TLSPlaintext.length` at `2^14` bytes; for a simple HTTP GET request this plan's own realistic payloads are far under that, but implement the splitting loop correctly rather than assuming a single request always fits), writes each resulting record (5-byte outer header + `record_encrypt`'s output) via `tcp_send`.

`tls_recv`: reads a record's 5-byte outer header via `tcp_recv` (looping until all 5 bytes arrive — `tcp_recv` may return fewer bytes than requested, matching normal POSIX `recv` semantics; do not assume one `tcp_recv` call returns a whole header), reads `length` more bytes for the encrypted fragment (looping the same way), calls `record_decrypt`. If the decoded content type is `TLS_CONTENT_TYPE_APPLICATION_DATA`, copies the decrypted plaintext into the caller's `buf` (up to `cap`) and returns the byte count. If it's a handshake-layer message and `is_new_session_ticket` of its type tag is true, discard and loop to read the next record instead of returning. If `is_key_update` is true: re-derive the connection's *own-direction* traffic secret via `derive_secret(hash_algo, current_traffic_secret, ..., "traffic upd", ..., <empty context>, ..., Hash.length)` — independently verify this exact label string (`"traffic upd"`) against RFC 8446 §7.2 before hardcoding it — call `record_layer_rekey` with the new key/IV (derived via `tls13_traffic_key_and_iv` from the new secret) for the read direction only, then loop to read the next record. If it's `TLS_CONTENT_TYPE_ALERT` and the alert is `close_notify` (independently verify `close_notify`'s exact 2-byte alert encoding — level + description — against RFC 8446 §6 before hardcoding), return 0 (EOF). Any other alert or malformed record: return a negative sentinel.

`tls_close`: sends a `close_notify` alert record via `record_encrypt`/`tcp_send` (best-effort — if the send fails, proceed to close anyway, don't hang the caller over a peer that already disappeared), then `tcp_close`.

- [ ] **Step 1: Extend Task 7's loopback test server usage to actually serve content.** `openssl s_server -www` (already used in Task 7) responds to any HTTP-like request with a status page — this is enough to exercise a real request/response round-trip without needing a real HTTP server.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
// Completes a handshake (reusing Task 7's approach), sends a minimal
// HTTP/1.1 request, and reads back a real response over the encrypted
// connection -- proving the full application-data lifecycle works
// against a real peer.
use tls::client::*
use x509::trust_store::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in: same tls_connect setup as Task 7's test, against the
    // same or a fresh openssl s_server -www instance>
    let request_str = "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n"
    // <build a RawBuf from request_str, tls_send it>
    // <loop calling tls_recv into a buffer until it returns 0 (EOF) or
    // a negative error, accumulating the response bytes>
    // <assert the accumulated response contains an HTTP status line,
    // e.g. checks for "HTTP/1." at the start of the accumulated bytes>
    // tls_close(conn)
    println("tls_client_application_data_loopback: sent request, received real HTTP response over TLS")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails, implement `tls_send`/`tls_recv`/`tls_close`, run test to verify it passes.**

- [ ] **Step 4: NewSessionTicket-discard test.** `openssl s_server` sends NewSessionTicket messages immediately after the handshake by default — Step 2's test already exercises this path implicitly if `tls_recv`'s first call(s) have to skip past one or more tickets before returning real application data; make this explicit by asserting the test's `tls_recv` loop actually observes and discards at least one NewSessionTicket (e.g. via a debug counter or log the implementer adds temporarily to confirm, then removes, or a permanent return-value/counter Task 8 finds a clean way to expose for testing — implementer's judgment on the cleanest way to make this observable without polluting the public API for a test-only concern).

- [ ] **Step 5: close_notify-as-EOF test.** After receiving the full response, confirm the server's connection close is observed by `tls_recv` returning exactly 0, not a negative error code and not hanging.

- [ ] **Step 6: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix tls_client_ --jobs 2 --verbose`

- [ ] **Step 7: Commit**

```bash
git add stdlib/tls/client.sio tests/run-pass/tls_client_application_data_loopback.sio
git commit -m "feat(tls): add application-data lifecycle (tls_send/tls_recv/tls_close)"
```

---

## After This Plan

`tls_connect`/`tls_send`/`tls_recv`/`tls_close` (`stdlib/tls/client.sio`) are the complete, final public interface of the TLS-on-Madaros effort. A future, separate piece of work (not part of this plan) wires HTTP/1.1 request/response framing over this connection the way `stdlib/net/http_client.sio` already does over plain `TcpSocket` — likely a thin `https_get()` built on top, or a direct integration into the Conclave web-search tool. TLS 1.2 fallback support, if ever needed, is its own future sub-project requiring a fresh brainstorming session (its own PRF, its own handshake message set, largely disjoint from everything built here).

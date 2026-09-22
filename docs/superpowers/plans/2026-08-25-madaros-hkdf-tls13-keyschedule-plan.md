<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-25-madaros-hkdf-tls13-keyschedule-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-25-madaros-hkdf-tls13-keyschedule-plan
-->

# Madaros HKDF + TLS 1.3 Key Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** implement RFC 5869 HKDF, RFC 8446 §7.1's `HKDF-Expand-Label`/`Derive-Secret` wrappers, and the minimal TLS 1.3 key-derivation ladder a fresh HTTPS client connection needs, verified against a complete, real, published TLS 1.3 handshake trace (RFC 8448).

**Architecture:** one new module (`stdlib/crypto/hkdf.sio`), built bottom-up in three tasks — HMAC, then generic HKDF, then the TLS 1.3-specific wrappers and secret ladder, each verified against its own published test vectors before the next task builds on it.

**Tech Stack:** Sounio/Madaros (`./bin/souc`), no non-Sounio code anywhere in this plan's deliverables.

**Spec:** `docs/superpowers/specs/2026-08-25-madaros-hkdf-tls13-keyschedule-design.md`

## Global Constraints

- **RawBuf-based throughout** — `net::socket::*` for `RawBuf`/`rawbuf_new`/`rawbuf_get`/`rawbuf_set`.
- **Does not touch `stdlib/crypto/hmac.sio`** (existing, file-private, fixed-array-based, untested by this branch's work) — left exactly as-is. This plan's `hmac_sha256`/`hmac_sha384` are fresh functions built on `stdlib/hash/sha256.sio`'s `sha256(buf: &RawBuf, len: i64) -> [u8;32] with IO` and `stdlib/hash/sha384.sio`'s `sha384(buf: &RawBuf, len: i64) -> [u8;48] with IO` (both already `pub`, already confirmed working — no verification spike needed for these two).
- Scope is deliberately minimal: only the key-schedule path a fresh, no-PSK, no-early-data, no-resumption connection uses. `binder_key`, `early_exporter_master_secret`, `client_early_traffic_secret`, `exporter_master_secret`, `resumption_master_secret`, post-handshake key updates, and TLS 1.2's PRF are all explicitly out of scope.
- Sentinel `bool`/`i64` error convention if any genuine failure mode is found during implementation — never `Result<T,E>`/`Option<T>`. No new error codes are anticipated by the spec; do not invent one speculatively.
- Every test's expected value comes from a published, independent source (RFC 5869, RFC 4231, RFC 8448) — never self-generated. **Every cited vector below must be independently re-verified against the actual RFC text before being trusted in committed test code** — this plan's own transcriptions are a starting point, not a substitute for checking the primary source, per this project's standing discipline (the AEAD ciphers and X25519 sub-projects on this exact branch both found real, self-consistent bugs that were invisible to mandated test vectors and only caught by adversarial, beyond-the-mandate scrutiny).
- Module imports: bare `use <filename>::{name|*}`, no `module` declaration.
- Commit convention: Conventional Commits (`feat(crypto): ...`). **Never add an AI-attribution line to any commit.**
- Test invocation: `bash scripts/run_sio_test_suite.sh --filter-prefix hmac_` / `hkdf_` / `tls13_keyschedule_` as appropriate — never the whole-repo suite. Use `--jobs 2` (this exact branch has repeatedly found high parallelism produces spurious 30-second timeouts on this test runner).
- No Rust, Python, or any non-Sounio code in this plan's shipped deliverables (Python used only to independently cross-check a test vector during development is fine — the shipped `.sio` files must be pure Sounio).
- Run every shell command as a plain foreground command — never a background/Monitor-style wait mechanism.
- Two real Madaros compiler defects are documented and apply here if relevant: (a) `var x = *ref` on any fixed-array reference ALIASES the caller's array instead of copying it (`docs/handoff/souc_v0800_defects.md` §D8) — copy fixed arrays element-by-element, never via dereference-assignment; (b) native `u32` arithmetic doesn't wrap mod 2^32 — not directly relevant to this plan's byte-oriented HMAC/HKDF logic, but be aware if any 32-bit-word-shaped sub-problem arises (`stdlib/hash/word32.sio` has the safe primitives if needed).

---

## Task 1: `stdlib/crypto/hkdf.sio` (part 1) — RawBuf-based HMAC-SHA256/SHA384

**Files:**
- Create: `stdlib/crypto/hkdf.sio`
- Test: `tests/run-pass/hmac_sha256_sha384_vectors.sio`

**Interfaces:**
- Consumes: `sha256(buf: &RawBuf, len: i64) -> [u8;32] with IO` (`stdlib/hash/sha256.sio`), `sha384(buf: &RawBuf, len: i64) -> [u8;48] with IO` (`stdlib/hash/sha384.sio`).
- Produces: `pub fn hmac_sha256(key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> [u8;32] with IO`, `pub fn hmac_sha384(key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> [u8;48] with IO`. Tasks 2 and 3 import and call these exact names/signatures.

Standard HMAC construction (FIPS 198-1 / RFC 2104), for both hash functions: if `key_len` exceeds the hash's block size (64 bytes for both SHA-256 and SHA-384 — SHA-384's block size is 128 bytes for the underlying compression function, but **independently verify this specific detail before assuming it**: HMAC's own key-shortening step operates on the hash function's BLOCK size, and SHA-384 (a member of the SHA-512 family) uses 128-byte blocks, not 64 — this is a common point of confusion since SHA-256's block size IS 64 bytes; get this right for HMAC-SHA384 specifically), hash the key down to the hash's output length first. Otherwise zero-pad the key up to the block size. Then compute `Hash((key XOR opad) || Hash((key XOR ipad) || message))` with `ipad = 0x36` repeated and `opad = 0x5c` repeated for the block size.

- [ ] **Step 1: Write the failing test using RFC 4231's published test vectors**

```sio
//@ run-pass
// RFC 4231 Test Case 1 for HMAC-SHA-256 and HMAC-SHA-384.
// INDEPENDENTLY RE-VERIFY every hex string below against the actual RFC
// 4231 text before trusting this transcription.
use crypto::hkdf::*
use net::socket::*

fn main() -> i64 with IO {
    // Key = 0x0b repeated 20 times; Data = "Hi There" (ASCII, 8 bytes)
    var key_bytes: [u8;20] = [0x0b;20]
    let key_buf = rawbuf_new(20 + 8)
    var i: i64 = 0
    while i < 20 { rawbuf_set(&key_buf, i, key_bytes[i as usize] as i64); i = i + 1 }

    let msg_str = "Hi There"
    let msg_len: i64 = 8
    let msg_buf = rawbuf_new(msg_len + 8)
    var j: i64 = 0
    while j < msg_len { rawbuf_set(&msg_buf, j, str_char_at(msg_str, j) as i64); j = j + 1 }

    // RFC 4231 Test Case 1 expected HMAC-SHA-256:
    // b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7
    let expected_sha256: [u8;32] = [
        0xb0,0x34,0x4c,0x61,0xd8,0xdb,0x38,0x53,0x5c,0xa8,0xaf,0xce,0xaf,0x0b,0xf1,0x2b,
        0x88,0x1d,0xc2,0x00,0xc9,0x83,0x3d,0xa7,0x26,0xe9,0x37,0x6c,0x2e,0x32,0xcf,0xf7
    ]
    let result_sha256 = hmac_sha256(&key_buf, 20, &msg_buf, msg_len)
    var k: i64 = 0
    while k < 32 {
        if result_sha256[k as usize] != expected_sha256[k as usize] {
            print_int(1)
            return 1
        }
        k = k + 1
    }

    // RFC 4231 Test Case 1 expected HMAC-SHA-384:
    // afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6
    let expected_sha384: [u8;48] = [
        0xaf,0xd0,0x39,0x44,0xd8,0x48,0x95,0x62,0x6b,0x08,0x25,0xf4,0xa1,0xb4,0x69,0x07,
        0xf1,0x5f,0x9d,0xad,0xbe,0x41,0x01,0xec,0x68,0x2a,0xa0,0x34,0xc7,0xce,0xbc,0x59,
        0xcf,0xae,0xa9,0xea,0x90,0x76,0xed,0xe7,0xf4,0xaf,0x15,0x2e,0x8b,0x2f,0xa9,0xcb
    ]
    let result_sha384 = hmac_sha384(&key_buf, 20, &msg_buf, msg_len)
    var m: i64 = 0
    while m < 48 {
        if result_sha384[m as usize] != expected_sha384[m as usize] {
            print_int(2)
            return 1
        }
        m = m + 1
    }

    println("hmac_sha256_sha384_vectors: both match RFC 4231 Test Case 1")
    return 0
}
```

Verify `str_char_at`'s real signature/behavior against `stdlib/str/lib.sio` before trusting it compiles as sketched (grep for existing callers of `str_char_at` — e.g. `tests/run-pass/hash_sha256_vectors.sio` from an earlier plan on this branch used this exact pattern successfully).

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/hmac_sha256_sha384_vectors.sio`
Expected: unresolved-import error referencing `crypto::hkdf`.

- [ ] **Step 3: Implement `hmac_sha256`/`hmac_sha384` in `stdlib/crypto/hkdf.sio`**

```sio
use net::socket::*
use hash::sha256::*
use hash::sha384::*

const SHA256_BLOCK_LEN: i64 = 64
const SHA384_BLOCK_LEN: i64 = 128   // INDEPENDENTLY VERIFY: SHA-384's block size is 128 bytes (it's a SHA-512-family member), NOT 64 -- do not assume it matches SHA-256's block size.

pub fn hmac_sha256(key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> [u8;32] with IO {
    // <fill in: standard HMAC construction (RFC 2104/FIPS 198-1) using
    // SHA256_BLOCK_LEN=64, sha256() for both the key-shortening step (if
    // key_len > 64) and the inner/outer hash computations. Build the
    // key/ipad/opad-XORed blocks into fresh RawBufs (never `var x = *ref`
    // on any fixed-array intermediate), concatenate ipad-block ++ msg for
    // the inner hash, then opad-block ++ inner_hash_output for the outer
    // hash.>
}

pub fn hmac_sha384(key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> [u8;48] with IO {
    // <fill in: identical structure to hmac_sha256, using SHA384_BLOCK_LEN=128
    // and sha384() instead>
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/hmac_sha256_sha384_vectors.sio`
Expected: prints `hmac_sha256_sha384_vectors: both match RFC 4231 Test Case 1`, exit 0.

- [ ] **Step 5: Add a second RFC 4231 test case exercising the key-longer-than-block-size path**

RFC 4231 Test Case 6 or 7 uses a key longer than the hash's block size (forcing the key-shortening-via-hash step) — independently verify and use whichever RFC 4231 test case actually has this property for both SHA-256 and SHA-384 (check the RFC directly; don't assume which numbered case it is from memory). Add this as a second assertion block in the same test file or a new one.

- [ ] **Step 6: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix hmac_ --jobs 2 --verbose`
Expected: all passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add stdlib/crypto/hkdf.sio tests/run-pass/hmac_sha256_sha384_vectors.sio
git commit -m "feat(crypto): add RawBuf-based HMAC-SHA256/SHA384"
```

---

## Task 2: `stdlib/crypto/hkdf.sio` (part 2) — HKDF-Extract/Expand

**Files:**
- Modify: `stdlib/crypto/hkdf.sio` (append to the file Task 1 created)
- Test: `tests/run-pass/hkdf_rfc5869_vectors.sio`

**Interfaces:**
- Consumes: `hmac_sha256` (Task 1, same file).
- Produces: `pub const HKDF_HASH_SHA256: i32 = 0`, `pub const HKDF_HASH_SHA384: i32 = 1`, `pub fn hkdf_extract(hash_algo: i32, salt: &RawBuf, salt_len: i64, ikm: &RawBuf, ikm_len: i64) -> RawBuf with IO`, `pub fn hkdf_expand(hash_algo: i32, prk: &RawBuf, prk_len: i64, info: &RawBuf, info_len: i64, out_len: i64) -> RawBuf with IO`. Task 3 imports and calls these exact names/signatures.

- [ ] **Step 1: Write the failing test using RFC 5869 Appendix A's published HKDF-SHA256 vectors**

```sio
//@ run-pass
// RFC 5869 Appendix A.1's basic HKDF-SHA256 test case.
// INDEPENDENTLY RE-VERIFY every hex string below against the actual RFC
// 5869 text before trusting this transcription.
use crypto::hkdf::*
use net::socket::*

fn main() -> i64 with IO {
    // IKM = 0x0b repeated 22 times
    var ikm_bytes: [u8;22] = [0x0b;22]
    let ikm_buf = rawbuf_new(22 + 8)
    var i: i64 = 0
    while i < 22 { rawbuf_set(&ikm_buf, i, ikm_bytes[i as usize] as i64); i = i + 1 }

    // salt = 000102030405060708090a0b0c (13 bytes)
    let salt_bytes: [u8;13] = [0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c]
    let salt_buf = rawbuf_new(13 + 8)
    var j: i64 = 0
    while j < 13 { rawbuf_set(&salt_buf, j, salt_bytes[j as usize] as i64); j = j + 1 }

    // info = f0f1f2f3f4f5f6f7f8f9 (10 bytes)
    let info_bytes: [u8;10] = [0xf0,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9]
    let info_buf = rawbuf_new(10 + 8)
    var k: i64 = 0
    while k < 10 { rawbuf_set(&info_buf, k, info_bytes[k as usize] as i64); k = k + 1 }

    // Expected PRK (32 bytes): 077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5
    let expected_prk: [u8;32] = [
        0x07,0x77,0x09,0x36,0x2c,0x2e,0x32,0xdf,0x0d,0xdc,0x3f,0x0d,0xc4,0x7b,0xba,0x63,
        0x90,0xb6,0xc7,0x3b,0xb5,0x0f,0x9c,0x31,0x22,0xec,0x84,0x4a,0xd7,0xc2,0xb3,0xe5
    ]
    let prk = hkdf_extract(HKDF_HASH_SHA256, &salt_buf, 13, &ikm_buf, 22)
    var p: i64 = 0
    while p < 32 {
        if rawbuf_get(&prk, p) != (expected_prk[p as usize] as i64) {
            print_int(1)
            return 1
        }
        p = p + 1
    }

    // Expected OKM (42 bytes): 3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865
    let expected_okm: [u8;42] = [
        0x3c,0xb2,0x5f,0x25,0xfa,0xac,0xd5,0x7a,0x90,0x43,0x4f,0x64,0xd0,0x36,0x2f,0x2a,
        0x2d,0x2d,0x0a,0x90,0xcf,0x1a,0x5a,0x4c,0x5d,0xb0,0x2d,0x56,0xec,0xc4,0xc5,0xbf,
        0x34,0x00,0x72,0x08,0xd5,0xb8,0x87,0x18,0x58,0x65
    ]
    let okm = hkdf_expand(HKDF_HASH_SHA256, &prk, 32, &info_buf, 10, 42)
    var q: i64 = 0
    while q < 42 {
        if rawbuf_get(&okm, q) != (expected_okm[q as usize] as i64) {
            print_int(2)
            return 1
        }
        q = q + 1
    }

    println("hkdf_rfc5869_vectors: HKDF-Extract and HKDF-Expand match RFC 5869 Appendix A.1")
    return 0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/hkdf_rfc5869_vectors.sio`
Expected: unresolved-name error referencing `hkdf_extract`/`hkdf_expand`/`HKDF_HASH_SHA256`.

- [ ] **Step 3: Implement `hkdf_extract`/`hkdf_expand` — append to `stdlib/crypto/hkdf.sio`**

```sio
pub const HKDF_HASH_SHA256: i32 = 0
pub const HKDF_HASH_SHA384: i32 = 1

fn hkdf_hash_len(hash_algo: i32) -> i64 {
    if hash_algo == HKDF_HASH_SHA256 { 32 } else { 48 }
}

// Dispatches to hmac_sha256/hmac_sha384, copying the fixed-size result
// into a RawBuf (never `var x = *ref` -- element-by-element copy).
fn hkdf_hmac(hash_algo: i32, key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> RawBuf with IO {
    // <fill in: if hash_algo==HKDF_HASH_SHA256, call hmac_sha256 and copy
    // its [u8;32] result into a fresh RawBuf; else call hmac_sha384 and
    // copy its [u8;48] result>
}

pub fn hkdf_extract(hash_algo: i32, salt: &RawBuf, salt_len: i64, ikm: &RawBuf, ikm_len: i64) -> RawBuf with IO {
    // RFC 5869 2.2: PRK = HMAC-Hash(salt, IKM). Per the RFC, if salt is
    // not provided, it is set to a string of HashLen zeros -- this
    // plan's callers (Task 3) always provide an explicit salt (even if
    // it's a zero-filled buffer), so this function does not need to
    // synthesize a default salt itself; document this assumption.
    hkdf_hmac(hash_algo, salt, salt_len, ikm, ikm_len)
}

pub fn hkdf_expand(hash_algo: i32, prk: &RawBuf, prk_len: i64, info: &RawBuf, info_len: i64, out_len: i64) -> RawBuf with IO {
    // <fill in: RFC 5869 2.3's iterative construction: T(0) = empty,
    // T(i) = HMAC-Hash(PRK, T(i-1) | info | i) for i = 1..ceil(out_len/HashLen),
    // OKM = T(1) | T(2) | ... truncated to out_len bytes. Build each
    // iteration's input buffer explicitly (previous T-block ++ info ++
    // single byte holding the iteration counter i, per the RFC -- note i
    // is a ONE-BYTE counter, values 1..255, matching the RFC's own stated
    // limit of out_len <= 255*HashLen).>
}
```

- [ ] **Step 4: Fill in `hkdf_hmac`/`hkdf_expand`'s bodies, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/hkdf_rfc5869_vectors.sio`
Expected: prints `hkdf_rfc5869_vectors: HKDF-Extract and HKDF-Expand match RFC 5869 Appendix A.1`, exit 0.

- [ ] **Step 5: Run the test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix hkdf_ --jobs 2 --verbose`
Expected: 1 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add stdlib/crypto/hkdf.sio tests/run-pass/hkdf_rfc5869_vectors.sio
git commit -m "feat(crypto): add HKDF-Extract/Expand (RFC 5869)"
```

---

## Task 3: `stdlib/crypto/hkdf.sio` (part 3) — TLS 1.3 key schedule, verified against RFC 8448

**Files:**
- Modify: `stdlib/crypto/hkdf.sio` (append to the file Tasks 1-2 created)
- Test: `tests/run-pass/tls13_keyschedule_rfc8448.sio`

**Interfaces:**
- Consumes: `hkdf_extract`, `hkdf_expand`, `HKDF_HASH_SHA256` (Task 2, same file).
- Produces (public — this is the interface the not-yet-scoped handshake sub-project calls): `pub fn hkdf_expand_label(hash_algo: i32, secret: &RawBuf, secret_len: i64, label: &RawBuf, label_len: i64, context: &RawBuf, context_len: i64, out_len: i64) -> RawBuf with IO`, `pub fn derive_secret(hash_algo: i32, secret: &RawBuf, secret_len: i64, label: &RawBuf, label_len: i64, transcript_hash: &RawBuf, transcript_hash_len: i64) -> RawBuf with IO`, `pub fn tls13_early_secret(hash_algo: i32) -> RawBuf with IO`, `pub fn tls13_handshake_secret(hash_algo: i32, early_secret: &RawBuf, early_secret_len: i64, ecdhe_shared_secret: &RawBuf, ecdhe_len: i64) -> RawBuf with IO`, `pub fn tls13_master_secret(hash_algo: i32, handshake_secret: &RawBuf, handshake_secret_len: i64) -> RawBuf with IO`, `pub fn tls13_traffic_key_and_iv(hash_algo: i32, traffic_secret: &RawBuf, traffic_secret_len: i64, key_len: i64, iv_len: i64) -> (RawBuf, RawBuf) with IO`.

- [ ] **Step 1: Fetch and read RFC 8448's actual text before writing anything**

RFC 8448 ("Example Handshake Traces for TLS 1.3") is long (it's a full worked handshake trace). Fetch the actual RFC text (e.g. `https://www.rfc-editor.org/rfc/rfc8448.txt`) and locate the section using X25519 + AES-128-GCM-SHA256 (the RFC's "Simple 1-RTT Handshake" trace is the one matching this branch's own crypto stack — confirm this by checking the trace's stated cipher suite/group before extracting any values, don't assume the first trace in the document is the right one). Extract, and independently verify against the RFC text directly (do not trust any secondhand summary of RFC 8448, including anything in this plan or its spec):
- The ClientHello's `x25519` private key and the resulting shared secret (or the client's and server's key_share public values plus enough to derive the shared secret independently, if the RFC states the shared secret directly — check which the RFC actually publishes).
- The published Early Secret, the `derived` intermediate, and the Handshake Secret (all as raw hex).
- The published transcript hash through ServerHello (a SHA-256 hash the RFC states explicitly — you do NOT need to compute this from message bytes, just use the RFC's own published hash value directly, per this sub-project's explicit scope: transcript hashing itself is out of scope).
- The published `client_handshake_traffic_secret` and `server_handshake_traffic_secret`.
- The published `derived` step and Master Secret.
- The published transcript hash through the server's Finished message.
- The published `client_application_traffic_secret_0` and `server_application_traffic_secret_0`.
- The published derived key/IV for at least one traffic secret (to test `tls13_traffic_key_and_iv` against a real value, not just structurally).

- [ ] **Step 2: Write the failing test using the real, independently-verified RFC 8448 values**

```sio
//@ run-pass
// RFC 8448's full TLS 1.3 handshake trace (X25519 + AES-128-GCM-SHA256).
// Every value below MUST be independently re-verified against the actual
// RFC 8448 text (fetched fresh) before this test is trusted -- this is
// the single highest-risk transcription in this whole plan given RFC
// 8448's length and density.
use crypto::hkdf::*
use net::socket::*

fn main() -> i64 with IO {
    // <fill in from Step 1's verified extraction:>
    // let ecdhe_shared_secret: [u8;32] = [ ... ]
    // let expected_handshake_secret: [u8;32] = [ ... ]
    // let transcript_hash_through_server_hello: [u8;32] = [ ... ]
    // let expected_client_hs_traffic_secret: [u8;32] = [ ... ]
    // let expected_server_hs_traffic_secret: [u8;32] = [ ... ]
    // let expected_master_secret: [u8;32] = [ ... ]
    // let transcript_hash_through_server_finished: [u8;32] = [ ... ]
    // let expected_client_ap_traffic_secret: [u8;32] = [ ... ]
    // let expected_server_ap_traffic_secret: [u8;32] = [ ... ]
    // let expected_some_traffic_key: [u8;16] = [ ... ]  // AES-128-GCM key length
    // let expected_some_traffic_iv: [u8;12] = [ ... ]   // AES-128-GCM IV length

    let early_secret = tls13_early_secret(HKDF_HASH_SHA256)

    let ecdhe_buf = rawbuf_new(32 + 8)
    // <copy ecdhe_shared_secret into ecdhe_buf>

    let handshake_secret = tls13_handshake_secret(HKDF_HASH_SHA256, &early_secret, 32, &ecdhe_buf, 32)
    // <assert handshake_secret's 32 bytes == expected_handshake_secret>

    let sh_hash_buf = rawbuf_new(32 + 8)
    // <copy transcript_hash_through_server_hello into sh_hash_buf>

    let c_hs_label_str = "c hs traffic"
    // <build c_hs_label_buf from c_hs_label_str via str_char_at, length 12>
    // let client_hs_traffic_secret = derive_secret(HKDF_HASH_SHA256, &handshake_secret, 32, &c_hs_label_buf, 12, &sh_hash_buf, 32)
    // <assert against expected_client_hs_traffic_secret>

    // <same pattern for "s hs traffic" -> server_hs_traffic_secret>

    let master_secret = tls13_master_secret(HKDF_HASH_SHA256, &handshake_secret, 32)
    // <assert against expected_master_secret>

    let sf_hash_buf = rawbuf_new(32 + 8)
    // <copy transcript_hash_through_server_finished into sf_hash_buf>

    // <"c ap traffic" -> client_ap_traffic_secret, assert>
    // <"s ap traffic" -> server_ap_traffic_secret, assert>

    // <tls13_traffic_key_and_iv on one of the traffic secrets with
    // key_len=16, iv_len=12 (AES-128-GCM's sizes), assert against
    // expected_some_traffic_key/expected_some_traffic_iv>

    println("tls13_keyschedule_rfc8448: full key schedule matches RFC 8448's published handshake trace")
    return 0
}
```

Fill in every placeholder above with real code and real RFC 8448 values before running — this is a full worked example, so every intermediate assertion should be a real, independently-verified check, not left as a comment.

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/tls13_keyschedule_rfc8448.sio`
Expected: unresolved-name error referencing `hkdf_expand_label`/`derive_secret`/`tls13_early_secret`/etc.

- [ ] **Step 4: Implement `hkdf_expand_label`, `derive_secret`, and the secret ladder — append to `stdlib/crypto/hkdf.sio`**

```sio
// RFC 8446 section 7.1:
//   struct { uint16 length; opaque label<7..255> = "tls13 " + Label;
//             opaque context<0..255> = Context; } HkdfLabel;
pub fn hkdf_expand_label(
    hash_algo: i32,
    secret: &RawBuf,
    secret_len: i64,
    label: &RawBuf,
    label_len: i64,
    context: &RawBuf,
    context_len: i64,
    out_len: i64,
) -> RawBuf with IO {
    // <fill in: build the HkdfLabel wire structure --
    // 2-byte big-endian out_len, 1-byte length-prefix + "tls13 " (6
    // ASCII bytes) + label, 1-byte length-prefix + context -- into a
    // fresh RawBuf, then call hkdf_expand(hash_algo, secret, secret_len,
    // that_buf, its_length, out_len)>
}

pub fn derive_secret(
    hash_algo: i32,
    secret: &RawBuf,
    secret_len: i64,
    label: &RawBuf,
    label_len: i64,
    transcript_hash: &RawBuf,
    transcript_hash_len: i64,
) -> RawBuf with IO {
    hkdf_expand_label(hash_algo, secret, secret_len, label, label_len, transcript_hash, transcript_hash_len, hkdf_hash_len(hash_algo))
}

pub fn tls13_early_secret(hash_algo: i32) -> RawBuf with IO {
    let hash_len = hkdf_hash_len(hash_algo)
    let zero_salt = rawbuf_new(hash_len + 8)   // all-zero by construction (rawbuf_new does not guarantee zero-init -- VERIFY this against rawbuf_new's actual implementation before relying on it; if it doesn't zero-init, explicitly zero every byte)
    let zero_ikm = rawbuf_new(hash_len + 8)
    hkdf_extract(hash_algo, &zero_salt, hash_len, &zero_ikm, hash_len)
}

fn empty_string_hash(hash_algo: i32) -> RawBuf with IO {
    // <fill in: Hash("") -- the hash of a zero-length input. Compute this
    // directly (call sha256/sha384 with len=0 on an empty RawBuf) rather
    // than hard-coding the well-known constant value, so this stays
    // correct if the hash choice differs -- but INDEPENDENTLY VERIFY the
    // computed value matches the well-known SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (and the
    // analogous SHA-384("") value) as a sanity check during implementation.>
}

pub fn tls13_handshake_secret(hash_algo: i32, early_secret: &RawBuf, early_secret_len: i64, ecdhe_shared_secret: &RawBuf, ecdhe_len: i64) -> RawBuf with IO {
    let empty_hash = empty_string_hash(hash_algo)
    let derived_label_str = "derived"
    // <build derived_label_buf from derived_label_str, length 7>
    // let derived = derive_secret(hash_algo, early_secret, early_secret_len, &derived_label_buf, 7, &empty_hash, hkdf_hash_len(hash_algo))
    // hkdf_extract(hash_algo, &derived, hkdf_hash_len(hash_algo), ecdhe_shared_secret, ecdhe_len)
}

pub fn tls13_master_secret(hash_algo: i32, handshake_secret: &RawBuf, handshake_secret_len: i64) -> RawBuf with IO {
    let empty_hash = empty_string_hash(hash_algo)
    let derived_label_str = "derived"
    // <build derived_label_buf, same as above>
    // let derived = derive_secret(hash_algo, handshake_secret, handshake_secret_len, &derived_label_buf, 7, &empty_hash, hkdf_hash_len(hash_algo))
    // let zero_ikm = rawbuf_new(hkdf_hash_len(hash_algo) + 8)  // per RFC 8446's diagram: this stage's IKM is 0, NOT the previous secret -- INDEPENDENTLY VERIFY this against the RFC's own key-schedule diagram before trusting it, since it's easy to mistakenly reuse handshake_secret here instead of zero
    // hkdf_extract(hash_algo, &derived, hkdf_hash_len(hash_algo), &zero_ikm, hkdf_hash_len(hash_algo))
}

pub fn tls13_traffic_key_and_iv(hash_algo: i32, traffic_secret: &RawBuf, traffic_secret_len: i64, key_len: i64, iv_len: i64) -> (RawBuf, RawBuf) with IO {
    let key_label_str = "key"
    let iv_label_str = "iv"
    // <build key_label_buf (length 3), iv_label_buf (length 2)>
    // let empty_context = rawbuf_new(8)
    // let key = hkdf_expand_label(hash_algo, traffic_secret, traffic_secret_len, &key_label_buf, 3, &empty_context, 0, key_len)
    // let iv = hkdf_expand_label(hash_algo, traffic_secret, traffic_secret_len, &iv_label_buf, 2, &empty_context, 0, iv_len)
    // (key, iv)
}
```

Before trusting `rawbuf_new`'s assumed zero-initialization (used above for `zero_salt`/`zero_ikm`/`empty_context`), verify this directly: read `rawbuf_new`'s actual implementation in `stdlib/net/socket.sio` (or wherever it's defined) to confirm it zero-fills, or write a tiny throwaway probe (`let b = rawbuf_new(8); print_int(rawbuf_get(&b, 0))`) and check the output — do not assume, per this project's "measure, don't assume" discipline.

- [ ] **Step 5: Fill in every remaining `<fill in>` block, fill in the test's real RFC 8448 values, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/tls13_keyschedule_rfc8448.sio`
Expected: prints `tls13_keyschedule_rfc8448: full key schedule matches RFC 8448's published handshake trace`, exit 0.

- [ ] **Step 6: Run all three tasks' tests together**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix hmac_ --jobs 2 --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix hkdf_ --jobs 2 --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix tls13_keyschedule_ --jobs 2 --verbose`
Expected: every one reports all passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add stdlib/crypto/hkdf.sio tests/run-pass/tls13_keyschedule_rfc8448.sio
git commit -m "feat(crypto): add TLS 1.3 key schedule (HKDF-Expand-Label, Derive-Secret, secret ladder), verified against RFC 8448"
```

---

## After This Plan

`crypto::hkdf::derive_secret`/`tls13_early_secret`/`tls13_handshake_secret`/`tls13_master_secret`/`tls13_traffic_key_and_iv` are what the not-yet-scoped TLS 1.3 handshake state machine + record layer sub-project's own spec should reference directly — that sub-project also needs transcript-hash accumulation and the full handshake message flow (ClientHello through Finished), neither of which this plan covers, and its own brainstorming/spec process starts fresh once this plan's final whole-branch review (per `superpowers:subagent-driven-development`) is clean.

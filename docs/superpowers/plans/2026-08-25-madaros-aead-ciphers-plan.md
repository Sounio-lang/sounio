<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-25-madaros-aead-ciphers-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-25-madaros-aead-ciphers-plan
-->

# Madaros AEAD Ciphers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** implement AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305 behind one unified `aead_seal`/`aead_open` interface, in pure Sounio, verified against published cryptographic test vectors — the symmetric-encryption layer a future TLS handshake sub-project will call.

**Architecture:** five new modules under `stdlib/crypto/` (`aes.sio`, `gcm.sio`, `chacha20.sio`, `poly1305.sio`, `aead.sio`), built bottom-up — each cipher primitive first, verified standalone against a published test vector, then combined into the two AEAD constructions, then unified behind one dispatch interface.

**Tech Stack:** Sounio/Madaros (`./bin/souc`), no non-Sounio code anywhere in this plan's deliverables.

**Spec:** `docs/superpowers/specs/2026-08-25-madaros-aead-ciphers-design.md`

## Global Constraints

- **RawBuf-based, not fixed-array-based**, deliberately diverging from `stdlib/crypto/sha256.sio`'s older `[u8;256]`-array convention — TLS records carry up to 16384 bytes, far past a 256-byte fixed array. `RawBuf`/`rawbuf_new`/`rawbuf_get`/`rawbuf_set` come from `net::socket::*`.
- **Never use native `u32` arithmetic anywhere in this plan.** `stdlib/hash/word32.sio` already exists with exactly the primitives needed for every 32-bit-word operation this plan touches: `add32(a: i64, b: i64) -> i64`, `rotl32(x: i64, n: i64) -> i64`, `shr32`, `xor32`, `and32`, `or32`, `not32` — all operating on `i64` values masked to `0..4294967295`. Its header comment documents "Finding 13": native `u32` `+`/`-` do not wrap mod 2^32 on Madaros at all — a real, confirmed compiler defect, not a style preference. Every task touching a 32-bit word (ChaCha20's state/counter, AES's key-schedule word rotation/XOR, GCM's counter-block increment) reuses `word32.sio` via `use hash::word32::*` rather than reimplementing 32-bit arithmetic or using the native `u32` type. `rotr32` exists in the same module but is not needed by this plan.
- **GCM is encrypt-direction-only.** `aes.sio` implements ONLY `SubBytes`/`ShiftRows`/`MixColumns`/`AddRoundKey` and the forward key schedule — never `InvSubBytes`/`InvShiftRows`/`InvMixColumns`. GCM's counter-mode construction only ever calls AES in the encrypt direction, for both encryption and decryption.
- **12-byte nonces only** — TLS's own record-layer nonce construction always produces exactly 12 bytes for both AES-GCM and ChaCha20-Poly1305.
- **Nonce management is the caller's responsibility** — this plan does not track, validate, or reject nonce reuse.
- **Constant-time tag comparison.** Every `_open` function accumulates a bitwise difference across all 16 tag bytes and compares the accumulator to zero exactly once, at the end — never a short-circuiting per-byte `if` comparison. This is a security requirement (timing side-channel), not a style preference.
- **Fail-closed, zeroed-on-failure.** Every `_open` function returns `(RawBuf, bool)`; when the `bool` is `false`, the returned `RawBuf` is explicitly zeroed before returning — never left as partially-decrypted or garbage data a careless caller might read anyway.
- **Sentinel `bool`/`i64` error convention** — never `Result<T,E>`/`Option<T>`.
- **Every test's expected value comes from a published, independent source — never self-generated.** This plan cites specific real test vectors below; every task must independently re-verify its cited vector against a fresh, authoritative source (the actual RFC/FIPS text, fetched or looked up directly) before trusting it in committed test code, per this project's "measure, don't assume" discipline — doubly important here, since a self-consistent transcription error in a hex string would make a wrong implementation look correct.
- Module imports: bare `use <filename>::{name|*}`, no `module` declaration.
- Commit convention: Conventional Commits (`feat(crypto): ...`). **Never add an AI-attribution line to any commit.**
- Test invocation: `bash scripts/run_sio_test_suite.sh --filter-prefix aes_` / `gcm_` / `chacha20_` / `poly1305_` / `aead_` as appropriate — never the whole-repo suite for this plan's checkpoints. Use `--jobs 2` when running multiple test files together (this exact branch has repeatedly found high parallelism produces spurious 30-second timeouts on this test runner).
- No Rust, Python, or any non-Sounio code in this plan's shipped deliverables (Python/openssl-equivalent tooling used only to independently cross-check a test vector during development is fine — the shipped `.sio` files must be pure Sounio).
- Run every shell command as a plain foreground command — never a background/Monitor-style wait mechanism.

---

## Task 1: `stdlib/crypto/aes.sio` — AES-128/256 key schedule and single-block encryption

**Files:**
- Create: `stdlib/crypto/aes.sio`
- Test: `tests/run-pass/aes_block_encrypt.sio`

**Interfaces:**
- Produces: `pub const AES_128_ROUNDS: i64 = 10`, `pub const AES_256_ROUNDS: i64 = 14`, `pub struct AesRoundKeys { keys: [[u8;16];15], rounds: i64 }`, `pub fn aes128_key_schedule(key: &[u8;16]) -> AesRoundKeys`, `pub fn aes256_key_schedule(key: &[u8;32]) -> AesRoundKeys`, `pub fn aes_encrypt_block(round_keys: &AesRoundKeys, block: &[u8;16]) -> [u8;16]`. Task 2 imports and calls these exact names/signatures.

AES's S-box substitution and `MixColumns` step work over `GF(2^8)` (8-bit values) — this is NOT covered by `word32.sio` (which only handles 32-bit words), so this task needs its own from-scratch 8-bit bit manipulation. Before writing that code, verify the real bit-shift operand-type convention on this compiler, since two docs in this repo disagree (`docs/guide/SOUNIO_GOTCHAS.md` says shift amounts must be `u8`; `CLAUDE.md` §7 says this is stale as of 2026-08-20 and a bare `i32` shift amount works).

- [ ] **Step 1: Verify the real bit-shift operand-type convention before writing any GF(2^8) code**

```sio
// /tmp/shift_probe.sio
fn main() -> i64 with IO {
    let x: i32 = 200
    let a = x >> 4          // bare i32 shift amount
    let b = x >> (4 as u8)  // explicit u8 shift amount
    print_int(a)
    print(" ")
    print_int(b)
    return 0
}
```

Run: `./bin/souc run /tmp/shift_probe.sio`
Expected: both forms compile and print the same value (`12 12`). Record which form(s) actually work in a comment at the top of `aes.sio`, and use whichever form is confirmed working for every raw shift this task writes.

- [ ] **Step 2: Write the failing test using a real FIPS-197 vector**

```sio
//@ run-pass
use crypto::aes::*

fn main() -> i64 with IO {
    // FIPS-197 Appendix B / C.1 (AES-128): the canonical worked example.
    // INDEPENDENTLY RE-VERIFY these three hex strings against the actual
    // FIPS-197 PDF (or another primary source) before trusting this
    // transcription -- do not skip this because the numbers "look right".
    let key128: [u8;16] = [
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f
    ]
    let plaintext: [u8;16] = [
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
    ]
    let expected_ciphertext_128: [u8;16] = [
        0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
        0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a
    ]

    let rk128 = aes128_key_schedule(&key128)
    let ct128 = aes_encrypt_block(&rk128, &plaintext)
    var i: i64 = 0
    while i < 16 {
        if ct128[i as usize] != expected_ciphertext_128[i as usize] {
            print_int(1)
            return 1
        }
        i = i + 1
    }

    // FIPS-197 Appendix C.3 (AES-256), SAME plaintext as above, per FIPS-197's
    // own convention of reusing one plaintext across all three C.1/C.2/C.3
    // examples. INDEPENDENTLY RE-VERIFY this key and ciphertext too.
    let key256: [u8;32] = [
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
        0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
    ]
    let expected_ciphertext_256: [u8;16] = [
        0x8e,0xa2,0xb7,0xca,0x51,0x67,0x45,0xbf,
        0xea,0xfc,0x49,0x90,0x4b,0x49,0x60,0x89
    ]

    let rk256 = aes256_key_schedule(&key256)
    let ct256 = aes_encrypt_block(&rk256, &plaintext)
    var j: i64 = 0
    while j < 16 {
        if ct256[j as usize] != expected_ciphertext_256[j as usize] {
            print_int(2)
            return 1
        }
        j = j + 1
    }

    println("aes_block_encrypt: AES-128 and AES-256 match FIPS-197's worked examples")
    return 0
}
```

- [ ] **Step 2b: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/aes_block_encrypt.sio`
Expected: unresolved-import error referencing `crypto::aes`.

- [ ] **Step 3: Implement `aes.sio`**

Implement the standard AES forward cipher: S-box lookup table (the fixed 256-entry FIPS-197 Rijndael S-box — copy it from the FIPS-197 spec, verify byte-for-byte against a primary source, don't hand-derive it), `SubBytes` (per-byte S-box substitution), `ShiftRows` (row-wise cyclic left shift by 0/1/2/3 bytes), `MixColumns` (GF(2^8) matrix multiplication using the standard `{02}`/`{03}` multiplication-by-constant, implemented via the standard "xtime" conditional-XOR-with-0x1b construction — this is genuinely new 8-bit bit manipulation, use whichever shift-operand form Step 1 confirmed), `AddRoundKey` (XOR with the round key), and the key schedule (`RotWord`/`SubWord`/`Rcon`-driven expansion, 10 rounds for AES-128 producing 11 round keys into `keys[0..11]`, 14 rounds for AES-256 producing 15 round keys into `keys[0..15]`, with `rounds` set to `AES_128_ROUNDS`/`AES_256_ROUNDS` respectively so `aes_encrypt_block` knows how many of the 15 slots to use).

```sio
pub const AES_128_ROUNDS: i64 = 10
pub const AES_256_ROUNDS: i64 = 14

pub struct AesRoundKeys {
    keys: [[u8;16];15],
    rounds: i64,
}

// <fill in: S-box table, InvSbox not needed, RotWord/SubWord/Rcon,
// aes128_key_schedule, aes256_key_schedule, SubBytes/ShiftRows/MixColumns/
// AddRoundKey, aes_encrypt_block driving AddRoundKey(round0) then
// (rounds-1) full rounds then a final round without MixColumns>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/aes_block_encrypt.sio`
Expected: prints `aes_block_encrypt: AES-128 and AES-256 match FIPS-197's worked examples`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/crypto/aes.sio tests/run-pass/aes_block_encrypt.sio
git commit -m "feat(crypto): add AES-128/256 key schedule and single-block encryption"
```

---

## Task 2: `stdlib/crypto/gcm.sio` — Galois/Counter Mode

**Files:**
- Create: `stdlib/crypto/gcm.sio`
- Test: `tests/run-pass/gcm_seal_open.sio`

**Interfaces:**
- Consumes: `AesRoundKeys`, `aes_encrypt_block` (Task 1).
- Produces: `pub const GCM_TAG_LEN: i64 = 16`, `pub const GCM_NONCE_LEN: i64 = 12`, `pub fn gcm_seal(round_keys: &AesRoundKeys, nonce: &[u8;12], plaintext: &RawBuf, plaintext_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, i64) with IO`, `pub fn gcm_open(round_keys: &AesRoundKeys, nonce: &[u8;12], ciphertext_with_tag: &RawBuf, ciphertext_with_tag_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, bool) with IO`. Task 5 imports and calls these exact names/signatures.

- [ ] **Step 1: Write the failing test using real NIST GCM test vectors**

NIST-published GCM test vectors (originally from McGrew & Viega's "The Galois/Counter Mode of Operation" spec, widely mirrored in NIST's own CAVP GCM test vector files) include an all-zero AES-128 case. **INDEPENDENTLY RE-VERIFY every hex string below against a primary NIST/RFC source before trusting this transcription** — this is the single highest-risk transcription in this whole plan, since GCM vectors are long hex strings with no self-checking structure.

```sio
//@ run-pass
use crypto::aes::*
use crypto::gcm::*
use net::socket::*

fn main() -> i64 with IO {
    // GCM Test Case 2 (McGrew & Viega): AES-128, all-zero key, one all-zero
    // 16-byte plaintext block, empty AAD, the standard 12-byte all-zero-
    // except-final-0x01... NO: this vector's IV is the 12-byte all-zero
    // value with no trailing 1 -- VERIFY THE EXACT IV BYTES against the
    // primary source, do not assume.
    let key: [u8;16] = [0;16]
    let plaintext_bytes: [u8;16] = [0;16]
    let nonce: [u8;12] = [0;12]
    let expected_ciphertext: [u8;16] = [
        0x03,0x88,0xda,0xce,0x60,0xb6,0xa3,0x92,
        0xf3,0x28,0xc2,0xb9,0x71,0xb2,0xfe,0x78
    ]
    let expected_tag: [u8;16] = [
        0xab,0x6e,0x47,0xd4,0x2c,0xec,0x13,0xbd,
        0xf5,0x3a,0x67,0xb2,0x12,0x57,0xbd,0xdf
    ]

    let round_keys = aes128_key_schedule(&key)

    let pt_buf = rawbuf_new(16 + 8)
    var i: i64 = 0
    while i < 16 { rawbuf_set(&pt_buf, i, plaintext_bytes[i as usize] as i64); i = i + 1 }
    let empty_aad = rawbuf_new(1)

    let (sealed, sealed_len) = gcm_seal(&round_keys, &nonce, &pt_buf, 16, &empty_aad, 0)
    if sealed_len != 32 {   // 16 ciphertext + 16 tag
        print_int(1)
        return 1
    }
    var j: i64 = 0
    while j < 16 {
        if rawbuf_get(&sealed, j) != (expected_ciphertext[j as usize] as i64) {
            print_int(2)
            return 1
        }
        j = j + 1
    }
    var k: i64 = 0
    while k < 16 {
        if rawbuf_get(&sealed, 16 + k) != (expected_tag[k as usize] as i64) {
            print_int(3)
            return 1
        }
        k = k + 1
    }

    // Round-trip: gcm_open on our own sealed output must recover the
    // original plaintext and report ok=true.
    let (opened, ok) = gcm_open(&round_keys, &nonce, &sealed, sealed_len, &empty_aad, 0)
    if !ok {
        print_int(4)
        return 1
    }
    var m: i64 = 0
    while m < 16 {
        if rawbuf_get(&opened, m) != (plaintext_bytes[m as usize] as i64) {
            print_int(5)
            return 1
        }
        m = m + 1
    }

    // Adversarial: flip one tag byte, confirm gcm_open reports ok=false
    // and the returned plaintext is all-zero (fail-closed, per spec).
    rawbuf_set(&sealed, 16, (rawbuf_get(&sealed, 16) ^ 0xFF))
    let (opened_bad, ok_bad) = gcm_open(&round_keys, &nonce, &sealed, sealed_len, &empty_aad, 0)
    if ok_bad {
        print_int(6)
        return 1
    }
    var n: i64 = 0
    while n < 16 {
        if rawbuf_get(&opened_bad, n) != 0 {
            print_int(7)
            return 1
        }
        n = n + 1
    }

    println("gcm_seal_open: AES-128-GCM matches published test vector, round-trips, and fails closed on a tampered tag")
    return 0
}
```

Also add a second AES-256-GCM case to this same test (or a second test file `tests/run-pass/gcm_seal_open_256.sio` if the first grows too large) using a real NIST AES-256-GCM vector with non-trivial plaintext and non-empty AAD — look up a specific published NIST CAVP AES-256-GCM test case (key, IV, plaintext, AAD, expected ciphertext, expected tag) and independently verify it before writing the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/gcm_seal_open.sio`
Expected: unresolved-import error referencing `crypto::gcm`.

- [ ] **Step 3: Implement `gcm.sio`**

```sio
use hash::word32::*
use net::socket::*
use crypto::aes::*

pub const GCM_TAG_LEN: i64 = 16
pub const GCM_NONCE_LEN: i64 = 12

// GHASH: multiplication in GF(2^128) with the reduction polynomial
// x^128 + x^7 + x^2 + x + 1, per NIST SP 800-38D section 6.3. Standard
// bit-at-a-time shift-and-conditionally-XOR-with-0xE1000000...0 algorithm.
fn ghash_multiply(x: &[u8;16], h: &[u8;16]) -> [u8;16] {
    // <fill in: the standard 128-bit carryless multiplication + reduction>
}

fn ghash(h: &[u8;16], aad: &RawBuf, aad_len: i64, ciphertext: &RawBuf, ciphertext_len: i64) -> [u8;16] with IO {
    // <fill in: fold aad (zero-padded to 16-byte boundary), fold
    // ciphertext (zero-padded to 16-byte boundary), fold the final 16-byte
    // block encoding len(aad) and len(ciphertext) each as 64-bit BIG-ENDIAN
    // BIT counts (not byte counts -- multiply each length by 8 first),
    // multiplying by h via ghash_multiply after each fold, per NIST
    // SP 800-38D section 6.4>
}

// Increments only the low 32 bits of a 16-byte counter block (the high 96
// bits, which hold the nonce in GCM's 12-byte-nonce construction, never
// change during a single seal/open call) -- uses word32.sio's add32, never
// native u32, per this plan's Global Constraints.
fn gcm_increment_counter(block: &[u8;16]) -> [u8;16] {
    let low32 = ((block[12] as i64) << 24) | ((block[13] as i64) << 16) | ((block[14] as i64) << 8) | (block[15] as i64)
    let incremented = add32(low32, 1)
    var result = *block
    result[12] = ((incremented >> 24) & 255) as u8
    result[13] = ((incremented >> 16) & 255) as u8
    result[14] = ((incremented >> 8) & 255) as u8
    result[15] = (incremented & 255) as u8
    result
}

fn gcm_ctr_crypt(round_keys: &AesRoundKeys, j0: &[u8;16], data: &RawBuf, data_len: i64) -> RawBuf with IO {
    // <fill in: XOR data with the keystream generated by aes_encrypt_block
    // on j0+1, j0+2, ... (gcm_increment_counter chained), one 16-byte
    // block at a time, handling a final partial block correctly>
}

pub fn gcm_seal(
    round_keys: &AesRoundKeys,
    nonce: &[u8;12],
    plaintext: &RawBuf,
    plaintext_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, i64) with IO {
    let zero_block: [u8;16] = [0;16]
    let h = aes_encrypt_block(round_keys, &zero_block)

    var j0: [u8;16] = [0;16]
    var i: i64 = 0
    while i < 12 { j0[i as usize] = nonce[i as usize]; i = i + 1 }
    j0[15] = 1   // 12-byte-nonce construction: J0 = nonce || 0^31 || 1

    let ciphertext = gcm_ctr_crypt(round_keys, &j0, plaintext, plaintext_len)
    let hash = ghash(&h, aad, aad_len, &ciphertext, plaintext_len)
    let ek_j0 = aes_encrypt_block(round_keys, &j0)

    var tag: [u8;16] = [0;16]
    var t: i64 = 0
    while t < 16 { tag[t as usize] = hash[t as usize] ^ ek_j0[t as usize]; t = t + 1 }

    let out = rawbuf_new(plaintext_len + GCM_TAG_LEN + 8)
    var c: i64 = 0
    while c < plaintext_len { rawbuf_set(&out, c, rawbuf_get(&ciphertext, c)); c = c + 1 }
    var g: i64 = 0
    while g < 16 { rawbuf_set(&out, plaintext_len + g, tag[g as usize] as i64); g = g + 1 }

    (out, plaintext_len + GCM_TAG_LEN)
}

pub fn gcm_open(
    round_keys: &AesRoundKeys,
    nonce: &[u8;12],
    ciphertext_with_tag: &RawBuf,
    ciphertext_with_tag_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, bool) with IO {
    if ciphertext_with_tag_len < GCM_TAG_LEN {
        return (rawbuf_new(1), false)
    }
    let plaintext_len = ciphertext_with_tag_len - GCM_TAG_LEN

    let zero_block: [u8;16] = [0;16]
    let h = aes_encrypt_block(round_keys, &zero_block)
    var j0: [u8;16] = [0;16]
    var i: i64 = 0
    while i < 12 { j0[i as usize] = nonce[i as usize]; i = i + 1 }
    j0[15] = 1

    let ciphertext_only = rawbuf_new(plaintext_len + 8)
    var c: i64 = 0
    while c < plaintext_len { rawbuf_set(&ciphertext_only, c, rawbuf_get(ciphertext_with_tag, c)); c = c + 1 }

    let hash = ghash(&h, aad, aad_len, &ciphertext_only, plaintext_len)
    let ek_j0 = aes_encrypt_block(round_keys, &j0)
    var expected_tag: [u8;16] = [0;16]
    var t: i64 = 0
    while t < 16 { expected_tag[t as usize] = hash[t as usize] ^ ek_j0[t as usize]; t = t + 1 }

    // Constant-time comparison, per this plan's Global Constraints: OR all
    // byte differences together, branch on the accumulator exactly once.
    var diff: i64 = 0
    var d: i64 = 0
    while d < 16 {
        let received = rawbuf_get(ciphertext_with_tag, plaintext_len + d) as u8
        diff = diff | ((received ^ expected_tag[d as usize]) as i64)
        d = d + 1
    }

    let plaintext = gcm_ctr_crypt(round_keys, &j0, &ciphertext_only, plaintext_len)
    if diff != 0 {
        let zeroed = rawbuf_new(plaintext_len + 8)
        var z: i64 = 0
        while z < plaintext_len { rawbuf_set(&zeroed, z, 0); z = z + 1 }
        return (zeroed, false)
    }
    (plaintext, true)
}
```

- [ ] **Step 4: Fill in `ghash_multiply`, `ghash`, and `gcm_ctr_crypt`, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/gcm_seal_open.sio`
Expected: prints `gcm_seal_open: AES-128-GCM matches published test vector, round-trips, and fails closed on a tampered tag`, exit 0.

- [ ] **Step 5: Run the AES-256-GCM test too**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix gcm_ --jobs 2 --verbose`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add stdlib/crypto/gcm.sio tests/run-pass/gcm_seal_open.sio tests/run-pass/gcm_seal_open_256.sio
git commit -m "feat(crypto): add AES-GCM (128/256) authenticated encryption"
```

---

## Task 3: `stdlib/crypto/chacha20.sio` — ChaCha20 stream cipher

**Files:**
- Create: `stdlib/crypto/chacha20.sio`
- Test: `tests/run-pass/chacha20_block.sio`

**Interfaces:**
- Consumes: `add32`, `xor32`, `rotl32` (from `hash::word32`, already exists).
- Produces: `pub const CHACHA20_KEY_LEN: i64 = 32`, `pub const CHACHA20_NONCE_LEN: i64 = 12`, `pub fn chacha20_block(key: &[u8;32], counter: i64, nonce: &[u8;12]) -> [u8;64]`, `pub fn chacha20_encrypt(key: &[u8;32], counter: i64, nonce: &[u8;12], data: &RawBuf, data_len: i64) -> RawBuf with IO`. Task 4's sibling module and Task 5 both import and call these exact names/signatures. `counter` is `i64`, word32-bounded (`0..4294967295`), never native `u32`, per this plan's Global Constraints.

- [ ] **Step 1: Write the failing test using RFC 8439's published block-function vector**

```sio
//@ run-pass
use crypto::chacha20::*

fn main() -> i64 with IO {
    // RFC 8439 section 2.3.2's worked ChaCha20 block-function example.
    // INDEPENDENTLY RE-VERIFY every value below against the actual RFC text
    // before trusting this transcription -- key is 00..1f sequential,
    // nonce and counter come from the RFC's specific example (counter=1,
    // NOT counter=0 -- confirm this exact value against the RFC, some
    // ChaCha20 references use a different first example).
    let key: [u8;32] = [
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
        0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
    ]
    let nonce: [u8;12] = [
        0x00,0x00,0x00,0x09,0x00,0x00,0x00,0x4a,0x00,0x00,0x00,0x00
    ]
    let counter: i64 = 1
    let expected: [u8;64] = [
        0x10,0xf1,0xe7,0xe4,0xd1,0x3b,0x59,0x15,0x50,0x0f,0xdd,0x1f,0xa3,0x20,0x71,0xc4,
        0xc7,0xd1,0xf4,0xc7,0x33,0xc0,0x68,0x03,0x04,0x22,0xaa,0x9a,0xc3,0xd4,0x6c,0x4e,
        0xd2,0x82,0x64,0x46,0x07,0x9f,0xaa,0x09,0x14,0xc2,0xd7,0x05,0xd9,0x8b,0x02,0xa2,
        0xb5,0x12,0x9c,0xd1,0xde,0x16,0x4e,0xb9,0xcb,0xd0,0x83,0xe8,0xa2,0x50,0x3c,0x4e
    ]

    let block = chacha20_block(&key, counter, &nonce)
    var i: i64 = 0
    while i < 64 {
        if block[i as usize] != expected[i as usize] {
            print_int(1)
            return 1
        }
        i = i + 1
    }

    println("chacha20_block: matches RFC 8439 section 2.3.2's published test vector")
    return 0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/chacha20_block.sio`
Expected: unresolved-import error referencing `crypto::chacha20`.

- [ ] **Step 3: Implement `chacha20.sio`**

```sio
use hash::word32::*
use net::socket::*

pub const CHACHA20_KEY_LEN: i64 = 32
pub const CHACHA20_NONCE_LEN: i64 = 12

fn quarter_round(state: &[i64;16], a: i64, b: i64, c: i64, d: i64) -> [i64;16] {
    var s = *state
    s[a as usize] = add32(s[a as usize], s[b as usize])
    s[d as usize] = rotl32(xor32(s[d as usize], s[a as usize]), 16)
    s[c as usize] = add32(s[c as usize], s[d as usize])
    s[b as usize] = rotl32(xor32(s[b as usize], s[c as usize]), 12)
    s[a as usize] = add32(s[a as usize], s[b as usize])
    s[d as usize] = rotl32(xor32(s[d as usize], s[a as usize]), 8)
    s[c as usize] = add32(s[c as usize], s[d as usize])
    s[b as usize] = rotl32(xor32(s[b as usize], s[c as usize]), 7)
    s
}

pub fn chacha20_block(key: &[u8;32], counter: i64, nonce: &[u8;12]) -> [u8;64] {
    // "expa" "nd 3" "2-by" "te k" as four little-endian 32-bit constants.
    var state: [i64;16] = [0;16]
    state[0] = 0x61707865
    state[1] = 0x3320646e
    state[2] = 0x79622d32
    state[3] = 0x6b206574

    var i: i64 = 0
    while i < 8 {
        state[(4 + i) as usize] =
            (key[(i*4) as usize] as i64) |
            ((key[(i*4+1) as usize] as i64) << 8) |
            ((key[(i*4+2) as usize] as i64) << 16) |
            ((key[(i*4+3) as usize] as i64) << 24)
        i = i + 1
    }

    state[12] = counter
    var j: i64 = 0
    while j < 3 {
        state[(13 + j) as usize] =
            (nonce[(j*4) as usize] as i64) |
            ((nonce[(j*4+1) as usize] as i64) << 8) |
            ((nonce[(j*4+2) as usize] as i64) << 16) |
            ((nonce[(j*4+3) as usize] as i64) << 24)
        j = j + 1
    }

    let initial = state
    var working = state
    var round: i64 = 0
    while round < 10 {   // 10 double-rounds = 20 rounds total
        working = quarter_round(&working, 0, 4, 8, 12)
        working = quarter_round(&working, 1, 5, 9, 13)
        working = quarter_round(&working, 2, 6, 10, 14)
        working = quarter_round(&working, 3, 7, 11, 15)
        working = quarter_round(&working, 0, 5, 10, 15)
        working = quarter_round(&working, 1, 6, 11, 12)
        working = quarter_round(&working, 2, 7, 8, 13)
        working = quarter_round(&working, 3, 4, 9, 14)
        round = round + 1
    }

    var output: [u8;64] = [0;64]
    var k: i64 = 0
    while k < 16 {
        let word = add32(working[k as usize], initial[k as usize])
        output[(k*4) as usize] = (word & 255) as u8
        output[(k*4+1) as usize] = ((word >> 8) & 255) as u8
        output[(k*4+2) as usize] = ((word >> 16) & 255) as u8
        output[(k*4+3) as usize] = ((word >> 24) & 255) as u8
        k = k + 1
    }
    output
}

pub fn chacha20_encrypt(key: &[u8;32], counter: i64, nonce: &[u8;12], data: &RawBuf, data_len: i64) -> RawBuf with IO {
    let out = rawbuf_new(data_len + 8)
    var block_index: i64 = 0
    var pos: i64 = 0
    while pos < data_len {
        let ks = chacha20_block(key, add32(counter, block_index), nonce)
        var i: i64 = 0
        while i < 64 && pos + i < data_len {
            rawbuf_set(&out, pos + i, rawbuf_get(data, pos + i) ^ (ks[i as usize] as i64))
            i = i + 1
        }
        pos = pos + 64
        block_index = block_index + 1
    }
    out
}
```

Note: `rotl32`'s only accepted rotation amounts in this task are 16, 12, 8, 7 — all safely within `word32.sio`'s documented `1 <= n <= 31` precondition, no separate verification needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/chacha20_block.sio`
Expected: prints `chacha20_block: matches RFC 8439 section 2.3.2's published test vector`, exit 0.

- [ ] **Step 5: Add a `chacha20_encrypt` round-trip test at a non-64-byte-multiple length**

`tests/run-pass/chacha20_encrypt_roundtrip.sio`: encrypt a 100-byte plaintext (not a multiple of 64) with a fixed key/nonce/counter, then encrypt the resulting ciphertext AGAIN with the identical key/nonce/counter (ChaCha20 is a stream cipher — XOR-ing the keystream twice recovers the original), and assert the result equals the original plaintext.

- [ ] **Step 6: Run both tests**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix chacha20_ --jobs 2 --verbose`
Expected: 2 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add stdlib/crypto/chacha20.sio tests/run-pass/chacha20_block.sio tests/run-pass/chacha20_encrypt_roundtrip.sio
git commit -m "feat(crypto): add ChaCha20 stream cipher"
```

---

## Task 4: `stdlib/crypto/poly1305.sio` — Poly1305 one-time MAC

**Files:**
- Create: `stdlib/crypto/poly1305.sio`
- Test: `tests/run-pass/poly1305_mac.sio`

**Interfaces:**
- Produces: `pub const POLY1305_KEY_LEN: i64 = 32`, `pub const POLY1305_TAG_LEN: i64 = 16`, `pub fn poly1305_mac(key: &[u8;32], msg: &RawBuf, msg_len: i64) -> [u8;16] with IO`. Task 5 imports and calls this exact name/signature.

Poly1305's arithmetic is modulo `2^130 - 5` — a 130-bit modulus, wider than a single 64-bit word and past even `word32.sio`'s 32-bit scope. Before implementing, decide explicitly (in a code comment, matching this codebase's established `bigint.sio`-style overflow-margin reasoning) whether to reuse `stdlib/bignum/bigint.sio`'s existing `BigInt` (4096-bit, 16-bit limbs, fixed-width) or to build a small purpose-built limb representation for exactly this field. **Recommendation, not a mandate — verify it holds once you're implementing**: `BigInt`'s 4096-bit width is substantial overkill for 130-bit arithmetic (every operation pays for 256 limbs' worth of work when ~9 sixteen-bit limbs would cover 130 bits), and `BigInt`'s API (`bigint_mul`, `bigint_add` returning fresh full-width values) doesn't naturally express Poly1305's specific reduction step (reducing mod `2^130-5` after each block, which has a cheap near-Mersenne-prime-style reduction that generic `BigInt` multiplication/division doesn't exploit). Building a small, purpose-built accumulator (e.g. five 26-bit limbs, mirroring the well-known reference-implementation "radix 2^26" approach, or ten 13-bit limbs — pick whichever you can reason most confidently about the overflow margins for) is likely the better fit — but this is your call to make explicitly, not silently.

- [ ] **Step 1: Write the failing test using RFC 8439's published standalone MAC vector**

```sio
//@ run-pass
use crypto::poly1305::*
use net::socket::*

fn main() -> i64 with IO {
    // RFC 8439 section 2.5.2's standalone Poly1305 test vector.
    // INDEPENDENTLY RE-VERIFY the key, message, and expected tag below
    // against the actual RFC text before trusting this transcription.
    let key: [u8;32] = [
        0x85,0xd6,0xbe,0x78,0x57,0x55,0x6d,0x33,0x7f,0x44,0x52,0xfe,0x42,0xd5,0x06,0xa8,
        0x01,0x03,0x80,0x8a,0xfb,0x0d,0xb2,0xfd,0x4a,0xbf,0xf6,0xaf,0x41,0x49,0xf5,0x1b
    ]
    // "Cryptographic Forum Research Group" -- 34 ASCII bytes.
    let msg_bytes: [u8;34] = [
        67,114,121,112,116,111,103,114,97,112,104,105,99,32,70,111,
        114,117,109,32,82,101,115,101,97,114,99,104,32,71,114,111,
        117,112
    ]
    let expected_tag: [u8;16] = [
        0xa8,0x06,0x1d,0xc1,0x30,0x51,0x36,0xc6,0xc2,0x2b,0x8b,0xaf,0x0c,0x01,0x27,0xa9
    ]

    let msg_buf = rawbuf_new(34 + 8)
    var i: i64 = 0
    while i < 34 { rawbuf_set(&msg_buf, i, msg_bytes[i as usize] as i64); i = i + 1 }

    let tag = poly1305_mac(&key, &msg_buf, 34)
    var j: i64 = 0
    while j < 16 {
        if tag[j as usize] != expected_tag[j as usize] {
            print_int(1)
            return 1
        }
        j = j + 1
    }

    println("poly1305_mac: matches RFC 8439 section 2.5.2's published test vector")
    return 0
}
```

Double-check the 34-byte ASCII transcription of "Cryptographic Forum Research Group" above character-by-character against the RFC text (or regenerate it from the literal string yourself) before trusting the byte array — transcribing a 34-character string as 34 individual integer literals is exactly the kind of manual step where a single off-by-one silently breaks the test without looking wrong.

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/poly1305_mac.sio`
Expected: unresolved-import error referencing `crypto::poly1305`.

- [ ] **Step 3: Implement `poly1305.sio`**

```sio
use net::socket::*

pub const POLY1305_KEY_LEN: i64 = 32
pub const POLY1305_TAG_LEN: i64 = 16

// <fill in: your chosen limb representation (radix-2^26 5-limb accumulator
// recommended, see Interfaces note above), with a code comment explicitly
// reasoning about the overflow margin for every intermediate multiply/add
// the same way stdlib/bignum/bigint.sio's own comments do -- e.g. "each
// limb value stays below 2^26, multiplying two limbs is below 2^52, summing
// five such products is below 2^55, far under i64's 63-bit danger zone
// (Finding 11)". Implement clamp(r) (mask r per RFC 8439's r-clamping
// rule), the block-accumulation loop (process msg in 16-byte chunks, the
// final partial chunk gets a single 0x01 byte appended before the
// accumulation per RFC 8439's padding rule, NOT zero-padded), and the
// final s-addition + mod-2^128 truncation to produce the 16-byte tag.>

pub fn poly1305_mac(key: &[u8;32], msg: &RawBuf, msg_len: i64) -> [u8;16] with IO {
    // <fill in: r = clamp(key[0..16]), s = key[16..32], accumulate msg,
    // tag = (accumulator + s) mod 2^128>
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/poly1305_mac.sio`
Expected: prints `poly1305_mac: matches RFC 8439 section 2.5.2's published test vector`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/crypto/poly1305.sio tests/run-pass/poly1305_mac.sio
git commit -m "feat(crypto): add Poly1305 one-time MAC"
```

---

## Task 5: `stdlib/crypto/aead.sio` — unified AEAD interface

**Files:**
- Create: `stdlib/crypto/aead.sio`
- Test: `tests/run-pass/aead_chacha20poly1305.sio`, `tests/run-pass/aead_aes_gcm.sio`, `tests/run-pass/aead_adversarial.sio`

**Interfaces:**
- Consumes: `AesRoundKeys`, `aes128_key_schedule`, `aes256_key_schedule` (Task 1); `gcm_seal`, `gcm_open`, `GCM_TAG_LEN` (Task 2); `chacha20_encrypt`, `chacha20_block` (Task 3); `poly1305_mac` (Task 4).
- Produces (the interface a future TLS handshake sub-project calls): `pub const AEAD_AES_128_GCM: i32 = 0`, `pub const AEAD_AES_256_GCM: i32 = 1`, `pub const AEAD_CHACHA20_POLY1305: i32 = 2`, `pub fn aead_seal(algo: i32, key: &[u8;32], nonce: &[u8;12], plaintext: &RawBuf, plaintext_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, i64) with IO`, `pub fn aead_open(algo: i32, key: &[u8;32], nonce: &[u8;12], ciphertext_with_tag: &RawBuf, ciphertext_with_tag_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, bool) with IO`.

The ChaCha20-Poly1305 AEAD construction (RFC 8439 section 2.8) has its own padding and length-encoding rules — **little-endian** length encoding and pad-to-16-byte-boundary — which are DIFFERENT from GCM's own big-endian-bit-count length encoding and padding (already implemented inside `gcm.sio`'s `ghash` in Task 2). Do not attempt to share code between the two constructions' length/padding logic — implement RFC 8439 section 2.8's construction as its own, separate logic inside this task, even though it looks superficially similar to GCM's.

- [ ] **Step 1: Write the failing test for ChaCha20-Poly1305 using RFC 8439's full AEAD vector**

```sio
//@ run-pass
use crypto::aead::*
use net::socket::*

fn main() -> i64 with IO {
    // RFC 8439 section 2.8.2's complete worked AEAD example.
    // INDEPENDENTLY RE-VERIFY every value below against the actual RFC
    // text before trusting this transcription -- this is a long vector
    // and the highest-risk transcription in this task.
    let key: [u8;32] = [
        0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,
        0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0x9b,0x9c,0x9d,0x9e,0x9f
    ]
    let nonce: [u8;12] = [0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x4b]
    let aad_bytes: [u8;12] = [0x50,0x51,0x52,0x53,0xc0,0xc1,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7]

    // "Ladies and Gentlemen of the class of '99: If I could offer you only
    // one tip for the future, sunscreen would be it." -- 114 ASCII bytes.
    // Build this from the literal string rather than hand-transcribing 114
    // integer literals, to eliminate transcription risk for the plaintext
    // (the ciphertext/tag below still need independent verification).
    let plaintext_str = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it."
    let plaintext_len: i64 = 114   // VERIFY this length against the actual string above

    let expected_ciphertext: [u8;114] = [
        0xd3,0x1a,0x8d,0x34,0x64,0x8e,0x60,0xdb,0x7b,0x86,0xaf,0xbc,0x53,0xef,0x7e,0xc2,
        0xa4,0xad,0xed,0x51,0x29,0x6e,0x08,0xfe,0xa9,0xe2,0xb5,0xa7,0x36,0xee,0x62,0xd6,
        0x3d,0xbe,0xa4,0x5e,0x8c,0xa9,0x67,0x12,0x82,0xfa,0xfb,0x69,0xda,0x92,0x72,0x8b,
        0x1a,0x71,0xde,0x0a,0x9e,0x06,0x0b,0x29,0x05,0xd6,0xa5,0xb6,0x7e,0xcd,0x3b,0x36,
        0x92,0xdd,0xbd,0x7f,0x2d,0x77,0x8b,0x8c,0x98,0x03,0xae,0xe3,0x28,0x09,0x1b,0x58,
        0xfa,0xb3,0x24,0xe4,0xfa,0xd6,0x75,0x94,0x55,0x85,0x80,0x8b,0x48,0x31,0xd7,0xbc,
        0x3f,0xf4,0xde,0xf0,0x8e,0x4b,0x7a,0x9d,0xe5,0x76,0xd2,0x65,0x86,0xce,0xc6,0x4b,
        0x61,0x16
    ]
    let expected_tag: [u8;16] = [
        0x1a,0xe1,0x0b,0x59,0x4f,0x09,0xe2,0x6a,0x7e,0x90,0x2e,0xcb,0xd0,0x60,0x06,0x91
    ]

    let pt_buf = rawbuf_new(plaintext_len + 8)
    var i: i64 = 0
    while i < plaintext_len { rawbuf_set(&pt_buf, i, str_char_at(plaintext_str, i) as i64); i = i + 1 }
    let aad_buf = rawbuf_new(12 + 8)
    var j: i64 = 0
    while j < 12 { rawbuf_set(&aad_buf, j, aad_bytes[j as usize] as i64); j = j + 1 }

    let (sealed, sealed_len) = aead_seal(AEAD_CHACHA20_POLY1305, &key, &nonce, &pt_buf, plaintext_len, &aad_buf, 12)
    if sealed_len != plaintext_len + 16 {
        print_int(1)
        return 1
    }
    var c: i64 = 0
    while c < plaintext_len {
        if rawbuf_get(&sealed, c) != (expected_ciphertext[c as usize] as i64) {
            print_int(2)
            return 1
        }
        c = c + 1
    }
    var t: i64 = 0
    while t < 16 {
        if rawbuf_get(&sealed, plaintext_len + t) != (expected_tag[t as usize] as i64) {
            print_int(3)
            return 1
        }
        t = t + 1
    }

    println("aead_chacha20poly1305: matches RFC 8439 section 2.8.2's full worked AEAD example")
    return 0
}
```

Verify `str_char_at`'s real signature against `stdlib`'s string handling before using it (grep for existing callers — e.g. `stdlib/x509/trust_store.sio`'s `read_file`-to-`RawBuf` conversion from an earlier plan on this same branch used a similar string-to-bytes pattern) rather than assuming it compiles as sketched.

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/aead_chacha20poly1305.sio`
Expected: unresolved-import error referencing `crypto::aead`.

- [ ] **Step 3: Implement `aead.sio`**

```sio
use net::socket::*
use crypto::aes::*
use crypto::gcm::*
use crypto::chacha20::*
use crypto::poly1305::*

pub const AEAD_AES_128_GCM: i32 = 0
pub const AEAD_AES_256_GCM: i32 = 1
pub const AEAD_CHACHA20_POLY1305: i32 = 2

// RFC 8439 section 2.8's AEAD_CHACHA20_POLY1305 construction. Distinct
// padding/length-encoding from GCM's ghash (little-endian lengths, pad-to-
// 16 -- see this task's own note above) -- deliberately NOT shared code
// with gcm.sio's ghash.
fn chacha20poly1305_seal(key: &[u8;32], nonce: &[u8;12], plaintext: &RawBuf, plaintext_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, i64) with IO {
    let poly_key_block = chacha20_block(key, 0, nonce)
    var poly_key: [u8;32] = [0;32]
    var i: i64 = 0
    while i < 32 { poly_key[i as usize] = poly_key_block[i as usize]; i = i + 1 }

    let ciphertext = chacha20_encrypt(key, 1, nonce, plaintext, plaintext_len)

    // mac_data = aad || pad16(aad) || ciphertext || pad16(ciphertext) ||
    //            len(aad) as u64-LE || len(ciphertext) as u64-LE
    let aad_pad = (16 - (aad_len % 16)) % 16
    let ct_pad = (16 - (plaintext_len % 16)) % 16
    let mac_data_len = aad_len + aad_pad + plaintext_len + ct_pad + 8 + 8
    let mac_data = rawbuf_new(mac_data_len + 8)
    var pos: i64 = 0
    var a: i64 = 0
    while a < aad_len { rawbuf_set(&mac_data, pos, rawbuf_get(aad, a)); pos = pos + 1; a = a + 1 }
    var ap: i64 = 0
    while ap < aad_pad { rawbuf_set(&mac_data, pos, 0); pos = pos + 1; ap = ap + 1 }
    var c: i64 = 0
    while c < plaintext_len { rawbuf_set(&mac_data, pos, rawbuf_get(&ciphertext, c)); pos = pos + 1; c = c + 1 }
    var cp: i64 = 0
    while cp < ct_pad { rawbuf_set(&mac_data, pos, 0); pos = pos + 1; cp = cp + 1 }
    var lb: i64 = 0
    while lb < 8 { rawbuf_set(&mac_data, pos, (aad_len >> (lb * 8)) & 255); pos = pos + 1; lb = lb + 1 }
    var lc: i64 = 0
    while lc < 8 { rawbuf_set(&mac_data, pos, (plaintext_len >> (lc * 8)) & 255); pos = pos + 1; lc = lc + 1 }

    let tag = poly1305_mac(&poly_key, &mac_data, mac_data_len)

    let out = rawbuf_new(plaintext_len + 16 + 8)
    var oc: i64 = 0
    while oc < plaintext_len { rawbuf_set(&out, oc, rawbuf_get(&ciphertext, oc)); oc = oc + 1 }
    var ot: i64 = 0
    while ot < 16 { rawbuf_set(&out, plaintext_len + ot, tag[ot as usize] as i64); ot = ot + 1 }

    (out, plaintext_len + 16)
}

fn chacha20poly1305_open(key: &[u8;32], nonce: &[u8;12], ciphertext_with_tag: &RawBuf, ciphertext_with_tag_len: i64, aad: &RawBuf, aad_len: i64) -> (RawBuf, bool) with IO {
    if ciphertext_with_tag_len < 16 {
        return (rawbuf_new(1), false)
    }
    let plaintext_len = ciphertext_with_tag_len - 16

    let ciphertext_only = rawbuf_new(plaintext_len + 8)
    var c: i64 = 0
    while c < plaintext_len { rawbuf_set(&ciphertext_only, c, rawbuf_get(ciphertext_with_tag, c)); c = c + 1 }

    // <fill in: recompute poly_key and mac_data identically to
    // chacha20poly1305_seal, compute expected_tag, constant-time-compare
    // against ciphertext_with_tag's trailing 16 bytes (accumulate an OR of
    // byte differences, compare once), decrypt via chacha20_encrypt(key, 1,
    // nonce, ...) regardless, then return (plaintext, true) or
    // (all-zero RawBuf, false) per this plan's fail-closed constraint>
}

pub fn aead_seal(
    algo: i32,
    key: &[u8;32],
    nonce: &[u8;12],
    plaintext: &RawBuf,
    plaintext_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, i64) with IO {
    if algo == AEAD_AES_128_GCM {
        var key16: [u8;16] = [0;16]
        var i: i64 = 0
        while i < 16 { key16[i as usize] = key[i as usize]; i = i + 1 }
        let round_keys = aes128_key_schedule(&key16)
        gcm_seal(&round_keys, nonce, plaintext, plaintext_len, aad, aad_len)
    } else if algo == AEAD_AES_256_GCM {
        let round_keys = aes256_key_schedule(key)
        gcm_seal(&round_keys, nonce, plaintext, plaintext_len, aad, aad_len)
    } else {
        chacha20poly1305_seal(key, nonce, plaintext, plaintext_len, aad, aad_len)
    }
}

pub fn aead_open(
    algo: i32,
    key: &[u8;32],
    nonce: &[u8;12],
    ciphertext_with_tag: &RawBuf,
    ciphertext_with_tag_len: i64,
    aad: &RawBuf,
    aad_len: i64,
) -> (RawBuf, bool) with IO {
    if algo == AEAD_AES_128_GCM {
        var key16: [u8;16] = [0;16]
        var i: i64 = 0
        while i < 16 { key16[i as usize] = key[i as usize]; i = i + 1 }
        let round_keys = aes128_key_schedule(&key16)
        gcm_open(&round_keys, nonce, ciphertext_with_tag, ciphertext_with_tag_len, aad, aad_len)
    } else if algo == AEAD_AES_256_GCM {
        let round_keys = aes256_key_schedule(key)
        gcm_open(&round_keys, nonce, ciphertext_with_tag, ciphertext_with_tag_len, aad, aad_len)
    } else {
        chacha20poly1305_open(key, nonce, ciphertext_with_tag, ciphertext_with_tag_len, aad, aad_len)
    }
}
```

- [ ] **Step 4: Fill in `chacha20poly1305_open`, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/aead_chacha20poly1305.sio`
Expected: prints `aead_chacha20poly1305: matches RFC 8439 section 2.8.2's full worked AEAD example`, exit 0.

- [ ] **Step 5: Write `tests/run-pass/aead_aes_gcm.sio`**

Re-run Task 2's own AES-128-GCM NIST vector (same key/nonce/plaintext/expected-ciphertext/expected-tag as `tests/run-pass/gcm_seal_open.sio`, copied verbatim — not regenerated) through `aead_seal(AEAD_AES_128_GCM, ...)`/`aead_open`, asserting the identical byte-level result. This proves the dispatch layer's key-slicing (`key16` extraction) doesn't corrupt what `gcm.sio` already got right.

- [ ] **Step 6: Write `tests/run-pass/aead_adversarial.sio`**

For BOTH `AEAD_AES_128_GCM` and `AEAD_CHACHA20_POLY1305` (2 algorithms × 3 cases = 6 total, structure as separate helper functions per this branch's established pattern for avoiding the Madaros per-function IR-instruction budget wall when many fixtures live in one `main`):
1. Seal a message, flip one tag byte, confirm `aead_open` returns `ok == false` and an all-zero plaintext.
2. Seal a message, flip one AAD byte before calling `aead_open` (not before `aead_seal`), confirm `ok == false`.
3. Seal a message, truncate the sealed output to fewer bytes than the tag length, confirm `aead_open` returns `ok == false` (not a panic or out-of-bounds read).

- [ ] **Step 7: Write round-trip tests at non-block-aligned lengths**

Add to `aead_adversarial.sio` or a new `tests/run-pass/aead_roundtrip.sio`: for both algorithms, `aead_open(aead_seal(msg))` recovers `msg` exactly at plaintext lengths 1, 15, 17, 63, 65 bytes (spanning both AES's 16-byte and ChaCha20's 64-byte natural chunk sizes, on both sides of each boundary).

- [ ] **Step 8: Run the full test suite for this plan**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix aes_ --jobs 2 --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix gcm_ --jobs 2 --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix chacha20_ --jobs 2 --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix poly1305_ --jobs 2 --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix aead_ --jobs 2 --verbose`
Expected: every one reports all passed, 0 failed.

- [ ] **Step 9: Commit**

```bash
git add stdlib/crypto/aead.sio tests/run-pass/aead_chacha20poly1305.sio tests/run-pass/aead_aes_gcm.sio tests/run-pass/aead_adversarial.sio tests/run-pass/aead_roundtrip.sio
git commit -m "feat(crypto): add unified AEAD interface dispatching to AES-GCM and ChaCha20-Poly1305"
```

---

## After This Plan

`crypto::aead::aead_seal`/`aead_open` (the interface defined in Task 5) is what the not-yet-scoped TLS handshake sub-project's own spec should reference directly — that sub-project also needs X25519 key exchange and HKDF/TLS 1.2-PRF key derivation, neither of which this plan covers, and its own brainstorming/spec process starts fresh once this plan's final whole-branch review (per `superpowers:subagent-driven-development`) is clean.

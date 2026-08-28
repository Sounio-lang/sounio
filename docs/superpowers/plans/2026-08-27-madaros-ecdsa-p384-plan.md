<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-27-madaros-ecdsa-p384-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-27-madaros-ecdsa-p384-plan
-->

# NIST P-384 (secp384r1) ECDSA Chain-Verification Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `x509_verify_signature` able to validate a certificate signed `ecdsa-with-SHA384` by a **P-384** issuer key — the real, live gap blocking Cloudflare's `1.1.1.1:443` DoH chain (its intermediate has a P-384 key).

**Architecture:** Add P-384 curve constants and a shared-core ECDSA verifier (`stdlib/crypto/ecdsa_p384.sio`, mirroring `ecdsa_p256.sio`'s existing shared-core pattern) that reuses the already curve-generic point arithmetic (`ec_point_add`/`ec_scalar_mul`/`ec_mod_inverse`) unchanged. Add `secp384r1` OID recognition and 97-byte uncompressed-point parsing to `stdlib/x509/cert.sio`'s existing EC-key-parsing branch. Wire the new verifier into `x509_verify_signature`'s existing `is_ecdsa_sha384_sig_alg` branch, extending (not replacing) D20's algorithm-confusion guard to accept both P-256 and P-384 issuer keys, each still individually verified only through its own curve's arithmetic.

**Tech Stack:** Sounio stdlib (`stdlib/crypto/`, `stdlib/x509/`), Madaros compiler, OpenSSL 3.x for independent constant/fixture verification.

**Spec:** [`docs/superpowers/specs/2026-08-27-madaros-ecdsa-p384-design.md`](../specs/2026-08-27-madaros-ecdsa-p384-design.md)

## Global Constraints

- **Every hardcoded cryptographic constant (curve parameters, the new OID) MUST be independently re-verified against a real, live `openssl` invocation before being trusted** — this spec's own numbers were derived and triple-checked (curve-equation, primality, bit-length, Hasse bound), but per the spec's own mandatory-verification requirement, re-derive/re-confirm them again during implementation from a second, live source. This is not optional and not satisfied by re-asserting the same numbers a second time in code.
- Point-at-infinity sentinel, BigInt limb ordering (little-endian, limb 0 = least significant 16 bits), and the "no top-level `const [u8;N]`" rule (Finding 18, `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` — a top-level `const [u8;N]` is corrupted when its whole value is copied/referenced) all apply identically to every new function in this plan, exactly as they already apply in `stdlib/crypto/ecdsa_p256.sio` and `stdlib/x509/oid.sio`.
- TLS 1.3 `CertificateVerify` P-384 support (`ecdsa_secp384r1_sha384`, scheme 0x0503) is explicitly OUT OF SCOPE for this plan — see spec §Non-goals.
- No regressions: `x509`, `tls`, `crypto`, `pem`, `asn1`, `bignum` test-suite prefixes must show the same pass/fail counts before and after this plan's changes, plus the new tests passing.
- `bin/souc` / `SOUNIO_STDLIB_PATH` invocation matches this worktree's own established convention: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run <file>.sio` from the repo root (`/home/devsounio/sounio/.claude/worktrees/sounio-tls-on-madaros`).

---

### Task 1: P-384 curve constants, independently re-verified against OpenSSL

**Files:**
- Create: `stdlib/crypto/ecdsa_p384.sio`
- Test: `tests/run-pass/ecdsa_p384_curve_constants.sio`

**Interfaces:**
- Consumes: `stdlib/bignum/bigint.sio`'s `BigInt { limbs: [u16; 512], len: i32 }`, `bigint_zero()`, `bigint_cmp`, `bigint_from_u32`.
- Produces: `pub fn p384_field_prime() -> BigInt`, `pub fn p384_curve_a() -> BigInt`, `pub fn p384_curve_b() -> BigInt`, `pub fn p384_order() -> BigInt`, `pub fn p384_base_point() -> EcPoint` (uses `stdlib::crypto::ecdsa_p256::EcPoint`, imported — do not redefine `EcPoint`), and a private `build_bigint_24(values: [u16; 24]) -> BigInt` helper. These four functions and the point are consumed by Task 2.

- [ ] **Step 1: Independently re-verify the curve constants against a live OpenSSL, before writing any Sounio code**

Run, in a shell with `openssl` available:

```bash
openssl ecparam -name secp384r1 -param_enc explicit -text -noout
```

This prints `Prime:`, `A:`, `B:`, `Generator (uncompressed):`, and `Order:` as hex byte dumps. Confirm each of the following independently — do not skip this because the numbers below already look right:

1. `Prime:` matches `2^384 - 2^128 - 2^96 + 2^32 - 1` (compute this value directly, e.g. `python3 -c "print(hex(2**384 - 2**128 - 2**96 + 2**32 - 1))"`, and diff against OpenSSL's printed hex).
2. `A:` equals `Prime - 3`.
3. `Generator`'s uncompressed point (`04 || Gx || Gy`, 97 bytes) satisfies `Gy^2 ≡ Gx^3 + A*Gx + B (mod Prime)` — check this with a short Python script using `pow(x, 2, p)` etc., not by inspection.
4. `Order:` is prime (`sympy.isprime(n)` or equivalent) and satisfies the Hasse bound `abs(Prime + 1 - Order) <= 2 * isqrt(Prime)`.
5. Convert OpenSSL's own printed hex bytes for `Prime`, `A`, `B`, `Generator`, `Order` into 24-element `[u16; 24]` little-endian limb arrays (limb 0 = least-significant 16 bits) with a short Python script — e.g.:

```python
def hex_to_limbs24(hex_str):
    b = bytes.fromhex(hex_str.replace(":", "").replace(" ", ""))
    assert len(b) == 48, f"expected 48 bytes, got {len(b)}"
    val = int.from_bytes(b, "big")
    return [(val >> (16 * i)) & 0xFFFF for i in range(24)]
```

6. Compare the script's output against the limb arrays given in Step 3 below byte-for-byte. If OpenSSL's own live output disagrees with the arrays below in even one limb, OpenSSL is the authority — use OpenSSL's value and note the discrepancy in the task report.

Record the exact OpenSSL version used (`openssl version`) and the full command transcript in the task report, matching `ecdsa_p256.sio`'s own header-comment convention (see that file's lines 10–24 for the exact style to follow).

- [ ] **Step 2: Write the failing test**

```sio
// tests/run-pass/ecdsa_p384_curve_constants.sio
//@ run-pass

// Pins P-384's curve constants against the curve equation and basic
// structural invariants. This does NOT replace the OpenSSL cross-check
// from Task 1 Step 1 (a self-consistent-but-wrong curve would still pass
// a check that only uses its own numbers) -- it is a regression guard so
// a future edit to this file cannot silently corrupt a limb.

use crypto::ecdsa_p384::*
use bignum::bigint::*

fn main() -> i64 with IO, Mut, Div, Panic {
    let p = p384_field_prime()
    let a = p384_curve_a()
    let b = p384_curve_b()
    let n = p384_order()
    let g = p384_base_point()

    assert(p.len == 24)
    assert(a.len == 24)
    assert(n.len == 24)

    // a == p - 3
    let three = bigint_from_u32(3)
    let p_minus_3 = bigint_sub(&p, &three)
    assert(bigint_cmp(&a, &p_minus_3) == 0)

    // Curve equation: Gy^2 mod p == (Gx^3 + a*Gx + b) mod p
    let gy_sq = bigint_mod(&bigint_mul(&g.y, &g.y), &p)
    let gx_sq = bigint_mod(&bigint_mul(&g.x, &g.x), &p)
    let gx_cubed = bigint_mod(&bigint_mul(&gx_sq, &g.x), &p)
    let a_gx = bigint_mod(&bigint_mul(&a, &g.x), &p)
    let rhs_partial = bigint_mod(&bigint_add(&gx_cubed, &a_gx), &p)
    let rhs = bigint_mod(&bigint_add(&rhs_partial, &b), &p)
    assert(bigint_cmp(&gy_sq, &rhs) == 0)

    // n < p (true for every NIST prime curve; catches a gross transcription error)
    assert(bigint_cmp(&n, &p) < 0)

    println("ecdsa_p384_curve_constants: OK")
    0
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/ecdsa_p384_curve_constants.sio`
Expected: FAIL — `crypto::ecdsa_p384` module not found (file does not exist yet).

- [ ] **Step 4: Write the implementation**

The limb arrays below were derived from `p = 2^384 - 2^128 - 2^96 + 2^32 - 1` and cross-checked by curve equation, primality, bit-length, and Hasse bound (see spec §3). **Re-verify them against your own Step 1 OpenSSL output before trusting them — replace any limb that disagrees.**

```sio
// stdlib/crypto/ecdsa_p384.sio
//
// NIST P-384 (secp384r1) ECDSA signature verification (FIPS 186-4 SS6.4.2
// / SEC1 SS4.1.4), for X.509 certificates whose ISSUER key is P-384 and
// whose signature algorithm is ecdsa-with-SHA384 (OID 1.2.840.10045.4.3.3).
// See docs/superpowers/specs/2026-08-27-madaros-ecdsa-p384-design.md.
//
// CURVE PARAMETERS -- independently re-verified against a live
// `openssl ecparam -name secp384r1 -param_enc explicit -text -noout`
// (record OpenSSL version and transcript in this task's own report), same
// discipline as stdlib/crypto/ecdsa_p256.sio's own header comment:
//   - p == 2^384 - 2^128 - 2^96 + 2^32 - 1 (confirmed by direct computation)
//   - a == p - 3
//   - G = (Gx, Gy) satisfies Gy^2 == Gx^3 + a*Gx + b (mod p)
//   - n is prime and satisfies the Hasse bound relative to p
//
// Reuses stdlib::crypto::ecdsa_p256's EcPoint, ec_point_add, ec_scalar_mul,
// ec_mod_inverse, ec_is_infinity UNCHANGED -- those functions take the
// field prime and curve coefficient as parameters and perform generic
// short-Weierstrass-curve arithmetic, not P-256-specific arithmetic (see
// that file's own signatures). No new point/field arithmetic is needed
// here, only new curve-parameter constants and a verification entry point.
//
// Digest handling: SHA-384's 48-byte (384-bit) output exactly matches
// P-384's 384-bit curve order -- no FIPS 186-4 SS6.4.1 truncation is
// needed (unlike ecdsa_p256_verify_sha384's leftmost-32-of-48-byte
// truncation, needed there because P-256's order is only 256 bits). All
// 48 bytes are packed into a 24-limb BigInt directly.

use bignum::bigint::*
use hash::sha384::*
use asn1::der::*
use crypto::ecdsa_p256::EcPoint
use crypto::ecdsa_p256::ec_point_add
use crypto::ecdsa_p256::ec_scalar_mul
use crypto::ecdsa_p256::ec_mod_inverse
use crypto::ecdsa_p256::ec_is_infinity

// Builds a 384-bit BigInt (24 significant limbs) from a little-endian
// 16-bit limb array. Every curve constant below has a nonzero top limb
// (index 23), so len == 24 always, matching build_bigint_16's own
// no-trim convention in ecdsa_p256.sio.
fn build_bigint_24(values: [u16; 24]) -> BigInt {
    var result = bigint_zero()
    var i: i64 = 0
    while i < 24 {
        result.limbs[i as usize] = values[i as usize]
        i = i + 1
    }
    result.len = 24
    result
}

// p = 2^384 - 2^128 - 2^96 + 2^32 - 1 (FIPS 186-4 D.1.2.4 / SEC2 secp384r1).
pub fn p384_field_prime() -> BigInt {
    build_bigint_24([
        65535, 65535, 0, 0, 0, 0, 65535, 65535, 65534, 65535, 65535, 65535,
        65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535,
    ])
}

// a = p - 3.
pub fn p384_curve_a() -> BigInt {
    build_bigint_24([
        65532, 65535, 0, 0, 0, 0, 65535, 65535, 65534, 65535, 65535, 65535,
        65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535,
    ])
}

// b (curve equation y^2 = x^3 + a*x + b mod p).
pub fn p384_curve_b() -> BigInt {
    build_bigint_24([
        10991, 54252, 51437, 10885, 53661, 35374, 14733, 50774, 34650,
        20499, 2191, 788, 16658, 65153, 40046, 6173, 11545, 58360, 1387,
        39054, 59364, 57918, 12199, 45873,
    ])
}

// n, the group order.
pub fn p384_order() -> BigInt {
    build_bigint_24([
        10611, 52421, 6506, 60652, 42874, 18608, 3506, 22554, 11743, 62519,
        19841, 51043, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535, 65535, 65535, 65535,
    ])
}

// G = (Gx, Gy), the base point.
pub fn p384_base_point() -> EcPoint {
    let gx = build_bigint_24([
        2743, 29302, 24120, 14932, 10604, 48981, 62045, 21762, 10808,
        33364, 16864, 23031, 39832, 35751, 15202, 28189, 44404, 62240,
        50974, 36529, 1335, 48779, 51746, 43655,
    ])
    let gy = build_bigint_24([
        3679, 37098, 7548, 31299, 33181, 7550, 45518, 2656, 47296, 46576,
        12563, 59866, 5244, 10394, 7613, 63732, 56361, 37522, 39103, 23966,
        11375, 38438, 56906, 13847,
    ])
    EcPoint { x: gx, y: gy }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/ecdsa_p384_curve_constants.sio`
Expected: `ecdsa_p384_curve_constants: OK`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add stdlib/crypto/ecdsa_p384.sio tests/run-pass/ecdsa_p384_curve_constants.sio
git commit -m "feat(crypto): add P-384 (secp384r1) curve constants"
```

---

### Task 2: `ecdsa_p384_verify_sha384` — shared-core verifier

**Files:**
- Modify: `stdlib/crypto/ecdsa_p384.sio`
- Test: `tests/run-pass/ecdsa_p384_verify_vectors.sio`

**Interfaces:**
- Consumes: Task 1's `p384_field_prime`/`p384_curve_a`/`p384_order`/`p384_base_point`; `crypto::ecdsa_p256`'s `ec_point_add`/`ec_scalar_mul`/`ec_mod_inverse`/`ec_is_infinity`/`EcPoint`; `hash::sha384::sha384`; `asn1::der::*` (same DER reader functions `ecdsa_p256_verify_with_e` already uses: `der_reader_new`, `der_peek_tag`, `der_enter`, `der_next`, `der_read_integer`, `DER_OK`, `DER_TAG_SEQUENCE`).
- Produces: `pub fn ecdsa_p384_verify_sha384(pubkey: &EcPoint, message: &RawBuf, message_len: i64, der_signature: &RawBuf, der_signature_len: i64) -> bool with IO`. Consumed by Task 5.

- [ ] **Step 1: Verify no truncation is needed (do not just assert the spec's claim)**

Confirm directly: SHA-384 produces a 48-byte (384-bit) digest. `p384_order()`'s top limb (index 23) is `65535` (`0xFFFF`), meaning bits 383 down to at least 368 are all set — so `p384_order()` has bit-length exactly 384. Per FIPS 186-4 §6.4.1, when hash bit-length equals curve-order bit-length, the entire digest is used with no truncation (this is the same situation `ecdsa_p256_verify`'s own header comment documents for SHA-256/P-256, at 256 bits each). Record this check (the top-limb value and the resulting bit-length) in the task report — do not skip stating it explicitly, since a silent wrong-truncation-amount bug is exactly the class of defect D20's own header comment warns about.

- [ ] **Step 2: Write the failing test**

```sio
// tests/run-pass/ecdsa_p384_verify_vectors.sio
//@ run-pass

// Round-trip test using this file's OWN sign/verify math is not possible
// (no P-384 signing implementation exists in this codebase, deliberately
// -- this codebase only verifies, never signs, per every other ECDSA test
// in this repo). Task 6 supplies the real, openssl-independent positive
// and negative fixtures. This test only pins the DER-rejection paths that
// don't need a real valid signature: malformed DER must be rejected, not
// crash.

use crypto::ecdsa_p384::*
use crypto::ecdsa_p256::*
use bignum::bigint::*
use net::socket::*

fn main() -> i64 with IO, Mut, Div, Panic {
    let g = p384_base_point()
    let msg = rawbuf_new(16)
    rawbuf_set(&msg, 0, 104)  // 'h'
    rawbuf_set(&msg, 1, 105)  // 'i'

    // Not a valid DER SEQUENCE at all (tag byte 0x00 instead of 0x30) --
    // must return false, not crash.
    var bad_der = rawbuf_new(8)
    rawbuf_set(&bad_der, 0, 0)
    let ok = ecdsa_p384_verify_sha384(&g, &msg, 2, &bad_der, 2)
    assert(ok == false)

    println("ecdsa_p384_verify_vectors: OK")
    0
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/ecdsa_p384_verify_vectors.sio`
Expected: FAIL — `ecdsa_p384_verify_sha384` not found.

- [ ] **Step 4: Write the implementation**

Append to `stdlib/crypto/ecdsa_p384.sio`:

```sio
// Converts the 48 big-endian bytes of a SHA-384 digest into the digest
// integer `e`, in BigInt's little-endian 16-bit limb order (limb 0 =
// least significant = the LAST two bytes). No truncation -- see Task 2
// Step 1: SHA-384's 384-bit output exactly matches p384_order()'s 384-bit
// length, unlike P-256/SHA-384 which truncates 48 bytes down to 32.
fn p384_e_from_be48(digest48: &[u8; 48]) -> BigInt {
    var e = bigint_zero()
    var ei: i64 = 0
    while ei < 24 {
        let hi_idx = 46 - 2 * ei
        let lo_idx = 47 - 2 * ei
        let hi: i64 = digest48[hi_idx as usize] as i64
        let lo: i64 = digest48[lo_idx as usize] as i64
        e.limbs[ei as usize] = ((hi << 8) | lo) as u16
        ei = ei + 1
    }
    e.len = 24
    while e.len > 0 && e.limbs[(e.len - 1) as usize] == 0 {
        e.len = e.len - 1
    }
    e
}

// The ECDSA verification core, shared by (currently the only) public entry
// point below -- mirrors ecdsa_p256_verify_with_e's structure exactly
// (stdlib/crypto/ecdsa_p256.sio), parameterized by P-384's own curve
// constants instead of P-256's.
fn ecdsa_p384_verify_with_e(
    pubkey: &EcPoint,
    der_signature: &RawBuf,
    der_signature_len: i64,
    e: &BigInt,
) -> bool with IO {
    let top = der_reader_new(der_signature, 0, der_signature_len)
    let (seq_tag, seq_status) = der_peek_tag(&top)
    if seq_status != DER_OK {
        return false
    }
    if seq_tag.tag_number != DER_TAG_SEQUENCE || seq_tag.constructed == false {
        return false
    }
    let (body, enter_status) = der_enter(&top, &seq_tag)
    if enter_status != DER_OK {
        return false
    }

    let (r_tag, r_tag_status) = der_peek_tag(&body)
    if r_tag_status != DER_OK {
        return false
    }
    let (r, r_status) = der_read_integer(&body, &r_tag)
    if r_status != DER_OK {
        return false
    }
    let (body_after_r, next_status) = der_next(&body, &r_tag)
    if next_status != DER_OK {
        return false
    }

    let (s_tag, s_tag_status) = der_peek_tag(&body_after_r)
    if s_tag_status != DER_OK {
        return false
    }
    let (s, s_status) = der_read_integer(&body_after_r, &s_tag)
    if s_status != DER_OK {
        return false
    }

    let n = p384_order()
    let p = p384_field_prime()
    let curve_a = p384_curve_a()
    let g = p384_base_point()

    let one = bigint_from_u32(1)
    if bigint_cmp(&r, &one) < 0 || bigint_cmp(&r, &n) >= 0 {
        return false
    }
    if bigint_cmp(&s, &one) < 0 || bigint_cmp(&s, &n) >= 0 {
        return false
    }

    let w = ec_mod_inverse(&s, &n)
    let u1 = bigint_mod(&bigint_mul(e, &w), &n)
    let u2 = bigint_mod(&bigint_mul(&r, &w), &n)

    let p1 = ec_scalar_mul(&u1, &g, &curve_a, &p)
    let p2 = ec_scalar_mul(&u2, pubkey, &curve_a, &p)
    let sum_point = ec_point_add(&p1, &p2, &p)
    if ec_is_infinity(&sum_point) {
        return false
    }

    let v = bigint_mod(&sum_point.x, &n)
    bigint_cmp(&v, &r) == 0
}

// ECDSA-with-SHA-384 signature verification over P-384 (FIPS 186-4
// SS6.4.2 / SEC1 SS4.1.4). `der_signature` must be a DER
// SEQUENCE { r INTEGER, s INTEGER } (RFC 3279 SS2.2.3).
//
// Wire use: X.509 certificates signed ecdsa-with-SHA384 (OID
// 1.2.840.10045.4.3.3) whose ISSUER key is P-384 -- e.g. SSL.com's ECC
// intermediates (Cloudflare's 1.1.1.1 chain among them).
pub fn ecdsa_p384_verify_sha384(
    pubkey: &EcPoint,
    message: &RawBuf,
    message_len: i64,
    der_signature: &RawBuf,
    der_signature_len: i64,
) -> bool with IO {
    let digest = sha384(message, message_len)
    let e = p384_e_from_be48(&digest)
    ecdsa_p384_verify_with_e(pubkey, der_signature, der_signature_len, &e)
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/ecdsa_p384_verify_vectors.sio`
Expected: `ecdsa_p384_verify_vectors: OK`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add stdlib/crypto/ecdsa_p384.sio tests/run-pass/ecdsa_p384_verify_vectors.sio
git commit -m "feat(crypto): add ecdsa_p384_verify_sha384"
```

---

### Task 3: `oid_secp384r1()` — the P-384 named-curve OID

**Files:**
- Modify: `stdlib/x509/oid.sio`
- Modify: `stdlib/x509/oid.sio` (same file — new `oid_eq5` comparator)
- Test: `tests/run-pass/oid_secp384r1.sio`

**Interfaces:**
- Consumes: nothing new.
- Produces: `pub fn oid_secp384r1() -> [u8; 5]`, `pub fn oid_eq5(a: &[u8; 20], a_len: i32, b: &[u8; 5]) -> bool`. Consumed by Task 4.

- [ ] **Step 1: Independently re-derive and re-confirm the OID bytes**

secp384r1's curve-identifier OID is `1.3.132.0.34` (SEC 2's own `ansip384r1` identifier — note this is arc `1.3.132`, a DIFFERENT arc from P-256's `1.2.840.10045.3.1.7`, so none of the existing `2A 86 48 CE 3D` prefix bytes apply here). Derive the DER content bytes by hand using the same base-128 rule `stdlib/x509/oid.sio`'s own header comment documents (`first two arcs combined as 40*X+Y, each later arc base-128 with high-bit continuation`):

- Arcs 1, 3 → `40*1 + 3 = 43 = 0x2B`
- Arc 132 → `132 = 1*128 + 4`, base-128 with continuation: `0x81, 0x04`
- Arc 0 → `0x00`
- Arc 34 → `0x22`
- Content: `2B 81 04 00 22` (5 bytes)

Then independently confirm this against a live source — generate a real P-384 self-signed certificate and dump its raw DER:

```bash
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:secp384r1 \
  -keyout /tmp/p384_key.pem -out /tmp/p384_cert.pem -days 1 -nodes \
  -subj "/CN=p384-oid-check"
openssl asn1parse -in /tmp/p384_cert.pem -i | grep -A2 "OBJECT.*ecPublicKey" 
openssl x509 -in /tmp/p384_cert.pem -outform DER -out /tmp/p384_cert.der
openssl asn1parse -inform DER -in /tmp/p384_cert.der -i
```

Find the OID immediately following `id-ecPublicKey`'s OID in the `SubjectPublicKeyInfo.algorithm` field (this is the curve OID) and confirm its raw bytes (visible in `asn1parse`'s hex dump, or re-dump with `-dump`) exactly match `2B 81 04 00 22`. Record the full transcript in the task report. If they disagree, OpenSSL's live output is the authority — use it instead.

- [ ] **Step 2: Write the failing test**

```sio
// tests/run-pass/oid_secp384r1.sio
//@ run-pass

use x509::oid::*

fn main() -> i64 with IO, Mut, Div, Panic {
    let oid = oid_secp384r1()
    assert(oid[0] == 0x2B)
    assert(oid[1] == 0x81)
    assert(oid[2] == 0x04)
    assert(oid[3] == 0x00)
    assert(oid[4] == 0x22)

    var buf20: [u8; 20] = [0; 20]
    buf20[0] = 0x2B
    buf20[1] = 0x81
    buf20[2] = 0x04
    buf20[3] = 0x00
    buf20[4] = 0x22
    assert(oid_eq5(&buf20, 5, &oid) == true)

    // Wrong length must not match.
    assert(oid_eq5(&buf20, 4, &oid) == false)
    // Wrong bytes must not match.
    var buf20_wrong: [u8; 20] = [0; 20]
    buf20_wrong[0] = 0x2B
    buf20_wrong[1] = 0x81
    buf20_wrong[2] = 0x04
    buf20_wrong[3] = 0x00
    buf20_wrong[4] = 0x23
    assert(oid_eq5(&buf20_wrong, 5, &oid) == false)

    println("oid_secp384r1: OK")
    0
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/oid_secp384r1.sio`
Expected: FAIL — `oid_secp384r1` not found.

- [ ] **Step 4: Write the implementation**

Append to `stdlib/x509/oid.sio`, near `oid_prime256v1()`:

```sio
// secp384r1/ansip384r1 named-curve OID = 1.3.132.0.34 (SEC 2), a
// DIFFERENT arc from the 1.2.840.10045.x prefix oid_ec_public_key()/
// oid_prime256v1() share -- independently re-derived via the standard
// ASN.1 OID byte-encoding rule (see this file's own header comment) AND
// cross-checked against a real openssl-generated P-384 certificate's own
// DER bytes (this task's own report has the full transcript):
// secp384r1 = 1.3.132.0.34 -> 2B 81 04 00 22 (5 bytes).
pub fn oid_secp384r1() -> [u8; 5] {
    var r: [u8; 5] = [0; 5]
    r[0] = 0x2B; r[1] = 0x81; r[2] = 0x04; r[3] = 0x00; r[4] = 0x22
    r
}

pub fn oid_eq5(a: &[u8; 20], a_len: i32, b: &[u8; 5]) -> bool {
    if a_len != 5 {
        return false
    }
    var i: i32 = 0
    while i < 5 {
        if a[i as usize] != b[i as usize] {
            return false
        }
        i = i + 1
    }
    true
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/oid_secp384r1.sio`
Expected: `oid_secp384r1: OK`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add stdlib/x509/oid.sio tests/run-pass/oid_secp384r1.sio
git commit -m "feat(x509): add secp384r1 curve OID"
```

---

### Task 4: `PUBKEY_ALG_EC_P384` + 97-byte uncompressed-point parsing

**Files:**
- Modify: `stdlib/x509/cert.sio:44-46` (new constant), `:571-649` (parsing branch), near `:456` (new byte-to-BigInt helper)
- Test: `tests/run-pass/x509_parse_p384_public_key.sio`

**Interfaces:**
- Consumes: Task 3's `oid_secp384r1()`/`oid_eq5`; `crypto::ecdsa_p256::EcPoint`; `bignum::bigint::*`.
- Produces: `pub const PUBKEY_ALG_EC_P384: i32 = 3`; `Certificate.public_key_algorithm` can now be `PUBKEY_ALG_EC_P384`, with `Certificate.ec_public_key: EcPoint` populated the same way it already is for P-256 (same field, no struct change — `EcPoint`'s `x`/`y: BigInt` fields are curve-generic). Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

This test needs a real P-384 certificate's raw DER bytes. Generate one and convert it to a Sounio byte-array literal with a short script — same approach `tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`'s own header comment documents (DER bytes "converted to Sounio byte array literals by a script, never retyped by hand"):

```bash
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:secp384r1 \
  -keyout /tmp/p384_key.pem -out /tmp/p384_cert.pem -days 1 -nodes \
  -subj "/CN=p384-parse-test"
openssl x509 -in /tmp/p384_cert.pem -outform DER -out /tmp/p384_cert.der
python3 -c "
data = open('/tmp/p384_cert.der', 'rb').read()
print(f'// {len(data)} bytes')
print(', '.join(str(b) for b in data))
"
```

Also print the certificate's own public key coordinates independently, to assert against in the test:

```bash
openssl x509 -in /tmp/p384_cert.pem -pubkey -noout | \
  openssl ec -pubin -text -noout | grep -A6 "pub:"
```

Write the test using the printed DER bytes (replace `<DER_BYTES_HERE>` with the actual script output) and the printed X/Y coordinates (as hex, converted to decimal or compared as the raw bytes at the known offset):

```sio
// tests/run-pass/x509_parse_p384_public_key.sio
//@ run-pass

// Pins parsing of a real, openssl-generated P-384 SubjectPublicKeyInfo:
// PUBKEY_ALG_EC_P384 recognition and the 97-byte uncompressed-point
// (0x04 || 48-byte X || 48-byte Y) parse. Fixture generation transcript
// in this task's own report.

use x509::cert::*
use net::socket::*

const CERT_DER: [u8; <LEN>] = [<DER_BYTES_HERE>]

fn main() -> i64 with IO, Mut, Div, Panic {
    let buf = rawbuf_new(<LEN>)
    var i: i64 = 0
    while i < <LEN> {
        rawbuf_set(&buf, i, CERT_DER[i as usize] as i64)
        i = i + 1
    }

    let (cert, after, status) = x509_parse_certificate(&buf, 0, <LEN>)
    assert(status == X509_OK)
    assert(cert.public_key_algorithm == PUBKEY_ALG_EC_P384)

    println("x509_parse_p384_public_key: OK")
    0
}
```

(The implementer fills in `<LEN>`/`<DER_BYTES_HERE>` from the actual script output — this is fixture generation, not a placeholder in the sense the "No Placeholders" rule forbids, since the exact procedure and expected shape are fully specified above; the concrete bytes are only knowable once actually generated.)

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_parse_p384_public_key.sio`
Expected: FAIL — `PUBKEY_ALG_EC_P384` not found (constant doesn't exist yet), or `cert.public_key_algorithm == PUBKEY_ALG_UNKNOWN` (parsing falls through since the curve OID isn't recognized).

- [ ] **Step 3: Write the implementation**

In `stdlib/x509/cert.sio`, add the constant next to the existing three (line ~46):

```sio
pub const PUBKEY_ALG_RSA: i32 = 0
pub const PUBKEY_ALG_EC_P256: i32 = 1
pub const PUBKEY_ALG_UNKNOWN: i32 = 2
pub const PUBKEY_ALG_EC_P384: i32 = 3
```

Add a 48-byte-to-BigInt helper near the existing `bigint_from_be_bytes32` (line ~456), mirroring it exactly but for 48 bytes / 24 limbs:

```sio
fn bigint_from_be_bytes48(bytes: &[u8; 48]) -> BigInt {
    var result = bigint_zero()
    var limb_idx: i64 = 0
    var byte_pos: i64 = 46
    while byte_pos >= 0 {
        let hi = bytes[byte_pos as usize] as u16
        let lo = bytes[(byte_pos + 1) as usize] as u16
        result.limbs[limb_idx as usize] = (hi << 8) | lo
        limb_idx = limb_idx + 1
        byte_pos = byte_pos - 2
    }
    var len: i32 = 24
    while len > 0 && result.limbs[(len - 1) as usize] == 0 {
        len = len - 1
    }
    result.len = len
    result
}
```

Extend the curve-OID recognition block (lines 579–596 currently only checks `is_p256_curve`) to also compute `is_p384_curve`:

```sio
    var is_p256_curve = false
    var is_p384_curve = false
    if is_ec_alg {
        let (after_alg_oid, s_after_alg_oid) = der_next(&alg_id_inner, &alg_oid_tag)
        if s_after_alg_oid != DER_OK { return (cert, after_subject, s_after_alg_oid) }
        let (curve_r, curve_oid_tag, s_curve) = der_read_tlv(&after_alg_oid)
        if s_curve != DER_OK { return (cert, after_subject, s_curve) }
        if curve_oid_tag.tag_number == DER_TAG_OBJECT_IDENTIFIER {
            var curve_oid_buf: [u8; 20] = [0; 20]
            var curve_oi: i64 = 0
            while curve_oi < curve_oid_tag.content_len && curve_oi < 20 {
                curve_oid_buf[curve_oi as usize] = (rawbuf_get(buf, curve_oid_tag.content_start + curve_oi) & 255) as u8
                curve_oi = curve_oi + 1
            }
            let curve_oid_len = curve_oid_tag.content_len as i32
            let p256_oid = oid_prime256v1()
            let p384_oid = oid_secp384r1()
            is_p256_curve = oid_eq8(&curve_oid_buf, curve_oid_len, &p256_oid)
            is_p384_curve = oid_eq5(&curve_oid_buf, curve_oid_len, &p384_oid)
        }
    }
```

Extend the point-parsing branch (lines 627–646) to add a parallel P-384 arm — uncompressed point is `0x04 || X (48 bytes) || Y (48 bytes)` = 97 bytes total per SEC1 §2.3.3, same shape as P-256's 65-byte case, just wider:

```sio
    } else if is_ec_alg && is_p256_curve {
        // Uncompressed point per SEC1 SS2.3.3: 0x04 || X (32 bytes) || Y (32
        // bytes) -- a compressed point (0x02/0x03 prefix) is a valid DER
        // encoding this task does not support (per the task brief: not what
        // OpenSSL or any modern CA/server emits by default). Leave
        // public_key_algorithm at UNKNOWN rather than guess at decoding it.
        let point_prefix = rawbuf_get(buf, key_start) & 255
        if point_prefix == 4 && key_len == 65 {
            var x_bytes: [u8; 32] = [0; 32]
            var y_bytes: [u8; 32] = [0; 32]
            var bi: i64 = 0
            while bi < 32 {
                x_bytes[bi as usize] = (rawbuf_get(buf, key_start + 1 + bi) & 255) as u8
                y_bytes[bi as usize] = (rawbuf_get(buf, key_start + 1 + 32 + bi) & 255) as u8
                bi = bi + 1
            }
            cert.ec_public_key = EcPoint { x: bigint_from_be_bytes32(&x_bytes), y: bigint_from_be_bytes32(&y_bytes) }
            cert.public_key_algorithm = PUBKEY_ALG_EC_P256
        }
    } else if is_ec_alg && is_p384_curve {
        // Same uncompressed-point shape as the P-256 arm above, sized for
        // P-384's 48-byte coordinates: 0x04 || X (48 bytes) || Y (48
        // bytes) = 97 bytes total.
        let point_prefix2 = rawbuf_get(buf, key_start) & 255
        if point_prefix2 == 4 && key_len == 97 {
            var x_bytes2: [u8; 48] = [0; 48]
            var y_bytes2: [u8; 48] = [0; 48]
            var bi2: i64 = 0
            while bi2 < 48 {
                x_bytes2[bi2 as usize] = (rawbuf_get(buf, key_start + 1 + bi2) & 255) as u8
                y_bytes2[bi2 as usize] = (rawbuf_get(buf, key_start + 1 + 48 + bi2) & 255) as u8
                bi2 = bi2 + 1
            }
            cert.ec_public_key = EcPoint { x: bigint_from_be_bytes48(&x_bytes2), y: bigint_from_be_bytes48(&y_bytes2) }
            cert.public_key_algorithm = PUBKEY_ALG_EC_P384
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_parse_p384_public_key.sio`
Expected: `x509_parse_p384_public_key: OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/x509/cert.sio tests/run-pass/x509_parse_p384_public_key.sio
git commit -m "feat(x509): parse P-384 EC public keys (PUBKEY_ALG_EC_P384)"
```

---

### Task 5: Wire P-384 into `x509_verify_signature`, extend the algorithm-confusion guard

**Files:**
- Modify: `stdlib/x509/cert.sio:1584` (guard), `:1622-1625` (dispatch)
- Test: `tests/run-pass/x509_signature_p384_algorithm_confusion.sio`

**Interfaces:**
- Consumes: Task 2's `ecdsa_p384_verify_sha384`; Task 4's `PUBKEY_ALG_EC_P384`.
- Produces: `x509_verify_signature` now accepts a P-384 issuer signing `ecdsa-with-SHA384`. Consumed by Task 6's end-to-end chain tests.

- [ ] **Step 1: Write the failing test**

This is a pure algorithm-confusion unit test — it does not need real signature bytes, only a `Certificate` struct with a chosen `public_key_algorithm` and a chosen `outer_sig_alg_oid`, to prove the guard logic itself (not full chain verification, which Task 6 covers end-to-end):

```sio
// tests/run-pass/x509_signature_p384_algorithm_confusion.sio
//@ run-pass

// D20 added a guard: an ecdsa-with-SHA384-signed certificate is rejected
// unless the issuer's key algorithm is one this implementation actually
// verifies against. Before this task, ANY issuer other than P-256 was
// rejected (P-384 included, which was simply unimplemented). This test
// pins that after Task 5, P-384 issuers are ACCEPTED into the dispatch
// (reaching ecdsa_p384_verify_sha384, which then fails closed on a
// garbage signature -- proving the guard now lets P-384 through to real
// verification, not that a garbage signature ever "verifies").
//
// It also re-confirms the ORIGINAL guard property still holds both ways:
// an RSA issuer claiming to have signed ecdsa-with-SHA384 must still be
// rejected before reaching ANY curve's arithmetic.

use x509::cert::*
use net::socket::*

fn main() -> i64 with IO, Mut, Div, Panic {
    // We cannot easily hand-construct a full Certificate with a valid
    // outer_sig_alg_oid/tbs_start without a real DER cert (those fields
    // are populated by x509_parse_certificate, not settable standalone
    // in a meaningful way) -- so this task's OWN adversarial proof is
    // Task 6's real end-to-end fixtures (a P-384-issued cert accepted,
    // and a mismatched-curve cert rejected). This file instead pins the
    // constant-level contract the guard now depends on, so a future edit
    // removing PUBKEY_ALG_EC_P384 from the guard's accepted set is caught
    // immediately without needing network/openssl fixtures.
    assert(PUBKEY_ALG_EC_P384 != PUBKEY_ALG_EC_P256)
    assert(PUBKEY_ALG_EC_P384 != PUBKEY_ALG_RSA)
    assert(PUBKEY_ALG_EC_P384 != PUBKEY_ALG_UNKNOWN)

    println("x509_signature_p384_algorithm_confusion: OK (see Task 6 for the real end-to-end adversarial proof)")
    0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_signature_p384_algorithm_confusion.sio`
Expected: PASS already (this file only checks constants from Task 4) — confirms Task 4 landed correctly before Task 5's own wiring change. This is expected; proceed to Step 3's real change and Task 6's real adversarial fixtures.

- [ ] **Step 3: Write the implementation**

In `stdlib/x509/cert.sio`, extend the guard at line 1584 (currently rejects every non-P256 issuer) to also accept P-384:

```sio
    // Same guard for ecdsa-with-SHA384, extended (2026-08-27) to accept
    // BOTH curves this implementation verifies: P-256 (D20) and P-384
    // (this task). An issuer key that is neither is still rejected here,
    // before reaching either curve's arithmetic. Note this constrains the
    // ISSUER'S KEY, not the hash: per RFC 5480 SS2 the signature hash and
    // the key's curve are independent in X.509, so P-256+SHA-384 AND
    // P-384+SHA-384 are both valid, deployed combinations (D15, and
    // Cloudflare's own 1.1.1.1 intermediate for the P-384 case).
    if is_ecdsa_sha384_sig_alg
        && issuer.public_key_algorithm != PUBKEY_ALG_EC_P256
        && issuer.public_key_algorithm != PUBKEY_ALG_EC_P384 {
        return false
    }
```

Extend the dispatch at lines 1622–1625 to call the correct curve's verifier based on the issuer's ACTUAL key algorithm (not just the signature's hash OID — a P-256 issuer must still only ever go through `ecdsa_p256_verify_sha384`, and a P-384 issuer only through `ecdsa_p384_verify_sha384`):

```sio
        if is_ecdsa_sha384_sig_alg {
            if issuer.public_key_algorithm == PUBKEY_ALG_EC_P384 {
                return ecdsa_p384_verify_sha384(&issuer.ec_public_key, &tbs_copy, cert.tbs_len, &esig_buf, cert.outer_signature_len as i64)
            }
            return ecdsa_p256_verify_sha384(&issuer.ec_public_key, &tbs_copy, cert.tbs_len, &esig_buf, cert.outer_signature_len as i64)
        }
        return ecdsa_p256_verify(&issuer.ec_public_key, &tbs_copy, cert.tbs_len, &esig_buf, cert.outer_signature_len as i64)
```

(The guard above already ensures `issuer.public_key_algorithm` is P-256 or P-384 by the time this code runs, so the `if`/`else` here is a safe, total dispatch — no third case can reach it.)

Add the import at the top of `stdlib/x509/cert.sio` if not already present (check the existing `use crypto::ecdsa_p256::*` line and add a sibling):

```sio
use crypto::ecdsa_p384::ecdsa_p384_verify_sha384
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_signature_p384_algorithm_confusion.sio`
Expected: `x509_signature_p384_algorithm_confusion: OK`, exit 0. Also re-run the existing D20 tests to confirm the P-256 path is untouched:

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`
Expected: still passes, unchanged from before this task.

- [ ] **Step 5: Commit**

```bash
git add stdlib/x509/cert.sio tests/run-pass/x509_signature_p384_algorithm_confusion.sio
git commit -m "feat(x509): wire P-384 into x509_verify_signature, extend algorithm-confusion guard"
```

---

### Task 6: Real end-to-end chain tests, adversarial cross-curve rejection, and full regression

**Files:**
- Create: `tests/run-pass/x509_chain_ecdsa_p384_signature.sio`
- Create: `tests/run-pass/x509_chain_ecdsa_p384_cross_curve_rejected.sio`
- Modify: `tests/interop/tls_connect_1111_sha384_probe.sio` (or create `tests/interop/tls_connect_1111_p384_probe.sio` if the existing file is scoped narrowly to the SHA-384-only diagnosis — check its current content first)

**Interfaces:**
- Consumes: Task 5's fully-wired `x509_verify_signature`.
- Produces: final proof the whole plan works — no new public interface.

- [ ] **Step 1: Generate a real P-384 chain and its openssl-independent verification**

```bash
mkdir -p /tmp/p384_chain && cd /tmp/p384_chain

openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:secp384r1 \
  -sha384 -keyout root_key.pem -out root_cert.pem -days 3650 -nodes \
  -subj "/CN=p384-root-test" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout leaf_key.pem -out leaf.csr -nodes -subj "/CN=p384-leaf-test"

cat > leaf_ext.cnf <<'EOF'
basicConstraints=CA:FALSE
subjectAltName=DNS:p384-leaf-test.example
EOF

openssl x509 -req -in leaf.csr -CA root_cert.pem -CAkey root_key.pem \
  -CAcreateserial -out leaf_cert.pem -days 3650 -sha384 \
  -extfile leaf_ext.cnf

# Independent confirmation the chain is genuinely valid before trusting
# this implementation's own verdict:
openssl verify -CAfile root_cert.pem leaf_cert.pem

openssl x509 -in root_cert.pem -outform DER -out root_cert.der
openssl x509 -in leaf_cert.pem -outform DER -out leaf_cert.der
openssl x509 -in root_cert.pem -noout -text | grep -A1 "Public Key Algorithm\|Signature Algorithm"
openssl x509 -in leaf_cert.pem -noout -text | grep -A1 "Public Key Algorithm\|Signature Algorithm"
```

Confirm: `openssl verify` prints `leaf_cert.pem: OK`; the root's Public Key Algorithm is `id-ecPublicKey` with `ASN1 OID: secp384r1`; both certs' Signature Algorithm is `ecdsa-with-SHA384`. The leaf intentionally uses P-256 (matching the real Cloudflare shape this plan targets: P-256 leaf, P-384 issuer) — this proves the guard correctly lets a P-256-keyed cert be *issued by* a P-384 signer, which is exactly Task 5's dispatch-by-issuer-algorithm logic being exercised for real.

- [ ] **Step 2: Write the positive end-to-end test**

Convert both DER files to Sounio byte arrays with the same script pattern as Task 4 Step 1, then write, mirroring `tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`'s own structure exactly (that file's full content is the template — read it before writing this one, since its harness calls `x509_parse_certificate` on both DER blobs then `x509_verify_signature(&leaf_buf, &leaf_cert, &root_cert)` and asserts `true`):

```sio
// tests/run-pass/x509_chain_ecdsa_p384_signature.sio
//@ run-pass

// This task's closeout: `x509_verify_signature` can now validate a
// certificate signed ecdsa-with-SHA384 by a P-384 issuer key -- the exact
// shape of Cloudflare's real 1.1.1.1:443 intermediate (see
// docs/superpowers/specs/2026-08-27-madaros-ecdsa-p384-design.md).
//
// Fixture generation transcript: see this task's own report (Task 6 Step
// 1) -- a real openssl-generated P-384 root signing a P-256 leaf,
// independently confirmed valid via `openssl verify`.

use x509::cert::*
use net::socket::*

const ROOT_DER: [u8; <ROOT_LEN>] = [<ROOT_DER_BYTES>]
const LEAF_DER: [u8; <LEAF_LEN>] = [<LEAF_DER_BYTES>]

fn main() -> i64 with IO, Mut, Div, Panic {
    let root_buf = rawbuf_new(<ROOT_LEN>)
    var i: i64 = 0
    while i < <ROOT_LEN> {
        rawbuf_set(&root_buf, i, ROOT_DER[i as usize] as i64)
        i = i + 1
    }
    let leaf_buf = rawbuf_new(<LEAF_LEN>)
    var j: i64 = 0
    while j < <LEAF_LEN> {
        rawbuf_set(&leaf_buf, j, LEAF_DER[j as usize] as i64)
        j = j + 1
    }

    let (root_cert, _ra, root_status) = x509_parse_certificate(&root_buf, 0, <ROOT_LEN>)
    assert(root_status == X509_OK)
    assert(root_cert.public_key_algorithm == PUBKEY_ALG_EC_P384)

    let (leaf_cert, _la, leaf_status) = x509_parse_certificate(&leaf_buf, 0, <LEAF_LEN>)
    assert(leaf_status == X509_OK)

    let verified = x509_verify_signature(&leaf_buf, &leaf_cert, &root_cert)
    assert(verified == true)

    println("x509_chain_ecdsa_p384_signature: OK")
    0
}
```

- [ ] **Step 3: Write the adversarial cross-curve-rejection test**

Take the SAME leaf DER bytes from Step 2, but verify them against a DIFFERENT, unrelated P-256 root (any existing P-256 root fixture already in the repo, e.g. reuse `tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`'s own root bytes, or generate a fresh unrelated one) — this must be rejected, proving a P-384-signed certificate cannot be waved through by a mismatched P-256 key:

```sio
// tests/run-pass/x509_chain_ecdsa_p384_cross_curve_rejected.sio
//@ run-pass

// Adversarial proof that Task 5's dispatch is genuinely gated on the
// ACTUAL issuer key algorithm, not merely on the signature's hash OID:
// the real P-384-signed leaf from x509_chain_ecdsa_p384_signature.sio,
// verified against an UNRELATED P-256 root, must be rejected -- proving
// this is not "any EC key accepted for ecdsa-with-SHA384" but a genuine
// per-curve check with real curve arithmetic underneath (a P-256 point
// run through P-384's group law does not satisfy the verification
// equation for arbitrary r/s, and even if it coincidentally would, the
// keys/signature here are from entirely unrelated certificates).

use x509::cert::*
use net::socket::*

// Same LEAF_DER as x509_chain_ecdsa_p384_signature.sio.
const LEAF_DER: [u8; <LEAF_LEN>] = [<LEAF_DER_BYTES>]
// An UNRELATED, real P-256 root -- reuse the root bytes already fixtured
// in tests/run-pass/x509_chain_ecdsa_sha384_signature.sio (read that
// file for its own ROOT_DER constant and copy its byte array here).
const UNRELATED_P256_ROOT_DER: [u8; <ROOT2_LEN>] = [<ROOT2_DER_BYTES>]

fn main() -> i64 with IO, Mut, Div, Panic {
    let leaf_buf = rawbuf_new(<LEAF_LEN>)
    var j: i64 = 0
    while j < <LEAF_LEN> {
        rawbuf_set(&leaf_buf, j, LEAF_DER[j as usize] as i64)
        j = j + 1
    }
    let root2_buf = rawbuf_new(<ROOT2_LEN>)
    var k: i64 = 0
    while k < <ROOT2_LEN> {
        rawbuf_set(&root2_buf, k, UNRELATED_P256_ROOT_DER[k as usize] as i64)
        k = k + 1
    }

    let (leaf_cert, _la, leaf_status) = x509_parse_certificate(&leaf_buf, 0, <LEAF_LEN>)
    assert(leaf_status == X509_OK)
    let (root2_cert, _r2a, root2_status) = x509_parse_certificate(&root2_buf, 0, <ROOT2_LEN>)
    assert(root2_status == X509_OK)
    assert(root2_cert.public_key_algorithm == PUBKEY_ALG_EC_P256)

    let verified = x509_verify_signature(&leaf_buf, &leaf_cert, &root2_cert)
    assert(verified == false)

    println("x509_chain_ecdsa_p384_cross_curve_rejected: OK")
    0
}
```

- [ ] **Step 4: Run both new tests, confirm pass**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_chain_ecdsa_p384_signature.sio`
Expected: `x509_chain_ecdsa_p384_signature: OK`, exit 0.

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_chain_ecdsa_p384_cross_curve_rejected.sio`
Expected: `x509_chain_ecdsa_p384_cross_curve_rejected: OK`, exit 0.

- [ ] **Step 5: Full regression, before/after comparison**

```bash
git stash push -u -m "p384-plan-baseline-check"
bash scripts/run_sio_test_suite.sh x509 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh tls 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh crypto 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh pem 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh asn1 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh bignum 2>&1 | tail -20
git stash pop
bash scripts/run_sio_test_suite.sh x509 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh tls 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh crypto 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh pem 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh asn1 2>&1 | tail -20
bash scripts/run_sio_test_suite.sh bignum 2>&1 | tail -20
```

Expected: pass/fail counts identical before and after for every suite prefix, except `x509` gaining exactly the new tests added in Tasks 1–6 (all passing) and no other change.

**Note on `git stash`:** per this workspace's own shared-stash-stack discipline, use `git stash push -u -m "p384-plan-baseline-check"`, then immediately `git stash list --format='%H %gs'` to capture this entry's SHA, and restore with `git stash apply <sha>` (not bare `pop`) followed by `git stash drop <sha>` — never bare `git stash pop`, since the stash stack is shared with other concurrent sessions.

- [ ] **Step 6: Live retest against the real motivating case (if network-reachable)**

If this execution environment has outbound internet access, re-run (or write, if it does not already exist in the form needed) a live probe against `1.1.1.1:443`, mirroring `tests/interop/tls_connect_1111_sha384_probe.sio`'s own existing structure — read that file first to see its current probe shape and extend it, or add a sibling `tests/interop/tls_connect_1111_p384_probe.sio` if the existing file is narrowly scoped to only the SHA-384 diagnosis and not the full chain:

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/interop/tls_connect_1111_sha384_probe.sio` (or the new sibling file)
Expected: `tls_connect` now returns `0` (success) where it previously returned `TLS_CONNECT_ERR_CERT_CHAIN` — the P-256 leaf + P-384 intermediate chain now validates end-to-end. If this environment has no live internet egress at execution time, note this explicitly in the task report as "not executed, no network reachability" rather than skipping the note — do not claim this step succeeded without having actually run it.

- [ ] **Step 7: Update the defect log and commit**

Add a closing note to `docs/handoff/souc_v0800_defects.md`'s D15/D20 entries (or a new entry if the file's own convention calls for one — read its existing D20 entry's structure first) stating that the residual P-384 gap identified during D20's investigation is now closed by this plan, with a pointer to this plan's spec and the new tests.

```bash
git add tests/run-pass/x509_chain_ecdsa_p384_signature.sio \
        tests/run-pass/x509_chain_ecdsa_p384_cross_curve_rejected.sio \
        docs/handoff/souc_v0800_defects.md
# If Step 6 modified/added an interop probe file, add it too:
# git add tests/interop/tls_connect_1111_sha384_probe.sio (or the new sibling)
git commit -m "test(x509): pin real P-384 chain verification + adversarial cross-curve rejection

Closes the residual P-384 gap D20 identified: Cloudflare's 1.1.1.1
intermediate uses a P-384 key, now verifiable end to end."
```

---

## Self-Review Notes (writing-plans skill, run against this plan before handoff)

**Spec coverage:** §3 (curve constants) → Task 1. §2 (shared-core verifier, digest handling) → Task 2. §4 (OID) → Task 3. §5 (public-key parsing, `PUBKEY_ALG_EC_P384`) → Task 4. §6 (signature-verification wiring, guard extension) → Task 5. §Testing strategy (real openssl chain, adversarial mutation/cross-curve, regression, live retest) → Task 6. §Non-goals (TLS 1.3 `CertificateVerify`) — explicitly excluded, stated in Global Constraints; no task touches `stdlib/tls/client.sio`.

**Placeholder scan:** the only bracketed placeholders (`<LEN>`, `<DER_BYTES_HERE>`, etc.) appear exclusively where the underlying value is a real, openssl-generated artifact that does not exist until the implementer runs the specified command — every such spot names the exact command to run and the exact shape of the substitution, which is fixture generation, not an unspecified requirement. No task contains "add appropriate error handling," "similar to Task N," or an undefined type/function reference — every function a later task consumes is fully defined, with its complete body, in an earlier task.

**Type consistency:** `EcPoint`, `BigInt`, `PUBKEY_ALG_EC_P384: i32`, `ecdsa_p384_verify_sha384`'s signature, and `oid_secp384r1() -> [u8; 5]` are each defined once (Tasks 1/2/4/3 respectively) and referenced identically by every later task that consumes them.

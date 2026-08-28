<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-23-madaros-hash-functions-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-23-madaros-hash-functions-plan
-->

# Madaros Hash Functions (SHA-1/256/384/512) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `stdlib/hash/{word32,sha1,sha256,word64,sha512_core,sha384,sha512}.sio` — SHA-1, SHA-256, SHA-384, and SHA-512, each tested against published NIST/FIPS test vectors, as a prerequisite for the X.509 semantic layer's PKCS#1 v1.5 signature verification.

**Architecture:** Two independent phases, each opening with an empirical arithmetic audit before any hash code depends on its outcome. Phase 1 (32-bit family: SHA-1, SHA-256) audited `u32` — the audit found native `u32` `+`/`-` do not wrap mod 2^32 at all (**Finding 13**, more severe than anticipated), so Phase 1's words are represented as masked `i64` scalars, never native `u32`. Phase 2 (64-bit family: SHA-384, SHA-512, sharing one compression core) audits 64-bit arithmetic and — per this plan's own ruling below — represents every 64-bit "word" as a **pair of plain `i64` scalars** (high/low 32-bit halves), never as a struct or an array of structs, to avoid two entirely untested Sounio patterns (struct-typed arrays, and taking a reference to an array element) on top of the already-uncertain arithmetic.

**Tech Stack:** Sounio (Madaros v0.80.0). Byte input via `stdlib/net/socket.sio`'s `RawBuf`/`rawbuf_get` (already proven).

**Spec:** `docs/superpowers/specs/2026-08-23-madaros-hash-functions-design.md`

## Global Constraints

- **Module import path:** a caller outside `stdlib/` uses `use hash::sha256::*` (etc.); a sibling file inside `stdlib/hash/` uses the bare `use word32::*` form (per the audit doc's Findings 6/8).
- **Ruling — 64-bit words are `(i64, i64)` scalar pairs, not a struct.** The spec described a `U64Pair { hi, lo }` struct. This plan deviates: represent every 64-bit value as two independent plain `i64` values (high half, low half), and every `word64`-family function takes/returns them as separate scalar parameters or a `(i64, i64)` tuple — never a named struct, and never an array of such a struct. Reason: this branch has proven arrays of scalars (`BigInt.limbs: [u16;512]`) and tuples of scalars (`(TcpSocket, i64)`, `(DerReader, DerTag, i64)`) extensively, but has never once used an array of structs, or taken a reference to a single element of an array, anywhere in ~15 commits of prior work on this exact compiler. Introducing BOTH of those untested patterns simultaneously, in a plan that cannot be interactively tested before dispatch, is an avoidable risk for zero benefit — a struct wrapper adds naming convenience, not new capability. The message schedule for the 64-bit algorithms is therefore two parallel plain arrays, `w_hi: [i64; 80]` and `w_lo: [i64; 80]`, not one array of a pair type.
- **All 32-bit "words" are masked `i64` scalars, never native `u32`.** Task 1's audit already ran and found native `u32` `+`/`-` do not wrap mod 2^32 (Finding 13) — this is settled, not pending. All 64-bit "words" are `i64` scalar pairs regardless of Task 4's audit outcome, UNLESS Task 4's audit finds native 64-bit arithmetic fully safe for every operation these algorithms need, in which case the implementer may simplify to native `u64`/`i64`-native and must document that decision plus still pass every primitive-level test this plan specifies.
- **Every arithmetic step that could exceed 32 bits (or, for the scalar-pair 64-bit code, exceed the `i64` type's own bit-63 boundary) must carry a one-line comment justifying why it stays within bounds**, matching this branch's established documentation discipline (`stdlib/bignum/bigint.sio`, `stdlib/asn1/der.sio`).
- **No AI attribution in any commit message** (this repo's CLAUDE.md rule, zero exceptions).
- **Conventional-Commits-style commit messages**: `feat(hash): ...`, `test(hash): ...`, `docs(audit): ...`.
- **Test files:** `tests/run-pass/hash_*.sio`, run via `bash scripts/run_sio_test_suite.sh --filter-prefix hash_`. **Never run the whole-repo suite.**
- **Run every shell command as a plain foreground command, one at a time. Never use any Monitor/background-wait mechanism.**
- **Every test asserts a real, independently-reverified expected value** — for hash digests, this means an implementer must independently confirm each expected hex digest against its cited published source (FIPS 180-4 / RFC 3174 / a NIST test-vector page) before committing it, per this project's "measure, don't assume" discipline. This plan's own recollection of these digests (given below) is a best-effort starting point, not a substitute for that independent check.
- **Both arithmetic audits (Task 1, Task 4) get recorded as new, explicitly-numbered findings in `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`**, continuing that document's existing numbering (it currently ends at Finding 12).
- **Task 1's `u32` audit already ran and found native `u32` broken (Finding 13)** — this is why Task 1's plan text below uses masked `i64`, not native `u32`. If Task 1's `hash_word32_primitives.sio` test STILL fails against the `i64`-masked design, that is a new, second surprise beyond Finding 13 — STOP and report BLOCKED, do not invent a further workaround; plain `i64` arithmetic bounded away from bit 63 has been extensively proven correct elsewhere on this branch (`BigInt`, `word64.sio`), so a failure here would need a controller ruling.
- **No Sounio string literal in this plan (or any future code on this branch) may exceed ~120 characters.** Finding 17: literals over ~126 content characters are silently truncated at compile time (warning only, not an error) and can crash the compiler's output stage. Task 5 hit this with 128-hex-character SHA-512 digests and worked around it by splitting each into two 64-char halves compared separately. Task 6's SHA-384 digests (96 hex chars) are safely under this limit and need no such split — this note is for any future task/edit that might otherwise embed a longer literal.
- **Every narrowing `as u8` cast in this whole plan is masked with `& 255` immediately beforehand** (e.g. `(shr32(h0, 16) & 255) as u8`, never bare `shr32(h0, 16) as u8`), even where the source value is already provably ≤255. This is **Finding 14**: Task 2's implementer discovered that `i64 as u8` on Madaros does not truncate to the low 8 bits when the source exceeds 255 — the standard big-endian-byte-extraction idiom every C-family language relies on is silently wrong here. Every digest-output-writing code block in this plan (Tasks 2, 3, 5, 6) has already been updated with this masking; if you find an unmasked `as u8` cast anywhere in this plan's remaining code blocks that this revision missed, add the mask yourself rather than trusting the literal text — do not treat this omission as evidence the cast is safe in that specific spot.

## File Structure

```
stdlib/hash/word32.sio       -- Task 1: u32 primitives (add32/rotl32/rotr32/shr32/xor32/and32/or32/not32)
stdlib/hash/sha1.sio          -- Task 2: SHA-1
stdlib/hash/sha256.sio        -- Task 3: SHA-256
stdlib/hash/word64.sio        -- Task 4: 64-bit scalar-pair primitives
stdlib/hash/sha512_core.sio   -- Task 5: shared SHA-384/512 compression core
stdlib/hash/sha512.sio        -- Task 5: SHA-512 (thin wrapper over the core)
stdlib/hash/sha384.sio        -- Task 6: SHA-384 (thin wrapper over the core, different IV + truncation)
```

---

### Task 1: `word32.sio` primitives (`i64`-masked — see Finding 13 below)

**Revised after the audit ran.** The audit this task originally opened with has already been executed once (by an implementer dispatch) and found native `u32` arithmetic broken in a way more severe than anticipated: `u32 + u32` does not wrap mod 2^32 at all (`4294967295u32 + 1u32` evaluates to `4294967296`, the unbounded sum, not `0`), and `u32 - u32` underflows with signed semantics instead of wrapping to `0xFFFFFFFF`. This is now **Finding 13** in `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` (see Step 0 below — the controller independently reproduced this before revising this plan). This is a different, more fundamental failure than the bit-31-shift analogue of Finding 11 this task was designed to check for: it affects a `u32` `+`/`-` with no shift/rotate involved at all.

**Ruling:** `word32.sio` is redesigned to represent every 32-bit "word" as a plain `i64`, always masked (`& 0xFFFFFFFF`) after any operation that could exceed 32 bits (addition, left shift) — exactly the technique Task 4 already plans to use for 64-bit words (there, applied to 32-bit halves of a 64-bit value; here, applied directly to the 32-bit word itself). This is not a new, unverified technique: masked `i64` arithmetic bounded away from bit 63 is the same discipline `stdlib/bignum/bigint.sio`'s entire (already reviewed, tested, and shipped) design rests on, and `word64.sio` (Task 4) independently re-confirms it again for a different bit width. No further audit step is needed before adopting it here — the audit already happened, found `u32` untrustworthy, and the fix is a proven pattern from elsewhere on this branch, not a new gamble.

**Files:**
- Create: `stdlib/hash/word32.sio`
- Modify: `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` (Finding 13 — may already be present if the controller added it; confirm before appending a duplicate)
- Test: `tests/run-pass/hash_word32_primitives.sio`

**Interfaces:**
- Produces: `fn add32(a: i64, b: i64) -> i64`, `fn rotl32(x: i64, n: i64) -> i64`, `fn rotr32(x: i64, n: i64) -> i64`, `fn shr32(x: i64, n: i64) -> i64`, `fn xor32(a: i64, b: i64) -> i64`, `fn and32(a: i64, b: i64) -> i64`, `fn or32(a: i64, b: i64) -> i64`, `fn not32(a: i64) -> i64`. Every parameter and return value is a 32-bit word STORED in an `i64`, always in range `0..4294967295` — never a native `u32`/`u64` value.

- [ ] **Step 0: Confirm the audit finding is recorded**

Read the end of `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`. If a Finding 13 describing the `u32` non-wraparound bug is already present, proceed to Step 1. If not, append it now, continuing the document's existing numbering and formatting style (match Finding 11/12's structure: a minimal reproducer, the actual vs. expected behavior, and the workaround this task adopts). Minimal reproducer to include:

```sio
fn main() with IO {
    let x: u32 = 4294967295
    let y: u32 = 1
    let z = x + y
    if (z as i64) == 4294967296 { println("BUG: z as i64 == 4294967296 (not truncated to 32 bits)") }
    if z == 0 { println("correct: wrapped to 0") }
}
```
Actual: prints the BUG line. Expected (correct `u32` semantics): prints "correct: wrapped to 0". Workaround: represent 32-bit words as masked `i64` (this task's `word32.sio`), never as native `u32`.

- [ ] **Step 1: Write the failing primitives test**

Create `tests/run-pass/hash_word32_primitives.sio`:

```sio
//@ run-pass
use hash::word32::*

fn main() with IO {
    // Wraparound addition: the defining property of mod-2^32 arithmetic
    // every SHA round depends on. This is the exact case that failed
    // under native u32 (Finding 13) -- word32.sio's masking must fix it.
    assert(add32(4294967295, 1) == 0)
    assert(add32(1, 1) == 2)
    assert(add32(2147483648, 2147483648) == 0)   // 0x80000000 + 0x80000000 wraps to 0

    // Logical right shift at the bit-31 boundary -- the shape Finding 11
    // documents for u64 at bit 63, checked here for this module's i64-
    // masked 32-bit words. Since every value here is always 0..4294967295
    // (never negative), i64's `>>` behaves as a logical shift regardless
    // of arithmetic/logical semantics -- confirmed empirically here anyway.
    assert(shr32(2147483648, 1) == 1073741824)   // 0x80000000 >> 1 == 0x40000000
    assert(shr32(1, 1) == 0)

    // Rotations, both directions, including rotating a value whose top bit
    // is set.
    assert(rotr32(2147483648, 1) == 1073741824)  // 0x80000000 ROTR 1 == 0x40000000
    assert(rotr32(1, 1) == 2147483648)            // 1 ROTR 1 == 0x80000000
    assert(rotl32(2147483648, 1) == 1)            // 0x80000000 ROTL 1 == 1
    assert(rotl32(1, 31) == 2147483648)           // 1 ROTL 31 == 0x80000000

    // Bitwise ops at the top-bit boundary.
    assert(xor32(2147483648, 2147483648) == 0)
    assert(xor32(2147483648, 0) == 2147483648)
    assert(and32(4294967295, 2147483648) == 2147483648)
    assert(or32(0, 2147483648) == 2147483648)
    assert(not32(0) == 4294967295)
    assert(not32(4294967295) == 0)
    assert(not32(2147483648) == 2147483647)   // NOT 0x80000000 == 0x7FFFFFFF

    println("hash_word32_primitives: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/hash_word32_primitives.sio`
Expected: FAIL to compile — `stdlib/hash/word32.sio` doesn't exist yet.

- [ ] **Step 2: Implement `stdlib/hash/word32.sio`**

```sio
// stdlib/hash/word32.sio
//
// 32-bit word arithmetic primitives shared by SHA-1 and SHA-256. Every word
// is a plain i64, ALWAYS masked to 0..4294967295 after any operation that
// could exceed 32 bits -- native u32 arithmetic is NOT used here. Finding
// 13 (docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md) found
// that native u32 + and - do not wrap mod 2^32 at all on Madaros (a more
// severe failure than Finding 11's u64 shift/divide/modulo bug). This
// module sidesteps that entirely by never letting arithmetic run in the
// u32 type: every value here is bounded to 32 bits by explicit masking on
// a plain i64, the same discipline stdlib/bignum/bigint.sio's 16-bit-limb
// design and this project's own word64.sio (Task 4) both already use --
// values this small never approach i64's own bit-63 danger zone (Finding
// 11), so plain i64 +/-/shift/bitwise-ops are trustworthy here.

pub fn add32(a: i64, b: i64) -> i64 {
    (a + b) & 4294967295   // a,b are always 0..4294967295, so a+b is at
                             // most ~2^33 -- far under bit 63, masking
                             // truncates to the correct mod-2^32 result
}

pub fn rotl32(x: i64, n: i64) -> i64 {
    ((x << n) | (x >> (32 - n))) & 4294967295   // n always in [1,31] here;
                                                   // x << n can reach ~2^63... no,
                                                   // x <= 0xFFFFFFFF and n <= 31,
                                                   // so x << n <= 0xFFFFFFFF << 31,
                                                   // well under bit 63 -- mask
                                                   // discards the overflow bits
                                                   // above bit 31, which is exactly
                                                   // the rotation's intended effect
}

pub fn rotr32(x: i64, n: i64) -> i64 {
    ((x >> n) | (x << (32 - n))) & 4294967295
}

pub fn shr32(x: i64, n: i64) -> i64 {
    x >> n   // x is always 0..4294967295 (non-negative, 32-bit bounded);
              // shifting right only ever shrinks it, never approaches bit 63
}

pub fn xor32(a: i64, b: i64) -> i64 {
    a ^ b   // XOR of two 32-bit-bounded values is itself 32-bit bounded
}

pub fn and32(a: i64, b: i64) -> i64 {
    a & b
}

pub fn or32(a: i64, b: i64) -> i64 {
    a | b
}

pub fn not32(a: i64) -> i64 {
    a ^ 4294967295   // XOR with all-ones == bitwise NOT within 32 bits
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/hash_word32_primitives.sio`
Expected: prints `hash_word32_primitives: all cases passed`, exits 0.

If ANY assertion fails here, this is a genuinely new surprise beyond Finding 13 (plain `i64` `+`/`-`/shift/bitwise-ops at small, 32-bit-bounded magnitudes have been extensively proven correct elsewhere on this branch) — STOP and report BLOCKED with the exact failing assertion; do not patch around it.

- [ ] **Step 4: Commit**

```bash
git add stdlib/hash/word32.sio tests/run-pass/hash_word32_primitives.sio docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md
git commit -m "feat(hash): add word32 primitives as i64-masked words (native u32 arithmetic is broken -- Finding 13)"
```

---

### Task 2: SHA-1

**Files:**
- Create: `stdlib/hash/sha1.sio`
- Test: `tests/run-pass/hash_sha1_vectors.sio`

**Interfaces:**
- Consumes: `stdlib/hash/word32.sio`'s `add32`/`rotl32`/`xor32`/`and32`/`or32`/`not32`; `stdlib/net/socket.sio`'s `RawBuf`/`rawbuf_new`/`rawbuf_set`/`rawbuf_get`.
- Produces: `pub fn sha1(buf: &RawBuf, len: i64) -> [u8; 20] with IO`.

- [ ] **Step 1: Independently re-verify the three test vectors before writing them into a committed test**

This project's standing discipline requires confirming these against their published source before trusting them. This plan's best recollection (verify against FIPS 180-4 / RFC 3174 or an equivalent authoritative source):
- SHA-1("") = `da39a3ee5e6b4b0d3255bfef95601890afd80709`
- SHA-1("abc") = `a9993e364706816aba3e25717850c26c9cd0d89d`
- SHA-1("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") = `84983e441c3bd26ebaae4aa1f95129e5e54670f1` (corrected -- the original transcription was missing the trailing "1", confirmed via three independent tools during Task 2)

- [ ] **Step 2: Write the failing test**

Create `tests/run-pass/hash_sha1_vectors.sio`:

```sio
//@ run-pass
use hash::sha1::*
use net::socket::*

fn write_str(buf: &RawBuf, s: string) -> i64 with IO {
    let n = str_len(s)
    var i: i64 = 0
    while i < n {
        rawbuf_set(buf, i, str_char_at(s, i))
        i = i + 1
    }
    n
}

fn assert_digest_hex(digest: [u8; 20], expected_hex: string) with IO {
    var i: i64 = 0
    while i < 20 {
        let byte_val = digest[i as usize] as i64
        let hi_nibble = byte_val / 16
        let lo_nibble = byte_val - hi_nibble * 16
        let expected_hi = str_char_at(expected_hex, i * 2)
        let expected_lo = str_char_at(expected_hex, i * 2 + 1)
        let actual_hi = hex_nibble_to_char(hi_nibble)
        let actual_lo = hex_nibble_to_char(lo_nibble)
        assert(actual_hi == expected_hi)
        assert(actual_lo == expected_lo)
        i = i + 1
    }
}

fn hex_nibble_to_char(nibble: i64) -> i64 {
    if nibble < 10 {
        48 + nibble          // '0'..'9'
    } else {
        87 + nibble           // 'a'..'f' ('a' is 97, 97 - 10 = 87)
    }
}

fn main() with IO {
    // Vector 1: empty string.
    let buf1 = rawbuf_new(1)
    let d1 = sha1(&buf1, 0)
    assert_digest_hex(d1, "da39a3ee5e6b4b0d3255bfef95601890afd80709")

    // Vector 2: "abc".
    let buf2 = rawbuf_new(3)
    let len2 = write_str(&buf2, "abc")
    let d2 = sha1(&buf2, len2)
    assert_digest_hex(d2, "a9993e364706816aba3e25717850c26c9cd0d89d")

    // Vector 3: multi-block NIST vector (56 bytes, spans two 512-bit blocks
    // once padded).
    let msg3 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    let buf3 = rawbuf_new(64)
    let len3 = write_str(&buf3, msg3)
    let d3 = sha1(&buf3, len3)
    assert_digest_hex(d3, "84983e441c3bd26ebaae4aa1f95129e5e54670f1")

    println("hash_sha1_vectors: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/hash_sha1_vectors.sio`
Expected: FAIL to compile — `stdlib/hash/sha1.sio` doesn't exist yet.

- [ ] **Step 3: Implement `stdlib/hash/sha1.sio`**

```sio
// stdlib/hash/sha1.sio
use hash::word32::*
use net::socket::*

// Every "u32" constant/word in this file is stored as an i64 in
// 0..4294967295 -- see word32.sio for why (Finding 13: native u32
// arithmetic does not wrap mod 2^32 on Madaros).
const SHA1_H0: i64 = 1732584193
const SHA1_H1: i64 = 4023233417
const SHA1_H2: i64 = 2562383102
const SHA1_H3: i64 = 271733878
const SHA1_H4: i64 = 3285377520

const SHA1_K0: i64 = 1518500249
const SHA1_K1: i64 = 1859775393
const SHA1_K2: i64 = 2400959708
const SHA1_K3: i64 = 3395469782

// Returns the byte at absolute offset `i` of the padded message: the real
// message for i<len, the 0x80 marker at i==len, zero padding, then the
// big-endian 64-bit original bit-length in the last 8 bytes.
fn sha1_padded_byte(buf: &RawBuf, len: i64, padded_len: i64, i: i64) -> i64 with IO {
    if i < len {
        rawbuf_get(buf, i)
    } else if i == len {
        128
    } else if i < padded_len - 8 {
        0
    } else {
        // len is bounded far below 2^61 in this project (every caller's
        // input is at most a few tens of KB), so len*8 fits comfortably in
        // i64 with room to spare -- not implicated by Finding 11, which is
        // u64-specific, and len is always non-negative so `>>` behaves as
        // a logical shift here regardless of signed/unsigned semantics.
        let bit_len = len * 8
        let byte_index_from_end = padded_len - i
        let shift_amount = (byte_index_from_end - 1) * 8
        (bit_len >> shift_amount) & 255
    }
}

pub fn sha1(buf: &RawBuf, len: i64) -> [u8; 20] with IO {
    var h0 = SHA1_H0
    var h1 = SHA1_H1
    var h2 = SHA1_H2
    var h3 = SHA1_H3
    var h4 = SHA1_H4

    // Padded length: smallest multiple of 64 that fits len bytes + 1 (0x80)
    // + 8 (length field). Plain i64 division on a small, non-negative
    // value -- not implicated by Finding 11 (u64-specific).
    let padded_len: i64 = ((len + 9 + 63) / 64) * 64

    var block_start: i64 = 0
    while block_start < padded_len {
        var w: [i64; 80] = [0; 80]
        var t: i64 = 0
        while t < 16 {
            let base = block_start + t * 4
            let b0 = sha1_padded_byte(buf, len, padded_len, base)
            let b1 = sha1_padded_byte(buf, len, padded_len, base + 1)
            let b2 = sha1_padded_byte(buf, len, padded_len, base + 2)
            let b3 = sha1_padded_byte(buf, len, padded_len, base + 3)
            w[t as usize] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
            t = t + 1
        }
        t = 16
        while t < 80 {
            let x = xor32(xor32(w[(t - 3) as usize], w[(t - 8) as usize]), xor32(w[(t - 14) as usize], w[(t - 16) as usize]))
            w[t as usize] = rotl32(x, 1)
            t = t + 1
        }

        var a = h0
        var b = h1
        var c = h2
        var d = h3
        var e = h4

        t = 0
        while t < 80 {
            var f: i64 = 0
            var k: i64 = 0
            if t < 20 {
                f = or32(and32(b, c), and32(not32(b), d))
                k = SHA1_K0
            } else if t < 40 {
                f = xor32(xor32(b, c), d)
                k = SHA1_K1
            } else if t < 60 {
                f = or32(or32(and32(b, c), and32(b, d)), and32(c, d))
                k = SHA1_K2
            } else {
                f = xor32(xor32(b, c), d)
                k = SHA1_K3
            }
            let temp = add32(add32(add32(add32(rotl32(a, 5), f), e), k), w[t as usize])
            e = d
            d = c
            c = rotl32(b, 30)
            b = a
            a = temp
            t = t + 1
        }

        h0 = add32(h0, a)
        h1 = add32(h1, b)
        h2 = add32(h2, c)
        h3 = add32(h3, d)
        h4 = add32(h4, e)

        block_start = block_start + 64
    }

    var out: [u8; 20] = [0; 20]
    out[0] = (shr32(h0, 24) & 255) as u8
    out[1] = (shr32(h0, 16) & 255) as u8
    out[2] = (shr32(h0, 8) & 255) as u8
    out[3] = (h0 & 255) as u8
    out[4] = (shr32(h1, 24) & 255) as u8
    out[5] = (shr32(h1, 16) & 255) as u8
    out[6] = (shr32(h1, 8) & 255) as u8
    out[7] = (h1 & 255) as u8
    out[8] = (shr32(h2, 24) & 255) as u8
    out[9] = (shr32(h2, 16) & 255) as u8
    out[10] = (shr32(h2, 8) & 255) as u8
    out[11] = (h2 & 255) as u8
    out[12] = (shr32(h3, 24) & 255) as u8
    out[13] = (shr32(h3, 16) & 255) as u8
    out[14] = (shr32(h3, 8) & 255) as u8
    out[15] = (h3 & 255) as u8
    out[16] = (shr32(h4, 24) & 255) as u8
    out[17] = (shr32(h4, 16) & 255) as u8
    out[18] = (shr32(h4, 8) & 255) as u8
    out[19] = (h4 & 255) as u8
    out
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/hash_sha1_vectors.sio`
Expected: prints `hash_sha1_vectors: all cases passed`, exits 0. If a digest mismatches, first suspect Step 1's transcribed expected hex (re-verify against the published source) before assuming the algorithm implementation is wrong.

- [ ] **Step 5: Commit**

```bash
git add stdlib/hash/sha1.sio tests/run-pass/hash_sha1_vectors.sio
git commit -m "feat(hash): add SHA-1"
```

---

### Task 3: SHA-256

**Files:**
- Create: `stdlib/hash/sha256.sio`
- Test: `tests/run-pass/hash_sha256_vectors.sio`

**Interfaces:**
- Consumes: `stdlib/hash/word32.sio` (same as Task 2).
- Produces: `pub fn sha256(buf: &RawBuf, len: i64) -> [u8; 32] with IO`.

- [ ] **Step 1: Independently re-verify the three test vectors**

- SHA-256("") = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85`
- SHA-256("abc") = `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`
- SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") = `248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1`

**These three lengths look one hex digit too long for a 32-byte (64 hex char) digest — recount each against its authoritative source before use; this is exactly the kind of transcription slip this project's independent-reverification discipline exists to catch.** Do not assume this plan's recollection is correct; treat it as a draft to check, not a value to trust.

- [ ] **Step 2: Write the failing test**

Create `tests/run-pass/hash_sha256_vectors.sio`, structured identically to Task 2's `hash_sha1_vectors.sio` (same `write_str`/`hex_nibble_to_char` helpers, reimplemented locally in this file since Sounio test files don't share helper code across files on this branch) but calling `sha256` and asserting 32-byte (64 hex char) digests against the three re-verified values from Step 1:

```sio
//@ run-pass
use hash::sha256::*
use net::socket::*

fn write_str(buf: &RawBuf, s: string) -> i64 with IO {
    let n = str_len(s)
    var i: i64 = 0
    while i < n {
        rawbuf_set(buf, i, str_char_at(s, i))
        i = i + 1
    }
    n
}

fn hex_nibble_to_char(nibble: i64) -> i64 {
    if nibble < 10 {
        48 + nibble
    } else {
        87 + nibble
    }
}

fn assert_digest_hex(digest: [u8; 32], expected_hex: string) with IO {
    var i: i64 = 0
    while i < 32 {
        let byte_val = digest[i as usize] as i64
        let hi_nibble = byte_val / 16
        let lo_nibble = byte_val - hi_nibble * 16
        let expected_hi = str_char_at(expected_hex, i * 2)
        let expected_lo = str_char_at(expected_hex, i * 2 + 1)
        assert(hex_nibble_to_char(hi_nibble) == expected_hi)
        assert(hex_nibble_to_char(lo_nibble) == expected_lo)
        i = i + 1
    }
}

fn main() with IO {
    let buf1 = rawbuf_new(1)
    let d1 = sha256(&buf1, 0)
    assert_digest_hex(d1, "PASTE_INDEPENDENTLY_VERIFIED_EMPTY_STRING_DIGEST_HERE")

    let buf2 = rawbuf_new(3)
    let len2 = write_str(&buf2, "abc")
    let d2 = sha256(&buf2, len2)
    assert_digest_hex(d2, "PASTE_INDEPENDENTLY_VERIFIED_ABC_DIGEST_HERE")

    let msg3 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    let buf3 = rawbuf_new(64)
    let len3 = write_str(&buf3, msg3)
    let d3 = sha256(&buf3, len3)
    assert_digest_hex(d3, "PASTE_INDEPENDENTLY_VERIFIED_MULTIBLOCK_DIGEST_HERE")

    println("hash_sha256_vectors: all cases passed")
}
```

Replace each `PASTE_INDEPENDENTLY_VERIFIED_..._HERE` placeholder with the exact 64-hex-character digest you independently confirmed in Step 1 — these three placeholders are the ONE deliberate exception to this plan's normal "no placeholders" rule, precisely because this plan's own transcription of these specific values is flagged above as unreliable; do not invent a value or copy this plan's Step 1 text without checking it first.

Run: `./bin/souc run tests/run-pass/hash_sha256_vectors.sio`
Expected: FAIL to compile — `stdlib/hash/sha256.sio` doesn't exist yet (also fails if a placeholder wasn't replaced — that's fine, fix it before proceeding).

- [ ] **Step 3: Implement `stdlib/hash/sha256.sio`**

```sio
// stdlib/hash/sha256.sio
use hash::word32::*
use net::socket::*

// Every "u32" constant/word in this file is stored as an i64 in
// 0..4294967295 -- see word32.sio for why (Finding 13: native u32
// arithmetic does not wrap mod 2^32 on Madaros).
const SHA256_H0: i64 = 0x6a09e667
const SHA256_H1: i64 = 0xbb67ae85
const SHA256_H2: i64 = 0x3c6ef372
const SHA256_H3: i64 = 0xa54ff53a
const SHA256_H4: i64 = 0x510e527f
const SHA256_H5: i64 = 0x9b05688c
const SHA256_H6: i64 = 0x1f83d9ab
const SHA256_H7: i64 = 0x5be0cd19

// FIPS 180-4 Table 4.1. Public, standardized constants -- cross-check
// against the published standard (or a second independent published
// source) before trusting this transcription; any single wrong constant
// is caught immediately by this task's own NIST vector tests failing.
//
// NOTE (updated after Task 3 actually ran): a top-level `const [i64; 64]`
// here does NOT compile -- Finding 15 (docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md)
// found that any top-level `const` array with more than 16 elements breaks
// Madaros's native-v2 IR lowering. Use a function returning the literal
// instead, called once into a local at the top of sha256():
fn sha256_k_table() -> [i64; 64] {
    [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]
}
// (This section documents what Task 3 actually shipped, commit 735a0cce8 --
// the file created there uses `sha256_k()` as the function name and calls
// it into a local `sha256_k_table` inside `sha256()`; the exact naming
// used here is illustrative, match the already-shipped file if referring
// to it, don't re-derive from this snippet.)

fn sha256_padded_byte(buf: &RawBuf, len: i64, padded_len: i64, i: i64) -> i64 with IO {
    if i < len {
        rawbuf_get(buf, i)
    } else if i == len {
        128
    } else if i < padded_len - 8 {
        0
    } else {
        let bit_len = len * 8
        let byte_index_from_end = padded_len - i
        let shift_amount = (byte_index_from_end - 1) * 8
        (bit_len >> shift_amount) & 255
    }
}

pub fn sha256(buf: &RawBuf, len: i64) -> [u8; 32] with IO {
    let sha256_k = sha256_k_table()
    var h0 = SHA256_H0
    var h1 = SHA256_H1
    var h2 = SHA256_H2
    var h3 = SHA256_H3
    var h4 = SHA256_H4
    var h5 = SHA256_H5
    var h6 = SHA256_H6
    var h7 = SHA256_H7

    let padded_len: i64 = ((len + 9 + 63) / 64) * 64
    var block_start: i64 = 0
    while block_start < padded_len {
        var w: [i64; 64] = [0; 64]
        var t: i64 = 0
        while t < 16 {
            let base = block_start + t * 4
            let b0 = sha256_padded_byte(buf, len, padded_len, base)
            let b1 = sha256_padded_byte(buf, len, padded_len, base + 1)
            let b2 = sha256_padded_byte(buf, len, padded_len, base + 2)
            let b3 = sha256_padded_byte(buf, len, padded_len, base + 3)
            w[t as usize] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
            t = t + 1
        }
        t = 16
        while t < 64 {
            let w15 = w[(t - 15) as usize]
            let w2 = w[(t - 2) as usize]
            let s0 = xor32(xor32(rotr32(w15, 7), rotr32(w15, 18)), shr32(w15, 3))
            let s1 = xor32(xor32(rotr32(w2, 17), rotr32(w2, 19)), shr32(w2, 10))
            w[t as usize] = add32(add32(add32(w[(t - 16) as usize], s0), w[(t - 7) as usize]), s1)
            t = t + 1
        }

        var a = h0
        var b = h1
        var c = h2
        var d = h3
        var e = h4
        var f = h5
        var g = h6
        var h = h7

        t = 0
        while t < 64 {
            let s1 = xor32(xor32(rotr32(e, 6), rotr32(e, 11)), rotr32(e, 25))
            let ch = xor32(and32(e, f), and32(not32(e), g))
            let temp1 = add32(add32(add32(add32(h, s1), ch), sha256_k[t as usize]), w[t as usize])
            let s0 = xor32(xor32(rotr32(a, 2), rotr32(a, 13)), rotr32(a, 22))
            let maj = xor32(xor32(and32(a, b), and32(a, c)), and32(b, c))
            let temp2 = add32(s0, maj)
            h = g
            g = f
            f = e
            e = add32(d, temp1)
            d = c
            c = b
            b = a
            a = add32(temp1, temp2)
            t = t + 1
        }

        h0 = add32(h0, a)
        h1 = add32(h1, b)
        h2 = add32(h2, c)
        h3 = add32(h3, d)
        h4 = add32(h4, e)
        h5 = add32(h5, f)
        h6 = add32(h6, g)
        h7 = add32(h7, h)

        block_start = block_start + 64
    }

    var out: [u8; 32] = [0; 32]
    out[0] = (shr32(h0, 24) & 255) as u8
    out[1] = (shr32(h0, 16) & 255) as u8
    out[2] = (shr32(h0, 8) & 255) as u8
    out[3] = (h0 & 255) as u8
    out[4] = (shr32(h1, 24) & 255) as u8
    out[5] = (shr32(h1, 16) & 255) as u8
    out[6] = (shr32(h1, 8) & 255) as u8
    out[7] = (h1 & 255) as u8
    out[8] = (shr32(h2, 24) & 255) as u8
    out[9] = (shr32(h2, 16) & 255) as u8
    out[10] = (shr32(h2, 8) & 255) as u8
    out[11] = (h2 & 255) as u8
    out[12] = (shr32(h3, 24) & 255) as u8
    out[13] = (shr32(h3, 16) & 255) as u8
    out[14] = (shr32(h3, 8) & 255) as u8
    out[15] = (h3 & 255) as u8
    out[16] = (shr32(h4, 24) & 255) as u8
    out[17] = (shr32(h4, 16) & 255) as u8
    out[18] = (shr32(h4, 8) & 255) as u8
    out[19] = (h4 & 255) as u8
    out[20] = (shr32(h5, 24) & 255) as u8
    out[21] = (shr32(h5, 16) & 255) as u8
    out[22] = (shr32(h5, 8) & 255) as u8
    out[23] = (h5 & 255) as u8
    out[24] = (shr32(h6, 24) & 255) as u8
    out[25] = (shr32(h6, 16) & 255) as u8
    out[26] = (shr32(h6, 8) & 255) as u8
    out[27] = (h6 & 255) as u8
    out[28] = (shr32(h7, 24) & 255) as u8
    out[29] = (shr32(h7, 16) & 255) as u8
    out[30] = (shr32(h7, 8) & 255) as u8
    out[31] = (h7 & 255) as u8
    out
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/hash_sha256_vectors.sio`
Expected: prints `hash_sha256_vectors: all cases passed`, exits 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/hash/sha256.sio tests/run-pass/hash_sha256_vectors.sio
git commit -m "feat(hash): add SHA-256"
```

---

### Task 4: 64-bit arithmetic audit + `word64.sio` scalar-pair primitives

**Files:**
- Create: `stdlib/hash/word64.sio`
- Modify: `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` (append a new numbered finding)
- Test: `tests/run-pass/hash_word64_primitives.sio`

**Interfaces:**
- Produces: `fn add64(a_hi: i64, a_lo: i64, b_hi: i64, b_lo: i64) -> (i64, i64)`, `fn xor64(a_hi: i64, a_lo: i64, b_hi: i64, b_lo: i64) -> (i64, i64)`, `fn and64(a_hi: i64, a_lo: i64, b_hi: i64, b_lo: i64) -> (i64, i64)`, `fn not64(a_hi: i64, a_lo: i64) -> (i64, i64)`, `fn shr64(a_hi: i64, a_lo: i64, n: i64) -> (i64, i64)`, `fn rotr64(a_hi: i64, a_lo: i64, n: i64) -> (i64, i64)`.

- [ ] **Step 1: Write the failing audit/primitives test**

This test deliberately exercises values with the high (32nd) bit of each half set — the scenario BigInt was designed to avoid entirely, and the scenario this whole module exists to handle correctly despite that.

Create `tests/run-pass/hash_word64_primitives.sio`:

```sio
//@ run-pass
use hash::word64::*

fn main() with IO {
    // add64: 0xFFFFFFFF_FFFFFFFF + 1 == 0 (full 64-bit wraparound, carry
    // propagating from the low half into the high half and then off the
    // top entirely).
    let (r1_hi, r1_lo) = add64(4294967295, 4294967295, 0, 1)
    assert(r1_hi == 0)
    assert(r1_lo == 0)

    // Carry propagation without full wraparound: 0x00000000_FFFFFFFF + 1
    // == 0x00000001_00000000.
    let (r2_hi, r2_lo) = add64(0, 4294967295, 0, 1)
    assert(r2_hi == 1)
    assert(r2_lo == 0)

    // shr64 by 1 on a value with the top bit of the high half set --
    // logical shift must produce 0x40000000_00000000, not sign-extend.
    let (r3_hi, r3_lo) = shr64(2147483648, 0, 1)
    assert(r3_hi == 1073741824)
    assert(r3_lo == 0)

    // shr64 by 33 (crosses the half boundary): the low half's new value
    // comes entirely from the high half's low bits.
    let (r4_hi, r4_lo) = shr64(3, 0, 33)
    assert(r4_hi == 0)
    assert(r4_lo == 1)

    // rotr64 by 32 exactly: swaps the two halves.
    let (r5_hi, r5_lo) = rotr64(1, 2, 32)
    assert(r5_hi == 2)
    assert(r5_lo == 1)

    // rotr64 by 1 on a value with the low bit of the low half set --
    // rotating right must carry that bit into the top of the high half.
    let (r6_hi, r6_lo) = rotr64(0, 1, 1)
    assert(r6_hi == 2147483648)
    assert(r6_lo == 0)

    // xor64/and64/not64 at the top-bit boundary of both halves.
    let (r7_hi, r7_lo) = xor64(2147483648, 2147483648, 2147483648, 2147483648)
    assert(r7_hi == 0)
    assert(r7_lo == 0)
    let (r8_hi, r8_lo) = and64(4294967295, 4294967295, 2147483648, 2147483648)
    assert(r8_hi == 2147483648)
    assert(r8_lo == 2147483648)
    let (r9_hi, r9_lo) = not64(0, 0)
    assert(r9_hi == 4294967295)
    assert(r9_lo == 4294967295)

    println("hash_word64_primitives: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/hash_word64_primitives.sio`
Expected: FAIL to compile — `stdlib/hash/word64.sio` doesn't exist yet.

- [ ] **Step 2: Implement `stdlib/hash/word64.sio`**

```sio
// stdlib/hash/word64.sio
//
// 64-bit word arithmetic for SHA-384/512, represented as two independent
// i64 scalars (high 32 bits, low 32 bits of the logical 64-bit value) --
// NOT a struct, NOT an array of a pair type. See this plan's Global
// Constraints for why: this branch has proven scalar arrays and scalar
// tuples extensively, but never a struct-typed array or a reference to a
// single array element, and this module has no need to introduce either.
//
// Every value here is masked to 0..0xFFFFFFFF after any operation that
// could exceed 32 bits, so no intermediate value in this file ever
// approaches i64's own bit-63 boundary (Finding 11's danger zone) --
// this file sidesteps Finding 11 by construction, the same discipline
// stdlib/bignum/bigint.sio established for 16-bit limbs, applied here to
// 32-bit halves instead.
//
// This file's own primitives are audited (Task 4) against exactly the
// case BigInt never needed to handle: values with the top bit of a half
// set, which is the COMMON case for hash state, not an edge case.

pub fn add64(a_hi: i64, a_lo: i64, b_hi: i64, b_lo: i64) -> (i64, i64) {
    let lo_sum = a_lo + b_lo                 // at most 2*0xFFFFFFFF, far under bit 63
    let carry = lo_sum >> 32                  // lo_sum < 2^33, safe shift, yields 0 or 1
    let lo_out = lo_sum & 4294967295
    let hi_sum = a_hi + b_hi + carry          // at most 2*0xFFFFFFFF + 1, far under bit 63
    let hi_out = hi_sum & 4294967295          // mod-2^32 wraparound on the high half,
                                                 // matching true mod-2^64 addition overall
    (hi_out, lo_out)
}

pub fn xor64(a_hi: i64, a_lo: i64, b_hi: i64, b_lo: i64) -> (i64, i64) {
    (a_hi ^ b_hi, a_lo ^ b_lo)   // XOR is unconditionally safe per Finding 11
}

pub fn and64(a_hi: i64, a_lo: i64, b_hi: i64, b_lo: i64) -> (i64, i64) {
    (a_hi & b_hi, a_lo & b_lo)
}

pub fn not64(a_hi: i64, a_lo: i64) -> (i64, i64) {
    (a_hi ^ 4294967295, a_lo ^ 4294967295)
}

// Logical right shift by n bits, 0 < n < 64, treating (hi,lo) as one
// 64-bit value with hi as the more-significant half.
pub fn shr64(a_hi: i64, a_lo: i64, n: i64) -> (i64, i64) {
    if n < 32 {
        // Bits shifted out of hi's bottom feed into lo's top; both shift
        // amounts here are in [1,31], safe on 32-bit-bounded values.
        let lo_out = ((a_lo >> n) | (a_hi << (32 - n))) & 4294967295
        let hi_out = a_hi >> n
        (hi_out, lo_out)
    } else {
        // n in [32,63]: everything from the low half is shifted off
        // entirely; only the high half's own bits (shifted further)
        // survive, landing in the new low half.
        let shift = n - 32
        (0, a_hi >> shift)
    }
}

// Left shift by n bits, 0 < n < 64 -- only used internally by rotr64.
fn shl64(a_hi: i64, a_lo: i64, n: i64) -> (i64, i64) {
    if n < 32 {
        let hi_out = ((a_hi << n) | (a_lo >> (32 - n))) & 4294967295
        let lo_out = (a_lo << n) & 4294967295
        (hi_out, lo_out)
    } else {
        let shift = n - 32
        (((a_lo << shift) & 4294967295), 0)
    }
}

pub fn rotr64(a_hi: i64, a_lo: i64, n: i64) -> (i64, i64) {
    let (r_hi, r_lo) = shr64(a_hi, a_lo, n)
    let (l_hi, l_lo) = shl64(a_hi, a_lo, 64 - n)
    (r_hi | l_hi, r_lo | l_lo)
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/hash_word64_primitives.sio`
Expected: prints `hash_word64_primitives: all cases passed`, exits 0.

**If this fails**, that is not a plan-blocking surprise the way Task 1 failing would be — this outcome was anticipated in the spec (a Finding-11-style bug reproducing at 64-bit scale is the EXPECTED result this design already defends against). Investigate which specific assertion fails and whether it points to an error in this file's own logic (fixable directly) versus a genuinely new compiler behavior not yet covered by the masking discipline above (escalate to BLOCKED only in the latter case, with the exact failing case).

- [ ] **Step 4: Record the audit finding**

Append a new numbered finding to `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`, stating the outcome: either "native 64-bit-scale `i64` arithmetic at the bit-63 boundary reproduces Finding 11's failure pattern, mitigated here via 32-bit-half decomposition" (expected), or, if the primitives in Step 2 turn out to work correctly even without the masking discipline (i.e. a follow-up experiment shows native `i64`/`u64` arithmetic was safe all along for these specific operations) — note that too, since it would be a meaningful update to Finding 11's known scope. Either way, point to `tests/run-pass/hash_word64_primitives.sio` as the evidence.

- [ ] **Step 5: Commit**

```bash
git add stdlib/hash/word64.sio tests/run-pass/hash_word64_primitives.sio docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md
git commit -m "feat(hash): add word64 scalar-pair primitives, audit 64-bit arithmetic at bit-63 boundary"
```

---

### Task 5: `sha512_core.sio` (shared compression core) + `sha512.sio`

**Files:**
- Create: `stdlib/hash/sha512_core.sio`
- Create: `stdlib/hash/sha512.sio`
- Test: `tests/run-pass/hash_sha512_vectors.sio`

**Interfaces:**
- Consumes: `stdlib/hash/word64.sio`'s `add64`/`xor64`/`and64`/`not64`/`rotr64`/`shr64` (all `(i64,i64) -> (i64,i64)`, taking two `(hi,lo)` pairs as four scalar arguments where binary); `stdlib/net/socket.sio`'s `RawBuf`/`rawbuf_get`.
- Produces: `pub fn sha512_compress(buf: &RawBuf, len: i64, iv_hi: &[i64; 8], iv_lo: &[i64; 8]) -> ([i64; 8], [i64; 8]) with IO` (in `sha512_core.sio`); `pub fn sha512(buf: &RawBuf, len: i64) -> [u8; 64] with IO` (in `sha512.sio`).

**Pre-flight ruling (made before dispatch, not discovered mid-task):** `sha512_compress` takes `iv_hi`/`iv_lo` BY REFERENCE (`&[i64; 8]`), not by value. Passing a bare fixed-size array as a function parameter — by value OR by reference — has never been exercised anywhere on this branch (BigInt/DER always embed arrays inside a struct and pass `&TheStruct`, never a bare array type as a parameter). By-reference is the safer default given this branch's overwhelming preference for passing compound data by reference (`&BigInt`, `&DerReader`, `&RawBuf`), so this plan commits to that form rather than by-value. **Step 1 of this task adds an explicit smoke check of this exact pattern** (a trivial function taking `&[i64; N]` and reading an element) before the real compression logic is written on top of it — mirroring Task 1/4's "verify the risky pattern first" discipline.

- [ ] **Step 0: Smoke-check passing a fixed-size array by reference as a function parameter**

Before writing any real code, create a scratch file `tests/run-pass/hash_scratch_smoke2.sio` (delete it at the end of this step) with:

```sio
fn read_third(arr: &[i64; 8]) -> i64 {
    arr[2]
}

fn main() with IO {
    let a: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80]
    let v = read_third(&a)
    assert(v == 30)
    println("smoke check passed")
}
```

Run: `./bin/souc run tests/run-pass/hash_scratch_smoke2.sio`
Expected: prints `smoke check passed`, exits 0. This confirms the exact pattern `sha512_compress`'s signature depends on — a bare fixed-size array passed by reference to a function — which has not been used anywhere else on this branch (BigInt/DER always pass arrays wrapped inside a struct's own reference, never a bare array type). If this fails to compile or behaves unexpectedly, STOP and report BLOCKED with the exact output — do not silently switch to by-value or another workaround; this affects the signature this whole task is built on and needs a controller ruling.

Once it passes: `rm tests/run-pass/hash_scratch_smoke2.sio`.

- [ ] **Step 1: Independently re-verify the SHA-512 test vectors and the round-constant table**

Test vectors (re-verify against FIPS 180-4 / an authoritative NIST source):
- SHA-512("") = `cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e`
- SHA-512("abc") = `ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f`

The 80 round constants (`SHA512_K_HI`/`SHA512_K_LO` below, in Step 2) are transcribed from FIPS 180-4 Table 4.11 (Section 4.2.3) — a public, standardized table, not derived arithmetic. Before trusting this plan's transcription, cross-check the full table against the published standard (or a second independent published source, e.g. a widely-used reference implementation's own published constant table) — a single wrong hex digit among 80 64-bit constants would silently produce a wrong hash that this task's own vector tests will catch, but tracking down which of 80 constants is wrong after the fact is far more expensive than checking the table once before committing it.

- [ ] **Step 2: Implement `stdlib/hash/sha512_core.sio`**

```sio
// stdlib/hash/sha512_core.sio
//
// Shared SHA-384/SHA-512 compression core (FIPS 180-4 6.3-6.4). SHA-384 and
// SHA-512 differ ONLY in initial hash values (IV) and final output
// truncation (SHA-384 keeps the first 48 of the would-be 64 output bytes)
// -- this file implements the compression function and message schedule
// once, parameterized on the caller-supplied IV.

use hash::word64::*
use net::socket::*

// FIPS 180-4 Table 4.11 (Section 4.2.3). See this task's brief for why
// these 80 public, standardized constants are sourced from the published
// table directly rather than hand-derived, and why that's safe here.
//
// NOTE: these are functions returning the array literal, NOT top-level
// `const` arrays. Finding 15 (docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md)
// found that a top-level `const` array with more than 16 elements breaks
// Madaros's native-v2 IR lowering ("IR instruction arena contract
// violated") -- discovered in Task 3 with SHA-256's 64-entry table, and
// these 80-entry tables would hit the identical failure. Call each once
// into a local binding at the top of sha512_compress (Step 3's usage
// below already does this).
fn sha512_k_hi_table() -> [i64; 80] {
    [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        0xca273ece, 0xd186b8c7, 0xeada7dd6, 0xf57d4f7f, 0x06f067aa, 0x0a637dc5, 0x113f9804, 0x1b710b35,
        0x28db77f5, 0x32caab7b, 0x3c9ebe0a, 0x431d67c4, 0x4cc5d4be, 0x597f299c, 0x5fcb6fab, 0x6c44198c,
    ]
}
fn sha512_k_lo_table() -> [i64; 80] {
    [
        0xd728ae22, 0x23ef65cd, 0xec4d3b2f, 0x8189dbbc, 0xf348b538, 0xb605d019, 0xaf194f9b, 0xda6d8118,
        0xa3030242, 0x45706fbe, 0x4ee4b28c, 0xd5ffb4e2, 0xf27b896f, 0x3b1696b1, 0x25c71235, 0xcf692694,
        0x9ef14ad2, 0x384f25e3, 0x8b8cd5b5, 0x77ac9c65, 0x592b0275, 0x6ea6e483, 0xbd41fbd4, 0x831153b5,
        0xee66dfab, 0x2db43210, 0x98fb213f, 0xbeef0ee4, 0x3da88fc2, 0x930aa725, 0xe003826f, 0x0a0e6e70,
        0x46d22ffc, 0x5c26c926, 0x5ac42aed, 0x9d95b3df, 0x8baf63de, 0x3c77b2a8, 0x47edaee6, 0x1482353b,
        0x4cf10364, 0xbc423001, 0xd0f89791, 0x0654be30, 0xd6ef5218, 0x5565a910, 0x5771202a, 0x32bbd1b8,
        0xb8d2d0c8, 0x5141ab53, 0xdf8eeb99, 0xe19b48a8,
        // Indices 52-79 corrected here (2026-08-23) -- the original draft
        // of this table had a transcription error across this entire
        // range, caught by Task 5's own independent re-derivation from
        // FIPS 180-4's definition (fractional part of cube roots of the
        // first 80 primes), confirmed via two independent high-precision
        // methods by the implementer and re-confirmed by the controller
        // via a third, independently-written computation. The values
        // below are the corrected ones; do not revert to any earlier
        // version of this table.
        0xc5c95a63, 0xe3418acb, 0x7763e373, 0xd6b2b8a3,
        0x5defb2fc, 0x43172f60, 0xa1f0ab72, 0x1a6439ec,
        0x23631e28, 0xde82bde9, 0xb2c67915, 0xe372532b,
        0xea26619c, 0x21c0c207, 0xcde0eb1e, 0xee6ed178,
        0x72176fba, 0xa2c898a6, 0xbef90dae, 0x131c471b,
        0x23047d84, 0x40c72493, 0x15c9bebc, 0x9c100d4c,
        0xcb3e42b6, 0xfc657e2a, 0x3ad6faec, 0x4a475817,
    ]
}

fn sha512_padded_byte(buf: &RawBuf, len: i64, padded_len: i64, i: i64) -> i64 with IO {
    // 1024-bit blocks, 128-bit length field -- but per this sub-project's
    // spec Non-Goals, the length is far below 2^61 bytes so the high 64
    // bits of that 128-bit field are always zero.
    if i < len {
        rawbuf_get(buf, i)
    } else if i == len {
        128
    } else if i < padded_len - 8 {
        0   // covers both the zero-padding zone AND the high (always-zero)
             // 64 bits of the 128-bit length field
    } else {
        let bit_len = len * 8
        let byte_index_from_end = padded_len - i
        let shift_amount = (byte_index_from_end - 1) * 8
        (bit_len >> shift_amount) & 255
    }
}

// Runs the full SHA-384/512 compression over buf[0..len), starting from the
// given 8-word IV (as parallel hi/lo arrays), returning the final state as
// parallel hi/lo arrays. The caller (sha384.sio / sha512.sio) supplies the
// IV and decides how many of the resulting 64 bytes to keep.
pub fn sha512_compress(buf: &RawBuf, len: i64, iv_hi: &[i64; 8], iv_lo: &[i64; 8]) -> ([i64; 8], [i64; 8]) with IO {
    let sha512_k_hi = sha512_k_hi_table()
    let sha512_k_lo = sha512_k_lo_table()

    var h_hi: [i64; 8] = [0; 8]
    var h_lo: [i64; 8] = [0; 8]
    var i: i64 = 0
    while i < 8 {
        h_hi[i as usize] = iv_hi[i as usize]
        h_lo[i as usize] = iv_lo[i as usize]
        i = i + 1
    }

    // 1 (0x80) + 16 (128-bit length field) = 17.
    let padded_len: i64 = ((len + 17 + 127) / 128) * 128
    var block_start: i64 = 0
    while block_start < padded_len {
        var w_hi: [i64; 80] = [0; 80]
        var w_lo: [i64; 80] = [0; 80]
        var t: i64 = 0
        while t < 16 {
            let base = block_start + t * 8
            let b0 = sha512_padded_byte(buf, len, padded_len, base)
            let b1 = sha512_padded_byte(buf, len, padded_len, base + 1)
            let b2 = sha512_padded_byte(buf, len, padded_len, base + 2)
            let b3 = sha512_padded_byte(buf, len, padded_len, base + 3)
            let b4 = sha512_padded_byte(buf, len, padded_len, base + 4)
            let b5 = sha512_padded_byte(buf, len, padded_len, base + 5)
            let b6 = sha512_padded_byte(buf, len, padded_len, base + 6)
            let b7 = sha512_padded_byte(buf, len, padded_len, base + 7)
            w_hi[t as usize] = ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) & 4294967295
            w_lo[t as usize] = ((b4 << 24) | (b5 << 16) | (b6 << 8) | b7) & 4294967295
            t = t + 1
        }
        t = 16
        while t < 80 {
            let w15_hi = w_hi[(t - 15) as usize]
            let w15_lo = w_lo[(t - 15) as usize]
            let w2_hi = w_hi[(t - 2) as usize]
            let w2_lo = w_lo[(t - 2) as usize]

            let (r1_hi, r1_lo) = rotr64(w15_hi, w15_lo, 1)
            let (r8_hi, r8_lo) = rotr64(w15_hi, w15_lo, 8)
            let (s7_hi, s7_lo) = shr64(w15_hi, w15_lo, 7)
            let (x1_hi, x1_lo) = xor64(r1_hi, r1_lo, r8_hi, r8_lo)
            let (s0_hi, s0_lo) = xor64(x1_hi, x1_lo, s7_hi, s7_lo)

            let (r19_hi, r19_lo) = rotr64(w2_hi, w2_lo, 19)
            let (r61_hi, r61_lo) = rotr64(w2_hi, w2_lo, 61)
            let (s6_hi, s6_lo) = shr64(w2_hi, w2_lo, 6)
            let (x2_hi, x2_lo) = xor64(r19_hi, r19_lo, r61_hi, r61_lo)
            let (s1_hi, s1_lo) = xor64(x2_hi, x2_lo, s6_hi, s6_lo)

            let (sum1_hi, sum1_lo) = add64(w_hi[(t - 16) as usize], w_lo[(t - 16) as usize], s0_hi, s0_lo)
            let (sum2_hi, sum2_lo) = add64(sum1_hi, sum1_lo, w_hi[(t - 7) as usize], w_lo[(t - 7) as usize])
            let (sum3_hi, sum3_lo) = add64(sum2_hi, sum2_lo, s1_hi, s1_lo)
            w_hi[t as usize] = sum3_hi
            w_lo[t as usize] = sum3_lo
            t = t + 1
        }

        var a_hi = h_hi[0]; var a_lo = h_lo[0]
        var b_hi = h_hi[1]; var b_lo = h_lo[1]
        var c_hi = h_hi[2]; var c_lo = h_lo[2]
        var d_hi = h_hi[3]; var d_lo = h_lo[3]
        var e_hi = h_hi[4]; var e_lo = h_lo[4]
        var f_hi = h_hi[5]; var f_lo = h_lo[5]
        var g_hi = h_hi[6]; var g_lo = h_lo[6]
        var hh_hi = h_hi[7]; var hh_lo = h_lo[7]

        t = 0
        while t < 80 {
            let (r14_hi, r14_lo) = rotr64(e_hi, e_lo, 14)
            let (r18_hi, r18_lo) = rotr64(e_hi, e_lo, 18)
            let (r41_hi, r41_lo) = rotr64(e_hi, e_lo, 41)
            let (xa_hi, xa_lo) = xor64(r14_hi, r14_lo, r18_hi, r18_lo)
            let (bigs1_hi, bigs1_lo) = xor64(xa_hi, xa_lo, r41_hi, r41_lo)

            let (ef_hi, ef_lo) = and64(e_hi, e_lo, f_hi, f_lo)
            let (notE_hi, notE_lo) = not64(e_hi, e_lo)
            let (negEg_hi, negEg_lo) = and64(notE_hi, notE_lo, g_hi, g_lo)
            let (ch_hi, ch_lo) = xor64(ef_hi, ef_lo, negEg_hi, negEg_lo)

            let (t1a_hi, t1a_lo) = add64(hh_hi, hh_lo, bigs1_hi, bigs1_lo)
            let (t1b_hi, t1b_lo) = add64(t1a_hi, t1a_lo, ch_hi, ch_lo)
            let (t1c_hi, t1c_lo) = add64(t1b_hi, t1b_lo, sha512_k_hi[t as usize], sha512_k_lo[t as usize])
            let (temp1_hi, temp1_lo) = add64(t1c_hi, t1c_lo, w_hi[t as usize], w_lo[t as usize])

            let (r28_hi, r28_lo) = rotr64(a_hi, a_lo, 28)
            let (r34_hi, r34_lo) = rotr64(a_hi, a_lo, 34)
            let (r39_hi, r39_lo) = rotr64(a_hi, a_lo, 39)
            let (xb_hi, xb_lo) = xor64(r28_hi, r28_lo, r34_hi, r34_lo)
            let (bigs0_hi, bigs0_lo) = xor64(xb_hi, xb_lo, r39_hi, r39_lo)

            let (ab_hi, ab_lo) = and64(a_hi, a_lo, b_hi, b_lo)
            let (ac_hi, ac_lo) = and64(a_hi, a_lo, c_hi, c_lo)
            let (bc_hi, bc_lo) = and64(b_hi, b_lo, c_hi, c_lo)
            let (mj_hi, mj_lo) = xor64(ab_hi, ab_lo, ac_hi, ac_lo)
            let (maj_hi, maj_lo) = xor64(mj_hi, mj_lo, bc_hi, bc_lo)

            let (temp2_hi, temp2_lo) = add64(bigs0_hi, bigs0_lo, maj_hi, maj_lo)

            hh_hi = g_hi; hh_lo = g_lo
            g_hi = f_hi; g_lo = f_lo
            f_hi = e_hi; f_lo = e_lo
            let (newE_hi, newE_lo) = add64(d_hi, d_lo, temp1_hi, temp1_lo)
            e_hi = newE_hi; e_lo = newE_lo
            d_hi = c_hi; d_lo = c_lo
            c_hi = b_hi; c_lo = b_lo
            b_hi = a_hi; b_lo = a_lo
            let (newA_hi, newA_lo) = add64(temp1_hi, temp1_lo, temp2_hi, temp2_lo)
            a_hi = newA_hi; a_lo = newA_lo

            t = t + 1
        }

        let (nh0_hi, nh0_lo) = add64(h_hi[0], h_lo[0], a_hi, a_lo)
        let (nh1_hi, nh1_lo) = add64(h_hi[1], h_lo[1], b_hi, b_lo)
        let (nh2_hi, nh2_lo) = add64(h_hi[2], h_lo[2], c_hi, c_lo)
        let (nh3_hi, nh3_lo) = add64(h_hi[3], h_lo[3], d_hi, d_lo)
        let (nh4_hi, nh4_lo) = add64(h_hi[4], h_lo[4], e_hi, e_lo)
        let (nh5_hi, nh5_lo) = add64(h_hi[5], h_lo[5], f_hi, f_lo)
        let (nh6_hi, nh6_lo) = add64(h_hi[6], h_lo[6], g_hi, g_lo)
        let (nh7_hi, nh7_lo) = add64(h_hi[7], h_lo[7], hh_hi, hh_lo)
        h_hi[0] = nh0_hi; h_lo[0] = nh0_lo
        h_hi[1] = nh1_hi; h_lo[1] = nh1_lo
        h_hi[2] = nh2_hi; h_lo[2] = nh2_lo
        h_hi[3] = nh3_hi; h_lo[3] = nh3_lo
        h_hi[4] = nh4_hi; h_lo[4] = nh4_lo
        h_hi[5] = nh5_hi; h_lo[5] = nh5_lo
        h_hi[6] = nh6_hi; h_lo[6] = nh6_lo
        h_hi[7] = nh7_hi; h_lo[7] = nh7_lo

        block_start = block_start + 128
    }

    (h_hi, h_lo)
}
```

- [ ] **Step 3: Implement `stdlib/hash/sha512.sio`**

```sio
// stdlib/hash/sha512.sio
use hash::sha512_core::*
use net::socket::*

const SHA512_IV_HI: [i64; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
const SHA512_IV_LO: [i64; 8] = [0xf3bcc908, 0x84caa73b, 0xfe94f82b, 0x5f1d36f1, 0xade682d1, 0x2b3e6c1f, 0xfb41bd6b, 0x137e2179]

pub fn sha512(buf: &RawBuf, len: i64) -> [u8; 64] with IO {
    let (h_hi, h_lo) = sha512_compress(buf, len, &SHA512_IV_HI, &SHA512_IV_LO)
    var out: [u8; 64] = [0; 64]
    var i: i64 = 0
    while i < 8 {
        let hv = h_hi[i as usize]
        let lv = h_lo[i as usize]
        let base = i * 8
        out[base as usize] = ((hv >> 24) & 255) as u8
        out[(base + 1) as usize] = ((hv >> 16) & 255) as u8
        out[(base + 2) as usize] = ((hv >> 8) & 255) as u8
        out[(base + 3) as usize] = (hv & 255) as u8
        out[(base + 4) as usize] = ((lv >> 24) & 255) as u8
        out[(base + 5) as usize] = ((lv >> 16) & 255) as u8
        out[(base + 6) as usize] = ((lv >> 8) & 255) as u8
        out[(base + 7) as usize] = (lv & 255) as u8
        i = i + 1
    }
    out
}
```

- [ ] **Step 4: Write the failing test**

Create `tests/run-pass/hash_sha512_vectors.sio` (same `write_str`/`hex_nibble_to_char`/digest-comparison helper pattern as Tasks 2-3, sized for 64-byte digests), asserting the two vectors from Step 1 (empty string, `"abc"`), PLUS a third, multi-block vector using the standard NIST 112-byte SHA-512 test message: `"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"` (112 bytes, spans two 1024-bit blocks once padded). Use `assert_digest_hex(d3, "PASTE_INDEPENDENTLY_VERIFIED_MULTIBLOCK_SHA512_DIGEST_HERE")` — the same deliberate placeholder-requiring-independent-verification exception Task 3 Step 2 uses, since this plan's own recollection of a 128-hex-character digest for this specific message is not reliable enough to commit directly. Replace the placeholder with the exact digest after confirming it against an authoritative published source. Follow the exact structure of `hash_sha256_vectors.sio`'s test file otherwise, adapted to `sha512`'s `[u8; 64]` output.

Run: `./bin/souc run tests/run-pass/hash_sha512_vectors.sio`
Expected: FAIL to compile initially (before Steps 2-3 exist), then pass once implemented.

- [ ] **Step 5: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/hash_sha512_vectors.sio`
Expected: prints `hash_sha512_vectors: all cases passed`, exits 0. If a digest mismatches, suspect (in order): a transcription error in Step 1's expected hex, a transcription error in the 80-constant table (re-verify against FIPS 180-4), then the core logic itself.

- [ ] **Step 6: Commit**

```bash
git add stdlib/hash/sha512_core.sio stdlib/hash/sha512.sio tests/run-pass/hash_sha512_vectors.sio
git commit -m "feat(hash): add shared SHA-384/512 compression core and SHA-512"
```

---

### Task 6: SHA-384

**Files:**
- Create: `stdlib/hash/sha384.sio`
- Test: `tests/run-pass/hash_sha384_vectors.sio`

**Interfaces:**
- Consumes: `stdlib/hash/sha512_core.sio`'s `sha512_compress`.
- Produces: `pub fn sha384(buf: &RawBuf, len: i64) -> [u8; 48] with IO`.

- [ ] **Step 1: Independently re-verify the SHA-384 test vectors**

- SHA-384("") = `38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b`
- SHA-384("abc") = `cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7`

- [ ] **Step 2: Write the failing test**

Create `tests/run-pass/hash_sha384_vectors.sio`, same pattern as Task 5's SHA-512 test but for `sha384`'s `[u8; 48]` output and the two vectors from Step 1, PLUS a third multi-block vector using the same 112-byte NIST message as Task 5's Step 4 (`"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"`) with its SHA-384 digest as a `PASTE_INDEPENDENTLY_VERIFIED_..._HERE` placeholder, following the same exception pattern as Task 3 Step 2 and Task 5 Step 4 — replace with the exact 96-hex-character digest after independent verification.

Run: `./bin/souc run tests/run-pass/hash_sha384_vectors.sio`
Expected: FAIL to compile — `stdlib/hash/sha384.sio` doesn't exist yet.

- [ ] **Step 3: Implement `stdlib/hash/sha384.sio`**

```sio
// stdlib/hash/sha384.sio
//
// SHA-384 is SHA-512's compression core with a DIFFERENT initial hash value
// (FIPS 180-4 5.3.4) and truncated to the first 48 of the resulting 64
// output bytes. The distinct IV is required by the standard specifically
// so SHA-384 never coincides with a truncated SHA-512 of the same message.
use hash::sha512_core::*
use net::socket::*

const SHA384_IV_HI: [i64; 8] = [0xcbbb9d5d, 0x629a292a, 0x9159015a, 0x152fecd8, 0x67332667, 0x8eb44a87, 0xdb0c2e0d, 0x47b5481d]
const SHA384_IV_LO: [i64; 8] = [0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939, 0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4]

pub fn sha384(buf: &RawBuf, len: i64) -> [u8; 48] with IO {
    let (h_hi, h_lo) = sha512_compress(buf, len, &SHA384_IV_HI, &SHA384_IV_LO)
    var out: [u8; 48] = [0; 48]
    var i: i64 = 0
    while i < 6 {
        // Only the first 6 of the 8 state words (48 bytes) are kept --
        // this IS SHA-384's defined truncation, per FIPS 180-4 6.5.
        let hv = h_hi[i as usize]
        let lv = h_lo[i as usize]
        let base = i * 8
        out[base as usize] = ((hv >> 24) & 255) as u8
        out[(base + 1) as usize] = ((hv >> 16) & 255) as u8
        out[(base + 2) as usize] = ((hv >> 8) & 255) as u8
        out[(base + 3) as usize] = (hv & 255) as u8
        out[(base + 4) as usize] = ((lv >> 24) & 255) as u8
        out[(base + 5) as usize] = ((lv >> 16) & 255) as u8
        out[(base + 6) as usize] = ((lv >> 8) & 255) as u8
        out[(base + 7) as usize] = (lv & 255) as u8
        i = i + 1
    }
    out
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/hash_sha384_vectors.sio`
Expected: prints `hash_sha384_vectors: all cases passed`, exits 0.

- [ ] **Step 5: Run the full hash test group together**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix hash_`
Expected: 6/6 passing (`hash_word32_primitives`, `hash_sha1_vectors`, `hash_sha256_vectors`, `hash_word64_primitives`, `hash_sha512_vectors`, `hash_sha384_vectors` — confirm the exact count matches the files actually present at this point). Note: Tasks 1-5's test files did not originally carry the `//@ run-pass` annotation this harness requires to recognize a test file (the same omission this plan made for its own earlier task text) — this must be added to all 5 pre-existing files (not just this task's own new one) before this step can show a real 6/6 rather than a false "0 found, 0 skipped" or a silent skip.

- [ ] **Step 6: Commit**

```bash
git add stdlib/hash/sha384.sio tests/run-pass/hash_sha384_vectors.sio
git commit -m "feat(hash): add SHA-384"
```

<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-23-madaros-asn1-der-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-23-madaros-asn1-der-plan
-->

# Madaros ASN.1 DER Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `stdlib/asn1/der.sio`, a generic, reusable ASN.1 DER decoder (cursor-based, not tree-based) covering the tag types X.509 needs, tested against hand-built fixtures, a real RSA-key-shaped DER structure, and adversarial malformed input.

**Architecture:** A `DerReader` cursor (raw pointer + position + end offset, all plain scalars — no nested `RawBuf`, no `linear` annotation) walks a flat byte buffer. Reading a TLV header returns a `DerTag` plus an advanced cursor; descending into a constructed value (`der_enter`) returns a narrower child cursor over the same buffer. No heap tree is ever built.

**Tech Stack:** Sounio (Madaros v0.80.0), reusing `stdlib/bignum/bigint.sio`'s `BigInt` for INTEGER decoding and `stdlib/net/socket.sio`'s `RawBuf`/`rawbuf_new`/`rawbuf_set`/`rawbuf_get` pattern for buffer I/O.

**Spec:** `docs/superpowers/specs/2026-08-23-madaros-asn1-der-design.md`

## Global Constraints

- **`DerReader` deviates from the spec's literal field list, by ruling — read this before Task 1.** The spec's Architecture section shows `struct DerReader { buf: RawBuf, pos: i64, end: i64 }`. This plan instead uses `struct DerReader { buf_ptr: *mut u8, pos: i64, end: i64 }` — storing the raw pointer directly rather than nesting a whole `RawBuf` struct. Reason: nowhere on this branch has a `RawBuf` (or any struct) been copied by dereferencing a `&RawBuf` reference (`*some_ref`) into a fresh struct value — every existing use either builds a `RawBuf` fresh via `rawbuf_new` or reads/writes it through `rawbuf_get`/`rawbuf_set`, never copies the whole struct by value through a reference. Storing the bare pointer sidesteps this untested operation entirely: pointer-field reads through a reference (`buf.ptr`) are already a proven-safe pattern (`rawbuf_get`/`rawbuf_set`/`build_sockaddr` in `stdlib/net/socket.sio` all do `buf.ptr as i64` through a `&RawBuf` parameter). This does not change any public behavior described in the spec — `DerReader` is still an opaque cursor to every caller, constructed via `der_reader_new(buf: &RawBuf, start: i64, end: i64) -> DerReader`.
- **Never copy a struct by dereferencing a reference (`*some_ref`).** Every place this plan needs "a copy of a `DerReader`/`DerTag` with one field changed" rebuilds it as a fresh struct literal, reading the source fields individually (`DerReader { buf_ptr: r.buf_ptr, pos: new_pos, end: r.end }`) — never `*r`. This mirrors Finding 9's destructure-and-rebuild pattern from the BigInt/socket work on this branch, applied preemptively rather than discovered as a bug.
- **`DerReader` and `DerTag` are plain (non-`linear`) structs.** Neither owns a resource that must be consumed exactly once (unlike `TcpSocket`), so nothing in Findings 4/5/9's linear-type restrictions applies to them. Task 1's first step includes a five-minute smoke check confirming a plain struct with a `bool` field can be freely reassigned in a loop and returned inside a tuple, before the rest of the file is written on that assumption. `bool` as a field/return type is *not* a fresh risk — it is already used throughout `stdlib/net/http_client.sio` and `stdlib/net/dns.sio` (e.g. `is_numeric_ip(...) -> bool with IO`) — but a `bool` *struct field* (as opposed to a return type) has not been exercised on this branch before, so Task 1 verifies it explicitly rather than assuming the return-type precedent extends to it.
- **Sentinel error convention, no `Result`/`Option`.** Every fallible function returns `(value, i64 status)` with `DER_OK = 0` on success, matching `stdlib/net/socket.sio`'s `(TcpSocket, i64)` shape. Status constants: `DER_OK`, `DER_ERR_TRUNCATED`, `DER_ERR_BAD_LENGTH_FORM`, `DER_ERR_LENGTH_TOO_LARGE`, `DER_ERR_NOT_CONSTRUCTED`, `DER_ERR_NESTING_TOO_DEEP` (defined in Task 1; `DER_ERR_NESTING_TOO_DEEP` is defined for callers to use — this decoder has no call-spanning state to enforce it itself, per the spec's Nesting-depth guard section — so it is declared but never returned by any function in this module; do not treat that as a bug).
- **DER strictness is enforced, not just BER tolerance.** Indefinite-length (`0x80` alone) is rejected. Non-minimal long-form lengths (a leading zero length-byte, or a long-form encoding of a value that would fit in short form) are rejected. Both use `DER_ERR_BAD_LENGTH_FORM`.
- **Hard length cap:** `DER_MAX_LENGTH: i64 = 65536`. Checked in `der_peek_tag` before any read of that many content bytes is attempted.
- **All arithmetic on tag/length bytes stays in `i64`, using only `<<`, `|`, `&`, `>>`, `+`, comparisons — never on a value anywhere near bit 63.** Per the audit doc's Finding 11, `u64` right-shift/divide/modulo are broken once bit 63 is set; this module works entirely in `i64` on values bounded by `DER_MAX_LENGTH` (65536, 17 bits) or smaller, so this is true by construction, but every step that shifts/masks a length accumulator must carry a one-line comment saying why it stays far below bit 63, matching this branch's established documentation discipline (see `stdlib/bignum/bigint.sio` for the style).
- **Module import path:** a caller outside `stdlib/` uses `use asn1::der::*`; a future sibling file inside `stdlib/asn1/` uses the bare `use der::*` form (per the audit doc's Findings 6/8).
- **No AI attribution in any commit message** (this repo's `CLAUDE.md` rule, enforced with zero exceptions).
- **Conventional-Commits-style commit messages**: `feat(asn1): ...`, `test(asn1): ...`, `docs(asn1): ...`.
- **Test files:** `tests/run-pass/asn1_*.sio`, run via `bash scripts/run_sio_test_suite.sh --filter-prefix asn1_`. **Never run the whole-repo test suite** — established throughout this branch's prior work as too slow for checkpoints.
- **Run every shell command as a plain foreground command, one at a time. Never use any Monitor/background-wait mechanism.** Multiple agents on this exact branch have previously stalled by waiting on a Monitor/background tool instead of running commands directly (e.g. waiting on port TIME_WAIT clearing, or a test run) — this is a standing, repeated instruction on this branch, not new caution.
- **Every test file asserts real expected values** (byte-by-byte fixture content, expected `DerTag` fields, expected `BigInt` values via `bigint_cmp`) — never a test that merely checks "it ran without crashing."

## File Structure

One new file for this whole sub-project: **`stdlib/asn1/der.sio`** — the entire generic DER decoder: types, constants, tag/length decoding, navigation (`der_enter`/`der_skip`/`der_at_end`), and typed primitive readers (`der_read_integer`, `der_read_boolean`, `der_read_null`, `der_read_raw_bytes`). It is one file, not split by responsibility across multiple files, because every piece operates on the same two small structs (`DerReader`, `DerTag`) and the whole module is expected to stay well under a thousand lines — matching `stdlib/bignum/bigint.sio`'s precedent of one file for one cohesive, small-surface-area module.

Five test files, one per task, each exercising exactly that task's new surface: `tests/run-pass/asn1_tag_length.sio`, `asn1_navigate.sio`, `asn1_primitives.sio`, `asn1_raw_bytes_and_errors.sio`, `asn1_rsa_pubkey_integration.sio`.

---

### Task 1: Types, constants, and tag/length header decoding

**Files:**
- Create: `stdlib/asn1/der.sio`
- Test: `tests/run-pass/asn1_tag_length.sio`

**Interfaces:**
- Produces: `struct DerReader { buf_ptr: *mut u8, pos: i64, end: i64 }`; `struct DerTag { class: i32, constructed: bool, tag_number: i32, content_start: i64, content_len: i64 }`; constants `DER_TAG_BOOLEAN=1, DER_TAG_INTEGER=2, DER_TAG_BIT_STRING=3, DER_TAG_OCTET_STRING=4, DER_TAG_NULL=5, DER_TAG_OBJECT_IDENTIFIER=6, DER_TAG_UTF8_STRING=12, DER_TAG_SEQUENCE=16, DER_TAG_SET=17, DER_TAG_PRINTABLE_STRING=19, DER_TAG_IA5_STRING=22, DER_TAG_UTC_TIME=23, DER_TAG_GENERALIZED_TIME=24` (all `i32`); `DER_CLASS_UNIVERSAL=0, DER_CLASS_APPLICATION=1, DER_CLASS_CONTEXT=2, DER_CLASS_PRIVATE=3` (all `i32`); `DER_MAX_LENGTH: i64 = 65536`; status constants `DER_OK=0, DER_ERR_TRUNCATED=-1, DER_ERR_BAD_LENGTH_FORM=-2, DER_ERR_LENGTH_TOO_LARGE=-3, DER_ERR_NOT_CONSTRUCTED=-4, DER_ERR_NESTING_TOO_DEEP=-5` (all `i64`); `fn der_reader_new(buf: &RawBuf, start: i64, end: i64) -> DerReader`; `fn der_peek_tag(r: &DerReader) -> (DerTag, i64) with IO`.

- [ ] **Step 1: Smoke-check the plain-struct assumptions this whole module rests on**

Before writing any real code, create a scratch file `tests/run-pass/asn1_scratch_smoke.sio` (delete it at the end of this step — it is not part of the module's permanent test suite) with exactly this content:

```sio
struct ScratchTag {
    flag: bool,
    n: i32,
}

fn make_tag(f: bool, n: i32) -> (ScratchTag, i64) {
    (ScratchTag { flag: f, n: n }, 0)
}

fn main() with IO {
    var current = ScratchTag { flag: false, n: 0 }
    var i: i32 = 0
    while i < 3 {
        let (t, status) = make_tag(i == 1, i)
        current = ScratchTag { flag: t.flag, n: t.n }
        i = i + 1
    }
    assert(current.n == 2)
    assert(current.flag == false)
    println("smoke check passed")
}
```

Run: `./bin/souc run tests/run-pass/asn1_scratch_smoke.sio`
Expected: prints `smoke check passed` and exits 0.

If this fails to compile or behaves unexpectedly, STOP — do not work around it by adding `linear` or changing the design. Report `BLOCKED` with the exact compiler output; this smoke check gates the entire rest of this task and this module's whole design, so it needs a ruling from the controller, not a silent workaround.

Once it passes, delete the scratch file: `rm tests/run-pass/asn1_scratch_smoke.sio`.

- [ ] **Step 2: Write the failing test for tag/length decoding**

Create `tests/run-pass/asn1_tag_length.sio`:

```sio
use asn1::der::*
use net::socket::*

fn main() with IO {
    // Case 1: short-form length. INTEGER (tag 0x02), length 3, content 01 00 01
    // (this is the value 65537, reused later as an RSA exponent shape).
    let buf1 = rawbuf_new(5)
    rawbuf_set(&buf1, 0, 0x02)
    rawbuf_set(&buf1, 1, 0x03)
    rawbuf_set(&buf1, 2, 0x01)
    rawbuf_set(&buf1, 3, 0x00)
    rawbuf_set(&buf1, 4, 0x01)
    let r1 = der_reader_new(&buf1, 0, 5)
    let (tag1, status1) = der_peek_tag(&r1)
    assert(status1 == DER_OK)
    assert(tag1.class == DER_CLASS_UNIVERSAL)
    assert(tag1.constructed == false)
    assert(tag1.tag_number == DER_TAG_INTEGER)
    assert(tag1.content_start == 2)
    assert(tag1.content_len == 3)

    // Case 2: universal SEQUENCE tag byte 0x30 = constructed, tag 16.
    let buf2 = rawbuf_new(3)
    rawbuf_set(&buf2, 0, 0x30)
    rawbuf_set(&buf2, 1, 0x01)
    rawbuf_set(&buf2, 2, 0xAA)
    let r2 = der_reader_new(&buf2, 0, 3)
    let (tag2, status2) = der_peek_tag(&r2)
    assert(status2 == DER_OK)
    assert(tag2.class == DER_CLASS_UNIVERSAL)
    assert(tag2.constructed == true)
    assert(tag2.tag_number == DER_TAG_SEQUENCE)
    assert(tag2.content_start == 2)
    assert(tag2.content_len == 1)

    // Case 3: context-specific constructed tag [0], byte 0xA0 = class CONTEXT
    // (bits 7-6 = 10), constructed (bit 5 = 1), tag number 0 (bits 4-0).
    let buf3 = rawbuf_new(5)
    rawbuf_set(&buf3, 0, 0xA0)
    rawbuf_set(&buf3, 1, 0x03)
    rawbuf_set(&buf3, 2, 0x02)
    rawbuf_set(&buf3, 3, 0x01)
    rawbuf_set(&buf3, 4, 0x05)
    let r3 = der_reader_new(&buf3, 0, 5)
    let (tag3, status3) = der_peek_tag(&r3)
    assert(status3 == DER_OK)
    assert(tag3.class == DER_CLASS_CONTEXT)
    assert(tag3.constructed == true)
    assert(tag3.tag_number == 0)
    assert(tag3.content_start == 2)
    assert(tag3.content_len == 3)

    // Case 4: long-form length. OCTET STRING (0x04), length byte 0x81 (long
    // form, 1 subsequent length byte), that byte is 0xC8 = 200 -- a value
    // that genuinely needs long form (>127), so this is minimally encoded.
    let buf4 = rawbuf_new(203)
    rawbuf_set(&buf4, 0, 0x04)
    rawbuf_set(&buf4, 1, 0x81)
    rawbuf_set(&buf4, 2, 0xC8)
    var i: i64 = 0
    while i < 200 {
        rawbuf_set(&buf4, 3 + i, 0)
        i = i + 1
    }
    let r4 = der_reader_new(&buf4, 0, 203)
    let (tag4, status4) = der_peek_tag(&r4)
    assert(status4 == DER_OK)
    assert(tag4.class == DER_CLASS_UNIVERSAL)
    assert(tag4.constructed == false)
    assert(tag4.tag_number == DER_TAG_OCTET_STRING)
    assert(tag4.content_start == 3)
    assert(tag4.content_len == 200)

    println("asn1_tag_length: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/asn1_tag_length.sio`
Expected: FAIL to compile — `stdlib/asn1/der.sio` does not exist yet, so `use asn1::der::*` and every `der_*`/`DER_*` symbol are undefined.

- [ ] **Step 3: Implement `stdlib/asn1/der.sio`'s types, constants, and tag/length decoding**

```sio
// stdlib/asn1/der.sio
//
// Generic ASN.1 DER (Distinguished Encoding Rules) decoder. Cursor-based,
// not tree-based: DerReader is a flat { pointer, pos, end } over a RawBuf's
// bytes, and "entering" a constructed value (SEQUENCE, SET, an explicit
// context-specific wrapper) just narrows that range into a fresh child
// cursor -- no heap-allocated tree of decoded nodes is ever built. See
// docs/superpowers/specs/2026-08-23-madaros-asn1-der-design.md.
//
// This decoder has no X.509-specific knowledge: it hands back raw tag/
// length/value information and typed readers for the ASN.1 universal types
// X.509 uses, leaving semantic interpretation (which OID means what, which
// context tag means what) to a separate X.509 layer built on top.
//
// DER, not BER: indefinite-length encoding and non-minimal length forms are
// both rejected as malformed input, never tolerated.

use bignum::bigint::*
use net::socket::*

pub struct DerReader {
    buf_ptr: *mut u8,   // NOT a RawBuf -- see this plan's Global Constraints
                          // for why the raw pointer is stored directly
    pos: i64,             // current read position, absolute offset into the buffer
    end: i64,             // exclusive upper bound of this cursor's valid range
}

pub struct DerTag {
    class: i32,          // 0=UNIVERSAL, 1=APPLICATION, 2=CONTEXT-SPECIFIC, 3=PRIVATE
    constructed: bool,    // true for SEQUENCE/SET/explicit wrappers; false for primitives
    tag_number: i32,      // universal tag number, or the raw context-specific [N] number
    content_start: i64,   // absolute offset where this tag's value bytes begin
    content_len: i64,     // length of the value in bytes
}

pub const DER_TAG_BOOLEAN: i32 = 1
pub const DER_TAG_INTEGER: i32 = 2
pub const DER_TAG_BIT_STRING: i32 = 3
pub const DER_TAG_OCTET_STRING: i32 = 4
pub const DER_TAG_NULL: i32 = 5
pub const DER_TAG_OBJECT_IDENTIFIER: i32 = 6
pub const DER_TAG_UTF8_STRING: i32 = 12
pub const DER_TAG_SEQUENCE: i32 = 16
pub const DER_TAG_SET: i32 = 17
pub const DER_TAG_PRINTABLE_STRING: i32 = 19
pub const DER_TAG_IA5_STRING: i32 = 22
pub const DER_TAG_UTC_TIME: i32 = 23
pub const DER_TAG_GENERALIZED_TIME: i32 = 24

pub const DER_CLASS_UNIVERSAL: i32 = 0
pub const DER_CLASS_APPLICATION: i32 = 1
pub const DER_CLASS_CONTEXT: i32 = 2
pub const DER_CLASS_PRIVATE: i32 = 3

pub const DER_MAX_LENGTH: i64 = 65536

pub const DER_OK: i64 = 0
pub const DER_ERR_TRUNCATED: i64 = 0 - 1
pub const DER_ERR_BAD_LENGTH_FORM: i64 = 0 - 2
pub const DER_ERR_LENGTH_TOO_LARGE: i64 = 0 - 3
pub const DER_ERR_NOT_CONSTRUCTED: i64 = 0 - 4
pub const DER_ERR_NESTING_TOO_DEEP: i64 = 0 - 5

pub fn der_reader_new(buf: &RawBuf, start: i64, end: i64) -> DerReader {
    DerReader { buf_ptr: buf.ptr, pos: start, end: end }
}

// Reads one byte at an absolute offset within the underlying buffer.
// Mirrors stdlib/net/socket.sio's rawbuf_get body exactly (masking with
// & 255 per Finding 3: a raw pointer dereference reads a full 8-byte word,
// not one byte), just parameterized on DerReader's own buf_ptr field.
fn der_byte_at(r: &DerReader, offset: i64) -> i64 with IO {
    let addr = r.buf_ptr as i64
    let p = (addr + offset) as *mut u8
    ((*p) as i64) & 255
}

fn der_tag_zero() -> DerTag {
    DerTag { class: 0, constructed: false, tag_number: 0, content_start: 0, content_len: 0 }
}

pub fn der_at_end(r: &DerReader) -> bool {
    r.pos >= r.end
}

// Decodes the tag+length header starting at r.pos, WITHOUT advancing r.pos
// (the caller inspects the returned DerTag and decides what to do next via
// der_read_tlv/der_enter/der_skip/one of the der_read_* functions).
pub fn der_peek_tag(r: &DerReader) -> (DerTag, i64) with IO {
    if r.pos >= r.end {
        return (der_tag_zero(), DER_ERR_TRUNCATED)
    }
    let b0 = der_byte_at(r, r.pos)

    // b0 is a single byte, 0-255 -- every shift/mask below operates on a
    // value far below bit 63, so Finding 11 does not apply.
    let class_bits = (b0 >> 6) & 3
    let constructed_bit = (b0 >> 5) & 1
    let constructed = constructed_bit == 1
    let tag_num = b0 & 31
    // tag_num == 31 would mean a multi-byte tag number (bits 4-0 all set);
    // this decoder does not parse multi-byte tag numbers -- no field this
    // project needs ever uses one. Recording the truncated low-5-bits value
    // here is a documented limitation, not a silent misparse.

    var p = r.pos + 1
    if p >= r.end {
        return (der_tag_zero(), DER_ERR_TRUNCATED)
    }
    let len_byte = der_byte_at(r, p)
    p = p + 1

    var content_len: i64 = 0
    if (len_byte & 128) == 0 {
        // Short form: len_byte itself is the length, 0-127.
        content_len = len_byte
    } else {
        let num_len_bytes = len_byte & 127
        if num_len_bytes == 0 {
            // 0x80 alone is BER's indefinite-length marker; DER forbids it.
            return (der_tag_zero(), DER_ERR_BAD_LENGTH_FORM)
        }
        if num_len_bytes > 4 {
            // More than 4 length bytes could only encode a value far past
            // DER_MAX_LENGTH (65536 fits in 3 bytes); reject before even
            // reading them so the accumulator below never needs more than
            // 32 bits, let alone approaches bit 63 (Finding 11's boundary).
            return (der_tag_zero(), DER_ERR_BAD_LENGTH_FORM)
        }
        if p + num_len_bytes > r.end {
            return (der_tag_zero(), DER_ERR_TRUNCATED)
        }
        let first_len_byte = der_byte_at(r, p)
        var acc: i64 = 0
        var i: i64 = 0
        while i < num_len_bytes {
            let lb = der_byte_at(r, p + i)
            // acc accumulates at most 4 bytes -> at most 32 bits; << 8 | lb
            // never approaches bit 63.
            acc = (acc << 8) | lb
            i = i + 1
        }
        p = p + num_len_bytes
        // DER requires minimal long-form encoding: a leading zero length
        // byte (when more than one length byte is used), or a long-form
        // encoding of a value that would fit in short form (<128), is
        // non-minimal and therefore malformed DER.
        if num_len_bytes > 1 && first_len_byte == 0 {
            return (der_tag_zero(), DER_ERR_BAD_LENGTH_FORM)
        }
        if acc < 128 {
            return (der_tag_zero(), DER_ERR_BAD_LENGTH_FORM)
        }
        content_len = acc
    }

    if content_len > DER_MAX_LENGTH {
        return (der_tag_zero(), DER_ERR_LENGTH_TOO_LARGE)
    }
    if p + content_len > r.end {
        return (der_tag_zero(), DER_ERR_TRUNCATED)
    }

    let tag = DerTag {
        class: class_bits as i32,
        constructed: constructed,
        tag_number: tag_num as i32,
        content_start: p,
        content_len: content_len,
    }
    (tag, DER_OK)
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/asn1_tag_length.sio`
Expected: prints `asn1_tag_length: all cases passed` and exits 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/asn1/der.sio tests/run-pass/asn1_tag_length.sio
git commit -m "feat(asn1): add DerReader/DerTag types and tag/length header decoding"
```

---

### Task 2: Navigation — `der_read_tlv`, `der_enter`, `der_skip`

**Files:**
- Modify: `stdlib/asn1/der.sio`
- Test: `tests/run-pass/asn1_navigate.sio`

**Interfaces:**
- Consumes: `DerReader`, `DerTag`, `der_peek_tag`, `der_tag_zero` (Task 1, same file — `der_tag_zero` is a private `fn`, not `pub`, but this task's new functions live in the same file so they can call it directly).
- Produces: `fn der_read_tlv(r: &DerReader) -> (DerReader, DerTag, i64) with IO`; `fn der_enter(r: &DerReader, tag: &DerTag) -> (DerReader, i64)`; `fn der_skip(r: &DerReader, tag: &DerTag) -> (DerReader, i64)`.

- [ ] **Step 1: Write the failing test**

Create `tests/run-pass/asn1_navigate.sio`. This builds, by hand, the bytes for:
`SEQUENCE { INTEGER(1), SEQUENCE { INTEGER(2), OCTET_STRING(3 bytes) }, OCTET_STRING(2 bytes, to be skipped unread) }`
— exercising `der_read_tlv` at the top level, `der_enter` into the nested SEQUENCE, `der_read_tlv` again inside it, and `der_skip` on the trailing OCTET STRING without ever reading its bytes.

```sio
use asn1::der::*
use net::socket::*

fn main() with IO {
    // Inner SEQUENCE content: INTEGER(2) = 02 01 02 (3 bytes), then
    // OCTET_STRING(3 bytes AA BB CC) = 04 03 AA BB CC (5 bytes). Inner
    // SEQUENCE content length = 3 + 5 = 8 bytes. Inner SEQUENCE TLV =
    // 30 08 <8 bytes> = 10 bytes total.
    //
    // Outer SEQUENCE content: INTEGER(1) = 02 01 01 (3 bytes), then the
    // 10-byte inner SEQUENCE TLV, then a trailing OCTET_STRING(2 bytes
    // DD EE) = 04 02 DD EE (4 bytes) that the test skips without reading.
    // Outer content length = 3 + 10 + 4 = 17 bytes. Outer TLV = 30 11
    // <17 bytes> = 19 bytes total (0x11 = 17).
    let total_len: i64 = 19
    let buf = rawbuf_new(total_len)
    let bytes = [
        0x30, 0x11,                   // outer SEQUENCE, len 17
        0x02, 0x01, 0x01,             // INTEGER(1)
        0x30, 0x08,                   // inner SEQUENCE, len 8
        0x02, 0x01, 0x02,             // INTEGER(2)
        0x04, 0x03, 0xAA, 0xBB, 0xCC, // OCTET STRING, 3 bytes
        0x04, 0x02, 0xDD, 0xEE,       // trailing OCTET STRING, 2 bytes (skipped)
    ]
    var i: i64 = 0
    while i < total_len {
        rawbuf_set(&buf, i, bytes[i as usize])
        i = i + 1
    }

    let top = der_reader_new(&buf, 0, total_len)
    let (top1, outer_tag, s0) = der_read_tlv(&top)
    assert(s0 == DER_OK)
    assert(outer_tag.tag_number == DER_TAG_SEQUENCE)
    assert(outer_tag.constructed == true)
    assert(outer_tag.content_len == 17)

    let (outer, e0) = der_enter(&top, &outer_tag)
    assert(e0 == DER_OK)
    assert(outer.pos == outer_tag.content_start)
    assert(outer.end == outer_tag.content_start + outer_tag.content_len)

    // First field: INTEGER(1).
    let (outer2, int_tag, s1) = der_read_tlv(&outer)
    assert(s1 == DER_OK)
    assert(int_tag.tag_number == DER_TAG_INTEGER)
    assert(int_tag.content_len == 1)

    // Second field: the nested SEQUENCE -- descend into it.
    let after_int = DerReader { buf_ptr: outer2.buf_ptr, pos: int_tag.content_start + int_tag.content_len, end: outer2.end }
    let (after_int2, inner_seq_tag, s2) = der_read_tlv(&after_int)
    assert(s2 == DER_OK)
    assert(inner_seq_tag.tag_number == DER_TAG_SEQUENCE)
    assert(inner_seq_tag.constructed == true)
    assert(inner_seq_tag.content_len == 8)

    let (inner, e1) = der_enter(&after_int, &inner_seq_tag)
    assert(e1 == DER_OK)

    let (inner2, inner_int_tag, s3) = der_read_tlv(&inner)
    assert(s3 == DER_OK)
    assert(inner_int_tag.tag_number == DER_TAG_INTEGER)
    assert(inner_int_tag.content_len == 1)

    let after_inner_int = DerReader { buf_ptr: inner2.buf_ptr, pos: inner_int_tag.content_start + inner_int_tag.content_len, end: inner2.end }
    let (after_inner_int2, inner_octet_tag, s4) = der_read_tlv(&after_inner_int)
    assert(s4 == DER_OK)
    assert(inner_octet_tag.tag_number == DER_TAG_OCTET_STRING)
    assert(inner_octet_tag.content_len == 3)
    assert(der_at_end(&after_inner_int2) == true)

    // Back at the outer level: the third field is the trailing OCTET
    // STRING. Skip it without reading its content.
    let after_nested = DerReader { buf_ptr: after_int2.buf_ptr, pos: inner_seq_tag.content_start + inner_seq_tag.content_len, end: after_int2.end }
    let (after_nested2, trailing_tag, s5) = der_read_tlv(&after_nested)
    assert(s5 == DER_OK)
    assert(trailing_tag.tag_number == DER_TAG_OCTET_STRING)
    assert(trailing_tag.content_len == 2)

    let (skipped, sk) = der_skip(&after_nested, &trailing_tag)
    assert(sk == DER_OK)
    assert(der_at_end(&skipped) == true)

    // der_enter on a primitive tag must be rejected.
    let (_, bad_enter) = der_enter(&outer, &int_tag)
    assert(bad_enter == DER_ERR_NOT_CONSTRUCTED)

    println("asn1_navigate: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/asn1_navigate.sio`
Expected: FAIL to compile — `der_read_tlv`, `der_enter`, `der_skip` are not defined yet.

- [ ] **Step 2: Implement navigation functions in `stdlib/asn1/der.sio`**

Append to `stdlib/asn1/der.sio`:

```sio
// Decodes the tag+length header starting at r.pos, same as der_peek_tag,
// but returns a new DerReader whose pos has advanced past the header to
// content_start. Per this plan's Global Constraints, this rebuilds a fresh
// DerReader by reading fields individually -- never `*r`.
pub fn der_read_tlv(r: &DerReader) -> (DerReader, DerTag, i64) with IO {
    let (tag, status) = der_peek_tag(r)
    if status != DER_OK {
        let unchanged = DerReader { buf_ptr: r.buf_ptr, pos: r.pos, end: r.end }
        return (unchanged, tag, status)
    }
    let advanced = DerReader { buf_ptr: r.buf_ptr, pos: tag.content_start, end: r.end }
    (advanced, tag, DER_OK)
}

// Returns a child cursor scoped to exactly `tag`'s value bytes. `r` must be
// positioned such that `tag` was decoded from it (i.e. `tag.content_start`
// falls within `r`'s range) -- this is always true when `tag` came from a
// der_peek_tag/der_read_tlv call on this same `r`.
pub fn der_enter(r: &DerReader, tag: &DerTag) -> (DerReader, i64) {
    if tag.constructed == false {
        let unchanged = DerReader { buf_ptr: r.buf_ptr, pos: r.pos, end: r.end }
        return (unchanged, DER_ERR_NOT_CONSTRUCTED)
    }
    let child = DerReader {
        buf_ptr: r.buf_ptr,
        pos: tag.content_start,
        end: tag.content_start + tag.content_len,
    }
    (child, DER_OK)
}

// Returns a cursor advanced past this whole TLV (header + value), without
// reading or copying the value's bytes.
pub fn der_skip(r: &DerReader, tag: &DerTag) -> (DerReader, i64) {
    let new_pos = tag.content_start + tag.content_len
    if new_pos > r.end {
        let unchanged = DerReader { buf_ptr: r.buf_ptr, pos: r.pos, end: r.end }
        return (unchanged, DER_ERR_TRUNCATED)
    }
    let advanced = DerReader { buf_ptr: r.buf_ptr, pos: new_pos, end: r.end }
    (advanced, DER_OK)
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/asn1_navigate.sio`
Expected: prints `asn1_navigate: all cases passed` and exits 0.

- [ ] **Step 4: Commit**

```bash
git add stdlib/asn1/der.sio tests/run-pass/asn1_navigate.sio
git commit -m "feat(asn1): add der_read_tlv/der_enter/der_skip navigation"
```

---

### Task 3: Typed readers — `der_read_boolean`, `der_read_null`, `der_read_integer`

**Files:**
- Modify: `stdlib/asn1/der.sio`
- Test: `tests/run-pass/asn1_primitives.sio`

**Interfaces:**
- Consumes: `stdlib/bignum/bigint.sio`'s `BigInt`, `bigint_zero()`, `bigint_from_u32(v: u32) -> BigInt`, `bigint_mul(a: &BigInt, b: &BigInt) -> BigInt`, `bigint_add(a: &BigInt, b: &BigInt) -> BigInt`, `bigint_cmp(a: &BigInt, b: &BigInt) -> i32`, and `BIGINT_MAX_LIMBS: i32 = 512` (all from Task 1's already-added `use bignum::bigint::*`).
- Produces: `fn der_read_boolean(r: &DerReader, tag: &DerTag) -> (bool, i64) with IO`; `fn der_read_null(r: &DerReader, tag: &DerTag) -> i64`; `fn der_read_integer(r: &DerReader, tag: &DerTag) -> (BigInt, i64) with IO`.

- [ ] **Step 1: Write the failing test**

Create `tests/run-pass/asn1_primitives.sio`:

```sio
use asn1::der::*
use bignum::bigint::*
use net::socket::*

fn read_one_tlv(buf: &RawBuf, len: i64) -> (DerReader, DerTag, i64) with IO {
    let r = der_reader_new(buf, 0, len)
    der_read_tlv(&r)
}

fn main() with IO {
    // BOOLEAN true: 01 01 FF
    let b_true = rawbuf_new(3)
    rawbuf_set(&b_true, 0, 0x01)
    rawbuf_set(&b_true, 1, 0x01)
    rawbuf_set(&b_true, 2, 0xFF)
    let (r1, t1, s1) = read_one_tlv(&b_true, 3)
    assert(s1 == DER_OK)
    let (v1, vs1) = der_read_boolean(&r1, &t1)
    assert(vs1 == DER_OK)
    assert(v1 == true)

    // BOOLEAN false: 01 01 00
    let b_false = rawbuf_new(3)
    rawbuf_set(&b_false, 0, 0x01)
    rawbuf_set(&b_false, 1, 0x01)
    rawbuf_set(&b_false, 2, 0x00)
    let (r2, t2, s2) = read_one_tlv(&b_false, 3)
    assert(s2 == DER_OK)
    let (v2, vs2) = der_read_boolean(&r2, &t2)
    assert(vs2 == DER_OK)
    assert(v2 == false)

    // BOOLEAN invalid content byte: 01 01 7F -- DER forbids any byte for
    // true other than 0xFF.
    let b_bad = rawbuf_new(3)
    rawbuf_set(&b_bad, 0, 0x01)
    rawbuf_set(&b_bad, 1, 0x01)
    rawbuf_set(&b_bad, 2, 0x7F)
    let (r3, t3, s3) = read_one_tlv(&b_bad, 3)
    assert(s3 == DER_OK)
    let (_, vs3) = der_read_boolean(&r3, &t3)
    assert(vs3 == DER_ERR_BAD_LENGTH_FORM)

    // NULL valid: 05 00
    let n_ok = rawbuf_new(2)
    rawbuf_set(&n_ok, 0, 0x05)
    rawbuf_set(&n_ok, 1, 0x00)
    let (r4, t4, s4) = read_one_tlv(&n_ok, 2)
    assert(s4 == DER_OK)
    assert(der_read_null(&r4, &t4) == DER_OK)

    // NULL invalid (nonzero length): 05 01 00
    let n_bad = rawbuf_new(3)
    rawbuf_set(&n_bad, 0, 0x05)
    rawbuf_set(&n_bad, 1, 0x01)
    rawbuf_set(&n_bad, 2, 0x00)
    let (r5, t5, s5) = read_one_tlv(&n_bad, 3)
    assert(s5 == DER_OK)
    assert(der_read_null(&r5, &t5) == DER_ERR_BAD_LENGTH_FORM)

    // INTEGER small: 02 01 05 -> 5
    let i_small = rawbuf_new(3)
    rawbuf_set(&i_small, 0, 0x02)
    rawbuf_set(&i_small, 1, 0x01)
    rawbuf_set(&i_small, 2, 0x05)
    let (r6, t6, s6) = read_one_tlv(&i_small, 3)
    assert(s6 == DER_OK)
    let (v6, vs6) = der_read_integer(&r6, &t6)
    assert(vs6 == DER_OK)
    assert(bigint_cmp(&v6, &bigint_from_u32(5)) == 0)

    // INTEGER with required 0x00 padding: 02 02 00 80 -> 128
    // (0x80 alone would look negative in two's-complement, so DER pads it).
    let i_pad = rawbuf_new(4)
    rawbuf_set(&i_pad, 0, 0x02)
    rawbuf_set(&i_pad, 1, 0x02)
    rawbuf_set(&i_pad, 2, 0x00)
    rawbuf_set(&i_pad, 3, 0x80)
    let (r7, t7, s7) = read_one_tlv(&i_pad, 4)
    assert(s7 == DER_OK)
    let (v7, vs7) = der_read_integer(&r7, &t7)
    assert(vs7 == DER_OK)
    assert(bigint_cmp(&v7, &bigint_from_u32(128)) == 0)

    // INTEGER malformed: two leading zero bytes, 02 03 00 00 01.
    let i_bad = rawbuf_new(5)
    rawbuf_set(&i_bad, 0, 0x02)
    rawbuf_set(&i_bad, 1, 0x03)
    rawbuf_set(&i_bad, 2, 0x00)
    rawbuf_set(&i_bad, 3, 0x00)
    rawbuf_set(&i_bad, 4, 0x01)
    let (r8, t8, s8) = read_one_tlv(&i_bad, 5)
    assert(s8 == DER_OK)
    let (_, vs8) = der_read_integer(&r8, &t8)
    assert(vs8 == DER_ERR_BAD_LENGTH_FORM)

    // INTEGER multi-byte, no padding needed: 02 03 01 00 01 -> 65537.
    let i_multi = rawbuf_new(5)
    rawbuf_set(&i_multi, 0, 0x02)
    rawbuf_set(&i_multi, 1, 0x03)
    rawbuf_set(&i_multi, 2, 0x01)
    rawbuf_set(&i_multi, 3, 0x00)
    rawbuf_set(&i_multi, 4, 0x01)
    let (r9, t9, s9) = read_one_tlv(&i_multi, 5)
    assert(s9 == DER_OK)
    let (v9, vs9) = der_read_integer(&r9, &t9)
    assert(vs9 == DER_OK)
    var expected65537 = bigint_mul(&bigint_from_u32(65536), &bigint_from_u32(1))
    expected65537 = bigint_add(&expected65537, &bigint_from_u32(1))
    assert(bigint_cmp(&v9, &expected65537) == 0)

    println("asn1_primitives: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/asn1_primitives.sio`
Expected: FAIL to compile — `der_read_boolean`, `der_read_null`, `der_read_integer` are not defined yet.

- [ ] **Step 2: Implement the typed readers in `stdlib/asn1/der.sio`**

Append to `stdlib/asn1/der.sio`:

```sio
pub fn der_read_boolean(r: &DerReader, tag: &DerTag) -> (bool, i64) with IO {
    if tag.tag_number != DER_TAG_BOOLEAN || tag.content_len != 1 {
        return (false, DER_ERR_BAD_LENGTH_FORM)
    }
    if tag.content_start + 1 > r.end {
        return (false, DER_ERR_TRUNCATED)
    }
    let b = der_byte_at(r, tag.content_start)
    if b == 0 {
        return (false, DER_OK)
    }
    if b == 255 {
        return (true, DER_OK)
    }
    // DER forbids any nonzero byte other than 0xFF for a true BOOLEAN.
    (false, DER_ERR_BAD_LENGTH_FORM)
}

pub fn der_read_null(r: &DerReader, tag: &DerTag) -> i64 {
    if tag.tag_number != DER_TAG_NULL || tag.content_len != 0 {
        return DER_ERR_BAD_LENGTH_FORM
    }
    DER_OK
}

// Decodes a DER INTEGER as an unsigned magnitude into a BigInt. Every
// INTEGER this project ever decodes (serial numbers, RSA moduli/exponents)
// is non-negative by construction, so this treats the content bytes as an
// unsigned big-endian magnitude after stripping DER's required single
// leading 0x00 padding byte (present whenever the true value's high bit
// would otherwise look like a two's-complement sign bit).
pub fn der_read_integer(r: &DerReader, tag: &DerTag) -> (BigInt, i64) with IO {
    if tag.tag_number != DER_TAG_INTEGER || tag.content_len == 0 {
        return (bigint_zero(), DER_ERR_BAD_LENGTH_FORM)
    }
    if tag.content_start + tag.content_len > r.end {
        return (bigint_zero(), DER_ERR_TRUNCATED)
    }

    var start = tag.content_start
    var len = tag.content_len

    if len > 1 {
        let b0 = der_byte_at(r, start)
        if b0 == 0 {
            start = start + 1
            len = len - 1
            let b1 = der_byte_at(r, start)
            if b1 == 0 {
                // A second leading zero byte is non-minimal DER -- reject.
                return (bigint_zero(), DER_ERR_BAD_LENGTH_FORM)
            }
        }
    }

    // BigInt's storage is BIGINT_MAX_LIMBS (512) limbs of 16 bits = 1024
    // bytes = 8192 bits. Reject before looping rather than silently
    // truncating a value wider than BigInt can hold.
    if len > (BIGINT_MAX_LIMBS as i64) * 2 {
        return (bigint_zero(), DER_ERR_LENGTH_TOO_LARGE)
    }

    var acc = bigint_zero()
    let base256 = bigint_from_u32(256)
    var i: i64 = 0
    while i < len {
        let byte_val = der_byte_at(r, start + i)
        let digit = bigint_from_u32(byte_val as u32)
        let scaled = bigint_mul(&acc, &base256)
        acc = bigint_add(&scaled, &digit)
        i = i + 1
    }
    (acc, DER_OK)
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/asn1_primitives.sio`
Expected: prints `asn1_primitives: all cases passed` and exits 0.

- [ ] **Step 4: Commit**

```bash
git add stdlib/asn1/der.sio tests/run-pass/asn1_primitives.sio
git commit -m "feat(asn1): add der_read_boolean/der_read_null/der_read_integer"
```

---

### Task 4: `der_read_raw_bytes` and adversarial malformed-input tests

**Files:**
- Modify: `stdlib/asn1/der.sio`
- Test: `tests/run-pass/asn1_raw_bytes_and_errors.sio`

**Interfaces:**
- Consumes: everything from Tasks 1-2 (`DerReader`, `DerTag`, `der_reader_new`, `der_peek_tag`, `der_read_tlv`, `der_enter`, `der_byte_at` — note `der_byte_at` is a private `fn` in the same file, callable directly).
- Produces: `fn der_read_raw_bytes(r: &DerReader, tag: &DerTag, out: &RawBuf, out_offset: i64) -> i64 with IO`.

- [ ] **Step 1: Write the failing test**

Create `tests/run-pass/asn1_raw_bytes_and_errors.sio`:

```sio
use asn1::der::*
use net::socket::*

fn main() with IO {
    // OCTET STRING copy: 04 03 AA BB CC
    let os_buf = rawbuf_new(5)
    rawbuf_set(&os_buf, 0, 0x04)
    rawbuf_set(&os_buf, 1, 0x03)
    rawbuf_set(&os_buf, 2, 0xAA)
    rawbuf_set(&os_buf, 3, 0xBB)
    rawbuf_set(&os_buf, 4, 0xCC)
    let r1 = der_reader_new(&os_buf, 0, 5)
    let (r1b, t1, s1) = der_read_tlv(&r1)
    assert(s1 == DER_OK)
    let out1 = rawbuf_new(3)
    let copy_status1 = der_read_raw_bytes(&r1b, &t1, &out1, 0)
    assert(copy_status1 == DER_OK)
    assert(rawbuf_get(&out1, 0) == 0xAA)
    assert(rawbuf_get(&out1, 1) == 0xBB)
    assert(rawbuf_get(&out1, 2) == 0xCC)

    // BIT STRING copy: 03 02 00 F0 -- the decoder makes no interpretation
    // of the leading "unused bits" byte; it is copied like any other byte.
    let bs_buf = rawbuf_new(4)
    rawbuf_set(&bs_buf, 0, 0x03)
    rawbuf_set(&bs_buf, 1, 0x02)
    rawbuf_set(&bs_buf, 2, 0x00)
    rawbuf_set(&bs_buf, 3, 0xF0)
    let r2 = der_reader_new(&bs_buf, 0, 4)
    let (r2b, t2, s2) = der_read_tlv(&r2)
    assert(s2 == DER_OK)
    let out2 = rawbuf_new(2)
    let copy_status2 = der_read_raw_bytes(&r2b, &t2, &out2, 0)
    assert(copy_status2 == DER_OK)
    assert(rawbuf_get(&out2, 0) == 0x00)
    assert(rawbuf_get(&out2, 1) == 0xF0)

    // Adversarial: length field claims 5 bytes of content but the buffer
    // (relative to the reader's own `end`) only has 2 left. 04 05 AA BB
    // inside a reader whose end == 4.
    let trunc_buf = rawbuf_new(4)
    rawbuf_set(&trunc_buf, 0, 0x04)
    rawbuf_set(&trunc_buf, 1, 0x05)
    rawbuf_set(&trunc_buf, 2, 0xAA)
    rawbuf_set(&trunc_buf, 3, 0xBB)
    let r3 = der_reader_new(&trunc_buf, 0, 4)
    let (_, s3) = der_peek_tag(&r3)
    assert(s3 == DER_ERR_TRUNCATED)

    // Adversarial: long-form length encoding 65537 (0x83 01 00 01), one
    // over DER_MAX_LENGTH (65536). Rejected before any content bytes are
    // even expected to be present.
    let huge_buf = rawbuf_new(4)
    rawbuf_set(&huge_buf, 0, 0x04)
    rawbuf_set(&huge_buf, 1, 0x83)
    rawbuf_set(&huge_buf, 2, 0x01)
    rawbuf_set(&huge_buf, 3, 0x00)
    // Note: only 4 bytes total; the 3rd length byte (0x01) is out of range
    // for this tiny buffer, which would itself be DER_ERR_TRUNCATED -- so
    // this fixture instead uses a buffer sized to hold all 3 length bytes,
    // proving DER_ERR_LENGTH_TOO_LARGE fires rather than DER_ERR_TRUNCATED.
    let huge_buf2 = rawbuf_new(5)
    rawbuf_set(&huge_buf2, 0, 0x04)
    rawbuf_set(&huge_buf2, 1, 0x83)
    rawbuf_set(&huge_buf2, 2, 0x01)
    rawbuf_set(&huge_buf2, 3, 0x00)
    rawbuf_set(&huge_buf2, 4, 0x01)
    let r4 = der_reader_new(&huge_buf2, 0, 5)
    let (_, s4) = der_peek_tag(&r4)
    assert(s4 == DER_ERR_LENGTH_TOO_LARGE)

    // Adversarial: non-minimal long-form length. 04 81 05 <5 bytes> encodes
    // length 5 using one long-form byte, but 5 fits in short form -- this
    // is non-minimal DER and must be rejected.
    let nonmin_buf = rawbuf_new(8)
    rawbuf_set(&nonmin_buf, 0, 0x04)
    rawbuf_set(&nonmin_buf, 1, 0x81)
    rawbuf_set(&nonmin_buf, 2, 0x05)
    var i: i64 = 0
    while i < 5 {
        rawbuf_set(&nonmin_buf, 3 + i, 0xAA)
        i = i + 1
    }
    let r5 = der_reader_new(&nonmin_buf, 0, 8)
    let (_, s5) = der_peek_tag(&r5)
    assert(s5 == DER_ERR_BAD_LENGTH_FORM)

    // Adversarial: der_enter on a primitive tag must fail.
    let prim_buf = rawbuf_new(3)
    rawbuf_set(&prim_buf, 0, 0x02)
    rawbuf_set(&prim_buf, 1, 0x01)
    rawbuf_set(&prim_buf, 2, 0x05)
    let r6 = der_reader_new(&prim_buf, 0, 3)
    let (r6b, t6, s6) = der_read_tlv(&r6)
    assert(s6 == DER_OK)
    let (_, enter_status) = der_enter(&r6, &t6)
    assert(enter_status == DER_ERR_NOT_CONSTRUCTED)

    println("asn1_raw_bytes_and_errors: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/asn1_raw_bytes_and_errors.sio`
Expected: FAIL to compile — `der_read_raw_bytes` is not defined yet.

- [ ] **Step 2: Implement `der_read_raw_bytes` in `stdlib/asn1/der.sio`**

Append to `stdlib/asn1/der.sio`:

```sio
// Copies tag.content_len bytes verbatim from r's underlying buffer into
// `out` starting at `out_offset`. The caller is responsible for sizing
// `out` large enough -- matching this branch's established convention
// that RawBuf capacity checks are the caller's responsibility (see every
// rawbuf_new call site in stdlib/net/).
pub fn der_read_raw_bytes(r: &DerReader, tag: &DerTag, out: &RawBuf, out_offset: i64) -> i64 with IO {
    if tag.content_start + tag.content_len > r.end {
        return DER_ERR_TRUNCATED
    }
    var i: i64 = 0
    while i < tag.content_len {
        let b = der_byte_at(r, tag.content_start + i)
        rawbuf_set(out, out_offset + i, b)
        i = i + 1
    }
    DER_OK
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/asn1_raw_bytes_and_errors.sio`
Expected: prints `asn1_raw_bytes_and_errors: all cases passed` and exits 0.

- [ ] **Step 4: Commit**

```bash
git add stdlib/asn1/der.sio tests/run-pass/asn1_raw_bytes_and_errors.sio
git commit -m "feat(asn1): add der_read_raw_bytes and cover adversarial malformed input"
```

---

### Task 5: Integration test — decoding a real RSA public-key-shaped DER structure

**Files:**
- Test: `tests/run-pass/asn1_rsa_pubkey_integration.sio`

No changes to `stdlib/asn1/der.sio` in this task — this proves the existing surface (`der_reader_new`, `der_read_tlv`, `der_enter`, `der_read_integer`) composes correctly end-to-end against a real, independently-verified DER structure shaped exactly like PKCS#1's `RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }` — the structure this decoder ultimately exists to support extracting from an X.509 certificate's SubjectPublicKeyInfo.

**Interfaces:**
- Consumes: `der_reader_new`, `der_read_tlv`, `der_enter`, `der_read_integer` (Tasks 1-3), `bigint_cmp`, `bigint_from_u32` (from `stdlib/bignum/bigint.sio`, already used in Task 3's test).

- [ ] **Step 1: Independently re-verify the test vector before writing it into a committed test**

This reuses the exact RSA textbook vector already independently verified three separate times during the BigInt sub-project (see `docs/superpowers/plans/2026-08-23-madaros-bignum-plan.md` and `tests/run-pass/bignum_modpow_rsa.sio`): p=61, q=53, n=p*q=3233, public exponent e=17. This task only needs the DER encoding of `n` and `e`, not the modpow itself:

- `n = 3233`. In hex: 3233 = 12*256 + 161 = 0x0CA1. Big-endian bytes: `0x0C, 0xA1`. High bit of the first byte (`0x0C` = `00001100`) is 0, so no leading zero padding is needed. DER INTEGER encoding: tag `0x02`, length `0x02`, content `0x0C 0xA1`.
- `e = 17`. In hex: `0x11` (`00010001`), high bit 0, no padding needed. DER INTEGER encoding: tag `0x02`, length `0x01`, content `0x11`.
- SEQUENCE content = the two INTEGER TLVs concatenated: `(02 02 0C A1)` (4 bytes) + `(02 01 11)` (3 bytes) = 7 bytes.
- Full SEQUENCE TLV: tag `0x30`, length `0x07`, content = the 7 bytes above. Total 9 bytes: `30 07 02 02 0C A1 02 01 11`.

Before writing this into the test file, an implementer must independently recompute `12*256 + 161` and confirm it equals `3233`, and confirm `0x11 == 17`, rather than trusting this plan's arithmetic — per this project's standing "measure, don't assume" discipline (the same discipline that caught the BigInt plan's own RSA vector needing independent reverification, and separately caught the mid-BigInt-plan 256-limb truncation bug). Report the independent recomputation in the task's completion report.

- [ ] **Step 2: Write the test**

Create `tests/run-pass/asn1_rsa_pubkey_integration.sio`:

```sio
use asn1::der::*
use bignum::bigint::*
use net::socket::*

fn main() with IO {
    // RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }
    // n = 3233 = 0x0CA1, e = 17 = 0x11. See this task's Step 1 for the
    // independently-reverified byte-level derivation.
    let total_len: i64 = 9
    let buf = rawbuf_new(total_len)
    let bytes = [0x30, 0x07, 0x02, 0x02, 0x0C, 0xA1, 0x02, 0x01, 0x11]
    var i: i64 = 0
    while i < total_len {
        rawbuf_set(&buf, i, bytes[i as usize])
        i = i + 1
    }

    let top = der_reader_new(&buf, 0, total_len)
    let (top2, seq_tag, s0) = der_read_tlv(&top)
    assert(s0 == DER_OK)
    assert(seq_tag.tag_number == DER_TAG_SEQUENCE)
    assert(seq_tag.constructed == true)

    let (inner, e0) = der_enter(&top, &seq_tag)
    assert(e0 == DER_OK)

    let (inner2, modulus_tag, s1) = der_read_tlv(&inner)
    assert(s1 == DER_OK)
    assert(modulus_tag.tag_number == DER_TAG_INTEGER)
    let (modulus, ms) = der_read_integer(&inner, &modulus_tag)
    assert(ms == DER_OK)
    assert(bigint_cmp(&modulus, &bigint_from_u32(3233)) == 0)

    let after_modulus = DerReader { buf_ptr: inner2.buf_ptr, pos: modulus_tag.content_start + modulus_tag.content_len, end: inner2.end }
    let (after_modulus2, exponent_tag, s2) = der_read_tlv(&after_modulus)
    assert(s2 == DER_OK)
    assert(exponent_tag.tag_number == DER_TAG_INTEGER)
    let (exponent, es) = der_read_integer(&after_modulus, &exponent_tag)
    assert(es == DER_OK)
    assert(bigint_cmp(&exponent, &bigint_from_u32(17)) == 0)

    assert(der_at_end(&after_modulus2) == true)

    println("asn1_rsa_pubkey_integration: modulus and exponent decoded correctly")
}
```

Run: `./bin/souc run tests/run-pass/asn1_rsa_pubkey_integration.sio`
Expected: FAIL initially only if Step 1's arithmetic was wrong (the function calls all already exist from Tasks 1-3) — if it fails, recheck the byte-level derivation in Step 1 before assuming a decoder bug.

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/asn1_rsa_pubkey_integration.sio`
Expected: prints `asn1_rsa_pubkey_integration: modulus and exponent decoded correctly` and exits 0.

- [ ] **Step 4: Run the full ASN.1 test group together**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix asn1_`
Expected: 5/5 passing (`asn1_tag_length`, `asn1_navigate`, `asn1_primitives`, `asn1_raw_bytes_and_errors`, `asn1_rsa_pubkey_integration`).

- [ ] **Step 5: Commit**

```bash
git add tests/run-pass/asn1_rsa_pubkey_integration.sio
git commit -m "test(asn1): integration test decoding a real RSA public-key DER structure"
```

<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-23-madaros-asn1-der-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-23-madaros-asn1-der-design
-->

# Madaros ASN.1 DER Decoder — Design Spec

## Context and Motivation

This is sub-project 0b-ii-a of the TLS 1.2 roadmap, and the first of two layers agreed for X.509 certificate parsing: a generic, reusable ASN.1 Distinguished Encoding Rules (DER) decoder, with an X.509-specific semantic layer to follow on top of it as a separate sub-project. The split was chosen explicitly (over building one monolithic X.509 parser) for cleaner boundaries — each layer is testable in isolation, and the DER decoder itself has no X.509-specific knowledge, so it could in principle also decode other DER-encoded structures this project touches later (e.g. PKCS#1 `RSAPublicKey`, which is itself a DER SEQUENCE of two INTEGERs).

X.509 certificates arrive over the network from servers this client does not control. Unlike every prior sub-project on this branch (sockets, HTTP client, BigInt), whose test inputs were all self-crafted fixtures, this decoder's real input is **untrusted, adversarial-by-default** wire data. Malformed-input handling is therefore a first-class design concern here, not an afterthought — a hostile or buggy server sending a truncated, over-length, or deeply-nested DER blob must produce a defined error, never a crash, an infinite loop, or a silent misparse.

This spec targets Madaros v0.80.0, building on:
- **`stdlib/bignum/bigint.sio`** (sub-project 0b-i, complete) — DER INTEGER values (X.509's RSA modulus, exponent, and serial number are all encoded this way) decode directly into `BigInt`.
- **`stdlib/net/socket.sio`'s `RawBuf`/`heap_alloc` pattern** (sub-project 0a, complete) — the confirmed-safe way to hold a flat byte buffer on Madaros. `struct RawBuf { ptr: *mut u8, cap: i64 }`, `rawbuf_new(cap: i64) -> RawBuf`, `rawbuf_set(buf: &RawBuf, i: i64, v: i64)`, `rawbuf_get(buf: &RawBuf, i: i64) -> i64` (masks with `& 255` per Finding 3 — a raw pointer dereference reads a full 8-byte word, not one byte).
- **`docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`** — all 12 findings apply as background constraints; the ones this module actually exercises are called out inline below.

## Non-Goals

- **BER, only DER.** DER is a strict subset of BER: length is always encoded in definite form (never the BER-only indefinite "0x80...0x00" form), and DER additionally requires the shortest possible length/tag encoding and sorted SET OF elements. This decoder assumes and enforces DER's stricter rules — a certificate using indefinite-length encoding is rejected as malformed, not tolerated. Real X.509 certificates are always DER, so this loses no real-world coverage while meaningfully simplifying the decoder (no unbounded "read until end-of-contents marker" logic).
- **No encoder.** This is read-only — decoding bytes off the wire into structured values. Nothing in this project's roadmap needs to *produce* DER (the client never constructs or signs a certificate).
- **No BIT STRING/OCTET STRING semantic interpretation.** The decoder hands back raw bytes for these types; interpreting *what those bytes mean* (e.g. a BIT STRING containing a DER-encoded SubjectPublicKeyInfo, or an OCTET STRING containing a DER-encoded extension value) is the X.509 semantic layer's job, not this generic layer's. This is the core of the two-layer split.
- **No general-purpose tree materialization.** As decided below, this decoder is cursor/range-based, not a decoded-once-into-a-tree API. A caller who wants a snapshot of "all top-level fields of this SEQUENCE" gets there by calling the cursor API in a loop, not by receiving a pre-built tree structure.
- **No OID-to-name resolution or OID arithmetic beyond raw-byte comparison.** The decoder surfaces an OBJECT IDENTIFIER's raw content bytes (and, per Non-Goal above, leaves interpreting/dotting/naming them to the X.509 layer, which knows the specific OIDs it cares about — e.g. `rsaEncryption`, `commonName`).

## Architecture

One new module: `stdlib/asn1/der.sio`. Import convention (per the audit doc's Findings 6/8): a caller outside `stdlib/` uses `use asn1::der::*`; a future sibling file inside `stdlib/asn1/` uses the bare `use der::*` form.

### The core design decision: a cursor over a `RawBuf`, not a decoded tree

Madaros structs are fixed-size and non-recursive (no pointers/references stored across calls beyond a borrow's lifetime — see Finding 4/5's linear-type constraints, and the general absence of a heap-tree idiom anywhere else on this branch). A generic ASN.1 decoder that eagerly builds a full recursive tree of decoded nodes would need exactly that: a dynamically-sized, arbitrarily-nested tree of heap-allocated nodes. That is not a pattern this codebase or compiler has demonstrated working, and inventing one is a large, separate risk this sub-project does not need to take on.

Instead, this decoder exposes a **cursor**: an object tracking a current byte position within a fixed valid range (`[start, end)`) inside a `RawBuf` holding the raw DER bytes. Reading a TLV (Tag-Length-Value) header advances the cursor past the header to the start of the value; the caller decides whether to read the value's bytes directly (for a primitive type) or to "descend" into it (for a constructed type, SEQUENCE/SET) by creating a **child cursor** scoped to exactly that value's byte range. This maps DER's actual recursive structure onto plain, non-recursive Sounio structs: a `DerReader` is just three `i64`s (buffer handle, position, end), and "entering a SEQUENCE" is nothing more than constructing a new `DerReader` with tighter bounds — no heap tree, no dynamic allocation beyond the original input buffer.

This also directly serves the untrusted-input requirement: every read is bounds-checked against `end` before touching a byte, so a truncated or lying length field is caught at the read site as an error return, not a buffer overrun.

### Types

```sio
pub struct DerReader {
    buf: RawBuf,   // shared across a parent cursor and all its children/siblings —
                    // ASN.1 decoding never mutates the input, only reads it
    pos: i64,       // current read position, absolute offset into buf
    end: i64,       // exclusive upper bound of this cursor's valid range, absolute offset into buf
}

// A decoded TLV header: everything needed to interpret or skip the value
// that follows, without having read that value yet.
pub struct DerTag {
    class: i32,         // 0 = UNIVERSAL, 1 = APPLICATION, 2 = CONTEXT-SPECIFIC, 3 = PRIVATE
    constructed: bool,   // true if this is a constructed type (SEQUENCE, SET, or an
                          // explicitly-tagged context-specific wrapper); false if primitive
    tag_number: i32,     // for UNIVERSAL: the ASN.1 universal tag number (see constants below).
                          // for CONTEXT-SPECIFIC: the raw [N] tag number from the encoding --
                          // interpreting what tag N *means* is the X.509 layer's job, since the
                          // same number means different things in different contexts (e.g.
                          // Certificate's [0] is version, Extension's [3] is extensions).
    content_start: i64,  // absolute offset where this tag's value bytes begin
    content_len: i64,    // length of the value in bytes
}
```

### Universal tag number constants

```sio
pub const DER_TAG_BOOLEAN: i32 = 1
pub const DER_TAG_INTEGER: i32 = 2
pub const DER_TAG_BIT_STRING: i32 = 3
pub const DER_TAG_OCTET_STRING: i32 = 4
pub const DER_TAG_NULL: i32 = 5
pub const DER_TAG_OBJECT_IDENTIFIER: i32 = 6
pub const DER_TAG_UTF8_STRING: i32 = 12
pub const DER_TAG_SEQUENCE: i32 = 16   // always constructed
pub const DER_TAG_SET: i32 = 17         // always constructed
pub const DER_TAG_PRINTABLE_STRING: i32 = 19
pub const DER_TAG_IA5_STRING: i32 = 22
pub const DER_TAG_UTC_TIME: i32 = 23
pub const DER_TAG_GENERALIZED_TIME: i32 = 24

pub const DER_CLASS_UNIVERSAL: i32 = 0
pub const DER_CLASS_APPLICATION: i32 = 1
pub const DER_CLASS_CONTEXT: i32 = 2
pub const DER_CLASS_PRIVATE: i32 = 3
```

This is exactly the tag set needed for broad X.509 coverage: BOOLEAN (CA flag, critical-extension flag), INTEGER (serial number, RSA modulus/exponent, version when present as an integer), BIT STRING (subjectPublicKey, signature value), OCTET STRING (extension values), NULL (algorithm parameters for RSA), OBJECT IDENTIFIER (algorithm identifiers, extension IDs, RDN attribute types), SEQUENCE/SET (nearly everything — TBSCertificate, RDNs, AlgorithmIdentifier, Extensions), the string types (Subject/Issuer DN attribute values — X.509 names use a mix of PrintableString, UTF8String, and occasionally IA5String depending on the issuing CA and field), and UTCTime/GeneralizedTime (validity period — X.509 uses UTCTime for years before 2050 and GeneralizedTime after, per RFC 5280, so both must be recognized even though this decoder does no date-semantic interpretation itself).

### Error convention

Following this branch's established no-`Result`/no-`Option` sentinel convention (used throughout `stdlib/net/socket.sio` and `stdlib/bignum/bigint.sio`):

```sio
pub const DER_OK: i64 = 0
pub const DER_ERR_TRUNCATED: i64 = 0 - 1        // read would go past the cursor's `end`
pub const DER_ERR_BAD_LENGTH_FORM: i64 = 0 - 2  // long-form length encodes an invalid/non-minimal value
pub const DER_ERR_LENGTH_TOO_LARGE: i64 = 0 - 3 // decoded length exceeds DER_MAX_LENGTH (see below)
pub const DER_ERR_NOT_CONSTRUCTED: i64 = 0 - 4  // caller tried to descend into a primitive tag
pub const DER_ERR_NESTING_TOO_DEEP: i64 = 0 - 5 // caller-tracked nesting exceeded a sane bound
```

Every function that can fail returns `(value, i64 status)` where `status == DER_OK` on success — mirroring the `(TcpSocket, i64)` shape `stdlib/net/socket.sio` already established, so callers on this branch see a consistent pattern across modules. Per Findings 4/5/9 (linear-type over-consumption and tuple-destructuring pitfalls), functions returning a tuple that includes a plain (non-linear) struct like `DerReader` do not need the destructure-then-reconstruct workaround Finding 9 documents for *linear* types — `DerReader` is an ordinary struct, not `linear`, since a cursor has no resource (fd, socket) to leak if silently dropped. This is called out explicitly in Task 1 of the implementation plan as a fact to verify empirically before relying on it, per this project's standing "measure, don't assume" discipline — but the design does not require `DerReader` to be linear, and making it linear would only reintroduce Finding 4's both/neither-arm restriction for no benefit.

**Hard length cap:** `pub const DER_MAX_LENGTH: i64 = 65536` (64 KiB). A single DER TLV value larger than this is rejected with `DER_ERR_LENGTH_TOO_LARGE` — real X.509 certificates (including their RSA-4096 keys and full certificate chains) are comfortably under a few KiB to low tens of KiB; this cap exists purely as a defense against a malicious or corrupt length field claiming an absurd size, and is checked *before* any attempt to read that many bytes, so it costs nothing on well-formed input and fails fast on hostile input.

### Operations

**Constructing a reader**: `der_reader_new(buf: &RawBuf, start: i64, end: i64) -> DerReader` — wraps an existing `RawBuf` (the caller owns reading it off the network into the buffer beforehand, exactly as `stdlib/net/http_client.sio` already does for HTTP response bodies) with a byte range. The top-level call for a freshly-received certificate is `der_reader_new(cert_buf, 0, cert_len)`.

**Peeking a tag without consuming it**: `der_peek_tag(r: &DerReader) -> (DerTag, i64 status)` — decodes the tag byte(s) and length field starting at `r.pos`, without advancing `r.pos`. Used when a caller needs to decide *how* to handle the next element (e.g. "is this a context-specific [0] wrapper, or does the SEQUENCE go straight into its first real field?") before committing to consume it.

**Reading and advancing**: `der_read_tlv(r: &DerReader) -> (DerReader, DerTag, i64 status)` — decodes the tag+length exactly as `der_peek_tag` does, but returns an updated `DerReader` (a fresh value, per the plain-struct convention above — no mutation-through-reference needed, since `DerReader` is small and cheap to copy) whose `pos` has advanced past the header to `content_start`. This is the primary iteration primitive: a caller processing a SEQUENCE's fields calls `der_read_tlv` repeatedly, each time getting back both the decoded header and an advanced reader positioned at the start of that field's value.

**Tag decoding detail** (what `der_peek_tag`/`der_read_tlv` do internally): DER's tag byte packs class (bits 7-6), constructed flag (bit 5), and tag number (bits 4-0) into one byte, unless the low 5 bits are all `1` (tag number ≥ 31), in which case the tag number continues in subsequent base-128 bytes with a high-bit continuation marker — a case that does not arise for any tag this module needs to recognize (all universal tags used here are ≤ 24, and this design intentionally does not attempt to parse multi-byte tag numbers, since no field on X.509's roadmap needs one; encountering a multi-byte tag number is treated as a decode the same way as any other tag, just recorded with the raw first-byte-derived `tag_number` truncated to those low 5 bits — this is a documented limitation, not a silent misparse, since nothing in this project's scope produces or expects a multi-byte tag number). The length byte follows one of two forms: **short form** (high bit clear, value is the length directly, 0-127) or **long form** (high bit set, low 7 bits give the count of subsequent big-endian length bytes, which are then read and combined). Per DER's strictness (vs. BER), a long-form length must use the minimum number of subsequent bytes needed (no leading zero byte) — `der_peek_tag` checks this and returns `DER_ERR_BAD_LENGTH_FORM` if violated, and separately checks the decoded value against `DER_MAX_LENGTH` before returning `DER_ERR_LENGTH_TOO_LARGE`.

**Byte-width safety for length decoding**: long-form length bytes are combined via `len = (len << 8) | byte`, accumulated as an `i64`. Per the audit doc's Finding 11, `u64` right-shift/divide/modulo are the operations proven broken near bit 63 — this code uses only `<<` and `|` (both confirmed safe unconditionally per Finding 11's own operation list) and works in `i64`, not `u64`, and the loop is additionally bounded to at most 4 length bytes (enough to represent `DER_MAX_LENGTH` many times over) before returning `DER_ERR_BAD_LENGTH_FORM`, so the accumulator never approaches even 32 bits, let alone bit 63.

**Reading primitive values**: rather than one generic "read N raw bytes" function that every caller re-interprets, the decoder provides typed readers that consume a `DerTag` (already obtained from `der_read_tlv`) plus the `DerReader` positioned at its content:
- `der_read_integer(r: &DerReader, tag: &DerTag) -> (BigInt, i64 status)` — DER INTEGER is big-endian two's-complement, but every INTEGER this project ever decodes (serial numbers, RSA moduli/exponents) is non-negative by construction (a negative serial number or modulus is nonsensical and, per RFC 5280, non-conformant), so this function treats the content bytes as an unsigned big-endian magnitude, with one required check: DER encodes a non-negative INTEGER whose high bit would otherwise look negative by prefixing a `0x00` padding byte — this function strips a single leading `0x00` byte if present (and only one — a second leading zero byte is invalid DER and returns `DER_ERR_BAD_LENGTH_FORM`, reusing that status for "malformed content", not just malformed length, to avoid multiplying the error-constant list) before converting the remaining bytes into a `BigInt` via repeated `bigint_from_u32`-then-shift-and-add composition (byte-by-byte: `acc = bigint_add(bigint_mul(acc, bigint_from_u32(256)), bigint_from_u32(byte))`), reusing `BigInt`'s own confirmed-safe arithmetic rather than reimplementing big-endian-bytes-to-bignum conversion by hand.
- `der_read_raw_bytes(r: &DerReader, tag: &DerTag, out: &RawBuf, out_offset: i64) -> i64 status` — for BIT STRING, OCTET STRING, and every string type: copies `tag.content_len` bytes verbatim from the input buffer into a caller-supplied output `RawBuf` at `out_offset`, byte by byte via `rawbuf_get`/`rawbuf_set` (mirroring the copy-loop pattern `stdlib/net/http_client.sio` already uses for buffer-to-buffer transfers). The caller is responsible for sizing `out` large enough — this function returns `DER_ERR_TRUNCATED` if `tag.content_len` would read past `r.end`, but does not itself bounds-check `out`'s capacity (matching this branch's established convention that `RawBuf` capacity checks are the allocator/caller's responsibility, as `rawbuf_new`'s own callers already do throughout `stdlib/net/`). BIT STRING has one DER-specific wrinkle worth documenting even though this layer doesn't interpret it: its first content byte is the "number of unused bits in the last content byte" (always `0x00` for a DER-encoded key or signature, since those are always whole-byte-aligned) — `der_read_raw_bytes` copies this leading byte as part of the raw content like any other, leaving it to the X.509 layer to skip it when it knows it's looking at a BIT STRING wrapping a byte-aligned value.
- `der_read_boolean(r: &DerReader, tag: &DerTag) -> (bool, i64 status)` — DER BOOLEAN is exactly one content byte, `0x00` for false and `0xFF` for true (DER, unlike BER, forbids any other nonzero byte for true); a value outside `{0x00, 0xFF}` returns `DER_ERR_BAD_LENGTH_FORM` (reused again as the general "malformed content" status).
- `der_read_null(r: &DerReader, tag: &DerTag) -> i64 status` — DER NULL always has `content_len == 0`; this function's only job is to confirm that and return `DER_OK` or `DER_ERR_BAD_LENGTH_FORM`. (Used by the X.509 layer to validate an RSA AlgorithmIdentifier's parameters field, which is conventionally NULL.)

**Descending into a constructed value**: `der_enter(r: &DerReader, tag: &DerTag) -> (DerReader, i64 status)` — given a `DerTag` with `constructed == true` (SEQUENCE, SET, or an explicitly-tagged wrapper) and the `DerReader` positioned at its `content_start`, returns a **child** `DerReader` with `pos = tag.content_start` and `end = tag.content_start + tag.content_len` — a narrower view over the same underlying `buf`. The caller then calls `der_read_tlv` against this child reader to walk the constructed value's own fields, exactly as it would at the top level. Returns `DER_ERR_NOT_CONSTRUCTED` if `tag.constructed` is false — descending into a primitive tag is always a caller bug or a malformed-input signal, never silently tolerated.

**Skipping a value without reading it**: `der_skip(r: &DerReader, tag: &DerTag) -> (DerReader, i64 status)` — returns an updated `DerReader` with `pos` advanced past `tag.content_len` bytes (i.e. to the position immediately after this whole TLV, whether primitive or constructed), without reading or copying anything. This is essential for broad-but-shallow X.509 coverage: a caller processing `TBSCertificate`'s SEQUENCE of fields, needing only (say) `serialNumber` and `validity`, calls `der_skip` on fields it does not care about (e.g. `issuerUniqueID`, or an extension it doesn't recognize) rather than being forced to descend into and interpret every field DER-encodes. Bounds-checked against `r.end` exactly like every other advance — returns `DER_ERR_TRUNCATED` if `tag.content_len` would overrun.

**Detecting end-of-contents**: `der_at_end(r: &DerReader) -> bool` — `r.pos >= r.end`. The idiomatic loop for "read every field of this SEQUENCE" is:
```sio
var cursor = child_reader   // from der_enter
while der_at_end(&cursor) == false {
    let (advanced, tag, status) = der_read_tlv(&cursor)
    if status != DER_OK {
        // handle error
    }
    cursor = advanced
    // ... inspect tag.class / tag.tag_number / tag.constructed, then either
    // der_enter, one of the der_read_* functions, or der_skip on `cursor` ...
}
```

**Nesting-depth guard**: since `der_enter` can be called recursively by a caller walking an arbitrarily-shaped certificate (SEQUENCE containing a SEQUENCE containing a SEQUENCE...), and this decoder has no tree to bound recursion structurally, the *caller* (the X.509 layer, or a test) is responsible for tracking its own recursion depth and treating an excessive depth as `DER_ERR_NESTING_TOO_DEEP` — this generic layer does not track call-stack depth itself (it has no state that spans nested `der_enter` calls to hang a counter off), but this constraint is documented here so the X.509 layer's design inherits it as a stated requirement rather than discovering it the hard way against a hostile deeply-nested certificate. A suggested bound (for the X.509 layer to adopt): 16 levels, comfortably deeper than any legitimate X.509 structure (which nests at most 5-6 levels: Certificate → TBSCertificate → Extensions → Extension → extnValue's inner SEQUENCE → that SEQUENCE's own fields), while still bounding a malicious "SEQUENCE containing a SEQUENCE containing a SEQUENCE... one thousand times" input from ever being fully descended into.

## Testing Strategy

Following this branch's established pattern (hand-computable cases → a real-world vector → a boundary/adversarial case):

1. **Hand-constructed minimal DER fixtures**, built byte-by-byte in the test file itself (writing the raw bytes for, e.g., `INTEGER 65537` — `02 03 01 00 01` — and asserting the decoder recovers `65537`), covering: short-form length, long-form length (a value forcing a >127-byte content, e.g. a 200-byte OCTET STRING), each universal tag type in the constants list, a nested SEQUENCE-within-a-SEQUENCE, a context-specific tag, and the INTEGER leading-zero-padding case (`INTEGER 128` must encode as `02 02 00 80`, not `02 01 80` which would look negative — decode both the correct and, separately in a `compile-fail`-style negative test, a deliberately-malformed doubled-zero-padding case and confirm it's rejected).
2. **A real DER-encoded structure decoded from bytes actually seen on the wire** — reuse the RSA test vector's spirit from the BigInt module: construct the DER bytes for a minimal `RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }` (PKCS#1 format) using a small, independently-verifiable modulus/exponent pair, and confirm `der_enter` + two `der_read_integer` calls recover exactly those two `BigInt` values. This is the closest this layer can get to an "it actually works for X.509" proof without the X.509 semantic layer existing yet.
3. **Adversarial/malformed-input cases**, each expected to return a specific error status, not a crash: a length field claiming more bytes than the buffer actually contains (`DER_ERR_TRUNCATED`), a length field exceeding `DER_MAX_LENGTH` (`DER_ERR_LENGTH_TOO_LARGE`), a non-minimal long-form length encoding (`DER_ERR_BAD_LENGTH_FORM`), and a `der_enter` call against a primitive (non-constructed) tag (`DER_ERR_NOT_CONSTRUCTED`).

All tests as `tests/run-pass/asn1_*.sio` (or `tests/compile-fail/` where a case is about a rejected input triggering a normal run-time error return rather than a compile-time failure — per this project's test-file convention, a run-time-rejected malformed input is still `run-pass` at the Sounio-compilation level, since the test program itself compiles and runs successfully to completion, asserting on the returned error status; `tests/compile-fail/` remains reserved for inputs that fail to *compile*, which is not what adversarial DER fixtures exercise), invoked via `bash scripts/run_sio_test_suite.sh --filter-prefix asn1_`.

## Open Implementation-Time Questions

- Whether `DerReader` genuinely needs no `linear` annotation (confirm empirically, as flagged in the Error Convention section above) — a five-minute check with a throwaway test file before committing to the non-linear design across every function signature in the plan.
- The exact byte-copy loop shape for `der_read_raw_bytes` (`rawbuf_get`/`rawbuf_set` per byte) should be checked against `stdlib/net/http_client.sio`'s existing buffer-copy code for the idiom Madaros compiles most reliably, rather than inventing a new loop shape from scratch.
- Whether Madaros's `bool` type and `true`/`false` literals behave as expected for `der_read_boolean`'s return type and `der_at_end`'s — this project has not yet used `bool` as a return type anywhere on this branch (`socket.sio` and `bigint.sio` both stick to `i64` sentinels throughout), so this should be verified empirically in Task 1 rather than assumed to just work.

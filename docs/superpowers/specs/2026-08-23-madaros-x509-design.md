<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-23-madaros-x509-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-23-madaros-x509-design
-->

# Madaros X.509 Semantic Layer — Design Spec

## Context and Motivation

This is sub-project 0b-ii-b of the TLS 1.2 roadmap: the second of two layers agreed for certificate handling, built directly on top of the now-complete, generic `stdlib/asn1/der.sio` decoder (sub-project 0b-ii-a) and the now-complete `stdlib/hash/{sha1,sha256,sha384,sha512}.sio` hash functions (sub-project, this session). Where the DER decoder has no X.509-specific knowledge, this layer's entire purpose is to know what an X.509 certificate's fields *mean*: it decodes a `Certificate` DER structure into a queryable, fixed-size struct, extracts the RSA public key and every field a TLS 1.2 client needs, and — per this project's explicit scope decision — verifies the certificate's own signature against its issuer's public key using RSASSA-PKCS1-v1_5-VERIFY (RFC 8017 §8.2.2). It does **not** validate a trust chain (no CA bundle, no root-of-trust walk) or perform hostname verification — those are later, separate sub-projects, exactly as originally scoped.

Coverage is **broad**, not minimal-viable-RSA-only, per the explicit decision made during brainstorming: this layer targets essentially every field and extension a real CA-issued TLS certificate carries in practice (RFC 5280), not just the handful strictly required to complete a bare RSA key exchange.

## Non-Goals

- **No trust-chain validation.** No CA bundle, no issuer-lookup, no path-building, no revocation checking (no live OCSP/CRL fetch — the URLs in `authorityInfoAccess`/`cRLDistributionPoints` are extracted as data, never dereferenced by this layer). A later sub-project consumes this layer's output to walk a chain.
- **No hostname verification** (matching a SAN/CN against the server name dialed). Also a later sub-project, consuming this layer's decoded SAN list.
- **No private-key RSA operations.** This layer only ever does the RSA *public*-key operation (verifying a signature with the issuer's public key) — never signs or decrypts with a private key. No constant-time requirement, per the same reasoning already established for the BigInt sub-project.
- **No SCT signature verification.** SCTs (Certificate Transparency) are fully decoded into structured entries (log ID, timestamp, signature bytes, per RFC 6962) — but this layer does not verify an SCT's own signature against a CT log's public key. That would require a CT-log public-key registry this project has no source for yet; decoding the structure is valuable on its own (exposing what's there) without that verification step.
- **No BER tolerance beyond what `stdlib/asn1/der.sio` already provides.** This layer inherits the DER decoder's strictness (indefinite-length, non-minimal encodings, negative INTEGERs, constructed-where-primitive-required all already rejected at that layer). This layer adds no additional leniency.
- **No support for non-RSA public keys or non-RSA signature algorithms** (no ECDSA, no Ed25519). The TLS 1.2 roadmap's chosen cipher suite is static RSA key exchange with RSA-signed certificates; a certificate using a different key/signature algorithm is decoded up to the point where its `AlgorithmIdentifier` OID is read, then rejected with a clear "unsupported algorithm" status rather than silently mis-parsing its key/signature bytes as if they were RSA's.

## Architecture

Three new files, plus one addition to the existing crypto module:

```
stdlib/crypto/pkcs1.sio    -- RSASSA-PKCS1-v1_5-VERIFY (RFC 8017 §8.2.2)
stdlib/x509/oid.sio         -- known OID byte-sequence constants + comparison
stdlib/x509/cert.sio        -- Certificate struct, parser, field extraction
stdlib/x509/sct.sio         -- RFC 6962 Signed Certificate Timestamp list decoder
```

`stdlib/crypto/pkcs1.sio` lives in the **pre-existing** `stdlib/crypto/` directory (which already holds an unrelated, incremental-API SHA-256+HMAC implementation, `stdlib/crypto/sha256.sio`/`hmac.sio` — see `stdlib/hash/README.md` for why that module and the newer `stdlib/hash/` module coexist). PKCS#1 verification is conceptually an RSA/crypto primitive, not X.509-specific, and belongs there even though it imports `stdlib/hash::shaN::*` (the `&RawBuf`-oriented module, needed because the bytes being hashed — a `TBSCertificate`) come from a `RawBuf`, not a fixed small buffer) and `stdlib/bignum::bigint::*` for the modular exponentiation and byte conversion.

Import convention (per the audit doc's Findings 6/8): callers outside `stdlib/` use `use x509::cert::*` (etc.); a sibling file inside `stdlib/x509/` uses the bare `use oid::*` form.

## The core open risk: arrays of nested structs

Every array-based data structure on this branch so far (`BigInt.limbs: [u16;512]`, the DER decoder's byte buffers, the hash modules' message schedules) has been an array of **plain scalars**. This layer's data model needs several **arrays of structs** — `[RdnEntry; 16]`, `[ExtensionEntry; 32]`, `[SctEntry; 8]`, `[GeneralName; 32]` — a pattern that has never been exercised anywhere on this branch. Worse, the richest design for `GeneralName` (see below) would need to embed one struct type (`X509Name`, for the `directoryName` SAN variant) inside another struct (`GeneralName`) that is itself an array element — array-of-struct-containing-struct, a still-deeper untested pattern.

Per this project's established discipline (mirroring the hash sub-project's arithmetic audits), the implementation plan's **first task is a dedicated empirical audit** of this pattern before any of the rest of this layer's code depends on it: declare a small array of a multi-field struct, populate it in a loop, read fields back by index, and — separately — declare a struct containing another struct as a field, inside an array. Both checks must pass before the richer design below is adopted as-is.

**If the audit finds this pattern broken** (a real possibility, given this branch's track record of struct/array interactions surfacing bugs — Finding 2's array-as-raw-pointer issue, the hash sub-project's five new findings), the fallback design is documented now, not discovered mid-implementation:
- Every `[Thing; N]` array-of-struct becomes `N` parallel scalar/small-fixed-array fields instead (the same technique `stdlib/hash/word64.sio` used to avoid `[U64Pair; 80]` — an array of a *pair* type — by using two parallel `[i64; 80]` arrays instead), OR, where the element count is genuinely needed with indexed access, the struct is flattened into several parallel arrays of its individual scalar fields (e.g. `rdn_oids: [[u8;20]; 16]`, `rdn_oid_lens: [i32; 16]`, `rdn_value_tags: [i32; 16]`, `rdn_values: [[u8;128]; 16]`, `rdn_value_lens: [i32; 16]` instead of `rdn_entries: [RdnEntry; 16]`) — more verbose, but built entirely from patterns (arrays of scalars, arrays of fixed-size byte arrays) already proven safe.
- `GeneralName`'s `directoryName` variant specifically falls back to **raw, unparsed bytes** (like the exotic `x400Address`/`ediPartyName` variants) instead of a nested `X509Name` — this is the one piece of "broad coverage" this spec is willing to give up if the nested-struct-in-array pattern doesn't work, since `directoryName` SANs are rare in practice (almost all real-world SANs are `dNSName`, `iPAddress`, or `rfc822Name`).

The implementation plan must record which of these two paths (rich or fallback) it actually took, in the same style as the hash sub-project's audit-driven findings.

## Data Structures

```sio
pub const X509_MAX_RDN_ENTRIES: i32 = 16
pub const X509_MAX_NAME_VALUE_LEN: i64 = 128
pub const X509_MAX_OID_BYTES: i64 = 20
pub const X509_MAX_SAN_ENTRIES: i32 = 32
pub const X509_MAX_GENERAL_NAME_VALUE_LEN: i64 = 253   // RFC 1035 max DNS name length; also comfortably covers IPv6 text and typical URIs/emails
pub const X509_MAX_EXTENSIONS: i32 = 32
pub const X509_MAX_SCT_ENTRIES: i32 = 8

pub struct RdnEntry {
    oid: [u8; 20],
    oid_len: i32,
    value_tag: i32,        // DER_TAG_PRINTABLE_STRING / UTF8_STRING / IA5_STRING / etc.
    value: [u8; 128],
    value_len: i32,
}

pub struct X509Name {
    entries: [RdnEntry; 16],
    count: i32,
}

// GeneralName tag values (the SAN/issuerAltName CHOICE, RFC 5280 §4.2.1.6):
pub const GENERAL_NAME_OTHER_NAME: i32 = 0
pub const GENERAL_NAME_RFC822_NAME: i32 = 1     // email
pub const GENERAL_NAME_DNS_NAME: i32 = 2
pub const GENERAL_NAME_X400_ADDRESS: i32 = 3
pub const GENERAL_NAME_DIRECTORY_NAME: i32 = 4
pub const GENERAL_NAME_EDI_PARTY_NAME: i32 = 5
pub const GENERAL_NAME_URI: i32 = 6
pub const GENERAL_NAME_IP_ADDRESS: i32 = 7
pub const GENERAL_NAME_REGISTERED_ID: i32 = 8

pub struct GeneralName {
    tag: i32,                          // one of the GENERAL_NAME_* constants above
    value: [u8; 253],                   // rfc822Name/dNSName/URI: ASCII text bytes.
                                          // iPAddress: 4 (IPv4) or 16 (IPv6) raw bytes, length in value_len.
                                          // otherName/registeredID: the OID's raw bytes (see oid/oid_len below) --
                                          //   otherName's associated value is NOT captured (rare in practice; the
                                          //   OID alone identifies the "other name" type, which is usually enough).
                                          // x400Address/ediPartyName: raw, unparsed content bytes (legacy formats
                                          //   this project has no need to interpret further).
    value_len: i32,
    oid: [u8; 20],                       // populated only for otherName/registeredID; oid_len == 0 otherwise
    oid_len: i32,
    directory_name: X509Name,            // populated only if tag == GENERAL_NAME_DIRECTORY_NAME AND the
                                           // array-of-nested-struct audit (above) passes; otherwise this field
                                           // is a fallback design and directoryName's raw bytes land in `value`
                                           // instead, identical to x400Address/ediPartyName's handling.
}

pub struct ExtensionEntry {
    oid: [u8; 20],
    oid_len: i32,
    critical: bool,
    value: [u8; 512],       // the extension's raw OCTET STRING content (extnValue), NOT yet semantically
                              // decoded -- decoding a specific extension's meaning (e.g. interpreting
                              // basicConstraints' inner SEQUENCE) is a function the caller invokes on
                              // demand, keyed by matching `oid` against stdlib/x509/oid.sio's constants,
                              // not something this generic extension-list struct does automatically.
    value_len: i32,
}

pub struct SctEntry {
    // RFC 6962 §3.2 SignedCertificateTimestamp structure.
    version: i32,            // must be 0 (v1) for any SCT this project recognizes
    log_id: [u8; 32],        // SHA-256 hash identifying the CT log
    timestamp_ms: i64,        // milliseconds since epoch, big-endian 64-bit in the wire format
    extensions_len: i32,      // SCT extensions are rare in practice; captured as a length only, not decoded
    sig_algorithm_hash: i32,   // hash algorithm ID byte (RFC 6962 §3.2, mirrors TLS 1.2 SignatureAndHashAlgorithm)
    sig_algorithm_sign: i32,   // signature algorithm ID byte
    signature: [u8; 128],      // raw signature bytes (RSA-2048 signatures are up to 256 bytes; ECDSA typically
                                 // under 80 -- 128 is a documented, adjustable bound, not a hard protocol limit)
    signature_len: i32,
}

pub struct Certificate {
    // -- Outer Certificate fields (RFC 5280 §4.1) --
    tbs_start: i64,             // absolute byte offset of the TBSCertificate SEQUENCE's own tag byte
    tbs_len: i64,                // length of the TBSCertificate SEQUENCE's full TLV (tag+length+content) --
                                   // this exact byte range, re-read from the original input buffer, is what
                                   // gets hashed for signature verification, per RFC 5280 §4.1.1.3
    outer_sig_alg_oid: [u8; 20],
    outer_sig_alg_oid_len: i32,
    outer_signature: [u8; 512],   // BIT STRING content (minus the leading unused-bits byte), RSA-4096 max is 512 bytes
    outer_signature_len: i32,

    // -- TBSCertificate fields --
    version: i32,                 // 0, 1, or 2 (meaning v1/v2/v3); DEFAULT 0 if the optional [0] field is absent
    serial_number: BigInt,
    tbs_sig_alg_oid: [u8; 20],     // must match outer_sig_alg_oid -- a mismatch is a malformed/tampered certificate
    tbs_sig_alg_oid_len: i32,
    issuer: X509Name,
    not_before_unix: i64,          // validity start, Unix timestamp (decoded from UTCTime only as shipped -- GeneralizedTime, needed for dates 2050+, is a documented gap; see stdlib/x509/cert.sio's parse_utc_time_to_unix)
    not_after_unix: i64,           // validity end, Unix timestamp
    subject: X509Name,
    modulus: BigInt,                // subjectPublicKeyInfo's RSA modulus
    public_exponent: BigInt,        // subjectPublicKeyInfo's RSA public exponent

    // -- Extensions (v3 only; absent on a v1 certificate) --
    extensions: [ExtensionEntry; 32],
    extension_count: i32,
    is_ca: bool,                    // decoded from basicConstraints, if present; false if absent (RFC 5280 default)
    path_len_constraint: i32,        // basicConstraints' pathLenConstraint; -1 if absent/not applicable
    key_usage_bits: i32,             // raw KeyUsage BIT STRING, as an integer bitmask (bit N = KeyUsage's bit N)
    san_entries: [GeneralName; 32],
    san_count: i32,
    issuer_alt_names: [GeneralName; 32],
    issuer_alt_name_count: i32,
    subject_key_id: [u8; 32],
    subject_key_id_len: i32,
    authority_key_id: [u8; 32],
    authority_key_id_len: i32,
    sct_entries: [SctEntry; 8],
    sct_count: i32,
    // authorityInfoAccess, cRLDistributionPoints, certificatePolicies, nameConstraints, policyConstraints,
    // inhibitAnyPolicy, freshestCRL, subjectDirectoryAttributes: all captured generically via the
    // `extensions` list above (matched by OID, decoded on demand by a caller) rather than given their own
    // dedicated struct fields -- these are lower-frequency-access fields for this project's immediate TLS
    // handshake needs (identity, validity, key material, CA/key-usage flags, SAN, SCT are the fields a
    // handshake actually consults every connection); a caller that needs one of these rarer fields finds
    // its OID in `extensions` and decodes the DER content itself via the ASN.1 decoder, exactly as this
    // module's own parser does internally.
}
```

**Zero representation / "absent" convention**: every `_count`/`_len` field of `0` means "this list/value is absent or empty" — mirroring `BigInt`'s `len == 0` convention. `path_len_constraint == -1` is the sentinel for "not applicable" (distinct from a legitimate `pathLenConstraint` of `0`, which is a valid, meaningful CA constraint).

## PKCS#1 v1.5 Signature Verification

`stdlib/crypto/pkcs1.sio`:

```sio
pub const PKCS1_HASH_SHA1: i32 = 1
pub const PKCS1_HASH_SHA256: i32 = 2
pub const PKCS1_HASH_SHA384: i32 = 3
pub const PKCS1_HASH_SHA512: i32 = 4

pub fn pkcs1_v15_verify(
    signature: &BigInt,
    modulus: &BigInt,
    exponent: &BigInt,
    modulus_byte_len: i64,     // k in RFC 8017's notation; the certificate's own encoded modulus length
    hash_algorithm: i32,        // one of the PKCS1_HASH_* constants above
    message_hash: &[u8; 64],    // the actual hash bytes; only the first N are meaningful, N depends on hash_algorithm
    message_hash_len: i64,
) -> bool
```

Behavior: computes `m = signature^exponent mod modulus` via `bigint_modpow` (already built), converts `m` to a big-endian byte string of exactly `modulus_byte_len` bytes (a new helper local to this file — per the brainstorm decision, `bigint_to_bytes_be` is NOT added to the shared `bigint.sio` module; it stays local to `pkcs1.sio` since it's a PKCS#1-specific serialization need, not a generic BigInt operation this project needs elsewhere), and compares that byte string against the expected `0x00 0x01 <0xFF padding> 0x00 <DigestInfo prefix for hash_algorithm> <message_hash bytes>` construction, byte for byte. Returns `true` only on an exact match of every byte, including the correct padding length (`modulus_byte_len - 3 - DigestInfo_length - hash_length` bytes of `0xFF`).

The four `DigestInfo` prefixes (RFC 8017 Appendix, well-known fixed byte sequences per hash algorithm) are defined as local constants in this file, one per `PKCS1_HASH_*` value — this is the "parameterized enough to support all four already-built hash algorithms" form of generality: no generic ASN.1 DigestInfo parsing (a caller with a fifth, hypothetical future hash algorithm would add one more prefix constant and one more `PKCS1_HASH_*` value, not change this function's logic).

## SCT List Decoding (RFC 6962 §3.2-3.3)

The SCT extension's `extnValue` OCTET STRING contains, itself, an OCTET STRING wrapping a `SignedCertificateTimestampList` — a **length-prefixed binary format**, not ASN.1/DER: a 2-byte big-endian total-length field, followed by a sequence of `(2-byte length, SCT bytes)` entries. Each individual SCT is: 1 byte version, 32 bytes log ID, 8 bytes big-endian timestamp (milliseconds since epoch), 2-byte extensions length + that many extension bytes, 1 byte hash algorithm, 1 byte signature algorithm, 2-byte signature length + that many signature bytes.

`stdlib/x509/sct.sio` implements this as a small, self-contained cursor-based reader — structurally similar in spirit to `stdlib/asn1/der.sio`'s `DerReader` (bounds-checked reads via a position+end cursor over a `RawBuf`) but for this entirely different, non-ASN.1 wire format. `sct_list_decode(buf: &RawBuf, start: i64, len: i64) -> ([SctEntry; 8], i32, i64)` returns the decoded entries, a count, and a status (reusing the `i64` sentinel convention: `0` for OK, negative for a bounds/format error) — capped at `X509_MAX_SCT_ENTRIES` (8); a certificate with more SCTs than that has the excess silently dropped with the count capped, which is `log()`-worthy per this project's "no silent caps" testing ethos but is not expected to occur in practice (real certificates carry 2-3 SCTs almost universally).

## Testing Strategy

Following this project's established pattern (hand-built fixtures → a real-world artifact → adversarial/malformed input):

1. **Hand-built DER fixtures** for each individual field-extraction function (a minimal `TBSCertificate` with just a serial number and validity dates, decoded and checked field-by-field; a minimal `Extensions` SEQUENCE with one `basicConstraints`; a minimal SAN with one of each `GeneralName` variant; a minimal SCT list with one synthetic SCT entry) — isolates each piece of logic from the others.
2. **A real, complete certificate.** Obtain the DER bytes of one real, publicly-issued TLS certificate (e.g. by connecting to a well-known site with the already-built `stdlib/net/http_client.sio`'s TCP layer and manually extracting the certificate bytes from a captured TLS handshake with an external tool during test authoring — NOT by implementing TLS itself yet, which is a later sub-project; or, more simply, exporting a real certificate's DER bytes via `openssl s_client`/`openssl x509` during test authoring and embedding the resulting byte array as a fixture, the same way the ASN.1 sub-project embedded a hand-derived RSA-key-shaped fixture). Decode it fully — every field this spec lists — and independently cross-check the decoded values (serial number, validity dates, subject/issuer CN, SAN entries, key modulus) against the same certificate's `openssl x509 -text` output. Then verify its signature via `pkcs1_v15_verify` against its own issuer's public key (which, for a real chain, means either finding the issuer's certificate too, or — more simply for this layer's own isolated test — using a certificate signed by a well-known, low-depth CA whose public key can be independently obtained, or a self-signed test certificate generated during test authoring specifically so the "issuer" key is the certificate's own key).
3. **Adversarial cases**: a certificate whose outer signature algorithm OID doesn't match its TBSCertificate's inner one (must be flagged); a tampered single byte inside the TBSCertificate range (signature verification must fail); a certificate exceeding one of the fixed-size bounds (e.g. more than 32 extensions, or an RDN value longer than 128 bytes) — must be rejected cleanly (a defined error status), not overflow.

All tests as `tests/run-pass/x509_*.sio`, run via `bash scripts/run_sio_test_suite.sh --filter-prefix x509_`.

## Open Implementation-Time Questions

- **The array-of-nested-struct audit's outcome** (see "The core open risk" section above) determines whether `GeneralName.directory_name: X509Name` ships as designed or falls back to raw bytes — this must be resolved as literally the first task of the implementation plan, before any other code in this layer is written, exactly as Findings 13/16 gated the hash sub-project's own two phases.
- Whether Madaros's `bool` struct field type (used here for `ExtensionEntry.critical`, `Certificate.is_ca`) behaves as expected — `bool` as a return type is already proven (`stdlib/asn1/der.sio`'s `der_at_end`), and `bool` as a struct field was already used successfully in `stdlib/asn1/der.sio`'s `DerTag.constructed` — this is a repeat of an already-proven pattern, not a new risk, but worth a one-line confirmation early rather than assumed silently.
- The exact real-world certificate fixture (which site/CA) to embed as the Testing Strategy's item 2 is left to the implementation plan to choose and freeze at plan-writing time, not decided in this spec.

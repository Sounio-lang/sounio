<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-24-madaros-x509-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-24-madaros-x509-plan
-->

# Madaros X.509 Semantic Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `stdlib/x509/{oid,cert,sct}.sio` and `stdlib/crypto/pkcs1.sio` — a full X.509 certificate parser (broad RFC 5280 field/extension coverage) with RSASSA-PKCS1-v1_5-VERIFY signature verification, on top of the already-complete `stdlib/asn1/der.sio` decoder, `stdlib/bignum/bigint.sio`, and `stdlib/hash/{sha1,sha256,sha384,sha512}.sio`.

**Architecture:** A cursor-driven parser walking a `Certificate`'s DER structure via `stdlib/asn1/der.sio`'s existing `der_read_tlv`/`der_enter`/`der_skip`/`der_next`/`der_read_integer`/`der_read_boolean`/`der_read_raw_bytes` primitives, populating one large fixed-size `Certificate` struct. Extensions are captured generically (OID + raw bytes) and a handful of high-frequency ones (basicConstraints, keyUsage, SAN, key identifiers, SCTs) get dedicated semantic decoding; the rest stay as raw bytes a caller can decode on demand. Signature verification re-hashes the certificate's own `TBSCertificate` byte range (captured during parsing, not re-derived) and checks it against the issuer's public key via PKCS#1 v1.5.

**Tech Stack:** Sounio (Madaros v0.80.0), reusing `stdlib/asn1/der.sio`, `stdlib/bignum/bigint.sio`, `stdlib/hash/*`, `stdlib/net/socket.sio`'s `RawBuf`.

**Spec:** `docs/superpowers/specs/2026-08-23-madaros-x509-design.md`

## Global Constraints

- **Task 1's array-of-nested-struct audit gates the whole plan's data-structure design.** Read its two-tier outcome before writing Task 2: (a) a plain array of a flat multi-field struct (no nested struct fields) must work, or the entire plan needs a controller-level redesign (STOP, report BLOCKED, do not improvise a full flatten yourself — this is bigger than one task's fix). (b) an array of a struct that itself contains ANOTHER struct as a field (the `GeneralName.directory_name: X509Name` case) is a narrower, independently gate-able risk — if (a) passes but (b) fails, only `GeneralName` needs the documented fallback (raw bytes instead of a nested `X509Name`); everything else in this plan (`RdnEntry`, `ExtensionEntry`, `SctEntry` — none of which nest another struct) is unaffected and proceeds as designed.
- **Never copy a struct by dereferencing a reference (`*some_ref`)** — rebuild via a fresh struct literal reading fields individually, per the ASN.1 sub-project's established discipline. Where a struct is an array element and needs a field updated, prefer whole-element reassignment (`arr[i] = SomeStruct { ...fields... }`) over a field write through the index (`arr[i].field = x`) unless Task 1's audit specifically confirms the latter works — whole-element reassignment is the same "rebuild fresh" discipline already proven safe elsewhere on this branch.
- **Sentinel error convention, no `Result`/`Option`.** Every fallible function returns `(value, i64 status)`, reusing `stdlib/asn1/der.sio`'s `DER_OK`/`DER_ERR_*` constants where the failure is genuinely a DER-decode failure (propagate the decoder's own status), and new `X509_ERR_*` constants (defined in `cert.sio`) for X.509-semantic failures (an OID that doesn't match any known algorithm, a field count exceeding a fixed bound, an outer/inner signature-algorithm mismatch).
- **Zero-representation convention**: every `_count`/`_len` field of `0` means "absent/empty", mirroring `BigInt`'s `len == 0` convention. `path_len_constraint == -1` is the sentinel for "not applicable" (a real `pathLenConstraint` of `0` is a valid, distinct CA constraint).
- **No silent truncation on overflow.** Any field whose real DER content exceeds this plan's fixed bounds (`X509_MAX_*` constants) is rejected with a defined `X509_ERR_*` status, never silently truncated or dropped without a caller-visible signal.
- **OID and other DER-published byte constants in `stdlib/x509/oid.sio`** are public, standardized values (from RFC 5280 / PKIX object identifier registrations) — cross-check them against a second independent source before trusting this plan's transcription, per this project's "measure, don't assume" discipline (the same discipline that caught real transcription errors in the BigInt and hash-functions sub-projects' own constant tables).
- **No AI attribution in any commit message** (this repo's CLAUDE.md rule, zero exceptions).
- **Conventional-Commits-style commit messages**: `feat(x509): ...`, `feat(crypto): ...`, `test(x509): ...`, `docs(audit): ...`.
- **Test files:** `tests/run-pass/x509_*.sio`, run via `bash scripts/run_sio_test_suite.sh --filter-prefix x509_`. **Never run the whole-repo suite.**
- **Run every shell command as a plain foreground command, one at a time. Never use any Monitor/background-wait mechanism.**
- **No string literal over ~120 characters** (Finding 17, `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`) — any long byte sequence (a real certificate's DER bytes, a long expected hex digest) must be built via a `let bytes = [0x30, 0x82, ...]` array literal and a `rawbuf_set` loop (the pattern every prior sub-project's tests already use for DER fixtures), never a single long string.
- **No top-level `const` array with more than 16 elements** (Finding 15) — any lookup table larger than 16 entries (there shouldn't be one in this plan, but if a task needs one) must be a function returning the array literal, called once into a local binding.
- **Every narrowing `as u8` cast is masked with `& 255` immediately beforehand** (Finding 14) — applies to any byte-serialization code this plan writes (e.g. `pkcs1.sio`'s local `bigint_to_bytes_be` helper).
- **A single-shot `bigint_to_bytes_be` helper is local to `stdlib/crypto/pkcs1.sio`**, per the spec's explicit ruling — do NOT add it to `stdlib/bignum/bigint.sio`.

## File Structure

```
stdlib/x509/oid.sio        -- Task 1 (second half): known OID byte-constants + comparison
stdlib/x509/cert.sio        -- Task 2: data structures; Tasks 5-7: parser
stdlib/crypto/pkcs1.sio      -- Task 3: RSASSA-PKCS1-v1_5-VERIFY
stdlib/x509/sct.sio          -- Task 4: RFC 6962 SCT list decoder
```

---

### Task 1: Array-of-struct audit + `stdlib/x509/oid.sio`

**Files:**
- Create: `stdlib/x509/oid.sio`
- Test: `tests/run-pass/x509_array_of_struct_audit.sio`, `tests/run-pass/x509_oid_compare.sio`

**Interfaces:**
- Produces: OID byte constants (see Step 3) and `fn oid_eq(a: &[u8; 20], a_len: i32, b: &[u8; 20], b_len: i32) -> bool` (for comparing two already-decoded OIDs) plus `oid_eq3`/`oid_eq8`/`oid_eq9`/`oid_eq10` (for comparing a decoded OID against one of this file's fixed-width constants, matching the constant's own declared width).

- [ ] **Step 1: Write the array-of-struct audit test**

Create `tests/run-pass/x509_array_of_struct_audit.sio`:

```sio
struct FlatEntry {
    a: i32,
    b: [u8; 4],
}

struct Inner {
    x: i32,
}

struct NestedEntry {
    tag: i32,
    inner: Inner,
}

fn main() with IO {
    // Check (a): array of a FLAT multi-field struct (no nested struct field).
    var flat: [FlatEntry; 3] = [
        FlatEntry { a: 0, b: [0, 0, 0, 0] },
        FlatEntry { a: 0, b: [0, 0, 0, 0] },
        FlatEntry { a: 0, b: [0, 0, 0, 0] },
    ]
    flat[0] = FlatEntry { a: 10, b: [1, 2, 3, 4] }
    flat[1] = FlatEntry { a: 20, b: [5, 6, 7, 8] }
    assert(flat[0].a == 10)
    assert(flat[0].b[2] == 3)
    assert(flat[1].a == 20)
    assert(flat[2].a == 0)
    println("check (a) flat array-of-struct: PASSED")

    // Check (b): array of a struct that itself contains ANOTHER struct field.
    var nested: [NestedEntry; 2] = [
        NestedEntry { tag: 0, inner: Inner { x: 0 } },
        NestedEntry { tag: 0, inner: Inner { x: 0 } },
    ]
    nested[0] = NestedEntry { tag: 1, inner: Inner { x: 42 } }
    assert(nested[0].tag == 1)
    assert(nested[0].inner.x == 42)
    assert(nested[1].tag == 0)
    println("check (b) nested struct-in-array: PASSED")

    // Check (c): field write THROUGH an array index (arr[i].field = x), as
    // opposed to whole-element reassignment -- a distinct, narrower pattern
    // this plan's parser code would like to use if available, but has a
    // known-safe fallback (whole-element reassignment, already proven in
    // checks (a)/(b) above) if this specific pattern doesn't work.
    var via_index: [FlatEntry; 1] = [FlatEntry { a: 0, b: [0, 0, 0, 0] }]
    via_index[0].a = 99
    assert(via_index[0].a == 99)
    println("check (c) field write through array index: PASSED")

    // Check (d): whole-element copy-by-index-READ into a local `let` --
    // e.g. `let entry = arr[i]` -- distinct from checks (a)/(c) above,
    // which read/write individual FIELDS through an index, not the whole
    // struct value at once. This exact pattern is used later in this
    // plan (Task 7's extension-list walk: `let entry = cert.extensions[xi
    // as usize]`). Low risk if (a)/(c) both passed, but worth its own
    // direct check rather than assuming it transfers.
    var copy_src: [FlatEntry; 2] = [
        FlatEntry { a: 111, b: [1, 1, 1, 1] },
        FlatEntry { a: 222, b: [2, 2, 2, 2] },
    ]
    let copied = copy_src[1]
    assert(copied.a == 222)
    assert(copied.b[0] == 2)
    println("check (d) whole-element copy-by-index-read: PASSED")

    println("x509_array_of_struct_audit: all checks passed")
}
```

Run: `./bin/souc run tests/run-pass/x509_array_of_struct_audit.sio`

- [ ] **Step 2: Interpret the result and rule**

Run the test. Three independent outcomes are possible for checks (a)/(b)/(c) — each may pass or fail independently:

- **If check (a) FAILS**: STOP immediately. Report BLOCKED. This means arrays of structs do not work AT ALL on this compiler, which invalidates this entire plan's data-structure design (`RdnEntry`, `ExtensionEntry`, `SctEntry` are all flat structs used in arrays) — this is a controller-level plan redesign, not something to improvise a workaround for yourself. Do not proceed to Step 3 or any later task.
- **If check (a) PASSES but check (b) FAILS**: proceed with the plan as designed, EXCEPT `GeneralName.directory_name: X509Name` (Task 2, Step 3) must use the documented fallback: drop the `directory_name: X509Name` field entirely and let `directoryName`-tagged SAN entries fall through to the same raw-bytes handling as `x400Address`/`ediPartyName` (i.e. `GeneralName` becomes a purely flat struct, no nested struct field, consistent with what check (a) proved safe). Record this ruling in your report; the controller will carry it into Task 2's dispatch.
- **If both (a) and (b) PASS**: proceed with the plan exactly as designed, including the nested `X509Name` field in `GeneralName`.
- **Checks (c) and (d)'s results do not gate anything** — they only determine which exact access forms later tasks may use directly. If check (c) failed, use whole-element reassignment instead of `arr[i].field = x` (already proven safe by (a)/(b) passing). If check (d) failed (whole-element copy-by-index-read, `let entry = arr[i]`, does not work even though (a)/(b) do), Task 7's extension-list walk (`let entry = cert.extensions[xi as usize]`) must instead access fields directly off the indexed expression without an intermediate copy (`cert.extensions[xi as usize].oid`, `cert.extensions[xi as usize].oid_len`, etc., matching whatever check (a)/(c) already proved works) — record this in your report so Task 7's dispatch can be precise about which form to use. Record which of (c)/(d) work so later task briefs can be precise.

- [ ] **Step 3: Implement `stdlib/x509/oid.sio`**

This step does not depend on Step 2's outcome (OIDs are plain byte arrays, no struct-in-array risk). Independently cross-check every OID byte sequence below against a second published source (e.g. an OID lookup tool, or a real certificate's `openssl asn1parse` output for a field you can identify) before trusting this plan's transcription — these are the raw content bytes of each OID's DER encoding (i.e. what appears inside an `OBJECT IDENTIFIER` TLV's value, not including the tag/length bytes).

```sio
// stdlib/x509/oid.sio
//
// Known OID byte-sequences (RFC 5280 / PKIX registrations) this project's
// X.509 layer recognizes, plus a length-aware byte comparison helper.
// These are public, standardized values -- cross-checked against a second
// source before being trusted here, per this project's "measure, don't
// assume" discipline (see this task's brief for the cross-check instruction).

// NOTE: every OID is a FUNCTION returning a freshly-built [u8;N] array, NOT
// a top-level `const [u8;N]`. Finding 18 (docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md)
// found that a top-level `const [u8;N]` is silently corrupted whenever its
// WHOLE value is used (copied to a var, or address-taken and passed by
// reference) -- only direct indexed reads of the const are safe, which is
// useless for this file's actual purpose (comparing a decoded OID's bytes
// against a whole known OID value via oid_eq3/8/9/10). Every call site
// (Tasks 5-7) must call the function once into a local `let` before taking
// a reference to the result -- see this task's brief for the exact pattern.

// -- Signature/key algorithms (RFC 8017, arc 1.2.840.113549.1.1) --
pub fn oid_rsa_encryption() -> [u8; 9] {
    var r: [u8; 9] = [0; 9]
    r[0] = 0x2A; r[1] = 0x86; r[2] = 0x48; r[3] = 0x86; r[4] = 0xF7
    r[5] = 0x0D; r[6] = 0x01; r[7] = 0x01; r[8] = 0x01
    r
}
pub fn oid_sha1_with_rsa() -> [u8; 9] {
    var r: [u8; 9] = [0; 9]
    r[0] = 0x2A; r[1] = 0x86; r[2] = 0x48; r[3] = 0x86; r[4] = 0xF7
    r[5] = 0x0D; r[6] = 0x01; r[7] = 0x01; r[8] = 0x05
    r
}
pub fn oid_sha256_with_rsa() -> [u8; 9] {
    var r: [u8; 9] = [0; 9]
    r[0] = 0x2A; r[1] = 0x86; r[2] = 0x48; r[3] = 0x86; r[4] = 0xF7
    r[5] = 0x0D; r[6] = 0x01; r[7] = 0x01; r[8] = 0x0B
    r
}
pub fn oid_sha384_with_rsa() -> [u8; 9] {
    var r: [u8; 9] = [0; 9]
    r[0] = 0x2A; r[1] = 0x86; r[2] = 0x48; r[3] = 0x86; r[4] = 0xF7
    r[5] = 0x0D; r[6] = 0x01; r[7] = 0x01; r[8] = 0x0C
    r
}
pub fn oid_sha512_with_rsa() -> [u8; 9] {
    var r: [u8; 9] = [0; 9]
    r[0] = 0x2A; r[1] = 0x86; r[2] = 0x48; r[3] = 0x86; r[4] = 0xF7
    r[5] = 0x0D; r[6] = 0x01; r[7] = 0x01; r[8] = 0x0D
    r
}

// -- RDN attribute types (arc 2.5.4) + PKCS#9 emailAddress --
pub fn oid_common_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x04; r[2] = 0x03
    r
}
pub fn oid_country_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x04; r[2] = 0x06
    r
}
pub fn oid_locality_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x04; r[2] = 0x07
    r
}
pub fn oid_state_or_province_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x04; r[2] = 0x08
    r
}
pub fn oid_organization_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x04; r[2] = 0x0A
    r
}
pub fn oid_organizational_unit_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x04; r[2] = 0x0B
    r
}
pub fn oid_email_address() -> [u8; 9] {
    var r: [u8; 9] = [0; 9]
    r[0] = 0x2A; r[1] = 0x86; r[2] = 0x48; r[3] = 0x86; r[4] = 0xF7
    r[5] = 0x0D; r[6] = 0x01; r[7] = 0x09; r[8] = 0x01
    r
}

// -- Standard v3 extensions (arc 2.5.29) --
pub fn oid_ext_subject_directory_attributes() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x09
    r
}
pub fn oid_ext_subject_key_identifier() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x0E
    r
}
pub fn oid_ext_key_usage() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x0F
    r
}
pub fn oid_ext_subject_alt_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x11
    r
}
pub fn oid_ext_issuer_alt_name() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x12
    r
}
pub fn oid_ext_basic_constraints() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x13
    r
}
pub fn oid_ext_name_constraints() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x1E
    r
}
pub fn oid_ext_crl_distribution_points() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x1F
    r
}
pub fn oid_ext_certificate_policies() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x20
    r
}
pub fn oid_ext_authority_key_identifier() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x23
    r
}
pub fn oid_ext_policy_constraints() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x24
    r
}
pub fn oid_ext_ext_key_usage() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x25
    r
}
pub fn oid_ext_freshest_crl() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x2E
    r
}
pub fn oid_ext_inhibit_any_policy() -> [u8; 3] {
    var r: [u8; 3] = [0; 3]
    r[0] = 0x55; r[1] = 0x1D; r[2] = 0x36
    r
}

// -- PKIX private extension (arc 1.3.6.1.5.5.7.1) --
pub fn oid_ext_authority_info_access() -> [u8; 8] {
    var r: [u8; 8] = [0; 8]
    r[0] = 0x2B; r[1] = 0x06; r[2] = 0x01; r[3] = 0x05
    r[4] = 0x05; r[5] = 0x07; r[6] = 0x01; r[7] = 0x01
    r
}

// -- RFC 6962 Certificate Transparency SCT extension --
pub fn oid_ext_sct_list() -> [u8; 10] {
    var r: [u8; 10] = [0; 10]
    r[0] = 0x2B; r[1] = 0x06; r[2] = 0x01; r[3] = 0x04; r[4] = 0x01
    r[5] = 0xD6; r[6] = 0x79; r[7] = 0x02; r[8] = 0x04; r[9] = 0x02
    r
}

// Compares the first a_len bytes of a against the first b_len bytes of b --
// equal only if both lengths AND all compared bytes match. Used to match a
// decoded OID's raw bytes against one of the constants above.
pub fn oid_eq(a: &[u8; 20], a_len: i32, b: &[u8; 20], b_len: i32) -> bool {
    if a_len != b_len {
        return false
    }
    var i: i32 = 0
    while i < a_len {
        if a[i as usize] != b[i as usize] {
            return false
        }
        i = i + 1
    }
    true
}
```

`oid_eq` alone only compares two already-`[u8;20]`-shaped buffers — every OID constant above is a differently-sized `[u8;N]` (`N` = 3, 8, 9, or 10), which cannot be passed directly where `oid_eq` expects `&[u8;20]` (Sounio array types are size-distinct: `[u8;3]` and `[u8;20]` are different types). Add one small width-specific comparison function per width this file's constants actually use, appended to the same code block above:

```sio
pub fn oid_eq3(a: &[u8; 20], a_len: i32, b: &[u8; 3]) -> bool {
    if a_len != 3 {
        return false
    }
    var i: i32 = 0
    while i < 3 {
        if a[i as usize] != b[i as usize] {
            return false
        }
        i = i + 1
    }
    true
}

pub fn oid_eq8(a: &[u8; 20], a_len: i32, b: &[u8; 8]) -> bool {
    if a_len != 8 {
        return false
    }
    var i: i32 = 0
    while i < 8 {
        if a[i as usize] != b[i as usize] {
            return false
        }
        i = i + 1
    }
    true
}

pub fn oid_eq9(a: &[u8; 20], a_len: i32, b: &[u8; 9]) -> bool {
    if a_len != 9 {
        return false
    }
    var i: i32 = 0
    while i < 9 {
        if a[i as usize] != b[i as usize] {
            return false
        }
        i = i + 1
    }
    true
}

pub fn oid_eq10(a: &[u8; 20], a_len: i32, b: &[u8; 10]) -> bool {
    if a_len != 10 {
        return false
    }
    var i: i32 = 0
    while i < 10 {
        if a[i as usize] != b[i as usize] {
            return false
        }
        i = i + 1
    }
    true
}
```

`oid_eq` (the original, `[u8;20]`-vs-`[u8;20]`) stays too — it's the right tool for comparing two ALREADY-DECODED OIDs against each other (e.g. Task 7's outer-vs-inner signature-algorithm cross-check, `cert.outer_sig_alg_oid` vs `cert.tbs_sig_alg_oid`, both `[u8;20]` fields), which none of the width-specific functions above handle.

- [ ] **Step 4: Write and run the OID comparison test**

Create `tests/run-pass/x509_oid_compare.sio`:

```sio
use x509::oid::*

fn main() with IO {
    var a: [u8; 20] = [0; 20]
    a[0] = 0x55
    a[1] = 0x04
    a[2] = 0x03
    var b: [u8; 20] = [0; 20]
    b[0] = 0x55
    b[1] = 0x04
    b[2] = 0x03
    assert(oid_eq(&a, 3, &b, 3) == true)

    var c: [u8; 20] = [0; 20]
    c[0] = 0x55
    c[1] = 0x04
    c[2] = 0x06   // countryName, not commonName
    assert(oid_eq(&a, 3, &c, 3) == false)

    assert(oid_eq(&a, 3, &b, 2) == false)   // length mismatch

    println("x509_oid_compare: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/x509_oid_compare.sio`
Expected: prints `x509_oid_compare: all cases passed`, exits 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/x509/oid.sio tests/run-pass/x509_array_of_struct_audit.sio tests/run-pass/x509_oid_compare.sio
git commit -m "feat(x509): add OID constants, audit array-of-struct support on Madaros"
```

If Step 2 found check (a) failing (the BLOCKED case), do not commit `stdlib/x509/oid.sio` as part of a "plan continues" story — report BLOCKED as instructed instead, and let the controller decide whether `oid.sio` alone (which has no array-of-struct dependency) is still worth committing on its own.

---

### Task 2: `stdlib/x509/cert.sio` data structures

**Files:**
- Create: `stdlib/x509/cert.sio`
- Test: `tests/run-pass/x509_struct_construction.sio`

**Interfaces:**
- Consumes: `stdlib/asn1/der.sio` (`DerReader`, `DerTag`, `DER_*` constants — already `pub`), `stdlib/bignum/bigint.sio` (`BigInt`, `bigint_zero`), `stdlib/x509/oid.sio` (Task 1).
- Produces: every struct type and `X509_*` constant from the spec's Data Structures section, plus `fn x509_name_zero() -> X509Name`, `fn extension_entry_zero() -> ExtensionEntry`, `fn general_name_zero() -> GeneralName`, `fn certificate_zero() -> Certificate` (zero-value constructors, mirroring `bigint_zero()`'s convention — every array-of-struct field initialized via an explicit literal list of zeroed elements, per Task 1's proven pattern).

**Before starting:** read Task 1's completion report for the array-of-struct audit's exact outcome (both check (a)/(b) results). This task's `GeneralName` struct definition depends on it:
- If check (b) passed: define `GeneralName` WITH the nested `directory_name: X509Name` field, exactly as the spec shows.
- If check (b) failed (but (a) passed): define `GeneralName` WITHOUT that field — `directoryName`-tagged entries use the same `value: [u8; 253]` raw-bytes field every other non-string variant uses. Document this in a comment at the struct definition citing Task 1's finding.

- [ ] **Step 1: Write the failing construction test**

Create `tests/run-pass/x509_struct_construction.sio` (adjust based on Task 1's `GeneralName` ruling — this version assumes check (b) PASSED; if it failed, drop the `directory_name` assertion and the `Inner`-equivalent nested-field check):

```sio
use x509::cert::*
use bignum::bigint::*

fn main() with IO {
    let name = x509_name_zero()
    assert(name.count == 0)

    let ext = extension_entry_zero()
    assert(ext.oid_len == 0)
    assert(ext.critical == false)

    let gn = general_name_zero()
    assert(gn.tag == 0)
    assert(gn.value_len == 0)

    let cert = certificate_zero()
    assert(cert.tbs_len == 0)
    assert(cert.version == 0)
    assert(cert.extension_count == 0)
    assert(cert.san_count == 0)
    assert(cert.sct_count == 0)
    assert(cert.path_len_constraint == -1)
    assert(bigint_cmp(&cert.serial_number, &bigint_zero()) == 0)

    // Array-of-struct field access on a freshly-zeroed Certificate: read an
    // element's field two levels deep (Certificate -> extensions[i] -> oid_len).
    assert(cert.extensions[0].oid_len == 0)
    assert(cert.san_entries[0].tag == 0)
    assert(cert.sct_entries[0].version == 0)

    println("x509_struct_construction: all cases passed")
}
```

Run: `./bin/souc run tests/run-pass/x509_struct_construction.sio`
Expected: FAIL to compile — `stdlib/x509/cert.sio` doesn't exist yet.

- [ ] **Step 2: Implement `stdlib/x509/cert.sio`'s data structures**

```sio
// stdlib/x509/cert.sio
//
// X.509 Certificate data structures (RFC 5280) and zero-value constructors.
// The parser functions (der -> Certificate) are added in later tasks; this
// file starts with just the data model, mirroring how stdlib/bignum/bigint.sio
// and stdlib/asn1/der.sio each established their own types before their own
// operations were built on top.

use asn1::der::*
use bignum::bigint::*
use net::socket::*

pub const X509_MAX_RDN_ENTRIES: i32 = 16
pub const X509_MAX_NAME_VALUE_LEN: i64 = 128
pub const X509_MAX_OID_BYTES: i64 = 20
pub const X509_MAX_SAN_ENTRIES: i32 = 32
pub const X509_MAX_GENERAL_NAME_VALUE_LEN: i64 = 253
pub const X509_MAX_EXTENSIONS: i32 = 32
pub const X509_MAX_EXTENSION_VALUE_LEN: i64 = 512
pub const X509_MAX_SCT_ENTRIES: i32 = 8
pub const X509_MAX_SIGNATURE_LEN: i64 = 512

pub const X509_OK: i64 = 0
pub const X509_ERR_UNSUPPORTED_ALGORITHM: i64 = 0 - 1
pub const X509_ERR_TOO_MANY_ENTRIES: i64 = 0 - 2
pub const X509_ERR_VALUE_TOO_LONG: i64 = 0 - 3
pub const X509_ERR_SIG_ALG_MISMATCH: i64 = 0 - 4
pub const X509_ERR_MALFORMED: i64 = 0 - 5

pub struct RdnEntry {
    oid: [u8; 20],
    oid_len: i32,
    value_tag: i32,
    value: [u8; 128],
    value_len: i32,
}

pub struct X509Name {
    entries: [RdnEntry; 16],
    count: i32,
}

pub const GENERAL_NAME_OTHER_NAME: i32 = 0
pub const GENERAL_NAME_RFC822_NAME: i32 = 1
pub const GENERAL_NAME_DNS_NAME: i32 = 2
pub const GENERAL_NAME_X400_ADDRESS: i32 = 3
pub const GENERAL_NAME_DIRECTORY_NAME: i32 = 4
pub const GENERAL_NAME_EDI_PARTY_NAME: i32 = 5
pub const GENERAL_NAME_URI: i32 = 6
pub const GENERAL_NAME_IP_ADDRESS: i32 = 7
pub const GENERAL_NAME_REGISTERED_ID: i32 = 8

// NOTE: this struct's shape depends on Task 1's array-of-nested-struct audit
// outcome. As written here it assumes check (b) PASSED (nested struct-in-array
// works) -- if Task 1 found check (b) failing, remove the `directory_name`
// field entirely; directoryName-tagged entries then use `value`/`value_len`
// like x400Address/ediPartyName do, with no dedicated decode.
pub struct GeneralName {
    tag: i32,
    value: [u8; 253],
    value_len: i32,
    oid: [u8; 20],
    oid_len: i32,
    directory_name: X509Name,
}

pub struct ExtensionEntry {
    oid: [u8; 20],
    oid_len: i32,
    critical: bool,
    value: [u8; 512],
    value_len: i32,
}

pub struct Certificate {
    tbs_start: i64,
    tbs_len: i64,
    outer_sig_alg_oid: [u8; 20],
    outer_sig_alg_oid_len: i32,
    outer_signature: [u8; 512],
    outer_signature_len: i32,

    version: i32,
    serial_number: BigInt,
    tbs_sig_alg_oid: [u8; 20],
    tbs_sig_alg_oid_len: i32,
    issuer: X509Name,
    not_before_unix: i64,
    not_after_unix: i64,
    subject: X509Name,
    modulus: BigInt,
    public_exponent: BigInt,

    extensions: [ExtensionEntry; 32],
    extension_count: i32,
    is_ca: bool,
    path_len_constraint: i32,
    key_usage_bits: i32,
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
}

pub fn x509_name_zero() -> X509Name {
    X509Name {
        entries: [
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
            RdnEntry { oid: [0; 20], oid_len: 0, value_tag: 0, value: [0; 128], value_len: 0 },
        ],
        count: 0,
    }
}

pub fn extension_entry_zero() -> ExtensionEntry {
    ExtensionEntry { oid: [0; 20], oid_len: 0, critical: false, value: [0; 512], value_len: 0 }
}

pub fn general_name_zero() -> GeneralName {
    GeneralName {
        tag: 0,
        value: [0; 253],
        value_len: 0,
        oid: [0; 20],
        oid_len: 0,
        directory_name: x509_name_zero(),
    }
}

pub fn sct_entry_zero() -> SctEntry {
    SctEntry {
        version: 0,
        log_id: [0; 32],
        timestamp_ms: 0,
        extensions_len: 0,
        sig_algorithm_hash: 0,
        sig_algorithm_sign: 0,
        signature: [0; 128],
        signature_len: 0,
    }
}

pub fn certificate_zero() -> Certificate {
    var extensions_zero: [ExtensionEntry; 32] = [
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
    ]
    var san_zero: [GeneralName; 32] = [
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
    ]
    var ian_zero: [GeneralName; 32] = [
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
    ]
    var scts_zero: [SctEntry; 8] = [
        sct_entry_zero(), sct_entry_zero(), sct_entry_zero(), sct_entry_zero(),
        sct_entry_zero(), sct_entry_zero(), sct_entry_zero(), sct_entry_zero(),
    ]

    Certificate {
        tbs_start: 0,
        tbs_len: 0,
        outer_sig_alg_oid: [0; 20],
        outer_sig_alg_oid_len: 0,
        outer_signature: [0; 512],
        outer_signature_len: 0,
        version: 0,
        serial_number: bigint_zero(),
        tbs_sig_alg_oid: [0; 20],
        tbs_sig_alg_oid_len: 0,
        issuer: x509_name_zero(),
        not_before_unix: 0,
        not_after_unix: 0,
        subject: x509_name_zero(),
        modulus: bigint_zero(),
        public_exponent: bigint_zero(),
        extensions: extensions_zero,
        extension_count: 0,
        is_ca: false,
        path_len_constraint: 0 - 1,
        key_usage_bits: 0,
        san_entries: san_zero,
        san_count: 0,
        issuer_alt_names: ian_zero,
        issuer_alt_name_count: 0,
        subject_key_id: [0; 32],
        subject_key_id_len: 0,
        authority_key_id: [0; 32],
        authority_key_id_len: 0,
        sct_entries: scts_zero,
        sct_count: 0,
    }
}
```

Note: `SctEntry` is referenced above (`Certificate.sct_entries`, `sct_entry_zero()`) but its own struct definition lives in Task 4's `stdlib/x509/sct.sio`, per the spec's file layout — this creates a circular-looking dependency (`cert.sio` needs `SctEntry` from `sct.sio`, but `sct.sio` doesn't need anything from `cert.sio`). Resolve this by defining `SctEntry` and `sct_entry_zero()` directly in THIS file (`cert.sio`) instead of `sct.sio`, since `Certificate` needs it and `cert.sio` has no reason to depend on `sct.sio` for just a type definition — Task 4's `sct.sio` will `use x509::cert::*` to get `SctEntry`, and provide only the RFC-6962-specific *decoding function* (`sct_list_decode`), not the type itself. Add this to the code block above (insert before `certificate_zero`, after `ExtensionEntry`):

```sio
pub struct SctEntry {
    version: i32,
    log_id: [u8; 32],
    timestamp_ms: i64,
    extensions_len: i32,
    sig_algorithm_hash: i32,
    sig_algorithm_sign: i32,
    signature: [u8; 128],
    signature_len: i32,
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/x509_struct_construction.sio`
Expected: prints `x509_struct_construction: all cases passed`, exits 0.

- [ ] **Step 4: Commit**

```bash
git add stdlib/x509/cert.sio tests/run-pass/x509_struct_construction.sio
git commit -m "feat(x509): add Certificate/X509Name/GeneralName/ExtensionEntry/SctEntry data structures"
```

---

### Task 3: `stdlib/crypto/pkcs1.sio` — RSASSA-PKCS1-v1_5-VERIFY

**Files:**
- Create: `stdlib/crypto/pkcs1.sio`
- Test: `tests/run-pass/pkcs1_verify.sio`

**Interfaces:**
- Consumes: `stdlib/bignum/bigint.sio` (`BigInt`, `bigint_modpow`, `bigint_from_u32`, `bigint_cmp`, `bigint_zero`).
- Produces: `pub const PKCS1_HASH_SHA1/SHA256/SHA384/SHA512: i32`; `pub fn pkcs1_v15_verify(signature: &BigInt, modulus: &BigInt, exponent: &BigInt, modulus_byte_len: i64, hash_algorithm: i32, message_hash: &[u8; 64], message_hash_len: i64) -> bool`.

- [ ] **Step 1: Derive and independently re-verify a real RSA-2048 PKCS#1 v1.5 SHA-256 signature test vector**

This task needs a genuine, independently-computable signature test vector: a real RSA key pair, a real message, its SHA-256 hash, and the PKCS#1 v1.5 signature of that hash — generated with an external tool (e.g. `openssl genrsa`, `openssl dgst -sha256 -sign`) during implementation, NOT invented by this plan. Concretely:

1. Generate a small (1024 or 2048-bit) RSA key pair with `openssl genrsa`.
2. Extract the modulus and public exponent with `openssl rsa -text` (or `-modulus`).
3. Sign a short, fixed message (e.g. the literal bytes `"pkcs1 test vector"`) with `openssl dgst -sha256 -sign <private key> <message file>`, producing a raw PKCS#1 v1.5 signature.
4. Verify independently with `openssl dgst -sha256 -verify <public key> -signature <sig file> <message file>` (must print `Verified OK`) before trusting the vector.
5. Convert the modulus, exponent, and signature to the decimal or hex `BigInt`-literal form this test needs (e.g. via `python3 -c "print(int.from_bytes(open('sig','rb').read(),'big'))"` for the signature, similarly for the modulus from `openssl rsa -text -noout | grep -A100 modulus`).

Record the exact modulus (hex), exponent (almost certainly `65537`), message bytes, SHA-256 hash of the message (independently computed, e.g. `sha256sum`), and signature (hex) in the report — this is the test vector every future PKCS#1 test on this branch can reuse.

- [ ] **Step 2: Write the failing test**

Create `tests/run-pass/pkcs1_verify.sio` using the Step 1 vector — construct `signature`/`modulus` as `BigInt` values (via repeated `bigint_from_u32`/`bigint_mul`/`bigint_add` byte-by-byte composition, exactly like `stdlib/asn1/der.sio`'s `der_read_integer` does, or via direct `.limbs[]`/`.len` field assignment for a large literal, matching `tests/run-pass/bignum_wide_scale.sio`'s established pattern for constructing large `BigInt` values from known limb data), populate a `[u8; 64]` with the SHA-256 hash bytes (via `stdlib/hash::sha256`, hashing the message bytes from a `RawBuf`), and call `pkcs1_v15_verify`, asserting `true`. Also assert `false` for at least one deliberately-corrupted case (flip one byte of the signature, or use the wrong hash) — a verify function that never returns `false` in any test is unproven.

Run: `./bin/souc run tests/run-pass/pkcs1_verify.sio`
Expected: FAIL to compile — `stdlib/crypto/pkcs1.sio` doesn't exist yet.

- [ ] **Step 3: Implement `stdlib/crypto/pkcs1.sio`**

```sio
// stdlib/crypto/pkcs1.sio
//
// RSASSA-PKCS1-v1_5-VERIFY (RFC 8017 SS8.2.2). Public-key verification only
// -- this project never signs or decrypts with an RSA private key, so no
// constant-time requirement applies (see docs/superpowers/specs/2026-08-23-
// madaros-hash-functions-design.md's Context section for the reasoning,
// which carries forward unchanged here).

use bignum::bigint::*

pub const PKCS1_HASH_SHA1: i32 = 1
pub const PKCS1_HASH_SHA256: i32 = 2
pub const PKCS1_HASH_SHA384: i32 = 3
pub const PKCS1_HASH_SHA512: i32 = 4

// RFC 8017 Appendix, Table (DigestInfo DER prefixes -- the ASN.1 encoding of
// "SEQUENCE { AlgorithmIdentifier, OCTET STRING }" up to but not including
// the hash bytes themselves, for each hash algorithm this project supports).
fn digest_info_prefix_sha1() -> [u8; 15] {
    [0x30, 0x21, 0x30, 0x09, 0x06, 0x05, 0x2b, 0x0e, 0x03, 0x02, 0x1a, 0x05, 0x00, 0x04, 0x14]
}
fn digest_info_prefix_sha256() -> [u8; 19] {
    [0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20]
}
fn digest_info_prefix_sha384() -> [u8; 19] {
    [0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30]
}
fn digest_info_prefix_sha512() -> [u8; 19] {
    [0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40]
}

// Independently cross-check these three (prefix bytes, prefix length, hash
// length) against RFC 8017 before trusting this transcription -- a wrong
// value here would make every signature verification silently fail (safe
// direction: false negative, never a false positive) or, worse, silently
// succeed on a wrong prefix if two algorithms' prefixes were transposed.
fn digest_info_len_and_hash_len(hash_algorithm: i32) -> (i64, i64) {
    if hash_algorithm == PKCS1_HASH_SHA1 {
        (15, 20)
    } else if hash_algorithm == PKCS1_HASH_SHA256 {
        (19, 32)
    } else if hash_algorithm == PKCS1_HASH_SHA384 {
        (19, 48)
    } else {
        (19, 64)   // PKCS1_HASH_SHA512
    }
}

fn digest_info_prefix_byte(hash_algorithm: i32, i: i64) -> i64 {
    if hash_algorithm == PKCS1_HASH_SHA1 {
        let p = digest_info_prefix_sha1()
        (p[i as usize] & 255)
    } else if hash_algorithm == PKCS1_HASH_SHA256 {
        let p = digest_info_prefix_sha256()
        (p[i as usize] & 255)
    } else if hash_algorithm == PKCS1_HASH_SHA384 {
        let p = digest_info_prefix_sha384()
        (p[i as usize] & 255)
    } else {
        let p = digest_info_prefix_sha512()
        (p[i as usize] & 255)
    }
}

// Converts a BigInt to a big-endian byte string of exactly `out_len` bytes,
// left-padded with zero bytes if the value's own magnitude is shorter.
// Local to this file per the spec's explicit ruling (not added to
// stdlib/bignum/bigint.sio). Returns false if the value is too WIDE to fit
// in out_len bytes (a real error: a signature value must never exceed the
// modulus's own byte length).
fn bigint_to_bytes_be(v: &BigInt, out: &RawBuf, out_len: i64) -> bool with IO {
    // Repeated divmod-by-256 would need bigint division by a non-power-of-2
    // limb count, which this project's BigInt module doesn't provide
    // generically -- instead, walk the limbs directly: v.limbs is little-
    // endian 16-bit, so limb i covers output bytes [out_len-1-2*i-1, out_len-1-2*i].
    var i: i64 = 0
    while i < out_len {
        out_set_zero_byte(out, i)
        i = i + 1
    }
    let total_bytes = (v.len as i64) * 2
    if total_bytes > out_len {
        return false   // value too wide for the requested output width
    }
    var limb_idx: i64 = 0
    while limb_idx < (v.len as i64) {
        let limb = v.limbs[limb_idx as usize]
        let lo = (limb as i64) & 255
        let hi = ((limb as i64) >> 8) & 255
        let byte_pos_lo = out_len - 1 - (limb_idx * 2)
        let byte_pos_hi = out_len - 1 - (limb_idx * 2) - 1
        rawbuf_set(out, byte_pos_lo, lo)
        if byte_pos_hi >= 0 {
            rawbuf_set(out, byte_pos_hi, hi)
        }
        limb_idx = limb_idx + 1
    }
    true
}

fn out_set_zero_byte(out: &RawBuf, i: i64) with IO {
    rawbuf_set(out, i, 0)
}

pub fn pkcs1_v15_verify(
    signature: &BigInt,
    modulus: &BigInt,
    exponent: &BigInt,
    modulus_byte_len: i64,
    hash_algorithm: i32,
    message_hash: &[u8; 64],
    message_hash_len: i64,
) -> bool with IO {
    if bigint_cmp(signature, modulus) >= 0 {
        return false   // a valid signature is always < modulus
    }

    let em = bigint_modpow(signature, exponent, modulus)

    let em_buf = rawbuf_new(modulus_byte_len)
    let converted = bigint_to_bytes_be(&em, &em_buf, modulus_byte_len)
    if converted == false {
        return false
    }

    let (prefix_len, expected_hash_len) = digest_info_len_and_hash_len(hash_algorithm)
    if message_hash_len != expected_hash_len {
        return false
    }

    // Expected EM = 0x00 0x01 <PS: 0xFF...> 0x00 <DigestInfo prefix> <hash>
    let ps_len = modulus_byte_len - 3 - prefix_len - expected_hash_len
    if ps_len < 8 {
        return false   // RFC 8017 requires at least 8 bytes of 0xFF padding
    }

    if (rawbuf_get(&em_buf, 0) & 255) != 0 {
        return false
    }
    if (rawbuf_get(&em_buf, 1) & 255) != 1 {
        return false
    }
    var i: i64 = 0
    while i < ps_len {
        if (rawbuf_get(&em_buf, 2 + i) & 255) != 255 {
            return false
        }
        i = i + 1
    }
    if (rawbuf_get(&em_buf, 2 + ps_len) & 255) != 0 {
        return false
    }
    var j: i64 = 0
    while j < prefix_len {
        if (rawbuf_get(&em_buf, 3 + ps_len + j) & 255) != digest_info_prefix_byte(hash_algorithm, j) {
            return false
        }
        j = j + 1
    }
    var k: i64 = 0
    while k < expected_hash_len {
        let em_byte = rawbuf_get(&em_buf, 3 + ps_len + prefix_len + k) & 255
        let hash_byte = (message_hash[k as usize] as i64) & 255
        if em_byte != hash_byte {
            return false
        }
        k = k + 1
    }

    true
}
```

Note: `use net::socket::*` (for `RawBuf`/`rawbuf_new`/`rawbuf_get`/`rawbuf_set`) is required at the top of this file alongside `use bignum::bigint::*` — add it.

- [ ] **Step 4: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/pkcs1_verify.sio`
Expected: prints a success message, exits 0. Both the true-case (real signature) and false-case (corrupted signature/hash) assertions must pass.

- [ ] **Step 5: Commit**

```bash
git add stdlib/crypto/pkcs1.sio tests/run-pass/pkcs1_verify.sio
git commit -m "feat(crypto): add PKCS#1 v1.5 signature verification"
```

---

### Task 4: `stdlib/x509/sct.sio` — RFC 6962 SCT list decoder

**Files:**
- Create: `stdlib/x509/sct.sio`
- Test: `tests/run-pass/x509_sct_decode.sio`

**Interfaces:**
- Consumes: `stdlib/x509/cert.sio`'s `SctEntry`/`sct_entry_zero` (Task 2), `stdlib/net/socket.sio`'s `RawBuf`/`rawbuf_get`.
- Produces: `pub fn sct_list_decode(buf: &RawBuf, start: i64, len: i64) -> ([SctEntry; 8], i32, i64)` (entries, count, status — reusing `X509_OK`/`X509_ERR_*` from `cert.sio`).

- [ ] **Step 1: Write the failing test with a hand-built synthetic SCT list**

This format is NOT ASN.1 — it's a custom length-prefixed binary wire format (RFC 6962 §3.2-3.3), independent of everything `stdlib/asn1/der.sio` does. Build a minimal, one-entry SCT list by hand: `total_len` (2 bytes) + one SCT of `1 (version=0) + 32 (log_id, arbitrary bytes) + 8 (timestamp_ms, big-endian) + 2 (extensions_len=0) + 1 (hash alg) + 1 (sig alg) + 2 (sig_len) + sig_len (signature bytes)`.

Create `tests/run-pass/x509_sct_decode.sio` with a hand-constructed buffer (use a small, e.g. 4-byte, synthetic signature to keep the fixture short), asserting the decoded `SctEntry`'s every field matches what was encoded (version, first/last byte of log_id, timestamp_ms, hash/sig algorithm bytes, signature bytes, and `count == 1`).

Run: `./bin/souc run tests/run-pass/x509_sct_decode.sio`
Expected: FAIL to compile — `stdlib/x509/sct.sio` doesn't exist yet.

- [ ] **Step 2: Implement `stdlib/x509/sct.sio`**

```sio
// stdlib/x509/sct.sio
//
// RFC 6962 SS3.2-3.3 SignedCertificateTimestampList decoder. This wire
// format is length-prefixed binary, NOT ASN.1/DER -- independent of
// stdlib/asn1/der.sio, which only ever sees this structure as an opaque
// OCTET STRING (the SCT extension's extnValue, itself wrapping another
// OCTET STRING containing the bytes this file actually parses).

use x509::cert::*
use net::socket::*

fn read_u8(buf: &RawBuf, pos: i64) -> i64 with IO {
    rawbuf_get(buf, pos) & 255
}

fn read_u16(buf: &RawBuf, pos: i64) -> i64 with IO {
    let hi = read_u8(buf, pos)
    let lo = read_u8(buf, pos + 1)
    (hi << 8) | lo
}

fn read_u64(buf: &RawBuf, pos: i64) -> i64 with IO {
    var v: i64 = 0
    var i: i64 = 0
    while i < 8 {
        v = (v << 8) | read_u8(buf, pos + i)
        i = i + 1
    }
    v
}

pub fn sct_list_decode(buf: &RawBuf, start: i64, len: i64) -> ([SctEntry; 8], i32, i64) with IO {
    var out: [SctEntry; 8] = [
        sct_entry_zero(), sct_entry_zero(), sct_entry_zero(), sct_entry_zero(),
        sct_entry_zero(), sct_entry_zero(), sct_entry_zero(), sct_entry_zero(),
    ]
    var count: i32 = 0

    if len < 2 {
        return (out, 0, X509_ERR_MALFORMED)
    }
    let total_len = read_u16(buf, start)
    if total_len != len - 2 {
        return (out, 0, X509_ERR_MALFORMED)
    }

    var pos: i64 = start + 2
    let end: i64 = start + len
    while pos < end {
        if count >= X509_MAX_SCT_ENTRIES {
            return (out, count, X509_ERR_TOO_MANY_ENTRIES)
        }
        if pos + 2 > end {
            return (out, count, X509_ERR_MALFORMED)
        }
        let sct_len = read_u16(buf, pos)
        pos = pos + 2
        if pos + sct_len > end {
            return (out, count, X509_ERR_MALFORMED)
        }
        let sct_start = pos

        if sct_len < 47 {
            // 1 (version) + 32 (log_id) + 8 (timestamp) + 2 (ext_len) + 1
            // (hash alg) + 1 (sig alg) + 2 (sig_len) = 47 bytes minimum
            // structure BEFORE any extension/signature bytes.
            return (out, count, X509_ERR_MALFORMED)
        }

        let version = read_u8(buf, sct_start)
        var log_id: [u8; 32] = [0; 32]
        var li: i64 = 0
        while li < 32 {
            log_id[li as usize] = (read_u8(buf, sct_start + 1 + li) as u8)
            li = li + 1
        }
        let timestamp_ms = read_u64(buf, sct_start + 33)
        let ext_len = read_u16(buf, sct_start + 41)
        let after_ext = sct_start + 43 + ext_len
        let hash_alg = read_u8(buf, after_ext)
        let sig_alg = read_u8(buf, after_ext + 1)
        let sig_len = read_u16(buf, after_ext + 2)
        var signature: [u8; 128] = [0; 128]
        if sig_len > 128 {
            return (out, count, X509_ERR_VALUE_TOO_LONG)
        }
        var si: i64 = 0
        while si < sig_len {
            signature[si as usize] = (read_u8(buf, after_ext + 4 + si) as u8)
            si = si + 1
        }

        out[count as usize] = SctEntry {
            version: version as i32,
            log_id: log_id,
            timestamp_ms: timestamp_ms,
            extensions_len: ext_len as i32,
            sig_algorithm_hash: hash_alg as i32,
            sig_algorithm_sign: sig_alg as i32,
            signature: signature,
            signature_len: sig_len as i32,
        }
        count = count + 1
        pos = sct_start + sct_len
    }

    (out, count, X509_OK)
}
```

- [ ] **Step 3: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/x509_sct_decode.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 4: Commit**

```bash
git add stdlib/x509/sct.sio tests/run-pass/x509_sct_decode.sio
git commit -m "feat(x509): add RFC 6962 SCT list decoder"
```

---

### Task 5: TBSCertificate core field parsing

**Files:**
- Modify: `stdlib/x509/cert.sio`
- Test: `tests/run-pass/x509_parse_tbs_core.sio`

**Interfaces:**
- Consumes: `stdlib/asn1/der.sio` (`der_reader_new`, `der_read_tlv`, `der_enter`, `der_skip`, `der_next`, `der_read_integer`, `der_read_boolean`, `DER_TAG_*`), `stdlib/x509/oid.sio`.
- Produces: `pub fn x509_parse_tbs_core(buf: &RawBuf, tbs_reader: &DerReader) -> (Certificate, DerReader, i64)` — decodes version, serialNumber, signature algorithm, issuer, validity, subject, subjectPublicKeyInfo into a `Certificate` (leaving extension-related fields at their zero values; Task 7 fills those in and assembles the full outer `Certificate`), returning alongside it the `DerReader` positioned immediately after `subjectPublicKeyInfo` (Task 7 needs this to locate the optional `[3]` extensions field without re-walking the same fields a second time).

- [ ] **Step 1: Independently derive a minimal hand-built TBSCertificate fixture**

Build, by hand, a minimal but structurally complete `TBSCertificate` SEQUENCE (RFC 5280 §4.1): an optional `[0] { INTEGER version }` (use version 2, meaning v3, since v3 is needed later for extensions even though this task doesn't parse them yet), `serialNumber INTEGER` (a small value, e.g. 12345), `signature AlgorithmIdentifier` (SEQUENCE containing `OID_SHA256_WITH_RSA` + a NULL parameters field), `issuer Name` (a SEQUENCE of one RDN SET containing one AttributeTypeAndValue: `OID_COMMON_NAME` + a PrintableString "Test CA"), `validity SEQUENCE { UTCTime notBefore, UTCTime notAfter }` (use two simple, independently-decodable UTCTime values, e.g. `250101000000Z` and `260101000000Z` in `YYMMDDHHMMSSZ` form), `subject Name` (same shape as issuer, different CN), `subjectPublicKeyInfo SEQUENCE { AlgorithmIdentifier (rsaEncryption + NULL), BIT STRING (wrapping a small RSAPublicKey SEQUENCE { INTEGER modulus, INTEGER publicExponent }) }`. This is a substantial hand-built fixture — build it byte-by-byte the same way `tests/run-pass/asn1_rsa_pubkey_integration.sio` built its RSAPublicKey fixture, nested one level deeper.

Independently verify: the outer SEQUENCE's declared length equals the sum of its contents' TLV lengths (the same manual arithmetic check every prior sub-project's fixtures have used).

- [ ] **Step 2: Write the failing test**

Create `tests/run-pass/x509_parse_tbs_core.sio` using the Step 1 fixture, asserting every field `x509_parse_tbs_core` returns: `version == 2`, `serial_number` compares equal to `bigint_from_u32(12345)`, `tbs_sig_alg_oid` matches `oid_sha256_with_rsa()` via `oid_eq9` (call the accessor into a local `let` first, per Finding 18), `issuer.count == 1` and `issuer.entries[0].value` bytes match `"Test CA"`, `not_before_unix`/`not_after_unix` are non-zero and `not_before_unix < not_after_unix`, `subject.count == 1`, `modulus`/`public_exponent` compare equal to the small values encoded in the fixture.

Run: `./bin/souc run tests/run-pass/x509_parse_tbs_core.sio`
Expected: FAIL to compile — `x509_parse_tbs_core` doesn't exist yet.

- [ ] **Step 3: Implement `x509_parse_tbs_core` in `stdlib/x509/cert.sio`**

```sio
// -- append to stdlib/x509/cert.sio --

// UTCTime (YYMMDDHHMMSSZ) -> Unix timestamp. Per RFC 5280 SS4.1.2.5.1, a
// two-digit year YY is 20YY if YY < 50, else 19YY -- this project only ever
// decodes CURRENT/recent certificates, so every real YY it sees is < 50
// (20xx), but the rule is implemented per the standard regardless. Uses
// only additions and small-constant multiplications on i64 day/second
// counts -- no operation here approaches bit 63.
fn parse_utc_time_to_unix(buf: &RawBuf, start: i64, len: i64) -> i64 with IO {
    // "YYMMDDHHMMSSZ" is exactly 13 bytes.
    if len != 13 {
        return 0 - 1
    }
    let yy = ((rawbuf_get(buf, start) & 255) - 48) * 10 + ((rawbuf_get(buf, start + 1) & 255) - 48)
    let year_full = if yy < 50 { 2000 + yy } else { 1900 + yy }
    let mm = ((rawbuf_get(buf, start + 2) & 255) - 48) * 10 + ((rawbuf_get(buf, start + 3) & 255) - 48)
    let dd = ((rawbuf_get(buf, start + 4) & 255) - 48) * 10 + ((rawbuf_get(buf, start + 5) & 255) - 48)
    let hh = ((rawbuf_get(buf, start + 6) & 255) - 48) * 10 + ((rawbuf_get(buf, start + 7) & 255) - 48)
    let min = ((rawbuf_get(buf, start + 8) & 255) - 48) * 10 + ((rawbuf_get(buf, start + 9) & 255) - 48)
    let ss = ((rawbuf_get(buf, start + 10) & 255) - 48) * 10 + ((rawbuf_get(buf, start + 11) & 255) - 48)
    unix_timestamp_from_ymdhms(year_full, mm, dd, hh, min, ss)
}

// Days-from-civil algorithm (Howard Hinnant's well-known public-domain
// civil_from_days inverse), avoiding any library date function this
// project doesn't have. All intermediate values stay far below 2^32, let
// alone i64's bit 63.
fn unix_timestamp_from_ymdhms(year: i64, month: i64, day: i64, hour: i64, minute: i64, second: i64) -> i64 {
    var y = year
    if month <= 2 {
        y = y - 1
    }
    let era = if y >= 0 { y } else { y - 399 }
    let era_div = era / 400
    let yoe = y - era_div * 400
    var mp: i64 = month + 9
    if mp >= 12 {
        mp = mp - 12
    } else {
        mp = month - 3
    }
    // (month+9)%12 computed via the branch above since this project avoids
    // relying on '%' with any operand shape not already proven safe.
    let doy = (153 * mp + 2) / 5 + day - 1
    let yoe_leap_adj = yoe / 4 - yoe / 100
    let doe = yoe * 365 + yoe_leap_adj + doy
    let days_since_epoch = era_div * 146097 + doe - 719468
    days_since_epoch * 86400 + hour * 3600 + minute * 60 + second
}

// Returns, alongside the parsed Certificate, the DerReader positioned
// immediately after subjectPublicKeyInfo -- i.e. exactly where an optional
// [3] extensions field would begin, if present. Task 7's outer-assembly
// function needs this position to find the extensions block without
// re-walking the same fields a second time.
pub fn x509_parse_tbs_core(buf: &RawBuf, tbs_reader: &DerReader) -> (Certificate, DerReader, i64) with IO {
    var cert = certificate_zero()
    var cursor = DerReader { buf_ptr: tbs_reader.buf_ptr, pos: tbs_reader.pos, end: tbs_reader.end }

    let (r1, t1, s1) = der_read_tlv(&cursor)
    if s1 != DER_OK {
        return (cert, cursor, s1)
    }

    var version: i32 = 0
    var after_version = cursor
    if t1.class == DER_CLASS_CONTEXT && t1.tag_number == 0 {
        // Explicit [0] version wrapper -- descend, read the inner INTEGER.
        let (inner, e0) = der_enter(&cursor, &t1)
        if e0 != DER_OK {
            return (cert, cursor, e0)
        }
        let (inner2, ver_tag, s2) = der_read_tlv(&inner)
        if s2 != DER_OK {
            return (cert, cursor, s2)
        }
        let (ver_bigint, s3) = der_read_integer(&inner, &ver_tag)
        if s3 != DER_OK {
            return (cert, cursor, s3)
        }
        version = bigint_to_small_i32(&ver_bigint)
        let (adv, s4) = der_next(&cursor, &t1)
        if s4 != DER_OK {
            return (cert, cursor, s4)
        }
        after_version = adv
        let (r2, t2, s5) = der_read_tlv(&after_version)
        if s5 != DER_OK {
            return (cert, cursor, s5)
        }
        cert.version = version
        let (serial, s6) = der_read_integer(&after_version, &t2)
        if s6 != DER_OK {
            return (cert, cursor, s6)
        }
        cert.serial_number = serial
        let (after_serial, s7) = der_next(&after_version, &t2)
        if s7 != DER_OK {
            return (cert, cursor, s7)
        }
        return x509_parse_tbs_after_serial(buf, &after_serial, cert)
    } else {
        // No [0] wrapper: version defaults to 0 (v1), and t1/r1 IS the
        // serialNumber INTEGER already read.
        cert.version = 0
        let (serial, s6) = der_read_integer(&cursor, &t1)
        if s6 != DER_OK {
            return (cert, cursor, s6)
        }
        cert.serial_number = serial
        let (after_serial, s7) = der_next(&cursor, &t1)
        if s7 != DER_OK {
            return (cert, cursor, s7)
        }
        return x509_parse_tbs_after_serial(buf, &after_serial, cert)
    }
}

fn bigint_to_small_i32(v: &BigInt) -> i32 {
    if v.len == 0 {
        return 0
    }
    v.limbs[0] as i32   // version is always 0/1/2 -- fits in the low limb trivially
}

fn x509_parse_tbs_after_serial(buf: &RawBuf, r: &DerReader, cert_in: Certificate) -> (Certificate, DerReader, i64) with IO {
    var cert = cert_in
    // Never copy a struct via `*r` (this branch's established discipline) --
    // rebuild a fresh DerReader by reading r's fields individually whenever
    // an error return needs "the reader unchanged" as its second tuple slot.
    let r_copy = DerReader { buf_ptr: r.buf_ptr, pos: r.pos, end: r.end }

    // signature AlgorithmIdentifier
    let (r1, t1, s1) = der_read_tlv(r)
    if s1 != DER_OK { return (cert, r_copy, s1) }
    let (alg_inner, e1) = der_enter(r, &t1)
    if e1 != DER_OK { return (cert, r_copy, e1) }
    let (alg_inner2, oid_tag, s2) = der_read_tlv(&alg_inner)
    if s2 != DER_OK { return (cert, r_copy, s2) }
    if oid_tag.tag_number != DER_TAG_OBJECT_IDENTIFIER { return (cert, r_copy, X509_ERR_MALFORMED) }
    var oid_buf: [u8; 20] = [0; 20]
    var oi: i64 = 0
    while oi < oid_tag.content_len {
        oid_buf[oi as usize] = (rawbuf_get(buf, oid_tag.content_start + oi) & 255) as u8
        oi = oi + 1
    }
    cert.tbs_sig_alg_oid = oid_buf
    cert.tbs_sig_alg_oid_len = oid_tag.content_len as i32
    let (after_sig_alg, s3) = der_next(r, &t1)
    if s3 != DER_OK { return (cert, r_copy, s3) }

    // issuer Name
    let (r2, t2, s4) = der_read_tlv(&after_sig_alg)
    if s4 != DER_OK { return (cert, after_sig_alg, s4) }
    let (issuer, s5) = x509_parse_name(buf, &after_sig_alg, &t2)
    if s5 != DER_OK { return (cert, after_sig_alg, s5) }
    cert.issuer = issuer
    let (after_issuer, s6) = der_next(&after_sig_alg, &t2)
    if s6 != DER_OK { return (cert, after_sig_alg, s6) }

    // validity SEQUENCE { Time, Time }
    let (r3, t3, s7) = der_read_tlv(&after_issuer)
    if s7 != DER_OK { return (cert, after_issuer, s7) }
    let (validity_inner, e2) = der_enter(&after_issuer, &t3)
    if e2 != DER_OK { return (cert, after_issuer, e2) }
    let (v1, nb_tag, s8) = der_read_tlv(&validity_inner)
    if s8 != DER_OK { return (cert, after_issuer, s8) }
    cert.not_before_unix = parse_utc_time_to_unix(buf, nb_tag.content_start, nb_tag.content_len)
    let (after_nb, s9) = der_next(&validity_inner, &nb_tag)
    if s9 != DER_OK { return (cert, after_issuer, s9) }
    let (v2, na_tag, s10) = der_read_tlv(&after_nb)
    if s10 != DER_OK { return (cert, after_issuer, s10) }
    cert.not_after_unix = parse_utc_time_to_unix(buf, na_tag.content_start, na_tag.content_len)
    let (after_validity, s11) = der_next(&after_issuer, &t3)
    if s11 != DER_OK { return (cert, after_issuer, s11) }

    // subject Name
    let (r4, t4, s12) = der_read_tlv(&after_validity)
    if s12 != DER_OK { return (cert, after_validity, s12) }
    let (subject, s13) = x509_parse_name(buf, &after_validity, &t4)
    if s13 != DER_OK { return (cert, after_validity, s13) }
    cert.subject = subject
    let (after_subject, s14) = der_next(&after_validity, &t4)
    if s14 != DER_OK { return (cert, after_validity, s14) }

    // subjectPublicKeyInfo SEQUENCE { AlgorithmIdentifier, BIT STRING }
    let (r5, t5, s15) = der_read_tlv(&after_subject)
    if s15 != DER_OK { return (cert, after_subject, s15) }
    let (spki_inner, e3) = der_enter(&after_subject, &t5)
    if e3 != DER_OK { return (cert, after_subject, e3) }
    let (spki_inner2, spki_alg_tag, s16) = der_read_tlv(&spki_inner)
    if s16 != DER_OK { return (cert, after_subject, s16) }
    let (after_spki_alg, s17) = der_next(&spki_inner, &spki_alg_tag)
    if s17 != DER_OK { return (cert, after_subject, s17) }
    let (spki_inner3, bitstr_tag, s18) = der_read_tlv(&after_spki_alg)
    if s18 != DER_OK { return (cert, after_subject, s18) }
    if bitstr_tag.tag_number != DER_TAG_BIT_STRING { return (cert, after_subject, X509_ERR_MALFORMED) }
    // BIT STRING's first content byte is the "unused bits" count (always 0
    // for a byte-aligned RSAPublicKey); the RSAPublicKey DER starts at the
    // next byte.
    let rsa_key_start = bitstr_tag.content_start + 1
    let rsa_key_len = bitstr_tag.content_len - 1
    let rsa_key_reader = DerReader { buf_ptr: after_spki_alg.buf_ptr, pos: rsa_key_start, end: rsa_key_start + rsa_key_len }
    let (r6, rsa_seq_tag, s19) = der_read_tlv(&rsa_key_reader)
    if s19 != DER_OK { return (cert, after_subject, s19) }
    let (rsa_inner, e4) = der_enter(&rsa_key_reader, &rsa_seq_tag)
    if e4 != DER_OK { return (cert, after_subject, e4) }
    let (rsa_inner2, mod_tag, s20) = der_read_tlv(&rsa_inner)
    if s20 != DER_OK { return (cert, after_subject, s20) }
    let (modulus, s21) = der_read_integer(&rsa_inner, &mod_tag)
    if s21 != DER_OK { return (cert, after_subject, s21) }
    cert.modulus = modulus
    let (after_mod, s22) = der_next(&rsa_inner, &mod_tag)
    if s22 != DER_OK { return (cert, after_subject, s22) }
    let (rsa_inner3, exp_tag, s23) = der_read_tlv(&after_mod)
    if s23 != DER_OK { return (cert, after_subject, s23) }
    let (exponent, s24) = der_read_integer(&after_mod, &exp_tag)
    if s24 != DER_OK { return (cert, after_subject, s24) }
    cert.public_exponent = exponent
    // The BIT STRING (containing the RSAPublicKey) was read via a SEPARATE
    // child reader (rsa_key_reader) scoped inside the BIT STRING's own
    // content -- advancing past the whole subjectPublicKeyInfo field in the
    // OUTER cursor uses der_skip on after_subject/t5 (the SPKI SEQUENCE's
    // own tag), not anything derived from rsa_key_reader.
    let (after_spki, s25) = der_skip(&after_subject, &t5)
    if s25 != DER_OK { return (cert, after_subject, s25) }

    (cert, after_spki, X509_OK)
}

// Parses a Name (SEQUENCE OF RelativeDistinguishedName, each RDN a SET OF
// AttributeTypeAndValue) into an X509Name. Only the first AttributeTypeAndValue
// of each RDN SET is read (multi-valued RDNs -- more than one attribute in a
// single SET -- are rare in practice; capturing just the first is a
// documented, acceptable simplification, not a silent drop, since real CA
// certificates essentially never issue genuinely multi-valued RDNs).
fn x509_parse_name(buf: &RawBuf, r: &DerReader, tag: &DerTag) -> (X509Name, i64) with IO {
    var name = x509_name_zero()
    let (seq_inner, e0) = der_enter(r, tag)
    if e0 != DER_OK { return (name, e0) }

    var cursor = seq_inner
    var count: i32 = 0
    while der_at_end(&cursor) == false {
        if count >= X509_MAX_RDN_ENTRIES {
            return (name, X509_ERR_TOO_MANY_ENTRIES)
        }
        let (c1, rdn_set_tag, s1) = der_read_tlv(&cursor)
        if s1 != DER_OK { return (name, s1) }
        let (rdn_inner, e1) = der_enter(&cursor, &rdn_set_tag)
        if e1 != DER_OK { return (name, e1) }
        let (atv_inner, atv_tag, s2) = der_read_tlv(&rdn_inner)
        if s2 != DER_OK { return (name, s2) }
        let (atv_seq_inner, e2) = der_enter(&rdn_inner, &atv_tag)
        if e2 != DER_OK { return (name, e2) }
        let (oid_inner, oid_tag, s3) = der_read_tlv(&atv_seq_inner)
        if s3 != DER_OK { return (name, s3) }
        var oid_buf: [u8; 20] = [0; 20]
        var oi: i64 = 0
        while oi < oid_tag.content_len && oi < 20 {
            oid_buf[oi as usize] = (rawbuf_get(buf, oid_tag.content_start + oi) & 255) as u8
            oi = oi + 1
        }
        let (after_oid, s4) = der_next(&atv_seq_inner, &oid_tag)
        if s4 != DER_OK { return (name, s4) }
        let (val_inner, val_tag, s5) = der_read_tlv(&after_oid)
        if s5 != DER_OK { return (name, s5) }
        if val_tag.content_len > X509_MAX_NAME_VALUE_LEN {
            return (name, X509_ERR_VALUE_TOO_LONG)
        }
        var val_buf: [u8; 128] = [0; 128]
        var vi: i64 = 0
        while vi < val_tag.content_len {
            val_buf[vi as usize] = (rawbuf_get(buf, val_tag.content_start + vi) & 255) as u8
            vi = vi + 1
        }

        // Per Finding 22 (discovered during this task's own implementation,
        // not known when this section was first drafted): a whole-struct
        // literal written directly into an array-of-struct element
        // silently cross-contaminates which source array lands in which
        // destination field, once the struct has two or more [u8;N]
        // fields (RdnEntry has `oid:[u8;20]` and `value:[u8;128]`) and the
        // array is large enough (confirmed broken at RdnEntry's 16-entry
        // scale). Every field is assigned individually, directly into the
        // array element, instead.
        name.entries[count as usize].oid = oid_buf
        name.entries[count as usize].oid_len = oid_tag.content_len as i32
        name.entries[count as usize].value_tag = val_tag.tag_number
        name.entries[count as usize].value = val_buf
        name.entries[count as usize].value_len = val_tag.content_len as i32
        count = count + 1

        let (next_cursor, sk) = der_skip(&cursor, &rdn_set_tag)
        if sk != DER_OK { return (name, sk) }
        cursor = next_cursor
    }
    name.count = count
    (name, X509_OK)
}
```

**Note on the field-by-field array-element writes above**: this superseded the plan's original whole-struct-literal write (`name.entries[count as usize] = RdnEntry { ... }`), which Task 5's implementer found -- and the controller independently confirmed with a minimal repro -- silently cross-contaminates a struct's two `[u8;N]` fields when written whole into a sufficiently large array-of-struct element. See Finding 22 in `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`. This supersedes the now-stale claim, in the paragraph this replaced, that Task 1's check (a) had already confirmed this exact pattern safe -- check (a) tested a single-array-field struct; Finding 22 shows the two-array-field case is a distinct, narrower-scoped failure mode that check (a) did not cover.

**Note on `%`**: `unix_timestamp_from_ymdhms`'s comment mentions avoiding `%` — this project has not specifically audited `%` (modulo) on plain small non-negative `i64` values; it uses only `/` (already established safe for small values elsewhere on this branch, e.g. `der.sio`'s `padded_len` computation) and manual branch-based equivalents. If a future task finds `%` is in fact safe for small values, this is a candidate simplification, not a required one.

- [ ] **Step 4: Run the test and verify it passes**

Run: `./bin/souc run tests/run-pass/x509_parse_tbs_core.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/x509/cert.sio tests/run-pass/x509_parse_tbs_core.sio
git commit -m "feat(x509): parse TBSCertificate core fields (version, serial, algorithm, names, validity, SPKI)"
```

---

### Task 6: Extensions parsing (generic list + basicConstraints/keyUsage/extKeyUsage/key-identifiers) + SAN/issuerAltName

**Files:**
- Modify: `stdlib/x509/cert.sio`
- Test: `tests/run-pass/x509_parse_extensions.sio`, `tests/run-pass/x509_parse_san.sio`

**Interfaces:**
- Consumes: Task 5's parsing helpers and `DerReader` conventions; `stdlib/x509/oid.sio`.
- Produces: `pub fn x509_parse_extensions(buf: &RawBuf, r: &DerReader) -> ([ExtensionEntry; 32], i32, bool, i32, i32, [u8;32], i32, [u8;32], i32, i64)` (extensions list+count, `is_ca`, `path_len_constraint`, `key_usage_bits`, `subject_key_id`+len, `authority_key_id`+len, status — a wide return tuple, matching this project's established sentinel-tuple convention rather than inventing a new struct for it); `pub fn x509_parse_general_names(buf: &RawBuf, r: &DerReader, tag: &DerTag) -> ([GeneralName; 32], i32, i64)`.

- [ ] **Step 1: Write the failing extensions test**

Build a hand-crafted `Extensions` SEQUENCE (the `[3]`-tagged, explicitly-wrapped SEQUENCE OF Extension that appears at the end of a v3 TBSCertificate) containing: one `basicConstraints` extension (`cA=TRUE, pathLenConstraint=0`), one `keyUsage` extension (a BIT STRING with the `keyCertSign` and `digitalSignature` bits set), one `subjectKeyIdentifier` (a 20-byte OCTET STRING), one unrecognized/generic extension (an arbitrary OID this parser doesn't semantically decode, to prove the generic-capture path works for OIDs outside the "specially handled" set).

Create `tests/run-pass/x509_parse_extensions.sio` asserting: `extension_count == 4`, `is_ca == true`, `path_len_constraint == 0`, the `keyUsage` bits are set correctly (check the specific bit positions used), `subject_key_id_len == 20` and its bytes match, and that the 4th (generic) extension's OID/raw bytes are captured correctly in the `extensions` list even though it wasn't semantically decoded.

Run: `./bin/souc run tests/run-pass/x509_parse_extensions.sio`
Expected: FAIL to compile — `x509_parse_extensions` doesn't exist yet.

- [ ] **Step 2: Implement `x509_parse_extensions`**

```sio
// -- append to stdlib/x509/cert.sio --

pub fn x509_parse_extensions(buf: &RawBuf, r: &DerReader) -> ([ExtensionEntry; 32], i32, bool, i32, i32, [u8; 32], i32, [u8; 32], i32, i64) with IO {
    var extensions: [ExtensionEntry; 32] = [
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
        extension_entry_zero(), extension_entry_zero(), extension_entry_zero(), extension_entry_zero(),
    ]
    var count: i32 = 0
    var is_ca = false
    var path_len_constraint: i32 = 0 - 1
    var key_usage_bits: i32 = 0
    var subj_kid: [u8; 32] = [0; 32]
    var subj_kid_len: i32 = 0
    var auth_kid: [u8; 32] = [0; 32]
    var auth_kid_len: i32 = 0

    var cursor = DerReader { buf_ptr: r.buf_ptr, pos: r.pos, end: r.end }
    while der_at_end(&cursor) == false {
        if count >= X509_MAX_EXTENSIONS {
            return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, X509_ERR_TOO_MANY_ENTRIES)
        }
        let (c1, ext_seq_tag, s1) = der_read_tlv(&cursor)
        if s1 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s1) }
        let (ext_inner, e1) = der_enter(&cursor, &ext_seq_tag)
        if e1 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, e1) }

        let (oid_reader, oid_tag, s2) = der_read_tlv(&ext_inner)
        if s2 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s2) }
        var oid_buf: [u8; 20] = [0; 20]
        var oi: i64 = 0
        while oi < oid_tag.content_len && oi < 20 {
            oid_buf[oi as usize] = (rawbuf_get(buf, oid_tag.content_start + oi) & 255) as u8
            oi = oi + 1
        }
        let (after_oid, s3) = der_next(&ext_inner, &oid_tag)
        if s3 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s3) }

        let (c2, next_tag, s4) = der_read_tlv(&after_oid)
        if s4 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s4) }
        var critical = false
        var value_reader = after_oid
        var value_tag = next_tag
        if next_tag.tag_number == DER_TAG_BOOLEAN {
            let (crit_val, s5) = der_read_boolean(&after_oid, &next_tag)
            if s5 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s5) }
            critical = crit_val
            let (after_crit, s6) = der_next(&after_oid, &next_tag)
            if s6 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s6) }
            let (c3, val_tag2, s7) = der_read_tlv(&after_crit)
            if s7 != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, s7) }
            value_reader = after_crit
            value_tag = val_tag2
        }
        // value_tag is now the extnValue OCTET STRING's tag.
        if value_tag.content_len > X509_MAX_EXTENSION_VALUE_LEN {
            return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, X509_ERR_VALUE_TOO_LONG)
        }
        var val_buf: [u8; 512] = [0; 512]
        var vi: i64 = 0
        while vi < value_tag.content_len {
            val_buf[vi as usize] = (rawbuf_get(buf, value_tag.content_start + vi) & 255) as u8
            vi = vi + 1
        }

        // Per the newly discovered Finding 22 (a whole-struct copy -- literal
        // OR local var -- into an array-of-struct element silently
        // cross-contaminates which source array lands in which destination
        // field, once the struct has two or more [u8;N] fields and the
        // array is large enough; confirmed safe at SctEntry's ~160-byte/
        // entry x8 scale, confirmed BROKEN at ExtensionEntry's larger scale):
        // assign every field individually directly into the array element,
        // never via a struct literal or an intermediate local var copy.
        extensions[count as usize].oid = oid_buf
        extensions[count as usize].oid_len = oid_tag.content_len as i32
        extensions[count as usize].critical = critical
        extensions[count as usize].value = val_buf
        extensions[count as usize].value_len = value_tag.content_len as i32
        count = count + 1

        // Semantic decode for the extensions this layer specially recognizes.
        // extnValue's content is itself a DER-encoded structure (a nested
        // TLV inside the OCTET STRING we just captured raw bytes of) -- open
        // a fresh DerReader scoped to those bytes to decode it.
        var ext_value_reader = DerReader { buf_ptr: buf.ptr, pos: value_tag.content_start, end: value_tag.content_start + value_tag.content_len }

        // Per Finding 18: call each OID accessor once into a local `let`
        // before taking a reference to it -- never reference a would-be
        // top-level const directly (this file no longer has any).
        let oid_bc = oid_ext_basic_constraints()
        let oid_ku = oid_ext_key_usage()
        let oid_ski = oid_ext_subject_key_identifier()
        let oid_aki = oid_ext_authority_key_identifier()
        if oid_eq3(&oid_buf, oid_tag.content_len as i32, &oid_bc) {
            let (bc, s8) = x509_decode_basic_constraints(&ext_value_reader)
            let (bc_is_ca, bc_path_len) = bc
            is_ca = bc_is_ca
            path_len_constraint = bc_path_len
        } else if oid_eq3(&oid_buf, oid_tag.content_len as i32, &oid_ku) {
            let (ku, s9) = x509_decode_key_usage(buf, &ext_value_reader)
            key_usage_bits = ku
        } else if oid_eq3(&oid_buf, oid_tag.content_len as i32, &oid_ski) {
            let (kid, kid_len) = x509_decode_key_identifier(buf, &ext_value_reader)
            subj_kid = kid
            subj_kid_len = kid_len
        } else if oid_eq3(&oid_buf, oid_tag.content_len as i32, &oid_aki) {
            let (kid2, kid_len2) = x509_decode_key_identifier(buf, &ext_value_reader)
            auth_kid = kid2
            auth_kid_len = kid_len2
        }
        // Every other recognized-but-not-yet-specially-decoded OID (SAN,
        // issuerAltName, SCT list) is decoded by the CALLER (Task 7's
        // assembly function), which has access to this function's returned
        // `extensions` list to find them by OID and re-open a DerReader on
        // their captured `value`/`value_len` bytes -- this function's job
        // is the generic list plus the handful of extensions above, not
        // every extension this whole sub-project cares about.

        let (next_cursor, sk) = der_skip(&cursor, &ext_seq_tag)
        if sk != DER_OK { return (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, sk) }
        cursor = next_cursor
    }

    (extensions, count, is_ca, path_len_constraint, key_usage_bits, subj_kid, subj_kid_len, auth_kid, auth_kid_len, X509_OK)
}

// basicConstraints ::= SEQUENCE { cA BOOLEAN DEFAULT FALSE, pathLenConstraint INTEGER OPTIONAL }
fn x509_decode_basic_constraints(r: &DerReader) -> ((bool, i32), i64) with IO {
    if der_at_end(r) == true {
        return ((false, 0 - 1), X509_OK)   // empty SEQUENCE: both fields at their defaults
    }
    let (c1, t1, s1) = der_read_tlv(r)
    if s1 != DER_OK { return ((false, 0 - 1), s1) }
    var is_ca = false
    var cursor_after_ca = c1
    var have_ca_field = false
    if t1.tag_number == DER_TAG_BOOLEAN {
        let (ca_val, s2) = der_read_boolean(r, &t1)
        if s2 != DER_OK { return ((false, 0 - 1), s2) }
        is_ca = ca_val
        have_ca_field = true
        let (after_ca, s3) = der_next(r, &t1)
        if s3 != DER_OK { return ((false, 0 - 1), s3) }
        cursor_after_ca = after_ca
    }
    if der_at_end(&cursor_after_ca) == true {
        return ((is_ca, 0 - 1), X509_OK)
    }
    let (c2, t2, s4) = der_read_tlv(&cursor_after_ca)
    if s4 != DER_OK { return ((is_ca, 0 - 1), s4) }
    let (path_len_bigint, s5) = der_read_integer(&cursor_after_ca, &t2)
    if s5 != DER_OK { return ((is_ca, 0 - 1), s5) }
    ((is_ca, bigint_to_small_i32(&path_len_bigint)), X509_OK)
}

// keyUsage ::= BIT STRING. Returns the bits as an i32 bitmask, bit N of the
// mask == bit N of the DER BIT STRING (per its own bit-numbering, unrelated
// convention subtleties intentionally not reinterpreted here -- callers
// checking a specific named bit, e.g. keyCertSign, use the RFC 5280 SS4.2.1.3
// bit-position table directly against this raw mask).
fn x509_decode_key_usage(buf: &RawBuf, r: &DerReader) -> (i32, i64) with IO {
    let (c1, t1, s1) = der_read_tlv(r)
    if s1 != DER_OK { return (0, s1) }
    if t1.tag_number != DER_TAG_BIT_STRING { return (0, X509_ERR_MALFORMED) }
    if t1.content_len < 1 || t1.content_len > 3 { return (0, X509_ERR_MALFORMED) }
    // content[0] = unused-bit count; content[1..] = up to 2 bytes of bits,
    // MSB-first within each byte (bit 0 of the KeyUsage is the MSB of the
    // first content byte after the unused-bits count).
    var mask: i32 = 0
    var i: i64 = 1
    while i < t1.content_len {
        let byte_val = (rawbuf_get(buf, t1.content_start + i) & 255) as i32
        mask = (mask << 8) | byte_val
        i = i + 1
    }
    (mask, X509_OK)
}

// subjectKeyIdentifier / authorityKeyIdentifier's simple form ::= OCTET STRING.
// (authorityKeyIdentifier's full form is a SEQUENCE with an OPTIONAL
// keyIdentifier field among others -- this function assumes the common,
// simple bare-OCTET-STRING encoding real CAs almost universally use for
// subjectKeyIdentifier; a caller needing authorityKeyIdentifier's other
// optional fields, keyid aside, reads them from the raw `extensions` list
// entry directly.)
fn x509_decode_key_identifier(buf: &RawBuf, r: &DerReader) -> ([u8; 32], i32) with IO {
    var out: [u8; 32] = [0; 32]
    let (c1, t1, s1) = der_read_tlv(r)
    if s1 != DER_OK { return (out, 0) }
    if t1.tag_number != DER_TAG_OCTET_STRING || t1.content_len > 32 { return (out, 0) }
    var i: i64 = 0
    while i < t1.content_len {
        out[i as usize] = (rawbuf_get(buf, t1.content_start + i) & 255) as u8
        i = i + 1
    }
    (out, t1.content_len as i32)
}
```

- [ ] **Step 3: Run the extensions test and verify it passes**

Run: `./bin/souc run tests/run-pass/x509_parse_extensions.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 4: Write the failing SAN test**

Build a hand-crafted SAN `GeneralNames` SEQUENCE containing one of each of the 9 `GeneralName` variants (or, if Task 1's audit found check (b) failing, 8 variants — skip `directoryName`'s dedicated fixture and instead confirm it falls into the generic raw-bytes path like `x400Address`). Create `tests/run-pass/x509_parse_san.sio` asserting `x509_parse_general_names` returns the correct `tag`/`value`/`value_len` (or `oid`/`oid_len` for `otherName`/`registeredID`) for each entry, and `count == 9` (or `8`).

Run: `./bin/souc run tests/run-pass/x509_parse_san.sio`
Expected: FAIL to compile — `x509_parse_general_names` doesn't exist yet.

- [ ] **Step 5: Implement `x509_parse_general_names`**

```sio
// -- append to stdlib/x509/cert.sio --

// GeneralNames ::= SEQUENCE SIZE (1..MAX) OF GeneralName
// GeneralName ::= CHOICE { [0] otherName, [1] rfc822Name IA5String, [2]
//   dNSName IA5String, [3] x400Address, [4] directoryName Name, [5]
//   ediPartyName, [6] uniformResourceIdentifier IA5String, [7] iPAddress
//   OCTET STRING, [8] registeredID OBJECT IDENTIFIER } -- every variant is
//   an implicitly- or explicitly-tagged context-specific value; per the DER
//   decoder's own documented limitation, this parser reads each GeneralName
//   entry's raw tag/content directly (never needing der_enter for the
//   simple string/bytes variants, since those are IMPLICIT [primitive]
//   tags), only descending via der_enter for the CONSTRUCTED variants
//   (otherName, directoryName, ediPartyName, which are EXPLICIT/constructed
//   per RFC 5280's ASN.1 module).
pub fn x509_parse_general_names(buf: &RawBuf, r: &DerReader, tag: &DerTag) -> ([GeneralName; 32], i32, i64) with IO {
    var out: [GeneralName; 32] = [
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
        general_name_zero(), general_name_zero(), general_name_zero(), general_name_zero(),
    ]
    var count: i32 = 0

    let (seq_inner, e0) = der_enter(r, tag)
    if e0 != DER_OK { return (out, 0, e0) }

    var cursor = seq_inner
    while der_at_end(&cursor) == false {
        if count >= X509_MAX_SAN_ENTRIES {
            return (out, count, X509_ERR_TOO_MANY_ENTRIES)
        }
        let (c1, gn_tag, s1) = der_read_tlv(&cursor)
        if s1 != DER_OK { return (out, count, s1) }

        // Per the newly discovered Finding 22: a whole-struct copy (literal
        // OR local var, like the `entry` this comment used to build and
        // then assign whole into `out[count]`) silently cross-contaminates
        // which source array lands in which destination field, once the
        // struct has two or more [u8;N] fields and the array is large
        // enough (confirmed safe at SctEntry's ~160-byte/entry x8 scale,
        // confirmed BROKEN at GeneralName's larger 32-entry scale). Every
        // field is therefore assigned individually, directly into the
        // array element `out[count as usize]` -- never through an
        // intermediate local `entry` variable or a struct literal.
        out[count as usize].tag = gn_tag.tag_number   // the raw context-specific [N] number IS the GeneralName choice discriminant

        if gn_tag.tag_number == GENERAL_NAME_DIRECTORY_NAME && gn_tag.constructed == true {
            // Only attempted if the struct actually has this field --
            // Task 2's ruling (from Task 1's audit) determines whether this
            // branch is reachable code at all; if Task 1's check (b) failed,
            // this whole `if` branch is DELETED from the implementation
            // (directoryName falls through to the generic raw-bytes `else`
            // branch below instead, same as x400Address/ediPartyName).
            let (dn_reader, e1) = der_enter(&cursor, &gn_tag)
            if e1 == DER_OK {
                let (dn_seq_reader, dn_seq_tag, s2) = der_read_tlv(&dn_reader)
                if s2 == DER_OK {
                    let (dn, s3) = x509_parse_name(buf, &dn_reader, &dn_seq_tag)
                    if s3 == DER_OK {
                        out[count as usize].directory_name = dn
                    }
                }
            }
        } else if gn_tag.tag_number == GENERAL_NAME_OTHER_NAME || gn_tag.tag_number == GENERAL_NAME_REGISTERED_ID {
            if gn_tag.content_len > 20 {
                // truncate rather than fail the whole certificate over an
                // oversized otherName/registeredID OID -- documented, not silent
                out[count as usize].oid_len = 0
            } else {
                var oid_buf: [u8; 20] = [0; 20]
                var oi: i64 = 0
                while oi < gn_tag.content_len {
                    oid_buf[oi as usize] = (rawbuf_get(buf, gn_tag.content_start + oi) & 255) as u8
                    oi = oi + 1
                }
                out[count as usize].oid = oid_buf
                out[count as usize].oid_len = gn_tag.content_len as i32
            }
        } else {
            // rfc822Name, dNSName, x400Address, ediPartyName, URI,
            // iPAddress -- and directoryName too, if Task 1's check (b)
            // failed -- all captured as raw content bytes.
            if gn_tag.content_len > X509_MAX_GENERAL_NAME_VALUE_LEN {
                return (out, count, X509_ERR_VALUE_TOO_LONG)
            }
            var val_buf: [u8; 253] = [0; 253]
            var vi: i64 = 0
            while vi < gn_tag.content_len {
                val_buf[vi as usize] = (rawbuf_get(buf, gn_tag.content_start + vi) & 255) as u8
                vi = vi + 1
            }
            out[count as usize].value = val_buf
            out[count as usize].value_len = gn_tag.content_len as i32
        }

        count = count + 1

        let (next_cursor, sk) = der_skip(&cursor, &gn_tag)
        if sk != DER_OK { return (out, count, sk) }
        cursor = next_cursor
    }

    (out, count, X509_OK)
}
```

- [ ] **Step 6: Run the SAN test and verify it passes**

Run: `./bin/souc run tests/run-pass/x509_parse_san.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 7: Commit**

```bash
git add stdlib/x509/cert.sio tests/run-pass/x509_parse_extensions.sio tests/run-pass/x509_parse_san.sio
git commit -m "feat(x509): parse Extensions list, basicConstraints/keyUsage/key-identifiers, and SAN GeneralName entries"
```

---

### Task 7: Outer Certificate assembly + signature verification integration

**Files:**
- Modify: `stdlib/x509/cert.sio`
- Test: `tests/run-pass/x509_parse_full_certificate.sio`, `tests/run-pass/x509_verify_signature.sio`

**Interfaces:**
- Consumes: Tasks 2-6's parsing functions; `stdlib/crypto/pkcs1.sio`; `stdlib/hash::{sha1,sha256,sha384,sha512}`; `stdlib/x509/sct.sio`; `stdlib/x509/oid.sio`.
- Produces: `pub fn x509_parse_certificate(buf: &RawBuf, len: i64) -> (Certificate, i64)` (the full outer `Certificate SEQUENCE { TBSCertificate, AlgorithmIdentifier, BIT STRING }` parse, wiring `tbs_start`/`tbs_len`, the outer signature/algorithm, and cross-checking it against the TBS's own inner signature algorithm OID); `pub fn x509_verify_signature(buf: &RawBuf, cert: &Certificate, issuer_modulus: &BigInt, issuer_exponent: &BigInt) -> bool`.

- [ ] **Step 1: Write the failing full-certificate-parse test**

Extend Task 5's Step 1 fixture: wrap the TBSCertificate SEQUENCE in the full outer `Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signatureValue }` structure, add a minimal `Extensions [3]` block (reuse Task 6's fixture), and a plausible (not necessarily cryptographically valid — this test checks structural assembly, not verification) outer signature BIT STRING.

Create `tests/run-pass/x509_parse_full_certificate.sio` asserting: `x509_parse_certificate` returns `X509_OK`, `cert.tbs_start`/`cert.tbs_len` point at exactly the TBSCertificate SEQUENCE's own TLV bytes within the buffer (assert this by independently re-reading `rawbuf_get(buf, cert.tbs_start)` equals the TBSCertificate's own tag byte, `0x30`), `cert.outer_sig_alg_oid` matches `cert.tbs_sig_alg_oid` (the cross-check), and every field from Tasks 5/6's own tests is still correctly populated on the assembled result.

Run: `./bin/souc run tests/run-pass/x509_parse_full_certificate.sio`
Expected: FAIL to compile — `x509_parse_certificate` doesn't exist yet.

- [ ] **Step 2: Implement `x509_parse_certificate`**

**Note (superseded by the time this task is dispatched):** this step was originally drafted to widen `x509_parse_tbs_core`'s return type to `(Certificate, DerReader, i64)`. That widening already shipped as part of Task 5 -- `stdlib/x509/cert.sio`'s `pub fn x509_parse_tbs_core(buf: &RawBuf, tbs_reader: &DerReader) -> (Certificate, DerReader, i64) with IO` already returns exactly this 3-tuple, with the reader positioned right after `subjectPublicKeyInfo`. No signature change is needed here -- just call it as shown in Step 2's code below (which already expects the 3-tuple).

```sio
// -- append to stdlib/x509/cert.sio --

pub fn x509_parse_certificate(buf: &RawBuf, len: i64) -> (Certificate, i64) with IO {
    let top = der_reader_new(buf, 0, len)
    let (top2, cert_seq_tag, s1) = der_read_tlv(&top)
    if s1 != DER_OK { return (certificate_zero(), s1) }
    if cert_seq_tag.constructed == false { return (certificate_zero(), X509_ERR_MALFORMED) }

    let (cert_inner, e0) = der_enter(&top, &cert_seq_tag)
    if e0 != DER_OK { return (certificate_zero(), e0) }

    // Capture the TBSCertificate's own tag-byte position BEFORE der_read_tlv
    // advances past it -- this position, together with the TLV's own
    // content_start+content_len below, IS the exact byte range RFC 5280
    // SS4.1.1.3 says gets hashed for signature verification.
    let tbs_start_pos = cert_inner.pos
    let (tbs_reader, tbs_tag, s2) = der_read_tlv(&cert_inner)
    if s2 != DER_OK { return (certificate_zero(), s2) }

    // Per Finding 25 (docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md):
    // a tuple-destructured local (`let (parsed, ...) = x509_parse_tbs_core(...)`)
    // does not propagate `parsed`'s struct type, which would leave every
    // later `cert.field`/`cert.extensions[i].field` access in THIS function
    // exposed to Finding 24's corruption. Work around it the same way
    // tests/run-pass/x509_parse_tbs_core.sio does: declare `cert` FIRST via
    // a plain call whose single-struct return type IS resolved correctly
    // (certificate_zero() -> Certificate), then overwrite its value via
    // plain assignment (not `let`/`var`) -- assignment doesn't touch the
    // struct-type binding table, so the correct binding survives.
    var cert = certificate_zero()
    let (parsed, after_spki, s3) = x509_parse_tbs_core(buf, &cert_inner)
    if s3 != DER_OK { return (certificate_zero(), s3) }
    cert = parsed
    cert.tbs_start = tbs_start_pos
    cert.tbs_len = (tbs_tag.content_start + tbs_tag.content_len) - tbs_start_pos

    // Optional [3] extensions -- present iff cert.version == 2 (v3) and
    // after_spki isn't already at the end of the TBSCertificate's content.
    if cert.version == 2 && der_at_end(&after_spki) == false {
        let (ec, ext_wrap_tag, es) = der_read_tlv(&after_spki)
        if es != DER_OK { return (certificate_zero(), es) }
        if ext_wrap_tag.class == DER_CLASS_CONTEXT && ext_wrap_tag.tag_number == 3 {
            let (ext_wrap_inner, ee) = der_enter(&after_spki, &ext_wrap_tag)
            if ee != DER_OK { return (certificate_zero(), ee) }
            let (ext_list_inner_pre, ext_seq_tag, ess) = der_read_tlv(&ext_wrap_inner)
            if ess != DER_OK { return (certificate_zero(), ess) }
            let (ext_list_inner, eei) = der_enter(&ext_wrap_inner, &ext_seq_tag)
            if eei != DER_OK { return (certificate_zero(), eei) }

            let (extensions, ext_count, is_ca, path_len, key_usage, subj_kid, subj_kid_len, auth_kid, auth_kid_len, ext_status) = x509_parse_extensions(buf, &ext_list_inner)
            if ext_status != DER_OK { return (certificate_zero(), ext_status) }
            cert.extensions = extensions
            cert.extension_count = ext_count
            cert.is_ca = is_ca
            cert.path_len_constraint = path_len
            cert.key_usage_bits = key_usage
            cert.subject_key_id = subj_kid
            cert.subject_key_id_len = subj_kid_len
            cert.authority_key_id = auth_kid
            cert.authority_key_id_len = auth_kid_len

            // Find SAN/issuerAltName/SCT among the raw extensions list by
            // OID, copy each one's captured bytes into a fresh small RawBuf
            // (der_reader_new needs a RawBuf, and ExtensionEntry.value is a
            // plain [u8;512] array, not one), and decode it.
            var xi: i32 = 0
            while xi < ext_count {
                let entry = cert.extensions[xi as usize]
                let entry_buf = rawbuf_new(entry.value_len as i64)
                var cbi: i64 = 0
                while cbi < (entry.value_len as i64) {
                    rawbuf_set(&entry_buf, cbi, (entry.value[cbi as usize] as i64) & 255)
                    cbi = cbi + 1
                }
                let entry_reader = der_reader_new(&entry_buf, 0, entry.value_len as i64)

                // Per Finding 18: call each OID accessor once into a local
                // `let` before taking a reference to it.
                let oid_san = oid_ext_subject_alt_name()
                let oid_ian = oid_ext_issuer_alt_name()
                let oid_sct = oid_ext_sct_list()
                if oid_eq3(&entry.oid, entry.oid_len, &oid_san) {
                    let (san_c, san_tag, san_s) = der_read_tlv(&entry_reader)
                    if san_s == DER_OK {
                        let (san_list, san_count, san_status) = x509_parse_general_names(&entry_buf, &entry_reader, &san_tag)
                        if san_status == DER_OK {
                            cert.san_entries = san_list
                            cert.san_count = san_count
                        }
                    }
                } else if oid_eq3(&entry.oid, entry.oid_len, &oid_ian) {
                    let (ian_c, ian_tag, ian_s) = der_read_tlv(&entry_reader)
                    if ian_s == DER_OK {
                        let (ian_list, ian_count, ian_status) = x509_parse_general_names(&entry_buf, &entry_reader, &ian_tag)
                        if ian_status == DER_OK {
                            cert.issuer_alt_names = ian_list
                            cert.issuer_alt_name_count = ian_count
                        }
                    }
                } else if oid_eq10(&entry.oid, entry.oid_len, &oid_sct) {
                    // extnValue is itself an OCTET STRING wrapping the
                    // actual RFC 6962 length-prefixed bytes -- one more
                    // DER unwrap before sct_list_decode sees the real data.
                    let (sct_c, sct_octet_tag, sct_s) = der_read_tlv(&entry_reader)
                    if sct_s == DER_OK {
                        let (scts, sct_count, sct_status) = sct_list_decode(&entry_buf, sct_octet_tag.content_start, sct_octet_tag.content_len)
                        if sct_status == DER_OK {
                            cert.sct_entries = scts
                            cert.sct_count = sct_count
                        }
                    }
                }
                xi = xi + 1
            }
        }
    }

    let (after_tbs, sk1) = der_skip(&cert_inner, &tbs_tag)
    if sk1 != DER_OK { return (certificate_zero(), sk1) }

    let (c2, outer_alg_tag, s4) = der_read_tlv(&after_tbs)
    if s4 != DER_OK { return (certificate_zero(), s4) }
    let (outer_alg_inner, e1) = der_enter(&after_tbs, &outer_alg_tag)
    if e1 != DER_OK { return (certificate_zero(), e1) }
    let (outer_oid_reader, outer_oid_tag, s5) = der_read_tlv(&outer_alg_inner)
    if s5 != DER_OK { return (certificate_zero(), s5) }
    var outer_oid_buf: [u8; 20] = [0; 20]
    var ooi: i64 = 0
    while ooi < outer_oid_tag.content_len && ooi < 20 {
        outer_oid_buf[ooi as usize] = (rawbuf_get(buf, outer_oid_tag.content_start + ooi) & 255) as u8
        ooi = ooi + 1
    }
    cert.outer_sig_alg_oid = outer_oid_buf
    cert.outer_sig_alg_oid_len = outer_oid_tag.content_len as i32

    if oid_eq(&cert.outer_sig_alg_oid, cert.outer_sig_alg_oid_len, &cert.tbs_sig_alg_oid, cert.tbs_sig_alg_oid_len) == false {
        return (cert, X509_ERR_SIG_ALG_MISMATCH)
    }

    let (after_outer_alg, sk2) = der_skip(&after_tbs, &outer_alg_tag)
    if sk2 != DER_OK { return (certificate_zero(), sk2) }

    let (c3, sig_tag, s6) = der_read_tlv(&after_outer_alg)
    if s6 != DER_OK { return (certificate_zero(), s6) }
    if sig_tag.tag_number != DER_TAG_BIT_STRING { return (certificate_zero(), X509_ERR_MALFORMED) }
    // Skip the leading "unused bits" byte, same as the RSAPublicKey BIT
    // STRING handling in Task 5.
    let sig_bytes_start = sig_tag.content_start + 1
    let sig_bytes_len = sig_tag.content_len - 1
    if sig_bytes_len > X509_MAX_SIGNATURE_LEN { return (certificate_zero(), X509_ERR_VALUE_TOO_LONG) }
    var sig_buf: [u8; 512] = [0; 512]
    var si: i64 = 0
    while si < sig_bytes_len {
        sig_buf[si as usize] = (rawbuf_get(buf, sig_bytes_start + si) & 255) as u8
        si = si + 1
    }
    cert.outer_signature = sig_buf
    cert.outer_signature_len = sig_bytes_len as i32

    (cert, X509_OK)
}

pub fn x509_verify_signature(buf: &RawBuf, cert: &Certificate, issuer_modulus: &BigInt, issuer_exponent: &BigInt) -> bool with IO {
    var hash_algorithm: i32 = 0
    // Per Finding 18: call each OID accessor once into a local `let`
    // before taking a reference to it.
    let oid_s1 = oid_sha1_with_rsa()
    let oid_s256 = oid_sha256_with_rsa()
    let oid_s384 = oid_sha384_with_rsa()
    let oid_s512 = oid_sha512_with_rsa()
    if oid_eq9(&cert.outer_sig_alg_oid, cert.outer_sig_alg_oid_len, &oid_s1) {
        hash_algorithm = PKCS1_HASH_SHA1
    } else if oid_eq9(&cert.outer_sig_alg_oid, cert.outer_sig_alg_oid_len, &oid_s256) {
        hash_algorithm = PKCS1_HASH_SHA256
    } else if oid_eq9(&cert.outer_sig_alg_oid, cert.outer_sig_alg_oid_len, &oid_s384) {
        hash_algorithm = PKCS1_HASH_SHA384
    } else if oid_eq9(&cert.outer_sig_alg_oid, cert.outer_sig_alg_oid_len, &oid_s512) {
        hash_algorithm = PKCS1_HASH_SHA512
    } else {
        return false   // unsupported algorithm -- see spec Non-Goals (no ECDSA/Ed25519)
    }

    var message_hash: [u8; 64] = [0; 64]
    var message_hash_len: i64 = 0

    // stdlib/hash/*'s shaN(buf, len) functions always hash starting at
    // buf's own offset 0 -- they have no "start offset" parameter. Since
    // the TBSCertificate bytes to hash live at cert.tbs_start > 0 inside a
    // larger buffer, copy exactly that byte range into a fresh RawBuf
    // (offset 0) before hashing, rather than extending stdlib/hash/*'s
    // signatures (which would touch already-reviewed, already-shipped code
    // outside this plan's scope for no other benefit).
    let tbs_copy = rawbuf_new(cert.tbs_len)
    var ci: i64 = 0
    while ci < cert.tbs_len {
        rawbuf_set(&tbs_copy, ci, rawbuf_get(buf, cert.tbs_start + ci))
        ci = ci + 1
    }
    if hash_algorithm == PKCS1_HASH_SHA1 {
        let h = sha1(&tbs_copy, cert.tbs_len)
        var hi: i64 = 0
        while hi < 20 { message_hash[hi as usize] = h[hi as usize]; hi = hi + 1 }
        message_hash_len = 20
    } else if hash_algorithm == PKCS1_HASH_SHA256 {
        let h = sha256(&tbs_copy, cert.tbs_len)
        var hi: i64 = 0
        while hi < 32 { message_hash[hi as usize] = h[hi as usize]; hi = hi + 1 }
        message_hash_len = 32
    } else if hash_algorithm == PKCS1_HASH_SHA384 {
        let h = sha384(&tbs_copy, cert.tbs_len)
        var hi: i64 = 0
        while hi < 48 { message_hash[hi as usize] = h[hi as usize]; hi = hi + 1 }
        message_hash_len = 48
    } else {
        let h = sha512(&tbs_copy, cert.tbs_len)
        var hi: i64 = 0
        while hi < 64 { message_hash[hi as usize] = h[hi as usize]; hi = hi + 1 }
        message_hash_len = 64
    }

    let sig_buf = rawbuf_new(cert.outer_signature_len as i64)
    var sbi: i64 = 0
    while sbi < (cert.outer_signature_len as i64) {
        rawbuf_set(&sig_buf, sbi, (cert.outer_signature[sbi as usize] as i64) & 255)
        sbi = sbi + 1
    }
    let signature_bigint = bytes_be_to_bigint(&sig_buf, cert.outer_signature_len as i64)

    let modulus_byte_len = bigint_byte_length(issuer_modulus)
    pkcs1_v15_verify(&signature_bigint, issuer_modulus, issuer_exponent, modulus_byte_len, hash_algorithm, &message_hash, message_hash_len)
}
```

This task also needs two small new helpers, used at the end of `x509_verify_signature` above — add them to `stdlib/x509/cert.sio`:

```sio
// Big-endian bytes -> BigInt. Same byte-by-byte bigint_mul-by-256-then-
// bigint_add composition stdlib/asn1/der.sio's der_read_integer already
// uses for the identical conceptual conversion -- reused here rather than
// re-derived, since it's already proven correct and safe (Finding 11-aware:
// every intermediate stays within BigInt's own limb-bounded arithmetic).
fn bytes_be_to_bigint(buf: &RawBuf, len: i64) -> BigInt with IO {
    var acc = bigint_zero()
    let base256 = bigint_from_u32(256)
    var i: i64 = 0
    while i < len {
        let byte_val = rawbuf_get(buf, i) & 255
        let digit = bigint_from_u32(byte_val as u32)
        let scaled = bigint_mul(&acc, &base256)
        acc = bigint_add(&scaled, &digit)
        i = i + 1
    }
    acc
}

// The minimal big-endian byte length of v's magnitude -- v.len limbs is
// 2*v.len bytes UNLESS the most-significant limb's own high byte is zero,
// in which case the true minimal encoding is one byte shorter. This is
// PKCS#1's `k` parameter: the modulus's own encoded byte length, which
// must be exact (not merely "wide enough"), since pkcs1_v15_verify pads
// the recovered EM to exactly this many bytes.
fn bigint_byte_length(v: &BigInt) -> i64 {
    if v.len == 0 {
        return 0
    }
    let top_limb = v.limbs[(v.len - 1) as usize]
    let top_byte = (top_limb as i64) >> 8
    if top_byte == 0 {
        (v.len as i64) * 2 - 1
    } else {
        (v.len as i64) * 2
    }
}
```

- [ ] **Step 3: Run the full-certificate-parse test and verify it passes**

Run: `./bin/souc run tests/run-pass/x509_parse_full_certificate.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 4: Write and run the signature-verification test**

Create `tests/run-pass/x509_verify_signature.sio`: reuse Task 3's independently-derived real RSA test vector (Step 1 of Task 3) to build a SYNTHETIC but cryptographically genuine "certificate" fixture — the TBSCertificate bytes don't need to be a fully realistic certificate, only bytes that get hashed and whose hash is what Task 3's real signature actually signs (so either re-derive a fresh openssl-signed vector where the "message" IS a byte sequence shaped like a TBSCertificate SEQUENCE, or, more simply, sign the exact TBSCertificate bytes from Task 5/7's own fixture with `openssl dgst -sha256 -sign` during this task, using a freshly-generated key pair, and use THAT key pair's modulus/exponent as `issuer_modulus`/`issuer_exponent`). Assert `x509_verify_signature` returns `true` for the valid case, and `false` for at least one deliberately-corrupted case (flip a byte inside `tbs_start..tbs_start+tbs_len`, or use the wrong issuer key).

Run: `./bin/souc run tests/run-pass/x509_verify_signature.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/x509/cert.sio tests/run-pass/x509_parse_full_certificate.sio tests/run-pass/x509_verify_signature.sio
git commit -m "feat(x509): assemble full Certificate parse and wire up PKCS#1 signature verification"
```

---

### Task 8: Real-certificate end-to-end test + adversarial cases

**Files:**
- Test: `tests/run-pass/x509_real_certificate.sio`, `tests/run-pass/x509_adversarial.sio`

**Interfaces:**
- Consumes: everything from Tasks 1-7.

- [ ] **Step 1: Obtain and independently cross-check a real certificate's DER bytes**

Using an external tool during test authoring (e.g. `openssl s_client -connect <well-known-site>:443 -showcerts </dev/null 2>/dev/null | openssl x509 -outform DER > cert.der`, or generating a realistic self-signed test certificate with `openssl req -x509 -newkey rsa:2048 -days 365 -nodes -subj "/CN=test.example"`), obtain a real certificate's DER bytes and convert them into a Sounio byte-array literal (following this branch's established pattern for embedding a long byte sequence — a `let bytes = [0x30, 0x82, ...]` array literal built via a script, or `rawbuf_set` loop over a hardcoded array). Independently cross-check the parsed output against the same certificate's `openssl x509 -text -noout` output for: serial number, subject CN, issuer CN, validity dates (converted to the same Unix-timestamp representation for comparison), and the RSA modulus's first/last few bytes.

- [ ] **Step 2: Write and run the real-certificate test**

Create `tests/run-pass/x509_real_certificate.sio` calling `x509_parse_certificate` on the Step 1 bytes, asserting the cross-checked fields from Step 1 match, and — if the certificate is self-signed (its own modulus/exponent verify its own signature) or its issuer's public key is independently available — calling `x509_verify_signature` and asserting `true`.

Run: `./bin/souc run tests/run-pass/x509_real_certificate.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 3: Write and run adversarial tests**

Create `tests/run-pass/x509_adversarial.sio` covering: (a) outer `signatureAlgorithm` OID deliberately set to a different (but still recognized) algorithm than the TBS's own inner one — `x509_parse_certificate` must return `X509_ERR_SIG_ALG_MISMATCH`; (b) a single tampered byte inside the `tbs_start..tbs_start+tbs_len` range of an otherwise-valid signed fixture (reuse Task 7's Step 4 vector) — `x509_verify_signature` must return `false`; (c) a fixture with 33 extensions (one over `X509_MAX_EXTENSIONS`) — must return `X509_ERR_TOO_MANY_ENTRIES`, not silently drop the 33rd; (d) an RDN value 129 bytes long (one over `X509_MAX_NAME_VALUE_LEN`) — must return `X509_ERR_VALUE_TOO_LONG`.

Run: `./bin/souc run tests/run-pass/x509_adversarial.sio`
Expected: prints a success message, exits 0.

- [ ] **Step 4: Run the whole x509 test group together**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_` and separately `bash scripts/run_sio_test_suite.sh --filter-prefix pkcs1_`
Expected: all x509_ and pkcs1_ test files passing (confirm the exact count against what's actually present at this point in the plan — do not assume a specific number without counting the files yourself).

- [ ] **Step 5: Commit**

```bash
git add tests/run-pass/x509_real_certificate.sio tests/run-pass/x509_adversarial.sio
git commit -m "test(x509): add real-certificate end-to-end test and adversarial malformed-input coverage"
```

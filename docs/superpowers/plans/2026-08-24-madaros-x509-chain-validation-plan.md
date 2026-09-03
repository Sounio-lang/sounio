<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-24-madaros-x509-chain-validation-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-24-madaros-x509-chain-validation-plan
-->

# Madaros X.509 Chain Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** validate a certificate chain presented by a TLS server against the system's trusted root CA store, with generic path building, hostname verification, and fail-soft OCSP-stapling revocation checking — the first of two sub-projects needed to bring real HTTPS to Madaros.

**Architecture:** four new modules under `stdlib/x509/` (`pem.sio`, `trust_store.sio`, `chain.sio`, `ocsp.sio`), built entirely on already-existing, already-tested stdlib (`asn1::der`, `x509::cert`, `x509::oid`, `crypto::pkcs1`, `bignum::bigint`, `net::socket`'s `RawBuf`, `encoding::base64`). `chain.sio` is the only module the (separate, not-yet-planned) TLS handshake sub-project will call.

**Tech Stack:** Sounio/Madaros (`./bin/souc`, wrapping `bin/madaros-linux-x86_64`), no non-Sounio code anywhere.

**Spec:** `docs/superpowers/specs/2026-08-24-madaros-x509-chain-validation-design.md`

## Global Constraints

- Linux x86-64 only; trust store source is `/etc/ssl/certs/ca-certificates.crt` — no fallback path probing.
- Client role only — no certificate issuance, no server-side TLS concerns.
- Error convention: sentinel `i64` return values with named `pub const` error codes — **never** `Result<T,E>`/`Option<T>`.
- No network I/O anywhere in this sub-project. OCSP stapling verification consumes a byte blob the caller already has — this code never performs its own OCSP HTTP fetch.
- No CRL support, no direct (non-stapled) OCSP queries. Revocation checking is **fail-soft**: an empty/zero-length stapled OCSP response is accepted (chain trusted on signature/validity/hostname alone); a present-but-invalid-or-revoked response fails the chain.
- No path length limit beyond each cert's own `basicConstraints` `pathLenConstraint` (`Certificate.path_len_constraint`, already parsed by existing `cert.sio`; confirmed sentinel value for "absent" is `-1`, `0` or greater means "present, this many CAs may follow").
- Hostname verification: exact match, or single-level leftmost wildcard (`*.example.com` matches `foo.example.com`, not `foo.bar.example.com`, not bare `example.com`) — not full RFC 6125.
- Every `Certificate` this code handles must be kept paired with the `RawBuf` it was parsed from (`x509_verify_signature(buf, cert, issuer_modulus, issuer_exponent)` re-hashes `cert.tbs_start..cert.tbs_start+cert.tbs_len` out of that specific buffer — verified directly against `stdlib/x509/cert.sio`'s `x509_verify_signature` body; using the wrong buffer produces silently wrong hash input, not a crash).
- Fixed-size arrays throughout, matching this stdlib's existing style — never dynamic growth: `TRUST_STORE_MAX_CERTS: i64 = 512`, `MAX_INTERMEDIATES: i64 = 8`, `CHAIN_MAX_DEPTH: i64 = 10`.
- Module imports use the bare form `use <filename>::{name|*}` — no `module` declaration.
- Test invocation: `bash scripts/run_sio_test_suite.sh --filter-prefix pem_` / `--filter-prefix x509_chain_` / `--filter-prefix ocsp_` from the repo root — **never** the whole-repo suite for this plan's checkpoints. Every test file starts with `//@ run-pass` or `//@ compile-fail`.
- Every adversarial test asserts the **exact** expected error code (e.g. `CHAIN_ERR_EXPIRED`), never just "non-zero" — matching `tests/run-pass/x509_adversarial.sio`'s existing pattern.
- Commit convention: Conventional Commits (`feat(x509): ...`, `test(x509): ...`, `fix(x509): ...`). **Never add "Co-Authored-By" or any AI-attribution line to any commit** — this repo's `CLAUDE.md` states this explicitly.
- Run every shell command as a plain foreground command — never a background/Monitor-style wait mechanism (this caused real stalls in prior work on this exact branch).
- No Rust, Python, or any non-Sounio code anywhere in this plan's deliverables.

---

## Task 1: `stdlib/x509/pem.sio` — PEM block decoder

**Files:**
- Create: `stdlib/x509/pem.sio`
- Test: `tests/run-pass/pem_decode_roundtrip.sio`

**Interfaces:**
- Produces: `pub const PEM_ERR_NO_BEGIN_MARKER: i64 = 0 - 1`, `pub const PEM_ERR_NO_END_MARKER: i64 = 0 - 2`, `pub const PEM_ERR_BASE64_DECODE: i64 = 0 - 3`, `pub fn pem_decode_next(buf: &RawBuf, buf_len: i64, start: i64) -> (RawBuf, i64, i64, i64) with IO` — returns `(der_bytes, der_len, next_search_offset, error_code)`. `error_code == 0` on success. When no `-----BEGIN` marker exists at or after `start`, this is **not** an error: return `(rawbuf_new(1), 0, 0 - 1, 0)` (a throwaway empty `RawBuf`, `der_len = 0`, `next_search_offset = -1` signals "no more blocks", `error_code = 0`).
- Task 2 (`trust_store.sio`) calls `pem_decode_next` in a loop, feeding `next_search_offset` back in as `start` until it returns `-1`, to walk every block in a multi-certificate PEM bundle.

`stdlib/encoding/base64.sio`'s `base64_decode(data: &[u8; 256], data_len: i32, out: &![u8; 256]) -> i32` only accepts a **fixed 256-byte input array** and requires `data_len` to be a multiple of 4 (it decodes in 4-character quads; `'='` padding is only handled inside the final quad of a call). A PEM certificate body is almost always longer than 256 base64 characters, wrapped at 64 characters per line. So `pem_decode_next` cannot call `base64_decode` once on the whole body — it must:
1. Strip newlines from the base64 body between the `BEGIN`/`END` markers, copying only base64 alphabet characters (and `=`) into a scratch buffer.
2. Call `base64_decode` repeatedly on **256-character chunks** of that scratch buffer (each chunk copied into a local `[u8; 256]` array — `base64_decode` takes a reference to a fixed array, not a `RawBuf`), appending each call's decoded bytes into the output `RawBuf` at the correct offset. The stripped base64 body's total length is always a multiple of 4 (that's an invariant of valid base64), and 256 is itself a multiple of 4, so the final, possibly-shorter chunk (`total_len mod 256`, which is 0 when `total_len` is itself a multiple of 256) is still safely a multiple of 4 and may legally contain trailing `'='` padding.

- [ ] **Step 1: Write the failing test using a real cert's PEM/DER forms**

```bash
mkdir -p /tmp/pem_test_fixtures
openssl req -x509 -newkey rsa:2048 -keyout /tmp/pem_test_fixtures/key.pem -out /tmp/pem_test_fixtures/cert.pem -days 1 -nodes -subj "/CN=pem-test"
openssl x509 -in /tmp/pem_test_fixtures/cert.pem -outform DER -out /tmp/pem_test_fixtures/cert.der
xxd -i /tmp/pem_test_fixtures/cert.der | head -5
wc -c /tmp/pem_test_fixtures/cert.der
cat /tmp/pem_test_fixtures/cert.pem
```

Take the DER bytes from `xxd -i`'s output and the full PEM text from `cat`, and hand-write them as byte-array literals into the test file (this repo's established pattern for embedding real openssl-generated fixtures directly in `.sio` test source — see `tests/run-pass/x509_real_certificate.sio` from PR #2133 for the exact style to copy).

```sio
//@ run-pass
use x509::pem::*
use net::socket::*

fn main() -> i64 with IO {
    // PEM text below is the literal output of:
    //   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 1 -nodes -subj "/CN=pem-test"
    // copied verbatim, byte for byte (fill in from Step 1's `cat` output).
    let pem_text: [u8; 2048] = [ /* ... fill in ASCII byte values, one per PEM character, from `cat cert.pem` ... */ ]
    let pem_len: i64 = 0 // fill in: exact byte count of the PEM text above

    let pem_buf = rawbuf_new(pem_len + 8)
    var i: i64 = 0
    while i < pem_len {
        rawbuf_set(&pem_buf, i, pem_text[i as usize] as i64)
        i = i + 1
    }

    let (der, der_len, next, err) = pem_decode_next(&pem_buf, pem_len, 0)
    if err != 0 {
        print_int(1)
        return 1
    }

    // Expected DER length and first/last bytes below come from Step 1's
    // `wc -c cert.der` and `xxd -i cert.der` output -- fill in the real values.
    let expected_der_len: i64 = 0 // fill in from `wc -c`
    if der_len != expected_der_len {
        print_int(2)
        return 1
    }
    // A DER SEQUENCE always starts with tag byte 0x30.
    if rawbuf_get(&der, 0) != 0x30 {
        print_int(3)
        return 1
    }

    if next != 0 - 1 {
        print_int(4)
        return 1
    }

    println("pem_decode_roundtrip: PEM->DER matches openssl's own DER output")
    return 0
}
```

- [ ] **Step 2: Run test to verify it fails (module doesn't exist yet)**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/pem_decode_roundtrip.sio`
Expected: a "module not found" / unresolved-import style error referencing `x509::pem`.

- [ ] **Step 3: Implement `pem_decode_next`**

```sio
use net::socket::*
use encoding::base64::*

pub const PEM_ERR_NO_BEGIN_MARKER: i64 = 0 - 1
pub const PEM_ERR_NO_END_MARKER: i64 = 0 - 2
pub const PEM_ERR_BASE64_DECODE: i64 = 0 - 3

const PEM_BEGIN_MARKER_LEN: i64 = 27  // len("-----BEGIN CERTIFICATE-----")-1; recompute exactly in Step 3 body, do not trust this comment blindly
const PEM_END_MARKER_LEN: i64 = 25    // len("-----END CERTIFICATE-----")-1; same caveat

fn pem_is_base64_char(b: i64) -> bool {
    (b >= 65 && b <= 90) || (b >= 97 && b <= 122) || (b >= 48 && b <= 57) || b == 43 || b == 47 || b == 61
    // A-Z            a-z              0-9              '+'      '/'      '='
}

// Scans buf[from..buf_len) for the literal marker byte sequence; returns the
// absolute offset of the first byte AFTER the marker, or -1 if not found
// before buf_len. Compares byte-by-byte via rawbuf_get (no substring builtin
// exists in this stdlib for RawBuf).
fn pem_find_marker(buf: &RawBuf, buf_len: i64, from: i64, marker: &[u8; 32], marker_len: i64) -> i64 with IO {
    var i: i64 = from
    while i + marker_len <= buf_len {
        var matched: bool = true
        var j: i64 = 0
        while j < marker_len {
            if rawbuf_get(buf, i + j) != (marker[j as usize] as i64) {
                matched = false
                j = marker_len // break
            } else {
                j = j + 1
            }
        }
        if matched {
            return i + marker_len
        }
        i = i + 1
    }
    0 - 1
}

pub fn pem_decode_next(buf: &RawBuf, buf_len: i64, start: i64) -> (RawBuf, i64, i64, i64) with IO {
    let begin_marker: [u8; 32] = [
        45,45,45,45,45,66,69,71,73,78,32,67,69,82,84,73,70,73,67,65,84,69,45,45,45,45,45,
        0,0,0,0,0
    ] // "-----BEGIN CERTIFICATE-----" followed by zero padding to fill the array
    let begin_marker_len: i64 = 27
    let end_marker: [u8; 32] = [
        45,45,45,45,45,69,78,68,32,67,69,82,84,73,70,73,67,65,84,69,45,45,45,45,45,
        0,0,0,0,0,0,0
    ] // "-----END CERTIFICATE-----" followed by zero padding
    let end_marker_len: i64 = 25

    let body_start = pem_find_marker(buf, buf_len, start, &begin_marker, begin_marker_len)
    if body_start == (0 - 1) {
        return (rawbuf_new(1), 0, 0 - 1, 0)  // no more blocks -- not an error
    }
    let body_end_marker_pos = pem_find_marker(buf, buf_len, body_start, &end_marker, end_marker_len)
    if body_end_marker_pos == (0 - 1) {
        return (rawbuf_new(1), 0, 0, PEM_ERR_NO_END_MARKER)
    }
    let body_end = body_end_marker_pos - end_marker_len  // position of '-' that starts the END marker

    // Pass 1: strip non-base64-alphabet bytes (newlines, carriage returns)
    // from body_start..body_end into a scratch RawBuf, tracking its length.
    let scratch = rawbuf_new(body_end - body_start + 8)
    var scratch_len: i64 = 0
    var p: i64 = body_start
    while p < body_end {
        let b = rawbuf_get(buf, p)
        if pem_is_base64_char(b) {
            rawbuf_set(&scratch, scratch_len, b)
            scratch_len = scratch_len + 1
        }
        p = p + 1
    }

    // Pass 2: decode scratch in 256-byte chunks (base64_decode's own fixed-
    // array ceiling), each chunk's length a multiple of 4 by construction
    // (see this task's Interfaces note).
    let out = rawbuf_new((scratch_len / 4) * 3 + 8)
    var out_len: i64 = 0
    var chunk_start: i64 = 0
    while chunk_start < scratch_len {
        var chunk_len: i64 = scratch_len - chunk_start
        if chunk_len > 256 { chunk_len = 256 }

        var chunk_in: [u8; 256] = [0; 256]
        var k: i64 = 0
        while k < chunk_len {
            chunk_in[k as usize] = rawbuf_get(&scratch, chunk_start + k) as u8
            k = k + 1
        }
        var chunk_out: [u8; 256] = [0; 256]
        let decoded = base64_decode(&chunk_in, chunk_len as i32, &!chunk_out)
        if decoded < 0 {
            return (rawbuf_new(1), 0, 0, PEM_ERR_BASE64_DECODE)
        }
        var m: i64 = 0
        while m < (decoded as i64) {
            rawbuf_set(&out, out_len + m, chunk_out[m as usize] as i64)
            m = m + 1
        }
        out_len = out_len + (decoded as i64)
        chunk_start = chunk_start + chunk_len
    }

    (out, out_len, body_end_marker_pos, 0)
}
```

- [ ] **Step 4: Fill in the test's literal byte arrays from Step 1's real fixtures, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/pem_decode_roundtrip.sio`
Expected: prints `pem_decode_roundtrip: PEM->DER matches openssl's own DER output`, exit 0.

- [ ] **Step 5: Add a second test for multi-block bundles (what `trust_store.sio` will actually feed it)**

Reuse the same cert's PEM text twice, concatenated, as `tests/run-pass/pem_decode_multi_block.sio`: call `pem_decode_next` once with `start=0`, assert its `next_search_offset` is not `-1`, call it again with that offset as `start`, assert the second call's `der_len`/first-byte match the same expected values as Step 1, and assert the *third* call (using the second call's `next_search_offset`) returns `next_search_offset == 0 - 1`.

- [ ] **Step 6: Run both tests together**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix pem_ --verbose`
Expected: 2 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add stdlib/x509/pem.sio tests/run-pass/pem_decode_roundtrip.sio tests/run-pass/pem_decode_multi_block.sio
git commit -m "feat(x509): add PEM block decoder"
```

---

## Task 2: `stdlib/x509/trust_store.sio` — system CA bundle loader

**Files:**
- Create: `stdlib/x509/trust_store.sio`
- Test: `tests/run-pass/x509_trust_store_load.sio`

**Interfaces:**
- Consumes: `pem_decode_next` (Task 1), `x509_parse_certificate(buf: &RawBuf, len: i64) -> (Certificate, i64)` (existing, `stdlib/x509/cert.sio`).
- Produces: `pub const TRUST_STORE_MAX_CERTS: i64 = 512`, `pub const TRUST_STORE_ERR_READ_FAILED: i64 = 0 - 1`, `pub const TRUST_STORE_ERR_TOO_MANY_CERTS: i64 = 0 - 2`, `pub struct TrustStore { certs: [Certificate; 512], bufs: [RawBuf; 512], count: i32 }`, `pub fn trust_store_load() -> (TrustStore, i64) with IO`, `pub fn trust_store_find_by_subject(store: &TrustStore, subject: &X509Name) -> i64` (returns an index into `store.certs`, or `-1`), `pub fn trust_store_find_by_ski(store: &TrustStore, ski: &[u8; 32], ski_len: i32) -> i64` (same), `pub fn x509_name_eq(a: &X509Name, b: &X509Name) -> bool`. Tasks 3 and 4 import and use every one of these names/signatures verbatim.

- [ ] **Step 1: Verify `read_file`'s exact callable signature under Madaros before writing anything else**

`read_file`'s signature is inconsistent across this repository's history: the bootstrap-era compilers (`bootstrap/boot2.sio`, `boot3.sio`, `boot4.sio`) call it as a 1-argument builtin, `let data = read_file(path)`, returning a buffer directly; but `self-hosted/native/test_compile_to_elf.sio` and `self-hosted/native/codegen.sio` (Madaros's own compiler source) document it as a 3-argument builtin stub, `read_file(path, buf, max_len) -> i64`. These are different compiler generations — only the second one describes what Madaros itself recognizes. Verify directly:

```sio
// /tmp/read_file_spike.sio
fn main() -> i64 with IO {
    let n = read_file("/etc/hostname")
    print_int(n)
    return 0
}
```

Run: `./bin/souc check /tmp/read_file_spike.sio`

If that fails to typecheck (likely, since Madaros's codegen comments describe a 3-arg form), try:

```sio
extern "C" {
    fn heap_alloc(n: i64) -> *mut u8;
}
fn main() -> i64 with IO {
    let p = heap_alloc(65536)
    let n = read_file("/etc/hostname", p, 65536)
    print_int(n)
    return 0
}
```

Run: `./bin/souc run /tmp/read_file_spike.sio`
Expected: prints a small positive number (the byte length of `/etc/hostname`'s content) and exits 0. Record whichever form actually compiles and runs correctly as a comment at the top of `trust_store.sio`, and use that exact signature in Step 3 below — **do not assume the 3-arg form sketched there is correct until this step confirms it**; if the real signature differs (e.g. a `RawBuf` instead of a raw pointer for `buf`, or a different return convention for "file larger than buffer" / "file not found"), adjust Step 3's code to match what you actually observed, and note the discrepancy in the task's completion report.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
use x509::trust_store::*

fn main() -> i64 with IO {
    let (store, err) = trust_store_load()
    if err != 0 {
        print_int(1)
        return 1
    }
    // The real Debian/Ubuntu bundle has well over 50 root CAs as of 2026;
    // this is a loose sanity floor, not an exact count (the bundle changes
    // over time as CAs are added/removed upstream).
    if store.count < 50 {
        print_int(2)
        return 1
    }
    println("x509_trust_store_load: system CA bundle loaded")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/x509_trust_store_load.sio`
Expected: unresolved-import error referencing `x509::trust_store`.

- [ ] **Step 4: Implement `trust_store.sio`**

```sio
use net::socket::*
use x509::pem::*
use x509::cert::*

pub const TRUST_STORE_MAX_CERTS: i64 = 512
pub const TRUST_STORE_ERR_READ_FAILED: i64 = 0 - 1
pub const TRUST_STORE_ERR_TOO_MANY_CERTS: i64 = 0 - 2

pub struct TrustStore {
    certs: [Certificate; 512],
    bufs: [RawBuf; 512],
    count: i32,
}

pub fn x509_name_eq(a: &X509Name, b: &X509Name) -> bool {
    if a.count != b.count { return false }
    var i: i32 = 0
    while i < a.count {
        let ea = a.entries[i as usize]
        let eb = b.entries[i as usize]
        if ea.oid_len != eb.oid_len { return false }
        if ea.value_len != eb.value_len { return false }
        var j: i32 = 0
        while j < ea.oid_len {
            if ea.oid[j as usize] != eb.oid[j as usize] { return false }
            j = j + 1
        }
        var k: i32 = 0
        while k < ea.value_len {
            if ea.value[k as usize] != eb.value[k as usize] { return false }
            k = k + 1
        }
        i = i + 1
    }
    true
}

pub fn trust_store_find_by_subject(store: &TrustStore, subject: &X509Name) -> i64 {
    var i: i64 = 0
    while i < (store.count as i64) {
        if x509_name_eq(&store.certs[i as usize].subject, subject) {
            return i
        }
        i = i + 1
    }
    0 - 1
}

pub fn trust_store_find_by_ski(store: &TrustStore, ski: &[u8; 32], ski_len: i32) -> i64 {
    var i: i64 = 0
    while i < (store.count as i64) {
        let c = store.certs[i as usize]
        if c.subject_key_id_len == ski_len {
            var matched: bool = true
            var j: i32 = 0
            while j < ski_len {
                if c.subject_key_id[j as usize] != ski[j as usize] { matched = false }
                j = j + 1
            }
            if matched { return i }
        }
        i = i + 1
    }
    0 - 1
}

// PEM bundle size: fill in from Step 1's confirmed read_file signature --
// this uses the 3-arg (path, buf, max_len) -> i64 form pending that
// confirmation. 4MB comfortably covers the real Debian/Ubuntu bundle
// (measured well under 1MB as of 2026) with generous headroom.
const CA_BUNDLE_PATH: string = "/etc/ssl/certs/ca-certificates.crt"
const CA_BUNDLE_MAX_BYTES: i64 = 4194304

extern "C" {
    fn heap_alloc(n: i64) -> *mut u8;
}

pub fn trust_store_load() -> (TrustStore, i64) with IO {
    var store = TrustStore {
        certs: [certificate_zero(); 512],
        bufs: [rawbuf_new(1); 512],
        count: 0,
    }

    let raw_ptr = heap_alloc(CA_BUNDLE_MAX_BYTES)
    let bytes_read = read_file(CA_BUNDLE_PATH, raw_ptr, CA_BUNDLE_MAX_BYTES)
    if bytes_read <= 0 {
        return (store, TRUST_STORE_ERR_READ_FAILED)
    }

    let bundle = RawBuf { ptr: raw_ptr, cap: CA_BUNDLE_MAX_BYTES }

    var offset: i64 = 0
    while offset >= 0 {
        let (der, der_len, next, perr) = pem_decode_next(&bundle, bytes_read, offset)
        if perr != 0 {
            // A malformed block inside the system bundle is skipped, not
            // fatal -- the bundle is large and this loader's job is best-
            // effort coverage, matching how browsers treat OS trust stores.
            offset = next
        } else if der_len > 0 {
            if (store.count as i64) >= TRUST_STORE_MAX_CERTS {
                return (store, TRUST_STORE_ERR_TOO_MANY_CERTS)
            }
            let (cert, cerr) = x509_parse_certificate(&der, der_len)
            if cerr == 0 {
                store.certs[store.count as usize] = cert
                store.bufs[store.count as usize] = der
                store.count = store.count + 1
            }
            offset = next
        } else {
            offset = next  // -1: no more blocks, loop condition ends it
        }
    }

    (store, 0)
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_trust_store_load.sio`
Expected: prints `x509_trust_store_load: system CA bundle loaded`, exit 0. (This test genuinely reads the real system file — it is expected to run in this repo's own dev/CI environment, which has a standard `ca-certificates` package installed; if `store.count < 50` fails in some environment, that environment is missing the package, not a code defect — note this explicitly in the task report rather than loosening the threshold to mask it.)

- [ ] **Step 6: Add a `trust_store_find_by_subject`/`_by_ski` unit test**

`tests/run-pass/x509_trust_store_find.sio`: load the real store, take `store.certs[0].subject` (whatever the first real root happens to be — the test doesn't hardcode which CA it is), and assert `trust_store_find_by_subject(&store, &store.certs[0].subject) == 0`. Assert `trust_store_find_by_subject` against a X509Name with `count: 0` (an obviously-absent name) returns `-1`.

- [ ] **Step 7: Run both trust-store tests**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_trust_store_ --verbose`
Expected: 2 passed, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add stdlib/x509/trust_store.sio tests/run-pass/x509_trust_store_load.sio tests/run-pass/x509_trust_store_find.sio
git commit -m "feat(x509): add system CA trust store loader"
```

---

## Task 3: `stdlib/x509/chain.sio` (part 1) — path building

**Files:**
- Create: `stdlib/x509/chain.sio`
- Test: `tests/run-pass/x509_chain_path_building.sio`

**Interfaces:**
- Consumes: `TrustStore`, `x509_name_eq`, `trust_store_find_by_subject`, `trust_store_find_by_ski` (Task 2); `Certificate`, `X509Name` (existing `cert.sio`); `RawBuf` (existing `net::socket`).
- Produces (this task; file-private, no `pub`, but Task 4 in the *same file* relies on these exact names/signatures): `const CHAIN_MAX_DEPTH: i64 = 10`, `const MAX_INTERMEDIATES: i64 = 8`, `struct ChainCandidate { certs: [Certificate; 11], bufs: [RawBuf; 11], len: i32 }` (11 = `CHAIN_MAX_DEPTH + 1`, room for the leaf plus up to `CHAIN_MAX_DEPTH` issuers), `fn chain_build_candidates(leaf: &Certificate, leaf_buf: &RawBuf, intermediates: &[Certificate; 8], intermediate_bufs: &[RawBuf; 8], intermediate_count: i32, trust_store: &TrustStore) -> ([ChainCandidate; 16], i32) with IO` (up to 16 candidate paths; the cap exists to bound DFS output on adversarial/degenerate inputs, and is generous for any real-world chain shape).

This task's `path_len_constraint` sentinel check (used later, in Task 4, not here) must be independently confirmed rather than assumed: `grep -n "path_len_constraint" stdlib/x509/cert.sio` shows it's initialized as `var path_len_constraint: i32 = 0 - 1` and only overwritten when a `basicConstraints` extension with an explicit `pathLenConstraint` INTEGER is present — so `-1` means absent (no limit from this field) and any value `>= 0` is a real, present constraint. Confirm this still holds by re-running that grep before Task 4 relies on it; this task does not need the constraint value, only path building's candidate search.

- [ ] **Step 1: Write the failing test against a real 3-level chain**

```bash
mkdir -p /tmp/chain_test_fixtures && cd /tmp/chain_test_fixtures

# Root CA
openssl genrsa -out root.key 2048
openssl req -x509 -new -key root.key -days 3650 -out root.pem -subj "/CN=Test Root CA" -addext "basicConstraints=critical,CA:true"

# Intermediate CA, signed by root
openssl genrsa -out inter.key 2048
openssl req -new -key inter.key -out inter.csr -subj "/CN=Test Intermediate CA"
openssl x509 -req -in inter.csr -CA root.pem -CAkey root.key -CAcreateserial -days 1825 -out inter.pem -extfile <(echo "basicConstraints=critical,CA:true,pathlen:0")

# Leaf, signed by intermediate
openssl genrsa -out leaf.key 2048
openssl req -new -key leaf.key -out leaf.csr -subj "/CN=chain-test.example"
openssl x509 -req -in leaf.csr -CA inter.pem -CAkey inter.key -CAcreateserial -days 365 -out leaf.pem -extfile <(echo -e "basicConstraints=CA:false\nsubjectAltName=DNS:chain-test.example")

openssl x509 -in root.pem -outform DER -out root.der
openssl x509 -in inter.pem -outform DER -out inter.der
openssl x509 -in leaf.pem -outform DER -out leaf.der
xxd -i root.der; wc -c root.der
xxd -i inter.der; wc -c inter.der
xxd -i leaf.der; wc -c leaf.der
```

Copy each `.der`'s byte array (from `xxd -i`) and length (from `wc -c`) into the test file as literal `[u8; N]` arrays, following `tests/run-pass/x509_real_certificate.sio`'s established embedding style.

```sio
//@ run-pass
use x509::chain::*
use x509::trust_store::*
use x509::cert::*
use net::socket::*

fn main() -> i64 with IO {
    // DER bytes for root.der / inter.der / leaf.der -- fill in from Step 1.
    let root_der_bytes: [u8; 2048] = [ /* ... */ ]
    let root_der_len: i64 = 0 // fill in
    let inter_der_bytes: [u8; 2048] = [ /* ... */ ]
    let inter_der_len: i64 = 0 // fill in
    let leaf_der_bytes: [u8; 2048] = [ /* ... */ ]
    let leaf_der_len: i64 = 0 // fill in

    let root_buf = rawbuf_new(root_der_len + 8)
    var i: i64 = 0
    while i < root_der_len { rawbuf_set(&root_buf, i, root_der_bytes[i as usize] as i64); i = i + 1 }
    let inter_buf = rawbuf_new(inter_der_len + 8)
    var j: i64 = 0
    while j < inter_der_len { rawbuf_set(&inter_buf, j, inter_der_bytes[j as usize] as i64); j = j + 1 }
    let leaf_buf = rawbuf_new(leaf_der_len + 8)
    var k: i64 = 0
    while k < leaf_der_len { rawbuf_set(&leaf_buf, k, leaf_der_bytes[k as usize] as i64); k = k + 1 }

    let (root_cert, e1) = x509_parse_certificate(&root_buf, root_der_len)
    if e1 != 0 { print_int(1); return 1 }
    let (inter_cert, e2) = x509_parse_certificate(&inter_buf, inter_der_len)
    if e2 != 0 { print_int(2); return 1 }
    let (leaf_cert, e3) = x509_parse_certificate(&leaf_buf, leaf_der_len)
    if e3 != 0 { print_int(3); return 1 }

    var store = TrustStore { certs: [certificate_zero(); 512], bufs: [rawbuf_new(1); 512], count: 1 }
    store.certs[0] = root_cert
    store.bufs[0] = root_buf

    var intermediates: [Certificate; 8] = [certificate_zero(); 8]
    var intermediate_bufs: [RawBuf; 8] = [rawbuf_new(1); 8]
    intermediates[0] = inter_cert
    intermediate_bufs[0] = inter_buf

    let (candidates, count) = chain_build_candidates(&leaf_cert, &leaf_buf, &intermediates, &intermediate_bufs, 1, &store)
    if count < 1 {
        print_int(4)
        return 1
    }
    // At least one candidate path must reach the root: leaf -> intermediate -> root.
    var found_full_path: bool = false
    var c: i32 = 0
    while c < count {
        if candidates[c as usize].len == 3 {
            found_full_path = true
        }
        c = c + 1
    }
    if !found_full_path {
        print_int(5)
        return 1
    }

    println("x509_chain_path_building: found leaf->intermediate->root candidate")
    return 0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/x509_chain_path_building.sio`
Expected: unresolved-import error referencing `x509::chain`.

- [ ] **Step 3: Implement `chain_build_candidates`**

```sio
use net::socket::*
use x509::cert::*
use x509::trust_store::*

pub const CHAIN_MAX_DEPTH: i64 = 10
pub const MAX_INTERMEDIATES: i64 = 8
const MAX_CHAIN_CANDIDATES: i64 = 16

struct ChainCandidate {
    certs: [Certificate; 11],  // CHAIN_MAX_DEPTH + 1
    bufs: [RawBuf; 11],
    len: i32,
}

// Returns true and the found index/buf if `cert` (identified by AKI when
// present, else by issuer DN) has a matching issuer among `pool_certs`.
// `pool_certs`/`pool_bufs` covers intermediates ++ trust_store.certs in one
// flat, index-aligned pair of arrays -- callers build that pool once, not
// per recursive call.
fn chain_find_issuer_candidates(
    cert: &Certificate,
    pool_certs: &[Certificate; 520],   // MAX_INTERMEDIATES + TRUST_STORE_MAX_CERTS
    pool_bufs: &[RawBuf; 520],
    pool_len: i64,
    exclude_index: &[i64; 11],
    exclude_len: i32,
    out_indices: &![i64; 520],
) -> i32 {
    var out_count: i32 = 0
    var i: i64 = 0
    while i < pool_len {
        var already_used: bool = false
        var e: i32 = 0
        while e < exclude_len {
            if exclude_index[e as usize] == i { already_used = true }
            e = e + 1
        }
        if !already_used {
            let candidate = pool_certs[i as usize]
            var matches: bool = false
            if cert.authority_key_id_len > 0 && candidate.subject_key_id_len > 0 {
                if cert.authority_key_id_len == candidate.subject_key_id_len {
                    var same: bool = true
                    var b: i32 = 0
                    while b < cert.authority_key_id_len {
                        if cert.authority_key_id[b as usize] != candidate.subject_key_id[b as usize] { same = false }
                        b = b + 1
                    }
                    matches = same
                }
            } else {
                matches = x509_name_eq(&cert.issuer, &candidate.subject)
            }
            if matches {
                out_indices[out_count as usize] = i
                out_count = out_count + 1
            }
        }
        i = i + 1
    }
    out_count
}

fn chain_is_trusted_root(cert: &Certificate, trust_store: &TrustStore) -> bool {
    if trust_store_find_by_subject(trust_store, &cert.subject) >= 0 {
        return x509_name_eq(&cert.subject, &cert.issuer)
    }
    false
}

pub fn chain_build_candidates(
    leaf: &Certificate,
    leaf_buf: &RawBuf,
    intermediates: &[Certificate; 8],
    intermediate_bufs: &[RawBuf; 8],
    intermediate_count: i32,
    trust_store: &TrustStore,
) -> ([ChainCandidate; 16], i32) with IO {
    var results: [ChainCandidate; 16] = [
        ChainCandidate { certs: [certificate_zero(); 11], bufs: [rawbuf_new(1); 11], len: 0 };
        16
    ]
    var result_count: i32 = 0

    var pool_certs: [Certificate; 520] = [certificate_zero(); 520]
    var pool_bufs: [RawBuf; 520] = [rawbuf_new(1); 520]
    var pool_len: i64 = 0
    var pi: i64 = 0
    while pi < (intermediate_count as i64) {
        pool_certs[pool_len as usize] = intermediates[pi as usize]
        pool_bufs[pool_len as usize] = intermediate_bufs[pi as usize]
        pool_len = pool_len + 1
        pi = pi + 1
    }
    var ti: i64 = 0
    while ti < (trust_store.count as i64) {
        pool_certs[pool_len as usize] = trust_store.certs[ti as usize]
        pool_bufs[pool_len as usize] = trust_store.bufs[ti as usize]
        pool_len = pool_len + 1
        ti = ti + 1
    }

    // Explicit-stack DFS (no recursion needed): each stack frame is a path
    // prefix plus the list of pool indices already tried as its next issuer.
    var path_certs: [Certificate; 11] = [certificate_zero(); 11]
    var path_bufs: [RawBuf; 11] = [rawbuf_new(1); 11]
    var path_pool_index: [i64; 11] = [0 - 1; 11]  // pool index chosen at each depth (>=1); depth 0 is the leaf, has no pool index
    var depth: i64 = 0
    path_certs[0] = *leaf
    path_bufs[0] = *leaf_buf
    depth = 1

    // `tried[d]` = how many of chain_find_issuer_candidates' results at
    // depth d have already been attempted (drives backtracking: when a
    // depth's candidates are exhausted, pop back to depth-1 and advance
    // ITS tried-count instead).
    var tried: [i32; 11] = [0; 11]

    while depth >= 1 && depth <= CHAIN_MAX_DEPTH && (result_count as i64) < MAX_CHAIN_CANDIDATES {
        let current = path_certs[(depth - 1) as usize]
        if chain_is_trusted_root(&current, trust_store) && depth > 1 {
            var rc = ChainCandidate { certs: [certificate_zero(); 11], bufs: [rawbuf_new(1); 11], len: depth as i32 }
            var ci: i64 = 0
            while ci < depth {
                rc.certs[ci as usize] = path_certs[ci as usize]
                rc.bufs[ci as usize] = path_bufs[ci as usize]
                ci = ci + 1
            }
            results[result_count as usize] = rc
            result_count = result_count + 1
            // Backtrack after recording: a trusted root is a leaf of the
            // search tree, never itself an issuer we recurse through.
            depth = depth - 1
        } else {
            var candidate_indices: [i64; 520] = [0; 520]
            let n = chain_find_issuer_candidates(&current, &pool_certs, &pool_bufs, pool_len, &path_pool_index, (depth - 1) as i32, &!candidate_indices)
            if (tried[depth as usize] as i64) < (n as i64) {
                let chosen = candidate_indices[tried[depth as usize] as usize]
                tried[depth as usize] = tried[depth as usize] + 1
                path_pool_index[depth as usize] = chosen
                path_certs[depth as usize] = pool_certs[chosen as usize]
                path_bufs[depth as usize] = pool_bufs[chosen as usize]
                depth = depth + 1
                tried[depth as usize] = 0
            } else {
                // Exhausted this depth's candidates -- backtrack.
                tried[depth as usize] = 0
                depth = depth - 1
            }
        }
    }

    (results, result_count)
}
```

- [ ] **Step 4: Fill in the test's real DER byte arrays from Step 1, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_chain_path_building.sio`
Expected: prints `x509_chain_path_building: found leaf->intermediate->root candidate`, exit 0.

- [ ] **Step 5: Add the reversed-intermediate-order test**

`tests/run-pass/x509_chain_path_building_reordered.sio`: identical to Step 1's test, but swap which intermediate slot the (single, in this 3-level case, trivially "reordering" is a no-op with only one intermediate — extend the fixture to a 4-level chain: root → inter1 → inter2 → leaf, generated the same way as Step 1 but with an extra intermediate) and pass `intermediates = [inter2_cert, inter1_cert]` (deliberately out of the leaf-to-root order) into `chain_build_candidates`. Assert a candidate of `len == 4` is still found — this is the generic-path-building behavior the spec explicitly requires over a naive ordered-linear-scan implementation.

- [ ] **Step 6: Run both path-building tests**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_chain_path_building --verbose`
Expected: 2 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add stdlib/x509/chain.sio tests/run-pass/x509_chain_path_building.sio tests/run-pass/x509_chain_path_building_reordered.sio
git commit -m "feat(x509): add generic certificate chain path building"
```

---

## Task 4: `stdlib/x509/chain.sio` (part 2) — path verification, hostname check, public entry point

**Files:**
- Modify: `stdlib/x509/chain.sio` (append to the file Task 3 created)
- Test: `tests/run-pass/x509_chain_verify_positive.sio`, `tests/run-pass/x509_chain_adversarial.sio`

**Interfaces:**
- Consumes: everything Task 3 produced in `chain.sio`, plus `x509_verify_signature` (existing `cert.sio`).
- Produces (public — this is the interface the not-yet-planned TLS handshake sub-project will call): `pub const CHAIN_OK: i64 = 0`, `pub const CHAIN_ERR_NO_PATH_TO_ROOT: i64 = 0 - 1`, `pub const CHAIN_ERR_EXPIRED: i64 = 0 - 2`, `pub const CHAIN_ERR_NOT_YET_VALID: i64 = 0 - 3`, `pub const CHAIN_ERR_NOT_CA: i64 = 0 - 4`, `pub const CHAIN_ERR_PATH_LEN_EXCEEDED: i64 = 0 - 5`, `pub const CHAIN_ERR_BAD_SIGNATURE: i64 = 0 - 6`, `pub const CHAIN_ERR_HOSTNAME_MISMATCH: i64 = 0 - 7`, `pub fn x509_verify_hostname(leaf: &Certificate, hostname: &RawBuf, hostname_len: i64) -> bool with IO`, `pub fn x509_verify_chain(leaf: &Certificate, leaf_buf: &RawBuf, intermediates: &[Certificate; 8], intermediate_bufs: &[RawBuf; 8], intermediate_count: i32, trust_store: &TrustStore, hostname: &RawBuf, hostname_len: i64, now_unix: i64) -> i64 with IO`. (`CHAIN_ERR_OCSP_REVOKED`/`CHAIN_ERR_OCSP_INVALID` and the `ocsp_response`/`ocsp_response_len` parameters are added in Task 6, once `ocsp.sio` exists — `x509_verify_chain`'s signature here is deliberately the pre-OCSP shape; Task 6 extends it, and callers written against this task's shape need a two-argument addition, not a rewrite.)

First, confirm the `path_len_constraint` sentinel per Task 3's note:

```bash
grep -n "path_len_constraint" stdlib/x509/cert.sio
```

Confirm the output still shows `var path_len_constraint: i32 = 0 - 1` as the only initializer and that it's only reassigned inside the `basicConstraints` branch — if this has changed since this plan was written, adjust every `>= 0` / `== 0 - 1` check below to match reality, and note the discrepancy in this task's completion report.

- [ ] **Step 1: Write the failing positive-case test (reuses Task 3's 3-level fixture)**

```sio
//@ run-pass
use x509::chain::*
use x509::trust_store::*
use x509::cert::*
use net::socket::*

fn main() -> i64 with IO {
    // Same root/inter/leaf DER byte arrays as tests/run-pass/x509_chain_path_building.sio
    // -- copy them verbatim from that file rather than regenerating with openssl,
    // so both tests exercise byte-identical certificates.
    let root_der_bytes: [u8; 2048] = [ /* ... copy from x509_chain_path_building.sio ... */ ]
    let root_der_len: i64 = 0 // copy
    let inter_der_bytes: [u8; 2048] = [ /* ... */ ]
    let inter_der_len: i64 = 0
    let leaf_der_bytes: [u8; 2048] = [ /* ... */ ]
    let leaf_der_len: i64 = 0

    let root_buf = rawbuf_new(root_der_len + 8)
    var i: i64 = 0
    while i < root_der_len { rawbuf_set(&root_buf, i, root_der_bytes[i as usize] as i64); i = i + 1 }
    let inter_buf = rawbuf_new(inter_der_len + 8)
    var j: i64 = 0
    while j < inter_der_len { rawbuf_set(&inter_buf, j, inter_der_bytes[j as usize] as i64); j = j + 1 }
    let leaf_buf = rawbuf_new(leaf_der_len + 8)
    var k: i64 = 0
    while k < leaf_der_len { rawbuf_set(&leaf_buf, k, leaf_der_bytes[k as usize] as i64); k = k + 1 }

    let (root_cert, e1) = x509_parse_certificate(&root_buf, root_der_len)
    if e1 != 0 { print_int(1); return 1 }
    let (inter_cert, e2) = x509_parse_certificate(&inter_buf, inter_der_len)
    if e2 != 0 { print_int(2); return 1 }
    let (leaf_cert, e3) = x509_parse_certificate(&leaf_buf, leaf_der_len)
    if e3 != 0 { print_int(3); return 1 }

    var store = TrustStore { certs: [certificate_zero(); 512], bufs: [rawbuf_new(1); 512], count: 1 }
    store.certs[0] = root_cert
    store.bufs[0] = root_buf

    var intermediates: [Certificate; 8] = [certificate_zero(); 8]
    var intermediate_bufs: [RawBuf; 8] = [rawbuf_new(1); 8]
    intermediates[0] = inter_cert
    intermediate_bufs[0] = inter_buf

    let hostname_text: [u8; 32] = [99,104,97,105,110,45,116,101,115,116,46,101,120,97,109,112,108,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0] // "chain-test.example"
    let hostname_len: i64 = 19
    let hostname_buf = rawbuf_new(32)
    var h: i64 = 0
    while h < hostname_len { rawbuf_set(&hostname_buf, h, hostname_text[h as usize] as i64); h = h + 1 }

    // A fixed timestamp inside the leaf's validity window -- the leaf was
    // issued with `-days 365` from Task 3's Step 1, so "now" a few minutes
    // after generation is always valid; using unix_timestamp_from_ymdhms
    // (stdlib/x509/cert.sio, already exists) with a date shortly after this
    // test file was written avoids depending on wall-clock time at test-run.
    let now_unix: i64 = 1798761600 // 2026-12-01T00:00:00Z placeholder -- replace with a real `date -u +%s` value captured right after generating the fixtures in Task 3 Step 1, so it's guaranteed inside all three certs' validity windows

    let result = x509_verify_chain(&leaf_cert, &leaf_buf, &intermediates, &intermediate_bufs, 1, &store, &hostname_buf, hostname_len, now_unix)
    if result != CHAIN_OK {
        print_int(result)
        return 1
    }

    println("x509_chain_verify_positive: full chain verified")
    return 0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/x509_chain_verify_positive.sio`
Expected: unresolved-name error referencing `x509_verify_chain` / `CHAIN_OK` (not yet defined).

- [ ] **Step 3: Implement path verification, hostname check, and the public entry point — append to `stdlib/x509/chain.sio`**

```sio
use crypto::pkcs1::*
use x509::oid::*

pub const CHAIN_OK: i64 = 0
pub const CHAIN_ERR_NO_PATH_TO_ROOT: i64 = 0 - 1
pub const CHAIN_ERR_EXPIRED: i64 = 0 - 2
pub const CHAIN_ERR_NOT_YET_VALID: i64 = 0 - 3
pub const CHAIN_ERR_NOT_CA: i64 = 0 - 4
pub const CHAIN_ERR_PATH_LEN_EXCEEDED: i64 = 0 - 5
pub const CHAIN_ERR_BAD_SIGNATURE: i64 = 0 - 6
pub const CHAIN_ERR_HOSTNAME_MISMATCH: i64 = 0 - 7

const KEY_USAGE_KEY_CERT_SIGN_BYTE0: i32 = 4    // 0x04: keyCertSign when keyUsage is 1 usage-byte (the common case)
const KEY_USAGE_KEY_CERT_SIGN_BYTE0_SHIFTED: i32 = 1024  // 0x0400: same bit, when keyUsage is 2 usage-bytes (encipherOnly/decipherOnly present)

fn key_usage_allows_cert_sign(key_usage_bits: i32) -> bool {
    (key_usage_bits & KEY_USAGE_KEY_CERT_SIGN_BYTE0) != 0 || (key_usage_bits & KEY_USAGE_KEY_CERT_SIGN_BYTE0_SHIFTED) != 0
}

// Verifies one candidate path (path_certs[0] = leaf ... path_certs[len-1] =
// trusted root) fully. Returns (CHAIN_OK, depth_reached) on success, or
// (error_code, depth_reached) on the first failure -- depth_reached is how
// many links verified successfully before the failure, used by
// x509_verify_chain to pick the "most progress" candidate to report when
// every candidate fails for a different reason.
fn chain_verify_path(path_certs: &[Certificate; 11], path_bufs: &[RawBuf; 11], path_len: i32, now_unix: i64) -> (i64, i32) with IO {
    var i: i32 = 0
    while i < path_len {
        let cert = path_certs[i as usize]
        if now_unix < cert.not_before_unix {
            return (CHAIN_ERR_NOT_YET_VALID, i)
        }
        if now_unix > cert.not_after_unix {
            return (CHAIN_ERR_EXPIRED, i)
        }
        i = i + 1
    }

    var d: i32 = 0
    while d < path_len - 1 {
        let cert = path_certs[d as usize]
        let cert_buf = path_bufs[d as usize]
        let issuer = path_certs[(d + 1) as usize]

        if !x509_verify_signature(&cert_buf, &cert, &issuer.modulus, &issuer.public_exponent) {
            return (CHAIN_ERR_BAD_SIGNATURE, d)
        }

        if !issuer.is_ca {
            return (CHAIN_ERR_NOT_CA, d + 1)
        }
        if issuer.key_usage_bits != 0 && !key_usage_allows_cert_sign(issuer.key_usage_bits) {
            return (CHAIN_ERR_NOT_CA, d + 1)
        }
        if issuer.path_len_constraint >= 0 {
            // Number of CA certs strictly between `issuer` and the leaf:
            // path_certs[0..d] excludes the leaf itself (index 0) when d>0,
            // so the count of intermediates below `issuer` is exactly `d`.
            if (d as i64) > (issuer.path_len_constraint as i64) {
                return (CHAIN_ERR_PATH_LEN_EXCEEDED, d + 1)
            }
        }

        d = d + 1
    }

    (CHAIN_OK, path_len)
}

pub fn x509_verify_hostname(leaf: &Certificate, hostname: &RawBuf, hostname_len: i64) -> bool with IO {
    var i: i32 = 0
    while i < leaf.san_count {
        let san = leaf.san_entries[i as usize]
        if san.tag == GENERAL_NAME_DNS_NAME {
            if (san.value_len as i64) == hostname_len {
                var same: bool = true
                var j: i64 = 0
                while j < hostname_len {
                    if san.value[j as usize] != rawbuf_get(hostname, j) as u8 { same = false }
                    j = j + 1
                }
                if same { return true }
            } else if san.value_len >= 2 && san.value[0] == (42 as u8) && san.value[1] == (46 as u8) {
                // "*." prefix: match against the hostname's suffix starting
                // right after its own first '.' -- and require the hostname
                // to actually contain a '.' at all (a single-label hostname
                // can never match a wildcard).
                var dot_pos: i64 = 0 - 1
                var k: i64 = 0
                while k < hostname_len {
                    if (rawbuf_get(hostname, k) as u8) == (46 as u8) && dot_pos == (0 - 1) {
                        dot_pos = k
                    }
                    k = k + 1
                }
                if dot_pos >= 0 {
                    let suffix_len = hostname_len - dot_pos - 1
                    let pattern_suffix_len = (san.value_len as i64) - 2
                    if suffix_len == pattern_suffix_len {
                        var same2: bool = true
                        var m: i64 = 0
                        while m < suffix_len {
                            if san.value[(2 + m) as usize] != (rawbuf_get(hostname, dot_pos + 1 + m) as u8) { same2 = false }
                            m = m + 1
                        }
                        if same2 { return true }
                    }
                }
            }
        }
        i = i + 1
    }
    false
}

pub fn x509_verify_chain(
    leaf: &Certificate,
    leaf_buf: &RawBuf,
    intermediates: &[Certificate; 8],
    intermediate_bufs: &[RawBuf; 8],
    intermediate_count: i32,
    trust_store: &TrustStore,
    hostname: &RawBuf,
    hostname_len: i64,
    now_unix: i64,
) -> i64 with IO {
    let (candidates, candidate_count) = chain_build_candidates(leaf, leaf_buf, intermediates, intermediate_bufs, intermediate_count, trust_store)
    if candidate_count == 0 {
        return CHAIN_ERR_NO_PATH_TO_ROOT
    }

    var best_depth: i32 = 0 - 1
    var best_error: i64 = CHAIN_ERR_NO_PATH_TO_ROOT
    var c: i32 = 0
    while c < candidate_count {
        let cand = candidates[c as usize]
        let (result, depth_reached) = chain_verify_path(&cand.certs, &cand.bufs, cand.len, now_unix)
        if result == CHAIN_OK {
            if !x509_verify_hostname(leaf, hostname, hostname_len) {
                return CHAIN_ERR_HOSTNAME_MISMATCH
            }
            return CHAIN_OK
        }
        if depth_reached > best_depth {
            best_depth = depth_reached
            best_error = result
        }
        c = c + 1
    }

    best_error
}
```

- [ ] **Step 4: Fill in the test's DER arrays and `now_unix` from Task 3's Step 1 fixtures, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_chain_verify_positive.sio`
Expected: prints `x509_chain_verify_positive: full chain verified`, exit 0.

- [ ] **Step 5: Write the adversarial test suite**

```sio
//@ run-pass
// Reuses the same root/inter/leaf fixtures as x509_chain_verify_positive.sio.
// Each case below mutates exactly one property (expiry timestamp passed to
// x509_verify_chain, a byte in an intermediate's DER encoding, etc.) and
// asserts the EXACT expected CHAIN_ERR_* code, per this plan's Global
// Constraints ("never just non-zero").
use x509::chain::*
use x509::trust_store::*
use x509::cert::*
use net::socket::*

// ... (fill in the same setup as x509_chain_verify_positive.sio: parse
// root/inter/leaf, build store/intermediates/hostname_buf) ...

fn main() -> i64 with IO {
    // (setup identical to x509_chain_verify_positive.sio's Step 1 test --
    // copy it here rather than importing, since each case below needs its
    // own mutated copy of one input.)

    // Case (a): now_unix far in the future -> CHAIN_ERR_EXPIRED
    // let result_a = x509_verify_chain(..., now_unix: 4102444800 /* 2100-01-01 */)
    // if result_a != CHAIN_ERR_EXPIRED { print_int(1); return 1 }

    // Case (b): now_unix far in the past -> CHAIN_ERR_NOT_YET_VALID
    // if result_b != CHAIN_ERR_NOT_YET_VALID { print_int(2); return 1 }

    // Case (c): tamper one byte of leaf_buf's signature region (leaf.outer_signature
    // is the last ~256 bytes of a 2048-bit RSA cert's DER encoding -- flip
    // rawbuf_get/rawbuf_set at leaf_der_len - 10) -> CHAIN_ERR_BAD_SIGNATURE
    // if result_c != CHAIN_ERR_BAD_SIGNATURE { print_int(3); return 1 }

    // Case (d): wrong hostname ("not-chain-test.example") -> CHAIN_ERR_HOSTNAME_MISMATCH
    // if result_d != CHAIN_ERR_HOSTNAME_MISMATCH { print_int(4); return 1 }

    // Case (e): a 4-level fixture (root -> inter1[pathlen:0] -> inter2 -> leaf,
    // generated the same way as Task 3 Step 5's reordering fixture, but this
    // time inter1's pathlen:0 forbids any CA below it) -> CHAIN_ERR_PATH_LEN_EXCEEDED
    // if result_e != CHAIN_ERR_PATH_LEN_EXCEEDED { print_int(5); return 1 }

    // Case (f): an intermediate whose own cert was issued WITHOUT
    // "basicConstraints=CA:true" (regenerate inter.pem with
    // -extfile <(echo "basicConstraints=CA:false") for this one case only)
    // -> CHAIN_ERR_NOT_CA
    // if result_f != CHAIN_ERR_NOT_CA { print_int(6); return 1 }

    println("x509_chain_adversarial: all 6 cases (a-f) correctly rejected")
    return 0
}
```

Fill in every commented-out block above with real code (parse the mutated fixture, call `x509_verify_chain`, assert the exact error) before running — this plan's placeholder-scan self-review flags the comments as a structural outline for the implementer to complete, not a step to leave as-is; every `// if result_X ...` line must become live code with its own generated/mutated fixture.

- [ ] **Step 6: Run test to verify it fails, then implement each case, then run to verify it passes**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/x509_chain_adversarial.sio`
Expected (initially): fails on the first unimplemented case. After filling in all 6 cases: prints `x509_chain_adversarial: all 6 cases (a-f) correctly rejected`, exit 0.

- [ ] **Step 7: Run the full chain test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_chain_ --verbose`
Expected: all passed (path-building tests from Task 3 plus this task's positive + adversarial tests), 0 failed.

- [ ] **Step 8: Commit**

```bash
git add stdlib/x509/chain.sio tests/run-pass/x509_chain_verify_positive.sio tests/run-pass/x509_chain_adversarial.sio
git commit -m "feat(x509): add chain path verification, hostname check, and x509_verify_chain entry point"
```

---

## Task 5: `stdlib/x509/ocsp.sio` — OCSP stapled-response verification

**Files:**
- Create: `stdlib/x509/ocsp.sio`
- Test: `tests/run-pass/ocsp_verify_good.sio`, `tests/run-pass/ocsp_verify_adversarial.sio`

**Interfaces:**
- Consumes: `DerReader`, `DerTag`, `der_read_tlv`, `der_enter`, `DER_TAG_*`, `DER_OK` (existing `asn1::der`); `Certificate` (existing `x509::cert`); `x509_verify_signature`-style signature checking is **not** reused directly here (OCSP signs `tbsResponseData`, a different ASN.1 structure than a certificate's TBSCertificate) — this task hashes and PKCS#1-verifies its own `tbsResponseData` byte range using `crypto::pkcs1`'s primitives directly.
- Produces: `pub const OCSP_OK: i64 = 0`, `pub const OCSP_ERR_PARSE: i64 = 0 - 1`, `pub const OCSP_ERR_CERT_ID_MISMATCH: i64 = 0 - 2`, `pub const OCSP_ERR_BAD_SIGNATURE: i64 = 0 - 3`, `pub const OCSP_ERR_STALE: i64 = 0 - 4`, `pub const OCSP_STATUS_GOOD: i64 = 0`, `pub const OCSP_STATUS_REVOKED: i64 = 1`, `pub const OCSP_STATUS_UNKNOWN: i64 = 2`, `pub fn ocsp_verify_response(response: &RawBuf, response_len: i64, target: &Certificate, issuer: &Certificate, now_unix: i64) -> (i64, i64) with IO` (returns `(ocsp_status, error_code)`; `ocsp_status` only meaningful when `error_code == OCSP_OK`).

RFC 6960's `BasicOCSPResponse` structure, for reference while implementing (already-published, publicly documented ASN.1 — not invented for this plan):

```
OCSPResponse ::= SEQUENCE {
   responseStatus   ENUMERATED,
   responseBytes    [0] EXPLICIT ResponseBytes OPTIONAL }

ResponseBytes ::= SEQUENCE {
   responseType   OBJECT IDENTIFIER,
   response       OCTET STRING }   -- contains a DER-encoded BasicOCSPResponse

BasicOCSPResponse ::= SEQUENCE {
   tbsResponseData      ResponseData,
   signatureAlgorithm   AlgorithmIdentifier,
   signature            BIT STRING,
   certs            [0] EXPLICIT SEQUENCE OF Certificate OPTIONAL }

ResponseData ::= SEQUENCE {
   version              [0] EXPLICIT INTEGER DEFAULT v1,
   responderID              ResponderID,
   producedAt               GeneralizedTime,
   responses                SEQUENCE OF SingleResponse }

SingleResponse ::= SEQUENCE {
   certID                   CertID,
   certStatus               CertStatus,   -- CHOICE: good[0]/revoked[1]/unknown[2]
   thisUpdate               GeneralizedTime,
   nextUpdate           [0] EXPLICIT GeneralizedTime OPTIONAL }

CertID ::= SEQUENCE {
   hashAlgorithm    AlgorithmIdentifier,
   issuerNameHash   OCTET STRING,
   issuerKeyHash    OCTET STRING,
   serialNumber     INTEGER }
```

This plan's scope only reads the **first** `SingleResponse` in `responses` (a stapled OCSP response, by construction, only ever covers the one certificate the TLS server stapled it for — a server sending more than one `SingleResponse` in a staple is non-conformant and out of scope to handle specially). `certID` matching uses `serialNumber` only (comparing against `target.serial_number`, a `BigInt`) — `issuerNameHash`/`issuerKeyHash` re-verification against `issuer` is deliberately **not** implemented in this task (it would require replicating whichever hash algorithm `CertID.hashAlgorithm` names against `issuer`'s encoded name/key, adding real scope for a check that mostly guards against a confused-deputy scenario already prevented by this response only ever being handed to `ocsp_verify_response` alongside the specific `target`/`issuer` pair the chain-verification caller already trusts) — document this as a known limitation in the task's completion report, not a silent gap.

- [ ] **Step 1: Verify a real OCSP response's byte structure before writing the parser**

```bash
# Reuse Task 3/4's root/inter/leaf fixtures (still in /tmp/chain_test_fixtures).
cd /tmp/chain_test_fixtures
mkdir -p demoCA/newcerts
echo 01 > demoCA/serial
touch demoCA/index.txt
openssl ocsp -index demoCA/index.txt -CA inter.pem -rsigner inter.pem -rkey inter.key -port 8888 &
OCSP_PID=$!
sleep 1
openssl ocsp -issuer inter.pem -cert leaf.pem -url http://127.0.0.1:8888 -resp_text -respout ocsp_resp.der
kill $OCSP_PID
xxd -i ocsp_resp.der
wc -c ocsp_resp.der
openssl asn1parse -inform DER -in ocsp_resp.der
```

Cross-reference the `openssl asn1parse` output against the ASN.1 grammar above to confirm field offsets/tags line up with what the implementation below expects, before trusting the implementation against only the round-trip test's pass/fail signal.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass
use x509::ocsp::*
use x509::cert::*
use net::socket::*

fn main() -> i64 with IO {
    // OCSP response DER bytes from Step 1's ocsp_resp.der -- fill in.
    let resp_bytes: [u8; 2048] = [ /* ... */ ]
    let resp_len: i64 = 0 // fill in from `wc -c`

    // leaf/inter Certificate structs -- parse the same way as
    // x509_chain_verify_positive.sio (copy that setup here).
    // let (leaf_cert, _) = x509_parse_certificate(...)
    // let (inter_cert, _) = x509_parse_certificate(...)

    let resp_buf = rawbuf_new(resp_len + 8)
    var i: i64 = 0
    while i < resp_len { rawbuf_set(&resp_buf, i, resp_bytes[i as usize] as i64); i = i + 1 }

    let now_unix: i64 = 1798761600 // same placeholder convention as Task 4 -- replace with a real value captured right after Step 1's openssl ocsp run

    // let (status, err) = ocsp_verify_response(&resp_buf, resp_len, &leaf_cert, &inter_cert, now_unix)
    // if err != OCSP_OK { print_int(1); return 1 }
    // if status != OCSP_STATUS_GOOD { print_int(2); return 1 }

    println("ocsp_verify_good: stapled response verified as good")
    return 0
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check tests/run-pass/ocsp_verify_good.sio`
Expected: unresolved-import error referencing `x509::ocsp`.

- [ ] **Step 4: Implement `ocsp_verify_response`**

```sio
use asn1::der::*
use net::socket::*
use x509::cert::*
use crypto::pkcs1::*
use bignum::bigint::*

pub const OCSP_OK: i64 = 0
pub const OCSP_ERR_PARSE: i64 = 0 - 1
pub const OCSP_ERR_CERT_ID_MISMATCH: i64 = 0 - 2
pub const OCSP_ERR_BAD_SIGNATURE: i64 = 0 - 3
pub const OCSP_ERR_STALE: i64 = 0 - 4
pub const OCSP_STATUS_GOOD: i64 = 0
pub const OCSP_STATUS_REVOKED: i64 = 1
pub const OCSP_STATUS_UNKNOWN: i64 = 2

pub fn ocsp_verify_response(response: &RawBuf, response_len: i64, target: &Certificate, issuer: &Certificate, now_unix: i64) -> (i64, i64) with IO {
    var r = DerReader { buf_ptr: response.ptr, pos: 0, end: response_len }

    // OCSPResponse ::= SEQUENCE { responseStatus ENUMERATED, responseBytes [0] EXPLICIT ResponseBytes OPTIONAL }
    let (r1, outer_tag, s1) = der_read_tlv(&r)
    if s1 != DER_OK || outer_tag.tag_number != DER_TAG_SEQUENCE { return (0, OCSP_ERR_PARSE) }
    let (mut_r1, e1) = der_enter(&r, &outer_tag)
    if e1 != DER_OK { return (0, OCSP_ERR_PARSE) }
    var inner = mut_r1

    let (inner2, status_tag, s2) = der_read_tlv(&inner)
    if s2 != DER_OK { return (0, OCSP_ERR_PARSE) }
    inner = inner2
    // responseStatus must be 0 (successful) -- any other value has no
    // responseBytes to parse at all.
    let status_byte = rawbuf_get(response, status_tag.content_start)
    if status_byte != 0 { return (0, OCSP_ERR_PARSE) }

    let (inner3, response_bytes_outer_tag, s3) = der_read_tlv(&inner)
    if s3 != DER_OK { return (0, OCSP_ERR_PARSE) }
    inner = inner3
    let (response_bytes_reader, e2) = der_enter(&inner, &response_bytes_outer_tag)  // [0] EXPLICIT wrapper
    if e2 != DER_OK { return (0, OCSP_ERR_PARSE) }
    var rb = response_bytes_reader

    let (rb2, response_bytes_seq_tag, s4) = der_read_tlv(&rb)
    if s4 != DER_OK { return (0, OCSP_ERR_PARSE) }
    let (rb3, e3) = der_enter(&rb2, &response_bytes_seq_tag)
    if e3 != DER_OK { return (0, OCSP_ERR_PARSE) }
    var rbi = rb3

    let (rbi2, response_type_tag, s5) = der_read_tlv(&rbi)  // responseType OID -- not branched on; this task only implements id-pkix-ocsp-basic
    if s5 != DER_OK { return (0, OCSP_ERR_PARSE) }
    rbi = rbi2
    let (rbi3, response_octet_tag, s6) = der_read_tlv(&rbi)  // response OCTET STRING, containing DER-encoded BasicOCSPResponse
    if s6 != DER_OK || response_octet_tag.tag_number != DER_TAG_OCTET_STRING { return (0, OCSP_ERR_PARSE) }

    // BasicOCSPResponse ::= SEQUENCE { tbsResponseData, signatureAlgorithm, signature, certs OPTIONAL }
    var basic = DerReader { buf_ptr: response.ptr, pos: response_octet_tag.content_start, end: response_octet_tag.content_start + response_octet_tag.content_len }
    let (basic2, basic_seq_tag, s7) = der_read_tlv(&basic)
    if s7 != DER_OK { return (0, OCSP_ERR_PARSE) }
    let (basic3, e4) = der_enter(&basic2, &basic_seq_tag)
    if e4 != DER_OK { return (0, OCSP_ERR_PARSE) }
    var bi = basic3

    // tbsResponseData: capture its raw TLV span (tag+length+content) before
    // entering it, since the signature covers this whole encoded SEQUENCE.
    let (bi2, tbs_tag, s8) = der_read_tlv(&bi)
    if s8 != DER_OK { return (0, OCSP_ERR_PARSE) }
    bi = bi2
    // tbs_tag doesn't carry its own outer tag+length header bytes -- recover
    // them the same way stdlib/x509/cert.sio's x509_parse_tbs_core does for
    // TBSCertificate: DER_TAG_SEQUENCE with a definite-length header is
    // always exactly (content_start - tbs_header_start) bytes, computed by
    // re-reading from just before content_start. Since der_read_tlv already
    // advanced past the header, the header start is not directly available
    // here -- record it before the der_read_tlv call instead.
    // (Re-verify this against der_read_tlv's actual field semantics during
    // implementation; if DerTag exposes a tag-start field this plan's
    // grep of stdlib/asn1/der.sio's struct definition didn't find, use that
    // directly instead of recomputing.)

    let (tbs_data, e5) = der_enter(&DerReader { buf_ptr: response.ptr, pos: tbs_tag.content_start, end: tbs_tag.content_start + tbs_tag.content_len }, &tbs_tag)
    // responderID (CHOICE, skip -- not verified in this task's scope, same
    // limitation as issuerNameHash/issuerKeyHash noted above the Interfaces block)
    var tbi = tbs_data
    let (tbi2, _responder_id_tag, s9) = der_read_tlv(&tbi)
    if s9 != DER_OK { return (0, OCSP_ERR_PARSE) }
    tbi = tbi2
    let (tbi3, _produced_at_tag, s10) = der_read_tlv(&tbi)  // producedAt GeneralizedTime, not checked separately from thisUpdate/nextUpdate below
    if s10 != DER_OK { return (0, OCSP_ERR_PARSE) }
    tbi = tbi3

    let (tbi4, responses_seq_tag, s11) = der_read_tlv(&tbi)  // responses SEQUENCE OF SingleResponse
    if s11 != DER_OK { return (0, OCSP_ERR_PARSE) }
    let (responses_reader, e6) = der_enter(&tbi4, &responses_seq_tag)
    if e6 != DER_OK { return (0, OCSP_ERR_PARSE) }
    var resp_iter = responses_reader

    // First SingleResponse only -- see this task's Interfaces note.
    let (resp_iter2, single_response_tag, s12) = der_read_tlv(&resp_iter)
    if s12 != DER_OK { return (0, OCSP_ERR_PARSE) }
    let (sr, e7) = der_enter(&resp_iter2, &single_response_tag)
    if e7 != DER_OK { return (0, OCSP_ERR_PARSE) }
    var sri = sr

    // CertID ::= SEQUENCE { hashAlgorithm, issuerNameHash, issuerKeyHash, serialNumber }
    let (sri2, cert_id_tag, s13) = der_read_tlv(&sri)
    if s13 != DER_OK { return (0, OCSP_ERR_PARSE) }
    sri = sri2
    let (cert_id_reader, e8) = der_enter(&DerReader { buf_ptr: response.ptr, pos: cert_id_tag.content_start, end: cert_id_tag.content_start + cert_id_tag.content_len }, &cert_id_tag)
    var ci = cert_id_reader
    let (ci2, _hash_alg_tag, s14) = der_read_tlv(&ci)
    if s14 != DER_OK { return (0, OCSP_ERR_PARSE) }
    ci = ci2
    let (ci3, _issuer_name_hash_tag, s15) = der_read_tlv(&ci)
    if s15 != DER_OK { return (0, OCSP_ERR_PARSE) }
    ci = ci3
    let (ci4, _issuer_key_hash_tag, s16) = der_read_tlv(&ci)
    if s16 != DER_OK { return (0, OCSP_ERR_PARSE) }
    ci = ci4
    let (ci5, serial_tag, s17) = der_read_tlv(&ci)
    if s17 != DER_OK { return (0, OCSP_ERR_PARSE) }
    let (response_serial, s18) = der_read_integer(&ci, &serial_tag)
    if s18 != DER_OK { return (0, OCSP_ERR_PARSE) }
    if bigint_cmp(&response_serial, &target.serial_number) != 0 {
        return (0, OCSP_ERR_CERT_ID_MISMATCH)
    }

    // certStatus ::= CHOICE { good[0] IMPLICIT NULL, revoked[1] IMPLICIT ..., unknown[2] IMPLICIT NULL }
    let (sri3, status_choice_tag, s19) = der_read_tlv(&sri)
    if s19 != DER_OK { return (0, OCSP_ERR_PARSE) }
    sri = sri3
    var cert_status: i64 = OCSP_STATUS_UNKNOWN
    if status_choice_tag.tag_number == 0 { cert_status = OCSP_STATUS_GOOD }
    else if status_choice_tag.tag_number == 1 { cert_status = OCSP_STATUS_REVOKED }
    else { cert_status = OCSP_STATUS_UNKNOWN }

    let (sri4, this_update_tag, s20) = der_read_tlv(&sri)
    if s20 != DER_OK { return (0, OCSP_ERR_PARSE) }
    sri = sri4
    let (this_update_unix, s21) = x509_generalized_time_to_unix(response, &this_update_tag)
    if s21 != DER_OK { return (0, OCSP_ERR_PARSE) }
    if now_unix < this_update_unix {
        return (0, OCSP_ERR_STALE)
    }
    // nextUpdate is OPTIONAL -- only check staleness against it if a further
    // TLV remains in `sri`'s range; this task does not attempt to
    // distinguish "absent" from "parse error" beyond that presence check.
    if sri.pos < sri.end {
        let (_sri5, next_update_outer_tag, s22) = der_read_tlv(&sri)
        if s22 == DER_OK {
            let (next_update_reader, e9) = der_enter(&DerReader { buf_ptr: response.ptr, pos: next_update_outer_tag.content_start, end: next_update_outer_tag.content_start + next_update_outer_tag.content_len }, &next_update_outer_tag)
            if e9 == DER_OK {
                let (_nur2, next_update_tag, s23) = der_read_tlv(&next_update_reader)
                if s23 == DER_OK {
                    let (next_update_unix, s24) = x509_generalized_time_to_unix(response, &next_update_tag)
                    if s24 == DER_OK && now_unix > next_update_unix {
                        return (0, OCSP_ERR_STALE)
                    }
                }
            }
        }
    }

    // Signature verification: hash tbs_tag's full raw TLV bytes (tag+length+content,
    // matching how x509_verify_signature hashes a TBSCertificate's raw bytes)
    // and PKCS#1-verify against `issuer`'s public key -- OCSP responses in
    // this deployment are always signed directly by the certificate's own
    // issuer (no delegated responder certificate), matching the spec's
    // documented scope (a self-hosted server the caller already trusts).
    let tbs_len = tbs_tag.content_len  // NOTE: see the header-bytes caveat above this block -- if der_read_tlv exposes a tag-start offset, hash from THAT offset through content_start+content_len, not just content bytes, since the signature covers the full encoded SEQUENCE including its own tag+length header.
    let tbs_copy = rawbuf_new(tbs_len + 8)
    var tc: i64 = 0
    while tc < tbs_len {
        rawbuf_set(&tbs_copy, tc, rawbuf_get(response, tbs_tag.content_start + tc))
        tc = tc + 1
    }

    // signatureAlgorithm + signature BIT STRING, siblings of tbsResponseData
    // inside BasicOCSPResponse -- `bi` was left positioned right after
    // tbsResponseData by the der_read_tlv call above.
    let (bi3, _sig_alg_tag, s25) = der_read_tlv(&bi)
    if s25 != DER_OK { return (0, OCSP_ERR_PARSE) }
    let (bi4, sig_tag, s26) = der_read_tlv(&bi3)
    if s26 != DER_OK || sig_tag.tag_number != DER_TAG_BIT_STRING { return (0, OCSP_ERR_PARSE) }

    var sig_bytes: [u8; 512] = [0; 512]
    var sb: i64 = 1  // skip the BIT STRING's leading unused-bits-count byte
    while sb < sig_tag.content_len {
        sig_bytes[(sb - 1) as usize] = rawbuf_get(response, sig_tag.content_start + sb) as u8
        sb = sb + 1
    }

    let (message_hash, message_hash_len) = sha256_hash_rawbuf(&tbs_copy, tbs_len)  // adjust hash function/algorithm to whatever signatureAlgorithm's OID actually names -- Step 1's asn1parse output tells you which; sha256_hash_rawbuf here is a placeholder name for "whichever of stdlib/hash/*'s existing shaN functions matches", not a new function to write
    let verified = pkcs1_verify(&sig_bytes, sig_tag.content_len - 1, &message_hash, message_hash_len, &issuer.modulus, &issuer.public_exponent)
    if !verified {
        return (0, OCSP_ERR_BAD_SIGNATURE)
    }

    (cert_status, OCSP_OK)
}
```

This step's code block is denser and has more open verification points than earlier tasks (the `tbs_tag` header-bytes caveat, the exact hash-function call, `pkcs1_verify`'s exact name/signature) because OCSP response parsing genuinely has no prior art in this codebase to ground against, unlike `chain.sio`'s reuse of `x509_verify_signature`. Before considering Step 4 done, resolve every inline caveat comment against the real `stdlib/asn1/der.sio` (does `DerTag` expose a tag-start offset?) and `stdlib/crypto/pkcs1.sio` (what is the real name and parameter order of its signature-verification entry point? — PR #2133's `x509_verify_signature` calls into it, so read that call site rather than guessing) — replace every caveat comment with the confirmed real code, and note in the completion report which assumptions held and which needed correction.

- [ ] **Step 5: Fill in the test's real values from Step 1, then run**

Run: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run tests/run-pass/ocsp_verify_good.sio`
Expected: prints `ocsp_verify_good: stapled response verified as good`, exit 0.

- [ ] **Step 6: Write the adversarial OCSP test suite**

```sio
//@ run-pass
// Reuses Step 1's OCSP setup (openssl ocsp responder against the same CA).
use x509::ocsp::*
use x509::cert::*
use net::socket::*

fn main() -> i64 with IO {
    // Case (a): revoke the leaf in demoCA/index.txt (openssl ca -revoke),
    // regenerate the OCSP response, feed it in -> expect (OCSP_STATUS_REVOKED, OCSP_OK)
    // (a revoked response is NOT a parse/signature error -- err is OCSP_OK,
    // status is what tells the caller to reject it; chain.sio's Task 6
    // wiring is what turns OCSP_STATUS_REVOKED into a CHAIN_ERR_*)

    // Case (b): tamper one byte inside the response's signature BIT STRING
    // -> expect (_, OCSP_ERR_BAD_SIGNATURE)

    // Case (c): a response whose serialNumber doesn't match target's
    // (generate against a DIFFERENT leaf cert issued by the same
    // intermediate) -> expect (_, OCSP_ERR_CERT_ID_MISMATCH)

    // Case (d): now_unix far past nextUpdate -> expect (_, OCSP_ERR_STALE)

    println("ocsp_verify_adversarial: all 4 cases (a-d) correctly rejected")
    return 0
}
```

Fill in every commented case with real code and the exact expected tuple before running, same discipline as Task 4 Step 5.

- [ ] **Step 7: Run the full OCSP test suite**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix ocsp_ --verbose`
Expected: all passed, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add stdlib/x509/ocsp.sio tests/run-pass/ocsp_verify_good.sio tests/run-pass/ocsp_verify_adversarial.sio
git commit -m "feat(x509): add OCSP stapled-response verification"
```

---

## Task 6: wire OCSP into `x509_verify_chain`

**Files:**
- Modify: `stdlib/x509/chain.sio` (extend `x509_verify_chain`'s signature and body)
- Test: `tests/run-pass/x509_chain_ocsp_wiring.sio`

**Interfaces:**
- Consumes: `ocsp_verify_response`, `OCSP_OK`, `OCSP_STATUS_GOOD` (Task 5, `x509::ocsp`).
- Produces (the **final** public shape of this sub-project's one advertised entry point — this is what the TLS handshake sub-project's spec/plan should reference): `pub const CHAIN_ERR_OCSP_REVOKED: i64 = 0 - 8`, `pub const CHAIN_ERR_OCSP_INVALID: i64 = 0 - 9`, and `x509_verify_chain`'s signature grows two parameters, appended at the end (so existing call sites from Task 4's tests need only an append, not a reorder): `pub fn x509_verify_chain(leaf: &Certificate, leaf_buf: &RawBuf, intermediates: &[Certificate; 8], intermediate_bufs: &[RawBuf; 8], intermediate_count: i32, trust_store: &TrustStore, hostname: &RawBuf, hostname_len: i64, now_unix: i64, ocsp_response: &RawBuf, ocsp_response_len: i64) -> i64 with IO`.

- [ ] **Step 1: Update Task 4's existing tests to pass the two new parameters**

Every existing call to `x509_verify_chain` in `tests/run-pass/x509_chain_verify_positive.sio` and `tests/run-pass/x509_chain_adversarial.sio` needs `, &rawbuf_new(1), 0` appended to its argument list (an empty/zero-length OCSP response — the fail-soft "no stapling" case, which these tests' outcomes must NOT change).

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_chain_ --verbose`
Expected (before Step 2's implementation change): compile errors — `x509_verify_chain` doesn't have this many parameters yet. This is the "write the failing test(s) first" step for this task; Task 4's tests, now edited to call the *new* signature, serve as regression coverage that adding OCSP didn't change their outcomes.

- [ ] **Step 2: Write the new wiring test**

```sio
//@ run-pass
// Reuses the same 3-level fixture and OCSP response as Task 5's ocsp_verify_good.sio.
use x509::chain::*
use x509::trust_store::*
use x509::cert::*
use net::socket::*

fn main() -> i64 with IO {
    // ... (identical setup to x509_chain_verify_positive.sio, plus the OCSP
    // response bytes from ocsp_verify_good.sio) ...

    // Case: empty OCSP response (fail-soft) -> still CHAIN_OK.
    // let result_empty = x509_verify_chain(..., &rawbuf_new(1), 0)
    // if result_empty != CHAIN_OK { print_int(1); return 1 }

    // Case: good stapled OCSP response -> still CHAIN_OK.
    // let result_good = x509_verify_chain(..., &ocsp_resp_buf, ocsp_resp_len)
    // if result_good != CHAIN_OK { print_int(2); return 1 }

    // Case: revoked stapled OCSP response (Task 5's revoked fixture) -> CHAIN_ERR_OCSP_REVOKED.
    // let result_revoked = x509_verify_chain(..., &revoked_ocsp_resp_buf, revoked_ocsp_resp_len)
    // if result_revoked != CHAIN_ERR_OCSP_REVOKED { print_int(3); return 1 }

    println("x509_chain_ocsp_wiring: fail-soft, good, and revoked cases all correct")
    return 0
}
```

- [ ] **Step 3: Implement the wiring — modify `x509_verify_chain` in `stdlib/x509/chain.sio`**

```sio
use x509::ocsp::*

pub const CHAIN_ERR_OCSP_REVOKED: i64 = 0 - 8
pub const CHAIN_ERR_OCSP_INVALID: i64 = 0 - 9

pub fn x509_verify_chain(
    leaf: &Certificate,
    leaf_buf: &RawBuf,
    intermediates: &[Certificate; 8],
    intermediate_bufs: &[RawBuf; 8],
    intermediate_count: i32,
    trust_store: &TrustStore,
    hostname: &RawBuf,
    hostname_len: i64,
    now_unix: i64,
    ocsp_response: &RawBuf,
    ocsp_response_len: i64,
) -> i64 with IO {
    let (candidates, candidate_count) = chain_build_candidates(leaf, leaf_buf, intermediates, intermediate_bufs, intermediate_count, trust_store)
    if candidate_count == 0 {
        return CHAIN_ERR_NO_PATH_TO_ROOT
    }

    var best_depth: i32 = 0 - 1
    var best_error: i64 = CHAIN_ERR_NO_PATH_TO_ROOT
    var c: i32 = 0
    while c < candidate_count {
        let cand = candidates[c as usize]
        let (result, depth_reached) = chain_verify_path(&cand.certs, &cand.bufs, cand.len, now_unix)
        if result == CHAIN_OK {
            if !x509_verify_hostname(leaf, hostname, hostname_len) {
                return CHAIN_ERR_HOSTNAME_MISMATCH
            }
            if ocsp_response_len > 0 {
                let issuer_in_path = cand.certs[1]
                let (ocsp_status, ocsp_err) = ocsp_verify_response(ocsp_response, ocsp_response_len, leaf, &issuer_in_path, now_unix)
                if ocsp_err != OCSP_OK {
                    return CHAIN_ERR_OCSP_INVALID
                }
                if ocsp_status != OCSP_STATUS_GOOD {
                    return CHAIN_ERR_OCSP_REVOKED
                }
            }
            return CHAIN_OK
        }
        if depth_reached > best_depth {
            best_depth = depth_reached
            best_error = result
        }
        c = c + 1
    }

    best_error
}
```

(This replaces the function body Task 4 wrote — same name, extended signature, same core loop, with the OCSP branch inserted right before the `return CHAIN_OK` on the winning path.)

- [ ] **Step 4: Fill in Task 4's test files' new `&rawbuf_new(1), 0` arguments and this task's new test's real values, then run everything**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_chain_ --verbose`
Expected: all passed (Task 3's path-building tests + Task 4's positive/adversarial tests, now calling the extended signature + this task's new wiring test), 0 failed.

- [ ] **Step 5: Run every test this whole plan added, together, as a final checkpoint**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix pem_ --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_trust_store_ --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix x509_chain_ --verbose`
Run: `bash scripts/run_sio_test_suite.sh --filter-prefix ocsp_ --verbose`
Expected: every one reports all passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add stdlib/x509/chain.sio tests/run-pass/x509_chain_verify_positive.sio tests/run-pass/x509_chain_adversarial.sio tests/run-pass/x509_chain_ocsp_wiring.sio
git commit -m "feat(x509): wire OCSP stapling into x509_verify_chain"
```

---

## After This Plan

`x509::chain::x509_verify_chain` (final 11-parameter shape from Task 6) is the interface the not-yet-written TLS handshake sub-project's spec should reference directly — that sub-project's own brainstorming/spec process starts fresh once this plan's final whole-branch review (per `superpowers:subagent-driven-development`) is clean.

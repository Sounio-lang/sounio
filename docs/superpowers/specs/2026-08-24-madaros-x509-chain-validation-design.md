<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-24-madaros-x509-chain-validation-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-24-madaros-x509-chain-validation-design
-->

# Madaros X.509 Chain Validation Design

## Overview

**Goal:** validate a certificate chain presented by a TLS server against the
system's trusted root CA store, with generic path building, hostname
verification, and fail-soft OCSP-stapling revocation checking.

**Why:** this is the first of two sub-projects needed to bring real HTTPS to
Madaros, which unblocks a web-search tool being built for the Conclave chat
app (talks to local models; needs to fetch/search the web, and target search
backends run behind standard HTTPS reverse proxies — nginx/Caddy defaults,
not a config we control). The second sub-project (not covered here) is the
TLS handshake protocol itself — record layer, ECDHE key exchange, AEAD
cipher, key schedule — which will consume this sub-project's chain
verification as a black-box interface. This decomposition was chosen during
brainstorming because the combined scope (protocol state machine + chain
trust) was too large for one spec.

**Builds on** (already merged or in-flight, all in this repository):
- `stdlib/asn1/der.sio` — DER decoder (merged, `main`).
- `stdlib/x509/cert.sio` — `Certificate`/`X509Name`/`GeneralName`/`ExtensionEntry`
  structs, `x509_parse_certificate`, `x509_verify_signature` (PR #2133,
  `tls-on-madaros`, not yet merged to `integration/sounio-dev-ready-base` —
  this sub-project's implementation plan assumes that PR is merged first, or
  targets the same branch).
- `stdlib/x509/oid.sio` — 28 OID accessors including PKIX AIA OIDs.
- `stdlib/crypto/pkcs1.sio` — RSASSA-PKCS1-v1_5-VERIFY.
- `stdlib/bignum/bigint.sio` — fixed-width BigInt (4096-bit, 256×16-bit limbs).
- `stdlib/encoding/base64.sio` — base64 decode (merged, `main`).

**Tech stack:** Sounio/Madaros (`./bin/souc`), no non-Sounio code, `read_file`
from the confirmed `extern "C"` allow-list.

## Global Constraints

- Linux x86-64 only (per repo-wide constraint); trust store source is
  `/etc/ssl/certs/ca-certificates.crt` (Debian/Ubuntu path — the only
  supported system CA bundle location for this design; no fallback path
  probing).
- Client role only — no certificate chain *issuance* or server-side TLS
  concerns in this sub-project.
- Error convention: sentinel `i64` return values with named `pub const`
  error codes — never `Result<T,E>`/`Option<T>`, matching the rest of the
  X.509/crypto stdlib.
- No network I/O anywhere in this sub-project. OCSP stapling verification
  consumes a byte blob the caller already has (delivered by the TLS
  handshake's `status_request` extension in sub-project 2) — this module
  never performs its own OCSP HTTP fetch.
- No CRL support, no direct (non-stapled) OCSP queries. Revocation checking
  is fail-soft: absence of a stapled OCSP response is accepted (chain is
  trusted on signature/validity/hostname alone); presence of an invalid or
  revoked stapled response fails the chain. This asymmetry is a deliberate,
  documented limitation, not an oversight — hard-fail-without-stapling would
  reject the large fraction of real servers that don't staple.
- No path length limit beyond each cert's own `basicConstraints`
  `pathLenConstraint` (already parsed into `Certificate.path_len_constraint`
  by the existing `cert.sio`).
- Hostname verification supports exact match and single-level leftmost
  wildcard (`*.example.com` matches `foo.example.com`, not
  `foo.bar.example.com` and not the bare `example.com`) — not full RFC 6125.
- Commit convention: Conventional Commits (`feat(x509): ...`), no AI
  attribution line, matching this branch's established history.
- Module imports: bare `use <filename>::{name|*}`, no `module` declaration.
- Test invocation: `tests/run-pass/*.sio` / `tests/compile-fail/*.sio` via
  `bash scripts/run_sio_test_suite.sh --filter-prefix x509_chain_` (or
  `pem_`, `ocsp_` as appropriate) — never the whole-repo suite for this
  sub-project's checkpoints.

---

## Architecture

Four new modules under `stdlib/x509/`:

| Module | Responsibility |
|---|---|
| `pem.sio` | Decode concatenated PEM blocks (`-----BEGIN CERTIFICATE-----...-----END CERTIFICATE-----`) into raw DER byte buffers, using `stdlib/encoding/base64.sio` for the body. |
| `trust_store.sio` | Read `/etc/ssl/certs/ca-certificates.crt` via `read_file`, split into PEM blocks (`pem.sio`), parse each into a `Certificate` (`cert.sio`'s `x509_parse_certificate`), and hold them in an in-memory table indexed by Subject DN and by Subject Key Identifier. |
| `chain.sio` | Path building (generic, with backtracking) from a leaf certificate through caller-supplied intermediates to a root in the trust store, followed by full verification of the winning path. Exposes `x509_verify_chain` and `x509_verify_hostname`. |
| `ocsp.sio` | ASN.1 parsing of `OCSPResponse`/`BasicOCSPResponse` (RFC 6960), signature verification of the response, and status/freshness checks against one target certificate. |

`chain.sio` is the only module the TLS handshake sub-project calls directly;
`pem.sio`, `trust_store.sio`, and `ocsp.sio` are internal collaborators (all
still `pub` at the module level, since Sounio has no separate
crate-private visibility beyond file-level `pub`, but not part of the
sub-project's advertised public interface).

---

## Data Structures

### `pem.sio`

```sio
pub const PEM_ERR_NO_BEGIN_MARKER: i64 = -1
pub const PEM_ERR_NO_END_MARKER: i64 = -2
pub const PEM_ERR_BASE64_DECODE: i64 = -3

// Decodes the first PEM block found at or after `start` in `buf`.
// Returns (der_bytes, der_len, next_search_offset, error_code).
// error_code is 0 on success; next_search_offset is set past the block's
// END marker so callers can loop to find the next block, or -1 when no
// further BEGIN marker exists (normal end-of-bundle, not an error).
pub fn pem_decode_next(buf: &RawBuf, buf_len: i64, start: i64) -> (RawBuf, i64, i64, i64) with IO
```

### `trust_store.sio`

```sio
pub const TRUST_STORE_MAX_CERTS: i64 = 512
pub const TRUST_STORE_ERR_READ_FAILED: i64 = -1
pub const TRUST_STORE_ERR_TOO_MANY_CERTS: i64 = -2

pub struct TrustStore {
    certs: [Certificate; 512],   // TRUST_STORE_MAX_CERTS
    bufs: [RawBuf; 512],         // certs[i]'s own raw DER bytes -- x509_verify_signature
                                  // re-hashes cert.tbs_start..tbs_start+tbs_len out of the
                                  // SAME buffer a cert was parsed from (verified against
                                  // cert.sio's x509_verify_signature body), so every stored
                                  // cert needs its source buffer kept alongside it.
    count: i32,
}

// Reads and parses /etc/ssl/certs/ca-certificates.crt once. Call once per
// process; reuse the returned TrustStore across every x509_verify_chain
// call (never re-read per connection).
pub fn trust_store_load() -> (TrustStore, i64) with IO

// Linear scan helpers used by chain.sio's path building — kept as plain
// functions (not methods) matching this stdlib's existing style.
pub fn trust_store_find_by_subject(store: &TrustStore, subject: &X509Name) -> i64  // index, or -1
pub fn trust_store_find_by_ski(store: &TrustStore, ski: &[u8; 32], ski_len: i32) -> i64  // index, or -1
```

`TRUST_STORE_MAX_CERTS = 512` is sized against the real Debian/Ubuntu
`ca-certificates` package (~150 roots as of 2026); 512 leaves headroom
without the unbounded-growth risk of a dynamic structure in a stdlib that
otherwise uses fixed-size arrays throughout (`Certificate.extensions`,
`.san_entries`, etc. all follow this pattern already).

### `chain.sio`

```sio
pub const CHAIN_MAX_DEPTH: i64 = 10
pub const CHAIN_OK: i64 = 0
pub const CHAIN_ERR_NO_PATH_TO_ROOT: i64 = -1
pub const CHAIN_ERR_EXPIRED: i64 = -2
pub const CHAIN_ERR_NOT_YET_VALID: i64 = -3
pub const CHAIN_ERR_NOT_CA: i64 = -4
pub const CHAIN_ERR_PATH_LEN_EXCEEDED: i64 = -5
pub const CHAIN_ERR_BAD_SIGNATURE: i64 = -6
pub const CHAIN_ERR_HOSTNAME_MISMATCH: i64 = -7
pub const CHAIN_ERR_OCSP_REVOKED: i64 = -8
pub const CHAIN_ERR_OCSP_INVALID: i64 = -9
pub const CHAIN_ERR_TOO_DEEP: i64 = -10

pub const MAX_INTERMEDIATES: i64 = 8

// Top-level entry point. `intermediates`/`intermediate_count` are the certs
// the server presented besides the leaf, in whatever order it sent them.
// `ocsp_response`/`ocsp_response_len` may be zero-length when the server did
// not staple a response. `now_unix` is passed in by the caller (no direct
// clock read in this module, matching this stdlib's existing
// `unix_timestamp_from_ymdhms`-based, caller-supplies-time convention).
pub fn x509_verify_chain(
    leaf: &Certificate,
    leaf_buf: &RawBuf,                  // leaf's own raw DER bytes -- see TrustStore.bufs' comment
    intermediates: &[Certificate; 8],   // MAX_INTERMEDIATES
    intermediate_bufs: &[RawBuf; 8],    // intermediates[i]'s own raw DER bytes, same reason
    intermediate_count: i32,
    trust_store: &TrustStore,
    ocsp_response: &RawBuf,
    ocsp_response_len: i64,
    hostname: &RawBuf,
    hostname_len: i64,
    now_unix: i64,
) -> i64 with IO   // CHAIN_OK (0) or one of the CHAIN_ERR_* negative codes

pub fn x509_verify_hostname(leaf: &Certificate, hostname: &RawBuf, hostname_len: i64) -> bool
```

`x509_verify_chain` is deliberately a single function rather than
separately-exposed path-building/path-verification/hostname/OCSP calls: the
handshake sub-project only ever needs the one yes/no-plus-reason answer, and
keeping the staged internals (`chain_build_candidates`, `chain_verify_path`)
file-private to `chain.sio` avoids leaking the path-building data structures
(candidate stacks, visited-set for backtracking) into the sub-project's
public interface.

### `ocsp.sio`

```sio
pub const OCSP_OK: i64 = 0
pub const OCSP_ERR_PARSE: i64 = -1
pub const OCSP_ERR_CERT_ID_MISMATCH: i64 = -2
pub const OCSP_ERR_BAD_SIGNATURE: i64 = -3
pub const OCSP_ERR_STALE: i64 = -4
pub const OCSP_STATUS_GOOD: i64 = 0
pub const OCSP_STATUS_REVOKED: i64 = 1
pub const OCSP_STATUS_UNKNOWN: i64 = 2

// Verifies a stapled OCSPResponse against `target` (the leaf certificate)
// and `issuer` (the leaf's direct issuer, from the winning chain path).
// Returns (ocsp_status, error_code): error_code is OCSP_OK when the
// response parsed and its signature verified; ocsp_status is only
// meaningful when error_code == OCSP_OK, and OCSP_STATUS_GOOD is the only
// status chain.sio treats as passing.
pub fn ocsp_verify_response(
    response: &RawBuf,
    response_len: i64,
    target: &Certificate,
    issuer: &Certificate,
    now_unix: i64,
) -> (i64, i64) with IO
```

---

## Data Flow

1. Caller loads the trust store once: `let (store, err) = trust_store_load()`.
2. Per connection, after the TLS handshake sub-project has received the
   server's Certificate message and parsed each cert via
   `x509_parse_certificate`:
   `x509_verify_chain(&leaf, &intermediates, n, &store, &ocsp_bytes, ocsp_len, &hostname_buf, hostname_len, now)`.
3. Inside `x509_verify_chain`:
   - `chain_build_candidates` (file-private) treats `{leaf} ∪ intermediates ∪ store.certs` as the available cert pool (each entry carrying its `Certificate` alongside its own `RawBuf`, per the data-structure fix above) and performs a depth-bounded (`CHAIN_MAX_DEPTH`) DFS from the leaf: at each step, find candidate issuers by matching the current cert's `authority_key_id` against a candidate's `subject_key_id` (preferred, when both are present), falling back to `issuer` DN == candidate's `subject` DN when AKI/SKI is absent on either side. Each candidate issuer not already on the current path is pushed; a self-signed cert found in `store.certs` (subject == issuer, found via `trust_store_find_by_subject`/`_by_ski`) terminates a path successfully. Backtracks (pops and tries the next candidate) when a branch dead-ends before reaching a trusted root within `CHAIN_MAX_DEPTH`.
   - Each candidate path found is fed to `chain_verify_path` (file-private): walks the path from leaf to root checking, per link, `x509_verify_signature(cert_buf, cert, issuer.modulus, issuer.exponent)` (using that link's own stored buffer — never the leaf's or another link's), `now_unix` inside `[cert.not_before_unix, cert.not_after_unix]`, and — for every non-leaf link — `issuer.is_ca == true`, `issuer.key_usage_bits` has `keyCertSign` set when `key_usage_bits != 0` (absent key usage extension does not block, matching common real-world CA certs that omit it), and the number of CA certs between this link and the leaf does not exceed `issuer.path_len_constraint` when that field is set (`>= 0`; the existing parser encodes "absent" as a sentinel, matching the parser's established convention — verified against `cert.sio`'s parsing code, not assumed, during implementation).
   - The **first fully-passing candidate path wins**; if none pass, the error reported is the failure reason from whichever candidate got furthest (most links verified before failing) — tracked as a running `(best_depth, best_error)` pair updated whenever a new candidate's failure depth exceeds the stored best.
   - On a winning path: `x509_verify_hostname(&leaf, hostname, hostname_len)` — SAN `dNSName` entries only (not deprecated CN-based fallback), exact or single-level-wildcard match.
   - If `ocsp_response_len > 0`: `ocsp_verify_response` against `(leaf, winning_path[1])` (the leaf's direct issuer) — `OCSP_STATUS_GOOD` and `OCSP_OK` required to pass; anything else fails the whole chain.
4. Returns `CHAIN_OK` only if path verification, hostname, and (when present) OCSP all pass.

---

## Error Handling

All sentinel `i64`, all `pub const`, one flat namespace per module (`CHAIN_ERR_*`, `OCSP_ERR_*`/`OCSP_STATUS_*`, `PEM_ERR_*`, `TRUST_STORE_ERR_*`) — consistent with `stdlib/x509/cert.sio`'s existing `X509_ERR_*` family.

The "most-progress" failure-reporting heuristic (§ Data Flow, step 3) exists specifically so a single spurious name-collision candidate (e.g., two unrelated CAs sharing a subject DN substring) doesn't mask the real failure reason (e.g., "expired") on the path that was actually intended.

---

## Testing Strategy

Following the pattern established in PR #2133 (openssl-generated real certs, cross-checked against `openssl x509 -text` / `openssl verify`):

- **`pem_*` tests**: round-trip a real cert's PEM form (from `openssl x509 -outform PEM`) through `pem_decode_next` and diff the resulting DER bytes against `openssl x509 -outform DER` byte-for-byte. A second test feeds a bundle with 2+ concatenated blocks and confirms both are found via the `next_search_offset` chaining.
- **`x509_chain_*` tests**: build a real 3-level chain with `openssl` (self-signed test root, intermediate, leaf). Tests never touch `/etc/ssl/certs/` or `trust_store_load`: they construct a `TrustStore` directly in memory, parsing the test root via `x509_parse_certificate` and assigning it straight into a `TrustStore { certs: [...], count: 1 }` literal — `x509_verify_chain` takes a `&TrustStore` and has no idea whether it came from `trust_store_load()` or was built by hand, so this is a normal call, not a test-only code path. Positive case (full chain verifies), then adversarial cases: expired leaf, expired intermediate, intermediate missing `CA:true`, `pathLenConstraint` violated by an extra intermediate, tampered signature byte, hostname mismatch (plain and wildcard), intermediates presented in reversed order (exercises path-building, not just linear verification).
- **`ocsp_*` tests**: a real stapled-response byte sequence (captured via `openssl ocsp` against the test CA above, or hand-built per RFC 6960's ASN.1 grammar and cross-checked with `openssl asn1parse`) for the good/revoked/expired-response/bad-signature cases, plus the fail-soft empty-response case verified through `x509_verify_chain` directly (zero-length `ocsp_response` must not affect the result).
- Every adversarial test's expected rejection reason is asserted explicitly (the specific `CHAIN_ERR_*`/`OCSP_ERR_*` code), not just "non-zero" — matching this repo's established preference for exact-value assertions over loose bounds (per `docs/guide/SOUNIO_STYLE_GUIDE.md` conventions already followed in `tests/run-pass/x509_adversarial.sio`).

## Out of Scope (explicit)

- CRL support.
- Direct (non-stapled) OCSP queries — this module never makes a network call.
- Hard-fail-when-unstapled revocation policy.
- Full RFC 6125 hostname verification (multi-level wildcards, CN fallback, SRV-ID/URI-ID).
- Embedding a project-maintained CA bundle — the system bundle is the only source (§ Global Constraints).
- Certificate Transparency SCT verification against a log's public key (SCT *parsing* already exists in `stdlib/x509/sct.sio` from PR #2133; verifying an SCT's signature against a known log key is a separate, not-yet-scoped concern).
- Server-role TLS / certificate issuance.

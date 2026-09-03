<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-25-madaros-hkdf-tls13-keyschedule-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-25-madaros-hkdf-tls13-keyschedule-design
-->

# Madaros HKDF + TLS 1.3 Key Schedule Design

## Overview

**Goal:** implement RFC 5869 HKDF, RFC 8446 §7.1's `HKDF-Expand-Label`/
`Derive-Secret` wrappers, and the minimal TLS 1.3 key-derivation ladder a
fresh (no PSK, no session resumption, no early data) HTTPS client needs —
in pure Sounio, verified against a complete, real, published TLS 1.3
handshake trace.

**Why:** this is the first of two remaining sub-projects toward a working
TLS 1.3 client on Madaros. X.509 chain validation, AEAD ciphers
(AES-128/256-GCM, ChaCha20-Poly1305), and X25519 key exchange are already
merged on this branch (PR #2133). This sub-project produces the key
schedule that turns an X25519 shared secret plus a running transcript
hash into concrete AEAD traffic keys; the next, not-yet-scoped sub-project
(the handshake state machine + record layer) will call this sub-project's
functions at the right points in the handshake but never reimplement any
key-derivation math itself.

**Tech stack:** pure Sounio, no non-Sounio code anywhere in this
sub-project's deliverables.

## Global Constraints

- **RawBuf-based throughout**, consistent with this branch's AEAD/X25519
  work — TLS secrets and derived material have hash-output-dependent
  lengths (32 bytes for SHA-256, 48 for SHA-384), so a single set of
  RawBuf-returning functions parameterized by a `hash_algo` constant is
  used instead of separate fixed-array-typed function families per hash.
- **Does not touch the existing `stdlib/crypto/hmac.sio`.** That file's
  `hmac_sha256` is file-private, fixed-`[u8;256]`-array-based, and
  untested by this branch's own work — left exactly as-is, matching this
  branch's established precedent (e.g. `stdlib/crypto/sha256.sio` was
  similarly left alone when the AEAD sub-project needed a RawBuf-based
  SHA-256, which already existed separately at `stdlib/hash/sha256.sio`).
  This sub-project builds fresh, RawBuf-based
  `hmac_sha256_rawbuf`/`hmac_sha384_rawbuf` functions on top of the
  already-existing, already-`pub`,
  already-RawBuf-based `stdlib/hash/sha256.sio`'s `sha256(buf: &RawBuf,
  len: i64) -> [u8;32]` and `stdlib/hash/sha384.sio`'s `sha384(buf:
  &RawBuf, len: i64) -> [u8;48]`.
- **Scope is deliberately minimal** — only the key-schedule path a fresh,
  no-PSK, no-early-data, no-resumption client connection actually uses:
  Early Secret → Handshake Secret (via the X25519 shared secret) →
  handshake traffic secrets → Master Secret → application traffic
  secrets → per-direction key/IV. `binder_key`,
  `early_exporter_master_secret`, `client_early_traffic_secret`,
  `exporter_master_secret`, `resumption_master_secret`, and the
  post-handshake key-update mechanism (`"traffic upd"`) are explicitly
  out of scope — this client neither offers early data nor resumes
  sessions.
- Error convention: sentinel `bool`/`i64` — never
  `Result<T,E>`/`Option<T>`. Most functions here have no realistic
  failure mode given well-formed fixed-size inputs (hash functions and
  HMAC don't fail); no new sentinel error codes are introduced unless a
  genuine failure mode is found during implementation.
- Every test's expected value comes from a published, independent source
  — RFC 5869 (HKDF), RFC 4231 (HMAC-SHA256/SHA384), and **RFC 8448**
  ("Example Handshake Traces for TLS 1.3", which documents a complete,
  real TLS 1.3 handshake using X25519 + AES-128-GCM-SHA256 — exactly this
  branch's own already-built crypto stack — with every intermediate
  secret, key, and IV published) — never self-generated, and every
  implementer/reviewer independently re-verifies each cited vector
  against the actual RFC text before trusting it in committed test code,
  per this project's standing discipline (doubly important here: this
  branch's AEAD ciphers sub-project found a real, self-consistent Poly1305
  bug invisible to every mandated RFC vector, and its X25519 sub-project
  found a real entropy bias invisible to every task-level test — both
  were only caught by adversarial, beyond-the-mandated-vectors scrutiny
  at final review).
- Module imports: bare `use <filename>::{name|*}`, no `module`
  declaration.
- No AI-attribution line in commits; Conventional Commits style
  (`feat(crypto): ...`).

---

## Architecture

One new module, `stdlib/crypto/hkdf.sio`, in three layers:

| Layer | Responsibility |
|---|---|
| HMAC | `hmac_sha256_rawbuf`/`hmac_sha384_rawbuf` — fresh, RawBuf-based, built on `stdlib/hash/sha256.sio`/`sha384.sio`. Named distinctly from the pre-existing, differently-shaped, fixed-array-based `pub fn hmac_sha256` in `stdlib/crypto/sha256.sio` to avoid a public-symbol collision (renamed at final review; see progress ledger). |
| HKDF | `hkdf_extract`/`hkdf_expand` (RFC 5869 §2.2/2.3) — the generic HKDF primitives, hash-agnostic via the `hash_algo` parameter. |
| TLS 1.3 key schedule | `hkdf_expand_label`/`derive_secret` (RFC 8446 §7.1's TLS-specific wrappers over HKDF-Expand), then `tls13_early_secret`/`tls13_handshake_secret`/`tls13_master_secret` (the minimal secret ladder) and `tls13_traffic_key_and_iv` (the final key/IV derivation, RFC 8446 §7.3). |

This is the only module the not-yet-scoped handshake sub-project calls
into for any key-derivation need.

---

## Data Structures

```sio
pub const HKDF_HASH_SHA256: i32 = 0
pub const HKDF_HASH_SHA384: i32 = 1

pub fn hmac_sha256_rawbuf(key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> [u8;32] with IO
pub fn hmac_sha384_rawbuf(key: &RawBuf, key_len: i64, msg: &RawBuf, msg_len: i64) -> [u8;48] with IO

// RFC 5869 section 2.2/2.3. `hash_algo` selects HMAC-SHA256 (output 32
// bytes) or HMAC-SHA384 (output 48 bytes) internally; both functions
// return a RawBuf sized to the real hash output length (hkdf_extract) or
// the caller's requested `out_len` (hkdf_expand), never a fixed array,
// so callers never need to branch on which hash was used to know the
// buffer shape.
pub fn hkdf_extract(hash_algo: i32, salt: &RawBuf, salt_len: i64, ikm: &RawBuf, ikm_len: i64) -> RawBuf with IO
pub fn hkdf_expand(hash_algo: i32, prk: &RawBuf, prk_len: i64, info: &RawBuf, info_len: i64, out_len: i64) -> RawBuf with IO

// RFC 8446 section 7.1. `label` is the bare TLS 1.3 label text (e.g.
// "c hs traffic") WITHOUT the "tls13 " prefix -- hkdf_expand_label
// prepends it internally, per the RFC's own HkdfLabel structure:
//   struct { uint16 length; opaque label<7..255> = "tls13 " + Label;
//             opaque context<0..255> = Context; } HkdfLabel;
pub fn hkdf_expand_label(hash_algo: i32, secret: &RawBuf, secret_len: i64, label: &RawBuf, label_len: i64, context: &RawBuf, context_len: i64, out_len: i64) -> RawBuf with IO

// Derive-Secret(Secret, Label, Messages) = HKDF-Expand-Label(Secret,
// Label, Transcript-Hash(Messages), Hash.length) -- this function takes
// the transcript hash as an ALREADY-COMPUTED input (transcript hashing
// itself, over the accumulated handshake messages, is the handshake
// sub-project's responsibility, not this module's).
pub fn derive_secret(hash_algo: i32, secret: &RawBuf, secret_len: i64, label: &RawBuf, label_len: i64, transcript_hash: &RawBuf, transcript_hash_len: i64) -> RawBuf with IO

// The minimal secret ladder for a fresh, no-PSK connection.
pub fn tls13_early_secret(hash_algo: i32) -> RawBuf with IO
pub fn tls13_handshake_secret(hash_algo: i32, early_secret: &RawBuf, early_secret_len: i64, ecdhe_shared_secret: &RawBuf, ecdhe_len: i64) -> RawBuf with IO
pub fn tls13_master_secret(hash_algo: i32, handshake_secret: &RawBuf, handshake_secret_len: i64) -> RawBuf with IO

// RFC 8446 section 7.3: derives a direction's AEAD write key and IV from
// its traffic secret. key_len/iv_len are the negotiated AEAD's sizes
// (e.g. 16/12 for AES-128-GCM, 32/12 for AES-256-GCM or ChaCha20-Poly1305).
pub fn tls13_traffic_key_and_iv(hash_algo: i32, traffic_secret: &RawBuf, traffic_secret_len: i64, key_len: i64, iv_len: i64) -> (RawBuf, RawBuf) with IO
```

`tls13_handshake_secret`/`tls13_master_secret` each internally call
`derive_secret(..., "derived", "", ...)` (an empty-message transcript
hash, i.e. the hash of an empty string) before the next `hkdf_extract`
step, per RFC 8446's own key-schedule diagram — this internal
"derived"-label step is not separately exposed, since no caller outside
this module ever needs it standalone.

`Derive-Secret`'s remaining calls (for `"c hs traffic"`/`"s hs
traffic"`/`"c ap traffic"`/`"s ap traffic"`, each needing the REAL
transcript hash at that specific point in the handshake) are made
directly by the handshake sub-project via the public `derive_secret`
function above, not wrapped in additional convenience functions here —
the transcript hash input differs at every call, so there's no shared
logic left to factor out beyond what `derive_secret` itself already
provides.

---

## Data Flow

1. `early_secret = tls13_early_secret(hash_algo)` — internally
   `HKDF-Extract(salt=0, IKM=0)` (both zero-filled, `Hash.length` bytes
   each — HKDF's own "no salt provided" convention).
2. `handshake_secret = tls13_handshake_secret(hash_algo, &early_secret,
   ..., &ecdhe_shared_secret, ...)` — internally: `derived =
   derive_secret(hash_algo, &early_secret, ..., "derived", "", empty_hash,
   ...)` (where `empty_hash` is `Hash("")`, the hash of a zero-length
   input — computed once, a fixed constant per hash algorithm, verified
   directly against a known value rather than assumed), then
   `HKDF-Extract(salt=derived, IKM=ecdhe_shared_secret)`.
3. Handshake sub-project calls `derive_secret(hash_algo,
   &handshake_secret, ..., "c hs traffic", client_hello_through_server_hello_hash, ...)`
   and the `"s hs traffic"` equivalent directly, at the point in its own
   handshake state machine where the transcript hash through ServerHello
   is available.
4. `master_secret = tls13_master_secret(hash_algo, &handshake_secret,
   ...)` — internally: another `"derived"` step, then
   `HKDF-Extract(salt=derived, IKM=0)` (an all-zero IKM this time, per the
   RFC's diagram — NOT the same as step 1's zero-salt-and-zero-IKM case,
   double-check this distinction against the RFC directly during
   implementation rather than assuming symmetry with step 1).
5. Handshake sub-project calls `derive_secret(..., "c ap traffic", ...)`
   / `"s ap traffic"` directly, once the transcript hash through the
   server's Finished message is available.
6. For each traffic secret needed as actual AEAD keys: `(write_key,
   write_iv) = tls13_traffic_key_and_iv(hash_algo, &traffic_secret, ...,
   key_len, iv_len)` — internally two `hkdf_expand_label` calls with
   labels `"key"`/`"iv"` and an empty context.

---

## Error Handling

No new sentinel error codes are introduced by this spec — every function
here operates on well-formed, already-validated fixed-size or
caller-specified-length inputs, with no realistic failure mode (hashing
and HMAC never fail; HKDF-Expand's only documented failure condition per
RFC 5869 — requesting more output than `255 * HashLen` bytes — is not a
concern for TLS 1.3's own key schedule, whose largest single request is a
32-48 byte traffic secret or a cipher key/IV, both far under that limit).
If implementation reveals a genuine failure mode not anticipated here,
follow this codebase's established sentinel `bool`/`i64` convention for
it, documented explicitly in the implementation plan rather than silently
introduced.

---

## Testing Strategy

- **HMAC-SHA256/SHA384**: RFC 4231's official test vectors (multiple test
  cases per hash, covering different key/message lengths including a
  key longer than the hash's block size, which exercises HMAC's own
  key-shortening-via-hash step).
- **HKDF (raw, not TLS-specific)**: RFC 5869 Appendix A's official
  HKDF-SHA256 test cases (Extract and Expand, both with and without a
  salt).
- **The full TLS 1.3 key schedule, end to end, against RFC 8448's
  complete real handshake trace**: RFC 8448 documents an actual TLS 1.3
  handshake (X25519 key exchange, AES-128-GCM-SHA256 cipher suite —
  exactly this branch's own already-built stack) with every intermediate
  value published: the ECDHE shared secret, Early/Handshake/Master
  Secrets, every traffic secret, and the derived keys/IVs. This is
  significantly stronger evidence than isolated unit vectors, since it
  proves the ENTIRE ladder (all the "derived"-label intermediate steps,
  every `Derive-Secret` call with its real transcript-hash input) chains
  together correctly end to end, using the exact transcript-hash values
  RFC 8448 publishes at each step (this sub-project's tests supply those
  published hash values directly as inputs to `derive_secret` — computing
  the transcript hash FROM the handshake messages themselves is the
  handshake sub-project's job, not tested here).
- **Adversarial/edge cases**: `hkdf_expand`'s label/context length limits
  (RFC 8446's `opaque label<7..255>`/`opaque context<0..255>` — verify
  the implementation doesn't silently truncate or overflow if given
  inputs near these boundaries, even though this client's own labels are
  all short, fixed strings well under any limit).
- Every cited vector independently re-verified against the actual RFC
  text (5869, 4231, 8448) before being trusted in committed test code —
  not from a secondhand citation, and not from memory alone.

## Out of Scope (explicit)

- PSK-based resumption, 0-RTT/early data, and every key-schedule branch
  that depends on either (`binder_key`, `early_exporter_master_secret`,
  `client_early_traffic_secret`, `resumption_master_secret`).
- `exporter_master_secret` (TLS exporters, RFC 8446 §7.5) — not needed by
  a plain HTTPS client.
- Post-handshake key updates (`KeyUpdate` messages, the `"traffic upd"`
  label) — a future enhancement if long-lived connections ever need it.
- TLS 1.2's PRF (RFC 5246 §5) — a completely different key-derivation
  algorithm, explicitly deferred to its own future sub-project alongside
  TLS 1.2 protocol support generally (per this branch's own earlier
  decomposition decision to build TLS 1.3 first).
- Transcript-hash accumulation itself (hashing the running concatenation
  of handshake messages) — the handshake sub-project's responsibility;
  this sub-project only consumes already-computed transcript hash values.
- Any TLS record-layer, handshake-message-parsing, or network I/O
  concern — this sub-project is pure key-derivation math with no protocol
  awareness.

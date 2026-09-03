<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-27-madaros-ecdsa-p384-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-27-madaros-ecdsa-p384-design
-->

# ECDSA P-384 Certificate-Chain Verification Support

## Background

D15/D20 (`docs/handoff/souc_v0800_defects.md`) traced a real, live TLS
handshake failure against Cloudflare's DNS-over-HTTPS endpoint
(`1.1.1.1:443`) to its final root cause: the certificate chain's
intermediate (`CN=SSL.com SSL Intermediate CA ECC R2`) uses a **P-384**
public key, signing the leaf with `ecdsa-with-SHA384`. This codebase's
X.509/ECDSA stack only implements the P-256 curve — parsing that
intermediate's public key already returns `PUBKEY_ALG_UNKNOWN`, and
`x509_verify_signature` correctly (per D20's new algorithm-confusion
guard) refuses to run a P-384 key through P-256 group arithmetic. The
chain is therefore rejected, and `conclave-search`'s DNS-over-HTTPS
resolution stage (`search/dns_resolve.sio`, a separate repo) can never
succeed against a real-world CA using this common configuration.

**This is not a hypothetical curve to add "for completeness."** P-384 is
one of the three NIST curves TLS libraries commonly issue from, and this
exact configuration is live, in production, at one of the most-used public
DNS resolvers on the internet.

## Goals

- `x509_verify_chain` can validate a certificate signed by a P-384 key
  with `ecdsa-with-SHA384` (the observed real-world case).
- `1.1.1.1:443`'s real certificate chain validates end-to-end via
  `tls_connect` where it previously failed with `CHAIN_ERR_BAD_SIGNATURE`.
- No regression to the existing P-256 path, and no weakening of the
  algorithm-confusion guard D20 just added (a P-384-keyed certificate must
  still never be run through P-256 arithmetic, and vice versa).

## Non-goals

- TLS 1.3's `CertificateVerify` signature schemes (`ecdsa_secp384r1_sha384`,
  0x0503) — the real motivating case's own LEAF key is P-256 (confirmed by
  D20's own live measurement), so the handshake's own proof-of-possession
  step never needs a P-384 signature. Only the offline chain-signature
  verification (an issuer's key signing a subordinate certificate) needs
  P-384 here. Add TLS 1.3 P-384 `CertificateVerify` support separately if
  a real case for it ever surfaces.
- P-521 or any other curve. Out of scope until a real case demands it.
- RSA-in-the-same-chain interactions — unaffected by this change.

## Architecture

### 1. Curve arithmetic — no new code needed, only new parameters

`stdlib/crypto/ecdsa_p256.sio`'s point-arithmetic functions
(`ec_point_add(a, b, p)`, `ec_scalar_mul(k, point, curve_a, p)`,
`ec_mod_inverse`, `ec_is_infinity`) already take the field prime and curve
coefficient as parameters — they perform generic short-Weierstrass-curve
arithmetic, not P-256-specific arithmetic. `BigInt`
(`stdlib/bignum/bigint.sio`) has 512 sixteen-bit limbs (8192-bit capacity),
already exercised by this codebase's RSA-4096 support — a 384-bit value is
not a capacity concern.

This means P-384 support needs: new curve-parameter constants, a new
digest-truncation entry point, and new public-key parsing — **not** new
point/field arithmetic.

### 2. New file: `stdlib/crypto/ecdsa_p384.sio`

Mirrors `stdlib/crypto/ecdsa_p256.sio`'s own structure (a new file, not
appended to the P-256 file, matching this codebase's one-curve-per-file
convention). Exposes:

```sio
pub fn p384_field_prime() -> BigInt
pub fn p384_curve_a() -> BigInt
pub fn p384_curve_b() -> BigInt
pub fn p384_order() -> BigInt
pub fn p384_base_point() -> EcPoint
pub fn ecdsa_p384_verify_sha384(pubkey: &EcPoint, message: &RawBuf, message_len: i64, der_signature: &RawBuf, der_signature_len: i64) -> bool with IO
```

`EcPoint` (`{ x: BigInt, y: BigInt }`) is already curve-generic — imported
from `crypto::ecdsa_p256::*`, not redefined.

**The DER-signature-parsing and curve-math body must be its own,
generic-over-`(a, p, n)` shared function** (mirroring D20's own
`ecdsa_p256_verify_with_e` refactor pattern exactly) — do not copy-paste
`ecdsa_p256_verify`'s body into a P-384 variant; both curves' verify
functions should funnel through one shared core parameterized by curve
constants and a pre-computed digest integer `e`, so the security-critical
DER-parsing/range-check/curve-math steps cannot silently diverge between
curves.

**Digest handling**: SHA-384's 48-byte output exactly matches P-384's
384-bit curve order — no truncation needed (the same situation SHA-256
has with P-256, per `ecdsa_p256_verify`'s own existing comment). Pack all
48 bytes big-endian into a 24-limb `BigInt`, mirroring the existing
32-byte/16-limb `p256_e_from_be32` pattern exactly, but sized for 24
limbs.

### 3. Curve constants — MUST be independently re-derived and verified, not copied from this spec blindly

This spec's own author (an AI assistant) computed the following NIST
P-384 (secp384r1) constants and verified them three independent ways
before writing them here: (a) the field prime `p` was derived from its
defining formula `p = 2^384 - 2^128 - 2^96 + 2^32 - 1`, not typed from
memory; (b) `b`, `Gx`, `Gy` were checked against each other by confirming
the curve equation `Gy² ≡ Gx³ + a·Gx + b (mod p)` holds exactly; (c) the
order `n` was checked for primality and for satisfying the Hasse bound
`|p + 1 - n| ≤ 2√p` relative to `p`. All three checks passed. **Despite
this, the implementing task MUST independently re-verify every constant
below against a real, live source** (e.g. `openssl ecparam -name
secp384r1 -param_enc explicit -text -noout`, or a from-scratch derivation
using the same method) before trusting them — a transcription error in a
hardcoded elliptic-curve constant is exactly the kind of defect that can
look completely fine (compiles, passes a self-consistency test) while
being cryptographically wrong or exploitable. Do not skip this
verification step because the numbers are already provided here.

Each constant below is a 24-element `[u16; 24]` little-endian limb array
(limb 0 = least significant 16 bits), matching `stdlib/crypto/
ecdsa_p256.sio`'s own `build_bigint_16`/limb-ordering convention exactly,
just sized for 384 bits (24 × 16 = 384) instead of 256:

```
p384_field_prime (p):
  [65535, 65535, 0, 0, 0, 0, 65535, 65535, 65534, 65535, 65535, 65535,
   65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
   65535, 65535]

p384_curve_a (a = p - 3):
  [65532, 65535, 0, 0, 0, 0, 65535, 65535, 65534, 65535, 65535, 65535,
   65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
   65535, 65535]

p384_curve_b (b):
  [10991, 54252, 51437, 10885, 53661, 35374, 14733, 50774, 34650, 20499,
   2191, 788, 16658, 65153, 40046, 6173, 11545, 58360, 1387, 39054, 59364,
   57918, 12199, 45873]

p384_base_point Gx:
  [2743, 29302, 24120, 14932, 10604, 48981, 62045, 21762, 10808, 33364,
   16864, 23031, 39832, 35751, 15202, 28189, 44404, 62240, 50974, 36529,
   1335, 48779, 51746, 43655]

p384_base_point Gy:
  [3679, 37098, 7548, 31299, 33181, 7550, 45518, 2656, 47296, 46576,
   12563, 59866, 5244, 10394, 7613, 63732, 56361, 37522, 39103, 23966,
   11375, 38438, 56906, 13847]

p384_order (n):
  [10611, 52421, 6506, 60652, 42874, 18608, 3506, 22554, 11743, 62519,
   19841, 51043, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
   65535, 65535, 65535, 65535]
```

`BigInt` needs a `build_bigint_24` helper (a one-line copy of the existing
`build_bigint_16` sized for 24 limbs) — add it alongside, not instead of,
`build_bigint_16`.

### 4. X.509 OID recognition — `stdlib/x509/oid.sio`

secp384r1's curve-identifier OID lives under a **different arc** than
P-256's (`1.2.840.10045.3.1.7`): it is `1.3.132.0.34` (SEC 2's own
`ansip384r1`/`secp384r1` identifier, arc `1.3.132`, not
`1.2.840.10045.3.x`). DER-encode this the same way
`oid_prime256v1()`/`oid_ec_public_key()` already encode their own OIDs
(base-128 arc encoding per X.690 §8.19) — **derive and verify this
encoding independently** (e.g. against a real
`openssl asn1parse -in <a-real-p384-cert.pem> -i` dump of an actual P-384
certificate's `SubjectPublicKeyInfo.algorithm.parameters` field, which
will show the raw OID bytes directly), not by trusting a single derivation
attempt. Add:

```sio
pub fn oid_secp384r1() -> [u8; 5]  // DER content bytes only (no tag/length header), matching oid_prime256v1()'s own convention: 2B 81 04 00 22 -- confirm exact bytes independently
```

### 5. X.509 public-key parsing — `stdlib/x509/cert.sio`

- Add `pub const PUBKEY_ALG_EC_P384: i32 = 3` (the next unused value after
  the existing `RSA=0`, `EC_P256=1`, `UNKNOWN=2`).
- In the same code path that currently recognizes `oid_ec_public_key()` +
  `oid_prime256v1()` and sets `PUBKEY_ALG_EC_P256`, add a parallel branch:
  `oid_ec_public_key()` + `oid_secp384r1()` → `PUBKEY_ALG_EC_P384`.
- The EC point itself is encoded the same uncompressed-point way P-256
  already parses (a leading `0x04` byte, then the X coordinate, then the
  Y coordinate) — P-256 uses 32+32 bytes after the `0x04` prefix (65 bytes
  total); P-384 uses 48+48 bytes (97 bytes total). Reuse the existing
  point-parsing logic's *shape*, parameterized by coordinate width, rather
  than writing a second, separately-maintained parser — check how the
  existing P-256 path structures this parsing before deciding exactly how
  to share it (a shared helper taking the coordinate byte-width as a
  parameter is the likely shape, but read the real code first).
- `Certificate`'s own EC-key storage field (`ec_public_key: EcPoint` or
  similar — check its real name) already holds an `EcPoint`, which is
  curve-generic (`BigInt` x/y, not fixed-width) — no struct change needed
  there, only the algorithm-tag constant and the parsing branch.

### 6. Signature verification wiring — `stdlib/x509/cert.sio`

In `x509_verify_signature`, alongside the existing
`is_ecdsa_sha384_sig_alg` branch (which currently only accepts
`PUBKEY_ALG_EC_P256` issuers, per D20's own new guard), add handling for
a P-384 issuer:

```sio
if is_ecdsa_sha384_sig_alg && issuer.public_key_algorithm == PUBKEY_ALG_EC_P384 {
    // ... same TBS-copy / signature-buffer setup already used ...
    return ecdsa_p384_verify_sha384(&issuer.ec_public_key, &tbs_copy, cert.tbs_len, &esig_buf, cert.outer_signature_len as i64)
}
```

The existing algorithm-confusion guard (`is_ecdsa_sha384_sig_alg &&
issuer.public_key_algorithm != PUBKEY_ALG_EC_P256` → `return false`) must
be corrected to also accept `PUBKEY_ALG_EC_P384` as a valid pairing for
`ecdsa-with-SHA384` (it currently only names P-256, since P-384 support
didn't exist when D20 wrote it) — **not removed**, since the guard's whole
purpose (never running an issuer's real key through the wrong curve's
arithmetic) must keep holding for both curves independently: a P-256 key
must still never be run through P-384 math, and vice versa.

## Data flow

`x509_parse_certificate` (unchanged in its own DER-walking logic beyond
the new OID branch) → `Certificate.public_key_algorithm` now able to read
`PUBKEY_ALG_EC_P384` for a real P-384 issuer → `x509_verify_chain`'s
existing signature-verification step → `x509_verify_signature`'s new
P-384 branch → `ecdsa_p384_verify_sha384` → shared curve-math core
(parameterized by P-384's own `a`, `p`, `n`) → `CHAIN_OK` for a genuinely
valid chain, `CHAIN_ERR_BAD_SIGNATURE` for a genuinely invalid one — same
external behavior contract as the P-256 path, just reaching it via P-384
arithmetic.

## Error handling

- An OID that looks like an EC key but isn't recognized as either
  `prime256v1` or `secp384r1` continues to degrade to
  `PUBKEY_ALG_UNKNOWN`, matching the existing fail-closed convention (an
  unknown curve is never silently treated as P-256 or P-384).
- The algorithm-confusion guard must reject EVERY curve/algorithm
  mismatch explicitly enumerated in Section 6 — this is a real security
  property (D20's own PoC showed what happens when it's missing:
  RSA/P-256 confusion or, per this spec's own motivating case, running a
  P-384 key's bytes through P-256 group arithmetic, which corrupts
  silently rather than erroring).

## Testing strategy

- A real `openssl`-generated P-384 self-signed root + a leaf/intermediate
  it signs with `-sha384`, verified independently by `openssl verify`,
  exactly matching D20's own testing methodology (`tests/run-pass/
  x509_chain_ecdsa_sha384_signature.sio`'s own pattern, but with a P-384
  issuer key instead of P-256).
- **Adversarial tests, matching D20's own rigor**: mutate the truncation/
  digest-packing logic deliberately (there should be none needed here
  since SHA-384/P-384 need no truncation, but confirm a flipped signature
  byte and a wrong issuer key are both correctly rejected) — a passing
  positive test alone proves nothing without a corresponding test proving
  the negative case is caught.
- A real end-to-end check against `1.1.1.1:443` if network-reachable at
  implementation time — `tls_connect` should return `0` where it
  previously returned `-8` (`TLS_CONNECT_ERR_CERT_CHAIN`, tracing back to
  `CHAIN_ERR_BAD_SIGNATURE` on the P-384 intermediate).
- Full regression run: `x509`, `tls`, `crypto`, `pem`, `asn1`, `bignum`
  test-suite prefixes, with a clean-tree baseline count taken first
  (`git stash -u` before, restore after) so "same failures before and
  after" is a real comparison, not an assumption.

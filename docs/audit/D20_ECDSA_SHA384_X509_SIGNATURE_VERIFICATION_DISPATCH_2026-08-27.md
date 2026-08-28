<!-- docs:meta
topic_id: repo.docs.audit.d20-ecdsa-sha384-x509-signature-verification-dispatch-2026-08-27
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.d20-ecdsa-sha384-x509-signature-verification-dispatch-2026-08-27
-->

# Forensic dispatch — D15's `ecdsa-with-SHA384` fail-closed branch is closed, and D15's own root cause for `1.1.1.1` is corrected

**Filed:** 2026-08-27 · **Status:** the SHA-384 signature-verification gap is
**FIXED and proven against a real, openssl-generated, openssl-verified
certificate chain**; D15's stated root cause for Cloudflare's `1.1.1.1` is
**partially corrected by measurement** — the residual blocker there is a
**P-384 issuer key**, a different curve, not a different hash ·
**Protocol:** CLAUDE.md §8

**Related:** D15 (the dispatching entry — its `iPAddress`-SAN fix stands; its
root-cause statement for `1.1.1.1` is corrected here, not deleted), D17 (the
EC trust-anchor algorithm-awareness fix this extends), D18/D19 (the "a bare
−N is not evidence — instrument the real result" discipline applied here)

---

## Summary

`stdlib/x509/cert.sio`'s `x509_verify_signature` returned `false`
unconditionally for any certificate signed `ecdsa-with-SHA384`, with an
inline comment recording that `ecdsa_p256_verify` had only ever built a
SHA-256 hashing path. That branch is now a real verification path.

Three outcomes, in descending order of confidence:

1. **Fixed and proven.** `ecdsa-with-SHA384` signatures made by a **P-256**
   issuer key now verify, end to end through `x509_verify_chain`, against a
   real chain that OpenSSL 3.5.6 generated and independently confirms
   (`openssl verify … leaf_cert.pem: OK`). Pinned as
   `tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`.
2. **The truncation rule is proven load-bearing, not merely asserted.** Two
   deliberate mutations of the FIPS 186-4 §6.4.1 truncation both make the
   new test fail (see *Adversarial verification* below). A test that would
   pass under a wrong truncation would have proven nothing.
3. **D15's root cause for `1.1.1.1` was incomplete, and the fix does not
   unblock it.** Measured, not inferred: the SSL.com intermediate that signs
   Cloudflare's leaf holds a **P-384** key. Verifying that link needs P-384
   arithmetic, which this codebase does not implement. `tls_connect` against
   `1.1.1.1:443` still returns `-8`, correctly and fail-closed.

---

## The defect

`stdlib/x509/cert.sio` (~line 1566, before this change):

```sio
if is_ecdsa_sha384_sig_alg {
    // ecdsa-with-SHA384 is a REAL gap, not simply "unsupported": Task 2
    // of this plan only built ecdsa_p256_verify's SHA-256 hashing path
    // (it hashes internally with sha256(), unconditionally). ...
    return false
}
```

This was a correct, deliberate fail-closed decision at the time it was
written — the alternative (dispatching SHA-384 data at a SHA-256 verifier)
would have been worse. But per **RFC 5480 §2**, an X.509 signature's hash
algorithm and the issuer key's curve are **independent**: a P-256 issuer key
signing with SHA-384 is valid and deployed. Every such chain was rejected.

Note the contrast with TLS 1.3, which is why only one of the two call sites
changed (both were read before deciding):

| Call site | Keys hash and curve together? | Changed? |
|---|---|---|
| `stdlib/x509/cert.sio:~1616` (certificate chain) | No — RFC 5480 §2 makes them independent | **Yes** |
| `stdlib/tls/client.sio:685` (`CertificateVerify`) | Yes — RFC 8446 §4.2.3 signature schemes couple them | No |

TLS 1.3's `ecdsa_secp256r1_sha256` (0x0403) and `ecdsa_secp384r1_sha384`
(0x0503) are single tokens naming curve *and* hash. The SHA-384 scheme
demands a **P-384 key**, so it is an unimplemented *curve*, not this gap.
`verify_certificate_signature` correctly has no branch for it and falls
through to `false`. Nothing to fix there.

## The fix

**`stdlib/crypto/ecdsa_p256.sio`** — refactored, not duplicated. The DER
`SEQUENCE { r, s }` parsing, the `1 ≤ r,s < n` range checks and the curve
arithmetic moved into one private `ecdsa_p256_verify_with_e(…, e: &BigInt)`,
which takes the digest integer already computed and already truncated. Two
public entry points differ **only** in how they produce `e`:

- `ecdsa_p256_verify` — `e = SHA-256(m)`, all 32 bytes (unchanged behaviour).
- `ecdsa_p256_verify_sha384` — `e = leftmost 32 of SHA-384(m)`'s 48 bytes.

Both funnel their 32 bytes through one `p256_e_from_be32`, so the
big-endian/limb-packing convention cannot diverge between paths either. The
security-reviewed steps exist in exactly one copy.

**The truncation rule**, which is the entire substance of the new function.
FIPS 186-4 §6.4.1 (identically SEC1 v2 §4.1.3 step 3): when the hash is
**longer** than `bitlen(n)`, take the **leftmost** (most significant)
`bitlen(n)` bits. For P-256, `bitlen(n) = 256 = 32` bytes, so
`e = digest[0..32)` and `digest[32..48)` is **discarded**. Not the rightmost
32 bytes; not the full 384-bit value reduced mod n; not a right-shift. Because
`bitlen(n)` is exactly byte-aligned for P-256, the byte slice is the complete
rule with no residual bit-shift. `e` may exceed `n`; that is expected and
correct, and matches the SHA-256 path, which also does not reduce.

**`stdlib/x509/cert.sio`** — the fail-closed branch is replaced by a real
call, and the two ECDSA paths now share the TBS-copy and signature-buffer
handling verbatim, branching only on which verifier to call. The
algorithm-confusion guard was extended in the same shape as the existing
SHA-256 one:

```sio
if is_ecdsa_sha384_sig_alg && issuer.public_key_algorithm != PUBKEY_ALG_EC_P256 {
    return false
}
```

This guard is what keeps outcome 3 below safe, and it is load-bearing in
production traffic today — not a theoretical precaution.

---

## Verification

### 1. Against a real chain OpenSSL generated and OpenSSL validates

Self-consistency (sign and verify with our own code) would not catch a
wrong-end truncation, so the fixture is generated and independently
confirmed by a second implementation. OpenSSL 3.5.6:

```
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -sha384 \
  -keyout root_key.pem -out root_cert.pem -days 3650 -nodes \
  -subj "/CN=sha384-root-test" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"
openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout leaf_key.pem -out leaf.csr -nodes -subj "/CN=sha384-leaf-test"
openssl x509 -req -in leaf.csr -CA root_cert.pem -CAkey root_key.pem \
  -CAcreateserial -out leaf_cert.pem -days 3650 -sha384 -extfile leaf_ext.cnf

$ openssl verify -CAfile root_cert.pem leaf_cert.pem
leaf_cert.pem: OK
```

Both certificates: `Signature Algorithm: ecdsa-with-SHA384`, key
`prime256v1 (P-256)`. The DER of both is pinned byte-for-byte in
`tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`, which asserts:

| Assertion | Result |
|---|---|
| Root's own self-signature (SHA-384) verifies | pass |
| Leaf verifies against the **root's** key (the cross-certificate case) | pass |
| One flipped bit in the leaf's `s` INTEGER is rejected | pass |
| Leaf does **not** verify against the wrong issuer key | pass |
| Full `x509_verify_chain` → `CHAIN_OK` (was `CHAIN_ERR_BAD_SIGNATURE`) | pass |
| Corrupted leaf → not `CHAIN_OK` | pass |

### 2. Adversarial verification — the mutations that must fail

A test that passes under a *wrong* truncation is worthless, so the truncation
was deliberately broken twice and the test re-run:

| Mutation | Meaning | Result |
|---|---|---|
| `digest[ti + 16]` | rightmost 32 bytes instead of leftmost | **fails** at checkpoint 6 |
| `digest[ti + 1]` | leftmost 32 shifted one byte | **fails** at checkpoint 6 |
| `digest[ti]` (correct) | FIPS 186-4 §6.4.1 | passes |

The test is therefore sensitive to exactly the property it claims to pin.

### 3. Regression suite — clean-tree baseline, identical invocation

Both runs used the worktree's own compiler and bounded parallelism
(`SOUC_BIN=$PWD/bin/souc SOUNIO_TEST_JOBS=4`), per the harness hazards
recorded during D17/D18.

| Pattern | Baseline (clean tree) | After |
|---|---|---|
| `x509` | 24 pass / 0 fail / 3 skip | **26** pass / 0 fail / 3 skip |
| `tls` | 8 / 0 / 0 | 8 / 0 / 0 |
| `crypto` | 1 / 0 / 0 | 1 / 0 / 0 |
| `pem` | 2 / 0 / 0 | 2 / 0 / 0 |

No regressions; the deltas are this dispatch's two new tests.

---

## The correction to D15 — `1.1.1.1` is blocked on P-384, not on SHA-384

D15 recorded the SHA-384 hash gap as the reason Cloudflare's `1.1.1.1:443`
chain failed. **That is half the story, and the smaller half.** After the fix,
a live probe still returns `-8`:

```
$ souc run tests/interop/tls_connect_1111_sha384_probe.sio
probe: trust store roots: 150
probe: tls_connect rc: -8
```

Rather than infer why — the mistake D15, D17 and D18 each made once — the real
chain was fetched and the real certificates fed through the parser:

```
$ openssl s_client -connect 1.1.1.1:443 -servername cloudflare-dns.com -showcerts

  leaf          CN=cloudflare-dns.com                  key P-256
                Signature Algorithm: ecdsa-with-SHA384
  intermediate  CN=SSL.com SSL Intermediate CA ECC R2  key P-384
                Signature Algorithm: ecdsa-with-SHA384
```

Measured directly on those exact bytes:

```
leaf parse status (0 == X509_OK):          0
intermediate parse status (0 == X509_OK):  0
leaf public_key_algorithm  (1 == EC_P256): 1
INTERMEDIATE public_key_algorithm:         2   (PUBKEY_ALG_UNKNOWN)
x509_verify_signature(leaf, issuer=intermediate): FALSE
```

Both certificates parse cleanly — this is **not** a parser failure. The
intermediate's key is **P-384**, which `x509_parse_certificate` deliberately
declines to classify as P-256 (it matches the named-curve OID against
`prime256v1` and falls through to `UNKNOWN` for every other curve). The
leaf's signature was produced by that P-384 key, so verifying it requires
P-384 arithmetic. `stdlib/crypto/ecdsa_p256.sio` implements P-256 only.

**The `false` is correct.** It is the new algorithm-confusion guard refusing
to run a P-384 key through P-256 group arithmetic — the wrong order, the
wrong field prime, and a verdict that would mean nothing. Pinned against the
real Cloudflare certificates in
`tests/run-pass/x509_ecdsa_sha384_p384_issuer_rejected.sio`.

### What this means

- The fix is **real and useful**: P-256 issuer + SHA-384 is a valid, deployed
  configuration, now supported and proven against OpenSSL.
- It is **not sufficient for `1.1.1.1`**, and this dispatch does not claim
  otherwise. DNS-over-HTTPS against that endpoint stays blocked.
- The remaining work is **P-384 curve support** (a new curve: field prime,
  order, generator, and `ecdsa_p384_verify`), which is a substantially larger
  piece of work than this one and is **not** filed as done anywhere.
- `tests/interop/tls_connect_1111_sha384_probe.sio` is retained as the
  re-runnable measurement of that residue; it flips from `-8` to `0` the day
  P-384 lands, and nothing else needs to change to notice.

## Residual risk

- **P-384 (and every other curve) remains unimplemented.** Chains anchored in
  EC roots that are not P-256 fail closed. Safe, but limiting: ECC roots at
  the major CAs are commonly P-384.
- **Constant-time behaviour is out of scope**, unchanged by this dispatch.
  These are signature *verification* paths operating on public values only,
  so the timing exposure is not secret-dependent.
- **The two ECDSA paths must stay merged.** The value of this refactor is
  that DER parsing, range checks and curve math exist in one copy. A future
  hash algorithm should add an entry point that computes `e`, never a second
  copy of `ecdsa_p256_verify_with_e`.

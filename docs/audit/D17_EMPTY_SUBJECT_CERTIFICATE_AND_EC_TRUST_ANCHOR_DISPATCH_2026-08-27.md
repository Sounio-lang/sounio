<!-- docs:meta
topic_id: repo.docs.audit.d17-empty-subject-certificate-and-ec-trust-anchor-dispatch-2026-08-27
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.d17-empty-subject-certificate-and-ec-trust-anchor-dispatch-2026-08-27
-->

# Forensic dispatch — the empty-Subject parsing defect does not exist; the EC trust-anchor bypass found while disproving it does

**Filed:** 2026-08-27 · **Status:** reported defect **NOT REPRODUCED** (hypothesis refuted by measurement, four configurations); a **separate, security-critical chain-validation bypass** found during the investigation is **RESOLVED** · **Protocol:** CLAUDE.md §8 · **Related:** D15 ([`docs/handoff/souc_v0800_defects.md`](../handoff/souc_v0800_defects.md), "two wrong diagnoses of the same handshake failure"), and [`tests/run-pass/x509_chain_forged_root_dn_collision.sio`](../../tests/run-pass/x509_chain_forged_root_dn_collision.sio) (the RSA half of the bypass, closed 2026-08-25)

**Commits:** see `git log --oneline -- stdlib/x509/trust_store.sio stdlib/x509/chain.sio`

---

## Summary

Two findings, and they are not the same finding.

1. **The reported defect does not reproduce.** The report was that a real,
   currently-valid `cert-manager` leaf with an **empty Subject DN** is
   rejected by `x509_verify_chain` with `CHAIN_ERR_EXPIRED` (−2), on the
   hypothesis that a zero-length Subject `SEQUENCE` throws off
   `x509_parse_certificate`'s byte-offset computation for the Validity
   field. **That hypothesis is structurally impossible and was measured to
   be false.** In a `TBSCertificate` the Validity field *precedes* the
   Subject (RFC 5280 §4.1.2), so a zero-length Subject cannot move the
   Validity offsets — and in fact the certificate parses with its exact,
   correct dates and verifies to `CHAIN_OK` in every configuration tested,
   up to and including a live TLS 1.3 handshake.

2. **A different, security-critical defect was found while disproving it,
   in the code the reporter was exercising.** `chain_is_trusted_root`'s
   "byte-identical public key" requirement — added 2026-08-25 to close a
   trust-anchor forgery bypass — compares **only** `modulus` and
   `public_exponent`, which `x509_parse_certificate` **never populates for
   an EC certificate**. For every EC certificate the check was
   `bigint_cmp(zero, zero) == 0` twice: vacuously true. The bypass that fix
   closed for RSA was still **fully open for EC**, which is exactly the key
   type this whole sub-project uses (P-256 was chosen deliberately to avoid
   D15's ECDSA-SHA384 gap). Demonstrated with an executed proof-of-concept
   and fixed.

---

## Part 1 — the reported defect, refuted

### The certificate

Live `searxng-https-tls` secret, namespace `beagle`, captured 2026-08-27:

```bash
kubectl get secret searxng-https-tls -n beagle \
  -o jsonpath='{.data.tls\.crt}' | base64 -d > leaf.pem
kubectl get secret searxng-https-tls -n beagle \
  -o jsonpath='{.data.ca\.crt}'  | base64 -d > ca.pem
```

```
subject=                                    <- empty, cert-manager default
issuer=CN=conclave-search-internal-ca
notBefore=Aug 27 01:09:33 2026 GMT
notAfter =Nov 25 01:09:33 2026 GMT
```

Real, RFC 5280-conformant shape: §4.1.2.6 permits an empty subject provided
`subjectAltName` is present and critical, which it is (critical iPAddress
SAN, `10.96.250.10`). The leaf's PEM was verified byte-identical to the one
quoted in the dispatch brief.

### The structural refutation

`openssl asn1parse -in leaf.pem -i`, the actual byte layout:

```
    47:d=2  hl=2 l=  38 cons:   SEQUENCE          <- Issuer
    87:d=2  hl=2 l=  30 cons:   SEQUENCE          <- Validity
    89:d=3  hl=2 l=  13 prim:    UTCTIME  :260827010933Z
   104:d=3  hl=2 l=  13 prim:    UTCTIME  :261125010933Z
   119:d=2  hl=2 l=   0 cons:   SEQUENCE          <- Subject, EMPTY
   121:d=2  hl=2 l=  89 cons:   SEQUENCE          <- SubjectPublicKeyInfo
```

The empty Subject sits at **offset 119**; the dates are read at **89 and
104**. `x509_parse_tbs_after_serial` (`stdlib/x509/cert.sio:516–539`) walks
the fields in wire order — signature AlgorithmIdentifier, issuer, validity,
**then** subject — so the Subject's length is not an input to any offset the
date decode uses. The proposed mechanism cannot occur in this direction.

The one thing an empty Subject *could* plausibly break is everything read
**after** it, if a zero-length constructed Name were mishandled. It is not:
`der_enter` on a `30 00` yields `pos == end`, `x509_parse_name`'s
`der_at_end` loop body never executes, and the function returns
`(name{count:0}, X509_OK)`.

### The measurements

Four configurations, in increasing fidelity to production. All four were run
live on this worktree's `bin/souc` (Madaros v0.80.0):

| # | Configuration | Result |
|---|---|---|
| 1 | `x509_parse_certificate` on the leaf's DER, standalone | `status = 0`, `not_before_unix = 1787792973`, `not_after_unix = 1795568973` — **both exactly correct** |
| 2 | `x509_verify_chain`, trust store = the real internal CA only | **`CHAIN_OK`** |
| 3 | `x509_verify_chain`, trust store = `trust_store_load()` (150 real Debian roots) **+** the internal CA appended (151 entries) | **`CHAIN_OK`** |
| 4 | Full **live TLS 1.3 handshake** via `tls_connect`, against `openssl s_server` serving an equivalently-shaped empty-Subject P-256 leaf with an IP SAN | **`err = 0`** |

The expected values in row 1 were derived independently (Python `datetime`,
UTC: `1787792973` / `1795568973`), not by re-running this parser.

Configuration 4 is worth stating precisely, because it is the one that
covers the whole path the reporter was on: a locally-generated P-256 CA, an
**empty-Subject** leaf with `subjectAltName=critical,IP:127.0.0.1`, served
by `openssl s_server -tls1_3`, connected to by `tls_connect` with the CA as
the sole trust anchor and `now_unix()` from the real `time(2)` syscall. The
handshake completed, which means the Certificate-message decode, the
CertificateVerify signature check, `x509_verify_chain`, and the
IP-SAN hostname check all passed against an empty-Subject certificate.

### What the −2 most likely was

Not reproducible from here — the failing run's own output was not available
to this investigation, and its trust-store configuration is unknown. What
*is* measurable, and is the same lesson D15 filed, is that **`-2` is not a
unique value in this stack.** `grep -rn "= 0 - 2$" stdlib/` returns ten
distinct constants:

| Constant | Module |
|---|---|
| `CHAIN_ERR_EXPIRED` | `x509/chain.sio` |
| `X509_ERR_TOO_MANY_ENTRIES` | `x509/cert.sio` |
| `OCSP_ERR_CERT_ID_MISMATCH` | `x509/ocsp.sio` |
| `TRUST_STORE_ERR_TOO_MANY_CERTS` | `x509/trust_store.sio` |
| `PEM_ERR_NO_END_MARKER` | `x509/pem.sio` |
| `DER_ERR_BAD_LENGTH_FORM` | `asn1/der.sio` |
| `ERR_CONNECT` | `net/socket.sio` |
| `TLS_CONNECT_ERR_RANDOM` | `tls/client.sio` |
| `TLS_IO_ERR_RECORD_OVERFLOW` | `tls/client.sio` |

A bare `-2` printed by a caller that does not also name the function it came
from is therefore not evidence for `CHAIN_ERR_EXPIRED` specifically. Two
further measured data points bearing on the reported symptom:

* Feeding `x509_verify_chain` a `now_unix` **earlier** than `notBefore`
  yields `CHAIN_ERR_NOT_YET_VALID` (**−3**), not −2. A clock that is merely
  wrong-early does not produce the reported code.
* A `GeneralizedTime`-encoded validity field **does** produce
  `CHAIN_ERR_EXPIRED` at the correct current time — the parser implements
  UTCTime only and `parse_utc_time_to_unix` returns `-1` for any other
  length. This is already documented at
  `tests/interop/tls_client_handshake_loopback.sio:38–46`, and remains the
  single most plausible way to get a genuine, spurious `CHAIN_ERR_EXPIRED`
  out of this stack. It does not apply to the reported certificate (both of
  its dates are UTCTime, confirmed by `asn1parse` above), but it *would*
  apply to any certificate in the chain dated at or past 2050 — including a
  long-lived root in whatever bundle the failing process actually loaded.

**Recommended next step for whoever reproduces the original failure:** print
the `x509_verify_chain` return value alongside the name of the function that
produced it, and print `leaf.not_before_unix` / `leaf.not_after_unix` and
`now_unix` at the same site, in the same process. Configuration 3 above is a
ready-made harness for that.

---

## Part 2 — the EC trust-anchor bypass (real, fixed)

### Root cause

`chain_is_trusted_root` (`stdlib/x509/chain.sio:139`) decides whether a
terminal certificate is a trust anchor. Since the 2026-08-25 final-plan-review
Finding 1, it requires a trust-store entry with **both** a matching Subject DN
**and** a byte-identical public key — because Subject DNs are public
information, so DN equality alone lets an attacker forge a self-signed
certificate carrying a real root's DN and their own key.

The key comparison lives in `trust_store_has_matching_key_for_subject`, and
before this fix it was:

```sounio
if bigint_cmp(&entry.modulus, modulus) == 0
    && bigint_cmp(&entry.public_exponent, exponent) == 0 { return true }
```

`x509_parse_certificate` populates `modulus`/`public_exponent` **only** on
the RSA branch. The EC branch (`stdlib/x509/cert.sio:627–646`) decodes a
P-256 key into `cert.ec_public_key` and leaves both BigInts at
`bigint_zero()`, as that branch's own trailing comment states. So for **every
EC certificate** the guard evaluated `bigint_cmp(zero, zero) == 0` twice and
returned true on nothing but a DN match — i.e. it silently degraded to the
exact pre-Finding-1 behaviour the fix was written to eliminate.

`PUBKEY_ALG_UNKNOWN` certificates (unsupported curve, compressed EC point,
RSA-PSS, Ed25519, …) had the same property: all three key fields zero.

### Proof of concept — executed, before the fix

Three P-256 certificates generated with `openssl`:

* **genuine root** `CN=d17-ec-root-collision-test`, key A — the only entry in
  the trust store
* **forged root** `CN=d17-ec-root-collision-test`, key B (the attacker's own),
  self-signed, `CA:TRUE` — identical Subject DN, byte-different key
* **forged leaf** `CN=victim.example`, issued by the forged root

The forged root is supplied as an *intermediate*, exactly as a hostile server
would present it in its TLS Certificate message.

```
verify result (0 = BYPASS, attacker root accepted):
0                                    <- CHAIN_OK, pre-fix
```

A complete chain-validation bypass: every signature in the accepted path is
the attacker's own, and the genuine root's key was never consulted.

### Fix

`stdlib/x509/trust_store.sio` — `trust_store_has_matching_key_for_subject`
now takes the whole candidate `&Certificate` instead of
`(subject, modulus, exponent)`, so the comparison can be algorithm-aware.
Three requirements, all on the **same** trust-store entry:

1. `public_key_algorithm` matches, **and**
2. Subject DNs match, **and**
3. that algorithm's *own* key material is byte-identical — `modulus` +
   `public_exponent` for RSA, `ec_public_key.x` + `.y` for P-256.

Any algorithm this parser does not decode is rejected up front, fail-closed:
such a certificate has all three key fields at zero, so any two of them would
compare equal to each other.

The rewrite also drops a `let entry = store.certs[i]` by-value bind — a
`Certificate` is ~46 KB and this loop runs over all ~150 system roots on
every DFS step, so that bind burned ~6.9 MB of never-reclaimed arena per
call, against this module's own ARENA DISCIPLINE note and D16.

Sole non-test caller updated: `chain.sio:145`.

### After the fix

```
verify result (0 = BYPASS, attacker root accepted):
-6                                   <- CHAIN_ERR_BAD_SIGNATURE, rejected
```

−6 rather than −1 because the DFS, no longer able to stop at the forged root,
goes on to try the genuine root as the forged root's issuer and rejects the
path at the signature check instead. Either code is a rejection; the pinned
property is "not `CHAIN_OK`".

---

## Regression tests added

| Test | Pins |
|---|---|
| `tests/run-pass/x509_empty_subject_certificate.sio` | The real, live empty-Subject `searxng-https-tls` leaf + its CA. Asserts the exact real dates (`1787792973` / `1795568973`), `subject.count == 0`, `issuer.count == 1`, and that everything **after** the empty Subject still decodes (P-256 SPKI, v3, 4 extensions) — then `CHAIN_OK` at a fixed timestamp inside the window, **and** `CHAIN_ERR_EXPIRED` one second past `notAfter`, so the positive assertion is a live check rather than one that never runs. |
| `tests/run-pass/x509_chain_forged_ec_root_dn_collision.sio` | The EC half of the forged-root bypass, mirroring the existing RSA test. Asserts the preconditions explicitly (both roots EC, both RSA fields equal-and-zero, DNs collide, EC keys differ) so the test cannot silently stop testing the thing it was written for. `-days 8400` fixtures, fixed `now_unix` — it does not rot. |

Both certificates in the first test are real live cluster bytes, not
hand-built fixtures. The second test's fixtures are locally generated because
an attack certificate has to be.

## Test evidence

```
$ bash scripts/run_sio_test_suite.sh x509         Pass: 24  Fail: 0  Skip: 3
$ bash scripts/run_sio_test_suite.sh tls          Pass:  7  Fail: 0  Skip: 0
$ bash scripts/run_sio_test_suite.sh pem          Pass:  2  Fail: 0  Skip: 0
$ bash scripts/run_sio_test_suite.sh trust_store  Pass:  2  Fail: 0  Skip: 1
```

Plus, re-run after the fix and unchanged: configuration 3 above (151-entry
real system trust store → `CHAIN_OK`) and configuration 4 (live TLS 1.3
handshake against an empty-Subject leaf → `err = 0`). The fix does not break
any legitimate EC trust anchor.

---

## Residual risk

* **The originally reported failure is still unexplained.** This dispatch
  refutes the stated *mechanism* with measurement; it does not explain
  whatever the reporter observed. If it recurs, instrument per the
  "recommended next step" above before re-opening a parser hypothesis.
* **Empty Subject DNs are mutually equal.** `x509_name_eq` compares
  `count` then entries, so any two empty Names compare equal. That is correct
  DN semantics, and `chain_is_trusted_root` is no longer reachable through it
  (the key check is now real for EC too) — but a trust store that ever
  contained an empty-Subject entry would make DN matching useless for every
  other empty-Subject certificate. No such entry exists in the Debian bundle;
  flagged, not fixed.
* **`GeneralizedTime` is still unimplemented.** Any certificate dated at or
  past 2050 silently yields `-1` from `parse_utc_time_to_unix` and reads as
  expired. Fail-closed, but indistinguishable from a genuine expiry, and it
  is the leading candidate for any *real* spurious `CHAIN_ERR_EXPIRED`.
  Pre-existing and unchanged by this dispatch.

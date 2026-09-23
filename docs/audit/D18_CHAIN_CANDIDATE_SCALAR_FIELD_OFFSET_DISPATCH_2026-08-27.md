<!-- docs:meta
topic_id: repo.docs.audit.d18-chain-candidate-scalar-field-offset-dispatch-2026-08-27
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.d18-chain-candidate-scalar-field-offset-dispatch-2026-08-27
-->

# Forensic dispatch — the live-handshake `CHAIN_ERR_EXPIRED` is a scalar-field offset defect, not a handshake-framing defect

**Filed:** 2026-08-27 · **Status:** reported symptom **ROOT-CAUSED and FIXED**; the handshake is **still blocked**, on a second and distinct instance of the same defect class, which is characterised here and **NOT fixed** · **Protocol:** CLAUDE.md §8 · **Related:** D17 (the immediately preceding investigation, whose refutation of the empty-Subject hypothesis this dispatch confirms and does not revisit), D15 (the "a bare −2 is not evidence" lesson), D12/D16 (arena cost), and CLAUDE.md §13's imported-module native path (D3/D4) family

**Commits:** see `git log --oneline -- stdlib/x509/chain.sio`

---

## Summary

The reported symptom is real and reproduces exactly as described: a live
`tls_connect` against the deployed `10.96.250.10:443` (SearXNG) returns
`TLS_CONNECT_ERR_CERT_CHAIN` (−8), and direct instrumentation of
`stdlib/tls/client.sio`'s own `chain_result` shows `CHAIN_ERR_EXPIRED` (−2),
for a certificate chain that is entirely inside its validity window and that
`openssl verify` accepts.

**The stated hypothesis — that `stdlib/tls/`'s TLS 1.3 Certificate-message
decoder mis-frames the certificate bytes before handing them to
`x509_parse_certificate` — is refuted by measurement.** The extraction is
byte-perfect. The defect is downstream, and it is not in the X.509 parser's
logic either: it is a **compiler defect in scalar-field access on large
structs in the imported-module native path**, which corrupts the *path
length* that chain verification loops over.

Three findings:

1. **Refuted:** the Certificate-message framing is correct to the byte.
2. **Root cause of the reported −2:** `ChainCandidate`'s trailing scalar
   field `len`, read back through an array element, returns an arena
   pointer instead of the stored value. `chain_verify_path` then receives a
   garbage `path_len` (~5×10⁵ instead of 2) and walks off the end of the
   11-slot path arrays into zeroed slots, whose `not_after_unix` is `0`, and
   `now_unix > 0` — so it reports `CHAIN_ERR_EXPIRED`. **Fixed.**
3. **Still blocking, same defect class, different struct:** with the −2
   removed, the handshake advances and then fails
   `CHAIN_ERR_BAD_SIGNATURE` (−6), because `x509_parse_certificate` returns
   a `Certificate` whose `tbs_start`/`tbs_len` are garbage (`255`/`44`
   against a true `4`/`295`) **when called from `stdlib/tls/handshake.sio`**,
   while other fields of the same struct value are correct. **Not fixed —
   characterised only.**

---

## Part 1 — the framing hypothesis, refuted by measurement

`decode_certificate_message` (`stdlib/tls/handshake.sio`) was instrumented to
report exactly what it hands to `x509_parse_certificate` on the live wire
bytes from `10.96.250.10:443`.

| Measurement | Expected | Observed |
|---|---:|---:|
| `cert_len` (whole Certificate message) | 398 | **398** |
| First 4 bytes of `leaf_buf` | `0x30 0x82 0x01 0x7D` | **48, 130, 1, 125** |
| `leaf_buf.cap` (`cert_data_len + 8`) | 393 | **393** |
| Byte-sum of `leaf_buf[0..385]` | 30515 | **30515** |
| `intermediate_count` | 0 | **0** |

The reference byte-sum and length come from the DER of the very certificate
the server presented, captured independently with
`openssl s_client -showcerts` and converted with `openssl x509 -outform DER`
(385 bytes, byte-sum 30515).

The framing also reconciles arithmetically with RFC 8446 §4.4.2, with nothing
left over:

```
  4  handshake header (msg_type + 3-byte length)
+ 1  certificate_request_context length (0x00)
+ 3  certificate_list length
+ 3  CertificateEntry cert_data length
+385 cert_data (the DER itself)
+ 2  per-entry extensions length (0x0000)
=398 == cert_len
```

**The decoder strips the per-entry 3-byte length prefix and the 2-byte
extensions block correctly, and passes a buffer starting exactly at the DER
`SEQUENCE` tag `0x30`.** There is no off-by-N. The hypothesis is closed.

The server sends **one** certificate — the leaf only, no intermediates — so
the chain is `leaf → conclave-search-internal-ca` where the CA is a trust
anchor. (Incidentally, that CA appears **twice** in the deployed bundle,
byte-identically: same SHA-256 fingerprint. That is a deployment quirk, it is
harmless, and it is why the DFS finds exactly two candidate paths.)

---

## Part 2 — root cause of the reported −2

### The measurements

Instrumenting `x509_verify_chain` and `chain_verify_path` directly, during
the live handshake:

| Measurement | Expected | Observed |
|---|---:|---:|
| `candidate_count` | 2 | **2** ✓ |
| `rc.len` at record time (plain local) | 2 | **2** ✓ |
| `depth` at record time | 2 | **2** ✓ |
| `candidates[i].len` immediately after `candidates[i] = rc` | 2 | **503099** ✗ |
| `candidates[0].len` straight out of the `len: 0` initialiser | 0 | **495821** ✗ |
| `candidates[0].bufs[0].cap` (built by `rawbuf_new(1)`) | 1 | **255** ✗ |
| `path_len` received by `chain_verify_path` | 2 | **504725** ✗ |
| index at which the validity loop reported expiry | — | **2** (past the real path) |
| `not_before_unix`/`not_after_unix` at that index | — | **0 / 0** |

The garbage values are not random: they track arena state. They change from
run to run, and the two candidates' values differ by a constant 428 in every
run observed (495821/496249, 502627/503055, 503099/503527, 504725/505153).
They are arena pointers read out of the element's own `bufs` array.

### The mechanism

`ChainCandidate` is
`{ certs: [Certificate; 11], bufs: [RawBuf; 11], len: i32 }`. `Certificate` is
a large struct (D16 measured ~46 KB, since a `[u8; N]` field costs 8N), so one
`ChainCandidate` is roughly 506 KB and the `[ChainCandidate; 16]` array is
~8 MB.

Scalar-field access through an element of that array resolves to a **wrong
byte offset that lands inside the element's own `bufs` array**. Both the read
and the write use the same wrong offset, so they are self-consistent with each
other — which is precisely why this is hard to see, and why the natural
repair is a trap:

```
candidates[0].len = 7      -->  reads back 7        (looks fixed)
candidates[0].bufs[0].cap  -->  1 before, 0 after   (silently corrupted)
candidates[0].bufs[10].cap -->  SEGFAULT
```

So "store the trailing field again after the struct assignment" appears to
work and in fact corrupts buffer pointers. It was measured and rejected.

Taking a **reference** to a whole array field (`&candidates[c].certs`,
`&candidates[c].bufs`) is *not* affected — those reads were correct, which is
why the leaf and the CA at path indices 0 and 1 passed their date checks with
exactly the right timestamps.

### Why this is live-handshake-only

This is why D17's investigation, which was correct in everything it measured,
could not see it: it exercised `x509_parse_certificate` and
`x509_verify_chain` from a **test module**, and got `CHAIN_OK` including
against the full 151-root store. The defect only appears in the
**cross-module / imported-module native path** that a real `tls_connect`
goes through. This is the family CLAUDE.md §13 tracks as D3/D4, and it is the
same class `stdlib/x509/chain.sio`'s own comment (around the
`x509_verify_chain` entry point) already recorded on 2026-08-25 — where
routing this call through `chain_build_candidates` made a known-good chain
return **`CHAIN_ERR_EXPIRED` (−2) on every run**. That note guessed it was "a
second, still-live instance of the same defect class". It was right, and this
dispatch identifies the mechanism: the candidate array's trailing scalar.

**Consequence for the test suite: the existing `tests/run-pass/x509_*` tests
structurally cannot catch this.** They pass, they passed before this fix, and
they will keep passing, because a single-module test program does not
reproduce the offset divergence. Coverage for this defect has to come from a
live or cross-module handshake, not from `tests/run-pass/`.

### The fix

`stdlib/x509/chain.sio`, `x509_verify_chain`: the inlined DFS no longer
materialises paths into `[ChainCandidate; 16]` at all. Each completed path is
verified **in place**, at the moment the DFS reaches a trusted root, while
`path_certs` / `path_bufs` / `depth` are still plain locals that read back
correctly.

Preserved exactly: DFS search order, the first-`CHAIN_OK` early return, the
hostname and OCSP checks on success, the `MAX_CHAIN_CANDIDATES` cap, and the
"deepest failure wins" (`best_depth`/`best_error`) reporting rule.

Removed as a side effect: the ~8 MB of never-reclaimed arena the 16-element
candidate array cost on every single verification (D12/D16's ledger).

`chain_build_candidates` — the non-shipping duplicate that still returns
`[ChainCandidate; 16]` — is deliberately left alone, per the same standing
instruction that kept the inlined copy in the first place. It is only reached
from its own single-module tests, where the defect does not manifest. **Any
future caller that reaches it across a module boundary will hit this same
defect**, and should be routed through `x509_verify_chain` instead.

### Before / after, live against `10.96.250.10:443`

```
before:  chain_result = -2   (CHAIN_ERR_EXPIRED, path_len = 504725)
after:   path_len = 2 -- correct; the -2 is gone
```

---

## Part 3 — the next blocker, characterised but NOT fixed

With the path length correct, verification proceeds and the handshake fails
further along:

```
DBG candidate_found_depth  2
DBG verify_result          -6      (CHAIN_ERR_BAD_SIGNATURE)
DBG candidate_found_depth  2
madaros: arena full                (exit 181, inside the second verification)
```

The `-6` is **not** a cryptographic gap. The chain is
`ecdsa-with-SHA256` over P-256 throughout — the combination this sub-project
deliberately chose to dodge D15's ECDSA-SHA384 hole — and
`openssl verify -CAfile <ca> <leaf>` returns `OK`.

The cause is the same defect class, one struct over. Instrumented inside
`decode_certificate_message`, **at the point `x509_parse_certificate`
returns**:

| Field of the returned `Certificate` | True value | Observed |
|---|---:|---:|
| `tbs_start` | 4 | **255** ✗ |
| `tbs_len` | 295 | **44** ✗ |
| `not_before_unix` | 1787792973 | **1787792973** ✓ |
| `not_after_unix` | 1795568973 | **1795568973** ✓ |

(True values from `openssl asn1parse`: the TBS `SEQUENCE` sits at offset 4
with `hl=4 l=291`, i.e. 295 bytes of TBS DER.)

Two things make this a distinct finding rather than a restatement of Part 2:

- It is corrupt **at the callee's return**, before any tuple, any array, and
  any struct copy. Printing the same fields again just before
  `decode_certificate_message` returns, and again in `tls_connect` after the
  6-tuple destructure, gives the identical garbage — so the large
  `(Certificate, RawBuf, [Certificate; 8], [RawBuf; 8], i32, bool)` tuple
  return is **not** the corrupting step, and is exonerated.
- The corrupt fields are `Certificate`'s **second and third** members, while
  much later members (`not_before_unix`, `not_after_unix`) are correct. So
  this is **not** a monotonic "layout diverges from field N onward"
  divergence, and it is not explained by a simple size disagreement.

`x509_verify_signature` uses `tbs_start`/`tbs_len` to select the bytes it
hashes. Given `255`/`44` it hashes the wrong 44 bytes, so the ECDSA check
fails — `-6` — and the bogus window is the most likely driver of the
arena blow-up on the following candidate as well.

**This is left open deliberately.** It is a compiler defect, not a stdlib
defect: per CLAUDE.md §8, `self-hosted/` is not patched ad hoc, and this
needs to be recorded as evidence and a proposed fix first. There is no
stdlib-level workaround comparable to Part 2's — the values are already wrong
when the stdlib receives them.

### Reproduction for whoever picks this up

```bash
# from inside the cluster network
kubectl exec -n beagle sounio-workspace-control-0 -c workspace-ide -- bash -c \
  'cd /workspace/.wt/conclave-search-tls-verify && \
   SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run /tmp/probe_searxng.sio'
```

with `/tmp/probe_searxng.sio` doing `trust_store_load()` then
`tls_connect(&host, host_len, 443, &store, now_unix())` against
`10.96.250.10`. Instrument `stdlib/tls/handshake.sio` to print
`parsed_cert.tbs_start` / `parsed_cert.tbs_len` immediately after
`x509_parse_certificate`. The trust store is read from the hardcoded
`/etc/ssl/certs/ca-certificates.crt`; the pod's copy is already
byte-identical (md5 `bcc0018701b98ed1518f879cf07b9448`, 152 entries) to the
one the correctly-configured `epistemic-search` pod uses.

`10.96.250.20:443` (beagle-core) presents an identically-shaped leaf — empty
Subject, same internal CA issuer, same validity shape — and behaves the same.

### This is D2's family, and D2 already has a repro corpus

`docs/handoff/souc_v0800_defects.md`'s **D2 — large-aggregate struct
cross-module arity-mismatch segfault** describes the same shape: a struct
with a fixed-size array field, passed across a module boundary, corrupting
at runtime — and its isolated repros show the crash **only** cross-module
(`d2_main_crossmod_arity_mismatch_segv.sio`), never single-module
(`d2_boundary_singlemod_arity_mismatch_ok.sio`). That is exactly the split
seen here, and exactly why the four synthetic single-module repros below
came back clean.

Whoever takes the compiler defect should start from
`docs/handoff/repros/d2_*.sio` rather than from scratch: D2's variable is
aggregate-parameter *arity*, and the natural next experiment is whether the
same corruption reproduces for aggregate *return* values (which is the shape
`x509_parse_certificate` has — it returns `(Certificate, i32)` across a
module boundary).

### What a synthetic repro did and did not show

Four single-module reproductions were built and run, to try to isolate the
offset defect away from TLS:

| Shape | Result |
|---|---|
| `{ xs: [Inner; 11], ys: [Inner; 11], len: i32 }`, `Inner = [i64; 64]` | correct |
| `{ xs: [Inner; 11], len: i32 }`, `Inner = [i64; 64]` | **readback 255, not 2** |
| Same two-array shape scaled to `[i64; 5760]` (~46 KB, real `Certificate` magnitude) | correct |
| Real shape — two array fields, function-call initialisers, `[Cand; 16]` outer array, 46 KB elements | correct |

So element size alone does not trigger it, and neither does the field
pattern alone. The one single-module shape that *did* misbehave produced
**255** — the same value the live path produces for
`candidates[0].bufs[0].cap` and for `tbs_start`. That `255` recurring across
three unrelated reads is the sharpest single clue available for the compiler
investigation, and it is why the trigger is believed to require the
cross-module path rather than any particular struct geometry.

---

## Test evidence

> **Harness pitfall, hit during this investigation and worth one line of
> your attention.** `scripts/run_sio_test_suite.sh` defaults to
> `/workspace/sounio/bin/souc` — the *shared* checkout — not the compiler in
> the worktree you are testing. Run from a worktree without overriding it and
> **every** test fails with `run exited 1`, including tests that have nothing
> to do with your diff (`pem_decode_roundtrip`, `tls_record_rfc8448`), which
> reads exactly like a catastrophic regression you just caused. It is not one.
> Always:
> ```bash
> SOUC_BIN=$PWD/bin/souc SOUNIO_STDLIB_PATH=$PWD/stdlib \
>   bash scripts/run_sio_test_suite.sh <pattern>
> ```
> The same 37 tests went 0-pass/37-fail to full-pass on that flag alone, with
> no source change in between.

All runs below use the worktree's own compiler
(`SOUC_BIN=$PWD/bin/souc`, Madaros v0.80.0), `SOUNIO_TEST_JOBS=4`, on the
same pod, with **only `stdlib/x509/chain.sio` differing** between the two
columns.

| Pattern | Clean tree (`ece1420be`) | With the D18 fix |
|---|---|---|
| `x509` | 11 pass / **13 fail** / 3 skip | **13 pass** / **11 fail** / 3 skip |
| `tls` | — | **7 pass / 0 fail** |
| `pem` | — | **2 pass / 0 fail** |

**No regressions:** the 11 failures under the fix are a *strict subset* of
the baseline 13. Two tests flip from fail to pass —
`x509_chain_adversarial.sio` and `x509_chain_forged_root_dn_collision.sio`
(the latter being D17's own RSA forged-root regression test).

**On the 11 that still fail:** they fail identically at the clean D17 commit,
so they are pre-existing and were not introduced here. They are, however,
consistent with Part 3 rather than incidental — `tests/run-pass/` programs
`use` stdlib modules and are therefore *also* cross-module, which is why they
see this defect family at all. `x509_chain_verify_positive.sio`, for
instance, now returns `-7` (`CHAIN_ERR_HOSTNAME_MISMATCH`): the path itself
verifies, and it fails one step *later* than it used to. Diagnosing the
remainder is Part 3's work, not this fix's, and is deliberately left in
scope for whoever takes the compiler defect.

> **Discrepancy worth flagging.** D17 recorded `x509` at **24 pass / 0 fail**.
> Re-measured at D17's own commit on this worktree it is **11 pass / 13 fail /
> 3 skip**. The most likely explanation is the harness pitfall noted above —
> the default `SOUC_BIN` resolves to the *shared* `/workspace/sounio` compiler,
> not the worktree's — so the two numbers are probably measurements of two
> different binaries rather than a regression between them. Recorded here
> rather than silently reconciled, because whichever way it resolves, one of
> the two figures is not measuring what its entry claims it measures.

---

## Filed

Not a GitHub issue. Recorded here and in `docs/handoff/souc_v0800_defects.md`
per that file's convention.

**Next highest-value work:** the Part 3 compiler defect —
`x509_parse_certificate`'s returned `Certificate` having wrong
`tbs_start`/`tbs_len` across a module boundary. Until it is fixed, a real TLS
handshake against these endpoints cannot complete, regardless of anything
further done in `stdlib/x509/` or `stdlib/tls/`.

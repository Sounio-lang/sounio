<!-- docs:meta
topic_id: repo.docs.audit.d19-aggregate-return-refutation-and-stale-arena-dispatch-2026-08-27
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.d19-aggregate-return-refutation-and-stale-arena-dispatch-2026-08-27
-->

# Forensic dispatch — D18's aggregate-return compiler defect does not exist, and the `arena full` is a stale binary

**Filed:** 2026-08-27 · **Status:** the dispatched compiler defect is **REFUTED
by measurement on the very endpoint that produced it**; the live handshake
**now succeeds end to end**; the `arena full` symptom is **root-caused to a
stale deployed artefact**, not to any defect in this tree · **Protocol:**
CLAUDE.md §8 · **Outcome: no `self-hosted/` change is warranted, and none was
made.**

**Related:** D18 (the dispatching investigation — its Parts 1 and 2 stand
unchanged and are not revisited; only its **Part 3** is corrected here), D15
(the "a bare −N is not evidence" lesson, and the ECDSA-SHA384 hole that still
blocks DNS-over-HTTPS), D16/D12 (arena cost), D11 (the tuple-destructure
struct-type recovery, which is present and working), D2 (the
large-aggregate cross-module family this was assigned to)

---

## Summary

D18 Part 3 characterised, and this dispatch was opened to fix, a compiler
defect: `x509_parse_certificate` returning a `Certificate` whose `tbs_start` /
`tbs_len` were `255` / `44` against a true `4` / `295` when called across a
module boundary from `stdlib/tls/handshake.sio`. It concluded that "a real TLS
handshake against these endpoints cannot complete until this compiler defect is
fixed".

**It reproduces on no surface, and the conclusion is false.** On this branch,
as committed:

```
$ kubectl exec -n beagle epistemic-search-<pod> -- /tmp/d19_clean.elf
trust_store err 0 count 152
tls_connect rc 0
```

That is a real TLS 1.3 handshake against the real deployed
`10.96.250.10:443`, with full CA-chain verification against the pod's own
152-entry trust store, from a probe built out of this worktree with a
**pristine, uninstrumented `stdlib/`**. `10.96.250.20:443` (beagle-core)
returns `rc 0` as well.

Four findings:

1. **Refuted:** `tbs_start`/`tbs_len` are `4`/`295` everywhere they were
   measured — offline, and live at the exact call site D18 instrumented.
   `chain_result` is `CHAIN_OK`.
2. **Arithmetic that dissolves the premise:** `255 + 44 = 299`, and `299` is
   the certificate's true TBS content end. D18's two "independently corrupt"
   fields were never independent — `tbs_len` is a *subtraction* from
   `tbs_start`. There was at most **one** wrong scalar, so the evidence never
   supported an aggregate-**return** defect in the first place.
3. **Root cause of the `arena full` / empty-stdout regression:** the deployed
   `epistemic-search:v4` image carries a binary compiled by a **pre-8-GiB
   compiler**. Measured: it reserves **2.0 GiB** of arena, the workload's peak
   is **3.2 GiB**. Rebuild it and the abort disappears. Nothing in this tree is
   at fault.
4. **What actually still blocks a useful answer** (neither is a compiler
   defect, both measured): beagle-core answers the memory-context request with
   **HTTP 401**, and Cloudflare's `1.1.1.1` DoH leaf is
   **ecdsa-with-SHA384 over P-256** — D15's known unsupported combination — so
   every domain-name search result is dropped for want of DNS.

---

## Part 1 — the reported corruption, on every surface it was looked for

### 1.1 Offline, at the exact call site, on the exact bytes

The 385-byte leaf served by `10.96.250.10:443` was captured
(`openssl s_client -showcerts` → `openssl x509 -outform DER`) and confirmed
byte-identical to the certificate D18 measured: **385 bytes, byte-sum 30515**,
`openssl asn1parse` giving `4:d=1 hl=4 l=291` — TBS TLV at offset 4, length
295.

It was wrapped in a real RFC 8446 §4.4.2 Certificate message (398 bytes,
reconciling exactly as D18's Part 1 arithmetic does) and pushed through
`decode_certificate_message` — the very function D18 instrumented — with
`tls::client`, `x509::chain` and `x509::trust_store` in the module closure:

| Measurement | Expected | D18 Part 3 reported | D19 observed |
|---|---:|---:|---:|
| `tbs_start`, direct parse | 4 | — | **4** ✓ |
| `tbs_len`, direct parse | 295 | — | **295** ✓ |
| `tbs_start`, via `decode_certificate_message` | 4 | 255 | **4** ✓ |
| `tbs_len`, via `decode_certificate_message` | 295 | 44 | **295** ✓ |
| `not_after_unix` | 1795568973 | 1795568973 | **1795568973** ✓ |

Prepending `trust_store_load()` (D16's ~1.3 GB of permanent arena) before the
decode, to reproduce the live process's allocation state, changed nothing.

This offline case is now pinned as
`tests/run-pass/tls_decode_certificate_message_p256_leaf.sio`.

### 1.2 Live, instrumented, against the real endpoint

`stdlib/x509/cert.sio` was temporarily instrumented at D18's own site — the
statement `let tbs_start_pos = cert_inner.pos` — printing the inline read, a
read of the same field through a **typed-reference helper**
(`fn d19_reader_pos(r: &DerReader) -> i64 { r.pos }`, the standard
discriminator between field-index misresolution and real memory corruption),
and the `DerTag` values the length is derived from. `stdlib/tls/client.sio` was
instrumented at `chain_result`, and `stdlib/x509/chain.sio` inside
`chain_verify_path`.

The probe was compiled from this worktree and run **inside the
`epistemic-search` pod**, whose `/etc/ssl/certs/ca-certificates.crt` carries
the `conclave-search-internal-ca` trust anchor (152 entries):

```
P1 start
P2 trust_store err 0 count 152
P3 connecting
D19 inline_pos 4 typedref_pos 4 tbs_start_pos 4 top_pos 0 cs 8 cl 291
D19 stored tbs_start 4 tbs_len 295
D19 cvp path_len 2 c0.tbs 4/295 c0.pka 1
D19 chain_result 0 intermediate_count 0 leaf.tbs_start 4 leaf.tbs_len 295
P4 tls_connect rc 0
```

Every one of D18 Part 3's numbers is contradicted, on its own endpoint:
`255` → **4**, `44` → **295**, `-6` (`CHAIN_ERR_BAD_SIGNATURE`) → **0**
(`CHAIN_OK`), and the handshake completes.

The instrumentation was then removed and the probe rebuilt against a pristine
`stdlib/`, because added `print` statements change lowering and could in
principle have masked the defect:

```
trust_store err 0 count 152
tls_connect rc 0
```

Both instrumented and uninstrumented builds agree. **`chain_verify_path` runs
with `path_len = 2` and the ECDSA-P256/SHA-256 signature verifies.**

> One environment trap worth recording. The same probe run in
> `sounio-workspace-control-0` returns `chain_result = -1`
> (`CHAIN_ERR_NO_PATH_TO_ROOT`), because that pod's CA bundle has **146**
> entries and does not contain the internal CA. D18's repro instructions name
> that pod and a 152-entry bundle; the bundle there has since changed. A
> handshake probe against these endpoints must run where the trust anchor
> actually is — the `epistemic-search` pod — or it measures the trust store,
> not the parser.

### 1.3 Why the original reading was self-refuting

D18 reported `tbs_start = 255` and `tbs_len = 44` as two corrupt fields, and
built the "aggregate return value" hypothesis on the fact that they were the
struct's 2nd and 3rd members while later members were fine.

But `stdlib/x509/cert.sio` computes them as:

```sio
let tbs_start_pos = cert_inner.pos
...
cert.tbs_start = tbs_start_pos
cert.tbs_len = (tbs_tag.content_start + tbs_tag.content_len) - tbs_start_pos
```

For this certificate `content_start + content_len = 8 + 291 = 299`, and
`299 - 255 = 44`. The reported pair is exactly what a **single** wrong
`tbs_start_pos` produces, with `tbs_len` following arithmetically. It is *not*
two independent corruptions, so the "second and third members corrupt, later
members fine" shape — the entire basis for calling this a new,
non-monotonic, aggregate-return defect — never held.

### 1.4 The two standing compiler hypotheses, both eliminated

**`field_idx_from_name_simple`'s global first-match fallback** (D9 / Finding 24
/ D11 — the usual cause of a wrong-but-deterministic struct field read) cannot
be responsible for these two fields: `grep -rn "tbs_start" --include=*.sio` over
`stdlib/` and `self-hosted/` shows **exactly one** struct declaring
`tbs_start`/`tbs_len` anywhere in the tree, so a name-only scan has nothing to
collide with.

The one field that *could* collide is `DerReader.pos` (`pos` is declared by
17 structs across the tree, at differing indices — `HsBuf.pos` in
`stdlib/tls/client.sio` sits at index 3 against `DerReader.pos` at index 1),
and it is read through a **tuple-destructured** binding
(`let (cert_inner, e0) = der_enter(...)`), which is precisely the shape that
loses its struct type. But **D11's recovery is present and works.**
`lower_let_stmt_ref` records `LOWER_TUPLET_TMP_ID` / `LOWER_TUPLET_CALLEE_ID`
at the desugared `let __tupN = f()` and re-binds each element's struct type at
the following `let x = __tupN.k`. Two purpose-built repros confirm it:

| Repro | Shape | Result |
|---|---|---|
| single-module | decoy struct declaring `pos` at index 2 ahead of a reader struct with `pos` at index 1; `let (r, e) = mk()` then `r.pos` | **correct (4)** |
| cross-module | same, with the decoy and the reader in two separate imported modules | **correct (4)** |

and the live typed-reference control (`typedref_pos 4` alongside
`inline_pos 4`) closes it on the real path.

**D2's aggregate-arity family** is also not in play: the corrupting shape D2
documents is a segfault, the value here is not corrupt at all, and the
"aggregate *return* values are the untested case" experiment D18 proposed
returns clean results on every variant tried.

### 1.5 What most likely produced D18's numbers

Not established, and deliberately not guessed at length. Two possibilities are
consistent with everything above, and both are recorded rather than chosen
between:

- D18's Part 3 measurement was taken in a **different checkout** — its own
  repro instructions point at `/workspace/.wt/conclave-search-tls-verify`, not
  at this worktree — whose `stdlib/` state is not this branch's.
- The reported values came from **instrumentation that was itself misreading**.
  A debug `print_int(parsed_cert.tbs_start)` added at a site where the local's
  struct type is not bound is exposed to exactly the field-index fallback D18
  was hunting; the probe would then report a wrong number for a correct
  struct. This is the failure mode CLAUDE.md §6.1 exists for, and it is why
  §1.2 above carries a typed-reference control read next to every inline read.

Either way the actionable conclusion is the same: **there is no defect here to
fix, and `self-hosted/` must not be touched for it.**

---

## Part 2 — the `arena full` / empty-stdout symptom

### 2.1 Hypothesis (a) — buffered stdout hiding the status lines — refuted

A program was written that prints three lines and then deliberately exhausts
the arena, run with stdout redirected to a **file** (the fully-buffered case,
the only one where loss is plausible):

```
$ ./arena_flush > out 2> err ; echo $?
181
$ cat out
STATUS LINE 1
STATUS LINE 2
STATUS LINE 3
$ cat err
madaros: arena full
```

Madaros's `print` is unbuffered, and exit 181 loses nothing already written.
**Empty stdout therefore means genuinely nothing was printed** — the abort is
*earlier* than it looks, not later. For `conclave-search`, `main`'s first
possible output is after Stage 1, so the abort is inside
`fetch_memory_context`.

### 2.2 Hypothesis (b) — a regression introduced by D18's restructuring — refuted

`stdlib/x509/chain.sio` was temporarily reverted to its pre-D18 content
(`git show ece1420be:stdlib/x509/chain.sio`), `conclave-search` rebuilt against
it, and run in the same pod. It behaves **identically** to the post-D18 build:
the full pipeline runs, exit 1 with an honest report, **no `arena full`**.
D18's change is not the differentiator in either direction.

### 2.3 Hypothesis (c) — downstream of the `tbs_start` corruption — moot

There is no corruption (Part 1).

### 2.4 The actual cause: the deployed binary has a 2 GiB arena

Measured in the same pod, on the same query, differing only in which binary
runs:

| Binary | Arena reserved (`VmSize`) | Peak `VmHWM` | Result |
|---|---:|---:|---|
| deployed `/opt/conclave-search/bin/conclave-search` (image `v4`) | **2,100,000 kB** (2.0 GiB) | — | `madaros: arena full`, **exit 181**, empty stdout |
| built from this worktree with `bin/souc` | **8,394,152 kB** (8.0 GiB) | **3,351,036 kB** (3.2 GiB) | full pipeline, exit 1, honest report |

The deployed binary was compiled by a toolchain predating `eea3a449f`
(*fix(compiler): raise the native-v2 process arena from 2 GiB to 8 GiB
(Linux)*). The workload's real peak is **3.2 GiB** — comfortably inside 8 GiB,
and unreachable inside 2 GiB. It dies in Stage 1, before the first `println`,
which is exactly the observed empty stdout.

This worktree's own compiler emits the 8 GiB arena, confirmed independently by
walking a 1 MiB-per-iteration allocation loop to the wall: it reaches
**~7.2 GiB** before aborting.

**This is a deployment/stale-artefact problem, not a defect.** The remedy is to
rebuild `epistemic-search`'s image with the current toolchain. Doing so is out
of this dispatch's scope by its own terms.

### 2.5 The freshly built binary, end to end, against the live cluster

```
$ /tmp/d19_cs.elf "what is the speed of light" 10.96.250.10 10.96.250.20
conclave-search: fetch_memory_context failed (unreachable beagle-core, or no atoms)
conclave-search: continuing with empty enrichment
conclave-search: resolve_a_record failed for a candidate URL host -- dropping it   (x5)
conclave-search: zero usable sources obtained across the whole run
conclave-search: (no web page yielded a relevant claim, no memory atoms) -- aborting
EXIT=1
```

No abort, and every stage reports honestly. **SearXNG discovery works** — five
real candidate URLs came back over a verified TLS connection. Two things then
stop a useful answer, and neither is a compiler or `stdlib/x509` defect:

**(i) beagle-core returns HTTP 401.** TLS to `10.96.250.20:443` verifies to
`CHAIN_OK` (probe above, `rc 0`), so this is not a transport failure. Issuing
the same request `search/memory_context.sio` builds —
`POST /api/exocortex/v1/graphrag/query` with `X-Beagle-Consumer:
conclave-search` and no other credential — returns **401 with an empty body**.
The endpoint requires a Bearer token. `memory_context.sio`'s own source comment
predicted this precisely ("without SOME value here, a live call is
unconditionally rejected and this stage always fails-soft"); it needs a
configurable credential, which `conclave-search` has no mechanism for yet.

**(ii) DNS-over-HTTPS to `1.1.1.1` fails the chain check.** The probe returns
`tls_connect rc -8`, and Cloudflare's leaf is:

```
Signature Algorithm: ecdsa-with-SHA384
Public Key Algorithm: id-ecPublicKey   NIST CURVE: P-256
```

`ecdsa-with-SHA384` over a P-256 key is **D15's documented hole**. Every
SearXNG result URL is a domain name, `tls_connect` rejects non-dotted-decimal
hosts, so without DNS every candidate is dropped — hence "zero usable
sources".

Both are separately scoped follow-ups, stated with evidence rather than
guessed at, per this dispatch's own remit.

---

## Test evidence

No `self-hosted/` file and no `stdlib/` file was modified by this
investigation. The only source change is one **added** test,
`tests/run-pass/tls_decode_certificate_message_p256_leaf.sio`. A full-suite
before/after run is therefore not the gate it would be for a compiler change;
the affected prefixes were run instead, with the harness invoked correctly
(`SOUC_BIN` pinned to this worktree, `SOUNIO_TEST_JOBS=4` — see D18's own
harness note, which cost that investigation real time and is reproduced in
`docs/handoff/souc_v0800_defects.md`).

All runs: `SOUC_BIN=$PWD/bin/souc SOUNIO_STDLIB_PATH=$PWD/stdlib
SOUNIO_TEST_JOBS=<4|6> bash scripts/run_sio_test_suite.sh <pattern>`, Madaros
v0.80.0, on `t560-proxmox`. "Clean tree" columns were taken with the change set
stashed (`git stash -u`) and the identical invocation.

| Pattern | Clean tree | With D19's change set |
|---|---|---|
| `x509` | — | **24 pass / 0 fail / 3 skip** |
| `tls` | 7 pass / 0 fail | **8 pass / 0 fail** (the added test) |
| `pem` | — | **2 / 0** |
| `asn1` | — | **5 / 0** |
| `ocsp` | — | **3 / 0** |
| `crypto` | — | **1 / 0** |
| `socket` | — | **2 / 0** |
| `der` | 36 pass / **10 fail** / 42 skip | 36 pass / **10 fail** / 42 skip — *byte-identical* |
| `chain` | 21 pass / **2 fail** / 22 skip | 21 pass / **2 fail** / 22 skip — *byte-identical* |

Both patterns with failures were baselined explicitly, and both come back
identical, same test names, same reasons. Neither failure set is related to
this work: `der` is a **substring** match that sweeps in `pbpk28_deriv_zero`,
`lorenz_i256_derivative_num_smoke`, `unit_derived_*`, `closure_higher_order`
and friends; `chain` likewise sweeps in `import_chain_b` / `import_chain_main`
(module-import chain, not certificate chain). No file any of them touches was
modified.

> **`x509` at 24/0/3 is worth one line.** D18 recorded 13 pass / 11 fail / 3
> skip at its own fix, and re-measured D17's commit at 11/13/3. Measured here
> with the same harness flags on the current branch it is **24 pass / 0 fail /
> 3 skip** — i.e. D17's original 24/0 figure, not D18's. The x509 pass/fail
> counts in this file's D17 and D18 entries have now disagreed with each other
> three times across three sessions; treat any of them as a measurement of a
> particular binary on a particular pod, never as a property of the tree.

---

## Filed

Not a GitHub issue. Recorded here and in `docs/handoff/souc_v0800_defects.md`
per that file's convention, together with an amendment marking **D18 Part 3
withdrawn**.

**Next highest-value work**, in order, none of it in `self-hosted/`:

1. Rebuild and redeploy `epistemic-search` with the current toolchain. That
   alone removes the `arena full` abort.
2. Give `conclave-search` a way to carry a beagle-core Bearer credential.
3. Close D15's `ecdsa-with-SHA384` gap, or point `resolve_a_record` at a
   resolver whose certificate this stack can verify. Until one of the two
   happens, the pipeline cannot reach any web source.

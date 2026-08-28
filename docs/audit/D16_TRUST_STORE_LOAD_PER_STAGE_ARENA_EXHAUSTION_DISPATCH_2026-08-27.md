<!-- docs:meta
topic_id: repo.docs.audit.d16-trust-store-load-per-stage-arena-exhaustion-dispatch-2026-08-27
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.d16-trust-store-load-per-stage-arena-exhaustion-dispatch-2026-08-27
-->

# Forensic dispatch — `trust_store_load()` costs ~1.3 GB of arena per call, and a process survives only six

**Filed:** 2026-08-27 · **Status:** RESOLVED for the reported workload (ceiling **6 → unbounded** loads/process); the underlying lifetime defect is UNCHANGED and still open · **Protocol:** CLAUDE.md §8 · **Supersedes nothing in** [`ARENA_EXHAUSTION_TLS_HANDSHAKE_CHAIN_VERIFICATION_DISPATCH_2026-08-26.md`](ARENA_EXHAUSTION_TLS_HANDSHAKE_CHAIN_VERIFICATION_DISPATCH_2026-08-26.md) **(D12) — it measures a cost D12's probe was structurally blind to.**

**Commit:** see `git log --oneline -- stdlib/x509/trust_store.sio`

---

## Summary

`Sounio-lang/conclave-search`, compiled from this worktree's toolchain, aborts
with

```
conclave-search: fetch_memory_context failed (unreachable beagle-core, or no atoms)
conclave-search: continuing with empty enrichment
madaros: arena full
```

(exit 181) on a **single ordinary query** — far short of the 95-handshake
ceiling D12 established. Not an OOM: the k8s container has a 10 Gi limit, does
not crash, and is not cgroup-killed. The Madaros runtime aborts internally.

The cause is **not** per-handshake cost. It is that **`trust_store_load()`
itself costs ~1.3 GB of never-reclaimed arena per call**, and
`conclave-search` calls it once per TLS-using stage — including **once per
candidate URL** inside the DNS-over-HTTPS resolver. One query issues
`4 + N_candidates` loads (up to **14** at `MAX_RESULTS = 10`). A process
admits **six**.

D12 could not see this because its probe
(`tests/interop/tls_arena_multi_handshake_probe.sio:36`) loads the trust store
**exactly once, outside its handshake loop** — the correct usage, and
therefore the one that never measures the wrong usage's cost.

---

## Root cause

### The measurement

New probe, `tests/interop/trust_store_load_arena_cost_probe.sio` — same
methodology as D12's `x509_arena_cost_probe.sio` (print the iteration index
*before* the operation; the last number printed is the ceiling):

```
$ SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run \
    tests/interop/trust_store_load_arena_cost_probe.sio
trust_store_load iter 1 … 7
madaros: arena full          (exit 181)
```

**Six completed loads per process.** With `native_v2_arena_bytes() =
8589934592` (`self-hosted/native/gc.sio:94`), that is **~1.3 GB per load**.

### Where the 1.3 GB goes

Attribution, measured, not inferred:

| Component | Measurement | Cost |
|---|---|---:|
| `TrustStore` value construction (`[certificate_zero(); 512]` + `[rawbuf_new(1); 512]`) | 20,000 iterations, **no exhaustion** | ~0 |
| `150 × x509_parse_certificate` (real Debian bundle, `grep -c "BEGIN CERTIFICATE" /etc/ssl/certs/ca-certificates.crt` = 150) | D12's own attribution: ~8.8–11.5 MB/call | **~1.3 GB** |

The `[certificate_zero(); 512]` array-repeat form is *not* 512 allocations on
this compiler — a natural suspicion, measured and refuted. The whole cost is
D12's already-identified `x509_parse_certificate` scratch, multiplied by the
150 roots in the real system bundle.

### Why one query needs fourteen loads

`trust_store_load()` call sites in `conclave-search` (`/home/devsounio/conclave-search`):

| Call site | Times per query |
|---|---:|
| `search/memory_context.sio:270` (`fetch_memory_context`) | 1 |
| `search/discovery.sio:152` (`discover_candidates`) | 1 |
| `src/main.sio:610` (page-fetch store) | 1 |
| `search/dns_resolve.sio:248` (`resolve_a_record`) | **1 per candidate URL** (≤10) |
| `search/writeback.sio:338` (`writeback_synthesis`) | 1 |

The observed run therefore dies at load **#7 — the fourth candidate URL's DoH
resolution** — which is exactly the measured ceiling. The `fetch_memory_context
failed` line immediately preceding the abort is not incidental: that stage
burned the **first** 1.3 GB before failing, so a stage that accomplished nothing
still consumed a sixth of the process's entire lifetime budget.

---

## Leads investigated and closed

1. **Is `fetch_memory_context failed` masking an arena-heavy partial failure?**
   **Yes, but not in the way suspected.** It is not a "cheap 401". It performs a
   full `trust_store_load()` (~1.3 GB) before its handshake, so the failing
   stage is the single most expensive thing in the run after the loads
   themselves. Nothing unusual about beagle-core's internal-CA chain is
   required to explain the abort.

2. **Is Cloudflare's `1.1.1.1` chain unusually arena-expensive?**
   **Yes — measured, ~4.8×.** Same probe, one trust-store load, then repeated
   real `tls_connect` + chain verification:

   | Target | Result code | Handshakes/process | Implied cost/handshake |
   |---|---:|---:|---:|
   | `8.8.8.8:443` | 0 (success) | **95** | ~70 MB |
   | `1.1.1.1:443` | −8 (`TLS_CONNECT_ERR_CERT_CHAIN`, the known ECDSA-SHA384 gap of D15) | **20** | ~335 MB |

   A **failing** chain verification is not cheap: it parses the chain before it
   rejects it. This is a real secondary finding — the DoH path costs ~335 MB per
   candidate even though every one of its handshakes fails today.

3. **Does `conclave-search`'s larger linked program (509+ / 5,082 merged
   functions) inflate the per-handshake cost?** **No evidence for it, and no
   need for it.** The ceiling of 6 loads was measured in a 170-function program
   and predicts the real failure point exactly (abort at the 4th candidate).
   The per-load cost alone fully accounts for the symptom; no residual is left
   for program size to explain.

4. **Is the handshake count secretly higher (retries, redirect-follow)?**
   **No.** The count is as documented; the multiplier is `trust_store_load`,
   not `tls_connect`. Lead 4's own second hypothesis — "`trust_store_load()`
   called fresh inside every stage rather than threaded through" — is the
   correct one, and it is worse than a doubling: it is a **14×** on the single
   most expensive operation in the library.

---

## Fix

`stdlib/x509/trust_store.sio`: memoise the trust store for the process
lifetime.

- The previous body is preserved verbatim as **`trust_store_load_uncached()`**
  (still `pub`, for probes and for tests that need a genuine re-read).
- **`trust_store_load()`** now returns a module-level `var TRUST_STORE_CACHE`,
  populating it on first call. Failures are memoised too (state 2): the two
  failure modes are "bundle unreadable" and "bundle malformed", neither of
  which a retry inside the same process can fix, and a retry costs another
  1.3 GB.

Why this is sound rather than a workaround:

- The store is **immutable after load**, so there is nothing to invalidate.
- Its `bufs` are arena handles that are **never reclaimed** for the life of the
  process (the very defect that makes this bug possible), so a cached
  `RawBuf` can never dangle.
- **Returning the memo by value is arena-cheap on this compiler** — measured:
  3,000 consecutive returns of a cached `TrustStore` did not move the arena.
  Large aggregates are not deep-copied per element here. This fix does not
  trade one arena leak for another.

The security-critical half of this is that whole-struct assignment of a value
containing `[Certificate; 512]` into module-level state is *exactly* the shape
`stdlib/x509/cert.sio` documents as miscompiling in other positions. A garbled
or truncated cached root set is a chain-validation bypass, not a performance
bug, so it is pinned by a test rather than assumed.

### Before / after

| Measurement | Before | After |
|---|---:|---:|
| `trust_store_load()` calls per process | **6** | **≥500** (probe loop bound; no exhaustion) |
| `conclave-search` arena spend on trust stores, one query | ~18 GB demanded vs 8 GiB available | 1.3 GB |
| `tests/run-pass/{x509_*,tls*,pem_*}` | 34 pass | **35 pass, 0 fail** (incl. the new test) |

### Tests

- `tests/run-pass/trust_store_load_cached_identity.sio` (new) — deep
  field-for-field comparison of the cached store against a fresh
  `trust_store_load_uncached()` across all 150 roots (subject and issuer DN,
  RSA modulus and exponent, key algorithm, `is_ca`, key usage, validity bounds,
  subject key id bytes), a live `trust_store_find_by_subject` /
  `trust_store_has_matching_key_for_subject` query against the cached store,
  and 50 consecutive loads in one process — **fatal before this fix, free
  after it**.
- `tests/interop/trust_store_load_arena_cost_probe.sio` (new, MANUAL) — the
  ceiling probe above.

---

## Residual risk — the memo is necessary but not comfortably sufficient

With the memo, one `conclave-search` query at `MAX_RESULTS = 10` budgets:

```
1 × trust_store_load                        1.30 GB
1 × memory-context handshake                0.07–0.34 GB
1 × SearXNG handshake                       0.07–0.34 GB
10 × DoH handshake  @ ~0.335 GB (measured)  3.35 GB
10 × page fetch     @ 0.07–0.34 GB          0.70–3.35 GB
1 × writeback handshake                     0.07–0.34 GB
                                            ----------------
                                            5.6 – 9.0 GB   vs 8 GiB available
```

Best case fits with room; **worst case still overruns**. So:

- **`MAX_RESULTS` reduction is a real stopgap, not theatre.** At
  `MAX_RESULTS = 5` the worst case falls to ~4.4 GB and the query is safe under
  any chain shape. Recommended until the parser lever below lands.
- **Threading one `&TrustStore` through `conclave-search`'s stages is still the
  right architecture** even though the memo makes it unnecessary for
  correctness. It is a one-line-per-stage change in that repo and removes the
  library's reliance on a process-global.
- **The next real lever is D12's own nomination, unchanged:
  `x509_parse_certificate`'s ~8.8–11.5 MB of scratch per call.** It is now the
  sole term in the trust-store load *and* the dominant term in every handshake.
  Bringing it near the ~330 KB `Certificate` it returns would cut the load to
  ~50 MB and the per-handshake cost by a comparable factor, retiring this whole
  failure class rather than moving its wall again.
- The **ECDSA-SHA384 gap (D15)** compounds this: every DoH handshake pays
  ~335 MB *and then fails*. Closing D15 both fixes the resolver and removes the
  most expensive failing path in the run.

---

## What is still not fixed

The lifetime defect itself. The arena is still never reclaimed; this dispatch
removes a 14× multiplier on the largest single consumer of it. As D12 put it:
the wall moved, it did not disappear.

<!-- docs:meta
topic_id: repo.docs.audit.arena-exhaustion-tls-handshake-chain-verification-dispatch-2026-08-26
authority: repo_only
audience: users
last_validated: 2026-08-26
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.arena-exhaustion-tls-handshake-chain-verification-dispatch-2026-08-26
-->

# Forensic dispatch — real-CA TLS handshakes exhaust the never-reclaimed process arena after 2 connections

**Filed:** 2026-08-26 · **Status:** RESOLVED for the reported workload (ceiling 2 → 95 handshakes/process); the underlying lifetime defect is UNCHANGED and still open · **Protocol:** CLAUDE.md §8 · **Specialises** Finding 12 of [`TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`](TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md).

**Commits:** `62b466d17` (probe) · `c9bd996b2` (x509 struct footprint) · `976e3e399` (chain-verification copies) · `eea3a449f` (compiler arena capacity)

---

## Summary

A Madaros-compiled process could complete **two** real, CA-verified TLS 1.3
handshakes. The third died with

```
madaros: arena full
```

and exit status 181 — an uncatchable, unrecoverable abort, not a Sounio-level
error a program can observe or handle. Some certificate chains exhausted the
budget on the **first** handshake.

This blocked the `conclave-search` architecture outright, which needs a search
discovery call, a memory-context call, and one fetch per result page — a dozen
or more HTTPS connections — in a single CLI invocation.

After this dispatch the same probe completes **95** handshakes per process.

---

## Root cause

Two independent causes compose. Neither is a new defect class; the first is a
consequence of a known one.

### Cause 1 (pre-existing, unchanged) — the arena is never reclaimed

Finding 12 already established it: the native-v2 runtime takes **one fixed
anonymous mapping at process entry** and never frees or reuses anything in it.
Handles are allocated by a monotonic bump of `RuntimeContext.handle_count`; the
only reset emitter (`native_v2_emit_gc_empty_frame_reset`) is deliberately
unwired, because stack maps carry slot COUNTS rather than a root bitmap, so no
safe liveness point exists. On heap exhaustion the allocator requests a GC
(`native_v2_gc_reason_heap_limit()`), finds it cannot prove anything dead, and
exits 181.

The mapping's size is therefore not a working-set size. **It is the entire
per-process allocation budget**, and every value-returning function that
allocates spends it permanently.

### Cause 2 (this dispatch's specific finding) — certificate values are enormous, and chain verification copies them by the hundred

`stdlib/x509/cert.sio`'s `Certificate` carried, per certificate:

| Field | Logical size |
|---|---:|
| `san_entries: [GeneralName; 32]` | ~91 KB |
| `issuer_alt_names: [GeneralName; 32]` | ~91 KB |
| `extensions: [ExtensionEntry; 32]` | ~17 KB |
| `issuer` + `subject` (`X509Name`, 16 `RdnEntry` each) | ~5 KB |
| 3 × `BigInt` (`[u16; 512]` each) | ~3 KB |
| everything else | ~2 KB |
| **total** | **~210 KB** |

The dominant term was **a field that was never populated and never read**.
`GeneralName` carried `directory_name: X509Name` for the RFC 5280
directoryName choice ([4]). `x509_parse_general_names` deliberately skips
copying it — a Madaros defect corrupts that doubly-indexed struct-in-array
write, documented at length in that function — and nothing in `stdlib/` or
`tests/` ever reads it back. At 2,564 of `GeneralName`'s ~2,849 bytes, across
64 `GeneralName`s per `Certificate`, **one dead field was ~87% of every
certificate value in the program.**

`stdlib/x509/chain.sio` then copied `Certificate` *values*, not references, in
its hot path: a materialised `[Certificate; 520]` issuer pool rebuilt per
verification, plus `let candidate = pool_certs[i]` executed once per
trust-store root (150 on this host) per DFS step.

### The multiplier nobody had priced in

Measured here, and worth recording because it inflates every size estimate in
this repository by 8×: **a `[u8; N]` field occupies one 8-byte slot per
element**, not N bytes. `certificate_zero()` alone was measured at ~352 KB of
arena against a ~46 KB logical struct.

---

## Repro

`tests/interop/tls_arena_multi_handshake_probe.sio` (commit `62b466d17`) — N
sequential real, CA-verified handshakes against a real public HTTPS host in one
process, printing the 1-based attempt number *before* each attempt, so the last
number printed is the ceiling.

```
SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc run \
  tests/interop/tls_arena_multi_handshake_probe.sio
```

Baseline on this branch, 150-cert system trust store (`/etc/ssl/certs/ca-certificates.crt`):

```
arena_probe: trust store loaded, cert count: 150
arena_probe: handshake attempt 1  -> result code -8
arena_probe: handshake attempt 2  -> result code -8
arena_probe: handshake attempt 3
madaros: arena full                     (exit 181)
```

The `-8` verdict is a hostname mismatch and is irrelevant to the measurement:
`tls_connect` has no DNS resolver, so the host is given as dotted-decimal text
and reused for SNI and for `x509_verify_hostname`. All chain building and path
verification — the arena spend being measured — happens before that check.

### Cost attribution

`tests/interop/x509_arena_cost_probe.sio` (commit `976e3e399`) isolates one
operation per run and counts iterations to exhaustion. Measured per-process
ceilings, **before** the fixes below:

| Operation | Iterations/process | Implied cost |
|---|---:|---:|
| `certificate_zero()` | 5,254 | ~352 KB |
| `x509_parse_certificate()` | 161 | ~11.5 MB |
| `x509_verify_chain()` | 44 | ~30 MB |

This attribution is the load-bearing result of the investigation. It shows the
per-handshake spend is dominated by **certificate parsing and handshake
crypto**, not by path building — which is why the path-building fix below,
though real, does not move the end-to-end number on its own.

---

## Impact

- Any Sounio program making more than two real HTTPS requests per process died,
  uncatchably. Some certificate chains (e.g. `1.1.1.1:443`'s) died on the first.
- Not recoverable in-language: exit 181 is an abort, not a `Result`.
- No CLI escape hatch: the sizes are hardcoded compiler constants; `souc --help`
  exposes nothing.
- `SOUNIO_SOUC_ENGINE=lean_single` is not an alternative — it cannot resolve the
  `net`/`tls`/`x509` module graph at all.

---

## Resolution

Three changes, in the order they were measured.

### 1. `c9bd996b2` — drop `GeneralName.directory_name` (`stdlib/x509/cert.sio`)

Removed the never-populated, never-read `X509Name` field. `GeneralName`
~2,849 B → ~285 B; `Certificate` ~210 KB → ~46 KB. No behavioural change.
Field indices 0..4 are unchanged, so
`tests/run-pass/x509_generalname_field_index_ambiguity.sio` still locks the
same layout against the compiler's name-only field-resolution fallback.

**Ceiling: 2 → 11 handshakes.**

If directoryName support is ever genuinely needed, it must not go back inline:
park the decoded name in a side table indexed by GeneralName position, so only
certificates that actually use the [4] choice pay for it.

### 2. `976e3e399` — stop copying `Certificate` values in `stdlib/x509/chain.sio`

- The `[Certificate; 520]` issuer pool is now **virtual**: index `i < split` is
  `intermediates[i]`, else `trust_store.certs[i - split]`. Identical index
  ordering, so `path_pool_index` bookkeeping is untouched. Removes a ~24 MB
  array plus one copy per trust-store root per verification.
- Candidate matching extracted to
  `cert_is_plausible_issuer(&Certificate, &Certificate)` so the search compares
  through references taken on the array element, replacing
  `let candidate = pool_certs[i]`.
- `let current = path_certs[depth-1]`, the three by-value binds per link in
  `chain_verify_path`, and `let cand = candidates[c]` (a `ChainCandidate` holds
  11 `Certificate`s, ~506 KB) all became `&` references.
- `candidate_indices` hoisted out of the DFS loop.

Applied identically to both `chain_build_candidates` and the deliberately
inlined copy inside `x509_verify_chain`.

The `rc` temporary in the record-a-path branch was **deliberately kept**:
writing straight into `candidates[n].certs[ci]` is a doubly-indexed
struct-in-array write, a known-corrupting shape on this compiler.

**Ceiling: 11 → 11.** Reported as measured. The change removes a genuine
per-verification cost and is correct, but the attribution table above explains
why it is invisible end-to-end: parsing and crypto dominate. It is kept rather
than reverted because it is a real reduction and it makes the remaining cost
legible.

### 3. `eea3a449f` — raise the process arena 2 GiB → 8 GiB (`self-hosted/native/gc.sio`, `self-hosted/native/codegen_x86_linux.sio`)

New named constant `native_v2_arena_bytes()` replaces the four hardcoded
`2147483648` literals in the two Linux entry-trampoline emitters plus the
`nc_core_emit_alloc_into` bound check.

`native_v2_handle_table_capacity_default()` moved **2^22 → 2^24 in lockstep**.
Left at 2^22, the handle wall (exit 182) would still have fired at the same
4.19M objects with ~7.7 GiB of heap unused beneath it. At 2^24 filling the
table needs 939 MiB of heap against a 768 MiB table, so the table remains the
first wall reached — the invariant `gc.sio`'s comment exists to preserve, and
the size that comment already sanctions once the mmap grows.

**Linux only.** The mapping is `MAP_PRIVATE|MAP_ANONYMOUS`, so the kernel backs
pages lazily and 8 GiB costs address space, not RSS. The Windows entry path
uses `VirtualAlloc` with `MEM_COMMIT`, which is not lazy in the same way, and
the macOS path is untested at this size; both deliberately stay at 2 GiB.
`native_v2_ref_array_bss_window_end()` is a pointer-classification boundary,
not an arena size, and is untouched.

Accepted trade: a program that runs to exhaustion now touches ~8 GiB RSS before
aborting instead of ~2 GiB.

**Ceiling: 11 → 95 handshakes.**

Note the growth is superlinear in the arena (4× arena, 8.6× ceiling) because a
large one-time cost is amortised. Solving `1856 MiB = F + 11·P` against
`7424 MiB = F + 95·P` gives **P ≈ 66 MiB per handshake** and **F ≈ 1.1 GiB
one-time**, essentially all of it `trust_store_load()` parsing 150 root
certificates.

---

## Verification

| Check | Result |
|---|---|
| `tls_arena_multi_handshake_probe` | **2 → 95** handshakes/process |
| `run_sio_test_suite.sh --filter-prefix x509_` | 22 pass / 0 fail / 3 skip |
| `--filter-prefix tls` | 7 pass / 0 fail |
| `--filter-prefix array_` | 10 pass / 0 fail / 2 skip |
| `--filter-prefix gc` | 2 pass / 0 fail |
| `--filter-prefix rsa` | 1 pass / 0 fail |
| `scripts/epistemic_trust_gate.sh` | `EPISTEMIC_TRUST_GATE_OK`, k95i=2776 |
| `--filter-prefix struct_` | 5 pass / **1 fail** — pre-existing |
| `--filter-prefix knowledge_` | 15 pass / **8 fail** — pre-existing |

The compiler was rebuilt with `scripts/ci/build_modular_madaros.sh` (which
takes the global build lock itself and must not be wrapped).

**Every failure above was proven pre-existing** by re-running it against the
committed pre-change `bin/madaros-linux-x86_64` and comparing:

- `struct_missing_field.sio` — both binaries emit the identical
  `error[E046] ... struct literal has wrong number of fields`. The harness
  expects the literal text "missing field"; this is a harness-expectation
  mismatch, not a compiler behaviour change. The four `knowledge_*`
  "missing error:" failures are the same class.
- `knowledge_array.sio` (rc=139), `knowledge_unwrap.sio` (rc=139),
  `knowledge_octonion_inner.sio` (rc=1), `knowledge_value_with_epistemic.sio`
  (rc=1) — identical return codes and byte-identical output from both binaries.

---

## What remains open

1. **The lifetime defect itself is untouched.** The arena is still never
   reclaimed. This dispatch moved the wall ~47× and priced it; it did not
   remove it. A process doing unbounded HTTPS work will still die. The real fix
   is either arena reclamation (which needs stack maps to carry a root bitmap,
   not slot counts) or restructuring these modules around caller-supplied
   scratch buffers.

2. **`x509_parse_certificate` at ~11.5 MB per call is now the single largest
   lever** — 32× the cost of `certificate_zero()`, so the great majority is
   parser scratch rather than the returned value. It was not attacked here:
   that function is the subject of several open corruption findings
   (Findings 22/24/25 and the unfiled directoryName write defect), and
   restructuring its scratch allocations is a separate, riskier dispatch. Doing
   so is the highest-value next step and would benefit every X.509 consumer.

3. **`Certificate` still carries dead weight.** `issuer_alt_names:
   [GeneralName; 32]` (~9 KB post-fix) is written by the parser and read by
   nothing except a test asserting its count is zero. `extensions:
   [ExtensionEntry; 32]` (~17 KB, the largest remaining field) is parser
   scratch that survives into the returned value; chain verification reads only
   the already-decoded `is_ca` / `key_usage_bits` / `path_len_constraint` /
   `subject_key_id` / `authority_key_id` fields.

4. **The 8×-per-byte array representation** (`[u8; N]` costs 8N bytes) is worth
   filing on its own. Every buffer-sizing decision in `stdlib/x509/` and
   `stdlib/tls/` was made against the logical size.

---

## Trail

- Finding 12, [`TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`](TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md) — the root lifetime mechanism and the ~460,000-call `bigint_add` ceiling.
- [`X509_ARRAY_STRUCT_FIELD_CORRUPTION_DISPATCH_2026-08-24.md`](X509_ARRAY_STRUCT_FIELD_CORRUPTION_DISPATCH_2026-08-24.md) — the struct-in-array write defect that left `directory_name` unpopulated.
- [`D11_ARENA_SCRATCH_RESET_CROSS_MODULE_CORRUPTION_DISPATCH_2026-08-26.md`](D11_ARENA_SCRATCH_RESET_CROSS_MODULE_CORRUPTION_DISPATCH_2026-08-26.md) — a defect that *looked* like arena exhaustion and was not; useful contrast.
- `docs/handoff/souc_v0800_defects.md` — handoff index.

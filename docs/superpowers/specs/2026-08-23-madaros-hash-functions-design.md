<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-23-madaros-hash-functions-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-23-madaros-hash-functions-design
-->

# Madaros Hash Functions (SHA-1 / SHA-256 / SHA-384 / SHA-512) — Design Spec

## Context and Motivation

This is a prerequisite sub-project for the X.509 semantic layer (itself sub-project 0b-ii-b of the TLS 1.2 roadmap, currently paused pending this work). Verifying a certificate's signature (RSASSA-PKCS1-v1_5-VERIFY, RFC 8017 §8.2.2) requires hashing the certificate's signed bytes (the `TBSCertificate` DER encoding) and comparing that hash — wrapped in a `DigestInfo` structure — against the RSA public-key operation's output. TLS 1.2 certificates in practice use SHA-256 predominantly, SHA-1 on older/legacy certificates, and SHA-384/SHA-512 occasionally on higher-security chains. This sub-project builds all four, generalizing the PKCS#1 v1.5 verification this project needs. Note that a separate, incremental/fixed-buffer SHA-256 + HMAC-SHA256 implementation already exists at `stdlib/crypto/` (`sha256_hash`, `hmac_sha256`, plus `sha256_init`/`sha256_update`/`sha256_finalize`, all over fixed `[u8; 256]` buffers) — this is not this project's first cryptographic-hash capability. This new `stdlib/hash/` module exists specifically because the TLS/X.509 path needs a `&RawBuf`-oriented single-shot API over buffers up to `DER_MAX_LENGTH` (65536 bytes), which `stdlib/crypto/`'s small fixed `[u8; 256]` buffers cannot accommodate, and because it also needs SHA-1, SHA-384, and SHA-512, none of which `stdlib/crypto/` provides.

This sub-project produces pure computation over byte buffers — no networking, no linear resource types, no `RawBuf`-as-syscall-target concerns from `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`'s Finding 2. The one directly relevant finding from that audit is **Finding 11**: `u64` right-shift, division, and modulo silently produce wrong results whenever the operand's bit 63 is set, while `+`, `-`, `*`, `&`, `|`, `^`, `<<`, and comparisons are confirmed correct regardless. That finding was discovered and mitigated for `BigInt`'s narrow, carefully-bounded 16-bit-limb arithmetic (every intermediate value was designed to stay far below bit 63). **Hash-function internal state is the opposite case**: SHA-256's 32-bit words and SHA-512/384's 64-bit words are, by design, close to uniformly distributed bit patterns — a hash word having its top bit set is not a rare edge case here, it is the common case. This sub-project cannot design around Finding 11 the way BigInt did; it must either prove the relevant operations are safe at native width, or engineer around the unsafe ones directly.

## Non-Goals

- **No HMAC.** Out of scope for this module because one already exists at `stdlib/crypto/hmac.sio` (`hmac_sha256`, over `stdlib/crypto`'s fixed `[u8; 256]` buffers). A future TLS 1.2 PRF implementation, which will need HMAC-SHA256, should evaluate reusing or adapting that existing implementation rather than assuming none exists or building a second one against this module's `&RawBuf` types.
- **No streaming/incremental hashing API** (e.g. `sha256_init` / `sha256_update` / `sha256_final`). Every caller in this project's roadmap (hashing a DER-encoded `TBSCertificate`, at most tens of kilobytes, already bounded by `DER_MAX_LENGTH` = 65536 in the ASN.1 layer) has the complete message available as a single `RawBuf` + length before hashing begins. A single-shot API (`sha256(buf: &RawBuf, len: i64) -> [u8; 32]`, and analogously for the other three) is simpler and sufficient. An incremental API is a natural, backward-compatible future extension if a later use case needs one (e.g. hashing a TLS record stream), not a reason to build one speculatively now.
- **No generic/pluggable hash-algorithm abstraction inside this module.** Each algorithm gets its own module with its own concrete function name and fixed-size output type. Generality across hash algorithms belongs to the PKCS#1 v1.5 layer above this one (which the X.509 sub-project will build, parameterized on a `DigestInfo` prefix + hash-length pair), not to this module.
- **No message length beyond what an `i64` byte-count and `RawBuf` capacity can represent.** SHA-2's padding scheme appends the original message length as a 64-bit (SHA-256/SHA-1) or 128-bit (SHA-384/512) big-endian bit-count. This project bounds every message it will ever hash to `DER_MAX_LENGTH` (65536 bytes) or a small, similarly-bounded constant — nowhere near the range where the high 64 bits of a 128-bit length field would ever be nonzero. The SHA-384/512 padding logic writes zero for those unreachable high bits rather than implementing full 128-bit length arithmetic.

## Architecture

Five new files:

```
stdlib/hash/sha1.sio
stdlib/hash/sha256.sio
stdlib/hash/sha512_core.sio
stdlib/hash/sha384.sio
stdlib/hash/sha512.sio
```

Import convention (per the existing audit doc's Findings 6/8): a caller outside `stdlib/` uses `use hash::sha256::*` (etc.); a sibling file inside `stdlib/hash/` uses the bare `use sha512_core::*` form. `sha384.sio` and `sha512.sio` both depend on `sha512_core.sio`.

### Public API shape (uniform across all four algorithms)

```sio
pub fn sha1(buf: &RawBuf, len: i64) -> [u8; 20]
pub fn sha256(buf: &RawBuf, len: i64) -> [u8; 32]
pub fn sha384(buf: &RawBuf, len: i64) -> [u8; 48]
pub fn sha512(buf: &RawBuf, len: i64) -> [u8; 64]
```

Each function is a pure computation (no `IO` effect needed for the arithmetic itself, though reading `buf`'s bytes via the established `rawbuf_get`-style pattern does require `with IO`, matching every other `RawBuf`-reading function on this branch). No error return — unlike the ASN.1 decoder, there is no malformed-input concept here: any byte sequence of any length is a valid message to hash. `len` must be `>= 0` and `<= buf`'s allocated capacity; violating this is a caller bug (out-of-bounds read), not a recoverable error condition, matching this branch's established convention that `RawBuf` capacity is the caller's responsibility.

### Sequencing: 32-bit family first, 64-bit family second

Per the explicit decision made during brainstorming, this sub-project is sequenced in two phases with an empirical audit opening each:

**Phase 1 — 32-bit family (SHA-1, SHA-256).** Both operate on 8 (SHA-256) or 5 (SHA-1) 32-bit state words, an XOR/rotate-based message schedule, and modular addition mod 2³². The very first task of the implementation plan is a dedicated empirical audit of `u32` arithmetic on Madaros — every operator SHA-1/SHA-256 need (`+` with wraparound, `&`, `|`, `^`, `~` if available, `<<`, `>>`, comparisons) — mirroring exactly the audit that discovered Finding 11 for `u64` during the BigInt sub-project. The audit's outcome determines the concrete word representation:
- If native `u32` arithmetic (particularly wraparound-on-overflow for `+`, and `>>`/`<<` behavior at the full 32-bit range including the top bit) proves correct, use `u32` directly — the natural, idiomatic choice.
- If any operation is found broken (the established failure pattern on this branch is right-shift/divide/modulo breaking once a type's top bit is set — i.e. a `u32`-scale analogue of Finding 11), fall back to the BigInt-style mitigation: represent each 32-bit word as an `i64` that is masked (`& 0xFFFFFFFF`) after every operation that could overflow 32 bits (addition, left-shift), so the stored value never exceeds `2^32 - 1` — nowhere near `i64`'s own bit-63 danger zone, regardless of what the 32-bit value's own top bit (bit 31) is doing. This is a **different, weaker property** than BigInt's "stay far below bit 63" discipline: here the concern is a hypothetical `u32`-native bug (unconfirmed, to be settled by the audit), not `i64`'s own Finding 11, which does not apply to values this small either way.

This audit's findings and the resulting representation choice must be recorded in `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` as new, explicitly-numbered findings, continuing that document's existing numbering — it is this branch's single living record of empirically-verified compiler arithmetic behavior, and hash-function arithmetic is squarely within its scope even though it's a new sub-project.

**Phase 2 — 64-bit family (SHA-384, SHA-512).** These share one compression core (`sha512_core.sio`) operating on 8 64-bit state words, differing only in IV constants and final-output truncation (SHA-384 truncates the core's 64-byte output to 48 bytes and uses a distinct IV — it is not simply "SHA-512 with fewer bytes printed," the IV difference is required by the standard specifically so that SHA-384 and a truncated SHA-512 of the same message never coincide). Phase 2 opens with its own empirical audit, this time expected — not merely hedged — to find a `u64`-scale analogue of Finding 11: hash-function internal words are close to uniformly random and will routinely have bit 63 set, unlike BigInt's or SHA-256's carefully-bounded values, so this audit is testing "does Finding 11 actually bite here" rather than "might some unrelated bug exist." If confirmed (the expected outcome), the mitigation is a **split-halves representation**:

```sio
pub struct U64Pair {
    hi: i64,   // upper 32 bits, stored in the low 32 bits of this i64, masked
    lo: i64,   // lower 32 bits, stored in the low 32 bits of this i64, masked
}
```

with every SHA-512-core primitive (`add64`, `rotr64`, `shr64`, `xor64`, `and64`, `not64`) implemented as a function over `U64Pair` values performing the 64-bit operation via 32-bit-safe sub-operations and explicit carry propagation between halves — the same technique real C implementations of SHA-512 use on platforms without native 64-bit arithmetic. This is more code than a native-`u64` implementation would be, but every sub-operation it decomposes into is either already-proven-safe (Finding 11's own confirmed-safe list: `+`, `-`, `*`, `&`, `|`, `^`, `<<`, comparisons, all safe unconditionally) or is a 32-bit-scale operation resolved by Phase 1's audit. If Phase 2's audit unexpectedly finds native `u64` arithmetic safe for these specific operations (contradicting Finding 11's own text — the audit's job is to check this empirically, not assume it), use native `u64` instead and skip `U64Pair` entirely; either way, the finding and the resulting decision get recorded in the same audit document.

### Algorithm structure (standard, not novel)

All four algorithms follow their published specifications (FIPS 180-4) exactly — this sub-project is a from-scratch implementation of a well-defined standard, not new cryptographic design:

- **Padding**: append a single `0x80` byte, then zero-pad to the block boundary minus the length field (64 bits for SHA-1/256's 512-bit blocks; 128 bits for SHA-384/512's 1024-bit blocks — per this spec's Non-Goals, the high 64 bits of that 128-bit field are always written as zero), then append the original message bit-length as a big-endian integer of that width.
- **Message schedule**: each 512-bit (SHA-1/256) or 1024-bit (SHA-384/512) block expands into 80 (SHA-1) or 64 (SHA-256) or 80 (SHA-384/512) schedule words via the algorithm's defined XOR/rotate/shift recurrence.
- **Compression function**: the standard 80-round (SHA-1) or 64-round (SHA-256) or 80-round (SHA-384/512) main loop updating the state words using the algorithm's defined round function, message schedule word, and round constant.
- **Output**: the final state words, concatenated big-endian (and, for SHA-384, truncated to the first 48 of the would-be 64 output bytes).

The implementation plan will contain the actual round-constant tables and recurrence formulas verbatim (these are public, standardized constants — FIPS 180-4 Sections 4.1-4.2, 5.3, 6.1-6.4 — not design decisions this spec needs to argue for).

## Testing Strategy

For each of the four algorithms, test against official published vectors (FIPS 180-4 / RFC 3174 for SHA-1, well-known NIST test vectors for the SHA-2 family):
1. The empty string (`""`) — exercises padding-only, zero-block-of-real-data behavior.
2. `"abc"` — the standard short single-block test vector every SHA implementation's test suite includes; published expected digests exist for all four algorithms.
3. A message long enough to span at least two full blocks (e.g. the standard NIST "abcdbcdecdefdefgefghfghighij..." 448-bit vector for SHA-1/256, or an analogous multi-block vector for SHA-384/512) — exercises the message-schedule/multi-block chaining logic, which the single-block vectors above cannot.

Each expected digest is copied from the published standard/NIST source, not computed by this project — an implementer must independently confirm the exact hex digest string against the cited source before committing it as a test's expected value, per this project's standing "measure, don't assume" discipline (the same discipline that caught the BigInt sub-project's RSA vector needing independent reverification, and the ASN.1 sub-project's plan-authored test bugs).

All tests as `tests/run-pass/hash_*.sio` (e.g. `hash_sha256_vectors.sio`), invoked via `bash scripts/run_sio_test_suite.sh --filter-prefix hash_`.

## Open Implementation-Time Questions

- The exact outcome of both arithmetic audits (Phase 1's `u32` audit, Phase 2's `u64` audit) is unknown until the audits actually run — this spec deliberately does not commit to `u32`-native vs. `i64`-masked, or `u64`-native vs. `U64Pair`, in advance. The plan's first task in each phase is the audit itself; the representation choice is a ruling made from its results, recorded in the audit doc, and carried into every subsequent task in that phase.
- Whether Madaros's fixed-size byte-array return type (`[u8; 32]` etc., returned directly from a function rather than stored in a struct field) behaves as expected — this branch has used fixed arrays as struct fields (`BigInt.limbs: [u16; 512]`) but not yet as a bare function return type. Worth an early, explicit check (a trivial function returning a small fixed array, called and indexed by its caller) before committing the whole module's public API shape to this pattern.

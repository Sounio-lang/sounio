<!-- docs:meta
topic_id: repo.docs.audit.a64-aggregate-substrate-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a64-aggregate-substrate-2026-07-05
-->

# A64 aggregate substrate campaign — 2026-07-05

Status: **the aarch64-macos lane is fully green on real Apple Silicon**
(88/88 witnesses: 80 shape-matrix + 6 lean borrow/RMW + 2 read probes),
validated end-to-end via the Mac ssh soak harness. Commits `35fb7c908` +
`2a2fbdeb7` (copy direction), `125546ed4` + `d0ddc63a8` (borrow engine +
call guard).

## Baseline (first-ever a64 runtime validation)

The metal gate only ever ran x86 binaries; the first hardware soak
(macOS 27, Apple Silicon, 86 Mach-O arm64 witnesses) returned **3/86**:
every literal-initialized witness died at init (SIGBUS 138), borrows
segfaulted, aggregate reads returned slot pointers as values.

## The three defects, root-caused and fixed

1. **Struct-literal aggregate copy direction** (`copy_agg_into_struct_slots_a64`):
   the ascending copy loop started at slot `dst_start-(nslots-1)` — the
   HIGHEST address of the field region — so element 0 landed in the LAST
   element's cell and the remaining words overflowed the region, clobbering
   neighboring locals. Slot N lives at fp−N*8: the loop must start at slot
   `dst_start`. All three branches fixed. (The repeat-init path was always
   correct — why the two-level RMW witness passed while literals failed.)
2. **Missing a64 borrow engine**: the a64 `&`/`&!` primary handled only
   whole-var and slice-range borrows; field-chain/element borrows silently
   borrowed the root. New branches route `&(*name).chain` and `&name.chain`
   through `compile_place_projections_a64` with the kind-aware epilogue
   (aggregate slots borrow the LOADED element pointer).
3. **Missing same-line call guard**: the a64 primary treated ANY following
   `(` as a call on the preceding name — across lines — so `... = i` newline
   `(*hb).arr[...] = ...` compiled into `blr` to the VARIABLE'S VALUE and
   desynced the parser. Mirrors the x86 `TL[EP]==TL[name_tok]` guard.

## Forensic chain

Hardware soak (matrix partition: reads/assigns vs borrows) → local
byte-decode of Mach-O (signature encodings) → probe bisection
(v5/v6/v7/v10: two var-idx assigns = trigger) → compile-time dispatch
tracing (ENTER/GENERAL_ASSIGN/EXPR logs exposed the 7-token overconsume →
the phantom call).

## Gates

Fixed point + canonical PASS at each step (`a0eb403c…`, `5467f029…`);
x86 shape matrix 80/80 unchanged throughout; downstream imported matrix
12/12 at `d0ddc63a8`; full a64 hardware re-soak **88/88**.

## Soak harness (reusable)

Compile `--target aarch64-macos` → scp to the Mac
(user `demetriosagourakis`, Tailscale `macbook-pro-de-demetrios-2.tail21cbc4.ts.net`
or IP `100.91.184.41`; pod key `id_ed25519_dgx_spark` authorized) →
`codesign -s -` each binary → run. macOS has no `timeout`; remote default
shell is zsh (drive with `bash -s`). DNS to the tailnet is flaky from the
pod — prefer the IP.

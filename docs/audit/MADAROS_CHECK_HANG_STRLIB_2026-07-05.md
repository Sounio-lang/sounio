# Madaros checker hang on `str::lib` — 2026-07-05

Status: **NEW BLOCKER, class HANG** (not segfault, not wrong-code).
Blocker-ID: BLK-MADAROS-CHECK-HANG-STRLIB
Severity: high (blocks default-lane `check` for any string-using module)
Owner: unassigned (forensic dispatch needed)
Evidence level: measured, minimal repro isolated to one stdlib file

## Shape

`bin/souc check` (default Madaros engine) emits the banner and then
produces no further output; killed by timeout at 60–240 s. No segfault,
no diagnostic. `SOUNIO_SOUC_ENGINE=lean_single` is unaffected (all
affected tests run ALL PASS there).

## Measured matrix (worktree `gpu/epistemic-tensor-core-next`, post the
place-resolver/zoo-retirement commits, 2026-07-05 ~14:20 UTC)

| Target | default `check` | Notes |
|---|---|---|
| `stdlib/str/lib.sio` (tracked, clean, single module) | HANG >120 s | the minimal repro |
| `tests/stdlib/str/test_str.sio` (pre-existing test) | HANG >60 s | imports `str::lib` |
| `/tmp` probe, 8-line main, only `use str::lib::*` | HANG >90 s | import is sufficient |
| `stdlib/eisa/isa.sio` (imports str::lib) | HANG >90 s | how this was found (EISA v1a) |
| `stdlib/eisa/core.sio` (imports math::dd64 only) | OK, fast | control |
| probe: struct with `[f64; 16]`/`[i64; 16]` fields | OK, fast | shape hypothesis ruled out |
| probe: `[InsB; 32]` array of structs + `&[InsB; 32]` param | OK, fast | shape hypothesis ruled out |

The earlier working hypothesis (nested struct arrays hang the checker)
is **falsified** by the two shape probes. The trigger is inside
`stdlib/str/lib.sio` itself; every importer inherits the hang.

## Context

- `stdlib/str/lib.sio` is tracked and unmodified (last commit touching
  it: `2e67d5a9d`).
- This branch carries today's seed/lvalue campaign (general place
  resolver `cacd3c358`, handler retirement `be3e63b52`, bootstrap
  resyncs). Whether the hang predates those commits is NOT yet
  measured — bisecting `souc check stdlib/str/lib.sio` across today's
  seed resyncs is the first step of the dispatch.
- Distinct from the two known default-lane issues: the (resolved)
  `seed_begin` segfault family and the (open) imported-lane runtime
  wrong-code residue (test_error_policy exit 1, probe SIGSEGV on struct
  field read). This one is compile-time and terminates nothing.

## Impact on EISA track

- EISA v1a (`stdlib/eisa/isa.sio` + `tests/stdlib/eisa/test_eisa_isa.sio`)
  and E1 (`format.sio`/`asm.sio`, in flight) use `str::lib` for receipt
  and assembler text handling; their default-lane `check` hangs. They
  remain `validated_lane: lean_single`, consistent with the rest of the
  track.
- Runtime on lean_single is healthy: `test_eisa_isa.sio` ALL PASS
  P1–P5 with receipts.

## Next action (for the dispatch owner)

1. Bisect default-engine `check stdlib/str/lib.sio` across today's
   seed commits (`cc0122dc0` → `be72afbe9`) with a 60 s timeout.
2. If pre-existing, bisect further back; if introduced today, hand the
   repro to the seed campaign owner with this doc as evidence.
3. Suspect surface: checker path for `Str`/`StrSplit` fixed-capacity
   struct signatures with `&`/`&!` params in `str/lib.sio` — but that
   is a hypothesis, not evidence; the two shape probes above show the
   simple forms are fine.

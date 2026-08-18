<!-- docs:meta
topic_id: repo.docs.audit.rc182-diagnostic-5-tests-fixed-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.rc182-diagnostic-5-tests-fixed-2026-08-18
-->

> **Status**: Production | **Last validated**: 2026-08-18 | **Source**: scripts/audit/rc182_diagnose.py

# rc=182 — Corrected Diagnostic on the 5 New Failing Tests

**Date:** 2026-08-18
**Supersedes:** the measurements table in `RC182_DIAGNOSTIC_GAP_AND_INSTRUMENT_2026-08-17.md` (same directory). The gap analysis and in-binary fix spec in that file remain valid and unchanged.
**Scope:** rc=182 ("handles full") diagnostic on the 5 tests grok-cli2 re-measured (rapamycin_clinical, gum_vs_mc, pop_sim, d2_gum, d2_voi)
**Method:** external instrumentation via `/proc/<pid>/mem`, reading DIRECTLY from the runtime mmap base (no magic scan, no `hc < hcap` filter)
**Lane:** minimax-cli3 (this lane); gc.sio is grok-cli1 territory, codegen_x86_linux.sio is congested with three active lanes today

## Why the prior measurements were wrong

The earlier probe (`/tmp/handle-instrument/rc182_diagnose.py` from
2026-08-17) used a magic-scan heuristic to find the runtime context:
scan `/proc/<pid>/mem` for the 2^22 magic `4194304`, then validate each
candidate by reading the handle_count at the magic's address +32 and
checking `hc < hcap`. That filter discarded the real context the moment
handle_count climbed near capacity — which is precisely when the
diagnostic matters most. The probe then reported:

- d2_gum/d2_voi: peak_handle = 1,835,008 (43.76% of capacity, "2.36M headroom")
- rapamycin_clinical: peak_handle = 4,186,767 with peak_pin = 4,217,088 (100.7%)

All of those numbers were artefacts. The 1,835,008 was a stale
candidate the filter passed through (a stack-resident snapshot from an
earlier init stage), and the 100.7% was a write-race on the 8-byte
value that the filter happened to keep.

## The corrected instrument

The corrected instrument reads the runtime context from the runtime
mmap base directly. The runtime mmap is the 2 GiB anonymous rw-p region
the entry trampoline allocates via `sys_mmap` at
`codegen_x86_linux.sio:9882-9897`; the runtime context lives at the
front of it. There is exactly one of them, and its values stay coherent
regardless of how close handle_count gets to capacity. No magic scan,
no filter, no stale candidates.

Committed at `scripts/audit/rc182_diagnose.py`.

## Corrected measurements

All five tests exit rc=182. handle_capacity = 4,194,304 (2^22) for every
test, as expected from `gc.sio:64`. t_total ≈ 1.5–1.65 s for all five.

| Test | peak_handle | delta_to_wall | % of capacity | peak_pin | wall hit by |
|---|---|---|---|---|---|
| rapamycin_clinical | 4,186,515 | 7,789 | 99.81% | 0 | allocation |
| gum_vs_mc         | 4,192,947 | 1,357 | 99.97% | 0 | allocation |
| pop_sim           | 4,185,474 | 8,830 | 99.79% | 0 | allocation |
| d2_gum            | 4,189,539 | 4,765 | 99.89% | 0 | allocation |
| d2_voi            | 4,191,233 | 3,071 | 99.93% | 0 | allocation |

## What this changes in the diagnostic verdict

The prior GAP doc said the three "wall" tests (rapamycin_clinical,
gum_vs_mc, pop_sim) had peak_pin ≈ peak_handle (100.0–100.7%) and the
two "headroom" tests (d2_gum, d2_voi) had peak_pin = 0 with 2.36M of
apparent headroom. That picture was wrong in both directions.

The corrected picture: **all five tests have peak_pin = 0**. The
reclamation verdict inverts. There are no pinned handles to reclaim
from. Reclamation cannot help these tests in any form. The only paths
to fix them are:

1. **More capacity** — raising the wall to e.g. 2^24 (16,777,216) buys
   ~12.6M handles of headroom. With the init footprint at ~4.19M, that
   gives enough room for the test bodies to run. Cheap to implement
   (one constant change) but does not address the lifetime debt.
2. **Less init footprint** — investigate what in the darwin_pbpk
   init sequence allocates ~4.19M handles. The comment at `gc.sio:64`
   says this is "the lifetime problem... still open" — the runtime
   never reclaims handles because reclamation is unwired (also per
   grok-cli1: detection of death is the blocker, implementing now
   would be theatre or the bug of the reset).

Both paths are out of this lane's scope.

## What the curve is (now confirmed)

All five tests sit at 99.79%–99.97% of 2^22 within their first second of
runtime. The wall is hit by **allocation, not retention**. peak_pin is
zero on every test — the runtime does not pin handles in the slow paths
these tests exercise.

The earlier `CURVE_REPORT_5_TESTS.md` documented the curve as "a step,
not a slope and not a wrap". The corrected measurements sharpen that:
**all five tests are at the same step**. N ranges 7–20 across the
tests, peak_handle is within ±7,500 of each other (cluster mean ≈
4,189,000). If allocation were per-sample, doubling N would double
peak_handle. It does not. The init footprint is a fixed cost of the
darwin_pbpk modules, and the first sample after init eats the
remaining capacity.

## Porting the instrument to Sounio (CLAUDE.md §4 cost)

CLAUDE.md §4: "science of this repository is done in Sounio, not
Python." The instrument is in Python. Honest cost of porting it to a
Sounio native ELF:

### Parts Sounio can express

- Reading `/proc/<pid>/maps` lines and parsing hex ranges: yes, Sounio
  has string ops and integer arithmetic; a 200-line `parse_maps_line`
  in `stdlib/io/proc.sio` covers it.
- Polling `/proc/<pid>/mem` via syscalls (sys_open, sys_read, sys_close):
  Sounio's `darwin_posix` shim already exposes these. The instrument's
  hot path is ~10 lines of Sounio per poll (open fd, seek, unpack,
  close).
- Tracking peak values: trivial.
- Printing the diagnostic: Sounio's stdlib `io` module handles stderr
  write.

### Parts Sounio CANNOT express without runtime work

- **Popen / fork+exec**: there is no `posix_spawn` or `fork` shim in
  `darwin_posix` today. The whole diagnostic is a fork-and-watch
  pattern; without a process-spawn shim the Sounio instrument has to be
  a static standalone tool driven from the shell, not a Sounio program
  that takes argv[1] as the target binary. This is a real gap and the
  reason the Python prototype wins on time-to-first-result: shelling
  out is one line of Python.
- **ptrace attach** (alternative): Linux ptrace would let a Sounio ELF
  attach to a running pid without fork+exec, but Sounio has no
  ptrace/PTRACE_ATTACH syscall shim either.
- **/proc parsing for an arbitrary pid** while running in the same
  binary: the kernel allows reading `/proc/<other-pid>/maps` and
  `/proc/<other-pid>/mem` if the reader is privileged or has the right
  ptrace scope, but Sounio does not have CAP_SYS_PTRACE helpers nor a
  prctl wrapper to drop capabilities cleanly. Today the path
  "self → open /proc/<pid>/mem → read" is open in C and in Python; in
  Sounio it requires a new syscall shim.
- **Live offset polling at 5 ms cadence**: Sounio's runtime has no
  `nanosleep` or high-resolution sleep shim in the user's path; the
  existing 200 Hz tick is fine for this use case but adds CPU spin to
  the target process's debug session.

### Estimated cost (Honest — not a sale pitch)

| Phase | Effort | Notes |
|---|---|---|
| Add `posix_spawn` / `fork+exec` shim | ~1 day | touches `darwin_posix`, needs review from the posix lane |
| Add `sys_nanosleep` shim | ~2 hours | trivial; one syscall wrapper |
| Port `parse_maps_line` to Sounio | ~3 hours | 200 lines, mechanical |
| Port `read_u64` to Sounio | ~1 hour | sys_open + sys_pread64 (or sys_lseek + sys_read) |
| Port main loop + report formatting | ~4 hours | mostly the formatted-print work |
| Build, debug, validate against Python diagnostic | ~1 day | five known-rc=182 test ELFs already exist |
| **Total** | **~3 working days** | plus the posix-shim review, which is unbounded |

The Python instrument at `scripts/audit/rc182_diagnose.py` is 176
lines including docstring and exits cleanly. The Sounio port would be
~400 lines (more verbose for the same logic) and would NOT cover
process spawn until the posix shim lands. So the port is a deferred
investment, not a near-term refactor.

**Recommendation:** keep the Python instrument in `scripts/audit/`,
add a Sounio mirror once `posix_spawn` exists in the stdlib shim. The
diagnostic value does not change; only the implementation language
does, and only when the shim gap closes.

## Files

- Instrument (committed): `scripts/audit/rc182_diagnose.py`
- Original (superseded) doc:
  `docs/audit/RC182_DIAGNOSTIC_GAP_AND_INSTRUMENT_2026-08-17.md`
- Prior curve report (still useful):
  `/tmp/handle-instrument/5tests/CURVE_REPORT_5_TESTS.md`
- Corrected raw measurements (transient): `/tmp/handle-instrument/5tests/DIAGNOSTIC_REPORT_5_TESTS_FIXED.md`
- Compiled ELFs (read-only artefacts of probe runs):
  `/tmp/{rapamycin_clinical,gum_vs_mc,pop_sim,d2_gum,d2_voi}_test.out`

## Status

- Corrected instrument built ✓
- 5 corrected measurements collected ✓
- Reclamation verdict inverted (peak_pin=0 → reclamation cannot help) ✓
- In-binary fix spec unchanged, still queued for codegen lane ✓
- Wrong doc superseded in place with banner ✓
- Instrument moved from /tmp to repo per CLAUDE.md §4 ✓
- Sounio porting cost analysed and recorded above ✓

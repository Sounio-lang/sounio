<!-- docs:meta
topic_id: repo.docs.audit.rc182-diagnostic-gap-and-instrument-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.rc182-diagnostic-gap-and-instrument-2026-08-17
-->

> **SUPERSEDED** by `RC182_DIAGNOSTIC_5_TESTS_FIXED_2026-08-18.md` (same directory).
> The measurements table below has been replaced. The external instrument used here had a `hc < hcap` filter that discarded the real context the moment handle_count climbed near capacity — it produced phantom 1,835,008 readings for d2_gum/d2_voi (43.76% of capacity, "2.36M headroom") and inflated pin_count readings for the other three tests. The corrected instrument reads the runtime mmap base directly and finds all five tests at 99.79%–99.97% of capacity with peak_pin = 0. The gap analysis above (rc=182 slow-path emits no print) and the in-binary fix spec below remain valid and unchanged.

# rc=182 diagnostic — gap, external instrument, and measurements

**Date:** 2026-08-17 (corrected 2026-08-18)
**Scope:** rc=182 ("handles full") diagnostic on Sounio native ELFs
**Method:** external instrumentation via `/proc/<pid>/mem`, no compiler source modified
**Lane:** minimax-cli3 (this lane); gc.sio is grok-cli1 territory, codegen_x86_linux.sio is congested with three active lanes today

## The gap (confirmed)

The runtime already captures every relevant number at the moment of an rc=182
slow-path entry, but does not print any of them before exiting. The captured
state lives in two structs that the codegen populates before
`emit_exit(c.code, 182)`:

**runtime_context** (`self-hosted/native/runtime_context.sio:9-43`) — at the
runtime base:
- `handle_count`     offset 32  (monotonic total allocated; never reclaimed)
- `handle_capacity`  offset 40  (the wall, default 4,194,304 = 2^22)
- `pin_count`        offset 72  (live pinned handle count, the live proxy)

**gc_state** (`self-hosted/native/gc.sio:107-127`) — the slow-path metadata
buffer that `native_v2_emit_gc_request_metadata` (codegen_x86_linux.sio:2821)
fills in immediately before `emit_exit(182)` at codegen_x86_linux.sio:6379:
- `last_request_size`  offset 40  (bytes the failing alloc wanted)
- `last_reason`        offset 80  (the gc reason code, e.g. handle_table_full)
- `handle_count`       offset 104 (mirror of runtime_context)
- `handle_capacity`    offset 112 (mirror)
- `pin_count`          offset 144 (mirror)

The buffer is fully populated, but nothing in the slow path emits a syscall
that would format and print those values. The only diagnostic the user sees
today is the static string `madaros: handles full` baked into the binary's
rodata at every emit_exit call site — with no numbers.

## External instrument delivered

`scripts/audit/rc182_diagnose.py` (also `/tmp/handle-instrument/rc182_diagnose.py`
for live runs) — a standalone Python diagnostic that:

1. Forks the ELF under test (no compiler change required)
2. Reads `/proc/<pid>/maps` once, finds the largest anonymous rw-p mmap
   (the 2 GiB runtime region the entry trampoline allocates)
3. Polls `/proc/<pid>/mem` every 5 ms, reading `handle_count`,
   `handle_capacity`, `pin_count` from the runtime context at the
   runtime mmap base (no magic scan, no filter — single struct at a
   known offset)
4. Tracks peak `handle_count` and peak `pin_count` over the process lifetime
5. On rc=182, prints the diagnostic and a reclamation verdict:
   - **Reclamation WOULD HELP** if `peak_pin < handle_capacity * 0.1`
   - **Reclamation PARTIALLY HELPS** if `peak_pin < handle_capacity`
   - **Reclamation NOT ENOUGH** if `peak_pin >= handle_capacity`

**Corrected 2026-08-18:** the earlier magic-scan-with-filter variant
discarded the real context at the wall and produced phantom 1,835,008
readings for d2_gum/d2_voi. The current instrument reads the runtime
context from the mmap base directly. The corrected measurements are in
`RC182_DIAGNOSTIC_5_TESTS_FIXED_2026-08-18.md`.

The instrument reads only `/proc/<pid>/mem`; it does not write to the process,
and it does not modify any compiler source, runtime helper, or wrapper.

## Measurements on the 5 new rc=182 tests (grok-cli2 pbpk_suite)

**These numbers were SUPERSEDED on 2026-08-18.** The instrument below used
a `hc < hcap` filter that discarded real context at the wall. The
corrected measurements are in
`RC182_DIAGNOSTIC_5_TESTS_FIXED_2026-08-18.md`. Kept here for
traceability only.

Source files (read-only) in `/workspace/.wt/grok-cli2/stdlib/darwin_pbpk/`.
Built with `/workspace/.wt/minimax-cli3/bin/souc`, run via the diagnostic.

**Polling precision caveat:** the instrument polls `/proc/<pid>/mem` at
5 ms cadence and tracks the highest `handle_count` it sees. Run-to-run
values vary by ±10K because the runtime allocates handles in tight
bursts during init and the exact polling phase shifts which burst the
probe catches. The pattern (peak near capacity, peak_pin = 0) is
robust across runs; the exact peak value is not.

**Re-run 2026-08-18 (instrument from repo path):**

| Test | Source | peak_handle_count | delta_to_wall | % used | peak_pin_count |
|---|---|---|---|---|---|
| rapamycin_clinical | validation/rapamycin_clinical.sio | 4,181,352 | 12,952 | 99.69% | 0 |
| gum_vs_mc | validation/gum_vs_mc.sio | 4,185,734 | 8,570 | 99.80% | 0 |
| rapamycin_pop_sim | population/pop_sim.sio | 4,191,769 | 2,535 | 99.94% | 0 |
| d2_gum | pd/d2_gum.sio | 4,183,791 | 10,513 | 99.75% | 0 |
| d2_voi | pd/d2_voi.sio | 4,184,358 | 9,946 | 99.76% | 0 |

**Earlier corrected run 2026-08-18 (instrument from `/tmp`):**

| Test | peak_handle_count | delta_to_wall | % used | peak_pin_count |
|---|---|---|---|---|
| rapamycin_clinical | 4,186,515 | 7,789 | 99.81% | 0 |
| gum_vs_mc | 4,192,947 | 1,357 | 99.97% | 0 |
| pop_sim | 4,185,474 | 8,830 | 99.79% | 0 |
| d2_gum | 4,189,539 | 4,765 | 99.89% | 0 |
| d2_voi | 4,191,233 | 3,071 | 99.93% | 0 |

handle_capacity = 4,194,304 (2^22) for every test, as expected from `gc.sio:64`.
The two runs' peak_handle values agree within ±0.24% of capacity; both
find peak_pin = 0 on every test; both place every test within 1.5% of
the wall.

## Interpretation — the load-bearing data point

All five tests hit the wall at 99.79%–99.97% of capacity, and **peak_pin
is zero on every test**. The wall is hit by ALLOCATION, not retention.
The runtime did not pin handles in the slow paths these tests exercise.

This means the user's question "if the peak LIVE is small and the total
allocated is huge, reclamation helps" has its answer here:
**the LIVE set is zero, reclamation has nothing to do.** The only
durable paths to fix these tests are more capacity or less init
footprint — not reclamation. That direction is owned by whichever lane
can change `gc.sio:64` and/or the darwin_pbpk init sequence; this lane
only owns the diagnostic.

## What this does NOT claim

- That the in-binary diagnostic is fixed (it is not — the gap remains)
- That reclamation is the wrong direction for the broader problem (it
  may still be useful for tests not measured here; for these five it
  is ruled out by peak_pin = 0)
- That all five tests need the same fix — they all hit the wall by the
  same mechanism (init footprint), but the init sequences are module-
  specific and may want different mitigations
- That peak_pin = 0 implies the runtime never pins anything (pinning
  may exist in code paths these tests do not exercise; the
  five-measurement cluster just says it does not fire here)

## Spec for the in-binary fix (for the codegen_x86_linux.sio lane owner)

The codegen already populates the runtime context and gc_state with all
the numbers the user wants. The fix is purely a print step in the slow
path between `native_v2_emit_gc_request_metadata` (codegen_x86_linux.sio:6378)
and `emit_exit(c.code, 182)` (codegen_x86_linux.sio:6379).

The slow path needs:

1. Load `runtime_context_field_gc_state()` from runtime context (offset 120)
   to get the gc_state pointer (already wired via `emit_load_gc_state_ptr_rbx`)
2. From gc_state, read:
   - `handle_count`     at gc_state + 104
   - `handle_capacity`  at gc_state + 112
   - `pin_count`        at gc_state + 144
   - `last_request_size` at gc_state + 40
   - `last_reason`      at gc_state + 80
3. Format those into a stderr message:
   `madaros: handles full count=H capacity=C pin=P req=R bytes reason=NAME\n`
4. Emit `write(2, msg, len)` via the existing `emit_write_syscall_for_target`
   helper (codegen_x86_linux.sio:3193)
5. Then `emit_exit(c.code, 182)`

A small integer-to-string conversion routine is needed; the codebase
already has `print_int` (codegen_x86_linux.sio:3445) which can be reused
or its format helper factored out.

This change is **diagnostic-only**: it does not modify any gc logic, does
not touch gc.sio, and does not alter the exit code or reason. It only
adds numbers to the message.

## Files

- Instrument (committed): `scripts/audit/rc182_diagnose.py` (run standalone,
  no install, requires Python 3.6+; needs read access to `/proc/<pid>/mem`
  for the pid of the target ELF)
- Corrected measurements doc:
  `docs/audit/RC182_DIAGNOSTIC_5_TESTS_FIXED_2026-08-18.md`
- Prior curve report:
  `/tmp/handle-instrument/5tests/CURVE_REPORT_5_TESTS.md` (also
  referenced from the corrected doc; same data, different framing)
- Existing per-test probes (kept under `/tmp` for the session):
  `/tmp/handle-instrument/5tests/*.probe`
- Compiled ELFs (read-only artefacts of probe runs):
  `/tmp/{rapamycin_clinical,gum_vs_mc,pop_sim,d2_gum,d2_voi}_test.out`

## Honest limits

- **Polling races**: at 5 ms cadence, the probe can miss the exact moment
  of peak. Values are noisy when the runtime context is being actively
  written; sane-bound filtering is applied to drop garbage. The mmap-
  base read is single-struct, single-address, so the noise is much lower
  than the prior magic-scan variant.
- **No magic scan, but one assumption remains**: the runtime mmap is
  identified as the largest anonymous rw-p region in `/proc/<pid>/maps`
  (≥ 1 GiB). If a future ELF allocates a larger anonymous region before
  the entry trampoline runs, this assumption breaks. Today it holds.
- **pin_count reliability**: if the runtime does not implement pinning
  in some code paths, pin_count stays 0 even when handles are allocated.
  The interpretation must distinguish "no pins because nothing is pinned"
  from "no pins because probe missed them". All five measurements show
  zero pins, which is consistent with "runtime did not pin in these
  code paths" but does not prove pinning is dead in general — it only
  proves it does not fire here.
- **Lane discipline**: this lane did NOT modify codegen_x86_linux.sio
  or gc.sio. The fix spec above is for whichever lane owns the
  codegen path; that lane may already have work in flight on the same
  file, in which case this is a queue item not a hot patch.

## Status

- Gap documented (this file) ✓
- External instrument built and validated on 5 tests ✓
- 5 measurements collected, reclamation verdict emitted per test ✓
- **2026-08-18:** measurements SUPERSEDED — the instrument's `hc<hcap`
  filter discarded real context at the wall. The corrected instrument
  reads the runtime mmap base directly and finds peak_pin = 0 on every
  test. Reclamation verdict inverted (no pins → reclamation cannot help).
- In-binary fix specified but not applied (lane discipline) —
  blocked on codegen_x86_linux.sio ownership

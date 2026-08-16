<!-- docs:meta
topic_id: repo.docs.audit.read-after-fork-write-staleness-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.read-after-fork-write-staleness-dispatch-2026-08-16
-->

# "Read-after-fork-write staleness" — UNREPRODUCIBLE as described — dispatch

**Date:** 2026-08-16
**Engine:** lean_single, source-built fixed point (`make build` gen3, md5 `37c1cf8a43ab74143994ec77b9a45e5e`; identical to the refreshed `bin/souc-lean-single-x86_64`)
**Parent:** `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md` §"Secondary bugs", item #4: "`read_file()` on a path a `system()`-forked child had just written, called from the same parent process, reproducibly returned 0 bytes immediately after `wait4()` returned — even though the same path read correctly moments later from a fresh process. Inserting `sync()` … between the `system()` call and the `read_file()` call reliably fixed it."
**Owner:** unassigned
**Status:** **UNREPRODUCIBLE** — 11 attempts across two file sizes, two paths, and two child types produced zero anomalous reads. Recorded per the house rule that a failed reproduction is a result, not a gap. No `self-hosted/` change proposed.

## Why this dispatch

The parent dispatch reports this as "independently real (isolated with literal-string probes, confirmed via ffi_probe_readtest3)". Those probes no longer exist, and this session — armed with the Track B seed that the original investigation also ran on — could not make it happen once. Before anyone budgets engineering time against a filesystem-visibility hypothesis, or ships `sync()` workarounds as if they were load-bearing, the record should say what a fresh attempt finds.

## Reproduction attempts (all negative)

Common shape: `system()` runs a child that writes a file; the parent immediately calls `read_file()` on that path twice without `sync()`, then again after `syscall6(162,…)` (`sync()`); a fresh shell `wc -c` reads the same path after the process exits.

| # | child | bytes written | path | runs | anomalous reads |
|---|---|---|---|---|---|
| 1 | sh builtin (`printf … > f`) | 16 | `/tmp/sounio_f4_probe.txt` | 1 | 0 (16/16/16/16) |
| 2 | `python3` writer (with-open, close) | 37,232 | `/tmp/sounio_f4_probe.txt` | 5 | 0 (37232 every read, every run) |
| 3 | same python3 writer | 37,232 | `/workspace/.tmp/sounio_f4_probe.txt` | 5 | 0 (37232 every read, every run) |

37,232 bytes is deliberately the size of the original bridge CSV. `/tmp` and `/workspace` are the **same ext4 device** in this pod (`df -T`: both `/dev/rbd4 ext4`), which removes the cross-mount theory for this environment. Probe sources: `/tmp/ffi_probe/bug4_readfork.sio`, `bug4_big.sio`, `bug4_ws.sio` (+ `bug4_writer.py`).

## Ruled out (as explanations for the original sighting)

- **The module-const defect** (`MODULE_CONST_STRING_READ_FILE_ZERO_BYTES_DISPATCH_2026-08-16.md`) is the leading candidate for what the original probes actually saw: the original author's own code style at the time used a module-level `const BRIDGE_PATH: string`, and the parent dispatch item #5 already concedes that "this … was the actual cause of every '[read] bridge CSV length = 0 bytes' observed while debugging #4". A probe intended to isolate #4 whose path passed through a module-level const would reproduce "0 bytes after fork" perfectly — **and would be fixed by any change that also touched the read ordering**, such as inserting a `sync()` call, by the fragile-representation luck of recompilation. This is a hypothesis about history, clearly labelled as such; it cannot be proven without the original probes.
- **A genuine ext4/POSIX visibility window** between `wait4()` and a subsequent `open()` in the parent: not standard POSIX behaviour on ext4, and 11/11 clean reads here.

## Root-cause locus

None assign. The honest position: no defect is demonstrated to exist by today's evidence; the surviving candidate mechanism for the historical sighting is contamination by the const-string defect, which this session did reproduce 4-for-4 on its own symptoms.

## Proposed action

1. Keep the `sync()` call in `examples/cayley_dickson_lemon_g2_ffi.sio` — harmless belt-and-braces, and the file is another lane's claim anyway.
2. Do **not** budget work against a filesystem-visibility theory unless a reproducer re-emerges on a current seed. If one does, the first variable to control is the path binding (module const vs local let), given the const-string defect.
3. If the const-string defect is fixed and the LEMON pipeline still shows a 0-byte read after fork **without** a const path anywhere, reopen this dispatch with that probe.

## Impact if unaddressed

None demonstrated. The cost of the current state is minor: a `sync()` syscall per pipeline run and one paragraph of dispatch-doc caution.

## AI disclosure

Reproduction attempts and the contamination hypothesis by AI agent (Claude) under human direction, 2026-08-16, on lean_single gen3 (md5 `37c1cf8a…`). All probe sources regenerable from the table. No `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.

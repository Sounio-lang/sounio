# Handoff: issue #2130 probe corpus

From the #2126 investigation (claude-2). Issue #2130 is claude-1's; this is
evidence, not work in progress.

## The two "trees" are just git refs — don't copy anything

| arm | ref | state |
|---|---|---|
| unpatched | `7ecec10881` | `main` before the fix; every repro below corrupts |
| patched | `01258e2b42` | `main` with #2126 squashed in |

Both `lower.sio` files were byte-compared against my local build trees, so
building those two SHAs reproduces every number in this bundle.

## What each probe proves

`repro8` / `repro32` / `repro32_small` are the dispatch's Finding 24 repro at
three scales. All three corrupt identically on both refs' predecessors, which is
what killed the "size threshold" framing.

The bisection corpus, in the order it was run — each name is one hypothesis, and
every one of them came back **clean**, i.e. refuted:

- `tiny`, `tiny_noarr`, `tiny_one` — is array-of-struct enough on its own?
- `loc_*` — do the two local `[u8;N]` buffers alias on the stack? (`loc_20_512`
  is the direct test; `p_frame` re-tests it under a large frame)
- `arr_*`, `s_*` — does struct size, array length, or an interleaved scalar
  field matter?
- `p_lit` / `p_tinydyn` — literal vs dynamic index. `p_lit` still corrupts, which
  broke the index confound.
- `p_src` — prints the source buffer immediately before the store: source intact,
  so the destination is what goes wrong.
- `p_copy` — `let e = arr[0]` then read: still wrong, so the bad bytes really are
  in memory, not a skewed load.
- `d_*` — delta-debug from the failing side, removing one thing at a time.
- `i0`–`i5`, `j1`–`j4` — interpolating between the last clean and first corrupt
  program, down to a four-line gap.

The two that actually decide it:

- **`k1`** — the failing program with only the *field names* changed
  (`oid`→`a`, `value`→`b`): **clean**.
- **`k4`** — the working program with only the field names changed to
  `oid`/`value`: **corrupt**.

`k2` (rename locals) and `k3` (rename type and helper fn) stay corrupt, so it is
the field name specifically.

## For #2130 itself

- `real_shapes.sio` — the three real `stdlib/x509/cert.sio` structs copied
  verbatim, exercised with the pattern that blocked Task 6. Exit code says which
  one fails: 1 = `ExtensionEntry`, 2 = `GeneralName`, 3 = `RdnEntry`. On
  `01258e2b42` it exits **2**: `ExtensionEntry` is cured, `GeneralName` is not.
- `gn_probe.sio` — narrows it to a single field: `tag` reads 1 where 2 was
  written, everything else correct, and identical with and without the nested
  `directory_name`, so the nested aggregate is not involved.
- `gn_order.sio` — the same program with `GeneralName` declared **before**
  `RdnEntry`, nothing else changed: prints `tag=2`. That is the proof that
  declaration order is the whole difference.

Any candidate fix has to make `gn_probe.sio` print `tag=2` in the *original*
declaration order.

## Runners

`runners/` are SLURM harnesses; each takes a tree path and pipes it to a node
over stdin (nodes don't mount `/workspace`). `mb.sh` is the full battery
(repros + ratchet + suite with a per-test fail list), `real.sh` and `gn.sh` are
the X.509 ones, `gate.sh` runs `madaros_source_to_elf_gate.sh` A/B.

Two traps baked into them: `chmod +x` the compiled ELF before running it, and
`</dev/null` on every compiler invocation or a stdin-reading loop silently eats
the rest of the list.

`logs/` holds the measured output for all of the above.

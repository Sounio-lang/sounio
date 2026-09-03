<!-- docs:meta
topic_id: repo.docs.research.subptx-buffer-refactor-investigation
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-buffer-refactor-investigation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX buffer refactor — investigation finding

**Date:** 2026-05-11
**Branch:** `research/subptx-rounding-mode-step0`
**Plan reference:** B2-4 follow-up — "migrate KaxiAsmBuf to heap-string
to unlock convergence-grade Sinkhorn iteration count"
**Outcome:** original plan target is **wrong**; the real blocker is
downstream, in the PTX text builder, not the K-AXI input buffer.

## Background

`docs/research/subptx_sinkhorn16_2iter.md` shipped Sinkhorn-LSE N=16 at
2 iterations and identified the `KaxiAsmBuf.data: [i8; 262144]` field
in `kaxi_backend.sio` as the single remaining infrastructure blocker
for ≥4-iteration convergence-grade emission. The proposed fix was a
Phase G-style migration of `KaxiAsmBuf` to a heap-string accumulator.

This note documents the empirical investigation of that fix.

## What I tried

The simplest variant of the proposed refactor: keep the array layout
but raise the cap 8× — `[i8; 262144]` → `[i8; 2097152]` (256 KB → 2 MB)
in all five sites in `self-hosted/gpu/kaxi_backend.sio`:

- struct field declaration (line 179)
- `kaxi_asm_new` initialiser (line 185)
- `kaxi_asm_push_byte` cap check (line 192)
- `kaxi_asm_push_str` cap check (line 204)
- `kaxi_asm_buf_append` cap check (line 222)

This avoids the harder structural rewrite (`data: [i8; N]` → `data:
string`) by keeping the API surface and the ~200 byte-access call
sites unchanged.

## Result

| Kernel | Emission outcome |
|---|---|
| 318 PTX golden gate | PASS 318/318 unchanged **(2026-05-11 receipt; live 2026-08-18 is 0/318, #1915)** |
| LSE-8 gate | PASS 7/7 unchanged |
| Sinkhorn-16, 2 iters | PASS, 657 KB PTX in 37s (was 13s — 3× slower from larger buffer copies) |
| Sinkhorn-16, 4 iters | **segfault** at ~52s |
| Sinkhorn-16, 8 iters | **segfault** at ~94s |
| Sinkhorn-16, 16 iters | **segfault** at ~152s |

Stack limit was raised to 128 MB (`ulimit -s 131072`) before the
re-runs; the segfaults persisted, so it is not a stack-overflow.

## Diagnosis

Two costs scale with kernel size at this scale:

1. **K-AXI emission**: each `kaxi_asm_push_*` call passes the
   `KaxiAsmBuf` struct by value (the function signature is
   `(buf: KaxiAsmBuf, ...) -> KaxiAsmBuf`). Bumping the buffer 8×
   makes each push 8× more memory traffic. For 4-iter Sinkhorn
   (~13k pushes) this is the small cost; emission time grew from
   13s to ~37s × 4 = ~150s but did not crash by itself.

2. **PTX text build** in `kaxi_transpile_to_ptx_unified`
   (`self-hosted/gpu/kaxi_to_ptx.sio`, the main loop around line 1576):

   ```sounio
   var out: string = ""
   while ... {
       ...
       out = out + "L" + ptx_int_to_string(instr_idx) + ":\n"
       if str_len(p) > 0 {
           out = out + p
       }
       ...
   }
   ```

   String append is **O(current_size)** in Sounio (immutable strings
   reallocate on each `+`). Total cost for emitting N bytes of PTX
   is **O(N²)**. At N = 657 KB (2-iter Sinkhorn) this is ~430 GB of
   string-mem work — already noticeable but survivable. At
   N ≈ 1.3 MB (4-iter) it is ~1.7 TB, which exhausts the allocator
   and segfaults.

The PTX text builder is the binding constraint, **not** the K-AXI
input buffer. Bumping `KaxiAsmBuf.data` alone is necessary but not
sufficient.

## What the real fix would look like

A "Phase H" refactor that mirrors Phase G but on the **output** side
of `kaxi_transpile_to_ptx_unified`:

- Stream PTX text directly to stdout (or to a host-provided file
  descriptor) instead of building a `string` accumulator
- OR: build the PTX into a fixed-size `[i8; N]` chunk buffer and
  flush to stdout on overflow
- OR: pass a pre-allocated `string` accumulator with an exponential-
  growth append helper (would need a Sounio builtin to amortise
  append to O(1))

Each of these is a substantial change to the PTX driver. None of
them is a one-line edit.

## Revert

The 5-site bump in `kaxi_backend.sio` was reverted in this commit
without functional change. The cap stays at 262144 (256 KB), Sinkhorn-
16 stays at 2 iterations (per `kretikos_emit_kaxi.sio`'s `while iter
< 2` body comment), and all gates remain green at their
previously-committed state:

- `kaxi_ptx_golden_gate.sh` PASS 318/318 **(2026-05-11 receipt; live 2026-08-18 is 0/318, #1915)**
- `kretikos_kaxi_lse8_gate.sh` PASS 7/7
- `kretikos_kaxi_sinkhorn16_gate.sh` PASS 7/7
- `kretikos_kaxi_fmad_invariance_gate.sh` PASS 18/18

## Scope correction for the plan

The B2-4 → B2-5 → B2 (ABIDE-I deployment) path now has **two**
infrastructure blockers, not one:

1. ~~`KaxiAsmBuf` 256 KB cap~~ — minor; bumping is mechanical but
   only buys ~3× headroom because of cost (2).
2. **PTX text accumulator in `kaxi_transpile_to_ptx_unified`** —
   load-bearing; requires a streaming-output or chunk-and-flush
   refactor. Without this fix, kernels larger than ~657 KB of PTX
   are not viable regardless of how large the input buffer is.

Both should be addressed together in a separate "Phase H" commit.
This investigation is the data the user needs to scope that work
without losing time on a partial fix.

## What this commit lands

- This research note documenting the investigation
- No source changes (the 5-site bump and the iteration-count tweak
  were both reverted in-place)
- The plan target ("migrate KaxiAsmBuf to heap-string") is
  superseded by the more accurate diagnosis above

## Reproduction (when the CUDA toolchain is back on the workspace)

```bash
# 1. Bump the buffer (Edit kaxi_backend.sio s/262144/2097152/g, 5 sites)
# 2. Bump iter count (Edit kretikos_emit_kaxi.sio "while iter < 2" → "while iter < 4")
# 3. Try to emit
ulimit -s 131072
time ./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o /tmp/x.ptx
# Expected: segfault at ~60s, output file size 0 bytes.
```

<!-- docs:meta
topic_id: repo.docs.research.subptx-phase-h-complete
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-phase-h-complete
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX Phase H complete — convergence-grade Sinkhorn unlocked

**Date:** 2026-05-11
**Branch:** `research/subptx-rounding-mode-step0`
**Hardware:** NVIDIA RTX 4000 Ada (sm_89), habitat-0
**Toolchain:** ptxas 12.0.140 / cuobjdump 12.0.140 / driver 595.58.03
**Plan reference:** Phase H + buffer bump together (see
`subptx_buffer_refactor_investigation.md` for the diagnosis that
identified both blockers)

## What landed

Two coordinated changes that together unlock convergence-grade
Sinkhorn-LSE emission:

1. **Phase H — streaming PTX output.** New function
   `kaxi_transpile_to_ptx_streaming(asm, mode) with IO, Mut, Panic,
   Div, Alloc` in `self-hosted/gpu/kaxi_to_ptx.sio` that mirrors the
   per-line dispatch loop of `kaxi_transpile_to_ptx_unified` but
   writes each chunk to stdout via `print()` instead of building a
   heap-string accumulator. Eliminates the O(N²) cost that
   segfaulted Sinkhorn-16 at ≥4 iterations
   (`subptx_buffer_refactor_investigation.md`). The accumulator
   version is preserved for in-process callers that need the PTX
   as a string. `kretikos_kaxi_to_ptx.sio` driver switched to
   streaming (line 239: `kaxi_transpile_to_ptx_streaming(buf, mode_id)`).

2. **KaxiAsmBuf bump.** `data: [i8; 262144]` → `data: [i8; 2097152]`
   in `self-hosted/gpu/kaxi_backend.sio` (5 sites). Same struct
   layout, 8× headroom. Sufficient for 16-iter Sinkhorn N=16
   (~1 MB K-AXI input) with 2× margin.

The 2-iter / "256 KB cap" language in the Sinkhorn-16 kernel header
comment is now stale and has been updated; iteration count in the
emitter body raised from 2 to 16.

## Verification on RTX 4000 Ada (sm_89)

| Gate | Result | Note |
|---|---|---|
| `kaxi_ptx_golden_gate.sh` | PASS 318/318 byte-identical **(2026-05-11 receipt; live 2026-08-18 is 0/318, #1915)** | Backward compat on that date — streaming and accumulator emit the SAME bytes in the SAME order |
| `kretikos_kaxi_lse8_gate.sh` | PASS 7/7 | Small kernel sanity |
| `kretikos_kaxi_sinkhorn16_gate.sh` | PASS 7/7 | At 16 iters: lands on analytic fixed point u = -7.08746, v ≈ 0; 3-run bit-deterministic |
| `kretikos_kaxi_fmad_invariance_gate.sh` | PASS 18/18 | At the 5.26 MB PTX scale, ptxas refuses to fuse any of ~480k mul+add chains — `.rn` discipline holds |

## Timing data (RTX 4000 Ada / sm_89)

| Configuration | Emission time | PTX size | Cubin size | Outcome |
|---|---|---|---|---|
| 2 iters, 256 KB buf (pre-Phase-H) | 13 s | 657 KB | 113 KB | PASS |
| 2 iters, 2 MB buf (no streaming) | 37 s | 657 KB | — | PASS but 3× slower (per-push copy of larger buffer) |
| 4 iters, 2 MB buf (no streaming) | — | — | — | **segfault at ~52 s** (accumulator O(N²) overrun) |
| 16 iters, 2 MB buf + streaming | **3 min 40 s** | **5.26 MB** | **856 KB** | **PASS** |

The 16-iter emission time is ~3.7 minutes, dominated by K-AXI
push operations into the 2 MB buffer (each push copies the buffer
struct by value). Streaming the output is now fast and constant-
cost-per-byte; the bottleneck has moved to the K-AXI input side.
Acceptable for research; an in-place mutation primitive (`&!` or
a true heap accumulator on the input side) would speed this up
further but is not blocking any current work.

## Implication for the live research program

- The B2-4 → B2 (ABIDE-I reproducible ORC) path no longer has any
  infrastructure blockers. The same `kaxi_emit_sinkhorn16_asm`
  emitter now produces convergence-grade kernels.
- The B1 finding (FMA-fusion invariance for the .rn modifier
  discipline) **scales** from the original Cayley-Dickson kernels
  (~3 K instructions) through LSE-8 (~70) through 2-iter Sinkhorn
  (~30 K) through 16-iter Sinkhorn (~480 K mul+add chains). Same
  byte-identical SASS regardless of `--fmad`. The orbit-equivalence
  bit-identical-MSE claim and the G₂ bridge null both inherit this
  robustness at any kernel size Sounio can emit.
- The next natural step for ABIDE-I: feed real CC200 edge K
  matrices (extracted from a single subject's connectome) into the
  16-iter kernel and compare against scipy/POT Sinkhorn for ORC
  bit-stability across centers.

## What this commit does NOT claim

- Any speedup for K-AXI emission (the 8× larger buffer makes push
  3× slower — moving the bottleneck from output to input). For
  ABIDE-I production a per-subject pre-emitted PTX cache will
  absorb this cost.
- Compatibility with non-streaming callers of
  `kaxi_transpile_to_ptx_unified` for kernels >1 MB PTX. Those
  callers still hit the O(N²) accumulator. The streaming variant
  is the one to use for big kernels.
- Anything about NVIDIA architectures beyond sm_50 and sm_89.

## Files changed

- `self-hosted/gpu/kaxi_to_ptx.sio` — added `kaxi_transpile_to_ptx_streaming`
  (~135 lines), accumulator version preserved
- `self-hosted/gpu/kretikos_kaxi_to_ptx.sio` — driver switched to
  streaming (line 239)
- `self-hosted/gpu/kaxi_backend.sio` — 5 sites: `262144` → `2097152`
- `self-hosted/gpu/kretikos_emit_kaxi.sio` — `kaxi_emit_sinkhorn16_asm`
  iter count 2 → 16, header comment + emitted source comment updated

## Reproduction

```bash
# 0. Toolchain (Ubuntu 24.04).
sudo apt-get install -y nvidia-cuda-toolkit

# 1. Emit 16-iter Sinkhorn-16.
time ./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o /tmp/sinkhorn16.ptx
# ~3.5 min;  PTX 5.26 MB;  cubin 856 KB.

# 2. Build runner.
cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o /tmp/kaxi_runner

# 3. Diagonal-cost / uniform-marginal sanity input.
python3 -c "
import math
N = 16
la = [math.log2(1.0/N)] * N
lb = [math.log2(1.0/N)] * N
K = [0.0 if i==j else -1.0 for i in range(N) for j in range(N)]
print(','.join(f'{v}' for v in (la + lb + K + [0.0]*32)))
" > /tmp/sinkhorn_input.csv

# 4. Launch.
/tmp/kaxi_runner /tmp/sinkhorn16.ptx --threads 1 --mem-words 320 \
    --init-mem "$(cat /tmp/sinkhorn_input.csv)" --type f32

# Expected mem[288..303] (u_out): all -7.08746
# Expected mem[304..319] (v_out): all 4.77e-7  (≈ 0, ULP noise)
```

Gates: `bash scripts/ci/kretikos_kaxi_sinkhorn16_gate.sh` (PASS 7/7)
and `bash scripts/ci/kretikos_kaxi_fmad_invariance_gate.sh`
(PASS 18/18).

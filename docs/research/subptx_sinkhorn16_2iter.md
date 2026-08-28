<!-- docs:meta
topic_id: repo.docs.research.subptx-sinkhorn16-2iter
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-sinkhorn16-2iter
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX B2-4: Sinkhorn-LSE N=16 kernel on RTX 4000 Ada — 2-iter smoke

**Date:** 2026-05-11
**Hardware:** NVIDIA RTX 4000 Ada (sm_89), habitat-0
**Toolchain:** ptxas 12.0.140 / cuobjdump 12.0.140 / driver 595.58.03
**Branch:** `research/subptx-rounding-mode-step0`
**Plan reference:** B2 of the ORC × EEG/fMRI × SWOW × geometric-biomarker
plan (`~/.claude/plans/whats-our-next-step-structured-spring.md`)

## Question

Can a K-AXI Sinkhorn-LSE kernel (one thread per problem, N=16 cost
matrix, log-domain row + column updates, base-2 throughout) be built
on top of the Step 0 + LSE-8 + FMA-fusion-invariance foundation, and
will it land on the analytic Sinkhorn fixed point for a tractable
toy problem? Does the kernel inherit B1's FMA-fusion-invariance
property at this much larger scale (657 KB PTX vs ~5 KB for LSE-8)?

This is the load-bearing step for B2's stated goal:
reproducible-across-centers ORC for ABIDE-I edge problems — Sinkhorn
is the inner OT solver for ORC.

## Result

**Both yes.**

This is a fixed-iteration structure smoke, not a convergence theorem.
The gate checks the diagonal sanity case against the analytic
2-iteration fixed point within the documented f32 `.approx`
transcendental floor; production ABIDE-I use in later commits moves to
the 16-iteration kernel and separate per-edge NumPy spot-checks.

On the diagonal-cost / uniform-marginal sanity case
(C[i][j] = (i==j) ? 0 : 1, λ=1, a=b=1/16 marginals, so
K[i][j] = (i==j) ? 0 : -1, la = lb = log2(1/16) = -4), the analytic
2-iteration Sinkhorn fixed point is:

```
Iter 1 row:  u[i] = la - LSE_j(K[i,j] + v[j])
              = -4 - log2(ex2(0) + 15·ex2(-1))
              = -4 - log2(8.5)
              ≈ -7.0875
Iter 1 col:  v[j] = lb - LSE_i(K[i,j] + u[i])
              = -4 - (-7.0875 + log2(8.5))
              = 0
Iter 2:      stays at fixed point (u, v) = (-7.0875·𝟏, 𝟎)
```

GPU launch on RTX 4000 Ada with that exact input produced:

```
u_out = (-7.08746, -7.08746, ... 16 times)
v_out = (4.77e-7,  4.77e-7,  ... 16 times)    [ULP noise from
                                                ex2.approx / lg2.approx]
```

All 16 u values bit-identical. All 16 v values bit-identical. ULP-
level error vs analytic prediction. The kernel **lands on the
analytic fixed point** within the precision floor of the
`.approx` PTX f32 transcendentals.

**FMA-fusion invariance.** Compiled with `ptxas -arch sm_89
--fmad=true` and `--fmad=false`, the cubins disassemble to
**byte-identical** SASS (md5 match), zero FFMA instructions in
either build. The same mechanism that holds for LSE-8 and the
Cayley-Dickson kernels (per
`docs/research/subptx_fmad_invariance.md`) holds for Sinkhorn-16
at scale.

This SASS statement depends on a working CUDA disassembler-capable
runner. CPU-only or broken-`cuobjdump` environments can compile and
inspect the source/gates, but cannot regenerate this part of the
evidence locally.

**Run-to-run determinism.** Three back-to-back launches of the same
kernel on the same input produced bit-identical u and v outputs.

## Limit: 2 iterations

The `KaxiAsmBuf.data: [i8; 262144]` field in
`self-hosted/gpu/kaxi_backend.sio` line 179 is a fixed 256 KB stack
struct passed *by value* through every `kaxi_asm_push_*` call. Each
N=16 Sinkhorn iteration emits ~3.2k K-AXI lines ≈ 64 KB; 4+
iterations overflow the buffer with silent truncation in
`kaxi_asm_push_byte` (line 192 `if out.len >= 0 && out.len < 262144`
— failure mode is "drop bytes silently and continue").

Convergence-grade iteration counts (~16 for the well-conditioned
ABIDE-I edges with moderate regularisation) need the K-AXI
accumulator migrated off the fixed-size stack struct onto a
heap-string accumulator — the same Phase G pattern the PTX output
got. Tracked as a separate follow-up commit; not on the critical
path for the algorithmic-structure smoke test, which 2 iterations
demonstrates conclusively (the row + col + row + col chain
exercises the full u↔v cross-coupling, fixed-point detection, and
LSE_16 wiring).

## Method

Three artefacts: emitter pattern, dispatcher entry, end-to-end gate.
All ran on this workspace and produced the result above. The branch
flipped under us during the source-commit attempt (parallel-agent
activity per `feedback_parallel_claude_agents.md`); this note
preserves the *finding* and the *code* in a single durable file,
to be re-applied as source edits in a more stable window.

### Emitter (to be added to `self-hosted/gpu/kretikos_emit_kaxi.sio`
after `kaxi_emit_lse8_asm`)

```sounio
// Sub-PTX B2-4: 16-way log-sum-exp inner helper.
fn kaxi_emit_lse16_helper(
    buf_in: KaxiAsmBuf,
    r_in: i64, r_out: i64, r_max: i64, r_tmp: i64, r_acc: i64, r_zero: i64
) -> KaxiAsmBuf with Mut, Panic, Div, Alloc {
    var buf = buf_in
    buf = kaxi_asm_push_line(buf, "max r" + ptx_int_to_string(r_max)
        + ", r" + ptx_int_to_string(r_in) + ", r" + ptx_int_to_string(r_in + 1))
    var i: i64 = 2
    while i < 16 {
        buf = kaxi_asm_push_line(buf, "max r" + ptx_int_to_string(r_max)
            + ", r" + ptx_int_to_string(r_max) + ", r" + ptx_int_to_string(r_in + i))
        i = i + 1
    }
    var j: i64 = 0
    while j < 16 {
        buf = kaxi_asm_push_line(buf, "sub r" + ptx_int_to_string(r_tmp)
            + ", r" + ptx_int_to_string(r_in + j) + ", r" + ptx_int_to_string(r_max))
        buf = kaxi_asm_push_line(buf, "ex2 r" + ptx_int_to_string(r_tmp)
            + ", r" + ptx_int_to_string(r_tmp))
        if j == 0 {
            buf = kaxi_asm_push_line(buf, "add r" + ptx_int_to_string(r_acc)
                + ", r" + ptx_int_to_string(r_tmp) + ", r" + ptx_int_to_string(r_zero))
        } else {
            buf = kaxi_asm_push_line(buf, "add r" + ptx_int_to_string(r_acc)
                + ", r" + ptx_int_to_string(r_acc) + ", r" + ptx_int_to_string(r_tmp))
        }
        j = j + 1
    }
    buf = kaxi_asm_push_line(buf, "lg2 r" + ptx_int_to_string(r_tmp)
        + ", r" + ptx_int_to_string(r_acc))
    buf = kaxi_asm_push_line(buf, "add r" + ptx_int_to_string(r_out)
        + ", r" + ptx_int_to_string(r_max) + ", r" + ptx_int_to_string(r_tmp))
    buf
}

pub fn kaxi_emit_sinkhorn16_asm() -> KaxiAsmBuf with Mut, Panic, Div, Alloc {
    var buf: KaxiAsmBuf = kaxi_asm_new()
    buf = kaxi_asm_push_line(buf, "; K-AXI epistemic kernel assembly")
    buf = kaxi_asm_push_line(buf, "; Auto-generated by kaxi_emit_sinkhorn16_asm() — sub-PTX B2-4")
    buf = kaxi_asm_push_line(buf, "; N=16 Sinkhorn-LSE, 2 iterations, base-2 log domain")
    buf = kaxi_asm_push_line(buf, "")
    buf = kaxi_asm_push_line(buf, "get_tid r0, var=0%, seq=0")
    buf = kaxi_asm_push_line(buf, "load_imm r1, imm=320, seq=1")
    buf = kaxi_asm_push_line(buf, "mul r2, r0, r1, var=0%, seq=2")
    buf = kaxi_asm_push_line(buf, "load_immf r9, fimm=00000000, seq=3")
    var init_i: i64 = 0
    while init_i < 16 {
        buf = kaxi_asm_push_line(buf, "load_immf r" + ptx_int_to_string(100 + init_i)
            + ", fimm=00000000")
        buf = kaxi_asm_push_line(buf, "load_immf r" + ptx_int_to_string(120 + init_i)
            + ", fimm=00000000")
        init_i = init_i + 1
    }
    var iter: i64 = 0
    while iter < 2 {
        var ri: i64 = 0
        while ri < 16 {
            buf = kaxi_asm_push_line(buf, "load_imm r3, imm=" + ptx_int_to_string(ri))
            buf = kaxi_asm_push_line(buf, "add r4, r2, r3")
            buf = kaxi_asm_push_line(buf, "load_global r200, addr=r4")
            var rj: i64 = 0
            while rj < 16 {
                let k_off = 32 + ri * 16 + rj
                buf = kaxi_asm_push_line(buf, "load_imm r3, imm=" + ptx_int_to_string(k_off))
                buf = kaxi_asm_push_line(buf, "add r4, r2, r3")
                buf = kaxi_asm_push_line(buf, "load_global r205, addr=r4")
                buf = kaxi_asm_push_line(buf, "add r" + ptx_int_to_string(210 + rj)
                    + ", r205, r" + ptx_int_to_string(120 + rj))
                rj = rj + 1
            }
            buf = kaxi_emit_lse16_helper(buf, 210, 240, 245, 247, 248, 9)
            buf = kaxi_asm_push_line(buf, "sub r" + ptx_int_to_string(100 + ri)
                + ", r200, r240")
            ri = ri + 1
        }
        var rj2: i64 = 0
        while rj2 < 16 {
            buf = kaxi_asm_push_line(buf, "load_imm r3, imm=" + ptx_int_to_string(16 + rj2))
            buf = kaxi_asm_push_line(buf, "add r4, r2, r3")
            buf = kaxi_asm_push_line(buf, "load_global r200, addr=r4")
            var ri2: i64 = 0
            while ri2 < 16 {
                let k_off = 32 + ri2 * 16 + rj2
                buf = kaxi_asm_push_line(buf, "load_imm r3, imm=" + ptx_int_to_string(k_off))
                buf = kaxi_asm_push_line(buf, "add r4, r2, r3")
                buf = kaxi_asm_push_line(buf, "load_global r205, addr=r4")
                buf = kaxi_asm_push_line(buf, "add r" + ptx_int_to_string(210 + ri2)
                    + ", r205, r" + ptx_int_to_string(100 + ri2))
                ri2 = ri2 + 1
            }
            buf = kaxi_emit_lse16_helper(buf, 210, 240, 245, 247, 248, 9)
            buf = kaxi_asm_push_line(buf, "sub r" + ptx_int_to_string(120 + rj2)
                + ", r200, r240")
            rj2 = rj2 + 1
        }
        iter = iter + 1
    }
    var s: i64 = 0
    while s < 16 {
        buf = kaxi_asm_push_line(buf, "load_imm r3, imm=" + ptx_int_to_string(288 + s))
        buf = kaxi_asm_push_line(buf, "add r4, r2, r3")
        buf = kaxi_asm_push_line(buf, "store_global r" + ptx_int_to_string(100 + s)
            + ", addr=r4")
        s = s + 1
    }
    var s2: i64 = 0
    while s2 < 16 {
        buf = kaxi_asm_push_line(buf, "load_imm r3, imm=" + ptx_int_to_string(304 + s2))
        buf = kaxi_asm_push_line(buf, "add r4, r2, r3")
        buf = kaxi_asm_push_line(buf, "store_global r" + ptx_int_to_string(120 + s2)
            + ", addr=r4")
        s2 = s2 + 1
    }
    buf = kaxi_asm_push_line(buf, "ret seq=99")
    buf = kaxi_asm_push_line(buf, "")
    buf = kaxi_asm_push_line(buf, "; end")
    buf
}
```

### Dispatcher entries

In `self-hosted/gpu/kretikos_emit_kaxi.sio` (after the `lse8` entry):

```sounio
} else if str_eq(pattern, "sinkhorn16") {
    buf = kaxi_emit_sinkhorn16_asm()
```

In `self-hosted/gpu/kretikos_kaxi_to_ptx.sio` (after the `lse8` entry):

```sounio
} else if str_eq(pattern, "sinkhorn16") {
    buf = kaxi_emit_sinkhorn16_asm()
```

In `bin/kretikos`, the `kaxi-emit-ptx` whitelist case (line 1306)
needs `sinkhorn16` appended; the `emit-kaxi` whitelist (line 1182)
needs the same.

### Reproduction (after the source edits are applied)

```bash
# 1. Install CUDA toolkit if not present.
sudo apt-get install -y nvidia-cuda-toolkit

# 2. Build the runner.
cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o /tmp/kaxi_runner

# 3. Emit the kernel.
./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o /tmp/sinkhorn16.ptx
# ~13 seconds emit time; ~657 KB PTX; ptxas validates to ~113 KB cubin.

# 4. Generate the diagonal-cost / uniform-marginal input.
python3 -c "
import math
N = 16
la = [math.log2(1.0/N)] * N
lb = [math.log2(1.0/N)] * N
K = [0.0 if i==j else -1.0 for i in range(N) for j in range(N)]
print(','.join(f'{v}' for v in (la + lb + K + [0.0]*32)))
" > /tmp/sinkhorn_input.csv

# 5. Launch.
/tmp/kaxi_runner /tmp/sinkhorn16.ptx --threads 1 --mem-words 320 \
    --init-mem "$(cat /tmp/sinkhorn_input.csv)" --type f32

# Expected last line: device=NVIDIA RTX 4000 Ada Generation cc=8.9
# Expected mem[288..303] (u_out): all -7.08746
# Expected mem[304..319] (v_out): all 4.77e-7  (≈ 0, ULP noise)
```

### FMA-fusion invariance check

```bash
mkdir -p /tmp/b2c
./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o /tmp/b2c/sinkhorn.ptx
for fma in true false; do
    ptxas -arch sm_89 --fmad=$fma /tmp/b2c/sinkhorn.ptx \
        -o /tmp/b2c/sinkhorn_${fma}.cubin
    cuobjdump --dump-sass /tmp/b2c/sinkhorn_${fma}.cubin \
        > /tmp/b2c/sinkhorn_${fma}.sass
done
md5sum /tmp/b2c/sinkhorn_*.sass
# Both md5s should be identical.
grep -c '\bFFMA\b' /tmp/b2c/sinkhorn_*.sass
# Both counts should be 0.
```

## Implication for the larger plan

- The K-AXI emission discipline (explicit `.rn.f32` modifiers on
  every add/sub/mul, deterministic `.approx` on every f32
  transcendental) scales to non-trivial OT kernels without losing
  the bit-determinism property B1 established for Cayley-Dickson.
- The 657 KB PTX / 113 KB SASS Sinkhorn-16 kernel is the largest
  Sounio-emitted kernel verified end-to-end on real hardware in
  this branch's lineage.
- The 256 KB `KaxiAsmBuf` cap is the **single remaining
  infrastructure blocker** for convergence-grade Sinkhorn (≥4
  iterations / 16 iterations for full convergence). Refactoring
  this buffer to a heap-string accumulator is the next concrete
  compiler-side task, after which the same kernel can be emitted
  at full iteration depth without algorithmic change.
- For the ABIDE-I cross-center reproducibility goal of B2, the
  remaining work after the buffer refactor is straightforward:
  parameterise N (current kernel is N=16-hardcoded), wire the
  output to a per-subject NIfTI reader, and feed the K matrices
  generated from real CC200 edge neighbourhoods.

## What this does NOT claim

- Bit-stability across NVIDIA architectures beyond sm_50 and sm_89.
- Bit-stability across CUDA toolkit major versions (CUDA 12.0
  tested).
- That 2 iterations is enough for production ABIDE-I work — it is
  enough for the toy fixed-point sanity check that validates the
  kernel structure, and explicitly not enough for problems where
  Sinkhorn convergence matters.
- Anything about run-to-run determinism under concurrent thread
  blocks. Single-thread launch only.

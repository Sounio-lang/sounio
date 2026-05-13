<!-- docs:meta
topic_id: repo.docs.research.subptx-fmad-invariance
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-fmad-invariance
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX FMA-fusion invariance on Sounio's Cayley-Dickson GPU kernels

**Date:** 2026-05-11
**Hardware:** RTX 4000 Ada (sm_89) on habitat-0 workspace
**Toolchain:** ptxas 12.0.140 / cuobjdump 12.0.140 / NVIDIA driver 595.58.03
**Branch:** `research/subptx-rounding-mode-step0`
**Plan reference:** B1 Path C of the ORC × EEG/fMRI × SWOW × geometric biomarker plan

**Evidence boundary.** The SASS/FFMA table below is a recorded
GPU-toolchain result from the habitat-0 RTX 4000 Ada run. The CI gate
can only re-check it when both `ptxas` and a usable `cuobjdump` are
available; if a local CUDA disassembler is absent or crashes while
dumping the generated cubin, the gate reports SKIPPED rather than
pretending that SASS evidence was regenerated.

## Question

The S-SSM zero-divisor regularization on real EEG reports a 5.6×
effect at α=0.2 with bit-identical MSE across five ZD pairs (orbit
equivalence). A reviewer could plausibly argue that the
bit-identicality is a ptxas determinism artifact: specifically, that
ptxas's FMA-contracting choice (`mul + add` → `fma`) happens to land
where it does and the orbit equivalence rides on that coincidence.

This note tests that artifact concern empirically by compiling the
shipped Sounio GPU sedenion / octonion kernels with `ptxas --fmad=true`
(default) and `ptxas --fmad=false`, disassembling both, and comparing.

## Method

1. For each kernel in `{octonion_mul, sedenion_mul, octonion_associator,
   sedenion_associator}`, emit the f32 PTX via the shipped path:

   ```bash
   ./bin/kretikos kaxi-emit-ptx <pattern> --f32 -o /tmp/<pattern>.ptx
   ```

2. Compile each PTX file to a cubin twice, targeting sm_89, toggling
   only `--fmad`:

   ```bash
   ptxas -arch sm_89 --fmad=true  <pattern>.ptx -o <pattern>_true.cubin
   ptxas -arch sm_89 --fmad=false <pattern>.ptx -o <pattern>_false.cubin
   ```

3. Disassemble each cubin to SASS text with
   `cuobjdump --dump-sass`.

4. md5 the two SASS files and count `FMUL`, `FADD`, `FFMA` opcodes.

5. Cross-check on sm_50 (Sounio's hardcoded `.target` baseline) to
   confirm the finding is not arch-specific to Ada.

## Result

| Kernel | FMUL | FADD | FFMA | md5 fmad=true == md5 fmad=false |
|---|---|---|---|---|
| octonion_mul | 64 | 64 | 0 | YES |
| sedenion_mul | 256 | 256 | 0 | YES |
| octonion_associator | 256 | 288 | 0 | YES |
| sedenion_associator | 1024 | 1040 | 0 | YES |

Same result on sm_50: SASS byte-identical, zero FFMA.

## Why ptxas refuses to fuse

The K-AXI lowerer (`self-hosted/gpu/kaxi_to_ptx.sio` lines 687/750)
emits every f32 mul as `mul.rn.f32` and every f32 add as `add.rn.f32`
with the rounding-mode suffix explicit. For ptxas to fuse a `mul.rn`
followed by an `add.rn` into a single `fma.rn`, the rounding contract
would have to change — from "round after mul, round after add" (two
rounding events) to "single round after the fused multiply-add"
(one rounding event). These are different operations under IEEE-754
binary32 semantics, and ptxas correctly refuses to silently change
them. The explicit `.rn` modifiers act as an implicit
fma-contract-off directive *per pair of operations*, without any
need for a CUDA compile flag.

The post-2026-05-11 Sub-PTX Step 0 work (`research/subptx-rounding-mode-step0`,
commit `e611776b`) added the same control for f64 ops as a
`round=rN` attribute on K-AXI lines. The f32 path already had it
hardcoded.

## Interpretation

1. **The orbit-equivalence bit-identical-MSE claim on the 5 ZD pairs
   is robust to FMA-contracting on the tested CUDA 12.0 / sm_50 /
   sm_89 toolchain**, the most-commonly-cited ptxas sub-PTX freedom.
   The 5.6× regularization on real EEG inherits this tested robustness
   for the algebra layer underneath.

2. **The G₂ bridge null result** (`project_g2_bridge.md`: no ASD/TD
   diff at d=0.06 on ABIDE-I CC200 eigenmodes; z≈2 is a combinatorial
   artifact) is similarly robust — it cannot be explained by ptxas
   fusing FMAs differently across runs. The null is a property of
   the algebra and the eigenmodes, not a property of the compiler.

3. **The 168-theorem dual-pathway computation** (`project_168_theorem.md`:
   E2E Sounio verification complete) inherits the same robustness for
   any GPU portion that uses these algebra kernels.

4. **Going forward**: the Sinkhorn-LSE kernel for B2 (reproducible-
   across-centers ORC) does NOT yet have explicit per-op rounding
   modes because it does not yet exist. When it is built, the same
   `.rn` discipline should be applied to its mul/add chains, AND
   special care is needed for its `exp`, `log`, and `div` ops where
   ptxas has additional sub-PTX freedoms (`.approx` vs `.full` vs
   `.rn`) that the FMA test does not cover.

## What this does NOT claim

- It does not claim the Cayley-Dickson kernels are bit-stable across
  CUDA major versions. CUDA 12.0 was tested. ptxas's internal
  scheduling may differ across major versions (the K-AXI golden gate
  bytewise-checks the *PTX text*, not the cubin or the SASS).
- It does not claim bit-stability across NVIDIA architectures.
  sm_50 and sm_89 were tested and match. Hopper (sm_90) and Blackwell
  (sm_100) were not.
- It does not address the run-to-run determinism question
  (cooperative-thread-arrangement, atomic ordering, etc.) — those
  would be different experiments. This note only addresses the
  *compile-time* deterministic-binary question.
- It does not measure numerical accuracy. The kernels are bit-stable
  in their binary form; whether that binary form is the *most
  accurate* possible is a separate question.
- It does not claim that every future CI run has regenerated the SASS
  comparison. CPU-only or broken-CUDA environments must treat the gate
  as a skip and rely on the recorded GPU evidence until a CUDA
  disassembler-capable runner is available.

## Reproduction

```bash
# Toolchain (Ubuntu 24.04):
sudo apt-get install -y nvidia-cuda-toolkit

# Emit + compile + dump:
mkdir -p /tmp/b1c
for pat in octonion_mul sedenion_mul octonion_associator sedenion_associator; do
  ./bin/kretikos kaxi-emit-ptx "$pat" --f32 -o "/tmp/b1c/${pat}.ptx"
  for fma in true false; do
    ptxas -arch sm_89 --fmad="$fma" "/tmp/b1c/${pat}.ptx" \
      -o "/tmp/b1c/${pat}_${fma}.cubin"
    cuobjdump --dump-sass "/tmp/b1c/${pat}_${fma}.cubin" \
      > "/tmp/b1c/${pat}_${fma}.sass"
  done
  echo "$pat fmad=true:  $(md5sum /tmp/b1c/${pat}_true.sass | awk '{print $1}')"
  echo "$pat fmad=false: $(md5sum /tmp/b1c/${pat}_false.sass | awk '{print $1}')"
done
```

All four pairs return identical md5 hashes.

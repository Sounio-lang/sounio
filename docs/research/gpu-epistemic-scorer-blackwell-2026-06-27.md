<!-- docs:meta
topic_id: repo.docs.research.gpu-epistemic-scorer-blackwell-2026-06-27
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.gpu-epistemic-scorer-blackwell-2026-06-27
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# GPU epistemic solver scorer on Blackwell — VERIFIED (2026-06-27)

The Sounio solver's epistemic variable-scoring heuristic runs **per-variable parallel on a
real NVIDIA Blackwell GB10 (sm_121)**, computed exactly. This is the genuine GPU-solver
showcase (vs the earlier scalar/broadcast demonstrators).

Evidence tags: **MEASURED** / **BOUNDED**. No claim rests on a projection.

## Artifact
- New K-AXI emitter pattern `epistemic_scorer` (`self-hosted/gpu/kretikos_emit_kaxi.sio`
  `kaxi_emit_epistemic_scorer_asm()` + dispatch in `kretikos_kaxi_to_ptx.sio` + `bin/kretikos`
  whitelist). Routes through `gpu_lower_to_kaxi` mode-2 auto-promotion → `sqrt.approx.f32`,
  `add.rn.f32`/`mul.rn.f32`, `ld/st.global.f32`.
- `./bin/kretikos kaxi-emit-ptx epistemic_scorer` emits a 61-line ASCII-clean
  `.visible .entry kaxi_kernel` (single packed-f32 `param_mem`:
  `[0..N)=act_mean, [N..2N)=act_var, [2N..3N)=score(out), [3N]=beta`); per thread
  `i = %tid`: `score[i] = act_mean[i] + beta*sqrt(act_var[i])` — the smt.sio Mode-0 GUM-UCB
  formula (`beta = 0.6*(1-density)^2*regime_explore_trust`; fresh state beta=0.6).
- Launcher receipt: `benchmarks/solver/gpu/run_epistemic_scorer.c`.

## MEASURED on real Blackwell GB10 (sm_121, CUDA 13)
```
ptxas -arch=sm_121 scorer.ptx -> 5696-byte CUBIN
device: NVIDIA GB10 sm_121   (native cubin AND PTX JIT)
score[i] = mean[i] + 0.6*sqrt(var[i]):
  i  mean   var     measured   expected   delta
  0  0.100  0.0400  0.220000   0.220000   1.49e-08
  1  0.200  0.0900  0.380000   0.380000   2.98e-08
  2  0.300  0.1600  0.540000   0.540000   0.00e+00
  3  0.400  0.2500  0.700000   0.700000   0.00e+00
RESULT: PASS
```

## Reproduce
```bash
./bin/kretikos kaxi-emit-ptx epistemic_scorer --no-ptxas -o scorer.ptx
# on a Blackwell host (CUDA 13):
/usr/local/cuda-13.0/bin/ptxas -arch=sm_121 scorer.ptx -o scorer.cubin
nvcc -O2 benchmarks/solver/gpu/run_epistemic_scorer.c -lcuda -o run_sc
./run_sc scorer.cubin     # expect score=[0.22,0.38,0.54,0.70], all PASS
```

## Caveats / follow-up
- f32 precision (the GPU path uses `sqrt.approx.f32`); delta ~1e-8 vs CPU f64 reference.
- Single-block launch (N = `ntid.x`); multi-block would need `ctaid*ntid+tid` indexing.
- Golden-gate coverage: `epistemic_scorer` is not yet in `scripts/ci/kaxi_ptx_golden_gate.sh`
  PATTERNS (would need captured goldens × 6 modes) — follow-up.

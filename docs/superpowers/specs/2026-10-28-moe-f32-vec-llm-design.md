<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-10-28-moe-f32-vec-llm-design
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/superpowers/specs/2026-10-28-moe-f32-vec-llm-design.md
-->

# Design: MoE F32 vec chase + LLM FFN block (plan C)

**Date:** 2026-10-28
**Orthography:** EN-UK
**Status:** Approved design-base (approach C)
**Depends on:** `docs/design/moe-f32-mmid.md` (A+B landed: dp4a, vec n=1, GraphIR, `moe_ffn_q4k`)
**Branch:** `cursor/moe-f32-mmid`
**Implementation plan:** `docs/superpowers/plans/2026-10-28-moe-f32-vec-llm.md`
**Backlog:** Notion Tasks + this plan.
**Hardware gate:** NVIDIA L4 on `gpuorangefs-r770-proxmox` via `srun` (not sbatch)
**Toolchain:** nvcc 12.6, PTX `.version 8.5` / arch `sm_89` for driver JIT

---

## Goal

Two gated steps, in order, without a forever µs chase:

1. **Vec decode:** push `mmid_q4k_vec_f32` n=1 toward **<=120 µs/run** on L4 for the lab shape; if one L4 cycle does not land it, record the ceiling and move on.
2. **MoE LLM block:** wire one MoE FFN into the Sounio llama forward path (router + 3x Q4_K `mul_mat_id` + SiLU combine), demo with packed synthetic or minimal safetensors MoE weights.

No llama.cpp runtime. No merge to `main` without human. No AI-authored PR/commit text.

---

## Non-goals

- Matching llama.cpp ~72 µs on the first cycle (aspirational reference only).
- Q5_K / Q8_0 MoE kernels.
- Full Mixtral/Qwen-MoE production loader + serving.
- Changing the dp4a n>16/64 gates already green.
- Host SiLU/combine GPU port (keep host for this plan unless it blocks the demo).

---

## Baseline (OBSERVED, L4 2026-10-27)

Lab shape: m=4096, k=2048, 128 experts, 4 slots, amax=1e5.

| path | NMSE | us/run |
|------|------|--------|
| vec n=1 | ~4.58e-13 | ~308 |
| dp4a n=16 | ~1.54e-5 | ~2294 |
| dp4a n=64 | ~1.34e-5 | ~4474 |

llama.cpp vec reference on same class of problem: ~72 µs (not a hard gate).

Existing APIs:

- Kernel: `stdlib/hetero/kernels/mmid_q4k_dp4a_f32.cu` (`mmid_q4k_vec_f32`, `mmid_q4k_dp4a_f32`)
- Runtime: `rt_mul_mat_id_q4k` (auto vec when `ntokens==1`)
- Layer: `moe_ffn_q4k` in `stdlib/hetero/moe.sio`
- Demo: `examples/hetero_moe_layer_demo.sio`
- LLM: `stdlib/llm/llama.sio` (dense FFN today; no MoE branch yet)

---

## Architecture

```
                 +------------------+
  lab / demo --> | mmid_q4k_vec_f32 | --(1 cycle L4)--> ceiling or <=120 us
                 +--------+---------+
                          |
                 rt_mul_mat_id_q4k (n=1)
                          |
                 +--------v---------+
                 |   moe_ffn_q4k    |  gate/up/down + host SiLU + weighted sum
                 +--------+---------+
                          |
                 +--------v---------+
                 | llama MoE block  |  router logits -> topk -> moe_ffn_q4k
                 +--------+---------+
                          |
                 +--------v---------+
                 | demo / tiny MoE  |  packed Q4_K or minimal safetensors
                 +------------------+
```

Two units, clear boundaries:

1. **Vec unit** - only the n=1 CUDA entry + lab harness. Input: Q4_K weights, F32 x, ids. Output: F32 dst + NMSE/us. No LLM knowledge.
2. **LLM MoE unit** - Sounio glue in `llama.sio` + demo. Calls existing `moe_topk` / `moe_ffn_q4k`. Does not reimplement kernels.

---

## Step 1 - Vec <=120 µs (one L4 cycle)

### Shape and gates

- Shape fixed: m=4096, k=2048, experts=128, slots=4, ntokens=1, amax=1e5.
- Accuracy: NMSE <= 5e-4 vs CPU dequant+GEMM (unchanged).
- Speed target: **us/run <= 120**.
- Budget: **one** L4 measure cycle (build + srun lab + record). If miss: write measured ceiling into `docs/design/moe-f32-mmid.md` and `*_lab_out.txt`, leave vec functional, proceed to Step 2. Do not open a second chase cycle under this plan.

### Allowed tactics (in-cycle)

Prefer reuse of patterns already in the file / CUDA 12.6 sm_89:

1. Larger row tiles / better occupancy for `mmid_q4k_vec_f32`.
2. Stronger use of `__ldg` / vectorised weight loads if not already saturating.
3. Optional `cp.async` staging of activation or weight nibbles into shared (only if the one-cycle edit stays small).
4. Grid remap that keeps `slots` on a fast axis without changing the public ABI of `rt_mul_mat_id_q4k`.

Out of cycle (defer): new kernel file, warp-specialised pipelines, CUTLASS, changing Q4_K layout.

### Exit criteria Step 1

| Result | Action |
|--------|--------|
| NMSE ok and us <= 120 | Update lab gate to enforce time for n=1; document in moe-f32-mmid.md |
| NMSE ok and us > 120 | Document ceiling; keep NMSE-only time gate; continue Step 2 |
| NMSE fail | Stop; fix before Step 2 |

---

## Step 2 - MoE FFN in llama forward

### Behaviour

Add a MoE FFN path beside the dense FFN in `stdlib/llm/llama.sio`:

1. Router: hidden -> logits over `n_experts` (existing dense matmul or small host GEMM for the demo).
2. `moe_topk` -> ids + route weights.
3. `moe_ffn_q4k` for gate/up/down Q4_K expert packs.
4. Write residual-style output buffer the dense FFN already uses (same dst contract as current FFN block).

Config surface (explicit, no magic):

- `n_experts`, `n_expert_used` (slots), `moe_intermediate`
- pointers/byte sizes + Q4_K strides for gate/up/down packs
- flag or dtype tag selecting dense vs MoE FFN for the layer

### Data flow

```
x[hidden]
  -> router_logits[n_experts]
  -> moe_topk -> ids[slots], w[slots]
  -> moe_ffn_q4k(x, ids, w, gate_w, up_w, down_w) -> y[hidden]
```

### Demo

- New or extended example under `examples/` (e.g. `llama_moe_ffn_demo.sio`).
- Weights: prefer **synthetic packed Q4_K** generated in-process (matches hetero_moe_layer_demo). Optional: load a tiny safetensors MoE slice via existing `stdlib/llm/safetensors.sio` if it fits without new loader work.
- Print: topk ids, ms_gate/ms_up/ms_down, and a host reference residual check (NMSE or max-abs) on a tiny shape if CPU ref is cheap; otherwise smoke `ok` + timings only for large packs.

### Error handling

- Propagate `MoeFfnResult.status` / `ok` to the llama MoE caller; do not swallow CUDA launch failures.
- Refuse MoE path when packs/strides inconsistent (experts, inter, strides) with the same validation style as `rt_mul_mat_id_q4k`.
- Dense FFN remains default when MoE config is absent.

### Exit criteria Step 2

- Demo runs on L4 (or CPU-fallback path only if CUDA unavailable - prefer L4 for the receipt).
- One forward MoE FFN call returns `ok` with finite outputs.
- Design doc + `docs/design/moe-f32-mmid.md` updated with the wiring pointer.
- No llama.cpp linkage.

---

## Testing and measurement

| Gate | How |
|------|-----|
| Vec NMSE/time | `scripts/gpu/run_mmid_q4k_l4_lab.sh` / host lab binary; artifact `mmid_q4k_dp4a_f32_lab_out.txt` |
| MoE layer smoke | `examples/hetero_moe_layer_demo.sio` still green |
| Llama MoE demo | new demo; receipt under `stdlib/hetero/kernels/` or `artifacts/` as existing demos do |
| Regress dp4a | n=16 and n=64 still pass existing us + NMSE gates |

Measure with `srun` on L4. Conda/nvcc on node as already documented in moe-f32-mmid.md.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Vec is memory-bound near ~300 µs | One-cycle rule; document ceiling; ship LLM path anyway |
| Llama MoE config sprawl | Thin wrapper over `moe_ffn_q4k`; no second MoE stack |
| Safetensors MoE real weights too heavy | Synthetic packs default |
| Claim / branch drift | `sounio-coord` claim on touched files; branch `cursor/moe-f32-mmid` |

---

## Success definition (plan C complete)

1. Step 1 executed once on L4 with a written result (hit 120 µs **or** documented ceiling).
2. Step 2: llama forward can run one MoE FFN via native hetero Q4_K F32 path.
3. Docs updated; demos runnable; NMSE gates for dp4a unchanged.

---

## Open decisions locked by this spec

- Approach **C** (short gated sequence), not A (hard kernel-first) or B (LLM-first stub).
- Speed miss does **not** block Step 2.
- Host SiLU/combine stays host for this plan.
- Synthetic Q4_K packs are the default demo weight source.

# Mixtral multi-layer MoE + prefill Implementation Plan (Plan F)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MoE on every Llama layer, support `ntok>1` via dp4a, add `llama_load_moe` for Mixtral-shaped safetensors, and prove it with synthetic + optional real L4 demos.

**Architecture:** Promote single-layer MoE fields to per-layer arrays (cap 64). Drop `li==0` and `ntok==1` from the MoE gate. Dispatch `moe_ffn_q4k`/`moe_ffn_q5k` from a model quant tag. Loader fills routers + Qx_K expert packs per layer; dense SwiGLU remains the per-layer fallback.

**Tech Stack:** Sounio (`.sio`), hetero mmid Q4_K/Q5_K (Plan E), L4 via `srun --gres=gpu:1`, branch `cursor/moe-f32-mixtral-forward`.

**Spec:** `docs/superpowers/specs/2026-10-28-moe-f32-mixtral-forward-design.md`

## Global Constraints

- No llama.cpp runtime linkage.
- No GGUF; no GPU SiLU/combine.
- Q8_0 stays hetero-only; llm MoE uses Q4_K or Q5_K via `moe_quant`.
- Dense default when `cfg.n_experts == 0`.
- Layer MoE failure => dense fallback for that layer only; do not abort forward.
- L4 with `srun` and `gpu:1`; demos `exit(0)` after DONE (CUDA teardown hang).
- ASCII only in code/docs.
- Commits human-gated: propose message only unless human says commit. Use `Assisted-by:`.
- Coord lane `moe-f32-mixtral-forward`; never merge `main`.
- Notion prefix `SIO-moe-f32` (controller).

## File map

| File | Role |
|------|------|
| `stdlib/llm/llama.sio` | Per-layer MoE fields, ready/forward, load_moe, quant dispatch in block |
| `examples/llama_moe_forward_demo.sio` | Update to fill layer-0 slot of new arrays (regression) |
| `examples/llama_moe_multilayer_demo.sio` | New synthetic multi-layer + ntok 1/4 |
| `examples/llama_mixtral_forward_demo.sio` | New real/dir-based Mixtral smoke |
| `docs/design/moe-f32-mmid.md` | Plan F receipts |
| `docs/superpowers/plans/2026-10-28-moe-f32-mixtral-forward.md` | This plan |

---

### Task 1: Per-layer MoE data model + ready helper

**Files:**
- Modify: `stdlib/llm/llama.sio` (`Llama`, `LlamaCfg`, constructors, `ll_moe_ready`)
- Modify: `examples/llama_moe_forward_demo.sio` (write into layer 0 of new arrays)
- Test: `bin/souc check examples/llama_moe_forward_demo.sio examples/llama_moe_ffn_demo.sio`

**Interfaces:**
- Consumes: existing MoE cfg fields
- Produces:
  - `LlamaCfg.moe_quant: i64` (0=Q4_K, 1=Q5_K), default 0
  - `Llama.moe_router: [CpuBuf; 64]`
  - `Llama.moe_gate_w/up_w/down_w: [*mut u8; 64]` + parallel byte/stride arrays (or stride scalars if shared)
  - `Llama.last_moe_layers_ok: i64`
  - `ll_moe_ready_layer(m, rt, use_gpu, ntok, li) -> bool` (no `li==0`, no `ntok==1`; checks packs for `li` + GPU + mmid for quant)
  - Keep thin `ll_moe_ready(...)` as `ll_moe_ready_layer(..., 0)` only if demos need it; prefer updating call sites to `_layer`

- [ ] **Step 1: Extend structs**

In `LlamaCfg` add `moe_quant: i64` default 0 in `llama_cfg` / empty cfg.

In `Llama` replace scalar MoE pack fields with arrays. Initialize all 64 slots empty/null in every `Llama { ... }` literal (`llama_load`, demos, probes).

Add `last_moe_layers_ok: i64 = 0`.

- [ ] **Step 2: Ready helper**

```sio
fn ll_moe_ready_layer(m: &Llama, rt: &Runtime, use_gpu: bool, ntok: i64, li: i32) -> bool {
    if !use_gpu || !rt.exec.ok || ntok <= 0 { return false }
    if m.cfg.n_experts <= 0 || m.cfg.n_expert_used <= 0 { return false }
    if m.cfg.n_expert_used > 16 || m.cfg.n_expert_used > m.cfg.n_experts { return false }
    if li < 0 || li >= m.cfg.layers || li >= 64 { return false }
    // quant mmid loaded:
    if m.cfg.moe_quant == 1 {
        if rt.mmid_q5k_func == 0 || rt.mmid_q5k_vec_func == 0 { return false }
    } else {
        if rt.mmid_func == 0 || rt.mmid_vec_func == 0 { return false }
    }
    let i = li as usize
    if m.moe_gate_w[i] == 0 as *mut u8 || m.moe_up_w[i] == 0 as *mut u8 || m.moe_down_w[i] == 0 as *mut u8 { return false }
    if m.moe_gate_bytes[i] <= 0 || m.moe_up_bytes[i] <= 0 || m.moe_down_bytes[i] <= 0 { return false }
    if m.moe_router[i].len <= 0 { return false }
    true
}
```

Adapt field names if you choose shared strides (document choice in report).

- [ ] **Step 3: Fix Plan D demo**

`examples/llama_moe_forward_demo.sio`: assign synthetic packs into index 0 of the new arrays; load Q4 PTX as today.

- [ ] **Step 4: Typecheck**

```bash
bin/souc check examples/llama_moe_forward_demo.sio examples/llama_moe_ffn_demo.sio
```

Expected: `check: OK`.

- [ ] **Step 5: Propose commit only**

```
llm: per-layer MoE fields and ready_layer helper

Assisted-by: Composer
```

---

### Task 2: Forward multi-layer + prefill gate

**Files:**
- Modify: `stdlib/llm/llama.sio` (`ll_forward_impl` MoE branch ~450)
- Test: typecheck demos from Task 1; full L4 in Task 4

**Interfaces:**
- Consumes: `ll_moe_ready_layer`, per-layer packs
- Produces: MoE attempted on every ready layer; `last_moe_layers_ok` count; `last_moe_used = last_moe_layers_ok > 0`

- [ ] **Step 1: Reset telemetry**

At forward entry:

```sio
m.last_moe_used = false
m.last_moe_status = 0
m.last_moe_layers_ok = 0
```

- [ ] **Step 2: Replace `li == 0 && ll_moe_ready` branch**

For each layer, after post-norm:

```sio
var used_moe = false
if ll_moe_ready_layer(m, rt, use_gpu, ntok, li) {
    // router proj using m.moe_router[li]
    // llama_moe_ffn_block with layer packs / strides / bytes
    if r.ok {
        ll_add_into(&!x, &dn)
        m.last_moe_layers_ok = m.last_moe_layers_ok + 1
        m.last_moe_used = true
        m.last_moe_status = 0
        used_moe = true
    } else {
        m.last_moe_status = r.status
    }
}
if !used_moe {
    // existing dense SwiGLU
}
```

Do **not** gate on `ntok == 1`.

- [ ] **Step 3: Typecheck**

```bash
bin/souc check examples/llama_moe_forward_demo.sio
```

Expected: `check: OK`.

- [ ] **Step 4: Propose commit only**

```
llm: MoE on all layers; allow ntok>1

Assisted-by: Composer
```

---

### Task 3: Quant dispatch in `llama_moe_ffn_block`

**Files:**
- Modify: `stdlib/llm/llama.sio` (`LlamaMoeFfnCfg` + `llama_moe_ffn_block`)
- Test: `bin/souc check examples/llama_moe_ffn_demo.sio`

**Interfaces:**
- Consumes: `moe_ffn_q4k`, `moe_ffn_q5k` from `hetero::moe`
- Produces: `LlamaMoeFfnCfg.moe_quant: i64`; block calls the matching `moe_ffn_*`

- [ ] **Step 1: Extend cfg + imports**

```sio
use hetero::moe::{moe_topk, moe_ffn_q4k, moe_ffn_q5k, MoeFfnResult}
// in LlamaMoeFfnCfg:
pub moe_quant: i64,
```

- [ ] **Step 2: Dispatch**

After topk, if `cfg.moe_quant == 1` call `moe_ffn_q5k(...)` else `moe_ffn_q4k(...)` with the same argument list.

- [ ] **Step 3: Call sites**

Forward and `llama_moe_ffn_demo` set `moe_quant` from `m.cfg.moe_quant` (0).

- [ ] **Step 4: Typecheck**

```bash
bin/souc check examples/llama_moe_ffn_demo.sio examples/llama_moe_forward_demo.sio
```

Expected: `check: OK`.

- [ ] **Step 5: Propose commit only**

```
llm: llama_moe_ffn_block Q4_K/Q5_K dispatch

Assisted-by: Composer
```

---

### Task 4: Synthetic multi-layer + prefill demo

**Files:**
- Create: `examples/llama_moe_multilayer_demo.sio`
- Test: `bin/souc check` + L4 srun `gpu:1`

**Interfaces:**
- Consumes: Tasks 1-3 APIs, `rt_load_mmid`, `llama_forward_rt`
- Produces: demo binary printing MoE telemetry for `ntok=1` and `ntok=4`

- [ ] **Step 1: Build synthetic model**

`layers=2` (or 3), `hidden=256`, `moe_inter=512`, `n_experts=8`, `n_expert_used=2`. Fill MoE packs + router for **every** layer. Dense attn/embed synthetic as Plan D demo.

- [ ] **Step 2: Run twice**

```sio
// ntok=1 then ntok=4 (fresh kv each time)
llama_forward_rt(&!m, &!kv, &tokens, ntok, &rt)
// require last_moe_used && last_moe_layers_ok == layers && last_moe_status == 0 && finite
```

Print `LLAMA MOE MULTILAYER DONE` and `exit(0)`.

- [ ] **Step 3: Verify**

```bash
bin/souc check examples/llama_moe_multilayer_demo.sio examples/llama_moe_forward_demo.sio
# L4 tar+srun --gres=gpu:1 ; expect DONE and RC=0 for both ntok modes
```

- [ ] **Step 4: Propose commit only**

```
examples: multi-layer MoE forward demo (ntok 1 and 4)

Assisted-by: Composer
```

---

### Task 5: `llama_load_moe` + Mixtral demo

**Files:**
- Modify: `stdlib/llm/llama.sio` (`llama_cfg`, add `llama_load_moe`)
- Modify: `stdlib/llm/mod.sio` (export)
- Create: `examples/llama_mixtral_forward_demo.sio`
- Test: typecheck; L4 if model dir present

**Interfaces:**
- Consumes: `st_scan_i64`, `st_load_f64`, `st_read_all`, pack load helpers
- Produces: `pub fn llama_load_moe(dir: string) -> Llama with ...`

- [ ] **Step 1: Cfg scans**

In `llama_cfg`, when present:

```sio
cfg.n_experts = st_scan_i64(f.data, f.len, "num_local_experts")
cfg.n_expert_used = st_scan_i64(f.data, f.len, "num_experts_per_tok")
cfg.moe_inter = st_scan_i64(f.data, f.len, "moe_intermediate_size")
```

Leave 0 if missing (dense).

- [ ] **Step 2: Implement `llama_load_moe`**

Clone `llama_load` attn path. For each layer, load Mixtral gate:

`model.layers.{i}.block_sparse_moe.gate.weight` -> `moe_router[i]` (transpose for proj as other weights).

Expert packs: load from documented sidecar layout under `dir`:

```text
dir/moe_q4k/layer{i}_gate.bin
dir/moe_q4k/layer{i}_up.bin
dir/moe_q4k/layer{i}_down.bin
```

Or, if raw expert tensors exist as u8/Q blobs in safetensors with known names, bind those pointers. Record exact naming in the design note. Set strides from `n_experts`, `moe_inter`, `hidden`, QK block size.

If sidecars missing: attn still loads; MoE packs stay null => dense FFN fallback.

- [ ] **Step 3: Demo**

`examples/llama_mixtral_forward_demo.sio`:

- Read model dir from first CLI arg or default path string constant documented in comments.
- `rt_open`, load mmid for `moe_quant`, `llama_load_moe`, BPE if tokenizer present.
- `llama_forward_rt` ntok=1; require finite logits when `m.loaded` and GPU ready.
- If dir missing: print `MIXTRAL DIR MISSING` and `exit(0)` (honest non-claim).
- If GPU MoE path claimed: require `last_moe_layers_ok > 0` or document why dense.

- [ ] **Step 4: Verify**

```bash
bin/souc check examples/llama_mixtral_forward_demo.sio examples/llama_moe_multilayer_demo.sio
# L4 optional real weights; always run missing-dir path locally
```

- [ ] **Step 5: Propose commit only**

```
llm: llama_load_moe + Mixtral forward demo

Assisted-by: Composer
```

---

### Task 6: Docs + Notion

**Files:**
- Modify: `docs/design/moe-f32-mmid.md`
- Modify: this plan checkboxes to `[x]` when done
- Notion: controller creates/Done `SIO-moe-f32-mixtral-forward`

- [ ] **Step 1: Plan F section** with multi-layer + prefill + loader contract + L4 numbers / blocker
- [ ] **Step 2: Mark plan checkboxes**
- [ ] **Step 3: Notion IDs for controller**
- [ ] **Step 4: Propose docs commit only**

```
docs: Plan F Mixtral multi-layer receipts

Assisted-by: Composer
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Per-layer packs | 1 |
| MoE all layers | 2 |
| ntok>1 / prefill | 2, 4 |
| Quant dispatch Q4/Q5 | 3 |
| llama_load_moe | 5 |
| Synthetic multi-layer demo | 4 |
| Mixtral demo / blocker | 5 |
| Docs + Notion | 6 |
| No GGUF / no GPU SiLU | Global |
| Dense fallback per layer | 2 |

## Execution handoff

Plan complete. Choose Subagent-Driven (recommended) or Inline Execution.

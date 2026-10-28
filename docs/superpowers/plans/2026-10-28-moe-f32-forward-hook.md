# MoE FFN forward hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire layer-0 MoE FFN into `ll_forward_impl` via existing `llama_moe_ffn_block`, with dense default and a L4 demo proving `last_moe_used`.

**Architecture:** Extend `LlamaCfg`/`Llama` with MoE fields (null/zero = dense). In `ll_forward_impl`, after post-attn norm on layer 0, if `moe_ready` and `ntok==1`, router-proj + f32 bridge + `llama_moe_ffn_block` + residual; else existing SwiGLU. Failures fall back to dense.

**Tech Stack:** Sounio `.sio`, hetero Runtime/mmid PTX, L4 via `srun`, branch `cursor/moe-f32-mmid`.

**Spec:** `docs/superpowers/specs/2026-10-28-moe-f32-forward-hook-design.md`
**Notion:** https://app.notion.com/p/4073f37f9371818d99c8d804862042a0

## Global Constraints

- No llama.cpp runtime.
- Dense default when `n_experts == 0`.
- MoE only on **layer 0**; other layers stay dense SwiGLU.
- MoE branch requires `ntok == 1` (else dense).
- On `!llama_moe_ffn_block.ok` (or not ready): dense fallback; set `last_moe_status`.
- Reuse `llama_moe_ffn_block` / `LlamaMoeFfnCfg`; do not fork MoE stack.
- Synthetic Q4_K packs in demo; no Mixtral safetensors loader.
- ASCII comments only; human-gated commits (propose message; do not commit/push/PR).
- Work only on `sounio-workspace:/workspace/sounio`; heartbeat `bin/sounio-coord heartbeat --agent cursor --lane moe-f32-mmid`.
- Notion: move forward card to In progress when Task 1 starts; Done when acceptance met.

## File map

| File | Role |
|------|------|
| `stdlib/llm/llama.sio` | Cfg/Llama fields, moe_ready helper, forward branch, last_moe_* |
| `examples/llama_moe_forward_demo.sio` | New L4/CPU demo |
| `docs/design/moe-f32-mmid.md` | Receipts |
| `examples/llama_moe_ffn_demo.sio` | Regression only (do not break) |

---

### Task 1: Data model + moe_ready

**Files:**
- Modify: `stdlib/llm/llama.sio` (`LlamaCfg`, `Llama`, `llama_cfg`, constructors)
- Test: `bin/souc check examples/llama_gpu_demo.sio` (dense path still typechecks)

**Interfaces:**
- Produces on `LlamaCfg`: `n_experts: i64`, `n_expert_used: i64`, `moe_inter: i64` (all default 0 in `llama_cfg`)
- Produces on `Llama`: `moe_router: CpuBuf`, `moe_gate_w`/`moe_up_w`/`moe_down_w: *mut u8`, byte sizes + strides, `last_moe_used: bool`, `last_moe_status: i64`
- Produces: `fn ll_moe_ready(m: &Llama, rt: &Runtime, use_gpu: bool, ntok: i64) -> bool`

- [x] **Step 1: Extend `LlamaCfg`**

Add fields after `tied` (keep `ok` last):

```sio
pub n_experts: i64,
pub n_expert_used: i64,
pub moe_inter: i64,
```

In `llama_cfg` initializer set all three to `0`.

- [x] **Step 2: Extend `Llama`**

Add MoE pack fields + telemetry. Initialize empties/nulls/`false`/`0` wherever `Llama { ... }` is constructed (including early-return unloaded shells).

```sio
pub moe_router: CpuBuf,
pub moe_gate_w: *mut u8,
pub moe_up_w: *mut u8,
pub moe_down_w: *mut u8,
pub moe_gate_bytes: i64,
pub moe_up_bytes: i64,
pub moe_down_bytes: i64,
pub moe_expert_stride_g: i64,
pub moe_row_stride_g: i64,
pub moe_expert_stride_u: i64,
pub moe_row_stride_u: i64,
pub moe_expert_stride_d: i64,
pub moe_row_stride_d: i64,
pub last_moe_used: bool,
pub last_moe_status: i64,
```

- [x] **Step 3: Add `ll_moe_ready`**

```sio
fn ll_moe_ready(m: &Llama, rt: &Runtime, use_gpu: bool, ntok: i64) -> bool {
    if !use_gpu { return false }
    if ntok != 1 { return false }
    if m.cfg.n_experts <= 0 || m.cfg.n_expert_used <= 0 { return false }
    if m.cfg.n_expert_used > 16 || m.cfg.n_expert_used > m.cfg.n_experts { return false }
    if !rt.exec.ok { return false }
    if rt.mmid_vec_func == 0 { return false }
    if m.moe_gate_w == 0 as *mut u8 || m.moe_up_w == 0 as *mut u8 || m.moe_down_w == 0 as *mut u8 { return false }
    if m.moe_gate_bytes <= 0 || m.moe_up_bytes <= 0 || m.moe_down_bytes <= 0 { return false }
    if m.moe_router.len <= 0 { return false }
    true
}
```

Adjust null checks to Sounio idioms already used in the file (pointer compare style).

- [x] **Step 4: Typecheck dense demos**

```bash
bin/souc check examples/llama_gpu_demo.sio
```

Expected: `check: OK`

- [x] **Step 5: Propose commit only**

```
llm: MoE cfg fields and ll_moe_ready for forward hook

Assisted-by: Composer
```

---

### Task 2: Hook in `ll_forward_impl`

**Files:**
- Modify: `stdlib/llm/llama.sio` (`ll_forward_impl` FFN section ~post_norm)
- Test: `bin/souc check examples/llama_gpu_demo.sio` + `examples/llama_moe_ffn_demo.sio`

**Interfaces:**
- Consumes: `ll_moe_ready`, `ll_proj`, `llama_moe_ffn_block`, f64/f32 cast helpers (add private helpers if needed)
- Produces: residual via MoE or dense; updates `m.last_moe_used` / `m.last_moe_status` each call (reset at start of forward)

- [x] **Step 1: Reset telemetry at forward entry**

At start of `ll_forward_impl`:

```sio
m.last_moe_used = false
m.last_moe_status = 0
```

(If `m` is immutable `&Llama`, change signature to `&!Llama` for forward APIs that need it — prefer mutating through existing `&!KvCache` pattern; if `m` is `&Llama`, use interior fields only if already mutable. Spec expects updates: make `llama_forward` / `llama_forward_rt` take `&!Llama` if required for `last_moe_*`.)

- [x] **Step 2: Replace layer-0 FFN with branch**

Where dense gate/up/silu/down currently runs after `h2 = ll_rmsnorm(...post_norm...)`, wrap:

```sio
var used_moe = false
if li == 0 && ll_moe_ready(m, rt, use_gpu, ntok) {
    // 1) router logits f64 via ll_proj into ck_alloc(ntok * n_experts)
    // 2) cast h2 and logits to f32 host buffers (ntok*hidden, ntok*n_experts)
    // 3) build LlamaMoeFfnCfg from m.cfg + strides/bytes
    // 4) r = llama_moe_ffn_block(...)
    // 5) if r.ok: cast dst f32 -> f64 CpuBuf dn; ll_add_into(x, dn);
    //       m.last_moe_used = true; m.last_moe_status = 0; used_moe = true
    //    else: m.last_moe_status = r.status
}
if !used_moe {
    // existing dense SwiGLU block unchanged
}
```

Helper suggestions (private in same file):

```sio
fn ll_f64_to_f32(src: &CpuBuf, n: i64) -> *mut u8  // heap_alloc 4*n, write f32 bits
fn ll_f32_to_f64(src: *mut u8, n: i64) -> CpuBuf
```

Use existing float write patterns from hetero demos / cublas_bench if available (`gb_host_set` is f64-oriented — prefer raw f32 store consistent with `moe_ffn` buffers).

`moe_inter`: if `m.cfg.moe_inter > 0` use it, else `m.cfg.inter`.

- [x] **Step 3: Typecheck**

```bash
bin/souc check examples/llama_gpu_demo.sio
bin/souc check examples/llama_moe_ffn_demo.sio
```

Expected: both `check: OK`

- [x] **Step 4: Propose commit only**

```
llm: layer-0 MoE branch in ll_forward_impl

Assisted-by: Composer
```

---

### Task 3: Demo `llama_moe_forward_demo.sio`

**Files:**
- Create: `examples/llama_moe_forward_demo.sio`
- Reuse packer from: `examples/llama_moe_ffn_demo.sio` / `hetero_moe_layer_demo.sio`
- Test: L4 `srun` + local no-GPU path

**Interfaces:**
- Consumes: `llama_forward_rt`, MoE fields on `Llama`, `rt_open`/`rt_load_mmid`
- Produces: stdout with `last_moe_used`, finite logits check, `LLAMA MOE FORWARD DONE`

- [x] **Step 1: Build minimal 1-layer Llama with MoE packs**

Compact dims: `hidden=256`, `moe_inter=512`, `experts=8`, `slots=2`, `layers=1`, tiny `vocab`/`heads` so attn is cheap. Synthetic embed/attn/FFN dense weights (non-zero small values). Fill MoE Q4_K packs + `moe_router` (transposed for `ll_proj`).

Set `cfg.n_experts=8`, `cfg.n_expert_used=2`, `cfg.moe_inter=512`.

- [x] **Step 2: Run forward**

```sio
let logits = llama_forward_rt(&!m, &!kv, &tokens, 1, &rt)
```

When GPU ready: require `m.last_moe_used` and `m.last_moe_status == 0`; scan logits for NaN/Inf; print timings if available via status; `exit(0)` after DONE (CUDA teardown hang workaround).

When GPU unavailable: print fallback path, DONE, exit 0 (dense may still run if weights exist).

- [x] **Step 3: Verify**

```bash
bin/souc check examples/llama_moe_forward_demo.sio
# L4:
# srun ... souc run examples/llama_moe_forward_demo.sio
```

Expected L4: MoE used true, finite logits, `LLAMA MOE FORWARD DONE`, RC=0.

- [x] **Step 4: Regression**

Re-run or check `examples/llama_moe_ffn_demo.sio` still OK.

- [x] **Step 5: Propose commit only**

```
examples: llama_moe_forward_demo for layer-0 MoE hook

Assisted-by: Composer
```

---

### Task 4: Docs + Notion

**Files:**
- Modify: `docs/design/moe-f32-mmid.md`
- Notion card: `4073f37f-9371-818d-99c8-d804862042a0`

- [x] **Step 1: Update design note** with forward-hook measured/demo result and that layer-0 hook landed.
- [x] **Step 2: Notion** — Status Done, Note with L4 receipt; Acceptance confirmed.
- [x] **Step 3: Propose docs commit only**

```
docs: MoE forward-hook receipts

Assisted-by: Composer
```

---

## Self-review (plan vs spec)

| Spec | Task |
|------|------|
| Cfg n_experts default 0 | Task 1 |
| Layer-0 only + ntok==1 | Task 2 |
| Dense fallback | Task 2 |
| last_moe_* | Tasks 1-2 |
| Demo synthetic + L4 | Task 3 |
| Docs + Notion | Task 4 |
| No llama.cpp / no Mixtral loader | Global |

Placeholder scan: none intentional. Commits human-gated.
